"""
Classical CV ball detector, v2.

What changed from v1, and why (see diagnosis in project notes):
  - v1 filtered candidates by circularity, assuming the ball looks round.
    Visual inspection showed the real ball is usually motion-blurred into
    an elongated streak, which FAILS a circularity test. This let the
    filter reject the real ball on most frames, leaving the tracker to
    lock onto round-but-wrong objects (a paddle blade, in our case).
  - v2 relaxes the shape filter (tolerates elongated blobs) and adds a
    brightness/saturation filter instead: the ball is bright and pale
    (low color saturation) even when blurred, which distinguishes it
    from a red/black paddle or skin tones. This survives motion blur
    much better than a "is it a circle" test does.

Pipeline stages are unchanged in structure:
    1. Motion candidates  -> three_frame_diff()
    2. Shape + color filtering -> find_candidates()
    3. Motion continuity   -> pick_best_candidate()  (unchanged from v1)

Usage:
    python detect_ball_classical_v2.py --video test_2.mp4 --out trajectory_v2.csv
"""

import argparse
import csv
import math

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Stage 1: motion candidates via 3-frame differencing (unchanged from v1)
# ----------------------------------------------------------------------
def three_frame_diff(prev_gray, curr_gray, next_gray, diff_thresh=15):
    diff1 = cv2.absdiff(prev_gray, curr_gray)
    diff2 = cv2.absdiff(curr_gray, next_gray)
    _, mask1 = cv2.threshold(diff1, diff_thresh, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(diff2, diff_thresh, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.bitwise_and(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    return motion_mask


# ----------------------------------------------------------------------
# Stage 2: shape + color filtering
# ----------------------------------------------------------------------
def find_candidates(motion_mask, curr_frame_bgr, min_area=4, max_area=600,
                     max_aspect_ratio=6.0, max_saturation=140, min_value=130):
    """
    curr_frame_bgr: the ORIGINAL color frame (not the diff mask), used to
    read the true color under each candidate blob.

    Shape filter: replaced circularity with a much looser aspect-ratio
    check via the minimum-area bounding rectangle. This accepts both
    round blobs (aspect ratio ~1) and elongated motion-blur streaks
    (aspect ratio up to max_aspect_ratio), rejecting only extremely
    thin/long shapes that are almost certainly not the ball (e.g. a
    thin edge or line artifact).

    Color filter: the ball is bright and low-saturation (whitish/pale)
    even when blurred. We reject anything too saturated (a red paddle,
    skin tone) or too dark (shadows, dark clothing).
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_hsv = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2HSV)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # aspect ratio via minimum-area rotated rectangle: robust to
        # elongated streaks in any orientation (not just axis-aligned)
        (rect_cx, rect_cy), (rect_w, rect_h), _ = cv2.minAreaRect(cnt)
        if rect_w == 0 or rect_h == 0:
            continue
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect_ratio > max_aspect_ratio:
            continue

        # sample the true color under this blob from the original frame
        mask = np.zeros(motion_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_hsv = cv2.mean(frame_hsv, mask=mask)  # (H, S, V, _)
        mean_s, mean_v = mean_hsv[1], mean_hsv[2]

        if mean_s > max_saturation or mean_v < min_value:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        candidates.append({
            "x": cx, "y": cy, "area": area,
            "aspect_ratio": aspect_ratio, "mean_s": mean_s, "mean_v": mean_v,
        })

    return candidates


# ----------------------------------------------------------------------
# Stage 3: motion-continuity tracking (unchanged from v1)
# ----------------------------------------------------------------------
def predict_next_position(history):
    known = [h for h in history if h is not None]
    if len(known) < 2:
        return None
    (f1, x1, y1), (f2, x2, y2) = known[-2], known[-1]
    df = f2 - f1
    if df == 0:
        return (x2, y2)
    vx = (x2 - x1) / df
    vy = (y2 - y1) / df
    return (x2 + vx, y2 + vy)


def pick_best_candidate(candidates, predicted_pos, max_pred_dist=60):
    if not candidates:
        return None

    if predicted_pos is not None:
        px, py = predicted_pos
        best, best_dist = None, None
        for c in candidates:
            dist = math.hypot(c["x"] - px, c["y"] - py)
            if best_dist is None or dist < best_dist:
                best, best_dist = c, dist

        if best_dist is not None and best_dist <= max_pred_dist:
            confidence = max(0.3, 1.0 - best_dist / max_pred_dist)
            return (best["x"], best["y"], confidence)
        else:
            return None

    # no prediction yet: fall back to the brightest, most saturation-pure
    # candidate (lowest saturation = closest to "white") rather than most
    # circular, since v2 no longer trusts circularity as a strong signal
    best = min(candidates, key=lambda c: c["mean_s"])
    return (best["x"], best["y"], 0.4)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="trajectory_v2.csv")
    parser.add_argument("--min_area", type=float, default=4)
    parser.add_argument("--max_area", type=float, default=600)
    parser.add_argument("--max_aspect_ratio", type=float, default=6.0)
    parser.add_argument("--max_saturation", type=float, default=140)
    parser.add_argument("--min_value", type=float, default=130)
    parser.add_argument("--max_pred_dist", type=float, default=60)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0

    ok1, frame_prev_color = cap.read()
    ok2, frame_curr_color = cap.read()
    if not (ok1 and ok2):
        print("Video too short to process.")
        return

    gray_prev = cv2.cvtColor(frame_prev_color, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr_color, cv2.COLOR_BGR2GRAY)

    history = []
    results = []
    frame_idx = 1

    while True:
        ok_next, frame_next_color = cap.read()
        if not ok_next:
            break
        gray_next = cv2.cvtColor(frame_next_color, cv2.COLOR_BGR2GRAY)

        motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
        # NOTE: candidates are sampled from frame_curr_color, matching
        # gray_curr (the frame the motion mask is centered on)
        candidates = find_candidates(
            motion_mask, frame_curr_color,
            min_area=args.min_area, max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
            max_saturation=args.max_saturation, min_value=args.min_value,
        )

        predicted_pos = predict_next_position(history)
        result = pick_best_candidate(candidates, predicted_pos, max_pred_dist=args.max_pred_dist)

        timestamp_s = frame_idx / fps

        if result is not None:
            x, y, confidence = result
            results.append((frame_idx, timestamp_s, x, y, confidence))
            history.append((frame_idx, x, y))
        else:
            results.append((frame_idx, timestamp_s, -1, -1, 0.0))
            history.append(None)

        history = history[-5:]

        gray_prev, gray_curr = gray_curr, gray_next
        frame_curr_color = frame_next_color
        frame_idx += 1

    cap.release()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_s", "x", "y", "confidence"])
        for row in results:
            writer.writerow(row)

    n_found = sum(1 for r in results if r[3] != -1)
    print(f"Processed {len(results)} frames, ball found in {n_found} ({100*n_found/len(results):.1f}%).")
    print(f"Saved trajectory to {args.out}")


if __name__ == "__main__":
    main()
