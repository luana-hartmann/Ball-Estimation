"""
Classical CV ball detector, v3.

Changes from v2:
  - Thresholds moved from extreme (1st/99th percentile) to moderate
    (~85-90th percentile), based on calibrate_color_filter.py output.
    v2's extreme thresholds technically covered 99% of true ball
    appearances, but that acceptance window overlapped heavily with
    non-ball objects (paddle edges, skin), causing far more WRONG_POS
    matches. Moderate thresholds trade a little recall ceiling for a
    much more selective candidate pool.

  - pick_best_candidate() now scores candidates using BOTH spatial
    distance to the predicted position AND "typicality" (how close the
    candidate's color/size is to the median values measured during
    calibration). Previously we picked purely by spatial nearest-
    neighbor, which breaks down when multiple candidates (ball +
    paddle edge, say) are both near the predicted position.

Reference stats below come directly from calibrate_color_filter.py's
output on test_2.mp4 -- median values, and a robust "std" estimated from
the interquartile range (IQR / 1.349, the standard robust-std formula).
If you calibrate on a different video, recompute these.
"""

import argparse
import csv
import math

import cv2
import numpy as np


# ---- reference stats from calibration (test_2.mp4) ----
REF_MEDIAN_S = 118.0
REF_STD_S = (142 - 77) / 1.349     # from p25/p75
REF_MEDIAN_V = 153.0
REF_STD_V = (170 - 131) / 1.349
REF_MEDIAN_AREA = 380.0
REF_STD_AREA = (496 - 183) / 1.349


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


def find_candidates(motion_mask, curr_frame_bgr, min_area=4, max_area=650,
                     max_aspect_ratio=3.2, max_saturation=155, min_value=112):
    """Hard filter -- same idea as v2, thresholds now moderate (~85-90th pctile)."""
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_hsv = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2HSV)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(cnt)
        if rect_w == 0 or rect_h == 0:
            continue
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect_ratio > max_aspect_ratio:
            continue

        mask = np.zeros(motion_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_hsv = cv2.mean(frame_hsv, mask=mask)
        mean_s, mean_v = mean_hsv[1], mean_hsv[2]
        if mean_s > max_saturation or mean_v < min_value:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        candidates.append({
            "x": cx, "y": cy, "area": area, "aspect_ratio": aspect_ratio,
            "mean_s": mean_s, "mean_v": mean_v,
        })

    return candidates


def typicality_score(c):
    """
    Lower = more 'ball-like'. Measures how many (robust) standard
    deviations this candidate's color/size is away from the calibrated
    median ball. This is what lets us break ties between multiple
    candidates that are all spatially close to the prediction.
    """
    z_s = abs(c["mean_s"] - REF_MEDIAN_S) / REF_STD_S
    z_v = abs(c["mean_v"] - REF_MEDIAN_V) / REF_STD_V
    z_area = abs(c["area"] - REF_MEDIAN_AREA) / REF_STD_AREA
    return z_s + z_v + z_area


def predict_next_position(history):
    known = [h for h in history if h is not None]
    if len(known) < 2:
        return None
    (f1, x1, y1), (f2, x2, y2) = known[-2], known[-1]
    df = f2 - f1
    if df == 0:
        return (x2, y2)
    vx, vy = (x2 - x1) / df, (y2 - y1) / df
    return (x2 + vx, y2 + vy)


def pick_best_candidate(candidates, predicted_pos, max_pred_dist=60, dist_weight=0.05):
    """
    Among candidates within max_pred_dist of the prediction (hard gate,
    same safety principle as before: don't guess wildly far from where
    physics says the ball should be), pick the one with the best combined
    score of spatial distance (scaled down by dist_weight, since pixel
    distances and typicality z-scores are on different scales) and color
    typicality -- NOT just the spatially nearest one.
    """
    if not candidates:
        return None

    if predicted_pos is not None:
        px, py = predicted_pos
        scored = []
        for c in candidates:
            dist = math.hypot(c["x"] - px, c["y"] - py)
            if dist > max_pred_dist:
                continue
            score = dist * dist_weight + typicality_score(c)
            scored.append((score, dist, c))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0])
        _, best_dist, best = scored[0]
        confidence = max(0.3, 1.0 - best_dist / max_pred_dist)
        return (best["x"], best["y"], confidence)

    # no prediction yet: pick the most typical candidate outright
    best = min(candidates, key=typicality_score)
    return (best["x"], best["y"], 0.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="trajectory_v3.csv")
    parser.add_argument("--min_area", type=float, default=4)
    parser.add_argument("--max_area", type=float, default=650)
    parser.add_argument("--max_aspect_ratio", type=float, default=3.2)
    parser.add_argument("--max_saturation", type=float, default=155)
    parser.add_argument("--min_value", type=float, default=112)
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
        writer.writerows(results)

    n_found = sum(1 for r in results if r[3] != -1)
    print(f"Processed {len(results)} frames, ball found in {n_found} ({100*n_found/len(results):.1f}%).")
    print(f"Saved trajectory to {args.out}")


if __name__ == "__main__":
    main()
