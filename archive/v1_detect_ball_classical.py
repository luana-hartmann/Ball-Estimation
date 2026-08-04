"""
Classical computer vision ball detector for table tennis footage.

Pipeline (3 stages):
    1. Motion candidates  -> three_frame_diff()
    2. Shape filtering     -> find_candidates()
    3. Motion continuity    -> pick_best_candidate()

Output: a CSV with one row per frame:
    frame_idx, timestamp_s, x, y, confidence

confidence is in [0, 1]. A missing detection is written as x=-1, y=-1, confidence=0
(same convention as the OpenTTGames dataset, so it's easy to compare against
ground truth and easy for downstream code to check "is this frame usable").

Usage:
    python detect_ball_classical.py --video test_2.mp4 --out trajectory.csv
"""

import argparse
import csv
import math

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Stage 1: motion candidates via 3-frame differencing
# ----------------------------------------------------------------------
def three_frame_diff(prev_gray, curr_gray, next_gray, diff_thresh=15):
    """
    Returns a binary mask where a pixel is 255 if it changed significantly
    in BOTH the (prev -> curr) and (curr -> next) transitions.

    Why the AND of two diffs instead of one: something that changed only
    once (e.g. a lighting flicker, or the tail end of a player's motion
    settling) tends to disappear in one of the two diffs. A fast-moving
    small object like the ball keeps showing up as "changed" across both
    transitions, so intersecting the two masks suppresses a lot of noise
    before we even look at shape.
    """
    diff1 = cv2.absdiff(prev_gray, curr_gray)
    diff2 = cv2.absdiff(curr_gray, next_gray)

    _, mask1 = cv2.threshold(diff1, diff_thresh, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(diff2, diff_thresh, 255, cv2.THRESH_BINARY)

    motion_mask = cv2.bitwise_and(mask1, mask2)

    # Morphological cleanup: remove single-pixel noise (erode) then
    # restore/merge nearby fragments of the same blob (dilate).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

    return motion_mask


# ----------------------------------------------------------------------
# Stage 2: shape filtering on the motion mask
# ----------------------------------------------------------------------
def find_candidates(motion_mask, min_area=4, max_area=250, min_circularity=0.5):
    """
    Finds connected components in the motion mask and keeps only the ones
    that plausibly look like a ball: small, and close to circular.

    circularity = 4*pi*area / perimeter^2
    A perfect circle scores 1.0. Elongated blobs (motion blur streaks,
    edges of limbs) score much lower.

    Returns a list of dicts: {"x", "y", "area", "circularity"}
    (x, y) is the centroid of the contour, i.e. the candidate ball center.
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        candidates.append({"x": cx, "y": cy, "area": area, "circularity": circularity})

    return candidates


# ----------------------------------------------------------------------
# Stage 3: motion-continuity tracking across frames
# ----------------------------------------------------------------------
def predict_next_position(history):
    """
    Linear extrapolation from the last two known ball positions.
    history is a list of (frame_idx, x, y) for frames where the ball
    WAS found, most recent last. Returns a predicted (x, y) or None if
    we don't have enough history yet.
    """
    known = [h for h in history if h is not None]
    if len(known) < 2:
        return None

    (f1, x1, y1), (f2, x2, y2) = known[-2], known[-1]
    df = f2 - f1
    if df == 0:
        return (x2, y2)

    vx = (x2 - x1) / df
    vy = (y2 - y1) / df

    # predict one frame ahead of the last known position
    pred_x = x2 + vx
    pred_y = y2 + vy
    return (pred_x, pred_y)


def pick_best_candidate(candidates, predicted_pos, max_pred_dist=60):
    """
    Chooses which candidate is the ball.

    - If we have a motion prediction: pick the candidate closest to it,
      but only if it's within max_pred_dist pixels. Otherwise we'd rather
      report "not found" than confidently pick the wrong blob.
    - If we have no prediction yet (start of track / no track): fall back
      to the most circular candidate, as a reasonable first guess.

    Returns (x, y, confidence) or None.
    """
    if not candidates:
        return None

    if predicted_pos is not None:
        px, py = predicted_pos
        best = None
        best_dist = None
        for c in candidates:
            dist = math.hypot(c["x"] - px, c["y"] - py)
            if best_dist is None or dist < best_dist:
                best, best_dist = c, dist

        if best_dist is not None and best_dist <= max_pred_dist:
            # confidence decays with distance from prediction: exact match -> 1.0,
            # at the max allowed distance -> ~0.3
            confidence = max(0.3, 1.0 - best_dist / max_pred_dist)
            return (best["x"], best["y"], confidence)
        else:
            return None  # nothing close enough to the prediction -> not found

    # no prediction available yet: fall back to most circular candidate
    best = max(candidates, key=lambda c: c["circularity"])
    return (best["x"], best["y"], 0.4)  # lower confidence, this is a weak fallback


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="trajectory.csv")
    parser.add_argument("--min_area", type=float, default=4)
    parser.add_argument("--max_area", type=float, default=250)
    parser.add_argument("--min_circularity", type=float, default=0.5)
    parser.add_argument("--max_pred_dist", type=float, default=60)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0  # OpenTTGames videos are 120fps

    # Read the first two frames to prime the 3-frame diff.
    ok1, frame_prev = cap.read()
    ok2, frame_curr = cap.read()
    if not (ok1 and ok2):
        print("Video too short to process.")
        return

    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

    history = []  # list of (frame_idx, x, y) for frames where ball was found
    results = []  # list of (frame_idx, timestamp_s, x, y, confidence) for ALL frames

    frame_idx = 1  # frame_curr is frame index 1 (0-indexed), we need frame_idx+1 = "next"

    while True:
        ok_next, frame_next = cap.read()
        if not ok_next:
            break
        gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)

        # Stage 1 + 2
        motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
        candidates = find_candidates(
            motion_mask,
            min_area=args.min_area,
            max_area=args.max_area,
            min_circularity=args.min_circularity,
        )

        # Stage 3
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

        # keep history from growing forever; we only ever need the last 2 known points
        history = history[-5:]

        # slide the window forward
        gray_prev, gray_curr = gray_curr, gray_next
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
