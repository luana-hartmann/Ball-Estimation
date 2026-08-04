"""
Calibrates RELATIVE color features: for the ball's true motion contour,
measure how its brightness/saturation compares to the ring of pixels
immediately surrounding it (not the whole frame, not a fixed number).

Why local, not global-frame: lighting can vary across a single frame too
(shadows, spotlights), so comparing a candidate to its own immediate
neighborhood is a more robust invariant than comparing to a whole-frame
average, and it doesn't require knowing anything about overall exposure.

Usage:
    python calibrate_relative_color.py --video test_2.mp4 --markup ball_markup_test2.json --n_samples 300
"""

import argparse
import json
import math
import random

import cv2
import numpy as np

from detector import three_frame_diff


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def relative_stats_for_nearest_contour(gray_prev, gray_curr, gray_next, curr_color,
                                        gt_x, gt_y, near_thresh, ring_margin=10):
    motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_hsv = cv2.cvtColor(curr_color, cv2.COLOR_BGR2HSV)
    h, w = motion_mask.shape

    best_cnt, best_dist = None, None
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist = math.hypot(cx - gt_x, cy - gt_y)
        if best_dist is None or dist < best_dist:
            best_dist, best_cnt = dist, cnt

    if best_dist is None or best_dist > near_thresh:
        return None

    # candidate mask
    cand_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cand_mask, [best_cnt], -1, 255, -1)

    # local region = bounding box of the candidate, expanded by ring_margin
    x, y, bw, bh = cv2.boundingRect(best_cnt)
    x0, y0 = max(0, x - ring_margin), max(0, y - ring_margin)
    x1, y1 = min(w, x + bw + ring_margin), min(h, y + bh + ring_margin)

    local_region_mask = np.zeros((h, w), dtype=np.uint8)
    local_region_mask[y0:y1, x0:x1] = 255

    # ring = local region minus the candidate itself
    surrounding_mask = cv2.bitwise_and(local_region_mask, cv2.bitwise_not(cand_mask))
    if cv2.countNonZero(surrounding_mask) == 0:
        return None

    cand_hsv = cv2.mean(frame_hsv, mask=cand_mask)
    surr_hsv = cv2.mean(frame_hsv, mask=surrounding_mask)

    rel_v = cand_hsv[2] - surr_hsv[2]   # how much BRIGHTER than surroundings (expect positive)
    rel_s = cand_hsv[1] - surr_hsv[1]   # how much MORE/LESS saturated (expect negative -- paler)

    return {"rel_v": rel_v, "rel_s": rel_s, "dist": best_dist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--markup", required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--near_thresh", type=float, default=25)
    parser.add_argument("--ring_margin", type=int, default=10)
    args = parser.parse_args()

    markup = load_markup(args.markup)
    visible_frames = [f for f, c in markup.items() if c["x"] != -1 and c["y"] != -1]
    sample_frames = sorted(random.sample(visible_frames, min(args.n_samples, len(visible_frames))))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    rel_v_values, rel_s_values = [], []
    skipped = 0

    for frame_idx in sample_frames:
        gt_x, gt_y = markup[frame_idx]["x"], markup[frame_idx]["y"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ok1, prev_c = cap.read()
        ok2, curr_c = cap.read()
        ok3, next_c = cap.read()
        if not (ok1 and ok2 and ok3):
            skipped += 1
            continue

        gray_prev = cv2.cvtColor(prev_c, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_c, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(next_c, cv2.COLOR_BGR2GRAY)

        stats = relative_stats_for_nearest_contour(
            gray_prev, gray_curr, gray_next, curr_c, gt_x, gt_y,
            args.near_thresh, args.ring_margin,
        )
        if stats is None:
            skipped += 1
            continue

        rel_v_values.append(stats["rel_v"])
        rel_s_values.append(stats["rel_s"])

    cap.release()

    n = len(rel_v_values)
    print(f"Collected {n} samples ({skipped} skipped).\n")
    if n == 0:
        return

    rv, rs = np.array(rel_v_values), np.array(rel_s_values)

    def report(name, arr):
        p = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name}: min={arr.min():.1f} p1={p[0]:.1f} p5={p[1]:.1f} p25={p[2]:.1f} "
              f"median={p[3]:.1f} p75={p[4]:.1f} p95={p[5]:.1f} p99={p[6]:.1f} max={arr.max():.1f}")

    report("rel_v (candidate V - surrounding V)", rv)
    report("rel_s (candidate S - surrounding S)", rs)

    suggested_min_rel_v = np.percentile(rv, 10)
    suggested_max_rel_s = np.percentile(rs, 90)

    print(f"\nSuggested thresholds (~90% coverage):")
    print(f"  --min_rel_v {suggested_min_rel_v:.1f}")
    print(f"  --max_rel_s {suggested_max_rel_s:.1f}")


if __name__ == "__main__":
    main()
