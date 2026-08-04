"""
Proper color-filter calibration: for a large sample of ground-truth
frames, find the ball's actual motion contour (nearest to GT, within
near_thresh pixels) and record its mean saturation and value. This gives
us the REAL distribution of these values across the video, so we can set
thresholds from percentiles instead of guessing.

This reuses the exact same contour-finding logic as detect_ball_classical_v2
and diagnose_fn, so the calibration matches what the detector actually sees.

Usage:
    python calibrate_color_filter.py --video test_2.mp4 --markup ball_markup.json --n_samples 300
"""

import argparse
import json
import math
import random

import cv2
import numpy as np

from detect_ball_classical_v2 import three_frame_diff


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def nearest_contour_stats(gray_prev, gray_curr, gray_next, curr_color, gt_x, gt_y, near_thresh):
    motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_hsv = cv2.cvtColor(curr_color, cv2.COLOR_BGR2HSV)

    best = None
    best_dist = None
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist = math.hypot(cx - gt_x, cy - gt_y)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_cnt = cnt

    if best_dist is None or best_dist > near_thresh:
        return None

    mask = np.zeros(motion_mask.shape, dtype=np.uint8)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    mean_hsv = cv2.mean(frame_hsv, mask=mask)
    area = cv2.contourArea(best_cnt)
    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(best_cnt)
    aspect = max(rect_w, rect_h) / min(rect_w, rect_h) if rect_w and rect_h else None

    return {"mean_s": mean_hsv[1], "mean_v": mean_hsv[2], "area": area, "aspect": aspect, "dist": best_dist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--markup", required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--near_thresh", type=float, default=25)
    args = parser.parse_args()

    markup = load_markup(args.markup)
    visible_frames = [f for f, c in markup.items() if c["x"] != -1 and c["y"] != -1]
    sample_frames = sorted(random.sample(visible_frames, min(args.n_samples, len(visible_frames))))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    s_values, v_values, area_values, aspect_values = [], [], [], []
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

        stats = nearest_contour_stats(gray_prev, gray_curr, gray_next, curr_c, gt_x, gt_y, args.near_thresh)
        if stats is None:
            skipped += 1
            continue

        s_values.append(stats["mean_s"])
        v_values.append(stats["mean_v"])
        area_values.append(stats["area"])
        if stats["aspect"] is not None:
            aspect_values.append(stats["aspect"])

    cap.release()

    n = len(s_values)
    print(f"Collected {n} real ball-contour samples ({skipped} frames skipped -- no contour found near GT).\n")

    if n == 0:
        print("No samples collected, cannot calibrate.")
        return

    s_arr, v_arr = np.array(s_values), np.array(v_values)
    area_arr, aspect_arr = np.array(area_values), np.array(aspect_values)

    def report(name, arr):
        p = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name}: min={arr.min():.0f} p1={p[0]:.0f} p5={p[1]:.0f} p25={p[2]:.0f} "
              f"median={p[3]:.0f} p75={p[4]:.0f} p95={p[5]:.0f} p99={p[6]:.0f} max={arr.max():.0f}")

    report("saturation", s_arr)
    report("value", v_arr)
    report("area", area_arr)
    report("aspect_ratio", aspect_arr)

    # suggest thresholds at the 1st/99th percentile with a small safety margin,
    # so we keep ~98% of real ball detections while still filtering distractors
    suggested_max_sat = np.percentile(s_arr, 99) + 5
    suggested_min_val = np.percentile(v_arr, 1) - 5
    suggested_max_area = np.percentile(area_arr, 99) * 1.2
    suggested_max_aspect = np.percentile(aspect_arr, 99) + 0.5

    print(f"\nSuggested thresholds (covers ~98% of real ball detections + margin):")
    print(f"  --max_saturation {suggested_max_sat:.0f}")
    print(f"  --min_value {suggested_min_val:.0f}")
    print(f"  --max_area {suggested_max_area:.0f}")
    print(f"  --max_aspect_ratio {suggested_max_aspect:.1f}")


if __name__ == "__main__":
    main()
