"""
Calibrate what color the ball actually is in this footage, by sampling a
small patch of pixels around the ground-truth ball position on several
frames. This gives us real numbers to set a color filter with, instead
of guessing "the ball is probably white".

Usage:
    python calibrate_ball_color.py --video test_2.mp4 --markup ball_markup.json --n_samples 15
"""

import argparse
import json
import random

import cv2
import numpy as np


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--markup", required=True)
    parser.add_argument("--n_samples", type=int, default=15)
    parser.add_argument("--patch_radius", type=int, default=4,
                         help="Half-size of the pixel patch to sample around the ball center")
    args = parser.parse_args()

    markup = load_markup(args.markup)
    visible_frames = [f for f, c in markup.items() if c["x"] != -1 and c["y"] != -1]
    sample_frames = random.sample(visible_frames, min(args.n_samples, len(visible_frames)))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    hsv_samples = []
    bgr_samples = []

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        x, y = int(markup[frame_idx]["x"]), int(markup[frame_idx]["y"])
        r = args.patch_radius
        h, w = frame.shape[:2]

        # clip patch to stay inside the frame
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        patch_bgr = frame[y0:y1, x0:x1]

        if patch_bgr.size == 0:
            continue

        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

        bgr_samples.append(patch_bgr.reshape(-1, 3))
        hsv_samples.append(patch_hsv.reshape(-1, 3))

    if not bgr_samples:
        print("No valid samples collected.")
        return

    all_bgr = np.concatenate(bgr_samples, axis=0)
    all_hsv = np.concatenate(hsv_samples, axis=0)

    print(f"Sampled {len(all_bgr)} pixels from {len(sample_frames)} frames at ground-truth ball positions.\n")

    print("BGR  mean:", all_bgr.mean(axis=0).round(1), " std:", all_bgr.std(axis=0).round(1))
    print("HSV  mean:", all_hsv.mean(axis=0).round(1), " std:", all_hsv.std(axis=0).round(1))

    # suggested HSV range: mean +/- 2.5 std, clipped to valid ranges
    h_mean, s_mean, v_mean = all_hsv.mean(axis=0)
    h_std, s_std, v_std = all_hsv.std(axis=0)

    lower = np.array([
        max(0, h_mean - 2.5 * h_std),
        max(0, s_mean - 2.5 * s_std),
        max(0, v_mean - 2.5 * v_std),
    ]).astype(int)
    upper = np.array([
        min(179, h_mean + 2.5 * h_std),
        min(255, s_mean + 2.5 * s_std),
        min(255, v_mean + 2.5 * v_std),
    ]).astype(int)

    print(f"\nSuggested HSV range for a color filter:")
    print(f"  lower = {tuple(lower)}")
    print(f"  upper = {tuple(upper)}")
    print("\n(H is 0-179 in OpenCV, not 0-360 -- keep that in mind when you use this range later.)")


if __name__ == "__main__":
    main()
