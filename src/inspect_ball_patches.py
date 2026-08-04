"""
Saves magnified crops around the ground-truth ball position, so you can
visually check whether the color calibration patches actually show the
ball (vs. background bleeding in due to motion blur or patch size).

Usage:
    python inspect_ball_patches.py --video test_2.mp4 --markup ball_markup.json --n_samples 12
"""

import argparse
import json
import os
import random

import cv2


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--markup", required=True)
    parser.add_argument("--n_samples", type=int, default=12)
    parser.add_argument("--patch_radius", type=int, default=12,
                         help="Half-size of the raw crop before magnifying (bigger than the color-sampling radius, for context)")
    parser.add_argument("--scale", type=int, default=10, help="Magnification factor")
    parser.add_argument("--out_dir", default="ball_patches")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    markup = load_markup(args.markup)
    visible_frames = [f for f, c in markup.items() if c["x"] != -1 and c["y"] != -1]
    sample_frames = sorted(random.sample(visible_frames, min(args.n_samples, len(visible_frames))))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        x, y = int(markup[frame_idx]["x"]), int(markup[frame_idx]["y"])
        r = args.patch_radius
        h, w = frame.shape[:2]
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        patch = frame[y0:y1, x0:x1]

        if patch.size == 0:
            continue

        big = cv2.resize(patch, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)

        # mark the exact ground-truth center with a 1px crosshair for reference
        center_x = (x - x0) * args.scale
        center_y = (y - y0) * args.scale
        cv2.drawMarker(big, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)

        out_path = os.path.join(args.out_dir, f"patch_{frame_idx}.jpg")
        cv2.imwrite(out_path, big)
        print(f"Saved {out_path}")

    cap.release()
    print(f"\nOpen the images in {args.out_dir}/ -- the red crosshair marks the exact ground-truth center.")


if __name__ == "__main__":
    main()
