"""
Sanity-check script: verify that video frames line up correctly with
the OpenTTGames ball_markup.json annotations.

Usage:
    python verify_ball_markup.py --video test_2.mp4 --markup ball_markup.json --out_dir checks/

This does NOT do any ball detection. It just draws the ground-truth
ball position (from the dataset annotation) onto a handful of frames,
so you can visually confirm frame indexing / coordinate system before
writing your own detector.
"""

import argparse
import json
import os
import random

import cv2


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    # keys are frame numbers as strings -> convert to int
    return {int(k): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file (e.g. test_2.mp4)")
    parser.add_argument("--markup", required=True, help="Path to ball_markup.json")
    parser.add_argument("--out_dir", default="checks", help="Where to save annotated frames")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of frames to check")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    markup = load_markup(args.markup)

    # only keep frames where the ball is actually visible (not -1,-1)
    visible_frames = [f for f, coord in markup.items() if coord["x"] != -1 and coord["y"] != -1]

    if not visible_frames:
        print("No frames with a visible ball found in markup file.")
        return

    sample_frames = sorted(random.sample(visible_frames, min(args.n_samples, len(visible_frames))))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames total.")

    for frame_idx in sample_frames:
        if frame_idx >= total_frames:
            print(f"Frame {frame_idx} is out of range for this video, skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Failed to read frame {frame_idx}, skipping.")
            continue

        x, y = markup[frame_idx]["x"], markup[frame_idx]["y"]
        cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"frame {frame_idx} ball=({x},{y})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        out_path = os.path.join(args.out_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

    cap.release()
    print("Done. Open the saved images and check the red circle actually sits on the ball.")


if __name__ == "__main__":
    main()
