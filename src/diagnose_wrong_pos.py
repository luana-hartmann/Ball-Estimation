"""
Diagnostic tool: for the worst WRONG_POS frames (biggest pixel error),
draw BOTH the ground-truth ball position (red) and our detection (blue)
on the frame, so you can visually see what we're actually locking onto
instead of the ball.

Usage:
    python diagnose_wrong_pos.py --video test_2.mp4 --eval_csv eval_per_frame.csv --out_dir diagnosis/ --n 10
"""

import argparse
import csv
import os

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--eval_csv", required=True, help="eval_per_frame.csv from eval_trajectory.py")
    parser.add_argument("--out_dir", default="diagnosis")
    parser.add_argument("--n", type=int, default=10, help="How many worst frames to inspect")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all WRONG_POS rows, since those are the interesting failure mode
    wrong_rows = []
    with open(args.eval_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == "WRONG_POS":
                # pixel_error wasn't saved for WRONG_POS in the eval script's
                # CSV write (only TP got it) -- compute it here instead.
                gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
                pred_x, pred_y = float(row["pred_x"]), float(row["pred_y"])
                dist = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
                wrong_rows.append({**row, "dist": dist})

    if not wrong_rows:
        print("No WRONG_POS frames found -- nothing to diagnose.")
        return

    # sort by distance descending: worst mismatches first
    wrong_rows.sort(key=lambda r: r["dist"], reverse=True)
    worst = wrong_rows[: args.n]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    for row in worst:
        frame_idx = int(row["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        gt_x, gt_y = int(float(row["gt_x"])), int(float(row["gt_y"]))
        pred_x, pred_y = int(float(row["pred_x"])), int(float(row["pred_y"]))

        # ground truth: red
        cv2.circle(frame, (gt_x, gt_y), 10, (0, 0, 255), 2)
        cv2.putText(frame, "GT", (gt_x + 12, gt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # our prediction: blue
        cv2.circle(frame, (pred_x, pred_y), 10, (255, 0, 0), 2)
        cv2.putText(frame, "PRED", (pred_x + 12, pred_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"frame {frame_idx}  dist={row['dist']:.0f}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_path = os.path.join(args.out_dir, f"wrong_{frame_idx}_dist{int(row['dist'])}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

    cap.release()
    print(f"\nOpen the images in {args.out_dir}/ and look at what the blue circle landed on.")


if __name__ == "__main__":
    main()
