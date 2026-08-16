"""
For frames where the ML pipeline MISSED the ball (FN), finds the raw
motion contour closest to ground truth and reports the CNN's predicted
probability for it. Distinguishes two very different failure modes:

  - "no contour found near GT" -> stage 1 (motion detection) didn't even
    segment the ball as something that moved. Not a classifier problem.
  - "found, but classifier gave it probability X" -> stage 1 worked,
    the CNN saw the real ball's patch and confidently said "not ball".
    This IS a classifier problem -- likely appearance domain shift
    (different lighting/ball/background than training videos).

Usage:
    python src/ml/diagnose_fn_ml.py --video data/test7/test7.mp4 --model outputs/ml_dataset/ball_classifier.pt --eval_csv outputs/eval/eval_per_frame_test7_ml.csv --n 15
"""

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detector import three_frame_diff, find_candidates  # noqa: E402
from model import BallPatchCNN  # noqa: E402
from detect_ball_ml import crop_patch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--near_thresh", type=float, default=25)
    parser.add_argument("--out_dir", default="outputs/diagnosis/fn_diagnosis_ml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallPatchCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    fn_rows = []
    with open(args.eval_csv, "r") as f:
        for row in csv.DictReader(f):
            if row["category"] == "FN":
                fn_rows.append(row)

    if not fn_rows:
        print("No FN frames found.")
        return

    sample = fn_rows[:: max(1, len(fn_rows) // args.n)][: args.n]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    not_found_at_all = 0
    rejected_by_classifier = 0
    probs_when_found = []

    for row in sample:
        frame_idx = int(row["frame_idx"])
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ok1, prev_c = cap.read()
        ok2, curr_c = cap.read()
        ok3, next_c = cap.read()
        if not (ok1 and ok2 and ok3):
            continue

        gray_prev = cv2.cvtColor(prev_c, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_c, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(next_c, cv2.COLOR_BGR2GRAY)

        motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
        candidates = find_candidates(
            motion_mask, curr_c,
            min_area=0, max_area=float("inf"), max_aspect_ratio=float("inf"),
            min_rel_v=-float("inf"), max_rel_s=float("inf"),
        )

        print(f"\nframe {frame_idx}: {len(candidates)} raw motion contours total")

        if not candidates:
            not_found_at_all += 1
            print("  -> NOT FOUND AT ALL (no motion contours this frame)")
            continue

        dists = [math.hypot(c["x"] - gt_x, c["y"] - gt_y) for c in candidates]
        best_i = int(np.argmin(dists))
        best_dist = dists[best_i]

        if best_dist > args.near_thresh:
            not_found_at_all += 1
            print(f"  -> NOT FOUND AT ALL near GT (nearest contour is {best_dist:.0f}px away). "
                  f"Stage 1 (motion detection) missed it.")
            continue

        # the nearest-to-GT candidate DOES exist -- check what the CNN thinks of it
        best_c = candidates[best_i]
        patch = crop_patch(curr_c, best_c["x"], best_c["y"])
        patch_t = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = F.softmax(model(patch_t), dim=1)[0, 1].item()

        probs_when_found.append(prob)
        print(f"  -> Found at {best_dist:.1f}px from GT. CNN probability of 'ball': {prob:.3f}")
        if prob < 0.5:
            rejected_by_classifier += 1

        cv2.imwrite(os.path.join(args.out_dir, f"fn_{frame_idx}_prob{prob:.2f}.jpg"),
                    cv2.resize(patch, (128, 128), interpolation=cv2.INTER_NEAREST))

    print(f"\n--- Summary over {len(sample)} sampled FN frames ---")
    print(f"Not found at all (motion stage missed it): {not_found_at_all}")
    print(f"Found but classifier rejected it (prob < 0.5): {rejected_by_classifier}")
    if probs_when_found:
        print(f"Mean CNN probability on the true ball's patch, when found: "
              f"{sum(probs_when_found)/len(probs_when_found):.3f}")
    print(f"Saved patch crops to {args.out_dir}/")


if __name__ == "__main__":
    main()
