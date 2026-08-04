"""
Builds a labeled dataset of small image patches (ball vs not-ball) from a
video's motion-detection candidates, for training a CNN classifier to
replace the hand-tuned filtering stage of the classical detector.

Reuses detector.py's REAL three_frame_diff + find_candidates (called
permissively, with pass-through thresholds, to get every raw candidate --
same pattern as diagnose_fn.py) so the classifier trains on exactly the
kind of candidates the real pipeline produces at inference time.

For each labeled (ground-truth-visible) frame:
  - find ALL raw motion candidates
  - the one closest to ground truth, if within pos_thresh px, is POSITIVE
  - every other candidate in that frame is NEGATIVE (free hard negatives:
    paddle edges, hands, shadows -- exactly what a hand-tuned threshold
    struggles with)
  - crop a fixed-size patch around each candidate's center, save it

Usage:
    python src/ml/generate_dataset.py --video data/test2/test2.mp4 --markup data/test2/ball_markup_test2.json --out outputs/ml_dataset/patches_test2.npz
"""

import argparse
import json
import math
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detector import three_frame_diff, find_candidates  # noqa: E402


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def crop_patch(frame, cx, cy, patch_size=32):
    """Crop a patch_size x patch_size patch centered at (cx, cy), padding
    with edge-replication if the crop would run off the frame border."""
    h, w = frame.shape[:2]
    half = patch_size // 2
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = x0 + patch_size, y0 + patch_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    crop = frame[y0c:y1c, x0c:x1c]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size))

    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--markup", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--pos_thresh", type=float, default=6.0,
                         help="Max px distance to ground truth to count a candidate as positive")
    args = parser.parse_args()

    markup = load_markup(args.markup)
    visible_frames = sorted(f for f, c in markup.items() if c["x"] != -1 and c["y"] != -1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return

    patches, labels, dists, frame_idxs = [], [], [], []
    n_pos, n_neg = 0, 0

    for frame_idx in visible_frames:
        gt_x, gt_y = markup[frame_idx]["x"], markup[frame_idx]["y"]

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

        # permissive call: every raw candidate, unfiltered by our hand-tuned
        # thresholds -- this IS the dataset the classifier needs to see
        candidates = find_candidates(
            motion_mask, curr_c,
            min_area=0, max_area=float("inf"), max_aspect_ratio=float("inf"),
            min_rel_v=-float("inf"), max_rel_s=float("inf"),
        )

        if not candidates:
            continue

        dists_this_frame = [math.hypot(c["x"] - gt_x, c["y"] - gt_y) for c in candidates]
        best_idx = int(np.argmin(dists_this_frame))

        for i, c in enumerate(candidates):
            is_positive = (i == best_idx) and (dists_this_frame[i] <= args.pos_thresh)
            patch = crop_patch(curr_c, c["x"], c["y"], args.patch_size)
            patches.append(patch)
            labels.append(1 if is_positive else 0)
            dists.append(dists_this_frame[i])
            frame_idxs.append(frame_idx)
            n_pos += int(is_positive)
            n_neg += int(not is_positive)

    cap.release()

    patches = np.array(patches, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    dists = np.array(dists, dtype=np.float32)
    frame_idxs = np.array(frame_idxs, dtype=np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, patches=patches, labels=labels, dists=dists, frame_idxs=frame_idxs)

    print(f"Saved {len(patches)} patches to {args.out}")
    print(f"  positives: {n_pos}  negatives: {n_neg}  ({100*n_pos/max(1,n_pos+n_neg):.1f}% positive)")


if __name__ == "__main__":
    main()
