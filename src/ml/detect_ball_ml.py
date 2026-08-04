"""
Same three-stage pipeline as detector.py, but stage 2 (candidate
filtering) is replaced by the trained CNN classifier instead of hand-
tuned brightness/contrast thresholds. Stage 1 (motion detection) and
stage 3 (tracking) are otherwise unchanged, so any difference in results
vs. detector.py isolates the effect of learned vs. hand-tuned filtering.

Usage:
    python src/ml/detect_ball_ml.py --video ../data/test6/test6.mp4 --model ../outputs/ml_dataset/ball_classifier.pt --out ../outputs/trajectories/trajectory_test6_ml.csv
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
from detector import three_frame_diff, find_candidates as find_raw_candidates  # noqa: E402
from model import BallPatchCNN  # noqa: E402


def crop_patch(frame, cx, cy, patch_size=32):
    h, w = frame.shape[:2]
    half = patch_size // 2
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = x0 + patch_size, y0 + patch_size

    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = frame[y0c:y1c, x0c:x1c]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
        crop = cv2.resize(crop, (patch_size, patch_size))
    return crop


def classify_candidates(candidates, curr_frame, model, device, patch_size=32, prob_thresh=0.5):
    """
    Runs every raw candidate's patch through the CNN, keeps only the ones
    classified "ball" with confidence >= prob_thresh. prob_thresh is a
    tunable knob, same role as the hand-tuned thresholds in detector.py --
    lower it to favor recall, raise it to favor precision.
    """
    if not candidates:
        return []

    patches = np.stack([crop_patch(curr_frame, c["x"], c["y"], patch_size) for c in candidates])
    patches_t = torch.from_numpy(patches.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        logits = model(patches_t)
        probs = F.softmax(logits, dim=1)[:, 1]  # probability of class "ball"

    kept = []
    for c, p in zip(candidates, probs.cpu().numpy()):
        if p >= prob_thresh:
            kept.append({**c, "ball_prob": float(p)})
    return kept


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
    Same shape as detector.py's tracker, but instead of a hand-crafted
    typicality score, we use the CNN's own confidence: prefer whichever
    candidate is both close to the prediction AND confidently "ball"
    according to the model.
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
            score = dist * dist_weight - c["ball_prob"]  # lower is better: close AND confident
            scored.append((score, dist, c))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        _, best_dist, best = scored[0]
        confidence = best["ball_prob"]
        return (best["x"], best["y"], confidence)

    best = max(candidates, key=lambda c: c["ball_prob"])
    return (best["x"], best["y"], best["ball_prob"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="trajectory_ml.csv")
    parser.add_argument("--prob_thresh", type=float, default=0.5)
    parser.add_argument("--max_pred_dist", type=float, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallPatchCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

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

    history, results = [], []
    frame_idx = 1

    while True:
        ok_next, frame_next_color = cap.read()
        if not ok_next:
            break
        gray_next = cv2.cvtColor(frame_next_color, cv2.COLOR_BGR2GRAY)

        motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
        raw_candidates = find_raw_candidates(
            motion_mask, frame_curr_color,
            min_area=0, max_area=float("inf"), max_aspect_ratio=float("inf"),
            min_rel_v=-float("inf"), max_rel_s=float("inf"),
        )
        candidates = classify_candidates(raw_candidates, frame_curr_color, model, device,
                                          prob_thresh=args.prob_thresh)

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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_s", "x", "y", "confidence"])
        writer.writerows(results)

    n_found = sum(1 for r in results if r[3] != -1)
    print(f"Processed {len(results)} frames, ball found in {n_found} ({100*n_found/len(results):.1f}%).")
    print(f"Saved trajectory to {args.out}")


if __name__ == "__main__":
    main()
