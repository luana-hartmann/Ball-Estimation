"""
Evaluate a detected ball trajectory (from detect_ball_classical.py) against
the OpenTTGames ground-truth ball_markup.json.

For every frame, both sources say either "ball is here at (x,y)" or
"ball not visible" (-1,-1). We compare frame by frame and classify each
one into exactly one of four buckets:

    TP (true positive)  : both say visible, AND our position is close enough
    WRONG_POS            : both say visible, but our position is too far off
                            (we found *something*, just not the ball)
    FN (false negative)  : ground truth says visible, we said not found
    FP (false positive)  : we said visible, ground truth says not visible
    TN (true negative)   : both agree the ball isn't visible (not very
                            informative, we don't score on this)

From these counts we derive the metrics that actually matter for your
report: recall, precision, mean localization error, and the longest
streak of consecutive misses (which matters a lot for Alexandre's model,
since a long gap breaks trajectory continuity more than scattered
single-frame misses).

Usage:
    python eval_trajectory.py --pred trajectory.csv --markup ball_markup.json --threshold 10
"""

import argparse
import csv
import json
import math


def load_markup(markup_path):
    with open(markup_path, "r") as f:
        data = json.load(f)
    return {int(k): (v["x"], v["y"]) for k, v in data.items()}


def load_predictions(pred_path):
    preds = {}
    with open(pred_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            x, y = float(row["x"]), float(row["y"])
            preds[frame_idx] = (x, y)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="CSV output from detect_ball_classical.py")
    parser.add_argument("--markup", required=True, help="ball_markup.json ground truth")
    parser.add_argument("--threshold", type=float, default=10.0,
                         help="Max pixel distance to count a detection as correct")
    parser.add_argument("--per_frame_out", default="eval_per_frame.csv",
                         help="Where to save the per-frame breakdown")
    args = parser.parse_args()

    ground_truth = load_markup(args.markup)
    predictions = load_predictions(args.pred)

    # Only evaluate frames we actually have BOTH a ground-truth label for
    # and a prediction for. In practice ball_markup.json only labels frames
    # near events (see the dataset description), so this is usually a
    # subset of the full video.
    common_frames = sorted(set(ground_truth.keys()) & set(predictions.keys()))
    if not common_frames:
        print("No overlapping frames between predictions and ground truth. "
              "Did you run detection on the same video the markup file describes?")
        return

    per_frame_rows = []
    counts = {"TP": 0, "WRONG_POS": 0, "FN": 0, "FP": 0, "TN": 0}
    tp_errors = []          # pixel error for correctly matched detections
    miss_streak = 0         # current run length of FN (missed) frames
    longest_miss_streak = 0

    for frame_idx in common_frames:
        gt_x, gt_y = ground_truth[frame_idx]
        pred_x, pred_y = predictions[frame_idx]

        gt_visible = (gt_x != -1 and gt_y != -1)
        pred_visible = (pred_x != -1 and pred_y != -1)

        if gt_visible and pred_visible:
            dist = math.hypot(pred_x - gt_x, pred_y - gt_y)
            if dist <= args.threshold:
                category = "TP"
                tp_errors.append(dist)
            else:
                category = "WRONG_POS"
                dist = dist  # keep for logging even though it's not a TP
        elif gt_visible and not pred_visible:
            category = "FN"
            dist = None
        elif not gt_visible and pred_visible:
            category = "FP"
            dist = None
        else:
            category = "TN"
            dist = None

        counts[category] += 1

        # track longest run of missed frames (FN specifically -- ball was
        # there and we lost it, which is what breaks trajectory continuity)
        if category == "FN":
            miss_streak += 1
            longest_miss_streak = max(longest_miss_streak, miss_streak)
        else:
            miss_streak = 0

        per_frame_rows.append({
            "frame_idx": frame_idx,
            "category": category,
            "gt_x": gt_x, "gt_y": gt_y,
            "pred_x": pred_x, "pred_y": pred_y,
            "pixel_error": dist if dist is not None else "",
        })

    # ---- save per-frame breakdown for later plotting / report figures ----
    with open(args.per_frame_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_frame_rows)

    # ---- summary metrics ----
    n_gt_visible = counts["TP"] + counts["WRONG_POS"] + counts["FN"]
    n_pred_visible = counts["TP"] + counts["WRONG_POS"] + counts["FP"]

    recall = counts["TP"] / n_gt_visible if n_gt_visible else float("nan")
    precision = counts["TP"] / n_pred_visible if n_pred_visible else float("nan")
    mean_error = sum(tp_errors) / len(tp_errors) if tp_errors else float("nan")

    print(f"Evaluated {len(common_frames)} frames (threshold = {args.threshold}px)\n")
    print(f"Ground-truth visible frames : {n_gt_visible}")
    print(f"  TP  (correct detection)   : {counts['TP']}")
    print(f"  WRONG_POS (found, but off): {counts['WRONG_POS']}")
    print(f"  FN  (missed entirely)     : {counts['FN']}")
    print(f"Ground-truth NOT visible    : {counts['FP'] + counts['TN']}")
    print(f"  FP  (false alarm)         : {counts['FP']}")
    print(f"  TN  (correctly quiet)     : {counts['TN']}\n")
    print(f"Recall    : {recall:.3f}  (fraction of real ball appearances we caught)")
    print(f"Precision : {precision:.3f}  (fraction of our detections that were correct)")
    print(f"Mean pixel error on correct detections: {mean_error:.2f}px")
    print(f"Longest consecutive missed-frame streak: {longest_miss_streak}")
    print(f"\nPer-frame breakdown saved to {args.per_frame_out}")


if __name__ == "__main__":
    main()
