"""
Reprints the summary block (recall/precision/mean error/longest miss
streak) from an already-generated eval_per_frame_*.csv, without rerunning
detection or evaluation. Meant to be called unconditionally from the
Makefile so `make eval`/`make all` always show results, even when Make
skipped regenerating the underlying files because they're already
up to date.

Usage:
    python src/summarize_eval.py --eval_csv outputs/eval/eval_per_frame_test2.csv
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", required=True)
    args = parser.parse_args()

    counts = {"TP": 0, "WRONG_POS": 0, "FN": 0, "FP": 0, "TN": 0}
    tp_errors = []
    miss_streak = 0
    longest_miss_streak = 0

    with open(args.eval_csv, "r") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        cat = row["category"]
        counts[cat] += 1
        if cat == "TP" and row["pixel_error"]:
            tp_errors.append(float(row["pixel_error"]))
        if cat == "FN":
            miss_streak += 1
            longest_miss_streak = max(longest_miss_streak, miss_streak)
        else:
            miss_streak = 0

    n_gt_visible = counts["TP"] + counts["WRONG_POS"] + counts["FN"]
    n_pred_visible = counts["TP"] + counts["WRONG_POS"] + counts["FP"]
    recall = counts["TP"] / n_gt_visible if n_gt_visible else float("nan")
    precision = counts["TP"] / n_pred_visible if n_pred_visible else float("nan")
    mean_error = sum(tp_errors) / len(tp_errors) if tp_errors else float("nan")

    print(f"[{args.eval_csv}]")
    print(f"  Recall: {recall:.3f}  Precision: {precision:.3f}  "
          f"Mean error: {mean_error:.2f}px  Longest miss streak: {longest_miss_streak}")
    print(f"  TP={counts['TP']} WRONG_POS={counts['WRONG_POS']} FN={counts['FN']} "
          f"FP={counts['FP']} TN={counts['TN']}")


if __name__ == "__main__":
    main()
