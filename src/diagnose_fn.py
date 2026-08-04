"""
For frames where we MISSED the ball entirely (FN), this finds the raw
motion contour closest to the ground-truth position and reports whether
the CURRENT detector (detector.py's find_candidates) would accept or
reject it, and why.

Unlike earlier versions of this script, this one calls detector.py's
actual find_candidates() function directly -- twice: once with
pass-through thresholds (to measure every contour's stats, filtered or
not) and once with the real thresholds (to know what the detector
actually decided). This guarantees the diagnosis can never silently
drift out of sync with the real detector logic again (see project
history: this exact bug happened twice before with hand-duplicated
filter logic).

  - "no contour found near GT" -> stage 1 (motion detection) never even
    segmented the ball as something that moved.
  - "found, but failed filters" -> stage 1 worked, stage 2 (shape/color
    filtering) is rejecting a legitimate candidate. Fixable case.

Usage:
    python diagnose_fn.py --video ../data/test_2/test_2.mp4 --eval_csv ../outputs/eval/eval_per_frame_test2.csv --n 15
"""

import argparse
import csv
import math
import os

import cv2

from detector import three_frame_diff, find_candidates


def analyze_frame(gray_prev, gray_curr, gray_next, curr_color, gt_x, gt_y,
                   min_area, max_area, max_aspect_ratio, min_rel_v, max_rel_s):
    motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)

    # pass-through call: get EVERY contour's real measurements, unfiltered
    all_candidates = find_candidates(
        motion_mask, curr_color,
        min_area=0, max_area=float("inf"), max_aspect_ratio=float("inf"),
        min_rel_v=-float("inf"), max_rel_s=float("inf"),
    )

    best, best_dist = None, None
    for c in all_candidates:
        dist = math.hypot(c["x"] - gt_x, c["y"] - gt_y)
        if best_dist is None or dist < best_dist:
            best, best_dist = c, dist

    if best is None:
        return None, len(all_candidates), motion_mask

    # real call: what the actual detector would keep with real thresholds
    real_candidates = find_candidates(
        motion_mask, curr_color,
        min_area=min_area, max_area=max_area, max_aspect_ratio=max_aspect_ratio,
        min_rel_v=min_rel_v, max_rel_s=max_rel_s,
    )
    passed = any(math.hypot(c["x"] - best["x"], c["y"] - best["y"]) < 1.0 for c in real_candidates)

    reasons = []
    if not passed:
        if best["area"] < min_area or best["area"] > max_area:
            reasons.append(f"area={best['area']:.0f} outside [{min_area},{max_area}]")
        # aspect ratio isn't stored on the candidate dict by find_candidates,
        # so we can only report rel_v/rel_s/area here -- still authoritative
        # for the pass/fail decision itself, which is what matters.
        if best["rel_v"] < min_rel_v:
            reasons.append(f"rel_v={best['rel_v']:.1f} < {min_rel_v} (not bright enough vs surroundings)")
        if best["rel_s"] > max_rel_s:
            reasons.append(f"rel_s={best['rel_s']:.1f} > {max_rel_s} (too saturated vs surroundings)")
        if not reasons:
            reasons.append("aspect_ratio too high (elongated beyond max_aspect_ratio)")

    best["dist"] = best_dist
    best["reasons"] = reasons
    return best, len(all_candidates), motion_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--near_thresh", type=float, default=25,
                         help="If nearest contour is farther than this from GT, treat as 'not found at all'")
    parser.add_argument("--min_area", type=float, default=4)
    parser.add_argument("--max_area", type=float, default=650)
    parser.add_argument("--max_aspect_ratio", type=float, default=3.2)
    parser.add_argument("--min_rel_v", type=float, default=5.0)
    parser.add_argument("--max_rel_s", type=float, default=-4.0)
    parser.add_argument("--out_dir", default="../outputs/diagnosis/fn_diagnosis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
    filtered_out = 0

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

        best, n_contours, motion_mask = analyze_frame(
            gray_prev, gray_curr, gray_next, curr_c, gt_x, gt_y,
            args.min_area, args.max_area, args.max_aspect_ratio,
            args.min_rel_v, args.max_rel_s,
        )

        print(f"\nframe {frame_idx}: {n_contours} raw motion contours total")
        if best is None or best["dist"] > args.near_thresh:
            not_found_at_all += 1
            dist_str = f"{best['dist']:.0f}px" if best else "n/a"
            print(f"  -> NOT FOUND AT ALL near GT (nearest contour is {dist_str} away). "
                  f"Stage 1 (motion detection) likely missed it.")
        else:
            if best["reasons"]:
                filtered_out += 1
                print(f"  -> Found at {best['dist']:.1f}px from GT, but REJECTED by filters:")
                for r in best["reasons"]:
                    print(f"       - {r}")
            else:
                print(f"  -> Found at {best['dist']:.1f}px from GT and PASSED all filters "
                      f"(this shouldn't be an FN -- check stage 3 tracking logic / max_pred_dist)")

        r = 60
        h, w = curr_c.shape[:2]
        x0, x1 = max(0, int(gt_x) - r), min(w, int(gt_x) + r)
        y0, y1 = max(0, int(gt_y) - r), min(h, int(gt_y) + r)
        crop = curr_c[y0:y1, x0:x1].copy()
        mask_crop = motion_mask[y0:y1, x0:x1]
        overlay = crop.copy()
        overlay[mask_crop > 0] = (0, 255, 255)
        blended = cv2.addWeighted(overlay, 0.4, crop, 0.6, 0)
        cv2.drawMarker(blended, (int(gt_x) - x0, int(gt_y) - y0), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.imwrite(os.path.join(args.out_dir, f"fn_{frame_idx}.jpg"), blended)

    print(f"\n--- Summary over {len(sample)} sampled FN frames ---")
    print(f"Not found at all (motion stage missed it): {not_found_at_all}")
    print(f"Found but filtered out (too strict): {filtered_out}")
    print(f"Saved crops to {args.out_dir}/")


if __name__ == "__main__":
    main()
