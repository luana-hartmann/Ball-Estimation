"""
Generates a single figure: a full video frame with the raw motion mask
(from three_frame_diff, before any filtering) overlaid in semi-transparent
yellow. Meant to illustrate that stage 1 alone produces multiple candidate
blobs (ball AND paddles/hands), motivating stage 2's filtering.

Usage:
    python src/make_motion_mask_figure.py --video data/test2/test2.mp4 --frame 923 --out outputs/diagnosis/motion_mask_example.jpg
"""

import argparse

import cv2

from detector import three_frame_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True,
                         help="Frame index to illustrate (pick one where the ball is mid-rally, clearly visible)")
    parser.add_argument("--out", default="outputs/diagnosis/motion_mask_example.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame - 1)
    ok1, prev_c = cap.read()
    ok2, curr_c = cap.read()
    ok3, next_c = cap.read()
    cap.release()
    if not (ok1 and ok2 and ok3):
        print("Could not read the requested frame triplet.")
        return

    gray_prev = cv2.cvtColor(prev_c, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_c, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(next_c, cv2.COLOR_BGR2GRAY)
    motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)

    overlay = curr_c.copy()
    overlay[motion_mask > 0] = (0, 255, 255)  # yellow
    blended = cv2.addWeighted(overlay, 0.5, curr_c, 0.5, 0)

    cv2.imwrite(args.out, blended)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
