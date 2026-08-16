"""
Overlays the detected ball trajectory on top of the original video, with
a fading trail showing recent motion -- produces a short, visually
convincing demo clip instead of raw numbers.

Usage:
    python src/visualize_trajectory.py --video data/test6/test6.mp4 --trajectory outputs/trajectories/trajectory_test6_ml_aug.csv --out outputs/demo_test6.mp4 --start_frame 900 --end_frame 1050
"""

import argparse
import csv

import cv2


def load_trajectory(path):
    traj = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            frame_idx = int(row["frame_idx"])
            x, y = float(row["x"]), float(row["y"])
            if x != -1 and y != -1:
                traj[frame_idx] = (x, y)
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", default="demo.mp4")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=300,
                         help="Keep this short (a few hundred frames = a few seconds) for a punchy demo clip")
    parser.add_argument("--tail_length", type=int, default=15,
                         help="How many previous positions to show as a fading trail")
    parser.add_argument("--output_fps", type=float, default=None,
                         help="Playback fps for the output file. Omit to match the source video. "
                              "Set lower than the source fps (e.g. 30 for 120fps source) for a "
                              "slow-motion effect -- makes fast rallies much easier to follow in a demo.")
    args = parser.parse_args()

    traj = load_trajectory(args.trajectory)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    write_fps = args.output_fps if args.output_fps else fps
    out = cv2.VideoWriter(args.out, fourcc, write_fps, (w, h))

    recent_positions = []  # list of (frame_idx, x, y), most recent last

    for frame_idx in range(args.start_frame, args.end_frame):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in traj:
            recent_positions.append((frame_idx,) + traj[frame_idx])
        recent_positions = [p for p in recent_positions if frame_idx - p[0] <= args.tail_length]

        # fading trail: older points smaller and more transparent
        n = len(recent_positions)
        for i, (f_i, x, y) in enumerate(recent_positions):
            age_frac = (n - i) / max(n, 1)  # 1.0 = most recent, smaller = older
            radius = max(2, int(10 * age_frac))
            overlay = frame.copy()
            cv2.circle(overlay, (int(x), int(y)), radius, (0, 255, 0), -1)
            alpha = 0.15 + 0.7 * age_frac
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # current position: solid marker + crosshair for visibility
        if recent_positions and recent_positions[-1][0] == frame_idx:
            _, x, y = recent_positions[-1]
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)
            cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved demo video to {args.out}")


if __name__ == "__main__":
    main()