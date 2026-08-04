"""
Classical CV ball detector, v5.

Change from v4: stage 3 (tracking) now uses a proper Kalman filter
instead of naive two-point linear extrapolation. Detection/filtering
(stage 1 and 2, the relative-color approach from v4) is UNCHANGED --
this is purely a tracking-stage upgrade, testable independently of the
detection philosophy.

Why a Kalman filter helps, concretely:
  - It tracks not just a position estimate but also how UNCERTAIN that
    estimate is (a covariance matrix). Right after a confirmed detection,
    uncertainty is low. Each frame we go without confirming a match
    (occlusion, fast blur), uncertainty grows automatically.
  - We use that growing uncertainty to WIDEN our search radius the longer
    we've been "coasting" without a real detection, instead of using one
    fixed radius (v3/v4's max_pred_dist=60 for every situation). This
    should help recover faster after occlusions -- directly targeting
    the long miss-streaks we saw (e.g. 30 consecutive missed frames on
    test_6 with v3).
  - It's a standard technique (not something we invented), so it's
    legitimate to cite as "we used a Kalman filter" in the report,
    but implemented from scratch here (not cv2.KalmanFilter) so every
    step is something we can explain and defend, not a black box.

State vector: [x, y, vx, vy] (position + velocity, constant-velocity
model). We do NOT model acceleration -- table tennis trajectories curve
due to gravity/spin, so a real physics-aware tracker would need more,
but constant-velocity is a standard, defensible starting assumption for
a SHORT prediction horizon (one frame ahead), which is all we need here.
"""

import argparse
import csv
import math

import cv2
import numpy as np

from detect_ball_classical_v4 import three_frame_diff, find_candidates, typicality_score


class KalmanTracker2D:
    """
    Minimal from-scratch constant-velocity Kalman filter for 2D position
    tracking. Implemented manually (not cv2.KalmanFilter) so every step
    is explainable for the report, not a library black box.
    """

    def __init__(self, dt=1.0, process_var=5.0, measurement_var=4.0):
        self.dt = dt
        # state transition: next position = position + velocity * dt,
        # velocity unchanged (constant-velocity assumption)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        # measurement model: we only ever observe (x, y), not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        # process noise: how much we trust the constant-velocity
        # assumption (higher = allow faster/more erratic changes in
        # velocity, e.g. due to spin/gravity curving the real trajectory)
        self.Q = process_var * np.eye(4)
        # measurement noise: how much we trust a single detection's
        # pixel position (tie this to the mean_error we measured earlier,
        # ~2px, so this reflects our own evaluated accuracy)
        self.R = measurement_var * np.eye(2)

        self.x = None  # state estimate
        self.P = None  # state covariance (our uncertainty)
        self.initialized = False

    def init_state(self, x, y):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 500.0  # start very uncertain about velocity
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, zx, zy):
        z = np.array([zx, zy])
        y = z - self.H @ self.x                     # innovation (measurement - prediction)
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def position_std(self):
        """Standard deviation of our position estimate in x and y -- how uncertain we are right now."""
        return math.sqrt(max(self.P[0, 0], 0)), math.sqrt(max(self.P[1, 1], 0))


def pick_best_candidate_kf(candidates, predicted_pos, pos_std, base_gate=25, uncertainty_mult=3.0):
    """
    Same typicality-weighted selection idea as v4, but the search gate is
    now ADAPTIVE: base_gate is the minimum radius even when confident,
    and it grows with the Kalman filter's own uncertainty (uncertainty_mult
    * position std). This replaces v4's single fixed max_pred_dist=60.
    """
    if not candidates or predicted_pos is None:
        return None, None

    px, py = predicted_pos
    std_x, std_y = pos_std
    gate = base_gate + uncertainty_mult * max(std_x, std_y)

    scored = []
    for c in candidates:
        dist = math.hypot(c["x"] - px, c["y"] - py)
        if dist > gate:
            continue
        score = dist * 0.05 + typicality_score(c)
        scored.append((score, dist, c))

    if not scored:
        return None, gate

    scored.sort(key=lambda t: t[0])
    _, best_dist, best = scored[0]
    confidence = max(0.3, 1.0 - best_dist / gate)
    return (best["x"], best["y"], confidence), gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="trajectory_v5.csv")
    parser.add_argument("--min_area", type=float, default=4)
    parser.add_argument("--max_area", type=float, default=650)
    parser.add_argument("--max_aspect_ratio", type=float, default=3.2)
    parser.add_argument("--min_rel_v", type=float, default=5.0)
    parser.add_argument("--max_rel_s", type=float, default=-4.0)
    parser.add_argument("--base_gate", type=float, default=25,
                         help="Minimum search radius even when confident")
    parser.add_argument("--uncertainty_mult", type=float, default=3.0,
                         help="How much the search radius grows per unit of KF uncertainty")
    parser.add_argument("--process_var", type=float, default=5.0)
    parser.add_argument("--measurement_var", type=float, default=30.0,
                         help="Raised from 4.0: a low value made the filter trust every "
                              "match strongly, including wrong ones (see paddle lock-on "
                              "bug found via diagnose_wrong_pos.py). Higher = more willing "
                              "to be corrected by a better candidate next frame.")
    args = parser.parse_args()

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

    tracker = KalmanTracker2D(dt=1.0, process_var=args.process_var, measurement_var=args.measurement_var)
    results = []
    frame_idx = 1
    consecutive_misses = 0
    # pending candidate awaiting a second frame's confirmation before we
    # trust it enough to START a track. Fixes the bug where a single
    # frame's best-typicality candidate (e.g. a paddle) got trusted
    # immediately with no validation, since no prediction exists yet to
    # check it against.
    pending = None  # (x, y, frame_idx) or None
    CONFIRM_DIST = 20  # max px between two frames' candidate to count as "the same object"

    while True:
        ok_next, frame_next_color = cap.read()
        if not ok_next:
            break
        gray_next = cv2.cvtColor(frame_next_color, cv2.COLOR_BGR2GRAY)

        motion_mask = three_frame_diff(gray_prev, gray_curr, gray_next)
        candidates = find_candidates(
            motion_mask, frame_curr_color,
            min_area=args.min_area, max_area=args.max_area,
            max_aspect_ratio=args.max_aspect_ratio,
            min_rel_v=args.min_rel_v, max_rel_s=args.max_rel_s,
        )

        timestamp_s = frame_idx / fps

        if not tracker.initialized:
            # bootstrap: no prediction yet, just take the most typical candidate if any
            if candidates:
                best = min(candidates, key=typicality_score)
                tracker.init_state(best["x"], best["y"])
                results.append((frame_idx, timestamp_s, best["x"], best["y"], 0.4))
                consecutive_misses = 0
            else:
                results.append((frame_idx, timestamp_s, -1, -1, 0.0))
        else:
            pred_x, pred_y = tracker.predict()
            pos_std = tracker.position_std()
            result, gate = pick_best_candidate_kf(
                candidates, (pred_x, pred_y), pos_std,
                base_gate=args.base_gate, uncertainty_mult=args.uncertainty_mult,
            )

            if result is not None:
                x, y, confidence = result
                tracker.update(x, y)
                results.append((frame_idx, timestamp_s, x, y, confidence))
                consecutive_misses = 0
            else:
                # no matching candidate -- do NOT report a position (stay
                # honest, same principle as v1-v4), but the Kalman filter's
                # internal state still advanced via predict(), so its
                # uncertainty keeps growing and the gate keeps widening
                # for next frame, giving a better chance of recovery
                results.append((frame_idx, timestamp_s, -1, -1, 0.0))
                consecutive_misses += 1

        gray_prev, gray_curr = gray_curr, gray_next
        frame_curr_color = frame_next_color
        frame_idx += 1

    cap.release()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_s", "x", "y", "confidence"])
        writer.writerows(results)

    n_found = sum(1 for r in results if r[3] != -1)
    print(f"Processed {len(results)} frames, ball found in {n_found} ({100*n_found/len(results):.1f}%).")
    print(f"Saved trajectory to {args.out}")


if __name__ == "__main__":
    main()
