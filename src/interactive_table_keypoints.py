"""
Interactive & Automatic Table Keypoint Tool (13 keypoints per Kienzle et al. CVPRW 2025).

If automatic detection succeeds, press ENTER/SPACE to accept.
If corners need adjustment, click the 4 corners manually in order:
  1: Far-Left (top-left)
  2: Far-Right (top-right)
  3: Near-Right (bottom-right)
  4: Near-Left (bottom-left)

Usage:
    python src/interactive_table_keypoints.py --video data/test2/test2.mp4 --frame 50 --out outputs/keypoints_test2.jpg --json_out outputs/keypoints_test2.json
"""

import argparse
import json
import os
import cv2
import numpy as np

from table_keypoints import (
    find_table_corners,
    compute_all_13_keypoints,
    TABLE_LENGTH_CM,
    TABLE_WIDTH_CM,
)


def render_keypoints_on_frame(frame, corners):
    keypoints = compute_all_13_keypoints(corners)
    vis = frame.copy()

    # Draw table border polygon
    table_poly = np.array([
        corners["far_left"], corners["far_right"],
        corners["near_right"], corners["near_left"]
    ], dtype=np.int32)
    cv2.polylines(vis, [table_poly], isClosed=True, color=(0, 255, 255), thickness=2)

    # Draw centerline & net line
    c_far = (int(keypoints["centerline_far_edge"][0]), int(keypoints["centerline_far_edge"][1]))
    c_near = (int(keypoints["centerline_near_edge"][0]), int(keypoints["centerline_near_edge"][1]))
    cv2.line(vis, c_far, c_near, (255, 255, 0), 2)

    n_left = (int(keypoints["net_left_edge"][0]), int(keypoints["net_left_edge"][1]))
    n_right = (int(keypoints["net_right_edge"][0]), int(keypoints["net_right_edge"][1]))
    cv2.line(vis, n_left, n_right, (0, 165, 255), 2)

    # Draw keypoint dots
    for name, (x, y) in keypoints.items():
        color = (0, 0, 255) if "post" not in name else (255, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 6, color, -1)
        cv2.circle(vis, (int(x), int(y)), 8, (255, 255, 255), 1)

    return vis, keypoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--roi_top_frac", type=float, default=0.45)
    parser.add_argument("--out", default="outputs/keypoints.jpg")
    parser.add_argument("--json_out", default="outputs/keypoints.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"Could not read frame {args.frame} from {args.video}")
        return

    # Attempt automatic detection
    corners, _ = find_table_corners(frame, roi_top_frac=args.roi_top_frac)

    manual_clicks = []
    window_name = f"Keypoint Calibrator - {os.path.basename(args.video)} (ENTER=Save, R=Reset, ESC=Exit)"

    current_corners = corners

    def on_mouse(event, x, y, flags, param):
        nonlocal current_corners, manual_clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(manual_clicks) < 4:
                manual_clicks.append([float(x), float(y)])
                print(f"Point {len(manual_clicks)}/4: ({x}, {y})")

            if len(manual_clicks) == 4:
                current_corners = {
                    "far_left": manual_clicks[0],
                    "far_right": manual_clicks[1],
                    "near_right": manual_clicks[2],
                    "near_left": manual_clicks[3],
                }

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if current_corners is not None and len(manual_clicks) in [0, 4]:
            vis, keypoints = render_keypoints_on_frame(frame, current_corners)
        else:
            vis = frame.copy()
            for idx, pt in enumerate(manual_clicks):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
                cv2.putText(vis, f"P{idx+1}", (int(pt[0]) + 10, int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 32]:  # ENTER or SPACE to save
            if current_corners is not None:
                vis, keypoints = render_keypoints_on_frame(frame, current_corners)
                cv2.imwrite(args.out, vis)
                if args.json_out:
                    with open(args.json_out, "w") as f:
                        json.dump({k: [float(v[0]), float(v[1])] for k, v in keypoints.items()}, f, indent=2)
                print(f"[✓] Successfully saved keypoints to {args.out} and {args.json_out}")
                break
        elif key in [ord("r"), ord("R")]:  # Reset clicks
            manual_clicks = []
            current_corners = None
            print("[*] Reset points. Click the 4 corners: far-left, far-right, near-right, near-left.")
        elif key == 27:  # ESC to cancel
            print("Cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()