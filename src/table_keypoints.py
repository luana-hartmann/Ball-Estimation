"""
Robust Table Keypoint Detection (13 keypoints per Kienzle et al. CVPRW 2025).

Outputs are structured into:
  - outputs/keypoints/vis/
  - outputs/keypoints/masks/
  - outputs/keypoints/json/

Usage:
    # Batch run all tests automatically:
    python src/table_keypoints.py --batch --data_dir data --out_dir outputs/keypoints

    # Interactive adjustment (4 clicks: Top-Left, Top-Right, Bottom-Right, Bottom-Left):
    python src/table_keypoints.py --video data/test2/test2.mp4 --frame 50 --interactive --out_dir outputs/keypoints
"""

import argparse
import glob
import json
import os
import cv2
import numpy as np

# Official ITTF Table Dimensions (in cm)
TABLE_WIDTH_CM = 152.5    # along X (across table, left-to-right from camera view)
TABLE_LENGTH_CM = 274.0   # along Y (down table length, far-to-near from camera view)
NET_HEIGHT_CM = 15.25
NET_OVERHANG_CM = 15.25


def find_table_corners_auto(frame, roi_top_frac=0.45):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([95, 80, 50])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask[: int(h * roi_top_frac), :] = 0
    mask[:, : int(w * 0.05)] = 0
    mask[:, int(w * 0.95):] = 0
    mask[int(h * 0.88):, :] = 0

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    valid = [c for c in contours if cv2.contourArea(c) > (h * w * 0.03)]
    if not valid:
        return None, mask

    largest = max(valid, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = None
    for eps in [0.015, 0.02, 0.03, 0.04, 0.06]:
        cand = cv2.approxPolyDP(largest, eps * peri, True)
        if len(cand) == 4:
            approx = cand
            break

    if approx is None:
        rect = cv2.minAreaRect(largest)
        approx = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

    pts = approx.reshape(4, 2).astype(np.float32)
    pts = pts[np.argsort(pts[:, 1])]
    far_pair = pts[:2][np.argsort(pts[:2, 0])]
    near_pair = pts[2:][np.argsort(pts[2:, 0])]

    return {
        "far_left": far_pair[0],
        "far_right": far_pair[1],
        "near_left": near_pair[0],
        "near_right": near_pair[1],
    }, mask


def compute_all_13_keypoints(corners):
    """
    Computes all 13 canonical keypoints via planar homography mapping.
    Broadcast camera geometry:
      - X-axis: Table LENGTH (274.0 cm, from Left Player to Right Player)
      - Y-axis: Table WIDTH (152.5 cm, from Far edge to Near edge)
      - Net: Runs vertically at X = 137.0 cm (across the width Y)
      - Net Posts: Physically located at the Far (top) and Near (bottom) ends of the net
    """
    TABLE_LENGTH_CM = 274.0   # Left to Right
    TABLE_WIDTH_CM = 152.5    # Far to Near
    NET_HEIGHT_CM = 15.25
    NET_OVERHANG_CM = 15.25

    # 1. Canonical 2D coordinates of the 4 table corners
    src_canonical = np.array([
        [0.0, 0.0],                          # Far Left
        [TABLE_LENGTH_CM, 0.0],              # Far Right
        [TABLE_LENGTH_CM, TABLE_WIDTH_CM],   # Near Right
        [0.0, TABLE_WIDTH_CM]                # Near Left
    ], dtype=np.float32)

    dst_image = np.array([
        corners["far_left"],
        corners["far_right"],
        corners["near_right"],
        corners["near_left"]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_canonical, dst_image)

    half_l = TABLE_LENGTH_CM / 2.0  # 137.0 cm
    half_w = TABLE_WIDTH_CM / 2.0   # 76.25 cm

    planar_points = {
        "corner_far_left": [0.0, 0.0],
        "corner_far_right": [TABLE_LENGTH_CM, 0.0],
        "corner_near_right": [TABLE_LENGTH_CM, TABLE_WIDTH_CM],
        "corner_near_left": [0.0, TABLE_WIDTH_CM],
        "centerline_far_edge": [0.0, half_w],             # Left edge centerline
        "centerline_near_edge": [TABLE_LENGTH_CM, half_w], # Right edge centerline
        "center": [half_l, half_w],                       # Exact table center
        "net_left_edge": [half_l, 0.0],                   # Net at Far top edge
        "net_right_edge": [half_l, TABLE_WIDTH_CM],       # Net at Near bottom edge
        "net_post_left_base": [half_l, -NET_OVERHANG_CM], # Far net post base (extends top)
        "net_post_right_base": [half_l, TABLE_WIDTH_CM + NET_OVERHANG_CM], # Near net post base (extends bottom)
    }

    pts_names = list(planar_points.keys())
    pts_coords = np.array([planar_points[k] for k in pts_names], dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts_coords, H).reshape(-1, 2)

    keypoints = {}
    for name, pt in zip(pts_names, projected):
        keypoints[name] = (float(pt[0]), float(pt[1]))

    # Post tops: vertical elevation in image plane
    depth_left = np.linalg.norm(np.array(corners["near_left"]) - np.array(corners["far_left"]))
    depth_right = np.linalg.norm(np.array(corners["near_right"]) - np.array(corners["far_right"]))
    avg_depth = (depth_left + depth_right) / 2.0
    post_h = (NET_HEIGHT_CM / TABLE_WIDTH_CM) * avg_depth

    far_base_x, far_base_y = keypoints["net_post_left_base"]
    near_base_x, near_base_y = keypoints["net_post_right_base"]

    keypoints["net_post_left_top"] = (far_base_x, far_base_y - post_h)
    keypoints["net_post_right_top"] = (near_base_x, near_base_y - post_h)

    return keypoints


def draw_keypoints_vis(frame, corners, keypoints):
    vis = frame.copy()
    table_poly = np.array([
        corners["far_left"], corners["far_right"],
        corners["near_right"], corners["near_left"]
    ], dtype=np.int32)
    cv2.polylines(vis, [table_poly], isClosed=True, color=(0, 255, 255), thickness=2)

    c_far = (int(keypoints["centerline_far_edge"][0]), int(keypoints["centerline_far_edge"][1]))
    c_near = (int(keypoints["centerline_near_edge"][0]), int(keypoints["centerline_near_edge"][1]))
    cv2.line(vis, c_far, c_near, (255, 255, 0), 2)

    n_left = (int(keypoints["net_left_edge"][0]), int(keypoints["net_left_edge"][1]))
    n_right = (int(keypoints["net_right_edge"][0]), int(keypoints["net_right_edge"][1]))
    cv2.line(vis, n_left, n_right, (0, 165, 255), 2)

    for name, (x, y) in keypoints.items():
        color = (0, 0, 255) if "post" not in name else (255, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 6, color, -1)
        cv2.circle(vis, (int(x), int(y)), 8, (255, 255, 255), 1)

    return vis


def run_interactive(frame, initial_corners=None):
    manual_clicks = []
    current_corners = initial_corners
    window_name = "Table Keypoint Setup (Click 4 corners: TL, TR, BR, BL | ENTER=Save, R=Reset, ESC=Exit)"

    def on_mouse(event, x, y, flags, param):
        nonlocal manual_clicks, current_corners
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
            kp = compute_all_13_keypoints(current_corners)
            vis = draw_keypoints_vis(frame, current_corners, kp)
        else:
            vis = frame.copy()
            for idx, pt in enumerate(manual_clicks):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
                cv2.putText(vis, f"P{idx+1}", (int(pt[0]) + 10, int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 32]:
            break
        elif key in [ord("r"), ord("R")]:
            manual_clicks = []
            current_corners = None
            print("[*] Reset. Click 4 corners in order: Far-Left, Far-Right, Near-Right, Near-Left")
        elif key == 27:
            current_corners = None
            break

    cv2.destroyAllWindows()
    return current_corners


def process_single_video(video_path, frame_idx, out_dir, roi_top_frac=0.45, interactive=False):
    vis_dir = os.path.join(out_dir, "vis")
    masks_dir = os.path.join(out_dir, "masks")
    json_dir = os.path.join(out_dir, "json")
    for d in [vis_dir, masks_dir, json_dir]:
        os.makedirs(d, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_vis = os.path.join(vis_dir, f"{base_name}_keypoints.jpg")
    out_mask = os.path.join(masks_dir, f"{base_name}_mask.jpg")
    out_json = os.path.join(json_dir, f"{base_name}_keypoints.json")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"[-] Could not read frame {frame_idx} from {video_path}")
        return None

    corners, mask = find_table_corners_auto(frame, roi_top_frac=roi_top_frac)
    cv2.imwrite(out_mask, mask)

    if interactive:
        corners = run_interactive(frame, initial_corners=corners)

    if corners is None:
        print(f"[-] Detection failed for {base_name}")
        return None

    keypoints = compute_all_13_keypoints(corners)
    vis = draw_keypoints_vis(frame, corners, keypoints)

    cv2.imwrite(out_vis, vis)
    with open(out_json, "w") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in keypoints.items()}, f, indent=2)

    print(f"[✓] Processed {base_name:10s} -> Saved to {out_json}")
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Process all videos in data_dir")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--video", default=None)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--roi_top_frac", type=float, default=0.45)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--out_dir", default="outputs/keypoints")
    args = parser.parse_args()

    if args.batch:
        video_files = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.mp4")))
        print(f"Found {len(video_files)} videos. Processing all in batch...\n")
        rendered_previews = []

        for vpath in video_files:
            test_name = os.path.basename(os.path.dirname(vpath))
            vis = process_single_video(vpath, args.frame, args.out_dir, args.roi_top_frac, interactive=False)
            if vis is not None:
                small = cv2.resize(vis, (640, 360))
                cv2.putText(small, test_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                rendered_previews.append(small)

        if rendered_previews:
            n = len(rendered_previews)
            cols = 2
            rows = (n + cols - 1) // cols
            grid = np.zeros((rows * 360, cols * 640, 3), dtype=np.uint8)
            for idx, img in enumerate(rendered_previews):
                r, c = idx // cols, idx % cols
                grid[r * 360:(r + 1) * 360, c * 640:(c + 1) * 640] = img
            grid_out = os.path.join(args.out_dir, "vis", "ALL_TESTS_GRID_COMPARISON.jpg")
            cv2.imwrite(grid_out, grid)
            print(f"\n[✓] Saved complete comparison grid of all tests to: {grid_out}")
    else:
        if not args.video:
            print("Please specify --video <path> or use --batch")
            return
        process_single_video(args.video, args.frame, args.out_dir, args.roi_top_frac, interactive=args.interactive)


if __name__ == "__main__":
    main()