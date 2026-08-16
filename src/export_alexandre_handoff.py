"""
Prepares and bundles 2D ball trajectories + 13 static table keypoints
for 3D Trajectory & Spin estimation model (Kienzle et al. CVPRW 2025).

Outputs generated in handoff_alexandre/:
  - handoff_alexandre/tables/          (JSON copies of the 13 table keypoints)
  - handoff_alexandre/combined_csv/    (Full 28-dim per-frame concatenated CSV)
  - handoff_alexandre/tensors_npz/     (Ready-to-load NumPy arrays [T, 14, 2])

Usage:
    python src/export_alexandre_handoff.py \
        --trajectories_dir outputs/trajectories \
        --keypoints_dir outputs/keypoints/json \
        --out_dir handoff_alexandre
"""

import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

# Canonical ordering of the 13 table keypoints matching Kienzle et al.
KEYPOINT_ORDER = [
    "corner_far_left",
    "corner_far_right",
    "corner_near_right",
    "corner_near_left",
    "centerline_far_edge",
    "centerline_near_edge",
    "center",
    "net_left_edge",
    "net_right_edge",
    "net_post_far_base",
    "net_post_near_base",
    "net_post_far_top",
    "net_post_near_top",
]


def load_keypoints(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Return as flat (13, 2) array following standard order
    coords = []
    for k in KEYPOINT_ORDER:
        coords.append(data[k])
    return np.array(coords, dtype=np.float32)


def process_dataset(traj_csv, kp_json, out_dir):
    test_id = os.path.splitext(os.path.basename(traj_csv))[0].replace("trajectory_", "")

    df_traj = pd.read_csv(traj_csv)
    kp_array = load_keypoints(kp_json)  # shape: (13, 2)

    n_frames = len(df_traj)
    ball_xy = df_traj[["x", "y"]].to_numpy(dtype=np.float32)  # shape: (T, 2)

    # 1. Build Tensor (T, 14, 2): ball is index 0, keypoints are 1..13
    tensor_14pts = np.zeros((n_frames, 14, 2), dtype=np.float32)
    tensor_14pts[:, 0, :] = ball_xy
    tensor_14pts[:, 1:, :] = np.tile(kp_array, (n_frames, 1, 1))

    # 2. Build full 28-feature flat CSV table
    flat_cols = {}
    flat_cols["frame_idx"] = df_traj["frame_idx"]
    flat_cols["timestamp_s"] = df_traj["timestamp_s"]
    flat_cols["confidence"] = df_traj["confidence"]
    flat_cols["ball_x"] = df_traj["x"]
    flat_cols["ball_y"] = df_traj["y"]

    for idx, name in enumerate(KEYPOINT_ORDER):
        flat_cols[f"{name}_x"] = kp_array[idx, 0]
        flat_cols[f"{name}_y"] = kp_array[idx, 1]

    df_combined = pd.DataFrame(flat_cols)

    # Save outputs
    combined_csv_dir = os.path.join(out_dir, "combined_csv")
    npz_dir = os.path.join(out_dir, "tensors_npz")
    tables_dir = os.path.join(out_dir, "tables")
    for d in [combined_csv_dir, npz_dir, tables_dir]:
        os.makedirs(d, exist_ok=True)

    csv_out = os.path.join(combined_csv_dir, f"input_3d_{test_id}.csv")
    npz_out = os.path.join(npz_dir, f"input_3d_{test_id}.npz")
    json_out = os.path.join(tables_dir, f"table_keypoints_{test_id}.json")

    df_combined.to_csv(csv_out, index=False)
    np.savez_compressed(
        npz_out,
        inputs=tensor_14pts,                      # (T, 14, 2)
        inputs_flat=tensor_14pts.reshape(n_frames, 28), # (T, 28)
        frame_idx=df_traj["frame_idx"].to_numpy(),
        timestamp_s=df_traj["timestamp_s"].to_numpy(),
        confidence=df_traj["confidence"].to_numpy(),
        keypoint_names=["ball"] + KEYPOINT_ORDER,
    )
    with open(json_out, "w") as f:
        json.dump({k: kp_array[i].tolist() for i, k in enumerate(KEYPOINT_ORDER)}, f, indent=2)

    print(f"[✓] Processed {test_id:8s} -> {n_frames} frames bundled to {csv_out} and {npz_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories_dir", default="handoff_alexandre/trajectories")
    parser.add_argument("--keypoints_dir", default="outputs/keypoints/json")
    parser.add_argument("--out_dir", default="handoff_alexandre")
    args = parser.parse_args()

    traj_files = sorted(glob.glob(os.path.join(args.trajectories_dir, "trajectory_*.csv")))
    if not traj_files:
        # Fallback to local search if paths differ
        traj_files = sorted(glob.glob("trajectory_*.csv"))

    if not traj_files:
        print(f"[-] No trajectory CSVs found in {args.trajectories_dir}")
        return

    print(f"Bundling {len(traj_files)} sequences for Alexandre's 3D & Spin module...\n")

    for traj_path in traj_files:
        base = os.path.splitext(os.path.basename(traj_path))[0]
        test_id = base.replace("trajectory_", "")

        # Look for corresponding keypoint json
        kp_json = os.path.join(args.keypoints_dir, f"{test_id}_keypoints.json")
        if not os.path.exists(kp_json):
            kp_json = os.path.join(args.keypoints_dir, f"keypoints_{test_id}.json")

        if not os.path.exists(kp_json):
            print(f"[!] Warning: Missing keypoint JSON for {test_id} (looked for {kp_json}). Skipping.")
            continue

        process_dataset(traj_path, kp_json, args.out_dir)

    print(f"\n[✓] All handoff files successfully packaged in: {args.out_dir}/")


if __name__ == "__main__":
    main()