"""
Script to rebuild the final 5-video comparison grid from existing saved images.

Usage:
    python src/rebuild_grid.py
"""
import glob
import os
import cv2
import numpy as np

def main():
    vis_dir = "outputs/keypoints/vis"
    tests = ["test2", "test3", "test5", "test6", "test7"]
    rendered = []

    for t in tests:
        path = os.path.join(vis_dir, f"{t}_keypoints.jpg")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        small = cv2.resize(img, (640, 360))
        cv2.putText(small, t, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        rendered.append(small)

    if rendered:
        n = len(rendered)
        cols = 2
        rows = (n + cols - 1) // cols
        grid = np.zeros((rows * 360, cols * 640, 3), dtype=np.uint8)
        for idx, img in enumerate(rendered):
            r, c = idx // cols, idx % cols
            grid[r * 360:(r + 1) * 360, c * 640:(c + 1) * 640] = img
        
        out_grid = os.path.join(vis_dir, "ALL_TESTS_GRID_COMPARISON.jpg")
        cv2.imwrite(out_grid, grid)
        print(f"[✓] Rebuilt final grid at: {out_grid}")

if __name__ == "__main__":
    main()