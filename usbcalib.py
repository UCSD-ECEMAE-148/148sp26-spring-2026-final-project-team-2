"""
Camera Calibration from Image Directory
-----------------------------------------
Usage:
  python calibrate_camera.py                          # uses ./calibration_images/
  python calibrate_camera.py --input my_imgs          # custom folder
  python calibrate_camera.py --output calibration.json
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import glob
import os
import json

SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")

SQUARE_SIZE = 0.035   # metres — update if your print differs
MARKER_SIZE = 0.026

def load_image_paths(directory):
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera from a directory of ChArUco images.")
    parser.add_argument("--input",  default="calibration_images", help="Folder with calibration images")
    parser.add_argument("--output", default="calibration.json",   help="Where to save calibration results")
    args = parser.parse_args()

    image_paths = load_image_paths(args.input)
    if not image_paths:
        print(f"No images found in '{args.input}'. Supported: {SUPPORTED_EXTS}")
        return

    print(f"Found {len(image_paths)} images in '{args.input}'")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board      = aruco.CharucoBoard((7, 5), SQUARE_SIZE, MARKER_SIZE, dictionary)
    detector   = aruco.CharucoDetector(board)

    all_corners, all_ids = [], []
    img_size = None
    skipped  = []

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{i+1}/{len(image_paths)}] SKIP (could not read): {img_path}")
            skipped.append(img_path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) >= 4:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print(f"  [{i+1}/{len(image_paths)}] OK  — {len(charuco_ids)} corners: {os.path.basename(img_path)}")
        else:
            count = len(charuco_ids) if charuco_ids is not None else 0
            print(f"  [{i+1}/{len(image_paths)}] SKIP (only {count} corners): {os.path.basename(img_path)}")
            skipped.append(img_path)

    print(f"\n{len(all_corners)} usable images out of {len(image_paths)}")

    if len(all_corners) < 5:
        print("Not enough usable images to calibrate (need at least 5). Capture more images.")
        return

    print("Running calibration...")
    # calibrateCameraCharuco moved in OpenCV 4.8+ — try both APIs
    try:
        ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_size, None, None
        )
    except AttributeError:
        # OpenCV 4.8+: use calibrateCamera with board.chessboardCorners
        obj_points = []
        img_points = []
        for corners, ids in zip(all_corners, all_ids):
            obj_pts = board.getChessboardCorners()[ids.ravel()]
            obj_points.append(obj_pts.astype(np.float32))
            img_points.append(corners)
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )

    print(f"\n{'='*45}")
    print(f"RMS reprojection error : {ret:.4f} px")
    print(f"  (< 1.0 acceptable, < 0.5 excellent)")
    print(f"\nCamera matrix K:\n{K}")
    print(f"\nDistortion coefficients:\n{dist.ravel()}")
    print(f"{'='*45}\n")

    # Save results to JSON
    results = {
        "rms_error":              ret,
        "image_size":             list(img_size),
        "square_size_m":          SQUARE_SIZE,
        "marker_size_m":          MARKER_SIZE,
        "camera_matrix":          K.tolist(),
        "dist_coeffs":            dist.tolist(),
        "usable_images":          len(all_corners),
        "total_images":           len(image_paths),
        "skipped_images":         skipped,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Calibration saved to: {args.output}")

if __name__ == "__main__":
    main()