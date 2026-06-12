import cv2
import numpy as np
import glob

# Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.02, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

all_corners = []
all_ids = []
image_size = None

# Process images
for image_path in glob.glob("calibration_images/*.png"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None: image_size = gray.shape[::-1]
    
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is not None:
        all_corners.append(corners)
        all_ids.append(ids)

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_corners, all_ids, board, image_size, None, None
)

# Save results
np.save("camera_matrix.npy", mtx)
np.save("dist_coeffs.npy", dist)
print("Calibration complete. Matrix and distortion coefficients saved.")
