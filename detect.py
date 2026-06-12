import cv2
import numpy as np
import time

# Camera calibration constants
fx, fy = 600, 600
cx, cy = 320, 240
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)
tag_size = 0.05  # meters

# AprilTag dictionary and detector setup
aruco = cv2.aruco
tag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
params = aruco.DetectorParameters_create()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("AprilTag detection with X11 feed started. Press 'q' to stop.")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags
        corners, ids, rejected = aruco.detectMarkers(gray, tag_dict, parameters=params)

        if ids is not None:
            # Draw outlines around detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, tag_id in enumerate(ids.flatten()):
                c = corners[i].reshape((4, 2))
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([c], tag_size, camera_matrix, dist_coeffs)
                
                # Draw 3D axis on the tag to visualize orientation
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.03)
                
                tvec = tvecs[0].ravel()
                print(f"ID:{tag_id} tvec:{tvec.round(3)}")

        # Display the live feed via X11
        # cv2.imshow("AprilTag Live Feed", frame)

        # FPS calculation
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[INFO] Approx FPS: {fps:.1f}")

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Detection stopped by user.")
finally:
    # Proper cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
