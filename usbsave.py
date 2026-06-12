"""
Camera Calibration Image Capture
---------------------------------
Controls:
  SPACE  - save current frame
  R      - toggle live ArUco detection overlay
  Q/ESC  - quit

Usage:
  python capture_calibration_images.py
  python capture_calibration_images.py --camera 1        # use camera index 1
  python capture_calibration_images.py --output my_imgs  # custom output folder
"""

import cv2
import cv2.aruco as aruco
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Capture calibration images from camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", type=str, default="calibration_images", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Try to set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")
    print(f"Saving images to: ./{args.output}/")
    print("SPACE = save frame | R = toggle detection overlay | Q/ESC = quit\n")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard((7, 5), 0.035, 0.026, dictionary)
    detector = aruco.CharucoDetector(board)

    show_overlay = True
    saved_count = 0
    flash_until = 0  # timestamp for green flash feedback

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        display = frame.copy()

        # --- ArUco detection overlay ---
        markers_found = 0
        charuco_corners_found = 0
        if show_overlay:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            if marker_ids is not None:
                markers_found = len(marker_ids)
                aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if charuco_ids is not None:
                charuco_corners_found = len(charuco_ids)
                aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

        # --- Flash feedback on save ---
        if time.time() < flash_until:
            cv2.rectangle(display, (0, 0), (actual_w, actual_h), (0, 255, 0), 12)

        # --- HUD overlay ---
        def put(text, y, color=(255, 255, 255)):
            cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)

        put(f"Saved: {saved_count} images", 30)
        put(f"Overlay: {'ON' if show_overlay else 'OFF'}  [R to toggle]", 58)
        if show_overlay:
            quality_color = (0, 200, 0) if charuco_corners_found >= 10 else (0, 140, 255)
            put(f"Markers: {markers_found}  ChArUco corners: {charuco_corners_found}", 86, quality_color)
            if charuco_corners_found < 6:
                put("Move board into view", 114, (0, 80, 255))
        put("SPACE=save  R=overlay  Q=quit", actual_h - 14)

        cv2.imshow("Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            filename = os.path.join(args.output, f"calib_{saved_count:03d}.png")
            cv2.imwrite(filename, frame)  # save original, not the annotated display
            saved_count += 1
            flash_until = time.time() + 0.4
            print(f"Saved {filename}  (total: {saved_count})")

        elif key == ord('r'):
            show_overlay = not show_overlay

        elif key in (ord('q'), 27):  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved_count} images saved to ./{args.output}/")
    if saved_count < 20:
        print(f"Tip: aim for 20-30 images for a good calibration (you have {saved_count}).")

if __name__ == "__main__":
    main()