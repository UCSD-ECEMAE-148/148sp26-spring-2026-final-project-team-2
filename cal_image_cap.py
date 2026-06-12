import cv2
import os

cap = cv2.VideoCapture(0)
output_dir = "calibration_images"
os.makedirs(output_dir, exist_ok=True)
count = 0

print("Press 's' to save a frame, 'q' to quit.")
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"{output_dir}/calib_{count}.png", frame)
        print(f"Saved {count}")
        count += 1
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
