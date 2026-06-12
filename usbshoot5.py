#!/usr/bin/env python3

import cv2
import numpy as np
import time
import math
import sys
import threading
import json
from adafruit_servokit import ServoKit

# ── TRACKING CONFIGURATION ───────────────────────────────────────────────────
TROUBLESHOOT_MODE = "TRACKING" 

# ── Camera settings ───────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
CALIBRATION_FILE = "calibration.json"
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
DOWNSAMPLE_SCALE = 0.5  # Drops processing resolution to 640x360 to ensure zero lag

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL  = 0
TILT_CHANNEL = 1

PAN_CENTER    = 135
PAN_ACTUATION = 270
PAN_PULSE_MIN = 552
PAN_PULSE_MAX = 2282
PAN_MIN          = 20
PAN_MAX          = 250

TILT_CENTER      =   0
TILT_ACTUATION   = 180
TILT_PULSE_MIN   = 1180
TILT_PULSE_MAX   = 2525
TILT_MIN         = -15
TILT_MAX         =  80

# ── PID CONTROLLER TUNING (YOUR VALUES + AGGRESSION BOOST) ───────────────────
# Max speed control: Allowed degrees change per frame (Doubled from 4.0)
MAX_SERVO_STEP = 8.0  

KP_PAN  = 4.0
KD_PAN  = 0.5

KP_TILT = 2.5
KD_TILT = 0.02

# Deadzone in normalized units (0.01 = 1% off-center)
DEADZONE_NORM = 0.012 

# ── TIMEOUT CONFIGURATION ────────────────────────────────────────────────────
RETURN_TO_CENTER_TIMEOUT = 2.0  # Seconds to wait before returning to center

# ── FILTER AGGRESSION CONFIGURATION ──────────────────────────────────────────
# 0.0 = completely frozen/heavy filter, 1.0 = raw/instant reaction.
D_FILTER_ALPHA = 0.30  # Increased from 0.15 to let velocity updates pass faster

# ── Shooter configuration ────────────────────────────────────────────────────
FLYWHEEL_CHANNEL   = 3
TRIGGER_CHANNEL    = 4
FLYWHEEL_PULSE_MIN = 1000
FLYWHEEL_PULSE_MAX = 2000
TRIGGER_PULSE_MIN  = 1000
TRIGGER_PULSE_MAX  = 2000


class VideoStream:
    """Threaded camera stream handler to completely bypass OpenCV's internal frame buffer."""
    def __init__(self, src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.001)

    def read(self):
        with self.read_lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def true_tilt_to_servo_degrees(true_deg):
    return clamp(90.0 - true_deg, 0.0, 180.0)


def set_tilt(kit, true_deg):
    kit.servo[TILT_CHANNEL].angle = true_tilt_to_servo_degrees(true_deg)


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    K    = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["dist_coeffs"],   dtype=np.float64)
    print(f"Loaded calibration from '{path}'")
    return K, dist


def main():
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

    vs = VideoStream(src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
    time.sleep(1.0) 

    actual_w = FRAME_WIDTH
    actual_h = FRAME_HEIGHT
    print(f"USB Threaded Camera Opened: {actual_w}x{actual_h}")

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (actual_w, actual_h), alpha=0
    )

    aim_x = new_K[0, 2]   
    aim_y = new_K[1, 2]

    detector_params = cv2.aruco.DetectorParameters()
    tag_dict        = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_detector    = cv2.aruco.ArucoDetector(tag_dict, detector_params)

    kit = ServoKit(channels=16)
    kit.servo[PAN_CHANNEL].actuation_range = PAN_ACTUATION
    kit.servo[PAN_CHANNEL].set_pulse_width_range(PAN_PULSE_MIN, PAN_PULSE_MAX)
    kit.servo[TILT_CHANNEL].actuation_range = TILT_ACTUATION
    kit.servo[TILT_CHANNEL].set_pulse_width_range(TILT_PULSE_MIN, TILT_PULSE_MAX)

    kit.servo[PAN_CHANNEL].angle = PAN_CENTER
    set_tilt(kit, TILT_CENTER)   

    kit.servo[FLYWHEEL_CHANNEL].set_pulse_width_range(FLYWHEEL_PULSE_MIN, FLYWHEEL_PULSE_MAX)
    kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
    kit.servo[TRIGGER_CHANNEL].set_pulse_width_range(TRIGGER_PULSE_MIN, TRIGGER_PULSE_MAX)
    kit.servo[TRIGGER_CHANNEL].fraction = 0.0

    # Tracking State variables
    pan_servo  = float(PAN_CENTER)
    tilt_servo = float(TILT_CENTER)   
    
    last_error_x = 0.0
    last_error_y = 0.0
    
    filtered_dx = 0.0
    filtered_dy = 0.0
    
    last_time = time.monotonic()
    last_detection_time = time.monotonic()

    fps_accum_start  = time.monotonic()
    fps_accum_count  = 0
    fps              = 0.0

    print(f"\n=======================================================")
    print(f"RUNNING IN MODE: {TROUBLESHOOT_MODE}")
    print(f"=======================================================")
    print("Press Ctrl-C to quit safely.\n")

    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if dt <= 0:
                dt = 0.001

            ret, frame = vs.read()
            if not ret or frame is None:
                continue

            fps_accum_count += 1
            elapsed = now - fps_accum_start
            if elapsed >= 1.0:
                fps = fps_accum_count / elapsed
                fps_accum_count = 0
                fps_accum_start = now

            if TROUBLESHOOT_MODE == "SERVO_CENTER":
                kit.servo[PAN_CHANNEL].angle = PAN_CENTER
                set_tilt(kit, TILT_CENTER)
                print(f"\r[CENTER MODE] Pan configured to {PAN_CENTER}°, Tilt configured to {TILT_CENTER}°", end="")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=DOWNSAMPLE_SCALE, fy=DOWNSAMPLE_SCALE, interpolation=cv2.INTER_LINEAR)
            corners_list, ids, _ = tag_detector.detectMarkers(small_gray)

            if ids is not None and len(ids) > 0:
                # Update time tracker because we have a valid detection
                last_detection_time = now

                corners_list = [c / DOWNSAMPLE_SCALE for c in corners_list]
                
                pts = corners_list[0].reshape(4, 2)
                raw_cx = pts[:, 0].mean()
                raw_cy = pts[:, 1].mean()

                # Calculate normalized error (-1.0 to 1.0)
                error_x = (aim_x - raw_cx) / actual_w
                error_y = (aim_y - raw_cy) / actual_h  

                # Apply normalized deadzone
                if abs(error_x) < DEADZONE_NORM: error_x = 0.0
                if abs(error_y) < DEADZONE_NORM: error_y = 0.0

                # ── PAN PID LOGIC ─────────────────────────────────────────────────
                raw_dx = (error_x - last_error_x) / dt
                filtered_dx = (D_FILTER_ALPHA * raw_dx) + ((1.0 - D_FILTER_ALPHA) * filtered_dx)
                
                pan_output = (error_x * KP_PAN) + (filtered_dx * KD_PAN)
                pan_output = clamp(pan_output, -MAX_SERVO_STEP, MAX_SERVO_STEP) 
                
                pan_servo = clamp(pan_servo + pan_output, PAN_MIN, PAN_MAX)
                kit.servo[PAN_CHANNEL].angle = pan_servo
                last_error_x = error_x

                # ── TILT PID LOGIC ────────────────────────────────────────────────
                raw_dy = (error_y - last_error_y) / dt
                filtered_dy = (D_FILTER_ALPHA * raw_dy) + ((1.0 - D_FILTER_ALPHA) * filtered_dy)
                
                tilt_output = (error_y * KP_TILT) + (filtered_dy * KD_TILT)
                tilt_output = clamp(tilt_output, -MAX_SERVO_STEP, MAX_SERVO_STEP) 
                
                tilt_servo = clamp(tilt_servo + tilt_output, TILT_MIN, TILT_MAX)
                set_tilt(kit, tilt_servo)
                last_error_y = error_y

                print(f"\r[TRACKING] FPS: {fps:4.1f} | Pan Err: {error_x: .3f} | Tilt Err: {error_y: .3f}", end="")
            
            else:
                # Target lost: reset PID historical buffers
                last_error_x = 0.0
                last_error_y = 0.0
                filtered_dx = 0.0
                filtered_dy = 0.0

                # Check if timeout has expired
                if now - last_detection_time > RETURN_TO_CENTER_TIMEOUT:
                    # Linearly interpolate or step servos smoothly back to center using MAX_SERVO_STEP limits
                    pan_error_to_center = PAN_CENTER - pan_servo
                    tilt_error_to_center = TILT_CENTER - tilt_servo

                    # Use a basic proportional snap back constrained by our frame limits
                    pan_servo += clamp(pan_error_to_center * 0.1, -MAX_SERVO_STEP, MAX_SERVO_STEP)
                    tilt_servo += clamp(tilt_error_to_center * 0.1, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                    kit.servo[PAN_CHANNEL].angle = pan_servo
                    set_tilt(kit, tilt_servo)
                    print(f"\r[FPS {fps:4.1f}] Lost target > {RETURN_TO_CENTER_TIMEOUT}s. Returning to center...  ", end="")
                else:
                    # Target lost but within grace period; hold position
                    print(f"\r[FPS {fps:4.1f}] Scanning... No AprilTag visible (Holding position).           ", end="")

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nInterrupted tracking.")

    finally:
        print("Resetting hardware configuration...")
        vs.stop()
        kit.servo[PAN_CHANNEL].angle = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        print("Done.")


if __name__ == "__main__":
    main()