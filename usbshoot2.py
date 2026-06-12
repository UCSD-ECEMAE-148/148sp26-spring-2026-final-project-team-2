#!/usr/bin/env python3

import cv2
import numpy as np
import time
import math
import sys
import threading
import json
import os  
from collections import deque
from adafruit_servokit import ServoKit

# ── TRACKING CONFIGURATION ───────────────────────────────────────────────────
TROUBLESHOOT_MODE = "TRACKING" 

# ── Camera settings ───────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
CALIBRATION_FILE = "calibration.json"
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
DOWNSAMPLE_SCALE = 0.5  

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

# ── PID CONTROLLER TUNING ────────────────────────────────────────────────────
MAX_SERVO_STEP = 12.0  

KP_PAN  = 7.0   
KI_PAN  = 3.5   
KD_PAN  = 0.25  

KP_TILT = 4.5   
KI_TILT = 2.0   
KD_TILT = 0.01

INTEGRAL_MAX = 5.0  
DEADZONE_NORM = 0.003  
TRIGGER_THRESHOLD = 0.015  

# ── TIMEOUT CONFIGURATION ────────────────────────────────────────────────────
# STRICT CUTOFF: Drop tracking completely if no real detection in 1.0 second
RETURN_TO_CENTER_TIMEOUT = 1.0  

# ── FILTER CONFIGURATION ─────────────────────────────────────────────────────
D_FILTER_ALPHA = 0.40  

# ── Shooter configuration ────────────────────────────────────────────────────
FLYWHEEL_CHANNEL   = 3
TRIGGER_CHANNEL    = 4
FLYWHEEL_PULSE_MIN = 1000
FLYWHEEL_PULSE_MAX = 1000
TRIGGER_PULSE_MIN  = 1000
TRIGGER_PULSE_MAX  = 1400


class TagKalmanFilter:
    """Linear Kalman Filter to model target Position and Velocity in 2D space."""
    def __init__(self):
        # 4 State variables (x, y, dx, dy), 2 Measurement variables (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement Matrix (We only observe position x and y directly)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        
        # Transition Matrix (State update: pos = pos + vel * dt)
        self.kf.transitionMatrix = np.array([[1, 0, 0.033, 0],
                                             [0, 1, 0, 0.033],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        
        # Process Noise Covariance (Trust in physics model)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        
        # Measurement Noise Covariance (Trust in raw camera coordinates)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
        
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict(self, dt):
        """Project state vector forward using dynamic delta time."""
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        prediction = self.kf.predict()
        return float(prediction[0]), float(prediction[1])

    def update(self, raw_x, raw_y):
        """Correct estimated state using valid camera measurements."""
        measurement = np.array([[np.float32(raw_x)], [np.float32(raw_y)]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePost = np.array([[np.float32(raw_x)], [np.float32(raw_y)], [0], [0]], dtype=np.float32)
            self.initialized = True
            return raw_x, raw_y
        
        corrected = self.kf.correct(measurement)
        return float(corrected[0]), float(corrected[1])
        
    def reset(self):
        self.initialized = False


def configure_camera_hardware(index):
    """Configures camera driver directly via system V4L2 hooks to kill motion blur."""
    print(f"Applying hardware exposure configurations to /dev/video{index}...")
    try:
        os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_auto=1")
        os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_time_absolute=60")
        os.system(f"v4l2-ctl -d /dev/video{index} -c gain=35")
        print("Hardware exposure settings successfully locked.")
    except Exception as e:
        print(f"Warning: Failed to apply hardware camera configurations: {e}")


class VideoStream:
    """Threaded camera handler that continuously purges hardware buffers to prevent backlog lag."""
    def __init__(self, src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.started = False
        self.read_lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.frame_timestamp = 0.0

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            while True:
                grabbed = self.cap.grab()
                if not grabbed:
                    break
                arrival_time = time.perf_counter()
                ret, frame = self.cap.retrieve()
                if ret:
                    with self.read_lock:
                        self.ret = ret
                        self.frame = frame
                        self.frame_timestamp = arrival_time
            time.sleep(0.001)

    def read(self):
        with self.read_lock:
            if self.frame is None:
                return False, None, 0.0
            return self.ret, self.frame.copy(), self.frame_timestamp

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
    configure_camera_hardware(CAMERA_INDEX)
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
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.polygonalApproxAccuracyRate = 0.05

    tag_dict        = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_detector    = cv2.aruco.ArucoDetector(tag_dict, detector_params)

    # Initialize Kalman Filter
    tracker_kf = TagKalmanFilter()

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

    pan_servo  = float(PAN_CENTER)
    tilt_servo = float(TILT_CENTER)   
    
    last_error_x = 0.0
    last_error_y = 0.0
    
    integral_x = 0.0
    integral_y = 0.0
    
    filtered_dx = 0.0
    filtered_dy = 0.0
    
    last_time = time.monotonic()
    last_detection_time = time.monotonic()

    trigger_start_time = 0.0
    trigger_active = False

    fps_accum_start  = time.monotonic()
    fps_accum_count  = 0
    fps              = 0.0

    latency_history = deque(maxlen=5)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    print(f"\n=======================================================")
    print(f"RUNNING WITH REAL-TIME KALMAN TRAJECTORY PREDICTION")
    print(f"=======================================================")
    print("Press Ctrl-C to quit safely.\n")

    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            if dt <= 0:
                dt = 0.001

            ret, frame, frame_t = vs.read()
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
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=DOWNSAMPLE_SCALE, fy=DOWNSAMPLE_SCALE, interpolation=cv2.INTER_LINEAR)
            enhanced_gray = clahe.apply(small_gray)
            
            corners_list, ids, _ = tag_detector.detectMarkers(enhanced_gray)

            # ── KALMAN PREDICTION STEP ───────────────────────────────────────
            # Project where the tag should be located based on current velocity vectors
            pred_error_x, pred_error_y = tracker_kf.predict(dt)

            # Check if we have passed our strict 1-second physical limit
            time_since_last_detection = now - last_detection_time
            tracking_valid = time_since_last_detection <= RETURN_TO_CENTER_TIMEOUT

            if ids is not None and len(ids) > 0:
                # Real Tag Visible: Extract exact measurement data
                last_detection_time = now
                kit.servo[FLYWHEEL_CHANNEL].fraction = 1.0

                corners_list = [c / DOWNSAMPLE_SCALE for c in corners_list]
                pts = corners_list[0].reshape(4, 2)
                raw_cx = pts[:, 0].mean()
                raw_cy = pts[:, 1].mean()

                raw_error_x = (aim_x - raw_cx) / actual_w
                raw_error_y = (aim_y - raw_cy) / actual_h  

                # Update the Kalman physics engine using raw camera measurements
                error_x, error_y = tracker_kf.update(raw_error_x, raw_error_y)
                status_string = "TRACKING"

            elif tracking_valid:
                # Spotty Drop / Occlusion Pocket: Feed forward via prediction mechanics
                error_x = pred_error_x
                error_y = pred_error_y
                status_string = "PREDICTING"
            else:
                # Absolute Cutoff Triggered (> 1.0 second limit exceeded)
                status_string = "LOST"

            if status_string != "LOST":
                # Execute standard tracking and servo driving pathways
                if abs(error_x) < DEADZONE_NORM: error_x = 0.0
                if abs(error_y) < DEADZONE_NORM: error_y = 0.0

                # ── TRIGGER EVALUATION ────────────────────────────────────────────
                # Only allow firing capabilities if the tag is actually visible
                if ids is not None and len(ids) > 0:
                    if abs(error_x) <= TRIGGER_THRESHOLD and abs(error_y) <= TRIGGER_THRESHOLD and not trigger_active:
                        kit.servo[TRIGGER_CHANNEL].fraction = 1.0
                        trigger_start_time = now
                        trigger_active = True

                # ── PAN PID LOGIC ─────────────────────────────────────────────────
                raw_dx = (error_x - last_error_x) / dt
                filtered_dx = (D_FILTER_ALPHA * raw_dx) + ((1.0 - D_FILTER_ALPHA) * filtered_dx)
                integral_x = clamp(integral_x + (error_x * dt), -INTEGRAL_MAX, INTEGRAL_MAX)
                
                pan_output = (error_x * KP_PAN) + (integral_x * KI_PAN) + (filtered_dx * KD_PAN)
                pan_output = clamp(pan_output, -MAX_SERVO_STEP, MAX_SERVO_STEP) 
                
                pan_servo = clamp(pan_servo + pan_output, PAN_MIN, PAN_MAX)
                kit.servo[PAN_CHANNEL].angle = pan_servo
                last_error_x = error_x

                # ── TILT PID LOGIC ────────────────────────────────────────────────
                raw_dy = (error_y - last_error_y) / dt
                filtered_dy = (D_FILTER_ALPHA * raw_dy) + ((1.0 - D_FILTER_ALPHA) * filtered_dy)
                integral_y = clamp(integral_y + (error_y * dt), -INTEGRAL_MAX, INTEGRAL_MAX)
                
                tilt_output = (error_y * KP_TILT) + (integral_y * KI_TILT) + (filtered_dy * KD_TILT)
                tilt_output = clamp(tilt_output, -MAX_SERVO_STEP, MAX_SERVO_STEP) 
                
                tilt_servo = clamp(tilt_servo + tilt_output, TILT_MIN, TILT_MAX)
                set_tilt(kit, tilt_servo)
                last_error_y = error_y

                raw_latency_ms = (time.perf_counter() - frame_t) * 1000.0
                latency_history.append(raw_latency_ms)
                avg_latency_ms = sum(latency_history) / len(latency_history)

                print(f"\r[{status_string}] Latency: {avg_latency_ms:5.1f}ms | Pan Err: {error_x: .3f} | Trig: {trigger_active}  ", end="")
            
            else:
                # Target completely dropped: Clear Kalman memory and clean PID variables
                tracker_kf.reset()
                last_error_x = 0.0
                last_error_y = 0.0
                integral_x = 0.0
                integral_y = 0.0
                filtered_dx = 0.0
                filtered_dy = 0.0

                kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
                pan_error_to_center = PAN_CENTER - pan_servo
                tilt_error_to_center = TILT_CENTER - tilt_servo

                pan_servo += clamp(pan_error_to_center * 0.1, -MAX_SERVO_STEP, MAX_SERVO_STEP)
                tilt_servo += clamp(tilt_error_to_center * 0.1, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                kit.servo[PAN_CHANNEL].angle = pan_servo
                set_tilt(kit, tilt_servo)
                print(f"\r[FPS {fps:4.1f}] Timeout reached ({time_since_last_detection:.1f}s). Returning to center...  ", end="")

            if trigger_active and (now - trigger_start_time >= 0.5):
                kit.servo[TRIGGER_CHANNEL].fraction = 0.0
                trigger_active = False

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nInterrupted tracking.")

    finally:
        print("Resetting hardware configuration...")
        vs.stop()
        kit.servo[PAN_CHANNEL].angle = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        kit.servo[TRIGGER_CHANNEL].fraction = 0.0
        print("Done.")


if __name__ == "__main__":
    main()