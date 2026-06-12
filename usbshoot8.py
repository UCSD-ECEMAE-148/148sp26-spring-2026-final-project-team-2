#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
import json
from adafruit_servokit import ServoKit

# ── CONFIGURATION ───────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
CALIBRATION_FILE = "calibration.json"

# RESTORED ORIGINAL HIGH-RES FOOTPRINT
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720

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

# ── PD CONTROLLER TUNING ─────────────────────────────────────────────────────
KP_PAN  = 10.0   
KD_PAN  = 0.15   

KP_TILT = 7.0   
KD_TILT = 0.10   

MAX_SERVO_STEP = 10.0  
TRIGGER_THRESHOLD = 0.02

# ── TIMEOUT CONFIGURATION ────────────────────────────────────────────────────
LOST_TIMEOUT = 2.0  # Holds position AND keeps flywheels hot for exactly 2.0s

# ── Shooter configuration ────────────────────────────────────────────────────
FLYWHEEL_CHANNEL   = 3
TRIGGER_CHANNEL    = 4

FLYWHEEL_PULSE_MIN = 1000
FLYWHEEL_PULSE_MAX = 2000

TRIGGER_PULSE_MIN  = 1000
TRIGGER_PULSE_MAX  = 2000


class VideoStream:
    def __init__(self, src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2) 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.started = False
        self.read_lock = threading.Lock()
        self.frame = None
        self.ret = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            while self.started:
                grabbed = self.cap.grab()
                if not grabbed:
                    break
                ret, frame = self.cap.retrieve()
                if ret:
                    with self.read_lock:
                        self.ret = ret
                        self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.read_lock:
            return self.ret, self.frame

    def stop(self):
        self.started = False
        if self.cap.isOpened():
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
    return K, dist


def main():
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

    vs = VideoStream(src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
    print("Waiting for camera initialization...")
    time.sleep(2.0) 

    ret, test_frame = vs.read()
    if ret and test_frame is not None:
        actual_h, actual_w = test_frame.shape[:2]
    else:
        actual_w, actual_h = FRAME_WIDTH, FRAME_HEIGHT

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (actual_w, actual_h), alpha=0
    )
    aim_x = new_K[0, 2]   
    aim_y = new_K[1, 2]

    tag_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_detector = cv2.aruco.ArucoDetector(tag_dict, cv2.aruco.DetectorParameters())

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
    
    trigger_start_time = 0.0
    trigger_active = False
    
    last_error_x = 0.0
    last_error_y = 0.0
    last_time = time.monotonic()
    last_detection_time = time.monotonic()

    print("\n=======================================================")
    print("RUNNING RE-TUNED ENGINE — EXPLICIT GRACE PERIOD LOCK")
    print("=======================================================")
    print("Press Ctrl-C to safely stop.\n")

    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now
            if dt <= 0: dt = 0.001

            ret, frame = vs.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_list, ids, _ = tag_detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                last_detection_time = now
                
                # Active lock found: target full speed
                flywheel_target = 0.5

                pts = corners_list[0].reshape(4, 2)
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()

                error_x = (aim_x - cx) / actual_w
                error_y = (aim_y - cy) / actual_h  

                deriv_x = (error_x - last_error_x) / dt
                deriv_y = (error_y - last_error_y) / dt

                pan_step  = (error_x * KP_PAN) + (deriv_x * KD_PAN)
                tilt_step = (error_y * KP_TILT) + (deriv_y * KD_TILT)

                pan_step  = clamp(pan_step, -MAX_SERVO_STEP, MAX_SERVO_STEP)
                tilt_step = clamp(tilt_step, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                pan_servo  = clamp(pan_servo + pan_step, PAN_MIN, PAN_MAX)
                tilt_servo = clamp(tilt_servo + tilt_step, TILT_MIN, TILT_MAX)

                kit.servo[PAN_CHANNEL].angle = pan_servo
                set_tilt(kit, tilt_servo)

                last_error_x = error_x
                last_error_y = error_y

                if abs(error_x) <= TRIGGER_THRESHOLD and abs(error_y) <= TRIGGER_THRESHOLD and not trigger_active:
                    kit.servo[TRIGGER_CHANNEL].fraction = 0.7
                    trigger_start_time = now
                    trigger_active = True

                print(f"\r[LOCK ON] Pan Err: {error_x: .3f} | Tilt Err: {error_y: .3f} | Flywheel: HOT (0.5)  ", end="")
            
            else:
                time_since_lost = now - last_detection_time

                # If we are within the 2-second grace window
                if time_since_lost < LOST_TIMEOUT:
                    # FORCE flywheels to stay active during this state
                    flywheel_target = 0.5
                    
                    last_error_x = 0.0
                    last_error_y = 0.0
                    print(f"\r[HOLD POSITION] Lost tag. Grace Period Active: Holding flywheels HOT for {LOST_TIMEOUT - time_since_lost:.1f}s...  ", end="")
                else:
                    # Grace window has fully expired. Turn off flywheels and head home.
                    flywheel_target = 0.0
                    
                    last_error_x = 0.0
                    last_error_y = 0.0
                    
                    pan_servo  += clamp((PAN_CENTER - pan_servo) * 0.05, -MAX_SERVO_STEP, MAX_SERVO_STEP)
                    tilt_servo += clamp((TILT_CENTER - tilt_servo) * 0.05, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                    kit.servo[PAN_CHANNEL].angle = pan_servo
                    set_tilt(kit, tilt_servo)
                    print(f"\r[SEARCHING] Returning to center home. Flywheels: OFF              ", end="")

            # Manage automatic trigger reset windows (Back to 0.0 rest state)
            if trigger_active and (now - trigger_start_time >= 0.5):
                kit.servo[TRIGGER_CHANNEL].fraction = 0.0
                trigger_active = False

            # CRITICAL FIX: Explicitly enforce the flywheel speed at the absolute end of the loop execution
            # This ensures no intermediate conditional blocks override the hardware fraction duty cycles
            kit.servo[FLYWHEEL_CHANNEL].fraction = flywheel_target

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nScript manually stopped via terminal interface.")

    finally:
        print("Executing graceful safe shutdown sequences...")
        vs.stop()
        kit.servo[PAN_CHANNEL].angle = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        kit.servo[TRIGGER_CHANNEL].fraction = 0.0
        print("Done.")


if __name__ == "__main__":
    main()