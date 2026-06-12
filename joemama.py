#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
import json
from adafruit_servokit import ServoKit
import sys
import tty
import termios

# ── CONFIGURATION ───────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
CALIBRATION_FILE = "calibration.json"

FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL  = 0
TILT_CHANNEL = 1

PAN_CENTER    = 135
PAN_ACTUATION = 270
PAN_PULSE_MIN = 552
PAN_PULSE_MAX = 2282
PAN_MIN       = 20
PAN_MAX       = 250

TILT_CENTER    =   0
TILT_ACTUATION = 180
TILT_PULSE_MIN = 1180
TILT_PULSE_MAX = 2525
TILT_MIN       = -15
TILT_MAX       =  80

# ── PD CONTROLLER TUNING ─────────────────────────────────────────────────────
KP_PAN  = 10.0
KD_PAN  = 0.15

KP_TILT = 7.0
KD_TILT = 0.10

MAX_SERVO_STEP    = 10.0
TRIGGER_THRESHOLD = 0.05    

# ── TIMEOUT CONFIGURATION ────────────────────────────────────────────────────
LOST_TIMEOUT = 2.0

# ── Shooter configuration ────────────────────────────────────────────────────
FLYWHEEL_CHANNEL   = 3
TRIGGER_CHANNEL    = 4

FLYWHEEL_PULSE_MIN = 1000
FLYWHEEL_PULSE_MAX = 2000

TRIGGER_PULSE_MIN  = 1000
TRIGGER_PULSE_MAX  = 1500

FLYWHEEL_SPINUP_DELAY = 3.0  # seconds after enabling before trigger is allowed

# ── Flywheel toggle state ─────────────────────────────────────────────────────
flywheel_enabled    = False   # Off by default; user must press F to enable
flywheel_enabled_at = 0.0     # monotonic timestamp of the last enable event
toggle_lock         = threading.Lock()


def keyboard_listener():
    """Background thread: press 'f' to toggle flywheels on/off."""
    global flywheel_enabled, flywheel_enabled_at
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('f', 'F'):
                with toggle_lock:
                    flywheel_enabled = not flywheel_enabled
                    if flywheel_enabled:
                        flywheel_enabled_at = time.monotonic()
                        print(f"\n[FLYWHEEL TOGGLE] Flywheels: ON  (trigger armed in {FLYWHEEL_SPINUP_DELAY:.0f}s)")
                    else:
                        flywheel_enabled_at = 0.0
                        print("\n[FLYWHEEL TOGGLE] Flywheels: OFF")
            elif ch == '\x03':  # Ctrl+C passthrough
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class VideoStream:
    def __init__(self, src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.started   = False
        self.read_lock = threading.Lock()
        self.frame     = None
        self.ret       = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread  = threading.Thread(target=self.update, args=(), daemon=True)
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
                        self.ret   = ret
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

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

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
    kit.servo[PAN_CHANNEL].actuation_range  = PAN_ACTUATION
    kit.servo[PAN_CHANNEL].set_pulse_width_range(PAN_PULSE_MIN, PAN_PULSE_MAX)
    kit.servo[TILT_CHANNEL].actuation_range = TILT_ACTUATION
    kit.servo[TILT_CHANNEL].set_pulse_width_range(TILT_PULSE_MIN, TILT_PULSE_MAX)

    kit.servo[PAN_CHANNEL].angle = PAN_CENTER
    set_tilt(kit, TILT_CENTER)

    kit.servo[FLYWHEEL_CHANNEL].set_pulse_width_range(FLYWHEEL_PULSE_MIN, FLYWHEEL_PULSE_MAX)
    kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
    kit.servo[TRIGGER_CHANNEL].set_pulse_width_range(TRIGGER_PULSE_MIN, TRIGGER_PULSE_MAX)
    kit.servo[TRIGGER_CHANNEL].fraction  = 0.0

    pan_servo  = float(PAN_CENTER)
    tilt_servo = float(TILT_CENTER)

    trigger_start_time  = 0.0
    trigger_active      = False

    last_error_x        = 0.0
    last_error_y        = 0.0
    last_time           = time.monotonic()
    last_detection_time = time.monotonic()

    print("\n=======================================================")
    print("Press 'F' to toggle flywheels ON/OFF.")
    print(f"Trigger is locked for {FLYWHEEL_SPINUP_DELAY:.0f}s after enabling flywheels.")
    print("Press Ctrl-C to safely stop.")
    print("=======================================================\n")

    try:
        while True:
            now = time.monotonic()
            dt  = now - last_time
            last_time = now
            if dt <= 0:
                dt = 0.001

            ret, frame = vs.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_list, ids, _ = tag_detector.detectMarkers(gray)

            # Read toggle state once per loop
            with toggle_lock:
                flywheels_on    = flywheel_enabled
                fw_enabled_at   = flywheel_enabled_at

            # Trigger is only allowed once flywheels have been on long enough
            spinup_elapsed  = (now - fw_enabled_at) if flywheels_on else 0.0
            trigger_armed   = flywheels_on and (spinup_elapsed >= FLYWHEEL_SPINUP_DELAY)
            spinup_remaining = max(0.0, FLYWHEEL_SPINUP_DELAY - spinup_elapsed)

            if ids is not None and len(ids) > 0:
                last_detection_time = now

                pts = corners_list[0].reshape(4, 2)
                cx  = pts[:, 0].mean()
                cy  = pts[:, 1].mean()

                error_x = (aim_x - cx) / actual_w
                error_y = (aim_y - cy) / actual_h

                deriv_x = (error_x - last_error_x) / dt
                deriv_y = (error_y - last_error_y) / dt

                pan_step  = (error_x * KP_PAN)  + (deriv_x * KD_PAN)
                tilt_step = (error_y * KP_TILT) + (deriv_y * KD_TILT)

                pan_step  = clamp(pan_step,  -MAX_SERVO_STEP, MAX_SERVO_STEP)
                tilt_step = clamp(tilt_step, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                pan_servo  = clamp(pan_servo  + pan_step,  PAN_MIN,  PAN_MAX)
                tilt_servo = clamp(tilt_servo + tilt_step, TILT_MIN, TILT_MAX)

                kit.servo[PAN_CHANNEL].angle = pan_servo
                set_tilt(kit, tilt_servo)

                last_error_x = error_x
                last_error_y = error_y

                if (abs(error_x) <= TRIGGER_THRESHOLD and abs(error_y) <= TRIGGER_THRESHOLD
                        and not trigger_active and trigger_armed):
                    kit.servo[TRIGGER_CHANNEL].fraction = 0.7
                    trigger_start_time = now
                    trigger_active     = True

                if not flywheels_on:
                    fw_status = "OFF"
                elif not trigger_armed:
                    fw_status = f"SPINNING UP ({spinup_remaining:.1f}s)"
                else:
                    fw_status = "ARMED"

                print(f"\r[LOCK ON] Pan: {error_x: .3f} | Tilt: {error_y: .3f} | FW: {fw_status}  ", end="")

            else:
                time_since_lost = now - last_detection_time

                if time_since_lost < LOST_TIMEOUT:
                    last_error_x = 0.0
                    last_error_y = 0.0

                    if not flywheels_on:
                        fw_status = "OFF"
                    elif not trigger_armed:
                        fw_status = f"SPINNING UP ({spinup_remaining:.1f}s)"
                    else:
                        fw_status = "ARMED"

                    print(f"\r[HOLD] Grace: {LOST_TIMEOUT - time_since_lost:.1f}s | FW: {fw_status}  ", end="")

                else:
                    last_error_x = 0.0
                    last_error_y = 0.0

                    pan_servo  += clamp((PAN_CENTER  - pan_servo)  * 0.05, -MAX_SERVO_STEP, MAX_SERVO_STEP)
                    tilt_servo += clamp((TILT_CENTER - tilt_servo) * 0.05, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                    kit.servo[PAN_CHANNEL].angle = pan_servo
                    set_tilt(kit, tilt_servo)

                    if not flywheels_on:
                        fw_status = "OFF"
                    elif not trigger_armed:
                        fw_status = f"SPINNING UP ({spinup_remaining:.1f}s)"
                    else:
                        fw_status = "ARMED"

                    print(f"\r[SEARCHING] Returning home | FW: {fw_status}              ", end="")

            # Trigger auto-reset
            if trigger_active and (now - trigger_start_time >= 0.5):
                kit.servo[TRIGGER_CHANNEL].fraction = 0.0
                trigger_active = False

            # Flywheels: user toggle is the only authority
            kit.servo[FLYWHEEL_CHANNEL].fraction = 0.5 if flywheels_on else 0.0

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        print("Shutting down...")
        vs.stop()
        kit.servo[PAN_CHANNEL].angle = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        kit.servo[TRIGGER_CHANNEL].fraction  = 0.0
        print("Done.")


if __name__ == "__main__":
    main()
