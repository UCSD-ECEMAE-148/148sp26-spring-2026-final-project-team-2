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

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Detection runs on a downscaled copy for speed; aim math uses full-res coords
DETECT_SCALE = 0.5  # 640x360 for detection

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

# ── PID CONTROLLER TUNING ────────────────────────────────────────────────────
#
# Gains operate on angular error in degrees and produce servo POSITION DELTAS
# in degrees per loop tick (not per second). At 200Hz loop rate a KP of 0.08
# means: 1° of aim error → 0.08° of servo movement per tick, which is safe.
#
# Rule of thumb starting point:
#   KP  ~  MAX_SERVO_STEP / (max expected angle error in degrees)
#         e.g. 2.0 / 25° ≈ 0.08
#   KD  ~  KP * 0.05 to 0.15  (start low, increase if sluggish)
#   KI  ~  KP * 0.01 to 0.03  (start very low)

KP_PAN  = 0.08
KD_PAN  = 0.006

KP_TILT = 0.08
KD_TILT = 0.006

KI_PAN  = 0.001
KI_TILT = 0.001
INTEGRAL_CLAMP = 8.0   # degrees·ticks — prevents windup at limits

MAX_SERVO_STEP = 2.0   # hard cap on movement per loop tick (degrees)
                       # keeps motion smooth; raise if tracking feels too slow

# ── DERIVATIVE LOW-PASS FILTER ────────────────────────────────────────────────
# Blends raw derivative with previous filtered value.
# Lower alpha = more smoothing but slower derivative response.
DERIV_ALPHA = 0.25

# ── TRIGGER / DWELL CONFIGURATION ────────────────────────────────────────────
TRIGGER_THRESHOLD   = 1.5   # degrees — radial aim error must be within this cone
DWELL_REQUIRED_SECS = 0.10  # must hold inside cone for 100 ms before firing

# ── TIMEOUT / GRACE PERIOD ───────────────────────────────────────────────────
LOST_TIMEOUT = 2.0   # seconds to keep predicting after tag disappears

# ── Shooter configuration ────────────────────────────────────────────────────
FLYWHEEL_CHANNEL   = 3
TRIGGER_CHANNEL    = 4

FLYWHEEL_PULSE_MIN = 1000
FLYWHEEL_PULSE_MAX = 2000

TRIGGER_PULSE_MIN  = 1000
TRIGGER_PULSE_MAX  = 2000

# ── Flywheel toggle state ─────────────────────────────────────────────────────
flywheel_enabled = False
toggle_lock      = threading.Lock()


# ════════════════════════════════════════════════════════════════════════════
# KEYBOARD LISTENER
# ════════════════════════════════════════════════════════════════════════════

def keyboard_listener():
    """Background thread: press F to toggle flywheels on/off."""
    global flywheel_enabled
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('f', 'F'):
                with toggle_lock:
                    flywheel_enabled = not flywheel_enabled
                    state = "ON" if flywheel_enabled else "OFF"
                print(f"\n[FLYWHEEL TOGGLE] Flywheels: {state}")
            elif ch == '\x03':  # Ctrl+C passthrough
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ════════════════════════════════════════════════════════════════════════════
# THREADED CAMERA
# ════════════════════════════════════════════════════════════════════════════

class VideoStream:
    def __init__(self, src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
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
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.001)
                continue
            ret, frame = self.cap.retrieve()
            if ret:
                with self.read_lock:
                    self.ret   = ret
                    self.frame = frame

    def read(self):
        with self.read_lock:
            return self.ret, self.frame

    def stop(self):
        self.started = False
        if self.cap.isOpened():
            self.cap.release()


# ════════════════════════════════════════════════════════════════════════════
# THREADED DETECTOR
# Runs AprilTag detection on its own thread so it never blocks servo writes.
# The main loop calls detector.get() which returns the latest result instantly.
# ════════════════════════════════════════════════════════════════════════════

class TagDetector:
    def __init__(self, vs: VideoStream, scale: float = DETECT_SCALE):
        self.vs        = vs
        self.scale     = scale
        self.started   = False
        self.lock      = threading.Lock()
        self._cx       = None
        self._cy       = None
        self._detected = False
        self._ts       = 0.0

        tag_dict      = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(tag_dict, cv2.aruco.DetectorParameters())

    def start(self):
        self.started = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while self.started:
            ret, frame = self.vs.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            corners_list, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                pts = corners_list[0].reshape(4, 2)
                # Scale corner coordinates back to full-resolution space
                cx = float(pts[:, 0].mean()) / self.scale
                cy = float(pts[:, 1].mean()) / self.scale
                with self.lock:
                    self._cx       = cx
                    self._cy       = cy
                    self._detected = True
                    self._ts       = time.monotonic()
            else:
                with self.lock:
                    self._detected = False
                    self._ts       = time.monotonic()

    def get(self):
        """Returns (cx, cy, detected, timestamp). Non-blocking."""
        with self.lock:
            return self._cx, self._cy, self._detected, self._ts

    def stop(self):
        self.started = False


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

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

def pixel_to_angle_error(cx, cy, aim_x, aim_y, fx, fy):
    """
    Convert pixel offset from aim point into angular error in degrees using
    the calibrated focal length. This makes the error metric physically
    consistent across the full frame (not just at center).

    Sign convention:
      pan_err  > 0  → target is to the RIGHT  → servo must move right
      tilt_err > 0  → target is ABOVE center  → servo must move up
    """
    dx = cx - aim_x    # positive = target is right of aim point
    dy = cy - aim_y    # positive = target is below aim point (image Y flipped)

    pan_err_deg  =  np.degrees(np.arctan2(dx, fx))
    tilt_err_deg = -np.degrees(np.arctan2(dy, fy))  # negate: image Y is inverted
    return pan_err_deg, tilt_err_deg


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    vs = VideoStream(src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
    print("Waiting for camera initialization...")
    time.sleep(2.0)

    ret, test_frame = vs.read()
    actual_h, actual_w = test_frame.shape[:2] if (ret and test_frame is not None) \
                         else (FRAME_HEIGHT, FRAME_WIDTH)

    new_K, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (actual_w, actual_h), alpha=0
    )
    aim_x = new_K[0, 2]   # optical center x (pixels)
    aim_y = new_K[1, 2]   # optical center y (pixels)
    fx    = new_K[0, 0]   # focal length x (pixels)
    fy    = new_K[1, 1]   # focal length y (pixels)

    detector = TagDetector(vs).start()

    # ── Servo init ──────────────────────────────────────────────────────────
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

    # ── Controller state ────────────────────────────────────────────────────
    pan_servo  = float(PAN_CENTER)
    tilt_servo = float(TILT_CENTER)

    integral_pan  = 0.0
    integral_tilt = 0.0

    filtered_deriv_pan  = 0.0
    filtered_deriv_tilt = 0.0

    # Seeded to None so we can detect the very first frame and skip the
    # derivative calculation (avoids the giant spike on first acquisition).
    last_error_pan  = None
    last_error_tilt = None

    last_time = time.monotonic()

    # Predictive grace-period state
    vel_pan_dps         = 0.0
    vel_tilt_dps        = 0.0
    last_known_pan_err  = 0.0
    last_known_tilt_err = 0.0
    last_detection_time = time.monotonic()
    last_seen_time      = time.monotonic()

    # Trigger / dwell state
    trigger_start_time = 0.0
    trigger_active     = False
    dwell_start_time   = 0.0
    dwell_active       = False

    print("\n=======================================================")
    print("TURRET — Angular PID + Filtered Deriv + Dwell + Predict")
    print("Press 'F' to toggle flywheels ON/OFF.")
    print("Press Ctrl-C to safely stop.")
    print("=======================================================\n")

    try:
        while True:
            now = time.monotonic()
            dt  = now - last_time
            last_time = now
            # Clamp dt: ignore very long gaps (e.g. startup stall) that would
            # produce enormous derivative spikes.
            dt = clamp(dt, 0.001, 0.05)

            with toggle_lock:
                flywheels_on = flywheel_enabled

            cx, cy, detected, det_ts = detector.get()

            # ── TAG VISIBLE ─────────────────────────────────────────────────
            if detected and cx is not None:
                last_detection_time = now
                last_seen_time      = now

                error_pan, error_tilt = pixel_to_angle_error(cx, cy, aim_x, aim_y, fx, fy)

                # On the very first detection after being lost, seed last_error
                # to the current error so the derivative starts at zero instead
                # of producing a giant one-frame spike.
                if last_error_pan is None:
                    last_error_pan  = error_pan
                    last_error_tilt = error_tilt
                    filtered_deriv_pan  = 0.0
                    filtered_deriv_tilt = 0.0

                # ── Filtered derivative ──────────────────────────────────────
                raw_deriv_pan  = (error_pan  - last_error_pan)  / dt
                raw_deriv_tilt = (error_tilt - last_error_tilt) / dt

                filtered_deriv_pan  = (DERIV_ALPHA * raw_deriv_pan
                                       + (1.0 - DERIV_ALPHA) * filtered_deriv_pan)
                filtered_deriv_tilt = (DERIV_ALPHA * raw_deriv_tilt
                                       + (1.0 - DERIV_ALPHA) * filtered_deriv_tilt)

                # ── Integral with anti-windup ────────────────────────────────
                # Only accumulate when not already saturated at a servo limit.
                # This is "back-calculation" anti-windup: if the servo is already
                # at PAN_MIN/MAX, further integral growth in that direction is useless.
                at_pan_limit  = (pan_servo  <= PAN_MIN  and error_pan  < 0) or \
                                (pan_servo  >= PAN_MAX  and error_pan  > 0)
                at_tilt_limit = (tilt_servo <= TILT_MIN and error_tilt < 0) or \
                                (tilt_servo >= TILT_MAX and error_tilt > 0)

                if not at_pan_limit:
                    integral_pan  = clamp(integral_pan  + error_pan  * dt,
                                          -INTEGRAL_CLAMP, INTEGRAL_CLAMP)
                if not at_tilt_limit:
                    integral_tilt = clamp(integral_tilt + error_tilt * dt,
                                          -INTEGRAL_CLAMP, INTEGRAL_CLAMP)

                # ── PID step (degrees of servo movement this tick) ───────────
                pan_step  = ((error_pan  * KP_PAN)
                             + (filtered_deriv_pan  * KD_PAN)
                             + (integral_pan        * KI_PAN))
                tilt_step = ((error_tilt * KP_TILT)
                             + (filtered_deriv_tilt * KD_TILT)
                             + (integral_tilt       * KI_TILT))

                # Hard cap: no single tick moves the servo more than MAX_SERVO_STEP
                pan_step  = clamp(pan_step,  -MAX_SERVO_STEP, MAX_SERVO_STEP)
                tilt_step = clamp(tilt_step, -MAX_SERVO_STEP, MAX_SERVO_STEP)

                pan_servo  = clamp(pan_servo  + pan_step,  PAN_MIN,  PAN_MAX)
                tilt_servo = clamp(tilt_servo + tilt_step, TILT_MIN, TILT_MAX)

                kit.servo[PAN_CHANNEL].angle = pan_servo
                set_tilt(kit, tilt_servo)

                # Save for dead-reckoning
                vel_pan_dps         = filtered_deriv_pan
                vel_tilt_dps        = filtered_deriv_tilt
                last_known_pan_err  = error_pan
                last_known_tilt_err = error_tilt

                last_error_pan  = error_pan
                last_error_tilt = error_tilt

                # ── Dwell-before-fire ────────────────────────────────────────
                aim_error_deg = (error_pan**2 + error_tilt**2) ** 0.5
                if aim_error_deg <= TRIGGER_THRESHOLD and flywheels_on:
                    if not dwell_active:
                        dwell_active     = True
                        dwell_start_time = now
                    elif (now - dwell_start_time >= DWELL_REQUIRED_SECS) and not trigger_active:
                        kit.servo[TRIGGER_CHANNEL].fraction = 0.7
                        trigger_start_time = now
                        trigger_active     = True
                        dwell_active       = False
                else:
                    dwell_active = False

                fw_status = "ON" if flywheels_on else "OFF"
                dwell_pct = (min(100, int((now - dwell_start_time)
                             / DWELL_REQUIRED_SECS * 100)) if dwell_active else 0)
                print(
                    f"\r[LOCK ON] Pan: {error_pan:+.2f}° Tilt: {error_tilt:+.2f}°"
                    f" | Dwell: {dwell_pct:3d}% | FW: {fw_status}  ",
                    end=""
                )

            # ── TAG LOST ────────────────────────────────────────────────────
            else:
                # Reset PID state so stale accumulation doesn't kick in the
                # instant the tag reappears.
                integral_pan    = 0.0
                integral_tilt   = 0.0
                last_error_pan  = None   # forces derivative seed on re-acquisition
                last_error_tilt = None
                filtered_deriv_pan  = 0.0
                filtered_deriv_tilt = 0.0
                dwell_active    = False

                time_since_lost = now - last_detection_time

                if time_since_lost < LOST_TIMEOUT:
                    # Dead-reckoning: project last known position forward using
                    # last measured velocity, with linearly decaying confidence.
                    elapsed    = now - last_seen_time
                    confidence = max(0.0, 1.0 - (elapsed / LOST_TIMEOUT))

                    predicted_pan_err  = last_known_pan_err  + vel_pan_dps  * elapsed
                    predicted_tilt_err = last_known_tilt_err + vel_tilt_dps * elapsed

                    # Proportional-only nudge (no I/D during prediction)
                    pan_step  = clamp(predicted_pan_err  * KP_PAN  * confidence,
                                      -MAX_SERVO_STEP, MAX_SERVO_STEP)
                    tilt_step = clamp(predicted_tilt_err * KP_TILT * confidence,
                                      -MAX_SERVO_STEP, MAX_SERVO_STEP)

                    pan_servo  = clamp(pan_servo  + pan_step,  PAN_MIN,  PAN_MAX)
                    tilt_servo = clamp(tilt_servo + tilt_step, TILT_MIN, TILT_MAX)

                    kit.servo[PAN_CHANNEL].angle = pan_servo
                    set_tilt(kit, tilt_servo)

                    fw_status = "ON" if flywheels_on else "OFF"
                    print(
                        f"\r[PREDICTING] Grace: {LOST_TIMEOUT - time_since_lost:.1f}s"
                        f" | Conf: {confidence:.0%} | FW: {fw_status}  ",
                        end=""
                    )

                else:
                    # Grace expired — reset velocity memory and drift home
                    vel_pan_dps         = 0.0
                    vel_tilt_dps        = 0.0
                    last_known_pan_err  = 0.0
                    last_known_tilt_err = 0.0

                    pan_servo  += clamp((PAN_CENTER  - pan_servo)  * 0.05,
                                        -MAX_SERVO_STEP, MAX_SERVO_STEP)
                    tilt_servo += clamp((TILT_CENTER - tilt_servo) * 0.05,
                                        -MAX_SERVO_STEP, MAX_SERVO_STEP)

                    kit.servo[PAN_CHANNEL].angle = pan_servo
                    set_tilt(kit, tilt_servo)

                    fw_status = "ON" if flywheels_on else "OFF"
                    print(f"\r[SEARCHING] Returning home | FW: {fw_status}              ",
                          end="")

            # ── Trigger auto-reset ───────────────────────────────────────────
            if trigger_active and (now - trigger_start_time >= 0.5):
                kit.servo[TRIGGER_CHANNEL].fraction = 0.0
                trigger_active = False

            # ── Flywheels: user toggle is the only authority ─────────────────
            kit.servo[FLYWHEEL_CHANNEL].fraction = 0.5 if flywheels_on else 0.0

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        print("Shutting down...")
        detector.stop()
        vs.stop()
        kit.servo[PAN_CHANNEL].angle = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        kit.servo[TRIGGER_CHANNEL].fraction  = 0.0
        print("Done.")


if __name__ == "__main__":
    main()