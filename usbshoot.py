#!/usr/bin/env python3

import cv2
import numpy as np
import time
import math
import sys
import tty
import termios
import threading
import json
from adafruit_servokit import ServoKit

# ── AprilTag geometry ────────────────────────────────────────────────────────
TAG_SIZE = 0.173

# ── Camera settings ───────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
CALIBRATION_FILE = "calibration.json"

# ── PID gains ────────────────────────────────────────────────────────────────
# Error is in normalised units: pixel_error / image_half_dimension
# so gains are in servo-degrees per unit error
PAN_KP   = 7.0
PAN_KI   = 0.0
PAN_KD   = 1.0

TILT_KP  = 2.0
TILT_KI  = 0.0
TILT_KD  = 0.0

# Integrator anti-windup clamp (normalised units)
PAN_I_CLAMP  = 0.3
TILT_I_CLAMP = 0.3

# ── Auto-shoot threshold ─────────────────────────────────────────────────────
# Combined error magnitude (normalised) below which we fire
# 0.05 ≈ 5% of half-frame width — tune to taste
AIM_ERROR_THRESHOLD = 0.05
# Must hold lock for this many seconds before firing
AIM_HOLD_DURATION   = 0.3

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL  = 0
TILT_CHANNEL = 1

PAN_CENTER       = 135
PAN_ACTUATION    = 270
PAN_PULSE_MIN    = 552
PAN_PULSE_MAX    = 2282
PAN_MIN          = 20
PAN_MAX          = 250

TILT_CENTER      =   0
TILT_ACTUATION   = 180
TILT_PULSE_MIN   = 1180
TILT_PULSE_MAX   = 2525
TILT_MIN         = -15
TILT_MAX         =  80

# ── Shooter configuration ────────────────────────────────────────────────────
FLYWHEEL_CHANNEL   = 3
TRIGGER_CHANNEL    = 4

FLYWHEEL_PULSE_MIN = 1000
FLYWHEEL_PULSE_MAX = 2000
FLYWHEEL_FRACTION  = 0.5

TRIGGER_PULSE_MIN  = 1000
TRIGGER_PULSE_MAX  = 2000
TRIGGER_FRACTION   = 0.7
TRIGGER_DURATION   = 1.0

FLYWHEEL_SPINDOWN_DELAY = 1.5

PROCESS_NOISE     = 1e-2
MEASUREMENT_NOISE = 1e-1
MAX_COAST         = 0.5

# ── Shared state ─────────────────────────────────────────────────────────────
_shooting      = False
_shoot_lock    = threading.Lock()
_flywheel_on   = False
_flywheel_lock = threading.Lock()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def true_tilt_to_servo_degrees(true_deg):
    """Convert true tilt angle (0=level) to servo command angle."""
    return clamp(90.0 - true_deg, 0.0, 180.0)


def set_tilt(kit, true_deg):
    """Write a true tilt angle to the servo, applying the inversion."""
    kit.servo[TILT_CHANNEL].angle = true_tilt_to_servo_degrees(true_deg)


def set_flywheel(kit, on):
    global _flywheel_on
    with _flywheel_lock:
        if _flywheel_on == on:
            return
        _flywheel_on = on
        kit.servo[FLYWHEEL_CHANNEL].fraction = FLYWHEEL_FRACTION if on else 0.0
    print(f"\r[FLYWHEEL {'ON ' if on else 'OFF'}]  ")


def trigger_shoot(kit):
    global _shooting
    with _shoot_lock:
        if _shooting:
            return
        _shooting = True
    try:
        with _flywheel_lock:
            fw_on = _flywheel_on
        if not fw_on:
            print("\r[FLYWHEEL OFF — spinning up first]  ")
            return
        print("\r[SHOOT]  ")
        kit.servo[TRIGGER_CHANNEL].fraction = TRIGGER_FRACTION
        time.sleep(TRIGGER_DURATION)
        kit.servo[TRIGGER_CHANNEL].fraction = 0.0
        print("\r[READY]  ")
    finally:
        with _shoot_lock:
            _shooting = False


# ── Kalman filter (tracks tag centre in image pixels) ────────────────────────
def make_kalman():
    # State: [cx, cy, vx, vy]  Measurement: [cx, cy]
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    kf.transitionMatrix    = np.eye(4, dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


def kalman_predict(kf, dt):
    kf.transitionMatrix[0, 2] = dt
    kf.transitionMatrix[1, 3] = dt
    return kf.predict()[:2].flatten()


def kalman_correct(kf, cx, cy):
    return kf.correct(
        np.array([[cx], [cy]], dtype=np.float32)
    )[:2].flatten()


# ── PID controller ────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, i_clamp):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_clamp = i_clamp
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral = clamp(
            self.integral + error * dt,
            -self.i_clamp, self.i_clamp
        )
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    K    = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["dist_coeffs"],   dtype=np.float64)
    print(f"Loaded calibration from '{path}'  (RMS was {cal.get('rms_error', '?'):.4f})")
    return K, dist


def main():
    # ── Load calibration ─────────────────────────────────────────────────────
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)

    # ── Open USB camera ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: could not open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"USB camera opened: {actual_w}x{actual_h}")

    # Precompute undistortion maps so we don't redo it every frame
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (actual_w, actual_h), alpha=0
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_K, (actual_w, actual_h), cv2.CV_16SC2
    )

    # Image centre (barrel aimpoint) — use undistorted camera centre
    # aim_x = actual_w / 2.0
    # aim_y = actual_h / 2.0

    aim_x = new_K[0, 2]   # principal point from undistorted camera matrix
    aim_y = new_K[1, 2]

    # Normalisation factors (error will be in range [-1, 1])
    norm_x = actual_w / 2.0
    norm_y = actual_h / 2.0

    # ── AprilTag detector ────────────────────────────────────────────────────
    detector_params = cv2.aruco.DetectorParameters()
    tag_dict        = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_detector    = cv2.aruco.ArucoDetector(tag_dict, detector_params)

    # ── Servo setup ──────────────────────────────────────────────────────────
    kit = ServoKit(channels=16)

    kit.servo[PAN_CHANNEL].actuation_range = PAN_ACTUATION
    kit.servo[PAN_CHANNEL].set_pulse_width_range(PAN_PULSE_MIN, PAN_PULSE_MAX)

    kit.servo[TILT_CHANNEL].actuation_range = TILT_ACTUATION
    kit.servo[TILT_CHANNEL].set_pulse_width_range(TILT_PULSE_MIN, TILT_PULSE_MAX)

    kit.servo[PAN_CHANNEL].angle = PAN_CENTER
    set_tilt(kit, TILT_CENTER)   # always goes through conversion

    kit.servo[FLYWHEEL_CHANNEL].set_pulse_width_range(FLYWHEEL_PULSE_MIN, FLYWHEEL_PULSE_MAX)
    kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0

    kit.servo[TRIGGER_CHANNEL].set_pulse_width_range(TRIGGER_PULSE_MIN, TRIGGER_PULSE_MAX)
    kit.servo[TRIGGER_CHANNEL].fraction = 0.0

    pan_servo  = float(PAN_CENTER)
    tilt_servo = float(TILT_CENTER)   # always in true degrees

    pan_pid  = PID(PAN_KP,  PAN_KI,  PAN_KD,  PAN_I_CLAMP)
    tilt_pid = PID(TILT_KP, TILT_KI, TILT_KD, TILT_I_CLAMP)

    kf             = make_kalman()
    kf_initialised = False
    last_seen_t    = None
    last_t         = time.monotonic()

    aim_locked_since = None
    fps_accum_start  = time.monotonic()
    fps_accum_count  = 0
    fps              = 0.0

    # ── Key listener (manual shoot) ──────────────────────────────────────────
    def _key_listener():
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == 'a':
                    threading.Thread(target=trigger_shoot, args=(kit,), daemon=True).start()
                elif ch in ('\x03', '\x1b'):
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=_key_listener, daemon=True).start()

    print(f"Pan  centre={PAN_CENTER}°  |  Tilt centre=0°")
    print(f"Auto-shoot threshold: {AIM_ERROR_THRESHOLD:.3f} normalised  "
          f"({AIM_ERROR_THRESHOLD * norm_x:.1f} px x  /  "
          f"{AIM_ERROR_THRESHOLD * norm_y:.1f} px y)")
    print("Press 'a' to shoot manually, Ctrl-C to quit.")

    try:
        while True:
            now = time.monotonic()
            dt  = max(now - last_t, 1e-4)
            last_t = now

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # FPS — accumulate over whole seconds for a stable reading
            fps_accum_count += 1
            elapsed = now - fps_accum_start
            if elapsed >= 1.0:
                fps = fps_accum_count / elapsed
                fps_accum_count = 0
                fps_accum_start = now

            # Undistort frame so tag centres are geometrically correct
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

            corners_list, ids, _ = tag_detector.detectMarkers(gray)

            # ── No detection ─────────────────────────────────────────────────
            if ids is None or len(ids) == 0:
                aim_locked_since = None
                if last_seen_t is not None:
                    coasting = now - last_seen_t
                    if coasting >= FLYWHEEL_SPINDOWN_DELAY:
                        set_flywheel(kit, False)
                    if kf_initialised and coasting < MAX_COAST:
                        pred = kalman_predict(kf, dt)
                        cx_est, cy_est = pred[0], pred[1]
                        err_x = (cx_est - aim_x) / norm_x
                        err_y = (cy_est - aim_y) / norm_y
                        pan_servo  = clamp(pan_servo  - pan_pid.update(err_x, dt), PAN_MIN,  PAN_MAX)
                        tilt_servo = clamp(tilt_servo + tilt_pid.update(err_y, dt), TILT_MIN, TILT_MAX)
                        kit.servo[PAN_CHANNEL].angle = pan_servo
                        if TILT_KP != 0 or TILT_KI != 0 or TILT_KD != 0:
                            set_tilt(kit, tilt_servo)
                    else:
                        pan_pid.reset()
                        tilt_pid.reset()
                        kf_initialised = False
                continue

            # ── Pick the tag whose centre is closest to the aim point ────────
            best_dist = float("inf")
            best_cx = best_cy = None

            for tag_corners in corners_list:
                pts = tag_corners.reshape(4, 2)
                cx  = pts[:, 0].mean()
                cy  = pts[:, 1].mean()
                d   = math.hypot(cx - aim_x, cy - aim_y)
                if d < best_dist:
                    best_dist = d
                    best_cx, best_cy = cx, cy

            # ── Kalman update ────────────────────────────────────────────────
            if not kf_initialised:
                kf.statePost = np.array(
                    [best_cx, best_cy, 0.0, 0.0], dtype=np.float32
                ).reshape(4, 1)
                kf_initialised = True
                pan_pid.reset()
                tilt_pid.reset()

            kalman_predict(kf, dt)
            est = kalman_correct(kf, best_cx, best_cy)
            cx_est, cy_est = est[0], est[1]

            last_seen_t = now
            set_flywheel(kit, True)

            # ── PID errors (normalised, positive = target right/down) ────────
            err_x = (cx_est - aim_x) / norm_x
            err_y = (cy_est - aim_y) / norm_y

            pan_servo  = clamp(pan_servo  - pan_pid.update(err_x, dt), PAN_MIN,  PAN_MAX)
            tilt_servo = clamp(tilt_servo + tilt_pid.update(err_y, dt), TILT_MIN, TILT_MAX)

            kit.servo[PAN_CHANNEL].angle = pan_servo
            if TILT_KP != 0 or TILT_KI != 0 or TILT_KD != 0:
                set_tilt(kit, tilt_servo)

            # ── Auto-shoot logic ─────────────────────────────────────────────
            error_mag = math.hypot(err_x, err_y)
            if error_mag < AIM_ERROR_THRESHOLD:
                if aim_locked_since is None:
                    aim_locked_since = now
                held = now - aim_locked_since
                lock_str = f"LOCKED {held*1000:.0f}ms"
                if held >= AIM_HOLD_DURATION:
                    with _flywheel_lock:
                        fw_ready = _flywheel_on
                    if fw_ready:
                        threading.Thread(
                            target=trigger_shoot, args=(kit,), daemon=True
                        ).start()
                        aim_locked_since = None
            else:
                aim_locked_since = None
                lock_str = f"err {error_mag:.3f}"

            print(
                f"[FPS {fps:5.1f}]  "
                f"cx {cx_est:6.1f}  cy {cy_est:6.1f}  |  "
                f"ex {err_x:+.3f}  ey {err_y:+.3f}  |  "
                f"pan {pan_servo:6.1f}°  tilt {tilt_servo:+5.1f}°  |  {lock_str}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Releasing servos...")
        cap.release()
        kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        kit.servo[TRIGGER_CHANNEL].fraction  = 0.0
        kit.servo[PAN_CHANNEL].angle         = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        print("Done.")


if __name__ == "__main__":
    main()