#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import math
import sys
import tty
import termios
import threading
from adafruit_servokit import ServoKit

# ── AprilTag geometry ────────────────────────────────────────────────────────
TAG_SIZE = 0.173
half = TAG_SIZE / 2.0
obj_pts = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)

# ── Camera settings ───────────────────────────────────────────────────────────
MONO_W, MONO_H = 640, 400
MONO_FPS = 117

# ── Pivot offset (metres, camera-frame axes) ─────────────────────────────────
PIVOT_OFFSET = np.array([0.1, -0.35, -0.2])

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

FLYWHEEL_SPINDOWN_DELAY = 1.5  # seconds after tag lost before spinning down

DEADBAND = 0.0
ALPHA    = 0.3

PROCESS_NOISE     = 1e-2
MEASUREMENT_NOISE = 1e-1

MAX_COAST = 0.5

# ── Shared state ─────────────────────────────────────────────────────────────
_shooting      = False
_shoot_lock    = threading.Lock()
_flywheel_on   = False
_flywheel_lock = threading.Lock()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def true_tilt_to_servo_degrees(true_deg):
    return clamp(90.0 - true_deg, 0.0, 180.0)


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
            print("\r[FLYWHEEL OFF — target needed to spin up]  ")
            return
        print("\r[SHOOT]  ")
        kit.servo[TRIGGER_CHANNEL].fraction = TRIGGER_FRACTION
        time.sleep(TRIGGER_DURATION)
        kit.servo[TRIGGER_CHANNEL].fraction = 0.0
        print("\r[READY]  ")
    finally:
        with _shoot_lock:
            _shooting = False


def make_kalman():
    kf = cv2.KalmanFilter(6, 3)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=np.float32)
    kf.transitionMatrix    = np.eye(6, dtype=np.float32)
    kf.processNoiseCov     = np.eye(6, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * MEASUREMENT_NOISE
    kf.errorCovPost        = np.eye(6, dtype=np.float32) * 1.0
    return kf


def kalman_predict(kf, dt):
    kf.transitionMatrix[0, 3] = dt
    kf.transitionMatrix[1, 4] = dt
    kf.transitionMatrix[2, 5] = dt
    return kf.predict()[:3].flatten()


def kalman_correct(kf, tvec):
    return kf.correct(tvec.astype(np.float32).reshape(3, 1))[:3].flatten()


def tvec_to_servo_angles(tvec):
    t_pivot = tvec - PIVOT_OFFSET
    px, py, pz = t_pivot
    if pz <= 0:
        return None, None

    yaw_deg   =  math.degrees(math.atan2(px,  pz))
    pitch_deg =  math.degrees(math.atan2(-py, pz))

    pan_servo = clamp(PAN_CENTER - yaw_deg,  PAN_MIN,  PAN_MAX)
    tilt_true = clamp(pitch_deg,             TILT_MIN, TILT_MAX)

    return pan_servo, tilt_true


def set_tilt(kit, true_deg):
    kit.servo[TILT_CHANNEL].angle = true_tilt_to_servo_degrees(true_deg)


def main():
    # ── Servo setup ──────────────────────────────────────────────────────────
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

    smooth_pan  = float(PAN_CENTER)
    smooth_tilt = float(TILT_CENTER)

    kf             = make_kalman()
    kf_initialised = False
    last_seen_t    = None
    last_t         = time.monotonic()

    # ── Key listener ─────────────────────────────────────────────────────────
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

    # ── DepthAI pipeline ─────────────────────────────────────────────────────
    try:
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)

            camOut = camera.requestOutput(
                (MONO_W, MONO_H),
                dai.ImgFrame.Type.GRAY8
            )

            aprilTagNode = pipeline.create(dai.node.AprilTag)
            aprilTagNode.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)
            camOut.link(aprilTagNode.inputImage)

            outQueue = aprilTagNode.out.createOutputQueue(maxSize=1, blocking=False)

            device = pipeline.getDefaultDevice()
            calib  = device.readCalibration()
            M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, MONO_W, MONO_H)
            D = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B)

            camera_matrix = np.array(M, dtype=np.float64)
            dist_coeffs   = np.array(D, dtype=np.float64)

            startTime = time.monotonic()
            counter   = 0
            fps       = 0.0

            pipeline.start()
            print(f"Pipeline started — CAM_B @ {MONO_FPS}fps {MONO_W}x{MONO_H} GRAY8")
            print(f"Pan  centre={PAN_CENTER}°  pulse {PAN_PULSE_MIN}–{PAN_PULSE_MAX} µs")
            print(f"Tilt centre=0° (level)    pulse {TILT_PULSE_MIN}–{TILT_PULSE_MAX} µs  (inverted)")
            print("Flywheel spins up on tag detection. Press 'a' to shoot, Ctrl-C to quit.")

            while pipeline.isRunning():
                now = time.monotonic()
                dt  = now - last_t
                last_t = now

                aprilTagMessage = outQueue.get()

                if aprilTagMessage is None:
                    # Spindown check — runs regardless of kf state
                    if last_seen_t is not None:
                        coasting = now - last_seen_t
                        if coasting >= FLYWHEEL_SPINDOWN_DELAY:
                            set_flywheel(kit, False)
                        if kf_initialised and coasting < MAX_COAST:
                            estimated = kalman_predict(kf, dt)
                            pan_angle, tilt_true = tvec_to_servo_angles(estimated)
                            if pan_angle is not None:
                                smooth_pan  = ALPHA * pan_angle  + (1.0 - ALPHA) * smooth_pan
                                smooth_tilt = ALPHA * tilt_true  + (1.0 - ALPHA) * smooth_tilt
                                if abs(pan_angle - smooth_pan)  > DEADBAND:
                                    kit.servo[PAN_CHANNEL].angle = smooth_pan
                                if abs(tilt_true - smooth_tilt) > DEADBAND:
                                    set_tilt(kit, smooth_tilt)
                    continue

                counter += 1
                if now - startTime > 1.0:
                    fps = counter / (now - startTime)
                    counter   = 0
                    startTime = now

                best_tag  = None
                best_dist = float("inf")

                for tag in aprilTagMessage.aprilTags:
                    img_pts = np.array([
                        [tag.topLeft.x,     tag.topLeft.y],
                        [tag.topRight.x,    tag.topRight.y],
                        [tag.bottomRight.x, tag.bottomRight.y],
                        [tag.bottomLeft.x,  tag.bottomLeft.y],
                    ], dtype=np.float64)

                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if not ok:
                        continue

                    dist = np.linalg.norm(tvec)
                    if dist < best_dist:
                        best_dist = dist
                        best_tag  = tvec.flatten()

                if best_tag is not None:
                    last_seen_t = now

                    set_flywheel(kit, True)

                    if not kf_initialised:
                        kf.statePost = np.array(
                            [best_tag[0], best_tag[1], best_tag[2], 0, 0, 0],
                            dtype=np.float32
                        ).reshape(6, 1)
                        kf_initialised = True

                    kalman_predict(kf, dt)
                    estimated = kalman_correct(kf, best_tag)

                    print(
                        f"[FPS {fps:5.1f}]  "
                        f"raw  x {best_tag[0]:+.3f}  y {best_tag[1]:+.3f}  z {best_tag[2]:.3f}  |  "
                        f"filt x {estimated[0]:+.3f}  y {estimated[1]:+.3f}  z {estimated[2]:.3f}"
                    )

                elif kf_initialised:
                    coasting = now - last_seen_t
                    if coasting >= MAX_COAST:
                        kf_initialised = False
                        print("Tag lost — Kalman reset")
                        continue
                    estimated = kalman_predict(kf, dt)
                    print(f"[FPS {fps:5.1f}]  coasting {coasting*1000:.0f}ms")
                else:
                    continue

                pan_angle, tilt_true = tvec_to_servo_angles(estimated)
                if pan_angle is not None:
                    smooth_pan  = ALPHA * pan_angle  + (1.0 - ALPHA) * smooth_pan
                    smooth_tilt = ALPHA * tilt_true  + (1.0 - ALPHA) * smooth_tilt
                    if abs(pan_angle - smooth_pan)  > DEADBAND:
                        kit.servo[PAN_CHANNEL].angle = smooth_pan
                    if abs(tilt_true - smooth_tilt) > DEADBAND:
                        set_tilt(kit, smooth_tilt)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Releasing servos...")
        kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        kit.servo[TRIGGER_CHANNEL].fraction  = 0.0
        kit.servo[PAN_CHANNEL].angle         = PAN_CENTER
        set_tilt(kit, TILT_CENTER)
        print("Done.")


if __name__ == "__main__":
    main()