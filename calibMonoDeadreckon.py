#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import math
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
# PIVOT_OFFSET = np.array([0.0, -0.1514, -0.2165])

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL  = 0
TILT_CHANNEL = 1

# Pan: 135° is forward (centre). 0° is hard-right from turret perspective.
# Calibration fit: pulse = 6.4067 * angle + 552.65  → range 552–2282 µs
PAN_CENTER       = 135
PAN_ACTUATION    = 270
PAN_PULSE_MIN    = 552
PAN_PULSE_MAX    = 2282
PAN_MIN          = 20
PAN_MAX          = 250

# Tilt: 0° is forward/level. Negative angles point DOWN.
# Calibration fit: pulse = -7.4222 * angle + 1867.00
# The servo is physically inverted — increasing pulse decreases angle.
# We store tilt as a "true degrees" value (0 = level, + = up, - = down)
# and convert to servo library degrees before writing.
#
# Usable mechanical range from cal points: -90° to +90°
# At  +90°: pulse ≈ 1180 µs  → servo lib "low" end
# At  -90°: pulse ≈ 2525 µs  → servo lib "high" end
# Actuation range we tell the library: 180° (covers -90..+90)
# pulse_min = 1180, pulse_max = 2525
TILT_CENTER      =   0    # degrees (level)
TILT_ACTUATION   = 180
TILT_PULSE_MIN   = 1180   # corresponds to +90° (up)
TILT_PULSE_MAX   = 2525   # corresponds to -90° (down)
TILT_MIN         = -15    # true degrees, negative = down
TILT_MAX         =  80    # true degrees, positive = up

DEADBAND = 0.0            # degrees — ignore jitter smaller than this
ALPHA    = 0.3

PROCESS_NOISE     = 1e-2
MEASUREMENT_NOISE = 1e-1

MAX_COAST = 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def true_tilt_to_servo_degrees(true_deg):
    """
    Convert intuitive tilt angle (0=level, +up, -down) to the servo-library
    degree value that produces the correct pulse.

    The library maps:  0° → PULSE_MIN (1180 µs, which is physical +90°/up)
                      180° → PULSE_MAX (2525 µs, which is physical -90°/down)

    So servo_lib_deg = 90 - true_deg   (clipped to 0–180)
    """
    return clamp(90.0 - true_deg, 0.0, 180.0)


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
    """
    Returns (pan_deg, tilt_true_deg) in the intuitive coordinate systems:
      pan_deg       : servo-library degrees (0–270), 135 = forward
      tilt_true_deg : true degrees (0 = level, + up, - down)
    """
    t_pivot = tvec - PIVOT_OFFSET
    px, py, pz = t_pivot
    if pz <= 0:
        return None, None

    yaw_deg   =  math.degrees(math.atan2(px,  pz))   # + = target right of turret
    pitch_deg =  math.degrees(math.atan2(-py, pz))    # + = target above camera

    pan_servo  = clamp(PAN_CENTER - yaw_deg,  PAN_MIN,  PAN_MAX)
    tilt_true  = clamp(pitch_deg,             TILT_MIN, TILT_MAX)

    return pan_servo, tilt_true


def set_tilt(kit, true_deg):
    kit.servo[TILT_CHANNEL].angle = true_tilt_to_servo_degrees(true_deg)


def main():
    # ── Servo setup ──────────────────────────────────────────────────────────
    kit = ServoKit(channels=16)

    # Pan
    kit.servo[PAN_CHANNEL].actuation_range = PAN_ACTUATION
    kit.servo[PAN_CHANNEL].set_pulse_width_range(PAN_PULSE_MIN, PAN_PULSE_MAX)

    # Tilt — inverted; library min/max are swapped relative to physical angle
    kit.servo[TILT_CHANNEL].actuation_range = TILT_ACTUATION
    kit.servo[TILT_CHANNEL].set_pulse_width_range(TILT_PULSE_MIN, TILT_PULSE_MAX)

    kit.servo[PAN_CHANNEL].angle = PAN_CENTER
    set_tilt(kit, TILT_CENTER)   # 0° = level

    smooth_pan  = float(PAN_CENTER)
    smooth_tilt = float(TILT_CENTER)   # true degrees

    kf             = make_kalman()
    kf_initialised = False
    last_seen_t    = None
    last_t         = time.monotonic()

    # ── DepthAI pipeline ─────────────────────────────────────────────────────
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

        while pipeline.isRunning():
            now = time.monotonic()
            dt  = now - last_t
            last_t = now

            aprilTagMessage = outQueue.get()

            if aprilTagMessage is None:
                if kf_initialised:
                    coasting = now - last_seen_t
                    if coasting < MAX_COAST:
                        estimated = kalman_predict(kf, dt)
                        pan_angle, tilt_true = tvec_to_servo_angles(estimated)
                        if pan_angle is not None:
                            smooth_pan  = ALPHA * pan_angle  + (1.0 - ALPHA) * smooth_pan
                            smooth_tilt = ALPHA * tilt_true  + (1.0 - ALPHA) * smooth_tilt
                            if abs(pan_angle - smooth_pan)   > DEADBAND:
                                kit.servo[PAN_CHANNEL].angle = smooth_pan
                            if abs(tilt_true - smooth_tilt)  > DEADBAND:
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
                if abs(pan_angle - smooth_pan)   > DEADBAND:
                    kit.servo[PAN_CHANNEL].angle = smooth_pan
                if abs(tilt_true - smooth_tilt)  > DEADBAND:
                    set_tilt(kit, smooth_tilt)


if __name__ == "__main__":
    main()