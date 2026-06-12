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
MONO_FPS = 117  # capped by OV7251 sensor at this resolution

# ── Pivot offset (metres, camera-frame axes) ─────────────────────────────────
# PIVOT_OFFSET = np.array([0.0, 0.0, 0.0])
PIVOT_OFFSET = np.array([-0.04, -0.35, -0.3])
# PIVOT_OFFSET = np.array([0.0, -0.2165, -0.1514])

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL  = 0
TILT_CHANNEL = 1

PAN_MIN,  PAN_MAX  = 20, 250
TILT_MIN, TILT_MAX = 20, 250

PAN_CENTER  = 135
TILT_CENTER = 180

ACTUATION_RANGE = 270
PULSE_MIN, PULSE_MAX = 600, 2400

DEADBAND = 0.0
ALPHA    = 0.3

PROCESS_NOISE     = 1e-2
MEASUREMENT_NOISE = 1e-1

MAX_COAST = 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
    pan_servo  = clamp(PAN_CENTER  - yaw_deg,        PAN_MIN,  PAN_MAX)
    tilt_servo = clamp(TILT_CENTER - pitch_deg, TILT_MIN, TILT_MAX)
    return pan_servo, tilt_servo


def main():
    # ── Servo setup ──────────────────────────────────────────────────────────
    kit = ServoKit(channels=16)
    for ch in (PAN_CHANNEL, TILT_CHANNEL):
        kit.servo[ch].actuation_range = ACTUATION_RANGE
        kit.servo[ch].set_pulse_width_range(PULSE_MIN, PULSE_MAX)

    kit.servo[PAN_CHANNEL].angle  = PAN_CENTER
    kit.servo[TILT_CHANNEL].angle = TILT_CENTER

    smooth_pan  = float(PAN_CENTER)
    smooth_tilt = float(TILT_CENTER)

    kf             = make_kalman()
    kf_initialised = False
    last_seen_t    = None
    last_t         = time.monotonic()

    # ── DepthAI pipeline ─────────────────────────────────────────────────────
    with dai.Pipeline() as pipeline:
        # Use left mono socket via Camera node
        camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        # camera.setFps(MONO_FPS)

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
                        pan_angle, tilt_angle = tvec_to_servo_angles(estimated)
                        if pan_angle is not None:
                            smooth_pan  = ALPHA * pan_angle  + (1.0 - ALPHA) * smooth_pan
                            smooth_tilt = ALPHA * tilt_angle + (1.0 - ALPHA) * smooth_tilt
                            if abs(pan_angle  - smooth_pan)  > DEADBAND:
                                kit.servo[PAN_CHANNEL].angle  = smooth_pan
                            if abs(tilt_angle - smooth_tilt) > DEADBAND:
                                kit.servo[TILT_CHANNEL].angle = smooth_tilt
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

            pan_angle, tilt_angle = tvec_to_servo_angles(estimated)
            if pan_angle is not None:
                smooth_pan  = ALPHA * pan_angle  + (1.0 - ALPHA) * smooth_pan
                smooth_tilt = ALPHA * tilt_angle + (1.0 - ALPHA) * smooth_tilt
                if abs(pan_angle  - smooth_pan)  > DEADBAND:
                    kit.servo[PAN_CHANNEL].angle  = smooth_pan
                if abs(tilt_angle - smooth_tilt) > DEADBAND:
                    kit.servo[TILT_CHANNEL].angle = smooth_tilt


if __name__ == "__main__":
    main()
