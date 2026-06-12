#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import math
from adafruit_servokit import ServoKit

# ── AprilTag geometry ────────────────────────────────────────────────────────
TAG_SIZE = 0.162
half = TAG_SIZE / 2.0
obj_pts = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)

# ── Camera / capture settings ────────────────────────────────────────────────
FULL_RES = (640, 480)
# FULL_RES = (418, 418)

# ── Pivot offset (metres, camera-frame axes) ─────────────────────────────────
PIVOT_OFFSET = np.array([0.0, -0.1514, -0.2165])

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL  = 0
TILT_CHANNEL = 1

PAN_MIN,  PAN_MAX  = 20, 250
TILT_MIN, TILT_MAX = 20, 250

PAN_CENTER  = 135
TILT_CENTER = 180

ACTUATION_RANGE = 270
PULSE_MIN, PULSE_MAX = 600, 2400

DEADBAND = 0.5

# ── Smoothing (applied after Kalman, catches servo buzz) ─────────────────────
ALPHA = 0.3  # can be higher now that Kalman handles the noise


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Kalman filter ─────────────────────────────────────────────────────────────
# State:       [x, y, z, vx, vy, vz]  (position + velocity in camera frame)
# Measurement: [x, y, z]              (raw tvec from solvePnP)
#
# Tuning:
#   PROCESS_NOISE    — how much we trust the constant-velocity model.
#                      increase if the tag accelerates quickly.
#   MEASUREMENT_NOISE — how noisy solvePnP is.
#                      increase if you see jitter at a fixed pose.

PROCESS_NOISE    = 1e-2
MEASUREMENT_NOISE = 1e-1

def make_kalman():
    kf = cv2.KalmanFilter(6, 3)  # 6 state dims, 3 measurement dims

    # Measurement matrix H: we observe x, y, z directly
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=np.float32)

    # Transition matrix F: constant-velocity model, dt filled in each step
    kf.transitionMatrix = np.eye(6, dtype=np.float32)

    # Process noise Q
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * PROCESS_NOISE

    # Measurement noise R
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * MEASUREMENT_NOISE

    # Initial state covariance — high uncertainty at start
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

    return kf


def kalman_predict(kf, dt: float) -> np.ndarray:
    """Update transition matrix with current dt, then predict."""
    kf.transitionMatrix[0, 3] = dt
    kf.transitionMatrix[1, 4] = dt
    kf.transitionMatrix[2, 5] = dt
    predicted = kf.predict()
    return predicted[:3].flatten()


def kalman_correct(kf, tvec: np.ndarray) -> np.ndarray:
    """Feed a measurement in, get the corrected position back."""
    measurement = tvec.astype(np.float32).reshape(3, 1)
    corrected = kf.correct(measurement)
    return corrected[:3].flatten()


def tvec_to_servo_angles(tvec: np.ndarray):
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

    # ── Kalman state ─────────────────────────────────────────────────────────
    kf            = make_kalman()
    kf_initialised = False
    last_seen_t   = None   # monotonic time of last detection
    last_t        = time.monotonic()

    # How long (seconds) to keep predicting forward without a detection
    # before we consider the tag lost and reset
    MAX_COAST = 0.5

    # ── DepthAI pipeline ─────────────────────────────────────────────────────
    with dai.Pipeline() as pipeline:
        hostCamera   = pipeline.create(dai.node.Camera).build()
        aprilTagNode = pipeline.create(dai.node.AprilTag)
        outputCam    = hostCamera.requestOutput(FULL_RES)
        outputCam.link(aprilTagNode.inputImage)
        outQueue     = aprilTagNode.out.createOutputQueue(maxSize=1, blocking=False)

        device = pipeline.getDefaultDevice()
        calib  = device.readCalibration()
        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
                                      FULL_RES[0], FULL_RES[1])
        D = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)

        camera_matrix = np.array(M, dtype=np.float64)
        dist_coeffs   = np.array(D, dtype=np.float64)

        startTime = time.monotonic()
        counter   = 0
        fps       = 0.0

        pipeline.start()

        while pipeline.isRunning():
            now = time.monotonic()
            dt  = now - last_t
            last_t = now

            aprilTagMessage = outQueue.get()
            if aprilTagMessage is None:
                # No new frame — if Kalman is running, coast on prediction
                if kf_initialised:
                    coasting = (now - last_seen_t) if last_seen_t else 0
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
                    # Seed state with first measurement, zero velocity
                    kf.statePost = np.array(
                        [best_tag[0], best_tag[1], best_tag[2], 0, 0, 0],
                        dtype=np.float32
                    ).reshape(6, 1)
                    kf_initialised = True

                # Predict forward then correct with measurement
                kalman_predict(kf, dt)
                estimated = kalman_correct(kf, best_tag)

                print(
                    f"[FPS {fps:5.1f}]  "
                    f"raw  x {best_tag[0]:+.3f}  y {best_tag[1]:+.3f}  z {best_tag[2]:.3f}  |  "
                    f"filt x {estimated[0]:+.3f}  y {estimated[1]:+.3f}  z {estimated[2]:.3f}"
                )

            elif kf_initialised:
                # Tag not visible this frame — coast on prediction only
                coasting = (now - last_seen_t) if last_seen_t else 0
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