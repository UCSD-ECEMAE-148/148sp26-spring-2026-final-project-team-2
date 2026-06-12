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

# ── Pivot offset relative to camera (metres, camera-frame axes) ──────────────
# Camera axes (OpenCV): X right, Y down, Z forward
# Pivot is 151.4 mm ABOVE  → -Y
# Pivot is 216.5 mm BEHIND → -Z
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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def tvec_to_servo_angles(tvec: np.ndarray):
    t_pivot = tvec - PIVOT_OFFSET
    px, py, pz = t_pivot

    if pz <= 0:
        return None, None

    yaw_deg   =  math.degrees(math.atan2(px,  pz))
    pitch_deg =  math.degrees(math.atan2(-py, pz))

    pan_servo  = clamp(PAN_CENTER  - yaw_deg,         PAN_MIN,  PAN_MAX)   # flipped
    tilt_servo = clamp(TILT_CENTER - pitch_deg,  TILT_MIN, TILT_MAX)  # offset -90

    return pan_servo, tilt_servo

def main():
    # ── Servo setup ──────────────────────────────────────────────────────────
    kit = ServoKit(channels=16)
    for ch in (PAN_CHANNEL, TILT_CHANNEL):
        kit.servo[ch].actuation_range = ACTUATION_RANGE
        kit.servo[ch].set_pulse_width_range(PULSE_MIN, PULSE_MAX)

    kit.servo[PAN_CHANNEL].angle  = PAN_CENTER
    kit.servo[TILT_CHANNEL].angle = TILT_CENTER

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
            t0 = time.monotonic()
            aprilTagMessage = outQueue.get()
            if aprilTagMessage is None:
                continue
            t1 = time.monotonic()
            assert isinstance(aprilTagMessage, dai.AprilTags)

            counter += 1
            now = time.monotonic()
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

                t2 = time.monotonic()

                x, y, z = tvec.flatten()
                dist = np.linalg.norm(tvec)
                print(
                    f"[FPS {fps:5.1f}]  Tag {tag.id:3d} | "
                    f"dist {dist:.3f} m | "
                    f"x {x:+.3f}  y {y:+.3f}  z {z:.3f}"
                )

                if dist < best_dist:
                    best_dist = dist
                    best_tag  = tvec.flatten()

            if best_tag is not None:
                pan_angle, tilt_angle = tvec_to_servo_angles(best_tag)
                if pan_angle is not None:
                    kit.servo[PAN_CHANNEL].angle  = pan_angle
                    kit.servo[TILT_CHANNEL].angle = tilt_angle
                    print(f"  → servo  pan {pan_angle:.1f}°  tilt {tilt_angle:.1f}°")
                t3 = time.monotonic()

                # print(
                #     f"[FPS {fps:5.1f}]  "
                #     f"queue {(t1-t0)*1000:.1f}ms  "
                #     f"solve {(t2-t1)*1000:.1f}ms  "
                #     f"servo {(t3-t2)*1000:.1f}ms  "
                #     f"total {(t3-t0)*1000:.1f}ms"
                # )


if __name__ == "__main__":
    main()