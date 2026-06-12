import cv2
import numpy as np
import time
import math
import json
import os
from adafruit_servokit import ServoKit

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose

# ── AprilTag geometry ────────────────────────────────────────────────────────
TAG_SIZE = 0.173
half = TAG_SIZE / 2.0
OBJ_PTS = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)

# ── Camera settings ───────────────────────────────────────────────────────────
CAMERA_INDEX = 0
W, H = 320, 240

# ── Servo configuration ──────────────────────────────────────────────────────
PAN_CHANNEL, TILT_CHANNEL = 0, 1
PAN_CENTER, PAN_MIN, PAN_MAX = 135, 20, 250
TILT_CENTER, TILT_MIN, TILT_MAX = 0, -15, 80

FLYWHEEL_CHANNEL = 3
FLYWHEEL_FRACTION = 0.5
FLYWHEEL_SPINDOWN_DELAY = 1.5

# --- TUNED PID GAINS ---
# Reduced Kp for stability, increased Kd for damping, lowered Ki to prevent hunting
KP_PAN, KI_PAN, KD_PAN = 0.25, 0.01, 0.05
KP_TILT, KI_TILT, KD_TILT = 0.25, 0.01, 0.05
DEADBAND_DEG = 0.5 # Ignore errors smaller than 0.5 degrees to stop jitter

class PID:
    def __init__(self, kp, ki, kd, center_val):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.center = center_val
        self.integral = 0
        self.prev_error = 0

    def update(self, current_val, dt):
        error = current_val - self.center
        
        # Apply deadband
        if abs(error) < DEADBAND_DEG:
            return 0
            
        self.integral += error * dt
        self.integral = max(-5, min(5, self.integral)) # Tighter clamp
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def clamp(v, lo, hi): return max(lo, min(hi, v))
def true_tilt_to_servo_degrees(true_deg): return clamp(90.0 - true_deg, 0.0, 180.0)

class MultiCamShooterNode(Node):
    def __init__(self):
        super().__init__("usb_cam_shooter_pid")
        
        try:
            self.kit = ServoKit(channels=16)
            self.kit.servo[PAN_CHANNEL].set_pulse_width_range(552, 2282)
            self.kit.servo[TILT_CHANNEL].set_pulse_width_range(1180, 2525)
            self.kit.servo[PAN_CHANNEL].angle = PAN_CENTER
            self.kit.servo[TILT_CHANNEL].angle = true_tilt_to_servo_degrees(TILT_CENTER)
            self.kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
        except Exception as e:
            self.get_logger().error(f"ServoKit fail: {e}")

        self.pan_pid = PID(KP_PAN, KI_PAN, KD_PAN, 0)
        self.tilt_pid = PID(KP_TILT, KI_TILT, KD_TILT, 0)
        
        self.cur_pan = float(PAN_CENTER)
        self.cur_tilt = float(TILT_CENTER)
        
        self.last_seen_t = None
        self.last_t = time.monotonic()
        self.fw_on = False
        self.tag_pub = self.create_publisher(PoseArray, "/april_tags", 10)

    def run_loop(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Scaled Calibration Defaults
        K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)

        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE 
        
        print(f"USB PID LOOP STARTED (STABILIZED) @ {W}x{H}")
        
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: continue
            
            now = time.monotonic()
            dt = now - self.last_t
            self.last_t = now
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_list, ids, _ = cv2.aruco.detectMarkers(gray, tag_dict, parameters=params)

            best_tag = None
            best_dist = float('inf')
            if ids is not None:
                for tag_corners in corners_list:
                    img_pts = tag_corners.reshape(4, 2).astype(np.float64)
                    ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        tvec = tvec.flatten()
                        dist = np.linalg.norm(tvec)
                        if dist < best_dist:
                            best_dist, best_tag = dist, tvec

            if best_tag is not None:
                self.last_seen_t = now
                if not self.fw_on:
                    try: self.kit.servo[FLYWHEEL_CHANNEL].fraction = FLYWHEEL_FRACTION
                    except: pass
                    self.fw_on = True
                
                pz = best_tag[2]
                if pz > 0:
                    yaw_err = math.degrees(math.atan2(best_tag[0], pz))
                    pitch_err = math.degrees(math.atan2(-best_tag[1], pz))
                    
                    pan_corr = self.pan_pid.update(yaw_err, dt)
                    tilt_corr = self.tilt_pid.update(pitch_err, dt)
                    
                    self.cur_pan = clamp(self.cur_pan - pan_corr, PAN_MIN, PAN_MAX)
                    self.cur_tilt = clamp(self.cur_tilt + tilt_corr, TILT_MIN, TILT_MAX)
                    
                    try:
                        self.kit.servo[PAN_CHANNEL].angle = self.cur_pan
                        self.kit.servo[TILT_CHANNEL].angle = true_tilt_to_servo_degrees(self.cur_tilt)
                    except: pass
            else:
                if self.last_seen_t and (now - self.last_seen_t) >= FLYWHEEL_SPINDOWN_DELAY:
                    if self.fw_on:
                        try: self.kit.servo[FLYWHEEL_CHANNEL].fraction = 0.0
                        except: pass
                        self.fw_on = False
                    
        cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = MultiCamShooterNode()
    try: node.run_loop()
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == "__main__": main()
