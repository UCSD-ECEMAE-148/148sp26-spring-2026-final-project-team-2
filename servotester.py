#!/usr/bin/env python3

import math
from adafruit_servokit import ServoKit

# --------------------
# Configuration
# --------------------

PAN_CHANNEL = 0
TILT_CHANNEL = 1

# DS3240 effective range
PAN_MIN = 20
PAN_MAX = 250

TILT_MIN = 20
TILT_MAX = 250

STEP = 5

CENTER = 135  # 270° servo center


def clamp(value, low, high):
    return max(low, min(high, value))


def print_state(pan, tilt):
    pan_rad = math.radians(pan - CENTER)
    
    # FLIPPED PITCH HERE:
    tilt_rad = math.radians(CENTER - tilt)

    print("\n------------------------")
    print(f"PAN  Servo : {pan:.1f}°")
    print(f"TILT Servo : {tilt:.1f}° (FLIPPED)")
    print(f"PAN  ROS   : {pan_rad:.3f} rad")
    print(f"TILT ROS   : {tilt_rad:.3f} rad")
    print("------------------------")


def main():
    kit = ServoKit(channels=16)

    # 🔧 IMPORTANT: range + pulse tuning improves accuracy
    kit.servo[PAN_CHANNEL].actuation_range = 270
    kit.servo[TILT_CHANNEL].actuation_range = 270

    # tighter pulse range often improves DS3240 linearity
    kit.servo[PAN_CHANNEL].set_pulse_width_range(600, 2400)
    kit.servo[TILT_CHANNEL].set_pulse_width_range(600, 2400)

    pan = CENTER
    tilt = CENTER

    kit.servo[PAN_CHANNEL].angle = pan
    kit.servo[TILT_CHANNEL].angle = tilt

    print("Servo Test (FLIPPED PITCH)")
    print("a/d = pan left/right")
    print("w/s = tilt (FLIPPED)")
    print("c = center")
    print("q = quit")

    print_state(pan, tilt)

    while True:
        cmd = input("> ").strip().lower()

        if cmd == "q":
            break

        elif cmd == "a":
            pan -= STEP

        elif cmd == "d":
            pan += STEP

        elif cmd == "w":
            tilt -= STEP   # 🔥 FLIPPED HERE

        elif cmd == "s":
            tilt += STEP   # 🔥 FLIPPED HERE

        elif cmd == "c":
            pan = CENTER
            tilt = CENTER

        else:
            continue

        pan = clamp(pan, PAN_MIN, PAN_MAX)
        tilt = clamp(tilt, TILT_MIN, TILT_MAX)

        kit.servo[PAN_CHANNEL].angle = pan
        kit.servo[TILT_CHANNEL].angle = tilt

        print_state(pan, tilt)


if __name__ == "__main__":
    main()