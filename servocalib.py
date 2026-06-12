#!/usr/bin/env python3

from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

PAN_CHANNEL  = 0
TILT_CHANNEL = 1

ACTUATION_RANGE = 270

for ch in (PAN_CHANNEL, TILT_CHANNEL):
    kit.servo[ch].actuation_range = ACTUATION_RANGE
    kit.servo[ch].set_pulse_width_range(600, 2400)

# ── Calibration points ────────────────────────────────────────────────────────
# We will find the pulse width (us) that corresponds to each known angle.
# Use a protractor or reference marks on your mount.
# Fill this in as you go — at minimum you need min, center, max.
#
# angle_deg -> pulse_us
CALIBRATION = {
    0:   None,   # fill in after testing
    45:  None,
    90:  None,
    135: None,   # should be center / forward
    180: None,
    225: None,
    270: None,
}

def set_pulse(channel, pulse_us):
    """Directly set pulse width in microseconds, bypassing angle mapping."""
    kit.servo[channel]._pwm_out.duty_cycle = int(pulse_us / 20000 * 65535)

def main():
    channel = int(input("Channel to calibrate (0=pan, 1=tilt): "))
    print("Commands:")
    print("  <number>   — set pulse width in microseconds (e.g. 1500)")
    print("  a <angle>  — record current pulse for a known angle (e.g. a 135)")
    print("  q          — quit and print calibration table")

    current_pulse = 1500
    set_pulse(channel, current_pulse)

    recorded = {}

    while True:
        cmd = input(f"[{current_pulse}us] > ").strip()

        if cmd == "q":
            break

        elif cmd.startswith("a "):
            try:
                angle = float(cmd[2:])
                recorded[angle] = current_pulse
                print(f"  recorded {angle}° = {current_pulse}us")
            except ValueError:
                print("  invalid angle")

        else:
            try:
                pulse = int(cmd)
                if 400 <= pulse <= 2600:
                    current_pulse = pulse
                    set_pulse(channel, current_pulse)
                else:
                    print("  pulse out of safe range (400–2600)")
            except ValueError:
                print("  enter a pulse width in us or 'a <angle>'")

    print("\n── Calibration results ──────────────────────────")
    print("CALIBRATION = {")
    for angle, pulse in sorted(recorded.items()):
        print(f"    {angle}: {pulse},")
    print("}")

    if len(recorded) >= 2:
        angles = sorted(recorded.keys())
        pulses = [recorded[a] for a in angles]
        print("\n── Linear fit ───────────────────────────────────")
        import numpy as np
        coeffs = np.polyfit(angles, pulses, 1)
        print(f"  pulse = {coeffs[0]:.4f} * angle + {coeffs[1]:.2f}")
        min_pulse = int(np.polyval(coeffs, 0))
        max_pulse = int(np.polyval(coeffs, 270))
        print(f"  → set_pulse_width_range({min_pulse}, {max_pulse})")


if __name__ == "__main__":
    main()