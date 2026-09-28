"""
MorphoAqua - Stage 1B
Simple motor dynamics
"""

import numpy as np

from robot_parameters import (
    KF,
    MAX_RPM,
)


class Motor:

    def __init__(self, time_constant=0.08):

        self.rpm = 0.0
        self.time_constant = time_constant

    def update(self, commanded_rpm, dt):

        commanded_rpm = np.clip(
            commanded_rpm,
            0,
            MAX_RPM
        )

        # First-order motor response
        rpm_rate = (
            commanded_rpm - self.rpm
        ) / self.time_constant

        self.rpm += rpm_rate * dt

        return self.rpm


def rpm_to_rad_s(rpm):

    return rpm * 2.0 * np.pi / 60.0


def thrust_from_rpm(rpm):

    omega = rpm_to_rad_s(rpm)

    return KF * omega ** 2