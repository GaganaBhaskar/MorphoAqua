"""
MorphoAqua - Stage 1B
Improved altitude controller
"""

import numpy as np


class AltitudeController:

    def __init__(
        self,
        kp,
        ki,
        kv,
        mass,
        gravity,
        max_thrust
    ):

        self.kp = kp
        self.ki = ki
        self.kv = kv

        self.mass = mass
        self.gravity = gravity

        self.max_thrust = max_thrust

        self.integral = 0.0

    def update(
        self,
        target_altitude,
        altitude,
        vertical_velocity,
        dt
    ):

        # --------------------------------
        # Position error
        # --------------------------------

        error = (
            target_altitude
            - altitude
        )

        # --------------------------------
        # Integral
        # --------------------------------

        self.integral += error * dt

        # Anti-windup
        self.integral = np.clip(
            self.integral,
            -2.0,
            2.0
        )

        # --------------------------------
        # Gravity compensation
        # --------------------------------

        hover_thrust = (
            self.mass * self.gravity
        )

        # --------------------------------
        # Feedback control
        # --------------------------------

        correction = (
            self.kp * error
            + self.ki * self.integral
            - self.kv * vertical_velocity
        )

        thrust = (
            hover_thrust
            + correction
        )

        # --------------------------------
        # Thrust limits
        # --------------------------------

        thrust = np.clip(
            thrust,
            0.0,
            self.max_thrust
        )

        return thrust