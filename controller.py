"""
MorphoAqua Controller Module

Stage 2A:
3-D Position / Waypoint Controller

This controller uses a world-frame desired force vector.

The position controller computes:

    desired acceleration
        =
    feedforward acceleration
    + position feedback
    + velocity feedback

The desired force vector is then converted into:

    total thrust
    desired roll
    desired pitch
    desired yaw

The mapping is consistent with the ZYX rotation convention used
by the 6-DOF rigid-body simulation.
"""

import numpy as np


# ============================================================
# ANGLE UTILITIES
# ============================================================

def wrap_angle(angle):
    """
    Wrap an angle to the interval [-pi, pi].
    """

    return np.arctan2(
        np.sin(angle),
        np.cos(angle)
    )


# ============================================================
# ROTATION MATRIX
# ============================================================

def rotation_matrix(
    roll,
    pitch,
    yaw
):
    """
    Body-to-world rotation matrix.

    ZYX Euler convention:

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    The third column represents the body +Z axis expressed
    in world coordinates.
    """

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    return np.array([
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr
        ],
        [
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr
        ],
        [
            -sp,
            cp * sr,
            cp * cr
        ]
    ])


# ============================================================
# POSITION CONTROLLER
# ============================================================

class PositionController:

    def __init__(
        self,

        kp_x,
        kp_y,
        kp_z,

        kd_x,
        kd_y,
        kd_z,

        mass,
        gravity,

        max_roll,
        max_pitch,

        max_horizontal_acceleration=2.5,
        max_vertical_acceleration=3.0
    ):

        self.kp = np.array([
            kp_x,
            kp_y,
            kp_z
        ], dtype=float)

        self.kd = np.array([
            kd_x,
            kd_y,
            kd_z
        ], dtype=float)

        self.mass = float(mass)

        self.gravity = float(gravity)

        self.max_roll = float(max_roll)

        self.max_pitch = float(max_pitch)

        self.max_horizontal_acceleration = float(
            max_horizontal_acceleration
        )

        self.max_vertical_acceleration = float(
            max_vertical_acceleration
        )


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        target_position,
        current_position,
        current_velocity,
        target_yaw=0.0,
        target_velocity=None,
        target_acceleration=None
    ):

        target_position = np.asarray(
            target_position,
            dtype=float
        )

        current_position = np.asarray(
            current_position,
            dtype=float
        )

        current_velocity = np.asarray(
            current_velocity,
            dtype=float
        )


        # ----------------------------------------------------
        # DEFAULT REFERENCE VELOCITY
        # ----------------------------------------------------

        if target_velocity is None:

            target_velocity = np.zeros(3)

        else:

            target_velocity = np.asarray(
                target_velocity,
                dtype=float
            )


        # ----------------------------------------------------
        # DEFAULT FEEDFORWARD ACCELERATION
        # ----------------------------------------------------

        if target_acceleration is None:

            target_acceleration = np.zeros(3)

        else:

            target_acceleration = np.asarray(
                target_acceleration,
                dtype=float
            )


        # ----------------------------------------------------
        # POSITION ERROR
        # ----------------------------------------------------

        position_error = (
            target_position
            - current_position
        )


        # ----------------------------------------------------
        # VELOCITY ERROR
        # ----------------------------------------------------

        velocity_error = (
            target_velocity
            - current_velocity
        )


        # ----------------------------------------------------
        # PD + FEEDFORWARD ACCELERATION
        # ----------------------------------------------------

        acceleration_command = (
            target_acceleration
            + self.kp * position_error
            + self.kd * velocity_error
        )


        # ----------------------------------------------------
        # HORIZONTAL ACCELERATION LIMIT
        # ----------------------------------------------------

        horizontal_acceleration = np.array([
            acceleration_command[0],
            acceleration_command[1]
        ])

        horizontal_magnitude = np.linalg.norm(
            horizontal_acceleration
        )

        if (
            horizontal_magnitude
            > self.max_horizontal_acceleration
        ):

            horizontal_acceleration = (
                horizontal_acceleration
                * self.max_horizontal_acceleration
                / horizontal_magnitude
            )

            acceleration_command[0] = (
                horizontal_acceleration[0]
            )

            acceleration_command[1] = (
                horizontal_acceleration[1]
            )


        # ----------------------------------------------------
        # VERTICAL ACCELERATION LIMIT
        # ----------------------------------------------------

        vertical_feedback = (
            acceleration_command[2]
            - self.gravity
        )

        vertical_feedback = np.clip(
            vertical_feedback,
            -self.max_vertical_acceleration,
            self.max_vertical_acceleration
        )

        acceleration_command[2] = (
            self.gravity
            + vertical_feedback
        )


        # ----------------------------------------------------
        # DESIRED FORCE VECTOR
        # ----------------------------------------------------

        desired_force = (
            self.mass
            * acceleration_command
        )


        # ----------------------------------------------------
        # TOTAL THRUST
        # ----------------------------------------------------

        total_thrust = np.linalg.norm(
            desired_force
        )


        # Prevent division by zero.
        if total_thrust < 1e-9:

            total_thrust = (
                self.mass
                * self.gravity
            )

            desired_force = np.array([
                0.0,
                0.0,
                total_thrust
            ])


        # ----------------------------------------------------
        # DESIRED BODY Z AXIS
        # ----------------------------------------------------

        body_z_desired = (
            desired_force
            / total_thrust
        )


        # ----------------------------------------------------
        # DESIRED ATTITUDE
        #
        # For the ZYX convention:
        #
        # bx = cpsi*s_theta*c_phi + spsi*s_phi
        # by = spsi*s_theta*c_phi - cpsi*s_phi
        #
        # Therefore:
        #
        # roll  = asin(
        #           bx*sin(psi)
        #           - by*cos(psi)
        #         )
        #
        # pitch = atan2(
        #           bx*cos(psi)
        #           + by*sin(psi),
        #           bz
        #         )
        # ----------------------------------------------------

        yaw = float(target_yaw)

        bx = body_z_desired[0]
        by = body_z_desired[1]
        bz = body_z_desired[2]

        roll_argument = (
            bx * np.sin(yaw)
            - by * np.cos(yaw)
        )

        roll_argument = np.clip(
            roll_argument,
            -1.0,
            1.0
        )

        desired_roll = np.arcsin(
            roll_argument
        )

        desired_pitch = np.arctan2(
            (
                bx * np.cos(yaw)
                + by * np.sin(yaw)
            ),
            bz
        )


        # ----------------------------------------------------
        # ATTITUDE LIMITS
        # ----------------------------------------------------

        desired_roll = np.clip(
            desired_roll,
            -self.max_roll,
            self.max_roll
        )

        desired_pitch = np.clip(
            desired_pitch,
            -self.max_pitch,
            self.max_pitch
        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return (
            total_thrust,
            desired_roll,
            desired_pitch,
            yaw
        )


# ============================================================
# ATTITUDE CONTROLLER
# ============================================================

class AttitudeController:

    def __init__(
        self,

        kp_roll,
        kp_pitch,
        kp_yaw,

        kd_roll,
        kd_pitch,
        kd_yaw,

        max_roll_torque=None,
        max_pitch_torque=None,
        max_yaw_torque=None
    ):

        self.kp = np.array([
            kp_roll,
            kp_pitch,
            kp_yaw
        ], dtype=float)

        self.kd = np.array([
            kd_roll,
            kd_pitch,
            kd_yaw
        ], dtype=float)


        self.max_torque = None

        if (
            max_roll_torque is not None
            and max_pitch_torque is not None
            and max_yaw_torque is not None
        ):

            self.max_torque = np.array([
                max_roll_torque,
                max_pitch_torque,
                max_yaw_torque
            ], dtype=float)


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        target_angles,
        current_angles,
        angular_rates
    ):

        target_angles = np.asarray(
            target_angles,
            dtype=float
        )

        current_angles = np.asarray(
            current_angles,
            dtype=float
        )

        angular_rates = np.asarray(
            angular_rates,
            dtype=float
        )


        # ----------------------------------------------------
        # ATTITUDE ERROR
        # ----------------------------------------------------

        angle_error = np.array([
            wrap_angle(
                target_angles[0]
                - current_angles[0]
            ),

            wrap_angle(
                target_angles[1]
                - current_angles[1]
            ),

            wrap_angle(
                target_angles[2]
                - current_angles[2]
            )
        ])


        # ----------------------------------------------------
        # PD ATTITUDE CONTROL
        # ----------------------------------------------------

        torque_command = (
            self.kp * angle_error
            - self.kd * angular_rates
        )


        # ----------------------------------------------------
        # TORQUE LIMIT
        # ----------------------------------------------------

        if self.max_torque is not None:

            torque_command = np.clip(
                torque_command,
                -self.max_torque,
                self.max_torque
            )


        return torque_command