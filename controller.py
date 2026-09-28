"""
MorphoAqua - Stage 1C
6-DOF Position and Attitude Controller
"""

import numpy as np


# ============================================================
# ROTATION MATRIX
# ============================================================

def rotation_matrix(phi, theta, psi):

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array([

        [
            cpsi * ctheta,
            cpsi * stheta * sphi - spsi * cphi,
            cpsi * stheta * cphi + spsi * sphi
        ],

        [
            spsi * ctheta,
            spsi * stheta * sphi + cpsi * cphi,
            spsi * stheta * cphi - cpsi * sphi
        ],

        [
            -stheta,
            ctheta * sphi,
            ctheta * cphi
        ]

    ])

    return R


# ============================================================
# EXTRACT EULER ANGLES FROM ROTATION MATRIX
# ============================================================

def rotation_to_euler(R):

    theta = np.arcsin(
        np.clip(
            -R[2, 0],
            -1.0,
            1.0
        )
    )

    phi = np.arctan2(
        R[2, 1],
        R[2, 2]
    )

    psi = np.arctan2(
        R[1, 0],
        R[0, 0]
    )

    return phi, theta, psi


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
        max_pitch
    ):

        self.kp_x = kp_x
        self.kp_y = kp_y
        self.kp_z = kp_z

        self.kd_x = kd_x
        self.kd_y = kd_y
        self.kd_z = kd_z

        self.mass = mass
        self.gravity = gravity

        self.max_roll = max_roll
        self.max_pitch = max_pitch


    def update(
        self,
        target_position,
        position,
        velocity,
        yaw
    ):

        # ====================================================
        # POSITION ERROR
        # ====================================================

        error = (
            target_position
            - position
        )


        # ====================================================
        # DESIRED ACCELERATION
        # ====================================================

        ax = (

            self.kp_x * error[0]

            - self.kd_x * velocity[0]

        )


        ay = (

            self.kp_y * error[1]

            - self.kd_y * velocity[1]

        )


        az = (

            self.kp_z * error[2]

            - self.kd_z * velocity[2]

            + self.gravity

        )


        # ====================================================
        # LIMIT LATERAL ACCELERATION
        # ====================================================

        max_lateral_acceleration = (
            self.gravity
            * np.tan(self.max_pitch)
        )


        ax = np.clip(
            ax,
            -max_lateral_acceleration,
            max_lateral_acceleration
        )


        ay = np.clip(
            ay,
            -max_lateral_acceleration,
            max_lateral_acceleration
        )


        # ====================================================
        # DESIRED FORCE
        # ====================================================

        desired_force = self.mass * np.array([

            ax,
            ay,
            az

        ])


        force_magnitude = np.linalg.norm(
            desired_force
        )


        if force_magnitude < 1e-9:

            force_magnitude = (
                self.mass
                * self.gravity
            )


        # ====================================================
        # DESIRED BODY Z AXIS
        # ====================================================

        body_z = (
            desired_force
            / force_magnitude
        )


        # ====================================================
        # DESIRED YAW AXIS
        # ====================================================

        yaw_reference = np.array([

            np.cos(yaw),
            np.sin(yaw),
            0.0

        ])


        # ====================================================
        # CONSTRUCT DESIRED BODY X AXIS
        # ====================================================

        body_y = np.cross(
            body_z,
            yaw_reference
        )


        body_y_norm = np.linalg.norm(
            body_y
        )


        if body_y_norm < 1e-9:

            body_y = np.array([
                -np.sin(yaw),
                np.cos(yaw),
                0.0
            ])

        else:

            body_y = (
                body_y
                / body_y_norm
            )


        # ====================================================
        # DESIRED BODY X AXIS
        # ====================================================

        body_x = np.cross(
            body_y,
            body_z
        )


        body_x = (
            body_x
            / np.linalg.norm(body_x)
        )


        # ====================================================
        # DESIRED ROTATION MATRIX
        # ====================================================

        R_desired = np.column_stack([

            body_x,
            body_y,
            body_z

        ])


        # ====================================================
        # DESIRED EULER ANGLES
        # ====================================================

        desired_phi, desired_theta, desired_psi = (

            rotation_to_euler(
                R_desired
            )

        )


        desired_phi = np.clip(
            desired_phi,
            -self.max_roll,
            self.max_roll
        )


        desired_theta = np.clip(
            desired_theta,
            -self.max_pitch,
            self.max_pitch
        )


        return (

            force_magnitude,

            desired_phi,

            desired_theta,

            desired_psi

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
        kd_yaw

    ):

        self.kp = np.array([

            kp_roll,
            kp_pitch,
            kp_yaw

        ])


        self.kd = np.array([

            kd_roll,
            kd_pitch,
            kd_yaw

        ])


    def update(

        self,

        desired_angles,

        actual_angles,

        angular_rates

    ):

        # ====================================================
        # ANGLE ERROR
        # ====================================================

        angle_error = (

            desired_angles
            - actual_angles

        )


        # Wrap yaw error to [-pi, pi]

        angle_error[2] = np.arctan2(

            np.sin(angle_error[2]),

            np.cos(angle_error[2])

        )


        # ====================================================
        # PD ATTITUDE CONTROL
        # ====================================================

        torque = (

            self.kp * angle_error

            - self.kd * angular_rates

        )


        return torque