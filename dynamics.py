"""
MorphoAqua - Stage 1
Basic quadrotor dynamics

State:

[x, y, z,
 vx, vy, vz,
 roll, pitch, yaw,
 p, q, r]
"""

import numpy as np

from robot_parameters import (
    MASS,
    GRAVITY,
    AIR_DENSITY,
    CD,
    REFERENCE_AREA,
)


def rotation_matrix(roll, pitch, yaw):
    """
    Body-to-world rotation matrix.
    """

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,    0, 1]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])

    Rx = np.array([
        [1, 0,  0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    return Rz @ Ry @ Rx


def aerodynamic_drag(velocity):
    """
    Quadratic aerodynamic drag.

    Fd = -0.5 * rho * Cd * A * |v| * v
    """

    speed = np.linalg.norm(velocity)

    if speed < 1e-8:
        return np.zeros(3)

    return (
        -0.5
        * AIR_DENSITY
        * CD
        * REFERENCE_AREA
        * speed
        * velocity
    )


def translational_acceleration(
    velocity,
    roll,
    pitch,
    yaw,
    total_thrust,
):
    """
    Calculate world-frame translational acceleration.
    """

    R = rotation_matrix(roll, pitch, yaw)

    # Thrust acts along the +Z body axis
    thrust_body = np.array([
        0.0,
        0.0,
        total_thrust
    ])

    thrust_world = R @ thrust_body

    gravity_force = np.array([
        0.0,
        0.0,
        -MASS * GRAVITY
    ])

    drag_force = aerodynamic_drag(velocity)

    total_force = (
        thrust_world
        + gravity_force
        + drag_force
    )

    acceleration = total_force / MASS

    return acceleration


def rotational_acceleration(
    angular_velocity,
    torque,
    inertia,
):
    """
    Rigid-body rotational dynamics:

    I * omega_dot =
        torque - omega x (I omega)
    """

    omega = np.asarray(angular_velocity)

    angular_momentum = inertia @ omega

    gyroscopic_term = np.cross(
        omega,
        angular_momentum
    )

    angular_acceleration = np.linalg.solve(
        inertia,
        torque - gyroscopic_term
    )

    return angular_acceleration