"""
============================================================
MorphoAqua - Stage 2A
3-D Waypoint Navigation
============================================================

Stage 2A extends the validated Stage 1C 6-DOF rigid-body
simulation to 3-D waypoint navigation.

Mission:
P0 = (0, 0, 0)
P1 = (0, 0, 2)
P2 = (2, 0, 2)
P3 = (2, 2, 2)
P4 = (0, 2, 2)
P5 = (0, 0, 2)
P6 = (0, 0, 0)

This is a numerical simulation only.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from robot_parameters import (
    MASS,
    GRAVITY,
    ARM_LENGTH,
    INERTIA,
    KF,
    KM,
    MAX_RPM,
    DT,
    POSITION_KP_X,
    POSITION_KP_Y,
    POSITION_KP_Z,
    POSITION_KD_X,
    POSITION_KD_Y,
    POSITION_KD_Z,
    ATTITUDE_KP_ROLL,
    ATTITUDE_KP_PITCH,
    ATTITUDE_KP_YAW,
    ATTITUDE_KD_ROLL,
    ATTITUDE_KD_PITCH,
    ATTITUDE_KD_YAW,
    MAX_ROLL,
    MAX_PITCH,
    GROUND_ALTITUDE,
)

from controller import (
    PositionController,
    AttitudeController,
    rotation_matrix,
)

from motor_model import (
    Motor,
    thrust_from_rpm,
)


# ============================================================
# CONFIGURATION
# ============================================================

STAGE2_SIMULATION_TIME = 20.0
WAYPOINT_TIME = 3.0
TARGET_YAW = 0.0

MAX_HORIZONTAL_ACCELERATION = 2.0
MAX_VERTICAL_ACCELERATION = 2.5

RESULTS_DIRECTORY = "results"

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True,
)


# ============================================================
# WAYPOINTS
# ============================================================

WAYPOINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)

WAYPOINT_NAMES = [
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
]


# ============================================================
# THRUST / RPM
# ============================================================

def thrust_to_rpm(thrust):

    if thrust <= 0.0:
        return 0.0

    omega = np.sqrt(
        thrust / KF
    )

    rpm = (
        omega
        * 60.0
        / (2.0 * np.pi)
    )

    return float(
        np.clip(
            rpm,
            0.0,
            MAX_RPM,
        )
    )


def maximum_motor_thrust():

    omega_max = (
        MAX_RPM
        * 2.0
        * np.pi
        / 60.0
    )

    return KF * omega_max**2


# ============================================================
# QUADROTOR MIXER
# ============================================================

def calculate_motor_thrusts(
    total_thrust,
    roll_torque,
    pitch_torque,
    yaw_torque,
):

    arm = (
        ARM_LENGTH
        / np.sqrt(2.0)
    )

    mixer = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [arm, -arm, -arm, arm],
            [-arm, -arm, arm, arm],
            [KM, -KM, KM, -KM],
        ]
    )

    desired = np.array(
        [
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        ]
    )

    return np.linalg.solve(
        mixer,
        desired,
    )


def motor_mixer(
    total_thrust,
    roll_torque,
    pitch_torque,
    yaw_torque,
):

    maximum_thrust = (
        maximum_motor_thrust()
    )

    total_thrust = float(
        np.clip(
            total_thrust,
            0.0,
            4.0 * maximum_thrust,
        )
    )

    torque = np.array(
        [
            roll_torque,
            pitch_torque,
            yaw_torque,
        ],
        dtype=float,
    )

    thrusts = calculate_motor_thrusts(
        total_thrust,
        torque[0],
        torque[1],
        torque[2],
    )

    if (
        np.all(thrusts >= 0.0)
        and np.all(thrusts <= maximum_thrust)
    ):
        return thrusts

    low = 0.0
    high = 1.0

    best = calculate_motor_thrusts(
        total_thrust,
        0.0,
        0.0,
        0.0,
    )

    for _ in range(30):

        scale = (
            low + high
        ) / 2.0

        candidate = calculate_motor_thrusts(
            total_thrust,
            torque[0] * scale,
            torque[1] * scale,
            torque[2] * scale,
        )

        if (
            np.all(candidate >= 0.0)
            and np.all(candidate <= maximum_thrust)
        ):

            best = candidate
            low = scale

        else:

            high = scale

    return np.clip(
        best,
        0.0,
        maximum_thrust,
    )


# ============================================================
# ACTUAL MOTOR TORQUES
# ============================================================

def calculate_actual_torques(
    motor_thrusts,
):

    arm = (
        ARM_LENGTH
        / np.sqrt(2.0)
    )

    roll_torque = (
        arm
        * (
            motor_thrusts[0]
            - motor_thrusts[1]
            - motor_thrusts[2]
            + motor_thrusts[3]
        )
    )

    pitch_torque = (
        arm
        * (
            -motor_thrusts[0]
            -motor_thrusts[1]
            + motor_thrusts[2]
            + motor_thrusts[3]
        )
    )

    yaw_torque = (
        KM
        * (
            motor_thrusts[0]
            - motor_thrusts[1]
            + motor_thrusts[2]
            - motor_thrusts[3]
        )
    )

    return np.array(
        [
            roll_torque,
            pitch_torque,
            yaw_torque,
        ]
    )


# ============================================================
# QUINTIC SMOOTHSTEP
# ============================================================

def smoothstep_profile(u):

    u = np.clip(
        u,
        0.0,
        1.0,
    )

    s = (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )

    ds = (
        30.0 * u**2
        - 60.0 * u**3
        + 30.0 * u**4
    )

    d2s = (
        60.0 * u
        - 180.0 * u**2
        + 120.0 * u**3
    )

    return s, ds, d2s


# ============================================================
# REFERENCE TRAJECTORY
# ============================================================

def reference_trajectory(t):

    number_of_segments = (
        len(WAYPOINTS) - 1
    )

    mission_duration = (
        number_of_segments
        * WAYPOINT_TIME
    )

    if t >= mission_duration:

        return (
            WAYPOINTS[-1].copy(),
            np.zeros(3),
            np.zeros(3),
            number_of_segments - 1,
        )

    segment_index = int(
        t / WAYPOINT_TIME
    )

    segment_index = min(
        segment_index,
        number_of_segments - 1,
    )

    segment_start_time = (
        segment_index
        * WAYPOINT_TIME
    )

    local_time = (
        t
        - segment_start_time
    )

    u = (
        local_time
        / WAYPOINT_TIME
    )

    start = WAYPOINTS[
        segment_index
    ]

    end = WAYPOINTS[
        segment_index + 1
    ]

    delta = end - start

    s, ds, d2s = (
        smoothstep_profile(u)
    )

    position = (
        start
        + delta * s
    )

    velocity = (
        delta
        * ds
        / WAYPOINT_TIME
    )

    acceleration = (
        delta
        * d2s
        / WAYPOINT_TIME**2
    )

    return (
        position,
        velocity,
        acceleration,
        segment_index,
    )


# ============================================================
# SIMULATION
# ============================================================

def run_simulation():

    position = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    velocity = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    angles = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    angular_rates = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    motors = [
        Motor(),
        Motor(),
        Motor(),
        Motor(),
    ]

    position_controller = PositionController(

        kp_x=POSITION_KP_X,
        kp_y=POSITION_KP_Y,
        kp_z=POSITION_KP_Z,

        kd_x=POSITION_KD_X,
        kd_y=POSITION_KD_Y,
        kd_z=POSITION_KD_Z,

        mass=MASS,
        gravity=GRAVITY,

        max_roll=MAX_ROLL,
        max_pitch=MAX_PITCH,

        max_horizontal_acceleration=(
            MAX_HORIZONTAL_ACCELERATION
        ),

        max_vertical_acceleration=(
            MAX_VERTICAL_ACCELERATION
        ),
    )

    maximum_thrust = (
        maximum_motor_thrust()
    )

    arm = (
        ARM_LENGTH
        / np.sqrt(2.0)
    )

    maximum_roll_torque = (
        2.0
        * arm
        * maximum_thrust
    )

    maximum_pitch_torque = (
        2.0
        * arm
        * maximum_thrust
    )

    maximum_yaw_torque = (
        2.0
        * KM
        * maximum_thrust
    )

    attitude_controller = AttitudeController(

        kp_roll=ATTITUDE_KP_ROLL,
        kp_pitch=ATTITUDE_KP_PITCH,
        kp_yaw=ATTITUDE_KP_YAW,

        kd_roll=ATTITUDE_KD_ROLL,
        kd_pitch=ATTITUDE_KD_PITCH,
        kd_yaw=ATTITUDE_KD_YAW,

        max_roll_torque=maximum_roll_torque,
        max_pitch_torque=maximum_pitch_torque,
        max_yaw_torque=maximum_yaw_torque,
    )

    steps = int(
        STAGE2_SIMULATION_TIME
        / DT
    )

    time = (
        np.arange(steps)
        * DT
    )

    position_history = np.zeros(
        (steps, 3)
    )

    velocity_history = np.zeros(
        (steps, 3)
    )

    angle_history = np.zeros(
        (steps, 3)
    )

    angular_rate_history = np.zeros(
        (steps, 3)
    )

    target_position_history = np.zeros(
        (steps, 3)
    )

    target_velocity_history = np.zeros(
        (steps, 3)
    )

    target_acceleration_history = np.zeros(
        (steps, 3)
    )

    desired_angle_history = np.zeros(
        (steps, 3)
    )

    thrust_history = np.zeros(
        steps
    )

    torque_history = np.zeros(
        (steps, 3)
    )

    rpm_history = np.zeros(
        (steps, 4)
    )

    for i, t in enumerate(time):

        (
            target_position,
            target_velocity,
            target_acceleration,
            _,
        ) = reference_trajectory(t)

        target_position_history[i] = (
            target_position
        )

        target_velocity_history[i] = (
            target_velocity
        )

        target_acceleration_history[i] = (
            target_acceleration
        )

        (
            _controller_thrust,
            desired_roll,
            desired_pitch,
            desired_yaw,
        ) = position_controller.update(

            target_position,
            position,
            velocity,

            TARGET_YAW,

            target_velocity,
            target_acceleration,
        )

        desired_angles = np.array(
            [
                desired_roll,
                desired_pitch,
                desired_yaw,
            ]
        )

        desired_angle_history[i] = (
            desired_angles
        )

        torque_command = (
            attitude_controller.update(

                desired_angles,
                angles,
                angular_rates,
            )
        )

        # ----------------------------------------------------
        # Position errors
        # ----------------------------------------------------

        position_error = (
            target_position
            - position
        )

        velocity_error = (
            target_velocity
            - velocity
        )

        # ----------------------------------------------------
        # Vertical control
        # ----------------------------------------------------

        vertical_acceleration = (

            target_acceleration[2]

            + POSITION_KP_Z
            * position_error[2]

            + POSITION_KD_Z
            * velocity_error[2]

        )

        vertical_acceleration = np.clip(

            vertical_acceleration,

            -MAX_VERTICAL_ACCELERATION,

            MAX_VERTICAL_ACCELERATION,

        )

        desired_vertical_force = (
            MASS
            * (
                GRAVITY
                + vertical_acceleration
            )
        )

        # ----------------------------------------------------
        # Current attitude
        # ----------------------------------------------------

        R = rotation_matrix(

            angles[0],
            angles[1],
            angles[2],

        )

        vertical_thrust_factor = (
            R[2, 2]
        )

        vertical_thrust_factor = max(
            vertical_thrust_factor,
            0.50,
        )

        required_total_thrust = (

            desired_vertical_force
            / vertical_thrust_factor

        )

        # ----------------------------------------------------
        # Horizontal acceleration
        # ----------------------------------------------------

        horizontal_acceleration = (

            target_acceleration[:2]

            + np.array(
                [
                    POSITION_KP_X,
                    POSITION_KP_Y,
                ]
            )
            * position_error[:2]

            + np.array(
                [
                    POSITION_KD_X,
                    POSITION_KD_Y,
                ]
            )
            * velocity_error[:2]

        )

        horizontal_magnitude = (
            np.linalg.norm(
                horizontal_acceleration
            )
        )

        if (
            horizontal_magnitude
            > MAX_HORIZONTAL_ACCELERATION
        ):

            horizontal_acceleration *= (

                MAX_HORIZONTAL_ACCELERATION
                / horizontal_magnitude

            )

        horizontal_force = (
            MASS
            * np.linalg.norm(
                horizontal_acceleration
            )
        )

        required_total_thrust = np.sqrt(

            required_total_thrust**2
            + horizontal_force**2

        )

        required_total_thrust = np.clip(

            required_total_thrust,

            0.0,

            4.0 * maximum_thrust,

        )

        # ----------------------------------------------------
        # Motor mixing
        # ----------------------------------------------------

        commanded_motor_thrusts = (
            motor_mixer(

                required_total_thrust,

                torque_command[0],
                torque_command[1],
                torque_command[2],

            )
        )

        commanded_rpms = np.array(

            [
                thrust_to_rpm(
                    thrust
                )

                for thrust
                in commanded_motor_thrusts
            ]

        )

        # ----------------------------------------------------
        # Motor dynamics
        # ----------------------------------------------------

        actual_rpms = np.zeros(4)

        for motor_index in range(4):

            actual_rpms[motor_index] = (

                motors[motor_index].update(

                    commanded_rpms[motor_index],

                    DT,

                )

            )

        actual_motor_thrusts = np.array(

            [
                thrust_from_rpm(
                    rpm
                )

                for rpm
                in actual_rpms
            ]

        )

        actual_total_thrust = (
            np.sum(
                actual_motor_thrusts
            )
        )

        actual_torque = (
            calculate_actual_torques(
                actual_motor_thrusts
            )
        )

        # ----------------------------------------------------
        # Translational dynamics
        # ----------------------------------------------------

        R = rotation_matrix(

            angles[0],
            angles[1],
            angles[2],

        )

        thrust_body = np.array(
            [
                0.0,
                0.0,
                actual_total_thrust,
            ]
        )

        thrust_world = (
            R
            @ thrust_body
        )

        gravity_force = np.array(
            [
                0.0,
                0.0,
                -MASS * GRAVITY,
            ]
        )

        total_force = (
            thrust_world
            + gravity_force
        )

        acceleration = (
            total_force
            / MASS
        )

        velocity += (
            acceleration
            * DT
        )

        position += (
            velocity
            * DT
        )

        # ----------------------------------------------------
        # Ground constraint
        # ----------------------------------------------------

        if position[2] <= GROUND_ALTITUDE:

            position[2] = (
                GROUND_ALTITUDE
            )

            if velocity[2] < 0.0:
                velocity[2] = 0.0

        # ----------------------------------------------------
        # Rotational dynamics
        # ----------------------------------------------------

        angular_momentum = (
            INERTIA
            @ angular_rates
        )

        angular_acceleration = np.linalg.solve(

            INERTIA,

            actual_torque
            -
            np.cross(
                angular_rates,
                angular_momentum,
            ),

        )

        angular_rates += (
            angular_acceleration
            * DT
        )

        # ----------------------------------------------------
        # Euler-angle kinematics
        # ----------------------------------------------------

        phi = angles[0]
        theta = angles[1]

        cos_theta = np.cos(
            theta
        )

        if abs(cos_theta) < 1e-5:
            cos_theta = 1e-5

        tan_theta = (
            np.sin(theta)
            / cos_theta
        )

        euler_rate_matrix = np.array(

            [
                [
                    1.0,
                    np.sin(phi)
                    * tan_theta,

                    np.cos(phi)
                    * tan_theta,
                ],

                [
                    0.0,
                    np.cos(phi),

                    -np.sin(phi),
                ],

                [
                    0.0,

                    np.sin(phi)
                    / cos_theta,

                    np.cos(phi)
                    / cos_theta,
                ],
            ]

        )

        angle_rates = (
            euler_rate_matrix
            @ angular_rates
        )

        angles += (
            angle_rates
            * DT
        )

        angles[2] = np.arctan2(

            np.sin(
                angles[2]
            ),

            np.cos(
                angles[2]
            ),

        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        position_history[i] = (
            position
        )

        velocity_history[i] = (
            velocity
        )

        angle_history[i] = (
            angles
        )

        angular_rate_history[i] = (
            angular_rates
        )

        thrust_history[i] = (
            actual_total_thrust
        )

        torque_history[i] = (
            actual_torque
        )

        rpm_history[i] = (
            actual_rpms
        )

    return (

        time,
        position_history,
        velocity_history,
        angle_history,
        angular_rate_history,
        target_position_history,
        target_velocity_history,
        target_acceleration_history,
        thrust_history,
        torque_history,
        rpm_history,
        desired_angle_history,

    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    position,
    velocity,
    angles,
    target_position,
):

    position_error = (
        target_position
        - position
    )

    error_magnitude = np.linalg.norm(
        position_error,
        axis=1,
    )

    horizontal_error = np.linalg.norm(
        position_error[:, :2],
        axis=1,
    )

    speed = np.linalg.norm(
        velocity,
        axis=1,
    )

    roll_deg = np.rad2deg(
        angles[:, 0]
    )

    pitch_deg = np.rad2deg(
        angles[:, 1]
    )

    yaw_deg = np.rad2deg(
        angles[:, 2]
    )

    final_position = (
        position[-1]
    )

    final_velocity = (
        velocity[-1]
    )

    final_angles = (
        angles[-1]
    )

    return {

        "Maximum 3-D position error (m)":
            np.max(error_magnitude),

        "RMS 3-D position error (m)":
            np.sqrt(
                np.mean(
                    error_magnitude**2
                )
            ),

        "Maximum horizontal error (m)":
            np.max(horizontal_error),

        "RMS horizontal error (m)":
            np.sqrt(
                np.mean(
                    horizontal_error**2
                )
            ),

        "Maximum X error (m)":
            np.max(
                np.abs(
                    position_error[:, 0]
                )
            ),

        "Maximum Y error (m)":
            np.max(
                np.abs(
                    position_error[:, 1]
                )
            ),

        "Maximum Z error (m)":
            np.max(
                np.abs(
                    position_error[:, 2]
                )
            ),

        "Maximum altitude (m)":
            np.max(
                position[:, 2]
            ),

        "Maximum speed (m/s)":
            np.max(speed),

        "RMS speed (m/s)":
            np.sqrt(
                np.mean(
                    speed**2
                )
            ),

        "Maximum roll (deg)":
            np.max(
                np.abs(roll_deg)
            ),

        "Maximum pitch (deg)":
            np.max(
                np.abs(pitch_deg)
            ),

        "Maximum yaw (deg)":
            np.max(
                np.abs(yaw_deg)
            ),

        "Final position error (m)":
            np.linalg.norm(
                position_error[-1]
            ),

        "Final X (m)":
            final_position[0],

        "Final Y (m)":
            final_position[1],

        "Final Z (m)":
            final_position[2],

        "Final Vx (m/s)":
            final_velocity[0],

        "Final Vy (m/s)":
            final_velocity[1],

        "Final Vz (m/s)":
            final_velocity[2],

        "Final roll (deg)":
            np.rad2deg(
                final_angles[0]
            ),

        "Final pitch (deg)":
            np.rad2deg(
                final_angles[1]
            ),

        "Final yaw (deg)":
            np.rad2deg(
                final_angles[2]
            ),

    }


# ============================================================
# POSITION RESULTS
# ============================================================

def plot_position_results(
    time,
    position,
    target,
):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 11),
        sharex=True,
    )

    axes[0].plot(
        time,
        position[:, 0],
        label="Actual X",
    )

    axes[0].plot(
        time,
        target[:, 0],
        "--",
        label="Target X",
    )

    axes[0].set_ylabel(
        "X Position (m)"
    )

    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        time,
        position[:, 1],
        label="Actual Y",
    )

    axes[1].plot(
        time,
        target[:, 1],
        "--",
        label="Target Y",
    )

    axes[1].set_ylabel(
        "Y Position (m)"
    )

    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        time,
        position[:, 2],
        label="Actual Z",
    )

    axes[2].plot(
        time,
        target[:, 2],
        "--",
        label="Target Z",
    )

    axes[2].set_ylabel(
        "Z Position (m)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(
        "MorphoAqua - Stage 2A "
        "3-D Waypoint Navigation"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_2A_3D_waypoint_navigation.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# 3-D TRAJECTORY
# ============================================================

def plot_3d_trajectory(
    position,
    target,
):

    fig = plt.figure(
        figsize=(12, 10)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        label="Actual trajectory",
    )

    ax.plot(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        "--",
        label="Target trajectory",
    )

    ax.scatter(
        WAYPOINTS[:, 0],
        WAYPOINTS[:, 1],
        WAYPOINTS[:, 2],
        s=50,
        label="Waypoints",
    )

    for i, waypoint in enumerate(
        WAYPOINTS
    ):

        ax.text(
            waypoint[0],
            waypoint[1],
            waypoint[2],
            WAYPOINT_NAMES[i],
        )

    ax.set_xlabel(
        "X (m)"
    )

    ax.set_ylabel(
        "Y (m)"
    )

    ax.set_zlabel(
        "Z (m)"
    )

    ax.set_title(
        "MorphoAqua - Stage 2A "
        "3-D Waypoint Trajectory"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_2A_3D_trajectory.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# CONTROL RESPONSE
# ============================================================

def plot_control_results(
    time,
    velocity,
    angles,
    thrust,
    torque,
    rpm,
    desired_angles,
):

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13, 16),
        sharex=True,
    )

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    axes[0].plot(
        time,
        velocity[:, 0],
        label="Vx",
    )

    axes[0].plot(
        time,
        velocity[:, 1],
        label="Vy",
    )

    axes[0].plot(
        time,
        velocity[:, 2],
        label="Vz",
    )

    axes[0].set_ylabel(
        "Velocity (m/s)"
    )

    axes[0].legend()
    axes[0].grid(True)

    # --------------------------------------------------------
    # Attitude
    # --------------------------------------------------------

    axes[1].plot(
        time,
        np.rad2deg(
            angles[:, 0]
        ),
        label="Actual Roll",
    )

    axes[1].plot(
        time,
        np.rad2deg(
            desired_angles[:, 0]
        ),
        "--",
        label="Desired Roll",
    )

    axes[1].plot(
        time,
        np.rad2deg(
            angles[:, 1]
        ),
        label="Actual Pitch",
    )

    axes[1].plot(
        time,
        np.rad2deg(
            desired_angles[:, 1]
        ),
        "--",
        label="Desired Pitch",
    )

    axes[1].set_ylabel(
        "Angle (deg)"
    )

    axes[1].legend()
    axes[1].grid(True)

    # --------------------------------------------------------
    # Thrust
    # --------------------------------------------------------

    axes[2].plot(
        time,
        thrust,
        label="Total thrust",
    )

    # IMPORTANT:
    # linestyle must be supplied by keyword.
    # Passing "--" as the second positional argument to
    # axhline() causes Matplotlib to interpret it as xmin.
    axes[2].axhline(
        MASS * GRAVITY,
        linestyle="--",
        label="Hover thrust",
    )

    axes[2].set_ylabel(
        "Thrust (N)"
    )

    axes[2].legend()
    axes[2].grid(True)

    # --------------------------------------------------------
    # Torque
    # --------------------------------------------------------

    axes[3].plot(
        time,
        torque[:, 0],
        label="Roll torque",
    )

    axes[3].plot(
        time,
        torque[:, 1],
        label="Pitch torque",
    )

    axes[3].plot(
        time,
        torque[:, 2],
        label="Yaw torque",
    )

    axes[3].set_ylabel(
        "Torque (N m)"
    )

    axes[3].legend()
    axes[3].grid(True)

    # --------------------------------------------------------
    # RPM
    # --------------------------------------------------------

    for motor_index in range(4):

        axes[4].plot(
            time,
            rpm[:, motor_index],
            label=(
                f"Motor {motor_index + 1}"
            ),
        )

    axes[4].set_ylabel(
        "RPM"
    )

    axes[4].set_xlabel(
        "Time (s)"
    )

    axes[4].legend()
    axes[4].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 2A "
        "Control and Actuator Response"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_2A_control_response.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "MORPHOAQUA - STAGE 2A"
    )

    print(
        "3-D WAYPOINT NAVIGATION"
    )

    print("=" * 70)

    print()

    print(
        "Mission waypoints:"
    )

    for name, waypoint in zip(
        WAYPOINT_NAMES,
        WAYPOINTS,
    ):

        print(
            f"{name} = "
            f"({waypoint[0]:.2f}, "
            f"{waypoint[1]:.2f}, "
            f"{waypoint[2]:.2f})"
        )

    print()

    print(
        f"Simulation time: "
        f"{STAGE2_SIMULATION_TIME:.1f} s"
    )

    print()

    results = run_simulation()

    (
        time,
        position,
        velocity,
        angles,
        angular_rates,
        target_position,
        target_velocity,
        target_acceleration,
        thrust,
        torque,
        rpm,
        desired_angles,
    ) = results

    metrics = calculate_metrics(
        position,
        velocity,
        angles,
        target_position,
    )

    print()

    print(
        "3-D TRAJECTORY PERFORMANCE METRICS"
    )

    print(
        "-" * 70
    )

    for name, value in metrics.items():

        print(
            f"{name:<45}: "
            f"{value:.6f}"
        )

    print(
        "-" * 70
    )

    print()

    print(
        "Controller configuration:"
    )

    print(
        "Position controller: "
        "World-frame PD + trajectory feedforward"
    )

    print(
        "Attitude controller: PD"
    )

    print(
        "Degrees of freedom: 6"
    )

    print(
        "Trajectory profile: Quintic smoothstep"
    )

    print(
        "Vertical control: "
        "Tilt-compensated thrust"
    )

    print(
        "Motor allocation: "
        "Torque-limited collective-preserving mixer"
    )

    print()

    print(
        "Stage 2A simulation completed."
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_2A_3D_waypoint_navigation.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_2A_3D_trajectory.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_2A_control_response.png",
        )
    )

    plot_position_results(
        time,
        position,
        target_position,
    )

    plot_3d_trajectory(
        position,
        target_position,
    )

    plot_control_results(
        time,
        velocity,
        angles,
        thrust,
        torque,
        rpm,
        desired_angles,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()