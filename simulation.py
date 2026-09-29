"""
======================================================================
MorphoAqua - Stage 3A
In-Flight Morphing Dynamics
======================================================================

Stage 3A extends the validated Stage 2B 6-DOF simulation by introducing
a prescribed, smooth change in vehicle morphology during flight.

The vehicle continuously changes arm length while simultaneously
tracking the Stage 2B continuous 3-D trajectory.

Morphology model:
    - Compact configuration: 0.80 x nominal arm length
    - Extended configuration: 1.20 x nominal arm length
    - Smooth quintic morphing transitions
    - Rotational inertia is parameterized as proportional to arm_length^2

The changing arm length directly affects:
    1. Quadrotor motor mixing
    2. Available roll/pitch torque
    3. Actual motor-generated torque
    4. Rigid-body rotational dynamics through time-varying inertia

This is a numerical simulation only. The morphology and inertia
relationship is a parameterized model, not an experimental measurement.
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


# ======================================================================
# STAGE 3A CONFIGURATION
# ======================================================================

STAGE3A_SIMULATION_TIME = 20.0

TAKEOFF_TIME = 3.0

TRAJECTORY_START_TIME = 3.0

TRAJECTORY_DURATION = 12.0

LANDING_START_TIME = 15.0

LANDING_DURATION = 3.0


# ----------------------------------------------------------------------
# Morphing schedule
# ----------------------------------------------------------------------

MORPHING_START_TIME = 3.0

MORPHING_OUT_DURATION = 3.0

EXTENDED_HOLD_START_TIME = 6.0

EXTENDED_HOLD_END_TIME = 9.0

MORPHING_IN_DURATION = 3.0

COMPACT_HOLD_END_TIME = 15.0


COMPACT_ARM_RATIO = 0.80

EXTENDED_ARM_RATIO = 1.20


# ----------------------------------------------------------------------
# Continuous trajectory parameters
# ----------------------------------------------------------------------

TRAJECTORY_CENTER_Z = 2.0

TRAJECTORY_X_RADIUS = 1.5

TRAJECTORY_Y_RADIUS = 0.9

TRAJECTORY_Z_AMPLITUDE = 0.3


TRAJECTORY_ANGULAR_FREQUENCY = (
    2.0 * np.pi / TRAJECTORY_DURATION
)


TARGET_YAW = 0.0


# ----------------------------------------------------------------------
# Controller limits
# ----------------------------------------------------------------------

MAX_HORIZONTAL_ACCELERATION = 2.0

MAX_VERTICAL_ACCELERATION = 2.5


# ----------------------------------------------------------------------
# Results directory
# ----------------------------------------------------------------------

RESULTS_DIRECTORY = "results"

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True,
)


# ======================================================================
# TRAJECTORY DESCRIPTION
# ======================================================================

"""
The continuous trajectory is defined parametrically.

During the tracking phase:

    theta = 2*pi*u

    X = A * (1 - cos(theta))

    Y = B * sin(theta) * (1 - cos(theta))

    Z = Z0 + C * (1 - cos(theta))

where:

    A = TRAJECTORY_X_RADIUS
    B = TRAJECTORY_Y_RADIUS
    C = TRAJECTORY_Z_AMPLITUDE

The trajectory starts and ends at:

    (0, 0, 2)

with zero velocity at both ends.

This gives smooth transition into and out of the
continuous trajectory phase.
"""


# ======================================================================
# THRUST / RPM CONVERSION
# ======================================================================

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


# ======================================================================
# QUADROTOR MOTOR MIXER
# ======================================================================

def calculate_motor_thrusts(
    total_thrust,
    roll_torque,
    pitch_torque,
    yaw_torque,
    arm_length,
):

    arm = (
        arm_length
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
    arm_length,
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
        arm_length,
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
        arm_length,
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
            arm_length,
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


# ======================================================================
# ACTUAL MOTOR TORQUES
# ======================================================================

def calculate_actual_torques(
    motor_thrusts,
    arm_length,
):

    arm = (
        arm_length
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


# ======================================================================
# QUINTIC SMOOTHSTEP
# ======================================================================

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


# ======================================================================
# MORPHOLOGY MODEL
# ======================================================================

def morphology_profile(t):
    """
    Return the normalized morphology state.

    morphing_state:
        0.0 -> compact
        1.0 -> extended

    The transitions use the same quintic smoothstep profile used by
    the flight trajectory, giving zero first and second derivatives
    at the endpoints.
    """

    if t < MORPHING_START_TIME:

        return 0.0

    if t < (
        MORPHING_START_TIME
        + MORPHING_OUT_DURATION
    ):

        u = (
            t - MORPHING_START_TIME
        ) / MORPHING_OUT_DURATION

        s, _, _ = smoothstep_profile(u)

        return float(s)

    if t < EXTENDED_HOLD_END_TIME:

        return 1.0

    if t < (
        EXTENDED_HOLD_END_TIME
        + MORPHING_IN_DURATION
    ):

        u = (
            t - EXTENDED_HOLD_END_TIME
        ) / MORPHING_IN_DURATION

        s, _, _ = smoothstep_profile(u)

        return float(1.0 - s)

    return 0.0


def morphology_parameters(t):
    """
    Return current arm length and parameterized inertia.

    Arm length changes linearly with the normalized morphology state.

    The inertia model assumes the dominant morphology-dependent
    contribution scales with the square of the radial distance:

        I(t) = I_nominal * (L(t) / L_nominal)^2

    This is a parameterized numerical model, not a measured
    mass-property model.
    """

    morphing_state = morphology_profile(t)

    arm_ratio = (
        COMPACT_ARM_RATIO
        + morphing_state
        * (
            EXTENDED_ARM_RATIO
            - COMPACT_ARM_RATIO
        )
    )

    current_arm_length = (
        ARM_LENGTH
        * arm_ratio
    )

    inertia_scale = (
        arm_ratio**2
    )

    current_inertia = (
        INERTIA
        * inertia_scale
    )

    return (
        morphing_state,
        current_arm_length,
        current_inertia,
        inertia_scale,
    )


# ======================================================================
# CONTINUOUS 3-D TRAJECTORY
# ======================================================================

def continuous_trajectory(t):

    omega = (
        TRAJECTORY_ANGULAR_FREQUENCY
    )

    # --------------------------------------------------------------
    # PHASE 1: TAKEOFF
    # --------------------------------------------------------------

    if t < TAKEOFF_TIME:

        u = (
            t
            / TAKEOFF_TIME
        )

        s, ds, d2s = (
            smoothstep_profile(u)
        )

        position = np.array(
            [
                0.0,
                0.0,
                TRAJECTORY_CENTER_Z * s,
            ]
        )

        velocity = np.array(
            [
                0.0,
                0.0,
                (
                    TRAJECTORY_CENTER_Z
                    * ds
                    / TAKEOFF_TIME
                ),
            ]
        )

        acceleration = np.array(
            [
                0.0,
                0.0,
                (
                    TRAJECTORY_CENTER_Z
                    * d2s
                    / TAKEOFF_TIME**2
                ),
            ]
        )

        return (
            position,
            velocity,
            acceleration,
        )


    # --------------------------------------------------------------
    # PHASE 2: CONTINUOUS 3-D LOOP
    # --------------------------------------------------------------

    if t < LANDING_START_TIME:

        local_time = (
            t
            - TRAJECTORY_START_TIME
        )

        u = (
            local_time
            / TRAJECTORY_DURATION
        )

        u = np.clip(
            u,
            0.0,
            1.0,
        )

        theta = (
            2.0
            * np.pi
            * u
        )

        sin_theta = np.sin(theta)

        cos_theta = np.cos(theta)

        sin_2theta = np.sin(
            2.0 * theta
        )

        cos_2theta = np.cos(
            2.0 * theta
        )

        # ----------------------------------------------------------
        # Position
        # ----------------------------------------------------------

        x = (
            TRAJECTORY_X_RADIUS
            * (
                1.0
                - cos_theta
            )
        )

        y = (
            TRAJECTORY_Y_RADIUS
            * sin_theta
            * (
                1.0
                - cos_theta
            )
        )

        z = (
            TRAJECTORY_CENTER_Z
            + TRAJECTORY_Z_AMPLITUDE
            * (
                1.0
                - cos_theta
            )
        )

        position = np.array(
            [
                x,
                y,
                z,
            ]
        )


        # ----------------------------------------------------------
        # First derivatives with respect to theta
        # ----------------------------------------------------------

        dx_dtheta = (
            TRAJECTORY_X_RADIUS
            * sin_theta
        )

        dy_dtheta = (
            TRAJECTORY_Y_RADIUS
            * (
                cos_theta
                - cos_2theta
            )
        )

        dz_dtheta = (
            TRAJECTORY_Z_AMPLITUDE
            * sin_theta
        )


        # ----------------------------------------------------------
        # Second derivatives with respect to theta
        # ----------------------------------------------------------

        d2x_dtheta2 = (
            TRAJECTORY_X_RADIUS
            * cos_theta
        )

        d2y_dtheta2 = (
            TRAJECTORY_Y_RADIUS
            * (
                -sin_theta
                + 2.0 * sin_2theta
            )
        )

        d2z_dtheta2 = (
            TRAJECTORY_Z_AMPLITUDE
            * cos_theta
        )


        # ----------------------------------------------------------
        # Convert theta derivatives to time derivatives
        # ----------------------------------------------------------

        velocity = np.array(
            [
                dx_dtheta * omega,
                dy_dtheta * omega,
                dz_dtheta * omega,
            ]
        )

        acceleration = np.array(
            [
                d2x_dtheta2 * omega**2,
                d2y_dtheta2 * omega**2,
                d2z_dtheta2 * omega**2,
            ]
        )

        return (
            position,
            velocity,
            acceleration,
        )


    # --------------------------------------------------------------
    # PHASE 3: LANDING
    # --------------------------------------------------------------

    local_time = (
        t
        - LANDING_START_TIME
    )

    u = (
        local_time
        / LANDING_DURATION
    )

    s, ds, d2s = (
        smoothstep_profile(u)
    )

    position = np.array(
        [
            0.0,
            0.0,
            TRAJECTORY_CENTER_Z
            * (1.0 - s),
        ]
    )

    velocity = np.array(
        [
            0.0,
            0.0,
            -(
                TRAJECTORY_CENTER_Z
                * ds
                / LANDING_DURATION
            ),
        ]
    )

    acceleration = np.array(
        [
            0.0,
            0.0,
            -(
                TRAJECTORY_CENTER_Z
                * d2s
                / LANDING_DURATION**2
            ),
        ]
    )

    return (
        position,
        velocity,
        acceleration,
    )


# ======================================================================
# SIMULATION
# ======================================================================

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

    # The attitude controller is given the maximum torque
    # capability over the full morphology range. The dynamic
    # mixer subsequently enforces the actual torque capability
    # at the current arm length.
    maximum_arm_length = (
        ARM_LENGTH
        * EXTENDED_ARM_RATIO
    )

    maximum_arm = (
        maximum_arm_length
        / np.sqrt(2.0)
    )

    maximum_roll_torque = (
        2.0
        * maximum_arm
        * maximum_thrust
    )

    maximum_pitch_torque = (
        2.0
        * maximum_arm
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
        STAGE3A_SIMULATION_TIME
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

    morphology_history = np.zeros(
        steps
    )

    arm_length_history = np.zeros(
        steps
    )

    inertia_scale_history = np.zeros(
        steps
    )

    for i, t in enumerate(time):

        (
            target_position,
            target_velocity,
            target_acceleration,
        ) = continuous_trajectory(t)

        (
            morphing_state,
            current_arm_length,
            current_inertia,
            inertia_scale,
        ) = morphology_parameters(t)

        target_position_history[i] = (
            target_position
        )

        target_velocity_history[i] = (
            target_velocity
        )

        target_acceleration_history[i] = (
            target_acceleration
        )

        morphology_history[i] = (
            morphing_state
        )

        arm_length_history[i] = (
            current_arm_length
        )

        inertia_scale_history[i] = (
            inertia_scale
        )

        # --------------------------------------------------------------
        # Position controller
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Attitude controller
        # --------------------------------------------------------------

        torque_command = (
            attitude_controller.update(

                desired_angles,
                angles,
                angular_rates,
            )
        )

        # --------------------------------------------------------------
        # Position errors
        # --------------------------------------------------------------

        position_error = (
            target_position
            - position
        )

        velocity_error = (
            target_velocity
            - velocity
        )

        # --------------------------------------------------------------
        # Vertical control
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Current attitude
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Horizontal control
        # --------------------------------------------------------------

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

        # Combine vertical and horizontal force requirements.

        required_total_thrust = np.sqrt(

            required_total_thrust**2
            + horizontal_force**2
        )

        required_total_thrust = np.clip(

            required_total_thrust,

            0.0,

            4.0 * maximum_thrust,
        )

        # --------------------------------------------------------------
        # Dynamic motor mixing
        # --------------------------------------------------------------

        commanded_motor_thrusts = (
            motor_mixer(

                required_total_thrust,

                torque_command[0],
                torque_command[1],
                torque_command[2],

                current_arm_length,
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

        # --------------------------------------------------------------
        # Motor dynamics
        # --------------------------------------------------------------

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
                actual_motor_thrusts,
                current_arm_length,
            )
        )

        # --------------------------------------------------------------
        # Translational dynamics
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Ground constraint
        # --------------------------------------------------------------

        if position[2] <= GROUND_ALTITUDE:

            position[2] = (
                GROUND_ALTITUDE
            )

            if velocity[2] < 0.0:

                velocity[2] = 0.0

        # --------------------------------------------------------------
        # Rotational dynamics with time-varying inertia
        # --------------------------------------------------------------

        angular_momentum = (
            current_inertia
            @ angular_rates
        )

        angular_acceleration = np.linalg.solve(

            current_inertia,

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

        # --------------------------------------------------------------
        # Euler-angle kinematics
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Store results
        # --------------------------------------------------------------

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

        morphology_history,

        arm_length_history,

        inertia_scale_history,
    )


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    position,
    velocity,
    angles,
    target_position,
    morphology,
    arm_length,
    inertia_scale,
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

        "Maximum 3-D tracking error (m)":
            np.max(error_magnitude),

        "RMS 3-D tracking error (m)":
            np.sqrt(
                np.mean(
                    error_magnitude**2
                )
            ),

        "Maximum horizontal tracking error (m)":
            np.max(horizontal_error),

        "RMS horizontal tracking error (m)":
            np.sqrt(
                np.mean(
                    horizontal_error**2
                )
            ),

        "Maximum X tracking error (m)":
            np.max(
                np.abs(
                    position_error[:, 0]
                )
            ),

        "Maximum Y tracking error (m)":
            np.max(
                np.abs(
                    position_error[:, 1]
                )
            ),

        "Maximum Z tracking error (m)":
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
                np.abs(
                    roll_deg
                )
            ),

        "Maximum pitch (deg)":
            np.max(
                np.abs(
                    pitch_deg
                )
            ),

        "Maximum yaw (deg)":
            np.max(
                np.abs(
                    yaw_deg
                )
            ),

        "Minimum arm length (m)":
            np.min(arm_length),

        "Maximum arm length (m)":
            np.max(arm_length),

        "Minimum inertia scale":
            np.min(inertia_scale),

        "Maximum inertia scale":
            np.max(inertia_scale),

        "Final morphology state":
            morphology[-1],

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


# ======================================================================
# POSITION RESULTS
# ======================================================================

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
        "MorphoAqua - Stage 3A "
        "In-Flight Morphing + 3-D Tracking"
    )


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_3A_position_tracking.png",
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.show()

    plt.close(fig)


# ======================================================================
# 3-D TRAJECTORY PLOT
# ======================================================================

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
        [0.0],
        [0.0],
        [0.0],
        s=60,
        label="Start / Landing",
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
        "MorphoAqua - Stage 3A "
        "In-Flight Morphing Trajectory"
    )


    ax.legend()

    ax.grid(True)


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_3A_3D_trajectory.png",
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.show()

    plt.close(fig)


# ======================================================================
# CONTROL RESPONSE
# ======================================================================

def plot_control_results(
    time,
    velocity,
    angles,
    thrust,
    torque,
    rpm,
    desired_angles,
    morphology,
    arm_length,
    inertia_scale,
):

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(13, 19),
        sharex=True,
    )

    # --------------------------------------------------------------
    # Velocity
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Attitude
    # --------------------------------------------------------------

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

    axes[1].plot(
        time,
        np.rad2deg(
            angles[:, 2]
        ),
        label="Yaw",
    )

    axes[1].set_ylabel(
        "Angle (deg)"
    )

    axes[1].legend()
    axes[1].grid(True)

    # --------------------------------------------------------------
    # Thrust
    # --------------------------------------------------------------

    axes[2].plot(
        time,
        thrust,
        label="Total thrust",
    )

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

    # --------------------------------------------------------------
    # Torque
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Motor RPM
    # --------------------------------------------------------------

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

    axes[4].legend()
    axes[4].grid(True)

    # --------------------------------------------------------------
    # Morphology
    # --------------------------------------------------------------

    ax_morph = axes[5]

    ax_morph.plot(
        time,
        arm_length,
        label="Arm length (m)",
    )

    ax_morph.set_ylabel(
        "Arm length (m)"
    )

    ax_morph.set_xlabel(
        "Time (s)"
    )

    ax_morph.grid(True)

    ax_state = ax_morph.twinx()

    ax_state.plot(
        time,
        morphology,
        "--",
        label="Morphing state",
    )

    ax_state.set_ylabel(
        "Morphing state"
    )

    ax_morph.set_title(
        "Morphology transition and arm length"
    )

    lines_1, labels_1 = (
        ax_morph.get_legend_handles_labels()
    )

    lines_2, labels_2 = (
        ax_state.get_legend_handles_labels()
    )

    ax_morph.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )

    fig.suptitle(
        "MorphoAqua - Stage 3A "
        "Control, Actuator and Morphing Response"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_3A_control_morphing_response.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)

    print(
        "MORPHOAQUA - STAGE 3A"
    )

    print(
        "IN-FLIGHT MORPHING DYNAMICS"
    )

    print("=" * 70)

    print()

    print(
        "Mission:"
    )

    print(
        "P0 = (0.00, 0.00, 0.00)"
    )

    print(
        "Smooth takeoff to Z = 2.00 m"
    )

    print(
        "Continuous closed 3-D trajectory"
    )

    print(
        "Compact -> extended morphing"
    )

    print(
        "Extended -> compact morphing"
    )

    print(
        "Smooth landing to Z = 0.00 m"
    )

    print()

    print(
        "Morphology:"
    )

    print(
        f"Compact arm ratio: "
        f"{COMPACT_ARM_RATIO:.2f}"
    )

    print(
        f"Extended arm ratio: "
        f"{EXTENDED_ARM_RATIO:.2f}"
    )

    print(
        "Inertia scaling: "
        "proportional to arm-length ratio squared"
    )

    print()

    print(
        f"Simulation time: "
        f"{STAGE3A_SIMULATION_TIME:.1f} s"
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
        morphology,
        arm_length,
        inertia_scale,
    ) = results

    metrics = calculate_metrics(
        position,
        velocity,
        angles,
        target_position,
        morphology,
        arm_length,
        inertia_scale,
    )

    print()

    print(
        "STAGE 3A PERFORMANCE METRICS"
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
        "Trajectory: Continuous closed 3-D loop"
    )

    print(
        "Morphology: Prescribed smooth arm-length transition"
    )

    print(
        "Inertia: Time-varying parameterized model"
    )

    print(
        "Motor allocation: "
        "Dynamic torque-limited mixer"
    )

    print(
        "Rotational dynamics: "
        "Time-varying rigid-body inertia"
    )

    print()

    print(
        "Stage 3A simulation completed."
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_3A_position_tracking.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_3A_3D_trajectory.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_3A_control_morphing_response.png",
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
        morphology,
        arm_length,
        inertia_scale,
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()