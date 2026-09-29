"""
======================================================================
MorphoAqua - Stage 3B
Morphology-Aware Continuous 3-D Trajectory Tracking
======================================================================

Stage 3B extends the validated Stage 3A simulation.

Stage 3A:
    - Continuous 3-D trajectory tracking
    - Compact -> extended morphing
    - Extended -> compact morphing
    - Time-varying arm length
    - Time-varying inertia
    - Dynamic motor mixing

Stage 3B adds morphology-aware attitude control:

    1. Inertia-dependent torque scaling
    2. Time-varying inertia derivative dI/dt
    3. Dynamic compensation for:
           omega x (I omega)
           dI/dt omega
    4. Dedicated tracking metrics during morphing
    5. Morphology-rate and compensation logging

The translational trajectory remains the same as Stage 3A so that
Stage 3A and Stage 3B can be compared directly.

This is a numerical simulation only.

The morphology and inertia relationships are parameterized models,
not experimentally measured vehicle properties.
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
# STAGE 3B CONFIGURATION
# ======================================================================

STAGE3B_SIMULATION_TIME = 20.0

TAKEOFF_TIME = 3.0

TRAJECTORY_START_TIME = 3.0

TRAJECTORY_DURATION = 12.0

LANDING_START_TIME = 15.0

LANDING_DURATION = 3.0


# ======================================================================
# MORPHING SCHEDULE
# ======================================================================

MORPHING_START_TIME = 3.0

MORPHING_OUT_DURATION = 3.0

EXTENDED_HOLD_START_TIME = 6.0

EXTENDED_HOLD_END_TIME = 9.0

MORPHING_IN_DURATION = 3.0

COMPACT_HOLD_END_TIME = 15.0


COMPACT_ARM_RATIO = 0.80

EXTENDED_ARM_RATIO = 1.20


# ======================================================================
# CONTINUOUS TRAJECTORY
# ======================================================================

TRAJECTORY_CENTER_Z = 2.0

TRAJECTORY_X_RADIUS = 1.5

TRAJECTORY_Y_RADIUS = 0.9

TRAJECTORY_Z_AMPLITUDE = 0.3

TRAJECTORY_ANGULAR_FREQUENCY = (
    2.0 * np.pi / TRAJECTORY_DURATION
)

TARGET_YAW = 0.0


# ======================================================================
# CONTROL LIMITS
# ======================================================================

MAX_HORIZONTAL_ACCELERATION = 2.0

MAX_VERTICAL_ACCELERATION = 2.5


# ======================================================================
# RESULTS
# ======================================================================

RESULTS_DIRECTORY = "results"

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True,
)


# ======================================================================
# THRUST / RPM
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
# MOTOR MIXER
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
    Return:

        state
        state_rate
        state_acceleration

    State:
        0.0 = compact
        1.0 = extended

    Quintic smoothstep is used so that the morphology transition
    begins and ends with zero velocity and zero acceleration.
    """

    if t < MORPHING_START_TIME:

        return 0.0, 0.0, 0.0


    # --------------------------------------------------------------
    # Compact -> Extended
    # --------------------------------------------------------------

    morph_out_end = (
        MORPHING_START_TIME
        + MORPHING_OUT_DURATION
    )

    if t < morph_out_end:

        u = (
            t
            - MORPHING_START_TIME
        ) / MORPHING_OUT_DURATION

        s, ds, d2s = (
            smoothstep_profile(u)
        )

        state = s

        state_rate = (
            ds
            / MORPHING_OUT_DURATION
        )

        state_acceleration = (
            d2s
            / MORPHING_OUT_DURATION**2
        )

        return (
            float(state),
            float(state_rate),
            float(state_acceleration),
        )


    # --------------------------------------------------------------
    # Extended hold
    # --------------------------------------------------------------

    if t < EXTENDED_HOLD_END_TIME:

        return 1.0, 0.0, 0.0


    # --------------------------------------------------------------
    # Extended -> Compact
    # --------------------------------------------------------------

    morph_in_end = (
        EXTENDED_HOLD_END_TIME
        + MORPHING_IN_DURATION
    )

    if t < morph_in_end:

        u = (
            t
            - EXTENDED_HOLD_END_TIME
        ) / MORPHING_IN_DURATION

        s, ds, d2s = (
            smoothstep_profile(u)
        )

        state = (
            1.0 - s
        )

        state_rate = (
            -ds
            / MORPHING_IN_DURATION
        )

        state_acceleration = (
            -d2s
            / MORPHING_IN_DURATION**2
        )

        return (
            float(state),
            float(state_rate),
            float(state_acceleration),
        )


    return 0.0, 0.0, 0.0


def morphology_parameters(t):
    """
    Calculate morphology-dependent physical parameters.

    Arm length:

        L(t) = L_nominal * arm_ratio

    Inertia model:

        I(t) = I_nominal * arm_ratio^2

    Therefore:

        dI/dt = 2 * I_nominal * arm_ratio * d(arm_ratio)/dt

    This is a parameterized numerical model.
    """

    (
        morphology_state,
        morphology_rate,
        morphology_acceleration,
    ) = morphology_profile(t)


    arm_ratio = (
        COMPACT_ARM_RATIO
        + morphology_state
        * (
            EXTENDED_ARM_RATIO
            - COMPACT_ARM_RATIO
        )
    )


    arm_ratio_rate = (
        morphology_rate
        * (
            EXTENDED_ARM_RATIO
            - COMPACT_ARM_RATIO
        )
    )


    arm_ratio_acceleration = (
        morphology_acceleration
        * (
            EXTENDED_ARM_RATIO
            - COMPACT_ARM_RATIO
        )
    )


    current_arm_length = (
        ARM_LENGTH
        * arm_ratio
    )


    arm_length_rate = (
        ARM_LENGTH
        * arm_ratio_rate
    )


    inertia_scale = (
        arm_ratio**2
    )


    current_inertia = (
        INERTIA
        * inertia_scale
    )


    inertia_rate = (
        INERTIA
        * (
            2.0
            * arm_ratio
            * arm_ratio_rate
        )
    )


    return (
        morphology_state,
        morphology_rate,
        morphology_acceleration,
        current_arm_length,
        arm_length_rate,
        current_inertia,
        inertia_rate,
        inertia_scale,
        arm_ratio_acceleration,
    )


# ======================================================================
# CONTINUOUS 3-D TRAJECTORY
# ======================================================================

def continuous_trajectory(t):

    omega = (
        TRAJECTORY_ANGULAR_FREQUENCY
    )


    # --------------------------------------------------------------
    # TAKEOFF
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
    # CONTINUOUS 3-D LOOP
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


        # Position

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


        # First derivatives

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


        # Second derivatives

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
    # LANDING
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
# MORPHOLOGY-AWARE ATTITUDE CONTROL
# ======================================================================

def morphology_aware_attitude_control(
    attitude_controller,
    desired_angles,
    angles,
    angular_rates,
    current_inertia,
    inertia_rate,
    inertia_scale,
):
    """
    Stage 3B morphology-aware attitude controller.

    The Stage 3A attitude controller provides the base PD torque.

    Stage 3B introduces:

        1. Inertia scaling

           tau_feedback = inertia_scale * tau_PD

        2. Dynamic compensation

           tau_comp =
               omega x (I omega)
               + dI/dt omega

    The compensation corresponds to the additional terms in:

        I * domega/dt
        + dI/dt * omega
        + omega x (I omega)
        = tau

    This allows the controller to explicitly account for
    morphology-dependent rotational dynamics.
    """

    base_torque = (
        attitude_controller.update(
            desired_angles,
            angles,
            angular_rates,
        )
    )


    # --------------------------------------------------------------
    # Inertia-aware feedback scaling
    # --------------------------------------------------------------

    adaptive_torque = (
        inertia_scale
        * base_torque
    )


    # --------------------------------------------------------------
    # Dynamic rotational compensation
    # --------------------------------------------------------------

    angular_momentum = (
        current_inertia
        @ angular_rates
    )


    gyroscopic_term = np.cross(
        angular_rates,
        angular_momentum,
    )


    inertia_rate_term = (
        inertia_rate
        @ angular_rates
    )


    dynamic_compensation = (
        gyroscopic_term
        + inertia_rate_term
    )


    commanded_torque = (
        adaptive_torque
        + dynamic_compensation
    )


    return (
        commanded_torque,
        base_torque,
        dynamic_compensation,
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


    # --------------------------------------------------------------
    # Position controller
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # Maximum motor / torque capability
    # --------------------------------------------------------------

    maximum_thrust = (
        maximum_motor_thrust()
    )


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


    # --------------------------------------------------------------
    # Attitude controller
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # Time
    # --------------------------------------------------------------

    steps = int(
        STAGE3B_SIMULATION_TIME
        / DT
    )

    time = (
        np.arange(steps)
        * DT
    )


    # --------------------------------------------------------------
    # History arrays
    # --------------------------------------------------------------

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

    base_torque_history = np.zeros(
        (steps, 3)
    )

    dynamic_compensation_history = np.zeros(
        (steps, 3)
    )


    rpm_history = np.zeros(
        (steps, 4)
    )


    morphology_history = np.zeros(
        steps
    )

    morphology_rate_history = np.zeros(
        steps
    )

    morphology_acceleration_history = np.zeros(
        steps
    )


    arm_length_history = np.zeros(
        steps
    )

    arm_length_rate_history = np.zeros(
        steps
    )


    inertia_scale_history = np.zeros(
        steps
    )


    inertia_rate_history = np.zeros(
        (steps, 3, 3)
    )


    # ==================================================================
    # MAIN SIMULATION LOOP
    # ==================================================================

    for i, t in enumerate(time):


        # --------------------------------------------------------------
        # Target trajectory
        # --------------------------------------------------------------

        (
            target_position,
            target_velocity,
            target_acceleration,
        ) = continuous_trajectory(t)


        # --------------------------------------------------------------
        # Morphology
        # --------------------------------------------------------------

        (
            morphology_state,
            morphology_rate,
            morphology_acceleration,
            current_arm_length,
            arm_length_rate,
            current_inertia,
            inertia_rate,
            inertia_scale,
            _arm_ratio_acceleration,
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
            morphology_state
        )

        morphology_rate_history[i] = (
            morphology_rate
        )

        morphology_acceleration_history[i] = (
            morphology_acceleration
        )


        arm_length_history[i] = (
            current_arm_length
        )

        arm_length_rate_history[i] = (
            arm_length_rate
        )

        inertia_scale_history[i] = (
            inertia_scale
        )

        inertia_rate_history[i] = (
            inertia_rate
        )


        # --------------------------------------------------------------
        # Position control
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
        # Morphology-aware attitude control
        # --------------------------------------------------------------

        (
            torque_command,
            base_torque,
            dynamic_compensation,
        ) = morphology_aware_attitude_control(

            attitude_controller,

            desired_angles,
            angles,
            angular_rates,

            current_inertia,
            inertia_rate,
            inertia_scale,
        )


        base_torque_history[i] = (
            base_torque
        )

        dynamic_compensation_history[i] = (
            dynamic_compensation
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

        actual_rpms = np.zeros(
            4
        )


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
        # Time-varying rotational dynamics
        #
        # I * omega_dot
        # + I_dot * omega
        # + omega x (I omega)
        # = tau
        # --------------------------------------------------------------

        angular_momentum = (
            current_inertia
            @ angular_rates
        )


        angular_acceleration = np.linalg.solve(

            current_inertia,

            actual_torque

            - inertia_rate
            @ angular_rates

            - np.cross(
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

        morphology_rate_history,

        morphology_acceleration_history,

        arm_length_history,

        arm_length_rate_history,

        inertia_scale_history,

        inertia_rate_history,

        base_torque_history,

        dynamic_compensation_history,
    )


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    time,
    position,
    velocity,
    angles,
    target_position,
    morphology,
    morphology_rate,
    arm_length,
    arm_length_rate,
    inertia_scale,
    base_torque,
    dynamic_compensation,
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


    # --------------------------------------------------------------
    # Morphing transition mask
    # --------------------------------------------------------------

    morphing_mask = (
        (
            (time >= MORPHING_START_TIME)
            & (
                time
                <= MORPHING_START_TIME
                + MORPHING_OUT_DURATION
            )
        )
        |
        (
            (time >= EXTENDED_HOLD_END_TIME)
            & (
                time
                <= EXTENDED_HOLD_END_TIME
                + MORPHING_IN_DURATION
            )
        )
    )


    tracking_mask = (
        (time >= TRAJECTORY_START_TIME)
        &
        (time <= LANDING_START_TIME)
    )


    if np.any(morphing_mask):

        morphing_errors = (
            error_magnitude[morphing_mask]
        )

        morphing_horizontal_errors = (
            horizontal_error[morphing_mask]
        )

    else:

        morphing_errors = np.array(
            [0.0]
        )

        morphing_horizontal_errors = np.array(
            [0.0]
        )


    if np.any(tracking_mask):

        tracking_errors = (
            error_magnitude[tracking_mask]
        )

    else:

        tracking_errors = np.array(
            [0.0]
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


    base_torque_magnitude = (
        np.linalg.norm(
            base_torque,
            axis=1,
        )
    )


    compensation_magnitude = (
        np.linalg.norm(
            dynamic_compensation,
            axis=1,
        )
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

        "Maximum tracking error during trajectory (m)":
            np.max(tracking_errors),

        "RMS tracking error during trajectory (m)":
            np.sqrt(
                np.mean(
                    tracking_errors**2
                )
            ),

        "Maximum 3-D error during morphing (m)":
            np.max(morphing_errors),

        "RMS 3-D error during morphing (m)":
            np.sqrt(
                np.mean(
                    morphing_errors**2
                )
            ),

        "Maximum horizontal error during morphing (m)":
            np.max(
                morphing_horizontal_errors
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

        "Maximum arm-length rate (m/s)":
            np.max(
                np.abs(
                    arm_length_rate
                )
            ),

        "Minimum inertia scale":
            np.min(inertia_scale),

        "Maximum inertia scale":
            np.max(inertia_scale),

        "Maximum morphology rate (1/s)":
            np.max(
                np.abs(
                    morphology_rate
                )
            ),

        "Maximum base torque (N m)":
            np.max(
                base_torque_magnitude
            ),

        "Maximum dynamic compensation (N m)":
            np.max(
                compensation_magnitude
            ),

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
# POSITION PLOT
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


    labels = [
        ("X", 0),
        ("Y", 1),
        ("Z", 2),
    ]


    for axis, (
        label,
        index,
    ) in zip(
        axes,
        labels,
    ):

        axis.plot(
            time,
            position[:, index],
            label=f"Actual {label}",
        )

        axis.plot(
            time,
            target[:, index],
            "--",
            label=f"Target {label}",
        )

        axis.set_ylabel(
            f"{label} Position (m)"
        )

        axis.grid(True)

        axis.legend()


    axes[-1].set_xlabel(
        "Time (s)"
    )


    fig.suptitle(
        "MorphoAqua - Stage 3B "
        "Morphology-Aware 3-D Tracking"
    )


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_3B_position_tracking.png",
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.show()

    plt.close(fig)


# ======================================================================
# 3-D TRAJECTORY
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
        "MorphoAqua - Stage 3B "
        "Morphology-Aware 3-D Trajectory"
    )


    ax.legend()

    ax.grid(True)


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_3B_3D_trajectory.png",
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.show()

    plt.close(fig)


# ======================================================================
# CONTROL / MORPHING PLOT
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
    morphology_rate,
    arm_length,
    arm_length_rate,
    inertia_scale,
    dynamic_compensation,
):

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(13, 22),
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

    axes[1].legend(
        ncol=3
    )

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
    # Actual torque
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

    axes[4].legend(
        ncol=4
    )

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


    ax_morph.grid(True)


    ax_morph_rate = (
        ax_morph.twinx()
    )


    ax_morph_rate.plot(
        time,
        morphology_rate,
        "--",
        label="Morphology rate",
    )


    ax_morph_rate.set_ylabel(
        "Morphology rate (1/s)"
    )


    lines_1, labels_1 = (
        ax_morph.get_legend_handles_labels()
    )

    lines_2, labels_2 = (
        ax_morph_rate.get_legend_handles_labels()
    )


    ax_morph.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )


    ax_morph.set_title(
        "Morphology transition"
    )


    # --------------------------------------------------------------
    # Inertia / dynamic compensation
    # --------------------------------------------------------------

    axes[6].plot(
        time,
        inertia_scale,
        label="Inertia scale",
    )


    compensation_magnitude = (
        np.linalg.norm(
            dynamic_compensation,
            axis=1,
        )
    )


    ax_comp = axes[6].twinx()


    ax_comp.plot(
        time,
        compensation_magnitude,
        "--",
        label="Dynamic compensation",
    )


    axes[6].set_ylabel(
        "Inertia scale"
    )


    ax_comp.set_ylabel(
        "Compensation torque (N m)"
    )


    axes[6].set_xlabel(
        "Time (s)"
    )


    lines_1, labels_1 = (
        axes[6].get_legend_handles_labels()
    )

    lines_2, labels_2 = (
        ax_comp.get_legend_handles_labels()
    )


    axes[6].legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )


    axes[6].set_title(
        "Time-varying inertia and dynamic compensation"
    )


    axes[6].grid(True)


    fig.suptitle(
        "MorphoAqua - Stage 3B "
        "Control, Actuation and Morphology-Aware Dynamics"
    )


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_3B_control_morphing_response.png",
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
        "MORPHOAQUA - STAGE 3B"
    )

    print(
        "MORPHOLOGY-AWARE CONTINUOUS 3-D TRAJECTORY TRACKING"
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
        "Stage 3B control:"
    )

    print(
        "Morphology-aware inertia scaling"
    )

    print(
        "Time-varying inertia compensation"
    )

    print(
        "Dynamic rotational compensation"
    )

    print()


    print(
        f"Simulation time: "
        f"{STAGE3B_SIMULATION_TIME:.1f} s"
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
        morphology_rate,
        morphology_acceleration,
        arm_length,
        arm_length_rate,
        inertia_scale,
        inertia_rate,
        base_torque,
        dynamic_compensation,
    ) = results


    metrics = calculate_metrics(

        time,

        position,

        velocity,

        angles,

        target_position,

        morphology,

        morphology_rate,

        arm_length,

        arm_length_rate,

        inertia_scale,

        base_torque,

        dynamic_compensation,
    )


    print()

    print(
        "STAGE 3B PERFORMANCE METRICS"
    )

    print(
        "-" * 70
    )


    for name, value in metrics.items():

        print(
            f"{name:<50}: "
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
        "Attitude controller: "
        "Morphology-aware PD"
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

    print(
        "Dynamic compensation: "
        "Gyroscopic + dI/dt compensation"
    )


    print()

    print(
        "Stage 3B simulation completed."
    )


    print()

    print(
        "Results saved to:"
    )


    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_3B_position_tracking.png",
        )
    )


    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_3B_3D_trajectory.png",
        )
    )


    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_3B_control_morphing_response.png",
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

        morphology_rate,

        arm_length,

        arm_length_rate,

        inertia_scale,

        dynamic_compensation,
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()