"""
======================================================================
MorphoAqua - Stage 4B
Fully Submerged Stabilization and Underwater Control
======================================================================

Stage 4B extends the validated Stage 4A air-water transition model.

Mission:
    0-4 s     Smooth descent from z = -0.60 m to z = -1.00 m
    4-8 s     Submerged stabilization at z = -1.00 m
    8-12 s    Smooth underwater translation to x = +1.00 m
    12-16 s   Smooth return to x = 0.00 m
    16-20 s   Final submerged stabilization

Underwater model:
    - Fully submerged immersion = 1
    - Buoyancy
    - Quadratic hydrodynamic drag
    - 30% retained propulsion effectiveness
    - Water-induced rotational damping
    - 6-DOF rigid-body dynamics

Control model:
    - World-frame position PD + analytical trajectory feedforward
    - Attitude-feasible force projection
    - Morphology-aware attitude control
    - Torque-limited quadrotor motor allocation
    - Explicit actuator saturation reporting

Hydrodynamic quantities are parameterized simulation assumptions.
======================================================================
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# EXISTING MORPHOAQUA MODULES
# ======================================================================

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
)

from controller import (
    AttitudeController,
    rotation_matrix,
)

from motor_model import (
    Motor,
    thrust_from_rpm,
)


# ======================================================================
# SIMULATION CONFIGURATION
# ======================================================================

STAGE4B_SIMULATION_TIME = 20.0


# ======================================================================
# MISSION TIMELINE
# ======================================================================

DESCENT_START_TIME = 0.0
DESCENT_END_TIME = 4.0

SUBMERGED_HOLD_START_TIME = 4.0
SUBMERGED_HOLD_END_TIME = 8.0

OUTBOUND_START_TIME = 8.0
OUTBOUND_END_TIME = 13.0

RETURN_START_TIME = 13.0
RETURN_END_TIME = 18.0

FINAL_HOLD_START_TIME = 18.0
FINAL_HOLD_END_TIME = 20.0

# ======================================================================
# TARGETS
# ======================================================================

INITIAL_SUBMERGED_DEPTH = -0.60
SUBMERGED_TARGET_DEPTH = -1.00
OUTBOUND_TARGET_X = 1.00
TARGET_Y = 0.0
TARGET_YAW = 0.0


# ======================================================================
# WATER MODEL
# ======================================================================

FULLY_SUBMERGED_IMMERSION = 1.0

WATER_DENSITY = 1000.0
DISPLACED_VOLUME = 0.00110
WATER_DRAG_COEFFICIENT = 0.90
WATER_REFERENCE_AREA = 0.025
WATER_ROTATIONAL_DAMPING = 0.020

FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS = 0.30


# ======================================================================
# CONTROL LIMITS
# ======================================================================

MAX_HORIZONTAL_ACCELERATION = 2.0
MAX_VERTICAL_ACCELERATION = 2.5


# ======================================================================
# MORPHOLOGY
# ======================================================================

COMPACT_ARM_RATIO = 0.80

CURRENT_ARM_LENGTH = (
    ARM_LENGTH
    * COMPACT_ARM_RATIO
)

CURRENT_INERTIA = (
    INERTIA
    * COMPACT_ARM_RATIO**2
)

INERTIA_RATE = np.zeros((3, 3))

INERTIA_SCALE = COMPACT_ARM_RATIO**2


# ======================================================================
# RESULTS DIRECTORY
# ======================================================================

RESULTS_DIRECTORY = "results"

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True,
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
# STAGE 4B TARGET TRAJECTORY
# ======================================================================

def submerged_trajectory(t):

    if t < DESCENT_END_TIME:

        duration = (
            DESCENT_END_TIME
            - DESCENT_START_TIME
        )

        u = (
            t
            - DESCENT_START_TIME
        ) / duration

        s, ds, d2s = smoothstep_profile(u)

        delta_z = (
            SUBMERGED_TARGET_DEPTH
            - INITIAL_SUBMERGED_DEPTH
        )

        position = np.array(
            [
                0.0,
                0.0,
                INITIAL_SUBMERGED_DEPTH
                + delta_z * s,
            ]
        )

        velocity = np.array(
            [
                0.0,
                0.0,
                delta_z * ds / duration,
            ]
        )

        acceleration = np.array(
            [
                0.0,
                0.0,
                delta_z * d2s / duration**2,
            ]
        )

        return (
            position,
            velocity,
            acceleration,
        )

    if t < OUTBOUND_START_TIME:

        return (
            np.array(
                [
                    0.0,
                    TARGET_Y,
                    SUBMERGED_TARGET_DEPTH,
                ]
            ),
            np.zeros(3),
            np.zeros(3),
        )

    if t < OUTBOUND_END_TIME:

        duration = (
            OUTBOUND_END_TIME
            - OUTBOUND_START_TIME
        )

        u = (
            t
            - OUTBOUND_START_TIME
        ) / duration

        s, ds, d2s = smoothstep_profile(u)

        position = np.array(
            [
                OUTBOUND_TARGET_X * s,
                TARGET_Y,
                SUBMERGED_TARGET_DEPTH,
            ]
        )

        velocity = np.array(
            [
                OUTBOUND_TARGET_X * ds / duration,
                0.0,
                0.0,
            ]
        )

        acceleration = np.array(
            [
                OUTBOUND_TARGET_X * d2s / duration**2,
                0.0,
                0.0,
            ]
        )

        return (
            position,
            velocity,
            acceleration,
        )

    if t < RETURN_END_TIME:

        duration = (
            RETURN_END_TIME
            - RETURN_START_TIME
        )

        u = (
            t
            - RETURN_START_TIME
        ) / duration

        s, ds, d2s = smoothstep_profile(u)

        delta_x = -OUTBOUND_TARGET_X

        position = np.array(
            [
                OUTBOUND_TARGET_X
                + delta_x * s,
                TARGET_Y,
                SUBMERGED_TARGET_DEPTH,
            ]
        )

        velocity = np.array(
            [
                delta_x * ds / duration,
                0.0,
                0.0,
            ]
        )

        acceleration = np.array(
            [
                delta_x * d2s / duration**2,
                0.0,
                0.0,
            ]
        )

        return (
            position,
            velocity,
            acceleration,
        )

    return (
        np.array(
            [
                0.0,
                TARGET_Y,
                SUBMERGED_TARGET_DEPTH,
            ]
        ),
        np.zeros(3),
        np.zeros(3),
    )


# ======================================================================
# UNDERWATER FORCE MODEL
# ======================================================================

def calculate_underwater_effects(velocity):

    buoyancy_magnitude = (
        WATER_DENSITY
        * GRAVITY
        * DISPLACED_VOLUME
    )

    buoyancy_force = np.array(
        [
            0.0,
            0.0,
            buoyancy_magnitude,
        ]
    )

    speed = np.linalg.norm(
        velocity
    )

    if speed > 1e-12:

        drag_magnitude = (
            0.5
            * WATER_DENSITY
            * WATER_DRAG_COEFFICIENT
            * WATER_REFERENCE_AREA
            * speed**2
        )

        drag_force = (
            -drag_magnitude
            * velocity
            / speed
        )

    else:

        drag_magnitude = 0.0
        drag_force = np.zeros(3)

    return (
        buoyancy_force,
        drag_force,
        drag_magnitude,
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

    return (
        KF
        * omega_max**2
    )


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
            [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                arm,
                -arm,
                -arm,
                arm,
            ],
            [
                -arm,
                -arm,
                arm,
                arm,
            ],
            [
                KM,
                -KM,
                KM,
                -KM,
            ],
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

    requested_total = float(
        total_thrust
    )

    collective_clipped = (
        requested_total
        > 4.0 * maximum_thrust
    )

    total_thrust = float(
        np.clip(
            requested_total,
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
        and np.all(
            thrusts <= maximum_thrust
        )
    ):
        return (
            thrusts,
            False,
            1.0,
            collective_clipped,
        )

    low = 0.0
    high = 1.0

    best = calculate_motor_thrusts(
        total_thrust,
        0.0,
        0.0,
        0.0,
        arm_length,
    )

    for _ in range(40):

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
            and np.all(
                candidate <= maximum_thrust
            )
        ):
            best = candidate
            low = scale
        else:
            high = scale

    return (
        np.clip(
            best,
            0.0,
            maximum_thrust,
        ),
        True,
        low,
        collective_clipped,
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
# FORCE-TO-ATTITUDE MAPPING
# ======================================================================

def force_to_desired_angles(required_force):

    magnitude = np.linalg.norm(
        required_force
    )

    if magnitude < 1e-12:
        return np.array(
            [
                0.0,
                0.0,
                TARGET_YAW,
            ]
        )

    direction = (
        required_force
        / magnitude
    )

    desired_roll = np.arctan2(
        -direction[1],
        np.sqrt(
            direction[0]**2
            + direction[2]**2
        ),
    )

    desired_pitch = np.arctan2(
        direction[0],
        direction[2],
    )

    return np.array(
        [
            np.clip(
                desired_roll,
                -MAX_ROLL,
                MAX_ROLL,
            ),
            np.clip(
                desired_pitch,
                -MAX_PITCH,
                MAX_PITCH,
            ),
            TARGET_YAW,
        ]
    )


# ======================================================================
# ATTITUDE-FEASIBLE FORCE PROJECTION
# ======================================================================

def constrain_force_to_attitude_limits(required_force):
    """
    Project the requested world-frame force onto the physically
    achievable thrust direction defined by the configured roll/pitch
    limits.

    The procedure is:

        1. Determine the requested force direction.
        2. Convert it to desired roll/pitch.
        3. Apply the vehicle attitude limits.
        4. Reconstruct the feasible body-Z direction.
        5. Compute the least-squares scalar thrust along that direction.

    The least-squares solution is:

        T = F dot b3

    for:

        min_T ||F - T*b3||^2

    Negative thrust is not physically available, so T is clipped
    to zero.
    """

    force = np.asarray(
        required_force,
        dtype=float,
    )

    magnitude = np.linalg.norm(
        force
    )

    if magnitude <= 1e-12:
        return np.zeros(3)

    raw_angles = force_to_desired_angles(
        force
    )

    feasible_angles = np.array(
        [
            np.clip(
                raw_angles[0],
                -abs(MAX_ROLL),
                abs(MAX_ROLL),
            ),
            np.clip(
                raw_angles[1],
                -abs(MAX_PITCH),
                abs(MAX_PITCH),
            ),
            TARGET_YAW,
        ]
    )

    feasible_rotation = rotation_matrix(
        feasible_angles[0],
        feasible_angles[1],
        feasible_angles[2],
    )

    feasible_body_z = (
        feasible_rotation[:, 2]
    )

    feasible_thrust = max(
        float(
            np.dot(
                force,
                feasible_body_z,
            )
        ),
        0.0,
    )

    return (
        feasible_thrust
        * feasible_body_z
    )


# ======================================================================
# MORPHOLOGY-AWARE ATTITUDE CONTROL
# ======================================================================

def morphology_aware_attitude_control(
    attitude_controller,
    desired_angles,
    angles,
    angular_rates,
):

    base_torque = (
        attitude_controller.update(
            desired_angles,
            angles,
            angular_rates,
        )
    )

    adaptive_torque = (
        INERTIA_SCALE
        * base_torque
    )

    angular_momentum = (
        CURRENT_INERTIA
        @ angular_rates
    )

    gyroscopic_term = np.cross(
        angular_rates,
        angular_momentum,
    )

    inertia_rate_term = (
        INERTIA_RATE
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
        [
            0.0,
            0.0,
            INITIAL_SUBMERGED_DEPTH,
        ],
        dtype=float,
    )

    velocity = np.zeros(3)

    angles = np.zeros(3)

    angular_rates = np.zeros(3)

    motors = [
        Motor(),
        Motor(),
        Motor(),
        Motor(),
    ]

    maximum_thrust = (
        maximum_motor_thrust()
    )

    maximum_arm = (
        CURRENT_ARM_LENGTH
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
        STAGE4B_SIMULATION_TIME
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

    requested_thrust_history = np.zeros(
        steps
    )

    commanded_thrust_history = np.zeros(
        steps
    )

    effective_thrust_history = np.zeros(
        steps
    )

    torque_history = np.zeros(
        (steps, 3)
    )

    rpm_history = np.zeros(
        (steps, 4)
    )

    motor_thrust_history = np.zeros(
        (steps, 4)
    )

    buoyancy_history = np.zeros(
        steps
    )

    drag_history = np.zeros(
        steps
    )

    required_force_history = np.zeros(
        (steps, 3)
    )

    commanded_direction_error_history = np.zeros(
        steps
    )

    thrust_direction_error_history = np.zeros(
        steps
    )

    torque_scale_history = np.ones(
        steps
    )

    torque_saturation_history = np.zeros(
        steps,
        dtype=bool,
    )

    collective_clipped_history = np.zeros(
        steps,
        dtype=bool,
    )

    gravity_force_history = np.zeros(
        (steps, 3)
    )

    buoyancy_force_history = np.zeros(
        (steps, 3)
    )

    drag_force_history = np.zeros(
        (steps, 3)
    )

    for i, t in enumerate(time):

        (
            target_position,
            target_velocity,
            target_acceleration,
        ) = submerged_trajectory(t)

        target_position_history[i] = target_position
        target_velocity_history[i] = target_velocity
        target_acceleration_history[i] = target_acceleration

        position_error = (
            target_position
            - position
        )

        velocity_error = (
            target_velocity
            - velocity
        )

        feedback_acceleration = (
            target_acceleration
            + np.array(
                [
                    POSITION_KP_X,
                    POSITION_KP_Y,
                    POSITION_KP_Z,
                ]
            )
            * position_error
            + np.array(
                [
                    POSITION_KD_X,
                    POSITION_KD_Y,
                    POSITION_KD_Z,
                ]
            )
            * velocity_error
        )

        horizontal_acceleration = (
            feedback_acceleration[:2]
        )

        horizontal_magnitude = np.linalg.norm(
            horizontal_acceleration
        )

        if (
            horizontal_magnitude
            > MAX_HORIZONTAL_ACCELERATION
        ):

            horizontal_acceleration *= (
                MAX_HORIZONTAL_ACCELERATION
                / horizontal_magnitude
            )

        feedback_acceleration[0] = (
            horizontal_acceleration[0]
        )

        feedback_acceleration[1] = (
            horizontal_acceleration[1]
        )

        feedback_acceleration[2] = np.clip(
            feedback_acceleration[2],
            -MAX_VERTICAL_ACCELERATION,
            MAX_VERTICAL_ACCELERATION,
        )

        (
            buoyancy_force,
            hydrodynamic_drag,
            drag_magnitude,
        ) = calculate_underwater_effects(
            velocity
        )

        gravity_force = np.array(
            [
                0.0,
                0.0,
                -MASS * GRAVITY,
            ]
        )

        gravity_force_history[i] = gravity_force
        buoyancy_force_history[i] = buoyancy_force
        drag_force_history[i] = hydrodynamic_drag

        buoyancy_history[i] = (
            buoyancy_force[2]
        )

        drag_history[i] = (
            drag_magnitude
        )

        # --------------------------------------------------------------
        # Raw required world-frame thrust
        # --------------------------------------------------------------

        raw_required_thrust_world = (
            MASS * feedback_acceleration
            - gravity_force
            - buoyancy_force
            - hydrodynamic_drag
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # The raw position controller may request a force direction
        # outside the vehicle's roll/pitch envelope.
        #
        # Project it onto the physically feasible attitude envelope
        # before generating the desired attitude.
        # --------------------------------------------------------------

        required_thrust_world = (
            constrain_force_to_attitude_limits(
                raw_required_thrust_world
            )
        )

        required_force_history[i] = (
            required_thrust_world
        )

        desired_angles = (
            force_to_desired_angles(
                required_thrust_world
            )
        )

        desired_angle_history[i] = (
            desired_angles
        )

        (
            torque_command,
            _base_torque,
            _dynamic_compensation,
        ) = morphology_aware_attitude_control(
            attitude_controller,
            desired_angles,
            angles,
            angular_rates,
        )

        # --------------------------------------------------------------
        # Thrust-direction metrics
        # --------------------------------------------------------------

        required_force_magnitude = np.linalg.norm(
            required_thrust_world
        )

        desired_rotation = rotation_matrix(
            desired_angles[0],
            desired_angles[1],
            desired_angles[2],
        )

        desired_body_z = (
            desired_rotation[:, 2]
        )

        actual_rotation = rotation_matrix(
            angles[0],
            angles[1],
            angles[2],
        )

        actual_body_z = (
            actual_rotation[:, 2]
        )

        if required_force_magnitude > 1e-12:

            required_direction = (
                required_thrust_world
                / required_force_magnitude
            )

            commanded_direction_alignment = np.dot(
                desired_body_z,
                required_direction,
            )

            commanded_direction_alignment = np.clip(
                commanded_direction_alignment,
                -1.0,
                1.0,
            )

            commanded_direction_error_history[i] = (
                np.rad2deg(
                    np.arccos(
                        commanded_direction_alignment
                    )
                )
            )

            actual_direction_alignment = np.dot(
                actual_body_z,
                required_direction,
            )

            actual_direction_alignment = np.clip(
                actual_direction_alignment,
                -1.0,
                1.0,
            )

            thrust_direction_error_history[i] = (
                np.rad2deg(
                    np.arccos(
                        actual_direction_alignment
                    )
                )
            )

        else:

            commanded_direction_alignment = 1.0

            commanded_direction_error_history[i] = 0.0

            thrust_direction_error_history[i] = 0.0

        # --------------------------------------------------------------
        # Feasible underwater thrust
        #
        # required_thrust_world has already been projected onto the
        # feasible attitude envelope.
        #
        # Therefore DO NOT apply another projection here.
        # --------------------------------------------------------------

        requested_aerial_thrust = (
            required_force_magnitude
            / FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS
        )

        requested_thrust_history[i] = (
            requested_aerial_thrust
        )

        (
            commanded_motor_thrusts,
            torque_saturated,
            torque_scale,
            collective_clipped,
        ) = motor_mixer(
            requested_aerial_thrust,
            torque_command[0],
            torque_command[1],
            torque_command[2],
            CURRENT_ARM_LENGTH,
        )

        commanded_total_thrust = np.sum(
            commanded_motor_thrusts
        )

        commanded_thrust_history[i] = (
            commanded_total_thrust
        )

        torque_scale_history[i] = (
            torque_scale
        )

        torque_saturation_history[i] = (
            torque_saturated
        )

        collective_clipped_history[i] = (
            collective_clipped
        )

        commanded_rpms = np.array(
            [
                thrust_to_rpm(thrust)
                for thrust
                in commanded_motor_thrusts
            ]
        )

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
                thrust_from_rpm(rpm)
                for rpm
                in actual_rpms
            ]
        )

        actual_total_aerial_thrust = np.sum(
            actual_motor_thrusts
        )

        actual_total_effective_thrust = (
            actual_total_aerial_thrust
            * FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS
        )

        actual_aerial_torque = (
            calculate_actual_torques(
                actual_motor_thrusts,
                CURRENT_ARM_LENGTH,
            )
        )

        actual_effective_torque = (
            actual_aerial_torque
            * FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS
        )

        water_damping_torque = (
            -WATER_ROTATIONAL_DAMPING
            * angular_rates
        )

        total_actual_torque = (
            actual_effective_torque
            + water_damping_torque
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
                actual_total_effective_thrust,
            ]
        )

        thrust_world = (
            R
            @ thrust_body
        )

        total_force = (
            thrust_world
            + gravity_force
            + buoyancy_force
            + hydrodynamic_drag
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
        # Rotational dynamics
        #
        # I*w_dot + I_dot*w + w x (I*w) = tau
        # --------------------------------------------------------------

        angular_momentum = (
            CURRENT_INERTIA
            @ angular_rates
        )

        angular_acceleration = np.linalg.solve(
            CURRENT_INERTIA,
            total_actual_torque
            - INERTIA_RATE @ angular_rates
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

        cos_theta = np.cos(theta)

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
                    np.sin(phi) * tan_theta,
                    np.cos(phi) * tan_theta,
                ],
                [
                    0.0,
                    np.cos(phi),
                    -np.sin(phi),
                ],
                [
                    0.0,
                    np.sin(phi) / cos_theta,
                    np.cos(phi) / cos_theta,
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
            np.sin(angles[2]),
            np.cos(angles[2]),
        )

        position_history[i] = position
        velocity_history[i] = velocity
        angle_history[i] = angles
        angular_rate_history[i] = angular_rates
        rpm_history[i] = actual_rpms
        motor_thrust_history[i] = actual_motor_thrusts

        effective_thrust_history[i] = (
            actual_total_effective_thrust
        )

        torque_history[i] = (
            total_actual_torque
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
        desired_angle_history,
        requested_thrust_history,
        commanded_thrust_history,
        effective_thrust_history,
        torque_history,
        rpm_history,
        motor_thrust_history,
        buoyancy_history,
        drag_history,
        required_force_history,
        commanded_direction_error_history,
        thrust_direction_error_history,
        torque_scale_history,
        torque_saturation_history,
        collective_clipped_history,
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
    requested_thrust,
    commanded_thrust,
    effective_thrust,
    torque,
    rpm,
    motor_thrust,
    buoyancy,
    drag,
    commanded_direction_error,
    thrust_direction_error,
    torque_scale,
    torque_saturation,
    collective_clipped,
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

    depth_error = np.abs(
        position_error[:, 2]
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

    descent_mask = (
        time < DESCENT_END_TIME
    )

    outbound_mask = (
        (time >= OUTBOUND_START_TIME)
        & (time < OUTBOUND_END_TIME)
    )

    return_mask = (
        (time >= RETURN_START_TIME)
        & (time < RETURN_END_TIME)
    )

    final_hold_mask = (
        time >= FINAL_HOLD_START_TIME
    )

    maximum_motor_thrust_value = (
        maximum_motor_thrust()
    )

    maximum_rpm = np.max(
        rpm
    )

    maximum_motor_thrust_value_used = np.max(
        motor_thrust
    )

    rpm_margin = (
        MAX_RPM
        - maximum_rpm
    )

    motor_thrust_margin = (
        maximum_motor_thrust_value
        - maximum_motor_thrust_value_used
    )

    if np.any(descent_mask):
        descent_error = error_magnitude[
            descent_mask
        ]
    else:
        descent_error = np.array([0.0])

    if np.any(outbound_mask):
        outbound_error = error_magnitude[
            outbound_mask
        ]
    else:
        outbound_error = np.array([0.0])

    if np.any(return_mask):
        return_error = error_magnitude[
            return_mask
        ]
    else:
        return_error = np.array([0.0])

    if np.any(final_hold_mask):
        final_hold_error = error_magnitude[
            final_hold_mask
        ]
    else:
        final_hold_error = np.array([0.0])

    final_position = position[-1]
    final_velocity = velocity[-1]
    final_angles = angles[-1]

    return {
        "Maximum 3-D tracking error (m)": np.max(
            error_magnitude
        ),

        "RMS 3-D tracking error (m)": np.sqrt(
            np.mean(
                error_magnitude**2
            )
        ),

        "Maximum horizontal tracking error (m)": np.max(
            horizontal_error
        ),

        "RMS horizontal tracking error (m)": np.sqrt(
            np.mean(
                horizontal_error**2
            )
        ),

        "Maximum depth tracking error (m)": np.max(
            depth_error
        ),

        "RMS depth tracking error (m)": np.sqrt(
            np.mean(
                depth_error**2
            )
        ),

        "Maximum descent-phase error (m)": np.max(
            descent_error
        ),

        "Maximum outbound-phase error (m)": np.max(
            outbound_error
        ),

        "Maximum return-phase error (m)": np.max(
            return_error
        ),

        "Maximum final-hold error (m)": np.max(
            final_hold_error
        ),

        "Maximum underwater speed (m/s)": np.max(
            speed
        ),

        "Maximum roll (deg)": np.max(
            np.abs(roll_deg)
        ),

        "Maximum pitch (deg)": np.max(
            np.abs(pitch_deg)
        ),

        "Maximum yaw (deg)": np.max(
            np.abs(yaw_deg)
        ),

        "Maximum buoyancy (N)": np.max(
            buoyancy
        ),

        "Maximum hydrodynamic drag (N)": np.max(
            drag
        ),

        "Maximum requested aerial thrust (N)": np.max(
            requested_thrust
        ),

        "Maximum commanded aerial thrust (N)": np.max(
            commanded_thrust
        ),

        "Maximum effective thrust (N)": np.max(
            effective_thrust
        ),

        "Maximum motor RPM": maximum_rpm,

        "RPM margin to MAX_RPM": rpm_margin,

        "Maximum individual motor thrust (N)": (
            maximum_motor_thrust_value_used
        ),

        "Motor thrust margin (N)": (
            motor_thrust_margin
        ),

        "Torque saturation events": int(
            np.count_nonzero(
                torque_saturation
            )
        ),

        "Collective thrust clipping events": int(
            np.count_nonzero(
                collective_clipped
            )
        ),

        "Minimum torque allocation scale": np.min(
            torque_scale
        ),

        "Maximum commanded force-direction error (deg)": np.max(
            commanded_direction_error
        ),

        "Maximum actual thrust-direction error (deg)": np.max(
            thrust_direction_error
        ),

        "Final position error (m)": np.linalg.norm(
            position_error[-1]
        ),

        "Final X (m)": final_position[0],

        "Final Y (m)": final_position[1],

        "Final Z (m)": final_position[2],

        "Final Vx (m/s)": final_velocity[0],

        "Final Vy (m/s)": final_velocity[1],

        "Final Vz (m/s)": final_velocity[2],

        "Final roll (deg)": np.rad2deg(
            final_angles[0]
        ),

        "Final pitch (deg)": np.rad2deg(
            final_angles[1]
        ),

        "Final yaw (deg)": np.rad2deg(
            final_angles[2]
        ),
    }


# ======================================================================
# POSITION PLOT
# ======================================================================

def plot_position_results(
    time,
    position,
    target_position,
):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
    )

    labels = [
        ("X", 0),
        ("Y", 1),
        ("Z", 2),
    ]

    for ax, (label, index) in zip(
        axes,
        labels,
    ):

        ax.plot(
            time,
            position[:, index],
            label=f"Actual {label}",
        )

        ax.plot(
            time,
            target_position[:, index],
            "--",
            label=f"Target {label}",
        )

        ax.set_ylabel(
            f"{label} Position (m)"
        )

        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        "MorphoAqua - Stage 4B "
        "Submerged Position Tracking"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_4B_position_tracking.png",
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
    target_position,
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
        target_position[:, 0],
        target_position[:, 1],
        target_position[:, 2],
        "--",
        label="Target trajectory",
    )

    ax.scatter(
        [0.0],
        [0.0],
        [INITIAL_SUBMERGED_DEPTH],
        s=60,
        label="Initial submerged state",
    )

    ax.scatter(
        [position[-1, 0]],
        [position[-1, 1]],
        [position[-1, 2]],
        s=60,
        label="Final submerged state",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "MorphoAqua - Stage 4B "
        "Underwater 3-D Trajectory"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_4B_3D_trajectory.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ======================================================================
# UNDERWATER CONTROL / ACTUATOR PLOT
# ======================================================================

def plot_stage4b_results(
    time,
    velocity,
    angles,
    desired_angles,
    requested_thrust,
    commanded_thrust,
    effective_thrust,
    torque,
    rpm,
    buoyancy,
    drag,
):

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(13, 19),
        sharex=True,
    )

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

    axes[1].legend(ncol=3)
    axes[1].grid(True)

    axes[2].plot(
        time,
        requested_thrust,
        label="Requested aerial-equivalent thrust",
    )

    axes[2].plot(
        time,
        commanded_thrust,
        "--",
        label="Commanded aerial thrust",
    )

    axes[2].plot(
        time,
        effective_thrust,
        label="Effective underwater thrust",
    )

    axes[2].axhline(
        MASS * GRAVITY,
        linestyle=":",
        label="Weight",
    )

    axes[2].set_ylabel(
        "Thrust (N)"
    )

    axes[2].legend()
    axes[2].grid(True)

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

    for motor_index in range(4):

        axes[4].plot(
            time,
            rpm[:, motor_index],
            label=f"Motor {motor_index + 1}",
        )

    axes[4].axhline(
        MAX_RPM,
        linestyle=":",
        label="MAX_RPM",
    )

    axes[4].set_ylabel(
        "RPM"
    )

    axes[4].legend(ncol=5)
    axes[4].grid(True)

    axes[5].plot(
        time,
        buoyancy,
        label="Buoyancy",
    )

    axes[5].plot(
        time,
        drag,
        "--",
        label="Hydrodynamic drag",
    )

    axes[5].set_ylabel(
        "Force (N)"
    )

    axes[5].set_xlabel(
        "Time (s)"
    )

    axes[5].legend()
    axes[5].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 4B "
        "Underwater Control and Actuator Response"
    )

    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.98,
        ]
    )

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_4B_control_water_response.png",
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
    print("MORPHOAQUA - STAGE 4B")
    print("FULLY SUBMERGED STABILIZATION AND UNDERWATER CONTROL")
    print("=" * 70)
    print()

    print("Mission:")
    print(
        "Initial state: "
        "(0.00, 0.00, -0.60) m"
    )
    print(
        "Smooth descent to Z = -1.00 m"
    )
    print(
        "Submerged stabilization"
    )
    print(
        "Smooth underwater translation to X = +1.00 m"
    )
    print(
        "Smooth return to X = 0.00 m"
    )
    print(
        "Final submerged stabilization"
    )
    print()

    print("Underwater model:")
    print(
        f"Water density: "
        f"{WATER_DENSITY:.1f} kg/m^3"
    )
    print(
        f"Displaced volume: "
        f"{DISPLACED_VOLUME:.6f} m^3"
    )
    print(
        f"Drag coefficient: "
        f"{WATER_DRAG_COEFFICIENT:.2f}"
    )
    print(
        f"Reference area: "
        f"{WATER_REFERENCE_AREA:.4f} m^2"
    )
    print(
        f"Propulsion effectiveness: "
        f"{FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS:.2f}"
    )
    print(
        f"Compact arm ratio: "
        f"{COMPACT_ARM_RATIO:.2f}"
    )
    print()

    print("Important:")
    print(
        "Hydrodynamic parameters are "
        "parameterized simulation assumptions."
    )
    print(
        "They are not experimentally measured "
        "vehicle properties."
    )
    print()

    print(
        f"Simulation time: "
        f"{STAGE4B_SIMULATION_TIME:.1f} s"
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
        desired_angles,
        requested_thrust,
        commanded_thrust,
        effective_thrust,
        torque,
        rpm,
        motor_thrust,
        buoyancy,
        drag,
        required_force,
        commanded_direction_error,
        thrust_direction_error,
        torque_scale,
        torque_saturation,
        collective_clipped,
    ) = results

    metrics = calculate_metrics(
        time,
        position,
        velocity,
        angles,
        target_position,
        requested_thrust,
        commanded_thrust,
        effective_thrust,
        torque,
        rpm,
        motor_thrust,
        buoyancy,
        drag,
        commanded_direction_error,
        thrust_direction_error,
        torque_scale,
        torque_saturation,
        collective_clipped,
    )

    print("STAGE 4B PERFORMANCE METRICS")
    print("-" * 70)

    for name, value in metrics.items():

        if isinstance(
            value,
            (int, np.integer),
        ):

            print(
                f"{name:<50}: "
                f"{value}"
            )

        else:

            print(
                f"{name:<50}: "
                f"{value:.6f}"
            )

    print("-" * 70)
    print()

    print("Controller configuration:")
    print(
        "Position controller: "
        "World-frame PD + analytical trajectory feedforward"
    )
    print(
        "Attitude controller: "
        "Morphology-aware PD"
    )
    print("Degrees of freedom: 6")
    print(
        "Morphology: "
        "Fixed compact underwater configuration"
    )
    print(
        "Rotational dynamics: "
        "Rigid-body inertia with gyroscopic compensation"
    )
    print(
        "Water dynamics: "
        "Buoyancy + quadratic hydrodynamic drag"
    )
    print(
        "Propulsion: "
        "30% fully submerged effectiveness"
    )
    print(
        "Rotational water effect: "
        "Immersion-dependent damping, fully submerged"
    )
    print()

    print(
        "Stage 4B simulation completed."
    )
    print()

    print("Results saved to:")
    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_4B_position_tracking.png",
        )
    )
    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_4B_3D_trajectory.png",
        )
    )
    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_4B_control_water_response.png",
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

    plot_stage4b_results(
        time,
        velocity,
        angles,
        desired_angles,
        requested_thrust,
        commanded_thrust,
        effective_thrust,
        torque,
        rpm,
        buoyancy,
        drag,
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()