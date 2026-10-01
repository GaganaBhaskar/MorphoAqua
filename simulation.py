"""
======================================================================
MorphoAqua - Stage 5B-1
Integrated Aerial -> Water-Entry -> Underwater Mission
======================================================================

Purpose
-------
Stage 5B-1 is the first integration step toward the final Stage 5B
research framework.

It combines the validated physical elements developed earlier:

    - 6-DOF rigid-body dynamics
    - 3-D trajectory tracking
    - in-flight morphology change
    - time-varying inertia and dI/dt compensation
    - air-water interface model
    - buoyancy
    - quadratic hydrodynamic drag
    - medium-dependent propulsion effectiveness
    - underwater 3-D trajectory tracking
    - actuator and motor dynamics
    - actuator saturation reporting

Mission
-------
    0-3 s      Aerial takeoff: z = 0 -> +2.0 m
    3-6 s      Compact -> extended morphology
    6-8 s      Extended aerial hold
    8-11 s     Extended -> compact morphology
    11-14 s    Controlled descent through air-water interface
    14-18 s    Submerged stabilization:
               z = -0.60 m -> -1.00 m
    18-23 s    Underwater trajectory 1:
               (0.0, 0.0, -1.0) ->
               (+0.8, +0.5, -1.2) m
    23-29 s    Underwater trajectory 2:
               (+0.8, +0.5, -1.2) ->
               (-0.6, -0.5, -0.8) m
    29-33 s    Return:
               (-0.6, -0.5, -0.8) ->
               (0.0, 0.0, -1.0) m
    33-35 s    Final submerged stabilization

Important modeling note
-----------------------
This is still a numerical simulation. Hydrodynamic and water-entry
properties are parameterized assumptions, not experimentally measured
vehicle data.

Stage 5B-1 intentionally does NOT yet claim:
    - full CFD water-entry slamming
    - real sensor hardware
    - GPS-denied state estimation
    - MPC/RL/learning-based adaptation
    - experimental structural/material validation

Those are later integration layers of the final Stage 5B build.
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

STAGE5B_SIMULATION_TIME = 35.0


# ======================================================================
# MISSION TIMELINE
# ======================================================================

TAKEOFF_START_TIME = 0.0
TAKEOFF_END_TIME = 3.0

AERIAL_MORPHING_START_TIME = 3.0
AERIAL_MORPHING_END_TIME = 6.0

AERIAL_HOLD_START_TIME = 6.0
AERIAL_HOLD_END_TIME = 8.0

WATER_ENTRY_MORPHING_START_TIME = 8.0
WATER_ENTRY_MORPHING_END_TIME = 11.0

WATER_DESCENT_START_TIME = 11.0
WATER_DESCENT_END_TIME = 14.0

SUBMERGED_STABILIZATION_START_TIME = 14.0
SUBMERGED_STABILIZATION_END_TIME = 18.0

UNDERWATER_TRAJECTORY_1_START_TIME = 18.0
UNDERWATER_TRAJECTORY_1_END_TIME = 23.0

UNDERWATER_TRAJECTORY_2_START_TIME = 23.0
UNDERWATER_TRAJECTORY_2_END_TIME = 29.0

RETURN_START_TIME = 29.0
RETURN_END_TIME = 33.0

FINAL_HOLD_START_TIME = 33.0
FINAL_HOLD_END_TIME = 35.0


# ======================================================================
# TARGET POSITIONS
# ======================================================================

INITIAL_POSITION = np.array(
    [
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


TAKEOFF_TARGET = np.array(
    [
        0.0,
        0.0,
        2.0,
    ],
    dtype=float,
)


WATER_ENTRY_TARGET = np.array(
    [
        0.0,
        0.0,
        -0.60,
    ],
    dtype=float,
)


UNDERWATER_HOLD_TARGET = np.array(
    [
        0.0,
        0.0,
        -1.00,
    ],
    dtype=float,
)


TRAJECTORY_1_TARGET = np.array(
    [
        0.80,
        0.50,
        -1.20,
    ],
    dtype=float,
)


TRAJECTORY_2_TARGET = np.array(
    [
        -0.60,
        -0.50,
        -0.80,
    ],
    dtype=float,
)


FINAL_TARGET = np.array(
    [
        0.0,
        0.0,
        -1.00,
    ],
    dtype=float,
)


TARGET_YAW = 0.0


# ======================================================================
# MORPHOLOGY MODEL
# ======================================================================

COMPACT_ARM_RATIO = 0.80
EXTENDED_ARM_RATIO = 1.20


# ======================================================================
# AIR-WATER MODEL
# ======================================================================

WATER_SURFACE_Z = 0.0
VEHICLE_HALF_HEIGHT = 0.15

WATER_DENSITY = 1000.0
DISPLACED_VOLUME = 0.00110

WATER_DRAG_COEFFICIENT = 0.90
WATER_REFERENCE_AREA = 0.025

WATER_ROTATIONAL_DAMPING = 0.020

AIR_PROPULSION_EFFECTIVENESS = 1.00
FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS = 0.30


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
# GENERIC TRAJECTORY PROFILE
# ======================================================================

def smoothstep_profile(u):
    """
    Quintic profile with zero first and second derivatives at endpoints.
    """

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

    return (
        s,
        ds,
        d2s,
    )


def interpolate_segment(
    t,
    start_time,
    end_time,
    start_position,
    end_position,
):
    """
    Quintic smooth interpolation between two 3-D points.
    """

    duration = (
        end_time
        - start_time
    )

    if duration <= 0.0:
        raise ValueError(
            "Trajectory segment duration must be positive."
        )

    u = (
        t
        - start_time
    ) / duration

    s, ds, d2s = (
        smoothstep_profile(u)
    )

    delta = (
        end_position
        - start_position
    )

    position = (
        start_position
        + delta * s
    )

    velocity = (
        delta
        * ds
        / duration
    )

    acceleration = (
        delta
        * d2s
        / duration**2
    )

    return (
        position,
        velocity,
        acceleration,
    )


# ======================================================================
# MORPHOLOGY
# ======================================================================

def morphology_profile(t):
    """
    Returns:

        morphology_state
            0 = compact
            1 = extended

        morphology_rate
        morphology_acceleration
    """

    if t < AERIAL_MORPHING_START_TIME:
        return 0.0, 0.0, 0.0

    if t < AERIAL_MORPHING_END_TIME:

        u = (
            t
            - AERIAL_MORPHING_START_TIME
        ) / (
            AERIAL_MORPHING_END_TIME
            - AERIAL_MORPHING_START_TIME
        )

        s, ds, d2s = (
            smoothstep_profile(u)
        )

        duration = (
            AERIAL_MORPHING_END_TIME
            - AERIAL_MORPHING_START_TIME
        )

        return (
            float(s),
            float(ds / duration),
            float(d2s / duration**2),
        )

    if t < WATER_ENTRY_MORPHING_START_TIME:
        return 1.0, 0.0, 0.0

    if t < WATER_ENTRY_MORPHING_END_TIME:

        u = (
            t
            - WATER_ENTRY_MORPHING_START_TIME
        ) / (
            WATER_ENTRY_MORPHING_END_TIME
            - WATER_ENTRY_MORPHING_START_TIME
        )

        s, ds, d2s = (
            smoothstep_profile(u)
        )

        duration = (
            WATER_ENTRY_MORPHING_END_TIME
            - WATER_ENTRY_MORPHING_START_TIME
        )

        return (
            float(1.0 - s),
            float(-ds / duration),
            float(-d2s / duration**2),
        )

    return 0.0, 0.0, 0.0


def morphology_parameters(t):
    """
    Calculate time-varying arm length and diagonal inertia.

    I(t) = I0 * arm_ratio^2

    dI/dt = I0 * 2 * arm_ratio * d(arm_ratio)/dt
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
# IMMERSION / MEDIUM MODEL
# ======================================================================

def calculate_immersion_fraction(z):
    """
    Continuous immersion fraction:

        0.0 = vehicle fully above water
        0.5 = approximately half immersed
        1.0 = vehicle fully submerged
    """

    upper_transition = (
        WATER_SURFACE_Z
        + VEHICLE_HALF_HEIGHT
    )

    lower_transition = (
        WATER_SURFACE_Z
        - VEHICLE_HALF_HEIGHT
    )

    if z >= upper_transition:
        return 0.0

    if z <= lower_transition:
        return 1.0

    u = (
        upper_transition
        - z
    ) / (
        upper_transition
        - lower_transition
    )

    s, _, _ = (
        smoothstep_profile(u)
    )

    return float(s)


def calculate_medium_effects(
    position,
    velocity,
):
    """
    Compute buoyancy, hydrodynamic drag and propulsion effectiveness.
    """

    immersion = (
        calculate_immersion_fraction(
            position[2]
        )
    )

    buoyancy_magnitude = (
        WATER_DENSITY
        * GRAVITY
        * DISPLACED_VOLUME
        * immersion
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

    if (
        immersion > 0.0
        and speed > 1e-12
    ):

        drag_magnitude = (
            0.5
            * WATER_DENSITY
            * WATER_DRAG_COEFFICIENT
            * WATER_REFERENCE_AREA
            * speed**2
            * immersion
        )

        drag_force = (
            -drag_magnitude
            * velocity
            / speed
        )

    else:

        drag_magnitude = 0.0
        drag_force = np.zeros(3)

    propulsion_effectiveness = (
        AIR_PROPULSION_EFFECTIVENESS
        - immersion
        * (
            AIR_PROPULSION_EFFECTIVENESS
            - FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS
        )
    )

    return (
        immersion,
        buoyancy_force,
        drag_force,
        drag_magnitude,
        propulsion_effectiveness,
    )


# ======================================================================
# INTEGRATED REFERENCE TRAJECTORY
# ======================================================================

def stage5b_trajectory(t):
    """
    Integrated aerial + transition + underwater reference trajectory.
    """

    # --------------------------------------------------------------
    # Aerial takeoff
    # --------------------------------------------------------------

    if t < TAKEOFF_END_TIME:

        return interpolate_segment(
            t,
            TAKEOFF_START_TIME,
            TAKEOFF_END_TIME,
            INITIAL_POSITION,
            TAKEOFF_TARGET,
        )

    # --------------------------------------------------------------
    # Aerial hold while morphology extends
    # --------------------------------------------------------------

    if t < WATER_ENTRY_MORPHING_START_TIME:

        return (
            TAKEOFF_TARGET.copy(),
            np.zeros(3),
            np.zeros(3),
        )

    # --------------------------------------------------------------
    # Controlled descent through water
    # --------------------------------------------------------------

    if t < WATER_DESCENT_END_TIME:

        return interpolate_segment(
            t,
            WATER_ENTRY_MORPHING_START_TIME,
            WATER_DESCENT_END_TIME,
            TAKEOFF_TARGET,
            WATER_ENTRY_TARGET,
        )

    # --------------------------------------------------------------
    # Submerged stabilization / controlled depth transition
    #
    # The target changes smoothly from -0.60 m to -1.00 m rather
    # than introducing an artificial position step at t = 14 s.
    # --------------------------------------------------------------

    if t < SUBMERGED_STABILIZATION_END_TIME:

        return interpolate_segment(
            t,
            SUBMERGED_STABILIZATION_START_TIME,
            SUBMERGED_STABILIZATION_END_TIME,
            WATER_ENTRY_TARGET,
            UNDERWATER_HOLD_TARGET,
        )

    # --------------------------------------------------------------
    # Underwater trajectory 1
    # --------------------------------------------------------------

    if t < UNDERWATER_TRAJECTORY_1_END_TIME:

        return interpolate_segment(
            t,
            UNDERWATER_TRAJECTORY_1_START_TIME,
            UNDERWATER_TRAJECTORY_1_END_TIME,
            UNDERWATER_HOLD_TARGET,
            TRAJECTORY_1_TARGET,
        )

    # --------------------------------------------------------------
    # Underwater trajectory 2
    # --------------------------------------------------------------

    if t < UNDERWATER_TRAJECTORY_2_END_TIME:

        return interpolate_segment(
            t,
            UNDERWATER_TRAJECTORY_2_START_TIME,
            UNDERWATER_TRAJECTORY_2_END_TIME,
            TRAJECTORY_1_TARGET,
            TRAJECTORY_2_TARGET,
        )

    # --------------------------------------------------------------
    # Return
    # --------------------------------------------------------------

    if t < RETURN_END_TIME:

        return interpolate_segment(
            t,
            RETURN_START_TIME,
            RETURN_END_TIME,
            TRAJECTORY_2_TARGET,
            FINAL_TARGET,
        )

    # --------------------------------------------------------------
    # Final submerged hold
    # --------------------------------------------------------------

    return (
        FINAL_TARGET.copy(),
        np.zeros(3),
        np.zeros(3),
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
    """
    Allocate thrust while respecting individual motor limits.

    Collective thrust is preserved as far as possible. When the
    requested torque makes individual motor thrust infeasible,
    torque is scaled down instead of creating impossible commands.
    """

    maximum_thrust = (
        maximum_motor_thrust()
    )

    requested_total = float(
        total_thrust
    )

    collective_clipped = (
        requested_total
        > 4.0
        * maximum_thrust
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

    candidate = calculate_motor_thrusts(
        total_thrust,
        torque[0],
        torque[1],
        torque[2],
        arm_length,
    )

    if (
        np.all(candidate >= 0.0)
        and np.all(
            candidate <= maximum_thrust
        )
    ):

        return (
            candidate,
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
            low
            + high
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
# FORCE -> DESIRED ATTITUDE
# ======================================================================

def force_to_desired_angles(
    required_force,
):
    magnitude = np.linalg.norm(
        required_force
    )

    if magnitude <= 1e-12:

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


def constrain_force_to_attitude_limits(
    required_force,
):
    """
    Project requested force onto the force physically feasible under
    the available roll/pitch attitude limits.

    This avoids artificially increasing thrust to compensate for an
    unattainable thrust direction.
    """

    magnitude = np.linalg.norm(
        required_force
    )

    if magnitude <= 1e-12:
        return np.zeros(3)

    desired_angles = (
        force_to_desired_angles(
            required_force
        )
    )

    feasible_rotation = rotation_matrix(
        desired_angles[0],
        desired_angles[1],
        desired_angles[2],
    )

    feasible_body_z = (
        feasible_rotation[:, 2]
    )

    feasible_thrust = max(
        float(
            np.dot(
                required_force,
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
    current_inertia,
    inertia_rate,
    inertia_scale,
):
    base_torque = (
        attitude_controller.update(
            desired_angles,
            angles,
            angular_rates,
        )
    )

    adaptive_torque = (
        inertia_scale
        * base_torque
    )

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

    position = (
        INITIAL_POSITION.copy()
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
        STAGE5B_SIMULATION_TIME
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

    immersion_history = np.zeros(
        steps
    )

    propulsion_effectiveness_history = np.zeros(
        steps
    )

    buoyancy_history = np.zeros(
        steps
    )

    drag_history = np.zeros(
        steps
    )

    morphology_history = np.zeros(
        steps
    )

    morphology_rate_history = np.zeros(
        steps
    )

    arm_length_history = np.zeros(
        steps
    )

    inertia_scale_history = np.zeros(
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

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    for i, t in enumerate(time):

        (
            target_position,
            target_velocity,
            target_acceleration,
        ) = stage5b_trajectory(t)

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
            morphology_state,
            morphology_rate,
            _morphology_acceleration,
            current_arm_length,
            _arm_length_rate,
            current_inertia,
            inertia_rate,
            inertia_scale,
            _arm_ratio_acceleration,
        ) = morphology_parameters(t)

        morphology_history[i] = (
            morphology_state
        )

        morphology_rate_history[i] = (
            morphology_rate
        )

        arm_length_history[i] = (
            current_arm_length
        )

        inertia_scale_history[i] = (
            inertia_scale
        )

        (
            immersion,
            buoyancy_force,
            hydrodynamic_drag,
            drag_magnitude,
            propulsion_effectiveness,
        ) = calculate_medium_effects(
            position,
            velocity,
        )

        immersion_history[i] = (
            immersion
        )

        propulsion_effectiveness_history[i] = (
            propulsion_effectiveness
        )

        buoyancy_history[i] = (
            buoyancy_force[2]
        )

        drag_history[i] = (
            drag_magnitude
        )

        # --------------------------------------------------------------
        # Position feedback
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # External forces
        # --------------------------------------------------------------

        gravity_force = np.array(
            [
                0.0,
                0.0,
                -MASS * GRAVITY,
            ]
        )

        # --------------------------------------------------------------
        # Required world-frame thrust
        # --------------------------------------------------------------

        raw_required_force = (
            MASS
            * feedback_acceleration
            - gravity_force
            - buoyancy_force
            - hydrodynamic_drag
        )

        required_thrust_world = (
            constrain_force_to_attitude_limits(
                raw_required_force
            )
        )

        required_force_history[i] = (
            required_thrust_world
        )

        # --------------------------------------------------------------
        # Desired attitude
        # --------------------------------------------------------------

        desired_angles = (
            force_to_desired_angles(
                required_thrust_world
            )
        )

        desired_angle_history[i] = (
            desired_angles
        )

        # --------------------------------------------------------------
        # Attitude control
        # --------------------------------------------------------------

        (
            torque_command,
            _base_torque,
            _dynamic_compensation,
        ) = morphology_aware_attitude_control(
            attitude_controller,
            desired_angles,
            angles,
            angular_rates,
            current_inertia,
            inertia_rate,
            inertia_scale,
        )

        # --------------------------------------------------------------
        # Thrust-direction metrics
        # --------------------------------------------------------------

        required_force_magnitude = (
            np.linalg.norm(
                required_thrust_world
            )
        )

        desired_rotation = (
            rotation_matrix(
                desired_angles[0],
                desired_angles[1],
                desired_angles[2],
            )
        )

        desired_body_z = (
            desired_rotation[:, 2]
        )

        actual_rotation = (
            rotation_matrix(
                angles[0],
                angles[1],
                angles[2],
            )
        )

        actual_body_z = (
            actual_rotation[:, 2]
        )

        if required_force_magnitude > 1e-12:

            required_direction = (
                required_thrust_world
                / required_force_magnitude
            )

            commanded_alignment = np.clip(
                np.dot(
                    desired_body_z,
                    required_direction,
                ),
                -1.0,
                1.0,
            )

            actual_alignment = np.clip(
                np.dot(
                    actual_body_z,
                    required_direction,
                ),
                -1.0,
                1.0,
            )

            commanded_direction_error_history[i] = (
                np.rad2deg(
                    np.arccos(
                        commanded_alignment
                    )
                )
            )

            thrust_direction_error_history[i] = (
                np.rad2deg(
                    np.arccos(
                        actual_alignment
                    )
                )
            )

        # --------------------------------------------------------------
        # Medium-dependent propulsion
        # --------------------------------------------------------------

        required_aerial_equivalent_thrust = (
            required_force_magnitude
            / max(
                propulsion_effectiveness,
                0.05,
            )
        )

        requested_thrust_history[i] = (
            required_aerial_equivalent_thrust
        )

        # --------------------------------------------------------------
        # Motor allocation
        # --------------------------------------------------------------

        (
            commanded_motor_thrusts,
            torque_saturated,
            torque_scale,
            collective_clipped,
        ) = motor_mixer(
            required_aerial_equivalent_thrust,
            torque_command[0],
            torque_command[1],
            torque_command[2],
            current_arm_length,
        )

        commanded_thrust_history[i] = (
            np.sum(
                commanded_motor_thrusts
            )
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

        # --------------------------------------------------------------
        # Motor dynamics
        # --------------------------------------------------------------

        commanded_rpms = np.array(
            [
                thrust_to_rpm(
                    thrust
                )
                for thrust
                in commanded_motor_thrusts
            ]
        )

        actual_rpms = np.array(
            [
                motors[j].update(
                    commanded_rpms[j],
                    DT,
                )
                for j in range(4)
            ]
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

        rpm_history[i] = (
            actual_rpms
        )

        motor_thrust_history[i] = (
            actual_motor_thrusts
        )

        actual_total_aerial_thrust = (
            np.sum(
                actual_motor_thrusts
            )
        )

        effective_motor_thrusts = (
            actual_motor_thrusts
            * propulsion_effectiveness
        )

        actual_total_effective_thrust = (
            np.sum(
                effective_motor_thrusts
            )
        )

        effective_thrust_history[i] = (
            actual_total_effective_thrust
        )

        # --------------------------------------------------------------
        # Actual torques
        # --------------------------------------------------------------

        actual_aerial_torque = (
            calculate_actual_torques(
                actual_motor_thrusts,
                current_arm_length,
            )
        )

        actual_effective_torque = (
            actual_aerial_torque
            * propulsion_effectiveness
        )

        water_damping_torque = (
            -WATER_ROTATIONAL_DAMPING
            * immersion
            * angular_rates
        )

        total_actual_torque = (
            actual_effective_torque
            + water_damping_torque
        )

        torque_history[i] = (
            total_actual_torque
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
        # --------------------------------------------------------------

        angular_momentum = (
            current_inertia
            @ angular_rates
        )

        angular_acceleration = np.linalg.solve(
            current_inertia,
            total_actual_torque
            - (
                inertia_rate
                @ angular_rates
            )
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
        # Store state
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

    return {
        "time": time,
        "position": position_history,
        "velocity": velocity_history,
        "angles": angle_history,
        "angular_rates": angular_rate_history,
        "target_position": target_position_history,
        "target_velocity": target_velocity_history,
        "target_acceleration": target_acceleration_history,
        "desired_angles": desired_angle_history,
        "requested_thrust": requested_thrust_history,
        "commanded_thrust": commanded_thrust_history,
        "effective_thrust": effective_thrust_history,
        "torque": torque_history,
        "rpm": rpm_history,
        "motor_thrust": motor_thrust_history,
        "immersion": immersion_history,
        "propulsion_effectiveness": propulsion_effectiveness_history,
        "buoyancy": buoyancy_history,
        "drag": drag_history,
        "morphology": morphology_history,
        "morphology_rate": morphology_rate_history,
        "arm_length": arm_length_history,
        "inertia_scale": inertia_scale_history,
        "required_force": required_force_history,
        "commanded_direction_error": commanded_direction_error_history,
        "thrust_direction_error": thrust_direction_error_history,
        "torque_scale": torque_scale_history,
        "torque_saturation": torque_saturation_history,
        "collective_clipped": collective_clipped_history,
    }


# ======================================================================
# METRICS
# ======================================================================

def _masked_max(
    values,
    mask,
):
    if np.any(mask):
        return float(
            np.max(
                values[mask]
            )
        )
    return 0.0


def _masked_rms(
    values,
    mask,
):
    if np.any(mask):
        return float(
            np.sqrt(
                np.mean(
                    values[mask]**2
                )
            )
        )
    return 0.0


def calculate_metrics(results):

    time = results["time"]
    position = results["position"]
    velocity = results["velocity"]
    angles = results["angles"]
    target_position = results["target_position"]

    requested_thrust = results["requested_thrust"]
    commanded_thrust = results["commanded_thrust"]
    effective_thrust = results["effective_thrust"]

    rpm = results["rpm"]
    motor_thrust = results["motor_thrust"]

    buoyancy = results["buoyancy"]
    drag = results["drag"]

    commanded_direction_error = (
        results["commanded_direction_error"]
    )

    thrust_direction_error = (
        results["thrust_direction_error"]
    )

    torque_scale = (
        results["torque_scale"]
    )

    torque_saturation = (
        results["torque_saturation"]
    )

    collective_clipped = (
        results["collective_clipped"]
    )

    position_error = (
        target_position
        - position
    )

    error_magnitude = (
        np.linalg.norm(
            position_error,
            axis=1,
        )
    )

    horizontal_error = (
        np.linalg.norm(
            position_error[:, :2],
            axis=1,
        )
    )

    speed = (
        np.linalg.norm(
            velocity,
            axis=1,
        )
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

    water_entry_mask = (
        (time >= WATER_DESCENT_START_TIME)
        &
        (time < WATER_DESCENT_END_TIME)
    )

    stabilization_mask = (
        (time >= SUBMERGED_STABILIZATION_START_TIME)
        &
        (time < SUBMERGED_STABILIZATION_END_TIME)
    )

    trajectory_1_mask = (
        (time >= UNDERWATER_TRAJECTORY_1_START_TIME)
        &
        (time < UNDERWATER_TRAJECTORY_1_END_TIME)
    )

    trajectory_2_mask = (
        (time >= UNDERWATER_TRAJECTORY_2_START_TIME)
        &
        (time < UNDERWATER_TRAJECTORY_2_END_TIME)
    )

    return_mask = (
        (time >= RETURN_START_TIME)
        &
        (time < RETURN_END_TIME)
    )

    final_hold_mask = (
        time >= FINAL_HOLD_START_TIME
    )

    submerged_mask = (
        time >= SUBMERGED_STABILIZATION_START_TIME
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

    final_position_error = (
        target_position[-1]
        - final_position
    )

    maximum_motor_thrust_value = (
        maximum_motor_thrust()
    )

    maximum_rpm = float(
        np.max(rpm)
    )

    maximum_motor_thrust_used = float(
        np.max(motor_thrust)
    )

    if np.any(submerged_mask):
        underwater_speed = speed[
            submerged_mask
        ]
    else:
        underwater_speed = np.array(
            [0.0]
        )

    return {

        "Maximum 3-D tracking error (m)":
            float(
                np.max(
                    error_magnitude
                )
            ),

        "RMS 3-D tracking error (m)":
            float(
                np.sqrt(
                    np.mean(
                        error_magnitude**2
                    )
                )
            ),

        "Maximum horizontal tracking error (m)":
            float(
                np.max(
                    horizontal_error
                )
            ),

        "RMS horizontal tracking error (m)":
            float(
                np.sqrt(
                    np.mean(
                        horizontal_error**2
                    )
                )
            ),

        "Maximum pre-entry tracking error (m)":
            _masked_max(
                error_magnitude,
                time < WATER_DESCENT_START_TIME,
            ),

        "Maximum water-entry error (m)":
            _masked_max(
                error_magnitude,
                water_entry_mask,
            ),

        "Maximum submerged-stabilization error (m)":
            _masked_max(
                error_magnitude,
                stabilization_mask,
            ),

        "RMS submerged-stabilization error (m)":
            _masked_rms(
                error_magnitude,
                stabilization_mask,
            ),

        "Maximum trajectory-1 error (m)":
            _masked_max(
                error_magnitude,
                trajectory_1_mask,
            ),

        "Maximum trajectory-2 error (m)":
            _masked_max(
                error_magnitude,
                trajectory_2_mask,
            ),

        "Maximum return-phase error (m)":
            _masked_max(
                error_magnitude,
                return_mask,
            ),

        "Maximum final-stabilization error (m)":
            _masked_max(
                error_magnitude,
                final_hold_mask,
            ),

        "RMS final-stabilization error (m)":
            _masked_rms(
                error_magnitude,
                final_hold_mask,
            ),

        "Maximum final-stabilization speed (m/s)":
            _masked_max(
                speed,
                final_hold_mask,
            ),

        "Maximum underwater speed (m/s)":
            float(
                np.max(
                    underwater_speed
                )
            ),

        "Maximum roll (deg)":
            float(
                np.max(
                    np.abs(
                        roll_deg
                    )
                )
            ),

        "Maximum pitch (deg)":
            float(
                np.max(
                    np.abs(
                        pitch_deg
                    )
                )
            ),

        "Maximum yaw (deg)":
            float(
                np.max(
                    np.abs(
                        yaw_deg
                    )
                )
            ),

        "Maximum immersion fraction":
            float(
                np.max(
                    results["immersion"]
                )
            ),

        "Minimum propulsion effectiveness":
            float(
                np.min(
                    results[
                        "propulsion_effectiveness"
                    ]
                )
            ),

        "Maximum buoyancy (N)":
            float(
                np.max(
                    buoyancy
                )
            ),

        "Maximum hydrodynamic drag (N)":
            float(
                np.max(
                    drag
                )
            ),

        "Maximum requested aerial-equivalent thrust (N)":
            float(
                np.max(
                    requested_thrust
                )
            ),

        "Maximum commanded aerial thrust (N)":
            float(
                np.max(
                    commanded_thrust
                )
            ),

        "Maximum effective thrust (N)":
            float(
                np.max(
                    effective_thrust
                )
            ),

        "Maximum motor RPM":
            maximum_rpm,

        "RPM margin to MAX_RPM":
            float(
                MAX_RPM
                - maximum_rpm
            ),

        "Maximum individual motor thrust (N)":
            maximum_motor_thrust_used,

        "Motor thrust margin (N)":
            float(
                maximum_motor_thrust_value
                - maximum_motor_thrust_used
            ),

        "Torque saturation events":
            int(
                np.count_nonzero(
                    torque_saturation
                )
            ),

        "Collective thrust clipping events":
            int(
                np.count_nonzero(
                    collective_clipped
                )
            ),

        "Minimum torque allocation scale":
            float(
                np.min(
                    torque_scale
                )
            ),

        "Maximum commanded force-direction error (deg)":
            float(
                np.max(
                    commanded_direction_error
                )
            ),

        "Maximum actual thrust-direction error (deg)":
            float(
                np.max(
                    thrust_direction_error
                )
            ),

        "Minimum arm length (m)":
            float(
                np.min(
                    results["arm_length"]
                )
            ),

        "Maximum arm length (m)":
            float(
                np.max(
                    results["arm_length"]
                )
            ),

        "Minimum inertia scale":
            float(
                np.min(
                    results["inertia_scale"]
                )
            ),

        "Maximum inertia scale":
            float(
                np.max(
                    results["inertia_scale"]
                )
            ),

        "Maximum morphology rate (1/s)":
            float(
                np.max(
                    np.abs(
                        results["morphology_rate"]
                    )
                )
            ),

        "Final position error (m)":
            float(
                np.linalg.norm(
                    final_position_error
                )
            ),

        "Final X (m)":
            float(
                final_position[0]
            ),

        "Final Y (m)":
            float(
                final_position[1]
            ),

        "Final Z (m)":
            float(
                final_position[2]
            ),

        "Final Vx (m/s)":
            float(
                final_velocity[0]
            ),

        "Final Vy (m/s)":
            float(
                final_velocity[1]
            ),

        "Final Vz (m/s)":
            float(
                final_velocity[2]
            ),

        "Final roll (deg)":
            float(
                np.rad2deg(
                    final_angles[0]
                )
            ),

        "Final pitch (deg)":
            float(
                np.rad2deg(
                    final_angles[1]
                )
            ),

        "Final yaw (deg)":
            float(
                np.rad2deg(
                    final_angles[2]
                )
            ),
    }


# ======================================================================
# POSITION PLOT
# ======================================================================

def plot_position_results(
    results,
):

    time = results["time"]
    position = results["position"]
    target_position = results["target_position"]

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

        if index == 2:

            ax.axhline(
                WATER_SURFACE_Z,
                linestyle=":",
                label="Water surface",
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
        "MorphoAqua - Stage 5B-1 "
        "Integrated Mission Position Tracking"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_5B1_position_tracking.png",
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
    results,
):

    position = results["position"]
    target_position = results["target_position"]

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
        [position[0, 0]],
        [position[0, 1]],
        [position[0, 2]],
        s=60,
        label="Mission start",
    )

    ax.scatter(
        [position[-1, 0]],
        [position[-1, 1]],
        [position[-1, 2]],
        s=60,
        label="Final state",
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
        "MorphoAqua - Stage 5B-1 "
        "Integrated Aerial-Aquatic 3-D Trajectory"
    )

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_5B1_3D_trajectory.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ======================================================================
# CONTROL / MEDIUM RESPONSE PLOT
# ======================================================================

def plot_control_response(
    results,
):

    time = results["time"]

    velocity = results["velocity"]
    angles = results["angles"]
    desired_angles = results["desired_angles"]

    requested_thrust = results["requested_thrust"]
    commanded_thrust = results["commanded_thrust"]
    effective_thrust = results["effective_thrust"]

    torque = results["torque"]
    rpm = results["rpm"]

    buoyancy = results["buoyancy"]
    drag = results["drag"]

    immersion = results["immersion"]
    propulsion_effectiveness = (
        results["propulsion_effectiveness"]
    )

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(13, 23),
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
        label="Effective thrust",
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

    axes[4].legend(
        ncol=5
    )

    axes[4].grid(True)

    # --------------------------------------------------------------
    # Medium
    # --------------------------------------------------------------

    axes[5].plot(
        time,
        immersion,
        label="Immersion fraction",
    )

    axes[5].plot(
        time,
        propulsion_effectiveness,
        "--",
        label="Propulsion effectiveness",
    )

    axes[5].set_ylabel(
        "Immersion / effectiveness"
    )

    axes[5].legend()
    axes[5].grid(True)

    # --------------------------------------------------------------
    # Water forces
    # --------------------------------------------------------------

    axes[6].plot(
        time,
        buoyancy,
        label="Buoyancy",
    )

    axes[6].plot(
        time,
        drag,
        "--",
        label="Hydrodynamic drag",
    )

    axes[6].set_ylabel(
        "Force (N)"
    )

    axes[6].set_xlabel(
        "Time (s)"
    )

    axes[6].legend()
    axes[6].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 5B-1 "
        "Integrated Control, Actuation and Medium Response"
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
        "Stage_5B1_control_medium_response.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ======================================================================
# MORPHOLOGY PLOT
# ======================================================================

def plot_morphology(
    results,
):

    time = results["time"]

    morphology = results["morphology"]
    morphology_rate = results["morphology_rate"]

    arm_length = results["arm_length"]
    inertia_scale = results["inertia_scale"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        arm_length,
        label="Arm length",
    )

    axes[0].plot(
        time,
        morphology,
        "--",
        label="Morphology state",
    )

    axes[0].plot(
        time,
        morphology_rate,
        ":",
        label="Morphology rate",
    )

    axes[0].set_ylabel(
        "Morphology / length"
    )

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        inertia_scale,
        label="Inertia scale",
    )

    axes[1].set_ylabel(
        "I / I_nominal"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 5B-1 "
        "Morphology and Time-Varying Inertia"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_5B1_morphology_inertia.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ======================================================================
# DIRECTION / ACTUATOR PLOT
# ======================================================================

def plot_direction_error(
    results,
):

    time = results["time"]

    commanded_error = (
        results["commanded_direction_error"]
    )

    actual_error = (
        results["thrust_direction_error"]
    )

    torque_scale = (
        results["torque_scale"]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )

    axes[0].plot(
        time,
        commanded_error,
        label="Commanded direction error",
    )

    axes[0].plot(
        time,
        actual_error,
        "--",
        label="Actual thrust-direction error",
    )

    axes[0].set_ylabel(
        "Direction error (deg)"
    )

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time,
        torque_scale,
        label="Torque allocation scale",
    )

    axes[1].set_ylabel(
        "Torque scale"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 5B-1 "
        "Attitude Feasibility and Actuator Allocation"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_5B1_direction_actuator_response.png",
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

    print(
        "=" * 76
    )

    print(
        "MORPHOAQUA - STAGE 5B-1"
    )

    print(
        "INTEGRATED AERIAL -> WATER -> UNDERWATER MISSION"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Mission:"
    )

    print(
        "0-3 s   : Aerial takeoff to Z = +2.00 m"
    )

    print(
        "3-6 s   : Compact -> extended morphology"
    )

    print(
        "6-8 s   : Extended aerial hold"
    )

    print(
        "8-11 s  : Extended -> compact morphology"
    )

    print(
        "11-14 s : Controlled descent through water interface"
    )

    print(
        "14-18 s : Submerged stabilization to Z = -1.00 m"
    )

    print(
        "18-23 s : Underwater trajectory 1"
    )

    print(
        "23-29 s : Underwater trajectory 2"
    )

    print(
        "29-33 s : Return to final submerged point"
    )

    print(
        "33-35 s : Final submerged stabilization"
    )

    print()

    print(
        "Physics:"
    )

    print(
        "6-DOF rigid-body dynamics"
    )

    print(
        "Morphology-dependent inertia with dI/dt compensation"
    )

    print(
        "Air-water immersion model"
    )

    print(
        "Buoyancy + quadratic hydrodynamic drag"
    )

    print(
        "Medium-dependent propulsion effectiveness"
    )

    print(
        "Motor dynamics + actuator saturation reporting"
    )

    print()

    print(
        "Important:"
    )

    print(
        "Hydrodynamic quantities are parameterized simulation assumptions."
    )

    print(
        "Stage 5B-1 does not yet claim CFD, real sensors or learning-based control."
    )

    print()

    results = run_simulation()

    metrics = calculate_metrics(
        results
    )

    print(
        "STAGE 5B-1 PERFORMANCE METRICS"
    )

    print(
        "-" * 76
    )

    for name, value in metrics.items():

        if isinstance(
            value,
            (
                int,
                np.integer,
            ),
        ):

            print(
                f"{name:<58}: {value}"
            )

        else:

            print(
                f"{name:<58}: {value:.6f}"
            )

    print(
        "-" * 76
    )

    print()

    print(
        "Controller:"
    )

    print(
        "Position control: world-frame PD + analytical trajectory feedforward"
    )

    print(
        "Attitude control: morphology-aware PD + dynamic compensation"
    )

    print(
        "Force feasibility: attitude-limit-aware force projection"
    )

    print(
        "Degrees of freedom: 6"
    )

    print(
        "Morphology: compact -> extended -> compact -> compact underwater"
    )

    print(
        "Medium model: air / interface / fully submerged"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B1_position_tracking.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B1_3D_trajectory.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B1_control_medium_response.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B1_morphology_inertia.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B1_direction_actuator_response.png",
        )
    )

    plot_position_results(
        results
    )

    plot_3d_trajectory(
        results
    )

    plot_control_response(
        results
    )

    plot_morphology(
        results
    )

    plot_direction_error(
        results
    )

    print()

    print(
        "Stage 5B-1 simulation completed."
    )


if __name__ == "__main__":
    main()