"""
======================================================================
MorphoAqua - Stage 5B-2
Integrated Aerial -> Water-Entry -> Underwater Mission
======================================================================

Stage 5B-2 additions:

    - Reduced-order analytical water-entry disturbance
    - Cross-medium propulsion interface penalty
    - Water-entry impact metrics
    - Impact-induced error metrics
    - Baseline-relative impact recovery metric
    - Dedicated water-entry response visualization

Validated physical elements retained:

    - 6-DOF rigid-body dynamics
    - 3-D trajectory tracking
    - in-flight morphology change
    - time-varying inertia
    - dI/dt compensation
    - air-water interface model
    - buoyancy
    - quadratic hydrodynamic drag
    - medium-dependent propulsion effectiveness
    - underwater 3-D trajectory tracking
    - motor dynamics
    - actuator saturation reporting
    - attitude-limit-aware force projection

Important modeling note
-----------------------
This remains a numerical simulation.

Hydrodynamic and water-entry parameters are parameterized assumptions.
They are not experimentally measured properties of a physical vehicle.

The water-entry disturbance is a reduced-order analytical proxy.
It is NOT CFD/VoF.

The impact force is intentionally excluded from the controller's
feedforward calculation and is applied only to the plant dynamics.

This stage does not claim:

    - full CFD water-entry slamming
    - real sensor hardware
    - GPS-denied state estimation
    - MPC/RL/learning-based adaptation
    - experimental structural/material validation
======================================================================
"""

import os

# ----------------------------------------------------------------------
# Use a non-interactive backend so plots are written directly to files.
# ----------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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


# ======================================================================
# CROSS-MEDIUM PROPULSION
# ======================================================================

AIR_PROPULSION_EFFECTIVENESS = 1.00

FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS = 0.30

INTERFACE_PROPULSION_PENALTY = 0.015


# ======================================================================
# REDUCED-ORDER WATER-ENTRY DISTURBANCE
# ======================================================================

WATER_ENTRY_IMPACT_COEFFICIENT = 0.80

WATER_ENTRY_IMPACT_REFERENCE_AREA = 0.015

WATER_ENTRY_IMPACT_TIME_SCALE = 0.05

WATER_ENTRY_MIN_NORMAL_SPEED = 0.05

WATER_ENTRY_ACTIVE_FRACTION = 0.01


# ======================================================================
# IMPACT RECOVERY DEFINITION
# ======================================================================

# Recovery threshold is defined relative to the measured impact-induced
# tracking-error excursion rather than an arbitrary absolute distance.
IMPACT_RECOVERY_RESIDUAL_FRACTION = 0.10

# Error must remain below the recovery threshold for this duration.
IMPACT_RECOVERY_PERSISTENCE_TIME = 0.50

# Pre-impact interval used to estimate the local tracking-error baseline.
IMPACT_BASELINE_WINDOW = 0.50

# Duration after the force peak used to characterize the immediate
# tracking-error response to the disturbance.
IMPACT_RESPONSE_ANALYSIS_WINDOW = 2.00


# ======================================================================
# CONTROL LIMITS
# ======================================================================

MAX_HORIZONTAL_ACCELERATION = 2.0
MAX_VERTICAL_ACCELERATION = 2.5


# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================

RESULTS_DIRECTORY = "results"

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True,
)


# ======================================================================
# TRAJECTORY PROFILE
# ======================================================================

def smoothstep_profile(u):
    """
    Quintic smoothstep profile.

    Returns:
        s
        ds/du
        d2s/du2
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
    Morphology timeline:

        0-3 s   compact
        3-6 s   compact -> extended
        6-8 s   extended
        8-11 s  extended -> compact
        >11 s   compact
    """

    if t < AERIAL_MORPHING_START_TIME:

        return (
            0.0,
            0.0,
            0.0,
        )

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
            float(
                ds
                / duration
            ),
            float(
                d2s
                / duration**2
            ),
        )

    if t < WATER_ENTRY_MORPHING_START_TIME:

        return (
            1.0,
            0.0,
            0.0,
        )

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
            float(
                1.0
                - s
            ),
            float(
                -ds
                / duration
            ),
            float(
                -d2s
                / duration**2
            ),
        )

    return (
        0.0,
        0.0,
        0.0,
    )


def morphology_parameters(t):
    """
    Calculate time-varying arm length and inertia.

        I(t) = I0 * arm_ratio^2

        dI/dt =
            I0 * 2 * arm_ratio * d(arm_ratio)/dt
    """

    (
        morphology_state,
        morphology_rate,
        morphology_acceleration,
    ) = morphology_profile(
        t
    )

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
# IMMERSION MODEL
# ======================================================================

def calculate_immersion_fraction(z):
    """
    Continuous immersion fraction:

        0.0 = fully above water
        0.5 = approximately half immersed
        1.0 = fully submerged
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


def calculate_immersion_rate(
    z,
    vertical_velocity,
):
    """
    Analytical derivative of immersion fraction.
    """

    upper_transition = (
        WATER_SURFACE_Z
        + VEHICLE_HALF_HEIGHT
    )

    lower_transition = (
        WATER_SURFACE_Z
        - VEHICLE_HALF_HEIGHT
    )

    if (
        z >= upper_transition
        or z <= lower_transition
    ):

        return 0.0

    u = (
        upper_transition
        - z
    ) / (
        upper_transition
        - lower_transition
    )

    _, ds_du, _ = (
        smoothstep_profile(u)
    )

    du_dz = (
        -1.0
        / (
            upper_transition
            - lower_transition
        )
    )

    d_immersion_dz = (
        ds_du
        * du_dz
    )

    return float(
        d_immersion_dz
        * vertical_velocity
    )


# ======================================================================
# MEDIUM EFFECTS
# ======================================================================

def calculate_medium_effects(
    position,
    velocity,
):
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

    # Air -> water effectiveness.
    base_effectiveness = (
        AIR_PROPULSION_EFFECTIVENESS
        - immersion
        * (
            AIR_PROPULSION_EFFECTIVENESS
            - FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS
        )
    )

    # Additional loss exists only at the interface.
    interface_factor = (
        4.0
        * immersion
        * (
            1.0
            - immersion
        )
    )

    propulsion_effectiveness = np.clip(
        base_effectiveness
        - (
            INTERFACE_PROPULSION_PENALTY
            * interface_factor
        ),
        FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS,
        AIR_PROPULSION_EFFECTIVENESS,
    )

    return (
        immersion,
        buoyancy_force,
        drag_force,
        drag_magnitude,
        float(
            propulsion_effectiveness
        ),
    )


# ======================================================================
# WATER-ENTRY IMPACT
# ======================================================================

def calculate_water_entry_impact(
    t,
    position,
    velocity,
):
    """
    Reduced-order analytical water-entry disturbance.

        F_impact =
            0.5 * rho * C_impact * A_entry * Vn^2 * activation

    where:

        Vn =
            max(-Vz, 0)

        activation =
            clip(
                immersion_rate
                * impact_time_scale,
                0,
                1
            )

    The disturbance acts upward because it opposes downward entry.

    It is active ONLY during the intended 11-14 s water-entry phase.

    It is NOT supplied to the controller.
    """

    if not (
        WATER_DESCENT_START_TIME
        <= t
        < WATER_DESCENT_END_TIME
    ):

        return (
            np.zeros(3),
            0.0,
            0.0,
            0.0,
            False,
        )

    z = float(
        position[2]
    )

    vertical_velocity = float(
        velocity[2]
    )

    immersion = (
        calculate_immersion_fraction(
            z
        )
    )

    immersion_rate = (
        calculate_immersion_rate(
            z,
            vertical_velocity,
        )
    )

    normal_speed = max(
        -vertical_velocity,
        0.0,
    )

    in_interface = (
        immersion
        > WATER_ENTRY_ACTIVE_FRACTION
        and
        immersion
        < (
            1.0
            - WATER_ENTRY_ACTIVE_FRACTION
        )
    )

    active = (
        in_interface
        and
        immersion_rate > 0.0
        and
        normal_speed
        >= WATER_ENTRY_MIN_NORMAL_SPEED
    )

    if not active:

        return (
            np.zeros(3),
            0.0,
            normal_speed,
            max(
                immersion_rate,
                0.0,
            ),
            False,
        )

    activation = np.clip(
        immersion_rate
        * WATER_ENTRY_IMPACT_TIME_SCALE,
        0.0,
        1.0,
    )

    impact_magnitude = (
        0.5
        * WATER_DENSITY
        * WATER_ENTRY_IMPACT_COEFFICIENT
        * WATER_ENTRY_IMPACT_REFERENCE_AREA
        * normal_speed**2
        * activation
    )

    impact_force = np.array(
        [
            0.0,
            0.0,
            impact_magnitude,
        ]
    )

    return (
        impact_force,
        float(
            impact_magnitude
        ),
        float(
            normal_speed
        ),
        float(
            immersion_rate
        ),
        True,
    )


# ======================================================================
# INTEGRATED STAGE 5B REFERENCE TRAJECTORY
# ======================================================================

def stage5b_trajectory(t):
    """
    Exact integrated 35 s mission.

        0-3 s:
            Aerial takeoff
            (0,0,0) -> (0,0,+2.0)

        3-6 s:
            Morphology compact -> extended

        6-8 s:
            Extended aerial hold

        8-11 s:
            Morphology extended -> compact
            Position remains at +2.0 m

        11-14 s:
            Controlled descent through air-water interface
            (0,0,+2.0) -> (0,0,-0.60)

        14-18 s:
            Submerged stabilization
            (0,0,-0.60) -> (0,0,-1.00)

        18-23 s:
            Underwater trajectory 1
            (0,0,-1.00) -> (+0.80,+0.50,-1.20)

        23-29 s:
            Underwater trajectory 2
            (+0.80,+0.50,-1.20) -> (-0.60,-0.50,-0.80)

        29-33 s:
            Return
            (-0.60,-0.50,-0.80) -> (0,0,-1.00)

        33-35 s:
            Final stabilization
    """

    # --------------------------------------------------------------
    # 0-3 s: takeoff
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
    # 3-11 s: aerial position hold
    #
    # Morphology changes internally, but position remains at +2.0 m.
    # --------------------------------------------------------------

    if t < WATER_DESCENT_START_TIME:

        return (
            TAKEOFF_TARGET.copy(),
            np.zeros(3),
            np.zeros(3),
        )

    # --------------------------------------------------------------
    # 11-14 s: water-entry descent
    # --------------------------------------------------------------

    if t < WATER_DESCENT_END_TIME:

        return interpolate_segment(
            t,
            WATER_DESCENT_START_TIME,
            WATER_DESCENT_END_TIME,
            TAKEOFF_TARGET,
            WATER_ENTRY_TARGET,
        )

    # --------------------------------------------------------------
    # 14-18 s: submerged stabilization
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
    # 18-23 s: underwater trajectory 1
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
    # 23-29 s: underwater trajectory 2
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
    # 29-33 s: return
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
    # 33-35 s: final hold
    # --------------------------------------------------------------

    return (
        FINAL_TARGET.copy(),
        np.zeros(3),
        np.zeros(3),
    )


# ======================================================================
# THRUST / RPM
# ======================================================================

def thrust_to_rpm(
    thrust,
):
    if thrust <= 0.0:
        return 0.0

    omega = np.sqrt(
        thrust
        / KF
    )

    rpm = (
        omega
        * 60.0
        / (
            2.0
            * np.pi
        )
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
    Allocate thrust under individual-motor limits.

    Torque is scaled down when the complete requested allocation is
    infeasible. Collective thrust remains preserved as far as possible.
    """

    maximum_thrust = (
        maximum_motor_thrust()
    )

    requested_total = float(
        total_thrust
    )

    collective_clipped = (
        requested_total
        > (
            4.0
            * maximum_thrust
        )
    )

    total_thrust = float(
        np.clip(
            requested_total,
            0.0,
            4.0
            * maximum_thrust,
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

    candidate = (
        calculate_motor_thrusts(
            total_thrust,
            torque[0],
            torque[1],
            torque[2],
            arm_length,
        )
    )

    if (
        np.all(
            candidate >= 0.0
        )
        and
        np.all(
            candidate
            <= maximum_thrust
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

    best = (
        calculate_motor_thrusts(
            total_thrust,
            0.0,
            0.0,
            0.0,
            arm_length,
        )
    )

    for _ in range(40):

        scale = (
            low
            + high
        ) / 2.0

        candidate = (
            calculate_motor_thrusts(
                total_thrust,
                torque[0] * scale,
                torque[1] * scale,
                torque[2] * scale,
                arm_length,
            )
        )

        if (
            np.all(
                candidate >= 0.0
            )
            and
            np.all(
                candidate
                <= maximum_thrust
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
    Project required force onto the feasible body-Z direction.
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

    feasible_rotation = (
        rotation_matrix(
            desired_angles[0],
            desired_angles[1],
            desired_angles[2],
        )
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

    gyroscopic_term = (
        np.cross(
            angular_rates,
            angular_momentum,
        )
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
# REFERENCE TRAJECTORY VALIDATION
# ======================================================================

def validate_reference_trajectory():
    """
    Print the reference values at important mission times.
    """

    checkpoint_times = [
        0.0,
        3.0,
        6.0,
        8.0,
        10.0,
        11.0,
        12.5,
        14.0,
        18.0,
        23.0,
        29.0,
        33.0,
        35.0,
    ]

    print(
        "REFERENCE TRAJECTORY CHECKPOINTS"
    )

    print(
        "-" * 76
    )

    for t in checkpoint_times:

        (
            target_position,
            target_velocity,
            _target_acceleration,
        ) = stage5b_trajectory(
            t
        )

        print(
            f"t = {t:5.1f} s | "
            f"X = {target_position[0]:8.4f} | "
            f"Y = {target_position[1]:8.4f} | "
            f"Z = {target_position[2]:8.4f} | "
            f"|V| = {np.linalg.norm(target_velocity):8.4f}"
        )

    print(
        "-" * 76
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

    attitude_controller = (
        AttitudeController(
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

    acceleration_history = np.zeros(
        (steps, 3)
    )

    impact_force_history = np.zeros(
        steps
    )

    impact_normal_speed_history = np.zeros(
        steps
    )

    impact_active_history = np.zeros(
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
        ) = stage5b_trajectory(
            t
        )

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
        ) = morphology_parameters(
            t
        )

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

        (
            water_entry_impact_force,
            water_entry_impact_magnitude,
            water_entry_normal_speed,
            _water_entry_immersion_rate,
            water_entry_impact_active,
        ) = calculate_water_entry_impact(
            t,
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

        impact_force_history[i] = (
            water_entry_impact_magnitude
        )

        impact_normal_speed_history[i] = (
            water_entry_normal_speed
        )

        impact_active_history[i] = (
            water_entry_impact_active
        )

        # --------------------------------------------------------------
        # Position controller
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
        # Gravity
        # --------------------------------------------------------------

        gravity_force = np.array(
            [
                0.0,
                0.0,
                -MASS * GRAVITY,
            ]
        )

        # --------------------------------------------------------------
        # Controller required force
        #
        # Impact force remains deliberately excluded.
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
        # Attitude controller
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

        if (
            required_force_magnitude
            > 1e-12
        ):

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
        # Cross-medium propulsion
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
            + water_entry_impact_force
        )

        acceleration = (
            total_force
            / MASS
        )

        acceleration_history[i] = (
            acceleration
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
        "acceleration": acceleration_history,
        "impact_force": impact_force_history,
        "impact_normal_speed": impact_normal_speed_history,
        "impact_active": impact_active_history,
    }


# ======================================================================
# BASIC METRIC HELPERS
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


# ======================================================================
# IMPACT RECOVERY ANALYSIS
# ======================================================================

def calculate_impact_recovery_metrics(
    time,
    error_magnitude,
    impact_force,
    impact_active,
):
    """
    Calculate impact-relative tracking-error metrics.

    Method
    ------
    1. Detect the first active impact sample.
    2. Calculate local pre-impact RMS tracking error over the preceding
       IMPACT_BASELINE_WINDOW seconds.
    3. Detect the impact-force peak.
    4. Examine the tracking-error response after the force peak for
       IMPACT_RESPONSE_ANALYSIS_WINDOW seconds.
    5. Determine the peak post-impact error.
    6. Define impact-induced error excursion:

           excess =
               max(
                   peak_post_impact_error
                   - baseline_error,
                   0
               )

    7. Define recovery threshold:

           threshold =
               baseline_error
               + 0.10 * excess

       This means recovery corresponds to returning to within 10% of
       the disturbance-induced tracking-error excursion.
    8. Require the error to remain below that threshold for
       IMPACT_RECOVERY_PERSISTENCE_TIME.
    """

    active_indices = np.flatnonzero(
        impact_active
    )

    if active_indices.size == 0:

        return {
            "Impact pre-event RMS error (m)":
                float("nan"),

            "Peak post-impact error (m)":
                float("nan"),

            "Peak impact-induced error excursion (m)":
                float("nan"),

            "Impact recovery threshold (m)":
                float("nan"),

            "Impact response peak time (s)":
                float("nan"),

            "Impact recovery time to 10% residual (s)":
                float("nan"),
        }

    # --------------------------------------------------------------
    # First active impact sample
    # --------------------------------------------------------------

    first_active_index = int(
        active_indices[0]
    )

    first_active_time = float(
        time[
            first_active_index
        ]
    )

    baseline_start_time = max(
        0.0,
        first_active_time
        - IMPACT_BASELINE_WINDOW,
    )

    baseline_mask = (
        (time >= baseline_start_time)
        &
        (time < first_active_time)
    )

    if np.any(
        baseline_mask
    ):

        baseline_error = float(
            np.sqrt(
                np.mean(
                    error_magnitude[
                        baseline_mask
                    ]**2
                )
            )
        )

    else:

        baseline_error = float(
            error_magnitude[
                first_active_index
            ]
        )

    # --------------------------------------------------------------
    # Impact force peak
    # --------------------------------------------------------------

    impact_peak_index = int(
        np.argmax(
            impact_force
        )
    )

    impact_peak_force = float(
        impact_force[
            impact_peak_index
        ]
    )

    if impact_peak_force <= 0.0:

        return {
            "Impact pre-event RMS error (m)":
                baseline_error,

            "Peak post-impact error (m)":
                float("nan"),

            "Peak impact-induced error excursion (m)":
                float("nan"),

            "Impact recovery threshold (m)":
                float("nan"),

            "Impact response peak time (s)":
                float("nan"),

            "Impact recovery time to 10% residual (s)":
                float("nan"),
        }

    impact_peak_time = float(
        time[
            impact_peak_index
        ]
    )

    # --------------------------------------------------------------
    # Immediate post-impact response window
    # --------------------------------------------------------------

    response_end_time = min(
        STAGE5B_SIMULATION_TIME,
        impact_peak_time
        + IMPACT_RESPONSE_ANALYSIS_WINDOW,
    )

    response_mask = (
        (time >= impact_peak_time)
        &
        (time <= response_end_time)
    )

    response_indices = np.flatnonzero(
        response_mask
    )

    if response_indices.size == 0:

        return {
            "Impact pre-event RMS error (m)":
                baseline_error,

            "Peak post-impact error (m)":
                float("nan"),

            "Peak impact-induced error excursion (m)":
                float("nan"),

            "Impact recovery threshold (m)":
                float("nan"),

            "Impact response peak time (s)":
                float("nan"),

            "Impact recovery time to 10% residual (s)":
                float("nan"),
        }

    local_peak_offset = int(
        np.argmax(
            error_magnitude[
                response_indices
            ]
        )
    )

    response_peak_index = int(
        response_indices[
            local_peak_offset
        ]
    )

    response_peak_time = float(
        time[
            response_peak_index
        ]
    )

    peak_post_impact_error = float(
        error_magnitude[
            response_peak_index
        ]
    )

    impact_error_excursion = max(
        peak_post_impact_error
        - baseline_error,
        0.0,
    )

    recovery_threshold = (
        baseline_error
        + (
            IMPACT_RECOVERY_RESIDUAL_FRACTION
            * impact_error_excursion
        )
    )

    # --------------------------------------------------------------
    # Recovery detection
    # --------------------------------------------------------------

    persistence_steps = max(
        1,
        int(
            np.ceil(
                IMPACT_RECOVERY_PERSISTENCE_TIME
                / DT
            )
        ),
    )

    recovery_time = float(
        "nan"
    )

    search_start_index = (
        response_peak_index
        + 1
    )

    for i in range(
        search_start_index,
        len(time)
        - persistence_steps
        + 1,
    ):

        window = error_magnitude[
            i:
            i + persistence_steps
        ]

        if np.all(
            window
            <= recovery_threshold
        ):

            recovery_time = float(
                time[i]
                - response_peak_time
            )

            break

    return {
        "Impact pre-event RMS error (m)":
            baseline_error,

        "Peak post-impact error (m)":
            peak_post_impact_error,

        "Peak impact-induced error excursion (m)":
            impact_error_excursion,

        "Impact recovery threshold (m)":
            recovery_threshold,

        "Impact response peak time (s)":
            response_peak_time,

        "Impact recovery time to 10% residual (s)":
            recovery_time,
    }


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    results,
):
    time = results[
        "time"
    ]

    position = results[
        "position"
    ]

    velocity = results[
        "velocity"
    ]

    angles = results[
        "angles"
    ]

    target_position = results[
        "target_position"
    ]

    requested_thrust = results[
        "requested_thrust"
    ]

    commanded_thrust = results[
        "commanded_thrust"
    ]

    effective_thrust = results[
        "effective_thrust"
    ]

    rpm = results[
        "rpm"
    ]

    motor_thrust = results[
        "motor_thrust"
    ]

    buoyancy = results[
        "buoyancy"
    ]

    drag = results[
        "drag"
    ]

    commanded_direction_error = results[
        "commanded_direction_error"
    ]

    thrust_direction_error = results[
        "thrust_direction_error"
    ]

    torque_scale = results[
        "torque_scale"
    ]

    torque_saturation = results[
        "torque_saturation"
    ]

    collective_clipped = results[
        "collective_clipped"
    ]

    impact_force = results[
        "impact_force"
    ]

    impact_normal_speed = results[
        "impact_normal_speed"
    ]

    acceleration = results[
        "acceleration"
    ]

    impact_active = results[
        "impact_active"
    ]

    # --------------------------------------------------------------
    # Tracking error
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Phase masks
    # --------------------------------------------------------------

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

    post_impact_stabilization_mask = (
        (time >= WATER_DESCENT_END_TIME)
        &
        (time < SUBMERGED_STABILIZATION_END_TIME)
    )

    # --------------------------------------------------------------
    # Final state
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Motor statistics
    # --------------------------------------------------------------

    maximum_motor_thrust_value = (
        maximum_motor_thrust()
    )

    maximum_rpm = float(
        np.max(
            rpm
        )
    )

    maximum_motor_thrust_used = float(
        np.max(
            motor_thrust
        )
    )

    if np.any(
        submerged_mask
    ):

        underwater_speed = (
            speed[
                submerged_mask
            ]
        )

    else:

        underwater_speed = np.array(
            [0.0]
        )

    # --------------------------------------------------------------
    # Water-entry impact metrics
    # --------------------------------------------------------------

    if np.any(
        water_entry_mask
    ):

        local_impact_force = (
            impact_force[
                water_entry_mask
            ]
        )

        local_impact_speed = (
            impact_normal_speed[
                water_entry_mask
            ]
        )

        local_acceleration = (
            np.linalg.norm(
                acceleration[
                    water_entry_mask
                ],
                axis=1,
            )
        )

        peak_force = float(
            np.max(
                local_impact_force
            )
        )

        if peak_force > 0.0:

            window_indices = np.flatnonzero(
                water_entry_mask
            )

            peak_local_index = int(
                np.argmax(
                    local_impact_force
                )
            )

            impact_peak_global_index = int(
                window_indices[
                    peak_local_index
                ]
            )

            impact_peak_time = float(
                time[
                    impact_peak_global_index
                ]
            )

        else:

            impact_peak_time = None

        active_indices = np.flatnonzero(
            impact_active[
                water_entry_mask
            ]
        )

        if active_indices.size > 0:

            window_indices = np.flatnonzero(
                water_entry_mask
            )

            first_active = int(
                active_indices[0]
            )

            last_active = int(
                active_indices[-1]
            )

            active_times = time[
                window_indices[
                    first_active:
                    last_active + 1
                ]
            ]

            active_duration = float(
                active_times[-1]
                - active_times[0]
                + DT
            )

        else:

            active_duration = 0.0

        impact_time = time[
            water_entry_mask
        ]

        try:

            impact_impulse = float(
                np.trapezoid(
                    local_impact_force,
                    impact_time,
                )
            )

        except AttributeError:

            impact_impulse = float(
                np.trapz(
                    local_impact_force,
                    impact_time,
                )
            )

        peak_normal_speed = float(
            np.max(
                local_impact_speed
            )
        )

        peak_impact_acceleration = float(
            np.max(
                local_acceleration
            )
        )

    else:

        impact_peak_time = None
        active_duration = 0.0
        impact_impulse = 0.0
        peak_force = 0.0
        peak_normal_speed = 0.0
        peak_impact_acceleration = 0.0

    # --------------------------------------------------------------
    # Baseline-relative impact recovery
    # --------------------------------------------------------------

    impact_recovery_metrics = (
        calculate_impact_recovery_metrics(
            time,
            error_magnitude,
            impact_force,
            impact_active,
        )
    )

    # --------------------------------------------------------------
    # Other phase metrics
    # --------------------------------------------------------------

    maximum_post_impact_error = (
        _masked_max(
            error_magnitude,
            post_impact_stabilization_mask,
        )
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
                time
                < WATER_DESCENT_START_TIME,
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
                    results[
                        "immersion"
                    ]
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

        "Maximum water-entry impact force (N)":
            peak_force,

        "Water-entry impact impulse (N s)":
            impact_impulse,

        "Maximum water-entry normal speed (m/s)":
            peak_normal_speed,

        "Water-entry active duration (s)":
            active_duration,

        "Peak water-entry acceleration (m/s^2)":
            peak_impact_acceleration,

        "Water-entry impact peak time (s)":
            (
                float(
                    impact_peak_time
                )
                if impact_peak_time is not None
                else float("nan")
            ),

        "Impact pre-event RMS error (m)":
            impact_recovery_metrics[
                "Impact pre-event RMS error (m)"
            ],

        "Peak post-impact error (m)":
            impact_recovery_metrics[
                "Peak post-impact error (m)"
            ],

        "Peak impact-induced error excursion (m)":
            impact_recovery_metrics[
                "Peak impact-induced error excursion (m)"
            ],

        "Impact recovery threshold (m)":
            impact_recovery_metrics[
                "Impact recovery threshold (m)"
            ],

        "Impact response peak time (s)":
            impact_recovery_metrics[
                "Impact response peak time (s)"
            ],

        "Impact recovery time to 10% residual (s)":
            impact_recovery_metrics[
                "Impact recovery time to 10% residual (s)"
            ],

        "Maximum post-impact stabilization error (m)":
            maximum_post_impact_error,

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
                    results[
                        "arm_length"
                    ]
                )
            ),

        "Maximum arm length (m)":
            float(
                np.max(
                    results[
                        "arm_length"
                    ]
                )
            ),

        "Minimum inertia scale":
            float(
                np.min(
                    results[
                        "inertia_scale"
                    ]
                )
            ),

        "Maximum inertia scale":
            float(
                np.max(
                    results[
                        "inertia_scale"
                    ]
                )
            ),

        "Maximum morphology rate (1/s)":
            float(
                np.max(
                    np.abs(
                        results[
                            "morphology_rate"
                        ]
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
# PLOT SAVE HELPER
# ======================================================================

def save_figure(
    fig,
    filename,
):
    output_path = os.path.join(
        RESULTS_DIRECTORY,
        filename,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ======================================================================
# POSITION PLOT
# ======================================================================

def plot_position_results(
    results,
):
    time = results[
        "time"
    ]

    position = results[
        "position"
    ]

    target_position = results[
        "target_position"
    ]

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

    for ax, (
        label,
        index,
    ) in zip(
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
        "MorphoAqua - Stage 5B-2 "
        "Integrated Mission Position Tracking"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B2_position_tracking.png",
    )


# ======================================================================
# 3-D TRAJECTORY
# ======================================================================

def plot_3d_trajectory(
    results,
):
    position = results[
        "position"
    ]

    target_position = results[
        "target_position"
    ]

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
        "MorphoAqua - Stage 5B-2 "
        "Integrated Aerial-Aquatic 3-D Trajectory"
    )

    ax.legend()

    ax.grid(True)

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B2_3D_trajectory.png",
    )


# ======================================================================
# CONTROL / MEDIUM RESPONSE PLOT
# ======================================================================

def plot_control_response(
    results,
):
    time = results[
        "time"
    ]

    velocity = results[
        "velocity"
    ]

    angles = results[
        "angles"
    ]

    desired_angles = results[
        "desired_angles"
    ]

    requested_thrust = results[
        "requested_thrust"
    ]

    commanded_thrust = results[
        "commanded_thrust"
    ]

    effective_thrust = results[
        "effective_thrust"
    ]

    torque = results[
        "torque"
    ]

    rpm = results[
        "rpm"
    ]

    immersion = results[
        "immersion"
    ]

    propulsion_effectiveness = results[
        "propulsion_effectiveness"
    ]

    buoyancy = results[
        "buoyancy"
    ]

    drag = results[
        "drag"
    ]

    impact_force = results[
        "impact_force"
    ]

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
    # RPM
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
    # Immersion / propulsion
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

    axes[6].plot(
        time,
        impact_force,
        ":",
        label="Water-entry impact",
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
        "MorphoAqua - Stage 5B-2 "
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

    save_figure(
        fig,
        "Stage_5B2_control_medium_response.png",
    )


# ======================================================================
# MORPHOLOGY / INERTIA PLOT
# ======================================================================

def plot_morphology(
    results,
):
    time = results[
        "time"
    ]

    morphology = results[
        "morphology"
    ]

    morphology_rate = results[
        "morphology_rate"
    ]

    arm_length = results[
        "arm_length"
    ]

    inertia_scale = results[
        "inertia_scale"
    ]

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
        "MorphoAqua - Stage 5B-2 "
        "Morphology and Time-Varying Inertia"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B2_morphology_inertia.png",
    )


# ======================================================================
# DIRECTION / IMPACT RESPONSE PLOT
# ======================================================================

def plot_direction_and_impact(
    results,
):
    time = results[
        "time"
    ]

    acceleration_magnitude = (
        np.linalg.norm(
            results[
                "acceleration"
            ],
            axis=1,
        )
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 12),
        sharex=True,
    )

    # --------------------------------------------------------------
    # Direction error
    # --------------------------------------------------------------

    axes[0].plot(
        time,
        results[
            "commanded_direction_error"
        ],
        label="Commanded direction error",
    )

    axes[0].plot(
        time,
        results[
            "thrust_direction_error"
        ],
        "--",
        label="Actual thrust-direction error",
    )

    axes[0].set_ylabel(
        "Direction error (deg)"
    )

    axes[0].legend()

    axes[0].grid(True)

    # --------------------------------------------------------------
    # Water-entry response
    # --------------------------------------------------------------

    axes[1].plot(
        time,
        results[
            "impact_force"
        ],
        label="Water-entry impact force",
    )

    axes[1].plot(
        time,
        results[
            "impact_normal_speed"
        ],
        "--",
        label="Water-entry normal speed",
    )

    axes[1].set_ylabel(
        "Impact force / normal speed"
    )

    axes[1].legend()

    axes[1].grid(True)

    # --------------------------------------------------------------
    # Allocation / acceleration
    # --------------------------------------------------------------

    axes[2].plot(
        time,
        results[
            "torque_scale"
        ],
        label="Torque allocation scale",
    )

    axes[2].plot(
        time,
        acceleration_magnitude,
        "--",
        label="Acceleration magnitude",
    )

    axes[2].set_ylabel(
        "Scale / acceleration"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].legend()

    axes[2].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 5B-2 "
        "Direction Feasibility and Water-Entry Response"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B2_direction_impact_response.png",
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "=" * 76
    )

    print(
        "MORPHOAQUA - STAGE 5B-2"
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

    print(
        "Reduced-order analytical water-entry impact disturbance"
    )

    print(
        "Cross-medium propulsion with interface penalty"
    )

    print()

    print(
        "Impact recovery definition:"
    )

    print(
        "Return to baseline + "
        f"{IMPACT_RECOVERY_RESIDUAL_FRACTION * 100:.0f}% "
        "of the impact-induced tracking-error excursion."
    )

    print(
        f"Required persistence: "
        f"{IMPACT_RECOVERY_PERSISTENCE_TIME:.2f} s"
    )

    print()

    print(
        "Important:"
    )

    print(
        "Hydrodynamic and water-entry quantities are "
        "parameterized simulation assumptions."
    )

    print(
        "Water-entry model = reduced-order analytical proxy, "
        "not CFD/VoF."
    )

    print(
        "Impact force is not supplied to the controller "
        "as feedforward."
    )

    print(
        "Matplotlib backend = Agg."
    )

    print()

    validate_reference_trajectory()

    print()

    results = (
        run_simulation()
    )

    metrics = (
        calculate_metrics(
            results
        )
    )

    print()

    print(
        "STAGE 5B-2 PERFORMANCE METRICS"
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
                f"{name:<62}: {value}"
            )

        else:

            print(
                f"{name:<62}: {value:.6f}"
            )

    print(
        "-" * 76
    )

    print()

    print(
        "Controller:"
    )

    print(
        "Position control: world-frame PD + "
        "analytical trajectory feedforward"
    )

    print(
        "Attitude control: morphology-aware PD + "
        "dynamic compensation"
    )

    print(
        "Force feasibility: attitude-limit-aware "
        "force projection"
    )

    print(
        "Degrees of freedom: 6"
    )

    print(
        "Morphology: compact -> extended -> compact "
        "-> compact underwater"
    )

    print(
        "Medium model: air / interface / fully submerged"
    )

    print()

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

    plot_direction_and_impact(
        results
    )

    print(
        "Results saved to:"
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B2_position_tracking.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B2_3D_trajectory.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B2_control_medium_response.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B2_morphology_inertia.png",
        )
    )

    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_5B2_direction_impact_response.png",
        )
    )

    print()

    print(
        "Stage 5B-2 simulation completed."
    )


if __name__ == "__main__":
    main()