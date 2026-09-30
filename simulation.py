"""
======================================================================
MorphoAqua - Stage 4A
Air-Water Transition Simulation
======================================================================

Stage 4A extends the validated Stage 3B aerial simulation into an
aerial-aquatic transition model.

Mission:

    0-3 s     Smooth takeoff to 2 m
    3-6 s     Compact -> extended morphing
    6-8 s     Extended aerial flight / hover
    8-11 s    Extended -> compact morphing
    11-14 s   Controlled descent through the air-water interface
    14-20 s   Fully submerged stabilization and hold

New Stage 4A physics:

    - Air-water interface at z = 0
    - Partial immersion model
    - Buoyancy
    - Hydrodynamic drag
    - Reduced propulsion effectiveness during immersion
    - Water-induced rotational damping
    - Continuous force blending during water entry
    - Time-varying morphology and inertia
    - 6-DOF rigid-body dynamics

Important modeling note:

    The hydrodynamic quantities used here are parameterized simulation
    assumptions. They are NOT experimentally measured properties of a
    physical MorphoAqua vehicle.

This stage is numerical simulation only.
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
    PositionController,
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

STAGE4A_SIMULATION_TIME = 20.0


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
SUBMERGED_HOLD_END_TIME = 20.0


# ======================================================================
# MORPHOLOGY CONFIGURATION
# ======================================================================

COMPACT_ARM_RATIO = 0.80

EXTENDED_ARM_RATIO = 1.20


# ======================================================================
# AERIAL ALTITUDE
# ======================================================================

AERIAL_ALTITUDE = 2.0


# ======================================================================
# SUBMERGED TARGET
# ======================================================================

SUBMERGED_TARGET_DEPTH = -0.60


# ======================================================================
# WATER SURFACE
# ======================================================================

WATER_SURFACE_Z = 0.0


# ======================================================================
# VEHICLE IMMERSION MODEL
# ======================================================================

# The center of mass is used as the main vehicle position.
#
# The vehicle is considered:
#
#     fully in air        z >= +half_height
#     partially immersed  -half_height < z < +half_height
#     fully submerged     z <= -half_height

VEHICLE_HALF_HEIGHT = 0.15


# ======================================================================
# WATER PARAMETERS
# ======================================================================

WATER_DENSITY = 1000.0

GRAVITY_ACCELERATION = GRAVITY


# Parameterized displaced volume.
#
# This is intentionally a modeling parameter rather than a measured
# vehicle volume.

DISPLACED_VOLUME = 0.00110


# Hydrodynamic drag coefficient.

WATER_DRAG_COEFFICIENT = 0.90


# Reference frontal area used for the simplified drag model.

WATER_REFERENCE_AREA = 0.025


# Rotational damping coefficient applied when immersed.

WATER_ROTATIONAL_DAMPING = 0.020


# Propulsion effectiveness.
#
# 1.0  -> full aerial effectiveness
# 0.30 -> 30% of the aerial thrust model retained when fully submerged

FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS = 0.30


# ======================================================================
# CONTROL LIMITS
# ======================================================================

MAX_HORIZONTAL_ACCELERATION = 2.0

MAX_VERTICAL_ACCELERATION = 2.5


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
    """
    Quintic smoothstep.

    Returns:

        s       normalized position
        ds      first derivative
        d2s     second derivative
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


    return s, ds, d2s


# ======================================================================
# MORPHOLOGY MODEL
# ======================================================================

def morphology_profile(t):
    """
    Stage 4A morphology schedule.

    0-3 s:
        Compact

    3-6 s:
        Compact -> Extended

    6-8 s:
        Extended

    8-11 s:
        Extended -> Compact

    11-20 s:
        Compact

    Returns:

        morphology_state
        morphology_rate
        morphology_acceleration

    State:

        0 = compact
        1 = extended
    """

    # --------------------------------------------------------------
    # Compact before aerial morphing
    # --------------------------------------------------------------

    if t < AERIAL_MORPHING_START_TIME:

        return 0.0, 0.0, 0.0


    # --------------------------------------------------------------
    # Compact -> Extended
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # Extended hold
    # --------------------------------------------------------------

    if t < WATER_ENTRY_MORPHING_START_TIME:

        return 1.0, 0.0, 0.0


    # --------------------------------------------------------------
    # Extended -> Compact
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # Compact configuration before water entry
    # --------------------------------------------------------------

    return 0.0, 0.0, 0.0


# ======================================================================
# MORPHOLOGY-DEPENDENT PHYSICAL PARAMETERS
# ======================================================================

def morphology_parameters(t):
    """
    Calculate time-varying arm length and inertia.

    Arm length:

        L(t) = L_nominal * arm_ratio

    Inertia:

        I(t) = I_nominal * arm_ratio^2

    Therefore:

        dI/dt =
            2 * I_nominal * arm_ratio * arm_ratio_rate
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
# WATER IMMERSION MODEL
# ======================================================================

def calculate_immersion_fraction(z):
    """
    Calculate continuous immersion fraction.

    Returns:

        0.0 -> fully in air
        0.5 -> approximately half immersed
        1.0 -> fully submerged

    The transition is smoothed with a quintic profile.
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


    # Normalize:

    # z = upper_transition -> 0
    # z = lower_transition -> 1

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


# ======================================================================
# WATER / MEDIUM MODEL
# ======================================================================

def calculate_medium_effects(
    position,
    velocity,
):
    """
    Calculate buoyancy, hydrodynamic drag and propulsion effectiveness.

    Water is assumed stationary.

    Drag:

        F_D =
            -0.5 * rho * Cd * A * |v| * v * immersion

    Buoyancy:

        F_B =
            rho_water * g * displaced_volume * immersion
    """

    z = position[2]


    immersion = (
        calculate_immersion_fraction(z)
    )


    # --------------------------------------------------------------
    # Buoyancy
    # --------------------------------------------------------------

    buoyancy_magnitude = (

        WATER_DENSITY
        * GRAVITY_ACCELERATION
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


    # --------------------------------------------------------------
    # Hydrodynamic drag
    # --------------------------------------------------------------

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

        drag_force = np.zeros(
            3
        )


    # --------------------------------------------------------------
    # Propulsion effectiveness
    # --------------------------------------------------------------

    propulsion_effectiveness = (

        1.0
        - immersion
        * (
            1.0
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
        and np.all(
            thrusts <= maximum_thrust
        )
    ):

        return thrusts


    # --------------------------------------------------------------
    # Scale torque demand until all motors are feasible
    # --------------------------------------------------------------

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
            and np.all(
                candidate <= maximum_thrust
            )
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
# STAGE 4A TARGET TRAJECTORY
# ======================================================================

def transition_trajectory(t):
    """
    Stage 4A trajectory.

    0-3 s:
        Smooth takeoff from z=0 to z=2

    3-8 s:
        Aerial hold at z=2

    8-11 s:
        Continue aerial hold while morphology changes

    11-14 s:
        Smooth descent from z=2 to z=-0.6

    14-20 s:
        Fully submerged stabilization at z=-0.6
    """

    # --------------------------------------------------------------
    # TAKEOFF
    # --------------------------------------------------------------

    if t < TAKEOFF_END_TIME:

        u = (
            t
            / (
                TAKEOFF_END_TIME
                - TAKEOFF_START_TIME
            )
        )


        s, ds, d2s = (
            smoothstep_profile(u)
        )


        duration = (
            TAKEOFF_END_TIME
            - TAKEOFF_START_TIME
        )


        position = np.array(
            [
                0.0,
                0.0,
                AERIAL_ALTITUDE * s,
            ]
        )


        velocity = np.array(
            [
                0.0,
                0.0,
                (
                    AERIAL_ALTITUDE
                    * ds
                    / duration
                ),
            ]
        )


        acceleration = np.array(
            [
                0.0,
                0.0,
                (
                    AERIAL_ALTITUDE
                    * d2s
                    / duration**2
                ),
            ]
        )


        return (
            position,
            velocity,
            acceleration,
        )


    # --------------------------------------------------------------
    # AERIAL HOLD
    # --------------------------------------------------------------

    if t < WATER_DESCENT_START_TIME:

        return (

            np.array(
                [
                    0.0,
                    0.0,
                    AERIAL_ALTITUDE,
                ]
            ),

            np.zeros(3),

            np.zeros(3),
        )


    # --------------------------------------------------------------
    # DESCENT THROUGH WATER SURFACE
    # --------------------------------------------------------------

    if t < WATER_DESCENT_END_TIME:

        u = (
            t
            - WATER_DESCENT_START_TIME
        ) / (
            WATER_DESCENT_END_TIME
            - WATER_DESCENT_START_TIME
        )


        s, ds, d2s = (
            smoothstep_profile(u)
        )


        duration = (
            WATER_DESCENT_END_TIME
            - WATER_DESCENT_START_TIME
        )


        z = (

            AERIAL_ALTITUDE
            + (
                SUBMERGED_TARGET_DEPTH
                - AERIAL_ALTITUDE
            )
            * s
        )


        vz = (

            (
                SUBMERGED_TARGET_DEPTH
                - AERIAL_ALTITUDE
            )
            * ds
            / duration
        )


        az = (

            (
                SUBMERGED_TARGET_DEPTH
                - AERIAL_ALTITUDE
            )
            * d2s
            / duration**2
        )


        position = np.array(
            [
                0.0,
                0.0,
                z,
            ]
        )


        velocity = np.array(
            [
                0.0,
                0.0,
                vz,
            ]
        )


        acceleration = np.array(
            [
                0.0,
                0.0,
                az,
            ]
        )


        return (
            position,
            velocity,
            acceleration,
        )


    # --------------------------------------------------------------
    # FULLY SUBMERGED HOLD
    # --------------------------------------------------------------

    return (

        np.array(
            [
                0.0,
                0.0,
                SUBMERGED_TARGET_DEPTH,
            ]
        ),

        np.zeros(3),

        np.zeros(3),
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

    # --------------------------------------------------------------
    # Initial state
    # --------------------------------------------------------------

    position = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    velocity = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    angles = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    angular_rates = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    # --------------------------------------------------------------
    # Motors
    # --------------------------------------------------------------

    motors = [
        Motor(),
        Motor(),
        Motor(),
        Motor(),
    ]


    # --------------------------------------------------------------
    # Controllers
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
    # Motor capability
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
    # Simulation time
    # --------------------------------------------------------------

    steps = int(
        STAGE4A_SIMULATION_TIME
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


    effective_thrust_history = np.zeros(
        steps
    )


    torque_history = np.zeros(
        (steps, 3)
    )


    rpm_history = np.zeros(
        (steps, 4)
    )


    # --------------------------------------------------------------
    # Morphology history
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # Medium history
    # --------------------------------------------------------------

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


    # --------------------------------------------------------------
    # Force history
    # --------------------------------------------------------------

    gravity_force_history = np.zeros(
        (steps, 3)
    )


    buoyancy_force_history = np.zeros(
        (steps, 3)
    )


    hydrodynamic_drag_history = np.zeros(
        (steps, 3)
    )


    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    for i, t in enumerate(time):


        # --------------------------------------------------------------
        # Target
        # --------------------------------------------------------------

        (
            target_position,
            target_velocity,
            target_acceleration,
        ) = transition_trajectory(t)


        target_position_history[i] = (
            target_position
        )


        target_velocity_history[i] = (
            target_velocity
        )


        target_acceleration_history[i] = (
            target_acceleration
        )


        # --------------------------------------------------------------
        # Morphology
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # Medium / water effects
        # --------------------------------------------------------------

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

            0.0,

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

        (
            torque_command,
            _base_torque,
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


        # --------------------------------------------------------------
        # Tracking errors
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
        # Desired translational acceleration
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # Limit horizontal acceleration
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # Limit vertical acceleration
        # --------------------------------------------------------------

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


        gravity_force_history[i] = (
            gravity_force
        )


        buoyancy_force_history[i] = (
            buoyancy_force
        )


        hydrodynamic_drag_history[i] = (
            hydrodynamic_drag
        )


        # --------------------------------------------------------------
        # Required total thrust vector
        #
        # M*a = F_thrust + F_gravity
        #       + F_buoyancy + F_drag
        #
        # Therefore:
        #
        # F_thrust =
        #       M*a
        #       - F_gravity
        #       - F_buoyancy
        #       - F_drag
        # --------------------------------------------------------------

        required_thrust_world = (

            MASS
            * feedback_acceleration

            - gravity_force

            - buoyancy_force

            - hydrodynamic_drag
        )


        required_aerial_equivalent_thrust = (

            np.linalg.norm(
                required_thrust_world
            )
            / max(
                propulsion_effectiveness,
                0.05,
            )
        )


        required_aerial_equivalent_thrust = np.clip(

            required_aerial_equivalent_thrust,

            0.0,

            4.0 * maximum_thrust,
        )


        # --------------------------------------------------------------
        # Motor mixing
        # --------------------------------------------------------------

        commanded_motor_thrusts = (

            motor_mixer(

                required_aerial_equivalent_thrust,

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


        # --------------------------------------------------------------
        # Aerial-equivalent motor thrust
        # --------------------------------------------------------------

        actual_motor_thrusts = np.array(

            [
                thrust_from_rpm(
                    rpm
                )

                for rpm
                in actual_rpms
            ]
        )


        actual_total_aerial_thrust = (
            np.sum(
                actual_motor_thrusts
            )
        )


        # --------------------------------------------------------------
        # Medium-dependent effective thrust
        # --------------------------------------------------------------

        effective_motor_thrusts = (

            actual_motor_thrusts
            * propulsion_effectiveness
        )


        actual_total_effective_thrust = (
            np.sum(
                effective_motor_thrusts
            )
        )


        # --------------------------------------------------------------
        # Medium-dependent motor torque
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


        # --------------------------------------------------------------
        # Water rotational damping
        # --------------------------------------------------------------

        water_damping_torque = (

            -WATER_ROTATIONAL_DAMPING
            * immersion
            * angular_rates
        )


        total_actual_torque = (

            actual_effective_torque
            + water_damping_torque
        )


        # --------------------------------------------------------------
        # Body-to-world rotation
        # --------------------------------------------------------------

        R = rotation_matrix(

            angles[0],
            angles[1],
            angles[2],
        )


        # --------------------------------------------------------------
        # Effective thrust in body frame
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # Total translational force
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # Integrate translation
        # --------------------------------------------------------------

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
        # I*w_dot
        # + I_dot*w
        # + w x (I*w)
        # = tau
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
        # Store
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
            actual_total_aerial_thrust
        )


        effective_thrust_history[i] = (
            actual_total_effective_thrust
        )


        torque_history[i] = (
            total_actual_torque
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

        desired_angle_history,

        thrust_history,

        effective_thrust_history,

        torque_history,

        rpm_history,

        morphology_history,

        morphology_rate_history,

        arm_length_history,

        inertia_scale_history,

        immersion_history,

        propulsion_effectiveness_history,

        buoyancy_history,

        drag_history,

        gravity_force_history,

        buoyancy_force_history,

        hydrodynamic_drag_history,
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
    immersion,
    propulsion_effectiveness,
    buoyancy,
    drag,
    arm_length,
):

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
    # Water-entry mask
    # --------------------------------------------------------------

    water_entry_mask = (

        (
            time
            >= WATER_DESCENT_START_TIME
        )

        &

        (
            time
            <= WATER_DESCENT_END_TIME
        )
    )


    submerged_mask = (

        time
        >= SUBMERGED_STABILIZATION_START_TIME
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


    if np.any(water_entry_mask):

        water_entry_error = (

            error_magnitude[
                water_entry_mask
            ]
        )

    else:

        water_entry_error = np.array(
            [0.0]
        )


    if np.any(submerged_mask):

        submerged_error = (

            error_magnitude[
                submerged_mask
            ]
        )

    else:

        submerged_error = np.array(
            [0.0]
        )


    return {

        "Maximum 3-D tracking error (m)":
            np.max(
                error_magnitude
            ),

        "RMS 3-D tracking error (m)":
            np.sqrt(
                np.mean(
                    error_magnitude**2
                )
            ),

        "Maximum tracking error during water entry (m)":
            np.max(
                water_entry_error
            ),

        "RMS tracking error during water entry (m)":
            np.sqrt(
                np.mean(
                    water_entry_error**2
                )
            ),

        "Maximum tracking error while submerged (m)":
            np.max(
                submerged_error
            ),

        "RMS tracking error while submerged (m)":
            np.sqrt(
                np.mean(
                    submerged_error**2
                )
            ),

        "Maximum altitude (m)":
            np.max(
                position[:, 2]
            ),

        "Maximum depth (m)":
            -np.min(
                position[:, 2]
            ),

        "Maximum speed (m/s)":
            np.max(
                speed
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

        "Maximum immersion fraction":
            np.max(
                immersion
            ),

        "Minimum propulsion effectiveness":
            np.min(
                propulsion_effectiveness
            ),

        "Maximum buoyancy (N)":
            np.max(
                buoyancy
            ),

        "Maximum hydrodynamic drag (N)":
            np.max(
                drag
            ),

        "Minimum arm length (m)":
            np.min(
                arm_length
            ),

        "Maximum arm length (m)":
            np.max(
                arm_length
            ),

        "Final immersion fraction":
            immersion[-1],

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


        ax.axhline(
            WATER_SURFACE_Z,
            linestyle=":",
            label="Water surface"
            if index == 2
            else None,
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
        "MorphoAqua - Stage 4A "
        "Air-Water Transition Position Tracking"
    )


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_4A_position_tracking.png",
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
        [0.0],
        s=60,
        label="Start",
    )


    ax.scatter(
        [position[-1, 0]],
        [position[-1, 1]],
        [position[-1, 2]],
        s=60,
        label="Final submerged state",
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
        "MorphoAqua - Stage 4A "
        "Air-Water Transition Trajectory"
    )


    ax.legend()

    ax.grid(True)


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIRECTORY,
        "Stage_4A_3D_trajectory.png",
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.show()

    plt.close(fig)


# ======================================================================
# MEDIUM / MORPHOLOGY / CONTROL PLOT
# ======================================================================

def plot_stage4a_results(
    time,
    velocity,
    angles,
    thrust,
    effective_thrust,
    torque,
    rpm,
    morphology,
    morphology_rate,
    arm_length,
    immersion,
    propulsion_effectiveness,
    buoyancy,
    drag,
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
        label="Roll",
    )


    axes[1].plot(
        time,
        np.rad2deg(
            angles[:, 1]
        ),
        label="Pitch",
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
        label="Aerial-equivalent thrust",
    )


    axes[2].plot(
        time,
        effective_thrust,
        "--",
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


    ax_morph_right = (
        ax_morph.twinx()
    )


    ax_morph_right.plot(
        time,
        morphology,
        "--",
        label="Morphology state",
    )


    ax_morph_right.plot(
        time,
        morphology_rate,
        ":",
        label="Morphology rate",
    )


    ax_morph_right.set_ylabel(
        "Morphology"
    )


    lines_1, labels_1 = (
        ax_morph.get_legend_handles_labels()
    )


    lines_2, labels_2 = (
        ax_morph_right.get_legend_handles_labels()
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
    # Water transition
    # --------------------------------------------------------------

    ax_medium = axes[6]


    ax_medium.plot(
        time,
        immersion,
        label="Immersion fraction",
    )


    ax_medium.plot(
        time,
        propulsion_effectiveness,
        "--",
        label="Propulsion effectiveness",
    )


    ax_medium.set_ylabel(
        "Immersion / effectiveness"
    )


    ax_medium.set_xlabel(
        "Time (s)"
    )


    ax_medium.grid(True)


    ax_medium_right = (
        ax_medium.twinx()
    )


    ax_medium_right.plot(
        time,
        buoyancy,
        label="Buoyancy",
    )


    ax_medium_right.plot(
        time,
        drag,
        "--",
        label="Hydrodynamic drag",
    )


    ax_medium_right.set_ylabel(
        "Force (N)"
    )


    lines_1, labels_1 = (
        ax_medium.get_legend_handles_labels()
    )


    lines_2, labels_2 = (
        ax_medium_right.get_legend_handles_labels()
    )


    ax_medium.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )


    ax_medium.set_title(
        "Air-Water Interface and Hydrodynamic Effects"
    )


    fig.suptitle(
        "MorphoAqua - Stage 4A "
        "Air-Water Transition Dynamics"
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
        "Stage_4A_control_water_response.png",
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
        "MORPHOAQUA - STAGE 4A"
    )

    print(
        "AIR-WATER TRANSITION SIMULATION"
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
        "Compact -> extended aerial morphing"
    )


    print(
        "Extended aerial hold"
    )


    print(
        "Extended -> compact preparation for water entry"
    )


    print(
        "Controlled descent through water surface"
    )


    print(
        "Submerged stabilization at Z = -0.60 m"
    )


    print()


    print(
        "Air-water interface:"
    )


    print(
        "Water surface Z = 0.00 m"
    )


    print(
        f"Immersion transition: "
        f"+{VEHICLE_HALF_HEIGHT:.2f} m "
        f"to "
        f"-{VEHICLE_HALF_HEIGHT:.2f} m"
    )


    print()


    print(
        "Water model:"
    )


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
        f"Fully submerged propulsion effectiveness: "
        f"{FULLY_SUBMERGED_PROPULSION_EFFECTIVENESS:.2f}"
    )


    print()


    print(
        "Important:"
    )


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
        f"{STAGE4A_SIMULATION_TIME:.1f} s"
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

        thrust,

        effective_thrust,

        torque,

        rpm,

        morphology,

        morphology_rate,

        arm_length,

        inertia_scale,

        immersion,

        propulsion_effectiveness,

        buoyancy,

        drag,

        gravity_force,

        buoyancy_force,

        hydrodynamic_drag,

    ) = results


    metrics = calculate_metrics(

        time,

        position,

        velocity,

        angles,

        target_position,

        immersion,

        propulsion_effectiveness,

        buoyancy,

        drag,

        arm_length,
    )


    print()

    print(
        "STAGE 4A PERFORMANCE METRICS"
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
        "Morphology: "
        "Prescribed smooth arm-length transition"
    )


    print(
        "Rotational dynamics: "
        "Time-varying rigid-body inertia"
    )


    print(
        "Water dynamics: "
        "Buoyancy + hydrodynamic drag"
    )


    print(
        "Water entry: "
        "Continuous immersion blending"
    )


    print(
        "Propulsion: "
        "Immersion-dependent effectiveness"
    )


    print(
        "Rotational water effect: "
        "Immersion-dependent damping"
    )


    print()

    print(
        "Stage 4A simulation completed."
    )


    print()

    print(
        "Results saved to:"
    )


    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_4A_position_tracking.png",
        )
    )


    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_4A_3D_trajectory.png",
        )
    )


    print(
        os.path.join(
            RESULTS_DIRECTORY,
            "Stage_4A_control_water_response.png",
        )
    )


    # --------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------

    plot_position_results(

        time,

        position,

        target_position,
    )


    plot_3d_trajectory(

        position,

        target_position,
    )


    plot_stage4a_results(

        time,

        velocity,

        angles,

        thrust,

        effective_thrust,

        torque,

        rpm,

        morphology,

        morphology_rate,

        arm_length,

        immersion,

        propulsion_effectiveness,

        buoyancy,

        drag,
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()