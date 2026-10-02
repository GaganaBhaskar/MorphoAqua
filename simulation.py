"""
======================================================================
MorphoAqua - Stage 5B-3
Simulated Sensing, GPS Loss and State Estimation
======================================================================

Stage 5B-3 extends the integrated Stage 5B-2 aerial-aquatic mission
with simulated sensing and state estimation.

Added:
    - IMU-like acceleration and gyro measurements
    - GPS-like aerial position sensing
    - GPS degradation and loss
    - Vision-like position sensing
    - Vision-like attitude sensing
    - Depth sensing during immersion
    - Sonar-like underwater local-position sensing
    - Sensor noise and dropout
    - 9-state position/velocity/accelerometer-bias Kalman filter
    - Gyro-integrated attitude estimator with visual correction
    - Innovation gating
    - Estimator confidence
    - GPS-denied estimation metrics
    - Sensor availability metrics based on scheduled opportunities

Retained from Stage 5B-2:
    - 6-DOF rigid-body dynamics
    - continuous 3-D trajectory tracking
    - morphology-dependent arm length
    - time-varying inertia
    - dI/dt compensation
    - air-water immersion
    - buoyancy
    - quadratic hydrodynamic drag
    - medium-dependent propulsion effectiveness
    - reduced-order water-entry impact
    - actuator saturation reporting
    - attitude-limit-aware force projection
    - motor dynamics

Important:
    Sensor data are simulated numerical measurements.
    No physical sensor hardware is claimed.

    The accelerometer bias is defined in the body frame and estimated
    in the same frame.

    Stage 5B-3 does not implement adaptive/predictive control.
    That is reserved for Stage 5B-4.
======================================================================
"""

import os

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
# CONFIGURATION
# ======================================================================

SIMULATION_TIME = 35.0

RESULTS_DIRECTORY = "results"

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True,
)


# ======================================================================
# MISSION TIMELINE
# ======================================================================

TAKEOFF_START = 0.0
TAKEOFF_END = 3.0

MORPH_OUT_START = 3.0
MORPH_OUT_END = 6.0

AERIAL_HOLD_START = 6.0
AERIAL_HOLD_END = 8.0

MORPH_IN_START = 8.0
MORPH_IN_END = 11.0

WATER_DESCENT_START = 11.0
WATER_DESCENT_END = 14.0

SUBMERGED_STABILIZATION_START = 14.0
SUBMERGED_STABILIZATION_END = 18.0

TRAJECTORY_1_START = 18.0
TRAJECTORY_1_END = 23.0

TRAJECTORY_2_START = 23.0
TRAJECTORY_2_END = 29.0

RETURN_START = 29.0
RETURN_END = 33.0

FINAL_HOLD_START = 33.0
FINAL_HOLD_END = 35.0


# ======================================================================
# MISSION TARGETS
# ======================================================================

P_INITIAL = np.array(
    [0.0, 0.0, 0.0],
    dtype=float,
)

P_AERIAL = np.array(
    [0.0, 0.0, 2.0],
    dtype=float,
)

P_ENTRY = np.array(
    [0.0, 0.0, -0.60],
    dtype=float,
)

P_SUBMERGED = np.array(
    [0.0, 0.0, -1.00],
    dtype=float,
)

P_TRAJ1 = np.array(
    [0.80, 0.50, -1.20],
    dtype=float,
)

P_TRAJ2 = np.array(
    [-0.60, -0.50, -0.80],
    dtype=float,
)

P_FINAL = np.array(
    [0.0, 0.0, -1.00],
    dtype=float,
)


# ======================================================================
# MORPHOLOGY
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
# PROPULSION
# ======================================================================

AIR_PROPULSION_EFFECTIVENESS = 1.00

WATER_PROPULSION_EFFECTIVENESS = 0.30

INTERFACE_PROPULSION_PENALTY = 0.015


# ======================================================================
# WATER-ENTRY IMPACT
# ======================================================================

WATER_ENTRY_IMPACT_COEFFICIENT = 0.80

WATER_ENTRY_IMPACT_REFERENCE_AREA = 0.015

WATER_ENTRY_IMPACT_TIME_SCALE = 0.05

WATER_ENTRY_MIN_NORMAL_SPEED = 0.05

WATER_ENTRY_ACTIVE_FRACTION = 0.01


# ======================================================================
# IMPACT RECOVERY
# ======================================================================

IMPACT_BASELINE_WINDOW = 0.50

IMPACT_RESPONSE_WINDOW = 2.00

IMPACT_RECOVERY_RESIDUAL_FRACTION = 0.10

IMPACT_RECOVERY_PERSISTENCE_TIME = 0.50


# ======================================================================
# CONTROL LIMITS
# ======================================================================

MAX_HORIZONTAL_ACCELERATION = 2.0

MAX_VERTICAL_ACCELERATION = 2.5


# ======================================================================
# SENSOR MODEL
# ======================================================================

SENSOR_RANDOM_SEED = 20261002


# ----------------------------------------------------------------------
# IMU
# ----------------------------------------------------------------------

IMU_ACCEL_NOISE_STD = 0.08

IMU_GYRO_NOISE_STD = np.deg2rad(
    0.15
)

# Fixed BODY-FRAME accelerometer bias.
IMU_ACCEL_BIAS_BODY = np.array(
    [
        0.015,
        -0.010,
        0.020,
    ],
    dtype=float,
)

IMU_GYRO_BIAS = np.deg2rad(
    np.array(
        [
            0.020,
            -0.015,
            0.025,
        ],
        dtype=float,
    )
)


# ----------------------------------------------------------------------
# GPS
# ----------------------------------------------------------------------

GPS_POSITION_NOISE_STD = 0.030

GPS_RATE_HZ = 10.0

GPS_HANDOFF_START = 10.0

GPS_HANDOFF_END = 11.0


# ----------------------------------------------------------------------
# Vision position
# ----------------------------------------------------------------------

VISION_POSITION_NOISE_STD = 0.015

VISION_POSITION_RATE_HZ = 20.0

VISION_POSITION_MAX_IMMERSION = 0.20

VISION_POSITION_DROPOUT = 0.08


# ----------------------------------------------------------------------
# Vision attitude
# ----------------------------------------------------------------------

VISION_ATTITUDE_NOISE_STD = np.deg2rad(
    0.50
)

VISION_ATTITUDE_RATE_HZ = 20.0

VISION_ATTITUDE_MAX_IMMERSION = 0.30

VISION_ATTITUDE_DROPOUT = 0.10


# ----------------------------------------------------------------------
# Depth
# ----------------------------------------------------------------------

DEPTH_NOISE_STD = 0.012

DEPTH_RATE_HZ = 20.0

DEPTH_MIN_IMMERSION = 0.05

DEPTH_DROPOUT = 0.05


# ----------------------------------------------------------------------
# Sonar-like underwater local position
# ----------------------------------------------------------------------

SONAR_POSITION_NOISE_STD = 0.080

SONAR_RATE_HZ = 10.0

SONAR_MIN_IMMERSION = 0.80

SONAR_DROPOUT = 0.15


# ======================================================================
# STATE ESTIMATOR
# ======================================================================

ESTIMATOR_ACCEL_PROCESS_STD = 0.30

ESTIMATOR_INITIAL_POSITION_STD = 0.020

ESTIMATOR_INITIAL_VELOCITY_STD = 0.050

ESTIMATOR_INITIAL_BIAS_STD = 0.030

ESTIMATOR_BIAS_RANDOM_WALK_STD = 0.002

POSITION_INNOVATION_GATE = 25.0


# ----------------------------------------------------------------------
# Attitude estimator
# ----------------------------------------------------------------------

ATTITUDE_VISUAL_BLEND = 0.18

ATTITUDE_GYRO_BIAS_GAIN = 0.015


# ======================================================================
# MATH HELPERS
# ======================================================================

def wrap_angle_vector(
    values,
):
    return np.arctan2(
        np.sin(values),
        np.cos(values),
    )


def euler_rate_matrix(
    angles,
):
    phi = float(
        angles[0]
    )

    theta = float(
        angles[1]
    )

    cos_theta = np.cos(
        theta
    )

    if abs(
        cos_theta
    ) < 1e-5:

        cos_theta = 1e-5

    tan_theta = (
        np.sin(theta)
        / cos_theta
    )

    return np.array(
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
        ],
        dtype=float,
    )


# ======================================================================
# SMOOTH TRAJECTORY
# ======================================================================

def smoothstep(
    u,
):
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
    t0,
    t1,
    p0,
    p1,
):
    duration = (
        t1 - t0
    )

    if duration <= 0.0:

        raise ValueError(
            "Trajectory duration must be positive."
        )

    u = (
        t - t0
    ) / duration

    s, ds, d2s = smoothstep(
        u
    )

    delta = (
        p1 - p0
    )

    position = (
        p0
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


def periodic_update(
    t,
    rate_hz,
):
    if rate_hz <= 0.0:

        return False

    period = (
        1.0
        / rate_hz
    )

    phase = np.mod(
        t,
        period,
    )

    tolerance = max(
        0.51 * DT,
        1e-9,
    )

    return bool(
        phase <= tolerance
        or
        abs(
            phase - period
        ) <= tolerance
    )


# ======================================================================
# MISSION TRAJECTORY
# ======================================================================

def mission_trajectory(
    t,
):
    if t < TAKEOFF_END:

        return interpolate_segment(
            t,
            TAKEOFF_START,
            TAKEOFF_END,
            P_INITIAL,
            P_AERIAL,
        )

    if t < WATER_DESCENT_START:

        return (
            P_AERIAL.copy(),
            np.zeros(3),
            np.zeros(3),
        )

    if t < WATER_DESCENT_END:

        return interpolate_segment(
            t,
            WATER_DESCENT_START,
            WATER_DESCENT_END,
            P_AERIAL,
            P_ENTRY,
        )

    if t < SUBMERGED_STABILIZATION_END:

        return interpolate_segment(
            t,
            SUBMERGED_STABILIZATION_START,
            SUBMERGED_STABILIZATION_END,
            P_ENTRY,
            P_SUBMERGED,
        )

    if t < TRAJECTORY_1_END:

        return interpolate_segment(
            t,
            TRAJECTORY_1_START,
            TRAJECTORY_1_END,
            P_SUBMERGED,
            P_TRAJ1,
        )

    if t < TRAJECTORY_2_END:

        return interpolate_segment(
            t,
            TRAJECTORY_2_START,
            TRAJECTORY_2_END,
            P_TRAJ1,
            P_TRAJ2,
        )

    if t < RETURN_END:

        return interpolate_segment(
            t,
            RETURN_START,
            RETURN_END,
            P_TRAJ2,
            P_FINAL,
        )

    return (
        P_FINAL.copy(),
        np.zeros(3),
        np.zeros(3),
    )


# ======================================================================
# MORPHOLOGY
# ======================================================================

def morphology_profile(
    t,
):
    if t < MORPH_OUT_START:

        return (
            0.0,
            0.0,
            0.0,
        )

    if t < MORPH_OUT_END:

        u = (
            t - MORPH_OUT_START
        ) / (
            MORPH_OUT_END
            - MORPH_OUT_START
        )

        s, ds, d2s = smoothstep(
            u
        )

        duration = (
            MORPH_OUT_END
            - MORPH_OUT_START
        )

        return (
            float(s),
            float(
                ds / duration
            ),
            float(
                d2s / duration**2
            ),
        )

    if t < MORPH_IN_START:

        return (
            1.0,
            0.0,
            0.0,
        )

    if t < MORPH_IN_END:

        u = (
            t - MORPH_IN_START
        ) / (
            MORPH_IN_END
            - MORPH_IN_START
        )

        s, ds, d2s = smoothstep(
            u
        )

        duration = (
            MORPH_IN_END
            - MORPH_IN_START
        )

        return (
            float(1.0 - s),
            float(
                -ds / duration
            ),
            float(
                -d2s / duration**2
            ),
        )

    return (
        0.0,
        0.0,
        0.0,
    )


def morphology_parameters(
    t,
):
    (
        morphology_state,
        morphology_rate,
        morphology_acceleration,
    ) = morphology_profile(
        t
    )

    morphology_delta = (
        EXTENDED_ARM_RATIO
        - COMPACT_ARM_RATIO
    )

    arm_ratio = (
        COMPACT_ARM_RATIO
        + morphology_delta
        * morphology_state
    )

    arm_ratio_rate = (
        morphology_delta
        * morphology_rate
    )

    arm_ratio_acceleration = (
        morphology_delta
        * morphology_acceleration
    )

    arm_length = (
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
        * 2.0
        * arm_ratio
        * arm_ratio_rate
    )

    return (
        morphology_state,
        morphology_rate,
        morphology_acceleration,
        arm_length,
        arm_length_rate,
        current_inertia,
        inertia_rate,
        inertia_scale,
        arm_ratio_acceleration,
    )


# ======================================================================
# IMMERSION
# ======================================================================

def immersion_fraction(
    z,
):
    upper = (
        WATER_SURFACE_Z
        + VEHICLE_HALF_HEIGHT
    )

    lower = (
        WATER_SURFACE_Z
        - VEHICLE_HALF_HEIGHT
    )

    if z >= upper:

        return 0.0

    if z <= lower:

        return 1.0

    u = (
        upper - z
    ) / (
        upper - lower
    )

    s, _, _ = smoothstep(
        u
    )

    return float(
        s
    )


def immersion_rate(
    z,
    vertical_velocity,
):
    upper = (
        WATER_SURFACE_Z
        + VEHICLE_HALF_HEIGHT
    )

    lower = (
        WATER_SURFACE_Z
        - VEHICLE_HALF_HEIGHT
    )

    if (
        z >= upper
        or
        z <= lower
    ):

        return 0.0

    u = (
        upper - z
    ) / (
        upper - lower
    )

    _, ds_du, _ = smoothstep(
        u
    )

    du_dz = (
        -1.0
        / (
            upper - lower
        )
    )

    return float(
        ds_du
        * du_dz
        * vertical_velocity
    )


# ======================================================================
# MEDIUM EFFECTS
# ======================================================================

def medium_effects(
    position,
    velocity,
):
    immersion = immersion_fraction(
        position[2]
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
        ],
        dtype=float,
    )

    speed = np.linalg.norm(
        velocity
    )

    if (
        immersion > 0.0
        and
        speed > 1e-12
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

    base_effectiveness = (
        AIR_PROPULSION_EFFECTIVENESS
        - immersion
        * (
            AIR_PROPULSION_EFFECTIVENESS
            - WATER_PROPULSION_EFFECTIVENESS
        )
    )

    interface_factor = (
        4.0
        * immersion
        * (
            1.0 - immersion
        )
    )

    propulsion_effectiveness = np.clip(
        base_effectiveness
        - (
            INTERFACE_PROPULSION_PENALTY
            * interface_factor
        ),
        WATER_PROPULSION_EFFECTIVENESS,
        AIR_PROPULSION_EFFECTIVENESS,
    )

    return (
        immersion,
        buoyancy_force,
        drag_force,
        float(drag_magnitude),
        float(propulsion_effectiveness),
    )


# ======================================================================
# WATER-ENTRY IMPACT
# ======================================================================

def water_entry_impact(
    t,
    position,
    velocity,
):
    if not (
        WATER_DESCENT_START
        <= t
        < WATER_DESCENT_END
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

    vz = float(
        velocity[2]
    )

    immersion = immersion_fraction(
        z
    )

    entry_rate = immersion_rate(
        z,
        vz,
    )

    normal_speed = max(
        -vz,
        0.0,
    )

    interface_active = (
        immersion
        > WATER_ENTRY_ACTIVE_FRACTION
        and
        immersion
        <
        1.0
        - WATER_ENTRY_ACTIVE_FRACTION
    )

    active = (
        interface_active
        and
        entry_rate > 0.0
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
                entry_rate,
                0.0,
            ),
            False,
        )

    activation = np.clip(
        entry_rate
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
        ],
        dtype=float,
    )

    return (
        impact_force,
        float(impact_magnitude),
        float(normal_speed),
        float(entry_rate),
        True,
    )


# ======================================================================
# SENSOR SIMULATOR
# ======================================================================

class SensorSimulator:

    def __init__(
        self,
        seed=SENSOR_RANDOM_SEED,
    ):
        self.rng = (
            np.random.default_rng(
                seed
            )
        )

    def available(
        self,
        probability,
    ):
        return bool(
            self.rng.random()
            <=
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

    def gps_probability(
        self,
        t,
        immersion,
    ):
        if immersion > 0.05:

            return 0.0

        if t < GPS_HANDOFF_START:

            return 1.0

        if t < GPS_HANDOFF_END:

            fraction = (
                t
                - GPS_HANDOFF_START
            ) / (
                GPS_HANDOFF_END
                - GPS_HANDOFF_START
            )

            return float(
                1.0
                - fraction
            )

        return 0.0

    def measure(
        self,
        t,
        true_position,
        true_velocity,
        true_angles,
        true_angular_rates,
        true_acceleration,
        immersion,
    ):
        gravity_vector = np.array(
            [
                0.0,
                0.0,
                -GRAVITY,
            ],
            dtype=float,
        )

        true_rotation = (
            rotation_matrix(
                true_angles[0],
                true_angles[1],
                true_angles[2],
            )
        )

        # ==============================================================
        # IMU
        # ==============================================================

        specific_force_body = (
            true_rotation.T
            @ (
                true_acceleration
                - gravity_vector
            )
        )

        imu_acceleration = (
            specific_force_body
            + IMU_ACCEL_BIAS_BODY
            + self.rng.normal(
                0.0,
                IMU_ACCEL_NOISE_STD,
                3,
            )
        )

        imu_gyro = (
            true_angular_rates
            + IMU_GYRO_BIAS
            + self.rng.normal(
                0.0,
                IMU_GYRO_NOISE_STD,
                3,
            )
        )

        # ==============================================================
        # GPS
        # ==============================================================

        gps_probability = (
            self.gps_probability(
                t,
                immersion,
            )
        )

        gps_available = (
            periodic_update(
                t,
                GPS_RATE_HZ,
            )
            and
            self.available(
                gps_probability
            )
        )

        gps_position = None

        if gps_available:

            gps_position = (
                true_position
                + self.rng.normal(
                    0.0,
                    GPS_POSITION_NOISE_STD,
                    3,
                )
            )

        # ==============================================================
        # Vision position
        # ==============================================================

        if (
            immersion
            <=
            VISION_POSITION_MAX_IMMERSION
        ):

            vision_probability = (
                1.0
                - VISION_POSITION_DROPOUT
            )

            if immersion > 0.01:

                vision_probability *= 0.70

        else:

            vision_probability = 0.0

        vision_position_available = (
            periodic_update(
                t,
                VISION_POSITION_RATE_HZ,
            )
            and
            self.available(
                vision_probability
            )
        )

        vision_position = None

        if vision_position_available:

            vision_position = (
                true_position
                + self.rng.normal(
                    0.0,
                    VISION_POSITION_NOISE_STD,
                    3,
                )
            )

        # ==============================================================
        # Vision attitude
        # ==============================================================

        if (
            immersion
            <=
            VISION_ATTITUDE_MAX_IMMERSION
        ):

            attitude_probability = (
                1.0
                - VISION_ATTITUDE_DROPOUT
            )

            if immersion > 0.05:

                attitude_probability *= 0.60

        else:

            attitude_probability = 0.0

        vision_attitude_available = (
            periodic_update(
                t,
                VISION_ATTITUDE_RATE_HZ,
            )
            and
            self.available(
                attitude_probability
            )
        )

        vision_attitude = None

        if vision_attitude_available:

            vision_attitude = wrap_angle_vector(
                true_angles
                + self.rng.normal(
                    0.0,
                    VISION_ATTITUDE_NOISE_STD,
                    3,
                )
            )

        # ==============================================================
        # Depth
        # ==============================================================

        depth_available = (
            periodic_update(
                t,
                DEPTH_RATE_HZ,
            )
            and
            immersion
            >= DEPTH_MIN_IMMERSION
            and
            self.available(
                1.0
                - DEPTH_DROPOUT
            )
        )

        depth = None

        if depth_available:

            depth = float(
                true_position[2]
                + self.rng.normal(
                    0.0,
                    DEPTH_NOISE_STD,
                )
            )

        # ==============================================================
        # Sonar-like underwater position
        # ==============================================================

        sonar_available = (
            periodic_update(
                t,
                SONAR_RATE_HZ,
            )
            and
            immersion
            >= SONAR_MIN_IMMERSION
            and
            self.available(
                1.0
                - SONAR_DROPOUT
            )
        )

        sonar_position = None

        if sonar_available:

            sonar_position = (
                true_position
                + self.rng.normal(
                    0.0,
                    SONAR_POSITION_NOISE_STD,
                    3,
                )
            )

        return {
            "imu_acceleration":
                imu_acceleration,

            "imu_gyro":
                imu_gyro,

            "gps_available":
                gps_available,

            "gps_position":
                gps_position,

            "vision_position_available":
                vision_position_available,

            "vision_position":
                vision_position,

            "vision_attitude_available":
                vision_attitude_available,

            "vision_attitude":
                vision_attitude,

            "depth_available":
                depth_available,

            "depth":
                depth,

            "sonar_available":
                sonar_available,

            "sonar_position":
                sonar_position,
        }


# ======================================================================
# 9-STATE POSITION / VELOCITY / BODY-FRAME ACCELERATION-BIAS KF
# ======================================================================

class PositionVelocityBiasKalmanFilter:
    """
    State:

        [px, py, pz,
         vx, vy, vz,
         bax, bay, baz]

    The accelerometer bias is defined in the BODY FRAME.

    Measurement model:

        a_measured_body
            =
        a_true_body
            + b_a_body
            + noise

    Therefore:

        a_world =
            R * (a_measured_body - b_a_body)
            + g
    """

    def __init__(
        self,
    ):
        self.state = np.zeros(
            9,
            dtype=float,
        )

        self.P = np.diag(
            [
                ESTIMATOR_INITIAL_POSITION_STD**2,
                ESTIMATOR_INITIAL_POSITION_STD**2,
                ESTIMATOR_INITIAL_POSITION_STD**2,

                ESTIMATOR_INITIAL_VELOCITY_STD**2,
                ESTIMATOR_INITIAL_VELOCITY_STD**2,
                ESTIMATOR_INITIAL_VELOCITY_STD**2,

                ESTIMATOR_INITIAL_BIAS_STD**2,
                ESTIMATOR_INITIAL_BIAS_STD**2,
                ESTIMATOR_INITIAL_BIAS_STD**2,
            ]
        )

    @property
    def position(
        self,
    ):
        return self.state[
            :3
        ].copy()

    @property
    def velocity(
        self,
    ):
        return self.state[
            3:6
        ].copy()

    @property
    def acceleration_bias_body(
        self,
    ):
        return self.state[
            6:9
        ].copy()

    def predict(
        self,
        acceleration_body,
        body_to_world_rotation,
        dt,
    ):
        R = np.asarray(
            body_to_world_rotation,
            dtype=float,
        )

        I3 = np.eye(
            3
        )

        half_dt_squared = (
            0.5
            * dt**2
        )

        # --------------------------------------------------------------
        # State transition matrix.
        #
        # Bias is represented in body coordinates.
        # --------------------------------------------------------------

        F = np.eye(
            9
        )

        F[
            :3,
            3:6
        ] = (
            I3
            * dt
        )

        F[
            :3,
            6:9
        ] = (
            -half_dt_squared
            * R
        )

        F[
            3:6,
            6:9
        ] = (
            -dt
            * R
        )

        # --------------------------------------------------------------
        # Input mapping
        # --------------------------------------------------------------

        G = np.zeros(
            (
                9,
                3,
            )
        )

        G[
            :3,
            :
        ] = (
            half_dt_squared
            * R
        )

        G[
            3:6,
            :
        ] = (
            dt
            * R
        )

        gravity = np.array(
            [
                0.0,
                0.0,
                -GRAVITY,
            ]
        )

        bias_body = (
            self.state[
                6:9
            ]
        )

        unbiased_acceleration_world = (
            R
            @ (
                acceleration_body
                - bias_body
            )
        )

        acceleration_world = (
            unbiased_acceleration_world
            + gravity
        )

        # --------------------------------------------------------------
        # Nonlinear-free state propagation
        # --------------------------------------------------------------

        self.state[
            :3
        ] = (
            self.state[
                :3
            ]
            + self.state[
                3:6
            ]
            * dt
            + 0.5
            * acceleration_world
            * dt**2
        )

        self.state[
            3:6
        ] = (
            self.state[
                3:6
            ]
            + acceleration_world
            * dt
        )

        # --------------------------------------------------------------
        # Covariance
        # --------------------------------------------------------------

        Q_acc = (
            ESTIMATOR_ACCEL_PROCESS_STD**2
            * (
                G
                @ G.T
            )
        )

        Q_bias = np.zeros(
            (
                9,
                9,
            )
        )

        Q_bias[
            6:9,
            6:9
        ] = (
            I3
            * (
                ESTIMATOR_BIAS_RANDOM_WALK_STD**2
                * dt
            )
        )

        self.P = (
            F
            @ self.P
            @ F.T
            + Q_acc
            + Q_bias
        )

        self.P = (
            0.5
            * (
                self.P
                + self.P.T
            )
        )

    def update_position(
        self,
        measurement,
        measurement_std,
    ):
        H = np.zeros(
            (
                3,
                9,
            )
        )

        H[
            :,
            :3
        ] = np.eye(
            3
        )

        R = (
            np.eye(3)
            * measurement_std**2
        )

        innovation = (
            measurement
            - H
            @ self.state
        )

        S = (
            H
            @ self.P
            @ H.T
            + R
        )

        try:

            S_inv = np.linalg.inv(
                S
            )

        except np.linalg.LinAlgError:

            return False

        mahalanobis = float(
            innovation.T
            @ S_inv
            @ innovation
        )

        if (
            mahalanobis
            >
            POSITION_INNOVATION_GATE
        ):

            return False

        K = (
            self.P
            @ H.T
            @ S_inv
        )

        self.state = (
            self.state
            + K
            @ innovation
        )

        I9 = np.eye(
            9
        )

        IKH = (
            I9
            - K
            @ H
        )

        # Joseph form
        self.P = (
            IKH
            @ self.P
            @ IKH.T
            + K
            @ R
            @ K.T
        )

        self.P = (
            0.5
            * (
                self.P
                + self.P.T
            )
        )

        return True

    def update_depth(
        self,
        measurement,
        measurement_std,
    ):
        H = np.zeros(
            (
                1,
                9,
            )
        )

        H[
            0,
            2
        ] = 1.0

        R = np.array(
            [
                [
                    measurement_std**2
                ]
            ]
        )

        innovation = np.array(
            [
                measurement
                - self.state[
                    2
                ]
            ]
        )

        S = (
            H
            @ self.P
            @ H.T
            + R
        )

        try:

            S_inv = np.linalg.inv(
                S
            )

        except np.linalg.LinAlgError:

            return False

        mahalanobis = float(
            innovation.T
            @ S_inv
            @ innovation
        )

        if (
            mahalanobis
            >
            POSITION_INNOVATION_GATE
        ):

            return False

        K = (
            self.P
            @ H.T
            @ S_inv
        )

        self.state = (
            self.state
            + (
                K
                @ innovation
            ).reshape(
                -1
            )
        )

        I9 = np.eye(
            9
        )

        IKH = (
            I9
            - K
            @ H
        )

        self.P = (
            IKH
            @ self.P
            @ IKH.T
            + K
            @ R
            @ K.T
        )

        self.P = (
            0.5
            * (
                self.P
                + self.P.T
            )
        )

        return True

    def confidence(
        self,
    ):
        position_variance = float(
            np.trace(
                self.P[
                    :3,
                    :3
                ]
            )
        )

        reference_variance = (
            3.0
            * 0.25**2
        )

        confidence = np.exp(
            -position_variance
            / reference_variance
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )


# ======================================================================
# ATTITUDE ESTIMATOR
# ======================================================================

class AttitudeEstimator:

    def __init__(
        self,
    ):
        self.angles = np.zeros(
            3
        )

        self.gyro_bias = np.zeros(
            3
        )

        self.angular_rates = np.zeros(
            3
        )

    def predict(
        self,
        gyro_measurement,
        dt,
    ):
        corrected_rates = (
            gyro_measurement
            - self.gyro_bias
        )

        self.angular_rates = (
            corrected_rates.copy()
        )

        self.angles += (
            euler_rate_matrix(
                self.angles
            )
            @ corrected_rates
            * dt
        )

        self.angles = (
            wrap_angle_vector(
                self.angles
            )
        )

    def update_visual_attitude(
        self,
        measurement,
    ):
        innovation = (
            wrap_angle_vector(
                measurement
                - self.angles
            )
        )

        self.angles = (
            wrap_angle_vector(
                self.angles
                + ATTITUDE_VISUAL_BLEND
                * innovation
            )
        )

        self.gyro_bias += (
            ATTITUDE_GYRO_BIAS_GAIN
            * innovation
        )

        return innovation


# ======================================================================
# MULTI-MODAL ESTIMATOR
# ======================================================================

class MultiModalStateEstimator:

    def __init__(
        self,
    ):
        self.position_filter = (
            PositionVelocityBiasKalmanFilter()
        )

        self.attitude_filter = (
            AttitudeEstimator()
        )

        self.accepted_updates = {
            "gps": 0,
            "vision_position": 0,
            "depth": 0,
            "sonar": 0,
            "vision_attitude": 0,
        }

        self.rejected_updates = {
            "gps": 0,
            "vision_position": 0,
            "depth": 0,
            "sonar": 0,
            "vision_attitude": 0,
        }

    @property
    def position(
        self,
    ):
        return (
            self.position_filter.position
        )

    @property
    def velocity(
        self,
    ):
        return (
            self.position_filter.velocity
        )

    @property
    def acceleration_bias_body(
        self,
    ):
        return (
            self.position_filter
            .acceleration_bias_body
        )

    @property
    def angles(
        self,
    ):
        return (
            self.attitude_filter.angles.copy()
        )

    @property
    def angular_rates(
        self,
    ):
        return (
            self.attitude_filter
            .angular_rates.copy()
        )

    def update(
        self,
        sensor_measurements,
        dt,
    ):
        # --------------------------------------------------------------
        # Attitude prediction
        # --------------------------------------------------------------

        self.attitude_filter.predict(
            sensor_measurements[
                "imu_gyro"
            ],
            dt,
        )

        estimated_rotation = (
            rotation_matrix(
                self.attitude_filter.angles[0],
                self.attitude_filter.angles[1],
                self.attitude_filter.angles[2],
            )
        )

        # --------------------------------------------------------------
        # Translational prediction
        # --------------------------------------------------------------

        self.position_filter.predict(
            sensor_measurements[
                "imu_acceleration"
            ],
            estimated_rotation,
            dt,
        )

        # --------------------------------------------------------------
        # GPS update
        # --------------------------------------------------------------

        if sensor_measurements[
            "gps_available"
        ]:

            accepted = (
                self.position_filter
                .update_position(
                    sensor_measurements[
                        "gps_position"
                    ],
                    GPS_POSITION_NOISE_STD,
                )
            )

            if accepted:

                self.accepted_updates[
                    "gps"
                ] += 1

            else:

                self.rejected_updates[
                    "gps"
                ] += 1

        # --------------------------------------------------------------
        # Vision-position update
        # --------------------------------------------------------------

        if sensor_measurements[
            "vision_position_available"
        ]:

            accepted = (
                self.position_filter
                .update_position(
                    sensor_measurements[
                        "vision_position"
                    ],
                    VISION_POSITION_NOISE_STD,
                )
            )

            if accepted:

                self.accepted_updates[
                    "vision_position"
                ] += 1

            else:

                self.rejected_updates[
                    "vision_position"
                ] += 1

        # --------------------------------------------------------------
        # Depth update
        # --------------------------------------------------------------

        if sensor_measurements[
            "depth_available"
        ]:

            accepted = (
                self.position_filter
                .update_depth(
                    sensor_measurements[
                        "depth"
                    ],
                    DEPTH_NOISE_STD,
                )
            )

            if accepted:

                self.accepted_updates[
                    "depth"
                ] += 1

            else:

                self.rejected_updates[
                    "depth"
                ] += 1

        # --------------------------------------------------------------
        # Sonar-like position update
        # --------------------------------------------------------------

        if sensor_measurements[
            "sonar_available"
        ]:

            accepted = (
                self.position_filter
                .update_position(
                    sensor_measurements[
                        "sonar_position"
                    ],
                    SONAR_POSITION_NOISE_STD,
                )
            )

            if accepted:

                self.accepted_updates[
                    "sonar"
                ] += 1

            else:

                self.rejected_updates[
                    "sonar"
                ] += 1

        # --------------------------------------------------------------
        # Visual attitude update
        # --------------------------------------------------------------

        if sensor_measurements[
            "vision_attitude_available"
        ]:

            innovation = (
                wrap_angle_vector(
                    sensor_measurements[
                        "vision_attitude"
                    ]
                    -
                    self.attitude_filter.angles
                )
            )

            if (
                np.linalg.norm(
                    innovation
                )
                <=
                np.deg2rad(
                    10.0
                )
            ):

                self.attitude_filter.update_visual_attitude(
                    sensor_measurements[
                        "vision_attitude"
                    ]
                )

                self.accepted_updates[
                    "vision_attitude"
                ] += 1

            else:

                self.rejected_updates[
                    "vision_attitude"
                ] += 1

    def confidence(
        self,
    ):
        """
        Combined translational estimator confidence.

        This method was missing from the previous replacement and is
        now explicitly defined.
        """

        return (
            self.position_filter.confidence()
        )


# ======================================================================
# MOTOR FUNCTIONS
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
        ],
        dtype=float,
    )

    desired = np.array(
        [
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        ],
        dtype=float,
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
    max_thrust = (
        maximum_motor_thrust()
    )

    requested_total = float(
        total_thrust
    )

    collective_clipped = (
        requested_total
        >
        4.0
        * max_thrust
    )

    total_thrust = float(
        np.clip(
            requested_total,
            0.0,
            4.0
            * max_thrust,
        )
    )

    torques = np.array(
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
            torques[0],
            torques[1],
            torques[2],
            arm_length,
        )
    )

    if (
        np.all(
            candidate >= 0.0
        )
        and
        np.all(
            candidate <= max_thrust
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

    for _ in range(
        40
    ):

        scale = (
            low + high
        ) / 2.0

        candidate = (
            calculate_motor_thrusts(
                total_thrust,
                torques[0] * scale,
                torques[1] * scale,
                torques[2] * scale,
                arm_length,
            )
        )

        if (
            np.all(
                candidate >= 0.0
            )
            and
            np.all(
                candidate <= max_thrust
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
            max_thrust,
        ),
        True,
        low,
        collective_clipped,
    )


def actual_torques(
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
        ],
        dtype=float,
    )


# ======================================================================
# FORCE / ATTITUDE
# ======================================================================

def force_to_desired_angles(
    force,
):
    magnitude = np.linalg.norm(
        force
    )

    if magnitude <= 1e-12:

        return np.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        )

    direction = (
        force
        / magnitude
    )

    roll = np.arctan2(
        -direction[1],
        np.sqrt(
            direction[0]**2
            + direction[2]**2
        ),
    )

    pitch = np.arctan2(
        direction[0],
        direction[2],
    )

    return np.array(
        [
            np.clip(
                roll,
                -MAX_ROLL,
                MAX_ROLL,
            ),
            np.clip(
                pitch,
                -MAX_PITCH,
                MAX_PITCH,
            ),
            0.0,
        ]
    )


def feasible_force(
    force,
):
    magnitude = np.linalg.norm(
        force
    )

    if magnitude <= 1e-12:

        return np.zeros(
            3
        )

    desired_angles = (
        force_to_desired_angles(
            force
        )
    )

    feasible_rotation = (
        rotation_matrix(
            desired_angles[0],
            desired_angles[1],
            desired_angles[2],
        )
    )

    body_z_world = (
        feasible_rotation[
            :,
            2
        ]
    )

    thrust = max(
        float(
            np.dot(
                force,
                body_z_world,
            )
        ),
        0.0,
    )

    return (
        thrust
        * body_z_world
    )


def morphology_aware_attitude_control(
    controller,
    desired_angles,
    estimated_angles,
    estimated_rates,
    inertia,
    inertia_rate,
    inertia_scale,
):
    base_torque = (
        controller.update(
            desired_angles,
            estimated_angles,
            estimated_rates,
        )
    )

    scaled_torque = (
        inertia_scale
        * base_torque
    )

    angular_momentum = (
        inertia
        @ estimated_rates
    )

    gyroscopic_term = (
        np.cross(
            estimated_rates,
            angular_momentum,
        )
    )

    inertia_rate_term = (
        inertia_rate
        @ estimated_rates
    )

    return (
        scaled_torque
        + gyroscopic_term
        + inertia_rate_term
    )


# ======================================================================
# SIMULATION
# ======================================================================

def run_simulation():

    true_position = (
        P_INITIAL.copy()
    )

    true_velocity = np.zeros(
        3
    )

    true_angles = np.zeros(
        3
    )

    true_angular_rates = np.zeros(
        3
    )

    previous_acceleration = np.zeros(
        3
    )

    motors = [
        Motor(),
        Motor(),
        Motor(),
        Motor(),
    ]

    sensor_simulator = (
        SensorSimulator()
    )

    estimator = (
        MultiModalStateEstimator()
    )

    max_motor_thrust = (
        maximum_motor_thrust()
    )

    max_arm_length = (
        ARM_LENGTH
        * EXTENDED_ARM_RATIO
    )

    max_arm = (
        max_arm_length
        / np.sqrt(2.0)
    )

    max_roll_torque = (
        2.0
        * max_arm
        * max_motor_thrust
    )

    max_pitch_torque = (
        2.0
        * max_arm
        * max_motor_thrust
    )

    max_yaw_torque = (
        2.0
        * KM
        * max_motor_thrust
    )

    attitude_controller = (
        AttitudeController(
            kp_roll=ATTITUDE_KP_ROLL,
            kp_pitch=ATTITUDE_KP_PITCH,
            kp_yaw=ATTITUDE_KP_YAW,

            kd_roll=ATTITUDE_KD_ROLL,
            kd_pitch=ATTITUDE_KD_PITCH,
            kd_yaw=ATTITUDE_KD_YAW,

            max_roll_torque=max_roll_torque,
            max_pitch_torque=max_pitch_torque,
            max_yaw_torque=max_yaw_torque,
        )
    )

    steps = int(
        SIMULATION_TIME
        / DT
    )

    time = (
        np.arange(
            steps
        )
        * DT
    )

    # ------------------------------------------------------------------
    # Vector histories
    # ------------------------------------------------------------------

    vector_history_names = [
        "position",
        "velocity",
        "angles",
        "angular_rates",

        "estimated_position",
        "estimated_velocity",
        "estimated_angles",
        "estimated_angular_rates",
        "estimated_acceleration_bias",

        "target_position",
        "target_velocity",
        "target_acceleration",

        "desired_angles",
        "required_force",
        "torque",
        "acceleration",
    ]

    history = {
        name: np.zeros(
            (
                steps,
                3,
            )
        )
        for name in vector_history_names
    }

    # ------------------------------------------------------------------
    # Scalar histories
    # ------------------------------------------------------------------

    scalar_names = [
        "requested_thrust",
        "commanded_thrust",
        "effective_thrust",

        "buoyancy",
        "drag",

        "immersion",
        "estimated_immersion",
        "propulsion_effectiveness",

        "morphology",
        "morphology_rate",
        "arm_length",
        "inertia_scale",

        "commanded_direction_error",
        "actual_direction_error",

        "torque_scale",

        "impact_force",
        "impact_normal_speed",

        "position_estimation_error",
        "velocity_estimation_error",
        "attitude_estimation_error",

        "estimator_confidence",
        "position_covariance_trace",
    ]

    scalar = {
        name: np.zeros(
            steps
        )
        for name in scalar_names
    }

    # ------------------------------------------------------------------
    # Actuator histories
    # ------------------------------------------------------------------

    rpm_history = np.zeros(
        (
            steps,
            4,
        )
    )

    motor_thrust_history = np.zeros(
        (
            steps,
            4,
        )
    )

    # ------------------------------------------------------------------
    # Boolean histories
    # ------------------------------------------------------------------

    bool_names = [
        "torque_saturation",
        "collective_clipped",
        "impact_active",

        "gps_available",
        "vision_position_available",
        "depth_available",
        "sonar_available",
        "vision_attitude_available",
    ]

    boolean = {
        name: np.zeros(
            steps,
            dtype=bool,
        )
        for name in bool_names
    }

    # ==================================================================
    # LOOP
    # ==================================================================

    for i, t in enumerate(
        time
    ):

        # --------------------------------------------------------------
        # Reference trajectory
        # --------------------------------------------------------------

        (
            target_position,
            target_velocity,
            target_acceleration,
        ) = mission_trajectory(
            t
        )

        history[
            "target_position"
        ][i] = (
            target_position
        )

        history[
            "target_velocity"
        ][i] = (
            target_velocity
        )

        history[
            "target_acceleration"
        ][i] = (
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
        ) = morphology_parameters(
            t
        )

        scalar[
            "morphology"
        ][i] = morphology_state

        scalar[
            "morphology_rate"
        ][i] = morphology_rate

        scalar[
            "arm_length"
        ][i] = current_arm_length

        scalar[
            "inertia_scale"
        ][i] = inertia_scale

        # --------------------------------------------------------------
        # TRUE medium
        # --------------------------------------------------------------

        (
            true_immersion,
            true_buoyancy_force,
            true_drag_force,
            true_drag_magnitude,
            true_propulsion_effectiveness,
        ) = medium_effects(
            true_position,
            true_velocity,
        )

        scalar[
            "immersion"
        ][i] = true_immersion

        scalar[
            "buoyancy"
        ][i] = true_buoyancy_force[2]

        scalar[
            "drag"
        ][i] = true_drag_magnitude

        scalar[
            "propulsion_effectiveness"
        ][i] = true_propulsion_effectiveness

        # --------------------------------------------------------------
        # Sensors
        # --------------------------------------------------------------

        measurements = (
            sensor_simulator.measure(
                t,
                true_position,
                true_velocity,
                true_angles,
                true_angular_rates,
                previous_acceleration,
                true_immersion,
            )
        )

        for key in [
            "gps_available",
            "vision_position_available",
            "depth_available",
            "sonar_available",
            "vision_attitude_available",
        ]:

            boolean[
                key
            ][i] = measurements[
                key
            ]

        # --------------------------------------------------------------
        # State estimation
        # --------------------------------------------------------------

        estimator.update(
            measurements,
            DT,
        )

        estimated_position = (
            estimator.position
        )

        estimated_velocity = (
            estimator.velocity
        )

        estimated_angles = (
            estimator.angles
        )

        estimated_rates = (
            estimator.angular_rates
        )

        estimated_bias_body = (
            estimator.acceleration_bias_body
        )

        history[
            "estimated_position"
        ][i] = estimated_position

        history[
            "estimated_velocity"
        ][i] = estimated_velocity

        history[
            "estimated_angles"
        ][i] = estimated_angles

        history[
            "estimated_angular_rates"
        ][i] = estimated_rates

        history[
            "estimated_acceleration_bias"
        ][i] = estimated_bias_body

        # --------------------------------------------------------------
        # Estimation errors
        # --------------------------------------------------------------

        scalar[
            "position_estimation_error"
        ][i] = (
            np.linalg.norm(
                estimated_position
                - true_position
            )
        )

        scalar[
            "velocity_estimation_error"
        ][i] = (
            np.linalg.norm(
                estimated_velocity
                - true_velocity
            )
        )

        scalar[
            "attitude_estimation_error"
        ][i] = np.rad2deg(
            np.linalg.norm(
                wrap_angle_vector(
                    estimated_angles
                    - true_angles
                )
            )
        )

        scalar[
            "estimator_confidence"
        ][i] = (
            estimator.confidence()
        )

        scalar[
            "position_covariance_trace"
        ][i] = float(
            np.trace(
                estimator
                .position_filter
                .P[
                    :3,
                    :3
                ]
            )
        )

        # --------------------------------------------------------------
        # Estimated medium
        # --------------------------------------------------------------

        (
            estimated_immersion,
            estimated_buoyancy_force,
            estimated_drag_force,
            _estimated_drag_magnitude,
            estimated_propulsion_effectiveness,
        ) = medium_effects(
            estimated_position,
            estimated_velocity,
        )

        scalar[
            "estimated_immersion"
        ][i] = estimated_immersion

        # --------------------------------------------------------------
        # Water-entry disturbance
        # --------------------------------------------------------------

        (
            impact_force,
            impact_magnitude,
            impact_normal_speed,
            _impact_rate,
            impact_active,
        ) = water_entry_impact(
            t,
            true_position,
            true_velocity,
        )

        scalar[
            "impact_force"
        ][i] = impact_magnitude

        scalar[
            "impact_normal_speed"
        ][i] = impact_normal_speed

        boolean[
            "impact_active"
        ][i] = impact_active

        # ==============================================================
        # ESTIMATED-STATE POSITION CONTROL
        # ==============================================================

        position_error = (
            target_position
            - estimated_position
        )

        velocity_error = (
            target_velocity
            - estimated_velocity
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

        # --------------------------------------------------------------
        # Horizontal limit
        # --------------------------------------------------------------

        horizontal_acceleration = (
            feedback_acceleration[
                :2
            ]
        )

        horizontal_norm = (
            np.linalg.norm(
                horizontal_acceleration
            )
        )

        if (
            horizontal_norm
            >
            MAX_HORIZONTAL_ACCELERATION
        ):

            horizontal_acceleration *= (
                MAX_HORIZONTAL_ACCELERATION
                / horizontal_norm
            )

        feedback_acceleration[
            0
        ] = horizontal_acceleration[
            0
        ]

        feedback_acceleration[
            1
        ] = horizontal_acceleration[
            1
        ]

        # --------------------------------------------------------------
        # Vertical limit
        # --------------------------------------------------------------

        feedback_acceleration[
            2
        ] = np.clip(
            feedback_acceleration[
                2
            ],
            -MAX_VERTICAL_ACCELERATION,
            MAX_VERTICAL_ACCELERATION,
        )

        gravity_force = np.array(
            [
                0.0,
                0.0,
                -MASS * GRAVITY,
            ]
        )

        # --------------------------------------------------------------
        # Required controller force
        #
        # Water-entry impact remains excluded from controller
        # feedforward.
        # --------------------------------------------------------------

        raw_required_force = (
            MASS
            * feedback_acceleration
            - gravity_force
            - estimated_buoyancy_force
            - estimated_drag_force
        )

        required_thrust_world = (
            feasible_force(
                raw_required_force
            )
        )

        history[
            "required_force"
        ][i] = required_thrust_world

        required_thrust_magnitude = (
            np.linalg.norm(
                required_thrust_world
            )
        )

        # --------------------------------------------------------------
        # Desired attitude
        # --------------------------------------------------------------

        desired_angles = (
            force_to_desired_angles(
                required_thrust_world
            )
        )

        history[
            "desired_angles"
        ][i] = desired_angles

        # --------------------------------------------------------------
        # Attitude controller
        # --------------------------------------------------------------

        commanded_torque = (
            morphology_aware_attitude_control(
                attitude_controller,
                desired_angles,
                estimated_angles,
                estimated_rates,
                current_inertia,
                inertia_rate,
                inertia_scale,
            )
        )

        # --------------------------------------------------------------
        # Direction error
        # --------------------------------------------------------------

        if (
            required_thrust_magnitude
            >
            1e-12
        ):

            required_direction = (
                required_thrust_world
                / required_thrust_magnitude
            )

            desired_rotation = (
                rotation_matrix(
                    desired_angles[0],
                    desired_angles[1],
                    desired_angles[2],
                )
            )

            true_rotation = (
                rotation_matrix(
                    true_angles[0],
                    true_angles[1],
                    true_angles[2],
                )
            )

            desired_body_z = (
                desired_rotation[
                    :,
                    2
                ]
            )

            true_body_z = (
                true_rotation[
                    :,
                    2
                ]
            )

            commanded_dot = np.clip(
                np.dot(
                    desired_body_z,
                    required_direction,
                ),
                -1.0,
                1.0,
            )

            actual_dot = np.clip(
                np.dot(
                    true_body_z,
                    required_direction,
                ),
                -1.0,
                1.0,
            )

            scalar[
                "commanded_direction_error"
            ][i] = np.rad2deg(
                np.arccos(
                    commanded_dot
                )
            )

            scalar[
                "actual_direction_error"
            ][i] = np.rad2deg(
                np.arccos(
                    actual_dot
                )
            )

        # --------------------------------------------------------------
        # Cross-medium propulsion
        # --------------------------------------------------------------

        requested_aerial_equivalent_thrust = (
            required_thrust_magnitude
            / max(
                estimated_propulsion_effectiveness,
                0.05,
            )
        )

        scalar[
            "requested_thrust"
        ][i] = (
            requested_aerial_equivalent_thrust
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
            requested_aerial_equivalent_thrust,
            commanded_torque[0],
            commanded_torque[1],
            commanded_torque[2],
            current_arm_length,
        )

        scalar[
            "commanded_thrust"
        ][i] = np.sum(
            commanded_motor_thrusts
        )

        scalar[
            "torque_scale"
        ][i] = torque_scale

        boolean[
            "torque_saturation"
        ][i] = torque_saturated

        boolean[
            "collective_clipped"
        ][i] = collective_clipped

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

        rpm_history[
            i
        ] = actual_rpms

        motor_thrust_history[
            i
        ] = actual_motor_thrusts

        effective_motor_thrusts = (
            actual_motor_thrusts
            * true_propulsion_effectiveness
        )

        effective_total_thrust = (
            np.sum(
                effective_motor_thrusts
            )
        )

        scalar[
            "effective_thrust"
        ][i] = (
            effective_total_thrust
        )

        # --------------------------------------------------------------
        # Plant torque
        # --------------------------------------------------------------

        actual_aerial_torque = (
            actual_torques(
                actual_motor_thrusts,
                current_arm_length,
            )
        )

        actual_effective_torque = (
            actual_aerial_torque
            * true_propulsion_effectiveness
        )

        water_damping_torque = (
            -WATER_ROTATIONAL_DAMPING
            * true_immersion
            * true_angular_rates
        )

        total_torque = (
            actual_effective_torque
            + water_damping_torque
        )

        history[
            "torque"
        ][i] = total_torque

        # ==============================================================
        # TRUE TRANSLATIONAL DYNAMICS
        # ==============================================================

        true_rotation = (
            rotation_matrix(
                true_angles[0],
                true_angles[1],
                true_angles[2],
            )
        )

        thrust_body = np.array(
            [
                0.0,
                0.0,
                effective_total_thrust,
            ]
        )

        thrust_world = (
            true_rotation
            @ thrust_body
        )

        total_force = (
            thrust_world
            + gravity_force
            + true_buoyancy_force
            + true_drag_force
            + impact_force
        )

        true_acceleration = (
            total_force
            / MASS
        )

        history[
            "acceleration"
        ][i] = true_acceleration

        true_velocity += (
            true_acceleration
            * DT
        )

        true_position += (
            true_velocity
            * DT
        )

        # ==============================================================
        # TRUE ROTATIONAL DYNAMICS
        # ==============================================================

        angular_momentum = (
            current_inertia
            @ true_angular_rates
        )

        angular_acceleration = (
            np.linalg.solve(
                current_inertia,
                total_torque
                - (
                    inertia_rate
                    @ true_angular_rates
                )
                - np.cross(
                    true_angular_rates,
                    angular_momentum,
                ),
            )
        )

        true_angular_rates += (
            angular_acceleration
            * DT
        )

        true_angles += (
            euler_rate_matrix(
                true_angles
            )
            @ true_angular_rates
            * DT
        )

        true_angles = (
            wrap_angle_vector(
                true_angles
            )
        )

        # --------------------------------------------------------------
        # Store true state
        # --------------------------------------------------------------

        history[
            "position"
        ][i] = true_position

        history[
            "velocity"
        ][i] = true_velocity

        history[
            "angles"
        ][i] = true_angles

        history[
            "angular_rates"
        ][i] = true_angular_rates

        previous_acceleration = (
            true_acceleration.copy()
        )

    return {
        "time":
            time,

        **history,
        **scalar,

        "rpm":
            rpm_history,

        "motor_thrust":
            motor_thrust_history,

        **boolean,
    }


# ======================================================================
# METRIC HELPERS
# ======================================================================

def masked_max(
    values,
    mask,
):
    if np.any(
        mask
    ):

        return float(
            np.max(
                values[
                    mask
                ]
            )
        )

    return 0.0


def masked_rms(
    values,
    mask,
):
    if np.any(
        mask
    ):

        return float(
            np.sqrt(
                np.mean(
                    values[
                        mask
                    ]**2
                )
            )
        )

    return 0.0


def scheduled_sensor_mask(
    time,
    rate_hz,
):
    return np.array(
        [
            periodic_update(
                t,
                rate_hz,
            )
            for t in time
        ],
        dtype=bool,
    )


def availability_percentage(
    available,
    scheduled,
    applicable_mask=None,
):
    effective_schedule = (
        scheduled.copy()
    )

    if applicable_mask is not None:

        effective_schedule &= (
            applicable_mask
        )

    scheduled_count = (
        np.count_nonzero(
            effective_schedule
        )
    )

    if scheduled_count == 0:

        return 0.0

    available_count = (
        np.count_nonzero(
            available
            &
            effective_schedule
        )
    )

    return float(
        100.0
        * available_count
        / scheduled_count
    )


# ======================================================================
# IMPACT RECOVERY METRICS
# ======================================================================

def calculate_impact_recovery_metrics(
    time,
    tracking_error,
    impact_force,
    impact_active,
):
    active_indices = np.flatnonzero(
        impact_active
    )

    if active_indices.size == 0:

        return {
            "Impact pre-event RMS error (m)":
                np.nan,

            "Peak post-impact error (m)":
                np.nan,

            "Peak impact-induced error excursion (m)":
                np.nan,

            "Impact recovery threshold (m)":
                np.nan,

            "Impact response peak time (s)":
                np.nan,

            "Impact recovery time to 10% residual (s)":
                np.nan,
        }

    first_index = int(
        active_indices[0]
    )

    first_time = float(
        time[
            first_index
        ]
    )

    baseline_mask = (
        (
            time
            >= max(
                0.0,
                first_time
                - IMPACT_BASELINE_WINDOW,
            )
        )
        &
        (
            time
            < first_time
        )
    )

    if np.any(
        baseline_mask
    ):

        baseline_error = float(
            np.sqrt(
                np.mean(
                    tracking_error[
                        baseline_mask
                    ]**2
                )
            )
        )

    else:

        baseline_error = float(
            tracking_error[
                first_index
            ]
        )

    peak_force_index = int(
        np.argmax(
            impact_force
        )
    )

    if (
        impact_force[
            peak_force_index
        ]
        <=
        0.0
    ):

        return {
            "Impact pre-event RMS error (m)":
                baseline_error,

            "Peak post-impact error (m)":
                np.nan,

            "Peak impact-induced error excursion (m)":
                np.nan,

            "Impact recovery threshold (m)":
                np.nan,

            "Impact response peak time (s)":
                np.nan,

            "Impact recovery time to 10% residual (s)":
                np.nan,
        }

    peak_force_time = float(
        time[
            peak_force_index
        ]
    )

    response_mask = (
        (
            time
            >= peak_force_time
        )
        &
        (
            time
            <=
            peak_force_time
            + IMPACT_RESPONSE_WINDOW
        )
    )

    response_indices = np.flatnonzero(
        response_mask
    )

    if response_indices.size == 0:

        return {
            "Impact pre-event RMS error (m)":
                baseline_error,

            "Peak post-impact error (m)":
                np.nan,

            "Peak impact-induced error excursion (m)":
                np.nan,

            "Impact recovery threshold (m)":
                np.nan,

            "Impact response peak time (s)":
                np.nan,

            "Impact recovery time to 10% residual (s)":
                np.nan,
        }

    peak_error_index = int(
        response_indices[
            np.argmax(
                tracking_error[
                    response_indices
                ]
            )
        ]
    )

    peak_error = float(
        tracking_error[
            peak_error_index
        ]
    )

    peak_error_time = float(
        time[
            peak_error_index
        ]
    )

    excursion = max(
        peak_error
        - baseline_error,
        0.0,
    )

    threshold = (
        baseline_error
        + (
            IMPACT_RECOVERY_RESIDUAL_FRACTION
            * excursion
        )
    )

    persistence_steps = max(
        1,
        int(
            np.ceil(
                IMPACT_RECOVERY_PERSISTENCE_TIME
                / DT
            )
        ),
    )

    recovery_time = np.nan

    for index in range(
        peak_error_index + 1,
        len(time)
        - persistence_steps
        + 1,
    ):

        window = (
            tracking_error[
                index:
                index
                + persistence_steps
            ]
        )

        if np.all(
            window <= threshold
        ):

            recovery_time = float(
                time[
                    index
                ]
                - peak_error_time
            )

            break

    return {
        "Impact pre-event RMS error (m)":
            baseline_error,

        "Peak post-impact error (m)":
            peak_error,

        "Peak impact-induced error excursion (m)":
            excursion,

        "Impact recovery threshold (m)":
            threshold,

        "Impact response peak time (s)":
            peak_error_time,

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

    tracking_error_vector = (
        target_position
        - position
    )

    tracking_error = (
        np.linalg.norm(
            tracking_error_vector,
            axis=1,
        )
    )

    horizontal_error = (
        np.linalg.norm(
            tracking_error_vector[
                :,
                :2
            ],
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

    position_estimation_error = (
        results[
            "position_estimation_error"
        ]
    )

    velocity_estimation_error = (
        results[
            "velocity_estimation_error"
        ]
    )

    underwater_mask = (
        time
        >= SUBMERGED_STABILIZATION_START
    )

    water_entry_mask = (
        (
            time
            >= WATER_DESCENT_START
        )
        &
        (
            time
            < WATER_DESCENT_END
        )
    )

    submerged_stabilization_mask = (
        (
            time
            >= SUBMERGED_STABILIZATION_START
        )
        &
        (
            time
            < SUBMERGED_STABILIZATION_END
        )
    )

    trajectory_1_mask = (
        (
            time
            >= TRAJECTORY_1_START
        )
        &
        (
            time
            < TRAJECTORY_1_END
        )
    )

    trajectory_2_mask = (
        (
            time
            >= TRAJECTORY_2_START
        )
        &
        (
            time
            < TRAJECTORY_2_END
        )
    )

    return_mask = (
        (
            time
            >= RETURN_START
        )
        &
        (
            time
            < RETURN_END
        )
    )

    final_mask = (
        time
        >= FINAL_HOLD_START
    )

    gps_denied_mask = (
        time
        >= GPS_HANDOFF_END
    )

    # ------------------------------------------------------------------
    # Sensor schedules
    # ------------------------------------------------------------------

    scheduled_gps = (
        scheduled_sensor_mask(
            time,
            GPS_RATE_HZ,
        )
    )

    scheduled_vision_position = (
        scheduled_sensor_mask(
            time,
            VISION_POSITION_RATE_HZ,
        )
    )

    scheduled_depth = (
        scheduled_sensor_mask(
            time,
            DEPTH_RATE_HZ,
        )
    )

    scheduled_sonar = (
        scheduled_sensor_mask(
            time,
            SONAR_RATE_HZ,
        )
    )

    scheduled_vision_attitude = (
        scheduled_sensor_mask(
            time,
            VISION_ATTITUDE_RATE_HZ,
        )
    )

    # ------------------------------------------------------------------
    # Sensor applicability
    # ------------------------------------------------------------------

    gps_aerial_applicable = (
        time < GPS_HANDOFF_END
    )

    vision_position_applicable = (
        results[
            "immersion"
        ]
        <= VISION_POSITION_MAX_IMMERSION
    )

    depth_applicable = (
        results[
            "immersion"
        ]
        >= DEPTH_MIN_IMMERSION
    )

    sonar_underwater_applicable = (
        underwater_mask
        &
        (
            results[
                "immersion"
            ]
            >= SONAR_MIN_IMMERSION
        )
    )

    vision_attitude_applicable = (
        results[
            "immersion"
        ]
        <= VISION_ATTITUDE_MAX_IMMERSION
    )

    # ------------------------------------------------------------------
    # Impact
    # ------------------------------------------------------------------

    impact_metrics = (
        calculate_impact_recovery_metrics(
            time,
            tracking_error,
            results[
                "impact_force"
            ],
            results[
                "impact_active"
            ],
        )
    )

    # ------------------------------------------------------------------
    # Bias
    # ------------------------------------------------------------------

    estimated_bias_magnitude = (
        np.linalg.norm(
            results[
                "estimated_acceleration_bias"
            ],
            axis=1,
        )
    )

    injected_bias_magnitude = float(
        np.linalg.norm(
            IMU_ACCEL_BIAS_BODY
        )
    )

    # ------------------------------------------------------------------
    # Underwater / GPS-denied estimation
    # ------------------------------------------------------------------

    underwater_estimation_error = (
        position_estimation_error[
            underwater_mask
        ]
    )

    gps_denied_estimation_error = (
        position_estimation_error[
            gps_denied_mask
        ]
    )

    # ------------------------------------------------------------------
    # Final state
    # ------------------------------------------------------------------

    final_position = (
        position[-1]
    )

    final_velocity = (
        velocity[-1]
    )

    final_angles = (
        angles[-1]
    )

    final_target_error = (
        target_position[-1]
        - final_position
    )

    final_estimation_error = (
        results[
            "estimated_position"
        ][-1]
        - final_position
    )

    maximum_motor_thrust_used = float(
        np.max(
            results[
                "motor_thrust"
            ]
        )
    )

    maximum_rpm = float(
        np.max(
            results[
                "rpm"
            ]
        )
    )

    available_motor_thrust = (
        maximum_motor_thrust()
    )

    metrics = {

        # ==============================================================
        # Tracking
        # ==============================================================

        "Maximum 3-D tracking error (m)":
            float(
                np.max(
                    tracking_error
                )
            ),

        "RMS 3-D tracking error (m)":
            float(
                np.sqrt(
                    np.mean(
                        tracking_error**2
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
            masked_max(
                tracking_error,
                time < WATER_DESCENT_START,
            ),

        "Maximum water-entry error (m)":
            masked_max(
                tracking_error,
                water_entry_mask,
            ),

        "Maximum submerged-stabilization error (m)":
            masked_max(
                tracking_error,
                submerged_stabilization_mask,
            ),

        "RMS submerged-stabilization error (m)":
            masked_rms(
                tracking_error,
                submerged_stabilization_mask,
            ),

        "Maximum trajectory-1 error (m)":
            masked_max(
                tracking_error,
                trajectory_1_mask,
            ),

        "Maximum trajectory-2 error (m)":
            masked_max(
                tracking_error,
                trajectory_2_mask,
            ),

        "Maximum return-phase error (m)":
            masked_max(
                tracking_error,
                return_mask,
            ),

        "Maximum final-stabilization error (m)":
            masked_max(
                tracking_error,
                final_mask,
            ),

        "RMS final-stabilization error (m)":
            masked_rms(
                tracking_error,
                final_mask,
            ),

        "Maximum final-stabilization speed (m/s)":
            masked_max(
                speed,
                final_mask,
            ),

        "Maximum underwater speed (m/s)":
            masked_max(
                speed,
                underwater_mask,
            ),

        # ==============================================================
        # Attitude
        # ==============================================================

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

        # ==============================================================
        # Medium
        # ==============================================================

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
                    results[
                        "buoyancy"
                    ]
                )
            ),

        "Maximum hydrodynamic drag (N)":
            float(
                np.max(
                    results[
                        "drag"
                    ]
                )
            ),

        # ==============================================================
        # Water entry
        # ==============================================================

        "Maximum water-entry impact force (N)":
            float(
                np.max(
                    results[
                        "impact_force"
                    ]
                )
            ),

        "Water-entry impact impulse (N s)":
            float(
                np.trapz(
                    results[
                        "impact_force"
                    ],
                    time,
                )
            ),

        "Maximum water-entry normal speed (m/s)":
            float(
                np.max(
                    results[
                        "impact_normal_speed"
                    ]
                )
            ),

        "Water-entry active duration (s)":
            float(
                np.count_nonzero(
                    results[
                        "impact_active"
                    ]
                )
                * DT
            ),

        "Peak water-entry acceleration (m/s^2)":
            masked_max(
                np.linalg.norm(
                    results[
                        "acceleration"
                    ],
                    axis=1,
                ),
                water_entry_mask,
            ),

        "Water-entry impact peak time (s)":
            float(
                time[
                    np.argmax(
                        results[
                            "impact_force"
                        ]
                    )
                ]
            ),

        **impact_metrics,

        "Maximum post-impact stabilization error (m)":
            masked_max(
                tracking_error,
                submerged_stabilization_mask,
            ),

        # ==============================================================
        # Actuation
        # ==============================================================

        "Maximum requested aerial-equivalent thrust (N)":
            float(
                np.max(
                    results[
                        "requested_thrust"
                    ]
                )
            ),

        "Maximum commanded aerial thrust (N)":
            float(
                np.max(
                    results[
                        "commanded_thrust"
                    ]
                )
            ),

        "Maximum effective thrust (N)":
            float(
                np.max(
                    results[
                        "effective_thrust"
                    ]
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
                available_motor_thrust
                - maximum_motor_thrust_used
            ),

        "Torque saturation events":
            int(
                np.count_nonzero(
                    results[
                        "torque_saturation"
                    ]
                )
            ),

        "Collective thrust clipping events":
            int(
                np.count_nonzero(
                    results[
                        "collective_clipped"
                    ]
                )
            ),

        "Minimum torque allocation scale":
            float(
                np.min(
                    results[
                        "torque_scale"
                    ]
                )
            ),

        "Maximum commanded force-direction error (deg)":
            float(
                np.max(
                    results[
                        "commanded_direction_error"
                    ]
                )
            ),

        "Maximum actual thrust-direction error (deg)":
            float(
                np.max(
                    results[
                        "actual_direction_error"
                    ]
                )
            ),

        # ==============================================================
        # Morphology
        # ==============================================================

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

        # ==============================================================
        # State estimation
        # ==============================================================

        "Maximum position estimation error (m)":
            float(
                np.max(
                    position_estimation_error
                )
            ),

        "RMS position estimation error (m)":
            float(
                np.sqrt(
                    np.mean(
                        position_estimation_error**2
                    )
                )
            ),

        "Maximum underwater position estimation error (m)":
            float(
                np.max(
                    underwater_estimation_error
                )
            ),

        "RMS underwater position estimation error (m)":
            float(
                np.sqrt(
                    np.mean(
                        underwater_estimation_error**2
                    )
                )
            ),

        "Maximum GPS-denied position estimation error (m)":
            float(
                np.max(
                    gps_denied_estimation_error
                )
            ),

        "RMS GPS-denied position estimation error (m)":
            float(
                np.sqrt(
                    np.mean(
                        gps_denied_estimation_error**2
                    )
                )
            ),

        "Maximum velocity estimation error (m/s)":
            float(
                np.max(
                    velocity_estimation_error
                )
            ),

        "RMS velocity estimation error (m/s)":
            float(
                np.sqrt(
                    np.mean(
                        velocity_estimation_error**2
                    )
                )
            ),

        "Maximum estimated accel-bias magnitude (m/s^2)":
            float(
                np.max(
                    estimated_bias_magnitude
                )
            ),

        "Injected accel-bias magnitude (m/s^2)":
            injected_bias_magnitude,

        "Final estimated accel-bias magnitude (m/s^2)":
            float(
                estimated_bias_magnitude[-1]
            ),

        "Maximum attitude estimation error (deg)":
            float(
                np.max(
                    results[
                        "attitude_estimation_error"
                    ]
                )
            ),

        "RMS attitude estimation error (deg)":
            float(
                np.sqrt(
                    np.mean(
                        results[
                            "attitude_estimation_error"
                        ]**2
                    )
                )
            ),

        "Minimum estimator confidence":
            float(
                np.min(
                    results[
                        "estimator_confidence"
                    ]
                )
            ),

        "Mean estimator confidence":
            float(
                np.mean(
                    results[
                        "estimator_confidence"
                    ]
                )
            ),

        # ==============================================================
        # Sensor availability
        #
        # These are percentages of scheduled opportunities in the
        # applicable operating regime, not percentages of all
        # simulation time steps.
        # ==============================================================

        "Aerial GPS availability (%)":
            availability_percentage(
                results[
                    "gps_available"
                ],
                scheduled_gps,
                gps_aerial_applicable,
            ),

        "GPS availability underwater (%)":
            availability_percentage(
                results[
                    "gps_available"
                ],
                scheduled_gps,
                underwater_mask,
            ),

        "Vision position availability (%)":
            availability_percentage(
                results[
                    "vision_position_available"
                ],
                scheduled_vision_position,
                vision_position_applicable,
            ),

        "Depth sensor availability (%)":
            availability_percentage(
                results[
                    "depth_available"
                ],
                scheduled_depth,
                depth_applicable,
            ),

        "Sonar-like availability underwater (%)":
            availability_percentage(
                results[
                    "sonar_available"
                ],
                scheduled_sonar,
                sonar_underwater_applicable,
            ),

        "Vision attitude availability (%)":
            availability_percentage(
                results[
                    "vision_attitude_available"
                ],
                scheduled_vision_attitude,
                vision_attitude_applicable,
            ),

        # ==============================================================
        # Final state
        # ==============================================================

        "Final position estimation error (m)":
            float(
                np.linalg.norm(
                    final_estimation_error
                )
            ),

        "Final position error (m)":
            float(
                np.linalg.norm(
                    final_target_error
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

    return metrics


# ======================================================================
# PLOT HELPER
# ======================================================================

def save_figure(
    fig,
    filename,
):
    path = os.path.join(
        RESULTS_DIRECTORY,
        filename,
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ======================================================================
# POSITION PLOT
# ======================================================================

def plot_position(
    results,
):
    time = results[
        "time"
    ]

    true_position = results[
        "position"
    ]

    estimated_position = results[
        "estimated_position"
    ]

    target_position = results[
        "target_position"
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            13,
            10,
        ),
        sharex=True,
    )

    labels = [
        (
            "X",
            0,
        ),
        (
            "Y",
            1,
        ),
        (
            "Z",
            2,
        ),
    ]

    for (
        ax,
        (
            label,
            index,
        ),
    ) in zip(
        axes,
        labels,
    ):

        ax.plot(
            time,
            true_position[
                :,
                index
            ],
            label=f"True {label}",
        )

        ax.plot(
            time,
            estimated_position[
                :,
                index
            ],
            ":",
            label=f"Estimated {label}",
        )

        ax.plot(
            time,
            target_position[
                :,
                index
            ],
            "--",
            label=f"Target {label}",
        )

        if index == 2:

            ax.axhline(
                WATER_SURFACE_Z,
                linestyle="-.",
                label="Water surface",
            )

        ax.set_ylabel(
            f"{label} Position (m)"
        )

        ax.grid(
            True
        )

        ax.legend()

    axes[
        -1
    ].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        "MorphoAqua - Stage 5B-3 "
        "True, Estimated and Target Position"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B3_position_estimation_tracking.png",
    )


# ======================================================================
# 3-D TRAJECTORY
# ======================================================================

def plot_3d_trajectory(
    results,
):
    true_position = results[
        "position"
    ]

    estimated_position = results[
        "estimated_position"
    ]

    target_position = results[
        "target_position"
    ]

    fig = plt.figure(
        figsize=(
            12,
            10,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        true_position[:, 0],
        true_position[:, 1],
        true_position[:, 2],
        label="True trajectory",
    )

    ax.plot(
        estimated_position[:, 0],
        estimated_position[:, 1],
        estimated_position[:, 2],
        ":",
        label="Estimated trajectory",
    )

    ax.plot(
        target_position[:, 0],
        target_position[:, 1],
        target_position[:, 2],
        "--",
        label="Target trajectory",
    )

    ax.scatter(
        [
            true_position[
                0,
                0
            ]
        ],
        [
            true_position[
                0,
                1
            ]
        ],
        [
            true_position[
                0,
                2
            ]
        ],
        s=60,
        label="Mission start",
    )

    ax.scatter(
        [
            true_position[
                -1,
                0
            ]
        ],
        [
            true_position[
                -1,
                1
            ]
        ],
        [
            true_position[
                -1,
                2
            ]
        ],
        s=60,
        label="Final true state",
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
        "MorphoAqua - Stage 5B-3 "
        "True vs Estimated 3-D Trajectory"
    )

    ax.legend()

    ax.grid(
        True
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B3_3D_true_estimated_trajectory.png",
    )


# ======================================================================
# CONTROL / MEDIUM RESPONSE
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

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(
            13,
            23,
        ),
        sharex=True,
    )

    # Velocity
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

    # Attitude
    axes[1].plot(
        time,
        np.rad2deg(
            angles[:, 0]
        ),
        label="True Roll",
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
        label="True Pitch",
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

    # Thrust
    axes[2].plot(
        time,
        results[
            "requested_thrust"
        ],
        label="Requested aerial-equivalent thrust",
    )

    axes[2].plot(
        time,
        results[
            "commanded_thrust"
        ],
        "--",
        label="Commanded aerial thrust",
    )

    axes[2].plot(
        time,
        results[
            "effective_thrust"
        ],
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

    # Torque
    axes[3].plot(
        time,
        results[
            "torque"
        ][:, 0],
        label="Roll torque",
    )

    axes[3].plot(
        time,
        results[
            "torque"
        ][:, 1],
        label="Pitch torque",
    )

    axes[3].plot(
        time,
        results[
            "torque"
        ][:, 2],
        label="Yaw torque",
    )

    axes[3].set_ylabel(
        "Torque (N m)"
    )

    axes[3].legend()
    axes[3].grid(True)

    # RPM
    for motor_index in range(
        4
    ):

        axes[4].plot(
            time,
            results[
                "rpm"
            ][:, motor_index],
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

    # Medium
    axes[5].plot(
        time,
        results[
            "immersion"
        ],
        label="True immersion",
    )

    axes[5].plot(
        time,
        results[
            "estimated_immersion"
        ],
        ":",
        label="Estimated immersion",
    )

    axes[5].plot(
        time,
        results[
            "propulsion_effectiveness"
        ],
        "--",
        label="Propulsion effectiveness",
    )

    axes[5].set_ylabel(
        "Medium state"
    )

    axes[5].legend()
    axes[5].grid(True)

    # Forces
    axes[6].plot(
        time,
        results[
            "buoyancy"
        ],
        label="Buoyancy",
    )

    axes[6].plot(
        time,
        results[
            "drag"
        ],
        "--",
        label="Hydrodynamic drag",
    )

    axes[6].plot(
        time,
        results[
            "impact_force"
        ],
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
        "MorphoAqua - Stage 5B-3 "
        "Control and Medium Response"
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
        "Stage_5B3_control_medium_response.png",
    )


# ======================================================================
# MORPHOLOGY PLOT
# ======================================================================

def plot_morphology(
    results,
):
    time = results[
        "time"
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            13,
            8,
        ),
        sharex=True,
    )

    axes[0].plot(
        time,
        results[
            "arm_length"
        ],
        label="Arm length",
    )

    axes[0].plot(
        time,
        results[
            "morphology"
        ],
        "--",
        label="Morphology state",
    )

    axes[0].plot(
        time,
        results[
            "morphology_rate"
        ],
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
        results[
            "inertia_scale"
        ],
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
        "MorphoAqua - Stage 5B-3 "
        "Morphology and Time-Varying Inertia"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B3_morphology_inertia.png",
    )


# ======================================================================
# DIRECTION / IMPACT PLOT
# ======================================================================

def plot_direction_impact(
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
        figsize=(
            13,
            12,
        ),
        sharex=True,
    )

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
            "actual_direction_error"
        ],
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
        "Impact / speed"
    )

    axes[1].legend()
    axes[1].grid(True)

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
        "MorphoAqua - Stage 5B-3 "
        "Direction Feasibility and Water-Entry Response"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B3_direction_impact_response.png",
    )


# ======================================================================
# SENSOR / ESTIMATION PLOT
# ======================================================================

def plot_sensor_estimation(
    results,
):
    time = results[
        "time"
    ]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            13,
            18,
        ),
        sharex=True,
    )

    # --------------------------------------------------------------
    # Position / velocity estimation
    # --------------------------------------------------------------

    axes[0].plot(
        time,
        results[
            "position_estimation_error"
        ],
        label="3-D position estimation error",
    )

    axes[0].plot(
        time,
        results[
            "velocity_estimation_error"
        ],
        "--",
        label="Velocity estimation error",
    )

    axes[0].set_ylabel(
        "Error"
    )

    axes[0].legend()
    axes[0].grid(True)

    # --------------------------------------------------------------
    # Attitude
    # --------------------------------------------------------------

    axes[1].plot(
        time,
        results[
            "attitude_estimation_error"
        ],
        label="Attitude estimation error",
    )

    axes[1].set_ylabel(
        "Angle error (deg)"
    )

    axes[1].legend()
    axes[1].grid(True)

    # --------------------------------------------------------------
    # Sensor availability
    # --------------------------------------------------------------

    sensor_curves = [
        (
            "gps_available",
            "GPS",
        ),
        (
            "vision_position_available",
            "Vision position",
        ),
        (
            "depth_available",
            "Depth",
        ),
        (
            "sonar_available",
            "Sonar-like position",
        ),
        (
            "vision_attitude_available",
            "Vision attitude",
        ),
    ]

    for (
        key,
        label,
    ) in sensor_curves:

        axes[2].plot(
            time,
            results[
                key
            ].astype(
                float
            ),
            label=label,
        )

    axes[2].set_ylabel(
        "Available (0/1)"
    )

    axes[2].legend(
        ncol=3
    )

    axes[2].grid(True)

    # --------------------------------------------------------------
    # Confidence / covariance
    # --------------------------------------------------------------

    axes[3].plot(
        time,
        results[
            "estimator_confidence"
        ],
        label="Estimator confidence",
    )

    axes[3].plot(
        time,
        results[
            "position_covariance_trace"
        ],
        "--",
        label="Position covariance trace",
    )

    axes[3].set_ylabel(
        "Confidence / covariance"
    )

    axes[3].legend()
    axes[3].grid(True)

    # --------------------------------------------------------------
    # Estimated body-frame accelerometer bias
    # --------------------------------------------------------------

    bias = results[
        "estimated_acceleration_bias"
    ]

    axes[4].plot(
        time,
        bias[:, 0],
        label="Estimated bax",
    )

    axes[4].plot(
        time,
        bias[:, 1],
        label="Estimated bay",
    )

    axes[4].plot(
        time,
        bias[:, 2],
        label="Estimated baz",
    )

    axes[4].axhline(
        IMU_ACCEL_BIAS_BODY[0],
        linestyle=":",
        label="True bax",
    )

    axes[4].axhline(
        IMU_ACCEL_BIAS_BODY[1],
        linestyle=":",
        label="True bay",
    )

    axes[4].axhline(
        IMU_ACCEL_BIAS_BODY[2],
        linestyle=":",
        label="True baz",
    )

    axes[4].set_ylabel(
        "Accel bias (m/s²)"
    )

    axes[4].set_xlabel(
        "Time (s)"
    )

    axes[4].legend(
        ncol=3
    )

    axes[4].grid(True)

    fig.suptitle(
        "MorphoAqua - Stage 5B-3 "
        "Simulated Sensors, Estimation and Bias"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "Stage_5B3_sensor_estimation.png",
    )


# ======================================================================
# REFERENCE TRAJECTORY CHECK
# ======================================================================

def print_reference_checkpoints():

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
            position,
            velocity,
            _acceleration,
        ) = mission_trajectory(
            t
        )

        print(
            f"t = {t:5.1f} s | "
            f"X = {position[0]:8.4f} | "
            f"Y = {position[1]:8.4f} | "
            f"Z = {position[2]:8.4f} | "
            f"|V| = {np.linalg.norm(velocity):8.4f}"
        )

    print(
        "-" * 76
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "=" * 76
    )

    print(
        "MORPHOAQUA - STAGE 5B-3"
    )

    print(
        "SIMULATED SENSING + GPS LOSS + STATE ESTIMATION"
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
        "Sensor model:"
    )

    print(
        "IMU-like acceleration + gyro"
    )

    print(
        "GPS-like aerial position"
    )

    print(
        "GPS degradation from 10-11 s and complete loss after 11 s"
    )

    print(
        "Vision-like position and attitude"
    )

    print(
        "Depth sensing during immersion"
    )

    print(
        "Sonar-like local 3-D position fixes underwater"
    )

    print()

    print(
        "Estimator:"
    )

    print(
        "9-state position/velocity/body-frame acceleration-bias KF"
    )

    print(
        "Gyro-integrated attitude estimate with visual correction"
    )

    print(
        "Innovation gating"
    )

    print(
        "Controller feedback source: estimated state"
    )

    print()

    print(
        "Important:"
    )

    print(
        "Accelerometer bias is modeled in the BODY FRAME."
    )

    print(
        "All sensor signals are simulated numerical measurements."
    )

    print(
        "No physical sensor hardware is claimed."
    )

    print(
        "Adaptive/predictive control is reserved for Stage 5B-4."
    )

    print()

    print_reference_checkpoints()

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
        "STAGE 5B-3 PERFORMANCE METRICS"
    )

    print(
        "-" * 76
    )

    for (
        name,
        value,
    ) in metrics.items():

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
        "State feedback: estimated position, velocity, "
        "attitude and angular rate"
    )

    print(
        "Force feasibility: attitude-limit-aware "
        "force projection"
    )

    print(
        "Degrees of freedom: 6"
    )

    print()

    plot_position(
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

    plot_direction_impact(
        results
    )

    plot_sensor_estimation(
        results
    )

    print(
        "Results saved to:"
    )

    output_files = [
        "Stage_5B3_position_estimation_tracking.png",
        "Stage_5B3_3D_true_estimated_trajectory.png",
        "Stage_5B3_control_medium_response.png",
        "Stage_5B3_morphology_inertia.png",
        "Stage_5B3_direction_impact_response.png",
        "Stage_5B3_sensor_estimation.png",
    ]

    for filename in output_files:

        print(
            os.path.join(
                RESULTS_DIRECTORY,
                filename,
            )
        )

    print()

    print(
        "Stage 5B-3 simulation completed."
    )


if __name__ == "__main__":
    main()