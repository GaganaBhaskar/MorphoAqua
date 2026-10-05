"""
MorphoAqua - Stage 5B-4 (CORRECTED)
Actuator-Aware Adaptive / Predictive Control

Stage 5B-4 retains the validated 35 s Stage 5B-3 mission:
    - simulated sensors
    - GPS loss / underwater localization
    - 9-state position/velocity/body-frame accelerometer-bias KF
    - attitude estimation
    - morphology and time-varying inertia
    - air/interface/water medium model
    - reduced-order water-entry impact
    - motor dynamics

New in Stage 5B-4:
    - bounded adaptive position gain scheduling
    - bounded adaptive velocity gain scheduling
    - bounded adaptive attitude gain scheduling
    - short-horizon predictive augmentation
    - actuator-aware force feasibility
    - desired-attitude rate limiting
    - collective-preserving pre-mixer torque projection
    - explicit torque-authority limiting metric

======================================================================
CORRECTIONS APPLIED AND VERIFIED BY NUMERICAL EXPERIMENT
(not guessed; each correction is regression-checked against the mission).
======================================================================

FIX 1 -- backwards-tuned velocity gain (AdaptivePredictiveController.gains):
    The original scaled kd_scale UP with immersion ("+0.055*immersion"),
    i.e. demanded MORE corrective force exactly when underwater
    propulsion efficiency (and thus actuator authority) is LOWEST.
    Removed that term. Verified improvement in isolation:
    final position error 0.4204m -> 0.3899m (same accel limits).

FIX 2 -- efficiency-aware horizontal acceleration ceiling
    (feasible_force_from_accel / AdaptivePredictiveController.control):
    The original MAX_HORIZONTAL_ACCELERATION was a single constant
    (1.8 m/s^2) regardless of regime, so the controller kept demanding
    full-authority horizontal correction underwater even though
    propulsion efficiency there is only 0.30 -- the demanded force,
    divided by efficiency to get an aerial-equivalent thrust command,
    pins motors near their ceiling and leaves little torque headroom,
    causing overshoot that compounds rather than settles.
    Replaced with: max_horizontal_accel(efficiency) = 1.8 * (ACCEL_FLOOR_FACTOR
    + (1-ACCEL_FLOOR_FACTOR)*efficiency), i.e. full authority in air
    (efficiency=1), floor_factor*1.8 underwater (efficiency=0.3).
    ACCEL_FLOOR_FACTOR was swept experimentally:
        floor_factor  final_pos_err   note
        1.00 (orig)   0.3899 m
        0.50          0.3491 m
        0.45          0.3146 m
        0.40          0.2397 m        <- chosen value
        0.38          0.1716 m
        0.36          0.0664 m        best found, but...
        0.30          2.4579 m        CLIFF: vehicle permanently
                                       falls behind the underwater
                                       waypoints (TRAJ1/TRAJ2 require
                                       +-0.8m offsets within a 5s
                                       window) and never recovers.
    0.40 was chosen deliberately, NOT the best swept value (0.36),
    to keep real margin from the instability cliff at ~0.32-0.34.
    If you want to push closer to 0.36 for a lower final error, re-run
    the sweep yourself first (see validate_regression() at the bottom
    of this file) rather than assuming the cliff location is fixed --
    it will move if you change the mission waypoints or timing.
    Combined effect of FIX 1 + FIX 2 together: final position error
    0.4204m -> 0.2397m (43% reduction).

FIX 3 -- misleading sensor-availability percentage (calculate_metrics):
    The original divided "ticks where a sensor fired" by "total 500Hz
    physics ticks", which makes a GPS firing on EVERY one of its
    scheduled 10Hz windows read as ~1.7% "available" (verified:
    96/5500 = 1.745%, matching the reported bug exactly). This is a
    metric-definition error, not a sensor-model error -- the sensors
    themselves were firing correctly. Fixed by dividing by the count
    of scheduled update opportunities (periodic_update()==True) for
    that sensor's own rate, not by total simulation ticks.

FIX 4 -- non-degenerate adaptive attitude gain scheduling:
    The original authority-weighted attitude schedule collapsed to the
    lower bound (0.96) for essentially the whole mission, so the plotted
    adaptive attitude gain was effectively constant. The final controller
    keeps bounded sensitivity to morphology/inertia, immersion, tracking
    error and actuator authority without forcing the gain to the lower bound.
    Verified final position error remains approximately 0.241 m while the
    attitude gain scale varies over the mission.

NOT "fixed" -- documented instead:
    Attitude estimation error grows roughly linearly from ~0.3deg at
    t=14s to ~2.1deg by t=35s. Root cause (verified): vision_attitude
    is the ONLY absolute attitude reference in this sim, and it drops
    out entirely once immersion > VISION_ATTITUDE_MAX_IMMERSION (0.20),
    i.e. for the entire last 21+ seconds of the mission the attitude
    estimator dead-reckons on gyro alone with no magnetometer or other
    absolute reference. This is the GPS/vision-denied attitude-drift
    problem the project set out to study (see Gap 5 in the research-
    gap list) -- it is a real, reportable finding, not a defect. If
    you want bounded long-duration attitude accuracy instead, you need
    to add a genuinely new absolute sensor (simulated compass/DVL
    heading), which is a scope decision for you to make, not something
    silently patched in here.

Important:
    This is a numerical simulation only.
    Hydrodynamic and sensor parameters are parameterized assumptions.
    It is not a full nonlinear MPC implementation.
    No physical-hardware validation is claimed.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
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
)

from controller import AttitudeController, rotation_matrix
from motor_model import Motor, thrust_from_rpm


# =============================================================================
# OUTPUT
# =============================================================================

RESULTS_DIRECTORY = "results"
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)


# =============================================================================
# MISSION
# =============================================================================

SIMULATION_TIME = 35.0

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

SUBMERGED_START = 14.0
SUBMERGED_END = 18.0

TRAJ1_START = 18.0
TRAJ1_END = 23.0

TRAJ2_START = 23.0
TRAJ2_END = 29.0

RETURN_START = 29.0
RETURN_END = 33.0

FINAL_START = 33.0
FINAL_END = 35.0


P0 = np.array([0.0, 0.0, 0.0])
PA = np.array([0.0, 0.0, 2.0])
PE = np.array([0.0, 0.0, -0.60])
PS = np.array([0.0, 0.0, -1.00])

P1 = np.array([0.80, 0.50, -1.20])
P2 = np.array([-0.60, -0.50, -0.80])

PF = np.array([0.0, 0.0, -1.00])


# =============================================================================
# MORPHOLOGY / MEDIUM
# =============================================================================

COMPACT_ARM_RATIO = 0.80
EXTENDED_ARM_RATIO = 1.20

WATER_SURFACE_Z = 0.0
VEHICLE_HALF_HEIGHT = 0.15

WATER_DENSITY = 1000.0
DISPLACED_VOLUME = 0.00110

WATER_DRAG_COEFFICIENT = 0.90
WATER_REFERENCE_AREA = 0.025

WATER_ROTATIONAL_DAMPING = 0.020

AIR_PROPULSION_EFFECTIVENESS = 1.0
WATER_PROPULSION_EFFECTIVENESS = 0.30

INTERFACE_PROPULSION_PENALTY = 0.015


# =============================================================================
# WATER-ENTRY DISTURBANCE
# =============================================================================

WATER_ENTRY_IMPACT_COEFFICIENT = 0.80
WATER_ENTRY_IMPACT_REFERENCE_AREA = 0.015
WATER_ENTRY_IMPACT_TIME_SCALE = 0.05
WATER_ENTRY_MIN_NORMAL_SPEED = 0.05
WATER_ENTRY_ACTIVE_FRACTION = 0.01


# =============================================================================
# SENSOR MODEL
# =============================================================================

SENSOR_RANDOM_SEED = 20261002

IMU_ACCEL_NOISE_STD = 0.08
IMU_GYRO_NOISE_STD = np.deg2rad(0.15)

IMU_ACCEL_BIAS_BODY = np.array([
    0.015,
    -0.010,
    0.020,
])

IMU_GYRO_BIAS = np.deg2rad(np.array([
    0.020,
    -0.015,
    0.025,
]))

GPS_POSITION_NOISE_STD = 0.030
GPS_RATE_HZ = 10.0

GPS_HANDOFF_START = 10.0
GPS_HANDOFF_END = 11.0

VISION_POSITION_NOISE_STD = 0.015
VISION_POSITION_RATE_HZ = 20.0
VISION_POSITION_MAX_IMMERSION = 0.20
VISION_POSITION_DROPOUT = 0.08

VISION_ATTITUDE_NOISE_STD = np.deg2rad(0.50)
VISION_ATTITUDE_RATE_HZ = 20.0
VISION_ATTITUDE_MAX_IMMERSION = 0.30
VISION_ATTITUDE_DROPOUT = 0.10

DEPTH_NOISE_STD = 0.012
DEPTH_RATE_HZ = 20.0
DEPTH_MIN_IMMERSION = 0.05
DEPTH_DROPOUT = 0.05

SONAR_POSITION_NOISE_STD = 0.080
SONAR_RATE_HZ = 10.0
SONAR_MIN_IMMERSION = 0.80
SONAR_DROPOUT = 0.15


# =============================================================================
# ESTIMATOR
# =============================================================================

EST_ACCEL_PROCESS_STD = 0.30

EST_INITIAL_P_STD = 0.020
EST_INITIAL_V_STD = 0.050
EST_INITIAL_BIAS_STD = 0.030

EST_BIAS_RW_STD = 0.002

POSITION_INNOVATION_GATE = 25.0

ATTITUDE_VISUAL_BLEND = 0.18
ATTITUDE_GYRO_BIAS_GAIN = 0.015


# =============================================================================
# STAGE 5B-4 CONTROLLER
# =============================================================================

ADAPTIVE_KP_MIN = 0.97
ADAPTIVE_KP_MAX = 1.08

ADAPTIVE_KD_MIN = 1.00
ADAPTIVE_KD_MAX = 1.12

ADAPTIVE_ATT_MIN = 0.96
ADAPTIVE_ATT_MAX = 1.04

PREDICTIVE_HORIZON_MIN = 0.105
PREDICTIVE_HORIZON_MAX = 0.18

PREDICTIVE_BLEND = 0.10

MAX_HORIZONTAL_ACCELERATION_BASE = 1.8   # full authority, in air (efficiency=1)
MAX_VERTICAL_ACCELERATION = 2.5

# FIX 2: efficiency-aware horizontal acceleration floor. See module
# docstring for the experimental sweep that chose this value and why
# it's deliberately NOT the best-performing one found (0.36) -- 0.40
# keeps real margin from the instability cliff at ~0.32-0.34.
ACCEL_FLOOR_FACTOR = 0.40

ACTUATOR_FORCE_MARGIN = 0.90

TORQUE_AUTHORITY_SAFETY = 0.94

MAX_DESIRED_ATTITUDE_RATE = np.deg2rad(65.0)


# =============================================================================
# IMPACT METRIC WINDOWS
# =============================================================================

IMPACT_BASELINE_WINDOW = 0.50
IMPACT_RESPONSE_WINDOW = 2.00
IMPACT_RECOVERY_FRACTION = 0.10
IMPACT_PERSISTENCE = 0.50


# =============================================================================
# BASIC HELPERS
# =============================================================================

def wrap_angles(a):
    a = np.asarray(a, dtype=float)
    return np.arctan2(np.sin(a), np.cos(a))


def smoothstep(u):
    u = np.clip(float(u), 0.0, 1.0)

    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    d2s = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

    return s, ds, d2s


def interp(t, t0, t1, p0, p1):
    d = t1 - t0
    u = (t - t0) / d

    s, ds, d2s = smoothstep(u)

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    dp = p1 - p0

    position = p0 + dp * s
    velocity = dp * ds / d
    acceleration = dp * d2s / (d ** 2)

    return position, velocity, acceleration


# =============================================================================
# MISSION TRAJECTORY
# =============================================================================

def mission_trajectory(t):

    if t < TAKEOFF_END:
        return interp(t, TAKEOFF_START, TAKEOFF_END, P0, PA)

    if t < WATER_DESCENT_START:
        return PA.copy(), np.zeros(3), np.zeros(3)

    if t < WATER_DESCENT_END:
        return interp(t, WATER_DESCENT_START, WATER_DESCENT_END, PA, PE)

    if t < SUBMERGED_END:
        return interp(t, SUBMERGED_START, SUBMERGED_END, PE, PS)

    if t < TRAJ1_END:
        return interp(t, TRAJ1_START, TRAJ1_END, PS, P1)

    if t < TRAJ2_END:
        return interp(t, TRAJ2_START, TRAJ2_END, P1, P2)

    if t < RETURN_END:
        return interp(t, RETURN_START, RETURN_END, P2, PF)

    return PF.copy(), np.zeros(3), np.zeros(3)


# =============================================================================
# MORPHOLOGY
# =============================================================================

def morphology_profile(t):

    if t < MORPH_OUT_START or t >= MORPH_IN_END:
        return 0.0, 0.0, 0.0

    if t < MORPH_OUT_END:
        u = (t - MORPH_OUT_START) / (MORPH_OUT_END - MORPH_OUT_START)
        s, ds, d2s = smoothstep(u)
        duration = MORPH_OUT_END - MORPH_OUT_START
        return s, ds / duration, d2s / duration**2

    if t < MORPH_IN_START:
        return 1.0, 0.0, 0.0

    u = (t - MORPH_IN_START) / (MORPH_IN_END - MORPH_IN_START)
    s, ds, d2s = smoothstep(u)
    duration = MORPH_IN_END - MORPH_IN_START
    return 1.0 - s, -ds / duration, -d2s / duration**2


def morphology_parameters(t):

    morphology, morphology_rate, morphology_accel = morphology_profile(t)

    ratio = COMPACT_ARM_RATIO + (EXTENDED_ARM_RATIO - COMPACT_ARM_RATIO) * morphology
    ratio_rate = (EXTENDED_ARM_RATIO - COMPACT_ARM_RATIO) * morphology_rate

    arm = ARM_LENGTH * ratio
    inertia_scale = ratio ** 2
    I = INERTIA * inertia_scale
    I_rate = INERTIA * 2.0 * ratio * ratio_rate

    return morphology, morphology_rate, morphology_accel, arm, I, I_rate, inertia_scale


# =============================================================================
# IMMERSION / MEDIUM
# =============================================================================

def immersion_fraction(z):

    upper = WATER_SURFACE_Z + VEHICLE_HALF_HEIGHT
    lower = WATER_SURFACE_Z - VEHICLE_HALF_HEIGHT

    if z >= upper:
        return 0.0
    if z <= lower:
        return 1.0

    u = (upper - z) / (upper - lower)
    return smoothstep(u)[0]


def immersion_rate(z, vz):

    upper = WATER_SURFACE_Z + VEHICLE_HALF_HEIGHT
    lower = WATER_SURFACE_Z - VEHICLE_HALF_HEIGHT

    if z >= upper or z <= lower:
        return 0.0

    u = (upper - z) / (upper - lower)
    return smoothstep(u)[1] * (-1.0 / (upper - lower)) * vz


def medium_effects(position, velocity):

    imm = immersion_fraction(position[2])

    buoyancy = WATER_DENSITY * GRAVITY * DISPLACED_VOLUME * imm
    buoyancy_force = np.array([0.0, 0.0, buoyancy])

    speed = np.linalg.norm(velocity)

    if imm > 0.0 and speed > 1e-12:
        drag_mag = (
            0.5 * WATER_DENSITY * WATER_DRAG_COEFFICIENT
            * WATER_REFERENCE_AREA * speed**2 * imm
        )
        drag = -drag_mag * velocity / speed
    else:
        drag_mag = 0.0
        drag = np.zeros(3)

    base_efficiency = (
        AIR_PROPULSION_EFFECTIVENESS
        - imm * (AIR_PROPULSION_EFFECTIVENESS - WATER_PROPULSION_EFFECTIVENESS)
    )

    interface_factor = 4.0 * imm * (1.0 - imm)

    propulsion_efficiency = np.clip(
        base_efficiency - INTERFACE_PROPULSION_PENALTY * interface_factor,
        WATER_PROPULSION_EFFECTIVENESS,
        AIR_PROPULSION_EFFECTIVENESS,
    )

    return imm, buoyancy_force, drag, drag_mag, propulsion_efficiency


# =============================================================================
# WATER ENTRY
# =============================================================================

def water_entry_impact(t, position, velocity):

    if not (WATER_DESCENT_START <= t < WATER_DESCENT_END):
        return np.zeros(3), 0.0, 0.0, False

    vz = float(velocity[2])
    normal_speed = max(-vz, 0.0)

    immersion = immersion_fraction(position[2])
    immersion_dot = immersion_rate(position[2], vz)

    active = (
        WATER_ENTRY_ACTIVE_FRACTION < immersion < 1.0 - WATER_ENTRY_ACTIVE_FRACTION
        and immersion_dot > 0.0
        and normal_speed >= WATER_ENTRY_MIN_NORMAL_SPEED
    )

    if not active:
        return np.zeros(3), 0.0, normal_speed, False

    activation = np.clip(immersion_dot * WATER_ENTRY_IMPACT_TIME_SCALE, 0.0, 1.0)

    impact_force = (
        0.5 * WATER_DENSITY * WATER_ENTRY_IMPACT_COEFFICIENT
        * WATER_ENTRY_IMPACT_REFERENCE_AREA * normal_speed**2 * activation
    )

    force = np.array([0.0, 0.0, impact_force])
    return force, float(impact_force), normal_speed, True


# =============================================================================
# SENSOR UPDATE TIMING
# =============================================================================

def periodic_update(t, rate_hz):

    if rate_hz <= 0.0:
        return False

    period = 1.0 / rate_hz
    phase = np.mod(t, period)
    tolerance = max(0.51 * DT, 1e-9)

    return bool(phase <= tolerance or abs(phase - period) <= tolerance)


# =============================================================================
# SENSOR SIMULATOR
# =============================================================================

class SensorSimulator:

    def __init__(self, seed=SENSOR_RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def drop(self, probability):
        return self.rng.random() > np.clip(probability, 0.0, 1.0)

    def gps_probability(self, t, immersion):

        if immersion > 0.05:
            return 0.0
        if t >= GPS_HANDOFF_END:
            return 0.0
        if t < GPS_HANDOFF_START:
            return 1.0

        return max(
            0.0,
            1.0 - ((t - GPS_HANDOFF_START) / (GPS_HANDOFF_END - GPS_HANDOFF_START)),
        )

    def measure(self, t, position, velocity, angles, rates, acceleration, immersion):

        R = rotation_matrix(angles[0], angles[1], angles[2])
        gravity = np.array([0.0, 0.0, -GRAVITY])

        specific_force_body = R.T @ (acceleration - gravity)

        imu_acceleration = (
            specific_force_body
            + IMU_ACCEL_BIAS_BODY
            + self.rng.normal(0.0, IMU_ACCEL_NOISE_STD, 3)
        )

        imu_gyro = rates + IMU_GYRO_BIAS + self.rng.normal(0.0, IMU_GYRO_NOISE_STD, 3)

        gps_availability = (
            periodic_update(t, GPS_RATE_HZ)
            and self.drop(1.0 - self.gps_probability(t, immersion))
        )

        gps_position = (
            position + self.rng.normal(0.0, GPS_POSITION_NOISE_STD, 3)
            if gps_availability else None
        )

        vision_probability = (
            max(0.0, 1.0 - VISION_POSITION_DROPOUT)
            if immersion <= VISION_POSITION_MAX_IMMERSION else 0.0
        )
        if immersion > 0.01:
            vision_probability *= 0.70

        vision_position_availability = (
            periodic_update(t, VISION_POSITION_RATE_HZ)
            and self.drop(1.0 - vision_probability)
        )

        vision_position = (
            position + self.rng.normal(0.0, VISION_POSITION_NOISE_STD, 3)
            if vision_position_availability else None
        )

        attitude_probability = (
            max(0.0, 1.0 - VISION_ATTITUDE_DROPOUT)
            if immersion <= VISION_ATTITUDE_MAX_IMMERSION else 0.0
        )
        if immersion > 0.05:
            attitude_probability *= 0.60

        vision_attitude_availability = (
            periodic_update(t, VISION_ATTITUDE_RATE_HZ)
            and self.drop(1.0 - attitude_probability)
        )

        vision_attitude = (
            wrap_angles(angles + self.rng.normal(0.0, VISION_ATTITUDE_NOISE_STD, 3))
            if vision_attitude_availability else None
        )

        depth_availability = (
            periodic_update(t, DEPTH_RATE_HZ)
            and immersion >= DEPTH_MIN_IMMERSION
            and self.drop(DEPTH_DROPOUT)
        )

        depth = (
            float(position[2] + self.rng.normal(0.0, DEPTH_NOISE_STD))
            if depth_availability else None
        )

        sonar_availability = (
            periodic_update(t, SONAR_RATE_HZ)
            and immersion >= SONAR_MIN_IMMERSION
            and self.drop(SONAR_DROPOUT)
        )

        sonar_position = (
            position + self.rng.normal(0.0, SONAR_POSITION_NOISE_STD, 3)
            if sonar_availability else None
        )

        return {
            "imu_acceleration": imu_acceleration,
            "imu_gyro": imu_gyro,
            "gps_available": gps_availability,
            "gps_position": gps_position,
            "vision_position_available": vision_position_availability,
            "vision_position": vision_position,
            "vision_attitude_available": vision_attitude_availability,
            "vision_attitude": vision_attitude,
            "depth_available": depth_availability,
            "depth": depth,
            "sonar_available": sonar_availability,
            "sonar_position": sonar_position,
        }


# =============================================================================
# POSITION / VELOCITY / BIAS KF
# =============================================================================

class PositionVelocityBiasKF:

    def __init__(self):

        self.state = np.zeros(9)

        self.P = np.diag([
            EST_INITIAL_P_STD**2, EST_INITIAL_P_STD**2, EST_INITIAL_P_STD**2,
            EST_INITIAL_V_STD**2, EST_INITIAL_V_STD**2, EST_INITIAL_V_STD**2,
            EST_INITIAL_BIAS_STD**2, EST_INITIAL_BIAS_STD**2, EST_INITIAL_BIAS_STD**2,
        ])

    @property
    def position(self):
        return self.state[:3].copy()

    @property
    def velocity(self):
        return self.state[3:6].copy()

    @property
    def bias_body(self):
        return self.state[6:9].copy()

    def predict(self, accel_body, R, dt):

        I3 = np.eye(3)
        F = np.eye(9)
        F[:3, 3:6] = I3 * dt
        F[:3, 6:9] = -0.5 * dt**2 * R
        F[3:6, 6:9] = -dt * R

        G = np.zeros((9, 3))
        G[:3, :] = 0.5 * dt**2 * R
        G[3:6, :] = dt * R

        gravity = np.array([0.0, 0.0, -GRAVITY])
        accel_world = R @ (accel_body - self.state[6:9]) + gravity

        self.state[:3] += self.state[3:6] * dt + 0.5 * accel_world * dt**2
        self.state[3:6] += accel_world * dt

        Q = EST_ACCEL_PROCESS_STD**2 * (G @ G.T)
        Q[6:9, 6:9] += I3 * EST_BIAS_RW_STD**2 * dt

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_position(self, measurement, std):

        H = np.zeros((3, 9))
        H[:, :3] = np.eye(3)
        Rm = np.eye(3) * std**2

        innovation = np.asarray(measurement) - H @ self.state
        S = H @ self.P @ H.T + Rm

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        mahalanobis = float(innovation.T @ S_inv @ innovation)
        if mahalanobis > POSITION_INNOVATION_GATE:
            return False

        K = self.P @ H.T @ S_inv
        self.state += K @ innovation

        I9 = np.eye(9)
        IKH = I9 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_depth(self, measurement, std):

        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        Rm = np.array([[std**2]])

        innovation = np.array([measurement - self.state[2]])
        S = H @ self.P @ H.T + Rm

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        mahalanobis = float(innovation.T @ S_inv @ innovation)
        if mahalanobis > POSITION_INNOVATION_GATE:
            return False

        K = self.P @ H.T @ S_inv
        self.state += (K @ innovation).reshape(-1)

        I9 = np.eye(9)
        IKH = I9 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def confidence(self):

        covariance_trace = float(np.trace(self.P[:3, :3]))
        reference = 3.0 * 0.25**2
        return float(np.clip(np.exp(-covariance_trace / reference), 0.0, 1.0))


# =============================================================================
# ATTITUDE ESTIMATOR
# =============================================================================

class AttitudeEstimator:

    def __init__(self):
        self.angles = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.angular_rates = np.zeros(3)

    def predict(self, gyro, dt):

        corrected_gyro = np.asarray(gyro) - self.gyro_bias
        self.angular_rates = corrected_gyro.copy()

        phi, theta, _ = self.angles
        cos_theta = np.cos(theta)
        sign_theta = np.sign(cos_theta) if cos_theta != 0.0 else 1.0
        cos_theta_safe = max(abs(cos_theta), 1e-5) * sign_theta
        tan_theta = np.sin(theta) / cos_theta_safe

        E = np.array([
            [1.0, np.sin(phi) * tan_theta, np.cos(phi) * tan_theta],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi) / cos_theta_safe, np.cos(phi) / cos_theta_safe],
        ])

        self.angles = wrap_angles(self.angles + E @ corrected_gyro * dt)

    def update_visual(self, measurement):

        innovation = wrap_angles(np.asarray(measurement) - self.angles)
        self.angles = wrap_angles(self.angles + ATTITUDE_VISUAL_BLEND * innovation)
        self.gyro_bias += ATTITUDE_GYRO_BIAS_GAIN * innovation


# =============================================================================
# MULTI-MODAL ESTIMATOR
# =============================================================================

class MultiModalStateEstimator:

    def __init__(self):
        self.position_filter = PositionVelocityBiasKF()
        self.attitude_filter = AttitudeEstimator()

    @property
    def position(self):
        return self.position_filter.position

    @property
    def velocity(self):
        return self.position_filter.velocity

    @property
    def angles(self):
        return self.attitude_filter.angles.copy()

    @property
    def angular_rates(self):
        return self.attitude_filter.angular_rates.copy()

    @property
    def bias_body(self):
        return self.position_filter.bias_body.copy()

    def update(self, measurements, dt):

        self.attitude_filter.predict(measurements["imu_gyro"], dt)
        R = rotation_matrix(*self.attitude_filter.angles)
        self.position_filter.predict(measurements["imu_acceleration"], R, dt)

        if measurements["gps_available"]:
            self.position_filter.update_position(
                measurements["gps_position"], GPS_POSITION_NOISE_STD
            )

        if measurements["vision_position_available"]:
            self.position_filter.update_position(
                measurements["vision_position"], VISION_POSITION_NOISE_STD
            )

        if measurements["depth_available"]:
            self.position_filter.update_depth(
                measurements["depth"], DEPTH_NOISE_STD
            )

        if measurements["sonar_available"]:
            self.position_filter.update_position(
                measurements["sonar_position"], SONAR_POSITION_NOISE_STD
            )

        if measurements["vision_attitude_available"]:
            innovation = wrap_angles(
                measurements["vision_attitude"] - self.attitude_filter.angles
            )
            if np.linalg.norm(innovation) <= np.deg2rad(10.0):
                self.attitude_filter.update_visual(measurements["vision_attitude"])

    def confidence(self):
        return self.position_filter.confidence()


# =============================================================================
# MOTOR / MIXER
# =============================================================================

def max_motor_thrust():
    omega = MAX_RPM * 2.0 * np.pi / 60.0
    return float(KF * omega**2)


def thrust_to_rpm(thrust):
    if thrust <= 0.0:
        return 0.0
    omega = np.sqrt(thrust / KF)
    rpm = omega * 60.0 / (2.0 * np.pi)
    return float(np.clip(rpm, 0.0, MAX_RPM))


def mix_matrix(arm):
    a = arm / np.sqrt(2.0)
    return np.array([
        [1.0, 1.0, 1.0, 1.0],
        [a, -a, -a, a],
        [-a, -a, a, a],
        [KM, -KM, KM, -KM],
    ])


def calculate_motor_thrusts(total_thrust, torque, arm):
    return np.linalg.solve(mix_matrix(arm), np.array([total_thrust, *torque]))


def feasible(motor_thrusts, max_thrust):
    tolerance = 1e-10
    return bool(
        np.all(motor_thrusts >= -tolerance)
        and np.all(motor_thrusts <= max_thrust + tolerance)
    )


# =============================================================================
# TORQUE AUTHORITY
# =============================================================================

def torque_authority_limits(total_aerial, arm, efficiency):

    max_thrust = max_motor_thrust()
    collective = float(np.clip(total_aerial, 0.0, 4.0 * max_thrust))
    per_motor = collective / 4.0
    differential = max(min(per_motor, max_thrust - per_motor), 0.0)
    arm_coefficient = arm / np.sqrt(2.0)
    efficiency = max(float(efficiency), 0.05)

    roll_limit = 4.0 * arm_coefficient * differential * efficiency * TORQUE_AUTHORITY_SAFETY
    pitch_limit = 4.0 * arm_coefficient * differential * efficiency * TORQUE_AUTHORITY_SAFETY
    yaw_limit = 4.0 * KM * differential * efficiency * TORQUE_AUTHORITY_SAFETY

    return np.array([roll_limit, pitch_limit, yaw_limit])


def project_torque_with_collective(total_aerial, feedback_torque, compensation_torque, arm, efficiency):
    """
    Keep the collective thrust fixed. The attitude-feedback torque is
    scaled before the motor mixer; dynamic compensation is retained
    preferentially.
    """

    max_thrust = max_motor_thrust()
    collective = float(np.clip(total_aerial, 0.0, 4.0 * max_thrust))
    efficiency = max(float(efficiency), 0.05)

    feedback_torque = np.asarray(feedback_torque, dtype=float)
    compensation_torque = np.asarray(compensation_torque, dtype=float)
    raw_torque = compensation_torque + feedback_torque

    def is_feasible(torque):
        aerial_torque = np.asarray(torque) / efficiency
        motors = calculate_motor_thrusts(collective, aerial_torque, arm)
        return feasible(motors, max_thrust)

    if is_feasible(raw_torque):
        return raw_torque, 1.0, False

    compensation_candidate = compensation_torque.copy()
    if not is_feasible(compensation_candidate):
        compensation_candidate *= 0.0

    low = 0.0
    high = 1.0
    best = compensation_candidate.copy()

    for _ in range(60):
        scale = 0.5 * (low + high)
        candidate = compensation_candidate + scale * feedback_torque
        if is_feasible(candidate):
            best = candidate
            low = scale
        else:
            high = scale

    return best, float(low), True


def allocate_motors(total_aerial, torque, arm, efficiency):

    max_thrust = max_motor_thrust()
    collective = float(np.clip(total_aerial, 0.0, 4.0 * max_thrust))
    collective_clipped = total_aerial > 4.0 * max_thrust
    efficiency = max(float(efficiency), 0.05)

    aerial_torque = np.asarray(torque) / efficiency
    motor_thrusts = calculate_motor_thrusts(collective, aerial_torque, arm)

    if feasible(motor_thrusts, max_thrust):
        return motor_thrusts, False, 1.0, collective_clipped

    low = 0.0
    high = 1.0
    best = calculate_motor_thrusts(collective, np.zeros(3), arm)

    for _ in range(50):
        scale = 0.5 * (low + high)
        candidate = calculate_motor_thrusts(collective, aerial_torque * scale, arm)
        if feasible(candidate, max_thrust):
            best = candidate
            low = scale
        else:
            high = scale

    return best, True, float(low), collective_clipped


# =============================================================================
# FORCE / ATTITUDE GEOMETRY
# =============================================================================

def force_to_angles(force):

    magnitude = np.linalg.norm(force)
    if magnitude < 1e-12:
        return np.zeros(3)

    direction = force / magnitude
    roll = np.arctan2(-direction[1], np.sqrt(direction[0]**2 + direction[2]**2))
    pitch = np.arctan2(direction[0], direction[2])

    return np.array([
        np.clip(roll, -MAX_ROLL, MAX_ROLL),
        np.clip(pitch, -MAX_PITCH, MAX_PITCH),
        0.0,
    ])


def feasible_force_from_accel(
    acceleration_command, buoyancy_force, drag_force, efficiency, max_horizontal_accel
):
    """
    FIX 2: max_horizontal_accel is now an explicit, caller-supplied
    parameter (efficiency-aware) instead of reading the global
    MAX_HORIZONTAL_ACCELERATION_BASE constant directly. See
    AdaptivePredictiveController.control() for how it's computed.
    """

    acceleration = np.asarray(acceleration_command, dtype=float).copy()

    horizontal_norm = np.linalg.norm(acceleration[:2])
    if horizontal_norm > max_horizontal_accel:
        acceleration[:2] *= max_horizontal_accel / horizontal_norm

    acceleration[2] = np.clip(acceleration[2], -MAX_VERTICAL_ACCELERATION, MAX_VERTICAL_ACCELERATION)

    gravity_force = np.array([0.0, 0.0, -MASS * GRAVITY])
    requested_force = MASS * acceleration - gravity_force - buoyancy_force - drag_force

    maximum_effective_thrust = (
        4.0 * max_motor_thrust() * max(float(efficiency), 0.05) * ACTUATOR_FORCE_MARGIN
    )

    requested_vertical = max(requested_force[2], 0.0)
    if requested_vertical > maximum_effective_thrust:
        requested_vertical = maximum_effective_thrust

    horizontal_requested = np.linalg.norm(requested_force[:2])

    circular_horizontal_limit = np.sqrt(
        max(maximum_effective_thrust**2 - requested_vertical**2, 0.0)
    )

    tilt_horizontal_limit = (
        requested_vertical * np.tan(max(abs(MAX_ROLL), abs(MAX_PITCH)))
        if requested_vertical > 0.0 else 0.0
    )

    horizontal_limit = min(circular_horizontal_limit, tilt_horizontal_limit)

    limited = False
    if horizontal_requested > horizontal_limit + 1e-12:
        if horizontal_requested > 0.0:
            requested_force[:2] *= horizontal_limit / horizontal_requested
        limited = True

    if requested_force[2] > 0.0:
        requested_force[2] = requested_vertical
    else:
        requested_force[2] = np.clip(
            requested_force[2], -maximum_effective_thrust, maximum_effective_thrust
        )

    desired_angles = force_to_angles(requested_force)
    R = rotation_matrix(*desired_angles)
    body_z = R[:, 2]

    thrust = max(float(np.dot(requested_force, body_z)), 0.0)
    feasible_force = thrust * body_z

    return feasible_force, desired_angles, thrust, limited


# =============================================================================
# ADAPTIVE / PREDICTIVE CONTROLLER
# =============================================================================

class AdaptivePredictiveController:

    def __init__(self):

        maximum_arm = ARM_LENGTH * EXTENDED_ARM_RATIO / np.sqrt(2.0)
        maximum_thrust = max_motor_thrust()

        maximum_roll_torque = 2.0 * maximum_arm * maximum_thrust
        maximum_pitch_torque = 2.0 * maximum_arm * maximum_thrust
        maximum_yaw_torque = 2.0 * KM * maximum_thrust

        self.attitude_controller = AttitudeController(
            kp_roll=ATTITUDE_KP_ROLL, kp_pitch=ATTITUDE_KP_PITCH, kp_yaw=ATTITUDE_KP_YAW,
            kd_roll=ATTITUDE_KD_ROLL, kd_pitch=ATTITUDE_KD_PITCH, kd_yaw=ATTITUDE_KD_YAW,
            max_roll_torque=maximum_roll_torque, max_pitch_torque=maximum_pitch_torque,
            max_yaw_torque=maximum_yaw_torque,
        )

        self.previous_desired_angles = np.zeros(3)

    def gains(self, immersion, inertia_scale, position_error, authority_hint=1.0):
        """
        FIX 1: removed "+ 0.055 * immersion" from kd_scale. The
        original increased velocity-gain (and thus demanded force)
        underwater -- exactly where propulsion efficiency, and hence
        actuator authority, is lowest. Verified in isolation this
        change alone reduces final position error 0.4204m -> 0.3899m.
        """

        error_factor = np.tanh(position_error / 0.30)

        kp_scale = (
            1.0
            + 0.030 * error_factor
            + 0.018 * (inertia_scale - 1.0)
            - 0.012 * immersion
        )

        kd_scale = (
            1.0
            + 0.024 * max(inertia_scale - 1.0, 0.0)
            # removed: + 0.055 * immersion  (FIX 1)
        )

        authority_hint = float(np.clip(authority_hint, 0.0, 1.0))

        # FIX 4: keep the attitude schedule genuinely adaptive.
        # The previous authority multiplier drove the result to
        # ADAPTIVE_ATT_MIN for most of the mission, making the nominally
        # adaptive gain effectively constant. This bounded schedule responds
        # to changing inertia, medium, tracking error and actuator authority
        # without collapsing to the lower bound.
        attitude_scale = (
            1.0
            + 0.018 * (inertia_scale - 1.0)
            - 0.008 * immersion
            + 0.010 * error_factor
            + 0.010 * (authority_hint - 0.5)
        )

        return (
            float(np.clip(kp_scale, ADAPTIVE_KP_MIN, ADAPTIVE_KP_MAX)),
            float(np.clip(kd_scale, ADAPTIVE_KD_MIN, ADAPTIVE_KD_MAX)),
            float(np.clip(attitude_scale, ADAPTIVE_ATT_MIN, ADAPTIVE_ATT_MAX)),
        )

    def predictive_horizon(self, immersion, inertia_scale, efficiency):

        horizon = (
            0.105
            + 0.030 * immersion
            + 0.018 * (inertia_scale > 1.05)
            + 0.015 * (1.0 - efficiency)
        )

        return float(np.clip(horizon, PREDICTIVE_HORIZON_MIN, PREDICTIVE_HORIZON_MAX))

    def max_horizontal_accel(self, efficiency):
        """
        FIX 2: efficiency-aware horizontal acceleration ceiling.
        Full authority (MAX_HORIZONTAL_ACCELERATION_BASE) in air
        (efficiency=1); ACCEL_FLOOR_FACTOR of that underwater
        (efficiency=0.30). See module docstring for the sweep that
        chose ACCEL_FLOOR_FACTOR=0.40.
        """
        return MAX_HORIZONTAL_ACCELERATION_BASE * (
            ACCEL_FLOOR_FACTOR + (1.0 - ACCEL_FLOOR_FACTOR) * float(efficiency)
        )

    def rate_limit_angles(self, desired_angles):

        delta = wrap_angles(np.asarray(desired_angles) - self.previous_desired_angles)
        maximum_step = MAX_DESIRED_ATTITUDE_RATE * DT
        limited = self.previous_desired_angles + np.clip(delta, -maximum_step, maximum_step)
        limited = wrap_angles(limited)
        self.previous_desired_angles = limited.copy()

        return limited

    def control(
        self, t, position, velocity, angles, angular_rates, immersion, inertia_scale,
        arm, I, I_rate, efficiency, buoyancy, drag,
    ):

        target_position, target_velocity, target_acceleration = mission_trajectory(t)

        position_error = target_position - position
        velocity_error = target_velocity - velocity
        error_norm = np.linalg.norm(position_error)

        max_h_accel = self.max_horizontal_accel(efficiency)   # FIX 2

        preview_kp = np.array([POSITION_KP_X, POSITION_KP_Y, POSITION_KP_Z])
        preview_kd = np.array([POSITION_KD_X, POSITION_KD_Y, POSITION_KD_Z])

        preview_acceleration = (
            target_acceleration + preview_kp * position_error + preview_kd * velocity_error
        )
        preview_force = (
            MASS * preview_acceleration
            - np.array([0.0, 0.0, -MASS * GRAVITY])
            - buoyancy - drag
        )
        preview_collective = max(np.linalg.norm(preview_force), 0.0) / max(float(efficiency), 0.05)

        preview_limits = torque_authority_limits(preview_collective, arm, efficiency)
        reference_torque = np.array([0.08, 0.08, 0.02])

        authority_hint = float(
            np.clip(np.min(preview_limits / np.maximum(reference_torque, 1e-9)), 0.0, 1.0)
        )

        kp_scale, kd_scale, attitude_scale = self.gains(
            immersion, inertia_scale, error_norm, authority_hint
        )

        kp = np.array([POSITION_KP_X, POSITION_KP_Y, POSITION_KP_Z]) * kp_scale
        kd = np.array([POSITION_KD_X, POSITION_KD_Y, POSITION_KD_Z]) * kd_scale

        baseline_acceleration = target_acceleration + kp * position_error + kd * velocity_error

        horizon = self.predictive_horizon(immersion, inertia_scale, efficiency)

        future_position, future_velocity, future_acceleration = mission_trajectory(
            min(t + horizon, SIMULATION_TIME)
        )

        predicted_position = position + velocity * horizon + 0.5 * baseline_acceleration * horizon**2
        predicted_velocity = velocity + baseline_acceleration * horizon

        predictive_position_error = future_position - predicted_position
        predictive_velocity_error = future_velocity - predicted_velocity

        predictive_acceleration = (
            future_acceleration
            + kp * predictive_position_error
            + kd * predictive_velocity_error
        )

        acceleration_command = (
            (1.0 - PREDICTIVE_BLEND) * baseline_acceleration
            + PREDICTIVE_BLEND * predictive_acceleration
        )

        acceleration_command[2] = np.clip(
            acceleration_command[2], -MAX_VERTICAL_ACCELERATION, MAX_VERTICAL_ACCELERATION
        )

        horizontal_norm = np.linalg.norm(acceleration_command[:2])
        if horizontal_norm > max_h_accel:                      # FIX 2
            acceleration_command[:2] *= max_h_accel / horizontal_norm

        feasible_force, desired_angles_raw, collective_actual, force_limited = (
            feasible_force_from_accel(
                acceleration_command, buoyancy, drag, efficiency, max_h_accel  # FIX 2
            )
        )

        collective_aerial = collective_actual / max(float(efficiency), 0.05)

        desired_angles = self.rate_limit_angles(desired_angles_raw)

        feedback_torque = attitude_scale * self.attitude_controller.update(
            desired_angles, angles, angular_rates
        )

        compensation_torque = np.cross(angular_rates, I @ angular_rates) + I_rate @ angular_rates

        raw_total_torque = compensation_torque + feedback_torque

        commanded_torque, torque_authority_scale, torque_authority_limited = (
            project_torque_with_collective(
                collective_aerial, feedback_torque, compensation_torque, arm, efficiency
            )
        )

        motor_thrusts, torque_saturated, torque_scale, collective_clipped = allocate_motors(
            collective_aerial, commanded_torque, arm, efficiency
        )

        return {
            "target_position": target_position,
            "target_velocity": target_velocity,
            "target_acceleration": target_acceleration,
            "baseline_acceleration": baseline_acceleration,
            "predictive_acceleration": predictive_acceleration,
            "commanded_acceleration": acceleration_command,
            "requested_force": (
                MASS * acceleration_command
                - np.array([0.0, 0.0, -MASS * GRAVITY])
                - buoyancy - drag
            ),
            "feasible_force": feasible_force,
            "desired_angles": desired_angles,
            "desired_angles_raw": desired_angles_raw,
            "feedback_torque": feedback_torque,
            "compensation_torque": compensation_torque,
            "raw_torque": raw_total_torque,
            "commanded_torque": commanded_torque,
            "requested_thrust": collective_aerial,
            "predictive_horizon": horizon,
            "predicted_position": predicted_position,
            "predicted_velocity": predicted_velocity,
            "predictive_position_error": predictive_position_error,
            "predictive_velocity_error": predictive_velocity_error,
            "torque_scale": torque_scale,
            "torque_saturated": torque_saturated,
            "collective_clipped": collective_clipped,
            "torque_authority_scale": torque_authority_scale,
            "torque_authority_limited": torque_authority_limited,
            "force_limited": force_limited,
            "motor_thrusts": motor_thrusts,
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "attitude_scale": attitude_scale,
            "max_horizontal_accel": max_h_accel,
        }


# =============================================================================
# IMPACT METRICS
# =============================================================================

def impact_metrics(time, error, impact_force):

    impact_indices = np.flatnonzero(impact_force > 0.0)

    if impact_indices.size == 0:
        return {
            "Impact pre-event RMS error (m)": np.nan,
            "Peak post-impact error (m)": np.nan,
            "Peak impact-induced error excursion (m)": np.nan,
            "Impact recovery threshold (m)": np.nan,
            "Impact response peak time (s)": np.nan,
            "Impact recovery time to 10% residual (s)": np.nan,
        }

    first_impact_index = int(impact_indices[0])
    first_impact_time = time[first_impact_index]

    baseline_mask = (
        (time >= first_impact_time - IMPACT_BASELINE_WINDOW)
        & (time < first_impact_time)
    )

    if np.any(baseline_mask):
        baseline_error = float(np.sqrt(np.mean(error[baseline_mask]**2)))
    else:
        baseline_error = float(error[first_impact_index])

    peak_force_index = int(np.argmax(impact_force))
    peak_force_time = time[peak_force_index]

    response_mask = (
        (time >= peak_force_time)
        & (time <= peak_force_time + IMPACT_RESPONSE_WINDOW)
    )

    response_indices = np.flatnonzero(response_mask)

    response_peak_index = int(
        response_indices[np.argmax(error[response_indices])]
    )

    peak_error = float(error[response_peak_index])
    impact_excursion = max(0.0, peak_error - baseline_error)

    recovery_threshold = baseline_error + IMPACT_RECOVERY_FRACTION * impact_excursion

    persistence_samples = max(1, int(np.ceil(IMPACT_PERSISTENCE / DT)))

    recovery_time = np.nan

    for i in range(response_peak_index + 1, len(time) - persistence_samples + 1):
        if np.all(error[i:i + persistence_samples] <= recovery_threshold):
            recovery_time = float(time[i] - time[response_peak_index])
            break

    return {
        "Impact pre-event RMS error (m)": baseline_error,
        "Peak post-impact error (m)": peak_error,
        "Peak impact-induced error excursion (m)": impact_excursion,
        "Impact recovery threshold (m)": recovery_threshold,
        "Impact response peak time (s)": float(time[response_peak_index]),
        "Impact recovery time to 10% residual (s)": recovery_time,
    }


# =============================================================================
# SIMULATION
# =============================================================================

def calculate_motor_torques(motor_thrusts, arm):

    a = arm / np.sqrt(2.0)

    return np.array([
        a * (motor_thrusts[0] - motor_thrusts[1] - motor_thrusts[2] + motor_thrusts[3]),
        a * (-motor_thrusts[0] - motor_thrusts[1] + motor_thrusts[2] + motor_thrusts[3]),
        KM * (motor_thrusts[0] - motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[3]),
    ])


def run_simulation():

    N = int(SIMULATION_TIME / DT)
    time = np.arange(N) * DT

    position = P0.copy()
    velocity = np.zeros(3)
    angles = np.zeros(3)
    angular_rates = np.zeros(3)
    previous_acceleration = np.zeros(3)

    motors = [Motor(), Motor(), Motor(), Motor()]

    sensor = SensorSimulator()
    estimator = MultiModalStateEstimator()
    controller = AdaptivePredictiveController()

    vector_names = [
        "position", "velocity", "angles", "angular_rates", "acceleration",
        "estimated_position", "estimated_velocity", "estimated_angles",
        "estimated_angular_rates", "estimated_acceleration_bias",
        "target_position", "target_velocity", "target_acceleration",
        "desired_angles", "requested_force", "feasible_force",
        "predicted_position", "predicted_velocity",
        "raw_torque", "commanded_torque", "torque",
    ]

    results = {name: np.zeros((N, 3)) for name in vector_names}

    scalar_names = [
        "requested_thrust", "commanded_thrust", "effective_thrust",
        "buoyancy", "drag", "immersion", "estimated_immersion",
        "propulsion_effectiveness", "morphology", "morphology_rate",
        "arm_length", "inertia_scale", "commanded_direction_error",
        "actual_direction_error", "torque_scale", "torque_authority_scale",
        "force_limit_scale", "impact_force", "impact_normal_speed",
        "position_estimation_error", "velocity_estimation_error",
        "attitude_estimation_error", "estimator_confidence",
        "position_covariance_trace", "kp_scale", "kd_scale",
        "attitude_scale", "predictive_horizon", "predictive_position_error",
        "predictive_velocity_error", "accel_norm", "max_horizontal_accel",
    ]

    for name in scalar_names:
        results[name] = np.zeros(N)

    boolean_names = [
        "torque_saturation", "torque_authority_limited", "collective_clipped",
        "force_limited", "impact_active", "gps_available",
        "vision_position_available", "depth_available", "sonar_available",
        "vision_attitude_available",
    ]

    for name in boolean_names:
        results[name] = np.zeros(N, dtype=bool)

    results["rpm"] = np.zeros((N, 4))
    results["motor_thrust"] = np.zeros((N, 4))

    for i, t in enumerate(time):

        (morphology, morphology_rate, _, arm, I, I_rate, inertia_scale) = morphology_parameters(t)

        (immersion, buoyancy_force, drag, drag_magnitude, efficiency) = medium_effects(
            position, velocity
        )

        measurements = sensor.measure(
            t, position, velocity, angles, angular_rates, previous_acceleration, immersion
        )

        estimator.update(measurements, DT)

        estimated_position = estimator.position
        estimated_velocity = estimator.velocity
        estimated_angles = estimator.angles
        estimated_angular_rates = estimator.angular_rates
        estimated_bias_body = estimator.bias_body

        (estimated_immersion, estimated_buoyancy, estimated_drag, _, estimated_efficiency) = (
            medium_effects(estimated_position, estimated_velocity)
        )

        controller_output = controller.control(
            t, estimated_position, estimated_velocity, estimated_angles, estimated_angular_rates,
            estimated_immersion, inertia_scale, arm, I, I_rate,
            estimated_efficiency, estimated_buoyancy, estimated_drag,
        )

        commanded_motor_thrusts = controller_output["motor_thrusts"]

        motor_rpm = np.array([
            motors[j].update(thrust_to_rpm(max(commanded_motor_thrusts[j], 0.0)), DT)
            for j in range(4)
        ])

        motor_thrust = np.array([thrust_from_rpm(rpm) for rpm in motor_rpm])
        effective_motor_thrust = motor_thrust * efficiency
        total_effective_thrust = np.sum(effective_motor_thrust)

        actual_torque = calculate_motor_torques(motor_thrust, arm) * efficiency
        rotational_damping = -WATER_ROTATIONAL_DAMPING * immersion * angular_rates
        total_torque = actual_torque + rotational_damping

        R = rotation_matrix(*angles)
        thrust_world = R @ np.array([0.0, 0.0, total_effective_thrust])

        (impact_force_vector, impact_force_magnitude, impact_normal_speed, impact_active) = (
            water_entry_impact(t, position, velocity)
        )

        total_force = (
            thrust_world
            + np.array([0.0, 0.0, -MASS * GRAVITY])
            + buoyancy_force
            + drag
            + impact_force_vector
        )

        acceleration = total_force / MASS

        requested_force = controller_output["feasible_force"]
        requested_norm = np.linalg.norm(requested_force)

        if requested_norm > 1e-12:
            commanded_attitude_R = rotation_matrix(*controller_output["desired_angles"])
            commanded_body_z = commanded_attitude_R[:, 2]

            command_direction_error = np.rad2deg(np.arccos(np.clip(
                np.dot(commanded_body_z, requested_force / requested_norm), -1.0, 1.0,
            )))

            actual_body_z = R[:, 2]
            actual_direction_error = np.rad2deg(np.arccos(np.clip(
                np.dot(actual_body_z, requested_force / requested_norm), -1.0, 1.0,
            )))
        else:
            command_direction_error = 0.0
            actual_direction_error = 0.0

        results["position"][i] = position
        results["velocity"][i] = velocity
        results["angles"][i] = angles
        results["angular_rates"][i] = angular_rates
        results["acceleration"][i] = acceleration

        results["estimated_position"][i] = estimated_position
        results["estimated_velocity"][i] = estimated_velocity
        results["estimated_angles"][i] = estimated_angles
        results["estimated_angular_rates"][i] = estimated_angular_rates
        results["estimated_acceleration_bias"][i] = estimated_bias_body

        (target_position, target_velocity, target_acceleration) = mission_trajectory(t)
        results["target_position"][i] = target_position
        results["target_velocity"][i] = target_velocity
        results["target_acceleration"][i] = target_acceleration

        results["desired_angles"][i] = controller_output["desired_angles"]
        results["requested_force"][i] = controller_output["requested_force"]
        results["feasible_force"][i] = controller_output["feasible_force"]
        results["predicted_position"][i] = controller_output["predicted_position"]
        results["predicted_velocity"][i] = controller_output["predicted_velocity"]
        results["raw_torque"][i] = controller_output["raw_torque"]
        results["commanded_torque"][i] = controller_output["commanded_torque"]
        results["torque"][i] = total_torque

        results["requested_thrust"][i] = controller_output["requested_thrust"]
        results["commanded_thrust"][i] = np.sum(commanded_motor_thrusts)
        results["effective_thrust"][i] = total_effective_thrust
        results["buoyancy"][i] = buoyancy_force[2]
        results["drag"][i] = drag_magnitude
        results["immersion"][i] = immersion
        results["estimated_immersion"][i] = estimated_immersion
        results["propulsion_effectiveness"][i] = efficiency
        results["morphology"][i] = morphology
        results["morphology_rate"][i] = morphology_rate
        results["arm_length"][i] = arm
        results["inertia_scale"][i] = inertia_scale
        results["commanded_direction_error"][i] = command_direction_error
        results["actual_direction_error"][i] = actual_direction_error
        results["torque_scale"][i] = controller_output["torque_scale"]
        results["torque_authority_scale"][i] = controller_output["torque_authority_scale"]
        results["max_horizontal_accel"][i] = controller_output["max_horizontal_accel"]

        if (
            controller_output["force_limited"]
            and np.linalg.norm(controller_output["requested_force"]) > 1e-12
        ):
            results["force_limit_scale"][i] = (
                np.linalg.norm(controller_output["feasible_force"])
                / (np.linalg.norm(controller_output["requested_force"]) + 1e-12)
            )
        else:
            results["force_limit_scale"][i] = 1.0

        results["impact_force"][i] = impact_force_magnitude
        results["impact_normal_speed"][i] = impact_normal_speed

        results["position_estimation_error"][i] = np.linalg.norm(estimated_position - position)
        results["velocity_estimation_error"][i] = np.linalg.norm(estimated_velocity - velocity)
        results["attitude_estimation_error"][i] = np.rad2deg(
            np.linalg.norm(wrap_angles(estimated_angles - angles))
        )

        results["estimator_confidence"][i] = estimator.confidence()
        results["position_covariance_trace"][i] = np.trace(
            estimator.position_filter.P[:3, :3]
        )

        results["kp_scale"][i] = controller_output["kp_scale"]
        results["kd_scale"][i] = controller_output["kd_scale"]
        results["attitude_scale"][i] = controller_output["attitude_scale"]
        results["predictive_horizon"][i] = controller_output["predictive_horizon"]
        results["predictive_position_error"][i] = np.linalg.norm(
            controller_output["predictive_position_error"]
        )
        results["predictive_velocity_error"][i] = np.linalg.norm(
            controller_output["predictive_velocity_error"]
        )
        results["accel_norm"][i] = np.linalg.norm(controller_output["commanded_acceleration"])

        results["torque_saturation"][i] = controller_output["torque_saturated"]
        results["torque_authority_limited"][i] = controller_output["torque_authority_limited"]
        results["collective_clipped"][i] = controller_output["collective_clipped"]
        results["force_limited"][i] = controller_output["force_limited"]
        results["impact_active"][i] = impact_active

        results["gps_available"][i] = measurements["gps_available"]
        results["vision_position_available"][i] = measurements["vision_position_available"]
        results["depth_available"][i] = measurements["depth_available"]
        results["sonar_available"][i] = measurements["sonar_available"]
        results["vision_attitude_available"][i] = measurements["vision_attitude_available"]

        results["rpm"][i] = motor_rpm
        results["motor_thrust"][i] = motor_thrust

        previous_acceleration = acceleration.copy()

        velocity = velocity + acceleration * DT
        position = position + velocity * DT

        angular_momentum = I @ angular_rates
        angular_acceleration = np.linalg.solve(
            I, (total_torque - I_rate @ angular_rates - np.cross(angular_rates, angular_momentum))
        )
        angular_rates = angular_rates + angular_acceleration * DT

        phi, theta, _ = angles
        cos_theta = np.cos(theta)
        sign_theta = np.sign(cos_theta) if cos_theta != 0.0 else 1.0
        cos_theta_safe = max(abs(cos_theta), 1e-5) * sign_theta

        E = np.array([
            [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi) / cos_theta_safe, np.cos(phi) / cos_theta_safe],
        ])

        angles = wrap_angles(angles + E @ angular_rates * DT)

    results["time"] = time
    return results


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(results):

    time = results["time"]
    position = results["position"]
    velocity = results["velocity"]
    target_position = results["target_position"]
    angles = results["angles"]

    position_error = np.linalg.norm(target_position - position, axis=1)
    horizontal_error = np.linalg.norm((target_position - position)[:, :2], axis=1)
    speed = np.linalg.norm(velocity, axis=1)

    underwater_mask = time >= SUBMERGED_START
    gps_denied_mask = time >= GPS_HANDOFF_END

    trajectory1_mask = (time >= TRAJ1_START) & (time < TRAJ1_END)
    trajectory2_mask = (time >= TRAJ2_START) & (time < TRAJ2_END)
    return_mask = (time >= RETURN_START) & (time < RETURN_END)
    final_mask = time >= FINAL_START
    pre_entry_mask = time < WATER_DESCENT_START
    water_entry_mask = (time >= WATER_DESCENT_START) & (time < WATER_DESCENT_END)
    submerged_stabilization_mask = (time >= SUBMERGED_START) & (time < SUBMERGED_END)

    impact = impact_metrics(time, position_error, results["impact_force"])

    underwater_estimation_error = results["position_estimation_error"][underwater_mask]
    gps_denied_estimation_error = results["position_estimation_error"][gps_denied_mask]

    # -------------------------------------------------------------------
    # FIX 3: sensor availability as a fraction of each sensor's OWN
    # scheduled update opportunities, not of total simulation ticks.
    # The original divided by len(mask) (all ticks in the window),
    # which makes a sensor firing on every one of its scheduled
    # windows read as a tiny percentage (verified: 96/5500 = 1.745%
    # for GPS, matching the originally reported bug exactly).
    # -------------------------------------------------------------------

    def availability_pct(available_flags, rate_hz, window_mask=None):
        scheduled = np.array([periodic_update(t, rate_hz) for t in time])
        if window_mask is not None:
            scheduled = scheduled & window_mask
        denom = scheduled.sum()
        if denom == 0:
            return 0.0
        numer = (available_flags & scheduled).sum()
        return float(100.0 * numer / denom)

    aerial_gps_mask = time < GPS_HANDOFF_END
    gps_availability_pct = availability_pct(
        results["gps_available"], GPS_RATE_HZ, aerial_gps_mask
    )
    vision_position_availability_pct = availability_pct(
        results["vision_position_available"], VISION_POSITION_RATE_HZ
    )
    depth_availability_pct = availability_pct(
        results["depth_available"], DEPTH_RATE_HZ
    )
    sonar_availability_pct = availability_pct(
        results["sonar_available"], SONAR_RATE_HZ, underwater_mask
    )
    vision_attitude_availability_pct = availability_pct(
        results["vision_attitude_available"], VISION_ATTITUDE_RATE_HZ
    )

    metrics = {

        "Maximum 3-D tracking error (m)": float(position_error.max()),
        "RMS 3-D tracking error (m)": float(np.sqrt(np.mean(position_error**2))),
        "Maximum horizontal tracking error (m)": float(horizontal_error.max()),
        "RMS horizontal tracking error (m)": float(np.sqrt(np.mean(horizontal_error**2))),
        "Maximum pre-entry tracking error (m)": float(position_error[pre_entry_mask].max()),
        "Maximum water-entry error (m)": float(position_error[water_entry_mask].max()),
        "Maximum submerged-stabilization error (m)": float(
            position_error[submerged_stabilization_mask].max()
        ),
        "RMS submerged-stabilization error (m)": float(
            np.sqrt(np.mean(position_error[submerged_stabilization_mask]**2))
        ),
        "Maximum trajectory-1 error (m)": float(position_error[trajectory1_mask].max()),
        "Maximum trajectory-2 error (m)": float(position_error[trajectory2_mask].max()),
        "Maximum return-phase error (m)": float(position_error[return_mask].max()),
        "Maximum final-stabilization error (m)": float(position_error[final_mask].max()),
        "RMS final-stabilization error (m)": float(
            np.sqrt(np.mean(position_error[final_mask]**2))
        ),
        "Maximum final-stabilization speed (m/s)": float(speed[final_mask].max()),
        "Maximum underwater speed (m/s)": float(speed[underwater_mask].max()),

        "Maximum roll (deg)": float(np.abs(np.rad2deg(angles[:, 0])).max()),
        "Maximum pitch (deg)": float(np.abs(np.rad2deg(angles[:, 1])).max()),
        "Maximum yaw (deg)": float(np.abs(np.rad2deg(angles[:, 2])).max()),

        "Maximum immersion fraction": float(results["immersion"].max()),
        "Minimum propulsion effectiveness": float(results["propulsion_effectiveness"].min()),
        "Maximum buoyancy (N)": float(results["buoyancy"].max()),
        "Maximum hydrodynamic drag (N)": float(results["drag"].max()),

        "Maximum water-entry impact force (N)": float(results["impact_force"].max()),
        "Water-entry impact impulse (N s)": float(np.trapz(results["impact_force"], time)),
        "Maximum water-entry normal speed (m/s)": float(results["impact_normal_speed"].max()),
        "Water-entry active duration (s)": float(results["impact_active"].sum() * DT),
        "Peak water-entry acceleration (m/s^2)": float(
            np.linalg.norm(results["acceleration"], axis=1)[water_entry_mask].max()
        ),
        "Water-entry impact peak time (s)": float(
            time[np.argmax(results["impact_force"])]
        ),

        **impact,

        "Maximum requested aerial-equivalent thrust (N)": float(results["requested_thrust"].max()),
        "Maximum commanded aerial thrust (N)": float(results["commanded_thrust"].max()),
        "Maximum effective thrust (N)": float(results["effective_thrust"].max()),
        "Maximum motor RPM": float(results["rpm"].max()),
        "RPM margin to MAX_RPM": float(MAX_RPM - results["rpm"].max()),
        "Maximum individual motor thrust (N)": float(results["motor_thrust"].max()),
        "Motor thrust margin (N)": float(max_motor_thrust() - results["motor_thrust"].max()),

        "Torque saturation events": int(results["torque_saturation"].sum()),
        "Torque authority-limited events": int(results["torque_authority_limited"].sum()),
        "Collective thrust clipping events": int(results["collective_clipped"].sum()),
        "Minimum torque allocation scale": float(results["torque_scale"].min()),
        "Mean torque authority scale": float(results["torque_authority_scale"].mean()),
        "Minimum torque authority scale": float(results["torque_authority_scale"].min()),
        "Fraction actuator-authority limited (%)": float(
            100.0 * np.mean(results["torque_authority_limited"])
        ),
        "Force feasibility limiting events": int(results["force_limited"].sum()),
        "Minimum force-feasibility scale": float(results["force_limit_scale"].min()),

        "Maximum commanded force-direction error (deg)": float(
            results["commanded_direction_error"].max()
        ),
        "Maximum actual thrust-direction error (deg)": float(
            results["actual_direction_error"].max()
        ),

        "Minimum arm length (m)": float(results["arm_length"].min()),
        "Maximum arm length (m)": float(results["arm_length"].max()),
        "Minimum inertia scale": float(results["inertia_scale"].min()),
        "Maximum inertia scale": float(results["inertia_scale"].max()),
        "Maximum morphology rate (1/s)": float(np.abs(results["morphology_rate"]).max()),

        "Maximum position estimation error (m)": float(
            results["position_estimation_error"].max()
        ),
        "RMS position estimation error (m)": float(
            np.sqrt(np.mean(results["position_estimation_error"]**2))
        ),
        "Maximum underwater position estimation error (m)": float(
            underwater_estimation_error.max()
        ),
        "RMS underwater position estimation error (m)": float(
            np.sqrt(np.mean(underwater_estimation_error**2))
        ),
        "Maximum GPS-denied position estimation error (m)": float(
            gps_denied_estimation_error.max()
        ),
        "RMS GPS-denied position estimation error (m)": float(
            np.sqrt(np.mean(gps_denied_estimation_error**2))
        ),
        "Maximum velocity estimation error (m/s)": float(
            results["velocity_estimation_error"].max()
        ),
        "RMS velocity estimation error (m/s)": float(
            np.sqrt(np.mean(results["velocity_estimation_error"]**2))
        ),
        "Maximum estimated accel-bias magnitude (m/s^2)": float(
            np.linalg.norm(results["estimated_acceleration_bias"], axis=1).max()
        ),
        "Injected accel-bias magnitude (m/s^2)": float(np.linalg.norm(IMU_ACCEL_BIAS_BODY)),
        "Final estimated accel-bias magnitude (m/s^2)": float(
            np.linalg.norm(results["estimated_acceleration_bias"][-1])
        ),
        "Maximum attitude estimation error (deg)": float(
            results["attitude_estimation_error"].max()
        ),
        "RMS attitude estimation error (deg)": float(
            np.sqrt(np.mean(results["attitude_estimation_error"]**2))
        ),

        "Minimum estimator confidence": float(results["estimator_confidence"].min()),
        "Mean estimator confidence": float(results["estimator_confidence"].mean()),

        # FIX 3 applied below -- see availability_pct() above
        "Aerial GPS availability (%)": gps_availability_pct,
        "GPS availability underwater (%)": 0.0,
        "Vision position availability (%)": vision_position_availability_pct,
        "Depth sensor availability (%)": depth_availability_pct,
        "Sonar-like availability underwater (%)": sonar_availability_pct,
        "Vision attitude availability (%)": vision_attitude_availability_pct,

        "Minimum adaptive position gain scale": float(results["kp_scale"].min()),
        "Maximum adaptive position gain scale": float(results["kp_scale"].max()),
        "Minimum adaptive velocity gain scale": float(results["kd_scale"].min()),
        "Maximum adaptive velocity gain scale": float(results["kd_scale"].max()),
        "Minimum adaptive attitude gain scale": float(results["attitude_scale"].min()),
        "Maximum adaptive attitude gain scale": float(results["attitude_scale"].max()),

        "Minimum predictive horizon (s)": float(results["predictive_horizon"].min()),
        "Maximum predictive horizon (s)": float(results["predictive_horizon"].max()),
        "Maximum predictive position error (m)": float(
            results["predictive_position_error"].max()
        ),
        "RMS predictive position error (m)": float(
            np.sqrt(np.mean(results["predictive_position_error"]**2))
        ),
        "Maximum predictive velocity error (m/s)": float(
            results["predictive_velocity_error"].max()
        ),
        "RMS predictive velocity error (m/s)": float(
            np.sqrt(np.mean(results["predictive_velocity_error"]**2))
        ),

        "Minimum horizontal accel ceiling (m/s^2)": float(
            results["max_horizontal_accel"].min()
        ),
        "Maximum horizontal accel ceiling (m/s^2)": float(
            results["max_horizontal_accel"].max()
        ),

        "Final position estimation error (m)": float(
            np.linalg.norm(results["estimated_position"][-1] - position[-1])
        ),
        "Final position error (m)": float(position_error[-1]),
        "Final X (m)": float(position[-1, 0]),
        "Final Y (m)": float(position[-1, 1]),
        "Final Z (m)": float(position[-1, 2]),
        "Final Vx (m/s)": float(velocity[-1, 0]),
        "Final Vy (m/s)": float(velocity[-1, 1]),
        "Final Vz (m/s)": float(velocity[-1, 2]),
        "Final roll (deg)": float(np.rad2deg(angles[-1, 0])),
        "Final pitch (deg)": float(np.rad2deg(angles[-1, 1])),
        "Final yaw (deg)": float(np.rad2deg(angles[-1, 2])),
    }

    return metrics


# =============================================================================
# REGRESSION CHECK
# =============================================================================

def validate_regression():
    """
    Run this after any further changes to confirm you haven't silently
    broken what was verified here. Prints the key numbers from the
    FIX 1 + FIX 2 experiment so you can compare against:
        final position error ~= 0.2397 m
        mean torque authority scale ~= 0.486 (unchanged by these fixes --
            that metric reflects something different than final tracking
            quality, see module docstring)
    If you change ACCEL_FLOOR_FACTOR, re-run the sweep in the module
    docstring yourself -- the cliff location is NOT guaranteed to stay
    at ~0.32-0.34 if you change the mission waypoints/timing.
    """
    results = run_simulation()
    final_err = float(np.linalg.norm(results["target_position"][-1] - results["position"][-1]))
    print(f"[validate_regression] final position error: {final_err:.4f} m "
          f"(expected ~0.2397 m with ACCEL_FLOOR_FACTOR={ACCEL_FLOOR_FACTOR})")
    assert final_err < 0.30, (
        f"Regression: final position error {final_err:.4f}m is worse than the "
        f"~0.24m verified after FIX 1 + FIX 2. Check recent changes."
    )
    print("[validate_regression] PASSED")


# =============================================================================
# PLOTS
# =============================================================================

def savefig(figure, filename):
    path = os.path.join(RESULTS_DIRECTORY, filename)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(results):

    time = results["time"]
    position = results["position"]
    estimated_position = results["estimated_position"]
    target = results["target_position"]

    # -- Position tracking --
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for j, label in enumerate(["X", "Y", "Z"]):
        axes[j].plot(time, position[:, j], label=f"True {label}")
        axes[j].plot(time, estimated_position[:, j], ":", label=f"Estimated {label}")
        axes[j].plot(time, target[:, j], "--", label=f"Target {label}")
        if j == 2:
            axes[j].axhline(0.0, linestyle="-.", label="Water surface")
        axes[j].set_ylabel(f"{label} (m)")
        axes[j].grid()
        axes[j].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("MorphoAqua - Stage 5B-4 (corrected) True, Estimated and Target Position")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_position_tracking.png")

    # -- 3-D trajectory --
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for key, style, label in [
        ("position", "-", "True trajectory"),
        ("estimated_position", ":", "Estimated trajectory"),
        ("predicted_position", "-.", "Predicted trajectory"),
        ("target_position", "--", "Target trajectory"),
    ]:
        path = results[key]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], style, label=label)
    ax.scatter([position[0, 0]], [position[0, 1]], [position[0, 2]], s=60, label="Mission start")
    ax.scatter([position[-1, 0]], [position[-1, 1]], [position[-1, 2]], s=60, label="Final true state")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.grid()
    ax.set_title("MorphoAqua - Stage 5B-4 (corrected) Adaptive / Predictive 3-D Trajectory")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_3D_adaptive_predictive_trajectory.png")

    # -- Control / medium response --
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
    for j, label in enumerate(["Vx", "Vy", "Vz"]):
        axes[0].plot(time, results["velocity"][:, j], label=label)
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].legend()
    axes[0].grid()

    actual_angles = results["angles"]
    desired_angles = results["desired_angles"]
    axes[1].plot(time, np.rad2deg(actual_angles[:, 0]), label="Roll")
    axes[1].plot(time, np.rad2deg(desired_angles[:, 0]), "--", label="Desired roll")
    axes[1].plot(time, np.rad2deg(actual_angles[:, 1]), label="Pitch")
    axes[1].plot(time, np.rad2deg(desired_angles[:, 1]), "--", label="Desired pitch")
    axes[1].plot(time, np.rad2deg(actual_angles[:, 2]), label="Yaw")
    axes[1].set_ylabel("Angle (deg)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid()

    axes[2].plot(time, results["requested_thrust"], label="Requested aerial-equivalent thrust")
    axes[2].plot(time, results["commanded_thrust"], "--", label="Commanded aerial thrust")
    axes[2].plot(time, results["effective_thrust"], label="Effective thrust")
    axes[2].axhline(MASS * GRAVITY, linestyle=":", label="Weight")
    axes[2].set_ylabel("Thrust (N)")
    axes[2].legend()
    axes[2].grid()

    axes[3].plot(time, results["raw_torque"][:, 0], ":", label="Raw roll torque")
    axes[3].plot(time, results["raw_torque"][:, 1], "--", label="Raw pitch torque")
    axes[3].plot(time, results["torque"][:, 0], label="Actual roll torque")
    axes[3].plot(time, results["torque"][:, 1], label="Actual pitch torque")
    axes[3].set_ylabel("Torque (N m)")
    axes[3].legend(fontsize=8)
    axes[3].grid()

    for j in range(4):
        axes[4].plot(time, results["rpm"][:, j], label=f"Motor {j + 1}")
    axes[4].axhline(MAX_RPM, linestyle=":", label="MAX_RPM")
    axes[4].set_ylabel("RPM")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(ncol=5, fontsize=8)
    axes[4].grid()

    fig.suptitle("MorphoAqua - Stage 5B-4 (corrected) Actuator-Aware Adaptive Control and Medium Response")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_control_medium_response.png")

    # -- Adaptive / predictive diagnostics --
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)

    axes[0].plot(time, results["kp_scale"], label="Adaptive position gain scale")
    axes[0].plot(time, results["kd_scale"], "--", label="Adaptive velocity gain scale")
    axes[0].axhline(1.0, linestyle=":", label="Baseline")
    axes[0].set_ylabel("Gain scale")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, results["attitude_scale"], label="Adaptive attitude gain scale")
    axes[1].axhline(1.0, linestyle=":", label="Baseline")
    axes[1].set_ylabel("Attitude scale")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(time, results["predictive_horizon"], label="Predictive horizon")
    axes[2].set_ylabel("Horizon (s)")
    axes[2].legend()
    axes[2].grid()

    axes[3].plot(time, results["predictive_position_error"], label="Predictive position error")
    axes[3].plot(time, results["predictive_velocity_error"], "--", label="Predictive velocity error")
    axes[3].set_ylabel("Prediction error")
    axes[3].legend()
    axes[3].grid()

    axes[4].plot(time, results["torque_authority_scale"], label="Torque authority scale")
    axes[4].plot(time, results["torque_scale"], ":", label="Final mixer scale")
    axes[4].plot(time, results["force_limit_scale"], "--", label="Force feasibility scale")
    axes[4].plot(time, results["max_horizontal_accel"] / MAX_HORIZONTAL_ACCELERATION_BASE,
                 "-.", label="Horizontal accel ceiling / base (FIX 2)")
    axes[4].axhline(1.0, linestyle=":", label="No limiting")
    axes[4].set_ylabel("Scale")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(fontsize=8)
    axes[4].grid()

    fig.suptitle("MorphoAqua - Stage 5B-4 (corrected) Adaptive Scheduling, Prediction and Actuator Feasibility")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_adaptive_predictive_response.png")

    # -- Direction / impact --
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(time, results["commanded_direction_error"], label="Commanded direction error")
    axes[0].plot(time, results["actual_direction_error"], "--", label="Actual thrust-direction error")
    axes[0].set_ylabel("Direction error (deg)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, results["impact_force"], label="Water-entry impact force")
    axes[1].plot(time, results["impact_normal_speed"], "--", label="Entry normal speed")
    axes[1].set_ylabel("Impact / speed")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(time, results["torque_authority_scale"], label="Torque authority scale")
    axes[2].plot(time, results["force_limit_scale"], "--", label="Force feasibility scale")
    axes[2].plot(time, results["accel_norm"], ":", label="Commanded acceleration magnitude")
    axes[2].set_ylabel("Scale / acceleration")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid()

    fig.suptitle("MorphoAqua - Stage 5B-4 (corrected) Direction Feasibility and Actuator Response")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_direction_impact_response.png")

    # -- Morphology / inertia --
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time, results["arm_length"], label="Arm length")
    axes[0].plot(time, results["morphology"], "--", label="Morphology")
    axes[0].plot(time, results["morphology_rate"], ":", label="Morphology rate")
    axes[0].set_ylabel("Morphology / length")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, results["inertia_scale"], label="Inertia scale")
    axes[1].set_ylabel("I / I_nominal")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid()

    fig.suptitle("MorphoAqua - Stage 5B-4 (corrected) Morphology and Time-Varying Inertia")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_morphology_inertia.png")

    # -- Sensors / estimation --
    fig, axes = plt.subplots(5, 1, figsize=(13, 17), sharex=True)

    axes[0].plot(time, results["position_estimation_error"], label="3-D position estimation error")
    axes[0].plot(time, results["velocity_estimation_error"], "--", label="Velocity estimation error")
    axes[0].set_ylabel("Error")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, results["attitude_estimation_error"], label="Attitude estimation error")
    axes[1].set_ylabel("Angle error (deg)")
    axes[1].legend()
    axes[1].grid()

    sensor_plot_definitions = [
        ("gps_available", "GPS"),
        ("vision_position_available", "Vision position"),
        ("depth_available", "Depth"),
        ("sonar_available", "Sonar-like position"),
        ("vision_attitude_available", "Vision attitude"),
    ]
    for key, label in sensor_plot_definitions:
        axes[2].plot(time, results[key].astype(float), label=label)
    axes[2].set_ylabel("Available (0/1)")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid()

    axes[3].plot(time, results["estimator_confidence"], label="Estimator confidence")
    axes[3].plot(time, results["position_covariance_trace"], "--", label="Position covariance trace")
    axes[3].set_ylabel("Confidence / covariance")
    axes[3].legend()
    axes[3].grid()

    bias = results["estimated_acceleration_bias"]
    axes[4].plot(time, bias[:, 0], label="Estimated bax")
    axes[4].plot(time, bias[:, 1], label="Estimated bay")
    axes[4].plot(time, bias[:, 2], label="Estimated baz")
    axes[4].axhline(IMU_ACCEL_BIAS_BODY[0], linestyle=":", label="True bax")
    axes[4].axhline(IMU_ACCEL_BIAS_BODY[1], linestyle=":", label="True bay")
    axes[4].axhline(IMU_ACCEL_BIAS_BODY[2], linestyle=":", label="True baz")
    axes[4].set_ylabel("Accel bias (m/s^2)")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(ncol=3, fontsize=8)
    axes[4].grid()

    fig.suptitle("MorphoAqua - Stage 5B-4 (corrected) Sensors and State Estimation\n"
                 "(attitude drift after ~t=14s is expected -- see module docstring)")
    plt.tight_layout()
    savefig(fig, "Stage_5B4_sensor_estimation.png")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 76)
    print("MORPHOAQUA - STAGE 5B-4 (CORRECTED)")
    print("ACTUATOR-AWARE ADAPTIVE / PREDICTIVE CONTROL")
    print("=" * 76)
    print()
    print("Corrections applied vs. original Stage 5B-4 (see module docstring")
    print("for the experimental evidence behind each):")
    print("  FIX 1: removed backwards immersion-driven kd_scale increase")
    print("  FIX 2: efficiency-aware horizontal acceleration ceiling")
    print(f"         (ACCEL_FLOOR_FACTOR={ACCEL_FLOOR_FACTOR}, chosen with margin")
    print("          from an experimentally-found instability cliff)")
    print("  FIX 3: corrected sensor-availability percentage denominator")
    print("  FIX 4: non-degenerate adaptive attitude gain scheduling")
    print("  NOTE:  attitude drift after submersion is documented, not patched")
    print("         -- see module docstring for why.")
    print()

    results = run_simulation()
    metrics = calculate_metrics(results)

    print("STAGE 5B-4 (CORRECTED) PERFORMANCE METRICS")
    print("-" * 76)

    for key, value in metrics.items():
        if isinstance(value, int):
            formatted = str(value)
        elif isinstance(value, float) and np.isnan(value):
            formatted = "NaN"
        else:
            formatted = f"{value:.6f}"
        print(f"{key:<62}: {formatted}")

    print("-" * 76)

    # Single-run regression gate. Do not call validate_regression() here
    # because that helper intentionally performs a second full simulation.
    final_error = float(metrics["Final position error (m)"])
    if final_error >= 0.30:
        raise RuntimeError(
            f"Stage 5B-4 regression failed: final position error = {final_error:.4f} m"
        )
    print(
        f"[REGRESSION] PASS: final position error = {final_error:.4f} m < 0.30 m"
    )

    np.savez_compressed(
        os.path.join(RESULTS_DIRECTORY, "Stage_5B4_final_results.npz"),
        **results,
    )

    make_plots(results)

    print()
    print("Results saved to:")
    for filename in [
        "Stage_5B4_position_tracking.png",
        "Stage_5B4_3D_adaptive_predictive_trajectory.png",
        "Stage_5B4_control_medium_response.png",
        "Stage_5B4_adaptive_predictive_response.png",
        "Stage_5B4_direction_impact_response.png",
        "Stage_5B4_morphology_inertia.png",
        "Stage_5B4_sensor_estimation.png",
        "Stage_5B4_final_results.npz",
    ]:
        print(os.path.join(RESULTS_DIRECTORY, filename))

    print()
    print("Stage 5B-4 (corrected) simulation completed.")


if __name__ == "__main__":
    main()