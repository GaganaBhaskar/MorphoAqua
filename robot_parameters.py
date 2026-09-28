"""
MorphoAqua - Stage 1C
6-DOF Rigid Body Simulation Parameters
"""

import numpy as np


# ============================================================
# ROBOT PARAMETERS
# ============================================================

MASS = 1.5
GRAVITY = 9.81

ARM_LENGTH = 0.25

BODY_LENGTH = 0.25
BODY_WIDTH = 0.20
BODY_HEIGHT = 0.10


# ============================================================
# ROTOR PARAMETERS
# ============================================================

ROTOR_DIAMETER = 0.20

MAX_RPM = 8000

# Thrust coefficient used by the simulation model.
# This is an assumed model parameter, not an experimental
# motor/propeller measurement.
KF = 1.0e-5

# Motor reaction torque coefficient.
KM = 1.5e-7


# ============================================================
# AIR PARAMETERS
# ============================================================

AIR_DENSITY = 1.225

CD = 0.8

REFERENCE_AREA = 0.05


# ============================================================
# WATER PARAMETERS
# ============================================================

WATER_DENSITY = 1000.0


# ============================================================
# MASS MOMENT OF INERTIA
# ============================================================

IXX = 0.030
IYY = 0.030
IZZ = 0.050

INERTIA = np.diag([
    IXX,
    IYY,
    IZZ
])


# ============================================================
# SIMULATION
# ============================================================

DT = 0.002

SIMULATION_TIME = 12.0


# ============================================================
# POSITION MISSION
# ============================================================

TARGET_X = 0.0

TARGET_Y = 0.0

TARGET_Z = 2.0

TARGET_YAW = 0.0


# ============================================================
# POSITION CONTROLLER
# ============================================================

POSITION_KP_X = 2.0

POSITION_KP_Y = 2.0

POSITION_KP_Z = 6.0


POSITION_KD_X = 2.5

POSITION_KD_Y = 2.5

POSITION_KD_Z = 4.0


# ============================================================
# ATTITUDE CONTROLLER
# ============================================================

ATTITUDE_KP_ROLL = 4.0

ATTITUDE_KP_PITCH = 4.0

ATTITUDE_KP_YAW = 2.0


ATTITUDE_KD_ROLL = 0.8

ATTITUDE_KD_PITCH = 0.8

ATTITUDE_KD_YAW = 0.5


# ============================================================
# ANGLE LIMITS
# ============================================================

MAX_ROLL = np.deg2rad(20.0)

MAX_PITCH = np.deg2rad(20.0)


# ============================================================
# GROUND
# ============================================================

GROUND_ALTITUDE = 0.0