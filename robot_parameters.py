


import numpy as np

# =========================
# ROBOT PARAMETERS
# =========================

MASS = 1.5                 # kg
GRAVITY = 9.81             # m/s^2

ARM_LENGTH = 0.25         # m
BODY_LENGTH = 0.25        # m
BODY_WIDTH = 0.20         # m
BODY_HEIGHT = 0.10        # m

# Rotor parameters
ROTOR_DIAMETER = 0.20     # m
MAX_RPM = 8000

# Approximate thrust coefficient.
# This is a simulation parameter and will later
# be replaced/refined using a motor/propeller model.
KF = 1.0e-5

# Approximate drag coefficient
CD = 0.8

# Reference projected area
REFERENCE_AREA = 0.05     # m^2

# Fluid properties
AIR_DENSITY = 1.225       # kg/m^3
WATER_DENSITY = 1000.0    # kg/m^3

# =========================
# MORPHOLOGY
# =========================

# Arm configuration angle in degrees
MORPHOLOGY_ANGLE = 90.0

# =========================
# INERTIA
# =========================

# Approximate diagonal inertia matrix
# These are initial simulation assumptions.
IXX = 0.030
IYY = 0.030
IZZ = 0.050

INERTIA = np.diag([IXX, IYY, IZZ])

# =========================
# SIMULATION
# =========================

DT = 0.002              # seconds
SIMULATION_TIME = 10.0  # seconds

# =========================
# CONTROL PARAMETERS
# =========================

TARGET_ALTITUDE = 2.0       # m

# Maximum commanded total thrust
MAX_TOTAL_THRUST = 4.0 * (
    KF * (MAX_RPM * 2.0 * np.pi / 60.0) ** 2
)

# =========================
# ALTITUDE CONTROLLER
# =========================

TARGET_ALTITUDE = 2.0

ALTITUDE_KP = 6.0
ALTITUDE_KI = 0.5
ALTITUDE_KV = 4.0