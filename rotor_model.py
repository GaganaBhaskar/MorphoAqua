"""
MorphoAqua - Stage 1
Rotor thrust model
"""

import numpy as np
from robot_parameters import KF, MAX_RPM


def rpm_to_rad_s(rpm):
    """Convert RPM to rad/s."""
    return rpm * 2.0 * np.pi / 60.0


def rad_s_to_rpm(omega):
    """Convert rad/s to RPM."""
    return omega * 60.0 / (2.0 * np.pi)


def rotor_thrust(omega):
    """
    Calculate rotor thrust.

    T = Kf * omega^2
    """

    return KF * omega ** 2


def total_thrust(rotor_rpm):
    """
    Calculate total thrust from four rotor speeds.

    rotor_rpm:
        [RPM1, RPM2, RPM3, RPM4]
    """

    rotor_rpm = np.asarray(rotor_rpm)

    omega = rpm_to_rad_s(rotor_rpm)

    thrusts = rotor_thrust(omega)

    return np.sum(thrusts), thrusts


def hover_rpm(mass, gravity):
    """
    Calculate ideal rotor RPM required for hover.
    """

    required_total_thrust = mass * gravity

    thrust_per_rotor = required_total_thrust / 4.0

    omega = np.sqrt(thrust_per_rotor / KF)

    rpm = rad_s_to_rpm(omega)

    return rpm