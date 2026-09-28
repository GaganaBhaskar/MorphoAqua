"""
MorphoAqua - Stage 1B
Closed-loop altitude control

Mission:

0 m
 ↓
Takeoff
 ↓
2 m
 ↓
Hover
 ↓
0 m
 ↓
Land
"""

import numpy as np
import matplotlib.pyplot as plt

from robot_parameters import (
    MASS,
    GRAVITY,
    DT,
    SIMULATION_TIME,
    TARGET_ALTITUDE,
    ALTITUDE_KP,
    ALTITUDE_KI,
    ALTITUDE_KV,
    MAX_RPM,
    KF,
)

from controller import AltitudeController
from motor_model import Motor, thrust_from_rpm


def thrust_to_rpm(thrust):

    if thrust <= 0:
        return 0.0

    omega = np.sqrt(
        thrust / KF
    )

    rpm = (
        omega * 60.0
        / (2.0 * np.pi)
    )

    return np.clip(
        rpm,
        0,
        MAX_RPM
    )


def run_simulation():

    # =========================================
    # INITIAL CONDITIONS
    # =========================================

    altitude = 0.0
    vertical_velocity = 0.0

    # =========================================
    # MOTORS
    # =========================================

    motors = [
        Motor(),
        Motor(),
        Motor(),
        Motor()
    ]

    # =========================================
    # CONTROLLER
    # =========================================

    max_motor_thrust = (
        KF *
        (
            MAX_RPM
            * 2.0
            * np.pi
            / 60.0
        ) ** 2
    )

    max_total_thrust = (
        4.0 * max_motor_thrust
    )

    controller = AltitudeController(
        kp=ALTITUDE_KP,
        ki=ALTITUDE_KI,
        kv=ALTITUDE_KV,
        mass=MASS,
        gravity=GRAVITY,
        max_thrust=max_total_thrust
    )

    # =========================================
    # STORAGE
    # =========================================

    steps = int(
        SIMULATION_TIME / DT
    )

    time = np.arange(steps) * DT

    altitude_history = np.zeros(steps)
    velocity_history = np.zeros(steps)

    target_history = np.zeros(steps)

    thrust_history = np.zeros(steps)

    rpm_history = np.zeros(
        (steps, 4)
    )

    # =========================================
    # LOOP
    # =========================================

    for i, t in enumerate(time):

        # -------------------------------------
        # Mission profile
        # -------------------------------------

        if t < 2.0:

            target_altitude = 2.0

        elif t < 7.0:

            target_altitude = 2.0

        else:

            target_altitude = 0.0

        target_history[i] = target_altitude

        # -------------------------------------
        # Controller
        # -------------------------------------

        desired_total_thrust = controller.update(
            target_altitude,
            altitude,
            vertical_velocity,
            DT
        )

        # -------------------------------------
        # Equal thrust distribution
        # -------------------------------------

        desired_motor_thrust = (
            desired_total_thrust / 4.0
        )

        commanded_rpm = thrust_to_rpm(
            desired_motor_thrust
        )

        # -------------------------------------
        # Motor dynamics
        # -------------------------------------

        actual_rpms = []

        for motor in motors:

            actual_rpm = motor.update(
                commanded_rpm,
                DT
            )

            actual_rpms.append(
                actual_rpm
            )

        actual_rpms = np.array(
            actual_rpms
        )

        # -------------------------------------
        # Actual thrust
        # -------------------------------------

        actual_thrusts = np.array([
            thrust_from_rpm(rpm)
            for rpm in actual_rpms
        ])

        total_actual_thrust = np.sum(
            actual_thrusts
        )

        # -------------------------------------
        # Vertical dynamics
        # -------------------------------------

        net_force = (
            total_actual_thrust
            - MASS * GRAVITY
        )

        acceleration = (
            net_force / MASS
        )

        # -------------------------------------
        # Integrate
        # -------------------------------------

        vertical_velocity += (
            acceleration * DT
        )

        altitude += (
            vertical_velocity * DT
        )

        # -------------------------------------
        # Ground constraint
        # -------------------------------------

        if altitude <= 0:

            altitude = 0.0

            if vertical_velocity < 0:

                vertical_velocity = 0.0

        # -------------------------------------
        # Store
        # -------------------------------------

        altitude_history[i] = altitude

        velocity_history[i] = (
            vertical_velocity
        )

        thrust_history[i] = (
            total_actual_thrust
        )

        rpm_history[i, :] = (
            actual_rpms
        )

    return (
        time,
        altitude_history,
        velocity_history,
        target_history,
        thrust_history,
        rpm_history
    )


def plot_results(
    time,
    altitude,
    velocity,
    target,
    thrust,
    rpm
):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 12),
        sharex=True
    )

    # =========================================
    # ALTITUDE
    # =========================================

    axes[0].plot(
        time,
        altitude,
        label="Actual altitude"
    )

    axes[0].plot(
        time,
        target,
        "--",
        label="Target altitude"
    )

    axes[0].set_ylabel(
        "Altitude (m)"
    )

    axes[0].set_title(
        "MorphoAqua - Stage 1B"
    )

    axes[0].legend()
    axes[0].grid(True)

    # =========================================
    # VELOCITY
    # =========================================

    axes[1].plot(
        time,
        velocity
    )

    axes[1].set_ylabel(
        "Vertical velocity (m/s)"
    )

    axes[1].grid(True)

    # =========================================
    # THRUST
    # =========================================

    axes[2].plot(
        time,
        thrust,
        label="Actual thrust"
    )

    axes[2].axhline(
        MASS * GRAVITY,
        linestyle="--",
        label="Hover thrust"
    )

    axes[2].set_ylabel(
        "Thrust (N)"
    )

    axes[2].legend()
    axes[2].grid(True)

    # =========================================
    # RPM
    # =========================================

    for i in range(4):

        axes[3].plot(
            time,
            rpm[:, i],
            label=f"Motor {i+1}"
        )

    axes[3].set_xlabel(
        "Time (s)"
    )

    axes[3].set_ylabel(
        "RPM"
    )

    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()

    plt.show()


def main():

    print("=" * 60)
    print("MORPHOAQUA - STAGE 1B")
    print("IMPROVED CLOSED-LOOP ALTITUDE CONTROL")
    print("=" * 60)

    results = run_simulation()

    (
        time,
        altitude,
        velocity,
        target,
        thrust,
        rpm
    ) = results

    print("\nSimulation complete.")

    print(
        f"Maximum altitude : "
        f"{np.max(altitude):.3f} m"
    )

    print(
        f"Final altitude   : "
        f"{altitude[-1]:.3f} m"
    )

    print(
        f"Maximum velocity : "
        f"{np.max(np.abs(velocity)):.3f} m/s"
    )

    print(
        f"Final velocity   : "
        f"{velocity[-1]:.3f} m/s"
    )

    print(
        f"Final motor RPM  : "
        f"{rpm[-1, 0]:.2f}"
    )

    plot_results(
        time,
        altitude,
        velocity,
        target,
        thrust,
        rpm
    )


if __name__ == "__main__":
    main()