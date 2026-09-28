"""
MorphoAqua - Stage 1B
Closed-loop altitude control

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
    ALTITUDE_KD,
    MAX_RPM,
)

from controller import PIDController

from motor_model import (
    Motor,
    thrust_from_rpm,
)


def run_simulation():

    # =========================================
    # INITIAL STATE
    # =========================================

    altitude = 0.0
    vertical_velocity = 0.0

    # Four motors
    motors = [
        Motor(),
        Motor(),
        Motor(),
        Motor()
    ]

    # =========================================
    # PID CONTROLLER
    # =========================================

    pid = PIDController(
        kp=ALTITUDE_KP,
        ki=ALTITUDE_KI,
        kd=ALTITUDE_KD,

        output_min=0.0,
        output_max=MAX_RPM
    )

    # =========================================
    # DATA STORAGE
    # =========================================

    steps = int(SIMULATION_TIME / DT)

    time = np.arange(steps) * DT

    altitude_history = np.zeros(steps)
    velocity_history = np.zeros(steps)

    target_history = np.zeros(steps)

    thrust_history = np.zeros(steps)

    rpm_history = np.zeros((steps, 4))

    # =========================================
    # SIMULATION LOOP
    # =========================================

    for i, t in enumerate(time):

        # -------------------------------------
        # Desired altitude
        # -------------------------------------

        if t < 2.0:

            # Takeoff
            target_altitude = 2.0

        elif t < 7.0:

            # Hover
            target_altitude = 2.0

        else:

            # Landing
            target_altitude = 0.0

        target_history[i] = target_altitude

        # -------------------------------------
        # Controller
        # -------------------------------------

        # PID output is additional thrust command
        thrust_command = pid.update(
            target_altitude,
            altitude,
            DT
        )

        # -------------------------------------
        # Add gravity compensation
        # -------------------------------------

        required_hover_thrust = MASS * GRAVITY

        total_desired_thrust = (
            required_hover_thrust
            + thrust_command
        )

        # -------------------------------------
        # Convert total thrust to
        # per-motor thrust
        # -------------------------------------

        motor_thrust = (
            total_desired_thrust / 4.0
        )

        # Convert thrust to RPM
        omega = np.sqrt(
            max(motor_thrust, 0.0)
            / 1.0e-5
        )

        commanded_rpm = (
            omega * 60.0
            / (2.0 * np.pi)
        )

        commanded_rpm = np.clip(
            commanded_rpm,
            0,
            MAX_RPM
        )

        # -------------------------------------
        # Update motor dynamics
        # -------------------------------------

        actual_rpms = []

        for motor in motors:

            rpm = motor.update(
                commanded_rpm,
                DT
            )

            actual_rpms.append(rpm)

        actual_rpms = np.array(actual_rpms)

        # -------------------------------------
        # Calculate actual thrust
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
        # Integrate motion
        # -------------------------------------

        vertical_velocity += (
            acceleration * DT
        )

        altitude += (
            vertical_velocity * DT
        )

        # Prevent falling below ground
        if altitude < 0:

            altitude = 0

            if vertical_velocity < 0:
                vertical_velocity = 0

        # -------------------------------------
        # Store results
        # -------------------------------------

        altitude_history[i] = altitude
        velocity_history[i] = vertical_velocity

        thrust_history[i] = total_actual_thrust

        rpm_history[i, :] = actual_rpms

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
        "MorphoAqua - Stage 1B Closed-Loop Flight"
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
        thrust
    )

    axes[2].axhline(
        MASS * GRAVITY,
        linestyle="--",
        label="Hover thrust"
    )

    axes[2].set_ylabel(
        "Total thrust (N)"
    )

    axes[2].legend()
    axes[2].grid(True)

    # =========================================
    # MOTOR RPM
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
    print("CLOSED-LOOP ALTITUDE CONTROL")
    print("=" * 60)

    (
        time,
        altitude,
        velocity,
        target,
        thrust,
        rpm
    ) = run_simulation()

    print("\nSimulation complete.")

    print(
        f"Maximum altitude: "
        f"{np.max(altitude):.3f} m"
    )

    print(
        f"Final altitude: "
        f"{altitude[-1]:.3f} m"
    )

    print(
        f"Maximum velocity: "
        f"{np.max(np.abs(velocity)):.3f} m/s"
    )

    print(
        f"Final motor RPM: "
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