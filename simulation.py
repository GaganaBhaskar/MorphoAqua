"""
MorphoAqua - Stage 1
Basic quadrotor simulation
"""

import numpy as np
import matplotlib.pyplot as plt

from robot_parameters import (
    MASS,
    GRAVITY,
    DT,
    SIMULATION_TIME,
    INERTIA,
)

from rotor_model import (
    total_thrust,
    hover_rpm,
)

from dynamics import (
    translational_acceleration,
)


def run_simulation(
    rpm,
    initial_state=None,
    simulation_time=SIMULATION_TIME,
):
    """
    Run a basic translational simulation.

    State:

    [x, y, z,
     vx, vy, vz,
     roll, pitch, yaw,
     p, q, r]
    """

    steps = int(simulation_time / DT)

    if initial_state is None:
        state = np.zeros(12)
    else:
        state = np.array(initial_state, dtype=float)

    time = np.arange(steps) * DT

    history = np.zeros((steps, 12))
    thrust_history = np.zeros(steps)

    for i in range(steps):

        # --------------------------------
        # Rotor thrust
        # --------------------------------

        total_T, _ = total_thrust(
            [rpm, rpm, rpm, rpm]
        )

        # --------------------------------
        # Current state
        # --------------------------------

        position = state[0:3]

        velocity = state[3:6]

        roll = state[6]
        pitch = state[7]
        yaw = state[8]

        # --------------------------------
        # Translational dynamics
        # --------------------------------

        acceleration = translational_acceleration(
            velocity,
            roll,
            pitch,
            yaw,
            total_T,
        )

        # --------------------------------
        # Euler integration
        # --------------------------------

        state[0:3] += velocity * DT

        state[3:6] += acceleration * DT

        history[i] = state

        thrust_history[i] = total_T

    return time, history, thrust_history


def plot_results(
    time,
    history,
    thrust_history,
    title,
):
    """
    Plot simulation results.
    """

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # Position
    axes[0].plot(
        time,
        history[:, 0],
        label="X"
    )

    axes[0].plot(
        time,
        history[:, 1],
        label="Y"
    )

    axes[0].plot(
        time,
        history[:, 2],
        label="Z"
    )

    axes[0].set_ylabel("Position (m)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)

    # Velocity
    axes[1].plot(
        time,
        history[:, 3],
        label="Vx"
    )

    axes[1].plot(
        time,
        history[:, 4],
        label="Vy"
    )

    axes[1].plot(
        time,
        history[:, 5],
        label="Vz"
    )

    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()
    axes[1].grid(True)

    # Thrust
    axes[2].plot(
        time,
        thrust_history,
    )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Total thrust (N)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def main():

    # --------------------------------
    # Calculate hover RPM
    # --------------------------------

    rpm = hover_rpm(
        MASS,
        GRAVITY
    )

    print("=" * 50)
    print("MORPHOAQUA - STAGE 1")
    print("=" * 50)

    print(f"Robot mass       : {MASS:.2f} kg")
    print(f"Gravity          : {GRAVITY:.2f} m/s²")
    print(f"Weight           : {MASS * GRAVITY:.3f} N")
    print(f"Hover RPM/rotor  : {rpm:.2f}")

    total_T, thrusts = total_thrust(
        [rpm, rpm, rpm, rpm]
    )

    print(f"Total thrust     : {total_T:.3f} N")
    print(f"Thrust/rotor     : {thrusts[0]:.3f} N")

    # --------------------------------
    # Run hover simulation
    # --------------------------------

    time, history, thrust_history = run_simulation(
        rpm=rpm
    )

    # --------------------------------
    # Final state
    # --------------------------------

    print("\nFinal state:")
    print(f"X  = {history[-1, 0]:.4f} m")
    print(f"Y  = {history[-1, 1]:.4f} m")
    print(f"Z  = {history[-1, 2]:.4f} m")

    print("\nFinal velocity:")
    print(f"Vx = {history[-1, 3]:.4f} m/s")
    print(f"Vy = {history[-1, 4]:.4f} m/s")
    print(f"Vz = {history[-1, 5]:.4f} m/s")

    plot_results(
        time,
        history,
        thrust_history,
        "MorphoAqua - Stage 1 Hover Test"
    )


if __name__ == "__main__":
    main()