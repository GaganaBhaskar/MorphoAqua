"""
MorphoAqua - Stage 1C

Full 6-DOF Rigid-Body Hover Simulation

State:

Position:
    x, y, z

Velocity:
    vx, vy, vz

Attitude:
    roll, pitch, yaw

Angular velocity:
    p, q, r


"""

import numpy as np
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
    SIMULATION_TIME,

    TARGET_X,
    TARGET_Y,
    TARGET_Z,
    TARGET_YAW,

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

    GROUND_ALTITUDE

)


from controller import (

    PositionController,
    AttitudeController,
    rotation_matrix

)


from motor_model import (

    Motor,
    thrust_from_rpm

)


# ============================================================
# ROTOR THRUST -> RPM
# ============================================================

def thrust_to_rpm(thrust):

    if thrust <= 0:

        return 0.0


    omega = np.sqrt(

        thrust / KF

    )


    rpm = (

        omega
        * 60.0
        / (2.0 * np.pi)

    )


    return np.clip(

        rpm,
        0.0,
        MAX_RPM

    )


# ============================================================
# MOTOR MIXER
# ============================================================

def motor_mixer(

    total_thrust,
    roll_torque,
    pitch_torque,
    yaw_torque

):

    """

    X configuration:

             Front

       M1 -------- M2
          \      /
           \    /
           /    \
          /      \
       M4 -------- M3

    The mixer converts desired total thrust and
    body torques into individual rotor thrusts.
    """

    arm = (

        ARM_LENGTH
        / np.sqrt(2.0)

    )


    mixer = np.array([

        [1.0,  1.0 / arm, -1.0 / arm,  1.0 / KM],

        [1.0, -1.0 / arm, -1.0 / arm, -1.0 / KM],

        [1.0, -1.0 / arm,  1.0 / arm,  1.0 / KM],

        [1.0,  1.0 / arm,  1.0 / arm, -1.0 / KM]

    ])


    desired = np.array([

        total_thrust,
        roll_torque,
        pitch_torque,
        yaw_torque

    ])


    try:

        thrusts = np.linalg.solve(

            mixer,

            desired

        )

    except np.linalg.LinAlgError:

        thrusts = np.ones(4) * (

            total_thrust / 4.0

        )


    thrusts = np.clip(

        thrusts,
        0.0,
        KF * (
            MAX_RPM
            * 2.0
            * np.pi
            / 60.0
        ) ** 2

    )


    return thrusts


# ============================================================
# MISSION REFERENCE
# ============================================================

def reference_position(t):

    """

    0 - 2 seconds:
        smooth takeoff to 2 m

    2 - 9 seconds:
        hover

    9 - 11 seconds:
        smooth landing

    """

    if t < 2.0:

        progress = t / 2.0

        z = TARGET_Z * progress

    elif t < 9.0:

        z = TARGET_Z

    elif t < 11.0:

        progress = (

            (t - 9.0)
            / 2.0

        )

        z = TARGET_Z * (

            1.0 - progress

        )

    else:

        z = 0.0


    return np.array([

        TARGET_X,
        TARGET_Y,
        z

    ])


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation():

    # ========================================================
    # INITIAL STATE
    # ========================================================

    position = np.array([

        0.0,
        0.0,
        0.0

    ])


    velocity = np.array([

        0.0,
        0.0,
        0.0

    ])


    angles = np.array([

        0.0,
        0.0,
        0.0

    ])


    angular_rates = np.array([

        0.0,
        0.0,
        0.0

    ])


    # ========================================================
    # MOTORS
    # ========================================================

    motors = [

        Motor(),
        Motor(),
        Motor(),
        Motor()

    ]


    # ========================================================
    # CONTROLLERS
    # ========================================================

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
        max_pitch=MAX_PITCH

    )


    attitude_controller = AttitudeController(

        kp_roll=ATTITUDE_KP_ROLL,
        kp_pitch=ATTITUDE_KP_PITCH,
        kp_yaw=ATTITUDE_KP_YAW,

        kd_roll=ATTITUDE_KD_ROLL,
        kd_pitch=ATTITUDE_KD_PITCH,
        kd_yaw=ATTITUDE_KD_YAW

    )


    # ========================================================
    # STORAGE
    # ========================================================

    steps = int(

        SIMULATION_TIME
        / DT

    )


    time = (

        np.arange(steps)
        * DT

    )


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


    target_history = np.zeros(

        (steps, 3)

    )


    thrust_history = np.zeros(

        steps

    )


    torque_history = np.zeros(

        (steps, 3)

    )


    rpm_history = np.zeros(

        (steps, 4)

    )


    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for i, t in enumerate(time):


        # ====================================================
        # TARGET
        # ====================================================

        target_position = (

            reference_position(t)

        )


        target_history[i, :] = (

            target_position

        )


        # ====================================================
        # POSITION CONTROLLER
        # ====================================================

        (

            desired_force,

            desired_roll,

            desired_pitch,

            desired_yaw

        ) = position_controller.update(

            target_position,

            position,

            velocity,

            TARGET_YAW

        )


        desired_angles = np.array([

            desired_roll,
            desired_pitch,
            desired_yaw

        ])


        # ====================================================
        # ATTITUDE CONTROLLER
        # ====================================================

        torque = attitude_controller.update(

            desired_angles,

            angles,

            angular_rates

        )


        # ====================================================
        # ATTITUDE-DEPENDENT THRUST
        # ====================================================

        # The controller requests a force magnitude.
        # Convert desired world-frame force to required
        # body thrust along the current body Z axis.

        R = rotation_matrix(

            angles[0],
            angles[1],
            angles[2]

        )


        body_z = R[:, 2]


        thrust = (

            desired_force
            * np.dot(
                body_z,
                np.array([
                    0.0,
                    0.0,
                    1.0
                ])
            )

        )


        # Prevent negative thrust.

        thrust = max(

            0.0,

            thrust

        )


        # ====================================================
        # MOTOR MIXING
        # ====================================================

        motor_thrusts = motor_mixer(

            thrust,

            torque[0],

            torque[1],

            torque[2]

        )


        commanded_rpms = np.array([

            thrust_to_rpm(
                motor_thrust
            )

            for motor_thrust
            in motor_thrusts

        ])


        # ====================================================
        # MOTOR DYNAMICS
        # ====================================================

        actual_rpms = np.zeros(4)


        for motor_index in range(4):

            actual_rpms[
                motor_index
            ] = motors[
                motor_index
            ].update(

                commanded_rpms[
                    motor_index
                ],

                DT

            )


        # ====================================================
        # ACTUAL MOTOR THRUST
        # ========================================================

        actual_motor_thrusts = np.array([

            thrust_from_rpm(rpm)

            for rpm in actual_rpms

        ])


        actual_total_thrust = (

            np.sum(
                actual_motor_thrusts
            )

        )


        # ====================================================
        # ACTUAL MOTOR TORQUES
        # ====================================================

        actual_torques = np.array([

            arm := (

                ARM_LENGTH
                / np.sqrt(2.0)

            ),

            0.0,
            0.0

        ])


        # Calculate actual torques explicitly.

        actual_roll_torque = (

            arm
            * (

                actual_motor_thrusts[0]
                + actual_motor_thrusts[3]

                - actual_motor_thrusts[1]
                - actual_motor_thrusts[2]

            )

        )


        actual_pitch_torque = (

            arm
            * (

                -actual_motor_thrusts[0]
                -actual_motor_thrusts[1]

                + actual_motor_thrusts[2]
                + actual_motor_thrusts[3]

            )

        )


        actual_yaw_torque = (

            KM
            * (

                actual_motor_thrusts[0]
                - actual_motor_thrusts[1]
                + actual_motor_thrusts[2]
                - actual_motor_thrusts[3]

            )

        )


        actual_torque = np.array([

            actual_roll_torque,
            actual_pitch_torque,
            actual_yaw_torque

        ])


        # ====================================================
        # TRANSLATIONAL DYNAMICS
        # ====================================================

        thrust_body = np.array([

            0.0,
            0.0,
            actual_total_thrust

        ])


        thrust_world = (

            R
            @ thrust_body

        )


        gravity_force = np.array([

            0.0,
            0.0,
            -MASS * GRAVITY

        ])


        total_force = (

            thrust_world
            + gravity_force

        )


        acceleration = (

            total_force
            / MASS

        )


        # ====================================================
        # TRANSLATIONAL INTEGRATION
        # ====================================================

        velocity += (

            acceleration
            * DT

        )


        position += (

            velocity
            * DT

        )


        # ====================================================
        # GROUND CONSTRAINT
        # ====================================================

        if position[2] <= GROUND_ALTITUDE:

            position[2] = GROUND_ALTITUDE

            if velocity[2] < 0.0:

                velocity[2] = 0.0


        # ====================================================
        # ROTATIONAL DYNAMICS
        # ====================================================

        angular_momentum = (

            INERTIA
            @ angular_rates

        )


        angular_acceleration = (

            np.linalg.solve(

                INERTIA,

                actual_torque
                - np.cross(
                    angular_rates,
                    angular_momentum
                )

            )

        )


        # ====================================================
        # ANGULAR INTEGRATION
        # ====================================================

        angular_rates += (

            angular_acceleration
            * DT

        )


        # ====================================================
        # EULER ANGLE KINEMATICS
        # ====================================================

        phi = angles[0]

        theta = angles[1]


        cos_theta = np.cos(theta)


        if abs(cos_theta) < 1e-5:

            cos_theta = 1e-5


        tan_theta = (

            np.sin(theta)
            / cos_theta

        )


        euler_rate_matrix = np.array([

            [
                1.0,
                np.sin(phi) * tan_theta,
                np.cos(phi) * tan_theta
            ],

            [
                0.0,
                np.cos(phi),
                -np.sin(phi)
            ],

            [
                0.0,
                np.sin(phi) / cos_theta,
                np.cos(phi) / cos_theta
            ]

        ])


        angle_rates = (

            euler_rate_matrix
            @ angular_rates

        )


        angles += (

            angle_rates
            * DT

        )


        # Keep yaw within [-pi, pi]

        angles[2] = np.arctan2(

            np.sin(angles[2]),
            np.cos(angles[2])

        )


        # ====================================================
        # STORE
        # ====================================================

        position_history[i, :] = position

        velocity_history[i, :] = velocity

        angle_history[i, :] = angles

        angular_rate_history[i, :] = angular_rates

        thrust_history[i] = actual_total_thrust

        torque_history[i, :] = actual_torque

        rpm_history[i, :] = actual_rpms


    return (

        time,

        position_history,

        velocity_history,

        angle_history,

        angular_rate_history,

        target_history,

        thrust_history,

        torque_history,

        rpm_history

    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(

    time,

    position,

    velocity,

    angles,

    target

):

    # ========================================================
    # POSITION ERRORS
    # ========================================================

    position_error = (

        target
        - position

    )


    final_position_error = (

        np.linalg.norm(
            position_error[-1]
        )

    )


    maximum_horizontal_position_error = np.max(

        np.linalg.norm(

            position[:, :2]
            - target[:, :2],

            axis=1

        )

    )


    maximum_altitude = np.max(

        position[:, 2]

    )


    altitude_error = np.abs(

        position[:, 2]
        - TARGET_Z

    )


    hover_indices = np.where(

        (time >= 3.0)
        &
        (time <= 9.0)

    )[0]


    if len(hover_indices) > 0:

        mean_hover_altitude = np.mean(

            position[
                hover_indices,
                2
            ]

        )

        mean_hover_altitude_error = np.mean(

            np.abs(

                position[
                    hover_indices,
                    2
                ]
                - TARGET_Z

            )

        )

    else:

        mean_hover_altitude = np.nan

        mean_hover_altitude_error = np.nan


    # ========================================================
    # ATTITUDE
    # ========================================================

    roll_deg = np.rad2deg(

        angles[:, 0]

    )


    pitch_deg = np.rad2deg(

        angles[:, 1]

    )


    yaw_deg = np.rad2deg(

        angles[:, 2]

    )


    maximum_roll = np.max(

        np.abs(roll_deg)

    )


    maximum_pitch = np.max(

        np.abs(pitch_deg)

    )


    maximum_yaw = np.max(

        np.abs(yaw_deg)

    )


    # ========================================================
    # VELOCITY
    # ========================================================

    speed = np.linalg.norm(

        velocity,

        axis=1

    )


    maximum_speed = np.max(

        speed

    )


    # ========================================================
    # FINAL STATE
    # ========================================================

    final_position = position[-1]

    final_velocity = velocity[-1]

    final_angles = angles[-1]


    return {

        "Maximum altitude (m)":
            maximum_altitude,

        "Mean hover altitude (m)":
            mean_hover_altitude,

        "Mean hover altitude error (m)":
            mean_hover_altitude_error,

        "Maximum horizontal position error (m)":
            maximum_horizontal_position_error,

        "Final position error (m)":
            final_position_error,

        "Maximum speed (m/s)":
            maximum_speed,

        "Maximum roll (deg)":
            maximum_roll,

        "Maximum pitch (deg)":
            maximum_pitch,

        "Maximum yaw (deg)":
            maximum_yaw,

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
            np.rad2deg(final_angles[0]),

        "Final pitch (deg)":
            np.rad2deg(final_angles[1]),

        "Final yaw (deg)":
            np.rad2deg(final_angles[2])

    }


# ============================================================
# PLOT
# ============================================================

def plot_results(

    time,

    position,

    velocity,

    angles,

    target,

    thrust,

    torque,

    rpm

):

    fig, axes = plt.subplots(

        6,
        1,

        figsize=(12, 16),

        sharex=True

    )


    # ========================================================
    # POSITION
    # ========================================================

    axes[0].plot(

        time,
        position[:, 0],
        label="X"

    )


    axes[0].plot(

        time,
        position[:, 1],
        label="Y"

    )


    axes[0].plot(

        time,
        position[:, 2],
        label="Z"

    )


    axes[0].plot(

        time,
        target[:, 2],
        "--",
        label="Target Z"

    )


    axes[0].set_ylabel(

        "Position (m)"

    )


    axes[0].set_title(

        "MorphoAqua - Stage 1C "
        "6-DOF Hover Simulation"

    )


    axes[0].legend()

    axes[0].grid(True)


    # ========================================================
    # VELOCITY
    # ========================================================

    axes[1].plot(

        time,
        velocity[:, 0],
        label="Vx"

    )


    axes[1].plot(

        time,
        velocity[:, 1],
        label="Vy"

    )


    axes[1].plot(

        time,
        velocity[:, 2],
        label="Vz"

    )


    axes[1].set_ylabel(

        "Velocity (m/s)"

    )


    axes[1].legend()

    axes[1].grid(True)


    # ========================================================
    # ATTITUDE
    # ========================================================

    axes[2].plot(

        time,
        np.rad2deg(
            angles[:, 0]
        ),
        label="Roll"

    )


    axes[2].plot(

        time,
        np.rad2deg(
            angles[:, 1]
        ),
        label="Pitch"

    )


    axes[2].plot(

        time,
        np.rad2deg(
            angles[:, 2]
        ),
        label="Yaw"

    )


    axes[2].set_ylabel(

        "Angle (deg)"

    )


    axes[2].legend()

    axes[2].grid(True)


    # ========================================================
    # THRUST
    # ========================================================

    axes[3].plot(

        time,
        thrust,

        label="Total thrust"

    )


    axes[3].axhline(

        MASS * GRAVITY,

        linestyle="--",

        label="Hover thrust"

    )


    axes[3].set_ylabel(

        "Thrust (N)"

    )


    axes[3].legend()

    axes[3].grid(True)


    # ========================================================
    # TORQUES
    # ========================================================

    axes[4].plot(

        time,
        torque[:, 0],
        label="Roll torque"

    )


    axes[4].plot(

        time,
        torque[:, 1],
        label="Pitch torque"

    )


    axes[4].plot(

        time,
        torque[:, 2],
        label="Yaw torque"

    )


    axes[4].set_ylabel(

        "Torque (N m)"

    )


    axes[4].legend()

    axes[4].grid(True)


    # ========================================================
    # MOTOR RPM
    # ========================================================

    for i in range(4):

        axes[5].plot(

            time,
            rpm[:, i],

            label=f"Motor {i + 1}"

        )


    axes[5].set_xlabel(

        "Time (s)"

    )


    axes[5].set_ylabel(

        "RPM"

    )


    axes[5].legend()

    axes[5].grid(True)


    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(

        "MORPHOAQUA - STAGE 1C"

    )

    print(

        "FULL 6-DOF RIGID-BODY "
        "HOVER SIMULATION"

    )

    print("=" * 70)


    results = run_simulation()


    (

        time,

        position,

        velocity,

        angles,

        angular_rates,

        target,

        thrust,

        torque,

        rpm

    ) = results


    metrics = calculate_metrics(

        time,

        position,

        velocity,

        angles,

        target

    )


    print()

    print(

        "6-DOF PERFORMANCE METRICS"

    )

    print(

        "-" * 70

    )


    for name, value in metrics.items():

        if np.isnan(value):

            print(

                f"{name:<45}: NOT AVAILABLE"

            )

        else:

            print(

                f"{name:<45}: {value:.6f}"

            )


    print(

        "-" * 70

    )


    print()

    print(

        "Stage 1C simulation completed."

    )


    print()

    print(

        "Controller configuration:"

    )


    print(

        "Position controller: PD"

    )


    print(

        "Attitude controller: PD"

    )


    print(

        "Degrees of freedom: 6"

    )


    print()

    plot_results(

        time,

        position,

        velocity,

        angles,

        target,

        thrust,

        torque,

        rpm

    )


if __name__ == "__main__":

    main()