# MorphoAqua: Physics-Based Simulation and Control Framework
## Detailed Project Report with Formulae and Sources

---

## Executive Summary

**MorphoAqua** is a Python-based physics simulation and control framework for a morphing aerial-aquatic robot. The project implements a comprehensive 6-DOF (degree of freedom) rigid-body dynamics model, a hierarchical multirotor control architecture, and a cross-medium simulation environment spanning air, the air-water interface, and submerged operation. The framework explicitly includes rotor allocation, actuator authority limiting, medium-dependent drag, buoyancy, and variable inertia due to morphology changes.

The report below documents the underlying equations and references, while also distinguishing between literature-grounded dynamics and the heuristic tuning used in the later adaptive and predictive controller stages.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Stage 1: Core Dynamics Models](#stage-1-core-dynamics-models)
3. [Stage 2A: Control Architecture](#stage-2a-control-architecture)
4. [Stage 3A: Rotor Mixer and Actuator Allocation](#stage-3a-rotor-mixer-and-actuator-allocation)
5. [Stage 3B: Torque Authority, Force Feasibility, and Command Limiting](#stage-3b-torque-authority-force-feasibility-and-command-limiting)
6. [Implementation Details](#implementation-details)
7. [Stage 4A: Cross-Medium Hydrodynamic Transition](#stage-4a-cross-medium-hydrodynamic-transition)
8. [Stage 4B: Water-Phase Control with Buoyancy and Drag Compensation](#stage-4b-water-phase-control-with-buoyancy-and-drag-compensation)
9. [Stage 5: Morphing and Time-Varying Inertia Control](#stage-5-morphing-and-time-varying-inertia-control)
10. [Stage 5B: Adaptive Morphology-Aware Control](#stage-5b-adaptive-morphology-aware-control)
11. [Stage 5C: Robust Cross-Medium Control](#stage-5c-robust-cross-medium-control)
12. [Formula Sources and References](#formula-sources-and-references)

---

## Project Architecture

### Directory Structure

```
MorphoAqua/
├── robot_parameters.py      # Configuration: mass, geometry, control gains
├── dynamics.py              # Stage 1: Rigid-body translational & rotational dynamics
├── motor_model.py           # Stage 1B: Motor response dynamics (first-order)
├── rotor_model.py           # Stage 1: Rotor thrust model
├── controller.py            # Stage 2A: Hierarchical position & attitude control
├── simulation.py            # Main simulation loop with visualization
└── results/                 # Output directory for simulation results
```

### Control Hierarchy

```
Mission/Trajectory Planning
         ↓
Position Controller (3-DOF)
  - Computes desired acceleration
  - Outputs: total thrust, desired roll, desired pitch, desired yaw
         ↓
Attitude Controller (3-DOF)
  - Computes torque commands
  - Outputs: motor RPM commands
         ↓
Motor & Rotor Dynamics
  - First-order motor response
  - Thrust generation
```

---

## Stage 1: Core Dynamics Models

### 1.1 Coordinate Frames and Rotation Matrices

#### 1.1.1 ZYX Euler Convention (Aerospace Standard)

The system uses the **ZYX Euler angle convention** (also called yaw-pitch-roll or 3-2-1 sequence), standard in aerospace applications.

**Rotation Sequence:**
- **Yaw (ψ)**: Rotation about Z-axis (heading)
- **Pitch (θ)**: Rotation about the new Y-axis (lateral tilt)
- **Roll (φ)**: Rotation about the new X-axis (forward tilt)

**Combined Body-to-World Rotation Matrix:**

$$R_{ZYX} = R_Z(ψ) \cdot R_Y(θ) \cdot R_X(φ)$$

**Expanded Form:**

$$R_{ZYX} = \begin{bmatrix}
c_ψ c_θ & c_ψ s_θ s_φ - s_ψ c_φ & c_ψ s_θ c_φ + s_ψ s_φ \\
s_ψ c_θ & s_ψ s_θ s_φ + c_ψ c_φ & s_ψ s_θ c_φ - c_ψ s_φ \\
-s_θ & c_θ s_φ & c_θ c_φ
\end{bmatrix}$$

Where: $c_φ = \cos(φ)$, $s_φ = \sin(φ)$, and similarly for θ and ψ.

**Source:** 
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). *Robotics: Modelling, Planning, and Control*. Springer.
- Bouabdallah, S., Murrieri, P., & Siegwart, R. (2004). Design and control of an indoor micro quadrotor. *Proceedings of the 2004 IEEE ICRA*.

**Implementation:** `controller.py` and `dynamics.py`.

---

### 1.2 Translational Dynamics

#### 1.2.1 Equation of Motion

The world-frame translational acceleration is computed from the dominant force components:

$$\mathbf{a}_{world} = \frac{1}{m} \left( \mathbf{F}_{thrust} + \mathbf{F}_{gravity} + \mathbf{F}_{drag} \right)$$

Where:
- $m$ = robot mass (kg)
- $\mathbf{F}_{thrust}$ = thrust force in world frame
- $\mathbf{F}_{gravity}$ = gravitational force
- $\mathbf{F}_{drag}$ = aerodynamic drag

**Source:** 
- Craig, J. J. (1989). *Introduction to Robotics: Mechanics and Control*. Addison-Wesley.
- Standard rigid-body mechanics formulation.

**Implementation:** `dynamics.py`.

---

#### 1.2.2 Thrust Transformation

Thrust acts along the body +Z axis in body-fixed frame and must be transformed to world frame:

$$\mathbf{F}_{thrust,world} = R_{ZYX} \cdot \mathbf{F}_{thrust,body}$$

Where:
$$\mathbf{F}_{thrust,body} = \begin{bmatrix} 0 \\ 0 \\ T_{total} \end{bmatrix}$$

$T_{total}$ = total thrust from all four rotors (Newtons).

**Source:** Standard coordinate transformation for multirotor vehicles.

**Implementation:** `dynamics.py`.

---

#### 1.2.3 Gravitational Force

$$\mathbf{F}_{gravity} = \begin{bmatrix} 0 \\ 0 \\ -m \cdot g \end{bmatrix}$$

Where $g = 9.81$ m/s² (standard Earth gravity).

**Source:** Classical mechanics.

**Implementation:** `dynamics.py`.

---

#### 1.2.4 Aerodynamic Drag Force (Quadratic Drag)

For speeds typical of drone flight, drag follows a **quadratic relationship** with velocity:

$$\mathbf{F}_{drag} = -\frac{1}{2} ρ C_D A \|\mathbf{v}\| \mathbf{v}$$

Where:
- $ρ$ = air density (kg/m³, typically 1.225 at sea level)
- $C_D$ = drag coefficient (dimensionless)
- $A$ = reference cross-sectional area (m²)
- $\|\mathbf{v}\|$ = speed (magnitude of velocity)
- $\mathbf{v}$ = velocity vector (m/s)

The structure $\|\mathbf{v}\|\mathbf{v}$ ensures the drag force opposes motion.

**Typical Parameter Values (MorphoAqua):**
- Air density: ρ = 1.225 kg/m³
- Drag coefficient: $C_D$ = 0.8
- Reference area: $A$ = 0.05 m²

**Source:**
- Anderson, J. D. (2000). *Introduction to Flight* (4th ed.). McGraw-Hill.
- Hoerner, S. F. (1965). *Fluid-Dynamic Drag*. Hoerner Fluid Dynamics.

**Note:** For a body-drag model such as this one, Anderson and Hoerner are more directly relevant than the wing-focused biology paper by Lentink & Dickinson.

**Implementation:** `dynamics.py`.

---

### 1.3 Rotational Dynamics

#### 1.3.1 Rigid-Body Euler Equations (Body-Fixed Frame)

The fundamental equation for rigid-body rotation about the body-fixed principal axes is:

$$I \dot{\mathbf{ω}} = τ - \mathbf{ω} \times (I \mathbf{ω})$$

Expanded in component form:

$$I_x \dot{p} = τ_x - (I_z - I_y) q r$$
$$I_y \dot{q} = τ_y - (I_x - I_z) r p$$
$$I_z \dot{r} = τ_z - (I_y - I_x) p q$$

Where:
- $I_x, I_y, I_z$ = principal moments of inertia (kg·m²)
- $p, q, r$ = angular velocity components about body X, Y, Z axes (rad/s)
- $\dot{p}, \dot{q}, \dot{r}$ = angular accelerations (rad/s²)
- $τ_x, τ_y, τ_z$ = applied torques about body axes (N·m)
- The cross-product terms represent **gyroscopic coupling** effects

**Gyroscopic Term Interpretation:**

The term $\mathbf{ω} \times (I \mathbf{ω})$ is the gyroscopic torque. When a spinning object experiences a torque perpendicular to its spin axis, the gyroscopic effect causes precession.

**Source:**
- Goldstein, H. (1980). *Classical Mechanics* (2nd ed.). Addison-Wesley.
- Murray, R. M., Sastry, S. S., & Zexiang, L. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
- Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*. Princeton University Press.

**Implementation:** `dynamics.py`.

---

#### 1.3.2 Moment of Inertia Tensor

For a rigid body with principal moments of inertia:

$$I = \begin{bmatrix} I_x & 0 & 0 \\ 0 & I_y & 0 \\ 0 & 0 & I_z \end{bmatrix}$$

**MorphoAqua Parameters:**
- $I_x = 0.030$ kg·m²
- $I_y = 0.030$ kg·m²
- $I_z = 0.050$ kg·m²

These represent the rotational inertia about each body-fixed axis. For a symmetric rectangular body with dimensions length × width × height and uniform mass distribution:

$$I_x ≈ \frac{m}{12}(w^2 + h^2)$$
$$I_y ≈ \frac{m}{12}(l^2 + h^2)$$
$$I_z ≈ \frac{m}{12}(l^2 + w^2)$$

**Source:** 
- Classical mechanics and rigid-body inertia calculations.

**Implementation:** `robot_parameters.py`.

---

### 1.4 Rotor and Motor Models

#### 1.4.1 Rotor Thrust Generation

The thrust produced by a rotor depends quadratically on its angular velocity:

$$T_i = K_F \cdot ω_i^2$$

Where:
- $T_i$ = thrust from rotor $i$ (Newtons)
- $K_F$ = thrust coefficient (N·s²/rad²), typically determined experimentally
- $ω_i$ = angular velocity of rotor $i$ (rad/s)

**Physical Basis:**

Rotor thrust arises from aerodynamic lift on the rotating blades. The lift force is proportional to dynamic pressure $\frac{1}{2}ρ v^2$ and blade area. Since $v = ω \cdot r$, thrust is expected to scale with $ω^2$.

**Hover Condition:**

For a quadrotor to hover, total thrust must approximately equal weight:

$$T_{total} = \sum_{i=1}^{4} T_i = m \cdot g$$

For equal rotor speeds (symmetric hover):

$$T_{per\_rotor} = \frac{m \cdot g}{4} = K_F \cdot ω_{hover}^2$$

Solving for hover angular velocity:

$$ω_{hover} = \sqrt{\frac{m \cdot g}{4 \cdot K_F}}$$

**MorphoAqua Parameters:**
- $K_F = 1.0 \times 10^{-5}$ N·s²/rad² (assumed model parameter)
- Mass = 1.5 kg
- Hover RPM calculation using this coefficient: about 5,800 RPM per rotor

This differs from the earlier informal ~3700 RPM estimate; using the repository's parameter values, $ω_{hover}$ is approximately 606 rad/s, which corresponds to roughly 5,790 RPM. This is therefore a model-parameter and calibration matter, not a universal rotor constant.

**Source:**
- Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors. *IEEE ICRA*, pp. 2520-2525.
- Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*.

**Implementation:** `rotor_model.py` and `simulation.py`.

---

#### 1.4.2 Motor Dynamics (First-Order Response)

Real motors have finite response times and cannot instantaneously reach commanded speeds. This is modeled as a **first-order system**:

$$τ_m \frac{dω}{dt} + ω = ω_{cmd}$$

Rewriting for discrete-time simulation:

$$ω_{rate} = \frac{ω_{cmd} - ω}{τ_m}$$
$$ω_{new} = ω + ω_{rate} \cdot Δt$$

Where:
- $τ_m$ = motor time constant (seconds, typically 0.05–0.1 s)
- $ω_{cmd}$ = commanded angular speed
- $ω$ = actual angular speed (state variable)
- $Δt$ = simulation time step

**Physical Interpretation:**

The time constant $τ_m$ represents the combined effect of:
- Motor winding inductance and resistance
- Propeller inertia and aerodynamic damping
- Electronic speed controller (ESC) response

**MorphoAqua Parameter:**
- Motor time constant = 0.08 seconds

**Source:**
- Standard first-order system modeling from control theory; widely used in multirotor dynamics literature.

**Implementation:** `motor_model.py`.

---

#### 1.4.3 RPM to Angular Velocity Conversion

$$ω \text{ (rad/s)} = \text{RPM} \times \frac{2π}{60}$$

$$\text{RPM} = ω \text{ (rad/s)} \times \frac{60}{2π}$$

**Source:** Unit conversion (standard).

**Implementation:** `rotor_model.py` and `motor_model.py`.

---

### 1.5 Motor Reaction Torque (Yaw Dynamics)

In addition to thrust, each rotor produces a reaction torque about the Z-axis (proportional to its angular velocity or rotor speed, depending on the chosen simplification). The report uses the common low-order approximation:

$$τ_{yaw,i} = K_M \cdot ω_i$$

Where:
- $τ_{yaw,i}$ = reaction torque from rotor $i$ (N·m)
- $K_M$ = motor reaction torque coefficient (N·m·s/rad)
- $ω_i$ = rotor angular velocity (rad/s)

**MorphoAqua Parameter:**
- $K_M = 1.5 \times 10^{-7}$ N·m·s/rad

**Important modeling note:** In the code, the same constant $K_M$ also appears in the rotor mixing matrix as a moment-arm-like torque coefficient in the yaw row, so the report treats it as a practical actuator coefficient rather than a unique physically derived rotor-drag constant. In more detailed propeller models, yaw moment is often represented as a function of rotor drag torque, which can be modeled differently from the linear first-order approximation above. The project uses $K_M$ as a simplified allocation coefficient for the simulation.

**Source:**
- Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*.

**Implementation:** `robot_parameters.py` and `simulation.py`.

---

## Stage 2A: Control Architecture

### 2.1 Position Controller (3-DOF)

The **position controller** is a cascaded PD feedback system that:
1. Measures position and velocity errors
2. Computes desired acceleration (feedforward + feedback)
3. Converts desired acceleration to a desired force vector
4. Decomposes that force into thrust and attitude commands

#### 2.1.1 Position and Velocity Errors

$$\mathbf{e}_p = \mathbf{r}_{target} - \mathbf{r}_{current}$$
$$\mathbf{e}_v = \mathbf{v}_{target} - \mathbf{v}_{current}$$

Where:
- $\mathbf{e}_p$ = position error (m)
- $\mathbf{e}_v$ = velocity error (m/s)

---

#### 2.1.2 PD Acceleration Command

$$\mathbf{a}_{cmd} = \mathbf{a}_{ff} + K_p \circ \mathbf{e}_p + K_d \circ \mathbf{e}_v$$

Where:
- $\mathbf{a}_{ff}$ = feedforward acceleration (m/s²)
- $K_p$ = proportional gain vector (1/s²), element-wise
- $K_d$ = derivative gain vector (1/s), element-wise
- $\circ$ = element-wise (Hadamard) multiplication

**MorphoAqua Gains:**
```
K_p = [2.0, 2.0, 6.0]  (x, y, z components)
K_d = [2.5, 2.5, 4.0]  (x, y, z components)
```

**Source:**
- Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.
- PD control is standard in trajectory tracking; gains are tuned for desired responsiveness and stability.

**Implementation:** `controller.py` and `robot_parameters.py`.

---

#### 2.1.3 Acceleration Limits

**Horizontal (XY) Acceleration Limit:**

Limit the horizontal acceleration magnitude to prevent excessive tilting:

$$\text{if } \|\mathbf{a}_{h}\| > a_{h,max}: \quad \mathbf{a}_{h} \leftarrow \mathbf{a}_{h} \cdot \frac{a_{h,max}}{\|\mathbf{a}_{h}\|}$$

Where:
- $\mathbf{a}_{h} = [a_x, a_y]$ = horizontal acceleration components
- $a_{h,max}$ = maximum horizontal acceleration (m/s²), default = 2.5 m/s²

**Vertical (Z) Acceleration Limit:**

Clamp vertical acceleration feedback separately:

$$a_{z,fb} = \text{clip}(a_{z,cmd} - g, -a_{z,max}, a_{z,max})$$
$$a_{z,cmd,final} = g + a_{z,fb}$$

Where:
- $g$ = gravitational acceleration (9.81 m/s²)
- $a_{z,max}$ = maximum vertical acceleration (m/s²), default = 3.0 m/s²

**Rationale:** Vertical acceleration is referenced to gravity (null-thrust equilibrium). Limits prevent excessive vertical dynamics.

**Source:** Standard practice in multirotor control to maintain feasible attitudes.

**Implementation:** `controller.py`.

---

#### 2.1.4 Desired Force Vector

Once the acceleration command is finalized:

$$\mathbf{F}_{desired} = m \cdot \mathbf{a}_{cmd}$$

Where $m$ = robot mass (1.5 kg).

**Implementation:** `controller.py`.

---

#### 2.1.5 Thrust and Attitude Extraction

**Total Thrust Magnitude:**

$$T_{total} = \|\mathbf{F}_{desired}\|$$

The total thrust is constrained to avoid singularities and unrealistic near-zero-thrust conditions:

$$T_{total} = \max(T_{total}, m \cdot g)$$

**Desired Body Z-Axis (Unit Vector):**

$$\hat{z}_{body,desired} = \frac{\mathbf{F}_{desired}}{T_{total}}$$

This unit vector points in the direction the body must tilt to generate the desired force.

**Desired Roll and Pitch (from Desired Body Orientation):**

Using the ZYX Euler convention, the desired body Z-axis components in world frame are:

$$b_x = \cos(ψ) \sin(θ) \cos(φ) + \sin(ψ) \sin(φ)$$
$$b_y = \sin(ψ) \sin(θ) \cos(φ) - \cos(ψ) \sin(φ)$$
$$b_z = \cos(θ) \cos(φ)$$

Given $b_x, b_y, b_z$ and a desired yaw $ψ$, we can extract roll and pitch:

$$φ_{desired} = \arcsin(b_x \sin(ψ) - b_y \cos(ψ))$$
$$θ_{desired} = \arctan2(b_x \cos(ψ) + b_y \sin(ψ), b_z)$$

**Attitude Saturation:**

$$φ_{desired} \leftarrow \text{clip}(φ_{desired}, -φ_{max}, φ_{max})$$
$$θ_{desired} \leftarrow \text{clip}(θ_{desired}, -θ_{max}, θ_{max})$$

**MorphoAqua Limits:**
- $φ_{max} = 20°$ (0.349 rad)
- $θ_{max} = 20°$ (0.349 rad)

**Source:**
- Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors.
- Beard, R. W., & McLain, T. W. (2012).

**Implementation:** `controller.py`.

---

### 2.2 Attitude Controller (3-DOF)

The **attitude controller** uses PD feedback on Euler angle errors to generate torque commands.

#### 2.2.1 Angle Errors

Each angle error is wrapped to the interval $[-π, π]$ to handle discontinuities at $±π$:

$$e_φ = \text{atan2}(\sin(φ_{desired} - φ_{current}), \cos(φ_{desired} - φ_{current}))$$

Similarly for pitch and yaw. This is equivalent to:

$$e = \arctan2(\sin(e), \cos(e))$$

**Rationale:** Euler angles are periodic (mod $2π$). A raw difference can be ambiguous (e.g., -350° vs +10°). Wrapping ensures the shortest angular distance.

**Source:** Standard practice in attitude control.

**Implementation:** `controller.py`.

---

#### 2.2.2 PD Torque Command

$$\mathbf{τ} = K_{p,att} \circ \mathbf{e}_{angle} - K_{d,att} \circ \mathbf{ω}_{body}$$

Where:
- $K_{p,att}$ = proportional attitude gains (N·m/rad)
- $K_{d,att}$ = derivative attitude gains (N·m·s/rad)
- $\mathbf{ω}_{body}$ = angular velocity in body frame (rad/s)
- $\mathbf{e}_{angle}$ = wrapped angle errors (rad)

**MorphoAqua Gains:**
```
K_p,att = [4.0, 4.0, 2.0]  (roll, pitch, yaw)
K_d,att = [0.8, 0.8, 0.5]  (roll, pitch, yaw)
```

**Physical Interpretation:**

- **Proportional Term:** Directly opposes angle errors; acts like a rotational "spring."
- **Derivative Term:** Opposes angular velocity; acts like rotational "damping."

Together, they produce a stable second-order response.

**Source:**
- Ogata, K. (2010). *Modern Control Engineering*.
- Standard PD attitude control in robotics and aerospace.

**Implementation:** `controller.py`.

---

## Stage 3A: Rotor Mixer and Actuator Allocation

### 3A.1 Rotor force-to-moment mapping

The low-level actuation block converts a commanded collective thrust and body moments into four rotor thrusts. In MorphoAqua, each rotor is assumed to generate a vertical force along the body +Z axis and a corresponding yawing reaction effect consistent with the simplified mixer model.

For a square rotor layout with arm length $a$, the geometry yields the standard mixing matrix:

$$
\begin{bmatrix}
T \\
\tau_x \\
\tau_y \\
\tau_z
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 & 1 & 1 \\
a & -a & -a & a \\
-a & -a & a & a \\
K_M & -K_M & K_M & -K_M
\end{bmatrix}
\begin{bmatrix}
F_1 \\
F_2 \\
F_3 \\
F_4
\end{bmatrix}
$$

with:

$$a = \frac{L}{\sqrt{2}}$$

where:
- $F_i$ = thrust from rotor $i$
- $T$ = total collective thrust
- $\tau_x, \tau_y, \tau_z$ = roll, pitch, and yaw moments
- $L$ = rotor arm length
- $K_M$ = rotor reaction-torque coefficient used in the simplified model

This matches the code implementation in `simulation.py`, where:

$$M = \begin{bmatrix}
1 & 1 & 1 & 1 \\
a & -a & -a & a \\
-a & -a & a & a \\
K_M & -K_M & K_M & -K_M
\end{bmatrix}$$

and the mixer is solved by:

$$\mathbf{F}_{rotors} = M^{-1} \begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}$$

This is implemented by the functions:
- `mix_matrix(arm)`
- `calculate_motor_thrusts(total_thrust, torque, arm)`
- `np.linalg.solve(...)`

### 3A.2 Rotor thrust and rotor speed relation

Each rotor thrust is modeled as a quadratic function of rotor angular velocity:

$$T_i = K_F \omega_i^2$$

and the corresponding rotor speed conversion is:

$$\omega = \text{RPM} \cdot \frac{2\pi}{60}$$

The implementation uses:

$$\omega_{max} = \text{MAX\_RPM} \cdot \frac{2\pi}{60}$$

and converts commanded thrust to rotor RPM by:

$$\text{RPM} = \sqrt{\frac{T}{K_F}} \cdot \frac{60}{2\pi}$$

This is implemented in `simulation.py` via:
- `max_motor_thrust()`
- `thrust_to_rpm(thrust)`
- `Motor.update(...)`
- `thrust_from_rpm(rpm)`

### 3A.3 Feasibility and saturation logic

The low-level actuator is feasible only if each rotor thrust remains within the actuator bound:

$$0 \le F_i \le F_{max}$$

where:

$$F_{max} = K_F \omega_{max}^2$$

A command is rejected or scaled if any rotor exceeds this range. The simulation enforces this through:

$$\text{feasible}(F_1,...,F_4) = \big(\forall i,\ 0 \le F_i \le F_{max}\big)$$

This prevents invalid actuation when the commanded wrench is outside the reachable set of the four rotors.

### 3A.4 Source basis for Stage 3A

This stage is grounded in standard quadrotor rotor-mixing and motor modeling literature:
- Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*. Princeton University Press.
- Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors. *IEEE ICRA*.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). *Robotics: Modelling, Planning, and Control*.

**Implementation:** `simulation.py` contains the rotor mixer, allocation, feasibility checks, and motor timing logic used in the main simulation loop.

---

## Stage 3B: Torque Authority, Force Feasibility, and Command Limiting

### 3B.1 Torque authority envelope

Once the rotor thrusts are known, the maximum achievable roll, pitch, and yaw moments are constrained by both the total available thrust and the available differential thrust. The implementation uses a conservative estimate based on headroom around the collective thrust level.

For a rotor set operating near hover, the differential thrust headroom is approximated by:

$$\Delta T = \min\left(\frac{T_{collective}}{4},\ T_{max} - \frac{T_{collective}}{4}\right)$$

Then a conservative torque envelope is used:

$$\tau_{roll,max} = 4 \left(\frac{L}{\sqrt{2}}\right) \Delta T \cdot \eta_{eff} \cdot s_{torque}$$

$$\tau_{pitch,max} = 4 \left(\frac{L}{\sqrt{2}}\right) \Delta T \cdot \eta_{eff} \cdot s_{torque}$$

$$\tau_{yaw,max} = 4 K_M \Delta T \cdot \eta_{eff} \cdot s_{torque}$$

where:
- $\eta_{eff}$ = propulsion efficiency in the current medium
- $s_{torque}$ = safety factor (e.g., 0.94 in the code)
- $T_{collective}$ = total commanded thrust

This is implemented in `simulation.py` by the function:
- `torque_authority_limits(total_aerial, arm, efficiency)`

The code also computes the actuator-aware limiting metric:

$$\text{authority\_scale} = \frac{\tau_{commanded}}{\tau_{available}}$$

and clips or scales the torque command if the requested wrench exceeds the feasible set.

### 3B.2 Force feasibility under acceleration limits

The controller does not command arbitrary force vectors; it projects the desired force into a feasible throttle and attitude envelope. The implementation starts from the desired acceleration:

$$\mathbf{F}_{req} = m\mathbf{a}_{cmd} - \mathbf{F}_{gravity} - \mathbf{F}_{buoyancy} - \mathbf{F}_{drag}$$

Then it enforces:

$$\|\mathbf{a}_h\| \le a_{h,max}$$

and a vertical acceleration bound:

$$a_z \in [-a_{z,max}, a_{z,max}]$$

The feasible force set is bounded by the maximum effective thrust:

$$\|\mathbf{F}_{req}\| \le T_{eff,max} = 4F_{max}\,\eta_{eff}\,s_{force}$$

where $s_{force}$ is the force safety margin from the implementation.

For the horizontal-plane feasibility check, the code uses both a circular limit and a tilt limit:

$$\|\mathbf{F}_{h}\| \le \sqrt{T_{eff,max}^2 - F_z^2}$$

and

$$\|\mathbf{F}_{h}\| \le F_z \tan(\theta_{max})$$

The final command is then scaled to satisfy both inequalities. This logic is implemented in:
- `feasible_force_from_accel(...)`
- `ACTUATOR_FORCE_MARGIN`
- `MAX_ROLL`, `MAX_PITCH`

### 3B.3 Torque projection while preserving collective thrust

The implementation also includes a command-limiting step that maintains the collective thrust while scaling the feedback torque if necessary. The logic is:

$$\mathbf{\tau}_{total} = \mathbf{\tau}_{comp} + \mathbf{\tau}_{fb}$$

and then the code solves for a factor $s \in [0,1]$ such that:

$$\mathbf{\tau}_{proj} = \mathbf{\tau}_{comp} + s\mathbf{\tau}_{fb}$$

while ensuring the four motor thrusts remain feasible.

This is captured by the function:

$$\text{project\_torque\_with\_collective}(T_{collective},\ \tau_{fb},\ \tau_{comp},\ arm,\ \eta)$$

The reason for this step is practical: if the attitude controller requests a torque that exceeds the rotor authority, the controller should not simply saturate all torques independently, because that can disrupt the collective thrust and de-stabilize the flight envelope. The implementation therefore preserves the total thrust while scaling only the feedback contribution when needed.

### 3B.4 Implementation note in MorphoAqua

The simulation performs the full low-level feasibility chain:
1. Compute desired acceleration and desired force
2. Clip acceleration to horizontal and vertical limits
3. Project the force to the feasible thrust envelope
4. Map desired force to desired roll and pitch
5. Compute PD attitude torque
6. Calculate compensation torque from changing inertia
7. Project torque to the feasible motor authority set
8. Allocate the final rotor thrusts
9. Apply first-order motor lag and update the rigid-body dynamics

This chain is implemented in `simulation.py` through:
- `project_torque_with_collective(...)`
- `allocate_motors(...)`
- `feasible_force_from_accel(...)`
- `AdaptivePredictiveController.control(...)`

### 3B.5 Source basis for Stage 3B

This section is grounded in standard multirotor actuator allocation and constrained control formulations:
- Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*.
- Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors. *IEEE ICRA*.
- Mahony, R., Kumar, V., & Corke, P. (2012). Multirotor aerial vehicles: modeling, estimation, and control of quadrotor. *IEEE Robotics & Automation Magazine*.
- Ogata, K. (2010). *Modern Control Engineering*.

**Implementation:** `simulation.py` contains the full actuator-aware command-limiting logic and torque projection used by the later controller stages.

---

## Implementation Details

### 3.1 Simulation Loop Structure

The main `simulation.py` orchestrates:

1. **Initialization:** Set up robot parameters, controllers, initial state.
2. **Control Loop (dt = 0.002 s):**
   - Read current state (position, velocity, attitude, angular rates)
   - Update position controller → thrust + attitude commands
   - Update attitude controller → torque commands
   - Update motor dynamics (first-order lag)
   - Compute rotor thrusts from motor RPM
   - Update rigid-body dynamics (6-DOF ODE)
   - Log state and visualization data
3. **Visualization & Analysis:** Plot trajectories, attitude, thrust, etc.

**Simulation Parameters:**
- Time step: $Δt = 0.002$ s (500 Hz)
- Total duration: 15 seconds (for the earlier Stage 1–5 variants; the Stage 5C simulation extends to 35 seconds in the validated code)
- Integration method: Forward Euler (simple explicit Runge-Kutta available)

---

### 3.2 State Vector

The full simulation state is:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z, φ, θ, ψ, p, q, r]^T$$

Where:
- Position: $(x, y, z)$ in world frame (m)
- Velocity: $(v_x, v_y, v_z)$ in world frame (m/s)
- Euler angles: $(φ, θ, ψ)$ in world frame (rad)
- Angular velocity: $(p, q, r)$ in body frame (rad/s)

**Total dimension: 12 states**

---

### 3.3 Numerical Integration

For the full nonlinear ODE:

$$\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, t)$$

The system integrates using the **forward Euler method** by default:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + f(\mathbf{x}_k, \mathbf{u}_k, t_k) \cdot Δt$$

**Stability Consideration:** Forward Euler is explicit and requires small time steps for stability. For the tested gains and $Δt = 0.002$ s, the simulation is stable in the validated project configuration.

---

## Stage 4A: Cross-Medium Hydrodynamic Transition

Stage 4A extends the air-only dynamics model to the transition regime where the robot interacts with water. In this stage, the vehicle is no longer treated as a purely aerial rigid body; instead, the model includes buoyancy, hydrodynamic drag, immersion, and reduced propulsion effectiveness.

### 4A.1 Immersion and Medium Effects

The immersion state is represented by a smooth indicator function:

$$
\eta(z) = \begin{cases}
0, & z \ge z_{upper} \\
1, & z \le z_{lower} \\
s(u), & z_{lower} < z < z_{upper}
\end{cases}
$$

where:
- $\eta(z)$ = immersion fraction
- $z_{upper}$ and $z_{lower}$ define the upper and lower water-interface boundaries
- $s(u)$ is a smoothstep function used to avoid abrupt force switching

The total force acting on the vehicle is:

$$
\mathbf{F}_{total} = \mathbf{F}_{thrust} + \mathbf{F}_{gravity} + \mathbf{F}_{buoyancy} + \mathbf{F}_{drag}
$$

with buoyancy:

$$
\mathbf{F}_{buoyancy} = \rho_{water} V_{disp} g \,\hat{z}
$$

and hydrodynamic drag:

$$
\mathbf{F}_{drag} = -\frac{1}{2}\rho_{water} C_D A \|\mathbf{v}_{rel}\| \mathbf{v}_{rel}
$$

where:
- $\rho_{water} = 1000 \,\text{kg/m}^3$
- $V_{disp}$ = displaced volume
- $\mathbf{v}_{rel} = \mathbf{v}_{vehicle} - \mathbf{v}_{water}$

### 4A.2 Water-Entry Interface Model

The simulation includes a reduced-order water-entry impact model to capture the dominant transient effect of partial immersion crossing. Rather than solving fully coupled CFD, the model approximates the dominant effects of:
- increased buoyancy
- increased drag
- reduced propulsion effectiveness
- a brief impact impulse during water entry

The implementation uses:
- water density: $\rho_{water} = 1000$ kg/m³
- drag coefficient: $C_D = 0.90$
- reference area: $A = 0.025$ m²
- air propulsion effectiveness: $1.0$
- water propulsion effectiveness: $0.30$
- interface penalty: $0.015$

**Implementation note:** The code implements immersion scaling explicitly in the medium model (see `medium_effects()` in `simulation.py`), rather than assuming a single fixed buoyancy term for all phases.

### 4A.3 Source Basis

This stage is grounded in classical fluid mechanics and aerospace dynamics:
- Anderson, J. D. (2000). *Introduction to Flight*. McGraw-Hill.
- Hoerner, S. F. (1965). *Fluid-Dynamic Drag*. Hoerner Fluid Dynamics.
- Standard rigid-body force decomposition used in multirotor and marine vehicle modeling.

**Implementation:** `simulation.py` contains the immersion model, water-current drag model, and water-entry disturbance logic.

---

## Stage 4B: Water-Phase Control with Buoyancy and Drag Compensation

Stage 4B addresses the control problem after the interface transition, when the vehicle is operating in water and must compensate for the altered force balance.

### 4B.1 Water-Phase Force Budget

The translational dynamics in water are written as:

$$
m \dot{\mathbf{v}} = \eta_{water} \mathbf{F}_{thrust} + \mathbf{F}_{gravity} + \mathbf{F}_{buoyancy} + \mathbf{F}_{drag}
$$

where:
- $\eta_{water}$ is the effective propulsion efficiency in water
- $\mathbf{F}_{buoyancy}$ changes the equilibrium condition from the air hover condition
- $\mathbf{F}_{drag}$ becomes a dominant disturbance for lateral and vertical motion

This means the control law used in air cannot be directly reused without accounting for the shifted operating point.

### 4B.2 Reduced Propulsion Authority

The repo explicitly models reduced underwater thrust effectiveness:

$$
T_{water} = \eta_{prop,water} T_{air}
$$

with:

$$
\eta_{prop,water} \approx 0.30
$$

and additional interface penalties near the air-water boundary. In practical terms, the controller must demand more total force than in air to achieve the same acceleration under water.

### 4B.3 Control Interpretation

The position controller still follows the PD structure:

$$
\mathbf{a}_{cmd} = \mathbf{a}_{ff} + K_p \circ \mathbf{e}_p + K_d \circ \mathbf{e}_v
$$

but the actual plant dynamics now include:

$$
\mathbf{F}_{plant} = m\mathbf{a}_{cmd} - \mathbf{F}_{buoyancy} - \mathbf{F}_{drag}
$$

This changes the feasible set of commanded acceleration and therefore the controller must account for water-resistance and buoyancy compensation.

**Implementation note:** In the code, this is implemented as an immersed-force balance and not as a direct hardware-validated underwater model. It is a simulation-level approximation.

### 4B.4 Source Basis

This is consistent with:
- Ogata, K. (2010). *Modern Control Engineering*.
- Anderson, J. D. (2000). *Introduction to Flight*.
- Standard hydrodynamic force modeling for submerged vehicles and robot platforms.

**Implementation:** The repository includes Stage 4B artifacts:
- `results/Stage_4B_3D_trajectory.png`
- `results/Stage_4B_control_water_response.png`
- `results/Stage_4B_position_tracking.png`

---

## Stage 5: Morphing and Time-Varying Inertia Control

Stage 5 introduces morphology change and time-varying inertial properties. The vehicle is no longer treated as a fixed-geometry rigid body throughout the mission. Instead, the arm geometry changes over time and the inertia matrix changes accordingly.

### 5.1 Morphology Parameterization

The repo models a variable arm ratio:

$$
r(t) = r_{compact} + (r_{extended} - r_{compact})\,m(t)
$$

with:
- $r_{compact} = 0.80$
- $r_{extended} = 1.20$

The effective arm length becomes:

$$
L(t) = L_0\,r(t)
$$

and the inertia scales according to:

$$
I(t) = I_0\,r(t)^2
$$

with the corresponding derivative:

$$
\dot{I}(t) = I_0 \cdot 2r(t)\dot{r}(t)
$$

This is implemented in `simulation.py` through:
- `COMPACT_ARM_RATIO = 0.80`
- `EXTENDED_ARM_RATIO = 1.20`
- `arm = ARM_LENGTH * ratio`
- `inertia_scale = ratio ** 2`

### 5.2 Control Implication

The rigid-body rotational dynamics under changing inertia are:

$$
I(t)\dot{\boldsymbol{\omega}} + \dot{I}(t)\boldsymbol{\omega}
= \boldsymbol{\tau} - \boldsymbol{\omega} \times (I(t)\boldsymbol{\omega})
$$

This means the controller must react not only to attitude error but also to:
- changing rotational inertia
- changing arm geometry
- changing torque allocation capability
- changing plant response under air-water transition and morphology changes

### 5.3 Source Basis

This stage follows well-established literature on:
- rigid-body dynamics with changing inertia
- multirotor flight control
- morphology-aware robot motion planning and adaption

**Implementation:** `simulation.py` includes the morphology profile and time-varying inertia model used for the later adaptive stages.

---

## Stage 5B: Adaptive Morphology-Aware Control

Stage 5B adds adaptive scheduling and predictive augmentation to handle the combined effects of changing inertia, immersion, and actuator authority.

### 5B.1 Gain Scheduling

The controller computes adaptive gain scaling terms based on position error, inertia ratio, and immersion state. A representative scaling is:

$$
\text{error\_factor} = \tanh\left(\frac{\|\mathbf{e}_p\|}{0.30}\right)
$$

Then:

$$
k_{p,scale} = 1.0 + 0.030\,\text{error\_factor} + 0.018\left(\frac{I}{I_0} -1\right) - 0.012\eta
$$

$$
k_{d,scale} = 1.0 + 0.024\max\left(\frac{I}{I_0} -1, 0\right)
$$

and the attitude gain factor is:

$$
k_{att,scale} = 1.0 + 0.018\left(\frac{I}{I_0} -1\right) - 0.008\eta + 0.010\,\text{error\_factor} + 0.010(\alpha - 0.5)
$$

where:
- $\eta$ = immersion fraction
- $\alpha$ = authority hint (normalized actuator feasibility)

The repo explicitly bounds these schedules with:
- `ADAPTIVE_KP_MIN = 0.97`
- `ADAPTIVE_KP_MAX = 1.08`
- `ADAPTIVE_KD_MIN = 1.00`
- `ADAPTIVE_KD_MAX = 1.12`
- `ADAPTIVE_ATT_MIN = 0.96`
- `ADAPTIVE_ATT_MAX = 1.04`

### 5B.2 Predictive Horizon and Feasible Force Envelope

To improve robustness during morphology and medium transitions, the controller uses a short-horizon prediction:

$$
T_h \in [T_{h,min},T_{h,max}]
$$

with:
- `PREDICTIVE_HORIZON_MIN = 0.105 s`
- `PREDICTIVE_HORIZON_MAX = 0.18 s`

The controller also imposes an efficiency-aware horizontal acceleration ceiling:

$$
a_{h,max} = a_{h,max,base}\left(\beta + (1-\beta)\eta_{eff}\right)
$$

where:
- $\eta_{eff}$ = propulsion efficiency
- $\beta$ is a floor factor to avoid unrealistic underwater demand

### 5B.3 Implementation Notes

The Stage 5B controller is designed to remain physically plausible:
- gain schedules are bounded
- predictive horizon is bounded
- desired attitude rates are rate-limited
- torque commands are projected to feasible actuator space
- force commands are clipped based on actuator authority

This is essential because in water and morphing conditions, a nominal aerial controller would otherwise request forces and moments beyond what the motors can safely provide.

### 5B.4 Source Basis and Honest Interpretation

The Stage 5B gain-scheduling expressions are a practical engineering design, not a direct derivation from a single cited literature source. They are tuned to the project-specific medium-transition and morphology-scaling problem, and should therefore be interpreted as a heuristic schedule rather than an exact theorem-derived formula.

**Implementation:** The repository includes Stage 5B output sets and adaptive schedule logic in `simulation.py`.

---

## Stage 5C: Robust Cross-Medium Control

Stage 5C is the robust controller variant for the full cross-medium mission. It retains the validated mission structure while extending the adaptive and predictive architecture with explicit robustness against water-current and uncertainty effects.

### 5C.1 Disturbance Model

The water current is modeled as a plant-side disturbance rather than a direct measurement input:

$$
\mathbf{v}_{rel} = \mathbf{v}_{vehicle} - \mathbf{v}_{water}
$$

Hydrodynamic drag is therefore based on relative water velocity, not only vehicle velocity. The water current is applied only after the vehicle is sufficiently submerged:

$$
\mathbf{v}_{water}(t) = \mathbf{v}_{current}\,r_{ramp}(t)
$$

with:
- `STAGE5C_CURRENT_VECTOR = [0.10, -0.06, 0.0]` m/s
- current ramp starts at immersion roughly 0.80
- current is deliberately hidden from the controller

### 5C.2 Robustness and Uncertainty

The repo includes a bounded uncertainty study using:
- drag uncertainty: $\pm 15\%$
- propulsion uncertainty: $\pm 10\%$
- current uncertainty: $\pm 20\%$

The Monte Carlo robustness analysis uses randomized scaling factors and measures final tracking error, RMS trajectory error, and actuator-limit events.

### 5C.3 Controller Enhancements

The Stage 5C controller adds:
- bounded adaptive position gain scheduling
- bounded adaptive velocity gain scheduling
- bounded adaptive attitude gain scheduling
- short-horizon predictive augmentation
- actuator-aware force feasibility
- desired-attitude rate limiting
- collective-preserving torque projection
- explicit torque-authority limiting metric

These features are implemented in `simulation.py` inside the `AdaptivePredictiveController` class and the torque/force feasibility logic.

### 5C.4 Regression Gate

The repo defines the nominal pass criterion as:

$$
\text{Final position error} < 0.30 \,\text{m}
$$

This is used as the frozen nominal regression threshold. Disturbed-current and uncertainty cases are treated as robustness evaluations rather than separate tracking objectives.

### 5C.5 Source Basis and Honest Interpretation

This stage builds on standard methods from:
- adaptive control
- gain scheduling
- robust disturbance rejection
- actuator-aware constraint handling for underactuated aerial-aquatic robots

However, the specific Stage 5C gain schedule and acceleration-floor logic are tuned engineering choices, not direct theorem-based formulas derived from a single literature source. They are physically plausible and simulation-validated, but should be described as heuristically tuned, not strictly literature-derived.

**Implementation:** The repository contains the Stage 5C artifacts and summaries:
- `results/Stage_5C_3D_adaptive_predictive_trajectory.png`
- `results/Stage_5C_control_medium_response.png`
- `results/Stage_5C_adaptive_predictive_response.png`
- `results/Stage_5C_direction_impact_response.png`
- `results/Stage_5C_morphology_inertia.png`
- `results/Stage_5C_sensor_estimation.png`
- `results/Stage_5C_nominal_vs_current.png`
- `results/Stage_5C_disturbance_response.png`

---

## Formula Sources and References

### Primary References

1. **Beard, R. W., & McLain, T. W. (2012).** *Small Unmanned Aircraft: Theory and Practice*. Princeton University Press.
   - Comprehensive coverage of multirotor dynamics, control, and estimation.
   - Source for Euler equations, thrust models, and PD control.

2. **Mellinger, D., & Kumar, V. (2011).** Minimum snap trajectory generation and control for quadrotors. *IEEE ICRA*, pp. 2520-2525.
   - Foundational work on quadrotor trajectory tracking and attitude decomposition.
   - Source for force-to-attitude mapping.

3. **Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009).** *Robotics: Modelling, Planning, and Control*. Springer-Verlag.
   - Rigorous mathematical treatment of rotation matrices and rigid-body dynamics.

4. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice Hall.
   - Standard textbook on PD and PID control theory.

5. **Goldstein, H. (1980).** *Classical Mechanics* (2nd ed.). Addison-Wesley.
   - Derivation of Euler's equations and gyroscopic effects.

6. **Anderson, J. D. (2000).** *Introduction to Flight* (4th ed.). McGraw-Hill.
   - Quadratic drag and aerodynamic coefficients.

7. **Mahony, R., Kumar, V., & Corke, P. (2012).** Multirotor aerial vehicles: Modeling, estimation, and control of quadrotor. *IEEE Robotics & Automation Magazine*, 19(3), 20-32.
   - Directly relevant to actuator allocation, rotor mixing, and quadrotor attitude dynamics.

8. **Hoffmann, G. M., Huang, H., Waslander, S. L., & Tomlin, C. J. (2007).** Quadrotor helicopter flight dynamics and control: Theory and experiment. *AIAA Guidance, Navigation, and Control Conference*.
   - Useful for rotor-thrust/moment mapping and low-level feasibility constraints.

### Supplementary References

- **Craig, J. J. (1989).** *Introduction to Robotics: Mechanics and Control*. Addison-Wesley.
- **Murray, R. M., Sastry, S. S., & Zexiang, L. (1994).** *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
- **Hoerner, S. F. (1965).** *Fluid-Dynamic Drag*. Hoerner Fluid Dynamics.

---

## Conclusion

MorphoAqua implements a complete, physically grounded model of multirotor dynamics and hierarchical control. The core rigid-body dynamics, rotor-thrust relations, motor lag, Euler angle kinematics, and standard PD multirotor control structure are well supported by established aerospace and robotics literature.

At the same time, the later stages—especially the adaptive gain scheduling and medium-transition heuristics—should be interpreted as project-specific engineering design choices rather than exact formulae from a single literature source. This is not a weakness of the project; it is a realistic and transparent description of how simulation models are often constructed. The report is therefore most accurate when it distinguishes:

- literature-grounded physical equations,
- implementation-specific modeling assumptions, and
- tuned heuristic controller schedules.

---

## Appendix: Key Dimensionless Parameters

| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Mass ($m$) | 1.5 | kg | Robot total mass |
| Gravity ($g$) | 9.81 | m/s² | Standard Earth |
| Air Density ($ρ$) | 1.225 | kg/m³ | Sea level, 15°C |
| Drag Coefficient ($C_D$) | 0.8 | - | Assumed for body |
| Reference Area ($A$) | 0.05 | m² | Cross-sectional area |
| Thrust Coeff. ($K_F$) | 1.0e-5 | N·s²/rad² | Assumed; requires calibration |
| Motor Reaction Coeff. ($K_M$) | 1.5e-7 | N·m·s/rad | Simplified reaction-torque coefficient |
| Motor Time Constant ($τ_m$) | 0.08 | s | First-order response |
| Simulation Step ($Δt$) | 0.002 | s | 500 Hz loop rate |

---

**Report Generated:** October 2026  
**Repository:** https://github.com/GaganaBhaskar/MorphoAqua  
**Project:** MorphoAqua - Morphing Aerial-Aquatic Robot Simulation Framework
