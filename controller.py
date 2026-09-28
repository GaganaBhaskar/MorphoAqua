"""
MorphoAqua - Stage 1B
PID altitude controller
"""

class PIDController:

    def __init__(self, kp, ki, kd, output_min, output_max):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, target, measurement, dt):

        error = target - measurement

        # Integral term
        self.integral += error * dt

        # Derivative term
        derivative = (
            (error - self.previous_error) / dt
            if dt > 0
            else 0.0
        )

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # Prevent excessive controller output
        output = max(
            self.output_min,
            min(self.output_max, output)
        )

        self.previous_error = error

        return output