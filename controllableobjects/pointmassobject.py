import copy

import numpy as np

from .controlableobject import ControllableObject


class PointMassObject(ControllableObject):
    """
    a 1 dimensional point mass object. It's position is expressed in two dimensions, but it can only move over a predefined track. For that reason, it's
    velocity and acceleration are only one dimensional.
    Negative velocities are neglected.
    A resistance depending on the velocity squared is taken into account this is simplified to c * v ** 2 where c is a resistance coefficient
    Default value is based on the air resistance of a generic hatchback
    """

    def __init__(self, track, initial_position=np.array([0.0, 0.0]), initial_velocity=10., resistance_coefficient=0.0005, constant_resistance=0.1,
                 use_discrete_inputs=True, cruise_control_velocity=0.):

        self.track = track
        self.resistance_coefficient = resistance_coefficient
        self.use_discrete_inputs = use_discrete_inputs
        self.constant_resistance = constant_resistance

        # initial_values
        self.initial_position = initial_position.copy()
        self.initial_velocity = initial_velocity

        # state variables
        self.position = initial_position
        self.velocity = initial_velocity
        self.traveled_distance = track.coordinates_to_traveled_distance(initial_position, )

        # inputs
        self._discrete_acceleration_command = 0  # -1, 0 or 1
        self.max_acceleration = 2.5
        self._discrete_acceleration_magnitude = self.max_acceleration * 0.8
        self.acceleration = 0.0

        self.cruise_control_velocity = cruise_control_velocity
        self._cruise_control_last_error = cruise_control_velocity - initial_velocity
        self.cruise_control_active = False
        self._kp = 5.0
        self._kd = 0.05

    def reset(self):
        self.position = copy.copy(self.initial_position)
        self.velocity = copy.copy(self.initial_velocity)
        self.traveled_distance = self.track.coordinates_to_traveled_distance(self.initial_position, )
        self._discrete_acceleration_command = 0  # -1, 0 or 1
        self.acceleration = 0.0

        self._cruise_control_last_error = self.cruise_control_velocity - self.initial_velocity
        self.cruise_control_active = False

    def enable_cruise_control(self, boolean):
        self.cruise_control_active = boolean

    def update_model(self, dt):
        if self.use_discrete_inputs:
            self.acceleration = self._discrete_acceleration_command * self._discrete_acceleration_magnitude
        elif self.cruise_control_active:
            error = (self.cruise_control_velocity - self.velocity)
            control_input = self._kp * error - self._kd * (self._cruise_control_last_error - error) / dt
            self._cruise_control_last_error = error
            self.acceleration = max(-1.0, min(control_input, 1.0)) * self.max_acceleration
        else:
            self._cruise_control_last_error = (self.cruise_control_velocity - self.velocity)

        self.traveled_distance, _ = self.calculate_time_step_1d(dt, self.traveled_distance, self.velocity, self.acceleration, self.resistance_coefficient,
                                                                self.constant_resistance)
        self.position, self.velocity = self.calculate_time_step_2d(dt, self.position, self.velocity, self.heading, self.acceleration,
                                                                   self.resistance_coefficient, self.constant_resistance)

    @staticmethod
    def calculate_time_step_2d(dt, position, velocity, heading, acceleration, resistance_coefficient, constant_resistance):
        direction_vector = np.array([np.cos(heading), np.sin(heading)])

        net_acceleration = acceleration - resistance_coefficient * velocity ** 2 - constant_resistance

        new_velocity = velocity + net_acceleration * dt
        if new_velocity < 0:
            new_velocity = 0.0

        new_position = position + (velocity * dt + (net_acceleration / 2) * dt ** 2) * direction_vector

        return new_position, new_velocity

    @staticmethod
    def calculate_time_step_1d(dt, position, velocity, acceleration, resistance_coefficient, constant_resistance):
        net_acceleration = acceleration - resistance_coefficient * velocity ** 2 - constant_resistance

        new_velocity = velocity + net_acceleration * dt
        if new_velocity < 0:
            new_velocity = 0.0

        new_position = position + (velocity * dt + (net_acceleration / 2) * dt ** 2)

        return new_position, new_velocity

    def reset_to_initial_values(self):
        self.position = self.initial_position
        self.velocity = self.initial_velocity

        # inputs
        self._discrete_acceleration_command = 0
        self._discrete_acceleration_magnitude = 1.
        self.acceleration = 0.0

    def _ignore_control_input(self):
        return self.cruise_control_active

    def set_continuous_acceleration(self, value):
        if self.use_discrete_inputs:
            raise RuntimeError("The point mass object was configured to use discrete inputs")
        if self._ignore_control_input():
            self.acceleration = 0.
        else:
            self.acceleration = value * self.max_acceleration

    def set_discrete_acceleration(self, value):
        if not self.use_discrete_inputs:
            raise RuntimeError("The point mass object was configured to use continuous inputs")
        if self._ignore_control_input():
            self._discrete_acceleration_command = 0
        else:
            self._discrete_acceleration_command = value

    @property
    def heading(self):
        return self.track.get_heading(self.position)
