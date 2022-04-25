import abc


class ControllableObject(abc.ABC):
    @abc.abstractmethod
    def set_continuous_acceleration(self, value):
        """
        Sets the continuous acceleration for this agent; the command should be between -1 and 1
        :param value: acceleration command between -1 and 1
        :return:
        """
        pass

    @abc.abstractmethod
    def set_discrete_acceleration(self, value):
        """
        Sets the discrete acceleration for this agent; the command should be one of (-1, 0, 1)
        :param value: acceleration command one of (-1, 0, 1)
        :return:
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def calculate_time_step_2d(dt, position, velocity, heading, acceleration, resistance_coefficient, constant_resistance):
        """
        Calculates and returns the position and velocity after this time step but does not apply this to the object. This model is used in MPC

        :param constant_resistance:
        :param resistance_coefficient:
        :param dt: duration of the time step in s
        :param position: old position
        :param velocity: old velocity
        :param heading: current heading
        :param acceleration: acceleration command
        :return: new position, new velocity
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def calculate_time_step_1d(dt, position, velocity, acceleration, resistance_coefficient, constant_resistance):
        """
        Calculates and returns the position and velocity after this time step but does not apply this to the object. This model is used in MPC

        :param constant_resistance:
        :param resistance_coefficient:
        :param dt: duration of the time step in s
        :param position: old position
        :param velocity: old velocity
        :param acceleration: acceleration command
        :return: new position, new velocity
        """
        pass

    @abc.abstractmethod
    def reset_to_initial_values(self):
        """
        should reset the complete object to its initial state. Including the control inputs
        :return:
        """
        pass
