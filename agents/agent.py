import abc


class Agent(abc.ABC):
    @abc.abstractmethod
    def compute_discrete_input(self, dt):
        """
        This method should calculate a discrete acceleration input; either -1, 0 or 1.

        :param dt: the length of the time step (needed for some forms of MPC)
        :return: one of (-1, 0, 1)
        """
        pass

    @abc.abstractmethod
    def compute_continuous_input(self, dt):
        """
        This method should calculate an acceleration input; expressed between -1 and 1.

        :param dt: the length of the time step (needed for some forms of MPC)
        :return: acceleration between -1 and 1
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        This method should reset the agent object so it can start a new data collection run

        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass
