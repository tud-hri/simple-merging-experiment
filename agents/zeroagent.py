from .agent import Agent


class ZeroAgent(Agent):
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute_discrete_input(self, dt):
        raise NotImplementedError('The PD agent can only calculate a continuous input')

    def compute_continuous_input(self, dt):
        return 0.0

    @property
    def name(self):
        pass
