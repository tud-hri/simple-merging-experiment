from .agent import Agent


class PDAgent(Agent):
    def __init__(self, controllable_object, target_velocity, kd, kp):
        self.controllable_object = controllable_object
        self.target_velocity = target_velocity
        self.kd = kd
        self.kp = kp
        self.last_error = self.target_velocity - self.controllable_object.initial_velocity

    def reset(self):
        self.last_error = self.target_velocity - self.controllable_object.initial_velocity

    def compute_discrete_input(self, dt):
        raise NotImplementedError('The PD agent can only calculate a continuous input')

    def compute_continuous_input(self, dt):
        error = (self.target_velocity - self.controllable_object.velocity)
        control_input = self.kp * error + self.kd * (self.last_error - error) / dt
        self.last_error = error

        return max(-1.0, min(control_input, 1.0))

    @property
    def name(self):
        pass
