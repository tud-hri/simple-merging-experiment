import logitech_steering_wheel as lsw

from agents.agent import Agent


class SteeringWheelAgent(Agent):
    def __init__(self, steering_wheel_index, use_vibration_feedback=False, desired_velocity=None, controllable_object=None):
        connected = lsw.is_connected(steering_wheel_index)

        if not connected:
            raise RuntimeError('Could not connect to steering wheel ' + str(steering_wheel_index) + '. Make sure you initialize the logitech steering wheel '
                                                                                                    'sdk before creating a SteeringWheelAgent object.')

        self.index = steering_wheel_index
        self.use_vibration_feedback = use_vibration_feedback
        self.desired_velocity = desired_velocity
        self.controllable_object = controllable_object

        lsw.update()

    def compute_discrete_input(self, dt):
        raise NotImplemented('A steering wheel agent can only compute continuous inputs')

    def compute_continuous_input(self, dt):
        lsw.update()
        state = lsw.get_state(self.index)

        value = (2 ** 15 - state.lY) - (2 ** 15 - state.lRz)
        value /= 2 ** 16

        if self.use_vibration_feedback:
            velocity_difference = abs(self.controllable_object.velocity - self.desired_velocity)

            vibration_percentage = (velocity_difference * 2 / self.desired_velocity) * 100  # vibrate at 100% if the velocity difference is 50% of the desired v
            vibration_percentage = int(vibration_percentage)
            if vibration_percentage > 5.:
                self.set_vibration(vibration_percentage)
            else:
                self.stop_vibration()

        return value

    def set_vibration(self, percentage: int):
        percentage = min(100, max(0, percentage))
        lsw.play_dirt_road_effect(self.index, percentage)

    def stop_vibration(self):
        lsw.stop_dirt_road_effect(self.index)

    def reset(self):
        pass

    @property
    def name(self):
        pass
