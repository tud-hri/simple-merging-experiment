"""
Copyright 2022, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of simple-merging-experiment.

simple-merging-experiment is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

simple-merging-experiment is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with simple-merging-experiment.  If not, see <https://www.gnu.org/licenses/>.
"""
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
