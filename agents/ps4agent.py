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
import enum

import inputs
from PyQt5 import QtCore

from .agent import Agent


class JoystickInputButton(enum.Enum):
    RIGHT_JOYSTICK = 0
    LEFT_JOYSTICK = 1
    TRIGGERS = 2


class Joystick(QtCore.QThread):
    joy_stick_move_event = QtCore.pyqtSignal(float)

    def __init__(self, game_pad_index, input_button):
        super().__init__()
        self._raw_value = 0
        self._normalized_value = 0.
        self.game_pad_index = game_pad_index

        if input_button is JoystickInputButton.RIGHT_JOYSTICK:
            self._process = self._process_thumb_joystick
            self._event_codes = ['ABS_RY']
        elif input_button is JoystickInputButton.LEFT_JOYSTICK:
            self._process = self._process_thumb_joystick
            self._event_codes = ['ABS_Y']
        elif input_button is JoystickInputButton.TRIGGERS:
            self._process = self._process_trigger
            self._event_codes = ['ABS_Z', 'ABS_RZ']

        try:
            self.device = inputs.devices.gamepads[self.game_pad_index]
        except IndexError:
            raise inputs.UnpluggedError("No ps4 controller found.")

        self.start()

    def set_vibration(self, value):
        self.device._start_vibration_win(value, value)

    def stop_vibration(self):
        self.device._stop_vibration_win()

    def run(self):
        while True:
            events = self.device.read()
            for event in events:
                if event.ev_type == 'Absolute' and event.code in self._event_codes:
                    self._process(event)

    def _process_thumb_joystick(self, event):
        if self._raw_value != event.state:
            self._raw_value = event.state
            self._normalized_value = self._raw_value / 32767

            self.joy_stick_move_event.emit(self._normalized_value)

    def _process_trigger(self, event):
        if event.code == 'ABS_Z':
            raw_value = -event.state
        else:
            raw_value = event.state

        self._raw_value = raw_value
        self._normalized_value = self._raw_value / 255

        self.joy_stick_move_event.emit(self._normalized_value)


class PS4Agent(Agent):

    def __init__(self, game_pad_index=0, input_button=JoystickInputButton.RIGHT_JOYSTICK, use_vibration_feedback=False, desired_velocity=0.0,
                 controllable_object=None, name="PS4 Agent"):
        self.joystick = Joystick(game_pad_index, input_button)
        self.joystick.joy_stick_move_event.connect(self.handle_joystick_move)
        self._name = name

        self.use_vibration_feedback = use_vibration_feedback
        self.desired_velocity = desired_velocity
        self.controllable_object = controllable_object

        self._acceleration = 0.0

    def reset(self):
        pass

    def handle_joystick_move(self, acc):
        self._acceleration = acc
        if abs(self._acceleration) < 0.05:
            self._acceleration = 0.

    def compute_discrete_input(self, dt):
        raise NotImplementedError("the ps4 agent can only generate continuous inputs")

    def compute_continuous_input(self, dt):

        if self.use_vibration_feedback:
            velocity_difference = abs(self.controllable_object.velocity - self.desired_velocity)
            vibration_percentage = velocity_difference / 5.0
            vibration_percentage = min(1., max(vibration_percentage, 0.))
            if vibration_percentage > 0.1:
                self.joystick.set_vibration(vibration_percentage)
            else:
                self.joystick.stop_vibration()

        return self._acceleration

    def stop_vibration(self):
        self.joystick.stop_vibration()

    @property
    def name(self):
        return self._name
