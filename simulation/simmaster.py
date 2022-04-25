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
import os
import random

from PyQt5 import QtCore

from agents import SteeringWheelAgent
from agents.agent import Agent
from controllableobjects.controlableobject import ControllableObject
from experiment.autosave import auto_save_experiment
from simulation.abstractsimmaster import AbstractSimMaster
from trackobjects import TunnelMergingTrack
from trackobjects.trackside import TrackSide


class SimMaster(AbstractSimMaster):
    def __init__(self, gui, track, simulation_constants, *, file_name=None, sub_folder=None, save_to_mat_and_csv=True, use_collision_punishment=False,
                 auto_reset=False, reset_delay=3., experimental_conditions=None, number_of_training_runs=0, _use_cruise_control_in_tunnel=True):
        super().__init__(track, simulation_constants, file_name, sub_folder=sub_folder, save_to_mat_and_csv=save_to_mat_and_csv)

        self.main_timer = QtCore.QTimer()
        self.main_timer.setInterval(self.dt)
        self.main_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.main_timer.setSingleShot(False)
        self.main_timer.timeout.connect(self.do_time_step)

        self.count_down_timer = QtCore.QTimer()
        self.count_down_timer.setInterval(1000)
        self.count_down_timer.timeout.connect(self.count_down)

        self.reset_timer = QtCore.QTimer()
        self.reset_timer.setInterval(reset_delay * 1000)
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.reset)

        self.auto_reset = auto_reset
        self.use_collision_punishment = use_collision_punishment
        self._use_cruise_control_in_tunnel = _use_cruise_control_in_tunnel
        self.experimental_conditions = experimental_conditions
        self.extra_conditions = []  # to be executed after all regular conditions have been done
        self.condition_number = 0
        self.current_condition = experimental_conditions[0] if experimental_conditions else None
        self.training_run_number = 1
        self.number_of_training_runs = number_of_training_runs
        self.end_training_message_is_displayed = False
        self.is_training = self.number_of_training_runs > 0

        if self.experimental_conditions is not None:
            self._experimental_conditions_iterator = self.experimental_conditions.__iter__()

            if number_of_training_runs == 0:
                # call the first set of initial conditions because they are already applied for the first round
                self._experimental_conditions_iterator.__next__()

        self.count_down_clock = 3  # counts down from 3
        self.history_length = 5

        self.gui = gui

        self.velocity_history = {TrackSide.LEFT: [],
                                 TrackSide.RIGHT: []}

    def start(self):
        self._store_current_status()
        self.count_down()
        self.count_down_timer.start()

    def pause(self):
        self.main_timer.stop()

    def count_down(self):
        if self.count_down_clock == 0:
            self.main_timer.start()
            self.count_down_timer.stop()
            self.gui.show_overlay()
        else:
            self.gui.show_overlay(str(self.count_down_clock))
            self.count_down_clock -= 1

    def reset(self):
        super().reset()

        self.count_down_clock = 3  # counts down from 3

        for side in TrackSide:
            self._vehicles[side].reset()
            self._agents[side].reset()
            self.velocity_history[side] = [self._vehicles[side].velocity] * self.history_length

        if self.experimental_conditions is not None:
            self._apply_experimental_conditions()

        self.gui.update_all_graphics(left_velocity=self._vehicles[TrackSide.LEFT].velocity,
                                     right_velocity=self._vehicles[TrackSide.RIGHT].velocity)
        self.gui.reset()

        if self.number_of_training_runs and not self.is_training and not self.end_training_message_is_displayed:
            self.gui.show_overlay('End of Training')
            self.end_training_message_is_displayed = True
        else:
            self.gui.show_overlay('Not started')

    def test_vibration(self, side, boolean):
        if isinstance(self._agents[side], SteeringWheelAgent):
            if boolean:
                self._agents[side].set_vibration(20)
            else:
                self._agents[side].stop_vibration()

    def _check_extra_conditions(self):
        if self.extra_conditions:
            self._experimental_conditions_iterator = self.extra_conditions.__iter__()
            self.extra_conditions = []
            return self._experimental_conditions_iterator.__next__()
        else:
            return None

    def _apply_experimental_conditions(self):
        if self.training_run_number < self.number_of_training_runs:
            # apply new training run

            new_condition = random.choice(self.experimental_conditions)
            self._apply_condition(new_condition)
            self.training_run_number += 1
            self.is_training = True
        else:
            try:
                new_condition = self._experimental_conditions_iterator.__next__()
            except StopIteration:
                new_condition = self._check_extra_conditions()
                if not new_condition:
                    self.gui.finish_experiment()
                    return

            self._apply_condition(new_condition)
            self.condition_number += 1
            self.is_training = False

        try:
            self.gui.left_participant_dialog.speed_dial.initialize(0., 2 * new_condition.left_initial_velocity)
            self.gui.right_participant_dialog.speed_dial.initialize(0., 2 * new_condition.right_initial_velocity)

            self.gui.left_participant_dialog.view.randomize_mirror()
            self.gui.right_participant_dialog.view.randomize_mirror()
        except ValueError:
            pass

        auto_save_experiment(self.gui.surroundings, self, os.path.join('data', self._sub_folder))

    def _apply_condition(self, condition):
        self.current_condition = condition
        for side in TrackSide:
            self._vehicles[side].velocity = condition.get_initial_velocity(side)
            self._vehicles[side].cruise_control_velocity = condition.get_initial_velocity(side)

            position_offset = condition.get_position_offset(side)
            self._vehicles[side].position[:] = self._track.traveled_distance_to_coordinates(position_offset, track_side=side)[:]
            self._vehicles[side].traveled_distance = position_offset

            self._agents[side].desired_velocity = condition.get_initial_velocity(side)

    def add_vehicle(self, side: TrackSide, controllable_object: ControllableObject, agent: Agent):
        self._vehicles[side] = controllable_object
        self._agents[side] = agent
        self.agent_types[side] = type(agent)

        self.velocity_history[side] = [controllable_object.velocity] * self.history_length

    def get_velocity_history(self, side: TrackSide):
        return self.velocity_history[side]

    def _update_history(self):
        for side in TrackSide:
            try:
                self.velocity_history[side] = [self._vehicles[side].velocity] + self.velocity_history[side][:-1]
            except KeyError:
                # no vehicle exists on that side
                pass

    def _end_simulation(self):
        self.gui.show_overlay(self.end_state)

        if self.is_training:
            file_name_extension = '_training_' + str(self.training_run_number)
        else:
            file_name_extension = '_iter_' + str(self.condition_number)

        self._save_to_file(file_name_extension=file_name_extension)

        if self._is_recording:
            self.gui.record_frame()
            self.gui.stop_recording()

        for agent in self._agents.values():
            try:
                agent.stop_vibration()
            except AttributeError:
                pass

    def _run_cruise_control_check(self):
        enable_cruise_control = False
        if isinstance(self._track, TunnelMergingTrack):
            if self._track.is_in_tunnel(self._vehicles[TrackSide.LEFT].traveled_distance) or \
                    self._track.is_in_tunnel(self._vehicles[TrackSide.RIGHT].traveled_distance):
                enable_cruise_control = self._use_cruise_control_in_tunnel

        for vehicle in self._vehicles.values():
            vehicle.cruise_control_active = enable_cruise_control

    def do_time_step(self, reverse=False):
        self._run_cruise_control_check()

        for controllable_object, agent in zip(self._vehicles.values(), self._agents.values()):
            if controllable_object.use_discrete_inputs:
                controllable_object.set_discrete_acceleration(agent.compute_discrete_input(self.dt / 1000.0))
            else:
                controllable_object.set_continuous_acceleration(agent.compute_continuous_input(self.dt / 1000.0))

            controllable_object.update_model(self.dt / 1000.0)

            if self._track.is_beyond_track_bounds(controllable_object.position):
                self.main_timer.stop()
                self.end_state = "Beyond track bounds"
            elif self._track.is_beyond_finish(controllable_object.position):
                self.main_timer.stop()
                self.end_state = "Finished"

        lb, ub = self._track.get_collision_bounds(self._vehicles[TrackSide.LEFT].traveled_distance, self.vehicle_width, self.vehicle_length, )
        if lb and ub:
            try:
                if lb <= self._vehicles[TrackSide.RIGHT].traveled_distance <= ub:
                    self.main_timer.stop()
                    if self.use_collision_punishment:
                        self.gui.show_collision_punishment(reset_afterwards=self.auto_reset)
                    self.end_state = "Collided"
            except KeyError:
                # no right side vehicle exists
                pass

        self._update_history()

        try:
            self.gui.update_all_graphics(left_velocity=self._vehicles[TrackSide.LEFT].velocity,
                                         right_velocity=self._vehicles[TrackSide.RIGHT].velocity)
        except KeyError:
            # no right side vehicle
            self.gui.update_all_graphics(left_velocity=self._vehicles[TrackSide.LEFT].velocity,
                                         right_velocity=0.0)

        self.gui.update_time_label(self.t / 1000.0)
        self._t += self.dt
        self.time_index += 1
        self._store_current_status()

        if self._t >= self.max_time:
            self.end_state = self.end_state = "Time ran out"

        if self.end_state != 'Not finished':
            self._end_simulation()
            if self.auto_reset and not (self.end_state == "Collided" and self.use_collision_punishment):
                self.reset_timer.start()

        if self._is_recording:
            self.gui.record_frame()
