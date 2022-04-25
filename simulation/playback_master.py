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
import numpy as np
import pickle

from controllableobjects.controlableobject import ControllableObject
from simulation.simmaster import SimMaster
from trackobjects.trackside import TrackSide
from plotting.conflict_signal import determine_critical_points, calculate_level_of_conflict_signal


class PlaybackMaster(SimMaster):
    def __init__(self, gui, track, simulation_constants, playback_data):
        super().__init__(gui, track, simulation_constants, file_name=None)

        self.playback_data = playback_data
        self.maxtime_index = len([p for p in playback_data['positions'][TrackSide.LEFT] if p is not None]) - 1

    def add_vehicle(self, side: TrackSide, controllable_object: ControllableObject, agent=None):
        super(PlaybackMaster, self).add_vehicle(side, controllable_object, agent=None)

    def set_time(self, time_promille):
        new_index = int((time_promille / 1000.) * self.maxtime_index)
        self.time_index = new_index - 1
        self._t = self.time_index * self.dt
        self.do_time_step()

    def initialize_plots(self):
        left_velocity_trace = self.playback_data['velocities'][TrackSide.LEFT]
        right_velocity_trace = self.playback_data['velocities'][TrackSide.RIGHT]
        position_left = self.playback_data['travelled_distance'][TrackSide.LEFT]
        position_right = self.playback_data['travelled_distance'][TrackSide.RIGHT]

        headway_trace = np.array(position_left) - np.array(position_right)
        average_travelled_distance = (np.array(position_left) + np.array(position_right)) / 2

        time = [(self.playback_data['dt'] / 1000.) * index for index in range(len(position_left))]

        with open(os.path.join('data', 'headway_bounds.pkl'), 'rb') as f:
            headway_bounds = pickle.load(f)

        critical_points = determine_critical_points(headway_bounds, self.simulation_constants)
        conflict_traces = calculate_level_of_conflict_signal(average_travelled_distance, headway_trace, critical_points)

        self.gui.initialize_plots(left_velocity_trace, right_velocity_trace, time, headway_trace, average_travelled_distance, conflict_traces[TrackSide.LEFT],
                                  conflict_traces[TrackSide.RIGHT], headway_bounds, 'b', 'r')

    def do_time_step(self, reverse=False):
        if reverse and self.time_index > 0:
            self._t -= self.dt
            self.time_index -= 1
            self.gui.show_overlay()
        elif not reverse and self.time_index < self.maxtime_index:
            self._t += self.dt
            self.time_index += 1
            self.gui.show_overlay()
        elif not reverse:
            if self.main_timer.isActive():
                self.gui.toggle_play()
            self.gui.show_overlay(self.playback_data['end_state'])
            if self._is_recording:
                self.gui.record_frame()
                self.gui.stop_recording()
            return

        self.gui.update_time_label(self.t / 1000.0)

        for side in self._vehicles.keys():
            self._vehicles[side].position = self.playback_data['positions'][side][self.time_index]
            self._vehicles[side].velocity = self.playback_data['velocities'][side][self.time_index]

        self.gui.update_all_graphics(left_velocity=self.playback_data['velocities'][TrackSide.LEFT][self.time_index],
                                     right_velocity=self.playback_data['velocities'][TrackSide.RIGHT][self.time_index])

        if self._is_recording:
            self.gui.record_frame()
