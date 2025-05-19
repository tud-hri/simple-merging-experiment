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
import abc
import copy
import csv
import os
import pickle

import numpy as np
import scipy.io

from agents import CEIAgentV2
from simulation.simulationconstants import SimulationConstants
from trackobjects.trackside import TrackSide


class AbstractSimMaster(abc.ABC):
    def __init__(self, track, simulation_constants, file_name=None, sub_folder=None, save_to_mat_and_csv=True):
        self.vehicle_width = simulation_constants.vehicle_width
        self.vehicle_length = simulation_constants.vehicle_length
        self.simulation_constants = simulation_constants

        self._vehicles = {}
        self._agents = {}

        self._t = 0.  # [ms]
        self.time_index = 0
        self.dt = simulation_constants.dt
        self.max_time = simulation_constants.max_time

        self._track = track

        self._file_name = file_name
        self._save_to_mat_and_csv = save_to_mat_and_csv
        self._sub_folder = sub_folder
        self.end_state = 'Not finished'
        self.agent_types = {}

        self.experimental_conditions = None
        self.current_condition = None

        self._is_recording = False

        # dicts for saving to file and a list that contains all attributes of the sim master object that will be saved
        self.beliefs = {}
        self.observed_velocities = {}
        self.perceived_risks = {}
        self.is_replanning = {}
        self.position_plans = {}
        self.action_plans = {}
        self.positions = {}
        self.travelled_distance = {}
        self.raw_input = {}
        self.velocities = {}
        self.accelerations = {}
        self.net_accelerations = {}
        self.belief_time_stamps = {}
        self.belief_point_contributing_to_risk = {}
        self.risk_bounds = {}

        self._attributes_to_save = ['dt', 'max_time', 'simulation_constants', 'vehicle_width', 'vehicle_length', 'agent_types', 'end_state',
                                    'beliefs', 'observed_velocities', 'perceived_risks', 'is_replanning', 'position_plans', 'action_plans', 'positions',
                                    'travelled_distance', 'raw_input', 'velocities', 'accelerations', 'net_accelerations', 'belief_time_stamps',
                                    'belief_point_contributing_to_risk', 'risk_bounds', 'current_condition']

        number_of_time_steps = int(simulation_constants.max_time / simulation_constants.dt) + 1
        for side in TrackSide:
            self.beliefs[side] = [None] * number_of_time_steps
            self.observed_velocities[side] = [None] * number_of_time_steps
            self.position_plans[side] = [None] * number_of_time_steps
            self.action_plans[side] = [None] * number_of_time_steps
            self.perceived_risks[side] = [None] * number_of_time_steps
            self.is_replanning[side] = [None] * number_of_time_steps
            self.positions[side] = [None] * number_of_time_steps
            self.travelled_distance[side] = [None] * number_of_time_steps
            self.raw_input[side] = [None] * number_of_time_steps
            self.velocities[side] = [None] * number_of_time_steps
            self.accelerations[side] = [None] * number_of_time_steps
            self.net_accelerations[side] = [None] * number_of_time_steps
            self.belief_time_stamps[side] = [None] * number_of_time_steps
            self.belief_point_contributing_to_risk[side] = [None] * number_of_time_steps

    def reset(self):
        self._t = 0.  # [ms]
        self.time_index = 0
        self.end_state = 'Not finished'

        self.beliefs = {}
        self.observed_velocities = {}
        self.perceived_risks = {}
        self.is_replanning = {}
        self.position_plans = {}
        self.action_plans = {}
        self.positions = {}
        self.travelled_distance = {}
        self.raw_input = {}
        self.velocities = {}
        self.accelerations = {}
        self.net_accelerations = {}
        self.belief_time_stamps = {}
        self.belief_point_contributing_to_risk = {}
        self.risk_bounds = {}

        self._attributes_to_save = ['dt', 'max_time', 'simulation_constants', 'vehicle_width', 'vehicle_length', 'agent_types', 'end_state',
                                    'beliefs', 'observed_velocities', 'perceived_risks', 'is_replanning', 'position_plans', 'action_plans', 'positions',
                                    'travelled_distance', 'raw_input', 'velocities', 'accelerations', 'net_accelerations', 'belief_time_stamps',
                                    'belief_point_contributing_to_risk', 'risk_bounds', 'current_condition']

        number_of_time_steps = int(self.simulation_constants.max_time / self.simulation_constants.dt)
        for side in TrackSide:
            self.beliefs[side] = [None] * number_of_time_steps
            self.observed_velocities[side] = [None] * number_of_time_steps
            self.position_plans[side] = [None] * number_of_time_steps
            self.action_plans[side] = [None] * number_of_time_steps
            self.perceived_risks[side] = [None] * number_of_time_steps
            self.is_replanning[side] = [None] * number_of_time_steps
            self.positions[side] = [None] * number_of_time_steps
            self.travelled_distance[side] = [None] * number_of_time_steps
            self.raw_input[side] = [None] * number_of_time_steps
            self.velocities[side] = [None] * number_of_time_steps
            self.accelerations[side] = [None] * number_of_time_steps
            self.net_accelerations[side] = [None] * number_of_time_steps
            self.belief_time_stamps[side] = [None] * number_of_time_steps
            self.belief_point_contributing_to_risk[side] = [None] * number_of_time_steps

    @abc.abstractmethod
    def do_time_step(self, reverse=False):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def add_vehicle(self, side: TrackSide, controllable_object, agent):
        pass

    def get_current_state(self, side: TrackSide):
        try:
            return self._vehicles[side].traveled_distance, self._vehicles[side].velocity, self._vehicles[side].last_net_acceleration
        except KeyError:
            # no vehicle exists on that side
            return None, None, None

    def enable_recording(self, boolean):
        self._is_recording = boolean

    @property
    def t(self):
        return self._t

    def _store_current_status(self):
        for side in self._agents.keys():
            if self.agent_types[side] == CEIAgentV2:
                self.beliefs[side][self.time_index] = copy.deepcopy(self._agents[side].belief)
                self.observed_velocities[side][self.time_index] = self._agents[side].observed_velocity
                self.action_plans[side][self.time_index] = copy.deepcopy(self._agents[side].action_plan)
                self.position_plans[side][self.time_index] = copy.deepcopy(self._agents[side].position_plan)
                self.perceived_risks[side][self.time_index] = copy.deepcopy(self._agents[side].perceived_risk)
                self.is_replanning[side][self.time_index] = copy.deepcopy(self._agents[side].did_plan_update_on_last_tick)
                self.belief_time_stamps[side][self.time_index] = copy.deepcopy(self._agents[side].belief_time_stamps)
                self.belief_point_contributing_to_risk[side][self.time_index] = copy.deepcopy(self._agents[side].belief_point_contributing_to_risk)

            self.positions[side][self.time_index] = self._vehicles[side].position
            self.velocities[side][self.time_index] = self._vehicles[side].velocity
            self.travelled_distance[side][self.time_index] = self._vehicles[side].traveled_distance
            self.raw_input[side][self.time_index] = self._vehicles[side].acceleration / self._vehicles[side].max_acceleration
            self.accelerations[side][self.time_index] = self._vehicles[side].acceleration
            self.net_accelerations[side][self.time_index] = self._vehicles[side].acceleration - self._vehicles[side].resistance_coefficient * \
                                                            self._vehicles[side].velocity ** 2 - self._vehicles[side].constant_resistance

    def _save_to_file(self, file_name_extension=''):
        save_dict = {}
        for variable_name in self._attributes_to_save:
            variable_to_save = self.__getattribute__(variable_name)
            if isinstance(variable_to_save, dict):
                for side in TrackSide:
                    try:
                        if isinstance(variable_to_save[side], list):
                            variable_to_save[side] = [value for value in variable_to_save[side] if value is not None]
                    except KeyError:
                        pass

            save_dict[variable_name] = variable_to_save

        if self._file_name is not None:
            if self._sub_folder:
                folder = os.path.join('data', self._sub_folder)
            else:
                folder = 'data'

            os.makedirs(folder, exist_ok=True)

            pkl_file_name = os.path.join(folder, self._file_name + file_name_extension + '.pkl')
            csv_file_name = os.path.join(folder, self._file_name + file_name_extension + '.csv')
            mat_file_name = os.path.join(folder, self._file_name + file_name_extension + '.mat')

            self._save_pkl(save_dict, pkl_file_name)
            if self._save_to_mat_and_csv:
                self._save_mat(save_dict, mat_file_name)
                self._save_csv(save_dict, csv_file_name)
        return save_dict

    def _save_pkl(self, save_dict, pkl_file_name):
        pkl_dict = copy.deepcopy(save_dict)
        pkl_dict['track'] = self._track
        pkl_dict['experimental_conditions'] = self.experimental_conditions

        try:
            pkl_dict['surroundings'] = self.gui.surroundings
        except AttributeError:
            pkl_dict['surroundings'] = None

        with open(pkl_file_name, 'wb') as f:
            pickle.dump(pkl_dict, f)

    def _save_mat(self, save_dict, mat_file_name):
        mat_dict = self._convert_dict_to_mat_savable_dict(save_dict)
        scipy.io.savemat(mat_file_name, mat_dict, long_field_names=True)

    def _save_csv(self, save_dict, csv_file_name):
        csv_dict = self._convert_dict_to_csv_savable_dict(save_dict)

        with open(csv_file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_dict.keys())

            length = max([len(v) for v in csv_dict.values() if isinstance(v, list)])
            for index in range(length):
                row = []
                for value in csv_dict.values():
                    if isinstance(value, list):
                        try:
                            row.append(value[index])
                        except IndexError:
                            row.append('')
                    elif index == 0:
                        row.append(value)
                    else:
                        row.append('')
                writer.writerow(row)

    def _convert_dict_to_mat_savable_dict(self, d):
        new_dict = {}
        for key, value in d.items():

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    new_keys, new_values = self._get_mat_dict_values(key + '_' + str(sub_key), sub_value)
                    for new_key, new_value in zip(new_keys, new_values):
                        new_dict[new_key] = new_value
            else:
                new_keys, new_values = self._get_mat_dict_values(key, value)
                for new_key, new_value in zip(new_keys, new_values):
                    new_dict[new_key] = new_value

        return new_dict

    @staticmethod
    def _get_mat_dict_values(old_key, old_value):
        keys = []
        values = []

        if isinstance(old_value, list):
            values += [np.array(old_value)]
            keys += [old_key]
        elif isinstance(old_value, SimulationConstants):
            for sim_constants_key, sim_constants_value in old_value.__dict__.items():
                keys += [old_key + '_' + sim_constants_key]
                values += [sim_constants_value]
        elif type(old_value) not in [int, float, str, bool]:
            values += [str(old_value)]
            keys += [old_key]
        else:
            values += [old_value]
            keys += [old_key]
        return keys, values

    def _convert_dict_to_csv_savable_dict(self, d):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, list):
                new_dict[key] = self._convert_list_to_csv_savable_list(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        new_dict[key + '.' + str(sub_key)] = self._convert_list_to_csv_savable_list(sub_value)
                    else:
                        new_dict[key + '.' + str(sub_key)] = sub_value
            elif isinstance(value, SimulationConstants):
                for sim_constants_key, sim_constants_value in value.__dict__.items():
                    new_dict[key + '.' + sim_constants_key] = sim_constants_value
            else:
                new_dict[key] = value

        return new_dict

    def _convert_list_to_csv_savable_list(self, l):
        for index in range(len(l)):
            item = l[index]
            if isinstance(item, list):
                l[index] = self._convert_list_to_csv_savable_list(item)
            elif isinstance(item, np.ndarray):
                l[index] = item.tolist()
        return l
