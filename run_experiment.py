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
import glob
import os
import random
import sys

import logitech_steering_wheel as lsw
from PyQt5 import QtWidgets, QtCore

from agents import SteeringWheelAgent, ZeroAgent
from controllableobjects import PointMassObject
from experiment.experiment_conditions import get_experiment_conditions
from gui import ExperimentGUI, ParticipantInfoDialog
from simulation.simmaster import SimMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import TunnelMergingTrack
from trackobjects.surroundings import Surroundings
from trackobjects.trackside import TrackSide


def write_participant_numbers(numbers: list):
    path_to_file = os.path.join('data', 'all_participants.txt')

    with open(path_to_file, 'a') as f:
        for number in numbers:
            f.write(str(number) + '\n')


def get_used_participant_numbers():
    all_numbers = []

    path_to_file = os.path.join('data', 'all_participants.txt')

    if os.path.exists(path_to_file):
        with open(path_to_file, 'r') as f:
            for line in f:
                all_numbers.append(int(line))
    else:
        with open(os.path.join('data', 'all_participants.txt'), 'w'):
            pass

    return all_numbers


def save_participant_info(path, participant_info):
    with open(os.path.join(path, 'participant_info.txt'), 'a') as f:
        for key, value in participant_info.items():
            f.write(str(key) + ' = ' + str(value) + '\n')
        f.write('\n')

    if app.desktop().screenCount() == 3:
        if participant_info['side'] == TrackSide.LEFT:
            gui.left_participant_dialog.move(left_center)
            gui.left_participant_dialog.showNormal()
            gui.left_participant_dialog.showFullScreen()
        else:
            gui.right_participant_dialog.move(right_center)
            gui.right_participant_dialog.showNormal()
            gui.right_participant_dialog.showFullScreen()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # generate experiment number
    folders = glob.glob(os.path.join('data', 'experiments', 'experiment_*'))

    existing_experiment_numbers = [int(f.split('experiment_')[-1]) for f in folders]

    if existing_experiment_numbers:
        experiment_number = max(existing_experiment_numbers) + 1
    else:
        experiment_number = 1

    os.makedirs(os.path.join('data', 'experiments', 'experiment_' + str(experiment_number)))

    # generate participant numbers
    all_used_numbers = get_used_participant_numbers()
    participant_numbers = {}

    for side in TrackSide:
        participant_numbers[side] = random.randint(1000, 9999)
        while participant_numbers[side] in all_used_numbers:
            participant_numbers[side] = random.randint(1000, 9999)

    write_participant_numbers(participant_numbers.values())

    # setup simulation constants
    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50,
                                               max_time=30e3)

    time_horizon = 4.
    belief_frequency = 4

    condition_list = get_experiment_conditions(simulation_constants)

    # setup track and surroundings
    track = TunnelMergingTrack(simulation_constants)
    surroundings = Surroundings.load_from_file('surroundings.pkl')

    gui = ExperimentGUI(track, surroundings=surroundings)
    sim_master = SimMaster(gui, track, simulation_constants,
                           file_name='experiment_' + str(experiment_number),
                           sub_folder=os.path.join('experiments', 'experiment_' + str(experiment_number)),
                           use_collision_punishment=True, auto_reset=True,
                           experimental_conditions=condition_list, number_of_training_runs=5)

    initial_conditions = condition_list[0]

    left_point_mass_object = PointMassObject(track,
                                             initial_position=track.traveled_distance_to_coordinates(
                                                 initial_conditions.left_initial_position_offset,
                                                 track_side=TrackSide.LEFT),
                                             initial_velocity=initial_conditions.left_initial_velocity,
                                             use_discrete_inputs=False,
                                             cruise_control_velocity=initial_conditions.left_initial_velocity,
                                             resistance_coefficient=0.005, constant_resistance=0.5)
    right_point_mass_object = PointMassObject(track,
                                              initial_position=track.traveled_distance_to_coordinates(
                                                  initial_conditions.right_initial_position_offset,
                                                  track_side=TrackSide.RIGHT),
                                              initial_velocity=initial_conditions.right_initial_velocity,
                                              use_discrete_inputs=False,
                                              cruise_control_velocity=initial_conditions.right_initial_velocity,
                                              resistance_coefficient=0.005, constant_resistance=0.5)

    lsw.initialize_with_window(True, int(gui.winId()))

    # steering_wheel_right = SteeringWheelAgent(0, use_vibration_feedback=True, desired_velocity=initial_conditions.right_initial_velocity,
    #                                           controllable_object=right_point_mass_object)
    #
    # steering_wheel_left = SteeringWheelAgent(1, use_vibration_feedback=True, desired_velocity=initial_conditions.left_initial_velocity,
    #                                          controllable_object=left_point_mass_object)

    gui.register_controllable_cars(left_point_mass_object, right_point_mass_object, simulation_constants.vehicle_length,
                                   simulation_constants.vehicle_width)

    # sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object, steering_wheel_left)
    # sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object, steering_wheel_right)
    sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object, ZeroAgent())
    sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object, ZeroAgent())

    gui.register_sim_master(sim_master)

    # setup info dialogs
    left_dialog = ParticipantInfoDialog(participant_numbers[TrackSide.LEFT], TrackSide.LEFT, parent=gui)
    left_dialog.accepted.connect(
        lambda: save_participant_info(os.path.join('data', 'experiments', 'experiment_' + str(experiment_number)),
                                      left_dialog.participant_info))
    left_dialog.rejected.connect(
        lambda: save_participant_info(os.path.join('data', 'experiments', 'experiment_' + str(experiment_number)),
                                      left_dialog.participant_info))

    right_dialog = ParticipantInfoDialog(participant_numbers[TrackSide.RIGHT], TrackSide.RIGHT, parent=gui)
    right_dialog.accepted.connect(
        lambda: save_participant_info(os.path.join('data', 'experiments', 'experiment_' + str(experiment_number)),
                                      right_dialog.participant_info))
    right_dialog.rejected.connect(
        lambda: save_participant_info(os.path.join('data', 'experiments', 'experiment_' + str(experiment_number)),
                                      right_dialog.participant_info))

    if app.desktop().screenCount() == 3:
        left_center = app.desktop().screenGeometry(1).center()
        right_center = app.desktop().screenGeometry(2).center()

        gui.left_participant_dialog.showMinimized()
        left_dialog.move(left_center - QtCore.QPoint(int(left_dialog.width() / 2), int(left_dialog.height() / 2)))

        gui.right_participant_dialog.showMinimized()
        right_dialog.move(right_center - QtCore.QPoint(int(right_dialog.width() / 2), int(right_dialog.height() / 2)))

        right_dialog.activateWindow()
        left_dialog.activateWindow()

    return_code = app.exec_()
    lsw.shutdown()
    sys.exit(return_code)
