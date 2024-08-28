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
import pickle
import sys

from PyQt5 import QtWidgets

from controllableobjects import PointMassObject
from gui import SimulationGui
from simulation.playback_master import PlaybackMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects.trackside import TrackSide

simulation_constants: SimulationConstants

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    file_name = os.path.join('experiment_data', 'experiment_4', 'experiment_4_training_1.pkl')
    with open(os.path.join('data', file_name), 'rb') as f:
        playback_data = pickle.load(f)

    simulation_constants = playback_data['simulation_constants']

    dt = simulation_constants.dt  # ms
    vehicle_width = simulation_constants.vehicle_width
    vehicle_length = simulation_constants.vehicle_length

    track = playback_data['track']
    surroundings = playback_data['surroundings']

    gui = SimulationGui(track, surroundings=surroundings)
    sim_master = PlaybackMaster(gui, track, simulation_constants, playback_data)

    left_point_mass_object = PointMassObject(track, initial_position=track.get_start_position(TrackSide.LEFT), use_discrete_inputs=False)
    gui.add_controllable_car(left_point_mass_object, vehicle_length, vehicle_width, side_for_dial=TrackSide.LEFT, color='blue')
    sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object)

    if TrackSide.RIGHT in playback_data['agent_types'].keys():
        right_point_mass_object = PointMassObject(track, initial_position=track.get_start_position(TrackSide.RIGHT), use_discrete_inputs=False)
        gui.add_controllable_car(right_point_mass_object, vehicle_length, vehicle_width, side_for_dial=TrackSide.RIGHT, color='red')
        sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object)

    gui.register_sim_master(sim_master)
    sim_master.initialize_plots()
    sys.exit(app.exec_())
