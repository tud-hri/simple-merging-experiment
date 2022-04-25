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
import sys
import os
import pickle

from PyQt5 import QtWidgets

from gui import ExperimentGUI
from trackobjects.trackside import TrackSide
from simulation.simmaster import SimMaster

if __name__ == '__main__':
    experiment_number = 1

    path = os.path.join('data', 'experiments', 'experiment_' + str(experiment_number), 'auto_save.pkl')

    app = QtWidgets.QApplication(sys.argv)

    with open(path, 'br') as f:
        auto_save_dict = pickle.load(f)

    gui = ExperimentGUI(auto_save_dict['_track'], surroundings=auto_save_dict['gui_surrounding'])

    gui.register_controllable_cars(auto_save_dict['_vehicles'][TrackSide.LEFT],
                                   auto_save_dict['_vehicles'][TrackSide.RIGHT],
                                   auto_save_dict['vehicle_length'],
                                   auto_save_dict['vehicle_width'])

    sim_master = SimMaster(gui, auto_save_dict['_track'], auto_save_dict['simulation_constants'], file_name=auto_save_dict['_file_name'])

    for key, value in auto_save_dict.items():
        sim_master.__setattr__(key, value)

    gui.register_sim_master(sim_master)
    sys.exit(app.exec_())
