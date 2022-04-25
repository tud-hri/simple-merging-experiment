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
