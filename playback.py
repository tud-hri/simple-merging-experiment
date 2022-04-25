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

    file_name = os.path.join('experiments', 'experiment_1', 'experiment_1_training_1.pkl')
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
