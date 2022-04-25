import os
import pickle

import tqdm
from matplotlib import pyplot as plt
import numpy as np

from simulation.simulationconstants import SimulationConstants
from trackobjects import TunnelMergingTrack


def generate_data_dict():
    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50.,
                                               max_time=30e3)

    track = TunnelMergingTrack(simulation_constants)
    end_point = track.total_distance

    data_dict = {'positive_headway_bound': [],
                 'negative_headway_bound': [],
                 'average_travelled_distance': []}

    for average_y_position_in_mm in tqdm.trange(int(end_point * 1000)):
        average_y_position = average_y_position_in_mm / 1000.

        lb, ub = track.get_headway_bounds(average_y_position,
                                          vehicle_length=simulation_constants.vehicle_length,
                                          vehicle_width=simulation_constants.vehicle_width)

        data_dict['positive_headway_bound'].append(ub)
        data_dict['negative_headway_bound'].append(lb)
        data_dict['average_travelled_distance'].append(average_y_position)

    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key], dtype=float)
    return data_dict


if __name__ == '__main__':

    path_to_saved_dict = os.path.join('..', 'data', 'headway_bounds.pkl')
    if not os.path.isfile(path_to_saved_dict):
        data_dict = generate_data_dict()

        with open(path_to_saved_dict, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        with open(path_to_saved_dict, 'rb') as f:
            data_dict = pickle.load(f)

    plt.figure()
    plt.plot(data_dict['average_travelled_distance'], data_dict['positive_headway_bound'])
    plt.plot(data_dict['average_travelled_distance'], data_dict['negative_headway_bound'])
    plt.show()
