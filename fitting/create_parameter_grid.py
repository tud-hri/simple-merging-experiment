import pickle
import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import tqdm

from agents import PDAgent, CEIAgentV2
from trackobjects import TunnelMergingTrack
from simulation.simulationconstants import SimulationConstants
from simulation.offlinesimmaster import OfflineSimMaster
from controllableobjects import PointMassObject
from trackobjects.trackside import TrackSide
from experiment.experiment_conditions import get_experiment_conditions


def run_sim_for_condition(lower_thresholds, upper_thresholds, simulation_constants, experimental_condition):
    lower_threshold = lower_thresholds[0]
    upper_threshold = upper_thresholds[0]

    results = {'lower': [],
               'upper': [],
               'deviation': []}

    time = np.array([t * simulation_constants.dt/1000. for t in range(int(simulation_constants.max_time / simulation_constants.dt))])
    track = TunnelMergingTrack(simulation_constants)

    sim_master = OfflineSimMaster(track, simulation_constants, None, experimental_conditions=[experimental_condition], verbose=False,
                                  save_to_mat_and_csv=False)

    left_point_mass_object = PointMassObject(track,
                                             initial_position=track.traveled_distance_to_coordinates(
                                                 experimental_condition.left_initial_position_offset,
                                                 track_side=TrackSide.LEFT),
                                             initial_velocity=experimental_condition.left_initial_velocity,
                                             use_discrete_inputs=False,
                                             cruise_control_velocity=experimental_condition.left_initial_velocity,
                                             resistance_coefficient=0.005, constant_resistance=0.5)
    right_point_mass_object = PointMassObject(track,
                                              initial_position=track.traveled_distance_to_coordinates(
                                                  experimental_condition.right_initial_position_offset,
                                                  track_side=TrackSide.RIGHT),
                                              initial_velocity=experimental_condition.right_initial_velocity,
                                              use_discrete_inputs=False,
                                              cruise_control_velocity=experimental_condition.right_initial_velocity,
                                              resistance_coefficient=0.005, constant_resistance=0.5)

    pd_agent = PDAgent(right_point_mass_object, experimental_condition.right_initial_velocity, kd=0.05, kp=5.0)

    cei_agent = CEIAgentV2(left_point_mass_object, TrackSide.LEFT, simulation_constants.dt, sim_master, track,
                           risk_bounds=(lower_threshold, upper_threshold),
                           time_horizon=6.,
                           preferred_velocity=experimental_condition.left_initial_velocity,
                           vehicle_width=simulation_constants.vehicle_width,
                           belief_frequency=4,
                           vehicle_length=simulation_constants.vehicle_length,
                           theta=1.,
                           memory_length=2,
                           saturation_time=2.,
                           max_comfortable_acceleration=1.6,
                           use_noise=False,
                           use_incentive=False)
    with tqdm.tqdm(total=len(lower_thresholds) * len(upper_thresholds)) as progress_bar:
        for upper_threshold in upper_thresholds:
            for lower_threshold in lower_thresholds:
                cei_agent._risk_bounds = (lower_threshold, upper_threshold)

                left_point_mass_object.reset()
                right_point_mass_object.reset()

                sim_master = OfflineSimMaster(track, simulation_constants, None, experimental_conditions=[experimental_condition], verbose=False,
                                              save_to_mat_and_csv=False)

                cei_agent.sim_master = sim_master
                cei_agent.reset()
                pd_agent.reset()
                sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object, cei_agent)
                sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object, pd_agent)

                simulated_data = sim_master.start()

                mask = (np.array(simulated_data['travelled_distance'][TrackSide.LEFT]) > 50.) & \
                       (np.array(simulated_data['travelled_distance'][TrackSide.LEFT]) < 100.) & \
                       (np.array(simulated_data['travelled_distance'][TrackSide.RIGHT]) > 50.) & \
                       (np.array(simulated_data['travelled_distance'][TrackSide.RIGHT]) < 100.)

                velocity_trace = np.array(simulated_data['velocities'][TrackSide.LEFT])[mask]
                velocity_deviation = velocity_trace - experimental_condition.left_initial_velocity

                first_index = np.argmax(np.abs(velocity_deviation))
                max_deviation = velocity_trace[20]
                time_of_deviation = time[first_index]

                results['lower'].append(lower_threshold)
                results['upper'].append(upper_threshold)
                results['deviation'].append(max_deviation)

                # axes_index = 0 if experimental_condition.name == 'L_4_0' else 1
                # velocity_axes[axes_index].plot(velocity_trace, label=str(upper_threshold))

                progress_bar.update(1)

        results = pd.DataFrame(results)
        folder = os.path.join('..', 'data', 'parameter_effect_grids')
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, '%s.pkl' % experimental_condition.name), 'wb') as f:
            pickle.dump(results, f)

        return results


if __name__ == '__main__':
    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50.,
                                               max_time=30e3)

    all_conditions = get_experiment_conditions(simulation_constants, occurrences=1)
    selected_conditions = []

    n = 25

    upper_thresholds = np.linspace(0.3, 0.9, n)
    lower_thresholds = np.linspace(0.01, 0.4, n)

    args = zip([lower_thresholds] * len(all_conditions),
            [upper_thresholds] * len(all_conditions),
            [simulation_constants] * len(all_conditions),
            all_conditions)

    with mp.Pool(6) as p:
        all_results = p.starmap(run_sim_for_condition, args)
