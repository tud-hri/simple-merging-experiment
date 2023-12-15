import multiprocessing as mp
import os
import pickle

from agents import CEIAgentV2
from controllableobjects import PointMassObject
from experiment.experiment_conditions import get_experiment_conditions
from simulation.offlinesimmaster import OfflineSimMaster
from simulation.simulationconstants import SimulationConstants
from trackobjects import TunnelMergingTrack
from trackobjects.trackside import TrackSide


def simulate_offline(simulation_constants, file_suffix, experiment_number, experimental_condition, threshold_pair, left_incentives, right_incentives):
    folder = 'simulated_data\\experiment_%d' % experiment_number
    os.makedirs(os.path.join('data', folder), exist_ok=True)

    file_name = folder + '\\simulation%s' % file_suffix

    track = TunnelMergingTrack(simulation_constants)
    sim_master = OfflineSimMaster(track, simulation_constants, file_name, experimental_conditions=[experimental_condition], verbose=False,
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

    cei_agent_1 = CEIAgentV2(left_point_mass_object, TrackSide.LEFT, simulation_constants.dt, sim_master, track,
                             risk_bounds=threshold_pair[0],
                             time_horizon=6.,
                             preferred_velocity=experimental_condition.left_initial_velocity,
                             vehicle_width=simulation_constants.vehicle_width,
                             belief_frequency=4,
                             vehicle_length=simulation_constants.vehicle_length,
                             theta=1.,
                             memory_length=2,
                             saturation_time=1.6,
                             max_comfortable_acceleration=1.,
                             use_noise=True,
                             use_incentive=True,
                             incentive_lower_headway=left_incentives[0],
                             incentive_lower_dv=left_incentives[1],
                             incentive_lower_interaction=left_incentives[2],
                             incentive_upper_headway=left_incentives[3],
                             incentive_upper_dv=left_incentives[4],
                             incentive_upper_interaction=left_incentives[5])

    cei_agent_2 = CEIAgentV2(right_point_mass_object, TrackSide.RIGHT, simulation_constants.dt, sim_master, track,
                             risk_bounds=threshold_pair[1],
                             time_horizon=6.,
                             preferred_velocity=experimental_condition.right_initial_velocity,
                             vehicle_width=simulation_constants.vehicle_width,
                             belief_frequency=4,
                             vehicle_length=simulation_constants.vehicle_length,
                             theta=1.,
                             memory_length=2,
                             saturation_time=1.6,
                             max_comfortable_acceleration=1.,
                             use_noise=True,
                             use_incentive=True,
                             incentive_lower_headway=right_incentives[0],
                             incentive_lower_dv=right_incentives[1],
                             incentive_lower_interaction=right_incentives[2],
                             incentive_upper_headway=right_incentives[3],
                             incentive_upper_dv=right_incentives[4],
                             incentive_upper_interaction=right_incentives[5])

    sim_master.add_vehicle(TrackSide.LEFT, left_point_mass_object, cei_agent_1)
    sim_master.add_vehicle(TrackSide.RIGHT, right_point_mass_object, cei_agent_2)

    sim_master.start()

    print('')
    print('simulation ended with exit status: ' + sim_master.end_state)


if __name__ == '__main__':
    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50.,
                                               max_time=30e3)

    all_conditions = get_experiment_conditions(simulation_constants, occurrences=10)

    with open(os.path.join('data', 'fitted_thresholds.pkl'), 'rb') as f:
        fitted_parameters = pickle.load(f)

    incentives = (float(fitted_parameters['lower_parameters']['tunnel_exit_headway']),
                  float(fitted_parameters['lower_parameters']['dv']),
                  float(fitted_parameters['lower_parameters']['tunnel_exit_headway:dv']),
                  float(fitted_parameters['upper_parameters']['tunnel_exit_headway']),
                  float(fitted_parameters['upper_parameters']['dv']),
                  float(fitted_parameters['upper_parameters']['tunnel_exit_headway:dv']))

    thresholds = {TrackSide.LEFT: {},
                  TrackSide.RIGHT: {}}

    for side in TrackSide:
        for pair in [4, 5, 6, 8, 9, 10, 11, 12, 13]:
            key = '%s-%d' % (str(side), pair)
            thresholds[side][pair] = (float(fitted_parameters['lower_parameters']['Intercept']) +
                                      float(fitted_parameters['lower_random_effects'][key]),
                                      float(fitted_parameters['upper_parameters']['Intercept']) +
                                      float(fitted_parameters['upper_random_effects'][key]))

    for participant_pair in [4, 5, 6, 8, 9, 10, 11, 12, 13]:
        threshold_pair = (thresholds[TrackSide.LEFT][participant_pair], thresholds[TrackSide.RIGHT][participant_pair])

        number_of_trials = len(all_conditions)
        print('TODO: ' + str(number_of_trials))

        file_suffixes = ['_%d_iter_%d' % (participant_pair, i) for i in range(number_of_trials)]
        args = zip([simulation_constants] * number_of_trials, file_suffixes, [participant_pair] * number_of_trials, all_conditions,
                   [threshold_pair] * number_of_trials, [incentives] * number_of_trials, [incentives] * number_of_trials)

        with mp.Pool(8) as p:
            p.starmap(simulate_offline, args)
