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
import copy
import os
import pickle

import numpy as np
import pandas as pd

from tools.append_experiment_with_belief import append_experiment_data_with_belief
from plotting.conflict_signal import calculate_level_of_conflict_signal, determine_critical_points
from trackobjects.trackside import TrackSide


def find_who_exits_the_tunnel_first(experimental_conditions, track_section_length):
    unique_conditions = set(experimental_conditions)
    results = {}
    for condition in unique_conditions:
        left_time = (track_section_length - condition.left_initial_position_offset) / condition.left_initial_velocity
        right_time = (track_section_length - condition.right_initial_position_offset) / condition.right_initial_velocity
        results[condition.name] = TrackSide.LEFT if left_time < right_time else TrackSide.RIGHT

    return results


def get_first(data):
    if data['end_state'] == 'Collided':
        return None
    elif data['travelled_distance'][TrackSide.LEFT][-1] > data['travelled_distance'][TrackSide.RIGHT][-1]:
        return TrackSide.LEFT
    else:
        return TrackSide.RIGHT


def get_time_gap(data):
    time_array = np.array([t * data['dt'] / 1000 for t in range(len(data['travelled_distance'][TrackSide.LEFT]))])

    if data['end_state'] == 'Collided':
        return None
    else:
        try:
            left_c_t = time_array[np.array(data['travelled_distance'][TrackSide.LEFT]) > data['track'].total_distance * (2. / 3.)][0]
        except IndexError:
            left_c_t = np.inf
        try:
            right_c_t = time_array[np.array(data['travelled_distance'][TrackSide.RIGHT]) > data['track'].total_distance * (2. / 3.)][0]
        except IndexError:
            right_c_t = np.inf

        return right_c_t - left_c_t


def get_inv_ttc_trace(data):
    gap = np.array(data['travelled_distance'][TrackSide.LEFT]) - np.array(data['travelled_distance'][TrackSide.RIGHT])
    gap[abs(gap) < data['vehicle_length']] = 0.

    signs = np.sign(gap)
    gap -= signs * data['vehicle_length']

    relative_velocity = np.array(data['velocities'][TrackSide.RIGHT]) - np.array(data['velocities'][TrackSide.LEFT])

    # suppress warnings when dividing by 0.
    np.seterr(invalid='ignore', divide='ignore')
    inv_ttc = list(1 / (gap / relative_velocity))

    return inv_ttc


def get_raw_input_integral(data, include_positive=True, include_negative=True):
    input_integrals = {TrackSide.LEFT: None,
                       TrackSide.RIGHT: None}

    if data['end_state'] == 'Collided':
        return input_integrals

    for side in TrackSide:
        is_outside_tunnel = np.array(data['travelled_distance'][side]) > data['simulation_constants'].track_section_length

        include_data = np.array(data['raw_input'][side]) == 0.

        if include_positive:
            include_data |= np.array(data['raw_input'][side]) > 0.
        if include_negative:
            include_data |= np.array(data['raw_input'][side]) < 0.

        user_input = np.array(data['raw_input'][side])[include_data & is_outside_tunnel]
        integral = sum(abs(user_input) * data['dt'] / 1000.)
        input_integrals[side] = integral

    return input_integrals


def get_acceleration_integral(data, include_positive=True, include_negative=True):
    acceleration_integrals = {TrackSide.LEFT: None,
                              TrackSide.RIGHT: None}

    if data['end_state'] == 'Collided':
        return acceleration_integrals

    for side in TrackSide:
        is_outside_tunnel = np.array(data['travelled_distance'][side]) > data['simulation_constants'].track_section_length

        acceleration_data = np.array(data['accelerations'][side]) - 0.005 * np.array(data['velocities'][side]) ** 2 - 0.5
        include_data = acceleration_data == 0.

        if include_positive:
            include_data |= acceleration_data > 0.
        if include_negative:
            include_data |= acceleration_data < 0.

        accelerations = acceleration_data[include_data & is_outside_tunnel]
        integral = sum(abs(accelerations) * data['dt'] / 1000.)
        acceleration_integrals[side] = integral

    return acceleration_integrals


def get_velocity_reversals(data):
    reversals = {}
    for side in TrackSide:
        true_acceleration = np.gradient(data['velocities'][side], data['dt'] / 1000)
        sign_changes = true_acceleration * np.roll(true_acceleration, -1)
        reversal_count = sum(sign_changes < 0.)
        reversals[side] = reversal_count

    return reversals


def get_min_ttc_after_merge(data):
    if data['end_state'] == 'Collided':
        return None

    gap = np.array(data['travelled_distance'][TrackSide.LEFT]) - np.array(data['travelled_distance'][TrackSide.RIGHT])
    gap[abs(gap) < data['vehicle_length']] = 0.

    signs = np.sign(gap)
    gap -= signs * data['vehicle_length']

    relative_velocity = np.array(data['velocities'][TrackSide.RIGHT]) - np.array(data['velocities'][TrackSide.LEFT])

    # suppress warnings when dividing by 0.
    np.seterr(invalid='ignore', divide='ignore')
    ttc = gap / relative_velocity

    after_merge = (np.array(data['travelled_distance'][TrackSide.LEFT]) > 2 * data['simulation_constants'].track_section_length) | (
            np.array(data['travelled_distance'][TrackSide.RIGHT]) > 2 * data['simulation_constants'].track_section_length)

    positive_ttc_after_merge = ttc[after_merge]
    positive_ttc_after_merge = positive_ttc_after_merge[positive_ttc_after_merge > 0]

    if not len(positive_ttc_after_merge):
        # all ttc's were negative
        return np.inf
    else:
        return min(positive_ttc_after_merge)


def check_if_on_collision_course_for_point(travelled_distance_collision_point, data):
    point_predictions = {}

    for side in TrackSide:
        point_predictions[side] = np.array(data['travelled_distance'][side]) + np.array(data['velocities'][side]) * (
                travelled_distance_collision_point - np.array(data['travelled_distance'][side.other])) / np.array(data['velocities'][side.other])

    lb, ub = data['track'].get_collision_bounds(travelled_distance_collision_point, data['vehicle_width'], data['vehicle_length'])

    on_collision_course = ((lb < point_predictions[TrackSide.LEFT]) & (point_predictions[TrackSide.LEFT] < ub)) | \
                          ((lb < point_predictions[TrackSide.RIGHT]) & (point_predictions[TrackSide.RIGHT] < ub))
    return on_collision_course


def calculate_conflict_resolved_time(data):
    if data['end_state'] == 'Collided':
        return None, None

    time = [t * data['dt'] / 1000 for t in range(len(data['velocities'][TrackSide.LEFT]))]
    track = data['track']

    merge_point_collision_course = check_if_on_collision_course_for_point(2 * data['simulation_constants'].track_section_length, data)
    try:
        threshold_collision_course = check_if_on_collision_course_for_point(track._upper_bound_threshold + 1e-3, data)
    except AttributeError:  # a straight track has no threshold
        threshold_collision_course = merge_point_collision_course
    end_point_collision_course = check_if_on_collision_course_for_point(data['track'].total_distance, data)

    on_collision_course = merge_point_collision_course | threshold_collision_course | end_point_collision_course

    approach_mask = ((np.array(data['travelled_distance'][TrackSide.RIGHT]) > data['simulation_constants'].track_section_length) &
                     (np.array(data['travelled_distance'][TrackSide.LEFT]) > data['simulation_constants'].track_section_length))

    indices_of_conflict_resolved = ((on_collision_course == False) & approach_mask)

    try:
        conflict_resolved_index = np.where(indices_of_conflict_resolved)[0][0]
        time_of_conflict_resolved = np.array(time)[conflict_resolved_index]
    except IndexError:
        conflict_resolved_index = None
        time_of_conflict_resolved = None

    return time_of_conflict_resolved, conflict_resolved_index


def calculate_time_of_first_action(data, threshold=10e-3):
    first_action_times = {}
    time_array = np.array([t * data['dt'] / 1000 for t in range(len(data['velocities'][TrackSide.LEFT]))])

    track_section_length = data['simulation_constants'].track_section_length

    for side in TrackSide:
        true_acceleration = np.array(data['net_accelerations'][side])
        participant_action_times = time_array[(np.abs(true_acceleration) > threshold) & (np.array(data['travelled_distance'][side]) > track_section_length)]

        try:
            time_of_first_action = participant_action_times[0]
        except IndexError:
            time_of_first_action = time_array[-1]

        first_action_times[side] = time_of_first_action

    return first_action_times


def get_data_as_dicts(data, condition_name, experiment_iteration, is_simulation):
    global_metrics_dict = {'who went first': [],
                           'time_gap_at_merge': [],
                           'min_ttc_after_merge': [],
                           'conflict_resolved_time': [],
                           'nett crt': [],
                           'out_of_tunnel_time': [],
                           'merge_time': [],
                           'condition': [],
                           'end state': [],
                           'std of difficulty': [],
                           'headway at merge point': [],
                           'trial_number': [],
                           'is_simulation': [],
                           'experiment_number': [],
                           'left risk bounds': [],
                           'right risk bounds': [],
                           'realistic velocities': []}

    individual_metrics_dict = {'raw input integral': [],
                               'raw positive input integral': [],
                               'raw negative input integral': [],
                               'acceleration integral': [],
                               'positive acceleration integral': [],
                               'negative acceleration integral': [],
                               'max acceleration': [],
                               'v after one second': [],
                               'velocity deviation after one second': [],
                               'moment of first response': [],
                               'number of replans': [],
                               'max abs deviation from desired v': [],
                               'extreme deviation from desired v': [],
                               'condition': [],
                               'end state': [],
                               'trial_number': [],
                               'is_simulation': [],
                               'experiment_number': [],
                               'side': [],
                               'risk bounds': [],
                               'other risk bounds': []}

    global_traces_dict = {'inverse ttc [1/s]': [],
                          'headway [m]': [],
                          'relative velocity [m/s]': [],
                          'y position of right vehicle [m]': [],
                          'average travelled distance [m]': [],
                          'difficulty left': [],
                          'difficulty right': [],
                          'time [s]': [],
                          'condition': [],
                          'trial_number': [],
                          'is_simulation': [],
                          'experiment_number': [],
                          'end state': [],
                          'left risk bounds': [],
                          'right risk bounds': []}

    individual_traces_dict = {'velocity [m/s]': [],
                              'input': [],
                              'perceived risk': [],
                              'acceleration [m/s^2]': [],
                              'jerk [m/s^3]': [],
                              'travelled distance [m]': [],
                              'deviation from constant v [m]': [],
                              'time [s]': [],
                              'side': [],
                              'condition': [],
                              'trial_number': [],
                              'is_simulation': [],
                              'experiment_number': [],
                              'end state': [],
                              'risk bounds': [],
                              'other risk bounds': []}

    end_state = data['end_state']

    time = [t * data['dt'] / 1000 for t in range(len(data['velocities'][TrackSide.LEFT]))]
    global_metrics_dict['condition'].append(condition_name)

    if data['risk_bounds']:
        left_risk_bounds = data['risk_bounds'][TrackSide.LEFT]
        right_risk_bounds = data['risk_bounds'][TrackSide.RIGHT]
    else:
        left_risk_bounds = None
        right_risk_bounds = None

    if get_first(data):
        global_metrics_dict['who went first'].append(get_first(data))
        global_metrics_dict['time_gap_at_merge'].append(get_time_gap(data))
    else:
        global_metrics_dict['who went first'].append('Collision')
        global_metrics_dict['time_gap_at_merge'].append(None)

    global_metrics_dict['left risk bounds'].append(left_risk_bounds)
    global_metrics_dict['right risk bounds'].append(right_risk_bounds)
    global_metrics_dict['min_ttc_after_merge'].append(get_min_ttc_after_merge(data))
    global_metrics_dict['end state'].append(end_state)
    global_metrics_dict['trial_number'].append(experiment_iteration)
    global_metrics_dict['is_simulation'].append(is_simulation)
    global_metrics_dict['experiment_number'].append(experiment_iteration.split('-')[0])

    both_out_of_tunnel_mask = (np.array(data['travelled_distance'][TrackSide.LEFT]) > data['simulation_constants'].track_section_length) & \
                              (np.array(data['travelled_distance'][TrackSide.RIGHT]) > data['simulation_constants'].track_section_length)
    both_before_merge_mask = (np.array(data['travelled_distance'][TrackSide.LEFT]) < 2 * data['simulation_constants'].track_section_length) & \
                             (np.array(data['travelled_distance'][TrackSide.RIGHT]) < 2 * data['simulation_constants'].track_section_length)
    out_of_tunnel_time = np.array(time)[both_out_of_tunnel_mask][0]
    global_metrics_dict['out_of_tunnel_time'].append(out_of_tunnel_time)
    out_of_tunnel_index = np.where(both_out_of_tunnel_mask)[0][0]

    participants_have_control_mask = copy.copy(both_out_of_tunnel_mask)
    participants_have_control_mask[np.where(participants_have_control_mask)[0][0]] = False

    average_travelled_distance = (np.array(data['travelled_distance'][TrackSide.LEFT]) + np.array(data['travelled_distance'][TrackSide.RIGHT])) / 2.
    try:
        merge_point_index = np.where(average_travelled_distance >= 2 * data['simulation_constants'].track_section_length)[0][0]
        global_metrics_dict['merge_time'].append(time[merge_point_index])
    except IndexError:
        global_metrics_dict['merge_time'].append(None)

    time_of_conflict_resolved, conflict_resolved_index = calculate_conflict_resolved_time(data)
    global_metrics_dict['conflict_resolved_time'].append(time_of_conflict_resolved)
    try:
        global_metrics_dict['nett crt'].append(time_of_conflict_resolved - out_of_tunnel_time)
    except TypeError:
        global_metrics_dict['nett crt'].append(None)

    realistic_velocity_bounds = (3, 14.5)
    velocities_within_bounds = True
    for side in TrackSide:
        velocities_within_bounds &= min(data['velocities'][side]) >= realistic_velocity_bounds[0]
        velocities_within_bounds &= max(data['velocities'][side]) <= realistic_velocity_bounds[1]

    global_metrics_dict['realistic velocities'].append(velocities_within_bounds)

    raw_input_integrals = get_raw_input_integral(data)
    raw_acceleration_integrals = get_raw_input_integral(data, include_negative=False)
    raw_braking_integrals = get_raw_input_integral(data, include_positive=False)

    times_of_first_action = calculate_time_of_first_action(data, threshold=0.01)

    acceleration_integrals = get_acceleration_integral(data)
    positive_acceleration_integrals = get_acceleration_integral(data, include_negative=False)
    negative_acceleration_integrals = get_acceleration_integral(data, include_positive=False)

    ttc_trace = get_inv_ttc_trace(data)
    condition_list = [condition_name] * len(time)
    trial_number_list = [experiment_iteration] * len(time)
    is_simulation_list = [is_simulation] * len(time)
    experiment_number_list = [experiment_iteration.split('-')[0]] * len(time)

    global_traces_dict['inverse ttc [1/s]'] += ttc_trace

    headway = np.array(data['travelled_distance'][TrackSide.LEFT]) - np.array(data['travelled_distance'][TrackSide.RIGHT])

    global_traces_dict['y position of right vehicle [m]'] += list(np.array(data['positions'][TrackSide.RIGHT])[:, 1])
    global_traces_dict['relative velocity [m/s]'] += list(np.array(data['velocities'][TrackSide.LEFT]) - np.array(data['velocities'][TrackSide.RIGHT]))
    global_traces_dict['headway [m]'] += list(headway)
    global_traces_dict['time [s]'] += time
    average_travelled_distance = (np.array(data['travelled_distance'][TrackSide.LEFT]) + np.array(data['travelled_distance'][TrackSide.RIGHT])) / 2.
    global_traces_dict['average travelled distance [m]'] += list(average_travelled_distance)
    global_traces_dict['condition'] += condition_list
    global_traces_dict['trial_number'] += trial_number_list
    global_traces_dict['is_simulation'] += is_simulation_list
    global_traces_dict['experiment_number'] += experiment_number_list
    global_traces_dict['end state'] += [end_state] * len(time)
    global_traces_dict['left risk bounds'] += [left_risk_bounds] * len(time)
    global_traces_dict['right risk bounds'] += [right_risk_bounds] * len(time)

    simulation_constants = data['simulation_constants']
    with open(os.path.join('..', 'data', 'headway_bounds.pkl'), 'rb') as f:
        headway_bounds = pickle.load(f)

    critical_points = determine_critical_points(headway_bounds, simulation_constants)
    level_of_conflict = calculate_level_of_conflict_signal(average_travelled_distance, headway, critical_points)

    global_traces_dict['difficulty right'] += list(level_of_conflict[TrackSide.RIGHT])
    global_traces_dict['difficulty left'] += list(level_of_conflict[TrackSide.LEFT])

    if get_first(data):
        global_metrics_dict['std of difficulty'] += [np.std(np.array(level_of_conflict[get_first(data)])[both_before_merge_mask & both_out_of_tunnel_mask])]
    else:
        global_metrics_dict['std of difficulty'] += [None]

    try:
        merge_point_index = np.where(average_travelled_distance >= 2 * data['simulation_constants'].track_section_length)[0][0]
        global_metrics_dict['headway at merge point'] += [headway[merge_point_index]]
    except IndexError:
        global_metrics_dict['headway at merge point'] += [None]

    for side in TrackSide:
        individual_metrics_dict['raw input integral'] += [raw_input_integrals[side]]
        individual_metrics_dict['raw positive input integral'] += [raw_acceleration_integrals[side]]
        individual_metrics_dict['raw negative input integral'] += [raw_braking_integrals[side]]
        individual_metrics_dict['acceleration integral'] += [acceleration_integrals[side]]
        individual_metrics_dict['positive acceleration integral'] += [positive_acceleration_integrals[side]]
        individual_metrics_dict['negative acceleration integral'] += [negative_acceleration_integrals[side]]
        individual_metrics_dict['max acceleration'] += [max(data['accelerations'][side])]
        second_index = int(1000 / data['dt'])
        individual_metrics_dict['v after one second'] += [np.array(data['velocities'][side])[participants_have_control_mask][second_index]]
        individual_metrics_dict['velocity deviation after one second'] += [
            np.array(data['velocities'][side])[participants_have_control_mask][second_index] - np.array(data['velocities'][side])[0]]
        individual_metrics_dict['moment of first response'] += [times_of_first_action[side]]
        individual_metrics_dict['number of replans'] += [np.abs(data['is_replanning'][side]).sum()]
        velocity_deviation = np.array(data['velocities'][side])[participants_have_control_mask] - data['current_condition'].get_initial_velocity(side)
        individual_metrics_dict['max abs deviation from desired v'] += [abs(velocity_deviation).max()]
        extreme_deviation = velocity_deviation.max() if velocity_deviation.max() > -velocity_deviation.min() else velocity_deviation.min()
        individual_metrics_dict['extreme deviation from desired v'] += [extreme_deviation]
        trace_length = len(velocity_deviation)
        individual_metrics_dict['condition'] += [condition_name]
        individual_metrics_dict['trial_number'] += [experiment_iteration]
        individual_metrics_dict['is_simulation'] += [is_simulation]
        individual_metrics_dict['experiment_number'] += [experiment_iteration.split('-')[0]]
        individual_metrics_dict['end state'].append(end_state)
        individual_metrics_dict['side'] += [str(side)]
        constant_v_travelled_distance = np.array(time) * data['current_condition'].get_initial_velocity(side) + data['travelled_distance'][side][0]

        individual_traces_dict['velocity [m/s]'] += data['velocities'][side]
        individual_traces_dict['travelled distance [m]'] += data['travelled_distance'][side]
        individual_traces_dict['deviation from constant v [m]'] += list(data['travelled_distance'][side] - constant_v_travelled_distance)
        individual_traces_dict['input'] += data['raw_input'][side]
        individual_traces_dict['perceived risk'] += data['perceived_risks'][side]
        individual_traces_dict['acceleration [m/s^2]'] += data['net_accelerations'][side]

        jerk = list(np.array(data['net_accelerations'][side])[~participants_have_control_mask] * 0.0) + list(
            np.gradient(np.array(data['net_accelerations'][side])[participants_have_control_mask], data['dt']))

        individual_traces_dict['jerk [m/s^3]'] += jerk
        individual_traces_dict['time [s]'] += time
        individual_traces_dict['condition'] += [condition_name] * len(data['velocities'][side])
        individual_traces_dict['trial_number'] += [experiment_iteration] * len(data['velocities'][side])
        individual_traces_dict['is_simulation'] += [is_simulation] * len(data['velocities'][side])
        individual_traces_dict['experiment_number'] += [experiment_iteration.split('-')[0]] * len(data['velocities'][side])
        individual_traces_dict['end state'] += [end_state] * len(time)

        if side is TrackSide.LEFT:
            individual_metrics_dict['risk bounds'] += [left_risk_bounds]
            individual_traces_dict['risk bounds'] += [left_risk_bounds] * len(time)
            individual_metrics_dict['other risk bounds'] += [right_risk_bounds]
            individual_traces_dict['other risk bounds'] += [right_risk_bounds] * len(time)
        else:
            individual_metrics_dict['risk bounds'] += [right_risk_bounds]
            individual_traces_dict['risk bounds'] += [right_risk_bounds] * len(time)
            individual_metrics_dict['other risk bounds'] += [left_risk_bounds]
            individual_traces_dict['other risk bounds'] += [left_risk_bounds] * len(time)

        individual_traces_dict['side'] += [str(side)] * len(data['velocities'][side])
    return global_metrics_dict, global_traces_dict, individual_metrics_dict, individual_traces_dict


def load_data_from_file(index, file, override_trial_numbers, included_conditions, add_reconstructed_risk, progress_queue=None):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    if add_reconstructed_risk:
        saturation_times = {TrackSide.LEFT: 1.6,
                            TrackSide.RIGHT: 1.6}
        time_horizons = {TrackSide.LEFT: 6.,
                         TrackSide.RIGHT: 6.}
        memory_lengths = {TrackSide.LEFT: 2,
                          TrackSide.RIGHT: 2}
        belief_frequencies = {TrackSide.LEFT: 4,
                              TrackSide.RIGHT: 4}
        expected_accelerations = {TrackSide.LEFT: 1.,
                                  TrackSide.RIGHT: 1.}
        append_experiment_data_with_belief(data, saturation_times, time_horizons, memory_lengths, belief_frequencies,
                                           expected_accelerations, )

    is_simulation = 'simulation' in file or override_trial_numbers

    if override_trial_numbers:
        experiment_number = 0
        experiment_iteration = index + 1
    else:
        try:
            experiment_number = int(file.split('_')[-3])
            experiment_iteration = int(file.split('_')[-1].replace('.pkl', ''))
        except ValueError:
            experiment_number = 0
            experiment_iteration = 0

    try:
        condition_name = data['current_condition'].name
    except KeyError:
        condition_name = data['experimental_conditions'][experiment_iteration - 1].name

    if condition_name in included_conditions:
        iteration_name = str(experiment_number) + '-' + str(experiment_iteration)
        if is_simulation:
            iteration_name += '-sim'
        global_metrics_dict, global_traces_dict, individual_metrics_dict, individual_traces_dict = \
            get_data_as_dicts(data, condition_name, iteration_name, is_simulation)
    else:
        global_metrics_dict, global_traces_dict, individual_metrics_dict, individual_traces_dict = {}, {}, {}, {}, {}

    if progress_queue is not None:
        progress_queue.put(1)

    return pd.DataFrame(global_metrics_dict), pd.DataFrame(individual_metrics_dict), pd.DataFrame(global_traces_dict), pd.DataFrame(individual_traces_dict)
