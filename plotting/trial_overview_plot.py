import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.lines import Line2D

from conflict_signal import calculate_level_of_conflict_signal, determine_critical_points
from plotting.load_data_from_file import calculate_conflict_resolved_time, get_first, get_time_gap
from trackobjects.trackside import TrackSide


def plot_trial(data, plot_v_rel=False, plot_acc=False):
    freq = int(1000 / data['dt'])
    plot_risks = bool(data['perceived_risks'][TrackSide.LEFT] and data['perceived_risks'][TrackSide.RIGHT])

    figure = plt.figure(figsize=(8, 9.8))
    # figure.suptitle(title)

    time = [t * data['dt'] / 1000 for t in range(len(data['velocities'][TrackSide.LEFT]))]
    crt = calculate_conflict_resolved_time(data)

    try:
        crt_index = time.index(crt)
    except ValueError:
        crt_index = -1

    tunnel_y = data['track'].traveled_distance_to_coordinates(data['simulation_constants'].track_section_length, TrackSide.LEFT)[1]
    merge_point_y = data['track'].traveled_distance_to_coordinates(2 * data['simulation_constants'].track_section_length, TrackSide.LEFT)[1]

    tunnel_exit_index = {}
    merge_point_index = {}

    for side in TrackSide:
        tunnel_exit_index[side] = np.argmax(np.array(data['travelled_distance'][side]) >= data['simulation_constants'].track_section_length)
        merge_point_index[side] = np.argmax(np.array(data['travelled_distance'][side]) >= 2 * data['simulation_constants'].track_section_length)

    number_of_plots = 4 + sum([plot_acc, plot_v_rel, plot_risks])
    grid = plt.GridSpec(number_of_plots, 1, wspace=0.1, hspace=0.7)

    plot_number = 0

    pos_plot = plt.subplot(grid[plot_number, 0])
    plot_number += 1
    vel_plot = plt.subplot(grid[plot_number, 0])
    plot_number += 1

    if plot_acc:
        input_plot = plt.subplot(grid[plot_number, 0])
        plot_number += 1
    if plot_risks:
        risk_plot = plt.subplot(grid[plot_number, 0])
        plot_number += 1
    if plot_v_rel:
        rel_vel_plot = plt.subplot(grid[plot_number, 0], sharex=pos_plot)
        plot_number += 1

    gap_plot = plt.subplot(grid[plot_number, 0], sharex=pos_plot)
    plot_number += 1
    difficulty_plot = plt.subplot(grid[plot_number, 0], sharex=pos_plot)
    plot_number += 1

    plot_colors = {TrackSide.LEFT: 'b',
                   TrackSide.RIGHT: 'r'}

    x_position_limits = (40., 150.)
    x_time_limits = (4., 15.)

    average_travelled_distance_trace = (np.array(data['travelled_distance'][TrackSide.LEFT]) + np.array(data['travelled_distance'][TrackSide.RIGHT])) / 2.

    # Position plot
    lines = {}
    for side in TrackSide:
        positions = np.array(data['positions'][side])

        if side is TrackSide.LEFT:
            positions[:, 0] = positions[:, 0] - 2.
        else:
            positions[:, 0] = positions[:, 0] + 2.

        lines[side], = pos_plot.plot(positions[:, 1], -positions[:, 0], color=plot_colors[side])

        pos_plot.scatter(positions[crt_index, 1], -positions[crt_index, 0], label=str(side), c=plot_colors[side],
                         marker='X', s=80.)
        pos_plot.scatter(positions[tunnel_exit_index[side], 1], -positions[tunnel_exit_index[side], 0], c=plot_colors[side],
                         marker='>', s=80.)
        pos_plot.scatter(positions[merge_point_index[side], 1], -positions[merge_point_index[side], 0], c=plot_colors[side],
                         marker='s', s=80.)

    left_positions_per_second = np.array(data['positions'][TrackSide.LEFT][0::freq])
    left_positions_per_second[:, 0] = left_positions_per_second[:, 0] - 2.

    right_positions_per_second = np.array(data['positions'][TrackSide.RIGHT][0::freq])
    right_positions_per_second[:, 0] = right_positions_per_second[:, 0] + 2.

    for left_point, right_point in zip(left_positions_per_second, right_positions_per_second):
        pos_plot.plot([left_point[1], right_point[1]], [-left_point[0], -right_point[0]], c='lightgrey', linestyle='dashed')
        pos_plot.scatter([left_point[1], right_point[1]], [-left_point[0], -right_point[0]])

    y_bounds = (-1.2 * max(max(np.array(data['positions'][TrackSide.LEFT])[:, 0]), max(np.array(data['positions'][TrackSide.RIGHT])[:, 0])) - 2.,
                -1.2 * min(min(np.array(data['positions'][TrackSide.LEFT])[:, 0]), min(np.array(data['positions'][TrackSide.RIGHT])[:, 0])) + 2.)

    pos_plot.set_yticks([-12, -2, 2, 12])
    pos_plot.set_xlim(x_position_limits)
    pos_plot.set_yticklabels([10, 0, 0, -10])
    pos_plot.set_ylim(y_bounds)

    pos_plot.set_xlabel('Y position [m]')
    pos_plot.set_ylabel('X position [m]')
    lines['crt'] = Line2D([0], [0], c='k', marker='X', markersize=6., linestyle='none')
    lines['tunnel exit'] = Line2D([0], [0], c='k', marker='>', markersize=6., linestyle='none')
    lines['merge point'] = Line2D([0], [0], c='k', marker='s', markersize=6., linestyle='none')
    pos_plot.legend(lines.values(), lines.keys())

    # Velocity plot
    for side in TrackSide:
        vel_plot.plot(time, data['velocities'][side], label=str(side), c=plot_colors[side])
        vel_plot.scatter(time[crt_index], data['velocities'][side][crt_index], label=str(side), c=plot_colors[side],
                         marker='X', s=80.)

    for side in TrackSide:
        vel_plot.scatter(time[tunnel_exit_index[side]], data['velocities'][side][tunnel_exit_index[side]], c=plot_colors[side],
                         marker='>', s=80.)
        vel_plot.scatter(time[merge_point_index[side]], data['velocities'][side][merge_point_index[side]], c=plot_colors[side],
                         marker='s', s=80.)

    vel_plot.plot([time[tunnel_exit_index[TrackSide.LEFT]], time[tunnel_exit_index[TrackSide.RIGHT]]],
                  [data['velocities'][TrackSide.LEFT][tunnel_exit_index[TrackSide.LEFT]],
                   data['velocities'][TrackSide.RIGHT][tunnel_exit_index[TrackSide.RIGHT]]], linestyle='dashed', c='gray')
    vel_plot.plot([time[merge_point_index[TrackSide.LEFT]], time[merge_point_index[TrackSide.RIGHT]]],
                  [data['velocities'][TrackSide.LEFT][merge_point_index[TrackSide.LEFT]],
                   data['velocities'][TrackSide.RIGHT][merge_point_index[TrackSide.RIGHT]]], linestyle='dashed', c='gray')

    upper_y_bound = max(max(data['velocities'][TrackSide.RIGHT]) + 0.5, max(data['velocities'][TrackSide.LEFT]) + 0.5)
    lower_y_bound = min(min(data['velocities'][TrackSide.RIGHT]) - 0.5, min(data['velocities'][TrackSide.LEFT]) - 0.5)
    y_bounds = (lower_y_bound, upper_y_bound)

    vel_plot.set_ylabel('Velocity [m/s]')
    vel_plot.set_xlabel('time [s]')
    vel_plot.set_ylim(y_bounds)
    vel_plot.set_xlim(x_time_limits)

    for side in TrackSide:
        if data['is_replanning'][side]:
            upper_indices = np.array(data['is_replanning'][side]) == 1
            vel_plot.scatter(np.array(time)[upper_indices],
                             np.array(data['velocities'][side])[upper_indices],
                             marker='*', c=plot_colors[side])

            lower_indices = np.array(data['is_replanning'][side]) == -1
            vel_plot.scatter(np.array(time)[lower_indices],
                             np.array(data['velocities'][side])[lower_indices],
                             marker='o', c=plot_colors[side])

    # Input (acceleration) plot
    if plot_acc:
        true_acceleration = {}
        for side in TrackSide:
            jerk = np.gradient(data['accelerations'][side], data['dt'] / 1000)
            true_acceleration[side] = np.array(data['accelerations'][side]) - 0.005 * np.array(data['velocities'][side]) ** 2 - 0.5
            # input_plot.plot(np.array(data['positions'][side])[:, 1], data['accelerations'][side], label=str(side), c=plot_colors[side])
            input_plot.plot(time, true_acceleration[side], label=str(side), c=plot_colors[side])
            input_plot.scatter(time[crt_index], true_acceleration[side][crt_index], label=str(side), c=plot_colors[side],
                               marker='X', s=80.)
            # input_plot.plot(np.array(data['positions'][side])[:, 1], jerk, label=str(side), c=plot_colors[side])

        for side in TrackSide:
            input_plot.scatter(time[tunnel_exit_index[side]], true_acceleration[side][tunnel_exit_index[side]], c=plot_colors[side],
                               marker='>', s=80.)
            input_plot.scatter(time[merge_point_index[side]], true_acceleration[side][merge_point_index[side]], c=plot_colors[side],
                               marker='s', s=80.)

        y_bounds = (-2., 2.)

        input_plot.set_ylabel('Acceleration [m/s^2]')
        input_plot.set_xlabel('time [s]')
        input_plot.set_ylim(y_bounds)
        input_plot.set_xlim(x_time_limits)

    # risk plot
    if plot_risks:
        for side in TrackSide:
            risk_plot.plot(time, data['perceived_risks'][side], c=plot_colors[side])
            risk_plot.hlines(data['risk_bounds'][side], [0], [max(time)], linestyles='dashed', colors=plot_colors[side])

        risk_plot.set_xlabel('time [s]')
        risk_plot.set_ylabel('perceived risk')
        risk_plot.set_xlim(x_time_limits)

    # relative velocity plot
    if plot_v_rel:
        relative_velocity = np.array(data['velocities'][TrackSide.LEFT]) - np.array(data['velocities'][TrackSide.RIGHT])
        rel_vel_line, = rel_vel_plot.plot(average_travelled_distance_trace, relative_velocity, c='k')
        rel_vel_plot.scatter(average_travelled_distance_trace[crt_index], relative_velocity[crt_index], c='k', marker='X', s=80.)

        rel_vel_plot.vlines(tunnel_y, -5, 5, linestyles='dashed', linewidths=1., colors='grey')
        rel_vel_plot.vlines(merge_point_y, -5, 5, linestyles='dashed', linewidths=1., colors='grey')

        y_bounds = (-4., 4.)

        rel_vel_plot.set_ylabel('Relative Velocity [m/s]')
        rel_vel_plot.set_xlabel('Average traveled distance [m]')
        rel_vel_plot.set_ylim(y_bounds)
        rel_vel_plot.set_xlim(x_position_limits)

    # Gap plot
    gap_data = np.array(data['travelled_distance'][TrackSide.LEFT]) - np.array(data['travelled_distance'][TrackSide.RIGHT])

    gap_plot.plot(average_travelled_distance_trace[:crt_index], gap_data[:crt_index], c='k')
    gap_plot.plot(average_travelled_distance_trace[crt_index:], gap_data[crt_index:], c='k', linewidth=0.3)
    crt_marker = gap_plot.scatter(average_travelled_distance_trace[crt_index], gap_data[crt_index], c='k', marker='X', s=80.)

    for side in TrackSide:
        gap_plot.scatter(average_travelled_distance_trace[tunnel_exit_index[side]], gap_data[tunnel_exit_index[side]], c=plot_colors[side],
                         marker='>', s=80.)
        gap_plot.scatter(average_travelled_distance_trace[merge_point_index[side]], gap_data[merge_point_index[side]], c=plot_colors[side],
                         marker='s', s=80.)

    with open(os.path.join('data', 'headway_bounds.pkl'), 'rb') as f:
        headway_bounds = pickle.load(f)

    gap_plot.plot(np.array(headway_bounds['average_travelled_distance'], dtype=object), np.array(headway_bounds['negative_headway_bound'], dtype=object),
                  linestyle='dashed', c='grey')
    gap_plot.plot(np.array(headway_bounds['average_travelled_distance'], dtype=object), np.array(headway_bounds['positive_headway_bound'], dtype=object),
                  linestyle='dashed', c='grey')

    gap_plot.fill_between(headway_bounds['average_travelled_distance'], headway_bounds['negative_headway_bound'], headway_bounds['positive_headway_bound'],
                          color='lightgrey')
    gap_plot.text(110., -2, 'Collision area', verticalalignment='center', clip_on=True)
    gap_plot.text(110., 2, 'Collision area', verticalalignment='center', clip_on=True)
    gap_plot.set_ylabel('Headway [m]')
    gap_plot.set_xlabel('Average traveled distance [m]')

    lower_y_limit = min(0., min(gap_data) - 1.)
    upper_y_limit = max(0., max(gap_data) + 1.)

    gap_plot.set_ylim((lower_y_limit, upper_y_limit))
    gap_plot.set_xlim(x_position_limits)

    level_of_conflict = calculate_level_of_conflict_signal(average_travelled_distance_trace, gap_data,
                                                      determine_critical_points(headway_bounds, data['simulation_constants']))
    difficulty_plot.hlines(0.0, 0., max(headway_bounds['average_travelled_distance']), linestyles='dashed', colors='lightgray')

    chosen_solution = get_first(data)

    for side in TrackSide:
        if side == chosen_solution:
            difficulty_plot.plot(average_travelled_distance_trace, level_of_conflict[side], label=str(side) + ' first difficulty', c=plot_colors[side])
        else:
            difficulty_plot.plot(average_travelled_distance_trace[:crt_index], level_of_conflict[side][:crt_index], label=str(side) + ' first difficulty',
                                 c=plot_colors[side])
            difficulty_plot.plot(average_travelled_distance_trace[crt_index - 1:], level_of_conflict[side][crt_index - 1:], label=str(side) + ' first difficulty',
                                 c=plot_colors[side], linewidth=0.2)

        difficulty_plot.scatter(average_travelled_distance_trace[crt_index], level_of_conflict[side][crt_index], label=str(side), c=plot_colors[side],
                                marker='X', s=80.)
        difficulty_plot.scatter(average_travelled_distance_trace[tunnel_exit_index[side]], level_of_conflict[side][tunnel_exit_index[side]], c=plot_colors[side],
                                marker='>', s=80.)
        difficulty_plot.scatter(average_travelled_distance_trace[merge_point_index[side]], level_of_conflict[side][merge_point_index[side]], c=plot_colors[side],
                                marker='s', s=80.)

    difficulty_plot.set_ylabel('Level of conflict [-]')
    difficulty_plot.set_xlabel('Average traveled distance [m]')

    return figure


if __name__ == '__main__':
    os.chdir(os.getcwd() + '\\..')

    experiment_number = 4
    iteration_number = 7

    path_data = os.path.join('data', 'experiment_data', 'experiment_%d' % experiment_number, 'experiment_%d_iter_%d.pkl' % (experiment_number, iteration_number))

    with open(path_data, 'rb') as f:
        loaded_data = pickle.load(f)
        print(loaded_data['current_condition'].name)
        figure = plot_trial(loaded_data)

    # show results
    plt.show()
