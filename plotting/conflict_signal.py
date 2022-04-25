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

import matplotlib.pyplot as plt
import numpy as np

from experiment.conditiondefinition import ConditionDefinition
from simulation.simulationconstants import SimulationConstants
from trackobjects.trackside import TrackSide


def determine_critical_points(headway_bounds_dict, simulation_constants):
    index_of_collision_threshold = np.where(np.array(headway_bounds_dict['positive_headway_bound'], dtype=float) >= simulation_constants.vehicle_length)[0][0]
    index_of_merge_point = np.where(np.array(headway_bounds_dict['average_travelled_distance'], dtype=float) >=
                                    2 * simulation_constants.track_section_length)[0][0]

    left_first_threshold = np.array(
        [headway_bounds_dict['average_travelled_distance'][index_of_collision_threshold],
         headway_bounds_dict['positive_headway_bound'][index_of_collision_threshold]])
    right_first_threshold = np.array(
        [headway_bounds_dict['average_travelled_distance'][index_of_collision_threshold],
         headway_bounds_dict['negative_headway_bound'][index_of_collision_threshold]])

    left_first_merge_point = np.array(
        [headway_bounds_dict['average_travelled_distance'][index_of_merge_point], headway_bounds_dict['positive_headway_bound'][index_of_merge_point]])
    right_first_merge_point = np.array(
        [headway_bounds_dict['average_travelled_distance'][index_of_merge_point], headway_bounds_dict['negative_headway_bound'][index_of_merge_point]])

    left_first_end_point = np.array([headway_bounds_dict['average_travelled_distance'][-1], headway_bounds_dict['positive_headway_bound'][-1]])
    right_first_end_point = np.array([headway_bounds_dict['average_travelled_distance'][-1], headway_bounds_dict['negative_headway_bound'][-1]])

    critical_points = {TrackSide.LEFT: [left_first_threshold, left_first_merge_point, left_first_end_point],
                       TrackSide.RIGHT: [right_first_threshold, right_first_merge_point, right_first_end_point]}

    return critical_points


def calculate_conflict_at_tunnel_exit(condition: ConditionDefinition, simulation_constants: SimulationConstants):
    left_tunnel_exit_time = (simulation_constants.track_section_length - condition.left_initial_position_offset) / condition.left_initial_velocity
    right_tunnel_exit_time = (simulation_constants.track_section_length - condition.right_initial_position_offset) / condition.right_initial_velocity

    tunnel_exit_time = max(left_tunnel_exit_time, right_tunnel_exit_time)

    left_position_at_tunnel_exit = condition.left_initial_position_offset + condition.left_initial_velocity * tunnel_exit_time
    right_position_at_tunnel_exit = condition.right_initial_position_offset + condition.right_initial_velocity * tunnel_exit_time

    average_travelled_distance_at_tunnel_exit = (left_position_at_tunnel_exit + right_position_at_tunnel_exit) / 2.
    head_way_at_tunnel_exit = left_position_at_tunnel_exit - right_position_at_tunnel_exit

    point = (average_travelled_distance_at_tunnel_exit, head_way_at_tunnel_exit)
    gradient = (condition.left_initial_velocity - condition.right_initial_velocity) / (
                0.5 * (condition.left_initial_velocity + condition.right_initial_velocity))

    max_headway_angle = np.arctan(2)
    path_to_saved_dict = os.path.join('..', 'data', 'headway_bounds.pkl')
    with open(path_to_saved_dict, 'rb') as f:
        headway_bounds = pickle.load(f)

    critical_points = determine_critical_points(headway_bounds, simulation_constants)

    level_of_conflict_dict = {}

    for side in TrackSide:
        difficulties_per_critical_point = []
        for critical_point in critical_points[side]:
            vector_to_critical_point = critical_point - point
            solve_angle = np.arctan2(vector_to_critical_point[1], vector_to_critical_point[0])
            current_angle = np.arctan(gradient)

            difficulty = solve_angle - current_angle
            if side is TrackSide.RIGHT:
                difficulty *= -1.

            normalized_difficulty = difficulty / max_headway_angle
            difficulties_per_critical_point.append(np.clip(normalized_difficulty, -1.0, 1.0))

        level_of_conflict_dict[side] = max(difficulties_per_critical_point)

    return level_of_conflict_dict


def calculate_level_of_conflict_signal(average_travelled_distance_trace, head_way_trace, critical_points):
    gradients = np.gradient(head_way_trace, average_travelled_distance_trace)
    data_points = np.stack([average_travelled_distance_trace, head_way_trace]).T

    """the Maximum angle is based on the concept that one of the two vehicles remains stationary. The headway can then increase or decrease precisely twice as 
    fast as the average travelled distance"""
    max_headway_angle = np.arctan(2)

    level_of_conflict = {TrackSide.LEFT: np.zeros(head_way_trace.shape),
                         TrackSide.RIGHT: np.zeros(head_way_trace.shape)}

    for index, (point, gradient) in enumerate(zip(data_points, gradients)):
        for side in TrackSide:
            difficulties = []
            for critical_point in critical_points[side]:
                vector_to_critical_point = critical_point - point
                solve_angle = np.arctan2(vector_to_critical_point[1], vector_to_critical_point[0])
                current_angle = np.arctan(gradient)

                difficulty = solve_angle - current_angle
                if side is TrackSide.RIGHT:
                    difficulty *= -1.

                normalized_difficulty = difficulty / max_headway_angle
                difficulties.append(np.clip(normalized_difficulty, -1.0, 1.0))

            level_of_conflict[side][index] = max(difficulties)

    return level_of_conflict


if __name__ == '__main__':
    experiment_number = 4
    iteration = 1

    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50,
                                               max_time=30e3)

    path_to_saved_dict = os.path.join('..', 'data', 'headway_bounds.pkl')
    with open(path_to_saved_dict, 'rb') as f:
        headway_bounds = pickle.load(f)

    with open(os.path.join('..', 'data', 'experiment_data', 'experiment_%d' % experiment_number, 'experiment_%d_iter_%d.pkl' % (experiment_number, iteration)),
              'rb') as f:
        loaded_data = pickle.load(f)

    average_travelled_distance_trace = ((np.array(loaded_data['travelled_distance'][TrackSide.LEFT]) +
                                         np.array(loaded_data['travelled_distance'][TrackSide.RIGHT])) / 2.)

    head_way_trace = np.array(loaded_data['travelled_distance'][TrackSide.LEFT]) - np.array(loaded_data['travelled_distance'][TrackSide.RIGHT])

    critical_points = determine_critical_points(headway_bounds, simulation_constants)
    level_of_conflict = calculate_level_of_conflict_signal(average_travelled_distance_trace, head_way_trace, critical_points)

    v_l = np.array(loaded_data['velocities'][TrackSide.LEFT])
    v_r = np.array(loaded_data['velocities'][TrackSide.RIGHT])
    d_l = np.array(loaded_data['travelled_distance'][TrackSide.LEFT])
    d_r = np.array(loaded_data['travelled_distance'][TrackSide.RIGHT])

    projected_gap = np.abs(2 * (100 - (200 - d_l + (v_l / v_r) * d_r) / ((v_l / v_r) + 1))) - simulation_constants.vehicle_length
    projected_time_gap = projected_gap / (v_l - v_r)

    projected_time_gap = projected_time_gap[average_travelled_distance_trace < 100.]
    # --------------------------- Plotting ---------------------------------
    fig_1 = plt.figure()
    fig_2 = plt.figure()

    raw_headway_axes = fig_1.add_subplot(1, 1, 1)
    raw_headway_axes.set_aspect('equal')

    check_point_axes = fig_2.add_subplot(1, 1, 1)
    check_point_axes.set_aspect('equal')

    for axes in [raw_headway_axes, check_point_axes]:
        axes.plot(headway_bounds['average_travelled_distance'], headway_bounds['positive_headway_bound'], c='gray')
        axes.plot(headway_bounds['average_travelled_distance'], headway_bounds['negative_headway_bound'], c='gray')

        axes.fill_between(np.array(headway_bounds['average_travelled_distance'], dtype=float),
                          np.array(headway_bounds['positive_headway_bound'], dtype=float),
                          np.array(headway_bounds['negative_headway_bound'], dtype=float),
                          color='lightgray')
        axes.text(110., 0., 'Collision area', verticalalignment='center', clip_on=True)

        axes.plot(average_travelled_distance_trace, head_way_trace, color='k')
        axes.set_ylim((-15, 15))

    raw_headway_axes.vlines([50, 100], 20, -20, linestyles='dashed', colors='lightgray', zorder=0.)
    raw_headway_axes.text(25., 11., 'Tunnel', verticalalignment='center', horizontalalignment='center', clip_on=True)
    raw_headway_axes.text(75., 11., 'Approach', verticalalignment='center', horizontalalignment='center', clip_on=True)
    raw_headway_axes.text(125., 11., 'Car Following', verticalalignment='center', horizontalalignment='center', clip_on=True)

    merge_point_influence_mask = headway_bounds['average_travelled_distance'] <= 102.25

    raw_headway_axes.fill_between(headway_bounds['average_travelled_distance'][merge_point_influence_mask],
                                  headway_bounds['positive_headway_bound'][merge_point_influence_mask],
                                  headway_bounds['negative_headway_bound'][merge_point_influence_mask],
                                  color='gray')

    all_critical_points = np.stack(critical_points[TrackSide.LEFT] + critical_points[TrackSide.RIGHT])
    slope_trace = np.gradient(head_way_trace, average_travelled_distance_trace)

    check_point_axes.scatter(all_critical_points[:, 0], all_critical_points[:, 1], marker='o', color='tab:orange')
    check_point_axes.vlines(all_critical_points[:, 0], -20, 20, colors='tab:orange', zorder=0)

    current_frame = 80
    check_point_axes.scatter([average_travelled_distance_trace[current_frame]], [head_way_trace[current_frame]], marker='d', c='tab:blue', s=60, zorder=3.)
    tangent_line_slope = slope_trace[current_frame]
    tangent_line_intersect = head_way_trace[current_frame] - tangent_line_slope * average_travelled_distance_trace[current_frame]

    check_point_axes.arrow(average_travelled_distance_trace[current_frame], head_way_trace[current_frame],
                           30., 30. * tangent_line_slope, color='tab:blue',
                           width=.5, head_width=2., zorder=3.)

    for side in TrackSide:
        difficulty = level_of_conflict[side][current_frame]
        if side is TrackSide.RIGHT:
            difficulty *= -1

        current_slope = np.tan(difficulty * np.arctan(2) + np.arctan(slope_trace[current_frame]))
        intersect = head_way_trace[current_frame] - current_slope * average_travelled_distance_trace[current_frame]

        check_point_axes.plot([average_travelled_distance_trace[current_frame],
                               average_travelled_distance_trace[current_frame] + 200],
                              [average_travelled_distance_trace[current_frame] * current_slope + intersect,
                               (average_travelled_distance_trace[current_frame] + 200) * current_slope + intersect],
                              c='tab:green', zorder=0)

    raw_headway_axes.set_xlabel('average traveled distance [m]')
    raw_headway_axes.set_ylabel('headway [m]')
    check_point_axes.set_xlabel('average traveled distance [m]')
    check_point_axes.set_ylabel('headway [m]')

    raw_headway_axes.set_xlim((0., 160))
    check_point_axes.set_xlim((0., 160))

    plt.show()
