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
import glob
import multiprocessing as mp
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import matplotlib.lines
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from plotting.load_data_from_file import load_data_from_file
from simulation.simulationconstants import SimulationConstants
from tools import ProgressProcess
from trackobjects.trackside import TrackSide


def load_data(all_files, included_conditions, override_trial_numbers=False, show_tqdm=True):
    global_metrics_df_list = []
    global_traces_df_list = []
    individual_metrics_df_list = []
    individual_traces_df_list = []

    if show_tqdm:
        iterator = tqdm.tqdm(enumerate(all_files), total=len(all_files))
    else:
        iterator = enumerate(all_files)

    for index, file in iterator:
        global_metrics, individual_metrics, global_traces, individual_traces = load_data_from_file(index, file,
                                                                                                                     override_trial_numbers,
                                                                                                                     invert_scenarios=False,
                                                                                                                     included_conditions=included_conditions)

        global_metrics_df_list.append(global_metrics)
        global_traces_df_list.append(global_traces)
        individual_metrics_df_list.append(individual_metrics)
        individual_traces_df_list.append(individual_traces)

    global_metrics = pd.concat(global_metrics_df_list, ignore_index=True).convert_dtypes()
    individual_metrics = pd.concat(individual_metrics_df_list, ignore_index=True).convert_dtypes()
    global_traces = pd.concat(global_traces_df_list, ignore_index=True).convert_dtypes()
    individual_traces = pd.concat(individual_traces_df_list, ignore_index=True).convert_dtypes()

    return global_metrics, individual_metrics, global_traces, individual_traces


def load_experiment_data(conditions_to_consider):
    data_folder = '..\\data\\experiment_data\\'

    try:
        with open(data_folder + 'all_traces.pkl', 'rb') as f:
            all_global_traces = pickle.load(f)
        with open(data_folder + 'all_metrics.pkl', 'rb') as f:
            all_global_metrics = pickle.load(f)
        with open(data_folder + 'all_individual_traces.pkl', 'rb') as f:
            all_individual_traces = pickle.load(f)
        with open(data_folder + 'all_individual_metrics.pkl', 'rb') as f:
            all_individual_metrics = pickle.load(f)
    except FileNotFoundError:
        all_global_metrics = []
        all_global_traces = []
        all_individual_traces = []
        all_individual_metrics = []

        for experiment_number in [4, 5, 6, 8, 9, 10, 11, 12, 13]:
            all_experiment_iterations = glob.glob('..\\data\\experiment_data\\experiment_%d\\experiment_%d_iter_*.pkl' % (experiment_number, experiment_number))

            global_metrics, individual_metrics, global_traces, individual_traces = load_data(all_experiment_iterations, conditions_to_consider)
            all_global_metrics.append(global_metrics)
            all_global_traces.append(global_traces)
            all_individual_traces.append(individual_traces)
            all_individual_metrics.append(individual_metrics)

        all_global_traces = pd.concat(all_global_traces).reset_index()
        all_global_metrics = pd.concat(all_global_metrics).reset_index()
        all_individual_traces = pd.concat(all_individual_traces).reset_index()
        all_individual_metrics = pd.concat(all_individual_metrics).reset_index()

        with open(data_folder + 'all_traces.pkl', 'wb') as f:
            pickle.dump(all_global_traces, f)
        with open(data_folder + 'all_metrics.pkl', 'wb') as f:
            pickle.dump(all_global_metrics, f)
        with open(data_folder + 'all_individual_traces.pkl', 'wb') as f:
            pickle.dump(all_individual_traces, f)
        with open(data_folder + 'all_individual_metrics.pkl', 'wb') as f:
            pickle.dump(all_individual_metrics, f)

    return all_global_traces, all_global_metrics, all_individual_traces, all_individual_metrics


def load_data_with_multi_processing(all_files, included_conditions, invert_scenarios=False, override_trial_numbers=False, workers=8):
    if not all_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    manager = mp.Manager()
    progress_process = ProgressProcess(len(all_files), manager)
    progress_process.start()

    arguments = zip([r for r in range(len(all_files))], all_files, [override_trial_numbers] * len(all_files), [invert_scenarios] * len(all_files),
                    [included_conditions] * len(all_files), [progress_process.queue] * len(all_files))

    with mp.Pool(workers) as p:
        results = p.starmap(load_data_from_file, arguments)
    global_metrics_list, individual_metrics_list, global_traces_list, individual_traces_list = list(zip(*results))

    global_metrics = pd.concat(global_metrics_list, ignore_index=True).convert_dtypes()
    individual_metrics = pd.concat(individual_metrics_list, ignore_index=True).convert_dtypes()
    global_traces = pd.concat(global_traces_list, ignore_index=True).convert_dtypes()
    individual_traces = pd.concat(individual_traces_list, ignore_index=True).convert_dtypes()

    return global_metrics, individual_metrics, global_traces, individual_traces


def plot_global_metrics(global_metric_data, palette, conditions_to_consider):
    plt.figure()
    sns.countplot(data=global_metric_data, y="condition", hue='who went first', hue_order=[TrackSide.LEFT, TrackSide.RIGHT, 'Collision'],
                  order=['R_-4_-8', 'R_-4_0', 'R_-4_8', 'R_-2_8', 'R_0_8', 'N_0_0', 'L_0_-8', 'L_2_-8', 'L_4_-8', 'L_4_0', 'L_4_8'], orient='v')
    plt.title('First car at merge point')

    plt.figure()
    sns.boxplot(data=global_metric_data, x='condition', y='time_gap_at_merge', order=conditions_to_consider, palette=palette)
    plt.title('Time gap at merge point')

    plt.figure()
    sns.boxplot(x=global_metric_data['condition'], y=global_metric_data['nett crt'],
                order=conditions_to_consider, palette=palette)
    plt.ylim((0., 4.5))
    plt.ylabel('CRT [s]')
    plt.title('Time when conflict is resolved')

    plt.figure()
    sns.boxplot(x=global_metric_data['condition'], y=global_metric_data['time_gap_at_merge'].abs(), order=conditions_to_consider, palette=palette)
    plt.title('Absolute time gap at merge point')

    plt.figure()
    sns.boxplot(data=global_metric_data, x='condition', y='min_ttc_after_merge', order=conditions_to_consider, palette=palette)
    plt.title('Minimum positive ttc after merge point')

    plt.figure()
    sns.boxplot(data=global_metric_data, x='condition', y='headway at merge point', order=conditions_to_consider, palette=palette)
    plt.title('headway at merge point [m]')


def plot_global_traces(global_trace_data, palette):
    plt.figure()
    sns.lineplot(data=global_trace_data, x='time [s]', y='inverse ttc [1/s]', hue='condition', palette=palette)
    plt.hlines(0, 0, global_trace_data['time [s]'].max(), colors='lightgray', linestyles='dashed')
    plt.ylim((-2., 2.))
    plt.title('Inverse ttc traces for conditions')


def plot_all_headway_traces(global_trace_data, global_metrics_data, random_file, palette):
    fig, axes = plt.subplots(1, 1)
    with open(random_file, 'rb') as f:
        data = pickle.load(f)

    collision_bounds_in_axis(axes)

    axes.set_ylabel('headway [m] \n (positive is left ahead)')

    conflict_resolved_plot_points = {'condition': [],
                                     'average travelled distance [m]': [],
                                     'headway [m]': []}

    out_of_tunnel_plot_points = {'condition': [],
                                 'average travelled distance [m]': [],
                                 'headway [m]': []}

    for condition in global_metrics_data['condition'].unique():
        conflict_resolved_times = global_metrics_data.loc[global_metrics_data['condition'] == condition, ('conflict_resolved_time', 'trial_number')]
        condition_traces = global_trace_data.loc[global_trace_data['condition'] == condition, :]

        for trial_number in conflict_resolved_times['trial_number']:
            conflict_resolved_time = conflict_resolved_times.loc[conflict_resolved_times['trial_number'] == trial_number, 'conflict_resolved_time'].iat[0]

            if not pd.isnull(conflict_resolved_time):
                trial_traces = condition_traces.loc[condition_traces['trial_number'] == trial_number, :]
                conflict_resolved_data = trial_traces.loc[trial_traces['time [s]'] == conflict_resolved_time, :]

                conflict_resolved_plot_points['condition'].append(condition)
                conflict_resolved_plot_points['average travelled distance [m]'].append(conflict_resolved_data['average travelled distance [m]'].iat[0])
                conflict_resolved_plot_points['headway [m]'].append(conflict_resolved_data['headway [m]'].iat[0])

                global_trace_data.loc[global_trace_data['trial_number'] == trial_number, 'status'] = 'conflict'
                global_trace_data.loc[(global_trace_data['trial_number'] == trial_number) &
                                      (global_trace_data['time [s]'] > conflict_resolved_time), 'status'] = 'resolved'
                out_of_tunnel_time = global_metrics_data.loc[global_metrics_data['trial_number'] == trial_number, 'out_of_tunnel_time'].iat[0]
                global_trace_data.loc[(global_trace_data['trial_number'] == trial_number) &
                                      (global_trace_data['time [s]'] < out_of_tunnel_time), 'status'] = 'in tunnel'

                global_trace_data.loc[global_trace_data['trial_number'] == trial_number, 'conflict_is_resolved'] = \
                    global_trace_data.loc[global_trace_data['trial_number'] == trial_number, 'time [s]'] > conflict_resolved_time

        out_of_tunnel_index = round(
            global_metrics_data.loc[global_metrics_data['trial_number'] == trial_number, 'out_of_tunnel_time'].iat[0] / (data['dt'] / 1000))
        out_of_tunnel_plot_points['condition'].append(condition)
        out_of_tunnel_plot_points['average travelled distance [m]'].append(
            global_trace_data.loc[global_trace_data['trial_number'] == trial_number, 'average travelled distance [m]'].iat[out_of_tunnel_index])
        out_of_tunnel_plot_points['headway [m]'].append(
            global_trace_data.loc[global_trace_data['trial_number'] == trial_number, 'headway [m]'].iat[out_of_tunnel_index])

    global_trace_data['trial_number'] = global_trace_data['trial_number'].astype(object)

    sns.lineplot(data=global_trace_data,
                 x='average travelled distance [m]',
                 y='headway [m]',
                 hue='condition', units='trial_number',
                 size='status', sizes={'conflict': 1., 'resolved': 0.1, 'in tunnel': 0.1}, estimator=None, ax=axes,
                 palette=palette)
    sns.scatterplot(data=out_of_tunnel_plot_points, x='average travelled distance [m]', y='headway [m]', hue='condition', ax=axes,
                    palette=palette, marker=5)

    sns.scatterplot(data=conflict_resolved_plot_points, x='average travelled distance [m]', y='headway [m]', hue='condition', ax=axes,
                    palette=palette, marker='X')

    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()

    new_labels = []
    line_labels = []
    handle_dict = {}

    for handle, label in zip(legend_handles, legend_labels):
        if not isinstance(handle, mpl.collections.PathCollection):
            label = label.replace('_', ' ').capitalize()

            handle_dict[label] = handle

            if len(label) > 3:
                new_labels.append(label)
            else:
                line_labels.append(label)

    preferred_order = ['L4', 'L2', 'L0', 'L-2', 'L-4', 'R-4', 'R-2', 'R0', 'R2', 'R4']
    label_order_indices = []
    for line_label in line_labels:
        label_order_indices.append(preferred_order.index(line_label))

    line_labels = [l for _, l in sorted(zip(label_order_indices, line_labels))]
    new_labels = new_labels[0:1] + line_labels + new_labels[1:]

    new_handles = []
    for label in new_labels:
        new_handles.append(handle_dict[label])

    plt.legend(new_handles, new_labels)
    plt.xlim((40., 150.))


def plot_individual_metrics(individual_metric_data, conditions_to_consider):
    fig, axes = plt.subplots(1, 3, sharey=True)
    sns.boxplot(data=individual_metric_data, x='condition', y='raw input integral', order=conditions_to_consider, hue='side', ax=axes[0])
    sns.boxplot(data=individual_metric_data, x='condition', y='raw positive input integral', order=conditions_to_consider, hue='side', ax=axes[1])
    sns.boxplot(data=individual_metric_data, x='condition', y='raw negative input integral', order=conditions_to_consider, hue='side', ax=axes[2])
    fig.suptitle('Integrals of the raw user input per participant per condition')

    fig, axes = plt.subplots(1, 3, sharey=True)
    sns.boxplot(data=individual_metric_data, x='condition', y='acceleration integral', order=conditions_to_consider, hue='side', ax=axes[0])
    sns.boxplot(data=individual_metric_data, x='condition', y='positive acceleration integral', order=conditions_to_consider, hue='side', ax=axes[1])
    sns.boxplot(data=individual_metric_data, x='condition', y='negative acceleration integral', order=conditions_to_consider, hue='side', ax=axes[2])
    fig.suptitle('Integrals of acceleration per participant per condition')


def plot_individual_traces(individual_traces_data, palette):
    fig, axes = plt.subplots(2, 1)
    for index, side in enumerate(TrackSide):
        side = str(side)
        sns.lineplot(data=individual_traces_data.loc[individual_traces_data['side'] == side], x='time [s]', y='velocity [m/s]', hue='condition', ax=axes[index],
                     palette=palette)
        axes[index].title.set_text(side + ' velocity traces')

    fig, axes = plt.subplots(2, 1)
    individual_traces_data['trial_number'] = individual_traces_data['trial_number'].astype(object)
    for index, side in enumerate(TrackSide):
        side = str(side)
        sns.lineplot(data=individual_traces_data.loc[individual_traces_data['side'] == side], x='time [s]', y='velocity [m/s]', units='trial_number',
                     estimator=None, hue='condition', ax=axes[index], palette=palette)
        axes[index].title.set_text(side + ' velocity traces')

    fig, axes = plt.subplots(2, 1)
    for index, side in enumerate(TrackSide):
        side = str(side)
        sns.lineplot(data=individual_traces_data.loc[individual_traces_data['side'] == side], x='time [s]', y='input', hue='condition', ax=axes[index]
                     , palette=palette)
        axes[index].title.set_text(side + ' input traces')

    fig, axes = plt.subplots(2, 1)
    for index, side in enumerate(TrackSide):
        side = str(side)
        sns.lineplot(data=individual_traces_data.loc[individual_traces_data['side'] == side], x='time [s]', y='input', units='trial_number', estimator=None,
                     hue='condition', ax=axes[index], palette=palette)
        axes[index].title.set_text(side + ' input traces')


def collision_bounds_in_axis(ax):
    with open(os.path.join('..', 'data', 'headway_bounds.pkl'), 'rb') as f:
        headway_bounds = pickle.load(f)

    ax.plot(np.array(headway_bounds['average_travelled_distance'], dtype=object), np.array(headway_bounds['negative_headway_bound'], dtype=object),
            linestyle='dashed', c='lightgrey')
    ax.plot(np.array(headway_bounds['average_travelled_distance'], dtype=object), np.array(headway_bounds['positive_headway_bound'], dtype=object),
            linestyle='dashed', c='lightgrey')


def plot_spaghetti_paper(individual_traces_data, global_metrics, experiment_number_bold):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    for index, side in enumerate(TrackSide):
        side = str(side)
        data_mask = (individual_traces_data['side'] == side) & \
                    (individual_traces_data['condition'] == 'N_0_0') & \
                    (individual_traces_data['time [s]'] > 4.5) & \
                    (individual_traces_data['travelled distance [m]'] < 2 * 50 - 2)
        sizes = {}
        for experiment_number in individual_traces_data['experiment_number'].unique():
            sizes[experiment_number] = 3. if experiment_number == experiment_number_bold else 0.5

        individual_traces_data['trial_number'] = individual_traces_data['trial_number'].astype(str)

        sns.lineplot(data=individual_traces_data.loc[data_mask], x='time [s]', y='velocity [m/s]', units='trial_number', hue='experiment_number',
                     estimator=None, ax=axes[index], color='k', size='experiment_number', sizes=sizes, legend='full')

        axes[index].title.set_text(side.capitalize() + ' Driver')
        # axes[index].set_ylim(y_limits)
        # axes[index].set_ylim(y_limits)

        all_trials = individual_traces_data.loc[data_mask, 'trial_number'].unique()

        def _get_final_point(row):
            trace = individual_traces_data.loc[data_mask & (individual_traces_data['trial_number'] == row['trial_number']), :]
            row['velocity [m/s]'] = trace['velocity [m/s]'].iat[-1]
            row['time [s]'] = trace['time [s]'].iat[-1]

            return row

        outcomes = global_metrics.loc[global_metrics['condition'] == 'N_0_0', ['trial_number', 'who went first']]
        outcomes = outcomes.apply(_get_final_point, axis=1)

        palette = {TrackSide.LEFT: 'tab:blue',
                   TrackSide.RIGHT: 'tab:orange',
                   'Collision': 'tab:red'}
        marker_dict = {TrackSide.LEFT: '<',
                       TrackSide.RIGHT: '>',
                       'Collision': 'd'}
        sns.scatterplot(data=outcomes, x='time [s]', y='velocity [m/s]', hue='who went first', marker='o', s=20., palette=palette, ax=axes[index], legend=False,
                        zorder=10, style='who went first', markers=marker_dict)

        legend = axes[index].get_legend()
        lines = legend.get_lines()
        new_labels = ['Pair 1', 'Pair 2', 'Pair 3', 'Pair 4', 'Pair 5', 'Pair 6', 'Pair 7', 'Pair 8', 'Pair 9']
        axes[index].legend(lines, new_labels, title='Participant Pair', loc=3)

        handles = []
        labels = []
        for key, value in palette.items():
            handles.append(matplotlib.lines.Line2D([0], [0], marker='o', c=value, lw=0))
            label = str(key).capitalize() + (' first' if key != 'Collision' else '')
            labels.append(label)

        second_legend = Legend(axes[index], handles, labels, title='Trial Outcome', loc=1)
        axes[index].add_artist(second_legend)
        fig.tight_layout()


def plot_initial_response_vs_who_first(global_metric_data, individual_metric_data):
    condition_iterator = [global_metric_data['condition'].unique()]

    variable = 'initial nett acceleration'
    limits = (individual_metric_data[variable].min() * 0.9, individual_metric_data[variable].max() * 1.1)

    for condition_list in condition_iterator:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax in axes.flatten():
            ax.set_aspect(1)
            ax.plot(limits, limits, linestyle='dashed', color='grey')
            ax.hlines(0.0, *limits, linestyles='dashed', colors='lightgrey')
            ax.vlines(0.0, *limits, linestyles='dashed', colors='lightgrey')
            ax.set_xlim(limits)
            ax.set_ylim(limits)

        global_metric_plot_data = global_metric_data.loc[global_metric_data['condition'].isin(condition_list), :]
        individual_metric_plot_data = individual_metric_data.loc[individual_metric_data['condition'].isin(condition_list), :]

        left_initial_input = individual_metric_plot_data.loc[individual_metric_plot_data['side'] == 'left', ['trial_number', variable]]
        left_initial_input.rename(columns={variable: 'left ' + variable}, inplace=True)
        right_initial_input = individual_metric_plot_data.loc[individual_metric_plot_data['side'] == 'right', ['trial_number', variable]]
        right_initial_input.rename(columns={variable: 'right ' + variable}, inplace=True)

        plot_data = pd.merge(global_metric_plot_data.loc[:, ['who went first', 'trial_number']], left_initial_input, on='trial_number')
        plot_data = pd.merge(plot_data, right_initial_input, on='trial_number')

        who_went_first_palette = {TrackSide.LEFT: 'tab:blue',
                                  TrackSide.RIGHT: 'tab:orange',
                                  'Collision': 'tab:red'}
        marker_dict = {TrackSide.LEFT: '<',
                       TrackSide.RIGHT: '>',
                       'Collision': 'd'}

        who_went_first_legend_dict = {}
        sns.scatterplot(data=plot_data, x='left ' + variable, y='right ' + variable, hue='who went first', s=40, alpha=0.6,
                        ax=axes[0], palette=who_went_first_palette, style='who went first', markers=marker_dict, linewidth=0.1)
        sns.kdeplot(x=plot_data.loc[plot_data['who went first'] == TrackSide.LEFT, 'left ' + variable].to_numpy(),
                    y=plot_data.loc[plot_data['who went first'] == TrackSide.LEFT, 'right ' + variable].to_numpy(),
                    ax=axes[1], color=who_went_first_palette[TrackSide.LEFT], fill=True)
        sns.kdeplot(x=plot_data.loc[plot_data['who went first'] == TrackSide.RIGHT, 'left ' + variable].to_numpy(),
                    y=plot_data.loc[plot_data['who went first'] == TrackSide.RIGHT, 'right ' + variable].to_numpy(),
                    ax=axes[2], color=who_went_first_palette[TrackSide.RIGHT], fill=True)
        sns.kdeplot(x=plot_data.loc[plot_data['who went first'] == 'Collision', 'left ' + variable].to_numpy(),
                    y=plot_data.loc[plot_data['who went first'] == 'Collision', 'right ' + variable].to_numpy(),
                    ax=axes[3], color=who_went_first_palette['Collision'], fill=True)

        who_went_first_legend_dict = {}

        for label, ax in [(TrackSide.LEFT, axes[1]),
                          (TrackSide.RIGHT, axes[2]),
                          ('Collision', axes[3])]:
            who_went_first_legend_dict[label] = matplotlib.lines.Line2D([0], [0], lw=0, marker=marker_dict[label], c=who_went_first_palette[label])

        axes[0].legend(who_went_first_legend_dict.values(), who_went_first_legend_dict.keys(), title='Who Went First?', loc='center left',
                       bbox_to_anchor=(0.001, 0.15))

        axis_label = 'initial acceleration [m/s]'
        axes[0].set_xlabel('Left ' + axis_label)
        axes[0].set_ylabel('Right ' + axis_label)
        axes[0].set_title('All data, n=' + str(len(plot_data.loc[:, 'who went first'])))
        axes[1].set_xlabel('Left ' + axis_label)
        axes[1].set_ylabel('')
        axes[1].set_title('Left first, n=' + str(len(plot_data.loc[plot_data['who went first'] == TrackSide.LEFT, 'who went first'])))
        axes[2].set_xlabel('Left ' + axis_label)
        axes[2].set_ylabel('')
        axes[2].set_title('Right first, n=' + str(len(plot_data.loc[plot_data['who went first'] == TrackSide.RIGHT, 'who went first'])))
        axes[3].set_xlabel('Left ' + axis_label)
        axes[3].set_ylabel('')
        axes[3].set_title('Collisions, n=' + str(len(plot_data.loc[plot_data['who went first'] == 'Collision', 'who went first'])))

        plt.tight_layout()


if __name__ == '__main__':

    conditions_to_consider = ['R_-4_-8', 'R_-4_0', 'R_-4_8', 'R_-2_8', 'R_0_8', 'N_0_0', 'L_0_-8', 'L_2_-8', 'L_4_-8', 'L_4_0', 'L_4_8']
    color_dict = {'N_0_0': '#000000',
                  'L_0_-8': '#16193B',
                  'L_4_-8': '#ADD5F7',
                  'L_4_0': '#7FB2F0',
                  'L_2_-8': '#4E7AC7',
                  'L_4_8': '#35478C',
                  'R_-4_8': '#96ED89',
                  'R_0_8': '#00261C',
                  'R_-2_8': '#168039',
                  'R_-4_0': '#45BF55',
                  'R_-4_-8': '#044D29'}

    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50.,
                                               max_time=30e3)

    all_headway_figures = []
    all_v_rel_figures = []

    global_traces, global_metrics, individual_traces, individual_metrics = load_experiment_data(conditions_to_consider)

    # plot_global_metrics(global_metrics, color_dict, conditions_to_consider)

    # other plotting options below
    # -----------------------------------------
    plot_spaghetti_paper(individual_traces, global_metrics, experiment_number_bold='12')
    plot_initial_response_vs_who_first(global_metrics, individual_metrics)
    # plot_global_traces(global_traces, color_dict)
    # plot_individual_metrics(individual_metrics, conditions_to_consider)
    # plot_individual_traces(individual_traces, color_dict)

    # plot_all_headway_traces(global_traces, global_metrics, example_file, color_dict)

    plt.show()
