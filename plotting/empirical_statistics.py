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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymer4.models
import rpy2.robjects as robjects
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from pymer4.bridge import pandas2R, R2pandas, R2numpy
from rpy2.robjects.packages import importr
from scipy.stats._result_classes import BinomTestResult

from plotting.compare_conditions import load_experiment_data
from plotting.generate_html_plots import plot_logitic_plotly
from simulation.simulationconstants import SimulationConstants


def regression_on_merge_point_prediction(data, velocity_color_map, vehicle_color_map, use_matplotlib=True):
    headway = 'projected_headway'
    rel_v = 'relative_velocity'

    model_data = data.loc[:, ['who_first_int', 'experiment_number', headway, rel_v]]
    pymer4_model = pymer4.models.Lmer(data=model_data, formula="who_first_int ~ " + headway + " + " + rel_v + " + (1|experiment_number)",
                                      family='binomial')
    pymer4_model.fit(summary=False)
    print(pymer4_model.coefs.to_string())
    print(pymer4_model.fixef.to_string())

    if use_matplotlib:
        plot_logitic_mpl(data, headway, rel_v, pymer4_model, velocity_color_map, vehicle_color_map)
    else:
        plot_logitic_plotly(data, headway, rel_v, pymer4_model)
    return pymer4_model


def get_bootstrapped_confidence_interval_for_population(pymer4_model, new_data, number_of_simulations=200):
    importr("merTools")
    r_bootstrap_code = """
        function(model,new){
        
        mySumm <- function(.) {
            predict(.,new,allow.new.levels=TRUE,type='response')
        }         
        
        out <- bootMer(model, mySumm, nsim =""" + str(number_of_simulations) + """)
        out
        }
    """

    r_confidence_interval_code = """
        function(boostrapResults){
        out <- confint(boostrapResults,level=0.95)
        out
        }
    """

    new_data['experiment_number'] = np.NAN

    bootstrap_function = robjects.r(r_bootstrap_code)
    confidence_interval_function = robjects.r(r_confidence_interval_code)

    bootstrap_results = bootstrap_function(pymer4_model.model_obj, pandas2R(new_data))
    confidence_interval = confidence_interval_function(bootstrap_results)

    bootstrap_results_as_dict = {}
    for name, value in zip(bootstrap_results.names, bootstrap_results):
        bootstrap_results_as_dict[name] = R2pandas(value)

    confidence_dataframe = pd.DataFrame(R2numpy(confidence_interval), columns=confidence_interval.names[1])
    output_frame = new_data.join(confidence_dataframe)
    output_frame['mean_prediction'] = bootstrap_results_as_dict['t'].mean(axis=0)

    return output_frame


def _get_binomial_confidence_interval(row):
    results = BinomTestResult(row['sum'], row['samples'], alternative='two-sided', statistic=row['sum'] / row['samples'], pvalue=0.01).proportion_ci(
        confidence_level=0.95, method='wilson')
    row['ci_high'] = results.high
    row['ci_low'] = results.low
    return row


def plot_logitic_mpl(data, headway, rel_v, regression_results: pymer4.models.Lmer, velocity_color_map, vehicle_color_map):
    sum_data = data.groupby([headway, rel_v])['who_first_int'].sum()
    samples = data.groupby([headway, rel_v])['who_first_int'].count()
    percentage_data = sum_data / samples
    percentage_data = percentage_data.reset_index()
    percentage_data['sum'] = sum_data.to_numpy()
    percentage_data['samples'] = samples.to_numpy()
    percentage_data = percentage_data.apply(_get_binomial_confidence_interval, axis=1)

    # plot in 2D
    x = np.linspace(-4, 4, 100)
    y = np.array([-0.8, 0.0, 0.8])
    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, 300))

    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])
    conf_int_data = get_bootstrapped_confidence_interval_for_population(regression_results, as_df, number_of_simulations=20)

    plt.figure()
    palette = {0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'}
    sns.scatterplot(percentage_data, x=headway, y='who_first_int', hue=rel_v,
                    style=rel_v, s=300., linewidth=3,
                    palette=palette,
                    markers={0.0: '2', 0.8: '4', -0.8: '3'}, zorder=10)
    ci_palette = {0.0: '#7f7f7faa', 0.8: '#bcbd22aa', -0.8: '#17becfaa'}
    plt.vlines(percentage_data['projected_headway'], percentage_data['ci_low'], percentage_data['ci_high'], colors=percentage_data[rel_v].replace(ci_palette),
               linewidths=2., zorder=1)
    sns.lineplot(conf_int_data, x=headway, y='mean_prediction', hue=rel_v, palette=palette, zorder=9)
    for relative_velocity in [-0.8, 0.0, 0.8]:
        velocity_data = conf_int_data.loc[conf_int_data[rel_v] == relative_velocity]
        plt.fill_between(velocity_data[headway], velocity_data['2.5 %'], velocity_data['97.5 %'], color=palette[relative_velocity], zorder=2,
                         alpha=0.3)
    plt.xticks([-4, -2, 0, 2, 4])
    plt.yticks([0., 0.2, 0.4, 0.6, 0.8, 1.0], [0, 20, 40, 60, 80, 100])
    plt.xlabel('Projected headway advantage for the left vehicle [m]')
    plt.ylabel('Percentage of the left vehicle merging first')
    legend = plt.legend()

    texts = legend.get_texts()
    handles = legend.legendHandles

    plt.legend([(handles[0], handles[3]), (handles[1], handles[4]), (handles[2], handles[5])], [t.get_text() for t in texts[0:3]],
               title='Relative Velocity [m/s]', numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
    # plot in 3D
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-0.8, 0.8, 100)
    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, 10000))
    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])

    regression_prediction = np.array(regression_results.predict(as_df, verify_predictions=False, use_rfx=False)).reshape((100, 100)) * 100

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax_3 = fig.add_subplot(1, 3, 3, projection='3d')

    dx = 0.2
    dy = 0.0375

    for ax in [ax_1, ax_2, ax_3]:
        ax.plot_surface(X=grid[0], Y=grid[1], Z=regression_prediction, edgecolor='k', color='k', lw=0.5,
                        rstride=8,
                        cstride=8,
                        alpha=0.3, zorder=2)
        ax.bar3d(x=percentage_data[headway] - dx / 2., y=percentage_data[rel_v] - dy / 2., z=0., dx=dx, dy=dy,
                 dz=percentage_data['who_first_int'] * 100.,
                 color='tab:blue', alpha=0.5, zorder=1)
        ax.bar3d(x=percentage_data[headway] - dx / 2., y=percentage_data[rel_v] - dy / 2., z=percentage_data['who_first_int'] * 100., dx=dx, dy=dy,
                 dz=100 - percentage_data['who_first_int'] * 100.,
                 color='tab:orange', alpha=0.5, zorder=1)

        ax.contourf(X=grid[0], Y=grid[1], Z=regression_prediction, zdir='y', offset=1, cmap=velocity_color_map, alpha=1.)
        # ax.contourf(X=-grid[0], Y=grid[1], Z=regression_prediction, zdir='x', offset=4.5, cmap=color_map,
        #             alpha=1.)

        ax.set_xlabel('Projected Headway [m]')
        ax.set_ylabel('Relative Velocity [m/s]')
        ax.set_zlabel('Percentage left vehicle \n merging first')

    ax_1.view_init(elev=25., azim=-150.)
    ax_2.view_init(elev=12., azim=-115.)
    ax_3.view_init(elev=50., azim=-100.)

    colors = [mpl.lines.Line2D([0], [0], color='tab:orange', lw=4),
              mpl.lines.Line2D([0], [0], color='tab:blue', lw=4)]

    ax_2.legend(colors, ['Right vehicle merges first', 'Left vehicle merges first'])

    plt.tight_layout()

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-3, 3, 100)
    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, 10000))
    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])
    as_df[headway + ":" + rel_v] = as_df[headway] * as_df[rel_v]

    regression_prediction = np.array(regression_results.predict(as_df, verify_predictions=False, use_rfx=False)).reshape((100, 100)) * 100

    fig = plt.figure()
    plt.pcolormesh(grid[0], grid[1], regression_prediction, cmap=vehicle_color_map, alpha=0.5)

    vehicle_length = 4.5
    end_point = 10.  # used to plot lines that go out of frame

    plt.vlines(0, -end_point, end_point, colors='lightgray', linestyles='dashed', zorder=0.)
    plt.hlines(0, -end_point, end_point, colors='lightgray', linestyles='dashed', zorder=0.)

    plt.vlines(vehicle_length, -end_point, end_point, color='gray')
    plt.vlines(-vehicle_length, -end_point, end_point, color='gray')

    plt.fill_between([-vehicle_length, -end_point], [-end_point, -end_point], [end_point, end_point], color='lightgray')
    plt.fill_between([vehicle_length, end_point], [-end_point, -end_point], [end_point, end_point], color='lightgray')

    new_conditions = np.array(
        [[0., 0.], [0., -.8], [2., -.8], [4., -.8], [4., 0.], [4., 0.8], [0., .8], [-2., .8], [-4., .8], [-4., 0.], [-4., -0.8]])
    labels = ['0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8', '0_8', '-2_8', '-4_8', '-4_0', '-4_-8']
    plt.scatter(new_conditions[:, 0], new_conditions[:, 1], color='k', zorder=100., marker='s')

    plt.text(0.5, 0.9, 'Right vehicle merges first', horizontalalignment='center', verticalalignment='center', transform=fig.get_axes()[0].transAxes,
             bbox=dict(boxstyle="square,pad=0.5", fc="#EEEEEE", ec="#999999", lw=2))
    plt.text(0.5, 0.1, 'Left vehicle merges first', horizontalalignment='center', verticalalignment='center', transform=fig.get_axes()[0].transAxes,
             bbox=dict(boxstyle="square,pad=0.5", fc="#EEEEEE", ec="#999999", lw=2))

    fig.axes[0].set_aspect('equal')
    plt.xlim((-5.5, 5.5))
    plt.ylim((-3., 3.))
    plt.xlabel('projected $\\Delta x_l$ [m]', usetex=True)
    plt.ylabel('$\\Delta v_l$ [m/s]', usetex=True)

    _ = fig.colorbar(mpl.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap=vehicle_color_map), location="top", ax=fig.axes[0])
    for c in fig.axes[1].collections:
        c.set_alpha(0.5)
    fig.axes[1].set_xlabel('Percentage of left vehicle merging first \n Prediction of the logistic regression model')

    return fig


def two_d_linear_regression_on_crt(data):
    headway = 'projected_headway_crt'
    rel_v = 'relative_velocity_crt'

    model_data = data.loc[:, ['nett_crt', 'experiment_number', headway, rel_v]]
    pymer4_model = pymer4.models.Lmer(data=model_data, formula="nett_crt ~ " + headway + " * " + rel_v + " + (1|experiment_number)")
    pymer4_model.fit(summary=False)
    print(pymer4_model.coefs.to_string())
    print(pymer4_model.fixef.to_string())

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-0.8, 0.8, 100)
    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, 10000))
    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])
    as_df[headway + ":" + rel_v] = as_df[headway] * as_df[rel_v]
    as_df['experiment_number'] = np.NaN

    predictions = R2numpy(pymer4_model.predict(as_df, verify_predictions=False)).reshape((100, 100))

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax_3 = fig.add_subplot(1, 3, 3, projection='3d')

    for ax in [ax_1, ax_2, ax_3]:
        ax.plot_surface(X=grid[0], Y=grid[1], Z=predictions, edgecolor='k', color='k', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)

        grouped_data = data.groupby([headway, rel_v])['nett_crt']

        for (headway_value, rel_v_value) in grouped_data.indices.keys():
            crt_data = data.loc[(data[headway] == headway_value) & (data[rel_v] == rel_v_value), 'nett_crt'].to_numpy()
            plot_3d_box(headway_value, rel_v_value, crt_data, ax)

        if ax is not ax_1:
            ax.set_xlabel('Projected Headway [m]')
        if ax is not ax_3:
            ax.set_ylabel('Relative Velocity [m/s]')
        ax.set_zlabel('CRT [s]')

    ax_1.view_init(elev=0., azim=0.)
    ax_2.view_init(elev=30., azim=45.)
    ax_3.view_init(elev=0., azim=90.)

    plt.tight_layout()

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-3, 3, 100)

    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, 10000))
    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])
    as_df[headway + ":" + rel_v] = as_df[headway] * as_df[rel_v]
    as_df['experiment_number'] = np.NaN

    regression_prediction = R2numpy(pymer4_model.predict(as_df, verify_predictions=False)).reshape((100, 100))

    fig = plt.figure()
    plt.pcolormesh(grid[0], grid[1], regression_prediction, cmap='plasma')

    vehicle_length = 4.5
    end_point = 10.  # used to plot lines that go out of frame

    plt.vlines(0, -end_point, end_point, colors='lightgray', linestyles='dashed', zorder=0.)
    plt.hlines(0, -end_point, end_point, colors='lightgray', linestyles='dashed', zorder=0.)

    plt.vlines(vehicle_length, -end_point, end_point, color='gray')
    plt.vlines(-vehicle_length, -end_point, end_point, color='gray')

    plt.fill_between([-vehicle_length, -end_point], [-end_point, -end_point], [end_point, end_point], color='lightgray')
    plt.fill_between([vehicle_length, end_point], [-end_point, -end_point], [end_point, end_point], color='lightgray')

    new_conditions = np.array(
        [[0., 0.], [0., -.8], [2., -.8], [4., -.8], [4., 0.], [4., 0.8], [0., .8], [-2., .8], [-4., .8], [-4., 0.], [-4., -0.8]])
    labels = ['0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8', '0_8', '-2_8', '-4_8', '-4_0', '-4_-8']
    plt.scatter(new_conditions[:, 0], new_conditions[:, 1], color='k', zorder=100., marker='s')

    fig.axes[0].set_aspect('equal')
    plt.xlim((-5.5, 5.5))
    plt.ylim((-3., 3.))
    plt.xlabel('projected $\\Delta x_l$ [m]', usetex=True)
    plt.ylabel('$\\Delta v_l$ [m/s]', usetex=True)

    color_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=plt.Normalize(0, 5), cmap='plasma'), location="top", ax=fig.axes[0])
    fig.axes[1].set_xlabel('CRT  \n Prediction of the logistic regression model')

    x_coordinates = {-4: {-0.8: -0.25, 0.0: 0.0, 0.8: 0.25},
                     -2: {-0.8: 0.75, 0.0: 1.0, 0.8: 1.25},
                     0: {-0.8: 1.75, 0.0: 2.0, 0.8: 2.25},
                     2: {-0.8: 2.75, 0.0: 3.0, 0.8: 3.25},
                     4: {-0.8: 3.75, 0.0: 4.0, 0.8: 4.25}}

    x = np.array([-4., -2., 0., 2., 4.])
    y = np.array([-.8, 0., .8])

    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, 15))
    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])
    as_df[headway + ":" + rel_v] = as_df[headway] * as_df[rel_v]
    as_df['experiment_number'] = np.NaN
    as_df['x_coordinates'] = as_df.apply(lambda r: x_coordinates[r[headway]][r[rel_v]], axis=1)
    as_df = get_bootstrapped_confidence_interval_for_population(pymer4_model, as_df, number_of_simulations=200)

    mean_merge_time_data = {'condition': [],
                            'projected_headway': [],
                            'relative_velocity': [],
                            'mean_merge_time': []}

    for condition in data['condition'].unique():
        condition_data = data.loc[data['condition'] == condition, :]
        mean_merge_time_data['condition'].append(condition)
        mean_merge_time_data['projected_headway'].append(condition_data['projected_headway'].iat[0])
        mean_merge_time_data['relative_velocity'].append(condition_data['relative_velocity'].iat[0])
        mean_merge_time_data['mean_merge_time'].append((condition_data['merge_time'] - condition_data['out_of_tunnel_time']).mean())

    mean_merge_time_data = pd.DataFrame(mean_merge_time_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'}

    sns.boxplot(data=data, x='projected_headway', y='nett crt', hue='relative_velocity', dodge=True,
                palette=palette, ax=axes[0])
    sns.stripplot(data=mean_merge_time_data, x='projected_headway', y='mean_merge_time', hue='relative_velocity',
                  palette={0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'}, marker='$\_$', dodge=True, linewidth=0., s=20., legend=False, ax=axes[0])
    axes[0].legend([], [], frameon=False)

    sns.boxplot(data=data, x=headway, y='nett crt', hue=rel_v, dodge=True,
                palette=palette, ax=axes[1])
    sns.lineplot(data=as_df, x='x_coordinates', y='mean_prediction', hue=rel_v, palette=palette, ax=axes[1])
    for relative_velocity in [-0.8, 0.0, 0.8]:
        velocity_data = as_df.loc[as_df[rel_v] == relative_velocity]
        axes[1].fill_between(velocity_data['x_coordinates'], velocity_data['2.5 %'], velocity_data['97.5 %'], color=palette[relative_velocity], zorder=0,
                             alpha=0.3)
    for ax in axes:
        ax.set_xlabel('Projected Headway [m]')
        ax.set_ylabel('CRT [s]')

    old_legend = axes[1].get_legend()
    axes[1].legend(old_legend.legendHandles[0:3], [t.get_text() for t in old_legend.texts[0:3]], title='Relative Velocity [m/s]',
                   bbox_to_anchor=(-.07, 0.6))

    def index_to_label(index):
        return index

    def label_to_index(label):
        return label

    second_axis = axes[1].secondary_xaxis('top', functions=(index_to_label, label_to_index))

    ticks = []
    tick_labels = []
    data['x_coordinates'] = data.apply(lambda r: x_coordinates[r[headway]][r[rel_v]], axis=1)

    for tick in [-0.25, 0.0, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.75, 3.0, 3.25, 3.75, 4., 4.25]:
        count = data.loc[data['x_coordinates'] == tick, 'x_coordinates'].count()
        if count:
            label = 'n=%d' % count
            tick_labels.append(label)
            ticks.append(tick)

    second_axis.set_ticks(ticks, tick_labels, rotation=40, ha="left")
    plt.tight_layout()

    return fig


def plot_3d_box(x, y, z, ax, median_bar_height=0.01, box_width=0.5, box_depth=0.1, whisker_width=0.02, whisker_depth=0.004, color='tab:blue'):
    q0, q1, median_z, q3, q4 = np.percentile(z, [0, 25, 50, 75, 100])

    # median line
    ax.bar3d(x=x - (box_width * 1.2) / 2,
             y=y - (box_depth * 1.2) / 2,
             z=median_z - median_bar_height / 2.,
             dx=box_width * 1.2, dy=box_depth * 1.2, dz=median_bar_height,
             color='red')

    # whisker 1
    ax.bar3d(x=x - whisker_width / 2,
             y=y - whisker_depth / 2,
             z=q0,
             dx=whisker_width, dy=whisker_depth, dz=q1 - q0,
             color='k', alpha=0.5)

    # lower box
    ax.bar3d(x=x - box_width / 2,
             y=y - box_depth / 2,
             z=q1,
             dx=box_width, dy=box_depth, dz=median_z - median_bar_height / 2. - q1,
             color=color, alpha=0.5)

    # upper box
    ax.bar3d(x=x - box_width / 2,
             y=y - box_depth / 2,
             z=median_z + median_bar_height / 2.,
             dx=box_width, dy=box_depth, dz=q3 - median_z - median_bar_height / 2.,
             color=color, alpha=0.5)

    # whisker 1
    ax.bar3d(x=x - whisker_width / 2,
             y=y - whisker_depth / 2,
             z=q3,
             dx=whisker_width, dy=whisker_depth, dz=q4 - q3,
             color='k', alpha=0.5)


def _get_projected_headway(row):
    condition = str(row['condition'])
    elements = condition.split('_')
    return float(elements[1])


def _get_relative_velocity(row):
    condition = str(row['condition'])
    elements = condition.split('_')
    return float(elements[2]) / 10.


def _get_normalized_crt(row, minimum_crts):
    return row['nett crt'] - minimum_crts[row['condition']]


def _load_train_data():

    conditions_to_consider = ['R_-4_-8', 'R_-4_0', 'R_-4_8', 'R_-2_8', 'R_0_8', 'N_0_0', 'L_0_-8', 'L_2_-8', 'L_4_-8', 'L_4_0', 'L_4_8']
    # setup simulation constants
    simulation_constants = SimulationConstants(dt=50,
                                               vehicle_width=1.8,
                                               vehicle_length=4.5,
                                               track_start_point_distance=25.,
                                               track_section_length=50,
                                               max_time=30e3)

    _, global_metrics, _, individual_metrics = load_experiment_data(conditions_to_consider)
    global_metrics['who first str'] = global_metrics['who went first'].astype(str)
    global_metrics['nett_crt'] = global_metrics['nett crt'].astype(float)

    train_data = global_metrics.loc[global_metrics['who first str'] != 'Collision', :].copy()

    train_data.loc[:, 'who_first_int'] = (train_data['who went first'].apply(lambda v: -1 * v.value + 1)).astype(int)
    train_data.loc[:, 'projected_headway'] = train_data.apply(_get_projected_headway, axis=1)
    train_data.loc[:, 'relative_velocity'] = train_data.apply(_get_relative_velocity, axis=1)

    who_went_first_sign_data = train_data['who_first_int'] * 2 - 1

    train_data.loc[:, 'relative_velocity_crt'] = train_data.loc[:, 'relative_velocity'] * who_went_first_sign_data
    train_data.loc[:, 'projected_headway_crt'] = train_data.loc[:, 'projected_headway'] * who_went_first_sign_data

    velocity_cdict = {'red': [[0.0, 23 / 255, 23 / 255],
                              [0.5, 127 / 255, 127 / 255],
                              [1.0, 180 / 255, 180 / 255]],
                      'green': [[0.0, 190 / 255, 190 / 255],
                                [0.5, 127 / 255, 127 / 255],
                                [1.0, 189 / 255, 189 / 255]],
                      'blue': [[0.0, 207 / 255, 207 / 255],
                               [0.5, 127 / 255, 127 / 255],
                               [1.0, 34 / 255, 34 / 255]]}

    vehicle_cdict = {'red': [[0.0, 1.0, 1.0],
                             [0.5, 127 / 255, 127 / 255],
                             [1.0, 31 / 255, 31 / 255]],
                     'green': [[0.0, 127 / 255, 127 / 255],
                               [0.5, 127 / 255, 127 / 255],
                               [1.0, 118 / 255, 118 / 255]],
                     'blue': [[0.0, 14 / 255, 14 / 255],
                              [0.5, 127 / 255, 127 / 255],
                              [1.0, 180 / 255, 180 / 255]]}

    velocity_color_map = mpl.colors.LinearSegmentedColormap('velocity_cmap', segmentdata=velocity_cdict)
    vehicle_color_map = mpl.colors.LinearSegmentedColormap('vehicle_cmap', segmentdata=vehicle_cdict)

    return global_metrics, train_data, velocity_color_map, vehicle_color_map


if __name__ == '__main__':

    global_metrics, train_data, velocity_color_map, vehicle_color_map = _load_train_data()

    two_d_linear_regression_on_crt(train_data)
    fitted_model = regression_on_merge_point_prediction(train_data, velocity_color_map, vehicle_color_map, use_matplotlib=True)

    collisions_per_condition = {}

    for condition in train_data['condition'].unique():
        collisions_per_condition[condition] = global_metrics.loc[
            (global_metrics['condition'] == condition) & (global_metrics['end state'] == 'Collided'), 'end state'].count()

    print(collisions_per_condition)

    plt.show()
