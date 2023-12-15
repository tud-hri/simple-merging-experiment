import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from plotting.compare_conditions import load_experiment_data
from plotting.generate_html_plots import rename_to_paper_conventions
from plotting.empirical_statistics import _get_binomial_confidence_interval
from trackobjects.trackside import TrackSide


def _get_projected_headway_from_condition(row):
    condition = row['condition']
    headway = float(condition.split('_')[0])
    return headway


def _get_relative_velocity_from_condition(row):
    condition = row['condition']
    relative_velocity = float(condition.split('_')[1]) / 10
    return relative_velocity


def plot_velocity_traces(trace_data, global_metric_data, condition, highlighted_pair):
    plot_data = pd.merge(trace_data, global_metric_data[['who went first', 'trial_number']], how='inner', on='trial_number')
    plot_data = plot_data.loc[(plot_data['condition'] == condition) &
                              (plot_data['who went first'] != 'Collision'), :]

    plot_data['ego first'] = (plot_data['side'] == plot_data['who went first'].astype(str))
    plot_data['ego first'] = plot_data['ego first'].astype(str)

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, right=0.99)

    for side_index, action_side in enumerate(TrackSide):
        for first_index, first_bool in enumerate(['True', 'False']):
            for sim_index, is_simulation in enumerate([True, False]):
                sns.lineplot(plot_data.loc[(plot_data['side'] == str(action_side)) &
                                           (plot_data['ego first'] == first_bool) &
                                           (plot_data['is_simulation'] == is_simulation), :],
                             x='time [s]',
                             y='velocity [m/s]',
                             units='trial_number',
                             estimator=None,
                             color='gray',
                             linewidth=0.3,
                             ax=axes[sim_index, 2 * side_index + first_index])

    # overlay highlight

    for side_index, side in enumerate(TrackSide):
        for first_index, first_bool in enumerate(['True', 'False']):
            for sim_index, is_simulation in enumerate([True, False]):
                if is_simulation:
                    highlight_color = 'tab:orange'
                else:
                    highlight_color = 'tab:blue'

                sns.lineplot(plot_data.loc[(plot_data['side'] == str(side)) &
                                           (plot_data['is_simulation'] == is_simulation) &
                                           (plot_data['ego first'] == first_bool) &
                                           (plot_data['experiment_number'] == highlighted_pair), :],
                             x='time [s]',
                             y='velocity [m/s]',
                             units='trial_number',
                             estimator=None,
                             color=highlight_color,
                             linewidth=2.,
                             legend=None,
                             ax=axes[sim_index, 2 * side_index + first_index])

    axes[0, 0].set_xlim((4.5, 15.0))
    axes[0, 0].set_ylim((5., 14.0))

    axes[0, 0].set_title('Going First')
    axes[0, 1].set_title('Going Second')
    axes[0, 2].set_title('Going First')
    axes[0, 3].set_title('Going Second')

    plt.gcf().text(0.01, 0.7, 'Model', fontsize=14)
    plt.gcf().text(0.01, 0.3, 'Human', fontsize=14)

    plt.gcf().text(0.3, 0.95, 'Left Driver', fontsize=14)
    plt.gcf().text(0.75, 0.95, 'Right Driver', fontsize=14)

    for ax in np.array(axes).flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.tick_params(axis='both', which='both', length=0)
        # ax.patch.set_alpha(0.)
        # ax.set_xticks([0, 5])
    # plt.tight_layout()


def plot_deviation_from_v_medians(individual_metrics_data):
    plot_data = {'condition': [],
                 'sim': [],
                 'pair': [],
                 'side': [],
                 'mean': []}

    for experiment_number in individual_metrics_data['experiment_number'].unique():
        for sim in [True, False]:
            for side in TrackSide:
                data = individual_metrics_data.loc[(individual_metrics_data['experiment_number'] == experiment_number) &
                                                   (individual_metrics_data['side'] == str(side)) &
                                                   (individual_metrics_data['is_simulation'] == sim) &
                                                   (individual_metrics_data['who went first'] != 'Collision'),
                                                   'max abs deviation from desired v']

                plot_data['condition'].append('all')
                plot_data['sim'].append(sim)
                plot_data['pair'].append(experiment_number)
                plot_data['side'].append(str(side))
                plot_data['mean'].append(data.mean())

                for condition in individual_metrics_data['condition'].unique():
                    data = individual_metrics_data.loc[(individual_metrics_data['condition'] == condition) &
                                                       (individual_metrics_data['experiment_number'] == experiment_number) &
                                                       (individual_metrics_data['side'] == str(side)) &
                                                       (individual_metrics_data['is_simulation'] == sim) &
                                                       (individual_metrics_data['who went first'] != 'Collision'),
                                                       'max abs deviation from desired v']

                    plot_data['condition'].append(condition)
                    plot_data['sim'].append(sim)
                    plot_data['pair'].append(experiment_number)
                    plot_data['side'].append(str(side))
                    plot_data['mean'].append(data.mean())

    plot_data = pd.DataFrame(plot_data)

    y_max = plot_data['mean'].max() * 1.1
    order = {'left': ['-4_-8', '-4_0', '-4_8', '-2_8', '0_8', '0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8', 'all'],
             'right': ['all', '-4_-8', '-4_0', '-4_8', '-2_8', '0_8', '0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8']}

    fig, axes = plt.subplots(2, 9, sharex=True, sharey='row', figsize=(15, 5))
    plt.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.92, top=0.9, bottom=0.1)

    for row_index in range(2):
        for condition_index in range(12):
            con = mpl.patches.ConnectionPatch(xyA=(0., condition_index), coordsA=axes[row_index, 0].transData,
                                              xyB=(6., condition_index), coordsB=axes[row_index, 8].transData,
                                              zorder=0., linewidth=0.5, color='lightgray')
            fig.add_artist(con)

    for column_index in range(9):
        con = mpl.patches.ConnectionPatch(xyA=(0., 11.), coordsA=axes[1, column_index].transData,
                                          xyB=(0., 0.), coordsB=axes[0, column_index].transData,
                                          zorder=0., linewidth=0.5)
        fig.add_artist(con)

        con = mpl.patches.ConnectionPatch(xyA=(5., 11.), coordsA=axes[1, column_index].transData,
                                          xyB=(5., 0.), coordsB=axes[0, column_index].transData,
                                          zorder=0., linewidth=0.5, ls='dashed', color='lightgray')
        fig.add_artist(con)

    for pair_index, pair in enumerate(plot_data['pair'].unique()):
        pair_data = plot_data.loc[plot_data['pair'] == pair, :]
        for side_index, side in enumerate(TrackSide):
            selected_data = pair_data.loc[pair_data['side'] == str(side), :]
            ax = axes[side_index, pair_index]
            sns.stripplot(data=selected_data.loc[selected_data['sim']], y='condition', x='mean', ax=ax, color='tab:orange',
                          marker='o', dodge=False, jitter=0., edgecolor=None, linewidth=1., order=order[str(side)])
            sns.stripplot(data=selected_data.loc[~selected_data['sim']], y='condition', x='mean', ax=ax, color='tab:blue',
                          marker='s', dodge=False, jitter=0., edgecolor=None, linewidth=1., order=order[str(side)])

            ax.hlines(np.array(range(11)) + side_index, -1, 6.5, linewidth=1., colors='k')
            ax.set_xlim(0, y_max)
            ax.set_ylabel('')
            ax.set_xlabel('')

    pair_3_data = individual_metrics_data.loc[individual_metrics_data['experiment_number'] == '3', :]
    for side_index, side in enumerate(TrackSide):
        ax = axes[side_index, 2]
        side_data = pair_3_data.loc[pair_3_data['side'] == str(side), :]
        sns.stripplot(data=side_data, y='condition', x='max abs deviation from desired v', ax=ax, hue='is_simulation', order=order[str(side)], dodge=True,
                      marker='d', s=5, jitter=.25, alpha=0.5, zorder=0, palette={True: 'tab:orange', False: 'tab:blue'})
        ax.set_xlabel('')

    for ax in np.array(axes).flatten():
        ax.legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.patch.set_alpha(0.)
        ax.set_xticks([0, 5])
    # plt.tight_layout()

    fig.text(0.02, 0.92, "Participant pair: ")
    for i in range(1, 10):
        fig.text(0.04 + 0.093 * i, 0.92, str(i))

    fig.text(0.02, 0.7, "Left Driver")
    fig.text(0.02, 0.275, "Right Driver")

    axes[1, 4].set_xlabel('Absolute maximum deviation from initial velocity [m/s]')
    human_marker = mpl.lines.Line2D(linewidth=0., marker='s', color='tab:blue', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(linewidth=0., marker='o', color='tab:orange', xdata=[0], ydata=[0])
    legend = axes[0, 8].legend([human_marker, model_marker], ['Human', 'Model'], loc='center left', bbox_to_anchor=(1, 0.5))

    plot_data['coordinates'] = plot_data.apply(lambda r: 'y' if r['sim'] else 'x', axis=1)
    plot_data['id'] = plot_data['condition'] + '-' + plot_data['pair'] + '-' + plot_data['side']
    r_squared_data = plot_data.pivot(index='id', columns='coordinates', values='mean')

    model = smf.ols('y ~ x', data=r_squared_data).fit()
    print(model.summary())
    print(model.pvalues)

    x_max = 5.5

    plt.figure()
    sns.scatterplot(r_squared_data, x='x', y='y', color='gray')
    plt.xlim((0, x_max))
    plt.ylim((0, x_max))
    x_y_line, = plt.plot([0, x_max], [0, x_max], color='lightgrey', linestyle='dashed')
    regression_line, = plt.plot([0, x_max], [model.params['Intercept'], model.params['Intercept'] + x_max * model.params['x']], color='k', linestyle='-.')
    plt.gca().set_aspect('equal')
    plt.title('Absolute maximum deviation from initial velocity [m/s]')
    plt.ylabel('Model')
    plt.xlabel('Human')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    residual_sum_of_squares = ((r_squared_data['y'] - r_squared_data['x']) ** 2).sum()
    total_sum_of_squares = ((r_squared_data['y'] - r_squared_data['y'].mean()) ** 2).sum()
    r_squared = 1 - residual_sum_of_squares / total_sum_of_squares

    print(r_squared)

    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])
    plt.legend([x_y_line, none_marker, regression_line, none_marker, none_marker],
               ['x = y', '$R^{2}$: %.1f' % r_squared, 'Ordinary least-squares \n linear regression fit',
                'Intercept: %.2f' % model.params['Intercept'], 'Effect: %.2f' % model.params['x']])

    individual_metrics_data.loc[:, 'projected_headway'] = individual_metrics_data.apply(_get_projected_headway_from_condition, axis=1)
    individual_metrics_data.loc[:, 'relative_velocity'] = individual_metrics_data.apply(_get_relative_velocity_from_condition, axis=1)

    median_data = {'median': [],
                   'lower_iqr': [],
                   'upper_iqr': [],
                   'headway': [],
                   'dv': [],
                   'is_sim': []}

    regression_data = {'abs_v_deviation': [],
                       'headway': [],
                       'dv': [],
                       'driver_tag': [],
                       'is_sim': []}

    individual_metrics_data['driver_tag'] = individual_metrics_data['experiment_number'] + '-' + individual_metrics_data['side']
    for is_sim in [True, False]:
        for headway in [-4., -2, 0., 2., 4.]:
            for dv in [-0.8, 0., 0.8]:
                if (headway, dv) not in [(2., 0.8), (2., 0.), (-2., -0.8), (-2., 0.)]:
                    left_data = individual_metrics_data.loc[(individual_metrics_data['projected_headway'] == headway) &
                                                            (individual_metrics_data['relative_velocity'] == dv) &
                                                            (individual_metrics_data['is_simulation'] == is_sim) &
                                                            (individual_metrics_data['side'] == 'left'), ['max abs deviation from desired v',
                                                                                                          'driver_tag']]
                    right_data = individual_metrics_data.loc[(individual_metrics_data['projected_headway'] == -headway) &
                                                             (individual_metrics_data['relative_velocity'] == -dv) &
                                                             (individual_metrics_data['is_simulation'] == is_sim) &
                                                             (individual_metrics_data['side'] == 'right'), ['max abs deviation from desired v',
                                                                                                            'driver_tag']]

                    v_data_points = pd.concat([left_data['max abs deviation from desired v'],
                                               right_data['max abs deviation from desired v']])
                    driver_tags = pd.concat([left_data['driver_tag'],
                                             right_data['driver_tag']]).to_list()

                    median_data['median'].append(v_data_points.median())
                    median_data['lower_iqr'].append(v_data_points.quantile(0.25))
                    median_data['upper_iqr'].append(v_data_points.quantile(0.75))
                    median_data['headway'].append(headway)
                    median_data['dv'].append(dv)
                    median_data['is_sim'].append(is_sim)

                    regression_data['abs_v_deviation'] += v_data_points.to_list()
                    regression_data['driver_tag'] += driver_tags
                    regression_data['headway'] += [abs(headway)] * len(driver_tags)
                    regression_data['dv'] += [abs(dv)] * len(driver_tags)
                    regression_data['is_sim'] += [is_sim] * len(driver_tags)

    median_data = pd.DataFrame(median_data)
    regression_data = pd.DataFrame(regression_data)

    median_data['headway'] += 0.5 * median_data['dv'] * ((median_data['headway'].abs() - 2) / 2).abs()

    median_data.loc[median_data['is_sim'], 'headway'] += 0.08
    median_data.loc[~median_data['is_sim'], 'headway'] -= 0.08

    plt.figure()
    sns.scatterplot(median_data,
                    x='headway', y='median', hue='dv', style='is_sim',
                    markers={True: 'o', False: 's'}, s=50., linewidth=0, palette={0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'})

    iqr_palette = {0.0: '#7f7f7faa', 0.8: '#bcbd22aa', -0.8: '#17becfaa'}
    plt.vlines(median_data['headway'], median_data['lower_iqr'], median_data['upper_iqr'], colors=median_data['dv'].replace(iqr_palette),
               linewidths=2., zorder=1)

    plt.xticks([-4., -2., 0., 2., 4.])
    plt.xlabel('Projected headway advantage for the individual driver [m]')
    plt.ylabel('Maximum absolute deviation from initial velocity [m/s]')
    plt.ylim((0., 4.))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    human_marker = mpl.lines.Line2D(marker='s', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(marker='o', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    color_m8 = mpl.lines.Line2D(c='tab:cyan', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_8 = mpl.lines.Line2D(c='tab:olive', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_0 = mpl.lines.Line2D(c='tab:grey', xdata=[0], ydata=[0], marker='s', linewidth=0)
    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])

    plt.legend([human_marker, model_marker, none_marker, color_m8, color_0, color_8],
               ['Human', 'Model', 'Relative Velocity:', '-0.8 m/s', '0.0 m/s', '0.8 m/s'])

    model_regression_data = regression_data.loc[regression_data['is_sim']]
    human_regression_data = regression_data.loc[~regression_data['is_sim']]
    print('Model data: ')
    md = smf.mixedlm("abs_v_deviation ~ headway * dv", model_regression_data, groups=model_regression_data["driver_tag"]).fit()
    print(md.summary())
    print(pd.DataFrame(md.random_effects).to_string())
    print(md.pvalues.to_string())
    print('------------------------------------------')
    print('Human data: ')
    md = smf.mixedlm("abs_v_deviation ~ headway * dv", human_regression_data, groups=human_regression_data["driver_tag"]).fit()
    print(md.summary())
    print(pd.DataFrame(md.random_effects).to_string())
    print(md.pvalues.to_string())


def plot_safety_margin(global_metrics_data):
    global_metrics_data['gap at merge point'] = global_metrics_data['headway at merge point'].abs() - 4.5
    global_metrics_data = global_metrics_data.loc[global_metrics_data['who went first'] != 'Collision', :]
    plot_data = {'condition': [],
                 'sim': [],
                 'pair': [],
                 'mean': []}

    for experiment_number in global_metrics_data['experiment_number'].unique():
        for condition in global_metrics_data['condition'].unique():
            for sim in [True, False]:
                data = global_metrics_data.loc[(global_metrics_data['condition'] == condition) &
                                               (global_metrics_data['experiment_number'] == experiment_number) &
                                               (global_metrics_data['is_simulation'] == sim),
                                               'gap at merge point']

                plot_data['condition'].append(condition)
                plot_data['sim'].append(sim)
                plot_data['pair'].append(experiment_number)
                plot_data['mean'].append(data.mean())

    plot_data = pd.DataFrame(plot_data)
    y_max = plot_data['mean'].max() * 1.1
    order = ['-4_-8', '-4_0', '-4_8', '-2_8', '0_8', '0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8']

    fig, axes = plt.subplots(1, 9, sharex=True, sharey=True, figsize=(15, 2.7))
    plt.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.92, top=0.9, bottom=0.15)

    for condition_index in range(11):
        con = mpl.patches.ConnectionPatch(xyA=(0., condition_index), coordsA=axes[0].transData,
                                          xyB=(y_max, condition_index), coordsB=axes[8].transData,
                                          zorder=0., linewidth=0.5, color='lightgray')
        fig.add_artist(con)

    for column_index in range(9):
        con = mpl.patches.ConnectionPatch(xyA=(0., 10.), coordsA=axes[column_index].transData,
                                          xyB=(0., 0.), coordsB=axes[column_index].transData,
                                          zorder=0., linewidth=0.5)
        fig.add_artist(con)

        con = mpl.patches.ConnectionPatch(xyA=(9., 10.), coordsA=axes[column_index].transData,
                                          xyB=(9., 0.), coordsB=axes[column_index].transData,
                                          zorder=0., linewidth=0.5, ls='dashed', color='lightgray')
        fig.add_artist(con)

    for pair_index, pair in enumerate(plot_data['pair'].unique()):
        pair_data = plot_data.loc[plot_data['pair'] == pair, :]
        ax = axes[pair_index]
        sns.stripplot(data=pair_data.loc[pair_data['sim']], y='condition', x='mean', ax=ax, color='tab:orange', marker='o',
                      dodge=False, jitter=0., edgecolor=None, linewidth=1., order=order)
        sns.stripplot(data=pair_data.loc[~pair_data['sim']], y='condition', x='mean', ax=ax, color='tab:blue', marker='s',
                      dodge=False, jitter=0., edgecolor=None, linewidth=1., order=order)

        ax.hlines(range(11), 0., y_max, linewidth=1., colors='k')
        ax.set_xlim(0., y_max)
        ax.set_ylabel('')
        ax.set_xlabel('')

    pair_3_data = global_metrics_data.loc[global_metrics_data['experiment_number'] == '3', :]
    ax = axes[2]
    sns.stripplot(data=pair_3_data, y='condition', x='gap at merge point', ax=ax, hue='is_simulation', order=order, dodge=True,
                  marker='d', s=5, jitter=.25, alpha=0.5, zorder=0, palette={True: 'tab:orange', False: 'tab:blue'})
    ax.set_xlabel('')

    for ax in axes:
        ax.legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.patch.set_alpha(0.)
        ax.set_xticks([0.0, 9.0])

    fig.text(0.02, 0.92, "Participant pair: ")
    for i in range(1, 10):
        fig.text(0.04 + 0.093 * i, 0.92, str(i))

    fig.text(0.02, 0.5, "Condition")

    axes[4].set_xlabel('Gap at the merge point [m]')

    human_marker = mpl.lines.Line2D(linewidth=0., marker='s', color='tab:blue', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(linewidth=0., marker='o', color='tab:orange', xdata=[0], ydata=[0])
    legend = axes[8].legend([human_marker, model_marker], ['Human', 'Model'], loc='center left', bbox_to_anchor=(1, 0.5))

    legend.get_texts()[0].set_text('Human')
    legend.get_texts()[1].set_text('Model')

    plot_data['coordinates'] = plot_data.apply(lambda r: 'y' if r['sim'] else 'x', axis=1)
    plot_data['id'] = plot_data['condition'] + '-' + plot_data['pair']
    r_squared_data = plot_data.pivot(index='id', columns='coordinates', values='mean')

    model = smf.ols('y ~ x', data=r_squared_data).fit()

    x_min = 0.
    x_max = 10.
    print(model.summary())
    print(model.pvalues)
    print(model.params)

    residual_sum_of_squares = ((r_squared_data['y'] - r_squared_data['x']) ** 2).sum()
    total_sum_of_squares = ((r_squared_data['y'] - r_squared_data['y'].mean()) ** 2).sum()
    r_squared = 1 - residual_sum_of_squares / total_sum_of_squares

    print(r_squared)

    plt.figure()
    sns.scatterplot(data=r_squared_data, x='x', y='y', color='gray')
    plt.xlim((x_min, x_max))
    plt.ylim((x_min, x_max))
    x_y_line, = plt.plot([0, x_max], [0, x_max], color='lightgrey', linestyle='dashed')
    regression_line, = plt.plot([0, x_max], [model.params['Intercept'], model.params['Intercept'] + x_max * model.params['x']], color='k', linestyle='-.')
    plt.gca().set_aspect('equal')
    plt.title('Gap at merge point [m]')
    plt.ylabel('Model')
    plt.xlabel('Human')

    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])
    plt.legend([x_y_line, none_marker, regression_line, none_marker, none_marker],
               ['x = y', '$R^{2}$: %.1f' % r_squared, 'Ordinary least-squares \n linear regression fit',
                'Intercept: %.2f' % model.params['Intercept'], 'Effect: %.2f' % model.params['x']])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    global_metrics_data.loc[:, 'projected_headway'] = global_metrics_data.apply(_get_projected_headway_from_condition, axis=1)
    global_metrics_data.loc[:, 'relative_velocity'] = global_metrics_data.apply(_get_relative_velocity_from_condition, axis=1)

    median_data = {'median': [],
                   'lower_iqr': [],
                   'upper_iqr': [],
                   'headway': [],
                   'dv': [],
                   'is_sim': []}

    for condition in global_metrics_data['condition'].unique():
        condition_data = global_metrics_data.loc[global_metrics_data['condition'] == condition]
        for is_sim in [True, False]:
            sim_data = condition_data.loc[condition_data['is_simulation'] == is_sim]

            median_data['median'].append(sim_data['gap at merge point'].median())
            median_data['lower_iqr'].append(sim_data['gap at merge point'].quantile(0.25))
            median_data['upper_iqr'].append(sim_data['gap at merge point'].quantile(0.75))
            median_data['headway'].append(sim_data['projected_headway'].iat[0])
            median_data['dv'].append(sim_data['relative_velocity'].iat[0])
            median_data['is_sim'].append(is_sim)

    median_data = pd.DataFrame(median_data)

    median_data['headway'] += 0.5 * median_data['dv'] * ((median_data['headway'].abs() - 2) / 2).abs()

    median_data.loc[median_data['is_sim'], 'headway'] += 0.08
    median_data.loc[~median_data['is_sim'], 'headway'] -= 0.08

    plt.figure()
    sns.scatterplot(median_data,
                    x='headway', y='median', hue='dv', style='is_sim',
                    markers={True: 'o', False: 's'}, s=50., linewidth=0, palette={0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'})

    iqr_palette = {0.0: '#7f7f7faa', 0.8: '#bcbd22aa', -0.8: '#17becfaa'}
    plt.vlines(median_data['headway'], median_data['lower_iqr'], median_data['upper_iqr'], colors=median_data['dv'].replace(iqr_palette),
               linewidths=2., zorder=1)

    plt.xticks([-4., -2., 0., 2., 4.])
    plt.xlabel('Projected headway advantage for the left vehicle [m]')
    plt.ylabel('Gap at the merge point [m]')
    plt.ylim((0, 10.))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    human_marker = mpl.lines.Line2D(marker='s', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(marker='o', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    color_m8 = mpl.lines.Line2D(c='tab:cyan', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_8 = mpl.lines.Line2D(c='tab:olive', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_0 = mpl.lines.Line2D(c='tab:grey', xdata=[0], ydata=[0], marker='s', linewidth=0)
    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])

    plt.legend([human_marker, model_marker, none_marker, color_m8, color_0, color_8],
               ['Human', 'Model', 'Relative Velocity:', '-0.8 m/s', '0.0 m/s', '0.8 m/s'])

    global_metrics_data['gap_at_merge_point'] = global_metrics_data['gap at merge point'].astype(float)
    global_metrics_data['abs_hw'] = global_metrics_data['projected_headway'].abs()
    global_metrics_data['abs_dv'] = global_metrics_data['relative_velocity'].abs()
    model_regression_data = global_metrics_data.loc[global_metrics_data['is_simulation']]
    human_regression_data = global_metrics_data.loc[~global_metrics_data['is_simulation']]
    md = smf.mixedlm("gap_at_merge_point ~ abs_hw + abs_dv + projected_headway:relative_velocity", model_regression_data,
                     groups=model_regression_data["experiment_number"]).fit()
    print('Model data: ')
    print(md.summary())
    print(pd.DataFrame(md.random_effects).to_string())
    print(md.pvalues.to_string())
    print('------------------------------------------')
    print('Human data: ')
    md = smf.mixedlm("gap_at_merge_point ~  abs_hw + abs_dv + projected_headway:relative_velocity", human_regression_data,
                     groups=human_regression_data["experiment_number"]).fit()
    print(md.summary())
    print(pd.DataFrame(md.random_effects).to_string())
    print(model_regression_data['gap_at_merge_point'].mean())
    print(human_regression_data['gap_at_merge_point'].mean())
    print(md.pvalues.to_string())


def plot_who_first_vs_action(individual_metric_data, global_metrics_data):
    metrics = individual_metric_data.loc[individual_metric_data['who went first'] != 'Collision']
    metrics['first-side'] = metrics['who went first'].astype(str) + ' first - ' + metrics['side'].astype(str) + ' action'

    plot_data = {'first-side': [],
                 'sim': [],
                 'pair': [],
                 'mean': []}

    for experiment_number in metrics['experiment_number'].unique():
        for sim in [True, False]:
            for side in metrics['first-side'].unique():
                data = metrics.loc[(metrics['experiment_number'] == experiment_number) &
                                   (metrics['first-side'] == side) &
                                   (metrics['is_simulation'] == sim),
                                   'extreme deviation from desired v']

                plot_data['sim'].append(sim)
                plot_data['pair'].append(experiment_number)
                plot_data['first-side'].append(str(side))
                plot_data['mean'].append(data.mean())

    plot_data = pd.DataFrame(plot_data)
    fig, axes = plt.subplots(1, 9, sharex=True, sharey=True, figsize=(15, 2.7))
    plt.subplots_adjust(wspace=0.2, hspace=0.1, left=0.13, right=0.92, top=0.9, bottom=0.15)

    for outcome_index in range(4):
        con = mpl.patches.ConnectionPatch(xyA=(-3.5, outcome_index), coordsA=axes[0].transData,
                                          xyB=(3.5, outcome_index), coordsB=axes[8].transData,
                                          zorder=0., linewidth=0.5, color='lightgrey')
        fig.add_artist(con)

    for pair_index, pair in enumerate(plot_data['pair'].unique()):
        ax = axes[pair_index]

        ax.hlines([0, 1, 2, 3], -5, 5, linewidth=1., colors='k')
        ax.vlines(0., -0.5, 4.5, zorder=0., linewidth=0.5, colors='k')
        ax.vlines([-3, 3], -0.5, 4.5, zorder=0., linewidth=0.5, linestyles='dashed', colors='lightgray')
        sns.stripplot(data=plot_data.loc[(plot_data['pair'] == pair) & (plot_data['sim']), :],
                      y='first-side', x='mean', ax=ax, color='tab:orange', marker='o',
                      dodge=False, jitter=0., edgecolor=None, linewidth=1.)
        sns.stripplot(data=plot_data.loc[(plot_data['pair'] == pair) & (~plot_data['sim']), :],
                      y='first-side', x='mean', ax=ax, color='tab:blue', marker='s',
                      dodge=False, jitter=0., edgecolor=None, linewidth=1.)

        ax.set_ylabel('')
        ax.set_xlabel('')

    data = metrics.loc[metrics['experiment_number'] == '3', :]
    ax = axes[2]
    sns.stripplot(data=data, y='first-side', x='extreme deviation from desired v', ax=ax, hue='is_simulation', dodge=True,
                  marker='d', s=5, jitter=.25, alpha=0.5, zorder=0, palette={True: 'tab:orange', False: 'tab:blue'})
    ax.set_xlabel('')

    for ax in np.array(axes).flatten():
        ax.legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.patch.set_alpha(0.)
        ax.set_xticks([-3, 0, 3])
        ax.set_xlim(-4, 4)

    fig.text(0.05, 0.92, "Participant pair: ")
    for i in range(1, 10):
        fig.text(0.07 + 0.088 * i, 0.92, str(i))

    fig.text(0.05, 0.55, "Right \n Merges First", rotation=90, horizontalalignment='center')
    fig.text(0.05, 0.2, "Left \n Merges First", rotation=90, horizontalalignment='center')

    fig.text(0.05, 0.92, "Participant pair: ")
    axes[0].set_yticks(ax.get_yticks(), ['Left Driver', 'Right Driver', 'Left Driver', 'Right Driver'])
    axes[4].set_xlabel('Extremum deviation from initial velocity [m/s]')

    human_marker = mpl.lines.Line2D(linewidth=0., marker='s', color='tab:blue', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(linewidth=0., marker='o', color='tab:orange', xdata=[0], ydata=[0])
    legend = axes[8].legend([human_marker, model_marker], ['Human', 'Model'], loc='center left', bbox_to_anchor=(1, 0.5))

    plot_data['coordinates'] = plot_data.apply(lambda r: 'y' if r['sim'] else 'x', axis=1)
    plot_data['id'] = plot_data['pair'] + '-' + plot_data['first-side']
    r_squared_data = plot_data.pivot(index='id', columns='coordinates', values='mean')

    residual_sum_of_squares = ((r_squared_data['y'] - r_squared_data['x']) ** 2).sum()
    total_sum_of_squares = ((r_squared_data['y'] - r_squared_data['y'].mean()) ** 2).sum()
    r_squared = 1 - residual_sum_of_squares / total_sum_of_squares
    print(r_squared)

    model = smf.ols('y ~ x', data=r_squared_data).fit()
    print(model.summary())
    print(model.pvalues)

    x_min = -3.
    x_max = 3.

    plt.figure()
    sns.scatterplot(r_squared_data, x='x', y='y', color='gray')
    plt.xlim((x_min, x_max))
    plt.ylim((x_min, x_max))
    x_y_line, = plt.plot([x_min, x_max], [x_min, x_max], color='lightgrey', linestyle='dashed')
    regression_line, = plt.plot([x_min, x_max], [model.params['Intercept'] + x_min * model.params['x'], model.params['Intercept'] + x_max * model.params['x']],
             color='k',
             linestyle='-.')
    plt.gca().set_aspect('equal')
    plt.title('Extremum deviation from initial velocity [m/s]')
    plt.ylabel('Model')
    plt.xlabel('Human')

    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])
    plt.legend([x_y_line, none_marker, regression_line, none_marker, none_marker],
               ['x = y', '$R^{2}$: %.1f' % r_squared, 'Ordinary least-squares \n linear regression fit',
                'Intercept: %.2f' % model.params['Intercept'], 'Effect: %.2f' % model.params['x']])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def plot_who_first_sim_vs_real(global_metric_data):
    percentage_results = []

    ci_data = {'condition': [],
               'left_over_right': [],
               'sum': [],
               'samples': [],
               'is_sim': []}
    for is_sim in [True, False]:
        percentage_data = {}
        for condition in global_metric_data['condition'].unique():
            condition_data = global_metric_data.loc[(global_metric_data['condition'] == condition) & (global_metric_data['is_simulation'] == is_sim),
                                                    'who went first']

            percentage_data[condition] = {}

            total = condition_data.count()
            for key in ['Collision', TrackSide.LEFT, TrackSide.RIGHT]:
                percentage_data[condition][key] = ((condition_data.loc[condition_data == key]).count() / total) * 100
            left_over_right = ((condition_data.loc[condition_data == TrackSide.LEFT]).count() /
                               (condition_data.loc[condition_data != 'Collision']).count()) * 100
            percentage_data[condition]['left_over_right'] = left_over_right

            ci_data['condition'].append(condition)
            ci_data['left_over_right'].append((left_over_right))
            ci_data['sum'].append((condition_data == TrackSide.LEFT).sum())
            ci_data['samples'].append((condition_data != 'Collision').sum())
            ci_data['is_sim'].append(is_sim)

        percentage_data = pd.DataFrame(percentage_data)
        percentage_data = percentage_data.stack().reset_index()
        percentage_data.columns = ['who went first', 'condition', 'percentage']
        percentage_data['is_sim'] = is_sim

        percentage_results.append(percentage_data)

    ci_data = pd.DataFrame(ci_data)
    percentage_results = pd.concat(percentage_results, ignore_index=True)

    percentage_results.loc[:, 'projected_headway'] = percentage_results.apply(_get_projected_headway_from_condition, axis=1)
    percentage_results.loc[:, 'relative_velocity'] = percentage_results.apply(_get_relative_velocity_from_condition, axis=1)

    ci_data = ci_data.apply(_get_binomial_confidence_interval, axis=1)
    ci_data.loc[:, 'projected_headway'] = ci_data.apply(_get_projected_headway_from_condition, axis=1)
    ci_data.loc[:, 'relative_velocity'] = ci_data.apply(_get_relative_velocity_from_condition, axis=1)

    percentage_results.loc[percentage_results['is_sim'], 'projected_headway'] += 0.1
    percentage_results.loc[~percentage_results['is_sim'], 'projected_headway'] -= 0.1

    ci_data.loc[ci_data['is_sim'], 'projected_headway'] += 0.1
    ci_data.loc[~ci_data['is_sim'], 'projected_headway'] -= 0.1

    plt.figure()
    sns.scatterplot(percentage_results.loc[(percentage_results['who went first'] == 'left_over_right')],
                    x='projected_headway', y='percentage',
                    hue='relative_velocity', style='is_sim',
                    markers={True: 'o', False: 's'}, s=50., linewidth=0, palette={0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'})
    # sns.scatterplot(percentage_results.loc[(percentage_results['who went first'] == 'left_over_right') & ~percentage_results['is_sim']],
    #                 x='projected_headway', y='percentage',
    #                 hue='relative_velocity',
    #                 marker='s', s=50., linewidth=0, palette={0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'})

    ci_palette = {0.0: '#7f7f7faa', 0.8: '#bcbd22aa', -0.8: '#17becfaa'}
    plt.vlines(ci_data['projected_headway'], ci_data['ci_low'] * 100., ci_data['ci_high'] * 100.,
               colors=ci_data['relative_velocity'].replace(ci_palette),
               linewidths=2., zorder=1)
    plt.xticks([-4, -2, 0, 2, 4])
    plt.xlabel('Projected headway advantage for the left vehicle [m]')
    plt.ylabel('Percentage for the left vehicle merging first')

    human_marker = mpl.lines.Line2D(marker='s', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(marker='o', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    color_m8 = mpl.lines.Line2D(c='tab:cyan', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_8 = mpl.lines.Line2D(c='tab:olive', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_0 = mpl.lines.Line2D(c='tab:grey', xdata=[0], ydata=[0], marker='s', linewidth=0)
    none_marker = mpl.lines.Line2D(c='k', xdata=[0], ydata=[0], linewidth=0)

    plt.legend([human_marker, model_marker, none_marker, color_m8, color_0, color_8],
               ['Human', 'Model', 'Relative Velocity:', '-0.8 m/s', '0.0 m/s', '0.8 m/s'])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def plot_traces_example():
    fig, ax = plt.subplots(3, 2, figsize=(14, 5), sharey='row')

    sim_file = os.path.join('..', 'data', 'simulated_data', 'experiment_6', 'simulation_6_iter_94.pkl')
    human_file = os.path.join('..', 'data', 'experiment_data', 'experiment_6', 'experiment_6_iter_47.pkl')

    for axis_column, file in enumerate([sim_file, human_file]):
        with open(file, 'rb') as f:
            trial_data = pickle.load(f)

        time_vector = np.array([t * 0.05 for t in range(len(trial_data['travelled_distance'][TrackSide.LEFT]))])

        tunnel_exit_index = np.where((np.array(trial_data['travelled_distance'][TrackSide.LEFT]) > 50.) &
                                     (np.array(trial_data['travelled_distance'][TrackSide.RIGHT]) > 50.))[0][0]
        merge_index = np.where((np.array(trial_data['travelled_distance'][TrackSide.LEFT]) > 100.) |
                               (np.array(trial_data['travelled_distance'][TrackSide.RIGHT]) > 100.))[0][0]

        scatter_times = [0, tunnel_exit_index, merge_index, -1]
        left_positions = np.array(trial_data['positions'][TrackSide.LEFT])
        right_positions = np.array(trial_data['positions'][TrackSide.RIGHT])

        ax[0, axis_column].plot(left_positions[:, 1], left_positions[:, 0] - 1., color='tab:green', linewidth=2., zorder=3)
        ax[0, axis_column].scatter(left_positions[scatter_times, 1], left_positions[scatter_times, 0] - 1., color='tab:green', zorder=5)
        ax[0, axis_column].plot(right_positions[:, 1], right_positions[:, 0] + 1., color='tab:purple', linewidth=2., zorder=4)
        ax[0, axis_column].scatter(right_positions[scatter_times, 1], right_positions[scatter_times, 0] + 1., color='tab:purple', zorder=6)

        # ax[0].add_patch(mpl.patches.FancyBboxPatch((183, 1.6), 50, 3.4, color='#55555577', zorder=1, boxstyle='Round,pad=5.', mutation_aspect=.1,
        #                                            linewidth=0))
        #
        for scatter_index in scatter_times:
            ax[0, axis_column].plot([left_positions[scatter_index, 1], right_positions[scatter_index, 1]],
                                    [left_positions[scatter_index, 0] - 1., right_positions[scatter_index, 0] + 1.],
                                    color='k', linestyle='-', linewidth=0.4)

        gap = np.abs(np.array(trial_data['travelled_distance'][TrackSide.LEFT]) - np.array(trial_data['travelled_distance'][TrackSide.RIGHT]))
        gap[np.where(gap <= 4.5)] = 0.
        gap[np.where(gap > 0.)] -= 4.5

        ax[1, axis_column].plot(time_vector, gap, c='k', linewidth=2.)
        ax[1, axis_column].scatter(time_vector[scatter_times], gap[scatter_times], c='k')

        left_velocities = np.array(trial_data['velocities'][TrackSide.LEFT])
        right_velocities = np.array(trial_data['velocities'][TrackSide.RIGHT])

        ax[2, axis_column].plot(time_vector, left_velocities, color='tab:purple', linewidth=2., zorder=2)
        ax[2, axis_column].scatter(time_vector[scatter_times], left_velocities[scatter_times], color='tab:purple', zorder=4)
        ax[2, axis_column].plot(time_vector, right_velocities, color='tab:green', linewidth=2., zorder=3)
        ax[2, axis_column].scatter(time_vector[scatter_times], right_velocities[scatter_times], color='tab:green', zorder=5)
        #
        # ax[2].add_patch(mpl.patches.FancyBboxPatch((4., 14.), 2., 10., color='#55555577', zorder=0, boxstyle='Round,pad=.2', mutation_aspect=8.,
        #                                            linewidth=0))
        # ax[2].add_patch(mpl.patches.FancyBboxPatch((7., 17.), 1.73, 7., color='#55555577', zorder=1, boxstyle='Round,pad=.2', mutation_aspect=8.,
        #                                            linewidth=0))

        ax[2, axis_column].set_xlabel('time [s]')
        ax[0, axis_column].xaxis.set_label_position('top')
        ax[0, axis_column].xaxis.tick_top()
        ax[0, axis_column].set_xlabel('Longitudinal position [m]')

        ax[1, axis_column].set_xticks(ax[2, axis_column].get_xticks(), [])

        ax[0, axis_column].set_facecolor('#d4aa00')
        ax[1, axis_column].set_facecolor('#ffdd55')
        ax[2, axis_column].set_facecolor('#ffeeaa')

    ax[0, 0].set_ylabel('Lateral \n position [m]', labelpad=45., verticalalignment='center')
    ax[1, 0].set_ylabel('Gap [m]', labelpad=45., verticalalignment='center')
    ax[2, 0].set_ylabel('Velocity [m/s]', labelpad=45., verticalalignment='center')

    ax[1, 0].set_xlim((-0.5, 14.5))
    ax[1, 1].set_xlim((-0.5, 14.5))
    ax[2, 0].set_xlim((-0.5, 14.5))
    ax[2, 1].set_xlim((-0.5, 14.5))
    ax[1, 0].set_xticks(range(0, 15, 2), [])
    ax[1, 1].set_xticks(range(0, 15, 2), [])
    ax[2, 0].set_xticks(range(0, 15, 2), range(0, 15, 2))
    ax[2, 1].set_xticks(range(0, 15, 2), range(0, 15, 2))

    for column in range(2):
        for row in range(3):
            ax[row, column].spines['top'].set_visible(row == 0)
            ax[row, column].spines['right'].set_visible(False)
            ax[row, column].spines['bottom'].set_visible(row != 0)

    fig.subplots_adjust(left=0.1, right=0.99, hspace=0.1, wspace=0.1)
    plt.rcParams.update({
        "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })


def plot_crt(global_metrics_data):
    global_metrics_data = global_metrics_data.loc[global_metrics_data['who went first'] != 'Collision', :]
    plot_data = {'condition': [],
                 'sim': [],
                 'pair': [],
                 'mean': []}

    for experiment_number in global_metrics_data['experiment_number'].unique():
        for condition in global_metrics_data['condition'].unique():
            for sim in [True, False]:
                data = global_metrics_data.loc[(global_metrics_data['condition'] == condition) &
                                               (global_metrics_data['experiment_number'] == experiment_number) &
                                               (global_metrics_data['is_simulation'] == sim),
                                               'nett crt']

                plot_data['condition'].append(condition)
                plot_data['sim'].append(sim)
                plot_data['pair'].append(experiment_number)
                plot_data['mean'].append(data.mean())

    plot_data = pd.DataFrame(plot_data)
    y_max = plot_data['mean'].max() * 1.1
    order = ['-4_-8', '-4_0', '-4_8', '-2_8', '0_8', '0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8']

    fig, axes = plt.subplots(1, 9, sharex=True, sharey=True, figsize=(15, 2.7))
    plt.subplots_adjust(wspace=0.2, hspace=0.1, left=0.1, right=0.92, top=0.9, bottom=0.15)

    for condition_index in range(11):
        con = mpl.patches.ConnectionPatch(xyA=(0., condition_index), coordsA=axes[0].transData,
                                          xyB=(y_max, condition_index), coordsB=axes[8].transData,
                                          zorder=0., linewidth=0.5, color='lightgray')
        fig.add_artist(con)

    for column_index in range(9):
        con = mpl.patches.ConnectionPatch(xyA=(0., 10.), coordsA=axes[column_index].transData,
                                          xyB=(0., 0.), coordsB=axes[column_index].transData,
                                          zorder=0., linewidth=0.5)
        fig.add_artist(con)

        con = mpl.patches.ConnectionPatch(xyA=(4., 10.), coordsA=axes[column_index].transData,
                                          xyB=(4., 0.), coordsB=axes[column_index].transData,
                                          zorder=0., linewidth=0.5, ls='dashed', color='lightgray')
        fig.add_artist(con)

    for pair_index, pair in enumerate(plot_data['pair'].unique()):
        pair_data = plot_data.loc[plot_data['pair'] == pair, :]
        ax = axes[pair_index]
        sns.stripplot(data=pair_data.loc[pair_data['sim']], y='condition', x='mean', ax=ax, color='tab:orange', marker='o',
                      dodge=False, jitter=0., edgecolor=None, linewidth=1., order=order)
        sns.stripplot(data=pair_data.loc[~pair_data['sim']], y='condition', x='mean', ax=ax, color='tab:blue', marker='s',
                      dodge=False, jitter=0., edgecolor=None, linewidth=1., order=order)

        ax.hlines(range(11), 0., y_max, linewidth=1., colors='k')
        ax.set_xlim(0., y_max)
        ax.set_ylabel('')
        ax.set_xlabel('')

    pair_3_data = global_metrics_data.loc[global_metrics_data['experiment_number'] == '3', :]
    ax = axes[2]
    sns.stripplot(data=pair_3_data, y='condition', x='nett crt', ax=ax, hue='is_simulation', order=order, dodge=True,
                  marker='d', s=5, jitter=.25, alpha=0.5, zorder=0, palette={True: 'tab:orange', False: 'tab:blue'})
    ax.set_xlabel('')

    for ax in axes:
        ax.legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.patch.set_alpha(0.)
        ax.set_xticks([0.0, 4.0])

    fig.text(0.02, 0.92, "Participant pair: ")
    for i in range(1, 10):
        fig.text(0.04 + 0.093 * i, 0.92, str(i))

    fig.text(0.02, 0.5, "Condition")

    axes[4].set_xlabel('CRT [S]')

    human_marker = mpl.lines.Line2D(linewidth=0., marker='s', color='tab:blue', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(linewidth=0., marker='o', color='tab:orange', xdata=[0], ydata=[0])
    legend = axes[8].legend([human_marker, model_marker], ['Human', 'Model'], loc='center left', bbox_to_anchor=(1, 0.5))

    legend.get_texts()[0].set_text('Human')
    legend.get_texts()[1].set_text('Model')

    plot_data['coordinates'] = plot_data.apply(lambda r: 'y' if r['sim'] else 'x', axis=1)
    plot_data['id'] = plot_data['condition'] + '-' + plot_data['pair']
    r_squared_data = plot_data.pivot(index='id', columns='coordinates', values='mean')

    model = smf.ols('y ~ x', data=r_squared_data).fit()

    x_min = 0.
    x_max = 3.5
    print(model.params)
    print(model.summary())
    print(model.pvalues)

    residual_sum_of_squares = ((r_squared_data['y'] - r_squared_data['x']) ** 2).sum()
    total_sum_of_squares = ((r_squared_data['y'] - r_squared_data['y'].mean()) ** 2).sum()
    r_squared = 1 - residual_sum_of_squares / total_sum_of_squares

    print(r_squared)

    plt.figure()
    sns.scatterplot(data=r_squared_data, x='x', y='y', color='gray')
    plt.xlim((x_min, x_max))
    plt.ylim((x_min, x_max))
    x_y_line, = plt.plot([0, x_max], [0, x_max], color='lightgrey', linestyle='dashed')
    regression_line, = plt.plot([0, x_max], [model.params['Intercept'], model.params['Intercept'] + x_max * model.params['x']], color='k',
                                linestyle='-.')
    plt.gca().set_aspect('equal')
    plt.title('CRT [s]')
    plt.ylabel('Model')
    plt.xlabel('Human')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])
    plt.legend([x_y_line, none_marker, regression_line, none_marker, none_marker],
               ['x = y', '$R^{2}$: %.1f' % r_squared, 'Ordinary least-squares \n linear regression fit',
                'Intercept: %.2f' % model.params['Intercept'], 'Effect: %.2f' % model.params['x']])

    global_metrics_data.loc[:, 'projected_headway'] = global_metrics_data.apply(_get_projected_headway_from_condition, axis=1)
    global_metrics_data.loc[:, 'relative_velocity'] = global_metrics_data.apply(_get_relative_velocity_from_condition, axis=1)

    global_metrics_data.loc[:, 'who_first_int'] = (global_metrics_data['who went first'].apply(lambda v: -1 * v.value + 1)).astype(int)
    who_went_first_sign_data = global_metrics_data['who_first_int'] * 2 - 1

    global_metrics_data.loc[:, 'relative_velocity_crt'] = global_metrics_data.loc[:, 'relative_velocity'] * who_went_first_sign_data
    global_metrics_data.loc[:, 'projected_headway_crt'] = (global_metrics_data.loc[:, 'projected_headway'] * who_went_first_sign_data).astype(int)

    median_data = {'median': [],
                   'lower_iqr': [],
                   'upper_iqr': [],
                   'headway': [],
                   'dv': [],
                   'is_sim': [],
                   'count': [],}

    for h in [-4, -2, 0, 2, 4]:
        for v in [-0.8, 0., 0.8]:
            selected_data = global_metrics_data.loc[(global_metrics_data['projected_headway_crt'] == h) &
                                                    (global_metrics_data['relative_velocity_crt'] == v)]
            if not selected_data.empty:
                for is_sim in [True, False]:
                    sim_data = selected_data.loc[selected_data['is_simulation'] == is_sim]

                    median_data['median'].append(sim_data['nett crt'].median())
                    median_data['lower_iqr'].append(sim_data['nett crt'].quantile(0.25))
                    median_data['upper_iqr'].append(sim_data['nett crt'].quantile(0.75))
                    median_data['headway'].append(h)
                    median_data['dv'].append(v)
                    median_data['is_sim'].append(is_sim)
                    median_data['count'].append('n=%d' % sim_data['nett crt'].count())

    median_data = pd.DataFrame(median_data)

    median_data['headway'] += 0.5 * median_data['dv'] * ((median_data['headway'].abs() - 2) / 2).abs()

    median_data.loc[median_data['is_sim'], 'headway'] += 0.08
    median_data.loc[~median_data['is_sim'], 'headway'] -= 0.08

    plt.figure()
    sns.scatterplot(median_data,
                    x='headway', y='median', hue='dv', style='is_sim',
                    markers={True: 'o', False: 's'}, s=50., linewidth=0, palette={0.0: 'tab:grey', 0.8: 'tab:olive', -0.8: 'tab:cyan'})

    iqr_palette = {0.0: '#7f7f7faa', 0.8: '#bcbd22aa', -0.8: '#17becfaa'}
    plt.vlines(median_data['headway'], median_data['lower_iqr'], median_data['upper_iqr'], colors=median_data['dv'].replace(iqr_palette),
               linewidths=2., zorder=1)

    plt.xticks([-4., -2., 0., 2., 4.])
    plt.xlabel('Projected headway advantage for the left vehicle [m]')
    plt.ylabel('CRT [s]')
    plt.ylim((0, 3.))
    plt.gca().spines['right'].set_visible(False)

    def index_to_label(index):
        return index

    def label_to_index(label):
        return label

    second_axis = plt.gca().secondary_xaxis('top', functions=(index_to_label, label_to_index))
    second_axis.set_ticks(median_data['headway'], median_data['count'], rotation=60, ha="left", size='x-small', horizontalalignment='left')

    human_marker = mpl.lines.Line2D(marker='s', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    model_marker = mpl.lines.Line2D(marker='o', linewidth=0, linestyle=None, c='k', xdata=[0], ydata=[0])
    color_m8 = mpl.lines.Line2D(c='tab:cyan', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_8 = mpl.lines.Line2D(c='tab:olive', xdata=[0], ydata=[0], marker='s', linewidth=0)
    color_0 = mpl.lines.Line2D(c='tab:grey', xdata=[0], ydata=[0], marker='s', linewidth=0)
    none_marker = mpl.lines.Line2D(linewidth=0, linestyle=None, xdata=[0], ydata=[0])

    plt.legend([human_marker, model_marker, none_marker, color_m8, color_0, color_8],
               ['Human', 'Model', 'Relative Velocity:', '-0.8 m/s', '0.0 m/s', '0.8 m/s'])

    global_metrics_data['nett_crt'] = global_metrics_data['nett crt'].astype(float)
    global_metrics_data['abs_hw'] = global_metrics_data['projected_headway'].abs()
    global_metrics_data['abs_dv'] = global_metrics_data['relative_velocity'].abs()
    model_regression_data = global_metrics_data.loc[global_metrics_data['is_simulation']]
    human_regression_data = global_metrics_data.loc[~global_metrics_data['is_simulation']]

    md = smf.mixedlm("nett_crt ~ projected_headway_crt + relative_velocity_crt + projected_headway:relative_velocity", model_regression_data,
                     groups=model_regression_data["experiment_number"]).fit()
    print(md.summary())
    print(md.pvalues.to_string())
    md = smf.mixedlm("nett_crt ~  projected_headway_crt + relative_velocity_crt + projected_headway:relative_velocity", human_regression_data,
                     groups=human_regression_data["experiment_number"]).fit()
    print(md.summary())
    print(md.pvalues.to_string())
    print(model_regression_data['nett_crt'].mean())
    print(human_regression_data['nett_crt'].mean())


if __name__ == '__main__':
    conditions_to_consider = ['L_4_8', 'L_4_0', 'L_4_-8', 'L_2_-8', 'L_0_-8', 'N_0_0', 'R_0_8', 'R_-2_8', 'R_-4_8', 'R_-4_0', 'R_-4_-8']

    global_traces, global_metrics, individual_traces, individual_metrics = load_experiment_data(conditions_to_consider,
                                                                                                                  include_simulations=True)
    rename_to_paper_conventions(global_traces, global_metrics, individual_traces, individual_metrics)
    individual_metrics = pd.merge(individual_metrics, global_metrics[['who went first', 'trial_number']], how='inner', on='trial_number')

    plt.rcParams["font.family"] = "Century Gothic"
    # plt.rcParams['font.size'] = 16

    plot_velocity_traces(individual_traces, global_metrics, '0_0', highlighted_pair='3')
    plot_deviation_from_v_medians(individual_metrics)
    plot_who_first_sim_vs_real(global_metrics)
    plot_safety_margin(global_metrics)
    plot_who_first_vs_action(individual_metrics, global_metrics)
    plot_crt(global_metrics)
    plot_traces_example()

    plt.show()
