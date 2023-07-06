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
import pandas as pd
import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots

import numpy as np

from plotting.compare_conditions import load_experiment_data
from trackobjects.trackside import TrackSide


def _get_projected_headway(row):
    condition = str(row['condition'])
    elements = condition.split('_')
    return float(elements[0])


def _get_relative_velocity(row):
    condition = str(row['condition'])
    elements = condition.split('_')
    return float(elements[1]) / 10.


def rename_to_paper_conventions(global_traces, global_metrics, individual_traces, individual_metrics):
    condition_names = {'L_4_8': '4_8',
                       'L_4_0': '4_0',
                       'L_4_-8': '4_-8',
                       'L_2_-8': '2_-8',
                       'L_0_-8': '0_-8',
                       'N_0_0': '0_0',
                       'R_0_8': '0_8',
                       'R_-2_8': '-2_8',
                       'R_-4_8': '-4_8',
                       'R_-4_0': '-4_0',
                       'R_-4_-8': '-4_-8'}

    experiment_numbers = {'4': '1',
                          '5': '2',
                          '6': '3',
                          '8': '4',
                          '9': '5',
                          '10': '6',
                          '11': '7',
                          '12': '8',
                          '13': '9'}

    for df in [global_traces, global_metrics, individual_traces, individual_metrics]:
        df.replace(condition_names, inplace=True)

        df['experiment_number'] = df['experiment_number'].apply(lambda v: experiment_numbers[v])
        df['trial_number'] = df['trial_number'].apply(lambda v: experiment_numbers[v.split('-')[0]] + '-' + v.split('-')[1])


def get_metrics_excluding_collisions(metrics_data):
    excluding_collisions = metrics_data.loc[metrics_data['who went first'] != 'Collision', :].copy()

    excluding_collisions.loc[:, 'projected_headway'] = excluding_collisions.apply(_get_projected_headway, axis=1)
    excluding_collisions.loc[:, 'relative_velocity'] = excluding_collisions.apply(_get_relative_velocity, axis=1)

    return excluding_collisions


def generate_aggregated_velocity_plots(individual_traces, global_metrics):
    plot_data = pd.merge(individual_traces, global_metrics[['who went first', 'trial_number']], how='inner', on='trial_number')
    plot_data['ego first'] = (plot_data['side'] == plot_data['who went first'].astype(str))
    plot_data['ego first'] = plot_data['ego first'].astype(str)
    plot_data.loc[plot_data['who went first'] == 'Collision', 'ego first'] = 'Collision'

    line_styles = {'True': 'solid',
                   'False': 'dash',
                   'Collision': 'dot'}

    for condition_name in tqdm.tqdm(plot_data['condition'].unique()):
        df = plot_data.loc[plot_data['condition'] == condition_name, :]
        fig = subplots.make_subplots(rows=1, cols=2,
                                     x_title='Time [s]',
                                     y_title='Velocity [m\s]',
                                     subplot_titles=("Left Vehicle", "Right Vehicle"))

        for color_index, experiment_pair in enumerate(df['experiment_number'].unique()):
            is_first = True
            for col, side in enumerate(TrackSide):
                data = df.loc[(df['side'] == str(side)) & (df['experiment_number'] == experiment_pair), :]
                for trial in data['trial_number'].unique():
                    trial_data = data.loc[data['trial_number'] == trial, :]

                    if not trial_data.empty:
                        fig.add_trace(go.Scatter(x=trial_data['time [s]'],
                                                 y=trial_data['velocity [m/s]'],
                                                 name=experiment_pair, legendgroup=experiment_pair,
                                                 showlegend=is_first and trial_data['ego first'].iat[0] == 'True',
                                                 mode='lines',
                                                 hovertemplate=trial,
                                                 line=dict(color=px.colors.qualitative.D3[color_index],
                                                           dash=line_styles[trial_data['ego first'].iat[0]])),
                                      row=1, col=col + 1)
                        if is_first and trial_data['ego first'].iat[0] == 'True':
                            is_first = False

        fig.add_trace(go.Scatter(x=[0], y=[0], name="Went First", mode="lines", line=dict(color="Grey", dash='solid'), showlegend=True,
                                 legendgroup="Outcome", legendgrouptitle_text="Outcome"))
        fig.add_trace(go.Scatter(x=[0], y=[0], name="Went Second", mode="lines", line=dict(color="Grey", dash='dash'), showlegend=True,
                                 legendgroup="Outcome", legendgrouptitle_text="Outcome"))
        fig.add_trace(go.Scatter(x=[0], y=[0], name="Collision", mode="lines", line=dict(color="Grey", dash='dot'), showlegend=True,
                                 legendgroup="Outcome", legendgrouptitle_text="Outcome"))

        fig.update_xaxes(range=[4.7, df['time [s]'].max() * 1.01])
        fig.update_yaxes(range=[df['velocity [m/s]'].min() * 0.99, df['velocity [m/s]'].max() * 1.01])
        fig.update_layout(legend_title_text='Participant Pair')
        fig.write_html('html_output\\' + condition_name + '.html')


def generate_who_first_bar(global_metrics):
    global_metrics['who went first'] = global_metrics['who went first'].astype(str)
    fig = px.histogram(data_frame=global_metrics, y="condition", color='who went first', barmode='group',
                       color_discrete_map={'Collision': px.colors.qualitative.D3[3],
                                           str(TrackSide.LEFT): px.colors.qualitative.D3[0],
                                           str(TrackSide.RIGHT): px.colors.qualitative.D3[1]},
                       category_orders={'condition': ['-4_-8', '-4_0', '-4_8', '-2_8', '0_8', '0_0', '0_-8', '2_-8', '4_-8', '4_0', '4_8'],
                                        'who went first': ['Collision', 'left', 'right']})
    fig.write_html('html_output\\who_first_box.html')


def plot_logitic_plotly(data, headway, rel_v, regression_results):
    res = 25
    x = np.linspace(-4, 4, res)
    y = np.linspace(-0.8, 0.8, res)
    grid = np.meshgrid(x, y)
    as_2d_array = np.array(grid).reshape((2, res ** 2))
    as_df = pd.DataFrame(as_2d_array.T, columns=[headway, rel_v])
    as_df[headway + ":" + rel_v] = as_df[headway] * as_df[rel_v]
    regression_prediction = np.array(regression_results.predict(as_df, verify_predictions=False, use_rfx=False)).reshape((res, res)) * 100

    # Creating the plot
    lines = []
    line_marker = dict(color='#000000', width=4)
    is_first = True
    for i in range(res):
        lines.append(go.Scatter3d(x=grid[0][i, :], y=grid[1][i, :], z=regression_prediction[i, :], mode='lines', line=line_marker,
                                  name='Logistic Regression Model', legendgroup='surf', showlegend=is_first))
        is_first = False
    for j in range(res):
        lines.append(go.Scatter3d(x=grid[0][:, j], y=grid[1][:, j], z=regression_prediction[:, j], mode='lines', line=line_marker,
                                  name='Logistic Regression Model', legendgroup='surf', showlegend=False))

    layout = go.Layout(hovermode="x",
                       scene={'xaxis_title': "Projected Headway [m]",
                              'yaxis_title': "Relative Velocity [m/s]",
                              'zaxis_title': "Percentage Left Vehicle Merging First"})

    fig = go.Figure(data=lines, layout=layout)

    sum_data = data.groupby([headway, rel_v])['who_first_int'].sum()
    count_data = data.groupby([headway, rel_v])['who_first_int'].count()
    percentage_data = (sum_data / count_data) * 100
    percentage_data = percentage_data.reset_index()

    dx = 0.2
    dy = 0.0375
    is_first = True

    for index in percentage_data.index:
        x, y, z = percentage_data.loc[index, [headway, rel_v, 'who_first_int']]
        fig.add_mesh3d(
            # 8 vertices of a cube
            x=[x - dx, x - dx, x + dx, x + dx, x - dx, x - dx, x + dx, x + dx],
            y=[y - dy, y + dy, y + dy, y - dy, y - dy, y + dy, y + dy, y - dy],
            z=[0, 0, 0, 0, z, z, z, z],
            color=px.colors.qualitative.D3[0],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.6,
            flatshading=True,
            hovertemplate=("Condition: %.f_%.f <br>Percentage left first: %.2f " % (x, y * 10, z)),
            name='Left first',
            legendgroup='left',
            showlegend=is_first,
        )

        fig.add_mesh3d(
            # 8 vertices of a cube
            x=[x - dx, x - dx, x + dx, x + dx, x - dx, x - dx, x + dx, x + dx],
            y=[y - dy, y + dy, y + dy, y - dy, y - dy, y + dy, y + dy, y - dy],
            z=[z, z, z, z, 100, 100, 100, 100],
            color=px.colors.qualitative.D3[1],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.6,
            flatshading=True,
            hovertemplate=("Condition: %.f_%.f <br>Percentage right first: %.2f " % (x, y * 10, 100 - z)),
            name='Right first',
            legendgroup='right',
            showlegend=is_first,
        )
        is_first = False

    fig.write_html('html_output\\surf.html')


def generate_initial_action_scatter(global_metric_data, individual_metric_data):
    variable = 'initial nett acceleration'

    left_initial_input = individual_metric_data.loc[individual_metric_data['side'] == 'left', ['trial_number', variable]]
    left_initial_input.rename(columns={variable: 'left ' + variable}, inplace=True)
    right_initial_input = individual_metric_data.loc[individual_metric_data['side'] == 'right', ['trial_number', variable]]
    right_initial_input.rename(columns={variable: 'right ' + variable}, inplace=True)

    plot_data = pd.merge(global_metric_data.loc[:, ['who went first', 'trial_number']], left_initial_input, on='trial_number')
    plot_data = pd.merge(plot_data, right_initial_input, on='trial_number')

    fig = px.scatter(plot_data,
                     x='left initial nett acceleration',
                     y='right initial nett acceleration', color='who went first',
                     color_discrete_map={'Collision': px.colors.qualitative.D3[3],
                                         TrackSide.LEFT: px.colors.qualitative.D3[0],
                                         TrackSide.RIGHT: px.colors.qualitative.D3[1]})

    fig.update_traces(marker=dict(size=14, symbol="diamond-tall", opacity=0.7), selector=dict(mode="markers"))

    fig.add_trace(
        go.Scatter(x=np.linspace(-4, 4, 10), y=np.linspace(-4, 4, 10), mode='lines', line={'color': 'gray', 'dash': 'dash'}, showlegend=False))

    fig.update_layout(height=800, width=800)

    fig.update_xaxes(range=[-3.5, 2.])
    fig.update_yaxes(range=[-3.5, 2.])

    fig.write_html('html_output\\initial_action.html')


def generate_3d_boxplot(data_excluding_collisions):
    headway = 'projected_headway'
    rel_v = 'relative_velocity'

    fig = go.Figure()
    color_dict = {'0_0': '#999999',
                  '0_-8': '#1f77b4',
                  '2_-8': '#3d81b4',
                  '4_-8': '#548bb4',
                  '4_0': '#6a95b4',
                  '4_8': '#7e9eb4',
                  '0_8': '#ff7f0e',
                  '-2_8': '#ff9740',
                  '-4_8': '#ffad66',
                  '-4_0': '#ffc28c',
                  '-4_-8': '#ffd6b3'}

    grouped_data = data_excluding_collisions.groupby([headway, rel_v])['nett crt']

    for (headway_value, rel_v_value) in grouped_data.indices.keys():
        crt_data = data_excluding_collisions.loc[(data_excluding_collisions[headway] == headway_value) &
                                                 (data_excluding_collisions[rel_v] == rel_v_value), 'nett crt'].to_numpy()
        condition = data_excluding_collisions.loc[(data_excluding_collisions[headway] == headway_value) &
                                                  (data_excluding_collisions[rel_v] == rel_v_value), 'condition'].unique()[0]
        plot_3d_box(headway_value, rel_v_value, crt_data, fig, condition, box_depth=0.2, whisker_width=0.05, whisker_depth=0.05,
                    color=color_dict[condition])

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-4.5, 4.5], showbackground=False),
            yaxis=dict(range=[-4.5, 4.5], ),
            zaxis=dict(range=[0, 5.5], ),
            xaxis_title="Projected Headway [m]",
            yaxis_title="Relative Velocity [m/s]",
            zaxis_title="Percentage Left Vehicle Merging First")
    )

    fig.write_html('html_output\\3d_boxplot.html')


def plot_3d_box(x, y, z, fig, condition_label, median_bar_height=0.05, box_width=0.2, box_depth=0.2, whisker_width=0.02, whisker_depth=0.02,
                color=px.colors.qualitative.D3[0]):
    q0, q1, median_z, q3, q4 = np.percentile(z.astype(float), [0, 25, 50, 75, 100])

    # median line
    dx = box_width * 1.2 / 2
    dy = box_depth * 1.2 / 2
    dz = median_bar_height / 2
    median_bar = go.Mesh3d(x=[x - dx, x - dx, x + dx, x + dx, x - dx, x - dx, x + dx, x + dx],
                           y=[y - dy, y + dy, y + dy, y - dy, y - dy, y + dy, y + dy, y - dy],
                           z=[median_z - dz, median_z - dz, median_z - dz, median_z - dz, median_z + dz, median_z + dz, median_z + dz, median_z + dz],
                           color='red',
                           i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                           j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                           k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                           opacity=0.8,
                           showlegend=False,
                           hovertemplate="Condition: %s <br>Q1: %.2f <br>Median: %.2f <br>Q3: %.2f" % (condition_label, q1, median_z, q3))

    # whisker bottom
    dx = whisker_width / 2
    dy = whisker_depth / 2
    whisker_bottom = go.Mesh3d(x=[x - dx, x - dx, x + dx, x + dx, x - dx, x - dx, x + dx, x + dx],
                               y=[y - dy, y + dy, y + dy, y - dy, y - dy, y + dy, y + dy, y - dy],
                               z=[q0, q0, q0, q0, q1, q1, q1, q1],
                               color='black',
                               i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                               j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                               k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                               opacity=0.6,
                               showlegend=False,
                               hovertemplate="Condition: %s <br>Q1: %.2f <br>Median: %.2f <br>Q3: %.2f" % (condition_label, q1, median_z, q3))

    # box
    dx = box_width / 2
    dy = box_depth / 2
    box = go.Mesh3d(x=[x - dx, x - dx, x + dx, x + dx, x - dx, x - dx, x + dx, x + dx],
                    y=[y - dy, y + dy, y + dy, y - dy, y - dy, y + dy, y + dy, y - dy],
                    z=[q1, q1, q1, q1, q3, q3, q3, q3],
                    color=color,
                    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    opacity=0.8,
                    showlegend=False,
                    hovertemplate="Condition: %s <br>Q1: %.2f <br>Median: %.2f <br>Q3: %.2f" % (condition_label, q1, median_z, q3))

    # whisker top
    dx = whisker_width / 2
    dy = whisker_depth / 2
    whisker_top = go.Mesh3d(x=[x - dx, x - dx, x + dx, x + dx, x - dx, x - dx, x + dx, x + dx],
                            y=[y - dy, y + dy, y + dy, y - dy, y - dy, y + dy, y + dy, y - dy],
                            z=[q3, q3, q3, q3, q4, q4, q4, q4],
                            color='black',
                            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                            opacity=0.6,
                            showlegend=False,
                            hovertemplate="Condition: %s <br>Q1: %.2f <br>Median: %.2f <br>Q3: %.2f" % (condition_label, q1, median_z, q3))

    fig.add_trace(whisker_top)
    fig.add_trace(whisker_bottom)
    fig.add_trace(box)
    fig.add_trace(median_bar)


if __name__ == '__main__':
    if not os.path.isdir('html_output'):
        os.mkdir('html_output')

    pd.options.plotting.backend = "plotly"
    plot_experiment = True

    conditions_to_consider = ['L_4_8', 'L_4_0', 'L_4_-8', 'L_2_-8', 'L_0_-8', 'N_0_0', 'R_0_8', 'R_-2_8', 'R_-4_8', 'R_-4_0', 'R_-4_-8']

    global_traces, global_metrics, individual_traces, individual_metrics = load_experiment_data(conditions_to_consider)
    rename_to_paper_conventions(global_traces, global_metrics, individual_traces, individual_metrics)
    excluding_collisions = get_metrics_excluding_collisions(global_metrics)

    generate_aggregated_velocity_plots(individual_traces, global_metrics)
    generate_who_first_bar(global_metrics)
    generate_initial_action_scatter(global_metrics, individual_metrics)
    generate_3d_boxplot(excluding_collisions)
