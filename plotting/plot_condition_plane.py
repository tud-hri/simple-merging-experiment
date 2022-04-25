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
import matplotlib.pyplot as plt
import numpy as np


def find_new_conditions_around_point(x0, distance, direction, offset):
    polynomial = np.polynomial.polynomial.Polynomial([(1 + direction ** 2) * x0 ** 2 - distance ** 2, (-2 - 2 * direction ** 2) * x0, 1 + direction ** 2])

    solution = polynomial.roots()

    first_point = np.array([solution[0], solution[0] * direction + offset])
    second_point = np.array([solution[1], solution[1] * direction + offset])

    return first_point, second_point


if __name__ == '__main__':
    fig = plt.figure()
    plt.xlabel('projected $\\Delta x_l$ [m]', usetex=True)
    plt.ylabel('$\\Delta v_l$ [m/s]', usetex=True)

    vehicle_length = 4.5
    end_point = 10.  # used to plot lines that go out of frame

    plt.vlines(0, -end_point, end_point, colors='lightgray', linestyles='dashed', zorder=0.)
    plt.hlines(0, -end_point, end_point, colors='lightgray', linestyles='dashed', zorder=0.)

    plt.vlines(vehicle_length, -end_point, end_point, color='gray')
    plt.vlines(-vehicle_length, -end_point, end_point, color='gray')

    plt.fill_between([-vehicle_length, -end_point], [-end_point, -end_point],[end_point, end_point], color='lightgray')
    plt.fill_between([vehicle_length, end_point], [-end_point, -end_point],[end_point, end_point], color='lightgray')

    l_conditions = np.array([[-4, -2, 0, 2, 4], [-1, -1, -1, -1, -1]])
    r_conditions = np.array([[-4, -2, 0, 2, 4], [1, 1, 1, 1, 1]])

    estimated_inflection_value = 2.5

    estimated_inflection_points = np.array([[-estimated_inflection_value, estimated_inflection_value], [-1, 1]])
    # plt.scatter(estimated_inflection_points[0, :], estimated_inflection_points[1, :], c='k')

    direction_of_estimate = 1. / estimated_inflection_value
    x_array_for_estimate_line = np.array([v / 10. for v in range(-45, 46)])
    estimated_inflection_line = direction_of_estimate * x_array_for_estimate_line

    estimated_inflection_line_plot, = plt.plot(x_array_for_estimate_line, estimated_inflection_line, c='k', zorder=0.)

    direction_of_perpendicular = -1. / direction_of_estimate
    top_perpendicular_offset = 1. - direction_of_perpendicular * estimated_inflection_value
    bottom_perpendicular_offset = -1. + direction_of_perpendicular * estimated_inflection_value

    perpendicular_estimate_top = direction_of_perpendicular * x_array_for_estimate_line + top_perpendicular_offset
    perpendicular_estimate_bottom = direction_of_perpendicular * x_array_for_estimate_line + bottom_perpendicular_offset

    # plt.plot(x_array_for_estimate_line, perpendicular_estimate_top, c='grey', zorder=0.)
    # plt.plot(x_array_for_estimate_line, perpendicular_estimate_bottom, c='grey', zorder=0.)

    # find and plot new conditions:
    x0 = estimated_inflection_value
    top_offset = x0 * direction_of_estimate - direction_of_perpendicular * x0
    bottom_offset = -x0 * direction_of_estimate + direction_of_perpendicular * x0

    # new_conditions = [[0., 0.]]
    #
    # new_conditions += find_new_conditions_around_point(x0, 0.75, direction_of_perpendicular, top_offset)
    # new_conditions += find_new_conditions_around_point(x0, 1.5, direction_of_perpendicular, top_offset)
    #
    # new_conditions += find_new_conditions_around_point(-x0, 0.75, direction_of_perpendicular, bottom_offset)
    # new_conditions += find_new_conditions_around_point(-x0, 1.5, direction_of_perpendicular, bottom_offset)
    #
    # new_conditions = np.array(new_conditions)

    new_conditions = np.array([[0., 0.], [0., -.8], [2., -.8], [4., -.8], [4., 0.], [4., 0.8], [0., .8], [-2., .8], [-4., .8], [-4., 0.], [-4., -0.8]])
    labels = ['N_0_0', 'L_0_-8', 'L_2_-8', 'L_4_-8', 'L_4_0', 'L_4_8', 'R_0_8', 'R_-2_8', 'R_-4_8', 'R_-4_0', 'R_-4_-8']
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

    new_conditions_plot_objects = plt.scatter(new_conditions[:, 0], new_conditions[:, 1], color='k', zorder=100., marker='s')

    for point, text in zip(new_conditions, labels):
        plt.annotate(text, point, bbox=dict(boxstyle="square,pad=0.3", fc="#EEEEEE", ec=color_dict[text], lw=2), xytext=(12, point[1] * 12), textcoords='offset points')

    condition_line_1 = (2 / 5) * x_array_for_estimate_line + (12 / 5)
    condition_line_2 = (2 / 5) * x_array_for_estimate_line + (8 / 5)
    condition_line_3 = (2 / 5) * x_array_for_estimate_line + (4 / 5)
    condition_line_4 = (2 / 5) * x_array_for_estimate_line - (12 / 5)
    condition_line_5 = (2 / 5) * x_array_for_estimate_line - (8 / 5)
    condition_line_6 = (2 / 5) * x_array_for_estimate_line - (4 / 5)

    # plt.plot(x_array_for_estimate_line, condition_line_1, c='lightgrey', linestyle='dashed', linewidth=0.5)
    # plt.plot(x_array_for_estimate_line, condition_line_2, c='lightgrey', linestyle='dashed', linewidth=0.5)
    # plt.plot(x_array_for_estimate_line, condition_line_3, c='lightgrey', linestyle='dashed', linewidth=0.5)
    # plt.plot(x_array_for_estimate_line, condition_line_4, c='lightgrey', linestyle='dashed', linewidth=0.5)
    # plt.plot(x_array_for_estimate_line, condition_line_5, c='lightgrey', linestyle='dashed', linewidth=0.5)
    # constant_conflict, = plt.plot(x_array_for_estimate_line, condition_line_6, c='lightgrey', linestyle='dashed', linewidth=0.5)

    for new_condition in new_conditions:
        print('dx %.3f - dv %.3f' % (new_condition[0], new_condition[1]))

    # plot tunnel exit line
    tunnel_exit_line = (1 / 5) * x_array_for_estimate_line
    # equal_tunnel_exit_plot, = plt.plot(x_array_for_estimate_line, tunnel_exit_line, c='black', linestyle='dashed', linewidth=0.8)

    plt.text(0.5, 0.9, 'Right vehicle merges first',  horizontalalignment='center', verticalalignment='center', transform=fig.get_axes()[0].transAxes,
             bbox=dict(boxstyle="square,pad=0.5", fc="#EEEEEE", ec="#999999", lw=2))
    plt.fill_between(x_array_for_estimate_line, estimated_inflection_line, -10, color='#ADD5F733')
    plt.fill_between(x_array_for_estimate_line, estimated_inflection_line, 10, color='#96ED8933')
    plt.text(0.5, 0.1, 'Left vehicle merges first',  horizontalalignment='center', verticalalignment='center', transform=fig.get_axes()[0].transAxes,
             bbox=dict(boxstyle="square,pad=0.5", fc="#EEEEEE", ec="#999999", lw=2))

    # plt.title('experiment conditions - projected conflict plane')
    fig.axes[0].set_aspect('equal')
    plt.xlim((-5.5, 5.5))
    plt.ylim((-3., 3.))
    # plt.legend([old_conditions, new_conditions_plot_objects, estimated_inflection_line_plot, equal_tunnel_exit_plot, constant_conflict],
    #            ['Old experiment conditions', 'New experiment conditions', 'Estimated inflection line', 'Equal tunnel exit', 'Lines of constant conflict'])
    # plt.legend([new_conditions_plot_objects, estimated_inflection_line_plot],
    #            ['Experimental conditions', 'Estimated inflection line'])

    plt.show()
