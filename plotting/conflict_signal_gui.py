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
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from shapely.geometry import LineString, Point

from conflict_ui import Ui_MainWindow
from plotting.load_data_from_file import calculate_conflict_resolved_time
from plotting.conflict_signal import calculate_level_of_conflict_signal, determine_critical_points
from trackobjects.trackside import TrackSide


class ConflictGui(QtWidgets.QMainWindow):
    def __init__(self, data_file_path, parent=None):
        super().__init__(parent=parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._current_frame = 0

        with open(data_file_path, 'rb') as f:
            self.data = pickle.load(f)

        with open(os.path.join('..', 'data', 'headway_bounds.pkl'), 'rb') as f:
            self.headway_bounds = pickle.load(f)

        self.headway_data = np.array(self.data['travelled_distance'][TrackSide.LEFT]) - np.array(self.data['travelled_distance'][TrackSide.RIGHT])
        self.average_travelled_distance_trace = (np.array(self.data['travelled_distance'][TrackSide.LEFT]) +
                                                 np.array(self.data['travelled_distance'][TrackSide.RIGHT])) / 2.

        self.slope = np.gradient(self.headway_data, self.average_travelled_distance_trace)

        self._total_frames = len(self.data['velocities'][TrackSide.LEFT]) - 1
        self.ui.timeSlider.setMaximum(self._total_frames)
        self.ui.timeSlider.valueChanged.connect(self._update_current_frame)
        self.ui.criticalPointsCheckBox.stateChanged.connect(self._update_current_frame)
        self.ui.currentAngleCheckBox.stateChanged.connect(self._update_current_frame)
        self.ui.difficultyLinesCheckBox.stateChanged.connect(self._update_current_frame)
        self.ui.predictedCollisionCheckBox.stateChanged.connect(self._update_current_frame)
        self.ui.reachableSpaceCheckBox.stateChanged.connect(self._update_current_frame)

        self.time = [t * self.data['dt'] / 1000 for t in range(self._total_frames + 1)]
        self.crt = calculate_conflict_resolved_time(self.data)

        self.tangent_line_slope = self.slope[0]
        self._tangent_line_intersect = self.headway_data[0]

        self.intersection_point = (np.nan, np.nan)
        self._find_intersection()

        self.critical_points = determine_critical_points(self.headway_bounds, self.data['simulation_constants'])
        self.level_of_conflict = calculate_level_of_conflict_signal(self.average_travelled_distance_trace, self.headway_data, self.critical_points)

        self._plot_collision_block()
        self._plot_data()
        self._frame_indicator, self._tangent_line_graphics = self._initialize_frame_indicator()
        self._intersection_graphics = self._initialize_intersection_point()
        self._critical_points_graphic = self._initialize_critical_points()
        self.ui.graphicsView.disableAutoRange()
        self.ui.graphicsView.autoRange()
        self._reachable_line_graphics = self._initialize_reachable_lines()
        self._difficulty_line_graphics = self._initialize_difficulty_lines()
        self._update_current_frame()


    def _update_current_frame(self):
        self._current_frame = self.ui.timeSlider.value()
        self._frame_indicator.setData([self.average_travelled_distance_trace[self._current_frame]], [self.headway_data[self._current_frame]])
        self.tangent_line_slope = self.slope[self._current_frame]
        self._tangent_line_intersect = self.headway_data[self._current_frame] - self.tangent_line_slope * self.average_travelled_distance_trace[
            self._current_frame]
        self._find_intersection()

        self._intersection_graphics.setVisible(self.intersection_point is not None and self.ui.predictedCollisionCheckBox.isChecked())
        self._tangent_line_graphics.setVisible(self.ui.currentAngleCheckBox.isChecked())
        self._difficulty_line_graphics[TrackSide.LEFT].setVisible(self.ui.difficultyLinesCheckBox.isChecked())
        self._difficulty_line_graphics[TrackSide.RIGHT].setVisible(self.ui.difficultyLinesCheckBox.isChecked())
        self._critical_points_graphic.setVisible(self.ui.criticalPointsCheckBox.isChecked())
        self._reachable_line_graphics[0].setVisible(self.ui.reachableSpaceCheckBox.isChecked())
        self._reachable_line_graphics[1].setVisible(self.ui.reachableSpaceCheckBox.isChecked())

        if self.ui.currentAngleCheckBox.isChecked():
            tangent_line_x = [self.average_travelled_distance_trace[self._current_frame], self.average_travelled_distance_trace[self._current_frame] + 100]
            tangent_line_y = [tangent_line_x[0] * self.tangent_line_slope + self._tangent_line_intersect,
                              tangent_line_x[1] * self.tangent_line_slope + self._tangent_line_intersect]
            self._tangent_line_graphics.setData(tangent_line_x, tangent_line_y)

        if self.intersection_point is not None:
            self._intersection_graphics.setData([self.intersection_point.x], [self.intersection_point.y])

            self.ui.dxDoubleSpinBox.setValue(self.intersection_point.x - self.average_travelled_distance_trace[self._current_frame])
            self.ui.dyDoubleSpinBox.setValue(self.intersection_point.y - self.headway_data[self._current_frame])
        else:
            self.ui.dxDoubleSpinBox.setValue(0.)
            self.ui.dyDoubleSpinBox.setValue(0.)

        if self.ui.difficultyLinesCheckBox.isChecked():
            for side in TrackSide:
                difficulty = self.level_of_conflict[side][self._current_frame]
                if side is TrackSide.RIGHT:
                    difficulty *= -1

                slope = np.tan(difficulty * np.arctan(2) + np.arctan(self.slope[self._current_frame]))
                intersect = self.headway_data[self._current_frame] - slope * self.average_travelled_distance_trace[self._current_frame]

                self._difficulty_line_graphics[side].setData([self.average_travelled_distance_trace[self._current_frame],
                                                              self.average_travelled_distance_trace[self._current_frame] + 200],
                                                             [self.average_travelled_distance_trace[self._current_frame] * slope + intersect,
                                                              (self.average_travelled_distance_trace[self._current_frame] + 200) * slope + intersect])

        if self.ui.reachableSpaceCheckBox.isChecked():
            for index, extra_angle in enumerate([np.arctan(-2), np.arctan(2)]):
                slope = np.tan(np.arctan(self.slope[self._current_frame]) + extra_angle)
                intersect = self.headway_data[self._current_frame] - slope * self.average_travelled_distance_trace[self._current_frame]

                self._reachable_line_graphics[index].setData([self.average_travelled_distance_trace[self._current_frame],
                                                              self.average_travelled_distance_trace[self._current_frame] + 200],
                                                             [self.average_travelled_distance_trace[self._current_frame] * slope + intersect,
                                                              (self.average_travelled_distance_trace[self._current_frame] + 200) * slope + intersect])

        self.ui.currentAngleDoubleSpinBox.setValue(np.arctan(self.slope[self._current_frame]))
        self.ui.rightDifficultyDoubleSpinBox.setValue(self.level_of_conflict[TrackSide.RIGHT][self._current_frame])
        self.ui.leftDifficultyDoubleSpinBox.setValue(self.level_of_conflict[TrackSide.LEFT][self._current_frame])

    def _plot_collision_block(self):
        average_travelled_distance = np.array(self.headway_bounds['average_travelled_distance'], dtype=float)
        negative_headway_bound = np.array(self.headway_bounds['negative_headway_bound'], dtype=float)
        positive_headway_bound = np.array(self.headway_bounds['positive_headway_bound'], dtype=float)

        pen = pg.mkPen(0.5, width=3, style=QtCore.Qt.DashLine)
        brush = pg.mkBrush(0.3)
        self.ui.graphicsView.plot(average_travelled_distance, negative_headway_bound, pen=pen, brush=brush, fillLevel=0.)
        self.ui.graphicsView.plot(average_travelled_distance, positive_headway_bound, pen=pen, brush=brush, fillLevel=0.)

    def _plot_data(self):
        try:
            crt_index = self.time.index(self.crt)
        except ValueError:
            crt_index = -1

        fat_pen = pg.mkPen('r', width=3.)
        thin_pen = pg.mkPen('r', width=1.)

        symbol_pen = pg.mkPen('r')
        symbol_brush = pg.mkBrush('r')

        self.ui.graphicsView.plot(self.average_travelled_distance_trace[:crt_index], self.headway_data[:crt_index], pen=fat_pen)
        self.ui.graphicsView.plot(self.average_travelled_distance_trace[crt_index:], self.headway_data[crt_index:], pen=thin_pen)
        self.ui.graphicsView.plot([self.average_travelled_distance_trace[crt_index]], [self.headway_data[crt_index]], symbol='o', pen=None,
                                  symbolPen=symbol_pen,
                                  symbolBrush=symbol_brush)
        self.ui.graphicsView.setLabel('bottom', 'Average travelled distance [m]')
        self.ui.graphicsView.setLabel('left', 'headway [m]')

    def _initialize_frame_indicator(self):
        frame_indicator = self.ui.graphicsView.plot([], [], symbol='o', pen=None)
        tangent_pen = pg.mkPen('g', width=2.)
        tangent_line_graphics = self.ui.graphicsView.plot([], [], pen=tangent_pen)

        return frame_indicator, tangent_line_graphics

    def _initialize_intersection_point(self):
        symbol_pen = pg.mkPen((0, 255, 0))
        symbol_brush = pg.mkBrush((0, 255, 0, 255))

        intersection_graphics = self.ui.graphicsView.plot([0.0], [0.0], pen=None, symbol='x', symbolSize=15., symbolPen=symbol_pen, symbolBrush=symbol_brush)
        return intersection_graphics

    def _initialize_critical_points(self):
        symbol_pen = pg.mkPen('c')
        symbol_brush = pg.mkBrush('c')
        all_critical_points = np.stack(self.critical_points[TrackSide.LEFT] + self.critical_points[TrackSide.RIGHT])

        critical_point_graphics = self.ui.graphicsView.plot(all_critical_points[:, 0], all_critical_points[:, 1], symbol='o', pen=None, symbolPen=symbol_pen,
                                                            symbolBrush=symbol_brush)
        return critical_point_graphics

    def _initialize_difficulty_lines(self):
        difficulty_lines = {}
        for side in TrackSide:
            line = self.ui.graphicsView.plot([], [], pen=pg.mkPen('b'))
            difficulty_lines[side] = line

        return difficulty_lines

    def _initialize_reachable_lines(self):
        lines = [self.ui.graphicsView.plot([], [], pen=pg.mkPen('y')),
                 self.ui.graphicsView.plot([], [], pen=pg.mkPen('y'))]

        return lines

    def _find_intersection(self):

        c = (self.average_travelled_distance_trace[self._current_frame],
             self.average_travelled_distance_trace[self._current_frame] * self.tangent_line_slope + self._tangent_line_intersect)
        d = (self.average_travelled_distance_trace[self._current_frame] + 100,
             (self.average_travelled_distance_trace[self._current_frame] + 100) * self.tangent_line_slope + self._tangent_line_intersect)
        tangent_line = LineString([c, d])
        upper_bound_np = np.column_stack([self.headway_bounds['average_travelled_distance'], self.headway_bounds['positive_headway_bound']])
        upper_bound_np = upper_bound_np[~np.isnan(upper_bound_np[:, 1]), :]
        lower_bound_np = np.column_stack([self.headway_bounds['average_travelled_distance'], self.headway_bounds['negative_headway_bound']])
        lower_bound_np = lower_bound_np[~np.isnan(lower_bound_np[:, 1]), :]

        bounds_np = np.concatenate((np.flip(lower_bound_np, axis=0), upper_bound_np))
        bounds = LineString(bounds_np)

        intersection = tangent_line.intersection(bounds)

        if intersection.is_empty:
            self.intersection_point = None
        elif isinstance(intersection, Point):
            self.intersection_point = intersection
        else:
            # points in a multipoint object are automatically sorted on their x value, this selects the point with the lowest x value
            self.intersection_point = intersection.geoms[0]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    data_file = os.path.join('..', 'data', 'experiment_data', 'experiment_4', 'experiment_4_iter_7.pkl')

    main_windows = ConflictGui(data_file)
    main_windows.show()

    app.exec_()
