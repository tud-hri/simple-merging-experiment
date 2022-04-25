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
import cv2
import numpy as np
import datetime
import pyqtgraph
import pickle
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore, QtGui

from .simulation_gui_ui import Ui_SimpleMerging
from trackobjects.trackside import TrackSide


class SimulationGui(QtWidgets.QMainWindow):
    def __init__(self, track, in_replay_mode=True, surroundings=None, parent=None):
        super().__init__(parent)

        self.ui = Ui_SimpleMerging()
        self.ui.setupUi(self)

        self.ui.world_view.initialize(track, self)
        if surroundings:
            self.ui.world_view.scene.addItem(surroundings.get_graphics_objects())

        self.ui.leftSpeedDial.initialize(min_velocity=0.0, max_velocity=20.)
        self.ui.rightSpeedDial.initialize(min_velocity=0.0, max_velocity=20.)

        self.track = track
        self.in_replay_mode = in_replay_mode

        self.ui.play_button.clicked.connect(self.toggle_play)
        self.ui.previous_button.clicked.connect(self.previous_frame)
        self.ui.next_button.clicked.connect(self.next_frame)
        self.ui.next_button.setEnabled(False)
        self.ui.previous_button.setEnabled(False)
        self.ui.timeSlider.setEnabled(in_replay_mode)

        self.ui.timeSlider.sliderReleased.connect(self._set_time)

        self._time_indicator_lines = []
        self._average_distance_indicator_lines = []
        self.average_travelled_distance_trace = None
        self.time_trace = None

        self.video_writer = None
        self.is_recording = False
        self.path_to_video_file = ''

        self.ui.actionEnable_recording.triggered.connect(self._enable_recording)

        self.show()

    def register_sim_master(self, sim_master):
        self.sim_master = sim_master

    def toggle_play(self):
        if self.sim_master and not self.sim_master.main_timer.isActive():
            self.sim_master.start()
            if self.in_replay_mode:
                self.ui.play_button.setText('Pause')
                self.ui.next_button.setEnabled(False)
                self.ui.previous_button.setEnabled(False)
            else:
                self.ui.play_button.setEnabled(False)
        elif self.sim_master:
            self.sim_master.pause()
            self.ui.play_button.setText('Play')
            if self.in_replay_mode:
                self.ui.next_button.setEnabled(True)
                self.ui.previous_button.setEnabled(True)

    def reset(self):
        self.ui.play_button.setText('Play')
        self.ui.play_button.setEnabled(True)
        if self.in_replay_mode:
            self.ui.next_button.setEnabled(True)
            self.ui.previous_button.setEnabled(True)

    def next_frame(self):
        if self.in_replay_mode:
            self.sim_master.do_time_step()

    def previous_frame(self):
        if self.in_replay_mode:
            self.sim_master.do_time_step(reverse=True)

    def add_controllable_dot(self, controllable_object, color=QtCore.Qt.red):
        self.ui.world_view.add_controllable_dot(controllable_object, color)

    def add_controllable_car(self, controllable_object, vehicle_length, vehicle_width, side_for_dial, color='red'):
        self.ui.world_view.add_controllable_car(controllable_object, vehicle_length, vehicle_width, color)

        if side_for_dial is TrackSide.LEFT:
            self.ui.leftSpeedDial.set_velocity(controllable_object.velocity)
        elif side_for_dial is TrackSide.RIGHT:
            self.ui.rightSpeedDial.set_velocity(controllable_object.velocity)

    def update_all_graphics(self, left_velocity, right_velocity):
        self.ui.world_view.update_all_graphics_positions()

        self.ui.leftSpeedDial.set_velocity(left_velocity)
        self.ui.rightSpeedDial.set_velocity(right_velocity)

    def update_time_label(self, time):
        self.ui.statusbar.showMessage('time: %0.2f s' % time)
        if self.in_replay_mode:
            time_promille = int(self.sim_master.time_index * 1000 / self.sim_master.maxtime_index)
            self.ui.timeSlider.setValue(time_promille)
            self._update_trace_plots()

    def show_overlay(self, message=None):
        if message:
            self.ui.world_view.set_overlay_message(message)
            self.ui.world_view.draw_overlay(True)
        else:
            self.ui.world_view.draw_overlay(False)

    def _set_time(self):
        time_promille = self.ui.timeSlider.value()
        self.sim_master.set_time(time_promille=time_promille)

    @staticmethod
    def _add_padding_to_plot_widget(plot_widget, padding=0.1):
        """
        zooms out the view of a plot widget to show 'padding' around the contents of a PlotWidget
        :param plot_widget: The widget to add padding to
        :param padding: the percentage of padding expressed between 0.0 and 1.0
        :return:
        """

        width = plot_widget.sceneRect().width() * (1. + padding)
        height = plot_widget.sceneRect().height() * (1. + padding)
        center = plot_widget.sceneRect().center()
        zoom_rect = QtCore.QRectF(center.x() - width / 2., center.y() - height / 2., width, height)

        plot_widget.fitInView(zoom_rect)

    def initialize_plots(self, left_velocity_trace, right_velocity_trace, time, headway_trace, average_travelled_distance, left_conflict, right_conflict,
                                headway_bounds, left_color, right_color):

        self._initialize_trace_plots(left_velocity_trace, right_velocity_trace, time, headway_trace, average_travelled_distance, left_conflict, right_conflict,
                                headway_bounds, left_color, right_color)

        self.ui.velocityGraphicsView.setTitle('Velocity [m/s]')
        self.ui.headwayGraphicsView.setTitle('Headway [m]')
        self.ui.conflictGraphicsView.setTitle('Level of Conflict [-]')

        self._add_padding_to_plot_widget(self.ui.velocityGraphicsView)
        self._add_padding_to_plot_widget(self.ui.headwayGraphicsView)
        self._add_padding_to_plot_widget(self.ui.conflictGraphicsView)

    def _initialize_trace_plots(self, left_velocity_trace, right_velocity_trace, time, headway_trace, average_travelled_distance, left_conflict, right_conflict,
                                headway_bounds, left_color, right_color):

        self.average_travelled_distance_trace = average_travelled_distance
        self.time_trace = time

        left_pen = pyqtgraph.mkPen(left_color, width=1.5)
        right_pen = pyqtgraph.mkPen(right_color, width=1.5)

        self.ui.velocityGraphicsView.plot(time, left_velocity_trace, pen=left_pen)
        self.ui.velocityGraphicsView.plot(time, right_velocity_trace, pen=right_pen)

        pen = pg.mkPen(0.5, width=3, style=QtCore.Qt.DashLine)
        brush = pg.mkBrush(0.3)
        self.ui.headwayGraphicsView.plot(np.array(headway_bounds['average_travelled_distance'], dtype=float),
                                         np.array(headway_bounds['negative_headway_bound'], dtype=float),
                                         pen=pen, brush=brush, fillLevel=0.)
        self.ui.headwayGraphicsView.plot(np.array(headway_bounds['average_travelled_distance'], dtype=float),
                                         np.array(headway_bounds['positive_headway_bound'], dtype=float),
                                         pen=pen, brush=brush, fillLevel=0.)

        self.ui.headwayGraphicsView.plot(average_travelled_distance, headway_trace, pen=pyqtgraph.mkPen('w', width=1.5))

        self.ui.conflictGraphicsView.plot(time, left_conflict, pen=left_pen)
        self.ui.conflictGraphicsView.plot(time, right_conflict, pen=right_pen)

        for plot_view in [self.ui.conflictGraphicsView, self.ui.velocityGraphicsView]:
            time_indicator_line = pyqtgraph.InfiniteLine(pos=0., pen=pyqtgraph.mkPen('w', width=1.5, style=QtCore.Qt.DashLine))
            self._time_indicator_lines.append(time_indicator_line)
            plot_view.addItem(time_indicator_line)

        distance_indicator_line = pyqtgraph.InfiniteLine(pos=0., pen=pyqtgraph.mkPen('w', width=1.5, style=QtCore.Qt.DashLine))
        self._average_distance_indicator_lines.append(distance_indicator_line)
        self.ui.headwayGraphicsView.addItem(distance_indicator_line)

    def _update_trace_plots(self):
        for time_indicator_line in self._time_indicator_lines:
            time_indicator_line.setValue(self.sim_master.t / 1000.)

        for distance_indicator_line in self._average_distance_indicator_lines:
            index = np.abs(np.array(self.time_trace) - (self.sim_master.t / 1000.)).argmin()
            distance_indicator_line.setValue(self.average_travelled_distance_trace[index])

    def _enable_recording(self):
        if not self.is_recording:
            self.initialize_recording()
        else:
            self.stop_recording()

    def initialize_recording(self):
        file_name = datetime.datetime.now().strftime('-%Y%m%d-%Hh%Mm%Ss.avi')

        self.path_to_video_file = os.path.join('data', 'videos', file_name)
        os.makedirs(os.path.dirname(self.path_to_video_file), exist_ok=True)

        fps = 1 / (self.sim_master.dt / 1000.)

        frame_size = self._get_image_of_current_gui().size()
        self.video_writer = cv2.VideoWriter(self.path_to_video_file, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), fps, (frame_size.width(), frame_size.height()))
        self.is_recording = True
        self.sim_master.enable_recording(True)

    def stop_recording(self):
        self.video_writer.release()
        QtWidgets.QMessageBox.information(self, 'Video Saved', 'A video capture of the visualisation was saved to ' + self.path_to_video_file)
        self.is_recording = False
        self.sim_master.enable_recording(False)
        self.ui.actionEnable_recording.setChecked(False)

    def record_frame(self):
        if self.is_recording:
            image = self._get_image_of_current_gui()
            frame_size = image.size()
            bits = image.bits()

            bits.setsize(frame_size.height() * frame_size.width() * 4)
            image_array = np.frombuffer(bits, np.uint8).reshape((frame_size.height(), frame_size.width(), 4))
            color_convert_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            self.video_writer.write(color_convert_image)

    def _save_screen_shot(self):
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        file_name = os.path.join('data', 'images', time_stamp + '.png')

        image = self._get_image_of_current_gui()
        image.save(file_name)

    def _get_image_of_current_gui(self):
        image = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
        region = QtGui.QRegion(self.rect())

        painter = QtGui.QPainter(image)
        self.render(painter, QtCore.QPoint(), region)
        painter.end()

        return image
