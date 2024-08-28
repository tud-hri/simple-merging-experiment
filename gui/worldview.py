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
import random

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from trackobjects import TunnelMergingTrack
from trackobjects.trackside import TrackSide


class WorldView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_gui = None
        self.road_graphics = None
        self.track = None

        self.scene = QtWidgets.QGraphicsScene()
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setScene(self.scene)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(14, 150, 22)))

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self.horizontal_scale_factor = 0.0
        self.vertical_scale_factor = 0.0

        # Scaled size zoomRect
        self.max_zoom_size = 0.0
        self.min_zoom_size = 0.0
        self.zoom_level = 0.0
        self.zoom_center = None

        # intialize overlay
        self._overlay_message = 'Not started'
        self._draw_overlay = True

        self.controllable_objects = []
        self.graphics_objects = []

        # initialize plan and believe point lists
        self.plan_graphics_objects = []
        self.belief_graphics_objects = []

        # banana graphics
        self._banana = QtGui.QMovie('images/banana-dance-dancing-banana.gif')
        self._banana.frameChanged.connect(self.scene.update)
        self._show_banana = False

        self.is_mirrored = False

    def initialize(self, track, main_gui, show_run_up=False):
        self.main_gui = main_gui
        self.road_graphics = QtWidgets.QGraphicsItemGroup()
        self.track = track

        if isinstance(self.track, TunnelMergingTrack):
            x1, y1, x2, y2 = track.get_track_bounding_rect()
            left_exit, right_exit = self.track.tunnel_exit_points()

            tunnel_pen = QtGui.QPen(QtCore.Qt.NoPen)
            tunnel_brush = QtGui.QBrush(QtGui.QColor(135, 81, 31))

            left_tunnel_background = QtWidgets.QGraphicsRectItem(0.0, 0.0, x2 * 12, left_exit[1] * 3)
            left_tunnel_background.setTransformOriginPoint(x2 * 12 / 2., 0.0)
            left_tunnel_background.setRotation(90-np.degrees(self.track.get_heading(left_exit)))
            left_tunnel_background.setPos(left_exit[0] - x2 * 12 / 2., -left_exit[1])
            left_tunnel_background.setPen(tunnel_pen)
            left_tunnel_background.setBrush(tunnel_brush)

            right_tunnel_background = QtWidgets.QGraphicsRectItem(0.0, 0.0, x2 * 12, right_exit[1] * 3)
            right_tunnel_background.setTransformOriginPoint(x2 * 12 / 2., 0.0)
            right_tunnel_background.setRotation(90 - np.degrees(self.track.get_heading(right_exit)))
            right_tunnel_background.setPos(right_exit[0] - x2 * 12 / 2., -right_exit[1])
            right_tunnel_background.setPen(tunnel_pen)
            right_tunnel_background.setBrush(tunnel_brush)

            self.scene.addItem(left_tunnel_background)
            self.scene.addItem(right_tunnel_background)

            black_tunnel_exit_pixmap = QtGui.QPixmap(os.path.join('images', 'tunnel_black.png'))
            left_tunnel_exit = QtWidgets.QGraphicsPixmapItem(black_tunnel_exit_pixmap)

            transform = QtGui.QTransform()
            scale_factor = self.track.track_width * 2. / black_tunnel_exit_pixmap.width()
            transform.scale(scale_factor, scale_factor)
            transform.rotate(- (np.degrees(self.track.get_heading(left_exit)) + 90))

            left_tunnel_exit.setOffset(-left_tunnel_exit.sceneBoundingRect().width() / 2,
                                       -left_tunnel_exit.sceneBoundingRect().height())
            left_tunnel_exit.setTransform(transform)
            left_tunnel_exit.setPos(left_exit[0], -left_exit[1])
            left_tunnel_exit.setZValue(4.0)
            self.scene.addItem(left_tunnel_exit)

            right_tunnel_exit = QtWidgets.QGraphicsPixmapItem(black_tunnel_exit_pixmap)

            transform = QtGui.QTransform()
            scale_factor = self.track.track_width * 2. / black_tunnel_exit_pixmap.width()
            transform.scale(scale_factor, scale_factor)
            transform.rotate(-(np.degrees(self.track.get_heading(right_exit)) + 90))

            right_tunnel_exit.setOffset(-right_tunnel_exit.sceneBoundingRect().width() / 2,
                                        -right_tunnel_exit.sceneBoundingRect().height())
            right_tunnel_exit.setTransform(transform)
            right_tunnel_exit.setPos(right_exit[0], -right_exit[1])
            right_tunnel_exit.setZValue(4.0)

            self.scene.addItem(right_tunnel_exit)

        for way_point_set in [track.get_way_points(TrackSide.LEFT, show_run_up), track.get_way_points(TrackSide.RIGHT, show_run_up)]:
            road_path = QtWidgets.QGraphicsPathItem()
            road_painter = QtGui.QPainterPath()
            pen = QtGui.QPen()
            pen.setWidthF(track.track_width)
            pen.setColor(QtGui.QColor(50, 50, 50))
            road_path.setPen(pen)

            center_line_path = QtWidgets.QGraphicsPathItem()
            center_line_painter = QtGui.QPainterPath()
            pen = QtGui.QPen()
            pen.setWidthF(0.02)
            pen.setColor(QtGui.QColor(255, 255, 255))
            center_line_path.setPen(pen)

            road_painter.moveTo(way_point_set[0][0], -way_point_set[0][1])
            center_line_painter.moveTo(way_point_set[0][0], -way_point_set[0][1])

            for way_point in way_point_set[1:]:
                road_painter.lineTo(way_point[0], -way_point[1])
                center_line_painter.lineTo(way_point[0], -way_point[1])

            road_path.setPath(road_painter)
            center_line_path.setPath(center_line_painter)
            center_line_path.setZValue(1.0)

            self.road_graphics.addToGroup(road_path)
            self.road_graphics.addToGroup(center_line_path)

        self.road_graphics.setZValue(1.0)
        self.scene.addItem(self.road_graphics)
        padding_rect_size = self.road_graphics.sceneBoundingRect().size() * 4.0
        padding_rect_top_left_x = self.road_graphics.sceneBoundingRect().center().x() - padding_rect_size.width() / 2
        padding_rect_top_left_y = -self.road_graphics.sceneBoundingRect().center().y() - padding_rect_size.height() / 2
        scroll_padding_rect = QtWidgets.QGraphicsRectItem(padding_rect_top_left_x, padding_rect_top_left_y, padding_rect_size.width(),
                                                          padding_rect_size.height())
        scroll_padding_rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.scene.addItem(scroll_padding_rect)

        # Scaled size zoomRect
        self.max_zoom_size = self.road_graphics.sceneBoundingRect().size() * 1
        self.min_zoom_size = self.road_graphics.sceneBoundingRect().size() * 0.01
        self.zoom_level = 0.0
        self.zoom_center = self.road_graphics.sceneBoundingRect().center()
        self.update_zoom()

    def hide_track_side_in_tunnel(self, track_side: TrackSide, show_run_up=False):
        if not isinstance(self.track, TunnelMergingTrack):
            raise ValueError('can only hide vehicles in a tunnel if the track has a tunnel')

        left_exit, right_exit = self.track.tunnel_exit_points()

        hide_pen = QtGui.QPen(QtCore.Qt.NoPen)
        hide_brush = QtGui.QBrush(QtGui.QColor(135, 81, 31))

        if track_side is TrackSide.LEFT:
            start_point = self.track.get_way_points(TrackSide.LEFT, show_run_up)[0]
            x0 = start_point[0] - self.track.track_width
            y0 = -left_exit[1]
            width = left_exit[0] - start_point[0] + 2 * self.track.track_width
            height = left_exit[1] - start_point[1] + self.track.track_width
            angle = 90 - np.degrees(self.track.get_heading(left_exit))
        else:
            start_point = self.track.get_way_points(TrackSide.RIGHT, show_run_up)[0]
            x0 = right_exit[0] - self.track.track_width
            y0 = -right_exit[1]
            width = start_point[0] - right_exit[0] + 2 * self.track.track_width
            height = right_exit[1] - start_point[1] + self.track.track_width
            angle = 90 - np.degrees(self.track.get_heading(right_exit))

        hide_graphics = QtWidgets.QGraphicsRectItem(x0, y0, width, height)
        hide_graphics.setTransformOriginPoint(x0 + width/2, 0.)
        hide_graphics.setRotation(angle)

        hide_graphics.setPen(hide_pen)
        hide_graphics.setBrush(hide_brush)

        hide_graphics.setZValue(3.)

        self.scene.addItem(hide_graphics)

    def add_controllable_dot(self, controllable_object, color=QtCore.Qt.red):
        radius = 0.1
        graphics = QtWidgets.QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)

        graphics.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        graphics.setBrush(color)

        graphics.setZValue(2.0)
        self.scene.addItem(graphics)
        self.controllable_objects.append(controllable_object)
        self.graphics_objects.append(graphics)
        self.update_all_graphics_positions()

    def add_controllable_car(self, controllable_object, vehicle_length, vehicle_width, color='red'):
        """
        adds a controllable car object to the scene, color should be one of {blue, green, purple, red, white, yellow}
        """
        file_name = color.lower().strip() + '_car.png'
        file_path = 'images/' + file_name

        if not os.path.isfile(file_path):
            raise RuntimeError('Car graphics for color: ' + color + ' can not be found. Looking for the file: ' + file_path +
                               '. Please specify a different color.')

        vehicle_pixmap = QtGui.QPixmap(file_path)
        graphics_object = QtWidgets.QGraphicsPixmapItem(vehicle_pixmap)
        graphics_object.setOffset(-vehicle_pixmap.width() / 2, -vehicle_pixmap.height() / 2)
        graphics_object.setTransformationMode(QtCore.Qt.SmoothTransformation)
        graphics_object.setZValue(2.0)

        self.horizontal_scale_factor = vehicle_length / vehicle_pixmap.width()
        self.vertical_scale_factor = vehicle_width * 1.2 / vehicle_pixmap.height()
        # the mirrors on the vehicle graphics account for 20% of the width but are not included in the vehicle_width, hence the 1.2 factor

        self.scene.addItem(graphics_object)
        self.controllable_objects.append(controllable_object)
        self.graphics_objects.append(graphics_object)
        self.update_all_graphics_positions()

    def update_all_graphics_positions(self):
        for controllable_object, graphics_object in zip(self.controllable_objects, self.graphics_objects):
            transform = QtGui.QTransform()
            transform.rotate(-np.degrees(controllable_object.heading))
            transform.scale(self.horizontal_scale_factor, self.vertical_scale_factor)

            graphics_object.setTransform(transform)

            graphics_object.setPos(controllable_object.position[0], -controllable_object.position[1])

    def update_zoom(self):
        # Compute scale factors (in x- and y-direction)
        zoom = (1.0 - self.zoom_level) ** 2
        scale1 = zoom + (self.min_zoom_size.width() / self.max_zoom_size.width()) * (1.0 - zoom)
        scale2 = zoom + (self.min_zoom_size.height() / self.max_zoom_size.height()) * (1.0 - zoom)

        # Scaled size zoomRect
        scaled_w = self.max_zoom_size.width() * scale1
        scaled_h = self.max_zoom_size.height() * scale2

        # Set zoomRect
        view_zoom_rect = QtCore.QRectF(self.zoom_center.x() - scaled_w / 2, self.zoom_center.y() - scaled_h / 2, scaled_w, scaled_h)

        # Set view (including padding)
        self.fitInView(view_zoom_rect, QtCore.Qt.KeepAspectRatio)

    def randomize_mirror(self):
        if bool(random.getrandbits(1)):
            self.scale(-1, 1)
            self.is_mirrored = not self.is_mirrored

    def set_overlay_message(self, message):
        self._overlay_message = message

    def set_dancing_banana(self, boolean):
        self._show_banana = boolean

    def draw_overlay(self, bool):
        self._draw_overlay = bool
        if bool and self._show_banana:
            self._banana.start()
        else:
            self._banana.stop()
        self.scene.update()

    def drawForeground(self, painter, rect):
        if self._draw_overlay:

            if self.is_mirrored:
                painter.scale(-1., 1.)

            if self._show_banana:
                painter.drawPixmap(rect, self._banana.currentPixmap(), QtCore.QRectF(self._banana.currentPixmap().rect()))

            painter.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
            painter.setPen(QtGui.QPen())
            painter.setOpacity(0.3)

            # create rectangle with 20% margin around the edges for smooth panning
            if self.is_mirrored:
                corner = self.mapToScene(QtCore.QPoint(int(1.4 * self.width()), int(-0.2 * self.height())))
            else:
                corner = self.mapToScene(QtCore.QPoint(int(-0.2 * self.width()), int(-0.2 * self.height())))

            painter.drawRect(int(corner.x()), int(corner.y()), int(1.4 * self.width()), int(1.4 * self.height()))

            painter.setOpacity(1.0)
            font = QtGui.QFont()
            font.setPointSize(3)
            font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 130.)

            painter_path = QtGui.QPainterPath()
            painter.setBrush(QtCore.Qt.white)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)

            text_width = QtGui.QFontMetrics(font).horizontalAdvance(self._overlay_message)
            if self.is_mirrored:
                painter_path.addText(-rect.center().x() - text_width / 2, rect.center().y(), font, self._overlay_message)
            else:
                painter_path.addText(rect.center().x() - text_width / 2, rect.center().y(), font, self._overlay_message)

            painter.drawPath(painter_path)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_zoom()

    def wheelEvent(self, event):
        direction = np.sign(event.angleDelta().y())
        self.zoom_level = max(min(self.zoom_level + direction * 0.1, 1.0), 0.0)
        self.update_zoom()

    def enterEvent(self, e):
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        super().enterEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:  # Drag scene
            self.zoom_center = self.mapToScene(self.rect().center())
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.buttons() == QtCore.Qt.MiddleButton:  # Drag scene
            self.main_gui.statusBar.showMessage('position of mouse: %0.1f, %0.1f  -  position of point mass: %0.1f, %0.1f  ' % (
                self.mapToScene(e.pos()).x(), -self.mapToScene(e.pos()).y(), self.controllable_objects[0].position[0],
                -self.controllable_objects[0].position[1]))
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        self.update_zoom()
