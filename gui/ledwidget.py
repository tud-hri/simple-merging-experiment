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
import enum

from PyQt5 import QtGui, QtCore, QtWidgets


class LEDColor(enum.Enum):
    RED = 0
    GREEN = 1


class LEDWidget(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(LEDWidget, self).__init__(parent)

        self.red_on_pixmap = QtGui.QPixmap(os.path.join('images', 'led_red_on.png'))
        self.red_on_pixmap_item = QtWidgets.QGraphicsPixmapItem(self.red_on_pixmap)
        self.red_on_pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        self.red_off_pixmap = QtGui.QPixmap(os.path.join('images', 'led_red_off.png'))
        self.red_off_pixmap_item = QtWidgets.QGraphicsPixmapItem(self.red_off_pixmap)
        self.red_off_pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        self.green_on_pixmap = QtGui.QPixmap(os.path.join('images', 'led_green_on.png'))
        self.green_on_pixmap_item = QtWidgets.QGraphicsPixmapItem(self.green_on_pixmap)
        self.green_on_pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        self.green_off_pixmap = QtGui.QPixmap(os.path.join('images', 'led_green_off.png'))
        self.green_off_pixmap_item = QtWidgets.QGraphicsPixmapItem(self.green_off_pixmap)
        self.green_off_pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        self._scene = QtWidgets.QGraphicsScene()
        self.setScene(self._scene)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.scene().addItem(self.red_off_pixmap_item)
        self.scene().addItem(self.red_on_pixmap_item)
        self.scene().addItem(self.green_on_pixmap_item)
        self.scene().addItem(self.green_off_pixmap_item)

        self.led_color = LEDColor.GREEN
        self.is_on = False

        background = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        self.setBackgroundBrush(background)
        self.setStyleSheet("border: 0;")

        self._update_pixmap()
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def _update_pixmap(self):
        self.red_on_pixmap_item.setVisible(self.led_color is LEDColor.RED and self.is_on)
        self.red_off_pixmap_item.setVisible(self.led_color is LEDColor.RED and not self.is_on)
        self.green_on_pixmap_item.setVisible(self.led_color is LEDColor.GREEN and self.is_on)
        self.green_off_pixmap_item.setVisible(self.led_color is LEDColor.GREEN and not self.is_on)

    def set_color(self, color: LEDColor):
        self.color = color
        self._update_pixmap()

    def toggle_on_off(self):
        self.is_on = not self.is_on
        self._update_pixmap()

    def set_on(self, boolean):
        self.is_on = boolean
        self._update_pixmap()
