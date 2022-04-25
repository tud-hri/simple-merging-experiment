import os

from PyQt5 import QtCore, QtWidgets, QtGui


class SpeedDialGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(SpeedDialGraphicsView, self).__init__(parent)
        self.min_velocity = None
        self.max_velocity = None

        self.scene = QtWidgets.QGraphicsScene()
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setScene(self.scene)

        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        background_pixmap = QtGui.QPixmap(os.path.join('images', 'speed_dial_background.png'))
        self.background = QtWidgets.QGraphicsPixmapItem(background_pixmap)
        self.background.setScale(2.4 / background_pixmap.width())
        self.background.setPos(-1.2, -1.1)
        self.scene.addItem(self.background)

        needle_pen = QtGui.QPen()
        needle_pen.setColor(QtCore.Qt.red)
        needle_pen.setWidthF(0.05)

        self.needle = QtWidgets.QGraphicsLineItem(0.0, 0.0, -1.0, 0.0)
        self.needle.setPen(needle_pen)
        self.scene.addItem(self.needle)

        origin_pen = QtGui.QPen()
        origin_pen.setColor(QtCore.Qt.black)
        origin_pen.setWidthF(0.03)

        origin_brush = QtGui.QBrush(QtCore.Qt.red)

        self.origin = QtWidgets.QGraphicsEllipseItem(-0.025, -0.025, 0.05, 0.05)
        self.origin.setPen(origin_pen)
        self.origin.setBrush(origin_brush)
        self.scene.addItem(self.origin)

        view_box_pen = QtGui.QPen()
        view_box_pen.setColor(QtCore.Qt.black)
        view_box_pen.setWidthF(0.01)

        self.view_box = QtWidgets.QGraphicsRectItem(QtCore.QRectF(-1.2, -1.1, 2.4, 1.2))
        self.view_box.setPen(view_box_pen)
        self.scene.addItem(self.view_box)
        self.fitInView(self.view_box)

    def initialize(self, min_velocity, max_velocity):
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        self.set_velocity((max_velocity - min_velocity) / 2)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super(SpeedDialGraphicsView, self).resizeEvent(event)
        self.fitInView(self.view_box, QtCore.Qt.KeepAspectRatio)

    def set_velocity(self, v):
        v_as_ratio = v / (self.max_velocity - self.min_velocity)

        angle = v_as_ratio * 180.
        self.needle.setRotation(angle)
