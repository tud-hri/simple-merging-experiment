from PyQt5 import QtGui, QtCore, QtWidgets
from gui import WorldView

from trackobjects import TunnelMergingTrack
from gui.speeddialgraphicsview import SpeedDialGraphicsView


class ParticipantDialog(QtWidgets.QDialog):
    def __init__(self, track, surroundings=None, show_speed_dial=False, hide_cars_in_tunnel=False, parent=None):
        super(ParticipantDialog, self).__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.resize(1000, 1000)
        self.setWindowTitle("Simple Merging")

        self.layout = QtWidgets.QVBoxLayout()
        self.sim_master = None
        self.show_speed_dial = show_speed_dial
        self.view = WorldView()
        self.view.initialize(track, self, show_run_up=True)
        if surroundings:
            self.view.scene.addItem(surroundings.get_graphics_objects())

        self.view.zoom_level = 0.5
        self.view.update_zoom()

        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

        if self.show_speed_dial:
            self.speed_dial = SpeedDialGraphicsView()
            self.speed_dial.setFixedHeight(200)
            self.layout.addWidget(self.speed_dial)
        else:
            self.speed_dial = None

        self.ego_vehicle = None
        self.other_vehicle = None

        self.is_tunnel_track = isinstance(track, TunnelMergingTrack)
        self.hide_cars_in_tunnel = hide_cars_in_tunnel

        self.show()

    def set_ego_vehicle(self, controllable_object, vehicle_length, vehicle_width, color, ego_track_side):
        self.view.add_controllable_car(controllable_object, vehicle_length, vehicle_width, color)
        self.ego_vehicle = controllable_object

        self.view.zoom_center = QtCore.QPointF(self.ego_vehicle.position[0], -self.ego_vehicle.position[1])
        self.view.update_zoom()

        if self.is_tunnel_track and self.hide_cars_in_tunnel:
            self.view.hide_track_side_in_tunnel(ego_track_side.other, show_run_up=True)

        if self.show_speed_dial:
            self.speed_dial.initialize(0.0, controllable_object.cruise_control_velocity * 2.)

    def set_other_vehicle(self, controllable_object, vehicle_length, vehicle_width, color):
        self.view.add_controllable_car(controllable_object, vehicle_length, vehicle_width, color)
        self.other_vehicle = controllable_object

    def update_all_graphics(self):
        self.view.update_all_graphics_positions()

        if self.show_speed_dial:
            self.speed_dial.set_velocity(self.ego_vehicle.velocity)

        self.view.zoom_center = QtCore.QPointF(self.ego_vehicle.position[0], -self.ego_vehicle.position[1])
        self.view.update_zoom()

    def show_overlay(self, message=None):
        if message:
            self.view.set_overlay_message(message)
            self.view.draw_overlay(True)
        else:
            self.view.draw_overlay(False)
