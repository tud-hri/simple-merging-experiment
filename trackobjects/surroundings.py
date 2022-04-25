import os
import pickle
import random

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from trackobjects import TunnelMergingTrack
from trackobjects.surroundingobjects import *
from trackobjects.trackside import TrackSide


class Surroundings:
    def __init__(self, track):
        self.track = track

        self.trees = []
        self.rocks = []
        self.markers = []

        self._minimum_tree_distance = 7.
        self._minimum_tunnel_distance = 7.
        self._minimum_rock_distance = 2.
        self._minimum_road_distance = track.track_width * 2.

    def get_graphics_objects(self):
        group = QtWidgets.QGraphicsItemGroup()
        for surrounding_object in self.trees + self.rocks + self.markers:
            pixmap = QtGui.QPixmap(os.path.join('images', surrounding_object.type.file_name()))
            graphics_object = QtWidgets.QGraphicsPixmapItem(pixmap)
            graphics_object.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)

            graphics_object.setPos(surrounding_object.position[0], -surrounding_object.position[1])
            graphics_object.setScale(surrounding_object.scale)
            graphics_object.setRotation(np.degrees(surrounding_object.rotation))

            if isinstance(surrounding_object, Marker):
                graphics_object.setTransformationMode(QtCore.Qt.SmoothTransformation)
                graphics_object.setZValue(0.0)

            group.addToGroup(graphics_object)

        return group

    @staticmethod
    def initialize_randomly(track, tree_spawn_factor=1., rock_spawn_factor=1.0, marker_interval=10.):
        surroundings = Surroundings(track)

        x1, y1, x2, y2 = track.get_track_bounding_rect()
        x_width = (x2 - x1)
        x_center = x1 + x_width / 2.

        position_x_bounds = (x_center - x_width, x_center + x_width)
        position_y_bounds = (y1, y2)

        number_of_trees = int(tree_spawn_factor * track.total_distance / 4.)
        number_of_rocks = int(rock_spawn_factor * track.total_distance / 8.)

        if isinstance(track, TunnelMergingTrack):
            number_of_trees = int(number_of_trees / 2.)
            number_of_rocks = int(number_of_rocks / 2.)

        if isinstance(track, TunnelMergingTrack):
            number_of_markers = int((4 / 3) * track.total_distance / marker_interval)
        else:
            number_of_markers = int(*track.total_distance / marker_interval)

        for tree_index in range(number_of_trees):
            tree = Tree()
            tree.position = surroundings._generate_random_position(position_x_bounds, position_y_bounds)

            tree.scale_factor = random.uniform(0.5, 1.5)
            tree.rotation = random.uniform(-np.pi, np.pi)

            surroundings.trees.append(tree)

        for rock_index in range(number_of_rocks):
            rock = Rock()
            rock.position = surroundings._generate_random_position(position_x_bounds, position_y_bounds)

            rock.scale_factor = random.uniform(0.5, 1.5)

            surroundings.rocks.append(rock)

        if isinstance(track, TunnelMergingTrack):
            marker_distance = -track._section_length
        else:
            marker_distance = 0.

        for marker_index in range(number_of_markers):
            for side in TrackSide:
                track_position = track.traveled_distance_to_coordinates(marker_distance, track_side=side)
                a = (np.pi / 2.) - track.get_heading(track_position)
                w = 1.2 * (track.track_width / 2.)

                left_position = track_position - np.array([np.cos(a) * w, -np.sin(a) * w])
                right_position = track_position + np.array([np.cos(a) * w, -np.sin(a) * w])

                white_marker = Marker(MarkerType.WHITE)
                white_marker.position = left_position
                white_marker.rotation = a

                red_marker = Marker(MarkerType.RED)
                red_marker.position = right_position
                red_marker.rotation = a

                surroundings.markers.append(white_marker)
                surroundings.markers.append(red_marker)

            marker_distance += marker_interval

        return surroundings

    def _generate_random_position(self, position_x_bounds, position_y_bounds):
        position_x = random.uniform(position_x_bounds[0], position_x_bounds[1])
        position_y = random.uniform(position_y_bounds[0], position_y_bounds[1])
        position = np.array([position_x, position_y])

        while self._object_is_too_close(position):
            position_x = random.uniform(position_x_bounds[0], position_x_bounds[1])
            position_y = random.uniform(position_y_bounds[0], position_y_bounds[1])
            position = np.array([position_x, position_y])

        return position

    def _object_is_too_close(self, new_position):
        tree_booleans = [np.linalg.norm(t.position - new_position) < self._minimum_tree_distance for t in self.trees]
        rock_booleans = [np.linalg.norm(r.position - new_position) < self._minimum_rock_distance for r in self.rocks]
        track_boolean = [self.track.closest_point_on_route(new_position)[1] < self._minimum_road_distance]

        if isinstance(self.track, TunnelMergingTrack):
            left_exit, _ = self.track.tunnel_exit_points()
            tunnel_boolean = [new_position[1] < left_exit[1] + self._minimum_tunnel_distance]
        else:
            tunnel_boolean = []

        return any(tree_booleans + rock_booleans + track_boolean + tunnel_boolean)

    @staticmethod
    def load_from_file(file_name):
        with open(os.path.join('data', file_name), 'rb') as f:
            surroundings_data = pickle.load(f)

        surroundings = Surroundings(surroundings_data['track'])

        surroundings.trees = surroundings_data['trees']
        surroundings.rocks = surroundings_data['rocks']
        surroundings.markers = surroundings_data['markers']

        return surroundings

    def save_to_file(self, file_name):
        save_dict = {'track': self.track,
                     'trees': self.trees,
                     'rocks': self.rocks,
                     'markers': self.markers}

        with open(file_name, 'wb') as f:
            pickle.dump(save_dict, f)
