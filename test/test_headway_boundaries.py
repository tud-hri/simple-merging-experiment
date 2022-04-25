import random
import unittest

import numpy as np
import shapely.affinity
import shapely.geometry

from simulation.simulationconstants import SimulationConstants
from trackobjects import TunnelMergingTrack
from trackobjects.trackside import TrackSide


class TestHeadWayBoundaries(unittest.TestCase):

    @staticmethod
    def _traveled_distance_to_coordinates(distance, section_length, approach_angle, base_distance, vehicle='left'):
        if vehicle == 'left':
            x_axis = -1
        elif vehicle == 'right':
            x_axis = 1

        if distance <= 2 * section_length:
            x = ((base_distance / 2.) - np.cos(approach_angle) * distance) * x_axis
            y = np.sin(approach_angle) * distance
        else:
            x = 0.0
            y = np.sin(approach_angle) * 2 * section_length + (distance - 2 * section_length)
        return np.array([x, y])

    def _plot_result(self, vehicle_left, vehicle_right):
        from matplotlib import pyplot
        from descartes import PolygonPatch

        fig = pyplot.figure()
        ax = fig.add_subplot(111)

        ax.add_patch(PolygonPatch(vehicle_left, fc='#990000', alpha=0.7))
        ax.add_patch(PolygonPatch(vehicle_right, fc='#000099', alpha=0.7))
        ax.autoscale(enable=True)
        pyplot.show()

    def _test_method(self, head_way, average_travelled_distance, should_collide):
        left_coordinates = self._traveled_distance_to_coordinates(average_travelled_distance + head_way / 2., self.section_length, self.approach_angle,
                                                                  self.base_distance, vehicle='left')
        right_coordinates = self._traveled_distance_to_coordinates(average_travelled_distance - head_way / 2., self.section_length, self.approach_angle,
                                                                   self.base_distance, vehicle='right')

        if average_travelled_distance + head_way / 2. <= 2 * self.section_length:
            translated_vehicle_left = shapely.affinity.translate(self.rotated_vehicle_left, left_coordinates[0], left_coordinates[1])
        else:
            translated_vehicle_left = shapely.affinity.translate(self.vehicle_left, left_coordinates[0], left_coordinates[1])

        if average_travelled_distance - head_way / 2. <= 2 * self.section_length:
            translated_vehicle_right = shapely.affinity.translate(self.rotated_vehicle_right, right_coordinates[0], right_coordinates[1])
        else:
            translated_vehicle_right = shapely.affinity.translate(self.vehicle_right, right_coordinates[0], right_coordinates[1])

        intersects = translated_vehicle_left.intersects(translated_vehicle_right)

        if should_collide:
            if not intersects:
                print('should collide')
                print(average_travelled_distance - 2 * self.section_length)
                print(self.track.get_headway_bounds(average_travelled_distance, self.vehicle_width, self.vehicle_length))
                print(self.vehicle_length)
                self._plot_result(translated_vehicle_left, translated_vehicle_right)
            self.assertTrue(intersects)
        else:
            if intersects:
                print('should not collide')
                print(average_travelled_distance - 2 * self.section_length)
                print(self.track.get_headway_bounds(average_travelled_distance, self.vehicle_width, self.vehicle_length))
                print(self.vehicle_length)
                self._plot_result(translated_vehicle_left, translated_vehicle_right)
            self.assertFalse(intersects)

    def setUp(self):
        self.section_length = random.uniform(10.0, 100.)
        self.base_distance = random.uniform(0.3 * self.section_length, self.section_length - .1)

        self.approach_angle = np.arccos((self.base_distance / 2) / (2 * self.section_length))

        self.vehicle_length = random.uniform(3., 8.)
        self.vehicle_width = random.uniform(self.vehicle_length / 2., self.vehicle_length)

        simulation_constants = SimulationConstants(dt=50,
                                                   vehicle_width=self.vehicle_width,
                                                   vehicle_length=self.vehicle_length,
                                                   track_start_point_distance=self.base_distance,
                                                   track_section_length=self.section_length,
                                                   max_time=30e3)

        self.track = TunnelMergingTrack(simulation_constants)

        self.bound_offset = 0.002

        self.vehicle_left = shapely.geometry.box(-self.vehicle_width / 2.0, -self.vehicle_length / 2.0, self.vehicle_width / 2.0, self.vehicle_length / 2.0)
        self.vehicle_right = shapely.geometry.box(-self.vehicle_width / 2.0, -self.vehicle_length / 2.0, self.vehicle_width / 2.0, self.vehicle_length / 2.0)

        self.rotated_vehicle_left = shapely.affinity.rotate(self.vehicle_left, -((np.pi / 2) - self.approach_angle), use_radians=True)
        self.rotated_vehicle_right = shapely.affinity.rotate(self.vehicle_right, (np.pi / 2) - self.approach_angle, use_radians=True)

    def test_conversions(self):
        l = self.vehicle_width / 2.
        for n in range(1000):
            distance = random.uniform(0.0, 3 * self.section_length)
            coordinates = self.track.traveled_distance_to_coordinates(distance, TrackSide.LEFT)
            distance_new = self.track.coordinates_to_traveled_distance(coordinates)
            self.assertAlmostEqual(distance, distance_new)

            coordinates_right = self.track.traveled_distance_to_coordinates(distance, track_side=TrackSide.RIGHT)
            distance_new_right = self.track.coordinates_to_traveled_distance(coordinates_right)
            self.assertAlmostEqual(distance, distance_new_right)

            for track_side in TrackSide:
                for track_section in ['before', 'after']:
                    coordinates_without_l = self.track._traveled_distance_to_coordinates_forced(distance, track_side=track_side,
                                                                                                before_or_after_merge=track_section)
                    coordinates_with_l = self.track._traveled_distance_to_coordinates_forced(distance + l, track_side=track_side,
                                                                                             before_or_after_merge=track_section)

                    self.assertAlmostEqual(np.linalg.norm(coordinates_without_l - coordinates_with_l), l,
                                           msg='goes wrong for {%s, %s}' % (track_side, track_section))

            coordinates_without_l = self.track._traveled_distance_to_coordinates_forced(2 * self.section_length + 0.1, track_side=track_side,
                                                                                        before_or_after_merge=track_section)
            coordinates_with_l = self.track._traveled_distance_to_coordinates_forced(2 * self.section_length + 0.1 - l, track_side=track_side,
                                                                                     before_or_after_merge=track_section)

            self.assertAlmostEqual(np.linalg.norm(coordinates_without_l - coordinates_with_l), l, msg='goes wrong for {%s, %s}' % (track_side, track_section))

            coordinates_without_l = self.track._traveled_distance_to_coordinates_forced(2 * self.section_length - 0.1, track_side=track_side,
                                                                                        before_or_after_merge=track_section)
            coordinates_with_l = self.track._traveled_distance_to_coordinates_forced(2 * self.section_length - 0.1 + l, track_side=track_side,
                                                                                     before_or_after_merge=track_section)

            self.assertAlmostEqual(np.linalg.norm(coordinates_without_l - coordinates_with_l), l, msg='goes wrong for {%s, %s}' % (track_side, track_section))

    def test_positive_bound_before_merge(self):
        distance_samples = np.linspace(2 * self.section_length - self.section_length * .5, 2 * self.section_length, 100)

        for distance in distance_samples:
            nb, pb = self.track.get_headway_bounds(distance, self.vehicle_width, self.vehicle_length)
            if pb:
                self._test_method(pb - self.bound_offset, distance, True)
                self._test_method(pb + self.bound_offset, distance, False)

    def test_negative_bound_before_merge(self):
        distance_samples = np.linspace(2 * self.section_length - self.section_length * .5, 2 * self.section_length, 100)

        for distance in distance_samples:
            nb, pb = self.track.get_headway_bounds(distance, self.vehicle_width, self.vehicle_length)

            if nb:
                self._test_method(nb - self.bound_offset, distance, False)
                self._test_method(nb + self.bound_offset, distance, True)

    def test_positive_bound_after_merge(self):
        distance_samples = np.linspace(2 * self.section_length + 0.01, 3 * self.section_length, 100)

        for distance in distance_samples:
            nb, pb = self.track.get_headway_bounds(distance, self.vehicle_width, self.vehicle_length)

            if pb:
                self._test_method(pb - self.bound_offset, distance, True)
                self._test_method(pb + self.bound_offset, distance, False)

    def test_negative_bound_after_merge(self):
        distance_samples = np.linspace(2 * self.section_length + 0.01, 3 * self.section_length, 100)

        for distance in distance_samples:
            nb, pb = self.track.get_headway_bounds(distance, self.vehicle_width, self.vehicle_length)

            if nb:
                self._test_method(nb - self.bound_offset, distance, False)
                self._test_method(nb + self.bound_offset, distance, True)
