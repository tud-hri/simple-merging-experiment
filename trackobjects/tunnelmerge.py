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
import numpy as np
import shapely.affinity
import shapely.geometry
import shapely.ops
from scipy import optimize
from scipy import stats

from trackobjects import SymmetricMergingTrack
from trackobjects.trackside import TrackSide


class TunnelMergingTrack(SymmetricMergingTrack):
    def __init__(self, simulation_constants, track_width=4.):
        super(TunnelMergingTrack, self).__init__(simulation_constants, track_width)

        self._start_point_distance = simulation_constants.track_start_point_distance
        self._section_length = simulation_constants.track_section_length
        self._track_width = track_width
        self._approach_angle = np.arccos((self._start_point_distance / 2) / (self._section_length * 2))

        if not np.pi / 4 < self._approach_angle < np.pi / 2:
            raise ValueError('The approach angle for the symmetric merging track cannot be larger then 45 degree, please decrease the start point distance or '
                             'increase the section length.')

        self._merge_point = np.array([0.0, np.sqrt((self._section_length * 2) ** 2 - (self._start_point_distance / 2) ** 2)])
        self._end_point = np.array([0.0, self._merge_point[1] + self._section_length])

        self._left_way_points = [np.array([-self._start_point_distance / 2., 0.0]), self._merge_point, self._end_point]
        self._left_run_up_point = np.array([- 3 * self._start_point_distance / 4, -np.sqrt(self._section_length ** 2 - (self._start_point_distance / 2) ** 2)])
        self._right_way_points = [np.array([self._start_point_distance / 2., 0.0]), self._merge_point, self._end_point]
        self._right_run_up_point = np.array([3 * self._start_point_distance / 4, -np.sqrt(self._section_length ** 2 - (self._start_point_distance / 2) ** 2)])

        self._lower_bound_threshold = None
        self._upper_bound_threshold = None

        self._upper_bound_approximation_slope = None
        self._upper_bound_approximation_intersect = None
        self._lower_bound_approximation_slope = None
        self._lower_bound_approximation_intersect = None
        self._lower_bound_constant_value = None

        self._initialize_linear_bound_approximation(simulation_constants.vehicle_width, simulation_constants.vehicle_length)

    def _initialize_linear_bound_approximation(self, vehicle_width, vehicle_length):
        self._upper_bound_threshold = 2 * self._section_length - (vehicle_width / 2.) / np.tan((np.pi / 2) - self._approach_angle) - (vehicle_length / 2)
        self._lower_bound_threshold = 2 * self._section_length - (vehicle_width / 2.) / np.tan((np.pi / 2) - self._approach_angle) + (vehicle_length / 2)
        if self._lower_bound_threshold > 2 * self._section_length:
            self._lower_bound_threshold = 2 * self._section_length

        last_point = 3 * self._section_length

        # 10 cm resolution lookup
        entries = [i for i in range(int(self._upper_bound_threshold * 10 + 1), int(last_point * 10))]

        look_up_table = np.zeros((len(entries), 2))

        for index in range(len(entries)):
            travelled_distance = entries[index] / 10.
            look_up_table[index, :] = self.get_collision_bounds(travelled_distance, vehicle_width, vehicle_length)

        self._upper_bound_approximation_slope, self._upper_bound_approximation_intersect, _, _, _ = stats.linregress(np.array(entries) / 10.,
                                                                                                                     look_up_table[:, 1])
        lower_bound_index = np.where(entries > self._lower_bound_threshold * 10)[0][0]

        self._lower_bound_approximation_slope, self._lower_bound_approximation_intersect, _, _, _ = stats.linregress(
            np.array(entries[lower_bound_index:]) / 10.,
            look_up_table[lower_bound_index:, 0])

        self._lower_bound_constant_value, _ = self.get_collision_bounds(self._lower_bound_threshold - 0.1, vehicle_width, vehicle_length)

    def traveled_distance_to_coordinates(self, distance, track_side):
        if distance <= 2 * self._section_length:
            before_or_after = 'before'
        else:
            before_or_after = 'after'

        return self._traveled_distance_to_coordinates_forced(distance, track_side=track_side, before_or_after_merge=before_or_after)

    def _traveled_distance_to_coordinates_forced(self, distance, track_side: TrackSide, before_or_after_merge):
        if track_side is TrackSide.LEFT:
            x_axis = -1
        elif track_side is TrackSide.RIGHT:
            x_axis = 1

        if before_or_after_merge == 'before':
            x = ((self._start_point_distance / 2.) - np.cos(self._approach_angle) * distance) * x_axis
            y = np.sin(self._approach_angle) * distance
        elif before_or_after_merge == 'after':
            x = 0.0
            y = np.sin(self._approach_angle) * 2 * self._section_length + (distance - 2 * self._section_length)
        return np.array([x, y])

    def _coordinates_to_traveled_distance_forced(self, point, track_side: TrackSide, before_or_after_merge):
        if before_or_after_merge == 'after':
            distance = 2 * self._section_length + (point[1] - self._merge_point[1])
        elif before_or_after_merge == 'before':
            if track_side is TrackSide.LEFT:
                distance = np.linalg.norm(point - self._left_way_points[0])
            elif track_side is TrackSide.RIGHT:
                distance = np.linalg.norm(point - self._right_way_points[0])
        return distance

    def get_headway_bounds(self, average_travelled_distance, vehicle_width, vehicle_length):
        """
        Returns the bounds on the headway that spans the set of all collision positions. Assumes both vehicles have the same dimensions.
        returns (None, None) when no collisions are possible.

        This method uses scipy optimize to find the minimal headway without a collision, this is inefficient but this method is only used for plotting purposes.
        In the simulations, please use the get_collision_bounds method, it has a closed form solution.

        :param average_travelled_distance:
        :param vehicle_width:
        :param vehicle_length:
        :return:
        """
        if average_travelled_distance > 2 * self._section_length + vehicle_length / 2.:
            # both vehicles are on the straight section
            return -vehicle_length, vehicle_length
        elif average_travelled_distance < self._upper_bound_threshold:
            # at least one of the vehicles is on the approach on a position where it cannot collide
            return None, None
        else:
            # find the minimal headway (x) where the overlap between the vehicles is negative (no collision) and x is positive
            solution = optimize.minimize(lambda x: abs(x), np.array([0.]), constraints=[{'type': 'ineq',
                                                                            'fun': self._collision_constraint,
                                                                            'args': (average_travelled_distance, vehicle_width, vehicle_length)},
                                                                           {'type': 'ineq',
                                                                            'fun': lambda x: x}])
            headway = solution.x[0]
            # the headway bounds are completely symmetrical
            return -headway, headway

    def _collision_constraint(self, head_way, average_travelled_distance, vehicle_width, vehicle_length):
        left = average_travelled_distance + head_way / 2.
        right = average_travelled_distance - head_way / 2.

        lb, ub = self.get_collision_bounds(left, vehicle_width, vehicle_length)

        if lb is None:
            return 1.
        else:
            return lb - right

    def get_collision_bounds(self, traveled_distance_vehicle_1, vehicle_width, vehicle_length):
        """
        Returns the bounds on the position of the other vehicle that spans the set of all collision positions. Assumes both vehicles have the same dimensions.
        returns (None, None) when no collisions are possible

        :param traveled_distance_vehicle_1:
        :param vehicle_width:
        :param vehicle_length:
        :return:
        """

        # setup path_polygon and other pre-requisites
        a = self._approach_angle
        b = np.pi / 2 - self._approach_angle
        l = vehicle_length / 2
        w = vehicle_width / 2

        straight_part = shapely.geometry.box(-w, self._merge_point[1] - l, w, self._merge_point[1] + self._section_length + l)

        R = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])

        top_left = R @ np.array([-w, l]) + self._merge_point
        top_right = R @ np.array([w, l]) + self._merge_point

        start_point_right = self.traveled_distance_to_coordinates(0.0, track_side=TrackSide.RIGHT)
        bottom_left = R @ np.array([-w, -l]) + start_point_right
        bottom_right = R @ np.array([w, -l]) + start_point_right

        approach_part = shapely.geometry.Polygon([top_left, top_right, bottom_right, bottom_left])

        # setup polygon representing vehicle 1
        vehicle_1 = shapely.geometry.box(-w, -l, w, l)

        if traveled_distance_vehicle_1 <= 2 * self._section_length:
            vehicle_1 = shapely.affinity.rotate(vehicle_1, -b, use_radians=True)

        vehicle_1_position = self.traveled_distance_to_coordinates(traveled_distance_vehicle_1, track_side=TrackSide.LEFT)
        vehicle_1 = shapely.affinity.translate(vehicle_1, vehicle_1_position[0], vehicle_1_position[1])

        # get intersection between polygons
        straight_intersection = straight_part.intersection(vehicle_1)
        approach_intersection = approach_part.intersection(vehicle_1)

        if straight_intersection.is_empty and approach_intersection.is_empty:
            return None, None
        else:
            s_lower_bounds = []
            s_upper_bounds = []
            a_lower_bounds = []
            a_upper_bounds = []

            if not straight_intersection.is_empty:
                exterior_points_straight = np.array(straight_intersection.exterior.coords.xy).T
                for point in exterior_points_straight:
                    lb, ub, _ = self._get_straight_bounds_for_point(point, l)
                    s_lower_bounds += [lb]
                    s_upper_bounds += [ub]

            if not approach_intersection.is_empty:
                exterior_points_approach = np.array(approach_intersection.exterior.coords.xy).T
                for point in exterior_points_approach:
                    lb, ub, _ = self._get_approach_bounds_for_point(point, l)
                    a_lower_bounds += [lb]
                    a_upper_bounds += [ub]

            upper_bounds = s_upper_bounds + a_upper_bounds
            lower_bounds = s_lower_bounds + a_lower_bounds

            upper_bounds = [b for b in upper_bounds if not np.isnan(b)]
            lower_bounds = [b for b in lower_bounds if not np.isnan(b)]

            return min(lower_bounds), max(upper_bounds)

    def _get_straight_bounds_for_point(self, point, l):
        closest_point_on_route_after_merge, _ = self._closest_point_on_route_forced(point, track_side=TrackSide.RIGHT, before_or_after_merge='after')
        traveled_distance_after_merge = self._coordinates_to_traveled_distance_forced(closest_point_on_route_after_merge, track_side=TrackSide.RIGHT,
                                                                                      before_or_after_merge='after')

        after_merge_lb = traveled_distance_after_merge - l
        after_merge_ub = traveled_distance_after_merge + l

        if after_merge_lb < 2 * self._section_length:
            after_merge_lb = 2 * self._section_length
        if after_merge_ub < 2 * self._section_length:
            after_merge_ub = np.nan

        return after_merge_lb, after_merge_ub, closest_point_on_route_after_merge

    def _get_approach_bounds_for_point(self, point, l):
        closest_point_on_route_before_merge, _ = self._closest_point_on_route_forced(point, track_side=TrackSide.RIGHT, before_or_after_merge='before')
        traveled_distance_before_merge = self._coordinates_to_traveled_distance_forced(closest_point_on_route_before_merge, track_side=TrackSide.RIGHT,
                                                                                       before_or_after_merge='before')

        before_merge_lb = traveled_distance_before_merge - l
        before_merge_ub = traveled_distance_before_merge + l

        if before_merge_lb > 2 * self._section_length:
            before_merge_lb = np.nan
        if before_merge_ub > 2 * self._section_length:
            before_merge_ub = 2 * self._section_length

        return before_merge_lb, before_merge_ub, closest_point_on_route_before_merge

    def is_in_tunnel(self, travelled_distance):
        return travelled_distance <= self._section_length

    def tunnel_exit_points(self):
        left_exit = self.traveled_distance_to_coordinates(self._section_length, TrackSide.LEFT)
        right_exit = self.traveled_distance_to_coordinates(self._section_length, TrackSide.RIGHT)

        return left_exit, right_exit

    @property
    def total_distance(self) -> float:
        return self._section_length * 3.
