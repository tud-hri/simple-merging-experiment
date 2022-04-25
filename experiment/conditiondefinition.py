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
from trackobjects.trackside import TrackSide


class ConditionDefinition:
    def __init__(self, *, left_initial_velocity, right_initial_velocity, left_initial_position_offset, right_initial_position_offset, name):
        self.left_initial_velocity = left_initial_velocity
        self.right_initial_velocity = right_initial_velocity
        self.left_initial_position_offset = left_initial_position_offset
        self.right_initial_position_offset = right_initial_position_offset
        self.name = name

    def get_initial_velocity(self, side: TrackSide):
        if side is TrackSide.LEFT:
            return self.left_initial_velocity
        elif side is TrackSide.RIGHT:
            return self.right_initial_velocity

    def get_position_offset(self, side: TrackSide):
        if side is TrackSide.LEFT:
            return self.left_initial_position_offset
        elif side is TrackSide.RIGHT:
            return self.right_initial_position_offset

    @staticmethod
    def from_tunnel_exit_difference(*, left_velocity, right_velocity, left_headway, track_section_length, name):
        return ConditionDefinition._from_difference_at_point(left_velocity, right_velocity, left_headway, track_section_length, name)

    @staticmethod
    def from_merging_point_difference(*, left_velocity, right_velocity, left_headway, track_section_length, name):
        return ConditionDefinition._from_difference_at_point(left_velocity, right_velocity, left_headway, 2 * track_section_length, name)

    @staticmethod
    def _from_difference_at_point(left_velocity, right_velocity, left_headway, distance, name):

        if left_headway >= 0.:
            left_headway_at_start = -((left_velocity - right_velocity) / right_velocity) * distance + (left_velocity / right_velocity) * left_headway

            if left_headway_at_start > 0:
                left_initial_position_offset = left_headway_at_start
                right_initial_position_offset = 0.
            else:
                right_headway_at_start = (1 - (right_velocity / left_velocity)) * distance - left_headway

                left_initial_position_offset = 0.
                right_initial_position_offset = right_headway_at_start
        else:
            right_headway_at_start = -((right_velocity - left_velocity) / left_velocity) * distance - (right_velocity / left_velocity) * left_headway

            if right_headway_at_start > 0:
                left_initial_position_offset = 0.
                right_initial_position_offset = right_headway_at_start
            else:
                left_headway_at_start = (1 - (left_velocity / right_velocity)) * distance + left_headway

                left_initial_position_offset = left_headway_at_start
                right_initial_position_offset = 0.

        condition = ConditionDefinition(left_initial_velocity=left_velocity,
                                        right_initial_velocity=right_velocity,
                                        left_initial_position_offset=left_initial_position_offset,
                                        right_initial_position_offset=right_initial_position_offset,
                                        name=name)
        return condition
