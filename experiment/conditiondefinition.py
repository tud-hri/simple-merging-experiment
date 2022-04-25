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
