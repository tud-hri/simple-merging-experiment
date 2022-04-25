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
import random
from .conditiondefinition import ConditionDefinition


class ConditionList(list):
    def __init__(self, size=None):
        super().__init__()

        if size:
            self.randomly_initialize(size)

    def randomly_initialize(self, size, velocity_bounds=(8., 12.), position_bounds=(0., 1.)):
        for index in range(size):
            left_initial_velocity = random.uniform(*velocity_bounds)
            right_initial_velocity = random.uniform(*velocity_bounds)

            left_position_offset = random.uniform(*position_bounds)
            right_position_offset = random.uniform(*position_bounds)

            self.append(ConditionDefinition(left_initial_velocity=left_initial_velocity,
                                            right_initial_velocity=right_initial_velocity,
                                            left_initial_position_offset=left_position_offset,
                                            right_initial_position_offset=right_position_offset))

    def initialize_from_conditions(self, conditions, occurrences=1):
        self.clear()
        list_of_conditions = conditions * occurrences
        random.shuffle(list_of_conditions)
        self.extend(list_of_conditions)

