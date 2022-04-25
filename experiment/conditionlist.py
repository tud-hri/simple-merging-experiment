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

