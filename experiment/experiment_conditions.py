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
from .conditiondefinition import ConditionDefinition
from .conditionlist import ConditionList


def get_experiment_conditions(simulation_constants):
    N_0_0 = ConditionDefinition.from_merging_point_difference(left_velocity=10.,
                                                              right_velocity=10.,
                                                              left_headway=0.,
                                                              track_section_length=simulation_constants.track_section_length,
                                                              name='N_0_0')

    L_0_m8 = ConditionDefinition.from_merging_point_difference(left_velocity=9.6,
                                                               right_velocity=10.4,
                                                               left_headway=0.,
                                                               track_section_length=simulation_constants.track_section_length,
                                                               name='L_0_-8')
    L_2_m8 = ConditionDefinition.from_merging_point_difference(left_velocity=9.6,
                                                               right_velocity=10.4,
                                                               left_headway=2.,
                                                               track_section_length=simulation_constants.track_section_length,
                                                               name='L_2_-8')
    L_4_m8 = ConditionDefinition.from_merging_point_difference(left_velocity=9.6,
                                                               right_velocity=10.4,
                                                               left_headway=4.,
                                                               track_section_length=simulation_constants.track_section_length,
                                                               name='L_4_-8')
    L_4_0 = ConditionDefinition.from_merging_point_difference(left_velocity=10.,
                                                              right_velocity=10.,
                                                              left_headway=4.,
                                                              track_section_length=simulation_constants.track_section_length,
                                                              name='L_4_0')
    L_4_8 = ConditionDefinition.from_merging_point_difference(left_velocity=10.4,
                                                              right_velocity=9.6,
                                                              left_headway=4.,
                                                              track_section_length=simulation_constants.track_section_length,
                                                              name='L_4_8')

    R_0_8 = ConditionDefinition.from_merging_point_difference(left_velocity=10.4,
                                                              right_velocity=9.6,
                                                              left_headway=0.,
                                                              track_section_length=simulation_constants.track_section_length,
                                                              name='R_0_8')
    R_m2_8 = ConditionDefinition.from_merging_point_difference(left_velocity=10.4,
                                                               right_velocity=9.6,
                                                               left_headway=-2.,
                                                               track_section_length=simulation_constants.track_section_length,
                                                               name='R_-2_8')
    R_m4_8 = ConditionDefinition.from_merging_point_difference(left_velocity=10.4,
                                                               right_velocity=9.6,
                                                               left_headway=-4.,
                                                               track_section_length=simulation_constants.track_section_length,
                                                               name='R_-4_8')
    R_m4_0 = ConditionDefinition.from_merging_point_difference(left_velocity=10.,
                                                               right_velocity=10.,
                                                               left_headway=-4.,
                                                               track_section_length=simulation_constants.track_section_length,
                                                               name='R_-4_0')
    R_m4_m8 = ConditionDefinition.from_merging_point_difference(left_velocity=9.6,
                                                                right_velocity=10.4,
                                                                left_headway=-4.,
                                                                track_section_length=simulation_constants.track_section_length,
                                                                name='R_-4_-8')

    condition_list = ConditionList()
    condition_list.initialize_from_conditions([N_0_0, L_0_m8, L_2_m8, L_4_m8, L_4_0, L_4_8, R_0_8, R_m2_8, R_m4_8, R_m4_0, R_m4_m8], occurrences=10)

    return condition_list
