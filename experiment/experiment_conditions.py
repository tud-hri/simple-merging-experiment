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
