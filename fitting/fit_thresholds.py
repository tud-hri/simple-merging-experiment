import glob
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import tqdm

from trackobjects.trackside import TrackSide


def _inverse_condition(condition_name):
    return {'N_0_0': 'N_0_0',
            'L_0_-8': 'R_0_8',
            'L_2_-8': 'R_-2_8',
            'L_4_-8': 'R_-4_8',
            'L_4_0': 'R_-4_0',
            'L_4_8': 'R_-4_-8',
            'R_0_8': 'L_0_-8',
            'R_-2_8': 'L_2_-8',
            'R_-4_8': 'L_4_-8',
            'R_-4_0': 'L_4_0',
            'R_-4_-8': 'L_4_8'}[condition_name]


def estimate_thresholds_based_on_velocity(velocity_threshold=0.5):
    threshold_estimations = {'pair': [],
                             'person': [],
                             'condition': [],
                             'tunnel_exit_headway': [],
                             'dv': [],
                             'limit_reached': [],
                             'side': [],
                             'upper': [],
                             'min_upper': [],
                             'max_upper': [],
                             'lower': []}

    for pair_number in [4, 5, 6, 8, 9, 10, 11, 12, 13]:
    # for pair_number in [11]:
        all_experiment_files = glob.glob(
            os.path.join('..', 'data', 'experiment_data', 'experiment_%d' % pair_number, 'experiment_%d_iter_*.pkl' % pair_number))

        threshold_evidence = {'pair': [],
                              'side': [],
                              'condition': [],
                              'v_at_exit': [],
                              'v_after_one_second': [],
                              'tunnel_exit_headway': [],
                              'dv': [],
                              'deviation': [],
                              'limit_reached': [],
                              }

        for file in tqdm.tqdm(all_experiment_files):
            with open(file, 'rb') as f:
                loaded_data = pickle.load(f)

            if loaded_data['end_state'] == 'Finished':

                one_second_index = int(1000. / loaded_data['simulation_constants'].dt)

                merge_index = np.where((np.array(loaded_data['travelled_distance'][TrackSide.LEFT]) > 100.) | (
                        np.array(loaded_data['travelled_distance'][TrackSide.RIGHT]) > 100.))[0][0]

                exit_index = np.where((np.array(loaded_data['travelled_distance'][TrackSide.LEFT]) > 50.) & (
                        np.array(loaded_data['travelled_distance'][TrackSide.RIGHT]) > 50.))[0][0]

                for side in TrackSide:
                    velocity_trace = np.array(loaded_data['velocities'][side])[exit_index:merge_index]
                    velocity_deviation = velocity_trace - loaded_data['velocities'][side][0]

                    tunnel_exit_headway = loaded_data['travelled_distance'][side][exit_index] - loaded_data['travelled_distance'][side.other][exit_index]
                    dv = loaded_data['velocities'][side][exit_index] - loaded_data['velocities'][side.other][exit_index]
                    v_at_exit = velocity_trace[0]
                    v_after_one_second = velocity_trace[one_second_index]
                    deviation = velocity_deviation[one_second_index]

                    threshold_evidence['pair'].append(pair_number)
                    threshold_evidence['side'].append(side)
                    threshold_evidence['condition'].append(loaded_data['current_condition'].name)
                    threshold_evidence['v_after_one_second'].append(v_after_one_second)
                    threshold_evidence['tunnel_exit_headway'].append(tunnel_exit_headway)
                    threshold_evidence['dv'].append(dv)
                    threshold_evidence['v_at_exit'].append(v_at_exit)
                    threshold_evidence['deviation'].append(deviation)
                    threshold_evidence['limit_reached'].append((abs(velocity_deviation) > velocity_threshold).any())

        threshold_evidence = pd.DataFrame(threshold_evidence)
        folder = os.path.join('..', 'data', 'parameter_effect_grids')

        for condition in threshold_evidence['condition'].unique():
            for side in TrackSide:
                c = condition if side is TrackSide.LEFT else _inverse_condition(condition)
                with open(os.path.join(folder, '%s.pkl' % c), 'rb') as f:
                    model_data = pickle.load(f)

                side_data = threshold_evidence.loc[(threshold_evidence['side'] == side) & (threshold_evidence['condition'] == condition)]

                for index in side_data.index:
                    row = side_data.loc[index, :]
                    limit_reached = row['limit_reached']
                    if limit_reached:
                        model_difference = model_data['deviation'] - row['v_after_one_second']
                        minimum_difference = model_difference.abs().min()
                        if minimum_difference < velocity_threshold:
                            index_of_minimum_difference = model_difference.abs().argmin()
                            all_minima_indices = model_difference[(model_difference == model_difference[index_of_minimum_difference])].index
                            result = model_data.loc[all_minima_indices, :]
                            upper = result['upper'].max()
                            if upper == 0.85:
                                upper = result['upper'].min()
                            min_upper = result['upper'].min()
                            max_upper = result['upper'].max()
                            lower = result['lower'].max()
                        else:
                            upper = None
                            lower = None
                            min_upper = None
                            max_upper = None
                            limit_reached = False
                    else:
                        model_difference = model_data['deviation'] - row['v_at_exit']
                        index_of_minimum_difference = model_difference.abs().argmin()
                        all_minima_indices = model_difference[(model_difference == model_difference[index_of_minimum_difference])].index
                        result = model_data.loc[all_minima_indices, :]
                        upper = result['upper'].min()
                        min_upper = result['upper'].min()
                        max_upper = result['upper'].max()
                        lower = None

                    threshold_estimations['pair'].append(pair_number)
                    threshold_estimations['person'].append('%s-%d' % (side, pair_number))
                    threshold_estimations['condition'].append(condition)
                    threshold_estimations['limit_reached'].append(limit_reached)
                    threshold_estimations['tunnel_exit_headway'].append(row['tunnel_exit_headway'])
                    threshold_estimations['dv'].append(row['dv'])
                    threshold_estimations['side'].append(side)
                    threshold_estimations['upper'].append(upper)
                    threshold_estimations['min_upper'].append(min_upper)
                    threshold_estimations['max_upper'].append(max_upper)
                    threshold_estimations['lower'].append(lower)

    threshold_estimations = pd.DataFrame(threshold_estimations)

    for pair_number in threshold_estimations['pair'].unique():
        pair_data = threshold_estimations.loc[threshold_estimations['pair'] == pair_number]
        for side in TrackSide:
            conditions_where_limit_is_never_reached = []
            side_data = pair_data.loc[(pair_data['side'] == side)]
            for condition in side_data['condition'].unique():
                condition_data = side_data.loc[(side_data['condition'] == condition)]
                if (~condition_data['limit_reached'].to_numpy()).all():
                    conditions_where_limit_is_never_reached.append(condition)
            upper_estimate = side_data.loc[side_data['condition'].isin(conditions_where_limit_is_never_reached), 'upper'].max() * 1.1
            threshold_estimations.loc[(threshold_estimations['pair'] == pair_number) &
                                      (threshold_estimations['side'] == side) &
                                      (threshold_estimations['condition'].isin(conditions_where_limit_is_never_reached)), 'upper'] = upper_estimate

    return threshold_estimations


if __name__ == '__main__':
    results = estimate_thresholds_based_on_velocity()

    fitted_parameters = {}

    model_data = results.loc[results['upper'].notna()]
    model = smf.mixedlm(data=model_data, formula="upper ~ tunnel_exit_headway * dv", groups=model_data['person']).fit()
    fitted_parameters['upper_parameters'] = model.params
    fitted_parameters['upper_random_effects'] = model.random_effects
    print(model.summary())
    print(model.random_effects)

    model_data = results.loc[results['lower'].notna()]
    model = smf.mixedlm(data=model_data, formula="lower ~ tunnel_exit_headway * dv", groups=model_data['person']).fit()
    fitted_parameters['lower_parameters'] = model.params
    fitted_parameters['lower_random_effects'] = model.random_effects
    print(model.summary())
    print(model.random_effects)

    with open(os.path.join('..', 'data', 'fitted_thresholds.pkl'), 'wb') as f:
        pickle.dump(fitted_parameters, f)
