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
import os
import pickle


def auto_save_experiment(gui_surrounding, sim_master, path):

    save_dict = {'auto_reset': sim_master.auto_reset,
                 'use_collision_punishment': sim_master.use_collision_punishment,
                 'experimental_conditions': sim_master.experimental_conditions,
                 'condition_number': sim_master.condition_number,
                 'training_run_number': sim_master.training_run_number,
                 'number_of_training_runs': sim_master.number_of_training_runs,
                 'end_training_message_is_displayed': sim_master.end_training_message_is_displayed,
                 'is_training': sim_master.is_training,
                 'vehicle_width': sim_master.vehicle_width,
                 'vehicle_length': sim_master.vehicle_length,
                 'simulation_constants': sim_master.simulation_constants,
                 '_vehicles': sim_master._vehicles,
                 '_agents': sim_master._agents,
                 '_track': sim_master._track,
                 '_file_name': sim_master._file_name,
                 '_sub_folder': sim_master._sub_folder,
                 'end_state': sim_master.end_state,
                 'agent_types': sim_master.agent_types,
                 'gui_surrounding': gui_surrounding
                 }

    if sim_master.experimental_conditions is not None:
        save_dict['_experimental_conditions_iterator'] = sim_master._experimental_conditions_iterator

    with open(os.path.join(path, 'auto_save.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)
