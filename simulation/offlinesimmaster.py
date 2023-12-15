import copy
import datetime
import os
import pickle

import tqdm

from agents import CEIAgentV2
from trackobjects.trackside import TrackSide
from trackobjects.tunnelmerge import TunnelMergingTrack
from simulation.abstractsimmaster import AbstractSimMaster


class OfflineSimMaster(AbstractSimMaster):
    def __init__(self, track, simulation_constants, file_name, save_to_mat_and_csv=True, verbose=True, experimental_conditions=None,
                 _use_cruise_control_in_tunnel=True, disable_collisions=False):
        super().__init__(track, simulation_constants, file_name, save_to_mat_and_csv=save_to_mat_and_csv)
        self.verbose = verbose

        self.experimental_conditions = experimental_conditions
        self.current_condition = experimental_conditions[0] if experimental_conditions else None
        self._use_cruise_control_in_tunnel = _use_cruise_control_in_tunnel
        self.disable_collisions = disable_collisions

        if verbose:
            self._progress_bar = tqdm.tqdm()
        else:
            self._progress_bar = None

        self._stop = False

    def add_vehicle(self, side: TrackSide, controllable_object, agent):
        self._vehicles[side] = controllable_object
        self._agents[side] = agent
        self.agent_types[side] = type(agent)

        if type(agent) == CEIAgentV2:
            self.risk_bounds[side] = agent.risk_bounds
        else:
            self.risk_bounds[side] = None

    def start(self):
        self._store_current_status()

        while self.t <= self.max_time and not self._stop:
            self.do_time_step()
            self._t += self.dt
            self.time_index += 1
            if self.verbose:
                self._progress_bar.update()

        if not self._stop:
            self.end_state = "Time ran out"

        data_dict = self._save_to_file()
        return data_dict

    def _run_cruise_control_check(self):
        enable_cruise_control = False
        if isinstance(self._track, TunnelMergingTrack):
            if self._track.is_in_tunnel(self._vehicles[TrackSide.LEFT].traveled_distance) or \
                    self._track.is_in_tunnel(self._vehicles[TrackSide.RIGHT].traveled_distance):
                enable_cruise_control = self._use_cruise_control_in_tunnel

        for vehicle in self._vehicles.values():
            vehicle.cruise_control_active = enable_cruise_control

    def do_time_step(self, reverse=False):
        self._run_cruise_control_check()

        for controllable_object, agent in zip(self._vehicles.values(), self._agents.values()):
            if controllable_object.use_discrete_inputs:
                controllable_object.set_discrete_acceleration(agent.compute_discrete_input(self.dt / 1000.0))
            else:
                controllable_object.set_continuous_acceleration(agent.compute_continuous_input(self.dt / 1000.0))

        # This for loop over agents is done twice because the models that compute the new input need the current state of other vehicles.
        # So plan first for all vehicles before applying the accelerations and calculating the new state
        for controllable_object, agent in zip(self._vehicles.values(), self._agents.values()):
            controllable_object.update_model(self.dt / 1000.0)

            if self._track.is_beyond_track_bounds(controllable_object.position):
                self.end_state = "Beyond track bounds"
                self._stop = True
            elif self._track.is_beyond_finish(controllable_object.position):
                self.end_state = "Finished"
                self._stop = True

        lb, ub = self._track.get_collision_bounds(self._vehicles[TrackSide.LEFT].traveled_distance, self.vehicle_width, self.vehicle_length, )
        if lb is not None and ub is not None and not self.disable_collisions:
            try:
                if lb <= self._vehicles[TrackSide.RIGHT].traveled_distance <= ub:
                    self.end_state = "Collided"
                    self._stop = True
            except KeyError:
                # no right side vehicle exists
                pass

        self._store_current_status()
