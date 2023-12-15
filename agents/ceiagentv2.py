import copy
import math
import time
import warnings

import numpy as np
import colorednoise as cn
import casadi

from controllableobjects import ControllableObject
from trackobjects.trackside import TrackSide
from .agent import Agent


class CEIAgentV2(Agent):
    """
    An agent used in a Communication-Enabled Interaction model. This version is altered to better match empirical human data.
    """

    def __init__(self, controllable_object: ControllableObject, track_side: TrackSide, dt, sim_master, track, risk_bounds, vehicle_width,
                 vehicle_length, preferred_velocity, time_horizon, belief_frequency, theta, memory_length, saturation_time,
                 max_comfortable_acceleration=1.5, use_noise=False, use_incentive=False, incentive_upper_headway=0., incentive_upper_dv=0.,
                 incentive_upper_interaction=0., incentive_lower_headway=0., incentive_lower_dv=0., incentive_lower_interaction=0., sigma_factor=3.):
        self.controllable_object = controllable_object
        self.track_side = track_side
        self.dt = dt
        self.sim_master = sim_master
        self.track = track
        self._risk_bounds = risk_bounds
        self.theta = theta
        self.saturation_time = saturation_time
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.preferred_velocity = preferred_velocity
        self.last_velocity = preferred_velocity
        self.time_horizon = time_horizon
        self.belief_frequency = belief_frequency
        self.time_of_low_risk = np.inf
        self.memory_length = memory_length
        self.optimization_failed = False
        self.use_noise = use_noise
        self.use_incentive = use_incentive
        self.incentive_upper_headway = incentive_upper_headway
        self.incentive_upper_dv = incentive_upper_dv
        self.incentive_upper_interaction = incentive_upper_interaction
        self.incentive_lower_headway = incentive_lower_headway
        self.incentive_lower_dv = incentive_lower_dv
        self.incentive_lower_interaction = incentive_lower_interaction
        self.sigma_factor = sigma_factor

        # the action plan consists of the action (constant acceleration) to take at the coming time steps. The position plan is the set of positions along the
        # track where the ego vehicle will end up when taking these actions.
        self.action_plan = 0.0
        self.plan_length = int((1000 / dt) * time_horizon)
        self.velocity_plan = np.array([0.0] * self.plan_length)
        self.position_plan = np.array([0.0] * self.plan_length)
        self.action_bounds = [-1, 1]

        # the belief consists of sets of a mean and standard deviation for a distribution over positions at every time step.
        self.belief = []
        self.belief_time_stamps = []
        self.belief_point_contributing_to_risk = []
        for belief_index in range(int(belief_frequency * time_horizon)):
            self.belief.append([0., 0.])

        self.recent_acceleration_observations = np.array([0.] * self.memory_length * int(1000 / self.dt))

        self.did_plan_update_on_last_tick = 0
        self.perceived_risk = 0.
        self.projected_headway = 0.
        self.delta_v = 0.

        # the sigma of the likelihood function is fixed and assumed based on the bound of comfortable acceleration (Hoberock 1977)
        self.max_comfortable_acceleration = max_comfortable_acceleration  # [m/s^2]

        # The observed communication is the current position, velocity, and acceleration of the other vehicle
        self.observed_position = 0.0
        self.observed_velocity = self.controllable_object.initial_velocity
        self.observed_acceleration = 0.0

        samplerate = int(1000 / dt)
        samples = int(sim_master.simulation_constants.max_time * samplerate / 1000)

        if use_noise:
            self.perception_update_rate = (1 / samplerate) * 0.5  # 0.5 sec ref https://link.springer.com/article/10.3758/s13414-021-02372-4 and
            # https://link.springer.com/article/10.3758/s13414-018-1517-8
        else:
            self.perception_update_rate = 1

        self._is_initialized = False
        self._initialize_optimization()

    def reset(self):
        self.time_of_low_risk = np.inf
        self.action_plan = 0.0
        self.velocity_plan = np.array([0.0] * self.plan_length)
        self.position_plan = np.array([0.0] * self.plan_length)
        self.action_bounds = (-1., 1.)

        # the belief consists of sets of a mean and standard deviation for a distribution over positions at every time step.
        self.belief = []
        self.belief_time_stamps = []
        self.belief_point_contributing_to_risk = []
        for belief_index in range(int(self.belief_frequency * self.time_horizon)):
            self.belief.append([0., 0.])

        self.recent_acceleration_observations = np.array([0.] * self.memory_length * int(1000 / self.dt))

        self.did_plan_update_on_last_tick = 0
        self.perceived_risk = 0.

        # The observed communication is the current velocity of the other vehicle
        self.observed_position = 0.0
        self.observed_velocity = 0.0
        self.observed_acceleration = 0.0
        self._is_initialized = False

        self._initialize_optimization()

    def _observe_communication(self):
        other_position, other_velocity, other_acceleration = self.sim_master.get_current_state(self.track_side.other)

        if self.use_noise:
            noise = np.random.normal(scale=np.sqrt(self.dt / 1000.)) * 0.6
        else:
            noise = 0.
        velocity_update = self.perception_update_rate * (other_velocity - self.observed_velocity) + noise

        self.observed_position = other_position
        self.observed_velocity = self.observed_velocity + velocity_update
        self.observed_acceleration = other_acceleration

    def _initialize_optimization(self):
        # Optimization based on Casadi (Andersson2018), documentation: https://web.casadi.org/docs/, used the opti() stack
        # Optimization steps based on time horizon of belief points
        N = int(self.time_horizon * self.belief_frequency)

        # Initialize decision variables and parameters
        self.optimizer = casadi.Opti()
        self.x = self.optimizer.variable(2, N + 1)  # The state
        self.u = self.optimizer.variable()  # The control input
        total_cost = 0
        self.risk_bound_agent = self.optimizer.parameter()

        # Save to evaluate later on
        self.mu = []
        self.sd = []
        self.inequality = []

        for i in range(N + 1):
            # Dynamics, use as equality constraint
            x_now = self.x[:, i]

            if i < N:
                x_new = self._dynamics_casadi(x_now, self.u)
                self.optimizer.subject_to(self.x[:, i + 1] == x_new)

            # Cost evaluation
            total_cost += self._cost_function_casadi(x_now, self.u)

            # Inequality constraints
            if i > 0:
                mu_now = self.optimizer.parameter()
                sd_now = self.optimizer.parameter()
                lb, ub = self.track.get_collision_bounds_approximation_casadi(x_now[1])

                inequality_constraint = self._get_double_normal_probability(mu_now, sd_now, lb, ub, self.sigma_factor) - self.risk_bound_agent
                self.optimizer.subject_to(inequality_constraint <= 0)

                # Save for later evaluation
                self.mu.append(mu_now)
                self.sd.append(sd_now)
                self.inequality.append(inequality_constraint)

        # Set bounds
        self.optimizer.subject_to(self.u >= self.action_bounds[0])
        self.optimizer.subject_to(self.u <= self.action_bounds[1])

        # Set initial x
        self.x_initial = casadi.vertcat(self.optimizer.parameter(), self.optimizer.parameter())
        self.optimizer.subject_to(self.x[:, 0] == self.x_initial)

        # Set cost function
        self.optimizer.minimize(total_cost)

    @staticmethod
    def _get_double_normal_probability(mu, sigma, lower_bound, upper_bound, multiplication):
        # Compute the probability of collision
        p = (1 / 2) * (casadi.erf((upper_bound - mu) / (sigma * casadi.sqrt(2))) - casadi.erf((lower_bound - mu) / (sigma * casadi.sqrt(2))))
        # Compute and add the probability of getting to close
        p += (1 / 2) * (casadi.erf((upper_bound - mu) / (sigma * multiplication * casadi.sqrt(2))) - casadi.erf(
            (lower_bound - mu) / (sigma * multiplication * casadi.sqrt(2))))
        p /= 2.
        return p

    def _dynamics_casadi(self, x, u):
        # Equality constraints for the optimization
        alpha = self.controllable_object.resistance_coefficient
        beta = self.controllable_object.constant_resistance
        x1 = x[0]
        x2 = x[1]
        x1_dot = self.controllable_object.max_acceleration * u - alpha * x1 ** 2 - beta
        x2_dot = x1
        dt = 1 / self.belief_frequency
        x1_new = x1 + x1_dot * dt
        x2_new = x2 + x2_dot * dt + 1 / 2 * x1_dot * dt ** 2
        return casadi.vertcat(x1_new, x2_new)

    def _cost_function_casadi(self, x, u):
        # Cost function for the optimization
        desired_velocity = self.preferred_velocity
        current_velocity = x[0]
        alpha = self.controllable_object.resistance_coefficient
        beta = self.controllable_object.constant_resistance
        net_acceleration = self.controllable_object.max_acceleration * u - alpha * current_velocity ** 2 - beta
        return (current_velocity - desired_velocity) ** 2 + net_acceleration ** 2

    def _initialize_belief(self):
        if self.observed_position is None or self.observed_velocity is None:
            # no other vehicle exists, this can be approximated by assuming the other vehicle is stationary at 0.0
            observed_position = 0.0
            observed_velocity = 0.0

        upper_velocity_bound = lower_velocity_bound = self.observed_velocity
        upper_position_bound = lower_position_bound = self.observed_position

        for belief_index in range(len(self.belief)):
            upper_position_bound += upper_velocity_bound * (1 / self.belief_frequency) + (self.controllable_object.max_acceleration / 2.) * (
                    1 / self.belief_frequency) ** 2
            upper_velocity_bound += self.controllable_object.max_acceleration * (1 / self.belief_frequency)

            new_lower_position_bound = lower_position_bound + lower_velocity_bound * (1 / self.belief_frequency) + (
                    -self.controllable_object.max_acceleration / 2.) * (
                                               1 / self.belief_frequency) ** 2
            if new_lower_position_bound >= lower_position_bound:
                lower_position_bound = new_lower_position_bound

            lower_velocity_bound -= self.controllable_object.max_acceleration * (1 / self.belief_frequency)

            if lower_velocity_bound < 0.:
                lower_velocity_bound = 0.

            mean = ((upper_position_bound - lower_position_bound) / 2.) + lower_position_bound
            sd = (upper_position_bound - mean) / 3

            self.belief[belief_index][0] = mean
            self.belief[belief_index][1] = sd
            self.belief_time_stamps.append((1 / self.belief_frequency) * (belief_index + 1))

    def _update_belief(self):
        new_belief = []

        self.recent_acceleration_observations = np.roll(self.recent_acceleration_observations, 1)
        self.recent_acceleration_observations[0] = self.observed_acceleration

        acceleration_mu = sum(self.recent_acceleration_observations) / len(self.recent_acceleration_observations)
        acceleration_sigma = np.sqrt(
            sum((self.recent_acceleration_observations - acceleration_mu) ** 2) / len(self.recent_acceleration_observations)) + \
                             (self.max_comfortable_acceleration / 3.)

        for belief_point_index in range(len(self.belief)):
            # update observations
            time_to_point = self.belief_time_stamps[belief_point_index]

            belief_point_mu = ((time_to_point ** 2) / 2) * acceleration_mu + self.observed_position + self.observed_velocity * time_to_point
            belief_point_sigma = ((time_to_point ** 2) / 2) * acceleration_sigma

            new_belief.append([belief_point_mu, belief_point_sigma])

        self.belief = new_belief

    def _get_incentive(self):
        own_position, own_velocity, _ = self.sim_master.get_current_state(self.track_side)

        if self.observed_position is not None and own_position is not None and self.observed_velocity is not None and own_velocity is not None:
            headway = own_position - self.observed_position
            dv = own_velocity - self.observed_velocity

            upper_incentive = self.incentive_upper_headway * headway + self.incentive_upper_dv * dv + self.incentive_upper_interaction * headway * dv
            lower_incentive = self.incentive_lower_headway * headway + self.incentive_lower_dv * dv + self.incentive_lower_interaction * headway * dv
        else:
            print('WARNING: Incentive calculation failed')
            lower_incentive = upper_incentive = 0.

        return lower_incentive, upper_incentive

    def _evaluate_risk(self):
        max_risk, risk_per_point = self._get_collision_probability(self.belief, self.position_plan)
        self.belief_point_contributing_to_risk = [bool(p) for p in risk_per_point]
        return max_risk

    def _get_collision_probability(self, belief, position_plan):
        probabilities_over_plan = []

        for belief_index, belief_point in enumerate(belief):
            time_from_now = self.belief_time_stamps[belief_index]
            plan_index = int(time_from_now / (self.dt / 1000)) - 1

            position_plan_point = position_plan[plan_index]
            lower_bound, upper_bound = self.track.get_collision_bounds_approximation(position_plan_point)

            if lower_bound and upper_bound:
                collision_probability = self._get_double_normal_probability(belief_point[0], belief_point[1], lower_bound, upper_bound,
                                                                            self.sigma_factor)
                probabilities_over_plan += [collision_probability]
            else:
                probabilities_over_plan += [0.]

        return np.amax(probabilities_over_plan), probabilities_over_plan

    def _update_plan(self, constraint_risk_bound):
        # Set initial values
        initial_state = casadi.vertcat(self.controllable_object.velocity, self.controllable_object.traveled_distance)
        self.optimizer.set_value(self.x_initial, initial_state)

        # Set initial guess
        self.optimizer.set_initial(self.x[:, 0], initial_state)
        plan_indices_for_belief = (np.array(self.belief_time_stamps) / (self.dt / 1000)).astype(int) - 1
        self.optimizer.set_initial(self.x[0, 1:], self.velocity_plan[plan_indices_for_belief])
        self.optimizer.set_initial(self.x[1, 1:], self.position_plan[plan_indices_for_belief])
        self.optimizer.set_initial(self.u, self.action_plan)

        # Give the belief parameters to the solver as parameters
        for belief_index, belief_point in enumerate(self.belief):
            mu, sigma = belief_point

            self.optimizer.set_value(self.mu[belief_index], mu)
            self.optimizer.set_value(self.sd[belief_index], sigma)

        # Set the risk bounds of the agent as a parameter
        self.optimizer.set_value(self.risk_bound_agent, constraint_risk_bound)

        # Solver options
        p_opts = {"expand": True, 'ipopt.print_level': 0, 'print_time': 0}
        s_opts = {"max_iter": 20, "gamma_theta": 0.1, "constr_viol_tol": 1e-8}
        self.optimizer.solver("ipopt", p_opts, s_opts)

        # Solve, if not feasible, still take solution
        try:
            solution = self.optimizer.solve()
            self.action_plan = solution.value(self.u)
            if self.use_noise:
                self.action_plan += abs(np.random.normal(scale=0.1 / 4))

        except RuntimeError as e:
            warnings.warn("Optimization failed produced RuntimeError: ")
            print(e)
            self.optimization_failed = True
            observed_position, _, _ = self.sim_master.get_current_state(self.track_side.other)
            if self.controllable_object.traveled_distance > observed_position:
                self.action_plan = 1.
            else:
                self.action_plan = -1.

        self._update_position_plan()

    def _update_position_plan(self):
        """
        This function updates the current positions and velocity plans. The plan is made in terms of acceleration, but also stored in future positions and
        velocities for convenience. This function uses the current position and velocity and acceleration plan to update these predictions.
        :return:
        """
        previous_position = self.controllable_object.traveled_distance
        previous_velocity = copy.copy(self.controllable_object.velocity)
        acceleration = self.action_plan * self.controllable_object.max_acceleration

        for index in range(self.plan_length):
            previous_position, previous_velocity = self.controllable_object.calculate_time_step_1d(self.dt / 1000., previous_position,
                                                                                                   previous_velocity,
                                                                                                   acceleration,
                                                                                                   self.controllable_object.resistance_coefficient,
                                                                                                   self.controllable_object.constant_resistance)
            self.velocity_plan[index] = previous_velocity
            self.position_plan[index] = previous_position

    def _convert_plan_to_communicative_action(self):
        pass

    def compute_discrete_input(self, dt):
        pass

    def _check_if_crossing_preferred_velocity(self):
        current_velocity = self.controllable_object.velocity

        return (self.last_velocity < self.preferred_velocity <= current_velocity) or \
               (current_velocity <= self.preferred_velocity < self.last_velocity)

    def compute_continuous_input(self, dt):
        if not self._is_initialized:
            self._observe_communication()
            self._initialize_belief()
            self._update_position_plan()
            self._update_plan(constraint_risk_bound=0.8 * self.risk_bounds[1])
            self.perceived_risk = self._evaluate_risk()
            self._is_initialized = True
        else:
            self._observe_communication()
            self._update_belief()
            self._update_position_plan()
            self.perceived_risk = self._evaluate_risk()

            if not self.controllable_object.cruise_control_active:
                if self.perceived_risk < self.risk_bounds[0] and not np.isfinite(self.time_of_low_risk):
                    self.time_of_low_risk = self.sim_master.t

                if self.optimization_failed:
                    self.time_of_low_risk = np.inf
                    self.optimization_failed = False
                    self.did_plan_update_on_last_tick = 3
                    self._update_plan(constraint_risk_bound=0.8 * self.risk_bounds[0])
                    self.perceived_risk = self._evaluate_risk()
                elif self.perceived_risk < self.risk_bounds[0] and (self.sim_master.t - self.time_of_low_risk) > (self.saturation_time * 1000.):
                    self.did_plan_update_on_last_tick = -1
                    self.time_of_low_risk = self.sim_master.t
                    self._update_plan(constraint_risk_bound=0.6 * self.risk_bounds[1])
                    self.perceived_risk = self._evaluate_risk()
                elif self.perceived_risk > self.risk_bounds[1]:
                    self.time_of_low_risk = np.inf
                    self.did_plan_update_on_last_tick = 1
                    self._update_plan(constraint_risk_bound=0.8 * self.risk_bounds[0])
                    self.perceived_risk = self._evaluate_risk()
                elif self._check_if_crossing_preferred_velocity():
                    self.time_of_low_risk = np.inf
                    self.did_plan_update_on_last_tick = 2
                    self._update_plan(constraint_risk_bound=0.6 * self.risk_bounds[1])
                    self.perceived_risk = self._evaluate_risk()
                else:
                    if self.perceived_risk > self.risk_bounds[0]:
                        self.time_of_low_risk = np.inf
                    self.did_plan_update_on_last_tick = 0

        self.last_velocity = self.controllable_object.velocity

        return self.action_plan

    @staticmethod
    def _clip_bound(b):
        return min(max(b, 0.05), 1.0)

    @property
    def risk_bounds(self):
        if self.use_incentive:
            lower_incentive, upper_incentive = self._get_incentive()
            return self._clip_bound(self._risk_bounds[0] + lower_incentive), self._clip_bound(self._risk_bounds[1] + upper_incentive)
        else:
            return self._risk_bounds

    @property
    def _driving_close_to_desired_velocity(self):
        difference = abs(self.controllable_object.velocity - self.preferred_velocity)
        return difference <= 0.1 * self.preferred_velocity

    @property
    def name(self):
        pass
