from agents import CEIAgentV2
from controllableobjects import PointMassObject
from simulation.simulationconstants import SimulationConstants
from trackobjects.trackside import TrackSide

simulation_constants: SimulationConstants


class FakeSimMasterForBeliefConstruction:
    def __init__(self, simulation_constants, positions, velocities, net_accelerations):
        self.t = 0.
        self.time_index = 0

        self.simulation_constants = simulation_constants
        self.positions = positions
        self.velocities = velocities
        self.net_accelerations = net_accelerations

    def get_current_state(self, side: TrackSide):
        return self.positions[side][self.time_index], self.velocities[side][self.time_index], self.net_accelerations[side][self.time_index - 1]


def append_experiment_data_with_belief(experiment_data, saturation_times, time_horizons, memory_lengths, belief_frequencies,
                                       expected_accelerations, use_incentive=False):
    agents = {}
    vehicles = {}

    sim_master = FakeSimMasterForBeliefConstruction(experiment_data['simulation_constants'],
                                                    experiment_data['travelled_distance'],
                                                    experiment_data['velocities'],
                                                    experiment_data['net_accelerations'])

    simulation_constants = experiment_data['simulation_constants']
    experiment_data['collision_probabilities'] = {}

    for side in TrackSide:
        vehicles[side] = PointMassObject(experiment_data['track'],
                                         initial_position=[0., 0.],
                                         initial_velocity=0.,
                                         use_discrete_inputs=False,
                                         cruise_control_velocity=0.,
                                         resistance_coefficient=0.005, constant_resistance=0.5)

        agents[side] = CEIAgentV2(controllable_object=vehicles[side],
                                  track_side=side,
                                  dt=simulation_constants.dt,
                                  sim_master=sim_master,
                                  track=experiment_data['track'],
                                  risk_bounds=(0., 1.),
                                  vehicle_width=simulation_constants.vehicle_width,
                                  vehicle_length=simulation_constants.vehicle_length,
                                  preferred_velocity=experiment_data['velocities'][side][0],
                                  time_horizon=time_horizons[side],
                                  belief_frequency=belief_frequencies[side],
                                  theta=1.,
                                  memory_length=memory_lengths[side],
                                  saturation_time=saturation_times[side],
                                  max_comfortable_acceleration=expected_accelerations[side],
                                  use_incentive=use_incentive)

        agents[side]._initialize_belief()
        experiment_data['beliefs'][side] = [None] * len(experiment_data['velocities'][TrackSide.LEFT])
        experiment_data['position_plans'][side] = [None] * len(experiment_data['velocities'][TrackSide.LEFT])
        experiment_data['action_plans'][side] = [None] * len(experiment_data['velocities'][TrackSide.LEFT])
        experiment_data['perceived_risks'][side] = [None] * len(experiment_data['velocities'][TrackSide.LEFT])
        experiment_data['risk_bounds'][side] = (0., 0.)
        experiment_data['collision_probabilities'][side] = [None] * len(experiment_data['velocities'][TrackSide.LEFT])

    time = [t * simulation_constants.dt for t in range(1, len(experiment_data['velocities'][TrackSide.LEFT]))]

    for side in TrackSide:
        experiment_data['beliefs'][side][0] = 0.
        experiment_data['perceived_risks'][side][0] = 0.

    for t in time:
        sim_master.t = t
        sim_master.time_index += 1
        for side, agent in agents.items():
            agent._observe_communication()
            agent._update_belief()

            vehicles[side].traveled_distance = experiment_data['travelled_distance'][side][sim_master.time_index]
            vehicles[side].velocity = experiment_data['velocities'][side][sim_master.time_index]

            agent.action_plan = experiment_data['raw_input'][side][sim_master.time_index]
            agent._update_position_plan()
            perceived_risk = agent._evaluate_risk()
            max_risk, risk_per_point = agent._get_collision_probability(agent.belief, agent.position_plan)

            experiment_data['beliefs'][side][sim_master.time_index] = agent.belief
            experiment_data['position_plans'][side][sim_master.time_index] = agent.position_plan.copy()
            experiment_data['action_plans'][side][sim_master.time_index] = agent.action_plan
            experiment_data['perceived_risks'][side][sim_master.time_index] = perceived_risk
            experiment_data['collision_probabilities'][side][sim_master.time_index] = risk_per_point

    return experiment_data
