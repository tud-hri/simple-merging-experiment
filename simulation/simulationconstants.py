class SimulationConstants:
    """ object that stores all constants needed to recall a saved simulation. """

    def __init__(self, dt, vehicle_width, vehicle_length, track_start_point_distance, track_section_length, max_time):
        self.dt = dt
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.track_start_point_distance = track_start_point_distance
        self.track_section_length = track_section_length
        self.max_time = max_time
