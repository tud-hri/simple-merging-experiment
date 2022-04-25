class FakeSimMaster:
    def __init__(self, x0=0.0, v0=0.0):
        self.t = 0.0
        self._other_position = x0
        self._other_velocity = v0

    def update(self, dt):
        self.t += dt
        self._other_position = (dt / 1000.) * self._other_velocity

    def get_current_state(self, track_side):
        return self._other_position, self._other_velocity
