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
