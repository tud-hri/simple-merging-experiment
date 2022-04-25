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
from .agent import Agent


class ZeroAgent(Agent):
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute_discrete_input(self, dt):
        raise NotImplementedError('The PD agent can only calculate a continuous input')

    def compute_continuous_input(self, dt):
        return 0.0

    @property
    def name(self):
        pass
