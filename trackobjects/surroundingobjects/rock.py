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
import enum
import random


class RockType(enum.Enum):
    ROCK_1 = 1
    ROCK_2 = 2
    ROCK_3 = 3

    def __str__(self):
        return {RockType.ROCK_1: 'Rock 1',
                RockType.ROCK_2: 'Rock 2',
                RockType.ROCK_3: 'Rock 3', }[self]

    def file_name(self):
        return {RockType.ROCK_1: 'rock_1.png',
                RockType.ROCK_2: 'rock_2.png',
                RockType.ROCK_3: 'rock_3.png', }[self]

    @staticmethod
    def random():
        return random.choice(list(RockType))


class Rock:
    def __init__(self):
        self.position = [0.0, 0.0]
        self.rotation = 0.0
        self.scale_factor = 1.0
        self.type = RockType.random()

        self._default_scale = 0.004

    @property
    def scale(self):
        return self._default_scale * self.scale_factor
