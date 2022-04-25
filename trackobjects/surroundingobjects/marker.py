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


class MarkerType(enum.Enum):
    RED = 1
    WHITE = 2

    def __str__(self):
        return {MarkerType.RED: 'Red',
                MarkerType.WHITE: 'White', }[self]

    def file_name(self):
        return {MarkerType.RED: 'red_marker.png',
                MarkerType.WHITE: 'white_marker.png', }[self]


class Marker:
    def __init__(self, type):
        self.position = [0.0, 0.0]
        self.rotation = 0.0
        self.scale_factor = 1.0
        self.type = type

        self._default_scale = 0.0015

    @property
    def scale(self):
        return self._default_scale * self.scale_factor
