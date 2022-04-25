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


class TrackSide(enum.Enum):
    LEFT = 0
    RIGHT = 1

    @property
    def other(self):
        return {TrackSide.LEFT: TrackSide.RIGHT,
                TrackSide.RIGHT: TrackSide.LEFT,}[self]

    def __str__(self):
        return {TrackSide.LEFT: 'left',
                TrackSide.RIGHT: 'right',}[self]
