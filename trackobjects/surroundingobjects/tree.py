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


class TreeType(enum.Enum):
    TREE_1 = 1
    TREE_2 = 2
    TREE_3 = 3
    TREE_4 = 4

    def __str__(self):
        return {TreeType.TREE_1: 'Tree 1',
                TreeType.TREE_2: 'Tree 2',
                TreeType.TREE_3: 'Tree 3',
                TreeType.TREE_4: 'Tree 4', }[self]

    def file_name(self):
        return {TreeType.TREE_1: 'tree_1.png',
                TreeType.TREE_2: 'tree_2.png',
                TreeType.TREE_3: 'tree_3.png',
                TreeType.TREE_4: 'tree_4.png', }[self]

    @staticmethod
    def random():
        return random.choice(list(TreeType))


class Tree:
    def __init__(self):
        self.position = [0.0, 0.0]
        self.rotation = 0.0
        self.scale_factor = 1.0
        self.type = TreeType.random()

        self._default_scale = 0.015

    @property
    def scale(self):
        return self._default_scale * self.scale_factor
