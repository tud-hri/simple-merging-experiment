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
