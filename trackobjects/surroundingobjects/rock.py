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
