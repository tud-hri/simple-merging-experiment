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
