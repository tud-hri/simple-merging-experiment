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
