from PyQt5 import QtCore, QtGui, QtWidgets


class MouseAgent(QtCore.QObject):

    def __init__(self, name="Mouse agent", parent=None):
        super().__init__(parent)
        self._acceleration_command = 0
        self.app = None
        self._name = name

        self.mouse_pointer_y_location = 0.0

    def reset(self):
        pass

    def compute_discrete_input(self, dt):
        raise NotImplementedError("the mouse agent can only generate continues inputs")

    def compute_continuous_input(self, dt):
        return self._acceleration_command

    def connect_event_listener(self, app: QtWidgets.QApplication):
        app.installEventFilter(self)
        self.app = app

    def eventFilter(self, obj, event):
        if event.type() == QtGui.QMouseEvent.MouseMove:
            self.mouse_pointer_y_location = (event.pos().y() - self.app.allWindows()[0].height()/2) / (self.app.allWindows()[0].height() / 2)
            self._acceleration_command = -self.mouse_pointer_y_location
            return True
        return False

    @property
    def name(self):
        return self._name
