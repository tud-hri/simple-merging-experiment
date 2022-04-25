from PyQt5 import QtCore, QtGui, QtWidgets


class KeyBoardAgent(QtCore.QObject):

    def __init__(self, name="Keyboard agent", parent=None):
        super().__init__(parent)
        self._steering_command = 0

        self._name = name
        self.up_arrow_pressed = False
        self.down_arrow_pressed = False

    def reset(self):
        self._steering_command = 0
        self.up_arrow_pressed = False
        self.down_arrow_pressed = False

    def compute_discrete_input(self, dt):
        return self._steering_command

    def compute_continuous_input(self, dt):
        raise NotImplementedError("the keyboard agent can only generate discrete inputs")

    def connect_event_listener(self, app: QtWidgets.QApplication):
        app.installEventFilter(self)

    def _update_steering_command(self):
        if self.down_arrow_pressed and not self.up_arrow_pressed:
            self._steering_command = -1
        elif self.up_arrow_pressed and not self.down_arrow_pressed:
            self._steering_command = 1
        else:
            self._steering_command = 0

    def eventFilter(self, obj, event):
        if event.type() == QtGui.QKeyEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_D:
                self.down_arrow_pressed = True
                self._update_steering_command()
            elif event.key() == QtCore.Qt.Key_W:
                self.up_arrow_pressed = True
                self._update_steering_command()
            return True
        elif event.type() == QtGui.QKeyEvent.KeyRelease:
            if event.key() == QtCore.Qt.Key_D:
                self.down_arrow_pressed = False
                self._update_steering_command()
            elif event.key() == QtCore.Qt.Key_W:
                self.up_arrow_pressed = False
                self._update_steering_command()
            return True
        return False

    @property
    def name(self):
        return self._name
