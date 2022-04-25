from PyQt5 import QtWidgets, QtCore

from gui import ParticipantDialog
from .experiment_control_gui_ui import Ui_MainWindow
from trackobjects.trackside import TrackSide


class ExperimentGUI(QtWidgets.QMainWindow):
    def __init__(self, track, surroundings=None, parent=None):
        super(ExperimentGUI, self).__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.sim_master = None
        self.surroundings = surroundings

        self._punishment_count_down_count = 0
        self._reset_after_punishment = True
        self.punishment_timer = QtCore.QTimer()
        self.punishment_timer.setInterval(1000)
        self.punishment_timer.timeout.connect(self._punishment_count_down)

        self.ui.worldView.initialize(track, self)
        self.ui.worldView.draw_overlay(False)
        if surroundings:
            self.ui.worldView.scene.addItem(surroundings.get_graphics_objects())
        self.ui.startPushButton.clicked.connect(self.start_experiment)

        self.left_participant_dialog = ParticipantDialog(track, surroundings=surroundings, show_speed_dial=True, parent=self)
        self.right_participant_dialog = ParticipantDialog(track, surroundings=surroundings, show_speed_dial=True,  parent=self)

        self.left_vehicle = None
        self.right_vehicle = None

        self.ui.leftTestFFPushButton.pressed.connect(lambda: self.test_vibration(TrackSide.LEFT, True))
        self.ui.leftTestFFPushButton.released.connect(lambda: self.test_vibration(TrackSide.LEFT, False))
        self.ui.rightTestFFPushButton.pressed.connect(lambda: self.test_vibration(TrackSide.RIGHT, True))
        self.ui.rightTestFFPushButton.released.connect(lambda: self.test_vibration(TrackSide.RIGHT, False))

        self.ui.actionAdd_condition.triggered.connect(self.add_condition_to_experiment)

        self.show()

    def register_sim_master(self, sim_master):
        self.sim_master = sim_master
        self._set_session_message()

    def register_controllable_cars(self, left_controllable_object, right_controllable_object, vehicle_length, vehicle_width, left_color='red', right_color='white'):
        self.left_vehicle = left_controllable_object
        self.right_vehicle = right_controllable_object

        self.ui.worldView.add_controllable_car(left_controllable_object, vehicle_length, vehicle_width, left_color)
        self.ui.worldView.add_controllable_car(right_controllable_object, vehicle_length, vehicle_width, right_color)

        self.left_participant_dialog.set_ego_vehicle(left_controllable_object, vehicle_length, vehicle_width, left_color, TrackSide.LEFT)
        self.left_participant_dialog.set_other_vehicle(right_controllable_object, vehicle_length, vehicle_width, right_color)

        self.right_participant_dialog.set_ego_vehicle(right_controllable_object, vehicle_length, vehicle_width, right_color, TrackSide.RIGHT)
        self.right_participant_dialog.set_other_vehicle(left_controllable_object, vehicle_length, vehicle_width, left_color)

    def start_experiment(self):
        if self.sim_master and not self.sim_master.main_timer.isActive():
            self.sim_master.start()
            self.ui.startPushButton.setEnabled(False)

    def reset(self):
        self.ui.startPushButton.setEnabled(True)
        self._set_session_message()

    def add_condition_to_experiment(self):
        list_of_conditions = [condition.name for condition in set(self.sim_master.experimental_conditions)]
        selected_name, success = QtWidgets.QInputDialog.getItem(self, 'Add condition', 'Add a condition to the experiment', list_of_conditions)

        if success:
            for condition in set(self.sim_master.experimental_conditions):
                if condition.name == selected_name:
                    self.sim_master.extra_conditions.append(condition)
                    break

    def _set_session_message(self):
        if self.sim_master.is_training:
            message = 'training ' + str(self.sim_master.training_run_number)
        else:
            message = 'experiment ' + str(self.sim_master.condition_number)

        self.ui.sessionLineEdit.setText(message)
        try:
            self.ui.conditionLineEdit.setText(self.sim_master.current_condition.name)
        except AttributeError:
            self.ui.conditionLineEdit.setText('unknown')

    def show_collision_punishment(self, reset_afterwards=True):
        self._reset_after_punishment = reset_afterwards

        self.left_participant_dialog.view.set_dancing_banana(True)
        self.right_participant_dialog.view.set_dancing_banana(True)

        self._punishment_count_down_count = 20
        self.punishment_timer.start()

    def _punishment_count_down(self):
        if self._punishment_count_down_count != 0:
            self._punishment_count_down_count -= 1
            self.left_participant_dialog.show_overlay(str(self._punishment_count_down_count))
            self.right_participant_dialog.show_overlay(str(self._punishment_count_down_count))
            self.ui.messageLineEdit.setText(str(self._punishment_count_down_count))
        else:
            self.punishment_timer.stop()
            if self._reset_after_punishment:
                self.sim_master.reset()
            self.left_participant_dialog.view.set_dancing_banana(False)
            self.right_participant_dialog.view.set_dancing_banana(False)

    def update_all_graphics(self, left_velocity, right_velocity):
        self.ui.worldView.update_all_graphics_positions()

        self.left_participant_dialog.update_all_graphics()
        self.right_participant_dialog.update_all_graphics()

        left_acceleration = self.left_vehicle.acceleration / self.left_vehicle.max_acceleration
        self.ui.leftVerticalSlider.setValue(left_acceleration * 100)

        right_acceleration = self.right_vehicle.acceleration / self.right_vehicle.max_acceleration
        self.ui.rightVerticalSlider.setValue(right_acceleration * 100)

        self.ui.leftLED.set_on(self.left_vehicle.cruise_control_active)
        self.ui.rightLED.set_on(self.right_vehicle.cruise_control_active)

        self.ui.leftVDoubleSpinBox.setValue(self.left_vehicle.velocity)
        self.ui.rightVDoubleSpinBox.setValue(self.right_vehicle.velocity)

        self.ui.leftVdDoubleSpinBox.setValue(self.left_vehicle.cruise_control_velocity)
        self.ui.rightVdDoubleSpinBox.setValue(self.right_vehicle.cruise_control_velocity)

    def show_overlay(self, message=None):
        self.left_participant_dialog.show_overlay(message)
        self.right_participant_dialog.show_overlay(message)

        if message:
            self.ui.messageLineEdit.setText(message)
        else:
            self.ui.messageLineEdit.setText('')

    def finish_experiment(self):
        self.show_overlay("Experiment Finished")
        QtWidgets.QMessageBox.information(self, 'Experiment Finished', 'The Experiment has finished, the interface will close now.')
        self.close()

    def test_vibration(self, side, boolean):
        self.sim_master.test_vibration(side, boolean)

    def update_time_label(self, time):
        pass
