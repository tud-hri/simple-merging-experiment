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
from PyQt5 import QtWidgets, QtCore

from .participantinfodialog_ui import Ui_Dialog


class ParticipantInfoDialog(QtWidgets.QDialog):
    def __init__(self, participant_id, trackside, parent=None):
        super(ParticipantInfoDialog, self).__init__(parent=parent)

        self.participant_info = {'side': trackside,
                                 'id': participant_id,
                                 'age': None,
                                 'gender': None}

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.idLineEdit.setText(str(participant_id))
        self.ui.sideLineEdit.setText(str(trackside))

        self.ui.ageSpinBox.valueChanged.connect(self._set_age)

        for radio_button in [self.ui.femaleRadioButton, self.ui.maleRadioButton, self.ui.otherRadioButton, self.ui.unknownRadioButton]:
            radio_button.toggled.connect(self._set_gender)

        self.show()

    def _set_age(self):
        self.participant_info['age'] = self.ui.ageSpinBox.value()

    def _set_gender(self):
        if self.ui.femaleRadioButton.isChecked():
            self.participant_info['gender'] = 'female'
        elif self.ui.maleRadioButton.isChecked():
            self.participant_info['gender'] = 'male'
        elif self.ui.otherRadioButton.isChecked():
            self.participant_info['gender'] = 'other'
        elif self.ui.unknownRadioButton.isChecked():
            self.participant_info['gender'] = 'unknown'
        else:
            self.participant_info['gender'] = None
