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
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simulation_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SimpleMerging(object):
    def setupUi(self, SimpleMerging):
        SimpleMerging.setObjectName("SimpleMerging")
        SimpleMerging.resize(1300, 650)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SimpleMerging.sizePolicy().hasHeightForWidth())
        SimpleMerging.setSizePolicy(sizePolicy)
        SimpleMerging.setMinimumSize(QtCore.QSize(650, 650))
        self.centralwidget = QtWidgets.QWidget(SimpleMerging)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.world_view = WorldView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.world_view.sizePolicy().hasHeightForWidth())
        self.world_view.setSizePolicy(sizePolicy)
        self.world_view.setMinimumSize(QtCore.QSize(600, 420))
        self.world_view.setObjectName("world_view")
        self.verticalLayout.addWidget(self.world_view)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.leftSpeedDial = SpeedDialGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leftSpeedDial.sizePolicy().hasHeightForWidth())
        self.leftSpeedDial.setSizePolicy(sizePolicy)
        self.leftSpeedDial.setMinimumSize(QtCore.QSize(0, 95))
        self.leftSpeedDial.setObjectName("leftSpeedDial")
        self.horizontalLayout_4.addWidget(self.leftSpeedDial)
        self.rightSpeedDial = SpeedDialGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rightSpeedDial.sizePolicy().hasHeightForWidth())
        self.rightSpeedDial.setSizePolicy(sizePolicy)
        self.rightSpeedDial.setMinimumSize(QtCore.QSize(0, 95))
        self.rightSpeedDial.setObjectName("rightSpeedDial")
        self.horizontalLayout_4.addWidget(self.rightSpeedDial)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.previous_button = QtWidgets.QPushButton(self.centralwidget)
        self.previous_button.setObjectName("previous_button")
        self.horizontalLayout_3.addWidget(self.previous_button)
        self.play_button = QtWidgets.QPushButton(self.centralwidget)
        self.play_button.setObjectName("play_button")
        self.horizontalLayout_3.addWidget(self.play_button)
        self.next_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_button.setObjectName("next_button")
        self.horizontalLayout_3.addWidget(self.next_button)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.timeSlider = QtWidgets.QSlider(self.centralwidget)
        self.timeSlider.setMaximum(999)
        self.timeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.timeSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.timeSlider.setObjectName("timeSlider")
        self.verticalLayout.addWidget(self.timeSlider)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.velocityGraphicsView = PlotWidget(self.centralwidget)
        self.velocityGraphicsView.setObjectName("velocityGraphicsView")
        self.verticalLayout_2.addWidget(self.velocityGraphicsView)
        self.headwayGraphicsView = PlotWidget(self.centralwidget)
        self.headwayGraphicsView.setObjectName("headwayGraphicsView")
        self.verticalLayout_2.addWidget(self.headwayGraphicsView)
        self.conflictGraphicsView = PlotWidget(self.centralwidget)
        self.conflictGraphicsView.setObjectName("conflictGraphicsView")
        self.verticalLayout_2.addWidget(self.conflictGraphicsView)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        SimpleMerging.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SimpleMerging)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        SimpleMerging.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SimpleMerging)
        self.statusbar.setObjectName("statusbar")
        SimpleMerging.setStatusBar(self.statusbar)
        self.actionEnable_recording = QtWidgets.QAction(SimpleMerging)
        self.actionEnable_recording.setCheckable(True)
        self.actionEnable_recording.setObjectName("actionEnable_recording")
        self.menuFile.addAction(self.actionEnable_recording)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(SimpleMerging)
        QtCore.QMetaObject.connectSlotsByName(SimpleMerging)

    def retranslateUi(self, SimpleMerging):
        _translate = QtCore.QCoreApplication.translate
        SimpleMerging.setWindowTitle(_translate("SimpleMerging", "Simple Merging"))
        self.previous_button.setText(_translate("SimpleMerging", "<"))
        self.play_button.setText(_translate("SimpleMerging", "Start"))
        self.next_button.setText(_translate("SimpleMerging", ">"))
        self.menuFile.setTitle(_translate("SimpleMerging", "File"))
        self.actionEnable_recording.setText(_translate("SimpleMerging", "Enable recording"))
from gui.speeddialgraphicsview import SpeedDialGraphicsView
from gui.worldview import WorldView
from pyqtgraph import PlotWidget
