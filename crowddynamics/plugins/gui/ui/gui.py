# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer/gui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1000, 700)
        MainWindow.setStyleSheet(_fromUtf8(""))
        self.centralwidget = QtGui.QWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.sidebarLeft = QtGui.QVBoxLayout()
        self.sidebarLeft.setObjectName(_fromUtf8("sidebarLeft"))
        self.simulationsBox = QtGui.QComboBox(self.centralwidget)
        self.simulationsBox.setObjectName(_fromUtf8("simulationsBox"))
        self.sidebarLeft.addWidget(self.simulationsBox)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.sidebarLeft.addItem(spacerItem)
        self.gridLayout.addLayout(self.sidebarLeft, 0, 0, 2, 1)
        self.graphicsLayout = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsLayout.setStyleSheet(_fromUtf8(""))
        self.graphicsLayout.setObjectName(_fromUtf8("graphicsLayout"))
        self.gridLayout.addWidget(self.graphicsLayout, 0, 1, 1, 1)
        self.controlbarDown = QtGui.QWidget(self.centralwidget)
        self.controlbarDown.setObjectName(_fromUtf8("controlbarDown"))
        self.controlbar = QtGui.QHBoxLayout(self.controlbarDown)
        self.controlbar.setObjectName(_fromUtf8("controlbar"))
        self.startButton = QtGui.QPushButton(self.controlbarDown)
        self.startButton.setObjectName(_fromUtf8("startButton"))
        self.controlbar.addWidget(self.startButton)
        self.stopButton = QtGui.QPushButton(self.controlbarDown)
        self.stopButton.setObjectName(_fromUtf8("stopButton"))
        self.controlbar.addWidget(self.stopButton)
        self.saveButton = QtGui.QPushButton(self.controlbarDown)
        self.saveButton.setObjectName(_fromUtf8("saveButton"))
        self.controlbar.addWidget(self.saveButton)
        self.gridLayout.addWidget(self.controlbarDown, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 20))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.simulationMenu = QtGui.QMenu(self.menubar)
        self.simulationMenu.setObjectName(_fromUtf8("simulationMenu"))
        self.visualisationMenu = QtGui.QMenu(self.menubar)
        self.visualisationMenu.setObjectName(_fromUtf8("visualisationMenu"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionDensity = QtGui.QAction(MainWindow)
        self.actionDensity.setObjectName(_fromUtf8("actionDensity"))
        self.actionNavigation = QtGui.QAction(MainWindow)
        self.actionNavigation.setObjectName(_fromUtf8("actionNavigation"))
        self.actionNew = QtGui.QAction(MainWindow)
        self.actionNew.setObjectName(_fromUtf8("actionNew"))
        self.simulationMenu.addAction(self.actionOpen)
        self.simulationMenu.addAction(self.actionSave)
        self.visualisationMenu.addAction(self.actionDensity)
        self.visualisationMenu.addAction(self.actionNavigation)
        self.menubar.addAction(self.simulationMenu.menuAction())
        self.menubar.addAction(self.visualisationMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Crowd Dynamics", None))
        self.startButton.setText(_translate("MainWindow", "Start Process", None))
        self.stopButton.setText(_translate("MainWindow", "Stop Process", None))
        self.saveButton.setText(_translate("MainWindow", "Save", None))
        self.simulationMenu.setTitle(_translate("MainWindow", "Simulation", None))
        self.visualisationMenu.setTitle(_translate("MainWindow", "Visualisation", None))
        self.actionSave.setText(_translate("MainWindow", "Save As", None))
        self.actionOpen.setText(_translate("MainWindow", "Open", None))
        self.actionDensity.setText(_translate("MainWindow", "Density", None))
        self.actionNavigation.setText(_translate("MainWindow", "Navigation", None))
        self.actionNew.setText(_translate("MainWindow", "New", None))

from pyqtgraph import GraphicsLayoutWidget
