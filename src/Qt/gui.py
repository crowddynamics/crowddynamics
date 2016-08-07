# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
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
        MainWindow.resize(1000, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.simulationName = QtGui.QComboBox(self.centralwidget)
        self.simulationName.setObjectName(_fromUtf8("simulationName"))
        self.simulationName.addItem(_fromUtf8(""))
        self.simulationName.setItemText(0, _fromUtf8(""))
        self.simulationName.addItem(_fromUtf8(""))
        self.simulationName.addItem(_fromUtf8(""))
        self.simulationName.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.simulationName)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.agentSize = QtGui.QSpinBox(self.centralwidget)
        self.agentSize.setMinimum(1)
        self.agentSize.setMaximum(1000)
        self.agentSize.setObjectName(_fromUtf8("agentSize"))
        self.verticalLayout.addWidget(self.agentSize)
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout.addWidget(self.label_2)
        self.widthBox = QtGui.QDoubleSpinBox(self.centralwidget)
        self.widthBox.setMinimum(1.0)
        self.widthBox.setSingleStep(1.0)
        self.widthBox.setObjectName(_fromUtf8("widthBox"))
        self.verticalLayout.addWidget(self.widthBox)
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout.addWidget(self.label_3)
        self.heightBox = QtGui.QDoubleSpinBox(self.centralwidget)
        self.heightBox.setMinimum(1.0)
        self.heightBox.setSingleStep(1.0)
        self.heightBox.setObjectName(_fromUtf8("heightBox"))
        self.verticalLayout.addWidget(self.heightBox)
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout.addWidget(self.label_4)
        self.bodyType = QtGui.QComboBox(self.centralwidget)
        self.bodyType.setObjectName(_fromUtf8("bodyType"))
        self.bodyType.addItem(_fromUtf8(""))
        self.bodyType.addItem(_fromUtf8(""))
        self.bodyType.addItem(_fromUtf8(""))
        self.bodyType.addItem(_fromUtf8(""))
        self.bodyType.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.bodyType)
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout.addWidget(self.label_5)
        self.agentModel = QtGui.QComboBox(self.centralwidget)
        self.agentModel.setObjectName(_fromUtf8("agentModel"))
        self.agentModel.addItem(_fromUtf8(""))
        self.agentModel.addItem(_fromUtf8(""))
        self.verticalLayout.addWidget(self.agentModel)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout.addWidget(self.line)
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.verticalLayout.addWidget(self.label_6)
        self.initSimulation = QtGui.QPushButton(self.centralwidget)
        self.initSimulation.setObjectName(_fromUtf8("initSimulation"))
        self.verticalLayout.addWidget(self.initSimulation)
        self.runSimulation = QtGui.QPushButton(self.centralwidget)
        self.runSimulation.setObjectName(_fromUtf8("runSimulation"))
        self.verticalLayout.addWidget(self.runSimulation)
        self.saveSimulation = QtGui.QPushButton(self.centralwidget)
        self.saveSimulation.setObjectName(_fromUtf8("saveSimulation"))
        self.verticalLayout.addWidget(self.saveSimulation)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.plt = GraphicsLayoutWidget(self.centralwidget)
        self.plt.setObjectName(_fromUtf8("plt"))
        self.horizontalLayout.addWidget(self.plt)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Crowd Dynamics", None))
        self.simulationName.setItemText(1, _translate("MainWindow", "outdoor", None))
        self.simulationName.setItemText(2, _translate("MainWindow", "hallway", None))
        self.simulationName.setItemText(3, _translate("MainWindow", "evacuation", None))
        self.label.setText(_translate("MainWindow", "Size", None))
        self.label_2.setText(_translate("MainWindow", "Width", None))
        self.label_3.setText(_translate("MainWindow", "Height", None))
        self.label_4.setText(_translate("MainWindow", "Body type", None))
        self.bodyType.setItemText(0, _translate("MainWindow", "adult", None))
        self.bodyType.setItemText(1, _translate("MainWindow", "male", None))
        self.bodyType.setItemText(2, _translate("MainWindow", "female", None))
        self.bodyType.setItemText(3, _translate("MainWindow", "child", None))
        self.bodyType.setItemText(4, _translate("MainWindow", "eldery", None))
        self.label_5.setText(_translate("MainWindow", "Agent model", None))
        self.agentModel.setItemText(0, _translate("MainWindow", "circular", None))
        self.agentModel.setItemText(1, _translate("MainWindow", "three_circle", None))
        self.label_6.setText(_translate("MainWindow", "Simulation controls", None))
        self.initSimulation.setText(_translate("MainWindow", "Initialize", None))
        self.runSimulation.setText(_translate("MainWindow", "Run", None))
        self.saveSimulation.setText(_translate("MainWindow", "Save", None))

from pyqtgraph import GraphicsLayoutWidget
