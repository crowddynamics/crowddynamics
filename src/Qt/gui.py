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
        MainWindow.resize(1000, 600)
        MainWindow.setStyleSheet(_fromUtf8("\n"
"/*\n"
"    Android Material Dark\n"
"    COLOR_DARK     = #212121 Grey 900\n"
"    COLOR_MEDIUM   = #424242 Grey 800\n"
"    COLOR_MEDLIGHT = #757575 Grey 600\n"
"    COLOR_LIGHT    = #DDDDDD White\n"
"    COLOR_ACCENT   = #3F51B5 Indigo 500\n"
"*/\n"
"\n"
"* {\n"
"    background: #212121;\n"
"    color: #DDDDDD;\n"
"    border: 1px solid #757575;\n"
"}\n"
"\n"
"QWidget::item:selected {\n"
"    background: #3F51B5;\n"
"}\n"
"\n"
"QCheckBox, QRadioButton {\n"
"    border: none;\n"
"}\n"
"\n"
"QRadioButton::indicator, QCheckBox::indicator {\n"
"    width: 13px;\n"
"    height: 13px;\n"
"}\n"
"\n"
"QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {\n"
"    border: 1px solid #757575;\n"
"    background: none;\n"
"}\n"
"\n"
"QRadioButton::indicator:unchecked:hover, QCheckBox::indicator:unchecked:hover {\n"
"    border: 1px solid #DDDDDD;\n"
"}\n"
"\n"
"QRadioButton::indicator::checked, QCheckBox::indicator::checked {\n"
"    border: 1px solid #757575;\n"
"    background: #757575;\n"
"}\n"
"\n"
"QRadioButton::indicator:checked:hover, QCheckBox::indicator:checked:hover {\n"
"    border: 1px solid #DDDDDD;\n"
"    background: #DDDDDD;\n"
"}\n"
"\n"
"QGroupBox {\n"
"    margin-top: 6px;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    top: -7px;\n"
"    left: 7px;\n"
"}\n"
"\n"
"QScrollBar {\n"
"    border: 1px solid #757575;\n"
"    background: #212121;\n"
"}\n"
"\n"
"QScrollBar:horizontal {\n"
"    height: 15px;\n"
"    margin: 0px 0px 0px 32px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"    width: 15px;\n"
"    margin: 32px 0px 0px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle {\n"
"    background: #424242;\n"
"    border: 1px solid #757575;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"    border-width: 0px 1px 0px 1px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    border-width: 1px 0px 1px 0px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"    min-width: 20px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QScrollBar::add-line, QScrollBar::sub-line {\n"
"    background:#424242;\n"
"    border: 1px solid #757575;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line {\n"
"    position: absolute;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"    width: 15px;\n"
"    subcontrol-position: left;\n"
"    left: 15px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"    height: 15px;\n"
"    subcontrol-position: top;\n"
"    top: 15px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"    width: 15px;\n"
"    subcontrol-position: top left;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"    height: 15px;\n"
"    subcontrol-position: top;\n"
"}\n"
"\n"
"QScrollBar:left-arrow, QScrollBar::right-arrow, QScrollBar::up-arrow, QScrollBar::down-arrow {\n"
"    border: 1px solid #757575;\n"
"    width: 3px;\n"
"    height: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-page, QScrollBar::sub-page {\n"
"    background: none;\n"
"}\n"
"\n"
"QAbstractButton:hover {\n"
"    background: #424242;\n"
"}\n"
"\n"
"QAbstractButton:pressed {\n"
"    background: #757575;\n"
"}\n"
"\n"
"QAbstractItemView {\n"
"    show-decoration-selected: 1;\n"
"    selection-background-color: #3F51B5;\n"
"    selection-color: #DDDDDD;\n"
"    alternate-background-color: #424242;\n"
"}\n"
"\n"
"QHeaderView {\n"
"    border: 1px solid #757575;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"    background: #212121;\n"
"    border: 1px solid #757575;\n"
"    padding: 4px;\n"
"}\n"
"\n"
"QHeaderView::section:selected, QHeaderView::section::checked {\n"
"    background: #424242;\n"
"}\n"
"\n"
"QTableView {\n"
"    gridline-color: #757575;\n"
"}\n"
"\n"
"QTabBar {\n"
"    margin-left: 2px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    border-radius: 0px;\n"
"    padding: 4px;\n"
"    margin: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    background: #424242;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    border: 1px solid #757575;\n"
"    background: #424242;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    border: 1px solid #757575;\n"
"    background: #424242;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
"    width: 3px;\n"
"    height: 3px;\n"
"    border: 1px solid #757575;\n"
"}\n"
"\n"
"QAbstractSpinBox {\n"
"    padding-right: 15px;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {\n"
"    border: 1px solid #757575;\n"
"    background: #424242;\n"
"    subcontrol-origin: border;\n"
"}\n"
"\n"
"QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {\n"
"    width: 3px;\n"
"    height: 3px;\n"
"    border: 1px solid #757575;\n"
"}\n"
"\n"
"QSlider {\n"
"    border: none;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    margin: 4px 0px 4px 0px;\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    width: 5px;\n"
"    margin: 0px 4px 0px 4px;\n"
"}\n"
"\n"
"QSlider::handle {\n"
"    border: 1px solid #757575;\n"
"    background: #424242;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    width: 15px;\n"
"    margin: -4px 0px -4px 0px;\n"
"}\n"
"\n"
"QSlider::handle:vertical {\n"
"    height: 15px;\n"
"    margin: 0px -4px 0px -4px;\n"
"}\n"
"\n"
"QSlider::add-page:vertical, QSlider::sub-page:horizontal {\n"
"    background: #3F51B5;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, QSlider::add-page:horizontal {\n"
"    background: #424242;\n"
"}\n"
"\n"
"QLabel {\n"
"    border: none;\n"
"}\n"
"\n"
"QProgressBar {\n"
"    text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    width: 1px;\n"
"    background-color: #3F51B5;\n"
"}\n"
"\n"
"QMenu::separator {\n"
"    background: #424242;\n"
"}\n"
"\n"
"QStatusBar {\n"
"    border: 1px;\n"
"    color: #3F51B5;\n"
"}"))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.verticalLayout.addWidget(self.label_7)
        self.simulationName = QtGui.QComboBox(self.centralwidget)
        self.simulationName.setObjectName(_fromUtf8("simulationName"))
        self.simulationName.addItem(_fromUtf8(""))
        self.simulationName.setItemText(0, _fromUtf8(""))
        self.simulationName.addItem(_fromUtf8(""))
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
        self.plt.setStyleSheet(_fromUtf8(""))
        self.plt.setObjectName(_fromUtf8("plt"))
        self.horizontalLayout.addWidget(self.plt)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuMenu = QtGui.QMenu(self.menubar)
        self.menuMenu.setObjectName(_fromUtf8("menuMenu"))
        self.menuVisualisations = QtGui.QMenu(self.menubar)
        self.menuVisualisations.setObjectName(_fromUtf8("menuVisualisations"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionDensity_Grid = QtGui.QAction(MainWindow)
        self.actionDensity_Grid.setObjectName(_fromUtf8("actionDensity_Grid"))
        self.actionNavigation_Field = QtGui.QAction(MainWindow)
        self.actionNavigation_Field.setObjectName(_fromUtf8("actionNavigation_Field"))
        self.menuMenu.addAction(self.actionOpen)
        self.menuMenu.addAction(self.actionSave)
        self.menuVisualisations.addAction(self.actionDensity_Grid)
        self.menuVisualisations.addAction(self.actionNavigation_Field)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuVisualisations.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Crowd Dynamics", None))
        self.label_7.setText(_translate("MainWindow", "New Simulation", None))
        self.simulationName.setItemText(1, _translate("MainWindow", "outdoor", None))
        self.simulationName.setItemText(2, _translate("MainWindow", "hallway", None))
        self.simulationName.setItemText(3, _translate("MainWindow", "evacuation", None))
        self.simulationName.setItemText(4, _translate("MainWindow", "evacuation_game", None))
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
        self.menuMenu.setTitle(_translate("MainWindow", "File", None))
        self.menuVisualisations.setTitle(_translate("MainWindow", "Visualisation", None))
        self.actionSave.setText(_translate("MainWindow", "Save As", None))
        self.actionOpen.setText(_translate("MainWindow", "Open", None))
        self.actionDensity_Grid.setText(_translate("MainWindow", "Density", None))
        self.actionNavigation_Field.setText(_translate("MainWindow", "Navigation", None))

from pyqtgraph import GraphicsLayoutWidget
