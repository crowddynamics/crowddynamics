# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer/game.ui'
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

class Ui_Game(object):
    def setupUi(self, Game):
        Game.setObjectName(_fromUtf8("Game"))
        Game.resize(152, 105)
        self.verticalLayout = QtGui.QVBoxLayout(Game)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(Game)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.tasetBox = QtGui.QDoubleSpinBox(Game)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tasetBox.sizePolicy().hasHeightForWidth())
        self.tasetBox.setSizePolicy(sizePolicy)
        self.tasetBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tasetBox.setMaximum(16777215.0)
        self.tasetBox.setObjectName(_fromUtf8("tasetBox"))
        self.verticalLayout.addWidget(self.tasetBox)
        self.gameButton = QtGui.QPushButton(Game)
        self.gameButton.setObjectName(_fromUtf8("gameButton"))
        self.verticalLayout.addWidget(self.gameButton)

        self.retranslateUi(Game)
        QtCore.QMetaObject.connectSlotsByName(Game)

    def retranslateUi(self, Game):
        Game.setWindowTitle(_translate("Game", "Form", None))
        self.label.setText(_translate("Game", "T_aset_0", None))
        self.gameButton.setText(_translate("Game", "Update Game", None))

