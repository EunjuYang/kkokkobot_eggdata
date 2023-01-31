# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'feedback.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(895, 475)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.Previous = QPushButton(Form)
        self.Previous.setObjectName(u"Previous")
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Previous.sizePolicy().hasHeightForWidth())
        self.Previous.setSizePolicy(sizePolicy)
        self.Previous.setMaximumSize(QSize(20, 16777215))
        self.Previous.setAutoFillBackground(False)
        self.Previous.setFlat(True)

        self.horizontalLayout.addWidget(self.Previous)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_2)

        self.Original = QLabel(Form)
        self.Original.setObjectName(u"Original")
        self.Original.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.Original)

        self.verticalLayout_2.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_4.addWidget(self.label_3)

        self.index_label = QLabel(Form)
        self.index_label.setObjectName(u"index_label")

        self.horizontalLayout_4.addWidget(self.index_label)

        self.horizontalLayout_4.setStretch(1, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.horizontalLayout_3.addWidget(self.label)

        self.filePath_lineEdit = QLabel(Form)
        self.filePath_lineEdit.setObjectName(u"filePath_lineEdit")

        self.horizontalLayout_3.addWidget(self.filePath_lineEdit)

        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")

        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 10)

        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.Next = QPushButton(Form)
        self.Next.setObjectName(u"Next")
        sizePolicy.setHeightForWidth(self.Next.sizePolicy().hasHeightForWidth())
        self.Next.setSizePolicy(sizePolicy)
        self.Next.setMaximumSize(QSize(20, 16777215))
        self.Next.setFlat(True)

        self.horizontalLayout.addWidget(self.Next)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 4)
        self.horizontalLayout.setStretch(2, 4)
        self.horizontalLayout.setStretch(3, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalLayout.setStretch(0, 7)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.Previous.setText(QCoreApplication.translate("Form", u"<", None))
#if QT_CONFIG(shortcut)
        self.Previous.setShortcut("")
#endif // QT_CONFIG(shortcut)
        self.label_2.setText(QCoreApplication.translate("Form", u"Original Image", None))
        self.Original.setText(QCoreApplication.translate("Form", u"TextLabel", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Index: ", None))
        self.index_label.setText(QCoreApplication.translate("Form", u"index", None))
        self.label.setText(QCoreApplication.translate("Form", u"filePath:", None))
        self.filePath_lineEdit.setText(QCoreApplication.translate("Form", u"filePath", None))
        self.Next.setText(QCoreApplication.translate("Form", u">", None))
    # retranslateUi

