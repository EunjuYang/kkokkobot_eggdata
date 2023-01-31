# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication, QWidget, QFileSystemModel, QVBoxLayout, QComboBox, QSpacerItem, QSizePolicy, QMessageBox
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtCore import Slot, QModelIndex, QCoreApplication, Qt
from feedback_ui import Ui_Form
import os
import cv2 as cv
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np
import json
import paramiko
from stat import S_ISDIR, S_ISREG

# sudo dscl . create /Groups/docker
# sudo dseditgroup -o edit -a $USER -t user docker
# 맥을 기준으로 위 두개를 통해 docker를 sudo 없이 돌아가게 해놔야한다

# ssh 설정
target_IP = '' # ssh를 시도할 IP
target_Port =  # ssh를 시도할 Port 번호
target_ID = '' # ssh ID
target_PWD = '' # ssh PWD
dataPath = '' # ssh 대상 내 데이터셋 경로

grid_size = (3,3)
combo_box_index = {"Normal": 0, "Crack": 1, "Latent Crack": 2}
rootPath = '.'

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(target_IP, port=target_Port, username=target_ID, password=target_PWD)
sftp = cli.open_sftp()

remote_fileList = []

def listdir_r(sftp, remotedir):
    for entry in sftp.listdir_attr(remotedir):
        remotepath = remotedir + "/" + entry.filename
        mode = entry.st_mode
        if S_ISDIR(mode):
            listdir_r(sftp, remotepath)
        elif S_ISREG(mode):
            if '/0/' in remotepath or '/2/' in remotepath or '/6/' in remotepath:
                if os.path.splitext(remotepath)[1] == '.jpg':
                    remote_fileList.append(remotepath)

listdir_r(sftp, dataPath)

def read_json(jsonPath):
    with sftp.open(jsonPath, 'r') as f:
        json_data = json.load(f)
    return json_data

def write_json(dict, jsonPath):
    with sftp.open(jsonPath, 'w') as f:
        json.dump(dict, f)

def fileExists(sftp, path):
    try: 
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False

class MainWindow(QWidget, Ui_Form):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.curIndex = 0
        self.imgList = remote_fileList
        self.activated = True
        # 어노테이션 저장 위치
        self.annotation = []
        for i in range(grid_size[0]):
            self.annotation.append([])
        self.setupUi(self)
        
    def setupUi(self, Form):
        super(MainWindow, self).setupUi(self)
        self.setWindowTitle("꼬꼬봇 데이터 판독 도우미")
        # 우측 라벨
        # horizontal Layout이니 grid_size의 [1]만큼 먼저 vertical 만들기
        for i in range(grid_size[1]):
            verticalLine = QVBoxLayout()
            self.horizontalLayout_2.addLayout(verticalLine)
            # grid_size의 [0]만큼 vertical Layout에 comboBox 추가
            for j in range(grid_size[0]):
                comboBox = QComboBox()
                items = ['Normal', 'Crack', 'Latent Crack']
                comboBox.addItems(items)
                comboBox.currentTextChanged.connect(self.annotation_changed)
                verticalLine.addWidget(comboBox)
                self.annotation[j].append(comboBox)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacer)
        self.Next.clicked.connect(self.next_button_pressed)
        self.Previous.clicked.connect(self.previous_button_pressed)
        self.Next.setShortcut("right")
        self.Previous.setShortcut("left")

        self.openImage()

    def openImage(self):
        # ComboBox 초기화
        self.activated = False
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                self.annotation[i][j].setCurrentIndex(0)
        self.activated = True
        # 파일 읽어와 띄우기
        self.originalPixMap = self.draw_grid(self.imgList[self.curIndex], grid_size)
        w = self.Original.width()
        h = self.Original.height()
        self.Original.setPixmap(self.originalPixMap.scaled(w, h, Qt.KeepAspectRatio))
        self.filePath_lineEdit.setText(self.imgList[self.curIndex])
        self.index_label.setText(str(self.curIndex + 1) + '/' + str(len(self.imgList)))
        # 만약 이미 작성된 어노테이션이 있다면 불러온다
        # 어노테이션은 파일 이름과 동일한 JSON으로 저장
        jsonPath = os.path.splitext(self.imgList[self.curIndex])[0] + '.json'
        json_data = dict()
        json_data["grid_size"] = grid_size
        if fileExists(sftp, jsonPath):
            json_data = read_json(jsonPath)
        # 없거나 grid가 일치 하지 않는 경우
        if not fileExists(sftp, jsonPath) or json_data["grid_size"][0] != grid_size[0] or json_data["grid_size"][1] != grid_size[1]:
            json_data = dict()
            json_data["grid_size"] = grid_size
            json_data["annotations"] = []
            for i in range(grid_size[0]):
                json_data["annotations"].append([])
            # 값 초기화
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    json_data["annotations"][i].append(self.annotation[i][j].currentText())
            # 파일 저장
            write_json(json_data, jsonPath)
        # json 값 토대로 spinBox 구성
        self.activated = False
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                self.annotation[i][j].setCurrentIndex(combo_box_index[json_data["annotations"][i][j]])
        self.activated = True
            
    def resizeEvent(self, event: QResizeEvent) -> None:
        w = self.Original.width()
        h = self.Original.height()
        self.Original.setPixmap(self.originalPixMap.scaled(w, h, Qt.KeepAspectRatio))
        return super().resizeEvent(event)

    def draw_grid(self, imgpath, grid_shape, color=(0,255,0), thickness=1):
        with sftp.open(imgpath) as f:
            img = cv.imdecode(np.fromstring(f.read(), np.uint8), 1)
        h, w, _ = img.shape
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv.line(img, (x,0), (x,h), color=color, thickness=thickness)
        
        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv.line(img, (0,y), (w,y), color=color, thickness=thickness)
        
        #return img
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(rgb_img).convert('RGB')
        return QPixmap.fromImage(ImageQt(PIL_image))

    def next_button_pressed(self):
        if self.curIndex < len(self.imgList) - 1:
            self.curIndex += 1
            self.openImage()
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("The Last File")
            msg.exec_()

    def previous_button_pressed(self):
        if self.curIndex > 0:
            self.curIndex -= 1
            self.openImage()
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("The First File")
            msg.exec_()

    def annotation_changed(self):
        if (self.activated == True):
            sender = self.sender()
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if self.annotation[i][j] == sender:
                        break
                else:
                    continue
                break
            jsonPath = os.path.splitext(self.imgList[self.curIndex])[0] + '.json'
            json_data = read_json(jsonPath)
            json_data["annotations"][i][j] = self.annotation[i][j].currentText()
            write_json(json_data, jsonPath)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    #window.showMaximized()
    window.show()
    sys.exit(app.exec())
