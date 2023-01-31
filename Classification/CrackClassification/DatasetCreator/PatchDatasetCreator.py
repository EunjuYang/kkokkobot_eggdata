import cv2
from PIL import Image
from itertools import product
import os
import json
from IPython.display import display

# 파일 목록 가져오기
fileList = [] # Json 파일만 가져온다
for folderName, subfolders, filenames in os.walk('../Dataset/Crack_Dataset/clahe/'):
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.json':
            fileList.append(folderName + '/' + filename)

fileList[0:5]

# JSON에 해당하는 이미지 경로 반환하는 함수
def img_dir_from_json(json_dir):
    return os.path.splitext(json_dir)[0] + '.jpg'

img_dir_from_json(fileList[0])

# JSON을 읽어서 그리드 설정이랑 라벨을 가져오는 함수
def read_json(json_dir):
    with open(json_dir) as f:
        json_data = json.load(f)
    return json_data['grid_size'], json_data['annotations']

read_json(fileList[0])

import matplotlib.pyplot as plt
import numpy as np

def imgcrop(img_input, annotations, grid_size):
    xPieces = grid_size[0]
    yPieces = grid_size[1]
    filename, file_extension = os.path.splitext(img_input)
    im = Image.open(img_input)
    imgwidth, imgheight = im.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            # 파일 저장 -> ../Dataset/Crack_Dataset/patches            
            if annotations[i][j] == 'Normal':
                a.save("../Dataset/Crack_Dataset/clahe_patches/0/0_0/" + os.path.basename(filename) + "-" + str(i) + "-" + str(j) + file_extension)
            elif annotations[i][j] == 'Crack':
                a.save("../Dataset/Crack_Dataset/clahe_patches/1/1_0/" + os.path.basename(filename) + "-" + str(i) + "-" + str(j) + file_extension)
            else:
                a.save("../Dataset/Crack_Dataset/clahe_patches/2/2_0/" + os.path.basename(filename) + "-" + str(i) + "-" + str(j) + file_extension)

import shutil

# ../Dataset/Crack_Dataset/patches 폴더가 있으면 지우고 없으면 만든다
if os.path.isdir('../Dataset/Crack_Dataset/clahe_patches'):
    shutil.rmtree('../Dataset/Crack_Dataset/clahe_patches')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches/0')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches/1')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches/2')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches/0/0_0')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches/1/1_0')
os.mkdir('../Dataset/Crack_Dataset/clahe_patches/2/2_0')

for files in fileList:
    grid_size, annotations = read_json(files)
    imgcrop(img_dir_from_json(files), annotations, grid_size)