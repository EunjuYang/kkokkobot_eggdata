from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import pandas as pd
import os

def imgcrop(im, grid_size):
    # im은 이제 FloatTensor
    cropped = []
    for i in range(grid_size[0]):
        cropped.append([])
    xPieces = grid_size[0]
    yPieces = grid_size[1]
    _, imgheight, imgwidth = im.shape
    height = imgheight // yPieces
    width = imgwidth // xPieces
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            a = im[:, i * height:(i + 1) * height, j * width:(j + 1) * width]
            cropped[i].append(a)
    return cropped

class EggDataset_Patches(Dataset):

    def __init__(self, tgt_dir,transform=None, target_transform=None):
        # transform should pass image resizing
        dir = tgt_dir + '/bigclass'
        label_path = dir + '/test_labels.csv'
        self.img_dir = dir + '/test'
        self.img_labels = pd.read_csv(label_path, names=['file_name', 'label'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path).type(torch.FloatTensor)
        img = read_image(img_path).type(torch.FloatTensor)
        # GRID, ANNOTATIONS 정보
        grid_size = (3,3)
        label = self.img_labels.iloc[idx, 1]
        # 이미지 크롭 및 라벨 리스트 제공
        images = imgcrop(img, grid_size)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # 이미지 후처리
                if self.transform:
                    images[i][j] = self.transform(images[i][j])
        if self.target_transform:
            label = self.target_transform(label)
        return images, label