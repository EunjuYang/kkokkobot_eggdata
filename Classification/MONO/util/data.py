from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm
import os
import pandas as pd
import csv
import shutil
from PIL import Image
import torch
import torchvision


def construct_img_folder(root_dir, tgt_dir, is_big_class, train_ratio):

    ###########################################
    # Preparation for dataset construction
    ###########################################
    mkdir(tgt_dir, is_big_class)
    if (is_big_class and len(os.listdir(tgt_dir + '/bigclass/train')) != 0) or (not is_big_class and len(os.listdir(tgt_dir + '/subclass/train')) != 0):
        print('[*] Training/Test dataset is already constructed')
        return
    else:
        print('[*] Start to construct Training/Test dataset')

    ###########################################
    # create dataset (split train/test dataset & label file)
    ###########################################
    label = 0
    dir = tgt_dir + '/bigclass' if is_big_class else tgt_dir + '/subclass'
    f_tr = open(dir + '/train_labels.csv', 'w+', newline='')
    f_ts = open(dir + '/test_labels.csv', 'w+', newline='')
    wr_tr = csv.writer(f_tr)
    wr_ts = csv.writer(f_ts)

    num_big_classes = len(os.listdir(root_dir))
    big_classes = os.listdir(root_dir)
    for big_str in big_classes:
        sub_classes = os.listdir(root_dir+'/'+ big_str)

        for sub_str in sub_classes:
            # origin_path =root_dir+'/'+big_str+'/'+big_str + '_' + sub_str
            origin_path =root_dir+'/'+big_str+'/'+ sub_str
            img_list = os.listdir(origin_path)

            tgt_dir_name = big_str if is_big_class else sub_str
            print('[*] processing ' + origin_path + ' as label: ' + tgt_dir_name)
            os.makedirs(dir + '/train/' + tgt_dir_name, exist_ok=True)
            os.makedirs(dir + '/test/' + tgt_dir_name, exist_ok=True)

            for i, img_name in enumerate(tqdm(img_list)):

                # Copy image to target dir (instead of simple copy, we open and convert to RGB)
                dir_path = dir + '/train' if i <= len(img_list) * train_ratio else dir + '/test'
                image = Image.open(origin_path + '/' + img_name).convert("RGB")
                # image.save(dir_path + '/' + str(label) + '/' + img_name)
                image.save(dir_path + '/' + tgt_dir_name + '/' + img_name)

                # Add label to .csv
                writer = wr_tr if i <= len(img_list) * train_ratio else wr_ts
                writer.writerow([img_name, label])
            label = label + 1 if not is_big_class else label

        label = label + 1 if is_big_class else label

    f_tr.close()
    f_ts.close()

    return

def mkdir(path, is_bigclass):
    """
    create Training/Test folder for both bigclass and subclass
    :param path:
    :return:
    """
    if is_bigclass:
        os.makedirs(path + '/bigclass/train', exist_ok=True)
        os.makedirs(path + '/bigclass/test', exist_ok=True)
    else:
        os.makedirs(path + '/subclass/train', exist_ok=True)
        os.makedirs(path + '/subclass/test', exist_ok=True)


if __name__ == '__main__':

    # Backup
    construct_img_folder(root_dir='/hdd1/kkokkobot/TTA_Final/CroppedImage/MONO1224',
                         tgt_dir='/hdd/kkokkobot/TTA_Final/CLASSIFICATION/MONO',
                         is_big_class=False,
                         train_ratio=0.9)
