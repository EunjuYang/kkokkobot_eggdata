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


def construct_mutually_balanced_img_folder(root_dir, tgt_dir, is_big_class, balanced_list_tr=None, balanced_list_ts=None):

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
    for big in range(num_big_classes):
        big_str = str(big)
        sub_classes = os.listdir(root_dir+'/'+ big_str)

        for sub in range(len(sub_classes)):
            sub_str = str(sub)
            origin_path =root_dir+'/'+big_str+'/'+big_str + '_' + sub_str
            img_list = os.listdir(origin_path)

            train_len = balanced_list_tr[big][sub] if balanced_list_tr[big][sub] < len(img_list) else len(img_list)
            balanced_list_tr[big][sub] = train_len

            max_len = train_len + balanced_list_ts[big][sub] if train_len + balanced_list_ts[big][sub] < len(img_list) else len(img_list)
            img_list = img_list[:max_len]
            print('[*] processing ' + origin_path + ' as label: ' + str(label))
            os.makedirs(dir + '/train/' + str(label), exist_ok=True)
            os.makedirs(dir + '/test/' + str(label), exist_ok=True)

            for i, img_name in enumerate(tqdm(img_list)):

                # Copy image to target dir (instead of simple copy, we open and convert to RGB)
                dir_path = dir + '/train' if i < train_len else dir + '/test'
                image = Image.open(origin_path + '/' + img_name).convert("RGB")
                # enhancer = ImageEnhance.Contrast(image)
                # image = enhancer.enhance(1.7)
                image.save(dir_path + '/' + str(label) + '/' + img_name)

                # Add label to .csv
                writer = wr_tr if i < train_len else wr_ts
                writer.writerow([img_name, label])
            label = label + 1 if not is_big_class else label

        label = label + 1 if is_big_class else label

    f_tr.close()
    f_ts.close()

    return


def construct_img_folder(root_dir, tgt_dir, is_big_class, train_ratio, max_len=5000):

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

        tgt_dir_name = big_str

        my_max_len = max_len // len(sub_classes)

        for sub_str in sub_classes:
            # origin_path =root_dir+'/'+big_str+'/'+big_str + '_' + sub_str
            origin_path =root_dir+'/'+big_str+'/'+ sub_str
            img_list = os.listdir(origin_path)
            img_list = img_list[:my_max_len] if len(img_list) > my_max_len else img_list

            tgt_dir_name = tgt_dir_name if is_big_class else big_str + '_' + sub_str
            print('[*] processing ' + origin_path + ' as label: ' + str(label) + ' ' + tgt_dir_name)
            # os.makedirs(dir + '/train/' + str(label), exist_ok=True)
            # os.makedirs(dir + '/test/' + str(label), exist_ok=True)
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

def constructDataset(root_dir, tgt_dir, is_big_class, train_ratio):
    """
    Training / Test dataset construction with label.csv
    It takes hierarchical data directory and makes 'training' and 'test' dataset into
    # /roo_dir : egg_data/
    #               0/
    #                0_0/ ...
    # /tgt_dir:
    #   /big_classes
    #       test_labels.csv (imagepath, label)
    #       train_labels.csv (imagepath, label)
    #       /train
    #           ~~~.jpg
    #       /test
    #           ~~~.jpg
    #   /sub_classes
    #       test_labels.csv (imagepath, label)
    #       train_labels.csv (imagepath, label)
    #       /train
    #           ~~~.jpg
    #       /test
    #           ~~~.jpg
    :param root_dir: root folder name of hierarchical dataset
    :param tgt_dir:
    :param is_big_class: True if it constructs big_class dataset
    :param train_ratio: e.g., 0.8 * len(sub_classses)
    :return:
    """

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
    for big in range(num_big_classes):
        big = str(big)
        sub_classes = os.listdir(root_dir+'/'+ big)

        for sub in range(len(sub_classes)):
            sub = str(sub)
            origin_path =root_dir+'/'+big+'/'+big + '_' + sub
            img_list = os.listdir(origin_path)
            img_list.sort()
            print('[*] processing ' + origin_path + ' as label: ' + str(label))

            for i, img_name in enumerate(tqdm(img_list)):

                # Copy image to target dir (instead of simple copy, we open and convert to RGB)
                dir_path = dir + '/train' if i <= len(img_list) * train_ratio else dir + '/test'
                image = Image.open(origin_path + '/' + img_name).convert("RGB")
                image.save(dir_path + '/' + img_name)

                # Add label to .csv
                writer = wr_tr if i <= len(img_list) * train_ratio else wr_ts
                writer.writerow([img_name, label])
            label = label + 1 if not is_big_class else label

        label = label + 1 if is_big_class else label

    f_tr.close()
    f_ts.close()



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


class EggDataset(Dataset):

    def __init__(self, tgt_dir, big_class, train,auto_augment=False,transform=None, target_transform=None):
        # transform should pass image resizing
        dir = tgt_dir + '/bigclass' if big_class else tgt_dir + '/subclass'
        label_path = dir + '/train_labels.csv' if train else dir + '/test_labels.csv'
        self.img_dir = dir + '/train' if train else dir + '/test'
        self.img_labels = pd.read_csv(label_path, names=['file_name', 'label'])
        self.transform = transform
        self.target_transform = target_transform
        if auto_augment:
            print("Auto augment")
            self.auto_augment=torchvision.transforms.AutoAugment()
        else:
            self.auto_augment=None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.auto_augment:
            image=self.auto_augment(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':

    # Backup
    """
    construct_mutually_balanced_img_folder(root_dir='/hdd/kkokkobot/FinalEggData/CLASSIFICATION/COLOR_RAW',
                                           tgt_dir='/hdd/kkokkobot/FinalEggData/CLASSIFICATION/COLOR',
                                           is_big_class=True,
                                           balanced_list_tr=balanced_list_tr,
                                           balanced_list_ts=balanced_list_ts)
    """
    construct_img_folder(root_dir='/hdd/kkokkobot/FinalEggData/CroppedImage/MONO/은주분류',
                         tgt_dir='/hdd/kkokkobot/FinalEggData/CLASSIFICATION/MONO_EJ',
                         is_big_class=True,
                         train_ratio=0.9,
                         max_len=4000)

    # construct_img_folder(root_dir='/hdd/kkokkobot/FinalEggData/CroppedImage/MONO/반광사진',
    #                      tgt_dir='/hdd/kkokkobot/FinalEggData/CLASSIFICATION/MONO',
    #                      is_big_class=False,
    #                      train_ratio=0.9,
    #                      max_len=5000)
