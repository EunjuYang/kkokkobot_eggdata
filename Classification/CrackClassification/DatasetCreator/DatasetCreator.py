# 데이터셋 생성
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import csv
import shutil

# 설정값
origin_dir = ''
target_dir = ''

# 데이터셋 생성 함수
def constructDataset(root_dir, tgt_dir, is_big_class):
    ###########################################
    # Preparation for dataset construction
    ###########################################
    mkdir(tgt_dir, is_big_class)
    if len(os.listdir(tgt_dir + 'bigclass/test')) != 0:
        print('[*] Test dataset is already constructed')
        return
    else:
        print('[*] Start to construct Test dataset')

    ###########################################
    # create dataset (split train/test dataset & label file)
    ###########################################
    label = 0
    dir = tgt_dir + '/bigclass' if is_big_class else tgt_dir + '/subclass'
    f_ts = open(dir + '/test_labels.csv', 'w+', newline='')
    wr_ts = csv.writer(f_ts)

    num_big_classes = len(os.listdir(root_dir))
    print(num_big_classes)
    for big in range(num_big_classes):
        big = str(big)
        sub_classes = os.listdir(root_dir+'/'+ big)

        for sub in range(len(sub_classes)):
            sub = str(sub)
            origin_path =root_dir+'/'+big+'/'+big + '_' + sub
            img_list = os.listdir(origin_path)
            print('[*] processing ' + origin_path + ' as label: ' + str(label))

            for i, img_name in enumerate(tqdm(img_list)):
                if os.path.splitext(img_name)[1] == '.jpg':
                    # Copy image to target dir (instead of simple copy, we open and convert to RGB)
                    dir_path = dir + '/test'
                    image = Image.open(origin_path + '/' + img_name).convert("RGB")
                    image.save(dir_path + '/' + img_name)

                    # Add label to .csv
                    writer = wr_ts
                    writer.writerow([img_name, label])
            label = label + 1 if not is_big_class else label

        label = label + 1 if is_big_class else label

    f_ts.close()



def mkdir(path, is_bigclass):
    """
    create Training/Test folder for both bigclass and subclass
    :param path:
    :return:
    """
    if is_bigclass:
        os.makedirs(path + '/bigclass/test', exist_ok=True)
    else:
        os.makedirs(path + '/subclass/test', exist_ok=True)

# 데이터셋 생성
constructDataset(root_dir=origin_dir,
                 tgt_dir=target_dir,
                 is_big_class=True)