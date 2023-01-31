import os
import csv
from PIL import Image

#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True


def construct_DetectionDataset(root_dir, tgt_dir, train_ratio):
    """
    Find pair name.jpg and name.xml in the sub folders in root_dir
    And then copy them to tgt_dir
    :param root_dir: root directory of dataset
    :param tgt_dir: target directory of dataset
    :param train_ratio: ratio of train/test set
    :return:
    """
    os.makedirs(tgt_dir + '/train', exist_ok = True)
    os.makedirs(tgt_dir + '/test', exist_ok = True)
    os.makedirs(tgt_dir + '/val', exist_ok = True)
    if len(os.listdir(tgt_dir + '/train')) != 0:
        print('[*] Training/Test Detection dataset is already constructed !!')
        return
    else:
        print('[*] Start to construct Training/Test dataset ...')

    ######################################################
    # create dataset (split train/test dataset & root_dir)
    ######################################################
    f_tr = open(tgt_dir + '/train_labels.csv', 'w+', newline='')
    f_val = open(tgt_dir + '/val_labels.csv', 'w+', newline='')
    f_ts = open(tgt_dir + '/test_labels.csv', 'w+', newline='')

    wr_tr = csv.writer(f_tr)
    wr_val = csv.writer(f_val)
    wr_ts = csv.writer(f_ts)

    count =  0 # variable for counting number for spliting purpose
    threshold = train_ratio * 10 # threshold for spliting purpose

    # Get all directories in root_dir
    sub_folders = [x[0] for x in os.walk(os.path.abspath(root_dir))]
    for sub in (sub_folders):
        img_list = os.listdir(sub)
        for i, img_name in enumerate(img_list):
            # Check existence of pair image and xml files
            if (img_name[-3:] == 'jpg'):
                # Copy image to target dir (instead of simple copy, we open and convert to RGB)
                if (count % 10 < threshold):
                    dir_path = tgt_dir + '/train'
                    wr = wr_tr
                elif (count % 10 == threshold):
                    dir_path = tgt_dir + '/val'
                    wr = wr_val
                else:
                    dir_path = tgt_dir + '/test'
                    wr = wr_ts

                print(f'[{count}][{dir_path}] {img_name}')

                image = Image.open(sub + '/' + img_name).convert("RGB")
                if (image.load()):
                    image.save(dir_path + '/' + img_name)
                else:
                    print(f'[{count}][{dir_path}] {img_name} is not loaded')
                    continue


                # Add label to .csv file
                wr.writerow([img_name, sub])

                count += 1

    f_tr.close()
    f_val.close()
    f_ts.close()
    print('The total images:', count)


# Main function
if __name__ == '__main__':
   construct_DetectionDataset(root_dir ='/hdd_ext/kkokkobot/data/Classification/COLOR_RAW', tgt_dir='/hdd_ext/kkokkobot/data/Classification/COLOR', train_ratio=0.8)