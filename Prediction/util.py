from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import os

train_df_path = './freshness_train_dataset.csv'
test_df_path = './freshness_test_dataset.csv'

class parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='KKoKKo-BOT FRESHNESS dataset handling model')
        self._add_train()
        self._add_csv_path()
        self._add_data_path()
        self._add_seed()
        self._add_model()
        self._add_epochs()
        self._add_batch_size()
        self._add_save_model()
        self._add_gpus()

    def _add_gpus(self):
        self.parser.add_argument('--gpus',
                                 nargs='+',
                                 default=['/gpu:0'])

    def _add_model(self):
        self.parser.add_argument(
            '--model',
            dest='model',
            default='Full',
            help='The user can choose between "Full" and "ImageOnly"'
        )

    def _add_train(self):
        self.parser.add_argument(
            '--train',
            dest='train',
            default=False,
            help='If you want to train the model, please pass True '
                 'to train argument if you want to do an inference only, '
                 'please pass False to train argument',
            action='store_true'
        )

    def _add_batch_size(self):
        self.parser.add_argument(
            '--batch-size',
            dest='batch_size',
            default=512,
            type=int
        )

    def _add_epochs(self):
        self.parser.add_argument(
            '--epochs',
            dest='epochs',
            default=20,
            type=int
        )

    def _add_save_model(self):
        self.parser.add_argument(
            '--save-model',
            dest='save_model',
            default='./freshness_model_best.hdf5',
            type=str
        )

    def _add_data_path(self):
        self.parser.add_argument(
            '--data-path',
            dest='data_path',
            default='/data',
            type=str
        )

    def _add_csv_path(self):
        self.parser.add_argument(
            '--csv-path',
            dest='csv_path',
            default='./freshness_data.csv',
            type=str
        )

    def _add_seed(self):
        self.parser.add_argument(
            '--seed',
            dest='seed',
            default=99,
            type=int
        )

def construct_dataset(args):

    data_path = args.data_path

    if not os.path.isfile(train_df_path):

        print("Start to dataset construction")

        # load 'image' list & 'label list'
        img_list, label_list, CLASS = [], [], []
        for (path, dir_name, files) in os.walk(data_path):
            for file in files:

                # if there is 'image' and 'xml'
                if (os.path.splitext(file)[1] == '.jpg') and (os.path.isfile(path + '/' + os.path.splitext(file)[0]+'.xml')):
                    img_path = path + '/' + file
                    img_list.append(img_path)
                    label_list.append(path + '/' + os.path.splitext(file)[0]+'.xml')
                    paths = img_path.split(os.path.sep)
                    CLASS.append(paths[-3] + '/' + paths[-2])

        # parsing label
        DAY, WASH = [], []
        for label in label_list:
            xml_content = ET.parse(label)
            root = xml_content.getroot()
            DAY.append(int(root.findtext("days")))
            WASH.append(int(root.findtext("wash"))-1)

        raw_data = {'IMG': img_list, 'DAY': DAY, 'WASH': WASH, 'CLASS': CLASS}
        data = pd.DataFrame(raw_data)
        train_df, test_df = train_test_split(data, test_size=0.1, stratify=data['DAY'].values, random_state=args.seed)
        train_df.to_csv(train_df_path)
        test_df.to_csv(test_df_path)

    else:
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)

    return train_df, test_df



