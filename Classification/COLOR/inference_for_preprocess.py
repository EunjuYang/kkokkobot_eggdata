from torchvision.datasets import ImageFolder
from util.parser import parser, load_configuration
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from dotmap import DotMap
import torch
import numpy as np
import random
import mlflow
import csv
import os
from torchvision.transforms.functional import pad

labels = ['정상', '크랙', '이물질', '탈색', '외형이상', '잠재크랙']


class SquarePad:

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 0, 'constant')


def inference_for_preprocess(model, data_loader, dataset, device, tgt_file):

    print('Inference result is saved in ', tgt_file)
    f = open(tgt_file, 'w', newline='')
    wr = csv.writer(f)
    idx = 0
    total = len(dataset.imgs)
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_output = F.softmax(outputs, dim=-1)
            _, predicted = torch.max(softmax_output.data, 1)
            for r in predicted.data:
                img_name = dataset.imgs[idx][0].split('/')[-1]
                print('[', idx+1, '/', total, ']', dataset.imgs[idx][0], img_name, labels[r.data])
                idx += 1
                wr.writerow([img_name, labels[r.data]])

    f.close()



if __name__ == '__main__':


    ###################################################
    # Parameter settings & preparation
    ###################################################
    parser = parser()
    parser = parser.parser
    args = parser.parse_args()
    conf_ = load_configuration(args.conf_path)
    conf = DotMap(conf_)

    # Set random_seed
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    ###################################################
    # Pre-process hierarchical egg dataset into Dataset
    ###################################################
    # Create mlflow setting
    mlflow.set_tracking_uri('http://'+args.mlflow)
    mlflow.set_experiment(conf.exp)
    for (key, value) in conf_.items():
        mlflow.log_param(key, value)

    # train big class or subclass
    if conf.mode == "big_class":
        # labels = ['기형', '이물질', '정상', '크랙', '외형(탈색)']
        labels = ['정상', '크랙',  '이물질',  '탈색',  '외형이상',  '잠재크랙']

    ###################################################
    # Preparation
    ###################################################
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data transformation
    img_size = 224
    # transform_ = transforms.Compose([transforms.Resize([img_size, img_size]),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize(
    #                                      [0.5551, 0.4777, 0.4163],
    #                                      [0.2293, 0.1675, 0.1997])])

    transform_ = transforms.Compose([SquarePad(),
                                     transforms.Resize([img_size, img_size]),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         [0.5522, 0.4713, 0.4157],
                                         [0.2271, 0.1678, 0.1960])])

    ###################################################
    # Data Preparation
    ###################################################
    print('START TO INFERENCE in ' + conf.dataset_dir)
    test_dataset = ImageFolder(root=conf.dataset_dir, transform=transform_)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

    ###################################################
    # Model construction
    ###################################################
    import timm

    if conf.model == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True, num_classes=len(labels))

    elif conf.model == 'xception':
        model = timm.create_model('xception', pretrained=True, num_classes=len(labels))

    elif conf.model == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True, num_classes=len(labels))

    elif conf.model == 'vit':
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=len(labels), img_size=224)

    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    ###################################################
    # Test
    ###################################################
    print(test_dataset.class_to_idx)
    model.load_state_dict(torch.load(conf.checkpoint), strict=False)
    print('Complete to model load')
    # inference_for_preprocess(model, data_loader, dataset, device, tgt_file):
    inference_for_preprocess(model, test_loader, test_dataset, device, conf.inference_csv)
