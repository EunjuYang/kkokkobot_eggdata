from torchvision.datasets import ImageFolder
from util.parser import parser, load_configuration
from torch.utils.data import DataLoader
from torchvision import transforms
from dotmap import DotMap
from util.train import train, test_model
import torch
import numpy as np
import random
import mlflow
import os

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
    # Create checkpoints folder
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints", exist_ok=False)
    if not os.path.exists("./checkpoints/"+conf.model):
        os.makedirs("./checkpoints/"+conf.model)
    if not os.path.exists("./checkpoints/"+conf.model + '/'+conf.run_name):
        os.makedirs("./checkpoints/"+conf.model +'/'+conf.run_name)

    # Create mlflow setting
    mlflow.set_tracking_uri('http://'+args.mlflow)
    mlflow.set_experiment(conf.exp)
    for (key, value) in conf_.items():
        mlflow.log_param(key, value)

    # train big class or subclass
    if conf.mode == "big_class":
        labels = ['정상', '크랙', '이물질', '기형', '외형(탈색)']
    else:
        labels = ['핑크', '갈색', '옅은갈색', '진한갈색', '미세크랙', '완전크랙', '원형크랙', '일자형크랙', '깃털', '닭분변', '점박이', '계란유출', '곰보', '일반기형', '심한기형', '길쭉한계란', '사포란', '부분탈색', '위아래탈색', '칼슘침전물']

    ###################################################
    # Preparation
    ###################################################
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data transformation
    img_size = 224
    transform_ = transforms.Compose([transforms.Resize([img_size, img_size]),
                                     transforms.RandomAdjustSharpness(6, p=1.0),
                                     # transforms.RandomAutocontrast(p=1.0),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         [0.5551, 0.4777, 0.4163],
                                         [0.2293, 0.1675, 0.1997])])

    ###################################################
    # Data Preparation
    ###################################################
    train_dataset = ImageFolder(root=conf.dataset_dir+'/train', transform=transform_)
    test_dataset = ImageFolder(root=conf.dataset_dir+'/test', transform=transform_)

    n_train = int(len(train_dataset) * 0.9)
    n_val = len(train_dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(train_dataset, [n_train, n_val])
    dataloaders = {"train": DataLoader(train_set, batch_size=conf.batch_size, shuffle=True),
                   "val": DataLoader(val_set, batch_size=conf.batch_size, shuffle=True)}
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=True)

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

    if conf.pre_trained is not None:
        print('Pre-trained model is loaded from ' + conf.pre_trained)
        state_dict = torch.load(conf.pre_trained)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    ###################################################
    # Loss construction
    ###################################################
    if conf.loss == 'CrossEntropy':
        if conf.weighted_loss:
            import pandas as pd
            img_stats = pd.read_csv(conf.dataset_dir + "/train_labels.csv", names=['file_name', 'label'])
            stat=img_stats['label'].value_counts()
            print(stat)
            nSamples=[stat[i] for i in range(len(labels))]
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            normedWeights = torch.FloatTensor(normedWeights).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=normedWeights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    ###################################################
    # Optimizer construction
    ###################################################
    if conf.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.base_lr, weight_decay=conf.wd)
    elif conf.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf.base_lr, weight_decay=conf.wd)

    ###################################################
    # Learning Rate Scheduler
    ###################################################
    if conf.lr_scheduler == 'Cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(optimizer, conf.n_epochs)


    ###################################################
    # Training
    ###################################################
    train(conf, model, dataloaders, criterion, optimizer, lr_scheduler, device, conf.is_mixup)

    ###################################################
    # Test
    ###################################################
    model.load_state_dict(torch.load('./checkpoints/'+conf.model+ '/' + conf.run_name + '/ckp_best.pt'))
    test_model(model, test_loader, device)
