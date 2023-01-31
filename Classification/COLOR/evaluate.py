from torchvision.datasets import ImageFolder
from util.parser import parser, load_configuration
from torch.utils.data import DataLoader
from torchvision import transforms
from dotmap import DotMap
from util.train import train, test_model, evaluate
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

    if conf.pre_trained is not '':
        print('Pre-trained model is loaded from ' + conf.pre_trained)
        state_dict = torch.load(conf.pre_trained)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    ###################################################
    # Test
    ###################################################
    print(test_dataset.class_to_idx)
    labels = ['기형', '이물질', '정상', '크랙', '외형(탈색)']
    print(labels)
    model.load_state_dict(torch.load('./checkpoints/'+conf.model+ '/' + conf.run_name + '/ckp_best.pt'))
    print('model load ', './checkpoints/'+conf.model+ '/' + conf.run_name + '/ckp_best.pt')
    evaluate(model, test_loader, device, test_dataset, labels)
