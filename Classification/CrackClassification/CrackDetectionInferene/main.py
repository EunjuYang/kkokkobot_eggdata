import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
import argparse
from tqdm import tqdm
from torchvision import transforms
from mlflow import log_metric
from dataLoader import EggDataset_Patches
from img_process import pixel_map

# 패치 학습에 쓰인 그리드 크기
grid_size = [3,3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)

def run(parse):
    # 그리드 크기가 3 x 3 이라 기존의 1/3의 길이
    transform_ = transforms.Resize((80, 106))
    patches_test = EggDataset_Patches(tgt_dir=parse.tgt_dir, transform=transform_)
    patches_loader = DataLoader(patches_test, batch_size=1, shuffle=False)

    # 체크 포인트 로드
    model.load_state_dict(torch.load(parse.checkpoints))
    
    # 모델 성능 검증
    test_model_patches(model, patches_loader, parse.patch_test)

    # 크랙 탐지 결과 출력
    num_normal = parse.num_ex
    num_crack = parse.num_ex
    num_latent = parse.num_ex
    for images, labels in tqdm(patches_loader):
        if labels.item() == 1:
            if num_crack > 0:
                pixel_map(images, labels, model, device)
                num_crack -= 1

        elif labels.item() == 2:
            if num_latent > 0:
                pixel_map(images, labels, model, device)
                num_latent -= 1

        else:
            if num_normal > 0:
                pixel_map(images, labels, model, device)
                num_normal -= 1

        if (num_crack + num_latent) <= 0:
            break

def test_model_patches(model, test_loader, patch_test):
    model.eval()

    # 정답 카운트
    correct_images = 0
    total_images = 0
    
    normal_info = {"num": 0, "toNormal": 0, "toCrack": 0, "toLatent": 0}
    crack_info = {"num": 0, "toNormal": 0, "toCrack": 0, "toLatent": 0}
    latent_info = {"num": 0, "toNormal": 0, "toCrack": 0, "toLatent": 0}
    
    with torch.no_grad():
        for images, label in test_loader:
            prediction = []
            for i in range(grid_size[0]):
                prediction.append([])

            # images와 labels 이 경우 list로 온다
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    image = images[i][j].to(device)
                    outputs = model(image)
                    _, predicted = torch.max(outputs.data, 1)
                    # Prediction List에 내용 추가
                    prediction[i].append(predicted.item())

            # Prediction List를 갖고 이미지 예측
            image_prediction = 0
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if prediction[i][j] == 1:
                        image_prediction = 1
                    elif prediction[i][j] == 2 and image_prediction != 1:
                        image_prediction = 2
            total_images += 1
            correct_images += 1 if image_prediction == label.item() else 0

            # 정보 갱신
            if label.item() == 0:
                normal_info["num"] += 1
                if image_prediction == 0:
                    normal_info["toNormal"] += 1
                elif image_prediction == 1:
                    normal_info["toCrack"] += 1
                else:
                    normal_info["toLatent"] += 1
            elif label.item() ==1:
                crack_info["num"] += 1
                if image_prediction == 0:
                    crack_info["toNormal"] += 1
                elif image_prediction == 1:
                    crack_info["toCrack"] += 1
                else:
                    crack_info["toLatent"] += 1
            else:
                latent_info["num"] += 1
                if image_prediction == 0:
                    latent_info["toNormal"] += 1
                elif image_prediction == 1:
                    latent_info["toCrack"] += 1
                else:
                    latent_info["toLatent"] += 1
    
    print("Total Images: " + str(total_images))
    print('Accuracy of the model on the test images: {} %'.format(100 * correct_images / total_images))
    log_metric("test_acc", 100 * correct_images/total_images)

    # 정보 출력
    print(normal_info)
    print(crack_info)
    print(latent_info)

def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid int value" %value)
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_mutually_exclusive_group(required=False)

    parser.add_argument('--checkpoint', '-c', help='Patchwise Trained checkpoint', dest='checkpoints')
    parser.add_argument('--tgt_dir', '-t', help='Dataset Directory', dest='tgt_dir')
    
    parser.add_argument('--patch_test', dest='patch_test', action='store_true')
    parser.add_argument('--no-patch_test', dest='patch_test', action='store_false')
    parser.set_defaults(patch_test=False)

    parser.add_argument('--num_examples', dest='num_ex', type=check_positive, default='0')
    run(parser.parse_args())