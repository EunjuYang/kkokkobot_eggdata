import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch

labels_map = {0: "Normal", 1: "Crack", 2:"Latent Crack"}

def preprocess(img):
    processed = (img * 255).astype(np.uint8)
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    # _, processed = cv2.threshold(processed, 50, 255, cv2.THRESH_TOZERO)
    processed = cv2.erode(processed, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=5)
    return processed
    
def edge_detection(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, _ = cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret = ret*9//4 if ret*9//4 < 255 else 255
    _, f_cracked = cv2.threshold(image, ret, 255, cv2.THRESH_BINARY)
    processed = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    processed = cv2.fastNlMeansDenoising(processed, None, 3, 3, 27)
    processed = cv2.GaussianBlur(processed, (0,0), 1)
    cut1 = np.mean(processed) * 0.2
    cut2 = np.mean(processed) * 0.3
    processed = cv2.Canny(processed, cut1, cut2)
    added = cv2.add(processed, f_cracked)
    return added

def masking(mask, img):
    masked = cv2.bitwise_and(img, mask)
    masked = cv2.dilate(masked, (20, 20), iterations=3)
    masked = cv2.erode(masked, (20, 20), iterations=3)
    masked = cv2.dilate(masked, (20, 20), iterations=3)
    
    masked = cv2.fastNlMeansDenoising(masked, None, 13, 3, 27)
    masked = cv2.fastNlMeansDenoising(masked, None, 5, 3, 27)
    
    rets, _ = cv2.threshold(masked, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, masked = cv2.threshold(masked, rets//3, 255, cv2.THRESH_BINARY)
    
    masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    return masked

# 이제 위의 과정에 마스킹을 씌워보자
def single_cam_(image, model, device):
    original = (image / 256).T.squeeze()
    outputs = model(image.to(device))
    _, pred = torch.max(outputs, 1)
    
    targets = [ClassifierOutputTarget(pred)]
    # 캠 생성
    cam1 = GradCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=True)
    cam2 = GradCAM(model=model, target_layers=[model.layer2[-1]], use_cuda=True)
    cam3 = GradCAM(model=model, target_layers=[model.layer3[-1]], use_cuda=True)
    cam4 = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=True)
    grayscale_cam1 = cam1(input_tensor=image, targets=targets, aug_smooth=True)
    grayscale_cam2 = cam2(input_tensor=image, targets=targets, aug_smooth=True)
    grayscale_cam3 = cam3(input_tensor=image, targets=targets, aug_smooth=True)
    grayscale_cam4 = cam4(input_tensor=image, targets=targets, aug_smooth=True)
    
    # 캠 합성 - 평균
    weights = [1, 1, 2, 2]
    grayscale_cam = (weights[0] * grayscale_cam1 + weights[1] * grayscale_cam2 + weights[2] * grayscale_cam3 + weights[3] * grayscale_cam4) / sum(weights)
    grayscale_cam = grayscale_cam[0,:]
    
    if pred.item() == 0:
        grayscale_cam = grayscale_cam * 0.0
    
    # CAM 보여주기
    visualization = grayscale_cam.T.astype(np.float32)
    
    return visualization, pred

def pixel_map(images, label, model, device):
    # 각각의 캠 구하기
    camList = []
    image_prediction = 0
    for i in range(len(images)):
        for j in range(len(images[0])):
            cam_, pred_ = single_cam_(images[i][j], model, device)
            cam_ = preprocess(cam_)
            camList.append(cam_)
            if pred_.item() == 1:
                image_prediction = 1
            elif pred_.item() == 2 and image_prediction != 1:
                image_prediction = 2
    
    # 결과값이 안 맞는 경우만 출력
    if (image_prediction != label.item()):
        # 원본 이미지 합성
        imgCat = []
        for i in range(len(images)):
            for j in range(len(images[0])):
                imgCat.append(images[i][j].squeeze())

        temp = []
        num_rows = 3
        for i in range(num_rows):
            a = imgCat[int(len(imgCat)/num_rows)*i:int(len(imgCat)/num_rows)*(i+1)]
            temp.append(np.concatenate(tuple(a), axis=2))
        original = (np.concatenate(tuple(temp), axis=1).T).astype(np.uint8)

        # 원본 이미지 그리기
        fig, axes = plt.subplots(1, 2)
        axes[0].set_title("Label: " + labels_map[int(label)] + "\nPredict: " + labels_map[int(image_prediction)])
        axes[0].axis("off")
        axes[0].imshow(original, cmap="gray")

        # 크랙 찾은 결과
        temp = []
        num_rows = 3
        for i in range(num_rows):
            a = camList[int(len(camList)/num_rows)*i:int(len(camList)/num_rows)*(i+1)]
            temp.append(np.concatenate(tuple(a), axis=0))
        final = np.concatenate(tuple(temp), axis=1)
        axes[1].axis("off")
        axes[1].imshow(masking(final, edge_detection(original)), cmap="gray")
        return 0
    else:
        return 1