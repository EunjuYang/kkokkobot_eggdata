# 계란 데이터 구축 사업 데이터 활용 AI 모델 Repository

본 레포지토리는 NIA의 인공지능 데이터 셋 구축 사업의 일환으로 진행된 계란 데이터 구축 사업에서 수행된 결과물인 데이터 셋의 활용 예시 모델을 제공합니다.
구축된 데이터는 세 종류로, `Detection`, `Classification`, `Regression` 용 데이터셋입니다.
`Detection` 데이터와 `Classification` 데이터는 각각 `MONO` 이미지와 `COLOR` 이미지를 가지고 있습니다.
`Classification` 데이터는 `Detection`데이터셋으로부터 구성된 데이터로 동일한 원천 데이터에 대해 입.출력은 다른 포맷을 가집니다.

## 계란 상태 자동 탐지 데이터 셋

`Detection` 폴더는 계란 상태 자동 탐지 데이터 셋을 학습 & 추론하는 코드를 포함합니다.
우리는 Faster R-CNN 모드와 DETR 모델의 학습 & 추론 코드를 제공합니다.


## 계란 상태 분류 데이터 셋

`Classification` 폴더는 입력되는 하나의 계란 이미지로부터 상태를 분류하는 분류기 모델의 학습 및 추론 코드를 포함하고있습니다.
우리는 전형적인 CNN 모델인 ResNet, ResNext, Xception 네트워크와 Transformer 모델인  ViT, SwinTransformer의 학습 및 추론 코드를 제공합니다.

## 계란 보관 일 수 예측 데이터 셋

`Prediction` 폴더는 입력되는 하나이 계란 이미지로부터 상온에 보관된 일 수를 예측하는 회귀 모델의 학습 및 추론 코드를 포함하고있습니다.
우리는 ResNet 모델과 FC Layer를 사용한 모델을 제공하고있습니다.


