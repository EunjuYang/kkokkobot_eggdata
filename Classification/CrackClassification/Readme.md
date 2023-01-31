# 계란 균열 탐지 및 분류

## 환경 준비 및 실행

### 제공된 컨테이너 없이 로컬 환경에서 실행하기

#### 1. `Miniconda` 설치하기

```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

#### 2. `Miniconda`를 사용한 가상환경 생성
본 repository에 있는 conda_requirements.yaml은 학습에 필요한 python package 정보들을 포함하고 있습니다. 아래 명령어를 사용하여 자신의 로컬 환경에서 본 모델을 사용할 수 있는 환경을 구축할 수 있습니다.

```shell
$ conda env create -f conda_requirements.yaml
$ conda activate kko
```

### 제공된 컨테이너 환경에서 실행하기

#### 1. 컨테이너 이미지 설치하기
```shell
$ docker import kkokkobot-crack-classification.tar - kkokkobot/crackclassification:devel
```

#### 2. 컨테이너 이미지 실행하기
```shell
$ docker run -it --gpus="device=0" -v [datapath]:/root/sourceCode/data kkokkobot/crackclassification:devel /bin/bash
```

## 모델
본 repository에서는 데이터 셋 사용 예시로써 ResNet-50을 backbone으로 ㅎ나 patch-wise crack detection & classification 모델을 제공합니다. 탐지 및 분류를 위해서는 patch 데이터셋에서 학습된 모델의 pt 파일이 필요합니다. 본 repository에는 실행을 위한 예시 pt 파일이 하나 제공되며, 직접 patch 라벨링을 진행 후 모델을 학습한 뒤 실행해볼 수 있게 라벨링 툴 및 학습 툴을 제공합니다.

본 모델 실행에는 세가지 인자가 필요합니다.
`--checkpoint`는 학습된 pt파일의 경로를 요구합니다.
`--tgt_dir`은 탐지 및 추론에 사용될 데이터셋의 경로를 요구합니다.
`--num_examples`은 Normal, Latent Crack, Crack 별 탐지 결과 예시를 보일 개수를 요구합니다. Default 값은 0입니다.

`--num_examples` 인자를 통하여 예시를 보고 싶은 경우에는 vnc 또는 x11 등을 통한 Desktop 환경이 필요합니다.

### 실행
```shell
$ python main.py --checkpoint [checkpoint 경로] --tgt_dir [데이터셋 경로] (--num_examples [표시할 예시 개수])
```

#### 1. 라벨링 툴을 활용하여 Patch 데이터셋 라벨 생성
해당 라벨링 툴은 PyQT를 활용하기 때문에 Desktop 환경에서 실행되어야합니다.
해당 툴은 데이터셋이 저장된 서버와 별도의 컴퓨터에서 SSH를 통해 접속하여 사용할 것을 전제로 작성되었으며, 데이터셋이 다음과 같은 형태로 저장되어 있을 것을 요구합니다.

Root 경로
├── 0 // Normal 계란 경로
│   ├── 0_0
│   ├── 0_1
│   ├── 0_2
│   └── 0_3
├── 2 // Crack 계란 경로
│   ├── 2_0
│   ├── 2_1
│   ├── 2_2
│   ├── 2_3
│   ├── 2_4
│   ├── 2_5
│   ├── 2_6
│   └── 2_7
└── 6 // Latent Crack 계란 경로
    ├── 6_0
    ├── 6_1
    ├── 6_2
    └── 6_3

툴 실행에 앞서 feedback.py 파일 내의 SSH 정보를 작성해야합니다.

```shell
$ python feedback.py
```

#### 2. 본 모델용 학습 데이터셋 생성
본 모델은 Patch 기반 학습 때문에 학습을 위해 별도의 데이터셋을 필요로합니다.
앞서 라벨링 툴을 사용하여 라벨이 생성된 데이터셋을 필요로 하고 이를 패치 단위로 나누어 저장하는 툴을 활용하여 패치 데이터셋을 생성합니다

실행에 앞서 PatchDatasetCreator.py 내에서 타겟 데이터셋과 저장할 위치를 지정해줘야하고 타겟 데이터셋의 구조를 조금 수정할 필요가 있습니다.
타겟 데이터셋 구조
├── 0 // Normal 계란 경로
│   ├── 0_0
│   ├── 0_1
│   ├── 0_2
│   └── 0_3
├── 1 // Crack 계란 경로
│   ├── 1_0
│   ├── 1_1
│   ├── 1_2
│   ├── 1_3
│   ├── 1_4
│   ├── 1_5
│   ├── 1_6
│   └── 1_7
└── 2 // Latent Crack 계란 경로
    ├── 2_0
    ├── 2_1
    ├── 2_2
    └── 2_3

```shell
$ python PatchDatasetCreator.py
```

#### 3. 본 모델 학습

#### 4. 본 모델 탐지 및 분류 데이터셋 생성
본 모델을 활용하여 균열을 탐지 및 분류하기 위해서는 데이터셋을 알맞은 형태로 가공할 필요가 있다.
DatasetCreator.py 파일 내의 타겟 데이터셋 경로 및 저장 경로를 알맞게 설정한 다음 실행하면 탐지 및 분류에 필요한 형태로 데이터셋이 가공된다.
이 때, 타겟 데이터셋은 앞선 2에서의 타겟 데이터셋과 같은 형태로 폴더 구조가 짜여있어야한다.

```shell
$ python DatasetCreator.py
```

#### 5. 컨테이너 안에서 모델을 활용하여 탐지 및 분류하기
```shell
$ cd /root/sourceCode
$ python main.py --checkpoint ckp_best.pt --tgt_dir ./data
```
