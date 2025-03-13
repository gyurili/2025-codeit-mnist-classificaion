# 📌 MNIST Classification: From Scratch vs. VGGNet vs. ResNet

## 📖 프로젝트 개요
이 프로젝트는 **MNIST 손글씨 숫자 데이터셋**을 사용하여 숫자를 분류하는 다양한 모델을 비교하는 실험입니다.
다음 세 가지 방식으로 모델을 구현하여 성능을 비교합니다:

1. **From Scratch 모델**
2. **VGGNet 기반 모델**
3. **ResNet 기반 모델**

각 모델을 동일한 데이터셋으로 학습한 후, 성능을 비교하고 분석합니다.

---

## 📂 폴더 및 파일 구조
```
2025-codeit-mnist-classification/
│── .github/                 # GitHub 관련 설정 파일
│── data/
│   ├── dataset.py           # 데이터셋 로딩 및 전처리
│
│── models/                  # 모델 구현 파일
│   ├── ResNet.py            # ResNet 기반 모델
│   ├── Scratch.py           # From Scratch 방식 모델
│   ├── VGGNet.py            # VGGNet 기반 모델
│
│── src/                     # 주요 코드 모듈
│   ├── config.py            # 설정값 및 모델 로드
│   ├── train.py             # 모델 학습 코드
│   ├── visualization.py     # 결과 시각화 코드
│
│── utils/                   # 유틸리티 모듈
│   ├── data_loader.py       # 데이터 로더 구현
│
│── .gitignore               # Git 무시 파일 목록
│── environment.yml          # 프로젝트 환경 설정 파일
│── main.py                  # 메인 실행 스크립트
│── README.md                # 프로젝트 설명 (이 파일)
```

---

## 🛠 실행 방법
### 1️⃣ **프로젝트 환경 설정**
먼저 프로젝트 환경을 설정해야 합니다.
```bash
conda env create -f environment.yml
conda activate mnist-classification
```

### 2️⃣ ** git 클론 다운로드**
```bash
git clone https://github.com/gyurili/2025-codeit-mnist-classificaion.git
cd 2025-codeit-mnist-classificaion
```

### 3️⃣ **IDE 실행**
IDE를 통해 main.py를 열고 실행하면 됩니다.

---

## 🔍 실험 결과
| 모델         | 정확도 (%) | 학습 시간 |
|-------------|----------|----------|
| From Scratch | XX.XX%   | XX분 XX초 |
| VGGNet      | XX.XX%   | XX분 XX초 |
| ResNet      | XX.XX%   | XX분 XX초 |