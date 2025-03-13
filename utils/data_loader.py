import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import *
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# 난수 고정은 안 함 (난수 설정에 따라 데이터셋이 다르기 때문에 결과가 달라질 수 있음)
def make_loader(batch_size=64, val_ratio=0.2):
    train_dataset =load_train_dataset()
    test_dataset = load_test_dataset()

    # 데이터셋을 훈련용과 검증용으로 나누기
    train_size = int(len(train_dataset) * (1 - val_ratio))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 생성한 데이터 로더 시각화 함수
def visualize_train_loader(train_loader):

    data_iter = iter(train_loader)  # 데이터 로더에서 첫 배치 가져오기
    images, labels = next(data_iter)

    fig, axes = plt.subplots(1, 5, figsize=(12, 4))  # 1행 5열의 서브플롯 생성
    for i in range(5):
        img = images[i].squeeze(0)  # [1, 28, 28] -> [28, 28] 변환
        axes[i].imshow(img, cmap='gray')  # 흑백 이미지로 표시
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')  # 축 제거

    plt.show()