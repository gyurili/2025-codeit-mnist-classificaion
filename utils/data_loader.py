from data.dataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def make_loader(batch_size=64):
    train_dataset =load_train_dataset()
    test_dataset = load_test_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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