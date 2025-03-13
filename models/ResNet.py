import torch.nn as nn
from torchvision import models

# ResNet-18 모델 정의
class MNISTResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 흑백 이미지 지원
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # MNIST는 10개 클래스
    
    def forward(self, x):
        return self.model(x)