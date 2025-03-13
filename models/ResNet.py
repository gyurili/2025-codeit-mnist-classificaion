import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ResNet-18 모델 정의
class MNISTResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 흑백 이미지 지원
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # MNIST는 10개 클래스
    
    def forward(self, x):
        return self.model(x)

# 모델 함수
def load_ResNet_model(best=False, path='best_ResNet_model.pth'):
    model = MNISTResNet().to(device)
    if best:
        path = os.path.join(BASE_DIR, "models/ResNet_models", path)
        model.load_state_dict(torch.load(path))
    return model

# 학습 함수
def train(model, criterion, optimizer, train_loader, val_loader, epochs=10, path='best_ResNet_model.pth'):
    best_acc = 0.0
    os.makedirs("ResNet_models", exist_ok=True)  # 폴더가 없으면 생성
    path = os.path.join(BASE_DIR, "models/ResNet_models", path)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        val_acc = evaluate(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), path)
            print("Best model saved!")

# 검증 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

# 테스트 함수
def test_ResNet(test_loader):
    model = load_ResNet_model(best=True)
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# 학습 실행 함수
def run_ResNet(train_loader, val_loader, test_loader, lr=0.001, only_test=False):
    # 모델 초기화
    model = load_ResNet_model()
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if only_test:
        test_acc = test_ResNet(model, test_loader)
        return test_acc
    
    train(model, criterion, optimizer, train_loader, val_loader)
    test_ResNet(model, test_loader)

# 랜덤 샘플 시각화 함수
def ResNet_visualize_predictions(data_loader, num_samples=5):
    model = load_ResNet_model(best=True)
    
    model.eval()
    images, labels = next(iter(data_loader))
    indices = np.random.choice(len(images), num_samples, replace=False)
    images, labels = images[indices], labels[indices]
    
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Pred: {predictions[i].item()}\nLabel: {labels[i].item()}")
        ax.axis("off")
    plt.show()