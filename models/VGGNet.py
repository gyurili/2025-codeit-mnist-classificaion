import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# VGGNet 모델 정의
class MNISTVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTVGG, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 모델 로딩 함수
def load_VGG_model(best=False, path='best_VGG_model.pth'):
    model = MNISTVGG().to(device)
    if best:
        path = os.path.join(BASE_DIR, "models/VGG_models", path)
        model.load_state_dict(torch.load(path))
    return model


# 학습 함수
def train_VGG(model, criterion, optimizer, train_loader, val_loader, epochs=10, path='best_VGG_model.pth'):
    best_acc = 0.0
    os.makedirs("models/VGG_models", exist_ok=True)  # 폴더가 없으면 생성
    path = os.path.join(BASE_DIR, "models/VGG_models", path)
    
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
def test_VGG(test_loader, path='best_VGG_model.pth', print=True):
    model = load_VGG_model(best=True, path=path)
    test_acc = evaluate(model, test_loader)
    if print:
        print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


# 학습 실행 함수
def run_VGG(train_loader, val_loader, test_loader, lr=0.001, epochs=1, path='best_VGG_model.pth'):
    # 모델 초기화
    model = load_VGG_model()
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_VGG(model, criterion, optimizer, train_loader, val_loader, epochs, path)
    test_VGG(test_loader, path=path)


# 랜덤 샘플 시각화 함수
def VGG_visualize_predictions(data_loader, num_samples=5, path='best_VGG_model.pth'):
    model = load_VGG_model(best=True, path=path)
    
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
