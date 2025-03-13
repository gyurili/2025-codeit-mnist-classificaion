import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE, MODELS_DIR, model_dir, load_model

# 학습 함수
def train_model(model, model_type, criterion, optimizer, train_loader, val_loader, epochs=10, path='best_model.pth'):
    best_acc = 0.0
    loss_history = []
    
    dir = os.path.join(MODELS_DIR, model_dir[model_type])
    os.makedirs(dir, exist_ok=True)  # 폴더가 없으면 생성
    path = os.path.join(dir, path)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), path)
            print("Best model saved!")
    return loss_history

# 검증 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

# 테스트 함수
def test_model(model_type, test_loader, path='best_model.pth', print=True):
    model = load_model(model_type=model_type, best=True, path=path)
    test_acc = evaluate(model, test_loader)
    if print:
        print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# 학습 실행 함수
def run_model(model_type, train_loader, val_loader, test_loader, lr=0.001, epochs=10, path=''):
    # 모델 초기화
    model = load_model(model_type=model_type)
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = train_model(model, model_type, criterion, optimizer, train_loader, val_loader, epochs, path)
    test_model(model_type, test_loader, path=path)
    
    return loss_history