import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import DEVICE, load_model

# 랜덤 샘플 시각화 함수
def visualize_predictions(model_type, data_loader, num_samples=5, path=''):
    model = load_model(model_type, best=True, path=path)
    
    model.eval()
    images, labels = next(iter(data_loader))
    indices = np.random.choice(len(images), num_samples, replace=False)
    images, labels = images[indices], labels[indices]
    
    images = images.to(DEVICE)
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

# 학습 손실 시각화 함수
def visualize_loss(model_type, loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label=f"{model_type} Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss of {model_type}")
    plt.legend()
    plt.grid(True)
    plt.show()