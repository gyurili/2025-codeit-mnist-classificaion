import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import DEVICE, load_model

# 랜덤 샘플 시각화 함수
def visualize_predictions(data_loader, num_samples=5, path=''):
    model = load_model(best=True, path=path)
    
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