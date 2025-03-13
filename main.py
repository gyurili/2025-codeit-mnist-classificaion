from data import *
from models import *
from src.config import *
from src.train import *
from src.visualization import *
from utils.data_loader import *

# Data-loader 불러오기.
train_loader, val_loader, test_loader = make_loader()

# MNIST 데이터 시각화하기.
visualize_train_loader(train_loader)

# 모델 명 list.
models = ["Scratch", "VGGNet", "ResNet"]

# 모델 선택. 
while True:
    model_type = input("Select a model (Scratch, VGGNet, ResNet) : ")
    if model_type in models:
        break
    else:
        print("Invalid input. Please try again.")

model = load_model(model_type)

# 선택한 모델 학습하고 시각화하기.
run = run_model(model_type, train_loader, val_loader, test_loader, lr=0.001, epochs=10, path='best_model.pth')
visualize_predictions(model_type, test_loader, num_samples=5, path='best_model.pth')