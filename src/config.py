import os
import torch
from models.Scratch import FromScratch
from models.VGGNet import MNISTVGG
from models.ResNet import MNISTResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model_class = {
    'Scratch' : FromScratch(),
    'VGGNet' : MNISTVGG(),
    'ResNet' : MNISTResNet()
}

model_dir = {
    'Scratch' : 'Scratch_models',
    'VGGNet' : 'VGGNet_models',
    'ResNet' : 'ResNet_models'
}

# 모델 함수
def load_model(model_type, best=False, path='best_model.pth'):
    model = model_class[model_type].to(DEVICE)
    if best:
        path = os.path.join(MODELS_DIR, model_dir[model_type], path)
        model.load_state_dict(torch.load(path))
    return model