from models.VGGNet import run_VGG
from utils.data_loader import make_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_loader(batch_size=64)
    run_VGG(train_loader, test_loader, test_loader, lr=0.001, epochs=10)
