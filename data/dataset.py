from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomRotation, Resize


# 테스트 데이터셋 생성 함수 - 정규화 적용
def load_train_dataset(use_aug=True):
    transform = select_augmentation() if use_aug else Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    return dataset


# 테스트 데이터셋 생성 함수 - 정규화 적용
def load_test_dataset():
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))  # [-1, 1] 범위로 정규화
    ])

    dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    return dataset


def select_augmentation(image_size=28):
    """
    데이터 증강을 선택하는 함수
    return: 선택된 변환을 적용한 Compose 객체 - transform에 바로 적용 가능
    """
    crop_ratio, rotate_degree = None, None # 적용 여부 확인을 위한 None 초기화
    
    while True:  # 잘못된 값 입력 시 반복을 위함
        crop = input("Do you want to use Crop Augmentation? (y/n): ").strip().lower()
        # 입력값 에러 방지를 위해 strip과 소문자로 변환 적용
        
        if crop in ('y', 'n'):
            if crop == 'y':
                try:
                    crop_ratio = float(input("Enter the Crop Augmentation ratio (0~1): ").strip())
                    if 0 <= crop_ratio <= 1:
                        break
                except ValueError:
                    pass
            else:
                break
        print("Invalid input. Please try again.")
    
    while True:  # 잘못된 값 입력 시 반복을 위함
        rotate = input("Do you want to use Rotate Augmentation? (y/n): ").strip().lower()
        # 입력값 에러 방지를 위해 strip과 소문자로 변환 적용

        if rotate in ('y', 'n'):
            if rotate == 'y':
                try:
                    rotate_degree = int(input("Enter the Rotate Augmentation angle (0~360): ").strip())
                    if 0 <= rotate_degree <= 360:
                        break
                except ValueError:
                    pass
            else:
                break
        print("Invalid input. Please try again.")
    
    transforms = []
    if crop_ratio is not None:
        crop_size = max(1, int(image_size * crop_ratio))  # 최소 1x1 크기로 제한
        transforms.append(RandomCrop(int(crop_size)))  
        #  크롭 적용 - default image_size=28

        transforms.append(Resize((image_size, image_size)))
        # 크롭 후 이미지 크기 복원
        
    if rotate_degree is not None:
        
        transforms.append(RandomRotation(rotate_degree))
    
    transforms.extend([ToTensor(), Normalize((0.5,), (0.5,))])
    return Compose(transforms)