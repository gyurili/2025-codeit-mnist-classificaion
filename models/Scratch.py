import torch
import torch.nn as nn

class FromScratch(nn.Module):
    def __init__(self, num_classes = 10):
        super(FromScratch, self).__init__()
        self.features = nn.Sequential(
            # 첫 번째 Convolution: 입력 채널 1, 출력 채널 96, 커널 크기 11, padding = "same".
            nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 11, padding = "same"),
            nn.BatchNorm2d(num_features = 96,  eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(kernel_size = (2, 2)),
            nn.Dropout(p = 0.3, inplace = True),

            # 두 번째 Convolution: 입력 96, 출력 384, 커널 크기 3, padding = "same".
            nn.Conv2d(in_channels = 96, out_channels = 384, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(num_features = 384,  eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),

            # 세 번째 Convolution: 입력 384, 출력 384, 커널 크기 3, padding = "same".
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(num_features = 384,  eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),

            # 네 번째 Convolution: 입력 384, 출력 256, 커널 크기 3, padding = "same".
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(num_features = 256,  eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(kernel_size = (2, 2)),
            nn.Dropout(p = 0.3, inplace = True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 3200),
            nn.BatchNorm1d(num_features = 3200,  eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(3200, 3200),
            nn.BatchNorm1d(num_features = 3200,  eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(3200, num_classes)
        )

    def forward(self, input):
        output = self.features(input)
        # 첫 번째 차원을 제외하고 flatten합니다.
        output = torch.flatten(output, 1)
        probabilities = self.classifier(output)
        return probabilities

# 모델 인스턴스 생성 및 테스트 예시
if __name__ == "__main__":
    model = FromScratch(num_classes = 10)
    print(model)

    # 임의의 입력 (배치 크기 64, 채널 1, 28 x 8 이미지)
    test = torch.randn(64, 1, 28, 28)
    output = model(test)
    print("출력 크기:", output.shape)  # 예상 출력: torch.Size([64, 10])