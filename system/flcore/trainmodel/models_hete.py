import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


# ====================================================================================================================


"""
模型异构的情况
"""
class FedAvgCNN_Hetero(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024, num_data=0):
        mid_dim = 0
        super().__init__()
        num_conv_layers = 0
        if num_classes == 10:
            if num_data <= 2000:
                FedAvgCNN_2conv_layers(in_features, num_classes, dim, num_data)
            elif num_data <= 4000:
                FedAvgCNN_3conv_layers(in_features, num_classes, dim, num_data)
            else:
                FedAvgCNN_4conv_layers(in_features, num_classes, dim, num_data)
        elif num_classes == 100:
            if num_data <= 2100:
                FedAvgCNN_2conv_layers(in_features, num_classes, dim, num_data)
            elif num_conv_layers <= 2300:
                FedAvgCNN_3conv_layers(in_features, num_classes, dim, num_data)
            else:
                FedAvgCNN_4conv_layers(in_features, num_classes, dim, num_data)
        elif num_classes == 200:
            if num_data <= 4000:
                FedAvgCNN_2conv_layers(in_features, num_classes, dim, num_data)
            elif num_conv_layers <= 4200:
                FedAvgCNN_3conv_layers(in_features, num_classes, dim, num_data)
            else:
                FedAvgCNN_4conv_layers(in_features, num_classes, dim, num_data)

class FedAvgCNN_2conv_layers(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
        ])
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc(x)

        return x

class FedAvgCNN_3conv_layers(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # 卷积层+激活函数+池化
        x = self.conv_layers(x)
        # 将特征图展平
        x = x.view(-1, 128 * 4 * 4)
        # 全连接层+激活函数
        x = self.fc(x)
        return x


class FedAvgCNN_4conv_layers(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_features, 64, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            ),

        ])
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc(x)

        return x
