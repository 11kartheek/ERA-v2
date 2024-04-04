import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1. prep layer
        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 2. layer 1
        # layer 1 x
        self.X1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # layer1 res
        self.R1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 3. layer 2
        self.L2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 4. layer 3
        # layer 3 x
        self.X2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # layer1 res
        self.R2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # 5 max pool
        self.pool1 = nn.MaxPool2d(4, 4)

        # 6 fc
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # 1 prep
        x = self.prep(x)
        # 2
        x = self.X1(x)
        y = self.R1(x)
        x = x + y
        # 3
        x = self.L2(x)
        # 4
        x = self.X2(x)
        y = self.R2(x)
        x = x + y
        # 5 maxpool
        x = self.pool1(x)
        # 6 fc
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # 7 softmax
        return F.log_softmax(x, 1)
