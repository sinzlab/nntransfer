import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LeNet5(
    nn.Module
):  # adapted from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(
        self,
        num_classes: int = 10,
        input_width: int = 28,
        input_height: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
    ):
        super(LeNet5, self).__init__()
        self.input_size = (input_height, input_width)
        conv_out_width = int(
            ((((input_width - 3) + 1) / 2 - 3) + 1) / 2
        )  # [(W-K+2P)/S]+1 / MP
        conv_out_height = int(
            ((((input_height - 3) + 1) / 2 - 3) + 1) / 2
        )  # [(H-K+2P)/S]+1 / MP
        self.flat_feature_size = (conv_out_height * conv_out_width) * 16
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.flat_feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x) if self.dropout else x
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = self.dropout(x) if self.dropout else x
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.flat_feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if self.dropout else x
        x = self.fc3(x)
        return x


class LeNet300100(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_width: int = 28,
        input_height: int = 28,
        input_channels: int = 1,
        dropout: float = 0.0,
    ):
        super(LeNet300100, self).__init__()
        self.input_size = (input_height, input_width)
        self.flat_input_size = input_width * input_height * input_channels
        self.fc1 = nn.Linear(self.flat_input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):
        x = x.view(x.size(0), self.flat_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout(x) if self.dropout else x
        x = self.fc3(x)
        return x


def lenet_builder(seed: int, config):
    if "5" in config.type:
        lenet = LeNet5
    elif "300-100" in config.type:
        lenet = LeNet300100

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    model = lenet(
        num_classes=config.num_classes,
        input_width=config.input_width,
        input_height=config.input_height,
        input_channels=config.input_channels,
        dropout=config.dropout,
    )
    return model
