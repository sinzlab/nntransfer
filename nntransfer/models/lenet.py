import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    """
    Adapted from: https://discuss.pytorch.org/t/locally-connected-layers/26979/2
    """

    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=False
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size ** 2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = 0
        self.dilation = 1

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


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
        core: str = "conv",
    ):
        super(LeNet5, self).__init__()
        self.input_size = (input_height, input_width)
        conv_out_width1 = (input_width - 3) + 1
        conv_out_width = (
            conv_out_width1 // 2 - 3
        ) + 1  # [(W-K+2P)/S]+1  (includes max pool before)
        conv_out_height1 = (input_height - 3) + 1
        conv_out_height = (
            conv_out_height1 // 2 - 3
        ) + 1  # [(H-K+2P)/S]+1  (includes max pool before)
        flat_feature_size = ((conv_out_height // 2) * (conv_out_width // 2)) * 16
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.core_type = core
        if core == "lc":
            self.conv1 = LocallyConnected2d(
                input_channels,
                6,
                output_size=(conv_out_height1, conv_out_width1),
                kernel_size=3,
            )
            self.conv2 = LocallyConnected2d(
                6,
                16,
                output_size=(conv_out_height, conv_out_width),
                kernel_size=3,
            )
        elif core == "fc":
            flat_input_size = input_width * input_height * input_channels
            intermediate_size = (conv_out_height1 // 2) * (conv_out_width1 // 2) * 6
            self.conv1 = nn.Sequential(
                nn.Flatten(), nn.Linear(flat_input_size, intermediate_size)
            )
            self.conv2 = nn.Linear(intermediate_size, flat_feature_size)
        else:  # core == "conv":
            self.conv1 = nn.Conv2d(input_channels, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(flat_feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x) if self.dropout else x
        if not self.core_type == "fc":
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = self.dropout(x) if self.dropout else x
        if not self.core_type == "fc":
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
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
        core_type: str = "fc",
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
        core=config.core_type,
    )
    return model
