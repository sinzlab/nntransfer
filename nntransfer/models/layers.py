import torch
from torch import nn as nn
from torch.nn.modules.utils import _pair


class LocallyConnected1d(nn.Module):
    """
    Adapted from: https://discuss.pytorch.org/t/locally-connected-layers/26979/2
    """

    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=False
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size,
                kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size)
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 0
        self.dilation = 1

    def forward(self, x):
        _, c, l = x.size()
        kl = self.kernel_size
        dl = self.stride
        x = x.unfold(2, kl, dl)
        # x = x.contiguous().view(*x.size()[:-1], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


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