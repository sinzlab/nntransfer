from .resnet import ResNet, compute_corr_matrix, Bottleneck

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# Used the implementation from https://github.com/CuthbertCai/pytorch_DANN
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.constant = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None


def grad_reverse(x, lambda_p):
    return GradReverse.apply(x, lambda_p)


class NoiseAdvResNet(ResNet):
    def __init__(self, block, num_blocks, num_classes: int = 10, classification: bool = False):
        super().__init__(block, num_blocks, num_classes=num_classes)
        self.noise_readout = nn.Linear(512 * block.expansion, 1)
        self.classification = classification

    def forward(self, x, compute_corr: bool = False, seed: int = None, noise_lambda=None):
        core_out, corr_matrices = self.core(x, compute_corr=compute_corr, seed=seed)
        out = self.linear_readout(core_out)
        noise_out = self.noise_readout(grad_reverse(core_out, noise_lambda))  # additional noise prediction
        if self.classification:
            noise_out = F.sigmoid(noise_out)
        return out, corr_matrices, noise_out
