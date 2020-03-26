from functools import partial

from .base import BaseConfig
from nnfabrik.main import *


class ModelConfig(BaseConfig):
    config_name = "model"
    table = Model()
    fn = "bias_transfer.models.resnet_builder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        self.type = kwargs.pop("type", 50)
        self.input_size = kwargs.pop("input_size", 32)
        if self.input_size == 32:
            self.core_stride = 1
        elif self.input_size == 64:
            self.core_stride = 2
        self.conv_stem_kernel_size = kwargs.pop("conv_stem_kernel_size", 3)
        self.num_classes = kwargs.pop("num_classes", 100)
        self.update(**kwargs)


class CIFAR100(ModelConfig):
    pass


class CIFAR10(ModelConfig):
    def __init__(self, **kwargs):
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=10, **kwargs)


class TinyImageNet(ModelConfig):
    def __init__(self, **kwargs):
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=200, input_size=64, conv_stem_kernel_size=5, **kwargs)
