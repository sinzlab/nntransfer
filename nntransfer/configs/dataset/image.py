from typing import Dict, Tuple

from nntransfer.configs.dataset.base import DatasetConfig
from nntransfer.tables.nnfabrik import Dataset


class ImageDatasetConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.torchvision_dataset_loader"

    data_mean_defaults = {
        "CIFAR100": (
            0.5070751592371323,
            0.48654887331495095,
            0.4409178433670343,
        ),
        "CIFAR10": (0.49139968, 0.48215841, 0.44653091),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "MNIST": (0.1307,),
    }
    data_std_defaults = {
        "CIFAR100": (
            0.2673342858792401,
            0.2564384629170883,
            0.27615047132568404,
        ),
        "CIFAR10": (0.24703223, 0.24348513, 0.26158784),
        "SVHN": (0.1980, 0.2010, 0.1970),
        "MNIST": (0.3081,),
    }

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.dataset_cls: str = "CIFAR10"
        self.apply_augmentation: bool = True
        self.apply_normalization: bool = True
        self.apply_grayscale: bool = False
        self.apply_noise: Dict = {}
        self.convert_to_rgb: bool = False
        self.input_width: int = 32
        self.input_height: int = 32
        self.add_corrupted_test: bool = False
        self.add_stylized_test: bool = False
        self.use_c_test_as_val: bool = False
        self.show_sample: bool = False
        self.filter_classes: Tuple = ()  # (start,end)
        self.data_dir: str = "./data/image_classification/torchvision/"
        self.num_workers: int = 1
        dataset_id = (
            f"{self.dataset_sub_cls}_{self.bias}" if self.bias else self.dataset_cls
        )
        dataset_id += "_bw" if self.apply_grayscale else ""
        if not self.train_data_mean:
            self.train_data_mean: Tuple[float] = self.data_mean_defaults[dataset_id]
        if not self.train_data_std:
            self.train_data_std: Tuple[float] = self.data_std_defaults[dataset_id]

        super().__init__(**kwargs)

    @property
    def filters(self):
        filters = []
        if self.filter_classes:
            filters.append("ClassesFilter")
        return filters
