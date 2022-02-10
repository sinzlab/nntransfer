import numpy as np
import torch

from .dataset_filter import DatasetFilter


class ExamplesFilter(DatasetFilter):
    def __init__(self, config, train_dataset):
        super().__init__(config, train_dataset)
        examples_per_class = config.examples_per_class
        repeats = config.repeats_per_epoch

        new_data_idx = []
        for cls in train_dataset.classes:
            if not isinstance(cls, int):
                cls = int(''.join(c for c in cls if c.isdigit()))
            idx = (train_dataset.targets == cls).nonzero(as_tuple=False)
            new_data_idx.append(idx[:examples_per_class])
        new_data_idx = torch.cat(new_data_idx, dim=0)
        new_data_idx = new_data_idx.flatten()
        new_data_idx = new_data_idx.repeat((repeats,))
        if hasattr(train_dataset, "data"):
            train_dataset.data = train_dataset.data[new_data_idx]
        else:
            train_dataset.samples = train_dataset.samples[new_data_idx]
        train_dataset.targets = train_dataset.targets[new_data_idx]
        print(train_dataset)
        print(train_dataset.targets.shape)
        print(train_dataset.data.shape)

    def apply(self, dataset):
        pass
