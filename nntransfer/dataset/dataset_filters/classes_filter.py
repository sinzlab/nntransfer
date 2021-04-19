import numpy as np
import torch

from .dataset_filter import DatasetFilter


class ClassesFilter(DatasetFilter):
    def __init__(self, config, train_dataset):
        super().__init__(config, train_dataset)
        start, end = config.filter_classes

        if config.reduce_to_filtered_classes:  # e.g. if using separate heads
            self.filtered_classes = train_dataset.classes[start:end]
            self.start = start
        else:  # e.g. for shared head (all classes are kept)
            self.filtered_classes = train_dataset.classes
            self.start = 0

        self.filtered_classes_idx = list(range(start, end))
        self.percent_start = start / len(train_dataset.classes)
        self.percent_end = end / len(train_dataset.classes)

    def apply(self, dataset):
        if hasattr(dataset, "data"):
            samples = dataset.data
        else:
            samples = dataset.samples
        filtered_samples, filtered_targets = [], []
        for i, sample in enumerate(samples):
            if dataset.targets[i] in self.filtered_classes_idx:
                filtered_samples.append(sample)
                filtered_targets.append(dataset.targets[i] - self.start)
        if isinstance(samples,np.ndarray):
            filtered_samples = np.stack(filtered_samples)
        elif torch.is_tensor(samples):
            filtered_samples = torch.stack(filtered_samples)
        if isinstance(dataset.targets,np.ndarray):
            filtered_targets = np.stack(filtered_targets)
        elif torch.is_tensor(dataset.targets):
            filtered_targets = torch.stack(filtered_targets)
        if hasattr(dataset, "data"):
            dataset.data = filtered_samples
        else:
            dataset.samples = filtered_samples
        dataset.targets = filtered_targets
        dataset.classes = self.filtered_classes
        if hasattr(dataset, "start"):
            old_length = len(dataset)
            dataset.end = dataset.start + int(self.percent_end * old_length)
            dataset.start += int(self.percent_start * old_length)
        if hasattr(dataset, "target_transform"):
            dataset.target_transform = lambda x: x - self.start
