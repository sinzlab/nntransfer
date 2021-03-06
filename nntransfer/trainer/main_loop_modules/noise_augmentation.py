import torch
import numpy as np
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms

from .main_loop_module import MainLoopModule


class NoiseAugmentation(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.rnd_gen = None
        if isinstance(self.train_loader, dict):
            loaders = self.train_loader
        else:
            loaders = self.train_loader.loaders
        for k, v in loaders.items():
            if "img_classification" in k:
                train_loader = v
        dataset = train_loader.dataset
        if isinstance(dataset, ConcatDataset):
            dataset = dataset.datasets[0]
        transform_list = dataset.transforms.transform.transforms
        # go through StandardTransform and Compose to get to  the actual transforms
        normalization = None
        for trans in transform_list:
            if isinstance(trans, transforms.Normalize):
                normalization = trans
        if normalization:
            # image = (image - mean) / std
            # => noisy_image = (image + noise - mean) / std
            #                = (image - mean) / std + noise/ std
            img_mean = torch.tensor(normalization.mean)
            img_std = torch.tensor(normalization.std)
            self.img_min = -img_mean / img_std
            self.img_max = (1 - img_mean) / img_std
            self.noise_scale = 1 / img_std
            self.noise_scale = self.noise_scale.view(1, -1, 1, 1).to(self.device)
        else:
            self.img_min = 0
            self.img_max = 1
            self.noise_scale = None
        self.apply_to_eval = self.config.apply_noise_to_validation

    @staticmethod
    def apply_noise(
        x,
        device,
        std: dict = None,
        snr: dict = None,
        rnd_gen=None,
        img_min=0,
        img_max=1,
        noise_scale=None,
        in_place=False,
    ):
        """

        Args:
            x: input batch
            device: gpu/cpu
            std: dictionary of the for {noise_level_a: percent_a, noise_level_b: percent_b, ...}
                 determining the mixture of noise levels (in standard deviation) we will apply
            snr: dictionary of the for {noise_level_a: percent_a, noise_level_b: percent_b, ...}
                 determining the mixture of noise levels (in signal-to-noise ratio) we will apply
            rnd_gen: optional random generator
            img_min: minimum pixel value (to compensate for applying this after normalization)
            img_max: maximum pixel value (to compensate for applying this after normalization)
            noise_scale: optional factor to scale the noise magnitude
            in_place: option to apply the noise in place

        Returns:

            noise augmented batch, tensor of standard deviations that were applied to each sample
        """
        if not std and not snr:
            return x, torch.zeros([x.shape[0], 1], device=device)
        if not in_place:
            x = x.clone()
        if len(x.shape) == 3:
            x.unsqueeze(0)  # if we only have a single element
        with torch.no_grad():
            if std:
                noise_levels = std
            elif snr:
                noise_levels = snr
            else:
                noise_levels = {-1: 1.0}
            assert (
                abs(sum(noise_levels.values()) - 1.0) < 0.00001
            ), "Percentage for noise levels should sum to one!"
            indices = torch.randperm(x.shape[0])
            applied_std = torch.zeros([x.shape[0], 1], device=device)
            start = 0
            for (
                level,
                percentage,
            ) in noise_levels.items():  # TODO: is this efficient enough?
                end = start + int(percentage * x.shape[0])
                if isinstance(level, tuple):  # select level randomly from range
                    level = torch.empty(1, device=device, dtype=torch.float32).uniform_(
                        level[0], level[1]
                    )
                if level > 0:  # select specified noise level for a fraction of the data
                    if std is None:  # are we doing snr or std?
                        signal = torch.mean(
                            x[indices[start:end]] * x[indices[start:end]],
                            dim=[1, 2, 3],
                            keepdim=True,
                        )  # for each dimension except batch
                        std = signal / level
                    else:
                        if not isinstance(level, torch.Tensor):
                            std = torch.tensor(level, device=device)
                        else:
                            std = level
                    applied_std[indices[start:end]] = std.squeeze().unsqueeze(-1)
                    std = std.expand_as(x[start:end])
                    noise = torch.normal(mean=0.0, std=std, generator=rnd_gen)
                    if noise_scale is not None:
                        noise *= noise_scale
                    x[indices[start:end]] += noise
                # else: deactivate noise for a fraction of the data
                start = end
            if isinstance(img_max, torch.Tensor):
                for i in range(
                    img_max.shape[0]
                ):  # clamp each color channel individually
                    x[:, i] = torch.clamp(x[:, i], max=img_max[i], min=img_min[i])
            else:
                x = torch.clamp(x, max=img_max, min=img_min)
        return x, applied_std

    def pre_epoch(self, model, mode, **options):
        super().pre_epoch(model, mode, **options)
        if not self.train_mode:
            rnd_gen = torch.Generator(device=self.device)
            if isinstance(self.seed, np.generic):
                self.seed = np.asscalar(self.seed)
            self.rnd_gen = rnd_gen.manual_seed(
                self.seed
            )  # so that we always have the same noise for evaluation!

    def pre_forward(self, model, inputs, task_key, shared_memory):
        super().pre_forward(model, inputs, task_key, shared_memory)
        noise_std = self.options.get("noise_std", self.config.noise_std)
        noise_snr = self.options.get("noise_snr", self.config.noise_snr)
        inputs, shared_memory["applied_std"] = self.apply_noise(
            inputs,
            self.device,
            std=noise_std,
            snr=noise_snr,
            rnd_gen=self.rnd_gen if not self.train_mode else None,
            img_min=self.img_min,
            img_max=self.img_max,
            noise_scale=self.noise_scale,
        )
        return model, inputs
