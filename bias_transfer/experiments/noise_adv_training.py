from . import Description
from bias_transfer.configs.transfer_experiment import TransferExperiment
from bias_transfer.configs.experiment import Experiment
from bias_transfer.configs import model, dataset, trainer
from . import baseline

experiments = {}
transfer_experiments = {}

for seed in (8,
             13,
             42):
    # Noise augmentation:
    for noise_type in (
            {"add_noise": True, "noise_snr": None,
             "noise_std": {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}},
    ):
        for noise_adv in (
                {"noise_adv_classification": False, "noise_adv_regression": False},
                {"noise_adv_classification": True, "noise_adv_regression": False},
                {"noise_adv_classification": False, "noise_adv_regression": True},):
            for loss_factor in (1.0,
                                0.1,
                                10.0,
                                2.0,
                                0.5,
                                5.0
                                ):
                for gamma in (
                        10.0,
                        50.0,
                        1.0,
                        5.0
                ):
                    name = "Noise"
                    if noise_adv["noise_adv_classification"]:
                        name += " + Adv Classification"
                        m = model.CIFAR100(description="Noise Adv Classification",
                                           noise_adv_classification=True)
                    else:
                        name += " + Adv Regression"
                        m = model.CIFAR100(
                            description="Noise Adv Regression", noise_adv_regression=True)
                    name += " (lambda {} gamma {})".format(loss_factor, gamma)
                    experiments[
                        Description(name=name, seed=seed)] = Experiment(
                        dataset=dataset.CIFAR100(description=""),
                        model=m,
                        trainer=trainer.TrainerConfig(description=name,
                                                      noise_adv_loss_factor=loss_factor,
                                                      noise_adv_gamma=10.0,
                                                      **noise_adv,
                                                      **noise_type),
                        seed=seed)

                    # The transfer experiments:
                    transfer_experiments[Description(name=name, seed=seed)] = TransferExperiment(
                        [baseline.experiments[Description(name="Clean", seed=seed)],
                         experiments[Description(name=name, seed=seed)],
                         baseline.experiments[Description(name="Transfer", seed=seed)]])
                    transfer_experiments[Description(name=name + "-> Reset", seed=seed)] = TransferExperiment(
                        [baseline.experiments[Description(name="Clean", seed=seed)],
                         experiments[Description(name=name, seed=seed)],
                         baseline.experiments[Description(name="Transfer + Reset", seed=seed)]])