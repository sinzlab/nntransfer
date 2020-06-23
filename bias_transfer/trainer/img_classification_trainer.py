from functools import partial

from torch import nn

from bias_transfer.trainer.trainer import Trainer
from bias_transfer.trainer.utils import NBLossWrapper, get_subdict
from bias_transfer.utils import stringify
from mlutils import measures as mlmeasures
from mlutils.tracking import AdvancedMultipleObjectiveTracker
from mlutils.training import LongCycler
from nnvision.utility import measures
from nnvision.utility.measures import get_poisson_loss

from torch import nn, optim


def trainer(model, dataloaders, seed, uid, cb, eval_only=False, **kwargs):
    t = ImgClassificationTrainer(dataloaders, model, seed, uid, **kwargs)
    return t.train(cb)


class ImgClassificationTrainer(Trainer):
    def get_tracker(self):
        objectives = {
            "LR": 0,
            "Training": {
                "img_classification": {"loss": 0, "accuracy": 0, "normalization": 0}
            },
            "Validation": {
                "img_classification": {"loss": 0, "accuracy": 0, "normalization": 0}
            },
            "Test": {
                "img_classification": {"loss": 0, "accuracy": 0, "normalization": 0}
            },
        }
        tracker = AdvancedMultipleObjectiveTracker(
            main_objective=("img_classification", "accuracy"), **objectives
        )
        return tracker

    def get_training_controls(self):
        criterion, stop_closure = {}, {}
        for k in self.task_keys:
            if self.config.loss_weighing:
                pass
                # self.criterion[k] = XEntropyLossWrapper(
                #     getattr(nn, self.config.loss_functions[k])()
                # ).to(self.device)
            else:
                criterion[k] = getattr(nn, self.config.loss_functions[k])()
            stop_closure[k] = partial(
                self.main_loop,
                data_loader=get_subdict(self.data_loaders["validation"], [k]),
                mode="Validation",
                epoch=0,
                cycler_args={},
                cycler="LongCycler",
            )
        optimizer = getattr(optim, self.config.optimizer)(
            self.model.parameters(), **self.config.optimizer_options
        )
        return optimizer, stop_closure, criterion

    def compute_loss(
        self, mode, task_key, loss, outputs, targets,
    ):
        if task_key in self.criterion:
            loss += self.criterion[task_key](outputs, targets)
            _, predicted = outputs.max(1)
            batch_size = targets.size(0)
            self.tracker.log_objective(
                batch_size, keys=(mode, task_key, "normalization"),
            )
            self.tracker.log_objective(
                100 * predicted.eq(targets).sum().item(),
                keys=(mode, task_key, "accuracy"),
            )
            self.tracker.log_objective(
                loss.item() * batch_size, keys=(mode, task_key, "loss"),
            )
        return loss

    def test_final_model(self, epoch):
        # test the final model with noise on the dev-set
        # test the final model on the test set
        for k in self.task_keys:
            if "rep_matching" not in k and self.config.noise_test:
                for n_type, n_vals in self.config.noise_test.items():
                    for val in n_vals:
                        val_str = stringify(val)
                        mode = "Noise {} {}".format(n_type, val_str)
                        objectives = {
                            mode: {k: {"accuracy": 0, "loss": 0, "normalization": 0,}}
                        }
                        self.tracker.add_objectives(objectives, init_epoch=True)
                        module_options = {"noise_snr": None, "noise_std": None, "rep_matching":False}
                        module_options[n_type] = val
                        self.main_loop(
                            epoch=epoch,
                            data_loader=get_subdict(
                                self.data_loaders["validation"], [k]
                            ),
                            mode=mode,
                            cycler_args={},
                            cycler="LongCycler",
                            module_options=module_options,
                        )

            test_result = self.main_loop(
                epoch=epoch,
                data_loader=get_subdict(self.data_loaders["test"], [k]),
                mode="Test",
                cycler_args={},
                cycler="LongCycler",
                module_options={"noise_snr": None, "noise_std": None},
            )
        if "c_test" in self.data_loaders:
            for k in self.task_keys:
                if "rep_matching" not in k:
                    for c_category in list(self.data_loaders["c_test"][k].keys()):
                        for c_level, data_loader in self.data_loaders["c_test"][k][
                            c_category
                        ].items():

                            objectives = {
                                c_category: {
                                    c_level: {
                                        "accuracy": 0,
                                        "loss": 0,
                                        "normalization": 0,
                                    }
                                }
                            }
                            self.tracker.add_objectives(objectives, init_epoch=True)
                            results, _ = self.main_loop(
                                epoch=epoch,
                                data_loader={c_level: data_loader},
                                mode=c_category,
                                cycler_args={},
                                cycler="LongCycler",
                                module_options={"noise_snr": None, "noise_std": None, "rep_matching":False},
                            )
        if "st_test" in self.data_loaders:
            self.main_loop(
                epoch=epoch,
                data_loader={"img_classification": self.data_loaders["st_test"]},
                mode="Test-ST",
                cycler_args={},
                cycler="LongCycler",
                module_options={"noise_snr": None, "noise_std": None, "rep_matching": False},
            )
        return test_result