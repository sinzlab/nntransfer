import collections
import copy
import time
from collections import defaultdict
from typing import Dict

import numpy as np


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Tracker:
    def log_objective(self, obj):
        raise NotImplementedError()

    def finalize(self, obj):
        pass


class TimeObjectiveTracker(Tracker):
    def __init__(self):
        self.tracker = np.array([[time.time(), 0.0]])

    def log_objective(self, obj):
        new_track_point = np.array([[time.time(), obj]])
        self.tracker = np.concatenate((self.tracker, new_track_point), axis=0)

    def finalize(self):
        self.tracker[:, 0] -= self.tracker[0, 0]


class MultipleObjectiveTracker(Tracker):
    def __init__(self, **objectives):
        self.objectives = objectives
        self.log = defaultdict(list)
        self.time = []

    def log_objective(self, obj):
        t = time.time()
        self.time.append(t)
        for name, objective in self.objectives.items():
            self.log[name].append(objective())

    def finalize(self):
        self.time = np.array(self.time)
        self.time -= self.time[0]
        for k, l in self.log.items():
            self.log[k] = np.array(l)


class AdvancedMultipleObjectiveTracker(Tracker):
    def _initialize_log(self, objectives):
        log = {}
        for key, objective in objectives.items():
            if isinstance(objective, dict):
                log[key] = self._initialize_log(objective)
            elif not callable(objective):
                log[key] = []
        return log

    def __init__(self, main_objective=(), save_each_epoch=True, **objectives):
        """
        In principle, `objectives` is expected to be a dictionary of dictionaries
        The hierarchy can in principle be arbitrary deep.
        The only restriction is that the lowest level has to be a dictionary with values being
        either a numerical value which will be interpreted as the intial value for this objective
        (to be accumulated manually) or a callable (e.g. a function) that returns the objective value.

        :param objectives: e.g. {"dataset": {"objective1": o_fct1, "objective2": 0, "normalization": 0},...}
                           or {"dataset": {"task_key": {"objective1": o_fct1, "objective2": 0},...},...}

        """
        self.objectives = objectives
        self.log = self._initialize_log(objectives)
        self.time = []
        self.main_objective = main_objective
        self.epoch = -1
        self.best_epoch = -1
        if not save_each_epoch:
            self.save_each_epoch = True
            self.start_epoch()  # add first element to arrays
        self.save_each_epoch = save_each_epoch

    def add_objectives(self, objectives, init_epoch=False):
        deep_update(self.objectives, objectives)
        new_log = self._initialize_log(objectives)
        if init_epoch:
            _save_each_epoch = self.save_each_epoch
            self.save_each_epoch = True
            self._initialize_epoch(new_log, objectives)
            self.save_each_epoch = _save_each_epoch
        deep_update(self.log, new_log)

    def _initialize_epoch(self, log, objectives):
        for key, objective in objectives.items():
            if isinstance(objective, dict):
                self._initialize_epoch(log[key], objective)
            elif not callable(objective):
                if self.save_each_epoch:
                    while len(log[key]) <= self.epoch:
                        log[key].append(objective)
                else:
                    log[key][0] = objective

    def start_epoch(self):
        t = time.time()
        if self.save_each_epoch:
            self.time.append(t)
            self.epoch += 1
        self._initialize_epoch(self.log, self.objectives)

    def _log_objective_value(self, obj, log, keys=()):
        if len(keys) > 1:
            self._log_objective_value(obj, log[keys[0]], keys[1:])
        else:
            log[keys[0]][self.epoch] += obj

    def _log_objective_callables(self, log, objectives):
        for key, objective in objectives.items():
            if isinstance(objective, dict):
                self._log_objective_callables(log[key], objective)
            elif callable(objective):
                log[key].append(objective())

    def log_objective(self, obj, keys=()):
        if keys:
            self._log_objective_value(obj, self.log, keys)
        else:
            self._log_objective_callables(self.log, self.objectives)

    def _normalize_log(self, log):
        if isinstance(log, dict):
            n_log = {}
            norm = None
            for key, l in log.items():
                res = self._normalize_log(l)  # to turn into arrays
                if key == "normalization":
                    assert isinstance(res, np.ndarray)
                    norm = res
                else:
                    n_log[key] = res
            if norm is not None:
                nonzero_start = (norm != 0).argmax(axis=0)
                norm = norm[nonzero_start:]
                for key, l in n_log.items():
                    l = l[nonzero_start:]
                    if isinstance(l, np.ndarray):
                        n_log[key] = l / np.where(norm > 0, norm, np.ones_like(norm))
            return n_log
        else:
            return np.array(log)

    def _gather_log(self, log, keys, index=-1):
        if len(keys) > 1:
            return self._gather_log(log[keys[0]], keys[1:], index)
        elif keys:
            return self._gather_log(log[keys[0]], (), index)
        elif isinstance(log, dict):
            gathered = {}
            for key, l in log.items():
                logs = self._gather_log(l, (), index)
                for k, v in logs.items():
                    gathered[key + " " + k] = v
            return gathered
        else:
            return {"": "{:03.4f}".format(log[index])}

    def display_log(self, keys=(), tqdm_iterator=None):
        # normalize (if possible) and turn into np.arrays:
        n_log = self._normalize_log(self.log)
        # flatten a subset of the dictionary:
        current_log = self._gather_log(n_log, keys, index=-1)
        if tqdm_iterator:
            tqdm_iterator.set_postfix(**current_log)
        else:
            print(current_log)

    def _check_isfinite(self, log):
        if isinstance(log, dict):
            for k, l in log.items():
                if not self._check_isfinite(l):
                    return False
        else:
            return np.isfinite(log).any()
        return True

    def check_isfinite(self):
        return self._check_isfinite(self._normalize_log(self.log))

    def _get_objective(self, log, keys):
        if len(keys) > 1:
            return self._get_objective(log[keys[0]], keys[1:])
        else:
            return log[keys[0]]

    def get_objective(self, *keys):
        return self._get_objective(self._normalize_log(self.log), keys)

    def get_current_objective(self, *keys):
        return self._get_objective(self._normalize_log(self.log), keys)[-1]

    def get_current_main_objective(self, *keys):
        return self.get_current_objective(*keys, *self.main_objective)

    def finalize(self):
        self.time = np.array(self.time)
        self.time -= self.time[0]
        self.log = self._normalize_log(self.log)

    def state_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def load_state_dict(self, tracker_dict: Dict):
        self.main_objective = tracker_dict["main_objective"]
        self.objectives = tracker_dict["objectives"]
        self.log = tracker_dict["log"]
        self.time = tracker_dict["time"]
        self.epoch = tracker_dict["epoch"]
        self.best_epoch = tracker_dict["best_epoch"]

    @classmethod
    def from_dict(cls, tracker_dict: Dict) -> "AdvancedMultipleObjectiveTracker":
        tracker = cls(
            main_objective=tracker_dict["main_objective"], **tracker_dict["objectives"]
        )
        tracker.log = tracker_dict["log"]
        tracker.time = tracker_dict["time"]
        tracker.epoch = tracker_dict["epoch"]
        tracker.best_epoch = tracker_dict["best_epoch"]
        return tracker
