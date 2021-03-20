from typing import Dict

from nntransfer.configs.base import BaseConfig
from nntransfer.tables.nnfabrik import Model


class ModelConfig(BaseConfig):
    config_name = "model"
    table = Model()
    fn = None

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.dropout: float = 0.0
        self.get_intermediate_rep: Dict = {}
        super().__init__(**kwargs)


