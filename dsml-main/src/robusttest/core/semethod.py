import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from robusttest.core.grids import grid_ts


@dataclass
class SEResult:
    """Class for storing results from the train function."""

    # TODO  rework that once the Method implementation is reorganised

    report_df: pd.DataFrame  # evaluation report values
    pred: pd.DataFrame  # V predictions for each bus
    training_params: Any  # parameters used for training


class SEMethod(ABC):
    def __init__(self, method: str, grid : grid_ts , verbose: int = 1, ):
        self.verbose = verbose
        self.method = method
        self.grid = grid

    def evaluate(self):
        pass

    @staticmethod
    def guess_topology_NN(self):
        pass

    @staticmethod
    def prepare_measurments(self) -> pd.DataFrame:
        temp_df = self.net.measurments 

        measurements_df = pd.DataFrame()
        
        return measurements_df

    @abstractmethod
    def train(
        self,
        train,
    ):
        pass

    @abstractmethod
    def predict(self, test,):
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, model_params_path: str, verbose: int = 1):
        pass

    @abstractmethod
    def save_model(self):
        pass
