from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class EpochMetaData:
    name: str
    epoch_type: str
    epoch_s2p_idx: int
    epoch_daq_idx: int = None

@dataclass
class Epoch:
    """A dataclass representing an epoch in a 2p imaging experiment."""
    metadata: EpochMetaData
    traces: np.ndarray
    psths: np.ndarray
    psths_are_time_aligned: bool = False
    trial_labels: pd.DataFrame = None
    cell_labels: pd.DataFrame = None

    def overall_mresp(self):
        m = self.psths.mean(axis=1)
        return m.mean(axis=0)

@dataclass
class OriEpoch(Epoch):
    """A dataclass representing an orientation tuning epoch in a 2p imaging experiment."""
    ori_file: str
    ori_data: dict