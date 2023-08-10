from dataclasses import dataclass
import pandas as pd
import numpy as np

def make_holo_df(hr_targets):
    holos = pd.DataFrame({
        'hr targets': hr_targets,
    })
    
    # add holo size lengths
    vals = holos['hr targets'].apply(len).values
    holos['num targets requested'] = vals
    
    return holos

@dataclass
class Holo:
    """A hologram is a set of targets."""
    targs: np.ndarray
    ntargeted: int
    nmatched: int
    off_target_range: int
    off_targets: np.ndarray
    xyz: np.ndarray
    xyz_matched: np.ndarray
    
    def __post_init__(self):
        pass