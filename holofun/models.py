import numpy as np


class HoloOptimizer:
    def __init__(self) -> None:
        self.score_funcs = None
        self.update_func = None


class GenericModel:
    pass


class Gauss2dModel:
    pass


gauss_sample_data = np.array(
    [[0, 0, 0, 0, 0],
     [0, 1, 2, 1, 0],
     [0, 2, 4, 2, 0],
     [0, 1, 2, 1, 0],
     [0, 0, 0, 0, 0]]
)

def test_gauss2d():
    pass