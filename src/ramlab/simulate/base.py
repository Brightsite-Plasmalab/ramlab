import numpy as np
from ramlab.simulate.linespreadfunction.base import Lineshape


class SimulationMethod:
    def __init__(self):
        pass

    def simulate(
        self,
        x: np.ndarray,
        x_stick: np.ndarray,
        I_stick: np.ndarray,
        lineshape: Lineshape,
    ):
        pass
