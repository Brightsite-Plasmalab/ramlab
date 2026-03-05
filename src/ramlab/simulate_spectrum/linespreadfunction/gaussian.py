from typing_extensions import override

import numpy as np
from ramlab.simulate_spectrum.linespreadfunction.base import Lineshape


def gaussian(x, sigma=1):
    """Returns a normalized Gaussian with unit integral."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)


class Gaussian(Lineshape):
    w_g: float = 1
    vary_g: bool = True

    @override
    def prepare_fitparameters(self, parameters):
        parameters.add("w_g", value=1, min=0, vary=True)

    def apply(self, parameters, **kwargs):
        self.w_l = self._get_parameter(parameters, kwargs, "w_l")
        if (
            self._get_parameter(parameters, kwargs, "vary_l", allow_none=True)
            is not None
        ):
            self.vary_l = self._get_parameter(parameters, kwargs, "vary_l")

    @override
    def w_typical(self):
        return self.w_g

    @override
    def y(self, x):
        return gaussian(x, self.w_g)
