from typing_extensions import override

import numpy as np
from ramlab.simulate_spectrum.linespreadfunction.base import Lineshape


def lorentzian(x, gamma=1):
    """Returns a normalized Lorentzian with unit integral."""
    return (1 / (np.pi * gamma)) * (gamma**2 / ((x) ** 2 + gamma**2))


class Lorentzian(Lineshape):
    w_l: float = 1
    vary_l: bool = True

    @override
    def prepare_fitparameters(self, parameters):
        parameters.add("w_l", value=1, min=0, vary=True)

    def apply(self, parameters, **kwargs):
        self.w_l = self._get_parameter(parameters, kwargs, "w_l")
        if (
            self._get_parameter(parameters, kwargs, "vary_l", allow_none=True)
            is not None
        ):
            self.vary_l = self._get_parameter(parameters, kwargs, "vary_l")

    @override
    def w_typical(self):
        return self.w_g + self.w_l

    @override
    def y(self, x):
        return lorentzian(x, self.w_l)
