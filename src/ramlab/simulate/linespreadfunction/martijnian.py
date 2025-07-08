from typing_extensions import override
from src.ramlab.simulate.linespreadfunction.base import Lineshape

import numpy as np
from scipy.special import erf


def martijnian(x, g, s):
    m = (
        erf((x + s / 2) / (np.sqrt(2) * g)) / 2
        - erf((x - s / 2) / (np.sqrt(2) * g)) / 2
    )
    m /= s

    return m


class Martijnian(Lineshape):
    w_g: float = 1
    w_s: float = 1
    vary_l: bool = True

    @override
    def prepare_fitparameters(self, parameters):
        parameters.add("w_g", value=self.w_g, min=0, vary=True)
        parameters.add("w_s", value=self.w_s, min=0, vary=True)

    def apply(self, parameters, **kwargs):
        self.w_g = self._get_parameter(parameters, kwargs, "w_g")
        vary_g = self._get_parameter(parameters, kwargs, "vary_g", allow_none=True)
        if vary_g is not None:
            self.vary_g = vary_g

        self.w_s = self._get_parameter(parameters, kwargs, "w_s")
        vary_s = self._get_parameter(parameters, kwargs, "vary_s", allow_none=True)
        if vary_s is not None:
            self.vary_s = vary_s

    @override
    def w_typical(self):
        return self.w_g + self.w_s

    @override
    def y(self, x):
        return martijnian(x, self.w_g, self.w_s)
