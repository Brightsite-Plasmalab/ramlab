from typing_extensions import override

from src.ramlab.simulate.linespreadfunction.base import Lineshape

import numpy as np
import scipy.integrate as integrate


def hubbert(x, w=1, q=1):
    yl = lambda x: np.cosh(x / w) ** (-q)
    yi = yl(x)
    I = abs(integrate.quad(yl, -np.inf, np.inf)[0])
    return yi / I


class Hubbert(Lineshape):
    w: float = 1
    q: float = 1
    vary_w: bool = True
    vary_q: bool = True

    @override
    def prepare_fitparameters(self, parameters):
        parameters.add("w", value=self.w, min=0, vary=self.vary_w)
        parameters.add("q", value=self.q, min=0, vary=self.vary_q)

    def apply(self, parameters, **kwargs):
        self.w = self._get_parameter(parameters, kwargs, "w")
        self.q = self._get_parameter(parameters, kwargs, "q")
        if (
            self._get_parameter(parameters, kwargs, "vary_w", allow_none=True)
            is not None
        ):
            self.vary_w = self._get_parameter(parameters, kwargs, "vary_w")

        if (
            self._get_parameter(parameters, kwargs, "vary_q", allow_none=True)
            is not None
        ):
            self.vary_q = self._get_parameter(parameters, kwargs, "vary_q")

    @override
    def w_typical(self):
        return self.w

    @override
    def y(self, x):
        return hubbert(x, self.w, self.q)
