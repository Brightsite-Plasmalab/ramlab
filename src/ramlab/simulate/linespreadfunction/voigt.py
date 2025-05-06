from typing_extensions import override
from scipy.special import voigt_profile as voigt
import numpy as np

from ramlab.simulate.linespreadfunction.base import Lineshape


class Voigt(Lineshape):
    w_g: float = 1
    w_l: float = 0
    vary_g: bool = True
    vary_l: bool = True

    @override
    def prepare_fitparameters(self, parameters):
        parameters.add("w_g", value=self.w_g, min=0, vary=self.vary_g)
        parameters.add("w_l", value=self.w_l, min=0, vary=self.vary_l)

    def apply(self, parameters, **kwargs):
        self.w_g = self._get_parameter(parameters, kwargs, "w_g")
        self.w_l = self._get_parameter(parameters, kwargs, "w_l")
        if (
            self._get_parameter(parameters, kwargs, "vary_g", allow_none=True)
            is not None
        ):
            self.vary_g = self._get_parameter(parameters, kwargs, "vary_g")
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
        return voigt(x, self.w_g, self.w_l)
