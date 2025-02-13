from typing_extensions import override

import numpy as np
from ramlab.simulate.linespreadfunction.base import Lineshape


def gaussian(x, sigma=1):
    """Returns a normalized Gaussian with unit integral."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x) / sigma) ** 2)


class Gaussian(Lineshape):
    @override
    def prepare(self, parameters):
        parameters.add("w_g", value=1, min=0, vary=True)

    @override
    def y(self, x, parameters, **kwargs):
        if "w_g" in parameters.keys():
            w_g = parameters["w_g"]
        elif "w_g" in kwargs.keys():
            w_g = kwargs["w_g"]
        else:
            raise ValueError("Gaussian width `w_g` not found in parameters or kwargs")

        return gaussian(x, w_g)
