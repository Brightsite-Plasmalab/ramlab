from typing_extensions import override

import numpy as np
from ramlab.simulate.linespreadfunction.base import Lineshape


def lorentzian(x, gamma=1):
    """Returns a normalized Lorentzian with unit integral."""
    return (1 / (np.pi * gamma)) * (gamma**2 / ((x) ** 2 + gamma**2))


class Lorentzian(Lineshape):
    @override
    def prepare(self, parameters):
        parameters.add("w_l", value=1, min=0, vary=True)

    @override
    def y(self, x, parameters, **kwargs):
        if "w_l" in parameters.keys():
            w_l = parameters["w_l"]
        elif "w_l" in kwargs.keys():
            w_l = kwargs["w_l"]
        else:
            raise ValueError("Lorentzian width `w_l` not found in parameters or kwargs")

        return lorentzian(x, w_l)
