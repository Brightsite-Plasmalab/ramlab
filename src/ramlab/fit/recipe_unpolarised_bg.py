from typing_extensions import override
from lmfit import minimize, Parameters
import numpy as np
import numpy as np

from ramlab.fit.recipe import SingleMoleculeFitRecipe
from ramlab.fit.modifiers import *


class UnpolarisedBackgroundFit(SingleMoleculeFitRecipe):
    s_hor: Spectrum

    @override
    def prepare(self, s_hor: Spectrum = None):
        super().prepare()
        if s_hor is not None:
            self.s_hor = s_hor
        elif self.s_hor is None:
            raise ValueError("No horizontal spectrum given")

        self.fitparameters["A"].set(value=1, min=0.1, max=1.1, vary=True)
        self.fitparameters.add("A_bg", value=1e-5, min=0, max=1, vary=True)

    @override
    def make(self, pars: Parameters, meas: MeasurementSpectrum):
        I_sim_broad = super().make(pars, meas)
        I_sim_broad = I_sim_broad / np.nanmax(I_sim_broad)

        I_bg = self.s_hor.c.sdata
        I_bg = I_bg / np.nanmax(I_bg)

        I = I_sim_broad * pars["A"].value + pars["A_bg"].value * I_bg
        # I = I_sim_broad + pars["A_bg"].value * I_bg
        # I = I / np.nanmax(I) * pars["A"].value

        return I
