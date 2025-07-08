from typing_extensions import override
from lmfit import minimize, Parameters
from src.ramlab.molecules.base import Molecule
from src.ramlab.molecules.transitions import Transitions
from src.ramlab.simulate.base import SimulationMethod
from src.ramlab.simulate.linespreadfunction.base import Lineshape
import numpy as np
import lmfit
import numpy as np

from src.ramlab.fit.result import FitResult

#from ttictoc import tic, toc


from src.ramlab.fit.modifiers import *
from src.ramlab.molecules.polarisation import Polarisation


class FitRecipe:
    def prepare(self):
        raise NotImplementedError()

    def fit(self, meas: MeasurementSpectrum):
        raise NotImplementedError()

    def make(self, pars: Parameters, meas: MeasurementSpectrum):
        raise NotImplementedError()

    def fit_residuals(self, pars: Parameters, meas: MeasurementSpectrum):
        I_meas = meas.c.normalize(axis=0).sdata
        I_sim = self.make(pars, meas)
        return (I_meas - I_sim) ** 2

    def apply_parameters(self, pars: Parameters, **kwargs):
        pass


class SingleMoleculeFitRecipe(FitRecipe):
    molecule: Molecule
    transitions: Transitions
    polarisation: Polarisation
    lineshape: Lineshape
    simulation_method: SimulationMethod
    meas_modifiers: List[MeasurementModifier]
    fit_modifiers: List[MeasurementModifier]

    I_const: np.ndarray
    fitparameters: Parameters

    def __init__(
        self,
        molecule: Molecule,
        transitions: Transitions,
        polarisation: Polarisation,
        lineshape: Lineshape,
        simulation_method: SimulationMethod,
        meas_modifiers: List[MeasurementModifier] = [],
        fit_modifiers: List[MeasurementModifier] = [],
    ):
        self.molecule = molecule
        self.transitions = transitions
        self.polarisation = polarisation
        self.lineshape = lineshape
        self.meas_modifiers = meas_modifiers
        self.fit_modifiers = fit_modifiers
        self.simulation_method = simulation_method

    @override
    def prepare(self):
        self.I_const = self.molecule.get_intensity_constant(
            self.transitions, laser_wavelength=532e-9, polarisation=self.polarisation
        )

        self.fitparameters = Parameters()
        self.fitparameters.add("A", value=1, vary=True)
        for T_name in self._get_temperature_names():
            self.fitparameters.add(T_name, value=3000, min=250, max=10e3, vary=True)
        self.lineshape.prepare_fitparameters(self.fitparameters)

        for mod in self.meas_modifiers:
            mod.add_parameters(self.fitparameters)
        for mod in self.fit_modifiers:
            mod.add_parameters(self.fitparameters)

    @classmethod
    def _get_temperature_names(cls):
        return ["T"]

    @override
    def make(self, pars: Parameters, meas: MeasurementSpectrum):
        A = pars["A"]
        T_dict = {name: pars[name] for name in self._get_temperature_names()}

        meas = MeasurementSpectrum(meas)
        for mod in self.meas_modifiers:
            meas = mod.apply_and_modify(meas, pars)

        meas = meas.data

        dnu_meas = (meas.dnu() / 1e2,)
        lambda_meas = meas.lambdanm

        I_var = self.molecule.get_intensity_variable(self.transitions, **T_dict)
        I_stick_sim = (self.I_const) * I_var
        I_stick_sim /= np.nanmax(I_stick_sim)

        dnu_stick = self.transitions.vacuum_wavenumber
        dnu_scat = 1 / 532e-9 - dnu_stick * 1e2
        lambda_stick = 1e9 / dnu_scat

        self.lineshape.apply(pars)

        I_sim_broad = self.simulation_method.simulate(
            lambda_meas, lambda_stick, I_stick_sim, self.lineshape
        )
        I_sim_broad *= A / np.nanmax(I_sim_broad)

        sim = MeasurementSpectrum(Spectrum(data=I_sim_broad, lambda_=lambda_meas))
        for mod in self.fit_modifiers:
            sim = mod.apply_and_modify(sim, pars)
        I_sim_broad = sim.data.data

        return I_sim_broad

    @override
    def apply_parameters(self, pars: Parameters, **kwargs):
        super().apply_parameters(pars)

        self.lineshape.apply(pars, **kwargs)

        for mod in [*self.meas_modifiers, *self.fit_modifiers]:
            mod.apply(pars, **kwargs)

    @override
    def fit(self, meas: MeasurementSpectrum):
        out: lmfit.minimizer.MinimizerResult = minimize(
            self.fit_residuals,
            self.fitparameters,
            args=(meas.original.c.normalize(axis=0),),
        )

        for mod in self.meas_modifiers:
            meas = mod.apply_and_modify(meas, out.params)

        fit_normalized = self.make(out.params, meas.original)
        data_normalized = meas.data.c.normalize(axis=0).sdata
        residuals = data_normalized - fit_normalized

        return FitResult(
            fit_params=out.params,
            meas_original=meas.original,
            meas=meas.data.c.normalize(axis=0),
            fit=Spectrum(data=fit_normalized, lambda_=meas.data.lambda_),
            residuals=Spectrum(data=residuals, lambda_=meas.data.lambda_),
            **{name: out.params[name].value for name in self._get_temperature_names()},
            **{
                name + "_err": out.params[name].stderr
                for name in self._get_temperature_names()
            },
        )


class SingleMoleculeMultiTemperatureFitRecipe(SingleMoleculeFitRecipe):
    @override
    @classmethod
    def _get_temperature_names(cls):
        return ["T_rot", "T_vib"]
