from typing import Tuple, List
from typing_extensions import override

from lmfit import minimize, Parameters
from lmfit.minimizer import MinimizerResult
import numpy as np

from ramlab.fit.result import FitResult
from ramlab.fit.modifiers import MeasurementModifier, MeasurementSpectrum, Spectrum
from ramlab.state.polarisation import Polarisation
from ramlab.fit.recipe import FitRecipe
from ramlab.calculate_molecules.Molecule import Molecule
from ramlab.state.transitions import Transitions
from ramlab.simulate_spectrum.base import SimulationMethod
from ramlab.simulate_spectrum.linespreadfunction.base import Lineshape
from ramlab.simulate_spectrum.combine import CombinedMolecule


class MultiMoleculeFitRecipe(FitRecipe):
    polarisation: Polarisation
    lineshape: Lineshape
    simulation_method: SimulationMethod
    meas_modifiers: List[MeasurementModifier]
    fit_modifiers: List[MeasurementModifier]

    molecules: CombinedMolecule
    fitparameters: Parameters

    def __init__(
        self,
        molecules: List[Tuple[Molecule, Transitions]],
        polarisation: Polarisation,
        lineshape: Lineshape,
        simulation_method: SimulationMethod,
        meas_modifiers: List[MeasurementModifier] = None,
        fit_modifiers: List[MeasurementModifier] = None,
    ):
        self.molecules = CombinedMolecule(
            polarisation=polarisation,
            **{
                molecule.molecule_name: (molecule, transitions)
                for molecule, transitions in molecules
            },
        )
        self.polarisation = polarisation
        self.lineshape = lineshape
        self.meas_modifiers = meas_modifiers or []
        self.fit_modifiers = fit_modifiers or []
        self.simulation_method = simulation_method

    @override
    def prepare(self):
        self.fitparameters = Parameters()
        self.fitparameters.add("A", value=1, vary=True)
        for T_name in self._get_temperature_names():
            self.fitparameters.add(T_name, value=3000, min=250, max=10e3, vary=True)

        for molecule in self.molecules.molecule_names():
            self.fitparameters.add(f"X_{molecule}", value=1, min=0, max=1, vary=True)

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

        dnu_stick, lambda_stick, I_stick_sim = self.molecules.stick(
            T_dict["T"],
            **{x: pars[f"X_{x}"] for x in self.molecules.molecule_names()},
        )
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
        # Fit the parameters to the measurement
        out: MinimizerResult = minimize(
            self.fit_residuals,
            self.fitparameters,
            args=(meas.original.c.normalize(axis=0),),
        )

        # Synthesize the fitted spectrum
        fit_normalized = self.make(out.params, meas.original)

        # Apply the measurement modifiers according to the fit parameters
        for mod in self.meas_modifiers:
            meas = mod.apply_and_modify(meas, out.params)
        data_normalized = meas.data.c.normalize(axis=0).sdata

        # Calculate the residuals between the measured and fitted spectrum
        residuals = data_normalized - fit_normalized

        # Return the fit result
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
