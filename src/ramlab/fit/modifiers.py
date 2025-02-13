from typing_extensions import List, override
from lmfit import Parameters
from toddler.data.spectrum import Spectrum
import numpy as np


class MeasurementSpectrum:
    original: Spectrum
    data: Spectrum

    def __init__(self, spectrum):
        self.original = spectrum.c
        self.data = spectrum.c


class MeasurementModifier:
    def add_parameters(self, pars: Parameters):
        raise NotImplementedError()

    def modify(
        self, measurement: MeasurementSpectrum, pars: Parameters
    ) -> MeasurementSpectrum:
        raise NotImplementedError()


class WavelengthAxisCorrection(MeasurementModifier):
    order: int

    def __init__(self, order):
        self.order = order

    @override
    def add_parameters(self, pars: Parameters, initial_values=None, vary=True):
        for i in range(self.order + 1):
            initial = (
                0
                if not (len(initial_values) > i or initial_values[i] == np.nan)
                else initial_values[i]
            )
            pars.add(
                f"lambda_correction_{i:d}", value=initial, min=-3, max=3, vary=False
            )

    @override
    def modify(self, measurement: MeasurementSpectrum, pars: Parameters):
        N_center = np.size(measurement.data.lambda_) / 2
        lambda_relative = np.arange(np.size(measurement.data.lambda_)) - N_center
        lambda_relative /= np.max(lambda_relative)

        corr = np.zeros_like(measurement.data.lambda_)

        for i in range(self.order + 1):
            corr += pars[f"lambda_correction_{i:d}"] * (lambda_relative**i) * 1e-9

        measurement.data.lambda_ = measurement.data.lambda_ + corr

        return measurement


class Fit:
    measurement: MeasurementSpectrum
    modifiers: List[MeasurementModifier]

    def __init__(self, measurement, modifier):
        pass

    def apply_modifiers():
        pass
