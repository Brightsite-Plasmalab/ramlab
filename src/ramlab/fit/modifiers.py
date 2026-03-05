from abc import ABC, abstractmethod
from typing_extensions import List, override, Union

from lmfit import Parameters
import numpy as np

from ramlab.fit.spectrum import Spectrum


class MeasurementSpectrum:
    original: Spectrum
    data: Spectrum

    def __init__(self, spectrum: Spectrum):
        self.original = spectrum.c
        self.data = spectrum.c


class MeasurementModifier(ABC):
    @abstractmethod
    def add_parameters(self, pars: Parameters):
        pass

    @abstractmethod
    def apply(self, pars: Parameters):
        pass

    @abstractmethod
    def get_parameter_names(self):
        pass

    def get_parameter_dict(self, pars: Parameters):
        return {name: pars[name] for name in self.get_parameter_names()}

    @abstractmethod
    def modify(self, measurement: MeasurementSpectrum) -> MeasurementSpectrum:
        pass

    def apply_and_modify(self, measurement: MeasurementSpectrum, pars: Parameters):
        self.apply(pars)
        return self.modify(measurement)

    @staticmethod
    def _get_parameter(params, kwargs, key, allow_none=False):
        if params is not None and type(params) is not Parameters:
            raise TypeError(
                "Parameters should be of type Parameters, not {}".format(type(params))
            )
        if key in kwargs.keys():
            return kwargs[key]
        elif params is not None and key in params.keys():
            return params[key].value
        elif not allow_none:
            raise ValueError(f"Parameter `{key}` not found in parameters or kwargs")
        else:
            return None


class WavelengthAxisCorrection(MeasurementModifier):
    order: int
    coefficients: Union[List[float], None]
    vary: Union[List[bool], bool, None]

    VARY_NAME = "vary_wl"

    def __init__(self, order, initial_values=None, vary_wl=None):
        self.order = order

        if initial_values is None:
            initial_values = []

        coefficient_names = self.get_parameter_names()
        self.apply(
            None,
            **{
                coefficient_names[i]: (
                    initial_values[i] if i < len(initial_values) else 0
                )
                for i in range(order + 1)
            },
            vary=vary_wl,
        )

    @override
    def add_parameters(self, pars: Parameters):
        for i in range(self.order + 1):
            pars.add(
                self.get_parameter_names()[i],
                value=self.coefficients[i],
                min=-3,
                max=3,
                vary=self.vary[i],
            )

    @override
    def get_parameter_names(self):
        return [f"lambda_correction_{i:d}" for i in range(self.order + 1)]

    @override
    def apply(self, pars: Parameters, **kwargs):
        self.coefficients = []

        for i in range(self.order + 1):
            coefficient_i = self._get_parameter(
                pars, kwargs, self.get_parameter_names()[i], allow_none=True
            )
            self.coefficients.append(coefficient_i)

        if kwargs.get(self.VARY_NAME) is None:
            vary = True
        else:
            vary = kwargs.get(self.VARY_NAME)
        if type(vary) is bool:
            vary = [vary] * (self.order + 1)
        if len(vary) != self.order + 1:
            # Pad with True's
            vary = vary + [True] * (self.order + 1 - len(vary))
        self.vary = vary

        return self

    @override
    def modify(self, measurement: MeasurementSpectrum):
        N_center = np.size(measurement.data.lambda_) / 2
        lambda_relative = np.arange(np.size(measurement.data.lambda_)) - N_center
        lambda_relative /= np.max(lambda_relative)

        corr = np.zeros_like(measurement.data.lambda_)

        for i in range(self.order + 1):
            corr += self.coefficients[i] * (lambda_relative**i) * 1e-9

        measurement.data.lambda_ = measurement.data.lambda_ + corr

        return measurement


class BackgroundCorrection(MeasurementModifier):
    order: int
    coefficients: Union[List[float], None]
    vary: Union[List[bool], bool, None]

    VARY_NAME = "vary_bg"

    def __init__(self, order, initial_values=None, vary_bg=None):
        self.order = order

        if initial_values is None:
            initial_values = []

        coefficient_names = self.get_parameter_names()
        self.apply(
            None,
            **{
                coefficient_names[i]: (
                    initial_values[i] if i < len(initial_values) else 0
                )
                for i in range(order + 1)
            },
            vary_bg=vary_bg,
        )

    @override
    def add_parameters(self, pars: Parameters):
        for i in range(self.order + 1):
            pars.add(
                self.get_parameter_names()[i],
                value=self.coefficients[i],
                vary=self.vary[i],
            )

    @override
    def get_parameter_names(self):
        return [f"bg_correction_{i:d}" for i in range(self.order + 1)]

    @override
    def apply(self, pars: Parameters, **kwargs):
        self.coefficients = []

        for i in range(self.order + 1):
            coefficient_i = self._get_parameter(
                pars, kwargs, self.get_parameter_names()[i], allow_none=True
            )
            self.coefficients.append(coefficient_i)

        if kwargs.get(self.VARY_NAME) is None:
            vary = True
        else:
            vary = kwargs.get(self.VARY_NAME)
        if type(vary) is bool:
            vary = [vary] * (self.order + 1)
        if len(vary) != self.order + 1:
            # Pad with True's
            vary = vary + [True] * (self.order + 1 - len(vary))
        self.vary = vary

        return self

    @override
    def modify(self, measurement: MeasurementSpectrum):
        N_center = np.size(measurement.data.lambda_) / 2
        lambda_relative = np.arange(np.size(measurement.data.lambda_)) - N_center
        lambda_relative /= np.max(lambda_relative)

        corr = np.zeros_like(measurement.data.lambda_)

        for i in range(self.order + 1):
            corr += (
                self.coefficients[i]
                * (lambda_relative**i)
                * np.nanmax(measurement.data.data)
            )

        measurement.data.data = measurement.data.data + corr

        return measurement


class Fit:
    measurement: MeasurementSpectrum
    modifiers: List[MeasurementModifier]

    def __init__(self, measurement, modifier):
        pass

    def apply_modifiers():
        pass
