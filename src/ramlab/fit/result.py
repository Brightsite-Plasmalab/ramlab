from lmfit import Parameters
from toddler.data.spectrum import Spectrum


class FitResult:
    fit_params: Parameters
    meas_original: Spectrum
    meas: Spectrum
    fit: Spectrum
    residuals: Spectrum
    Ts: dict
    T: float
    T_err: float

    def __init__(
        self,
        fit_params: Parameters,
        meas_original: Spectrum,
        meas: Spectrum,
        fit: Spectrum,
        residuals: Spectrum,
        T: float = None,
        T_err: float = None,
        **kwargs,
    ):
        self.fit_params = fit_params
        self.meas_original = meas_original
        self.meas = meas
        self.fit = fit
        self.residuals = residuals
        self.T = T
        self.T_err = T_err
        self.Ts = {**kwargs, "T": self.T, "T_err": self.T_err}

    def to_dict(self):
        return {
            "T": self.T,
            "T_err": self.T_err,
            "Ts": self.Ts,
            "data_orig": {
                "lambda": self.meas_original.lambdanm,
                "data": self.meas_original.sdata,
            },
            "data": {
                "lambda": self.meas.lambdanm,
                "data": self.meas.sdata,
            },
            "fit": {
                "lambda": self.fit.lambdanm,
                "data": self.fit.sdata,
            },
            "residuals": {
                "lambda": self.residuals.lambdanm,
                "data": self.residuals.sdata,
            },
            "fitparameters": self.fit_params,
        }
