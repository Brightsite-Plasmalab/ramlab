from collections.abc import Callable
import functools

import numpy as np
import scipy

from ramlab._type_hints import floatNDArray1D
from ramlab.simulate_spectrum.linespreadfunction.gaussian import gaussian


def _bin_indexes(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    # TODO: Optimize with numba or by using np.searchsorted
    """
    Get the indexes of the bins for the given edges.

    Parameters
    ----------
    edges : np.ndarray
        The edges of the bins. Must be sorted in ascending order.
    values : np.ndarray
        The values to bin. Must be sorted in ascending order.

    Returns
    -------
    np.ndarray
        The indexes of the edges in the values array.
    """
    indexes = np.empty(len(edges), dtype=int)
    index = 0
    for i in range(len(indexes)):
        while index < len(values):
            if values[index] < edges[i]:
                index += 1
            else:
                break
        indexes[i] = index
    return indexes

def _bin_edges(values: np.ndarray) -> np.ndarray:
    """
    Get the edges of the bins for the given values.

    Parameters
    ----------
    values : np.ndarray
        The values to bin. Must be sorted in ascending order.

    Returns
    -------
    np.ndarray
        The edges of the bins for the given values.
    """
    shape = (values.shape[0] + 1,) + values.shape[1:]
    edges = np.zeros(shape, dtype=values.dtype)
    edges[0] = 1.5*values[0] - 0.5*values[1]
    edges[-1] = 1.5*values[-1] - 0.5*values[-2]
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    return edges


class SpectrumSimulator:
    def __init__(self, simulated_wavelengths: floatNDArray1D, spectrum_wavelengths: floatNDArray1D, peak_width: float,
                 *, relative_resolution: int = 10):
        if not np.all(np.diff(simulated_wavelengths) > 0):
            raise ValueError("`simulated_wavelengths` must be sorted in ascending order.")
        if not np.all(np.diff(spectrum_wavelengths) > 0):
            raise ValueError("`spectrum_wavelengths` must be sorted in ascending order.")

        self.simulated_wavelengths = simulated_wavelengths
        self.spectrum_wavelengths = spectrum_wavelengths
        self._spectrum_wavelengths_edges = _bin_edges(spectrum_wavelengths)

        self.sample_wavs = np.empty((len(spectrum_wavelengths)-1)*relative_resolution)
        for i in range(relative_resolution):
            self.sample_wavs[i::relative_resolution] = ((i/relative_resolution)*spectrum_wavelengths[1:]
                                                   + (1 - i/relative_resolution)*spectrum_wavelengths[:-1])

        sample_wavs_bin_edges = _bin_edges(self.sample_wavs)
        self._indexes = _bin_indexes(sample_wavs_bin_edges, simulated_wavelengths)

        dv_median = np.median(np.diff(self.spectrum_wavelengths))

        if peak_width < (10 * dv_median):
            peak_width_new = 10 * dv_median
            point_num = max(int(10 * peak_width_new / peak_width), 100)
            peak_width = peak_width_new
        else:
            point_num = max((int(peak_width / dv_median) * 10), 100)

        self._peak_width = peak_width
        self._peak_point_num = point_num
        self.peak_wavs, self._peak_wavs_dv = np.linspace(-peak_width / 2, peak_width / 2, point_num, retstep=True)

        self._starts = self.sample_wavs - peak_width / 2
        self._ends = self.sample_wavs + peak_width / 2
        self._starts_idx = _bin_indexes(self._starts, self._spectrum_wavelengths_edges)
        self._ends_idx = _bin_indexes(self._ends, self._spectrum_wavelengths_edges)


    def convolute(self, simulated_intensities: floatNDArray1D, peak_func: Callable[[np.ndarray], floatNDArray1D]) -> np.ndarray:
        out = np.zeros_like(self.spectrum_wavelengths)

        peak_cumsum = np.cumsum(peak_func(self.peak_wavs))
        peak_cumsum /= peak_cumsum[-1]
        # print(len(peak_cumsum), self.peak_wavs, self._peak_wavs_dv)
        print(len(simulated_intensities), len(self._indexes))
        for index in range(len(self.sample_wavs)):
            intensities = simulated_intensities[self._indexes[index]:self._indexes[index + 1]]
            if len(intensities) == 0:
                continue
            intensity = np.sum(intensities)

            # Interpolate the cumsum values
            wav_edges = self._spectrum_wavelengths_edges[self._starts_idx[index]:self._ends_idx[index]]
            centered_wav_edges = wav_edges - self.sample_wavs[index] + self._peak_width/2
            peak_wav_indexes = centered_wav_edges / self._peak_wavs_dv
            peak_wav_indexes_int = peak_wav_indexes.astype(int)
            peak_wav_indexes_offset = peak_wav_indexes - peak_wav_indexes_int
            cumsum_value = ((1 - peak_wav_indexes_offset) * peak_cumsum[peak_wav_indexes_int]
                            + peak_wav_indexes_offset * peak_cumsum[peak_wav_indexes_int + 1])

            # print(cumsum_value)

            peak_inten = intensity * np.diff(cumsum_value)
            out[self._starts_idx[index]:self._ends_idx[index] - 1] += peak_inten
        return out

    @classmethod
    def do_convolution(
            cls,
            simulated_wavelengths: floatNDArray1D,
            spectrum_wavelengths: floatNDArray1D,
            simulated_intensities: floatNDArray1D,
            peak_width: float,
            peak_func: Callable[[np.ndarray], floatNDArray1D],
            *,
            relative_resolution: int = 10
        ) -> np.ndarray:

        simulator = cls(simulated_wavelengths, spectrum_wavelengths, peak_width,
                        relative_resolution=relative_resolution)
        return simulator.convolute(simulated_intensities, peak_func)

    @classmethod
    def do_gaussian_convolution(
            cls,
            simulated_wavelengths: floatNDArray1D,
            spectrum_wavelengths: floatNDArray1D,
            simulated_intensities: floatNDArray1D,
            gaussian_sigma: float,
            *,
            relative_resolution: int = 10,
            relative_width: float = 8,
        ) -> np.ndarray:

        func = functools.partial(gaussian, sigma=gaussian_sigma)
        return cls.do_convolution(simulated_wavelengths,
                                  spectrum_wavelengths,
                                  simulated_intensities,
                                  relative_width * gaussian_sigma,
                                  func,
                                  relative_resolution=relative_resolution)


class BandpassSimulator:
    def __init__(self, wavelengths: floatNDArray1D, transmission: floatNDArray1D, simulated_wavelengths: floatNDArray1D,
                 interpolate: bool = False):
        if not np.all(np.diff(wavelengths) > 0):
            raise ValueError("`wavelengths` must be sorted in ascending order.")
        self.wavelengths = wavelengths
        self.transmission = transmission

        if interpolate:
            interpolator = scipy.interpolate.interp1d(wavelengths, transmission, kind='linear', assume_sorted=True)
            self._multiplier = interpolator(simulated_wavelengths)
        else:
            _wavelength_edges = _bin_edges(wavelengths)
            _indexes = np.searchsorted(_wavelength_edges, simulated_wavelengths) - 1
            self._multiplier = self.transmission[_indexes]

    def apply_bandpass(self, simulated_intensities: floatNDArray1D) -> np.ndarray:
        return np.sum(simulated_intensities * self._multiplier)
