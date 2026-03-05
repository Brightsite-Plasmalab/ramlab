import math

import numba
import numpy as np
from scipy.special import voigt_profile

from ramlab.simulate_spectrum.common import project_to_axis


def simulate_convolution(x, x_stick, I_stick, sigma=1, gamma=0):
    """This method generates a synthetic spectrum by convoluting a Voigt profile over a stick spectrum.

    Args:
        x (np.array): The desired x-axis for the spectrum. Must be equally spaced.
        x_stick (np.array): The x-axis of the stick spectrum.
        I_stick (np.array): The intensity of the stick spectrum.
        sigma (float): The standard deviation of the Gaussian profile.
        gamma (float, optional): The FWHM of the Lorentzian profile. Defaults to 0.
    """
    assert len(x_stick) == len(
        I_stick
    ), "x_stick and I_stick must have the same length."
    diffs_x = np.diff(x)
    assert np.all(diffs_x > 0), "x must be sorted in ascending order."
    assert np.all(~np.isnan(x)), "x must not contain NaNs."
    assert (np.max(diffs_x) - np.min(diffs_x)) < np.min(
        diffs_x
    ) / 1e10, "x must be equally spaced."

    dx = np.median(np.diff(x))

    pix_per_std = (sigma + gamma) / dx
    if pix_per_std < 5:
        print(
            f"sigma+gamma = {sigma+gamma}, dx = {dx}, so the FWHM is about {pix_per_std:.1f} steps. That's quite coarse (:\n Increase the number of steps or decrease the FWHM. If you can't, consider changing the simulation method to something more robust at low resolution."
        )

    N_pix_voigt = (sigma + gamma) / dx * 10
    N_pix_voigt = int(min(np.size(x) / 2 - 4, N_pix_voigt))

    x_voigt = np.arange(-N_pix_voigt, N_pix_voigt + 2) * dx
    x_voigt -= np.median(x_voigt)

    I_voigt = voigt_profile(x_voigt, sigma, gamma)
    # I_voigt /= I_voigt.sum()

    I_stick_projected = project_to_axis(x, x_stick, I_stick)
    I_x = np.convolve(I_stick_projected, I_voigt, mode="same")

    return I_x


@numba.njit
def _make_stick(out, wavelengths_out, peak_wavelengths, peak_intensities):
    """
    Generate a stick spectrum by placing the intensities at the nearest
    wavelength bins and distributing the intensity linearly between the two
    nearest bins.

    Parameters
    ----------
    out: np.ndarray
        The output array to add the stick spectrum to.
    wavelengths_out: np.ndarray
        The wavelengths corresponding to the output array.
    peak_wavelengths: np.ndarray
        The peak wavelengths.
    peak_intensities: np.ndarray
        The intensities at the peak wavelengths.

    Returns
    -------
    np.ndarray
        The output array with the stick spectrum added.
    """
    dx = wavelengths_out[1] - wavelengths_out[0]
    max_index = len(out) - 1
    for i, wav in enumerate(peak_wavelengths):
        wav_loc = (wav - wavelengths_out[0]) / dx
        wav_index = math.floor(wav_loc)
        if 0 <= wav_index <= max_index:
            inten = peak_intensities[i]
            rel = wav_loc - wav_index
            out[wav_index] += (1 - rel) * inten
            if wav_index != max_index:
                out[wav_index + 1] += rel * inten
    return out

def convolute_spectrum(spectrum_wavelengths, peak_wavelengths, peak_intensities, peak_shape):
    """
    Generate a spectrum by convolving a stick spectrum with a peak shape.

    Parameters
    ----------
    spectrum_wavelengths: np.ndarray
        The wavelengths at which to generate the spectrum.
    peak_wavelengths: np.ndarray
        The wavelengths of the peaks.
    peak_intensities: np.ndarray
        The intensities of the peaks.
    peak_shape: np.ndarray
        The shape of the peaks to convolve the stick spectrum with. Must have the same wavelength steps as spectrum_wavelengths.

    Returns
    -------
    np.ndarray
        The generated spectrum.
    """
    out = np.zeros_like(spectrum_wavelengths)
    stick = _make_stick(out, spectrum_wavelengths, peak_wavelengths, peak_intensities)
    return np.convolve(stick, peak_shape, mode="same")


def generate_spectrum(spectrum_wavelengths, simulated_wavelengths, simulated_intensities, peak_width, peak_func):
    if not np.all(np.diff(simulated_wavelengths) > 0):
        raise ValueError("`simulated_wavelengths` must be sorted in ascending order.")
    if not np.all(np.diff(spectrum_wavelengths) > 0):
        raise ValueError("`spectrum_wavelengths` must be sorted in ascending order.")

    wavelength_bin_edges = np.zeros(len(spectrum_wavelengths) + 1)
    wavelength_bin_edges[0] = spectrum_wavelengths[0] - 0.5 * (spectrum_wavelengths[1] - spectrum_wavelengths[0])
    wavelength_bin_edges[-1] = spectrum_wavelengths[-1] + 0.5 * (spectrum_wavelengths[-1] - spectrum_wavelengths[-2])
    wavelength_bin_edges[1:-1] = 0.5 * (spectrum_wavelengths[1:] + spectrum_wavelengths[:-1])

    dv_median = np.median(np.diff(spectrum_wavelengths))

    if peak_width < 10*dv_median:
        peak_width_new = 10*dv_median
        point_num = max(int(10*peak_width_new / peak_width), 100)
        peak_width = peak_width_new
    else:
        point_num = max((int(peak_width / dv_median)*10), 100)
    peak_wavs = np.linspace(-peak_width/2, peak_width/2, point_num)
    peak_cumsum = np.cumsum(peak_func(peak_wavs))
    peak_dv = peak_wavs[1] - peak_wavs[0]
    peak_cumsum /= peak_cumsum[-1]

    # @numba.njit
    def func2(spec_wavs, sim_wavs, sim_intens, pe_cumsum, wav_bin_edges, pe_dv, pe_width):
        out = np.zeros_like(spec_wavs)
        start_index = 0
        end_index = 0
        for index, (sim_wav, sim_inten) in enumerate(zip(sim_wavs, sim_intens)):
            # The start and end wavelength of the full peak
            start = sim_wav - pe_width
            end = sim_wav + pe_width

            # Test if the peak is out of bounds
            if (start > wav_bin_edges[-1]) or (end < wav_bin_edges[0]):
                continue

            # Update the start and end indexes for this simulated_wavelength
            while start_index < len(wav_bin_edges) and start > wav_bin_edges[start_index]:
                start_index += 1
            while end_index < len(wav_bin_edges) and end > wav_bin_edges[end_index]:
                end_index += 1

            # Interpolate the cumsum values
            wav_edges = wav_bin_edges[start_index:end_index]
            centered_wav_edges = wav_edges - sim_wav + pe_width
            peak_wav_indexes = centered_wav_edges / pe_dv
            peak_wav_indexes_int = peak_wav_indexes.astype(int)
            peak_wav_indexes_offset = peak_wav_indexes - peak_wav_indexes_int
            cumsum_value = ((1-peak_wav_indexes_offset)*pe_cumsum[peak_wav_indexes_int]
                            + peak_wav_indexes_offset*pe_cumsum[peak_wav_indexes_int+1])

            peak_inten = sim_inten*np.diff(cumsum_value)
            out[start_index:end_index-1] += peak_inten
        return out
    return func2(spectrum_wavelengths, simulated_wavelengths, simulated_intensities, peak_cumsum,
                 wavelength_bin_edges, peak_dv, peak_width/2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_stick = np.linspace(0, 20, 13)
    I_stick = np.random.rand(x_stick.size)
    x = np.arange(0, 20)

    I_stick_projected = project_to_axis(x, x_stick, I_stick)

    plt.figure()
    plt.vlines(x_stick, 0, I_stick, color="C0")
    plt.vlines(x, 0, I_stick_projected, color="C2")
    plt.plot(x, np.zeros_like(x), ".", color="C1")
    plt.show()
