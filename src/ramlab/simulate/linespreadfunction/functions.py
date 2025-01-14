from scipy.special import voigt_profile as voigt
from scipy.special import erf
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
import numpy as np


def gaussian(x, sigma=1):
    """Returns a normalized Gaussian with unit integral."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x) / sigma) ** 2)


def lorentzian(x, gamma=1):
    """Returns a normalized Lorentzian with unit integral."""
    return (1 / (np.pi * gamma)) * (gamma**2 / ((x) ** 2 + gamma**2))


def rect(x, width=1):
    """Returns a normalized rectangular function centered at zero with unit integral."""
    return np.where(np.abs(x) <= width / 2, 1 / width, 0)


def slit(
    x,
    fudgefactor=1,
    pixel_spectral_width=1e-9,
    fibersize=100e-6,
    slitwidth=65e-3,
    pixel_size=13.4e-6,
):
    """Returns a normalized slit function."""
    fibersize *= fudgefactor
    slitwidth *= fudgefactor

    y = np.cos(x * np.pi / fibersize)
    y = np.where(np.abs(x - np.minimum(slitwidth, fibersize) / 2) > 0, 0, y)

    y /= y * np.diff(x, append=np.nan, axis=1)

    return slit


def custom(x, g, s):
    combine(x, gaussian(x, g), slit(x, s))


def martijnian(x, g, w):
    m = (
        erf((x + w / 2) / (np.sqrt(2) * g)) / 2
        - erf((x - w / 2) / (np.sqrt(2) * g)) / 2
    )
    m /= w

    return m


def combine(x, *args):
    y = args[0]
    for yi in args[1:]:
        # y = convolve1d(y, yi, mode="same")
        y = np.convolve(y, yi, mode="same")

    # normalize integral to 1
    y /= np.nansum(y * np.diff(x), axis=1)
    return y
