# Subsample onto a evenly spaced grid
# Project each stick onto the nearest two grid points
# fft-convolute the sticks with the line spread function
# trapezoid integration of the subsampled evenly spaced grid to get the pixel bins
#       (this last step is independent of the stick intensity, so can be calculated upfront)


import numpy as np
import scipy
from ramlab.simulate.base import SimulationMethod
from ramlab.simulate.linespreadfunction.base import Lineshape
from ramlab.simulate.linespreadfunction import voigt, Voigt

from ttictoc import tic, toc

# TODO: Make the subsampler a wrapper around other SimulationMethods


class Subsampled(SimulationMethod):
    x_sim: np.ndarray = None
    x_lsf: np.ndarray = None
    I_lsf: np.ndarray = None
    id_closest: np.ndarray = None
    coefficients: np.ndarray = None
    dx_subsample: float = None

    def __init__(self, x, x_stick, N_subsample=10):
        """Call this method to prepare the simulation, in case these do not include fit parameters. This significantly speeds up the simulation."""

        # Make a subsampled evenly spaced grid
        dx_min = np.min(np.diff(x))
        dx_max = np.max(np.diff(x))
        dx = np.diff(x, append=np.nan)
        dx[-1] = dx[-2]
        self.dx_subsample = dx_min / N_subsample

        x_min = min(np.min(x_stick), np.min(x))
        x_max = max(np.max(x_stick), np.max(x))
        self.x_sim = np.arange(
            x_min - dx_max / 2, x_max + dx_max / 2, self.dx_subsample
        )

        # Project each stick onto the nearest two grid points using np.argmin and no for loop
        self.id_closest = np.argmin(np.abs(self.x_sim[:, np.newaxis] - x_stick), axis=0)
        # perc_closest = (x_stick - x_sim[id_closest]) / dx[id_closest]
        # I_stick_subsampled[id_closest] += I_stick * (1 - perc_closest)
        # I_stick_subsampled[id_closest + 1] += I_stick * perc_closest

        # Trapezoid integration of the subsampled evenly spaced grid to get the pixel bins
        # Construct a matrix of coefficients for the trapezoid integration
        # points that are less than 1/2 the distance from the bin center are 1
        # points that are more than 1/2 the distance from the bin center are 0
        # points that are exactly at the bin center are 1/2

        # Construct the matrix of coefficients
        # SLOW, precalculate this
        self.coefficients = np.where(
            np.abs(x[:, np.newaxis] - self.x_sim) < dx[:, np.newaxis], 1 / 2, 0
        )
        self.coefficients = np.where(
            np.abs(x[:, np.newaxis] - self.x_sim) < dx[:, np.newaxis] / 2,
            1,
            self.coefficients,
        )
        self.coefficients = self.coefficients * self.dx_subsample

    def simulate(self, x, x_stick, I_stick, lineshape: Lineshape):
        # tic()
        I_stick_subsampled = np.zeros_like(self.x_sim)
        I_stick_subsampled[self.id_closest] += I_stick

        # Calculate LSF
        w_sim = 20 * lineshape.w_typical()
        self.x_lsf = np.arange(
            -w_sim,
            w_sim,
            self.dx_subsample,
        )
        # x_lsf = x_sim - np.mean(x_sim)
        self.I_lsf = lineshape.y(self.x_lsf)

        # Convolute the sticks with the line spread function
        # I_sim_subsampled = np.convolve(I_stick_subsampled, I_lsf, mode="same")
        I_sim_subsampled = scipy.signal.fftconvolve(
            I_stick_subsampled, self.I_lsf, mode="same"
        )

        # I_sim = np.sum(I_sim_subsampled * self.coefficients, axis=1)  # SLOW
        I_sim = np.matmul(self.coefficients, I_sim_subsampled)
        # print(f"Execution time: {toc()*1e3:.0f}ms")

        return I_sim


def subsampled(x, x_stick, I_stick, sigma=1, gamma=0, N_subsample=20):
    sim_method = Subsampled(x, x_stick, N_subsample)
    return sim_method.simulate(x, x_stick, I_stick, Voigt(w_g=sigma, w_l=gamma))
