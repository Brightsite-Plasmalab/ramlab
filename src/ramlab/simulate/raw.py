import numpy as np

from ramlab.simulate.base import SimulationMethod
from ramlab.simulate.linespreadfunction.base import Lineshape
from ramlab.simulate.linespreadfunction.voigt import voigt
from ramlab.simulate.convolution import simulate_convolution2

from ttictoc import tic, toc


class RawSimulationMethod(SimulationMethod):
    def simulate(self, x, x_stick, I_stick, lineshape: Lineshape):
        return simulate_raw_general(x, x_stick, I_stick, lineshape)


class RawBinnedSimulationMethod(SimulationMethod):
    N_bin = 50

    def __init__(self, N_bin=50):
        self.N_bin = N_bin

    def simulate(self, x, x_full, x_stick, I_stick, sigma, gamma):
        # tic()
        x_new, y = simulate_raw_stijn(x, x_full, x_stick, I_stick, sigma, gamma, self.N_bin)
        # print(f"Execution time: {toc()*1e3:.0f}ms")

        return x_new, y


def simulate_raw_general(x, x_stick, I_stick, lineshape: Lineshape):
    x_stick = np.array(x_stick)[:, np.newaxis]
    I_stick = np.array(I_stick)[:, np.newaxis]

    # Calculate the intensity of every line at every point
    Ii2 = I_stick * lineshape.y(x - x_stick)

    # Sum the intensities of all lines at every point
    I_x2 = np.sum(Ii2, axis=0)

    return I_x2


def simulate_raw_binned_general(x, x_stick, I_stick, lineshape: Lineshape, N_bin=50):
    x_stick = np.array(x_stick)[:, np.newaxis]
    I_stick = np.array(I_stick)[:, np.newaxis]

    dx = np.median(np.diff(x))
    x2 = x + dx / N_bin * (np.arange(N_bin)[:, np.newaxis] - (N_bin - 1) / 2)
    x2 = x2.T.ravel()

    # Ii2 = I_stick * gaussian(x2 - x_stick, sigma)
    # Ii2 = I_stick * custom(x2 - x_stick, sigma, gamma)
    Ii2 = I_stick * lineshape.y(x2 - x_stick)
    I_x2 = np.sum(Ii2, axis=0)

    x3 = x2.reshape(-1, N_bin).mean(axis=1)
    I_x3 = I_x2.reshape(-1, N_bin).mean(axis=1)

    # assert np.all(x == x3)
    return I_x3


def simulate_raw_stijn(x,x_full, x_stick, I_stick, sigma, gamma, N_bin=40):
    x_stick = np.array(x_stick)[:, np.newaxis]
    I_stick = np.array(I_stick)[:, np.newaxis]


    x2 = np.linspace(min(x_full), max(x_full), len(x_full) * N_bin)

    Ii2 = simulate_convolution2(x2, x_stick, I_stick, sigma, gamma)

    x3 = x2[(x2>=np.min(x)) & (x2<=np.max(x))]
    Ii3 = Ii2[(x2>=np.min(x)) & (x2<=np.max(x))]

    return x3, Ii3

def simulate_raw(x, x_stick, I_stick, sigma=1, gamma=0, N_bin=50):
    x_stick = np.array(x_stick)[:, np.newaxis]
    I_stick = np.array(I_stick)[:, np.newaxis]

    dx = np.median(np.diff(x))
    x2 = x + dx / N_bin * (np.arange(N_bin)[:, np.newaxis] - (N_bin - 1) / 2)
    x2 = x2.T.ravel()

    # Ii2 = I_stick * gaussian(x2 - x_stick, sigma)
    # Ii2 = I_stick * custom(x2 - x_stick, sigma, gamma)
    Ii2 = I_stick * voigt(x2 - x_stick, sigma, gamma)
    I_x2 = np.sum(Ii2, axis=0)

    x3 = x2.reshape(-1, N_bin).mean(axis=1)
    I_x3 = I_x2.reshape(-1, N_bin).mean(axis=1)

    # assert np.all(x == x3)
    return I_x3


def simulate_raw_binned(x, x_stick, I_stick, sigma=1, gamma=0, N_bin=50):
    x_stick = np.array(x_stick)[:, np.newaxis]
    I_stick = np.array(I_stick)[:, np.newaxis]

    dx = np.median(np.diff(x))
    x2 = x + dx / N_bin * (np.arange(N_bin)[:, np.newaxis] - (N_bin - 1) / 2)
    x2 = x2.T.ravel()

    # Ii2 = I_stick * gaussian(x2 - x_stick, sigma)
    # Ii2 = I_stick * custom(x2 - x_stick, sigma, gamma)
    Ii2 = I_stick * voigt(x2 - x_stick, sigma, gamma)
    I_x2 = np.sum(Ii2, axis=0)

    x3 = x2.reshape(-1, N_bin).mean(axis=1)
    I_x3 = I_x2.reshape(-1, N_bin).mean(axis=1)

    # assert np.all(x == x3)
    return I_x3
