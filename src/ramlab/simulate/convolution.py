import numpy as np
from lmfit.models import skewed_voigt
from scipy.special import voigt_profile
from src.ramlab.simulate.common import project_to_axis, project_to_axis2


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
    #assert (np.max(diffs_x) - np.min(diffs_x)) < np.min(
    #    diffs_x
    #) / 1e10, "x must be equally spaced."

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

def simulate_convolution2(x, x_stick, I_stick, sigma=1, gamma=0):
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
    #diffs_x = np.diff(x)
    #assert np.all(diffs_x > 0), "x must be sorted in ascending order."
    #assert np.all(~np.isnan(x)), "x must not contain NaNs."
    #assert (np.max(diffs_x) - np.min(diffs_x)) < np.min(
    #    diffs_x
    #) / 1e10, "x must be equally spaced."

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

    I_stick_projected = project_to_axis2(x, x_stick, I_stick)
    I_x = np.convolve(I_stick_projected, I_voigt, mode="same")

    return I_x

def simulate_convolution_skewed(x, x_stick, I_stick, sigma=1, gamma=0, skew=0):
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
    #diffs_x = np.diff(x)
    #assert np.all(diffs_x > 0), "x must be sorted in ascending order."
    #assert np.all(~np.isnan(x)), "x must not contain NaNs."
    #assert (np.max(diffs_x) - np.min(diffs_x)) < np.min(
    #    diffs_x
    #) / 1e10, "x must be equally spaced."

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

    I_voigt = skewed_voigt(x=x_voigt,amplitude=1,center=0, sigma=sigma, gamma=gamma, skew=skew)
    # I_voigt /= I_voigt.sum()

    I_stick_projected = project_to_axis2(x, x_stick, I_stick)
    I_x = np.convolve(I_stick_projected, I_voigt, mode="same")

    return I_x

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
