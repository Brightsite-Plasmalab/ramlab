import numpy as np


def project_to_axis(x, x_stick, I_stick):
    """Projects a stick spectrum onto a given x-axis by choosing the closest grid point.

    Args:
        x (np.array): The desired x-axis for the spectrum. Must be equally spaced.
        x_stick (np.array): The x-axis of the stick spectrum.
        I_stick (np.array): The intensity of the stick spectrum.
    """

    assert len(x_stick) == len(
        I_stick
    ), "x_stick and I_stick must have the same length."
    diffs_x = np.diff(x)
    assert np.all(diffs_x > 0), "x must be sorted in ascending order."
    assert np.all(~np.isnan(x)), "x must not contain NaNs."
    assert (np.max(diffs_x) - np.min(diffs_x)) < np.min(diffs_x) / 1e10, "x must be equally spaced."

    I_x = np.zeros_like(x, dtype=np.float32)

    for i, xi in enumerate(x_stick):
        idx = np.argmin(np.abs(x - xi))
        I_x[idx] += I_stick[i]

    return I_x
