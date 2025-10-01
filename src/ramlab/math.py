import numpy as np


def make_equal_size(*x):
    # Make all input arrays the same size by broadcasting the smaller ones
    xp = np.broadcast_arrays(*x)
    return list(np.broadcast_to(xi, xp.shape) for xi in x)


def make_quantum_numbers(*x):
    X = np.meshgrid(*x)
    return [np.ravel(Xi) for Xi in X]


def wheren(*x):
    """
    Call with N boolean arrays to be paired with N value arrays. Returns the values where the boolean arrays are True.
    """
    assert (
        (len(x) % 2) == 0,
        "Same number of boolean and value arrays must be provided.",
    )
    N = len(x) // 2

    criteria, values = *x[:N], *x[N:]

    final_shape = np.broadcast_shapes(
        *[c.shape for c in criteria], *[v.shape for v in values]
    )
    values = np.ones(final_shape) * np.nan

    # Make all arrays the same size
    for i in range(N):
        criteria[i], values[i] = np.broadcast_arrays(criteria[i], values[i])

    # Loop over the arrays and set the values where the criteria are True
    for i in range(N):
        values = np.where(criteria[i], values[i], values)

    return values


def orthogonal(*x):
    return [xi[slice(None, *([np.newaxis] * i))] for i, xi in enumerate(x)]
