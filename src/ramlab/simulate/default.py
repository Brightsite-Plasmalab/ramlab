from ramlab.simulate.convolution import simulate_convolution


def simulate_default(x, x_stick, I_stick, sigma=1, gamma=0):
    return simulate_convolution(x, x_stick, I_stick, sigma, gamma)
