from src.ramlab.molecules import N2, O2
import matplotlib.pyplot as plt
import wedme.apply.dev
import numpy as np
from src.ramlab.simulate import simulate

M_N2 = N2
M_O2 = O2
print(M_O2)
print(M_N2)
# Select the 5000 most intense transitions
transitions = M_O2.get_all_transitions(laser_wavelength=532e-9, force_recalculate=True)

I_stick = M_O2.get_intensity(transitions, T=300)
dnu_stick = transitions.vacuum_wavenumber

# Simulate the spectrum with Gaussian and Lorentzian broadening
dnu_sim = np.linspace(dnu_stick.min(), dnu_stick.max(), 10000)
I_sim = simulate(dnu_sim, dnu_stick, I_stick, 0.8, 1)

plt.figure()
plt.vlines(dnu_stick, 0, I_stick, color="r")
# plt.ylim(0, 1e21)

plt.figure()
plt.plot(dnu_sim, I_sim)
# plt.xlim(-500, 500)
# plt.ylim(0, 1e21)
plt.show()