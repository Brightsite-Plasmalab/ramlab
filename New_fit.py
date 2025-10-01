import numpy as np
from ramlab.fit.multi import MultiMoleculeFitRecipe
from ramlab.molecules.polarisation import Polarisation
from ramlab.simulate.linespreadfunction import Voigt, Hubbert
from ramlab.fit.modifiers import WavelengthAxisCorrection, MeasurementSpectrum
from ramlab.simulate.raw import RawSimulationMethod, RawBinnedSimulationMethod
from ramlab.molecules import N2, O2, O
from ramlab.simulate.combine import CombinedMolecule
from Fit_Functions_General import get_data_for_meas_new, calc_new_w
import matplotlib.pyplot as plt
from toddler.data.spectrum import Spectrum
from lmfit import minimize, Parameters, fit_report
import wedme
from ttictoc import tic,toc
from pathlib import Path
import os
from ramlab.display import plot_results
#from labctl.experiments import SimpleLaserExperiment
from matplotlib.gridspec import GridSpec
wedme.dev()



directory = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\2024\\12\\6\\15mm_400W\\"
directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Rotational_Raman\\"

name_sig = "534nm_10s_60um_1700lmm_2mm"
name_bg = "534nm_10s_60um_1700lmm_2mm_bg"

directory = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\2025\\3\\19\\Discharge_Check\\"
directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Rotational_Raman\\"

name_sig = "534nm_60s_60um_1700lmm_3mm_1000mbar"
name_bg = "534nm_60s_60um_1700lmm_3mm_1000mbar_bg"

f_sig = directory + name_sig + ".sif"
f_bg = directory + name_bg + ".sif"


grating = "1700lmm"
slit = "60um"
laser_middle = 532.0
cutoff_min = 535.2
cutoff_max = 545
wav_len_av = (cutoff_min + cutoff_max)/2
cutoff_max_full = 550

directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Rotational_Raman\\"
params = np.loadtxt(directory_params + "Rayleigh_1000mbar_1700lmm_60um.txt",delimiter="\t")
o_params = np.loadtxt(directory_params + "O_" + grating + "_" + slit + ".txt",delimiter="\t")

ver = get_data_for_meas_new(f_sig,f_bg, l_start=cutoff_min*1e-9,l_end=cutoff_max*1e-9).normalize(axis=0)
ver_full = get_data_for_meas_new(f_sig,f_bg, l_start=laser_middle*1e-9,l_end=cutoff_max_full*1e-9).normalize(axis=0)

#plt.figure()
#plt.plot(ver.lambdanm, ver.sdata, label="Spectrum")
# plt.xlim(532, 540)
#plt.show()

# Define the molecules to be simulated
combined = CombinedMolecule(
    N2 = (N2, N2.get_all_transitions(laser_wavelength=532e-9)),
    O2 = (O2, O2.get_all_transitions(laser_wavelength=532e-9)),
    O = (O, O.get_all_transitions(laser_wavelength=532e-9)) # THe absolute value of the O cross-section is currently wrong.
)


fitParameters = Parameters()
fitParameters.add("A", value=1, min=0.9, max=1.2, vary=True)
fitParameters.add("T", value=4000, min=292, max=5000, vary=True)
fitParameters.add("y0", value=0.0, min=0.0, max=0.0001, vary=True)
fitParameters.add("w_g_r", value=0.0, min=-0.1, max=0.1, vary=True)
fitParameters.add("w_l_r", value=0.0, min=-0.1, max=0.1, vary=True)

fitParameters.add('x_N2', value=0, min=0,max=1, vary=False)
fitParameters.add('x_O2', value=0.4, min=0,max=1, vary=True)
fitParameters.add('x_O', value=0.6, min=0,max=1, vary=True)

fitParameters.add('k0', value=10, min=0, max=12, vary=True)
fitParameters.add('k1', value=1.0144, vary=False, min=0.9*1.0195, max=1.1*1.0195)
fitParameters.add('k2', value=0.3493, vary=True, min=0.8*0.3493, max=1.2*0.3493)


def spectral_fit_residuals(pars: Parameters, meas, meas_full, test1=False, test2=False):
    tic()
    A = pars['A']
    T_sim = pars['T']
    y0 = pars['y0']
    w_g = np.sqrt(params[0]**2 + calc_new_w(T_sim,wav_len_av, "Combined")**2) + pars['w_g_r']
    w_l = params[1] + pars['w_l_r']
    k0 = pars['k0']
    k1 = pars['k1']
    k2 = pars['k2']
    k3 = 0

    if T_sim >= 2200:
        X = {
            'N2': pars['x_N2'],
            'O2': pars['x_O2'],
            'O': pars['x_O']
        }
    elif T_sim < 2200:
        X = {
            'N2': pars['x_N2'],
            'O2': pars['x_O2'],
            'O': 0
        }

    def wav_cor(lam):
        lam_full = ((k3 * 10 ** -8) * lam ** 3
                    - (k2 * 10 ** -5) * lam ** 2
                    + k1 * lam - k0)
        return lam_full

    meas = MeasurementSpectrum(meas)
    meas_full = MeasurementSpectrum(meas_full)
    # meas = wavelengthCorrection.modify(meas)

    meas = meas.data
    dnu_meas = meas.dnu() / 1e2
    I_meas = meas.c.normalize(axis=0).sdata

    meas_full = meas_full.data
    dnu_meas_full = meas_full.dnu() / 1e2
    I_meas_full = meas_full.c.normalize(axis=0).sdata

    simulation = RawBinnedSimulationMethod(N_bin=40)
    dnu_stick_comb, _, I_stick_comb = combined.stick(T=T_sim, **X)

    dnu_stick_temp = wav_cor(dnu_stick_comb)


    dnu_new, I_tot_comb = simulation.simulate(dnu_meas, dnu_meas_full, dnu_stick_temp, I_stick_comb, w_g, w_l)

    spec_binned = np.zeros(len(dnu_meas))
    for i in range(len(dnu_meas)):
        spec_binned[i] = np.mean(I_tot_comb[np.abs(dnu_meas[i] - dnu_new[:]) <= 0.4])

    spec_binned = A/np.nanmax(spec_binned)*spec_binned + y0

    if test1 is True:
        plt.plot(dnu_meas, I_meas)
        plt.plot(dnu_meas, spec_binned)
        plt.show()
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2
    elif test2 is True:
        return spec_binned
    else:
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2

spectral_fit_residuals(fitParameters, ver.c.normalize(axis=0), ver_full.c.normalize(axis=0), test1=True)
out = minimize(spectral_fit_residuals, fitParameters,
               args=(ver.c.normalize(axis=0), ver_full.c.normalize(axis=0)), method='leastq')
print(fit_report(out))

fig = plt.figure()
gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

fit = spectral_fit_residuals(out.params, ver.c.normalize(axis=0), ver_full.c.normalize(axis=0), test2=True)
res = ver.c.normalize(axis=0).sdata - fit

ax0.plot(ver.c.normalize(axis=0).dnu() / 1e2, ver.c.normalize(axis=0).sdata, label='Signal')
ax0.plot(ver.c.normalize(axis=0).dnu() / 1e2, fit, label='Fit')

ax0.set_title('Tr=Tv=' + str(round(out.params['T'].value, 0)) + 'K')
plt.tick_params('x', labelbottom=False)
ax0.set_ylabel('Intensity (a.u.)')
ax0.legend(loc='upper right', fontsize=12)
ax0.grid(True)

ax1.plot(ver.c.normalize(axis=0).dnu() / 1e2, res, label='Res2', linewidth=1)
ax1.set_xlabel('Raman shift ($cm^{-1}$)')
ax1.set_ylabel('Residual')
ax1.set_xlim(80, 500)
ax1.grid(True)
fig.tight_layout()


x_N2 = out.params['x_N2'].value / (out.params['x_N2'].value + out.params['x_O2'].value + out.params['x_O'].value)
x_O2 = out.params['x_O2'].value / (out.params['x_N2'].value + out.params['x_O2'].value + out.params['x_O'].value)
x_O = out.params['x_O'].value / (out.params['x_N2'].value + out.params['x_O2'].value + out.params['x_O'].value)

print("Gas Temperature:\t\t", str(round(out.params['T'].value,0)))
print("FWHM of Gaussian:\t\t",str(round(np.sqrt(params[0]**2 + calc_new_w(out.params['T'].value,wav_len_av, "Combined")**2) + out.params['w_g_r'],4)))
print("FWHM of Lorentzian:\t\t", str(round(params[1] + out.params['w_l_r'],4)))
print('k0:\t\t', str(round(out.params['k0'].value,4)))
print('k1:\t\t', str(round(out.params['k1'].value,4)))
print('k2:\t\t', str(round(out.params['k2'].value,7)))
print("Mole Fraction Nitrogen (N2):\t\t", str(round(x_N2,4)))
print("Mole Fraction Oxygen (O2):\t\t", str(round(x_O2,4)))
print("Mole Fraction Oxygen Atom (O):\t\t", str(round(x_O,4)))

plt.show()