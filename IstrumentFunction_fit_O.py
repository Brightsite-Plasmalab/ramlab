import lmfit as lm
from lmfit import minimize, Parameters, fit_report
from src.ramlab.simulate.linespreadfunction import gaussian, voigt, lorentzian
from src.ramlab.molecules.H2 import H2
import matplotlib.pyplot as plt
import wedme.apply.dev
import numpy as np
from src.ramlab.molecules.calculated_linelist_molecule import CalculatedLinelistMolecule
from src.ramlab.simulate import simulate, simulate_raw, subsampled, simulate_raw_stijn, simulate_raw_stijn_full, simulate_raw_skew
import pickle
import sif_parser
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sif_reader
import scipy.constants as cons
from matplotlib.gridspec import GridSpec
import scipy.integrate as inte
from scipy.special import voigt_profile as voigt2
# import csv
from src.ramlab.fit.modifiers import *
from Fit_Functions_General import get_data_for_meas_new, load_spe_sif, load_spe_sif_matrix, calc_new_w
from src.ramlab.molecules import N2, O2
import wedme.apply.dev
import numpy as np
from src.ramlab.simulate import simulate
from src.ramlab.molecules.polarisation import Polarisation
from ttictoc import tic, toc
from lmfit.models import skewed_voigt

# Important information
directory = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\2025\\3\\19\\Discharge_Check\\"
directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Rotational_Raman\\"

name_sig = "534nm_60s_60um_1700lmm_3mm_1000mbar"
name_bg = "534nm_60s_60um_1700lmm_3mm_1000mbar_bg"

molecule_list = ["O2","O"]
grating = "1700lmm"
slit = "60um"
pressure = 1000
laser_middle = 532.0
cutoff_min = 534.9
cutoff_max = 545
cutoff_max_full = 550
skewed_bool = False
if skewed_bool == True:
    params = np.loadtxt(directory_params + "Rayleigh_1000mbar_1700lmm_60um_skewed.txt",delimiter="\t")
else:
    params = np.loadtxt(directory_params + "Rayleigh_1000mbar_1700lmm_60um.txt",delimiter="\t")


fsig = directory + name_sig + ".sif"
fbg = directory + name_bg + ".sif"
filter_data = np.loadtxt("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Filter_characterized.csv"
                         , delimiter="\t")

# ver_full = get_data_for_meas_new(fsig,fbg, l_start=laser_middle*1e-9).normalize(axis=0)
ver = get_data_for_meas_new(fsig, fbg, l_start=cutoff_min * 1e-9, l_end=cutoff_max * 1e-9).normalize(axis=0)
ver_full = get_data_for_meas_new(fsig, fbg, l_start=530.5 * 1e-9, l_end=cutoff_max_full * 1e-9).normalize(axis=0)
# plt.plot(s_roomtemp_full.lambdanm, s_roomtemp_full.sdata, label="Full spectrum")
#plt.plot(ver.lambdanm, ver.sdata, label="Spectrum, cut-off")
# plt.legend()
#plt.show()

M = O2
transitions = M.get_all_transitions(laser_wavelength=532e-9,force_recalculate=True)
transitions = transitions[transitions.crosssection != np.nan]

I_const_par = M.get_intensity_constant(transitions, laser_wavelength=532e-9, polarisation=Polarisation.PARALLEL)
I_const_per = M.get_intensity_constant(transitions, laser_wavelength=532e-9, polarisation=Polarisation.PERPENDICULAR)
I_var = M.get_intensity_variable(transitions, T=300)

I = (I_const_par + I_const_per) * I_var
I /= np.nanmax(I)
dnu = M.dE(transitions)

# plt.vlines(dnu, 0, I, 'r')
# plt.plot(ver.dnu()/1e2, ver.c.normalize(axis=0).sdata)
# plt.xlim(50, 500)
# plt.show()

fitParameters = Parameters()
fitParameters.add("A", value=1, min=0.1, max=1.1, vary=True)
fitParameters.add("y0", value=0, min=0, max=0.001, vary=True)
fitParameters.add("T", value=4061, min=4000, max=4100, vary=True)

fitParameters.add("w_g_r", value=0, min=-params[0]*0.05, max=params[0]*0.05, vary=True)
fitParameters.add("w_l_r", value=0, min=-params[1]*0.05, max=params[1]*0.05, vary=True)

fitParameters.add("c_O", value=1.69, min=0, max=3, vary=True)

fitParameters.add('mid_O_1', value=158, min=155, max=159, vary=True)
fitParameters.add('mid_O_2', value=226, min=224, max=227, vary=True)

fitParameters.add('k0', value=10, vary=True, min=4, max=13)
fitParameters.add('k1', value=1.02, vary=True, min=0.99, max=1.05)
fitParameters.add('k2', value=0.3372, vary=True, min=-0.1, max=2)
fitParameters.add('k3', value=4.99999, vary=True, min=0, max=10)

# wavelengthCorrection = WavelengthAxisCorrection(1, initial_values=[1.,0.], vary_wl=True)
# wavelengthCorrection.add_parameters(pars=fitParameters)

def spectral_fit_residuals(pars: Parameters, meas, meas_full, I_const_per, I_const_par, test1=False, test2=False):
    tic()
    A = pars['A']
    y0 = pars['y0']
    T = pars['T']
    w_g = np.sqrt(params[0]**2 + calc_new_w(T, "O2")**2) + pars['w_g_r']
    w_g_o_158 = np.sqrt(params[0] ** 2 + calc_new_w(T, "O_158") ** 2)+ pars['w_g_r']
    w_g_o_226 = np.sqrt(params[0] ** 2 + calc_new_w(T, "O_226") ** 2) + pars['w_g_r']
    w_l = params[1] + pars['w_l_r']
    c_O = pars['c_O']
    mid_O_1 = pars['mid_O_1']
    mid_O_2 = pars['mid_O_2']
    k0 = pars['k0']
    k1 = pars['k1']
    k2 = pars['k2']
    k3 = pars['k3']

    def wav_cor(lam):
        lam_full = ((k3 * 10 ** -8) * lam ** 3
                    - (k2 * 10 ** -5) * lam ** 2
                    + k1 * lam- k0)
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

    """Oxygen Molecule"""
    I_var = M.get_intensity_variable(transitions, T=T)
    I_stick_sim = (I_const_per + I_const_par) * I_var
    I_stick_sim /= np.nanmax(I_stick_sim)

    dnu_stick = transitions.vacuum_wavenumber
    dnu_stick_temp = dnu_stick.copy()
    dnu_stick_temp = wav_cor(dnu_stick_temp)
    if skewed_bool==True:
        dnu_new, I_sim_O2 = simulate_raw_skew(dnu_meas,dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l,
                                              skew=params[2])
        I_sim_O = (
                    2.5 * skewed_voigt(x=(dnu_new - (k2 * mid_O_1 ** 2 + mid_O_1 * k1 - k0)), amplitude=1, center=0,
                                       sigma=w_g, gamma=w_l, skew=params[2]) +
                    skewed_voigt(x=(dnu_new - (k2 * mid_O_2 ** 2 + mid_O_2 * k1 - k0)), amplitude=1, center=0,
                                 sigma=w_g, gamma=w_l, skew=params[2]))  # Here find correcter values
    else:
        dnu_new, I_sim_O2 = simulate_raw_stijn(dnu_meas,dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l)
        I_sim_O = (2.5 * voigt2(dnu_new - wav_cor(mid_O_1), w_g_o_158, w_l) +
                         voigt2(dnu_new - wav_cor(mid_O_2), w_g_o_226, w_l))
    I_sim_O2_max = np.nanmax(I_sim_O2)
    I_sim_O2 /= I_sim_O2_max
    I_sim_O /= np.nanmax(I_sim_O)

    I_sim_mix = I_sim_O2 + c_O*I_sim_O
    I_sim_mix = A / np.nanmax(I_sim_mix) * I_sim_mix + y0

    spec_binned = np.zeros(len(dnu_meas))
    for i in range(len(dnu_meas)):
        spec_binned[i]=np.mean(I_sim_mix[np.abs(dnu_meas[i]-dnu_new[:])<=0.4])

    if test1 is True:
        plt.plot(dnu_meas, I_meas)
        plt.plot(dnu_meas, spec_binned)
        plt.show()
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2
    elif test2 is True:
        return [spec_binned, I_sim_O2_max]
    else:
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2


spectral_fit_residuals(fitParameters, ver.c.normalize(axis=0), ver_full.c.normalize(axis=0), I_const_per, I_const_par,
                       test1=True)
out = minimize(spectral_fit_residuals, fitParameters,
               args=(ver.c.normalize(axis=0), ver_full.c.normalize(axis=0), I_const_per, I_const_par), method='leastq')
print(fit_report(out))

fig = plt.figure()
gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

[fit, fit_O2_max] = spectral_fit_residuals(out.params, ver.c.normalize(axis=0), ver_full.c.normalize(axis=0), I_const_per,
                             I_const_par, test2=True)
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

meas = MeasurementSpectrum(ver.c.normalize(axis=0))
meas = meas.data
dnu_meas = meas.dnu() / 1e2
I_meas = meas.c.normalize(axis=0).sdata


meas_full = MeasurementSpectrum(ver_full.c.normalize(axis=0))
meas_full = meas_full.data
dnu_meas_full = meas_full.dnu() / 1e2
I_var = M.get_intensity_variable(transitions, T=out.params['T'])
I_stick_sim = (I_const_per + I_const_par) * I_var
I_stick_sim /= np.nanmax(I_stick_sim)

k3 = out.params['k3']
k2 = out.params['k2']
k1 = out.params['k1']
k0 = out.params['k0']
mid_O_1 = 158
mid_O_2 = 226
def wav_cor(lam):
    lam_full = ((k3 * 10 ** -8) * lam ** 3
                - (k2 * 10 ** -6) * lam ** 2
                + k1 * lam - k0)
    return lam_full

w_g = np.sqrt(params[0]**2 + calc_new_w(out.params['T'], "O2")**2) + out.params['w_g_r']
w_g_o_158 = np.sqrt(params[0]**2 + calc_new_w(out.params['T'], "O_158")**2) + out.params['w_g_r']
w_g_o_226 = np.sqrt(params[0]**2 + calc_new_w(out.params['T'], "O_226")**2) + out.params['w_g_r']
w_l = params[1] #+ out.params['w_l_r']

dnu_stick = transitions.vacuum_wavenumber
dnu_stick_temp = wav_cor(dnu_stick.copy())
if skewed_bool==True:
    dnu_new, I_sim_O2 = simulate_raw_skew(dnu_meas_full[32:-32], dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l, skew=params[2])
    I_sim_O = 2.5 * skewed_voigt(x=(dnu_new - (k2 * out.params['mid_O_1'] ** 2 + out.params['mid_O_1'] * k1 - k0)), amplitude=1,
                       center=0, sigma=w_g, gamma=w_l, skew=params[2])
else:
    dnu_new, I_sim_O2 = simulate_raw_stijn(dnu_meas_full[32:-32], dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l)
    I_sim_O = 2.5 * voigt2(dnu_new - wav_cor(mid_O_1), w_g_o_158, w_l)
    I_sim_O_2 = voigt2(dnu_new - wav_cor(mid_O_2), w_g_o_226, w_l)
    I_sim_O_both = (2.5 * voigt2(dnu_new - wav_cor(out.params['mid_O_1']), w_g_o_158, w_l)
                 + voigt2(dnu_new - wav_cor(out.params['mid_O_2']), w_g_o_226, w_l))

I_sim_O2 = I_sim_O2/fit_O2_max
I_sim_O_max = np.nanmax(I_sim_O)
I_sim_O = out.params['c_O']/I_sim_O_max*I_sim_O
I_sim_O_2 = out.params['c_O']/I_sim_O_max*I_sim_O_2
I_sim_O_both = out.params['c_O']/np.nanmax(I_sim_O_both)*I_sim_O_both

"""Oxygen Atom"""
#I_sim_O = voigt2(dnu_new - (out.params['mid_O_1'] * k1 - k0), w_g, w_l) + voigt2(dnu_new - (out.params['mid_O_2'] * k1 - k0), w_g,w_l)
#I_sim_O_2 = 2.5 * voigt2(dnu_new - (k2*out.params['mid_O_1']**2 + out.params['mid_O_1'] * k1 - k0), w_g, w_l) #+ voigt2(dnu_new - (o_params[6] * k1 - k0), w_g,w_l)
#I_sim_O_2 = 2.5 * skewed_voigt(x = (dnu_new - (k2*out.params['mid_O_1']**2 + out.params['mid_O_1']*k1 - k0)),amplitude=1,center=0,sigma=w_g, gamma=w_l, skew=0.06544268)
#I_sim_O /= np.nanmax(I_sim_O)

print(out.params['c_O'].stderr)

abs_cros_O2 = 2.61 * 1.64 * 10 ** -29
abs_cros_O = 5.27 * 10 ** -31
abs_cros_O_2 = 5.27 * 10 ** -31 / 2.5
rel_cros_O = abs_cros_O2/abs_cros_O
rel_cros_O_2 = abs_cros_O2/abs_cros_O_2
#rel_cros_O2 = abs_cros_O2 / abs_cros_N2


integral_O= inte.simpson(I_sim_O, x=dnu_new)
integral_O_2= inte.simpson(I_sim_O_2, x=dnu_new)
integral_O_both = inte.simpson(I_sim_O_both, x=dnu_new)
integral_O2 = inte.simpson(I_sim_O2, x=dnu_new)


ratio_O_O2 = (integral_O/integral_O2)*rel_cros_O
ratio_O_O2_1 = (integral_O_2/integral_O2)*rel_cros_O_2
ratio_O_O2_2 = (integral_O_both/integral_O2)*rel_cros_O



print("Ratio O over O2 (first O peak, only stokes):\t\t" + str(round(ratio_O_O2,4)) + " : 1")
print("Ratio O over O2 (second O peak, only stokes):\t\t" + str(round(ratio_O_O2_1,4)) + " : 1")
print("Ratio O over O2 (both peaks, only stokes):\t\t" + str(round(ratio_O_O2_2,4)) + " : 1")

print("Gas Temperature: ", str(round(out.params['T'].value,0)))
print("FWHM of Gaussian: ",str(round(w_g,4)))
print("FWHM of Lorentzian: ", str(round(w_l,4)))
print('k0: ', str(round(out.params['k0'].value,4)))
print('k1: ', str(round(out.params['k1'].value,4)))
print('k2: ', str(round(out.params['k2'].value,7)))
print('k3: ', str(round(out.params['k3'].value,10)))
plt.show()

params_to_save = np.column_stack(([round(out.params['T'].value,0),
                                   round(w_g,4),
                                   round(w_l,4),
                                   round(out.params["k0"].value,4),
                                   round(out.params["k1"].value,4),
                                   round(out.params["k2"].value,4),
                                   round(out.params["k3"].value,4),
                                   round(out.params['mid_O_1'].value,4),
                                   round(out.params['mid_O_2'].value,4)]))

np.savetxt(directory_params + molecule_list[1] + "_" + grating + "_" + slit + ".txt", params_to_save,
           delimiter="\t", header="Temp\tw_G\tw_L\tk0\tk1\tk2\tk3\tmid_O_1\tmid_O_2")

