import lmfit as lm
from lmfit import minimize, Parameters, fit_report
from src.ramlab.simulate.linespreadfunction import gaussian, voigt, lorentzian
from src.ramlab.molecules.H2 import H2
import matplotlib.pyplot as plt
import wedme.apply.dev
import numpy as np
from src.ramlab.molecules.calculated_linelist_molecule import CalculatedLinelistMolecule
from src.ramlab.simulate import simulate, simulate_raw, subsampled, simulate_raw_stijn, simulate_raw_stijn_full, simulate_raw_skew
from lmfit.models import skewed_voigt
import pickle
import sif_parser
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sif_reader
import cantera as ct
import scipy.constants as cons
from matplotlib.gridspec import GridSpec
import scipy.integrate as inte
from scipy.special import voigt_profile as voigt2
# import csv
from src.ramlab.fit.modifiers import *
from Fit_Functions_General import get_data_for_meas_new, calc_new_w
from src.ramlab.molecules import N2, O2
import wedme.apply.dev
import numpy as np
from src.ramlab.simulate import simulate
from src.ramlab.molecules.polarisation import Polarisation
from ttictoc import tic, toc

# Important information
directory = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\2024\\12\\21\\100mm_600W\\"
directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Rotational_Raman\\"

name_sig = "534nm_20s_60um_1700lmm_2mm"
name_bg = "534nm_20s_60um_1700lmm_2mm_bg"

molecule_list = ["O2","O"]
grating = "1700lmm"
slit = "60um"
pressure = 1000
laser_middle = 532.0
cutoff_min = 535.0
cutoff_max = 546
wav_len_av = (cutoff_min + cutoff_max)/2
cutoff_max_full = 550
skewed_bool = False
if skewed_bool == True:
    params = np.loadtxt(directory_params + "Rayleigh_1000mbar_1700lmm_60um_skewed.txt",delimiter="\t")
else:
    params = np.loadtxt(directory_params + "Rayleigh_1000mbar_1700lmm_60um.txt",delimiter="\t")
o_params = np.loadtxt(directory_params + "O_" + grating + "_" + slit + ".txt",delimiter="\t")

fsig = directory + name_sig + ".sif"
fbg = directory + name_bg + ".sif"


# ver_full = get_data_for_meas_new(fsig,fbg, l_start=laser_middle*1e-9).normalize(axis=0)
ver = get_data_for_meas_new(fsig, fbg, l_start=cutoff_min * 1e-9, l_end=cutoff_max * 1e-9).normalize(axis=0)
ver_full = get_data_for_meas_new(fsig, fbg, l_start=530.5 * 1e-9, l_end=cutoff_max_full * 1e-9).normalize(axis=0)
#ver_full2 = get_data_for_meas_new(fsig, fbg, l_start=520 * 1e-9, l_end=550 * 1e-9).normalize(axis=0)
# plt.plot(s_roomtemp_full.lambdanm, s_roomtemp_full.sdata, label="Full spectrum")
# plt.plot(ver.lambdanm, ver.sdata, label="Spectrum, cut-off")
# plt.legend()
# plt.show()

M = O2
transitions = M.get_all_transitions(laser_wavelength=532e-9, force_recalculate=True)
transitions = transitions[transitions.crosssection != np.nan]


I_const_par = M.get_intensity_constant(transitions, laser_wavelength=532e-9, polarisation=Polarisation.PARALLEL)
I_const_per = M.get_intensity_constant(transitions, laser_wavelength=532e-9, polarisation=Polarisation.PERPENDICULAR)
I_var = M.get_intensity_variable(transitions, T=300)


I = (I_const_par + I_const_per) * I_var
I /= np.nanmax(I)
dnu = M.dE(transitions)

#plt.vlines(dnu,\\ 0, I, 'r')
#plt.plot(I)
#plt.plot(ver_full.dnu()[20:-20]/1e2, ver_full.c.normalize(axis=0).sdata[20:-20])
#plt.xlim(0, 800)
#plt.show()
fitParameters = Parameters()
fitParameters.add("A", value=1, min=0.9, max=1.1, vary=True)
fitParameters.add("y0",value=0.0015, min=.0, max=0.0018, vary=True)
fitParameters.add("T", value=300, min=295, max=3500, vary=True)

fitParameters.add("w_g_r", value=0, min=-params[0]*0.1, max=params[0]*0.1, vary=True)
fitParameters.add("w_l_r", value=0, min=-params[1]*0.1, max=params[1]*0.1, vary=True)

fitParameters.add("c_O", value=0.01, min=0.0, max=0.1, vary=True)

fitParameters.add('k0', value=6, vary=True, min=2, max=13)
fitParameters.add('k1', value=o_params[4], vary=True, min=0.9*o_params[4], max=1.1*o_params[4])
fitParameters.add('k2', value=o_params[5], vary=True, min=0.5*o_params[5], max=2*o_params[5])

#fitParameters.add('k3', value=o_params[6], vary=True, min=0.9*o_params[6], max=1.1*o_params[6])
# wavelengthCorrection = WavelengthAxisCorrection(1, initial_values=[1.,1.], vary_wl=True)
# wavelengthCorrection.add_parameters(pars=fitParameters)


def spectral_fit_residuals(pars: Parameters, meas, meas_full, I_const_per, I_const_par, test1=False, test2=False):
    tic()
    A = pars['A']
    y0 = pars['y0']
    T = pars['T']
    w_g = np.sqrt(params[0]**2 + calc_new_w(T,wav_len_av, "O2")**2) + pars['w_g_r']
    w_g_o_158 = np.sqrt(params[0] ** 2 + calc_new_w(T,wav_len_av, "O_158") ** 2)+ pars['w_g_r']
    w_g_o_226 = np.sqrt(params[0] ** 2 + calc_new_w(T,wav_len_av, "O_226") ** 2) + pars['w_g_r']
    w_l = params[1] + pars['w_l_r']
    mid_O_1 = o_params[7]
    mid_O_2 = o_params[8]
    if T < 2200:
        c_O = 0
    else:
        c_O = pars['c_O']
    k0 = pars['k0']
    k1 = pars['k1']
    k2 = pars['k2']
    k3 = o_params[6]
    meas = MeasurementSpectrum(meas)
    meas_full = MeasurementSpectrum(meas_full)
    # meas = wavelengthCorrection.modify(meas)
    def wav_cor(lam):
        lam_full = ((k3 * 10 ** -8) * lam ** 3
                    - (k2 * 10 ** -5) * lam ** 2
                    + k1 * lam- k0)
        return lam_full

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
    #dnu_stick_temp = k3*dnu_stick_temp**3 + k2*dnu_stick_temp**2 + dnu_stick_temp * k1 - k0
    dnu_stick_temp = wav_cor(dnu_stick_temp)

    #plt.scatter(dnu_stick_temp, I_stick_sim)
    if skewed_bool==True:
        dnu_new, I_sim_O2 = simulate_raw_skew(dnu_meas,dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l,
                                              skew=params[2])
        I_sim_O = (
                    2.5 * skewed_voigt(x=(dnu_new - (k2 * mid_O_1 ** 2 + mid_O_1 * k1 - k0)), amplitude=1, center=0,
                                       sigma=w_g_o, gamma=w_l, skew=params[2]) +
                    skewed_voigt(x=(dnu_new - (k2 * mid_O_2 ** 2 + mid_O_2 * k1 - k0)), amplitude=1, center=0,
                                 sigma=w_g_o, gamma=w_l, skew=params[2]))  # Here find correcter values
    else:
        dnu_new, I_sim_O2 = simulate_raw_stijn(dnu_meas,dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l)
        #I_sim_O = (2.5 * voigt(dnu_new - (k3*mid_O_1**3 + k2 * mid_O_1 ** 2 + mid_O_1 * k1 - k0), w_g, w_l) +
        #                 voigt2(dnu_new - (k3*mid_O_2**3 + k2 * mid_O_2 ** 2 + mid_O_2 * k1 - k0), w_g,w_l))
        I_sim_O = (2.5 * voigt2(dnu_new - wav_cor(mid_O_1), w_g_o_158, w_l) +
                         voigt2(dnu_new - wav_cor(mid_O_2), w_g_o_226,w_l))
    I_sim_O2_max = np.nanmax(I_sim_O2)
    I_sim_O2 /= I_sim_O2_max
    I_sim_O /= np.nanmax(I_sim_O)

    I_sim_mix = I_sim_O2 + c_O*I_sim_O
    I_sim_mix = A / np.nanmax(I_sim_mix) * I_sim_mix + y0

    spec_binned = np.zeros(len(dnu_meas))
    for i in range(len(dnu_meas)):
        spec_binned[i]=np.mean(I_sim_mix[np.abs(dnu_meas[i]-dnu_new[:])<=0.4])

    if test1 is True:
        print(I_sim_O2_max)
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

[fit,fit_O2_max] = spectral_fit_residuals(out.params, ver.c.normalize(axis=0), ver_full.c.normalize(axis=0), I_const_per,
                             I_const_par, test2=True)
res = ver.c.normalize(axis=0).sdata - fit

ax0.plot(ver.c.normalize(axis=0).dnu() / 1e2, ver.c.normalize(axis=0).sdata, label='Signal')
ax0.plot(ver.c.normalize(axis=0).dnu() / 1e2, fit, label='Fit')

#ax0.set_title('Tr=Tv=' + str(round(out.params['T'].value, 0)) + 'K')
plt.tick_params('x', labelbottom=False)
ax0.set_ylabel('Intensity (a.u.)')
ax0.legend(loc='upper right', fontsize=12)
ax0.grid(True)

ax1.plot(ver.c.normalize(axis=0).dnu() / 1e2, res, label='Res2', linewidth=1)
ax1.set_xlabel('Raman shift ($cm^{-1}$)')
ax1.set_ylabel('Residual')
ax1.set_xlim(80, 600)
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

k0 = out.params['k0']
k1 = out.params['k1']
k2 = out.params['k2']
k3 = o_params[6]


def wav_cor(lam):
    lam_full = ((k3 * 10 ** -8) * lam ** 3
                - (k2 * 10 ** -6) * lam ** 2
                + k1 * lam - k0)
    return lam_full

w_g = np.sqrt(params[0]**2 + calc_new_w(out.params['T'], wav_len_av,"O2")**2) + out.params['w_g_r']
w_g_o_158 = np.sqrt(params[0]**2 + calc_new_w(out.params['T'],wav_len_av, "O_158")**2) + out.params['w_g_r']
w_g_o_226 = np.sqrt(params[0]**2 + calc_new_w(out.params['T'],wav_len_av, "O_226")**2) + out.params['w_g_r']
w_l = params[1] + out.params['w_l_r']

dnu_stick = transitions.vacuum_wavenumber
dnu_stick_temp = dnu_stick.copy()
dnu_stick_temp = wav_cor(dnu_stick_temp)
if skewed_bool==True:
    dnu_new, I_sim_O2 = simulate_raw_skew(dnu_meas_full[32:-32], dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l, skew=params[2])
    I_sim_O = 2.5 * skewed_voigt(x=(dnu_new - (k2 * o_params[6] ** 2 + o_params[6] * k1 - k0)), amplitude=1,
                       center=0, sigma=w_g, gamma=w_l, skew=params[2])
else:
    dnu_new, I_sim_O2 = simulate_raw_stijn(dnu_meas_full[32:-32], dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l)
    I_sim_O = 2.5 * voigt2(dnu_new - wav_cor(o_params[7]), w_g_o_158, w_l)
    I_sim_O_2 = voigt2(dnu_new - wav_cor(o_params[8]), w_g_o_226, w_l)
    I_sim_O_both = (2.5 * voigt2(dnu_new - wav_cor(o_params[7]), w_g_o_158, w_l)
                + voigt2(dnu_new - wav_cor(o_params[8]), w_g_o_226, w_l))



I_sim_O2 = I_sim_O2/fit_O2_max
I_sim_O_max = np.nanmax(I_sim_O)

if out.params['T'] < 2200:
    c_O = 0
else:
    c_O = out.params['c_O']
I_sim_O = c_O/I_sim_O_max*I_sim_O
I_sim_O_2 = c_O/I_sim_O_max*I_sim_O_2
I_sim_O_both = c_O/np.nanmax(I_sim_O_both)*I_sim_O_both

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
y0_list = np.ones(len(dnu_new))*out.params['y0']
integral_y0 = inte.simpson(y0_list, x=dnu_new)
integral_O2_y0 = integral_y0

ratio_O_O2 = (integral_O/integral_O2)*rel_cros_O
ratio_O_O2_1 = (integral_O_2/integral_O2)*rel_cros_O_2
ratio_O_O2_2 = (integral_O_both/integral_O2)*rel_cros_O
ratio_O_O2_y0 = (integral_O/integral_O2_y0)
ratio_av = (ratio_O_O2 + ratio_O_O2_1)/2

print(integral_O, integral_O_2, integral_O2, integral_y0, integral_O2_y0)
print("Percentage O:\t\t" + str(round(ratio_av/(1+ratio_av),4)) + " : 1")
print("Percentage O2:\t\t" + str(round(1/(1+ratio_av),4)) + " : 1")
if ratio_O_O2_y0 <= 0.1:
    print("Integral y0 much bigger than O atom. O atom data untrustworthy:\t\t\t\t" + str(round(ratio_O_O2_y0,4)))



print("Gas Temperature: ", str(round(out.params['T'].value,0)))
print("FWHM of Gaussian: ",str(round(w_g,4)))
print("FWHM of Lorentzian: ", str(round(w_l,4)))
print('k0: ', str(round(out.params['k0'].value,4)))
print('k1: ', str(round(out.params['k1'].value,4)))
print('k2: ', str(round(out.params['k2'].value,4)))
#print('k3: ', str(round(out.params['k3'].value,4)))


plt.show()

