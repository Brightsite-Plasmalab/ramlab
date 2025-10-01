import lmfit as lm
from lmfit import minimize, Parameters, fit_report
from src.ramlab.simulate.linespreadfunction import gaussian, voigt, lorentzian
from src.ramlab.molecules.H2 import H2
import matplotlib.pyplot as plt
import wedme.apply.dev
import numpy as np
from src.ramlab.molecules.calculated_linelist_molecule import CalculatedLinelistMolecule
from src.ramlab.simulate import simulate, simulate_raw, subsampled, simulate_raw_stijn,simulate_raw_skew
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
#import csv
from src.ramlab.fit.modifiers import *
from Fit_Functions_General import get_data_for_meas_new, calc_new_w
from src.ramlab.molecules import N2, O2
import wedme.apply.dev
import numpy as np
from src.ramlab.simulate import simulate
from src.ramlab.molecules.polarisation import Polarisation
from ttictoc import tic,toc


# Important information
directory = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\2025\\3\\31\\"
directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Rotational_Raman\\"
filter_char = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Filter_characterized.txt"


name_sig = "Room_Temp_Meas_8mm"
name_bg = "Room_Temp_Meas_8mm_bg"

molecule_list = ["O2"]
grating = "1700lmm"
slit = "60um"
pressure = 1000
laser_middle = 532.0
cutoff_min = 535
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

#ver_full = get_data_for_meas_new(fsig,fbg, l_start=laser_middle*1e-9).normalize(axis=0)
ver = get_data_for_meas_new(fsig,fbg, l_start=cutoff_min*1e-9,l_end=cutoff_max*1e-9).normalize(axis=0)
ver_full = get_data_for_meas_new(fsig,fbg, l_start=laser_middle*1e-9,l_end=cutoff_max_full*1e-9).normalize(axis=0)
#plt.plot(s_roomtemp_full.lambdanm, s_roomtemp_full.sdata, label="Full spectrum")
#plt.plot(ver.lambdanm, ver.sdata, label="Spectrum, cut-off")
#plt.legend()
#plt.show()

M = O2
transitions = M.get_all_transitions(laser_wavelength=532e-9, force_recalculate=True)
transitions = transitions[transitions.crosssection != np.nan]

I_const_par = M.get_intensity_constant(transitions, laser_wavelength=532e-9, polarisation=Polarisation.PARALLEL)
I_const_per = M.get_intensity_constant(transitions, laser_wavelength=532e-9, polarisation=Polarisation.PERPENDICULAR)
I_var = M.get_intensity_variable(transitions, T=300)

I = (I_const_par + I_const_per) * I_var
I /= np.nanmax(I)
dnu = M.dE(transitions)

#plt.vlines(dnu, 0, I, 'r')
#plt.plot(ver.dnu()/1e2, ver.c.normalize(axis=0).sdata)
#plt.xlim(50, 500)
#plt.show()

fitParameters = Parameters()
fitParameters.add("A", value=1, min=0.9, max=1.2, vary=True)
fitParameters.add("y0", value=0, min=0, max=0.001, vary=True)
fitParameters.add("T", value=296, min=292, max=300, vary=True)
fitParameters.add("w_g_r", value=0, min=-params[0]*0.05, max=params[0]*0.05, vary=True)
fitParameters.add("w_l_r", value=0, min=-params[1]*0.05, max=params[1]*0.05, vary=True)

fitParameters.add('k0', value=9.5, min=5, max=12, vary=True)
fitParameters.add('k1', value=1.023, min=1.01, max=1.03, vary=True)
fitParameters.add('k2', value=0.00001, vary=True, min=-0.001, max=0.001)

#wavelengthCorrection = WavelengthAxisCorrection(1, initial_values=[1.,0.], vary_wl=True)
#wavelengthCorrection.add_parameters(pars=fitParameters)

def spectral_fit_residuals(pars: Parameters, meas, meas_full, I_const_per,I_const_par, test1=False, test2=False):
    tic()
    A = pars['A']
    y0 = pars['y0']
    T = pars['T']
    w_g = np.sqrt(params[0]**2 + calc_new_w(T, "O2")**2) + pars['w_g_r']
    w_g_o = np.sqrt(params[0]**2 + calc_new_w(T, "O")**2) + pars['w_g_r']
    w_l = params[1] + pars['w_l_r']
    k0 = pars['k0'].value
    k1 = pars['k1'].value
    k2 = pars['k2']


    meas = MeasurementSpectrum(meas)
    meas_full = MeasurementSpectrum(meas_full)
    #meas = wavelengthCorrection.modify(meas)

    meas = meas.data
    dnu_meas = meas.dnu()/1e2
    I_meas = meas.c.normalize(axis=0).sdata

    meas_full = meas_full.data
    dnu_meas_full = meas_full.dnu() / 1e2
    I_meas_full = meas_full.c.normalize(axis=0).sdata

    I_var = M.get_intensity_variable(transitions, T=T)
    I_stick_sim = (I_const_per + I_const_par) * I_var
    I_stick_sim /= np.nanmax(I_stick_sim)
    
    dnu_stick = transitions.vacuum_wavenumber
    dnu_stick_temp = dnu_stick.copy()
    dnu_stick_temp = k2*dnu_stick_temp**2 + dnu_stick_temp*k1 - k0

    if skewed_bool==True:
        dnu_new, I_sim_O2 = simulate_raw_skew(dnu_meas,dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l,
                                              skew=params[2])
    else:
        dnu_new, I_sim_O2 = simulate_raw_stijn(dnu_meas,dnu_meas_full, dnu_stick_temp, I_stick_sim, w_g, w_l)
    I_sim_O2 = A / np.nanmax(I_sim_O2)*I_sim_O2+ y0
    
    spec_binned = np.zeros(len(dnu_meas))
    for i in range(len(dnu_meas)):
        spec_binned[i]=np.mean(I_sim_O2[np.abs(dnu_meas[i]-dnu_new[:])<=0.4])
    
    if test1 is True:
        plt.plot(dnu_meas,I_meas)
        plt.plot(dnu_meas,spec_binned)
        plt.show()
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2
    elif test2 is True:
        return spec_binned
    else:
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas)**2


spectral_fit_residuals(fitParameters, ver.c.normalize(axis=0),ver_full.c.normalize(axis=0), I_const_per, I_const_par, test1=True)
out = minimize(spectral_fit_residuals, fitParameters, args=(ver.c.normalize(axis=0),ver_full.c.normalize(axis=0), I_const_per,I_const_par),method='leastq')
print(fit_report(out))

fig = plt.figure()
gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

fit = spectral_fit_residuals(out.params, ver.c.normalize(axis=0),ver_full.c.normalize(axis=0), I_const_per,I_const_par, test2=True)
res = ver.c.normalize(axis=0).sdata - fit

ax0.plot(ver.c.normalize(axis=0).dnu()/1e2, ver.c.normalize(axis=0).sdata, label='Signal')
ax0.plot(ver.c.normalize(axis=0).dnu()/1e2, fit, label='Fit')

ax0.set_title('Tr=Tv=' + str(round(out.params['T'].value,0)) + 'K')
plt.tick_params('x', labelbottom=False)
ax0.set_ylabel('Intensity (a.u.)')
ax0.legend(loc='lower right', fontsize=12)
ax0.grid(True)

ax1.plot(ver.c.normalize(axis=0).dnu()/1e2, res, label='Res2', linewidth=1)
ax1.set_xlabel('Raman shift ($cm^{-1}$)')
ax1.set_ylabel('Residual')
ax1.set_xlim(80, 500)
ax1.grid(True)
fig.tight_layout()


w_g = np.sqrt(params[0]**2 + calc_new_w(out.params['T'], "O2")**2) + out.params['w_g_r']
w_l = params[1] + out.params['w_l_r']

print("Gas Temperature: ", str(round(out.params['T'].value,0)))
print("FWHM of Gaussian: ",str(round(w_g,4)))
print("FWHM of Lorentzian: ", str(round(w_l,4)))
print('k0: ', str(round(out.params['k0'].value,4)))
print('k1: ', str(round(out.params['k1'].value,4)))
print('k2: ', str(round(out.params['k2'].value,7)))
plt.show()