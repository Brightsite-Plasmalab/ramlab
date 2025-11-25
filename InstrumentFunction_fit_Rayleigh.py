import lmfit as lm
from lmfit import minimize, Parameters, fit_report
from src.ramlab.simulate.linespreadfunction import gaussian, voigt, lorentzian
from src.ramlab.molecules.H2 import H2
import matplotlib.pyplot as plt
import wedme.apply.dev
import numpy as np
from src.ramlab.molecules.calculated_linelist_molecule import CalculatedLinelistMolecule
from src.ramlab.simulate import simulate, simulate_raw, subsampled, simulate_raw_stijn
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
from lmfit.models import skewed_voigt
#import csv
from src.ramlab.fit.modifiers import *
from Fit_Functions_General import get_data_for_meas_new, load_spe_sif, load_spe_sif_matrix
from src.ramlab.molecules import N2, O2
import wedme.apply.dev
import numpy as np
from Fit_Functions_General import calc_new_w
from ttictoc import tic,toc

# Important information
directory = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\2024\\12\\16\\300K_1000mbar_20slm\\"
directory_params = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\"
filter_char = "C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Filter_characterized.txt"

name_sig = "520nm_1s_60um_1700lmm"
name_bg = "520nm_1s_60um_1700lmm_bg"


molecule = "O2"
grating = "1700lmm"
slit = "60um"
pressure = 1000
laser_middle = 532.0
cutoff_min = 530.5
cutoff_max = 532.5

fsig = directory + name_sig + ".sif"
fbg = directory + name_bg + ".sif"
filter_data = np.loadtxt("C:\\Users\\P70085588\\Data\\Raman_Spectoscopy\\Parameters\\Filter_characterized.csv"
                         , delimiter="\t")

#ver_full = get_data_for_meas_new(fsig,fbg, l_start=laser_middle*1e-9).normalize(axis=0)

ver = get_data_for_meas_new(fsig,fbg, l_start=cutoff_min*1e-9,l_end=cutoff_max*1e-9).normalize(axis=0)


#I_filter = ver.sdata.copy() / filter_data[:,1][(filter_data[:,0] >= np.min(ver.lambdanm.copy())) & (filter_data[:,0] <= np.max(ver.lambdanm.copy()))]
#plt.plot(s_roomtemp_full.lambdanm, s_roomtemp_full.sdata, label="Full spectrum")
#plt.plot(ver.lambdanm, ver.sdata, label="Spectrum, cut-off")
#plt.plot(ver.lambdanm, I_filter, label="Spectrum, cut-off")
#plt.legend()
#plt.show()

fitParameters = Parameters()
fitParameters.add("A", value=1.06, min=0.9, max=1.5, vary=True)
fitParameters.add("y0", value=0.002, min=0, max=0.005, vary=False)
fitParameters.add("w_g", value=0.90, min=0, max=1.1, vary=True)
fitParameters.add("w_l", value=1.04, min=0, max=1.8, vary=True)
fitParameters.add("s0", value=0.06, min=0, max=0.2, vary=True)

fitParameters.add('k0', value=-2.4, min=0, max=-10, vary=True)
fitParameters.add('k1', value=1.02, min=1.01, max=1.06, vary=True)
fitParameters.add('k2', value=0.00001, vary=True, min=0, max=0.0001)

def spectral_fit_residuals(pars: Parameters, meas,N_bin, test1=False, test2=False):
    tic()
    A = pars['A']
    y0 = pars['y0']
    w_g = np.sqrt(pars['w_g']**2 + calc_new_w(296, "O2")**2)
    w_l = pars['w_l']
    k0 = pars['k0']
    k1 = pars['k1']
    k2 = pars['k2']
    s0 = pars['s0']

    meas = MeasurementSpectrum(meas)
    meas = meas.data
    dnu_meas = meas.dnu() / 1e2
    I_meas = meas.c.normalize(axis=0).sdata

    xmin = np.min(dnu_meas)
    xmax = np.max(dnu_meas)

    x2 = np.linspace(xmin, xmax, len(dnu_meas) * N_bin)

    #voigt_rayleigh = voigt2(x2**2*k2 + x2*k1 - k0,w_g,w_l)
    voigt_rayleigh = skewed_voigt(x=x2**2*k2 + x2*k1 - k0,amplitude=1,center=0,sigma=w_g,gamma=w_l, skew=s0)
    voigt_max = voigt_rayleigh.max(axis=0)
    spec = voigt_rayleigh/voigt_max*A + y0

    x_binned = np.zeros(len(dnu_meas))
    spec_binned = np.zeros(len(dnu_meas))
    for i in range(len(dnu_meas)):
        x_binned[i] = np.mean(x2[np.abs(dnu_meas[i]-x2[:])<=0.4])
        spec_binned[i]=np.mean(spec[np.abs(dnu_meas[i]-x2[:])<=0.4])

    if test1 is True:
        plt.plot(dnu_meas, I_meas)
        plt.plot(dnu_meas, spec_binned)
        plt.plot(x2, spec)
        plt.show()
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2
    elif test2 is True:
        plt.plot(dnu_meas, I_meas)
        plt.plot(dnu_meas, spec_binned)
        plt.plot(x2, spec)
        plt.show()
        return spec_binned
    else:
        print(f"Execution time: {toc() * 1e3:.0f}ms")
        return (spec_binned - I_meas) ** 2


spectral_fit_residuals(fitParameters, ver.c.normalize(axis=0),50,test1=True)
out = minimize(spectral_fit_residuals, fitParameters, args=(ver.c.normalize(axis=0),50))
print(fit_report(out))
fit = spectral_fit_residuals(out.params, ver.c.normalize(axis=0),50, test2=True)
res = ver.c.normalize(axis=0).sdata - fit
fig = plt.figure()
gs = GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)


ax0.plot(ver.c.normalize(axis=0).dnu() / 1e2, ver.c.normalize(axis=0).sdata, label='Signal')
ax0.plot(ver.c.normalize(axis=0).dnu() / 1e2, fit, label='Fit')

plt.tick_params('x', labelbottom=False)
ax0.set_ylabel('Intensity (a.u.)')
ax0.legend(loc='lower right', fontsize=12)
ax0.grid(True)

ax1.plot(ver.c.normalize(axis=0).dnu() / 1e2, res, label='Res2', linewidth=1)
ax1.set_xlabel('Raman shift ($cm^{-1}$)')
ax1.set_ylabel('Residual')
ax1.set_xlim(-50, 50)
ax1.grid(True)
fig.tight_layout()

plt.show()

params_to_save = np.column_stack(([round(out.params["w_g"].value, 4),
                                   round(out.params["w_l"].value, 4),
                                   round(out.params['s0'].value,4),
                                   round(out.params["k0"].value, 4),
                                   round(out.params["k1"].value, 4),
                                   round(out.params['k2'].value,4)
                                   ]))

np.savetxt(directory_params + "Rotational_Raman\\Rayleigh_1000mbar_1700lmm_60um_skewed.txt", params_to_save,
           delimiter="\t", header="w_G\tw_L\tk0\tk1\tk2")