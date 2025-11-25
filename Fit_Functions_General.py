import pickle
import sif_parser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from toddler.data.spectrum import Spectrum
import wedme
import sif_reader
import scipy.constants as cons

def get_data_for_meas_new(f_fg,):
    f_bg = f_fg.with_stem(f_fg.stem + "_bg")

    s_fg = Spectrum.from_file(f_fg, new_axes=True).median(axis=2)
    s_fg._axis_lambda = 0

    s_bg = Spectrum.from_file(f_bg, new_axes=True).median(axis=2)
    s_bg._axis_lambda = 0

    s_raman = s_fg - s_bg

    # plt.figure()
    # plt.plot(s_fg.lambdanm, s_fg.sdata, label="fg")
    # plt.plot(s_bg.lambdanm, s_bg.sdata, label="bg")
    # plt.plot(s_raman.lambdanm, s_raman.sdata, label="fg-bg")

    return s_raman.squeeze().slice(lambda_start=555e-9).normalize(axis=0)

def get_data_for_meas_new(f_fg, f_bg, l_start, l_end):
    s_fg = Spectrum.from_file(f_fg, new_axes=True).median(axis=2)
    s_fg._axis_lambda = 0

    s_bg = Spectrum.from_file(f_bg, new_axes=True).median(axis=2)
    s_bg._axis_lambda = 0

    s_raman = s_fg - s_bg

    # plt.figure()
    # plt.plot(s_fg.lambdanm, s_fg.sdata, label="fg")
    # plt.plot(s_bg.lambdanm, s_bg.sdata, label="bg")
    # plt.plot(s_raman.lambdanm, s_raman.sdata, label="fg-bg")

    return s_raman.squeeze().slice(lambda_start=l_start,lambda_end=l_end).normalize(axis=0)

def load_spe_sif(fname):
    data, info = sif_reader.np_open(fname)
    print(info)
    print(data)
    wavelengths = sif_reader.utils.extract_calibration(info)
    y = np.transpose(np.sum(data[0, :, :], 0))
    return y, wavelengths[0:len(wavelengths):1]


def load_spe_sif_matrix(fname):
    data, info = sif_reader.np_open(fname)
    print(info)
    wavelengths = sif_reader.utils.extract_calibration(info)
    y = np.transpose(data[:, 0, :])
    print(y)
    return y, wavelengths[0:len(wavelengths):1]


def calc_new_w(T,av_lambda, molecule):
    if molecule == "O_158":
        lambda_0 = 1e-2/(536.33*10**-9)
        mass = cons.m_u * 16
    elif molecule =="O_226":
        lambda_0 = 1e-2/(538.39*10**-9)
        mass = cons.m_u * 16
    else:
        if molecule=="O2":
            mass = 2. * cons.m_u * 16
        elif molecule=="N2":
            mass = 2*cons.m_u*18
        elif molecule=="Combined":
            mass = 2. * cons.m_u * (17)
        lambda_0 = 1e-2/(av_lambda*10**-9)

    w_DG = (2./cons.c) * lambda_0 *np.sqrt(2*cons.k*T*np.log(2)/mass)
    return 2*w_DG

def calc_new_w2(T,molecule):
    if molecule=="O2":
        mass = 2. * cons.m_u * 16
    elif molecule=="N2":
        mass = 2*cons.m_u*18

    lambda_0 = np.sqrt((1e-4/ (532*10**-9)**2) + (1e-2/ (532*10**-9))) #- levels_Ram_vib[0])**2)
    #lambda_0 = np.sqrt((1e-4/ (532*10**-9)**2))
    w_DG = (2./cons.c) * lambda_0 *np.sqrt(2*cons.k*T*np.log(2)/mass)
    return 2*w_DG

print(calc_new_w(3000,541,"O2"))