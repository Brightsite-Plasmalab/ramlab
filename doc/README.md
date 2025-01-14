# Ramlab

The goal of this project is to provide researchers with a robust way of fitting Raman spectra. It achieves this using three modules:
- A library of **Raman cross-section line lists**. These are calculated ab initio for simple molecules (such as H2), or pre-calculated for complex molecules (such as CH4)
- Algorithms for generating **synthetic spectra** for a given composition and temperature.
- Utilities for **fitting** the synthetic spectra to experimental data.

This document discusses how these are achieved on a physics level, and gives an high-level overview of the code structure.

## Line lists

Calculation of line lists is by far the most difficult step of Raman spectroscopy.

RamLab includes pre-calculated line lists for the following molecules:
- CH4 (MeCaSDa)

RamLab includes methods for calculating line list ab initio for the following molecules:
- H2
- N2
- O2
- NO
- CO2
- CO
- O

Line lists contain the following information (contained in the `Transitions` class):
- Initial & final level quantum numbers (contained in the `State` class)
- Initial level energy
- Transition wavenumber
- Raman scattering cross-section
- Depolarization ratio

## Synthetic spectra

Generating spectra consists of two parts:
- Stick spectra calculation
  - Raman cross-section line lists
  - State population distributions
- Broadening function

### Stick spectra
The cross-sections are taken from the line list step. For the purposes of the developer, the broadening function is often dominated by the instrument function and can therefore be taken from a fit at a known temperature. 

What remains for this step is the proper calculation of population distributions. The population distribution is in general dependent on the quantum numbers and energy of the upper state, combined with the molecule environment. The molecule invironment could be calculated using a state-to-state plasma-kinetic model, but is more generally simplified to a multi-temperature Boltzmann distribution in one of the following ways:
- One temperature for all species and states (thermal equilibrium).  
  In this case, the population $n_i$ of state $i$ with degeneracy $g_i$ can be expressed as  
  $$
  n_i = g_i \exp{\left(-E_i / kT\right)} / \sum_j g_j \exp{\left(-E_j / kT\right)}
  $$

- Seperate temperatures $T_v$ and $T_r$ for the vibrational and rotational distribution functions.  
  In this case, the above formula can be applied to the vibrational energy levels first, and then again to the rotational levels within each vibrational level.



### Broadened spectra

We implemented a couple of calculation methods for a broadened spectrum.
Analytically, the intensity stick spectrum can be expressed as 
$$
I(\lambda)=\Sigma_i \delta_{\lambda_i,\ \lambda} I_i
$$

The broadened spectrum, following a linespread function $f(\Delta\lambda)$ is the expressed as 
$$
\begin{equation}
I(\lambda)=\Sigma_i f(\lambda - \lambda_i) I_i
\end{equation}
$$

#### Raw
1-to-1 implementation of the above equation.  
Pro's: most accurate, can deal with non-nonlinear spectral axis.
Con's: terribly slow for long linelists.

#### Convolution
Eq. (1) is equivalent to 
$$
I(\lambda)=\Sigma_i f(\lambda - \lambda_i) I_i \equiv f(\lambda) \circledast \left(
\Sigma_i \delta_{\lambda_i,\ \lambda} I_i
\right)
$$

If the measurement wavelengths $\{\lambda\}$ are approximately linear, this can be executed using a linear convolution.  
Pro's: accurate, fast.
Con's: only appliccable with linear spectral axis.


#### Convolution in the frequency domain
A convolution in the space domain is equivalent to a multiplication in the frequency domain. 

Article: [A discrete integral transform for rapid spectral synthesis](https://doi.org/10.1016/j.jqsrt.2020.107476)



## Spectral fitting

This step is comparatively easy: repeatedly compare synthetic spectra to experimental spectra, and adjust the simulation parameters (composition, temperatures, lineshape) until a good fit is achieved.