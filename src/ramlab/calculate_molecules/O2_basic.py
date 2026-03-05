import numpy as np
from ramlab.calculate_molecules.diatomic import SimpleDiatomicMolecule
from ramlab.state.state import State
from ramlab.state.transitions import Transitions


class O2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 1 #Needs to be checked
    molecule_name = "O2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 1  # nuclear degeneracy for even J
    g_o = 1  # nuclear degeneracy for odd J

    # Energy constants
    w_e = 1580.19  # /m Source: NIST
    w_ex_e = 11.98  # /m Source: NIST
    B_e = 1.4376   # /m Source: NIST
    alpha0_e_1 = 1.59e-02  # /m Source: NIST
    D1_e = 4.839e-06  # /m 
    alpha1_e_1 = -1.0e-08   # /m
    D2_e = -1e-10#-3.02983e-10  
    alpha2_e_1 = -1e-11#-9.16227e-11  
    D3_e = 0
    alpha3_e_1 = 0

    # Polarizability constants
    alpha_p_sq = 2.90e-60 #Isotropy invariant squared - C^4m^2 / J^2, Taken from Lucht, Ch 7, P42
    gamma_p_sq = 8.73-60 #Anisotropy invariant squared - C^4m^2 / J^2, Taken from Lucht, Ch 7, P42

    #Herman-Wallis factor need to be added 