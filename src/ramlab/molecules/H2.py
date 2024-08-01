import numpy as np
from ramlab.molecules.diatomic import SimpleDiatomicMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions


class H2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 45
    molecule_name = "H2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 1  # nuclear degeneracy for even J
    g_o = 3  # nuclear degeneracy for odd J

    # Energy constants
    w_e = 4.39449e03  # /cm
    w_ex_e = 1.16986e02  # /cm
    B_e = 6.07823e01  # /cm
    alpha0_e_1 = 2.94099e00  # /cm
    D1_e = 4.66150e-02  # /cm
    alpha1_e_1 = 2.37251e-03  # /cm
    D2_e = 5.08134e-05  # /cm
    alpha2_e_1 = 1.35508e-05  # /cm
    D3_e = 0
    alpha3_e_1 = 0

    # Polarizability constants - Need to be implemented for H2
    alpha_p_sq = 3.79e-60 #Isotropy invariant squared - C^4m^2 / J^2, Taken from Lucht, Ch 7, P42
    gamma_p_sq = 5.15e-60 #Anisotropy invariant squared - C^4m^2 / J^2, Taken from Lucht, Ch 7, P42

    #Herman-Wallis factor need to be added 
