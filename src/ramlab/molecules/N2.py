import numpy as np

from ramlab.molecules.diatomic import SimpleDiatomicMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions


class N2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 45
    molecule_name = "N2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 6  # nuclear degeneracy for even J
    g_o = 3  # nuclear degeneracy for odd J

    # Energy constants
    w_e = 2.35862e03  # /m
    w_ex_e = 1.43444e01  # /m
    B_e = 1.99829e00  # /m
    alpha0_e_1 = 1.74130e-02  # /m
    D1_e = 5.49723e-06  # /m
    alpha1_e_1 = -7.67491e-08  # /m
    D2_e = -3.02983e-10  # /m
    alpha2_e_1 = -9.16227e-11  # /m
    D3_e = 0
    alpha3_e_1 = 0

    # Polarizability constants
    alpha_p_sq = 3.79e-60  # Isotropy invariant squared - C^4m^2 / J^2, Taken from Lucht, Ch 7, P42
    gamma_p_sq = 5.15e-60  # Anisotropy invariant squared - C^4m^2 / J^2, Taken from Lucht, Ch 7, P42
