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
    w_e = 2.35862e03 / 1e-2  # /m
    w_ex_e = 1.43444e01 / 1e-2  # /m
    B_e = 1.99829e00 / 1e-2  # /m
    alpha0_e_1 = 1.74130e-02 / 1e-2  # /m
    D1_e = 5.49723e-06 / 1e-2  # /m
    alpha1_e_1 = -7.67491e-08 / 1e-2  # /m
    D2_e = -3.02983e-10 / 1e-2  # /m
    alpha2_e_1 = -9.16227e-11 / 1e-2  # /m
    D3_e = 0
    alpha3_e_1 = 0

    @classmethod
    def _calc_degeneracy(cls, state: State):
        degeneracy_nuclear = (state.J % 2) * cls.g_o + ((state.J + 1) % 2) * cls.g_e
        degeneracy = (2 * state.J + 1) * degeneracy_nuclear
        return degeneracy

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        return 1

    @classmethod
    def _calc_intensity_roomtemp(cls, transitions: Transitions):
        return 1
