import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from itertools import product

# # Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ramlab.math import make_quantum_numbers
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.molecules.ab_initio_molecule import AbInitioMolecule
from sympy.physics.wigner import wigner_3j


class NO(AbInitioMolecule):
    def __init__(self):
        print("Loading NO molecule...")

        # self.states = self._get_all_transition_states()

    # inital,final = self.states
    # self.transitions = self._make_transitions(laser_wavelength= 532e-9,state_initial=inital, state_final=final)
    # self.transitions = self._get_all_transitions()
    #     self._make_transitions(laser_wavelength= 532e-9,state_initial=inital_state, state_final=final_state)

    # Molecule properties
    molecule_name = "NO"
    molecule_number = 8
    isotope_number = 1

    # Degeneracy constants
    g_e = 2  # nuclear degeneracy for even J - needs verification
    g_o = 2  # nuclear degeneracy for odd J - needs verification

    # Polarisability tensor operator is molecular fixed reference frame, {\hat a_q^k}
    a_q0 = np.sqrt(9.1e-81)  # C^4 m^4 J^-2 - Source: Satija and Lucht, p16
    a_q0_to_N2 = 1.5  # Ratio of polarisability of NO to N2 - Source: Satija and Lucht
    a_q2_to_aq0 = 0.02  # Ratio of aq2 to aq0 - Source: Satija and Lucht

    # Energy constants
    # For the electroic ground state: Omega = +/- 1/2
    w_e = 1904.20  # cm^-1 Vibrational constant at eq. Source:NIST
    w_ex_e = 14.075  # cm^-1 Vibrational anharmonicity at eq. Source:NIST
    B_e = 1.67195  # cm^-1 Rotational constant at eq. Source:NIST
    alpha0_e_1 = 0.0171  # cm^-1. Vibrational anharmonicty correction of rotational constant Source: NIST

    # For the electronic higher state: Omega = +/- 3/2
    A = 123.160  # cm^-1. Spin-orbit coupling constant. Source:
    w_e_high = 1904.04  # cm^-1 Vibrational constant at eq. Source:NIST
    w_ex_e_high = 14.075  # cm^-1 Vibrational anharmonicity at eq. Source:NIST
    B_e_high = 1.72016  # cm^-1 Rotational constant at eq. Source:NIST
    alpha0_e_1_high = 0.0182  # cm^-1. Vibrational anharmonicty correction of rotational constant  Source: NIST

    level_energies = {}

    @classmethod
    def parity(cls, J, S):  # Determine parity
        return (-1) ** (J - S)

    # Intensity and population calculations
    @classmethod
    def _calc_degeneracy(cls, state: State):  # Determine rotational degeneracy
        """Rotational degenarcy - validated in Satija and Lucht"""
        return 2 * (state.J + 1)

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        """Cross sections - many terms need to be implemented"""
        b2 = cls.linestrength(transitions)  # Plazcek-Teller Coeffecients, squared
        print("Cross sections complete")
        # R2 = cls.R2(transitions) #Intensity correction factor for transitions in omega
        return b2  # *R2

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        print("Depolarization ratio complete")
        return 1.0

    @classmethod
    def _rc(
        cls, state, type
    ):  # Rotational coeffecients, used in calculation of linestrengths for Satija and Lucht method, taken from Zare - Angular Momentum, P303
        # Has been validatad against Fig. 3 from Satija and Lucht
        Bv = cls.B_v(state.v, np.where(state.F == 1, 1 / 2, 3 / 2))
        Y = cls.A / Bv  # Ratio of spin-orbit coupling constant to rotational constant
        X = (4 * (state.J - 0.5) * (state.J + 1.5) + (Y - 2) ** 2) ** 0.5

        if type in ["a", "d"]:
            return ((X + (Y - 2)) / (2 * X)) ** 0.5
        elif type == "b":
            return ((X - (Y - 2)) / (2 * X)) ** 0.5
        elif type == "c":
            return -(((X - (Y - 2)) / (2 * X)) ** 0.5)
        else:
            raise ValueError("Invalid type. Type must be one of ['a', 'b', 'c', 'd'].")

    @classmethod
    # Testing new method based on Satija and Lucht
    def linestrength(cls, transitions: Transitions):
        intensities = np.ones_like(transitions.initial_J) * np.nan

        return intensities

    # Energy calculations
    @classmethod
    def E(cls, state: State):
        # E_r = cls.E_rot(state)
        # E_v = cls.E_vib(state)
        # E = E_r + E_v
        # state.E = {'value': E, 'units': 'cm⁻¹'}
        # print(E)

        idF1 = state.F == 1
        idF2 = state.F == 2

        E = np.zeros(len(state.J)) * np.nan

        E_F1 = (
            -9.488e02 * 1
            + 1.682e00
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v1 + 1 / 2) ** 0
            - 1.743e-02
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v1 + 1 / 2) ** 1
            - 2.973e-06
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v1 + 1 / 2) ** 0
            + 1.002e-07
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v1 + 1 / 2) ** 1
            - 1.599e-10
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v1 + 1 / 2) ** 0
            - 2.798e-11
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v1 + 1 / 2) ** 1
            + 1.904e03 * (state.v1 + 1 / 2) ** 1
            - 1.409e01 * (state.v1 + 1 / 2) ** 2
            + 8.161e-03 * (state.v1 + 1 / 2) ** 3
        )
        E_F2 = (
            -8.286e02 * 1
            + 1.728e00
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v1 + 1 / 2) ** 0
            - 1.783e-02
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v1 + 1 / 2) ** 1
            - 8.013e-06
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v1 + 1 / 2) ** 0
            - 1.149e-07
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v1 + 1 / 2) ** 1
            + 1.680e-10
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v1 + 1 / 2) ** 0
            + 2.895e-11
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v1 + 1 / 2) ** 1
            + 1.904e03 * (state.v1 + 1 / 2) ** 1
            - 1.412e01 * (state.v1 + 1 / 2) ** 2
            + 1.097e-02 * (state.v1 + 1 / 2) ** 3
        )

        E[idF1] = E_F1[idF1]
        E[idF2] = E_F2[idF2]

        #  TODO: About time we start to enfore this
        assert np.all(~np.isnan(E))

        return E

    @classmethod
    def B_v(
        cls, v, O
    ) -> (
        float
    ):  # Used in rotational energy calculation and in calculation of wavefunction coeffecients
        """Calculates the rotational constant for a given vibrational level and electronic ground state."""
        # See Derek A. Long, eq. 6.6.14
        v = np.atleast_1d(v)
        O = np.atleast_1d(O)
        result = np.zeros_like(v, dtype=float)
        # Handle the case where abs(O) == 0.5
        mask_05 = np.abs(O) == 0.5
        result[mask_05] = cls.B_e - cls.alpha0_e_1 * (v[mask_05] + 1 / 2)

        # Handle the case where abs(O) == 1.5
        mask_15 = np.abs(O) == 1.5
        result[mask_15] = cls.B_e_high - cls.alpha0_e_1_high * (v[mask_15] + 1 / 2)

        if not np.all(mask_05 | mask_15):
            raise ValueError("Invalid Omega value")

        return result

    @classmethod
    def E_rot(cls, state: State) -> float:
        raise NotImplementedError(
            "Rotational energy calculation not implemented: we used fitted HITRAN data for total energy only."
        )

    @classmethod
    def E_vib(cls, state: State) -> float:
        raise NotImplementedError(
            "Vibrational energy calculation not implemented: we used fitted HITRAN data for total energy only."
        )

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        # Quantum numbers definitions - see Zare - Angular Momentum, P29

        vi = np.arange(0, 1)  # Vibrational quantum number
        Ji = np.arange(1 / 2, 81 / 2)  # Rotational quantum number
        Fi = np.array([1, 2])
        pi = np.array([1, -1])

        # Define transitions for each quantum number
        dv = np.array([0])
        dJ = np.array([-2, -1, 0, 1, 2])
        dF = np.array([-1, 0, 1])
        dp = np.array([-2, 0, 2])

        vi, Ji, Fi, pi, dv, dJ, dF, dp = np.meshgrid(vi, Ji, Fi, pi, dv, dJ, dF, dp)

        vf = vi + dv
        pf = pi + dp
        Ff = Fi + dF
        Jf = Ji + dJ

        states_initial = State(v=vi, v1=vi, J=Ji, F=Fi, p=pi)
        states_final = State(v=vf, v1=vf, J=Jf, F=Ff, p=pf)

        # Apply validity conditions
        legal = vf >= 0  # Final v and J states are positive

        # J must be at least 1/2
        legal &= (states_initial.J >= 0.5) & (states_final.J >= 0.5)
        legal &= np.isin(Ff, [1, 2])  # F is either -1 or +1
        legal &= np.abs(pf) == 1  # p is either -1 or +1

        # We'll allow everything else for now, but the transitions might still have 0 probability.

        # Remove states where final state == initial state
        rayleigh = states_initial.J == states_final.J
        for x in states_initial.keys():
            rayleigh &= states_initial[x] == states_final[x]
        legal &= ~rayleigh

        return states_initial[legal], states_final[legal]

    @classmethod
    def _format_quanta_global(cls, state: State):
        return np.char.mod("%2d", state.v)

    @classmethod
    def _format_quanta_local(cls, state: State):
        return np.char.mod("%2d", state.J)

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Parse the quantum numbers from the HITRAN format for local/global quantum numbers
        # e.g.
        #   df["lower_v"] = df["lower_quanta_global"].str[0:2].astype("Int64")
        raise NotImplementedError()

    @classmethod
    def _validate_transitions(cls, transitions: Transitions):
        return super()._validate_transitions(transitions)


if __name__ == "__main__":
    NO._get_all_transition_states()
