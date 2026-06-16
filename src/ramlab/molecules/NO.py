import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from itertools import product
from typing_extensions import override
#from scipy.constants import k, hbar, c, pi, epsilon_0, fine_structure
import scipy.constants as cons
from ramlab.hitran.parser import parse_hitran_data

# # Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ramlab.math import make_quantum_numbers
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.molecules.ab_initio_molecule import AbInitioMolecule
from sympy.physics.wigner import wigner_3j
from ramlab.raman.common import akp2_perturbed2
from ramlab.molecules.polarisation import Polarisation
from ramlab.molecules.intensity import Intensity


class NO(AbInitioMolecule):
    # Molecule properties
    molecule_name = "NO"
    molecule_number = 8
    isotope_number = 1

    # Degeneracy constants
    # Nuclear spin of 16O = 0, nuclear spin of 14N = 1
    g_nuclear = 3  # nuclear degeneracy for the heteronuclear molecule

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
    '''
    # Intensity and population calculations
    @classmethod
    def _calc_degeneracy(cls, state: State):  # Determine rotational degeneracy
        """Rotational degenarcy - validated in Satija and Lucht"""
        return 2 * (state.J + 1)
    '''
    
    @classmethod
    @override
    # Lambda doubling is supposed to give 2 separated states
    def _calc_degeneracy(cls, state):
        degeneracy_nuclear = cls.g_nuclear
        degeneracy_rotational = 2 * state.J + 1
        return cls.g_nuclear * degeneracy_rotational
    
    # Returns squares of polarizabilities in directions parallel and perpendicula to the laser polarization.
    def a_sq_perturbed(transitions):
        # terms with k=1 are zero because laser photon energy is far from resonance, Satija eq. 26
        # a21  = akp_sq(2,  1, transitions.initial_J, transitions.final_J, np.abs(transitions.initial_O), np.abs(transitions.final_O))
        # a21  = akp2(2,  1, transitions.initial_J, transitions.final_J, np.abs(transitions.initial_O), np.abs(transitions.final_O))

        # Find the `major` and `minor` (perturbing) components of the initial and final states
        id_F1i = (
            transitions.initial_F == 1
        )  # id's of the transitions originating from an F1 state
        id_F1f = transitions.final_F == 1  # id's of the transitions to an F1 state

        # Find the a and b components of the initial and final states
        aji = NO._rc(transitions.state_initial, type="a")
        bji = NO._rc(transitions.state_initial, type="b")
        ajf = NO._rc(transitions.state_final, type="a")
        bjf = NO._rc(transitions.state_final, type="b")

        p_tot_i = transitions.initial_p * (-1) ** (transitions.initial_J - 1 / 2)
        p_tot_f = transitions.final_p * (-1) ** (transitions.final_J - 1 / 2)

        temp = np.ones_like(id_F1i)
        
        # ToDo: Check, what akp2_perturbed2 actually calculates.
        # If I understand correctly, it must be a square of the space-fixed frame irreducible polarizability tensor element.
        
        # Now we are concerned by pure rotational transitions and can skip calculation of a2_00
        a2_00 = 0
        '''
        a2_00 = akp2_perturbed2(
            0,
            0,
            transitions.initial_J,
            transitions.final_J,
            [  # Omega components of the initial state
                (np.where(id_F1i, aji, -bji) / np.sqrt(2), temp * 1 / 2),
                (np.where(id_F1i, aji, -bji) / np.sqrt(2) * p_tot_i, temp * -1 / 2),
                (np.where(id_F1i, bji, aji) / np.sqrt(2), temp * 3 / 2),
                (np.where(id_F1i, bji, aji) / np.sqrt(2) * p_tot_i, temp * -3 / 2),
            ],
            [  # Omega components of the final state
                (np.where(id_F1f, ajf, -bjf) / np.sqrt(2), temp * 1 / 2),
                (np.where(id_F1f, ajf, -bjf) / np.sqrt(2) * p_tot_f, temp * -1 / 2),
                (np.where(id_F1f, bjf, ajf) / np.sqrt(2), temp * 3 / 2),
                (np.where(id_F1f, bjf, ajf) / np.sqrt(2) * p_tot_f, temp * -3 / 2),
            ],
        )
        '''
        a2_21 = akp2_perturbed2(
            2,
            1,
            transitions.initial_J,
            transitions.final_J,
            [  # Omega components of the initial state
                (np.where(id_F1i, aji, -bji) / np.sqrt(2), temp * 1 / 2),
                (np.where(id_F1i, aji, -bji) / np.sqrt(2) * p_tot_i, temp * -1 / 2),
                (np.where(id_F1i, bji, aji) / np.sqrt(2), temp * 3 / 2),
                (np.where(id_F1i, bji, aji) / np.sqrt(2) * p_tot_i, temp * -3 / 2),
            ],
            [  # Omega components of the final state
                (np.where(id_F1f, ajf, -bjf) / np.sqrt(2), temp * 1 / 2),
                (np.where(id_F1f, ajf, -bjf) / np.sqrt(2) * p_tot_f, temp * -1 / 2),
                (np.where(id_F1f, bjf, ajf) / np.sqrt(2), temp * 3 / 2),
                (np.where(id_F1f, bjf, ajf) / np.sqrt(2) * p_tot_f, temp * -3 / 2),
            ],
        )

        # a21  = akp2_perturbed2(2, 1, transitions.initial_J, transitions.final_J, \
        #                        [(aji, O_initial_major), (np.where(id_F1i, bji, -bji), O_initial_minor)],
        #                        [(ajf, O_final_major), (np.where(id_F1f, bjf, -bjf), O_final_minor)])
        
        
        # Rotational part of the polarizability must be multiplied by the vibrational and electronic part, which we don't know.
        # According to Satija, <v=0 | MFRF a20 | v=0>^2 for NO is 1.5 times larger than for N2.
        # Taking (without any reasoning) that other elements behave same way,
        # we can calculate the vibrational part from alpha and gamma N2 values provided by Buldakov.
        # alpha^2 = 1/3 <v| a00 |v>^2 = 1.5 * (1.78e-24)^2
        # gamma^2 = 3/2 summa <v| a2i |v>^2 = 15/2 <v| a21 |v>^2 = (0.72e-24)^2
        # ToDo: determine this values experimentally.
        # What to do for v != 0?
        # We aim to present squares of polarizabilities in cm^6
        a2_00 *= 14.3e-48   # 3 * 1.5 * (1.78e-24)^2
        a2_21 *= 0.104e-48      # 2/15 * 1.5 * (0.72e-24)^2
        
        # See Long table A14.9 and definitions of G0 and G2 (eq. 14.7.6, 14.7.8)
        a2_xx = a2_00 / 3. + 2 * a2_21 / 3.
        a2_xy = a2_21 / 2.
        
        return a2_xx, a2_xy

    @classmethod
    def _calc_crosssection(
        cls, transitions: Transitions, laser_wavelength=532.083e-9, polarisation=Polarisation.COMBINED
    ):
        Polarisation.validate(polarisation)
        assert np.all(transitions.dv == 0), "Changes with dV != 0 are not yet implemented"
        
        a2xx, a2xy = cls.a_sq_perturbed(transitions)
        # Now we implement equations 20-21 from Satija.
        # Those equations are in SI, but we use CGS.
        # Let's use the equation from diatomic.py
        # Equations by Satija don't contain Placzek-Teller coefficient, they seems to be included into a2xx and a2xy
        # I'm not sure if 1/(2Ja + 1) is also included.
        nu = 1e-2/laser_wavelength - transitions.vacuum_wavenumber  # [cm^-1]
        
        # [cm^2/sr]
        return 16 * np.pi**4 * (nu**4) * a2xx, 16 * np.pi**4 * (nu**4) * a2xy

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        print("Depolarization ratio complete")
        return transitions.crosssection_parallel / transitions.crosssection_perpendicular

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
            * (state.v + 1 / 2) ** 0
            - 1.743e-02
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v + 1 / 2) ** 1
            - 2.973e-06
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v + 1 / 2) ** 0
            + 1.002e-07
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v + 1 / 2) ** 1
            - 1.599e-10
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v + 1 / 2) ** 0
            - 2.798e-11
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v + 1 / 2) ** 1
            + 1.904e03 * (state.v + 1 / 2) ** 1
            - 1.409e01 * (state.v + 1 / 2) ** 2
            + 8.161e-03 * (state.v + 1 / 2) ** 3
        )
        E_F2 = (
            -8.286e02 * 1
            + 1.728e00
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v + 1 / 2) ** 0
            - 1.783e-02
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 1
            * (state.v + 1 / 2) ** 1
            - 8.013e-06
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v + 1 / 2) ** 0
            - 1.149e-07
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 2
            * (state.v + 1 / 2) ** 1
            + 1.680e-10
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v + 1 / 2) ** 0
            + 2.895e-11
            * ((state.J - 1 / 2) * (state.J + 3 / 2)) ** 3
            * (state.v + 1 / 2) ** 1
            + 1.904e03 * (state.v + 1 / 2) ** 1
            - 1.412e01 * (state.v + 1 / 2) ** 2
            + 1.097e-02 * (state.v + 1 / 2) ** 3
        )

        E[idF1] = E_F1[idF1]
        E[idF2] = E_F2[idF2]
        
        # This fit gives negative energies for lower states,
        # as a result, some Raman transitions are discarded.
        # Let's add zeroth vibrational energy to prevent it.
        # Difference of zeroth vibrational energy between F1 and F2 must be already considered in the fit.
        E += cls.w_e / 2. + cls.w_ex_e / 4.

        #  TODO: About time we start to enforce this
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
        # Quantum numbers definitions - see Zare - Angular Momentum, P297

        vi = np.arange(0, 5)  # Vibrational quantum number
        Ji = np.arange(0.5, 80)  # Rotational quantum number
        Fi = np.array([1, 2])
        pi = np.array([1, -1]) # Is it e and f states for the lambda doubling?

        # Define transitions for each quantum number
        dv = np.array([0])
        dJ = np.array([-2, -1, 0, 1, 2])
        #dF = np.array([-1, 0, 1])
        dF = np.array([0,]) # electronic transitions are not implemented yet
        dp = np.array([-2, 0, 2])

        vi, Ji, Fi, pi, dv, dJ, dF, dp = make_quantum_numbers(
            vi, Ji, Fi, pi, dv, dJ, dF, dp
        )

        vf = vi + dv
        pf = pi + dp
        Ff = Fi + dF
        Jf = Ji + dJ

        states_initial = State(v=vi, J=Ji, F=Fi, p=pi)
        states_final = State(v=vf, J=Jf, F=Ff, p=pf)

        # Apply validity conditions
        legal = vf >= 0  # Final v and J states are positive

        # J must be at least 1/2
        legal &= (states_initial.J >= 0.5) & (states_final.J >= 0.5)
        # State F2 J=1/2 corresponds to non-existent state with omega=3/2 and must be excluded
        legal &= ~((states_initial.J == 0.5) & (states_initial.F == 2))
        legal &= ~((states_final.J == 0.5) & (states_final.F == 2))
        legal &= np.isin(Ff, [1, 2])  # F is either -1 or +1
        legal &= np.abs(pf) == 1  # p is either -1 or +1

        # We'll allow everything else for now, but the transitions might still have 0 probability.

        # Remove states where final state == initial state
        rayleigh = states_initial.J == states_final.J
        for x in states_initial.keys():
            rayleigh &= states_initial[x] == states_final[x]
        legal &= ~rayleigh

        return states_initial[legal], states_final[legal]
    
    # Returns all the states
    # Neglect hyperfine splitting
    @override
    @classmethod
    def _get_all_states(cls):
        vi = np.arange(0, 5)  # Vibrational quantum number
        Ji = np.arange(0.5, 80)  # Rotational quantum number
        Fi = np.array([1, 2])
        pi = np.array([1, -1])
        
        states = State(J=Ji).add_each(v=vi).add_each(F=Fi).add_each(p=pi)
        states = states[~((states.J == 0.5) & (states.F == 2))]
        return states

    @override
    @classmethod
    def get_partition_sum(cls, **temperatures) -> float:
        """Returns the partition sum of the molecule.

        Args:
            **temperatures: The temperatures in Kelvin.

        Returns:
            float: The partition sum of the molecule.
        """
        if len(temperatures) <= 1:
            T = temperatures["T"]
        else:
            raise NotImplementedError("Non-equilibrium is not implemented for NO.")

        state = cls._get_all_states()
        E = cls.E(state) * 100
        g = cls._calc_degeneracy(state)
        weights = g * np.exp(-cons.h * cons.c * E / (cons.k * T))
        return np.sum(weights)
     
    @classmethod
    def _format_quanta_global(cls, state: State):
        # According to HITRAN documentation, this field must contain omega.
        # However, we classify NO state in a different way.
        # Let's use this field to distinguish F1 and F2 states.
        F_str = np.char.mod("        %3d", state.F)
        v_str = np.char.mod("  %2d", state.v)
        return np.char.add(F_str, v_str)

    @classmethod
    def _format_quanta_local(cls, state: State):
        # Only one character is reserved for NO symmetry.
        # It is supposed to be "e" or "f", but I am not 100% sure if it is exactly our p=+/-1 states.
        # Let's write "0" for "+" states and "1" for "-" states 
        J_str = np.char.mod("    %5.1f", state.J)
        p_str = np.char.mod("%1d", np.signbit(state.p))
        return np.char.add(J_str, p_str)


    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Parse the quantum numbers from the HITRAN format for local/global quantum numbers
        # e.g.
        #   df["lower_v"] = df["lower_quanta_global"].str[0:2].astype("Int64")
        for state in ["initial", "final"]:
            # Global quanta
            df[f"{state}_F"] = df[f"{state}_quanta_global"].str[8:11].astype("Int64")
            df[f"{state}_v"] = df[f"{state}_quanta_global"].str[13:15].astype("Int64")

            df[f"{state}_J"] = df[f"{state}_quanta_local"].str[4:9].astype("float")
            df[f"{state}_p"] = (-1) ** df[f"{state}_quanta_local"].str[9].astype("Int64")
        
        df['dv'] = df['final_v'] - df['initial_v']
        df['dJ'] = df['final_J'] - df['initial_J']
        df['dp'] = df['final_p'] - df['initial_p']
        df['dF'] = df['final_F'] - df['initial_F']
        df['depolarization_ratio'] = df['einstein_A_coefficient']
        df['crosssection'] = df['intensity']
        df['crosssection_perpendicular'] = df['intensity'] / (1 + df['depolarization_ratio'])
        df['crosssection_parallel'] = df['depolarization_ratio'] * df['intensity'] / (1 + df['depolarization_ratio'])
        
        return df

    @classmethod
    def _validate_transitions(cls, transitions: Transitions):
        return super()._validate_transitions(transitions)

if __name__ == "__main__":
    NO._get_all_transition_states()
