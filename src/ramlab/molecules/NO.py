import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from itertools import product

# # Add the src directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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

    level_energies = {}

    @classmethod
    def parity(cls, J, S):  # Determine parity
        return (-1) ** (J - S)

    # Intensity and population calculations
    @classmethod
    def _calc_degeneracy(
        cls, transitions: Transitions
    ):  # Determine rotational degeneracy
        """Rotational degenarcy - validated in Satija and Lucht"""
        J = transitions.J
        g = 2 * (J + 1)
        # g = (J * 0) +1
        return g

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
        cls, J, v, O, type
    ):  # Rotational coeffecients, used in calculation of linestrengths for Satija and Lucht method, taken from Zare - Angular Momentum, P303
        # Has been validatad against Fig. 3 from Satija and Lucht
        A = cls.A
        v = 0
        Bv = cls.B_v(v, O)
        # B_v = cls.B_v(v)
        Y = A / Bv  # Ratio of spin-orbit coupling constant to rotational constant
        X = (4 * (J - 0.5) * (J + 1.5) + (Y - 2) ** 2) ** 0.5
        if type in ["a", "d"]:
            return ((X + (Y - 2)) / (2 * X)) ** 0.5
        elif type in ["b", "c"]:
            coeff = ((X - (Y - 2)) / (2 * X)) ** 0.5
            if type == "c":
                coeff = -coeff
            return coeff
        else:
            raise ValueError("Invalid type. Type must be one of ['a', 'b', 'c', 'd'].")

    @classmethod
    # Testing new method based on Satija and Lucht
    def linestrength(cls, transitions: Transitions):
        rc = cls._rc
        intensities = []
        aq_0 = cls.a_q0
        aq_2 = 0  # np.sqrt(cls.a_q2_to_aq0  * cls.a_q0**2)
        a_j = []
        b_j = []
        J_list = []
        tran_length = len(transitions.linelist)
        print(f"Calculating linestrengths for {tran_length}...")
        for index, row in transitions.linelist.iterrows():
            O_l = row["initial_O"]
            O_u = row["final_O"]
            dO = row["dO"]

            J_l = row["initial_J"]
            J_u = row["final_J"]
            v_l = row["initial_v"]
            v_u = row["final_v"]
            dS = row["dS"]
            dN = row["dN"]
            S_l = row["initial_S"]
            S_u = row["final_S"]

            rc_i = (J_l, v_l, O_l)
            rc_f = (J_u, v_u, O_u)
            # Definition of F1 and F2 states are not yet correctly defined.
            # Attention also needs to be paid to the +/- term that preceeds the aq_2 term
            if (np.abs(O_l) == 0.5) & (
                np.abs(O_u) == 0.5
            ):  # For F1 -> F1 pure rotational transitions - Eq. 40 from Satija and Lucht
                intensity = (
                    (2 * J_l + 1)
                    * (2 * J_u + 1)
                    / 10
                    * (
                        (
                            aq_0
                            * (
                                (
                                    rc(*rc_i, "a")
                                    * rc(*rc_f, "a")
                                    * wigner_3j(J_l, J_u, 2, 1 / 2, -1 / 2, 0)
                                )
                                - (
                                    rc(*rc_i, "b")
                                    * rc(*rc_f, "b")
                                    * wigner_3j(J_l, J_u, 2, 3 / 2, -3 / 2, 0)
                                )
                            )  #   + aq_2 * ((-1)**(J_l - 0.5)) * (-(rc(*rc_i,'a')*rc(*rc_f,'b')* wigner_3j(J_l, J_u, 2, -1/2, -3/2, 2)) + (rc(*rc_i,'b')*rc(*rc_f,'a') * wigner_3j(J_l, J_u, 2, -3/2, -1/2, 2)))\
                        )
                        ** 2
                    )
                )
            elif (np.abs(O_l) == 1.5) & (
                np.abs(O_u) == 1.5
            ):  # For F2 -> F2 pure rotational transitions - Eq. 41 from Satija and Lucht
                intensity = (
                    (2 * J_l + 1)
                    * (2 * J_u + 1)
                    / 10
                    * (
                        (
                            aq_0
                            * (
                                (
                                    rc(*rc_i, "c")
                                    * rc(*rc_f, "c")
                                    * wigner_3j(J_l, J_u, 2, 1 / 2, -1 / 2, 0)
                                )
                                - (
                                    rc(*rc_i, "d")
                                    * rc(*rc_f, "d")
                                    * wigner_3j(J_l, J_u, 2, 3 / 2, -3 / 2, 0)
                                )
                            )  #   + aq_2 * ((-1)**(J_l - 0.5)) * (-(rc(*rc_i,'c')*rc(*rc_f,'d')* wigner_3j(J_l, J_u, 2, -1/2, -3/2, 2)) + (rc(*rc_i,'d')*rc(*rc_f,'c') * wigner_3j(J_l, J_u, 2, -3/2, -1/2, 2))) \
                        )
                        ** 2
                    )
                )
            # For F1 -> F2 electronic-rotational transitions - Eq. 60 from Satija and Lucht
            # elif (dO == 1) or (dO == -1): #Note place holder for F1 -> F2 transitions, equation is correctly implemented but rules need to be defined
            elif (np.abs(O_l) == 0.5) & (np.abs(O_u) == 1.5):
                intensity = (
                    (2 * J_l + 1)
                    * (2 * J_u + 1)
                    / 5
                    * (
                        (
                            aq_0
                            * (
                                (
                                    rc(*rc_i, "a")
                                    * rc(*rc_f, "c")
                                    * wigner_3j(J_l, J_u, 2, 1 / 2, -1 / 2, 0)
                                )
                                - (
                                    rc(*rc_i, "b")
                                    * rc(*rc_f, "d")
                                    * wigner_3j(J_l, J_u, 2, 3 / 2, -3 / 2, 0)
                                )
                            )  # + aq_2 * ((-1)**(J_l - 0.5)) * (-(rc(*rc_i,'a')*rc(*rc_f,'d')* wigner_3j(J_l, J_u, 2, -1/2, -3/2, 2)) + (rc(*rc_i,'b')*rc(*rc_f,'c') * wigner_3j(J_l, J_u, 2, -3/2, -1/2, 2)))\
                        )
                        ** 2
                    )
                )
            else:
                print(
                    f"No valid transition for: Omega_lower = {O_l} and delta_Omega = {dO}. Intensity = {intensity}"
                )
                intensity = 0
            a_j.append(rc(*rc_i, "a"))
            b_j.append(rc(*rc_i, "b"))
            J_list.append(J_l)
            intensities.append(float(intensity / ((2 * J_l) + 1)))
            # intensities.append(float(intensity / ((2*J_l)+1)))
        plt.figure()
        plt.scatter(J_list, a_j, label=r"$a_j$, $d_j$", marker=".")
        plt.scatter(J_list, b_j, label=r"$b_j$, $-c_j$", marker=".")
        plt.xlabel("J")
        plt.legend()
        plt.ylabel("Wavefunction Coeffecient")
        plt.show()
        return intensities

    # Energy calculations
    @classmethod
    def E(cls, state: State):  # Perhaps check if this is unessary duplication?
        # E_r = cls.E_rot(state)
        # E_v = cls.E_vib(state)
        # E = E_r + E_v
        # state.E = {'value': E, 'units': 'cm⁻¹'}
        # print(E)
        E = np.zeros(len(state.J)) * np.nan

        E_O_12 = (
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
        E_O_32 = (
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
        E[state.O == 1 / 2] = E_O_12[state.O == 1 / 2]
        E[state.O == 3 / 2] = E_O_32[state.O == 3 / 2]

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

    # #State calculations
    # @classmethod
    # def _get_all_transition_states(cls) -> tuple[State, State]:
    # # Quantum numbers definitions - see Zare - Angular Momentum, P297
    #     vi = np.arange(0, 1)  # Vibrational
    #     Ri = np.arange(0, 40, 1)  # Nuclear rotational angular momentum
    #     Si = np.array([0.5, -0.5])    # Electronic spin angular momentum
    #     Li = np.array([1, -1])   #  Electronic orbital angular momentum

    #     # Generate all possible combinations of these quantum numbers
    #     V, R, S, L = np.meshgrid(vi, Ri, Si, Li, indexing='ij')
    #     V, R, S, L = V.flatten(), R.flatten(), S.flatten(), L.flatten()
    #     # Calculate other Quantum numbers
    #     O = L + S  # Calculate Omega for each state
    #     J = R + S + L # Total angular momentum
    #     N = J - S  # Determine quantum number N

    #     # Define allowed changes (dv, dJ, dO)
    #     dv = np.array([ 0])  # Allowed change in vibrational quantum number
    #     dJ = np.array([-2, -1, 0, 1, 2])  # Allowed change in total angular momentum
    #     dL = np.array([-2, 0, 2])  # Allowed change in electronic orbital angular momentum
    #     dS = np.array([-1,0,1])  # Allowed change in electronic spin angular momentum

    #     # Calculate transitions - first remake list of inital states, having the same length as the final states list
    #     shape = (len(V), len(dv), len(dJ), len(dS))
    #     VI = np.broadcast_to(V[:, None, None, None], shape)
    #     RI = np.broadcast_to(R[:, None, None, None], shape)
    #     SI = np.broadcast_to(S[:, None, None, None], shape)
    #     LI = np.broadcast_to(L[:, None, None, None], shape)
    #     JI = RI + SI + LI # Total angular momentum
    #     OI = LI + SI  # Calculate Omega for each state: +3/2, +1/2, -1/2, -3/2
    #     NI = JI - SI # Determine quantum number N
    #     PI = cls.parity(JI, SI) #Calculate parity

    #     VF = VI + dv[:, None, None]
    #     JF = JI + dJ[:, None]
    #     SF = SI + dS
    #     LF = LI + dL
    #     OF = LF + SF
    #     NF = JF - SF
    #     RF = NF - LF
    #     PF = cls.parity(JF, SF)

    #     # Flatten the arrays to create a list of final states
    #     vi, ri, ji, si, li, oi, ni, pi = VI.flatten(), RI.flatten(), JI.flatten(), SI.flatten(), LI.flatten(), OI.flatten(), NI.flatten(), PI.flatten()
    #     vf, rf, jf, sf, lf, of, nf, pf = VF.flatten(), RF.flatten(), JF.flatten(), SF.flatten(), LF.flatten(), OF.flatten(), NF.flatten(), PF.flatten()

    #     print ("Unique states:")
    #     for i in [vf, jf, sf, lf, of]:
    #         print(np.unique(i))
    #     print ("Unique transitions: [v,j,s,l,o]")
    #     for i in [vf-vi, jf-ji, sf-si, lf-li, of-oi]:
    #         print(np.unique(i))

    #     # Apply validity conditions
    #     legal = (vf >= 0)                 #Final v and J states are positive
    #     legal = legal & (jf >= 0.5) & (ji >= 0.5)      # J must be at least 1/2
    #   #  legal = legal & (-1.5 <= of) & (of <= 1.5 )     # Omega is between -3/2 and +3/2
    #     legal = legal &  np.isin(sf, [-1/2, 1/2])      # S is either -1/2 or +1/20
    #     legal = legal &  np.isin(lf, [-1, 1])       # L is either -1 or +1
    #     #legal = legal & (np.abs(nf-ni) <=2)            # Delta N is 1 or 0
    #     #legal = legal & (pi == pf)                      # Parity must be conserved, i.e. + -> + or - -> -
    #     #rayleigh = (vi == vf) & (ri == rf) & (ji == jf) & (si == sf) & (li == lf) & (oi == of) & (ni == nf)
    #    # legal &= ~rayleigh # Remove identical states
    #     print ("Unique legal states:")
    #     for i in [vf[legal],jf[legal],sf[legal],lf[legal],of[legal]]:
    #         print(np.unique(i))
    #     print ("Unique legal transitions: [v,j,s,l,o]")
    #     for i in [vf[legal]-vi[legal], jf[legal]-ji[legal], sf[legal]-si[legal], lf[legal]-li[legal], of[legal]-oi[legal]]:
    #         print(np.unique(i))
    #     states_inital= State(v=vi[legal], R=ri[legal], S=si[legal], L=li[legal], J=ji[legal], O=oi[legal], N=ni[legal], p = pi[legal])
    #     states_final = State(v=vf[legal], R=rf[legal], S=sf[legal], L=lf[legal], J=jf[legal], O=of[legal], N=nf[legal], p = pf[legal])

    #     return states_inital, states_final

    # State calculations
    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        # Quantum numbers definitions - see Zare - Angular Momentum, P29

        # Define initial quantum states
        vi = np.arange(0, 1)  # Vibrational quantum number
        Ri = np.arange(0, 40)  # Rotational quantum number
        Si = np.array([0.5, -0.5])  # Spin quantum number
        Li = np.array([1, -1])  # Orbital quantum number

        # Define transitions for each quantum number
        dv = np.array([0])
        dR = np.array([-2, -1, 0, 1, 2])
        dL = np.array([-2, 0, 2])
        dS = np.array([-1, 0, 1])

        # Calculate the total number of transitions
        total_transitions = len(dv) * len(dR) * len(dS) * len(dL)

        # Generate initial states
        initial_states = np.array(list(product(vi, Ri, Si, Li)))
        i_states_all = np.repeat(initial_states, total_transitions, axis=0)
        VI, RI, SI, LI = (
            i_states_all[:, 0],
            i_states_all[:, 1],
            i_states_all[:, 2],
            i_states_all[:, 3],
        )

        # Generate transitions for each quantum number and tile them appropriately
        dv_full = np.tile(dv, len(initial_states) * len(dR) * len(dS) * len(dL))
        dR_full = np.tile(
            np.repeat(dR, len(dv)), len(initial_states) * len(dS) * len(dL)
        )
        dS_full = np.tile(
            np.repeat(dS, len(dv) * len(dR)), len(initial_states) * len(dL)
        )
        dL_full = np.tile(
            np.repeat(dL, len(dv) * len(dR) * len(dS)), len(initial_states)
        )

        # Apply transitions
        VF = i_states_all[:, 0] + dv_full
        RF = i_states_all[:, 1] + dR_full
        SF = i_states_all[:, 2] + dS_full
        LF = i_states_all[:, 3] + dL_full

        # Calculate the final states
        # Calculate the final states
        OI = LI + SI
        OF = LF + SF
        NI = RI + LI
        NF = RF + LF
        JI = RI + SI + LI
        JF = RF + SF + LF
        PI = (-1) ** (JI - SI)
        PF = (-1) ** (JF - SF)

        # # Flatten the arrays to create a list of final states
        # vi, ri, ji, si, li, oi, ni, pi = VI.flatten(), RI.flatten(), JI.flatten(), SI.flatten(), LI.flatten(), OI.flatten(), NI.flatten(), PI.flatten()
        # vf, rf, jf, sf, lf, of, nf, pf = VF.flatten(), RF.flatten(), JF.flatten(), SF.flatten(), LF.flatten(), OF.flatten(), NF.flatten(), PF.flatten()

        # print ("Unique states:")
        # for i in [vf, jf, sf, lf, of, nf]:
        #     print(np.unique(i))
        # print ("Unique transitions: [v,j,s,l,o,n]")
        # for i in [vf-vi, jf-ji, sf-si, lf-li, of-oi, nf - ni]:
        #     print(np.unique(i))

        # Apply validity conditions
        legal = VF >= 0  # Final v and J states are positive
        legal = legal & (JF >= 0.5) & (JI >= 0.5)  # J must be at least 1/2
        legal = legal & (-1.5 <= OF) & (OF <= 1.5)  # Omega is between -3/2 and +3/2
        legal = legal & np.isin(SF, [-1 / 2, 1 / 2])  # S is either -1/2 or +1/2
        legal = legal & np.isin(LF, [-1, 1])  # L is either -1 or +1
        # legal = legal & (np.abs(NF - NI) <=3)            # Delta N is less than 3
        # legal = legal & (pi == pf)                      # Parity must be conserved, i.e. + -> + or - -> -
        rayleigh = (
            (VI == VF)
            & (RI == RF)
            & (JI == JF)
            & (SI == SF)
            & (LI == LF)
            & (OI == OF)
            & (NI == NF)
        )
        legal &= ~rayleigh  # Remove states where final state == initial state
        # print(len(vf[legal]))
        # print ("Unique legal states:")
        # for i in [vf[legal],jf[legal],sf[legal],lf[legal],of[legal]]:
        #     print(np.unique(i))
        # print ("Unique legal transitions: [v,j,s,l,o,n]")
        # for i in [vf[legal]-vi[legal], jf[legal]-ji[legal], sf[legal]-si[legal], lf[legal]-li[legal], of[legal]-oi[legal]]:
        #     print(np.unique(i))
        states_inital = State(
            v=VI[legal],
            R=RI[legal],
            S=SI[legal],
            L=LI[legal],
            J=JI[legal],
            O=OI[legal],
            N=NI[legal],
            p=PI[legal],
        )
        states_final = State(
            v=VF[legal],
            R=RF[legal],
            S=SF[legal],
            L=LF[legal],
            J=JF[legal],
            O=OF[legal],
            N=NF[legal],
            p=PF[legal],
        )

        return states_inital, states_final

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
