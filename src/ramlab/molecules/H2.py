from pathlib import Path
from typing_extensions import override
import numpy as np
from ramlab.molecules.custom_linelist_molecule import CustomLinelistMolecule
from ramlab.molecules.diatomic2 import SimpleDiatomicMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.dirs import dir_data
import h2_rovib_me


class H2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 45
    molecule_name = "H2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 1  # nuclear degeneracy for even J
    g_o = 3  # nuclear degeneracy for odd J

    # Energy constants
    # w_e = 4.39449e03  # /cm
    # w_ex_e = 1.16986e02  # /cm
    # B_e = 6.07823e01  # /cm
    # alpha0_e_1 = 2.94099e00  # /cm
    # D1_e = 4.66150e-02  # /cm
    # alpha1_e_1 = 2.37251e-03  # /cm
    # D2_e = 5.08134e-05  # /cm
    # alpha2_e_1 = 1.35508e-05  # /cm
    # D3_e = 0
    # alpha3_e_1 = 0

    @classmethod
    @override
    def _get_all_transition_states(cls) -> tuple[State, State]:
        state_initial, state_final = super()._get_all_transition_states()

        # Filter for 0<=v<=4 and 0<=J<=15
        idx_valid = (
            (state_initial.v <= 4)
            & (state_final.v <= 4)
            & (state_initial.J <= 15)
            & (state_final.J <= 15)
            & (state_initial.v >= 0)
            & (state_final.v >= 0)
            & (state_initial.J >= 0)
            & (state_final.J >= 0)
        )

        state_initial = state_initial[idx_valid]
        state_final = state_final[idx_valid]

        return state_initial, state_final

    @classmethod
    @override
    def polarisability_mean(cls, transitions: Transitions) -> np.ndarray:
        return h2_rovib_me.compute_batch(
            "H2",
            transitions.initial_v,
            transitions.initial_J,
            transitions.final_v,
            transitions.final_J,
            532.083,
            "nm",
            "iso",
        )

    @classmethod
    @override
    def polarisability_anisotropy(cls, transitions):
        return h2_rovib_me.compute_batch(
            "H2",
            transitions.initial_v,
            transitions.initial_J,
            transitions.final_v,
            transitions.final_J,
            532.083,
            "nm",
            "aniso",
        )

    # @override
    # @classmethod
    # def dE(cls, transitions: Transitions):
    #     # H2 has a very complex energy level structure, and the energy levels are not well described by the simple diatomic model.
    #     # Therefore, we use HITRAN data to calculate the energy levels.

    #     M_hitran = CustomLinelistMolecule(
    #         Path(dir_data / "H2" / "HITRAN_linebyline_0_6000_cm-1.txt")
    #     )
    #     transitions_hitran = M_hitran.get_all_transitions()

    #     # The HITRAN database is meant for absorption, so the initial state is always the lower energy state.
    #     # We need to flip the transitions to get the correct energy difference.
    #     idx_Stokes: np.ndarray = super().dE(transitions) > 0
    #     idx_aStokes = ~idx_Stokes

    #     dE_hitran = np.zeros_like(transitions.vacuum_wavenumber)
    #     for i in range(len(transitions)):
    #         if idx_Stokes[i]:
    #             mask = (
    #                 (transitions_hitran.initial_v == transitions.initial_v[i])
    #                 & (transitions_hitran.initial_J == transitions.initial_J[i])
    #                 & (transitions_hitran.final_v == transitions.final_v[i])
    #                 & (transitions_hitran.final_J == transitions.final_J[i])
    #             )

    #         else:
    #             mask = (
    #                 (transitions_hitran.initial_v == transitions.final_v[i])
    #                 & (transitions_hitran.initial_J == transitions.final_J[i])
    #                 & (transitions_hitran.final_v == transitions.initial_v[i])
    #                 & (transitions_hitran.final_J == transitions.initial_J[i])
    #             )
    #         if not np.any(mask):
    #             dE_hitran[i] = super().dE(transitions[i])
    #             continue
    #         elif np.sum(mask) > 1:
    #             idx = np.where(mask)[0][0]
    #         else:
    #             idx = np.where(mask)[0]

    #         dE_hitran[i] = transitions_hitran[idx].vacuum_wavenumber
    #     dE_hitran[idx_aStokes] *= -1

    #     return dE_hitran

    @override
    @classmethod
    def E_vib(cls, v) -> float:
        """Calculate the vibrational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Vibrational energy in cm^-1"""

        # See Derek A. Long, eq. 5.9.3
        return (
            (-1) ** (2) * 4397.303795698471 * (v + 1 / 2) ** 1
            + (-1) ** (3) * 119.28584412106706 * (v + 1 / 2) ** 2
            + (-1) ** (4) * 0.5018108459849808 * (v + 1 / 2) ** 3
        )

    @override
    @classmethod
    def E_rot(cls, v, J) -> float:
        """Calculate the rotational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number
            J (int): rotational quantum number

        Returns:
            float: Rotational energy in cm^-1"""

        # Expand the treatment of Long (2002) eq 6.6.17 to include higher order terms
        # This is neccessary for accurate representation of energies, especially for H2.

        # It is recommended to fit these constants to experimental data from HITRAN.
        # NB: Do not combine constants from different sources, as these sources may have determined those constants from different orders of the expansion.

        # NB: Here we deviate from Long's notation. Long uses B, D, H, while we use B, D_1, D_2, D_3
        return (
            (
                60.66175715112368 * (v + 1 / 2) ** 0
                + -2.745221465499087 * (v + 1 / 2) ** 1
                + -0.14879473831652712 * (v + 1 / 2) ** 2
                + 0.04805471833732873 * (v + 1 / 2) ** 3
                + -0.004495620316741733 * (v + 1 / 2) ** 4
            )
            * (J * (J + 1)) ** 1
            + (
                -0.04341881186754313 * (v + 1 / 2) ** 0
                + -0.0025803528493919207 * (v + 1 / 2) ** 1
                + 0.0028049227888147206 * (v + 1 / 2) ** 2
                + -0.0007903056393882357 * (v + 1 / 2) ** 3
                + 7.277681412956464e-05 * (v + 1 / 2) ** 4
            )
            * (J * (J + 1)) ** 2
            + (
                2.8481364168655227e-05 * (v + 1 / 2) ** 0
                + 2.43932649605932e-05 * (v + 1 / 2) ** 1
                + -1.9099970426825623e-05 * (v + 1 / 2) ** 2
                + 5.880766672921694e-06 * (v + 1 / 2) ** 3
                + -6.01082681460338e-07 * (v + 1 / 2) ** 4
            )
            * (J * (J + 1)) ** 3
            + (
                3.463584515881339e-09 * (v + 1 / 2) ** 0
                + -7.21410404518034e-08 * (v + 1 / 2) ** 1
                + 6.223016504799459e-08 * (v + 1 / 2) ** 2
                + -2.121885488355858e-08 * (v + 1 / 2) ** 3
                + 2.3980475792481776e-09 * (v + 1 / 2) ** 4
            )
            * (J * (J + 1)) ** 4
        )


if __name__ == "__main__":
    print(H2.get_all_transitions())
