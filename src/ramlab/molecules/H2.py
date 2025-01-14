from typing import override
import numpy as np
from ramlab.molecules.diatomic2 import SimpleDiatomicMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
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
            532,
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
            532,
            "nm",
            "aniso",
        )


if __name__ == "__main__":
    print(H2.get_all_transitions())
