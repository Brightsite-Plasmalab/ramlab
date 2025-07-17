from typing_extensions import override
import numpy as np
from ramlab.molecules.ab_initio_molecule2 import AbInitioMolecule
from ramlab.molecules.intensity import Intensity
from ramlab.molecules.polarisation import Polarisation
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from scipy import constants


class O(AbInitioMolecule):
    """
    From vd Steeg et al (2021)
    https://doi.org/10.1364/OL.424102
    """

    # Molecule properties
    molecule_number = 99
    molecule_name = "O"
    isotope_number = 1

    @classmethod
    def crosssection_polarised(cls, transitions: Transitions) -> Intensity:
        _, dJ1 = transitions.filter(return_mask=True, dJ=-1)
        _, dJ2 = transitions.filter(return_mask=True, dJ=-2)

        sigma_dJ1 = dJ1 * 5.27e-31  # cm^2/sr
        sigma_dJ2 = dJ2 * 2.11e-31  # cm^2/sr
        sigma = (sigma_dJ1 + sigma_dJ2) * 1e-4  # m^2/sr

        conv = 4 * (np.pi**2) * (constants.fine_structure**2)  # See Long eq 5.10.5
        nu = transitions.scattering_wavenumber * 100  # Convert from cm^-1 to m^-1

        alpha2 = sigma / (nu**4) / conv  # m^6
        alpha2_au = alpha2 * 1e60  # A^6

        return Intensity(transitions.initial_J * 0.0, alpha2_au)  # m^6

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        state_initial = State(J=np.array([2, 2]))
        state_final = State(J=np.array([0, 1]))

        return state_initial, state_final

    @override
    @classmethod
    def E(cls, state: State) -> float:
        """Returns the energy of a given state in cm^-1."""

        return (state.J == 1) * (226 - 158) + (state.J == 2) * (226)

    @classmethod
    def degeneracy(cls, state: State) -> int:
        return state.J * 0 + 1

    @override
    @classmethod
    def transitions_metadata(
        cls,
        transitions: Transitions,
        laser_wavelength: float = 532.083e-9,
        polarisation: str = Polarisation.COMBINED,
    ) -> Transitions:
        transitions = super().transitions_metadata(
            transitions, laser_wavelength, polarisation
        )

        transitions.dJ = transitions.final_J - transitions.initial_J

        return transitions


if __name__ == "__main__":
    # Example usage
    transitions = O.get_all_transitions()
    intensity = O.crosssection_polarised(transitions)
    print(intensity)
