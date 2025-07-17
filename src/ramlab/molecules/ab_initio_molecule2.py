from ramlab.molecules.hitran_compatible_molecule import HitranCompatibleMolecule

from ramlab.molecules.intensity import Intensity
from ramlab.molecules.polarisation import Polarisation
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions


class AbInitioMolecule(HitranCompatibleMolecule):
    """
    Subclasses should:
    - override (and super-call) transitions() to include quantum number changes (e.g. dJ, dv) in the transitions
    - override _get_all_transition_states to return all possible combinations of initial and final states
    - implement crosssection_polarised()

    From HitranCompatibleMolecule, they should also:
    - implement molecule_number, molecule_name, molecule_formula, isotope_number
    - implement _format_quanta_global, _format_quanta_local
    - implement process_hitran_data

    From the Molecule base class, they should also implement:
    - E()
    - degeneracy()
    """

    @classmethod
    def crosssection_polarised(cls, transitions: Transitions) -> Intensity:
        raise NotImplementedError()

    @classmethod
    def crosssection(
        cls,
        transitions: Transitions,
        laser_wavelength: float,
        polarisation: Polarisation = Polarisation.COMBINED,
    ) -> float:
        print("Calculating cross-section")
        intensity = cls.crosssection_polarised(transitions)
        return intensity.for_polarisation(polarisation)

    @classmethod
    def depolarization_ratio(cls, transitions: Transitions) -> float:
        intensity = cls.crosssection_polarised(transitions)

        # Assume the incident light is polarised perpendicular to the scattering plane
        rho = intensity.intensity_parallel / intensity.intensity_perpendicular
        return rho

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        raise NotImplementedError()

    @classmethod
    def get_all_transitions(
        cls, laser_wavelength=532e-9, polarisation=Polarisation.COMBINED
    ):
        state_initial, state_final = cls._get_all_transition_states()

        return cls.transitions(
            laser_wavelength, state_initial, state_final, polarisation
        )

    @classmethod
    def transitions_calculate(
        cls,
        laser_wavelength: float,
        transitions: Transitions,
        polarisation: str = Polarisation.COMBINED,
    ) -> Transitions:
        crosssection_polarised = cls.crosssection_polarised(transitions)
        transitions.crosssection = crosssection_polarised.I
        transitions.depolarization_ratio = (
            crosssection_polarised.I_parallel / crosssection_polarised.I_perpendicular
        )

        return transitions

    @classmethod
    def transitions_metadata(
        cls,
        transitions: Transitions,
        laser_wavelength: float,
        polarisation: str = Polarisation.COMBINED,
    ) -> Transitions:
        state_initial, state_final = transitions.to_states()

        transitions.initial_degeneracy = cls.degeneracy(state_initial)
        transitions.final_degeneracy = cls.degeneracy(state_final)

        transitions.initial_E = cls.E(state_initial)
        transitions.final_E = cls.E(state_final)

        transitions.dE = (
            transitions.final_E - transitions.initial_E
        )  # The change in energy of the molecule
        transitions.vacuum_wavenumber = transitions.dE

        laser_frequency = 1e-2 / (laser_wavelength)  # Convert from nm to 1/cm
        transitions.scattering_wavenumber = (
            laser_frequency - transitions.vacuum_wavenumber
        )

        # Discard invalid transitions
        id_invalid = transitions.initial_E < 0
        id_invalid &= transitions.final_E < 0
        transitions = transitions[~id_invalid]
        if id_invalid.sum() > 0:
            print(f"Discarded {id_invalid.sum()} invalid transitions")

        transitions.molecule_number = int(cls.molecule_number)
        transitions.isotope_number = cls.isotope_number
        return transitions

    @classmethod
    def transitions(
        cls,
        laser_wavelength: float,
        state_initial: State,
        state_final: State,
        polarisation: str = Polarisation.COMBINED,
    ) -> Transitions:
        for state in [state_initial, state_final]:
            state.degeneracy = cls.degeneracy(state)
            state.E = cls.E(state)

        transitions = Transitions.for_states(state_initial, state_final)

        transitions = cls.transitions_metadata(
            transitions, laser_wavelength, polarisation
        )
        return cls.transitions_calculate(laser_wavelength, transitions, polarisation)
