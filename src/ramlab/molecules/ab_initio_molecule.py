from pathlib import Path
from ramlab.hitran.formatter import format_transitions_initial_final
from ramlab.molecules.hitran_linelist_molecule import LineListMolecule
import pandas as pd
import numpy as np

from ramlab.dirs import dir_data
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.util.decorators import abstractproperty


class AbInitioMolecule(LineListMolecule):
    """A class for molecules with ab initio linelists. The line list are not shipped with the package, but are generated.

    Subclasses must implement the following methods:
    - _make_transitions(laser_wavelength: float, state_initial: State, state_final: State) -> Transitions
    - _get_all_transitions() -> tuple[State, State]
    """

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        return (
            dir_data
            / "ab_initio"
            / cls.__name__
            / f"lambda_{laser_wavelength*1e9:.2f}nm.txt"
        )

    @classmethod
    def _make_linelist_file(
        cls,
        laser_wavelength: float,
        state_initial: State = None,
        state_final: State = None,
        polarisation: str = "= + T",
    ) -> Transitions:
        if state_initial is None or state_final is None:
            state_initial, state_final = cls._get_all_transition_states()

        transitions = cls._make_transitions(
            laser_wavelength, state_initial, state_final, polarisation
        )
        cls._save_hitran_linelist(transitions, cls.get_linelist_file(laser_wavelength))

        return transitions

    @classmethod
    def _save_hitran_linelist(cls, transitions: Transitions, path):
        hitran_format = cls._transitions_to_hitran(transitions)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for line in hitran_format.values:
                f.write(line + "\n")

    @abstractproperty
    def molecule_number(cls) -> int:
        """The molecule number according to the HITRAN database."""
        raise NotImplementedError()

    @abstractproperty
    def molecule_name(cls) -> str:
        """The molecule name according to the HITRAN database."""
        raise NotImplementedError()

    @abstractproperty
    def isotope_number(cls) -> int:
        """The isotope number according to the HITRAN database."""
        raise NotImplementedError()

    @classmethod
    def _make_transitions(
        cls, laser_wavelength: float, state_initial: State, state_final: State, polarisation: str = "= + T"
    ) -> Transitions:

        for state in [state_initial, state_final]:
            state.degeneracy = cls._calc_degeneracy(state)
            state.quanta_global = cls._format_quanta_global(state)
            state.quanta_local = cls._format_quanta_local(state)
            state.quanta = np.char.add(state.quanta_global, state.quanta_local)
            state.E = cls.E(state)

        transitions = Transitions()

        # Make sure the states have the same keys
        assert list(state_initial.keys()) == list(state_final.keys())
        # Copy the quantum numbers
        for col in state_initial.state.keys():
            transitions["initial_" + col] = state_initial.state[col]
            transitions["final_" + col] = state_final.state[col]

        transitions.dv = state_final.v - state_initial.v
        transitions.dJ = state_final.J - state_initial.J
        transitions.dE = state_final.E - state_initial.E
        if 'O' in state_initial.state.keys():
            transitions.dO = state_final.O - state_initial.O #Sum of spin and electronic orbital angular momentum
            transitions.dL = state_final.L - state_initial.L #Electronic orbital angular momentum
            transitions.dR = state_final.R - state_initial.R #Rotational angular momentum
            transitions.dN = state_final.N - state_initial.N #Sum of rotational and electronic angular momentum
            transitions.dS = state_final.S - state_initial.S #Electron spin
            transitions.dp = state_final.p - state_initial.p #Parity, either +1 or -1

        transitions.vacuum_wavenumber = state_final.E - state_initial.E
        transitions.crosssection = cls._calc_crosssection(transitions, laser_wavelength, polarisation)
        transitions.depolarization_ratio = cls._calc_depolarization_ratio(transitions)
        transitions.molecule_number = cls.molecule_number
        transitions.isotope_number = cls.isotope_number

        return transitions

    @classmethod
    def _format_quanta_global(cls, state: State):
        """Format the global quanta in a (max) 15-character string.

        Args:
            state (State): the state to format
        """
        raise NotImplementedError()

    @classmethod
    def _format_quanta_local(cls, state: State):
        """Format the local quanta in a (max) 15-character string.

        Args:
            state (State): the state to format
        """
        raise NotImplementedError()

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        raise NotImplementedError()

    @classmethod
    def _calc_degeneracy(cls, state: State):
        raise NotImplementedError()

    @classmethod
    def _calc_crosssection(cls, transitions: Transitions):
        raise NotImplementedError()

    @classmethod
    def _calc_depolarization_ratio(cls, transitions: Transitions):
        raise NotImplementedError()

    @classmethod
    def _transitions_to_hitran(cls, transitions: Transitions) -> pd.Series:
        return format_transitions_initial_final(transitions)
