from abc import abstractmethod
import pandas as pd
import numpy as np
from src.ramlab.hitran.lineformatter import format_transitions_initial_final
from src.ramlab.molecules.base import Molecule
from src.ramlab.molecules.state import State
from src.ramlab.molecules.transitions import Transitions


class HitranCompatibleMolecule(Molecule):
    """
    Subclassing molecules should:
    - implement molecule_number, molecule_name, molecule_formula, isotope_number
    - implement _format_quanta_global, _format_quanta_local
    - implement process_hitran_data

    From the Molecule base class, they should also implement:
    - E()
    - degeneracy()
    - crosssection(), depolarization_ratio()
    - get_all_transitions()
    """

    @property
    @staticmethod
    @abstractmethod
    def molecule_number(subclass) -> int:
        raise NotImplementedError()

    @property
    @staticmethod
    @abstractmethod
    def molecule_name(subclass) -> str:
        raise NotImplementedError()

    @property
    @staticmethod
    @abstractmethod
    def molecule_formula() -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def isotope_number(cls) -> int:
        raise NotImplementedError()

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
    def process_hitran_data(cls, transitions: Transitions) -> Transitions:
        """Processes the HITRAN data.

        Args:
            Transitions: The HITRAN data, containing some unprocessed fields (e.g. local_quanta_upper)

        Returns:
            Transitions: The processed data.
        """
        raise NotImplementedError()

    @classmethod
    def prepare_hitran_format(cls, transitions: Transitions) -> pd.Series:
        """Prepare the HITRAN format.

        Args:
            transitions (Transitions): The transitions, missing some data needed for the modified HITRAN format.

        Returns:
            Transitions: Transitions, containing all the data needed for our modified HITRAN format.
        """

        transitions.initial_quanta_local = cls._format_quanta_local(
            transitions.state_initial
        )
        transitions.final_quanta_local = cls._format_quanta_local(
            transitions.state_final
        )

        transitions.initial_quanta_global = cls._format_quanta_global(
            transitions.state_initial
        )
        transitions.final_quanta_global = cls._format_quanta_global(
            transitions.state_final
        )

        transitions.initial_quanta = np.char.add(
            transitions.initial_quanta_global, transitions.initial_quanta_local
        )
        transitions.final_quanta = np.char.add(
            transitions.final_quanta_global, transitions.final_quanta_local
        )

        transitions.molecule_number = cls.molecule_number
        transitions.isotope_number = cls.isotope_number
        transitions.molecule_name = cls.molecule_name
        transitions.molecule_formula = cls.molecule_formula

        return transitions

    @classmethod
    def format_to_hitran(cls, transitions: Transitions) -> pd.Series:
        """Format the transitions to HITRAN format.

        Args:
            transitions (Transitions): The transitions.

        Returns:
            pd.Series: The formatted transitions.
        """
        transitions = cls.prepare_hitran_format(transitions)
        return format_transitions_initial_final(transitions)
