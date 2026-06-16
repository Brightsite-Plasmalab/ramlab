import pandas as pd
from ramlab.hitran.parser import parse_hitran_data
from ramlab.molecules.base import Molecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.molecules.polarisation import Polarisation
from ramlab.dirs import dir_data


class LineListMolecule(Molecule):

    @classmethod
    def _has_linelist_file(cls, laser_wavelength: float) -> bool:
        return cls.get_linelist_file(laser_wavelength).exists()

    @classmethod
    def get_linelist_file(
        cls, laser_wavelength: float = None, polarisation: str = "= + T"
    ) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        return dir_data / cls.__name__ / f"lambda_{laser_wavelength*1e9:.2f}nm.txt"

    @classmethod
    def _make_linelist_file(
        cls,
        laser_wavelength: float,
        state_initial: State = None,
        state_final: State = None,
    ) -> Transitions:
        raise NotImplementedError()

    @classmethod
    def get_all_transitions(
        cls,
        laser_wavelength: float = None,
        force_recalculate: bool = False,
        polarisation: str = "= + T",
    ) -> Transitions:
        """Returns all possible transitions for the molecule.

        Returns:
            Transitions: The transitions.
        """

        if not force_recalculate and cls._has_linelist_file(laser_wavelength):
            df: pd.DataFrame = parse_hitran_data(
                cls.get_linelist_file(
                    laser_wavelength=laser_wavelength, polarisation=polarisation
                )
            )
            df = cls.process_hitran_data(df)

            return Transitions(df)
        else:
            print("Generating linelist file...")
            return cls._make_linelist_file(laser_wavelength, polarisation=polarisation)

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the HITRAN data.

        Args:
            df: The HITRAN data.

        Returns:
            pd.DataFrame: The processed data.
        """
        raise NotImplementedError()

    @classmethod
    def E(cls, state: State) -> float:
        return state.E

    @classmethod
    def dE(cls, transitions: Transitions) -> float:
        return transitions.vacuum_wavenumber

    @classmethod
    def degeneracy(cls, state) -> int:
        return state.degeneracy

    @classmethod
    def crosssection(
        cls, transitions: Transitions, lambda_laser: float, polarisation: str
    ) -> float:
        if type(polarisation) is str:
            polarisation = Polarisation.str_to_enum(polarisation)
        if polarisation == Polarisation.COMBINED:
            return transitions.crosssection
        elif polarisation == Polarisation.PARALLEL:
            return transitions.crosssection_parallel
        elif polarisation == Polarisation.PERPENDICULAR:
            return transitions.crosssection_perpendicular
        else:
            raise ValueError(f"Invalid polarisation: {polarisation}")
