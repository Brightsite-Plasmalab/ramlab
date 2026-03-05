from pathlib import Path

from typing_extensions import override
import pandas as pd

from ramlab.data_parsing.parser import parse_hitran_data
from ramlab.calculate_molecules.ab_initio_molecule2 import AbInitioMolecule
from ramlab.calculate_molecules.Molecule import Molecule
from ramlab.calculate_molecules.hitran_compatible_molecule import HitranCompatibleMolecule
from ramlab.state.intensity import Intensity
from ramlab.state.polarisation import Polarisation
from ramlab.state.transitions import Transitions
from ramlab.dirs import dir_data


class CalculatedLinelistMolecule(Molecule):
    """
    A wrapper class for calculate_molecules with calculated linelists.
    """

    M: AbInitioMolecule

    def __init__(self, M: HitranCompatibleMolecule):
        self.M = M

    def all_transitions(
        self,
        laser_wavelength: float = None,
        force_recalculate: bool = False,
        force_file: Path = None,
        polarisation: str = "= + T",
    ) -> Transitions:
        """Returns all possible transitions for the molecule.

        Returns:
            Transitions: The transitions.
        """

        if force_file:
            force_recalculate = False
            linelist_file = force_file
        else:
            linelist_file = self.get_linelist_file(
                laser_wavelength=laser_wavelength, polarisation=polarisation
            )

        if (not force_recalculate and linelist_file.exists()) or force_file:
            # Load the cached HITRAN data
            if force_file:
                linelist_file = force_file
            df: pd.DataFrame = parse_hitran_data(linelist_file)

            transitions = Transitions(df)
            transitions = self.M.process_hitran_data(transitions)
            transitions = self.M.transitions_metadata(transitions)
        else:
            print("Generating linelist file...")
            transitions = self.M.all_transitions()

            # Add all HITRAN data to the transitions
            hitran_format = self.M.format_to_hitran(transitions)

            # Save the transitions to a file
            Path(linelist_file).parent.mkdir(parents=True, exist_ok=True)
            with open(linelist_file, "w") as f:
                for line in hitran_format.values:
                    f.write(line + "\n")

        return transitions

    def get_linelist_file(
        self, laser_wavelength: float = None, polarisation: str = None
    ) -> str:
        """Returns the path to the line list file.

        Returns:
            str: The path to the line list file.
        """
        return (
            dir_data
            / "ab_initio"
            / self.M.__name__
            / f"lambda_{laser_wavelength*1e9:.2f}nm.txt"
        )

    @override
    def get_intensity(
        self, transitions, laser_wavelength=None, polarisation="= + T", **temperatures
    ):
        return self.M.get_intensity(
            transitions, laser_wavelength, polarisation, **temperatures
        )

    @classmethod
    def dE(cls, transitions: Transitions) -> float:
        return transitions.dE

    def population(self, state_initial, **temperatures) -> float:
        return self.M.population(state_initial, **temperatures)

    @override
    @classmethod
    def crosssection(
        cls, transitions: Transitions, laser_wavelength: float, polarisation: str
    ) -> float:
        cs = transitions.crosssection
        dr = transitions.depolarization_ratio
        polarisation = Polarisation.from_string(polarisation)
        # Depolarization ratio is defined as dr =  I_par / I_perp
        # and I = I_par + I_perp
        # so that I_perp = (1-I_perp)/dr
        #         I_perp * (1+1/dr) = 1/dr
        #         I_perp = 1/dr / (1 + 1/dr)
        # or
        # I_par = dr * (1-I_par)
        # I_par * (1+dr) = dr
        # I_par = dr / (1+dr)

        par = dr / (1 + dr)
        per = 1 - par

        # per = (1 / dr) / (1 + 1 / dr)
        # par = (1-per)

        I = Intensity(cs * per, cs * par)

        # project
        return I.for_polarisation(polarisation)
