from pathlib import Path
import pandas as pd
from src.ramlab.molecules.base import Molecule
from src.ramlab.molecules.transitions import Transitions
import re


class CustomLinelistMolecule:
    """
    This class is made for integration of the HITRAN / HITEMP database (https://hitran.org/) with Python.
    For this class to work, configure a custom output format with at least following parameters:
        Molecule ID,Transition ID,ν,A,qns',qns",g'g"
    """

    df: pd.DataFrame

    def __init__(self, file=None) -> None:
        assert file, "Please provide a file with the HITRAN data"

        self.df = pd.read_csv(file)

        # Parse (v,J) data for upper and lower states from the state string
        for i in ["v", "J"]:
            for j in ["p", "pp"]:
                # p, pp stand for upper, lower state respectively
                self.df[f"{i}{j}"] = self.df.apply(
                    lambda row: int(
                        re.search(rf"(?<={i}=)\d+", row[f"state{j}"]).group()
                    ),
                    axis=1,
                )

        self.df.rename(
            {
                "elower": "initial_E",
                "gpp": "initial_degeneracy",
                "vpp": "initial_v",
                "gp": "final_degeneracy",
                "vp": "final_v",
                "Jpp": "initial_J",
                "Jp": "final_J",
                "nu": "vacuum_wavenumber",
            },
            axis=1,
            inplace=True,
        )

    def get_all_transitions(
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

        transitions = Transitions(self.df)

        return transitions

    def dE(self, transitions):
        return transitions.vacuum_wavenumber
