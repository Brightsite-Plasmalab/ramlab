from ramlab.molecules.hitran_linelist_molecule import LineListMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
import pandas as pd
import numpy as np
from ramlab.dirs import dir_data


class NO_Hitran(LineListMolecule):
    # Molecule properties
    molecule_number = 8
    molecule_name = "NO"
    molecule_formula = "NO"
    isotope_number = 1

    @classmethod
    def process_hitran_data(cls, df2: pd.DataFrame) -> pd.DataFrame:
        ## Parse data specific to the NO molecule
        # See https://hitran.org/media/refs/HITRAN_QN_formats.pdf:
        #       NO "global" quanta are of class 1b. "local" quanta are of group 7.
        # Upper global quanta
        df2 = df2[df2["isotope_no"] == 1].copy()
        df2["final_quanta"] = df2["final_quanta_global"] + df2["final_quanta_local"]

        for state in ["initial", "final"]:
            # Global quanta
            df2[f"{state}_X"] = df2[f"{state}_quanta_global"].str[6:8]
            df2[f"{state}_Omega"] = df2[f"{state}_quanta_global"].str[
                8:11
            ]  # Is either 3/2 or 1/2

            # Convert state_Omega to a numeric value, stored in state_O
            df2[f"{state}_O"] = np.nan
            df2.loc[df2[f"{state}_Omega"] == "3/2", f"{state}_O"] = 3 / 2
            df2.loc[df2[f"{state}_Omega"] == "1/2", f"{state}_O"] = 1 / 2

            df2[f"{state}_v1"] = pd.to_numeric(
                df2[f"{state}_quanta_global"].str[13:15],
                errors="coerce",
            )

            # Local quanta
            df2[f"{state}_multiplex"] = df2[f"{state}_quanta_local"].str[0:1]
            df2[f"{state}_Br"] = df2[f"{state}_quanta_local"].str[2:4]
            df2[f"{state}_J"] = pd.to_numeric(
                df2[f"{state}_quanta_local"].str[4:9],
                errors="coerce",
            )
            df2[f"{state}_Sym"] = df2[f"{state}_quanta_local"].str[9]
            df2[f"{state}_F"] = pd.to_numeric(
                df2[f"{state}_quanta_local"].str[10:15],
                errors="coerce",
            )

        df2["crosssection"] = 1
        return df2

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        return dir_data / "NO" / "08_HITEMP2019_0.0-3500.0.par"


if __name__ == "__main__":
    M = NO_Hitran()
