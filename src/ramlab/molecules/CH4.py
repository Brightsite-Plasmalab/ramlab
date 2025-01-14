import pandas as pd
from ramlab.molecules.base import Molecule
from ramlab.dirs import dir_data
from ramlab.hitran.parser import parse_hitran_data
from ramlab.molecules.hitran_linelist_molecule import LineListMolecule
from ramlab.molecules.state import State
import numpy as np
from scipy.constants import k, h, c

from ramlab.molecules.transitions import Transitions


# TODO: Make a class for downloading and/or extracting data from the HITRAN database
class CH4(LineListMolecule):
    # Molecule properties
    molecule_number = -1
    molecule_name = "CH4"
    molecule_formula = "CH_4"
    isotope_number = 1

    @classmethod
    def process_hitran_data(cls, df2: pd.DataFrame) -> pd.DataFrame:
        ## Parse data specific to the methane molecule
        # Final global quanta
        df2["final_quanta"] = df2["final_quanta_global"] + df2["final_quanta_local"]

        for state in ["initial", "final"]:
            # Global quanta
            df2[f"{state}_v1"] = df2[f"{state}_quanta_global"].str[3:5].astype("Int64")
            df2[f"{state}_v2"] = df2[f"{state}_quanta_global"].str[5:7].astype("Int64")
            df2[f"{state}_v3"] = df2[f"{state}_quanta_global"].str[7:9].astype("Int64")
            df2[f"{state}_v4"] = df2[f"{state}_quanta_global"].str[9:11].astype("Int64")
            df2[f"{state}_mp"] = (
                df2[f"{state}_quanta_global"].str[11:13].astype("Int64")
            )
            df2[f"{state}_sy"] = df2[f"{state}_quanta_global"].str[13:15]

            # Local quanta
            # See doi:10.1016/S0022-4073(03)00155-9 for meaning
            # Made some adjustments in the format as it appears they did not strictly follow the HITRAN2004 format:
            #         I3 1X A2 1X I3 A5
            # HITRAN: 2X I3 A2    I3 A5
            df2[f"{state}_J"] = df2[f"{state}_quanta_local"].str[0:3].astype("Int64")
            df2[f"{state}_C"] = df2[f"{state}_quanta_local"].str[4:6]
            df2[f"{state}_a"] = df2[f"{state}_quanta_local"].str[7:10].astype("Int64")
            df2[f"{state}_F"] = df2[f"{state}_quanta_local"].str[10:15]

        ## Calculate rotational and vibrational energy levels
        # Filter for the ground rotational state
        idx_ground_rotational_state = df2["initial_J"] == 0

        # Map the ground state transitions to the ground state energy
        ground_state_transitions = pd.DataFrame(
            index=df2["initial_quanta_global"][idx_ground_rotational_state].values,
            data=df2["initial_E"][idx_ground_rotational_state].values,
            columns=["initial_E"],
        ).drop_duplicates()

        # For each transition, get the energy of the ground state
        idx_global_quanta = df2["initial_quanta_global"]

        # Add the energy of the vibrational state
        df2["initial_E_vib"] = ground_state_transitions.loc[idx_global_quanta][
            "initial_E"
        ].values
        df2["initial_E_rot"] = df2["initial_E"] - df2["initial_E_vib"]

        # TODO: Contact maintainers of MeCaSDa to confirm the einstein coefficient is the cross-section.
        # The "room temperature intensity" divided by the boltzmann distribution and degeneracies results in a constant relation with the Einstein coefficient.
        # So I suspect that the Einstein coefficient is the cross-section.
        df2["crosssection"] = df2["einstein_A_coefficient"].values
        # df2["crosssection"] = df2["intensity"].values / np.exp(
        #     -100 * h * c * df2["initial_E"].values / (k * 296)
        # )

        return df2

    @classmethod
    def get_linelist_file(cls, laser_wavelength: float = None) -> str:
        return dir_data / "mecasda" / "extract_12CH4_2850_3000_0_pol.txt"


if __name__ == "__main__":
    M = CH4()
