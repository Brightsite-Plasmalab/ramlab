import numpy as np
from typing_extensions import List, Tuple, Dict, Type, AnyStr
from ramlab.molecules.base import Molecule
from ramlab.molecules.polarisation import Polarisation
from ramlab.molecules.transitions import Transitions


class CombinedMolecule:
    """
    Class to combine multiple molecular spectra into a single stick spectrum.

    Attributes:
        molecules (Dict[str, Tuple[Type[Molecule], Transitions]]):
            Dictionary mapping molecule names to their class and transitions.
        polarisation (Polarisation):
            Polarisation setting for the combined spectrum.
        I_const (List[np.ndarray]):
            List of intensity constants for each molecule.

    Example usage:
        ```python
        from ramlab.molecules.base import Molecule
        from ramlab.molecules.transitions import Transitions
        from ramlab.molecules.polarisation import Polarisation

        H2 = Molecule()
        H2_trans = Transitions(...)
        N2 = Molecule()
        N2_trans = Transitions(...)

        combined = CombinedMolecule(
            polarisation=Polarisation.COMBINED,
            H2=(H2, H2_trans),
            N2=(N2, N2_trans)
        )
        dnu_stick, lambda_stick, I_stick_sim = combined.stick(T=300, H2=0.7, N2=0.3)
        ```
    """

    molecules: Dict[AnyStr, Tuple[Type[Molecule], Transitions]]
    polarisation: Polarisation
    I_const: List[np.ndarray]

    def __init__(
        self,
        polarisation: Polarisation = Polarisation.COMBINED,
        **molecules: Dict[AnyStr, Tuple[type[Molecule], Transitions]],
    ):
        """
        Initialize CombinedMolecule with given molecules and polarisation.

        Args:
            polarisation (Polarisation): Polarisation setting.
            **molecules: Keyword arguments mapping molecule names to (Molecule, Transitions) tuples.
        """
        self.molecules = molecules
        self.polarisation = polarisation

        # Initialize intensity constants for each molecule
        # This is done to avoid recalculating them for each fit
        self.I_const = {}
        for molecule_name, (molecule, transitions) in molecules.items():
            I_c = molecule.get_intensity_constant(
                transitions, laser_wavelength=532e-9, polarisation=polarisation
            )
            self.I_const[molecule_name] = I_c

    def stick(self, T, **X):
        """
        Generate the combined stick spectrum for the specified temperature and molecule fractions.

        Args:
            T (float): Temperature in Kelvin.
            **X: Keyword arguments mapping molecule names to their fractions.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (dnu_stick, lambda_stick, I_stick_sim)
        """
        molecule_list = [
            (X.get(molecule), *self.molecules[molecule]) for molecule in X.keys()
        ]

        return combine_molecules(
            molecule_list,
            T,
            polarisation=self.polarisation,
            I_const=[self.I_const[x] for x in X.keys()],
        )

    def contributions(self, T, **X):
        """
        Generate the contributions of each molecule to the combined stick spectrum.

        Args:
            T (float): Temperature in Kelvin.
            **X: Keyword arguments mapping molecule names to their fractions.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]: Dictionary with molecule names as keys and
            tuples of (dnu_stick, lambda_stick, I_stick_sim) as values.
        """
        return {
            molecule: self.stick(T, **{molecule: X.get(molecule)})
            for molecule in self.molecules.keys()
        }

    def molecule_names(self):
        """
        Return a list of molecule names in the combined spectrum.
        """
        return list(self.molecules.keys())


def combine_molecules(
    molecules: List[Tuple[float, type[Molecule], Transitions]],
    T,
    polarisation: Polarisation = Polarisation.COMBINED,
    I_const: List[np.ndarray] = None,
):
    """
    Combine stick spectra from multiple molecules into a single spectrum.

    Args:
        molecules (List[Tuple[float, Molecule, Transitions]]): List of (fraction, Molecule, Transitions) tuples.
        T (float): Temperature in Kelvin.
        polarisation (Polarisation): Polarisation setting.
        I_const (List[np.ndarray], optional): List of intensity constants for each molecule.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (dnu_stick, lambda_stick, I_stick_sim)

    Example usage:
        H2 = Molecule()
        H2_trans = Transitions(...)
        N2 = Molecule()
        N2_trans = Transitions(...)

        molecules = [
            (0.7, H2, H2_trans),
            (0.3, N2, N2_trans)
        ]
        dnu_stick, lambda_stick, I_stick_sim = combine_molecules(molecules, T=300)
    """
    I_stick_sim = []
    dnu_stick = []
    if I_const is None:
        I_const = []

    for i, (X, molecule, transitions) in enumerate(molecules):
        I_var_i = molecule.get_intensity_variable(transitions, T=T)

        if len(I_const) <= i:
            I_const.append(
                molecule.get_intensity_constant(
                    transitions, laser_wavelength=532e-9, polarisation=polarisation
                )
            )

        I_stick_sim_i = (I_const[i]) * I_var_i
        I_stick_sim.append(I_stick_sim_i * X)
        dnu_stick.append(transitions.vacuum_wavenumber)

    # Merge the stick spectra by creating one numpy array
    I_stick_sim = np.concatenate(I_stick_sim)
    dnu_stick = np.concatenate(dnu_stick)

    dnu_scat = 1 / 532e-9 - dnu_stick * 1e2
    lambda_stick = 1e9 / dnu_scat

    return dnu_stick, lambda_stick, I_stick_sim
