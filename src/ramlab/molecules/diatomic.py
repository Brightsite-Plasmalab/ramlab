import numpy as np
import pandas as pd
from ramlab.molecules.ab_initio_molecule import AbInitioMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.util.decorators import abstractproperty


class SimpleDiatomicMolecule(AbInitioMolecule):
    @classmethod
    def E(cls, state: State) -> float:
        """Calculate the energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number
            J (int): rotational quantum number

        Returns:
            float: Energy in cm^-1
        """
        return cls.E_rot(state.v, state.J) + cls.E_vib(state.v)

    @classmethod
    def E_rot(cls, v, J) -> float:
        """Calculate the rotational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number
            J (int): rotational quantum number

        Returns:
            float: Rotational energy in cm^-1"""

        # Expand the treatment of Long (2002) eq 6.6.17 to include higher order terms
        # This is neccessary for accurate representation of energies, especially for H2.

        # It is recommended to fit these constants to experimental data from HITRAN.
        # NB: Do not combine constants from different sources, as these sources may have determined those constants from different orders of the expansion.

        # NB: Here we deviate from Long's notation. Long uses B, D, H, while we use B, D_1, D_2, D_3
        return (
            cls.B_v(v) * J * (J + 1)
            - cls.D1_v(v) * (J * (J + 1)) ** 2
            + cls.D2_v(v) * (J * (J + 1)) ** 3
            - cls.D3_v(v) * (J * (J + 1)) ** 4
        )

    @classmethod
    def B_v(cls, v) -> float:
        """Calculate the rotational constant in a given vibrational state.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Rotational constant in cm^-1
        """

        # See Derek A. Long, eq. 6.6.14
        return cls.B_e - cls.alpha0_e_1 * (v + 1 / 2)

    @abstractproperty
    def B_e(cls) -> float:
        """Calculate the rotational constant for the equilibrium internuclear separation.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Rotational constant in cm^-1
        """
        raise NotImplementedError

    @abstractproperty
    def alpha0_e_1(cls) -> float:
        """Calculate the first anharmonicity constant of a diatomic molecule.
        In our notation, this is the 1st order vibrational correction to the rotational constant.

        Returns:
            float: First anharmonicity constant in cm^-1

        """
        raise NotImplementedError

    @classmethod
    def D1_v(cls, v) -> float:
        """Calculate the (first) centrifugal distortion constant of a diatomic molecule in a given vibrational state.
        In our notation, this is the 1st order rotational correction to the rotational constant.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: (first) Centrifugal distortion constant in cm^-1
        """
        # See Derek A. Long, eq. 6.6.18
        # Again, we deviate from Long's notation.
        return cls.D1_e - cls.alpha1_e_1 * (v + 1 / 2)  # TODO: + or -?

    @abstractproperty
    def D1_e(cls) -> float:
        """Calculate the (first) centrifugal distortion constant for the equilibrium internuclear separation.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Centrifugal distortian constant in cm^-1
        """
        raise NotImplementedError

    @abstractproperty
    def alpha1_e_1(cls) -> float:
        """Get the (1st-order) anharmonicity correction to the first centrifugal distortion constant.

        Returns:
            float: First (1st-order) anharmonicity constant in cm^-1

        """
        raise NotImplementedError

    @classmethod
    def D2_v(cls, v) -> float:
        """Calculate the (second) centrifugal distortion constant of a diatomic molecule in a given vibrational state.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: (second) Centrifugal distortion constant in cm^-1
        """
        return cls.D2_e - cls.alpha2_e_1 * (v + 1 / 2)

    @abstractproperty
    def D2_e(cls) -> float:
        """Calculate the (second) centrifugal distortion constant for the equilibrium internuclear separation.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Centrifugal distortian constant in cm^-1
        """
        raise NotImplementedError

    @abstractproperty
    def alpha2_e_1(cls) -> float:
        """Get the (1st-order) anharmonicity correction to the second centrifugal distortion constant.

        Returns:
            float: Second (1st-order) anharmonicity constant in cm^-1

        """
        raise NotImplementedError

    @classmethod
    def D3_v(cls, v) -> float:
        """Calculate the (third) centrifugal distortion constant of a diatomic molecule in a given vibrational state.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: (third) Centrifugal distortion constant in cm^-1
        """
        return cls.D3_e - cls.alpha3_e_1 * (v + 1 / 2)

    @abstractproperty
    def D3_e(cls) -> float:
        """Calculate the (third) centrifugal distortion constant for the equilibrium internuclear separation.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Centrifugal distortian constant in cm^-1
        """
        raise NotImplementedError

    @abstractproperty
    def alpha3_e_1(cls) -> float:
        """Get the (1st-order) anharmonicity correction to the third centrifugal distortion constant.

        Returns:
            float: Third (1st-order) anharmonicity constant in cm^-1
        """
        raise NotImplementedError

    @classmethod
    def E_vib(cls, v) -> float:
        """Calculate the vibrational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Vibrational energy in cm^-1"""

        # See Derek A. Long, eq. 5.9.3
        return (v + 1 / 2) * cls.w_e - cls.w_ex_e * (v + 1 / 2) ** 2

    @abstractproperty
    def w_e(cls) -> float:
        """Harmonic vibration wavenumber (wavenumber associated with infinitely small vibrations about the equilibrium internuclear separation).

        Returns:
            float: Vibrational anharmonicity constant in cm^-1
        """
        raise NotImplementedError

    @abstractproperty
    def w_ex_e(cls) -> float:
        """Get the (1st-order) vibrational anharmonicity constant.

        Returns:
            float: Vibrational anharmonicity constant in cm^-1
        """
        raise NotImplementedError

    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:
        vi = np.arange(0, 4)
        Ji = np.arange(0, 14)[:, np.newaxis]
        dv = np.array([-1, 0, 1])[:, np.newaxis, np.newaxis]
        dJ = np.array([-2, -1, 0, 1, 2])[:, np.newaxis, np.newaxis, np.newaxis]

        endshape = (len(dJ), len(dv), len(Ji), len(vi))
        vi = vi * np.ones(endshape)
        Ji = Ji * np.ones(endshape)
        vf = (vi + dv) * np.ones(endshape)
        Jf = (Ji + dJ) * np.ones(endshape)
        vi, Ji, vf, Jf = vi.flatten(), Ji.flatten(), vf.flatten(), Jf.flatten()

        state_initial = State(v=vi, J=Ji)
        state_final = State(v=vf, J=Jf)

        return state_initial, state_final

    @classmethod
    def _format_quanta_global(cls, state: State):
        return np.char.mod("%2d", state.v)

    @classmethod
    def _format_quanta_local(cls, state: State):
        return np.char.mod("%2d", state.J)

    @classmethod
    def process_hitran_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["initial_v"] = df["initial_quanta_global"].str[0:2].astype("Int64")
        df["initial_J"] = df["initial_quanta_local"].str[0:2].astype("Int64")
        df["final_v"] = df["final_quanta_global"].str[0:2].astype("Int64")
        df["final_J"] = df["final_quanta_local"].str[0:2].astype("Int64")

        # Filter for the ground rotational state
        idx_ground_rotational_state = df["initial_J"] == 0

        # Map the ground state transitions to the ground state energy
        ground_state_transitions = pd.DataFrame(
            index=df["initial_quanta_global"][idx_ground_rotational_state].values,
            data=df["initial_E"][idx_ground_rotational_state].values,
            columns=["initial_E"],
        ).drop_duplicates()

        # For each transition, get the energy of the ground state
        idx_global_quanta = df["initial_quanta_global"]

        # Add the energy of the vibrational state
        df["initial_E_vib"] = ground_state_transitions.loc[idx_global_quanta][
            "initial_E"
        ].values
        df["initial_E_rot"] = df["initial_E"] - df["initial_E_vib"]

        df["crosssection"] = df["einstein_A_coefficient"].values

        return df

