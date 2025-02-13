from typing_extensions import override
import numpy as np
import pandas as pd
from itertools import product
from ramlab.molecules.ab_initio_molecule2 import AbInitioMolecule
from ramlab.molecules.intensity import Intensity
from ramlab.molecules.polarisation import Polarisation
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from ramlab.util.decorators import abstractproperty
from scipy.constants import k, h, hbar, c, pi, epsilon_0


class SimpleDiatomicMolecule(AbInitioMolecule):
    """Applicable to homonuclear diatomic molecules.

    Subclasses should:
    - implement polarisability_mean()
    - implement polarisability_anisotropy()

    Subclasses could:
    - override (and super-call) _get_all_transition_states() to filter

    From HitranCompatibleMolecule, they should also:
    - implement molecule_number, molecule_name, molecule_formula, isotope_number

    """

    @override
    @classmethod
    def transitions_metadata(
        cls,
        transitions: Transitions,
        laser_wavelength: float = 532e-9,
        polarisation: str = Polarisation.COMBINED,
    ) -> Transitions:
        transitions = super().transitions_metadata(
            transitions, laser_wavelength, polarisation
        )

        transitions.dJ = transitions.final_J - transitions.initial_J
        transitions.dv = transitions.final_v - transitions.initial_v

        return transitions

    # Intensity calculations
    @override
    @classmethod
    def crosssection_polarised(
        cls,
        transitions: Transitions,
        laser_wavelength=532e-9,
        polarisation=Polarisation.COMBINED,
    ):
        Polarisation.validate(polarisation)

        print("Calculating cross-section")

        # Molecule specific constants
        mu_r = 1.169e-26  # Reduced mass of nitrogen in kg, see eq 7.9 in Lucht - needs to be set for each species
        alpha0 = cls.polarisability_mean(transitions)  # Mean polarisability isotropy
        gamma0 = cls.polarisability_anisotropy(
            transitions
        )  # Mean polarisability anisotropy

        v = transitions.initial_v
        dV = transitions.dv
        nu = transitions.vacuum_wavenumber * 100  # Convert from cm^-1 to m^-1

        # Initialise values for calculation
        id_vib_Stokes = dV == 1
        id_vib_aStokes = dV == -1
        id_rot = dV == 0

        assert np.all(
            id_vib_Stokes + id_vib_aStokes + id_rot == 1
        ), "Changes with dV != 1, 0, -1 not yet implemented"

        id_Q = transitions.dJ == 0
        pt = cls.placzekteller(transitions)

        # b_v_k from Long, eq. 5.7.8
        constants = np.sqrt(h / (8 * pi**2 * c * np.abs(nu)))
        constants = 1
        # constants = hbar / (4 * (pi) * mu_r * c * np.abs(nu))
        # constants = hbar / (8 * (pi**2) * mu_r * c * np.abs(nu))

        alpha = id_Q * (  # Only Q-branches
            (id_rot * alpha0)  # Rot
            + (id_vib_Stokes * alpha0 * np.sqrt((v + 1) * constants))  # Vib Stokes
            + (id_vib_aStokes * alpha0 * np.sqrt(v * constants))  # Vib a-Stokes
        )
        # alpha = 1  # alpha_0

        gamma = (  # All branches
            (gamma0 * id_rot)  # Rot
            + (gamma0 * np.sqrt((v + 1) * constants) * id_vib_Stokes)  # Vib Stokes
            + (gamma0 * np.sqrt(v * constants) * id_vib_aStokes)  # Vib a-Stokes
        )
        # gamma = 1  # gamma_0

        return Intensity(alpha**2 + (4 / 45) * pt * gamma**2, (1 / 15) * pt * gamma**2)

    @classmethod
    def placzekteller(cls, transitions: Transitions) -> float:
        """Calculate the Placzek-Teller factor - See Long, P.183"""
        dJ = transitions.dJ
        J = transitions.initial_J
        PT = np.zeros_like(dJ, dtype=float)
        mask_Q = dJ == 0
        mask_S = dJ == 2
        mask_O = dJ == -2
        PT[mask_Q] = (J * (J + 1) / ((2 * J - 1) * (2 * J + 3)))[mask_Q]
        PT[mask_S] = ((3 * (J + 1) * (J + 2)) / (2 * (2 * J + 1) * (2 * J + 3)))[mask_S]
        PT[mask_O] = ((3 * J * (J - 1)) / (2 * (2 * J - 1) * (2 * J + 1)))[mask_O]
        return PT

    @classmethod
    def polarisability_mean(cls, transitions: Transitions) -> float:
        raise NotImplementedError()

    @classmethod
    def polarisability_anisotropy(cls, transitions: Transitions) -> float:
        raise NotImplementedError()

    # Energy calculations
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

    # Defining the possible transitions
    @classmethod
    def _get_all_transition_states(cls) -> tuple[State, State]:

        # Define initial quantum states
        vi = np.arange(0, 12)  # Vibrational quantum number
        Ji = np.arange(0, 80)  # Rotational quantum number

        # Define transitions for each quantum number
        dv = np.array([-1, 0, 1])
        dJ = np.array([-2, 0, 2])

        # Calculate the total number of transitions
        total_transitions = len(dv) * len(dJ)

        # Generate initial states
        initial_states = np.array(list(product(vi, Ji)))
        i_states_all = np.repeat(initial_states, total_transitions, axis=0)
        VI, JI = i_states_all[:, 0], i_states_all[:, 1]

        # Generate transitions for each quantum number and tile them appropriately
        dv_full = np.tile(dv, len(initial_states) * len(dJ))
        dJ_full = np.tile(np.repeat(dJ, len(dv)), len(initial_states))

        # Apply transitions
        VF = i_states_all[:, 0] + dv_full
        JF = i_states_all[:, 1] + dJ_full

        rayleigh = (VI == VF) & (JI == JF)
        legal = VF >= 0
        legal &= JF >= 0
        legal &= ~rayleigh
        vi, Ji, vf, Jf = VI[legal], JI[legal], VF[legal], JF[legal]
        state_initial = State(v=vi, J=Ji)
        state_final = State(v=vf, J=Jf)

        return state_initial, state_final

    @classmethod
    def _format_quanta_global(cls, state: State):
        return np.char.mod(" " * 13 + "%2d", state.v)

    @classmethod
    def _format_quanta_local(cls, state: State):
        return np.char.mod(" " * 6 + "%3d" + " " * 6, state.J)

    @classmethod
    @override
    def degeneracy(cls, state):
        degeneracy_nuclear = (state.J % 2) * cls.g_o + ((state.J + 1) % 2) * cls.g_e
        degeneracy_rotational = 2 * state.J + 1
        return degeneracy_nuclear * degeneracy_rotational

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
