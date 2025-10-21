from typing_extensions import override
import numpy as np
from ramlab.molecules.diatomic import SimpleDiatomicMolecule
from ramlab.molecules.state import State
from ramlab.molecules.transitions import Transitions
from scipy.constants import k, h, hbar, c, pi, epsilon_0, fine_structure


class N2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 22
    molecule_name = "N2"
    isotope_number = None

    # Energy constants
    B_e = None  # /m
    w_e = None  # /m

    # Store energies and degeneracies and degeneracies of states
    # which could have non-negligible populations at certain temperature range
    # ToDo: Should it relate to an instance instead of the class?
    selected_E_vib = None
    selected_E_rot = None
    selected_degeneracy = None
    
    @override
    @classmethod
    def _get_all_states(cls):
        return State(J=np.arange(0, 200)).add_each(v=np.arange(0, 20))
    
    @classmethod
    @override
    def select_states(cls, maxT):
        state = cls._get_all_states()
        E_vib = cls.E_vib(state.v) * 100  # 100*E converts E from cm^-1 to m^-1
        E_rot = cls.E_rot(state.v, state.J) * 100
        g = cls.degeneracy(state)
        mask = False
        for T in range(300, maxT+499, 500):
            p = g * np.exp(-h * c * (E_vib + E_rot) / (k * T))
            p /= np.max(p)
            mask = np.logical_or(mask, p>1e-5)
        
        # Additional states that could be populated in case Tvib > Trot
        for Trot in range(300, maxT+499, 500):
            p = g * np.exp(-h * c * E_rot / (k * Trot)) * np.exp(-h * c * E_vib / (k * maxT))
            p /= np.max(p)
            mask = np.logical_or(mask, p>1e-5)
        
        cls.selected_E_vib = E_vib[mask]
        cls.selected_E_rot = E_rot[mask]
        cls.selected_degeneracy = g[mask]

    @override
    @classmethod
    def get_partition_sum(cls, **temperatures) -> float:
        """Returns the partition sum of the molecule using precalculated energies if possible.

        Args:
            **temperatures: The temperatures in Kelvin.

        Returns:
            float: The partition sum of the molecule.
        """
        if (cls.selected_E_vib is None) or (cls.selected_E_rot is None) or (cls.selected_degeneracy is None):
            return super().get_partition_sum(**temperatures)
        
        if len(temperatures) <= 1:
            # Assume populations are described by a Boltzmann distribution if only one temperature is given
            weights = cls.selected_degeneracy * np.exp(-h * c * (cls.selected_E_vib + cls.selected_E_rot) / (k * temperatures["T"]))
        else:
            T_vib = temperatures["T_vib"]
            T_rot = temperatures["T_rot"]
            weights = (
                cls.selected_degeneracy
                * np.exp(-h * c * cls.selected_E_vib / (k * T_vib))
                * np.exp(-h * c * cls.selected_E_rot / (k * T_rot))
            )
        return np.nansum(weights)


    @classmethod
    @override
    def polarisability_mean(cls, transitions: Transitions) -> np.ndarray:
        # Straight from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7
        v = transitions.initial_v
        dv = np.abs(transitions.dv)
        J = transitions.initial_J
        dJ = transitions.dJ

        alpha = np.zeros_like(dv, dtype=np.float64)
        v_l = np.minimum(transitions.initial_v, transitions.final_v)
        
        # Herman–Wallis factors
        mQ = J*(J+1)
        FalphaQ_dv0 = 1 + (6.08e-6 + 0.86e-7*v) * mQ
        FalphaQ_dv1 = 1 + (1.10e-5 - 0.61e-7*v) * mQ
        
        alpha_dv0 = 1.777 + 0.01389 * v_l + 0.000098 * v_l**2
        alpha_dv1 = (
            ((v_l + 1) / 2) ** (1 / 2)
            * (2 * cls.B_e / cls.w_e) ** (1 / 2)
            * (1.871 + 0.0105 * v_l)
        )

        alpha[np.logical_and((dv == 0), (dJ == 0))] = np.sqrt(FalphaQ_dv0[np.logical_and((dv == 0), (dJ == 0))]) * alpha_dv0[np.logical_and((dv == 0), (dJ == 0))]
        alpha[np.logical_and((dv == 1), (dJ == 0))] = np.sqrt(FalphaQ_dv1[np.logical_and((dv == 1), (dJ == 0))]) * alpha_dv1[np.logical_and((dv == 1), (dJ == 0))]

        return alpha

    @classmethod
    @override
    def polarisability_anisotropy(cls, transitions):
        # Straight from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7
        v = transitions.initial_v
        dv = np.abs(transitions.dv)
        J = transitions.initial_J
        dJ = transitions.dJ

        gamma = np.zeros_like(dv, dtype=np.float64)
        v_l = np.minimum(transitions.initial_v, transitions.final_v)

        gamma_dv0 = 0.719 + 0.0177 * v_l + 0.00015 * v_l**2
        gamma_dv1 = (
            ((v_l + 1) / 2) ** (1 / 2)
            * (2 * cls.B_e / cls.w_e) ** (1 / 2)
            * (2.25 + 0.019 * v_l)
        )
		
        # Herman–Wallis factors
        mS = 2*J + 3
        mO = -2*J + 1
        mQ = J*(J+1)
        
        FgammaS_dv0 = 1.0000135 + 0.03e-6*v + (4.50e-6 - 0.10e-7*v) * mS**2
        FgammaS_dv1 = 1.0000104 - 0.09e-6*v - (2.20e-3 + 0.37e-4*v) * mS + (0.47e-5 + 0.11e-7*v) * mS**2
        FgammaO_dv0 = 1.0000135 + 0.03e-6*v + (4.50e-6 - 0.10e-7*v) * mO**2
        FgammaO_dv1 = 1.0000104 - 0.09e-6*v - (2.20e-3 + 0.37e-4*v) * mO + (0.47e-5 + 0.11e-7*v) * mO**2

        FgammaQ_dv0 = 1 + (1.81e-5 + 0.04e-6*v) * mQ
        FgammaQ_dv1 = 1 + (0.14e-4 - 0.12e-6*v) * mQ


        gamma[dv == 0] = gamma_dv0[dv == 0]
        gamma[np.logical_and((dv == 0), (dJ == -2))] *= np.sqrt(FgammaO_dv0[np.logical_and((dv == 0), (dJ == -2))])
        gamma[np.logical_and((dv == 0), (dJ ==  0))] *= np.sqrt(FgammaQ_dv0[np.logical_and((dv == 0), (dJ ==  0))])
        gamma[np.logical_and((dv == 0), (dJ ==  2))] *= np.sqrt(FgammaS_dv0[np.logical_and((dv == 0), (dJ ==  2))])
        gamma[dv == 1] = gamma_dv1[dv == 1]
        gamma[np.logical_and((dv == 1), (dJ == -2))] *= np.sqrt(FgammaO_dv1[np.logical_and((dv == 1), (dJ == -2))])
        gamma[np.logical_and((dv == 1), (dJ ==  0))] *= np.sqrt(FgammaQ_dv1[np.logical_and((dv == 1), (dJ ==  0))])
        gamma[np.logical_and((dv == 1), (dJ ==  2))] *= np.sqrt(FgammaS_dv1[np.logical_and((dv == 1), (dJ ==  2))])
        return gamma

    @override
    @classmethod
    def E_vib(cls, v) -> float:
        """Calculate the vibrational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Vibrational energy in cm^-1"""

        raise NotImplementedError()

    @override
    @classmethod
    def E_rot(cls, v, J) -> float:
        raise NotImplementedError()


class N2_28(N2):
    # Molecule properties
    isotope_number = 1

    # Degeneracy constants
    g_e = 6  # nuclear degeneracy for even J
    g_o = 3  # nuclear degeneracy for odd J

    # Energy constants
    B_e = 1.998259  # /m
    w_e = 2358.53373  # /m

    @override
    @classmethod
    def E_vib(cls, v) -> float:
        """Calculate the vibrational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Vibrational energy in cm^-1"""

        return (2358.53373 * (v + 0.5)**1 +
				 -14.30056 * (v + 0.5)**2 +
				 -65.22e-4 * (v + 0.5)**3 + 
			       0.60e-4 * (v + 0.5)**4 +
			      -0.0715e-4 * (v + 0.5)**5)

    @override
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

        return (
            1.998259 * (v + 0.5) ** 0
            + -0.0173243 * (v + 0.5) ** 1
            + -2.27e-5 * (v + 0.5) ** 2
			+ -0.053e-5 * (v + 0.5) ** 3
        ) * (J * (J + 1)) ** 1 + (
            -5.75e-06 * (v + 0.5) ** 0
            + -0.98e-08 * (v + 0.5) ** 1
        ) * (J * (J + 1)) ** 2
			

class N2_29(N2):
    # Molecule properties
    isotope_number = 2

    # Degeneracy constants
    # s(N14) = 1; s(N15) = 1/2; s(N2_29) = 3/2 or 1/2
    g_e = 6  # nuclear degeneracy for even J
    g_o = 6  # nuclear degeneracy for odd J

    # Energy constants
    B_e = 1.9318484  # /m
    w_e = 2319.00790  # /m

    @override
    @classmethod
    def E_vib(cls, v) -> float:
        """Calculate the vibrational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Vibrational energy in cm^-1"""

        return (2319.00790 * (v + 0.5)**1 +
				 -13.825105 * (v + 0.5)**2 +
				 -61.996e-4 * (v + 0.5)**3 + 
			       0.5608e-4 * (v + 0.5)**4 +
			      -0.06571e-4 * (v + 0.5)**5)
    
    @override
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

        return (
            1.9318484 * (v + 0.5) ** 0
            + -0.01646788 * (v + 0.5) ** 1
            + -2.122e-5 * (v + 0.5) ** 2
			+ -0.0487e-5 * (v + 0.5) ** 3
        ) * (J * (J + 1)) ** 1 + (
              -5.3742e-06 * (v + 0.5) ** 0
            + -0.901e-08 * (v + 0.5) ** 1
        ) * (J * (J + 1)) ** 2