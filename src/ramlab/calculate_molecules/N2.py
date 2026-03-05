from typing_extensions import override
import numpy as np
from ramlab.calculate_molecules.diatomic import SimpleDiatomicMolecule
from ramlab.state.transitions import Transitions


class N2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 22
    molecule_name = "N2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 6  # nuclear degeneracy for even J
    g_o = 3  # nuclear degeneracy for odd J

    # Energy constants
    B_e = 1.99829e00  # /m
    w_e = 2.35862e03  # /m
    w_ex_e = 1.43444e01  # /m
    alpha0_e_1 = 1.74130e-02  # /m
    D1_e = 5.49723e-06  # /m
    alpha1_e_1 = -7.67491e-08  # /m
    D2_e = -3.02983e-10  # /m
    alpha2_e_1 = -9.16227e-11  # /m
    D3_e = 0
    alpha3_e_1 = 0

    @classmethod
    @override
    def polarisability_mean(cls, transitions: Transitions) -> np.ndarray:
        # Straight from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7
        dV = np.abs(transitions.dv)

        alpha = np.zeros_like(dV, dtype=np.float64)
        v_l = np.minimum(transitions.initial_v, transitions.final_v)

        alpha_dv0 = 1.777 + 0.01389 * v_l + 0.000098 * v_l**2
        alpha_dv1 = (
            ((v_l + 1) / 2) ** (1 / 2)
            * (2 * cls.B_e / cls.w_e) ** (1 / 2)
            * (1.871 + 0.0105 * v_l)
        )

        alpha[dV == 0] = alpha_dv0[dV == 0]
        alpha[dV == 1] = alpha_dv1[dV == 1]

        return alpha

    @classmethod
    @override
    def polarisability_anisotropy(cls, transitions):
        # Straight from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7
        dV = np.abs(transitions.dv)

        gamma = np.zeros_like(dV, dtype=np.float64)
        v_l = np.minimum(transitions.initial_v, transitions.final_v)

        gamma_dv0 = 0.719 + 0.0177 * v_l + 0.00015 * v_l**2
        gamma_dv1 = (
            ((v_l + 1) / 2) ** (1 / 2)
            * (2 * cls.B_e / cls.w_e) ** (1 / 2)
            * (2.25 + 0.019 * v_l)
        )

        gamma[dV == 0] = gamma_dv0[dV == 0]
        gamma[dV == 1] = gamma_dv1[dV == 1]

        return gamma

    @override
    @classmethod
    def E_vib(cls, v) -> float:
        """Calculate the vibrational energy of a diatomic molecule.

        Args:
            v (int): vibrational quantum number

        Returns:
            float: Vibrational energy in cm^-1"""

        return (-1) ** (2) * 2358.565029936973 * (v + 1 / 2) ** 1 + (-1) ** (
            3
        ) * 14.327781128145256 * (v + 1 / 2) ** 2

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
            1.9982310472466613 * (v + 1 / 2) ** 0
            + -0.01730063137226103 * (v + 1 / 2) ** 1
            + -2.9664008924682414e-05 * (v + 1 / 2) ** 2
        ) * (J * (J + 1)) ** 1 + (
            -5.714693409383646e-06 * (v + 1 / 2) ** 0
            + -1.6900332756085206e-08 * (v + 1 / 2) ** 1
            + -1.0240627129671262e-10 * (v + 1 / 2) ** 2
        ) * (
            J * (J + 1)
        ) ** 2
