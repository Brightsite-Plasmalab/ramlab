from typing_extensions import override
import numpy as np
from ramlab.molecules.diatomic import SimpleDiatomicMolecule
from ramlab.molecules.transitions import Transitions


class O2(SimpleDiatomicMolecule):
    # Molecule properties
    molecule_number = 7
    molecule_name = "O2"
    isotope_number = 1

    # Degeneracy constants
    g_e = 0  # nuclear degeneracy for even J
    g_o = 1  # nuclear degeneracy for odd J

    # Energy constants
    B_e = 1.44563  # /m
    w_e = 1580.193  # /m
    # w_ex_e = 1.16986e02  # /cm
    # alpha0_e_1 = 2.94099e00  # /cm
    # D1_e = 4.66150e-02  # /cm
    # alpha1_e_1 = 2.37251e-03  # /cm
    # D2_e = 5.08134e-05  # /cm
    # alpha2_e_1 = 1.35508e-05  # /cm
    # D3_e = 0
    # alpha3_e_1 = 0

    @classmethod
    @override
    def polarisability_mean(cls, transitions: Transitions) -> np.ndarray:
        # Straight from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7
        dV = np.abs(transitions.dv)

        alpha = np.zeros_like(dV, dtype=np.float64)
        v_l = np.minimum(transitions.initial_v, transitions.final_v)

        alpha_dv0 = 1.619 + 0.01773 * v_l + 0.00009 * v_l**2
        alpha_dv1 = (
            ((v_l + 1) / 2) ** (1 / 2)
            * (2 * cls.B_e / cls.w_e) ** (1 / 2)
            * (1.779 + 0.019 * v_l)
        )

        alpha[dV == 0] = alpha_dv0[dV == 0]
        alpha[dV == 1] = alpha_dv1[dV == 1]

        # [A^3] = [10^-30 m^3] = [10^-24 cm^3], atomig oxygen has a total cross-section of 1e-31 cm2/sr
        # Contribution of alpha to total cross-section ~ 1e-48 cm^6
        # [w0 = 19 cm-1] -> [130 cm^-1]
        # [16pi^4 = 1558]
        # See Long eq 4.13.3 and Long eq. 5.10.3
        # Question: what's included in the reduced polarizability tensor?

        return alpha

    @classmethod
    @override
    def polarisability_anisotropy(cls, transitions):
        # Straight from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7
        dV = np.abs(transitions.dv)

        gamma = np.zeros_like(dV, dtype=np.float64)
        v_l = np.minimum(transitions.initial_v, transitions.final_v)

        gamma_dv0 = 1.097 + 0.0339 * v_l + 0.00040 * v_l**2
        gamma_dv1 = (
            ((v_l + 1) / 2) ** (1 / 2)
            * (2 * cls.B_e / cls.w_e) ** (1 / 2)
            * (3.25 + 0.057 * v_l)
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

        # See Derek A. Long, eq. 5.9.3
        return (-1) ** (2) * 1556.35820196816 * (v + 1 / 2) ** 1

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

        # NB: Here we deviate from Long's notation. Long uses B, D, H, while we use B, D_1, D_2, D_3
        return (
            1.445587560355248 * (v + 1 / 2) ** 0
            + -0.015694718136882425 * (v + 1 / 2) ** 1
        ) * (J * (J + 1)) ** 1 + (
            -4.815829641535194e-06 * (v + 1 / 2) ** 0
            + -8.276084200953973e-08 * (v + 1 / 2) ** 1
        ) * (
            J * (J + 1)
        ) ** 2


if __name__ == "__main__":
    print(O2.get_all_transitions())
