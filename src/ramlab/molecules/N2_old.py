import numpy as np

from ramlab.molecules.diatomic import SimpleDiatomicMolecule


class N2(SimpleDiatomicMolecule):
    # The implementation of this class is based on Derek A. Long, The Raman Effect.
    # Polarizibility matrix elements are taken from Buldakov (2003), https://doi.org/10.1016/S0022-2852(02)00012-7

    g_e = 6
    g_o = 3
    # g_p = 7.91
    # D_v = 5.76e-6 / 1e-2
    # H_v = 0
    # w_e = 0
    # w_ex_e = 0
    # a_e = 0.017 / 1e-2

    # Fitted using transition_n2.ipynb (using Hitran line-by-line database)
    w_e = 2.35862e03 / 1e-2  # /m
    w_ex_e = 1.43444e01 / 1e-2  # /m
    B_e = 1.99829e00 / 1e-2  # /m
    a_e = 1.74130e-02 / 1e-2  # /m
    D_e = 5.49723e-06 / 1e-2  # /m
    b_e = -7.67491e-08 / 1e-2  # /m
    H_e = -3.02983e-10 / 1e-2  # /m
    c_e = -9.16227e-11 / 1e-2  # /m
    J_e = 0
    d_e = 0

    @classmethod
    def polarizibility_mean(cls, vi, Ji, vf, Jf, lambda_laser=532e-9) -> float:
        # Determine lower (xp) and upper (xpp) energy levels and quantum numbers
        Ei = cls.E(vi, Ji)
        Ef = cls.E(vf, Jf)
        fi_pp = Ef > Ei

        vp = vf * fi_pp + vi * ~fi_pp
        Jp = Jf * fi_pp + Ji * ~fi_pp

        vpp = vf * ~fi_pp + vi * fi_pp
        Jpp = Jf * ~fi_pp + Ji * fi_pp

        # Find the locations matching O, Q, S branches
        dv = vp - vpp
        dJ = Jp - Jpp

        # The polarizibility mean only plays a role for the Q branch
        idQ = dJ == 0

        # Overtones are listed in Buldakov (2003), but not taken into account here.

        dv0 = 1.777 + 0.01389 * vp * 0.000098 * vp**2
        dv1 = (
            ((vp + 1) / 2) ** (1 / 2)
            * (cls.B_e / cls.w_e) ** (1 / 2)
            * (1.871 + 0.0105 * vp)
        )
        M = (dv == 0) * dv0 + (dv == 1) * dv1

        F = cls.HermanWallis_a(vp, Jp, vpp, Jpp)

        return np.sqrt(F) * M

    @classmethod
    def polarizibility_anisotropy(cls, vi, Ji, vf, Jf, lambda_laser=532e-9) -> float:
        # Determine lower (xp) and upper (xpp) energy levels and quantum numbers
        Ei = cls.E(vi, Ji)
        Ef = cls.E(vf, Jf)
        fi_pp = Ef > Ei

        vp = vf * fi_pp + vi * ~fi_pp
        Jp = Jf * fi_pp + Ji * ~fi_pp

        vpp = vf * ~fi_pp + vi * fi_pp
        Jpp = Jf * ~fi_pp + Ji * fi_pp

        # Find the locations matching O, Q, S branches
        dv = vf - vi

        # The polarizibility mean only plays a role for the Q branch, so make sure the rest is nan

        # Overtones are listed in Buldakov (2003), but not taken into account here.

        # Variables M are the matrix elements <vp|a,y|vpp>
        M0 = 0.719 + 0.0177 * vp * 0.00015 * vp**2
        M1 = (
            ((vp + 1) / 2) ** (1 / 2)
            * (cls.B_e / cls.w_e) ** (1 / 2)
            * (2.25 + 0.019 * vp)
        )
        M = (dv == 0) * M0 + (dv == 1) * M1

        F = cls.HermanWallis_y(vp, Jp, vpp, Jpp)

        return np.sqrt(F) * M

    @classmethod
    def HermanWallis_y(cls, vp, Jp, vpp, Jpp):
        # Find the locations matching O, Q, S branches
        dv = vp - vpp
        dJ = Jp - Jpp
        idO = dJ == -2
        idQ = dJ == 0
        idS = dJ == 2

        m = idQ * (Jp * (Jp + 1)) + idO * (-2 * Jp + 1) + idS * (2 * Jp + 3)

        FnQ0 = 1 + 1.35e-5 + 0.03e-6 * m + (4.50e-6 - 0.10e-7 * vp) * m**2
        FQ0 = 1 + (1.81e-5 + 0.04e-6 * vp) * m

        FnQ1 = (
            (1 + 1.04e-5 - 0.09e-6 * vp)
            - (2.20e-3 + 0.37e-4 * vp) * m
            + (0.47e-5 + 0.11e-7 * vp) * m**2
        )
        FQ1 = 1 + (0.14e-4 - 0.12e-6 * vp) * m

        F0 = (idQ) * FQ0 + (~idQ) * FnQ0
        F1 = (idQ) * FQ1 + (~idQ) * FnQ1

        F = (dv == 0) * F0 + (dv == 1) * F1

        return F

    @classmethod
    def HermanWallis_a(cls, vp, Jp, vpp, Jpp):
        # Find the locations matching O, Q, S branches
        dv = vp - vpp
        dJ = Jp - Jpp
        idO = dJ == -2
        idQ = dJ == 0
        idS = dJ == 2

        m = idQ * (Jp * (Jp + 1)) + idO * (-2 * Jp + 1) + idS * (2 * Jp + 3)

        F0 = 1 + (6.08e-6 + 0.86e-7 * vp) * m
        F1 = 1 + (1.10e-5 - 0.61e-7 * vp) * m
        F = (dv == 0) * F0 + (dv == 1) * F1

        return F
