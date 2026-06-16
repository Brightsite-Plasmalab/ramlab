from sympy.physics import wigner
import numpy as np
from ramlab.math import make_equal_size, wheren

WARNING_PY3NJ = True


def placzek_teller():
    raise NotImplementedError("This function is not implemented yet.")


def wigner_3j_py3nj(j1, j2, j3, m1, m2, m3):
    import py3nj

    # For py3nj's wigner3j, the arguments must be multiplied by 2
    j1, j2, j3, m1, m2, m3 = [
        (2 * x).astype(np.int16) for x in [j1, j2, j3, m1, m2, m3]
    ]

    id_invalid = (
        (np.abs(m1) > j1)
        | (np.abs(m2) > j2)
        | (np.abs(m3) > j3)
        | (j1 < 0)
        | (j2 < 0)
        | (j3 < 0)
    )
    j1[id_invalid] = 0
    j2[id_invalid] = 0
    j3[id_invalid] = 0
    m1[id_invalid] = 0
    m2[id_invalid] = 0
    m3[id_invalid] = 0

    W = py3nj.wigner3j(j1, j2, j3, m1, m2, m3)
    W[id_invalid] = 0

    return W


def wigner_3j_sympy(j1, j2, j3, m1, m2, m3):
    W = np.zeros_like(j1, dtype=np.float32)
    for i in range(W.size):
        idx = np.unravel_index(i, W.shape)
        W[idx] = wigner.wigner_3j(j1[idx], j2[idx], j3[idx], m1[idx], m2[idx], m3[idx])

    return W


def wigner_3j(j1, j2, j3, m1, m2, m3):
    # Assume j1, j2, j3, m1, m2, m3 are numpy arrays of size N, OR scalars.
    # Loop over the arrays and calculate the Wigner 3j symbol for each set of values.
    # Return the result as a numpy array of size N.

    # Cast scalars to numpy arrays of the same size
    j1, j2, j3, m1, m2, m3 = [
        np.atleast_1d(x) for x in np.broadcast_arrays(j1, j2, j3, m1, m2, m3)
    ]

    global WARNING_PY3NJ
    try:
        W = wigner_3j_py3nj(j1, j2, j3, m1, m2, m3)
    except ImportError as e:
        if WARNING_PY3NJ:
            print(
                "py3nj is not installed, falling back to the (>100x slower) sympy implementation."
            )
            WARNING_PY3NJ = False
        W = wigner_3j_sympy(j1, j2, j3, m1, m2, m3)

    return W


def akp_sq(k, p, JA, JB, OmegaA, OmegaB):  # Eq. 14 from Satija
    """
    Implementation of <|a^k_p|^2>, excluding the term |<VA|a^k_p|VB>|^2. See Eq.14 from Satija (2020) for details, Robert Lucht's Book.
    In cases with perturbed states, such as NO (see eq. 14), this result should be weighted accordingly.
    """
    q = OmegaB - OmegaA
    wig_omega = wigner_3j(JA, JB, k, OmegaA, -OmegaB, q)

    wig_magnetic_sum_of_squares = np.zeros(np.size(JA))
    for i in range(np.size(JA)):
        Ma = np.arange(-JA[i], JA[i] + 1, 1)[:, np.newaxis]
        Mb = np.arange(-JB[i], JB[i] + 1, 1)[:, np.newaxis, np.newaxis]
        wig_magnetic = wigner_3j(JA[i], JB[i], k, Ma, -Mb, p)  # replace with p
        wig_magnetic_sum_of_squares[i] = np.sum(wig_magnetic**2)

    akp = (2 * JB + 1) * wig_omega**2 * wig_magnetic_sum_of_squares
    return akp


def Dkp(k, p, JA, JB, OmegaA, OmegaB, Ma, Mb):  # Eq. 14 from Satija
    """ """
    q = OmegaB - OmegaA

    return (
        (-1) ** (p - q)
        * (-1) ** (Ma - OmegaA)
        * np.sqrt((2 * JA + 1) * (2 * JB + 1))
        * wigner_3j(JA, JB, k, Ma, -Mb, p)
        * wigner_3j(JA, JB, k, OmegaA, -OmegaB, q)
    ), q


def akp2(k, p, JA, JB, OmegaA, OmegaB):
    akp2_av_i = np.zeros(np.size(JA))
    q = np.zeros_like(akp2_av_i)
    akq = 1  # TODO: Make value dependent on q

    for i in range(np.size(JA)):
        Ma, Mb = get_valid_magnetic_quantum_numbers(JA[i], JB[i], p)

        Dkp_i_components, q[i] = Dkp(k, p, JA[i], JB[i], OmegaA[i], OmegaB[i], Ma, Mb)

        akp2_av_i[i] = (akq**2) * np.sum(Dkp_i_components**2) / (2 * JA[i] + 1)

    return akp2_av_i

# Where is it from?
# Shouldn't it be independent of p as suggested by equation 23 from Satija?
# For k=2, p= +/-2, there is an error: "ValueError: zero-size array to reduction operation maximum which has no identity"
# 7.70 Lucht or 19 in Satija?
def akp2_perturbed2(k, p, JA, JB, perturbationA, perturbationB):
    """
    akp2, except we assume that each state is a perturbed state, and we weight the result accordingly.
    the arguments (perturbationA) and (perturbationA) should contain coefficients ((c1i, OmegaA1i), (c2i, OmegaA2i), ...)
    """
    akp2_av_i = np.zeros(np.size(JA))

    # Is it components of the irreducible polarizability tensor elements in molecule-fixed frame?
    # Shouldn't these values be molecule-specific?
    akq_values = np.array([1, 0, 0])  # akq for q = 0, 1, 2

    Na = len(perturbationA)
    Nb = len(perturbationB)

    # Loop over all transitions
    for l in range(np.size(JA)):
        # Find valid magnetic quantum numbers
        Ma, Mb = get_valid_magnetic_quantum_numbers(JA[l], JB[l], p)

        akp_l = np.zeros((Na, Nb, np.size(Ma)))

        # Loop over all combinations of OmegaAi and OmegaBj
        for i, (cAi, OmegaA) in enumerate(perturbationA):
            for j, (cBj, OmegaB) in enumerate(perturbationB):
                Dkp_lij, qij = Dkp(k, p, JA[l], JB[l], OmegaA[l], OmegaB[l], Ma, Mb)

                # Find the value of <a^k_q> corresponding to this transition
                assert np.isclose(qij % 1, 0), f"qij must be an integer, but is {qij} for JA={JA[l]}, JB={JB[l]}, OmegaA={OmegaA[l]}, OmegaB={OmegaB[l]}"

                qij = int(np.abs(qij))
                if qij >= len(akq_values):
                    continue

                akq_lij = akq_values[qij]
                akp_l[i, j] = cAi[l] * cBj[l] * akq_lij * Dkp_lij

        # NB: Latest addition is to comment this out. I think this is not what Lucht's book does, but it seems to work the best so far.
        # Perhaps better to follow Long.
        # akp_l = akp_l.sum(axis=(0, 1))
        akp2_av_i[l] = np.sum(akp_l**2) / (2 * JA[l] + 1)

    return akp2_av_i


def get_valid_magnetic_quantum_numbers(JA, JB, p):
    Ma = np.arange(-JA, JA + 1, 1)[:, np.newaxis]
    Mb = np.arange(-JB, JB + 1, 1)[:, np.newaxis, np.newaxis]
    Ma, Mb = np.broadcast_arrays(Ma, Mb)

    # Check for which values Ma - Mb + p = 0
    mask = Ma - Mb + p == 0

    # Flatten Ma and Mb with the valid values
    Ma = Ma[mask]
    Mb = Mb[mask]

    return Ma, Mb


def branchname(d):
    # N = -3
    # O = -2
    # P = -1
    # Q = 0
    # R = 1
    # S = 2
    # T = 3
    assert isinstance(d, int), "d must be an integer."
    assert d >= -3 and d <= 3, "d must be between -3 and 3."

    return ["N", "O", "P", "Q", "R", "S", "T"][d + 3]
