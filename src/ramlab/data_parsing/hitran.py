from typing import Collection

from .reader_generator import HitranSchemaEntry, add_subschema
from .copy_on_read import CopyAttributes


def full_schema_dict(
    globals_: dict[str, HitranSchemaEntry] | str,
    locals_: dict[str, HitranSchemaEntry] | str,
    general_schema: dict[str, HitranSchemaEntry] | str = 'hitran',
    global_quanta_i: dict[str, HitranSchemaEntry] | str = None,
    local_quanta_i: dict[str, HitranSchemaEntry] | str = None,
    entries: Collection[str] = None):
    # TODO: Add docstring

    if general_schema == 'hitran':
        general_schema = hitran_schema_dict()
    elif general_schema == 'ramlab':
        general_schema = ramlab_schema_dict()

    def set_value(value, quanta_type, startswith, endswith):
        if isinstance(value, str):
            if not value.startswith(startswith):
                value = startswith + value
            if not value.endswith(endswith):
                value += endswith

            try:
                return getattr(quanta_type, value)
            except AttributeError:
                msg = (f"Invalid quanta_type string: {value} for {quanta_type.__name__}."
                       f" Must be one of {', '.join((x for x in dir(quanta_type) if not x.startswith('_')))}")
                raise ValueError(msg)
        return value

    global_quanta_f = set_value(globals_, GlobalQuanta, "class", "_f")
    if global_quanta_i is None:
        global_quanta_i = globals_
    global_quanta_i = set_value(global_quanta_i, GlobalQuanta, "class",  "_i")

    local_quanta_f = set_value(locals_, LocalQuanta, "group", "_f")
    if local_quanta_i is None:
        local_quanta_i = locals_
    local_quanta_i = set_value(local_quanta_i, LocalQuanta, "group", "_i")

    schema = add_subschema(
        general_schema,
        global_quanta_f = global_quanta_f,
        global_quanta_i = global_quanta_i,
        local_quanta_f = local_quanta_f,
        local_quanta_i = local_quanta_i,
    )
    if entries is not None:
        schema = {key: schema[key] for key in entries}
    return schema


def hitran_schema_dict():
    """
    Returns a dictionary of Hitran schema entries.

    Notes
    -----
    The schema is:
        - molec_id: digit of length 2
        - iso_id: digit of length 1
        - nu: float, wavenumber, length 12, 6 decimals
        - sw: float, line intensity, length 10, scientific notation
        - a: float, Einstein A coefficient, length 10, scientific notation
        - gamma_air: float, air-broadened half-width, length 5, 4 decimals
        - gamma_self: float, self-broadened half-width, length 5, 3 decimals
        - E_i: float, initial state energy, length 10, 4 decimals
        - n_air: float, temperature exponent, length 4, 2 decimals
        - delta_air: float, pressure shift, length 8, 6 decimals
        - global_quanta_f: string, final state global quanta, length 15
        - global_quanta_i: string, initial state global quanta, length 15
        - local_quanta_f: string, final state local quanta, length 15
        - local_quanta_i: string, initial state local quanta, length 15
        - ierr: integer, error code, length 6
        - iref: string, reference, length 12
        - mixing: string, line mixing flag, length 1
        - g_f: float, final state statistical weight, length 7, 1 decimal
        - g_i: float, initial state statistical weight, length 7, 1 decimal
    """
    return {
        'molec_id': HitranSchemaEntry(start=0, fmt='2d', explanation="Molecular species identification (ID) number"),
        'iso_id': HitranSchemaEntry(start=2, fmt='1d', explanation="Isotopologue ID number"),
        'nu': HitranSchemaEntry(start=3, fmt='12.6f',
                                explanation="Wavenumber of the spectral line transition in vacuum [$\"cm\"^(-1)$]"),
        'sw': HitranSchemaEntry(start=15, fmt='10.3e',
                                explanation="The spectral line intensity [$\"cm\"^(-1)\\/(\"molecule\" \"cm\"^(-2))$]"),
        'a': HitranSchemaEntry(start=25, fmt='10.3e', explanation="Einstein-A coefficient [$s^(-1)$]"),
        'gamma_air': HitranSchemaEntry(start=35, fmt='5.4f',
                                       explanation=("Air-broadened half width at half maximum (HWHM) "
                                                    "[$\"cm\"^(-1)\\/\"atm\"$]")),
        'gamma_self': HitranSchemaEntry(start=40, fmt='5.3f',
                                        explanation=("Self-broadened half width at half maximum (HWHM) "
                                                     "[$\"cm\"^(-1)\\/\"atm\"$]")),
        'E_i': HitranSchemaEntry(start=45, fmt='10.4f',
                                 explanation="Initial state energy [$\"cm\"^(-1)$]"),
        'n_air': HitranSchemaEntry(start=55, fmt='4.2f',
                               explanation="Coefficient of the temperature dependence of the air-broadened half width"),
        'delta_air': HitranSchemaEntry(start=59, fmt='8.6f',
                                       explanation=("Pressure shift of the line position with respect to the vacuum "
                                                    "transition wavenumber [$\"cm\"^(-1)\\/\"atm\"$]")),
        'global_quanta_f': HitranSchemaEntry(start=67, fmt='15s', explanation="Species specific quantum numbers for final state"),
        'global_quanta_i': HitranSchemaEntry(start=82, fmt='15s', explanation="Species specific quantum numbers for initial state"),
        'local_quanta_f': HitranSchemaEntry(start=97, fmt='15s', explanation="Species specific quantum numbers for final state"),
        'local_quanta_i': HitranSchemaEntry(start=112, fmt='15s', explanation="Species specific quantum numbers for initial state"),
        'ierr': HitranSchemaEntry(start=127, fmt='6d', explanation="Uncertainty codes"),
        'iref': HitranSchemaEntry(start=133, fmt='12s', explanation="Reference codes"),
        'mixing': HitranSchemaEntry(start=145, fmt='1s', explanation="Line mixing flag"),
        'g_f': HitranSchemaEntry(start=146, fmt='7.1f', explanation="Final state statistical weight"),
        'g_i': HitranSchemaEntry(start=153, fmt='7.1f', explanation="Initial state statistical weight")
    }


def ramlab_schema_dict():
    """
    Returns a dictionary of ramlab schema entries.

    Which is a compressed version of ``hitran_schema_dict()`` and it includes the depolarization ratio `rho` instead of
    the Einstein factor `a`

    Notes
    -----
    The schema is:
        - molec_id: digit of length 2
        - iso_id: digit of length 1
        - nu: float, wavenumber, length 12, 6 decimals
        - sw: float, line intensity, length 10, scientific notation
        - rho: float, depolarisation ratio, length 10, 8 decimals
        - E_i: float, initial state energy, length 10, 4 decimals
        - global_quanta_f: string, final state global quanta, length 15
        - global_quanta_i: string, initial state global quanta, length 15
        - local_quanta_f: string, final state local quanta, length 15
        - local_quanta_i: string, initial state local quanta, length 15
        - g_f: float, final state statistical weight, length 7, 1 decimal
        - g_i: float, initial state statistical weight, length 7, 1 decimal

    """
    return {
        'molec_id': HitranSchemaEntry(start=0, fmt='2d'),
        'iso_id': HitranSchemaEntry(start=2, fmt='1d'),
        'nu': HitranSchemaEntry(start=3, fmt='12.6f' ),
        'sw': HitranSchemaEntry(start=15, fmt='10.3e' ),
        'rho': HitranSchemaEntry(start=25, fmt='10.8f' ),
        'E_i': HitranSchemaEntry(start=35, fmt='10.4f' ),
        'global_quanta_f': HitranSchemaEntry(start=45, fmt='15s' ),
        'global_quanta_i': HitranSchemaEntry(start=60, fmt='15s' ),
        'local_quanta_f': HitranSchemaEntry(start=75, fmt='15s' ),
        'local_quanta_i': HitranSchemaEntry(start=90, fmt='15s' ),
        'g_f': HitranSchemaEntry(start=105, fmt='7.1f' ),
        'g_i': HitranSchemaEntry(start=112, fmt='7.1f' )
    }

def _to_final_initial(data: dict[str, HitranSchemaEntry], value: str):
    new_data = {}
    for key, entry in data.items():
        key += value

        if entry.explanation is None:
            explanation = None
        else:
            prepend = "Initial" if value == "_i" else "Final"
            explanation_case = entry.explanation[0].lower() + entry.explanation[1:]
            explanation = f"{prepend} state {explanation_case}"

        new_entry = HitranSchemaEntry(
            entry.start,
            entry.fmt,
            explanation,
        )
        new_data[key] = new_entry
    return new_data


class GlobalQuanta(CopyAttributes):
    """
    Contains the global quanta schemas for all classes of calculate_molecules in the HITRAN database.
    """

    # Class 1: Diatomic calculate_molecules
    _class1a = {
        'v1': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class1a_f = _to_final_initial(_class1a, '_f')
    class1a_i = _to_final_initial(_class1a, '_i')

    _class1b = {
        'A2': HitranSchemaEntry(start=6, fmt='2s'),
        'Omega': HitranSchemaEntry(start=8, fmt='3d'),
        'v1': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class1b_f = _to_final_initial(_class1b, '_f')
    class1b_i = _to_final_initial(_class1b, '_i')

    # Class 2: Linear triatomic calculate_molecules
    _class2a = {
        'v1': HitranSchemaEntry(start=6, fmt='2d'),
        'v2': HitranSchemaEntry(start=8, fmt='2d'),
        'l2': HitranSchemaEntry(start=10, fmt='2d'),
        'v3': HitranSchemaEntry(start=12, fmt='2d'),
        'r': HitranSchemaEntry(start=14, fmt='1d'),
    }
    class2a_f = _to_final_initial(_class2a, '_f')
    class2a_i = _to_final_initial(_class2a, '_i')

    _class2b = {
        'v1': HitranSchemaEntry(start=7, fmt='2d'),
        'v2': HitranSchemaEntry(start=9, fmt='2d'),
        'l2': HitranSchemaEntry(start=11, fmt='2d'),
        'v3': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class2b_f = _to_final_initial(_class2b, '_f')
    class2b_i = _to_final_initial(_class2b, '_i')

    # Class 3: Non-linear triatomic calculate_molecules
    _class3 = {
        'v1': HitranSchemaEntry(start=9, fmt='2d'),
        'v2': HitranSchemaEntry(start=11, fmt='2d'),
        'v3': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class3_f = _to_final_initial(_class3, '_f')
    class3_i = _to_final_initial(_class3, '_i')

    # Class 4: Pyramidal tetratomic calculate_molecules
    _class4a = {
        'v1': HitranSchemaEntry(start=5, fmt='2d'),
        'v2': HitranSchemaEntry(start=7, fmt='2d'),
        'v3': HitranSchemaEntry(start=9, fmt='2d'),
        'v4': HitranSchemaEntry(start=11, fmt='2d'),
        'S': HitranSchemaEntry(start=13, fmt='2s'),
    }
    class4a_f = _to_final_initial(_class4a, '_f')
    class4a_i = _to_final_initial(_class4a, '_i')

    _class4b = {
        'v1': HitranSchemaEntry(start=1, fmt='1d'),
        'v2': HitranSchemaEntry(start=2, fmt='1d'),
        'v3': HitranSchemaEntry(start=3, fmt='1d'),
        'v4': HitranSchemaEntry(start=4, fmt='1d'),
        'l3': HitranSchemaEntry(start=6, fmt='1d'),
        'l4': HitranSchemaEntry(start=7, fmt='1d'),
        'l': HitranSchemaEntry(start=9, fmt='1d'),
        'G_v': HitranSchemaEntry(start=11, fmt='4s'),
    }
    class4b_f = _to_final_initial(_class4b, '_f')
    class4b_i = _to_final_initial(_class4b, '_i')

    # Class 5: Linear polyatomic calculate_molecules
    _class5a = {
        'v1': HitranSchemaEntry(start=1, fmt='1d'),
        'v2': HitranSchemaEntry(start=2, fmt='1d'),
        'v3': HitranSchemaEntry(start=3, fmt='2d'),
        'v4': HitranSchemaEntry(start=5, fmt='2d'),
        'v5': HitranSchemaEntry(start=7, fmt='1d'),
        'v5s': HitranSchemaEntry(start=8, fmt='1s'),
        'l4': HitranSchemaEntry(start=9, fmt='1d'),
        'l4s': HitranSchemaEntry(start=10, fmt='1s'),
        'l5': HitranSchemaEntry(start=11, fmt='1d'),
        'l5s': HitranSchemaEntry(start=12, fmt='1s'),
        'pm': HitranSchemaEntry(start=13, fmt='1s'),
        'S': HitranSchemaEntry(start=14, fmt='1s'),
    }
    class5a_f = _to_final_initial(_class5a, '_f')
    class5a_i = _to_final_initial(_class5a, '_i')

    _class5b = {
        'v1': HitranSchemaEntry(start=1, fmt='1d'),
        'v2': HitranSchemaEntry(start=2, fmt='1d'),
        'v3': HitranSchemaEntry(start=3, fmt='1d'),
        'v4': HitranSchemaEntry(start=4, fmt='1d'),
        'v5': HitranSchemaEntry(start=5, fmt='1d'),
        'v6': HitranSchemaEntry(start=6, fmt='1d'),
        'v7': HitranSchemaEntry(start=7, fmt='1d'),
        'v8': HitranSchemaEntry(start=8, fmt='1d'),
        'v9': HitranSchemaEntry(start=9, fmt='1d'),
        'G_v': HitranSchemaEntry(start=11, fmt='1s'),
        'S': HitranSchemaEntry(start=13, fmt='2s'),
    }
    class5b_f = _to_final_initial(_class5b, '_f')
    class5b_i = _to_final_initial(_class5b, '_i')

    _class5c = {
        'v1': HitranSchemaEntry(start=2, fmt='1d'),
        'v2': HitranSchemaEntry(start=3, fmt='1d'),
        'v3': HitranSchemaEntry(start=4, fmt='1d'),
        'v4': HitranSchemaEntry(start=5, fmt='1d'),
        'v5': HitranSchemaEntry(start=6, fmt='1d'),
        'v6': HitranSchemaEntry(start=7, fmt='1d'),
        'v7': HitranSchemaEntry(start=8, fmt='1d'),
        'l5': HitranSchemaEntry(start=9, fmt='2d'),
        'l6': HitranSchemaEntry(start=11, fmt='2d'),
        'l7': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class5c_f = _to_final_initial(_class5c, '_f')
    class5c_i = _to_final_initial(_class5c, '_i')

    _class5d = {
        'v1': HitranSchemaEntry(start=0, fmt='2d'),
        'v2': HitranSchemaEntry(start=2, fmt='2d'),
        'v3': HitranSchemaEntry(start=4, fmt='2d'),
        'v4': HitranSchemaEntry(start=6, fmt='2d'),
        'v5': HitranSchemaEntry(start=8, fmt='2d'),
        'l': HitranSchemaEntry(start=10, fmt='2d'),
        'pm': HitranSchemaEntry(start=12, fmt='1s'),
        'r': HitranSchemaEntry(start=13, fmt='1d'),
        'S': HitranSchemaEntry(start=14, fmt='1s'),
    }
    class5d_f = _to_final_initial(_class5d, '_f')
    class5d_i = _to_final_initial(_class5d, '_i')

    # Class 6: Asymmetric top calculate_molecules
    _class6a = {
        'v1': HitranSchemaEntry(start=3, fmt='2d'),
        'v2': HitranSchemaEntry(start=5, fmt='2d'),
        'v3': HitranSchemaEntry(start=7, fmt='2d'),
        'v4': HitranSchemaEntry(start=9, fmt='2d'),
        'v5': HitranSchemaEntry(start=11, fmt='2d'),
        'v6': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class6a_f = _to_final_initial(_class6a, '_f')
    class6a_i = _to_final_initial(_class6a, '_i')

    _class6b = {
        'v1': HitranSchemaEntry(start=3, fmt='2d'),
        'v2': HitranSchemaEntry(start=5, fmt='2d'),
        'v3': HitranSchemaEntry(start=7, fmt='2d'),
        'n': HitranSchemaEntry(start=9, fmt='1d'),
        'tau': HitranSchemaEntry(start=10, fmt='1d'),
        'v5': HitranSchemaEntry(start=11, fmt='2d'),
        'v6': HitranSchemaEntry(start=13, fmt='2d'),
    }
    class6b_f = _to_final_initial(_class6b, '_f')
    class6b_i = _to_final_initial(_class6b, '_i')

    # Class 7: Planar symmetric calculate_molecules
    _class7 = {
        'v1': HitranSchemaEntry(start=0, fmt='2d'),
        'v2': HitranSchemaEntry(start=2, fmt='2d'),
        'v3': HitranSchemaEntry(start=4, fmt='2d'),
        'l3': HitranSchemaEntry(start=6, fmt='2d'),
        'v4': HitranSchemaEntry(start=8, fmt='2d'),
        'l4': HitranSchemaEntry(start=10, fmt='2d'),
        'G_v': HitranSchemaEntry(start=12, fmt='3s'),
    }
    class7_f = _to_final_initial(_class7, '_f')
    class7_i = _to_final_initial(_class7, '_i')

    # Class 8: Spherical top calculate_molecules/isotopologues
    _class8 = {
        'v1': HitranSchemaEntry(start=3, fmt='2d', explanation="Normal mode vibration"),
        'v2': HitranSchemaEntry(start=5, fmt='2d', explanation="Normal mode vibration"),
        'v3': HitranSchemaEntry(start=7, fmt='2d', explanation="Normal mode vibration"),
        'v4': HitranSchemaEntry(start=9, fmt='2d', explanation="Normal mode vibration"),
        'n': HitranSchemaEntry(start=11, fmt='2d', explanation="Polyad counting number"),
        'G_v': HitranSchemaEntry(start=13, fmt='2s', explanation="Vibrational symmetry"),
    }
    class8_f = _to_final_initial(_class8, '_f')
    class8_i = _to_final_initial(_class8, '_i')

    # Class 9: Explicit global quantum number notation
    _class9 = {
        'A15': HitranSchemaEntry(start=0, fmt='15s', explanation="Species specific explicit quantum number notation"),
    }
    class9_f = _to_final_initial(_class9, '_f')
    class9_i = _to_final_initial(_class9, '_i')
    _class9_ecasda = {
        'n1': HitranSchemaEntry(start=0, fmt='1d', explanation="Vibrational quantum number"),
        'n2': HitranSchemaEntry(start=1, fmt='1d', explanation="Vibrational quantum number"),
        'n3': HitranSchemaEntry(start=2, fmt='1d', explanation="Vibrational quantum number"),
        'n4': HitranSchemaEntry(start=3, fmt='1d', explanation="Vibrational quantum number"),
        'n5': HitranSchemaEntry(start=4, fmt='1d', explanation="Vibrational quantum number"),
        'n6': HitranSchemaEntry(start=5, fmt='1d', explanation="Vibrational quantum number"),
        'n7': HitranSchemaEntry(start=6, fmt='1d', explanation="Vibrational quantum number"),
        'n8': HitranSchemaEntry(start=7, fmt='1d', explanation="Vibrational quantum number"),
        'n9': HitranSchemaEntry(start=8, fmt='1d', explanation="Vibrational quantum number"),
        'n10': HitranSchemaEntry(start=9, fmt='1d', explanation="Vibrational quantum number"),
        'n11': HitranSchemaEntry(start=10, fmt='1d', explanation="Vibrational quantum number"),
        'n12': HitranSchemaEntry(start=11, fmt='1d', explanation="Vibrational quantum number"),
        'G_v': HitranSchemaEntry(start=12, fmt='3s', explanation="Vibrational symmetry"),
    }
    class9_ecasda_f = _to_final_initial(_class9_ecasda, '_f')
    class9_ecasda_i = _to_final_initial(_class9_ecasda, '_i')

class LocalQuanta(CopyAttributes):
    """
    Contains the local quanta schemas for all classes of calculate_molecules in the HITRAN database.
    """
    # Group 1: Asymmetric rotors
    _group1 = {
        "J": HitranSchemaEntry(start=0, fmt="3d", explanation="Total rotational quantum number, excluding nuclear spin"),
        "K_a": HitranSchemaEntry(start=3, fmt="3d", explanation="Projection of the angular momenum on the a-axis"),
        "K_c": HitranSchemaEntry(start=6, fmt="3d", explanation="Projection of the angular momenum on the c-axis"),
        "F": HitranSchemaEntry(start=9, fmt="5s", explanation="Total angular momentum, including nuclear spin"),
        "G_v": HitranSchemaEntry(start=14, fmt="1s", explanation="Rotational symmetry"),
    }
    group1_f = _to_final_initial(_group1, '_f')
    group1_i = _to_final_initial(_group1, '_i')
    _group1_ecasda = {
        "J": HitranSchemaEntry(start=0, fmt="3d", explanation="Total rotational quantum number, excluding nuclear spin"),
        "G_v": HitranSchemaEntry(start=3, fmt="4s", explanation="Rotational symmetry"),
        "n_J": HitranSchemaEntry(start=7, fmt="3d", explanation="Counting number"),
    }
    group1_ecasda_f = _to_final_initial(_group1_ecasda, '_f')
    group1_ecasda_i = _to_final_initial(_group1_ecasda, '_i')

    # Group 2: Closed-shell diatomic and linear calculate_molecules
    group2a_f = {
        "m": HitranSchemaEntry(start=0, fmt="1s"),
        "F": HitranSchemaEntry(start=10, fmt="5s"),
    }
    group2a_i = {
        "Br": HitranSchemaEntry(start=5, fmt="1s"),
        "J_i": HitranSchemaEntry(start=6, fmt="3d"),
        "Sym_i": HitranSchemaEntry(start=9, fmt="1s"),
        "F_i": HitranSchemaEntry(start=10, fmt="5s"),
    }

    group2b_f = {
        "l6_f": HitranSchemaEntry(start=0, fmt="2s"),
        "l7_f": HitranSchemaEntry(start=2, fmt="2s"),
        "l8_f": HitranSchemaEntry(start=4, fmt="2s"),
        "l9_f": HitranSchemaEntry(start=6, fmt="2s"),
    }
    group2b_i = {
        "l6_i": HitranSchemaEntry(start=0, fmt="2s"),
        "l7_i": HitranSchemaEntry(start=2, fmt="2s"),
        "l8_i": HitranSchemaEntry(start=4, fmt="2s"),
        "l9_i": HitranSchemaEntry(start=6, fmt="2s"),
        "Br": HitranSchemaEntry(start=9, fmt="1s"),
        "J_i": HitranSchemaEntry(start=10, fmt="3d"),
        "Sym_i": HitranSchemaEntry(start=13, fmt="1s"),
    }

    # Group 3: Spherical rotors
    _group3 = {
        "J": HitranSchemaEntry(start=2, fmt="3d"),
        "G_r": HitranSchemaEntry(start=5, fmt="2s"),
        "alpha": HitranSchemaEntry(start=7, fmt="3d"),
        "F": HitranSchemaEntry(start=10, fmt="5s"),
    }
    group3_f = _to_final_initial(_group3, '_f')
    group3_i = _to_final_initial(_group3, '_i')

    _group3_mecasda = {
        "J": HitranSchemaEntry(start=0, fmt="3d", explanation="Total rotational quantum number, excluding nuclear spin"),
        "G_r": HitranSchemaEntry(start=4, fmt="2s", explanation="Rotational symmetry"),
        "alpha": HitranSchemaEntry(start=7, fmt="3d", explanation="Counting number"),
        "F": HitranSchemaEntry(start=10, fmt="5s", explanation="Total angular momentum, including nuclear spin"),
    }
    group3_mecasda_f = _to_final_initial(_group3_mecasda, '_f')
    group3_mecasda_i = _to_final_initial(_group3_mecasda, '_i')

    # Group 4: Symmetric rotors
    _group4a = {
        "J": HitranSchemaEntry(start=0, fmt="3d"),
        "K": HitranSchemaEntry(start=3, fmt="3d"),
        "l": HitranSchemaEntry(start=6, fmt="2d"),
        "C": HitranSchemaEntry(start=8, fmt="2s"),
        "Sym": HitranSchemaEntry(start=10, fmt="1s"),
        "F": HitranSchemaEntry(start=11, fmt="4s"),
    }
    group4a_f = _to_final_initial(_group4a, '_f')
    group4a_i = _to_final_initial(_group4a, '_i')

    _group4b = {
        "J": HitranSchemaEntry(start=0, fmt="2d"),
        "K": HitranSchemaEntry(start=2, fmt="3d"),
        "l": HitranSchemaEntry(start=5, fmt="2d"),
        "G_r": HitranSchemaEntry(start=8, fmt="3s"),
        "G_tot": HitranSchemaEntry(start=11, fmt="3s"),
    }
    group4b_f = _to_final_initial(_group4b, '_f')
    group4b_i = _to_final_initial(_group4b, '_i')

    _group4c = {
        "J": HitranSchemaEntry(start=0, fmt="3d"),
        "K": HitranSchemaEntry(start=3, fmt="3d"),
        "l": HitranSchemaEntry(start=6, fmt="2d"),
        "Sym": HitranSchemaEntry(start=8, fmt="3s"),
        "F": HitranSchemaEntry(start=11, fmt="4s"),
    }
    group4c_f = _to_final_initial(_group4c, '_f')
    group4c_i = _to_final_initial(_group4c, '_i')

    # Group 5: Planar symmetric calculate_molecules
    group5 = {
        "J": HitranSchemaEntry(start=3, fmt="3d"),
        "K": HitranSchemaEntry(start=6, fmt="3d"),
        "G_tot": HitranSchemaEntry(start=11, fmt="3s"),
    }
    group5_f = _to_final_initial(group5, '_f')
    group5_i = _to_final_initial(group5, '_i')

    # Group 6: Open-shell diatomics with 3Σ ground states
    group6_f = {
        "F": HitranSchemaEntry(start=10, fmt="5s"),
    }
    group6_i = {
        "Br": HitranSchemaEntry(start=1, fmt="1s"),
        "N_i": HitranSchemaEntry(start=2, fmt="3d"),
        "Br2": HitranSchemaEntry(start=5, fmt="1s"),
        "J_i": HitranSchemaEntry(start=6, fmt="3d"),
        "F_i": HitranSchemaEntry(start=9, fmt="5s"),
        "M": HitranSchemaEntry(start=14, fmt="1s"),
    }

    # Group 7: Open-shell diatomics with 2Π ground states
    group7a_f = {
        "m": HitranSchemaEntry(start=0, fmt="1s"),
        "F": HitranSchemaEntry(start=10, fmt="5s"),
    }
    group7a_i = {
        "Br": HitranSchemaEntry(start=2, fmt="2s"),
        "J_i": HitranSchemaEntry(start=4, fmt="5.1f"),
        "Sym_i": HitranSchemaEntry(start=9, fmt="1s"),
        "F_i": HitranSchemaEntry(start=10, fmt="5s"),
    }

    group7b_f = {
        "F": HitranSchemaEntry(start=10, fmt="5s"),
    }
    group7b_i = {
        "Br": HitranSchemaEntry(start=1, fmt="2s"),
        "J_i": HitranSchemaEntry(start=3, fmt="5.1f"),
        "Sym_i": HitranSchemaEntry(start=8, fmt="2s"),
        "F_i": HitranSchemaEntry(start=10, fmt="5s"),
    }
