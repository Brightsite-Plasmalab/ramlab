import json
import warnings

from .reader_generator import HitranSchemaEntry


def exomol_ch4_schema_dict() -> dict[str, HitranSchemaEntry]:
    return {
        'ID': HitranSchemaEntry(start=0, fmt=':12d', explanation="State ID, starting at 1"),
        'E': HitranSchemaEntry(start=13, fmt=':12.6f', explanation="State energy [$\"cm\"^(-1)$]"),
        'g': HitranSchemaEntry(start=26, fmt=':6d', explanation="Total state degeneracy"),
        'J': HitranSchemaEntry(start=33, fmt=':7d',
                  explanation="Total rotational quantum number, excluding nuclear spin"),
        'unc': HitranSchemaEntry(start=41, fmt=':11.6f', explanation="Energy uncertainty [$\"cm\"^(-1)$]"),
        'tau': HitranSchemaEntry(start=53, fmt=':12.4e', explanation="Lifetime [$\"s\"^(-1)$]"),
        'G': HitranSchemaEntry(start=65, fmt=':4s',
                  explanation="$Gamma_\"tot\"$, Total symmetry in $T_d (M)$"),
        'P_n': HitranSchemaEntry(start=69, fmt=':4d', explanation="Polyad number"),
        'n_P': HitranSchemaEntry(start=73, fmt=':6d', explanation="Polyad counting number"),
        'v1': HitranSchemaEntry(start=79, fmt=':4d', explanation="$A_1$-symmetry normal mode"),
        'v2': HitranSchemaEntry(start=83, fmt=':4d', explanation="E-symmetry normal mode"),
        'L2': HitranSchemaEntry(start=87, fmt=':4d', explanation="$L_2$, vibrational angular momentum"),
        'v3': HitranSchemaEntry(start=93, fmt=':4d', explanation="$F_1$-symmetry normal mode"),
        'L3': HitranSchemaEntry(start=97, fmt=':4d', explanation="$L_3$, Vibrational angular momentum"),
        'M3': HitranSchemaEntry(start=99, fmt=':4d', explanation="$M_3$, Multiplicity index"),
        'v4': HitranSchemaEntry(start=103, fmt=':4d', explanation="$F_2$-symmetry normal mode"),
        'L4': HitranSchemaEntry(start=105, fmt=':4d', explanation="$L_4$, Vibrational angular momentum"),
        'M4': HitranSchemaEntry(start=111, fmt=':4d', explanation="$M_4$, Multiplicity index"),
        'G_v': HitranSchemaEntry(start=115, fmt=':4s', explanation="$Gamma_v$, Symmetry of the vibrational contribution in $T_d (M)$"),
        'n_J': HitranSchemaEntry(start=119, fmt=':4d', explanation="Rotational counting number"),
        'G_r': HitranSchemaEntry(start=123, fmt=':4s', explanation="$Gamma_r$, Symmetry of the rotational contribution in $T_d (M)$"),
        'i_v': HitranSchemaEntry(start=127, fmt=':6d', explanation="Vibrational state ID"),
        'Coef': HitranSchemaEntry(start=133, fmt=':6.2f', explanation="Largest coefficient used in the assignment"),
        'n1': HitranSchemaEntry(start=139, fmt=':4d', explanation="Local mode vibration"),
        'n2': HitranSchemaEntry(start=143, fmt=':4d', explanation="Local mode vibration"),
        'n3': HitranSchemaEntry(start=147, fmt=':4d', explanation="Local mode vibration"),
        'n4': HitranSchemaEntry(start=151, fmt=':4d', explanation="Local mode vibration"),
        'n5': HitranSchemaEntry(start=155, fmt=':4d', explanation="Local mode vibration"),
        'n6': HitranSchemaEntry(start=159, fmt=':4d', explanation="Local mode vibration"),
        'n7': HitranSchemaEntry(start=163, fmt=':4d', explanation="Local mode vibration"),
        'n8': HitranSchemaEntry(start=167, fmt=':4d', explanation="Local mode vibration"),
        'n9': HitranSchemaEntry(start=171, fmt=':4d', explanation="Local mode vibration"),
        'MaCa': HitranSchemaEntry(start=175, fmt=':4s', explanation="The source of the line, either 'Ma' for MARVEL or 'Ca' for Calculated"),
        'E_cal': HitranSchemaEntry(start=179, fmt=':12.6f', explanation="The calculated (TROVE) lower energy [$\"cm\"^(-1)$]"),
    }

def exomol_h2_schema_dict() -> dict[str, HitranSchemaEntry]:
    return {
        'ID': HitranSchemaEntry(start=0, fmt=':12d', explanation="State ID, starting at 1"),
        'E': HitranSchemaEntry(start=13, fmt=':12.6f', explanation="State energy [$\"cm\"^(-1)$]"),
        'g_tot': HitranSchemaEntry(start=26, fmt=':6d', explanation="Total state degeneracy"),
        'J': HitranSchemaEntry(start=33, fmt=':7d',
                               explanation="Total rotational quantum number, excluding nuclear spin"),
        'v': HitranSchemaEntry(start=40, fmt=':7d', explanation="Vibrational quantum number"),
    }

def exomol_c2h4_schema_dict() -> dict[str, HitranSchemaEntry]:
    return {
        'ID': HitranSchemaEntry(start=0, fmt=':12d', explanation="State ID, starting at 1"),
        'E': HitranSchemaEntry(start=13, fmt=':12.6f', explanation="State energy [$\"cm\"^(-1)$]"),
        'g': HitranSchemaEntry(start=26, fmt=':6d', explanation="Total state degeneracy"),
        'J': HitranSchemaEntry(start=33, fmt=':7d',
                explanation="Total rotational quantum number, excluding nuclear spin"),
        'G': HitranSchemaEntry(start=40, fmt=':5d',
                explanation="$Gamma_\"tot\"$, Index for total symmetry in $D_(2h) (M)$, $Gamma$ = $A_g$, $A_u$, $B_(1g)$, $B_(1u)$, "
                            "$B_(2g)$, $B_(2u)$, $B_(3g)$, $B_(3u)$",
                mapping=(None, "A_g", "A_u", "B_1g", "B_1u", "B_2g", "B_2u", "B_3g", "B_3u")),
        'v1': HitranSchemaEntry(start=45, fmt=':6d', explanation="C-C local mode stretch"),
        'v2': HitranSchemaEntry(start=51, fmt=':4d', explanation="C-H local mode stretch"),
        'v3': HitranSchemaEntry(start=55, fmt=':4d', explanation="C-H local mode stretch"),
        'v4': HitranSchemaEntry(start=59, fmt=':4d', explanation="C-H local mode stretch"),
        'v5': HitranSchemaEntry(start=63, fmt=':4d', explanation="C-H local mode stretch"),
        'v6': HitranSchemaEntry(start=67, fmt=':4d', explanation="CCH local mode bend"),
        'v7': HitranSchemaEntry(start=71, fmt=':4d', explanation="CCH local mode bend"),
        'v8': HitranSchemaEntry(start=75, fmt=':4d', explanation="CCH local mode bend"),
        'v9': HitranSchemaEntry(start=79, fmt=':4d', explanation="CCH local mode bend"),
        'v10': HitranSchemaEntry(start=83, fmt=':4d', explanation="Beta1 local mode bend"),
        'v11': HitranSchemaEntry(start=87, fmt=':4d', explanation="Beta2 local mode bend"),
        'v12': HitranSchemaEntry(start=91, fmt=':4d', explanation="tau local mode torsion"),
        'G_v': HitranSchemaEntry(start=95, fmt=':5d',
                explanation="$Gamma_v$, Index for symmetry of the vibrational contribution in $D_(2h) (M)$",
                mapping=(None, "A_g", "A_u", "B_1g", "B_1u", "B_2g", "B_2u", "B_3g", "B_3u")),
        'J_1': HitranSchemaEntry(start=100, fmt=':7d',
                explanation="Same as `J`"),
        'K': HitranSchemaEntry(start=107, fmt=':4d', explanation="Projection of J on molecule fixed $z$-axis"),
        'P_r': HitranSchemaEntry(start=111, fmt=':4d', explanation="$tau_(\"rot\")$, Rotational parity (0 or 1)"),
        'G_r': HitranSchemaEntry(start=115, fmt=':4d',
                explanation="$Gamma_r$, Index for symmetry of the rotational contribution in$D_(2h) (M)$",
                mapping=(None, "A_g", "A_u", "B_1g", "B_1u", "B_2g", "B_2u", "B_3g", "B_3u")),
    }

def exomol_c2h4_schema_dict_extra() -> dict[str, HitranSchemaEntry]:
    return {
        'U1': HitranSchemaEntry(start=119, fmt=':10d', explanation="Unknown, all 0, may be counting number"),
        'U2': HitranSchemaEntry(start=129, fmt=':7.2f', explanation="Unknown float between 0 and 1"),
        'n1_2': HitranSchemaEntry(start=136, fmt=':7d', explanation="Same as `n1`"),
        'n2_2': HitranSchemaEntry(start=143, fmt=':4d', explanation="Same as `n2`"),
        'n3_2': HitranSchemaEntry(start=147, fmt=':4d', explanation="Same as `n3`"),
        'n4_2': HitranSchemaEntry(start=151, fmt=':4d', explanation="Same as `n4`"),
        'n5_2': HitranSchemaEntry(start=155, fmt=':4d', explanation="Same as `n5`"),
        'n6_2': HitranSchemaEntry(start=159, fmt=':4d', explanation="Same as `n6`"),
        'n7_2': HitranSchemaEntry(start=163, fmt=':4d', explanation="Same as `n7`"),
        'n8_2': HitranSchemaEntry(start=167, fmt=':4d', explanation="Same as `n8`"),
        'n9_2': HitranSchemaEntry(start=171, fmt=':4d', explanation="Same as `n9`"),
        'n10_2': HitranSchemaEntry(start=175, fmt=':4d', explanation="Same as `n10`"),
        'n11_2': HitranSchemaEntry(start=179, fmt=':4d', explanation="Same as `n11`"),
        'n12_2': HitranSchemaEntry(start=183, fmt=':4d', explanation="Same as `n12`"),
    }

def exomol_c2h2_schema_dict() -> dict[str, HitranSchemaEntry]:
    return {
        'ID': HitranSchemaEntry(start=0, fmt=':12d', explanation="State ID, starting at 1"),
        'E': HitranSchemaEntry(start=13, fmt=':12.6f', explanation="State energy [$\"cm\"^(-1)$]"),
        'g': HitranSchemaEntry(start=26, fmt=':6d', explanation="Total state degeneracy"),
        'J': HitranSchemaEntry(start=33, fmt=':7d',
                explanation="Total rotational quantum number, excluding nuclear spin"),
        'unc': HitranSchemaEntry(start=41, fmt=':12.6f', explanation="Energy uncertainty [$\"cm\"^(-1)$]"),
        'G': HitranSchemaEntry(start=53, fmt=':4s',
                explanation="$Gamma_\"tot\"$, Total symmetry in $D_(infinity h) (M)$"),
        'v1': HitranSchemaEntry(start=57, fmt=':3d', explanation="Symmetric CC stretching"),
        'v2': HitranSchemaEntry(start=60, fmt=':3d', explanation="CH stretching"),
        'v3': HitranSchemaEntry(start=63, fmt=':3d', explanation="CH stretching"),
        'v4': HitranSchemaEntry(start=66, fmt=':3d', explanation="CHH bending"),
        'v5': HitranSchemaEntry(start=69, fmt=':3d', explanation="CHH bending"),
        'v6': HitranSchemaEntry(start=72, fmt=':3d', explanation="CHH bending"),
        'v7': HitranSchemaEntry(start=75, fmt=':3d', explanation="CHH bending"),
        'G_v': HitranSchemaEntry(start=78, fmt=':4s',
                explanation="$Gamma_v$, Symmetry of the vibrational contribution in $D_(infinity h) (M)$"),
        'K': HitranSchemaEntry(start=82, fmt=':3d',
                explanation="Projection of J on the axis of molecular rotational symmetry"),
        'tau_rot': HitranSchemaEntry(start=85, fmt=':2s', explanation="Rotational parity"),
        'G_r': HitranSchemaEntry(start=87, fmt=':4s',
                explanation="$Gamma_r$, Symmetry of the rotational contribution in $D_(infinity h) (M)$"),
        'Coef': HitranSchemaEntry(start=91, fmt=':6.2f',
                explanation="Coefficient with the largest contribution to the (J = 0) contracted set"),
        'MaCa': HitranSchemaEntry(start=97, fmt=':3s',
                explanation="Indicates if the value is from MARVEL (Ma) or calculated (Ca)"),
        'E_cal': HitranSchemaEntry(start=100, fmt=':13.6f',
                explanation="Energy in cm-1 from variational spectroscopic model"),
    }

def exomol_h2o_schema_dict() -> dict[str, HitranSchemaEntry]:
    return {
        'ID': HitranSchemaEntry(start=0, fmt=':12d', explanation="State ID, starting at 1"),
        'E': HitranSchemaEntry(start=13, fmt=':12.6f', explanation="State energy [$\"cm\"^(-1)$]"),
        'g_tot': HitranSchemaEntry(start=26, fmt=':6d', explanation="Total state degeneracy"),
        'J': HitranSchemaEntry(start=33, fmt=':7d',
                               explanation="Total rotational quantum number, excluding nuclear spin"),
        'unc': HitranSchemaEntry(start=41, fmt=':13.6f', explanation="Energy uncertainty [$\"cm\"^(-1)$]"),
        'Ka': HitranSchemaEntry(start=54, fmt=':5d', explanation="Asymmetric top quantum number Ka"),
        'Kc': HitranSchemaEntry(start=59, fmt=':5d', explanation="Asymmetric top quantum number Kc"),
        'v1': HitranSchemaEntry(start=64, fmt=':5d', explanation="Symmetric OH stretching vibration quantum number"),
        'v2': HitranSchemaEntry(start=69, fmt=':5d', explanation="Bending vibration quantum number"),
        'v3': HitranSchemaEntry(start=74, fmt=':5d', explanation="Asymmetric OH stretching vibration quantum number"),
        'G_r': HitranSchemaEntry(start=79, fmt=':3s',
                explanation="$Gamma_r$, Symmetry of the rotational contribution in $C_(2 v) (M)$"),
        'E_cal': HitranSchemaEntry(start=82, fmt=':17.6f',
                explanation="Energy in cm-1 from variational spectroscopic model"),
        'MaCa': HitranSchemaEntry(start=99, fmt=':5s',
                explanation="Indicates if the value is from MARVEL (Ma) or calculated (Ca)"),
    }


def schema_dict_from_json(json_loc):
    with open(json_loc, 'r') as f:
        data = json.load(f)

    fields = data['dataset']['states']['states_file_fields']

    schema_dict = {}
    start = 0
    for index, field in enumerate(fields):
        fmt = field['cfmt'].replace("%", ":")
        if "/" in fmt:
            warnings.warn(f"The entry {field['name']} has multiple formats: {field['cfmt']}, the first will be used")
            fmt = fmt.split("/", 1)[0]
        name = field['name'].split(":")[-1]
        if name in schema_dict:
            warnings.warn(f"The entry {name} has multiple instances, the second one will get a suffix")
            index = 1
            while True:
                if f"{name}_{index}" not in schema_dict:
                    name = f"{name}_{index}"
                    break
                index += 1
        schema_dict[name] = HitranSchemaEntry(start=start, fmt=fmt)
        start += schema_dict[name].length
        if index < 3:
            start += 1
    return schema_dict

schemas = {
    'CH4': exomol_ch4_schema_dict,
    'C2H2': exomol_c2h2_schema_dict,
    'C2H4': exomol_c2h4_schema_dict,
    'C2H4_EXTRA': exomol_c2h4_schema_dict_extra,
    'H2': exomol_h2_schema_dict,
    'H2O': exomol_h2o_schema_dict,
}

def get_schema(name: str) -> dict:
    """
    Retrieve the ExoMol schema dictionary for a given molecule name.

    Parameters
    ----------
    name : str
        The name of the molecule (e.g., 'CH4', 'H2O').

    Returns
    -------
    dict
        The schema dictionary corresponding to the specified molecule.

    Raises
    ------
    ValueError
        If the specified molecule name does not have an associated schema.
    """
    try:
        return schemas[name.strip().upper()]()
    except KeyError:
        raise ValueError(f"No schema found for molecule '{name}'. Available molecules: {list(schemas.keys())}")

