import pandas as pd
from ramlab.util.pandas import load_csv_with_progress

# Hitran format:
# line[0:2], # Molecule no
# line[2], # Isotope no
# line[3:15],  # Vacuum wavenumber
# line[15:25],  # Intensity
# line[25:35],  # Einstein A_coefficient
# line[35:40], # gamma_air
# line[40:45], # gamma_self
# line[45:55],  # E'' lower state
# line[55:59], # Temperature-dependence coefficient
# line[59:67], # Air-pressure-induced line shift
# line[67:82],  # V' Upper-state “global” quanta
# line[82:97],  # V'' Lower-state “global” quanta
# line[97:112],  # q′	Upper-state “local” quanta
# line[112:127],  # Q″	Lower-state “local” quanta
# line[127:133], # Ierr	Uncertainty indices
# line[133:145], # Iref	Reference indices
# line[145], # *	Flag
# line[146:153],  # g′	Statistical weight of the upper state
# line[153:160],  # g″	Statistical weight of the lower state


def parse_hitran_data(file_path):
    # See https://hitran.org/media/refs/HITRAN_QN_formats.pdf
    df = load_csv_with_progress(file_path, header=None)

    idx_reject = df[0].str.len() != 160
    df = df[~idx_reject]
    print(f"Rejected {idx_reject.sum()} lines with a length not equal to 160.")

    df2 = pd.DataFrame()
    df2["molecule_no"] = df[0].str[0:2].astype("Int64")
    df2["isotope_no"] = df[0].str[2].astype("Int64")
    df2["vacuum_wavenumber"] = df[0].str[3:15].astype(float)
    df2["intensity"] = df[0].str[15:25].astype(float)
    df2["einstein_A_coefficient"] = df[0].str[25:35].astype(float)
    # df2["gamma_air"] = df[0].str[35:40].astype(float)
    # df2["gamma_self"] = df[0].str[40:45].astype(float)
    df2["initial_E"] = df[0].str[45:55].astype(float)
    # df2["temperature_dependence_coefficient"] = df[0].str[55:59].astype(float)
    # df2["air_pressure_induced_line_shift"] = df[0].str[59:67].astype(float)
    df2["final_quanta_global"] = df[0].str[67:82]
    df2["initial_quanta_global"] = df[0].str[82:97]
    df2["final_quanta_local"] = df[0].str[97:112]
    df2["initial_quanta_local"] = df[0].str[112:127]
    # df["Ierr"] = df[0].str[127:133]
    # df2["Iref"] = df[0].str[133:145]
    # df2["flag"] = df[0].str[145]
    df2["final_degeneracy"] = df[0].str[146:153].astype(float)
    df2["initial_degeneracy"] = df[0].str[153:160].astype(float)

    return df2


def main():
    from pathlib import Path

    path_data = Path(
        "/Users/mruijzendaal/Projects/UM/Software/data/extract_12CH4_2850_3000_0.000001_pol.txt"
    )

    print(parse_hitran_data(path_data))


if __name__ == "__main__":
    main()
