import pandas as pd

from ramlab.molecules.transitions import Transitions


# HITRAN format:
# df2["molecule_no"] = df[0].str[0:2].astype("Int64")
# df2["isotope_no"] = df[0].str[2].astype("Int64")
# df2["vacuum_wavenumber"] = df[0].str[3:15].astype(float)
# df2["intensity"] = df[0].str[15:25].astype(float)
# df2["einstein_A_coefficient"] = df[0].str[25:35].astype(float)
# df2["gamma_air"] = df[0].str[35:40].astype(float)
# df2["gamma_self"] = df[0].str[40:45].astype(float)
# df2["lower_E"] = df[0].str[45:55].astype(float)
# df2["temperature_dependence_coefficient"] = df[0].str[55:59].astype(float)
# df2["air_pressure_induced_line_shift"] = df[0].str[59:67].astype(float)
# df2["upper_quanta_global"] = df[0].str[67:82]
# df2["lower_quanta_global"] = df[0].str[82:97]
# df2["upper_quanta_local"] = df[0].str[97:112]
# df2["lower_quanta_local"] = df[0].str[112:127]
# df["Ierr"] = df[0].str[127:133]
# df2["Iref"] = df[0].str[133:145]
# df2["flag"] = df[0].str[145]
# df2["upper_degeneracy"] = df[0].str[146:153].astype(float)
# df2["lower_degeneracy"] = df[0].str[153:160].astype(float)
# Changes: format initial,final as lower,upper


def format_transitions_initial_final(transitions: Transitions) -> Transitions:
    # Define row formatter according to the above format
    def format_row(row):
        return (
            f"{row.molecule_number:02d}"
            f"{row.isotope_number:1d}"
            f"{row.vacuum_wavenumber:12.6f}"
            f"{row.crosssection:10.3e}"
            f"{row.depolarization_ratio:10.3e}"
            f"{0:5.2f}"  # gamma_air
            f"{0:5.2f}"  # gamma_self
            f"{row.initial_E:10.3f}"  # Lower state energy
            f"{0:4.2f}"  # Temperature-dependence coefficient
            f"{0:8.2f}"  # Air-pressure-induced line shift
            f"{row.final_quanta_global:15s}"
            f"{row.initial_quanta_global:15s}"
            f"{row.final_quanta_local:15s}"
            f"{row.initial_quanta_local:15s}"
            f"{'':6s}"  # Ierr
            f"{'':12s}"  # Iref
            f"{'':1s}"  # flag
            f"{row.final_degeneracy:7.0f}"
            f"{row.initial_degeneracy:7.0f}"
        )

    # Format rows
    return transitions.linelist.apply(format_row, axis=1)
