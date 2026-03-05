import polars as pl

from ramlab.data_parsing.hitran import full_schema_dict
from ramlab.molecule_base.filters import levels_filter_and
from ramlab.molecule_base.Molecule import Molecule

ch4_mecasda_schema = full_schema_dict(
    general_schema="hitran",
    globals_="class8",
    locals_="group3_mecasda",
)

ch4_ramlab_schema = full_schema_dict(
    general_schema="ramlab",
    globals_="class8",
    locals_="group3_mecasda",
)

class Methane(Molecule):
    """
    A class representing the methane (CH4) molecule.
    """
    _default_schema = ch4_mecasda_schema
    _must_params = Molecule._must_params + ("v1_i", "v2_i", "v3_i", "v4_i", "J_i","v1_f", "v2_f", "v3_f", "v4_f", "J_f")
    _temperatures = {"vib": "vib", "rot": "rot", "v": "vib", "r": "rot"}
    _unique_entries = ('v1_i', 'v2_i', 'v3_i', 'v4_i', 'J_i', 'G_v_i', 'G_r_i', 'alpha_i')

    def __init__(self, data, T_ref, partition_interpolator=None):
        super().__init__(data, T_ref, partition_interpolator)

        vib_levels = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 1, 0, 1],
        ]
        
        cond = pl.when(pl.col("v1_i") == -1).then(None)
        for vib in vib_levels:
            vib_filter = levels_filter_and(v1_i=vib[0], v2_i=vib[1], v3_i=vib[2], v4_i=vib[3])
            data_filtered = self._data.filter(vib_filter)
            syms = data_filtered.select("G_v_i").unique()["G_v_i"].to_list()
            for sym in syms:
                e_vib = data_filtered.filter((pl.col("G_v_i") == sym) & levels_filter_and(J_i=0))["E_i"].mean()
                cond = cond.when(vib_filter & (pl.col("G_v_i") == sym)).then(e_vib)

        self._data = self._data.with_columns(
            cond.alias("E_i_vib")
        )

        if self._data["E_i_vib"].has_nulls():
            pl.col("E_i_vib").has_nulls()
            msg = ("Some vibrational energies could not be assigned. Please check the input data."
                   f"\n{self._data.filter((pl.col(['E_i_vib']).is_null()))}")
            raise ValueError(msg)

        self._data = self._data.with_columns(
            (pl.col("E_i") - pl.col("E_i_vib")).alias("E_i_rot")
        )

        self._data = self._data.with_columns(
            (2*pl.col("v1_i") + pl.col("v2_i") + 2*pl.col("v3_i") + pl.col("v4_i")).alias("P_n_i"),
            (2*pl.col("v1_f") + pl.col("v2_f") + 2*pl.col("v3_f") + pl.col("v4_f")).alias("P_n_f"),
        )