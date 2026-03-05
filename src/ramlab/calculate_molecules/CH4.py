from typing import Iterable
import operator
import functools

import polars as pl
import pathlib

from ramlab.data_parsing.hitran import hitran_schema_dict, LocalQuanta, GlobalQuanta
from ramlab.data_parsing.reader_generator import add_subschema, read_data
from ramlab.simulate_spectrum.convolution import convolute_spectrum
from ramlab.calculate_molecules.ab_initio_molecule2 import AbInitioMolecule


def _mk_eq_fil(other, val):
    if isinstance(val, Iterable):
        vals = [other == v for v in val]
        return functools.reduce(operator.or_, vals)
    else:
        return other == val,

def _extend_eq_filter(filter_list, **kwargs):
    for key, val in kwargs.items():
        if val is not None:
            filter_list.extend(_mk_eq_fil(pl.col(key), val))


class CH4(AbInitioMolecule):
    def __init__(self, loc=r"data/CH4/extract_12CH4_0_10000_0_pol.txt", T_ref=296.0, data_filter=None):
        self._data_path = pathlib.Path(loc)
        schema = add_subschema(
            hitran_schema_dict(),
            global_upper_quanta=GlobalQuanta.class8_u,
            global_lower_quanta=GlobalQuanta.class8_l,
            local_upper_quanta=LocalQuanta.group3_macesda_u,
            local_lower_quanta=LocalQuanta.group3_macesda_l,
        )
        entries = (
            'nu', 'sw', 'elower', 'v1_u', 'v2_u', 'v3_u', 'v4_u', 'v1_l', 'v2_l', 'v3_l', 'v4_l', 'C_l', 'J_u', 'J_l'
        )
        self.data = read_data(self._data_path, schema, entries=entries)
        self.T_ref = T_ref

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

        cond = pl.when(pl.col("v1_l") == -1).then(None)
        for vib in vib_levels:
            vib_filter = self.vib_levels_filter(v1_l=vib[0], v2_l=vib[1], v3_l=vib[2], v4_l=vib[3])
            data_filtered = self.data.filter(vib_filter)
            syms = data_filtered.select("C_l").unique().collect()["C_l"].to_list()
            for sym in syms:
                e_vib = data_filtered.filter((pl.col("C_l") == sym) & self.rot_levels_filter(J_l=0)).collect()["elower"].mean()
                cond = cond.when(vib_filter & (pl.col("C_l") == sym)).then(e_vib)

        self.data = self.data.with_columns(
            cond.alias("E_vib")
        )
        self.data = self.data.with_columns(
            (pl.col("elower") - pl.col("E_vib")).alias("E_rot")
        )
        if data_filter is not None:
            self.data = self.data.filter(data_filter)
        self.data = self.data.collect()

    def intensity(self, T_vib, T_rot=None, data_filter=None, energy='exact') -> tuple[pl.Series, pl.Series]:
        def set_filter(data_filter):
            if data_filter is not None:
                data = self.data.filter(data_filter)
            else:
                data = self.data
            return data

        data = set_filter(data_filter)

        if energy.lower() == "exact":
            if T_rot is None:
                boltz = (1.4387770 * data["elower"]*(1 / self.T_ref - 1 / T_vib)).exp()
            else:
                boltz = (1.4387770 * (data["E_vib"]*(1 / self.T_ref - 1 / T_vib)
                                       + data["E_rot"]*(1 / self.T_ref - 1 / T_rot))).exp()
        elif energy.lower() == "tom":
            T_rot = T_vib if T_rot is None else T_rot
            if "E_vib_tom" not in data.columns:
                self.data = self.data.with_columns(
                    (5.2412 * pl.col("J_l") * (pl.col("J_l") + 1)).alias("E_rot_tom")
                ).with_columns(
                    (2932.369 * pl.col("v1_l") + 1533.332567 * pl.col("v2_l") + 3018.5292 * pl.col(
                        "v3_l") + 1310.761458 * pl.col("v4_l")).alias("E_vib_tom")
                )
                data = set_filter(data_filter)
            boltz = (1.4387770 * (data["E_vib_tom"] * (1 / self.T_ref - 1 / T_vib)
                                  + data["E_rot_tom"] * (1 / self.T_ref - 1 / T_rot))).exp()
        else:
            raise ValueError(f"Unknown energy calculation method: {energy}, use 'exact' or 'tom'")
        return data['nu'], data["sw"] * boltz

    @staticmethod
    def vib_levels_filter(*, v1_l=None, v2_l=None, v3_l=None, v4_l=None, v1_u=None, v2_u=None, v3_u=None, v4_u=None,
                          dv1=None, dv2=None, dv3=None, dv4=None) -> pl.Expr:
        """
        Create a filter for selecting vibrational levels based on the provided parameters.
        """
        filter = []
        _extend_eq_filter(filter, v1_l=v1_l, v2_l=v2_l, v3_l=v3_l, v4_l=v4_l,
                         v1_u=v1_u, v2_u=v2_u, v3_u=v3_u, v4_u=v4_u)

        if dv1 is not None:
            filter.extend(_mk_eq_fil((pl.col("v1_u") - pl.col("v1_l")), dv1))
        if dv2 is not None:
            filter.extend(_mk_eq_fil((pl.col("v2_u") - pl.col("v2_l")), dv2))
        if dv3 is not None:
            filter.extend(_mk_eq_fil((pl.col("v3_u") - pl.col("v3_l")), dv3))
        if dv4 is not None:
            filter.extend(_mk_eq_fil((pl.col("v4_u") - pl.col("v4_l")), dv4))

        if len(filter) == 0:
            raise ValueError("No vibrational levels selected")

        return functools.reduce(operator.and_, filter)

    @staticmethod
    def rot_levels_filter(*, J_u=None, J_l=None, dJ=None) -> pl.Expr:
        filter = []
        _extend_eq_filter(filter, J_u=J_u, J_l=J_l)
        if dJ is not None:
            filter.extend(_mk_eq_fil((pl.col("J_u") - pl.col("J_l")), dJ))

        if len(filter) == 0:
            raise ValueError("No rotational levels selected")

        return functools.reduce(operator.and_, filter)

    def generate_spectrum(self, wavelengths, peak_shape, T_vib, T_rot=None, data_filter=None, energy="exact"):
        dx = wavelengths[1] - wavelengths[0]
        wav_min = wavelengths[0] - dx*len(peak_shape)
        wav_max = wavelengths[-1] + dx*len(peak_shape)
        if data_filter is not None:
            data_filter = data_filter & (pl.col("nu") >= wav_min) & (pl.col("nu") <= wav_max)
        else:
            data_filter = (pl.col("nu") >= wav_min) & (pl.col("nu") <= wav_max)

        peak_wav, peak_inten = self.intensity(T_vib, T_rot, data_filter, energy=energy)
        return convolute_spectrum(wavelengths, peak_wav.to_numpy(), peak_inten.to_numpy(), peak_shape)

if __name__ == "__main__":
    ch4 = CH4()