import os
from abc import ABC, abstractmethod

import polars as pl
import numpy as np

from ramlab.molecule_base.Molecule import Population
from ramlab.state.polarisation import Polarisation
from ramlab.state.state2 import RoVibState
from ramlab.data_parsing.reader_generator import read_data
from ramlab.molecule_base.molecule_base import DataHandler, missing_message
from ramlab.molecule_base.filters import levels_filter_and


# class MustAttributes(ABC):
#     _must_attributes = ()
#
#     def __init__(self):
#         missing = []
#         for attr in self._must_attributes:
#             if not hasattr(self, attr) or getattr(self, attr) is None:
#                 missing.append(attr)
#         if not missing:
#             return
#
#         msg = missing_message(missing, "The attribute{s} {param} must be implemented in the subclass '{class_name}'",)
#         raise NotImplementedError(msg)


class CalculateMolecule(ABC, DataHandler):
    @abstractmethod
    def cross_section(self, wavelength: float, polarization: Polarisation | str, *, filters: pl.Expr,
                     save: str | bool = True) -> pl.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def calculate_degeneracies(data: pl.DataFrame) -> pl.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def import_energies(cls, hitran_loc, hitran_schema=None) -> pl.DataFrame:
        pass


class CalculateRoVibMolecule(CalculateMolecule, Population):
    _hitran_schema = None
    _vib_names = None

    _must_params = Population._must_params + ("E_i_vib", "E_i_rot", "J_i", "v1_i")

    def __init__(self, data: pl.DataFrame | str | os.PathLike):
        data = self.from_txt(data)
        if "g_i" not in data.columns:
            data = self.calculate_degeneracies(data)
        super().__init__(data)

    # @staticmethod
    # def calculate_energies(vib_constants, rot_constants, v_max, J_max) -> pl.DataFrame:
    #     """Calculate rovibrational energies for a grid of vibrational and rotational levels.
    #
    #     The function constructs all pairs of vibrational (v) and rotational (J) quantum
    #     numbers in the ranges 0..v_max and 0..J_max (inclusive) and computes vibrational
    #     and rotational energy contributions using simple polynomial expansions provided
    #     by `vib_constants` and `rot_constants`.
    #
    #     Parameters
    #     ----------
    #     vib_constants : Sequence[float]
    #         Coefficients for the vibrational-energy polynomial. The first element is
    #         multiplied by (v + 0.5) (i.e. the harmonic term), and subsequent elements
    #         are applied to higher powers: coeffs[1] * (v + 0.5)**2, coeffs[2] *
    #         (v + 0.5)**3, etc.
    #     rot_constants : float or Sequence[float]
    #         Either a scalar rotational constant (applied to all vibrational levels) or
    #         a sequence of coefficients that define a vibrational-dependant rotational
    #         constant via a polynomial in (v + 0.5) using the same convention as
    #         `vib_constants` (first element multiplies (v + 0.5), etc.). The resulting
    #         per-vibrational-level rotational constant(s) are then used to compute
    #         rotational energies.
    #     v_max : int
    #         Maximum vibrational quantum number (inclusive).
    #     J_max : int
    #         Maximum rotational quantum number (inclusive).
    #
    #     Returns
    #     -------
    #     pl.DataFrame
    #         A DataFrame with the following columns for each (v, J) pair:
    #         - "v1_i": vibrational quantum number v
    #         - "J_i": rotational quantum number J
    #         - "E_i_vib": computed vibrational energy for the vibrational level
    #         - "E_i_rot": computed rotational energy for the rotational level
    #         - "E_i": total energy (E_i_vib + E_i_rot)
    #
    #     Notes
    #     -----
    #     - Energies are in the same units as the provided constants; the function does
    #       not perform unit conversions.
    #     - The polynomial model is a simple expansion and may not capture all
    #       high-order physical effects; it is intended for quick, analytic estimates
    #       or as a building block for further fitting.
    #     - Both `v_max` and `J_max` should be non-negative integers.
    #     - If `rot_constants` is a sequence, the implementation computes a vibrationally
    #       dependent rotational-constant polynomial. If it is a scalar, the same
    #       constant is used for all vibrational levels.
    #     """
    #     state = RoVibState.for_each(J=np.arange(0, J_max + 1), v=np.arange(0, v_max + 1))
    #     vib_states, rot_states = state["v"], state["J"]
    #
    #     vib_energy = (vib_states + 0.5) * vib_constants[0]
    #     for i, value in enumerate(vib_constants[1:], start=2):
    #         vib_energy += value * (vib_states + 0.5) ** i
    #
    #     if isinstance(rot_constants, (float, int)):
    #         calc_rot_constants = rot_constants
    #     else:
    #         calc_rot_constants = (vib_states+0.5) * rot_constants[0]
    #         for i, value in enumerate(rot_constants[1:], start=2):
    #             rot_constants += value * (vib_states + 0.5) ** i
    #
    #     rot_energy = calc_rot_constants * rot_states * (rot_states + 1)
    #     for i, value in enumerate(calc_rot_constants, start=2):
    #         rot_energy += value * (rot_states * (rot_states + 1)) ** i
    #
    #     df = pl.DataFrame({
    #         "v1_i": vib_states,
    #         "J_i": rot_states,
    #         "E_i_vib": vib_energy,
    #         "E_i_rot": rot_energy,
    #         "E_i": vib_energy + rot_energy,
    #     })
    #     return df

    def extrapolate_levels(self, v_polynomial, J_polynomial, **max_vals: int):
        raise NotImplementedError()  # TODO

    @classmethod
    def import_energies(cls, hitran_loc, hitran_schema=None):
        # TODO: Currently this assumes there is a J=0 level for each vibrational level.
        # This may not be the case for some molecules.
        # In that case we may need to do a fitting of the rotational levels to find the vibrational energy.

        if hitran_schema is None:
            if cls._hitran_schema is None:
                raise ValueError("No hitran schema provided for importing energies.")
            hitran_schema = cls._hitran_schema

        data = read_data(hitran_loc, schema=hitran_schema, collect=True)
        vib_names = cls._get_vib_names(data)

        sym = True if "G_v_i" in data.columns else False

        vibs = data.select(vib_names).unique()
        cond = pl.when(pl.col("v1_i") == -1).then(None)
        for vib in vibs.iter_rows():
            vib_filter_kwargs = {
                name: val for name, val in zip(vibs.columns, vib)
            }
            vib_fiter = levels_filter_and(**vib_filter_kwargs)
            data_filtered = data.filter(vib_fiter)
            if sym:
                syms = data_filtered.select("G_v_i").unique().collect()["G_v_i"].to_list()
                for s in syms:
                    vib_mask = (pl.col("G_v_i") == s) & levels_filter_and(J_i=0)
                    e_vib = data_filtered.filter(vib_mask).collect()["E_i"].mean()
                    cond = cond.when(vib_fiter & (pl.col("G_v_i") == s)).then(e_vib)
            else:
                e_vib = data_filtered.filter(levels_filter_and(J_i=0)).collect()["E_i"].mean()
                cond = cond.when(vib_fiter).then(e_vib)

        data = data.with_columns(
            cond.alias("E_i_vib")
        )
        return data

    @classmethod
    def _get_vib_names(cls, data: pl.DataFrame) -> list[str]:
        if cls._vib_names is not None:
            return cls._vib_names

        names = []
        num = 1
        while True:
            vib_name = f'v{num}_i'
            if vib_name in data.columns:
                names.append(vib_name)
                num += 1
            else:
                break
        return names
