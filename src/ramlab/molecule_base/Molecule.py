from typing import Self, Mapping, ClassVar, Unpack, overload

import numpy as np
import scipy
import polars as pl

from ramlab._type_hints import floatNDArray1D
from ramlab.molecule_base.molecule_base import DataHandler, _TypeReadTxtLoc
from ramlab.molecule_base._temperature_typehints import OneTemperatureKwargs, RoVibTemperatureKwargs

kB_per_mK = scipy.constants.value('Boltzmann constant in inverse meter per kelvin')


class Population(DataHandler):
    _temperatures: ClassVar[Mapping[str, str]] = {"T_r": "T_r", "Tr": "T_r", "T_rot": "T_r", "Trot": "T_r",
                                                  "T_v": "T_v", "Tv": "T_v", "T_vib": "T_v", "Tvib": "T_v"}
    _unique_entries: ClassVar[tuple[str, ...]]
    _must_params: ClassVar[tuple[str, ...]] = ("E_i", "g_i")

    def __init__(self, data: pl.DataFrame | _TypeReadTxtLoc, partition_interpolator = None):
        super().__init__(data)
        self._partition_interpolator = partition_interpolator

        names = [p for p in self._must_params if p.startswith("d")]
        if names:
            msg = f"Column names starting with 'd' are reserved for differences. Found '{names}'."
            raise ValueError(msg)

    def unique(self):
        unique_data = self._data.unique(self._unique_entries)
        return self.__class__(unique_data, partition_interpolator=self._partition_interpolator)

    def _filter_data(self, data_filter: pl.Expr) -> pl.DataFrame:
        if data_filter is not None:
            return self._data.filter(data_filter)
        else:
            return self._data

    def filtered_column(self, column_name: str, data_filter: pl.Expr) -> np.ndarray:
        """Returns a column from the internal dataframe, filtered by the provided expression.

        Parameters
        ----------
        column_name : str
            The name of the column to return.
        data_filter : pl.Expr
            This is used to filter the internal dataframe `_data` to get the states of interest.
            For example, to get the energies of states with J=1, use `pl.col("J") == 1`.

        Returns
        -------
        np.ndarray
            The values of the specified column after filtering.
        """
        if column_name not in self._data.columns:
            raise ValueError(f"Column '{column_name}' not found in data, available columns: {self._data.columns}")
        return self._filter_data(data_filter)[column_name].to_numpy()

    @overload
    def partition_sum(self, **temperature: Unpack[OneTemperatureKwargs]) -> float:
        pass

    @overload
    def partition_sum(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> float:
        pass

    def partition_sum(self, **temperatures: float) -> float:
        """Returns the partition sum of the molecule.

        Parameters
        ------------
        **temperatures:
            The temperatures in Kelvin.

        Returns
        ------------
        float:
            The partition sum of all the states in the molecule.
        """
        if (self._partition_interpolator is not None) and (len(temperatures) == 1):
            temperatures = self._check_temperatures(**temperatures)
            return self._partition_interpolator(temperatures["T"])
        return np.sum(self._relative_populations(**temperatures))

    @overload
    def _relative_populations(self, **temperature: Unpack[OneTemperatureKwargs]) -> floatNDArray1D:
        pass

    @overload
    def _relative_populations(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> floatNDArray1D:
        pass

    def _relative_populations(self, **temperatures: float) \
            -> floatNDArray1D:
        """Returns the relative populations of a state.

        Parameters
        ----------
        temperatures: float
            The temperatures in Kelvin.
        """
        boltz = self._boltz_expr(**temperatures)
        return self._data.select(pl.col("g_i") * boltz)["g_i"].to_numpy()

    @overload
    def population(self, **temperature: Unpack[OneTemperatureKwargs]) -> floatNDArray1D:
        pass

    @overload
    def population(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> floatNDArray1D:
        pass

    def population(self, **temperatures: float) -> floatNDArray1D:
        """Returns the populations of a state.

        Parameters
        ----------
        temperatures: float
            The temperatures in Kelvin.

        Returns
        ------------
        np.ndarray
            The populations of the state.
        """
        rel_pop = self._relative_populations(**temperatures)
        partition_sum = self.partition_sum(**temperatures)
        n = rel_pop / partition_sum
        return n

    @overload
    def intensity_variable(self, **temperature: Unpack[OneTemperatureKwargs]) -> floatNDArray1D:
        pass

    @overload
    def intensity_variable(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> floatNDArray1D:
        pass

    def intensity_variable(self, **temperatures: float) -> floatNDArray1D:
        """Returns the variable part of the intensity calculation. These involve populations, but not physical
        constants or cross-sections. When fitting temperatures, this is the only part that changes. Calculating this
        separately greatly speeds up the fitting process.

        Parameters
        ----------
        temperatures: float
            The temperatures in Kelvin.

        Returns
        ------------
            float: The intensity variable of the transition.
        """
        return self.population(**temperatures)

    def filter(self, data_filter: pl.Expr) -> Self:
        """Returns a new Molecule instance with the data filtered by the provided expression.

        Parameters
        ----------
        data_filter : pl.Expr
            A Polars expression to filter the data. Default is None, which means no filtering.

        Returns
        -------
        Self
            A new Molecule instance with the filtered data.
        """
        return self.__class__(self._filter_data(data_filter), self._partition_interpolator)

    @overload
    def _boltz_expr(self, **temperature: Unpack[OneTemperatureKwargs]) -> pl.Expr:
        pass

    @overload
    def _boltz_expr(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> pl.Expr:
        pass

    def _boltz_expr(self, **temperatures: float) -> pl.Expr:
        temperatures = self._check_temperatures(**temperatures)
        if len(temperatures) == 1:
            boltz = (-1.4387770 * pl.col("E_i") / temperatures["T"]).exp()
        else:
            boltz: pl.Expr = 0  # noqa
            for temp_name, temp_value in temperatures.items():
                boltz += (-1.4387770 * pl.col(f"E_i_{temp_name}") / temp_value)
            boltz = boltz.exp()
        return boltz

    @overload
    def _check_temperatures(self, **temperature: Unpack[OneTemperatureKwargs]) -> OneTemperatureKwargs:
        pass

    @overload
    def _check_temperatures(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> RoVibTemperatureKwargs:
        pass

    def _check_temperatures(self, **temperatures: float) -> dict[str, float]:
        if len(temperatures) == 0:
            raise TypeError("Missing temperature argument(s)")

        if len(temperatures) == 1:
            if "T" not in temperatures:
                raise ValueError("If only one temperature is provided, it must be named 'T'")
            return temperatures

        out = {}
        for temp, value in temperatures.items():
            temp = temp.split("_", 1)[1]
            if temp not in self._temperatures:
                raise ValueError(f"Unknown temperature '{temp}', must be one of {self._temperatures}")
            out[self._temperatures[temp]] = value
        return out

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]


class Molecule(Population):
    _must_params = Population._must_params + ("sw",)

    def __init__(self, data: pl.DataFrame | _TypeReadTxtLoc, T_ref: float = 296.0, partition_interpolator = None):
        super().__init__(data, partition_interpolator)
        self.T_ref = T_ref

    def crosssection(self) -> floatNDArray1D:
        """Returns the cross-section of a transition between two states.

        NOTE: CURRENTLY RETURNS THE RAMAN LINE STRENGTH, NOT THE CROSS-SECTION.


        Returns
        -------
        Quantity
            The cross-sections of the transitions.
        """

        return self._data["sw"].to_numpy()

    @overload
    def _relative_boltz_expr(self, **temperature: Unpack[OneTemperatureKwargs]) -> pl.Expr:
        pass

    @overload
    def _relative_boltz_expr(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> pl.Expr:
        pass

    def _relative_boltz_expr(self, **temperatures: float) -> pl.Expr:
        temperatures = self._check_temperatures(**temperatures)
        if len(temperatures) == 1:
            boltz = 1.4387770 * pl.col("E_i") * (1 / self.T_ref - 1 / temperatures["T"])
        else:
            boltz: pl.Expr = 0  # noqa
            for temp_name, temperature in temperatures.items():
                boltz += 1.4387770 * pl.col(f"E_i_{temp_name}") * (1 / self.T_ref - 1 / temperature)
        return boltz.exp()

    @overload
    def intensity(self, **temperature: Unpack[OneTemperatureKwargs]) -> floatNDArray1D:
        pass

    @overload
    def intensity(self, **temperature: Unpack[RoVibTemperatureKwargs]) -> floatNDArray1D:
        pass

    def intensity(self, **temperatures: float) -> floatNDArray1D:
        """Returns the intensity of a transition.

        Returns
        ------------
        np.ndarray:
            The intensity of the transition.
        """
        boltz = self._relative_boltz_expr(**temperatures)
        return self._data.select(pl.col("sw") * boltz)["sw"].to_numpy()


    def intensity_constant(self) -> floatNDArray1D:
        """Returns the constant part of the intensity calculation. These involve physical constants and cross-sections,
        but not the populations.

        CURRENTLY NOT IMPLEMENTED.

        Returns
        ------------
        float: The intensity constant of the transition.
        """
        return self._data['sw'] / self.population(T = self.T_ref)

    def initial_population(self) -> Population:
        """
        Returns a Population instance with the data of the initial state.

        Returns
        -------
        Population
            A Population instance with the initial state data.
        """
        cols = [x for x in self._data.columns if not x.endswith("_f")]
        data_init = self._data.select(cols).unique(self._unique_entries)
        return Population(data_init, T_ref=self.T_ref, partition_interpolator=self._partition_interpolator)

