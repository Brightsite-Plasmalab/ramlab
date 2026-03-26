from typing import ClassVar
from collections.abc import Iterable, Collection, Sequence
import os

import polars as pl

from ramlab.data_parsing.reader_generator import read_data, write_txt, SchemaDict
from ramlab.data_parsing.missing import missing_message, missing_check

_TypeReadTxtLoc = str | os.PathLike | Collection[str | os.PathLike]
_TypeReadTxtLoc_tester = str | os.PathLike | Collection


class DataHandler:
    _default_schema: ClassVar[SchemaDict]
    _must_params: ClassVar[tuple[str, ...]]

    _data: pl.DataFrame

    def __init__(self, data: pl.DataFrame | _TypeReadTxtLoc):
        if isinstance(data, _TypeReadTxtLoc_tester):
            data = read_data(data, schema=self._default_schema)
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        elif not isinstance(data, pl.DataFrame):
            msg = f"Data must be a polars DataFrame, LazyFrame, a file path, or a collection of file paths, got {type(data)}."
            raise TypeError(msg)

        self._data = data

        msg = missing_check(self._must_params, self._data.columns,
                            "Missing required parameter{s} {param} in read data.")
        if msg:
            raise ValueError(msg)

        super().__init__()

    @classmethod
    def from_txt(cls, loc: _TypeReadTxtLoc, *, entries: Collection[str]=None,
                 filter_expr=None, strict=True, schema = None) -> pl.DataFrame:
        """Read data from a file according to the provided schema.

        Parameters
        ----------
        loc:
            The location of the file to read.
        entries : Collection[str]
            The entries to read from the file. Must include all entries in `cls._must_params`.
            If None, all entries in the schema are read.
        filter_expr : pl.Expr, optional
            A Polars expression to filter the data. Default is None, which means no filtering.
        strict : bool, optional
            Sets the strict parameter on the polars cast expression. Default is True.
        schema : dict, optional
            The schema to use for reading the data. If None, it uses the classes `_default_schema`.
        """
        if schema is None:
            schema = cls._default_schema

        def read(file_loc, collect=True):
            return read_data(file_loc, schema=schema, entries=entries, filter_expr=filter_expr, strict=strict,
                             collect=collect)

        if not isinstance(loc, (str, os.PathLike)):
            datas = []
            for single_loc in loc:
                data = read(single_loc, collect=False)
                datas.append(data)
            data = pl.concat(datas, rechunk=True, parallel=True).collect()
        else:
            data = read(loc)
        return data

    @classmethod
    def from_parquet(cls, loc: str | os.PathLike, *, entries: Collection[str]=None, filter_expr=None) -> pl.DataFrame:
        """Read data from a Parquet file.

        Parameters
        ----------
        loc : str | os.PathLike
            The location of the file to read.
        entries : Collection[str]
            The entries to read from the file. If None, all entries in the file are read.
        filter_expr : pl.Expr, optional
            A Polars expression to filter the data. Default is None, which means no filtering.
        """
        data = pl.read_parquet(loc)
        if entries is not None:
            msg = missing_check(entries, data.columns, "Column{s} {param} from entries {is_are} not in read data.")
            if msg:
                raise ValueError(msg)
            data = data.select(entries)
        if filter_expr is not None:
            data = data.filter(filter_expr)
        return data

    def write_txt(self, loc, entries: Collection[str]=None, schema: SchemaDict=None):
        """Writes the data to a text file.

        Parameters
        ----------
        loc : str or os.PathLike
            The location of the file to write.
        entries : Collection[str]
            The entries to write to the file. If None, all entries in the schema are written.
        schema : dict, optional
            The schema to use for writing the data. If None, it uses the classes `_default_schema`.
        """
        if schema is None:
            schema = self._default_schema
        if entries is None:
            entries = list(schema.keys())
        msg = missing_check(entries, self._data.columns, "Cannot write missing column{s} {param}.")
        if msg:
            raise ValueError(msg)
        schema = {k: v for k, v in schema.items() if k in entries}

        write_txt(loc, self._data, schema)

    def write_parquet(self, loc, entries: Collection[str]=None, **parquet_kwargs):
        """Writes the data to a Parquet file.

        Parameters
        ----------
        loc : str or os.PathLike
            The location of the file to write.
        entries : Collection[str]
            The entries to write to the file. If None, all entries in the schema are written.
        parquet_kwargs : dict
            Additional keyword arguments to pass to the Polars `write_parquet` method.
        """
        if entries is None:
            entries = self._data.columns

        msg = missing_check(entries, self._data.columns, "Cannot write missing column{s} {param} to Parquet file.")
        if msg:
            raise ValueError(msg)

        data = self._data.select(entries)
        data.write_parquet(loc, **parquet_kwargs)

    def _filter_data(self, data_filter: pl.Expr) -> pl.DataFrame:
        if data_filter is not None:
            return self._data.filter(data_filter)
        else:
            return self._data
