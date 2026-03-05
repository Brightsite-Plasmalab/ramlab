"""
This module provides functions to generate and manipulate schemas for Polars DataFrames,
read and write data according to these schemas, and handle string parsing and file operations.

Classes
-------
SchemaEntry
    A class representing an entry in a schema with start position, length, format, and data type.

Functions
---------
generate_schema(schema: dict[str, SchemaEntry]) -> pl.DataFrameSchema
    Generate a Polars DataFrame schema from a dictionary of SchemaEntry objects.

read_str(string: str, schema: dict[str, SchemaEntry], entries: Collection[str] = None) -> dict[str, any]
    Read a string according to the provided schema and return the parsed values.

write_line(file_handler, data: dict[str, any], schema: dict[str, SchemaEntry], delimiter: str = ',', end: str = '\n')
    Write a line to a file according to the provided schema.

read_data(loc: str or os.PathLike, schema: dict[str, SchemaEntry], entries: Collection[str] = None) -> dict[str, list[any]]
    Read data from a file according to the provided schema.

write_csv(loc: str or os.PathLike, data: dict[str, Sequence[any]], schema: dict[str, SchemaEntry], delimiter: str = ',')
    Write data to a CSV file according to the provided schema.
"""

from typing import Any, TextIO, Mapping, Literal, assert_never
from collections.abc import Collection, Sequence
import functools
import operator
import warnings
import re
import os

import polars as pl

SchemaDict = dict[str, 'SchemaEntry']


class SchemaEntry:  #TODO: change to frozen (data)class, with offset function to change the start value
    """
    A class representing an entry in a schema with start position, length, format, and data type.

    Attributes
    ----------
    start : int
        The start position of the entry in the string.
    fmt : str
        The format of the entry.
    dtype : pl.DataType
        The data type of the entry.
    length : int
        The length of the entry in characters.
    explanation: str | None
        The explanation for the meaning of the entry.
    """
    def __init__(self, start: int, fmt: str, dtype: pl.DataType, length: int, explanation: str | None = None,
                 mapping: dict | Collection | None = None):
        self.start = start
        self.fmt = fmt
        self.dtype = dtype
        self.length = length
        self.explanation = explanation
        if isinstance(mapping, Collection):
            mapping = {i: v for i, v in enumerate(mapping)}
        self.mapping = mapping

    def copy(self):
        """
        Create a copy of the SchemaEntry instance.

        Returns
        -------
        SchemaEntry
            A new instance of SchemaEntry with the same attributes.
        """
        return SchemaEntry(self.start, self.fmt, self.dtype, self.length, self.explanation)

    def __repr__(self):
        return (f"SchemaEntry(start={self.start}, fmt='{self.fmt}', dtype={self.dtype}, length={self.length},"
                f" explanation={self.explanation})")

    @property
    def end(self):
        return self.start + self.length


class HitranSchemaEntry(SchemaEntry):
    def __init__(self, start: int, fmt: str, explanation: str = None, mapping: tuple = None):
        match = re.search(r'(\d+)', fmt)
        if match:
            length = int(match.group(1))
        else:
            msg = f"Invalid format specifier: {fmt}"
            raise ValueError(msg)

        match fmt[-1]:
            case 's':
                dtype = pl.Utf8
            case 'e' | 'f':
                dtype = pl.Float64
            case "d":
                if length <= 2:
                    dtype = pl.UInt8
                elif length <= 4:
                    dtype = pl.UInt16
                elif length <= 8:
                    dtype = pl.UInt32
                elif length <= 16:
                    dtype = pl.UInt64
                else:
                    msg = f"Unsupported length for integer format string {fmt}: {length}. Maximum supported is 16."
                    raise ValueError(msg)
            case _:
                msg = (f"Unsupported format specifier letter: `{fmt}`: '{fmt[-1]}',"
                       f" supported are 's', 'f', 'e', and 'd'.")
                raise ValueError(msg)
        super().__init__(start, fmt, dtype, length, explanation)

    def copy(self):
        """
        Create a copy of the HitranSchemaEntry instance.

        Returns
        -------
        HitranSchemaEntry
            A new instance of HitranSchemaEntry with the same attributes.
        """
        return HitranSchemaEntry(self.start, self.fmt, self.explanation)

    def format_string(self) -> str:
        """
        Generate a format string for the SchemaEntry.

        Returns
        -------
        str
            A format string based on the entry's format and length.
        """
        match self.fmt:
            case 's':
                return f"<{self.length}"
            case 'e' | 'f':
                return f"{self.length}.{self.length - 7}e"
            case 'd':
                return f"{self.length}d"
            case _:
                raise ValueError(f"Unsupported format specifier: {self.fmt}")

    def parse(self, value):
        """
        Parse a string value according to the entry's data type.

        Parameters
        ----------
        value : str
            The string value to be parsed.

        Returns
        -------
        any
            The parsed value in the appropriate data type.
        """
        if self.dtype == pl.Utf8:
            return value.strip()
        elif self.dtype == pl.Float64:
            return float(value)
        elif self.dtype in {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            return int(value)
        else:
            assert_never("Unsupported data type for parsing.")

    def __repr__(self):
        return f"HitranSchemaEntry(start={self.start}, fmt='{self.fmt}')"


def add_subschema(schema: dict[str, SchemaEntry],* , del_old=True, **subschemas: dict[str, SchemaEntry]):
    """
    Add a subschema to the main schema.

    Parameters
    ----------
    schema : dict[str, SchemaEntry]
        The main schema where the subschema will be added.
    del_old : bool, optional
        If True, the parent schema entry will be deleted after adding the subschema. Default is True.
    subschemas : dict[str, SchemaEntry]
        The subschema to be added. The kwarg name is the name of the parent schema entry.
    """
    for name, subschema in subschemas.items():
        if name not in schema:
            msg = f"Subschema name '{name}' not found in the main schema. Available entries: {tuple(schema.keys())}."
            raise ValueError(msg)
        for sub_name in subschema:
            if sub_name in schema:
                msg = f"Subschema entry '{sub_name}' already exists in the main schema."
                raise ValueError(msg)

        entry = schema[name]
        offset = entry.start
        for key, value in subschema.items():
            value.start += offset
            if (value.length + value.start) > (entry.start + entry.length):
                msg = f"Subschema entry '{key}' exceeds the length of the parent schema entry '{name}'."
                raise ValueError(msg)
            subschema[key] = value
        schema.update(subschema)
        if del_old:
            del schema[name]
    return schema


def generate_schema(schema: dict[str, SchemaEntry]):
    """
    Generate a Polars DataFrame schema from a dictionary of SchemaEntry objects.

    Parameters
    ----------
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.

    Returns
    -------
    pl.DataFrameSchema
        A Polars DataFrame schema.
    """
    return pl.DataFrameSchema({k: v.dtype for k, v in schema.items()})


def read_str(string, schema: dict[str, SchemaEntry], entries: Collection[str] = None):
    """
    Read a string according to the provided schema and return the parsed values.

    Parameters
    ----------
    string : str
        The input string to be parsed.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    entries : Collection[str], optional
        A collection of entries to be read from the string. If None, all entries in the schema are read.

    Returns
    -------
    dict[str, any]
        A dictionary with parsed values.
    """
    entries = _check_entries(schema, entries)
    return _read_str(string, schema, entries)


def _read_str(string, schema: dict[str, SchemaEntry], entries: Collection[str] = None):
    """
    Helper function to read a string according to the provided schema and return the parsed values.

    Parameters
    ----------
    string : str
        The input string to be parsed.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    entries : Collection[str], optional
        A collection of entries to be read from the string. If None, all entries in the schema are read.

    Returns
    -------
    dict[str, any]
        A dictionary with parsed values.
    """
    #TODO implement using the mapping of HitranSchemaEntry
    return {k: v.dtype.to_python(string[v.start:v.start + v.length]) for k, v in schema.items() if k in entries}


def _write_line(file_handler: TextIO, data: list[str], delimiter=',', end='\n'):
    file_handler.write(delimiter.join(data) + end)


def write_line(file_handler: TextIO, data: dict[str, Any], schema: dict[str, SchemaEntry], delimiter=',', end='\n'):
    """
    Write a line to a file formatted according to the schema, but does not take position into account.

    Parameters
    ----------
    file_handler : file-like object
        The file handler to write the line to.
    data : dict[str, any]
        A dictionary with data to be written.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    delimiter : str, optional
        The delimiter to use between values. Default is ','.
    end : str, optional
        The end character to use after the line. Default is '\n'.
    """
    #TODO implement using the mapping of HitranSchemaEntry
    sorted_schema = dict(sorted(schema.items(), key=lambda x: x[1].start))
    _write_line(file_handler, [f"{data[k]:{schema[k].fmt}}" for k in sorted_schema], delimiter=delimiter, end=end)


def write_txt(loc, data: pl.DataFrame, schema: dict[str, SchemaEntry], null_fill: str = ''):
    """
    Write data to a text file according to the provided schema.

    Parameters
    ----------
    loc : str or os.PathLike
        The location of the file to write.
    data : dict[str, Sequence[any]]
        A dictionary with data to be written.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    null_fill: str, optional
        The string to use for null values. Default is ''.

    Notes
    ------
    This function only works when all entries from the schema are present in the data.
    """
    #TODO implement using the mapping of HitranSchemaEntry
    sorted_entries = list(sorted(schema.items(), key=lambda x: x[1].start))
    result = []
    for key, entry in sorted_entries:
        # Cast value to string, pad with spaces to the left to ensure correct length, and handle null values
        result.append((
            pl.when(pl.col(key).is_null())
                .then(pl.lit(null_fill))
                .otherwise(pl.col(key).cast(pl.String).str.pad_start(entry.length))
        ).alias(key))
    adder = functools.reduce(operator.add, [pl.col(key) for key, entry in sorted_entries]).alias("map")
    data = data.with_columns(
        *result
    ).with_columns(adder).select("map")

    with open(loc, "w") as file:
        for line in data['map']:
            if line is None:
                raise ValueError("Line is None, something went wrong in the formatting process.")
            file.write(line + "\n")


def _check_entries(schema: dict[str, SchemaEntry], entries: Collection[str]):
    """
    Check if the provided entries are in the schema.

    Parameters
    ----------
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    entries : Collection[str]
        A collection of entries to be checked.

    Returns
    -------
    list[str]
        A list of valid entries.

    Raises
    ------
    ValueError
        If any entry is not in the schema.
    """
    if entries is None:
        return list(schema.keys())
    for entry in entries:
        if entry not in schema:
            raise ValueError(f"Entry {entry} not in schema, entries in schema are: {tuple(schema.keys())}.")
    return entries


def read_data(loc, schema: dict[str, SchemaEntry], entries: Collection[str] | Mapping[str, str] = None,
              filter_expr: pl.Expr = None, strict=False, collect=False):
    """
    Read data from a file according to the provided schema.

    Parameters
    ----------
    loc : str or os.PathLike
        The location of the file to read.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    entries : Collection[str] or Mapping[str, str], optional
        A collection of entries to be read from the file.
        If a mapping is provided, the keys are the entries to read and the values are the desired column names in the output DataFrame.
        If None, all entries in the schema are read.
    filter_expr : pl.Expr, optional
        A Polars expression to filter the data. Default is None, which means no filtering.
    strict : bool, optional
        Sets the strict parameter on the polars cast expression
    collect : bool, optional
        Whether to collect the LazyFrame to a DataFrame

    Returns
    -------
    polars.LazyFrame | polars.DataFrame
        A Polars DataFrame with the parsed data.

    Raises
    -------
    ValueError
        If the length of the schema is longer than the length of the first line
    """
    if isinstance(entries, Mapping):
        name_mapping = dict(entries)
        entries = list(name_mapping.keys())
    else:
        name_mapping = None

    entries = _check_entries(schema, entries)
    if name_mapping is None:
        name_mapping = {entry: entry for entry in entries}

    # checks for a out of bounds, i.e. that the schema entries do not exceed the line length
    total_end = max( schema[entry].end for entry in entries )
    with open(loc, 'r') as f:
        first_line = f.readline()
        if total_end > len(first_line):
            msg = f"The length of the schema ({total_end}) is bigger than the line length ({len(first_line)})."
            raise ValueError(msg)

    # the parsing is split in two steps, so that if there is an error in the casting, the error message contains the
    # name of the column that caused the error.
    df = pl.scan_csv(loc, has_header=False).with_columns(
        pl.col("column_1")
            .str.slice(schema[entry].start, schema[entry].length)
            .str.strip_chars(' ')
            .alias(name_mapping[entry])
        for entry in entries
    ).with_columns(
        *(pl.col(name_mapping[entry]).cast(schema[entry].dtype, strict=strict)
        for entry in entries),
        *(pl.col(name_mapping[entry]).replace(schema[entry].mapping)
        for entry in entries if schema[entry].mapping is not None)
    )
    df = df.drop("column_1")

    if filter_expr is not None:
        df = df.filter(filter_expr)

    if collect:
        df = df.collect()
    return df


def write_csv(loc, data: dict[str, Sequence[Any]], schema: dict[str, SchemaEntry], *, delimiter=',', write_header=True):
    """
    Write data to a CSV file according to the provided schema.

    Parameters
    ----------
    loc : str or os.PathLike
        The location of the file to write.
    data : dict[str, Sequence[any]]
        A dictionary with data to be written.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    delimiter : str, optional
        The delimiter to use between values. Default is ','.
    write_header : bool, optional
        Whether to write a header row. Default is True.
    """
    with open(loc, "w") as file:
        if write_header:
            _write_line(file, list(schema.keys()))
        for i in range(len(data[list(schema.keys())[0]])):
            write_line(file, {k: v[i] for k, v in data.items()}, schema, delimiter=delimiter)


def stream_file_to_csv(loc_in, loc_out, schema: dict[str, SchemaEntry], entries: Collection[str] = None):
    """
    Read data from a file according to the provided schema and write it to a CSV file.

    Parameters
    ----------
    loc_in : str or os.PathLike
        The location of the input file.
    loc_out : str or os.PathLike
        The location of the output CSV file.
    schema : dict[str, SchemaEntry]
        A dictionary where keys are column names and values are SchemaEntry objects.
    entries : Collection[str], optional
        A collection of entries to be read from the file. If None, all entries in the schema are read.
    """
    with open(loc_in, "r") as file, open(loc_out, "w") as file_out:
        for line in file:
            data = _read_str(line, schema, entries)
            write_line(file_out, data, schema)


def read_csv(loc, entries: Collection[str]=None, scan_csv_kwargs: dict[str, Any] = None):
    """
    Reads a CSV file and selects specified columns.

    This function reads data from a CSV file using the provided file location and optional parameters. It allows
    filtering specific columns from the CSV file by passing their names. Additional keyword arguments for reading the
    CSV file can also be specified.

    Parameters
    ----------
    loc: os.PathLike or str
        The location of the CSV file as a string.
    entries: Collection[str]
        A collection of column names to select from the CSV file
        (optional). If none is provided, no filtering is done.
    scan_csv_kwargs: dict[str, Any]
        A dictionary of additional keyword arguments passed
        to the scan_csv function (optional).

    Returns
    --------
    A Polars LazyFrame containing the filtered CSV data.

    Raises
    --------
    KeyError
        If specified entries are not found in the CSV file.
    """
    scan_csv_kwargs = scan_csv_kwargs or {}
    data = pl.scan_csv(loc, **scan_csv_kwargs)
    data = data.select(entries)
    return data

def write_avro(file, data: pl.DataFrame, entries: Collection[str],
               compression: Literal['snappy', 'deflate', 'uncompressed'] = 'snappy'):
    """
    Convert a Polars DataFrame to a string according to the provided schema.

    Parameters
    ----------
    file: str or os.PathLike
        The location of the Avro file to write.
    data : pl.DataFrame
        The Polars DataFrame to be converted.
    entries: Collection[str]
        A collection of column names to save to the file.
    compression: Literal['snappy', 'deflate', 'uncompressed'], optional
        The compression algorithm to use for the Avro file. Default is 'snappy'.

    Notes
    -----
    Not all polars data types are supported in Avro files. At least u8 is not supported.
    """
    data = data.select(entries)
    data.write_avro(file, compression=compression)

def read_avro(file, entries: Collection[str] = None, n_rows: int = None) -> pl.DataFrame:
    """
    Read an Avro file and return a Polars DataFrame.

    Parameters
    ----------
    file : str or os.PathLike
        The location of the Avro file to read.
    entries : Collection[str], optional
        A collection of column names to select from the Avro file. If None, all columns are read.
    n_rows : int, optional
        The number of rows to read from the Avro file. If None, all rows are read.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the data from the Avro file.
    """
    return pl.read_avro(file, columns=entries, n_rows=n_rows)

def schema_tester(file, schema: dict[str, HitranSchemaEntry], line_num=1):
    """
    Test if a txt file follows a given schema.
    """
    max_val = max(entry.end for entry in schema.values()) + 1

    exceptions = []
    with open(file, "r") as f:
        line = f.readline()
        print(f"Full line:\n{line}", end="")
        for i in range(1, line_num):
            print(f.readline(), end="")
        parts = [' ']*max_val
        names = [' ']*max_val
        for name, entry in schema.items():
            parts[entry.start] = "|"
            for i in range(entry.start + 1, entry.end):
                parts[i] = "-"
            parts[entry.end-1] = "|"

            offset = 0
            if len(name) < entry.length:
                offset = 1
            formatted_name = f"{name:^{entry.length-offset}}"
            for i in range(entry.start + offset, entry.end):
                names[i] = formatted_name[i - (entry.start + offset)]
        print("".join(parts))
        print("".join(names))

        f.seek(0)
        line = f.readline()
        for name, entry in schema.items():
            value_str = line[entry.start:entry.start + entry.length]
            print(f"{name}: '{value_str}'")
            try:
                entry.parse(value_str)
            except Exception as e:
                msg = f"Entry '{name}': '{value_str}' with schema {entry}.\n{e}"
                err = ValueError(msg)
                err.__cause__ = e
                exceptions.append(err)
    if exceptions:
        raise ExceptionGroup(f"Schema test failed for file {file}.", exceptions)

def schema_to_code(schema: dict[str, HitranSchemaEntry]):
    key_list = sorted(schema.keys(), key=lambda x: schema[x].start)
    output = "{"
    for key in key_list:
        output += f"    '{key}': {schema[key]},\n"
    output += "}"
    return output