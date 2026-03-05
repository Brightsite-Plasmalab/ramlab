from typing import Collection
import functools
import operator

import polars as pl


def levels_filter_and(**kwargs: object) -> pl.Expr:
    """
    Create a filter for polars DataFrame based on provided keyword arguments.

    All filters are combined using a logical AND operation.

    Each keyword argument corresponds to a column in the DataFrame. If the key starts with 'd', it indicates a
    difference between final and initial states. If the value is an iterable, it checks if the column value is in the
    iterable. Otherwise, it checks for equality. If no keyword arguments are provided, it returns a filter that always
    evaluates to True.

    For example:
    df.filter(levels_filter(J_i=1, v_i=[0, 1, 2], dJ=1))
    Will result in a filter equivalent to:
    (df["J_i"] == 1) & (df["v_i"].is_in([0, 1, 2])) & ((df["J_f"] - df["J_i"]) == 1)
    """
    return _levels_filter(True, operator.and_, **kwargs)

def levels_filter_or(**kwargs: object) -> pl.Expr:
    """
    Create a filter for polars DataFrame based on provided keyword arguments.

    All filters are combined using a logical OR operation.

    Each keyword argument corresponds to a column in the DataFrame. If the key starts with 'd', it indicates a
    difference between final and initial states. If the value is an iterable, it checks if the column value is in the
    iterable. Otherwise, it checks for equality. If no keyword arguments are provided, it returns a filter that always
    evaluates to False.

    For example:
    df.filter(levels_filter(J_i=1, v_i=[0, 1, 2], dJ=1))
    Will result in a filter equivalent to:
    (df["J_i"] == 1) | (df["v_i"].is_in([0, 1, 2])) | ((df["J_f"] - df["J_i"]) == 1)
    """
    return _levels_filter(False, operator.or_, **kwargs)

def _levels_filter(default: bool, logical_operator, **kwargs: float | Collection[float]) -> pl.Expr:
    if len(kwargs) == 0:
        return pl.lit(default)
    filters = []
    for key, value in kwargs.items():
        if key[0] == "d":
            key = key[1:]
            expr = pl.col(f"{key}_f") - pl.col(f"{key}_i")
        else:
            expr = pl.col(f"{key}")

        if isinstance(value, Collection):
            filters.append(expr.is_in(value))
        else:
            filters.append(expr == value)

    return functools.reduce(logical_operator, filters)