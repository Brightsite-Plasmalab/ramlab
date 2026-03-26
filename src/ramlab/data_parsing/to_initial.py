import polars as pl

from ramlab.data_parsing.missing import difference_check


def to_initial(dataframe):  # TODO: Think about putting it in some other file
    """
    Convert a DataFrame with final and initial state columns to a DataFrame with only initial state columns.

    For each column in the input DataFrame, if the column name ends with '_i' it is considered an initial state column,
    when ending in '_f', it is considered a final state column. The initial and final state columns should contain the
    same parameters (e.g. g_i and g_f). The only exception to this 'E_f', which is calculated as E_i + nu.

    It adds a new column 'origin' to indicate whether the row corresponds to an initial state ('i') or a final state
    ('f').

    Parameters
    ----------
    dataframe: pl.DataFrame
        The input DataFrame containing final and initial state columns.

    Returns
    --------
    pl.DataFrame:
        A new DataFrame containing only initial state columns with values copied from final state columns.

    Raises
    --------
    ValueError: If the initial and final state columns do not match.
    """
    dataframe = dataframe.with_columns(
        (pl.col("E_i") + pl.col("nu")).alias("E_f")
    )

    initial_columns = [col for col in dataframe.columns if col.endswith('_i')]
    final_columns = [col for col in dataframe.columns if col.endswith('_f')]

    def remove_suffix(xs):
        return [x[:-2] for x in xs]

    err_string = "Missing {name1} column{s} in {name2}: '{param}'. "
    missing = difference_check(remove_suffix(initial_columns), remove_suffix(final_columns), err_string,
                               name1="initial state", name2="final state")
    if missing:
        msg = "Both final and initial states should have the same columns. " + missing
        raise ValueError(msg)

    initial_frame = dataframe.select(initial_columns).with_columns(
        origin=pl.lit("i")
    )
    final_frame = dataframe.select(final_columns)
    final_frame = (final_frame.rename({col: col[:-2] + "_i" for col in final_frame.columns})
                              .with_columns(origin=pl.lit("f"))
                              .select(initial_frame.columns))
    return pl.concat([initial_frame, final_frame], how="vertical")
