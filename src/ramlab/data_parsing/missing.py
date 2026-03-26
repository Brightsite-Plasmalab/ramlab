def difference_check(values1, values2, base_string: str, name1: str, name2: str) -> str | None:
    """
    Build a formatted message listing any items from `value` that are present in `values`.

    Parameters
    ----------
    values1 : Iterable[str]
        Items to check for presence (e.g. requested column names or schema entries).
    values2 : Collection[str]
        Available items to check against (e.g. dataframe columns).
    base_string : str
        A format string used to construct the final message. It must contain the placeholders:
        - `{name1}` : the name of the first set of values (e.g. "requested columns")
        - `{name2}` : the name of the second set of values (e.g. "dataframe columns")
        - `{param}` : replaced with the extra parameter name(s)
        - `{s}` : replaced with `''` for singular or `'s'` for plural
        - `{is_are}` : replaced with `'is'` for singular or `'are'` for plural
        If you want extra names quoted, include the quotes in `base_string` (for example `\"'{param}'\"`).
    name1: str
        The name to use for the first set of values in the message (e.g. "requested columns") in `base_string`.
    name2: str
        The name to use for the second set of values in the message (e.g. "dataframe columns") in `base_string`.

    Returns
    -------
    str
        A formatted message describing the missing item(s) when any are missing; otherwise an empty string.

    Notes
    -----
    - The function only checks membership and does not raise; callers should raise an exception if desired.
    - For a single extra item the item is inserted as-is; for multiple extra items they are joined with `\", \"`.
    """
    extra1 = [param for param in values1 if param not in values2]
    extra2 = [param for param in values2 if param not in values1]
    return (missing_message(extra1, base_string, name1=name1, name2=name2)
            + missing_message(extra2, base_string, name1=name2, name2=name1))

def missing_check(value, values, base_string) -> str:
    """
    Build a formatted message listing any items from `value` that are not present in `values`.

    Parameters
    ----------
    value : Iterable[str]
        Items to check for presence (e.g. requested column names or schema entries).
    values : Collection[str]
        Available items to check against (e.g. dataframe columns).
    base_string : str
        A format string used to construct the final message. It must contain the placeholders:
        - `{param}` : replaced with the missing parameter name(s)
        - `{s}` : replaced with `''` for singular or `'s'` for plural
        - `{is_are}` : replaced with `'is'` for singular or `'are'` for plural
        If you want missing names quoted, include the quotes in `base_string` (for example `\"'{param}'\"`).

    Returns
    -------
    str
        A formatted message describing the missing item(s) when any are missing; otherwise an empty string.

    Notes
    -----
    - The function only checks membership and does not raise; callers should raise an exception if desired.
    - For a single missing item the item is inserted as-is; for multiple missing items they are joined with `\", \"`.
    """
    missing = [param for param in value if param not in values]
    return missing_message(missing, base_string)


def missing_message(missing_values, base_string, **format_kwargs) -> str:
    """
    Build a formatted message listing any items that are missing.

    Parameters
    ----------
    missing_values : Sequence[str]
        The missing values
    base_string : str
        A format string used to construct the final message. It must contain the placeholders:
        - `{param}` : replaced with the missing parameter name(s)
        - `{s}` : replaced with `''` for singular or `'s'` for plural
        - `{is_are}` : replaced with `'is'` for singular or `'are'` for plural
        If you want missing names quoted, include the quotes in `base_string` (for example `\"'{param}'\"`).
    format_kwargs: str
        Extra keyword arguments to be passed to `base_string.format()` in addition to the automatically generated
        `param`, `s`, and `is_are` parameters.

    Returns
    -------
    str
        A formatted message describing the missing item(s) when any are missing; otherwise an empty string.

    Notes
    -----
    - For a single missing item the item is inserted as-is; for multiple missing items they are joined with `\", \"`.
    """
    if len(missing_values) == 0:
        return ""
    elif len(missing_values) == 1:
        param = missing_values[0]
        return base_string.format(param=param, s="", is_are="is", **format_kwargs)
    else:
        params = "', '".join(missing_values)
        return base_string.format(param=params, s="s", is_are="are", **format_kwargs)