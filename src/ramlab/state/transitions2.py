from typing import TypeVar, Generic

import numpy as np
import numpy.typing as npt

from .state2 import State

StateType = TypeVar("StateType", bound=State, covariant=True)


class Transitions(Generic[StateType]):
    """
    Container of paired initial/final molecular states and per-transition properties.

    This class holds two State objects (initial and final) of equal length and allows
    vectorized access to their values, the changes between them (prefixed with 'd'),
    and any additional numeric per-transition properties provided by the user.

    Parameters
    ----------
    initial : State
        The initial State. Must have the same keys and length as ``final``.
    final : State
        The final State. Must have the same keys and length as ``initial``.
    **properties
        Additional per-transition numeric 1D arrays. Keys become attributes of the
        Transitions instance. Keys may not collide with reserved State keys based variables
        (``{key}_initial``, ``{key}_final``) nor change names (``d{key}``).

    Notes
    -----
    - For every state key present in the provided State objects, the following dynamic attributes are available
      on Transitions:
        - ``{key}_initial``: values from the initial state.
        - ``{key}_final``: values from the final state.
        - ``d{key}``: difference ``{key}_final-{key}_initial`` (read-only).
    - Additional user properties must be 1D numeric arrays with the same length as
      the states.
    """
    keys: tuple
    state_initial: StateType
    state_final: StateType
    state_keys: tuple

    _length: int
    _attrs = ("state_initial", "state_final", "keys", "state_keys", "_state_values", "_change_values", "_length")
    _state_values: tuple
    _change_values: tuple

    def __init__(self, initial: StateType, final: StateType, **properties: npt.ArrayLike):
        if State._check_keys_different(initial._keys, final._keys):
            msg = f"Initial and final states must have the same keys. Found keys: {initial._keys} and {final._keys}"
            raise ValueError(msg)
        if len(initial) != len(final):
            msg = f"Initial and final states must have the same length. Found lengths: {len(initial)} and {len(final)}"
            raise ValueError(msg)

        self.state_initial = initial
        self.state_final = final
        self._state_values = tuple([
            f"{name}_{suffix}" for name in initial._keys for suffix in ["initial", "final"]
        ])
        self._change_values = tuple([
            f"d{name}" for name in initial._keys
        ])
        self.keys = tuple(properties.keys())
        self._length = len(self.state_initial)

        for key, value in properties.items():
            arr = self._check_arr(value, key)
            if (key in self._state_values) or (key in self._change_values):
                msg = f"'{key}' is reserved for state values."
                raise ValueError(msg)
            setattr(self, key, arr)

        self.state_keys = initial._keys

    def _check_arr(self, value: npt.ArrayLike, name: str) -> np.ndarray:
        """
        Validate and coerce a property value to a 1D numeric numpy array.

        Parameters
        ----------
        value : Any
            The input to convert via ``np.asarray``.
        name : str
            The property name (used for error messages).

        Returns
        -------
        numpy.ndarray
            A 1D numeric array of length ``len(self)``.

        Raises
        ------
        ValueError
            If conversion fails, the length mismatches, dtype is non-numeric,
            or the array is not 1-dimensional.
        """
        try:
            arr = np.asarray(value)
        except Exception as err:
            msg = f"Could not convert property '{name}' to a numpy array."
            raise ValueError(msg) from err

        if len(arr) != self._length:
            msg = f"'{name}' has length {len(arr)}, but transitions has length {self._length}"
            raise ValueError(msg)
        if arr.dtype.kind not in "iuf":
            msg = f"'{name}' must be numeric, found dtype: {arr.dtype.name}"
            raise ValueError(msg)
        if arr.ndim != 1:
            msg = f"'{name}' must be 1-dimensional, found ndim: {arr.ndim}"
            raise ValueError(msg)
        return arr

    def to_states(self):
        """
        Return the underlying initial and final states.

        Returns
        -------
        tuple[State, State]
            A pair ``(state_initial, state_final)``.
        """
        return self.state_initial, self.state_final

    def make_mask(self, **kwargs: list | np.ndarray | int):
        """
        Build a boolean mask by matching values of attributes/properties.

        Each keyword must correspond to a valid attribute on this Transitions
        object, i.e. one of:
        - dynamic state values: ``{name}_initial``, ``{name}_final``
        - dynamic changes: ``d{name}``
        - user-provided properties: keys from ``self.keys``

        The provided value(s) are matched using ``numpy.isin``. Scalars are
        treated as singletons; iterables are converted to lists when needed.

        Parameters
        ----------
        **kwargs : list | numpy.ndarray | int
            Mapping of attribute name to allowed value(s).

        Returns
        -------
        numpy.ndarray
            A boolean array of shape ``(len(self),)`` where True marks rows that
            match all provided constraints.

        Raises
        ------
        ValueError
            If an unknown attribute name is supplied or a value cannot be
            converted to a list for membership checking.
        """
        for key in kwargs.keys():
            if (key not in self._state_values) and (key not in self._change_values) and (key not in self.keys):
                msg = f"'{key}' is not a valid state value or change value."
                raise ValueError(msg)

        total_mask = np.ones(self._length, dtype=bool)
        for key, value in kwargs.items():
            if not isinstance(value, list | np.ndarray | int):
                try:
                    value = list(value)
                except Exception as err:
                    raise ValueError(f"Could not convert '{key}' to a list.") from err
            total_mask &= np.isin(getattr(self, key), value)
        return total_mask

    def filter(self, *, return_mask=False, **kwargs):
        """
        Filter transitions by attribute values using ``make_mask``.

        Parameters
        ----------
        return_mask : bool, default False
            If True, also return the boolean mask used for filtering.
        **kwargs
            Passed to ``make_mask``.
            Each keyword must correspond to a valid attribute on this Transitions
            object, i.e. one of:
            - dynamic state values: ``{name}_initial``, ``{name}_final``
            - dynamic changes: ``d{name}``
            - user-provided properties: keys from ``self.keys``

        Returns
        -------
        Transitions or tuple[Transitions, numpy.ndarray]
            The filtered Transitions. If ``return_mask`` is True, also return the
            mask as a 1D boolean array of length ``len(self)``.
        """
        mask = self.make_mask(**kwargs)
        if return_mask:
            return self[mask], mask
        return self[mask]

    @classmethod
    def from_state_transition(cls, state_initial: State, transition_kwargs, **properties):
        """
        Create Transitions from applying a (range of) transition(s) to a State.

        Parameters
        ----------
        state_initial : State
            The initial State.
        transition_kwargs : dict
            Keyword arguments forwarded to ``State.transition``.
        **properties: array-like
            Additional per-transition 1D numeric arrays to attach.

        Returns
        -------
        Transitions
            A new Transitions instance with the produced initial/final states.
        """
        state_initial, state_final = state_initial.transition(**transition_kwargs)
        return cls(state_initial, state_final, **properties)

    @classmethod
    def from_state_transition_each(cls, state_initial: State, transition_each_kwargs, **properties):
        """
        Create Transitions by applying element-wise transitions to a State.

        Parameters
        ----------
        state_initial : State
            The initial State.
        transition_each_kwargs : dict
            Keyword arguments forwarded to ``State.transition_each``.
        **properties: array_like
            Additional per-transition 1D numeric arrays to attach.

        Returns
        -------
        Transitions
            A new Transitions instance with the produced initial/final states.
        """
        state_initial, state_final = state_initial.transition_each(**transition_each_kwargs)
        return cls(state_initial, state_final, **properties)

    def __getattr__(self, item):
        """
        Provide dynamic access to state values and their changes.

        Parameters
        ----------
        item : str
            Attribute name to resolve. Supported forms are ``{name}_initial``,
            ``{name}_final`` and ``d{name}``.

        Returns
        -------
        numpy.ndarray
            The requested 1D array.

        Raises
        ------
        AttributeError
            If the requested name is ambiguous or invalid.
        """
        if  item in self._state_values:
            name, suffix = item.rsplit("_", 1)
            return getattr(getattr(self, f"state_{suffix}"), name)
        elif item in self._change_values:
            name = item[1:]
            return getattr(self.state_final, name) - getattr(self.state_initial, name)
        elif item in self.state_initial._keys:
            raise AttributeError(f"'{item}' is a State attribute, not Transition attribute, "
                                 f"add '_initial' or '_final' to specify which state.")
        else:
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        """
        Assign to user properties or delegate to underlying states.

        Rules
        -----
        - If ``key`` is in internal attributes, set directly.
        - If ``key`` is a user property, validate and set the numeric array.
        - If ``key`` matches ``{name}_initial`` or ``{name}_final``, write into the
          corresponding underlying State.
        - If ``key`` matches ``d{name}``, raise AttributeError (read-only).

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Value to assign.

        Raises
        ------
        AttributeError
            If the attribute name is unknown or read-only.
        ValueError
            If validation of a user property fails.
        """
        if key in self._attrs:
            super().__setattr__(key, value)
        elif key in self.keys:
            value = self._check_arr(value, key)
            super().__setattr__(key, value)
        elif key in self._state_values:
            name, suffix = key.rsplit("_", 1)
            setattr(getattr(self, f"state_{suffix}"), name, value)
        elif key in self._change_values:
            msg = f"'{key}' is read only."
            raise AttributeError(msg)
        else:
            msg = f"'Transitions' object has no attribute '{key}'."
            raise AttributeError(msg)

    def __getitem__(self, item):
        """
        Index into transitions or retrieve an attribute by name.

        Parameters
        ----------
        item : int | slice | numpy.ndarray | list | str
            - If a string, return the named attribute/property array.
            - Otherwise, return a sliced Transitions with corresponding rows.

        Returns
        -------
        numpy.ndarray or Transitions
            Array when ``item`` is a string; otherwise a new Transitions of the
            selected rows.

        Raises
        ------
        KeyError
            If the string key is not a valid attribute/property.
        IndexError
            If row indexing fails.
        """
        if isinstance(item, str):
            if (item in self.keys) or (item in self._state_values) or (item in self._change_values):
                return getattr(self, item)
            else:
                msg = f"The state has no key {item}, valid keys are: {self.keys}"
                raise KeyError(msg)

        if isinstance(item, int):
            item = [item]

        try:
            return self.__class__(
                self.state_initial[item].copy(),
                self.state_final[item].copy(),
                **{ key: getattr(self, key)[item].copy() for key in self.keys }
            )
        except IndexError as err:
            msg = f"Could not index the Transitions (len = {len(self)}) using: {item} of type {type(item).__name__}"
            raise IndexError(msg) from err

    def __setitem__(self, key, value):
        """
        Assign to a property or state value by name.

        Parameters
        ----------
        key : str
            Name of the property or state value (``{name}_initial``/``{name}_final``).
        value : Any
            Value to assign. For properties, must be a 1D numeric array of the appropriate length.

        Raises
        ------
        KeyError
            If ``key`` is not a known property or state value name.
        AttributeError
            If attempting to assign a change value (``d{name}``).
        ValueError
            If property validation fails.
        """
        if isinstance(key, str):
            if (key in self.keys) or (key in self._state_values):
                setattr(self, key, value)
            else:
                msg = f"The state has no key {key}, valid keys are: {self.keys + self._state_values}"
                raise KeyError(msg)
        else:
            msg = f"The key must be a string, found: {type(key).__name__}"
            raise KeyError(msg)

    def __len__(self):
        """The number of transitions."""
        return self._length

    def unique(self, *, return_index=False, use_properties=False):
        """
        Return unique transitions based on state values and optional properties.

        By default, uniqueness is computed across all state value columns
        (``{name}_initial`` and ``{name}_final`` for every name). Optionally,
        you can include user properties in the uniqueness criterion.

        Parameters
        ----------
        return_index : bool, default False
            If True, also return the indices of the unique rows.
        use_properties : bool | str | list[str], default False
            - If True, include all user property arrays in the uniqueness key.
            - If False, only include state value columns, no user properties.
            - If a string, include just that property.
            - If a list of strings, include those properties.

        Returns
        -------
        Transitions or tuple[Transitions, numpy.ndarray]
            The unique transitions. If ``return_index`` is True, also return the
            indices as a 1D integer array.

        Raises
        ------
        ValueError
            If ``use_properties`` references a property name that does not exist.
        TypeError
            If ``use_properties`` is not a string, list of strings, or boolean.
        """
        base_list = [getattr(self, key) for key in self._state_values]
        if use_properties:
            if isinstance(use_properties, str):
                if use_properties not in self.keys:
                    msg = f"use_properties must be a key in the state, found: {use_properties}"
                    raise ValueError(msg)
                use_properties = [use_properties]
            elif isinstance(use_properties, list):
                for prop in use_properties:
                    if prop not in self.keys:
                        msg = f"use_properties must be a key in the state, found: {prop}"
                        raise ValueError(msg)
            elif use_properties is True:
                use_properties = self.keys
            else:
                msg = f"use_properties must be a string or list of strings, found: {type(use_properties).__name__}"
                raise TypeError(msg)

            base_list += [getattr(self, key) for key in use_properties]
        values = np.array(base_list).T
        _, idx = np.unique(values, axis=0, return_index=True)

        if return_index:
            return self[idx], idx
        return self[idx]
