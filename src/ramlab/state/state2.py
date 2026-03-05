import numpy as np

class State:
    """Container for synchronized 1D numeric arrays representing a molecular state.

    Notes
    -----
    - All attributes must be 1D numeric NumPy arrays (or array-like) of equal length.
    - Keys are fixed after initialization; use `add` to add new attributes with matching length.
    - Many dunder methods mirror NumPy-like semantics (indexing, concatenation, equality by element).
    """
    _keys: tuple
    _aliases: dict = {}

    def __init__(self, **state):
        """Create a State from named 1D numeric arrays.

        Parameters
        ----------
        **state: np.ndarray or list
            Mapping from attribute name to a 1D numeric array. All arrays must have the same length.

        Raises
        ------
        ValueError
            If any value cannot be converted to a 1D numeric array, or if the lengths differ.
        """
        self._keys = tuple(state.keys())

        for key, value in state.items():
            arr = self._check_arr(value, key)
            super().__setattr__(key, arr)

        vals = {key: getattr(self, key) for key in self._keys}
        self._raise_lengths(
            vals,"All state variables should have the same lengths, found lengths: ")

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    @property
    def aliases(self) -> dict[str, str]:
        """
        A copy of the aliases dictionary
        """
        return self._aliases.copy()


    @staticmethod
    def _raise_lengths(values: dict, msg: str) -> None:
        """Validate that all arrays share the same length.

        Parameters
        ----------
        values : dict
            Mapping from name to 1D arrays.
        msg : str
            Base message used when raising the error; details are appended.

        Raises
        ------
        ValueError
            If any lengths differ.
        """
        lengths = [len(value) for value in values.values()]
        if not all(l == lengths[0] for l in lengths[1:]):
            for length, name in zip(lengths, values.keys()):
                msg += f"{name}: {length}, "
            msg = msg[:-2] + "."
            raise ValueError(msg)

    @staticmethod
    def _check_arr(value, name):
        """Convert input to 1D numeric NumPy array and validate.

        Parameters
        ----------
        value : array-like
            Input to convert to NumPy array.
        name : str
            Attribute name used for informative error messages.

        Returns
        -------
        numpy.ndarray
            A 1D numeric NumPy array view/copy of the input.

        Raises
        ------
        ValueError
            If the conversion fails, the array is not 1D, or the dtype is non-numeric.
        """
        try:
            arr = np.asarray(value)
        except Exception as err:
            msg = f"Could not convert '{name}' to a numpy array."
            raise ValueError(msg) from err
        if arr.ndim != 1:
            msg = f"'{name}' must be 1-dimensional. Found ndim: {arr.ndim}."
            raise ValueError(msg)
        if arr.dtype.kind not in "iuf":
            msg = f"'{name}' must be numeric. Found dtype: {arr.dtype.name}"
            raise ValueError(msg)
        return arr

    def __len__(self):
        """Number of rows in the state.

        Returns
        -------
        int
            The number of elements for each attribute (common length).
        """
        if len(self.keys) == 0:
            return 0
        return len(getattr(self, self.keys[0]))

    def __setattr__(self, key, value):
        def setitem():
            nonlocal value, key, self
            arr = self._check_arr(value, key)
            if len(value) == len(self):
                super(State, self).__setattr__(key, arr)
            else:
                msg = f"{key} has length {len(value)}, but state has length {len(self)}"
                raise ValueError(msg)

        if key in ("_keys", "_aliases"):
            super().__setattr__(key, value)
        elif key in self._keys:
            setitem()
        elif key in self._aliases:
            key = self._aliases[key]
            setitem()
        else:
            msg = f"'State' object has no attribute '{key}'. When adding a new attribute, use `add`."
            raise AttributeError(msg)

    def __getattr__(self, item):
        if item in self._keys:
            return getattr(self, item)
        if item in self._aliases:
            return getattr(self, self._aliases[item])
        raise AttributeError(f"'State' object has no attribute '{item}'")

    def __getitem__(self, item):
        """Retrieve attribute array by name or slice rows to a new State.

        Parameters
        ----------
        item : str or int or slice or sequence of int/bool
            - If str: return the corresponding attribute array.
            - Else: index rows and return a new State with copied data.

        Returns
        -------
        numpy.ndarray or State
            The attribute array (for str) or a new State instance with selected rows.

        Raises
        ------
        KeyError
            If an unknown key string is provided.
        IndexError
            If row indexing fails.
        """
        if isinstance(item, str):
            if item in self._keys:
                return getattr(self, item)
            if item in self._aliases:
                return getattr(self, self._aliases[item])
            else:
                msg = f"The state has no key {item}, valid keys are: {self._keys}"
                raise KeyError(msg)

        if isinstance(item, int):
            item = [item]

        try:
            return self.__class__(
                **{key: getattr(self, key)[item].copy() for key in self._keys}
            )
        except IndexError as err:
            msg = f"Could not index the State (len = {len(self)}) using: {item} of type {type(item).__name__}"
            raise IndexError(msg) from err

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if (key in self._keys) or (key in self._aliases):
                setattr(self, key, value)
            else:
                msg = f"The state has no key {key}, valid keys are: {self._keys}. When adding a new key, use `add`."
                raise KeyError(msg)
        else:
            msg = f"The key must be a string, found: {type(key).__name__}"
            raise KeyError(msg)

    def __repr__(self):
        if len(self) == 1:
            state_str = (
                f"{name} = {getattr(self, name)[0]}" for name in self._keys
            )
            return f"{self.__class__.__name__}({', '.join(state_str)})"
        longer = "..." if self._state_lengths > 10 else ""
        state_str = (
            f"{name} = [{' '.join((str(x) for x in getattr(self, name)[:10]))} {longer}]" for name in self._keys
        )
        if len(self) >= 10:
            addition = f" of length {len(self)}"
        else:
            addition = ""
        return f"{self.__class__.__name__}({', '.join(state_str)})" + addition

    def _raise_keys_different(self, other):
        if self._check_keys_different(self._keys, other.keys):
            key_err_msg = f"The two states have different keys: {self._keys} and {other.keys}"
            raise ValueError(key_err_msg)

    @staticmethod
    def _check_keys_different(keys1, keys2):
        return len(keys1) != len(keys2) or any((k not in keys1) for k in keys2)

    def __add__(self, other):
        self._raise_keys_different(other)
        return self.__class__(
            **{k: np.concatenate([getattr(self, k), getattr(other, k)]) for k in self._keys}
        )

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        self._raise_keys_different(other)
        if len(self) != len(other) and len(self) != 1 and len(other) != 1:
            msg = (f"Cannot compare two State with different lengths, unless one has length 1. "
                   f"Found lengths {len(self)} and {len(other)}")
            raise ValueError(msg)

        mask = np.ones(len(self), dtype=bool)
        for key in self._keys:
            mask &= getattr(self, key) == getattr(other, key)
        return mask

    def __copy__(self):
        return self.__class__(**{key: getattr(self, key).copy() for key in self._keys})

    def copy(self):
        """Shallow copy of the state (arrays are copied).

        Returns
        -------
        State
            A new State with copied arrays.
        """
        return self.__copy__()

    def add(self, **kwargs):
        """Add new 1D numeric attributes to the state.

        Parameters
        ----------
        **kwargs
            Mapping of new attribute names to arrays. Each array must be length ``len(self)``.

        Raises
        ------
        ValueError
            If any provided array has the wrong length.
        """
        for key, value in kwargs.items():
            if len(self._keys) != 0 and len(value) != len(self):
                msg = f"New attribute '{key}' has length {len(value)}, but state has length {len(self)}"
                raise ValueError(msg)

        # Only add new values once all have been checked
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            self._keys += (key,)

    def add_each(self, **kwargs):
        """Cartesian-add a new attribute by repeating existing rows.

        Parameters
        ----------
        **kwargs
            Exactly one key-value pair where the key is the new attribute name
            and the value is a 1D array of values to expand with. If multiple
            pairs are passed, the operation is applied recursively in order.

        Returns
        -------
        State
            A new State where each existing row is repeated for every value in
            the provided array(s), and the new attribute(s) are filled accordingly.

        Raises
        ------
        ValueError
            If a provided key already exists in the state.
        """
        if len(kwargs) == 0:
            return self
        if len(kwargs) > 1:
            # If multiple attributes are provided, recursively call add_each
            a = self
            for k, v in kwargs.items():
                a = a.add_each(**{k: v})
            return a

        k, v = kwargs.popitem()

        if k in self._keys:
            msg = f"Key '{k}' already exists in the state. Did you mean to use `transition_each`?"
            raise ValueError(msg)

        # Expand the current state such that each attribute is repeated N times,
        # where N is the length of the new attribute
        values = {
            key: np.tile(getattr(self, key), len(v)) for key in self._keys
        }
        # Add the new attribute
        len_self = len(self) if len(self) > 0 else 1
        values[k] = np.repeat(v, len_self)

        return State(**values)

    def transition(self, **kwargs):
        """Create paired initial/final states for additive transitions.

        For each provided attribute ``k`` with values ``v`` (1D array), the
        final state is formed by tiling the original arrays and adding each value
        in ``v`` to the corresponding attribute. Attributes not listed are simply
        tiled. If multiple attributes are provided, they must all share the same
        length and transitions are applied per index (not Cartesian product).

        Parameters
        ----------
        **kwargs
            Mapping from existing attribute names to 1D arrays of additive deltas.

        Returns
        -------
        (State, State)
            A tuple of (initial_state, final_state), both expanded by tiling.

        Raises
        ------
        ValueError
            If no attributes provided, a key is unknown, or lengths mismatch.
        """
        if len(kwargs) == 0:
            raise ValueError("No attributes provided to transition.")

        if len(kwargs) >= 2:
            self._raise_lengths(
                kwargs, "All state variables should have the same lengths, found lengths: ")
        add_length = len(kwargs[next(iter(kwargs.keys()))])

        for key in kwargs.keys():
            if key not in self._keys:
                raise ValueError(f"Key '{key}' doesn't exists in the state.")

        initial = {}
        final = {}

        for key in self._keys:
            if key not in kwargs:
                new = np.tile(getattr(self, key), add_length)
                initial[key] = new.copy()
                final[key] = new
            else:
                old = getattr(self, key)
                initial[key] = np.tile(old, add_length)
                final[key] = (old + kwargs[key][:, None]).flatten()

        return self.__class__(**initial), self.__class__(**final)


    def transition_each(self, **kwargs):
        """
        Cartesian transitions across multiple attributes.

        Applies additive transitions for each provided attribute independently,
        producing the Cartesian product of transitions. This is equivalent to
        chaining ``transition`` per attribute in sequence.

        Parameters
        ----------
        **kwargs
            Mapping from existing attribute names to 1D arrays of additive deltas.

        Returns
        -------
        State
            The tiled original state (initials, repeated per combination).
        State
            The transitioned state (finals, after applying sums per combination).

        Raises
        ------
        ValueError
            If no attributes are provided.
        
        Example
        -------
        >>> state = State(v=[0], J=[0])
        >>> init, final = state.transition_each(v=[1, 2], J=[1, 2, 3])
        >>> init
        State(v = [0 0 0 0 0 0], J = [0 0 0 0 0 0])
        >>> final
        State(v = [1 1 1 2 2 2], J = [1 2 3 1 2 3])
        """
        if len(kwargs) == 0:
            raise ValueError("No attributes provided to transition.")

        # If multiple attributes are provided, recursively call transition_each
        if len(kwargs) >= 2:
            # Recursively apply transition_each for more than one kwarg
            k1, v1 = kwargs.popitem()
            a, b = self.transition(**{k1: v1})
            a, _ = a.transition_each(**kwargs)
            _, b = b.transition_each(**kwargs)
            return a, b
        return self.transition(**kwargs)

    def unique(self, return_index=False):
        """
        Return unique rows of the state.

        Parameters
        ----------
        return_index : bool, default False
            If True, also return the indices of the first occurrences.

        Returns
        -------
        State
            A state containing only unique rows.
        numpy.ndarray, optional
            Indices of the first occurrence of each unique row (if ``return_index`` is True).
        """
        values = np.array(
            [getattr(self, key) for key in self._keys]
        ).T
        _, idx = np.unique(values, axis=0, return_index=True)

        if return_index:
            return self[idx], idx
        return self[idx]

    def equivalent(self, other):
        """
        Check if two states contain the same rows, up to permutation.

        Parameters
        ----------
        other : State
            Another state with the same keys.

        Returns
        -------
        bool
            True if both states have identical multisets of rows.
        """
        if self._check_keys_different(self._keys, other._keys):
            return False
        if len(self) != len(other):
            return False

        values = np.asarray([getattr(self, x) for x in self._keys]).T
        values2 = np.asarray([getattr(other, x) for x in self._keys]).T

        max1 = np.max(values, axis=0)
        max2 = np.max(values2, axis=0)
        max_vals = [max(max1[i], max2[i])+1 for i in range(len(max1))]
        cumsum = [1] + list(np.cumprod(max_vals)[:-1])

        sorter1 = np.argsort(np.sum(values * cumsum, axis=1))
        sorter2 = np.argsort(np.sum(values2 * cumsum, axis=1))

        return np.all(values[sorter1] == values2[sorter2])

    def append(self, state=None, /, **values):
        """
        Append rows from another state or from provided arrays.

        Parameters
        ----------
        state : State, optional
            Another state with identical keys. If provided, ``**values`` must be empty.
        **values
            Arrays for each key to append. Must match existing keys and have equal lengths.

        Raises
        ------
        ValueError
            If both ``state`` and ``**values`` are provided, key mismatch, or length mismatch.
        """
        if state is not None:
            if values:
                raise ValueError("Cannot provide both state and kwargs.")
            if self._check_keys_different(self._keys, state.keys):
                msg = f"The keys ({state.keys}) do not match the keys of the state {self._keys}"
                raise ValueError(msg)
            for key in self._keys:
                super().__setattr__(key, np.concatenate([getattr(self, key), getattr(state, key)]))
        else:
            keys = list(values.keys())
            if self._check_keys_different(self._keys, keys):
                msg = f"The keys {keys} do not match the keys of the state {self._keys}"
                raise ValueError(msg)
            self._raise_lengths(
                values, "All state variables should have the same lengths, found lengths: ")

            for key in keys:
                super().__setattr__(key, np.concatenate([getattr(self, key), values[key]]))

    @classmethod
    def to_state(cls, instance):
        if not (all(k in instance.keys for k in cls._keys) and all(k in cls._keys for k in instance.keys)):
            raise ValueError(f"Expected the parameters {cls._keys} but got {instance.keys}")
        return cls(**{key: getattr(instance, key) for key in instance.keys})

    @classmethod
    def for_each(cls, **kwargs):
        if not (all(k in kwargs.keys() for k in cls._keys) and all(k in cls._keys for k in kwargs.keys())):
            raise TypeError(f"Expected the parameters {cls._keys} but got {kwargs.keys()}")
        return cls.to_state(State().add_each(**kwargs))


class RoVibState(State):
    """Container for synchronized 1D numeric arrays representing a molecular rovib state.

    Attributes
    ---------
    rot: numpy.ndarray
        The rotational quantum numbers
    vib: numpy.ndarray
        The vibrational quantum numbers
    r: np.ndarray
        Alias for ``rot``
    J: np.ndarray
        Alias for ``rot``
    j: np.ndarray
        Alias for ``rot``
    v: np.ndarray
        Alias for ``vib``

    Notes
    -----
    - All attributes must be 1D numeric NumPy arrays (or array-like) of equal length.
    - Keys are fixed after initialization; use `add` to add new attributes with matching length.
    - Many dunder methods mirror NumPy-like semantics (indexing, concatenation, equality by element).
    """
    _keys = ('rot', 'vib')
    _aliases = {'r': 'rot', 'v': 'vib', 'J': 'rot', 'j': 'rot'}

    def __init__(self, rot, vib):
        super().__init__(rot=rot, vib=vib)