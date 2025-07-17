from typing_extensions import Any, Dict
import numpy as np
import pandas as pd
import copy


class State:
    # TODO: Probably best to use pandas here too, see implementation of Transitions
    state: Dict[str, Any]

    def __init__(self, **state):
        self.state = state

    def keys(self):
        return self.state.keys()

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__.get("state", {}):
            return self.__dict__["state"][name]
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise AttributeError(
                f"No attribute '{name}' in {str(self)}.\nDid you add all necessary quantum numbers?"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "state":
            super().__setattr__(name, value)
        else:
            self.state[name] = value

    def __getitem__(self, key):
        if (
            isinstance(key, slice)
            or isinstance(key, int)
            or isinstance(key, np.ndarray)
        ):
            # Get the start, stop, and step from the slice
            return State(**{k: self.state[k][key] for k in list(self.state.keys())})
        elif isinstance(key, str):
            return self.state[key]
        else:
            raise TypeError(
                f"Invalid argument type `{type(key)}`. Use a slice or an integer."
            )

    def __len__(self):
        return np.size(list(self.state.values())[0])

    def __repr__(self):
        if len(self) == 1:
            nt = "\n\t"
            return f"State({','.join([f'{nt}{k}({str(type(v))})={v}' for k, v in self.state.items()])}\n)"
        return f"State({', '.join([f'{k}' for k in self.state.keys()])}) with length {len(self)}"

    def __add__(self, other):
        return State(
            **{k: np.concatenate([self.state[k], other.state[k]]) for k in self.state}
        )

    def __eq__(self, other):
        if not isinstance(other, State):
            return False

        # Check if other has at least the same keys as self
        if not set(self.state.keys()).issubset(set(other.state.keys())):
            if len(other) == 1:
                # Return a boolean mask of all False when required keys are missing
                return np.full(len(self), False)
            return False

        # If other is a single-element State, return a boolean mask
        if len(other) == 1:
            mask = np.full(len(self), True)
            for key in self.state.keys():
                try:
                    # Compare each element of self with the single element of other
                    element_mask = np.equal(self.state[key], other.state[key])
                    mask = mask & element_mask
                except (ValueError, TypeError):
                    # Fallback to regular equality comparison
                    try:
                        element_mask = self.state[key] == other.state[key]
                        mask = mask & element_mask
                    except Exception:
                        # If comparison fails, mark all as False
                        mask = np.full(len(self), False)
                        break
            return mask

        # For same-length States, check element-wise equality
        if len(self) != len(other):
            return False

        # Check if all values are equal (element-wise for arrays) for keys in self
        for key in self.state.keys():
            try:
                # Use np.array_equal to handle both scalar and array values
                if not np.array_equal(self.state[key], other.state[key]):
                    return False
            except (ValueError, TypeError):
                # Fallback to regular equality if np.array_equal fails
                try:
                    if self.state[key] != other.state[key]:
                        return False
                except Exception:
                    # If both comparison methods fail, consider them unequal
                    return False

        return True

    def unique(self, return_index=False):
        df = pd.DataFrame.from_dict(self.state, orient="index").reset_index()
        # df = pd.DataFrame.from_dict(self.state).reset_index()

        # Drop duplicate rows and keep the first occurrence
        df_unique = df.drop_duplicates(keep="first")

        # Get the indices of the unique rows
        idx = df_unique.index.tolist()

        # # Convert the dictionary values to a 2D numpy array
        # matrix = np.array(list(self.state.values()), dtype=np.float32)

        # # Find the unique rows and their indices
        # _, idx = np.unique(matrix, axis=1, return_index=True)

        if return_index:
            return self[np.ndarray(idx, dtype=int)], idx
        else:
            return self[idx]

    def copy(self):
        """
        Create a deep copy of the State object.

        Returns:
            State: A new State object with deep copies of all state data.
        """
        return State(**copy.deepcopy(self.state))


if __name__ == "__main__":
    # Test cases for the State class, particularly the __eq__ method
    print("Testing State class __eq__ method:")
    print("=" * 50)

    # Test 1: Basic equality with same-length states
    print("\nTest 1: Same-length state equality")
    state1 = State(J=np.array([1, 2, 3]), v=np.array([0, 1, 0]))
    state2 = State(J=np.array([1, 2, 3]), v=np.array([0, 1, 0]))
    state3 = State(J=np.array([1, 2, 4]), v=np.array([0, 1, 0]))

    print(f"state1 == state2: {state1 == state2}")  # Should be True
    print(f"state1 == state3: {state1 == state3}")  # Should be False

    # Test 2: Boolean mask with single-element state
    print("\nTest 2: Boolean mask with single-element state")
    multi_state = State(J=np.array([1, 2, 1, 3]), v=np.array([0, 1, 0, 2]))
    single_state = State(J=np.array([1]), v=np.array([0]))

    mask = multi_state == single_state
    print(f"Multi-state: J={multi_state.J}, v={multi_state.v}")
    print(f"Single-state: J={single_state.J}, v={single_state.v}")
    print(f"Boolean mask: {mask}")
    print(f"Mask type: {type(mask)}")

    # Test 3: Filtering using the boolean mask
    print("\nTest 3: Filtering using boolean mask")
    filtered_state = multi_state[mask]
    print(f"Filtered state: J={filtered_state.J}, v={filtered_state.v}")
    print(f"Filtered state length: {len(filtered_state)}")

    # Test 4: Different keys should return all False mask
    print("\nTest 4: Different keys")
    single_state_diff_keys = State(K=np.array([1]), v=np.array([0]))
    mask_diff_keys = multi_state == single_state_diff_keys
    print(f"Mask with different keys: {mask_diff_keys}")

    # Test 5: Non-State comparison
    print("\nTest 5: Non-State comparison")
    print(f"state1 == 'not a state': {state1 == 'not a state'}")

    # Test 6: Single element states
    print("\nTest 6: Single element state comparison")
    single1 = State(J=np.array([2]), v=np.array([1]))
    single2 = State(J=np.array([2]), v=np.array([1]))
    single3 = State(J=np.array([3]), v=np.array([1]))

    print(f"single1 == single2: {single1 == single2}")  # Should be True
    print(f"single1 == single3: {single1 == single3}")  # Should be False

    print("\nAll tests completed!")
