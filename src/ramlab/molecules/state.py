from typing import Any, Dict
import numpy as np
import pandas as pd


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
            nt = '\n\t'
            return f"State({','.join([f'{nt}{k}({str(type(v))})={v}' for k, v in self.state.items()])}\n)"
        return f"State({', '.join([f'{k}' for k in self.state.keys()])}) with length {len(self)}"

    def __add__(self, other):
        return State(
            **{k: np.concatenate([self.state[k], other.state[k]]) for k in self.state}
        )

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
