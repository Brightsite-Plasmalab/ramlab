from typing_extensions import Any, Dict, Tuple
import numpy as np
import pandas as pd

from ramlab.molecules.state import State


class Transitions:
    linelist: pd.DataFrame

    def __init__(self, linelist=None, **kwargs):
        if linelist is None:
            linelist = pd.DataFrame()
        self.linelist = linelist

        for k, v in kwargs.items():
            self[k] = v

    @classmethod
    def for_states(cls, state_initial: State, state_final: State) -> "Transitions":
        assert list(state_initial.keys()) == list(state_final.keys())
        return cls(
            **{f"initial_{k}": state_initial[k] for k in state_initial.keys()},
            **{f"final_{k}": state_final[k] for k in state_final.keys()},
        )

    @property
    def state_initial(self):
        return State(
            **{
                k.replace("initial_", ""): self.linelist[k].values
                for k in self.linelist.columns
                if k.startswith("initial_")
            }
        )

    @property
    def state_final(self):
        return State(
            **{
                k.replace("final_", ""): self.linelist[k].values
                for k in self.linelist.columns
                if k.startswith("final_")
            }
        )

    def to_states(self) -> Tuple[State, State]:
        return self.state_initial, self.state_final

    def sortby(self, column: str, **kwargs) -> "Transitions":
        return Transitions(self.linelist.sort_values(by=column, **kwargs))

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__["linelist"].columns:
            return self.__dict__["linelist"][name].values
        else:
            raise AttributeError(f"'Transitions' object has no attribute '{name}'.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "linelist":
            super().__setattr__(name, value)
        else:
            self.linelist.loc[:, name] = value

    def __setitem__(self, name: str, value: Any) -> None:
        self.linelist[name] = value

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, np.int64):
            key = np.array([key])
        if (
            isinstance(key, slice)
            or isinstance(key, np.ndarray)
            or isinstance(key, pd.arrays.BooleanArray)
        ):
            # Get the start, stop, and step from the slice
            return Transitions(self.linelist.iloc[key])
        elif isinstance(key, str):
            return self.linelist[key]
        else:
            raise TypeError(
                f"Invalid argument type `{type(key)}`. Use a slice or an integer."
            )

    def __len__(self):
        return len(self.linelist)

    def __repr__(self):
        return f"Transitions[{len(self)}]({', '.join([f'{k}' for k in self.linelist.columns])})"
