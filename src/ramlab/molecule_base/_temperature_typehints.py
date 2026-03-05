from typing import TypedDict


class OneTemperatureKwargs(TypedDict):
    T: float


class RoVibTemperatureKwargs(TypedDict):
    T_vib: float
    T_rot: float