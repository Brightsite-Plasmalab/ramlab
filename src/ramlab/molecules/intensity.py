from typing_extensions import Union
import numpy as np

from src.ramlab.molecules.polarisation import Polarisation


class Intensity:
    I_perpendicular: Union[float, np.ndarray]
    I_parallel: Union[float, np.ndarray]

    # Getter for the total intensity
    @property
    def I(self):
        return self.I_perpendicular + self.I_parallel

    def __init__(
        self,
        perpendicular: Union[float, np.ndarray],
        parallel: Union[float, np.ndarray],
    ):
        self.I_perpendicular = perpendicular
        self.I_parallel = parallel

    def __add__(self, other):
        return Intensity(
            self.I_perpendicular + other.I_perpendicular,
            self.I_parallel + other.I_parallel,
        )

    def for_polarisation(self, polarisation: Union[str, Polarisation]) -> float:
        if type(polarisation) is str:
            polarisation = Polarisation.str_to_enum(polarisation)
        if polarisation == Polarisation.COMBINED:
            return self.I
        elif polarisation == Polarisation.PARALLEL:
            return self.I_parallel
        elif polarisation == Polarisation.PERPENDICULAR:
            return self.I_perpendicular
        else:
            raise ValueError(f"Invalid polarisation: {polarisation}")
