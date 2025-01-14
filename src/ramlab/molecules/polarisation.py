from enum import Enum
from typing import Union

import numpy as np


class Axis(Enum):
    PERPENDICULAR = 0
    PARALLEL = 1


class Polarisation(Enum):
    # Relative to the scattering plane, i.e. the plane defined by the incident and scattered beam
    # See Derek A. Long, The Raman Effect
    PERPENDICULAR = "T"
    PARALLEL = "="
    COMBINED = "= + T"
    UNKNOWN = "unknown"

    @classmethod
    def validate(cls, polarisation: Union[str, "Polarisation"]):
        if type(polarisation) is str:
            cls.str_to_enum(polarisation)

    def project(self, projection: Axis):
        if self == Polarisation.COMBINED:
            return 0.5
        elif self == Polarisation.PERPENDICULAR:
            return 1 if projection == Axis.PERPENDICULAR else 0
        elif self == Polarisation.PARALLEL:
            return 1 if projection == Axis.PARALLEL else 0
        elif self == Polarisation.UNKNOWN:
            return np.nan
        else:
            raise ValueError(
                f"Invalid polarisation ({self}) or projection ({projection})"
            )

    @classmethod
    def str_to_enum(cls, polarisation: str) -> "Polarisation":
        if polarisation.__class__.__name__ == "Polarisation":
            return polarisation
        for p in Polarisation:
            if p.value == polarisation:
                return p
        raise ValueError(
            f"Invalid polarisation: {polarisation}. Must be one of {Polarisation.__members__}"
        )


if __name__ == "__main__":
    print(Polarisation.PARALLEL.project(Axis.PARALLEL))
