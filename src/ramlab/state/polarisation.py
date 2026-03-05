from enum import Flag
from typing_extensions import Union


class Axis(Flag):
    PERPENDICULAR = 1
    PARALLEL = 2


class Polarisation(Flag):
    # Relative to the scattering plane, i.e. the plane defined by the incident and scattered beam
    # See Derek A. Long, The Raman Effect
    PERPENDICULAR = 1
    PARALLEL = 2

    @classmethod
    def _string_map(cls):
        return {
            "+": cls.PERPENDICULAR,
            "=": cls.PARALLEL,
            "perpendicular": cls.PERPENDICULAR,
            "perp": cls.PERPENDICULAR,
            "parallel": cls.PARALLEL,
            "para": cls.PARALLEL,
            "both": cls.PERPENDICULAR | cls.PARALLEL,
            "combined": cls.PERPENDICULAR | cls.PARALLEL,
            "comb": cls.PERPENDICULAR | cls.PARALLEL,
        }

    @classmethod
    def validate(cls, polarisation: Union[str, "Polarisation"]):
        if isinstance(polarisation, str):
            return cls.from_string(polarisation)
        elif isinstance(polarisation, Polarisation):
            return polarisation
        else:
            raise TypeError(f"Expected str or Polarisation, got {type(polarisation).__name__}")

    def project(self, projection: Axis):
        if self == (Polarisation.PERPENDICULAR | Polarisation.PARALLEL):
            return 0.5
        elif self in (Polarisation.PERPENDICULAR, Polarisation.PARALLEL):
            return int(self == projection)  # False -> 0, True -> 1
        else:
            msg = f"Invalid polarisation ({self}) or projection ({projection})"
            raise ValueError(msg)

    @classmethod
    def from_string(cls, polarisation: str) -> "Polarisation":
        if isinstance(polarisation, Polarisation):
            return polarisation
        str_map = cls._string_map()
        if polarisation.lower() in str_map:
            return str_map[polarisation.lower()]
        str_map_keys = ', '.join(str_map.keys())

        msg = f'Invalid polarisation: {polarisation}. Must be one of {str_map_keys}'
        raise ValueError(msg)

    @property
    def COMBINED(self):
        return Polarisation.PERPENDICULAR | Polarisation.PARALLEL


if __name__ == "__main__":
    print(Polarisation.PARALLEL.project(Axis.PARALLEL))
