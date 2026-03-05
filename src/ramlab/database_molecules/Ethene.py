import polars as pl

from ramlab.data_parsing.hitran import full_schema_dict
from ramlab.molecule_base.filters import levels_filter_and
from ramlab.molecule_base.Molecule import Molecule

c2h4_mecasda_schema = full_schema_dict(
    general_schema="hitran",
    globals_="class9",
    locals_="group1",
)

# ch4_ramlab_schema = full_schema_dict(
#     general_schema="ramlab",
#     global_quanta="class8",
#     local_quanta="group3_mecasda",
# )


class Ethene(Molecule):
    """
    A class representing the Ethene/Ethylene (C2H4) molecule.
    """
    _default_schema = c2h4_mecasda_schema
    _must_params = Molecule._must_params + ()  # TODO
    _temperatures = {"vib": "vib", "rot": "rot", "v": "vib", "r": "rot"}
    _unique_entries = ('v1_i', 'v2_i', 'v3_i', 'v4_i', 'J_i', 'G_v_i', 'G_r_i', 'alpha_i')  # TODO

    def __init__(self, data, T_ref, partition_interpolator=None):
        super().__init__(data, T_ref, partition_interpolator)


Ethylene = Ethene # Alternative name