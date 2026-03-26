from dataclasses import FrozenInstanceError
import polars as pl

from ramlab.dirs import dir_data
from ramlab.data_parsing.reader_generator import SchemaEntry, HitranSchemaEntry, read_data, add_subschema
from ramlab.data_parsing.hitran import hitran_schema_dict, GlobalQuanta, LocalQuanta


def test_schema_entry_is_frozen_and_offset_returns_new_entry():
    entry = SchemaEntry(start=1, fmt='2d', dtype=pl.UInt8(), length=2, explanation='test', mapping=('a', 'b'))

    assert entry.mapping == {0: 'a', 1: 'b'}
    assert entry.end == 3

    shifted = entry.offset(4)
    copied = entry.copy()

    assert shifted.start == 5
    assert shifted.length == entry.length
    assert shifted.mapping == entry.mapping
    assert copied == entry
    assert copied is not entry

    try:
        setattr(entry, 'start', 2)
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('SchemaEntry should be immutable.')


def test_add_subschema_preserves_mapping_on_offset_entries():
    schema = {'parent': HitranSchemaEntry(start=10, fmt='4d')}
    subschema = {'child': HitranSchemaEntry(start=1, fmt='1d', mapping=('x', 'y'))}

    result = add_subschema(schema, parent=subschema)

    assert result['child'].start == 11
    assert result['child'].mapping == {0: 'x', 1: 'y'}
    assert subschema['child'].start == 1
    assert subschema['child'].mapping == {0: 'x', 1: 'y'}

def test_schema_immutability():
    # Do the ``add_subschema`` thing twice to test that the GlobalQuanta and LocalQuanta dicts are not modified.
    add_subschema(
        hitran_schema_dict(),
        global_quanta_f=GlobalQuanta.class8_f,
        global_quanta_i=GlobalQuanta.class8_i,
        local_quanta_f=LocalQuanta.group3_mecasda_f,
        local_quanta_i=LocalQuanta.group3_mecasda_i,
    )
    add_subschema(
        hitran_schema_dict(),
        global_quanta_f=GlobalQuanta.class8_f,
        global_quanta_i=GlobalQuanta.class8_i,
        local_quanta_f=LocalQuanta.group3_mecasda_f,
        local_quanta_i=LocalQuanta.group3_mecasda_i,
    )

def test_reader():
    schema = add_subschema(
        hitran_schema_dict(),
        global_quanta_f=GlobalQuanta.class8_f,
        global_quanta_i=GlobalQuanta.class8_i,
        local_quanta_f=LocalQuanta.group3_mecasda_f,
        local_quanta_i=LocalQuanta.group3_mecasda_i,
    )

    loc = dir_data / r'test/CH4_test.txt'
    read_data(loc, schema)

    entries = (
        'nu', 'sw', 'E_i', 'v1_f', 'v2_f', 'C_i', 'J_f'
    )
    data = read_data(loc, schema, entries=entries).collect()
    assert (len(data.columns) == len(entries)) and all(
        name in entries for name in data.columns
    ), "read_data entries not working correctly"

