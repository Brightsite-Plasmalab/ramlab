from ramlab.dirs import dir_data
from ramlab.data_parsing.reader_generator import read_data, add_subschema
from ramlab.data_parsing.hitran import hitran_schema_dict, GlobalQuanta, LocalQuanta

def test_schema_immutability():
    # Do the ``add_subschema`` thing twice to test that the GlobalQuanta and LocalQuanta dicts are not modified.
    add_subschema(
        hitran_schema_dict(),
        global_final_quanta=GlobalQuanta.class8_f,
        global_initial_quanta=GlobalQuanta.class8_i,
        local_final_quanta=LocalQuanta.group3_mecasda_f,
        local_initial_quanta=LocalQuanta.group3_mecasda_i,
    )
    add_subschema(
        hitran_schema_dict(),
        global_final_quanta=GlobalQuanta.class8_f,
        global_initial_quanta=GlobalQuanta.class8_i,
        local_final_quanta=LocalQuanta.group3_mecasda_f,
        local_initial_quanta=LocalQuanta.group3_mecasda_i,
    )

def test_reader():
    schema = add_subschema(
        hitran_schema_dict(),
        global_final_quanta=GlobalQuanta.class8_f,
        global_initial_quanta=GlobalQuanta.class8_i,
        local_final_quanta=LocalQuanta.group3_mecasda_f,
        local_initial_quanta=LocalQuanta.group3_mecasda_i,
    )

    loc = dir_data / r'test/CH4_test.txt'
    read_data(loc, schema)

    entries = (
        'nu', 'sw', 'E_i', 'v1_f', 'v2_f', 'C_i', 'J_f'
    )
    data = read_data(loc, schema, entries=entries).collect()
    assert (len(data.columns) == len(entries)) and all(name in entries for name in data.columns), "read_data entries not working correctly"

