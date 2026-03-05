from ramlab.database_molecules import Methane

loc = r"C:\Users\P70095200\PycharmProjects\ramlab_test\data\CH4\Methane Pentad 1450"

for p in p_vals:
    data_p = read_data(data_dir / f"spectr_{p}_R{num}_1450K.t", schema, entries)
    datas.append(data_p)

# %%