import pickle
import sif_parser
import numpy as np
import matplotlib.pyplot as plt

f_pickle = ""  # Replace with path of the ..._idx.pkl file
f_data = ""  # Replace with path of the .sif file

# Load pickle file
info = pickle.load(open(f_pickle, "rb"))
print(info.keys())
print(info["configs"])

# Load sif file
print("Loading image data...")
data, _ = sif_parser.np_open(f_data)


def get_data(data, info, config):
    # Get the keys for the signal and background indices
    sig_key = f"{config}_sig"
    bg_key = f"{config}_bg"

    if not sig_key in info:
        return None, None

    # Get the indices for the signal and background
    sig_ind = info[sig_key]
    bg_ind = info[bg_key]

    # Get the data for the signal and background
    sig_data = data[sig_ind[0], :, :]
    bg_data = data[bg_ind[0], :, :]

    sig_data_avg = np.median(sig_data, axis=0)
    bg_data_avg = np.median(bg_data, axis=0)

    return sig_data_avg, bg_data_avg, sig_data_avg - bg_data_avg


# Load the data of all config
# (sig, bg, sig-bg)
sig_x_0_000mm, bg_x_0_000mm, sig_x_0_000mm_corr = get_data(data, info, 0)
