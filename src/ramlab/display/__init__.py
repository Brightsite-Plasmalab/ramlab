import matplotlib.pyplot as plt


def plot_results(fitresults, T_name="T"):
    if type(fitresults) is not dict:
        fitresults = fitresults.to_dict()
    T_err = fitresults["Ts"][f"{T_name}_err"]
    T_stderr_str = f"{3*T_err:.0f}" if T_err is not None else "?"
    T_str = (
        f"{fitresults['Ts'][T_name]:.0f}"
        if fitresults["Ts"][T_name] is not None
        else "?"
    )

    plt.plot(
        fitresults["data"]["lambda"], fitresults["data"]["data"], "r.-", label="Data"
    )
    plt.plot(
        fitresults["fit"]["lambda"],
        fitresults["fit"]["data"],
        "k-",
        label=rf"Fit ($T={T_str} \pm {T_stderr_str}$)",
    )
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [a.u.]")

    plt.legend()
