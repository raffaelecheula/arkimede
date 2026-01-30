# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import numpy as np
from ase.db import connect

from arkimede.utilities import print_title
from arkimede.databases import get_atoms_list_from_db
from arkimede.reports import plot_calculation_times, plot_steps_distribution

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Results databases.
    calc_MLP_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN
    # Databases names.
    db_DFT_relax_name = f"databases/DFT_relax.db"
    db_DFT_after_name = f"databases/DFT_after_{calc_MLP_name}.db"
    db_AL_serial_name = f"databases/AL_serial_{calc_MLP_name}.db"
    db_AL_parall_name = f"databases/AL_parall_{calc_MLP_name}.db"
    db_MLP_relax_name = f"databases/MLP_relax_{calc_MLP_name}.db"

    # Read pure DFT results.
    print_title("Pure DFT")
    db_DFT_relax = connect(name=db_DFT_relax_name)
    db_kwargs = {"calculation": "sella", "status": "finished"}
    atoms_list = get_atoms_list_from_db(db_ase=db_DFT_relax, **db_kwargs)
    print(f"Number of structures: {len(atoms_list):5d}")
    # Get number of DFT steps.
    steps_pure_DFT = []
    for atoms in atoms_list:
        steps_pure_DFT.append(atoms.info["counter"])
    print(f"Mean number of steps: {np.mean(steps_pure_DFT):5.1f}")

    # Read pre-relaxation results.
    print_title("DFT after MLP")
    db_DFT_after = connect(name=db_DFT_after_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_DFT_after)
    print(f"Number of structures: {len(atoms_list):5d}")
    # Get number of DFT steps.
    steps_DFT_after = []
    for atoms in atoms_list:
        steps_DFT_after.append(atoms.info["counter"])
    print(f"Mean number of steps: {np.mean(steps_DFT_after):5.1f}")

    # Read active learning results.
    print_title("Active Learning (serial)")
    db_AL_serial = connect(name=db_AL_serial_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_AL_serial)
    #atoms_list = [atoms for atoms in atoms_list if atoms.info["DFT_steps"] < 10]
    print(f"Number of structures: {len(atoms_list):5d}")
    # Get number of DFT steps and computational times.
    steps_actlearn = []
    time_MLP = 0.
    time_DFT = 0.
    time_train = 0.
    for atoms in atoms_list:
        steps_actlearn.append(atoms.info["DFT_steps"])
        time_MLP += atoms.info["time_MLP"] / 60 # [min]
        time_DFT += atoms.info["time_DFT"] / 60 # [min]
        time_train += atoms.info["time_train"] / 60 # [min]
    time_DFT_per_step = time_DFT / sum(steps_actlearn) # [min]
    time_MLP /= len(atoms_list) # [min]
    time_DFT /= len(atoms_list) # [min]
    time_train /= len(atoms_list) # [min]
    print(f"Mean number of steps: {np.mean(steps_actlearn):5.1f}")

    steps_dict = {
        "Pure DFT": steps_pure_DFT,
        "DFT after MLP": steps_DFT_after,
        "Active learning\n(serial)": steps_actlearn,
        "Active learning\n(parallel)": np.array(steps_actlearn) * 3,
    }

    times_dict = {
        "Pure DFT": [0, time_DFT_per_step * np.mean(steps_pure_DFT), 0],
        "DFT after MLP": [0.8, time_DFT_per_step * np.mean(steps_DFT_after), 0],
        "Active learning\n(serial)": [2.1, time_DFT, time_train],
        #"Active learning\n(parallel)": []
    }

    # Steps calculation times.
    labels = ["MLP", "DFT", "fine-tuning"]
    plot_calculation_times(times_dict=times_dict, labels=labels)

    # Steps distribution plot.
    y_max = 300
    kwargs_violin = {"points": 300, "bw_method": 0.2}
    plot_steps_distribution(
        steps_dict=steps_dict,
        kwargs_violin=kwargs_violin,
        y_max=y_max,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
