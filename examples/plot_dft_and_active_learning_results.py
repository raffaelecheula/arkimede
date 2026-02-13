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
    db_AL_groups_name = f"databases/AL_groups_{calc_MLP_name}.db"
    db_MLP_relax_name = f"databases/MLP_relax_{calc_MLP_name}.db"

    # Labels.
    label_DFT_relax = "Pure DFT\nRelaxation"
    label_DFT_after = "DFT after MLP\nPre-relaxation"
    label_serial_AL = "Sequential\nActive Learning"
    label_groups_AL = "Batch\nActive Learning"

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
    print_title("Sequential Active Learning")
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

    # Read active learning results.
    print_title("Batch Active Learning")
    db_AL_groups = connect(name=db_AL_groups_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_AL_groups)
    #atoms_list = [atoms for atoms in atoms_list if atoms.info["DFT_steps"] < 10]
    print(f"Number of structures: {len(atoms_list):5d}")
    # Get number of DFT steps.
    steps_DFT_groups = []
    for atoms in atoms_list:
        steps_DFT_groups.append(atoms.info["counter"])
    steps_DFT_groups = [aa / 8 for aa in steps_DFT_groups] # TODO:
    print(f"Mean number of steps: {np.mean(steps_DFT_groups):5.1f}")

    steps_dict = {
        label_DFT_relax: steps_pure_DFT,
        label_DFT_after: steps_DFT_after,
        label_serial_AL: steps_actlearn,
        label_groups_AL: steps_DFT_groups,
    }

    times_dict = {
        label_DFT_relax: [0, time_DFT_per_step * np.mean(steps_pure_DFT), 0],
        label_DFT_after: [0.8, time_DFT_per_step * np.mean(steps_DFT_after), 0],
        label_serial_AL: [2.1, time_DFT, time_train],
        #label_groups_AL: []
    }

    # Steps calculation times.
    labels = ["MLP", "DFT", "fine-tuning"]
    plot_calculation_times(times_dict=times_dict, labels=labels)

    # Steps distribution plot.
    y_max = 300
    violin_kwargs = {"points": 300, "bw_method": 0.2}
    plot_steps_distribution(
        steps_dict=steps_dict,
        violin_kwargs=violin_kwargs,
        y_max=y_max,
        color="salmon",
        alpha_fill=1.0,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
