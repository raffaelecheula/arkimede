# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from ase.gui.gui import GUI
from arkimede.workflow.utilities import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Active learning step.
    step_actlearn = 0
    
    # Name of ase database.
    db_name = f"databases/ocp_{step_actlearn:02d}.db"
    
    # Select atoms from the database.
    calculation = "climbbonds"
    selection = ""
    selection += "status=finished,"
    selection += "surf_structure=fcc-Rh-100,"
    
    # Initialize ase database.
    db_ase = connect(name=db_name)

    reaction_dict = {}
    selection_ts = f"{selection},calculation={calculation}"
    for atoms in get_atoms_list_from_db(db_ase=db_ase, selection=selection_ts):
        species = atoms.info["species"]
        name_clean = atoms.info["name_clean"]
        selection_cl = f"{selection},name={name_clean}"
        atoms_clean = get_atoms_list_from_db(db_ase=db_ase, selection=selection_cl)[0]
        energy_clean = atoms_clean.get_potential_energy()
        images_energies = atoms.info["images_energies"]
        energy_IS = images_energies[0]-energy_clean
        energy_FS = images_energies[-1]-energy_clean
        energy_TS = atoms.get_potential_energy()-energy_clean
        if energy_TS < energy_IS or energy_TS < energy_FS:
            energy_TS = max(energy_IS, energy_FS)
        energies = [energy_IS, energy_IS, energy_TS, energy_TS, energy_FS, energy_FS]
        if species in reaction_dict:
            reaction_dict[species].append(energies)
        else:
            reaction_dict[species] = [energies]

    os.makedirs("figures", exist_ok=True)
    for ii, reaction in enumerate(reaction_dict):
        energies = reaction_dict[reaction]
        plt.figure(ii)
        plt.plot(np.array(energies).T)
        plt.savefig(f"figures/figure_{reaction}.png")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
