# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Active learning step.
    step_actlearn = 1
    
    # Name of ase database.
    db_name = f"databases/vasp_{step_actlearn:02d}.db"
    
    # Filter atoms structures.
    selection = ""
    
    # ---------------------------------------------------------------------------------
    # RUN DFT CALCULATIONS
    # ---------------------------------------------------------------------------------
    
    # Initialize ase database.
    db_ase = connect(name=db_name)

    # Get list of atoms from database.
    forces = None
    for id in [aa.id for aa in db_ase.select(selection=selection)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        max_forces = np.sqrt((atoms.get_forces() ** 2).sum(axis=1).max())
        print(f'{atoms.info["name"]:100s} {max_forces:7.4f}')
        if forces is None:
            forces = atoms.get_forces()
        else:
            forces = np.vstack([forces, atoms.get_forces()])

    fmax = np.sqrt((forces ** 2).sum(axis=1).max())
    fmean = np.sqrt((forces ** 2).sum(axis=1).mean())
    print(f"fmax: {fmax}")
    print(f"fave: {fmean}")

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
