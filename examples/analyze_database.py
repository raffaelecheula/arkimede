# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    
    # Name of ase database.
    db_ase_name = "database_ocp_2.db"
    
    # Select atoms from the database.
    calculation = "dimer"
    selection = f"calculation={calculation}"
    #selection = "Rh>0,status=finished"
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Print number of selected atoms.
    selected = list(db_ase.select(selection=selection))
    print(f"number of {calculation} calculations:", len(selected))
    selected = list(db_ase.select(selection=f"{selection},status=finished"))
    print(f"number of {calculation} converged:", len(selected))
    
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms_list.append(atoms)
    
    gui = GUI(atoms_list)
    gui.run()

# -----------------------------------------------------------------------------
# IF NAME MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
