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
    db_ase_name = "database.db"

    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    selection = "calculation=climbbonds"
    selected = list(db_ase.select(selection=selection))
    print(len(selected))
    
    selection = "calculation=climbbonds,status=finished"
    selected = list(db_ase.select(selection=selection))
    print(len(selected))
    
    atoms_list = []
    id_list = [aa.id for aa in db_ase.select(selection=selection)]
    for id in id_list:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms_list.append(atoms)
    
    #gui = GUI(atoms_list)
    #gui.run()
    

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
