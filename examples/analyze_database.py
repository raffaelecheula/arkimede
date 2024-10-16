# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from ase.gui.gui import GUI
from arkimede.utilities import get_atoms_list_from_db, success_curve

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Control parameters.
    delete_selection = False
    show_selection = False
    
    # Active learning step.
    step_actlearn = 0
    
    # Name of ase database.
    db_name = f"databases/ocp_{step_actlearn:02d}.db"
    
    # Select atoms from the database.
    calculation = "climbbonds"
    selection = ""
    selection += f"calculation={calculation},"
    
    # Initialize ase database.
    db_ase = connect(name=db_name)

    # Print number of selected atoms.
    selected = list(db_ase.select(selection=selection))
    print(f"number of calculations:", len(selected))
    selected = list(db_ase.select(selection=selection, status="finished"))
    print(f"number of converged:", len(selected))
    
    # Delete atoms from database.
    if delete_selection is True:
        selection += "status=unfinished,"
        db_ase.delete([aa.id for aa in db_ase.select(selection=selection)])
    
    # Read atoms from database.
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)
    
    # Plot success curve.
    success_curve(
        atoms_list=atoms_list,
        max_steps=1000,
        filename="success_curve.png",
    )
    
    # Show atoms.
    if show_selection is True and len(atoms_list) > 0:
        gui = GUI(atoms_list)
        gui.run()

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
