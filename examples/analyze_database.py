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

def main(parsed_args):
    
    # Active learning step.
    step_actlearn = 0
    
    # Name of ase database.
    db_name = f"databases/ocp_{step_actlearn:02d}.db"
    
    # Select atoms from the database.
    calculation = "climbbonds" # neb | dimer | climbbonds | climbfixint | sella
    selection = ""
    selection += f"calculation={calculation},"
    
    # Initialize ase database.
    db_ase = connect(name=db_name)

    # Print number of selected atoms.
    selected = list(db_ase.select(selection=selection))
    print(f"number of calculations:", len(selected))
    selected = list(db_ase.select(selection=selection, status="finished"))
    print(f"number of converged:", len(selected))
    
    # Read atoms from database.
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)
    
    # Plot success curve.
    if parsed_args.success_curve is True:
        success_steps = success_curve(
            atoms_list=atoms_list,
            max_steps=1000,
            filename="success_curve.png",
        )
    
    # Show atoms.
    if parsed_args.show_atoms is True and len(atoms_list) > 0:
        gui = GUI(atoms_list)
        gui.run()

    # Delete all from database.
    if parsed_args.delete_all is True:
        db_ase.delete([aa.id for aa in db_ase.select(selection=selection)])
    # Delete unfinished from database.
    if parsed_args.delete_unfinished is True:
        selection_new = f"{selection},status!=finished"
        db_ase.delete([aa.id for aa in db_ase.select(selection=selection_new)])

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete_all",
        "-da",
        action="store_true",
        help="Delete selected atoms structures.",
    )
    parser.add_argument(
        "--delete_unfinished",
        "-du",
        action="store_true",
        help="Delete selected unfinished atoms structures.",
    )
    parser.add_argument(
        "--success_curve",
        "-sc",
        action="store_true",
        help="Plot success curve.",
    )
    parser.add_argument(
        "--show_atoms",
        "-sa",
        action="store_true",
        help="Show selected atoms with ase.gui.",
    )
    parsed_args = parser.parse_args()

    import timeit
    time_start = timeit.default_timer()
    main(parsed_args)
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
