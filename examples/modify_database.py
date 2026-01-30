# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import timeit
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

from mlps_finetuning.databases import get_atoms_list_from_db, write_atoms_to_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Parameters.
    calc_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN
    db_name = f"databases/MLP_relax_{calc_name}.db"
    calculation = "sella-ba" # neb | dimer | climbfixint | sella | sella-ba
    delete_all = False
    delete_unfinished = False
    copy_data_to = None
    print_unfinished = True
    
    # Initialize ase database.
    db_ase = connect(name=db_name)
    kwargs = {"calculation": calculation}

    # Delete all from database.
    if delete_all is True:
        db_ase.delete([aa.id for aa in db_ase.select(**kwargs)])
    # Delete unfinished from database.
    if delete_unfinished is True:
        for status in ("unfinished", "desorbed", "failed", "wrong"):
            db_ase.delete([aa.id for aa in db_ase.select(status=status, **kwargs)])
    
    # Copy data.
    if copy_data_to is not None:
        kwargs["status"] = "finished"
        atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
        for atoms in atoms_list:
            keys_store = ("name", "image", "status", "neb_steps")
            kwargs = {key: atoms.info[key] for key in keys_store}
            kwargs["calculation"] = copy_data_to
            write_atoms_to_db(db_ase=db_ase, atoms=atoms, **kwargs)
    
    # Print names of unfinished calculations.
    if print_unfinished is True:
        for status in ("unfinished", "desorbed", "failed", "wrong"):
            kwargs["status"] = status
            atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
            print(status)
            for atoms in atoms_list:
                print(atoms.info["name"])

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
