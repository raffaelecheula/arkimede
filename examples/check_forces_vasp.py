# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.io import read
from ase.db import connect
from ase.calculators.vasp.vasp import Vasp
from arkimede.utils import templates_basedir
from arkimede.workflow.reaction_workflow import run_dft_calculations_k8s
from arkimede.workflow.dft_calculations import (
    write_input_vasp,
    check_finished_vasp,
    job_queued_k8s,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = "database_vasp.db"
    
    # ---------------------------------------------------------------------------------
    # RUN DFT CALCULATIONS
    # ---------------------------------------------------------------------------------
    
    # Initialize ase database.
    db_ase = connect(name=db_dft_name)

    # Get list of atoms from database.
    forces = None
    for id in [aa.id for aa in db_ase.select(selection="")]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
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
