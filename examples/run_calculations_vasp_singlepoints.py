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
    
    # Name of ase database containing the structures to read.
    db_ase_name = "database_ocp.db"

    # Select atoms from the ase database.
    selection = "calculation=climbbonds,status=finished,surf_structure=fcc-Rh-100"
    
    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = "database_vasp.db"
    db_dft_append = True
    
    # Name of the folder for dft single point calculations.
    basedir_dft_calc = "calculations_vasp"
    
    # Name of the template yaml file and namespace for kubernetes submission.
    template_yaml = templates_basedir() / "template_vasp_k8s.yaml"
    namespace = "raffaelecheula"
    
    # Filename and name of dft calculation.
    filename_out = "OUTCAR"
    calculation = "vasp-singlepoint"
    
    # Vasp parameters.
    vasp_flags = {
        "ibrion": -1,
        "nsw": 0,
        "isif": 0,
        "isym": 0,
        "lreal": "Auto",
        "ediffg": 0.0,
        "symprec": 1e-10,
        "encut": 350.0,
        "ncore": 2,
        "lcharg": False,
        "lwave": False,
        "gga": "RP",
        "pp": "PBE",
        "xc": "PBE",
    }
    
    # Setup vasp calculator.
    calc = Vasp(**vasp_flags)
    
    # ---------------------------------------------------------------------------------
    # RUN DFT CALCULATIONS
    # ---------------------------------------------------------------------------------
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Get list of atoms from database.
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms_list.append(atoms)

    # Initialize ase dft database.
    db_dft = connect(name=db_dft_name, append=db_dft_append)

    # Run the dft calculations.
    run_dft_calculations_k8s(
        atoms_list=atoms_list,
        calc=calc,
        template_yaml=template_yaml,
        namespace=namespace,
        basedir_dft_calc=basedir_dft_calc,
        filename_out=filename_out,
        calculation=calculation,
        db_dft=db_dft,
        write_input_fun=write_input_vasp,
        check_finished_fun=check_finished_vasp,
        job_queued_fun=job_queued_k8s,
    )

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
