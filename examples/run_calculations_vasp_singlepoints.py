# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.io import read
from ase.db import connect
from ase.calculators.vasp.vasp import Vasp
from arkimede.utilities import read_step_actlearn, get_atoms_list_from_db
from arkimede.utilities.paths import templates_basedir
from arkimede.workflow.reaction_workflow import run_dft_calculations_k8s
from arkimede.workflow.dft_calculations import (
    write_input_vasp,
    read_output_vasp,
    check_finished_vasp,
    job_queued_k8s,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Active learning step.
    filename_actlearn = "actlearn.yaml"
    step_actlearn = read_step_actlearn(filename=filename_actlearn)
    
    # Name of ase database containing the structures to read.
    db_ase_name = f"databases/ocp_{step_actlearn:02d}.db"

    # Select atoms from the ase database.
    selection = "status=finished,"
    
    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = f"databases/vasp_{step_actlearn:02d}.db"
    db_dft_append = True
    
    # Keywords used to match structures in the db and to be stored in db columns.
    keys_match = ["name", "calculation"]
    keys_store = ["name", "calculation", "status", "name_ref", "species"]
    
    # Name of the folder for dft single point calculations.
    basedir_dft_calc = f"calculations/vasp_{step_actlearn:02d}"
    
    # Name of the template yaml file and namespace for kubernetes submission.
    template_yaml = templates_basedir() / "template_k8s.yaml"
    namespace = "raffaelecheula"
    
    # Filename of calculation output.
    filename_out = "OUTCAR"
    
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
    
    # Vasp command.
    vasp_bin = "/opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std"
    vasp_command = f"mpirun -np 8 --map-by hwthread {vasp_bin} > vasp.out"
    
    # ---------------------------------------------------------------------------------
    # RUN DFT CALCULATIONS
    # ---------------------------------------------------------------------------------
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Get list of atoms from database.
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)

    # Initialize ase dft database.
    db_dft = connect(name=db_dft_name, append=db_dft_append)

    for atoms in atoms_list:
        atoms.pbc = True
        atoms.calc = calc
        atoms.info["calculation"] = "vasp-singlepoint"

    # Run the dft calculations.
    run_dft_calculations_k8s(
        atoms_list=atoms_list,
        template_yaml=template_yaml,
        namespace=namespace,
        basedir_dft_calc=basedir_dft_calc,
        filename_out=filename_out,
        db_dft=db_dft,
        write_input_fun=write_input_vasp,
        read_output_fun=read_output_vasp,
        check_finished_fun=check_finished_vasp,
        job_queued_fun=job_queued_k8s,
        command=vasp_command,
        keys_match=keys_match,
        keys_store=keys_store,
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
