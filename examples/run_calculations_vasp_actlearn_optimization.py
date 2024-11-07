# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.io import read
from ase.db import connect
from ase.calculators.vasp.vasp import Vasp
from arkimede.utilities import read_step_actlearn, get_atoms_list_from_db
from arkimede.utilities.paths import templates_basedir, scripts_basedir
from arkimede.ocp.ocp_utils import get_checkpoint_path, get_checkpoint_path_actlearn
from arkimede.workflow.reaction_workflow import run_dft_calculations_k8s
from arkimede.workflow.dft_calculations import (
    write_input_asevasp,
    read_output_asevasp,
    check_finished_asevasp,
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
    selection = "calculation=climbbonds,status=finished"
    
    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = "databases/vasp_actlearn.db"
    db_dft_append = True
    
    # Keywords used to match structures in the db and to be stored in db columns.
    keys_match = ["name", "calculation"]
    keys_store = ["name", "calculation", "status", "name_ref", "species"]
    
    # Name of the folder for dft single point calculations.
    basedir_dft_calc = "calculations/vasp_actlearn"
    
    # Name of the template yaml file and namespace for kubernetes submission.
    template_yaml = templates_basedir() / "template_k8s.yaml"
    namespace = "raffaelecheula"
    
    # Filename of calculation output.
    filename_out = "asevasp_actlearn.out"
    
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
    vasp_flags["lwave"] = True
    
    # Vasp command.
    vasp_bin = "/opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std"
    vasp_command = f"{vasp_bin} > vasp.out"
    
    # Python command.
    pythonfile = scripts_basedir() / "asevasp_actlearn.py"
    conda_command = "source /home/jovyan/.bashrc; conda activate ocp"
    command = f"{conda_command}; python {pythonfile} >> {filename_out}"
    
    # OCPmodels ase calculator.
    checkpoint_key = 'GemNet-OC OC20+OC22 v2'
    if step_actlearn == 0:
        checkpoint_path = get_checkpoint_path(checkpoint_key=checkpoint_key)
    else:
        checkpoint_path = get_checkpoint_path_actlearn(step_actlearn=step_actlearn)
    
    # GPU requested.
    gpu_req = 1
    
    # ---------------------------------------------------------------------------------
    # RUN DFT CALCULATIONS
    # ---------------------------------------------------------------------------------
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Get list of atoms from database.
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)

    # Initialize ase dft database.
    db_dft = connect(name=db_dft_name, append=db_dft_append)

    # Store calculation settings into atoms.info dictionaries.
    for atoms in atoms_list:
        atoms.pbc = True
        atoms.set_tags([1.]*len(atoms))
        atoms.info["calculation"] = "climbbonds"
        atoms.info["vasp_flags"] = vasp_flags
        atoms.info["vasp_command"] = vasp_command
        atoms.info["filename_out"] = filename_out
        atoms.info["save_trajs"] = True
        atoms.info["min_steps"] = 10
        atoms.info["max_steps"] = 1000
        atoms.info["fmax"] = 0.05
        atoms.info["max_step_actlearn"] = 100
        atoms.info["checkpoint_path"] = checkpoint_path

    # Run the dft calculations.
    run_dft_calculations_k8s(
        atoms_list=atoms_list,
        template_yaml=template_yaml,
        namespace=namespace,
        basedir_dft_calc=basedir_dft_calc,
        filename_out=filename_out,
        db_dft=db_dft,
        write_input_fun=write_input_asevasp,
        read_output_fun=read_output_asevasp,
        check_finished_fun=check_finished_asevasp,
        job_queued_fun=job_queued_k8s,
        command=command,
        keys_match=keys_match,
        keys_store=keys_store,
        gpu_req=gpu_req,
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
