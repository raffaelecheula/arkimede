# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.io import read
from ase.db import connect
from ase.calculators.vasp.vasp import Vasp
from arkimede.utils import templates_basedir, scripts_basedir
from arkimede.workflow.reaction_workflow import run_dft_calculations_k8s
from arkimede.workflow.utilities import read_step_actlearn, get_atoms_list_from_db
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
    selection = "calculation=neb"
    
    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = "databases/vasp_opt.db"
    db_dft_append = True
    
    # Name of the folder for dft single point calculations.
    basedir_dft_calc = "calculations/vasp_opt"
    
    # Name of the template yaml file and namespace for kubernetes submission.
    template_yaml = templates_basedir() / "template_k8s.yaml"
    namespace = "raffaelecheula"
    
    # Filename of calculation output.
    filename_out = "asevasp.out"
    
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
    
    # Vasp command.
    vasp_bin = "/opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std"
    vasp_command = f"{vasp_bin} > vasp.out"
    
    # Ase vasp command.
    asevasp_py = scripts_basedir() / "asevasp.py"
    conda_command = "source /home/jovyan/.bashrc; conda activate ocp"
    command = f"{conda_command}; python {asevasp_py} >> asevasp.out"
    
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
        atoms.info["calculation"] = "climbbonds"
        atoms.info["vasp_flags"] = vasp_flags
        atoms.info["vasp_command"] = vasp_command
        atoms.info["filename_out"] = filename_out
        atoms.info["save_trajs"] = True

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
