# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from ase.calculators.vasp.vasp import Vasp
from arkimede.utils import templates_basedir
from arkimede.workflow.utilities import get_atoms_list_from_db
from arkimede.workflow.reaction_workflow import run_dft_calculations_k8s
from arkimede.workflow.dft_calculations import (
    write_input_vasp_vib,
    check_finished_vasp_vib,
    job_queued_k8s,
)

# -----------------------------------------------------------------------------
# ASE VASP VIBRATIONS
# -----------------------------------------------------------------------------

def main():

    # Name of ase database containing the unrelaxed structures to read.
    db_init_name = "database_init.db"

    # Select atoms from the ase database.
    selection = ""

    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = "database_gas.db"
    db_dft_append = True
    
    # Name of the folder for gas vibrations calculations.
    basedir_dft_calc = "calculations_vib"
    
    # Name of the template yaml file and namespace for kubernetes submission.
    template_yaml = templates_basedir() / "template_k8s.yaml"
    namespace = "raffaelecheula"
    
    # Filename and name of dft calculation.
    filename_out = "vib.out"
    calculation = "vasp-vibrations"

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
    vasp_exe = "/opt/vasp.6.1.2_pgi_mkl_beef/bin/vasp_std"
    command = f"mpirun -np 8 --map-by hwthread {vasp_exe} > vasp.out"
    
    # ---------------------------------------------------------------------------------
    # RUN DFT CALCULATIONS
    # ---------------------------------------------------------------------------------
    
    # Initialize ase init database.
    db_init = connect(name=db_init_name)

    # Get lists of atoms from database.
    atoms_list = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="gas",
    )
    
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
        write_input_fun=write_input_vasp_vib,
        check_finished_fun=check_finished_vasp_vib,
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

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
