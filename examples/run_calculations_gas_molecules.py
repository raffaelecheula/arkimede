# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from ase.calculators.vasp.vasp import Vasp
from arkimede.utilities import get_atoms_list_from_db_metadata
from arkimede.utilities.paths import templates_basedir, scripts_basedir
from arkimede.workflow.reaction_workflow import run_dft_calculations_k8s
from arkimede.workflow.dft_calculations import (
    write_input_asevasp,
    read_output_asevasp,
    check_finished_asevasp,
    job_queued_k8s,
)

# -------------------------------------------------------------------------------------
# ASE VASP VIBRATIONS
# -------------------------------------------------------------------------------------

def main():

    # Name of ase database containing the unrelaxed structures to read.
    db_init_name = "databases/init.db"

    # Select atoms from the ase database.
    selection = ""

    # Name of ase dft database to store the results of the dft calculations.
    db_dft_name = "databases/vasp_gas.db"
    db_dft_append = True
    
    # Keywords used to match structures in the db and to be stored in db columns.
    keys_match = ["name", "calculation"]
    keys_store = ["name", "calculation", "status", "name_ref", "species"]
    
    # Name of the folder for asevasp gas calculations.
    basedir_dft_calc = "calculations/vasp_gas"
    
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
    
    # Initialize ase init database.
    db_init = connect(name=db_init_name)

    # Get lists of atoms from database.
    atoms_list = get_atoms_list_from_db_metadata(
        db_ase=db_init,
        selection=selection,
        metadata_key="gas",
    )
    
    # Initialize ase dft database.
    db_dft = connect(name=db_dft_name, append=db_dft_append)
    
    # Get only molecules of two of more atoms.
    atoms_list = [atoms for atoms in atoms_list if len(atoms) > 2]
    
    # Store calculation settings into atoms.info dictionaries.
    for atoms in atoms_list:
        atoms.pbc = True
        atoms.info["calculation"] = "relax+vibrations"
        atoms.info["vasp_flags"] = vasp_flags
        atoms.info["vasp_command"] = vasp_command
        atoms.info["filename_out"] = filename_out
        atoms.info["save_trajs"] = True
        atoms.info["indices"] = list(range(len(atoms)))
    
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
        write_all_to_db=True,
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

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
