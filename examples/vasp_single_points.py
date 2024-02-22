# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import re
import yaml
import numpy as np
from ase.io import read
from ase.db import connect
from ase.gui.gui import GUI
from ocdata.utils.vasp import calculate_surface_k_points
from ase.calculators.vasp.vasp import Vasp
from arkimede.workflow.utilities import (
    read_atoms_from_db,
    write_atoms_to_db,
)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    
    # Name of ase database.
    db_ase_name = "database.db"

    # Select atoms from the database.
    selection = "calculation=climbbonds,status=finished"
    
    # Name of the template yaml file and namespace for kubernetes submission.
    template_yaml = "template.yaml"
    namespace = "raffaelecheula"
    
    # Name of the folder for dft single point calculations.
    basedir_dft_calc = "vasp_singlepoints"
    
    # Filename and name of dft calculation.
    filename_out = "OUTCAR"
    calculation = "vasp-single-point"
    
    # Name of ase dft database.
    db_ase_dft_name = "database_vasp.db"

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
        "ncore": 1,
        "lcharg": False,
        "lwave": False,
        "gga": "RP",
        "pp": "PBE",
        "xc": "PBE",
    }
    
    # Setup vasp calculator.
    calc = Vasp(**vasp_flags)

    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Get list of atoms from database.
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms_list.append(atoms)

    # Run the dft calculations.
    run_dft_calculations(
        atoms_list=atoms_list,
        calc=calc,
        template_yaml=template_yaml,
        namespace=namespace,
        basedir_dft_calc=basedir_dft_calc,
        filename_out=filename_out,
        calculation=calculation,
        db_ase_dft_name=db_ase_dft_name,
    )

# -----------------------------------------------------------------------------
# RUN DFT CALCULATIONS
# -----------------------------------------------------------------------------

def run_dft_calculations(
    atoms_list,
    calc,
    template_yaml,
    namespace,
    basedir_dft_calc,
    filename_out,
    calculation,
    db_ase_dft_name,
    cpu_req=8,
    cpu_lim=12,
    mem_req='8Gi',
    mem_lim='24Gi',
):
    
    from arkimede.workflow.dft_calculations import (
        submit_k8s,
        read_dft_output,
    )
    
    # Initialize ase dft database.
    db_ase_dft = connect(name=db_ase_dft_name)
    
    # Read the template yaml file.
    with open(template_yaml, 'r') as fileobj:
        params_k8s = yaml.safe_load(fileobj)
    
    cwd = os.getcwd()
    for atoms in atoms_list:
        atoms.pbc = True
        atoms.info["calculation"] = calculation
        if read_atoms_from_db(atoms=atoms, db_ase=db_ase_dft) is None:
            directory = os.path.join(basedir_dft_calc, atoms.info["name"])
            os.makedirs(directory, exist_ok=True)
            os.chdir(directory)
            if os.path.isfile(filename_out):
                read_dft_output(
                    atoms=atoms,
                    filename=filename_out,
                )
                if atoms.info["converged"] is True:
                    atoms.set_tags(np.ones(len(atoms)))
                    write_atoms_to_db(atoms=atoms, db_ase=db_ase_dft)
            else:
                calc.set(kpts=calculate_surface_k_points(atoms))
                calc.write_input(atoms=atoms)
                submit_k8s(
                    params_k8s=params_k8s,
                    run_tag=atoms.info["name"],
                    dirpath=os.path.join(cwd, directory),
                    namespace=namespace,
                    cpu_req=cpu_req,
                    cpu_lim=cpu_lim,
                    mem_req=mem_req,
                    mem_lim=mem_lim,
                )

            os.chdir(cwd)

# -----------------------------------------------------------------------------
# IF NAME MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
