# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
from ase.io import read
from vasp_interactive import VaspInteractive
from arkimede.workflow.dft_calculations import check_file_contains
from arkimede.workflow.calculations import (
    run_relax_calculation,
    run_vibrations_calculation,
    run_calculation,
)

# -----------------------------------------------------------------------------
# VASP ASE CALCULATION
# -----------------------------------------------------------------------------

# Read atoms object.
atoms = read("atoms.traj")
vasp_flags = atoms.info["vasp_flags"]
kpts = atoms.info["kpts"]
filename_out = atoms.info["filename_out"]

# Setup vasp calculator.
with VaspInteractive(**vasp_flags, kpts=kpts) as calc:

    print("Calculations Started")

    if "relax" in atoms.info["calculations"]:
        print("Relax Calculation Started")
        key_string = "Relax Calculation Finished"
        trajname = "relax.traj"
        if check_file_contains(filename_out, key_string) and os.path.isfile(trajname):
            atoms = read(trajname)
        else:
            run_relax_calculation(atoms=atoms, calc=calc, **atoms.info)
        print(key_string)

    if "TS_search" in atoms.info["calculations"]:
        print("TS Search Calculation Started")
        key_string = "TS Search Calculation Finished"
        trajname = atoms.info['search_TS']+".traj"
        if check_file_contains(filename_out, key_string) and os.path.isfile(trajname):
            atoms = read(trajname)
        else:
            run_TS_search_calculation(atoms=atoms, calc=calc, **atoms.info)
        print(key_string)

    if "vibrations" in atoms.info["calculations"]:
        print("Vibrations Calculation Started")
        key_string = "Vibrations Calculation Finished"
        trajname = "vibrations.traj"
        if check_file_contains(filename_out, key_string) and os.path.isfile(trajname):
            atoms = read(trajname)
        else:
            run_vibrations_calculation(atoms=atoms, calc=calc, **atoms.info)
        print(key_string)

    print("Calculations Finished")

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
