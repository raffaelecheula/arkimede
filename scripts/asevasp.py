# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
from ase.io import read
from vasp_interactive import VaspInteractive
from arkimede.workflow.dft_calculations import check_file_contains
from arkimede.workflow.calculations import run_calculation

# -------------------------------------------------------------------------------------
# ASEVASP CALCULATION
# -------------------------------------------------------------------------------------

# Read atoms object.
atoms = read("asevasp.traj")
vasp_flags = atoms.info["vasp_flags"]
kpts = atoms.info["kpts"]
filename_out = atoms.info["filename_out"]

# Setup vasp calculator and run the calculations.
with VaspInteractive(**vasp_flags, directory="vasp", kpts=kpts) as calc:
    for calculation in atoms.info["calculation"].split("+"):
        key_string = calculation+" calculation finished"
        trajname = calculation+".traj"
        if (
            check_file_contains(filename=filename_out, key_string=key_string)
            and os.path.isfile(trajname)
        ):
            atoms = read(trajname)
        else:
            print(calculation+" calculation started")
            atoms.info["calculation"] = calculation
            run_calculation(atoms=atoms, calc=calc, **atoms.info)
            print(key_string)
    print("all calculations finished")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
