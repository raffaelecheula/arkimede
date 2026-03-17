# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
from ase.db import connect
os.environ["MKL_THREADING_LAYER"] = "GNU"

from arkimede.workflow.recipes import search_TS_from_atoms_IS_and_FS

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation_TS = "sella-ba" # neb | dimer | climbfixint | sella | sella-ba
    neb_steps = 0
    n_restarts_TSsearch = 0

    # Relax parameters.
    run_relax_kwargs = {
        "fmax": 0.05, # [eV/Å]
        "max_steps": 500,
    }
    # NEB parameters.
    run_NEB_kwargs = {
        "fmax": 0.05, # [eV/Å]
        "max_steps": neb_steps,
        "n_images": 10,
    }
    # TS-search parameters.
    run_TSsearch_kwargs = {
        "fmax": 0.05, # [eV/Å]
        "max_steps": 500,
        "max_forcecalls": 1000,
    }
    # IS and FS check parameters.
    check_IS_FS_kwargs = {
        "method": "vibrations",
        "indices_check": "adsorbate",
    }

    # Get IS and FS from database.
    from arkimede.databases import get_atoms_from_db
    name = "Rh(111)+Pt1_H2O*_to_OH*+H*_top008_to_3fh001+3fh014"
    db_inp = connect(name=f"../00_databases/SAA_DFT_reactions.db")
    atoms_IS = get_atoms_from_db(db_ase=db_inp, name=name, image="IS")
    atoms_FS = get_atoms_from_db(db_ase=db_inp, name=name, image="FS")
    # Add info on TS-search parameters.
    info = {"name": name, "bonds_TS": [[16, 17, "break"]], "indices_ads": [16, 17, 18]}
    atoms_IS.info = info
    atoms_FS.info = info
    
    # Get calculator.
    from mlps_finetuning.calculators import get_calculator
    calc = get_calculator(calc_name="OCP_eSEN", logfile="log.txt")

    # Run TS-search calculation from IS and FS atoms.
    atoms_TS = search_TS_from_atoms_IS_and_FS(
        atoms_IS=atoms_IS,
        atoms_FS=atoms_FS,
        calc=calc,
        db_out=None,
        calculation_TS=calculation_TS,
        run_relax_kwargs=run_relax_kwargs,
        run_NEB_kwargs=run_NEB_kwargs,
        run_TSsearch_kwargs=run_TSsearch_kwargs,
        check_IS_FS_kwargs=check_IS_FS_kwargs,
        n_restarts_TSsearch=n_restarts_TSsearch,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    # Run script and measure execution time.
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print(f"Execution time = {time_stop - time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------