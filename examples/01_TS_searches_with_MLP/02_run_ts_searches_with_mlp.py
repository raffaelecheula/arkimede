# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

from arkimede.databases import get_names_calculations_to_run
from arkimede.workflow.recipes import search_TS_from_names

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation_TS = "sella-ba" # neb | dimer | climbfixint | sella | sella-ba
    neb_steps = 0
    n_restarts_TSsearch = 5
    # MLP calculator parameters.
    calc_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN | CHGNet_MP | MACE_MPA0 | FAIRChem_UMA
    calc_kwargs = {"logfile": "log.txt"}
    # Databases parameters.
    db_inp_name = f"../00_databases/SAA_DFT_reactions.db"
    db_out_name = f"../00_databases/MLP_relax_{calc_name}.db"
    db_inp_kwargs = {}
    db_out_kwargs = {"neb_steps": neb_steps}

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
        "db_out_kwargs": db_out_kwargs,
    }
    # TS-search parameters.
    run_TSsearch_kwargs = {
        "fmax": 0.05, # [eV/Å]
        "max_steps": 1000,
        "max_forcecalls": 10000,
        "db_out_kwargs": db_out_kwargs,
    }
    # IS and FS check parameters.
    check_IS_FS_kwargs = {
        "method": "vibrations",
        "indices_check": "adsorbate",
    }

    # Get names of calculations to run.
    name_list = get_names_calculations_to_run(
        db_inp_name=db_inp_name,
        db_out_name=db_out_name,
        db_inp_kwargs=db_inp_kwargs,
        db_out_kwargs={**db_out_kwargs, "calculation": calculation_TS},
    )

    # Run the calculations.
    experiment_list = []
    for name in name_list:
        # Experiment parameters.
        experiment_kwargs = {
            "name": name,
            "db_inp_name": db_inp_name,
            "db_out_name": db_out_name,
            "calc_name": calc_name,
            "calc_kwargs": calc_kwargs,
            "calculation_TS": calculation_TS,
            "run_relax_kwargs": run_relax_kwargs,
            "run_NEB_kwargs": run_NEB_kwargs,
            "run_TSsearch_kwargs": run_TSsearch_kwargs,
            "check_IS_FS_kwargs": check_IS_FS_kwargs,
            "n_restarts_TSsearch": n_restarts_TSsearch,
            "db_inp_kwargs": db_inp_kwargs,
            "basedir": os.getcwd(),
        }
        # Run experiment.
        search_TS_from_names(**experiment_kwargs)

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