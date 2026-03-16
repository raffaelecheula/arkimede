# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import yaml

from arkimede.databases import get_names_calculations_to_run
from arkimede.workflow.recipes import run_calculation_from_names

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation = "sella-ba"
    # MLP calculator parameters.
    calc_MLP_name = "OCP_eSEN"
    # DFT calculator parameters.
    calc_DFT_name = "Espresso"
    calc_DFT_kwargs = {"basedir": os.getcwd(), "clean_directory": True, "socket": True}
    # Databases parameters.
    db_inp_name = f"../00_databases/MLP_relax_{calc_MLP_name}.db"
    db_out_name = f"../00_databases/DFT_after_{calc_MLP_name}.db"
    db_inp_kwargs = {"calculation": calculation, "neb_steps": 0, "status": "finished"}
    db_out_kwargs = {"calculation": calculation, "image": "TS"}
    # Shephex parameters.
    use_shephex = True
    
    # TS-search parameters.
    run_TSsearch_kwargs = {
        "fmax": 0.05, # [eV/Å]
        "max_steps": 1000,
        "max_forcecalls": 10000,
        "max_force_tot": 50.,
    }

    # Get names of calculations to run.
    name_list = get_names_calculations_to_run(
        db_inp_name=db_inp_name,
        db_out_name=db_out_name,
        db_inp_kwargs=db_inp_kwargs,
        db_out_kwargs={**db_out_kwargs, "calculation": calculation},
    )

    # Prepare Shephex executor for parallel Slurm jobs.
    if use_shephex is True:
        from shephex import SlurmExecutor, hexperiment
        slurm_config = yaml.safe_load(open("slurm.yaml", "r"))
        executor = SlurmExecutor(**slurm_config["parameters"])
        executor._commands_pre_execution = slurm_config["commands_pre_execution"]

    # Run calculations.
    experiment_list = []
    for name in name_list:
        # Experiment parameters.
        experiment_kwargs = {
            "name": name,
            "db_inp_name": db_inp_name,
            "db_out_name": db_out_name,
            "calculation": calculation,
            "calc_name": calc_DFT_name,
            "calc_kwargs": calc_DFT_kwargs,
            "run_kwargs": run_TSsearch_kwargs,
            "db_inp_kwargs": db_inp_kwargs,
            "db_out_kwargs": db_out_kwargs,
            "basedir": os.getcwd(),
        }
        if use_shephex is True:
            # Set up experiment for parallel Slurm jobs.
            experiment_list.append(
                hexperiment()(run_calculation_from_names)(**experiment_kwargs)
            )
        else:
            # Run experiment in serial.
            run_calculation_from_names(**experiment_kwargs)
    
    # Submit experiments.
    if use_shephex is True:
        print(f"Number of experiments submitted: {len(experiment_list)}")
        executor.execute(experiment_list)

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