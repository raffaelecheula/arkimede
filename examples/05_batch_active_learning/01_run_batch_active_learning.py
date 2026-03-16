# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import yaml

from arkimede.databases import get_names_calculations_to_run
from arkimede.workflow.active_learning import run_active_learning_list_from_names

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation = "sella-ba"
    group_by_surface = True
    # MLP calculator parameters.
    calc_MLP_name = "OCP_eSEN"
    calc_MLP_kwargs = {"logfile": "log.txt"}
    # DFT calculator parameters.
    calc_DFT_name = "Espresso"
    calc_DFT_kwargs = {"basedir": os.getcwd(), "clean_directory": True}
    # Databases parameters.
    db_inp_name = f"../00_databases/MLP_relax_{calc_MLP_name}.db"
    db_out_name = f"../00_databases/AL_groups_{calc_MLP_name}.db"
    db_inp_kwargs = {"calculation": calculation, "neb_steps": 0, "status": "finished"}
    db_out_kwargs = {"calculation": calculation, "image": "TS"}
    # Shephex parameters.
    use_shephex = True

    # MLP calculation parameters.
    run_MLP_kwargs = {
        "fmax": 0.01, # [eV/Å]
        "max_steps": 1000,
        "min_steps": 10,
    }
    # DFT calculation parameters.
    run_DFT_kwargs = {
        "fmax": 0.05, # [eV/Å]
    }
    # Fine-tuning parameters.
    finetune_kwargs = {
        "epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "logfile": "log.txt",
        "energy_coeff": 0.0,
        "forces_coeff": 1.0,
        "stress_coeff": 0.0,
        "calc_kwargs": calc_MLP_kwargs,
    }
    
    # Get names of calculations to run.
    name_list = get_names_calculations_to_run(
        db_inp_name=db_inp_name,
        db_out_name=db_out_name,
        db_inp_kwargs=db_inp_kwargs,
        db_out_kwargs=db_out_kwargs,
    )

    # Prepare Shephex executor for parallel Slurm jobs.
    if use_shephex is True:
        from shephex import SlurmExecutor, hexperiment
        slurm_config = yaml.safe_load(open("slurm.yaml", "r"))
        executor = SlurmExecutor(**slurm_config["parameters"])
        executor._commands_pre_execution = slurm_config["commands_pre_execution"]

    # Group names by surface.
    if group_by_surface is True:
        surface_dict = {}
        for name in name_list:
            surface = name.split("_")[0]
            if surface not in surface_dict:
                surface_dict[surface] = []
            surface_dict[surface].append(name)
        group_list = surface_dict.values()
    else:
        group_list = [name_list]
    
    # Run calculations.
    experiment_list = []
    for name_list in group_list:
        # Experiment parameters.
        experiment_kwargs = {
            "name_list": name_list,
            "db_inp_name": db_inp_name,
            "db_out_name": db_out_name,
            "calculation": calculation,
            "calc_MLP_name": calc_MLP_name,
            "calc_DFT_name": calc_DFT_name,
            "calc_MLP_kwargs": calc_MLP_kwargs,
            "calc_DFT_kwargs": calc_DFT_kwargs,
            "run_MLP_kwargs": run_MLP_kwargs,
            "run_DFT_kwargs": run_DFT_kwargs,
            "finetune_kwargs": finetune_kwargs,
            "db_inp_kwargs": db_inp_kwargs,
            "db_out_kwargs": db_out_kwargs,
            "basedir": os.getcwd(),
        }
        if use_shephex is True:
            # Set up experiment for parallel Slurm jobs.
            experiment_list.append(
                hexperiment()(run_active_learning_list_from_names)(**experiment_kwargs)
            )
        else:
            # Run experiment in serial.
            run_active_learning_list_from_names(**experiment_kwargs)

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
