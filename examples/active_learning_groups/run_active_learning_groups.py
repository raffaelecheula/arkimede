# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
from ase import Atoms
from ase.db import connect
from shephex import SlurmExecutor, hexperiment

from arkimede.databases import get_names_from_db, get_atoms_from_db
from arkimede.workflow.active_learning import run_active_learning_list_from_names

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation = "sella"
    # MLP calculator parameters.
    calc_MLP_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN
    calc_MLP_kwargs = {"logfile": "log.txt"}
    # DFT calculator parameters.
    calc_DFT_name = "Espresso"
    calc_DFT_kwargs = {"basedir": os.getcwd(), "clean_directory": True}
    # Databases parameters.
    db_inp_name = f"../databases/MLP_relax_{calc_MLP_name}.db"
    db_out_name = f"../databases/AL_groups_{calc_MLP_name}.db"
    db_inp_kwargs = {"calculation": calculation, "neb_steps": 0, "status": "finished"}
    db_out_kwargs = {"calculation": calculation, "image": "TS"}

    # Shephex parameters.
    use_shephex = True
    slurm_kwargs = {
        "partition": "ql40s",
        "nodes": 1,
        "ntasks-per-node": 32,
        "cpus-per-task" : 1,
        "gpus-per-node" : 1,
        "mem": "377G",
        "time": "72:00:00",
    }

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
    
    # Prepare input database and get calculation names.
    db_inp = connect(name=db_inp_name)
    name_list = get_names_from_db(db_ase=db_inp, **db_inp_kwargs)
    print(f"Number of calculations:", len(name_list))

    # Group names by surface.
    surface_dict = {}
    for name in name_list:
        surface = name.split("_")[0]
        if surface not in surface_dict:
            surface_dict[surface] = []
        surface_dict[surface].append(name)
    
    # Prepare output database.
    db_out = connect(name=db_out_name, use_lock_file=True, append=True)

    # Run calculations.
    experiment_list = []
    for surface, name_list in surface_dict.items():
        # Skip experiments if final results in output db.
        db_kwargs = {**db_out_kwargs, "name": name_list[0]}
        if get_atoms_from_db(db_ase=db_out, none_ok=True, **db_kwargs) is not None:
            continue
        # Set up experiment.
        exp_kwargs = {
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
        # Set up experiment.
        if use_shephex is True:
            # Set up experiment for parallel runs.
            experiment_list.append(
                hexperiment()(run_active_learning_list_from_names)(**exp_kwargs)
            )
        else:
            # Run experiment in serial.
            run_active_learning_list_from_names(**exp_kwargs)

    # Submit experiments.
    if use_shephex is True:
        print(f"Number of calculations submitted:", len(experiment_list))
        commands_pre_execution = [
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            "module load intel/2020.1 openmpi/4.0.3",
        ]
        executor = SlurmExecutor(**slurm_kwargs)
        executor._commands_pre_execution = commands_pre_execution
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
    print(f"Execution time = {time_stop-time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
