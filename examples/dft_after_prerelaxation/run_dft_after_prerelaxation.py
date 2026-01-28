# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from ase.db import connect
from shephex import SlurmExecutor, hexperiment

from arkimede.databases import get_names_from_db, get_atoms_from_db
from arkimede.workflow.recipes import run_calculation_from_names

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation = "sella"
    # MLP calculator parameters.
    calc_MLP_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN
    # DFT calculator parameters.
    calc_DFT_name = "Espresso"
    calc_DFT_kwargs = {"basedir": os.getcwd(), "clean_directory": True, "socket": True}
    # Databases parameters.
    db_inp_name = f"../databases/MLP_relax_{calc_MLP_name}.db"
    db_out_name = f"../databases/DFT_after_{calc_MLP_name}.db"
    db_inp_kwargs = {"calculation": calculation, "neb_steps": 0, "status": "finished"}
    db_out_kwargs = {"calculation": calculation, "image": "TS"}

    # Shephex parameters.
    use_shephex = True
    slurm_kwargs = {
        "partition": "q64",
        "nodes": 1,
        "ntasks-per-node": 32,
        "cpus-per-task" : 1,
        "mem": "251G",
        "time": "08:00:00",
    }
    
    # TS-search parameters.
    run_TSsearch_kwargs = {
        "fmax": 0.05, # [eV/Å]
        "max_steps": 1000,
        "max_forcecalls": 10000,
        "max_force_tot": 50.,
    }

    # Prepare input database and get calculation names.
    db_inp = connect(name=db_inp_name)
    name_list = get_names_from_db(db_ase=db_inp, **db_inp_kwargs)
    print(f"Number of calculations:", len(name_list))

    # Prepare output database.
    db_out = connect(name=db_out_name, use_lock_file=True, append=True)

    # Run calculations.
    experiment_list = []
    for name in name_list:
        # Skip experiments if final results in output db.
        db_kwargs = {**db_out_kwargs, "name": name, "calculation": calculation}
        if get_atoms_from_db(db_ase=db_out, none_ok=True, **db_kwargs) is not None:
            continue
        # Set up experiment.
        exp_kwargs = {
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
        # Set up experiment.
        if use_shephex is True:
            # Set up experiment for parallel runs.
            experiment_list.append(
                hexperiment()(run_calculation_from_names)(**exp_kwargs)
            )
        else:
            # Run experiment in serial.
            run_calculation_from_names(**exp_kwargs)
    
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