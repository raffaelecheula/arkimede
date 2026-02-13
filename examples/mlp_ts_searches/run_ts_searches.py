# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from ase.db import connect
from shephex import SlurmExecutor, hexperiment

from arkimede.databases import get_names_from_db, get_atoms_from_db
from arkimede.workflow.recipes import search_TS_from_names

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Calculation parameters.
    calculation_TS = "sella" # neb | dimer | climbfixint | sella | sella-ba
    neb_steps = 0
    # MLP calculator parameters.
    calc_MLP_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN
    calc_MLP_kwargs = {"logfile": "log.txt"}
    # Databases parameters.
    db_inp_name = f"../databases/SAA_DFT_reactions.db"
    db_out_name = f"../databases/MLP_relax_{calc_MLP_name}.db"
    db_inp_kwargs = {}
    db_out_kwargs = {"neb_steps": neb_steps}

    # Shephex parameters.
    use_shephex = False
    slurm_kwargs = {
        "partition": "ql40s",
        "nodes": 1,
        "ntasks-per-node": 16,
        "cpus-per-task" : 1,
        "gpus-per-node" : 1,
        "mem": "187G",
        "time": "02:00:00",
    }
    
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

    # Prepare input database and get calculation names.
    db_inp = connect(name=db_inp_name)
    name_list = get_names_from_db(db_ase=db_inp)
    print(f"Number of calculations:", len(name_list))

    # Prepare output database.
    db_out = connect(name=db_out_name, use_lock_file=True, append=True)

    # Run calculations.
    experiment_list = []
    for name in name_list:
        # Skip experiments if final results in output db.
        db_kwargs = {**db_out_kwargs, "name": name, "calculation": calculation_TS}
        if get_atoms_from_db(db_ase=db_out, none_ok=True, **db_kwargs) is not None:
            continue
        # Set up experiment.
        exp_kwargs = {
            "name": name,
            "db_inp_name": db_inp_name,
            "db_out_name": db_out_name,
            "calc_name": calc_MLP_name,
            "calc_kwargs": calc_MLP_kwargs,
            "calculation_TS": calculation_TS,
            "run_relax_kwargs": run_relax_kwargs,
            "run_NEB_kwargs": run_NEB_kwargs,
            "run_TSsearch_kwargs": run_TSsearch_kwargs,
            "check_IS_FS_kwargs": check_IS_FS_kwargs,
            "db_inp_kwargs": db_inp_kwargs,
            "basedir": os.getcwd(),
        }
        # Set up experiment.
        if use_shephex is True:
            # Set up experiment for parallel runs.
            experiment_list.append(
                hexperiment()(search_TS_from_names)(**exp_kwargs)
            )
        else:
            # Run experiment in serial.
            search_TS_from_names(**exp_kwargs)
    
    # Submit experiments.
    if use_shephex is True:
        print(f"Number of calculations submitted:", len(experiment_list))
        executor = SlurmExecutor(**slurm_kwargs)
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