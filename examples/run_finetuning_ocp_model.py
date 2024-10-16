# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
from arkimede.utilities import read_step_actlearn, write_step_actlearn
from arkimede.ocp.ocp_utils import (
    ocp_main,
    get_checkpoint_path,
    get_checkpoint_path_actlearn,
    split_database,
    merge_databases,
    update_config_yaml,
    fine_tune_ocp_model,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Active learning step.
    filename_actlearn = "actlearn.yaml"
    step_actlearn = read_step_actlearn(filename=filename_actlearn)
    
    # Name of dft databases.
    db_name_list = [f"databases/vasp_{ii:02d}.db" for ii in range(step_actlearn+1)]
    
    # Merge databases into one total database.
    db_tot_name = "databases/vasp_tot.db"
    merge_databases(db_name_list=db_name_list, db_new_name=db_tot_name)
    
    # Directory, identifier, and new yaml config file for fine-tuning.
    directory = "finetuning"
    identifier = f"step-{step_actlearn+1:02d}"
    config_yaml = f"{directory}/config.yml"
    
    # OCPmodels checkpoint path.
    checkpoint_key = 'GemNet-OC OC20+OC22 v2'
    if step_actlearn == 0:
        checkpoint_path = get_checkpoint_path(checkpoint_key=checkpoint_key)
    else:
        checkpoint_path = get_checkpoint_path_actlearn(step_actlearn=step_actlearn)
    
    # Split the database into train, test (optional), and val databases.
    split_database(
        db_ase_name=db_tot_name,
        fractions=(0.8, 0.2),
        filenames=("train.db", "val.db"),
        directory=directory,
        seed=step_actlearn*42,
        change_energy_ref_fun=None,
    )
    
    # Update config file.
    delete_keys = [
        'slurm',
        'cmd',
        'logger',
        'task',
        'model_attributes',
        'dataset',
        'test_dataset',
        'val_dataset',
        'optim.loss_force',
    ]
    update_keys = {
        'gpus': 1,
        'task.dataset': 'ase_db',
        'optim.eval_every': 4,
        'optim.max_epochs': 1000,
        'optim.lr_initial': 1e-5,
        'optim.batch_size': 1,
        'optim.num_workers': 4,
        'dataset.train.src': f"{directory}/train.db",
        'dataset.train.a2g_args.r_energy': True,
        'dataset.train.a2g_args.r_forces': True,
        'dataset.val.src': f"{directory}/val.db",
        'dataset.val.a2g_args.r_energy': True,
        'dataset.val.a2g_args.r_forces': True,
        'task.primary_metric': "forces_mae",
        'logger': 'wandb',
        #'optim.early_stopping_lr': 1e-9,
    }
    update_config_yaml(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        delete_keys=delete_keys,
        update_keys=update_keys,
    )
    
    # Run the fine-tuning.
    fine_tune_ocp_model(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        directory=directory,
        identifier=identifier,
    )

    # Update active learning step.
    step_actlearn += 1
    write_step_actlearn(filename=filename_actlearn, step=step_actlearn)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
