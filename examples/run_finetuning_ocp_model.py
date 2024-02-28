# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
from arkimede.ocp.ocp_utils import (
    ocp_main,
    get_checkpoint_path,
    checkpoints_basedir,
    train_test_val_split,
    update_config_yaml,
)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    
    # Name of ase dft database.
    db_dft_name = "database_vasp.db"
    
    # Directory and identifier for fine-tuning.
    directory = "fine-tuning"
    identifier = "fine-tuning"
    
    # Split the model into databases.
    train_test_val_split(
        db_ase_name=db_dft_name,
        fractions=(0.8, 0.1, 0.1),
        filenames=("train.db", "test.db", "val.db"),
        directory=directory,
        seed=42,
    )
    
    # New yaml config and output files.
    config_yaml = f"{directory}/config.yml"
    output = f"{directory}/{identifier}.txt"
    
    # OCPmodels checkpoint path.
    checkpoint_key = 'GemNet-OC OC20+OC22' # 'EquiformerV2 (31M) All+MD'
    checkpoint_path = get_checkpoint_path(checkpoint_key=checkpoint_key)
    
    
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
        #'optim.scheduler',
        #'optim.scheduler_params',
    ]
    
    update_keys = {
        'gpus': 1,
        'task.dataset': 'ase_db',
        'optim.eval_every': 4,
        'optim.max_epochs': 1000,
        'optim.lr_initial': 4e-5,
        'optim.batch_size': 4,
        'optim.num_workers': 4,
        'dataset.train.src': f"{directory}/train.db",
        'dataset.train.a2g_args.r_energy': True,
        'dataset.train.a2g_args.r_forces': True,
        'dataset.val.src': f"{directory}/val.db",
        'dataset.val.a2g_args.r_energy': True,
        'dataset.val.a2g_args.r_forces': True,
        'task.primary_metric': "forces_mae",
        'logger': 'wandb',
        'optim.early_stopping_lr': 1e-8,
    }
    
    update_config_yaml(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        delete_keys=delete_keys,
        update_keys=update_keys,
    )
    
    args_dict = {
        "mode": "train",
        "config-yml": config_yaml,
        "checkpoint": checkpoint_path,
        "run-dir": directory,
        "identifier": identifier,
        "amp": "",
    }
    
    command = f"python {ocp_main()}"
    for arg in args_dict:
        command += f" --{arg} {args_dict[arg]}"
    if output is not None:
        command += f" > {output} 2>&1"
    os.system(command)

# -----------------------------------------------------------------------------
# IF NAME MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
