# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
from ocp_utils import (
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
    identifier = "train-test"
    
    # Split the model into databases.
    train_test_val_split(
        db_ase_name=db_dft_name,
        fractions=(0.9, 0.1),
        filenames=("train.db", "val.db"),
        directory=directory,
        seed=42,
    )
    
    # New yaml config and output files.
    config_yaml = f"{directory}/config.yml"
    output = f"{directory}/{identifier}.txt"
    
    # OCPmodels checkpoint path.
    checkpoint_key = 'EquiformerV2 (31M) All+MD'
    checkpoint_path = get_checkpoint_path(
        key=checkpoint_key,
        basedir=checkpoints_basedir(),
    )
    
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
        'optim.scheduler',
        'optim.scheduler_params',
    ]
    
    update_keys = {
        'gpus': 1,
        'task.dataset': 'ase_db',
        'optim.batch_size': 1,
        'optim.eval_every': 1,
        'optim.max_epochs': 10,
        'optim.lr_initial': 4e-5,
        'optim.scheduler': 'ReduceLROnPlateau',
        'optim.scheduler_params.mode': 'min',
        'optim.scheduler_params.factor': 0.8,
        'optim.scheduler_params.patience': 3,
        'dataset.train.src': 'train.db',
        'dataset.train.a2g_args.r_energy': True,
        'dataset.train.a2g_args.r_forces': True,
        'dataset.train.key_mapping.y': 'energy',
        'dataset.train.key_mapping.force': 'forces',
        'dataset.train.key_mapping.stress': 'stress',
        'dataset.val.src': 'val.db',
        'dataset.val.a2g_args.r_energy': True,
        'dataset.val.a2g_args.r_forces': True,
        'dataset.val.key_mapping.y': 'energy',
        'dataset.val.key_mapping.force': 'forces',
        'dataset.val.key_mapping.stress': 'stress',
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
    command += " > train.txt"
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
