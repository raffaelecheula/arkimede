# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from ocp_utils import get_checkpoint_path, checkpoints_basedir

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    
    # TODO: finish this!
    
    # Name of ase dft database.
    db_ase_dft_name = "database_vasp.db"
    
    train, test, val = train_test_val_split(db_ase_dft_name)
    
    yml_config = generate_yml_config(
        checkpoint,
        'config.yml',
        delete=[
            'slurm',
            'cmd',
            'logger',
            'task',
            'model_attributes',
            'optim.loss_force',
            'dataset',
            'test_dataset',
            'val_dataset',
        ],
        update={
            'gpus': 1,
            'task.dataset': 'ase_db',
            'optim.eval_every': 1,
            'optim.max_epochs': 10,
            # Train data.
            'dataset.train.src': 'train.db',
            'dataset.train.a2g_args.r_energy': True,
            'dataset.train.a2g_args.r_forces': True,
            # Test data (prediction only so no regression).
            'dataset.test.src': 'test.db',
            'dataset.test.a2g_args.r_energy': False,
            'dataset.test.a2g_args.r_forces': False,
            # Val data.
            'dataset.val.src': 'val.db',
            'dataset.val.a2g_args.r_energy': True,
            'dataset.val.a2g_args.r_forces': True,
        }
    )
    

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
