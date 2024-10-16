# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
from ase.io import read
from ase.db import connect
from ase.io import Trajectory
from vasp_interactive import VaspInteractive
from arkimede.utilities import write_atoms_to_db
from arkimede.workflow.calculations import run_calculation
from arkimede.ocp.ocp_calc import OCPCalculatorCounter
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
# DEFAULTS
# -------------------------------------------------------------------------------------

# TODO: put these into arkimede.ocp.ocp_utils?
delete_keys_default = [
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
update_keys_default = {
    'gpus': 1,
    'task.dataset': 'ase_db',
    'optim.eval_every': 100,
    'optim.max_epochs': 100, # 10
    'optim.lr_initial': 1e-5, # 5e-6 # 1e-7 probably too low
    'optim.batch_size': 1,
    'optim.num_workers': 4, # 0
    #'optim.energy_coefficient': 0,
    'dataset.train.src': "finetuning/vasp.db",
    'dataset.train.a2g_args.r_energy': True,
    'dataset.train.a2g_args.r_forces': True,
    'dataset.val.src': "finetuning/vasp.db",
    'dataset.val.a2g_args.r_energy': True,
    'dataset.val.a2g_args.r_forces': True,
    'task.primary_metric': "forces_mae",
    'logger': 'wandb',
}

# -------------------------------------------------------------------------------------
# ACTIVE LEARNING OPTIMIZATION
# -------------------------------------------------------------------------------------

# Read atoms object.
atoms = read("aloptvasp.traj")

# TODO: just for testing, these should be already in atoms!
atoms.set_tags([1.]*len(atoms))
atoms.info["name"] = "aloptvasp"
atoms.info["min_steps"] = 10
atoms.info["max_steps"] = 1000
atoms.info["fmax"] = 0.05
atoms.info["max_step_actlearn"] = 100
atoms.info["vasp_flags"]["lwave"] = True

# Calculation settings.
vasp_flags = atoms.info["vasp_flags"]
kpts = atoms.info["kpts"]
calculation = atoms.info["calculation"]
max_step_actlearn = atoms.info["max_step_actlearn"]
fmax = atoms.info["fmax"]

# OCPmodels ase calculator.
checkpoint_key = 'GemNet-OC OC20+OC22 v2'
checkpoint_path = get_checkpoint_path(checkpoint_key=checkpoint_key)
calc = OCPCalculatorCounter(checkpoint_path=checkpoint_path, cpu=False)

# Initialize ase database.
os.makedirs("finetuning", exist_ok=True)
db_ase = connect(name="finetuning/vasp.db")

# Create a new config.yaml file.
update_config_yaml(
    checkpoint_path=checkpoint_path,
    config_dict=calc.config,
    config_yaml="finetuning/config.yml",
    delete_keys=delete_keys_default,
    update_keys=update_keys_default,
)

# Setup vasp calculator and run the calculation.
with VaspInteractive(**vasp_flags, directory="vasp", kpts=kpts) as calc_dft:
    print(calculation+" calculation started")
    for ii in range(max_step_actlearn):
        run_calculation(atoms=atoms, calc=calc, **atoms.info)
        atoms.calc = calc_dft
        forces = atoms.get_forces()
        max_force = (forces ** 2).sum(axis=1).max() ** 0.5
        print(f"vasp fmax = {max_force:+7.4f}")
        if max_force < fmax:
            break
        write_atoms_to_db(
            atoms=atoms,
            db_ase=db_ase,
            keys_store=[],
            get_status=False,
        )
        fine_tune_ocp_model(
            checkpoint_path=checkpoint_path,
            config_yaml="finetuning/config.yml",
            identifier=f"step-{ii+1:03d}"
        )
        checkpoint_path = get_checkpoint_path_actlearn(step_actlearn=ii+1)
        calc = OCPCalculatorCounter(checkpoint_path=checkpoint_path, cpu=False)
    print(calculation+" calculation finished")
    
# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
