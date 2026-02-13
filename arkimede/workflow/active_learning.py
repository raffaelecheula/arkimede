# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import shutil
import timeit
import numpy as np
from copy import deepcopy
from ase import Atoms, atoms
from ase.calculators.calculator import Calculator
from ase.io import Trajectory
from ase.db import connect

from arkimede.utilities import print_title, filter_results, copy_atoms_and_results
from arkimede.databases import write_atoms_list_to_db
from arkimede.workflow.calculations import run_calculation
from arkimede.databases import get_atoms_from_db, write_atoms_to_db

from mlps_finetuning.calculators import get_calculator, get_finetune_function

# -------------------------------------------------------------------------------------
# RUN ACTIVE LEARNING
# -------------------------------------------------------------------------------------

def run_active_learning(
    atoms: Atoms,
    calculation: str,
    calc_MLP: Calculator,
    calc_DFT: Calculator,
    finetune_fun: callable,
    finetune_kwargs: dict = {},
    directory: str = "finetuning",
    max_steps_actlearn: int = 300,
    run_MLP_kwargs: dict = {},
    run_DFT_kwargs: dict = {},
    train_on_all: bool = False,
    train_finetuned: bool = False,
    match_energies: bool = True,
    update_parameters_DFT: bool = False,
    save_trajs: bool = False,
    clean_directory: bool = True,
):
    """
    Run active learning calculation.
    """
    # Default finetuning parameters.
    finetune_kwargs = {
        "clean_directory": False,
        "val_fraction": 0.0,
        "test_fraction": 0.0,
        **finetune_kwargs,
    }
    # Store a copy of the MLP calculator.
    calc_MLP_copy = deepcopy(calc_MLP)
    # Get DFT fmax.
    fmax_DFT = run_DFT_kwargs.get("fmax", 0.05)
    # Prepare fine-tuning directory.
    if os.path.isdir(directory) and clean_directory:
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    # Trajectory.
    trajectory = Trajectory(f"{calculation}.traj", mode="w") if save_trajs else None
    # Update DFT calculator parameters.
    if update_parameters_DFT:
        parameters_DFT = calc_DFT.parameters.copy()
        calc_DFT.parameters.update(atoms.info.get("parameters_DFT", {}))
    # Run the active learning protocol.
    time_MLP = 0.
    time_DFT = 0.
    time_train = 0.
    atoms_train = []
    for ii in range(max_steps_actlearn + 1):
        # Run MLP calculation.
        print_title(f"MLP {calculation} calculation.")
        time_MLP_start = timeit.default_timer()
        atoms.info["task"] = "MLP"
        run_calculation(
            atoms=atoms,
            calculation=calculation,
            calc=calc_MLP,
            trajectory=trajectory,
            **run_MLP_kwargs,
        )
        energy_MLP = atoms.get_potential_energy()
        time_MLP += timeit.default_timer() - time_MLP_start
        # Run DFT single-point calculation.
        print_title("DFT single-point calculation.")
        time_DFT_start = timeit.default_timer()
        atoms.info["task"] = "DFT"
        run_calculation(
            atoms=atoms,
            calculation="singlepoint",
            calc=calc_DFT,
            no_constraints=True,
            **run_DFT_kwargs,
        )
        time_DFT += timeit.default_timer() - time_DFT_start
        # Write structure with DFT forces to trajectory.
        if save_trajs:
            trajectory.write(atoms=atoms, **filter_results(atoms.calc.results))
        # Check DFT max force.
        max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
        print(f"DFT fmax    = {max_force:+7.4f} [eV/Å]")
        # Stop if the calculation has converged.
        if max_force < fmax_DFT:
            atoms.info["status"] = "finished"
            break
        # Stop if the maximum number of steps is reached.
        if ii == max_steps_actlearn:
            atoms.info["status"] = "unfinished"
            break
        forces_DFT = atoms.get_forces(apply_constraint=False).copy()
        # Set DFT energy equal to MLP energy.
        if match_energies:
            atoms.calc.results["energy"] = energy_MLP
        # Train on all DFT single-points or only on the last one.
        if train_on_all:
            atoms_train.append(copy_atoms_and_results(atoms=atoms))
        else:
            atoms_train = [copy_atoms_and_results(atoms=atoms)]
        # Run MLP model fine-tuning.
        print_title("MLP fine-tuning.")
        time_train_start = timeit.default_timer()
        calc_MLP = finetune_fun(
            atoms_list=atoms_train,
            calc=calc_MLP if train_finetuned else calc_MLP_copy,
            label=f"model_{ii + 1:02d}",
            directory=directory,
            **finetune_kwargs,
        )
        time_train += timeit.default_timer() - time_train_start
        # Print fine-tuned MLP errors.
        atoms.calc = calc_MLP
        forces_MLP = atoms.get_forces(apply_constraint=False)
        deltaforces = forces_MLP - forces_DFT
        mae_forces = np.linalg.norm(deltaforces, axis=1).mean()
        emax_forces = np.linalg.norm(deltaforces, axis=1).max()
        print(f"MAE forces  = {mae_forces:+7.4f} [eV/Å]")
        print(f"Emax forces = {emax_forces:+7.4f} [eV/Å]")
    # Restore original DFT parameters.
    if update_parameters_DFT:
        calc_DFT.parameters = parameters_DFT
    # Calculation finished.
    print_title(f"Active-learning {calculation} calculation finished.")
    atoms.info.update({
        "DFT_steps": ii + 1,
        "time_MLP": time_MLP, 
        "time_DFT": time_DFT,
        "time_train": time_train,
    })
    print("Calculation results:")
    steps = atoms.info["DFT_steps"]
    print(f"DFT_steps  = {steps:6.0f}")
    print(f"time_MLP   = {time_MLP:6.1f} [s]")
    print(f"time_DFT   = {time_DFT:6.1f} [s]")
    print(f"time_train = {time_train:6.1f} [s]")
    # Return atoms.
    return atoms

# -------------------------------------------------------------------------------------
# RUN ACTIVE LEARNING LIST
# -------------------------------------------------------------------------------------

def run_active_learning_list(
    atoms_list: list,
    calculation: str,
    calc_MLP: Calculator,
    calc_DFT: Calculator,
    finetune_fun: callable,
    finetune_kwargs: dict = {},
    directory: str = "finetuning",
    max_steps_actlearn: int = 100,
    run_MLP_kwargs: dict = {},
    run_DFT_kwargs: dict = {},
    train_on_all: bool = False,
    train_finetuned: bool = False,
    match_energies: bool = True,
    update_parameters_DFT: bool = False,
    save_trajs: bool = False,
    clean_directory: bool = True,
):
    """
    Run active learning calculations.
    """
    # Default finetuning parameters.
    finetune_kwargs = {
        "clean_directory": False,
        "val_fraction": 0.0,
        "test_fraction": 0.0,
        **finetune_kwargs,
    }
    # Store a copy of the MLP calculator.
    calc_MLP_copy = deepcopy(calc_MLP)
    # Get DFT fmax.
    fmax_DFT = run_DFT_kwargs.get("fmax", 0.05)
    # Prepare fine-tuning directory.
    if os.path.isdir(directory) and clean_directory:
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    # Trajectories.
    trajectory_list = [
        Trajectory(f"{calculation}_{jj:02d}.traj", mode="w") if save_trajs else None
        for jj in range(len(atoms_list))
    ]
    # Store original DFT parameters.
    if update_parameters_DFT:
        parameters_DFT = calc_DFT.parameters.copy()
    # Run the active learning protocol.
    time_MLP = 0.
    time_DFT = 0.
    time_train = 0.
    atoms_train = []
    energy_dict_MLP = {}
    forces_dict_DFT = {}
    indices_active = list(range(len(atoms_list)))
    for ii in range(max_steps_actlearn + 1):
        # Run MLP calculation.
        print_title(f"MLP {calculation} calculations.")
        time_MLP_start = timeit.default_timer()
        for jj in indices_active:
            atoms_list[jj].info["task"] = "MLP"
            run_calculation(
                atoms=atoms_list[jj],
                calculation=calculation,
                calc=calc_MLP,
                trajectory=trajectory_list[jj],
                **run_MLP_kwargs,
            )
            energy_dict_MLP[jj] = atoms_list[jj].get_potential_energy()
        time_MLP += timeit.default_timer() - time_MLP_start
        # Train on all DFT single-points or only on the last one.
        if not train_on_all:
            atoms_train = []
        # Run DFT single-point calculation.
        print_title("DFT single-point calculations.")
        time_DFT_start = timeit.default_timer()
        for jj in indices_active.copy():
            # Update DFT calculator parameters.
            if update_parameters_DFT:
                calc_DFT.parameters = {
                    **parameters_DFT,
                    **atoms_list[jj].info.get("parameters_DFT", {}),
                }
            # Run DFT calculation.
            atoms_list[jj].info["task"] = "DFT"
            run_calculation(
                atoms=atoms_list[jj],
                calculation="singlepoint",
                calc=calc_DFT,
                no_constraints=True,
                **run_DFT_kwargs,
            )
            # Write structure with DFT forces to trajectory.
            if save_trajs:
                results = filter_results(atoms_list[jj].calc.results)
                trajectory_list[jj].write(atoms=atoms_list[jj], **results)
            # Check DFT max force.
            max_force = np.linalg.norm(atoms_list[jj].get_forces(), axis=1).max()
            print(f"DFT fmax    = {max_force:+7.4f} [eV/Å]")
            # Set DFT energy equal to MLP energy.
            if match_energies:
                atoms_list[jj].calc.results["energy"] = energy_dict_MLP[jj]
            # Remove index of converged calculations.
            if max_force < fmax_DFT:
                indices_active.remove(jj)
                atoms_list[jj].info["status"] = "finished"
                atoms_list[jj].info["DFT_steps"] = ii + 1
            else:
                # Add to training set.
                atoms_train.append(copy_atoms_and_results(atoms=atoms_list[jj]))
                # Calculate DFT forces.
                forces_dict_DFT[jj] = (
                    atoms_list[jj].get_forces(apply_constraint=False).copy()
                )
        time_DFT += timeit.default_timer() - time_DFT_start
        # Stop if all calculations have converged.
        if len(indices_active) == 0:
            break
        # Stop if the maximum number of steps is reached.
        if ii == max_steps_actlearn:
            for jj in indices_active:
                atoms_list[jj].info["status"] = "unfinished"
                atoms_list[jj].info["DFT_steps"] = ii + 1
            break
        # Run MLP model fine-tuning.
        print_title("MLP fine-tuning.")
        time_train_start = timeit.default_timer()
        calc_MLP = finetune_fun(
            atoms_list=atoms_train,
            calc=calc_MLP if train_finetuned else calc_MLP_copy,
            label=f"model_{ii + 1:02d}",
            directory=directory,
            **finetune_kwargs,
        )
        time_train += timeit.default_timer() - time_train_start
        # Print fine-tuned MLP errors.
        deltaforces = np.empty((0, 3))
        for jj in indices_active:
            atoms_list[jj].calc = calc_MLP
            forces_MLP = atoms_list[jj].get_forces(apply_constraint=False)
            deltaforces = np.vstack((deltaforces, forces_MLP - forces_dict_DFT[jj]))
        mae_forces = np.linalg.norm(deltaforces, axis=1).mean()
        emax_forces = np.linalg.norm(deltaforces, axis=1).max()
        print(f"MAE forces  = {mae_forces:+7.4f} [eV/Å]")
        print(f"Emax forces = {emax_forces:+7.4f} [eV/Å]")
    # Restore original DFT parameters.
    if update_parameters_DFT:
        calc_DFT.parameters = parameters_DFT
    # Calculation finished.
    print_title(f"Active-learning {calculation} calculation finished.")
    for atoms in atoms_list:
        atoms.info.update({
            "time_MLP": time_MLP / len(atoms_list), 
            "time_DFT": time_DFT / len(atoms_list),
            "time_train": time_train / len(atoms_list),
        })
    print("Average results:")
    average_steps = np.average([atoms.info["DFT_steps"] for atoms in atoms_list])
    print(f"DFT_steps  = {average_steps:6.1f}")
    print(f"time_MLP   = {time_MLP / len(atoms_list):6.1f} [s]")
    print(f"time_DFT   = {time_DFT / len(atoms_list):6.1f} [s]")
    print(f"time_train = {time_train / len(atoms_list):6.1f} [s]")
    # Return atoms.
    return atoms_list

# -------------------------------------------------------------------------------------
# RUN ACTIVE LEARNING FROM NAMES
# -------------------------------------------------------------------------------------

def run_active_learning_from_names(
    name: str,
    db_inp_name: str,
    db_out_name: str,
    calculation: str,
    calc_MLP_name: str,
    calc_DFT_name: str,
    calc_MLP_kwargs: dict = {},
    calc_DFT_kwargs: dict = {},
    run_MLP_kwargs: dict = {},
    run_DFT_kwargs: dict = {},
    finetune_kwargs: dict = {},
    directory: str = "finetuning",
    max_steps_actlearn: int = 100,
    train_on_all: bool = False,
    train_finetuned: bool = False,
    match_energies: bool = True,
    update_parameters_DFT: bool = False,
    save_trajs: bool = False,
    clean_directory: bool = True,
    db_inp_kwargs: dict = {},
    db_out_kwargs: dict = {},
    basedir: str = "",
):
    """
    Run active learning calculation from name and input database.
    """
    # Get atoms from database.
    db_inp = connect(name=os.path.join(basedir, db_inp_name))
    atoms = get_atoms_from_db(db_ase=db_inp, name=name, **db_inp_kwargs)
    # Get MLP calculator.
    calc_MLP = get_calculator(calc_name=calc_MLP_name, **calc_MLP_kwargs)
    # Get DFT calculator.
    calc_DFT = get_calculator(calc_name=calc_DFT_name, **calc_DFT_kwargs)
    # Get fine-tuning function.
    finetune_fun = get_finetune_function(calc_name=calc_MLP_name)
    # Run active learning.
    atoms = run_active_learning(
        atoms=atoms,
        calculation=calculation,
        calc_MLP=calc_MLP,
        calc_DFT=calc_DFT,
        finetune_fun=finetune_fun,
        finetune_kwargs=finetune_kwargs,
        directory=directory,
        max_steps_actlearn=max_steps_actlearn,
        run_MLP_kwargs=run_MLP_kwargs,
        run_DFT_kwargs=run_DFT_kwargs,
        train_on_all=train_on_all,
        train_finetuned=train_finetuned,
        match_energies=match_energies,
        update_parameters_DFT=update_parameters_DFT,
        save_trajs=save_trajs,
        clean_directory=clean_directory,
    )
    # Prepare results database.
    db_out = connect(name=os.path.join(basedir, db_out_name))
    # Write results to db.
    write_atoms_to_db(atoms=atoms, db_ase=db_out, name=name, **db_out_kwargs)

# -------------------------------------------------------------------------------------
# RUN ACTIVE LEARNING LIST FROM NAMES
# -------------------------------------------------------------------------------------

def run_active_learning_list_from_names(
    name_list: list,
    db_inp_name: str,
    db_out_name: str,
    calculation: str,
    calc_MLP_name: str,
    calc_DFT_name: str,
    calc_MLP_kwargs: dict = {},
    calc_DFT_kwargs: dict = {},
    run_MLP_kwargs: dict = {},
    run_DFT_kwargs: dict = {},
    finetune_kwargs: dict = {},
    directory: str = "finetuning",
    max_steps_actlearn: int = 100,
    train_on_all: bool = False,
    train_finetuned: bool = False,
    match_energies: bool = True,
    update_parameters_DFT: bool = False,
    save_trajs: bool = False,
    clean_directory: bool = True,
    db_inp_kwargs: dict = {},
    db_out_kwargs: dict = {},
    basedir: str = "",
):
    """
    Run active learning calculation from name and input database.
    """
    # Get atoms from database.
    db_inp = connect(name=os.path.join(basedir, db_inp_name))
    atoms_list = [
        get_atoms_from_db(db_ase=db_inp, name=name, **db_inp_kwargs)
        for name in name_list
    ]
    # Get MLP calculator.
    calc_MLP = get_calculator(calc_name=calc_MLP_name, **calc_MLP_kwargs)
    # Get DFT calculator.
    calc_DFT = get_calculator(calc_name=calc_DFT_name, **calc_DFT_kwargs)
    # Get fine-tuning function.
    finetune_fun = get_finetune_function(calc_name=calc_MLP_name)
    # Run active learning.
    atoms_list = run_active_learning_list(
        atoms_list=atoms_list,
        calculation=calculation,
        calc_MLP=calc_MLP,
        calc_DFT=calc_DFT,
        finetune_fun=finetune_fun,
        finetune_kwargs=finetune_kwargs,
        directory=directory,
        max_steps_actlearn=max_steps_actlearn,
        run_MLP_kwargs=run_MLP_kwargs,
        run_DFT_kwargs=run_DFT_kwargs,
        train_on_all=train_on_all,
        train_finetuned=train_finetuned,
        match_energies=match_energies,
        update_parameters_DFT=update_parameters_DFT,
        save_trajs=save_trajs,
        clean_directory=clean_directory,
    )
    # Prepare results database.
    db_out = connect(name=os.path.join(basedir, db_out_name))
    # Write results to db.
    for name, atoms in zip(name_list, atoms_list):
        write_atoms_to_db(atoms=atoms, db_ase=db_out, name=name, **db_out_kwargs)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
