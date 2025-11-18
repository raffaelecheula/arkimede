# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import timeit
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.io import Trajectory
from ase.db import connect

from arkimede.utilities import print_title, filter_results
from arkimede.databases import write_atoms_list_to_db
from arkimede.workflow.calculations import run_calculation

# -------------------------------------------------------------------------------------
# RUN ACTIVE LEARNING
# -------------------------------------------------------------------------------------

def run_active_learning(
    atoms: Atoms,
    calculation: str,
    calc_MLP: Calculator,
    calc_DFT: Calculator,
    finetuning_fun: callable,
    directory: str = "finetuning",
    max_steps_actlearn: int = 100,
    kwargs_MLP: dict = {},
    kwargs_DFT: dict = {},
    kwargs_finetuning = {},
    save_trajs: bool = True,
    save_metadata_npz: bool = True,
):
    """
    Run active learning calculation.
    """
    # Get DFT fmax.
    fmax_DFT = kwargs_DFT.pop("fmax", 0.05)
    # Prepare fine-tuning directory.
    os.makedirs(directory, exist_ok=True)
    # Trajectory.
    if save_trajs is True:
        trajectory = Trajectory(filename=f"{calculation}.traj", mode="w")
    # Run the active learning protocol.
    time_MLP = 0.
    time_DFT = 0.
    time_tun = 0.
    for ii in range(max_steps_actlearn):
        # Run MLP calculation.
        print_title(f"MLP {calculation} calculation.")
        time_MLP_start = timeit.default_timer()
        atoms.info["task"] = "MLP"
        run_calculation(
            atoms=atoms,
            calculation=calculation,
            calc=calc_MLP,
            trajectory=trajectory,
            **kwargs_MLP,
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
            **kwargs_DFT,
        )
        time_DFT += timeit.default_timer() - time_DFT_start
        # Check DFT max force.
        max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
        print(f"DFT fmax    = {max_force:+7.4f} [eV/Å]")
        if max_force < fmax_DFT:
            break
        # Set DFT energy equal to MLP energy.
        atoms.calc.results["energy"] = energy_MLP
        # Write structure with DFT forces to trajectory.
        if save_trajs is True:
            trajectory.write(atoms=atoms, **filter_results(atoms.calc.results))
        # Write structure with DFT forces to db.
        db_ase = connect(name=f"{directory}/dft.db", append=False)
        atoms_list = [atoms] * 1
        natoms = [len(atoms)] * 1
        write_atoms_list_to_db(
            atoms_list=atoms_list,
            db_ase=db_ase,
            fill_stress=True,
            no_constraints=True,
            use_tqdm=False,
        )
        if save_metadata_npz is True:
            np.savez_compressed(file=f"{directory}/metadata.npz", natoms=natoms)
        # Run MLP model fine-tuning.
        print_title("MLP fine-tuning.")
        time_tun_start = timeit.default_timer()
        label = f"model_{ii+1:02d}"
        calc_MLP = finetuning_fun(
            calc=calc_MLP,
            label=label,
            directory=directory,
            **kwargs_finetuning,
        )
        time_tun += timeit.default_timer() - time_tun_start
        # Print fine-tuned MLP errors.
        atoms.calc = calc_MLP
        atoms.get_forces(apply_constraint=False)
        deltaforces = calc_MLP.results["forces"] - calc_DFT.results["forces"]
        mae_forces = np.linalg.norm(deltaforces, axis=1).mean()
        emax_forces = np.linalg.norm(deltaforces, axis=1).max()
        print(f"MAE forces  = {mae_forces:+7.4f} [eV/Å]")
        print(f"Emax forces = {emax_forces:+7.4f} [eV/Å]")
    # Calculation finished.
    print_title(f"Active-learning {calculation} calculation finished.")
    atoms.info.update({
        "DFT_steps": ii+1,
        "time_MLP": time_MLP, 
        "time_DFT": time_DFT,
        "time_tun": time_tun,
    })
    print("DFT_steps =", atoms.info["DFT_steps"])
    for key in ["time_MLP", "time_DFT", "time_tun"]:
        print(f"{key} = {atoms.info[key]:6.1f} [s]")
    # Write final trajectory.
    if save_trajs is True:
        with Trajectory(filename=f"{calculation}_final.traj", mode="w") as trajectory:
            trajectory.write(atoms=atoms, **filter_results(atoms.calc.results))
    # Return atoms.
    return atoms

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
