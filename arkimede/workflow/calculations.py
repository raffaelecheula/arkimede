# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import uuid
import inspect
import numpy as np
from copy import deepcopy
from ase import Atoms
from ase.io import Trajectory
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS
try: from ase.mep.neb import NEB, NEBOptimizer
except: from ase.neb import NEB, NEBOptimizer
from ase.calculators.singlepoint import SinglePointCalculator

from arkimede.utilities import (
    get_indices_not_fixed,
    get_indices_adsorbate,
    filter_results,
    update_atoms_from_atoms_opt,
)
from arkimede.workflow.observers import (
    min_steps_obs,
    max_forcecalls_obs,
    max_displacement_obs,
    max_force_tot_obs,
    atoms_too_close_obs,
)

# -------------------------------------------------------------------------------------
# HAS CONVERGED
# -------------------------------------------------------------------------------------

def has_converged(
    opt: Optimizer,
) -> bool:
    """
    Check if a calculation has converged.
    """
    if "gradient" in inspect.signature(opt.converged).parameters:
        return opt.converged(gradient=opt.optimizable.get_gradient())
    else:
        return opt.converged()

# -------------------------------------------------------------------------------------
# RUN OPTIMIZATION
# -------------------------------------------------------------------------------------

def run_optimization(
    opt: Optimizer,
    fmax: float = 0.05,
    max_steps: int = 300,
    min_steps: int = None,
    run_kwargs: dict = {},
):
    """
    Run optimization.
    """
    try:
        fmax_start = fmax if min_steps is None else 0.
        opt.run(fmax=fmax_start, steps=max_steps, **run_kwargs)
        status = "finished" if has_converged(opt=opt) else "unfinished"
    except Exception as error:
        print(error)
        status = "failed"
    # Return status.
    return status

# -------------------------------------------------------------------------------------
# RUN SINGLE-POINT CALCULATION
# -------------------------------------------------------------------------------------

def run_singlepoint_calculation(
    atoms: Atoms,
    calc: Calculator,
    properties: list = ["energy", "forces"],
    no_constraints: bool = False,
    **kwargs: dict,
) -> None:
    """
    Run a single-point calculation.
    """
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    # Remove contraints for MLP fine-tuning from DFT calculations.
    if no_constraints is True:
        atoms_opt.constraints = []
    # Run the calculation.
    try:
        atoms_opt.get_potential_energy()
        status = "finished"
    except Exception as error:
        print(error)
        status = "failed"
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=False,
    )

# -------------------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -------------------------------------------------------------------------------------

def run_relax_calculation(
    atoms: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 300,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "relax",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    optimizer: Optimizer = BFGS,
    opt_kwargs: dict = {},
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a relax calculation.
    """
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    # Set up the optimizer.
    opt = optimizer(
        atoms=atoms_opt,
        logfile=logfile,
        trajectory=trajectory,
        **opt_kwargs,
    )
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the optimization.
    status = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=update_cell,
    )
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN NEB CALCULATION
# -------------------------------------------------------------------------------------

def run_neb_calculation(
    images: list,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 300,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "neb",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    optimizer: Optimizer = BFGS,
    opt_kwargs: dict = {},
    k_neb: float = 1.00,
    climb: bool = False,
    parallel: bool = False,
    allow_shared_calculator: bool = True,
    neb_kwargs: dict = {},
    activate_climb: bool = True,
    print_energies: bool = False,
    converged_TS: bool = False,
    ftres_climb: float = 0.10,
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a NEB calculation.
    """
    from arkimede.workflow.neb import (
        activate_climb_obs,
        print_energies_obs,
        converged_TS_obs,
        TSConverged,
    )
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Set up NEB method.
    neb = NEB(
        images=images,
        k=k_neb,
        climb=climb,
        parallel=parallel,
        allow_shared_calculator=allow_shared_calculator,
        **neb_kwargs,
    )
    # Set calculator to images.
    for ii in range(len(images)):
        images[ii].calc = calc if allow_shared_calculator else deepcopy(calc)
    # Substitute calculator of initial and final images.
    for ii in (0, -1):
        images[ii].get_potential_energy()
        results = filter_results(results=images[ii].calc.results)
        images[ii].calc = SinglePointCalculator(atoms=images[ii], **results)
    # Set up the optimizer.
    opt = optimizer(neb, logfile=logfile, **opt_kwargs)
    # Observer for activating the climbing image.
    if activate_climb is True:
        obs_kwargs = {"neb": neb, "ftres": ftres_climb}
        opt.attach(activate_climb_obs, interval=1, **obs_kwargs)
    # Observer to print the energies of the path images.
    if print_energies is True:
        obs_kwargs = {"neb": neb, "opt": opt, "print_path_energies": True}
        opt.attach(print_energies_obs, interval=1, **obs_kwargs)
    # Observer that stops the calculation if the TS forces are below the threshold.
    if converged_TS is True:
        obs_kwargs = {"neb": neb, "opt": opt, "fmax": fmax}
        opt.attach(converged_TS_obs, interval=1, **obs_kwargs)
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the calculation (and restart NEBOptimizer when it fails randomly).
    converged = False
    fmax_start = fmax if min_steps is None else 0.
    while opt.nsteps < max_steps and not converged:
        try:
            converged = opt.run(fmax=fmax, steps=max_steps-opt.nsteps)
        except TSConverged as stop:
            converged = True
    # Write trajectory.
    if save_trajs is True:
        with Trajectory(filename=trajectory, mode="w") as traj:
            for atoms in images:
                traj.write(atoms, **atoms.calc.results)
    # Update images with the results.
    for atoms in images:
        atoms.info["status"] = "finished" if converged else "unfinished"
        if "counter" in dir(calc):
            atoms.info["counter"] = calc.counter
    # Write images.
    if write_images is True:
        index_TS = np.argmax([atoms.get_potential_energy() for atoms in images])
        atoms = images[index_TS]
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN DIMER CALCULATION
# -------------------------------------------------------------------------------------

def run_dimer_calculation(
    atoms: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "dimer",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    bonds_TS: list = None,
    mode_TS: np.ndarray = None,
    control_kwargs: dict = {},
    max_displ: float = None,
    max_force_tot: float = 100.,
    reset_eigenmode: bool = False,
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a Dimer TS-search calculation.
    """
    try: from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
    except: from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
    from arkimede.workflow.observers import reset_eigenmode_obs
    # Get TS bonds from info dictionary.
    if bonds_TS is None:
        bonds_TS = atoms.info["bonds_TS"]
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Set up the dimer calculation.
    atoms.calc = calc
    mask = get_indices_not_fixed(atoms, return_mask=True)
    if mode_TS is None:
        mode_TS = atoms.info.get("mode_TS").copy()
    # Dimer control parameters.
    control_kwargs = {
        "initial_eigenmode_method": "displacement",
        "displacement_method": "vector",
        "logfile": None,
        "trial_trans_step": 0.06,
        "dimer_separation": 0.02,
        **control_kwargs,
    }
    # Set up the dimer atoms.
    dimer_control = DimerControl(**control_kwargs)
    atoms_opt = MinModeAtoms(
        atoms=atoms,
        control=dimer_control,
        mask=mask,
    )
    mode_TS *= control_kwargs.get("maximum_translation", 0.1) / np.linalg.norm(mode_TS)
    atoms_opt.displace(displacement_vector=mode_TS, mask=mask)
    # Set up the dimer optimizer.
    opt = MinModeTranslate(
        dimeratoms=atoms_opt,
        trajectory=trajectory,
        logfile=logfile,
    )
    # Observer that reset the eigenmode to the direction of the TS bonds.
    if bonds_TS and reset_eigenmode is True:
        obs_kwargs = {"atoms": atoms_opt, "bonds_TS": bonds_TS}
        opt.attach(reset_eigenmode_obs, interval=1, **obs_kwargs)
    # Observer that checks the displacement of the TS from the starting position.
    if max_displ is not None:
        obs_kwargs = {"atoms": atoms_opt, "atoms_zero": atoms, "max_displ": max_displ}
        opt.attach(function=max_displacement_obs, interval=10, **obs_kwargs)
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot is not None:
        obs_kwargs = {"atoms": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **obs_kwargs)
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the optimization.
    status = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=update_cell,
    )
    atoms.info["mode_TS"] = atoms_opt.eigenmodes[0]
    if "counter" in dir(calc):
        atoms.info["counter"] = calc.counter
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN CLIMB TS BONDS CALCULATION
# -------------------------------------------------------------------------------------

def run_climbtsbonds_calculation(
    atoms: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "climbtsbonds",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    optimizer: Optimizer = BFGS,
    opt_kwargs: dict = {"maxstep": 0.01},
    bonds_TS: list = None,
    max_displ: float = None,
    max_force_tot: float = None,
    atoms_too_close: bool = False,
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a ClimbTSBonds TS-search calculation.
    """
    from arkimede.optimize.climbtsbonds import ClimbTSBonds
    # Get TS bonds from info dictionary.
    if bonds_TS is None:
        bonds_TS = atoms.info["bonds_TS"]
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    atoms_opt.constraints.append(ClimbTSBonds(bonds=bonds_TS))
    # Set up the optimizer.
    opt = optimizer(
        atoms=atoms_opt,
        logfile=logfile,
        trajectory=trajectory,
        **opt_kwargs,
    )
    # Observer that checks the displacement of the TS from the starting position.
    if max_displ is not None:
        obs_kwargs = {"atoms": atoms_opt, "atoms_zero": atoms, "max_displ": max_displ}
        opt.attach(function=max_displacement_obs, interval=10, **obs_kwargs)
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot is not None:
        obs_kwargs = {"atoms": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **obs_kwargs)
    # Observer to stop the calculation if two atoms are too close to each other.
    if atoms_too_close is True:
        obs_kwargs = {"atoms": atoms_opt}
        opt.attach(function=atoms_too_close_obs, interval=10, **obs_kwargs)
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the calculation.
    failed = 0
    converged = False
    fmax_start = fmax if min_steps is None else 0.
    while opt.nsteps < max_steps and failed < 100 and not converged:
        try:
            converged = opt.run(fmax_start, steps=max_steps-opt.nsteps)
            status = "finished" if has_converged(opt=opt) else "unfinished"
        except Exception as error:
            print(error)
            status = "failed"
            failed += 1
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=update_cell,
    )
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN CLIMBFIXINT CALCULATION
# -------------------------------------------------------------------------------------

def run_climbfixint_calculation(
    atoms: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "climbfixint",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    bonds_TS: list = None,
    mic: bool = True,
    epsilon: float = 1e-7,
    opt_kwargs: dict = {},
    optB_kwargs: dict = {},
    optB_max_steps: int = 200,
    max_displ: float = None,
    max_force_tot: float = 100.,
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a ClimbFixInternals TS-search calculation.
    """
    from ase.constraints import FixInternals
    from arkimede.workflow.climbfixinternals import BFGSClimbFixInternals
    # Get TS bonds from info dictionary.
    if bonds_TS is None:
        bonds_TS = atoms.info["bonds_TS"]
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    # Set up the ClimbFixInternal calculation.
    climb_coordinate = [bond[:2] + [1.0] for bond in bonds_TS]
    bondcombos = [[None, climb_coordinate]]
    atoms_opt.constraints += [
        FixInternals(bondcombos=bondcombos, mic=mic, epsilon=epsilon)
    ]
    # Default parameters.
    opt_kwargs = {
        "maxstep": 0.05,
        "logfile": logfile,
        "trajectory": trajectory,
        **opt_kwargs,
    }
    optB_kwargs = {
        "maxstep": 0.05,
        "logfile": "-",
        "trajectory": trajectory,
        **optB_kwargs,
    }
    # Prepare ClimbFixInternal BFGS optimizer.
    opt = BFGSClimbFixInternals(
        atoms=atoms_opt,
        climb_coordinate=climb_coordinate,
        optB_kwargs=optB_kwargs,
        optB_max_steps=optB_max_steps,
        max_force_tot=max_force_tot,
        **opt_kwargs,
    )
    # Observer that checks the displacement of the TS from the starting position.
    if max_displ is not None:
        obs_kwargs = {"atoms": atoms_opt, "atoms_zero": atoms, "max_displ": max_displ}
        opt.attach(function=max_displacement_obs, interval=10, **obs_kwargs)
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the optimization.
    status = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=update_cell,
    )
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN SELLA CALCULATION
# -------------------------------------------------------------------------------------

def run_sella_calculation(
    atoms: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "sella",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    opt_kwargs: dict = {},
    bonds_TS: list = None,
    dot_prod_thr: float = 0.50,
    modify_hessian: bool = False,
    max_displ: float = None,
    max_force_tot: float = None,
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a Sella TS-search calculation.
    """
    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*TPU.*")
    warnings.filterwarnings("ignore", message=".*CUDA.*")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    from sella import Sella
    from arkimede.workflow.sella import modify_hessian_obs
    # Get TS bonds from info dictionary.
    if bonds_TS is None:
        bonds_TS = atoms.info["bonds_TS"]
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    # Default parameters.
    opt_kwargs = {
        "eta": 0.01,
        "gamma": 0.05,
        "delta0": 0.05,
        "nsteps_per_diag": 1,
        **opt_kwargs,
    }
    # Prepare Sella optimizer.
    opt = Sella(
        atoms=atoms_opt,
        trajectory=trajectory,
        logfile=logfile,
        **opt_kwargs,
    )
    # Observer that modifies the hessian adding curvature to TS bonds.
    if modify_hessian is True:
        obs_kwargs = {"opt": opt, "bonds_TS": bonds_TS, "dot_prod_thr": dot_prod_thr}
        opt.attach(modify_hessian_obs, interval=1, **obs_kwargs)
    # Observer that checks the displacement of the TS from the starting position.
    if max_displ is not None:
        obs_kwargs = {"atoms": atoms_opt, "atoms_zero": atoms, "max_displ": max_displ}
        opt.attach(function=max_displacement_obs, interval=10, **obs_kwargs)
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot is not None:
        obs_kwargs = {"atoms": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the optimization.
    status = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=update_cell,
    )
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN SELLA CALCULATION
# -------------------------------------------------------------------------------------

def run_irc_calculation(
    atoms: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "irc",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    direction: str = "forward",
    keep_going: bool = True,
    opt_kwargs: dict = {},
    max_forcecalls: int = None,
    reset_counter: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a Sella IRC calculation.
    """
    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*TPU.*")
    warnings.filterwarnings("ignore", message=".*CUDA.*")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    from sella import IRC
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    # Set up the IRC optimizer.
    opt = IRC(
        atoms=atoms_opt,
        trajectory=trajectory,
        logfile=logfile,
        keep_going=keep_going,
        **opt_kwargs,
    )
    # Observer to set a minimum number of steps.
    if min_steps is not None:
        obs_kwargs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **obs_kwargs)
    # Observer to set a maximum of forces calls.
    if max_forcecalls is not None:
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(calc) and reset_counter is True:
        calc.counter = 0
    # Run the optimization.
    status = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
        run_kwargs={"direction": direction},
    )
    # Update the input atoms with the results of the calculation.
    update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        status=status,
        properties=properties,
        update_cell=update_cell,
    )
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN VIBRATIONS CALCULATION
# -------------------------------------------------------------------------------------

def run_vibrations_calculation(
    atoms: Atoms,
    calc: Calculator,
    label: str = "vib-uuid",
    directory: str = ".",
    save_trajs: bool = False,
    properties: list = ["energy", "forces"],
    indices: str = "adsorbate",
    delta: float = 0.01,
    nfree: int = 2,
    remove_cache: bool = False,
    **kwargs: dict,
) -> None:
    """
    Run a vibrations calculation.
    """
    import shutil
    from arkimede.workflow.vibrations import Vibrations
    # Get indices of atoms to vibrate.
    if indices == "adsorbate":
        indices = get_indices_adsorbate(atoms=atoms)
    elif indices == "not_fixed":
        indices = get_indices_not_fixed(atoms=atoms)
    # Label with uuid to avoid conflicts.
    if label == "vib-uuid":
        label = f"vib-{uuid.uuid4().hex}"
    # Create directory to store the results.
    if save_trajs is True:
        os.makedirs(directory, exist_ok=True)
    # Initialize the vibrations calculation.
    atoms_vib = atoms.copy()
    atoms_vib.calc = calc
    # Prepare vibrations calculation.
    vib = Vibrations(
        atoms=atoms_vib,
        indices=indices,
        name=label,
        delta=delta,
        nfree=nfree,
    )
    # Run the calculation and get normal frequencies.
    vib.clean(empty_files=True)
    vib.run()
    vib_energies = vib.get_energies()
    # Get the atoms objects of the displacements.
    atoms_list = []
    for disp, atoms_vib in vib.iterdisplace(inplace=True):
        atoms_vib = atoms_vib.copy()
        results = filter_results(results=vib.cache[disp.name], properties=properties)
        atoms_vib.calc = SinglePointCalculator(atoms=atoms_vib, **results)
        atoms_vib.info["displacement"] = str(disp.name)
        atoms_list.append(atoms_vib)
    # Write Trajectory files for each displacement.
    if save_trajs is True:
        trajname = os.path.join(directory, f"{label}.traj")
        with Trajectory(filename=trajname, mode="w") as traj:
            for atoms_vib in atoms_list[1:] + atoms_list[0:1]:
                traj.write(atoms_vib, **atoms_vib.calc.results)
    # Remove the cache directory with the results of the calculation.
    if remove_cache is True:
        vib.clean()
        if os.path.isdir(label):
            shutil.rmtree(label)
    # Store vibrational energies in atoms.
    atoms.info["vib_energies"] = vib_energies

# -------------------------------------------------------------------------------------
# RUN CALCULATION
# -------------------------------------------------------------------------------------

def run_calculation(
    atoms: Atoms,
    calc: Calculator,
    calculation: str,
    **kwargs: dict,
) -> None:
    """
    Run calculation.
    """
    if calculation == "singlepoint":
        # Single-point calculation.
        run_singlepoint_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "relax":
        # Relax calculation.
        run_relax_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "neb":
        # NEB calculation.
        run_neb_calculation(
            images=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "dimer":
        # Dimer calculation.
        run_dimer_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "climbtsbonds":
        # ClimbTSbonds calculaton.
        run_climbtsbonds_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "climbfixint":
        # ClimbFixInternals calculaton.
        run_climbfixint_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "sella":
        # Sella calculaton.
        run_sella_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "sella-ba":
        # Sella calculaton.
        run_sella_calculation(
            atoms=atoms,
            calc=calc,
            modify_hessian=True,
            **kwargs,
        )
    elif calculation == "irc":
        # IRC calculaton.
        run_irc_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )
    elif calculation == "vibrations":
        # Vibrations calculaton.
        run_vibrations_calculation(
            atoms=atoms,
            calc=calc,
            **kwargs,
        )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
