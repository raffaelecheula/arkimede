# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import uuid
import inspect
import numpy as np
from copy import deepcopy
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS
try: from ase.mep.neb import NEB, NEBOptimizer
except: from ase.neb import NEB, NEBOptimizer
from ase.calculators.singlepoint import SinglePointCalculator

from arkimede.utilities import filter_results, update_atoms_from_atoms_opt
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
    copy_atoms: bool = True,
    **kwargs: dict,
) -> None:
    """
    Run a single-point calculation.
    """
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy() if copy_atoms is True else atoms
    if calc is not None:
        atoms_opt.calc = calc
    # Remove contraints for MLP fine-tuning from DFT calculations.
    if no_constraints is True:
        atoms_opt.constraints = []
    # Run the calculation.
    try:
        atoms_opt.get_potential_energy()
        atoms_opt.info["status"] = "finished"
    except Exception as error:
        print(error)
        atoms_opt.info["status"] = "failed"
    # Store the number of calculator calls.
    if "counter" in dir(atoms_opt.calc):
        atoms_opt.info["counter"] = atoms_opt.calc.counter
    # Update the input atoms with the results of the calculation.
    if copy_atoms is True:
        update_atoms_from_atoms_opt(
            atoms=atoms,
            atoms_opt=atoms_opt,
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
    max_steps: int = 500,
    min_steps: int = None,
    logfile: str = "-",
    label: str = "relax",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    copy_atoms: bool = True,
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
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy() if copy_atoms is True else atoms
    if calc is not None:
        atoms_opt.calc = calc
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    elif isinstance(trajectory, TrajectoryWriter):
        trajectory.atoms = atoms_opt
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
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(atoms_opt.calc) and reset_counter is True:
        atoms_opt.calc.counter = 0
    # Run the optimization.
    atoms_opt.info["status"] = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Store the number of calculator calls.
    if "counter" in dir(atoms_opt.calc):
        atoms_opt.info["counter"] = atoms_opt.calc.counter
    # Update the input atoms with the results of the calculation.
    if copy_atoms is True:
        update_atoms_from_atoms_opt(
            atoms=atoms,
            atoms_opt=atoms_opt,
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
    max_steps: int = 500,
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
    opt = optimizer(neb, logfile=logfile, trajectory=trajectory, **opt_kwargs)
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
            converged = opt.run(fmax=fmax, steps=max_steps - opt.nsteps)
        except TSConverged as stop:
            converged = True
    # Substitute calculator of intermediate images.
    for ii in range(1, len(images) - 1):
        images[ii].get_potential_energy()
        results = filter_results(results=images[ii].calc.results)
        images[ii].calc = SinglePointCalculator(atoms=images[ii], **results)
    # Write trajectory.
    if save_trajs is True and max_steps == 0:
        with TrajectoryWriter(filename=trajectory, mode="w") as traj:
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
    copy_atoms: bool = True,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    bonds_TS: list = None,
    mode_TS: np.ndarray = None,
    control_kwargs: dict = {},
    store_mode_TS: bool = True,
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
    from arkimede.utilities import get_indices_not_fixed
    from arkimede.workflow.observers import reset_eigenmode_obs
    # Get TS bonds from info dictionary.
    if bonds_TS is None:
        bonds_TS = atoms.info.get("bonds_TS", None)
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    # Set up the dimer calculation.
    if calc is not None:
        atoms.calc = calc
    mask = get_indices_not_fixed(atoms=atoms, return_mask=True)
    if mode_TS is None:
        mode_TS = atoms.info["mode_TS"].copy()
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
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(atoms_opt.calc) and reset_counter is True:
        atoms_opt.calc.counter = 0
    # Run the optimization.
    atoms_opt.info["status"] = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Store the number of calculator calls.
    if "counter" in dir(atoms_opt.calc):
        atoms_opt.info["counter"] = atoms_opt.calc.counter
    # Update the input atoms with the results of the calculation.
    if copy_atoms is True:
        update_atoms_from_atoms_opt(
            atoms=atoms,
            atoms_opt=atoms_opt,
            properties=properties,
            update_cell=update_cell,
        )
    if store_mode_TS is True:
        atoms.info["mode_TS"] = atoms_opt.eigenmodes[0]
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
    copy_atoms: bool = True,
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
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy() if copy_atoms is True else atoms
    if calc is not None:
        atoms_opt.calc = calc
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    elif isinstance(trajectory, TrajectoryWriter):
        trajectory.atoms = atoms_opt
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
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(atoms_opt.calc) and reset_counter is True:
        atoms_opt.calc.counter = 0
    # Run the optimization.
    atoms_opt.info["status"] = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Store the number of calculator calls.
    if "counter" in dir(atoms_opt.calc):
        atoms_opt.info["counter"] = atoms_opt.calc.counter
    # Update the input atoms with the results of the calculation.
    if copy_atoms is True:
        update_atoms_from_atoms_opt(
            atoms=atoms,
            atoms_opt=atoms_opt,
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
    copy_atoms: bool = True,
    update_cell: bool = False,
    properties: list = ["energy", "forces"],
    opt_kwargs: dict = {},
    bonds_TS: list = None,
    modify_hessian: bool = False,
    dot_prod_thr: float = 0.50,
    find_internals: bool = False,
    internal_kwargs: dict = {},
    store_hessian: bool = False,
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
    from arkimede.workflow.sella import modify_hessian_obs, get_internals
    # Get TS bonds from info dictionary.
    if bonds_TS is None:
        bonds_TS = atoms.info.get("bonds_TS", None)
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy() if copy_atoms is True else atoms
    if calc is not None:
        atoms_opt.calc = calc
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    elif isinstance(trajectory, TrajectoryWriter):
        trajectory.atoms = atoms_opt
    # Default parameters.
    opt_kwargs = {
        "eta": 0.01,
        "gamma": 0.05,
        "delta0": 0.05,
        "nsteps_per_diag": 1,
        **opt_kwargs,
    }
    # Find key internals.
    if find_internals is True:
        opt_kwargs["internal"] = get_internals(atoms=atoms_opt, **internal_kwargs)
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
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot is not None:
        obs_kwargs = {"atoms": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(atoms_opt.calc) and reset_counter is True:
        atoms_opt.calc.counter = 0
    # Run the optimization.
    atoms_opt.info["status"] = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
    )
    # Store the number of calculator calls.
    if "counter" in dir(atoms_opt.calc):
        atoms_opt.info["counter"] = atoms_opt.calc.counter
    # Update the input atoms with the results of the calculation.
    if copy_atoms is True:
        update_atoms_from_atoms_opt(
            atoms=atoms,
            atoms_opt=atoms_opt,
            properties=properties,
            update_cell=update_cell,
        )
    # Store Hessian.
    if store_hessian is True and opt.pes.H.B is not None:
        atoms.info["hessian"] = opt.pes.H.B.copy()
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN IRC CALCULATION
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
    copy_atoms: bool = True,
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
    # Prepare a copy of atoms.
    atoms_opt = atoms.copy() if copy_atoms is True else atoms
    if calc is not None:
        atoms_opt.calc = calc
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    elif isinstance(trajectory, TrajectoryWriter):
        trajectory.atoms = atoms_opt
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
        obs_kwargs = {"opt": opt, "max_forcecalls": max_forcecalls}
        opt.attach(max_forcecalls_obs, interval=1, **obs_kwargs)
    # Reset the number of calculator calls.
    if "counter" in dir(atoms_opt.calc) and reset_counter is True:
        atoms_opt.calc.counter = 0
    # Run the optimization.
    atoms_opt.info["status"] = run_optimization(
        opt=opt,
        fmax=fmax,
        max_steps=max_steps,
        min_steps=min_steps,
        run_kwargs={"direction": direction},
    )
    # Store the number of calculator calls.
    if "counter" in dir(atoms_opt.calc):
        atoms_opt.info["counter"] = atoms_opt.calc.counter
    # Update the input atoms with the results of the calculation.
    if copy_atoms is True:
        update_atoms_from_atoms_opt(
            atoms=atoms,
            atoms_opt=atoms_opt,
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
    label: str = "vibrations",
    vib_cache: str = "vib-uuid",
    directory: str = ".",
    trajectory: str = None,
    save_trajs: bool = False,
    indices_vib: str = "adsorbate",
    delta: float = 0.01,
    nfree: int = 2,
    remove_cache: bool = True,
    store_energies: bool = True,
    store_modes: bool = False,
    store_hessian: bool = False,
    unit_masses: bool = False,
    **kwargs: dict,
) -> list:
    """
    Run a vibrations calculation.
    """
    from arkimede.utilities import get_indices_from_name
    from arkimede.workflow.vibrations import Vibrations
    # Get indices of atoms to vibrate.
    indices_vib = get_indices_from_name(atoms=atoms, indices=indices_vib)
    # Cache directory with uuid to avoid conflicts.
    if vib_cache == "vib-uuid":
        vib_cache = f"vib-{uuid.uuid4().hex}"
    # Create directory to store the results.
    if save_trajs is True:
        os.makedirs(directory, exist_ok=True)
    # Initialize the vibrations calculation.
    atoms_vib = atoms.copy()
    if calc is not None:
        atoms_vib.calc = calc
    # Name of the trajectory file.
    if trajectory is None:
        trajectory = os.path.join(directory, f"{label}.traj") if save_trajs else None
    elif isinstance(trajectory, TrajectoryWriter):
        trajectory.atoms = atoms_vib
    # Set unit masses for optimization methods.
    if unit_masses is True:
        atoms_vib.set_masses([1.0] * len(atoms_vib))
    # Prepare vibrations calculation.
    vib = Vibrations(
        atoms=atoms_vib,
        indices=indices_vib,
        name=vib_cache,
        delta=delta,
        nfree=nfree,
        trajectory=trajectory,
    )
    # Run the calculation and get vibrational energies and modes.
    vib.clean()
    vib.run()
    data = vib.get_vibrations()
    vib_hessian = vib.H.copy()
    vib_energies, vib_modes = data.get_energies_and_modes(all_atoms=True)
    # Remove the cache directory.
    if remove_cache is True:
        vib.remove_cache()
    # Store vibrational energies and modes in atoms info dictionary.
    if store_energies is True:
        atoms.info["vib_energies"] = vib_energies
    if store_modes is True:
        atoms.info["vib_modes"] = vib_modes
    if store_hessian is True:
        atoms.info["vib_hessian"] = vib_hessian
    # Return vibrational Hessian, energies, and modes.
    return vib_hessian, vib_energies, vib_modes

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
