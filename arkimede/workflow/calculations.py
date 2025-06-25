# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.io import Trajectory
from ase.optimize import BFGS, LBFGS, ODE12r
try: from ase.neb import NEB, NEBOptimizer
except: from ase.mep.neb import NEB, NEBOptimizer
from ase.calculators.singlepoint import SinglePointCalculator
from arkimede.utilities import (
    get_indices_fixed,
    get_indices_not_fixed,
    get_indices_adsorbate,
    filter_results,
    update_atoms_from_atoms_opt,
)
from arkimede.optimize.observers import (
    min_steps_obs,
    max_forcecalls_obs,
    activate_climb_obs,
    print_energies_obs,
    reset_eigenmode_obs,
    max_displacement_obs,
    max_force_tot_obs,
    atoms_too_close_obs,
)

# -------------------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -------------------------------------------------------------------------------------

def run_relax_calculation(
    atoms,
    calc,
    fmax=0.05,
    max_steps=300,
    min_steps=None,
    logfile='-',
    label='relax',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    optimizer=BFGS,
    max_forcecalls=None,
    kwargs_opt={},
    properties=["energy", "forces"],
    **kwargs,
):
    """Run a relax calculation."""

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')

    # Set up the optimizer.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    opt = optimizer(
        atoms=atoms_opt,
        logfile=logfile,
        trajectory=trajname,
        **kwargs_opt,
    )
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)
    
    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=max_steps)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=converged,
        modified=False,
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
    images,
    calc,
    fmax=0.05,
    max_steps=300,
    min_steps=None,
    k_neb=1.00,
    label='neb',
    directory='.',
    save_trajs=False,
    write_images=False,
    logfile='-',
    optimizer=NEBOptimizer,
    kwargs_opt={},
    activate_climb=True,
    print_energies=False,
    ftres_climb=0.10,
    max_forcecalls=None,
    **kwargs,
):
    """Run a NEB calculation."""
    
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    
    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')
    
    # Set up NEB method.
    neb = NEB(
        images=images,
        k=k_neb,
        climb=False,
        parallel=False,
        method='aseneb',
        allow_shared_calculator=True,
    )
    for atoms in images:
        atoms.calc = calc
    for ii in (0, -1):
        images[ii].get_potential_energy()
        results = filter_results(results=images[ii].calc.results)
        images[ii].calc = SinglePointCalculator(atoms=images[ii], **results)
    
    # Set up the optimizer.
    opt = optimizer(neb, logfile=logfile, **kwargs_opt)
    
    # Observer for activating the climbing image.
    if activate_climb:
        kwargs_obs = {"neb": neb, "ftres": ftres_climb}
        opt.attach(activate_climb_obs, interval=1, **kwargs_obs)
    
    # Observer to print the energies of the path images.
    if print_energies:
        kwargs_obs = {"neb": neb, "opt": opt, "print_path_energies": True}
        opt.attach(print_energies_obs, interval=1, **kwargs_obs)
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)

    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)

    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0

    # Run the calculation. The while is to keep NEBOptimizer going when 
    # it fails randomly.
    converged = False
    while opt.nsteps < max_steps and not converged:
        converged = opt.run(fmax=fmax, steps=max_steps-opt.nsteps)
    
    # Write trajectory.
    if save_trajs is True:
        with Trajectory(filename=trajname, mode='w') as traj:
            for atoms in images:
                traj.write(atoms, **atoms.calc.results)
    
    # Update images with the results.
    for atoms in images:
        atoms.info["modified"] = False
        atoms.info["converged"] = bool(converged)
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
    atoms,
    calc,
    bonds_TS=None,
    vector=None,
    label='dimer',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.05,
    max_steps=500,
    min_steps=None,
    max_displacement=False,
    max_force_tot=50.,
    reset_eigenmode=False,
    max_forcecalls=None,
    sign_bond_dict={'break': +1, 'form': -1},
    properties=["energy", "forces"],
    kwargs_dimer={},
    **kwargs,
):
    """Run a dimer calculation."""
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
    from arkimede.utilities import get_vector_from_bonds_TS

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')

    if vector is None:
        vector = atoms.info.get("vector").copy()

    atoms.calc = calc
    mask = get_indices_not_fixed(atoms, return_mask=True)
    
    kwargs_dimer_opt = {
        "initial_eigenmode_method": 'displacement',
        "displacement_method": 'vector',
        "logfile": None,
        "cg_translation": True,
        "use_central_forces": True,
        "extrapolate_forces": False,
        "order": 1,
        "f_rot_min": 0.10,
        "f_rot_max": 1.00,
        "max_num_rot": 6, # modified
        "trial_angle": np.pi/6, # modified
        "trial_trans_step": 0.005, # modified
        "maximum_translation": 0.050, # modified
        "dimer_separation": 0.005, # modified
    }
    kwargs_dimer_opt.update(kwargs_dimer)
    
    dimer_control = DimerControl(**kwargs_dimer_opt)
    atoms_opt = MinModeAtoms(
        atoms=atoms,
        control=dimer_control,
        mask=mask,
    )
    vector *= kwargs_dimer_opt["maximum_translation"]/np.linalg.norm(vector)
    atoms_opt.displace(displacement_vector=vector, mask=mask)

    opt = MinModeTranslate(
        atoms=atoms_opt,
        trajectory=trajname,
        logfile=logfile,
    )
    
    # Observer that reset the eigenmode to the direction of the ts bonds.
    # Maybe there is a better way of doing this (e.g., selecting the 
    # eigenmode most similar to the directions of the ts bonds).
    if bonds_TS and reset_eigenmode:
        kwargs_obs = {
            "atoms_opt": atoms_opt,
            "bonds_TS": bonds_TS,
            "sign_bond_dict": sign_bond_dict,
        }
        opt.attach(reset_eigenmode_obs, interval=1, **kwargs_obs)
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        kwargs_obs = {
            "atoms_new": atoms_opt,
            "atoms_old": atoms,
            "max_displacement": max_displacement,
        }
        opt.attach(function=max_displacement_obs, interval=10, **kwargs_obs)
    
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot:
        kwargs_obs = {"atoms_opt": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **kwargs_obs)
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)
    
    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=max_steps)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=converged,
        modified=False,
        properties=properties,
        update_cell=update_cell,
    )
    atoms.info["vector"] = atoms_opt.eigenmodes[0]
    if "counter" in dir(calc):
        atoms.info["counter"] = calc.counter
    
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# RUN CLIMBBONDS CALCULATION
# -------------------------------------------------------------------------------------

def run_climbbonds_calculation(
    atoms,
    bonds_TS,
    calc,
    label='climbbonds',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.05,
    max_steps=500,
    min_steps=None,
    max_displacement=None,
    max_force_tot=50.,
    atoms_too_close=False,
    max_forcecalls=None,
    optimizer=ODE12r,
    atoms_IS_FS=None,
    kwargs_opt={},
    properties=["energy", "forces"],
    propagate_forces=True,
    only_bond_type="break",
    **kwargs,
):
    """Run a climbbonds calculation."""
    from arkimede.optimize.climbbonds import ClimbBonds
    
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    
    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')
    
    # Invert the forces only for a type of bond ("break" or "form").
    if only_bond_type is not None:
        bonds_TS_new = [bond for bond in bonds_TS if bond[2] == only_bond_type]
        if len(bonds_TS_new) > 0:
            bonds_TS = bonds_TS_new
    
    # Create a copy of the atoms object to not mess with constraints.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    atoms_opt.constraints.append(ClimbBonds(bonds=bonds_TS, atoms_IS_FS=atoms_IS_FS))
    
    # Add forces propagation to improve convergence.
    if propagate_forces is True:
        from arkimede.optimize.constraints import PropagateForces
        atoms_opt.constraints.append(PropagateForces())

    # Set up the optimizer.
    opt = optimizer(
        atoms=atoms_opt,
        logfile=logfile,
        trajectory=trajname,
        **kwargs_opt,
    )
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        kwargs_obs = {
            "atoms_new": atoms_opt,
            "atoms_old": atoms,
            "max_displacement": max_displacement,
        }
        opt.attach(function=max_displacement_obs, interval=10, **kwargs_obs)
    
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot:
        kwargs_obs = {"atoms_opt": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **kwargs_obs)
    
    ## Observer to stop the calculation if two atoms are too close to each other.
    if atoms_too_close:
        kwargs_obs = {"atoms_opt": atoms_opt}
        opt.attach(function=atoms_too_close_obs, interval=10, **kwargs_obs)
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)
    
    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    failed = 0
    converged = False
    while opt.nsteps < max_steps and failed < 100 and not converged:
        try:
            opt.run(fmax=fmax, steps=max_steps-opt.nsteps)
            converged = opt.converged()
        except Exception as error:
            print(error)
            failed += 1
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=converged,
        modified=False,
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
    atoms,
    bonds_TS,
    calc,
    label='climbfixint',
    index_constr2climb=0,
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.05,
    max_steps=500,
    min_steps=None,
    optB_kwargs={'logfile': '-', 'trajectory': None},
    max_displacement=None,
    max_force_tot=50.,
    max_forcecalls=None,
    properties=["energy", "forces"],
    kwargs_opt={},
    **kwargs,
):
    """Run a climbfixinternals calculation."""
    from arkimede.optimize.climbfixinternals import (
        FixInternals,
        BFGSClimbFixInternals,
    )
    
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    
    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')
    
    # Create a copy of the atoms object to not mess with constraints.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    
    bonds = [[None, bond[:2]] for bond in bonds_TS]
    atoms_opt.set_constraint([FixInternals(bonds=bonds)]+atoms.constraints)
    
    # Optimizer for ClimbFixInternal.
    opt = BFGSClimbFixInternals(
        atoms=atoms_opt,
        logfile=logfile,
        trajectory=trajname,
        index_constr2climb=index_constr2climb,
        optB_kwargs=optB_kwargs,
        **kwargs_opt,
    )
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        kwargs_obs = {
            "atoms_new": atoms_opt,
            "atoms_old": atoms,
            "max_displacement": max_displacement,
        }
        opt.attach(function=max_displacement_obs, interval=10, **kwargs_obs)
    
    # Observer to stop the calculation if the maximum force is too high.
    if max_force_tot:
        kwargs_obs = {"atoms_opt": atoms_opt, "max_force_tot": max_force_tot}
        opt.attach(function=max_force_tot_obs, interval=10, **kwargs_obs)
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)
    
    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=max_steps)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=converged,
        modified=False,
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
    atoms,
    calc,
    label='sella',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.05,
    max_steps=500,
    min_steps=None,
    max_forcecalls=None,
    properties=["energy", "forces"],
    kwargs_opt={},
    **kwargs,
):
    """Run a sella calculation."""
    from sella import Sella, Constraints

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')

    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    indices = get_indices_fixed(atoms_opt, return_mask=False)
    
    atoms_opt.constraints = []
    constraints = Constraints(atoms_opt)
    for ii in indices:
        constraints.fix_translation(ii)
    
    opt = Sella(
        atoms=atoms_opt,
        constraints=constraints,
        trajectory=trajname,
        logfile=logfile,
        **kwargs_opt,
    )
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)
    
    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=max_steps)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=converged,
        modified=False,
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
    atoms,
    calc,
    label='irc',
    direction="forward",
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.05,
    max_steps=500,
    min_steps=None,
    max_forcecalls=None,
    properties=["energy", "forces"],
    kwargs_opt={},
    **kwargs,
):
    """Run a sella calculation."""
    from sella import IRC, Constraints

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    trajname = None
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')

    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    indices = get_indices_fixed(atoms_opt, return_mask=False)
    
    atoms_opt.constraints = []
    constraints = Constraints(atoms_opt)
    for ii in indices:
        constraints.fix_translation(ii)
    
    opt = IRC(
        atoms=atoms_opt,
        constraints=constraints,
        trajectory=trajname,
        logfile=logfile,
        **kwargs_opt,
    )
    
    # Observer to set a minimum number of steps.
    if min_steps:
        kwargs_obs = {"opt": opt, "min_steps": min_steps, "fmax": fmax}
        opt.attach(min_steps_obs, interval=1, **kwargs_obs)
    
    # Observer to set a maximum of forces calls.
    if max_forcecalls:
        kwargs_obs = {"opt": opt, "max_forcecalls": max_forcecalls, "calc": calc}
        opt.attach(max_forcecalls_obs, interval=1, **kwargs_obs)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=max_steps, direction=direction)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=converged,
        modified=False,
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
    atoms,
    calc,
    indices="adsorbate",
    label="vibrations",
    delta=0.01,
    nfree=2,
    directory=".",
    remove_cache=False,
    save_trajs=False,
    properties=["energy", "forces"],
    **kwargs,
):
    """Run a vibrations calculation."""
    from ase.vibrations import Vibrations
    
    # Get indices of atoms to vibrate.
    if indices == "adsorbate":
        indices = get_indices_adsorbate(atoms)
    elif indices == "not_fixed":
        indices = get_indices_not_fixed(atoms)
    
    # Create directory to store the results.
    if save_trajs is True:
        os.makedirs(directory, exist_ok=True)

    # Initialize the vibrations calculation.
    atoms_vib = atoms.copy()
    atoms_vib.calc = calc
    
    class VibrationsWithEnergy(Vibrations):
        def calculate(self, atoms, disp):
            results = {}
            results['energy'] = float(self.calc.get_potential_energy(atoms))
            results['forces'] = self.calc.get_forces(atoms)
            if self.ir:
                results['dipole'] = self.calc.get_dipole_moment(atoms)
            return results
    
    vib = VibrationsWithEnergy(
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
        trajname = os.path.join(directory, f'{label}.traj')
        with Trajectory(filename=trajname, mode='w') as traj:
            for atoms_vib in atoms_list[1:]+atoms_list[0:1]:
                traj.write(atoms_vib, **atoms_vib.calc.results)
    
    # Remove the cache directory with the results of the calculation.
    if remove_cache is True:
        vib.clean()
        os.rmdir(label)

    atoms.info["vib_energies"] = vib_energies

# -------------------------------------------------------------------------------------
# RUN CALCULATION
# -------------------------------------------------------------------------------------

def run_calculation(
    atoms,
    calc,
    calculation,
    **kwargs,
):
    """Run calculation."""
    if calculation == "relax":
        # Relax calculation.
        run_relax_calculation(
            atoms=atoms,
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
    elif calculation == "climbbonds":
        # Climbbonds calculaton.
        run_climbbonds_calculation(
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
