# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import numpy as np
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from arkimede.workflow.utilities import get_atoms_not_fixed, get_atoms_not_surface

# -----------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -----------------------------------------------------------------------------

def run_relax_calculation(
    atoms,
    calc,
    fmax=0.01,
    steps_max=300,
    logfile='-',
    name='relax',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
):
    """Run a relax calculation."""
    from ase.optimize import BFGS

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    if save_trajs is True:
        trajectory = os.path.join(directory, f'{name}.traj')
    else:
        trajectory = None

    # Setup the BFGS solver.
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    opt = BFGS(
        atoms=atoms_copy,
        logfile=logfile,
        trajectory=trajectory,
    )
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=steps_max)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_copy.positions)
    if update_cell is True:
        atoms.set_cell(atoms_copy.cell)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        energy=atoms_copy.calc.results['energy'],
        forces=atoms_copy.calc.results['forces'],
    )
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# RUN NEB CALCULATION
# -----------------------------------------------------------------------------

def run_neb_calculation(
    images,
    calc,
    fmax=0.01,
    steps_max=300,
    k_neb=0.10,
    name='neb',
    directory='.',
    save_trajs=False,
    write_images=False,
    logfile='-',
    use_OCPdyNEB=False,
    use_NEBOptimizer=False,
    activate_climb=True,
    print_energies=True,
    ftres_climb=0.10,
):
    """Run a NEB calculation."""
    
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    
    # Name of the trajectory file.
    if save_trajs is True:
        trajectory = os.path.join(directory, f'{name}.traj')
    else:
        trajectory = None
    
    # Choose NEB method.
    if use_OCPdyNEB:
        from ocpneb.core import OCPdyNEB
        neb = OCPdyNEB(
            images=images,
            checkpoint_path=calc.config['checkpoint'],
            k=k_neb,
            climb=False,
            dynamic_relaxation=False,
            method="aseneb",
            cpu=False,
            batch_size=4,
        )
    else:
        from ase.neb import NEB
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
            images[ii].calc = SinglePointCalculator(
                atoms=images[ii],
                energy=images[ii].calc.results["energy"],
                forces=images[ii].calc.results["forces"],
            )
    
    # Choose optimizer.
    if use_NEBOptimizer:
        from ase.neb import NEBOptimizer
        opt = NEBOptimizer(neb, logfile=logfile)
    else:
        from ase.optimize import BFGS
        opt = BFGS(neb, logfile=logfile)
    
    def activate_climb_obs(neb = neb, ftres = ftres_climb):
        if neb.get_residual() < ftres:
            neb.climb = True
    
    def print_energies_obs(neb = neb, opt = opt, print_path_energies = True):
        fres = neb.get_residual()
        print(f'Step: {opt.nsteps:4d} Fmax: {fres:9.4f}', end = '  ')
        for ii in (0, -1):
            neb.energies[ii] = images[ii].calc.results['energy']
        act_energy = max(neb.energies)-neb.energies[0]
        print(f'Eact: {act_energy:+9.4f} eV', end = '  ')
        if print_path_energies is True:
            print(f'Epath:', end = '  ')
            for energy in neb.energies:
                print(f'{energy-neb.energies[0]:+9.4f}', end = ' ')
            print('eV')

    # Attach observers.
    if activate_climb:
        opt.attach(activate_climb_obs, interval = 1)
    if print_energies:
        opt.attach(print_energies_obs, interval = 1)

    # Run the calculation. The while is to keep NEBOptimizer going when 
    # it fails randomly.
    converged = False
    while opt.nsteps < steps_max and not converged:
        converged = opt.run(fmax=fmax, steps=steps_max-opt.nsteps)
    
    # Write trajectory.
    if save_trajs is True:
        traj = Trajectory(filename=trajectory, mode='w')
        for atoms in images:
            traj.write(atoms)
    
    # Update images with the results.
    for atoms in images:
        atoms.info["modified"] = False
        atoms.info["converged"] = bool(converged)

    # Write images.
    if write_images is True:
        index_TS = np.argmax([atoms.get_potential_energy() for atoms in images])
        atoms = images[index_TS]
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# RUN DIMER CALCULATION
# -----------------------------------------------------------------------------

def run_dimer_calculation(
    atoms,
    calc,
    bonds_TS=None,
    vector=None,
    name='dimer',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.01,
    steps_max=500,
    max_displacement=False,
    reset_eigenmode=True,
    sign_bond_dict={'break': +1, 'form': -1},
):
    """Run a dimer calculation."""
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
    from arkimede.workflow.utilities import get_vector_from_bonds_TS

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    if save_trajs is True:
        trajectory = os.path.join(directory, f'{name}.traj')
    else:
        trajectory = None

    if vector is None:
        vector = atoms.info.get("vector").copy()

    atoms.calc = calc
    mask = get_atoms_not_fixed(atoms, return_mask=True)
    
    dimer_control = DimerControl(
        initial_eigenmode_method='displacement',
        displacement_method='vector',
        logfile=None,
        cg_translation=True,
        use_central_forces=True,
        extrapolate_forces=False,
        order=1,
        f_rot_min=0.10,
        f_rot_max=1.00,
        max_num_rot=6,
        trial_angle=np.pi/6,
        trial_trans_step=0.005,
        maximum_translation=0.100,
        dimer_separation=0.005,
    )
    atoms_dimer = MinModeAtoms(
        atoms=atoms,
        control=dimer_control,
        mask=mask,
    )
    atoms_dimer.displace(displacement_vector=vector, mask=mask)

    opt = MinModeTranslate(
        atoms=atoms_dimer,
        trajectory=trajectory,
        logfile=logfile,
    )
    
    # Observer that reset the eigenmode to the direction of the ts bonds.
    # TODO: Maybe there is a better way of doind this (e.g., selecting the 
    # eigenmode most similar to the directions of the ts bonds).
    if bonds_TS and reset_eigenmode:
        def reset_eigenmode_obs(atoms_dimer = atoms_dimer):
            vector = get_vector_from_bonds_TS(
                atoms=atoms_dimer,
                bonds_TS=bonds_TS,
                sign_bond_dict=sign_bond_dict,
            )
            atoms_dimer.eigenmodes = [vector]
        opt.attach(reset_eigenmode_obs, interval=1)
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        def check_displacement():
            displ = np.linalg.norm(atoms_dimer.positions-atoms.positions)
            if displ > max_displacement:
                opt.max_steps = 0
        opt.insert_observer(function=check_displacement, interval=10)
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=steps_max)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_dimer.positions)
    if update_cell is True:
        atoms.set_cell(atoms_dimer.cell)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        energy=atoms_dimer.calc.results['energy'],
        forces=atoms_dimer.calc.results['forces'],
    )
    atoms.info["vector"] = atoms_dimer.eigenmodes[0]
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# RUN CLIMBBONDS CALCULATION
# -----------------------------------------------------------------------------

def run_climbbonds_calculation(
    atoms,
    bonds_TS,
    calc,
    name='climbbonds',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.01,
    steps_max=500,
    max_displacement=None,
):
    """Run a climbbonds calculation."""
    from arkimede.optimize.climbbonds import ClimbBonds
    from ase.optimize.ode import ODE12r
    
    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)
    
    # Name of the trajectory file.
    if save_trajs is True:
        trajectory = os.path.join(directory, f'{name}.traj')
    else:
        trajectory = None
    
    # Create a copy of the atoms object to not mess with constraints.
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    atoms_copy.set_constraint(
        atoms_copy.constraints+[ClimbBonds(bonds = bonds_TS)]
    )

    # We use an ODE solver to improve convergence.
    opt = ODE12r(
        atoms=atoms_copy,
        logfile=logfile,
        trajectory=trajectory,
    )
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        def check_displacement():
            displ = np.linalg.norm(atoms_copy.positions-atoms.positions)
            if displ > max_displacement:
                opt.max_steps = 0
        opt.insert_observer(function=check_displacement, interval=10)
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=steps_max)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_copy.positions)
    if update_cell is True:
        atoms.set_cell(atoms_copy.cell)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        energy=atoms_copy.calc.results['energy'],
        forces=atoms_copy.calc.results['forces'],
    )
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# RUN CLIMBFIXINT CALCULATION
# -----------------------------------------------------------------------------

def run_climbfixint_calculation(
    atoms,
    bonds_TS,
    calc,
    name='climbfixint',
    index_constr2climb=0,
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.01,
    steps_max=500,
    optB_kwargs={'logfile': '-', 'trajectory': None},
    max_displacement=None,
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
    if save_trajs is True:
        trajectory = os.path.join(directory, f'{name}.traj')
    else:
        trajectory = None
    
    # Create a copy of the atoms object to not mess with constraints.
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    
    bonds = [[None, bond[:2]] for bond in bonds_TS]
    atoms_copy.set_constraint([FixInternals(bonds=bonds)]+atoms.constraints)
    
    # We use an ODE solver to improve convergence.
    opt = BFGSClimbFixInternals(
        atoms=atoms_copy,
        logfile=logfile,
        trajectory=trajectory,
        index_constr2climb=index_constr2climb,
        optB_kwargs=optB_kwargs,
    )
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        def check_displacement():
            displ = np.linalg.norm(atoms_copy.positions-atoms.positions)
            if displ > max_displacement:
                opt.max_steps = 0
        opt.insert_observer(function=check_displacement, interval=10)
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=steps_max)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_copy.positions)
    if update_cell is True:
        atoms.set_cell(atoms_copy.cell)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        energy=atoms_copy.calc.results['energy'],
        forces=atoms_copy.calc.results['forces'],
    )
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# RUN SELLA CALCULATION
# -----------------------------------------------------------------------------

def run_sella_calculation(
    atoms,
    calc,
    bonds_TS,
    name='sella',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.01,
    steps_max=500,
):
    """Run a sella calculation."""
    from sella import Sella, Constraints

    # Create directory to store the results.
    if save_trajs is True or write_images is True:
        os.makedirs(directory, exist_ok=True)

    # Name of the trajectory file.
    if save_trajs is True:
        trajectory = os.path.join(directory, f'{name}.traj')
    else:
        trajectory = None

    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    indices = get_atoms_not_fixed(atoms_copy, return_mask=False)
    
    atoms_copy.constraints = []
    constraints = Constraints(atoms_copy)
    constraints.fix_translation(indices)
    
    opt = Sella(
        atoms=atoms_copy,
        constraints=constraints,
        trajectory=trajectory,
        logfile=logfile,
    )
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=steps_max)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_copy.positions)
    if update_cell is True:
        atoms.set_cell(atoms_copy.cell)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        energy=atoms_copy.calc.results['energy'],
        forces=atoms_copy.calc.results['forces'],
    )
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# RUN TS SEARCH CALCULATION
# -----------------------------------------------------------------------------

def run_TS_search_calculation(
    atoms,
    calc,
    bonds_TS,
    search_TS,
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.01,
    steps_max=500,
    **kwargs,
):
    """Run TS search calculation."""
    if search_TS == "dimer":
        # Dimer calculation.
        run_dimer_calculation(
            atoms=atoms,
            calc=calc,
            steps_max=steps_max,
            bonds_TS=bonds_TS,
            fmax=fmax,
            directory=directory,
            save_trajs=save_trajs,
            write_images=write_images,
            **kwargs,
        )
    elif search_TS == "climbbonds":
        # Climbbonds calculaton.
        run_climbbonds_calculation(
            atoms=atoms,
            calc=calc,
            steps_max=steps_max,
            bonds_TS=bonds_TS,
            fmax=fmax,
            directory=directory,
            save_trajs=save_trajs,
            write_images=write_images,
            **kwargs,
        )
    elif search_TS == "climbfixint":
        # ClimbFixInternals calculaton.
        run_climbfixint_calculation(
            atoms=atoms,
            calc=calc,
            steps_max=steps_max,
            bonds_TS=bonds_TS,
            fmax=fmax,
            directory=directory,
            save_trajs=save_trajs,
            write_images=write_images,
            **kwargs,
        )
    elif search_TS == "sella":
        # Sella calculaton.
        run_sella_calculation(
            atoms=atoms,
            calc=calc,
            steps_max=steps_max,
            bonds_TS=bonds_TS,
            fmax=fmax,
            directory=directory,
            save_trajs=save_trajs,
            write_images=write_images,
            **kwargs,
        )

# -----------------------------------------------------------------------------
# RUN VIBRATIONS CALCULATION
# -----------------------------------------------------------------------------

def run_vibrations_calculation(atoms, calc, indices="not_surface"):
    """Run a vibrations calculation."""
    from ase.vibrations import Vibrations
    
    # Get indices of atoms to vibrate.
    if indices == "not_surface":
        indices = get_atoms_not_surface(atoms)
    elif indices == "not_fixed":
        indices = get_atoms_not_fixed(atoms)
    
    # Initialize the vibrations calculation.
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    vib = Vibrations(atoms=atoms_copy, indices=indices, delta=0.01, nfree=2)
    
    # Run the calculation and get normal frequencies.
    vib.run()
    vib_energies = vib.get_energies()
    vib.clean()
    os.rmdir("vib")

    atoms.info["vib_energies"] = vib_energies

    return vib_energies

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
