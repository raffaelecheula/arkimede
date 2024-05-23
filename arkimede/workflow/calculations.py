# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.io import Trajectory
from ase.optimize import BFGS, LBFGS, ODE12r
from ase.neb import NEBOptimizer
from ase.calculators.singlepoint import SinglePointCalculator
from arkimede.workflow.utilities import (
    get_indices_not_fixed,
    get_indices_adsorbate,
    filter_results,
    get_atoms_too_close,
)

# -------------------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -------------------------------------------------------------------------------------

def run_relax_calculation(
    atoms,
    calc,
    fmax=0.01,
    steps_max=300,
    logfile='-',
    label='relax',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    optimizer=BFGS,
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

    # Setup the optimizer.
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    opt = optimizer(
        atoms=atoms_copy,
        logfile=logfile,
        trajectory=trajname,
        **kwargs_opt,
    )
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
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
    results = filter_results(results=atoms_copy.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)
    if "counter" in dir(calc):
        atoms.info["counter"] = calc.counter

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
    fmax=0.01,
    steps_max=300,
    k_neb=0.10,
    label='neb',
    directory='.',
    save_trajs=False,
    write_images=False,
    logfile='-',
    use_OCPdyNEB=False,
    optimizer=NEBOptimizer,
    kwargs_opt={},
    activate_climb=True,
    print_energies=True,
    ftres_climb=0.10,
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
    
    # Setup optimizer.
    opt = optimizer(neb, logfile=logfile, **kwargs_opt)
    
    def activate_climb_obs(neb=neb, ftres=ftres_climb):
        if neb.get_residual() < ftres:
            neb.climb = True
    
    def print_energies_obs(neb=neb, opt=opt, print_path_energies=True):
        fres = neb.get_residual()
        print(f'Step: {opt.nsteps:4d} Fmax: {fres:9.4f}', end='  ')
        for ii in (0, -1):
            neb.energies[ii] = images[ii].calc.results['energy']
        act_energy = max(neb.energies)-neb.energies[0]
        print(f'Eact: {act_energy:+9.4f} eV', end='  ')
        if print_path_energies is True:
            print(f'Epath:', end='  ')
            for energy in neb.energies:
                print(f'{energy-neb.energies[0]:+9.4f}', end=' ')
            print('eV')

    # Attach observers.
    if activate_climb:
        opt.attach(activate_climb_obs, interval=1)
    if print_energies:
        opt.attach(print_energies_obs, interval=1)

    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0

    # Run the calculation. The while is to keep NEBOptimizer going when 
    # it fails randomly.
    converged = False
    while opt.nsteps < steps_max and not converged:
        converged = opt.run(fmax=fmax, steps=steps_max-opt.nsteps)
    
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
    fmax=0.01,
    steps_max=500,
    max_displacement=False,
    max_forces=20.,
    reset_eigenmode=True,
    sign_bond_dict={'break': +1, 'form': -1},
    properties=["energy", "forces"],
    **kwargs,
):
    """Run a dimer calculation."""
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
    from arkimede.workflow.utilities import get_vector_from_bonds_TS

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
        trajectory=trajname,
        logfile=logfile,
    )
    
    # Observer that reset the eigenmode to the direction of the ts bonds.
    # Maybe there is a better way of doing this (e.g., selecting the 
    # eigenmode most similar to the directions of the ts bonds).
    if bonds_TS and reset_eigenmode:
        def reset_eigenmode_obs(atoms_dimer=atoms_dimer):
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
                raise RuntimeError("atoms displaced too much")
        opt.insert_observer(function=check_displacement, interval=10)
    
    if max_forces:
        def check_max_forces():
            if (atoms_dimer.get_forces() ** 2).sum(axis=1).max() > max_forces ** 2:
                raise RuntimeError("max force too high")
        opt.insert_observer(function=check_max_forces, interval=5)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
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
    results = filter_results(results=atoms_dimer.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    atoms.info["vector"] = atoms_dimer.eigenmodes[0]
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)
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
    fmax=0.01,
    steps_max=500,
    max_displacement=None,
    max_forces=20.,
    atoms_too_close=True,
    optimizer=ODE12r,
    kwargs_opt={},
    properties=["energy", "forces"],
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
    
    # Create a copy of the atoms object to not mess with constraints.
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    atoms_copy.set_constraint(atoms_copy.constraints+[ClimbBonds(bonds=bonds_TS)])

    # TODO:
    #from arkimede.optimize.climbbonds import AdaptiveBFGS
    #optimizer = AdaptiveBFGS
    #kwargs_opt = {}
    #from ase.optimize.sciopt import SciPyFminCG
    #optimizer = SciPyFminCG
    #kwargs_opt = {}

    # Setup the optimizer.
    opt = optimizer(
        atoms=atoms_copy,
        logfile=logfile,
        trajectory=trajname,
        **kwargs_opt,
    )
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        def check_displacement():
            displ = np.linalg.norm(atoms_copy.positions-atoms.positions)
            if displ > max_displacement:
                raise RuntimeError("atoms displaced too much")
        opt.insert_observer(function=check_displacement, interval=10)
    
    if max_forces:
        def check_max_forces():
            if (atoms_copy.get_forces() ** 2).sum(axis=1).max() > max_forces ** 2:
                raise RuntimeError("max force too high")
        opt.insert_observer(function=check_max_forces, interval=10)
    
    if atoms_too_close:
        def check_atoms_too_close():
            if get_atoms_too_close(atoms_copy, return_anomaly=True) is True:
                raise RuntimeError("atoms too close")
        opt.insert_observer(function=check_atoms_too_close, interval=10)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
    # Run the calculation.
    try:
        opt.run(fmax=fmax, steps=steps_max)
        converged = opt.converged()
    except Exception as error:
        print(error)
        converged = False
    
    # TODO:
    #print(calc.counter)
    #exit()
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_copy.positions)
    if update_cell is True:
        atoms.set_cell(atoms_copy.cell)
    results = filter_results(results=atoms_copy.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)
    if "counter" in dir(calc):
        atoms.info["counter"] = calc.counter

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
    fmax=0.01,
    steps_max=500,
    optB_kwargs={'logfile': '-', 'trajectory': None},
    max_displacement=None,
    max_forces=20.,
    properties=["energy", "forces"],
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
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    
    bonds = [[None, bond[:2]] for bond in bonds_TS]
    atoms_copy.set_constraint([FixInternals(bonds=bonds)]+atoms.constraints)
    
    # We use an ODE solver to improve convergence.
    opt = BFGSClimbFixInternals(
        atoms=atoms_copy,
        logfile=logfile,
        trajectory=trajname,
        index_constr2climb=index_constr2climb,
        optB_kwargs=optB_kwargs,
    )
    
    # Observer that checks the displacement of the ts from the starting position.
    if max_displacement:
        def check_displacement():
            displ = np.linalg.norm(atoms_copy.positions-atoms.positions)
            if displ > max_displacement:
                raise RuntimeError("atoms displaced too much")
        opt.insert_observer(function=check_displacement, interval=10)
    
    if max_forces:
        def check_max_forces():
            if (atoms_copy.get_forces() ** 2).sum(axis=1).max() > max_forces ** 2:
                raise RuntimeError("max force too high")
        opt.insert_observer(function=check_max_forces, interval=5)
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
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
    results = filter_results(results=atoms_copy.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)
    if "counter" in dir(calc):
        atoms.info["counter"] = calc.counter

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
    bonds_TS,
    label='sella',
    directory='.',
    save_trajs=False,
    write_images=False,
    update_cell=False,
    logfile='-',
    fmax=0.01,
    steps_max=500,
    properties=["energy", "forces"],
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

    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    indices = get_indices_not_fixed(atoms_copy, return_mask=False)
    
    atoms_copy.constraints = []
    constraints = Constraints(atoms_copy)
    constraints.fix_translation(indices)
    
    opt = Sella(
        atoms=atoms_copy,
        constraints=constraints,
        trajectory=trajname,
        logfile=logfile,
    )
    
    # Calculate the number of calculator calls.
    if "counter" in dir(calc):
        calc.counter = 0
    
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
    results = filter_results(results=atoms_copy.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    atoms.info["modified"] = False
    atoms.info["converged"] = bool(converged)
    if "counter" in dir(calc):
        atoms.info["counter"] = calc.counter

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
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    
    class VibrationsWithEnergy(Vibrations):
        def calculate(self, atoms, disp):
            self.calc.calculate(
                atoms=atoms,
                properties=properties,
                system_changes=["positions"],
            )
            if self.ir:
                self.calc.get_dipole_moment(atoms)
            return self.calc.results
    
    vib = VibrationsWithEnergy(
        atoms=atoms_copy,
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
        atoms_vib.info["displacement"] = disp.name
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
