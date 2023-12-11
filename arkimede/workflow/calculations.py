# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from .utilities import get_atoms_not_fixed

# -----------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -----------------------------------------------------------------------------

def run_relax_calculation(
    atoms,
    calc,
    fmax=0.01,
    steps_max=300,
    name=None,
):
    """Run a relax calculation."""
    from ase.optimize import BFGS

    atoms.calc = calc
    opt = BFGS(
        atoms = atoms,
        trajectory = f'{name}_relax.traj' if name else None,
    )
    opt.run(fmax = fmax)
    return atoms

# -----------------------------------------------------------------------------
# RUN NEB CALCULATION
# -----------------------------------------------------------------------------

def run_neb_calculation(
    images,
    calc,
    fmax=0.01,
    steps_max=300,
    k_neb=0.10,
    name=None,
    use_OCPdyNEB=False,
    use_NEBOptimizer=False,
    activate_climb=True,
    print_energies=True,
    ftres_climb=0.10,
):
    """Run a NEB calculation."""
    if use_OCPdyNEB:
        from ocpneb.core import OCPdyNEB
        neb = OCPdyNEB(
            images = images,
            checkpoint_path = calc.config['checkpoint'],
            k = k_neb,
            climb = False,
            dynamic_relaxation = False,
            method = "aseneb",
            cpu = False,
            batch_size = 4,
        )
    else:
        from ase.neb import NEB
        from ase.calculators.singlepoint import SinglePointCalculator
        neb = NEB(
            images = images,
            k = k_neb,
            climb = False,
            parallel = False,
            method = 'aseneb',
            allow_shared_calculator = True,
        )
        for image in images:
            image.calc = calc
        for ii in (0, -1):
            images[ii].get_potential_energy()
            images[ii].calc = SinglePointCalculator(
                atoms = images[ii],
                energy = images[ii].calc.results['energy'],
                forces = images[ii].calc.results['forces'],
            )
    
    if use_NEBOptimizer:
        from ase.neb import NEBOptimizer
        opt = NEBOptimizer(neb, logfile = None)
    else:
        from ase.optimize import BFGS
        opt = BFGS(neb, logfile = None)
    
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

    if activate_climb:
        opt.attach(activate_climb_obs, interval = 1)
    if print_energies:
        opt.attach(print_energies_obs, interval = 1)

    # The while is to keep NEBOptimizer going when it fails randomly.
    while opt.nsteps < steps_max:
        converged = opt.run(fmax = fmax, steps = steps_max-opt.nsteps)
    
    if name and converged:
        from ase.io import Trajectory
        traj = Trajectory(filename = f'{name}_neb.traj', mode = 'w')
        for image in images:
            traj.write(image)
    
    return images, converged

# -----------------------------------------------------------------------------
# RUN DIMER CALCULATION
# -----------------------------------------------------------------------------

def run_dimer_calculation(
    atoms,
    vector,
    calc,
    bonds_TS,
    name=None,
    fmax=0.01,
    reset_eigenmode=True,
    sign_bond_dict={'break': +1, 'form': -1},
):
    """Run a dimer calculation."""
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate

    atoms.calc = calc
    mask = get_atoms_not_fixed(atoms, return_mask=True)
    
    dimer_control = DimerControl(
        initial_eigenmode_method = 'displacement',
        displacement_method = 'vector',
        logfile = None,
        cg_translation = True,
        use_central_forces = True,
        extrapolate_forces = False,
        order = 1,
        f_rot_min = 0.10,
        f_rot_max = 1.00,
        max_num_rot = 1,
        trial_angle = np.pi / 4,
        trial_trans_step = 0.001,
        maximum_translation = 0.10,
        dimer_separation = 0.0001,
    )
    atoms_dimer = MinModeAtoms(
        atoms = atoms,
        control = dimer_control,
        mask = mask,
    )
    atoms_dimer.displace(displacement_vector = vector)

    opt = MinModeTranslate(
        atoms = atoms_dimer,
        trajectory = f'{name}_dimer.traj' if name else None,
        logfile = '-',
    )
    
    def reset_eigenmode_obs(opt = opt):
        eigenmode = np.zeros((len(atoms_dimer), 3))
        for bond in bonds_TS:
            index_a, index_b, sign_bond = bond
            if isinstance(sign_bond, str):
                sign_bond = sign_bond_dict[sign_bond]
            dir_bond = (
                atoms_dimer.positions[index_a]-atoms_dimer.positions[index_b]
            )
            eigenmode[index_a] += +dir_bond * sign_bond
            eigenmode[index_b] += -dir_bond * sign_bond
        eigenmode /= np.linalg.norm(eigenmode)
        opt.eigenmodes = [eigenmode/np.linalg.norm(eigenmode)]

    if bonds_TS and reset_eigenmode:
        opt.attach(reset_eigenmode_obs, interval = 1)
    opt.run(fmax = fmax)
    atoms.set_positions(atoms_dimer.positions)

    return atoms

# -----------------------------------------------------------------------------
# RUN CLIMBBONDS CALCULATION
# -----------------------------------------------------------------------------

def run_climbbonds_calculation(
    atoms,
    bonds_TS,
    calc,
    name=None,
    fmax=0.01,
):
    """Run a climbbonds calculation."""
    from .climbbonds import ClimbBondLengths
    from ase.optimize.ode import ODE12r
    
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    atoms_copy.set_constraint(
        atoms_copy.constraints+[ClimbBondLengths(bonds = bonds_TS)]
    )

    opt = ODE12r(
        atoms = atoms_copy,
        trajectory = f'{name}_climbbonds.traj' if name else None,
        alpha = 150,
    )
    opt.run(fmax = fmax)
    atoms.set_positions(atoms_copy.positions)

    return atoms

# -----------------------------------------------------------------------------
# RUN VIBRATIONS CALCULATION
# -----------------------------------------------------------------------------

def run_vibrations_calculation(atoms):
    
    from ase.vibrations import Vibrations
    
    indices = get_atoms_not_fixed(atoms)
    vib = Vibrations(atoms, indices = indices, delta = 0.01, nfree = 2)
    vib.run()

    return vib.get_frequencies()

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
