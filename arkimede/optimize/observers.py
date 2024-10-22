# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

# -------------------------------------------------------------------------------------
# MIN STEPS OBS
# -------------------------------------------------------------------------------------

def min_steps_obs(opt, min_steps, fmax):
    if opt.nsteps >= min_steps:
        opt.fmax = fmax
    else:
        opt.fmax = 0.

# -------------------------------------------------------------------------------------
# MAX FORCECALLS OBS
# -------------------------------------------------------------------------------------

def max_forcecalls_obs(opt, max_forcecalls, calc):
    if "counter" in dir(calc) and calc.counter > max_forcecalls:
        opt.max_steps = 0

# -------------------------------------------------------------------------------------
# ACTIVATE CLIMB OBS
# -------------------------------------------------------------------------------------

def activate_climb_obs(neb, ftres):
    if neb.get_residual() < ftres:
        neb.climb = True

# -------------------------------------------------------------------------------------
# PRINT ENERGIES OBS
# -------------------------------------------------------------------------------------

def print_energies_obs(neb, opt, print_path_energies=True):
    fres = neb.get_residual()
    print(f'Step: {opt.nsteps:4d} Fmax: {fres:9.4f}', end='  ')
    for ii in (0, -1):
        neb.energies[ii] = neb.images[ii].calc.results['energy']
    act_energy = max(neb.energies)-neb.energies[0]
    print(f'Eact: {act_energy:+9.4f} eV', end='  ')
    if print_path_energies is True:
        print(f'Epath:', end='  ')
        for energy in neb.energies:
            print(f'{energy-neb.energies[0]:+9.4f}', end=' ')
        print('eV')

# -------------------------------------------------------------------------------------
# RESET EIGENMODE OBS
# -------------------------------------------------------------------------------------

def reset_eigenmode_obs(atoms_opt, bonds_TS, sign_bond_dict):
    from arkimede.utilities import get_vector_from_bonds_TS
    vector = get_vector_from_bonds_TS(
        atoms=atoms_opt,
        bonds_TS=bonds_TS,
        sign_bond_dict=sign_bond_dict,
    )
    atoms_opt.eigenmodes = [vector]

# -------------------------------------------------------------------------------------
# MAX DISPLACEMENT OBS
# -------------------------------------------------------------------------------------

def max_displacement_obs(atoms_new, atoms_old, max_displacement):
    displ = np.linalg.norm(atoms_new.positions-atoms_old.positions)
    if displ > max_displacement:
        raise RuntimeError("atoms displaced too much")

# -------------------------------------------------------------------------------------
# MAX FORCE TOT OBS
# -------------------------------------------------------------------------------------

def max_force_tot_obs(atoms_opt, max_force_tot):
    if (atoms_opt.get_forces() ** 2).sum(axis=1).max() > max_force_tot ** 2:
        raise RuntimeError("max force too high")

# -------------------------------------------------------------------------------------
# ATOMS TOO CLOSE OBS
# -------------------------------------------------------------------------------------

def atoms_too_close_obs(atoms_opt, mindist=0.2):
    for ii, position in enumerate(atoms_opt.positions):
        distances = np.linalg.norm(atoms_opt.positions[ii+1:]-position, axis=1)
        duplicates = np.where(distances < mindist)[0]
        if len(duplicates) > 0:
            raise RuntimeError("atoms too close")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
