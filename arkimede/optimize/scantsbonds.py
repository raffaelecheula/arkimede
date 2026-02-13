# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS

# -------------------------------------------------------------------------------------
# RUN SCAN TS BONDS CALCULATION
# -------------------------------------------------------------------------------------

def run_scantsbonds_calculation(
    atoms: Atoms,
    calc: Calculator,
    bonds_TS: list = None,
    label: str = "scantsbonds",
    directory: str = ".",
    save_trajs: bool = False,
    write_images: bool = False,
    update_cell: bool = False,
    logfile: str = "-",
    fmax: float = 0.05,
    max_steps: int = 500,
    min_steps: int = None,
    max_displacement: float = None,
    max_force_tot: float = None,
    atoms_too_close: bool = False,
    max_forcecalls: int = None,
    optimizer: Optimizer = BFGS,
    opt_kwargs: dict = {"maxstep": 0.01},
    properties: list = ["energy", "forces"],
    **kwargs: dict,
):
    """
    Run a ScanTSBonds TS-search calculation.
    """
    from scipy.interpolate import InterpolatedUnivariateSpline
    from ase.constraints import FixBondLengths

    n_relax_max = 100
    n_root_max = 20
    delta = 0.02

    # Prepare a copy of atoms.
    atoms_opt = atoms.copy()
    atoms_opt.calc = calc
    
    pairs = [bond[:2] for bond in atoms_opt.info["bonds_TS"]]

    for jj in range(n_relax_max):
        # Relax TS structure.
        atoms_fix = atoms_opt.copy()
        atoms_fix.calc = calc
        constraint_bonds = FixBondLengths(pairs=pairs, tolerance=1e-5)
        atoms_fix.constraints.append(constraint_bonds)
        #opt = BFGS(atoms=atoms_fix)
        from ase.optimize import LBFGSLineSearch
        opt = LBFGSLineSearch(atoms=atoms_fix, trajectory="optimize.traj")
        try: opt.run(fmax=fmax)
        except: pass
        atoms_opt.positions = atoms_fix.positions.copy()
        # Calculate maximum forces.
        max_force = np.max(np.linalg.norm(atoms_opt.get_forces(), axis=1))
        print(f"fmax = {max_force:+7.4f} [eV/Å]")
        if max_force < fmax:
            status = "finished"
            break
        # Refine bond lengths.
        for bb, pair in enumerate(pairs):
            # Get bond direction and length.
            index_a, index_b = pair
            bond_vector = atoms_opt.positions[index_b] - atoms_opt.positions[index_a]
            bond_length_0 = np.linalg.norm(bond_vector)
            bond_unit = bond_vector / bond_length_0
            # Get projected forces.
            forces = atoms_opt.get_forces()
            forces_proj_0 = np.dot(forces[index_a], bond_unit)
            bond_length_list = [bond_length_0]
            forces_proj_list = [forces_proj_0]
            position_b = atoms_opt.positions[index_b].copy()
            # Displace atom B along bond direction in both directions.
            for ii in range(n_root_max):
                # Alternate sign of displacement.
                sign = +1 if ii % 2 == 0 else -1
                # Displace atom B along bond direction.
                atoms_opt[index_b].position += (ii + 1) * delta * sign * bond_unit
                bond_vector = atoms_opt.positions[index_b]-atoms_opt.positions[index_a]
                bond_length = np.linalg.norm(bond_vector)
                # Get projected forces.
                try: forces = atoms_opt.get_forces()
                except: continue
                forces_proj = np.dot(forces[index_a], bond_unit)
                bond_length_list.append(bond_length)
                forces_proj_list.append(forces_proj)
                # Stop if opposite sign is found.
                #if np.sign(forces_proj) != np.sign(forces_proj_0):
                #    break
                # Reset position of atom B.
                atoms_opt[index_b].position = position_b
            # Fit spline and get roots.
            x_vect, y_vect = np.array(bond_length_list), np.array(forces_proj_list)
            ii_sort = np.argsort(x_vect)
            spline = InterpolatedUnivariateSpline(x=x_vect[ii_sort], y=y_vect[ii_sort])
            bond_length_opt = spline.roots()[0]
            atoms_opt[index_b].position += (bond_length_opt-bond_length_0) * bond_unit
    else:
        status = "failed"
    # Return atoms.
    return atoms_opt

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
