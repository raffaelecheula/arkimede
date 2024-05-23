# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from scipy.integrate import LSODA
from ase.geometry import find_mic
from ase.constraints import FixConstraint

# -------------------------------------------------------------------------------------
# CLIMB BONDS
# -------------------------------------------------------------------------------------

class ClimbBonds(FixConstraint):
    
    def __init__(
        self,
        bonds,
        ftol=1e-8,
        rtol=1e-5,
        atol=1e-8,
        t_bound=1e+9,
    ):
        self.bonds = np.asarray([bond[:2] for bond in bonds])
        self.ftol = ftol
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound
        self.bondlengths = np.zeros(self.bonds.shape[0])
        
    def adjust_momenta(self, atoms, forces):
        
        masses = atoms.get_masses()
        for ii, bond in enumerate(self.bonds):
            self.bondlengths[ii] = atoms.get_distance(*bond, mic=True)

        def get_dforces(time, forces):
            x_dot_tot = 0.
            forces = forces.reshape(len(atoms), 3)
            dforces = np.zeros(forces.shape)
            for ii, bond in enumerate(self.bonds):
                index_a, index_b = bond
                dir_bond = atoms.positions[index_a] - atoms.positions[index_b]
                dir_bond, _ = find_mic(dir_bond, atoms.cell, atoms.pbc)
                dir_forces = (
                    +forces[index_a]/masses[index_a] 
                    -forces[index_b]/masses[index_b]
                )
                mass_red = 1 / (1/masses[index_a] + 1/masses[index_b])
                x_dot = -np.dot(dir_forces, dir_bond) / self.bondlengths[ii]**2
                dforces[index_a] = +x_dot * mass_red * dir_bond
                dforces[index_b] = -x_dot * mass_red * dir_bond
                x_dot_tot += abs(x_dot)
            if x_dot_tot < self.ftol:
                solver.status = 'finished'
            return dforces.flatten()

        solver = LSODA(
            fun=get_dforces,
            t0=0.,
            y0=forces.flatten(),
            rtol=self.rtol,
            atol=self.atol,
            t_bound=self.t_bound,
        )
        while solver.status != 'finished':
            solver.step()

        forces[:] = solver.y.reshape(len(atoms), 3)

    def adjust_forces(self, atoms, forces):
        forces0 = forces.copy()
        self.adjust_momenta(atoms, forces)
        self.projected_forces = forces-forces0
        forces += self.projected_forces

    def adjust_positions(self, atoms, forces):
        pass

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        bonds = map[self.bonds]
        self.bonds = bonds[(bonds != -1).all(1)]
        if len(self.bonds) == 0:
            raise IndexError('Constraint not part of slice')

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
