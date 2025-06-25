# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.optimize.optimize import Optimizer

# -------------------------------------------------------------------------------------
# TS OPTIMIZER
# -------------------------------------------------------------------------------------

class TSOptimizer(Optimizer):
    def __init__(self, atoms, bonds, alpha=0.1, memory=5, max_steps=100, **kwargs):
        super().__init__(atoms, **kwargs)
        self.alpha = alpha
        self.memory = memory
        self.max_steps = max_steps
        self.bonds = bonds

        self.s_list = []
        self.y_list = []
        self.rho_list = []

        # Identify movable (unconstrained) atoms
        mask = np.ones(len(atoms), dtype=bool)
        if atoms.constraints:
            for c in atoms.constraints:
                if hasattr(c, 'index'):
                    mask[c.index] = False
                elif hasattr(c, 'get_indices'):
                    mask[c.get_indices()] = False
        self.free_idx = np.where(mask)[0]

    def invert_forces(self, forces, positions):
        """Invert forces only along specific bonds among unconstrained atoms."""
        f_mod = forces.copy()
        for i, j in self.bonds:
            if i in self.free_idx and j in self.free_idx:
                vec = positions[j] - positions[i]
                norm = np.linalg.norm(vec)
                if norm < 1e-8:
                    continue
                vec /= norm
                f_i = np.dot(f_mod[i], vec)
                f_j = np.dot(f_mod[j], -vec)
                f_mod[i] -= 2 * f_i * vec
                f_mod[j] -= 2 * f_j * (-vec)
        return f_mod

    def two_loop_recursion(self, q):
        alphas = []
        q = q.copy()
        for s, y, rho in reversed(list(zip(self.s_list, self.y_list, self.rho_list))):
            alpha = rho * np.dot(s, q)
            q -= alpha * y
            alphas.append(alpha)
        if self.y_list:
            y, s = self.y_list[-1], self.s_list[-1]
            H0 = np.dot(y, s) / np.dot(y, y)
            q *= H0
        for s, y, rho, alpha in zip(
            self.s_list, self.y_list, self.rho_list, reversed(alphas)
        ):
            beta = rho * np.dot(y, q)
            q += s * (alpha - beta)
        return -q

    def step(self, f=None):
        pos = self.atoms.get_positions()
        forces = self.atoms.get_forces()
        f_mod = self.invert_forces(forces, pos)

        # Restrict optimization to unconstrained atoms
        pos_flat = pos[self.free_idx].flatten()
        f_mod_flat = f_mod[self.free_idx].flatten()

        if len(self.s_list) > 0:
            direction = self.two_loop_recursion(f_mod_flat)
        else:
            direction = -f_mod_flat

        step = self.alpha * direction.reshape((-1, 3))
        new_pos = pos.copy()
        new_pos[self.free_idx] += step

        self.atoms.set_positions(new_pos)

        # New forces and update vectors
        forces_new = self.atoms.get_forces()
        f_mod_new = self.invert_forces(forces_new, new_pos)

        s = (new_pos[self.free_idx] - pos[self.free_idx]).flatten()
        y = (f_mod_new[self.free_idx] - f_mod[self.free_idx]).flatten()

        if np.dot(s, y) > 1e-8:
            rho = 1.0 / np.dot(y, s)
            self.s_list.append(s)
            self.y_list.append(y)
            self.rho_list.append(rho)
            if len(self.s_list) > self.memory:
                self.s_list.pop(0)
                self.y_list.pop(0)
                self.rho_list.pop(0)
        else:
            print("Skipped BFGS update due to curvature condition.")

    def run(self, fmax=0.05, steps=None):
        if steps is None:
            steps = self.max_steps
        return super().run(fmax=fmax, steps=steps)

#from ase import Atoms
#from ase.constraints import FixAtoms
#from ase.calculators.emt import EMT
#
## Create molecule
#atoms = Atoms('H5', positions=np.random.randn(5, 3))
#atoms.set_calculator(EMT())
#
## Freeze atoms 0, 4
#constraint = FixAtoms(indices=[0, 4])
#atoms.set_constraint(constraint)
#
## Use TS-aware optimizer with bond (1, 2) inverted
#bonds = [(1, 2)]
#opt = TSOptimizer(atoms, bonds=bonds, alpha=0.05, max_steps=100, logfile='ts_opt.log')
#opt.run(fmax=0.05)
#
#print(atoms.get_positions())

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
