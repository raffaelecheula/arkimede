# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from scipy.integrate import LSODA
from ase.geometry import find_mic
from ase.constraints import FixConstraint
from ase.neighborlist import natural_cutoffs

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
        scale_forces=2.,
        doublehookean=True,
        atoms_IS_FS=None,
        bondratio_thr=1.0,
        k_spring_displ=1.,
        k_spring_IS_FS=1.,
        sign_bond_dict={'break': +1, 'form': -1},
    ):
        self.bonds = [bond[:2] for bond in bonds]
        self.signs = [sign_bond_dict[bond[2]] for bond in bonds]
        self.ftol = ftol
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound
        self.scale_forces = scale_forces
        self.bondlengths = np.zeros(len(self.bonds))
        # Doublehookean parameters.
        self.doublehookean = doublehookean if atoms_IS_FS else None
        if self.doublehookean:
            self.atoms_IS, self.atoms_FS = atoms_IS_FS
            self.bondratio_thr = bondratio_thr
            self.k_spring_displ = k_spring_displ
            self.k_spring_IS_FS = k_spring_IS_FS
            self.initialized = False
    
    def initialize_doublehookean(self, atoms):
        self.initialized = True
        self.bondlengths_min = []
        self.bondlengths_max = []
        self.bondlengths_ave = []
        self.too_short = []
        self.too_long = []
        for ii, bond in enumerate(self.bonds):
            bondlen_IS = self.atoms_IS.get_distance(*bond, mic=True)
            bondlen_FS = self.atoms_FS.get_distance(*bond, mic=True)
            self.too_short.append(False)
            self.too_long.append(False)
            if self.signs[ii] > 0:
                self.bondlengths_min.append(bondlen_IS)
                self.bondlengths_max.append(bondlen_FS)
            else:
                self.bondlengths_min.append(bondlen_FS)
                self.bondlengths_max.append(bondlen_IS)
            self.bondlengths_ave.append((bondlen_IS+bondlen_FS)/2.)
   
    def apply_doublehookean(self, atoms, forces):
        if not self.initialized:
            self.initialize_doublehookean(atoms=atoms)
        for ii, bond in enumerate(self.bonds):
            p1, p2 = atoms.positions[bond]
            if self.bondlengths[ii]/self.bondlengths_min[ii] < self.bondratio_thr:
                self.too_short[ii] = True
                print("warning: bond too short")
            if self.bondlengths_max[ii]/self.bondlengths[ii] < self.bondratio_thr:
                self.too_long[ii] = True
                print("warning: bond too long")
            if self.bondlengths[ii] > self.bondlengths_ave[ii]:
                self.too_short[ii] = False
            if self.bondlengths[ii] < self.bondlengths_ave[ii]:
                self.too_long[ii] = False
            # Add restoring force if bond is too short.
            if self.too_short[ii] is True:
                displ = find_mic(p2-p1, cell=atoms.cell, pbc=True)[0]
                if self.signs[ii] > 0:
                    p1_ref, p2_ref = self.atoms_FS.positions[bond]
                else:
                    p1_ref, p2_ref = self.atoms_IS.positions[bond]
                forces[bond[0]] += self.k_spring_IS_FS * (p1_ref - p1)
                forces[bond[1]] += self.k_spring_IS_FS * (p2_ref - p2)
                forces[bond[0]] -= self.k_spring_displ * displ
                forces[bond[1]] += self.k_spring_displ * displ
            # Add restoring force if bond is too long.
            if self.too_long[ii] is True:
                displ = find_mic(p2-p1, cell=atoms.cell, pbc=True)[0]
                if self.signs[ii] > 0:
                    p1_ref, p2_ref = self.atoms_IS.positions[bond]
                else:
                    p1_ref, p2_ref = self.atoms_FS.positions[bond]
                forces[bond[0]] += self.k_spring_IS_FS * (p1_ref - p1)
                forces[bond[1]] += self.k_spring_IS_FS * (p2_ref - p2)
                forces[bond[0]] += self.k_spring_displ * displ
                forces[bond[1]] -= self.k_spring_displ * displ

    def adjust_forces(self, atoms, forces):
        
        forces0 = forces.copy()
        masses = atoms.get_masses()
        for ii, bond in enumerate(self.bonds):
            self.bondlengths[ii] = atoms.get_distance(*bond, mic=True)
        
        if self.doublehookean:
            self.apply_doublehookean(atoms=atoms, forces=forces)
            if any(self.too_long+self.too_short):
                return

        def get_dforces(time, forces):
            x_dot_tot = 0.
            forces = forces.reshape(-1, 3)
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

        forces[:] = solver.y.reshape(-1, 3)
        self.projected_forces = forces-forces0
        forces += self.scale_forces * self.projected_forces

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
