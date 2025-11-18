# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from scipy.integrate import LSODA
from ase.geometry import find_mic
from ase.constraints import FixConstraint
from ase.neighborlist import natural_cutoffs

# -------------------------------------------------------------------------------------
# CLIMB TS BONDS
# -------------------------------------------------------------------------------------

class ClimbTSBonds(FixConstraint):
    
    def __init__(
        self,
        bonds: list,
        clip_coeff: float = 0.05,
        restore_bonds: bool = False,
        c1_bondmin: float = 0.8,
        c2_bondmax: float = 2.8,
        c3_repuls: float = 1.0,
        c4_repuls: float = 100.0,
        c5_attrac: float = 1.0,
        c6_attrac: float = 100.0,
        restore_clip: float = 1000.0,
        restore_mod: list = [+0.0, +0.0, -1.0],
    ):
        self.bonds = [bond[:2] for bond in bonds]
        self.restore_bonds = restore_bonds
        self.clip_coeff = clip_coeff
        self.c1_bondmin = c1_bondmin
        self.c2_bondmax = c2_bondmax
        self.c3_repuls = c3_repuls
        self.c4_repuls = c4_repuls
        self.c5_attrac = c5_attrac
        self.c6_attrac = c6_attrac
        self.restore_clip = restore_clip
        self.restore_mod = np.array(restore_mod)
    
    def adjust_forces(self, atoms, forces):
        """
        Mirror the forces acting on the TS atoms parallel to the TS bond.
        """
        # Copy the original forces.
        forces0 = forces.copy()
        # Initialize lists.
        bond_unit_list = []
        dot_prod_a_list = []
        dot_prod_b_list = []
        # Iterate over bonds and remove parallel forces.
        for ii, (index_a, index_b) in enumerate(self.bonds):
            # Get bond direction.
            bond_vector = atoms.positions[index_a] - atoms.positions[index_b]
            bond_unit = bond_vector / np.linalg.norm(bond_vector)
            # Get force acting on a parallel to the bond.
            forces_vector_a = forces0[index_a]
            dot_prod_a = np.dot(forces_vector_a, bond_unit)
            # Get force acting on b parallel to the bond.
            forces_vector_b = forces0[index_b]
            dot_prod_b = np.dot(forces_vector_b, bond_unit)
            # Remove parallel forces.
            forces[index_a] -= dot_prod_a * bond_unit
            forces[index_b] -= dot_prod_b * bond_unit
            # Store values of bond direction and dot products.
            bond_unit_list.append(bond_unit)
            dot_prod_a_list.append(dot_prod_a)
            dot_prod_b_list.append(dot_prod_b)
            # Add a restoring force if atoms are too close or too distant.
            if self.restore_bonds is True:
                bondlen = np.linalg.norm(bond_vector)
                bondlen_natural = sum(natural_cutoffs(atoms[index_a, index_b]))
                bondlen_min = self.c1_bondmin * bondlen_natural
                bondlen_max = self.c2_bondmax * bondlen_natural
                restore = (
                    + self.c3_repuls * np.exp(self.c4_repuls * (bondlen_min - bondlen))
                    - self.c5_attrac * np.exp(self.c6_attrac * (bondlen - bondlen_max))
                )
                if self.restore_clip is not None:
                    restore = np.clip(restore, -self.restore_clip, +self.restore_clip)
                forces[index_a] += restore * (bond_unit + self.restore_mod)
                forces[index_b] -= restore * (bond_unit - self.restore_mod)
        # Iterate over bonds and add modified parallel forces.
        for ii, (index_a, index_b) in enumerate(self.bonds):
            # Calculate maximum force.
            force_max = np.max(np.linalg.norm(forces, axis=1))
            # Get sum and difference of parallel forces.
            bond_unit = bond_unit_list[ii]
            dot_prod_dif = dot_prod_a_list[ii] - dot_prod_b_list[ii]
            dot_prod_sum = dot_prod_a_list[ii] + dot_prod_b_list[ii]
            # Clip sum and difference of parallel forces.
            if self.clip_coeff is not None:
                clip_value = self.clip_coeff / force_max
                if self.restore_bonds is True and np.abs(restore) > 0.05:
                    clip_value = 0.
                dot_prod_dif = np.clip(dot_prod_dif, -clip_value, +clip_value)
                dot_prod_sum = np.clip(dot_prod_sum, -clip_value, +clip_value)
            # Add force to climb to the saddle point.
            forces[index_a] -= dot_prod_dif * bond_unit
            forces[index_b] += dot_prod_dif * bond_unit
            # Add force to allow the bond to translate.
            forces[index_a] += dot_prod_sum * bond_unit
            forces[index_b] += dot_prod_sum * bond_unit

    def adjust_positions(self, atoms, forces):
        pass

    def index_shuffle(self, atoms, ind):
        """
        Shuffle the indices of the two atoms in this constraint.
        """
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        bonds = map[self.bonds]
        self.bonds = bonds[(bonds != -1).all(1)]
        if len(self.bonds) == 0:
            raise IndexError("Constraint not part of slice")

    def todict(self):
        return {
            "name": "ClimbTSBonds",
            "kwargs": {
                "bonds": list(self.bonds),
                "clip_coeff": float(self.clip_coeff),
                "restore_bonds": bool(self.restore_bonds),
                "c1_bondmin": float(self.c1_bondmin),
                "c2_bondmax": float(self.c2_bondmax),
                "c3_repuls": float(self.c3_repuls),
                "c4_repuls": float(self.c4_repuls),
                "c5_attrac": float(self.c5_attrac),
                "c6_attrac": float(self.c6_attrac),
                "restore_mod": list(self.restore_mod),
            },
        }

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
