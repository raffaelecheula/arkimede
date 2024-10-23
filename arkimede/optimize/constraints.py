# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.constraints import FixConstraint
from ase.geometry import find_mic, get_distances
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    natural_cutoffs,
)

# -------------------------------------------------------------------------------------
# PROPAGATE FORCES
# -------------------------------------------------------------------------------------

class PropagateForces(FixConstraint):
    
    def __init__(
        self,
        mult_1nn=0.50,
        mult_2nn=0.25,
        mult_nlist=1.25,
    ):
        self.mult_1nn = mult_1nn
        self.mult_2nn = mult_2nn
        self.mult_nlist = mult_nlist
        self.nlist = None

    def get_neighlist(self, atoms):
        cutoffs = natural_cutoffs(atoms=atoms, mult=self.mult_nlist)
        self.nlist = NeighborList(
            cutoffs=cutoffs,
            skin=0.3,
            self_interaction=False,
            bothways=False,
            primitive=NewPrimitiveNeighborList,
        )

    def modify_forces(self, forces, fixed=[]):
        deltaf = np.zeros(forces.shape)
        for aa in [aa for aa in range(forces.shape[0]) if aa not in fixed]:
            neighbors = [
                bb for bb in self.nlist.get_neighbors(aa)[0] if bb not in fixed
            ]
            if len(neighbors) > 0:
                deltaf[aa] = np.average(forces[neighbors], axis=0)
        return deltaf

    def adjust_forces(self, atoms, forces):
        if self.nlist is None:
            self.get_neighlist(atoms=atoms)
        self.nlist.update(atoms=atoms)
        fixed = np.where(np.linalg.norm(forces, axis=1) < 1e-20)[0]
        df_1nn = self.modify_forces(forces, fixed=fixed)
        df_2nn = self.modify_forces(df_1nn, fixed=fixed)
        forces += df_1nn * self.mult_1nn + df_2nn * self.mult_2nn

    def adjust_positions(self, atoms, forces):
        pass

    def todict(self):
        return {
            "name": "PropagateForces",
            "kwargs": {
                "mult_1nn": self.mult_1nn, 
                "mult_2nn": self.mult_2nn, 
                "mult_nlist":self.mult_nlist,
            }
        }

# -------------------------------------------------------------------------------------
# MIN BOND LENGTH
# -------------------------------------------------------------------------------------

class DoubleHookean(FixConstraint):
    
    def __init__(
        self,
        indices,
        bondlength,
        thr_min,
        thr_max,
        k_spring,
    ):
        self.indices = indices
        self.bondlength = bondlength
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.k_spring = k_spring
        self.too_short = False
        self.too_long = False

    def get_removed_dof(self, atoms):
        return 0

    def adjust_forces(self, atoms, forces):
        p1, p2 = atoms.positions[self.indices]
        displ, bondlength = find_mic(p1-p2, cell=atoms.cell, pbc=True)
        if bondlength/self.bondlength < self.thr_min:
            self.too_short = True
        elif bondlength/self.bondlength > self.thr_max:
            self.too_long = True
        if self.too_short is True and bondlength > self.bondlength:
            self.too_short = False
        if self.too_long is True and bondlength < self.bondlength:
            self.too_long = False
        # Add restoring force.
        if self.too_short is True or self.too_long is True:
            magnitude = self.k_spring * (bondlength - self.bondlength)
            direction = displ / bondlength
            forces[self.indices[0]] -= direction * magnitude
            forces[self.indices[1]] += direction * magnitude

    def adjust_positions(self, atoms, positions):
        pass

    def todict(self):
        return {
            "name": "DoubleHookean",
            "kwargs": {
                "indices": self.indices,
                "bondlength": self.bondlength,
                "thr_min": self.thr_min,
                "thr_max": self.thr_max,
                "k_spring": self.k_spring,
            }
        }

# -------------------------------------------------------------------------------------
# SET DOUBLEHOOKEANS
# -------------------------------------------------------------------------------------

def set_doublehookeans(
    atoms,
    bonds_TS,
    indices_ads,
    thr_min=0.85,
    thr_max=1.50,
    k_spring=50.,
):
    
    cutoffs = natural_cutoffs(atoms=atoms, mult=1.)
    for bond in bonds_TS:
        a1, a2 = bond[:2]
        atoms.constraints.append(
            DoubleHookean(
                indices=[a1, a2],
                bondlength=cutoffs[a1]+cutoffs[a2],
                thr_min=thr_min,
                thr_max=thr_max,
                k_spring=k_spring,
            )
        )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
