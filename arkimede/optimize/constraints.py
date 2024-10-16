# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.constraints import FixConstraint
from ase.geometry import find_mic
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
# DOUBLE HOOKEAN
# -------------------------------------------------------------------------------------


class DoubleHookean(FixConstraint):
    
    def __init__(
        self,
        indices,
        thr_min,
        thr_max,
        k_spring,
    ):
        self.indices = indices
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.k_spring = k_spring

    def get_removed_dof(self, atoms):
        return 0

    def adjust_forces(self, atoms, forces):
        pos1, pos2 = atoms.positions[self.indices]
        difference = find_mic(pos1 - pos2, atoms.cell, atoms.pbc)[0]
        bondlength = np.linalg.norm(difference)
        if bondlength > self.thr_max:
            magnitude = self.k_spring * (bondlength - self.thr_max)
            direction = difference / np.linalg.norm(bondlength)
            forces[self.indices[0]] += direction * magnitude
            forces[self.indices[1]] -= direction * magnitude
        elif bondlength < self.thr_min:
            magnitude = self.k_spring * (self.thr_min - bondlength)
            direction = difference / np.linalg.norm(bondlength)
            forces[self.indices[0]] += direction * magnitude
            forces[self.indices[1]] -= direction * magnitude

    def adjust_positions(self, atoms, forces):
        pass

    def todict(self):
        return {
            "name": "DoubleHookean",
            "kwargs": {
                "indices": self.indices,
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
    indices,
    mult_nlist=1.25,
    mult_thr_min=0.65,
    mult_thr_max=1.25,
    k_spring=10.,
):
    
    cutoffs = natural_cutoffs(atoms=atoms, mult=mult_nlist)
    nlist = NeighborList(
        cutoffs=cutoffs,
        skin=0.3,
        self_interaction=False,
        bothways=False,
        primitive=NewPrimitiveNeighborList,
    )
    nlist.update(atoms)
    for a1, a2 in nlist.get_connectivity_matrix().keys():
        if a1 not in indices or a2 not in indices:
            continue
        atoms.constraints.append(
            DoubleHookean(
                indices=[a1, a2],
                k_spring=k_spring,
                thr_min=(cutoffs[a1]+cutoffs[a2])*mult_thr_min,
                thr_max=(cutoffs[a1]+cutoffs[a2])*mult_thr_max,
            )
        )


# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
