# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.geometry import find_mic
from ase.constraints import FixConstraint
from ase.vibrations import Vibrations

# -------------------------------------------------------------------------------------
# EIGEN FOLLOW
# -------------------------------------------------------------------------------------

class EigenFollow(FixConstraint):
    
    def __init__(
        self,
        indices,
        bonds_TS=None,
        delta=0.01,
        nfree=2,
    ):
        self.indices = indices
        self.bonds_TS = bonds_TS
        self.delta = delta
        self.nfree = nfree
        self.vib = None

    def initialize(self, atoms):
        self.vib = Vibrations(
            atoms=atoms,
            indices=self.indices,
            delta=self.delta,
            nfree=self.nfree,
        )

    def get_eigenvector(self, atoms):
        print("vib")
        if self.vib is None:
            self.initialize(atoms)
        self.vib.clean(empty_files=False)
        self.vib.run()
        vibdata = self.vib.get_vibrations()
        energies, modes = vibdata.get_energies_and_modes(all_atoms=True)
        # TODO: make parallel to bonds_TS
        return modes[0]

    def adjust_forces(self, atoms, forces):
        eigenvector = self.get_eigenvector(atoms)
        forces -= eigenvector # TODO: how do we add the eigenvector to the forces?

    def adjust_positions(self, atoms, forces):
        pass

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        indices = map[self.indices]
        self.indices = indices[(indices != -1).all(1)]
        if len(self.indices) == 0:
            raise IndexError('Constraint not part of slice')

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
