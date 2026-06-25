# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator

from arkimede.utilities import filter_results

# -------------------------------------------------------------------------------------
# INVERSION SYMMETRY ATOMS
# -------------------------------------------------------------------------------------

class InversionSymmetryAtoms(Atoms):
    
    def __init__(
        self,
        atoms_full: Atoms,
        mapping_dict: dict = None,
        **kwargs,
    ):
        # Get mapping of inversion symmetry indices.
        self.atoms_full = atoms_full
        if mapping_dict is None:
            self.mapping_dict = self.get_indices_mapping()
        else:
            self.mapping_dict = mapping_dict
        self.indices_active = [
            ii for ii in range(len(atoms_full)) if ii not in self.mapping_dict
        ]
        # Initialize ASE Atoms with only the active indices.
        super().__init__(
            symbols=np.array(atoms_full.symbols)[self.indices_active],
            positions=np.array(atoms_full.get_positions())[self.indices_active],
            cell=np.array(atoms_full.cell),
            pbc=atoms_full.get_pbc(),
            **kwargs,
        )
        # Update Atoms constraints.
        self.update_constraints()

    @property
    def calc(self):
        return self.atoms_full.calc

    @calc.setter
    def calc(self, calc):
        if hasattr(self, "atoms_full"):
            self.atoms_full.calc = calc

    def get_indices_mapping(self, tol: float = 1e-6):
        scaled = self.atoms_full.get_scaled_positions(wrap=True)
        mapping_dict = {}
        for ii, pos in enumerate(scaled):
            pos_mapped = (-pos) % 1.0
            for jj in range(ii + 1, len(scaled)):
                diff = scaled[jj] - pos_mapped
                diff -= np.round(diff)
                if np.all(np.abs(diff) < tol):
                    mapping_dict[ii] = jj
                    break
        return mapping_dict

    def update_constraints(self):
        constraints = []
        for cc in deepcopy(self.atoms_full.constraints):
            try:
                cc.index_shuffle(self.atoms_full, self.indices_active)
            except:
                continue
            constraints.append(cc)
        self.set_constraint(constraints)

    def update_positions_full(self):
        scaled = self.get_scaled_positions()
        scaled_full = np.zeros((len(self.atoms_full), 3))
        scaled_full[self.indices_active] = scaled
        for ii, jj in self.mapping_dict.items():
            scaled_full[ii] = np.ones(3) - scaled_full[jj]
        self.atoms_full.set_scaled_positions(scaled_full)

    def get_forces(self, **kwargs):
        self.update_positions_full()
        return self.atoms_full.get_forces(**kwargs)[self.indices_active]

    def get_potential_energy(self, **kwargs):
        self.update_positions_full()
        return self.atoms_full.get_potential_energy(**kwargs)

    def set_positions(self, positions: np.ndarray, **kwargs):
        super().set_positions(positions, **kwargs)
        self.update_positions_full()

    def copy(self):
        return type(self)(
            atoms_full=self.atoms_full.copy(),
            info=self.info.copy(),
            mapping_dict=self.mapping_dict.copy(),
        )

    def __getitem__(self, i):
        return Atoms(
            symbols=np.array(self.symbols)[i],
            positions=np.array(self.get_positions())[i],
            cell=np.array(self.cell),
            pbc=self.get_pbc(),
        )

# -------------------------------------------------------------------------------------
# INVERSION SYMMETRY TRAJECTORY
# -------------------------------------------------------------------------------------

class InversionSymmetryTrajectory(TrajectoryWriter):
    
    def write(self, atoms: Atoms = None, **kwargs):
        if atoms is None:
            atoms = self.atoms
        if hasattr(atoms, "atoms_full"):
            results = filter_results(atoms.calc.results)
            atoms_full = atoms.atoms_full.copy()
            atoms_full.info = atoms.info.copy()
            atoms_full.calc = SinglePointCalculator(atoms=atoms_full, **results)
            atoms = atoms_full
        super().write(atoms=atoms, **kwargs)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------