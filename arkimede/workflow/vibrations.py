# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import shutil
from ase import Atoms
from ase.parallel import world
from ase.vibrations import Vibrations as VibrationsOriginal

# -------------------------------------------------------------------------------------
# VIBRATIONS
# -------------------------------------------------------------------------------------

class Vibrations(VibrationsOriginal):
    """
    Custom Vibrations class that stores energies and can write to trajectory files.
    """
    def __init__(
        self,
        trajectory: str = None,
        append_trajectory: bool = False,
        master: bool = None,
        comm: object = world,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        if isinstance(trajectory, str):
            from ase.io.trajectory import Trajectory
            mode = "a" if append_trajectory else "w"
            trajectory = Trajectory(trajectory, mode=mode, master=master, comm=comm)
        self.trajectory = trajectory

    def calculate(self, atoms: Atoms, disp: str):
        results = {}
        results["energy"] = float(self.calc.get_potential_energy(atoms))
        results["forces"] = self.calc.get_forces(atoms)
        if self.ir:
            results["dipole"] = self.calc.get_dipole_moment(atoms)
        if self.trajectory is not None:
            self.trajectory.write(atoms=atoms, disp=disp)
        return results
    
    def remove_cache(self):
        self.clean()
        if os.path.isdir(self.cache.directory):
            shutil.rmtree(self.cache.directory)
    
# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------