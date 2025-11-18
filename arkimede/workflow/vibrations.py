# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.vibrations import Vibrations as VibrationsOriginal

# -------------------------------------------------------------------------------------
# VIBRATIONS
# -------------------------------------------------------------------------------------

class Vibrations(VibrationsOriginal):
    """
    Custom Vibrations class that stores also energies.
    """
    def calculate(self, atoms, disp):
        results = {}
        results["energy"] = float(self.calc.get_potential_energy(atoms))
        results["forces"] = self.calc.get_forces(atoms)
        if self.ir:
            results["dipole"] = self.calc.get_dipole_moment(atoms)
        return results

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------