# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.io import read
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.calculators.vasp.vasp import Vasp
from ocdata.utils.vasp import calculate_surface_k_points

# -----------------------------------------------------------------------------
# ASE VASP VIBRATIONS
# -----------------------------------------------------------------------------

# Phase of the structure.
phase = "gas" # gas | surface

# Temperature and pressure.
temperature = 500 # [K]
pressure = 1e5 # [Pa]

# Properties of molecule.
if phase == "gas":
    geometry = "nonlinear" # monatomic | linear | nonlinear
    symmetrynumber = 1
    spin = None

# Read atoms object.
atoms = read('atoms.traj')
atoms.pbc = True
indices = list([aa.index for aa in atoms if aa.symbol in ("C", "O", "H")])

# Vasp parameters.
vasp_flags = {
    "ibrion": -1,
    "nsw": 0,
    "isif": 0,
    "isym": 0,
    "lreal": "Auto",
    "ediffg": 0.0,
    "symprec": 1e-10,
    "encut": 350.0,
    "ncore": 2,
    "lcharg": False,
    "lwave": False,
    "gga": "RP",
    "pp": "PBE",
    "xc": "PBE",
}

# Setup vasp calculator.
calc = Vasp(command=None, **vasp_flags)
if phase == "surface":
    kpts = calculate_surface_k_points(atoms)
    calc.set(kpts=kpts)
atoms.calc = calc

# Setup vibrations calculation.
vib = Vibrations(atoms=atoms, indices=indices, delta=0.01, nfree=2)
    
# Run the calculation and get normal frequencies.
vib.clean(empty_files=True)
vib.run()

# Write frequencies to file.
logfile = "vib.log"
open(logfile, "w").close()
vib.summary(method="standard", log=logfile)

# Calculate Helmholtz (Gibbs) free energy.
if phase == "gas":
    print(f"\IdealGasThermo calculation at T = {temperature} K")
    thermo = IdealGasThermo(
        vib_energies=vib.get_energies(),
        geometry=geometry,
        atoms=atoms,
        symmetrynumber=symmetrynumber,
        spin=spin,
    )
    thermo.get_gibbs_energy(
        temperature=temperature,
        pressure=pressure,
        verbose=True,
    )
elif phase == "surface":
    print(f"\nHarmonicThermo calculation at T = {temperature} K")
    vib_energies = []
    for ii, vib_energy in enumerate(vib.get_energies()):
        if np.iscomplex(vib_energy):
            print(f"Mode {ii:02d} has imaginary frequency")
        else:
            vib_energies.append(vib_energy)
    thermo = HarmonicThermo(vib_energies=vib_energies)
    thermo.get_helmholtz_energy(
        temperature=temperature,
        verbose=True,
    )

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
