# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import timeit
import numpy as np
import copy as cp
from ase.gui.gui import GUI
from arkimede.workflow.reaction_workflow import MechanismCalculator

time_start = timeit.default_timer()

# -----------------------------------------------------------------------------
# STRUCTURES PARAMETERS
# -----------------------------------------------------------------------------

# Materials parameters.
repetitions_dict = {
    "100": (3, 3),
    "110": (3, 2),
    "111": (3, 3),
    "210": (2, 2),
    "211": (3, 1),
    "221": (3, 1),
    "310": (2, 1),
    "320": (2, 1),
    "311": (3, 2),
    "321": (2, 1),
    "331": (3, 1),
    "110c": (3, 1),
    "311c": (3, 1),
}

# Catkit adsorption parameters.
delta_ncoord = 1
topology_sym = False
auto_construct = True

# Sites names parameters.
range_edges_fun = lambda lc: [1e-2, lc + 1e-2]

# Reactions parameters.
#displ_max_fun = lambda lc: lc * np.sqrt(2.0) / 4.0 + 1e-3
#range_coads_fun = lambda lc: [lc/3 + 1e-3, lc + 1e-3]
displ_max_fun = lambda lc: lc * np.sqrt(2.0) / 4.0 + 1e-3
range_coads_fun = lambda lc: [
    lc * np.sqrt(2.0) / 3.0 + 1e-3,
    lc * np.sqrt(2.0) / 2.0 + 1e-3,
]

# Show initial structures.
show_sites = False
show_gas_init = False
show_clean_init = False
show_ads_init = True
show_neb_init = True

# -----------------------------------------------------------------------------
# DATASET TASK
# -----------------------------------------------------------------------------

# Surface structures parameters.
element_bulk_list = [
    "Rh",
    #'Cu',
    #'Pd',
    #'Co',
    #'Ni',
    #'Au',
]
miller_index_list = [
    "100",
    #"110",
    #"111",
    #"210",
    #"211",
    #"221",
    #"310",
    #"320",
    #"311",
    #"321",
    #"331",
    #"311c",
    #"110c",
]

# Adsorbates names.
ads_name_list = [
    #"CO2[s]",
    "CO2[ss]",
    #"cCOOH[s]",
    #"cCOOH[ss]",
    #"tCOOH[s]",
    #"tCOOH[ss]",
    #"HCOO[s]",
    #"HCOO[ss]",
    #"HCO[s]",
    #"HCO[ss]",
    #"H2O[s]",
    "CO[s]",
    #"H[s]",
    #"OH[s]",
    #"O[s]",
    #"COH[s]",
    #"C[s]",
    #"CH[s]",
    #"CH2[s]",
    #"CH3[s]",
    #"H2COO[ss]",
    #"H2COOH[ss]", 
    #"H2CO[ss]",
    #"H3CO[s]",
    #"HCOH[ss]",
    #"H2COH[ss]",
]

# Reactions names.
reactions_list = [
    "CO2[ss]→CO[s]+O[s]",
    #"cCOOH[s]→CO[s]+OH[s]",
    #"tCOOH[s]→CO2[s]+H[s]",
    #"tCOOH[ss]→CO2[ss]+H[s]",
    #"HCOO[ss]→CO2[ss]+H[s]",
    #"HCOO[ss]→HCO[ss]+O[s]",
    #"H2O[s]→OH[s]+H[s]",
    #"OH[s]→O[s]+H[s]",
    #"cCOOH[ss]→COH[s]+O[s]",
    #"COH[s]→CO[s]+H[s]",
    #"HCO[s]→CO[s]+H[s]",
    #"HCO[ss]→CO[s]+H[s]",
]

# -----------------------------------------------------------------------------
# STRUCTURES
# -----------------------------------------------------------------------------

# Import functions to produce ase atoms structures.
from .ase_structures import get_atoms_slab, get_atoms_gas

atoms_gas_tot = []
atoms_clean_tot = []
atoms_ads_tot = []
atoms_neb_tot = []

# Get the atoms structures of gas molecules in vacuum.
for ads_name in ads_name_list:
    atoms_gas = get_atoms_gas(ads_name=ads_name)
    atoms_gas_tot.append(atoms_gas)

# Get the atoms structures of clean surfaces and surfaces with adsorbates.
for element_bulk in element_bulk_list:
    for miller_index in miller_index_list:

        repetitions = repetitions_dict[miller_index]

        # Get the clean slab (with possible saa).
        atoms_clean = get_atoms_slab(
            element_bulk=element_bulk,
            miller_index=miller_index,
            repetitions=repetitions,
            delta_ncoord=delta_ncoord,
        )
        atoms_clean_tot.append(atoms_clean)

        lattice_const = atoms_clean.info["lattice_const"]
        range_edges = range_edges_fun(lattice_const)
        displ_max_reaction = displ_max_fun(lattice_const)
        range_coadsorption = range_coads_fun(lattice_const)

        mech = MechanismCalculator(
            atoms_clean=atoms_clean,
            get_atoms_gas_fun=get_atoms_gas,
            range_edges=range_edges,
            displ_max_reaction=displ_max_reaction,
            range_coadsorption=range_coadsorption,
            auto_construct=auto_construct,
            topology_sym=topology_sym,
        )
        if show_sites:
            mech.builder.plot()

        atoms_ads_tot += mech.get_adsorbates(ads_name_list=ads_name_list)

        atoms_neb_tot += mech.get_neb_images(reactions_list=reactions_list)

# -----------------------------------------------------------------------------
# SHOW ATOMS
# -----------------------------------------------------------------------------

atoms_allnebs_tot = [atoms for atoms_neb in atoms_neb_tot for atoms in atoms_neb]

print(f"Number of adsorbates:     {len(atoms_ads_tot):.0f}")
print(f"Number of reaction paths: {len(atoms_allnebs_tot)/2:.0f}")

if show_gas_init is True and len(atoms_gas_tot) > 0:
    gui = GUI(atoms_gas_tot)
    gui.run()

if show_clean_init is True and len(atoms_clean_tot) > 0:
    gui = GUI(atoms_clean_tot)
    gui.run()

if show_ads_init is True and len(atoms_ads_tot) > 0:
    gui = GUI(atoms_ads_tot)
    gui.run()

if show_neb_init is True and len(atoms_allnebs_tot) > 0:
    gui = GUI(atoms_allnebs_tot)
    gui.run()

time_stop = timeit.default_timer()
print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
