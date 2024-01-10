# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import timeit
import numpy as np
import copy as cp
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocp_utils import get_checkpoint_path, checkpoints_basedir
from ase.gui.gui import GUI
from arkimede.workflow.reaction_workflow import MechanismCalculator

time_start = timeit.default_timer()

# -----------------------------------------------------------------------------
# STRUCTURES PARAMETERS
# -----------------------------------------------------------------------------

# Catkit adsorption parameters.
auto_construct = True
topology_sym = False
delta_ncoord = 1

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
show_ads_init = False
show_neb_init = False

# -----------------------------------------------------------------------------
# CALCULATOR
# -----------------------------------------------------------------------------

db_ase_name = "database_test.db"
db_ase_append = True

fmax = 0.01
steps_max_relax = 300
steps_max_neb = 100
n_images_neb = 10
search_TS = "dimer" # dimer | climbbonds | climbfixinter

checkpoint_path = get_checkpoint_path(
    key = 'GemNet-dT 2M',
    basedir = checkpoints_basedir(),
)
calc = OCPCalculator(
    checkpoint_path = checkpoint_path,
    trainer = 'forces',
    cpu = False,
)

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
from ase_structures import get_atoms_slab, get_atoms_gas

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

        # Get the clean slab (with possible saa).
        atoms_clean = get_atoms_slab(
            element_bulk=element_bulk,
            miller_index=miller_index,
            delta_ncoord=delta_ncoord,
        )
        atoms_clean_tot.append(atoms_clean)

        lattice_const = atoms_clean.info["lattice_const"]
        range_edges = range_edges_fun(lattice_const)
        displ_max_reaction = displ_max_fun(lattice_const)
        range_coadsorption = range_coads_fun(lattice_const)

        mech = MechanismCalculator(
            atoms_clean=atoms_clean,
            calc=calc,
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
# RUN CALCULATIONS
# -----------------------------------------------------------------------------

from ase.db import connect
from ase import Atoms
from arkimede.workflow.utilities import (
    get_idpp_interpolated_images,
    get_atoms_TS_from_images_neb,
    get_new_IS_and_FS,
)
from arkimede.workflow.calculations import (
    run_relax_calculation,
    run_neb_calculation,
    run_dimer_calculation,
    run_climbbonds_calculation,
    run_vibrations_calculation,
)

db_ase = connect(name=db_ase_name, append=db_ase_append)

def write_atoms_to_db(atoms, db_ase):
    """Write atoms to ase database."""
    name = atoms.info["name"]
    calculation = atoms.info["calculation"]
    if db_ase.count(name=name, calculation=calculation) == 0:
        db_ase.write(
            atoms=Atoms(atoms),
            name=name,
            calculation=calculation,
            data=atoms.info,
        )
    elif db_ase.count(name=name, calculation=calculation) == 1:
        row_id = db_ase.get(name=name, calculation=calculation).id
        db_ase.update(
            id=row_id,
            atoms=Atoms(atoms),
            calculation=calculation,
            data=atoms.info,
        )

def read_atoms_from_db(atoms, db_ase):
    """Read atoms from ase database."""
    name = atoms.info["name"]
    calculation = atoms.info["calculation"]
    if db_ase.count(name=name, calculation=calculation) == 1:
        atoms_row = db_ase.get(name=name, calculation=calculation)
        info = atoms_row.data.copy()
        info.update(atoms_row.key_value_pairs)
        atoms.info.update(info)
        atoms.set_positions(atoms_row.positions)
        return atoms
    else:
        return None

def update_clean_slab_positions(atoms, db_ase):
    """Update positions of relaxed clean slab."""
    if "atoms_clean" in atoms.info:
        if db_ase.count(name = atoms.info["atoms_clean"]) == 1:
            atoms_row = db_ase.get(name = atoms.info["atoms_clean"])
            atoms.set_positions(
                np.vstack((atoms_row.positions, atoms[len(atoms_clean):].positions))
            )
    return atoms

def print_results_calculations(atoms):
    print(atoms.info["name"], atoms.info["calculation"], atoms.info["converged"])

"""Relax clean slabs."""
for atoms in atoms_clean_tot:
    atoms.info["calculation"] = "relax"
    if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
        run_relax_calculation(
            atoms=atoms,
            calc=calc,
            fmax=fmax,
            steps_max=steps_max_relax,
            name=None,
        )
        write_atoms_to_db(atoms=atoms, db_ase=db_ase)
    print_results_calculations(atoms=atoms)

"""Relax slabs with adsorbates."""
for atoms in atoms_ads_tot:
    atoms.info["calculation"] = "relax"
    if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
        update_clean_slab_positions(atoms=atoms, db_ase=db_ase)
        run_relax_calculation(
            atoms=atoms,
            calc=calc,
            fmax=fmax,
            steps_max=steps_max_relax,
            name=None,
        )
        write_atoms_to_db(atoms=atoms, db_ase=db_ase)
    print_results_calculations(atoms=atoms)

"""Calculate transition states of the reactions."""
for atoms_IS_FS in atoms_neb_tot:
    
    # Initialize the atoms TS as copy of the initial state and update name.
    atoms_TS = atoms_IS_FS[0].copy()
    atoms_TS.info["name"] = "_".join([
        atoms_IS_FS[0].info["clean_name"],
        "→".join([atoms_IS_FS[0].info["ads_name"], atoms_IS_FS[1].info["ads_name"]]),
        "→".join([atoms_IS_FS[0].info["site_id"], atoms_IS_FS[1].info["site_id"]]),
    ])
    bonds_TS = atoms_IS_FS[1].info["bonds_TS"]

    # Relax initial and final states.
    for atoms in atoms_IS_FS:
        atoms.info["calculation"] = "relax"
        if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
            update_clean_slab_positions(atoms=atoms, db_ase=db_ase)
            run_relax_calculation(
                atoms=atoms,
                calc=calc,
                fmax=fmax,
                steps_max=steps_max_relax,
                name=None,
            )
            write_atoms_to_db(atoms=atoms, db_ase=db_ase)
        print_results_calculations(atoms=atoms)
    
    # Run the neb calculation.
    atoms_TS.info["calculation"] = "neb"
    if read_atoms_from_db(atoms=atoms_TS, db_ase=db_ase) is None:
        images = get_idpp_interpolated_images(
            atoms_IS = atoms_IS_FS[0],
            atoms_FS = atoms_IS_FS[1],
            name = None,
        )
        run_neb_calculation(
            images = images,
            calc = calc,
            steps_max = steps_max_neb,
            fmax = fmax,
            name = None,
        )
        get_atoms_TS_from_images_neb(images=images, atoms_TS=atoms_TS)
        write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
    print_results_calculations(atoms=atoms_TS)
        
    if atoms_TS.info["converged"] is False:
        if search_TS == "dimer":
            atoms_TS.info["calculation"] = "dimer"
            if read_atoms_from_db(atoms=atoms_TS, db_ase=db_ase) is None:
                run_dimer_calculation(
                    atoms = atoms_TS,
                    calc = calc,
                    bonds_TS = bonds_TS,
                    vector = atoms_TS.info["vector"].copy(),
                    fmax = fmax,
                    name = None,
                )
                write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
            print_results_calculations(atoms=atoms_TS)
        elif search_TS == "climbbonds":
            atoms_TS.info["calculation"] = "climbbonds"
            if read_atoms_from_db(atoms=atoms_TS, db_ase=db_ase) is None:
                run_climbbonds_calculation(
                    atoms = atoms_TS,
                    calc = calc,
                    bonds_TS = bonds_TS,
                    name = None,
                )
                write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
            print_results_calculations(atoms=atoms_TS)
    
    #freq = run_vibrations_calculation(atoms=atoms_TS, calc=calc)
    
    #atoms_IS_FS_new = get_new_IS_and_FS(
    #    calc = calc,
    #    atoms_TS = atoms_TS,
    #    bonds_TS = bonds_TS,
    #)
    #
    #for atoms in atoms_IS_FS_new:
    #    if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
    #        run_relax_calculation(
    #            atoms=atoms,
    #            calc=calc,
    #            fmax=fmax,
    #            steps_max=steps_max_relax,
    #            name=None,
    #        )
    #        #write_atoms_to_db(atoms, db_ase, calculation="relax")

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
