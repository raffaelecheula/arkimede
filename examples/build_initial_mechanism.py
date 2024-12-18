# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.gui.gui import GUI
from ase.db import connect
from arkimede.utilities import write_atoms_to_db, get_db_metadata
from arkimede.workflow.reaction_mechanism import MechanismCalculator

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Name of ase database to store unrelaxed structures.
    db_init_name = "databases/init.db"
    db_init_append = False

    # Arkimede adsorption parameters.
    auto_construct = True
    delta_ncoord = 1

     # Sites and reactions parameters.
    range_edges_fun = lambda lc: [1e-2, lc + 1e-2]
    displ_max_fun = lambda lc: lc * np.sqrt(2.0) / 4.0 + 1e-3
    range_coads_fun = lambda lc: [lc * 0.4 + 1e-3, lc * 0.8 + 1e-3]

    # Show initial structures.
    show_sites = False
    show_gas_init = False
    show_clean_init = False
    show_ads_init = False
    show_neb_init = False

    # ---------------------------------------------------------------------------------
    # REACTION MECHANISM PARAMETERS
    # ---------------------------------------------------------------------------------

    # Surface structures parameters.
    bulk_structure = "fcc"
    element_bulk_list = [
        "Rh",
        #"Cu",
        #"Co",
        #"Ni",
        #"Fe",
        #"Mn",
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
        #"511",
        #"100-100",
        #"100-110",
        #"100-111",
        #"110-111",
        #"111-111",
    ]

    # Adsorbates names.
    adsorbate_list = [
        "CO2**",
        "H2O*",
        "CO*",
        "H*",
        "OH*",
        "O*",
        "c-COOH**",
        "t-COOH**",
        #"H2*",
        #"b-HCOOH**"
        
        #"CO2*",
        #"c-COOH*",
        #"t-COOH*",
        #"HCOO*",
        #"HCOO**",
        #"HCO*",
        #"HCO**",
        #"COH*",
        #"C*",
        #"CH*",
        #"CH2*",
        #"CH3*",
        #"H2COO**",
        #"H2COOH**", 
        #"H2CO**",
        #"H3CO*",
        #"HCOH**",
        #"H2COH**",
    ]

    # Reactions names.
    reaction_list = [
        "CO2**→CO*+O*",
        "H2O*→OH*+H*",
        "OH*→O*+H*",
        "t-COOH**→CO2**+H*",
        "c-COOH**→CO*+OH*",
        #"H2*→H*+H*",
        #"b-HCOOH**→HCOO**+H*",
        #"b-HCOOH**→c-COOH**+H*",
        
        #"c-COOH**→COH*+O*",
        #"HCOO**→CO2**+H*",
        #"HCOO**→HCO**+O*",
        #"HCO**→CO*+H*",
        #"HCO**→CH*+O*",
        #"COH*→CO*+H*",
        #"COH*→C*+OH*",
        #"CO*→C*+O*",
        #"CH*→C*+H*",
        #"CH2*→CH*+H*",
        #"CH3*→CH2*+H*",
        #"H3CO*→H2CO*+H*",
        #"H2CO**→HCO**+H*",
        #"H2CO**→CH2**+O*",
        #"HCOH**→COH*+H*",
        #"HCOH**→HCO*+H*",
        #"HCOH**→CH*+OH*",
        #"H2COH**→HCOH**+H*",
        #"H2COH**→H2CO*+H*",
        #"H2COH**→CH2*+OH*",
        #"H2COO**→HCOO**+H*",
        #"H2COO**→H2CO**+O*",
        #"H2COOH**→H2COO**+H*",
        #"H2COOH**→H2CO**+OH*",
    ]

    # ---------------------------------------------------------------------------------
    # STRUCTURES
    # ---------------------------------------------------------------------------------

    # Import functions to produce ase atoms structures.
    from ase_structures import get_atoms_slab, get_atoms_gas

    atoms_gas_tot = []
    atoms_clean_tot = []
    atoms_ads_tot = []
    atoms_neb_tot = []

    # Get the structures of gas species.
    for species in adsorbate_list:
        atoms_gas_tot.append(get_atoms_gas(species=species))

    # Get the atoms structures of clean surfaces and surfaces with adsorbates.
    for element_bulk in element_bulk_list:
        for miller_index in miller_index_list:

            # Get the clean slab.
            atoms_clean = get_atoms_slab(
                bulk_structure=bulk_structure,
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
                get_atoms_gas_fun=get_atoms_gas,
                range_edges=range_edges,
                displ_max_reaction=displ_max_reaction,
                range_coadsorption=range_coadsorption,
                auto_construct=auto_construct,
            )
            if show_sites is True:
                mech.builder.plot()

            atoms_ads_tot += mech.get_adsorbates(adsorbate_list=adsorbate_list)
            atoms_neb_tot += mech.get_neb_images(reaction_list=reaction_list)

    atoms_allnebs_tot = [
        atoms for atoms_neb in atoms_neb_tot for atoms in atoms_neb
    ]
    
    # Initialize ase database.
    if len(os.path.split(db_init_name)[0]) > 0:
        os.makedirs(os.path.split(db_init_name)[0], exist_ok=True)
    db_init = connect(name=db_init_name, append=db_init_append)
    
    # Store structures names as metadata in the database.
    db_init.metadata = get_db_metadata(
        atoms_gas_tot=atoms_gas_tot,
        atoms_clean_tot=atoms_clean_tot,
        atoms_ads_tot=atoms_ads_tot,
        atoms_neb_tot=atoms_neb_tot,
    )
    
    # Store initial structures into ade database.
    for atoms in atoms_gas_tot+atoms_clean_tot+atoms_ads_tot+atoms_allnebs_tot:
        atoms.info["calculation"] = "initial"
        atoms.info["converged"] = False
        write_atoms_to_db(atoms=atoms, db_ase=db_init)
    
    print(f"Number of adsorbates: {len(atoms_ads_tot):.0f}")
    print(f"Number of reactions:  {len(atoms_neb_tot):.0f}")

    # Visualize initial structures.
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

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
