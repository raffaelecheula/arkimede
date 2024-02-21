# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import numpy as np
import copy as cp
from ase.gui.gui import GUI
from arkimede.workflow.reaction_workflow import MechanismCalculator

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():

    # Run the calculations.
    run_calculations = True

    # Arkimede adsorption parameters.
    auto_construct = True
    delta_ncoord = 1

    # Sites and reactions parameters.
    range_edges_fun = lambda lc: [1e-2, lc + 1e-2]
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
    # DATASET TASK
    # -----------------------------------------------------------------------------

    # Surface structures parameters.
    bulk_structure = "fcc"
    element_bulk_list = [
        "Rh",
        "Cu",
        "Pd",
        "Co",
        "Ni",
        "Au",
    ]
    miller_index_list = [
        "100",
        #"110",
        "111",
        
        #"311c",
        #"110c",
        #"210",
        #"211",
        #"221",
        #"310",
        #"320",
        #"311",
        #"321",
        #"331",
    ]

    # Adsorbates names.
    adsorbate_list = [
        #"CO2**",
        #"H2O*",
        #"CO*",
        #"H*",
        #"OH*",
        #"O*",
        "c-COOH**",
        "t-COOH**",
        
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
        #"CO2**→CO*+O*",
        #"H2O*→OH*+H*",
        #"OH*→O*+H*",
        "t-COOH**→CO2**+H*",
        "c-COOH**→CO*+OH*",
        
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

    # -----------------------------------------------------------------------------
    # STRUCTURES
    # -----------------------------------------------------------------------------

    # Import functions to produce ase atoms structures.
    from ase_structures import get_atoms_slab, get_atoms_gas

    atoms_clean_tot = []
    atoms_ads_tot = []
    atoms_neb_tot = []

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

    atoms_allnebs_tot = [atoms for atoms_neb in atoms_neb_tot for atoms in atoms_neb]

    print(f"Number of adsorbates: {len(atoms_ads_tot):.0f}")
    print(f"Number of reactions:  {len(atoms_allnebs_tot)/2:.0f}")

    if show_clean_init is True and len(atoms_clean_tot) > 0:
        gui = GUI(atoms_clean_tot)
        gui.run()

    if show_ads_init is True and len(atoms_ads_tot) > 0:
        gui = GUI(atoms_ads_tot)
        gui.run()

    if show_neb_init is True and len(atoms_allnebs_tot) > 0:
        gui = GUI(atoms_allnebs_tot)
        gui.run()

    # Run the calculations.
    if run_calculations is True:
        run_calculations_reaction_mechanism(
            atoms_clean_tot=atoms_clean_tot,
            atoms_ads_tot=atoms_ads_tot,
            atoms_neb_tot=atoms_neb_tot,
        )

# -----------------------------------------------------------------------------
# RUN CALCULATIONS
# -----------------------------------------------------------------------------

def run_calculations_reaction_mechanism(
    atoms_clean_tot,
    atoms_ads_tot,
    atoms_neb_tot,
):

    from ase.db import connect
    from ocpmodels.common.relaxation.ase_utils import OCPCalculator
    from ocp_utils import get_checkpoint_path, checkpoints_basedir
    from arkimede.workflow.utilities import (
        get_idpp_interpolated_images,
        get_atoms_TS_from_images_neb,
        get_new_IS_and_FS_from_images,
        get_new_IS_and_FS_from_atoms_TS,
    )
    from arkimede.workflow.calculations import (
        run_relax_calculation,
        run_neb_calculation,
        run_dimer_calculation,
        run_climbbonds_calculation,
        run_vibrations_calculation,
    )
    from arkimede.workflow.utilities import (
        read_atoms_from_db,
        write_atoms_to_db,
        update_clean_slab_positions,
        print_results_calculation,
        get_images_neb_from_atoms_TS,
        update_info_TS,
        check_same_connectivity,
    )

    # -----------------------------------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------------------------------

    # Save trajectories and write images.
    save_trajs = False
    write_images = False
    basedir_trajs = "calculatons"

    # Name of ase database.
    db_ase_name = "database.db"
    db_ase_append = True

    # Calculations parameters.
    fmax = 0.05
    steps_max_relax = 1000
    steps_max_neb = 10
    steps_max_ts_search = 1000
    n_images_neb = 10
    search_TS = "climbbonds" # dimer | climbbonds | climbfixinter

    # OCPmodels ase calculator.
    checkpoint_key = 'EquiformerV2 (31M) All+MD'
    checkpoint_path = get_checkpoint_path(
        key = checkpoint_key,
        basedir = checkpoints_basedir(),
    )
    calc = OCPCalculator(
        checkpoint_path = checkpoint_path,
        cpu = False,
    )

    # -----------------------------------------------------------------------------
    # CALCULATIONS
    # -----------------------------------------------------------------------------

    # Initialize ase database.
    db_ase = connect(name=db_ase_name, append=db_ase_append)

    # Relax clean slabs.
    for atoms in atoms_clean_tot:
        atoms.info["calculation"] = "relax"
        if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
            run_relax_calculation(
                atoms=atoms,
                calc=calc,
                fmax=fmax,
                steps_max=steps_max_relax,
                name='relax',
                directory=os.path.join(basedir_trajs, atoms.info["name"]),
                save_trajs=save_trajs,
                write_images=write_images,
            )
            write_atoms_to_db(atoms=atoms, db_ase=db_ase)
        print_results_calculation(atoms=atoms)

    # Relax slabs with adsorbates.
    for atoms in atoms_ads_tot:
        atoms.info["calculation"] = "relax"
        if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
            update_clean_slab_positions(atoms=atoms, db_ase=db_ase)
            run_relax_calculation(
                atoms=atoms,
                calc=calc,
                fmax=fmax,
                steps_max=steps_max_relax,
                name='relax',
                directory=os.path.join(basedir_trajs, atoms.info["name"]),
                save_trajs=save_trajs,
                write_images=write_images,
            )
            # Calculate vibrations.
            if atoms.info["converged"] and "vib_energies" not in atoms.info:
                run_vibrations_calculation(atoms=atoms, calc=calc)
            write_atoms_to_db(atoms=atoms, db_ase=db_ase)
        
        print_results_calculation(atoms=atoms)

    # Calculate transition states of the reactions.
    for idx, atoms_IS_FS in enumerate(atoms_neb_tot):

        # Initialize the atoms TS as copy of the initial state and update its name.
        atoms_IS, atoms_FS = atoms_IS_FS
        atoms_TS = atoms_IS.copy()
        update_info_TS(atoms_TS=atoms_TS, atoms_IS=atoms_IS, atoms_FS=atoms_FS)
        bonds_TS = atoms_FS.info["bonds_TS"]

        # Directory to store output trajectories and images.
        directory = os.path.join(basedir_trajs, atoms_TS.info["name"])

        # Relax initial and final states.
        for ss, atoms in {"IS_0": atoms_IS, "FS_0": atoms_FS}.items():
            atoms.info["calculation"] = "relax"
            if read_atoms_from_db(atoms=atoms, db_ase=db_ase) is None:
                update_clean_slab_positions(atoms=atoms, db_ase=db_ase)
                run_relax_calculation(
                    atoms=atoms,
                    calc=calc,
                    fmax=fmax,
                    steps_max=steps_max_relax,
                    name=f'relax_{ss}',
                    directory=directory,
                    save_trajs=save_trajs,
                    write_images=write_images,
                )
                write_atoms_to_db(atoms=atoms, db_ase=db_ase)
            print_results_calculation(atoms=atoms)

        # Run the NEB calculation.
        atoms_TS.info["calculation"] = "neb"
        if read_atoms_from_db(atoms=atoms_TS, db_ase=db_ase) is None:
            images = get_idpp_interpolated_images(
                atoms_IS=atoms_IS,
                atoms_FS=atoms_FS,
                n_images=n_images_neb,
                name=None,
            )
            run_neb_calculation(
                images=images,
                calc=calc,
                steps_max=steps_max_neb,
                fmax=fmax,
                k_neb=0.10,
                use_OCPdyNEB=False,
                use_NEBOptimizer=True,
                print_energies=True,
                name='neb',
                directory=directory,
                save_trajs=save_trajs,
                write_images=write_images,
            )
            get_atoms_TS_from_images_neb(images=images, atoms_TS=atoms_TS)

            # Check if relaxing the TS gives the initial IS and FS.
            #if atoms_TS.info["converged"] is False:
            #    atoms_IS_1, atoms_FS_1 = get_new_IS_and_FS_from_images(images=images)
            #    for ss, atoms in {"IS_1": atoms_IS_1, "FS_1": atoms_FS_1}.items():
            #        run_relax_calculation(
            #            atoms=atoms,
            #            calc=calc,
            #            fmax=fmax,
            #            steps_max=steps_max_relax,
            #            name=f"relax_{ss}",
            #            directory=directory,
            #            save_trajs=save_trajs,
            #            write_images=write_images,
            #        )
            #    check_IS = check_same_connectivity(atoms_IS, atoms_IS_1)
            #    check_FS = check_same_connectivity(atoms_FS, atoms_FS_1)
            #    if bool(check_IS*check_FS) is False:
            #        atoms_TS.info["modified"] = True
            write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
        print_results_calculation(atoms=atoms_TS)

        # If NEB is not converged, do a TS search calculation.
        if atoms_TS.info["converged"] is False and atoms_TS.info["modified"] is False:

            atoms_TS.info["calculation"] = search_TS
            if read_atoms_from_db(atoms=atoms_TS, db_ase=db_ase) is None:
                # Dimer calculation.
                if search_TS == "dimer":
                    run_dimer_calculation(
                        atoms=atoms_TS,
                        calc=calc,
                        steps_max=steps_max_ts_search,
                        bonds_TS=bonds_TS,
                        vector=atoms_TS.info["vector"].copy(),
                        fmax=fmax,
                        name="dimer",
                        directory=directory,
                        save_trajs=save_trajs,
                        write_images=write_images,
                    )
                # Climbbonds calculaton.
                elif search_TS == "climbbonds":
                    run_climbbonds_calculation(
                        atoms=atoms_TS,
                        calc=calc,
                        steps_max=steps_max_ts_search,
                        bonds_TS=bonds_TS,
                        fmax=fmax,
                        name="climbbonds",
                        directory=directory,
                        save_trajs=save_trajs,
                        write_images=write_images,
                    )

                # Check if relaxing the TS gives the initial IS and FS.
                #atoms_IS_2, atoms_FS_2 = get_new_IS_and_FS_from_atoms_TS(
                #    atoms_TS=atoms_TS,
                #    bonds_TS=bonds_TS,
                #)
                #for ss, atoms in {"IS_2": atoms_IS_2, "FS_2": atoms_FS_2}.items():
                #    run_relax_calculation(
                #        atoms=atoms,
                #        calc=calc,
                #        fmax=fmax,
                #        steps_max=steps_max_relax,
                #        name=f'relax_{ss}_2',
                #        directory=directory,
                #        save_trajs=save_trajs,
                #        write_images=write_images,
                #    )
                #check_IS = check_same_connectivity(atoms_IS, atoms_IS_2)
                #check_FS = check_same_connectivity(atoms_FS, atoms_FS_2)
                #if bool(check_IS*check_FS) is False:
                #    atoms_TS.info["modified"] = True

                write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
            print_results_calculation(atoms=atoms_TS)

        # If the TS is found, calculate vibrations.
        if atoms_TS.info["converged"] and atoms_TS.info["modified"] is False:
            if "vib_energies" not in atoms_TS.info:
                run_vibrations_calculation(atoms=atoms_TS, calc=calc)
                write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)

# -----------------------------------------------------------------------------
# IF NAME MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
