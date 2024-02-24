# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocp_utils import get_checkpoint_path, checkpoints_basedir
from arkimede.workflow.utilities import get_atoms_list_from_db

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():

    # Name of ase database containing the unrelaxed structures to read.
    db_init_name = "database_init.db"

    # Select atoms from the ase database.
    selection = ""

    # Name of ase database to store the results of the calculations.
    db_ase_name = "database_ocp.db"
    db_ase_append = True

    # Calculations parameters.
    fmax = 0.05
    steps_max_relax = 1000
    steps_max_neb = 10
    steps_max_ts_search = 1000
    n_images_neb = 10
    search_TS = "climbbonds" # dimer | climbbonds | climbfixinter

    # Save trajectories and write images.
    save_trajs = False
    write_images = False
    basedir_trajs = "calculations_ocp"

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
    # RUN ASE CALCULATIONS
    # -----------------------------------------------------------------------------

    # Initialize ase init database.
    db_init = connect(name=db_init_name)

    # Get lists of atoms from database.
    atoms_clean_tot = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="clean",
    )
    atoms_ads_tot = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="adsorbates",
    )
    atoms_neb_tot = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="reactions",
    )
    
    # Initialize ase dft database.
    db_ase = connect(name=db_ase_name, append=db_ase_append)

    run_ase_calculations_reaction_mechanism(
        atoms_clean_tot=atoms_clean_tot,
        atoms_ads_tot=atoms_ads_tot,
        atoms_neb_tot=atoms_neb_tot,
        db_ase=db_ase,
        fmax=fmax,
        steps_max_relax=steps_max_relax,
        steps_max_neb=steps_max_neb,
        steps_max_ts_search=steps_max_ts_search,
        n_images_neb=n_images_neb,
        search_TS=search_TS,
        save_trajs=save_trajs,
        write_images=write_images,
        basedir_trajs=basedir_trajs,
    )

# -----------------------------------------------------------------------------
# RUN ASE CALCULATIONS REACTION MECHANISM
# -----------------------------------------------------------------------------

def run_ase_calculations_reaction_mechanism(
    atoms_clean_tot,
    atoms_ads_tot,
    atoms_neb_tot,
    calc,
    db_ase,
    fmax=0.05,
    steps_max_relax=500,
    steps_max_neb=300,
    steps_max_ts_search=1000,
    n_images_neb=10,
    search_TS="climbbonds",
    save_trajs=False,
    write_images=False,
    basedir_trajs="calculations",
):

    from arkimede.workflow.calculations import (
        run_relax_calculation,
        run_neb_calculation,
        run_dimer_calculation,
        run_climbbonds_calculation,
        run_vibrations_calculation,
    )
    from arkimede.workflow.utilities import (
        get_idpp_interpolated_images,
        get_atoms_TS_from_images_neb,
        get_new_IS_and_FS_from_images,
        get_new_IS_and_FS_from_atoms_TS,
        read_atoms_from_db,
        write_atoms_to_db,
        update_clean_slab_positions,
        print_results_calculation,
        get_images_neb_from_atoms_TS,
        update_info_TS,
        check_same_connectivity,
    )

    # -----------------------------------------------------------------------------
    # CALCULATIONS
    # -----------------------------------------------------------------------------

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
