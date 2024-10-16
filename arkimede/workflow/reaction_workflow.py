# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np

# -------------------------------------------------------------------------------------
# RUN ASE CALCULATIONS MECHANISM
# -------------------------------------------------------------------------------------

def run_ase_calculations_mechanism(
    atoms_clean_tot,
    atoms_ads_tot,
    atoms_neb_tot,
    calc,
    db_ase,
    fmax=0.05,
    max_steps_relax=500,
    max_steps_neb=300,
    max_steps_ts_search=1000,
    n_images_neb=10,
    search_TS="climbbonds",
    save_trajs=False,
    write_images=False,
    basedir_trajs="calculations",
    update_info_TS_fun=None,
    check_modified_relax=False,
    check_modified_neb=False,
    check_modified_TS_search=False,
    keys_match=None,
    keys_store=None,
    kwargs_relax={},
    kwargs_neb={},
    kwargs_ts_search={},
    kwargs_read_db={},
    kwargs_write_db={},
):

    from arkimede.workflow.calculations import (
        run_relax_calculation,
        run_neb_calculation,
        run_calculation,
        run_vibrations_calculation,
    )
    from arkimede.utilities import (
        get_idpp_interpolated_images,
        get_atoms_TS_from_images_neb,
        get_new_IS_and_FS_from_images,
        get_new_IS_and_FS_from_atoms_TS,
        read_atoms_from_db,
        write_atoms_to_db,
        update_clean_slab_positions,
        print_results_calculation,
        get_images_neb_from_atoms_TS,
        update_info_TS_combo,
        check_same_connectivity,
    )

    # Update kwargs.
    kwargs_read_db.update({"db_ase": db_ase})
    kwargs_write_db.update({"db_ase": db_ase})
    if keys_match is not None:
        kwargs_read_db.update({"keys_match": keys_match})
        kwargs_write_db.update({"keys_match": keys_match})
    if keys_store is not None:
        kwargs_write_db.update({"keys_store": keys_store})

    if update_info_TS_fun is None:
        update_info_TS_fun = update_info_TS_combo

    # Relax clean slabs.
    for atoms in atoms_clean_tot:
        atoms.info["calculation"] = "relax"
        if read_atoms_from_db(atoms=atoms, **kwargs_read_db) is None:
            run_relax_calculation(
                atoms=atoms,
                calc=calc,
                fmax=fmax,
                max_steps=max_steps_relax,
                directory=os.path.join(basedir_trajs, atoms.info["name"]),
                save_trajs=save_trajs,
                write_images=write_images,
                **kwargs_relax,
            )
            write_atoms_to_db(atoms=atoms, **kwargs_write_db)
        print_results_calculation(atoms=atoms)

    # Relax slabs with adsorbates.
    for atoms in atoms_ads_tot:
        atoms.info["calculation"] = "relax"
        if read_atoms_from_db(atoms=atoms, **kwargs_read_db) is None:
            update_clean_slab_positions(atoms=atoms, db_ase=db_ase)
            atoms_copy = atoms.copy()
            run_relax_calculation(
                atoms=atoms,
                calc=calc,
                fmax=fmax,
                max_steps=max_steps_relax,
                directory=os.path.join(basedir_trajs, atoms.info["name"]),
                save_trajs=save_trajs,
                write_images=write_images,
                **kwargs_relax,
            )
            if check_modified_relax is True:
                if check_same_connectivity(atoms_copy, atoms) is False:
                    atoms.info["modified"] = True
            # Calculate vibrations.
            if atoms.info["converged"] and "vib_energies" not in atoms.info:
                run_vibrations_calculation(
                    atoms=atoms,
                    calc=calc,
                    remove_cache=True,
                    save_trajs=False,
                )
            write_atoms_to_db(atoms=atoms, **kwargs_write_db)
        print_results_calculation(atoms=atoms)

    # Calculate transition states of the reactions.
    for idx, atoms_IS_FS in enumerate(atoms_neb_tot):

        # Initialize the atoms TS as copy of the initial state and update its name.
        atoms_IS, atoms_FS = atoms_IS_FS
        atoms_TS = atoms_IS.copy()
        update_info_TS_fun(atoms_TS=atoms_TS, atoms_IS=atoms_IS, atoms_FS=atoms_FS)
        bonds_TS = atoms_TS.info["bonds_TS"]

        # Directory to store output trajectories and images.
        directory = os.path.join(basedir_trajs, atoms_TS.info["name"])

        # Relax initial and final states.
        for ss, atoms in {"IS_0": atoms_IS, "FS_0": atoms_FS}.items():
            atoms.info["calculation"] = "relax"
            if read_atoms_from_db(atoms=atoms, **kwargs_read_db) is None:
                update_clean_slab_positions(atoms=atoms, db_ase=db_ase)
                run_relax_calculation(
                    atoms=atoms,
                    calc=calc,
                    fmax=fmax,
                    max_steps=max_steps_relax,
                    label=f'relax_{ss}',
                    directory=directory,
                    save_trajs=save_trajs,
                    write_images=write_images,
                    **kwargs_relax,
                )
                write_atoms_to_db(atoms=atoms, **kwargs_write_db)
            print_results_calculation(atoms=atoms)

        # Run the NEB calculation.
        atoms_TS.info["calculation"] = "neb"
        if read_atoms_from_db(atoms=atoms_TS, **kwargs_read_db) is None:
            images = get_idpp_interpolated_images(
                atoms_IS=atoms_IS,
                atoms_FS=atoms_FS,
                n_images=n_images_neb,
                save_trajs=False,
            )
            run_neb_calculation(
                images=images,
                calc=calc,
                max_steps=max_steps_neb,
                fmax=fmax,
                directory=directory,
                save_trajs=save_trajs,
                write_images=write_images,
                **kwargs_neb,
            )
            get_atoms_TS_from_images_neb(images=images, atoms_TS=atoms_TS)

            # Check if relaxing the TS gives the initial IS and FS.
            if check_modified_neb is True:
                atoms_IS_1, atoms_FS_1 = get_new_IS_and_FS_from_images(images=images)
                for ss, atoms in {"IS": atoms_IS_1, "FS": atoms_FS_1}.items():
                    run_relax_calculation(
                        atoms=atoms,
                        calc=calc,
                        fmax=fmax,
                        max_steps=max_steps_relax,
                        label=f"relax_{ss}_1",
                        directory=directory,
                        save_trajs=save_trajs,
                        write_images=write_images,
                        **kwargs_relax,
                    )
                check_IS = check_same_connectivity(atoms_IS, atoms_IS_1)
                check_FS = check_same_connectivity(atoms_FS, atoms_FS_1)
                if bool(check_IS*check_FS) is False:
                    atoms_TS.info["modified"] = True
            
            write_atoms_to_db(atoms=atoms_TS, **kwargs_write_db)
        print_results_calculation(atoms=atoms_TS)

        # If NEB is not converged, do a TS search calculation.
        if atoms_TS.info["converged"] is False and atoms_TS.info["modified"] is False:

            atoms_TS.info["calculation"] = search_TS
            if read_atoms_from_db(atoms=atoms_TS, **kwargs_read_db) is None:
                run_calculation(
                    atoms=atoms_TS,
                    calc=calc,
                    calculation=search_TS,
                    max_steps=max_steps_ts_search,
                    bonds_TS=bonds_TS,
                    fmax=fmax,
                    directory=directory,
                    save_trajs=save_trajs,
                    write_images=write_images,
                    **kwargs_ts_search,
                )

                # Check if relaxing the TS gives the initial IS and FS.
                if check_modified_TS_search is True:
                    atoms_IS_2, atoms_FS_2 = get_new_IS_and_FS_from_atoms_TS(
                        atoms_TS=atoms_TS,
                        bonds_TS=bonds_TS,
                    )
                    for ss, atoms in {"IS": atoms_IS_2, "FS": atoms_FS_2}.items():
                        run_relax_calculation(
                            atoms=atoms,
                            calc=calc,
                            fmax=fmax,
                            max_steps=max_steps_relax,
                            label=f'relax_{ss}_2',
                            directory=directory,
                            save_trajs=save_trajs,
                            write_images=write_images,
                            **kwargs_relax,
                        )
                    check_IS = check_same_connectivity(atoms_IS, atoms_IS_2)
                    check_FS = check_same_connectivity(atoms_FS, atoms_FS_2)
                    if bool(check_IS*check_FS) is False:
                        atoms_TS.info["modified"] = True

                write_atoms_to_db(atoms=atoms_TS, **kwargs_write_db)
            print_results_calculation(atoms=atoms_TS)

        # If the TS is found, calculate vibrations.
        if atoms_TS.info["converged"] and atoms_TS.info["modified"] is False:
            if "vib_energies" not in atoms_TS.info:
                run_vibrations_calculation(
                    atoms=atoms_TS,
                    calc=calc,
                    remove_cache=True,
                    save_trajs=False,
                )
                write_atoms_to_db(atoms=atoms_TS, **kwargs_write_db)

# -------------------------------------------------------------------------------------
# RUN DFT CALCULATIONS K8S
# -------------------------------------------------------------------------------------

def run_dft_calculations_k8s(
    atoms_list,
    template_yaml,
    namespace,
    basedir_dft_calc,
    filename_out,
    db_dft,
    write_input_fun,
    read_output_fun,
    check_finished_fun,
    job_queued_fun,
    command=None,
    cpu_req=8,
    cpu_lim=12,
    mem_req='8Gi',
    mem_lim='24Gi',
    write_all_to_db=False,
    keys_match=None,
    keys_store=None,
    kwargs_read_db={},
    kwargs_write_db={},
    **kwargs,
):
    
    from arkimede.workflow.dft_calculations import submit_k8s
    from arkimede.utilities import (
        read_atoms_from_db,
        write_atoms_to_db,
        print_results_calculation,
    )
    
    # Update kwargs.
    kwargs_read_db.update({"db_ase": db_dft})
    kwargs_write_db.update({"db_ase": db_dft})
    if keys_match is not None:
        kwargs_read_db.update({"keys_match": keys_match})
        kwargs_write_db.update({"keys_match": keys_match})
    if keys_store is not None:
        kwargs_write_db.update({"keys_store": keys_match})
    
    # Read the template yaml file.
    with open(template_yaml, 'r') as fileobj:
        params_k8s = yaml.safe_load(fileobj)
    
    # Run the dft calculations.
    cwd = os.getcwd()
    for atoms in atoms_list:
        atoms.info["converged"] = False
        if read_atoms_from_db(atoms=atoms, **kwargs_read_db) is None:
            directory = os.path.join(basedir_dft_calc, atoms.info["name"])
            os.makedirs(directory, exist_ok=True)
            os.chdir(directory)
            if os.path.isfile(filename_out) and check_finished_fun(filename_out):
                atoms_all = read_output_fun(atoms=atoms, **kwargs)
                if atoms.info["converged"] is True:
                    if isinstance(atoms_all, list) and write_all_to_db is True:
                        for atoms_ii in atoms_all:
                            atoms_ii.set_tags(np.ones(len(atoms_ii)))
                            write_atoms_to_db(atoms=atoms_ii, **kwargs_write_db)
                    else:
                        atoms.set_tags(np.ones(len(atoms)))
                        write_atoms_to_db(atoms=atoms, **kwargs_write_db)
            elif job_queued_fun(atoms.info["name"]) is False:
                write_input_fun(atoms)
                submit_k8s(
                    params_k8s=params_k8s,
                    command=command,
                    name=atoms.info["name"],
                    dirpath=os.path.join(cwd, directory),
                    namespace=namespace,
                    cpu_req=cpu_req,
                    cpu_lim=cpu_lim,
                    mem_req=mem_req,
                    mem_lim=mem_lim,
                )
            os.chdir(cwd)
        print_results_calculation(atoms=atoms)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
