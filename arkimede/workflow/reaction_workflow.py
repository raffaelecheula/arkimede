# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import re
import yaml
import numpy as np
from arkimede.catkit.reaction_mechanism import MechanismBuilder

# -------------------------------------------------------------------------------------
# MECHANISM CALCULATOR
# -------------------------------------------------------------------------------------

class MechanismCalculator:

    def __init__(
        self,
        atoms_clean,
        get_atoms_gas_fun,
        range_edges,
        displ_max_reaction,
        range_coadsorption,
        auto_construct=True,
    ):
        self.atoms_clean = atoms_clean
        self.get_atoms_gas_fun = get_atoms_gas_fun
        self.range_edges = range_edges
        self.displ_max_reaction = displ_max_reaction
        self.range_coadsorption = range_coadsorption
        self.auto_construct = auto_construct
    
        self.lattice_const = atoms_clean.info["lattice_const"]
        self.surface_name = self.atoms_clean.info["name"]

        # Setup the builder of adsorption data.
        self.builder = MechanismBuilder(
            slab=self.atoms_clean,
            tol=1e-5,
            cutoff=self.lattice_const,
        )

    def get_adsorbates(self, adsorbate_list):
        
        # Get the atoms structures of the surface slab with adsorbates.
        atoms_ads_tot = []
        for species in adsorbate_list:

            adsorbate = self.get_atoms_gas_fun(species=species)

            # Get the slabs with adsorbates.
            atoms_ads_list = self.builder.add_adsorbate(
                adsorbate=adsorbate,
                sites_names=adsorbate.info["sites_names"],
                range_edges=self.range_edges,
                auto_construct=self.auto_construct,
                symmetric_ads=adsorbate.info["symmetric_ads"],
            )

            for atoms_ads in atoms_ads_list:
                atoms_ads.info["surface_name"] = self.surface_name
                atoms_ads.info["species"] = adsorbate.info["species"]
                atoms_ads.info["name"] = "_".join(
                    [
                        atoms_ads.info["surface_name"],
                        atoms_ads.info["species"],
                        atoms_ads.info["site_id"]
                    ]
                )

            atoms_ads_tot += atoms_ads_list

        return atoms_ads_tot

    def get_neb_images(self, reaction_list):
        
        atoms_neb_tot = []
        # Get the images for reaction calculations.
        for reaction in reaction_list:
        
            reactants_str, products_str = reaction.split("→")
            reactants = [
                self.get_atoms_gas_fun(species=name)
                for name in reactants_str.split("+")
            ]
            products = [
                self.get_atoms_gas_fun(species=name)
                for name in products_str.split("+")
            ]
            
            # Dissociation reactions.
            if len(reactants) == 1:

                reactant = reactants[0]
                dissoc_dict = reactant.info["dissoc_dict"]
                sites_names_list = [
                    product.info["sites_names"] for product in products
                ]

                slab_reactants_list = self.builder.add_adsorbate(
                    adsorbate=reactant,
                    sites_names=reactant.info["sites_names"],
                    range_edges=self.range_edges,
                    auto_construct=self.auto_construct,
                    symmetric_ads=reactant.info["symmetric_ads"],
                )
                
                for slab_reactants in slab_reactants_list:
                    slab_reactants.info["surface_name"] = self.surface_name
                    slab_reactants.info["species"] = reactants_str
                    slab_reactants.info["name"] = "_".join(
                        [
                            slab_reactants.info["surface_name"],
                            slab_reactants.info["species"],
                            slab_reactants.info["site_id"]
                        ]
                    )
                
                for slab_reactants in slab_reactants_list:

                    slab_products_list = self.builder.dissociation_reaction(
                        products=products,
                        bond_break=dissoc_dict[products_str]["bond_break"],
                        bonds_surf=dissoc_dict[products_str]["bonds_surf"],
                        slab_reactants=slab_reactants,
                        auto_construct=self.auto_construct,
                        range_edges=self.range_edges,
                        displ_max_reaction=self.displ_max_reaction,
                        range_coadsorption=self.range_coadsorption,
                        sites_names_list=sites_names_list,
                        site_contains_list=None,
                        sort_by_pathlength=False,
                    )

                    for slab_products in slab_products_list:
                        slab_products.info["surface_name"] = self.surface_name
                        slab_products.info["species"] = products_str
                        slab_products.info["name"] = "_".join(
                            [
                                slab_products.info["surface_name"],
                                slab_products.info["species"],
                                slab_products.info["site_id"]
                            ]
                        )
                        atoms_neb_tot.append([slab_reactants, slab_products])

        return atoms_neb_tot

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
    steps_max_relax=500,
    steps_max_neb=300,
    steps_max_ts_search=1000,
    n_images_neb=10,
    search_TS="climbbonds",
    save_trajs=False,
    write_images=False,
    basedir_trajs="calculations",
    check_modified_relax=False,
    check_modified_neb=False,
    check_modified_TS_search=False,
):

    from arkimede.workflow.calculations import (
        run_relax_calculation,
        run_neb_calculation,
        run_TS_search_calculation,
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

    # -------------------------------------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------------------------------------

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
            atoms_copy = atoms.copy()
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
            if check_modified_relax is True:
                if check_same_connectivity(atoms_copy, atoms) is False:
                    atoms.info["modified"] = True
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
            if check_modified_neb is True:
                atoms_IS_1, atoms_FS_1 = get_new_IS_and_FS_from_images(images=images)
                for ss, atoms in {"IS_1": atoms_IS_1, "FS_1": atoms_FS_1}.items():
                    run_relax_calculation(
                        atoms=atoms,
                        calc=calc,
                        fmax=fmax,
                        steps_max=steps_max_relax,
                        name=f"relax_{ss}",
                        directory=directory,
                        save_trajs=save_trajs,
                        write_images=write_images,
                    )
                check_IS = check_same_connectivity(atoms_IS, atoms_IS_1)
                check_FS = check_same_connectivity(atoms_FS, atoms_FS_1)
                if bool(check_IS*check_FS) is False:
                    atoms_TS.info["modified"] = True
            
            write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
        print_results_calculation(atoms=atoms_TS)

        # If NEB is not converged, do a TS search calculation.
        if atoms_TS.info["converged"] is False and atoms_TS.info["modified"] is False:

            atoms_TS.info["calculation"] = search_TS
            if read_atoms_from_db(atoms=atoms_TS, db_ase=db_ase) is None:
                run_TS_search_calculation(
                    atoms=atoms_TS,
                    calc=calc,
                    search_TS=search_TS,
                    steps_max=steps_max_ts_search,
                    bonds_TS=bonds_TS,
                    fmax=fmax,
                    directory=directory,
                    save_trajs=save_trajs,
                    write_images=write_images,
                )

                # Check if relaxing the TS gives the initial IS and FS.
                if check_modified_TS_search is True:
                    atoms_IS_2, atoms_FS_2 = get_new_IS_and_FS_from_atoms_TS(
                        atoms_TS=atoms_TS,
                        bonds_TS=bonds_TS,
                    )
                    for ss, atoms in {"IS_2": atoms_IS_2, "FS_2": atoms_FS_2}.items():
                        run_relax_calculation(
                            atoms=atoms,
                            calc=calc,
                            fmax=fmax,
                            steps_max=steps_max_relax,
                            name=f'relax_{ss}_2',
                            directory=directory,
                            save_trajs=save_trajs,
                            write_images=write_images,
                        )
                    check_IS = check_same_connectivity(atoms_IS, atoms_IS_2)
                    check_FS = check_same_connectivity(atoms_FS, atoms_FS_2)
                    if bool(check_IS*check_FS) is False:
                        atoms_TS.info["modified"] = True

                write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)
            print_results_calculation(atoms=atoms_TS)

        # If the TS is found, calculate vibrations.
        if atoms_TS.info["converged"] and atoms_TS.info["modified"] is False:
            if "vib_energies" not in atoms_TS.info:
                run_vibrations_calculation(atoms=atoms_TS, calc=calc)
                write_atoms_to_db(atoms=atoms_TS, db_ase=db_ase)

# -------------------------------------------------------------------------------------
# RUN DFT CALCULATIONS K8S
# -------------------------------------------------------------------------------------

def run_dft_calculations_k8s(
    atoms_list,
    calc,
    template_yaml,
    namespace,
    basedir_dft_calc,
    filename_out,
    calculation,
    db_dft,
    write_input_fun,
    check_finished_fun,
    job_queued_fun,
    cpu_req=8,
    cpu_lim=12,
    mem_req='8Gi',
    mem_lim='24Gi',
):
    
    from arkimede.workflow.utilities import (
        read_atoms_from_db,
        write_atoms_to_db,
        print_results_calculation,
    )
    from arkimede.workflow.dft_calculations import (
        submit_k8s,
        read_dft_output,
    )
    
    # Read the template yaml file.
    with open(template_yaml, 'r') as fileobj:
        params_k8s = yaml.safe_load(fileobj)
    
    # Run the dft calculations.
    cwd = os.getcwd()
    for atoms in atoms_list:
        atoms.pbc = True
        atoms.calc = calc
        atoms.info["converged"] = False
        atoms.info["calculation"] = calculation
        if read_atoms_from_db(atoms=atoms, db_ase=db_dft) is None:
            directory = os.path.join(basedir_dft_calc, atoms.info["name"])
            os.makedirs(directory, exist_ok=True)
            os.chdir(directory)
            if os.path.isfile(filename_out) and check_finished_fun(filename_out):
                read_dft_output(
                    atoms=atoms,
                    filename=filename_out,
                )
                if atoms.info["converged"] is True:
                    atoms.set_tags(np.ones(len(atoms)))
                    write_atoms_to_db(atoms=atoms, db_ase=db_dft)
            elif job_queued_fun(atoms.info["name"]) is False:
                write_input_fun(atoms)
                submit_k8s(
                    params_k8s=params_k8s,
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
