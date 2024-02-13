# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from arkimede.catkit.reaction_mechanism import MechanismBuilder

# -----------------------------------------------------------------------------
# STRUCTURES PARAMETERS
# -----------------------------------------------------------------------------

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
        self.get_atoms_gas_fun = get_atoms_gas_fun # This could be a dictionary
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

    def get_adsorbates(self, ads_name_list):
        
        # Get the atoms structures of the surface slab with adsorbates.
        atoms_ads_tot = []
        for ads_name in ads_name_list:

            adsorbate = self.get_atoms_gas_fun(ads_name=ads_name)

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
                atoms_ads.info["ads_name"] = adsorbate.info["ads_name"]
                atoms_ads.info["name"] = "_".join(
                    [
                        atoms_ads.info["surface_name"],
                        atoms_ads.info["ads_name"],
                        atoms_ads.info["site_id"]
                    ]
                )

            atoms_ads_tot += atoms_ads_list

        return atoms_ads_tot

    def get_neb_images(self, reactions_list):
        
        atoms_neb_tot = []
        # Get the images for reaction calculations.
        for reaction in reactions_list:
        
            reactants_str, products_str = reaction.split("→")
            reactants = [
                self.get_atoms_gas_fun(ads_name=name)
                for name in reactants_str.split("+")
            ]
            products = [
                self.get_atoms_gas_fun(ads_name=name)
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
                    slab_reactants.info["ads_name"] = reactants_str
                    slab_reactants.info["name"] = "_".join(
                        [
                            slab_reactants.info["surface_name"],
                            slab_reactants.info["ads_name"],
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
                        slab_products.info["ads_name"] = products_str
                        slab_products.info["name"] = "_".join(
                            [
                                slab_products.info["surface_name"],
                                slab_products.info["ads_name"],
                                slab_products.info["site_id"]
                            ]
                        )
                        atoms_neb_tot.append([slab_reactants, slab_products])

        return atoms_neb_tot

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
