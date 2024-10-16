# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
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

        # Setup the builder of adsorption data.
        self.builder = MechanismBuilder(
            slab=self.atoms_clean,
            tol=1e-5,
            cutoff=self.atoms_clean.info["lattice_const"],
        )

    def get_atoms_name(self, atoms):
        kwords = [
            atoms.info["name_ref"],
            atoms.info["species"],
            atoms.info["site_id"],
        ]
        return "_".join(kwords)

    def get_adsorbates(self, adsorbate_list):
        """Get the atoms structures of the surface slab with adsorbates."""
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
                atoms_ads.info["species"] = adsorbate.info["species"]
                atoms_ads.info["name"] = self.get_atoms_name(atoms_ads)
            atoms_ads_tot += atoms_ads_list

        return atoms_ads_tot

    def get_neb_images(self, reaction_list):
        """Get the initial and final structures for neb reaction calculations."""
        atoms_neb_tot = []
        for reaction in reaction_list:
            name_IS, name_FS = reaction.split("→")
            gas_IS_list = [
                self.get_atoms_gas_fun(species=name) for name in name_IS.split("+")
            ]
            gas_FS_list = [
                self.get_atoms_gas_fun(species=name) for name in name_FS.split("+")
            ]
            # Dissociation reactions.
            if len(gas_IS_list) == 1:
                gas_IS = gas_IS_list[0]
                dissoc_dict = gas_IS.info["dissoc_dict"]
                sites_names_list = [
                    atoms.info["sites_names"] for atoms in gas_FS_list
                ]
                atoms_IS_list = self.builder.add_adsorbate(
                    adsorbate=gas_IS,
                    sites_names=gas_IS.info["sites_names"],
                    range_edges=self.range_edges,
                    auto_construct=self.auto_construct,
                    symmetric_ads=gas_IS.info["symmetric_ads"],
                )
                for atoms_IS in atoms_IS_list:
                    atoms_IS.info["species"] = name_IS
                    atoms_IS.info["name"] = self.get_atoms_name(atoms_IS)
                    atoms_FS_list = self.builder.dissociation_reaction(
                        products=gas_FS_list,
                        bond_break=dissoc_dict[name_FS]["bond_break"],
                        bonds_surf=dissoc_dict[name_FS]["bonds_surf"],
                        slab_reactants=atoms_IS,
                        auto_construct=self.auto_construct,
                        range_edges=self.range_edges,
                        displ_max_reaction=self.displ_max_reaction,
                        range_coadsorption=self.range_coadsorption,
                        sites_names_list=sites_names_list,
                        site_contains_list=None,
                        sort_by_pathlength=False,
                    )
                    for atoms_FS in atoms_FS_list:
                        atoms_IS = atoms_IS.copy()
                        atoms_IS.info["bonds_TS"] = atoms_FS.info["bonds_TS"]
                        atoms_FS.info["species"] = name_FS
                        atoms_FS.info["name"] = self.get_atoms_name(atoms_FS)
                        atoms_neb_tot.append([atoms_IS, atoms_FS])

        return atoms_neb_tot

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
