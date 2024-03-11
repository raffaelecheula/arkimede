# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.formula import Formula
from ase.constraints import FixAtoms
from arkimede.catkit.build import bulk, surface, molecule
from arkimede.catkit.surface import cut_surface, reorder_surface
from arkimede.catkit.utils.connectivity import get_connectivity

# -------------------------------------------------------------------------------------
# LATTICE CONSTANTS
# -------------------------------------------------------------------------------------

lattice_const_dict = {
    "Co": 3.55,
    "Cu": 3.68,
    "Pd": 3.98,
    "Rh": 3.87,
    "Pd": 3.97,
    "Ag": 4.20,
    "Au": 4.20,
    "Ni": 3.53,
    "Pt": 3.98,
    "Ru": 3.82,
    "Os": 3.86,
    "Zn": 2.66,
    "Ga": 4.52,
}

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
    "511": (3, 1),
    "100-100": (2, 2),
    "100-110": (3, 1),
    "100-111": (3, 1),
    "110-111": (3, 1),
    "111-111": (3, 1),
}

# -------------------------------------------------------------------------------------
# GET ATOMS SLAB
# -------------------------------------------------------------------------------------


def get_atoms_slab(
    element_bulk,
    miller_index,
    element_dopant=None,
    number_dopant=0,
    bulk_structure="fcc",
    lattice_const=None,
    repetitions=None,
    magnetic_elements=("Co", "Ni", "Fe"),
    vacuum=10.0,
    delta_ncoord=0,
):

    if repetitions is None:
        repetitions = repetitions_dict[miller_index]

    if lattice_const is None:
        lattice_const = lattice_const_dict[element_bulk]

    if miller_index == "100":
        layers = 4
        miller = (1, 0, 0)
        fixed = 2
        orthogonal = False

    elif miller_index == "110":
        layers = 6
        miller = (1, 1, 0)
        fixed = 2
        orthogonal = True

    elif miller_index == "111":
        layers = 4
        miller = (1, 1, 1)
        fixed = 2
        orthogonal = False

    elif miller_index == "210":
        layers = 9
        miller = (2, 1, 0)
        fixed = 3
        orthogonal = True

    elif miller_index == "211":
        layers = 9 # 12
        miller = (2, 1, 1)
        fixed = 3 # 6
        orthogonal = True

    elif miller_index == "221":
        layers = 12
        miller = (2, 2, 1)
        fixed = 4
        orthogonal = True

    elif miller_index == "310":
        layers = 12
        miller = (3, 1, 0)
        fixed = 4
        orthogonal = True
    
    elif miller_index == "320":
        layers = 15
        miller = (3, 2, 0)
        fixed = 5
        orthogonal = True
    
    elif miller_index == "311":
        layers = 6 # 8
        miller = (3, 1, 1)
        fixed = 2 # 4
        orthogonal = True
    
    elif miller_index == "321":
        layers = 15
        miller = (3, 2, 1)
        fixed = 5
        orthogonal = True
    
    elif miller_index == "331":
        layers = 9
        miller = (3, 3, 1)
        fixed = 3
        orthogonal = True

    elif miller_index == "511":
        layers = 9
        miller = (5, 1, 1)
        fixed = 3
        orthogonal = True

    elif miller_index == "100-100":
        layers = 7
        miller = (1, 1, 0)
        fixed = 3
        orthogonal = True

    elif miller_index == "100-110":
        layers = 9
        miller = (2, 1, 0)
        fixed = 3
        orthogonal = True

    elif miller_index == "100-111":
        layers = 7
        miller = (1, 3, 1)
        fixed = 4
        orthogonal = True

    elif miller_index == "110-111":
        layers = 9
        miller = (3, 3, 1)
        fixed = 3
        orthogonal = True

    elif miller_index == "111-111":
        layers = 7
        miller = (1, 1, 0)
        fixed = 3
        orthogonal = True

    a_init = 1.0
    atoms_bulk = bulk(
        name="X",
        a=a_init,
        crystalstructure=bulk_structure,
        primitive=True,
    )
    atoms_bulk.info["name"] = "-".join([bulk_structure, element_bulk])

    atoms_slab = surface(
        bulk=atoms_bulk,
        size=(1, 1, layers),
        miller=miller,
        termination=0,
        fixed=fixed,
        vacuum=vacuum / 2.0,
        orthogonal=orthogonal,
        attach_graph=True,
        delta_ncoord=delta_ncoord,
    )

    if miller_index == "100-100":
        atoms_slab = cut_surface(atoms_slab, vectors=[[2, 0], [0, 1]])
        atoms_slab = reorder_surface(atoms_slab)
        del atoms_slab[[1, 12]]
        atoms_slab.set_constraint(FixAtoms(indices=[0, 1, 2, 4]))
        top = [7, 9, 10, 11]
        bottom = [0, 1, 2, 4]
        atoms_slab.set_surface_atoms(top=top, bottom=bottom)
        atoms_slab.edit()

    if miller_index == "100-110":
        atoms_slab = cut_surface(atoms_slab, vectors=[[1, 0], [-1, 2]])
        atoms_slab = reorder_surface(atoms_slab)
        del atoms_slab[[1, 16]]
        atoms_slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3, 4, 6]))
        top = [12, 13, 14, 15]
        bottom = [0, 1, 2, 4]
        if delta_ncoord == 0:
            top += [9, 11]
            bottom += [3, 6]
        atoms_slab.set_surface_atoms(top=top, bottom=bottom)

    elif miller_index == "100-111":
        atoms_slab = cut_surface(atoms_slab, vectors=[[1, 0], [-1, 2]])
        atoms_slab = reorder_surface(atoms_slab)
        del atoms_slab[[1, 12]]
        atoms_slab.set_constraint(FixAtoms(indices=[0, 1, 2, 4]))
        atoms_slab.set_surface_atoms(top=[7, 9, 10, 11], bottom=[0, 1, 2, 4])

    elif miller_index == "110-111":
        atoms_slab = cut_surface(atoms_slab, vectors=[[1, 0], [-1, 2]])
        atoms_slab = reorder_surface(atoms_slab)
        del atoms_slab[[1, 16]]
        atoms_slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3, 4, 6]))
        top = [11, 13, 14, 15]
        bottom = [0, 1, 2, 3]
        if delta_ncoord == 0:
            top += [10, 12]
            bottom += [4, 6]
        atoms_slab.set_surface_atoms(top=top, bottom=bottom)

    elif miller_index == "111-111":
        atoms_slab = cut_surface(atoms_slab, vectors=[[1, 0], [0, 2]])
        atoms_slab = reorder_surface(atoms_slab)
        del atoms_slab[[1, 12]]
        atoms_slab.set_constraint(FixAtoms(indices=[0, 1, 2, 4]))
        top = [9, 10, 11]
        bottom = [0, 1, 2]
        if delta_ncoord == 0:
            top += [7]
            bottom += [4]
        atoms_slab.set_surface_atoms(top=top, bottom=bottom)

    atoms_slab.set_cell(atoms_slab.cell * lattice_const / a_init, scale_atoms=True)
    atoms_slab.center(vacuum=vacuum / 2.0, axis=2)
    atoms_slab.symbols = [element_bulk] * len(atoms_slab)

    n_atoms_1x1 = len(atoms_slab)
    atoms_slab *= (repetitions[0], 1, 1)
    atoms_slab *= (1, repetitions[1], 1)
    sort_list = sorted([(a.index % n_atoms_1x1, a.index) for a in atoms_slab])
    indices = [i[1] for i in sort_list]
    atoms_slab = atoms_slab[indices]

    if number_dopant == 0 or element_dopant == element_bulk:
        element_dopant = element_bulk[:]
        number_dopant = 0

    get_connectivity(
        atoms=atoms_slab,
        method="cutoff",
        add_cov_radii=0.2,
        attach_graph=True,
    )

    surf_indices = [a.index for a in atoms_slab if a.tag == 1]
    surf_indices.reverse()

    for ii in range(number_dopant):
        jj = surf_indices[ii]
        atoms_slab[jj].symbol = element_dopant

    magmoms = [1.0 if a.symbol in magnetic_elements else 0.0 for a in atoms_slab]
    atoms_slab.set_initial_magnetic_moments(magmoms=magmoms)

    # Get name and build atoms.info
    surf_structure = atoms_bulk.info["name"][:]
    if number_dopant > 0:
        surf_structure += f"+{element_dopant}{number_dopant}"
    surf_structure += f"-{miller_index}"

    name = "_".join([surf_structure, "x".join([str(ii) for ii in repetitions])])

    atoms_slab.info = {
        "name": name,
        "species": "clean",
        "surf_structure": surf_structure,
        "n_atoms_clean": len(atoms_slab),
        "element_bulk": element_bulk,
        "miller_index": miller_index,
        "element_dopant": element_dopant,
        "number_dopant": number_dopant,
        "repetitions": repetitions,
        "lattice_const": lattice_const,
    }

    return atoms_slab


# -------------------------------------------------------------------------------------
# GET ATOMS GAS
# -------------------------------------------------------------------------------------


def get_atoms_gas(species, vacuum=10.0):

    top_brg_combinations = [
        {'bonds': ['top', 'top'], 'over': 'brg'},
        {'bonds': ['top', 'top'], 'over': '4fh'},
        {'bonds': ['brg', 'brg'], 'over': '4fh'},
        {'bonds': ['top', 'brg'], 'over': '3fh'},
        {'bonds': ['brg', 'top'], 'over': '3fh'},
    ]

    gas_name = species.replace("*", "")

    if "*" not in species:
        atoms_gas = molecule(gas_name)[0]
        bonds_surf = []
        sites_names = []
        symmetric_ads = False
        dissoc_dict = {}

    elif species in ("O*", "C*"):
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = ["brg", "3fh", "4fh"]
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "H*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "CO*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "C*+O*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "CH*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "C*+H*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "CH2*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "CH*+H*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "CH3*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "CH2*+H*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "OH*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "O*+H*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "H2O*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = ["top", "brg"]
        symmetric_ads = False
        dissoc_dict = {
            "OH*+H*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "CO2**":
        gas_name = "CO2"
        bonds_surf = [0, 1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {
            "CO*+O*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "CO2*":
        gas_name = "CO2"
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        atoms_gas.set_angle(1, 0, 2, 135)
        atoms_gas.set_angle(2, 0, 1, 150)
        sites_names = ["brg"]
        symmetric_ads = False
        dissoc_dict = {
            "CO*+O*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "COH*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "CO*+H*": {"bond_break": [1, 2], "bonds_surf": [0, 2]},
        }

    elif species == "HCO*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = ["top", "brg"]
        symmetric_ads = False
        dissoc_dict = {
            "CH*+O*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
            "CO*+H*": {"bond_break": [0, 2], "bonds_surf": [0, 2]},
        }

    elif species == "HCO**":
        bonds_surf = [0, 1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {
            "CH*+O*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
            "CO*+H*": {"bond_break": [0, 2], "bonds_surf": [0, 2]},
        }

    elif species == "c-COOH*":
        gas_name = "COOH"
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        sites_names = ["top", "brg"]
        symmetric_ads = False
        dissoc_dict = {
            "CO*+OH*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
        }

    elif species == "c-COOH**":
        gas_name = "COOH"
        bonds_surf = [0, 2]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {
            "CO*+OH*": {"bond_break": [0, 1], "bonds_surf": [0, 1]},
            "COH*+O*": {"bond_break": [0, 2], "bonds_surf": [0, 2]},
        }

    elif species == "t-COOH*":
        gas_name = "COOH"
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        atoms_gas.set_angle(0, 1, 3, -120)
        sites_names = ["top", "brg"]
        symmetric_ads = False
        dissoc_dict = {
            "CO2*+H*": {"bond_break": [1, 3], "bonds_surf": [0, 3]},
        }

    elif species == "t-COOH**":
        gas_name = "COOH"
        bonds_surf = [0, 2]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        atoms_gas.set_angle(0, 1, 3, -120)
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {
            "CO2*+H*": {"bond_break": [1, 3], "bonds_surf": [0, 3]},
            "CO2**+H*": {"bond_break": [1, 3], "bonds_surf": [0, 2, 3]},
        }

    elif species == "HCOO**":
        bonds_surf = [1, 2]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = top_brg_combinations[:]
        symmetric_ads = True
        dissoc_dict = {
            "HCO*+O*": {"bond_break": [0, 2], "bonds_surf": [0, 2]},
            "HCO**+O*": {"bond_break": [0, 2], "bonds_surf": [0, 1, 2]},
            "CO2**+H*": {"bond_break": [0, 3], "bonds_surf": [0, 2, 3]},
        }

    elif species == "HCOO*":
        bonds_surf = [1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {
            "CO2*+H*": {"bond_break": [0, 3], "bonds_surf": [0, 3]},
            "CO2**+H*": {"bond_break": [0, 3], "bonds_surf": [0, 2, 3]},
        }

    elif species == "H2COO**":
        bonds_surf = [1, 2]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = top_brg_combinations[:]
        symmetric_ads = True
        dissoc_dict = {}

    elif species == "H2COOH**":
        bonds_surf = [1, 2]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {
            "H2CO**+OH*": {"bond_break": [0, 1], "bonds_surf": [0, 1, 2]},
        }

    elif species == "H2CO**":
        bonds_surf = [0, 1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "H3CO*":
        bonds_surf = [1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[0]
        sites_names = None
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "HCOH**":
        bonds_surf = [0, 1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "H2COH**":
        bonds_surf = [0, 1]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "HCOOH*":
        bonds_surf = [1, 2]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[1]
        sites_names = top_brg_combinations[:]
        symmetric_ads = False
        dissoc_dict = {}

    elif species == "COHOH*":
        bonds_surf = [0]
        atoms_gas = molecule(gas_name, bond_index=bonds_surf)[2]
        sites_names = ["top", "brg"]
        symmetric_ads = False
        dissoc_dict = {}

    elif species is not None:
        raise ValueError(f"Adsorbate {species} not defined!")

    for i in bonds_surf:
        atoms_gas[i].tag = -1

    atoms_gas.center(vacuum / 2.0)

    formula = Formula(atoms_gas.get_chemical_formula())
    symbols_dict = formula.count()

    atoms_gas.info = {
        "species": species,
        "bonds_surf": bonds_surf,
        "sites_names": sites_names,
        "symmetric_ads": symmetric_ads,
        "dissoc_dict": dissoc_dict,
        "symbols_dict": symbols_dict,
    }

    return atoms_gas


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    from ase.gui.gui import GUI

    gas_species_list = [
        "CH3OH",
        "CO",
        "H2O",
        "CO2",
        "CH4",
    ]

    adsorbates_list = [
        "H*",
        "O*",
        "C*",
        "CO*",
        "CH*",
        "CH2*",
        "CH3*",
        "OH*",
        "H2O*",
        "CO2**",
        "CO2*",
        "COH*",
        "HCO*",
        "HCO**",
        "c-COOH*",
        "c-COOH**",
        "t-COOH*",
        "t-COOH**",
        "HCOO**",
        "HCOO*",
        "H2COO**",
        "H2COOH**",
        "H2CO**",
        "H3CO*",
        "HCOH**",
        "H2COH**",
        "HCOOH*",
        "COHOH*",
    ]

    atoms_list = []
    for species in gas_species_list:
        atoms_list.append(get_atoms_gas(species=species))
    for species in adsorbates_list:
        atoms_list.append(get_atoms_gas(species=species))

    gui = GUI(atoms_list)
    gui.run()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
