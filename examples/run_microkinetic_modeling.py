# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from ase.thermochemistry import HarmonicThermo
from mikimoto import units
from mikimoto.thermodynamics import ThermoFixedG0
from mikimoto.microkinetics import (
    Species, IdealGas, Interface, Reaction, Microkinetics,
)
from mikimoto.reactors import CSTReactor

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Temperature and pressure.
    temperature = 300+273.15 # [K]
    pressure = 1 * units.atm # [Pa]
    
    # Catalyst and reactor parameters.
    sites_conc = 1. # [kmol/m^3]
    reactor_volume = 1e-6 # [m^3]
    vol_flow_rate = 1e-6 # [m^3/s]
    
    # Name of ase database.
    db_ase_name = "databases/ocp_00.db"

    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Surface structures parameters.
    bulk_structure = "fcc"
    element_bulk_list = [
        "Rh",
        #"Cu",
        #"Pd",
        #"Co",
        #"Ni",
        #"Au",
    ]
    miller_index_list = [
        "100",
        #"110",
        #"111",
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
    ]

    # Reactions names.
    reaction_list = [
        "CO2**→CO*+O*",
        "H2O*→OH*+H*",
        "OH*→O*+H*",
        "t-COOH**→CO2**+H*",
        "c-COOH**→CO*+OH*",
    ]

    units_energy = units.eV/units.molecule

    # Gibbs free energy of gas phase species.
    Gibbs_gas_species = {
        "CO2": -0.490+0.309-1.222,
        "H2": +0.000+0.274-0.718,
        "CO": +0.000+0.141-1.114,
        "H2O": +0.000+0.569-1.053,
    }

    # Gibbs free energy of gas phase transition states.
    Gibbs_gas_reactions = {
    }

    # Gibbs free energy of surface species.
    Gibbs_surf_species_common = {
        "(X)": 0.0,
    }
    
    # Gibbs free energy of surface transition states.
    Gibbs_surf_reactions_common = {
        "CO2 + 2 (X) <=> CO2(X)(X)": Gibbs_gas_species["CO2"]+0.544,
        "H2 + 2 (X) <=> 2 H(X)": Gibbs_gas_species["H2"]+0.467,
        "CO + (X) <=> CO(X)": Gibbs_gas_species["CO"]+0.533,
        "H2O + (X) <=> H2O(X)": Gibbs_gas_species["H2O"]+0.522,
    }

    # Set the initial molar fractions of the gas phase.
    X_in_dict = {
        "CO2": 0.50,
        "H2": 0.50,
        "CO": 0.00,
        "H2O": 0.00,
    }

    # Set the initial coverages of the surface.
    θ_in_dict = {
        "(X)": 1.00,
    }

    productivities = {}

    # Build microkinetic models.
    for element_bulk in element_bulk_list:
        for miller_index in miller_index_list:
            
            Gibbs_surf_species = Gibbs_surf_species_common.copy()
            Gibbs_surf_reactions = Gibbs_surf_reactions_common.copy()
            
            # Surface structure.
            surf_structure = "-".join(
                [bulk_structure, element_bulk, miller_index]
            )
            
            # Get energy of clean surface.
            atoms_clean = db_ase.get(
                surf_structure=surf_structure,
                species="clean",
                status="finished",
            )

            # Get Gibbs free energy of the adsorbates.
            for species in adsorbate_list:
                atoms_list = list(db_ase.select(
                    surf_structure=surf_structure,
                    species=species,
                    status="finished",
                ))
                Gibbs_energy = get_Gibbs_energy_from_atoms_list(
                    atoms_list=atoms_list,
                    atoms_clean=atoms_clean,
                    temperature=temperature,
                ) # [eV/molecule]
                name = get_mikimoto_name_from_species_name(species=species)
                Gibbs_surf_species[name] = Gibbs_energy # [J/kmol]

            # Get the Gibbs free energy of the transition states.
            for species in reaction_list:
                atoms_list = list(db_ase.select(
                    surf_structure=surf_structure,
                    species=species,
                    status="finished",
                ))
                Gibbs_energy = get_Gibbs_energy_from_atoms_list(
                    atoms_list=atoms_list,
                    atoms_clean=atoms_clean,
                    temperature=temperature,
                ) # [eV/molecule]
                name = get_mikimoto_name_from_species_name(species=species)
                Gibbs_surf_reactions[name] = Gibbs_energy # [J/kmol]
            
            results = get_catalytic_performances(
                Gibbs_gas_species=Gibbs_gas_species,
                Gibbs_gas_reactions=Gibbs_gas_reactions,
                Gibbs_surf_species=Gibbs_surf_species,
                Gibbs_surf_reactions=Gibbs_surf_reactions,
                temperature=temperature,
                pressure=pressure,
                sites_conc=sites_conc,
                reactor_volume=reactor_volume,
                vol_flow_rate=vol_flow_rate,
                X_in_dict=X_in_dict,
                θ_in_dict=θ_in_dict,
                units_energy=units_energy,
            )

            print(f"Structure: {surf_structure}")
            print(f'productivity CO2: {results["productvity_CO"]:5.3e}')
            productivities[surf_structure] = results["productvity_CO"]

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-colorblind')
    plt.figure(figsize=(15, 5))
    plt.bar(
        productivities.keys(),
        productivities.values(),
    )
    plt.yscale("log")
    plt.ylabel("CO productivity [kmol/m$^3$/s]")
    plt.savefig("plot.png")

# -------------------------------------------------------------------------------------
# GET GIBBS SPECIES FROM ATOMS LIST
# -------------------------------------------------------------------------------------

def get_Gibbs_energy_from_atoms_list(atoms_list, atoms_clean, temperature):

    ii = np.argmin([atoms.energy for atoms in atoms_list])
    energy = atoms_list[ii].energy-atoms_clean.energy
    vib_energies = [
        vv for vv in atoms_list[ii].data["vib_energies"] if np.isreal(vv)
    ]
    thermo = HarmonicThermo(
        potentialenergy=energy,
        vib_energies=vib_energies,
    )
    Gibbs_energy = thermo.get_helmholtz_energy(
        temperature=temperature,
        verbose=False,
    )
    return Gibbs_energy

# -------------------------------------------------------------------------------------
# GET MIKIMOTO NAME FROM SPECIES NAME
# -------------------------------------------------------------------------------------

def get_mikimoto_name_from_species_name(species):

    name = species[:]
    replace_dict = {
        "+": " + ",
        "→": " <=> ",
        "*": "(X)",
    }
    for string in replace_dict:
        name = name.replace(string, replace_dict[string])
    name = adjust_number_active_sites(name=name)

    return name

# -------------------------------------------------------------------------------------
# ADJUST NUMBER ACTIVE SITES
# -------------------------------------------------------------------------------------

def adjust_number_active_sites(name):

    if " <=> " in name:
        reactants, products = name.split(" <=> ")
        n_sites_reactants = reactants.count("X")
        n_sites_products = products.count("X")
        if n_sites_reactants > n_sites_products:
            products += f" + {n_sites_reactants-n_sites_products} (X)"
        elif n_sites_reactants < n_sites_products:
            reactants += f" + {n_sites_products-n_sites_reactants} (X)"
        name = " <=> ".join([reactants, products])

    return name

# -------------------------------------------------------------------------------------
# GET CATALITIC ACTIVITY
# -------------------------------------------------------------------------------------

def get_catalytic_performances(
    Gibbs_gas_species,
    Gibbs_gas_reactions,
    Gibbs_surf_species,
    Gibbs_surf_reactions,
    temperature,
    pressure,
    sites_conc,
    reactor_volume,
    vol_flow_rate,
    X_in_dict,
    θ_in_dict,
    units_energy,
):
    
    gas_species = [
        Species(name, thermo=ThermoFixedG0(Gibbs_ref=Gibbs_gas_species[name]))
        for name in Gibbs_gas_species
    ]
    gas_reactions = [
        Reaction(name, thermo=ThermoFixedG0(Gibbs_ref=Gibbs_gas_reactions[name]))
        for name in Gibbs_gas_reactions
    ]

    gas = IdealGas(
        temperature=temperature,
        pressure=pressure,
        species=gas_species,
        reactions=gas_reactions,
    )

    surf_species = [
        Species(name, thermo=ThermoFixedG0(Gibbs_ref=Gibbs_surf_species[name]))
        for name in Gibbs_surf_species
    ]
    surf_reactions = [
        Reaction(name, thermo=ThermoFixedG0(Gibbs_ref=Gibbs_surf_reactions[name]))
        for name in Gibbs_surf_reactions
    ]

    surf = Interface(
        temperature=temperature,
        pressure=pressure,
        conc_tot=sites_conc,
        species=surf_species,
        reactions=surf_reactions,
    )

    microkin = Microkinetics(
        gas_phases=[gas],
        surf_phases=[surf],
        temperature=temperature,
        pressure=pressure,
        units_energy=units_energy,
    )

    for name in [spec.name for spec in surf_species if spec.name not in θ_in_dict]:
        θ_in_dict[name] = 0.

    gas.X_dict = X_in_dict
    surf.θ_dict = θ_in_dict

    reactor = CSTReactor(
        microkin=microkin,
        method='Radau',
        reactor_volume=reactor_volume,
        vol_flow_rate=vol_flow_rate,
        constant_temperature=True,
        constant_pressure=True,
        rtol=1e-13,
        atol=1e-15,
        t_bound=np.inf,
        conv_thr_ode=1e-5,
    )
    try: reactor.integrate_volume()
    except: pass

    productvity = (
        (gas.X_dict["CO"]-X_in_dict["CO"])*gas.conc_tot
        * vol_flow_rate/reactor_volume
    )
    conversion = (X_in_dict["CO2"]-gas.X_dict["CO2"])/X_in_dict["CO2"]
    results = {
        "conversion_CO2": conversion, # [%]
        "productvity_CO": productvity, # [kmol/m^3/s]
    }

    return results

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
