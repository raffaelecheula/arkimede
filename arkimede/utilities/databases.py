# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator, all_properties

# -------------------------------------------------------------------------------------
# WRITE ATOMS TO DB
# -------------------------------------------------------------------------------------

def write_atoms_to_db(
    atoms,
    db_ase,
    get_status=True,
    keys_match=["name", "calculation"],
    keys_store=["name", "calculation", "status"],
):
    """Write atoms to ase database."""
    if get_status is True:
        atoms.info["status"] = get_status_calculation(atoms)
    kwargs_match = {}
    for key in keys_match:
        kwargs_match[key] = atoms.info[key]
    kwargs_store = {}
    for key in keys_store:
        kwargs_store[key] = atoms.info[key]
    filter_constraints(atoms)
    if db_ase.count(**kwargs_match) == 0:
        db_ase.write(
            atoms=atoms,
            data=atoms.info,
            **kwargs_store,
        )
    elif db_ase.count(**kwargs_match) == 1:
        row_id = db_ase.get(**kwargs_match).id
        db_ase.update(
            id=row_id,
            atoms=atoms,
            data=atoms.info,
            **kwargs_store,
        )

# -------------------------------------------------------------------------------------
# READ ATOMS FROM DATABASE
# -------------------------------------------------------------------------------------

def read_atoms_from_db(atoms, db_ase, keys_match=["name", "calculation"]):
    """Read atoms from ase database. Does not create a new Atoms object."""
    kwargs_match = {}
    for key in keys_match:
        kwargs_match[key] = atoms.info[key]
    if db_ase.count(**kwargs_match) == 1:
        atoms_row = db_ase.get(**kwargs_match)
        info = atoms_row.data.copy()
        info.update(atoms_row.key_value_pairs)
        atoms.info.update(info)
        atoms.set_positions(atoms_row.positions)
        results = filter_results(results=atoms_row)
        atoms.calc = SinglePointCalculator(atoms=atoms, **results)
        return atoms
    else:
        return None

# -------------------------------------------------------------------------------------
# GET ATOMS LIST FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_list_from_db(db_ase, selection="", **kwargs):
    """Read list of atoms from database, according to selection string."""
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection, **kwargs)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms_list.append(atoms)
    return atoms_list

# -------------------------------------------------------------------------------------
# FILTER RESULTS
# -------------------------------------------------------------------------------------

def filter_results(results, properties=all_properties):
    return {pp: results[pp] for pp in results if pp in properties}

# -------------------------------------------------------------------------------------
# FILTER CONSTRAINTS
# -------------------------------------------------------------------------------------

def filter_constraints(atoms):
    """Filter constraints to be stored into an ase database."""
    from ase.constraints import __all__
    for ii, constraint in reversed(list(enumerate(atoms.constraints))):
        if constraint.todict()['name'] not in __all__:
            del atoms.constraints[ii]

# -------------------------------------------------------------------------------------
# GET METADATA REACTIONS
# -------------------------------------------------------------------------------------

def get_metadata_reaction(atoms_neb):
    """Get list of metadata for a reaction."""
    return [
        atoms_neb[0].info["name"],
        atoms_neb[1].info["name"],
        atoms_neb[1].info["bonds_TS"],
    ]

# -------------------------------------------------------------------------------------
# GET DB METADATA
# -------------------------------------------------------------------------------------

def get_db_metadata(
    atoms_gas_tot,
    atoms_clean_tot,
    atoms_ads_tot,
    atoms_neb_tot,
):
    """Get the structures names as dictionary."""
    metadata = {
        "gas": [atoms.info["name"] for atoms in atoms_gas_tot],
        "clean": [atoms.info["name"] for atoms in atoms_clean_tot],
        "adsorbate": [atoms.info["name"] for atoms in atoms_ads_tot],
        "reaction": [get_metadata_reaction(atoms_neb) for atoms_neb in atoms_neb_tot]
    }
    return metadata

# -------------------------------------------------------------------------------------
# GET ATOMS LIST FROM DB METADATA
# -------------------------------------------------------------------------------------

def get_atoms_list_from_db_metadata(db_ase, selection, metadata_key):
    """Read atoms from database according to metadata_key string."""
    atoms_list = []
    if metadata_key == "reaction":
        for obj in db_ase.metadata[metadata_key]:
            atoms_neb = []
            name_IS, name_FS, bonds_TS = obj
            for name in [name_IS, name_FS]:
                selection_new = ",".join([selection, f"name={name}"])
                atoms_row = db_ase.get(selection=selection_new)
                atoms = atoms_row.toatoms()
                atoms.info = atoms_row.data
                atoms.info["bonds_TS"] = bonds_TS
                atoms_neb.append(atoms)
            atoms_list.append(atoms_neb)
    else:
        for name in db_ase.metadata[metadata_key]:
            selection_new = ",".join([selection, f"name={name}"])
            atoms_row = db_ase.get(selection=selection_new)
            atoms = atoms_row.toatoms()
            atoms.info = atoms_row.data
            atoms_list.append(atoms)
    return atoms_list

# -------------------------------------------------------------------------------------
# UPDATE ATOMS FROM ATOMS OPT
# -------------------------------------------------------------------------------------

def update_atoms_from_atoms_opt(
    atoms,
    atoms_opt,
    converged,
    modified,
    properties=all_properties,
    update_cell=False,
):
    """Update the input atoms with the results of the optimized atoms."""
    atoms.set_positions(atoms_opt.get_positions())
    if update_cell is True:
        atoms.set_cell(atoms_opt.get_cell())
    results = filter_results(results=atoms_opt.calc.results, properties=properties)
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)
    atoms.info.update(atoms_opt.info)
    atoms.info["converged"] = bool(converged)
    atoms.info["modified"] = bool(modified)
    if "counter" in dir(atoms_opt.calc):
        atoms.info["counter"] = atoms_opt.calc.counter

    return atoms

# -------------------------------------------------------------------------------------
# GET STATUS CALCULATION
# -------------------------------------------------------------------------------------

def get_status_calculation(atoms):
    """Get the status of the calculation."""
    if atoms.info["converged"] is True and atoms.info["modified"] is False:
        status = "finished"
    elif atoms.info["converged"] is True and atoms.info["modified"] is True:
        status = "modified"
    else:
        status = "unfinished"
    return status

# -------------------------------------------------------------------------------------
# PRINT RESULTS CALCULATION
# -------------------------------------------------------------------------------------

def print_results_calculation(atoms, get_status=True):
    """Print results of the calculation to screen."""
    if get_status is True:
        atoms.info["status"] = get_status_calculation(atoms)
    status_str = f'{atoms.info["status"]:15s}'
    if atoms.calc and "energy" in atoms.calc.results:
        energy_str = f'{atoms.calc.results["energy"]:15.3f}'
    else:
        energy_str = None
    print(
        f'{atoms.info["name"]:100s}',
        f'{atoms.info["calculation"]:20s}',
        status_str,
        energy_str,
    )

# -------------------------------------------------------------------------------------
# UPDATE INFO TS COMBO
# -------------------------------------------------------------------------------------

def update_info_TS_combo(atoms_TS, atoms_IS, atoms_FS):
    """Update info of atoms_TS from info of atoms_IS and atoms_FS."""
    species = "→".join([str(atoms_IS.info["species"]), str(atoms_FS.info["species"])])
    site_id = "→".join([str(atoms_IS.info["site_id"]), str(atoms_FS.info["site_id"])])
    name_TS = "_".join([str(atoms_IS.info["name_ref"]), species, site_id])
    atoms_TS.info["species"] = species
    atoms_TS.info["site_id"] = site_id
    atoms_TS.info["name"] = name_TS
    if atoms_IS.info["bonds_TS"] != atoms_FS.info["bonds_TS"]:
        import warnings
        warnings.warn("atoms_IS and atoms_FS have different bonds_TS")
    atoms_TS.info["bonds_TS"] = atoms_FS.info["bonds_TS"]

# -------------------------------------------------------------------------------------
# UPDATE INFO TS FROM IS
# -------------------------------------------------------------------------------------

def update_info_TS_from_IS(atoms_TS, atoms_IS, atoms_FS):
    """Update info of atoms_TS from info of atoms_IS."""
    atoms_TS.info = atoms_IS.info.copy()
    atoms_TS.info["name"] = atoms_TS.info["name"].replace("_IS", "_TS")
    atoms_TS.info["bonds_TS"] = atoms_IS.info["bonds_TS"]

# -------------------------------------------------------------------------------------
# UPDATE INFO TS FROM FS
# -------------------------------------------------------------------------------------

def update_info_TS_from_FS(atoms_TS, atoms_IS, atoms_FS):
    """Update info of atoms_TS from info of atoms_FS."""
    atoms_TS.info = atoms_FS.info.copy()
    atoms_TS.info["name"] = atoms_TS.info["name"].replace("_FS", "_TS")
    atoms_TS.info["bonds_TS"] = atoms_FS.info["bonds_TS"]

# -------------------------------------------------------------------------------------
# SUCCESS CURVE
# -------------------------------------------------------------------------------------

def success_curve(atoms_list, max_steps=1000, filename=None):
    """Calculate success curve."""
    success_steps = np.zeros(max_steps)
    for atoms in atoms_list:
        if atoms.info["status"] == "finished":
            success_steps[atoms.info["counter"]:] += 1
    if len(atoms_list) > 0:
        success_steps *= 100/len(atoms_list)
    # Plot success curve.
    if filename is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(np.arange(max_steps), success_steps)
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, 100)
        ax.set_xlabel("single-point calculations [-]")
        ax.set_ylabel("success [%]")
        plt.savefig(filename)
    return success_steps

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
