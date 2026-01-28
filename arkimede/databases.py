# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from tqdm import tqdm
from ase import Atoms
from ase.db.core import Database

# -------------------------------------------------------------------------------------
# WRITE ATOMS TO DB
# -------------------------------------------------------------------------------------

def write_atoms_to_db(
    atoms: Atoms,
    db_ase: Database,
    keys_store: list = [],
    keys_match: list = None,
    deepcopy_all: bool = True,
    fill_stress: bool = False,
    fill_magmom: bool = False,
    no_constraints: bool = False,
    match_kwargs: bool = False,
    **kwargs: dict,
):
    """
    Write atoms to ase database.
    """
    # Deep copy atoms and calculator.
    if deepcopy_all is True:
        calc = deepcopy(atoms.calc)
        atoms = deepcopy(atoms)
        atoms.calc = calc
    # Fill with zeros stress and magmoms.
    if fill_stress and "stress" not in atoms.calc.results:
        atoms.calc.results["stress"] = np.zeros(6)
    if fill_magmom and "magmoms" not in atoms.calc.results:
        atoms.calc.results["magmoms"] = np.zeros(len(atoms))
    # Remove contraints.
    if no_constraints is True:
        atoms.constraints = []
        atoms.calc.atoms = atoms
    # Get dictionary to store atoms.info into the columns of the db.
    kwargs_store = {key: atoms.info[key] for key in keys_store} if keys_store else {}
    kwargs_store.update(**kwargs)
    # Get dictionary to check if structure is already in db.
    kwargs_match = {key: atoms.info[key] for key in keys_match} if keys_match else {}
    if match_kwargs is True:
        kwargs_match.update(**kwargs)
    # Write structure to db.
    if not kwargs_match or db_ase.count(**kwargs_match) == 0:
        db_ase.write(atoms=atoms, data=atoms.info, **kwargs_store)
    elif db_ase.count(**kwargs_match) == 1:
        row_id = db_ase.get(**kwargs_match).id
        db_ase.update(id=row_id, atoms=atoms, data=atoms.info, **kwargs_store)
    else:
        raise RuntimeError("More than one structure found in database.")

# -------------------------------------------------------------------------------------
# WRITE ATOMS LIST TO DB
# -------------------------------------------------------------------------------------

def write_atoms_list_to_db(
    atoms_list: list,
    db_ase: Database,
    keys_store: list = [],
    keys_match: list = None,
    deepcopy_all: bool = True,
    fill_stress: bool = False,
    fill_magmom: bool = False,
    no_constraints: bool = False,
    match_kwargs: bool = False,
    use_tqdm: bool = True,
    **kwargs: dict,
):
    """
    Write list of ase Atoms to ase database.
    """
    # Monitor progress with tqdm.
    if use_tqdm is True:
        atoms_list = tqdm(atoms_list, desc="Writing atoms to ASE DB", ncols=120)
    # Write all atoms structures in a single transaction.
    with db_ase:
        for atoms in atoms_list:
            write_atoms_to_db(
                atoms=atoms,
                db_ase=db_ase,
                keys_store=keys_store,
                keys_match=keys_match,
                deepcopy_all=deepcopy_all,
                fill_stress=fill_stress,
                fill_magmom=fill_magmom,
                no_constraints=no_constraints,
                match_kwargs=match_kwargs,
                **kwargs,
            )

# -------------------------------------------------------------------------------------
# GET ATOMS LIST FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_list_from_db(
    db_ase: Database,
    selection: str = "",
    **kwargs,
) -> list:
    """
    Get list of ase Atoms from ase database.
    """
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection, **kwargs)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms.info.update(atoms_row.key_value_pairs)
        atoms_list.append(atoms)
    return atoms_list

# -------------------------------------------------------------------------------------
# GET ATOMS FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_from_db(
    db_ase: Database,
    selection: str = "",
    none_ok: bool = False,
    **kwargs,
) -> Atoms:
    """
    Get ase Atoms from ase database.
    """
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection, **kwargs)
    if len(atoms_list) < 1:
        if none_ok is True:
            return None
        raise RuntimeError("No atoms structure found in database.")
    elif len(atoms_list) > 1:
        raise RuntimeError("More than one atoms structure found in database.")
    return atoms_list[0]

# -------------------------------------------------------------------------------------
# GET NAMES FROM DB
# -------------------------------------------------------------------------------------

def get_names_from_db(
    db_ase: Database,
    selection: str = "",
    unique: bool = True,
    **kwargs,
) -> Atoms:
    """
    Get names of structures in database.
    """
    names = [
        row.key_value_pairs["name"]
        for row in db_ase.select(selection=selection, **kwargs, include_data=False)
    ]
    if unique is True:
        names = list({key: None for key in names}.keys())
    return names

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
