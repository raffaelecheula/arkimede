# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
from ase.db import connect
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.db.core import Database

from arkimede.utilities import print_title, print_subtitle
from arkimede.databases import get_atoms_from_db, write_atoms_to_db
from arkimede.workflow.calculations import run_calculation
from arkimede.workflow.checks import (
    check_atoms_not_desorbed,
    check_TS_relax_into_IS_and_FS,
)
from arkimede.transtates import (
    get_idpp_interpolated_images,
    get_atoms_TS_from_images_neb,
)
from mlps_finetuning.calculators import get_calculator

# -------------------------------------------------------------------------------------
# SEARCH TS FROM ATOMS IS AND FS
# -------------------------------------------------------------------------------------

def search_TS_from_atoms_IS_and_FS(
    atoms_IS: Atoms,
    atoms_FS: Atoms,
    calc: Calculator,
    db_out: Database,
    calculation_TS: str = "sella",
    run_relax_kwargs: dict = {},
    run_NEB_kwargs: dict = {},
    run_TSsearch_kwargs: dict = {},
    check_IS_FS_kwargs: dict = {},
):
    """
    Run TS-search calculation from IS and FS atoms.
    """
    # Get and print name.
    name = atoms_IS.info["name"]
    print_title(name)
    # Relax initial and final states.
    for atoms, image in zip((atoms_IS, atoms_FS), ("IS", "FS")):
        run_relax_kwargs_ii = run_relax_kwargs.copy()
        db_kwargs = {"name": name, "image": image, "calculation": "relax"}
        db_kwargs.update(run_relax_kwargs_ii.pop("db_out_kwargs", {}))
        if get_atoms_from_db(db_ase=db_out, none_ok=True, **db_kwargs) is None:
            # Run relax calculation.
            print_subtitle(f"{image} relax calculation")
            run_calculation(
                atoms=atoms,
                calc=calc,
                calculation="relax",
                **run_relax_kwargs_ii,
            )
            # Write atoms to database.
            db_kwargs["status"] = atoms.info["status"]
            write_atoms_to_db(db_ase=db_out, atoms=atoms, **db_kwargs)
        else:
            # Get atoms from database.
            atoms = get_atoms_from_db(db_ase=db_out, **db_kwargs)
    # NEB calculation.
    run_NEB_kwargs = run_NEB_kwargs.copy()
    db_kwargs = {"name": name, "image": "TS", "calculation": "neb"}
    db_kwargs.update(run_NEB_kwargs.pop("db_out_kwargs", {}))
    if get_atoms_from_db(db_ase=db_out, none_ok=True, **db_kwargs) is None:
        # Get interpolated images.
        images = get_idpp_interpolated_images(
            atoms_IS=atoms_IS,
            atoms_FS=atoms_FS,
            n_images=run_NEB_kwargs.pop("n_images", 10),
        )
        # Run NEB calculation.
        print_subtitle("NEB calculation")
        run_calculation(
            atoms=images,
            calc=calc,
            calculation="neb",
            **run_NEB_kwargs,
        )
        # Write atoms to database.
        no_IS_or_FS = False if calculation_TS == "neb" else True
        atoms_TS = get_atoms_TS_from_images_neb(images=images, no_IS_or_FS=no_IS_or_FS)
        db_kwargs["status"] = atoms_TS.info["status"]
        write_atoms_to_db(db_ase=db_out, atoms=atoms_TS, **db_kwargs)
    else:
        # Get atoms from database.
        atoms_TS = get_atoms_from_db(db_ase=db_out, **db_kwargs)
    # Return if TS-search calculation is NEB.
    if calculation_TS == "neb":
        return atoms_TS
    # TS-search calculation.
    run_TSsearch_kwargs = run_TSsearch_kwargs.copy()
    db_kwargs = {"name": name, "image": "TS", "calculation": calculation_TS}
    db_kwargs.update(run_TSsearch_kwargs.pop("db_out_kwargs", {}))
    if get_atoms_from_db(db_ase=db_out, none_ok=True, **db_kwargs) is None:
        # Run TS-search calculation.
        print_subtitle(f"TS-search {calculation_TS} calculation")
        run_calculation(
            atoms=atoms_TS,
            calculation=calculation_TS,
            bonds_TS=atoms_TS.info["bonds_TS"],
            calc=calc,
            **run_TSsearch_kwargs,
        )
        # Check if the atoms has desorbed.
        if atoms_TS.info["status"] == "finished":
            check = check_atoms_not_desorbed(atoms=atoms_TS)
            if check is False:
                atoms_TS.info["status"] = "desorbed"
        # Check if the TS atoms relax into original IS and FS.
        if check_IS_FS_kwargs["method"] and atoms_TS.info["status"] == "finished":
            print_subtitle("Check-TS calculations")
            check = check_TS_relax_into_IS_and_FS(
                atoms_TS=atoms_TS,
                atoms_IS=atoms_IS,
                atoms_FS=atoms_FS,
                calc=calc,
                **check_IS_FS_kwargs,
            )
            if check is False:
                atoms_TS.info["status"] = "wrong"
        # Write atoms to database.
        db_kwargs["status"] = atoms_TS.info["status"]
        write_atoms_to_db(db_ase=db_out, atoms=atoms_TS, **db_kwargs)
        # Print TS-search status.
        print_subtitle("Status: " + atoms_TS.info["status"], newline_after=True)
    else:
        # Get atoms from database.
        atoms_TS = get_atoms_from_db(db_ase=db_out, **db_kwargs)
        # Print TS-search status.
        print_subtitle("Status: " + atoms_TS.info["status"], newline_after=True)
    # Return atoms.
    return atoms_TS

# -------------------------------------------------------------------------------------
# SEARCH TS FROM NAMES
# -------------------------------------------------------------------------------------

def search_TS_from_names(
    name: str,
    db_inp_name: str,
    db_out_name: str,
    calc_name: str,
    calc_kwargs: dict = {},
    calculation_TS: str = "sella",
    run_relax_kwargs: dict = {},
    run_NEB_kwargs: dict = {},
    run_TSsearch_kwargs: dict = {},
    check_IS_FS_kwargs: dict = {},
    db_inp_kwargs: dict = {},
    basedir: str = "",
):
    """
    Run TS-search calculation from reaction name and input database.
    """
    # Get IS and FS atoms.
    db_inp = connect(name=os.path.join(basedir, db_inp_name))
    atoms_IS = get_atoms_from_db(db_ase=db_inp, name=name, image="IS", **db_inp_kwargs)
    atoms_FS = get_atoms_from_db(db_ase=db_inp, name=name, image="FS", **db_inp_kwargs)
    # Get calculator.
    calc = get_calculator(calc_name=calc_name, **calc_kwargs)
    # Results database.
    db_out = connect(name=os.path.join(basedir, db_out_name))
    # Run TS-search calculation from IS and FS atoms.
    search_TS_from_atoms_IS_and_FS(
        atoms_IS=atoms_IS,
        atoms_FS=atoms_FS,
        calc=calc,
        db_out=db_out,
        calculation_TS=calculation_TS,
        run_relax_kwargs=run_relax_kwargs,
        run_NEB_kwargs=run_NEB_kwargs,
        run_TSsearch_kwargs=run_TSsearch_kwargs,
        check_IS_FS_kwargs=check_IS_FS_kwargs,
    )

# -------------------------------------------------------------------------------------
# RUN CALCULATION FROM NAMES
# -------------------------------------------------------------------------------------

def run_calculation_from_names(
    name: str,
    db_inp_name: str,
    db_out_name: str,
    calculation: str,
    calc_name: str,
    calc_kwargs: dict = {},
    run_kwargs: dict = {},
    db_inp_kwargs: dict = {},
    db_out_kwargs: dict = {},
    basedir: str = "",
):
    """
    Run calculation from name and input database.
    """
    # Get IS and FS atoms.
    db_inp = connect(name=os.path.join(basedir, db_inp_name))
    atoms = get_atoms_from_db(db_ase=db_inp, name=name, **db_inp_kwargs)
    # Get calculator.
    calc = get_calculator(calc_name=calc_name, **calc_kwargs)
    # Run calculation.
    print_title(name)
    print_subtitle(f"{calculation} calculation")
    run_calculation(
        atoms=atoms,
        calc=calc,
        calculation=calculation,
        **run_kwargs,
    )
    # Print Calculation status.
    print_subtitle("Status: " + atoms.info["status"], newline_after=True)
    # Results database.
    db_out = connect(name=os.path.join(basedir, db_out_name))
    # Write atoms to database.
    db_out_kwargs = {**db_out_kwargs, "name": name, "calculation": calculation}
    db_out_kwargs["status"] = atoms.info["status"]
    write_atoms_to_db(db_ase=db_out, atoms=atoms, **db_out_kwargs)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------