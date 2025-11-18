# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------
# SEARCH TS FROM ATOMS IS AND FS
# -------------------------------------------------------------------------------------

def search_TS_from_atoms_IS_and_FS(
    atoms_IS: Atoms,
    atoms_FS: Atoms,
    calc: Calculator,
    db_out: Database,
    calculation_TS: str = "sella",
    kwargs_relax: dict = {},
    kwargs_NEB: dict = {},
    kwargs_TSsearch: dict = {},
    kwargs_check: dict = {},
):
    """
    Run TS-search calculation from IS and FS atoms.
    """
    # Get and print name.
    name = atoms_IS.info["name"]
    print_title(name)
    # Relax initial and final states.
    for atoms, image in zip((atoms_IS, atoms_FS), ("IS", "FS")):
        kwargs = {"name": name, "image": image, "calculation": "relax"}
        kwargs.update(kwargs_relax.pop("kwargs_db", {}))
        if get_atoms_from_db(db_ase=db_out, none_ok=True, **kwargs) is None:
            # Run relax calculation.
            print_subtitle(f"{image} relax calculation")
            run_calculation(
                atoms=atoms,
                calc=calc,
                calculation="relax",
                **kwargs_relax,
            )
            # Write atoms to database.
            kwargs["status"] = atoms.info["status"]
            write_atoms_to_db(db_ase=db_out, atoms=atoms, **kwargs)
        else:
            # Get atoms from database.
            atoms = get_atoms_from_db(db_ase=db_out, **kwargs)
    # NEB calculation.
    kwargs = {"name": name, "image": "TS", "calculation": "neb"}
    kwargs.update(kwargs_NEB.pop("kwargs_db", {}))
    if get_atoms_from_db(db_ase=db_out, none_ok=True, **kwargs) is None:
        # Get interpolated images.
        images = get_idpp_interpolated_images(
            atoms_IS=atoms_IS,
            atoms_FS=atoms_FS,
            n_images=kwargs_NEB.pop("n_images", 10),
        )
        # Run NEB calculation.
        print_subtitle("NEB calculation")
        run_calculation(
            atoms=images,
            calc=calc,
            calculation="neb",
            **kwargs_NEB,
        )
        # Write atoms to database.
        no_IS_or_FS = False if calculation_TS == "neb" else True
        atoms_TS = get_atoms_TS_from_images_neb(images=images, no_IS_or_FS=no_IS_or_FS)
        kwargs["status"] = atoms_TS.info["status"]
        write_atoms_to_db(db_ase=db_out, atoms=atoms_TS, **kwargs)
    else:
        # Get atoms from database.
        atoms_TS = get_atoms_from_db(db_ase=db_out, **kwargs)
    # Return if TS-search calculation is NEB.
    if calculation_TS == "neb":
        return atoms_TS
    # TS-search calculation.
    kwargs = {"name": name, "image": "TS", "calculation": calculation_TS}
    kwargs.update(kwargs_TSsearch.pop("kwargs_db", {}))
    if get_atoms_from_db(db_ase=db_out, none_ok=True, **kwargs) is None:
        # Run TS-search calculation.
        print_subtitle(f"TS-search {calculation_TS} calculation")
        run_calculation(
            atoms=atoms_TS,
            calculation=calculation_TS,
            bonds_TS=atoms_TS.info["bonds_TS"],
            calc=calc,
            **kwargs_TSsearch,
        )
        # Check if the atoms has desorbed.
        if atoms_TS.info["status"] == "finished":
            check = check_atoms_not_desorbed(atoms=atoms_TS)
            if check is False:
                atoms_TS.info["status"] = "desorbed"
        # Check if the TS atoms relax into original IS and FS.
        if atoms_TS.info["status"] == "finished":
            print_subtitle("Check-TS calculations")
            check = check_TS_relax_into_IS_and_FS(
                atoms_TS=atoms_TS,
                atoms_IS=atoms_IS,
                atoms_FS=atoms_FS,
                calc=calc,
                **kwargs_check,
            )
            if check is False:
                atoms_TS.info["status"] = "wrong"
        # Write atoms to database.
        kwargs["status"] = atoms_TS.info["status"]
        write_atoms_to_db(db_ase=db_out, atoms=atoms_TS, **kwargs)
        # Print TS-search status.
        print_subtitle("Status: " + atoms_TS.info["status"], newline_after=True)
    else:
        # Get atoms from database.
        atoms_TS = get_atoms_from_db(db_ase=db_out, **kwargs)
        # Print TS-search status.
        print_subtitle("Status: " + atoms_TS.info["status"], newline_after=True)
    # Return atoms.
    return atoms_TS

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------