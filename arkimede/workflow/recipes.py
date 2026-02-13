# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.db.core import Database

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
from arkimede.utilities import (
    print_title,
    print_subtitle,
    get_indices_from_name,
    get_key_paths,
    set_key_path_value,
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
    n_restarts_TSsearch: int = 0,
    check_atoms_desorbed: bool = True,
    rattle_kwargs: dict = {},
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
        run_TSsearch_and_check(
            atoms_TS=atoms_TS,
            atoms_IS=atoms_IS,
            atoms_FS=atoms_FS,
            calc=calc,
            calculation_TS=calculation_TS,
            n_restarts_TSsearch=n_restarts_TSsearch,
            run_TSsearch_kwargs=run_TSsearch_kwargs,
            check_atoms_desorbed=check_atoms_desorbed,
            check_IS_FS_kwargs=check_IS_FS_kwargs,
            rattle_kwargs=rattle_kwargs,
        )
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
# RUN TSSEARCH AND CHECK
# -------------------------------------------------------------------------------------

def run_TSsearch_and_check(
    atoms_TS: Atoms,
    atoms_IS: Atoms,
    atoms_FS: Atoms,
    calc: Calculator,
    calculation_TS: str = "sella",
    n_restarts_TSsearch: int = 0,
    run_TSsearch_kwargs: dict = {},
    check_atoms_desorbed: bool = True,
    check_IS_FS_kwargs: dict = {},
    rattle_kwargs: dict = {},
) -> None:
    """
    Run TS-search calculation and check results.
    """
    # Run TS-search calculation.
    number_generator = None
    atoms_TS_zero = atoms_TS.copy()
    run_TSsearch_kwargs_zero = run_TSsearch_kwargs.copy()
    for ii in range(n_restarts_TSsearch + 1):
        print_subtitle(f"TS-search {calculation_TS} calculation")
        run_calculation(
            atoms=atoms_TS,
            calculation=calculation_TS,
            bonds_TS=atoms_TS.info["bonds_TS"],
            calc=calc,
            **run_TSsearch_kwargs,
        )
        # Check if the atoms has desorbed.
        if check_atoms_desorbed and atoms_TS.info["status"] == "finished":
            check = check_atoms_not_desorbed(atoms=atoms_TS)
            if check is False:
                atoms_TS.info["status"] = "desorbed"
        # Check if the TS atoms relax into original IS and FS.
        if check_IS_FS_kwargs.get("method") and atoms_TS.info["status"] == "finished":
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
        # Break if TS-search is successful.
        if atoms_TS.info["status"] == "finished":
            break
        elif ii < n_restarts_TSsearch:
            # Rattle atoms and parameters for next restart.
            print_subtitle("Restart")
            atoms_TS.positions = atoms_TS_zero.positions.copy()
            run_TSsearch_kwargs = run_TSsearch_kwargs_zero.copy()
            number_generator = rattle_atoms_and_parameters(
                atoms=atoms_TS,
                parameters=run_TSsearch_kwargs,
                calculation_TS=calculation_TS,
                number_generator=number_generator,
                **rattle_kwargs,
            )

# -------------------------------------------------------------------------------------
# RATTLE ATOMS AND PARAMETERS
# -------------------------------------------------------------------------------------

def rattle_atoms_and_parameters(
    atoms: Atoms,
    parameters: dict,
    calculation_TS: str,
    indices: list = "adsorbate",
    scale_transl: float = 0.5,
    scale_rattle: float = 0.2,
    parameters_ranges: dict = None,
    number_generator: object = None,
    sampling: str = "quasirandom",
):
    """
    Modify TS atoms by performing a random translation and rattle of the atoms.
    """
    # Get indices of atoms to rattle.
    indices = get_indices_from_name(atoms=atoms, indices=indices)
    # Random translation and rattle.
    positions_ads = atoms.positions[indices]
    transl = np.random.normal(scale=scale_transl, size=3)
    rattle = np.random.normal(scale=scale_rattle, size=positions_ads.shape)
    atoms.positions[indices] = positions_ads + transl + rattle
    # Get default parameters ranges.
    if parameters_ranges is None:
        parameters_ranges = default_parameters_ranges(calculation=calculation_TS)
    # Perturb the parameters of the TS-search calculation.
    number_generator = rattle_parameters(
        parameters=parameters,
        parameters_ranges=parameters_ranges,
        number_generator=number_generator,
    )
    # Return number generator.
    return number_generator

# -------------------------------------------------------------------------------------
# RATTLE PARAMETERS
# -------------------------------------------------------------------------------------

def rattle_parameters(
    parameters: dict,
    parameters_ranges: dict = {},
    sampling: str = "quasirandom",
    number_generator: object = None,
    seed: int = None,
):
    """
    Modify the parameters of the calculation by randomly sampling new values
    from given ranges.
    """
    key_paths, values = get_key_paths(dictionary=parameters_ranges)
    # Gererate sampling numbers.
    if sampling == "quasirandom":
        from scipy.stats.qmc import Sobol
        if number_generator is None:
            number_generator = Sobol(d=len(key_paths), scramble=True, seed=seed)
        numbers = number_generator.random(n=1)[0]
    else:
        if number_generator is None:
            number_generator = np.random.default_rng(seed=seed)
        numbers = number_generator.random(size=len(key_paths))
    # Set new parameters values.
    for num, path, (low, high) in zip(numbers, key_paths, values):
        if low > 0 and high / low > 5:
            value = 10 ** (np.log10(high) * num + np.log10(low) * (1 - num))
        else:
            value = high * num + low * (1 - num)
        if isinstance(low, int):
            value = int(round(value))
        set_key_path_value(dictionary=parameters, path=path, value=value)
    # Return number generator.
    return number_generator

# -------------------------------------------------------------------------------------
# DEFAULT PARAMETERS RANGES
# -------------------------------------------------------------------------------------

def default_parameters_ranges(
    calculation: str,
):
    """
    Return default parameters ranges.
    """
    if calculation == "dimer":
        return {
            "control_kwargs": {
                "trial_trans_step": (0.001, 0.10),
                "dimer_separation": (0.001, 0.10),
                "maximum_translation": (0.01, 0.10),
            },
        }
    elif calculation == "climbfixint":
        return {
            "opt_kwargs": {
                "maxstep": (0.01, 0.10),
            },
            "optB_kwargs": {
                "maxstep": (0.01, 0.10),
            },
        }
    elif calculation == "sella":
        return {
            "opt_kwargs": {
                "eta": (0.001, 0.01),
                "gamma": (0.01, 0.10),
                "delta0": (0.01, 0.10),
            },
        }
    elif calculation == "sella-ba":
        return {
            "opt_kwargs": {
                "eta": (0.001, 0.01),
                "gamma": (0.01, 0.10),
                "delta0": (0.01, 0.10),
            },
            "dot_prod_thr": (0.0, 0.8),
        }

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
    n_restarts_TSsearch: int = 0,
    db_inp_kwargs: dict = {},
    basedir: str = "",
):
    """
    Run TS-search calculation from reaction name and input database.
    """
    # Get IS and FS atoms.
    db_inp = connect(name=os.path.join(basedir, db_inp_name))
    db_inp_kwargs_IS = {"name": name, "image": "IS", **db_inp_kwargs}
    db_inp_kwargs_FS = {"name": name, "image": "FS", **db_inp_kwargs}
    atoms_IS = get_atoms_from_db(db_ase=db_inp, **db_inp_kwargs_IS)
    atoms_FS = get_atoms_from_db(db_ase=db_inp, **db_inp_kwargs_FS)
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
        n_restarts_TSsearch=n_restarts_TSsearch,
    )

# -------------------------------------------------------------------------------------
# SEARCH TS FROM NAMES
# -------------------------------------------------------------------------------------

def search_TS_from_names_list(
    name_list: list,
    db_inp_name: str,
    db_out_name: str,
    calc_name: str,
    calc_kwargs: dict = {},
    calculation_TS: str = "sella",
    run_relax_kwargs: dict = {},
    run_NEB_kwargs: dict = {},
    run_TSsearch_kwargs: dict = {},
    check_IS_FS_kwargs: dict = {},
    n_restarts_TSsearch: int = 0,
    db_inp_kwargs: dict = {},
    basedir: str = "",
):
    """
    Run TS-search calculation from reaction name and input database.
    """
    # Input database.
    db_inp = connect(name=os.path.join(basedir, db_inp_name))
    # Get calculator.
    calc = get_calculator(calc_name=calc_name, **calc_kwargs)
    # Results database.
    db_out = connect(name=os.path.join(basedir, db_out_name))
    # Run TS-search calculations.
    for name in name_list:
        # Get IS and FS atoms.
        db_inp_kwargs_IS = {"name": name, "image": "IS", **db_inp_kwargs}
        db_inp_kwargs_FS = {"name": name, "image": "FS", **db_inp_kwargs}
        atoms_IS = get_atoms_from_db(db_ase=db_inp, **db_inp_kwargs_IS)
        atoms_FS = get_atoms_from_db(db_ase=db_inp, **db_inp_kwargs_FS)
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
            n_restarts_TSsearch=n_restarts_TSsearch,
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