# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

from arkimede.workflow.recipes import search_TS_from_atoms_IS_and_FS

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Perform a transition state (TS) search calculation starting from the atoms
    # objects of the initial state (IS) and final state (FS) of a reaction step.
    # First, the IS and FS are relaxed, then a NEB calculation is run to find a TS
    # guess structure, and finally a single-image TS-search calculation is run
    # starting from the TS guess structure. After the TS-search, the TS structure is
    # relaxed following the two directions of the TS mode and the connectivity of the
    # obtained structures is checked against the original IS and FS structures (to
    # check if the identified TS structure corresponds to the starting reaction step).

    # Calculations parameters.
    calculation_TS = "sella-ba" # neb | dimer | climbfixint | sella | sella-ba
    neb_steps = 0 # Number of NEB steps to run before the TS-search.
    n_restarts_TSsearch = 0 # Number of restarts in case the TS-search fails.
    use_hessian_function = True # Set function to get the Hessian when using Sella.
    save_trajs = True # Save trajectory files of the calculations.
    directory = "trajectories" # Directory to save the trajectory files.

    # Parameters for the relaxation of the IS and FS of the reaction.
    run_relax_kwargs = {
        "fmax": 0.05, # Forces convergence threshold [eV/Å].
        "max_steps": 0, # Maximum number of steps in the relax calculation.
        "save_trajs": save_trajs,
        "directory": directory,
    }
    # Parameters for the NEB calculation to identify the TS or a TS guess structure.
    run_NEB_kwargs = {
        "fmax": 0.05, # Forces convergence threshold [eV/Å].
        "max_steps": neb_steps, # Maximum number of steps in the NEB calculation.
        "n_images": 10, # Number of images in the NEB calculation.
        "save_trajs": save_trajs,
        "directory": directory,
    }
    # Parameters for the single-image TS-search calculation to identify the TS.
    run_TSsearch_kwargs = {
        "fmax": 0.05, # Forces convergence threshold [eV/Å].
        "max_steps": 500, # Maximum number of steps in the TS-search calculation.
        "max_forcecalls": 1000, # Maximum number of force calls.
        "save_trajs": save_trajs,
        "directory": directory,
    }
    # Parameters for calculating the approximate Hessian for TS-search with Sella.
    hessian_kwargs = {
        "indices_vib": "adsorbate", # Indices of atoms to displace to get the Hessian.
        "save_trajs": save_trajs,
        "directory": directory,
    }
    # Parameters for checking the connectivity of the IS and FS structures.
    check_IS_FS_kwargs = {
        "method": "vibrations", # Method to find the TS mode after the TS-search.
        "indices_vib": "adsorbate", # Indices of atoms to displace to get the Hessian.
        "indices_check": "adsorbate", # Indices of atoms to match the connectivity.
        "save_trajs": save_trajs,
        "directory": directory,
    }

    # Here we get the IS and FS atoms objects from a database.
    from ase.db import connect
    from arkimede.databases import get_atoms_from_db
    name = "Rh(111)+Pt1_H2O*_to_OH*+H*_top008_to_3fh001+3fh014"
    db_inp = connect(name=f"../00_databases/SAA_DFT_reactions.db")
    atoms_IS = get_atoms_from_db(db_ase=db_inp, name=name, image="IS")
    atoms_FS = get_atoms_from_db(db_ase=db_inp, name=name, image="FS")
    
    # Some information on the structures must be added to the atoms objects.
    indices_ads = [16, 17, 18] # Indices of adsorbate atoms.
    bonds_TS = [[16, 17, "break"]] # List of bonds to break or form in TS-search.
    atoms_IS.info = {"name": name, "indices_ads": indices_ads, "bonds_TS": bonds_TS}
    atoms_FS.info = {"name": name, "indices_ads": indices_ads, "bonds_TS": bonds_TS}
    
    # Here we get the ASE calculator (eSEN-OAM machine-learning potential).
    from mlps_finetuning.calculators import get_calculator
    calc = get_calculator(calc_name="OCP_eSEN", logfile="log.txt")

    # Get function to calculate the approximate Hessian for TS-search with Sella.
    if use_hessian_function is True:
        from arkimede.workflow.sella import get_hessian_from_vibrations_fun
        hessian_function = get_hessian_from_vibrations_fun(**hessian_kwargs)
        run_TSsearch_kwargs["opt_kwargs"] = {"hessian_function": hessian_function}

    # Run the TS-search workflow.
    atoms_TS = search_TS_from_atoms_IS_and_FS(
        atoms_IS=atoms_IS,
        atoms_FS=atoms_FS,
        calc=calc,
        db_out=None,
        calculation_TS=calculation_TS,
        run_relax_kwargs=run_relax_kwargs,
        run_NEB_kwargs=run_NEB_kwargs,
        run_TSsearch_kwargs=run_TSsearch_kwargs,
        check_IS_FS_kwargs=check_IS_FS_kwargs,
        n_restarts_TSsearch=n_restarts_TSsearch,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    # Run script and measure execution time.
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print(f"Execution time = {time_stop - time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------