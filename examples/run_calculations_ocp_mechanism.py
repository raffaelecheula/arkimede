# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.db import connect
from arkimede.workflow.reaction_workflow import run_ase_calculations_mechanism
from arkimede.utilities import get_atoms_list_from_db_metadata, read_step_actlearn

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Active learning step.
    filename_actlearn = "actlearn.yaml"
    step_actlearn = read_step_actlearn(filename=filename_actlearn)

    # Name of ase database containing the unrelaxed structures to read.
    db_init_name = "databases/init.db"

    # Select atoms from the ase database.
    selection = ""

    # Name of ase database to store the results of the calculations.
    db_ase_name = f"databases/ocp_{step_actlearn:02d}.db"
    db_ase_append = True

    # Keywords used to match structures in the db and to be stored in db columns.
    keys_match = ["name", "calculation"]
    keys_store = ["name", "calculation", "status", "name_ref", "species"]

    # Calculations parameters.
    n_images_neb = 10
    search_TS = "climbbonds" # dimer | climbbonds | climbfixint | sella
    check_modified_neb = False
    check_modified_TS_search = True

    # Kwargs parameters of calculations.
    kwargs_relax = {
        "fmax": 0.05,
        "max_steps": 1000,
    }
    kwargs_neb = {
        "fmax": 0.05,
        "max_steps": 100,
        "k_neb": 1.0,
        "use_OCPdyNEB": True,
    }
    kwargs_ts_search = {
        "fmax": 0.05,
        "max_steps": 1000,
        "min_steps": 10,
        "max_forcecalls": 1000,
    }
    kwargs_check = {
        "fmax": 0.05,
        "max_steps": 1000,
    }

    # Save trajectories and write images.
    save_trajs = False
    write_images = False
    basedir_trajs = "calculations/ocp"

    mlp_name = "ocp"

    if mlp_name == "ocp":
        # OCPmodels ase calculator.
        from mlps_finetuning.ocp import OCPCalculator
        calc = OCPCalculator(
            model_name="GemNet-OC-S2EFS-OC20+OC22",
            local_cache="pretrained_models",
            cpu=False,
        )
    elif mlp_name == "chgnet":
        # CHGNetCalculator ase calculator.
        from mlps_finetuning.chgnet import CHGNetCalculator
        calc = CHGNetCalculator(use_device=None)
    elif mlp_name == "fairchem":
        # FairChem ase calculator.
        from mlps_finetuning.fairchem import FAIRChemCalculator, pretrained_mlip
        predict_unit = pretrained_mlip.get_predict_unit(
            model_name="uma-s-1p1",
            device="cuda",
            cache_dir="pretrained_models",
        )
        calc = FAIRChemCalculator(predict_unit=predict_unit, task_name="oc20")
    
    # ---------------------------------------------------------------------------------
    # RUN ASE CALCULATIONS
    # ---------------------------------------------------------------------------------

    # Initialize ase init database.
    db_init = connect(name=db_init_name)

    # Get lists of atoms from database.
    atoms_clean_tot = get_atoms_list_from_db_metadata(
        db_ase=db_init,
        selection=selection,
        metadata_key="clean",
    )
    atoms_ads_tot = get_atoms_list_from_db_metadata(
        db_ase=db_init,
        selection=selection,
        metadata_key="adsorbate",
    )
    atoms_neb_tot = get_atoms_list_from_db_metadata(
        db_ase=db_init,
        selection=selection,
        metadata_key="reaction",
    )
    
    # Initialize ase dft database.
    db_ase = connect(name=db_ase_name, append=db_ase_append)

    # Run the calculations.
    run_ase_calculations_mechanism(
        atoms_clean_tot=atoms_clean_tot,
        atoms_ads_tot=[],
        atoms_neb_tot=atoms_neb_tot,
        calc=calc,
        db_ase=db_ase,
        n_images_neb=n_images_neb,
        search_TS=search_TS,
        save_trajs=save_trajs,
        write_images=write_images,
        basedir_trajs=basedir_trajs,
        check_modified_neb=check_modified_neb,
        check_modified_TS_search=check_modified_TS_search,
        keys_match=keys_match,
        keys_store=keys_store,
        kwargs_relax=kwargs_relax,
        kwargs_neb=kwargs_neb,
        kwargs_ts_search=kwargs_ts_search,
        kwargs_check=kwargs_check,
    )

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
