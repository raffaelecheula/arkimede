# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from ase.db import connect
from arkimede.workflow.reaction_workflow import run_ase_calculations_mechanism
from arkimede.workflow.utilities import get_atoms_list_from_db
from arkimede.ocp.ocp_utils import get_checkpoint_path
from arkimede.ocp.ocp_calc import OCPCalculatorCounter

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():

    # Name of ase database containing the unrelaxed structures to read.
    db_init_name = "database_init.db"

    # Select atoms from the ase database.
    selection = ""

    # Name of ase database to store the results of the calculations.
    db_ase_name = "database_ocp_2.db"
    db_ase_append = True

    # Calculations parameters.
    fmax = 0.05
    steps_max_relax = 1000
    steps_max_neb = 10
    steps_max_ts_search = 1000
    n_images_neb = 10
    search_TS = "dimer" # dimer | climbbonds | climbfixint | sella

    # Save trajectories and write images.
    save_trajs = False
    write_images = False
    basedir_trajs = "calculations_ocp"

    # OCPmodels ase calculator.
    checkpoint_key = 'GemNet-OC OC20+OC22' # 'EquiformerV2 (31M) All+MD'
    checkpoint_path = get_checkpoint_path(checkpoint_key=checkpoint_key)
    calc = OCPCalculatorCounter(checkpoint_path=checkpoint_path, cpu=False)
    
    # -----------------------------------------------------------------------------
    # RUN ASE CALCULATIONS
    # -----------------------------------------------------------------------------

    # Initialize ase init database.
    db_init = connect(name=db_init_name)

    # Get lists of atoms from database.
    atoms_clean_tot = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="clean",
    )
    atoms_ads_tot = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="adsorbates",
    )
    atoms_neb_tot = get_atoms_list_from_db(
        db_ase=db_init,
        selection=selection,
        structure_type="reactions",
    )
    
    # Initialize ase dft database.
    db_ase = connect(name=db_ase_name, append=db_ase_append)

    run_ase_calculations_mechanism(
        atoms_clean_tot=atoms_clean_tot,
        atoms_ads_tot=atoms_ads_tot,
        atoms_neb_tot=atoms_neb_tot,
        calc=calc,
        db_ase=db_ase,
        fmax=fmax,
        steps_max_relax=steps_max_relax,
        steps_max_neb=steps_max_neb,
        steps_max_ts_search=steps_max_ts_search,
        n_images_neb=n_images_neb,
        search_TS=search_TS,
        save_trajs=save_trajs,
        write_images=write_images,
        basedir_trajs=basedir_trajs,
    )

# -----------------------------------------------------------------------------
# IF NAME MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print("Execution time = {0:6.3} s".format(time_stop - time_start))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
