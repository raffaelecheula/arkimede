# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import yaml
from ase.db import connect

from arkimede.reports import (
    calculations_report,
    plot_success_curves,
    plot_status_data,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Parameters.
    calc_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN | CHGNet_MP | MACE_MPA0 | FAIRChem_UMA
    db_name = f"../00_databases/MLP_relax_{calc_name}.db"
    percentages = True
    
    # Lists of calculation and status names.
    calculation_list = ["dimer", "climbfixint", "sella", "sella-ba"]
    status_list = ["finished", "failed", "desorbed", "wrong"]

    # Prepare ASE database.
    db_ase = connect(name=db_name)

    # Calculation report.
    results_dict, success_steps_dict = calculations_report(
        calculation_list=calculation_list,
        status_list=status_list,
        db_ase=db_ase,
        max_steps=10000 + 10,
        not_in_status_list="failed",
        percentages=percentages,
        print_report=True,
    )
    # Plot success curves.
    plot_success_curves(
        success_steps_dict=success_steps_dict,
        xlabel="Number of MLP single-points [-]",
        ylabel="Successful TS-search calculations [%]",
        xmed=1100,
        xmax=10000,
        ymax=100 if percentages is True else None,
        filename=f"success_curves_{calc_name}.png",
    )
    # Plot status data.
    plot_status_data(
        calculation_list=calculation_list,
        results_dict=results_dict,
        xlabel="TS-search method",
        ylabel="TS-search calculations [%]",
        ymax=100 if percentages is True else None,
        filename=f"status_data_{calc_name}.png",
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
