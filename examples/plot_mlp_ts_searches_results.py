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
    calc_name = "OCP_eSEN" # OCP_eSCN | OCP_eSEN
    db_name = f"databases/MLP_relax_{calc_name}.db"
    
    # Lists of calculation and status names.
    calculation_list = ["dimer", "climbfixint", "sella", "sella-ba"]
    status_list = ["finished", "desorbed", "failed", "wrong"]

    # Prepare ase database.
    db_ase = connect(name=db_name)

    # Read excluded calculations.
    excluded = yaml.safe_load(open("excluded.yaml", "r"))

    # Calculation report.
    results_dict, success_steps_dict = calculations_report(
        calculation_list=calculation_list,
        status_list=status_list,
        db_ase=db_ase,
        max_steps=10000 + 10,
        not_in_status_list="failed",
        percentages=True,
        print_report=True,
        excluded=excluded,
    )
    # Plot success curves.
    plot_success_curves(
        success_steps_dict=success_steps_dict,
        xlabel="MLP single-point calculations [-]",
        ylabel="Successful TS-search calculations [%]",
        filename="plot_success_curves.png",
        xmed=1100,
        xmax=10000,
        ymax=100,
    )
    # Plot status data.
    plot_status_data(
        calculation_list=calculation_list,
        results_dict=results_dict,
        xlabel="Methods",
        ylabel="TS-search calculations [%]",
        ymax=100,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
