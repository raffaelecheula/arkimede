# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
    
from arkimede.reports import (
    calculations_report,
    get_names_dict,
    get_colors_dict,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Parameters.
    percentages = True
    width = 0.19
    basedir = "../00_databases"
    # MLP calculator labels.
    calc_name_dict = {
        "CHGNet_MP": "CHGNet-0.3",
        "MACE_MPA0": "MACE-MPA",
        "OCP_eSCN": "eSCN-OC20",
        "OCP_eSEN": "eSEN-OAM",
        "FAIRChem_UMA": "UMA-M",
    }
    
    # Lists of calculation and status names.
    calculation_list = ["dimer", "climbfixint", "sella", "sella-ba"]
    status_list = ["finished", "failed", "desorbed", "wrong"]

    # Get results.
    finished = {}
    for calc_name in calc_name_dict:
        # Prepare ase database.
        db_name = f"{basedir}/MLP_relax_{calc_name}.db"
        db_ase = connect(name=db_name)
        # Calculation report.
        results_dict, success_steps_dict = calculations_report(
            calculation_list=calculation_list,
            status_list=status_list,
            db_ase=db_ase,
            max_steps=10000,
            not_in_status_list="failed",
            percentages=percentages,
            print_report=False,
        )
        finished[calc_name] = {
            key: results_dict[key]["finished"] for key in results_dict
        }
    
    # Plot calculators data.
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    # Update default names and colors dictionaries.
    names_dict = get_names_dict()
    colors_dict = get_colors_dict()
    # Plot bars.
    x_values = np.arange(len(calc_name_dict))
    for ii, calculation in enumerate(calculation_list):
        ax.bar(
            x_values + ii * width,
            [finished[calc_name][calculation] for calc_name in calc_name_dict],
            width=width,
            label=names_dict[calculation],
            color=colors_dict[calculation],
            edgecolor="black",
        )
    ax.set_xticks(x_values + width * (len(calculation_list) - 1) / 2)
    ax.set_xticklabels(calc_name_dict.values())
    # Set axes labels and y limit.
    ax.set_xlabel("MLP model")
    ax.set_ylabel("Successful TS-search calculations [%]")
    ax.set_ylim(0, 100 if percentages is True else None)
    ax.set_xlim(-0.22, len(calc_name_dict) - 0.22)
    # Add legend.
    ax.legend(
        loc="lower right",
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
    )
    # Save the figure.
    plt.savefig("success_all_MLPs.png")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
