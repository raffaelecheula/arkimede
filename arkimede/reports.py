# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ase.db.core import Database

from arkimede.utilities import print_title
from arkimede.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# GET NAMES DICT
# -------------------------------------------------------------------------------------

def get_names_dict(
    names_dict: dict = {},
) -> dict:
    """
    Get calculation names dictionary updating the default one.
    """
    return {
        "relax": "Relax",
        "neb": "NEB",
        "dimer": "Dimer",
        "climbfixint": "ARPESS",
        "sella": "Sella",
        "sella-ba": "BA-Sella",
        **names_dict,
    }

# -------------------------------------------------------------------------------------
# GET COLORS DICT
# -------------------------------------------------------------------------------------

def get_colors_dict(
    colors_dict: dict = {},
) -> dict:
    """
    Get calculation colors dictionary updating the default one.
    """
    return {
        "relax": "mediumblue",
        "neb": "mediumorchid",
        "dimer": "deepskyblue",
        "climbfixint": "crimson",
        "sella": "orange",
        "sella-ba": "forestgreen",
        **colors_dict,
    }

# -------------------------------------------------------------------------------------
# CALCULATIONS REPORT
# -------------------------------------------------------------------------------------

def calculations_report(
    calculation_list: list,
    status_list: list,
    db_ase: Database,
    max_steps: int,
    not_in_status_list: str = "other",
    excluded: list = [],
    percentages: bool = True,
    print_report: bool = True,
) -> dict:
    """
    Get dictionaries of status and success curve data from calculations in a db.
    """
    # Collect results data.
    results_dict = {}
    success_steps_dict = {}
    for calculation in calculation_list:
        # Read atoms from database.
        atoms_list = get_atoms_list_from_db(db_ase=db_ase, calculation=calculation)
        atoms_list = [
            atoms for atoms in atoms_list if atoms.info["name"] not in excluded
        ]
        # Substitute status if not in status_list.
        for atoms in atoms_list:
            if atoms.info["status"] not in status_list:
                atoms.info["status"] = not_in_status_list
        # Get results dictionary and success curve data.
        status_dict = {key: 0 for key in status_list}
        success_steps = np.zeros(max_steps)
        mult = 100 / len(atoms_list) if percentages and len(atoms_list) > 0 else 1
        for atoms in atoms_list:
            status_dict[atoms.info["status"]] += 1 * mult
            if atoms.info["status"] == "finished":
                success_steps[atoms.info["counter"]:] += 1 * mult
        success_steps_dict[calculation] = success_steps
        results_dict[calculation] = status_dict
        # Print report.
        if print_report is True:
            print_title(calculation, width=32)
            print(f"Number of calculations  : {len(atoms_list):6d} [-]")
            for key in status_list:
                if percentages is True:
                    print(f"Calculations {key:11s}: {status_dict[key]:6.2f} [%]")
                else:
                    print(f"Calculations {key:11s}: {status_dict[key]:6d} [-]")
    # Return dictionaries.
    return results_dict, success_steps_dict

# -------------------------------------------------------------------------------------
# PLOT SUCCESS CURVES
# -------------------------------------------------------------------------------------

def plot_success_curves(
    success_steps_dict: dict,
    axes: object = None,
    xlabel: str = "Number of steps [-]",
    ylabel: str = "Successful calculations [%]",
    title: str = None,
    xmed: int = 1100,
    xmax: int = 10000,
    ymax: float = 100,
    filename: str = "plot_success_curves.png",
    colors_dict: dict = {},
    names_dict: dict = {},
) -> object:
    """
    Plot success curves showing how many steps are required to the calculations
    to finish (and how many calculations have finished successfully).
    """
    # Prepare figure.
    if axes is None:
        fig_kwargs = {"dpi": 300, "gridspec_kw": {"width_ratios": (14, 1)}}
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.5, 4), **fig_kwargs)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.03)
    else:
        ax1, ax2 = axes
    # Update default names and colors dictionaries.
    names_dict = get_names_dict(names_dict=names_dict)
    colors_dict = get_colors_dict(colors_dict=colors_dict)
    # Plot success curve.
    x_data = np.arange(xmax + 10)
    for ii, (calculation, success_steps) in enumerate(success_steps_dict.items()):
        label = names_dict[calculation]
        for ax in (ax1, ax2):
            ax.plot(x_data, success_steps, color=colors_dict[calculation], label=label)
    # Customize axes.
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)
    ax1.set_xlim(0, xmed)
    ax2.set_xlim(xmax - 10, xmax + 10)
    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.set_yticks([])
    ax2.set_xticks([xmax])
    ax2.set_xticklabels([str(xmax)])
    # Draw diagonal break marks.
    px, py = 0.05, 0.01
    kwargs = {"transform": ax2.transAxes, "color": "k", "clip_on": False, "lw": 1}
    ax2.plot((-0.2 - px, -0.2 + px), (-py, +py), **kwargs)
    ax2.plot((-0.2 - px, -0.2 + px), (1 - py, 1 + py), **kwargs)
    ax2.plot((-px, +px), (-py, +py), **kwargs)
    ax2.plot((-px, +px), (1 - py, 1 + py), **kwargs)
    # Set axes labels and legend.
    ax1.set_xlabel(" " * 10 + xlabel)
    ax1.set_ylabel(ylabel)
    if title is not None:
        ax1.set_title(title)
    ax2.legend(
        loc="lower right",
        frameon=True,
        edgecolor="black",
    )
    # Save the figure.
    if filename is not None:
        plt.savefig(filename)
    # Return the figure object.
    return fig

# -------------------------------------------------------------------------------------
# PLOT STATUS DATA
# -------------------------------------------------------------------------------------

def plot_status_data(
    calculation_list: list,
    results_dict: dict,
    ax: object = None,
    xlabel: str = "Methods",
    ylabel: str = "Calculations [-]",
    title: str = None,
    ymax: float = 100,
    filename: str = "plot_status_data.png",
    colors_dict: dict = {},
    names_dict: dict = {},
) -> object:
    """
    Plot status data of calculations.
    """
    # Prepare figure.
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        fig.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.95)
    # Update default names and colors dictionaries.
    names_dict = get_names_dict(names_dict=names_dict)
    colors_dict = get_colors_dict(colors_dict=colors_dict)
    # Define hatch patterns for different statuses.
    hatch_dict = {
        "finished": "",
        "unfinished": "---",
        "failed": "////",
        "desorbed": "|||",
        "wrong": "ooo",
    }
    # Plot status data.
    bottom = np.zeros(len(calculation_list))
    status_list = list({key: None for dd in results_dict.values() for key in dd})
    for status in status_list:
        xx = [names_dict[calculation] for calculation in calculation_list]
        hh = [results_dict[calculation][status] for calculation in calculation_list]
        cc = [colors_dict[calculation] for calculation in calculation_list]
        plt.bar(
            x=xx,
            height=hh,
            bottom=bottom,
            color=cc if status == "finished" else "white",
            edgecolor=cc,
            hatch=hatch_dict[status],
        )
        plt.bar(
            x=xx,
            height=hh,
            bottom=bottom,
            color="none",
            edgecolor="black",
        )
        plt.bar(
            x=-1,
            height=0,
            label=status,
            color="darkgrey" if status == "finished" else "white",
            edgecolor="black",
            hatch=hatch_dict[status],
        )
        bottom += hh
    # Set axes labels and y limit.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(calculation_list)-0.5)
    ax.set_ylim(0, ymax)
    # Set title.
    if title is not None:
        ax.set_title(title)
    # Add legend.
    ax.legend(
        loc="lower right",
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
    )
    # Save the figure.
    if filename is not None:
        plt.savefig(filename)
    # Return the figure object.
    return fig

# -------------------------------------------------------------------------------------
# PLOT CALCULATION TIMES    
# -------------------------------------------------------------------------------------

def plot_calculation_times(
    times_dict: list,
    labels: list,
    ax: object = None,
    y_max: float = None,
    ylabel: str = "Time per calculation [min]",
    filename: str = "plot_calculation_times.png",
) -> object:
    """
    Plot the steps distribution.
    """
    # Prepare figure.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    # Plot half violins.
    data = np.array([times for times in times_dict.values()]).T
    n_layers, n_bars = data.shape
    # Plot bars.
    x_values = np.arange(n_bars)
    bottom = np.zeros(n_bars)
    for ii in range(n_layers):
        ax.bar(
            x=x_values,
            height=data[ii],
            bottom=bottom,
            label=labels[ii],
        )
        bottom += data[ii]
    # Set y max.
    y_max = y_max if y_max else max(bottom) * 1.10
    ax.set_ylim(0, y_max)
    # Set axis labels.
    ax.set_xticks(x_values)
    ax.set_xticklabels(times_dict.keys())
    ax.set_ylabel(ylabel)
    # Add legend.
    ax.legend(
        loc="upper right",
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
    )
    # Save figure.
    if filename is not None:
        plt.savefig(filename)
    # Return the figure object.
    return fig

# -------------------------------------------------------------------------------------
# PLOT STEPS DISTRIBUTION
# -------------------------------------------------------------------------------------

def plot_steps_distribution(
    steps_dict: list,
    ax: object = None,
    y_max: float = None,
    color: str = "crimson",
    edgecolor: str = "black",
    alpha_fill: float = 0.7,
    ylabel: str = "Number of DFT single-points [-]",
    filename: str = "plot_actlearn_steps.png",
    violin_kwargs: dict = {"points": 300},
    scatter_kwargs: dict = {"s": 20},
) -> object:
    """
    Plot the steps distribution.
    """
    # Prepare figure.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    # Plot half violins.
    dataset = [list(ii) for ii in steps_dict.values()]
    positions = range(len(dataset))
    violin_kwargs = {"showmeans": False, "showextrema": False, **violin_kwargs}
    parts = ax.violinplot(dataset=dataset, positions=positions, **violin_kwargs)
    # Customize violins.
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(edgecolor)
        pc.set_alpha(alpha_fill)
    # Plot mean values.
    mean_values = [np.mean(ii) for ii in dataset]
    scatter_kwargs = {"color": "black", "facecolors": "none", **scatter_kwargs}
    ax.scatter(x=positions, y=mean_values, **scatter_kwargs)
    # Set y max.
    y_max = y_max if y_max else max([ii for jj in dataset for ii in jj]) * 1.10
    ax.set_ylim(0, y_max)
    # Set axis labels.
    ax.set_xticks(positions)
    ax.set_xticklabels(steps_dict.keys())
    ax.set_ylabel(ylabel)
    # Save the figure.
    if filename is not None:
        plt.savefig(filename)
    # Return the figure object.
    return fig

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
