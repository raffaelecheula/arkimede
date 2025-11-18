# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.io import read

from arkimede.workflow.reaction_mechanism import MechanismCalculator
from arkimede.workflow.calculations import run_calculation, run_neb_calculation
from arkimede.utilities import (
    get_idpp_interpolated_images,
    get_atoms_TS_from_images_neb,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

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
            model_name="uma-s-1",
            device="cuda",
            cache_dir="pretrained_models",
        )
        calc = FAIRChemCalculator(predict_unit=predict_unit, task_name="oc20")

    from ase_structures import get_atoms_slab, get_atoms_gas

    # Reactions names.
    reaction_list = ["H2O*→OH*+H*"]

    if os.path.isfile("atoms_TS.traj"):
        # Read TS structure.
        atoms_TS = read("atoms_TS.traj")
        atoms_TS.pbc = True
    else:
        # Get the clean slab.
        atoms_clean = get_atoms_slab(
            bulk_structure="fcc",
            element_bulk="Rh",
            miller_index="100",
            delta_ncoord=1,
        )
        # Sites and reactions parameters.
        range_edges_fun = lambda lc: [1e-2, lc + 1e-2]
        displ_max_fun = lambda lc: lc * np.sqrt(2.0) / 4.0 + 1e-3
        range_coads_fun = lambda lc: [lc * 0.4 + 1e-3, lc * 0.8 + 1e-3]
        lattice_const = atoms_clean.info["lattice_const"]
        range_edges = range_edges_fun(lattice_const)
        displ_max_reaction = displ_max_fun(lattice_const)
        range_coadsorption = range_coads_fun(lattice_const)
        # Generate the pairs of IS and FS.
        mech = MechanismCalculator(
            atoms_clean=atoms_clean,
            get_atoms_gas_fun=get_atoms_gas,
            range_edges=range_edges,
            displ_max_reaction=displ_max_reaction,
            range_coadsorption=range_coadsorption,
            auto_construct=True,
        )
        atoms_neb_tot = mech.get_neb_images(reaction_list=reaction_list)
        # Use only the first pair for testing.
        atoms_IS, atoms_FS = atoms_neb_tot[0]
        # Relax IS and FS.
        for atoms in [atoms_IS, atoms_FS]:
            run_calculation(atoms=atoms, calc=calc, calculation="relax")
        # Get interpolated images.
        images = get_idpp_interpolated_images(atoms_IS=atoms_IS, atoms_FS=atoms_FS)
        # Run NEB calculation.
        run_neb_calculation(images=images, calc=calc, max_steps=10)
        # Get TS guess from NEB images.
        atoms_TS = get_atoms_TS_from_images_neb(images=images)
        atoms_TS.write("atoms_TS.traj")

    # Optimize TS.
    #run_calculation(atoms=atoms_TS.copy(), calc=calc, calculation="dimer")
    
    # Optimize TS.
    #run_calculation(atoms=atoms_TS.copy(), calc=calc, calculation="climbfixint")
    
    from arkimede.optimize.tsoptimizer import HybridBFGS
    from ase.optimize.lbfgs import LBFGS
    
    # Optimize TS.
    atoms_climbbonds = atoms_TS.copy()
    run_calculation(atoms=atoms_climbbonds, calc=calc, calculation="climbbonds", optimizer=HybridBFGS)
    atoms_climbbonds.write("atoms_LBFGS.traj")

    # Optimize TS.
    atoms_climbbonds = atoms_TS.copy()
    run_calculation(atoms=atoms_climbbonds, calc=calc, calculation="climbbonds")
    atoms_climbbonds.write("atoms_climbbonds.traj")

    #atoms = atoms_TS.copy()
    #vector = atoms.info["vector"]
    #bonds_TS = atoms.info["bonds_TS"]
    #
    #from arkimede.optimize.tsoptimizer import TSOptimizer
    #atoms.calc = calc
    #opt = TSOptimizer(
    #    atoms=atoms,
    #    bonds_TS=[bonds[:2] for bonds in bonds_TS],
    #)
    #opt.run(fmax=0.05)
    #atoms.write("atoms_hybrid.traj")
    
    """
    
    from ase.mep.dimer import MinModeAtoms, DimerControl, MinModeTranslate
    from arkimede.optimize.tsoptimizer import DirectedBondDimer
    from arkimede.utilities import get_indices_not_fixed

    atoms.calc = calc
    mask = get_indices_not_fixed(atoms, return_mask=True)
    
    kwargs_dimer_opt = {
        "initial_eigenmode_method": "displacement",
        "displacement_method": "vector",
        "logfile": None,
        "cg_translation": True,
        "use_central_forces": True,
        "extrapolate_forces": False,
        "order": 1,
        "f_rot_min": 0.10,
        "f_rot_max": 1.00,
        "max_num_rot": 6, # modified
        "trial_angle": np.pi/6, # modified
        "trial_trans_step": 0.005, # modified
        "maximum_translation": 0.050, # modified
        "dimer_separation": 0.005, # modified
    }
    
    dimer_control = DimerControl(**kwargs_dimer_opt)
    #atoms_opt = DirectedBondDimer(
    #    atoms=atoms,
    #    control=dimer_control,
    #    mask=mask,
    #    bond_pair=bonds_TS,
    #)
    atoms_opt = MinModeAtoms(
        atoms=atoms,
        control=dimer_control,
        mask=mask,
    )
    vector *= kwargs_dimer_opt["maximum_translation"]/np.linalg.norm(vector)
    atoms_opt.displace(displacement_vector=vector, mask=mask)

    opt = MinModeTranslate(
        dimeratoms=atoms_opt,
        trajectory=None,
        logfile="-",
    )
    opt.run(fmax=0.05, steps=1000)

    print(calc.counter)

    #dimer_atoms = DirectedBondDimer(
    #    atoms=atoms_TS,
    #    bond_pair=atoms_TS.info["bonds_TS"][0][:2],
    #    control=dimer_control,
    #)
    """

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
