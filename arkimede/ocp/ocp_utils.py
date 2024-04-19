#######################################################################################
# This is adapted from ocp/tutorial.
#######################################################################################

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from pathlib import Path
from arkimede.utils import checkpoints_basedir

# -------------------------------------------------------------------------------------
# OCP ROOT
# -------------------------------------------------------------------------------------

def ocp_root():
    """Return the root directory of the installed ocp package."""
    import ocpmodels
    return Path(ocpmodels.__file__).parent.parent

# -------------------------------------------------------------------------------------
# OCP MAIN
# -------------------------------------------------------------------------------------

def ocp_main():
    """Return the path to ocp main.py"""
    return ocp_root() / "main.py"

# -------------------------------------------------------------------------------------
# DESCRIBE OCP
# -------------------------------------------------------------------------------------

def describe_ocp():
    """Print some system information that could be useful in debugging."""
    
    import sys
    import ase
    import numba
    import e3nn
    import pymatgen.core
    import platform
    import torch
    import torch.cuda
    import torch_geometric
    import ocpmodels
    import subprocess
    import psutil
    
    print(sys.executable, sys.version)  
    print(f'ocp is installed at {ocp_root()}')
    
    command = ["git", "-C", ocpmodels.__path__[0], "describe", "--always"]
    commit_hash = subprocess.check_output(command).strip().decode("ascii")
    print(f'ocp repo is at git commit: {commit_hash}')
    print(f'numba: {numba.__version__}')
    print(f'numpy: {np.version.version}')
    print(f'ase: {ase.__version__}')
    print(f'e3nn: {e3nn.__version__}')
    print(f'pymatgen: {pymatgen.core.__version__}')
    print(f'torch: {torch.version.__version__}')
    print(f'torch.version.cuda: {torch.version.cuda}')
    print(f'torch.cuda: is_available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print('  __CUDNN VERSION:', torch.backends.cudnn.version())
        print('  __Number CUDA Devices:', torch.cuda.device_count())
        print('  __CUDA Device Name:', torch.cuda.get_device_name(0))
        print(
            '  __CUDA Device Total Memory [GB]:',
            torch.cuda.get_device_properties(0).total_memory/1e9,
        )
    print(f'torch geometric: {torch_geometric.__version__}')    
    print()
    print(f'Platform: {platform.platform()}')
    print(f'  Processor: {platform.processor()}')
    print(f'  Virtual memory: {psutil.virtual_memory()}')
    print(f'  Swap memory: {psutil.swap_memory()}')
    print(f'  Disk usage: {psutil.disk_usage("/")}')

# -------------------------------------------------------------------------------------
# GET CHECKPOINTS KEYS
# -------------------------------------------------------------------------------------

def get_checkpoints_keys():

    # Base address.
    ba = "https://dl.fbaipublicfiles.com/opencatalystproject/models/"

    checkpoints_keys = {
        # Open Catalyst 2020 (OC20)
        'CGCNN 200k': f'{ba}/2020_11/s2ef/cgcnn_200k.pt', 
        'CGCNN 2M': f'{ba}/2020_11/s2ef/cgcnn_2M.pt', 
        'CGCNN 20M': f'{ba}/2020_11/s2ef/cgcnn_20M.pt', 
        'CGCNN All': f'{ba}/2020_11/s2ef/cgcnn_all.pt', 
        'DimeNet 200k': f'{ba}/2020_11/s2ef/dimenet_200k.pt',
        'DimeNet 2M': f'{ba}/2020_11/s2ef/dimenet_2M.pt',
        'SchNet 200k': f'{ba}/2020_11/s2ef/schnet_200k.pt', 
        'SchNet 2M': f'{ba}/2020_11/s2ef/schnet_2M.pt', 
        'SchNet 20M': f'{ba}/2020_11/s2ef/schnet_20M.pt', 
        'SchNet All': f'{ba}/2020_11/s2ef/schnet_all_large.pt', 
        'DimeNet++ 200k': f'{ba}/2021_02/s2ef/dimenetpp_200k.pt', 
        'DimeNet++ 2M': f'{ba}/2021_02/s2ef/dimenetpp_2M.pt', 
        'DimeNet++ 20M': f'{ba}/2021_02/s2ef/dimenetpp_20M.pt', 
        'DimeNet++ All': f'{ba}/2021_02/s2ef/dimenetpp_all.pt', 
        'SpinConv 2M': f'{ba}/2021_12/s2ef/spinconv_force_centric_2M.pt', 
        'SpinConv All': f'{ba}/2021_08/s2ef/spinconv_force_centric_all.pt', 
        'GemNet-dT 2M': f'{ba}/2021_12/s2ef/gemnet_t_direct_h512_2M.pt', 
        'GemNet-dT All': f'{ba}/2021_08/s2ef/gemnet_t_direct_h512_all.pt', 
        'PaiNN All': f'{ba}/2022_05/s2ef/painn_h512_s2ef_all.pt', 
        'GemNet-OC 2M': f'{ba}/2022_07/s2ef/gemnet_oc_base_s2ef_2M.pt', 
        'GemNet-OC All': f'{ba}/2022_07/s2ef/gemnet_oc_base_s2ef_all.pt', 
        'GemNet-OC All+MD': f'{ba}/2023_03/s2ef/gemnet_oc_base_s2ef_all_md.pt', 
        'GemNet-OC-Large All+MD': f'{ba}/2022_07/s2ef/gemnet_oc_large_s2ef_all_md.pt', 
        'SCN 2M': f'{ba}/2023_03/s2ef/scn_t1_b1_s2ef_2M.pt', 
        'SCN-t4-b2 2M': f'{ba}/2023_03/s2ef/scn_t4_b2_s2ef_2M.pt', 
        'SCN All+MD': f'{ba}/2023_03/s2ef/scn_all_md_s2ef.pt', 
        'eSCN-L4-M2-Lay12 2M': f'{ba}/2023_03/s2ef/escn_l4_m2_lay12_2M_s2ef.pt', 
        'eSCN-L6-M2-Lay12 2M': f'{ba}/2023_03/s2ef/escn_l6_m2_lay12_2M_s2ef.pt', 
        'eSCN-L6-M2-Lay12 All+MD': f'{ba}/2023_03/s2ef/escn_l6_m2_lay12_all_md_s2ef.pt', 
        'eSCN-L6-M3-Lay20 All+MD': f'{ba}/2023_03/s2ef/escn_l6_m3_lay20_all_md_s2ef.pt', 
        'EquiformerV2 (83M) 2M': f'{ba}/2023_06/oc20/s2ef/eq2_83M_2M.pt', 
        'EquiformerV2 (31M) All+MD': f'{ba}/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt', 
        'EquiformerV2 (153M) All+MD': f'{ba}/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt', 
        # Open Catalyst 2022 (OC22)
        'GemNet-dT OC22': f'{ba}/2022_09/oc22/s2ef/gndt_oc22_all_s2ef.pt',
        'GemNet-OC OC22': f'{ba}/2022_09/oc22/s2ef/gnoc_oc22_all_s2ef.pt',
        'GemNet-OC OC20+OC22': f'{ba}/2022_09/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt',
        'GemNet-OC OC20+OC22 v2': f'{ba}/2023_05/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt',
        'GemNet-OC OC20->OC22': f'{ba}/2022_09/oc22/s2ef/gnoc_finetune_all_s2ef.pt',
        'EquiformerV2 OC22': f'{ba}/2023_10/oc22/s2ef/eq2_121M_e4_f100_oc22_s2ef.pt',
    }
    
    return checkpoints_keys

# -------------------------------------------------------------------------------------
# LIST CHECKPOINTS
# -------------------------------------------------------------------------------------

def list_checkpoints():
    """List checkpoints that are available to download."""
    print(
        'See https://github.com/Open-Catalyst-Project/ocp/blob/main/MODELS.md '+
        'for more details.'
    )
    checkpoints_keys = get_checkpoints_keys()
    for key in checkpoints_keys:
        print(key)

# -------------------------------------------------------------------------------------
# GET CHECKPOINT PATH
# -------------------------------------------------------------------------------------
        
def get_checkpoint_path(checkpoint_key, basedir=checkpoints_basedir()):
    """Download a checkpoint.
    
    checkpoint_key: string in checkpoints.
    basedir: directory where to store the checkpoints.
    
    Returns name of checkpoint that was saved.
    """
    import requests
    import urllib
    
    checkpoints_keys = get_checkpoints_keys()
    checkpoint_url = checkpoints_keys[checkpoint_key]
    if checkpoint_url is None:
        raise Exception(f'No url found for {checkpoint_key}')
            
    checkpoint_path = os.path.join(
        basedir,
        Path(urllib.parse.urlparse(checkpoint_url).path).name,
    )
    
    os.makedirs(basedir, exist_ok=True)
    
    if not os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'wb') as fileobj:
            print(f'Downloading {checkpoint_url}')
            fileobj.write(requests.get(checkpoint_url).content)
    
    return checkpoint_path      

# -------------------------------------------------------------------------------------
# GET CHECKPOINT PATH ACTLEARN
# -------------------------------------------------------------------------------------

def get_checkpoint_path_actlearn(step_actlearn, directory="finetuning"):
    """Get checkpoint path after fine tuning."""
    checkpoint_path = None
    basepath = os.path.join(directory, "checkpoints")
    for dirname in sorted(os.listdir(basepath)):
        if int(dirname.split("-")[-1]) == step_actlearn:
            checkpoint_path = os.path.join(basepath, dirname, "best_checkpoint.pt")
            break
    return checkpoint_path

# -------------------------------------------------------------------------------------
# SPLIT DATABASE
# -------------------------------------------------------------------------------------

def split_database(
    db_name,
    change_energy_ref_fun=None,
    fractions=(0.8, 0.1, 0.1),
    filenames=('train.db', 'test.db', 'val.db'),
    directory='.',
    seed=42,
):
    """Split an ase db into train, test and validation dbs.
    
    db_name: path to an ase db containing all the data.
    change_energy_ref_fun: function to change the reference energy for OCP models.
    fractions: a tuple containing the fraction of train, test and val data. 
        This will be normalized.
    filenames: a tuple of filenames to write the splits into.
    directory: directory in which store the databases.
    seed: an integer for the random number generator seed
    
    Returns the absolute path to files.
    """
    from ase.db import connect
    
    os.makedirs(directory, exist_ok=True)
    
    # Get filenames (and delete them).
    db_name_list = []
    for name in filenames:
        db_name = os.path.join(directory, name)
        if os.path.exists(db_name):
            os.remove(db_name)
        db_name_list.append(db_name)
    
    # Read source database.
    db_ase = connect(db_name)
    n_data = db_ase.count()
    
    # Set sum of fractions equal to 1 and get numbers of data.
    fractions = np.array(fractions)
    fractions /= fractions.sum()
    n_data_array = np.array(np.round(fractions*n_data), dtype=int)
    n_data_array[-1] = n_data-np.sum(n_data_array[:-1])
    
    # Shuffle the database ids.
    ids = np.arange(1, n_data+1)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(ids)
    
    # Write new databases.
    num = 0
    for ii, db_name in enumerate(db_name_list):
        db_new = connect(db_name)
        for id in ids[num:num+n_data_array[ii]]:
            row = db_ase.get(id=int(id))
            atoms = row.toatoms()
            if change_energy_ref_fun is not None:
                energy = change_energy_ref_fun(atoms=atoms)
                atoms.calc.results["energy"] = energy
            db_new.write(atoms)
        num += n_data_array[ii]

# -------------------------------------------------------------------------------------
# MERGE DATABASES
# -------------------------------------------------------------------------------------

def merge_databases(db_name_list, db_new_name):
    """Merge databases into one new database."""
    from ase.db import connect
    
    db_new = connect(db_new_name)
    for ii, db_name in enumerate(db_name_list):
        db_ase = connect(db_name)
        for id in range(1, db_ase.count()+1):
            row = db_ase.get(id=int(id))
            atoms = row.toatoms()
            db_new.write(atoms)

# -------------------------------------------------------------------------------------
# UPDATE CONFIG YAML
# -------------------------------------------------------------------------------------

def update_config_yaml(
    checkpoint_path,
    config_yaml='config.yml',
    delete_keys=(),
    update_keys={},
):
    """Generate a yml config file from an existing checkpoint file.
    
    checkpoint_path: string to path of an existing checkpoint
    yml: name of file to write to.
    delete_keys: list of keys to remove from the config
    update: dictionary of key:values to update
    """
    
    import yaml
    from ocpmodels.common.relaxation.ase_utils import OCPCalculator

    config_dict = OCPCalculator(checkpoint_path=checkpoint_path).config
    
    for key in delete_keys:
        if key in config_dict:
            del config_dict[key]
        elif "." in key and key.split(".")[0] in config_dict:
            key_list = key.split(".")
            nested_dict = config_dict[key_list[0]]
            for key_ii in key_list[1:-1]:
                if key_ii in nested_dict:
                    nested_dict = nested_dict[key_ii]
            if key_list[-1] in nested_dict:
                del nested_dict[key_list[-1]]
    
    for key in update_keys:
        if "." not in key:
            config_dict[key] = update_keys[key]
        else:
            key_list = key.split(".")
            if key_list[0] not in config_dict:
                config_dict[key_list[0]] = {}
            nested_dict = config_dict[key_list[0]]
            for key_ii in key_list[1:-1]:
                if key_ii not in nested_dict:
                    nested_dict[key_ii] = {}
                nested_dict = nested_dict[key_ii]
            nested_dict[key_list[-1]] = update_keys[key]
    
    with open(config_yaml, 'w') as fileobj:
        yaml.dump(config_dict, fileobj)

# -------------------------------------------------------------------------------------
# REF ENERGY ADS DICT
# -------------------------------------------------------------------------------------

def ref_energy_ads_dict():
    """Return a dictionary of the reference energies of adsorbate atoms.
    For adsorption energy ocp models."""
    return {
        "H": -3.477,
        "O": -7.204,
        "C": -7.282,
        "N": -8.083,
    }

# -------------------------------------------------------------------------------------
# REF ENERGY TOT ARRAY
# -------------------------------------------------------------------------------------

def ref_energy_tot_array():
    """Return an array with the reference energies for all atomic species.
    For total energy ocp models."""
    filename = f"{ocp_root()}/configs/oc22/linref/oc22_linfit_coeffs.npz"
    return np.array(np.load(filename)["coeff"])

# -------------------------------------------------------------------------------------
# REF FORM ENERGY ADS
# -------------------------------------------------------------------------------------

def get_form_energy_ads(
    atoms,
    energy_slab,
    indices_ads,
    ref_energy_dict=ref_energy_ads_dict(),
    energy=None,
):
    """Get the formation energy of an Atoms object for adsorption energy ocp models."""
    if energy is None:
        energy = atoms.get_potential_energy()
    ref_energy = energy_slab+0.
    for aa in atoms[indices_ads]:
        ref_energy += ref_energy_dict[aa.symbol]
    return energy-ref_energy

# -------------------------------------------------------------------------------------
# GET FORM ENERGY TOT
# -------------------------------------------------------------------------------------

def get_form_energy_tot(
    atoms,
    energy=None,
    ref_energy_array=ref_energy_tot_array(),
):
    """Get the formation energy of an Atoms object for total energy ocp models."""
    if energy is None:
        energy = atoms.get_potential_energy()
    ref_energy = ref_energy_array[atoms.get_atomic_numbers()].sum()
    return energy-ref_energy

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------