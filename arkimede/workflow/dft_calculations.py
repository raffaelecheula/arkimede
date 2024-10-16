# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import re
import yaml
import subprocess
import numpy as np
from ase.io import read
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from arkimede.utilities import update_atoms_from_atoms_opt

# -------------------------------------------------------------------------------------
# GET NAME K8S
# -------------------------------------------------------------------------------------

def get_name_k8s(name):
    """Replace forbidden characters in k8s name."""
    return "job-"+re.sub('[^0-9a-zA-Z]+', '-', name.lower())+"-job"

# -------------------------------------------------------------------------------------
# SUBMIT K8S
# -------------------------------------------------------------------------------------

def submit_k8s(
    params_k8s,
    name,
    dirpath,
    namespace=None,
    command=None,
    cpu_req=8,
    cpu_lim=12,
    mem_req='8Gi',
    mem_lim='24Gi',
):
    """Submit a calculation in k8s."""

    # Replace forbidden characters in name.
    name_k8s = get_name_k8s(name=name)
    
    # Modify some general parameters.
    params = params_k8s.copy()
    params['metadata']['name'] = name_k8s
    if namespace is not None:
        params['metadata']['namespace'] = namespace
    
    # Modify some parameters of the container.
    container = params['spec']['template']['spec']['containers'][0]
    if command is not None:
        container['args'][0] = command
    container['name'] = name_k8s
    container['workingDir'] = os.path.abspath(dirpath)
    container['resources']['limits']['cpu'] = cpu_lim
    container['resources']['requests']['cpu'] = cpu_req
    container['args'][0] = re.sub('-np \d+', f'-np {cpu_req}', container['args'][0])
    container['resources']['limits']['memory'] = mem_lim
    container['resources']['requests']['memory'] = mem_req
    
    # Write new job.yaml file.
    with open('job.yaml', 'w') as config_file:
        yaml.dump(params, config_file, default_flow_style=False, width=1000)
        
    # Submit the job.
    os.system('kubectl apply -f job.yaml > /dev/null')

# -------------------------------------------------------------------------------------
# JOB QUEUED K8S
# -------------------------------------------------------------------------------------

def job_queued_k8s(name):
    """Get kubectl jobs sumbitted or running."""
    name_k8s = get_name_k8s(name=name)
    output = subprocess.check_output(
        "kubectl get jobs -o custom-columns=:.metadata.name",
        shell=True,
    ).decode("utf-8")
    return name_k8s in output.split("\n")

# -------------------------------------------------------------------------------------
# CHECK FILE CONTAINS
# -------------------------------------------------------------------------------------

def check_file_contains(filename, key_string):
    """Check if a file contains a string."""
    contains = False
    with open(filename, 'r') as fileobj:
        for line in fileobj.readlines():
            if key_string in line:
                contains = True
                break
    return contains

# -------------------------------------------------------------------------------------
# AUTOMATIC KPTS
# -------------------------------------------------------------------------------------

def automatic_kpts(atoms):
    """Calculate kpts for a dft calculation."""
    if atoms.info["structure_type"] == "gas":
        kpts = (1, 1, 1)
    elif atoms.info["structure_type"] == "surface":
        from ocdata.utils.vasp import calculate_surface_k_points
        kpts = calculate_surface_k_points(atoms)
    return kpts

# -------------------------------------------------------------------------------------
# WRITE INPUT VASP
# -------------------------------------------------------------------------------------

def write_input_vasp(atoms, auto_kpts=True):
    """Write input files for vasp calculation."""
    if auto_kpts is True:
        atoms.info["kpts"] = automatic_kpts(atoms)
    # Write vasp input files.
    atoms.calc.set(kpts=atoms.info["kpts"])
    atoms.calc.write_input(atoms=atoms)
    # Write vector for vasp dimer calculation.
    params = atoms.calc.todict()
    if params.get("ibrion") == 3:
        write_vector_modecar_vasp(vector=atoms.info["vector"])
    elif params.get("ibrion") == 44:
        write_vector_poscar_vasp(vector=atoms.info["vector"])

# -------------------------------------------------------------------------------------
# READ OUTPUT VASP
# -------------------------------------------------------------------------------------

def read_output_vasp(
    atoms,
    filename="OUTCAR",
    label="vasp",
    directory=".",
    save_trajs=False,
    write_images=False,
    update_cell=False,
    properties=["energy", "forces"],
    **kwargs,
):
    """Read output of a vasp calculation."""

    # Read all atoms from output.
    atoms_opt_all = read(os.path.join(directory, filename), ":")
    atoms_opt = atoms_opt_all[-1]

    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=True, # TODO:
        modified=False,
        properties=properties,
        update_cell=update_cell,
    )
    
    # Get list of all atoms in the calculation output.
    atoms_all = []
    for ii, atoms_opt_ii in enumerate(atoms_opt_all):
        atoms_ii = update_atoms_from_atoms_opt(
            atoms=atoms.copy(),
            atoms_opt=atoms_opt_ii,
            converged=True, # TODO:
            modified=False,
            properties=properties,
            update_cell=update_cell,
        )
        # Update name of atoms.
        atoms_ii.info["name"] += f"-{ii:05d}"
        atoms_all.append(atoms_ii)
    
    # Save trajectory.
    if save_trajs is True:
        trajname = os.path.join(directory, f'{label}.traj')
        traj = Trajectory(filename=trajname, mode='w')
        for atoms_ii in atoms_all:
            traj.write(atoms, **atoms.calc.results)
    
    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{label}.png")
        atoms.write(filename, radii=0.9, scale=200)

    return atoms_all

# -------------------------------------------------------------------------------------
# CHECK FINISHED VASP
# -------------------------------------------------------------------------------------

def check_finished_vasp(filename, key_string="Total CPU time used"):
    """Check if a vasp calculation has finished."""
    return check_file_contains(filename=filename, key_string=key_string)

# -------------------------------------------------------------------------------------
# WRITE INPUT ASEVASP
# -------------------------------------------------------------------------------------

def write_input_asevasp(atoms, auto_kpts=True):
    """Write input files for asevasp calculation."""
    if auto_kpts is True:
        atoms.info["kpts"] = automatic_kpts(atoms)
    # Write trajectory file containing all the parameters.
    atoms.write("asevasp.traj")

# -------------------------------------------------------------------------------------
# READ OUTPUT ASEVASP
# -------------------------------------------------------------------------------------

def read_output_asevasp(
    atoms,
    directory=".",
    update_cell=False,
    properties=["energy", "forces"],
):
    """Read output of an asevasp calculation."""
    
    # Read all atoms from output.
    atoms_opt_all = []
    for calculation in atoms.info["calculation"].split("+"):
        atoms_opt_all += read(os.path.join(directory, calculation+".traj"), ":")
    atoms_opt = atoms_opt_all[-1]
    
    # Update the input atoms with the results of the calculation.
    atoms = update_atoms_from_atoms_opt(
        atoms=atoms,
        atoms_opt=atoms_opt,
        converged=atoms_opt.info["converged"],
        modified=atoms_opt.info["modified"],
        properties=properties,
        update_cell=update_cell,
    )
    
    # Get list of all atoms in the calculation output.
    atoms_all = []
    for ii, atoms_opt_ii in enumerate(atoms_opt_all):
        atoms_ii = update_atoms_from_atoms_opt(
            atoms=atoms.copy(),
            atoms_opt=atoms_opt_ii,
            converged=atoms_ii.info["converged"],
            modified=atoms_ii.info["modified"],
            properties=properties,
            update_cell=update_cell,
        )
        # Update name of atoms.
        if "displacement" in atoms_ii.info:
            atoms_ii.info["name"] += "-"+atoms_ii.info["displacement"]
        else:
            atoms_ii.info["name"] += f"-{ii:05d}"
        atoms_all.append(atoms_ii)
    
    return atoms_all

# -------------------------------------------------------------------------------------
# CHECK FINISHED ASEVASP
# -------------------------------------------------------------------------------------

def check_finished_asevasp(filename, key_string="all calculations finished"):
    """Check if an asevasp calculation has finished."""
    return check_file_contains(filename=filename, key_string=key_string)

# -------------------------------------------------------------------------------------
# WRITE VECTOR MODECAR VASP
# -------------------------------------------------------------------------------------

def write_vector_modecar_vasp(vector):
    """Write vector to modecar for dimer calculation in vasp."""
    with open('MODECAR', 'w+') as fileobj:
        for line in vector:
            print("{0:20.10E} {1:20.10E} {2:20.10E}".format(*line), file=fileobj)

# -------------------------------------------------------------------------------------
# WRITE VECTOR POSCAR VASP
# -------------------------------------------------------------------------------------

def write_vector_poscar_vasp(vector):
    """Write vector to poscar for improved dimer calculation in vasp."""
    with open('POSCAR', 'a') as fileobj:
        print("", file=fileobj)
        for line in vector:
            print("{0:20.10E} {1:20.10E} {2:20.10E}".format(*line), file=fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
