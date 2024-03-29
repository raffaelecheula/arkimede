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

# -------------------------------------------------------------------------------------
# GET NAME K8S
# -------------------------------------------------------------------------------------

def get_name_k8s(name):
    """Replace forbidden characters in k8s name and add random hex."""
    import secrets
    return re.sub('[^0-9a-zA-Z]+', '-', name.lower())+'-'+secrets.token_hex(5)

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
        yaml.dump(params, config_file, default_flow_style = False, width = 1000)
        
    # Submit the job.
    os.system('kubectl apply -f job.yaml > /dev/null')

# -------------------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -------------------------------------------------------------------------------------

def read_dft_output(
    atoms,
    filename="OUTCAR",
    name="vasp",
    directory=".",
    save_trajs=False,
    write_images=False,
    update_cell=False,
    **kwargs,
):
    """Read dft output."""

    # Read dft output.
    atoms_out = read(os.path.join(directory, filename))
    
    # Save trajectory.
    if save_trajs is True:
        traj = Trajectory(
            filename=os.path.join(directory, f'{name}.traj'),
            mode='w',
        )
        for atoms in read(os.path.join(directory, filename), ':'):
            traj.write(atoms)
    
    # Update the input atoms with the results.
    atoms.set_positions(atoms_out.positions)
    if update_cell is True:
        atoms.set_cell(atoms_out.cell)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        energy=atoms_out.calc.results['energy'],
        forces=atoms_out.calc.results['forces'],
    )
    atoms.info["modified"] = False
    atoms.info["converged"] = True

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -------------------------------------------------------------------------------------
# CHECK FILE CONTAINS
# -------------------------------------------------------------------------------------

def check_file_contains(filename, key_string):
    contains = False
    with open(filename, 'r') as fileobj:
        for line in fileobj.readlines():
            if key_string in line:
                contains = True
                break
    return contains

# -------------------------------------------------------------------------------------
# CHECK FINISHED VASP
# -------------------------------------------------------------------------------------

def check_finished_vasp(filename, key_string="Total CPU time used"):
    return check_file_contains(filename=filename, key_string=key_string)

# -------------------------------------------------------------------------------------
# WRITE INPUT VASP
# -------------------------------------------------------------------------------------

def write_input_vasp(atoms, auto_kpts=True):
    """Write input files for vasp calculation."""
    if auto_kpts is True:
        from ocdata.utils.vasp import calculate_surface_k_points
        kpts = calculate_surface_k_points(atoms)
    else:
        kpts = atoms.info["kpts"]
    # Write vasp input files.
    atoms.calc.set(kpts=kpts)
    atoms.calc.write_input(atoms=atoms)
    # Write vector for vasp dimer calculation.
    params = atoms.calc.todict()
    if params.get("ibrion") == 3:
        write_vector_modecar_vasp(vector=atoms.info["vector"])
    elif params.get("ibrion") == 44:
        write_vector_poscar_vasp(vector=atoms.info["vector"])

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
# END
# -------------------------------------------------------------------------------------
