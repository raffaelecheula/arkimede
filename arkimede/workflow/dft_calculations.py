# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import re
import yaml
import numpy as np
from ase.io import read
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

# -----------------------------------------------------------------------------
# SUBMIT K8S
# -----------------------------------------------------------------------------

def submit_k8s(
    params_k8s,
    run_tag,
    dirpath,
    namespace = None,
    command = None,
    cpu_req = 8,
    cpu_lim = 12,
    mem_req = '8Gi',
    mem_lim = '24Gi',
):
    
    # Replace forbidden characters in run_tag.
    run_tag = re.sub('[^0-9a-zA-Z]+', '-', run_tag.lower())
    
    # Modify some general parameters.
    params = params_k8s.copy()
    params['metadata']['name'] = run_tag
    if namespace is not None:
        params['metadata']['namespace'] = namespace
    
    # Modify some parameters of the container.
    container = params['spec']['template']['spec']['containers'][0]
    if command is not None:
        container['args'][0] = command
    container['name'] = run_tag
    container['workingDir'] = dirpath
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

# -----------------------------------------------------------------------------
# RUN RELAX CALCULATION
# -----------------------------------------------------------------------------

def read_dft_output(
    atoms,
    filename = "vasprun.xml",
    name = "vasp",
    directory = ".",
    save_trajs = False,
    write_images = False,
    update_cell = False,
):
    """Read dft output."""

    # Read dft output.
    atoms_out = read(os.path.join(directory, filename))
    converged = True # TODO:
    
    # Save trajectory.
    if save_trajs is True:
        traj = Trajectory(
            filename = os.path.join(directory, f'{name}.traj'),
            mode = 'w',
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
    atoms.info["converged"] = bool(converged)

    # Write image.
    if write_images is True:
        filename = os.path.join(directory, f"{name}.png")
        atoms.write(filename, radii=0.9, scale=200)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
