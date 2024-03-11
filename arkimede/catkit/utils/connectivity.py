import numpy as np
import scipy
import warnings
import networkx as nx
from ase.data import covalent_radii
from .coordinates import expand_cell


def get_connectivity_voronoi(
    atoms,
    radii=None,
    cutoff_max=5.0,
    use_cov_radii=False,
    scale_cov_radii=1.2,
    add_cov_radii=0.2,
    voronoi_4d=True,
):
    """Return the connectivity matrix from the Voronoi method.
    Multi-bonding occurs through periodic boundary conditions.
    """
    
    index, coords, offsets = expand_cell(atoms, cutoff=cutoff_max)

    xm, ym, zm = np.max(coords, axis=0) - np.min(coords, axis=0)

    if radii is not None:
        radii = np.tile(radii, len(offsets))
    elif use_cov_radii is True:
        radii = covalent_radii[atoms.numbers] * scale_cov_radii + add_cov_radii
        radii = np.tile(radii, len(offsets))
    else:
        radii = np.ones(len(index)) * cutoff_max / 2.0

    if voronoi_4d is True:
        coords = np.hstack([coords, radii.reshape(-1,1)])

    voronoi = scipy.spatial.Voronoi(coords, qhull_options="QbB Qc Qs")

    ll = int(len(offsets) / 2)
    original_indices = np.arange(ll * len(atoms), (ll + 1) * len(atoms))

    connectivity = np.zeros((len(atoms), len(atoms)))
    distances = []
    distance_indices = []
    for nn in original_indices:
        ridge_indices = np.where(voronoi.ridge_points == nn)[0]
        pp = voronoi.ridge_points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[:, :3][pp], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[pp])

        if not edges.size:
            warnings.warn(
                (
                    "scipy.spatial.Voronoi returned an atom which has "
                    "no neighbors. This may result in incorrect connectivity."
                )
            )
            continue

        unique_edge = np.unique(edges, axis=0)
        for edge in unique_edge:
            uu, vv = edge
            indices = np.where(np.all(edge == edges, axis=1))[0]
            cutoff = radii[uu]+radii[vv]
            dd = dist[indices][np.where(dist[indices] < cutoff)[0]]
            count = len(dd)
            if count == 0:
                continue
            distance_indices += [sorted([uu, vv])]
            distances += [sorted(dd)]
            connectivity[uu][vv] += count
            connectivity[vv][uu] += count

    connectivity /= 2
    return connectivity.astype(int)


def get_connectivity_cutoff(
    atoms,
    radii=None,
    cutoff_max=5.0,
    use_cov_radii=True,
    scale_cov_radii=1.0,
    add_cov_radii=0.2,
    atol=1e-8,
):
    """Return the connectivity matrix from a simple radial cutoff.
    Multi-bonding occurs through periodic boundary conditions.
    """
    
    index, coords, offsets = expand_cell(atoms)

    if radii is not None:
        radii = np.tile(radii, len(offsets))
    elif use_cov_radii is True:
        radii = covalent_radii[atoms.numbers] * scale_cov_radii + add_cov_radii
        radii = np.tile(radii, len(offsets))
    else:
        radii = np.ones(len(index)) * cutoff_max / 2.0

    connectivity = np.zeros((len(atoms), len(atoms)), dtype=int)
    for ii, center in enumerate(atoms.positions):
        norm = np.linalg.norm(coords - center, axis=1)
        boundary_distance = norm - radii - radii[ii]
        neighbors = index[np.where(boundary_distance < atol)]
        unique, counts = np.unique(neighbors, return_counts=True)
        connectivity[ii, unique] = counts
        # Remove self-interaction.
        connectivity[ii, ii] -= 1

    return connectivity


def get_connectivity_cutoff_ase(
    atoms,
    radii=None,
    cutoff_max=5.0,
    use_cov_radii=True,
    scale_cov_radii=1.0,
    add_cov_radii=0.2,
):
    """Return the connectivity matrix from ase.neighborlist.NeighborList.
    """
    
    from ase.neighborlist import NeighborList, get_connectivity_matrix
    
    if radii is None and use_cov_radii is True:
        radii = covalent_radii[atoms.numbers] * scale_cov_radii + add_cov_radii
    elif radii is None:
        radii = np.ones(len(atoms)) * cutoff_max / 2.0
    
    ase_neighbor_list = NeighborList(
        cutoffs=radii,
        self_interaction=False,
        bothways=True,
    )
    ase_neighbor_list.update(atoms)
    
    return get_connectivity_matrix(ase_neighbor_list.nl).toarray()


def get_connectivity_pymatgen(atoms, pymatgen_class_NN=None):
    """Return the connectivity matrix from pymatgen.analysis.local_env classes.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    
    structure = AseAtomsAdaptor().get_structure(atoms)
    structure.add_oxidation_state_by_guess()
    
    if pymatgen_class_NN is None:
        from pymatgen.analysis.local_env import CrystalNN
        pymatgen_class_NN = CrystalNN()
    
    connectivity = np.zeros((len(atoms), len(atoms)), dtype=int)
    for ii in range(len(atoms)):
        for site in pymatgen_class_NN.get_nn_info(structure, ii):
            connectivity[ii, site["site_index"]] += 1
    
    return connectivity


def get_connectivity(
    atoms,
    method="cutoff",
    radii=None,
    cutoff_max=5.0,
    use_cov_radii=True,
    scale_cov_radii=1.0,
    add_cov_radii=0.2,
    attach_graph=True,
    pymatgen_class_NN=None,
):

    if method == "cutoff":
        connectivity = get_connectivity_cutoff(
            atoms=atoms,
            radii=radii,
            cutoff_max=cutoff_max,
            use_cov_radii=use_cov_radii,
            scale_cov_radii=scale_cov_radii,
            add_cov_radii=add_cov_radii,
        )

    elif method == "voronoi":
        connectivity = get_connectivity_voronoi(
            atoms=atoms,
            radii=radii,
            cutoff_max=cutoff_max,
            use_cov_radii=use_cov_radii,
            scale_cov_radii=scale_cov_radii,
            add_cov_radii=add_cov_radii,
            voronoi_4d=True,
        )

    elif method == "cutoff-ase":
        connectivity = get_connectivity_cutoff_ase(
            atoms=atoms,
            radii=radii,
            cutoff_max=cutoff_max,
            use_cov_radii=use_cov_radii,
            scale_cov_radii=scale_cov_radii,
            add_cov_radii=add_cov_radii,
        )

    elif method == "pymatgen":
        connectivity = get_connectivity_pymatgen(
            atoms=atoms,
            pymatgen_class_NN=pymatgen_class_NN,
        )

    if attach_graph is True:
        atoms.connectivity = connectivity

    return connectivity


def plot_connectivity(atoms, connectivity=None):

    import matplotlib.pyplot as plt
    from itertools import product

    if connectivity is None:
        connectivity = atoms.connectivity

    displ_list = []
    for vect in list(product([-1, 0, +1], repeat=3)):
        displ_list.append(np.dot(atoms.cell, vect))

    edges_xyz = []
    index_atoms = range(len(atoms))
    for ii in index_atoms:
        for jj in [jj for jj in index_atoms if connectivity[ii, jj] > 0]:
            distances = np.array([
                np.linalg.norm(atoms[ii].position-(atoms[jj].position+displ))
                for displ in displ_list
            ])
            for kk in np.argsort(distances)[:connectivity[ii, jj]]:
                edges_xyz.append(
                    np.array([atoms[ii].position, atoms[jj].position+displ_list[kk]])
                )

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*atoms.positions.T, s=500, ec="w")

    for edge in edges_xyz:
        ax.plot(*edge.T, color="grey", alpha=0.5)

    ax.grid(False)
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    plt.show()
