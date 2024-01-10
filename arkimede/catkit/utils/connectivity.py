import numpy as np
import scipy
import warnings
import networkx as nx
from ase.data import atomic_numbers, covalent_radii
from networkx.convert_matrix import from_numpy_matrix
from .coordinates import expand_cell


def get_voronoi_neighbors(atoms, cutoff=5.0, return_distances=False):
    """Return the connectivity matrix from the Voronoi
    method. Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.

    Returns
    -------
    connectivity : ndarray (n, n)
        Number of edges formed between atoms in a system.
    """
    index, coords, offsets = expand_cell(atoms, cutoff=cutoff)

    xm, ym, zm = np.max(coords, axis=0) - np.min(coords, axis=0)

    L = int(len(offsets) / 2)
    origional_indices = np.arange(L * len(atoms), (L + 1) * len(atoms))

    voronoi = scipy.spatial.Voronoi(coords, qhull_options="QbB Qc Qs")
    points = voronoi.ridge_points

    connectivity = np.zeros((len(atoms), len(atoms)))
    distances = []
    distance_indices = []
    for i, n in enumerate(origional_indices):
        ridge_indices = np.where(points == n)[0]
        p = points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[p], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[p])

        if not edges.size:
            warnings.warn(
                (
                    "scipy.spatial.Voronoi returned an atom which has "
                    "no neighbors. This may result in incorrect connectivity."
                )
            )
            continue

        unique_edge = np.unique(edges, axis=0)

        for j, edge in enumerate(unique_edge):
            indices = np.where(np.all(edge == edges, axis=1))[0]
            d = dist[indices][np.where(dist[indices] < cutoff)[0]]
            count = len(d)
            if count == 0:
                continue

            u, v = edge

            distance_indices += [sorted([u, v])]
            distances += [sorted(d)]

            connectivity[u][v] += count
            connectivity[v][u] += count

    connectivity /= 2
    if not return_distances:
        return connectivity.astype(int)

    if len(distances) > 0:
        distance_indices, unique_idx_idx = np.unique(
            distance_indices, axis=0, return_index=True
        )
        distance_indices = distance_indices.tolist()

        distances = [distances[i] for i in unique_idx_idx]

    pair_distances = {"indices": distance_indices, "distances": distances}

    return connectivity.astype(int), pair_distances


def get_cutoff_neighbors(
    atoms, cutoff=None, scale_cov_radii=1.0, atol=1e-8, radii=None
):
    """Return the connectivity matrix from a simple radial cutoff.
    Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cutoff : float
        Cutoff radius to use when determining neighbors.
    scale_cov_radii : float
        Value close to 1 to scale the covalent radii used for automated
        cutoff values.
    atol: float
        Absolute tolerance to use when computing distances.

    Returns
    -------
    connectivity : ndarray (n, n)
        Number of edges formed between atoms in a system.
    """
    cov_radii = covalent_radii * scale_cov_radii
    numbers = atoms.numbers
    index, coords = expand_cell(atoms)[:2]

    if cutoff is None:
        if radii is None:
            radii = cov_radii[numbers]
            radii = np.repeat(radii[:, None], len(index) / len(radii), axis=1)
            radii = radii.T.flatten()
        else:
            radii = np.tile(radii, int(len(index) / len(atoms))) * scale_cov_radii
    else:
        radii = np.ones_like(index) * cutoff / 2.0

    connectivity = np.zeros((len(atoms), len(atoms)), dtype=int)
    for i, center in enumerate(atoms.positions):
        norm = np.linalg.norm(coords - center, axis=1)
        boundry_distance = norm - radii - radii[i]
        neighbors = index[np.where(boundry_distance < atol)]

        unique, counts = np.unique(neighbors, return_counts=True)
        connectivity[i, unique] = counts
        # Remove self-interaction
        connectivity[i, i] -= 1

    return connectivity


def get_connectivity(
    atoms,
    method="cutoff",
    cutoff_dict={},
    mask=None,
    scale_cov_radii=1.0,
    add_cov_radii=0.2,
    attach_graph=True,
    atoms_template=None,
    print_matrix=False,
    bond_indices=None,
):

    if method == "cutoff":

        cutoff_dict_copy = cutoff_dict.copy()
        for symbol in [
            symbol
            for symbol in atoms.get_chemical_symbols()
            if symbol not in cutoff_dict
        ]:
            cutoff_dict_copy[symbol] = covalent_radii[atomic_numbers[symbol]]

        radii = [
            cutoff_dict_copy[a.symbol] * scale_cov_radii + add_cov_radii for a in atoms
        ]

        if mask is not None:
            radii = [radii[i] if mask[i] else 0.0 for i in range(len(radii))]

        connectivity = get_cutoff_neighbors(atoms, radii=radii)

        if atoms_template is not None:
            n_atoms = len(atoms_template)
            connectivity[:n_atoms, :n_atoms] = atoms_template.connectivity

    elif method == "voronoi":

        connectivity = get_voronoi_neighbors(atoms)

    if bond_indices is not None:
        aa, bb = bond_indices
        connectivity[aa, bb] = 1.0
        connectivity[bb, aa] = 1.0

    if attach_graph is True:
        atoms._graph = from_numpy_matrix(
            A=connectivity,
            create_using=nx.MultiGraph,
            parallel_edges=True,
        )

    return connectivity


def plot_connectivity(atoms, connectivity=None, max_dist=None):

    import matplotlib.pyplot as plt

    atoms = atoms.copy()

    if connectivity is not None:
        graph = from_numpy_matrix(connectivity)
    else:
        graph = atoms.graph

    if max_dist is None:
        max_dist = np.max(atoms.cell.lengths())

    pos = atoms.positions
    nodes_xyz = np.array([pos[v] for v in sorted(graph)])
    edges_xyz = np.array(
        [
            (pos[u], pos[v])
            for u, v in graph.edges()
            if np.linalg.norm(pos[u] - pos[v]) < max_dist
        ]
    )

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*nodes_xyz.T, s=500, ec="w")

    for edge in edges_xyz:
        ax.plot(*edge.T, color="tab:gray")

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
