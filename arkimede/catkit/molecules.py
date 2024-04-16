import networkx as nx
import itertools
import numpy as np
from ase.build.tools import rotation_matrix
from ase.data import covalent_radii
from .gratoms import Gratoms
from .utils import get_basis_vectors, get_atomic_numbers


def bin_hydrogen(hydrogens=1, bins=1):
    """Recursive function for determining distributions of
    hydrogens across bins.
    """
    if bins == 1:
        yield [hydrogens]

    elif hydrogens == 0:
        yield [0] * bins

    else:
        for i in range(hydrogens + 1):
            for j in bin_hydrogen(hydrogens - i, 1):
                for k in bin_hydrogen(i, bins - 1):
                    yield j + k


def hydrogenate(atoms, bins, copy=True):
    """Add hydrogens to a gratoms object via provided bins"""
    h_index = len(atoms)

    edges = []
    for i, j in enumerate(bins):
        for _ in range(j):
            edges += [(i, h_index)]
            h_index += 1

    if copy:
        atoms = atoms.copy()
    atoms += Gratoms("H{}".format(sum(bins)))
    atoms.graph.add_edges_from(edges)

    return atoms


def get_topologies(symbols, saturate=False):
    """Return the possible topologies of a given chemical species.

    Parameters
    ----------
    symbols : str
        Atomic symbols to construct the topologies from.
    saturate : bool
        Saturate the molecule with hydrogen based on the
        default.radicals set.

    Returns
    -------
    molecules : list (N,)
        Gratoms objects with unique connectivity matrix attached.
        No 3D positions will be provided for these structures.
    """
    num, cnt = get_atomic_numbers(symbols, True)
    mcnt = cnt[num != 1]
    mnum = num[num != 1]

    if cnt[num == 1]:
        hcnt = cnt[num == 1][0]
    else:
        hcnt = 0

    radicals = np.ones(92)
    radicals[[6, 7, 8, 9, 15, 16]] = [4, 3, 2, 1, 3, 2]

    elements = np.repeat(mnum, mcnt)
    max_degree = radicals[elements]
    n = mcnt.sum()

    hmax = int(max_degree.sum() - (n - 1) * 2)
    if hcnt > hmax:
        hcnt = hmax

    if saturate:
        hcnt = hmax

    if n == 1:
        atoms = Gratoms(elements, cell=[1, 1, 1])
        hatoms = hydrogenate(atoms, np.array([hcnt]))
        return [hatoms]
    elif n == 0:
        hatoms = Gratoms("H{}".format(hcnt))
        if hcnt == 2:
            hatoms.graph.add_edge(0, 1, bonds=1)
        return [hatoms]

    ln = np.arange(n).sum()
    il = np.tril_indices(n, -1)

    backbones, molecules = [], []
    combos = itertools.combinations(np.arange(ln), n - 1)
    for c in combos:
        # Construct the connectivity matrix
        ltm = np.zeros(ln)
        ltm[np.atleast_2d(c)] = 1

        connectivity = np.zeros((n, n))
        connectivity[il] = ltm
        connectivity = np.maximum(connectivity, connectivity.T)

        degree = connectivity.sum(axis=0)

        # Not fully connected (subgraph)
        try:
            connectivity_nx = nx.from_numpy_matrix(connectivity)
        except:
            connectivity_nx = nx.from_numpy_array(connectivity)
        if np.any(degree == 0) or not nx.is_connected(connectivity_nx):
            continue

        # Overbonded atoms.
        remaining_bonds = (max_degree - degree).astype(int)
        if np.any(remaining_bonds < 0):
            continue

        atoms = Gratoms(numbers=elements, edges=connectivity, cell=[1, 1, 1])

        isomorph = False
        for G0 in backbones:
            if atoms.is_isomorph(G0):
                isomorph = True
                break

        if not isomorph:
            backbones += [atoms]

            # The backbone is saturated, do not enumerate
            if hcnt == hmax:
                hatoms = hydrogenate(atoms, remaining_bonds)
                molecules += [hatoms]
                continue

            # Enumerate hydrogens across backbone
            for bins in bin_hydrogen(hcnt, n):
                if not np.all(bins <= remaining_bonds):
                    continue

                hatoms = hydrogenate(atoms, bins)

                isomorph = False
                for G0 in molecules:
                    if hatoms.is_isomorph(G0):
                        isomorph = True
                        break

                if not isomorph:
                    molecules += [hatoms]

    return molecules


def get_3D_positions(atoms, bond_index=None, site_natoms=0):
    """Return an estimation of the 3D structure of a Gratoms object
    based on its graph.

    WARNING: This function operates on the atoms object in-place.

    Parameters
    ----------
    atoms : Gratoms object
        Structure with connectivity matrix to provide a 3D structure.
    bond_index : int, list
        Index of the atoms to consider as the origin of a surface.
        bonding site.
    site_natoms : int
        Number of atoms involved in the site at which the molecule is adsorbed.

    Returns
    -------
    atoms : Gratoms object
        Structure with updated 3D positions.
    """
    if isinstance(bond_index, list):
        bond_index_0 = bond_index[0]
    else:
        bond_index_0 = bond_index

    branches = nx.dfs_successors(atoms.graph, bond_index_0)

    complete = []
    for i, branch in enumerate(branches.items()):
        root, nodes = branch

        if len(nodes) == 0:
            continue

        c0 = atoms[root].position
        if i == 0:
            basis = get_basis_vectors([c0, [0, 0, -1]])
        else:
            for j, base_root in enumerate(complete):
                if root in branches[base_root]:
                    c1 = atoms[base_root].position
                    # Flip the basis for every alternate step down the chain.
                    basis = get_basis_vectors([c0, c1])
                    if (i - j) % 2 != 0:
                        basis[2] *= -1
                    break
        complete.insert(0, root)

        atoms.positions[nodes] = branch_molecule(atoms, branch, basis, site_natoms)
        site_natoms = 0

    # if two bonds with surface are present, rotate molecule.
    if isinstance(bond_index, list) and len(bond_index) == 2:
        atoms = rotate_bidentate(atoms, bond_index)
        atoms = branch_bidentate(atoms, bond_index)

    return atoms


def rotate_bidentate(atoms, bond_index):

    atoms.translate(-atoms[bond_index[0]].position)
    d_bond = atoms[bond_index[1]].position - atoms[bond_index[0]].position
    atoms.rotate(d_bond, [0, 1, 0])
    a1 = np.zeros(3)
    links = [
        bb
        for bb in atoms.graph.neighbors(bond_index[0])
        if bb in atoms.graph.neighbors(bond_index[1])
    ]
    if len(links) > 0:
        a1 += sum([atoms[int(ii)].position for ii in links])
    else:
        for bond in bond_index:
            succ = nx.dfs_successors(atoms.graph, bond)
            if bond in succ:
                a1 += sum([atoms[int(ii)].position for ii in succ[bond]])
    a1[1] = 0.0
    if (np.abs(a1) > 0.0).any():
        a2 = [0, 0, 1]
        b1 = [0, 1, 0]
        b2 = [0, 1, 0]
        rot_matrix = rotation_matrix(a1, a2, b1, b2)
        for kk, _ in enumerate(atoms):
            atoms[kk].position = np.dot(atoms[kk].position, rot_matrix.T)

    return atoms


def branch_bidentate(atoms, bond_index):

    links = [
        bb
        for bb in atoms.graph.neighbors(bond_index[0])
        if bb in atoms.graph.neighbors(bond_index[1])
    ]

    # Temporarily break adsorbate bond.
    if len(links) == 0:
        atoms.graph.remove_edge(*bond_index)
        centre = (
            +atoms[int(bond_index[1])].position - atoms[int(bond_index[0])].position
        ) / 2.0
    else:
        for bb in links:
            atoms.graph.remove_edge(bond_index[0], bb)
            atoms.graph.remove_edge(bond_index[1], bb)
        centre = sum([atoms[int(i)].position for i in links]) / len(links)

    for jj, bond in enumerate(bond_index):
        atoms.translate(-atoms[bond].position)
        indices = []
        succ = nx.dfs_successors(atoms.graph, bond)
        if bond in succ:
            a1 = sum([atoms[int(i)].position for i in succ[bond]])
        else:
            continue
        for ss in succ:
            indices += succ[ss]
        if (np.abs(a1) > 0.0).any():
            a2 = [0, -1, +1] if jj == 0 else [0, +1, +1]
            b1 = atoms[int(bond)].position - centre
            b2 = atoms[int(bond)].position - centre
            rot_matrix = rotation_matrix(a1, a2, b1, b2)
            for kk in indices:
                atoms[int(kk)].position = np.dot(atoms[int(kk)].position, rot_matrix.T)

    # Restore adsorbate bond.
    if len(links) == 0:
        atoms.graph.add_edge(*bond_index)
    else:
        for kk in links:
            atoms.graph.add_edge(bond_index[0], bb)
            atoms.graph.add_edge(bond_index[1], bb)

    atoms.translate(-atoms[bond_index[0]].position)

    return atoms


def branch_molecule(atoms, branch, basis=None, site_natoms=0):
    """Return the positions of a Gratoms object for a segment of its
    attached graph. This function is mean to be iterated over by a depth
    first search form NetworkX.

    Parameters
    ----------
    atoms : Gratoms object
        Gratoms object to be iterated over. Will have its positions
        altered in-place.
    branch : tuple (1, [N,])
        A single entry from the output of nx.bfs_successors. The
        first entry is the root node and the following list is the
        nodes branching from the root.
    basis : ndarray (3, 3)
        The basis vectors to use for this step of the branching.
    site_natoms : int
        Number of atoms involved in the site at which the molecule is adsorbed.

    Returns
    -------
    positions : ndarray (N, 3)
        Estimated positions for the branch nodes.
    """
    root, nodes = branch
    root_position = atoms[root].position

    atomic_numbers = atoms.numbers[[root] + nodes]
    atomic_radii = covalent_radii[atomic_numbers]
    dist = (atomic_radii[0] + atomic_radii[1:])[:, None]

    if len(nodes) == 1:
        # Linear bond arrangement
        angles = [180]
        dihedral = [0]
        if atoms[root].symbol == "O" and site_natoms < 3:
            angles = [120]
    elif len(nodes) == 2:
        # Trigonal-planer bond arrangement
        angles = [120, 120]
        dihedral = [0, 180]
        if atoms[root].symbol == "O" and site_natoms < 3:
            angles = [109.47, 109.47]
            dihedral = [0, 120]
    else:
        # Tetrahedral bond arrangement
        angles = [109.47, 109.47, 109.47, 0]
        dihedral = [0, 120, -120, 0]
    
    if site_natoms > 0:
        # Move adsorption structures away from surface
        angles = [aa+10 if aa != 180 else aa for aa in angles]

    # Place the atoms of this segment of the branch
    if basis is None:
        basis = get_basis_vectors([root_position, [0, 0, -1]])
    basis = np.repeat(basis[None, :, :], len(dist), axis=0)

    ang = np.deg2rad(angles)[: len(dist), None]
    tor = np.deg2rad(dihedral)[: len(dist), None]

    basis[:, 1] *= -np.sin(tor)
    basis[:, 2] *= np.cos(tor)

    vectors = basis[:, 1] + basis[:, 2]
    vectors *= dist * np.sin(ang)
    basis[:, 0] *= dist * np.cos(ang)

    positions = vectors + root_position - basis[:, 0]

    return positions
