######################################################
# This is copied from the development version of ase.
######################################################

from numpy.linalg import norm
from ase.optimize.bfgs import BFGS
import numpy as np
from warnings import warn
from ase.constraints import FixConstraint, slice2enlist
from ase.geometry import (
    get_distances_derivatives,
    get_angles_derivatives,
    get_dihedrals_derivatives,
    conditional_find_mic,
    get_angles,
    get_dihedrals,
)

class FixInternals(FixConstraint):
    """Constraint object for fixing multiple internal coordinates.

    Allows fixing bonds, angles, dihedrals as well as linear combinations
    of bonds (bondcombos).

    Please provide angular units in degrees using `angles_deg` and
    `dihedrals_deg`.
    Fixing planar angles is not supported at the moment.
    """

    def __init__(self, bonds=None, angles=None, dihedrals=None,
                 angles_deg=None, dihedrals_deg=None,
                 bondcombos=None,
                 mic=False, epsilon=1.e-7):
        """
        A constrained internal coordinate is defined as a nested list:
        '[value, [atom indices]]'. The constraint is initialized with a list of
        constrained internal coordinates, i.e. '[[value, [atom indices]], ...]'.
        If 'value' is None, the current value of the coordinate is constrained.

        Parameters
        ----------
        bonds: nested python list, optional
            List with targetvalue and atom indices defining the fixed bonds,
            i.e. [[targetvalue, [index0, index1]], ...]

        angles_deg: nested python list, optional
            List with targetvalue and atom indices defining the fixedangles,
            i.e. [[targetvalue, [index0, index1, index3]], ...]

        dihedrals_deg: nested python list, optional
            List with targetvalue and atom indices defining the fixed dihedrals,
            i.e. [[targetvalue, [index0, index1, index3]], ...]

        bondcombos: nested python list, optional
            List with targetvalue, atom indices and linear coefficient defining
            the fixed linear combination of bonds,
            i.e. [[targetvalue, [[index0, index1, coefficient_for_bond],
            [index1, index2, coefficient_for_bond]]], ...]

        mic: bool, optional, default: False
            Minimum image convention.

        epsilon: float, optional, default: 1e-7
            Convergence criterion.
        """
        warn_msg = 'Please specify {} in degrees using the {} argument.'
        if angles:
            warn(FutureWarning(warn_msg.format('angles', 'angle_deg')))
            angles = np.asarray(angles)
            angles[:, 0] = angles[:, 0] / np.pi * 180
            angles = angles.tolist()
        else:
            angles = angles_deg
        if dihedrals:
            warn(FutureWarning(warn_msg.format('dihedrals', 'dihedrals_deg')))
            dihedrals = np.asarray(dihedrals)
            dihedrals[:, 0] = dihedrals[:, 0] / np.pi * 180
            dihedrals = dihedrals.tolist()
        else:
            dihedrals = dihedrals_deg

        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.bondcombos = bondcombos or []
        self.mic = mic
        self.epsilon = epsilon

        self.n = (len(self.bonds) + len(self.angles) + len(self.dihedrals)
                  + len(self.bondcombos))

        # Initialize these at run-time:
        self.constraints = []
        self.initialized = False

    def get_removed_dof(self, atoms):
        return self.n

    def initialize(self, atoms):
        if self.initialized:
            return
        masses = np.repeat(atoms.get_masses(), 3)
        cell = None
        pbc = None
        if self.mic:
            cell = atoms.cell
            pbc = atoms.pbc
        self.constraints = []
        for data, ConstrClass in [(self.bonds, self.FixBondLengthAlt),
                                  (self.angles, self.FixAngle),
                                  (self.dihedrals, self.FixDihedral),
                                  (self.bondcombos, self.FixBondCombo)]:
            for datum in data:
                targetvalue = datum[0]
                if targetvalue is None:  # set to current value
                    targetvalue = ConstrClass.get_value(atoms, datum[1],
                                                        self.mic)
                constr = ConstrClass(targetvalue, datum[1], masses, cell, pbc)
                self.constraints.append(constr)
        self.initialized = True

    @staticmethod
    def get_bondcombo(atoms, indices, mic=False):
        """Convenience function to return the value of the bondcombo coordinate
        (linear combination of bond lengths) for the given Atoms object 'atoms'.
        Example: Get the current value of the linear combination of two bond
        lengths defined as `bondcombo = [[0, 1, 1.0], [2, 3, -1.0]]`."""
        c = sum(df[2] * atoms.get_distance(*df[:2], mic=mic) for df in indices)
        return c

    def get_subconstraint(self, atoms, definition):
        """Get pointer to a specific subconstraint.
        Identification by its definition via indices (and coefficients)."""
        self.initialize(atoms)
        for subconstr in self.constraints:
            if type(definition[0]) == list:  # identify Combo constraint
                defin = [d + [c] for d, c in zip(subconstr.indices,
                                                 subconstr.coefs)]
                if defin == definition:
                    return subconstr
            else:  # identify primitive constraints by their indices
                if subconstr.indices == [definition]:
                    return subconstr
        raise ValueError('Given `definition` not found on Atoms object.')

    def shuffle_definitions(self, shuffle_dic, internal_type):
        dfns = []  # definitions
        for dfn in internal_type:  # e.g. for bond in self.bonds
            append = True
            new_dfn = [dfn[0], list(dfn[1])]
            for old in dfn[1]:
                if old in shuffle_dic:
                    new_dfn[1][dfn[1].index(old)] = shuffle_dic[old]
                else:
                    append = False
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def shuffle_combos(self, shuffle_dic, internal_type):
        dfns = []  # definitions
        for dfn in internal_type:  # i.e. for bondcombo in self.bondcombos
            append = True
            all_indices = [idx[0:-1] for idx in dfn[1]]
            new_dfn = [dfn[0], list(dfn[1])]
            for i, indices in enumerate(all_indices):
                for old in indices:
                    if old in shuffle_dic:
                        new_dfn[1][i][indices.index(old)] = shuffle_dic[old]
                    else:
                        append = False
                        break
                if not append:
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        self.initialize(atoms)
        shuffle_dic = dict(slice2enlist(ind, len(atoms)))
        shuffle_dic = {old: new for new, old in shuffle_dic.items()}
        self.bonds = self.shuffle_definitions(shuffle_dic, self.bonds)
        self.angles = self.shuffle_definitions(shuffle_dic, self.angles)
        self.dihedrals = self.shuffle_definitions(shuffle_dic, self.dihedrals)
        self.bondcombos = self.shuffle_combos(shuffle_dic, self.bondcombos)
        self.initialized = False
        self.initialize(atoms)
        if len(self.constraints) == 0:
            raise IndexError('Constraint not part of slice')

    def get_indices(self):
        cons = []
        for dfn in self.bonds + self.dihedrals + self.angles:
            cons.extend(dfn[1])
        for dfn in self.bondcombos:
            for partial_dfn in dfn[1]:
                cons.extend(partial_dfn[0:-1])  # last index is the coefficient
        return list(set(cons))

    def todict(self):
        return {'name': 'FixInternals',
                'kwargs': {'bonds': self.bonds,
                           'angles_deg': self.angles,
                           'dihedrals_deg': self.dihedrals,
                           'bondcombos': self.bondcombos,
                           'mic': self.mic,
                           'epsilon': self.epsilon}}

    def adjust_positions(self, atoms, newpos):
        self.initialize(atoms)
        for constraint in self.constraints:
            constraint.setup_jacobian(atoms.positions)
        for j in range(50):
            maxerr = 0.0
            for constraint in self.constraints:
                constraint.adjust_positions(atoms.positions, newpos)
                maxerr = max(abs(constraint.sigma), maxerr)
            if maxerr < self.epsilon:
                return
        msg = 'FixInternals.adjust_positions did not converge.'
        if any([constr.targetvalue > 175. or constr.targetvalue < 5. for constr
                in self.constraints if isinstance(constr, self.FixAngle)]):
            msg += (' This may be caused by an almost planar angle.'
                    ' Support for planar angles would require the'
                    ' implementation of ghost, i.e. dummy, atoms.'
                    ' See issue #868.')
        raise ValueError(msg)

    def adjust_forces(self, atoms, forces):
        """Project out translations and rotations and all other constraints"""
        self.initialize(atoms)
        positions = atoms.positions
        N = len(forces)
        list2_constraints = list(np.zeros((6, N, 3)))
        tx, ty, tz, rx, ry, rz = list2_constraints

        list_constraints = [r.ravel() for r in list2_constraints]

        tx[:, 0] = 1.0
        ty[:, 1] = 1.0
        tz[:, 2] = 1.0
        ff = forces.ravel()

        # Calculate the center of mass
        center = positions.sum(axis=0) / N

        rx[:, 1] = -(positions[:, 2] - center[2])
        rx[:, 2] = positions[:, 1] - center[1]
        ry[:, 0] = positions[:, 2] - center[2]
        ry[:, 2] = -(positions[:, 0] - center[0])
        rz[:, 0] = -(positions[:, 1] - center[1])
        rz[:, 1] = positions[:, 0] - center[0]

        # Normalizing transl., rotat. constraints
        for r in list2_constraints:
            r /= np.linalg.norm(r.ravel())

        # Add all angle, etc. constraint vectors
        for constraint in self.constraints:
            constraint.setup_jacobian(positions)
            constraint.adjust_forces(positions, forces)
            list_constraints.insert(0, constraint.jacobian)
        # QR DECOMPOSITION - GRAM SCHMIDT

        list_constraints = [r.ravel() for r in list_constraints]
        aa = np.column_stack(list_constraints)
        (aa, bb) = np.linalg.qr(aa)
        # Projection
        hh = []
        for i, constraint in enumerate(self.constraints):
            hh.append(aa[:, i] * np.row_stack(aa[:, i]))

        txx = aa[:, self.n] * np.row_stack(aa[:, self.n])
        tyy = aa[:, self.n + 1] * np.row_stack(aa[:, self.n + 1])
        tzz = aa[:, self.n + 2] * np.row_stack(aa[:, self.n + 2])
        rxx = aa[:, self.n + 3] * np.row_stack(aa[:, self.n + 3])
        ryy = aa[:, self.n + 4] * np.row_stack(aa[:, self.n + 4])
        rzz = aa[:, self.n + 5] * np.row_stack(aa[:, self.n + 5])
        T = txx + tyy + tzz + rxx + ryy + rzz
        for vec in hh:
            T += vec
        ff = np.dot(T, np.row_stack(ff))
        forces[:, :] -= np.dot(T, np.row_stack(ff)).reshape(-1, 3)

    def __repr__(self):
        constraints = [repr(constr) for constr in self.constraints]
        return f'FixInternals(_copy_init={constraints}, epsilon={self.epsilon})'

    # Classes for internal use in FixInternals
    class FixInternalsBase:
        """Base class for subclasses of FixInternals."""

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            self.targetvalue = targetvalue  # constant target value
            self.indices = [defin[0:-1] for defin in indices]  # indices, defs
            self.coefs = np.asarray([defin[-1] for defin in indices])
            self.masses = masses
            self.jacobian = []  # geometric Jacobian matrix, Wilson B-matrix
            self.sigma = 1.  # difference between current and target value
            self.projected_forces = None  # helps optimizers scan along constr.
            self.cell = cell
            self.pbc = pbc

        def finalize_jacobian(self, pos, n_internals, n, derivs):
            """Populate jacobian with derivatives for `n_internals` defined
            internals. n = 2 (bonds), 3 (angles), 4 (dihedrals)."""
            jacobian = np.zeros((n_internals, *pos.shape))
            for i, idx in enumerate(self.indices):
                for j in range(n):
                    jacobian[i, idx[j]] = derivs[i, j]
            jacobian = jacobian.reshape((n_internals, 3 * len(pos)))
            return self.coefs @ jacobian

        def finalize_positions(self, newpos):
            jacobian = self.jacobian / self.masses
            lamda = -self.sigma / (jacobian @ self.get_jacobian(newpos))
            dnewpos = lamda * jacobian
            newpos += dnewpos.reshape(newpos.shape)

        def adjust_forces(self, positions, forces):
            self.projected_forces = ((self.jacobian @ forces.ravel())
                                     * self.jacobian)
            self.jacobian /= np.linalg.norm(self.jacobian)

    class FixBondCombo(FixInternalsBase):
        """Constraint subobject for fixing linear combination of bond lengths
        within FixInternals.

        sum_i( coef_i * bond_length_i ) = constant
        """

        def get_jacobian(self, pos):
            bondvectors = [pos[k] - pos[h] for h, k in self.indices]
            derivs = get_distances_derivatives(bondvectors, cell=self.cell,
                                               pbc=self.pbc)
            return self.finalize_jacobian(pos, len(bondvectors), 2, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            bondvectors = [newpos[k] - newpos[h] for h, k in self.indices]
            (_, ), (dists, ) = conditional_find_mic([bondvectors],
                                                    cell=self.cell,
                                                    pbc=self.pbc)
            value = self.coefs @ dists
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        @staticmethod
        def get_value(atoms, indices, mic):
            return FixInternals.get_bondcombo(atoms, indices, mic)

        def __repr__(self):
            return (f'FixBondCombo({self.targetvalue}, {self.indices}, '
                    '{self.coefs})')

    class FixBondLengthAlt(FixBondCombo):
        """Constraint subobject for fixing bond length within FixInternals.
        Fix distance between atoms with indices a1, a2."""

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            if targetvalue <= 0.:
                raise ZeroDivisionError('Invalid targetvalue for fixed bond')
            indices = [list(indices) + [1.]]  # bond definition with coef 1.
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        @staticmethod
        def get_value(atoms, indices, mic):
            return atoms.get_distance(*indices, mic=mic)

        def __repr__(self):
            return f'FixBondLengthAlt({self.targetvalue}, {self.indices})'

    class FixAngle(FixInternalsBase):
        """Constraint subobject for fixing an angle within FixInternals.

        Convergence is potentially problematic for angles very close to
        0 or 180 degrees as there is a singularity in the Cartesian derivative.
        Fixing planar angles is therefore not supported at the moment.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            """Fix atom movement to construct a constant angle."""
            if targetvalue <= 0. or targetvalue >= 180.:
                raise ZeroDivisionError('Invalid targetvalue for fixed angle')
            indices = [list(indices) + [1.]]  # angle definition with coef 1.
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def gather_vectors(self, pos):
            v0 = [pos[h] - pos[k] for h, k, l in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l in self.indices]
            return v0, v1

        def get_jacobian(self, pos):
            v0, v1 = self.gather_vectors(pos)
            derivs = get_angles_derivatives(v0, v1, cell=self.cell,
                                            pbc=self.pbc)
            return self.finalize_jacobian(pos, len(v0), 3, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            v0, v1 = self.gather_vectors(newpos)
            value = get_angles(v0, v1, cell=self.cell, pbc=self.pbc)
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        @staticmethod
        def get_value(atoms, indices, mic):
            return atoms.get_angle(*indices, mic=mic)

        def __repr__(self):
            return f'FixAngle({self.targetvalue}, {self.indices})'

    class FixDihedral(FixInternalsBase):
        """Constraint subobject for fixing a dihedral angle within FixInternals.

        A dihedral becomes undefined when at least one of the inner two angles
        becomes planar. Make sure to avoid this situation.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            indices = [list(indices) + [1.]]  # dihedral def. with coef 1.
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def gather_vectors(self, pos):
            v0 = [pos[k] - pos[h] for h, k, l, m in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l, m in self.indices]
            v2 = [pos[m] - pos[l] for h, k, l, m in self.indices]
            return v0, v1, v2

        def get_jacobian(self, pos):
            v0, v1, v2 = self.gather_vectors(pos)
            derivs = get_dihedrals_derivatives(v0, v1, v2, cell=self.cell,
                                               pbc=self.pbc)
            return self.finalize_jacobian(pos, len(v0), 4, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            v0, v1, v2 = self.gather_vectors(newpos)
            value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
            # apply minimum dihedral difference 'convention': (diff <= 180)
            self.sigma = (value - self.targetvalue + 180) % 360 - 180
            self.finalize_positions(newpos)

        @staticmethod
        def get_value(atoms, indices, mic):
            return atoms.get_dihedral(*indices, mic=mic)

        def __repr__(self):
            return f'FixDihedral({self.targetvalue}, {self.indices})'

class BFGSClimbFixInternals(BFGS):
    """Class for transition state search and optimization

    Climbs the 1D reaction coordinate defined as constrained internal coordinate
    via the :class:`~ase.constraints.FixInternals` class while minimizing all
    remaining degrees of freedom.

    Details: Two optimizers, 'A' and 'B', are applied orthogonal to each other.
    Optimizer 'A' climbs the constrained coordinate while optimizer 'B'
    optimizes the remaining degrees of freedom after each climbing step.
    Optimizer 'A' uses the BFGS algorithm to climb along the projected force of
    the selected constraint. Optimizer 'B' can be any ASE optimizer
    (default: BFGS).

    In combination with other constraints, the order of constraints matters.
    Generally, the FixInternals constraint should come first in the list of
    constraints.
    This has been tested with the :class:`~ase.constraints.FixAtoms` constraint.

    Inspired by concepts described by P. N. Plessow. [1]_

    .. [1] Plessow, P. N., Efficient Transition State Optimization of Periodic
           Structures through Automated Relaxed Potential Energy Surface Scans.
           J. Chem. Theory Comput. 2018, 14 (2), 981–990.
           https://doi.org/10.1021/acs.jctc.7b01070.

    .. note::
       Convergence is based on 'fmax' of the total forces which is the sum of
       the projected forces and the forces of the remaining degrees of freedom.
       This value is logged in the 'logfile'. Optimizer 'B' logs 'fmax' of the
       remaining degrees of freedom without the projected forces. The projected
       forces can be inspected using the :meth:`get_projected_forces` method:

       >>> for _ in dyn.irun():
       ...     projected_forces = dyn.get_projected_forces()

    Example
    -------
    .. literalinclude:: ../../ase/test/optimize/test_climb_fix_internals.py
       :end-before: # end example for documentation
    """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None,
                 climb_coordinate=None, index_constr2climb=None,
                 optB=BFGS, optB_kwargs=None, optB_fmax=0.05,
                 optB_fmax_scaling=0.0):
        """Allowed parameters are similar to the parent class
        :class:`~ase.optimize.bfgs.BFGS` with the following additions:

        Parameters
        ----------
        climb_coordinate: list
            Specifies which subconstraint of the
            :class:`~ase.constraints.FixInternals` constraint is to be climbed.
            Provide the corresponding nested list of indices
            (including coefficients in the case of Combo constraints).
            See the example above.

        optB: any ASE optimizer, optional
            Optimizer 'B' for optimization of the remaining degrees of freedom
            after each climbing step.

        optB_kwargs: dict, optional
            Specifies keyword arguments to be passed to optimizer 'B' at its
            initialization. By default, optimizer 'B' writes a logfile and
            trajectory (optB_{...}.log, optB_{...}.traj) where {...} is the
            current value of the ``climb_coordinate``. Set ``logfile`` to '-'
            for console output. Set ``trajectory`` to 'None' to suppress
            writing of the trajectory file.

        optB_fmax: float, optional
            Specifies the convergence criterion 'fmax' of optimizer 'B'.

        optB_fmax_scaling: float, optional
            Scaling factor to dynamically tighten 'fmax' of optimizer 'B' to
            the value of ``optB_fmax`` when close to convergence.
            Can speed up the climbing process. The scaling formula is

            'fmax' = ``optB_fmax`` + ``optB_fmax_scaling``
            :math:`\\cdot` norm_of_projected_forces

            The final optimization with optimizer 'B' is
            performed with ``optB_fmax`` independent of ``optB_fmax_scaling``.
        """
        self.targetvalue = None  # may be assigned during restart in self.read()
        super().__init__(atoms, restart=restart, logfile=logfile,
                         trajectory=trajectory, maxstep=maxstep, master=master,
                         alpha=alpha)

        if index_constr2climb is None:
            self.constr2climb = self.get_constr2climb(
                self.atoms, climb_coordinate)
        else:
            fixinternals = self.get_fixinternals(atoms)
            fixinternals.initialize(atoms)
            self.constr2climb = fixinternals.constraints[index_constr2climb]
        self.targetvalue = self.targetvalue or self.constr2climb.targetvalue

        self.optB = optB
        self.optB_kwargs = optB_kwargs or {}
        self.optB_fmax = optB_fmax
        self.scaling = optB_fmax_scaling
        # log optimizer 'B' in logfiles named after current value of constraint
        self.autolog = 'logfile' not in self.optB_kwargs
        self.autotraj = 'trajectory' not in self.optB_kwargs

    def get_constr2climb(self, atoms, climb_coordinate):
        """Get pointer to the subconstraint that is to be climbed.
        Identification by its definition via indices (and coefficients)."""
        constr = self.get_fixinternals(atoms)
        return constr.get_subconstraint(atoms, climb_coordinate)

    def get_fixinternals(self, atoms):
        """Get pointer to the FixInternals constraint on the atoms object."""
        all_constr_types = list(map(type, atoms.constraints))
        index = all_constr_types.index(FixInternals)  # locate constraint
        return atoms.constraints[index]

    def read(self):
        (self.H, self.pos0, self.forces0, self.maxstep,
         self.targetvalue) = self.load()

    def step(self):
        self.relax_remaining_dof()  # optimization with optimizer 'B'

        pos, dpos = self.pretend2climb()  # with optimizer 'A'
        self.update_positions_and_targetvalue(pos, dpos)  # obey other constr.

        self.dump((self.H, self.pos0, self.forces0, self.maxstep,
                   self.targetvalue))

    def pretend2climb(self):
        """Get directions for climbing and climb with optimizer 'A'."""
        proj_forces = self.get_projected_forces()
        pos = self.atoms.get_positions()
        dpos, steplengths = self.prepare_step(pos, proj_forces)
        dpos = self.determine_step(dpos, steplengths)
        return pos, dpos

    def update_positions_and_targetvalue(self, pos, dpos):
        """Adjust constrained targetvalue of constraint and update positions."""
        self.constr2climb.adjust_positions(pos, pos + dpos)  # update sigma
        self.targetvalue += self.constr2climb.sigma          # climb constraint
        self.constr2climb.targetvalue = self.targetvalue     # adjust positions
        self.atoms.set_positions(self.atoms.get_positions())   # to targetvalue

    def relax_remaining_dof(self):
        """Optimize remaining degrees of freedom with optimizer 'B'."""
        if self.autolog:
            self.optB_kwargs['logfile'] = f'optB_{self.targetvalue}.log'
        if self.autotraj:
            self.optB_kwargs['trajectory'] = f'optB_{self.targetvalue}.traj'
        fmax = self.get_scaled_fmax()
        with self.optB(self.atoms, **self.optB_kwargs) as opt:
            opt.run(fmax)  # optimize with scaled fmax
            if self.converged() and fmax > self.optB_fmax:
                # (final) optimization with desired fmax
                opt.run(self.optB_fmax)

    def get_scaled_fmax(self):
        """Return the adaptive 'fmax' based on the estimated distance to the
        transition state."""
        return (self.optB_fmax +
                self.scaling * norm(self.constr2climb.projected_forces))

    def get_projected_forces(self):
        """Return the projected forces along the constrained coordinate in
        uphill direction (negative sign)."""
        forces = self.constr2climb.projected_forces
        forces = -forces.reshape(self.atoms.positions.shape)
        return forces

    def get_total_forces(self):
        """Return forces obeying all constraints plus projected forces."""
        return self.atoms.get_forces() + self.get_projected_forces()

    def converged(self, forces=None):
        """Did the optimization converge based on the total forces?"""
        forces = forces or self.get_total_forces()
        return super().converged(forces=forces)

    def log(self, forces=None):
        forces = forces or self.get_total_forces()
        super().log(forces=forces)
