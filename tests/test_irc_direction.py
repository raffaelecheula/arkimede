"""
Tests for running the forward and reverse IRC with the same Sella IRC object.

Sella fixes the sign of the initial IRC displacement by making the first non-zero
component of the lowest Hessian eigenvector positive. On a slab with fixed atoms
that component is numerically zero, so its sign follows floating-point noise, and
two IRC calculations started from the same TS can follow the same side of the
barrier. The model is a Cu adatom at the bridge site of a 3x3 Cu(100) slab with EMT
and a fixed bottom layer (CPU only). A small, seeded noise on the forces stands in
for the run-to-run noise of GPU calculators.
"""

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os

# Run on CPU (read when Sella is imported).
os.environ.setdefault("SELLA_DISABLE_GPU", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
from ase.build import add_adsorbate, fcc100
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS
from sella import Sella

from arkimede.utilities import check_same_connectivity
from arkimede.workflow import calculations, checks
from arkimede.workflow.recipes import run_calculation

# -------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------

NOISE_SEEDS = range(8)
IRC_STEPS = 5
LABELS = {"forward": "fwd", "reverse": "rev"}

class NoisyEMT(Calculator):
    """
    EMT with seeded Gaussian noise on the forces (no noise if seed is None).
    The positions of every evaluation are stored in positions_list, and the number
    of evaluations in counter.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, seed=None, scale=1e-10):
        super().__init__()
        self.rng = None if seed is None else np.random.default_rng(seed)
        self.scale = scale
        self.positions_list = []
        self.counter = 0

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # New EMT calculator and arrays at each call (EMT reuses its force array).
        atoms_emt = self.atoms.copy()
        atoms_emt.calc = EMT()
        forces = np.array(atoms_emt.get_forces(), copy=True)
        if self.rng is not None:
            forces += self.scale * self.rng.standard_normal(forces.shape)
        self.results["energy"] = atoms_emt.get_potential_energy()
        self.results["forces"] = forces
        self.positions_list.append(np.array(self.atoms.positions, copy=True))
        self.counter += 1

class FailingEMT(NoisyEMT):
    """
    NoisyEMT raising an error at one evaluation (index_fail, counted from zero).
    """
    def __init__(self, index_fail, **kwargs):
        super().__init__(**kwargs)
        self.index_fail = index_fail
        self.n_calls = 0

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        self.n_calls += 1
        if self.n_calls - 1 == self.index_fail:
            raise RuntimeError("calculation failed")
        super().calculate(atoms, properties, system_changes)

@pytest.fixture(scope="module")
def reaction():
    """
    TS, IS and FS of a Cu adatom hopping between two hollow sites of Cu(100).
    """
    atoms = fcc100("Cu", size=(3, 3, 2), vacuum=6.0)
    add_adsorbate(atoms, "Cu", 1.9, "bridge")
    atoms.set_constraint(FixAtoms(indices=[aa.index for aa in atoms if aa.tag == 2]))
    atoms.positions[-1, 1] += 0.05
    atoms.calc = EMT()
    Sella(atoms, order=1, internal=False, logfile=None).run(fmax=1e-4, steps=300)
    assert np.linalg.norm(atoms.get_forces(), axis=1).max() < 1e-4
    atoms_list = [atoms]
    for shift in (-0.6, +0.6):
        atoms_min = atoms.copy()
        atoms_min.positions[-1, 1] += shift
        atoms_min.calc = EMT()
        BFGS(atoms_min, logfile=None).run(fmax=0.01, steps=200)
        atoms_list.append(atoms_min)
    # The IS and FS are different hollow sites.
    assert not check_same_connectivity(
        atoms_1=atoms_list[1],
        atoms_2=atoms_list[2],
        indices="all",
    )
    for atoms_ii in atoms_list:
        atoms_ii.calc = None
    return atoms_list

def cosine(vector_1, vector_2):
    """
    Cosine of the angle between two displacements.
    """
    vector_1, vector_2 = np.ravel(vector_1), np.ravel(vector_2)
    return vector_1 @ vector_2 / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)

def get_initial_steps(positions_list, positions_TS):
    """
    Initial forward and reverse displacements of a run with both directions.
    The forward run first diagonalizes the Hessian at the TS with finite
    differences (eta = 1e-4 Å), and its first larger displacement is the initial
    IRC step. The reverse run returns to the TS and takes its initial step right
    after the last evaluation at the TS.
    """
    step_fwd = next(
        positions for positions in positions_list
        if np.linalg.norm(positions - positions_TS) > 1e-3
    ) - positions_TS
    index_TS = max(
        ii for ii, positions in enumerate(positions_list)
        if np.array_equal(positions, positions_TS)
    )
    step_rev = positions_list[index_TS + 1] - positions_TS
    return step_fwd, step_rev

def read_trajectories(directory, label="irc"):
    """
    Read the forward and reverse trajectories.
    """
    return [
        read(os.path.join(directory, f"{label}_{LABELS[direction]}.traj"), ":")
        for direction in ("forward", "reverse")
    ]

# -------------------------------------------------------------------------------------
# RUN IRC BOTH DIRECTIONS
# -------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_both_directions_start_along_opposite_displacements(reaction, seed):
    atoms_TS = reaction[0]
    calc = NoisyEMT(seed=seed)
    calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=calc,
        max_steps=IRC_STEPS,
        logfile=None,
    )
    step_fwd, step_rev = get_initial_steps(calc.positions_list, atoms_TS.positions)
    assert np.linalg.norm(step_fwd) > 1e-3
    assert abs(cosine(step_fwd, step_rev) + 1.) <= 1e-6
    np.testing.assert_allclose(step_rev, -step_fwd, rtol=0., atol=1e-12)

@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_both_directions_write_one_trajectory_per_direction(reaction, tmp_path, seed):
    atoms_TS = reaction[0].copy()
    positions_TS = atoms_TS.positions.copy()
    atoms_irc_list = calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=NoisyEMT(seed=seed),
        max_steps=IRC_STEPS,
        logfile=None,
        save_trajs=True,
        directory=tmp_path,
    )
    images_fwd, images_rev = read_trajectories(tmp_path)
    # The input atoms are not modified.
    np.testing.assert_array_equal(atoms_TS.positions, positions_TS)
    assert atoms_TS.calc is None
    assert "status" not in atoms_TS.info
    # Each trajectory starts at the TS and ends at the returned end point.
    assert len(atoms_irc_list) == 2
    assert len(images_fwd) == len(images_rev) == IRC_STEPS + 1
    for images, atoms_irc in zip((images_fwd, images_rev), atoms_irc_list):
        np.testing.assert_array_equal(images[0].positions, positions_TS)
        np.testing.assert_array_equal(images[-1].positions, atoms_irc.positions)
        assert atoms_irc.info["status"] == "unfinished"
        assert atoms_irc.get_potential_energy() == images[-1].get_potential_energy()
        np.testing.assert_array_equal(atoms_irc.get_forces(), images[-1].get_forces())
    # Each trajectory contains only the images of its own direction.
    step_fwd = images_fwd[1].positions - positions_TS
    assert all(
        cosine(image.positions - positions_TS, step_fwd) > 0.
        for image in images_fwd[1:]
    )
    assert all(
        cosine(image.positions - positions_TS, step_fwd) < 0.
        for image in images_rev[1:]
    )

def test_both_directions_match_separate_calculations_without_noise(reaction, tmp_path):
    atoms_TS = reaction[0]
    # Two separate single-direction calculations.
    atoms_sep_list = []
    for direction in ("forward", "reverse"):
        atoms = atoms_TS.copy()
        calculations.run_irc_calculation(
            atoms=atoms,
            calc=NoisyEMT(),
            direction=direction,
            max_steps=IRC_STEPS,
            logfile=None,
            label=f"irc_{LABELS[direction]}",
            save_trajs=True,
            directory=tmp_path / "separate",
        )
        atoms_sep_list.append(atoms)
    # Both directions with one IRC object.
    atoms_irc_list = calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=NoisyEMT(),
        max_steps=IRC_STEPS,
        logfile=None,
        save_trajs=True,
        directory=tmp_path / "both",
    )
    traj_list_sep = read_trajectories(tmp_path / "separate")
    traj_list_both = read_trajectories(tmp_path / "both")
    for images_sep, images_both in zip(traj_list_sep, traj_list_both):
        assert len(images_both) == len(images_sep) == IRC_STEPS + 1
        for image_sep, image_both in zip(images_sep, images_both):
            np.testing.assert_allclose(
                image_both.positions, image_sep.positions, rtol=0., atol=1e-10,
            )
            np.testing.assert_allclose(
                image_both.get_potential_energy(),
                image_sep.get_potential_energy(),
                rtol=0.,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                image_both.get_forces(), image_sep.get_forces(), rtol=0., atol=1e-10,
            )
    for atoms_sep, atoms_irc in zip(atoms_sep_list, atoms_irc_list):
        np.testing.assert_allclose(
            atoms_irc.positions, atoms_sep.positions, rtol=0., atol=1e-10,
        )
        assert atoms_irc.info["status"] == atoms_sep.info["status"]
    # The reverse direction does not diagonalize the Hessian again.
    counter_sep = [atoms.info["counter"] for atoms in atoms_sep_list]
    counter_both = [atoms.info["counter"] for atoms in atoms_irc_list]
    assert counter_both[0] == counter_sep[0]
    assert counter_both[1] < counter_sep[1]

def test_both_directions_count_force_calls_per_direction(reaction, tmp_path):
    atoms_TS = reaction[0]
    # Reference run without limits.
    calc = NoisyEMT()
    atoms_irc_list = calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=calc,
        max_steps=IRC_STEPS,
        logfile=None,
    )
    counter_list = [atoms.info["counter"] for atoms in atoms_irc_list]
    assert sum(counter_list) == len(calc.positions_list)
    # Only the forward direction includes the diagonalization of the Hessian.
    assert counter_list[1] < counter_list[0]
    # With a limit equal to the calls of the reverse direction, the forward
    # direction stops early and the reverse one takes all its steps, since the
    # counter is reset at the start of each direction.
    for reset_counter in (True, False):
        directory = tmp_path / f"reset_{reset_counter}"
        atoms_irc_list = calculations.run_irc_both_directions(
            atoms=atoms_TS,
            calc=NoisyEMT(),
            max_steps=IRC_STEPS,
            logfile=None,
            max_forcecalls=counter_list[1],
            reset_counter=reset_counter,
            save_trajs=True,
            directory=directory,
        )
        images_fwd, images_rev = read_trajectories(directory)
        assert len(images_fwd) < IRC_STEPS + 1
        if reset_counter is True:
            assert len(images_rev) == IRC_STEPS + 1
            assert atoms_irc_list[1].info["counter"] <= counter_list[1]
        else:
            # Without the reset, the limit applies to both directions together.
            assert len(images_rev) == 1
            assert atoms_irc_list[1].info["counter"] > counter_list[1]

def test_both_directions_after_failed_forward_run(reaction, tmp_path):
    atoms_TS = reaction[0]
    # Reference run without failures.
    calc = NoisyEMT()
    calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=calc,
        max_steps=IRC_STEPS,
        logfile=None,
        save_trajs=True,
        directory=tmp_path / "reference",
    )
    images_rev_ref = read_trajectories(tmp_path / "reference")[1]
    index_step = next(
        ii for ii, positions in enumerate(calc.positions_list)
        if np.linalg.norm(positions - atoms_TS.positions) > 1e-3
    )
    # Failure during the diagonalization: there is no displacement to reverse.
    calc = FailingEMT(index_fail=1)
    atoms_irc_list = calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=calc,
        max_steps=IRC_STEPS,
        logfile=None,
    )
    assert [atoms.info["status"] for atoms in atoms_irc_list] == ["failed", "failed"]
    assert calc.n_calls == 2
    # Failure after the diagonalization: the reverse direction is not affected.
    # The input atoms carry a status and counter of their own.
    atoms_TS = atoms_TS.copy()
    atoms_TS.info.update({"status": "finished", "counter": -1})
    atoms_irc_list = calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=FailingEMT(index_fail=index_step + 1),
        max_steps=IRC_STEPS,
        logfile=None,
        save_trajs=True,
        directory=tmp_path / "failed",
    )
    images_fwd, images_rev = read_trajectories(tmp_path / "failed")
    status_list = [atoms.info["status"] for atoms in atoms_irc_list]
    assert status_list == ["failed", "unfinished"]
    assert atoms_TS.info["status"] == "finished"
    assert len(images_rev) == len(images_rev_ref) == IRC_STEPS + 1
    for image, image_ref in zip(images_rev, images_rev_ref):
        np.testing.assert_allclose(
            image.positions, image_ref.positions, rtol=0., atol=1e-10,
        )
    # The images carry the status and counter of the input atoms, not those of the
    # failed forward run.
    for image in images_fwd + images_rev:
        assert image.info["status"] == "finished"
        assert image.info["counter"] == -1

def test_both_directions_reject_single_trajectory(reaction, tmp_path):
    with pytest.raises(ValueError, match="trajectory"):
        calculations.run_irc_both_directions(
            atoms=reaction[0],
            calc=NoisyEMT(),
            max_steps=IRC_STEPS,
            logfile=None,
            trajectory=str(tmp_path / "irc.traj"),
        )

def test_run_calculation_irc_both_stores_end_points(reaction):
    atoms_TS = reaction[0]
    atoms = atoms_TS.copy()
    run_calculation(
        atoms=atoms,
        calc=NoisyEMT(),
        calculation="irc",
        direction="both",
        max_steps=IRC_STEPS,
        logfile=None,
    )
    atoms_irc_list = calculations.run_irc_both_directions(
        atoms=atoms_TS,
        calc=NoisyEMT(),
        max_steps=IRC_STEPS,
        logfile=None,
    )
    np.testing.assert_array_equal(atoms.positions, atoms_TS.positions)
    assert atoms.info["status_irc"] == ["unfinished", "unfinished"]
    for positions, atoms_irc in zip(atoms.info["positions_irc"], atoms_irc_list):
        np.testing.assert_array_equal(positions, atoms_irc.positions)

# -------------------------------------------------------------------------------------
# CHECK TS RELAX INTO IS AND FS
# -------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_check_irc_leaves_ts_in_opposite_directions(reaction, tmp_path, seed):
    atoms_TS, atoms_IS, atoms_FS = reaction
    checks.check_TS_relax_into_IS_and_FS(
        atoms_TS=atoms_TS.copy(),
        atoms_IS=atoms_IS,
        atoms_FS=atoms_FS,
        calc=NoisyEMT(seed=seed),
        method="irc",
        indices_check="all",
        max_steps=IRC_STEPS,
        logfile=None,
        save_trajs=True,
        directory=tmp_path,
    )
    images_fwd, images_rev = read_trajectories(tmp_path)
    for images in (images_fwd, images_rev):
        np.testing.assert_array_equal(images[0].positions, atoms_TS.positions)
    cosine_steps = cosine(
        images_fwd[1].positions - atoms_TS.positions,
        images_rev[1].positions - atoms_TS.positions,
    )
    assert cosine_steps < 0., (
        f"forward and reverse IRC leave the TS on the same side "
        f"(cosine {cosine_steps:+.4f})"
    )

@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_check_irc_connects_IS_and_FS(reaction, seed):
    atoms_TS, atoms_IS, atoms_FS = reaction
    atoms_TS = atoms_TS.copy()
    check = checks.check_TS_relax_into_IS_and_FS(
        atoms_TS=atoms_TS,
        atoms_IS=atoms_IS,
        atoms_FS=atoms_FS,
        calc=NoisyEMT(seed=seed),
        method="irc",
        indices_check="all",
        store_positions_relaxed=True,
        logfile=None,
    )
    positions_fwd, positions_rev = atoms_TS.info["positions_relaxed"]
    distance = np.linalg.norm(positions_fwd[-1] - positions_rev[-1])
    assert check is True, (
        f"IRC end points do not match IS and FS (adatom end points {distance:.3f} Å "
        f"apart)"
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
