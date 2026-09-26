# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import time
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.morse import MorsePotential

import sella
from sella import Sella
from arkimede.workflow.sella import modify_hessian_obs
from arkimede.workflow.calculations import run_sella_calculation

pytestmark = pytest.mark.filterwarnings(
    "ignore:Saddle point optimizations with eig=False"
)

# -------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------

BONDS_TS = [[0, 1, "break"]]

def make_system():
    """
    H2 molecule, bond vector v and a Hessian with a negative mode orthogonal to v.
    """
    atoms = Atoms("H2", positions=[[0., 0., 0.], [1.2, 0., 0.]])
    atoms.calc = MorsePotential()
    v = np.array([-1., 0., 0., 1., 0., 0.]) / np.sqrt(2.)
    w = np.array([0., -1., 0., 0., 1., 0.]) / np.sqrt(2.)
    hessian = np.eye(6) - 1.25 * np.outer(w, w)
    return atoms, v, hessian

def make_hessian_with_lowest_mode(v, dot_prod):
    """
    Make a Hessian whose lowest eigenvector has overlap dot_prod with v.
    """
    u = np.array([0., 0., -1., 0., 0., 1.]) / np.sqrt(2.)
    evec = dot_prod * v + np.sqrt(1. - dot_prod ** 2) * u
    return np.eye(6) - 2. * np.outer(evec, evec)

def read_hessian(opt):
    """
    Get a copy of the current Hessian.
    """
    return np.array(opt.pes.H.asarray(), copy=True)

def smoothed_threshold(dot_prod, dot_prod_thr=0.5):
    """
    Smoothed threshold used by modify_hessian_obs (smooth_thr=True).
    """
    p_exp = 2 + 1 / (1 - dot_prod_thr) ** 2
    return (dot_prod ** p_exp + dot_prod_thr ** p_exp) ** (1 / p_exp)

class DeviceResidentHessian:
    """
    Hessian stored on the GPU (B is None), as after a GPU quasi-Newton update.
    """
    def __init__(self, matrix):
        self.initialized = True
        self.B = None
        self.current = np.array(matrix, copy=True)

    def asarray(self):
        return self.current

class DeviceResidentSella(Sella):
    """
    Sella with a GPU-resident Hessian for the observers called after a step and
    at the end of the run.
    """
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        DeviceResidentSella.instances.append(self)

    def call_observers(self):
        if self.nsteps == 0 or not self.pes.H.initialized:
            super().call_observers()
            return
        approx_hessian = self.pes.H
        self.pes.H = DeviceResidentHessian(approx_hessian.asarray())
        super().call_observers()
        # Restore the Hessian if the observers did not set a new one.
        if isinstance(self.pes.H, DeviceResidentHessian):
            self.pes.H = approx_hessian

    def run(self, *args, **kwargs):
        converged = super().run(*args, **kwargs)
        if self.pes.H.initialized:
            self.pes.H = DeviceResidentHessian(self.pes.H.asarray())
        return converged

@pytest.fixture
def device_resident_sella(monkeypatch):
    DeviceResidentSella.instances = []
    monkeypatch.setattr(sella, "Sella", DeviceResidentSella)
    return DeviceResidentSella

def run_sella(atoms, hessian, tmp_path, opt_kwargs={}, **kwargs):
    """
    Run run_sella_calculation on the H2 system.
    """
    opt_kwargs = {"internal": False, "eig": False, **opt_kwargs}
    if hessian is not None:
        opt_kwargs["H0"] = hessian
    kwargs = {
        "atoms": atoms,
        "calc": atoms.calc,
        "bonds_TS": BONDS_TS,
        "max_steps": 0,
        "logfile": None,
        "directory": str(tmp_path),
        "opt_kwargs": opt_kwargs,
        **kwargs,
    }
    run_sella_calculation(**kwargs)

# -------------------------------------------------------------------------------------
# OBSERVER TESTS
# -------------------------------------------------------------------------------------

def test_first_observer_call_makes_bond_curvature_negative():
    atoms, v, hessian = make_system()
    opt = Sella(atoms, internal=False, eig=False, logfile=None)
    opt.pes.set_H(target=hessian)
    before = read_hessian(opt)
    assert v @ before @ v > 0.
    modify_hessian_obs(opt=opt, bonds_TS=BONDS_TS)
    after = read_hessian(opt)
    assert v @ after @ v < -1e-8
    values, vectors = np.linalg.eigh(after)
    assert values[0] < 0.
    assert abs(v @ vectors[:, 0]) >= 0.5

def test_observer_modifies_device_resident_hessian():
    atoms, v, hessian = make_system()
    opt = Sella(atoms, internal=False, eig=False, logfile=None)
    opt.pes.H = DeviceResidentHessian(hessian)
    modify_hessian_obs(opt=opt, bonds_TS=BONDS_TS)
    after = read_hessian(opt)
    assert v @ after @ v < -1e-8
    values, vectors = np.linalg.eigh(after)
    assert abs(v @ vectors[:, 0]) >= 0.5

def test_observer_same_result_for_cpu_and_device_resident_hessian():
    results = []
    for device_resident in (False, True):
        atoms, v, hessian = make_system()
        opt = Sella(atoms, internal=False, eig=False, logfile=None)
        if device_resident is True:
            opt.pes.H = DeviceResidentHessian(hessian)
        else:
            opt.pes.set_H(target=hessian)
        modify_hessian_obs(opt=opt, bonds_TS=BONDS_TS)
        results.append(read_hessian(opt))
    np.testing.assert_allclose(results[0], results[1], rtol=0., atol=1e-12)

def test_observer_skips_aligned_lowest_mode(monkeypatch):
    # The lowest eigenvector is the bond vector: the smoothed threshold is > 1.
    atoms, v, _ = make_system()
    hessian = np.eye(6) - 2. * np.outer(v, v)
    opt = Sella(atoms, internal=False, eig=False, logfile=None)
    opt.pes.set_H(target=hessian)
    calls = []
    monkeypatch.setattr(opt.pes, "set_H", lambda *args, **kw: calls.append(kw))
    start = time.perf_counter()
    modify_hessian_obs(opt=opt, bonds_TS=BONDS_TS)
    wall = time.perf_counter() - start
    assert calls == []
    assert wall < 0.5
    np.testing.assert_array_equal(read_hessian(opt), hessian)

@pytest.mark.parametrize(
    "dot_prod, modified",
    [(0.99738, False), (0.9973, True)],
)
def test_observer_smoothed_threshold_boundary(monkeypatch, dot_prod, modified):
    # With dot_prod_thr=0.5, the smoothed threshold reaches 1 at dot_prod ~ 0.99738.
    assert (smoothed_threshold(dot_prod) >= 1.) is not modified
    atoms, v, _ = make_system()
    hessian = make_hessian_with_lowest_mode(v=v, dot_prod=dot_prod)
    evec = np.linalg.eigh(hessian)[1][:, 0]
    assert abs(v @ evec) == pytest.approx(dot_prod, abs=1e-12)
    opt = Sella(atoms, internal=False, eig=False, logfile=None)
    opt.pes.set_H(target=hessian)
    set_H = opt.pes.set_H
    calls = []
    def set_H_spy(*args, **kwargs):
        calls.append(kwargs)
        set_H(*args, **kwargs)
    monkeypatch.setattr(opt.pes, "set_H", set_H_spy)
    modify_hessian_obs(opt=opt, bonds_TS=BONDS_TS)
    assert (len(calls) == 1) is modified
    if modified is True:
        after = read_hessian(opt)
        evec = np.linalg.eigh(after)[1][:, 0]
        assert abs(v @ evec) >= smoothed_threshold(dot_prod)

# -------------------------------------------------------------------------------------
# RUN SELLA CALCULATION TESTS
# -------------------------------------------------------------------------------------

@pytest.mark.parametrize("copy_atoms", [True, False])
def test_run_sella_stores_modified_hessian(tmp_path, copy_atoms):
    atoms, v, hessian = make_system()
    run_sella(
        atoms=atoms,
        hessian=hessian,
        tmp_path=tmp_path,
        copy_atoms=copy_atoms,
        modify_hessian=True,
        store_hessian=True,
    )
    assert atoms.info["status"] in ("finished", "unfinished")
    stored = atoms.info["hessian"]
    assert isinstance(stored, np.ndarray)
    assert stored.shape == (6, 6)
    assert np.isfinite(stored).all()
    assert v @ stored @ v < -1e-8

@pytest.mark.parametrize("modify_hessian", [True, False])
@pytest.mark.parametrize("copy_atoms", [True, False])
def test_run_sella_stores_device_resident_hessian(
    tmp_path, device_resident_sella, copy_atoms, modify_hessian,
):
    atoms, v, _ = make_system()
    # Sella 2.3.5 gives NaNs when projecting out the rotations of H2, and needs
    # a single negative mode to take a step.
    w = np.array([0., -1., 0., 0., 1., 0.]) / np.sqrt(2.)
    evec = 0.8 * v + 0.6 * w
    hessian = 0.5 * np.eye(6) - np.outer(evec, evec)
    run_sella(
        atoms=atoms,
        hessian=hessian,
        tmp_path=tmp_path,
        copy_atoms=copy_atoms,
        modify_hessian=modify_hessian,
        store_hessian=True,
        max_steps=1,
        opt_kwargs={"proj_rot": False},
    )
    assert atoms.info["status"] in ("finished", "unfinished")
    opt, = device_resident_sella.instances
    assert opt.nsteps == 1
    stored = atoms.info["hessian"]
    current = read_hessian(opt)
    np.testing.assert_array_equal(stored, current)
    assert not np.shares_memory(stored, opt.pes.H.asarray())
    if modify_hessian is True:
        values, vectors = np.linalg.eigh(stored)
        assert abs(v @ vectors[:, 0]) >= 0.5

def test_run_sella_stores_hessian_without_modify_hessian(tmp_path):
    atoms, v, hessian = make_system()
    run_sella(
        atoms=atoms,
        hessian=hessian,
        tmp_path=tmp_path,
        modify_hessian=False,
        store_hessian=True,
    )
    np.testing.assert_array_equal(atoms.info["hessian"], hessian)

def test_run_sella_does_not_store_hessian_by_default(tmp_path):
    atoms, v, hessian = make_system()
    run_sella(
        atoms=atoms,
        hessian=hessian,
        tmp_path=tmp_path,
        modify_hessian=True,
        store_hessian=False,
    )
    assert "hessian" not in atoms.info

def test_run_sella_does_not_store_uninitialized_hessian(tmp_path):
    atoms, v, _ = make_system()
    run_sella(
        atoms=atoms,
        hessian=None,
        tmp_path=tmp_path,
        modify_hessian=True,
        store_hessian=True,
    )
    assert atoms.info["status"] in ("finished", "unfinished")
    assert "hessian" not in atoms.info

@pytest.mark.parametrize(
    "modify_hessian_kwargs, curvature",
    [({}, -1.), ({"iter_max": 1}, 0.), ({"delta": 3.}, -2.)],
)
def test_run_sella_passes_modify_hessian_kwargs(
    tmp_path, modify_hessian_kwargs, curvature,
):
    # Curvature along v is 1: two subtractions are needed with the default delta.
    atoms, v, hessian = make_system()
    run_sella(
        atoms=atoms,
        hessian=hessian,
        tmp_path=tmp_path,
        modify_hessian=True,
        modify_hessian_kwargs=modify_hessian_kwargs,
        store_hessian=True,
    )
    assert v @ atoms.info["hessian"] @ v == pytest.approx(curvature, abs=1e-10)
