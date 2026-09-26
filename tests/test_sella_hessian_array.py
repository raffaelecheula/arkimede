# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import pytest
from types import SimpleNamespace

from arkimede.workflow.sella import get_hessian_array

# -------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------

def make_opt(approx_hessian):
    """
    Wrap a Hessian in an optimizer-like object.
    """
    return SimpleNamespace(pes=SimpleNamespace(H=approx_hessian))

class DeviceResidentHessian:
    """
    Hessian stored on the GPU: B is None, the matrix is given by asarray().
    """
    def __init__(self, matrix=None):
        self.initialized = True
        self.B = None
        self.current = np.diag([-2., 1., 3.]) if matrix is None else matrix
        self.calls = 0

    def asarray(self):
        self.calls += 1
        return self.current

# -------------------------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------------------------

def test_device_resident_hessian_uses_public_accessor():
    approx_hessian = DeviceResidentHessian()
    hessian = get_hessian_array(make_opt(approx_hessian))
    assert isinstance(hessian, np.ndarray)
    np.testing.assert_array_equal(hessian, approx_hessian.current)
    assert approx_hessian.calls == 1

def test_returned_hessian_is_an_independent_copy():
    approx_hessian = DeviceResidentHessian()
    hessian = get_hessian_array(make_opt(approx_hessian))
    assert not np.shares_memory(hessian, approx_hessian.current)
    hessian[0, 0] = 99.
    assert approx_hessian.current[0, 0] == -2.

@pytest.mark.parametrize("B", [None, np.eye(3)])
def test_uninitialized_hessian_returns_none_without_asarray(B):
    # asarray() returns an identity matrix here.
    approx_hessian = DeviceResidentHessian()
    approx_hessian.initialized = False
    approx_hessian.B = B
    assert get_hessian_array(make_opt(approx_hessian)) is None
    assert approx_hessian.calls == 0

@pytest.mark.parametrize("has_flag", [False, True])
def test_legacy_hessian_without_asarray(has_flag):
    approx_hessian = SimpleNamespace(B=np.diag([-1., 2.]))
    if has_flag:
        approx_hessian.initialized = True
    hessian = get_hessian_array(make_opt(approx_hessian))
    np.testing.assert_array_equal(hessian, approx_hessian.B)
    assert not np.shares_memory(hessian, approx_hessian.B)

def test_legacy_empty_hessian_returns_none():
    assert get_hessian_array(make_opt(SimpleNamespace(B=None))) is None

def test_legacy_empty_hessian_with_asarray_returns_none():
    # Without the initialized flag, an empty B means uninitialized.
    approx_hessian = DeviceResidentHessian(matrix=np.eye(3))
    del approx_hessian.initialized
    assert get_hessian_array(make_opt(approx_hessian)) is None
    assert approx_hessian.calls == 0

def test_legacy_hessian_prefers_asarray_without_flag():
    approx_hessian = DeviceResidentHessian()
    del approx_hessian.initialized
    approx_hessian.B = np.eye(3)
    hessian = get_hessian_array(make_opt(approx_hessian))
    np.testing.assert_array_equal(hessian, approx_hessian.current)

def test_stale_cpu_copy_is_not_preferred():
    approx_hessian = DeviceResidentHessian()
    approx_hessian.B = np.eye(3)
    hessian = get_hessian_array(make_opt(approx_hessian))
    np.testing.assert_array_equal(hessian, approx_hessian.current)

def test_initialized_hessian_without_matrix_returns_none():
    approx_hessian = SimpleNamespace(initialized=True, B=None)
    assert get_hessian_array(make_opt(approx_hessian)) is None

def test_accessor_error_propagates():
    class BrokenHessian:
        initialized = True
        B = np.eye(3)

        def asarray(self):
            raise RuntimeError("cannot read current Hessian")

    with pytest.raises(RuntimeError, match="cannot read current Hessian"):
        get_hessian_array(make_opt(BrokenHessian()))

def test_sella_approximate_hessian():
    from sella.linalg import ApproximateHessian
    matrix = np.diag([-1., 2., 3.])
    approx_hessian = ApproximateHessian(dim=3, ncart=3)
    assert get_hessian_array(make_opt(approx_hessian)) is None
    approx_hessian.set_B(matrix.copy())
    hessian = get_hessian_array(make_opt(approx_hessian))
    np.testing.assert_array_equal(hessian, matrix)
    assert not np.shares_memory(hessian, approx_hessian.asarray())
    hessian[0, 0] = 99.
    np.testing.assert_array_equal(approx_hessian.asarray(), matrix)
