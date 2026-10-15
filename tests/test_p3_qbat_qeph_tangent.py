"""Task P3.10 — QBAT/QEPH tangent + geometric stiffness: finite-difference proof.

The plan's hard constraint: the tangent is proven by a finite difference of
the (nonlinear) residual, not by a symmetry check.  A tangent that is merely
symmetric-positive fails these tests.

Residual under test: each module's ``static_internal_forces`` — the
updated-Lagrangian end-configuration assembly the M9 nonlinear-geometry
implicit path (``implicit/statics.py``) evaluates.  The FD procedure is the
M9/M11 pattern: fresh group -> midpoint ``forces()`` advance with the trial
displacement as pseudo-velocity (dt = 1) -> ``static_internal_forces`` on the
end geometry.  Its derivative at zero displacement is the material tangent;
without the advance (prestressed state held fixed) it is the geometric
stiffness.

Scope notes (all dusty corners documented, none silent):
* Drilling spins (DOFs 5, 11, 17, 23) and their rows are excluded: the
  explicit kernels run Idrill = 0 (no spin response) while the tangents
  carry the implicit-only 1e-3 spin regularization (upstream HZ/KZ_TOL,
  cmatc3.F).  Upstream's own implicit K does not linearize its explicit
  residual there either.
* In-plane kgeo columns are not FD-asserted: the corotational frame spins
  under in-plane perturbation and the frame-spin x prestress-force terms are
  in neither the port's nor upstream CZLKECG3's von-Karman operator.  The
  transverse columns (the buckling-relevant ones) are FD-asserted.
"""

import numpy as np
import pytest

from pyradioss.accel import _state, select_backend
from pyradioss.elements import shell_qbat, shell_qeph


@pytest.fixture(autouse=True)
def pin_numpy_backend(monkeypatch):
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    orig = dict(_state)
    select_backend("numpy")
    yield
    _state.update(orig)


class SimpleMat:
    def __init__(self, E=2.1e11, nu=0.3, rho0=7850.0, law=1):
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.law = law
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.G = E / (2.0 * (1.0 + nu))
        self.fail = None
        self.params = {"E": E, "nu": nu, "rho": rho0, "eps_p_max": 1e30}

    def sound_speed_shell(self):
        return np.sqrt(self.E / (self.rho0 * (1.0 - self.nu ** 2))) \
            if (self.rho0 > 0.0 and self.law != 0) else 0.0


class SimpleProp:
    def __init__(self, thick=0.01, nip=3):
        self.params = {"thick": thick, "nip": nip}
        self.thick = thick
        self.nip = nip


#: mildly distorted flat quad: PH1/PH2 != 0 so the hourglass/transverse
#: operators are proven off the parallelogram, not just on it.
COORDS = np.array([[0.0, 0.0, 0.0], [1.1, 0.05, 0.0],
                   [0.95, 1.02, 0.0], [-0.03, 0.9, 0.0]])

#: every DOF except the drilling spins (see module docstring).
PROVEN_DOFS = [j for j in range(24) if j % 6 != 5]
#: every row except the drilling-moment rows (implicit-only regularization).
PROVEN_ROWS = [j for j in range(24) if j % 6 != 5]
#: transverse DOFs, the buckling-relevant kgeo columns.
TRANSVERSE_DOFS = [2, 8, 14, 20]


def _qbat_group(coords):
    from types import SimpleNamespace
    from pyradioss.common.messages import MessageLog  # noqa: F401
    n = len(coords) // 4
    conn = np.arange(len(coords), dtype=np.int64).reshape(n, 4)
    group = SimpleNamespace(
        id=1, n=n, conn=conn, ids=np.arange(1, n + 1, dtype=np.int64),
        state={"slices": [(slice(0, n), SimpleMat(), SimpleProp())],
               "off": np.ones(n, dtype=float), "chk_fail": False},
    )
    model = SimpleNamespace(x0=coords.copy(),
                            nodes_idx=np.arange(len(coords), dtype=np.int64))
    log = SimpleNamespace(error=lambda msg, ctx="": None,
                          warning=lambda msg, ctx="": None)
    shell_qbat.init_group(group, model, log)
    return group


def _qeph_group(coords):
    from pyradioss.common.messages import MessageLog
    from pyradioss.model.model import ElementGroup, Model
    n = len(coords) // 4
    conn = np.arange(len(coords), dtype=np.int64).reshape(n, 4)
    model = Model()
    model.x0 = coords.copy()
    group = ElementGroup(ids=np.arange(1, n + 1, dtype=np.int64), conn=conn,
                         part=np.zeros(n, dtype=np.int64),
                         state={"slices": [(slice(0, n), SimpleMat(),
                                           SimpleProp())],
                                "part_ids": np.ones(n, dtype=np.int64)})
    shell_qeph.init_group(group, model, MessageLog())
    return group


_MAKERS = {"qbat": (shell_qbat, _qbat_group),
           "qeph": (shell_qeph, _qeph_group)}


def _gather(group, fint, mint):
    r = np.zeros(24)
    for a in range(4):
        r[6 * a:6 * a + 3] = -fint[group.conn[0, a]]
        r[6 * a + 3:6 * a + 6] = -mint[group.conn[0, a]]
    return r


def _nonlinear_residual(mod, maker, u_vec):
    """R(U): midpoint-advance the state to U, assemble static on X + U."""
    group = maker(COORDS)
    nn = COORDS.shape[0]
    u4 = np.zeros((nn, 3))
    ur4 = np.zeros((nn, 3))
    for a in range(4):
        u4[group.conn[0, a]] = u_vec[6 * a:6 * a + 3]
        ur4[group.conn[0, a]] = u_vec[6 * a + 3:6 * a + 6]
    x_mid = COORDS + 0.5 * u4
    x_end = COORDS + u4
    mod.forces(group, x_mid, u4, ur4, 1.0,
               np.zeros((nn, 3)), np.zeros((nn, 3)))
    fint = np.zeros((nn, 3))
    mint = np.zeros((nn, 3))
    mod.static_internal_forces(group, x_end, None, None, fint, mint)
    return _gather(group, fint, mint)


def _static_residual(mod, maker, x):
    """R(x): static assembly at geometry x from the as-built state."""
    group = maker(COORDS)
    nn = COORDS.shape[0]
    fint = np.zeros((nn, 3))
    mint = np.zeros((nn, 3))
    mod.static_internal_forces(group, x, None, None, fint, mint)
    return _gather(group, fint, mint)


@pytest.mark.parametrize("name", ["qbat", "qeph"])
def test_tangent_matches_a_finite_difference_of_the_residual(name):
    """P3.10 gate: central FD of the nonlinear residual == tangent column."""
    mod, maker = _MAKERS[name]
    group = maker(COORDS)
    K = mod.tangent(group, COORDS)[0][0]
    for dof in PROVEN_DOFS:
        h = 1e-6 if dof % 6 < 3 else 1e-5
        up = np.zeros(24)
        um = np.zeros(24)
        up[dof] = h
        um[dof] = -h
        fd = (_nonlinear_residual(mod, maker, up)
              - _nonlinear_residual(mod, maker, um)) / (2.0 * h)
        np.testing.assert_allclose(fd[PROVEN_ROWS], K[PROVEN_ROWS, dof],
                                   rtol=2e-5, atol=1e-6,
                                   err_msg=f"{name} dof {dof}")


@pytest.mark.parametrize("name", ["qbat", "qeph"])
def test_kgeo_transverse_matches_a_finite_difference_under_prestress(name):
    """Prestressed static assembly differentiated wrt geometry == kgeo."""
    mod, maker = _MAKERS[name]
    pre = np.array([5.0e7, 3.0e7, 1.0e6])
    probe = maker(COORDS)
    probe.state["sig"][:] = pre
    Kg = mod.kgeo(probe, COORDS)[0][0]

    h = 1e-6
    for dof in TRANSVERSE_DOFS:
        node, comp = dof // 6, dof % 6
        xp = COORDS.copy()
        xm = COORDS.copy()
        xp[node, comp] += h
        xm[node, comp] -= h
        fd = (_prestressed_static(mod, maker, pre, xp)
              - _prestressed_static(mod, maker, pre, xm)) / (2.0 * h)
        np.testing.assert_allclose(fd[PROVEN_ROWS], Kg[PROVEN_ROWS, dof],
                                   rtol=2e-5, atol=1e-6,
                                   err_msg=f"{name} kgeo dof {dof}")


def _prestressed_static(mod, maker, pre, x):
    g = maker(COORDS)
    g.state["sig"][:] = pre
    nn = COORDS.shape[0]
    fint = np.zeros((nn, 3))
    mint = np.zeros((nn, 3))
    mod.static_internal_forces(g, x, None, None, fint, mint)
    return _gather(g, fint, mint)


@pytest.mark.parametrize("name", ["qbat", "qeph"])
def test_static_internal_forces_preserves_state_and_balances(name):
    """static_internal_forces is a pure re-assembly: no state drift."""
    mod, maker = _MAKERS[name]
    group = maker(COORDS)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in group.state.items() if isinstance(v, np.ndarray)
              and v.dtype != bool}
    nn = COORDS.shape[0]
    fint = np.zeros((nn, 3))
    mint = np.zeros((nn, 3))
    mod.static_internal_forces(group, COORDS, None, None, fint, mint)
    for k, v in before.items():
        assert np.array_equal(group.state[k], v), f"{name}: state[{k}] drifted"
    assert np.allclose(fint, 0.0) and np.allclose(mint, 0.0)


@pytest.mark.parametrize("name", ["qbat", "qeph"])
def test_tangent_kgeo_satisfy_the_assembly_contract(name):
    """(ke, edofs) shapes and equation addressing for implicit/assembly.py."""
    mod, maker = _MAKERS[name]
    group = maker(COORDS)
    for fun in (mod.tangent, mod.kgeo):
        ke, edofs = fun(group, COORDS)
        assert ke.shape == (1, 24, 24)
        assert edofs.shape == (1, 24)
        assert np.allclose(ke[0], ke[0].T, atol=1e-6)
        expect = np.array([group.conn[0, a] * 6 + c
                           for a in range(4) for c in range(6)])
        assert np.array_equal(edofs[0], expect)
    # six rigid-body null modes of the material tangent
    K = mod.tangent(group, COORDS)[0][0]
    for d in range(3):
        u = np.zeros(24)
        u[d::6] = 1.0
        assert np.linalg.norm(K @ u) < 1e-4
