"""P3.7 — QBAT ``nip=1`` single-point branch (CBAFORI1 / CBAVISNP1).

**Synthetic decks.** No official deck runs QBAT at N=1 (the port's own
note in ``PORTING_GUIDE.md`` and the upstream corpus agree), so every case
here is a hand-built one-element SYNTHETIC deck. They exist to pin the
single-point branch's observable physics:

* the branch gate itself (a pure out-of-plane velocity mode produces the
  CBAVISNP1 node-3/4 z-force and its ehour bookkeeping at ``nip=1``,
  while at ``nip=2`` the same mode integrates into transverse-shear
  forces with no ehour work),
* the hand-computed CBAVISNP1 damping magnitude on the unit square,
* the ``NIP`` group-state interface and the removal of the Starter
  warning the port used to emit for ``nip=1`` decks.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.accel import _state, select_backend
from pyradioss.elements import shell_qbat


@pytest.fixture(autouse=True)
def pin_numpy_backend(monkeypatch):
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    orig = dict(_state)
    select_backend("numpy")
    yield
    _state.update(orig)


class _Mat:
    """Minimal LAW1 shell material (same contract as test_m513's)."""

    def __init__(self, E=2.1e11, nu=0.3, rho0=7850.0, law=1):
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.law = law
        self.G = E / (2.0 * (1.0 + nu))
        self.fail = None
        self.params = {"E": E, "nu": nu, "rho": rho0, "eps_p_max": 1e-30}

    def sound_speed_shell(self):
        if self.rho0 > 0.0 and self.law != 0:
            return np.sqrt(self.E / (self.rho0 * (1.0 - self.nu ** 2)))
        return 0.0


class _Prop:
    def __init__(self, thick=0.01, nip=1, dn=1e-3, ishell=12):
        self.params = {"thick": thick, "nip": nip, "dn": dn, "ishell": ishell}
        self.id = 12
        self.thick = thick
        self.nip = nip
        self.dn = dn
        self.ipinch = 0


class _Log:
    """Collecting log: the Starter warnings land in ``.warnings``."""

    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, ctx=""):
        self.warnings.append(msg)

    def error(self, msg, ctx=""):
        self.errors.append(msg)
        raise AssertionError(msg)


class _QbatCase:
    """A one-element QBAT group with a ``.forces(dt)`` convenience and the
    collecting log, standing in for the Starter-built group."""

    def __init__(self, nip, coords=None, thick=0.01, dn=1e-3, E=2.1e11,
                 nu=0.3, rho=7850.0):
        if coords is None:
            coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        self.coords = np.asarray(coords, dtype=float)
        self.log = _Log()
        mat = _Mat(E=E, nu=nu, rho0=rho)
        prop = _Prop(thick=thick, nip=nip, dn=dn)
        self.group = SimpleNamespace(
            id=1,
            n=1,
            conn=np.arange(4, dtype=np.int64).reshape(1, 4),
            ids=np.array([1], dtype=np.int64),
            state={"slices": [(slice(0, 1), mat, prop)],
                   "off": np.ones(1), "chk_fail": False},
        )
        self.model = SimpleNamespace(
            x0=self.coords.copy(),
            nodes_idx=np.arange(4, dtype=np.int64),
        )
        shell_qbat.init_group(self.group, self.model, self.log)
        self.prop = prop

    def forces(self, dt, v=None, vr=None):
        v = np.zeros((4, 3)) if v is None else np.asarray(v, dtype=float)
        vr = np.zeros((4, 3)) if vr is None else np.asarray(vr, dtype=float)
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))
        dt_e = shell_qbat.forces(self.group, self.coords, v, vr, dt,
                                 fint, mint)
        return dt_e, fint, mint


def test_qbat_nip1_reduces_to_the_single_point_path():
    g = _QbatCase(nip=1)
    out = g.forces(dt=1e-3)[0]
    assert np.all(np.isfinite(out)) and np.all(out > 0)
    assert not any("nip" in w.lower() for w in g.log.warnings)


def test_qbat_nip1_state_records_nip_and_warns_no_more():
    g1 = _QbatCase(nip=1)
    assert g1.group.state["NIP"] == 1
    assert g1.log.warnings == []
    g3 = _QbatCase(nip=3)
    assert g3.group.state["NIP"] == 3


def test_qbat_nip1_rotation_damping_force_and_energy():
    """A pure out-of-plane hourglass velocity mode (V13 = V24 = 0, VHI z != 0,
    zero nodal rotations) drives ONLY the CBAVISNP1 damping at nip=1: opposite
    z-forces at nodes 3/4 and the TESY work in ehour. At nip=2 the same mode
    carries transverse-shear rate, so the fully integrated path answers with
    shear forces (no ehour work — CBAVISC damps membrane/bending only)."""
    v = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])       # VHI z = 1.0
    g1 = _QbatCase(nip=1)
    _, fint1, _ = g1.forces(dt=1e-3, v=v)
    assert g1.group.state["ehour"][0] > 0.0
    assert fint1[2, 0] == 0.0 and fint1[2, 1] == 0.0
    assert fint1[2, 2] == pytest.approx(-fint1[3, 2])
    assert abs(fint1[2, 2]) > 0.0

    g2 = _QbatCase(nip=2)
    _, fint2, _ = g2.forces(dt=1e-3, v=v)
    assert g2.group.state["ehour"][0] == 0.0
    assert not np.all(fint2 == 0.0)


def test_qbat_nip1_damping_matches_the_hand_computed_cbavisnp1():
    """Hand-computed CBAVISNP1 on the unit square. With vr = 0 every
    rotation combination vanishes, so VHXZ = VHYZ = VHI z; the centered
    moment arms of the unit square are mx23 = my34 = 0.5 and
    mx13 = my13 = mx34 = my23 = 0."""
    E, nu, rho, thick = 2.1e11, 0.3, 7850.0, 0.01
    dn = 1e-3
    vhiz = 1.0
    dt = 1e-3
    area = 1.0

    g = E / (2.0 * (1.0 + nu))
    c2 = g * rho * area / 12.0
    # HVL = TWO1000*AMU*sqrt(C2)*OFF (cbavisc.F): TWO1000 = 2000
    # (common_source/modules/constant_mod.F:629), AMU = DN (cncoef3.F).
    hvl = 2000.0 * dn * math.sqrt(c2)
    cxz = (0.5 ** 2) * hvl          # MY34 = 0.5, MX34 = 0
    cyz = (0.5 ** 2) * hvl          # MX23 = 0.5, MY23 = 0
    hsura = thick / area
    sc5 = cxz * vhiz * hsura
    sc6 = cyz * vhiz * hsura
    ss3 = sc5 + sc6
    tesy = (sc5 * vhiz + sc6 * vhiz) * dt

    v = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    case = _QbatCase(nip=1, dn=dn, E=E, nu=nu, rho=rho, thick=thick)
    _, fint, _ = case.forces(dt=dt, v=v)

    assert case.group.state["ehour"][0] == pytest.approx(tesy, rel=1e-12)
    assert fint[2, 2] == pytest.approx(-ss3, rel=1e-12)
    assert fint[3, 2] == pytest.approx(ss3, rel=1e-12)
