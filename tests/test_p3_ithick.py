"""P3.6 — ``Ithick=1`` thickness update in the QBAT/QEPH families.

**Synthetic one-element cases.** Every existing corpus deck runs
``Ithick=0``, so the evolving-thickness path is covered by hand-built
single-element groups (a unit-square LAW1 shell under uniform x-stretch).

Upstream (all read off the Fortran, not recalled):

* ``cncoef3.F`` (``CNCOEF3`` / ``CNCOEF3B``): ``ITHK>0`` selects the
  evolving ``GBUF%THK`` as the cycle thickness ``THK0``; otherwise ``THK0``
  is reset to the deck constant ``THKE`` every cycle.
* ``sigeps01c.F:143-144``: per-layer thinning ``EZZ =
  -NU*(DEPSXX+DEPSYY)/(ONE-NU)``, ``THK += EZZ*THKLYL*OFF``.
* ``cbaforc3.F:863-867`` (QBAT only, ``NPINCH==0``): the relaxation
  ``GBUF%THK -= THREE_OVER_4*(GBUF%THK-THK0)`` with ``THK0`` following.
  Neither ``coque/cforc3.F`` nor ``coquez/czforc3.F`` has this block.

The port mapping is ``state["thick"]`` == ``THK0`` (assembly thickness)
and ``state["thk0"]`` == ``THKE`` (deck anchor).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.accel import _state, select_backend
from pyradioss.elements import shell_qbat, shell_qeph
from pyradioss.elements.shell_ithick import _ithick_relax


@pytest.fixture(autouse=True)
def pin_numpy_backend(monkeypatch):
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    orig = dict(_state)
    select_backend("numpy")
    yield
    _state.update(orig)


class _Mat:
    """Minimal LAW1 shell material."""

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
    def __init__(self, thick=0.01, nip=3, dn=1e-3, ithick=0):
        self.params = {"thick": thick, "nip": nip, "dn": dn,
                       "ithick": ithick}
        self.thick = thick
        self.nip = nip
        self.dn = dn
        self.ithick = ithick
        self.ipinch = 0


def _shell_group(group_name, ithick, thick=0.01, nip=3):
    """One-element unit-square group for the named kernel."""
    mod = {"shells_qbat": shell_qbat, "shells_qeph": shell_qeph}[group_name]
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mat = _Mat()
    prop = _Prop(thick=thick, nip=nip, ithick=ithick)
    group = SimpleNamespace(
        id=1,
        n=1,
        conn=np.arange(4, dtype=np.int64).reshape(1, 4),
        ids=np.array([1], dtype=np.int64),
        state={"slices": [(slice(0, 1), mat, prop)],
               "off": np.ones(1), "chk_fail": False},
    )
    model = SimpleNamespace(x0=coords.copy(),
                            nodes_idx=np.arange(4, dtype=np.int64))
    mod.init_group(group, model, None)
    return mod, group, coords


def _stretch_x(a=10.0):
    """Uniform x-stretch velocity: V13 = (-a,0,0), V24 = (a,0,0)."""
    return np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0],
                     [a, 0.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.mark.parametrize("group_name", ["shells_qbat", "shells_qeph"])
def test_ithick_one_updates_thickness(group_name):
    mod, group, coords = _shell_group(group_name, ithick=1)
    t0 = group.state["thick"].copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    mod.forces(group, coords, _stretch_x(), np.zeros((4, 3)), 1e-3,
               fint, mint)
    assert not np.array_equal(group.state["thick"], t0)


@pytest.mark.parametrize("group_name", ["shells_qbat", "shells_qeph"])
def test_ithick_zero_leaves_thickness_alone(group_name):
    mod, group, coords = _shell_group(group_name, ithick=0)
    t0 = group.state["thick"].copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    mod.forces(group, coords, _stretch_x(), np.zeros((4, 3)), 1e-3,
               fint, mint)
    assert np.array_equal(group.state["thick"], t0)


@pytest.mark.parametrize("group_name", ["shells_qbat", "shells_qeph"])
def test_ithick_one_thins_under_tension(group_name):
    """Biaxial tension thins the shell (sigeps01c plane-stress sign)."""
    mod, group, coords = _shell_group(group_name, ithick=1)
    t0 = group.state["thick"].copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    mod.forces(group, coords, _stretch_x(), np.zeros((4, 3)), 1e-3,
               fint, mint)
    assert np.all(group.state["thick"] < t0)


def test_ithick_relaxation_keeps_a_quarter_of_the_increment():
    """cbaforc3.F:865 pin: THK - 3/4*(THK-THK0), un-reassociated."""
    thk_star, thk0 = 1.1, 1.0
    got = _ithick_relax(np.array([thk_star]), np.array([thk0]))
    assert got[0] == pytest.approx(thk_star - 0.75 * (thk_star - thk0))
    assert got[0] == pytest.approx(1.025)


@pytest.mark.parametrize("group_name", ["shells_qbat", "shells_qeph"])
def test_ithick_one_skips_dead_elements(group_name):
    """sigeps01c.F:144 gates thinning on OFF: dead elements keep THK."""
    mod, group, coords = _shell_group(group_name, ithick=1)
    group.state["off"][:] = 0.0
    t0 = group.state["thick"].copy()
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    mod.forces(group, coords, _stretch_x(), np.zeros((4, 3)), 1e-3,
               fint, mint)
    assert np.array_equal(group.state["thick"], t0)
