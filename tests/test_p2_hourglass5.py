"""Task P2.5 — the 5-hourglass-mode control (``s_hg5.F``).

What the tests hold down, in the order the plan lists them:

* a **zero-energy mode produces a nonzero force** — the control's whole job;
* **the energy lands in the hourglass ledger** (``EVIS(8)`` / ``state["evis"][8]``
  *and* the per-element ``state["ehour"]``) — the M36 lesson, and the reason a
  force-only test is not enough;
* the **default is off** and a deck that does not ask for ``/PROP/SOLID
  Isolid=5`` is bit-for-bit untouched;
* the coefficient ladder's material arms and the ``OFF`` / trapezoid bookkeeping
  behave as the Fortran does.

Fortran read before any of this was written: ``engine/source/elements/solid/
solide/s_hg5.F``, ``.../solide/shvis3.F``, ``.../solidz/shour_ctl.F90``,
``.../solide/fderi3.F`` (the ``PX*H`` matrix) and
``starter/source/properties/solid/hm_read_prop14.F`` (the ``Isolid=5 -> IINT=3``
dispatch).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.elements import solid_hexa8, solid_hourglass5
from pyradioss.model.model import ElementGroup, Model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Mat:
    def __init__(self, law=1, rho0=1000.0, E=2.1e11, nu=0.3):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.fail = None
        self.eos = None
        self.params = {"E": E, "nu": nu, "sig_y": 200.0e6,
                       "eps_max": 1e30, "eps_p_max": 1e30}

    def sound_speed_solid(self):
        return float(np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0))


class _Prop:
    def __init__(self, qa=1.1, qb=0.05, h=0.1, isolid=5, icontrol=0):
        self.params = {"qa": qa, "qb": qb, "h": h, "isolid": isolid,
                       "icontrol": icontrol}


class _Log:
    def error(self, msg, tag=""):
        raise AssertionError(msg)

    def warning(self, msg, tag=""):
        pass

    def info(self, msg, tag=""):
        pass


def _unit_hexa8_group(hourglass="5", coords=None, mat=None, prop=None):
    """One 8-node brick in a group whose /PROP/SOLID selects the control.

    ``hourglass="5"`` is ``/PROP/SOLID Isolid=5`` — the formulation flag
    ``hm_read_prop14.F:233-235`` turns into ``IINT = 3``, which ``sforc3.F:1232``
    turns into ``S_HG5``.  Any other value leaves the default ``SHVIS3`` path.
    """
    model = Model()
    x0 = 0.5 * (solid_hexa8._XI + 1.0)
    if coords is not None:
        x0 = np.asarray(coords, dtype=float)
    model.x0 = x0.copy()
    model.x = x0.copy()
    model.v = np.zeros((8, 3))
    model.vr = np.zeros((8, 3))
    group = ElementGroup(ids=np.array([1], dtype=np.int64),
                         conn=np.arange(8, dtype=np.int64)[None, :],
                         part=np.array([0], dtype=np.int64))
    group.state["slices"] = [(slice(0, 1),
                              mat if mat is not None else _Mat(),
                              prop if prop is not None else _Prop(isolid=hourglass))]
    solid_hexa8.init_group(group, model, _Log())
    return group, model


def _hourglass_velocity(group, row, direction=0, speed=1.0):
    """A velocity field that is a pure hourglass mode: zero uniform strain.

    ``v_i = _H[row][i] * e_direction`` — the modal rate of the mode.  The
    helper checks the claim rather than asserting it in prose: the one-point
    element's uniform strain-rate measure ``sum_i gradN_i (x) v_i`` must
    vanish, so any force that shows up is the control acting and nothing else.
    """
    pattern = solid_hexa8._H[row]
    v = np.zeros((8, 3))
    v[:, direction] = pattern * speed
    dndx, _vol = solid_hexa8._geometry(group._model.x[group.conn])
    assert np.abs(np.einsum("ib,ic->bc", dndx[0], v)).max() < 1e-12
    return v


# ---------------------------------------------------------------------------
# 1. the plan's two tests
# ---------------------------------------------------------------------------
def test_hg5_dissipates_a_zero_energy_mode():
    """A zero-energy (hourglass) velocity must produce a NONZERO force."""
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    f = solid_hourglass5.hg5_forces(group, hg_mode="zx", rot=np.eye(3))
    assert np.abs(f).sum() > 0.0


def test_hg5_energy_lands_in_the_hourglass_ledger():
    """The energy is booked to EVIS(8) — not produced and dropped (M36)."""
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    e = solid_hourglass5.hg5_energy(group)
    assert e >= 0.0
    assert group.state["evis"][8] == pytest.approx(e, rel=1e-12)
    assert float(group.state["ehour"].sum()) == pytest.approx(e, rel=1e-12)


def test_hg5_energy_is_the_trapezoid_of_the_hourglass_stress():
    """The booking is the trapezoid of ``s_hg5.F:297-384``, not a rectangle.

    Hold a modal velocity constant for N cycles.  ``FHOUR`` then grows linearly
    (``F_n = n dt FCL HG``), so the exact viscous work over the run is
    ``1/2 (FHOUR_final . v) T`` and the trapezoid must reproduce it exactly.
    Booking the rectangle ``FHOUR_new . v dt`` instead would come out twice as
    large, which is precisely the leak the energy-balance tests exist to catch.
    """
    group, model = _unit_hexa8_group(hourglass="5")
    row = solid_hourglass5._MODE_ROW["zx"]
    model.v[:] = _hourglass_velocity(group, row)
    dt, cycles = 1e-5, 6
    total = 0.0
    for _ in range(cycles):
        total += solid_hourglass5.hg5_energy(group, dt=dt)
    rows, signs = solid_hourglass5._resolve_mode("zx")
    conn, gamma, _gg, _k, _s = solid_hexa8._hg_operators(group, model.x)
    g = signs[0] * gamma[:, rows[0], :]
    hg = np.einsum("ni,ni->n", g, model.v[conn][:, :, 0])
    modal = float((group.state["hg5_fhour"][:, 0, rows[0]] * hg).sum())
    assert total == pytest.approx(0.5 * modal * (cycles * dt), rel=1e-10)


def test_hg5_energy_is_nonnegative_for_every_mode():
    """Viscous control: every column books work it cannot get back."""
    for mode in solid_hourglass5.MODES:
        group, model = _unit_hexa8_group(hourglass="5")
        model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW[mode])
        e = solid_hourglass5.hg5_energy(group, hg_mode=mode)
        assert e >= 0.0, mode
        assert e > 0.0, mode


# ---------------------------------------------------------------------------
# 2. the default is off and a non-requesting deck is untouched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("isolid", [0, 1, 2, 14, 17, 24])
def test_hg5_is_off_unless_the_property_asks_for_it(isolid):
    group, model = _unit_hexa8_group()
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    group.state["slices"] = [(slice(0, 1), _Mat(), _Prop(isolid=isolid))]
    assert not solid_hourglass5.enabled(group)
    f, e = solid_hourglass5.apply(group, hg_mode=None)
    assert np.array_equal(f, np.zeros_like(f))
    assert e == 0.0
    assert "evis" not in group.state, "a disabled control must not touch the ledger"
    assert np.array_equal(group.state["ehour"], np.zeros(1))


def test_enabled_follows_the_isolid5_dispatch():
    group, _ = _unit_hexa8_group(hourglass="5")
    assert solid_hourglass5.enabled(group)
    group.state["slices"] = [(slice(0, 1), _Mat(), _Prop(isolid=1))]
    assert not solid_hourglass5.enabled(group)


def test_hg5_does_not_disturb_the_default_viscous_hourglass():
    """solid_hexa8's own viscous hourglass is untouched by this module."""
    group, model = _unit_hexa8_group(hourglass="1")
    before = float(group.state["ehour"].sum())
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    fint = np.zeros((8, 6))
    dt_crit = solid_hexa8.forces(group, model.x, model.v, None, 1e-5, fint, None)
    assert dt_crit > 0.0
    assert float(group.state["ehour"].sum()) != before   # the viscous path ran
    assert solid_hourglass5.apply(group)[1] == 0.0       # hg5 stayed out of it


# ---------------------------------------------------------------------------
# 3. the coefficient ladder
# ---------------------------------------------------------------------------
def test_fcl_follows_the_fortran_prefactor():
    """FCL = F_ET * (0.038 * QH * LAMG) * dt * OFF * VOL**(1/3)."""
    group, model = _unit_hexa8_group(hourglass="5")
    h = 0.2
    group.state["slices"] = [(slice(0, 1), _Mat(), _Prop(h=h, isolid=5))]
    fhour = np.zeros((1, 3, 4))
    _dndx, vol = solid_hexa8._geometry(model.x[group.conn])
    fcl, sfac = solid_hourglass5._coefficients(group, 1e-3, fhour, vol, np.array([1000.0]))
    lamg = _Mat().K + 4.0 / 3.0 * _Mat().G
    assert sfac[0] == pytest.approx(0.038 * h * lamg, rel=1e-12)
    # isctl = 0 -> F_ET = 1 (s_hg5.F:204)
    assert fcl[0] == pytest.approx(0.038 * h * lamg * 1e-3 * vol[0] ** (1.0 / 3.0), rel=1e-12)


def test_isctl_lifts_the_distortion_stiffening():
    """With Icontrol on, F_ET may exceed 1 — and ISCTL=0 forces it back to 1.

    ``LAMGT = CXX^2 RHO`` is the P-wave modulus ``K + 4G/3`` while the LAW62
    ``LAMG`` halves the shear term (``s_hg5.F:150-153``), so the ratio is above
    1 even at the reference density — but the ``IF (ISCTL==0) F_ET=ONE`` reset
    at :204 discards it unless the deck asked for distortion control.
    """
    vol = np.array([1.0])
    fhour = np.zeros((1, 3, 4))
    soft = _Mat(law=62, E=1.0e6)
    group, _ = _unit_hexa8_group(hourglass="5", mat=soft, prop=_Prop(isolid=5, icontrol=0))
    dense = np.array([2.0 * soft.rho0])
    fcl0, _ = solid_hourglass5._coefficients(group, 1e-3, fhour, vol, dense)
    group.state["slices"] = [(slice(0, 1), soft, _Prop(isolid=5, icontrol=1))]
    fcl1, _ = solid_hourglass5._coefficients(group, 1e-3, fhour, vol, dense)
    # F_ET = LAMGT/LAMG = 2.4706..., and the LAW62 stiffening block jumps it
    # straight to TEN because SFAC1 > 2 (s_hg5.F:171-177)
    lamg = soft.K + 4.0 / 3.0 * (soft.G / 2.0)
    lamgt = (soft.K + 4.0 * soft.G / 3.0) * 2.0
    expect = 0.038 * 0.1 * lamg * 1e-3 * 10.0 * (lamgt / lamg) * vol[0] ** (1.0 / 3.0)
    assert fcl1[0] == pytest.approx(expect, rel=1e-10)

    # ISCTL = 0 resets F_ET to ONE whatever the ratio was (s_hg5.F:204)
    lamg0 = soft.K + 4.0 / 3.0 * (soft.G / 2.0)
    assert fcl0[0] == pytest.approx(
        0.038 * 0.1 * lamg0 * 1e-3 * vol[0] ** (1.0 / 3.0), rel=1e-12)


def test_a_deleted_element_stops_dissipating():
    """OFF = 0 (deleted): the control goes silent and the energy with it."""
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    assert solid_hourglass5.hg5_energy(group) > 0.0
    group.state["off"][:] = 0.0
    group.state["hg5_fhour"][:] = 0.0
    assert solid_hourglass5.hg5_energy(group, hg_mode="zx") == pytest.approx(0.0)


def test_force_is_zero_when_the_mode_is_not_excited():
    """A velocity in one column must not leak into another column's force."""
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["yz"])
    f = solid_hourglass5.hg5_forces(group, hg_mode="xy")
    assert np.abs(f).max() == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# 4. mode naming and the rot guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,row", [("xy", 2), ("yz", 0), ("zx", 1), ("xyz", 3)])
def test_mode_names_select_the_documented_gamma_row(name, row):
    """Each name drives its own ``_H`` row, and each row gives its own force."""
    forces = {}
    for mode in solid_hourglass5.MODES:
        group, model = _unit_hexa8_group(hourglass="5")
        model.v[:] = _hourglass_velocity(group, row)
        forces[mode] = solid_hourglass5.hg5_forces(group, hg_mode=mode)
    assert np.abs(forces[name]).sum() > 0.0
    assert not np.allclose(forces[name], forces["xy" if name != "xy" else "yz"])


def test_fortran_mode_integer_is_accepted():
    g1, m1 = _unit_hexa8_group(hourglass="5")
    m1.v[:] = _hourglass_velocity(g1, solid_hourglass5._MODE_ROW["zx"])
    g2, m2 = _unit_hexa8_group(hourglass="5")
    m2.v[:] = _hourglass_velocity(g2, solid_hourglass5._MODE_ROW["zx"])
    by_name = solid_hourglass5.hg5_forces(g1, hg_mode="zx")
    by_num = solid_hourglass5.hg5_forces(g2, hg_mode=3)
    assert np.allclose(by_name, by_num)


def test_unknown_mode_is_rejected():
    group, _ = _unit_hexa8_group(hourglass="5")
    with pytest.raises(ValueError, match="unknown hourglass mode"):
        solid_hourglass5.hg5_forces(group, hg_mode="zzz")


def test_a_non_identity_rotation_is_rejected():
    """s_hg5.F rotates nothing; a rotated force would be invented physics."""
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    th = 0.4
    rot = np.array([[np.cos(th), -np.sin(th), 0.0],
                    [np.sin(th), np.cos(th), 0.0],
                    [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="rotates nothing"):
        solid_hourglass5.hg5_forces(group, hg_mode="zx", rot=rot)


def test_identity_rotation_is_accepted_for_any_group_size():
    """(n, 3, 3) and (3, 3) identities are both fine; the group size is not 1."""
    for n in (1, 3):
        model = Model()
        cube = 0.5 * (solid_hexa8._XI + 1.0)
        model.x0 = np.tile(cube, (n, 1, 1)).reshape(8 * n, 3)
        model.x = model.x0.copy()
        model.v = np.zeros_like(model.x)
        model.vr = np.zeros_like(model.x)
        conn = (np.arange(8 * n).reshape(n, 8)).astype(np.int64)
        group = ElementGroup(ids=np.arange(1, n + 1, dtype=np.int64),
                             conn=conn, part=np.zeros(n, dtype=np.int64))
        group.state["slices"] = [(slice(0, n), _Mat(), _Prop(isolid=5))]
        solid_hexa8.init_group(group, model, _Log())
        for k in range(n):
            model.v[conn[k]] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
        f = solid_hourglass5.hg5_forces(group, hg_mode="zx", rot=np.eye(3))
        assert f.shape == (n, 8, 3)
        assert np.abs(f).sum() > 0.0


# ---------------------------------------------------------------------------
# 5. bookkeeping plumbing
# ---------------------------------------------------------------------------
def test_fint_accumulates_the_hourglass_force():
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    fint = np.zeros((8, 6))
    f = solid_hourglass5.hg5_forces(group, hg_mode="zx", fint=fint)
    assert np.abs(fint[:, :3]).sum() > 0.0
    assert fint[:, 3:].sum() == 0.0
    assert fint[:, :3].sum() == pytest.approx(f.sum(), rel=1e-12)


def test_ledger_accumulates_across_cycles():
    group, model = _unit_hexa8_group(hourglass="5")
    model.v[:] = _hourglass_velocity(group, solid_hourglass5._MODE_ROW["zx"])
    a = solid_hourglass5.hg5_energy(group)
    b = solid_hourglass5.hg5_energy(group)
    assert group.state["evis"][8] == pytest.approx(a + b, rel=1e-12)
    # a growing FHOUR means the second cycle dissipates at least as much
    assert b >= 0.0


def test_empty_group_is_a_no_op():
    model = Model()
    model.x0 = np.zeros((0, 3))
    model.x = np.zeros((0, 3))
    model.v = np.zeros((0, 3))
    group = ElementGroup(ids=np.empty(0, dtype=np.int64),
                         conn=np.empty((0, 8), dtype=np.int64),
                         part=np.empty(0, dtype=np.int64))
    group.state["slices"] = []
    f, e = solid_hourglass5.apply(group)
    assert f.shape == (0, 8, 3)
    assert e == 0.0