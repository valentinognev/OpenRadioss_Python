"""BT Z2 ``ISMSTR == 3`` zeroing (CPXPY3) in the shipped kind-4 block.

**Synthetic decks.** No official deck resolves ``ISMSTR == 3`` on a kind-4
(``Ishell = 4``) shell — upstream's starter guard (``hm_read_prop01.F:234``)
resets ``Ismstr = 3`` to 2 for every card but 0/2 (measured on the oracle
starter: ``(4,3) -> 2`` + warning 319, ``(0,3) -> 2`` + 319, ``(2,3) -> 3``,
``(4,0) -> 2``) — so every ``ISMSTR == 3`` case here is a hand-built
one-element SYNTHETIC deck (free-format, after the P3.2 ``_one_element``
recipe), plus state-level injection where the starter guard leaves no
deck-level route. They exist to pin the one observable piece of the
``ISMSTR == 3`` stress-based path this task models:

THE UPSTREAM CONTRACT (read from the Fortran, measured on the oracle)
---------------------------------------------------------------------
``cforc3.F:430`` takes ``CPXPY3`` exactly when ``ISMSTR == 3`` (else
``CDERI3``), and ``CPXPY3`` sets ``Z2 = ZERO`` (``cpxpy3.F:77`` and l.88).
``Z2`` — the corotational out-of-plane coordinate of node 2 relative to
node 1, ``(x2-x1).e3`` (``cderi3.F:109``) — feeds the kind-4 warp half
``EXZZ2 = GZX*ZZ2`` / ``EYZZ2 = GZY*ZZ2`` (``cdefo3.F:183``, l.193), so on
an ``ISMSTR == 3`` element the warp half is identically zero while the
``EXZ2``/``EYZ2`` velocity half survives. The port always built ``Z2``
from corotational local geometry and never modelled that path.

``/PROP/SHELL`` resolution (``hm_read_prop01.F:203``, l.233-241, plus the
card-0 default ``defaults_mod.F90:167``): card ``Ismstr`` 0 takes the
``/DEF_SHELL`` default (``ISST_D``); ``ISST_D == -2`` forces ``-1``
(automatic — ``cgrtails.F`` picks per material, unported); leftover 0
becomes 2; ``ISMSTR == 3`` with (effective) card ``Ishell`` outside
``{0, 2}`` becomes 2 with warning 319 (card 0 means "default", whose
``ISHELL`` fallback is 1 — hence ``(0,3) -> 2`` too, as the oracle
confirms).

A CORRECTION TO THE P3.2 RECORD, kept here so the next reader does not
re-inherit it: the c45 parity case does NOT run ``ISMSTR == 3``. The
P3.2 note ("no ``/BRAT`` runs that path") is wrong on both halves — no
``/BRAT`` card exists anywhere upstream (starter, engine, or CFG tree),
and the oracle starter prints ``SMALL STRAIN FLAG = 2`` for the c45
deck. The port's corotational ``Z2`` therefore MATCHES upstream on c45,
and the fix is bitwise inert there (proven by ``cmp`` on before/after
output, not by quoting rel-RMS).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.accel import _state, select_backend
from pyradioss.elements import shell_bt4
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)


@pytest.fixture(autouse=True)
def pin_numpy_backend(monkeypatch):
    """Pin the NumPy backend: the no-op test below asserts bitwise
    identity, which only the reference kernels guarantee (same pattern
    as the P3.7 synthetic-deck file)."""
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    orig = dict(_state)
    select_backend("numpy")
    yield
    _state.update(orig)


# the RD-E-1000 BT recipe (same shape as the P3.2 c45 probe)
A_, B_, T_ = 16.0, 15.2, 20.0
DT = 1e-4
#: rigid roll rate about the element's in-plane y axis — gives non-zero
#: out-of-plane diagonal differences VZ13/VZ24, without which every
#: cdefo3 branch does nothing at all.
W = 0.05


def _deck_text(ishell, ismstr, warp=0.02):
    """Free-format one-element SYNTHETIC BT deck (cf. P3.2 ``_one_element``)."""
    z = (0.0, warp, -0.3 * warp, 0.0)
    return (
        "/BEGIN\nsynthetic z2 ismstr\n/NODE\n"
        f"1 {-A_/2} {-B_/2} {z[0]}\n2 {A_/2} {-B_/2} {z[1]}\n"
        f"3 {A_/2} {B_/2} {z[2]}\n4 {-A_/2} {B_/2} {z[3]}\n"
        "/SHELL/1\n1 1 2 3 4\n/PART/1\np\n1 1\n"
        "/MAT/LAW1/1\nm\n0.01\n1000. 0.\n"
        f"/PROP/SHELL/1\nsh\n{ishell} {ismstr} 0 0\n0 0 0 0 0\n3 0 {T_}\n/END\n")


def _one_element(tmp_path, ishell=4, ismstr=0, warp=0.02, name="SYN"):
    """Parse + init the SYNTHETIC deck; returns (model, group, log)."""
    tmp_path.mkdir(exist_ok=True)
    f = tmp_path / f"{name}_{ishell}_{ismstr}_0000.rad"
    f.write_text(_deck_text(ishell, ismstr, warp))
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model, model.shells, log


def _roty(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rolled(m, theta, w, dt):
    """Stencil-exact rigid roll about the in-plane y axis (P3.2 recipe)."""
    x0 = m.x0
    m.x = x0 @ _roty(theta).T
    m.v = (m.x - x0 @ _roty(theta - w * dt).T) / dt
    m.vr = np.zeros_like(m.v)
    m.vr[:, 1] = w


def _run_forces(m, g, theta=0.7):
    _rolled(m, theta, W, DT)
    fint = np.zeros((len(m.x), 3))
    mint = np.zeros((len(m.x), 3))
    shell_bt4.forces(g, m.x, m.v, m.vr, DT, fint, mint)
    return fint, mint


def _fresh_run(tmp_path, tag, warp, ismstr=None, ihbe_off=False):
    """One forces() evaluation on FRESH state — forces() accumulates
    persistent hourglass/stress state, so consecutive runs on one group
    are not comparable."""
    m, g, _log = _one_element(tmp_path / tag, ishell=4, ismstr=0,
                              warp=warp, name=tag)
    if ismstr is not None:
        g.state["ismstr"][:] = ismstr
    if ihbe_off:
        g.state["ihbe_mask"][:] = 99          # no cdefo3 branch at all
    return _run_forces(m, g)


# ---------------------------------------------------------------------------
# 1. ISMSTR resolution (hm_read_prop01.F:203, l.233-241; defaults_mod.F90:167)
# ---------------------------------------------------------------------------

def test_ismstr_defaults_to_2(tmp_path):
    """Card ``Ismstr = 0`` takes the ``/DEF_SHELL`` default (0 here) and
    lands on 2 — the c45 shape (oracle: ``SMALL STRAIN FLAG = 2``)."""
    _m, g, _log = _one_element(tmp_path, ishell=4, ismstr=0)
    assert int(g.state["ismstr"][0]) == 2


def test_ismstr3_with_ishell4_resolves_to_2_with_warning(tmp_path):
    """``hm_read_prop01.F:234``: ``ISMSTR == 3`` with card ``Ishell = 4``
    is reset to 2 with warning 319 (oracle C43 run: ``Ismstr = 2``)."""
    _m, g, log = _one_element(tmp_path, ishell=4, ismstr=3)
    assert int(g.state["ismstr"][0]) == 2
    assert any("319" in w or "ISMSTR" in w for w in log.warnings), log.warnings


def test_ismstr3_with_ishell2_survives(tmp_path):
    """Card ``Ishell = 2`` is inside the ``{0, 2}`` guard, so ``Ismstr = 3``
    survives to the engine (oracle C23 run: ``SMALL STRAIN FLAG = 3``,
    no warning 319)."""
    _m, g, log = _one_element(tmp_path, ishell=2, ismstr=3)
    assert int(g.state["ismstr"][0]) == 3
    assert not any("319" in w for w in log.warnings), log.warnings


def test_c45_deck_resolves_ismstr2():
    """The real c45 parity deck (fixed-format ``/PROP/SHELL/202``,
    ``Ishell = 4 Ismstr = 0``) resolves ``ISMSTR = 2`` — the fixed-format
    read path plus the oracle's ``SMALL STRAIN FLAG = 2``."""
    text = Path("tests/data/rd_decks/rd_e/RD-E-1000_Bending/10_Bending/BT/"
                "BT_type4/Sf_0.9/ROLLING_0000.rad")
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(text)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    assert model.shells is not None
    assert set(np.unique(model.shells.state["ismstr"])) == {2}


# ---------------------------------------------------------------------------
# 2. the discriminating SYNTHETIC deck: ISMSTR != 3, non-zero Z2 —
#    the warp correction is live and upstream's zeroing is not
# ---------------------------------------------------------------------------

def test_warp_correction_live_when_ismstr_not_3(tmp_path):
    """SYNTHETIC Ishell=4 warped deck at resolved ``ISMSTR = 2``: ``Z2``
    is non-zero and switching the kind-4 path off moves the resultant —
    the correction is live, exactly as upstream's ``CDERI3`` path is."""
    m, g, _log = _one_element(tmp_path, ishell=4, ismstr=0, warp=0.02)
    assert int(g.state["ismstr"][0]) == 2
    _e, xl, _a, _b1, _b2 = shell_bt4._local_geometry(m.x[g.conn])
    assert abs(float(xl[0, 1, 2] - xl[0, 0, 2])) > 1e-6  # Z2 != 0
    on, _ = _fresh_run(tmp_path, "live", warp=0.02)
    off, _ = _fresh_run(tmp_path, "off", warp=0.02, ihbe_off=True)
    assert not np.allclose(on, off)


# ---------------------------------------------------------------------------
# 3. ISMSTR == 3 zeroes the warp half (cforc3.F:430 / cpxpy3.F:77, :88)
# ---------------------------------------------------------------------------

def test_ismstr3_zeroes_the_warp_half(tmp_path):
    """Same SYNTHETIC warped deck with the resolved flag at 3 (injected at
    state level: the starter guard leaves no TYPE1 deck-level route to
    kind-4 with ``ISMSTR == 3`` — oracle C43) — the resultant moves,
    because the ``EXZZ2``/``EYZZ2`` warp half is gone while the
    ``EXZ2``/``EYZ2`` velocity half survives."""
    live, _ = _fresh_run(tmp_path, "live", warp=0.02)
    zeroed, _ = _fresh_run(tmp_path, "zeroed", warp=0.02, ismstr=3)
    assert not np.allclose(live, zeroed)


def test_ismstr3_is_a_noop_when_z2_is_zero(tmp_path):
    """Axis-aligned flat SYNTHETIC deck (``Z2`` bitwise zero) with a steady
    out-of-plane shear rate keeping the velocity half live: the flag
    changes nothing, bitwise — the zeroing touches only the ``Z2``-fed
    warp half, never the surrounding arithmetic."""
    runs = {}
    for tag, ism in (("flat2", None), ("flat3", 3)):
        m, g, _log = _one_element(tmp_path / tag, ishell=4, ismstr=0,
                                  warp=0.0, name=tag)
        if ism is not None:
            g.state["ismstr"][:] = ism
        m.v[:] = 0.0
        m.v[:, 2] = m.x0[:, 0] * 0.05  # vz ~ x: VZ13/VZ24 != 0, no rotation
        m.vr[:] = 0.0
        fint = np.zeros((len(m.x), 3))
        mint = np.zeros((len(m.x), 3))
        shell_bt4.forces(g, m.x, m.v, m.vr, DT, fint, mint)
        runs[tag] = (fint, mint)
    (live, live_m), (zeroed, zeroed_m) = runs["flat2"], runs["flat3"]
    assert np.abs(live).max() > 0.0          # the velocity half is live ...
    assert np.array_equal(live, zeroed)      # ... and the flag is a no-op
    assert np.array_equal(live_m, zeroed_m)


def test_z2_zero_kills_exactly_the_warp_half():
    """``EXZZ2 = GZX*ZZ2`` is the only ``Z2``-carrying term of the kind-4
    branch: at ``Z2 = 0`` the EVEN part (warp half, under the 1<->3/2<->4
    diagonal swap) vanishes and the ODD part (``EXZ2`` velocity half) is
    untouched."""
    px = np.array([[2.0, -1.0]])
    py = np.array([[3.0, 4.0]])
    vx = np.array([[0.3, 0.2]])
    vy = np.array([[1.0, 0.0]])
    dx_live, dy_live = shell_bt4._rot2_correction(
        px, py, vx, vy, DT, kind=4, z2=np.array([0.02]))
    dx_zero, dy_zero = shell_bt4._rot2_correction(
        px, py, vx, vy, DT, kind=4, z2=np.array([0.0]))
    swap = [2, 3, 0, 1]
    for live, zero in ((dx_live, dx_zero), (dy_live, dy_zero)):
        odd_live = 0.5 * (live[0] - live[0][swap])
        odd_zero = 0.5 * (zero[0] - zero[0][swap])
        even_zero = 0.5 * (zero[0] + zero[0][swap])
        assert np.abs(odd_live).max() > 0.0
        assert np.allclose(odd_live, odd_zero, rtol=1e-12)
        assert np.allclose(even_zero, 0.0, atol=1e-300)
