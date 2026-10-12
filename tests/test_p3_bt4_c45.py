"""P3.2 — ``cdefo3.F`` BT type 4 (parity case c45) Z2 warp correction.

THE UPSTREAM CONTRACT (read from the Fortran, not extrapolated)
--------------------------------------------------------------
``engine/source/elements/shell/coque/cdefo3.F`` l.174-213, the third and
last formulation branch::

    ELSEIF(IHBE==4)THEN
    C    old
    c wrapping z1 = z3 = 0 z4 = z2 => velocity modification for the lurglass membrane
          DT1V4 = HALF*DT1C(I)
          IF(IMPL_S>0) DT1V4=ZERO
          ZZ2(I)=HALF*Z2(I)
          GZX(I)     = EXZ(I)/AREA(I)
          EXZZ2(I)   = GZX(I)*ZZ2(I)
          EXZ2(I)    = GZX(I)*GZX(I)*DT1V4
          EXZ2PY2(I) = EXZ2(I)*PY2(I)
          EXZ2PY1(I) = EXZ2(I)*PY1(I)
          VX1(I)=VX1(I)-EXZZ2(I)-EXZ2PY2(I)
          VX3(I)=VX3(I)-EXZZ2(I)+EXZ2PY2(I)
          VX2(I)=VX2(I)+EXZZ2(I)+EXZ2PY1(I)
          VX4(I)=VX4(I)+EXZZ2(I)-EXZ2PY1(I)
          ... the same five lines in y, with GZY/EYZZ2/EYZ2 and PX ...  l.192-200

So — unlike kinds 1 and 3 — this branch does not yield a pair of
coefficient fields.  It **is** a per-node modification of the four in-plane
velocity components, with a sign pattern that alternates between the 1-3
and the 2-4 diagonals, and each half contracts with the OPPOSITE operator
pair (the VX rows with ``PY``, the VY rows with ``PX``).
``_rot2_correction(..., kind=4)`` therefore returns the two ``(..., 4)``
nodal corrections, in upstream's node order.

Parity case **c45** (``c45_E1000_Bending_BT_BT_type4_Sf_0``,
``Ishell = 4`` -> ``GEO(171) = IHBE = 4``, ``hm_read_prop01.F:306-315``
-> ``cgrtails.F:956`` -> ``cforc3.F``) lands in this branch.

WHERE Z2 COMES FROM — a plan correction
---------------------------------------
``plan/04_phase3_elements_shell.md`` Task P3.2 says "Z2 is the second
surface curvature mode; getting it from ``ccurv3.F`` rather than
recomputing it is the point".  Read from the Fortran, both halves of that
sentence are wrong, and ``AGENTS.md`` is explicit about which wins ("If
the Fortran disagrees with your expectation, the Fortran wins"):

* ``Z2`` is **not a curvature**.  It is the corotational OUT-OF-PLANE
  COORDINATE of node 2 relative to node 1, ``cderi3.F:109``
  ``Z2(I)=E3X(I)*X21GA+E3Y(I)*Y21GA+E3Z(I)*Z21GA`` with
  ``Z21GA = Z2G(I)-Z1G(I)`` (l.95) — i.e. ``(x2-x1).e3``.  It is zeroed
  outright on the stress-based flat-membrane path, ``cpxpy3.F:77`` and
  l.88 (``Z2(I) = ZERO``), the branch ``ISMSTR == 3`` takes
  (``cforc3.F`` l.443-452).
* ``ccurv3.F`` does not produce ``Z2`` — it never mentions it.  Its
  outputs are the hourglass curvature coefficients ``KXX/KYY/KXY``, and
  ``cdefo3`` runs **before** it (``cforc3.F`` l.471 then l.476), so it
  cannot be the source either.

So the curvature state cannot be "extended to carry Z2": Z2 was never in
it.  It is an extra INPUT to this branch, and it is passed as ``z2``.

THE PLAN'S STEP-1 IDENTITIES, AGAIN
-----------------------------------
Plan Step 1 asserts "the correction vanishes on a flat element".  For
kind 4 that is **false as stated**, for the same reason it was false for
kind 3: ``Z2 == 0`` kills the warp half (``EXZZ2 = GZX*ZZ2``,
``EYZZ2 = GZY*ZZ2``) but leaves the velocity half (``EXZ2 = GZX^2*DT1V4``),
which is non-zero for any out-of-plane motion.  The two halves are
separable in the output and are pinned separately here; the plan's two
assertions are kept under the condition that makes them true, a flat
element with no out-of-plane motion.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.common.constants import EM20
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)

# the RD-E-1000 BT recipe (the c43 tests use the same shape)
A_, B_, T_ = 16.0, 15.2, 20.0
DT = 1e-4
#: rigid roll rate about the element's in-plane y axis.  As a LOCAL
#: velocity field omega x r with omega = (0, w, 0) is (w*z, 0, -w*x), so
#: the out-of-plane diagonal differences VZ13/VZ24 are non-zero — without
#: them every cdefo3 branch does nothing at all.
W = 0.05


# ---------------------------------------------------------------------------
# the transcription's own closed form, in upstream's order
# ---------------------------------------------------------------------------

def _closed(px1, px2, py1, py2, vz13, vz24, dt, z2, implicit=False):
    """cdefo3.F l.179-200 evaluated line by line.  Returns the two nodal
    correction vectors and the four coefficient fields."""
    area = max(2.0 * (py2 * px1 - py1 * px2), EM20)
    gzx = (px1 * vz13 + px2 * vz24) / area
    gzy = (py1 * vz13 + py2 * vz24) / area
    dt1v4 = 0.0 if implicit else 0.5 * dt
    zz2 = 0.5 * z2
    exzz2 = gzx * zz2
    eyzz2 = gzy * zz2
    exz2 = gzx * gzx * dt1v4
    eyz2 = gzy * gzy * dt1v4
    exz2py2 = exz2 * py2
    exz2py1 = exz2 * py1
    eyz2px2 = eyz2 * px2
    eyz2px1 = eyz2 * px1
    dx = np.array([-exzz2 - exz2py2, exzz2 + exz2py1,
                   -exzz2 + exz2py2, exzz2 - exz2py1])
    dy = np.array([-eyzz2 + eyz2px2, eyzz2 - eyz2px1,
                   -eyzz2 - eyz2px2, eyzz2 + eyz2px1])
    return dx, dy, (exz2, eyz2, exzz2, eyzz2)


def _ops(px1, px2, py1, py2):
    """(px, py) as the (n, 2) operator pairs _rot2_correction consumes."""
    return (np.array([[px1, px2]]), np.array([[py1, py2]]))


#: the numeric fixture shared by the closed-form tests
FIX = dict(px1=2.0, px2=-1.0, py1=3.0, py2=4.0, vz13=0.30, vz24=-0.10, z2=0.05)
_ZEROS2 = np.zeros((1, 2))


def _fix_call(**kw):
    px, py = _ops(FIX["px1"], FIX["px2"], FIX["py1"], FIX["py2"])
    z2 = kw.pop("z2", FIX["z2"])
    return shell_bt4._rot2_correction(px, py, np.array([[FIX["vz13"],
                                                        FIX["vz24"]]]),
                                      _ZEROS2, dt=DT, kind=4,
                                      z2=np.array([z2]), **kw)


# ---------------------------------------------------------------------------
# the plan's group fixtures
# ---------------------------------------------------------------------------

def _coords(warp):
    """The c45 element: flat, or with nodes 2 and 3 lifted out of plane by
    the deck's imperfection profile."""
    z = (0.0, warp, -0.3 * warp, 0.0)
    return np.array([[-A_ / 2, -B_ / 2, z[0]],
                     [A_ / 2, -B_ / 2, z[1]],
                     [A_ / 2, B_ / 2, z[2]],
                     [-A_ / 2, B_ / 2, z[3]]], dtype=float)


def _group(warp=0.0, roll=W):
    """The (px, py, vx, vy, z2) a one-element BT group hands
    ``_rot2_correction`` — the plan's ``_flat_bt4_group`` /
    ``_warped_bt4_group`` pair, built from the local corotational
    geometry ``forces()`` itself uses."""
    xe = _coords(warp)[None, :, :]
    _e, xl, area, B1, B2 = shell_bt4._local_geometry(xe)
    px = area[:, None] * np.stack((B1[:, 0], B1[:, 1]), axis=1)
    py = area[:, None] * np.stack((B2[:, 0], B2[:, 1]), axis=1)
    v = np.zeros((1, 4, 3))
    v[:, :, 0] = roll * xl[:, :, 2]
    v[:, :, 2] = -roll * xl[:, :, 0]
    vx = np.stack((v[:, 0, 2] - v[:, 2, 2],
                   v[:, 1, 2] - v[:, 3, 2]), axis=1)
    vy = np.stack((v[:, 0, 0] - v[:, 2, 0],
                   v[:, 1, 0] - v[:, 3, 0]), axis=1)
    # cderi3.F:109 — the local out-of-plane coordinate of node 2 rel. node 1
    return SimpleNamespace(px=px, py=py, vx=vx, vy=vy,
                           z2=xl[:, 1, 2] - xl[:, 0, 2])


def _flat_bt4_group():
    return _group(warp=0.0)


def _warped_bt4_group(warp):
    return _group(warp=warp)


def _group_closed(g, dt=DT, implicit=False):
    return _closed(g.px[0, 0], g.px[0, 1], g.py[0, 0], g.py[0, 1],
                   g.vx[0, 0], g.vx[0, 1], dt, g.z2[0], implicit)


# ---------------------------------------------------------------------------
# 1. the plan's two identities, under the condition that makes them true
# ---------------------------------------------------------------------------

def test_bt4_z2_warp_correction_is_zero_on_a_flat_element():
    """cdefo3.F l.179-200: on a flat element ``ZZ2 = Z2/2 = 0`` kills the
    warp half, and with no out-of-plane motion ``EXZ = PX1*VZ13 +
    PX2*VZ24`` and ``EYZ`` vanish too, so ``GZX = GZY = 0`` and every
    term — EXZZ2, EYZZ2, EXZ2, EYZ2 — is exactly zero.

    The second half of that sentence is the qualifier the plan's Step 1
    is missing; see the next test."""
    g = _flat_bt4_group()
    assert g.z2[0] == 0.0                       # cderi3.F:109 on a flat quad
    assert np.abs(g.vx).max() > 0.0             # the fixture CAN roll
    dx, dy = shell_bt4._rot2_correction(g.px, g.py, g.vx, g.vy, dt=DT,
                                        kind=4, z2=g.z2)
    assert np.abs(dx).max() > 0.0               # rolling, so NOT zero

    dx, dy = shell_bt4._rot2_correction(g.px, g.py, _ZEROS2, _ZEROS2,
                                        dt=DT, kind=4, z2=g.z2)
    assert np.abs(dx).max() == 0.0 and np.abs(dy).max() == 0.0

    # ... and the same holds for every implemented branch, for the same
    # reason: all three are driven by the out-of-plane diagonal velocities.
    for kind in (0, 1, 3):
        dx, dy = shell_bt4._rot2_correction(g.px, g.py, _ZEROS2, _ZEROS2,
                                            dt=DT, kind=kind)
        assert np.abs(dx).max() == 0.0, kind
        assert np.abs(dy).max() == 0.0, kind


def test_bt4_z2_is_nonzero_on_a_warped_element():
    g = _warped_bt4_group(warp=0.02)
    assert g.z2[0] != 0.0
    dx, dy = shell_bt4._rot2_correction(g.px, g.py, g.vx, g.vy, dt=DT,
                                        kind=4, z2=g.z2)
    assert np.abs(dx).max() > 0.0
    assert np.abs(dy).max() > 0.0


def test_a_flat_element_kills_only_the_warp_half_not_the_whole_branch():
    """WHY the plan's "flat element -> vanishes" needs its qualifier here.

    ``ZZ2 = HALF*Z2`` (l.180) is the only place Z2 enters, and it feeds
    only ``EXZZ2 = GZX*ZZ2`` (l.183) and ``EYZZ2 = GZY*ZZ2`` (l.193).
    The velocity half ``EXZ2 = GZX*GZX*DT1V4`` (l.184) carries no Z2, so
    a FLAT element rolling out of plane is still corrected.  The two
    halves are separable in the output — that is what makes both testable.

    The separation is exact, not approximate: in the returned vector the
    warp half is the even part (d1 + d3, d2 + d4) and the velocity half
    the odd part (d1 - d3, d2 - d4).
    """
    flat = _flat_bt4_group()
    warped = _warped_bt4_group(warp=0.02)
    dx_f, dy_f, _c = _group_closed(flat)
    dx_w, _dy_w, _c = _group_closed(warped)

    assert np.abs(dx_f).max() > 0.0              # NOT zero: EXZ2 survives
    exz2, eyz2, _exzz2, _eyzz2 = _group_closed(flat)[2]
    assert dx_f == pytest.approx([-exz2 * flat.py[0, 1],
                                  exz2 * flat.py[0, 0],
                                  exz2 * flat.py[0, 1],
                                  -exz2 * flat.py[0, 0]], rel=1e-12)
    assert dy_f == pytest.approx([eyz2 * flat.px[0, 1],
                                  -eyz2 * flat.px[0, 0],
                                  -eyz2 * flat.px[0, 1],
                                  eyz2 * flat.px[0, 0]], rel=1e-12)
    assert not np.allclose(dx_f, dx_w)           # and Z2 does change it
    assert np.abs(dx_f[0] + dx_f[2]) < 1e-300    # even part ~ Z2 = 0


# ---------------------------------------------------------------------------
# 2. the closed form, node for node
# ---------------------------------------------------------------------------

def test_bt4_branch_is_the_four_node_velocity_modification():
    """The whole branch, node by node, in upstream's order.  The warp term
    ``EXZZ2`` goes ``-`` into nodes 1 and 3 and ``+`` into nodes 2 and 4,
    while ``EXZ2`` follows the opposite sign of its operator — ``PY2`` on
    the 1-3 pair, ``PY1`` on the 2-4 pair."""
    dx, dy = _fix_call()
    exz2, eyz2, exzz2, eyzz2 = _closed(**FIX, dt=DT)[2]
    assert dx.shape == (1, 4) and dy.shape == (1, 4)
    assert dx[0] == pytest.approx([-exzz2 - exz2 * FIX["py2"],
                                   exzz2 + exz2 * FIX["py1"],
                                   -exzz2 + exz2 * FIX["py2"],
                                   exzz2 - exz2 * FIX["py1"]], rel=1e-12)
    assert dy[0] == pytest.approx([-eyzz2 + eyz2 * FIX["px2"],
                                   eyzz2 - eyz2 * FIX["px1"],
                                   -eyzz2 - eyz2 * FIX["px2"],
                                   eyzz2 + eyz2 * FIX["px1"]], rel=1e-12)


def test_bt4_contracts_vx_rows_with_py_and_vy_rows_with_px():
    """``cdefo3.F`` l.185-190: the VX rows contract with ``PY`` and the VY
    rows with ``PX`` — the transpose of how a reader would guess it.  The
    whole branch is invisible if you swap them: the result is still a
    plausible non-zero vector, just the wrong one."""
    dx, _dy = _fix_call()
    exz2 = _closed(**FIX, dt=DT)[2][0]
    # node 2 is +EXZZ2 + EXZ2*PY1 and node 4 is +EXZZ2 - EXZ2*PY1, so the
    # 2-4 difference isolates 2*EXZ2*PY1 and NOT 2*EXZ2*PX1
    assert dx[0, 1] - dx[0, 3] == pytest.approx(2.0 * exz2 * FIX["py1"],
                                                rel=1e-12)
    assert dx[0, 1] - dx[0, 3] != pytest.approx(2.0 * exz2 * FIX["px1"],
                                                rel=1e-12)


def test_bt4_output_reconstructs_the_exz2_membrane_term():
    """``forces()``'s inline kind-4 block also subtracts ``EXZ2`` from the
    membrane rate ``dm[:, 0]``.  ``ZZ2`` cancels in the 2-4 difference, so
    ``EXZ2 = (VX2_corr - VX4_corr)/(2*PY1)``: the transcription carries
    that membrane term too, and this pins it."""
    dx, dy = _fix_call()
    exz2, eyz2, _exzz2, _eyzz2 = _closed(**FIX, dt=DT)[2]
    assert (dx[0, 1] - dx[0, 3]) / (2.0 * FIX["py1"]) == pytest.approx(
        exz2, rel=1e-12)
    # the y rows contract with PX and carry the OPPOSITE sign, because the
    # 2-4 pair is +EYZZ2-EYZ2*PX1 at node 2 and +EYZZ2+EYZ2*PX1 at node 4
    assert -(dy[0, 1] - dy[0, 3]) / (2.0 * FIX["px1"]) == pytest.approx(
        eyz2, rel=1e-12)


def test_bt4_is_not_kind3_or_kind1_on_the_same_input():
    """Plan Step 1's second assertion, for kind 4.  Kinds 1 and 3 return
    one coefficient per component; kind 4 returns four nodal corrections
    per component, and on the same rolled input no two agree."""
    px, py = _ops(FIX["px1"], FIX["px2"], FIX["py1"], FIX["py2"])
    vx = np.array([[FIX["vz13"], FIX["vz24"]]])
    dx4, dy4 = shell_bt4._rot2_correction(px, py, vx, _ZEROS2, dt=DT,
                                          kind=4, z2=np.array([FIX["z2"]]))
    dx3, _dy3 = shell_bt4._rot2_correction(px, py, vx, _ZEROS2, dt=DT,
                                           kind=3)
    dx1, _dy1 = shell_bt4._rot2_correction(px, py, vx, _ZEROS2, dt=DT,
                                           kind=1)
    assert dx4.shape == (1, 4)
    assert dx3.shape == (1,) and dx1.shape == (1,)
    assert np.abs(dx4).max() != np.abs(dx3).max()
    assert np.abs(dx4).max() != np.abs(dx1).max()
    assert dy4.shape == (1, 4)


# ---------------------------------------------------------------------------
# 3. the guards: Z2 is required, and IMPL_S>0 is NOT dt == 0 here
# ---------------------------------------------------------------------------

def test_kind4_without_z2_is_refused_loudly():
    """Z2 is an INPUT to this branch, not part of the curvature state
    (``cderi3.F:109``; ``cpxpy3.F:77``/l.88 zeroes it on the ISMSTR == 3
    path).  Guessing it would be guessing physics, so a missing Z2 must be
    a loud refusal, never a silent zero."""
    px, py = _ops(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="cderi3"):
        shell_bt4._rot2_correction(px, py, _ZEROS2, _ZEROS2, dt=DT, kind=4)


def test_implicit_disable_is_dt_zero_for_kinds_1_and_3():
    """cdefo3.F l.109 / l.137: ``IF(IMPL_S>0) DT1V4=ZERO`` and EVERY term
    of those branches carries ``DT1V4``, so the implicit static branch is
    exactly ``dt == 0`` there."""
    px, py = _ops(2.0, -1.0, 3.0, 4.0)
    vx, vy = np.array([[0.3, 0.2]]), np.array([[1.0, 0.0]])
    for kind in (0, 1, 3):
        for kwargs in ({"dt": 0.0}, {"dt": DT, "implicit": True}):
            dx, dy = shell_bt4._rot2_correction(px, py, vx, vy, kind=kind,
                                                **kwargs)
            assert np.abs(dx).max() == 0.0, (kind, kwargs)
            assert np.abs(dy).max() == 0.0, (kind, kwargs)


def test_implicit_disable_on_kind4_keeps_the_z2_terms():
    """THE POINT WHERE KIND 4 IS NOT LIKE KIND 3.

    ``IF(IMPL_S>0) DT1V4=ZERO`` (l.178) zeroes ``EXZ2`` and ``EYZ2`` only.
    ``EXZZ2 = GZX*ZZ2`` (l.183) and ``EYZZ2 = GZY*ZZ2`` (l.193) carry **no**
    ``DT1V4``, so the warp half survives the implicit disable: ``dt == 0``
    is NOT the disable on this branch, and treating it as one would delete
    real upstream behaviour.  The guard is therefore carried explicitly
    (``implicit=True``) instead of being folded into ``dt``, and this pins
    the asymmetry.
    """
    dx_on, _dy_on = _fix_call()
    dx_off, dy_off = _fix_call(implicit=True)
    _exz2, _eyz2, exzz2, eyzz2 = _closed(**FIX, dt=DT, implicit=True)[2]
    assert dx_off[0] == pytest.approx([-exzz2, exzz2, -exzz2, exzz2],
                                      rel=1e-12)
    assert dy_off[0] == pytest.approx([-eyzz2, eyzz2, -eyzz2, eyzz2],
                                      rel=1e-12)
    assert np.abs(dx_off).max() > 0.0
    assert not np.allclose(dx_on, dx_off)       # the velocity half is gone

    # ... and dt == 0 lands on the same state as the explicit disable here,
    # because on THIS branch every DT1V4-carrying term is already zero and
    # the Z2 half never carried DT1V4 in the first place
    px, py = _ops(FIX["px1"], FIX["px2"], FIX["py1"], FIX["py2"])
    vx = np.array([[FIX["vz13"], FIX["vz24"]]])
    dx_zero, _dy_zero = shell_bt4._rot2_correction(
        px, py, vx, _ZEROS2, dt=0.0, kind=4, z2=np.array([FIX["z2"]]))
    assert dx_zero[0] == pytest.approx(dx_off[0], rel=1e-12)


def test_z2_scales_the_warp_half_linearly():
    """``EXZZ2 = GZX*ZZ2`` is linear in ``Z2``; ``EXZ2`` does not carry it.
    In the returned vector the warp half is the EVEN part and the velocity
    half the ODD part, under the ``1<->3, 2<->4`` swap the branch's sign
    pattern lives on — so doubling the warp doubles the even part and
    leaves the odd one alone."""
    one, _ = _fix_call(z2=0.01)
    two, _ = _fix_call(z2=0.02)
    swap = [2, 3, 0, 1]                 # the 1-3 / 2-4 diagonal swap
    odd_one = 0.5 * (one[0] - one[0][swap])
    odd_two = 0.5 * (two[0] - two[0][swap])
    even_one = 0.5 * (one[0] + one[0][swap])
    even_two = 0.5 * (two[0] + two[0][swap])
    assert np.abs(odd_one).max() > 0.0          # the EXZ2 half is live ...
    assert np.allclose(odd_one, odd_two, rtol=1e-12)       # ... and Z2-free
    assert np.abs(even_one).max() > 0.0
    assert np.allclose(even_one, 0.5 * even_two, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. the production wiring: c45 (Ishell = 4) takes the kind-4 branch
# ---------------------------------------------------------------------------

def _one_element(tmp_path, ishell=4, warp=0.02):
    """One free BT shell with the given /PROP/SHELL Ishell card and the
    c45 imperfection profile."""
    z = (0.0, warp, -0.3 * warp, 0.0)
    deck = (
        "/BEGIN\np32 c45\n/NODE\n"
        f"1 {-A_/2} {-B_/2} {z[0]}\n2 {A_/2} {-B_/2} {z[1]}\n"
        f"3 {A_/2} {B_/2} {z[2]}\n4 {-A_/2} {B_/2} {z[3]}\n"
        "/SHELL/1\n1 1 2 3 4\n/PART/1\np\n1 1\n"
        "/MAT/LAW1/1\nm\n0.01\n1000. 0.\n"
        f"/PROP/SHELL/1\nsh\n{ishell} 0 0 0\n0 0 0 0 0\n3 0 {T_}\n/END\n")
    f = tmp_path / "P32_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model


def _roty(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rolled(m, theta, w, dt):
    """Stencil-exact rigid roll about the in-plane y axis:
    x = R(theta)x0 and v = (x^n - x^{n-1})/dt."""
    x0 = m.x0
    m.x = x0 @ _roty(theta).T
    m.v = (m.x - x0 @ _roty(theta - w * dt).T) / dt
    m.vr = np.zeros_like(m.v)
    m.vr[:, 1] = w


def test_ishell4_maps_to_engine_ihbe4_and_kind4(tmp_path):
    """``hm_read_prop01.F:306-315`` maps card 4 -> GEO(171) = 4 — the one
    card that is NOT ``card-1`` — so c45's ``Ishell = 4`` runs the
    ``ELSEIF(IHBE==4)`` branch and neither of its neighbours."""
    g = _one_element(tmp_path, ishell=4).shells
    assert int(g.state["ihbe_mask"][0]) == 4
    assert shell_bt4._ROT2_KIND[4] == 4


def test_production_kind4_block_is_the_transcribed_closed_form(tmp_path):
    """forces()'s ``mask4`` block is the transcribed kind-4 branch, kept
    inline for the reason P3.1 measured on c43 (one floating-point bit is
    worth 34 % of that deck's final internal energy).  This pins the two
    together so they cannot drift.

    They are built differently on purpose: the inline block forms
    ``GZX = np.sum(B1*VZ)`` — the raw rate — while the transcription
    rebuilds it from the AREA-scaled operator pair the way the module's
    other kinds do.  Algebraically equal, so this asserts to 1e-12.
    """
    m = _one_element(tmp_path, ishell=4, warp=0.02)
    g = m.shells
    _rolled(m, 0.7, W, DT)          # gives non-zero VZ13/VZ24
    E, area, _lc, B1, B2, _bb, _gam, V, _dm, _kap, _gs = shell_bt4._pre(
        m.x[g.conn], m.v[g.conn], m.vr[g.conn], g.state["off"])
    # cderi3.F:109 — _local_geometry returns node coordinates relative to
    # the element centre, so the node-1 offset cancels in the difference
    _e, xl, _a, _b1, _b2 = shell_bt4._local_geometry(m.x[g.conn])
    z2 = xl[:, 1, 2] - xl[:, 0, 2]

    gzx = np.sum(B1 * V[:, :, 2], axis=1)
    gzy = np.sum(B2 * V[:, :, 2], axis=1)
    zz2 = 0.5 * z2
    exzz2, eyzz2 = gzx * zz2, gzy * zz2
    exz2, eyz2 = gzx * gzx * (0.5 * DT), gzy * gzy * (0.5 * DT)
    py1, py2 = area * B2[:, 0], area * B2[:, 1]
    px1, px2 = area * B1[:, 0], area * B1[:, 1]
    inline_x = np.stack([-exzz2 - exz2 * py2, exzz2 + exz2 * py1,
                         -exzz2 + exz2 * py2, exzz2 - exz2 * py1], axis=1)
    inline_y = np.stack([-eyzz2 + eyz2 * px2, eyzz2 - eyz2 * px1,
                         -eyzz2 - eyz2 * px2, eyzz2 + eyz2 * px1], axis=1)

    px = area[:, None] * np.stack((B1[:, 0], B1[:, 1]), axis=1)
    py = area[:, None] * np.stack((B2[:, 0], B2[:, 1]), axis=1)
    vxz = np.stack((V[:, 0, 2] - V[:, 2, 2],
                    V[:, 1, 2] - V[:, 3, 2]), axis=1)

    # the two ways of forming GZX/GZY are algebraically equal but not
    # identically associated, and on this warped quad EXZ = PX1*VZ13 +
    # PX2*VZ24 cancels to ~7 significant digits.  So this asserts them equal
    # at that precision and the transcription itself exactly, in the
    # closed-form tests above.
    mod_gzx = (px[:, 0] * vxz[:, 0] + px[:, 1] * vxz[:, 1]) / area
    mod_gzy = (py[:, 0] * vxz[:, 0] + py[:, 1] * vxz[:, 1]) / area
    assert mod_gzx == pytest.approx(gzx, rel=1e-9)
    assert mod_gzy == pytest.approx(gzy, rel=1e-9)

    dx, dy = shell_bt4._rot2_correction(px, py, vxz, vxz, DT, kind=4,
                                        z2=z2)
    got_x = np.abs(dx - inline_x) / np.abs(inline_x)
    got_y = np.abs(dy - inline_y) / np.abs(inline_y)
    assert got_x.max() < 1e-9, f"inline/transcribed x disagree: {got_x}"
    assert got_y.max() < 1e-9, f"inline/transcribed y disagree: {got_y}"


def test_production_kind4_block_moves_the_resultants(tmp_path):
    """The branch's stated purpose — "velocity modification for the
    lurglass membrane" (cdefo3.F l.179) — is observable in ``forces()``:
    switching the ``mask4`` path off changes the element resultant, on a
    warped element AND on a flat one (the EXZ2 half needs no Z2)."""
    for warp in (0.02, 0.0):
        on = _resultant(tmp_path / f"w{warp}", ishell=4, warp=warp)
        off = _resultant(tmp_path / f"w{warp}", ishell=4, warp=warp,
                         force_off=True)
        assert not np.allclose(on, off), warp


def _resultant(tmp_path, ishell, warp, force_off=False):
    tmp_path.mkdir(exist_ok=True)
    m = _one_element(tmp_path, ishell=ishell, warp=warp)
    g = m.shells
    if force_off:
        g.state["ihbe_mask"][:] = 99          # no cdefo3 branch at all
    _rolled(m, 0.7, W, DT)
    fint, mint = np.zeros((1, 3)), np.zeros((1, 3))
    shell_bt4.forces(g, m.x, m.v, m.vr, DT, fint, mint)
    return fint.copy()


def test_c45_deck_runs_the_non_brat_path_that_zeroes_z2_upstream():
    """The c45 deck carries no ``/BRAT`` card, so it runs ``ISMSTR == 3``
    and ``cforc3.F`` takes the ``CPXPY3`` path, which sets ``Z2 = ZERO``
    (``cpxpy3.F:77`` and l.88).  Upstream's Z2 warp half is therefore
    identically zero on this very parity case — which is the reason the
    P3.2 production behaviour is provably untouched, and which also says
    the port's always-corotational ``z2`` does not model that zeroing (a
    pre-existing, wider ISMSTR gap, not this task's to close)."""
    text = Path("tests/data/rd_decks/rd_e/RD-E-1000_Bending/10_Bending/BT/"
                "BT_type4/Sf_0.9/ROLLING_0000.rad").read_text()
    assert "/BRAT" not in text.upper()
    assert "/PROP/SHELL/202" in text