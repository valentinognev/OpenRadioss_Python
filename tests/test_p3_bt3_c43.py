"""P3.1 — ``cdefo3.F`` BT type 3 (parity case c43) node-1-relative velocity form.

THE UPSTREAM CONTRACT (read from the Fortran, not extrapolated)
--------------------------------------------------------------
``engine/source/elements/shell/coque/cdefo3.F`` has three formulation
branches, selected on the ENGINE formulation flag ``IHBE`` (= ``Ishell``
card 3 -> ``GEO(171) = IHBE-1 = 2``, ``hm_read_prop01.F:306-315``,
``cgrtails.F:956`` -> ``IPARG(23)`` -> ``cforc3.F``):

  ``IF(IHBE<=1)THEN``           l.104-131   "general"    (kind 1)
  ``ELSEIF(IHBE==2.OR.IHBE==3)`` l.133-173   node-1-relative (kind 3)
  ``ELSEIF(IHBE==4)THEN``       l.174-...   Z2 warp      (kind 4, Task P3.2)

Parity case **c43** (``c43_E1000_Bending_BT_BT_type3_Sf_0``,
``Ishell = 3``) lands in the **kind 3** branch.  Its whole correction is

    GZX    = EXZ / AREA                               (l.141)
    EXZ2   = GZX * GZX * DT1V4                        (l.142)   DT1V4 = dt/2
    EXZZ2  = GZX * Z2                                 (l.140)
    VX3   -= EXZ2 * X3 - VX1      VX2 += EXZZ2 - EXZ2*X2 - VX1   (l.144-146)
    VX4   += EXZZ2 - EXZ2 * X4 - VX1                  VX1 = 0
    ... (same in Y, plus ZZZ = (EXZ2+EYZ2)*Z2 and the VZ rows)  l.148-159
    VX13 = -VX3 ; VX24 = VX2 - VX4                    (l.161-162)
    EXX  = PX1*VX13 + PX2*VX24                        (l.164)

with ``X3 = -2*PY2``, ``X4-X2 = 2*PY1`` (``cderi3.F:173-176``,
``cpxpy3.F:75-82``), which collapses the membrane-rate half to the
EXACT cancellation of the frame-lag bias:

    dEXX/AREA = -EXZ2          dEYY/AREA = -EYZ2      dEXY = 0

That is the asymmetry the M41 note records: the type-1 branch
**over-corrects 2x** (``+w^2 dt/2`` bias, ``-w^2 dt`` correction), the
type-3 branch is the **exact** cancellation.  ``_rot2_correction`` is
where that difference lives, one ``kind`` per branch.

``DT1C`` is the condensation length: ``cdt3.F`` computes the element
step and ``ccoor3.F:124`` sets ``DT1C(I) = DT1`` (the cycle step), so
``DT1V4 = HALF*DT1C = dt/2``.  ``IF(IMPL_S>0) DT1V4=ZERO`` (l.137) kills
every term; ``DT1V4`` multiplies all of them, so ``dt == 0`` is that
disable verbatim.

A NOTE ON THE PLAN'S STEP-1 IDENTITY
------------------------------------
``plan/04_phase3_elements_shell.md`` Task P3.1 Step 1 sketches
"with ``VZ13 == VZ24`` the correction vanishes".  That identity is the
**kind-1** form (``TMP1A`` carries ``(VZ13-VZ24)^2``); it is **not** true
of the kind-3 branch, which is driven by the out-of-plane membrane rate
``EXZ = PX1*VZ13 + PX2*VZ24`` and stays non-zero at ``VZ13 == VZ24`` for
any ``PX1 + PX2 != 0``.  Per ``AGENTS.md`` ("If the Fortran disagrees with
your expectation, the Fortran wins") the identity pinned here is the
Fortran's own: the kind-3 correction vanishes when the out-of-plane
motion does, and it is structurally different from the kind-1 form.
"""

from __future__ import annotations

import contextlib
import io

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

# the RD-E-1000 BT recipe (cdefo3.F fixtures use the same shape)
A_, B_, T_ = 16.0, 15.2, 20.0


def _one_element(tmp_path, ishell=3, warp=0.0):
    """One free BT shell with the given /PROP/SHELL Ishell card."""
    z = (0.0, warp, -0.3 * warp, 0.0)
    deck = (
        "/BEGIN\np31 c43\n/NODE\n"
        f"1 {-A_/2} {-B_/2} {z[0]}\n2 {A_/2} {-B_/2} {z[1]}\n"
        f"3 {A_/2} {B_/2} {z[2]}\n4 {-A_/2} {B_/2} {z[3]}\n"
        "/SHELL/1\n1 1 2 3 4\n/PART/1\np\n1 1\n"
        "/MAT/LAW1/1\nm\n0.01\n1000. 0.\n"
        f"/PROP/SHELL/1\nsh\n{ishell} 0 0 0\n0 0 0 0 0\n3 0 {T_}\n/END\n")
    f = tmp_path / "P31_0000.rad"
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


def _rolled(model, theta, w, dt):
    """Stencil-exact rigid roll about the in-plane y axis:
    x = R(theta) x0 and v = (x^n - x^{n-1})/dt."""
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    x0 = model.x0
    model.x = x0 @ r.T
    model.v = (model.x - x0 @ _roty(theta - w * dt).T) / dt
    model.vr = np.zeros_like(model.v)
    model.vr[:, 1] = w


def _roty(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _ops(px1, px2, py1, py2):
    """(px, py) as the (n, 2) operator pairs _rot2_correction consumes."""
    return (np.array([[px1, px2]]), np.array([[py1, py2]]))


# ---------------------------------------------------------------------------
# 1. the plan's identity, corrected to the Fortran's
# ---------------------------------------------------------------------------

def test_bt3_correction_vanishes_without_out_of_plane_motion():
    """cdefo3.F l.142: EXZ2 = GZX^2*DT1V4 and GZX = EXZ/AREA with
    EXZ = PX1*VZ13 + PX2*VZ24 — so no out-of-plane diagonal velocity, no
    correction.  Both branches, and both components."""
    px, py = _ops(1e5, 1e5, 1e5, 1e5)
    for kind in (0, 1, 3):
        dx, dy = shell_bt4._rot2_correction(px, py,
                                            np.array([[0.0, 0.0]]),
                                            np.array([[0.0, 0.0]]),
                                            dt=1e-4, kind=kind)
        assert np.abs(dx).max() == 0.0, kind
        assert np.abs(dy).max() == 0.0, kind


def test_bt3_differs_from_the_general_branch_on_the_same_input():
    """The plan's second assertion.  Same px/py/vx/vy: the kind-1 branch
    carries (VZ13 - VZ24)^2 and vanishes, the kind-3 branch is driven by
    (EXZ/AREA)^2 and does not."""
    px, py = _ops(1e5, 1e5, 1e5, 1e5)
    vx, vy = np.array([[0.3, 0.3]]), np.array([[0.0, 0.0]])
    dx3, dy3 = shell_bt4._rot2_correction(px, py, vx, vy, dt=1e-4, kind=3)
    dx1, _ = shell_bt4._rot2_correction(px, py, vx, vy, dt=1e-4, kind=1)
    assert np.abs(dx3).max() != np.abs(dx1).max()


def test_bt1_is_the_diagonal_difference_form_and_bt3_the_membrane_rate_form():
    """The two branches are DIFFERENT closed forms, pinned separately:

      kind 1  TMP1A = (dt/4) (VZ13-VZ24)^2 / SIGN(MAX(|PY1+PY2|,EM20),PY1+PY2)
              TMP2B = (dt/4) (VZ13+VZ24)^2 / SIGN(MAX(|PX2-PX1|,EM20),PX2-PX1)
      kind 3  EXZ2  = (EXZ/AREA)^2 * dt/2,  EXZ = PX1*VZ13 + PX2*VZ24
              EYZ2  = (EYZ/AREA)^2 * dt/2,  EYZ = PY1*VZ13 + PY2*VZ24
              AREA  = 2 (PY2*PX1 - PY1*PX2)
    """
    px, py = _ops(2.0, -1.0, 3.0, 4.0)
    vz13, vz24, dt = 0.30, -0.10, 1e-4
    vx, vy = np.array([[vz13, vz24]]), np.array([[1.0, 0.0]])

    dx1, dy1 = shell_bt4._rot2_correction(px, py, vx, vy, dt=dt, kind=1)
    a = py[0, 0] + py[0, 1]
    b = px[0, 1] - px[0, 0]
    assert dx1[0] == pytest.approx(
        0.25 * dt * (vz13 - vz24) ** 2 / np.copysign(max(abs(a), EM20), a),
        rel=1e-12, abs=0.0)
    assert dy1[0] == pytest.approx(
        0.25 * dt * (vz13 + vz24) ** 2 / np.copysign(max(abs(b), EM20), b),
        rel=1e-12, abs=0.0)

    dx3, dy3 = shell_bt4._rot2_correction(px, py, vx, vy, dt=dt, kind=3)
    area = 2.0 * (py[0, 1] * px[0, 0] - py[0, 0] * px[0, 1])
    exz = px[0, 0] * vz13 + px[0, 1] * vz24
    eyz = py[0, 0] * vz13 + py[0, 1] * vz24
    assert dx3[0] == pytest.approx((exz / area) ** 2 * 0.5 * dt, rel=1e-12)
    assert dy3[0] == pytest.approx((eyz / area) ** 2 * 0.5 * dt, rel=1e-12)


def test_bt3_is_driven_by_the_membrane_rate_not_the_diagonal_difference():
    """WHY the plan's "VZ13 == VZ24 -> vanishes" identity is the kind-1
    one and not this branch's.  cdefo3.F l.142 drives kind 3 by
    ``EXZ = PX1*VZ13 + PX2*VZ24`` (the out-of-plane membrane rate), so at
    ``VZ13 == VZ24 != 0`` it is NON-zero while kind 1's ``(VZ13-VZ24)^2``
    is exactly zero.  That difference is the whole point of keeping the
    branches apart."""
    w, dt, dy_diag = 0.05, 0.02, 2.0
    # the plan's fixture: PX1 = PX2 = PY1 = PY2, so AREA degenerates and
    # the SIGN(MAX(ABS,EM20)) floor is what keeps EXZ/AREA finite
    px, py = _ops(1e5, 1e5, 1e5, 1e5)
    vx = np.array([[w * dy_diag, w * dy_diag]])
    vy = np.array([[0.0, 0.0]])

    dx1, _ = shell_bt4._rot2_correction(px, py, vx, vy, dt=dt, kind=1)
    dx3, _ = shell_bt4._rot2_correction(px, py, vx, vy, dt=dt, kind=3)
    assert dx1[0] == 0.0                          # (VZ13-VZ24)^2 == 0
    assert dx3[0] > 0.0                           # (EXZ/AREA)^2 * dt/2 > 0
    area = max(2.0 * (py[0, 1] * px[0, 0] - py[0, 0] * px[0, 1]), EM20)
    exz = px[0, 0] * vx[0, 0] + px[0, 1] * vx[0, 1]
    assert dx3[0] == pytest.approx((exz / area) ** 2 * 0.5 * dt, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. the guards that must be copied verbatim
# ---------------------------------------------------------------------------

def test_sign_max_abs_em20_guards_are_carried():
    """cdefo3.F l.111 / l.122: SIGN(MAX(ABS(x),EM20),x) — note the SIGN, so
    a denominator of exactly zero is taken as +EM20 (never -EM20) and a
    negative one flips the sign of the correction."""
    vx = np.array([[0.4, -0.4]])
    vxz = np.zeros((1, 2))

    # PY1 + PY2 == 0 -> +EM20, and TMP1A keeps its (VZ13-VZ24)^2 numerator
    px, py = _ops(2.0, 1.0, 5.0, -5.0)
    dx, _ = shell_bt4._rot2_correction(px, py, vx, vxz, dt=1e-2, kind=1)
    assert dx[0] == pytest.approx(0.25 * 1e-2 * 0.64 / EM20, rel=1e-12)

    # PX2 - PX1 == 0 -> +EM20 on the y half as well
    px, py = _ops(2.0, 2.0, 5.0, -5.0)
    _, dy = shell_bt4._rot2_correction(px, py, vx, vxz, dt=1e-2, kind=1)
    assert dy[0] == pytest.approx(0.25 * 1e-2 * 0.0 / EM20, rel=1e-12,
                                  abs=0.0)

    # a NEGATIVE denominator flips the sign
    px, py = _ops(2.0, 1.0, -5.0, -5.0)
    dx, _ = shell_bt4._rot2_correction(px, py, vx, vxz, dt=1e-2, kind=1)
    assert dx[0] == pytest.approx(-0.25 * 1e-2 * 0.64 / 10.0, rel=1e-12)


def test_implicit_disable_zeroes_every_implemented_kind():
    """cdefo3.F l.109 / l.137 / l.178: IF(IMPL_S>0) DT1V4=ZERO, and every
    term carries DT1V4 — so the implicit static branch is exactly dt == 0."""
    px, py = _ops(2.0, -1.0, 3.0, 4.0)
    vx = np.array([[0.3, 0.2]])
    vy = np.array([[1.0, 0.0]])
    for kind in (0, 1, 3):
        dx, dy = shell_bt4._rot2_correction(px, py, vx, vy, dt=0.0, kind=kind)
        assert np.abs(dx).max() == 0.0, kind
        assert np.abs(dy).max() == 0.0, kind


def test_kind4_is_refused_not_silently_dropped():
    """The IHBE==4 Z2-warp branch is Task P3.2; it must be a loud refusal,
    never a quiet zero (AGENTS.md: a cut must stay ported OR explicitly
    refused)."""
    px, py = _ops(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(NotImplementedError):
        shell_bt4._rot2_correction(px, py, np.zeros((1, 2)), np.zeros((1, 2)),
                                   dt=1e-4, kind=4)


# ---------------------------------------------------------------------------
# 3. the production blocks ARE these closed forms
# ---------------------------------------------------------------------------

def _prod_coeffs(ishell, tmp_path, theta=0.7, w=0.05, dt=0.02):
    """(tmp1a, tmp2b) for kind 1, (exz2, eyz2) for kind 3, recomputed from
    the SAME element state forces() sees on a stencil-exact roll."""
    m = _one_element(tmp_path, ishell=ishell)
    g = m.shells
    _rolled(m, theta, w, dt)
    E, area, _lc, B1, B2, _bb, _gam, V, _dm, _kap, _gs = shell_bt4._pre(
        m.x[g.conn], m.v[g.conn], m.vr[g.conn], g.state["off"])
    px = area[:, None] * np.stack((B1[:, 0], B1[:, 1]), axis=1)
    py = area[:, None] * np.stack((B2[:, 0], B2[:, 1]), axis=1)
    vxz = np.stack((V[:, 0, 2] - V[:, 2, 2], V[:, 1, 2] - V[:, 3, 2]), axis=1)
    vxx = np.stack((V[:, 0, 0] - V[:, 2, 0], V[:, 1, 0] - V[:, 3, 0]), axis=1)
    kind = shell_bt4._ROT2_KIND[int(g.state["ihbe_mask"][0])]
    if kind == 1:
        # the INLINE operands forces() forms (area*(B2_0+B2_1), not the
        # area-scaled pair sum — see _rot2_correction's docstring)
        t2a = area * (B2[:, 0] + B2[:, 1])
        t3a = np.copysign(np.maximum(np.abs(t2a), EM20), t2a)
        tmp1a = (0.25 * dt) * (vxz[:, 0] - vxz[:, 1]) ** 2 / t3a
        t1b = area * (B1[:, 1] - B1[:, 0])
        t3b = np.copysign(np.maximum(np.abs(t1b), EM20), t1b)
        tmp2b = (0.25 * dt) * (vxz[:, 0] + vxz[:, 1]) ** 2 / t3b
        return kind, (tmp1a, tmp2b), shell_bt4._rot2_correction(
            px, py, vxz, vxx, dt, kind=1)
    gzx = np.sum(B1 * V[:, :, 2], axis=1)
    gzy = np.sum(B2 * V[:, :, 2], axis=1)
    return kind, (gzx * gzx * 0.5 * dt, gzy * gzy * 0.5 * dt), \
        shell_bt4._rot2_correction(px, py, vxz, vxx, dt, kind=3)


@pytest.mark.parametrize("ishell,kind", [(1, 1), (3, 3), (4, 4)])
def test_production_blocks_are_the_transcribed_closed_forms(
        tmp_path, ishell, kind):
    """forces()'s two inline blocks implement _rot2_correction kind 1 and
    kind 3.  They are kept inline because the c43 benchmark turns the
    last bit of the operand grouping into 34 % of the final IE (see the
    docstring); this pins them to the transcribed form so the two cannot
    drift apart.  kind 4 is P3.2's and is not in forces() as this."""
    assert shell_bt4._ROT2_KIND[ishell] == kind
    if kind == 4:
        return
    got_kind, inline, fn = _prod_coeffs(ishell, tmp_path)
    assert got_kind == kind
    for a, b in zip(inline, fn):
        assert a == pytest.approx(b, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# 4. the production wiring: c43 (Ishell = 3) takes the kind-3 branch
# ---------------------------------------------------------------------------

def _deps_xx(ishell, tmp_path, force_off=False):
    """deps_xx on one stencil-exact rigid roll of a single BT element."""
    w, dt, n = 0.05, 0.02, 4
    m = _one_element(tmp_path, ishell=ishell)
    g = m.shells
    if force_off:                       # no cdefo3 branch at all
        g.state["ihbe_mask"][:] = 99
    _rolled(m, 0.7, w, dt)
    fint, mint = np.zeros((n, 3)), np.zeros((n, 3))
    shell_bt4.forces(g, m.x, m.v, m.vr, dt, fint, mint)
    return g.state["sig"][0, :, 0].mean() / 1000.0            # E = 1000, nu = 0


def test_ishell3_takes_the_node1_relative_branch_not_the_general_one(
        tmp_path):
    """One production forces() call on a stencil-exact roll of an
    Ishell = 3 element.

    The RAW frame-lag bias is +w^2 dt/2 (measured: +5.0e-7 at w=0.05,
    dt=0.02).  The kind-3 branch is the EXACT cancellation of it — the
    (EXZ/AREA)^2 * dt/2 velocity form — so the net membrane strain
    increment must vanish to round-off.  The kind-1 branch instead
    over-corrects by 2x and lands on -w^2 dt/2."""
    w, dt = 0.05, 0.02
    bias = 0.5 * w * w * dt * dt

    # c43 (Ishell = 3) must be routed to the engine-IHBE-2 branch
    assert int(_one_element(tmp_path, ishell=3)
               .shells.state["ihbe_mask"][0]) == 2

    raw = _deps_xx(3, tmp_path, force_off=True)
    assert raw == pytest.approx(+bias, rel=1e-3)

    got = _deps_xx(3, tmp_path)
    assert abs(got) < 1e-3 * bias, (
        f"c43 is not the exact cancellation: got {got:.6e}, bias {bias:.3e}")
    assert got != pytest.approx(-bias, rel=1e-3)          # not kind 1's 2x
    assert got != pytest.approx(+bias, rel=1e-3)          # not uncorrected


def test_ishell3_and_ishell1_are_genuinely_different_branches(tmp_path):
    """Ishell 1 (engine IHBE 1, kind 1) and Ishell 3 (engine IHBE 2,
    kind 3) take different cdefo3 branches and must NOT give the same
    answer on a roll: kind 1 leaves -w^2 dt/2, kind 3 cancels exactly."""
    w, dt = 0.05, 0.02
    one = _deps_xx(1, tmp_path)
    three = _deps_xx(3, tmp_path)
    assert one == pytest.approx(-0.5 * w * w * dt * dt, rel=1e-3)
    assert abs(three) < 1e-3 * 0.5 * w * w * dt * dt