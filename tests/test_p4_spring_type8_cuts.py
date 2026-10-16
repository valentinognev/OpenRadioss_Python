"""Task P4.3 — the eight documented TYPE8 cuts, tests first.

Fortran origin
--------------
* engine/source/elements/spring/r2def3.F (the 6 uncoupled DOF force laws,
  rupture DLIM/CRIT, ISRATE smoothing, REPLA3 coupling)
* engine/source/elements/spring/redef3.F90 (per-channel constitutive update:
  elastic curve, isotropic/decoupled hardening, damping close)
* engine/source/elements/spring/r2coor3.F (frame gather)
* engine/source/elements/spring/r2len3.F (dt claim — unchanged by every cut)
* engine/source/elements/spring/r2len3law135.F90 (LAW135-only, not a cut)
* engine/source/elements/spring/r8ke3.F + r8sumg3.F (implicit tangent)
* engine/source/elements/spring/r23sens3.F + r1sens3.F (sensor gating)
* starter/source/properties/spring/hm_read_prop08.F (reader: per-DOF
  FUN_A/HFLAG/FUN_B/FUN_C/FUN_D/MIN_RUP/MAX_RUP/A/B/D/F/E/scale/GF3 plus the
  trailing ISRATE/Asrate card, cfg prop_p8_spr_gene.cfg radioss2018)

The eight cuts (w1d_routine_status.json ``@cut`` rows, PORTING_GUIDE.md:67):

1. fct_IDji ......... per-DOF force-function curves override the closed form
2. hardening ......... Hi/IECROU elasto-plastic channel updates
3. rupture ........... DeltaMin/DeltaMax limits delete the element
4. rate .............. ISRATE/Fcut smooths the rupture criterion
5. sensor ............ r2sens3 activation/deactivation gating
6. skew .............. skew_ID frame (wired by M39; locked by a deck test here)
7. corotational ...... TYPE13 per-cycle frame (lives in spring_beam.py, outside
                      P4.3's files; characterised as xfail, not implemented)
8. implicit .......... r8sumg3 tangent assembly (builder only; spring.py wiring
                      is Task P4.8's)

Written first per the task (Step 1); seven fail, the skew deck test passes
(M39 pre-wired it) and the corotational test xfails (out-of-file live path).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.elements import spring, spring_common, spring_general
from pyradioss.engine.sensors import Sensors
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.keywords import parse_starter_deck
from pyradioss.model.entities import Property
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter.initialization import build_element_groups, resolve_skews


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _type8_group(params, x0_nodes=None):
    """One 2-node TYPE8 element with ``params``; inits via the real driver."""
    if x0_nodes is None:
        x0_nodes = np.array([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
    m = Model()
    m.add_nodes(np.array([1, 2]), np.asarray(x0_nodes, dtype=float))
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]),
                     part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), None,
                          Property(id=1, type=8, params=dict(params)))]
    log = MessageLog()
    spring.init_group(g, m, log)
    assert not log.errors
    return m, g


def _forces(g, m, x, dt=1e-4):
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(g, np.asarray(x, dtype=float), None, None, dt, fint, mint)
    return fint, mint


# ---------------------------------------------------------------------------
# cut 1 — force-function curve override (fct_IDji)
# ---------------------------------------------------------------------------
def test_type8_force_function_uses_the_requested_curve():
    """fct_IDji overrides the closed form with an /FUNCT curve (finter.F)."""
    curve = FunctTable(7, [0.0, 1.0], [0.0, 100.0])
    kw = dict(mass=2.0, inertia=1.0, skew_id=0, k1=1000.0, fun_a1=7)
    m, g = _type8_group(kw)
    m.functions[7] = curve
    # re-init so the init-time curve resolution sees the function
    spring.init_group(g, m, MessageLog())

    fint, _ = _forces(g, m, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(50.0, rel=1e-12)

    # and: with fct_IDji=0 the closed form is used instead
    m0, g0 = _type8_group(dict(mass=2.0, inertia=1.0, skew_id=0, k1=1000.0))
    fint0, _ = _forces(g0, m0, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert fint0[0, 0] == pytest.approx(500.0, rel=1e-12)
    assert fint[0, 0] != pytest.approx(fint0[0, 0])


# ---------------------------------------------------------------------------
# cut 2 — hardening (Hi/IECROU)
# ---------------------------------------------------------------------------
def test_type8_isotropic_hardening_caps_at_the_curve():
    """IECROU=1: loading follows the curve past yield, unloading is elastic."""
    curve = FunctTable(7, [0.0, 0.1, 1.0], [0.0, 100.0, 100.0])
    kw = dict(mass=2.0, inertia=1.0, skew_id=0, k1=10000.0,
              fun_a1=7, hflag1=1)
    m, g = _type8_group(kw)
    m.functions[7] = curve
    spring.init_group(g, m, MessageLog())

    # load far past the knee in one step: capped at the curve, plastic flows
    fint, _ = _forces(g, m, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(100.0, rel=1e-9)
    dpx = g.state["gen6"]["t8"]["dpx"][0, 0]
    assert dpx == pytest.approx(0.49, rel=1e-9)

    # partial unload follows the elastic slope down from the cap
    fint, _ = _forces(g, m, [[0.0, 0.0, 0.0], [0.45, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(100.0 - 10000.0 * 0.05, rel=1e-9)

    # decoupled tension/compression (IECROU=2): cap each side, zero gap
    kw2 = dict(mass=2.0, inertia=1.0, skew_id=0, k1=10000.0,
               fun_a1=7, hflag1=2)
    m2, g2 = _type8_group(kw2)
    m2.functions[7] = FunctTable(7, [0.0, 0.1, 1.0], [0.0, 100.0, 100.0])
    spring.init_group(g2, m2, MessageLog())
    fint, _ = _forces(g2, m2, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(100.0, rel=1e-9)
    fint, _ = _forces(g2, m2, [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(0.0, abs=1e-12)
    fint, _ = _forces(g2, m2, [[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(-100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# cut 3 — rupture (DeltaMin/DeltaMax)
# ---------------------------------------------------------------------------
def test_type8_rupture_deletes_the_element_beyond_its_limits():
    """IFAIL2=0 displacement limits: CRIT>=1 deletes from the next cycle."""
    kw = dict(mass=2.0, inertia=1.0, skew_id=0, k1=1000.0,
              ifail=0, ifail2=0, min_rup1=-0.05, max_rup1=0.05)
    m, g = _type8_group(kw)

    # the exceedance step still carries its force (r2def3 computes first,
    # deletes after); the element flags dead
    fint, _ = _forces(g, m, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert fint[0, 0] == pytest.approx(100.0, rel=1e-12)
    assert g.state["gen6"]["t8"]["off"][0] == pytest.approx(0.0)

    # stays dead even back inside the limits
    fint, _ = _forces(g, m, [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    np.testing.assert_allclose(fint, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# cut 4 — rate smoothing (ISRATE/Fcut)
# ---------------------------------------------------------------------------
def test_type8_rate_smoothing_lags_the_rupture_criterion():
    """ISRATE=1: CRITNEW trails CRIT through the Fcut low-pass (r2def3)."""
    dt = 1e-3
    fcut = 10.0
    alpha = (2.0 * math.pi * fcut * dt) / (1.0 + 2.0 * math.pi * fcut * dt)
    kw = dict(mass=2.0, inertia=1.0, skew_id=0, k1=1000.0,
              ifail=0, ifail2=0, min_rup1=-0.05, max_rup1=0.05,
              israte=1, asrate=fcut)
    m, g = _type8_group(kw)
    fint, _ = _forces(g, m, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dt=dt)
    # CRIT = 2 would kill the element unsmoothed; smoothed it survives.
    # Fortran caps CRIT at 1+EM3 before the low-pass, so critnew trails
    # alpha * 1.001, not alpha * 2.
    assert g.state["gen6"]["t8"]["off"][0] == pytest.approx(1.0)
    assert g.state["gen6"]["t8"]["critnew"][0] == pytest.approx(1.001 * alpha)
    assert abs(fint).max() > 0.0

    # control: ISRATE=0 dies on the same step (force stands that step,
    # silence from the next, exactly like the rupture test above)
    kw0 = dict(kw, israte=0)
    m0, g0 = _type8_group(kw0)
    fint0, _ = _forces(g0, m0, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dt=dt)
    assert g0.state["gen6"]["t8"]["off"][0] == pytest.approx(0.0)
    fint0, _ = _forces(g0, m0, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dt=dt)
    np.testing.assert_allclose(fint0, 0.0, atol=1e-15)


def test_type8_reader_parses_the_trailing_israte_card(tmp_path: Path):
    """The /PROP/TYPE8 trailing card carries ISRATE + Asrate (hm_read_prop08)."""
    deck_text = """\
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
/PART/1
p1
         1         0         0
/PROP/TYPE8/1
gene
2.0 1.0 0 0 0 0 0 0
500.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
500.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
500.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
1 10.0
/SPRING/1
         1         1         2
/END
"""
    p = tmp_path / "T_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(p)), model, log)
    assert not log.errors
    prop = model.properties[1]
    assert prop.params["israte"] == 1
    assert prop.params["asrate"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# cut 5 — sensor activation (r2sens3)
# ---------------------------------------------------------------------------
def test_type8_sensor_gates_activation_and_deactivation():
    """isflag=0: dead until the sensor fires; isflag=1: dead after it fires."""
    kw = dict(mass=2.0, inertia=1.0, skew_id=0, k1=1000.0,
              sens_id=5, isflag=0)
    m, g = _type8_group(kw)
    x = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]

    # no sensor registry: never-fired, the spring stays inactive
    fint, _ = _forces(g, m, x)
    np.testing.assert_allclose(fint, 0.0, atol=1e-15)

    # sensor fires: the spring activates
    m.sensors_state = Sensors(m, MessageLog())
    m.sensors_state.status[5] = True
    fint, _ = _forces(g, m, x)
    assert fint[0, 0] == pytest.approx(100.0, rel=1e-12)

    # isflag=1 (hm negates ISENS): alive until the sensor fires, then dead
    kw1 = dict(kw, isflag=1)
    m1, g1 = _type8_group(kw1)
    fint1, _ = _forces(g1, m1, x)
    assert fint1[0, 0] == pytest.approx(100.0, rel=1e-12)
    m1.sensors_state = Sensors(m1, MessageLog())
    m1.sensors_state.status[5] = True
    fint1, _ = _forces(g1, m1, x)
    np.testing.assert_allclose(fint1, 0.0, atol=1e-15)

    # isflag=2 reactivates by re-referencing the deformed length (r2sens3:
    # X0 += DX, totals zeroed): the activation step carries no force,
    # afterwards the spring works about the new reference
    kw2 = dict(kw, isflag=2)
    m2, g2 = _type8_group(kw2)
    fint2, _ = _forces(g2, m2, x)
    np.testing.assert_allclose(fint2, 0.0, atol=1e-15)
    m2.sensors_state = Sensors(m2, MessageLog())
    m2.sensors_state.status[5] = True
    fint2, _ = _forces(g2, m2, x)
    np.testing.assert_allclose(fint2, 0.0, atol=1e-12)
    np.testing.assert_allclose(g2.state["gen6"]["L0"][0], [0.1, 0.0, 0.0],
                               rtol=1e-12)
    fint2, _ = _forces(g2, m2, [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    assert fint2[0, 0] == pytest.approx(50.0, rel=1e-9)


# ---------------------------------------------------------------------------
# cut 6 — skew frame (wired by M39; this test locks the deck path)
# ---------------------------------------------------------------------------
def test_type8_skew_frame_end_to_end_from_deck(tmp_path: Path):
    """/SKEW/FIX + /PROP/TYPE8 skew_ID: the DOFs ride the skew axes."""
    deck_text = """\
/NODE
         1                 0.0                 0.0                 0.0
         2                 0.0                 0.0                 0.0
/SKEW/FIX/5
skew five
0.0 0.0 0.0
-1.0 0.0 0.0
0.0 0.0 1.0
/PART/1
p1
         1         0         0
/PROP/TYPE8/1
gene
2.0 1.0 5 0 0 0 0 0
500.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
100.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
100.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0
0 0 0 0 0
0.0 0.0 1.0 1.0
/SPRING/1
         1         1         2
/END
"""
    p = tmp_path / "T_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(p)), model, log)
    assert not log.errors
    resolve_skews(model, log)
    assert not log.errors
    build_element_groups(model, log)
    assert not log.errors
    grp = model.springs
    spring.init_group(grp, model, log)
    assert not log.errors

    # stretch along global Y = the skew's local X' (k1 = 500)
    x = model.x0.copy()
    x[1] = np.array([0.0, 0.2, 0.0])
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    spring.forces(grp, x, None, None, 1e-4, fint, mint)
    np.testing.assert_allclose(fint[0], [0.0, 100.0, 0.0], rtol=1e-9)


# ---------------------------------------------------------------------------
# cut 7 — co-rotational beam frame (out of P4.3's files; characterised only)
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=False,
    reason="TYPE13's live path is spring_beam.py (outside P4.3's file list): "
           "it resolves the deformation on the initial frame, so a rigid "
           "rotation reads as spurious strain (probe: [-220,-1000,0] vs "
           "[0,100,0]). Tracked for the TYPE13 owner.",
)
def test_type13_corotational_frame_follows_large_rotation():
    """A rigidly rotated TYPE13 element carries no force (r4def3/r4evec3)."""
    from pyradioss.model.entities import Property as _Property
    prop = _Property(id=1, type=13, params={
        "mass": 2.0, "inertia": 1.0, "skew_id": 0,
        "k1": 1000.0, "k2": 200.0, "k3": 300.0,
    })
    m = Model()
    m.add_nodes(np.array([1, 2]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]),
                     part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), None, prop)]
    m.springs = g
    spring.init_group(g, m, MessageLog())

    th = math.pi / 2.0
    rot = np.array([[math.cos(th), -math.sin(th), 0.0],
                    [math.sin(th), math.cos(th), 0.0],
                    [0.0, 0.0, 1.0]])
    x = np.array([[0.0, 0.0, 0.0], rot @ np.array([1.1, 0.0, 0.0])])
    fint = np.zeros((2, 3))
    spring.forces(g, x, None, None, 1e-4, fint, np.zeros((2, 3)))
    # 0.1 of stretch along the rotated axis at k1 = 1000
    np.testing.assert_allclose(fint[0], [0.0, 100.0, 0.0], rtol=1e-9)


# ---------------------------------------------------------------------------
# cut 8 — implicit tangent (r8ke3/r8sumg3 builder; P4.8 wires it up)
# ---------------------------------------------------------------------------
def test_type8_implicit_tangent_matches_the_analytic_blocks():
    """KE11 = Q diag(K) Q', KE12 = -KE11, KE22 = KE11; ruptured rows vanish."""
    k = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    eye = (np.array([[1.0, 0.0, 0.0]]),
           np.array([[0.0, 1.0, 0.0]]),
           np.array([[0.0, 0.0, 1.0]]))
    ke = spring_common.type8_tangent(k, *eye, np.ones(1))
    assert ke.shape == (1, 12, 12)
    np.testing.assert_allclose(ke[0, :3, :3], np.diag([1.0, 2.0, 3.0]),
                               rtol=1e-15)
    np.testing.assert_allclose(ke[0, 3:6, 3:6], np.diag([4.0, 5.0, 6.0]),
                               rtol=1e-15)
    np.testing.assert_allclose(ke[0, :6, 6:], -ke[0, :6, :6], rtol=0, atol=0)
    np.testing.assert_allclose(ke[0, 6:, 6:], ke[0, :6, :6], rtol=0, atol=0)
    np.testing.assert_allclose(ke[0, :3, 3:], 0.0, atol=0)

    # a 90-degree frame moves local-X stiffness onto global Y
    rot = (np.array([[0.0, 1.0, 0.0]]),
           np.array([[-1.0, 0.0, 0.0]]),
           np.array([[0.0, 0.0, 1.0]]))
    ke_r = spring_common.type8_tangent(k, *rot, np.ones(1))
    assert ke_r[0, 1, 1] == pytest.approx(1.0)
    assert ke_r[0, 0, 0] == pytest.approx(2.0)

    # a ruptured element contributes no stiffness
    ke_dead = spring_common.type8_tangent(k, *eye, np.zeros(1))
    np.testing.assert_allclose(ke_dead, 0.0, atol=0)
