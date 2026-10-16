"""Task P4.4 — the TYPE32 pretensioner spring (``SPR_PRE``) physics.

Fortran origin:
  engine/source/elements/spring/ruser32.F (RUSER32 force laws, ITYP 1-4)
  engine/source/elements/spring/ruser32ke3.F (RUSER32KE3 tangent assembly)
  engine/source/elements/spring/ruser32mat3.F (RUSER32MAT3: KX = UVAR(4))
  engine/source/elements/spring/rforc3.F:1061 (the IGTYP == 32 branch)
  starter/source/properties/spring/hm_read_prop32.F (ITYP solve, D1, scales)

Acceptance (plan P4.4): the Starter no longer produces an ``InactiveProperty``
for TYPE32, and the pretensioner imposes its preload force at rest.
"""

import math

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring_pretensioner
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.keywords.properties import read_prop, read_prop_type32
from pyradioss.model.entities.elements import Property
from pyradioss.model.model import ElementGroup, Model

FIXED_BEGIN = (
    "/BEGIN\np44-type32\n      2021         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n")

#: c52 RD-V-0031 card 1 (ITYP1): Stif0=2000, F1=1000, D1=50 -> solved STIF1=20.
SPR_PRE_CARD1 = (
    "/PROP/SPR_PRE/1\nSpring pretentioner 1\n"
    "                1E-5                                       0"
    "         1\n"
    "                2000                1000                  50"
    "                   0                   0\n"
    "         0         0\n")

TYPE32_CARD1 = SPR_PRE_CARD1.replace("/PROP/SPR_PRE/1", "/PROP/TYPE32/1")


def _starter_prop(tmp_path, body):
    deck = tmp_path / "p_0000.rad"
    deck.write_text(FIXED_BEGIN + body + "/END\n")
    log, model = MessageLog(), Model()
    for b in read_deck(str(deck)):
        if b.parts and b.parts[0] == "PROP":
            read_prop(b, model, log)
    return model, log


def _type32_group(ityp=1, f1=1000.0, stif0=2000.0, stif1=20.0, d1=-50.0,
                  ilock=0, sens_id=0, length=0.02, mass=1e-5,
                  fct_id1=0, fct_id2=0):
    """Two-node axial pretensioner along +X with solved reader params."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    prop = Property(
        id=1, type=32, title="pre",
        params={"mass": mass, "sens_id": sens_id, "ilock": ilock,
                "stif0": stif0, "stiff0": stif0,
                "stif1": stif1, "stiff1": stif1,
                "f1": f1, "d1": d1, "e1": -0.5 * f1 * d1,
                "fct_id1": fct_id1, "fct_id2": fct_id2,
                "scale_t": 1.0, "scale_d": 1.0, "scale_f": 1.0,
                "ityp": ityp, "k": stif0 + stif1},
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]},
    )
    n = group.n
    group.state.update(
        L0=np.array([length]), mass=np.array([mass]), k=np.array([stif0 + stif1]),
        force=np.zeros(n), eint=np.zeros(n), off=np.ones(n),
    )
    spring_pretensioner.init_pretensioner_type32(
        group, model, None, np.array([0], dtype=np.int64))
    group.state["model"] = model
    return group, model


def _ctx(group, model, x=None, v=None, dt=0.0, t=0.0):
    nn = model.x0.shape[0]
    model.t = t
    return {
        "x": np.array(model.x0, dtype=float) if x is None else np.array(x, dtype=float),
        "v": None if v is None else np.array(v, dtype=float),
        "dt": dt,
        "fint": np.zeros((nn, 3)),
    }


@pytest.mark.parametrize("body", [SPR_PRE_CARD1, TYPE32_CARD1])
def test_type32_no_longer_refuses_at_the_starter(tmp_path, body):
    """P4.4: the Starter yields a live TYPE32 property, not InactiveProperty."""
    from pyradioss.input import prop_reader
    model, log = _starter_prop(tmp_path, body)
    assert not log.has_errors
    assert 1 in model.prop_type32s
    p = model.properties[1]
    assert not isinstance(p, prop_reader.InactiveProperty)
    assert not getattr(p, "inactive", False)
    assert p.type == 32
    assert p.params["ityp"] == 1
    assert p.params["d1"] == pytest.approx(-50.0)
    assert p.params["k"] == pytest.approx(2020.0)


def test_pretensioner_imposes_its_locked_length():
    """P4.4: at rest the pretensioner carries its preload force (FF > 0)."""
    group, model = _type32_group()
    ctx = _ctx(group, model, dt=0.0)
    dt_elem = spring_pretensioner.driver(group, model, ctx)
    assert dt_elem.shape == (1,)
    # FF = F1 + STIF1 * X = 1000 + 20 * 0 = 1000 N at rest along the axis.
    assert abs(ctx["fint"]).sum() > 0.0
    assert ctx["fint"][0, 0] == pytest.approx(1000.0, rel=1e-12)
    assert ctx["fint"][1, 0] == pytest.approx(-1000.0, rel=1e-12)


def test_driver_returns_the_explicit_critical_step():
    """dt_crit = sqrt(M / K) with K = STIF0 once the force law has run."""
    group, model = _type32_group()
    ctx = _ctx(group, model, dt=1e-6)
    dt_elem = spring_pretensioner.driver(group, model, ctx)
    assert dt_elem[0] == pytest.approx(math.sqrt(1e-5 / 2000.0), rel=1e-12)


def test_read_prop_type32_direct_matches_live_path(tmp_path):
    """The exported ``read_prop_type32`` transcribes hm_read_prop32.F itself:
    ITYP solve, D1 = -|D1|, inverted time/displacement scales, k = STIF0+STIF1."""
    deck = tmp_path / "p_0000.rad"
    deck.write_text(FIXED_BEGIN + SPR_PRE_CARD1 + "/END\n")
    log, model = MessageLog(), Model()
    blocks = [b for b in read_deck(str(deck)) if b.parts and b.parts[0] == "PROP"]
    assert len(blocks) == 1
    read_prop_type32(blocks[0], model, log)
    assert not log.has_errors
    p = model.properties[1]
    assert p.type == 32
    assert p.params["ityp"] == 1
    assert p.params["stif0"] == pytest.approx(2000.0)
    assert p.params["f1"] == pytest.approx(1000.0)
    assert p.params["d1"] == pytest.approx(-50.0)
    assert p.params["stif1"] == pytest.approx(20.0)
    assert p.params["k"] == pytest.approx(2020.0)
    assert 1 in model.prop_type32s


class _Curve:
    def __init__(self, fn):
        self._fn = fn

    def eval(self, x):
        return self._fn(x)


def test_driver_ityp2_follows_the_displacement_curve():
    """ITYP2 (ruser32.F:203-219): FF = Scale_f * func1(X * Scale_d)."""
    group, model = _type32_group(ityp=2, fct_id1=7)
    model.functions[7] = _Curve(lambda x: 1000.0 * abs(x))
    v = np.zeros_like(model.x0)
    v[1, 0] = -1.0
    ctx = _ctx(group, model, v=v, dt=0.05)
    spring_pretensioner.driver(group, model, ctx)
    # X = -0.05 -> FF = 1000 * 0.05 = 50 N; elastic part is -2.5 N.
    assert ctx["fint"][0, 0] == pytest.approx(50.0, rel=1e-12)


def test_driver_ityp3_follows_the_time_curve():
    """ITYP3 (ruser32.F:220-230): F0 = Scale_f * func2(tacti * Scale_t)."""
    group, model = _type32_group(ityp=3, fct_id2=9)
    model.functions[9] = _Curve(lambda t: min(300.0, 30000.0 * t))
    ctx = _ctx(group, model, dt=0.005, t=0.005)
    spring_pretensioner.driver(group, model, ctx)
    assert ctx["fint"][0, 0] == pytest.approx(150.0, rel=1e-12)


def test_ilock2_locks_when_external_tension_exceeds_pretension():
    """ILOCK == 2 (ruser32.F:191-194): FX > FF latches UVAR(3); FF not imposed."""
    group, model = _type32_group(ityp=1, f1=100.0, stif1=0.0, ilock=2)
    group.state["force"][0] = 200.0
    ctx = _ctx(group, model, dt=0.0)
    spring_pretensioner.driver(group, model, ctx)
    assert group.state["uvar3"][0] == 1.0
    assert ctx["fint"][0, 0] == pytest.approx(200.0, rel=1e-12)


def test_energy_uses_the_pre_update_force():
    """``r5def3.F`` books ``dE = F_old * Ldot * dt`` (left rule: R5DEF3 runs
    before RUSER32), not the trapezoidal average of old and new force."""
    group, model = _type32_group()
    assert group.state["eint"][0] == pytest.approx(0.0)
    v = np.zeros_like(model.x0)
    v[1, 0] = 1.0
    ctx = _ctx(group, model, v=v, dt=0.001)
    spring_pretensioner.driver(group, model, ctx)
    # First step from rest books no work (F_old = 0) even though the
    # pretension law imposes F = FF = 1000 + 20 * 0.001 by the end of it.
    # (RINI32's E(0) = 25000 here is not yet applied: pending spring.py
    # handoff, see the module docstring.)
    assert ctx["fint"][0, 0] == pytest.approx(1000.02, rel=1e-12)
    # Second identical step: F_old = 1000.02 now, dE = 1000.02 * 1 * 0.001.
    spring_pretensioner.driver(group, model, ctx)
    assert group.state["eint"][0] == pytest.approx(1.00002, rel=1e-12)


def test_rini32_initial_energy_matches_fortran_heads():
    """RINI32 (hm_read_prop32.F): EINT(0) per ITYP, as the oracle T01 shows."""
    assert spring_pretensioner._rini32_eint(1, 1000.0, -50.0, 2000.0, None, None, 1.0, 1.0) == pytest.approx(25000.0)
    xs = np.array([-50.0, -40.0, 0.0])
    ys = np.array([0.0, 700.0, 1000.0])
    assert spring_pretensioner._rini32_eint(2, 0.0, 0.0, 2000.0, (xs, ys), None, 1.0, 1.0) == pytest.approx(37500.0)
    tx = np.array([0.0, 0.4])
    ty = np.array([1000.0, 1000.0])
    assert spring_pretensioner._rini32_eint(3, 0.0, 0.0, 2000.0, None, (tx, ty), 1.0, 1.0) == pytest.approx(250.0)


def test_rini32_ityp1_linear_case_without_d1():
    """RINI32 ITYP1 with D1 == 0: EINT(0) = (F1/STIF0)*F1/2."""
    assert spring_pretensioner._rini32_eint(1, 1000.0, 0.0, 2000.0, None, None, 1.0, 1.0) == pytest.approx(250.0)


def test_zero_length_spring_is_a_starter_error():
    """RINI32 MSG406: a zero-length pretensioner is refused at init."""
    group, model = _type32_group(length=0.0)
    log = MessageLog()
    spring_pretensioner.init_pretensioner_type32(
        group, model, log, np.array([0], dtype=np.int64))
    assert log.has_errors
