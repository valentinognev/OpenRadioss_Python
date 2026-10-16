"""Task P4.1 — the ``r6def3`` spring kernel (``redef3``/``repla3``).

Fortran origin:
  engine/source/elements/spring/r6def3.F
  engine/source/elements/spring/redef3.F90
  engine/source/elements/spring/repla3.F
  engine/source/elements/spring/rforc3.F (the IGTYP == 25 call site, :785)
  starter/source/properties/spring/hm_read_prop25.F (GEO layout)

Scope note (P4.0 audit, ``w1d_routine_status.json``): upstream has no
``IGTYP == 6`` branch — ``rforc3.F`` dispatches ``R6DEF3`` from the
``IGTYP == 25`` (SPR_AXI) branch, and the starter ships no ``/PROP/TYPE6``
reader.  This module ports the ``r6def3`` *file* (axial channel) under the
plan's ``spring_type6`` name; the docstring there records the correction.
"""

import numpy as np
import pytest

from pyradioss.elements import spring_type6
from pyradioss.model.entities.elements import Property
from pyradioss.model.model import ElementGroup, Model


def _type6_group(k=1000.0, length=2.0, mass=2.0, c=0.0):
    """Two-node axial spring along +X: node 0 at origin, node 1 at ``length``."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    prop = Property(
        id=1, type=25, title="axi",
        params={"k": k, "c": c, "mass": mass},
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]},
    )
    spring_type6.init_group(group, model, None)
    return group, model


def _ctx(group, model, x=None, v=None, dt=0.0):
    nn = model.x0.shape[0]
    return {
        "x": np.array(model.x0, dtype=float) if x is None else np.array(x, dtype=float),
        "v": None if v is None else np.array(v, dtype=float),
        "dt": dt,
        "fint": np.zeros((nn, 3)),
    }


def test_type6_zero_stiffness_is_a_free_segment():
    group, model = _type6_group(k=0.0)
    v = np.zeros_like(model.x0)
    v[0, 0] = 1.0
    ctx = _ctx(group, model, v=v, dt=0.1)
    dt_elem = spring_type6.driver(group, model, ctx)
    assert np.abs(ctx["fint"]).max() == 0.0
    assert np.abs(group.state["t6_force"]).max() == 0.0
    assert np.abs(spring_type6.energy(group)).max() == 0.0
    assert dt_elem.shape == (1,)


def test_type6_linear_spring_matches_analytic():
    group, model = _type6_group(k=1000.0, length=2.0)
    x = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    ctx = _ctx(group, model, x=x)
    spring_type6.driver(group, model, ctx)
    # Tension F = k*dL = 500 N pulls the nodes together: node 0 is dragged
    # +X, node 1 is dragged -X.  Written in the redef3 strain form
    # k*L0*(dL/L0) so the XL0 normalisation is visible in the assertion.
    assert ctx["fint"][1, 0] == pytest.approx(-1000.0 * 2.0 * 0.5 / 2.0, rel=1e-12)
    assert ctx["fint"][0, 0] == pytest.approx(500.0, rel=1e-12)
    assert np.abs(ctx["fint"][0] + ctx["fint"][1]).max() == 0.0


def test_type6_tangent_is_the_axial_projector():
    group, model = _type6_group(k=1000.0, length=2.0)
    ke, edofs = spring_type6.tangent(group, model.x0)
    assert ke.shape == (1, 6, 6)
    assert edofs.tolist() == [[0, 1, 2, 6, 7, 8]]
    assert ke[0, 0, 0] == pytest.approx(1000.0, rel=1e-12)
    assert ke[0, 0, 3] == pytest.approx(-1000.0, rel=1e-12)
    assert ke[0, 3, 0] == pytest.approx(-1000.0, rel=1e-12)
    assert ke[0, 3, 3] == pytest.approx(1000.0, rel=1e-12)
    transverse = np.ix_([1, 2, 4, 5], [1, 2, 4, 5])
    assert np.abs(ke[0][transverse]).max() == 0.0


def test_type6_kgeo_vanishes_at_zero_force():
    group, model = _type6_group(k=1000.0, length=2.0)
    kg, edofs = spring_type6.kgeo(group, model.x0)
    assert kg.shape == (1, 6, 6)
    assert np.abs(kg).max() == 0.0
    assert edofs.tolist() == [[0, 1, 2, 6, 7, 8]]


def test_type6_energy_is_the_trapezoidal_work():
    group, model = _type6_group(k=1000.0, length=2.0)
    x = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    ctx = _ctx(group, model, x=x)
    spring_type6.driver(group, model, ctx)
    # First step from rest: E = (F_old + F)/2 * dL = (0 + 500)/2 * 0.5.
    assert spring_type6.energy(group)[0] == pytest.approx(125.0, rel=1e-12)


def test_common_interp_matches_fortran_segments():
    from pyradioss.elements import spring_common

    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 10.0, 30.0])
    f, tg = spring_common.interp_curve(xs, ys, 0.5)
    assert f == pytest.approx(5.0, rel=1e-12)
    assert tg == pytest.approx(10.0, rel=1e-12)
    # Beyond the last abscissa INTERP returns F = 0, TG = 0 (interp.F:52).
    f, tg = spring_common.interp_curve(xs, ys, 3.0)
    assert f == 0.0 and tg == 0.0


def test_common_repla3_fires_only_for_coupled_plasticity():
    from pyradioss.elements import spring_common

    curve = (np.array([0.0, 1.0]), np.array([0.0, 100.0]))
    # IECROU != 3 passes through untouched (repla3.F guard).
    assert spring_common.repla3_correct(1000.0, 0.5, curve, 0, 7, 0, 1.0) == pytest.approx(0.5)
    # IECROU == 3: FP = 50*1000/(1000-100), dpx += 1.0/FP.
    got = spring_common.repla3_correct(1000.0, 0.5, curve, 3, 7, 0, 1.0)
    assert got == pytest.approx(0.5 + 900.0 / (50.0 * 1000.0), rel=1e-12)


def test_common_channel_refuses_unfactored_hardening_loudly():
    from pyradioss.elements import spring_common

    with pytest.raises(NotImplementedError, match="IECROU=1"):
        spring_common.redef3_channel(0.1, 0.0, xk=1000.0, iecrou=1, ifunc=5)


def test_type6_nonlinear_property_is_refused_not_linearised():
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model._id2idx = {1: 0, 2: 1}
    model.x0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prop = Property(
        id=1, type=25, title="axi-curve",
        params={"k": 1000.0, "mass": 2.0,
                "tension": {"stiff": 1000.0, "fun_a": 7, "hflag": 1}},
    )
    group = ElementGroup(
        ids=np.array([1]),
        conn=np.array([[0, 1]], dtype=np.int64),
        part=np.array([0]),
        state={"slices": [(slice(0, 1), None, prop)]},
    )
    with pytest.raises(NotImplementedError) as excinfo:
        spring_type6.init_group(group, model, None)
    # Seed-independent: the property carries two live nonlinear keys
    # (fun_a=7, hflag=1) and _refuse_nonlinear iterates a frozenset, so
    # either may be named first depending on PYTHONHASHSEED.  Either names
    # a live nonlinear key on the loud-refusal path.
    msg = str(excinfo.value)
    assert "needs the nonlinear" in msg
    assert "'fun_a'" in msg or "'hflag'" in msg
