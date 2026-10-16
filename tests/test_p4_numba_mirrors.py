"""Task P4.9 — numba mirrors for the 1-D element-iterating branches.

Fortran origin: none (this task adds no physics). Each mirror below
reproduces, statement for statement, the per-element math of one
vectorised branch driver from Tasks P4.1–P4.5 (spring) or the beam
driver the plan names (beam_type3):

* ``spring_linear`` <-> ``elements/spring_type6.driver`` (P4.1, linear
  ``redef3`` channel + damping close + ``r2len3`` dt close);
* ``spring_t32_linear`` <-> ``elements/spring_pretensioner.
  forces_pretensioner_type32`` (P4.4) on the curve-free domain
  (sensor-inactive elements and ``ITYP == 1``);
* ``spring_t8_linear`` <-> ``elements/spring_common.type8_forces``
  (P4.3) on the closed-form domain (every DOF ``fun_a == 0`` or with
  no ``/FUNCT`` table — the linear ``k dx + c rate`` path);
* ``beam_type3_elastic`` <-> ``elements/beam_type3._forces_core``
  (the plan's ``beam_type3`` case) on the elastic domain (``LAW != 2``).

Branches that iterate over *curves* instead of *elements* get no
mirror by explicit plan decision (they are lookup-bound, not NumPy
overhead-bound): ``spring_python_funct`` evaluators (P4.5, iteration
over curve abscissae), the pretensioner ``ITYP 2/3/4`` curve laws,
TYPE8 nonlinear DOFs (``IECROU``/force-function tables), the user
spring callback (P4.2 — an arbitrary Python callable numba cannot
compile), and the beam LAW2 radial return (per-slice Python material
dispatch). ``test_only_element_branches_have_mirrors`` locks that
exclusion: the mirror modules must export exactly the four kernels
above and nothing else.

Parity contract (same as M7): the mirrors run the same formulas in
the same order, so element-wise expressions agree BITWISE with the
NumPy drivers (``tobytes()`` equality, not a tolerance) — except the
beam's global back-transform, which NumPy's einsum contracts with
SIMD unrolling (measured ``(t0+t2)+t1``, unreproducible by any
portable scalar order) and is therefore proven at the M7 kernel
tolerance (``rtol=1e-12``; measured ``<= 5e-16``). The drivers
themselves are untouched — the reference side of each comparison is
the real driver, run on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.accel.jit_kernels import beam as beam_mir
from pyradioss.accel.jit_kernels import spring as spring_mir
from pyradioss.common.messages import MessageLog
from pyradioss.elements import beam_type3, spring, spring_type6
from pyradioss.model.entities import Property
from pyradioss.model.model import ElementGroup, Model


def test_only_element_branches_have_mirrors():
    """Curve-iterating branches must not gain a mirror (plan P4.9)."""
    assert list(spring_mir.__all__) == [
        "spring_linear", "spring_t32_linear", "spring_t8_linear",
    ]
    assert list(beam_mir.__all__) == ["beam_type3_elastic"]


# ---------------------------------------------------------------------------
# spring_linear <-> spring_type6.driver (P4.1)
# ---------------------------------------------------------------------------

def _t6_group(n, rng):
    """N-element TYPE6 group with randomised linear properties."""
    nn = 2 * n
    x0 = np.zeros((nn, 3))
    x0[0::2, 0] = 0.0
    lengths = rng.uniform(0.5, 3.0, size=n)
    x0[1::2, 0] = lengths
    x0[1::2, 1] = rng.uniform(-0.5, 0.5, size=n)
    model = Model()
    model.node_ids = np.arange(1, nn + 1, dtype=np.int64)
    model._id2idx = {int(v): int(i) for i, v in enumerate(model.node_ids)}
    model.x0 = np.array(x0, dtype=float)
    conn = np.stack([np.arange(0, nn, 2), np.arange(1, nn, 2)], axis=1)
    k = rng.uniform(0.0, 5000.0, size=n)
    k[rng.random(n) < 0.1] = 0.0              # free segments included
    c = rng.uniform(0.0, 50.0, size=n)
    mass = rng.uniform(0.5, 5.0, size=n)
    ak = rng.uniform(0.5, 2.0, size=n)
    ileng = rng.integers(0, 2, size=n).astype(np.int64)
    props = []
    for i in range(n):
        props.append(Property(
            id=i + 1, type=25, title="axi",
            params={"k": float(k[i]), "c": float(c[i]),
                    "mass": float(mass[i]), "a_scale": float(ak[i]),
                    "ileng": int(ileng[i])},
        ))
    group = ElementGroup(
        ids=np.arange(1, n + 1, dtype=np.int64), conn=conn,
        part=np.zeros(n, dtype=np.int64),
        state={"slices": [(slice(i, i + 1), None, props[i])
                          for i in range(n)]},
    )
    spring_type6.init_group(group, model, None)
    group.state["off"][0] = 0.0      # a switched-off element exercises
    return group, model               # the tangent pass-through path


def test_1d_mirror_is_bitwise_spring_linear():
    rng = np.random.default_rng(40901)
    n = 37
    group, model = _t6_group(n, rng)
    x = np.array(model.x0, dtype=float)
    x[1::2, 0] += rng.uniform(-0.4, 0.6, size=n)   # stretch/compress
    x[1::2, 2] += rng.uniform(-0.3, 0.3, size=n)
    v = rng.standard_normal((2 * n, 3)) * 0.5
    dt = 1e-3
    ctx = {"x": x, "v": v, "dt": dt, "fint": np.zeros_like(x)}
    # multi-step: the mirror must track dx_old/F_old state exactly
    for _ in range(3):
        st = group.state
        F_old = st["t6_force"].copy()
        dx_old = st["t6_dx_old"].copy()
        eint_old = st["t6_eint"].copy()
        k_tan_old = st["t6_k_tan"].copy()
        dt_ref = spring_type6.driver(group, model, ctx)
        F_ref = st["t6_force"].copy()
        eint_ref = st["t6_eint"].copy()
        k_tan_ref = st["t6_k_tan"].copy()
        fint_ref = ctx["fint"].copy()
        # restore pre-driver state, run the mirror on identical inputs
        st["t6_force"][:] = F_old
        st["t6_dx_old"][:] = dx_old
        st["t6_eint"][:] = eint_old
        st["t6_k_tan"][:] = k_tan_old
        ctx["fint"][:] = 0.0
        conn = group.conn
        F, a, e_incr, k_tan, dt_elem = spring_mir.spring_linear(
            x[conn[:, 0]], x[conn[:, 1]], st["t6_L0"], dx_old,
            st["t6_k"], st["t6_c"], st["t6_ak"], st["t6_ileng"],
            F_old, dt, st["off"], st["t6_mass"], k_tan_old,
        )
        assert F.tobytes() == F_ref.tobytes()
        assert k_tan.tobytes() == k_tan_ref.tobytes()
        assert dt_elem.tobytes() == dt_ref.tobytes()
        assert (eint_old + e_incr).tobytes() == eint_ref.tobytes()
        fint_mir = np.zeros_like(x)
        np.add.at(fint_mir, conn[:, 0], F[:, None] * a)
        np.add.at(fint_mir, conn[:, 1], -F[:, None] * a)
        assert fint_mir.tobytes() == fint_ref.tobytes()
        # advance the driver state for the next step
        st["t6_force"][:] = F_ref
        st["t6_eint"][:] = eint_ref
        st["t6_k_tan"][:] = k_tan_ref
        x[1::2, 0] += rng.uniform(-0.1, 0.1, size=n)


# ---------------------------------------------------------------------------
# spring_t32_linear <-> pretensioner ITYP1 (P4.4)
# ---------------------------------------------------------------------------

def _t32_group(n, rng):
    from types import SimpleNamespace

    nn = 2 * n
    x0 = np.zeros((nn, 3))
    x0[1::2, 0] = rng.uniform(0.8, 1.5, size=n)
    conn = np.stack([np.arange(0, nn, 2), np.arange(1, nn, 2)], axis=1)
    group = ElementGroup(
        ids=np.arange(1, n + 1, dtype=np.int64), conn=conn,
        part=np.zeros(n, dtype=np.int64), state={},
    )
    st = group.state
    idx = np.arange(n)
    st["off"] = np.ones(n)
    st["off"][rng.random(n) < 0.15] = 0.0     # deleted elements included
    st["force"] = rng.uniform(-5.0, 5.0, size=n)
    st["eint"] = rng.uniform(0.0, 2.0, size=n)
    st["k"] = np.zeros(n)
    st["mass"] = rng.uniform(0.5, 3.0, size=n)
    st["stif0"] = rng.uniform(100.0, 2000.0, size=n)
    st["stif1"] = rng.uniform(0.0, 500.0, size=n)
    st["ityp"] = np.ones(n, dtype=np.int64)
    st["f1"] = rng.uniform(0.0, 50.0, size=n)
    st["d1"] = np.zeros(n)
    st["scale_t"] = np.ones(n)
    st["scale_d"] = np.ones(n)
    st["scale_f"] = np.ones(n)
    st["sens_id"] = np.zeros(n, dtype=np.int64)
    st["fct_id1"] = np.zeros(n, dtype=np.int64)
    st["fct_id2"] = np.zeros(n, dtype=np.int64)
    st["ilock"] = (rng.random(n) < 0.5).astype(np.int64) * 2
    st["uvar1"] = rng.uniform(-0.05, 0.2, size=n)
    st["uvar2"] = np.ones(n)                  # already activated
    st["uvar2"][rng.random(n) < 0.3] = 0.0    # some just-armed
    st["uvar3"] = np.zeros(n)
    st["uvar4"] = st["stif0"].copy()
    st["model"] = SimpleNamespace(t=1.0, sensors_state=None,
                                  functions={})
    return group, x0


def test_1d_mirror_is_bitwise_spring_t32_linear():
    from pyradioss.elements import spring_pretensioner

    rng = np.random.default_rng(40904)
    n = 29
    group, x0 = _t32_group(n, rng)
    st = group.state
    idx = np.arange(n)
    x = np.array(x0)
    x[1::2, 0] += rng.uniform(-0.1, 0.2, size=n)
    v = rng.standard_normal((2 * n, 3)) * 0.2
    dt = 1e-3
    fint = np.zeros_like(x)
    snapshot = {k: np.array(st[k], copy=True)
                for k in ("force", "eint", "k", "uvar1", "uvar2",
                          "uvar3", "uvar4")}
    dt_ref = spring_pretensioner.forces_pretensioner_type32(
        group, x, v, dt, fint, idx)
    ref = {k: np.array(st[k], copy=True)
           for k in ("force", "eint", "k", "uvar1", "uvar2", "uvar3")}
    # restore, run the mirror (all sensors inactive-armed -> iact True)
    for k, vals in snapshot.items():
        st[k][:] = vals
    iact = np.ones(n, dtype=bool)
    F, u1, u2, u3, k_tan, de_incr, dt_elem = spring_mir.spring_t32_linear(
        x[idx * 2], x[idx * 2 + 1], v[idx * 2], v[idx * 2 + 1],
        snapshot["force"], st["stif0"], st["stif1"], st["f1"], st["d1"],
        st["ilock"], st["ityp"], iact, snapshot["uvar1"],
        snapshot["uvar2"], snapshot["uvar3"], dt, st["off"], st["mass"],
    )
    assert F.tobytes() == ref["force"].tobytes()
    assert u1.tobytes() == ref["uvar1"].tobytes()
    assert u2.tobytes() == ref["uvar2"].tobytes()
    assert u3.tobytes() == ref["uvar3"].tobytes()
    assert k_tan.tobytes() == ref["k"].tobytes()
    assert (snapshot["eint"] + de_incr).tobytes() == ref["eint"].tobytes()
    assert dt_elem.tobytes() == dt_ref.tobytes()


def test_t32_mirror_refuses_curve_ityps_loudly():
    n = 3
    z = np.zeros((n, 3))
    o = np.zeros(n)
    i1 = np.ones(n, dtype=np.int64)
    for ityp in (2, 3, 4):
        with pytest.raises(ValueError, match="ITYP"):
            spring_mir.spring_t32_linear(
                z, z, z, z, o, o, o, o, o, i1, np.full(n, ityp),
                np.ones(n, dtype=bool), o, np.ones(n), o, 1e-3,
                np.ones(n), np.ones(n),
            )


# ---------------------------------------------------------------------------
# spring_t8_linear <-> type8_forces closed-form path (P4.3)
# ---------------------------------------------------------------------------

def _t8_group(n, rng):
    """N-element all-linear TYPE8 group built through the real reader."""
    x0_nodes = np.zeros((2 * n, 3))
    x0_nodes[1::2, 0] = rng.uniform(0.5, 2.0, size=n)
    x0_nodes[1::2, 1] = rng.uniform(-0.5, 0.5, size=n)
    m = Model()
    m.add_nodes(np.arange(1, 2 * n + 1, dtype=np.int64), x0_nodes)
    conn = np.stack([np.arange(0, 2 * n, 2),
                     np.arange(1, 2 * n, 2)], axis=1)
    g = ElementGroup(ids=np.arange(1, n + 1), conn=conn,
                     part=np.zeros(n, dtype=np.int64))
    params = dict(mass=2.0, inertia=1.0, skew_id=0)
    for d in range(1, 7):
        params[f"k{d}"] = float(rng.uniform(100.0, 5000.0))
        params[f"c{d}"] = float(rng.uniform(0.0, 20.0))
    if rng.random() < 0.5:                    # rupture limits, sometimes
        params["min_rup1"] = -0.5
        params["max_rup1"] = 0.5
    g.state["slices"] = [(slice(0, n), None,
                           Property(id=1, type=8, params=dict(params)))]
    log = MessageLog()
    spring.init_group(g, m, log)
    assert not log.errors
    return m, g


def test_1d_mirror_is_bitwise_spring_t8_linear():
    rng = np.random.default_rng(40903)
    n = 23
    m, g = _t8_group(n, rng)
    x = np.array(m.x0, dtype=float)
    x[1::2, 0] += rng.uniform(-0.2, 0.3, size=n)
    x[1::2, 1] += rng.uniform(-0.2, 0.2, size=n)
    dt = 1e-4
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    gen = g.state["gen6"]
    t8 = gen["t8"]
    snap = {k: np.array(t8[k], copy=True)
            for k in ("dx_old", "f_old", "k_tan", "e6", "off", "crit",
                      "critnew", "dead")}
    snap["theta"] = np.array(gen["theta"], copy=True)
    snap["eint"] = np.array(gen["eint"], copy=True)
    snap["force"] = np.array(gen["force"], copy=True)
    snap["moment"] = np.array(gen["moment"], copy=True)
    dt_first = spring.forces(g, x, None, None, dt, fint, mint)
    ref = {k: np.array(t8[k], copy=True)
           for k in ("dx_old", "f_old", "k_tan", "e6", "off", "crit",
                     "critnew", "dead")}
    ref["theta"] = np.array(gen["theta"], copy=True)
    ref["eint"] = np.array(gen["eint"], copy=True)
    ref["F"] = np.array(gen["force"], copy=True)
    ref["M"] = np.array(gen["moment"], copy=True)
    ref["fint"] = np.array(fint, copy=True)
    ref["mint"] = np.array(mint, copy=True)
    fint2 = np.zeros_like(x)
    mint2 = np.zeros_like(x)
    assert spring.forces(g, x, None, None, dt, fint2,
                         mint2).tobytes() == dt_first.tobytes()
    ref["dt"] = dt_first
    # restore pre-driver state, gather the mirror inputs
    for k in snap:
        if k in t8:
            t8[k][:] = snap[k]
        elif k == "theta":
            gen["theta"][:] = snap[k]
    conn = gen["conn"]
    n1, n2 = conn[:, 0], conn[:, 1]
    d = x[n2] - x[n1]
    dv = np.zeros_like(d)
    delta = np.stack([np.einsum("mb,mb->m", d, gen["e1"]) - gen["L0"][:, 0],
                      np.einsum("mb,mb->m", d, gen["e2"]) - gen["L0"][:, 1],
                      np.einsum("mb,mb->m", d, gen["e3"]) - gen["L0"][:, 2]],
                     axis=1)
    dvl = np.zeros_like(delta)
    dwl = np.zeros_like(delta)
    nonlinear = np.zeros((n, 6), dtype=bool)
    (F, Mx, theta_new, deint, dx_old, f_old, k_tan, e6, off, dead,
     crit, critnew, L0_out, rerefd_out, dt_elem, fvec, m1,
     m2) = spring_mir.spring_t8_linear(
        delta, dvl, snap["theta"], dwl, dt, gen["e1"], gen["e2"],
        gen["e3"], d, gen["L0"], t8["xk"], t8["xc"], t8["ak"],
        nonlinear, t8["dmin"], t8["dmax"], t8["ifail"], t8["ifail2"],
        t8["israte"], t8["fcut"], snap["off"], np.zeros(n, dtype=bool),
        np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64), snap["dx_old"], snap["f_old"],
        snap["k_tan"], snap["e6"], snap["crit"], snap["critnew"],
        t8["rerefd"], snap["dead"], gen["mass"], gen["inertia"],
    )
    assert L0_out.tobytes() == gen["L0"].tobytes()
    assert rerefd_out.tobytes() == t8["rerefd"].tobytes()
    assert F.tobytes() == ref["F"].tobytes()
    assert Mx.tobytes() == ref["M"].tobytes()
    assert theta_new.tobytes() == ref["theta"].tobytes()
    assert dx_old.tobytes() == ref["dx_old"].tobytes()
    assert f_old.tobytes() == ref["f_old"].tobytes()
    assert k_tan.tobytes() == ref["k_tan"].tobytes()
    assert e6.tobytes() == ref["e6"].tobytes()
    assert off.tobytes() == ref["off"].tobytes()
    assert dead.tobytes() == ref["dead"].tobytes()
    assert crit.tobytes() == ref["crit"].tobytes()
    assert critnew.tobytes() == ref["critnew"].tobytes()
    assert (snap["eint"] + deint).tobytes() == ref["eint"].tobytes()
    assert dt_elem.tobytes() == ref["dt"].tobytes()
    fint_mir = np.zeros_like(x)
    mint_mir = np.zeros_like(x)
    np.add.at(fint_mir, n1, fvec)
    np.add.at(fint_mir, n2, -fvec)
    np.add.at(mint_mir, n1, m1)
    np.add.at(mint_mir, n2, m2)
    assert fint_mir.tobytes() == ref["fint"].tobytes()
    assert mint_mir.tobytes() == ref["mint"].tobytes()


def test_t8_mirror_refuses_nonlinear_dofs_loudly():
    n = 2
    z3 = np.zeros((n, 3))
    z6 = np.zeros((n, 6))
    o = np.zeros(n)
    i = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=bool)
    nonlinear = np.zeros((n, 6), dtype=bool)
    nonlinear[1, 0] = True
    with pytest.raises(ValueError, match="deck curve"):
        spring_mir.spring_t8_linear(
            z3, z3, z3, z3, 1e-4, z3, z3, z3, z3, z3, z6, z6, z6,
            nonlinear, z6, z6, i, i, i, o, o, b, i, i, i, z6, z6,
            z6, z6, o, o, b, b, o, o,
        )


# ---------------------------------------------------------------------------
# beam_type3_elastic <-> beam_type3._forces_core LAW1 path
# ---------------------------------------------------------------------------

def _beam_group(n, rng):
    """N-element LAW1 beam group (axis +X, orientation +Y)."""
    from types import SimpleNamespace

    nn = 3 * n
    x0 = np.zeros((nn, 3))
    for e in range(n):
        x0[3 * e] = [10.0 * e, 0.0, 0.0]
        x0[3 * e + 1] = [10.0 * e + rng.uniform(4.0, 12.0), 0.0, 0.0]
        x0[3 * e + 2] = [10.0 * e, 1.0, 0.0]
    model = Model()
    model.node_ids = np.arange(1, nn + 1, dtype=np.int64)
    model._id2idx = {int(v): int(j) for j, v in enumerate(model.node_ids)}
    model.x0 = np.array(x0, dtype=float)
    model.x = np.array(x0, dtype=float)
    conn = np.stack([np.arange(0, nn, 3), np.arange(1, nn, 3),
                     np.arange(2, nn, 3)], axis=1)
    area = rng.uniform(10.0, 40.0, size=n)
    iyy = rng.uniform(20.0, 80.0, size=n)
    izz = rng.uniform(20.0, 80.0, size=n)
    ixx = iyy + izz
    mat = SimpleNamespace(law=1, E=210.0, G=80.0, rho0=7.8e-6,
                          params={})
    prop = SimpleNamespace(area=0.0, iyy=0.0, izz=0.0, ixx=0.0, params={
        "area": 0.0, "iyy": 0.0, "izz": 0.0, "ixx": 0.0})
    group = ElementGroup(
        ids=np.arange(1, n + 1, dtype=np.int64), conn=conn,
        part=np.zeros(n, dtype=np.int64),
        state={"slices": [(slice(0, n), mat, prop)]},
    )
    # per-element properties live on the slice params in the real reader;
    # emulate with one slice per element
    group.state["slices"] = [
        (slice(e, e + 1), mat, SimpleNamespace(
            area=0.0, iyy=0.0, izz=0.0, ixx=0.0, params={
                "area": float(area[e]), "iyy": float(iyy[e]),
                "izz": float(izz[e]), "ixx": float(ixx[e])}))
        for e in range(n)
    ]
    log = MessageLog()
    beam_type3.init_group(group, model, log)
    assert not log.errors
    group.state["_area"] = area
    group.state["_ixx"] = ixx
    group.state["_iyy"] = iyy
    group.state["_izz"] = izz
    group.state["_E"] = np.full(n, 210.0)
    group.state["_G"] = np.full(n, 80.0)
    return group, model


def test_1d_mirror_is_bitwise_beam_type3():
    rng = np.random.default_rng(40902)
    n = 19
    group, model = _beam_group(n, rng)
    st = group.state
    x = np.array(model.x0, dtype=float)
    x[1::3] += rng.uniform(-0.3, 0.5, size=(n, 3))   # deformed geometry
    v = rng.standard_normal((3 * n, 3)) * 0.1
    vr = rng.standard_normal((3 * n, 3)) * 0.01
    dt = 1e-4
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    for _ in range(2):                              # stateful resultants
        fres_old = st["fres"].copy()
        mres_old = st["mres"].copy()
        eint_old = st["eint"].copy()
        fint[:] = 0.0
        mint[:] = 0.0
        dt_ref = beam_type3.forces(group, x, v, vr, dt, fint, mint)
        fres_ref = st["fres"].copy()
        mres_ref = st["mres"].copy()
        eint_ref = st["eint"].copy()
        fint_ref = fint.copy()
        mint_ref = mint.copy()
        st["fres"][:] = fres_old
        st["mres"][:] = mres_old
        st["eint"][:] = eint_old
        fint[:] = 0.0
        mint[:] = 0.0
        conn = group.conn
        (fg1, fg2, mg1, mg2, fres_new, mres_new, deint,
         dt_elem) = beam_mir.beam_type3_elastic(
            x[conn[:, 0]], x[conn[:, 1]], x[conn[:, 2]],
            v[conn[:, 0]], v[conn[:, 1]],
            vr[conn[:, 0]], vr[conn[:, 1]],
            st["_area"], st["_ixx"], st["_iyy"], st["_izz"],
            st["_E"], st["_G"], fres_old, mres_old, st["L0"],
            st["dt0"], st["off"], dt, np.zeros(n, dtype=bool),
        )
        assert fres_new.tobytes() == fres_ref.tobytes()
        assert mres_new.tobytes() == mres_ref.tobytes()
        assert (eint_old + deint).tobytes() == eint_ref.tobytes()
        assert dt_elem.tobytes() == dt_ref.tobytes()
        fint_mir = np.zeros_like(x)
        mint_mir = np.zeros_like(x)
        np.add.at(fint_mir, conn[:, 0], fg1)
        np.add.at(fint_mir, conn[:, 1], fg2)
        np.add.at(mint_mir, conn[:, 0], mg1)
        np.add.at(mint_mir, conn[:, 1], mg2)
        # The global back-transform (E @ local) is contracted by NumPy's
        # einsum with SIMD unrolling — measured (t0+t2)+t1 association,
        # i.e. NOT reproducible by any portable scalar order — so it gets
        # the M7 kernel tolerance (rtol=1e-12; measured <= 5e-16 here),
        # not tobytes. Everything element-local above IS bitwise.
        assert np.allclose(fint_mir, fint_ref, rtol=1e-12, atol=0.0)
        assert np.allclose(mint_mir, mint_ref, rtol=1e-12, atol=0.0)
        st["fres"][:] = fres_ref
        st["mres"][:] = mres_ref
        st["eint"][:] = eint_ref


def test_beam_mirror_refuses_plastic_law_loudly():
    n = 2
    z3 = np.zeros((n, 3))
    o = np.zeros(n)
    with pytest.raises(ValueError, match="LAW2"):
        beam_mir.beam_type3_elastic(
            z3, z3, z3, z3, z3, z3, z3, o, o, o, o, o, o, z3, z3,
            o, o, o, 1e-4, np.ones(n, dtype=bool),
        )
