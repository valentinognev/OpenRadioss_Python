"""Task P2.9 -- ``solid_q1np``: hexahedra on a NURBS surface.

WHAT IS TESTED, AND WHAT IS NOT CLAIMED
---------------------------------------
The first test is the plan's own (``plan/03_phase2_elements_solid.md`` Task
P2.9, Step 1), verbatim apart from line wrapping.  The rest pin the evaluator
against an independent reference (``scipy.interpolate.BSpline``), against its
own parent-coordinate twin, and against the identities the upstream shape
functions must satisfy.

``forces`` is tested at the *transcription* level only: a hand-built element
and a caller-supplied stress.  **Those tests say nothing about reachability.**
No deck ``pyradioss`` reads produces a Q1NP element -- upstream builds the
surface in a Starter chain (``q1np_generate_main`` from an ``/INTER`` ``Ists``
card) that is not ported -- and the last block of this file fails if anything
in ``pyradioss`` starts calling the module, or if the census/registry stop
saying ``ported-unreachable``.  See ``docs/PORT_EXTENSIONS.md``.

The expected Fortran-side facts below were read, not recalled, from
``$OR_SRC/engine/source/elements/solid/solid_q1np/q1np_nurbs_surface_eval_mod.F90``,
``q1np_forc3.F90`` and ``$OR_SRC/common_source/modules/q1np_geom_mod.F90``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from pyradioss.elements import solid_q1np

REPO = Path(__file__).resolve().parents[1]
FORC3 = "engine/source/elements/solid/solid_q1np/q1np_forc3.F90"


def test_bilinear_nurbs_surface_evaluates_and_differentiates():
    ctrl = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    knots = np.array([0, 0, 1, 1], float)
    P = solid_q1np.evaluate_surface(ctrl, knots, degree=(1, 1), u=0.25, v=0.5)
    assert np.allclose(P.x, 0.25)
    d = solid_q1np.evaluate_surface(
        ctrl, knots, degree=(1, 1), u=0.25, v=0.5, derivs=(1, 0)
    )
    assert np.allclose(d.dx_du, 1.0) and np.allclose(d.dy_du, 0.0)


# ---------------------------------------------------------------------------
# the evaluator against an independent reference
# ---------------------------------------------------------------------------

scipy_interp = pytest.importorskip("scipy.interpolate")

#: p = 2, q = 3 with one interior knot in u and two in v: a genuinely
#: multi-span patch, so the span search and the control-point window matter.
P_DEG, Q_DEG = 2, 3
U_KNOTS = np.array([0, 0, 0, 0.4, 1, 1, 1], float)            # ncp_u = 4
V_KNOTS = np.array([0, 0, 0, 0, 0.3, 0.7, 1, 1, 1, 1], float)  # ncp_v = 6


def _patch():
    rng = np.random.default_rng(20261010)
    ncp_u = len(U_KNOTS) - P_DEG - 1
    ncp_v = len(V_KNOTS) - Q_DEG - 1
    return rng.normal(size=(ncp_v * ncp_u, 3)), ncp_u, ncp_v


def _reference(ctrl, ncp_u, ncp_v, u, v, a=0, b=0):
    """``sum_ij N_i^(a)(u) N_j^(b)(v) P_ij`` straight from ``BSpline`` basis objects."""
    out = np.zeros(3)
    for j in range(ncp_v):
        bv = scipy_interp.BSpline(V_KNOTS, np.eye(ncp_v)[j], Q_DEG)
        bv = bv.derivative(b) if b else bv
        for i in range(ncp_u):
            bu = scipy_interp.BSpline(U_KNOTS, np.eye(ncp_u)[i], P_DEG)
            bu = bu.derivative(a) if a else bu
            out += float(bu(u)) * float(bv(v)) * ctrl[j * ncp_u + i]
    return out


@pytest.mark.parametrize("u,v", [(0.0, 0.0), (0.2, 0.5), (0.4, 0.3), (0.77, 0.9), (0.999, 0.01)])
@pytest.mark.parametrize("derivs", [None, (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (0, 3)])
def test_evaluator_matches_scipy_bspline(u, v, derivs):
    ctrl, ncp_u, ncp_v = _patch()
    s = solid_q1np.evaluate_surface(ctrl, (U_KNOTS, V_KNOTS), (P_DEG, Q_DEG), u, v, derivs=derivs)
    assert np.allclose(s, _reference(ctrl, ncp_u, ncp_v, u, v), atol=1e-12)
    if derivs is not None:
        want = _reference(ctrl, ncp_u, ncp_v, u, v, *derivs)
        assert np.allclose(s.derivative, want, atol=1e-10), (derivs, s.derivative, want)


def test_the_closing_end_of_the_domain_is_evaluated_on_the_last_span():
    """``u = 1`` is not inside any half-open span; upstream's ``UVAL >= U(NK)`` branch handles it."""
    ctrl, ncp_u, ncp_v = _patch()
    s = solid_q1np.evaluate_surface(ctrl, (U_KNOTS, V_KNOTS), (P_DEG, Q_DEG), 1.0, 1.0)
    assert np.allclose(s, ctrl[-1], atol=1e-14)  # open knots interpolate the corner


def test_derivative_naming_and_order_above_degree():
    ctrl = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], float)
    knots = np.array([0, 0, 1, 1], float)
    # z = u*v on this twisted bilinear patch: mixed derivative is exactly 1.
    mixed = solid_q1np.evaluate_surface(ctrl, knots, (1, 1), 0.3, 0.6, derivs=(1, 1))
    assert mixed.dz_du_dv == pytest.approx(1.0) and mixed.dx_du_dv == pytest.approx(0.0)
    # a bilinear patch has no second derivative in u
    second = solid_q1np.evaluate_surface(ctrl, knots, (1, 1), 0.3, 0.6, derivs=(2, 0))
    assert np.allclose(second.derivative, 0.0)
    dv = solid_q1np.evaluate_surface(ctrl, knots, (1, 1), 0.3, 0.6, derivs=(0, 1))
    assert dv.dy_dv == pytest.approx(1.0) and dv.dz_dv == pytest.approx(0.3)


def test_position_alone_carries_no_derivative_and_still_is_an_array():
    ctrl = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    s = solid_q1np.evaluate_surface(ctrl, np.array([0, 0, 1, 1.0]), (1, 1), 0.25, 0.5)
    assert isinstance(s, np.ndarray) and s.shape == (3,)
    assert s.derivative is None and not hasattr(s, "dx_du")
    assert (s.x, s.y, s.z) == pytest.approx((0.25, 0.5, 0.0))


def test_control_point_count_must_match_the_knots():
    with pytest.raises(ValueError, match="control points"):
        solid_q1np.evaluate_surface(np.zeros((5, 3)), np.array([0, 0, 1, 1.0]), (1, 1), 0.5, 0.5)


def test_find_span_matches_the_half_open_rule():
    knots = np.array([0, 0, 0, 0.4, 1, 1, 1], float)
    nk = len(knots)
    assert solid_q1np.find_span(knots, nk, 2, 0.0) == 2
    assert solid_q1np.find_span(knots, nk, 2, 0.399) == 2
    assert solid_q1np.find_span(knots, nk, 2, 0.4) == 3     # the knot belongs to the right span
    assert solid_q1np.find_span(knots, nk, 2, 1.0) == 3     # NK-P-1 (1-based)


def test_basis_is_a_partition_of_unity_and_derivatives_sum_to_zero():
    for u in (0.0, 0.1, 0.4, 0.55, 0.99):
        span = solid_q1np.find_span(U_KNOTS, len(U_KNOTS), P_DEG, u)
        ders = solid_q1np.ders_basis_funs(span, u, P_DEG, U_KNOTS, 2)
        assert ders[0].sum() == pytest.approx(1.0, abs=1e-14)
        assert ders[1].sum() == pytest.approx(0.0, abs=1e-12)
        assert ders[2].sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# the upstream entry points (parent coordinates, element span indices)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zeta", [-1.0, -0.3, 0.0, 0.8, 1.0])
@pytest.mark.parametrize("elem_u,elem_v", [(0, 0), (1, 0), (0, 2)])
def test_shape_functions_are_a_partition_of_unity_with_zero_gradient_sum(zeta, elem_u, elem_v):
    n, dn = solid_q1np.shape_functions(0.3, -0.45, zeta, P_DEG, Q_DEG, U_KNOTS, V_KNOTS, elem_u, elem_v)
    assert n.shape == ((P_DEG + 1) * (Q_DEG + 1) + 4,) and dn.shape == (n.size, 3)
    assert n.sum() == pytest.approx(1.0, abs=1e-13)
    assert np.allclose(dn.sum(axis=0), 0.0, atol=1e-12)


def test_shape_functions_split_into_a_nurbs_top_and_a_bilinear_bottom():
    nt = (P_DEG + 1) * (Q_DEG + 1)
    top, _ = solid_q1np.shape_functions(0.2, 0.1, 1.0, P_DEG, Q_DEG, U_KNOTS, V_KNOTS, 1, 1)
    bot, _ = solid_q1np.shape_functions(0.2, 0.1, -1.0, P_DEG, Q_DEG, U_KNOTS, V_KNOTS, 1, 1)
    assert np.allclose(top[nt:], 0.0) and top[:nt].sum() == pytest.approx(1.0)
    assert np.allclose(bot[:nt], 0.0) and bot[nt:].sum() == pytest.approx(1.0)
    # the four bottom corners, in upstream's order: (-1,-1) (1,-1) (1,1) (-1,1)
    for corner, (xi, eta) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        b, _ = solid_q1np.shape_functions(xi, eta, -1.0, 1, 1, np.array([0, 0, 1, 1.0]),
                                          np.array([0, 0, 1, 1.0]), 0, 0)
        assert b[4 + corner] == pytest.approx(1.0)


def test_top_point_is_the_knot_space_evaluation_and_tangents_carry_the_span_factor():
    ctrl, ncp_u, ncp_v = _patch()
    ids = list(range(1, ctrl.shape[0] + 1))
    # element (1, 1): the knot spans [0.4, 1] x [0.3, 0.7] -> the (p+1)(q+1) = 12 window
    # the shape functions address is the global patch restricted to that window.
    elem_u, elem_v = 1, 1
    window = [(elem_v + j) * ncp_u + (elem_u + i) for j in range(Q_DEG + 1) for i in range(P_DEG + 1)]
    xi, eta = 0.37, -0.62
    uval = 0.4 + 0.6 * 0.5 * (xi + 1)
    vval = 0.3 + 0.4 * 0.5 * (eta + 1)
    pos, dxi, deta = solid_q1np.evaluate_top_surface_point_and_derivs(
        xi, eta, P_DEG, Q_DEG, elem_u, elem_v, U_KNOTS, V_KNOTS,
        [ids[k] for k in window], ctrl)
    want = solid_q1np.evaluate_surface(ctrl, (U_KNOTS, V_KNOTS), (P_DEG, Q_DEG), uval, vval, derivs=(1, 0))
    wantv = solid_q1np.evaluate_surface(ctrl, (U_KNOTS, V_KNOTS), (P_DEG, Q_DEG), uval, vval, derivs=(0, 1))
    assert np.allclose(pos, want, atol=1e-12)
    assert np.allclose(dxi, 0.5 * 0.6 * want.derivative, atol=1e-11)
    assert np.allclose(deta, 0.5 * 0.4 * wantv.derivative, atol=1e-11)
    only = solid_q1np.evaluate_top_surface_point(
        xi, eta, P_DEG, Q_DEG, elem_u, elem_v, U_KNOTS, V_KNOTS, [ids[k] for k in window], ctrl)
    assert np.allclose(only, pos)
    vals = solid_q1np.evaluate_shape_values(xi, eta, P_DEG, Q_DEG, elem_u, elem_v, U_KNOTS, V_KNOTS)
    assert vals.shape == (12,) and vals.sum() == pytest.approx(1.0)


def test_out_of_range_control_point_ids_are_skipped_as_upstream_skips_them():
    knots = np.array([0, 0, 1, 1.0])
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    full = solid_q1np.evaluate_top_surface_point(0, 0, 1, 1, 0, 0, knots, knots, [1, 2, 3, 4], x)
    skip = solid_q1np.evaluate_top_surface_point(0, 0, 1, 1, 0, 0, knots, knots, [1, 2, 0, 99], x)
    assert np.allclose(full, [0.5, 0.5, 0.0])
    assert np.allclose(skip, [0.25, 0.0, 0.0])  # 0.25*P1 + 0.25*P2; ids 0 and 99 contribute nothing


# ---------------------------------------------------------------------------
# the Gauss-point kernel of q1np_forc3.F90 (transcription checks only)
# ---------------------------------------------------------------------------


def _unit_cube(v=None):
    knots = np.array([0, 0, 1, 1.0])
    x = np.array([
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],          # NURBS top, u fastest
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],          # bulk bottom, upstream corner order
    ], float)
    return solid_q1np.Q1NPElement(1, 1, knots, knots, 0, 0, x, v)


def _warped_element():
    """p = q = 2 top patch, off-grid control points, tilted bottom: a general element."""
    knots = np.array([0, 0, 0, 1, 1, 1.0])
    rng = np.random.default_rng(7)
    gx, gy = np.meshgrid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    top = np.stack([gx.ravel(), gy.ravel(), 1.0 + 0.1 * np.sin(3 * gx.ravel() + gy.ravel())], 1)
    top += 0.03 * rng.normal(size=top.shape)
    bot = np.array([[0, 0, 0.05], [1, 0, 0.0], [1, 1, -0.05], [0, 1, 0.0]], float)
    return solid_q1np.Q1NPElement(2, 2, knots, knots, 0, 0, np.vstack([top, bot]))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_gauss_rules_integrate_polynomials_exactly(n):
    x, w = solid_q1np.gauss_1d(n)
    assert w.sum() == pytest.approx(2.0, abs=1e-12)
    for k in range(2 * n):                       # exact to degree 2n-1
        want = 0.0 if k % 2 else 2.0 / (k + 1)
        assert (w * x**k).sum() == pytest.approx(want, abs=2e-12)


def test_gauss_above_five_points_is_upstreams_uniform_grid_not_legendre():
    x, w = solid_q1np.gauss_1d(6)
    assert x[0] == -1.0 and x[-1] == 1.0 and np.allclose(w, 2.0 / 6.0)
    assert (w * x**2).sum() != pytest.approx(2.0 / 3.0, abs=1e-3)   # not exact: the Fortran wins


def test_unit_cube_gauss_volume_is_one_and_jacobian_is_half_the_edge():
    el = _unit_cube()
    res = solid_q1np.forces(el, rho0=1.0, dt1=0.0, stress_fn=lambda gp: solid_q1np.GaussPointStress([0] * 6, 0.0))
    assert res.volg == pytest.approx(1.0, abs=1e-14)
    assert res.vgauss.shape == (2 * 2 * 2,) and np.allclose(res.vgauss, 1.0 / 8.0)


def test_characteristic_length_uses_the_single_precision_point_two():
    el = _unit_cube()
    assert solid_q1np.SPAN_SCALE == float(np.float32(0.2)) != 0.2
    # four unit edges per direction -> SPAN_SCALE * 4, and it is NOT the mean (0.25 * 4 = 1)
    assert solid_q1np.char_len(el) == pytest.approx(4 * solid_q1np.SPAN_SCALE, rel=1e-14)


def test_jacobian_error_is_raised_not_swallowed():
    el = _unit_cube()
    el.x[:4, 2] = -1.0                                  # top below bottom: detJ < 0
    with pytest.raises(solid_q1np.Q1NPBadJacobian):
        solid_q1np.forces(el, rho0=1.0, dt1=0.0, stress_fn=lambda gp: solid_q1np.GaussPointStress([0] * 6, 0.0))


def test_rigid_translation_and_uniform_stretch_strain_rates():
    el = _unit_cube()
    geo = solid_q1np.gp_geometry(el, 0.1, -0.2, 0.3, 1.0)
    rigid = solid_q1np.strain_rate(geo.dn_global, np.tile([3.0, -2.0, 1.0], (8, 1)), dt1=0.1)
    assert all(abs(getattr(rigid, f)) < 1e-14 for f in ("dxx", "dyy", "dzz", "d4", "d5", "d6", "wxx", "wyy", "wzz"))
    a = 0.7
    stretch = solid_q1np.strain_rate(geo.dn_global, np.column_stack([a * el.x[:, 0], 0 * el.x[:, 0], 0 * el.x[:, 0]]), 0.0)
    assert stretch.dxx == pytest.approx(a) and abs(stretch.dyy) + abs(stretch.d4) < 1e-14
    # the DT1/2 correction: dxx -> dxx - dt/2 * dxx^2
    corrected = solid_q1np.strain_rate(geo.dn_global, np.column_stack([a * el.x[:, 0], 0 * el.x[:, 0], 0 * el.x[:, 0]]), 0.2)
    assert corrected.dxx == pytest.approx(a - 0.1 * a * a)


def test_pure_shear_has_no_spin_but_rotation_has_wzz():
    el = _unit_cube()
    geo = solid_q1np.gp_geometry(el, 0.0, 0.0, 0.0, 1.0)
    g = 0.3
    shear = np.column_stack([g * el.x[:, 1], g * el.x[:, 0], 0 * el.x[:, 0]])      # vx = g y, vy = g x
    s = solid_q1np.strain_rate(geo.dn_global, shear, 0.0)
    assert s.d4 == pytest.approx(2 * g) and s.wzz == pytest.approx(0.0, abs=1e-15)
    rot = np.column_stack([-g * el.x[:, 1], g * el.x[:, 0], 0 * el.x[:, 0]])       # rigid spin about z
    r = solid_q1np.strain_rate(geo.dn_global, rot, 0.2)
    assert r.wzz == pytest.approx(0.1 * (g + g))      # DT1/2 * (DYX - DXY)


@pytest.mark.parametrize("make", [_unit_cube, _warped_element])
def test_uniform_stress_equilibrium_and_the_isoparametric_identity(make):
    """``sum_k F_k = 0`` and ``sum_k F_k x_k^T = -V sigma`` for any element shape.

    The second is exact because ``B`` is the isoparametric gradient:
    ``sum_k B_ki x_kj = d x_j / d x_i = delta_ij`` at every Gauss point.
    """
    el = make()
    sxx, syy, szz, sxy, syz, szx = 3.0, -1.0, 0.5, 0.8, -0.4, 0.25
    sig = [sxx, syy, szz, sxy, syz, szx]
    res = solid_q1np.forces(el, rho0=2.0, dt1=0.0,
                            stress_fn=lambda gp: solid_q1np.GaussPointStress(sig, 5.0))
    assert np.allclose(res.f_int.sum(axis=0), 0.0, atol=1e-12)
    m = res.f_int.T @ el.x                     # m[i, j] = sum_k F_ki x_kj
    want = -res.volg * np.array([[sxx, sxy, szx], [sxy, syy, syz], [szx, syz, szz]])
    assert np.allclose(m, want, atol=1e-12)
    assert np.all(res.stig > 0.0)


def test_unit_cube_face_traction_is_minus_sigma_times_area():
    el = _unit_cube()
    res = solid_q1np.forces(el, rho0=1.0, dt1=0.0,
                            stress_fn=lambda gp: solid_q1np.GaussPointStress([4.0, 0, 0, 0, 0, 0], 1.0))
    on_x1 = np.isclose(el.x[:, 0], 1.0)
    on_x0 = np.isclose(el.x[:, 0], 0.0)
    assert res.f_int[on_x1, 0].sum() == pytest.approx(-4.0)   # upstream books the force negated
    assert res.f_int[on_x0, 0].sum() == pytest.approx(+4.0)
    assert np.allclose(res.f_int[:, 1:], 0.0, atol=1e-14)


def test_viscous_terms_enter_as_build_sig_does():
    """``SIG + SVIS - QVIS`` on the normals, ``SIG + SVIS`` on the shears."""
    el = _unit_cube()
    plain = solid_q1np.forces(el, rho0=1.0, dt1=0.0,
                              stress_fn=lambda gp: solid_q1np.GaussPointStress([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0))
    folded = solid_q1np.forces(
        el, rho0=1.0, dt1=0.0,
        stress_fn=lambda gp: solid_q1np.GaussPointStress(
            [0.0] * 6, 1.0, svis=[2.0, 3.0, 4.0, 4.0, 5.0, 6.0], qvis=1.0))
    assert np.allclose(plain.f_int, folded.f_int, atol=1e-14)


def test_deleted_element_books_no_force_but_keeps_its_volume():
    el = _unit_cube()
    calls = []
    res = solid_q1np.forces(el, rho0=1.0, dt1=0.0, off=0.0,
                            stress_fn=lambda gp: calls.append(gp) or solid_q1np.GaussPointStress([1] * 6, 1.0))
    assert calls == [] and np.all(res.f_int == 0.0) and res.volg == pytest.approx(1.0)


def test_gauss_points_are_visited_zeta_outer_xi_inner():
    el = _unit_cube()
    seen = []
    solid_q1np.forces(el, rho0=1.0, dt1=0.0,
                      stress_fn=lambda gp: seen.append((gp.it, gp.iv, gp.iu)) or solid_q1np.GaussPointStress([0] * 6, 0.0))
    assert seen == [(t, v, u) for t in range(2) for v in range(2) for u in range(2)]


def test_forces_has_no_default_material():
    """Upstream's stress is ``MMAIN`` over the element buffer; a default would be invented."""
    with pytest.raises(TypeError):
        solid_q1np.forces(_unit_cube(), rho0=1.0, dt1=0.0)          # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# honest reachability: this port must stay unreferenced and say so
# ---------------------------------------------------------------------------


def test_nothing_in_pyradioss_references_solid_q1np():
    """``forces`` is unreachable; the day something calls it this test must be rewritten on purpose."""
    pattern = re.compile(r"\bsolid_q1np\b")
    offenders = []
    for path in (REPO / "pyradioss").rglob("*.py"):
        if path.name == "solid_q1np.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert offenders == []

    from pyradioss import elements

    assert not any(m.__name__.endswith("solid_q1np") for m in elements.KERNELS.values())


def test_forc3_is_marked_ported_unreachable_in_the_allowlist_and_the_census():
    allow = json.loads((REPO / "tools" / "validation_data" / "port_status.json").read_text())
    assert allow.get(FORC3) == "ported-unreachable"
    assert "ported" not in allow.values()           # nothing here was promoted to a claim of physics
    census = json.loads((REPO / "tools" / "validation_data" / "census.json").read_text())
    row = census["files"][FORC3]
    assert row["status"] == "ported-unreachable"
    assert row["python_module"] == "pyradioss.elements.solid_q1np"
    # the projection that the plan attributes to forc3 lives in the contact file, which is NOT ported
    contact = census["files"]["engine/source/interfaces/ists_q1np/q1np_contact_algorithms.F90"]
    assert contact["status"] == "missing" and contact["python_module"] is None


def test_port_extensions_registry_records_q1np_unreachability():
    text = (REPO / "docs" / "PORT_EXTENSIONS.md").read_text(encoding="utf-8")
    assert "pyradioss/elements/solid_q1np.py" in text
    section = text.split("pyradioss/elements/solid_q1np.py", 1)[1].split("\n## ", 1)[0]
    for needle in ("ported-unreachable", "q1np_forc3.F90", "q1np_generate_main", "Ists",
                   "q1np_contact_project_point_newton"):
        assert needle in section, f"registry entry lost {needle!r}"


def test_module_docstring_declares_the_unreachability():
    doc = solid_q1np.__doc__ or ""
    assert "unreachable" in doc.lower() and "ported-unreachable" in doc
    assert "no projection" in doc.replace("**", "").lower() or "contains **no** projection" in doc


def test_upstream_files_this_module_cites_exist():
    from pyradioss import paths

    root = Path(paths.or_src())
    cited = set(re.findall(r"\$OR_SRC/([A-Za-z0-9_./-]+\.F90)", solid_q1np.__doc__ or ""))
    assert cited, "the module cites no upstream file"
    for rel in cited:
        assert (root / rel).is_file(), rel
