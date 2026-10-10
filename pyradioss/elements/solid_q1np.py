"""Task P2.9 -- Q1NP hexahedra on a B-spline ("NURBS") top surface.

Fortran origin (all read before this file was written)::

    $OR_SRC/engine/source/elements/solid/solid_q1np/q1np_nurbs_surface_eval_mod.F90
    $OR_SRC/engine/source/elements/solid/solid_q1np/q1np_forc3.F90
    $OR_SRC/common_source/modules/q1np_geom_mod.F90
    $OR_SRC/common_source/modules/q1np_restart_mod.F90

What a Q1NP element is
----------------------
A hexahedron whose **top face** (``zeta = +1``) is a tensor-product B-spline
patch of ``(p+1)*(q+1)`` control points and whose **bottom face**
(``zeta = -1``) is the plain bilinear quad on four "bulk" nodes.  The element
therefore has ``(p+1)*(q+1) + 4`` nodes and one shape function per node
(``q1np_geom_mod.F90`` ``Q1NP_SHAPE_FUNCTIONS``): the top ones are the
B-spline tensor basis times ``(1+zeta)/2``, the bottom four are the bilinear
corner functions times ``(1-zeta)/2``.

What is here
------------
* :func:`evaluate_surface` -- the plan's evaluator: a point on a B-spline
  patch in *knot space* ``(u, v)``, optionally with a partial derivative.
  Built on :func:`ders_basis_funs`, the line-for-line port of
  ``Q1NP_DERS_BASIS_FUNS`` (Piegl & Tiller A2.3 with upstream's ``1e-15``
  guards on the knot differences).
* :func:`shape_functions`, :func:`evaluate_top_surface_point`,
  :func:`evaluate_top_surface_point_and_derivs`,
  :func:`evaluate_shape_values` -- the upstream entry points, in the
  upstream's own *parent* coordinates ``(xi, eta) in [-1, 1]`` and element
  span indices.  ``dS/dxi`` there carries the ``du/dxi = (knot span)/2``
  chain factor that ``dS/du`` in :func:`evaluate_surface` does not.
* The Gauss-point kernel of ``Q1NP_FORC3`` that does not need the material
  library: :func:`gauss_1d`, :func:`gp_geometry` (``Q1NP_GP_GEOM`` +
  ``Q1NP_JACOBIAN``), :func:`strain_rate` (``Q1NP_EVAL_DEF``),
  :func:`char_len` (``Q1NP_CHAR_LEN``), :func:`accum_fint`
  (``Q1NP_ACCUM_FINT``) and the loop that drives them, :func:`forces`.

The upstream "NURBS" is **not rational**.  ``Q1NP_WTAB`` exists upstream, but
no routine in the files above multiplies a basis function by a weight; the
surface is evaluated as a plain B-spline.  This port does the same and has no
``weights`` argument -- adding one would be inventing a rational evaluator.

REACHABILITY -- READ BEFORE CITING THIS MODULE AS COVERAGE
----------------------------------------------------------
:func:`evaluate_surface` and the shape-function family are plain functions
that tests call directly.  **Nothing in ``pyradioss`` calls
:func:`forces`, and nothing can: no deck pyradioss reads produces a Q1NP
element.**  This is a port that is *unreachable*, and the census says so
(``ported-unreachable``, ``docs/PORT_EXTENSIONS.md`` §``solid_q1np``).

The reason is narrower than "the deck format has no NURBS syntax", and the
narrow version is the true one.  Upstream does not read a NURBS surface from
the deck at all.  ``starter/source/starter/lectur.F`` calls
``Q1NP_GENERATE_MAIN`` (``starter/.../solid_q1np/q1np_generate_main.F90``),
which, for an ``/INTER`` carrying the ``Ists`` flag, takes the two ordinary
mesh ``/SURF`` entities it names, fits a B-spline top surface to one of them
(``q1np_genelements.F90``, ``q1np_cholesky.F90``) and builds the
``KQ1NP_TAB`` / ``Q1NP_CPTAB`` / ``Q1NP_KTAB`` tables the Engine reads.  So a
deck *can* reach the Fortran kernel.  It cannot reach this one, because
``pyradioss`` ports none of that Starter chain (about 6,000 lines, and the
``/INTER`` ``Ists`` reader it hangs off).  Nothing here invents it.

What is **not** ported from ``q1np_forc3.F90``, and why
-------------------------------------------------------
* ``Q1NP_GP_MAT`` -- ``SROTA3`` / ``SRHO3`` / ``MMAIN`` / ``SSTRA3``, the
  constitutive update on the element buffer.  :func:`forces` takes the
  stress as a caller-supplied callback instead of reaching for a material.
  A default would be an invented material lookup.
* ``Q1NP_INIT_NODE_MAP`` / ``Q1NP_GET_BULK_NODE_IDS`` /
  ``Q1NP_REBUILD_BULK_*`` -- they consume ``KQ1NP_TAB`` and the node
  numbering, which only the (unported) Starter produces.
* ``SMALLA3`` / ``SMALLB3`` (``ISMSTR`` small-strain storage), the
  ``Q1NP_AVG_SIG_BILAN`` stress averaging and energy bookkeeping, the
  ``IDTMIN(101)`` ``MASS_ELEM`` accumulation (written, never read upstream).
* ``q1np_dump_hist_state.F90`` -- a debug CSV dump.

``q1np_forc3.F90`` contains **no** projection of nodes onto the surface (the
plan's wording to the contrary was checked against the file and is wrong).
The Newton point projection lives in
``engine/source/interfaces/ists_q1np/q1np_contact_algorithms.F90`` (named
without the ``$OR_SRC`` prefix on purpose: the census reads that prefix as "this
module implements the file", and it does not),
which this task does not port; it is the consumer of
:func:`evaluate_top_surface_point_and_derivs`.

Index conventions
-----------------
Upstream is 1-based Fortran.  Here every array index and every span is
0-based; control points are stored with **u fastest** (``(j * ncp_u) + i``),
the order of ``Q1NP_SHAPE_FUNCTIONS``' tensor loop and of
``Q1NP_BASIS_ROW_AT_UV``'s ``COL = (JJ-1)*NCP_U + II``.  Control points are
followed by the four bulk nodes in the element node list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

#: ``TOL_NDU`` in ``Q1NP_DERS_BASIS_FUNS``.
TOL_NDU = 1.0e-15

#: ``SPAN_SCALE`` in ``Q1NP_CHAR_LEN``.  Upstream writes the literal ``0.2``
#: with no ``_WP`` suffix, i.e. a *default real* converted to ``WP``.  The
#: single-precision value is what the Fortran really multiplies by, in both
#: the ``WP = 4`` and the ``WP = 8`` build, so it is kept (it is 0.2 to 7
#: digits: 0.20000000298...), not "corrected" to the double 0.2.
SPAN_SCALE = float(np.float32(0.2))

#: ``EPSILON(ONE)`` guard of ``Q1NP_JACOBIAN`` (double build).
_JAC_EPS = float(np.finfo(np.float64).eps)

#: Number of bilinear bulk nodes of the bottom face (``NBULK``).
NBULK = 4


# ---------------------------------------------------------------------------
# B-spline basis (q1np_geom_mod.F90)
# ---------------------------------------------------------------------------


def find_span(knots: np.ndarray, nk: int, p: int, u: float) -> int:
    """0-based knot span ``i`` with ``U[i] <= u < U[i+1]`` (``Q1NP_FIND_SPAN``).

    ``u >= U[nk-1]`` returns the last non-empty span, ``nk - p - 2`` (upstream
    ``NK - P - 1`` in 1-based numbering); a ``u`` that no span contains falls
    back to span 0 exactly as upstream falls back to ``SPAN = 1``.
    """
    if u >= knots[nk - 1]:
        return nk - p - 2
    # Fortran: DO I = 2, NK-1  with  U(I) <= UVAL < U(I+1)   (1-based)
    for i1 in range(2, nk):
        if knots[i1 - 1] <= u < knots[i1]:
            return i1 - 1
    return 0


def ders_basis_funs(span: int, u: float, p: int, knots: np.ndarray, nders: int) -> np.ndarray:
    """B-spline basis functions and derivatives (``Q1NP_DERS_BASIS_FUNS``).

    Returns ``DERS`` of shape ``(nders + 1, p + 1)``: ``DERS[k, j]`` is the
    ``k``-th derivative of the ``j``-th nonzero basis function
    ``N_{span-p+j, p}`` at ``u``.  ``span`` is 0-based.  Derivative orders
    above ``p`` are identically zero; upstream only ever asks for
    ``nders = 1`` and the loop below is undefined past ``p``, so the extra
    rows are returned as zeros rather than computed.
    """
    ders = np.zeros((nders + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        temp = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            ndu[j, r] = 0.0 if abs(denom) <= TOL_NDU else denom
            if abs(ndu[j, r]) > TOL_NDU:
                denom = ndu[r, j - 1] / ndu[j, r]
            else:
                denom = 0.0
            ndu[r, j] = temp + right[r + 1] * denom
            temp = left[j - r] * denom
        ndu[j, j] = temp

    for j in range(p + 1):
        ders[0, j] = ndu[j, p]
    if nders <= 0:
        return ders

    kmax = min(nders, p)
    for r in range(p + 1):
        s1, s2 = 0, 1
        a = np.zeros((2, p + 1))
        a[0, 0] = 1.0
        for k in range(1, kmax + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                denom = ndu[pk + 1, rk]
                a[s2, 0] = a[s1, 0] / denom if abs(denom) > TOL_NDU else 0.0
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1, rk + j]
                if abs(denom) > TOL_NDU:
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / denom
                else:
                    a[s2, j] = 0.0
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                denom = ndu[pk + 1, r]
                a[s2, k] = -a[s1, k - 1] / denom if abs(denom) > TOL_NDU else 0.0
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, 1 - s2

    rfact = float(p)
    for k in range(1, kmax + 1):
        ders[k, :] *= rfact
        rfact *= float(p - k)
    return ders


def shape_functions(
    xi: float,
    eta: float,
    zeta: float,
    p: int,
    q: int,
    u_knots: np.ndarray,
    v_knots: np.ndarray,
    elem_u: int,
    elem_v: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Q1NP shape functions and parent-coordinate derivatives.

    Port of ``Q1NP_SHAPE_FUNCTIONS``.  ``(xi, eta, zeta)`` are in
    ``[-1, 1]^3``; ``elem_u`` / ``elem_v`` are the **0-based** element
    indices along the knot vectors (the element occupies the knot span
    ``[U[p+elem_u], U[p+elem_u+1]]``, which assumes an open knot vector
    without repeated interior knots, as upstream does).

    Returns ``(N, DN)`` with ``N.shape == ((p+1)*(q+1) + 4,)`` -- the
    ``(p+1)*(q+1)`` top functions (u fastest), then the four bottom ones --
    and ``DN.shape == (nnode, 3)`` holding ``dN/dxi, dN/deta, dN/dzeta``.
    """
    n_top = (p + 1) * (q + 1)
    xi_loc = 0.5 * (xi + 1.0)
    eta_loc = 0.5 * (eta + 1.0)

    su = p + elem_u
    sv = q + elem_v
    au, bu = u_knots[su], u_knots[su + 1]
    av, bv = v_knots[sv], v_knots[sv + 1]
    uval = au + (bu - au) * xi_loc
    vval = av + (bv - av) * eta_loc

    kspan_u = bu - au
    kspan_v = bv - av
    du_dxi = 0.5 * kspan_u
    dv_deta = 0.5 * kspan_v

    nu_ders = ders_basis_funs(su, uval, p, u_knots, 1)
    nv_ders = ders_basis_funs(sv, vval, q, v_knots, 1)
    nu, dnu = nu_ders[0], nu_ders[1]
    nv, dnv = nv_ders[0], nv_ders[1]

    n_top_ary = np.empty(n_top)
    dn_top_du = np.empty(n_top)
    dn_top_dv = np.empty(n_top)
    idx = 0
    for j in range(q + 1):
        for i in range(p + 1):
            n_top_ary[idx] = nu[i] * nv[j]
            dn_top_du[idx] = dnu[i] * nv[j]
            dn_top_dv[idx] = nu[i] * dnv[j]
            idx += 1

    nz_top = 0.5 * (1.0 + zeta)
    nz_bot = 0.5 * (1.0 - zeta)
    dnz_top = 0.5
    dnz_bot = -0.5

    inv = 1.0 / (kspan_u * kspan_v)
    n_bot = np.array(
        [
            (bu - uval) * (bv - vval) * inv,
            (uval - au) * (bv - vval) * inv,
            (uval - au) * (vval - av) * inv,
            (bu - uval) * (vval - av) * inv,
        ]
    )
    dn_bot_du = np.array(
        [-(bv - vval) * inv, (bv - vval) * inv, (vval - av) * inv, -(vval - av) * inv]
    )
    dn_bot_dv = np.array(
        [-(bu - uval) * inv, -(uval - au) * inv, (uval - au) * inv, (bu - uval) * inv]
    )

    n = np.empty(n_top + NBULK)
    dn = np.empty((n_top + NBULK, 3))
    n[:n_top] = n_top_ary * nz_top
    dn[:n_top, 0] = dn_top_du * du_dxi * nz_top
    dn[:n_top, 1] = dn_top_dv * dv_deta * nz_top
    dn[:n_top, 2] = n_top_ary * dnz_top
    n[n_top:] = n_bot * nz_bot
    dn[n_top:, 0] = dn_bot_du * du_dxi * nz_bot
    dn[n_top:, 1] = dn_bot_dv * dv_deta * nz_bot
    dn[n_top:, 2] = n_bot * dnz_bot
    return n, dn


# ---------------------------------------------------------------------------
# The plan's evaluator, in knot space
# ---------------------------------------------------------------------------


class SurfacePoint(np.ndarray):
    """A point ``(x, y, z)`` on a surface, as a length-3 ``ndarray``.

    ``.x .y .z`` are the position.  When a derivative was requested the
    result also carries its components as ``dx_du``, ``dy_du``, ``dz_du``
    (for ``derivs=(1, 0)``), ``dx_dv`` (``(0, 1)``), ``dx_du_dv`` (``(1, 1)``),
    ``dx_du2`` (``(2, 0)``) ... and the vector as ``.derivative``.
    """

    derivs: Tuple[int, int]
    derivative: Optional[np.ndarray]

    def __new__(cls, xyz, derivs=(0, 0), derivative=None):
        obj = np.asarray(xyz, dtype=float).view(cls)
        obj.derivs = (int(derivs[0]), int(derivs[1]))
        obj.derivative = None if derivative is None else np.asarray(derivative, float)
        if obj.derivative is not None:
            suffix = _deriv_suffix(obj.derivs)
            for comp, val in zip("xyz", obj.derivative):
                setattr(obj, f"d{comp}_{suffix}", float(val))
        return obj

    def __array_finalize__(self, obj):
        self.derivs = getattr(obj, "derivs", (0, 0))
        self.derivative = getattr(obj, "derivative", None)

    @property
    def x(self) -> float:
        return float(self[0])

    @property
    def y(self) -> float:
        return float(self[1])

    @property
    def z(self) -> float:
        return float(self[2])


def _deriv_suffix(derivs: Tuple[int, int]) -> str:
    a, b = derivs
    parts = []
    if a:
        parts.append("du" if a == 1 else f"du{a}")
    if b:
        parts.append("dv" if b == 1 else f"dv{b}")
    return "_".join(parts)


def _split_knots(knots) -> Tuple[np.ndarray, np.ndarray]:
    """One knot vector serves both directions; a pair ``(U, V)`` is also accepted."""
    if (
        isinstance(knots, (tuple, list))
        and len(knots) == 2
        and all(np.ndim(k) == 1 for k in knots)
    ):
        return np.asarray(knots[0], float), np.asarray(knots[1], float)
    arr = np.asarray(knots, float)
    if arr.ndim != 1:
        raise ValueError("knots must be one vector, or a (U, V) pair of vectors")
    return arr, arr


def evaluate_surface(
    ctrl,
    knots,
    degree: Sequence[int],
    u: float,
    v: float,
    derivs: Optional[Tuple[int, int]] = None,
) -> SurfacePoint:
    """Evaluate a tensor-product B-spline surface at knot-space ``(u, v)``.

    ``ctrl`` is ``(ncp_u * ncp_v, 3)`` with **u fastest** (or the equivalent
    ``(ncp_v, ncp_u, 3)``).  ``knots`` is one open knot vector used for both
    directions, or a ``(U, V)`` pair; ``ncp_u = len(U) - p - 1``.  The
    position is always returned; ``derivs=(a, b)`` additionally attaches the
    partial derivative ``d^(a+b) S / du^a dv^b`` (see :class:`SurfacePoint`).

    This is the same sum :func:`evaluate_top_surface_point` forms
    (``sum_k N_k * X_k`` over the ``(p+1)*(q+1)`` nonzero basis functions)
    with the parent-coordinate map removed, so ``dS/du`` here is
    ``dS/dxi / (0.5 * knot span)``.
    """
    p, q = int(degree[0]), int(degree[1])
    u_knots, v_knots = _split_knots(knots)
    nku, nkv = len(u_knots), len(v_knots)
    ncp_u, ncp_v = nku - p - 1, nkv - q - 1
    if ncp_u < 1 or ncp_v < 1:
        raise ValueError(
            f"knot vectors of length {nku}/{nkv} cannot carry degree {p}/{q}"
        )
    pts = np.asarray(ctrl, float).reshape(-1, 3)
    if pts.shape[0] != ncp_u * ncp_v:
        raise ValueError(
            f"{pts.shape[0]} control points, but the knots and degree imply "
            f"{ncp_u} x {ncp_v} = {ncp_u * ncp_v}"
        )
    a, b = (0, 0) if derivs is None else (int(derivs[0]), int(derivs[1]))
    if a < 0 or b < 0:
        raise ValueError("derivative orders must be non-negative")

    su = find_span(u_knots, nku, p, u)
    sv = find_span(v_knots, nkv, q, v)
    nu = ders_basis_funs(su, u, p, u_knots, min(a, p))
    nv = ders_basis_funs(sv, v, q, v_knots, min(b, q))
    nu0 = nu[0]
    nv0 = nv[0]
    nua = nu[a] if a <= p else np.zeros(p + 1)
    nvb = nv[b] if b <= q else np.zeros(q + 1)

    pos = np.zeros(3)
    der = np.zeros(3)
    for j in range(q + 1):
        jj = sv - q + j
        for i in range(p + 1):
            ii = su - p + i
            cp = pts[jj * ncp_u + ii]
            pos += nu0[i] * nv0[j] * cp
            if derivs is not None:
                der += nua[i] * nvb[j] * cp
    if derivs is None:
        return SurfacePoint(pos)
    return SurfacePoint(pos, derivs=(a, b), derivative=der)


# ---------------------------------------------------------------------------
# Upstream entry points: q1np_nurbs_surface_eval_mod.F90
# ---------------------------------------------------------------------------


def _gather(ctrl_point_ids: Sequence[int], x_coords: np.ndarray, n: np.ndarray, dn: Optional[np.ndarray]):
    """The ``GID <= 0 .OR. GID > NUMNOD -> CYCLE`` accumulation loop (1-based ids)."""
    numnod = x_coords.shape[0]
    xyz = np.zeros(3)
    dxi = np.zeros(3)
    deta = np.zeros(3)
    for k, gid in enumerate(ctrl_point_ids):
        if gid <= 0 or gid > numnod:
            continue
        xk = x_coords[gid - 1]
        xyz += n[k] * xk
        if dn is not None:
            dxi += dn[k, 0] * xk
            deta += dn[k, 1] * xk
    return xyz, dxi, deta


def evaluate_top_surface_point(
    xi: float,
    eta: float,
    p: int,
    q: int,
    elem_u: int,
    elem_v: int,
    u_knots: np.ndarray,
    v_knots: np.ndarray,
    ctrl_point_ids: Sequence[int],
    x_coords: np.ndarray,
) -> np.ndarray:
    """Position on the ``zeta = +1`` surface (``Q1NP_EVALUATE_NURBS_TOP_SURFACE_POINT``).

    ``ctrl_point_ids`` are **1-based** global node ids as upstream stores
    them; ids outside ``1..numnod`` are skipped, as upstream does.
    ``x_coords`` is ``(numnod, 3)``.
    """
    n, dn = shape_functions(xi, eta, 1.0, p, q, u_knots, v_knots, elem_u, elem_v)
    xyz, _, _ = _gather(ctrl_point_ids, np.asarray(x_coords, float), n, None)
    return xyz


def evaluate_top_surface_point_and_derivs(
    xi: float,
    eta: float,
    p: int,
    q: int,
    elem_u: int,
    elem_v: int,
    u_knots: np.ndarray,
    v_knots: np.ndarray,
    ctrl_point_ids: Sequence[int],
    x_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(S, dS/dxi, dS/deta)`` on ``zeta = +1``
    (``Q1NP_EVALUATE_NURBS_TOP_SURFACE_POINT_AND_DERIVS``).

    The tangents are derivatives with respect to the **parent** coordinates,
    which is what upstream's Newton projection forms its residual from.
    """
    n, dn = shape_functions(xi, eta, 1.0, p, q, u_knots, v_knots, elem_u, elem_v)
    return _gather(ctrl_point_ids, np.asarray(x_coords, float), n, dn)


def evaluate_shape_values(
    xi: float,
    eta: float,
    p: int,
    q: int,
    elem_u: int,
    elem_v: int,
    u_knots: np.ndarray,
    v_knots: np.ndarray,
) -> np.ndarray:
    """The ``(p+1)*(q+1)`` top-surface weights (``Q1NP_EVALUATE_NURBS_SHAPE_VALUES``)."""
    n, _ = shape_functions(xi, eta, 1.0, p, q, u_knots, v_knots, elem_u, elem_v)
    return n[: (p + 1) * (q + 1)].copy()


# ---------------------------------------------------------------------------
# q1np_forc3.F90 -- the Gauss-point kernel, without the material library
# ---------------------------------------------------------------------------


def gauss_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre points and weights on ``[-1, 1]`` (``Q1NP_GAUSS_1D``).

    Upstream hard-codes ``n = 1..5`` and, above that, falls back to a
    **uniform** grid with weights ``2/n`` -- which is not Gauss-Legendre and
    does not integrate polynomials exactly.  Kept: the Fortran wins.
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        g = 0.577350269189626
        return np.array([-g, g]), np.array([1.0, 1.0])
    if n == 3:
        g = 0.774596669241483
        return (
            np.array([-g, 0.0, g]),
            np.array([0.555555555555556, 0.888888888888889, 0.555555555555556]),
        )
    if n == 4:
        a, b = 0.861136311594053, 0.339981043584856
        wa, wb = 0.347854845137454, 0.652145154862546
        return np.array([-a, -b, b, a]), np.array([wa, wb, wb, wa])
    if n == 5:
        a, b = 0.906179845938664, 0.538469310105683
        wa, wb, wc = 0.236926885056189, 0.478628670499366, 0.568888888888889
        return np.array([-a, -b, 0.0, b, a]), np.array([wa, wb, wc, wb, wa])
    i = np.arange(n, dtype=float)
    return -1.0 + 2.0 * i / float(n - 1), np.full(n, 2.0 / float(n))


class Q1NPBadJacobian(ArithmeticError):
    """``Q1NP_JACOBIAN`` returned ``IERR = 1`` (``detJ <= EPSILON``).

    Upstream prints a diagnostic and ``RETURN``s from ``Q1NP_FORC3`` *without*
    any error status, silently abandoning the group.  Raising is the
    deviation: a swallowed return would hand the caller a half-integrated
    force vector.
    """


def jacobian(dn_local: np.ndarray, xnode: np.ndarray):
    """``Q1NP_JACOBIAN``: ``(J, detJ, Jinv, dN/dx, ierr)``.

    ``J[i, j] = sum_k X[k, i] * dN[k, j]``; the inverse is the explicit
    cofactor form, and when ``detJ <= eps`` it is the zero matrix with
    ``ierr = 1`` (so ``dN/dx`` is zero too), as upstream.
    """
    jm = xnode.T @ dn_local
    detj = (
        jm[0, 0] * (jm[1, 1] * jm[2, 2] - jm[1, 2] * jm[2, 1])
        - jm[0, 1] * (jm[1, 0] * jm[2, 2] - jm[1, 2] * jm[2, 0])
        + jm[0, 2] * (jm[1, 0] * jm[2, 1] - jm[1, 1] * jm[2, 0])
    )
    if detj <= _JAC_EPS:
        jinv = np.zeros((3, 3))
        ierr = 1
    else:
        d = 1.0 / detj
        jinv = np.array(
            [
                [
                    (jm[1, 1] * jm[2, 2] - jm[1, 2] * jm[2, 1]) * d,
                    -(jm[0, 1] * jm[2, 2] - jm[0, 2] * jm[2, 1]) * d,
                    (jm[0, 1] * jm[1, 2] - jm[0, 2] * jm[1, 1]) * d,
                ],
                [
                    -(jm[1, 0] * jm[2, 2] - jm[1, 2] * jm[2, 0]) * d,
                    (jm[0, 0] * jm[2, 2] - jm[0, 2] * jm[2, 0]) * d,
                    -(jm[0, 0] * jm[1, 2] - jm[0, 2] * jm[1, 0]) * d,
                ],
                [
                    (jm[1, 0] * jm[2, 1] - jm[1, 1] * jm[2, 0]) * d,
                    -(jm[0, 0] * jm[2, 1] - jm[0, 1] * jm[2, 0]) * d,
                    (jm[0, 0] * jm[1, 1] - jm[0, 1] * jm[1, 0]) * d,
                ],
            ]
        )
        ierr = 0
    return jm, detj, jinv, dn_local @ jinv, ierr


@dataclass
class Q1NPElement:
    """One Q1NP element's geometry and kinematics.

    ``x`` and ``v`` are ``(nnode, 3)`` with the ``(p+1)*(q+1)`` control
    points first (u fastest) and the four bulk nodes last -- upstream's
    ``X_ELEM`` / ``V_ELEM`` order.  ``elem_u`` / ``elem_v`` are 0-based.
    """

    p: int
    q: int
    u_knots: np.ndarray
    v_knots: np.ndarray
    elem_u: int
    elem_v: int
    x: np.ndarray
    v: np.ndarray = field(default=None)

    def __post_init__(self):
        self.u_knots = np.asarray(self.u_knots, float)
        self.v_knots = np.asarray(self.v_knots, float)
        self.x = np.asarray(self.x, float)
        if self.x.shape != (self.nnode, 3):
            raise ValueError(
                f"x must be ({self.nnode}, 3) = (p+1)*(q+1) + {NBULK} nodes, got {self.x.shape}"
            )
        self.v = np.zeros_like(self.x) if self.v is None else np.asarray(self.v, float)
        if self.v.shape != self.x.shape:
            raise ValueError("v must have the shape of x")

    @property
    def nctrl(self) -> int:
        return (self.p + 1) * (self.q + 1)

    @property
    def nnode(self) -> int:
        return self.nctrl + NBULK


def eval_phys_point(elem: Q1NPElement, xi: float, eta: float, zeta: float) -> np.ndarray:
    """Current position at parent coordinates (``Q1NP_EVAL_PHYS_POINT``)."""
    n, _ = shape_functions(
        xi, eta, zeta, elem.p, elem.q, elem.u_knots, elem.v_knots, elem.elem_u, elem.elem_v
    )
    return n @ elem.x


def char_len(elem: Q1NPElement) -> float:
    """Characteristic length ``DELTAX`` (``Q1NP_CHAR_LEN``).

    ``SPAN_SCALE`` times the sum of the four edge lengths of each direction
    (note: ``0.2`` times *four* lengths, not the mean ``0.25``), minimum over
    the three directions.
    """

    def pt(xi, eta, zeta):
        return eval_phys_point(elem, xi, eta, zeta)

    def dist(a, b):
        d = b - a
        return float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))

    t1, t2, t3, t4 = pt(-1, -1, 1), pt(1, -1, 1), pt(1, 1, 1), pt(-1, 1, 1)
    b1, b2, b3, b4 = pt(-1, -1, -1), pt(1, -1, -1), pt(1, 1, -1), pt(-1, 1, -1)
    span_u = SPAN_SCALE * (dist(t1, t2) + dist(t4, t3) + dist(b1, b2) + dist(b4, b3))
    span_v = SPAN_SCALE * (dist(t1, t4) + dist(t2, t3) + dist(b1, b4) + dist(b2, b3))
    span_t = SPAN_SCALE * (dist(t1, b1) + dist(t2, b2) + dist(t3, b3) + dist(t4, b4))
    return min(span_u, min(span_v, span_t))


@dataclass
class GaussPointGeometry:
    """Output of :func:`gp_geometry` (one Gauss point of one element)."""

    n: np.ndarray            # NVAL
    dn_global: np.ndarray    # DN_GLOBAL == MATB_GP, (nnode, 3)
    detj: float
    voln: float              # GPW * detJ


def gp_geometry(elem: Q1NPElement, xi: float, eta: float, zeta: float, gpw: float) -> GaussPointGeometry:
    """``Q1NP_GP_GEOM`` for one lane: shape functions, Jacobian, ``VOLN = GPW * detJ``."""
    n, dn_local = shape_functions(
        xi, eta, zeta, elem.p, elem.q, elem.u_knots, elem.v_knots, elem.elem_u, elem.elem_v
    )
    _, detj, _, dn_global, ierr = jacobian(dn_local, elem.x)
    if ierr != 0:
        raise Q1NPBadJacobian(
            f"Q1NP bad Jacobian: detJ={detj:.4e} at (xi,eta,zeta)=({xi:.4f},{eta:.4f},{zeta:.4f}), "
            f"p={elem.p} q={elem.q} elem_u={elem.elem_u} elem_v={elem.elem_v}"
        )
    return GaussPointGeometry(n=n, dn_global=dn_global, detj=detj, voln=gpw * detj)


@dataclass
class StrainRate:
    """The ``DXX..D6`` / ``WXX..WZZ`` arrays of ``Q1NP_EVAL_DEF`` for one lane."""

    dxx: float = 0.0
    dyy: float = 0.0
    dzz: float = 0.0
    dxy: float = 0.0
    dyx: float = 0.0
    dyz: float = 0.0
    dzy: float = 0.0
    dzx: float = 0.0
    dxz: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    d6: float = 0.0
    wxx: float = 0.0
    wyy: float = 0.0
    wzz: float = 0.0


def strain_rate(matb: np.ndarray, v_elem: np.ndarray, dt1: float) -> StrainRate:
    """``Q1NP_EVAL_DEF``: velocity gradient, then the ``DT1/2`` quadratic correction.

    ``matb`` is ``dN/dx`` ``(nnode, 3)``; ``v_elem`` is ``(nnode, 3)``.
    The velocity-gradient terms are named ``D<row><col>`` upstream with the
    *derivative* index second: ``DXY = sum dN/dy * vx`` is ``dvx/dy``.
    """
    s = StrainRate()
    s.dxx = float(matb[:, 0] @ v_elem[:, 0])
    s.dyy = float(matb[:, 1] @ v_elem[:, 1])
    s.dzz = float(matb[:, 2] @ v_elem[:, 2])
    s.dxy = float(matb[:, 1] @ v_elem[:, 0])
    s.dyx = float(matb[:, 0] @ v_elem[:, 1])
    s.dyz = float(matb[:, 2] @ v_elem[:, 1])
    s.dzy = float(matb[:, 1] @ v_elem[:, 2])
    s.dzx = float(matb[:, 0] @ v_elem[:, 2])
    s.dxz = float(matb[:, 2] @ v_elem[:, 0])

    h = 0.5 * dt1
    s.dxx = s.dxx - h * (s.dxx * s.dxx + s.dyx * s.dyx + s.dzx * s.dzx)
    s.dyy = s.dyy - h * (s.dyy * s.dyy + s.dzy * s.dzy + s.dxy * s.dxy)
    s.dzz = s.dzz - h * (s.dzz * s.dzz + s.dxz * s.dxz + s.dyz * s.dyz)

    aaa = h * (s.dxx * s.dxy + s.dyx * s.dyy + s.dzx * s.dzy)
    s.dxy -= aaa
    s.dyx -= aaa
    s.d4 = s.dxy + s.dyx

    aaa = h * (s.dyy * s.dyz + s.dzy * s.dzz + s.dxy * s.dxz)
    s.dyz -= aaa
    s.dzy -= aaa
    s.d5 = s.dyz + s.dzy

    aaa = h * (s.dzz * s.dzx + s.dxz * s.dxx + s.dyz * s.dyx)
    s.dxz -= aaa
    s.dzx -= aaa
    s.d6 = s.dxz + s.dzx

    s.wxx = h * (s.dzy - s.dyz)
    s.wyy = h * (s.dxz - s.dzx)
    s.wzz = h * (s.dyx - s.dxy)
    return s


def accum_fint(
    matb: np.ndarray,
    voln: float,
    sig: Sequence[float],
    rho0: float,
    ssp_eq: float,
    f_int: np.ndarray,
    stig: np.ndarray,
) -> np.ndarray:
    """``Q1NP_ACCUM_FINT``: add one Gauss point to the element-local buffers.

    ``sig = (sxx, syy, szz, sxy, syz, szx)`` (``SIG1..SIG6``).  ``f_int``
    (``(nnode, 3)``) and ``stig`` (``(nnode,)``) are updated in place and
    ``f_int`` is returned.  ``rho0`` is ``PM(89, mat)`` -- the density the
    upstream stiffness estimate ``rho * ssp^2`` is built from.
    """
    s1, s2, s3, s4, s5, s6 = (float(t) for t in sig)
    bx, by, bz = matb[:, 0], matb[:, 1], matb[:, 2]
    sumx = float(np.sum(np.abs(bx)))
    sumy = float(np.sum(np.abs(by)))
    sumz = float(np.sum(np.abs(bz)))
    aa = rho0 * ssp_eq * ssp_eq

    # FCOMP(1,1..6) summed over the six columns, component by component.
    fx = -voln * (bx * s1 + by * s4 + bz * s6)
    fy = -voln * (by * s2 + bx * s4 + bz * s5)
    fz = -voln * (bz * s3 + by * s5 + bx * s6)
    f_int[:, 0] += fx
    f_int[:, 1] += fy
    f_int[:, 2] += fz

    stin = 0.5 * voln * (np.abs(bx) * sumx + np.abs(by) * sumy + np.abs(bz) * sumz)
    stig += stin * aa
    return f_int


@dataclass
class GaussPointState:
    """What the caller's constitutive update is handed at each Gauss point."""

    iu: int
    iv: int
    it: int
    xi: float
    eta: float
    zeta: float
    voln: float
    deltax: float
    strain_rate: StrainRate


@dataclass
class GaussPointStress:
    """What it hands back.

    ``sig`` is the material stress ``LBUF%SIG`` (``SIG1..SIG6``); ``svis``
    (``SVIS(:,1:6)``) and ``qvis`` (bulk viscosity pressure) are folded in by
    ``Q1NP_BUILD_SIG``: ``sxx = sig + svis - qvis`` on the three normals,
    ``sig + svis`` on the shears.  ``ssp_eq`` is the equivalent sound speed
    that feeds the nodal stiffness estimate.
    """

    sig: Sequence[float]
    ssp_eq: float
    svis: Sequence[float] = (0.0,) * 6
    qvis: float = 0.0


@dataclass
class Q1NPForces:
    """Result of :func:`forces` for one element."""

    f_int: np.ndarray        # (nnode, 3) -- already negated, as upstream accumulates it
    stig: np.ndarray         # (nnode,)   -- the STIFN contribution
    volg: float              # sum of the Gauss-point volumes
    vgauss: np.ndarray       # per-Gauss-point volumes, IPT order (u fastest, then v, then t)
    deltax: float


def forces(
    elem: Q1NPElement,
    *,
    rho0: float,
    dt1: float,
    stress_fn: Callable[[GaussPointState], GaussPointStress],
    off: float = 1.0,
    np_t: int = 2,
) -> Q1NPForces:
    """Internal force of one Q1NP element -- the ``Q1NP_FORC3`` Gauss loop.

    **Unreachable from a deck and unreferenced by the engine** -- see the
    module docstring.  There is deliberately no default ``stress_fn``: the
    upstream stress comes from ``MMAIN`` over the element buffer, and
    supplying a stand-in here would be a made-up material lookup.

    The Gauss scheme is ``(p+1) x (q+1) x np_t`` (``NP_T_L = 2`` upstream),
    iterated ``zeta`` outermost and ``xi`` innermost; ``off <= 0`` (a deleted
    element) integrates the geometry but books no force, as upstream's
    ``Q1NP_IS_ACTIVE`` gates ``Q1NP_EVAL_DEF`` / ``Q1NP_ACCUM_NFORCE`` /
    ``Q1NP_ASSEMBLE_FINT``.  The returned ``f_int`` is the element-local
    buffer; the scatter into the global ``A`` / ``STIFN``
    (``Q1NP_ASSEMBLE_FINT``) needs the node numbering and is the caller's.
    """
    np_u, np_v = elem.p + 1, elem.q + 1
    gp_u, gw_u = gauss_1d(np_u)
    gp_v, gw_v = gauss_1d(np_v)
    gp_t, gw_t = gauss_1d(np_t)

    deltax = char_len(elem)
    active = off > 0.0
    f_int = np.zeros((elem.nnode, 3))
    stig = np.zeros(elem.nnode)
    vgauss = np.zeros(np_u * np_v * np_t)
    volg = 0.0

    ipt = 0
    for it in range(np_t):
        zeta = gp_t[it]
        for iv in range(np_v):
            eta = gp_v[iv]
            for iu in range(np_u):
                xi = gp_u[iu]
                gpw = gw_u[iu] * gw_v[iv] * gw_t[it]
                geo = gp_geometry(elem, xi, eta, zeta, gpw)
                vgauss[ipt] = geo.voln
                volg += geo.voln
                ipt += 1
                if not active:
                    continue
                sr = strain_rate(geo.dn_global, elem.v, dt1)
                st = stress_fn(
                    GaussPointState(iu, iv, it, xi, eta, zeta, geo.voln, deltax, sr)
                )
                svis = list(st.svis)
                sig = [
                    st.sig[0] + svis[0] - st.qvis,
                    st.sig[1] + svis[1] - st.qvis,
                    st.sig[2] + svis[2] - st.qvis,
                    st.sig[3] + svis[3],
                    st.sig[4] + svis[4],
                    st.sig[5] + svis[5],
                ]
                accum_fint(geo.dn_global, geo.voln, sig, rho0, st.ssp_eq, f_int, stig)

    return Q1NPForces(f_int=f_int, stig=stig, volg=volg, vgauss=vgauss, deltax=deltax)


__all__ = [
    "NBULK",
    "GaussPointGeometry",
    "GaussPointState",
    "GaussPointStress",
    "Q1NPBadJacobian",
    "Q1NPElement",
    "Q1NPForces",
    "StrainRate",
    "SurfacePoint",
    "accum_fint",
    "char_len",
    "ders_basis_funs",
    "eval_phys_point",
    "evaluate_shape_values",
    "evaluate_surface",
    "evaluate_top_surface_point",
    "evaluate_top_surface_point_and_derivs",
    "find_span",
    "forces",
    "gauss_1d",
    "gp_geometry",
    "jacobian",
    "shape_functions",
    "strain_rate",
]
