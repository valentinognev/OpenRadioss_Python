"""
Catmull-Rom spline interpolation, knot calculation, arc length, and point projection.

Fortran source citations:
- cr_spline_interpol: $OR_SRC/common_source/tools/interpolation/cr_spline_interpol.F
- cr_spline_knots:    $OR_SRC/common_source/tools/interpolation/cr_spline_knots.F
- cr_spline_length:   $OR_SRC/common_source/tools/interpolation/cr_spline_length.F
- cr_spline_point_proj: $OR_SRC/common_source/tools/interpolation/cr_spline_point_proj.F

Theory notes:
- Barry and Goldman's pyramidal formulation for Catmull-Rom splines.
- Evaluates spline position C(t), 1st derivative C'(t), and 2nd derivative C''(t)
  on the segment between P1 and P2 for parameter t in [0, 1].
- Knots parameter alpha:
    alpha = 0.0 : uniform Catmull-Rom
    alpha = 0.5 : centripetal Catmull-Rom (avoids cusps and self-intersections)
    alpha = 1.0 : chordal Catmull-Rom
- Arc length calculation uses 20-step Simpson's 3rd-order quadrature.
- Point projection uses Newton-Raphson iteration on (C(t) - Z) . C'(t) = 0.
"""

from __future__ import annotations

import numpy as np


def cr_spline_knots(pts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Compute knots for 4-point Catmull-Rom spline.

    Ported from $OR_SRC/common_source/tools/interpolation/cr_spline_knots.F

    Parameters
    ----------
    pts : np.ndarray of shape (4, 3)
        The 4 control points P0, P1, P2, P3.
    alpha : float, optional
        Catmull-Rom parameter (0.0: uniform, 0.5: centripetal, 1.0: chordal). Default is 0.5.

    Returns
    -------
    knots : np.ndarray of shape (4,)
        Knots [t0, t1, t2, t3] with t0 = 0.0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    knots = np.zeros(4, dtype=np.float64)
    for i in range(1, 4):
        d = np.linalg.norm(pts[i] - pts[i - 1])
        if d <= 0.0:
            dt = 1.0 if alpha == 0.0 else 0.0
        elif alpha == 0.0:
            dt = 1.0
        else:
            dt = float(np.exp(alpha * np.log(d)))
        knots[i] = knots[i - 1] + dt
    return knots


def cr_spline_interpol(
    pts: np.ndarray, knots: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Catmull-Rom spline position and derivatives on segment [P1, P2].

    Ported from $OR_SRC/common_source/tools/interpolation/cr_spline_interpol.F

    Parameters
    ----------
    pts : np.ndarray of shape (4, 3)
        4 control points P0, P1, P2, P3.
    knots : np.ndarray of shape (4,)
        Knots [t0, t1, t2, t3].
    t : float
        Position in [0, 1] along segment [P1, P2] (t=0 -> P1, t=1 -> P2).

    Returns
    -------
    C : np.ndarray of shape (3,)
        Interpolated position on the spline curve.
    C_D : np.ndarray of shape (3,)
        First derivative dC/d(TT).
    C_DD : np.ndarray of shape (3,)
        Second derivative d^2C/d(TT)^2.
    """
    pts = np.asarray(pts, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)

    t0, t1, t2, t3 = knots[0], knots[1], knots[2], knots[3]
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]

    tt = t1 + t * (t2 - t1)

    dt10 = t1 - t0 if t1 != t0 else 1.0e-30
    dt21 = t2 - t1 if t2 != t1 else 1.0e-30
    dt32 = t3 - t2 if t3 != t2 else 1.0e-30
    dt20 = t2 - t0 if t2 != t0 else 1.0e-30
    dt31 = t3 - t1 if t3 != t1 else 1.0e-30

    a1 = ((t1 - tt) * p0 + (tt - t0) * p1) / dt10
    a2 = ((t2 - tt) * p1 + (tt - t1) * p2) / dt21
    a3 = ((t3 - tt) * p2 + (tt - t2) * p3) / dt32

    b1 = ((t2 - tt) * a1 + (tt - t0) * a2) / dt20
    b2 = ((t3 - tt) * a2 + (tt - t1) * a3) / dt31

    c = ((t2 - tt) * b1 + (tt - t1) * b2) / dt21

    a1p = (p1 - p0) / dt10
    a2p = (p2 - p1) / dt21
    a3p = (p3 - p2) / dt32

    b1p = (a2 - a1) / dt20 + ((t2 - tt) / dt20) * a1p + ((tt - t0) / dt20) * a2p
    b2p = (a3 - a2) / dt31 + ((t3 - tt) / dt31) * a2p + ((tt - t1) / dt31) * a3p

    c_d = (b2 - b1) / dt21 + ((t2 - tt) / dt21) * b1p + ((tt - t1) / dt21) * b2p

    b1pp = (2.0 * a2p - 2.0 * a1p) / dt20
    b2pp = (2.0 * a3p - 2.0 * a2p) / dt31

    c_dd = (2.0 * b2p - 2.0 * b1p + (t2 - tt) * b1pp + (tt - t1) * b2pp) / dt21

    return c, c_d, c_dd


def cr_spline_length(pts: np.ndarray, alpha: float = 0.5, t: float = 1.0, niter: int = 20) -> float:
    """Compute Catmull-Rom spline arc length using Simpson's rule.

    Ported from $OR_SRC/common_source/tools/interpolation/cr_spline_length.F

    Parameters
    ----------
    pts : np.ndarray of shape (4, 3)
        4 control points P0, P1, P2, P3.
    alpha : float, optional
        Catmull-Rom parameter (default 0.5).
    t : float, optional
        Upper parameter bound in [0, 1] (1.0 for full segment [P1, P2]). Default is 1.0.
    niter : int, optional
        Number of steps for Simpson quadrature (Fortran default is 20).

    Returns
    -------
    length : float
        Arc length of the curve from parameter 0 to t.
    """
    pts = np.asarray(pts, dtype=np.float64)
    knots = cr_spline_knots(pts, alpha)
    length = 0.0

    # Fortran: DO ITER=1, NITER-1 (NITER=20 -> 19 intervals)
    for i in range(1, niter):
        # First point of interval
        t_start = t * float(i - 1) / float(niter - 1)
        _, cd1, _ = cr_spline_interpol(pts, knots, t_start)
        val1 = float(np.linalg.norm(cd1))
        k1 = knots[1] + t_start * (knots[2] - knots[1])

        # Middle point of interval
        t_mid = (float(i - 1) / float(niter - 1) + 0.5 / float(niter - 1)) * t
        _, cd2, _ = cr_spline_interpol(pts, knots, t_mid)
        val2 = float(np.linalg.norm(cd2))

        # Last point of interval
        t_end = t * float(i) / float(niter - 1)
        _, cd3, _ = cr_spline_interpol(pts, knots, t_end)
        val3 = float(np.linalg.norm(cd3))
        k2 = knots[1] + t_end * (knots[2] - knots[1])

        length += (k2 - k1) / 6.0 * (val1 + 4.0 * val2 + val3)

    return length


def cr_spline_point_proj(
    pts: np.ndarray, z: np.ndarray, alpha: float = 0.5, max_iter: int = 20
) -> tuple[np.ndarray, float, float]:
    """Project 3D point Z onto Catmull-Rom spline segment [P1, P2].

    Ported from $OR_SRC/common_source/tools/interpolation/cr_spline_point_proj.F

    Parameters
    ----------
    pts : np.ndarray of shape (4, 3)
        4 control points P0, P1, P2, P3.
    z : np.ndarray of shape (3,)
        Point to project.
    alpha : float, optional
        Catmull-Rom parameter (default 0.5).
    max_iter : int, optional
        Maximum Newton-Raphson iterations (Fortran default 20).

    Returns
    -------
    zh : np.ndarray of shape (3,)
        Closest point on the spline curve.
    h : float
        Distance from Z to Zh: ||Zh - Z||.
    t : float
        Parameter position in [0, 1] corresponding to Zh.
    """
    pts = np.asarray(pts, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    knots = cr_spline_knots(pts, alpha)

    t = 0.5
    for _ in range(max_iter):
        c, cd, cdd = cr_spline_interpol(pts, knots, t)
        diff = c - z
        f = float(np.dot(diff, cd))
        f_d = float(np.dot(cd, cd) + np.dot(diff, cdd))
        if f_d != 0.0:
            t = t - f / f_d
        else:
            if f == 0.0:
                break
            # Fallback if derivative vanishes
            break

        if t < 0.0:
            t = 1.0e-10
        elif t > 1.0:
            t = 1.0 - 1.0e-10

    c, cd, cdd = cr_spline_interpol(pts, knots, t)
    zh = c
    h = float(np.linalg.norm(zh - z))
    return zh, h, t
