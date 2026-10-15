# Copyright (C) 2026 the pyradioss contributors.
#
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit FEM solver, and is licensed under the GNU Affero
# General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   engine/source/elements/shell/coqueba/cbacoor.F (added: the flat-pack
#     mid-sums, normalized half-diagonals, analytic Gauss Jacobians,
#     hourglass gradients and gama factors of lines 442-501 — the
#     geometry the operators below are built from)
#   engine/source/elements/shell/coqueba/cbadef.F (added: the flat-branch
#     membrane/bending rows of lines 79-105 and the transverse-shear BC
#     coefficients of lines 106-161 — restated here as stiffness rows
#     through the 13/24/HI slot maps)
#   engine/source/elements/shell/coqueba/cbaforc3.F (added: the CBADEFSH
#     constant assumed membrane shear and the CBAFORCT mean-shear force
#     correction whose exact linearization is mean_shear_block; upstream
#     states the same correction as CBALKEC3 in cbalke3.F lines 1635-1730)

"""QBAT implicit stiffness operators (P3.10).

``pyradioss/elements/shell_qbat.py`` is at its recorded line budget
(``tools/validation_data/size_budget.json``) and may only shrink, so the
per-Gauss-point strain-displacement rows live here and ``shell_qbat``
calls :func:`flat_gp_rows` / :func:`mean_shear_block`.

WHY NOT NAIVE SHAPE DERIVATIVES: the explicit flat kernel evaluates its
membrane/bending strains through the ``bm`` operator (constant part from
the normalized half-diagonals plus ``hxg``/``hyg`` hourglass-gradient
corrections), not through raw ``dN/dxi``.  On parallelograms both agree;
on distorted quads they differ by up to ~50 %, and the finite-difference
gate (``tests/test_p3_qbat_qeph_tangent.py``) fails with the naive rows.
The rows below transcribe the explicit operators (same slot maps), so the
tangent linearizes the assembly term by term.

The vdef3 assumed-shear row feeds ONLY :func:`mean_shear_block` (the
per-GP shear resultant is never assembled); ``Bm`` row 2 stays zero.
``mean_shear_block`` is the exact linearization of that correction.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20

#: 1/sqrt(3) — the Gauss abscissa PG of cbacoor.F/cbadef.F.
_PG = 0.577350269189626

#: 13/24-difference, hourglass and sum slot patterns over the 4 nodes.
_P13 = np.array([1.0, 0.0, -1.0, 0.0])
_P24 = np.array([0.0, 1.0, 0.0, -1.0])
_HPAT = np.array([1.0, -1.0, 1.0, -1.0])


def _shear_bc_ops(mx13, my13, mx23, my23, mx34, my34, detJ, ksi, eta):
    """Shear BC coefficients (cbadef.F 106-161), shared by ``_flat_gp`` and
    ``tangent()`` (P3.10): the stiffness linearizes the assembly term by
    term. Entries 0-11 ride the first beta set, 12-17 the second."""
    m = np.asarray(detJ).shape[0]
    a_1 = 0.25 / np.maximum(detJ, EM20)
    c11 = (my34 + my13 * ksi) * a_1
    c12 = -(my23 + my13 * eta) * a_1
    c21 = -(mx34 + mx13 * ksi) * a_1
    c22 = (mx23 + mx13 * eta) * a_1
    b1a = my13 + my23 * eta
    k1a = my13 + my34 * ksi
    b2a = mx13 + mx23 * eta
    k2a = mx13 + mx34 * ksi
    b1b = my23 + my13 * eta
    k1b = my34 + my13 * ksi
    b2b = mx23 + mx13 * eta
    k2b = mx34 + mx13 * ksi
    bc = np.empty((m, 18))
    bc[:, 0] = -c11 - c12
    bc[:, 1] = -c21 - c22
    bc[:, 2] = b1a * c11 + k1a * c12
    bc[:, 3] = b1a * c21 + k1a * c22
    bc[:, 4] = -b2a * c11 - k2a * c12
    bc[:, 5] = -b2a * c21 - k2a * c22
    bc[:, 6] = c11 - c12
    bc[:, 7] = c21 - c22
    bc[:, 8] = b1a * c11 - k1a * c12
    bc[:, 9] = b1a * c21 - k1a * c22
    bc[:, 10] = -b2a * c11 + k2a * c12
    bc[:, 11] = -b2a * c21 + k2a * c22
    bc[:, 12] = c11 * eta + c12 * ksi
    bc[:, 13] = c21 * eta + c22 * ksi
    bc[:, 14] = -b1b * c11 - k1b * c12
    bc[:, 15] = -b1b * c21 - k1b * c22
    bc[:, 16] = c11 * b2b + c12 * k2b
    bc[:, 17] = c21 * b2b + c22 * k2b
    return bc


def gauss_jac(cx, cy, area, g):
    """Analytic Gauss Jacobian at Gauss point ``g`` (cbacoor.F 478-486)."""
    mx13 = 0.5 * (cx[:, 0] + cx[:, 2])
    my13 = 0.5 * (cy[:, 0] + cy[:, 2])
    mx23 = 0.5 * (cx[:, 1] + cx[:, 2])
    my23 = 0.5 * (cy[:, 1] + cy[:, 2])
    mx34 = 0.5 * (cx[:, 2] + cx[:, 3])
    my34 = 0.5 * (cy[:, 2] + cy[:, 3])
    j1 = (mx23 * my13 - mx13 * my23) * _PG
    j2 = -(mx13 * my34 - mx34 * my13) * _PG
    j0 = 0.25 * area
    s2 = 1.0 if g < 2 else -1.0
    s1 = 1.0 if g in (1, 2) else -1.0
    return np.maximum(np.abs(j0 + s2 * j2 + s1 * j1), EM20)


def flat_gp_rows(cx, cy, area, g, xi_g, eta_g):
    """Per-Gauss-point (Bm, Bb, Bs, detJ) at Gauss point ``g``.

    ``cx``/``cy`` are centered corner coords (m,4), ``area`` (m,),
    ``xi_g``/``eta_g`` scalars. Flat-branch formulas throughout (warped
    elements reuse them — the same standing approximation as the previous
    tangent, now with the correct operators)."""
    m = np.shape(cx)[0]
    area_i = 1.0 / np.maximum(area, EM20)
    mx13 = 0.5 * (cx[:, 0] + cx[:, 2])
    my13 = 0.5 * (cy[:, 0] + cy[:, 2])
    mx23 = 0.5 * (cx[:, 1] + cx[:, 2])
    my23 = 0.5 * (cy[:, 1] + cy[:, 2])
    mx34 = 0.5 * (cx[:, 2] + cx[:, 3])
    my34 = 0.5 * (cy[:, 2] + cy[:, 3])
    x13 = 0.5 * (cx[:, 0] - cx[:, 2])
    y13 = 0.5 * (cy[:, 0] - cy[:, 2])
    x24 = 0.5 * (cx[:, 1] - cx[:, 3])
    y24 = 0.5 * (cy[:, 1] - cy[:, 3])
    x13n = x13 * area_i
    y13n = y13 * area_i
    x24n = x24 * area_i
    y24n = y24 * area_i
    gama1n = (-mx13 * y24 + my13 * x24) * area_i
    gama2n = (mx13 * y13 - my13 * x13) * area_i
    detJ = gauss_jac(cx, cy, area, g)
    j1h = (my23 - my34) * _PG
    j2h = -(my23 + my34) * _PG
    j1y = (mx34 - mx23) * _PG
    j2y = (mx34 + mx23) * _PG
    use1 = (g % 2 == 0)
    sh = 1.0 if g < 2 else -1.0
    hxg = np.where(use1, j1h, j2h) * sh / detJ
    hyg = np.where(use1, j1y, j2y) * sh / detJ
    # membrane/bending operator (cbadef.F 79-105 via _flat_gp)
    bm0 = y24n + hxg * gama1n
    bm1 = -y13n + hxg * gama2n
    bm2 = 0.25 * hxg
    bm4 = -x24n + hyg * gama1n
    bm5 = x13n + hyg * gama2n
    bm6 = 0.25 * hyg
    # transverse-shear BC operator (cbadef.F 106-161)
    bc = _shear_bc_ops(mx13, my13, mx23, my23, mx34, my34,
                       detJ, xi_g, eta_g)
    p13 = _P13[None, :]
    p24 = _P24[None, :]
    hi = _HPAT[None, :]
    Bm = np.zeros((m, 3, 24), dtype=float)
    Bm[:, 0, 0::6] = bm0[:, None] * p13 + bm1[:, None] * p24 + bm2[:, None] * hi
    Bm[:, 1, 1::6] = bm4[:, None] * p13 + bm5[:, None] * p24 + bm6[:, None] * hi
    # Bm row 2 (in-plane shear) stays zero: the explicit per-GP shear
    # resultant is never assembled (_fori_flat takes npg[0:2] only) — all
    # membrane shear stiffness is the mean operator (mean_shear_block).
    Bb = np.zeros((m, 3, 24), dtype=float)
    Bb[:, 0, 4::6] = bm0[:, None] * p13 + bm1[:, None] * p24 + bm2[:, None] * hi
    Bb[:, 1, 3::6] = -(bm4[:, None] * p13 + bm5[:, None] * p24 + bm6[:, None] * hi)
    Bb[:, 2, 3::6] = -(bm0[:, None] * p13 + bm1[:, None] * p24 + bm2[:, None] * hi)
    Bb[:, 2, 4::6] = bm4[:, None] * p13 + bm5[:, None] * p24 + bm6[:, None] * hi
    Bs = np.zeros((m, 2, 24), dtype=float)
    Bs[:, 0, 2::6] = bc[:, 0:1] * p13 + bc[:, 6:7] * p24 + bc[:, 12:13] * hi
    Bs[:, 1, 2::6] = bc[:, 1:2] * p13 + bc[:, 7:8] * p24 + bc[:, 13:14] * hi
    Bs[:, 0, 3::6] = bc[:, 2:3] * p13 + bc[:, 8:9] * p24 + bc[:, 14:15]
    Bs[:, 0, 4::6] = bc[:, 4:5] * p13 + bc[:, 10:11] * p24 + bc[:, 16:17]
    Bs[:, 1, 3::6] = bc[:, 3:4] * p13 + bc[:, 9:10] * p24 + bc[:, 15:16]
    Bs[:, 1, 4::6] = bc[:, 5:6] * p13 + bc[:, 11:12] * p24 + bc[:, 17:18]
    return Bm, Bb, Bs, detJ


def mean_shear_block(x13, y13, x24, y24, area, gshear, vol):
    """Constant membrane-shear stiffness (n,24,24), local frame: the exact
    linearization of the explicit CBAFORCT mean-shear force correction.

    ``th = volg * for_mean[2]`` with ``for_mean[2]`` the mean shear
    resultant gives ``d(th)/dU = volg * Gshear * S`` (``S`` the vdef3
    assumed-shear row); the flat slot-combine distributes it with the SAME
    pattern, so the block is an outer product. ``x13``… are UNNORMALIZED
    half-diagonal differences, ``area`` (n,), ``gshear`` (n,) the STRESS
    shear modulus, ``vol`` (n,) the element volume."""
    n = np.shape(x13)[0]
    ai = 1.0 / np.maximum(area, EM20)
    sx = np.stack([-x24 * ai, x13 * ai, x24 * ai, -x13 * ai], axis=1)
    sy = np.stack([y24 * ai, -y13 * ai, -y24 * ai, y13 * ai], axis=1)
    s = np.zeros((n, 24), dtype=float)
    s[:, 0::6] = sx
    s[:, 1::6] = sy
    c = (gshear * vol)[:, None]
    return c * s[:, :, None] * s[:, None, :]
