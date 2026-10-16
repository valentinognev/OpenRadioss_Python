# Copyright (C) 2026 the pyradioss contributors.
#
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit FEM solver, and is licensed under the GNU Affero
# General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   engine/source/elements/beam/pforc3.F, pcoor3.F, pdefo3.F, pmat3.F,
#   pfint3.F, pdlen3.F (the TYPE3 beam cycle path)

"""numba mirrors for the 1-D beam branches that iterate over *elements*.

Companion to :mod:`jit_kernels.spring`: same M7 parity contract
(bitwise against the NumPy driver), same rules (``cache=True``, no
``fastmath``, no ``parallel``). READ THE DRIVER FIRST.

Scope (plan P4.9 — element-iterating branches ONLY):

* ``beam_type3_elastic`` — the elastic (``LAW != 2``) core of
  ``beam_type3._forces_core``: corotational frame, rate kinematics,
  elastic resultant update, internal forces/moments, midpoint energy,
  and the length-rescaled dt claim. Frame, rates, resultants, energy
  and dt are BITWISE; the global back-transform (``E @ local``) is
  contracted by NumPy's einsum with SIMD unrolling — measured
  ``(t0+t2)+t1`` association, unreproducible by any portable scalar
  order — so it is proven at the M7 kernel tolerance (``rtol=1e-12``;
  measured ``<= 5e-16``).

Explicitly NOT mirrored: the LAW2 global-plasticity return (per-slice
Python material dispatch around a Newton loop over material-object
state — a slice-iterating branch, deferred like every other
material-law loop in the M7 backend).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover — NumPy fallback path
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def dec(fn):
            return fn

        return dec

    prange = range

from pyradioss.common.constants import EM20, EP30


from pyradioss.common.constants import EM20, EP30


__all__ = ["beam_type3_elastic"]


@njit(cache=True)
def beam_type3_elastic(x1, x2, x3, vn1, vn2, vrn1, vrn2, area, ixx, iyy,
                       izz, emod, gmod, fres_old, mres_old, L0, dt0, off,
                       dt, plastic):
    """Mirror of ``beam_type3._forces_core`` on the elastic domain.

    Statement-for-statement with the driver for ``LAW != 2`` elements:
    the ``pcoor3`` corotational frame (including the degenerate
    fallback), ``pdefo3`` rate kinematics, the elastic ``pmat3`` update,
    the ``pfint3`` internal forces/moments, midpoint energy booking,
    and the length-rescaled ``pdlen3`` dt claim. Any ``plastic``
    element raises ``ValueError`` — the LAW2 radial return dispatches
    per part slice over Python material objects (a slice-iterating
    branch, deferred like every material-law loop in the M7 backend).
    Returns ``(fg1, fg2, mg1, mg2, fres_new, mres_new, deint,
    dt_elem)`` — the exact node-1/node-2 scatter operands (the driver
    adds ``fg`` on N1 / ``-fg`` on N2, ``-m`` on both).
    """
    m = x1.shape[0]

    for e in range(m):
        if plastic[e]:
            raise ValueError(
                "beam_type3_elastic: LAW2 needs the global-plasticity "
                "return (no numba mirror by plan P4.9)")

    fg1 = np.empty((m, 3))
    fg2 = np.empty((m, 3))
    mg1 = np.empty((m, 3))
    mg2 = np.empty((m, 3))
    fres_new = np.empty((m, 3))
    mres_new = np.empty((m, 3))
    deint = np.empty(m)
    dt_elem = np.empty(m)

    for e in prange(m):
        # --- pcoor3 frame (beam_type3._frame, scalar transcription)
        d0 = x2[e, 0] - x1[e, 0]
        d1 = x2[e, 1] - x1[e, 1]
        d2 = x2[e, 2] - x1[e, 2]
        nd = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        deg_d = nd <= EM20
        L = nd if nd > EM20 else EM20
        e1x = d0 / L
        e1y = d1 / L
        e1z = d2 / L
        y0 = x3[e, 0] - x1[e, 0]
        y1 = x3[e, 1] - x1[e, 1]
        y2 = x3[e, 2] - x1[e, 2]
        dot = y0 * e1x + y1 * e1y + y2 * e1z
        q0 = y0 - dot * e1x
        q1 = y1 - dot * e1y
        q2 = y2 - dot * e1z
        ne2 = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2)
        deg = (ne2 <= 1e-12) or deg_d
        if deg:
            if abs(e1x) < 0.9:
                rx, ry, rz = 1.0, 0.0, 0.0
            else:
                rx, ry, rz = 0.0, 1.0, 0.0
            rd = rx * e1x + ry * e1y + rz * e1z
            c0 = rx - rd * e1x
            c1 = ry - rd * e1y
            c2 = rz - rd * e1z
            nn = np.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
            if nn > 1e-12:
                q0 = c0 / nn
                q1 = c1 / nn
                q2 = c2 / nn
                ne2 = 1.0
            else:
                if abs(e1y) < 0.9:
                    q0, q1, q2 = 0.0, 1.0, 0.0
                else:
                    q0, q1, q2 = 0.0, 0.0, 1.0
                ne2 = 1.0
        den = ne2 if ne2 > EM20 else EM20
        e2x = q0 / den
        e2y = q1 / den
        e2z = q2 / den
        # cross3(e1, e2) component-by-component (fastmath.cross3)
        e3x = e1y * e2z - e1z * e2y
        e3y = e1z * e2x - e1x * e2z
        e3z = e1x * e2y - e1y * e2x
        if deg_d:
            e1x, e1y, e1z = 1.0, 0.0, 0.0
            e2x, e2y, e2z = 0.0, 1.0, 0.0
            e3x, e3y, e3z = 0.0, 0.0, 1.0
        # E[b, a]: columns are e1 | e2 | e3
        E00, E10, E20 = e1x, e1y, e1z
        E01, E11, E21 = e2x, e2y, e2z
        E02, E12, E22 = e3x, e3y, e3z

        # --- pdefo3: local velocities then rates (einsum order: b runs 0..2)
        v1x = vn1[e, 0] * E00 + vn1[e, 1] * E10 + vn1[e, 2] * E20
        v1y = vn1[e, 0] * E01 + vn1[e, 1] * E11 + vn1[e, 2] * E21
        v1z = vn1[e, 0] * E02 + vn1[e, 1] * E12 + vn1[e, 2] * E22
        v2x = vn2[e, 0] * E00 + vn2[e, 1] * E10 + vn2[e, 2] * E20
        v2y = vn2[e, 0] * E01 + vn2[e, 1] * E11 + vn2[e, 2] * E21
        v2z = vn2[e, 0] * E02 + vn2[e, 1] * E12 + vn2[e, 2] * E22
        t1x = vrn1[e, 0] * E00 + vrn1[e, 1] * E10 + vrn1[e, 2] * E20
        t1y = vrn1[e, 0] * E01 + vrn1[e, 1] * E11 + vrn1[e, 2] * E21
        t1z = vrn1[e, 0] * E02 + vrn1[e, 1] * E12 + vrn1[e, 2] * E22
        t2x = vrn2[e, 0] * E00 + vrn2[e, 1] * E10 + vrn2[e, 2] * E20
        t2y = vrn2[e, 0] * E01 + vrn2[e, 1] * E11 + vrn2[e, 2] * E21
        t2z = vrn2[e, 0] * E02 + vrn2[e, 1] * E12 + vrn2[e, 2] * E22
        invL = 1.0 / L
        eps = (v2x - v1x) * invL
        gy = (v2y - v1y) * invL - 0.5 * (t1z + t2z)
        gz = (v2z - v1z) * invL + 0.5 * (t1y + t2y)
        kx = (t2x - t1x) * invL
        ky = (t2y - t1y) * invL
        kz = (t2z - t1z) * invL

        # --- pmat3 elastic update (driver per-slice body, LAW1 here)
        EA = emod[e] * area[e]
        GA = gmod[e] * area[e]
        GI = gmod[e] * ixx[e]
        EIy = emod[e] * iyy[e]
        EIz = emod[e] * izz[e]
        N = fres_old[e, 0] + EA * eps * dt
        Qy = fres_old[e, 1] + GA * gy * dt
        Qz = fres_old[e, 2] + GA * gz * dt
        Mx_ = mres_old[e, 0] + GI * kx * dt
        My = mres_old[e, 1] + EIy * ky * dt
        Mz = mres_old[e, 2] + EIz * kz * dt
        fres_new[e, 0] = N
        fres_new[e, 1] = Qy
        fres_new[e, 2] = Qz
        mres_new[e, 0] = Mx_
        mres_new[e, 1] = My
        mres_new[e, 2] = Mz

        # --- pfint3 internal forces/moments (alive-gated)
        alive = off[e] > 0.0
        al = 1.0 if alive else 0.0
        f2x = N * al
        f2y = Qy * al
        f2z = Qz * al
        hL = 0.5 * L
        m1x = -Mx_ * al
        m1y = (-My + Qz * hL) * al
        m1z = (-Mz - Qy * hL) * al
        m2x = Mx_ * al
        m2y = (My + Qz * hL) * al
        m2z = (Mz - Qy * hL) * al

        # --- midpoint energy (before scatter, like the driver)
        f0m = 0.5 * (fres_old[e, 0] + N)
        f1m = 0.5 * (fres_old[e, 1] + Qy)
        f2m = 0.5 * (fres_old[e, 2] + Qz)
        g0m = 0.5 * (mres_old[e, 0] + Mx_)
        g1m = 0.5 * (mres_old[e, 1] + My)
        g2m = 0.5 * (mres_old[e, 2] + Mz)
        pw = (f0m * eps + f1m * gy + f2m * gz
              + g0m * kx + g1m * ky + g2m * kz)
        deint[e] = L * dt * pw if alive else 0.0

        # --- back to global axes (driver einsums, b outer / a inner)
        gx = f2x * E00 + f2y * E01 + f2z * E02
        gy_ = f2x * E10 + f2y * E11 + f2z * E12
        gz_ = f2x * E20 + f2y * E21 + f2z * E22
        fg1[e, 0] = gx
        fg1[e, 1] = gy_
        fg1[e, 2] = gz_
        fg2[e, 0] = -gx
        fg2[e, 1] = -gy_
        fg2[e, 2] = -gz_
        h1x = m1x * E00 + m1y * E01 + m1z * E02
        h1y = m1x * E10 + m1y * E11 + m1z * E12
        h1z = m1x * E20 + m1y * E21 + m1z * E22
        mg1[e, 0] = -h1x
        mg1[e, 1] = -h1y
        mg1[e, 2] = -h1z
        h2x = m2x * E00 + m2y * E01 + m2z * E02
        h2y = m2x * E10 + m2y * E11 + m2z * E12
        h2z = m2x * E20 + m2y * E21 + m2z * E22
        mg2[e, 0] = -h2x
        mg2[e, 1] = -h2y
        mg2[e, 2] = -h2z

        # --- pdlen3: exact-dt0 rescaled with the current length
        L0e = L0[e] if L0[e] > EM20 else EM20
        ratio = L / L0e
        if ratio < 1.0:
            dt_c = dt0[e] * ratio
        else:
            dt_c = dt0[e] * ratio ** -0.5
        dt_elem[e] = dt_c if alive else EP30
    return fg1, fg2, mg1, mg2, fres_new, mres_new, deint, dt_elem
