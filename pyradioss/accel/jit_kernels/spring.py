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
#   engine/source/elements/spring/r6def3.F (the axial channel driver body)
#   engine/source/elements/spring/redef3.F90 (the linear branch :739-743,
#     the damping close :1141-1146, the XL0 normalisation :270-279)
#   engine/source/elements/spring/r2len3.F (the dt close)

"""numba mirrors for the 1-D spring branches that iterate over *elements*.

Each kernel here mirrors, statement for statement, the per-element math
of one vectorised branch driver from Tasks P4.1–P4.5. READ THE DRIVER
FIRST — the formulas are documented there; this module only replays
them as compiled scalar loops. The M7 parity contract applies: same
formulas, same branch structure, same evaluation order, so the mirrors
agree BITWISE with the NumPy drivers (``tests/test_p4_numba_mirrors.py``
asserts ``tobytes()`` equality, not a tolerance).

Scope (plan P4.9 — element-iterating branches ONLY):

* ``spring_linear`` — the ``spring_type6.driver`` axial body (P4.1):
  the linear ``redef3`` channel with the damping close plus the
  ``r2len3`` dt close. The driver's reachable domain IS linear
  (``init_group`` refuses every nonlinear property key loudly), so
  this mirror covers the whole driver.
* ``spring_t32_linear`` — the pretensioner curve-free core (P4.4).
* ``spring_t8_linear`` — the TYPE8 closed-form core (P4.3).

Explicitly NOT mirrored (branches that iterate over *curves*, not
elements — lookup-bound, so compiling the loop buys nothing — plus
one uncompilable branch):

* the ``redef3`` nonlinear-elastic / hardening channels (``IECROU``
  with a deck curve) and the pretensioner ``ITYP 2/3/4`` curve laws:
  per-element ``FunctTable`` lookups over ragged curve tables;
* ``spring_python_funct`` (P4.5): evaluation iterates over curve
  abscissae, there is no element loop at all;
* ``spring_user`` (P4.2): the element body is an arbitrary Python
  callback — numba cannot compile it, so no mirror can exist.

``cache=True`` everywhere (the M7 rule: the one-time JIT cost is paid
once). No ``fastmath``, no ``parallel`` (deterministic scatter order —
the restart-chaining canary).
"""

from __future__ import annotations

import math

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


__all__ = ["spring_linear", "spring_t32_linear", "spring_t8_linear"]


@njit(cache=True)
def spring_linear(x1, x2, L0, dx_old, k, c, ak, ileng, F_old, dt, off,
                  mass, k_tan_old):
    """Mirror of ``spring_type6.driver`` — axial ``r6def3`` channel.

    Statement-for-statement with the driver body plus the linear
    ``redef3_channel`` path (``redef3.F90:739-743``) through the damping
    close (``:1141-1146``) and the ``r2len3`` dt close. Inputs are the
    driver's own per-element arrays (``x1``/``x2`` the nodal positions,
    ``dx_old`` the previous ``L - L0``, ``F_old`` the previous force,
    ``k_tan_old`` the previous tangent for switched-off elements).
    Returns ``(F, a, e_incr, k_tan, dt_elem)`` — ``a`` the current axis
    so the caller scatters ``F*a`` exactly like the driver.
    """
    m = x1.shape[0]
    F = np.empty(m)
    a = np.empty((m, 3))
    e_incr = np.empty(m)
    k_tan = np.empty(m)
    dt_elem = np.empty(m)
    dt11 = dt if dt != 0.0 else EP30

    for e in prange(m):
        # --- gather: _axis (norm3 associates left to right, like fastmath)
        d0 = x2[e, 0] - x1[e, 0]
        d1 = x2[e, 1] - x1[e, 1]
        d2 = x2[e, 2] - x1[e, 2]
        norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        degen = norm < EM20
        L = EM20 if degen else norm
        if degen:
            a[e, 0] = 1.0
            a[e, 1] = 0.0
            a[e, 2] = 0.0
        else:
            a[e, 0] = d0 / L
            a[e, 1] = d1 / L
            a[e, 2] = d2 / L

        alive = off[e] > 0.0
        if not alive:
            # driver ``continue``: force silent, no work booked, the old
            # tangent stands, no step claimed
            F[e] = 0.0
            e_incr[e] = 0.0
            k_tan[e] = k_tan_old[e]
            dt_elem[e] = EP30
            continue

        # --- XL0 normalisation (driver lines 186-187)
        use_strain = (ileng[e] != 0) and (L0[e] > EM20)
        xl0 = L0[e] if use_strain else 1.0

        # --- linear redef3 channel (redef3.F90:739-743, driver call site
        # defaults: ifunc=0, iecrou=0, b=ee=0, ff=1, gf3=1, off=1)
        dx = L - L0[e]
        dxn = dx / xl0
        dxo = dx_old[e] / xl0
        dvx = (dxn - dxo) / dt11
        f = k[e] * dxn
        kt = k[e]
        # damping close (redef3.F90:1141-1146); gf3=1 fires it always,
        # but b=ee=gx=gx2=0 so dfac is exactly ak
        ad = dvx if dvx >= 0.0 else -dvx
        dvv = ad if ad > 1.0 else 1.0
        dfac = ak[e] + 0.0 * math.log(dvv) + 0.0
        f = (dfac * f + c[e] * dvx + 0.0) * 1.0
        kt = dfac * kt
        e_incr[e] = (dxn - dxo) * (f + F_old[e]) * 0.5 * xl0
        F[e] = f
        k_tan[e] = kt

        # --- r2len3 dt close (driver lines 218-225)
        mass_e = mass[e] if mass[e] > EM20 else EM20
        kk = k[e] if k[e] > 0.0 else 0.0
        cc = c[e]
        pos_k = (k[e] > 0.0) and (mass[e] > 0.0)
        pure_c = (k[e] <= 0.0) and (cc > 0.0) and (mass[e] > 0.0)
        s = kk / mass_e if pos_k else 1.0
        omega = 2.0 * np.sqrt(s)
        if pos_k:
            km = kk * mass_e
            if km < EM20:
                km = EM20
            xi = cc / np.sqrt(km)
        else:
            xi = 0.0
        dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi * xi) - xi)
        if pure_c:
            cden = cc if cc > EM20 else EM20
            dt_c = 0.5 * mass_e / cden
        else:
            dt_c = EP30
        dt_elem[e] = dt_crit if pos_k else dt_c
    return F, a, e_incr, k_tan, dt_elem


@njit(cache=True)
def spring_t32_linear(x1, x2, v1, v2, F_old, stif0, stif1, f1, d1, ilock,
                      ityp, iact, uvar1_in, uvar2_in, uvar3_in, dt, off,
                      mass):
    """Mirror of ``forces_pretensioner_type32`` on the curve-free domain.

    Statement-for-statement with the driver for sensor-inactive elements
    (``ruser32.F`` lines 162-171) and active ``ITYP == 1`` elements
    (lines 173-202: stroke accumulation, elastic baseline, linear
    pretension law, ``ILOCK``). An ACTIVE element with ``ITYP 2/3/4``
    raises ``ValueError`` — those branches evaluate deck ``/FUNCT``
    curves per element (curve iteration, explicitly not mirrored by
    plan P4.9); inactive elements of any ``ITYP`` take the curve-free
    path exactly like the driver. Returns ``(F, uvar1, uvar2, uvar3,
    k_tan, de_incr, dt_elem)``; the caller adds ``de_incr`` into
    ``eint`` and scatters ``F`` along the axis, as the driver does.
    """
    m = x1.shape[0]
    dt_val = dt if dt > 0.0 else 0.0

    # loud refusal first (a partial scatter must never look complete)
    for e in range(m):
        if iact[e] and ityp[e] != 1:
            raise ValueError(
                "spring_t32_linear: active ITYP 2/3/4 evaluates a deck "
                "curve (no numba mirror by plan P4.9)")

    F = np.empty(m)
    uvar1 = np.empty(m)
    uvar2 = np.empty(m)
    uvar3 = np.empty(m)
    k_tan = np.empty(m)
    de_incr = np.empty(m)
    dt_elem = np.empty(m)

    for e in prange(m):
        # --- gather: L, axis, Ldot (same formulas as the driver)
        d0 = x2[e, 0] - x1[e, 0]
        d1_ = x2[e, 1] - x1[e, 1]
        d2 = x2[e, 2] - x1[e, 2]
        norm = np.sqrt(d0 * d0 + d1_ * d1_ + d2 * d2)
        degen = norm < EM20
        L = EM20 if degen else norm
        if degen:
            ax0, ax1, ax2 = 1.0, 0.0, 0.0
        else:
            ax0, ax1, ax2 = d0 / L, d1_ / L, d2 / L
        dv0 = v2[e, 0] - v1[e, 0]
        dv1 = v2[e, 1] - v1[e, 1]
        dv2 = v2[e, 2] - v1[e, 2]
        Ldot = dv0 * ax0 + dv1 * ax1 + dv2 * ax2

        f = F_old[e]
        X = uvar1_in[e]
        u2 = uvar2_in[e]
        u3 = uvar3_in[e]
        if not iact[e]:
            # Case 1: inactive sensor — STIF0 baseline only
            u2 = 0.0
            f = f + stif0[e] * dt_val * Ldot
            kt = stif0[e]
        else:
            # Case 2: first activation cycle re-arms the stroke ...
            if u2 == 0.0:
                X = 0.0
                u2 = 1.0
            # ... then stroke and elastic baseline accumulate
            X = X + dt_val * Ldot
            f = f + stif0[e] * dt_val * Ldot
            kt = stif0[e]
            # ITYP 1: linear pretensioner FF = F1 + STIF1 * X
            FF = f1[e] + stif1[e] * X
            if (f > FF) and (ilock[e] == 2):
                u3 = 1.0
            if (FF > 0.0) and (u3 == 0.0):
                f = FF if FF > f else f

        alive = off[e] > 0.0
        F[e] = f if alive else 0.0
        uvar1[e] = X
        uvar2[e] = u2
        uvar3[e] = u3
        k_tan[e] = kt
        de_incr[e] = (0.5 * (F_old[e] + f) * Ldot * dt_val
                      if alive else 0.0)

        # --- critical step (ruser32mat3.F: Kt = uvar4 = stif0 here)
        mass_e = mass[e] if mass[e] > EM20 else EM20
        kdt = kt if kt > 0.0 else 0.0
        pos_k = (kt > 0.0) and (mass[e] > 0.0)
        s = kdt / mass_e if pos_k else 1.0
        dt_crit = 2.0 / (2.0 * np.sqrt(s))
        if alive and pos_k:
            dt_elem[e] = dt_crit
        else:
            dt_elem[e] = EP30
    return F, uvar1, uvar2, uvar3, k_tan, de_incr, dt_elem


@njit(cache=True)
def spring_t8_linear(delta, dvl, theta_in, dwl, dt, e1, e2, e3, dvec,
                     L0_in, xk, xc, ak, nonlinear, dmin, dmax, ifail,
                     ifail2, israte, fcut, off_in, fired, usens, isflag,
                     iequil, dx_old_in, f_old_in, k_tan_in, e6_in, crit_in,
                     critnew_in, rerefd_in, dead_in, mass, inertia):
    """Mirror of ``spring_common.type8_forces`` on the closed-form domain.

    Statement-for-statement with the driver for elements whose six DOFs
    are all closed-form (``nonlinear[e, i]`` False everywhere): the
    linear ``k dx + c rate`` force, sensor gating (``fired`` pre-resolved
    by the caller — sensors live on Python objects numba cannot touch),
    rupture DLIM/CRIT with ISRATE smoothing and latching, theta
    accumulation, trapezoidal-free linear work booking, the ``fvec`` /
    moment assembly with the ``iequil`` arm correction, and the exact
    ``r2len3`` dt close. Any ``nonlinear`` DOF raises ``ValueError`` —
    curve/plastic channels iterate over deck tables per element and are
    explicitly not mirrored (plan P4.9). Returns ``(F, M, theta_new,
    deint, dx_old, f_old, k_tan, e6, off, dead, crit, critnew, L0,
    rerefd, dt_elem, fvec, mvec1, mvec2)``.
    """
    m = delta.shape[0]
    dt_val = dt if dt > 0.0 else 0.0

    for e in range(m):
        for i in range(6):
            if nonlinear[e, i]:
                raise ValueError(
                    "spring_t8_linear: a DOF needs a deck curve or "
                    "hardening channel (no numba mirror by plan P4.9)")

    F = np.empty((m, 3))
    Mx = np.empty((m, 3))
    theta_new = np.empty((m, 3))
    deint = np.empty(m)
    dx_old = np.empty((m, 6))
    f_old = np.empty((m, 6))
    k_tan = np.empty((m, 6))
    e6 = np.empty((m, 6))
    off = np.empty(m)
    dead = np.empty(m, dtype=np.bool_)
    crit_a = np.empty(m)
    critnew = np.empty(m)
    L0 = np.empty((m, 3))
    rerefd = np.empty(m, dtype=np.bool_)
    dt_elem = np.empty(m)
    fvec = np.empty((m, 3))
    mvec1 = np.empty((m, 3))
    mvec2 = np.empty((m, 3))

    for e in prange(m):
        for b in range(3):
            theta_new[e, b] = theta_in[e, b] + dwl[e, b] * dt_val
            L0[e, b] = L0_in[e, b]
        for i in range(6):
            dx_old[e, i] = dx_old_in[e, i]
            f_old[e, i] = f_old_in[e, i]
            k_tan[e, i] = k_tan_in[e, i]
            e6[e, i] = e6_in[e, i]
        rerefd[e] = rerefd_in[e]
        crit_a[e] = crit_in[e]
        critnew[e] = critnew_in[e]
        dead[e] = dead_in[e]
        off[e] = off_in[e]

        if dead_in[e]:
            # ruptured: the driver ``continue``s — forces stay zero, no
            # work is booked, every state array keeps its old value
            # (theta_new above is the one exception: it is updated by
            # the driver's vectorised pre-loop, outside the dead check)
            for b in range(3):
                F[e, b] = 0.0
                Mx[e, b] = 0.0
                fvec[e, b] = 0.0
                mvec1[e, b] = 0.0
                mvec2[e, b] = 0.0
            deint[e] = 0.0
            dt_elem[e] = EP30
            continue

        # --- sensor gating (r23sens3/r1sens3 flavour, type8_sensor_off)
        ue = usens[e]
        if ue == 0:
            of = 1.0
            reref = False
        elif isflag[e] == 2:
            if fired[e]:
                of = 1.0
                reref = True
            else:
                of = 0.0
                reref = False
        elif ue > 0:
            if fired[e]:
                of = 1.0
                reref = False
            else:
                of = 0.0
                reref = False
        else:
            if fired[e]:
                of = 0.0
                reref = False
            else:
                of = 1.0
                reref = False
        prev_off = off_in[e]
        off[e] = of
        if reref and prev_off <= 0.0 and not rerefd[e]:
            for b in range(3):
                L0[e, b] = L0[e, b] + delta[e, b]
                theta_new[e, b] = 0.0
            dx_old[e, 0] = 0.0
            dx_old[e, 1] = 0.0
            dx_old[e, 2] = 0.0
            rerefd[e] = True
            zeroed = True
        else:
            zeroed = False

        ifail2e = ifail2[e]
        ifaile = ifail[e]
        crit = 0.0
        de = 0.0
        for i in range(6):
            if zeroed:
                dx = 0.0
                dxo = 0.0
                rate = 0.0
            elif i < 3:
                dx = delta[e, i]
                dxo = dx_old[e, i]
                rate = dvl[e, i]
            else:
                j = i - 3
                dx = theta_new[e, j]
                dxo = theta_in[e, j]
                rate = dwl[e, j]
            # closed form (the driver ``else`` branch): linear k/c only
            k_raw = xk[e, i] * ak[e, i]
            c_raw = xc[e, i]
            f = (k_raw * dx + c_raw * rate) * of
            kti = xk[e, i] * ak[e, i]
            e_incr = f * rate * dt_val
            if i < 3:
                F[e, i] = f
                dx_old[e, i] = dx
            else:
                Mx[e, i - 3] = f
            f_old[e, i] = f
            k_tan[e, i] = kti
            e6[e, i] = e6[e, i] + e_incr
            de = de + e_incr
            if of == 1.0:
                # DLIM accumulates on live elements only (r2def3 gate)
                if ifail2e == 0:
                    if dx > 0.0 and dmax[e, i] < 1e29:
                        dlim = dx / dmax[e, i]
                    elif dx < 0.0 and dmin[e, i] > -1e29:
                        dlim = dx / dmin[e, i]
                    else:
                        dlim = 0.0
                elif ifail2e == 1:
                    if f > 0.0 and dmax[e, i] < 1e29:
                        dlim = f / dmax[e, i]
                    elif f < 0.0 and dmin[e, i] > -1e29:
                        dlim = f / dmin[e, i]
                    else:
                        dlim = 0.0
                elif ifail2e == 2:
                    if dmax[e, i] < 1e29:
                        pos_e6 = e6[e, i] if e6[e, i] > 0.0 else 0.0
                        dlim = pos_e6 / dmax[e, i]
                    else:
                        dlim = 0.0
                else:
                    dlim = 0.0
                if ifaile == 0:
                    crit = dlim if dlim > crit else crit
                else:
                    crit = crit + dlim * dlim
        deint[e] = de
        crit_a[e] = crit
        # ISRATE smoothing (type8_smooth_crit), latches at 1
        cn = critnew_in[e]
        if israte[e] != 0:
            fc = fcut[e] if fcut[e] != 0.0 else EP30
            alpha = (2.0 * np.pi * fc * dt_val) / (1.0 + 2.0 * np.pi * fc * dt_val)
            if cn < 1.0:
                c = crit if crit < 1.0 + 1e-3 else 1.0 + 1e-3
                sm = alpha * c + (1.0 - alpha) * cn
                new = sm if sm < 1.0 else 1.0
            else:
                new = 1.0
        else:
            if cn < 1.0:
                new = crit if crit < 1.0 else 1.0
            else:
                new = 1.0
        critnew[e] = new
        if of == 1.0 and new >= 1.0:
            dead[e] = True
            off[e] = 0.0
        else:
            dead[e] = False

        # --- assembly to the nodes (forces6 frame + iequil arm)
        f0 = F[e, 0]
        f1 = F[e, 1]
        f2 = F[e, 2]
        g0 = Mx[e, 0]
        g1 = Mx[e, 1]
        g2 = Mx[e, 2]
        for b in range(3):
            fvec[e, b] = f0 * e1[e, b] + f1 * e2[e, b] + f2 * e3[e, b]
            mv = g0 * e1[e, b] + g1 * e2[e, b] + g2 * e3[e, b]
            mvec1[e, b] = mv
            mvec2[e, b] = -mv
        if iequil[e] == 1:
            # MM = 0.5 * (d x fvec) added to BOTH nodes (r2cum3.F:125-140)
            a0 = 0.5 * (dvec[e, 1] * fvec[e, 2] - dvec[e, 2] * fvec[e, 1])
            a1 = 0.5 * (dvec[e, 2] * fvec[e, 0] - dvec[e, 0] * fvec[e, 2])
            a2 = 0.5 * (dvec[e, 0] * fvec[e, 1] - dvec[e, 1] * fvec[e, 0])
            mvec1[e, 0] = mvec1[e, 0] + a0
            mvec1[e, 1] = mvec1[e, 1] + a1
            mvec1[e, 2] = mvec1[e, 2] + a2
            mvec2[e, 0] = mvec2[e, 0] + a0
            mvec2[e, 1] = mvec2[e, 1] + a1
            mvec2[e, 2] = mvec2[e, 2] + a2

        # --- r2len3 dt close (exact Fortran damping reduction)
        dt_tr0 = EP30
        dt_tr1 = EP30
        dt_tr2 = EP30
        dt_ro0 = EP30
        dt_ro1 = EP30
        dt_ro2 = EP30
        me = mass[e] if mass[e] > 0.0 else 0.0
        ine = inertia[e] if inertia[e] > 0.0 else 0.0
        for i in range(3):
            kt = xk[e, i] if xk[e, i] > 0.0 else 0.0
            ct = xc[e, i] if xc[e, i] > 0.0 else 0.0
            if (me > 0.0) and ((kt > 0.0) or (ct > 0.0)):
                den = np.sqrt(ct * ct + me * kt) + ct
                if den < EM20:
                    den = EM20
                v = me / den
                if i == 0:
                    dt_tr0 = v
                elif i == 1:
                    dt_tr1 = v
                else:
                    dt_tr2 = v
            kr = xk[e, i + 3] if xk[e, i + 3] > 0.0 else 0.0
            cr = xc[e, i + 3] if xc[e, i + 3] > 0.0 else 0.0
            if (ine > 0.0) and ((kr > 0.0) or (cr > 0.0)):
                den = np.sqrt(cr * cr + ine * kr) + cr
                if den < EM20:
                    den = EM20
                v = ine / den
                if i == 0:
                    dt_ro0 = v
                elif i == 1:
                    dt_ro1 = v
                else:
                    dt_ro2 = v
        best = dt_tr0
        if dt_tr1 < best:
            best = dt_tr1
        if dt_tr2 < best:
            best = dt_tr2
        if dt_ro0 < best:
            best = dt_ro0
        if dt_ro1 < best:
            best = dt_ro1
        if dt_ro2 < best:
            best = dt_ro2
        dt_elem[e] = best
    return (F, Mx, theta_new, deint, dx_old, f_old, k_tan, e6, off, dead,
            crit_a, critnew, L0, rerefd, dt_elem, fvec, mvec1, mvec2)
