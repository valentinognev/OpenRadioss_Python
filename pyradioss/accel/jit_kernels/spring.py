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


__all__ = ["spring_linear"]


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
