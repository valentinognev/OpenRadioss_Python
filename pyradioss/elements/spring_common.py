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
#   engine/source/elements/spring/redef3.F90  (the shared single-channel
#     constitutive update: linear branch redef3.F90:739-743, elastic-curve
#     branch :757, the rate/damping close :1140-1147, the XL0 normalisation
#     :270-279 and rescale :1148-1155)
#   engine/source/elements/spring/repla3.F  (the coupled-plasticity return,
#     transcribed statement-for-statement)
#   engine/source/tools/curve/interp.F  (INTERP value+slope, including the
#     beyond-last-point ZERO return :52)

"""Shared 1-D spring constitutive channels (``redef3``/``repla3``).

``redef3.F90`` is called by ``r1def3``, ``r2def3``, ``r3def3``, ``r4def3``
and ``r6def3`` — one channel per spring DOF.  It is factored here **once**
so the branches share it instead of drifting in per-branch copies
(Task P4.1 reviewer checklist).

Scope: the linear path (``IFUNC == 0``) and the nonlinear-elastic curve
path (``IECROU == 0`` with a deck curve), plus the rate/damping close and
the trapezoidal energy booking.  Every other ``IECROU`` value raises
``NotImplementedError`` naming the Fortran branch — a loud refusal, never
a silent linearisation.  (``spring_mat._update_dof_redef3`` already covers
the hardening flags 1/2/4/7 for the TYPE23 family; re-pointing that caller
at this module is later work, not this task's.)
"""

from __future__ import annotations

import math

import numpy as np

from ..common.constants import EM20, EP30


def interp_curve(xs: np.ndarray, ys: np.ndarray, x: float) -> tuple[float, float]:
    """Value and slope of a deck curve at ``x`` (``interp.F`` INTERP).

    Below the first abscissa the first segment is extrapolated; above the
    last abscissa the Fortran returns ``F = 0, TG = 0`` (``interp.F:52``)
    and so do we.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(ys[0]), 0.0
    if x > xs[-1]:
        return 0.0, 0.0
    m = 1
    while m < n - 1 and x > xs[m]:
        m += 1
    d1 = xs[m] - xs[m - 1]
    tg = 0.0 if d1 == 0.0 else (ys[m] - ys[m - 1]) / d1
    dt1 = x - xs[m - 1]
    dt2 = xs[m] - x
    f = ys[m - 1] + dt1 * tg if dt1 <= dt2 else ys[m] - dt2 * tg
    return float(f), float(tg)


def redef3_channel(
    dx: float,
    dx_old: float,
    *,
    xk: float,
    xc: float = 0.0,
    ak: float = 1.0,
    b: float = 0.0,
    d: float = 1.0,
    ee: float = 0.0,
    ff: float = 1.0,
    gf3: float = 0.0,
    lscale: float = 1.0,  # noqa: ARG001 — GEO(39) slot, applied by the caller curve scaling
    xl0: float = 1.0,
    iecrou: int = 0,
    ifunc: int = 0,
    ifunc2: int = 0,  # noqa: ARG001 — unloading-curve slot (IECROU 4/5/6 path, not yet ported)
    ifv: int = 0,
    ifunc3: int = 0,  # noqa: ARG001 — second-damping-curve slot (gx2 path, not yet ported)
    f_old: float = 0.0,
    dt: float = 0.0,
    off: float = 1.0,
    fn1: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """One ``redef3`` channel update; returns ``(force, energy_incr, k_tan)``.

    ``dx`` is the total channel displacement (``r6def3.F``: ``DX = ALDP -
    X0``), ``dx_old`` its value at the previous call.  The channel is
    normalised by ``XL0`` on entry (``redef3.F90:270-279``: ``XL0`` is the
    initial length when ``ILENG != 0``, else ``ONE``) and the energy is
    scaled back on exit (:1148-1155), so the returned energy increment is
    the physical trapezoidal work ``(f_old + f)/2 * (dx - dx_old)``
    (:1140-1147).
    """
    xl = xl0 if abs(xl0) >= EM20 else math.copysign(EM20, xl0 if xl0 != 0.0 else 1.0)
    dxn = dx / xl
    dxo = dx_old / xl
    dt11 = dt if dt != 0.0 else EP30
    dvx = (dxn - dxo) / dt11

    if ifunc == 0:
        # Linear elastic (redef3.F90:739-743): fx = xk * dx.
        f = xk * dxn
        k_tan = xk
    elif iecrou == 0 and fn1 is not None:
        # Nonlinear elastic, f = f(total displacement) (redef3.F90:757).
        f, tg = interp_curve(fn1[0], fn1[1], dxn)
        k_tan = tg
    elif iecrou == 0:
        raise NotImplementedError(
            "redef3 IECROU=0 with IFUNC != 0 needs the deck curve "
            "(redef3.F90:757, vinter path); no curve was supplied"
        )
    else:
        raise NotImplementedError(
            f"redef3 IECROU={iecrou} is not factored here yet "
            f"(redef3.F90:{_iecrou_line(iecrou)}); refusing loudly"
        )

    if ifv != 0 or b != 0.0 or ee != 0.0 or xc != 0.0 or gf3 != 0.0:
        # Rate/damping close (redef3.F90:1140-1143): without damping curves
        # gx = gx2 = 0 (:1131-1135), so dfac is the logarithmic rate factor.
        dvv = max(1.0, abs(dvx / d) if d != 0.0 else 1.0)
        dfac = ak + b * math.log(dvv)
        f = (dfac * f + xc * dvx) * off
        k_tan = dfac * k_tan
    else:
        # Rate-independent close (redef3.F90:1144-1147): fx * ak * off.
        f = f * ak * off
        k_tan = k_tan * ak

    e_incr = (dxn - dxo) * (f + f_old) * 0.5 * xl
    return float(f), float(e_incr), float(k_tan)


def _iecrou_line(iecrou: int) -> str:
    """Approximate ``redef3.F90`` line span owning an ``IECROU`` branch."""
    return {
        1: "762-775", 2: "779-797", 3: "799-812", 4: "814-840",
        5: "842-900", 6: "486-538", 7: "540-580", 8: "746-752",
        9: "581-686", 12: "687-696",
    }.get(iecrou, "405-442")


def repla3_correct(
    xk: float,
    dpx: float,
    curve: tuple[np.ndarray, np.ndarray],
    iecrou: int,
    ifunc: int,
    ifv: int,  # noqa: ARG001 — signature slot (repla3.F dummy arg, unused in the body)
    epla: float,
) -> float:
    """Coupled-plasticity return (``repla3.F``, statement-for-statement).

    Only fires for ``IFUNC != 0 .AND. IECROU == 3`` (tension-compression
    coupled elasto-plasticity); otherwise ``dpx`` passes through untouched.
    ``curve`` is the ``(xs, ys)`` deck function ``TF(NPF(IFUNC))``.
    """
    if ifunc != 0 and iecrou == 3:
        fp, xki = interp_curve(curve[0], curve[1], dpx)
        fp = fp * xk / max(EM20, (xk - xki))
        fp = max(abs(fp), EM20)
        dpx = dpx + epla / fp
    return float(dpx)
