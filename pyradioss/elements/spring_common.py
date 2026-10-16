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
    lscale: float = 1.0,
    xl0: float = 1.0,
    iecrou: int = 0,
    ifunc: int = 0,
    ifunc2: int = 0,  # noqa: ARG001 — unloading-curve slot (IECROU 4/5/6 path, not yet ported)
    ifv: int = 0,
    ifunc3: int = 0,
    f_old: float = 0.0,
    dt: float = 0.0,
    off: float = 1.0,
    fn1: tuple[np.ndarray, np.ndarray] | None = None,
    fnv: tuple[np.ndarray, np.ndarray] | None = None,
    fnd: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """One ``redef3`` channel update; returns ``(force, energy_incr, k_tan)``.

    ``dx`` is the total channel displacement (``r6def3.F``: ``DX = ALDP -
    X0``), ``dx_old`` its value at the previous call.  The channel is
    normalised by ``XL0`` on entry (``redef3.F90:270-279``: ``XL0`` is the
    initial length when ``ILENG != 0``, else ``ONE``); deck-curve abscissae
    are additionally scaled by ``lscale`` (``GEO(39) = 1/LSCALE``,
    ``redef3.F90:705-707``) and the energy is scaled back on exit
    (:1148-1155), so the returned energy increment is the physical
    trapezoidal work ``(f_old + f)/2 * (dx - dx_old)`` (:1140-1147).
    ``fnv``/``fnd`` are the damping F-V curves (``FUN_B``/``FUN_D``,
    evaluated at ``dvx * ff`` per :1022-1077); each only acts when its id
    (``ifv``/``ifunc3``) is set, exactly like ``jdmp``/``j2dmp``.
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
        f, tg = interp_curve(fn1[0], fn1[1], dxn * lscale)
        k_tan = tg * lscale
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

    gx = interp_curve(fnv[0], fnv[1], dvx * ff)[0] if (ifv != 0 and fnv is not None) else 0.0
    gx2 = interp_curve(fnd[0], fnd[1], dvx * ff)[0] if (ifunc3 != 0 and fnd is not None) else 0.0
    if ifv != 0 or b != 0.0 or ee != 0.0 or xc != 0.0 or gf3 != 0.0:
        # Rate/damping close (redef3.F90:1141-1146): dfac is the logarithmic
        # rate factor plus the F-V curve term; gf3 weights the additional
        # damping curve.  Without curves gx = gx2 = 0 (:1131-1135).
        dvv = max(1.0, abs(dvx / d) if d != 0.0 else 1.0)
        dfac = ak + b * math.log(dvv) + ee * gx
        f = (dfac * f + xc * dvx + gf3 * gx2) * off
        k_tan = dfac * k_tan
    else:
        # Rate-independent close (redef3.F90:1147-1151): fx * ak * off.
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


#: DOF indices of the six TYPE8 channels (0-2 translations, 3-5 rotations).
TYPE8_NDOF = 6


def _curve_of(funcs, fid):
    """``(xs, ys)`` deck curve for ``fid`` or ``None`` (``redef3`` vinter).

    Only array curves are honest here: ``redef3`` interpolates the table
    (``vinter2`` — zero past the last point, see :func:`interp_curve`),
    which needs the abscissae.  Objects without ``x``/``y`` tables fall
    back to ``None`` and the caller keeps the closed form with a warning.
    """
    if not fid:
        return None
    fn = None
    try:
        fn = funcs.get(fid)
    except AttributeError:
        return None
    if fn is None:
        return None
    xs = getattr(fn, "x", None)
    ys = getattr(fn, "y", None)
    if xs is None or ys is None:
        return None
    return (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))


def type8_init(m):
    """Zeroed TYPE8 nonlinear state for ``m`` elements (one row per DOF)."""
    return {
        "fun": np.zeros((m, 6), dtype=np.int64),
        "iecrou": np.zeros((m, 6), dtype=np.int64),
        "funb": np.zeros((m, 6), dtype=np.int64),
        "func": np.zeros((m, 6), dtype=np.int64),
        "fund": np.zeros((m, 6), dtype=np.int64),
        "xk": np.zeros((m, 6)),
        "xc": np.zeros((m, 6)),
        "ak": np.ones((m, 6)),
        "bk": np.zeros((m, 6)),
        "dk": np.ones((m, 6)),
        "ee": np.zeros((m, 6)),
        "gf3": np.ones((m, 6)),
        "ls": np.ones((m, 6)),
        "ff": np.ones((m, 6)),
        "cv_a": np.full((m, 6), None, dtype=object),
        "cv_b": np.full((m, 6), None, dtype=object),
        "cv_d": np.full((m, 6), None, dtype=object),
        "f_old": np.zeros((m, 6)),
        "k_tan": np.zeros((m, 6)),
    }


def type8_fill(t8, local, p, funcs, log, pid):
    """Per-DOF constitutive params for one property slice (hm_read_prop08).

    Applies the reader defaults (``A/D/GF3/LSCALE = 1`` when blank,
    ``A = 1, B = E = 0`` when ``FUN_A = 0``) and resolves the ``FUN_A/B/D``
    deck curves, warning once per property when a referenced curve is
    absent (closed form kept) or the hardening flag is not yet ported
    (elastic fallback).  Later cuts add rupture/sensor/rate keys here.
    """
    local = np.asarray(local, dtype=np.int64)
    if len(local) == 0:
        return
    for i in range(TYPE8_NDOF):
        d = i + 1
        fun_a = int(p.get(f"fun_a{d}", 0) or 0)
        hflag = int(p.get(f"hflag{d}", 0) or 0)
        fun_b = int(p.get(f"fun_b{d}", 0) or 0)
        fun_c = int(p.get(f"fun_c{d}", 0) or 0)
        fun_d = int(p.get(f"fun_d{d}", 0) or 0)
        k_raw = float(p.get(f"k{d}", 0.0) or 0.0)
        if fun_a == 0:
            a_eff, b_eff, ee_eff = 1.0, 0.0, 0.0
        else:
            a_raw = float(p.get(f"a{d}", 0.0) or 0.0)
            a_eff = a_raw if a_raw != 0.0 else 1.0
            b_eff = float(p.get(f"b{d}", 0.0) or 0.0)
            e_raw = float(p.get(f"f0_{d}", 0.0) or 0.0)
            ee_eff = e_raw if e_raw != 0.0 else 1.0
        d_raw = float(p.get(f"d{d}", 0.0) or 0.0)
        h_raw = float(p.get(f"hscale{d}", 0.0) or 0.0)
        s_raw = float(p.get(f"scale{d}", 0.0) or 0.0)
        v_raw = float(p.get(f"v0_{d}", 0.0) or 0.0)
        t8["fun"][local, i] = fun_a
        t8["iecrou"][local, i] = hflag
        t8["funb"][local, i] = fun_b
        t8["func"][local, i] = fun_c
        t8["fund"][local, i] = fun_d
        t8["xk"][local, i] = k_raw / a_eff
        t8["xc"][local, i] = float(p.get(f"c{d}", 0.0) or 0.0)
        t8["ak"][local, i] = a_eff
        t8["bk"][local, i] = b_eff
        t8["dk"][local, i] = d_raw if d_raw != 0.0 else 1.0
        t8["ee"][local, i] = ee_eff
        t8["gf3"][local, i] = h_raw if h_raw != 0.0 else 1.0
        t8["ls"][local, i] = 1.0 / s_raw if s_raw != 0.0 else 1.0
        t8["ff"][local, i] = 1.0 / v_raw if v_raw != 0.0 else 1.0
        for slot, fid, store in (("A", fun_a, "cv_a"), ("B", fun_b, "cv_b"),
                                 ("D", fun_d, "cv_d")):
            if fid == 0:
                continue
            cv = _curve_of(funcs, fid)
            if cv is None and log is not None:
                log.warning(f"/PROP/TYPE8/{pid}: DOF{d} FUN_{slot}={fid} "
                            f"has no /FUNCT table — closed form kept",
                            "PROP CHECK")
            for e in local:
                t8[store][e, i] = cv
        if hflag != 0 and log is not None:
            log.warning(f"/PROP/TYPE8/{pid}: DOF{d} IECROU={hflag} not "
                        f"ported yet (redef3.F90:{_iecrou_line(hflag)}) — "
                        f"elastic response kept", "PROP CHECK")
        if fun_c != 0 and log is not None:
            log.warning(f"/PROP/TYPE8/{pid}: DOF{d} FUN_C={fun_c} needs "
                        f"IECROU 4/5/7 (not ported) — ignored", "PROP CHECK")


def type8_forces(delta, dvl, theta, dwl, dt_val, t8):
    """One ``r2def3``-style update over the six uncoupled DOFs.

    ``delta``/``dvl`` are the (m, 3) translation displacements/rates,
    ``theta``/``dwl`` the accumulated/rate rotations.  Each DOF runs the
    shared :func:`redef3_channel` (linear or ``FUN_A`` elastic curve plus
    the damping close); returns ``(F, M, theta_new, deint)`` with the
    trapezoidal work booked per channel exactly like ``redef3``.
    """
    m = delta.shape[0]
    F = np.zeros((m, 3))
    M = np.zeros((m, 3))
    deint = np.zeros(m)
    theta_new = theta + dwl * dt_val
    for e in range(m):
        for i in range(TYPE8_NDOF):
            if i < 3:
                dx, dxo = delta[e, i], delta[e, i] - dvl[e, i] * dt_val
            else:
                j = i - 3
                dx, dxo = theta_new[e, j], theta[e, j]
            fun_a = int(t8["fun"][e, i])
            cv = t8["cv_a"][e, i]
            if fun_a != 0 and cv is not None and int(t8["iecrou"][e, i]) == 0:
                ifunc, fn1 = fun_a, cv
            else:
                ifunc, fn1 = 0, None
            f, e_incr, kti = redef3_channel(
                dx, dxo, xk=float(t8["xk"][e, i]), xc=float(t8["xc"][e, i]),
                ak=float(t8["ak"][e, i]), b=float(t8["bk"][e, i]),
                d=float(t8["dk"][e, i]), ee=float(t8["ee"][e, i]),
                ff=float(t8["ff"][e, i]), gf3=float(t8["gf3"][e, i]),
                lscale=float(t8["ls"][e, i]), xl0=1.0,
                iecrou=0, ifunc=ifunc,
                ifv=int(t8["funb"][e, i]), ifunc3=int(t8["fund"][e, i]),
                f_old=float(t8["f_old"][e, i]), dt=dt_val, off=1.0,
                fn1=fn1, fnv=t8["cv_b"][e, i], fnd=t8["cv_d"][e, i])
            if i < 3:
                F[e, i] = f
            else:
                M[e, i - 3] = f
            t8["f_old"][e, i] = f
            t8["k_tan"][e, i] = kti
            deint[e] += e_incr
    return F, M, theta_new, deint
