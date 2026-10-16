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
        f, k_tan = _rate_close(f, k_tan, dvx, ak=ak, b=b, d=d, ee=ee,
                               xc=xc, gf3=gf3, gx=gx, gx2=gx2, off=off)
    else:
        # Rate-independent close (redef3.F90:1147-1151): fx * ak * off.
        f = f * ak * off
        k_tan = k_tan * ak

    e_incr = (dxn - dxo) * (f + f_old) * 0.5 * xl
    return float(f), float(e_incr), float(k_tan)


def _rate_close(f, k_tan, dvx, *, ak, b, d, ee, xc, gf3, gx, gx2, off):
    """Shared damping close (``redef3.F90:1141-1146``).

    ``dvv = max(1, |dvx/d|)``; ``dfac = ak + b ln(dvv) + ee gx``;
    ``f = (dfac f + xc dvx + gf3 gx2) off``.  Used by the elastic channel
    and every hardening channel alike — one transcription, not one copy
    per branch.
    """
    dvv = max(1.0, abs(dvx / d) if d != 0.0 else 1.0)
    dfac = ak + b * math.log(dvv) + ee * gx
    return (dfac * f + xc * dvx + gf3 * gx2) * off, dfac * k_tan


def redef3_plastic(
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
    iecrou: int,
    fn1: tuple[np.ndarray, np.ndarray],
    ifv: int = 0,
    ifunc3: int = 0,
    fnv: tuple[np.ndarray, np.ndarray] | None = None,
    fnd: tuple[np.ndarray, np.ndarray] | None = None,
    dpx: float = 0.0,
    dpx2: float = 0.0,
    f_ep: float = 0.0,
    f_old: float = 0.0,
    dt: float = 0.0,
    off: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    """Elasto-plastic channel update (``redef3.F90`` hardening branches).

    Returns ``(force, energy_incr, k_tan, dpx, dpx2, f_ep)``.  Only
    ``IECROU = 1`` (isotropic, :763-776: trial ``fxep + xk ddx`` capped at
    the loading curve, ``dpx`` takes the overstress) and ``IECROU = 2``
    (decoupled tension/compression, :780-798: independent ``dpx``/``dpx2``
    memories, zero force in the gap) are factored; every other flag keeps
    the loud refusal of :func:`redef3_channel`.  The damping close and the
    trapezoidal work booking are shared with the elastic channel.
    """
    xl = xl0 if abs(xl0) >= EM20 else math.copysign(EM20, xl0 if xl0 != 0.0 else 1.0)
    dxn = dx / xl
    dxo = dx_old / xl
    dt11 = dt if dt != 0.0 else EP30
    dvx = (dxn - dxo) / dt11

    if iecrou == 1:
        f_trial = f_ep + xk * (dxn - dxo)
        if f_trial >= 0.0:
            xx = dpx + f_trial / xk
        else:
            xx = -dpx + f_trial / xk
        yy, tg = interp_curve(fn1[0], fn1[1], xx * lscale)
        if f_trial >= 0.0 and f_trial > yy:
            dpx = dpx + (f_trial - yy) / xk
            f, capped = yy, True
        elif f_trial < 0.0 and f_trial < yy:
            dpx = dpx + (yy - f_trial) / xk
            f, capped = yy, True
        else:
            f, capped = f_trial, False
        f_ep = f
        k_tan = tg * lscale if capped else xk
    elif iecrou == 2:
        yy, tg = interp_curve(fn1[0], fn1[1], dxn * lscale)
        if dxn > dpx:
            f_trial = xk * (dxn - dpx)
            if f_trial > yy:
                f, capped = yy, True
            else:
                f, capped = f_trial, False
            dpx = dxn - f / xk
        elif dxn < dpx2:
            f_trial = xk * (dxn - dpx2)
            if f_trial < yy:
                f, capped = yy, True
            else:
                f, capped = f_trial, False
            dpx2 = dxn - f / xk
        else:
            f, capped = 0.0, False
        k_tan = tg * lscale if capped else (xk if f != 0.0 else 0.0)
    else:
        raise NotImplementedError(
            f"redef3 IECROU={iecrou} is not factored here yet "
            f"(redef3.F90:{_iecrou_line(iecrou)}); refusing loudly"
        )

    gx = interp_curve(fnv[0], fnv[1], dvx * ff)[0] if (ifv != 0 and fnv is not None) else 0.0
    gx2 = interp_curve(fnd[0], fnd[1], dvx * ff)[0] if (ifunc3 != 0 and fnd is not None) else 0.0
    if ifv != 0 or b != 0.0 or ee != 0.0 or xc != 0.0 or gf3 != 0.0:
        f, k_tan = _rate_close(f, k_tan, dvx, ak=ak, b=b, d=d, ee=ee,
                               xc=xc, gf3=gf3, gx=gx, gx2=gx2, off=off)
    else:
        f = f * ak * off
        k_tan = k_tan * ak

    e_incr = (dxn - dxo) * (f + f_old) * 0.5 * xl
    return float(f), float(e_incr), float(k_tan), float(dpx), float(dpx2), float(f_ep)


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
        "dpx": np.zeros((m, 6)),
        "dpx2": np.zeros((m, 6)),
        "f_ep": np.zeros((m, 6)),
        "dx_old": np.zeros((m, 6)),
        "f_old": np.zeros((m, 6)),
        "k_tan": np.zeros((m, 6)),
        "e6": np.zeros((m, 6)),
        "dmin": np.full((m, 6), -1e30),
        "dmax": np.full((m, 6), 1e30),
        "ifail": np.zeros(m, dtype=np.int64),
        "ifail2": np.zeros(m, dtype=np.int64),
        "off": np.ones(m),
        "dead": np.zeros(m, dtype=bool),
        "crit": np.zeros(m),
        "critnew": np.zeros(m),
        "israte": np.zeros(m, dtype=np.int64),
        "fcut": np.zeros(m),
        "sens": np.zeros(m, dtype=np.int64),
        "isflag": np.zeros(m, dtype=np.int64),
    }


def type8_fill(t8, local, p, funcs, log, pid):
    """Per-DOF constitutive params for one property slice (hm_read_prop08).

    Applies the reader defaults (``A/D/GF3/LSCALE = 1`` when blank,
    ``A = 1, B = E = 0`` when ``FUN_A = 0``) and resolves the ``FUN_A/B/D``
    deck curves, warning once per property when a referenced curve is
    absent (closed form kept) or the hardening flag is not yet ported
    (elastic fallback), plus the rupture limits and failure modes.
    Sensor and rate keys arrive with their cuts.
    """
    local = np.asarray(local, dtype=np.int64)
    if len(local) == 0:
        return
    # header-level failure modes (hm_read_prop08: IFAIL2 outside {0,1,2}
    # is clamped to displacement at read time, :153)
    ifail = int(p.get("ifail", 0) or 0)
    ifail2 = int(p.get("ifail2", 0) or 0)
    if ifail2 not in (0, 1, 2):
        ifail2 = 0
    t8["ifail"][local] = ifail
    t8["ifail2"][local] = ifail2
    t8["israte"][local] = int(p.get("israte", 0) or 0)
    t8["fcut"][local] = float(p.get("asrate", 0.0) or 0.0)
    sens_raw = int(p.get("sens_id", p.get("isensor", 0)) or 0)
    isflag = int(p.get("isflag", 0) or 0)
    # hm_read_prop08: IFL == 1 negates ISENS at read time (deactivation)
    usens = -sens_raw if isflag == 1 else sens_raw
    t8["sens"][local] = usens
    t8["isflag"][local] = isflag
    if usens != 0 and isflag != 1:
        # activation modes start inactive (Fortran OFF = -10)
        t8["off"][local] = 0.0
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
        dn_raw = float(p.get(f"min_rup{i + 1}", 0.0) or 0.0)
        dx_raw = float(p.get(f"max_rup{i + 1}", 0.0) or 0.0)
        if ifail2 == 0:
            # hm_read_prop08: rupture displacements ride the LSCALE factor
            s_eff = s_raw if s_raw != 0.0 else 1.0
            dn_raw *= s_eff
            dx_raw *= s_eff
        t8["dmin"][local, i] = dn_raw if dn_raw != 0.0 else -1e30
        t8["dmax"][local, i] = dx_raw if dx_raw != 0.0 else 1e30
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
        if hflag not in (0, 1, 2) and log is not None:
            log.warning(f"/PROP/TYPE8/{pid}: DOF{d} IECROU={hflag} not "
                        f"ported yet (redef3.F90:{_iecrou_line(hflag)}) — "
                        f"elastic response kept", "PROP CHECK")
        if fun_c != 0 and log is not None:
            log.warning(f"/PROP/TYPE8/{pid}: DOF{d} FUN_C={fun_c} needs "
                        f"IECROU 4/5/7 (not ported) — ignored", "PROP CHECK")


def _type8_dlim(x, f, e6, dmin, dmax, ifail2):
    """One-DOF rupture ratio (``r2def3.F`` DLIM blocks).

    ``IFAIL2 = 0``: displacement (``x`` over the limit with its sign);
    ``1``: force; ``2``: accumulated work.  Any other mode contributes
    nothing, exactly like the Fortran ``IF/ELSEIF`` chain falling through.
    Unset limits (``+-1e30``) are skipped, mirroring the deck default.
    """
    if ifail2 == 0:
        if x > 0.0 and dmax < 1e29:
            return x / dmax
        if x < 0.0 and dmin > -1e29:
            return x / dmin
    elif ifail2 == 1:
        if f > 0.0 and dmax < 1e29:
            return f / dmax
        if f < 0.0 and dmin > -1e29:
            return f / dmin
    elif ifail2 == 2:
        if dmax < 1e29:
            return max(0.0, e6) / dmax
    return 0.0


def type8_smooth_crit(crit, critnew, israte, fcut, dt):
    """ISRATE rupture-criterion smoothing (``r2def3.F:714-732``).

    With smoothing on, ``CRITNEW`` trails ``CRIT`` through the ``Fcut``
    low-pass ``alpha = 2 pi Fcut dt / (1 + 2 pi Fcut dt)``; off, it tracks
    immediately.  Either way it latches at 1: a ruptured element never
    heals.  Rupture is decided on the smoothed value, exactly like the
    Fortran ``CRIT`` overwrite feeding ``CRIT >= ONE``.
    """
    if israte != 0:
        fc = fcut if fcut != 0.0 else EP30
        alpha = (2.0 * math.pi * fc * dt) / (1.0 + 2.0 * math.pi * fc * dt)
        if critnew < 1.0:
            c = min(crit, 1.0 + 1e-3)       # EM3 overshoot allowance
            return min(alpha * c + (1.0 - alpha) * critnew, 1.0)
        return 1.0
    if critnew < 1.0:
        return min(crit, 1.0)
    return 1.0


def type8_sensor_off(usens, isflag, fired):
    """Sensor OFF target for one element (``r23sens3.F``/``r1sens3.F``).

    ``usens`` is the signed sensor id (``hm_read_prop08`` negates it when
    ``IFL == 1``); ``fired`` latches like ``TT > TSTART``.  Returns
    ``(off, reref)``: ``off`` is 1/0, and ``reref`` asks the caller to
    re-reference the deformed length on this activation (the ``IFL == 2``
    both-mode ``X0 += DX``).  ``usens == 0`` is always active.
    """
    if usens == 0:
        return 1.0, False
    if isflag == 2:
        return (1.0, True) if fired else (0.0, False)
    if usens > 0:
        return (1.0, False) if fired else (0.0, False)
    return (0.0, False) if fired else (1.0, False)


def type8_forces(delta, dvl, theta, dwl, dt_val, t8, L0, sensors):
    """One ``r2def3``-style update over the six uncoupled DOFs.

    ``delta``/``dvl`` are the (m, 3) translation displacements/rates,
    ``theta``/``dwl`` the accumulated/rate rotations.  Linear DOFs keep the
    historical closed form ``k dx + c rate`` bit-for-bit (the suite pins
    ``C`` against the nodal velocity with no displacement); nonlinear DOFs
    run the shared :func:`redef3_channel`/:func:`redef3_plastic` with true
    previous-displacement state (``r2def3`` DXOLD) and displacement-derived
    rates, booking trapezoidal work.  Returns ``(F, M, theta_new, deint)``.
    """
    m = delta.shape[0]
    F = np.zeros((m, 3))
    M = np.zeros((m, 3))
    deint = np.zeros(m)
    theta_new = theta + dwl * dt_val
    for e in range(m):
        if t8["dead"][e]:
            continue                    # ruptured: forces stay zero
        usens = int(t8["sens"][e])
        sid = abs(usens)
        fired = False
        if sensors is not None and sid != 0:
            fired = bool(sensors.active(sid)) or \
                sid in getattr(sensors, "fire_time", {})
        off, reref = type8_sensor_off(usens, int(t8["isflag"][e]), fired)
        prev_off = float(t8["off"][e])
        t8["off"][e] = off
        if reref and prev_off <= 0.0:
            # IFL==2 activation re-references the deformed length (r2sens3:
            # X0 += DX, totals zeroed); this step restarts from zero force
            L0[e] += delta[e]
            theta_new[e] = 0.0
            t8["dx_old"][e, :3] = 0.0
            zeroed = True
        else:
            zeroed = False
        ifail2e = int(t8["ifail2"][e])
        ifaile = int(t8["ifail"][e])
        crit = 0.0
        for i in range(TYPE8_NDOF):
            if zeroed:
                dx, dxo, rate = 0.0, 0.0, 0.0
            elif i < 3:
                dx, rate = delta[e, i], dvl[e, i]
                dxo = float(t8["dx_old"][e, i])
            else:
                j = i - 3
                dx, dxo, rate = theta_new[e, j], theta[e, j], dwl[e, j]
            fun_a = int(t8["fun"][e, i])
            cv = t8["cv_a"][e, i]
            hflag = int(t8["iecrou"][e, i])
            if fun_a != 0 and cv is not None and hflag in (1, 2):
                f, e_incr, kti, ndpx, ndpx2, nf_ep = redef3_plastic(
                    dx, dxo, iecrou=hflag, fn1=cv,
                    xk=float(t8["xk"][e, i]), xc=float(t8["xc"][e, i]),
                    ak=float(t8["ak"][e, i]), b=float(t8["bk"][e, i]),
                    d=float(t8["dk"][e, i]), ee=float(t8["ee"][e, i]),
                    ff=float(t8["ff"][e, i]), gf3=float(t8["gf3"][e, i]),
                    lscale=float(t8["ls"][e, i]), xl0=1.0,
                    ifv=int(t8["funb"][e, i]), ifunc3=int(t8["fund"][e, i]),
                    fnv=t8["cv_b"][e, i], fnd=t8["cv_d"][e, i],
                    dpx=float(t8["dpx"][e, i]), dpx2=float(t8["dpx2"][e, i]),
                    f_ep=float(t8["f_ep"][e, i]),
                    f_old=float(t8["f_old"][e, i]), dt=dt_val, off=off)
                t8["dpx"][e, i], t8["dpx2"][e, i], t8["f_ep"][e, i] = \
                    ndpx, ndpx2, nf_ep
            elif fun_a != 0 and cv is not None and hflag == 0:
                f, e_incr, kti = redef3_channel(
                    dx, dxo, xk=float(t8["xk"][e, i]), xc=float(t8["xc"][e, i]),
                    ak=float(t8["ak"][e, i]), b=float(t8["bk"][e, i]),
                    d=float(t8["dk"][e, i]), ee=float(t8["ee"][e, i]),
                    ff=float(t8["ff"][e, i]), gf3=float(t8["gf3"][e, i]),
                    lscale=float(t8["ls"][e, i]), xl0=1.0,
                    iecrou=0, ifunc=fun_a,
                    ifv=int(t8["funb"][e, i]), ifunc3=int(t8["fund"][e, i]),
                    f_old=float(t8["f_old"][e, i]), dt=dt_val, off=off,
                    fn1=cv, fnv=t8["cv_b"][e, i], fnd=t8["cv_d"][e, i])
            else:
                k_raw = float(t8["xk"][e, i]) * float(t8["ak"][e, i])
                c_raw = float(t8["xc"][e, i])
                f = (k_raw * dx + c_raw * rate) * off
                kti = float(t8["xk"][e, i]) * float(t8["ak"][e, i])
                e_incr = f * rate * dt_val
            if i < 3:
                F[e, i] = f
                t8["dx_old"][e, i] = dx
            else:
                M[e, i - 3] = f
            t8["f_old"][e, i] = f
            t8["k_tan"][e, i] = kti
            t8["e6"][e, i] += e_incr
            deint[e] += e_incr
            if off == 1.0:
                # DLIM accumulates only on live elements (r2def3 gates every
                # DLIM block on OFF == ONE); smoothing still runs below
                dlim = _type8_dlim(dx, f, t8["e6"][e, i], t8["dmin"][e, i],
                                   t8["dmax"][e, i], ifail2e)
                if ifaile == 0:
                    # uniaxial failure: worst DOF wins (r2def3: MAX over DLIM)
                    crit = max(crit, dlim)
                else:
                    # multiaxial failure: coupled sum of squares
                    crit += dlim * dlim
        t8["crit"][e] = crit
        new = type8_smooth_crit(crit, t8["critnew"][e],
                                int(t8["israte"][e]), float(t8["fcut"][e]),
                                dt_val)
        t8["critnew"][e] = new
        if off == 1.0 and new >= 1.0:
            # coupled failure (r2def3:733-739): the element is deleted from
            # the NEXT cycle — this step's forces stand, then go silent
            t8["dead"][e] = True
            t8["off"][e] = 0.0
    return F, M, theta_new, deint
