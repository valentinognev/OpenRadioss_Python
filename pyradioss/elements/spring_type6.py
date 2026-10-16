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
#   engine/source/elements/spring/r6def3.F  (axial channel: DX = ALDP - X0,
#     the REDEF3 call, E = sum(E6); dt inputs XM/XKM/XCM for R2LEN3)
#   engine/source/elements/spring/redef3.F90  (via spring_common.redef3_channel)
#   engine/source/elements/spring/repla3.F  (via spring_common.repla3_correct)
#   engine/source/elements/spring/r2len3.F  (the dt close: STI from XKM/XCM)
#   starter/source/properties/spring/hm_read_prop25.F  (GEO(3)=STIFF1/A,
#     GEO(4)=DAMP, GEO(41)=A, GEO(93)=ILENG)

"""The ``r6def3`` axial spring kernel (Task P4.1).

Scope correction (P4.0 audit, ``tools/validation_data/w1d_routine_status.json``
row ``rforc3@gap:igtyp6``): upstream has **no** ``IGTYP == 6`` branch.
``rforc3.F:785`` calls ``R6DEF3`` from the ``IGTYP == 25`` (SPR_AXI) branch,
and the starter ships no ``/PROP/TYPE6`` reader.  This module keeps the
plan's ``spring_type6`` name — it follows the *file* ``r6def3`` — and ports
its axial translation channel; the shear/torsion/bending channels stay with
``spring_advanced.forces_axi_type25`` until a task owns them.

Axial force law (``ILENG == 0`` default, so ``XL0 = 1`` and the ``redef3``
normalisation is the identity)::

    F = ak * (k * (L - L0) + c * L_dot)

with the shared channel in :mod:`spring_common` (linear ``IFUNC == 0``
branch, ``redef3.F90:739-743`` + close ``:1144-1147``).  Rupture
(``DMN``/``DMX``, ``r6def3.F`` failure blocks), the nonlinear ``IECROU``
channels and the ``python_funct_mod`` callback (Task P4.5) are refused
loudly by :mod:`spring_common`, not silently linearised.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3
from . import spring_common


def _safe_param(params: dict, key: str, default: float = 0.0) -> float:
    val = params.get(key)
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


#: Property keys this kernel does NOT implement (curves, hardening, rate,
#: rupture, skew).  A property carrying any of them with a live value is
#: refused loudly by :func:`init_group` — silently running it linear would
#: be wrong physics (``redef3.F90`` branches, ``r6def3.F`` failure blocks).
_NONLINEAR_KEYS = frozenset({
    "fun_a", "fun_a1", "fun_a2", "fun_a_tens", "fun_a_shear",
    "fun_b1", "fun_c1", "fun_d1", "fun_b2", "fun_c2", "fun_d2",
    "hflag", "hflag1", "hflag2", "hflag_tens", "hflag_shear",
    "b", "d", "b_tens", "d_tens", "e_tens", "f_tens",
    "min_rup", "max_rup", "min_rup1", "max_rup1", "min_rup2", "max_rup2",
    "min_rup_tens", "max_rup_tens", "f_yield", "ifail", "ifail2",
    "skew_id", "lscale",
})


def _refuse_nonlinear(prop) -> None:
    """Raise on property content this kernel cannot compute (see above)."""
    dicts = [getattr(prop, "params", {}) or {}]
    for sub in ("tension", "shear"):
        sub_p = dicts[0].get(sub)
        if isinstance(sub_p, dict):
            dicts.append(sub_p)
    for p in dicts:
        for key in sorted(_NONLINEAR_KEYS):
            if key not in p:
                continue
            try:
                live = float(p[key]) != 0.0
            except (ValueError, TypeError):
                live = p[key] not in (None, "", 0)
            if live:
                raise NotImplementedError(
                    f"/PROP key {key!r}={p[key]!r} needs the nonlinear "
                    f"r6def3/redef3 path (not this linear kernel)"
                )
        if "a" in p:
            try:
                if float(p["a"]) not in (0.0, 1.0):
                    raise NotImplementedError(
                        f"/PROP key 'a'={p['a']!r} needs the rate-scaled "
                        f"redef3 path (not this linear kernel)"
                    )
            except (ValueError, TypeError):
                pass


def init_group(group, model, log):
    """Element buffer for the ``r6def3`` axial channel.

    Reads the flat property keys ``k`` (``GEO(3)``, card ``STIFF1``),
    ``c`` (``GEO(4)``, card ``DAMP``), ``mass``, ``ileng`` (``GEO(93)``)
    and ``a_scale`` (``GEO(41)``, card ``Acoeft1``) off each slice's
    property; ``L0`` is the initial node distance (``r6def3.F``: ``X0 =
    ALDP`` at ``TT == 0``).  Mass is lumped half per node by the caller.
    """
    st = group.state
    n = group.n
    if n == 0 or len(group.conn) == 0:
        st.update(
            t6_k=np.empty(0), t6_c=np.empty(0), t6_mass=np.empty(0),
            t6_L0=np.empty(0), t6_ileng=np.zeros(0, dtype=np.int64),
            t6_ak=np.empty(0), t6_force=np.empty(0), t6_eint=np.empty(0),
            t6_dx_old=np.empty(0), t6_k_tan=np.empty(0),
        )
        return
    xe = model.x0[group.conn[:, :2]]
    L0 = norm3(xe[:, 1] - xe[:, 0])
    k = np.zeros(n)
    c = np.zeros(n)
    mass = np.zeros(n)
    ileng = np.zeros(n, dtype=np.int64)
    ak = np.ones(n)
    for sl, mat, prop in st["slices"]:
        _refuse_nonlinear(prop)
        p = getattr(prop, "params", {}) or {}
        k[sl] = _safe_param(p, "k", _safe_param(p, "stiff", 0.0))
        c[sl] = _safe_param(p, "c", _safe_param(p, "damp", 0.0))
        mass[sl] = _safe_param(p, "mass", 0.0)
        ileng[sl] = int(_safe_param(p, "ileng", 0.0))
        ak[sl] = _safe_param(p, "a_scale", 1.0)
    if "off" not in st:
        st["off"] = np.ones(n)
    xl0_chan = np.where(ileng != 0, np.maximum(L0, EM20), 1.0)
    st.update(
        t6_k=k, t6_c=c, t6_mass=mass, t6_L0=L0, t6_ileng=ileng, t6_ak=ak,
        t6_force=np.zeros(n), t6_eint=np.zeros(n), t6_dx_old=np.zeros(n),
        t6_k_tan=ak * k / xl0_chan,
    )


def _axis(group, x):
    conn = group.conn[:, :2]
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    norm = norm3(dx)
    degen = norm < EM20
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])
    return conn, L, a


def driver(group, model, ctx):
    """Axial ``r6def3`` channel: ``driver(group, model, ctx) -> np.ndarray``.

    ``ctx`` carries ``x`` (current positions), ``dt`` and the ``fint``
    nodal-force accumulator (``v``, when present, is informational: the
    channel rate follows ``redef3`` — ``DVX = (DX - DXOLD) / DT``).  Forces are
    scattered tension-positive along the current axis (node 0 ``+F*a``,
    node 1 ``-F*a``); internal energy is booked trapezoidally by the shared
    channel.  Returns the per-element critical step (``r2len3.F`` close);
    a free segment (``k == 0, c == 0``) claims ``EP30``.
    """
    st = group.state
    x = np.asarray(ctx["x"], dtype=float)
    dt = float(ctx.get("dt", 0.0))
    fint = ctx["fint"]
    conn, L, a = _axis(group, x)
    n = group.n
    if n == 0:
        return np.empty(0)

    off = st.get("off", np.ones(n))
    alive = off > 0.0
    L0 = st["t6_L0"]
    use_strain = (st["t6_ileng"] != 0) & (L0 > EM20)
    xl0_chan = np.where(use_strain, L0, 1.0)

    F_old = st["t6_force"].copy()
    F = np.zeros(n)
    k_tan = st["t6_k_tan"].copy()
    for i in range(n):
        if not alive[i]:
            continue
        # The viscous part rides the shared channel's rate close
        # (``xc(i)*dvx(i)``, ``redef3.F90:1142``); the channel also books
        # the trapezoidal work of the total force, damping included.
        fi, e_incr, kti = spring_common.redef3_channel(
            L[i] - L0[i], st["t6_dx_old"][i],
            xk=st["t6_k"][i], xc=st["t6_c"][i], ak=st["t6_ak"][i],
            xl0=xl0_chan[i], f_old=F_old[i], dt=dt, off=1.0,
        )
        F[i] = fi
        st["t6_eint"][i] += e_incr
        k_tan[i] = kti
    F = np.where(alive, F, 0.0)
    st["t6_force"][:] = F
    st["t6_dx_old"][:] = L - L0
    st["t6_k_tan"][:] = k_tan

    fvec = F[:, None] * a
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)

    mass = np.maximum(st["t6_mass"], EM20)
    kk = np.maximum(st["t6_k"], 0.0)
    cc = st["t6_c"]
    pos_k = (st["t6_k"] > 0.0) & (st["t6_mass"] > 0.0)
    pure_c = (st["t6_k"] <= 0.0) & (cc > 0.0) & (st["t6_mass"] > 0.0)
    omega = 2.0 * np.sqrt(np.where(pos_k, kk / mass, 1.0))
    xi = np.where(pos_k, cc / np.sqrt(np.maximum(kk * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    dt_c = np.where(pure_c, 0.5 * mass / np.maximum(cc, EM20), EP30)
    dt_elem = np.where(pos_k, dt_crit, dt_c)
    return np.where(alive, dt_elem, EP30)


def _edofs(conn):
    if len(conn) == 0:
        return np.empty((0, 6), dtype=np.int64)
    edofs = np.empty((len(conn), 6), dtype=np.int64)
    for c in range(3):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 3 + c] = conn[:, 1] * 6 + c
    return edofs


def _blocks(kb):
    n = len(kb)
    if n == 0:
        return np.empty((0, 6, 6))
    ke = np.empty((n, 6, 6))
    ke[:, :3, :3] = kb
    ke[:, 3:, 3:] = kb
    ke[:, :3, 3:] = -kb
    ke[:, 3:, :3] = -kb
    return ke


def tangent(group, x):
    """Material tangent ``k_tan a a^T`` at geometry ``x`` (``(ke, edofs)``).

    ``k_tan`` is the shared channel's linearised stiffness
    (``ak * k / XL0`` on the linear path).
    """
    if group.n == 0 or len(group.conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)
    conn, L, a = _axis(group, np.asarray(x, dtype=float))
    kb = group.state["t6_k_tan"][:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks(kb), _edofs(conn)


def kgeo(group, x):
    """Geometric stiffness ``(F/L)(I - a a^T)`` from the current axial force.

    Identically zero at zero force — the same taut-string operator as the
    TYPE4 spring and the truss.
    """
    if group.n == 0 or len(group.conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)
    conn, L, a = _axis(group, np.asarray(x, dtype=float))
    f_over_l = group.state["t6_force"] / L
    eye = np.eye(3)
    kb = f_over_l[:, None, None] * (eye[None, :, :] - np.einsum("ni,nj->nij", a, a))
    return _blocks(kb), _edofs(conn)


def energy(group):
    """Per-element internal energy booked by :func:`driver` (the ``E`` sum
    of ``r6def3.F`` restricted to the axial channel)."""
    if group.n == 0:
        return np.empty(0)
    return np.array(group.state["t6_eint"], dtype=float)
