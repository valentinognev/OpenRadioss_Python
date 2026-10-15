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
#   engine/source/elements/shell/coqueba/cbadef.F  (CBADERIRZ0, CBPATCH,
#     CBADEFRZ; modified: transcribed to NumPy)
#   engine/source/elements/shell/coquez/czdef.F    (CZDERIRZ, CZDEFRZ;
#     modified: transcribed to NumPy)
#   engine/source/elements/sh3n/coquedk/cncoef3.F  (CNCOEF3/CNCOEF3B KRZ;
#     modified: transcribed to NumPy)

"""Drilling stiffness (``Idrill``/``Isrot``) shared operators — P3.5.

QB AT (``coqueba``) and QEPH (``coquez``) build their drilling
B-operators from the SAME formulas (``CBADERIRZ0`` = ``CZDERIRZ`` up to
the ``/AREA`` scaling the ``CBADEFRZ`` header notes as
"QEPH: Bi*A, RLZ: VRZ/A — QBAT: Bi, RLZ: VRZ", plus the patch-test
correction ``CBPATCH`` both call). This module carries that shared
core; the per-family strain/force glue lives in ``shell_qbat.py`` and
``shell_qeph.py`` next to the ``forces()`` paths it extends.

Magnitude (never a plausible formula): ``cncoef3.F:86-91`` —
``KFAC = EM03`` (1e-3) explicit for quads
("explicit KFAC=1.0e-3 for quad"), ``KRZ = KFAC * G`` with the
material shear modulus.

Scope (kept visible per the phase contract): FLAT rows only. Warped
rows with ``isrot > 0`` run without drill stiffness and the owning
kernel warns once at init — the warped drill legs (QBAT ``CBAFORRZ``
warped leg's ``BRZ`` rows from ``CBABE3``/``CBADEF``-warped, QEPH
``CZPROJN``'s ``VQN`` leg at ``czproj.F:192-196,243-271``) are
explicitly refused, not silently missing. The drill *hourglass*
stabilization (QEPH ``CZFINTNRZ``, ``czfintn.F:565``) is likewise not
ported: the ``CZFINTN1`` stabilization keeps running unchanged.
"""

from __future__ import annotations

import numpy as np

# cncoef3.F:86-91 — explicit KFAC for quads (implicit would be 0.1;
# the port's QBAT/QEPH paths are explicit-only).
KRZ_FAC = 1.0e-3

# Gauss-point weights shared with the flat QBAT operators
# (cbadef.F: DATA VPG/PG1,PG1,PG,PG1,PG,PG,PG1,PG/).
_PG = 0.577350269189626
VPG = np.array([[-_PG, -_PG], [_PG, -_PG], [_PG, _PG], [-_PG, _PG]])


def group_isrot(group) -> int:
    """Engine ``ISROT`` for a group: the max over its slices' props.

    The reader resolves the card ``Idrill`` through ``resolve_isrot``
    (``hm_read_prop01.F:214-215``) into ``params["isrot"]``; groups
    built without a reader default to 0 (drill off).
    """
    top = 0
    for _sl, _mat, prop in group.state.get("slices", []):
        params = getattr(prop, "params", {}) or {}
        try:
            top = max(top, int(params.get("isrot", 0) or 0))
        except (TypeError, ValueError):
            continue
    return top


def qbat_flat_mask(xe, E, tol=1.0e-12):
    """Flat/warped split for QBAT init-time checks (``cbacoor.F:442-451``).

    ``ZL1**2 < TOL*LL``; ``cbacoork.F:131`` raises ``TOL`` from ``EM12``
    to ``EM8`` when ``ISROT > 0``, so drill groups pass ``tol=1e-8``.
    ``force_flat`` (NPT==1) is a cycle-time input and reads False here.
    """
    d = np.asarray(xe) - np.asarray(xe)[:, 0:1, :]
    xl = np.einsum("njk,nka->nja", d, np.asarray(E))
    cx = xl[:, :, 0] - xl[:, :, 0].mean(axis=1)[:, None]
    cy = xl[:, :, 1] - xl[:, :, 1].mean(axis=1)[:, None]
    zl1 = -xl[:, :, 2].mean(axis=1)
    x13 = 0.5 * (cx[:, 0] - cx[:, 2])
    x24 = 0.5 * (cx[:, 1] - cx[:, 3])
    y13 = 0.5 * (cy[:, 0] - cy[:, 2])
    y24 = 0.5 * (cy[:, 1] - cy[:, 3])
    ll = np.maximum(x13 ** 2 + y13 ** 2, x24 ** 2 + y24 ** 2)
    return zl1 ** 2 < tol * ll


def cbpatch(bm0, bmk, bme, area, mx13, mx23, mx34, my13, my23, my34):
    """Patch-test correction, ``cbadef.F`` CBPATCH (live lines only).

    Called by both ``CBADERIRZ0`` and ``CZDERIRZ``. The ``IAC > 0``
    branch returns right after rebuilding ``BM0RZ`` analytically, so
    the Gauss-point loop below it is dead upstream and not ported.
    Operates in place on ``bm0``; all arrays ``(n, 4, 4)``.
    """
    jk = (4.0 / 3.0) * (mx23 * my13 - mx13 * my23) / np.maximum(area, 1e-20)
    je = (4.0 / 3.0) * (mx13 * my34 - mx34 * my13) / np.maximum(area, 1e-20)
    for r in range(4):
        bm0[:, r, :] = -bmk[:, r, :] * jk[:, None] - bme[:, r, :] * je[:, None]


def derirz(x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34,
           area, qbat=False):
    """Drilling B-operators ``(BM0RZ, BMKRZ, BMERZ)``, each ``(n, 4, 4)``.

    ``CZDERIRZ`` (``czdef.F:299``) for QEPH, ``CBADERIRZ0``
    (``cbadef.F``) for QBAT (``qbat=True``). The two differ ONLY in the
    ``XIJ/YIJ/XIL/YIL`` pre-factor (``/AREA`` for QBAT — the "Bi vs
    Bi*A" note in the ``CBADEFRZ`` header) and in the final ``A05``
    (``HALF*AREA`` QEPH vs ``HALF`` QBAT); the ``BM`` assembly, the
    ``CBPATCH`` call and the symmetrization patterns are shared.
    """
    n = len(np.asarray(area))
    if qbat:
        a = 0.5 / np.maximum(area, 1e-20)
        xij1 = a * (mx13 - mx34)
        xij2 = 0.5 * x13 + a * (mx23 - mx13)
        yij1 = a * (my13 - my34)
        yij2 = 0.5 * y13 + a * (my23 - my13)
        xil1 = a * (mx13 - mx23)
        xil3 = a * (mx13 - mx34) - 0.5 * x13
        yil1 = a * (my13 - my23)
        yil3 = a * (my13 - my34) - 0.5 * y13
        a05 = 0.5
    else:
        xij1 = 0.5 * (mx13 - mx34)
        xij2 = 0.5 * (x13 + mx23 - mx13)
        yij1 = 0.5 * (my13 - my34)
        yij2 = 0.5 * (y13 + my23 - my13)
        xil1 = 0.5 * (mx13 - mx23)
        xil3 = 0.5 * (mx13 - mx34 - x13)
        yil1 = 0.5 * (my13 - my23)
        yil3 = 0.5 * (my13 - my34 - y13)
        a05 = 0.5 * area
    xij3, xij4 = -xij2, -xij1
    yij3, yij4 = -yij2, -yij1
    xil2, xil4 = -xil1, -xil3
    yil2, yil4 = -yil1, -yil3

    bm0 = np.empty((n, 4, 4))
    bmk = np.empty((n, 4, 4))
    bme = np.empty((n, 4, 4))
    # row 0: NxI,x (MY34*YIJ KSI - MY23*YIL ETA, then KSI/ETA variants)
    bm0[:, 0, 0] = -my34 * yij1 + my23 * yil1
    bm0[:, 0, 1] = my34 * yij2 + my23 * yil2
    bm0[:, 0, 2] = my34 * yij3 - my23 * yil3
    bm0[:, 0, 3] = -my34 * yij4 - my23 * yil4
    bmk[:, 0, 0] = -my13 * yij1 - 2.0 * my34 * yil1
    bmk[:, 0, 1] = my13 * yij2 - 2.0 * my34 * yil2
    bmk[:, 0, 2] = my13 * yij3 - 2.0 * my34 * yil3
    bmk[:, 0, 3] = -my13 * yij4 - 2.0 * my34 * yil4
    bme[:, 0, 0] = my13 * yil1 + 2.0 * my23 * yij1
    bme[:, 0, 1] = my13 * yil2 + 2.0 * my23 * yij2
    bme[:, 0, 2] = -my13 * yil3 + 2.0 * my23 * yij3
    bme[:, 0, 3] = -my13 * yil4 + 2.0 * my23 * yij4
    # row 1: NyI,y
    bm0[:, 1, 0] = -mx34 * xij1 + mx23 * xil1
    bm0[:, 1, 1] = mx34 * xij2 + mx23 * xil2
    bm0[:, 1, 2] = mx34 * xij3 - mx23 * xil3
    bm0[:, 1, 3] = -mx34 * xij4 - mx23 * xil4
    bmk[:, 1, 0] = -mx13 * xij1 - 2.0 * mx34 * xil1
    bmk[:, 1, 1] = mx13 * xij2 - 2.0 * mx34 * xil2
    bmk[:, 1, 2] = mx13 * xij3 - 2.0 * mx34 * xil3
    bmk[:, 1, 3] = -mx13 * xij4 - 2.0 * mx34 * xil4
    bme[:, 1, 0] = -mx13 * xil1 + 2.0 * mx23 * xij1
    bme[:, 1, 1] = -mx13 * xil2 + 2.0 * mx23 * xij2
    bme[:, 1, 2] = -mx13 * xil3 + 2.0 * mx23 * xij3
    bme[:, 1, 3] = -mx13 * xil4 + 2.0 * mx23 * xij4
    # row 2: NxI,y
    bm0[:, 2, 0] = mx34 * yij1 - mx23 * yil1
    bm0[:, 2, 1] = -mx34 * yij2 - mx23 * yil2
    bm0[:, 2, 2] = -mx34 * yij3 + mx23 * yil3
    bm0[:, 2, 3] = mx34 * yij4 + mx23 * yil4
    bmk[:, 2, 0] = mx13 * yij1 + 2.0 * mx34 * yil1
    bmk[:, 2, 1] = -mx13 * yij2 + 2.0 * mx34 * yil2
    bmk[:, 2, 2] = -mx13 * yij3 + 2.0 * mx34 * yil3
    bmk[:, 2, 3] = mx13 * yij4 + 2.0 * mx34 * yil4
    bme[:, 2, 0] = -mx13 * yil1 - 2.0 * mx23 * yij1
    bme[:, 2, 1] = -mx13 * yil2 - 2.0 * mx23 * yij2
    bme[:, 2, 2] = mx13 * yil3 - 2.0 * mx23 * yij3
    bme[:, 2, 3] = mx13 * yil4 - 2.0 * mx23 * yij4
    # row 3: NyI,x
    bm0[:, 3, 0] = my34 * xij1 - my23 * xil1
    bm0[:, 3, 1] = -my34 * xij2 - my23 * xil2
    bm0[:, 3, 2] = -my34 * xij3 + my23 * xil3
    bm0[:, 3, 3] = my34 * xij4 + my23 * xil4
    bmk[:, 3, 0] = my13 * xij1 + 2.0 * my34 * xil1
    bmk[:, 3, 1] = -my13 * xij2 + 2.0 * my34 * xil2
    bmk[:, 3, 2] = -my13 * xij3 + 2.0 * my34 * xil3
    bmk[:, 3, 3] = my13 * xij4 + 2.0 * my34 * xil4
    bme[:, 3, 0] = -my13 * xil1 - 2.0 * my23 * xij1
    bme[:, 3, 1] = -my13 * xil2 - 2.0 * my23 * xij2
    bme[:, 3, 2] = my13 * xil3 - 2.0 * my23 * xij3
    bme[:, 3, 3] = my13 * xil4 - 2.0 * my23 * xij4

    cbpatch(bm0, bmk, bme, area, mx13, mx23, mx34, my13, my23, my34)

    # symmetrization to pass the patch test: row 2 -> (NxI,y+NyI,x),
    # row 3 -> (-NxI,y+NyI,x-2NI) with the KSI/ETA sign patterns.
    a05k = np.array([-a05, a05, a05, -a05])
    a05e = np.array([-a05, -a05, a05, a05])
    for j in range(4):
        nxy = bm0[:, 2, j].copy()
        nyx = bm0[:, 3, j].copy()
        bm0[:, 2, j] = nxy + nyx
        bm0[:, 3, j] = -nxy + nyx - a05
        nxy = bmk[:, 2, j].copy()
        nyx = bmk[:, 3, j].copy()
        bmk[:, 2, j] = nxy + nyx
        bmk[:, 3, j] = -nxy + nyx - a05k[j]
        nxy = bme[:, 2, j].copy()
        nyx = bme[:, 3, j].copy()
        bme[:, 2, j] = nxy + nyx
        bme[:, 3, j] = -nxy + nyx - a05e[j]
    return bm0, bmk, bme
