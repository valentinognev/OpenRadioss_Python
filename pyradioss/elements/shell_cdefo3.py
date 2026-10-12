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
#   engine/source/elements/shell/coque/cdefo3.F  (modified: transcribed to
#     NumPy, the three branches dispatched on one `kind` argument, the
#     IMPL_S>0 guard carried explicitly as `implicit=True` — for the
#     IHBE==4 branch it is NOT the same thing as dt == 0, and Z2 added
#     as an input)
#   engine/source/elements/shell/coque/cderi3.F   (added: the Z2 the
#     IHBE==4 branch consumes, l.95 / l.109)
#   engine/source/elements/shell/coque/cpxpy3.F   (added: the Z2 = ZERO
#     of the ISMSTR == 3 stress-based path, l.77 / l.88; and the PX/PY
#     operator pairs and AREA the coefficients are built from, l.75-82)
#   engine/source/elements/shell/coque/cdt3.F    (the DT1C condensation
#     length consumed as `dt`)

"""``cdefo3.F`` — the BT second-order rigid-rotation correction, one entry
per ENGINE formulation branch.

WHY THIS IS A SEPARATE MODULE
-----------------------------
``pyradioss/elements/shell_bt4.py`` is at its recorded line budget
(``tools/validation_data/size_budget.json``, 1913 lines) and may only
shrink, so the transcribed branches live here and
``shell_bt4._rot2_correction`` re-exports them.

THE UPSTREAM CONTRACT
---------------------
``cdefo3.F`` has three formulation branches, selected on the ENGINE flag
``IHBE``.  ``Ishell`` card 3 maps to engine ``IHBE = 2`` through
``GEO(171) = IHBE-1`` (``hm_read_prop01.F:306-315`` -> ``cgrtails.F:956``
``IPARG(23)`` -> ``cforc3.F``), so parity case c43
(``c43_E1000_Bending_BT_BT_type3_Sf_0``, ``Ishell = 3``) runs the
node-1-relative branch.  Card 4 maps to ``IHBE = 4`` unchanged (it is the
one card that is *not* ``card-1``), so parity case c45
(``c45_E1000_Bending_BT_BT_type4_Sf_0``, ``Ishell = 4``) runs the third:

  ``IF(IHBE<=1)THEN``            l.104-131  "general"              (kind 1)
  ``ELSEIF(IHBE==2.OR.IHBE==3)``  l.133-173  node-1-relative       (kind 3)
  ``ELSEIF(IHBE==4)THEN``        l.174-213  Z2 warp                (kind 4)

Kinds 1 and 3 fix the same frame-lag problem — the corotational frame is
evaluated on the END-of-step geometry while the velocities sit at
mid-step, so an element rotating rigidly at rate ``w`` about an in-plane
axis measures a spurious membrane stretching rate of ``+w^2 dt/2`` — in
DIFFERENT ways, and upstream's asymmetry is deliberate:

* kind 1 works on the DIFFERENCES of the out-of-plane diagonal
  velocities and **over-corrects 2x** (``TMP1A`` carries ``dt/4`` and
  ``(VZ13-VZ24)^2``, which on a stencil-exact roll is ``-w^2 dt``);
* kind 3 works on the out-of-plane membrane RATE
  ``EXZ = PX1*VZ13 + PX2*VZ24`` and is the **exact** cancellation
  (``EXZ2 = (EXZ/AREA)^2 * dt/2``).

M41 mirrored the kind-1 asymmetry on purpose; this module keeps all three
apart, because they are not interchangeable — a fact pinned by
``tests/test_p3_bt3_c43.py`` and ``tests/test_p3_bt4_c45.py``.

KIND 4 IS A DIFFERENT SHAPE OF ANSWER AGAIN
-------------------------------------------
Kinds 1 and 3 yield a coefficient field per component, which the caller
contracts with the membrane operators.  Kind 4 does not: its own header
calls it "velocity modification for the lurglass membrane" (l.179), and
it returns the four per-node in-plane velocity corrections, the warp term
``GZX*ZZ2`` alternating sign between the 1-3 and 2-4 diagonals and each
half contracting with the OPPOSITE operator pair (``PY`` for the x rows,
``PX`` for the y rows).  It also is the only branch the implicit guard
does not switch off: ``EXZZ2``/``EYZZ2`` carry no ``DT1V4``.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20

#: cdefo3.F branch selector -> ``_rot2_correction`` ``kind``: 0 none,
#: 1 the ``IHBE <= 1`` general form, 3 the ``IHBE == 2 .OR. IHBE == 3``
#: node-1-relative velocity form, 4 the ``IHBE == 4`` Z2-warp form.
ROT2_KIND = {0: 0, 1: 1, 2: 3, 3: 3, 4: 4}


def rot2_correction(px, py, vx, vy, dt, kind=1, z2=None, *, implicit=False):
    """``cdefo3.F`` — the second-order rigid-rotation correction, one entry
    per ENGINE ``IHBE`` formulation branch, returning the pair of upstream
    correction coefficients.

    Parameters
    ----------
    px, py : (..., 2) arrays
        ``(PX1, PX2)`` and ``(PY1, PY2)`` — the AREA-scaled membrane
        operators, ``PX_i = AREA*B1_i`` and ``PY_i = AREA*B2_i``
        (``cpxpy3.F`` l.75-82 / ``cderi3.F`` l.173-176).
    vx, vy : (..., 2) arrays
        ``(VZ13, VZ24)`` — the OUT-OF-PLANE diagonal velocity differences
        (``cdefo3.F`` l.97-98) — and ``(VX13, VX24)``, the in-plane x
        diagonal pair the returned x coefficient acts on.
    dt : ``DT1C``
        the condensation length ``cdt3.F`` computes the element step for;
        ``ccoor3.F`` l.124 sets ``DT1C(I) = DT1``, the cycle step.
    kind : {0, 1, 3, 4}
        0 no correction, 1 the ``IHBE <= 1`` general form, 3 the
        ``IHBE == 2 .OR. IHBE == 3`` node-1-relative velocity form
        (parity case c43), 4 the ``IHBE == 4`` Z2-warp form (c45).
    z2 : (...) array, REQUIRED for ``kind == 4``
        ``Z2`` — the corotational out-of-plane coordinate of node 2
        relative to node 1, ``(x2-x1).e3`` (``cderi3.F`` l.95 and l.109;
        zeroed outright on the ``ISMSTR == 3`` stress-based path,
        ``cpxpy3.F`` l.77 and l.88).  It is an INPUT, not curvature state
        — ``ccurv3.F`` never produces it and ``cdefo3`` runs before it
        (``cforc3.F`` l.471 then l.476).
    implicit : bool
        ``IMPL_S > 0``, carried explicitly rather than folded into ``dt``,
        because the two are only the same thing on kinds 1 and 3 (see
        below).

    Returns
    -------
    (dx, dy) : (..., ) or (..., 4) arrays
        the upstream correction fields, in upstream's own naming:

        ==========  ==============================  ==================
        kind        ``dx``                          ``dy``
        ==========  ==============================  ==================
        0           0                               0
        1           ``TMP1A`` (l.111)               ``TMP2B`` (l.122)
        3           ``EXZ2``  (l.142)               ``EYZ2``  (l.148)
        4           per-node VX (l.187-190)         per-node VY (l.197-200)
        ==========  ==============================  ==================

    Kinds 1 and 3 return ONE coefficient per component, which the caller
    contracts with the membrane operators.  Kind 4 is not that shape: the
    branch IS a per-node modification of the four in-plane velocities
    (its own header says "velocity modification for the lurglass
    membrane", l.179), so it returns the four nodal corrections per
    component, in upstream's node order.

    THE IMPLICIT DISABLE IS NOT ``dt == 0`` ON KIND 4
    -------------------------------------------------
    ``IF(IMPL_S>0) DT1V4=ZERO`` (l.109, l.137, l.178) zeroes the
    condensation length's half.  On kinds 1 and 3 EVERY term carries
    ``DT1V4``, so ``dt == 0`` is exactly that guard.  On kind 4 it is not:
    ``EXZZ2 = GZX*ZZ2`` (l.183) and ``EYZZ2 = GZY*ZZ2`` (l.193) carry no
    ``DT1V4`` and SURVIVE it.  So ``implicit=True`` is modelled here as
    ``DT1V4 = 0`` — upstream's guard — and not as a zero return; folding
    it into ``dt`` would silently delete real upstream behaviour on the
    branch.

    WHY ``shell_bt4.forces`` DOES NOT CALL IT (Phase 3 Task P3.1)
    --------------------------------------------------------------
    The inline blocks in ``forces()`` implement exactly these closed
    forms, with the operands grouped the way upstream forms them —
    ``area*(B2_0+B2_1)`` for ``PY1+PY2``, ``np.sum(B1*VZ)`` for
    ``EXZ/AREA``.  Routing them through here would reassociate those into
    ``area*B2_0 + area*B2_1`` and ``(PX1*VZ13 + PX2*VZ24)/AREA``:
    mathematically identical, one different last bit.  On the c43 rolling
    benchmark that single bit is worth **34 % of the final internal
    energy** (measured: IE 1.93e5 -> 2.58e5 over 1185 steps — a 1605 ms
    free roll is chaos-dominated), so the refactor has to stay inert and
    this module is the transcribed, cited and unit-tested form the inline
    blocks are pinned against.
    """
    px1, px2 = px[..., 0], px[..., 1]
    py1, py2 = py[..., 0], py[..., 1]
    vz13, vz24 = vx[..., 0], vx[..., 1]

    if kind == 4:
        # cdefo3.F l.174-213, ELSEIF(IHBE==4)THEN.  DT1V4 = HALF*DT1C;
        # GZX = EXZ/AREA with EXZ = PX1*VZ13 + PX2*VZ24 (l.100), EYZ with
        # PY (l.99), AREA = 2*(PY2*PX1 - PY1*PX2) (cpxpy3.F l.82) — the
        # same EM20 floor kind 3 applies.  Unlike the other two kinds this
        # branch returns the per-node VELOCITY corrections, each half
        # contracting with the OPPOSITE operator pair (VX rows with PY,
        # VY rows with PX).  Handled before the dt == 0 shortcut on
        # purpose: the implicit guard here zeroes DT1V4 only, and the Z2
        # half does not carry it.
        if z2 is None:
            raise ValueError(
                "_rot2_correction(kind=4): Z2 is required.  It is the "
                "corotational out-of-plane coordinate of node 2 relative "
                "to node 1, (x2-x1).e3 (cderi3.F l.95, l.109) — zeroed on "
                "the ISMSTR == 3 stress-based path (cpxpy3.F l.77, l.88) "
                "— not a curvature from ccurv3.F, which never produces it.")
        area = np.maximum(2.0 * (py2 * px1 - py1 * px2), EM20)
        gzx = (px1 * vz13 + px2 * vz24) / area
        gzy = (py1 * vz13 + py2 * vz24) / area
        dt1v4 = 0.0 if implicit else 0.5 * dt
        zz2 = 0.5 * z2                                      # l.180
        exzz2 = gzx * zz2                                   # l.183
        eyzz2 = gzy * zz2                                   # l.193
        exz2 = gzx * gzx * dt1v4                            # l.184
        eyz2 = gzy * gzy * dt1v4                            # l.194
        exz2py2 = exz2 * py2                                # l.185
        exz2py1 = exz2 * py1                                # l.186
        eyz2px2 = eyz2 * px2                                # l.195
        eyz2px1 = eyz2 * px1                                # l.196
        return (np.stack([-exzz2 - exz2py2, exzz2 + exz2py1,
                          -exzz2 + exz2py2, exzz2 - exz2py1], axis=-1),
                np.stack([-eyzz2 + eyz2px2, eyzz2 - eyz2px1,
                          -eyzz2 - eyz2px2, eyzz2 + eyz2px1], axis=-1))

    if kind == 0 or dt == 0.0 or implicit:
        zero = np.zeros(np.shape(vz13), dtype=float)
        return zero, zero

    if kind == 1:
        # cdefo3.F l.107-126.  DT1V4 = FOURTH*DT1C; both denominators
        # carry the Fortran SIGN(MAX(ABS,EM20)) guard, so a zero
        # denominator is taken as +EM20 and a negative one flips sign.
        t3a = py1 + py2
        t3a = np.copysign(np.maximum(np.abs(t3a), EM20), t3a)
        dx = (0.25 * dt) * (vz13 - vz24) ** 2 / t3a
        t3b = px2 - px1
        t3b = np.copysign(np.maximum(np.abs(t3b), EM20), t3b)
        dy = (0.25 * dt) * (vz13 + vz24) ** 2 / t3b
        return dx, dy

    if kind == 3:
        # cdefo3.F l.136-150.  DT1V4 = HALF*DT1C; GZX = EXZ/AREA with
        # EXZ = PX1*VZ13 + PX2*VZ24 (l.100), EYZ = PY1*VZ13 + PY2*VZ24
        # (l.99), and AREA = 2*(PY2*PX1 - PY1*PX2) (cpxpy3.F l.82), the
        # floor being the same EM20 guard _local_geometry applies.
        area = np.maximum(2.0 * (py2 * px1 - py1 * px2), EM20)
        gzx = (px1 * vz13 + px2 * vz24) / area
        gzy = (py1 * vz13 + py2 * vz24) / area
        return (0.5 * dt) * gzx * gzx, (0.5 * dt) * gzy * gzy

    raise ValueError(f"_rot2_correction: unknown cdefo3 kind {kind!r}")