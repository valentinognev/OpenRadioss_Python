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
#     IMPL_S>0 disable carried as DT1C == 0, the IHBE==4 branch NOT ported —
#     see the deviation note below)
#   engine/source/elements/shell/coque/cdt3.F    (the DT1C condensation
#     length consumed as `dt`)
#   engine/source/elements/shell/coque/cpxpy3.F and cderi3.F  (the PX/PY
#     operator pairs and AREA the coefficients are built from)

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
node-1-relative branch:

  ``IF(IHBE<=1)THEN``            l.104-131  "general"              (kind 1)
  ``ELSEIF(IHBE==2.OR.IHBE==3)``  l.133-173  node-1-relative       (kind 3)
  ``ELSEIF(IHBE==4)THEN``        l.174-...  Z2 warp (Task P3.2)   (kind 4)

Both implemented branches share the frame-lag problem.  The corotational
frame is evaluated on the END-of-step geometry while the velocities sit at
mid-step, so an element rotating rigidly at rate ``w`` about an in-plane
axis measures a spurious membrane stretching rate of ``+w^2 dt/2``.  The
two branches then fix it in *different* ways, and upstream's asymmetry is
deliberate:

* kind 1 works on the DIFFERENCES of the out-of-plane diagonal
  velocities and **over-corrects 2x** (``TMP1A`` carries ``dt/4`` and
  ``(VZ13-VZ24)^2``, which on a stencil-exact roll is ``-w^2 dt``);
* kind 3 works on the out-of-plane membrane RATE
  ``EXZ = PX1*VZ13 + PX2*VZ24`` and is the **exact** cancellation
  (``EXZ2 = (EXZ/AREA)^2 * dt/2``).

M41 mirrored the kind-1 asymmetry on purpose; this module keeps the two
apart, because they are not interchangeable — a fact pinned by
``tests/test_p3_bt3_c43.py``.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20

#: cdefo3.F branch selector -> ``_rot2_correction`` ``kind``: 0 none,
#: 1 the ``IHBE <= 1`` general form, 3 the ``IHBE == 2 .OR. IHBE == 3``
#: node-1-relative velocity form, 4 the ``IHBE == 4`` Z2-warp form.
ROT2_KIND = {0: 0, 1: 1, 2: 3, 3: 3, 4: 4}


def rot2_correction(px, py, vx, vy, dt, kind=1):
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
        (parity case c43), 4 the ``IHBE == 4`` Z2-warp form.

    Returns
    -------
    (dx, dy) : (..., ) arrays
        the two upstream coefficient fields the caller contracts with the
        membrane operators, in upstream's own naming:

        ==========  ==============================  ==================
        kind        ``dx``                          ``dy``
        ==========  ==============================  ==================
        0           0                               0
        1           ``TMP1A`` (l.111)               ``TMP2B`` (l.122)
        3           ``EXZ2``  (l.142)               ``EYZ2``  (l.148)
        ==========  ==============================  ==================

    ``dt == 0`` is the implicit kill switch: every branch multiplies by
    ``DT1V4`` and ``IF(IMPL_S>0) DT1V4=ZERO`` (l.109, l.137, l.178), so
    the disable rides on the condensation length itself.

    WHY ``shell_bt4.forces`` DOES NOT CALL IT (Phase 3 Task P3.1)
    --------------------------------------------------------------
    The two inline blocks in ``forces()`` implement exactly these closed
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
        # cdefo3.F l.174-... (IHBE == 4, Z2 warp) — Phase 3 Task P3.2.
        # Refused loudly: a quiet zero here would read as a ported branch.
        raise NotImplementedError(
            "cdefo3.F kind=4 (IHBE==4 Z2 warp correction) is Phase 3 "
            "Task P3.2, not ported; forces() carries its own IHBE==4 "
            "block until then")

    if kind == 0 or dt == 0.0:
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