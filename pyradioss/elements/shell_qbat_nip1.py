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
#   engine/source/elements/shell/coqueba/cbafori.F  (modified: transcribed to
#     NumPy — the flat CBAFORI leg, moved here verbatim from shell_qbat.py,
#     plus the NPT==1 CBAFORI1 leg)
#   engine/source/elements/shell/coqueba/cbavisc.F  (modified: transcribed to
#     NumPy — the NPT==1 CBAVISNP1 leg)
#   engine/source/elements/shell/coqueba/cbacoor.F  (the 'OR NPT==1' flat gate
#     the nip1 row subset rests on, l.444)

"""QBAT flat force assembly and the ``nip=1`` single-point branch (P3.7).

WHY THIS IS A SEPARATE MODULE
-----------------------------
``shell_qbat.py`` sits exactly at its P1.7 line budget, and the single-point
branch (CBAFORI1/CBAVISNP1) the plan mandates there does not fit inside it.
This module carries the flat force-assembly legs — the standard CBAFORI leg
beside the NPT==1 CBAFORI1/CBAVISNP1 legs, which share its BM operator layout
and VF slot convention — while ``shell_qbat.py`` keeps only the import plus
the dispatch. No logic was changed in the move; the bodies are verbatim.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20


def _nip_array(n, slices, nip_vals):
    """Per-element NIP for a mixed-nip group (uniform groups store the
    scalar directly under ``state["NIP"]``)."""
    out = np.zeros(n, dtype=np.int64)
    for (sl, _mat, _prop), nv in zip(slices, nip_vals):
        out[sl] = nv
    return out


def _fori_flat(vf, vm, g, bm, bc, cdet, npg, mpg, q_pg):
    """CBAFORI flat branch (cbafori.F lines 73-119): physical resultants,
    factors C2 -> CDET (forces) and C1 -> CDET (moments) after the
    FF = sigma / MM = M/t^2 unit reductions cancel THK0/TH12."""
    i = g["i_f"]
    c = cdet[i]
    n1, n2 = npg[i, 0], npg[i, 1]
    m1, m2, m3 = mpg[i, 0], mpg[i, 1], mpg[i, 2]
    qx, qy = q_pg[i, 0], q_pg[i, 1]
    cm1 = c * (bm[:, 6] * m2 + bm[:, 2] * m3)
    cm2 = c * (bm[:, 2] * m1 + bm[:, 6] * m3)
    cc1 = c * (bc[:, 14] * qx + bc[:, 15] * qy)
    cc2 = c * (bc[:, 16] * qx + bc[:, 17] * qy)
    vf[i, 0, 0] += c * bm[:, 0] * n1
    vf[i, 1, 0] += c * bm[:, 4] * n2
    vf[i, 2, 0] += c * (bc[:, 0] * qx + bc[:, 1] * qy)
    vm[i, 0, 0] += c * (bc[:, 2] * qx + bc[:, 3] * qy) \
        - c * (bm[:, 4] * m2 + bm[:, 0] * m3)
    vm[i, 1, 0] += c * (bc[:, 4] * qx + bc[:, 5] * qy) \
        + c * (bm[:, 0] * m1 + bm[:, 4] * m3)
    vf[i, 0, 2] += c * bm[:, 2] * n1
    vf[i, 1, 2] += c * bm[:, 6] * n2
    vf[i, 2, 2] += c * (bc[:, 12] * qx + bc[:, 13] * qy)
    vm[i, 0, 2] += cc1 - cm1
    vm[i, 1, 2] += cc2 + cm2
    vf[i, 0, 1] += c * bm[:, 1] * n1
    vf[i, 1, 1] += c * bm[:, 5] * n2
    vf[i, 2, 1] += c * (bc[:, 6] * qx + bc[:, 7] * qy)
    vm[i, 0, 1] += c * (bc[:, 8] * qx + bc[:, 9] * qy) \
        - c * (bm[:, 5] * m2 + bm[:, 1] * m3)
    vm[i, 1, 1] += c * (bc[:, 10] * qx + bc[:, 11] * qy) \
        + c * (bm[:, 1] * m1 + bm[:, 5] * m3)
    # slot 4 = -slot 3 for the forces (assigned per cycle in the Fortran,
    # equivalent to mirroring the accumulation)
    vf[i, 0, 3] = -vf[i, 0, 2]
    vf[i, 1, 3] = -vf[i, 1, 2]
    vf[i, 2, 3] = -vf[i, 2, 2]
    vm[i, 0, 3] += cc1 + cm1
    vm[i, 1, 3] += cc2 - cm2


def _fori_np1(vf, g, bm, cdet_, npg, m1):
    """CBAFORI1 (cbafori.F, the NPT==1 leg): membrane-ONLY force assembly.

    Not the CBAFORI operator with fewer layers — the antisymmetric part goes
    to slots 1-2 and the symmetric part to slots 3-4 from FF(1:2) alone, and
    VM is never touched. ``bm`` is the full flat compact operator (aligned
    with ``g["i_f"]``); ``m1`` is the positional mask of the NPT==1 rows.
    C2 = VOL0 = THK0*CDET times the stress-unit FF equals CDET times the
    physical resultants, so ``c``/``n1``/``n2`` below are that product."""
    rows = g["i_f"][m1]
    b = bm[m1]
    c = cdet_[rows]
    n1, n2 = npg[rows, 0], npg[rows, 1]
    vf[rows, 0, 0] += c * b[:, 0] * n1      # VF(1,1) += C2*BM(1)*FF(1)
    vf[rows, 1, 0] += c * b[:, 4] * n2      # VF(2,1) += C2*BM(5)*FF(2)
    vf[rows, 0, 2] += c * b[:, 2] * n1      # VF(1,3) += C2*BM(3)*FF(1)
    vf[rows, 1, 2] += c * b[:, 6] * n2      # VF(2,3) += C2*BM(7)*FF(2)
    vf[rows, 0, 1] += c * b[:, 1] * n1      # VF(1,2) += C2*BM(2)*FF(1)
    vf[rows, 1, 1] += c * b[:, 5] * n2      # VF(2,2) += C2*BM(6)*FF(2)
    # slot 4 = -slot 3, assigned per Gauss point like the Fortran
    vf[rows, 0, 3] = -vf[rows, 0, 2]
    vf[rows, 1, 3] = -vf[rows, 1, 2]


def _cbavisnp1(vf, g, nip1, gshear, rho0, amu, off, thick, area, dt, dehg):
    """CBAVISNP1 (cbavisc.F, the NPT==1 leg): rotation-damping z-forces.

    The single through-thickness point cannot carry bending, so the
    out-of-plane hourglass rate (VHI z minus the rigid-rotation rate built
    from the RXYZ slots and the VCORE moment arms) is damped directly:
    opposite z-forces +-SS3 on slots 3/4 and the TESY work into the
    hourglass (EVIS(8)) ledger. HVL = TWO1000*AMU*sqrt(C2)*OFF with
    TWO1000 = 2000 (common_source/modules/constant_mod.F) and AMU = DN
    (cncoef3.F); C2 = G*RHO*AREA/12; HSURA = THK/AREA."""
    rows = np.where(nip1)[0]
    v = g["vxyz_f"][rows]                       # (m, slot, comp): 13/24/HI
    r = g["rxyz"][rows]                         # (m, slot4, xy): 13/24/HI/TI
    mx23, my23 = g["mx23"][rows], g["my23"][rows]
    mx34, my34 = g["mx34"][rows], g["my34"][rows]
    mx13, my13 = g["mx13"][rows], g["my13"][rows]
    vhxz = v[:, 2, 2] - (my13 * r[:, 3, 0]
                         - my23 * (r[:, 0, 0] + r[:, 1, 0])
                         - mx13 * r[:, 3, 1]
                         + mx23 * (r[:, 0, 1] + r[:, 1, 1])) * 0.25
    vhyz = v[:, 2, 2] - (my13 * r[:, 3, 0]
                         - my34 * (r[:, 0, 0] - r[:, 1, 0])
                         - mx13 * r[:, 3, 1]
                         + mx34 * (r[:, 0, 1] - r[:, 1, 1])) * 0.25
    c2 = gshear[rows] * rho0[rows] * area[rows] / 12.0
    hvl = 2000.0 * amu[rows] * np.sqrt(np.maximum(c2, 0.0)) * off[rows]
    cxz = (my34 * my34 + mx34 * mx34) * hvl
    cyz = (my23 * my23 + mx23 * mx23) * hvl
    hsura = thick[rows] / np.maximum(area[rows], EM20)
    sc5 = cxz * vhxz * hsura
    sc6 = cyz * vhyz * hsura
    ss3 = sc5 + sc6
    vf[rows, 2, 2] += ss3
    vf[rows, 2, 3] -= ss3
    dehg[rows] += (sc5 * vhxz + sc6 * vhyz) * dt
