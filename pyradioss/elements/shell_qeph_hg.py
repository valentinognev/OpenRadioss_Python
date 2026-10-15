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
#   engine/source/elements/shell/coquez/czbe3.F (added: CZBE3 lines
#     60-104 — PX1/PX2/PY1/PY2, PH1/PH2, SX/SY/RX/RY and HXX/HYY/HXY —
#     transcribed as the hourglass shape data below)
#   engine/source/elements/shell/coquez/czlken3.F (added: CZLKEN3 lines
#     72-140 — the CHM/CHF moduli and the GAMA outer-product membrane/
#     bending hourglass assembly)
#   engine/source/elements/shell/coquez/czfintn.F (added: the shear-modal
#     chain of CZFINTN1 lines 353-434 — generalized stresses sc5/sc6/ss3
#     and their nodal assembly — linearized here; the damper, plastic
#     relaxation and energy booking do not enter a stiffness)

"""QEPH implicit hourglass stiffness (P3.10).

``pyradioss/elements/shell_qeph.py`` is at its recorded line budget
(``tools/validation_data/size_budget.json``, 1600 lines) and may only
shrink, so the hourglass stiffness lives here and ``shell_qeph.tangent``
adds :func:`hg_tangent` to its constant part.

Three pieces, all local-frame, all exact linearizations of the verified
explicit chain (``czdef``/``czfintn`` as ported in ``shell_qeph``), proven
by ``tests/test_p3_qbat_qeph_tangent.py``:

* membrane/bending: ``CHM``/``CHF`` over ``GAMA`` (``CZLKEN3``) with
  ``GAMA = h/4 - PH`` — the port's orthogonalized ``gam`` is exactly
  ``4*GAMA`` (FD-checked on distorted quads), so this is a rescale, not
  a new operator;
* transverse: the shear-modal chain (``vhg[4:5]`` -> ``dg`` -> ``sc5``/
  ``sc6``/``ss3`` -> nodal forces AND the ``s1geo`` shear-moment lever),
  which is the same stiffness upstream's ``CZLKECT3`` ``HS`` block
  states directly (both give the FD-measured response).

Spin (drilling) regularization lives in ``shell_qeph.tangent`` (the
spin-coupled penalty, implicit-only like the BT4 term): the explicit
``Idrill = 0`` path has no spin response, and the penalty's translation
coupling is what keeps the assembled K at rank 18.
"""

from __future__ import annotations

import numpy as np

#: hourglass sign pattern h = (1,-1,1,-1).
_HPAT = np.array([1.0, -1.0, 1.0, -1.0])


def hg_tangent(G, a11, a12, gs, thick, area):
    """Local-frame (n,24,24) hourglass stiffness (see module docstring)."""
    n = len(area)
    a_i = G["a_i"]
    x13, y13 = G["x13"], G["y13"]
    mx13, my13 = G["mx13"], G["my13"]
    mx23, my23 = G["mx23"], G["my23"]
    mx34, my34 = G["mx34"], G["my34"]

    # CZBE3: PX/PY (the mean-gradient operators), PH, SX/SY/RX/RY, HXX/HYY/HXY
    px1 = G["y24"] * a_i
    px2 = -y13 * a_i
    py1 = -G["x24"] * a_i
    py2 = x13 * a_i
    ph1 = mx13 * px1 + my13 * py1
    ph2 = mx13 * px2 + my13 * py2
    gam = np.stack([0.25 - ph1, -0.25 - ph2,
                    0.25 + ph1, -0.25 + ph2], axis=1)
    sx = mx34 * a_i
    sy = my34 * a_i
    rx = mx23 * a_i
    ry = my23 * a_i
    hxx = (16.0 / 3.0) * (sy * sy + ry * ry)
    hyy = (16.0 / 3.0) * (sx * sx + rx * rx)
    hxy = -(16.0 / 3.0) * (sx * sy + rx * ry)

    # CZLKEN3: CHM/CHF moduli (hourglass stays elastic-isotropic per CZKE3)
    vol = area * thick
    chm11 = a11 * vol * hxx
    chm22 = a11 * vol * hyy
    chm12 = a12 * vol * hxy
    facf = thick * thick / 12.0
    chf11 = facf * chm22
    chf22 = facf * chm11
    chf12 = -facf * chm12

    kl = np.zeros((n, 24, 24), dtype=float)
    gg = np.einsum("ni,nj->nij", gam, gam)
    for a in range(4):
        for b in range(4):
            gab = gg[:, a, b]
            kl[:, 6 * a, 6 * b] += chm11 * gab
            kl[:, 6 * a + 1, 6 * b + 1] += chm22 * gab
            kl[:, 6 * a, 6 * b + 1] += chm12 * gab
            kl[:, 6 * a + 1, 6 * b] += chm12 * gab
            kl[:, 6 * a + 3, 6 * b + 3] += chf11 * gab
            kl[:, 6 * a + 4, 6 * b + 4] += chf22 * gab
            kl[:, 6 * a + 3, 6 * b + 4] += chf12 * gab
            kl[:, 6 * a + 4, 6 * b + 3] += chf12 * gab

    # shear-modal chain (czfintn 353-434 linearized at zero modal stress:
    # vhg[4:5] -> dg -> sc5/sc6/ss3 -> nodal w-forces and moments; the
    # damper, relaxation and energy terms do not enter a stiffness)
    p13 = np.array([1.0, 0.0, -1.0, 0.0])
    p24 = np.array([0.0, 1.0, 0.0, -1.0])
    one = np.array([1.0, 1.0, 1.0, 1.0])
    hw = 4.0 * _HPAT[None, :]
    h0x = -4.0 * my13[:, None] * one + 4.0 * my23[:, None] * (p13 + p24)
    h0y = -4.0 * mx23[:, None] * (p13 + p24) + 4.0 * mx13[:, None] * one
    h1x = -4.0 * my13[:, None] * one + 4.0 * my34[:, None] * (p13 - p24)
    h1y = -4.0 * mx34[:, None] * (p13 - p24) + 4.0 * mx13[:, None] * one
    c2s = gs / 64.0
    qxx = c2s * (4.0 * a_i * my34)
    qyy = c2s * (4.0 * a_i * mx34)
    qxx_k = c2s * (4.0 * a_i * my23)
    qyy_k = c2s * (4.0 * a_i * mx23)
    c2t = (4.0 / 3.0) * thick
    # d(sc5)/dU, d(sc6)/dU, d(ss3)/dU as local 24-vectors (per element row
    # below: W slots 6a+2, TX 6a+3, TY 6a+4)
    l5 = np.zeros((n, 24), dtype=float)
    l6 = np.zeros((n, 24), dtype=float)
    c5 = (c2t * (my34 * qxx + mx34 * qyy))[:, None]
    c6 = (c2t * (my23 * qxx_k + mx23 * qyy_k))[:, None]
    l5[:, 2::6] = c5 * hw
    l5[:, 3::6] = c5 * h0x
    l5[:, 4::6] = c5 * h0y
    l6[:, 2::6] = c6 * hw
    l6[:, 3::6] = c6 * h1x
    l6[:, 4::6] = c6 * h1y
    lss = l5 + l6
    s = _HPAT[None, :]
    my23c = my23[:, None]
    my34c = my34[:, None]
    my13c = my13[:, None]
    mx23c = mx23[:, None]
    mx34c = mx34[:, None]
    mx13c = mx13[:, None]
    kl[:, 2, :] += s[:, 0:1] * lss
    kl[:, 8, :] += s[:, 1:2] * lss
    kl[:, 14, :] += s[:, 2:3] * lss
    kl[:, 20, :] += s[:, 3:4] * lss
    kl[:, 3, :] += my23c * l5 + my34c * l6 - my13c * lss
    kl[:, 9, :] += my23c * l5 - my34c * l6 - my13c * lss
    kl[:, 15, :] += -my23c * l5 - my34c * l6 - my13c * lss
    kl[:, 21, :] += -my23c * l5 + my34c * l6 - my13c * lss
    kl[:, 4, :] += -mx23c * l5 - mx34c * l6 + mx13c * lss
    kl[:, 10, :] += -mx23c * l5 + mx34c * l6 + mx13c * lss
    kl[:, 16, :] += mx23c * l5 + mx34c * l6 + mx13c * lss
    kl[:, 22, :] += mx23c * l5 - mx34c * l6 + mx13c * lss
    return kl
