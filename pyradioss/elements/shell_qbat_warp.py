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
#   engine/source/elements/shell/coqueba/cbacoor.F  (warped geometry;
#     modified: transcribed to NumPy)
#   engine/source/elements/shell/coqueba/cbadef.F   (warped operators;
#     modified: transcribed to NumPy)
#   engine/source/elements/shell/coqueba/cbafori.F  (warped assembly;
#     modified: transcribed to NumPy)

"""QBAT warped-element branch — split per the P1.7 budget.

``shell_qbat.py`` owns the ``forces()`` path and the flat branch; this
module owns the warped geometry (``_warp_geometry``), the warped
per-Gauss operators (``_warp_gp``) and the warped force assembly
(``_fori_warp``), moved verbatim, plus the bilinear shape tables they
are built on. ``shell_qbat.py`` re-imports what the flat path shares.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20


#: 1/sqrt(3) — the Gauss abscissa PG of cbacoor.F/cbadef.F
_PG = 0.577350269189626

#: Gauss-point (ksi, eta) coordinates, cbadef.F DATA VPG (column-major):
#: GP order 1..4 = (-,-), (+,-), (+,+), (-,+)
_VPG = np.array([[-_PG, -_PG], [_PG, -_PG], [_PG, _PG], [-_PG, _PG]])

#: bilinear corner signs
_KSI_N = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA_N = np.array([-1.0, -1.0, 1.0, 1.0])

#: VKSI[j, ng] = dN_j/dksi at GP ng ; VETA[j, ng] = dN_j/deta — exactly
#: the cbacoor.F lines 167-198 table (checked term by term against the
#: closed form dN_j/dksi = ksi_j (1 + eta eta_j)/4).
_VKSI = 0.25 * _KSI_N[:, None] * (1.0 + _VPG[None, :, 1] * _ETA_N[:, None])
_VETA = 0.25 * _ETA_N[:, None] * (1.0 + _VPG[None, :, 0] * _KSI_N[:, None])


def _warp_geometry(g, rr, v13, v24, vhi):
    """The warped-element geometry: nodal frames VQN, edge normals VNRM +
    rotation arms VASTN, Gauss frames VQG + inverse-Jacobian columns VJFI,
    Gauss Jacobians, the warped AREA, the DI rigid-projection matrix and
    the projected nodal velocities/rotations (cbacoor.F 549-927/945-1075).
    All arrays compact over ``i_w``; results stored into ``g``."""
    iw = g["i_w"]
    m = len(iw)
    cx, cy = g["cx"][iw], g["cy"][iw]
    z1 = g["zl1"][iw]
    z2 = z1 * z1
    area = g["area"][iw]
    x13, x24 = g["x13"][iw], g["x24"][iw]
    y13, y24 = g["y13"][iw], g["y24"][iw]
    l13, l24 = g["l13"][iw], g["l24"][iw]
    mx13, my13 = g["mx13"][iw], g["my13"][iw]
    mx23, my23 = g["mx23"][iw], g["my23"][iw]
    mx34, my34 = g["mx34"][iw], g["my34"][iw]
    gama1, gama2 = g["gama1"][iw], g["gama2"][iw]

    # half-edge vectors (with warp z)
    x21 = mx23 - mx13
    x34 = 0.5 * (cx[:, 2] - cx[:, 3])
    y21 = my23 - my13
    y34 = 0.5 * (cy[:, 2] - cy[:, 3])
    l12 = np.sqrt(x21 ** 2 + y21 ** 2 + z2)
    l34 = np.sqrt(x34 ** 2 + y34 ** 2 + z2)
    x41 = mx34 - mx13
    x32 = 0.5 * (cx[:, 2] - cx[:, 1])
    y41 = my34 - my13
    y32 = 0.5 * (cy[:, 2] - cy[:, 1])
    a4 = 0.25 * area

    # ---- nodal frames VQN (t1 | t2 | n) ---------------------------------
    vqn = np.empty((m, 4, 9))

    def _t2(nrm, t1):
        return np.cross(nrm, t1)

    sl = 1.0 / np.maximum(l12, EM20)
    t1_12 = np.stack([x21 * sl, y21 * sl, -z1 * sl], axis=1)
    sz24 = z2 * l24
    sz2 = a4 - gama1
    sln = 1.0 / np.sqrt(np.maximum(sz24 + sz2 ** 2, EM20))
    n1 = np.stack([-z1 * y24 * sln, z1 * x24 * sln, sz2 * sln], axis=1)
    vqn[:, 0, 0:3] = t1_12
    vqn[:, 0, 6:9] = n1
    vqn[:, 0, 3:6] = _t2(n1, t1_12)

    sl = 1.0 / np.maximum(l34, EM20)
    t1_34 = np.stack([x34 * sl, y34 * sl, z1 * sl], axis=1)
    sz2 = a4 + gama1
    sln = 1.0 / np.sqrt(np.maximum(sz24 + sz2 ** 2, EM20))
    n3 = np.stack([z1 * y24 * sln, -z1 * x24 * sln, sz2 * sln], axis=1)
    vqn[:, 2, 0:3] = t1_34
    vqn[:, 2, 6:9] = n3
    vqn[:, 2, 3:6] = _t2(n3, t1_34)

    sz13 = z2 * l13
    sz2 = a4 + gama2
    sln = 1.0 / np.sqrt(np.maximum(sz13 + sz2 ** 2, EM20))
    n2 = np.stack([-z1 * y13 * sln, z1 * x13 * sln, sz2 * sln], axis=1)
    vqn[:, 1, 0:3] = t1_12
    vqn[:, 1, 6:9] = n2
    vqn[:, 1, 3:6] = _t2(n2, t1_12)

    sz2 = a4 - gama2
    sln = 1.0 / np.sqrt(np.maximum(sz13 + sz2 ** 2, EM20))
    n4 = np.stack([z1 * y13 * sln, -z1 * x13 * sln, sz2 * sln], axis=1)
    vqn[:, 3, 0:3] = t1_34
    vqn[:, 3, 6:9] = n4
    vqn[:, 3, 3:6] = _t2(n4, t1_34)

    # ---- edge normals VNRM + rotation arms VASTN ------------------------
    vnrm = np.empty((m, 4, 3))
    for j, (a, b) in enumerate(((0, 1), (3, 2), (0, 3), (1, 2))):
        s = vqn[:, a, 6:9] + vqn[:, b, 6:9]
        vnrm[:, j] = s / np.maximum(
            np.sqrt(np.einsum("nk,nk->n", s, s)), EM20)[:, None]
    vastn = np.zeros((m, 4, 4))
    vastn[:, 0, 1] = l12
    vastn[:, 0, 3] = l12
    vastn[:, 1, 1] = l34
    vastn[:, 1, 3] = l34
    e41 = np.stack([x41, y41, -z1], axis=1)
    vastn[:, 2, 0] = -np.einsum("nk,nk->n", e41, vqn[:, 0, 3:6])
    vastn[:, 2, 1] = np.einsum("nk,nk->n", e41, vqn[:, 0, 0:3])
    vastn[:, 2, 2] = -np.einsum("nk,nk->n", e41, vqn[:, 3, 3:6])
    vastn[:, 2, 3] = np.einsum("nk,nk->n", e41, vqn[:, 3, 0:3])
    e32 = np.stack([x32, y32, z1], axis=1)
    vastn[:, 3, 0] = -np.einsum("nk,nk->n", e32, vqn[:, 1, 3:6])
    vastn[:, 3, 1] = np.einsum("nk,nk->n", e32, vqn[:, 1, 0:3])
    vastn[:, 3, 2] = -np.einsum("nk,nk->n", e32, vqn[:, 2, 3:6])
    vastn[:, 3, 3] = np.einsum("nk,nk->n", e32, vqn[:, 2, 0:3])

    # ---- Gauss frames VQG + VJFI + Jacobians ----------------------------
    a4pg = a4 / _PG
    jmx13 = mx13 * _PG
    jmy13 = my13 * _PG
    jmz13 = z1 * _PG
    j2myz = jmz13 ** 2
    g1x1 = mx23 - jmx13
    g1y1 = my23 - jmy13
    c1g = np.sqrt(g1x1 ** 2 + g1y1 ** 2 + j2myz)
    g2x1 = mx34 - jmx13
    g2y1 = my34 - jmy13
    c2g = np.sqrt(g2x1 ** 2 + g2y1 ** 2 + j2myz)
    g1x3 = mx23 + jmx13
    g1y3 = my23 + jmy13
    j1g = np.sqrt(g1x3 ** 2 + g1y3 ** 2 + j2myz)
    g2x2 = mx34 + jmx13
    g2y2 = my34 + jmy13
    j2g = np.sqrt(g2x2 ** 2 + g2y2 ** 2 + j2myz)

    vqg = np.empty((m, 4, 9))
    vjfi = np.empty((m, 4, 6))
    jac_w = np.empty((m, 4))

    # the four in-plane covariant vectors of the warped mid-surface, as
    # FULL 3-vectors (z = -+jmz13). Expanding cbacoor.F's hand-written
    # blocks (lines 773-925) against these shows the exact structure
    #   VJFI(1..3,NG) = (g2 x n) / (JAC)      [after the SL/PG scale]
    #   VJFI(4..6,NG) = (n x g1) / (JAC)
    #   t1_raw        = g1 * |g2| + (g2 x n) * |g1| ,  t2 = n x t1
    # with (g1, g2) per GP:  NG=1 (g1m, g2m), NG=3 (g1p, g2p),
    # NG=2 (g1m, g2p), NG=4 (g1p, g2m) — the 'm/p' vectors carry
    # z = -jmz13 / +jmz13 respectively.
    g1m = np.stack([g1x1, g1y1, -jmz13], axis=1)
    g1p = np.stack([g1x3, g1y3, jmz13], axis=1)
    g2m = np.stack([g2x1, g2y1, -jmz13], axis=1)
    g2p = np.stack([g2x2, g2y2, jmz13], axis=1)

    def _gp(ng, sz, sz2, nraw, g1v, g2v, ca, cb):
        """One Gauss point: nraw = unnormalized in-plane normal part;
        g1v/g2v the covariant vectors; ca = |g1v|, cb = |g2v|."""
        slr = np.sqrt(sz + sz2 ** 2)
        jac_w[:, ng] = slr * _PG
        sli = 1.0 / np.maximum(slr, EM20)
        nrm = np.empty((m, 3))
        nrm[:, 0] = nraw[:, 0] * sli
        nrm[:, 1] = nraw[:, 1] * sli
        nrm[:, 2] = sz2 * sli
        vqg[:, ng, 6:9] = nrm
        v1 = np.cross(g2v, nrm)                  # VJFI rows 1-3 (raw scale)
        t1 = g1v * cb[:, None] + v1 * ca[:, None]
        sl2 = (sli / _PG)[:, None]
        vjfi[:, ng, 0:3] = v1 * sl2
        vjfi[:, ng, 3:6] = np.cross(nrm, g1v) * sl2
        nt = np.sqrt(np.einsum("nk,nk->n", t1, t1))
        t1 /= np.where(nt == 0.0, 1.0, nt)[:, None]
        vqg[:, ng, 0:3] = t1
        vqg[:, ng, 3:6] = np.cross(nrm, t1)

    nraw1 = np.stack([-z1 * y24, z1 * x24], axis=1)
    nraw2 = np.stack([-z1 * y13, z1 * x13], axis=1)
    _gp(0, sz24, a4pg - gama1, nraw1, g1m, g2m, c1g, c2g)
    _gp(2, sz24, a4pg + gama1, -nraw1, g1p, g2p, j1g, j2g)
    _gp(1, sz13, a4pg + gama2, nraw2, g1m, g2p, c1g, j2g)
    _gp(3, sz13, a4pg - gama2, -nraw2, g1p, g2m, j1g, c2g)

    # warped element AREA := sum of Gauss Jacobians (cbacoor.F line 926)
    g["area"][iw] = jac_w.sum(axis=1)
    g["jac"][iw] = jac_w

    # ---- rigid-mode projection of the velocities (945-1075) -------------
    rr_w = rr[iw]                                # (m, node, comp)
    v13w, v24w, vhiw = v13[iw], v24[iw], vhi[iw]
    ar = np.empty((m, 3))
    ar[:, 0] = (-z1 * vhiw[:, 1] + y13 * v13w[:, 2] + y24 * v24w[:, 2]
                + my13 * vhiw[:, 2] + rr_w[:, :, 0].sum(axis=1))
    ar[:, 1] = (z1 * vhiw[:, 0] - x13 * v13w[:, 2] - x24 * v24w[:, 2]
                - mx13 * vhiw[:, 2] + rr_w[:, :, 1].sum(axis=1))
    ar[:, 2] = (x13 * v13w[:, 1] + x24 * v24w[:, 1] + mx13 * vhiw[:, 1]
                - y13 * v13w[:, 0] - y24 * v24w[:, 0] - my13 * vhiw[:, 0]
                + rr_w[:, :, 2].sum(axis=1))
    xx1 = (cx ** 2).sum(axis=1)
    yy = (cy ** 2).sum(axis=1)
    xy = (cx * cy).sum(axis=1)
    hi_x = cx[:, 0] - cx[:, 1] + cx[:, 2] - cx[:, 3]
    hi_y = cy[:, 0] - cy[:, 1] + cy[:, 2] - cy[:, 3]
    xz1 = hi_x * z1
    yz = hi_y * z1
    zz = 4.0 * z2
    d1 = yy + zz + 4.0
    d2 = xx1 + zz + 4.0
    d3 = xx1 + yy + 4.0
    d4 = -xy
    d5 = -xz1
    d6 = -yz
    abc = d1 * d2 * d3
    xxyz2 = d1 * d6 * d6
    yyxz2 = d2 * d5 * d5
    zzxy2 = d3 * d4 * d4
    deta = abc + 2.0 * d4 * d5 * d6 - xxyz2 - yyxz2 - zzxy2
    deta = np.where(deta < EM20, 1.0, 1.0 / np.where(deta < EM20, 1.0, deta))
    di = np.empty((m, 6))
    di[:, 0] = (abc - xxyz2) * deta / d1
    di[:, 1] = (abc - yyxz2) * deta / d2
    di[:, 2] = (abc - zzxy2) * deta / d3
    di[:, 3] = (d5 * d6 - d4 * d3) * deta
    di[:, 4] = (d6 * d4 - d5 * d2) * deta
    di[:, 5] = (d4 * d5 - d6 * d1) * deta
    alr = np.empty((m, 3))
    alr[:, 0] = di[:, 0] * ar[:, 0] + di[:, 3] * ar[:, 1] + di[:, 4] * ar[:, 2]
    alr[:, 1] = di[:, 3] * ar[:, 0] + di[:, 1] * ar[:, 1] + di[:, 5] * ar[:, 2]
    alr[:, 2] = di[:, 4] * ar[:, 0] + di[:, 5] * ar[:, 1] + di[:, 2] * ar[:, 2]

    v13p = np.empty((m, 3))
    v24p = np.empty((m, 3))
    vhip = np.empty((m, 3))
    v13p[:, 0] = 0.5 * v13w[:, 0] + alr[:, 2] * y13
    v24p[:, 0] = 0.5 * v24w[:, 0] + alr[:, 2] * y24
    vhip[:, 0] = 0.25 * vhiw[:, 0] + (alr[:, 2] * my13 - z1 * alr[:, 1])
    v13p[:, 1] = 0.5 * v13w[:, 1] - alr[:, 2] * x13
    v24p[:, 1] = 0.5 * v24w[:, 1] - alr[:, 2] * x24
    vhip[:, 1] = 0.25 * vhiw[:, 1] - (alr[:, 2] * mx13 - z1 * alr[:, 0])
    v13p[:, 2] = 0.5 * v13w[:, 2] - (y13 * alr[:, 0] - x13 * alr[:, 1])
    v24p[:, 2] = 0.5 * v24w[:, 2] - (y24 * alr[:, 0] - x24 * alr[:, 1])
    vhip[:, 2] = 0.25 * vhiw[:, 2] + (mx13 * alr[:, 1] - my13 * alr[:, 0])

    vxyz_w = np.empty((m, 4, 3))                 # per-NODE local velocities
    vxyz_w[:, 0] = v13p + vhip
    vxyz_w[:, 1] = v24p - vhip
    vxyz_w[:, 2] = -v13p + vhip
    vxyz_w[:, 3] = -v24p - vhip

    rrc = rr_w - alr[:, None, :]
    rxyz_w = np.empty((m, 4, 2))                 # nodal 2-dof rotations
    for j in range(4):
        rxyz_w[:, j, 0] = np.einsum("nk,nk->n", vqn[:, j, 0:3], rrc[:, j])
        rxyz_w[:, j, 1] = np.einsum("nk,nk->n", vqn[:, j, 3:6], rrc[:, j])

    corel = np.empty((m, 3, 4))                  # cbaproj COREL (comp,node)
    corel[:, 0, :] = cx
    corel[:, 1, :] = cy
    corel[:, 2, 0] = z1
    corel[:, 2, 1] = -z1
    corel[:, 2, 2] = z1
    corel[:, 2, 3] = -z1

    g.update(vqn=vqn, vnrm=vnrm, vastn=vastn, vqg=vqg, vjfi=vjfi,
             vxyz_w=vxyz_w, rxyz_w=rxyz_w, di=di, corel=corel)


# ----------------------------------------------------------------------------
# per-Gauss-point operators + velocity strains (cbadef.F)
# ----------------------------------------------------------------------------

def _warp_gp(g, ng):
    """WARPED branch of CBADEF for Gauss point ``ng``: the per-node
    membrane rows BM (m,4,3,2), bending rows BMF (m,4,3,3) and BF
    (m,4,2,3), the shear operator BCQ (m,4,5,2), the 2x2 TC map and the
    velocity strains (cbadef.F lines 176-714). Compact over i_w."""
    iw = g["i_w"]
    m = len(iw)
    vqn = g["vqn"]
    vqg = g["vqg"][:, ng]
    vjfi = g["vjfi"][:, ng]
    v = g["vxyz_w"]
    r = g["rxyz_w"]

    tfn = np.zeros((m, 3, 2))
    for j in range(4):
        tfn[:, :, 0] += _VKSI[j, ng] * vqn[:, j, 6:9]
        tfn[:, :, 1] += _VETA[j, ng] * vqn[:, j, 6:9]
    f1, f2 = vjfi[:, 0:3], vjfi[:, 3:6]
    tbi = np.empty((m, 2, 2))
    tbi[:, 1, 1] = np.einsum("nk,nk->n", f1, tfn[:, :, 0])
    tbi[:, 1, 0] = np.einsum("nk,nk->n", f2, tfn[:, :, 0])
    tbi[:, 0, 1] = np.einsum("nk,nk->n", f1, tfn[:, :, 1])
    tbi[:, 0, 0] = np.einsum("nk,nk->n", f2, tfn[:, :, 1])
    thk = -(tbi[:, 0, 0] + tbi[:, 1, 1])
    tbi[:, 0, 1] = -tbi[:, 0, 1]
    tbi[:, 1, 0] = -tbi[:, 1, 0]

    t1g, t2g = vqg[:, 0:3], vqg[:, 3:6]
    tc = np.empty((m, 2, 2))
    tc[:, 0, 0] = np.einsum("nk,nk->n", f1, t1g)
    tc[:, 1, 0] = np.einsum("nk,nk->n", f2, t1g)
    tc[:, 0, 1] = np.einsum("nk,nk->n", f1, t2g)
    tc[:, 1, 1] = np.einsum("nk,nk->n", f2, t2g)
    tbc = np.einsum("nab,nbc->nac", tbi, tc)

    bmw = np.empty((m, 4, 3, 2))
    bmfw = np.empty((m, 4, 3, 3))
    bfw = np.empty((m, 4, 2, 3))
    vdef = np.zeros((m, 8))
    for j in range(4):
        vt1 = np.einsum("nk,nk->n", t1g, v[:, j])
        vt2 = np.einsum("nk,nk->n", t2g, v[:, j])
        c1 = _VKSI[j, ng] * tc[:, 0, 0] + _VETA[j, ng] * tc[:, 1, 0]
        c2 = _VKSI[j, ng] * tc[:, 0, 1] + _VETA[j, ng] * tc[:, 1, 1]
        bc1 = _VKSI[j, ng] * tbc[:, 0, 0] + _VETA[j, ng] * tbc[:, 1, 0]
        bc2 = _VKSI[j, ng] * tbc[:, 0, 1] + _VETA[j, ng] * tbc[:, 1, 1]
        bmw[:, j, :, 0] = t1g * c1[:, None]
        bmw[:, j, :, 1] = t2g * c2[:, None]
        bmfw[:, j, :, 0] = thk[:, None] * bmw[:, j, :, 0] + t1g * bc1[:, None]
        bmfw[:, j, :, 1] = thk[:, None] * bmw[:, j, :, 1] + t2g * bc2[:, None]
        bmfw[:, j, :, 2] = t1g * bc2[:, None] + t2g * bc1[:, None]
        vdef[:, 0] += c1 * vt1
        vdef[:, 1] += c2 * vt2
        vdef[:, 5] += bc1 * vt1
        vdef[:, 6] += bc2 * vt2
        vdef[:, 7] += bc1 * vt2 + bc2 * vt1
        v12 = np.einsum("nk,nk->n", t1g, vqn[:, j, 0:3])   # V1(2)
        v11 = -np.einsum("nk,nk->n", t1g, vqn[:, j, 3:6])  # V1(1)
        v22 = np.einsum("nk,nk->n", t2g, vqn[:, j, 0:3])   # V2(2)
        v21 = -np.einsum("nk,nk->n", t2g, vqn[:, j, 3:6])  # V2(1)
        rv1 = v11 * r[:, j, 0] + v12 * r[:, j, 1]
        rv2 = v21 * r[:, j, 0] + v22 * r[:, j, 1]
        bfw[:, j, 0, 0] = v11 * c1
        bfw[:, j, 0, 1] = v21 * c2
        bfw[:, j, 0, 2] = v11 * c2 + v21 * c1
        bfw[:, j, 1, 0] = v12 * c1
        bfw[:, j, 1, 1] = v22 * c2
        bfw[:, j, 1, 2] = v12 * c2 + v22 * c1
        vdef[:, 5] += c1 * rv1
        vdef[:, 6] += c2 * rv2
        vdef[:, 7] += c1 * rv2 + c2 * rv1
    # closing 2H*[B0] contribution: uses the CONSTANT assumed membrane
    # shear vdef3 (set by CBADEFSH before the GP loop — cbadef.F line 569)
    vdef[:, 5] += thk * vdef[:, 0]
    vdef[:, 6] += thk * vdef[:, 1]
    vdef[:, 7] += thk * g["vdef3"][iw]

    # ---- assumed transverse shear BCQ (m, node, dof(x,y,z,r1,r2), row) --
    bcq = np.zeros((m, 4, 5, 2))
    v11e = np.array([_VKSI[1, ng], _VKSI[2, ng], _VETA[3, ng], _VETA[2, ng]])
    edges = ((0, 1, 0, 0), (3, 2, 1, 0), (0, 3, 2, 1), (1, 2, 3, 1))
    for (na, nb, je, row) in edges:
        w = v11e[je]
        for c in range(3):
            coef = w * g["vnrm"][:, je, c]
            bcq[:, na, c, row] = -coef
            bcq[:, nb, c, row] = coef
        bcq[:, na, 3, row] = w * g["vastn"][:, je, 0]
        bcq[:, na, 4, row] = w * g["vastn"][:, je, 1]
        bcq[:, nb, 3, row] = w * g["vastn"][:, je, 2]
        bcq[:, nb, 4, row] = w * g["vastn"][:, je, 3]
    bcx = np.zeros(m)
    bcy = np.zeros(m)
    for j in range(4):
        bcx += np.einsum("nk,nk->n", bcq[:, j, 0:3, 0], v[:, j]) \
            + bcq[:, j, 3, 0] * r[:, j, 0] + bcq[:, j, 4, 0] * r[:, j, 1]
        bcy += np.einsum("nk,nk->n", bcq[:, j, 0:3, 1], v[:, j]) \
            + bcq[:, j, 3, 1] * r[:, j, 0] + bcq[:, j, 4, 1] * r[:, j, 1]
    vdef[:, 3] = tc[:, 0, 0] * bcx + tc[:, 1, 0] * bcy
    vdef[:, 4] = tc[:, 0, 1] * bcx + tc[:, 1, 1] * bcy
    return bmw, bmfw, bfw, bcq, tc, vdef


# ----------------------------------------------------------------------------
# QBAT Pinching (Through-Thickness Stretch) Routines
# ----------------------------------------------------------------------------

def _fori_warp(vf, vm, g, ops, cdet, npg, mpg, q_pg):
    """CBAFORI warped branch (cbafori.F lines 120-272)."""
    iw = g["i_w"]
    bmw, bmfw, bfw, bcq, tc = ops
    c = cdet[iw]
    nn = npg[iw]
    mm = mpg[iw]
    qx, qy = q_pg[iw, 0], q_pg[iw, 1]
    bcx = tc[:, 0, 0] * qx + tc[:, 0, 1] * qy
    bcy = tc[:, 1, 0] * qx + tc[:, 1, 1] * qy
    for j in range(4):
        for comp in range(3):
            vf[iw, comp, j] += c * (
                bmw[:, j, comp, 0] * nn[:, 0] + bmw[:, j, comp, 1] * nn[:, 1]
                + bcq[:, j, comp, 0] * bcx + bcq[:, j, comp, 1] * bcy
                + bmfw[:, j, comp, 0] * mm[:, 0]
                + bmfw[:, j, comp, 1] * mm[:, 1]
                + bmfw[:, j, comp, 2] * mm[:, 2])
        for a in range(2):
            vm[iw, a, j] += c * (
                bcq[:, j, 3 + a, 0] * bcx + bcq[:, j, 3 + a, 1] * bcy
                + bfw[:, j, a, 0] * mm[:, 0] + bfw[:, j, a, 1] * mm[:, 1]
                + bfw[:, j, a, 2] * mm[:, 2])


