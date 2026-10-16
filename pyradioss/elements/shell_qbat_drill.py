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
#   engine/source/elements/shell/coqueba/cbadef.F  (CBADEFRZ; modified:
#     transcribed to NumPy)
#   engine/source/elements/shell/coqueba/cbafori.F (CBAFORRZ; modified:
#     transcribed to NumPy)

"""QBAT drilling stiffness, flat rows (P3.5) — split per the P1.7 budget.

``shell_qbat.py`` owns the ``forces()`` path and calls in here; this
module owns the drill branch bodies (``CBADEFRZ`` + ``CBAFORRZ``), like
``shell_qbat_nip1.py`` owns the single-point branch. The shared
B-operators live in ``shell_drill.py``.
"""

from __future__ import annotations

import numpy as np

from .shell_drill import KRZ_FAC, VPG, derirz, group_isrot, qbat_flat_mask


def warn_warped_drill(group, log, xe, E):
    """Loud init warning for the explicitly refused warped drill leg.

    The ``CBAFORRZ`` warped leg's ``BRZ`` rows (``CBABE3``/``CBADEF``
    warped) are unported, so warped rows run without drill stiffness.
    """
    if group_isrot(group) > 0 and log is not None:
        flat0 = qbat_flat_mask(xe, E, tol=1.0e-8)
        if not bool(np.all(flat0)):
            log.warning(
                f"QBAT Idrill>0: {int((~flat0).sum())} warped row(s) run "
                "WITHOUT drill stiffness (warped CBAFORRZ leg unported; "
                "flat rows carry the full CBADEFRZ/CBAFORRZ path)",
                "SHELL INIT")


# ----------------------------------------------------------------------------
# Drilling stiffness (P3.5): CBADEFRZ + CBAFORRZ, flat rows
# ----------------------------------------------------------------------------

def _drill_kinematics(g, vre):
    """P3.5 strain-side drill setup: RLZ + B-operators (CBADERIRZ).

    ``RLZ`` is the raw nodal spin about the element normal
    (``cbacoor.F:939-942`` — no ``/AREA``); the operators come from
    ``shell_drill.derirz`` (``CBADERIRZ0``). Warped rows keep zero
    drill state (explicitly refused — see the init warning).
    """
    n = g["n"]
    i_f = g["i_f"]
    rlz = np.zeros((n, 4))
    if len(i_f):
        rr = np.einsum("njk,nka->nja", vre, g["E"])
        rlz[i_f] = rr[i_f][:, :, 2]
        # NOTE: X13/X24/Y13/Y24 here are the AREA-NORMALIZED set
        # (cbaforc3.F passes CBACOOR's X13_T slot: cbacoor.F:470
        # X13_T = X13*AREA_I); MX/MY stay physical mid-sums.
        b0, bk, be = derirz(g["x13n"], g["x24n"], g["y13n"], g["y24n"],
                            g["mx13"], g["mx23"], g["mx34"],
                            g["my13"], g["my23"], g["my34"],
                            g["area"], qbat=True)
    else:
        b0 = bk = be = np.zeros((n, 4, 4))
    g.update(bm0rz=b0, bmksz=bk, bmerz=be, rlz=rlz,
             vrlz=np.zeros((n, 4)), vmz=np.zeros((n, 4)))


def _drill_gp_operators(g, ng):
    """BMRZ rows 1-3 + BRZ row 4 at Gauss point ``ng`` (CBADEFRZ)."""
    i = g["i_f"]
    ksi, eta = VPG[ng]
    b0, bk, be = g["bm0rz"][i], g["bmksz"][i], g["bmerz"][i]
    bmrz = np.empty((len(i), 3, 4))
    for r in range(3):
        bmrz[:, r, :] = b0[:, r, :] + bk[:, r, :] * ksi + be[:, r, :] * eta
    brz4 = b0[:, 3, :] + bk[:, 3, :] * ksi + be[:, 3, :] * eta
    return bmrz, brz4


def _drill_strain(g, vdef_, bm_ng, nip1, alive, ng):
    """CBADEFRZ flat leg: membrane augment + VRLZ, and the ISROT>0
    compatible-shear VDEF(3) replacing the CBADEFSH constant
    (``cbadef.F`` "no more constant shear")."""
    i = g["i_f"]
    v = g["vxyz_f"][i]
    bmrz, brz4 = _drill_gp_operators(g, ng)
    bxv2 = (bm_ng[:, 0] * v[:, 0, 1] + bm_ng[:, 1] * v[:, 1, 1]
            + bm_ng[:, 2] * v[:, 2, 1])
    byv1 = (bm_ng[:, 4] * v[:, 0, 0] + bm_ng[:, 5] * v[:, 1, 0]
            + bm_ng[:, 6] * v[:, 2, 0])
    rlz = g["rlz"][i]
    live = alive[i]
    use = live & ~nip1[i]
    if np.any(use):
        vdef_[i[use], 2] = (bxv2 + byv1)[use]
    if np.any(live):
        w = np.where(live)[0]
        rl = rlz[w]
        vdef_[i[w], 0] += (bmrz[w, 0] * rl).sum(axis=1)
        vdef_[i[w], 1] += (bmrz[w, 1] * rl).sum(axis=1)
        vdef_[i[w], 2] += (bmrz[w, 2] * rl).sum(axis=1)
        g["vrlz"][i[w], ng] = 0.5 * ((bxv2 - byv1)[w]
                                     + (brz4[w] * rl).sum(axis=1))


def _drill_forces(g, vf, bm_ng, npg_v, cdet_, thick, off, dt, ng, st, de,
                  alive, gshear):
    """CBAFORRZ flat leg (``cbafori.F:358``): VSRZ update with the
    trapezoidal EINT work, VMZ accumulation, and the shear+drill VF
    assembly (``C3``/``CSRZ``). Port assembly convention (``_fori_flat``
    docstring): ``C2 = VOL`` x stress-unit ``FF`` is ``cdet`` x the
    physical resultants ``npg_v``."""
    i = g["i_f"]
    live = alive[i]
    if not np.any(live):
        return
    w = np.where(live)[0]
    rows = i[w]
    vol = thick[rows] * cdet_[rows]
    erz = g["vrlz"][rows, ng] * dt
    dev = erz * vol * 0.5 * off[rows]
    krz = KRZ_FAC * gshear[rows]          # cncoef3.F: KRZ = KFAC * G
    vs_old = st["vsrz"][rows, ng]
    de[rows] += (2.0 * vs_old + krz * erz * off[rows]) * dev
    vs_new = vs_old + krz * erz * off[rows]
    st["vsrz"][rows, ng] = vs_new
    nn = npg_v[rows]
    c = cdet_[rows]
    bmrz, brz4 = _drill_gp_operators(g, ng)
    bmrz, brz4 = bmrz[w], brz4[w]
    b = bm_ng[w]
    c3 = c * nn[:, 2]
    csrz = vol * vs_new * 0.5
    vf[rows, 0, 0] += (c3 - csrz) * b[:, 4]
    vf[rows, 1, 0] += (c3 + csrz) * b[:, 0]
    fx3 = (c3 - csrz) * b[:, 6]
    fy3 = (c3 + csrz) * b[:, 2]
    vf[rows, 0, 2] += fx3
    vf[rows, 1, 2] += fy3
    vf[rows, 0, 1] += (c3 - csrz) * b[:, 5]
    vf[rows, 1, 1] += (c3 + csrz) * b[:, 1]
    vf[rows, 0, 3] = -vf[rows, 0, 2]
    vf[rows, 1, 3] = -vf[rows, 1, 2]
    vmz = g["vmz"]
    for j in range(4):
        vmz[rows, j] += (bmrz[:, 0, j] * nn[:, 0]
                         + bmrz[:, 1, j] * nn[:, 1]
                         + bmrz[:, 2, j] * nn[:, 2]) * c \
            + brz4[:, j] * (vol * vs_new * 0.5)
