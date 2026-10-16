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
#   engine/source/elements/shell/coquez/czdef.F    (CZDEFRZ; modified:
#     transcribed to NumPy)
#   engine/source/elements/shell/coquez/czfintce.F (CZFINTCRZ; modified:
#     transcribed to NumPy)

"""QEPH drilling stiffness, flat rows (P3.5) — split per the P1.7 budget.

``shell_qeph.py`` owns the ``forces()`` path and calls in here; this
module owns the drill branch bodies (``CZDEFRZ`` + ``CZFINTCRZ``). The
shared B-operators live in ``shell_drill.py``.
"""

from __future__ import annotations

import numpy as np

from .shell_drill import KRZ_FAC, derirz, group_isrot


def warn_warped_drill(group, log, G):
    """Loud init warning for the explicitly refused warped drill leg.

    The warped ``CZPROJN`` VQN leg (``czproj.F:192-196,243-271``) is
    unported, so warped rows run without drill stiffness. The plate
    gate is ``Z1**2 < LM*TOL`` with ``TOL = 1e-8`` (``czcorc.F:135``).
    """
    if group_isrot(group) > 0 and log is not None:
        warped0 = ~(G["z1"] ** 2 < G["lm"] * 1.0e-8)
        if bool(np.any(warped0)):
            log.warning(
                f"QEPH Isrot>0: {int(warped0.sum())} warped row(s) run "
                "WITHOUT drill stiffness (warped CZPROJN leg unported; "
                "flat rows carry the full CZDEFRZ/CZFINTCRZ path)",
                "SHELL INIT")


def _drill_strain(G, vdef, vre, thick, dt, alive, plat, st):
    """CZDEFRZ (``czdef.F:212``) with the ``CZDERIRZ`` operators.

    ``rlz`` is the nodal spin about the element normal over the area
    (``czcorc.F:518-534``); the membrane augment and ``VRLZ`` use the
    scaled ``v13s``/``v24s`` set, matching ``_rates``' convention. Also
    carries the ``CZFINTCRZ`` constant-part update (``VSRZ(19)`` drill
    stress + trapezoidal ``EINT`` work — needs only ``dt``, no
    resultants). Stashes ``rlz``/``vrlz``/``bm0rz``/``vmz`` inputs on
    ``G`` for ``_drill_moments``/``_post``. Flat live rows only.
    """
    n = len(G["area"])
    flat = np.asarray(plat) & np.asarray(alive)
    rlz = np.zeros((n, 4))
    vrlz = np.zeros(n)
    b0 = np.zeros((n, 4, 4))
    if np.any(flat):
        rr = np.einsum("njk,nka->nja", vre, G["E"])
        rlz[flat] = rr[flat][:, :, 2] * G["a_i"][flat, None]
        b0, _bk, _be = derirz(G["x13"], G["x24"], G["y13"], G["y24"],
                              G["mx13"], G["mx23"], G["mx34"],
                              G["my13"], G["my23"], G["my34"],
                              G["area"])
        v13s, v24s = G["v13s"], G["v24s"]
        bxv2 = G["y24"] * v13s[:, 1] - G["y13"] * v24s[:, 1]
        byv1 = -G["x24"] * v13s[:, 0] + G["x13"] * v24s[:, 0]
        rl = rlz[flat]
        vdef[flat, 0] += (b0[flat, 0] * rl).sum(axis=1)
        vdef[flat, 1] += (b0[flat, 1] * rl).sum(axis=1)
        vdef[flat, 2] += (b0[flat, 2] * rl).sum(axis=1)
        vrlz[flat] = 0.5 * ((bxv2 - byv1)[flat]
                            + (b0[flat, 3] * rl).sum(axis=1))
        vol = G["area"][flat] * thick[flat]
        erz = vrlz[flat] * dt
        off01 = alive[flat].astype(float)
        off05 = 0.5 * off01 * vol
        vs = st["vsrz"]
        krz = KRZ_FAC * st["gmod"][flat]      # cncoef3.F: KRZ = KFAC * G
        st["eint"][flat] += vs[flat] * erz * off05
        vs[flat] = vs[flat] + krz * erz * off01
        st["eint"][flat] += vs[flat] * erz * off05
    G["rlz"], G["vrlz"], G["bm0rz"] = rlz, vrlz, b0


def _drill_moments(G, Nres, thick, alive, plat, st):
    """Nodal drill moments ``VMZ`` (``CZFINTCRZ`` force part)."""
    n = len(G["area"])
    vmz = np.zeros((n, 4))
    flat = np.asarray(plat) & np.asarray(alive)
    if np.any(flat):
        b0 = G["bm0rz"]
        vs = np.where(np.asarray(alive), st["vsrz"], 0.0)
        c2 = thick * vs * 0.5
        n1, n2, n3 = Nres[:, 0], Nres[:, 1], Nres[:, 2]
        for j in range(4):
            vmz[flat, j] = (b0[flat, 0, j] * n1[flat]
                            + b0[flat, 1, j] * n2[flat]
                            + b0[flat, 2, j] * n3[flat]) * thick[flat] \
                + b0[flat, 3, j] * c2[flat]
    return vmz
