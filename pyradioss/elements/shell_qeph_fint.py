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
#   engine/source/elements/shell/coquez/czfintce.F  (CZFINTCE; modified:
#     transcribed to NumPy)

"""QEPH constant force assembly (CZFINTCE) — split per the P1.7 budget.

``shell_qeph.py`` owns the ``forces()`` path; this module owns the
constant-part generalized forces, moved verbatim.
"""

from __future__ import annotations

import numpy as np


def _fint_const(G, thick, Nres, Mres, qres):
    """VF (n,3,4) / VM (n,2,4) with slots 0,1 = ANTISYM (node1/node2
    patterns: f1 = -f3, f2 = -f4) and slots 2,3 = SYM — czfintce.F with
    GBUF%FOR = N/t and GBUF%MOM = M/t^2 folded into physical resultants
    (THK*VSTRE -> Nres, THK02*MSTRE -> Mres)."""
    x13, x24, y13, y24 = G["x13"], G["x24"], G["y13"], G["y24"]
    mx13, my13 = G["mx13"], G["my13"]
    n = len(x13)
    VF = np.zeros((n, 3, 4))
    VM = np.zeros((n, 2, 4))
    qx, qy = qres[:, 0], qres[:, 1]          # t*sig_xz, t*sig_yz
    N1, N2, N3 = Nres[:, 0], Nres[:, 1], Nres[:, 2]
    M1, M2, M3 = Mres[:, 0], Mres[:, 1], Mres[:, 2]

    s1geo = G["my34"] * G["mx23"] - G["my23"] * G["mx34"]

    VF[:, 0, 0] = y24 * N1 - x24 * N3
    VF[:, 1, 0] = -x24 * N2 + y24 * N3
    VF[:, 2, 0] = -x24 * qy + y24 * qx       # VSTRE(4)=yz, VSTRE(5)=xz
    VM[:, 0, 0] = x24 * M2 - y24 * M3 - my13 * VF[:, 2, 0]
    VM[:, 1, 0] = y24 * M1 - x24 * M3 + mx13 * VF[:, 2, 0]
    VM[:, 0, 2] = -s1geo * qy                # -S52S
    VM[:, 1, 2] = s1geo * qx                 # +S42S
    VF[:, 0, 1] = -y13 * N1 + x13 * N3
    VF[:, 1, 1] = x13 * N2 - y13 * N3
    VF[:, 2, 1] = x13 * qy - y13 * qx
    VM[:, 0, 1] = -x13 * M2 + y13 * M3 + my13 * VF[:, 2, 1]
    VM[:, 1, 1] = -y13 * M1 + x13 * M3 - mx13 * VF[:, 2, 1]
    VM[:, 0, 3] = VM[:, 0, 2]
    VM[:, 1, 3] = VM[:, 1, 2]
    # warp coupling of the moments into in-plane forces (czfintce 95-100)
    c2 = G["z1"] * 4.0 * G["a_i"]
    VF[:, 0, 0] += c2 * (x13 * M1 + y13 * M3)
    VF[:, 1, 0] += c2 * (y13 * M2 + x13 * M3)
    VF[:, 0, 1] -= c2 * (x24 * M1 + y24 * M3)
    VF[:, 1, 1] -= c2 * (y24 * M2 + x24 * M3)
    return VF, VM


# ---------------------------------------------------------------------------
# physical stabilization (czfintn.F CZFINTN1) — the QEPH heart
# ---------------------------------------------------------------------------
