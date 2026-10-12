# Copyright (C) 2026 the pyradioss contributors.
#
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit finite-element solver, and is licensed under the GNU
# Affero General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   starter/source/elements/sh3n/coque3n/c3inmas.F  (the PTG vertex-angle
#     fractions, c3inmas.F:1136-1150, and the element rotary inertia
#     XI = EM*(AREA/4.5 + THK^2/12), c3inmas.F:1116)
#   starter/source/elements/initia/spmd_msin.F      (the nodal assembly
#     MS += MSTG*PTG / IN += INTG*PTG, spmd_msin.F:557-558)
"""SH3N lumped-mass corner shares (P3.9).

Upstream distributes each triangle's lumped mass ``MSTG`` and rotary inertia
``INTG`` to its nodes by vertex-angle fractions ``PTG(K) = angle_K / PI``
(``c3inmas.F:1136-1150``, assembled in ``spmd_msin.F:557-558``).  A kernel
that scatters that way stores the table as ``group.state["ptg"]`` with the
element-total rotary inertia in ``group.state["dt_iner"]``; every other
kernel scatters uniform corner shares, keeps ``dt_iner`` as the per-node
share, and stores no table.

The helpers below read the table when present and evaluate the uniform
formulas with identical operation order when absent, so kernels without a
table are numerically untouched down to the last bit.
"""

from __future__ import annotations

import numpy as np


def _lump(state, k):
    """The PTG table when the angle-fraction path applies, else None."""
    lump = state.get("ptg")
    if lump is None or state.get("mass") is None:
        return None
    if lump.shape != (len(state["mass"]), k):
        return None
    return lump


def corner_shares(state, conn):
    """Nodal ``(mass-share, inertia-share)`` matrices for /DT/NODA claims.

    ``conn`` is the group's corner connectivity ``(n, k)``; both returns
    are ``(n, k)``, matching the contract ``NodalTimeStep`` assembles.
    """
    lump = _lump(state, conn.shape[1])
    mass = state["mass"][:, None]
    share = mass * lump if lump is not None else mass / conn.shape[1]
    iner = state.get("dt_iner")
    if iner is None:
        return share, None
    if lump is not None:
        return share, iner[:, None] * lump
    return share, np.broadcast_to(iner[:, None], conn.shape).copy()


def corner_ke(state, ve, vre, mask):
    """``(ke_trans, ke_rot)`` of the masked corners for /TH kinetic energy.

    ``ve``/``vre`` are the corner translational/rotational velocities
    gathered at the caller's (already masked) connectivity; ``vre`` is None
    when the model carries no rotational velocities.  ``mask`` selects the
    ``state`` rows the corners were gathered from.
    """
    mass = state.get("mass")
    lump = _lump(state, ve.shape[1]) if mass is not None else None
    ke_trans = 0.0
    if mass is not None:
        if lump is not None:
            v2c = np.einsum("nib,nib->ni", ve, ve)
            ke_trans = 0.5 * (mass[mask, None] * lump[mask] * v2c).sum()
        else:
            v2 = np.einsum("nib,nib->n", ve, ve) / ve.shape[1]
            ke_trans = 0.5 * (mass[mask] * v2).sum()
    ke_rot = 0.0
    if vre is not None and len(vre):
        if lump is not None and state.get("dt_iner") is not None:
            r2c = np.einsum("nib,nib->ni", vre, vre)
            ke_rot = 0.5 * (state["dt_iner"][mask, None] * lump[mask] * r2c).sum()
        elif "dt_iner" in state:
            vr2_sum = np.einsum("nib,nib->n", vre, vre)
            ke_rot = 0.5 * (state["dt_iner"][mask] * vr2_sum).sum()
        elif "inertia" in state:
            vr2_avg = np.einsum("nib,nib->n", vre, vre) / vre.shape[1]
            ke_rot = 0.5 * (state["inertia"][mask] * vr2_avg).sum()
    return ke_trans, ke_rot
