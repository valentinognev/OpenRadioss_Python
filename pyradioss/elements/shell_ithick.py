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
#   engine/source/elements/shell/coqueba/cbaforc3.F  (modified: transcribed
#     to NumPy — the NPINCH==0, ITHK>0 thickness relaxation, l.863-867)
#   engine/source/elements/sh3n/coquedk/cncoef3.F  (modified: transcribed to
#     NumPy — the CNCOEF3/CNCOEF3B ITHK gate, THK0 = THK vs THKE)
#   engine/source/materials/mat/mat001/sigeps01c.F  (modified: transcribed
#     to NumPy — the elastic per-layer thinning, l.143-144)
#   common_source/modules/constant_mod.F  (THREE_OVER_4, l.748)

"""Shared ``Ithick=1`` thickness update for the QBAT/QEPH families (P3.6).

WHY THIS IS A SEPARATE MODULE
-----------------------------
``shell_qbat.py`` and ``shell_qeph.py`` both sit exactly at their P1.7 line
budgets, and the thickness update the plan mandates in both does not fit
inside either. This module carries the shared pieces — the per-layer
thinning increment, the CNCOEF3/CNCOEF3B ITHK flag gate, and the QBAT-only
relaxation — while each kernel keeps one import, one accumulator line, and
one apply call. No logic lives here that is not cited above.

Port mapping: ``state["thick"]`` is the assembly thickness THK0,
``state["thk0"]`` (QBAT) is the deck anchor THKE. The ITHK flag itself is
read off the shell property (``params["ithick"]``) at apply time, so no
init-time state is needed.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20

#: THREE_OVER_4 (common_source/modules/constant_mod.F:748).
_THREE_OVER_4 = 0.75


def _ithick_relax(thk_star, thk0):
    """QBAT thickness relaxation, cbaforc3.F lines 863-867 verbatim
    (``NPINCH == 0``, ``ITHK > 0``)::

        GBUF%THK(I) = GBUF%THK(I) - THREE_OVER_4*(GBUF%THK(I)-THK0(I))
        THK0(I) = GBUF%THK(I)

    ``thk_star`` is GBUF%THK after the material thinning, ``thk0`` the
    start-of-cycle THK0; the return value is both the new GBUF%THK and
    the new THK0."""
    return thk_star - _THREE_OVER_4 * (thk_star - thk0)


def thin_layer_increment(deps, wk, nu, off):
    """One layer's thinning contribution, sigeps01c.F lines 143-144::

        EZZ = -NU*(DEPSXX+DEPSYY)/(ONE-NU)
        THK = THK + EZZ*THKLYL*OFF

    Elastic law: plastic laws' extra ``dezz_pl`` is not surfaced by
    ``materials.shell_update`` (documented approximation). ``wk`` is the
    layer thickness THKLYL, ``off`` the element OFF gate (multiplicative,
    exactly as upstream — dead elements do not thin)."""
    ezz = -nu * (deps[:, 0] + deps[:, 1]) / np.maximum(1.0 - nu, EM20)
    return ezz * wk * off


def _ithick_mask(slices, n):
    """CNCOEF3/CNCOEF3B ITHK gate per element (``ITHK>0``, ``ISMDISP==0`` —
    the port runs explicit only): flagged elements track the evolving
    thickness, the rest reset THK0 to the deck THKE every cycle (i.e. are
    left alone here)."""
    mask = np.zeros(n, dtype=bool)
    for sl, _mat, prop in slices:
        params = getattr(prop, "params", {})
        if int(params.get("ithick", getattr(prop, "ithick", 0)) or 0) > 0:
            mask[sl] = True
    return mask


def apply_qbat_thickness(st, dthk, alive, has_pinch):
    """End-of-cycle QBAT thickness update (P3.6).

    ``dthk`` sums :func:`thin_layer_increment` over the 4 Gauss points
    because CMAIN3 runs once per GP in cbaforc3's NG loop; the
    cbaforc3.F:863-867 relaxation then keeps exactly one quarter of it
    on uniform states. ``NPINCH>0`` skips this — pinching owns the
    thickness via ``_cbapinchthk``. Applied after assembly so the cycle's
    forces use the start-of-cycle THK0; ITHK==0 leaves ``state["thick"]``
    bit-identical."""
    if has_pinch:
        return
    mask = _ithick_mask(st["slices"], len(dthk)) & alive
    if np.any(mask):
        start = st["thick"][mask]
        st["thick"][mask] = _ithick_relax(start + dthk[mask], start)


def apply_qeph_thickness(st, dthk, alive):
    """End-of-cycle QEPH thickness update (P3.6).

    One CMAIN3 call on the single-point path, so the full increment is
    taken — czforc3.F has no relaxation block (that block exists only in
    cbaforc3.F). Same post-assembly placement and ITHK==0 no-op as QBAT.
    """
    mask = _ithick_mask(st["slices"], len(dthk)) & alive
    if np.any(mask):
        st["thick"][mask] += dthk[mask]
