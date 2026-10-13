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
#   engine/source/elements/shell/shell_offset_wm_ini.F90
#     (SUBROUTINE shell_offset_wm_ini, transcribed without reassociating
#     arithmetic: the fac -> lobatto -> pos -> wmi order below is the file's)

"""Shell offset / warping-membrane initialisation.

THE UPSTREAM CONTRACT
---------------------
``engine/source/elements/shell/shell_offset_wm_ini.F90`` computes the
warping-membrane (moment) weight of one through-thickness integration point
when the shell reference surface is offset from the mid-surface::

    wmi = 0
    if (npt > 1):
        fac = 1/(npt-1)
        lobatto = (1/2 - fac/3)*fac   if |Z01(ipt,npt)| == 1/2
        lobatto = fac                 otherwise
        pos = Z01(ipt,npt) + zoffset
        wmi = pos*lobatto

Its only caller in the open tree is
``engine/source/materials/mat_share/mulawc.F90:778`` (verified present;
``grep -n shell_offset_wm_ini mulawc.F90`` returns the ``use`` at l.246 and
the call at l.778), inside the ``zshift = GEO(199, pid)`` dispatch
(l.598, l.774-779): at ``zshift == 0`` the material takes the ``coqini_wm``
table instead, so this module is the offset branch of that ``if`` and
nothing else.  ``wmc`` accumulates the stress moments (l.2662-2664), hence
the offset shifts the moment weights while the lobatto (membrane) factor
takes no ``zoffset`` — ``initialise`` pins that direction.

The ``Z01`` abscissae are the canonical ``shell_basis.Z01()`` table (the
same ``coqini.F`` DATA block the routine's own ``z01`` parameter repeats
symbol by symbol); the edge entries ``±0.5`` are exact in both, which is
all the ``abs(z) == half`` branch needs.  Interior abscissae are the
table's 7-digit decimals, so agreement with the ``WM1`` table at zero
offset is to ``1e-6``, and the test says so.

WHAT IS NOT CLAIMED HERE
------------------------
* ``engine/source/interfaces/shell_offset/`` (``inter_offset_dim.F90``,
  ``inter_offset_ini.F90``, ``offset_nproj.F90``) is the contact offset
  treatment called by ``resol_init`` — a separate concern, not ported here.
* ``starter/source/elements/shell/shell_offset/shell_offset_ini.F90``
  (called by ``lectur``) flags offset elements and saves/zeroes
  ``GEO(199)`` after projection.  It is surveyed but deferred: its trigger
  (``/DEF_SHELL IOFFSET``) has no field in the port's ``read_def_shell``,
  so a port of it today would be unreachable.  Its reader-side half
  (``GEO(199) = ZSHIFT`` in ``hm_read_prop01.F:337`` and siblings) IS
  reachable — ``/PROP/TYPE51`` carries ``zshift`` and is legal on shells.

``initialise`` is the group-level entry point the Starter's element init
caches as ``group.state["wm_offset"]`` (and the engine material path
consumes per integration point): the per-element shifted moment-weight
table, zero rows wherever the part carries no offset keyword.
"""

from __future__ import annotations

import numpy as np

from .shell_basis import Z01

#: ``constant_mod`` values this routine uses (``use constant_mod, only``).
_HALF = 0.5
_THIRD = 1.0 / 3.0

#: ``Z01`` holds the rules ``1..11``; upstream would index out of bounds past
#: it, so the port refuses loudly instead (repo convention).
_MAXNPT = 11


def _check_rule(npt: int, ipt: int | None = None) -> None:
    """``ValueError`` where upstream would index past the ``Z01`` table."""
    if npt > _MAXNPT:
        raise ValueError(
            f"shell_offset_wm_ini: npt={npt} past the Z01 table "
            f"(rules 1..{_MAXNPT})"
        )
    if ipt is not None and npt > 1 and not 1 <= ipt <= npt:
        raise ValueError(
            f"shell_offset_wm_ini: ipt={ipt} names no point of the "
            f"{npt}-point rule"
        )


def membrane_weight(ipt: int, npt: int) -> float:
    """``shell_offset_wm_ini.F90``'s ``lobatto`` local: the membrane factor.

    Offset-free by construction — ``zoffset`` enters only through ``pos``
    (l.102), never here (l.95-101).  The ``npt <= 1`` zero mirrors the
    routine's ``IF(NPT>1)`` guard, which leaves ``wmi`` at its
    initialised zero.
    """
    _check_rule(npt, ipt)
    if npt <= 1:
        return 0.0
    fac = 1.0 / (npt - 1)
    if abs(Z01()[npt - 1, ipt - 1]) == _HALF:
        return (_HALF - _THIRD * fac) * fac
    return fac


def membrane_weights(npt: int) -> np.ndarray:
    """The ``lobatto`` factor at every point of the rule (offset-free)."""
    return np.array([membrane_weight(ipt, npt) for ipt in range(1, npt + 1)])


def wm_weight(ipt: int, npt: int, zoffset: float) -> float:
    """``shell_offset_wm_ini(ipt, npt, zoffset) -> wmi``, 1-based ``ipt``.

    Transcribed without reassociating arithmetic: ``fac``, then the
    ``abs(z01) == half`` branch, then ``pos``, then ``pos*lobatto``.
    """
    _check_rule(npt, ipt)
    wmi = 0.0
    if npt > 1:
        lobatto = membrane_weight(ipt, npt)
        pos = Z01()[npt - 1, ipt - 1] + zoffset
        wmi = pos * lobatto
    return wmi


def wm_weights(npt: int, zoffset: float) -> np.ndarray:
    """``wm_weight`` at every point ``1..npt`` of the rule."""
    return np.array([wm_weight(ipt, npt, zoffset) for ipt in range(1, npt + 1)])


def reference_positions(npt: int, zoffset: float) -> np.ndarray:
    """``pos = Z01(ipt,npt) + zoffset`` (l.102): the shifted reference
    abscissae the moment arm is measured from."""
    _check_rule(npt)
    if npt < 1:
        return np.zeros(0)
    return Z01()[npt - 1, :npt] + zoffset


def _part_zshift(prop) -> float:
    """``GEO(199, pid)``: ``zshift`` on the part property, ``0.0`` when the
    keyword carries none (``/PROP/SHELL`` has no such field)."""
    params = getattr(prop, "params", {}) or {}
    if "zshift" in params:
        value = params["zshift"]
    else:
        value = getattr(prop, "zshift", 0.0)
    return 0.0 if value is None else float(value)


def _part_npt(prop) -> int:
    """Through-thickness points on the part property (``/PROP/SHELL`` card
    3 ``N`` → ``nip``; composites carry no count in the port's reader, so
    the shell default 3 stands in, exactly as ``init_group`` resolves it).
    """
    params = getattr(prop, "params", {}) or {}
    value = params.get("nip", params.get("npt", getattr(prop, "nip",
                         getattr(prop, "npt", 3))))
    return int(value or 3)


def initialise(group, model=None) -> np.ndarray:
    """Per-element shifted warping-membrane weights, ``(nelem, npt)``.

    The group-level form of the ``mulawc.F90:774-779`` dispatch: rows are
    ``wm_weights(npt, zshift)`` where the part property carries an offset,
    and zeros where it does not — the exact ``zshift == 0`` comparison, so
    ``/PROP/SHELL`` groups (no keyword) come back all zeros.  Parts with
    fewer points than the widest are zero-padded past their ``npt``.  The
    table is cached as ``group.state["wm_offset"]`` for the engine
    material path and returned.  ``model`` is accepted for the Starter
    call signature and is currently unused.
    """
    nelem = group.n
    slices = group.state.get("slices", [])
    npts = [_part_npt(prop) for _, _, prop in slices]
    width = max(npts) if npts else 0
    table = np.zeros((nelem, width))
    for (sl, _, prop), npt in zip(slices, npts):
        zshift = _part_zshift(prop)
        if zshift == 0.0:
            continue  # the coqini_wm branch: no offset correction
        table[sl, :npt] = wm_weights(npt, zshift)
    group.state["wm_offset"] = table
    return table
