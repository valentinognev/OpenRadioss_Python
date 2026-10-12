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
#   engine/source/elements/shell/coqini.F    (the DATA Z01/WF1/WM1/ZN1
#     blocks, transcribed digit-for-digit; the column-major fill of the
#     Fortran declaration becomes the plain row-major reshape documented
#     below -- the ENGINE reads Z0(IT,NPT) as Z()[NPT-1][IT-1], so Python
#     row r is the (r+1)-point rule)
#   engine/source/elements/shell/err_thk.F  (the remesher thickness-error
#     estimator: shell_areas / node_thickness / thickness_error)
#   engine/source/elements/shell/fequilibre.F (the force-equilibrium
#     spread: force_equilibrium)

"""``coqini.F`` / ``err_thk.F`` / ``fequilibre.F`` — shell startup tables,
the remesher thickness-error estimator, and the force-equilibrium spread.

THE UPSTREAM CONTRACT
---------------------
``coqini.F`` (``SUBROUTINE COQINI``, called by ``radioss2.F``) builds four
tables at engine startup out of ``DATA`` blocks and copies them into the
globals ``Z0``, ``WF``, ``WM`` and ``ZTH``.  The blocks are **data, not
formulas** -- they are transcribed digit-for-digit below and the spot-value
tests pin the transcription, so a slip is caught rather than argued about.
Fortran fills the declared shapes column-major, so the flat value stream
runs one *rule* at a time, padded to the declared column height; the Python
tables are the plain row-major reshape of that stream, which is exactly how
the ENGINE indexes the globals (``coqueba/cmatc3.F:780-781``:
``WF0 = WF(IT,NPT)`` -> ``WF()[NPT-1][IT-1]``; ``cfint_reg.F:211``:
``ZTH(K,NDDL)`` -> ``[...][NDDL-1][K-1]``).  Row ``r`` is the ``(r+1)``-point
rule, column ``k-1`` its ``k``-th entry.  ``ZN1`` declares ``ZN1(12,11)``
upstream, so its Python counterpart is the ``(11, 12)`` transpose of that
declaration, for consistency with the other three.

``err_thk.F`` is the thickness error *estimator* the remesher consumes: it
accumulates element/nodal areas and ``area*thickness``, hands them to
``admthke.F`` for adapted parts, and then computes a per-element error
indicator.  It emits nothing and aborts nowhere -- there is no ``EXIT``,
no ``STOP`` and no message call in the file, so the plan's "port the abort"
is a misreading of the ``ERR_`` name and **nothing aborts** here.  The
``admthke.F`` hand-off (adapted parts, ``NADMESH /= 0``) has no port
counterpart yet and is not reproduced; the functions below cover the
``IADM == 0`` path.

``fequilibre.F`` is the pre-solve step that spreads nodal accelerations
onto the elements and frees ``A``.  It has no tolerance, no reference
field and emits nothing, so the plan's ``(fint, fint_ref, tol)`` signature
is not implementable from this routine and has not been invented; the port
keeps upstream's own behaviour under upstream's own name.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM30

# ---------------------------------------------------------------------------
# coqini.F -- the startup tables (DATA, transcribed digit-for-digit)
#
# Each block below mirrors the Fortran DATA statement line for line (the
# statement labels 1..B mark the continuation groups in the file).  The
# values are the file's own tokens; the reshape at build time is the only
# transformation, and it is the column-major-fill -> row-major-read mapping
# documented in the module docstring.
# ---------------------------------------------------------------------------

_Z01_DATA = [
    # coqini.F:46-68 -- DATA Z01/  ->  Z0(11,11)   (copied at l.149-153)
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, 0.5, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, 0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.1666667, 0.1666667, 0.5, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.25, 0.0, 0.25, 0.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.3, -0.1, 0.1, 0.3,
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.3333333, -0.1666667, 0.0, 0.1666667,
    0.3333333, 0.5, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.3571429, -0.2142857, -0.0714286, 0.0714286,
    0.2142857, 0.3571429, 0.5, 0.0, 0.0, 0.0,
    -0.5, -0.375, -0.25, -0.125, 0.0,
    0.125, 0.25, 0.375, 0.5, 0.0, 0.0,
    -0.5, -0.3888889, -0.2777778, -0.1666667, -0.0555555,
    0.0555555, 0.1666667, 0.2777778, 0.3888889, 0.5, 0.0,
    -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
]

_WF1_DATA = [
    # coqini.F:70-98 -- DATA WF1/  ->  WF(11,11)   (copied at l.150-153)
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.5, 0.5, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.25, 0.5, 0.25, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.1666667, 0.3333333, 0.3333333, 0.1666667, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.125, 0.25, 0.25, 0.25, 0.125,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.1, 0.2, 0.2, 0.2, 0.2,
    0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0833333, 0.1666667, 0.1666667, 0.1666667, 0.1666667,
    0.1666667, 0.0833333, 0.0, 0.0, 0.0, 0.0,
    0.0714286, 0.1428571, 0.1428571, 0.1428571, 0.1428571,
    0.1428571, 0.1428571, 0.0714286, 0.0, 0.0, 0.0,
    0.0625, 0.125, 0.125, 0.125, 0.125,
    0.125, 0.125, 0.125, 0.0625, 0.0, 0.0,
    0.0555556, 0.1111111, 0.1111111, 0.1111111, 0.1111111,
    0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.0555556, 0.0,
    0.05, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.05,
]

_WM1_DATA = [
    # coqini.F:100-122 -- DATA WM1/  ->  WM(11,11)  (copied at l.151-153)
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.0833333, 0.0833333, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.0833333, 0.0, 0.0833333, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.0648148, -0.0555556, 0.0555556, 0.0648148, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.0520833, -0.0625, 0.0, 0.0625, 0.0520833,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.0433333, -0.0600000, -0.0200000, 0.0200000, 0.0600000,
    0.0433333, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
    -0.0370370, -0.0555556, -0.0277778, 0.0000000, 0.0277778,
    0.0555556, 0.0370370, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
    -0.0323129, -0.0510204, -0.0306122, -0.0102041, 0.0102041,
    0.0306122, 0.0510204, 0.0323129, 0.0000000, 0.0000000, 0.0000000,
    -0.0286458, -0.0468750, -0.0312500, -0.0156250, 0.0000000,
    0.0156250, 0.0312500, 0.0468750, 0.0286458, 0.0000000, 0.0000000,
    -0.0257202, -0.0432099, -0.0308642, -0.0185185, -0.0061728,
    0.0061728, 0.0185185, 0.0308642, 0.0432099, 0.0257202, 0.0000000,
    -0.0233333, -0.0400000, -0.0300000, -0.0200000, -0.0100000,
    0.0000000, 0.0100000, 0.0200000, 0.0300000, 0.0400000, 0.0233333,
]

_ZN1_DATA = [
    # coqini.F:124-148 -- DATA ZN1/  ->  ZTH(12,11)  (copied at l.155-158);
    # declared ZN1(12,11) upstream, so the Python table is its (11,12)
    # transpose -- see the module docstring
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, 0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.25, 0.25, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.3333333, 0.0, 0.3333333, 0.5, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.375, -0.125, 0.125, 0.375, 0.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.4, -0.2, 0.0, 0.2, 0.4,
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.4166667, -0.25, -0.0833333, 0.0833333, 0.25,
    0.4166667, 0.5, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.4285715, -0.2857143, -0.1428572, 0.0, 0.1428572,
    0.2857143, 0.4285715, 0.5, 0.0, 0.0, 0.0,
    -0.5, -0.4375, -0.3125, -0.1875, -0.0625, 0.0625,
    0.1875, 0.3125, 0.4375, 0.5, 0.0, 0.0,
    -0.5, -0.4444444, -0.3333333, -0.2222222, -0.1111111, 0.0,
    0.1111111, 0.2222222, 0.3333333, 0.4444444, 0.5, 0.0,
    -0.5, -0.45, -0.35, -0.25, -0.15, -0.05,
    0.05, 0.15, 0.25, 0.35, 0.45, 0.5,
]


def _startup_table(data, shape):
    """Build one ``coqini`` table: the flat DATA stream reshaped row-major
    (Python row = rule, see the module docstring) and frozen read-only --
    the ENGINE builds these once at startup and hands them out shared."""
    table = np.array(data, dtype=float).reshape(shape)
    table.flags.writeable = False
    return table


_Z01 = _startup_table(_Z01_DATA, (11, 11))
_WF1 = _startup_table(_WF1_DATA, (11, 11))
_WM1 = _startup_table(_WM1_DATA, (11, 11))
_ZN1 = _startup_table(_ZN1_DATA, (11, 12))


def Z01():
    """``coqini.F:46-68`` -- the ``Z0`` integration-abscissa table, (11, 11).

    Row ``r`` holds the equispaced natural coordinates on [-0.5, +0.5] of
    the ``(r+1)``-point rule, zero-padded; read-only shared singleton.
    """
    return _Z01


def WF1():
    """``coqini.F:70-98`` -- the ``WF`` integration-weight table, (11, 11).

    Row ``r`` holds the ``(r+1)``-point rule's weights (each row sums to
    exactly 1); read-only shared singleton.
    """
    return _WF1


def WM1():
    """``coqini.F:100-122`` -- the ``WM`` table, (11, 11).

    The engine combines it with ``Z0`` as ``WM0 = Z0(IT,NPT)*WM(IT,NPT)``
    (``coqueba/cmatc3.F:780-781``); read-only shared singleton.
    """
    return _WM1


def ZN1():
    """``coqini.F:124-148`` -- the ``ZTH`` Gauss-coordinate table, (11, 12).

    Declared ``ZN1(12,11)`` upstream, so this is the transpose of that
    declaration: row ``r`` is the ``(r+1)``-point rule, column ``k-1`` its
    ``k``-th entry, matching ``cfint_reg.F:209-211``; read-only shared
    singleton.
    """
    return _ZN1


# ---------------------------------------------------------------------------
# err_thk.F -- the remesher thickness-error estimator
# ---------------------------------------------------------------------------


def shell_areas(conn, coords):
    """``err_thk.F:155-188`` (SH4) and ``205-239`` (SH3) -- per-element areas.

    SH4: ``AREA = |(X3-X1) x (X4-X2)| / 8``  (the diagonal cross product,
    ``ONE_OVER_8`` at l.170-172).  SH3: ``AREA = |(X3-X1) x (X3-X2)| / 6``
    (``ONE_OVER_6`` at l.223-225).  The element kind is inferred from the
    connectivity's column count.  The upstream scale is reproduced
    literally -- a unit square scores 0.25, not 1.0 -- because
    ``AREA_SH4``/``AREA_SH3`` are exported to the remesher as-is; the
    scale cancels only inside ``thickness_error``'s ratio.
    """
    conn = np.asarray(conn)
    coords = np.asarray(coords, dtype=float)
    if conn.shape[1] == 4:
        d1 = coords[conn[:, 2]] - coords[conn[:, 0]]
        d2 = coords[conn[:, 3]] - coords[conn[:, 1]]
        return np.linalg.norm(np.cross(d1, d2), axis=1) / 8.0
    d1 = coords[conn[:, 2]] - coords[conn[:, 0]]
    d2 = coords[conn[:, 2]] - coords[conn[:, 1]]
    return np.linalg.norm(np.cross(d1, d2), axis=1) / 6.0


def node_thickness(conn, coords, thick, active=None):
    """``err_thk.F:139-188`` (SH4) and ``199-239`` (SH3) -- the nodal
    deposits.

    Returns ``(AREA_NOD, THICK_NOD)`` over every node: each element adds
    its area to its corners' ``AREA_NOD`` and ``area * thickness`` to their
    ``THICK_NOD`` (l.181-188, l.234-239).  Elements with ``active=False``
    are skipped entirely, the port of upstream's ``GBUF%OFF(I) <= ZERO``
    guard (l.139, l.199).
    """
    conn = np.asarray(conn)
    coords = np.asarray(coords, dtype=float)
    thick = np.asarray(thick, dtype=float)
    if active is None:
        active = np.ones(conn.shape[0], dtype=bool)
    active = np.asarray(active, dtype=bool)
    area = np.where(active, shell_areas(conn, coords), 0.0)
    at = area * thick
    area_nod = np.zeros(coords.shape[0])
    thick_nod = np.zeros(coords.shape[0])
    for corner in range(conn.shape[1]):
        np.add.at(area_nod, conn[:, corner], area)
        np.add.at(thick_nod, conn[:, corner], at)
    return area_nod, thick_nod


def thickness_error(conn, area_nod, thick_nod, thickness, active=None):
    """``err_thk.F:296-334`` -- the per-element remesher error indicator.

    ``ERR_THK_SH4(N) = (1/4) * sum_k |THICK_NOD(K)/MAX(EM30, AREA_NOD(K))
    * (1/THICK_SH4(N)) - 1|`` over the element's four corners (l.307-313);
    the SH3 branch is the same with ``1/3`` and three corners (l.329-334).
    The indicator is normalised by the element's own thickness, so the
    same inconsistent neighbourhood flags a thin element harder than a
    thick one.  Elements with ``thickness <= 0`` are skipped -- the
    indicator stays at its initialised zero, exactly where upstream leaves
    ``GBUF%OFF(I) <= ZERO`` elements (l.109-110) -- because ``1/THICK``
    is a singularity there.  ``active`` masks further elements (eroded
    ones) without inventing upstream's ``MLW``-based skip.
    """
    conn = np.asarray(conn)
    thickness = np.asarray(thickness, dtype=float)
    if active is None:
        active = thickness > 0.0
    active = np.asarray(active, dtype=bool)
    err = np.zeros(conn.shape[0])
    act = np.nonzero(active)[0]
    if act.size == 0:
        return err
    corners = conn[act]
    unt = 1.0 / thickness[act]
    ratio = thick_nod[corners] / np.maximum(EM30, area_nod[corners])
    tn = np.abs(ratio * unt[:, None] - 1.0)
    if conn.shape[1] == 4:
        err[act] = 0.25 * tn.sum(axis=1)
    else:
        err[act] = (1.0 / 3.0) * tn.sum(axis=1)
    return err


# ---------------------------------------------------------------------------
# fequilibre.F -- the force-equilibrium spread
# ---------------------------------------------------------------------------


def force_equilibrium(acc, conn4=None, conn3=None):
    """``fequilibre.F:52-122`` -- spread nodal accelerations onto the shell
    elements and free ``A``.

    ``WA(K)`` counts every shell corner incidence on node K (l.52-64);
    ``FZERO(L,J,I) = A(L,K) / WA(K)`` for each quad corner (l.78-92) and
    each triangle corner (l.94-104); ``A(L,K) = 0`` at every shell node
    (l.106-122).  ``acc`` is the ``A(3,*)`` acceleration array, modified
    in place.  Returns ``(fzero4, fzero3)`` with shapes ``(3, 4, nquad)``
    and ``(3, 3, ntri)`` -- the Fortran stores both in one
    ``FZERO(3, 4, NUMELC+NUMELTG)`` array, SH3 in the first three corner
    slots at offset ``NUMELC``; each entry is ``None`` when the
    corresponding connectivity was not supplied.  Cheap enough to leave
    switched on: three bincounts' worth of work per call, no allocation
    beyond the two result arrays.
    """
    acc = np.asarray(acc, dtype=float)
    nnodes = acc.shape[1]
    wa = np.zeros(nnodes, dtype=float)
    if conn4 is not None:
        wa += np.bincount(np.asarray(conn4).ravel(), minlength=nnodes)
    if conn3 is not None:
        wa += np.bincount(np.asarray(conn3).ravel(), minlength=nnodes)

    fzero4 = None
    if conn4 is not None:
        conn4 = np.asarray(conn4)
        # upstream FZERO always has four corner slots per quad; a
        # degenerate connectivity is padded with its own edge column
        if conn4.shape[1] < 4:
            conn4 = np.pad(conn4, ((0, 0), (0, 4 - conn4.shape[1])), mode="edge")
        fzero4 = (acc[:, conn4] / wa[conn4]).transpose(0, 2, 1).copy()
    fzero3 = None
    if conn3 is not None:
        conn3 = np.asarray(conn3)
        if conn3.shape[1] < 3:
            conn3 = np.pad(conn3, ((0, 0), (0, 3 - conn3.shape[1])), mode="edge")
        fzero3 = (acc[:, conn3] / wa[conn3]).transpose(0, 2, 1).copy()

    shell_nodes = []
    if conn4 is not None:
        shell_nodes.append(np.asarray(conn4).ravel())
    if conn3 is not None:
        shell_nodes.append(np.asarray(conn3).ravel())
    if shell_nodes:
        acc[:, np.unique(np.concatenate(shell_nodes))] = 0.0
    return fzero4, fzero3
