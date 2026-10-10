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
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later),
# engine/source/elements/solid/solide8s/{crframe_imp,crtrans_imp,transk,
# getuloc,s8xref_imp,s8sav3_imp,srcoor3_imp,srcoork_imp,s8sfint3_crimp}.F,
# transcribed to NumPy.

"""
The ``solide8s`` co-rotational frame — Task P2.10.

Fortran origin: ``$OR_SRC/engine/source/elements/solid/solide8s/``::

    crframe_imp.F      the frame: polar rotation of F = x_cur / X_ref  (stage 1)
    crtrans_imp.F      the 24x24 projection TRM built FROM that frame  (stage 2)
    transk.F           K_global = TRM^T K_local TRM                    (stage 2)
    s8sfint3_crimp.F   f_global = TRM^T f_local, and the stored rotated force
    s8sav3_imp.F       the reference SAV (node-1-relative coordinates)
    s8xref_imp.F       the reference seen in the frame (``XREF``)
    getuloc.F          local displacement  R^T (x - x_1) - SAV
    srcoor3_imp.F      the force-cycle caller of all of the above
    srcoork_imp.F      the stiffness-cycle caller (``S8SCOORK_IMP``)

What "crimp" is, and what it is not
-----------------------------------
The plan and the task brief both call this path *crimp / geometrical
imperfection*.  The Fortran does not support that reading, and the Fortran
wins.  ``CR`` is **co-rotational** and ``IMP`` is **implicit**: every routine
above lives in the implicit branch of the HA8 solid-shell (``JHBE == 17`` with
``IPARG(36) == 3`` in ``forint.F``), and there is no imperfection, no
perturbation field and no fabric crimp anywhere in it.  (The only ``crimp``
strings elsewhere in the upstream tree are the fabric crimp strains of
``/MAT/LAW214``, which have nothing to do with this element.)  The name
``imperfection_frame`` is kept because the plan fixes it as the public
interface; what it returns is the co-rotational frame.

The trap: two stages, not one
-----------------------------
``crframe_imp`` returns an orthonormal ``R``.  ``crtrans_imp`` does **not**
compute it — ``R`` is an *input* there — and what it builds is a 24x24 matrix
whose diagonal 3x3 blocks are ``R^T`` and whose off-diagonal blocks are the
derivative of ``R`` with respect to every nodal displacement.  Porting only
the first stage yields a frame that is orthonormal, right-handed, and
useless: a stiffness rotated by ``blockdiag(R^T)`` is not invariant under a
rigid rotation of the element, because it ignores that ``R`` itself moves when
the nodes do.  ``transk`` then applies the second stage to the local
stiffness, and ``s8sfint3_crimp`` applies its transpose to the local forces.

``corotational_transform`` is stage two, and
``tests/test_p2_solid_crimp.py`` holds it to the one property that separates
the two readings: the transform of a rigid-body displacement field is zero.

Frame convention
----------------
``R[e]`` is Fortran's ``R(:,:,e)``: its **columns** are the local axes in
global coordinates, so ``x_local = R^T (x - x_1)``.  ``getuloc.F`` and
``s8xref_imp.F`` both use ``UL = R(1,1)*U + R(2,1)*V + R(3,1)*W``, which is
``(R^T u)_1``.  ``R`` is the rotation of the polar decomposition
``F = R U``, never a Gram-Schmidt of the element edges (that would agree with
``R`` on a rigid motion and disagree on shear).

Reachability
------------
This module is **ported and unreachable from a deck**, and the reason is a
finding, not an omission.  The plan asks for a keyword reader for
``crframe``/``crtrans``.  No such starter reader exists: ``starter/`` and
``reader/`` contain no ``crframe`` or ``crtrans`` card, and the CFG tree has
none either.  The routines are reached purely by property flags
(``/PROP/TYPE14`` with ``Isolid = 17``/``19``, which the Starter turns into
``IHBE = 17``, ``IINT = 3`` and, for ``ICPRE``, ``1`` in
``hm_read_prop14.F``) and by the implicit driver setting ``INCONV``.  Writing
a ``/CRFRAME`` card would invent a deck format upstream does not parse, so
none is added; wiring ``Isolid = 17`` to this module belongs to the dispatch
task (P2.11).

What is not here
----------------
``s8ske3.F`` / ``s8slke3.F`` / ``s8sksig.F`` (the implicit element stiffness,
its ANS ``B`` matrix and the geometric stiffness) and the gather/OFF/ISMSTR
bookkeeping of ``srcoor3_imp`` beyond :func:`convected_coordinates`.  The
stiffness operators consume this module's ``TRM`` and ``V`` through
:func:`transk`; they are the implicit branch's, not the frame's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Constants as spelled in ``common_source/modules/constant_mod.F`` and in
# ``crframe_imp.F`` itself (``MILLE24 = 1024.0``).
THIRD = 1.0 / 3.0
TWO_THIRD = 2.0 / 3.0

#: ``(sign_r, sign_s, sign_t)`` of the eight shape functions, i.e. the
#: ``DNk_DX = (+-INVJ(1,.) +- INVJ(2,.) +- INVJ(3,.))/8`` pattern of
#: ``crtrans_imp.F`` lines 136-163.  Node 1 is ``(-,-,-)``; ``r`` runs 1->4,
#: ``s`` runs 1->5 and ``t`` runs 1->2.
_NODE_SIGNS = np.array([
    [-1, -1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, -1, -1],
    [-1, +1, -1],
    [-1, +1, +1],
    [+1, +1, +1],
    [+1, +1, -1],
], dtype=float)

#: ``DX_DR`` = (nodes 3,4,7,8) - (nodes 1,2,5,6); ``DX_DS`` = (5,6,7,8) -
#: (1,2,3,4); ``DX_DT`` = (2,3,6,7) - (1,4,5,8)  (``crframe_imp.F``), 0-based.
_PLUS_R, _MINUS_R = (2, 3, 6, 7), (0, 1, 4, 5)
_PLUS_S, _MINUS_S = (4, 5, 6, 7), (0, 1, 2, 3)
_PLUS_T, _MINUS_T = (1, 2, 5, 6), (0, 3, 4, 7)


# ---------------------------------------------------------------------------
# reference state: s8sav3_imp / s8xref_imp / getuloc
# ---------------------------------------------------------------------------

def _as_nodes(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape == (8, 3):
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (8, 3):
        raise ValueError(f"expected (n, 8, 3) nodal coordinates, got {x.shape}")
    return x


def save_reference(x, offg=None, sav=None) -> np.ndarray:
    """``S8SAV3_IMP``: ``SAV(I, 3*(k-1)+c) = X_k(c) - X_1(c)``, ``k = 2..8``.

    The write is guarded per element by ``ABS(OFFG(I)) <= ONE``: an element
    that is off (``OFFG`` beyond 1 in magnitude) keeps whatever ``sav`` held.
    Pass the previous ``sav`` to get that behaviour; without it an element
    that is off gets zeros, which is what a freshly allocated ``SAV`` holds.
    """
    x = _as_nodes(x)
    n = x.shape[0]
    new = (x[:, 1:, :] - x[:, :1, :]).reshape(n, 21)
    out = np.zeros((n, 21)) if sav is None else np.array(sav, dtype=float)
    if offg is None:
        write = np.ones(n, dtype=bool)
    else:
        write = np.abs(np.asarray(offg, dtype=float)) <= 1.0
    out[write] = new[write]
    return out


def _sav_nodes(sav) -> np.ndarray:
    """``SAV(NEL,21)`` as ``(n, 8, 3)`` with node 1 at the origin — the
    ``X0(1:8), Y0, Z0`` of ``crframe_imp.F`` lines 76-101."""
    sav = np.asarray(sav, dtype=float)
    if sav.ndim == 1:
        sav = sav[None]
    out = np.zeros((sav.shape[0], 8, 3))
    out[:, 1:, :] = sav.reshape(sav.shape[0], 7, 3)
    return out


def local_coordinates(x, R) -> np.ndarray:
    """``(n, 8, 3)`` ``R^T (x_k - x_1)`` — the ``UL/VL/WL`` of ``getuloc.F``
    and ``s8xref_imp.F`` before the reference is subtracted.  Node 1 is
    exactly zero."""
    x = _as_nodes(x)
    R = np.asarray(R, dtype=float)
    rel = x - x[:, :1, :]
    return np.einsum("nji,nkj->nki", R, rel)


def local_reference(x, R) -> np.ndarray:
    """``S8XREF_IMP``: ``XREF(NEL,21)``, the geometry in the frame ``R``."""
    loc = local_coordinates(x, R)
    return loc[:, 1:, :].reshape(loc.shape[0], 21)


def local_displacements(x, sav, R) -> np.ndarray:
    """``GETULOC``: ``(n, 8, 3)`` ``R^T (x_k - x_1) - SAV_k``; node 1 is
    zero by construction (``ULX1 = ULY1 = ULZ1 = ZERO``)."""
    loc = local_coordinates(x, R)
    loc[:, 1:, :] -= _sav_nodes(sav)[:, 1:, :]
    loc[:, 0, :] = 0.0
    return loc


def convected_coordinates(x, R, sav=None, ismstr: int = 4) -> np.ndarray:
    """The "change to the convected frame" block of ``SRCOOR3_IMP``
    (lines 513-612): ``ISMSTR == 1`` replaces the coordinates by ``SAV``
    (node 1 at the origin); ``ISMSTR == 2`` or ``4`` replaces them by
    ``R^T (x_k - x_1)``; any other value leaves them untouched."""
    x = _as_nodes(x)
    if ismstr == 1:
        if sav is None:
            raise ValueError("ISMSTR == 1 needs the saved reference (sav)")
        return _sav_nodes(sav)
    if ismstr in (2, 4):
        return local_coordinates(x, R)
    return x.copy()


# ---------------------------------------------------------------------------
# stage 1: crframe_imp
# ---------------------------------------------------------------------------

def reference_inverse_jacobian(sav) -> np.ndarray:
    """``INVJ(I,3,3)`` of ``crframe_imp.F`` lines 103-134 (also computed,
    identically, by ``srcoork_imp.F`` lines 177-201): the inverse of the
    reference Jacobian at the element centre, times 8.

    ``DX_DR`` and friends are the *unscaled* corner sums (``!DX_DR/EIGHT`` is
    commented out for the reference), so ``DETM1 = 8/DETJ0`` supplies the
    ``1/8`` that makes ``INVJ = inv(J_ref)`` for ``J_ref = sums/8``.
    ``INVJ[e, k, j]`` is ``d xi_k / d X_j`` (Fortran ``INVJ(e,k,j)``).
    """
    x0 = _sav_nodes(sav)
    dr = x0[:, _PLUS_R].sum(1) - x0[:, _MINUS_R].sum(1)       # (n, 3)
    ds = x0[:, _PLUS_S].sum(1) - x0[:, _MINUS_S].sum(1)
    dt = x0[:, _PLUS_T].sum(1) - x0[:, _MINUS_T].sum(1)
    DX_DR, DY_DR, DZ_DR = dr[:, 0], dr[:, 1], dr[:, 2]
    DX_DS, DY_DS, DZ_DS = ds[:, 0], ds[:, 1], ds[:, 2]
    DX_DT, DY_DT, DZ_DT = dt[:, 0], dt[:, 1], dt[:, 2]
    detj0 = (DX_DR * (DY_DS * DZ_DT - DZ_DS * DY_DT)
             - DX_DS * (DY_DR * DZ_DT - DY_DT * DZ_DR)
             + DX_DT * (DY_DR * DZ_DS - DY_DS * DZ_DR))
    detm1 = 8.0 * (1.0 / detj0)
    invj = np.empty((x0.shape[0], 3, 3))
    invj[:, 0, 0] = (DY_DS * DZ_DT - DZ_DS * DY_DT) * detm1
    invj[:, 1, 0] = (DZ_DR * DY_DT - DY_DR * DZ_DT) * detm1
    invj[:, 2, 0] = (DY_DR * DZ_DS - DY_DS * DZ_DR) * detm1
    invj[:, 0, 1] = (DX_DT * DZ_DS - DX_DS * DZ_DT) * detm1
    invj[:, 1, 1] = (DX_DR * DZ_DT - DX_DT * DZ_DR) * detm1
    invj[:, 2, 1] = (DX_DS * DZ_DR - DX_DR * DZ_DS) * detm1
    invj[:, 0, 2] = (DX_DS * DY_DT - DX_DT * DY_DS) * detm1
    invj[:, 1, 2] = (DX_DT * DY_DR - DX_DR * DY_DT) * detm1
    invj[:, 2, 2] = (DX_DR * DY_DS - DX_DS * DY_DR) * detm1
    return invj


def deformation_gradient(x, invj) -> np.ndarray:
    """``FMAT(3,3)`` of ``crframe_imp.F`` lines 136-163: the centre
    deformation gradient ``F[i,j] = sum_k (dx_i/d xi_k) INVJ[k,j]`` with the
    current corner sums divided by 8."""
    x = _as_nodes(x)
    dr = (x[:, _PLUS_R].sum(1) - x[:, _MINUS_R].sum(1)) / 8.0
    ds = (x[:, _PLUS_S].sum(1) - x[:, _MINUS_S].sum(1)) / 8.0
    dt = (x[:, _PLUS_T].sum(1) - x[:, _MINUS_T].sum(1)) / 8.0
    dxdxi = np.stack([dr, ds, dt], axis=2)                    # [n, i, k]
    return np.einsum("nik,nkj->nij", dxdxi, invj)


def _inverse_stretch(Cm) -> np.ndarray:
    """``UM`` of ``crframe_imp.F`` lines 165-221: the inverse of the right
    stretch ``U = sqrt(C)``, by the closed form (Cardano for the largest
    invariant, then a Cayley-Hamilton expansion), vectorized.

    Every statement is the Fortran's, including ``B = MAX(B, 0)`` and the
    ``A == 0`` branch; there is no guard against a singular ``C``, as there
    is none upstream.
    """
    C11, C12, C13 = Cm[:, 0, 0], Cm[:, 0, 1], Cm[:, 0, 2]
    C21, C22, C23 = Cm[:, 1, 0], Cm[:, 1, 1], Cm[:, 1, 2]
    C31, C32, C33 = Cm[:, 2, 0], Cm[:, 2, 1], Cm[:, 2, 2]
    CC = Cm @ Cm

    IC = C11 + C22 + C33
    I2C = C11 * C22 + C22 * C33 + C11 * C33 - C21 * C12 - C13 * C31 - C23 * C32
    I3C = (C11 * C22 * C33 + C12 * C23 * C31 + C13 * C21 * C32
           - (C13 * C22 * C31 + C12 * C21 * C33 + C11 * C23 * C32))

    with np.errstate(divide="ignore", invalid="ignore"):
        A = (2.0 * IC ** 3 - 9.0 * IC * I2C + 27.0 * I3C) * 32.0 / 27.0
        B = (4.0 * (I2C ** 3 + IC ** 3 * I3C) - IC ** 2 * I2C ** 2
             - 18.0 * IC * I2C * I3C + 27.0 * I3C ** 2) * 1024.0 / 27.0
        B = np.maximum(B, 0.0)
        A1 = A + np.sqrt(B)
        A2 = A - np.sqrt(B)
        ZZ = (-TWO_THIRD * IC
              + np.sign(A1) * np.abs(A1) ** THIRD
              + np.sign(A2) * np.abs(A2) ** THIRD)
        A = 2.0 * IC + ZZ
        a_zero = A == 0.0
        sqrt_a = np.sqrt(np.where(a_zero, 1.0, A))
        IU = np.where(
            a_zero,
            np.sqrt(IC + 2.0 * np.sqrt(I2C)),
            0.5 * (sqrt_a + np.sqrt(2.0 * IC - ZZ + 16.0 * np.sqrt(I3C) / sqrt_a)),
        )
        I3U = np.sqrt(I3C)
        I2U = 0.5 * (IU * IU - IC)
        A = IU * I2U - I3U
        B = I3U + IU * IC
        A1 = IU * A
        A2 = A * B
        A3 = I2U * I3U * B + IU * IU * (I2U * I2C + I3C)
        A4 = 1.0 / (I3U * I3U * B + IU * IU * (IU * I3C + I3U * I2C))

    eye = np.eye(3)[None]
    return (A4[:, None, None]
            * (A1[:, None, None] * CC - A2[:, None, None] * Cm
               + A3[:, None, None] * eye))


def crframe(sav, x) -> Tuple[np.ndarray, np.ndarray]:
    """``CRFRAME_IMP``: the co-rotational frame ``(R, INVJ)``.

    ``F = (dx/d xi) INVJ_ref`` from the current corners ``x`` and the saved
    reference ``sav``; ``R = F U^{-1}`` with ``U = sqrt(F^T F)``.
    """
    x = _as_nodes(x)
    invj = reference_inverse_jacobian(sav)
    F = deformation_gradient(x, invj)
    Cm = np.einsum("nki,nkj->nij", F, F)                       # F^T F
    R = F @ _inverse_stretch(Cm)
    return R, invj


# ---------------------------------------------------------------------------
# stage 2: crtrans_imp, transk, s8sfint3_crimp
# ---------------------------------------------------------------------------

def _shape_gradients(invj) -> np.ndarray:
    """``DNk_DX/DY/DZ`` of ``crtrans_imp.F`` lines 136-163: ``(n, 8, 3)``,
    ``g[e, k, j] = sum_m sign[k, m] INVJ[e, m, j] / 8``."""
    return np.einsum("km,nmj->nkj", _NODE_SIGNS, invj) * (1.0 / 8.0)


def corotational_transform(x, R, invj) -> Tuple[np.ndarray, np.ndarray]:
    """``CRTRANS_IMP``: ``(TRM, V)``.

    ``V[e, k]`` (``V1..V8`` upstream) maps the global displacement of node
    ``k`` to the rotation of the frame; ``TRM[e]`` (24x24) maps the global
    displacement vector to the variation of the local node positions:

    * rows of node 1 are zero except the diagonal block,
    * the block ``(j, k)`` for ``j >= 2`` is ``S_j V_k`` with
      ``S_j[a, :] = R[:, a] x (x_j - x_1)`` (``SA``..``SG``),
    * every diagonal block gets ``+R^T`` (lines 1108-1120:
      ``TRM(JJ+a, JJ+b) += R(b, a)``).

    ``R`` is an input — see the module docstring.
    """
    x = _as_nodes(x)
    n = x.shape[0]
    R = np.asarray(R, dtype=float)
    xi = x - x[:, :1, :]                                       # XI1 = 0
    g = _shape_gradients(invj)                                 # (n, 8, 3)

    # S_j[a, :] = (column a of R) x xi_j, j = 2..8      (SA11 .. SG33)
    r_cols = R.transpose(0, 2, 1)                              # [n, a, :]
    S = np.empty((n, 7, 3, 3))
    for j in range(1, 8):
        S[:, j - 1] = np.cross(r_cols, xi[:, j, None, :])

    # B11 .. B33 (lines 230-290): sum over nodes 2..8 of g_k x S_k[:, c].
    Bm = np.zeros((n, 3, 3))
    for j in range(1, 8):
        # np.cross(g, S_j[:, :, c]) for each column c -> (n, c, p)
        Bm += np.cross(g[:, j, None, :], S[:, j - 1].transpose(0, 2, 1)
                       ).transpose(0, 2, 1)
    B11, B12, B13 = Bm[:, 0, 0], Bm[:, 0, 1], Bm[:, 0, 2]
    B21, B22, B23 = Bm[:, 1, 0], Bm[:, 1, 1], Bm[:, 1, 2]
    B31, B32, B33 = Bm[:, 2, 0], Bm[:, 2, 1], Bm[:, 2, 2]
    BB = (B11 * (B22 * B33 - B23 * B32)
          - B12 * (B21 * B33 - B31 * B23)
          + B13 * (B21 * B32 - B31 * B22))
    # IF (BB /= ZERO) BB = ONE/BB   -- a zero determinant leaves BB = 0 and
    # therefore an all-zero inverse.
    with np.errstate(divide="ignore"):
        BB = np.where(BB != 0.0, 1.0 / np.where(BB != 0.0, BB, 1.0), 0.0)
    BI = np.empty((n, 3, 3))
    BI[:, 0, 0] = (B22 * B33 - B32 * B23) * BB
    BI[:, 1, 0] = (B31 * B23 - B21 * B33) * BB
    BI[:, 2, 0] = (B21 * B32 - B31 * B22) * BB
    BI[:, 0, 1] = (B13 * B32 - B12 * B33) * BB
    BI[:, 1, 1] = (B11 * B33 - B13 * B31) * BB
    BI[:, 2, 1] = (B12 * B31 - B11 * B32) * BB
    BI[:, 0, 2] = (B12 * B23 - B13 * B22) * BB
    BI[:, 1, 2] = (B21 * B13 - B11 * B23) * BB
    BI[:, 2, 2] = (B11 * B22 - B12 * B21) * BB

    # V_k = -(BI . Sv_k),  Sv_k[p, m] = (g_k x R[m, :])_p      (lines 306-465)
    V = np.empty((n, 8, 3, 3))
    for k in range(8):
        # np.cross(g, R[:, m, :]) -> (n, m, p)
        Sv = np.cross(g[:, k, None, :], R).transpose(0, 2, 1)
        V[:, k] = -np.einsum("nrp,npm->nrm", BI, Sv)

    # TRM(I, 3*j+a, 3*k+b) = sum_c S_j[a, c] V_k[c, b]   (j = 2..8)
    TRM = np.zeros((n, 24, 24))
    blocks = np.einsum("njac,nkcb->njakb", S, V)               # (n,7,3,8,3)
    TRM[:, 3:, :] = blocks.reshape(n, 21, 24)
    # TRM(I, JJ+a, JJ+b) += R(b, a)
    Rt = R.transpose(0, 2, 1)
    for j in range(8):
        TRM[:, 3 * j:3 * j + 3, 3 * j:3 * j + 3] += Rt
    return TRM, V


def transk(KL, TRM) -> np.ndarray:
    """``TRANSK``: ``K_new = TRM^T K_local TRM`` for every element.

    ``KL`` is ``(n, 24, 24)`` (Fortran ``KL(24,24,NEL)``); a new array is
    returned.  ``KTEMP = KL . TRM``, then ``KL = TRM^T . KTEMP``.
    """
    KL = np.asarray(KL, dtype=float)
    TRM = np.asarray(TRM, dtype=float)
    ktemp = np.einsum("eln,enm->elm", KL, TRM)
    return np.einsum("enl,enm->elm", TRM, ktemp)


def internal_force_to_global(f_local, R, TRM) -> Tuple[np.ndarray, np.ndarray]:
    """``S8SFINT3_CRIMP``: ``(f_global, QF)``.

    ``f_local`` is ``(n, 8, 3)`` (``F1k, F2k, F3k`` of node ``k``).  ``QF``
    is ``(n, 24)``: minus the plain rotation ``R f`` of each node force
    (``GBUF%COR_NF``, what ``S8SKSIG`` reads back), and ``f_global`` is
    ``TRM^T f_local`` — the projected force that is actually assembled.
    """
    f_local = np.asarray(f_local, dtype=float)
    n = f_local.shape[0]
    qf = -np.einsum("eij,ekj->eki", R, f_local).reshape(n, 24)
    t = np.einsum("ekj,ek->ej", TRM, f_local.reshape(n, 24))
    return t.reshape(n, 8, 3), qf


# ---------------------------------------------------------------------------
# the plan's public interface
# ---------------------------------------------------------------------------

_REFERENCES = ("undeformed", "saved")


def _table(group, model, explicit, attr, key):
    """The node-coordinate table: explicit, else ``model.<attr>``, else
    ``group.state[key]``."""
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    if model is not None and getattr(model, attr, None) is not None:
        return np.asarray(getattr(model, attr), dtype=float)
    state = getattr(group, "state", None) or {}
    if key in state:
        return np.asarray(state[key], dtype=float)
    raise ValueError(f"no {key!r} coordinates: pass them explicitly, give a "
                     f"model, or set group.state[{key!r}]")


def _corners(group, table) -> np.ndarray:
    """``(n, 8, 3)`` corner coordinates of ``group`` out of a node table (a
    table that is already ``(n, 8, 3)`` is passed through)."""
    conn = np.asarray(group.conn, dtype=np.int64)
    if conn.ndim != 2 or conn.shape[1] != 8:
        raise ValueError("the solide8s frame needs an 8-node brick group, got "
                         f"connectivity of shape {conn.shape}")
    table = np.asarray(table, dtype=float)
    return table if table.ndim == 3 else table[conn]


@dataclass
class ReferenceState:
    """What ``S8SAV3_IMP`` and ``S8XREF_IMP`` leave behind for one group."""

    sav: np.ndarray                    # (n, 21)  node-1-relative reference
    xref: Optional[np.ndarray] = None  # (n, 21)  the reference in the frame


def reference_state(group, model=None, *, x_ref=None, offg=None, sav=None,
                    frame=None) -> ReferenceState:
    """The reference state of a group: ``SAV`` (``s8sav3_imp.F``) and, when a
    ``frame`` is given, ``XREF`` (``s8xref_imp.F``).

    ``x_ref`` is the geometry being saved: at ``ISMSTR == 1`` that is the
    undeformed one (``model.x0``, the default; ``srcoor3_imp`` writes it once,
    at ``DT1 == 0``), otherwise the coordinates at the start of each step
    (``INCONV == 1``) — pass them.  ``offg`` and the previous ``sav``
    reproduce the ``ABS(OFFG) <= 1`` write guard.  ``XREF`` is that same
    geometry seen in ``frame``.  The ``SAV`` is also stored in
    ``group.state['crimp_sav']`` — the engine's ``GBUF%SMSTR`` — for
    ``imperfection_frame(..., reference='saved')``.
    """
    corners = _corners(group, _table(group, model, x_ref, "x0", "x0"))
    sav_out = save_reference(corners, offg=offg, sav=sav)
    xref = None if frame is None else local_reference(corners, frame)
    state = getattr(group, "state", None)
    if state is not None:
        state["crimp_sav"] = sav_out
    return ReferenceState(sav=sav_out, xref=xref)


def imperfection_frame(group, model=None, *, reference: str = "undeformed",
                       x=None, x_ref=None) -> np.ndarray:
    """The ``(n, 3, 3)`` co-rotational frame of every element (``CRFRAME_IMP``).

    ``reference='undeformed'`` measures ``F`` against ``model.x0`` (the
    ``ISMSTR == 1`` case, where ``SAV`` is written once at ``DT1 == 0``);
    ``reference='saved'`` measures it against ``group.state['crimp_sav']``
    (every other ``ISMSTR``: ``SAV`` is rewritten at the start of each step).
    ``x`` / ``x_ref`` override the current / reference node coordinates; with
    no ``model`` they default to ``group.state['x']`` / ``['x0']``.

    ``R[e]`` has the local axes as columns: ``x_local = R[e].T @ (x - x_1)``.
    """
    if reference not in _REFERENCES:
        raise ValueError(f"reference must be one of {_REFERENCES}, got "
                         f"{reference!r}")
    cur = _corners(group, _table(group, model, x, "x", "x"))
    if reference == "saved" and x_ref is None:
        state = getattr(group, "state", None) or {}
        if "crimp_sav" not in state:
            raise ValueError("reference='saved' needs group.state['crimp_sav'] "
                             "(see reference_state)")
        sav = np.asarray(state["crimp_sav"], dtype=float)
    else:
        sav = save_reference(_corners(group, _table(group, model, x_ref,
                                                     "x0", "x0")))
    R, _ = crframe(sav, cur)
    return R
