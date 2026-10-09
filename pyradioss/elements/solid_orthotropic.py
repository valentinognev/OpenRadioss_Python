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

"""
The ``solidez`` orthotropic solid continuum — Task P2.2.

Fortran origin: ``$OR_SRC/engine/source/elements/solid/solidez/``, the driver
``szforc3.F`` and the routines it calls.  Upstream reaches this family through
``/PROP/SOL_ORTH`` (``/PROP/TYPE6``), the orthotropic solid continuum of
``Isolid`` 11/12; the P2.0 reconciliation audit recorded every routine here
as ``missing`` because ``solid_cohesive.py`` cites ``szforc3.F`` while
implementing a cohesive element, and because ``/PROP/SOL_ORTH`` parsed in this
port but its ``isolid`` defaulted to 14, which routes to ``tshells``.

The cycle this module reproduces
--------------------------------
Upstream's ``szforc3.F`` is the ``sforc3.F`` brick cycle with three
orthotropic insertions, and reading it is what fixes the order of operations:

1. **the material frame** — ``srcoor3.F`` builds ``GBUF%GAMA``, the two
   directions of the ``/PROP/SOL_ORTH`` fiber frame in element coordinates;
   ``gettransv.F`` turns them into ``E1``/``E2``/``E3 = E1 x E2`` and then
   into the four cosine products ``QC``/``QCG``/``QGC``/``QG``;
   ``storth3.F`` fills ``G1``/``G2``/``G3`` from ``GAMA`` **only when
   ``ISORTH /= 0``** and otherwise sets the identity frame;
2. **the moduli** — ``mmodul.F`` builds the normal-normal block ``CC`` and
   the shear moduli ``G3`` in the material frame, per material type, and for
   the orthotropic types rotates them into the element frame through
   ``cbatran3v.F`` in ``mstiforthv.F`` (``CG = ½A + 2B``, ``G33 = ¼A + B``,
   ``CC = A + 4B``).  ``mmod_norm.F`` divides all three by ``GG``;
3. **the rate into the material frame** — ``sordeft3.F`` computes
   ``M'_ab = Σ_cd e_a(c) M_cd e_b(d)``, i.e. ``R M Rᵀ`` with the ``e_a`` the
   ROWS of the material frame.  Upstream hands the result to ``MMAIN`` as the
   ``MFXX..MFZZ`` block together with ``GAMA``; the law then evaluates in the
   material frame and rotates the stress back (``sroto3.F``, the same ``R·Rᵀ``).
   This port does the two rotations explicitly around
   ``materials.solid_update``, which is the same computation with the same
   order.

``forces`` on a group with no orthotropy (``ISORTH == 0``) **is** the
``sforc3`` cycle, and is dispatched to ``solid_hexa8`` for that reason — the
same reduction upstream makes at ``szforc3.F:499`` (the material-frame rate
is only formed ``IF (... OR.ISORTH/=0)``) and ``szforc3.F:933`` (the
orthotropic hourglass is only reached ``ELSEIF (ISORTH>0)``).  The
isotropic-limit test in ``tests/test_p2_solid_orthotropic.py`` asserts that
dispatch bit for bit rather than approximately.

What this module does NOT contain
---------------------------------
``szhour3_or.F`` / ``szsvm_or.F`` / ``szhour_ctl.F`` / ``gfhour_or.F`` /
``szstrainhg.F`` — the **orthotropic hourglass** law, the consumer of the
``CC``/``CG``/``G33`` moduli built here — are **not** transcribed.  The
orthotropic branch therefore runs the isotropic viscous Flanagan–Belytschko
hourglass, which is the routine upstream itself calls at ``szforc3.F:970`` for
``ISORTH == 0``.  The moduli those routines consume *are* ported, exposed
(``hourglass_moduli``) and tested; the law that reads them is named here as
absent so no reader mistakes it for ported code.

Two corrections to the plan's reading of this family, both settled by the
Fortran:

* ``scoor_cp2sp.F`` is **not** a strain-rate-to-cotangent transform.  Read,
  it is a coordinate split: it copies ``X0(I,1..8)``/``Y0``/``Z0`` (declared
  ``DOUBLE PRECISION``) into the 24 separate ``my_real`` arrays ``X1..X8``,
  ``Y1..Y8``, ``Z1..Z8``.  It is ported as exactly that.  The
  strain-rate-to-material-frame transform the plan is reaching for is
  ``sordeft3.F``.
* ``mmodul.F``'s ``ELSE`` branch is **not** the constitutive Lamé stiffness.
  It reads ``C1 = 3E/(1+ν)``, ``LAMDA = C1·ν``, ``GG = C1(1-2ν)`` and writes
  ``CC11 = LAMDA + GG = 3E(1-ν)/(1+ν)`` — a SCALED hourglass moduli, not
  ``λ + 2G`` (``= E(1+ν)(1-2ν)/(1-νν)``); the two agree only at ``ν = 0.3``.
  It also sets ``G33(J,J) = GG/2`` and ``CG = 0`` and never calls
  ``mstiforthv``.  So ``CC``/``CG``/``G33`` are kept exactly as upstream builds
  them — they feed the hourglass — and the constitutive moduli ``D`` that the
  law and the tangent read are a separate quantity.

``mmodul24c.F`` is the ``Isolid = 24`` (HEPH) variant with its own defaults
(``PM(50)``, ``PM(53..55)`` armature, reinforced ``C33STIF2EL`` assembly) and
belongs to ``solid_heph``; none of it appears in this base kernel.
"""

from __future__ import annotations

import numpy as np

from .. import materials
from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3
from . import solid_hexa8

__all__ = [
    "init_group", "forces", "dt_claim", "tangent", "kgeo",
    "stress_from_strain", "hourglass_moduli", "material_frame",
    "gettransv", "cbatran3v", "mstiforthv", "mmod_norm", "sz_dt1",
    "scoor_cp2sp", "sordeft3", "sroto3", "szordef3",
]

#: ``mmod_norm.F``'s guard value, the Fortran ``EM20``.
_EM20 = 1.0e-20

#: Voigt order used throughout, matching ``solid_hexa8`` / ``materials``:
#: ``[xx, yy, zz, xy, yz, zx]`` with **engineering** shear (gamma = 2 eps).
_VOIGT = ("xx", "yy", "zz", "xy", "yz", "zx")


# ----------------------------------------------------------------------------
# Voigt <-> tensor (the two share their slot layout)
# ----------------------------------------------------------------------------

def _to_tensor(v: np.ndarray) -> np.ndarray:
    """(n, 6) Voigt -> (n, 3, 3).  Stress Voigt (tau_xy) and engineering
    strain Voigt (gamma_xy = 2 eps_xy) share the layout, and the map is
    linear, so the same helper serves both."""
    t = np.zeros(v.shape[:-1] + (3, 3))
    t[..., 0, 0] = v[..., 0]
    t[..., 1, 1] = v[..., 1]
    t[..., 2, 2] = v[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = v[..., 3]
    t[..., 1, 2] = t[..., 2, 1] = v[..., 4]
    t[..., 0, 2] = t[..., 2, 0] = v[..., 5]
    return t


def _from_tensor(t: np.ndarray) -> np.ndarray:
    """(n, 3, 3) -> (n, 6) Voigt — the inverse of :func:`_to_tensor`."""
    v = np.empty(t.shape[:-2] + (6,))
    v[..., 0] = t[..., 0, 0]
    v[..., 1] = t[..., 1, 1]
    v[..., 2] = t[..., 2, 2]
    v[..., 3] = t[..., 0, 1]
    v[..., 4] = t[..., 1, 2]
    v[..., 5] = t[..., 0, 2]
    return v


# ----------------------------------------------------------------------------
# gettransv.F — the cosine products of the material frame
# ----------------------------------------------------------------------------

def gettransv(gama: np.ndarray):
    """``gettransv.F``: the material frame and its four cosine products.

    ``gama`` is (n, 6): ``GAMA(:,1:3)`` is the first material direction and
    ``GAMA(:,4:6)`` the second, both in element coordinates (the layout
    ``srcoor3.F`` fills ``GBUF%GAMA`` with).  Returns
    ``(qc, qcg, qgc, qg)``, each (n, 3, 3), transcribed index for index:

    * ``E1(I,J) = GAMA(I,J)``, ``E2(I,J) = GAMA(I,J+3)``, ``E3 = E1 x E2``;
    * ``QC(I,·,J)   = (E1(I,J)^2, E2(I,J)^2, E3(I,J)^2)``
    * ``QGC(I,·,J)  = (E1(I,J)E2(I,J), E2(I,J)E3(I,J), E1(I,J)E3(I,J))``
    * ``QG(I,·,J)   = (E1(J)E2(J+1)+E2(J)E1(J+1), E2(J)E3(J+1)+E3(J)E2(J+1),
      E3(J)E1(J+1)+E1(J)E3(J+1))``
    * ``QCG(I,·,J)  = (2E1(J)E1(J+1), 2E2(J)E2(J+1), 2E3(J)E3(J+1))``

    with ``J+1`` the Fortran's cyclic ``K = J+1; IF (K>3) K = 1``.

    Note what the identity frame gives: ``QC`` is **not** the identity — its
    first row is ``(1,0,0)`` and its other two rows vanish, because the Fortran
    fills ``QC(I,·,J)`` from the *component* ``J`` of each direction.
    """
    gama = np.atleast_2d(np.asarray(gama, dtype=float))
    e1 = gama[:, 0:3]
    e2 = gama[:, 3:6]
    e3 = np.cross(e1, e2)
    nxt = np.array([1, 2, 0])                       # K = J+1, 3 -> 1

    def _rows(a, b):
        """rows[a(J)b(J)] and the same with b taken at K = J+1."""
        same = np.empty_like(e1)
        nxts = np.empty_like(e1)
        for j in range(3):
            same[:, j] = a[:, j] * b[:, j]
            nxts[:, j] = a[:, j] * b[:, nxt[j]]
        return same, nxts

    qc = np.stack([_rows(e1, e1)[0], _rows(e2, e2)[0], _rows(e3, e3)[0]], axis=1)
    qgc = np.stack([_rows(e1, e2)[0], _rows(e2, e3)[0], _rows(e1, e3)[0]], axis=1)

    qg_rows, qcg_rows = [], []
    for a, b, _c in ((e1, e2, e3), (e2, e3, e1), (e3, e1, e2)):
        _ab_same, ab_next = _rows(a, b)
        _ba_same, ba_next = _rows(b, a)
        qg_rows.append(ab_next + ba_next)
        _aa_same, aa_next = _rows(a, a)
        qcg_rows.append(2.0 * aa_next)
    qg = np.stack(qg_rows, axis=1)
    qcg = np.stack(qcg_rows, axis=1)
    return qc, qcg, qgc, qg


# ----------------------------------------------------------------------------
# cbatran3v.F — the sandwich
# ----------------------------------------------------------------------------

def cbatran3v(vqi: np.ndarray, kk: np.ndarray, vqj: np.ndarray,
              isym: int = 0) -> np.ndarray:
    """``cbatran3v.F``: ``K(·,I,J) = Σ_p Σ_q VQI(p,I) KK(p,q) VQJ(q,J)``.

    That is ``VQIᵀ KK VQJ`` — the transpose is on the **left**, which is the
    whole content of the routine and the reason a "``VQI KK VQJ``" reading
    silently produces the wrong moduli.  ``isym == 1`` fills only the upper
    triangle (the Fortran's ``DO J = I, 3``); ``isym == 0`` fills all nine.
    """
    full = np.einsum("npi,npq,nqj->nij", vqi, kk, vqj)
    if isym != 1:
        return full
    # ISYM == 1: only the upper triangle is written (the Fortran's
    # ``DO J = I, 3``); the lower one keeps whatever KK carried in, which is
    # why mstiforthv mirrors from the upper triangle afterwards
    out = kk.copy()
    idx = np.triu_indices(3)
    out[:, idx[0], idx[1]] = full[:, idx[0], idx[1]]
    return out


# ----------------------------------------------------------------------------
# mstiforthv.F — the moduli composition
# ----------------------------------------------------------------------------

def _symmetrize_upper(a: np.ndarray) -> np.ndarray:
    """The Fortran's closing ``CC(I,K,J) = CC(I,J,K); G33(I,K,J) = G33(I,J,K)``
    for ``K > J``: mirror the strict upper triangle."""
    out = a.copy()
    for i in range(3):
        for k in range(i + 1, 3):
            out[:, k, i] = out[:, i, k]
    return out


def mstiforthv(cc: np.ndarray, g3: np.ndarray, qc: np.ndarray,
               qcg: np.ndarray, qgc: np.ndarray, qg: np.ndarray):
    """``mstiforthv.F``: rotate ``(CC, G3)`` from the material frame into the
    ``(CC, CG, G33)`` triple the orthotropic hourglass law consumes.

    Three ``CBATRAN3V`` sandwiches, each with ``A = CC`` and ``B = diag(G3)``:

    1. ``ISYM = 0``: ``A = QCᵀ CC QCG``, ``B = QGCᵀ G QG`` → ``CG = ½A + 2B``;
    2. ``ISYM = 1``: ``A = QCGᵀ CC QCG``, ``B = QGᵀ G QG`` → ``G33 = ¼A + B``;
    3. ``ISYM = 1`` again (the Fortran never resets it): ``A = QCᵀ CC QC``,
       ``B = QGCᵀ G QGC`` → ``CC = A + 4B``.

    Returns ``(cc, cg, g33)``, each (n, 3, 3), the new matrices mirrored from
    the upper triangle as the Fortran does before returning.
    """
    n = cc.shape[0]
    b_diag = np.zeros((n, 3, 3))
    idx = np.arange(3)
    b_diag[:, idx, idx] = g3

    a1 = cbatran3v(qc, cc, qcg, isym=0)
    b1 = cbatran3v(qgc, b_diag, qg, isym=0)
    cg = 0.5 * a1 + 2.0 * b1

    a2 = cbatran3v(qcg, cc, qcg, isym=1)
    b2 = cbatran3v(qg, b_diag, qg, isym=1)
    g33 = 0.25 * a2 + b2

    a3 = cbatran3v(qc, cc, qc, isym=1)
    b3 = cbatran3v(qgc, b_diag, qgc, isym=1)
    cc_out = _symmetrize_upper(a3 + 4.0 * b3)
    return cc_out, cg, _symmetrize_upper(g33)


def mmod_norm(cc: np.ndarray, cg: np.ndarray, g33: np.ndarray, gg: np.ndarray):
    """``mmod_norm.F``: divide ``CC``, ``G33`` and ``CG`` by ``GG``, upper
    triangle first and the lower one mirrored."""
    g = np.asarray(gg, dtype=float)[:, None, None]
    cc_o, g33_o, cg_o = cc / g, g33 / g, cg / g
    return _symmetrize_upper(cc_o), cg_o, _symmetrize_upper(g33_o)


# ----------------------------------------------------------------------------
# sz_dt1.F90 — the hourglass characteristic length
# ----------------------------------------------------------------------------

def sz_dt1(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
           gfac: float) -> np.ndarray:
    """``sz_dt1.F90``: the orthotropic hourglass length ``DELTAX1``.

    With ``gfac = G/BULK = (1-2nu)/(1-nu)`` below one,

        pxx = 2 Σ_i px_i²   (and pyy, pzz; pxy, pxz, pyz likewise)
        aa  = -(pxx + pyy + pzz)
        bb  = gfac (pxx·pyy + pxx·pzz + pyy·pzz - pxy² - pxz² - pyz²)
        p   = bb - aa²/3
        d   = 4 sqrt(max(-p, 0)/3) - 2aa/3          (mas_j = mas_e/8)
        DELTAX1 = 1/sqrt(d)

    and for ``gfac >= 1`` it is exactly zero, as upstream.  This is a *length*,
    not a time step — ``solid_cohesive.py`` / ``solid_connect.py`` cite this
    file for an eigenvalue dt bound, which is a different quantity.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    pz = np.asarray(pz, dtype=float)
    if gfac >= 1.0:
        return np.zeros(px.shape[0])
    pxx = 2.0 * np.sum(px * px, axis=-1)
    pyy = 2.0 * np.sum(py * py, axis=-1)
    pzz = 2.0 * np.sum(pz * pz, axis=-1)
    pxy = 2.0 * np.sum(px * py, axis=-1)
    pxz = 2.0 * np.sum(px * pz, axis=-1)
    pyz = 2.0 * np.sum(py * pz, axis=-1)
    aa = -(pxx + pyy + pzz)
    bb = gfac * (pxx * pyy + pxx * pzz + pyy * pzz - pxy ** 2 - pxz ** 2 - pyz ** 2)
    p = bb - aa * aa / 3.0
    d = 4.0 * np.sqrt(np.maximum(-p, 0.0) / 3.0) - 2.0 * aa / 3.0
    return 1.0 / np.sqrt(np.maximum(d, EM20))


# ----------------------------------------------------------------------------
# scoor_cp2sp.F — the coordinate split
# ----------------------------------------------------------------------------

def scoor_cp2sp(x0: np.ndarray, y0: np.ndarray, z0: np.ndarray):
    """``scoor_cp2sp.F``: split the (nel, 8) coordinate arrays into the 24
    single-node arrays the cycle passes around.

    Upstream reads the gather into ``DOUBLE PRECISION X0(MVSIZ,8)`` (and Y0,
    Z0) and copies ``X0(I,1..8)`` into ``X1(I)..X8(I)``, ``Y0`` into
    ``Y1..Y8`` and ``Z0`` into ``Z1..Z8`` — a copy loop, with no rotation and
    no strain rate.  Returns the 24 arrays in the Fortran's own order.
    """
    cols = []
    for block in (x0, y0, z0):
        block = np.asarray(block)
        for j in range(block.shape[1]):
            cols.append(block[:, j])
    return cols


# ----------------------------------------------------------------------------
# The material-frame rotations: sordeft3.F / sroto3.F / szordef3.F
# ----------------------------------------------------------------------------

def _frame_rows(gama: np.ndarray) -> np.ndarray:
    """``storth3.F``: ``G1 = GAMA(:,1:3)``, ``G2 = GAMA(:,4:6)``,
    ``G3 = G1 x G2``, stacked as the ROWS of the (n, 3, 3) frame.  (The
    identity frame is what ``storth3.F`` fills for ``ISORTH == 0``, which is
    the ``GAMA`` :func:`material_frame` hands back in that case.)"""
    gama = np.atleast_2d(np.asarray(gama, dtype=float))
    e1 = gama[:, 0:3]
    e2 = gama[:, 3:6]
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1)


def sordeft3(m: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``sordeft3.F``: the 3x3 rate rotated **into** the material frame,
    ``M'_ab = Σ_cd e_a(c) M_cd e_b(d)`` — ``R M Rᵀ`` with the ``e_a`` the ROWS
    of the frame.  ``g`` is (n, 6) in the ``GAMA`` layout; returns (n, 3, 3).
    The input is a true tensor (the Fortran works on the nine ``MFXX..MFZZ``
    components), not a Voigt vector — use :func:`sroto3` for the stress.
    """
    r = _frame_rows(g)
    return np.einsum("...ij,...jk,...kl->...il", r, m, r.swapaxes(-1, -2),
                     optimize=True)


def sroto3(sig: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``sroto3.F``: the stress carried OUT of the material frame and back to
    the element frame — the INVERSE of :func:`sordeft3`.

Upstream writes the same ``R σ Rᵀ`` expression in both routines because
    it never needs the inverse there: ``szforc3.F`` hands ``MFXX..MFZZ`` and
    ``GAMA`` to ``MMAIN``, the law rotates its own stress back before
    ``GBUF%SIG`` is used, and ``SROTO3``'s ``SIGN`` output feeds the hourglass
    evaluation only.  This port does the two rotations itself, so it does need
    the inverse, and the inverse of ``M ↦ R M Rᵀ`` is ``σ ↦ Rᵀ σ R``.

    Using ``R σ Rᵀ`` here instead is the single most expensive mistake
    available in this family: it is self-consistent rather than loud, wrong for
    every anisotropic material, and **invisible for an isotropic one** — which
    is why the suite compares an isotropic material in two different frames
    rather than trusting the isotropic-limit equality.
    """
    r = _frame_rows(g)
    s3 = _to_tensor(np.asarray(sig, dtype=float))
    return _from_tensor(np.einsum("...ij,...jk,...kl->...il",
                                 r.swapaxes(-1, -2), s3, r, optimize=True))


def szordef3(defn: np.ndarray, g: np.ndarray) -> np.ndarray:
    """``szordef3.F``: the strain increment taken **out** of the material frame
    and back into the element frame — the same operator as :func:`sroto3`,
    named separately because ``szforc3.F`` calls it at a different point
    (line 659, guarded by ``JCVT == 2``)."""
    return sroto3(defn, g)


# ----------------------------------------------------------------------------
# The material frame of a /PROP/SOL_ORTH group
# ----------------------------------------------------------------------------

_IDENTITY_GAMA = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def material_frame(prop, n: int = 1) -> np.ndarray:
    """The (n, 6) ``GAMA`` of a ``/PROP/SOL_ORTH`` property.

    ``storth3.F`` reads ``GBUF%GAMA`` as two directions; the starter builds it
    from the property's fiber vector ``(vx, vy, vz)`` rotated by the in-plane
    angle ``phi`` about it (``PROP_SOL_ORTH_VEC`` / ``PROP_SOL_ORTH_ANG``).
    ``iorth == 0`` — upstream's ``ISORTH`` — carries no fiber frame, and
    ``storth3.F`` then sets the identity.
    """
    if getattr(prop, "orthtrop", 0) in (0, None):
        return np.tile(_IDENTITY_GAMA, (n, 1))
    v = np.array([float(getattr(prop, k, 0.0)) for k in ("vx", "vy", "vz")])
    nv = float(np.linalg.norm(v))
    if nv <= EM20:                                  # no fiber vector given
        return np.tile(_IDENTITY_GAMA, (n, 1))
    e1 = v / nv
    w = np.eye(3)[int(np.argmin(np.abs(e1)))]         # the axis least aligned
    t = np.cross(e1, w)
    t /= max(float(np.linalg.norm(t)), EM20)
    phi = np.deg2rad(float(getattr(prop, "phi", 0.0)))
    e2 = np.cos(phi) * w + np.sin(phi) * t
    return np.tile(np.concatenate([e1, e2]), (n, 1))


# ----------------------------------------------------------------------------
# mmodul.F — the material moduli
# ----------------------------------------------------------------------------

def _param(mat, *names, default=0.0):
    """First present value among ``names`` on the material object or in its
    ``params`` dict.  The port stores card values by name (``MAT_E11`` /
    ``E11`` / ``e11`` …) rather than in upstream's ``PM``/``UPARAM`` slots, so
    the branch table below names the slots it came from and this accessor
    does the lookup."""
    for name in names:
        val = getattr(mat, name, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    params = getattr(mat, "params", None)
    if isinstance(params, dict):
        for name in names:
            if name in params:
                try:
                    return float(params[name])
                except (TypeError, ValueError):
                    pass
    return float(default)


#: ``mmodul.F`` MTN values whose ``CC`` the material supplies directly, and
#: whether the Fortran leaves the normal coupling at zero for it.  MTN is the
#: material law number, which is how upstream dispatches.
DIRECT_BRANCHES = {
    12: False, 14: False,          # PM(40..45) — the nine moduli, coupling kept
    28: True, 68: True, 50: True,  # UPARAM(1..6) — diagonal, coupling ZERO
    53: True, 130: True,           # UPARAM(1..4) / (4..9) — diagonal
    93: False, 122: False, 123: False, 125: False, 127: False,
}

#: ``mmodul.F`` MTN values whose ``CC`` comes from the Poisson form.
NU_BRANCHES = (25, 107, 112)


def _direct_constants(mat) -> tuple[float, ...]:
    """(C11, C22, C33, C12, C13, C23, G12, G23, G31) from the material card."""
    c11 = _param(mat, "MAT_M11", "m11", "C11")
    c22 = _param(mat, "MAT_M22", "m22", "C22")
    c33 = _param(mat, "MAT_M33", "m33", "C33")
    c12 = _param(mat, "MAT_M12", "m12", "C12")
    c13 = _param(mat, "MAT_M13", "m13", "C13")
    c23 = _param(mat, "MAT_M23", "m23", "C23")
    g12 = _param(mat, "MAT_G12", "G12", "g12", "gab")
    g23 = _param(mat, "MAT_G23", "G23", "g23", "gbc")
    g31 = _param(mat, "MAT_G31", "G31", "g31", "gca")
    if c11 == 0.0 and c22 == 0.0 and c33 == 0.0:
        # a card that carries E / nu / G instead of the stiffnesses: build the
        # block from the compliance, the same place the port's orthotropic
        # elastic laws build it.
        c11, c22, c33, c12, c13, c23 = _from_compliance(mat)
    return c11, c22, c33, c12, c13, c23, g12, g23, g31


def _from_compliance(mat):
    """The 3x3 normal block of an orthotropic stiffness from
    ``(E1, E2, E3, nu12, nu21, nu13, nu31, nu23, nu32)`` — the reciprocal
    Poisson relations ``nu_ji/E_j = nu_ij/E_i`` filled in as the material
    laws do."""
    e1 = _param(mat, "MAT_E11", "E11", "e11")
    e2 = _param(mat, "MAT_E22", "E22", "e22")
    e3 = _param(mat, "MAT_E33", "E33", "e33")
    if min(e1, e2, e3) <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nu12 = _param(mat, "MAT_PRAB", "nu12")
    nu23 = _param(mat, "MAT_PRBC", "nu23")
    nu31 = _param(mat, "MAT_PRCA", "nu31")
    nu21 = _param(mat, "MAT_PRBA", "nu21", default=nu12 * e2 / e1)
    nu32 = _param(mat, "MAT_PRCB", "nu32", default=nu23 * e3 / e2)
    nu13 = _param(mat, "MAT_PRAC", "nu13", default=nu31 * e1 / e3)
    s = np.array([
        [1.0 / e1, -nu21 / e2, -nu31 / e3],
        [-nu12 / e1, 1.0 / e2, -nu32 / e3],
        [-nu13 / e1, -nu23 / e2, 1.0 / e3],
    ])
    try:
        c = np.linalg.inv(s)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (float(c[0, 0]), float(c[1, 1]), float(c[2, 2]),
            float(c[0, 1]), float(c[0, 2]), float(c[1, 2]))


def _nu_branch(mat) -> tuple[float, ...]:
    """``mmodul.F`` MTN = 25 / 107 / 112 — the Poisson construction, verbatim,
    including its singular guard:

        S1   = 1 - nu12 nu21
        CC11 = E1 / max(EM20, S1)
        CC22 = E2 / max(EM20, S1)
        CC33 = E3
        CC12 = ½ (nu21 CC11 + nu12 CC22)
        CC13 = CC23 = 0
        G3   = (G12, G23, G31)
    """
    e1 = _param(mat, "MAT_E11", "E11", "e11")
    e2 = _param(mat, "MAT_E22", "E22", "e22")
    e3 = _param(mat, "MAT_E33", "E33", "e33")
    nu12 = _param(mat, "MAT_PRAB", "nu12")
    nu21 = _param(mat, "MAT_PRBA", "nu21", default=nu12 * e2 / max(e1, EM20))
    s1 = 1.0 - nu12 * nu21
    guard = max(EM20, s1)
    c11 = e1 / guard
    c22 = e2 / guard
    c12 = 0.5 * (nu21 * c11 + nu12 * c22)
    g12 = _param(mat, "MAT_G12", "G12", "g12", "gab")
    g23 = _param(mat, "MAT_G23", "G23", "g23", "gbc")
    g31 = _param(mat, "MAT_G31", "G31", "g31", "gca")
    return c11, c22, e3, c12, 0.0, 0.0, g12, g23, g31


def _isotropic_hourglass_moduli(mat) -> tuple[float, ...]:
    """``mmodul.F`` MTN = 128 and its ``ELSE`` branch.

    MTN=128 takes ``LAMDA = 3 nu K/(1+nu)`` and ``CC11 = LAMDA + 2G`` from the
    material's ``BULK``/``SHEAR``; the ``ELSE`` branch takes Young's modulus
    and Poisson's ratio off ``PM(32)``/``PM(21)``:

        C1 = 3E/(1+nu),  LAMDA = C1 nu,  GG = C1 (1-2nu)
        CC11 = CC22 = CC33 = LAMDA + GG,  CC12 = CC13 = CC23 = LAMDA
        G33(J,J) = GG/2,  CG = 0

    and fills them **without** calling ``MSTIFORTHV``.

    WHAT THESE NUMBERS ARE NOT, because it is the single easiest wrong turn
    in this family: ``LAMDA + GG`` is **not** ``lam + 2G``.  For nu = 0.3 the
    Fortran's ``CC11`` is ``3(1-2nu) = 1.2`` times the physical
    ``lam + 2G``, and ``G33(J,J)`` is ``3(1-2nu)`` times ``G``.  They are the
    HOURGLASS moduli: ``szsvm_or.F`` and ``szsvm.F`` consume them only through
    differences and ratios (``CC(I,J,1) - CC(I,K,1)``, and everything after
    ``MMOD_NORM`` divides by ``GG``), where a common scale cancels.  Reading
    them as a constitutive tensor gives a stiffness that is wrong by 20% for a
    perfectly ordinary nu = 0.3 solid — so :func:`_ortho_stiffness` takes the
    constitutive moduli from the material law instead, and this function feeds
    only :func:`hourglass_moduli`.
    """
    bulk = _param(mat, "MAT_BULK", "bulk", "K")
    shear = _param(mat, "MAT_SHEAR", "shear", "G")
    nu = _param(mat, "MAT_NU", "nu")
    if bulk > 0.0 and shear > 0.0:
        lamda = 3.0 * nu * bulk / (1.0 + nu)
        c11 = lamda + 2.0 * shear
        return c11, c11, c11, lamda, lamda, lamda, shear, shear, shear
    E = _param(mat, "MAT_EN", "E", "E11", "e11")
    if E <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    c1 = 3.0 * E / (1.0 + nu)
    lamda = c1 * nu
    gg = c1 * (1.0 - 2.0 * nu)
    return lamda + gg, lamda + gg, lamda + gg, lamda, lamda, lamda, gg / 2.0, gg / 2.0, gg / 2.0


def mmodul(mat) -> tuple[tuple[float, ...], bool]:
    """``mmodul.F`` for one material: the nine orthotropic constants in the
    MATERIAL frame and whether the block is rotated into the element frame.

    Returns ``(constants, rotate)``.  The Fortran rotates for every orthotropic
    branch (``MTN`` 12/14/25/28/50/53/68/93/107/112/122/123/125/127/130) and
    does **not** for ``MTN = 24``, ``MTN = 128`` or the ``ELSE`` branch, where
    it fills ``CC``/``CG``/``G33`` in place.
    """
    mtn = int(getattr(mat, "law", 1) or 1)
    if mtn in DIRECT_BRANCHES:
        c = _direct_constants(mat)
        if DIRECT_BRANCHES[mtn]:
            c = (c[0], c[1], c[2], 0.0, 0.0, 0.0, c[6], c[7], c[8])
        return c, True
    if mtn in NU_BRANCHES:
        return _nu_branch(mat), True
    return _isotropic_hourglass_moduli(mat), False


def _ortho_stiffness(mat) -> tuple[float, ...]:
    """The nine CONSTITUTIVE constants ``(C11, C22, C33, C12, C13, C23,
    G12, G23, G31)`` in the material frame — what the stress law actually
    integrates.

    This is deliberately NOT :func:`_isotropic_hourglass_moduli`.  Where a
    material card carries its own orthotropic constants (``mmodul.F``'s
    ``MTN`` 12/14/25/28/50/53/68/93/107/112/122/123/125/127/130 branches —
    the ones the port reaches through ``MAT_M11``-style and
    ``MAT_E11``/``MAT_PRAB``/``MAT_GAB``-style parameter names) the same
    numbers are both the hourglass moduli and the constitutive ones, and they
    are used unchanged.  For an ISOTROPIC card the hourglass moduli are the
    ``3(1-2nu)``-scaled quantities ``mmodul.F`` builds, so the constitutive
    moduli are taken from the material law instead — from ``K`` and ``G``, the
    same ``materials.solid_tangent`` the isotropic brick kernel integrates,
    which is what makes the isotropic limit exact rather than 20% off.
    """
    mtn = int(getattr(mat, "law", 1) or 1)
    if mtn in DIRECT_BRANCHES or mtn in NU_BRANCHES:
        return _direct_constants(mat) if mtn in DIRECT_BRANCHES else _nu_branch(mat)
    lamda, shear = _lame_from_material(mat)
    if lamda == 0.0 and shear == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    c11 = lamda + 2.0 * shear
    return c11, c11, c11, lamda, lamda, lamda, shear, shear, shear


def _lame_from_material(mat) -> tuple[float, float]:
    """``(lam, G)`` of an isotropic material card, from whatever the law
    exposes: ``mat.K``/``mat.G`` (the fields every isotropic solid law in this
    port carries), else ``E`` and ``nu``."""
    K = getattr(mat, "K", None)
    G = getattr(mat, "G", None)
    if K is not None and G is not None and float(K) > 0.0 and float(G) > 0.0:
        return float(K) - 2.0 * float(G) / 3.0, float(G)
    E = _param(mat, "MAT_EN", "E")
    nu = _param(mat, "MAT_NU", "nu")
    if E <= 0.0 or 1.0 + nu <= 0.0:
        return 0.0, 0.0
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def hourglass_moduli(group) -> dict:
    """The per-element ``(CC, CG, G33)`` triple ``szhour3_or.F`` consumes,
    built by :func:`mmodul` + :func:`gettransv` + :func:`mstiforthv`.

    Exposed and tested here; the routine that *reads* it
    (``szsvm_or.F`` / ``szhour3_or.F``) is **not** part of this module — see
    the module docstring.
    """
    st = group.state
    n = group.n
    cc = np.zeros((n, 3, 3))
    cg = np.zeros((n, 3, 3))
    g33 = np.zeros((n, 3, 3))
    g3 = np.zeros((n, 3))
    for sl, mat, _prop in st.get("slices", []):
        consts, rotate = mmodul(mat)
        blk = np.zeros((sl.stop - sl.start, 3, 3))
        blk[:, 0, 0], blk[:, 1, 1], blk[:, 2, 2] = consts[0], consts[1], consts[2]
        blk[:, 0, 1] = blk[:, 1, 0] = consts[3]
        blk[:, 0, 2] = blk[:, 2, 0] = consts[4]
        blk[:, 1, 2] = blk[:, 2, 1] = consts[5]
        shear = np.tile(np.array(consts[6:9]), (sl.stop - sl.start, 1))
        gama = st["gama"][sl]
        if rotate:
            qc, qcg, qgc, qg = gettransv(gama)
            blk, cg_o, g33_o = mstiforthv(blk, shear, qc, qcg, qgc, qg)
        else:
            # mmodul.F's isotropic fill: CG = 0 and G33(J,J) = G, no rotation
            cg_o = np.zeros_like(blk)
            g33_o = np.zeros_like(blk)
            d = np.arange(3)
            g33_o[:, d, d] = shear
        cc[sl] = blk
        cg[sl] = cg_o
        g33[sl] = g33_o
        g3[sl] = shear
    return {"cc": cc, "cg": cg, "g33": g33, "g3": g3}


# ----------------------------------------------------------------------------
# The element-frame Voigt moduli (the constitutive law and the tangent)
# ----------------------------------------------------------------------------

def _voigt_from_constants(consts) -> np.ndarray:
    """The material-frame (6, 6) orthotropic stiffness in the ENGINEERING
    Voigt convention the element kernels use: ``sigma_v = C6 gamma_v`` with
    ``gamma = 2 eps``, so the shear diagonal carries ``G`` exactly as
    ``law01_elastic.solid_tangent`` writes it and as
    ``law01_elastic.solid_update`` integrates it
    (``sig[:, 3:] += G * deps[:, 3:]``).

    ``consts`` is the nine ``(C11, C22, C33, C12, C13, C23, G12, G23, G31)``.
    """
    c11, c22, c33, c12, c13, c23, g12, g23, g31 = consts
    c = np.zeros((6, 6))
    c[0, 0], c[1, 1], c[2, 2] = c11, c22, c33
    c[0, 1] = c[1, 0] = c12
    c[0, 2] = c[2, 0] = c13
    c[1, 2] = c[2, 1] = c23
    c[3, 3], c[4, 4], c[5, 5] = g12, g23, g31
    return c


#: Voigt shear slot -> the tensor pair it stands for, ``[xx, yy, zz, xy, yz, zx]``.
_SLOT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

#: The six unit strain DIRECTIONS as tensor strains: ``eps_aa = 1`` on the
#: normal slots and ``eps_ab = eps_ba = 1/2`` on the shear slots, which is what
#: the ENGINEERING Voigt unit direction means (``gamma_ab = 2 eps_ab = 1``).
_UNIT_STRAIN = np.zeros((6, 3, 3))
for _i, (_a, _b) in enumerate(_SLOT):
    if _a == _b:
        _UNIT_STRAIN[_i, _a, _b] = 1.0
    else:
        _UNIT_STRAIN[_i, _a, _b] = _UNIT_STRAIN[_i, _b, _a] = 0.5


def _tensor_to_voigt(t: np.ndarray) -> np.ndarray:
    """(..., 3, 3) tensor -> (..., 6) ENGINEERING Voigt: the shear slots are
DOUBLED, because ``gamma = 2 eps``."""
    v = np.empty(t.shape[:-2] + (6,))
    v[..., 0] = t[..., 0, 0]
    v[..., 1] = t[..., 1, 1]
    v[..., 2] = t[..., 2, 2]
    v[..., 3] = 2.0 * t[..., 0, 1]
    v[..., 4] = 2.0 * t[..., 1, 2]
    v[..., 5] = 2.0 * t[..., 0, 2]
    return v


def _voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """(..., 6) ENGINEERING Voigt -> (..., 3, 3) tensor strain: the shear
    slots are HALVED.  This and :func:`_tensor_to_voigt` are the only two
    places the engineering-shear factor appears."""
    t = np.zeros(v.shape[:-1] + (3, 3))
    t[..., 0, 0] = v[..., 0]
    t[..., 1, 1] = v[..., 1]
    t[..., 2, 2] = v[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = 0.5 * v[..., 3]
    t[..., 1, 2] = t[..., 2, 1] = 0.5 * v[..., 4]
    t[..., 0, 2] = t[..., 2, 0] = 0.5 * v[..., 5]
    return t


def _frame_as_gama(r: np.ndarray) -> np.ndarray:
    """(n, 3, 3) frame rows -> the (n, 6) ``GAMA`` layout :func:`sordeft3`
    takes, so the tensor rotation is written once and reused here."""
    return np.concatenate([r[:, 0, :], r[:, 1, :]], axis=1)


def _rot3(r: np.ndarray, m: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """``R M Rᵀ`` for a batch of 3x3 tensors — the tensor rule of
    ``sordeft3.F`` / ``sroto3.F``, with ``R`` the rows of the material frame
    and ``rt`` the frame's other side (the transpose for ``R M Rᵀ``)."""
    return np.einsum("iac,icd,ibd->iab", r, m, rt)


def _voigt_to_c4(c6: np.ndarray) -> np.ndarray:
    """(n, 6, 6) Voigt moduli -> (n, 3, 3, 3, 3) elasticity tensor.

    ``c6`` says ``sigma_v[i] = Σ_j c6[i, j] gamma_v[j]``.  The shear Voigt
    direction is ``gamma_ab = 2 eps_ab`` over ONE independent component, so a
    unit ``gamma_ab`` is a tensor strain ``eps_ab = 1/2`` and
    ``d sigma_ab / d eps_ab = 2 c6[i, j]`` — the factor of 2 lives on the shear
    strain direction, symmetrically across both reflected positions because
    ``eps_ab`` and ``eps_ba`` are the same independent component.
    """
    n = c6.shape[0]
    c4 = np.zeros((n, 3, 3, 3, 3))
    for i, (a, b) in enumerate(_SLOT):
        for j, (c, d) in enumerate(_SLOT):
            val = c6[:, i, j] / (1.0 if c == d else 2.0)
            c4[:, a, b, c, d] = val
            c4[:, b, a, c, d] = val
            c4[:, a, b, d, c] = val
            c4[:, b, a, d, c] = val
    return c4


def _c4_to_voigt(c4: np.ndarray) -> np.ndarray:
    """The inverse of :func:`_voigt_to_c4`."""
    n = c4.shape[0]
    c6 = np.zeros((n, 6, 6))
    for i, (a, b) in enumerate(_SLOT):
        for j, (c, d) in enumerate(_SLOT):
            c6[:, i, j] = c4[:, a, b, c, d] * (1.0 if c == d else 2.0)
    return c6


def _rotate_moduli(c6: np.ndarray, gama: np.ndarray) -> np.ndarray:
    """``c6`` (n, 6, 6) in the material frame -> the element frame.

    Rotated through the **4th-order elasticity tensor**, which is the only
    place the Voigt shear convention can be got right once: the moduli are a
    law on a TENSOR strain, the Voigt form is just a reading of it, and the two
    differ by the engineering-shear factor of 2 on the shear strain direction.

    Rotating the Voigt matrix directly — ``K C Kᵀ`` with a Kelvin rotation, or
    any similarity built from one — is wrong for a reason that is very hard to
    see: the Kelvin operators for strain and for stress are *not* inverses or
    transposes of each other (they differ by the shear factor), so no ordering
    of them turns a Voigt shear of ``G`` into the rotated answer.  Every wrong
    variant still passes the two obvious checks — the result is symmetric, and
    an axis-permuting frame still returns a diagonal — so the invariance
    asserted here is the load-bearing one:

    * an **isotropic** moduli matrix comes back unchanged under ANY frame, to
      round-off (a rotated isotropic material is the same material);
    * the result is **symmetric**, because an elasticity tensor is;
    * a frame that permutes the axes maps each modulus onto the axis its own
      material direction points at.
    """
    r = _frame_rows(gama)                            # (n, 3, 3)
    c4 = _voigt_to_c4(c6)
    # The frame's ROWS are the material directions expressed in element
    # coordinates (storth3.F), so the element direction x_j is material
    # component Σ_a R_ja eps_a, i.e. the tensor rotation here runs on the
    # frame's TRANSPOSE — the same operator sordeft3.F applies before it hands
    # MMAIN the rate.  With the rows instead of the transpose the axis mapping
    # silently comes out inverted on a permuted frame.
    rt = r.swapaxes(-1, -2)
    c4p = np.einsum("nap,nbq,ncr,nds,npqrs->nabcd", rt, rt, rt, rt, c4,
                    optimize=True)
    return _c4_to_voigt(c4p)


def _build_D(group) -> np.ndarray:
    """The (n, 6, 6) element-frame Voigt moduli, one block per slice."""
    st = group.state
    n = group.n
    D = np.zeros((n, 6, 6))
    for sl, mat, _prop in st.get("slices", []):
        c6 = _voigt_from_constants(_ortho_stiffness(mat))
        c6 = np.tile(c6, (sl.stop - sl.start, 1, 1))
        D[sl] = _rotate_moduli(c6, st["gama"][sl])
    return D


def stress_from_strain(group, eps: np.ndarray) -> np.ndarray:
    """``σ = D : ε`` with ``D`` the element-frame Voigt moduli.

    The constitutive evaluation the plan's test names.  ``eps`` and the result
    are (n, 6) in the element frame, engineering shear.  With a decoupled
    orthotropic card (``mmodul.F`` MTN=28/68 sets ``CC(1,2) = CC(1,3) =
    CC(2,3) = ZERO``) the strain state ``(1e-3, 0, 0, 0, 0, 0)`` returns
    ``σ_11 = C11·1e-3`` and zero lateral stress.
    """
    eps = np.atleast_2d(np.asarray(eps, dtype=float))
    return np.einsum("nij,nj->ni", group.state["D"], eps)


# ----------------------------------------------------------------------------
# The cycle entry points
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Element buffer and nodal mass for a ``solidez`` group.

    The geometry and lumping are ``srcoor3.F`` / ``smass3.F`` — the same
    one-point brick ``solid_hexa8.init_group`` builds — so that is where this
    goes through; what is added here is the orthotropic state the cycle needs:
    the ``GAMA`` material frame (``storth3.F``), the ``(CC, CG, G33)`` moduli of
    ``mmodul.F``, and the element-frame Voigt moduli ``D``.
    """
    out = solid_hexa8.init_group(group, model, log)
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state["gama"] = np.empty((0, 6))
        group.state["cc"] = np.empty((0, 3, 3))
        group.state["cg"] = np.empty((0, 3, 3))
        group.state["g33"] = np.empty((0, 3, 3))
        group.state["D"] = np.empty((0, 6, 6))
        return out

    slices = group.state.get("slices", [])
    gama = np.empty((n, 6))
    for sl, _mat, prop in slices:
        gama[sl] = material_frame(prop, sl.stop - sl.start)[0]
    group.state["gama"] = gama

    moduli = hourglass_moduli(group)
    group.state["cc"] = moduli["cc"]
    group.state["cg"] = moduli["cg"]
    group.state["g33"] = moduli["g33"]
    group.state["D"] = _build_D(group)

    # the Courant scalar sz_dt1.F90 is gated on (upstream calls it with
    # FAC_NU = (1-2nu)/(1-nu), the ratio the routine's comment names)
    gfac = np.zeros(n)
    for sl, mat, _prop in slices:
        nu = float(getattr(mat, "nu", 0.0) or 0.0)
        gfac[sl] = (1.0 - 2.0 * nu) / (1.0 - nu) if nu < 1.0 else 1.0
    group.state["gfac"] = gfac
    return out


def _is_orthotropic(group) -> bool:
    """``ISORTH > 0`` on any slice — the flag upstream reads as ``IORTH`` on
    ``/PROP/SOL_ORTH`` and tests at ``szforc3.F:499`` and ``:933``."""
    for _sl, _mat, prop in group.state.get("slices", []):
        if int(getattr(prop, "orthtrop", 0) or 0) > 0:
            return True
    return False


def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the whole group; returns the per-element
    critical time step.

    With no orthotropy this is ``solid_hexa8.forces`` — the ``szforc3.F`` /
    ``sforc3.F`` reduction described in the module docstring.  With
    orthotropy the rate is rotated into the material frame (``sordeft3.F``)
    before the law and the stress back out (``sroto3.F``) after it, the way
    upstream hands ``MFXX..MFZZ`` and ``GAMA`` to ``MMAIN``; the force
    assembly is ``sfint3.F`` and the viscous hourglass is ``szhour3.F``.
    """
    if not _is_orthotropic(group):
        return solid_hexa8.forces(group, x, v, vr, dt, fint, mint)

    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0)

    xe = x[conn]
    detJ = solid_hexa8._detJ(xe)
    if dt is None or dt <= 0.0 or v is None:
        return dt_claim(group, x)

    dndx, vol = solid_hexa8._geometry(xe)
    vol = np.maximum(vol, EM20)
    lc = solid_hexa8._char_length(xe, vol) * st.get("lc_scale", np.ones(n))
    ve = v[conn]
    alive = st["off"] > 0.0

    L = ve.transpose(0, 2, 1) @ dndx                       # sdefo3.F
    trD = L[:, 0, 0] + L[:, 1, 1] + L[:, 2, 2]
    vgm = np.abs(ve).max(axis=(1, 2)) * np.abs(dndx).max(axis=(1, 2))
    trD = np.where(np.abs(trD) <= 1e-14 * vgm, 0.0, trD)
    deps = np.empty((n, 6))
    deps[:, 0] = L[:, 0, 0] * dt
    deps[:, 1] = L[:, 1, 1] * dt
    deps[:, 2] = L[:, 2, 2] * dt
    deps[:, 3] = (L[:, 0, 1] + L[:, 1, 0]) * dt
    deps[:, 4] = (L[:, 1, 2] + L[:, 2, 1]) * dt
    deps[:, 5] = (L[:, 0, 2] + L[:, 2, 0]) * dt
    deps[~alive] = 0.0
    trD = np.where(alive, trD, 0.0)

    sig = st["sig"]
    sig_old = sig.copy()
    # Jaumann rotation of the old stress (srrota3.F), as solid_hexa8 does it
    wxy = 0.5 * (L[:, 0, 1] - L[:, 1, 0]) * dt
    wyz = 0.5 * (L[:, 1, 2] - L[:, 2, 1]) * dt
    wxz = 0.5 * (L[:, 0, 2] - L[:, 2, 0]) * dt
    sxx, syy, szz = sig[:, 0].copy(), sig[:, 1].copy(), sig[:, 2].copy()
    sxy, syz, szx = sig[:, 3].copy(), sig[:, 4].copy(), sig[:, 5].copy()
    sig[:, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    # The rate INTO the material frame.  ``deps`` is an ENGINEERING Voigt rate
    # (``gamma = 2 eps``), so it has to become a strain TENSOR first —
    # ``_voigt_to_tensor`` halves the shear — and only then is rotated by
    # sordeft3's R M R^T.  Passing the Voigt array through the stress-shaped
    # ``sroto3`` gets both the shear scale and the rotation direction wrong,
    # and is invisible for an isotropic material.
    deps_mat = _tensor_to_voigt(
        sordeft3(_voigt_to_tensor(deps), st["gama"]))

    # ---- material law, in the material frame ------------------------------
    rho = st["mass"] / vol
    c = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)
    hcoef = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        # the ELEMENT stress carried into the material frame: tau IS sigma_ab,
        # so it goes through the plain _to_tensor layout, and it comes back out
        # of the law by the INVERSE rotation (see sroto3's note).
        sig_mat = sroto3(sig[sl], st["gama"][sl])
        extra = {"off": st["off"][sl], "rho": rho[sl], "vol": vol[sl],
                 "vol0": st["vol0"][sl], "eint": st["eint"][sl]}
        sig_mat, _e, c_new = materials.solid_update(mat, sig_mat, deps_mat[sl],
                                                    st["epsp"][sl], dt, extra)
        sig[sl] = sroto3(sig_mat, st["gama"][sl])
        if c_new is not None:
            c[sl] = c_new
        else:
            D = st["D"][sl]
            c[sl] = np.sqrt(np.maximum(D[:, 0, 0].max(axis=1), 0.0)
                            / np.maximum(rho[sl], EM20))
        qa[sl] = getattr(prop, "params", {}).get("qa", 1.1)
        qb[sl] = getattr(prop, "params", {}).get("qb", 0.05)
        hcoef[sl] = getattr(prop, "params", {}).get("h", 0.1)

    alive = st["off"] > 0.0
    sig[~alive] = 0.0

    # ---- sfint3.F + sbulk3.F + szhOur3.F ----------------------------------
    compressing = (trD < 0.0) & alive
    qvisc = np.where(compressing,
                     rho * lc * (qa ** 2 * lc * trD ** 2 - qb * c * trD), 0.0)
    S = np.empty((n, 3, 3))
    S[:, 0, 0] = sig[:, 0] - qvisc
    S[:, 1, 1] = sig[:, 1] - qvisc
    S[:, 2, 2] = sig[:, 2] - qvisc
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]
    fe = (dndx @ S) * (-vol)[:, None, None]                # sfint3.F

    hx = solid_hexa8._H @ xe
    gamma = solid_hexa8._H[None, :, :] - hx @ dndx.transpose(0, 2, 1)
    qdot = gamma @ ve
    ah = hcoef * rho * c * vol ** (2.0 / 3.0) / 4.0 * alive
    fhg = (gamma.transpose(0, 2, 1) @ qdot) * (-ah)[:, None, None]
    fe += fhg

    # ---- energy (the EN ledger): midpoint sigma:deps, viscous trapezoid ---
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = 0.5 * vol * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    st["qvw_pend"] = 0.5 * vol * qvisc * dt
    deint = vol * np.einsum("nk,nk->n", sig_mid, deps)
    dehour = -np.einsum("nib,nib->n", fhg, ve) * dt
    st["eint"] += deint + w_visc
    st["ehour"] += dehour

    dt_crit = _courant(group, xe, vol, lc, c, rho, detJ)
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3),
                     st.get("color_indices"), st.get("color_offsets"))
    return dt_crit


def _courant(group, xe, vol, lc, c, rho, detJ) -> np.ndarray:
    """The per-element Courant claim ``lc/c`` the cycle returns, with the
    degenerate-element and void-element escapes ``solid_hexa8.forces`` uses."""
    st = group.state
    n = group.n
    alive = st["off"] > 0.0
    dtfac = st.get("dtfac", np.ones(n))
    if np.ndim(dtfac) == 0:
        dtfac = np.full(n, float(dtfac))
    dt_e = np.where(alive, dtfac * lc / np.maximum(c, EM20), EP30)
    dt_e = np.where(detJ <= EM20, EP30, dt_e)
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, _prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    return np.where(is_void, EP30, dt_e)


def dt_claim(group, x=None):
    """The per-element critical time step this kernel would return, without
    running a cycle.

    For an isotropic group this is exactly ``solid_hexa8``'s Courant probe.
    For an orthotropic one the length is ``sz_dt1.F90``'s ``DELTAX1`` — the
    orthotropic hourglass length, which is the B-matrix quantity upstream
    gates on ``gfac = (1-2nu)/(1-nu)`` — and the wave speed comes from the
    largest normal modulus, so stiffening the material shortens the step.
    """
    st = group.state
    if st.get("conn") is None and x is None:
        x = getattr(group, "_model", None)
        x = getattr(x, "x", None)
    if not _is_orthotropic(group):
        if x is None:
            return np.empty(group.n)
        return solid_hexa8.forces(group, x, None, None, 0.0, None, None)

    n = group.n
    if n == 0 or len(group.conn) == 0 or x is None:
        return np.empty(0)
    xe = x[group.conn]
    dndx, vol = solid_hexa8._geometry(xe)
    vol = np.maximum(vol, EM20)
    lc = solid_hexa8._char_length(xe, vol) * st.get("lc_scale", np.ones(n))
    rho = st["mass"] / vol
    D = st["D"]
    c = np.sqrt(np.maximum(D[:, 0, 0], 0.0) / np.maximum(rho, EM20))
    gfac = st.get("gfac", np.zeros(n))
    px, py, pz = dndx[:, :, 0], dndx[:, :, 1], dndx[:, :, 2]
    length = np.empty(n)
    for value in np.unique(gfac):
        sel = gfac == value
        length[sel] = sz_dt1(px[sel], py[sel], pz[sel], float(value))
    # sz_dt1 returns zero where its gate is closed; fall back to the
    # sdlen3.F length there rather than claiming no time step at all
    length = np.where(length > 0.0, length, lc)
    return np.where(rho > 0.0, length / np.maximum(c, EM20), EP30)


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness ``K_e = V Bᵀ D B`` plus the hourglass
    stabilization — the M8 entry point beside ``forces``, sharing its
    geometry, addressing and hourglass operators with ``solid_hexa8``.

    ``D`` is the element-frame Voigt moduli of :func:`stress_from_strain`, so
    the tangent is CONSISTENT with the residual by construction: for a linear
    orthotropic card it is exactly the derivative of the stress update.  It
    is never returned zero for a live element — a zero tangent is the failure
    the ``checks.py``-side implicit gate must never see, and the orthotropic
    hourglass that would replace part of it is **not** ported (see the module
    docstring), so the hourglass block is ``solid_hexa8``'s.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    xe = x[conn]
    dndx, vol = solid_hexa8._geometry(xe)
    vol = np.maximum(vol, EM20)

    ke = np.zeros((n, 24, 24))
    ix = np.arange(8)
    gx, gy, gz = dndx[:, :, 0], dndx[:, :, 1], dndx[:, :, 2]
    B = np.zeros((n, 6, 24))
    B[:, 0, 3 * ix + 0] = gx
    B[:, 1, 3 * ix + 1] = gy
    B[:, 2, 3 * ix + 2] = gz
    B[:, 3, 3 * ix + 0] = gy
    B[:, 3, 3 * ix + 1] = gx
    B[:, 4, 3 * ix + 1] = gz
    B[:, 4, 3 * ix + 2] = gy
    B[:, 5, 3 * ix + 0] = gz
    B[:, 5, 3 * ix + 2] = gx

    if "D" not in st:
        raise ValueError(
            "the element-frame Voigt moduli are absent from the element "
            "buffer: solid_orthotropic.init_group must run before tangent() "
            "(it builds D from mmodul.F). A missing D here would otherwise "
            "yield a silent ZERO tangent, which the implicit gate cannot "
            "distinguish from a void element."
        )
    D = st["D"]
    ke = vol[:, None, None] * np.einsum("nji,njk,nkl->nil", B, D, B, optimize=True)

    _conn, _gamma, GG, k_hg, _k_stiff = solid_hexa8._hg_operators(group, x)
    kh = k_hg[:, None, None] * GG
    for b in range(3):
        ke[:, (3 * ix + b)[:, None], (3 * ix + b)[None, :]] += kh

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, _prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0
    return ke, solid_hexa8._edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness ``g_ab = V ∇N_a · σ · ∇N_b``
    from the CURRENT stress state, shaped and addressed exactly like
    :func:`tangent`.  Zero where the stress is zero, which is where a deleted
    element's stress is zeroed.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    dndx, vol = solid_hexa8._geometry(x[conn])
    vol = np.maximum(vol, EM20)
    S = _to_tensor(np.asarray(st["sig"], dtype=float))
    g = vol[:, None, None] * np.einsum("nac,ncd,nbd->nab", dndx, S, dndx)
    ke = np.zeros((n, 24, 24))
    ix = np.arange(8)
    for b in range(3):
        ke[:, (3 * ix + b)[:, None], (3 * ix + b)[None, :]] += g

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, _prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0
    return ke, solid_hexa8._edofs(conn)
