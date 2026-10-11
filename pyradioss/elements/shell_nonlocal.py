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
#   engine/source/elements/sh3n/coquedk/cdkfint_reg.F   (modified: transposed
#     to NumPy, /PROP/SHELL thickness dofs instead of the NPTTOT common block,
#     the PARITH/ON FSKY branch dropped — see the deviation notes in
#     tools/validation_data/shell_routine_status.json)
#   engine/source/elements/sh3n/coquedk/dtcdk_reg.F      (modified: same)
#   engine/source/elements/sh3n/coquedk/cdkforc3.F        (VAR_REG pre-treatment,
#     cdkforc3.F:392-424)
#   starter/source/elements/sh3n/coquedk/ + cdkfint_reg_ini.F (the sweeps)
#   starter/source/materials/fail/nloc_dmg_init.F         (NLOC_DMG set-up)
#   starter/source/materials/nonlocal/hm_read_nonlocal.F   (/NONLOCAL read)
#   common_source/modules/nlocal_reg_mod.F                (NLOCAL_STR_, GET_LENGTH)
#   engine/source/elements/shell/coqini.F                 (the Z01/WF1/ZN1 DATA)
#
"""
Element-side non-local (regularised) damage path for DKT18 shells.

WHAT THIS IS
------------
The port had **no element-side non-local path at all**:
``pyradioss/materials/law130_nonlocal_damage.py`` is the Gauss-point-local
damage law, and ``/NONLOCAL/{id}`` and ``/MAT/NONLOCAL/{id}`` were parsed into
``model.nonlocal_models`` and ``model.mat_nonlocal_modifiers`` and then read by
nothing.  This module is the missing half: the additional (non-local) degrees
of freedom that the regularised field lives on, the force that drives them,
and the time step they claim.

Upstream shape, and where each piece lives
-----------------------------------------
::

    starter/source/materials/fail/nloc_dmg_init.F
        reads /NONLOCAL, derives DENS/DAMP/SSPNL, numbers the non-local dofs
        (IDXI/POSI/L_NLOC) and builds MASS/MASS0
    starter/source/materials/nonlocal/hm_read_nonlocal.F
        LEN  <-> LE_MAX   (GET_LENGTH / GET_LEMAX, mutually inverse)
    starter/source/elements/sh3n/coquedk/ + cdkfint_reg_ini.F
        the initial sweep: DNL -> VAR_REG -> FNL -> VNL -> UNL
    engine/source/elements/sh3n/coquedk/cdkforc3.F:392-424
        VAR_REG(I,K) = (1-ETA-KSI)*DNL(POS1+K-1) + ETA*DNL(POS2+K-1)
                             + KSI*DNL(POS3+K-1)          at each Hammer point
    engine/source/elements/sh3n/coquedk/cdkforc3.F:498, :573
        the two guarded calls this module transcribes

    engine/source/elements/sh3n/coquedk/cdkfint_reg.F
        the in-thickness regularisation over NDDL sub-dofs, then the plane
        non-local force, then the assembly
    engine/source/elements/sh3n/coquedk/dtcdk_reg.F
        the DT2T reduction

Why the init sweep is not optional
----------------------------------
``VAR_REG`` is built from ``DNL``, and ``DNL`` is produced by integrating
``VNL``, which is produced by integrating ``FNL``, whose only source is
``VAR_REG``.  That is a pure fixed point: from a zero field it stays zero
forever.  Upstream breaks it once, in the Starter, with
``cdkfint_reg_ini.F``.  Without that pass this module would be structurally
present and numerically dead, so it is transcribed too even though it is not
one of the two routines the audit named.

SCOPE — shell side only
-----------------------
The solid-side non-local routines (``solid/solide/sfint_reg.F``,
``s4fint_reg.F``, ``s6fint_reg.F90``, ``s8zfint_reg.F``) stay ``refused`` in
``tools/validation_data/solid_routine_status.json``; that is Phase 2's
decision and this module does not touch it.  ``NonlocalShell`` therefore owns
only the DKT18 (triangle shell) nodes and says so in its docstring.  The
PARITH/ON ``FSKY``/``STSKY`` assembly leg of ``cdkfint_reg.F`` is the
``IPARIT != 0`` branch; mpi4py is not installed in this environment so every
run is single-rank, which is upstream's ``IPARIT == 0`` branch — that is the
branch transcribed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..common.constants import EM20

# --------------------------------------------------------------------------
# Module parameters, each with its upstream site.
# --------------------------------------------------------------------------

#: nlocal_reg_mod.F:138 — safety coefficient, non-local vs mechanical step.
CSTA = 40.0
#: nlocal_reg_mod.F:141 — non-local stability vs the damping term.
CDAMP = 0.7
#: nloc_dmg_init.F:112 — damping ratio (0 < ETA < 1: underdamped).
ETA = 0.2
#: constant_mod.F:1004 ``PI = ATAN2(ZERO,-ONE)``.
PI = math.pi
#: cdkfint_reg.F:400 — the ``ONEP00001`` mass-addition safety factor.
ONEP00001 = 1.00001


def get_lemax(nl_length: float) -> float:
    """``nlocal_reg_mod.F:203-207`` — ``LE_MAX = SQRT(PI)*NLOC_LENGTH/THREE``."""
    return math.sqrt(PI) * nl_length / 3.0


def get_length(le_max: float) -> float:
    """``nlocal_reg_mod.F:218-222`` — ``NLOC_LENGTH = THREE*LE_MAX/SQRT(PI)``."""
    return 3.0 * le_max / math.sqrt(PI)


# --------------------------------------------------------------------------
# coqini.F DATA blocks — Z01/WF1/ZN1, transcribed digit for digit.
#
# These are DATA, not formulas.  Upstream builds them once at startup in
# ``coqini.F`` (``radioss2.F`` calls it) and every shell kernel and non-local
# routine reads them through the ``COM20`` common block.  Plan Task P3.4 owns
# the canonical home for them (``shell_basis.Z01/WF1/WM1/ZN1``); this module
# keeps its own copy so it does not depend on a task that has not landed, and
# the two must be reconciled when P3.4 does.
# --------------------------------------------------------------------------

#: The ``DATA`` block of ``coqini.F``, one Python tuple per Fortran *column*
#: (Fortran DATA fills in column-major order: ``Z01`` and ``WF1`` are 11x11 and
#: ``ZN1`` is 12x11, so each logical record below is column ``nddl - 1``).
_Z01_DATA = (
    (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.1666667, 0.1666667, 0.5, 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.25, 0., 0.25, 0.5, 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0., 0., 0., 0., 0.),
    (-0.5, -0.3333333, -0.1666667, 0.0, 0.1666667, 0.3333333, 0.5,
     0., 0., 0., 0.),
    (-0.5, -0.3571429, -0.2142857, -0.0714286, 0.0714286, 0.2142857,
     0.3571429, 0.5, 0., 0., 0.),
    (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0., 0.),
    (-0.5, -0.3888889, -0.2777778, -0.1666667, -0.0555555, 0.0555555,
     0.1666667, 0.2777778, 0.3888889, 0.5, 0.),
    (-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5),
)
_WF1_DATA = (
    (1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.),
    (0.1666667, 0.3333333, 0.3333333, 0.1666667, 0., 0., 0., 0., 0., 0., 0.),
    (0.125, 0.25, 0.25, 0.25, 0.125, 0., 0., 0., 0., 0., 0.),
    (0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0., 0., 0., 0., 0.),
    (0.0833333, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667,
     0.0833333, 0., 0., 0., 0.),
    (0.0714286, 0.1428571, 0.1428571, 0.1428571, 0.1428571, 0.1428571,
     0.1428571, 0.0714286, 0., 0., 0.),
    (0.0625, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0625,
     0., 0.),
    (0.0555556, 0.1111111, 0.1111111, 0.1111111, 0.1111111, 0.1111111,
     0.1111111, 0.1111111, 0.1111111, 0.0555556, 0.),
    (0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05),
)
_ZN1_DATA = (
    (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.25, 0.25, 0.5, 0., 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.3333333, 0., 0.3333333, 0.5, 0., 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.375, -0.125, 0.125, 0.375, 0.5, 0., 0., 0., 0., 0., 0.),
    (-0.5, -0.4, -0.2, 0., 0.2, 0.4, 0.5, 0., 0., 0., 0., 0.),
    (-0.5, -0.4166667, -0.25, -0.0833333, 0.0833333, 0.25, 0.4166667, 0.5,
     0., 0., 0., 0.),
    (-0.5, -0.4285715, -0.2857143, -0.1428572, 0., 0.1428572, 0.2857143,
     0.4285715, 0.5, 0., 0., 0.),
    (-0.5, -0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125,
     0.4375, 0.5, 0., 0.),
    (-0.5, -0.4444444, -0.3333333, -0.2222222, -0.1111111, 0., 0.1111111,
     0.2222222, 0.3333333, 0.4444444, 0.5, 0.),
    (-0.5, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35,
     0.45, 0.5),
)

Z01 = np.array(_Z01_DATA, dtype=float).T.copy()
#: ``COM20 Z0(11,11)`` — through-thickness Gauss station positions in [-0.5, 0.5].
Z0 = Z01
WF = np.array(_WF1_DATA, dtype=float).T.copy()
#: ``COM20 WF(11,11)`` — through-thickness weights, already normalised by the
#: thickness (every column sums to 1).
WF1 = WF
#: ``COM20 ZTH(12,11)`` — the in-thickness sub-dof node positions.
ZTH = np.array(_ZN1_DATA, dtype=float).T.copy()
ZN1 = ZTH


def _zth(nddl: int) -> np.ndarray:
    """``ZTH(1:nddl+1, nddl)`` — the in-thickness sub-dof node positions."""
    return ZTH[:nddl + 1, nddl - 1]


# --------------------------------------------------------------------------
# NLOC_DMG — the shell-side non-local store.
# --------------------------------------------------------------------------


@dataclass
class NlocalMaterial:
    """One column of the per-material non-local tables.

    Fortran origin: ``NLOCAL_STR_`` (``nlocal_reg_mod.F:143-191``) — ``LEN``,
    ``LE_MAX``, ``DENS``, ``DAMP`` and ``SSPNL``, all dimensioned ``(NUMMAT)``.
    """

    mat_id: int
    length: float = 0.0      # NLOC_DMG%LEN     — non-local internal length
    le_max: float = 0.0      # NLOC_DMG%LE_MAX  — maximal convergence length
    dens: float = 0.0        # NLOC_DMG%DENS    — non-local density
    damp: float = 0.0        # NLOC_DMG%DAMP    — non-local damping
    sspnl: float = 0.0       # NLOC_DMG%SSPNL   — non-local sound speed


class NonlocalThickness:
    """``BUFNL%MASSTH/UNLTH/VNLTH/FNLTH`` — the in-thickness sub-dof buffers.

    ``cdkfint_reg.F:206-213`` points at ``BUFNL`` for the element group, and
    the buffers are ``(NEL, NDNOD)`` where ``NDNOD`` is ``NDDL`` (two
    stations) or ``NDDL+1`` (the general case, ``cdkfint_reg.F:186-203``).
    """

    __slots__ = ("massth", "unlth", "vnlth", "fnlth")

    def __init__(self, nel: int, ndnod: int):
        self.massth = np.zeros((nel, ndnod))
        self.unlth = np.zeros((nel, ndnod))
        self.vnlth = np.zeros((nel, ndnod))
        self.fnlth = np.zeros((nel, ndnod))


class NonlocalShell:
    """``NLOC_DMG``, shell (triangle) side.

    **Shell side only.**  Upstream one global store serves solids, shells and
    triangles; the solid rows in ``solid_routine_status.json`` stay
    ``refused``, so this store owns the DKT18 nodes alone and its ``idxi`` is
    ``-1`` on every node no DKT18 element touches.  Indexing is 0-based here
    against 1-based Fortran: ``idxi[node]`` is the non-local node number (or
    ``-1``), ``posi[k]`` the first dof slot of non-local node ``k``, and
    ``posi[nnod] == l_nloc``.
    """

    def __init__(self) -> None:
        self.materials: Dict[int, NlocalMaterial] = {}
        self.idxi = np.zeros(0, dtype=np.int64)
        self.posi = np.zeros(1, dtype=np.int64)
        self.nmat = np.zeros(0, dtype=np.int64)
        self.nddl = np.zeros(0, dtype=np.int64)
        self.nnod = 0
        self.l_nloc = 0
        self.mass = np.zeros(0)
        self.mass0 = np.zeros(0)
        self.fnl = np.zeros(0)
        self.vnl = np.zeros(0)
        self.vnl_old = np.zeros(0)
        self.dnl = np.zeros(0)
        self.unl = np.zeros(0)
        self.stifnl = np.zeros(0)
        #: per-group in-thickness buffers, keyed by ``id(group)``.
        self._thickness: Dict[int, NonlocalThickness] = {}
        # -- engine controls ----------------------------------------------
        # Upstream reads these from the global SCR18/IDTMIN/DTFAC1 common
        # blocks.  The port's Engine writes them here once per run (see
        # engine.py); the Starter-side initial sweep runs before that, so
        # they default to the no-/DT/NODA case it computes under.
        self.nodadt = 0            # NODADT > 0 iff /DT/NODA is on
        self.dt_fac1 = 0.9         # DTFAC1(1)  — the global dT scale
        self.dt_fac_noda = 0.9     # DTFAC1(11) — the /DT/NODA dT scale
        self.dt_min_noda = 0.0     # DTMIN1(11)
        self.idtn_min = 0          # IDTMIN(11)  — 3/4/8 are the CSTX variants

    # -- storage ---------------------------------------------------------
    def allocate(self) -> None:
        """Zero every vector (``nloc_dmg_init.F:1009-1015``)."""
        n = self.l_nloc
        self.fnl = np.zeros(n)
        self.vnl = np.zeros(n)
        self.vnl_old = np.zeros(n)
        self.dnl = np.zeros(n)
        self.unl = np.zeros(n)
        self.stifnl = np.zeros(n)
        if self.mass.size != n:
            self.mass = np.zeros(n)
            self.mass0 = np.zeros(n)

    def thickness(self, group_id: int, nel: int, ndnod: int) -> NonlocalThickness:
        """Fetch (creating on first use) a group's in-thickness buffers."""
        bufnl = self._thickness.get(group_id)
        if bufnl is None or bufnl.massth.shape != (nel, ndnod):
            bufnl = NonlocalThickness(nel, ndnod)
            self._thickness[group_id] = bufnl
        return bufnl

    # -- the four engine phases (assembly/nlocal_{incr,acc,vel}.F) --------
    def incr(self, dt2: float) -> None:
        """``nlocal_incr.F:60-64`` — ``DNL = DT2*VNL`` then ``UNL += DNL``."""
        if self.l_nloc == 0:
            return
        self.dnl[:] = dt2 * self.vnl
        self.unl += self.dnl

    def acc(self) -> None:
        """``nlocal_acc.F:60-63`` — ``FNL /= MASS`` (acceleration in place)."""
        if self.l_nloc == 0:
            return
        self.fnl /= np.maximum(self.mass, EM20)

    def vel(self, dt12: float) -> None:
        """``nlocal_vel.F:66-71`` — save VNL0, ``VNL += DT12*FNL``, clear FNL."""
        if self.l_nloc == 0:
            return
        self.vnl_old[:] = self.vnl
        self.vnl += dt12 * self.fnl
        self.fnl[:] = 0.0

    def cycle_step(self, dt: float) -> None:
        """The three per-cycle phases, in resol.F's order.

        ``NLOCAL_ACC`` (resol.F:6948) -> ``NLOCAL_VEL`` (:8965) ->
        ``NLOCAL_INCR`` (:9064), run after every element group has assembled
        ``FNL``.  They touch the non-local dof vector only — never ``fint``.
        """
        self.acc()
        self.vel(0.5 * dt)
        self.incr(0.5 * dt)


# --------------------------------------------------------------------------
# /NONLOCAL and /MAT/NONLOCAL -> the per-material table.
# --------------------------------------------------------------------------


def materials_from_model(model) -> Dict[int, NlocalMaterial]:
    """Read the two keywords the Starter already parses into a material table.

    ``/NONLOCAL/{id}`` lands in ``model.nonlocal_models`` (``read_nonlocal``)
    and ``/MAT/NONLOCAL/{id}`` in ``model.mat_nonlocal_modifiers``
    (``read_mat_nonlocal``, M171).  Both describe the same thing upstream —
    ``hm_read_nonlocal.F`` is the only reader that writes ``NLOC_DMG%LEN`` —
    so both feed one table and ``/NONLOCAL`` wins where both are present.

    The ``LEN <-> LE_MAX`` pair is made mutually consistent here, exactly as
    ``hm_read_nonlocal.F:150-158`` does: a positive ``LE_MAX`` is converted to
    ``LEN``, a zero one derives ``LE_MAX`` from ``LEN``.
    """
    out: Dict[int, NlocalMaterial] = {}
    for mat_id, mod in getattr(model, "mat_nonlocal_modifiers", {}).items():
        out[int(mat_id)] = NlocalMaterial(
            mat_id=int(mat_id), length=float(mod.length),
            le_max=float(mod.le_max),
            dens=float(getattr(mod, "dens", 0.0) or 0.0),
            damp=float(getattr(mod, "damp", 0.0) or 0.0))
    for mat_id, nlm in getattr(model, "nonlocal_models", {}).items():
        out[int(mat_id)] = NlocalMaterial(
            mat_id=int(mat_id), length=float(nlm.length),
            le_max=float(nlm.le_max),
            dens=float(getattr(nlm, "dens", 0.0) or 0.0),
            damp=float(getattr(nlm, "damp", 0.0) or 0.0))
    for mat in out.values():
        if mat.le_max > 0.0:
            mat.length = get_length(mat.le_max)
        elif mat.length > 0.0:
            mat.le_max = get_lemax(mat.length)
    return out


def nlocal_material_parameters(length: float, le_max: float, *,
                              sound_speed: float, dt_min: float,
                              dtmin_ams: float = 0.0) -> NlocalMaterial:
    """``nloc_dmg_init.F:903-921`` — derive ``DENS``, ``DAMP`` and ``SSPNL``.

    ::

        DTMIN = MAX(LE_MAX/SSP, DTMINI)      DTMINI = max(DTMIN_AMS, DTMIN_CST_AMS)
        DENS  = CSTA*((LEN/LE_MAX)**2 + 1/12)*DTMIN**2
        DAMP  = (2*ETA/LE_MAX)*SQRT(DENS*(LEN**2*PI**2 + LE_MAX**2))
        SSPNL = SQRT((LEN**2 + LE_MAX**2/PI**2)/DENS)
    """
    dtmin = max(dt_min, dtmin_ams)
    length = max(length, EM20)
    le_max = max(le_max, EM20)
    dt_eff = max(le_max / max(sound_speed, EM20), dtmin)
    dens = CSTA * (((length / le_max) ** 2) + 1.0 / 12.0) * dt_eff ** 2
    damp = (2.0 * ETA / le_max) * math.sqrt(
        dens * ((length ** 2) * (PI ** 2) + le_max ** 2))
    sspnl = math.sqrt(((length ** 2) + le_max ** 2 / PI ** 2) / max(dens, EM20))
    return NlocalMaterial(mat_id=0, length=length, le_max=le_max,
                         dens=max(dens, 0.0), damp=max(damp, 0.0),
                         sspnl=max(sspnl, 0.0))


def _nddnod(nddl: int) -> int:
    """``cdkfint_reg.F:186-203`` — ``NDNOD`` for an ``NDDL``-station thickness."""
    return nddl + 1 if nddl > 2 else nddl


def _nth(k: int, nddl: int) -> tuple:
    """The ``NTH1``/``NTH2`` shape values at station ``k`` (0-based).

    ``cdkfint_reg.F:207-216``.  ``NDDL == 2`` needs a special case at its
    second station because the extra sub-dof sits *below* the pair, not above.
    """
    z0 = Z0[:, nddl - 1]
    zth = _zth(nddl)
    if nddl == 2 and k == 1:
        return ((z0[k] - zth[k]) / (zth[k - 1] - zth[k]),
                (z0[k] - zth[k - 1]) / (zth[k] - zth[k - 1]))
    return ((z0[k] - zth[k + 1]) / (zth[k] - zth[k + 1]),
            (z0[k] - zth[k]) / (zth[k + 1] - zth[k]))


def _bth(k: int, nddl: int, thk: np.ndarray) -> tuple:
    """The ``BTH1``/``BTH2`` weights (``cdkfint_reg.F:220-229``)."""
    zth = _zth(nddl)
    if nddl == 2 and k == 1:
        return ((1.0 / (zth[k - 1] - zth[k])) * (1.0 / thk),
                (1.0 / (zth[k] - zth[k - 1])) * (1.0 / thk))
    return ((1.0 / (zth[k] - zth[k + 1])) * (1.0 / thk),
            (1.0 / (zth[k + 1] - zth[k])) * (1.0 / thk))


# --------------------------------------------------------------------------
# cdkfint_reg.F — the virtual internal forces of the regularised non-local dofs
# --------------------------------------------------------------------------


def cdkfint_reg(store: NonlocalShell, *, var_reg: np.ndarray, off: np.ndarray,
                area: np.ndarray, thk: np.ndarray, thk0: np.ndarray,
                area0: np.ndarray, pos1: np.ndarray, pos2: np.ndarray,
                pos3: np.ndarray, px2: np.ndarray, py2: np.ndarray,
                px3: np.ndarray, py3: np.ndarray, ksi: float, eta: float,
                imat: int, nddl: int, dt2t: float, nodadt: int = 0,
                bufnl: Optional[NonlocalThickness] = None,
                dt12: float = 0.0, dt1: float = 0.0,
                dt_fac1: float = 0.9, dt_fac_noda: float = 0.9,
                dt_min_noda: float = 0.0, idtn_min: int = 0) -> float:
    """``cdkfint_reg.F`` — the regularised non-local force of one DKT18 group.

    ``var_reg`` is the ``(NEL, NDDL)`` local variable to regularise, already
    built by the caller from ``DNL`` (``cdkforc3.F:398-424``).  The routine

    1. regularises ``VAR_REG`` through the thickness when ``NDDL > 1`` and the
       internal length is non-zero (:186-349), which also books the ``NODADT``
       nodal step and the ``/DT/NODA/CSTX`` added mass;
    2. forms the plane non-local force ``F1/F2/F3`` per element and
       station (:352-431), with the broken-element absorbing branch;
    3. assembles into ``FNL`` — and ``STIFNL`` under ``NODADT`` — subtracting
       the force (:455-476).

    Returns the reduced ``DT2T``.  Upstream's ``DT2T`` is a running global
    minimum; the engine here reduces per-element dt claims, so this returns
    the group's claim and the caller folds it into its own element array.

    Not carried: the ``IPARIT != 0`` (``/PARITH/ON``) ``FSKY``/``STSKY``
    assembly leg.  mpi4py is not installed in this environment, so every run is
    single-rank, which is upstream's ``IPARIT == 0`` branch — the one above.
    """
    mat = store.materials[imat]
    l2 = mat.length ** 2                      # :110 internal length ** 2
    xi = mat.damp                             # :111
    zeta = mat.dens                           # :112
    sspnl = mat.sspnl                         # :113
    n = off.shape[0]

    h = np.array([1.0 - eta - ksi, eta, ksi])  # :115-117
    hx = np.outer(h, h)

    # -- element volume and BtB (cdkfint_reg.F:154-176) -------------------
    vol = (area * thk) / 3.0
    a1 = -px3 - px2
    a2 = -py2 - py3
    btb = np.empty((n, 3, 3))
    btb[:, 0, 0] = a1 ** 2 + a2 ** 2
    btb[:, 0, 1] = a1 * px2 + a2 * py2
    btb[:, 0, 2] = a1 * px3 + a2 * py3
    btb[:, 1, 0] = btb[:, 0, 1]
    btb[:, 1, 1] = px2 ** 2 + py2 ** 2
    btb[:, 1, 2] = px2 * px3 + py2 * py3
    btb[:, 2, 0] = btb[:, 0, 2]
    btb[:, 2, 1] = btb[:, 1, 2]
    btb[:, 2, 2] = px3 ** 2 + py3 ** 2

    # ------------------------------------------------------------------
    # 1. in-thickness regularisation (cdkfint_reg.F:186-349)
    # ------------------------------------------------------------------
    if nddl > 1 and l2 > 0.0:
        if bufnl is None:
            raise ValueError(
                "cdkfint_reg: NDDL > 1 needs the group's in-thickness buffers "
                "(cdkfint_reg.F:206-213 takes BUFNL%MASSTH/UNLTH/VNLTH/FNLTH)")
        ndnod = _nddnod(nddl)
        massth, unlth, vnlth, fnlth = (bufnl.massth, bufnl.unlth,
                                       bufnl.vnlth, bufnl.fnlth)
        if massth.shape != (n, ndnod):
            raise ValueError(
                f"cdkfint_reg: BUFNL buffers are {massth.shape}, the group "
                f"needs {(n, ndnod)} (NDNOD = NDDL+1 when NDDL > 2, "
                f"cdkfint_reg.F:194)")
        fnlth[:] = 0.0
        stifnlth = np.full((n, ndnod), EM20) if nodadt > 0 else None

        for k in range(nddl):                                   # :217
            nth1, nth2 = _nth(k, nddl)
            bth1, bth2 = _bth(k, nddl, thk)
            k1 = l2 * bth1 ** 2 + nth1 ** 2                     # :233-235
            k12 = l2 * (bth1 * bth2) + nth1 * nth2
            k2 = l2 * bth2 ** 2 + nth2 ** 2
            wvol = vol * WF[k, nddl - 1]                        # VOL*WF(K,NDDL)
            lo, hi = (k - 1, k) if (nddl == 2 and k == 1) else (k, k + 1)
            nl_a = nth1 * (nth1 * unlth[:, lo] + nth2 * unlth[:, hi])
            nl_b = nth2 * (nth1 * unlth[:, lo] + nth2 * unlth[:, hi])
            nv_a = nth1 * (nth1 * vnlth[:, lo] + nth2 * vnlth[:, hi])
            nv_b = nth2 * (nth1 * vnlth[:, lo] + nth2 * vnlth[:, hi])

            fnlth[:, lo] += (k1 * unlth[:, lo] + k12 * unlth[:, hi]
                             + xi * nv_a - nth1 * var_reg[:, k]) * wvol
            fnlth[:, hi] += (k12 * unlth[:, lo] + k2 * unlth[:, hi]
                             + xi * nv_b - nth2 * var_reg[:, k]) * wvol
            if nodadt > 0:                                     # :276-286
                add = (np.maximum(np.abs(k1) + np.abs(k12),
                                  np.abs(k12) + np.abs(k2)) * wvol)
                stifnlth[:, lo] += add
                stifnlth[:, hi] += add

        if nodadt > 0:                                         # :288-320
            dtn = dt_fac_noda * CDAMP * np.sqrt(
                2.0 * massth / np.maximum(stifnlth, EM20))
            dtnod = float(dtn.min())
            if idtn_min in (3, 4, 8) and dt_min_noda > 0.0:
                if dtnod < dt_min_noda * math.sqrt(CSTA):       # /DT/NODA/CSTX
                    dt2p = dt_min_noda / (dt_fac_noda * CDAMP)
                    short = dtn < dt_min_noda
                    bump = CSTA * 0.5 * stifnlth * dt2p * dt2p * ONEP00001
                    massth[:] = np.where(short, np.maximum(massth, bump), massth)
                    dtnod = dt_min_noda * math.sqrt(CSTA)
            if dtnod < dt2t:
                dt2t = min(dt2t, dtnod)

        vnlth -= (fnlth / np.maximum(massth, EM20)) * dt12      # :322-328
        alive = off != 0.0                                     # :330-340
        unlth += vnlth * dt1 * alive[:, None]

        for k in range(nddl):                                  # :344-349
            nth1, nth2 = _nth(k, nddl)
            lo, hi = (k - 1, k) if (nddl == 2 and k == 1) else (k, k + 1)
            var_reg[:, k] = nth1 * unlth[:, lo] + nth2 * unlth[:, hi]

    # ------------------------------------------------------------------
    # 2. plane non-local force (cdkfint_reg.F:352-431)
    # ------------------------------------------------------------------
    unl, vnl, vnl0 = store.unl, store.vnl, store.vnl_old
    alive = off != 0.0

    for k in range(nddl):                                      # :354
        wf = WF[k, nddl - 1]
        i0 = pos1 + k
        i1 = pos2 + k
        i2 = pos3 + k
        u = np.stack([unl[i0], unl[i1], unl[i2]], axis=1)      # (n, 3)
        v = np.stack([vnl[i0], vnl[i1], vnl[i2]], axis=1)

        b = (l2 * vol * wf)[:, None] * np.einsum("nij,nj->ni", btb, u)
        ntn_unl = (hx @ u.T).T * (vol * wf)[:, None]           # :369-372
        ntn_vnl = (hx @ v.T).T * (xi * vol * wf)[:, None]
        if nodadt > 0:                                         # :373-390
            ratio = np.minimum(
                np.minimum(np.sqrt(store.mass[i0] / np.maximum(store.mass0[i0], EM20)),
                           np.sqrt(store.mass[i1] / np.maximum(store.mass0[i1], EM20))),
                np.sqrt(store.mass[i2] / np.maximum(store.mass0[i2], EM20)))
            ntn_vnl = ntn_vnl * ratio[:, None]
        ntvar = (var_reg[:, k] * vol * wf)[:, None] * h[None, :]  # :411-413
        f = np.where(alive[:, None], ntn_unl + ntn_vnl - ntvar + b, 0.0)

        sti = None
        if nodadt > 0:                                         # :418-427
            sti = (np.abs(l2 * btb + hx[None, :, :]).sum(axis=2)
                   * (vol * wf)[:, None])

        # broken element: the non-local wave is absorbed (:433-451)
        if not alive.all():
            absorb = np.sqrt((4.0 / math.sqrt(3.0)) * area0) * thk0
            gain = wf * zeta * sspnl * absorb
            for j, idx in enumerate((i0, i1, i2)):
                half = 0.5 * (v[:, j] + vnl0[idx])
                if nodadt > 0:
                    half = half * np.sqrt(
                        store.mass[idx] / np.maximum(store.mass0[idx], EM20))
                f[:, j] = np.where(alive, f[:, j], h[j] * half * gain)
            if nodadt > 0:
                sti = np.where(alive, sti, EM20)

        # ------------------------------------------------------------------
        # 3. assembly (cdkfint_reg.F:455-476, the IPARIT == 0 branch)
        # ------------------------------------------------------------------
        np.subtract.at(store.fnl, i0, f[:, 0])
        np.subtract.at(store.fnl, i1, f[:, 1])
        np.subtract.at(store.fnl, i2, f[:, 2])
        if nodadt > 0:
            maxstif = np.maximum(np.maximum(sti[:, 0], sti[:, 1]), sti[:, 2])
            np.add.at(store.stifnl, i0, maxstif)
            np.add.at(store.stifnl, i1, maxstif)
            np.add.at(store.stifnl, i2, maxstif)

    return dt2t


# --------------------------------------------------------------------------
# dtcdk_reg.F — the non-local element time step
# --------------------------------------------------------------------------


def dtcdk_reg(store: NonlocalShell, thk: np.ndarray, off: np.ndarray,
              le: np.ndarray, imat: int, nddl: int, dt2t: float,
              *, nodadt: int = 0, dt_fac1: float = 0.9) -> float:
    """``dtcdk_reg.F`` — reduce ``DT2T`` by the non-local critical step.

    ::

        L2     = LEN(IMAT)**2            (non-local internal length squared)
        ZETA   = DENS(IMAT)              (non-local density)
        LE_MAX = LE_MAX(IMAT)            (maximal convergence length)

        IF (NODADT == 0) THEN
          IF (OFF(I)/=ZERO) THEN
            DTNL    = 2*MIN(LE,LE_MAX)*SQRT(3*ZETA)
                       / SQRT(12*L2 + MIN(LE,LE_MAX)**2)
            DTNL_TH = 2*MIN(THK/NDDL,LE_MAX)*SQRT(3*ZETA)      (NDDL > 2)
                    = 2*MIN(THK,LE_MAX)*SQRT(3*ZETA)          (NDDL == 2)
                    = EP20                                      (NDDL == 1)
            DT2T = MIN(DT2T, DTFAC1(1)*CDAMP*DTNL_TH, DTFAC1(1)*CDAMP*DTNL)

    ``NODADT == 0`` is the whole of the routine's body: upstream reaches
    ``DTCDK_REG`` unconditionally but its only guard is that one, so with
    ``/DT/NODA`` on the non-local step is bounded by the nodal leg inside
    ``cdkfint_reg.F`` instead.
    """
    if nodadt != 0:
        return dt2t
    mat = store.materials[imat]
    l2 = mat.length ** 2
    three_zeta = math.sqrt(3.0 * max(mat.dens, 0.0))
    le_max = mat.le_max
    spacing = (thk / nddl if nddl > 2 else thk) if nddl > 1 else None

    for i in range(off.shape[0]):                 # :68
        if off[i] == 0.0:                         # a deleted element claims nothing
            continue
        if nddl > 1:                              # :73-83
            dmin = min(float(spacing[i]), le_max)
            dt2t = min(dt2t, dt_fac1 * CDAMP * (
                2.0 * dmin * three_zeta
                / math.sqrt(12.0 * l2 + dmin ** 2)))
        lmin = min(float(le[i]), le_max)         # :70-72
        dt2t = min(dt2t, dt_fac1 * CDAMP * (
            2.0 * lmin * three_zeta
            / math.sqrt(12.0 * l2 + lmin ** 2)))
    return dt2t


# --------------------------------------------------------------------------
# Starter-side set-up: nloc_dmg_init.F + cdkfint_reg_ini.F
# --------------------------------------------------------------------------


def setup(model, log=None) -> Optional[NonlocalShell]:
    """Build the ``NLOC_DMG`` shell store from the model and its DKT18 groups.

    Returns ``None`` — and touches nothing — when no material carries a
    non-local modifier, which is what keeps ``cdkforc3.F``'s ``IF (INLOC > 0)``
    guard shut and the ``INLOC == 0`` cycle bit-identical.
    """
    materials = materials_from_model(model)
    if not materials:
        return None

    groups = [g for name, g in model.element_groups()
              if name == "sh3n_dkt18" and getattr(g, "state", None) is not None
              and g.state.get("mat_id") is not None
              and "nl_px2" in g.state]
    if not groups:
        return None
    group = groups[0]          # one store, one DKT18 group (see the class doc)

    store = NonlocalShell()
    store.materials = materials

    st = group.state
    conn = np.asarray(group.conn)
    mat_of = np.asarray(st["mat_id"], dtype=np.int64)
    nddl = int(st.get("nip_max", 1) or 1)
    area = np.asarray(st["area0"], dtype=float)
    thk = np.asarray(st["thick"], dtype=float)

    local = np.array([int(m) in materials for m in mat_of])
    if not local.any():
        return None

    # -- nloc_dmg_init.F:893-921: DENS / DAMP / SSPNL, per material --------
    for mat_id in sorted({int(m) for m in mat_of[local]}):
        sel = local & (mat_of == mat_id)
        le_min = float(np.sqrt((4.0 / math.sqrt(3.0)) * area[sel].min()))
        if nddl > 1:
            le_min = min(le_min, float(thk[sel].min()) / nddl)
        mat = materials[mat_id]
        young, nu, rho0 = 0.0, 0.3, 0.0
        for _sl, m, _prop in st.get("slices", []):
            if int(getattr(m, "id", -1)) == mat_id:
                young = float(getattr(m, "E", 0.0) or 0.0)
                nu = float(getattr(m, "nu", 0.3) or 0.3)
                rho0 = float(getattr(m, "rho0", 0.0) or 0.0)
                break
        ssp = math.sqrt((young / (1.0 - nu ** 2)) / rho0) if rho0 > 0.0 else 0.0
        dt_min = mat.le_max / ssp if ssp > 0.0 else 0.0
        derived = nlocal_material_parameters(mat.length, mat.le_max,
                                             sound_speed=ssp, dt_min=dt_min)
        if derived.dens > mat.dens or mat.dens <= 0.0:
            mat.dens = derived.dens
        if derived.damp > mat.damp or mat.damp <= 0.0:
            mat.damp = derived.damp
        if derived.sspnl > mat.sspnl or mat.sspnl <= 0.0:
            mat.sspnl = derived.sspnl
        if mat.le_max <= 0.0:
            mat.le_max = le_min

    # -- dof numbering (nloc_dmg_init.F:751-768) --------------------------
    tagged = np.unique(conn[local])
    idxi = np.full(model.numnod, -1, dtype=np.int64)
    idxi[tagged] = np.arange(tagged.size, dtype=np.int64)
    nnod = int(tagged.size)
    per = np.full(nnod, nddl, dtype=np.int64)
    posi = np.zeros(nnod + 1, dtype=np.int64)
    posi[:-1] = np.cumsum(per) - per
    posi[-1] = int(per.sum())
    nmat = np.zeros(nnod, dtype=np.int64)
    for i, node in enumerate(tagged):
        nmat[i] = mat_of[local & (conn == node).any(axis=1)][0]

    store.idxi = idxi
    store.posi = posi
    store.nmat = nmat
    store.nddl = per
    store.nnod = nnod
    store.l_nloc = int(posi[-1])
    store.allocate()

    # -- non-local masses (nloc_dmg_init.F:1018-1035 + :1070-1102) ------
    vol = area * thk / 3.0
    for i, node in enumerate(tagged):
        voln = float(vol[local & (conn == node).any(axis=1)].sum())
        dens = materials[int(nmat[i])].dens
        lo = int(posi[i])
        if nddl > 1:
            ws = 1.0                     # a DKT18 group has one plane station
            for k in range(nddl):
                nth1, nth2 = _nth(k, nddl)
                lo_k, hi_k = (k - 1, k) if (nddl == 2 and k == 1) else (k, k + 1)
                store.mass[lo + lo_k] += ((nth1 ** 2 + nth1 * nth2) * dens
                                          * voln * ws * WF[k, nddl - 1])
                store.mass[lo + hi_k] += ((nth2 ** 2 + nth1 * nth2) * dens
                                          * voln * ws * WF[k, nddl - 1])
        else:
            store.mass[lo] = WF[0, 0] * voln * dens
    store.mass0[:] = store.mass

    # -- per-element dof positions (cdkfint_reg.F:143-147) ---------------
    nl = idxi[conn[local]]
    st["nl_pos1"] = np.where(local, posi[nl[:, 0]], 0)
    st["nl_pos2"] = np.where(local, posi[nl[:, 1]], 0)
    st["nl_pos3"] = np.where(local, posi[nl[:, 2]], 0)
    st["nl_local"] = local
    st["nlocal"] = store
    st["nloc_id"] = id(group)
    st["nl_imat"] = mat_of
    st["nl_imat0"] = int(mat_of[local][0])
    st["nl_nddl"] = nddl
    st["inloc"] = 1

    nlocal_initialise(store, group, log)
    if log is not None:
        log.info(f"NON-LOCAL REGULARIZATION: {nnod} nodes, {store.l_nloc} "
                 f"additional d.o.fs (DKT18)")
    return store


def initialise_group(store: NonlocalShell, group, nddl: int) -> float:
    """``cdkfint_reg_ini.F`` — one Starter-side sweep of the group.

    Assembles the group's non-local force from the current ``UNL``/``VNL`` and
    the local variable at the integration points.  It returns the group's
    nominal non-local step ``DT_NL`` so the caller can drive the iteration
    (``nlocal_init_sta.F`` calls this routine once per pass; the ``MIN``
    against ``DT_NL`` accumulates across groups there and is a one-group
    no-op here).

    Without this sweep the non-local dofs sit at the zero fixed point of their
    own update rule for ever: ``VAR_REG`` is built from ``DNL``, ``DNL``
    integrates ``VNL``, ``VNL`` integrates ``FNL``, and ``FNL``'s only source
    is ``VAR_REG``.
    """
    st = group.state
    local = st["nl_local"]
    if not local.any():
        return 100.0
    n = int(local.sum())
    bufnl = store.thickness(st["nloc_id"], n, _nddnod(nddl))
    st["nl_bufnl"] = bufnl

    area = np.asarray(st["area0"])[local]
    thk = np.asarray(st["thick"])[local]
    off = np.asarray(st["off"])[local]
    pos1, pos2, pos3 = (st["nl_pos1"][local], st["nl_pos2"][local],
                        st["nl_pos3"][local])
    px2, py2 = st["nl_px2"][local], st["nl_py2"][local]
    px3, py3 = st["nl_px3"][local], st["nl_py3"][local]

    mat = store.materials[int(st["nl_imat"][local][0])]

    # -- the nominal non-local step (cdkfint_reg_ini.F:176-193) ----------
    le_min = float(np.sqrt((4.0 / math.sqrt(3.0)) * area.min()))
    if nddl > 1:
        le_min = min(le_min, float(thk.min()) / nddl)
    lmin = min(le_min, mat.le_max)
    dt_nl = 0.5 * ((2.0 * lmin * math.sqrt(3.0 * max(mat.dens, 0.0)))
                   / math.sqrt(12.0 * mat.length ** 2 + lmin ** 2))

    vol = area * thk                       # VOLS: the full element volume
    ks = np.arange(nddl)

    # -- VAR_REG at the integration points (cdkfint_reg_ini.F:262-268) ----
    var_reg = np.zeros((n, nddl))
    for node in (pos1, pos2, pos3):
        var_reg += store.dnl[node[:, None] + ks[None, :]]
    var_reg /= 3.0

    # -- in-thickness sweep (cdkfint_reg_ini.F:301-419) ------------------
    if nddl > 1 and mat.length > 0.0:
        vpred = bufnl.vnlth - (bufnl.fnlth
                               / np.maximum(bufnl.massth, EM20)) * (dt_nl * 0.5)
        bufnl.fnlth[:] = 0.0
        l2 = mat.length ** 2
        for k in range(nddl):
            nth1, nth2 = _nth(k, nddl)
            bth1, bth2 = _bth(k, nddl, thk)
            k1 = l2 * bth1 ** 2 + nth1 ** 2
            k12 = l2 * (bth1 * bth2) + nth1 * nth2
            k2 = l2 * bth2 ** 2 + nth2 ** 2
            wvol = vol * WF[k, nddl - 1]
            lo, hi = (k - 1, k) if (nddl == 2 and k == 1) else (k, k + 1)
            ul, uh = bufnl.unlth[:, lo], bufnl.unlth[:, hi]
            vl, vh = vpred[:, lo], vpred[:, hi]
            bufnl.fnlth[:, lo] += (k1 * ul + k12 * uh + mat.damp
                                   * (nth1 ** 2 * vl + nth1 * nth2 * vh)
                                   - nth1 * var_reg[:, k]) * wvol
            bufnl.fnlth[:, hi] += (k12 * ul + k2 * uh + mat.damp
                                   * (nth1 * nth2 * vl + nth2 ** 2 * vh)
                                   - nth2 * var_reg[:, k]) * wvol
        bufnl.vnlth -= (bufnl.fnlth
                        / np.maximum(bufnl.massth, EM20)) * dt_nl
        bufnl.unlth += bufnl.vnlth * dt_nl
        for k in range(nddl):
            nth1, nth2 = _nth(k, nddl)
            lo, hi = (k - 1, k) if (nddl == 2 and k == 1) else (k, k + 1)
            var_reg[:, k] = (nth1 * bufnl.unlth[:, lo]
                             + nth2 * bufnl.unlth[:, hi])

    # -- plane sweep (cdkfint_reg_ini.F:430-473) -------------------------
    # uniform 1/3 shape functions and the FULL element volume: a different
    # expression from cdkfint_reg.F's Hammer-point form, transcribed
    # separately rather than shared, as upstream does.
    unl, vnl = store.unl, store.vnl
    u = np.stack([unl[pos1[:, None] + ks[None, :]],
                  unl[pos2[:, None] + ks[None, :]],
                  unl[pos3[:, None] + ks[None, :]]], axis=1)      # (n, 3, NDDL)
    v = np.stack([vnl[pos1[:, None] + ks[None, :]],
                  vnl[pos2[:, None] + ks[None, :]],
                  vnl[pos3[:, None] + ks[None, :]]], axis=1)
    btb = _btb_init(px2, py2, px3, py3)
    alive = off != 0.0
    for k in range(nddl):
        wvol = vol * WF[k, nddl - 1]
        b = ((mat.length ** 2) * wvol)[:, None] * np.einsum(
            "nij,nj->ni", btb, u[:, :, k])
        ntn_unl = u[:, :, k].sum(axis=1) * (wvol / 9.0)
        ntn_vnl = v[:, :, k].sum(axis=1) * (mat.damp * wvol / 9.0)
        ntvar = var_reg[:, k] * (wvol / 3.0)
        f = ntn_unl + ntn_vnl - ntvar + b
        if not alive.all():
            absorb = (WF[k, nddl - 1] * mat.dens * mat.sspnl * mat.le_max
                      * math.sqrt(4.0 / math.sqrt(3.0)) * thk)
            f = np.where(alive[:, None], f, v[:, :, k] * absorb[:, None])
        np.subtract.at(store.fnl, pos1 + k, f[:, 0])
        np.subtract.at(store.fnl, pos2 + k, f[:, 1])
        np.subtract.at(store.fnl, pos3 + k, f[:, 2])

    return dt_nl


def _btb_init(px2: np.ndarray, py2: np.ndarray, px3: np.ndarray,
              py3: np.ndarray) -> np.ndarray:
    """``cdkfint_reg_ini.F:285-297`` — the Starter pass's own ``BtB``.

    ``cdkfint_reg_ini.F`` recomputes its gradients from the corner coordinates
    via ``CDKEVEC3`` and uses the **opposite** node-1 sign convention to
    ``cdkfint_reg.F`` — here ``PX1 = PX2 + PX3`` where the engine cycle uses
    ``-PX3 - PX2`` — and the resulting operator is not the one
    ``cdkfint_reg.F`` builds.  Transcribed as written; it is a one-shot
    bootstrap, and the per-cycle operator is ``cdkfint_reg.F``'s.
    """
    p1x = px2 + px3
    p1y = -(py2 + py3)
    btb = np.zeros((len(px2), 3, 3))
    btb[:, 0, 0] = p1x ** 2 + p1y ** 2
    btb[:, 0, 1] = -p1x ** 2 + p1y * py2
    btb[:, 0, 2] = -p1y * (p1y + py2)
    btb[:, 1, 1] = p1x ** 2 + py2 ** 2
    btb[:, 1, 2] = -py2 * (p1y + py2)
    btb[:, 2, 2] = (p1y + py2) ** 2
    return btb


def nlocal_initialise(store: NonlocalShell, group, log=None,
                      tol1: float = 5.0e-4, tol2: float = 1.0e-8,
                      max_iter: int = 10000) -> int:
    """``nlocal_init_sta.F`` — iterate the non-local field to steady state.

    Upstream's initialisation is a *quasi-static relaxation loop*, not one
    sweep::

        FNL = VNL = DNL = UNL = ZERO ; TIME = ZERO
        DO WHILE (.NOT.CONVERGED)
          <every group's CDKFINT_REG_INI assembles FNL>
          FNL = FNL/MNL                 ! nlocal_acc.F
          VNL = VNL + DT_NL*FNL         ! nlocal_vel.F
          FNL = ZERO
          NORM_1 = ||DNL||
          DNL = DT_NL*VNL               ! nlocal_incr.F
          UNL = UNL + DNL
          CONVERGED = NORM < 5e-4 .AND. |NORM-NORM_1| < 1e-8 .AND. TIME /= 0
          TIME = TIME + DT_NL
        END DO
        FNL = VNL = DNL = ZERO          ! UNL is KEPT — it is the initial state

    The loop exists because ``CDKFINT_REG_INI`` calls ``CNLOC_MATINI``, which
    hands ``VAR_REG`` to the material and lets it generate plastic strain at
    increasing ``TIME``; the non-local field is the converged answer to that.
    This port's material laws do not yet consume a regularised variable, so on
    a virgin deck the loop converges on the second pass with ``NORM == 0`` —
    which is what upstream does too when the material produces no plastic
    strain.  The iteration, the tolerances and the reset are upstream's; the
    absent part is the material coupling, not the element side.

    ``max_iter`` is a backstop upstream does not carry (it exits on a NaN
    instead); a non-converging relaxation here would otherwise spin.
    """
    st = group.state
    if not st["nl_local"].any():
        return 0

    store.fnl[:] = 0.0
    store.vnl[:] = 0.0
    store.dnl[:] = 0.0
    store.unl[:] = 0.0
    time = 0.0
    dt_nl = 100.0
    norm_prev = 0.0
    converged = False
    it = 0
    while not converged and it < max_iter:
        it += 1
        dt_nl = min(dt_nl, initialise_group(store, group, st["nl_nddl"]))
        store.acc()
        store.vel(dt_nl)                 # VNL += DT_NL*FNL, then FNL = 0
        norm_prev = float(np.linalg.norm(store.dnl))
        store.incr(dt_nl)                # DNL = DT_NL*VNL ; UNL += DNL
        norm = float(np.linalg.norm(store.dnl))
        if math.isnan(norm):
            raise FloatingPointError(
                "non-local initialisation diverged (NLOC init, cdkfint_reg_ini.F)")
        converged = (norm < tol1) and abs(norm - norm_prev) < tol2 and time != 0.0
        time += dt_nl

    store.fnl[:] = 0.0                  # :171-174 — UNL is deliberately kept
    store.vnl[:] = 0.0
    store.dnl[:] = 0.0
    if log is not None:
        log.info(f"NON-LOCAL INITIALIZATION: {it} relaxation passes, "
                 f"|DNL| = {float(np.linalg.norm(store.dnl)):.3E}")
    return it


def configure(model, controls=None, noda=None) -> Optional[NonlocalShell]:
    """Publish the Engine's time-step controls onto the store.

    Upstream reads ``DTFAC1(1)``, ``DTFAC1(11)``, ``DTMIN1(11)`` and
    ``IDTMIN(11)`` from the ``SCR18`` common block *inside* ``cdkfint_reg.F``,
    and ``NODADT`` from ``SCR02``.  In this port they live on the Engine's
    control object, so the Engine hands them over once.  Returns the store
    (or ``None``), which is all the Engine needs to keep.
    """
    store = getattr(model, "nloc_dmg", None)
    if store is None:
        return None
    store.nodadt = int(noda is not None)
    store.dt_fac1 = getattr(controls, "dt_scale", 0.9) if controls else 0.9
    if noda is not None:
        store.dt_fac_noda = noda.dt_sca
        store.dt_min_noda = noda.dt_min
        store.idtn_min = 3 if noda.cst else 0
    return store
