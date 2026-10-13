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
#   engine/source/elements/shell/coque/cdefo3.F  (modified: transcribed to
#     NumPy, the three branches dispatched on one `kind` argument, the
#     IMPL_S>0 guard carried explicitly as `implicit=True` — for the
#     IHBE==4 branch it is NOT the same thing as dt == 0, and Z2 added
#     as an input)
#   engine/source/elements/shell/coque/cderi3.F   (added: the Z2 the
#     IHBE==4 branch consumes, l.95 / l.109)
#   engine/source/elements/shell/coque/cpxpy3.F   (added: the Z2 = ZERO
#     of the ISMSTR == 3 stress-based path, l.77 / l.88; and the PX/PY
#     operator pairs and AREA the coefficients are built from, l.75-82)
#   engine/source/elements/shell/coque/cdt3.F    (the DT1C condensation
#     length consumed as `dt`)

"""``cdefo3.F`` — the BT second-order rigid-rotation correction, one entry
per ENGINE formulation branch.

WHY THIS IS A SEPARATE MODULE
-----------------------------
``pyradioss/elements/shell_bt4.py`` is at its recorded line budget
(``tools/validation_data/size_budget.json``, 1913 lines) and may only
shrink, so the transcribed branches live here and
``shell_bt4._rot2_correction`` re-exports them. The exact-dt eigenvalue
helpers (``_bend_shear_omega2`` / ``_exact_dt_factor``) were exiled here
later for the same reason — likewise re-exported, verbatim.

THE UPSTREAM CONTRACT
---------------------
``cdefo3.F`` has three formulation branches, selected on the ENGINE flag
``IHBE``.  ``Ishell`` card 3 maps to engine ``IHBE = 2`` through
``GEO(171) = IHBE-1`` (``hm_read_prop01.F:306-315`` -> ``cgrtails.F:956``
``IPARG(23)`` -> ``cforc3.F``), so parity case c43
(``c43_E1000_Bending_BT_BT_type3_Sf_0``, ``Ishell = 3``) runs the
node-1-relative branch.  Card 4 maps to ``IHBE = 4`` unchanged (it is the
one card that is *not* ``card-1``), so parity case c45
(``c45_E1000_Bending_BT_BT_type4_Sf_0``, ``Ishell = 4``) runs the third:

  ``IF(IHBE<=1)THEN``            l.104-131  "general"              (kind 1)
  ``ELSEIF(IHBE==2.OR.IHBE==3)``  l.133-173  node-1-relative       (kind 3)
  ``ELSEIF(IHBE==4)THEN``        l.174-213  Z2 warp                (kind 4)

Kinds 1 and 3 fix the same frame-lag problem — the corotational frame is
evaluated on the END-of-step geometry while the velocities sit at
mid-step, so an element rotating rigidly at rate ``w`` about an in-plane
axis measures a spurious membrane stretching rate of ``+w^2 dt/2`` — in
DIFFERENT ways, and upstream's asymmetry is deliberate:

* kind 1 works on the DIFFERENCES of the out-of-plane diagonal
  velocities and **over-corrects 2x** (``TMP1A`` carries ``dt/4`` and
  ``(VZ13-VZ24)^2``, which on a stencil-exact roll is ``-w^2 dt``);
* kind 3 works on the out-of-plane membrane RATE
  ``EXZ = PX1*VZ13 + PX2*VZ24`` and is the **exact** cancellation
  (``EXZ2 = (EXZ/AREA)^2 * dt/2``).

M41 mirrored the kind-1 asymmetry on purpose; this module keeps all three
apart, because they are not interchangeable — a fact pinned by
``tests/test_p3_bt3_c43.py`` and ``tests/test_p3_bt4_c45.py``.

KIND 4 IS A DIFFERENT SHAPE OF ANSWER AGAIN
-------------------------------------------
Kinds 1 and 3 yield a coefficient field per component, which the caller
contracts with the membrane operators.  Kind 4 does not: its own header
calls it "velocity modification for the lurglass membrane" (l.179), and
it returns the four per-node in-plane velocity corrections, the warp term
``GZX*ZZ2`` alternating sign between the 1-3 and 2-4 diagonals and each
half contracting with the OPPOSITE operator pair (``PY`` for the x rows,
``PX`` for the y rows).  It also is the only branch the implicit guard
does not switch off: ``EXZZ2``/``EYZZ2`` carry no ``DT1V4``.
"""

from __future__ import annotations

import numpy as np

from .. import materials
from ..common.constants import EM20, SHEAR_FACTOR

#: cdefo3.F branch selector -> ``_rot2_correction`` ``kind``: 0 none,
#: 1 the ``IHBE <= 1`` general form, 3 the ``IHBE == 2 .OR. IHBE == 3``
#: node-1-relative velocity form, 4 the ``IHBE == 4`` Z2-warp form.
ROT2_KIND = {0: 0, 1: 1, 2: 3, 3: 3, 4: 4}


def rot2_correction(px, py, vx, vy, dt, kind=1, z2=None, *, implicit=False):
    """``cdefo3.F`` — the second-order rigid-rotation correction, one entry
    per ENGINE ``IHBE`` formulation branch, returning the pair of upstream
    correction coefficients.

    Parameters
    ----------
    px, py : (..., 2) arrays
        ``(PX1, PX2)`` and ``(PY1, PY2)`` — the AREA-scaled membrane
        operators, ``PX_i = AREA*B1_i`` and ``PY_i = AREA*B2_i``
        (``cpxpy3.F`` l.75-82 / ``cderi3.F`` l.173-176).
    vx, vy : (..., 2) arrays
        ``(VZ13, VZ24)`` — the OUT-OF-PLANE diagonal velocity differences
        (``cdefo3.F`` l.97-98) — and ``(VX13, VX24)``, the in-plane x
        diagonal pair the returned x coefficient acts on.
    dt : ``DT1C``
        the condensation length ``cdt3.F`` computes the element step for;
        ``ccoor3.F`` l.124 sets ``DT1C(I) = DT1``, the cycle step.
    kind : {0, 1, 3, 4}
        0 no correction, 1 the ``IHBE <= 1`` general form, 3 the
        ``IHBE == 2 .OR. IHBE == 3`` node-1-relative velocity form
        (parity case c43), 4 the ``IHBE == 4`` Z2-warp form (c45).
    z2 : (...) array, REQUIRED for ``kind == 4``
        ``Z2`` — the corotational out-of-plane coordinate of node 2
        relative to node 1, ``(x2-x1).e3`` (``cderi3.F`` l.95 and l.109;
        zeroed outright on the ``ISMSTR == 3`` stress-based path,
        ``cpxpy3.F`` l.77 and l.88).  It is an INPUT, not curvature state
        — ``ccurv3.F`` never produces it and ``cdefo3`` runs before it
        (``cforc3.F`` l.471 then l.476).
    implicit : bool
        ``IMPL_S > 0``, carried explicitly rather than folded into ``dt``,
        because the two are only the same thing on kinds 1 and 3 (see
        below).

    Returns
    -------
    (dx, dy) : (..., ) or (..., 4) arrays
        the upstream correction fields, in upstream's own naming:

        ==========  ==============================  ==================
        kind        ``dx``                          ``dy``
        ==========  ==============================  ==================
        0           0                               0
        1           ``TMP1A`` (l.111)               ``TMP2B`` (l.122)
        3           ``EXZ2``  (l.142)               ``EYZ2``  (l.148)
        4           per-node VX (l.187-190)         per-node VY (l.197-200)
        ==========  ==============================  ==================

    Kinds 1 and 3 return ONE coefficient per component, which the caller
    contracts with the membrane operators.  Kind 4 is not that shape: the
    branch IS a per-node modification of the four in-plane velocities
    (its own header says "velocity modification for the lurglass
    membrane", l.179), so it returns the four nodal corrections per
    component, in upstream's node order.

    THE IMPLICIT DISABLE IS NOT ``dt == 0`` ON KIND 4
    -------------------------------------------------
    ``IF(IMPL_S>0) DT1V4=ZERO`` (l.109, l.137, l.178) zeroes the
    condensation length's half.  On kinds 1 and 3 EVERY term carries
    ``DT1V4``, so ``dt == 0`` is exactly that guard.  On kind 4 it is not:
    ``EXZZ2 = GZX*ZZ2`` (l.183) and ``EYZZ2 = GZY*ZZ2`` (l.193) carry no
    ``DT1V4`` and SURVIVE it.  So ``implicit=True`` is modelled here as
    ``DT1V4 = 0`` — upstream's guard — and not as a zero return; folding
    it into ``dt`` would silently delete real upstream behaviour on the
    branch.

    WHY ``shell_bt4.forces`` DOES NOT CALL IT (Phase 3 Task P3.1)
    --------------------------------------------------------------
    The inline blocks in ``forces()`` implement exactly these closed
    forms, with the operands grouped the way upstream forms them —
    ``area*(B2_0+B2_1)`` for ``PY1+PY2``, ``np.sum(B1*VZ)`` for
    ``EXZ/AREA``.  Routing them through here would reassociate those into
    ``area*B2_0 + area*B2_1`` and ``(PX1*VZ13 + PX2*VZ24)/AREA``:
    mathematically identical, one different last bit.  On the c43 rolling
    benchmark that single bit is worth **34 % of the final internal
    energy** (measured: IE 1.93e5 -> 2.58e5 over 1185 steps — a 1605 ms
    free roll is chaos-dominated), so the refactor has to stay inert and
    this module is the transcribed, cited and unit-tested form the inline
    blocks are pinned against.
    """
    px1, px2 = px[..., 0], px[..., 1]
    py1, py2 = py[..., 0], py[..., 1]
    vz13, vz24 = vx[..., 0], vx[..., 1]

    if kind == 4:
        # cdefo3.F l.174-213, ELSEIF(IHBE==4)THEN.  DT1V4 = HALF*DT1C;
        # GZX = EXZ/AREA with EXZ = PX1*VZ13 + PX2*VZ24 (l.100), EYZ with
        # PY (l.99), AREA = 2*(PY2*PX1 - PY1*PX2) (cpxpy3.F l.82) — the
        # same EM20 floor kind 3 applies.  Unlike the other two kinds this
        # branch returns the per-node VELOCITY corrections, each half
        # contracting with the OPPOSITE operator pair (VX rows with PY,
        # VY rows with PX).  Handled before the dt == 0 shortcut on
        # purpose: the implicit guard here zeroes DT1V4 only, and the Z2
        # half does not carry it.
        if z2 is None:
            raise ValueError(
                "_rot2_correction(kind=4): Z2 is required.  It is the "
                "corotational out-of-plane coordinate of node 2 relative "
                "to node 1, (x2-x1).e3 (cderi3.F l.95, l.109) — zeroed on "
                "the ISMSTR == 3 stress-based path (cpxpy3.F l.77, l.88) "
                "— not a curvature from ccurv3.F, which never produces it.")
        area = np.maximum(2.0 * (py2 * px1 - py1 * px2), EM20)
        gzx = (px1 * vz13 + px2 * vz24) / area
        gzy = (py1 * vz13 + py2 * vz24) / area
        dt1v4 = 0.0 if implicit else 0.5 * dt
        zz2 = 0.5 * z2                                      # l.180
        exzz2 = gzx * zz2                                   # l.183
        eyzz2 = gzy * zz2                                   # l.193
        exz2 = gzx * gzx * dt1v4                            # l.184
        eyz2 = gzy * gzy * dt1v4                            # l.194
        exz2py2 = exz2 * py2                                # l.185
        exz2py1 = exz2 * py1                                # l.186
        eyz2px2 = eyz2 * px2                                # l.195
        eyz2px1 = eyz2 * px1                                # l.196
        return (np.stack([-exzz2 - exz2py2, exzz2 + exz2py1,
                          -exzz2 + exz2py2, exzz2 - exz2py1], axis=-1),
                np.stack([-eyzz2 + eyz2px2, eyzz2 - eyz2px1,
                          -eyzz2 - eyz2px2, eyzz2 + eyz2px1], axis=-1))

    if kind == 0 or dt == 0.0 or implicit:
        zero = np.zeros(np.shape(vz13), dtype=float)
        return zero, zero

    if kind == 1:
        # cdefo3.F l.107-126.  DT1V4 = FOURTH*DT1C; both denominators
        # carry the Fortran SIGN(MAX(ABS,EM20)) guard, so a zero
        # denominator is taken as +EM20 and a negative one flips sign.
        t3a = py1 + py2
        t3a = np.copysign(np.maximum(np.abs(t3a), EM20), t3a)
        dx = (0.25 * dt) * (vz13 - vz24) ** 2 / t3a
        t3b = px2 - px1
        t3b = np.copysign(np.maximum(np.abs(t3b), EM20), t3b)
        dy = (0.25 * dt) * (vz13 + vz24) ** 2 / t3b
        return dx, dy

    if kind == 3:
        # cdefo3.F l.136-150.  DT1V4 = HALF*DT1C; GZX = EXZ/AREA with
        # EXZ = PX1*VZ13 + PX2*VZ24 (l.100), EYZ = PY1*VZ13 + PY2*VZ24
        # (l.99), and AREA = 2*(PY2*PX1 - PY1*PX2) (cpxpy3.F l.82), the
        # floor being the same EM20 guard _local_geometry applies.
        area = np.maximum(2.0 * (py2 * px1 - py1 * px2), EM20)
        gzx = (px1 * vz13 + px2 * vz24) / area
        gzy = (py1 * vz13 + py2 * vz24) / area
        return (0.5 * dt) * gzx * gzx, (0.5 * dt) * gzy * gzy

    raise ValueError(f"_rot2_correction: unknown cdefo3 kind {kind!r}")


def resolve_ismstr(prop, def_shell, log):
    """Resolve one shell property's engine strain flag ISMSTR — the switch
    ``cforc3.F:430`` branches on (``ISMSTR == 3`` takes ``CPXPY3``, which
    sets ``Z2 = ZERO`` at ``cpxpy3.F:77`` and l.88).

    Rules are ``hm_read_prop01.F:203``, l.233-241 for ``/PROP/SHELL``: card
    ``Ismstr`` 0 takes the ``/DEF_SHELL`` default ``ISST_D`` (``-2``
    forces automatic ``-1`` — ``cgrtails.F`` then picks per material, a
    wider gap carried here as-is; it never yields 3, so the ``== 3`` gate
    below stays exact); leftover 0 becomes 2; ``ISMSTR == 3`` with the
    EFFECTIVE card ``Ishell`` (0 means "default", whose ``ISHELL``
    fallback is 1 — ``defaults_mod.F90:167``) outside ``{0, 2}`` becomes
    2 with the MSGID 319 warning. Only ``TYPE1`` is guarded — other
    property types carry raw or no ``Ismstr`` (their own starter rules
    are unported) and pass through the same 0-to-default-to-2 chain.
    Measured on the oracle starter: ``(4,3) -> 2`` + 319, ``(0,3) -> 2``
    + 319, ``(2,3) -> 3``, ``(4,0) -> 2``.
    """
    params = getattr(prop, "params", {}) or {}
    def_shell = def_shell or {}
    def_ism = def_shell.get("ismstr", 0)
    raw = int(params.get("ismstr", 0) or 0)
    if raw == 0:
        raw = def_ism
    if def_ism == -2:
        raw = -1
    if raw == 0:
        raw = 2
    if raw == 3 and getattr(prop, "type", 1) == 1:
        card = int(params.get("ishell", 0) or 0)
        eff = card if card != 0 else (def_shell.get("ishell", 0) or 1)
        if eff not in (0, 2):
            raw = 2
            log.warning(f"/PROP/SHELL/{prop.id}: Ismstr = 3 incompatible "
                        f"with Ishell = {eff}: ISMSTR set to 2 (MSGID 319)",
                        "SHELL INIT")
    return raw

# ----------------------------------------------------------------------------
# Starter dt-claim eigenvalue helpers (cdt3.F flavour)
# ----------------------------------------------------------------------------
# Exiled here by the shell_bt4.py line budget, for the same reason the
# transcription above lives here rather than inline in forces(): the exact
# per-element dt correction init_group claims on lc/c. Verbatim move —
# shell_bt4 re-exports both names, so shell_tri3's import keeps working.


def _bend_shear_omega2(B1, B2, area, sl, mat, t, nnode, rho) -> np.ndarray:
    """Exact max eigenfrequency^2 of the bending/shear branch of a FLAT
    one-point shell element (membrane decouples on a flat element).

    Local dofs (w_i, thx_i, thy_i); generalized rates (kxx, kyy, kxy,
    gx, gy) from the SAME operators as forces() — see the kinematics
    there; K = A * B^T C_b B with C_b = diag(t^3/12 * C_planestress,
    kappa*G*t * I2); lumped mass m = rho t A/nnode and the (deliberately
    generous) rotational inertia I = m (t^2 + A)/12 actually used by
    init_group. This branch GOVERNS the time step for thick/large
    elements where kappa*G*t*A outruns the membrane stiffness — assuming
    'membrane always binds' was an M1 bug fixed in M2 (a nu=0 cantilever
    strip at /DT 0.9 diverged on the shear-rotation mode)."""
    b1, b2, A = B1[sl], B2[sl], area[sl]
    n = len(A)
    B = np.zeros((n, 5, 3 * nnode))
    B[:, 0, 2 * nnode:] = b1                    # kxx =  B1 . thy
    B[:, 1, nnode:2 * nnode] = -b2              # kyy = -B2 . thx
    B[:, 2, 2 * nnode:] = b2                    # kxy = B2.thy - B1.thx
    B[:, 2, nnode:2 * nnode] = -b1
    B[:, 3, :nnode] = b1                        # gx = B1.w + mean(thy)
    B[:, 3, 2 * nnode:] = 1.0 / nnode
    B[:, 4, :nnode] = b2                        # gy = B2.w - mean(thx)
    B[:, 4, nnode:2 * nnode] = -1.0 / nnode
    Ep = mat.E / (1.0 - mat.nu ** 2)
    Cb = np.zeros((5, 5))
    Cb[:3, :3] = t ** 3 / 12.0 * np.array(
        [[Ep, mat.nu * Ep, 0.0], [mat.nu * Ep, Ep, 0.0], [0.0, 0.0, mat.G]])
    Cb[3, 3] = Cb[4, 4] = SHEAR_FACTOR * mat.G * t
    K = A[:, None, None] * np.einsum("nai,ab,nbj->nij", B, Cb, B)
    m = np.maximum(rho * t * A / nnode, EM20)   # nodal mass
    inertia = np.maximum(m * (t ** 2 + A) / 12.0, EM20)  # nodal inertia (init_group)
    minv = np.empty((n, 3 * nnode))
    minv[:, :nnode] = 1.0 / np.sqrt(m)[:, None]
    minv[:, nnode:] = np.repeat(1.0 / np.sqrt(inertia), 2 * nnode
                                ).reshape(n, 2 * nnode)
    Ksym = minv[:, :, None] * K * minv[:, None, :]
    return np.linalg.eigvalsh(Ksym)[:, -1]


def _exact_dt_factor(B1, B2, area, lc, thick, slices) -> np.ndarray:
    """Per-element ratio dt_exact/(lc/c) over BOTH stiffness branches of
    the flat one-point element:

    * membrane — the 2-D analogue of solid_hexa8._exact_dt_factor: the
      membrane stiffness is K = t*A * B^T C B with constant B, so the
      exact max frequency is the 3x3 eigenproblem
      omega^2 = (4/rho) eig(C_planestress . B B^T)  (lumped m = rho t A/4);
    * bending/transverse-shear — the 12-dof (w, thx, thy) eigenproblem of
      _bend_shear_omega2, which takes over for thick or large elements.

    dt_exact = 2 / max(omega) is a strict bound for the linearized
    element; forces() rescales it by the running lc/c."""
    n = len(area)
    Sxx = np.einsum("ni,ni->n", B1, B1)
    Syy = np.einsum("ni,ni->n", B2, B2)
    Sxy = np.einsum("ni,ni->n", B1, B2)
    BBt = np.zeros((n, 3, 3))
    BBt[:, 0, 0], BBt[:, 1, 1] = Sxx, Syy
    BBt[:, 2, 2] = Sxx + Syy
    BBt[:, 0, 2] = BBt[:, 2, 0] = Sxy
    BBt[:, 1, 2] = BBt[:, 2, 1] = Sxy
    fac = np.ones(n)
    for sl, mat, prop in slices:
        if getattr(mat, "law", 1) == 0 or not (getattr(mat, "rho0", 0.0) > 0.0 and getattr(mat, "E", 0.0) > 0.0):
            # stiffness-free / massless material (a /MAT/VOID skin shell —
            # legally RHO0 = 0 and E = 0, see starter/checks.
            # _NULL_RHO0_OK_LAWS): the element claims no time step at all
            # (upstream lc/SSP with SSP = 0), so the correction ratio is
            # moot — keep 1 instead of dividing by the null density.  The
            # exact twin of the guard solid_hexa8._exact_dt_factor already
            # applies for the same material (M39 / M38-NEW-2).
            fac[sl] = 1.0
            continue
        if getattr(mat, "law", 1) in (104, "104", "LAW104", 106, "106", "LAW106", 107, "107", "LAW107", 109, "109", "LAW109", 110, "110", "LAW110") or getattr(mat, "law_name", None) in ("104", "LAW104", "106", "LAW106", "107", "LAW107", "109", "LAW109", "110", "LAW110"):
            try:
                c = float(materials.sound_speed(mat, is_shell=True))
            except Exception:
                c = float(mat.sound_speed_shell()) if hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0 else 0.0
        elif getattr(mat, "law", 1) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
            from ..materials import law57_barlat
            c = law57_barlat.sound_speed_shell_law57(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
            from ..materials import law73_hill_therm
            c = law73_hill_therm.sound_speed(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000"):
            from ..materials import law87_barlat2000
            c = law87_barlat2000.sound_speed(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP"):
            from ..materials import law88_tab_hyp
            c = law88_tab_hyp.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE"):
            from ..materials import law92_arruda_boyce
            c = law92_arruda_boyce.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
            from ..materials import law93_orth_hill
            c = law93_orth_hill.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH"):
            from ..materials import law94_yeoh
            c = law94_yeoh.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0:
            c = mat.sound_speed_shell()
        else:
            c = 0.0
        if c <= EM20:
            fac[sl] = 1.0
            continue
        Ep = mat.E / (1.0 - mat.nu ** 2)
        C = np.array([[Ep, mat.nu * Ep, 0.0],
                      [mat.nu * Ep, Ep, 0.0],
                      [0.0, 0.0, mat.G]])
        eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
        w2max = (4.0 / mat.rho0) * eig.real.max(axis=1)
        w2bend = _bend_shear_omega2(B1, B2, area, sl, mat,
                                    getattr(prop, "params", {}).get("thick", getattr(prop, "thick", 0.001)), 4, mat.rho0)
        w2max = np.maximum(w2max, w2bend)
        dt_exact = 2.0 / np.sqrt(np.maximum(w2max, EM20))
        fac[sl] = np.minimum(dt_exact / np.maximum(lc[sl] / c, EM20), 1.0)
    return fac
