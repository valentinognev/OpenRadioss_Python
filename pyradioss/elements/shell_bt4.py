"""
4-node Belytschko–Tsay shell element (/SHELL + /PROP/SHELL, Ishell=1).

Fortran origin: ``engine/source/elements/shell/coque/`` — the cycle path is

    cforc3.F   driver (gather, frame, call chain, scatter)
    ccoor3.F   corotational frame + projection to local coordinates
    cdefo3.F   membrane velocity strains
    cdlen3.F   characteristic length / time step
    czforc3.F / cbilan.F  bending + shear rates, resultants, forces
    chour3.F   hourglass control
    + the plane-stress material calls sigeps..c.F per integration layer

Theory (Belytschko, Lin & Tsay, CMAME 42 (1984) 225-251; also BLM ch. 9):

* **Corotational frame**: a local orthonormal triad (e1, e2, e3) is built
  from the current geometry each cycle: e3 is normal to the element
  (cross product of the diagonals — insensitive to in-plane node
  numbering), e1 is side 1-2 projected onto the plane, e2 = e3 x e1.
  All rates are measured in this frame, which removes the large rigid
  rotation from the formulation — objective stress rates are then
  unnecessary for the (small) in-frame rotations.  This is the classic
  explicit-shell trick: accuracy O(element rotation per step), perfectly
  adequate at explicit time steps.

* **One-point quadrature** in the plane (element center), NIP-point
  Gauss quadrature through the thickness. The mid-plane gradient operator
  for a quad with local corner coords (x_i, y_i):

      B1 = [y2-y4, y3-y1, y4-y2, y1-y3] / (2A)
      B2 = [x4-x2, x1-x3, x2-x4, x3-x1] / (2A)

* **Mindlin-Reissner kinematics** (first-order shear deformable):
  velocity of a point at distance z from the mid-plane is
  v(z) = v_m + z * (theta_dot x e3), giving

      membrane rates   d_xx = B1.vx, d_yy = B2.vy,
                       d_xy = B1.vy + B2.vx           (engineering)
      curvature rates  k_xx = B1.thy, k_yy = -B2.thx,
                       k_xy = B2.thy - B1.thx
      shear rates      g_xz = B1.vz + mean(thy),
                       g_yz = B2.vz - mean(thx)

  strain rate at layer z:  d(z) = d_m + z * k. Each layer is updated by
  the plane-stress material law; transverse shear is elastic with the 5/6
  correction factor (BT assumption, matches the original for elastic
  shear).

* **Resultants** (force/length and moment/length):
      N = sum_k w_k sigma_k,  M = sum_k w_k z_k sigma_k,
      q = kappa * G * t * gamma
  and the internal nodal forces/moments follow from the virtual power
  identity  P = A (N:d_m + M:k + q.g)  — the exact transpose of the rate
  operators above (each B-term in a rate produces the matching force
  term; see the code, it is written line by line against the rates).

* **Hourglass control** (chvis3.F): one-point quadrature leaves 5 zero-
  energy modes (2 membrane, 1 transverse 'w', 2 bending) with the pattern
  h = (1,-1,1,-1). The stabilizing shape vector is orthogonalized against
  the linear field (Flanagan-Belytschko), gamma_i = h_i - (h.x) B1i -
  (h.y) B2i, so pure deformation and rigid motion produce no hourglass
  force — this matches chvis3.F's GAMA1..GAMA4 (lines 122-140) exactly.

  cforc3.F (lines 593-638) dispatches the control by Ishell (IHBE):
  IHBE == 2 -> chsti3.F (pure stiffness + plastic FMAX caps), every other
  Ishell -> **chvis3.F**, which is the /PROP/SHELL default and what this
  file implements. The engine pins HELAS = HVISC = 1/2 and **HVLIN = 0**
  (radioss2.F lines 638-640), so chvis3's LINEAR (sound-speed) viscous
  branch vanishes identically and each mode carries an ELASTIC stiffness
  plus a **QUADRATIC viscous damper**:

      SHFPR3   = SHF / (3 (1 + nu))                        (chvis3 l.143)
      (B1+B2)  = PX1^2 + PY1^2 + PX2^2 + PY2^2             (chvis3 l.174)

      elastic   hh1 = hm E t / 8                  modes 0,1 (membrane)
                hh2 = hf E SHFPR3 t^3 / (8 (B1+B2))  mode 2 (bending)
      viscous   h1q = (25/2) rho hm t sqrt(A)      modes 0,1
                h2q = (25/2) rho hf sqrt(SHFPR3) t^2  mode 2
                h3q = (25/2) 0.072169 rho hr t^2 A  modes 3,4

      Q += hh * (gamma . v) * dt                   (elastic, persistent)
      F  = Q + qd * hq * |qd|                      (+ quadratic damper)
      f_i = -F * gamma_i

  The rotation modes 3,4 are **purely viscous** (chvis3 lines 332-334
  ASSIGN rather than accumulate: no elastic branch at all) and their modal
  rate uses the RAW h = (1,-1,1,-1) pattern, not gamma (lines 327-330).

  Upstream stores the gradient operators AREA-scaled (cderi3.F l.172:
  PX1 = (Y2-Y4)/2, i.e. PX = A * B with B this file's operator), so
  (B1+B2)_upstream = A^2 * bb / 2 and the ELASTIC coefficients carry no
  area factor whatever. Before M39 this file used the LS-DYNA **BLT84**
  calibration (k_m = hm E t A bb / 8, k_w = hf kappa G t A bb / 8,
  k_r = hr E t^3 A bb / 192) — a different code's hourglass. That was
  wrong three ways: the membrane stiffness ran A*bb (~2) high, the
  transverse one ~3 (B1+B2)^2/(A t^2) high (~300x at box_beam's L/t = 10),
  and — the one that showed up in validation — it was purely ELASTIC, so
  it STORED the hourglass energy and dissipated none: the M36 box_beam
  hourglass energy came out at 0.04 % of the total against the Fortran's
  4.4 %. The quadratic dampers above are where that 4.4 % lives; the
  earlier reading that the port's hourglass stiffness was "~2 orders low"
  had the sign of the stiffness error backwards — it was the DISSIPATION
  that was missing, not the stiffness (M39).

  Triangles (a degenerate quad, node 3 == node 4) take no hourglass at all
  — chvis3.F lines 181-192 zero every coefficient. shell_tri3.py is fully
  integrated and has no hourglass block for the same reason.

* **Lumped inertia**: m_i = rho t A / 4; rotational inertia per
  cinmas.F's family split (M41) — I_i = m_i (A/FAC + t^2/12) with
  FAC = 9 for the BT family (engine IHBE < 11) and FAC = 12 for
  QBAT/QEPH/DKT18 (IHBE >= 11) — deliberately generous (Key's trick) to
  push the rotational stability limit up toward the membrane one. It
  does NOT always clear it: for thick or large elements the
  transverse-shear / rotation branch (stiffness ~ kappa G t A) still
  governs, which is why the Starter computes the exact eigenvalue of
  BOTH branches (see _exact_dt_factor — an M2 fix after a nu=0 strip
  diverged at /DT 0.9; its estimate keeps the FAC=12 inertia for every
  family — the SMALLER value, so the bound stays on the safe side).

M7 performance structure
------------------------
forces() is split around the Python layer/material loop into ``_pre``
(corotational frame, local geometry, rate kinematics — ccoor3/cdefo3)
and ``_post`` (resultant nodal forces, BLT84 hourglass, back-transform —
czforc3/chour3), each with an optional numba mirror in
``pyradioss.accel.jit_kernels`` (see the accel package docstring for the
architecture and the parity contract). The NumPy code here is the
reference. The M7 profiling pass fused the per-rate einsum calls into
ONE stacked matmul (all ten B·v dot products at once: Bt (n,2,4) @
V (n,4,5), with V the local [vx,vy,vz,thx,thy]), replaced np.cross /
np.linalg.norm with the bitwise-identical fastmath forms, and replaced
np.add.at with the bincount scatter — see common/fastmath.py.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from . import shell_ortho, shell_cdefo3 as _shell_cdefo3
from ..accel import get as accel_get
from ..common.constants import EM20, EP30, SHEAR_FACTOR
from ..common.fastmath import cross3, norm3, scatter_add3

# side-index helper for the characteristic length: side i = (i, i+1)
_NEXT = np.array([1, 2, 3, 0])

#: raw hourglass pattern h = (1,-1,1,-1) — chvis3.F applies it UNMODIFIED
#: to the rotation modes (lines 327-330: HG1 = RX1-RX2+RX3-RX4), unlike the
#: translation modes which ride the FB-orthogonalized gamma.
_HRAW = np.array([1.0, -1.0, 1.0, -1.0])

#: chvis3.F quadratic-viscous prefactor: R1 * HVISC = (100/4 rho) * (1/2)
#: -> 25/2 per unit rho (chvis3 lines 144-157, radioss2.F HVISC = HALF).
_HQ = 12.5

#: constant_mod.F ZEP072169 = ZEP07+TWOEM3+EM04+SIXEM5+NINEEM6 — the
#: rotational hourglass calibration of chvis3.F/chsti3.F.
_ZEP072169 = 0.072169

#: IMPLICIT-ONLY rotational-hourglass regularization fraction (tangent()).
#: chvis3's rotation modes are purely viscous, so they contribute no
#: stiffness; a tangent still needs the (1,-1,1,-1) thx/thy mode
#: constrained or the implicit matrix is singular. 1.0 keeps the historic
#: (pre-M39) BLT84 rotational stiffness, which the M8/M9 implicit shell
#: validations were built on — it is a conditioning device, not physics.
_HG_ROT_REG = 1.0


# ----------------------------------------------------------------------------
# geometry: corotational frame and local coordinates (ccoor3.F)
# ----------------------------------------------------------------------------

def _edofs(conn: np.ndarray) -> np.ndarray:
    """Return (n, 24) array of global scalar DOF slot ids."""
    n = len(conn)
    if n == 0:
        return np.empty((0, 24), dtype=np.int64)
    edofs = np.empty((n, 24), dtype=np.int64)
    for i in range(4):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return edofs


def _frame(xe: np.ndarray):
    """Build the corotational triad E = [e1|e2|e3] per element.

    xe: (n, 4, 3). Returns E (n, 3, 3) with COLUMNS e1, e2, e3.
    """
    n = len(xe)
    if n == 0:
        return np.empty((0, 3, 3))
    r31 = xe[:, 2] - xe[:, 0]
    r42 = xe[:, 3] - xe[:, 1]
    e3 = cross3(r31, r42)
    n3 = norm3(e3)
    degen3 = n3 <= EM20
    e3 = np.where(degen3[:, None], np.array([0.0, 0.0, 1.0]), e3 / np.maximum(n3, EM20)[:, None])

    s1 = xe[:, 1] - xe[:, 0]
    proj = np.einsum("nb,nb->n", s1, e3)
    e1 = s1 - proj[:, None] * e3
    n1 = norm3(e1)
    degen1 = n1 <= EM20
    if np.any(degen1):
        cand_x = np.array([1.0, 0.0, 0.0])
        cand1 = cand_x - np.einsum("nb,b->n", e3, cand_x)[:, None] * e3
        nc1 = norm3(cand1)
        use_x = nc1 > 0.1
        cand_y = np.array([0.0, 1.0, 0.0])
        cand2 = cand_y - np.einsum("nb,b->n", e3, cand_y)[:, None] * e3
        nc2 = norm3(cand2)
        cand1_norm = cand1 / np.maximum(nc1, EM20)[:, None]
        cand2_norm = cand2 / np.maximum(nc2, EM20)[:, None]
        fallback_e1 = np.where(use_x[:, None], cand1_norm, cand2_norm)
        e1 = np.where(degen1[:, None], fallback_e1, e1 / np.maximum(n1, EM20)[:, None])
    else:
        e1 /= np.maximum(n1, EM20)[:, None]

    e2 = cross3(e3, e1)
    return np.stack([e1, e2, e3], axis=2)


def _local_geometry(xe: np.ndarray):
    """Frame, local corner coordinates, area and gradient operators."""
    n = len(xe)
    if n == 0:
        return np.empty((0, 3, 3)), np.empty((0, 4, 3)), np.empty(0), np.empty((0, 4)), np.empty((0, 4))
    E = _frame(xe)
    center = xe.mean(axis=1)
    # local coords: xl[n,i,a] = (x_i - c) . e_a  — one stacked matmul
    xl = (xe - center[:, None, :]) @ E
    x, y = xl[:, :, 0], xl[:, :, 1]
    area = 0.5 * ((x[:, 2] - x[:, 0]) * (y[:, 3] - y[:, 1])
                  + (x[:, 1] - x[:, 3]) * (y[:, 2] - y[:, 0]))
    inv2A = 1.0 / np.maximum(np.abs(2.0 * area), EM20)
    B1 = np.empty((len(xe), 4))
    B1[:, 0] = y[:, 1] - y[:, 3]
    B1[:, 1] = y[:, 2] - y[:, 0]
    B1[:, 2] = y[:, 3] - y[:, 1]
    B1[:, 3] = y[:, 0] - y[:, 2]
    B1 *= inv2A[:, None]
    B2 = np.empty((len(xe), 4))
    B2[:, 0] = x[:, 3] - x[:, 1]
    B2[:, 1] = x[:, 0] - x[:, 2]
    B2[:, 2] = x[:, 1] - x[:, 3]
    B2[:, 3] = x[:, 2] - x[:, 0]
    B2 *= inv2A[:, None]
    bad = np.abs(2.0 * area) <= EM20
    if np.any(bad):
        B1[bad] = 0.0
        B2[bad] = 0.0
    return E, xl, area, B1, B2


def _char_length(xl: np.ndarray, area: np.ndarray) -> np.ndarray:
    """lc = A / longest side (cdlen3.F flavour) — all 4 sides at once."""
    d = xl[:, _NEXT, :2] - xl[:, :, :2]              # (n, 4, 2) side vectors
    lmax = (d[:, :, 0] ** 2 + d[:, :, 1] ** 2).max(axis=1)
    return np.maximum(area, 0.0) / np.maximum(np.sqrt(lmax), EM20)



#: card Ishell values whose ENGINE formulation flag is IHBE <= 1 — the
#: hm_read_prop01.F lines 302-315 double storage maps card -> GEO(171):
#: 0 -> 0, 1 -> 1, 2 -> 0, >= 3 except 4 -> card-1, 4 -> 4.  The BT
#: type-1 family (cards 0/1/2) therefore runs cdefo3.F's IHBE <= 1
#: membrane-rate branch, which carries the SECOND-ORDER rigid-rotation
#: correction of forces() (see there); type 3 (engine 2) runs the
#: node-1-relative branch and type 4 (engine 4) the Z2 warp one, both
#: transcribed in :mod:`shell_cdefo3` (kind 4 = Phase 3 Task P3.2).
_IHBE_LE1_CARDS = (0, 1, 2)
_rot2_correction = _shell_cdefo3.rot2_correction
_ROT2_KIND = _shell_cdefo3.ROT2_KIND



# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

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


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter cinit3/cmass3)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.zeros((0, 1, 3)),
            qshear=np.zeros((0, 2)),
            epsp=np.zeros((0, 1)),
            thick=np.zeros(0),
            area0=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            hgq=np.zeros((0, 5)),
            hgq_rot=np.zeros((0, 2)),
            zw=[],
            dtfac=np.zeros(0),
            off=np.zeros(0),
            dt_iner=np.zeros(0),
            ihbe_mask=np.zeros(0, dtype=int),
        )
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    xe = model.x0[group.conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SHELL {eid}: zero or negative area", "SHELL INIT")

    n = group.n
    thick = np.zeros(n)
    rho0 = np.zeros(n)
    nip_max = 1
    for sl, mat, prop in group.state["slices"]:
        p = getattr(prop, "params", {})
        t_val = p.get("thick", getattr(prop, "thick", 0.001))
        thick[sl] = t_val
        rho0[sl] = mat.rho0
        nip_val = int(p.get("nip", getattr(prop, "nip", 3)))
        nip_max = max(nip_max, nip_val)
    mass = rho0 * thick * area

    # Through-thickness Gauss stations per part slice: z_k in [-t/2, t/2],
    # weights scaled so sum(w_k) = t. Stored per slice (nip may differ).
    zw = []
    for sl, mat, prop in group.state["slices"]:
        p = getattr(prop, "params", {})
        nip = int(p.get("nip", getattr(prop, "nip", 3)))
        gp, gw = np.polynomial.legendre.leggauss(nip)
        zw.append((gp * 0.5, gw * 0.5))  # relative to thickness
    # dt-claim correction factor on lc/c (static, from initial geometry —
    # the established M2 convention): the exact BT eigenvalue bound.
    lc0 = _char_length(xl, area)
    dtfac = _exact_dt_factor(B1, B2, area, lc0, thick,
                             group.state["slices"])

    group.state.update(
        sig=np.zeros((n, nip_max, 3)),   # in-plane stress per layer
        qshear=np.zeros((n, 2)),         # transverse shear stress (elastic)
        epsp=np.zeros((n, nip_max)),     # plastic strain per layer
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        # persistent hourglass generalized forces (stiffness control):
        # columns = [membrane-x, membrane-y, transverse-w, theta-x, theta-y]
        hgq=np.zeros((n, 5)),
        # accumulated IMPLICIT rotation-hourglass modal displacement
        # (local theta-x, theta-y) — the static-stabilization state, the
        # rotation-mode analogue of solid_hexa8's hgq. chvis3's rotation
        # hourglass is purely viscous, so the implicit residual must supply
        # its own ELASTIC rotation stabilization to match tangent()'s k_r
        # regularization (see static_stabilization). Unused by the explicit
        # path (its rotation hourglass is the viscous damper of _post).
        hgq_rot=np.zeros((n, 2)),
        zw=zw,
        # exact stability correction to the lc/c estimate (see above)
        dtfac=dtfac,
    )
    _init_material_state(group, nip_max)
    # orthotropy fiber frame (/PROP/TYPE9 SH_ORTH, TYPE16): per-element
    # (cos, sin) of the fiber axis in the INITIAL corotational frame E,
    # frozen for the run (IREP==0) — None when no slice is orthotropic
    group.state["ortho"] = shell_ortho.build_group_ortho(
        group.state["slices"], E, n, log, group.ids)
    # element engine IHBE formulation flags for cdefo3.F kinematics branching
    ihbe_mask = np.zeros(n, dtype=int)
    for sl, mat, prop in group.state["slices"]:
        card = int(prop.params.get("ishell", 0))
        ihbe = {0: 0, 1: 1, 2: 0, 4: 4}.get(card, card - 1 if card > 0 else 0)
        ihbe_mask[sl] = ihbe
    group.state["ihbe_mask"] = ihbe_mask
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    # lumped rotational inertia — upstream's cinmas.F "INERTIES ELEMENTS /4"
    # (lines 916-924 + 1379-1387), which is FAMILY-dependent:
    #
    #     IF(IHBE>=11) FAC=TWELVE ELSE FAC=NINE     (engine-numbering IHBE)
    #     XI = EMS*(AREA/FAC + THK^2/12)            EMS = rho t A / 4
    #
    # i.e. the BT family (engine IHBE < 11: user cards 0..4 and 11) lumps
    # the AREA share as A/9 — 4/3 MORE than the QBAT/QEPH/DKT A/12 — and
    # the engine's chvis3.F nodal-stiffness claim mirrors exactly that
    # (STIR = STI*(THK02/12 + AREA/9), chvis3 'STIFFNESS - DT' block), so
    # dt_rot == dt_tra stays exact on element-lumped nodes for EVERY
    # family.  Before M41 this file lumped (t^2+A)/12 — the FAC=TWELVE
    # value — for ALL shells; measured on the RD-E-1000 c41 mini-roll rig
    # the FAC=9 value reproduces the Fortran starter's printed /RBODY
    # principal inertia EXACTLY (Ixx 4403.541 = sum of the members'
    # m/4*(A/9+t^2/12) — nodes collinear with the axis, so the m*d^2 term
    # vanishes and the print isolates the nodal lumping), and it moves the
    # BT-family /DT/NODA + /RBODY-transported dt floor 1.86540e-2 ->
    # 1.95667e-2 against the Fortran 2.004e-2 (the residual is the
    # chvis3.F STI/STIR claim formulas, still the documented M40 cut).
    # dt_iner: the per-NODE inertia share, kept for the ROTATIONAL
    # nodal-dt claim of /DT/NODA (M40, engine/mass_scaling.py) — the same
    # array the nodal inertia below is built from, so the claimed
    # rotational spring kr = 2 I/dt_e^2 mirrors upstream's STIR with the
    # factor that MATCHES this lumping: on a free element-lumped node the
    # rotational dt equals the translational one and never binds; it
    # bites through the /RBODY master transport.
    fac = np.full(n, 9.0)                    # cinmas.F FAC=NINE (BT, DKT18)
    for sl, mat, prop in group.state["slices"]:
        card = int(prop.params.get("ishell", 0))
        # hm_read_prop01.F 302-315 card -> engine IHBE: 0/2 -> 0, 1 -> 1,
        # 4 -> 4, else (3..99) -> card-1.  IHBE >= 11 <=> card >= 12.
        ihbe = {0: 0, 1: 1, 2: 0, 4: 4}.get(card,
                                            card - 1 if card >= 3 else 0)
        if ihbe >= 11:
            fac[sl] = 12.0                   # QBAT/QEPH/DKT18 keep A/12
    group.state["dt_iner"] = mass / 4.0 * (area / fac + thick ** 2 / 12.0)
    inertia_c = np.repeat(group.state["dt_iner"], 4)
    group._model = model
    return node_idx, mass_c, inertia_c


# ----------------------------------------------------------------------------
# M3 material/failure plumbing shared by both shell kernels
# ----------------------------------------------------------------------------

def _init_material_state(group, nip_max=None, n=None):
    """Allocate the per-layer material/failure state (see the materials
    and failure package docstrings):

    * ``off``      (n,)          1 alive / 0 deleted (GBUF%OFF)
    * ``layfail``  (n, nip)      1 intact / 0 broken, per layer — written
                                 by /FAIL criteria AND by layer-breaking
                                 laws (LAW27 rupture strain)
    * ``dama``     (n, nip)      /FAIL damage per layer (when needed)
    * ``mat_extra``{name: array} law-specific state (LAW27 crack memory)
    * ``chk_fail``               precomputed 'anything can delete here'
    """
    if hasattr(group, "state"):
        st = group.state
        if n is None:
            n = group.n
    else:
        st = group
        if n is None:
            n = len(st["layfail"]) if "layfail" in st else (len(st["off"]) if "off" in st else 1)
    if nip_max is None:
        if "layfail" in st and hasattr(st["layfail"], "shape") and len(st["layfail"].shape) > 1:
            nip_max = st["layfail"].shape[1]
        else:
            nip_max = 1
    if "off" not in st:
        st["off"] = np.ones(n)
    if "layfail" not in st:
        st["layfail"] = np.ones((n, nip_max))
    if "mat_extra" not in st or st["mat_extra"] is None:
        st["mat_extra"] = {}
    for sl, mat, prop in st["slices"]:
        for name, shape in materials.extra_shapes(mat, nip_max).items():
            if name not in st["mat_extra"]:
                if name.startswith("off") or name.startswith("damt") or name.startswith("alpe") or name.startswith("uvar82") or name.startswith("uvar_lam3"):
                    st["mat_extra"][name] = np.ones((n,) + shape)
                elif name.startswith("thk") and name != "thk43":
                    thk_arr = st.get("thick")
                    if thk_arr is None:
                        thk_arr = np.full(n, getattr(prop, "thick", 1.0))
                    st["mat_extra"][name] = np.ones((n,) + shape) * thk_arr.reshape((n,) + (1,) * len(shape))
                elif name == "temp":
                    t0_val = float(mat.params.get("t0", mat.params.get("T0", mat.params.get("T_i", 293.0)))) if hasattr(mat, "params") else 293.0
                    st["mat_extra"][name] = np.full((n,) + shape, t0_val)
                elif name in ("uvar", "uv69", "uvar69") and getattr(mat, "law", 1) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "HYP_EXT_COMP", "HYPER_EXT_COMP"):
                    arr = np.zeros((n,) + shape)
                    arr[..., 2] = 1.0
                    st["mat_extra"][name] = arr
                else:
                    st["mat_extra"][name] = np.zeros((n,) + shape)
    for sl, mat, prop in st["slices"]:
        if (getattr(mat, "law", 1) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL")
                or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL")):
            if "uvar73" not in st:
                st["uvar73"] = np.zeros((n, 7))
            if "uvar73" not in st["mat_extra"]:
                st["mat_extra"]["uvar73"] = np.zeros((n, nip_max, 7))
        if (getattr(mat, "law", 1) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000")
                or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000")):
            if "uvar87" not in st:
                st["uvar87"] = np.zeros((n, 7))
            if "uvar87" not in st["mat_extra"]:
                st["mat_extra"]["uvar87"] = np.zeros((n, nip_max, 7))
            if "sigb87" not in st["mat_extra"]:
                st["mat_extra"]["sigb87"] = np.zeros((n, nip_max, 12))
            if "thk87" not in st["mat_extra"]:
                thk_arr = st.get("thick")
                if thk_arr is None:
                    thk_arr = np.full(n, getattr(prop, "thick", 1.0))
                st["mat_extra"]["thk87"] = np.ones((n, nip_max)) * thk_arr.reshape((n, 1))
            if "pla87" not in st["mat_extra"]:
                st["mat_extra"]["pla87"] = np.zeros((n, nip_max))
            if "off87" not in st["mat_extra"]:
                st["mat_extra"]["off87"] = np.ones((n, nip_max))
        if (getattr(mat, "law", 1) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB")
                or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB")):
            if "uvar66" not in st:
                st["uvar66"] = np.zeros((n, 8))
            if "uvar66" not in st["mat_extra"]:
                st["mat_extra"]["uvar66"] = np.zeros((n, nip_max, 8))
        if (getattr(mat, "law", 1) in (88, "88", "LAW88", "HYP_TAB", "TAB_HYP", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TABULATED_HYP")
                or getattr(mat, "law_name", None) in ("88", "LAW88", "HYP_TAB", "TAB_HYP", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TABULATED_HYP", "MAT_LAW88", "MAT_HYP_TAB", "MAT_TAB_HYP", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC")):
            if "uvar88" not in st:
                st["uvar88"] = np.zeros((n, 30))
            if "uvar88" not in st["mat_extra"]:
                st["mat_extra"]["uvar88"] = np.zeros((n, nip_max, 30))
    if any(getattr(mat, "fail", None) is not None for _, mat, _ in st["slices"]):
        st["dama"] = np.zeros((n, nip_max))
    st["chk_fail"] = any(
        getattr(mat, "fail", None) is not None or getattr(mat, "law", 1) in (15, 22, 25, 27, 43, 48, 52, 57, 60, 66, 69, 73, 87)
        or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "73", "LAW73", "HILL_THERM", "THERM_HILL", "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000")
        or mat.params.get("eps_p_max", EP30) < 1e30
        or mat.params.get("eps_max", EP30) < 1e30
        or mat.params.get("EPSMAX", EP30) < 1e30
        or mat.params.get("tenscut", 1e30) < 1e30
        or mat.params.get("TENSCUT", 1e30) < 1e30
        for _, mat, _ in st["slices"])


def _layer_extra(st, sl, k, area=None):
    """The ``extra`` dict for one layer of one part slice: views into the
    law-specific arrays plus the shared layer-failure flags."""
    extra = {name: (arr[sl, k] if arr.ndim >= 2 else arr[sl]) for name, arr in st["mat_extra"].items()}
    extra["layfail"] = st["layfail"][sl, k]
    if "uvar73" in st and "uvar73" not in extra:
        extra["uvar73"] = st["uvar73"][sl]
    if "uvar73" in extra and "uvar" not in extra:
        extra["uvar"] = extra["uvar73"]
    if "uvar87" in st and "uvar87" not in extra:
        extra["uvar87"] = st["uvar87"][sl]
    if "uvar87" in extra and "uvar" not in extra:
        extra["uvar"] = extra["uvar87"]
    if "uvar66" in st and "uvar66" not in extra:
        extra["uvar66"] = st["uvar66"][sl]
    if "uvar66" in extra and "uvar" not in extra:
        extra["uvar"] = extra["uvar66"]
    if "uvar88" in st and "uvar88" not in extra:
        extra["uvar88"] = st["uvar88"][sl]
    if "uvar88" in extra and "uvar" not in extra:
        extra["uvar"] = extra["uvar88"]
    if "time" in st:
        extra["time"] = st["time"]
    if "thick" in st:
        if "thkn" not in extra:
            extra["thkn"] = st["thick"][sl]
        if "thk" not in extra:
            extra["thk"] = st["thick"][sl]
    if "thick0" in st:
        extra["thklyl"] = st["thick0"][sl]
    elif "thick" in st:
        extra["thklyl"] = st["thick"][sl]
    if area is not None:
        extra["area"] = area[sl]
    elif "area" in st:
        extra["area"] = st["area"][sl]
    if "vol" not in extra and "area" in extra and "thk" in extra:
        extra["vol"] = extra["area"] * extra["thk"]
    return extra


def _layer_failure(st, sl, mat, k, sig_k, epsp_old, deps_k, dt):
    """/FAIL damage + eps_p_max for one layer; breaks layers in place and
    zeroes their stress so the resultant integration never sees them."""
    layf = st["layfail"][sl, k]
    if mat.fail is not None:
        d_ep = st["epsp"][sl, k] - epsp_old[sl, k]
        tstar = None                 # /FAIL/JOHNSON D5 (M6): homologous
        if "temp" in st["mat_extra"] and "mT" in mat.params:
            tstar = np.clip(
                st["mat_extra"]["temp"][sl, k]
                / (mat.params["T_melt"] - mat.params["T_i"]), 0.0, 1.0)
        eps_tot = None
        if "eps_fld" in st["mat_extra"]:
            st["mat_extra"]["eps_fld"][sl, k] += deps_k
            eps_tot = st["mat_extra"]["eps_fld"][sl, k]
        broken = failure.shell_step(mat.fail, sig_k, d_ep, deps_k, dt,
                                    st["dama"][sl, k], tstar, eps_tot=eps_tot)
        layf[broken] = 0.0
    if getattr(mat, "law", 1) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "HYP_EXT_COMP", "HYPER_EXT_COMP"):
        tenscut = float(getattr(mat, "tenscut", 1e30) or (mat.params.get("tenscut", 1e30) if hasattr(mat, "params") else 1e30) or 1e30)
        if tenscut < 1e30:
            broken = (sig_k[:, 0] > tenscut) | (sig_k[:, 1] > tenscut)
            if np.any(broken):
                layf[broken] = 0.0
    law = getattr(mat, "law", 1)
    law_name = getattr(mat, "law_name", None)
    if (law in (52, "52", "LAW52", "GURSON", "PLAS_GURS") or law_name in ("52", "LAW52", "GURSON", "PLAS_GURS")) and "off52" in st["mat_extra"]:
        layf[st["mat_extra"]["off52"][sl, k] == 0.0] = 0.0
    elif (law in (57, "57", "LAW57", "BARLAT", "BARLAT3") or law_name in ("57", "LAW57", "BARLAT", "BARLAT3")) and "off57" in st["mat_extra"]:
        layf[st["mat_extra"]["off57"][sl, k] == 0.0] = 0.0
    elif (law in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL") or law_name in ("73", "LAW73", "HILL_THERM", "THERM_HILL")) and "off73" in st["mat_extra"]:
        layf[st["mat_extra"]["off73"][sl, k] <= 0.8] = 0.0
    elif (law in (87, "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or law_name in ("87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000")) and "off87" in st["mat_extra"]:
        layf[st["mat_extra"]["off87"][sl, k] <= 0.8] = 0.0
    elif "off" in st["mat_extra"] and (law in (52, "52", "LAW52", "GURSON", "PLAS_GURS", 57, "57", "LAW57", "BARLAT", "BARLAT3", 73, "73", "LAW73", "HILL_THERM", "THERM_HILL", 87, "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or law_name in ("52", "LAW52", "GURSON", "PLAS_GURS", "57", "LAW57", "BARLAT", "BARLAT3", "73", "LAW73", "HILL_THERM", "THERM_HILL", "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000")):
        layf[st["mat_extra"]["off"][sl, k] == 0.0] = 0.0
    if getattr(mat, "law", 1) != 43:
        eps_max = mat.params.get("eps_p_max", mat.params.get("eps_max", EP30))
        if eps_max < 1e30:
            layf[st["epsp"][sl, k] > eps_max] = 0.0
    sig_k[layf == 0.0] = 0.0


def _element_deletion(st, nip_of):
    """Element OFF from the layer flags, per part slice.

    Deletion rule: /FAIL's Ifail_sh (1 = one broken layer kills the
    element — the Radioss default, also used for the material eps_p_max
    thresholds; 2 = all layers; 3 = membrane criterion; 4 = no deletion),
    while LAW27 uses the all-layers rule of the original brittle law,
    and LAW25 uses the composite layer ratio/ioff criteria.
    Returns the updated alive mask."""
    off = st["off"]
    layfail = st["layfail"]
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        law = getattr(mat, "law", 1)
        if law == 0 or isl >= len(nip_of):
            continue
        if not (mat.fail is not None or law in (15, 22, 25, 27, 43, 48, 52, 57, 60, 66, 69, 73, 87)
                or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS", "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "73", "LAW73", "HILL_THERM", "THERM_HILL", "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000")
                or mat.params.get("eps_p_max", EP30) < 1e30
                or mat.params.get("eps_max", EP30) < 1e30
                or mat.params.get("EPSMAX", EP30) < 1e30
                or mat.params.get("tenscut", 1e30) < 1e30
                or mat.params.get("TENSCUT", 1e30) < 1e30):
            continue
        nip = nip_of[isl]
        if nip <= 0:
            continue
        nbroken = (layfail[sl, :nip] == 0.0).sum(axis=1)
        if law == 25:
            ratio = float(mat.params.get("ratio", 1.0))
            if ratio < 0:
                fail_npt = max(1, nip - 1)
            else:
                fail_npt = max(1, nip - round(nip * (1.0 - ratio)))
            ioff = int(mat.params.get("ioff", 0))
            if ioff == 0:
                dead = nbroken >= 1
            else:
                dead = nbroken >= fail_npt
        elif law == 15:
            ioff = int(mat.params.get("ioff", mat.params.get("itype", 0)))
            if ioff == 2 and nip > 1:
                dead = nbroken == nip
            else:
                dead = nbroken >= 1
        elif mat.fail is not None and getattr(mat.fail, "ifail_sh", 1) == 4:
            dead = np.zeros(len(off[sl]), dtype=bool)
        elif mat.fail is not None and getattr(mat.fail, "ifail_sh", 1) == 3:
            # Membrane criterion: mid-surface layer (or all layers)
            mid = nip // 2
            dead = layfail[sl, mid] == 0.0
        elif law in (27, 43) or (mat.fail is not None and getattr(mat.fail, "ifail_sh", 1) == 2):
            dead = nbroken == nip
        else:
            dead = nbroken >= 1
        off[sl][dead] = 0.0
    return off > 0.0


# ----------------------------------------------------------------------------
# Engine-side forces (cforc3.F)
# ----------------------------------------------------------------------------

def _pre(xe, ve, vre, off):
    """Corotational frame + local geometry + rate kinematics — the
    ccoor3/cdefo3(+czforc3 kinematics) part of the cycle, everything
    BEFORE the layer/material loop. Returns
    (E, area, lc, B1, B2, bb, gam, V, dm, kap, gs) where

    * ``V`` (n, 4, 5) are the LOCAL nodal rates [vx, vy, vz, thx, thy] —
      reused verbatim by the hourglass block of _post;
    * ``gam`` (n, 4) is the FB-orthogonalized hourglass shape vector;
    * ``bb`` = B1.B1 + B2.B2 feeds the per-slice hourglass stiffness.

    Mirrored by accel.jit_kernels.shell_pre (the M7 parity contract)."""
    n = len(xe)
    E, xl, area, B1, B2 = _local_geometry(xe)
    area = np.maximum(area, EM20)
    lc = _char_length(xl, area)

    # velocities in the corotational frame: one matmul for the three
    # translations, one for the rotations (only thx/thy are used)
    V = np.empty((n, 4, 5))
    V[:, :, :3] = ve @ E
    V[:, :, 3:] = (vre @ E)[:, :, :2]

    # ---- rate of deformation (cdefo3 + czforc3 kinematics) ----------------
    # ALL ten B.v dot products in one stacked matmul: Bt (n,2,4) @ V
    # (n,4,5) -> M[n, which-B, which-field]; the rate lines below then
    # mirror the kinematics table of the module docstring entry by entry.
    Bt = np.empty((n, 2, 4))
    Bt[:, 0, :] = B1
    Bt[:, 1, :] = B2
    M = Bt @ V
    dm = np.empty((n, 3))            # membrane rates [xx, yy, xy(eng)]
    dm[:, 0] = M[:, 0, 0]                              # B1 . vx
    dm[:, 1] = M[:, 1, 1]                              # B2 . vy
    dm[:, 2] = M[:, 0, 1] + M[:, 1, 0]                 # B1.vy + B2.vx
    kap = np.empty((n, 3))           # curvature rates
    kap[:, 0] = M[:, 0, 4]                             # B1 . thy
    kap[:, 1] = -M[:, 1, 3]                            # -B2 . thx
    kap[:, 2] = M[:, 1, 4] - M[:, 0, 3]                # B2.thy - B1.thx
    gs = np.empty((n, 2))            # transverse shear rates
    gs[:, 0] = M[:, 0, 2] + V[:, :, 4].mean(axis=1)    # B1.vz + mean(thy)
    gs[:, 1] = M[:, 1, 2] - V[:, :, 3].mean(axis=1)    # B2.vz - mean(thx)

    # hourglass geometry (chour3): the FB shape vector gamma and the
    # B1.B1+B2.B2 stiffness factor — pure geometry, computed here so the
    # post block never re-touches xl
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0] = 1.0
    gam[:, 1] = -1.0
    gam[:, 2] = 1.0
    gam[:, 3] = -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2
    bb = (np.einsum("ni,ni->n", B1, B1)
          + np.einsum("ni,ni->n", B2, B2))             # B1.B1 + B2.B2

    # deleted elements (GBUF%OFF = 0): freeze their state — no straining,
    # and downstream no resultants, hourglass force or time-step claim
    alive = off > 0.0
    if not alive.all():
        dm[~alive] = 0.0
        kap[~alive] = 0.0
        gs[~alive] = 0.0
    return E, area, lc, B1, B2, bb, gam, V, dm, kap, gs


def _post(E, area, B1, B2, gam, V, Nres, Mres, qres, Q,
          k_m, k_w, hqm, hqb, hqr, dt):
    """Resultants -> nodal forces/moments, chvis3.F hourglass and the
    back-transform to global axes — the czforc3/chvis3 part of the
    cycle, everything AFTER the layer loop. ``Q`` is the persistent
    ELASTIC hourglass state st["hgq"], updated IN PLACE; the five
    coefficients arrive pre-masked by ``alive``:

    * ``k_m``/``k_w`` — elastic stiffness of the membrane / bending modes
      (chvis3 HH1/HH2); the rotation modes have NO elastic branch;
    * ``hqm``/``hqb``/``hqr`` — quadratic viscous coefficients (H1Q/H2Q/
      H3Q), the dissipative branch.

    Returns (fg, mg, dehg): global nodal forces/moments (n,4,3) ready to
    scatter, and the hourglass energy increment (elastic stored + viscous
    dissipated, exactly chvis3's EHOU). Mirrored by
    accel.jit_kernels.shell_post."""
    n = len(area)

    # ---- internal nodal forces & moments (transpose of the rates) ---------
    # each line mirrors one line of the rate kinematics in _pre.
    f = np.empty((n, 4, 3))
    m = np.zeros((n, 4, 3))
    A_ = area[:, None]
    f[:, :, 0] = A_ * (B1 * Nres[:, 0:1] + B2 * Nres[:, 2:3])
    f[:, :, 1] = A_ * (B2 * Nres[:, 1:2] + B1 * Nres[:, 2:3])
    f[:, :, 2] = A_ * (B1 * qres[:, 0:1] + B2 * qres[:, 1:2])
    m[:, :, 0] = A_ * (-B2 * Mres[:, 1:2] - B1 * Mres[:, 2:3]
                       - 0.25 * qres[:, 1:2])
    m[:, :, 1] = A_ * (B1 * Mres[:, 0:1] + B2 * Mres[:, 2:3]
                       + 0.25 * qres[:, 0:1])

    # ---- hourglass control (chvis3.F — module doc) ------------------------
    # translation modes 0,1,2 ride the FB-orthogonalized gamma (chvis3
    # lines 277-284 / 308-313); the rotation modes 3,4 ride the RAW
    # h = (1,-1,1,-1) pattern (chvis3 lines 327-330 use RX1-RX2+RX3-RX4).
    qd = np.einsum("ni,nik->nk", gam, V)               # (n, 5)
    qd[:, 3:] = (V[:, 0, 3:] - V[:, 1, 3:]
                 + V[:, 2, 3:] - V[:, 3, 3:])          # raw h . (thx, thy)

    # elastic branch: persistent, modes 0-2 only (Q += HH*qd*dt).
    Q[:, 0] += k_m * qd[:, 0] * dt
    Q[:, 1] += k_m * qd[:, 1] * dt
    Q[:, 2] += k_w * qd[:, 2] * dt
    Q[:, 3:] = 0.0            # rotation carries no elastic hourglass state

    # total modal force = elastic + quadratic viscous damper qd*HQ*|qd|
    # (chvis3 lines 288/293/316/333-334). The rotation force is the damper
    # alone — chvis3 ASSIGNS HOUR(4..5) rather than accumulating them.
    F = np.empty((n, 5))
    F[:, 0] = Q[:, 0] + qd[:, 0] * hqm * np.abs(qd[:, 0])
    F[:, 1] = Q[:, 1] + qd[:, 1] * hqm * np.abs(qd[:, 1])
    F[:, 2] = Q[:, 2] + qd[:, 2] * hqb * np.abs(qd[:, 2])
    F[:, 3] = qd[:, 3] * hqr * np.abs(qd[:, 3])
    F[:, 4] = qd[:, 4] * hqr * np.abs(qd[:, 4])
    # hourglass energy: EHOU = dt * sum_modes F * qd (chvis3 l.298/321/335
    # -337). The elastic part swings both ways (stored), the viscous part
    # is sign-definite (dissipated) — their sum is the reported HE.
    dehg = (F * qd).sum(axis=1) * dt

    # total local force = -(internal) + hourglass, back to global frame
    fl = -f
    fl -= gam[:, :, None] * F[:, None, :3]
    ml = -m
    ml[:, :, 0] -= _HRAW * F[:, 3:4]
    ml[:, :, 1] -= _HRAW * F[:, 4:5]
    # back to global axes: fg[n,i,b] = sum_a fl[n,i,a] E[n,b,a]
    Et = E.transpose(0, 2, 1)
    fg = fl @ Et
    mg = ml @ Et
    return fg, mg, dehg


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0)

    xe = x[conn]
    thick = st["thick"]

    # Cycle 0 Courant step probe or evaluation without velocity
    if dt is None or dt <= 0.0 or v is None:
        E, xl, area, B1, B2 = _local_geometry(xe)
        lc = _char_length(xl, area)
        c = np.zeros(n)
        is_void = np.zeros(n, dtype=bool)
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) == 0:
                is_void[sl] = True
            if getattr(mat, "law", 1) in (104, "104", "LAW104", 106, "106", "LAW106", 107, "107", "LAW107", 109, "109", "LAW109", 110, "110", "LAW110") or getattr(mat, "law_name", None) in ("104", "LAW104", "106", "LAW106", "107", "LAW107", "109", "LAW109", "110", "LAW110"):
                try:
                    c[sl] = materials.sound_speed(mat, is_shell=True)
                except Exception:
                    c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0 else 0.0
            elif getattr(mat, "law", 1) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
                from ..materials import law52_gurson
                c[sl] = law52_gurson.sound_speed_shell_law52(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A"):
                from ..materials import law58_fabr_a
                c[sl] = law58_fabr_a.sound_speed_shell_law58(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
                from ..materials import law57_barlat
                c[sl] = law57_barlat.sound_speed_shell_law57(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
                from ..materials import law73_hill_therm
                c[sl] = law73_hill_therm.sound_speed(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000"):
                from ..materials import law87_barlat2000
                c[sl] = law87_barlat2000.sound_speed(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP"):
                from ..materials import law88_tab_hyp
                c[sl] = law88_tab_hyp.sound_speed_shell(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE"):
                from ..materials import law92_arruda_boyce
                c[sl] = law92_arruda_boyce.sound_speed_shell(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
                from ..materials import law93_orth_hill
                c[sl] = law93_orth_hill.sound_speed_shell(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH"):
                from ..materials import law94_yeoh
                c[sl] = law94_yeoh.sound_speed_shell(mat, getattr(mat, "rho0", None))
            elif getattr(mat, "law", 1) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
                c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0 else 0.0
            elif hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0:
                c[sl] = mat.sound_speed_shell()
            else:
                c[sl] = 0.0
        alive = st["off"] > 0.0
        dt_e = np.where(alive, st["dtfac"] * lc / np.maximum(c, EM20), EP30)
        dt_e = np.where(area <= EM20, EP30, dt_e)
        return np.where(is_void, EP30, dt_e)

    # ---- pre block: frame, geometry, rates (numba mirror when active) -----
    jit = accel_get("shell_pre")
    if jit is not None:
        E, area, lc, B1, B2, bb, gam, V, dm, kap, gs = jit(
            xe, v[conn], vr[conn], st["off"])
    else:
        E, area, lc, B1, B2, bb, gam, V, dm, kap, gs = _pre(
            xe, v[conn], vr[conn], st["off"])
    alive = st["off"] > 0.0

    # ---- second-order rigid-rotation membrane correction (cdefo3.F) --------
    # cdefo3.F lines 103-130, the IHBE <= 1 branch (BT type 1, see
    # _IHBE_LE1_CARDS): the corotational frame is evaluated on the END-of-
    # step geometry while the velocities sit at mid-step, so an element
    # rotating rigidly at rate w about an IN-PLANE axis measures a spurious
    # membrane stretching rate of O(w^2*dt) along the rotated direction —
    # the frame lags the velocity field by w*dt/2 and the out-of-plane
    # nodal velocity leaks into the in-plane rates.  Upstream compensates
    # with a quadratic term built from the DIAGONAL vz differences (the
    # out-of-plane rotation-rate measures):
    #
    #     DT1V4 = dt/4                                   (cdefo3 l.107)
    #     TMP1A = DT1V4*(VZ13-VZ24)^2/(PY1+PY2)          (l.109-111)
    #     VX13 -= TMP1A ; VX24 += TMP1A                  (l.114-115)
    #     TMP2B = DT1V4*(VZ13+VZ24)^2/(PX2-PX1)          (l.120-122)
    #     VY13 += TMP2B ; VY24 += TMP2B                  (l.125-126)
    #
    # (PX/PY are the AREA-scaled operators, PX_i = A*B1_i; both
    # denominators carry the Fortran SIGN(MAX(ABS,EM20)) guard.)  On a
    # stencil-exact rigid roll (x at n, v = (x^n - x^{n-1})/dt) the raw
    # frame-lag bias is d_xx = +w^2 dt/2 and this term adds EXACTLY
    # -w^2 dt: the type-1 branch OVERCORRECTS 2x, leaving -w^2 dt/2
    # (the IHBE==2/3 branch's velocity form IS the exact cancellation —
    # a real upstream family asymmetry, mirrored bit-for-bit; pinned in
    # closed form by tests/test_m41_bt_rotation.py).  Beyond the static
    # O(w^2*dt) bias, the term LINEARIZED about a steady roll is an
    # O(w*dt) skew coupling between the transverse-vibration rates and
    # the membrane rates: without it the port's rolled BT elements sat
    # in a NEGATIVE-DAMPING flutter growing exp(2.7e-3/cycle) from
    # round-off (RD-E-1000 c41, HE 1e-29 -> 2.4e5 by t=1050).  The M41
    # forensics established the flutter is REAL PHYSICS of the rolled
    # strip in BOTH engines — the Fortran c41 holds HE at 2e-2 at
    # t=1050 but its OWN blow-up follows at t~1100-1140 (HE 3.1e5 =
    # 76% of IE, printed ENERGY ERROR -40.5%), surviving to TSTOP only
    # because with no /STOP card upstream's energy-error stop threshold
    # is infinite (ecrit.F l.563, freform.F DEMXS=EP30) while the port
    # keeps a live 15% guard.  WITH this correction the port's flutter
    # onset sits AT/BELOW the Fortran engine's on the differential
    # mini-roll rig (onset-window growth fits 3.8-5.5 vs Fortran's
    # 4.6-6.5 dec/100ms; late-window exponents EQUAL at ~11-12 where
    # the PRE-fix port ran 18.5; HE ~100x lower at equal late time).
    # On c41 the IE tracking extends t~900 -> ~1000 ms (port IE at
    # t=1000 within 0.8% of Fortran vs 19% off before), HE 40x lower
    # at t=900 and 3+ decades lower by t=950-980, both-engine
    # max_rel_rms 0.2305 -> 0.1542 (c40 NO-CHANNELS -> 0.1892 with the
    # M41 guard startup conditioning, c43 0.2713 -> 0.2566, c45 0.6244
    # -> 0.1496 — the whole RD-E-1000 BT family).
    # Upstream kills the term for implicit (IMPL_S>0 -> DT1V4=0, l.108):
    # the same gate as the viscous-damper disable below.  Membrane rates
    # only — kap/gs are untouched, exactly as upstream.
    ihbe_mask = st["ihbe_mask"]
    rot2 = (ihbe_mask <= 1) * alive
    if rot2.any() and not st.get("_impl_static_hg"):
        vz13 = V[:, 0, 2] - V[:, 2, 2]
        vz24 = V[:, 1, 2] - V[:, 3, 2]
        t2a = area * (B2[:, 0] + B2[:, 1])               # PY1 + PY2
        t3a = np.copysign(np.maximum(np.abs(t2a), EM20), t2a)
        tmp1a = (0.25 * dt) * (vz13 - vz24) ** 2 / t3a * rot2
        t1b = area * (B1[:, 1] - B1[:, 0])               # PX2 - PX1
        t3b = np.copysign(np.maximum(np.abs(t1b), EM20), t1b)
        tmp2b = (0.25 * dt) * (vz13 + vz24) ** 2 / t3b * rot2
        dm[:, 0] += tmp1a * (B1[:, 1] - B1[:, 0])
        dm[:, 1] += tmp2b * (B2[:, 0] + B2[:, 1])
        dm[:, 2] += tmp1a * (B2[:, 1] - B2[:, 0]) + tmp2b * (B1[:, 0]
                                                             + B1[:, 1])

    # cdefo3.F IHBE == 2/3 and 4 branches: Z2 warping modifications to VX/VY/VZ
    # These branches correct membrane rates AND modify the nodal velocities passed
    # to the hourglass loop (chvis3.F), unlike the type-1 branch which only
    # touches the membrane rates.
    mask23 = ((ihbe_mask == 2) | (ihbe_mask == 3)) * alive
    mask4 = (ihbe_mask == 4) * alive
    if (mask23.any() or mask4.any()) and not st.get("_impl_static_hg"):
        _, xl, _, _, _ = _local_geometry(xe)

    if mask23.any() and not st.get("_impl_static_hg"):
        idx = mask23
        z2 = xl[idx, 1, 2] - xl[idx, 0, 2]
        gzx = np.sum(B1[idx] * V[idx, :, 2], axis=1)
        gzy = np.sum(B2[idx] * V[idx, :, 2], axis=1)
        exzz2 = gzx * z2
        eyzz2 = gzy * z2
        dt1v4 = 0.5 * dt
        exz2 = gzx * gzx * dt1v4
        eyz2 = gzy * gzy * dt1v4

        dm[idx, 0] -= exz2
        dm[idx, 1] -= eyz2

        zzz = np.zeros_like(exz2)
        ihbe2 = (ihbe_mask[idx] == 2)
        if ihbe2.any():
            zzz[ihbe2] = (exz2[ihbe2] + eyz2[ihbe2]) * z2[ihbe2]

        x_rel = xl[idx, :, 0] - xl[idx, 0:1, 0]
        y_rel = xl[idx, :, 1] - xl[idx, 0:1, 1]

        corr_x = np.zeros((np.count_nonzero(mask23), 4))
        corr_x[:, 1] = exzz2
        corr_x[:, 3] = exzz2
        corr_x -= exz2[:, None] * x_rel
        V[idx, :, 0] += corr_x

        corr_y = np.zeros((np.count_nonzero(mask23), 4))
        corr_y[:, 1] = eyzz2
        corr_y[:, 3] = eyzz2
        corr_y -= eyz2[:, None] * y_rel
        V[idx, :, 1] += corr_y

        corr_z = np.zeros((np.count_nonzero(mask23), 4))
        corr_z[:, 1] = -zzz
        corr_z[:, 3] = -zzz
        corr_z -= gzx[:, None] * x_rel + gzy[:, None] * y_rel
        V[idx, :, 2] += corr_z

    mask4 = (ihbe_mask == 4) * alive
    if mask4.any() and not st.get("_impl_static_hg"):
        idx = mask4
        z2 = xl[idx, 1, 2] - xl[idx, 0, 2]
        zz2 = 0.5 * z2
        gzx = np.sum(B1[idx] * V[idx, :, 2], axis=1)
        gzy = np.sum(B2[idx] * V[idx, :, 2], axis=1)
        exzz2 = gzx * zz2
        eyzz2 = gzy * zz2
        dt1v4 = 0.5 * dt
        exz2 = gzx * gzx * dt1v4
        eyz2 = gzy * gzy * dt1v4

        px1 = B1[idx, 0] * area[idx]
        px2 = B1[idx, 1] * area[idx]
        py1 = B2[idx, 0] * area[idx]
        py2 = B2[idx, 1] * area[idx]

        dm[idx, 0] -= exz2
        dm[idx, 1] -= eyz2

        corr_x = np.zeros((np.count_nonzero(mask4), 4))
        corr_x[:, 0] = -exzz2 - exz2 * py2
        corr_x[:, 2] = -exzz2 + exz2 * py2
        corr_x[:, 1] =  exzz2 + exz2 * py1
        corr_x[:, 3] =  exzz2 - exz2 * py1
        V[idx, :, 0] += corr_x

        corr_y = np.zeros((np.count_nonzero(mask4), 4))
        corr_y[:, 0] = -eyzz2 + eyz2 * px2
        corr_y[:, 2] = -eyzz2 - eyz2 * px2
        corr_y[:, 1] =  eyzz2 - eyz2 * px1
        corr_y[:, 3] =  eyzz2 + eyz2 * px1
        V[idx, :, 1] += corr_y

    # ---- layer stress updates + resultants ---------------------------------
    if hasattr(group, "_model") and hasattr(group._model, "t"):
        st["time"] = group._model.t
    sig = st["sig"]
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    Nres = np.zeros((n, 3))     # membrane force / length
    Mres = np.zeros((n, 3))     # moment / length
    de_layers = np.zeros(n)     # internal energy density accumulation
    c = np.zeros(n)
    nip_of = []
    ortho_all = st.get("ortho")                     # (n, 2) fiber cos/sin
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            nip_of.append(0)
            continue
        zrel, wrel = st["zw"][isl]
        nip_of.append(len(zrel))
        t_sl = thick[sl]
        # orthotropic slice: rotate the strain into the fiber frame before
        # the law and the stress back for the resultants (shell_ortho).
        # The stored sig is then in the FIBER frame (consistent with the
        # law's own frame state, e.g. LAW19's eps19/sigi19) — the energy
        # sig:deps is frame invariant either way.
        cs = ortho_all[sl] if (ortho_all is not None and getattr(
            prop, "type", 0) in shell_ortho.ORTHO_PROP_TYPES) else None
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl                     # layer position
            wk = wrel[k] * t_sl                     # layer weight (sums to t)
            deps = (dm[sl] + zk[:, None] * kap[sl]) * dt
            if cs is not None:
                deps = shell_ortho.rot_strain_e2m(deps, cs)   # elem -> fiber
            s_old = sig[sl, k, :].copy()
            s_new, ep_new = materials.shell_update(
                mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                _layer_extra(st, sl, k, area=area))
            if ep_new is not None:
                st["epsp"][sl, k] = ep_new
            if st["chk_fail"]:
                # /FAIL damage + eps_p_max: break layers, zero their stress
                # BEFORE they enter the resultants
                _layer_failure(st, sl, mat, k, s_new, epsp_old, deps, dt)
            sig[sl, k, :] = s_new
            s_mid = 0.5 * (s_old + s_new)
            de_layers[sl] += wk * np.einsum("nk,nk->n", s_mid, deps)
            s_res = shell_ortho.rot_stress_m2e(s_new, cs) \
                if cs is not None else s_new        # fiber -> elem
            Nres[sl] += wk[:, None] * s_res
            Mres[sl] += (wk * zk)[:, None] * s_res
        if getattr(mat, "law", 1) in (104, "104", "LAW104", 106, "106", "LAW106", 107, "107", "LAW107", 109, "109", "LAW109", 110, "110", "LAW110") or getattr(mat, "law_name", None) in ("104", "LAW104", "106", "LAW106", "107", "LAW107", "109", "LAW109", "110", "LAW110"):
            try:
                c[sl] = materials.sound_speed(mat, is_shell=True)
            except Exception:
                c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0 else 0.0
        elif getattr(mat, "law", 1) in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS") or getattr(mat, "law_name", None) in ("52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
            from ..materials import law52_gurson
            c[sl] = law52_gurson.sound_speed_shell_law52(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (58, "58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A") or getattr(mat, "law_name", None) in ("58", "LAW58", "FABR_A", "FABRIC_A", "MAT_LAW58", "MAT_FABR_A", "LAW58_FABR_A"):
            from ..materials import law58_fabr_a
            c[sl] = law58_fabr_a.sound_speed_shell_law58(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (57, "57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3") or getattr(mat, "law_name", None) in ("57", "LAW57", "BARLAT", "BARLAT3", "MAT_LAW57", "MAT_BARLAT", "MAT_BARLAT3", "LAW57_BARLAT", "LAW57_BARLAT3"):
            from ..materials import law57_barlat
            c[sl] = law57_barlat.sound_speed_shell_law57(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL") or getattr(mat, "law_name", None) in ("73", "LAW73", "HILL_THERM", "THERM_HILL", "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL", "LAW73_HILL_THERM", "LAW73_THERM_HILL"):
            from ..materials import law73_hill_therm
            c[sl] = law73_hill_therm.sound_speed(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000") or getattr(mat, "law_name", None) in ("87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000"):
            from ..materials import law87_barlat2000
            c[sl] = law87_barlat2000.sound_speed(mat, getattr(mat, "rho0", None))
            if "uvar87" in st and "uvar87" in st.get("mat_extra", {}):
                u87 = st["mat_extra"]["uvar87"]
                st["uvar87"][sl] = u87[sl, 0] if u87.ndim == 3 else u87[sl]
        elif getattr(mat, "law", 1) in (88, "88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP") or getattr(mat, "law_name", None) in ("88", "LAW88", "HYPER_ELAS", "TABULATED_HYPERELASTIC", "TAB_HYP", "TABULATED_HYP", "MAT_LAW88", "MAT_HYPER_ELAS", "MAT_TABULATED_HYPERELASTIC", "MAT_TAB_HYP"):
            from ..materials import law88_tab_hyp
            c[sl] = law88_tab_hyp.sound_speed_shell(mat, getattr(mat, "rho0", None))
            if "uvar88" in st.get("mat_extra", {}):
                u88 = st["mat_extra"]["uvar88"]
                if "uvar88" not in st:
                    st["uvar88"] = np.zeros((n, 30))
                st["uvar88"][sl] = u88[sl, 0] if u88.ndim == 3 else u88[sl]
        elif getattr(mat, "law", 1) in (92, "92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE") or getattr(mat, "law_name", None) in ("92", "LAW92", "ARRUDA_BOYCE", "ARRUDA-BOYCE", "MAT_LAW92", "MAT_ARRUDA_BOYCE"):
            from ..materials import law92_arruda_boyce
            c[sl] = law92_arruda_boyce.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (93, "93", "LAW93", "ORTH_HILL") or getattr(mat, "law_name", None) in ("93", "LAW93", "ORTH_HILL", "MAT_LAW93", "MAT_ORTH_HILL", "LAW93_ORTH_HILL"):
            from ..materials import law93_orth_hill
            c[sl] = law93_orth_hill.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (94, "94", "LAW94", "YEOH") or getattr(mat, "law_name", None) in ("94", "LAW94", "YEOH", "MAT_LAW94", "MAT_YEOH"):
            from ..materials import law94_yeoh
            c[sl] = law94_yeoh.sound_speed_shell(mat, getattr(mat, "rho0", None))
        elif getattr(mat, "law", 1) in (66, "66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB") or getattr(mat, "law_name", None) in ("66", "LAW66", "PLAS_TAB_COSSER", "PLAS_COSSER", "FOAM_TAB", "MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER", "MAT_FOAM_TAB"):
            c[sl] = mat.sound_speed_shell() if hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0 else 0.0
            if "uvar66" in st and "uvar66" in st.get("mat_extra", {}):
                st["uvar66"][sl] = st["mat_extra"]["uvar66"][sl, 0]
        elif hasattr(mat, "sound_speed_shell") and getattr(mat, "rho0", 0.0) > 0.0:
            c[sl] = mat.sound_speed_shell()
        else:
            c[sl] = 0.0
        # elastic transverse shear resultant stress (with 5/6 factor)
        qold = st["qshear"][sl].copy()
        g_val = getattr(mat, "G", 0.0) or getattr(mat, "g5", 0.0) or getattr(mat, "g0", 0.0)
        st["qshear"][sl] += SHEAR_FACTOR * g_val * gs[sl] * dt
        de_layers[sl] += t_sl * np.einsum(
            "nk,nk->n", 0.5 * (qold + st["qshear"][sl]), gs[sl] * dt)

    # ---- element deletion from the layer flags -----------------------------
    if st["chk_fail"]:
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            # a deleted element carries nothing: wipe this cycle's
            # resultants and every bit of persistent stress state
            Nres[dead] = 0.0
            Mres[dead] = 0.0
            sig[dead] = 0.0
            st["qshear"][dead] = 0.0
            st["hgq"][dead] = 0.0
            if "mat_extra" in st:
                for name in st["mat_extra"]:
                    if name.startswith("off"):
                        st["mat_extra"][name][dead] = 0.0
    qres = st["qshear"] * thick[:, None]            # shear force / length

    # ---- hourglass coefficients (chvis3.F — see the module docstring) ------
    # elastic HH1/HH2 and quadratic-viscous H1Q/H2Q/H3Q, with the engine's
    # HELAS = HVISC = 1/2, HVLIN = 0. (B1+B2) is upstream's AREA-scaled
    # PX1^2+PY1^2+PX2^2+PY2^2 = A^2 * bb / 2 (cderi3.F l.172).
    # Deleted elements exert no hourglass force (their Q was wiped above).
    k_m = np.zeros(n)      # HH1 elastic membrane stiffness  (modes 0,1)
    k_w = np.zeros(n)      # HH2 elastic bending  stiffness  (mode 2)
    hqm = np.zeros(n)      # H1Q quadratic viscous, membrane (modes 0,1)
    hqb = np.zeros(n)      # H2Q quadratic viscous, bending  (mode 2)
    hqr = np.zeros(n)      # H3Q quadratic viscous, rotation (modes 3,4)
    b12 = np.maximum(area ** 2 * bb * 0.5, EM20)       # (B1+B2) upstream
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        p = prop.params
        t_sl = thick[sl]
        rho = getattr(mat, "rho0", 0.0)
        E_mat = getattr(mat, "E", 0.0)
        nu = getattr(mat, "nu", 0.0)
        shfpr3 = SHEAR_FACTOR / (3.0 * (1.0 + nu))
        hm = p.get("hm", 0.01)
        hf = p.get("hf", 0.01)
        hr = p.get("hr", 0.01)
        k_m[sl] = hm * E_mat * t_sl / 8.0
        k_w[sl] = hf * E_mat * shfpr3 * t_sl ** 3 / (8.0 * b12[sl])
        hqm[sl] = _HQ * rho * hm * t_sl * np.sqrt(np.maximum(area[sl], 0.0))
        hqb[sl] = _HQ * rho * hf * np.sqrt(shfpr3) * t_sl ** 2
        hqr[sl] = _HQ * _ZEP072169 * rho * hr * t_sl ** 2 * area[sl]
    if st.get("_impl_static_hg"):
        # IMPLICIT residual (statics/dynamics): the quadratic viscous
        # hourglass damper qd*HQ*|qd| is a RATE device. The implicit driver
        # feeds the displacement INCREMENT as a pseudo-velocity at dt = 1
        # (see implicit/statics._internal_forces), so the O(v^2) damper
        # becomes a spurious O(u^2) static force that does NOT vanish at the
        # equilibrium displacement — it grows the residual after an otherwise
        # correct Newton step and the solve never converges. It is disabled
        # here exactly as the driver disables the solid bulk viscosity
        # (qa/qb): the ELASTIC chvis3 hourglass (HH1/HH2, linear in u) is the
        # genuine stabilization and stays, matched bit-for-bit by tangent().
        # The explicit engine never sets this flag (byte-identical path).
        hqm[:] = 0.0
        hqb[:] = 0.0
        hqr[:] = 0.0
    for _c in (k_m, k_w, hqm, hqb, hqr):
        _c *= alive

    # ---- post block: forces, hourglass, back-transform ---------------------
    jit = accel_get("shell_post")
    if jit is not None:
        fg, mg, dehg = jit(E, area, B1, B2, gam, V, Nres, Mres, qres,
                           st["hgq"], k_m, k_w, hqm, hqb, hqr, dt)
    else:
        fg, mg, dehg = _post(E, area, B1, B2, gam, V, Nres, Mres, qres,
                             st["hgq"], k_m, k_w, hqm, hqb, hqr, dt)

    st["ehour"] += dehg
    st["eint"] += area * de_layers

    # ---- scatter to global arrays (asspar) ---------------------------------
    flat = conn.reshape(-1)
    if fint is not None:
        scatter_add3(fint, flat, fg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    if mint is not None:
        scatter_add3(mint, flat, mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    # ---- critical time step ------------------------------------------------
    # deleted and void elements no longer constrain the global step
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dt_crit = np.where(alive & (~is_void), st["dtfac"] * lc / np.maximum(c, EM20), EP30)
    dt_crit = np.where(area <= EM20, EP30, dt_crit)
    return dt_crit


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M8) — a NEW entry point alongside forces()
# ----------------------------------------------------------------------------
# The Belytschko-Tsay tangent is assembled in the corotational LOCAL frame and
# rotated to global, exactly like the force path. Per node the element carries
# five LOCAL dofs — three translations (vx, vy, vz) and two bending rotations
# (thx, thy); there is no local drilling (thz) stiffness (the classic BT
# feature). The local element stiffness is the sum of four consistent blocks,
# each the exact linearization of the matching rate operator in _pre/_post:
#
#   membrane   K_m = A t     B_m^T C B_m         (N   = t   C  dm)
#   bending    K_b = A t^3/12 B_b^T C B_b        (M   = t^3/12 C kappa)
#   shear      K_s = A kG t   B_s^T B_s          (q   = kG t gamma_s)
#   hourglass  K_h = sum_modes k_mode g (x) g    (BLT84 stiffness control)
#
# with C the plane-stress membrane tangent (materials.shell_membrane_tangent,
# LAW1 in M8). The local dofs are then mapped to the six global dofs per node
# (three translations + three rotations) by the corotational frame E, and a
# small DRILLING stiffness about the local normal e3 is added so the global
# rotation block is non-singular (a flat shell gives no stiffness to rotation
# about its normal — the standard shell drilling-DOF penalty; the residual
# never loads it, so it stays zero and does not affect the solution).
#
# LAW2 shells (M11 — this removes the M8 deferral): the constant-C membrane/
# bending blocks above are the CLOSED-FORM thickness integration of an
# elastic layer stack (sum w_k = t, sum w_k z_k^2 = t^3/12 for the Gauss
# stations). With elastoplastic layers each station carries its OWN
# consistent plane-stress tangent D_k (materials.shell_layer_tangent — the
# algorithmic tangent of the Iplas=2 radial projection, derived in
# law02.consistent_shell_tangent), so the tangent integrates the SAME
# quadrature the force path uses for the resultants:
#
#   A_m = sum_k w_k D_k     B_m = sum_k w_k z_k D_k    D_m = sum_k w_k z_k^2 D_k
#   K   = A [ B_mem^T A_m B_mem + B_mem^T B_m B_bend + B_bend^T B_m B_mem
#             + B_bend^T D_m B_bend ] + shear + hourglass
#
# — the membrane/bending COUPLING block B_m switches on exactly when the
# stack yields asymmetrically through the thickness (a plastified outer
# fiber shifts the section's neutral surface), which the constant-C form
# cannot represent. LAW1 keeps the closed-form path bit-for-bit (M8
# contract). The transverse shear stays elastic (the force path integrates
# it elastically too) and the BLT84 hourglass keeps its elastic modulus (a
# stabilization, not a constitutive term). The geometric / initial-stress
# stiffness is the M9 addition — ``kgeo()`` below, added to this tangent by
# the assembler when /IMPL/NONLIN is active.

#: drilling-stiffness fraction of the bending stiffness (conditioning only —
#: the drilling DOF carries no load on the M8 validations, so the exact value
#: does not change results; small enough not to pollute a curved-shell answer).
_DRILL_COEF = 1.0e-3


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole shell group (LAW1 elastic
    closed-form; LAW2 per-layer consistent integration — see the note above).

    Returns ``(ke, edofs)``:

    * ``ke``    (n, 24, 24) dense element tangents over the 4 nodes x 6 global
      dofs (translations + rotations);
    * ``edofs`` (n, 24) global scalar DOF slot ids (node*6 + component) in the
      ``implicit.dofmap`` numbering — 0,1,2 = ux,uy,uz ; 3,4,5 = rx,ry,rz.

    ``epsp_incr`` (n, nip) is the increment's PER-LAYER plastic-strain step
    (None / zeros = all elastic), used by the LAW2 consistent layer
    tangents; LAW1 slices ignore it."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)

    xe = x[conn]

    E, xl, area, B1, B2 = _local_geometry(xe)
    area = np.maximum(area, EM20)
    thick = st["thick"]

    # ---- generalized strain-displacement operators in the LOCAL frame -----
    # local dof layout is COMPONENT-grouped: d = [vx(4), vy(4), vz(4),
    # thx(4), thy(4)] (20 dofs). Each B maps d -> the generalized strains,
    # written line-for-line against the rate kinematics of _pre.
    z = np.zeros((n, 4))
    Bm = np.zeros((n, 3, 20))            # membrane [dm_xx, dm_yy, dm_xy]
    Bm[:, 0, 0:4] = B1                                   # dm_xx = B1.vx
    Bm[:, 1, 4:8] = B2                                   # dm_yy = B2.vy
    Bm[:, 2, 0:4] = B2                                   # dm_xy = B2.vx +
    Bm[:, 2, 4:8] = B1                                   #         B1.vy
    Bb = np.zeros((n, 3, 20))            # curvature [k_xx, k_yy, k_xy]
    Bb[:, 0, 16:20] = B1                                 # k_xx = B1.thy
    Bb[:, 1, 12:16] = -B2                                # k_yy = -B2.thx
    Bb[:, 2, 12:16] = -B1                                # k_xy = B2.thy -
    Bb[:, 2, 16:20] = B2                                 #        B1.thx
    Bs = np.zeros((n, 2, 20))            # shear [g_x, g_y]
    Bs[:, 0, 8:12] = B1                                  # g_x = B1.vz +
    Bs[:, 0, 16:20] = 0.25                               #       mean(thy)
    Bs[:, 1, 8:12] = B2                                  # g_y = B2.vz -
    Bs[:, 1, 12:16] = -0.25                              #       mean(thx)

    # ---- constitutive blocks (membrane + bending + shear) -----------------
    Kl = np.zeros((n, 20, 20))
    kdrill = np.zeros(n)
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            continue
        t_sl = thick[sl]
        A_sl = area[sl]
        kGt = SHEAR_FACTOR * mat.G * t_sl                # transverse shear
        Bms, Bbs, Bss = Bm[sl], Bb[sl], Bs[sl]
        if mat.law == 1:
            # LAW1: closed-form thickness integration of the constant C —
            # the M8 elastic path, kept bit-for-bit
            C = materials.shell_membrane_tangent(mat)    # (3, 3) plane stress
            # membrane: A t B_m^T C B_m
            Kl[sl] += (A_sl * t_sl)[:, None, None] * np.einsum(
                "nai,ab,nbj->nij", Bms, C, Bms)
            # bending: A t^3/12 B_b^T C B_b
            Kl[sl] += (A_sl * t_sl ** 3 / 12.0)[:, None, None] * np.einsum(
                "nai,ab,nbj->nij", Bbs, C, Bbs)
        else:
            # M11 elastoplastic layers: per-layer consistent tangents D_k
            # integrated with the FORCE PATH's own quadrature (see the
            # module note — A_m/B_m/D_m thickness moments; the B_m coupling
            # block carries a plastified stack's neutral-surface shift)
            zrel, wrel = st["zw"][isl]
            m = sl.stop - sl.start
            Am_ = np.zeros((m, 3, 3))
            Bm_ = np.zeros((m, 3, 3))
            Dm_ = np.zeros((m, 3, 3))
            for k in range(len(zrel)):
                zk = zrel[k] * t_sl
                wk = wrel[k] * t_sl
                dep_k = None if epsp_incr is None else epsp_incr[sl, k]
                Dk = materials.shell_layer_tangent(
                    mat, st["sig"][sl, k, :], st["epsp"][sl, k], dep_k,
                    extra=_layer_extra(st, sl, k))
                Am_ += wk[:, None, None] * Dk
                Bm_ += (wk * zk)[:, None, None] * Dk
                Dm_ += (wk * zk * zk)[:, None, None] * Dk
            Kl[sl] += A_sl[:, None, None] * (
                np.einsum("nai,nab,nbj->nij", Bms, Am_, Bms)
                + np.einsum("nai,nab,nbj->nij", Bms, Bm_, Bbs)
                + np.einsum("nai,nab,nbj->nij", Bbs, Bm_, Bms)
                + np.einsum("nai,nab,nbj->nij", Bbs, Dm_, Bbs))
        # shear: A kGt B_s^T B_s (elastic, matching the force path)
        Kl[sl] += (A_sl * kGt)[:, None, None] * np.einsum(
            "nai,naj->nij", Bss, Bss)
        # drilling penalty scale (bending stiffness order, see module note)
        kdrill[sl] = _DRILL_COEF * mat.E * t_sl ** 3 * A_sl / 12.0

    # ---- hourglass stiffness (BLT84 stiffness control, same as _post) -----
    # gamma: FB-orthogonalized shape vector (identical to _pre's construction)
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0] = 1.0
    gam[:, 1] = -1.0
    gam[:, 2] = 1.0
    gam[:, 3] = -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2
    GG = np.einsum("ni,nj->nij", gam, gam)               # (n, 4, 4) Gram
    bb = (np.einsum("ni,ni->n", B1, B1)
          + np.einsum("ni,ni->n", B2, B2))
    # per-mode stiffness (identical formulas to forces()), fields ordered
    # [vx, vy, vz, thx, thy] -> [k_m, k_m, k_w, k_r, k_r]
    kfield = np.zeros((n, 5))
    b12 = np.maximum(area ** 2 * bb * 0.5, EM20)       # (B1+B2) upstream
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        p = prop.params
        t_sl = thick[sl]
        A_sl = area[sl]
        E_val = getattr(mat, "E", 0.0)
        nu_val = getattr(mat, "nu", 0.0)
        shfpr3 = SHEAR_FACTOR / (3.0 * (1.0 + nu_val))
        # membrane/bending: the ELASTIC chvis3 branch (HH1/HH2), the exact
        # linearization of the ELASTIC hourglass forces() integrates. The
        # quadratic viscous damper forces() also emits is a RATE device
        # (qd*HQ*|qd|, O(v^2)) that the implicit residual disables — so the
        # tangent carries the elastic stiffness alone (see the module note
        # on the implicit static hourglass, and forces()' _impl_static gate).
        k_m = p["hm"] * E_val * t_sl / 8.0
        k_w = p["hf"] * E_val * shfpr3 * t_sl ** 3 / (8.0 * b12[sl])
        # rotation: chvis3's rotational hourglass is PURELY VISCOUS, so it
        # contributes no stiffness and cannot appear in a tangent. Left
        # unconstrained the (1,-1,1,-1) thx/thy pattern is a zero-energy
        # mode of the whole local matrix (B1/B2 annihilate it) and the
        # implicit solve goes singular, so the tangent keeps an explicit
        # REGULARIZATION here of bending-stiffness order — the same role
        # _DRILL_COEF plays for the drilling DOF, and implicit-only (the
        # explicit path never sees it).
        k_r = _HG_ROT_REG * p["hr"] * E_val * t_sl ** 3 * A_sl * bb[sl] / 192.0
        kfield[sl, 0] = k_m
        kfield[sl, 1] = k_m
        kfield[sl, 2] = k_w
        kfield[sl, 3] = k_r
        kfield[sl, 4] = k_r
    # scatter k_field * GG onto each field's 4x4 block of the local matrix
    fi = np.arange(4)
    for f in range(5):
        rows = (f * 4 + fi)[:, None]
        cols = (f * 4 + fi)[None, :]
        Kl[:, rows, cols] += kfield[:, f, None, None] * GG

    # ---- local (20) -> global (24) transformation via the frame E ----------
    # local layout [vx(4),vy(4),vz(4),thx(4),thy(4)]; global node-major
    # [ux,uy,uz,rx,ry,rz] per node. local translation = E^T global_trans,
    # local (thx,thy) = (e1,e2) . global_rot.
    e1, e2, e3 = E[:, :, 0], E[:, :, 1], E[:, :, 2]      # (n, 3) each
    Tg = np.zeros((n, 20, 24))
    for i in range(4):
        c = np.arange(3)
        Tg[:, 0 * 4 + i, i * 6 + c] = e1[:, c]       # vx = e1.trans
        Tg[:, 1 * 4 + i, i * 6 + c] = e2[:, c]       # vy = e2.trans
        Tg[:, 2 * 4 + i, i * 6 + c] = e3[:, c]       # vz = e3.trans
        Tg[:, 3 * 4 + i, i * 6 + 3 + c] = e1[:, c]   # thx = e1.rot
        Tg[:, 4 * 4 + i, i * 6 + 3 + c] = e2[:, c]   # thy = e2.rot
    ke = np.einsum("nki,nkl,nlj->nij", Tg, Kl, Tg)       # (n, 24, 24)

    # ---- drilling penalty about the local normal e3 (global rot block) -----
    # add k_d (e3 (x) e3) to each node's rotation block so the drilling DOF
    # (rotation about the shell normal, unstiffened by BT) has a small,
    # non-singular stiffness. e3 (x) e3 restricts exactly that rotation.
    e3e3 = np.einsum("ni,nj->nij", e3, e3)               # (n, 3, 3)
    for i in range(4):
        r = i * 6 + 3
        ke[:, r:r + 3, r:r + 3] += kdrill[:, None, None] * e3e3

    # ---- zero dead elements -----------------------------------------------
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


# ----------------------------------------------------------------------------
# Implicit static hourglass stabilization — the residual counterpart of
# tangent()'s rotation-hourglass regularization (M39).
# ----------------------------------------------------------------------------
# chvis3's rotation-hourglass modes (theta-x, theta-y with the (1,-1,1,-1)
# pattern) are PURELY VISCOUS: the explicit _post emits them as the quadratic
# damper qd*H3Q*|qd|, which the implicit residual disables (a rate device fed a
# pseudo-velocity at dt = 1 — see forces()' _impl_static_hg gate). The membrane
# and transverse-w hourglass modes carry a chvis3 ELASTIC stiffness (HH1/HH2)
# that forces() emits linearly and tangent() matches bit-for-bit; the rotation
# modes have NO elastic branch. tangent() therefore stabilizes the rotation
# zero-energy mode with an explicit regularization k_r (of bending-stiffness
# order — else the implicit matrix is singular on the (1,-1,1,-1) theta pattern,
# and the modal/buckling eigensolvers would pick it up as a spurious near-zero
# frequency). That k_r has no counterpart in the (viscous-disabled) residual, so
# the two disagree and Newton stalls on the rotation DOFs. This restores the
# MATCHING elastic rotation-hourglass MOMENT to the implicit residual —
# exactly the consistent pre-M39 pairing, now split cleanly from the explicit
# viscous damper — mirroring solid_hexa8.static_stabilization for the solid
# stiffness hourglass. The stored modal displacement (hgq_rot) is persistent
# across committed increments (the driver's snapshot machinery commits/restores
# it with the stress), so a multi-increment run does not ratchet.

def _static_rot_hourglass(group, x, ur, mint):
    """Assemble the implicit elastic rotation-hourglass MOMENT into ``mint``,
    the residual counterpart of tangent()'s k_r regularization (see the note
    above). Shared by both implicit residual assemblers — static_stabilization
    (the M8 small-strain path) and static_internal_forces (the M9 nonlinear-
    geometry path) — so the rotation zero-energy mode is stabilized
    CONSISTENTLY with the tangent in either geometry mode. The persistent modal
    displacement ``hgq_rot`` is committed/restored with the stress by the
    driver's snapshot machinery, so a multi-increment run does not ratchet.
    ``x`` is the geometry the tangent is linearized at (x_ref for M8, the
    trial end configuration for NLGEOM); ``ur`` the rotation increment."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)
    thick = st["thick"]

    # FB-orthogonalized hourglass shape vector gam (identical to tangent()/_pre)
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0], gam[:, 1], gam[:, 2], gam[:, 3] = 1.0, -1.0, 1.0, -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2
    bb = (np.einsum("ni,ni->n", B1, B1)
          + np.einsum("ni,ni->n", B2, B2))

    # per-element rotation-hourglass stiffness k_r — the SAME formula as
    # tangent(), masked by the alive flag (deleted elements exert no moment)
    alive = st["off"] > 0.0
    k_r = np.zeros(n)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        p = prop.params
        t_sl = thick[sl]
        k_r[sl] = (_HG_ROT_REG * p["hr"] * getattr(mat, "E", 0.0) * t_sl ** 3
                   * area[sl] * bb[sl] / 192.0)
    k_r *= alive

    # local rotation increment (theta about e1, e2) and its modal amplitude
    # a = gam . theta_local (the same projection tangent()'s GG = gam (x) gam
    # linearizes; for a rectangle gam = (1,-1,1,-1), the raw _post pattern)
    Vr = (ur[conn] @ E)[:, :, :2]                      # (n, 4, 2) local thx,thy
    a = np.einsum("ni,nid->nd", gam, Vr)               # (n, 2) modal amplitude
    q0 = st["hgq_rot"]                                 # committed base
    F = k_r[:, None] * (q0 + a)                        # total elastic moment
    q0[...] = q0 + a                                   # trial state, in place

    # -gam (x) F back to global axes: moment about e1 <- F[:,0], e2 <- F[:,1]
    ml = np.zeros((n, 4, 3))
    ml[:, :, 0] = -gam * F[:, 0:1]
    ml[:, :, 1] = -gam * F[:, 1:2]
    mg = ml @ E.transpose(0, 2, 1)                     # local -> global
    scatter_add3(mint, conn.reshape(-1), mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))


def static_stabilization(group, x, u, ur, fint, mint):
    """Add the implicit elastic rotation-hourglass MOMENT to ``mint`` (the part
    of the SMALL-STRAIN static residual forces() does not supply, because
    chvis3's rotation hourglass is viscous — see the note above). Consistent
    with the k_r block of tangent(); ``u``/``fint`` unused (the membrane/
    transverse elastic hourglass forces() already emits and tangent() already
    matches). Called by implicit/statics._internal_forces after forces() on the
    M8 (nlgeom=False) path; the NLGEOM path folds the same term into
    static_internal_forces instead (both go through _static_rot_hourglass)."""
    _static_rot_hourglass(group, x, ur, mint)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the lumped mass/inertia is ``starter/source/elements/shell/
# coque/cmass3.F`` (the m/4 nodal mass and the generous rotary inertia
# ``init_group`` returns). The CONSISTENT mass is the shell shape-function
# integral M = ∫_A ρ (t Nᵀ_u N_u + t³/12 Nᵀ_θ N_θ) dA; ported for the M16 modal
# eigensolver alongside — never mutating — the lumped path.
#
# Theory (Cook, Malkus & Plesha ch. 11 — bilinear-quad consistent mass). The
# BT4's midsurface displacement and section rotation are interpolated with the
# SAME bilinear shape functions N_i in every global direction, so both blocks
# are ISOTROPIC (∝ I3) and the mass needs NO corotational rotation (unlike the
# stiffness): it is assembled directly in global node-major DOF order. Writing
# S_ij = ∫ N_i N_j dA (units of area),
#
#     translation block (i,j) = ρ t   S_ij I3     (membrane + transverse)
#     rotation    block (i,j) = ρ t³/12 S_ij I3   (bending rotary inertia)
#
# The rotation block uses the PHYSICAL section rotary inertia ρt³/12 — WITHOUT
# the lumped path's deliberate "+A" time-step boost (module docstring): that
# boost is a stability device that would wreck the natural frequencies, so the
# consistent mass drops it and applies the same isotropic ρt³/12 to all three
# rotation components (including the drilling DOF, so the reduced mass stays
# positive-definite for the eigensolver — the mass analogue of the drilling
# penalty stiffness). For a rectangular / parallelogram element the exact
# bilinear integral is the closed form
#
#     S = A/36 [[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]
#
# (used here); each translational row then sums to ρtA/4 = m/4 (the lumped
# nodal mass) so ½ vᵀMv = ½ m|v|² is exact for rigid v. A distorted quad
# carries the O(distortion) tributary-area error of this closed form — the same
# one-point-integration character as the element's stiffness; a full
# isoparametric 2×2 integration is the documented refinement (PORTING_GUIDE
# M16).

#: bilinear-quad ∫ Nᵀ N dA in units of the element area (parallelogram-exact).
_S_QUAD = np.array([[4.0, 2.0, 1.0, 2.0],
                    [2.0, 4.0, 2.0, 1.0],
                    [1.0, 2.0, 4.0, 2.0],
                    [2.0, 1.0, 2.0, 4.0]]) / 36.0


def consistent_mass(group, x=None):
    """Consistent element mass of the BT4 shell (see the note above):
    ρt S ⊗ I3 on the translations and ρt³/12 S ⊗ I3 on the rotations, S the
    bilinear-quad area integral. Built in global node-major DOF order (the
    blocks are isotropic, so no frame rotation is needed).

    Returns ``(me (n,24,24), edofs (n,24))``. ``x`` unused (mass conserved on
    the reference area, frame-invariant)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    mass = st["mass"]                                  # ρ t A, per element
    thick = st["thick"]
    m_trans = mass                                     # ρtA scalar
    m_rot = mass * thick ** 2 / 12.0                   # ρ (t³/12) A
    me = np.zeros((n, 24, 24))
    for a in range(4):
        for b in range(4):
            s = _S_QUAD[a, b]                          # dimensionless factor
            ft = m_trans * s                           # (n,)
            fr = m_rot * s
            for c in range(3):
                me[:, a * 6 + c, b * 6 + c] = ft        # translations
                me[:, a * 6 + 3 + c, b * 6 + 3 + c] = fr  # rotations
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        me[dead] = 0.0
    return me, _edofs(conn)


# ----------------------------------------------------------------------------
# Geometric (initial-stress) stiffness K_geo (M9)
# ----------------------------------------------------------------------------
# Fortran origin: the geometric-stiffness branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F`` + the shell KGEO routines — the
# ``imp_kgeo`` path of OpenRadioss's /IMPL/NONLIN large-displacement branch).
#
# Theory. For a shell the dominant initial-stress term carries the MEMBRANE
# force resultants N (force/length): linearizing the internal virtual work of
# the current membrane state against a transverse (or in-plane) perturbation
# of the midsurface gives, per element and in the local frame,
#
#     K_geo[a i, b j] = δ_ij A ( B1_a B1_b N_xx + B2_a B2_b N_yy
#                                + (B1_a B2_b + B2_a B1_b) N_xy )
#
# on the TRANSLATIONS — the classic von-Kármán initial-stress matrix built
# from the same one-point gradient operators B1, B2 the force path uses.
# It is δ_ij (isotropic over the translation components), so rotating the
# local block to global axes leaves it unchanged: it can be added directly to
# the global translation blocks (frame invariance of c_ab * I3). Compressive
# N makes it negative — the plate/column buckling driver; tensile N stiffens
# (stress stiffening / membranes). At zero stress it vanishes: the M8 path is
# untouched. The higher-order geometric couplings through the ROTATIONAL dofs
# (moment-resultant terms) are O(t/L) of the membrane term and are omitted —
# the standard BT shell buckling practice (they matter only for problems
# dominated by pre-stress moments, out of M9 scope, see PORTING_GUIDE).

def _membrane_resultants(st, thick):
    """Membrane force resultants N = sum_k w_k sigma_k (force/length, local
    [xx, yy, xy]) from the stored layer stresses — the same quadrature the
    force path applies (deleted elements have wiped stress => N = 0)."""
    sig = st["sig"]
    Nres = np.zeros((len(thick), 3))
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            continue
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            wk = wrel[k] * t_sl                 # layer weight (sums to t)
            Nres[sl] += wk[:, None] * sig[sl, k, :]
    return Nres


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the shell group,
    from the current layer stresses at geometry ``x``. Returns
    ``(ke, edofs)`` shaped exactly like ``tangent()`` (n, 24, 24)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)
    Nres = _membrane_resultants(st, st["thick"])

    # g_ab = A (B1a B1b Nxx + B2a B2b Nyy + (B1a B2b + B2a B1b) Nxy)  (n,4,4)
    g = area[:, None, None] * (
        Nres[:, 0, None, None] * B1[:, :, None] * B1[:, None, :]
        + Nres[:, 1, None, None] * B2[:, :, None] * B2[:, None, :]
        + Nres[:, 2, None, None] * (B1[:, :, None] * B2[:, None, :]
                                    + B2[:, :, None] * B1[:, None, :]))
    ke = np.zeros((n, 24, 24))
    ni = 6 * np.arange(4)
    for c in range(3):                     # delta_ij over the translations
        rows = (ni + c)[:, None]
        cols = (ni + c)[None, :]
        ke[:, rows, cols] += g

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0
    return ke, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal forces/moments at configuration ``x`` from the CURRENT
    resultant state — the updated-Lagrangian force assembly of the M9
    implicit residual (the stress/hourglass state was just updated by a
    ``forces()`` call at the MIDPOINT geometry; this re-states the czforc3
    force expressions on the END geometry, where equilibrium holds).

    Implementation: rebuild N/M from the stored layer stresses (the same
    quadrature the force path uses), q from the stored shear state, then
    call the existing ``_post`` with dt = 0 — at dt = 0 the hourglass state
    Q is NOT advanced (it already holds the trial values from the midpoint
    call) and _post reduces to exactly the resultant->nodal-force transpose
    plus the -gamma*Q hourglass push-back, all on this geometry."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    thick = st["thick"]
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)

    sig = st["sig"]
    Nres = np.zeros((n, 3))
    Mres = np.zeros((n, 3))
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if getattr(mat, "law", 1) == 0:
            continue
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl
            wk = wrel[k] * t_sl
            Nres[sl] += wk[:, None] * sig[sl, k, :]
            Mres[sl] += (wk * zk)[:, None] * sig[sl, k, :]
    qres = st["qshear"] * thick[:, None]

    # hourglass shape vector at THIS geometry (same construction as _pre)
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0], gam[:, 1], gam[:, 2], gam[:, 3] = 1.0, -1.0, 1.0, -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2

    # dt = 0 and zero rates: every coefficient drops out, so _post reduces
    # to the resultant->nodal-force transpose plus the -gamma*Q push-back of
    # the ELASTIC membrane/transverse hourglass state (chvis3's rotation
    # modes are viscous, so _post contributes nothing to them in a static
    # residual — see the module doc).
    zeros_n = np.zeros(n)
    fg, mg, _ = _post(E, area, B1, B2, gam, np.zeros((n, 4, 5)),
                      Nres, Mres, qres, st["hgq"],
                      zeros_n, zeros_n, zeros_n, zeros_n, zeros_n, 0.0)
    flat = conn.reshape(-1)
    if fint is not None:
        scatter_add3(fint, flat, fg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    if mint is not None:
        scatter_add3(mint, flat, mg.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
        # the elastic ROTATION-hourglass moment — the residual counterpart of
        # tangent()'s k_r regularization, on the END geometry (the M8 path adds
        # it via static_stabilization; NLGEOM folds it in here). Without it the
        # tangent's k_r has no residual match and the nonlinear-geometry Newton
        # stalls on the rotation DOFs exactly as the small-strain path did.
        _static_rot_hourglass(group, x, ur, mint)
