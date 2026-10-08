# Ported from $OR_SRC/engine/source/elements/xfem/cforc3_crk.F
# Subroutine: CFORC3_CRK (lines 66-641)
# Ported from $OR_SRC/engine/source/elements/xfem/czforc3_crk.F
# Subroutine: CZFORC3_CRK (lines 65-645)
# Ported from $OR_SRC/engine/source/elements/xfem/c3forc3_crk.F
# Subroutine: C3FORC3_CRK (lines 68-597)
# Ported from $OR_SRC/engine/source/elements/xfem/ccoor3_crk.F
# Subroutine: CCOOR3_CRK (lines 31-183)
# Ported from $OR_SRC/engine/source/elements/xfem/xfemfsky.F
# Subroutine: CUPDT3_CRK (lines 31-263), CUPDTN3_CRK (lines 272-380)
# Ported from $OR_SRC/engine/source/elements/xfem/crk_velocity.F
# Subroutine: CRK_VELOCITY (lines 30-127)
"""XFEM (Extended Finite Element Method) Enriched Shell Element Force Integration.

This module implements the phantom-node XFEM shell element formulation used in OpenRadioss:
1. Phantom Kinematics Gathering (ccoor3_crk.F, crk_velocity.F):
   - For uncracked elements, standard nodal coordinates and velocities are used.
   - For cracked elements, each phantom element component gathers coordinates and velocities
     from standard nodes (if un-enriched on that side, ENR <= 0) or from dedicated enriched
     phantom node DOFs (if enriched, ENR > 0).
2. Area-Weighted Sub-Element Force Integration (cforc3_crk.F, czforc3_crk.F):
   - Mindlin-Reissner kinematics and corotational shell formulation evaluated on phantom components.
   - Internal nodal forces and moments scaled by phantom area fraction w_k = A_k / A (sum w_k = 1.0).
3. Force Scatter to Standard and Enriched DOFs (xfemfsky.F, CUPDT3_CRK):
   - Un-enriched node contributions accumulate into the global standard mesh force array fint/mint.
   - Enriched phantom node contributions accumulate into the enriched DOF force array fenr/menr.
4. Energy Accounting (cbilan.F, cforc3_crk.F):
   - Load-bearing energy balance booking internal strain energy work into EINT and hourglass dissipation into EHG:
     dE_int = sum_k w_k * dE_int^(k)
     dE_hg  = sum_k w_k * dE_hg^(k)
     Work done by standard and enriched forces exactly balances internal energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM20, EP30, SHEAR_FACTOR
from ..common.fastmath import cross3, norm3, scatter_add3
from .xfem_crack import EnrichmentDOFManager, XfemElementCut


# ---------------------------------------------------------------------------
# Corotational Frame and Local Kinematics (shared with shell_bt4)
# ---------------------------------------------------------------------------

def _build_frame(xe: np.ndarray) -> np.ndarray:
    """Build orthonormal corotational triad E = [e1, e2, e3] for a quad (4, 3).

    Upstream: cderi3.F, ccoor3.F.
    """
    r31 = xe[2] - xe[0]
    r42 = xe[3] - xe[1]
    e3 = np.cross(r31, r42)
    n3 = float(np.linalg.norm(e3))
    if n3 > EM20:
        e3 /= n3
    else:
        e3 = np.array([0.0, 0.0, 1.0])

    s1 = xe[1] - xe[0]
    proj = float(np.dot(s1, e3))
    e1 = s1 - proj * e3
    n1 = float(np.linalg.norm(e1))
    if n1 > EM20:
        e1 /= n1
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(e3, ref)
        e1 /= max(float(np.linalg.norm(e1)), EM20)

    e2 = np.cross(e3, e1)
    e2 /= max(float(np.linalg.norm(e2)), EM20)
    return np.column_stack([e1, e2, e3])


def _quad_local_geometry(xe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute local coords, element area, and gradient operators B1, B2 for quad (4, 3).

    Upstream: cderi3.F lines 170-205.
      B1 = [y2-y4, y3-y1, y4-y2, y1-y3] / (2A)
      B2 = [x4-x2, x1-x3, x2-x4, x3-x1] / (2A)
    """
    E = _build_frame(xe)
    center = np.mean(xe, axis=0)
    xl = (xe - center) @ E
    x, y = xl[:, 0], xl[:, 1]

    area = 0.5 * ((x[2] - x[0]) * (y[3] - y[1]) + (x[1] - x[3]) * (y[2] - y[0]))
    inv2a = 1.0 / max(abs(2.0 * area), EM20)

    b1 = np.array([y[1] - y[3], y[2] - y[0], y[3] - y[1], y[0] - y[2]]) * inv2a
    b2 = np.array([x[3] - x[1], x[0] - x[2], x[1] - x[3], x[2] - x[0]]) * inv2a
    return E, xl, abs(area), b1, b2


# ---------------------------------------------------------------------------
# 1. Phantom Kinematics Gathering (ccoor3_crk.F lines 31-183, crk_velocity.F)
# ---------------------------------------------------------------------------

def gather_phantom_kinematics(
    xe_std: np.ndarray,
    ve_std: np.ndarray,
    vre_std: np.ndarray,
    enr_node_ids: Sequence[int],
    x_enr_dict: Dict[int, np.ndarray],
    v_enr_dict: Dict[int, np.ndarray],
    vr_enr_dict: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather nodal coordinates and velocities for one phantom element component.

    Ported from OpenRadioss subroutine CCOOR3_CRK:
      ``$OR_SRC/engine/source/elements/xfem/ccoor3_crk.F``
      lines 78-134.
    And CRK_VELOCITY:
      ``$OR_SRC/engine/source/elements/xfem/crk_velocity.F``
      lines 70-120.

    Parameters
    ----------
    xe_std : ndarray of shape (4, 3)
        Standard mesh coordinates of the 4 element nodes.
    ve_std : ndarray of shape (4, 3)
        Standard translational velocities of the 4 element nodes.
    vre_std : ndarray of shape (4, 3)
        Standard rotational velocities of the 4 element nodes.
    enr_node_ids : sequence of 4 ints
        Enrichment status per node for this phantom component:
          <= 0: node is tied to standard mesh (physical domain).
          >  0: node has an independent enriched phantom DOF (ghost domain).
    x_enr_dict : dict
        Mapping enriched_id -> ndarray of shape (3,) storing phantom coordinates.
    v_enr_dict : dict
        Mapping enriched_id -> ndarray of shape (3,) storing phantom translational velocities.
    vr_enr_dict : dict
        Mapping enriched_id -> ndarray of shape (3,) storing phantom rotational velocities.

    Returns
    -------
    xe_phantom : ndarray of shape (4, 3)
        Nodal coordinates for this phantom component.
    ve_phantom : ndarray of shape (4, 3)
        Translational velocities for this phantom component.
    vre_phantom : ndarray of shape (4, 3)
        Rotational velocities for this phantom component.
    """
    xe_phantom = xe_std.copy()
    ve_phantom = ve_std.copy()
    vre_phantom = vre_std.copy()

    for i in range(4):
        enr_id = enr_node_ids[i]
        if enr_id > 0:
            # Free phantom node DOF (ccoor3_crk.F lines 88-134)
            if enr_id in x_enr_dict:
                xe_phantom[i] = x_enr_dict[enr_id].copy()
            if enr_id in v_enr_dict:
                ve_phantom[i] = v_enr_dict[enr_id].copy()
            if enr_id in vr_enr_dict:
                vre_phantom[i] = vr_enr_dict[enr_id].copy()

    return xe_phantom, ve_phantom, vre_phantom


# ---------------------------------------------------------------------------
# 2. Force Scatter to Standard and Enriched DOFs (xfemfsky.F / CUPDT3_CRK)
# ---------------------------------------------------------------------------

def scatter_phantom_forces(
    fg_phantom: np.ndarray,
    mg_phantom: np.ndarray,
    area_weight: float,
    enr_node_ids: Sequence[int],
    f_std: np.ndarray,
    m_std: np.ndarray,
    f_enr_dict: Dict[int, np.ndarray],
    m_enr_dict: Dict[int, np.ndarray],
) -> None:
    """Scatter area-scaled phantom element forces and moments into standard and enriched arrays.

    Ported from OpenRadioss subroutine CUPDT3_CRK:
      ``$OR_SRC/engine/source/elements/xfem/xfemfsky.F``
      lines 123-260:
        AREAP = CRKLVSET(ILEV)%AREA(ELCRK)
        CRKSKY%FSKY(1..3) = -F(1..3) * AREAP
        CRKSKY%FSKY(4..6) = -M(1..3) * AREAP
        IF (ENR <= 0) THEN
          FSKY = FSKY + CRKSKY%FSKY  ! Standard node accumulation
        ELSE
          ! Remains on CRKSKY (enriched phantom node DOF)
        ENDIF

    Parameters
    ----------
    fg_phantom : ndarray of shape (4, 3)
        Global translational forces on the 4 nodes of the phantom element.
    mg_phantom : ndarray of shape (4, 3)
        Global moments on the 4 nodes of the phantom element.
    area_weight : float
        Normalized area fraction w_k = A_k / A.
    enr_node_ids : sequence of 4 ints
        Enrichment tag per node: <= 0 for standard node, > 0 for enriched phantom node.
    f_std : ndarray of shape (4, 3)
        Standard element nodal translational force accumulator.
    m_std : ndarray of shape (4, 3)
        Standard element nodal moment accumulator.
    f_enr_dict : dict
        Dictionary mapping enriched_id -> accumulated translational force (3,).
    m_enr_dict : dict
        Dictionary mapping enriched_id -> accumulated moment (3,).
    """
    fg_scaled = fg_phantom * area_weight
    mg_scaled = mg_phantom * area_weight

    for i in range(4):
        enr_id = enr_node_ids[i]
        if enr_id <= 0:
            # Physical side: accumulate into standard mesh nodes (lines 178-259)
            f_std[i] += fg_scaled[i]
            m_std[i] += mg_scaled[i]
        else:
            # Ghost/cut side: accumulate onto enriched phantom DOF (lines 126-163)
            if enr_id not in f_enr_dict:
                f_enr_dict[enr_id] = np.zeros(3)
                m_enr_dict[enr_id] = np.zeros(3)
            f_enr_dict[enr_id] += fg_scaled[i]
            m_enr_dict[enr_id] += mg_scaled[i]


# ---------------------------------------------------------------------------
# 3. Single-Element Shell Force Kernel (BT4 / Mindlin-Reissner)
# ---------------------------------------------------------------------------

def _single_shell_forces(
    xe: np.ndarray,
    ve: np.ndarray,
    vre: np.ndarray,
    thick: float,
    youngs_modulus: float,
    poisson_ratio: float,
    shear_modulus: float,
    density: float,
    dt: float,
    hg_coeff: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute internal nodal forces and moments for a single 4-node BT shell element.

    Upstream: cforc3.F, cdefo3.F, ccurv3.F, cstra3.F, cfint3.F, chvis3.F.

    Returns
    -------
    fg : ndarray of shape (4, 3)
        Global translational nodal internal forces.
    mg : ndarray of shape (4, 3)
        Global nodal internal moments.
    de_int : float
        Internal strain energy increment during time step dt.
    de_hg : float
        Hourglass stabilization energy increment.
    dt_crit : float
        Characteristic critical time step.
    """
    E_mat = youngs_modulus
    nu = poisson_ratio
    G = shear_modulus
    rho = density
    t = thick

    # Local geometry and corotational frame
    E_triad, xl, area, b1, b2 = _quad_local_geometry(xe)

    # Local translational velocities and angular velocities
    # vl[i] = v_i @ E_triad, vrl[i] = vr_i @ E_triad
    vl = ve @ E_triad
    vrl = vre @ E_triad

    vx, vy, vz = vl[:, 0], vl[:, 1], vl[:, 2]
    thx, thy = vrl[:, 0], vrl[:, 1]

    # Membrane velocity strains (cdefo3.F lines 65-80)
    # d_xx = B1 . vx, d_yy = B2 . vy, d_xy = B1 . vy + B2 . vx
    d_xx = float(np.dot(b1, vx))
    d_yy = float(np.dot(b2, vy))
    d_xy = float(np.dot(b1, vy) + np.dot(b2, vx))

    # Curvature rates (ccurv3.F lines 55-65)
    # k_xx = B1 . thy, k_yy = -B2 . thx, k_xy = B2 . thy - B1 . thx
    k_xx = float(np.dot(b1, thy))
    k_yy = float(-np.dot(b2, thx))
    k_xy = float(np.dot(b2, thy) - np.dot(b1, thx))

    # Transverse shear rates (cdefo3.F lines 85-92)
    # g_xz = B1 . vz + mean(thy), g_yz = B2 . vz - mean(thx)
    mean_thy = float(np.mean(thy))
    mean_thx = float(np.mean(thx))
    g_xz = float(np.dot(b1, vz) + mean_thy)
    g_yz = float(np.dot(b2, vz) - mean_thx)

    # Elastic plane-stress constitutive law (cmain3.F, mulawc.F90)
    c11 = E_mat / (1.0 - nu * nu)
    c12 = nu * c11
    c33 = G

    # Membrane resultants N (force/length)
    n_xx = t * (c11 * d_xx * dt + c12 * d_yy * dt)
    n_yy = t * (c12 * d_xx * dt + c11 * d_yy * dt)
    n_xy = t * c33 * d_xy * dt

    # Bending resultants M (moment/length): D = E*t^3 / (12*(1-nu^2))
    d_bend = (t ** 3 / 12.0) * c11
    m_xx = (t ** 3 / 12.0) * (c11 * k_xx * dt + c12 * k_yy * dt)
    m_yy = (t ** 3 / 12.0) * (c12 * k_xx * dt + c11 * k_yy * dt)
    m_xy = (t ** 3 / 12.0) * c33 * k_xy * dt

    # Transverse shear resultants Q = kappa * G * t * gamma
    q_xz = SHEAR_FACTOR * G * t * g_xz * dt
    q_yz = SHEAR_FACTOR * G * t * g_yz * dt

    # Virtual work rate / internal force transposition (cfint3.F lines 40-120)
    # fl[i, 0] = A * (b1[i]*N_xx + b2[i]*N_xy)
    # fl[i, 1] = A * (b2[i]*N_yy + b1[i]*N_xy)
    # fl[i, 2] = A * (b1[i]*Q_xz + b2[i]*Q_yz)
    # ml[i, 0] = A * (-b2[i]*M_yy - b1[i]*M_xy - Q_yz / 4)
    # ml[i, 1] = A * ( b1[i]*M_xx + b2[i]*M_xy + Q_xz / 4)
    # ml[i, 2] = 0 (drilling)
    fl = np.zeros((4, 3))
    ml = np.zeros((4, 3))

    for i in range(4):
        fl[i, 0] = area * (b1[i] * n_xx + b2[i] * n_xy)
        fl[i, 1] = area * (b2[i] * n_yy + b1[i] * n_xy)
        fl[i, 2] = area * (b1[i] * q_xz + b2[i] * q_yz)

        ml[i, 0] = area * (-b2[i] * m_yy - b1[i] * m_xy - 0.25 * q_yz)
        ml[i, 1] = area * (b1[i] * m_xx + b2[i] * m_xy + 0.25 * q_xz)
        ml[i, 2] = 0.0

    # Hourglass control (chvis3.F)
    # Flanagan-Belytschko gamma vector: gamma = h - (h.x)*B1 - (h.y)*B2
    h = np.array([1.0, -1.0, 1.0, -1.0])
    gamma = h - np.dot(h, xl[:, 0]) * b1 - np.dot(h, xl[:, 1]) * b2
    k_hg = hg_coeff * E_mat * t / 8.0
    vhg_x = float(np.dot(gamma, vx))
    vhg_y = float(np.dot(gamma, vy))
    vhg_z = float(np.dot(gamma, vz))

    f_hg = np.zeros((4, 3))
    f_hg[:, 0] = k_hg * vhg_x * dt * gamma
    f_hg[:, 1] = k_hg * vhg_y * dt * gamma
    f_hg[:, 2] = k_hg * vhg_z * dt * gamma

    de_hg = float(np.sum(f_hg * vl)) * dt

    fl += f_hg

    # Transform forces/moments back to global coordinates (E_triad @ f_local)
    fg = fl @ E_triad.T
    mg = ml @ E_triad.T

    # Strain energy increment: dE_int = A * (N:dm + M:k + Q:gamma) (cbilan.F)
    de_int = area * (
        n_xx * d_xx + n_yy * d_yy + n_xy * d_xy
        + m_xx * k_xx + m_yy * k_yy + m_xy * k_xy
        + q_xz * g_xz + q_yz * g_yz
    )

    # Characteristic length and critical time step (cdlen3.F, cdt3.F)
    dx12 = np.linalg.norm(xe[1] - xe[0])
    dx23 = np.linalg.norm(xe[2] - xe[1])
    dx34 = np.linalg.norm(xe[3] - xe[2])
    dx41 = np.linalg.norm(xe[0] - xe[3])
    l_max = max(dx12, dx23, dx34, dx41, EM20)
    lc = area / l_max
    c_sound = math.sqrt(E_mat / max(rho * (1.0 - nu * nu), EM20))
    dt_crit = lc / max(c_sound, EM20)

    return fg, mg, de_int, de_hg, dt_crit


# ---------------------------------------------------------------------------
# 4. XFEM Enriched Shell Single-Element Force Integration
# ---------------------------------------------------------------------------

def xfem_shell_quad4_forces(
    xe_std: np.ndarray,
    ve_std: np.ndarray,
    vre_std: np.ndarray,
    cut: Optional[XfemElementCut],
    thick: float,
    youngs_modulus: float,
    poisson_ratio: float,
    shear_modulus: float,
    density: float,
    dt: float,
    x_enr_dict: Optional[Dict[int, np.ndarray]] = None,
    v_enr_dict: Optional[Dict[int, np.ndarray]] = None,
    vr_enr_dict: Optional[Dict[int, np.ndarray]] = None,
    hg_coeff: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], float, float, float]:
    """Calculate internal forces, moments, and energy for a standard or XFEM-cracked shell.

    Ported from OpenRadioss subroutines:
      - ``$OR_SRC/engine/source/elements/xfem/cforc3_crk.F``
      - ``$OR_SRC/engine/source/elements/xfem/ccoor3_crk.F``
      - ``$OR_SRC/engine/source/elements/xfem/xfemfsky.F``

    If the element is uncut:
      Standard Belytschko-Tsay 4-node shell evaluation.
    If the element is cut by XFEM crack:
      Iterates over active phantom element levels (IXEL=1..3) with area fractions w_k.
      Evaluates forces on phantom kinematics and scatters:
        - to standard nodes (f_std, m_std) if un-enriched (ENR <= 0).
        - to enriched DOFs (f_enr_dict, m_enr_dict) if enriched (ENR > 0).
      Books internal energy work dE_int and hourglass work dE_hg (Critical Rule 3).

    Returns
    -------
    f_std : ndarray of shape (4, 3)
        Accumulated internal translational forces on the 4 standard mesh nodes.
    m_std : ndarray of shape (4, 3)
        Accumulated internal moments on the 4 standard mesh nodes.
    f_enr_dict : dict
        Enriched translational forces: enriched_id -> ndarray of shape (3,).
    m_enr_dict : dict
        Enriched moments: enriched_id -> ndarray of shape (3,).
    total_de_int : float
        Total internal strain energy increment for the element.
    total_de_hg : float
        Total hourglass energy increment.
    dt_crit : float
        Critical stable time step.
    """
    if x_enr_dict is None:
        x_enr_dict = {}
    if v_enr_dict is None:
        v_enr_dict = {}
    if vr_enr_dict is None:
        vr_enr_dict = {}

    f_std = np.zeros((4, 3))
    m_std = np.zeros((4, 3))
    f_enr_dict: Dict[int, np.ndarray] = {}
    m_enr_dict: Dict[int, np.ndarray] = {}

    # Fast path: Uncut element (standard shell force integration)
    if cut is None or not cut.is_cut:
        fg, mg, de_int, de_hg, dt_crit = _single_shell_forces(
            xe=xe_std,
            ve=ve_std,
            vre=vre_std,
            thick=thick,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            shear_modulus=shear_modulus,
            density=density,
            dt=dt,
            hg_coeff=hg_coeff,
        )
        f_std += fg
        m_std += mg
        return f_std, m_std, f_enr_dict, m_enr_dict, de_int, de_hg, dt_crit

    # Enriched XFEM path (cforc3_crk.F lines 301-630)
    # Loop over phantom elements:
    # Level 1 (IXEL=1, positive domain), Level 2 (IXEL=2, negative domain),
    # and Level 3 if cut.itri != 0
    num_phantoms = 3 if cut.itri != 0 else 2
    total_de_int = 0.0
    total_de_hg = 0.0
    min_dt = EP30

    for k in range(num_phantoms):
        w_k = float(cut.area_fractions[k])
        if w_k <= 1.0e-6:
            continue

        enr_ids = cut.enr_ids[k]

        # 1. Gather phantom element kinematics (ccoor3_crk.F)
        xe_ph, ve_ph, vre_ph = gather_phantom_kinematics(
            xe_std=xe_std,
            ve_std=ve_std,
            vre_std=vre_std,
            enr_node_ids=enr_ids,
            x_enr_dict=x_enr_dict,
            v_enr_dict=v_enr_dict,
            vr_enr_dict=vr_enr_dict,
        )

        # 2. Evaluate shell forces on phantom component
        fg_ph, mg_ph, de_int_k, de_hg_k, dt_k = _single_shell_forces(
            xe=xe_ph,
            ve=ve_ph,
            vre=vre_ph,
            thick=thick,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            shear_modulus=shear_modulus,
            density=density,
            dt=dt,
            hg_coeff=hg_coeff,
        )

        # 3. Scatter forces to standard and enriched DOFs (xfemfsky.F lines 123-260)
        scatter_phantom_forces(
            fg_phantom=fg_ph,
            mg_phantom=mg_ph,
            area_weight=w_k,
            enr_node_ids=enr_ids,
            f_std=f_std,
            m_std=m_std,
            f_enr_dict=f_enr_dict,
            m_enr_dict=m_enr_dict,
        )

        # 4. Energy accounting (Critical Rule 3): book work in energy ledgers
        total_de_int += w_k * de_int_k
        total_de_hg += w_k * de_hg_k
        min_dt = min(min_dt, dt_k)

    return f_std, m_std, f_enr_dict, m_enr_dict, total_de_int, total_de_hg, min_dt


# ---------------------------------------------------------------------------
# 5. XfemShellGroup: Multi-Element Group Manager
# ---------------------------------------------------------------------------

class XfemShellGroup:
    """Manages an assembly of quadrilateral shell elements with XFEM crack enrichment.

    Follows standard pyradioss element kernel architecture with full support
    for enrichment DOFs, phantom node kinematics, and energy accounting.
    """

    def __init__(
        self,
        connectivity: np.ndarray,
        thick: float = 1.0e-3,
        youngs_modulus: float = 2.1e11,
        poisson_ratio: float = 0.3,
        density: float = 7800.0,
        hg_coeff: float = 0.01,
    ) -> None:
        self.conn = np.asarray(connectivity, dtype=int)
        self.n_elements = len(self.conn)
        self.thick = float(thick)
        self.E = float(youngs_modulus)
        self.nu = float(poisson_ratio)
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.rho = float(density)
        self.hg_coeff = float(hg_coeff)

        # XFEM infrastructure
        self.dof_manager = EnrichmentDOFManager()
        self.cuts: Dict[int, XfemElementCut] = {}

        # Enriched state vectors
        self.x_enr: Dict[int, np.ndarray] = {}
        self.v_enr: Dict[int, np.ndarray] = {}
        self.vr_enr: Dict[int, np.ndarray] = {}
        self.a_enr: Dict[int, np.ndarray] = {}
        self.ar_enr: Dict[int, np.ndarray] = {}
        self.m_enr: Dict[int, float] = {}

        # Energy ledgers
        self.eint = 0.0
        self.ehour = 0.0

    def add_crack(self, elem_id: int, cut: XfemElementCut, mesh_x: np.ndarray) -> None:
        """Register a crack cut on an element and initialize its enriched phantom DOFs."""
        assert 0 <= elem_id < self.n_elements, f"Invalid elem_id {elem_id}"
        self.cuts[elem_id] = cut
        nodes = self.conn[elem_id]
        self.dof_manager.assign_element_enrichments(elem_id, cut, nodes)

        # Initialize enriched node coordinates and velocities from standard nodes
        for k in range(3):
            for i in range(4):
                enr_id = cut.enr_ids[k, i]
                if enr_id > 0 and enr_id not in self.x_enr:
                    std_node = nodes[i]
                    self.x_enr[enr_id] = mesh_x[std_node].copy()
                    self.v_enr[enr_id] = np.zeros(3)
                    self.vr_enr[enr_id] = np.zeros(3)
                    self.a_enr[enr_id] = np.zeros(3)
                    self.ar_enr[enr_id] = np.zeros(3)
                    # Nodal mass share
                    self.m_enr[enr_id] = self.rho * self.thick * 1.0 / 4.0

    def forces(
        self,
        x: np.ndarray,
        v: np.ndarray,
        vr: np.ndarray,
        dt: float,
        fint: Optional[np.ndarray] = None,
        mint: Optional[np.ndarray] = None,
        fenr: Optional[Dict[int, np.ndarray]] = None,
        menr: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], float]:
        """Compute internal forces on standard nodes and enriched DOFs.

        Sign convention: internal forces ACCUMULATED NEGATED into fint/mint:
          fint[i] -= F_internal_i
        Matching OpenRadioss cupdt3_crk.F and standard pyradioss convention.

        Returns
        -------
        dt_crit_arr : ndarray of shape (n_elements,)
            Critical stable time step per element.
        fenr_out : dict
            Accumulated forces on enriched phantom DOFs.
        menr_out : dict
            Accumulated moments on enriched phantom DOFs.
        de_int_total : float
            Total internal energy increment for the group.
        """
        if fint is None:
            fint = np.zeros_like(x)
        if mint is None:
            mint = np.zeros_like(x)
        if fenr is None:
            fenr = {}
        if menr is None:
            menr = {}

        dt_crits = np.empty(self.n_elements)
        de_int_total = 0.0

        for e in range(self.n_elements):
            nodes = self.conn[e]
            xe = x[nodes]
            ve = v[nodes]
            vre = vr[nodes]
            cut = self.cuts.get(e, None)

            fe_std, me_std, fe_enr, me_enr, de_i, de_h, dt_c = xfem_shell_quad4_forces(
                xe_std=xe,
                ve_std=ve,
                vre_std=vre,
                cut=cut,
                thick=self.thick,
                youngs_modulus=self.E,
                poisson_ratio=self.nu,
                shear_modulus=self.G,
                density=self.rho,
                dt=dt,
                x_enr_dict=self.x_enr,
                v_enr_dict=self.v_enr,
                vr_enr_dict=self.vr_enr,
                hg_coeff=self.hg_coeff,
            )

            dt_crits[e] = dt_c
            de_int_total += de_i
            self.eint += de_i
            self.ehour += de_h

            # Accumulate negated internal forces into global arrays
            for i, n_id in enumerate(nodes):
                fint[n_id] -= fe_std[i]
                mint[n_id] -= me_std[i]

            for enr_id, f_val in fe_enr.items():
                fenr[enr_id] = fenr.get(enr_id, np.zeros(3)) - f_val
            for enr_id, m_val in me_enr.items():
                menr[enr_id] = menr.get(enr_id, np.zeros(3)) - m_val

        return dt_crits, fenr, menr, de_int_total


# ---------------------------------------------------------------------------
# 6. Explicit Dynamic Integration Step (crk_velocity.F lines 30-127)
# ---------------------------------------------------------------------------

def integrate_xfem_step(
    group: XfemShellGroup,
    x: np.ndarray,
    v: np.ndarray,
    m_nodes: np.ndarray,
    f_ext: np.ndarray,
    dt: float,
    vr: Optional[np.ndarray] = None,
    mr_nodes: Optional[np.ndarray] = None,
    m_ext: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Advance mesh and enriched phantom DOFs by one explicit time step.

    Ported from OpenRadioss subroutine CRK_VELOCITY:
      ``$OR_SRC/engine/source/elements/xfem/crk_velocity.F``
      lines 74-100:
        CRKAVX%V  = CRKAVX%V + DT12 * A
        CRKAVX%VR = CRKAVX%VR + DT12 * AR
        DX = DT2 * CRKAVX%V
        CRKAVX%X = CRKAVX%X + DX

    Parameters
    ----------
    group : XfemShellGroup
        The XFEM shell group.
    x : ndarray of shape (N, 3)
        Standard nodal coordinates.
    v : ndarray of shape (N, 3)
        Standard nodal velocities.
    m_nodes : ndarray of shape (N,)
        Lumped masses at standard nodes.
    f_ext : ndarray of shape (N, 3)
        External forces on standard nodes.
    dt : float
        Time step increment.
    vr : ndarray, optional
        Standard rotational velocities.
    mr_nodes : ndarray, optional
        Rotational inertia at standard nodes.
    m_ext : ndarray, optional
        External moments on standard nodes.

    Returns
    -------
    x_new : ndarray of shape (N, 3)
        Updated nodal coordinates.
    v_new : ndarray of shape (N, 3)
        Updated nodal velocities.
    de_int : float
        Internal strain energy increment during step.
    """
    if vr is None:
        vr = np.zeros_like(v)
    if mr_nodes is None:
        mr_nodes = m_nodes * (group.thick ** 2) / 12.0
    if m_ext is None:
        m_ext = np.zeros_like(vr)

    f_int = np.zeros_like(x)
    m_int = np.zeros_like(x)
    f_enr: Dict[int, np.ndarray] = {}
    m_enr: Dict[int, np.ndarray] = {}

    # 1. Compute internal forces on standard nodes and enriched DOFs
    _, f_enr, m_enr, de_int = group.forces(
        x=x,
        v=v,
        vr=vr,
        dt=dt,
        fint=f_int,
        mint=m_int,
        fenr=f_enr,
        menr=m_enr,
    )

    # 2. Standard nodes acceleration: a = (f_ext + f_int) / m  (f_int is already negated)
    a = (f_ext + f_int) / np.maximum(m_nodes[:, None], EM20)
    ar = (m_ext + m_int) / np.maximum(mr_nodes[:, None], EM20)

    # Velocity and coordinate update: central difference
    v_new = v + 0.5 * dt * a
    x_new = x + dt * v_new
    vr_new = vr + 0.5 * dt * ar

    # 3. Enriched phantom DOFs update (crk_velocity.F lines 74-100)
    for enr_id in group.x_enr:
        m_enr_val = group.m_enr.get(enr_id, 1.0)
        f_val = f_enr.get(enr_id, np.zeros(3))
        a_val = f_val / max(m_enr_val, EM20)
        group.a_enr[enr_id] = a_val

        # Update phantom velocity and position
        v_ph = group.v_enr[enr_id] + 0.5 * dt * a_val
        x_ph = group.x_enr[enr_id] + dt * v_ph

        group.v_enr[enr_id] = v_ph
        group.x_enr[enr_id] = x_ph

    return x_new, v_new, de_int
