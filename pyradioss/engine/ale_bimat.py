# pyradioss/engine/ale_bimat.py
"""
ALE Bi-material Volume-of-Fluid (VOF) Subsystem.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/bimat/balph2.F: bi-material VOF evolution equation and volume fraction limiting
- engine/source/ale/bimat/amulf2.F: interface-weighted multi-material face fluxes
- engine/source/ale/bimat/bcumu2.F: mixture property computation and nodal force accumulation
- engine/source/ale/bimat/bafil2.F: nodal fill function convective variation
- engine/source/ale/bimat/bmultn.F: nodal fill function update and clamping
- engine/source/ale/bimat/blero2.F: phase density, volume change, and re-initialization
- engine/source/ale/bimat/brest2.F: bi-material restart data
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List, Any

import numpy as np

_EM30 = 1.0e-30
_EM20 = 1.0e-20
_EM15 = 1.0e-15
_ZEP99 = 0.99
_ZEP999 = 0.999
_THIRD = 1.0 / 3.0
_HALF = 0.5
_FOURTH = 0.25


# =============================================================================
# VOF Element Volume Fraction from Nodal Fill Function
# =============================================================================

def element_vof_from_nodal_fill(fill_nodes: np.ndarray) -> float:
    """Calculate element volume fraction alpha from 4-node fill function values.

    Ported from $OR_SRC/engine/source/ale/bimat/balph2.F
    lines 136-147:
      ALPN = sum_{n=1..4} max(0, FILL_n)
      ALPD = sum_{n=1..4} |FILL_n|
      If ALPD > 1e-20:
        alpha = ALPN / ALPD
      Else:
        alpha = 0.0

    Args:
        fill_nodes: (4,) array of nodal fill function values in [-1, 1].
                    Positive indicates inside material; negative outside.

    Returns:
        alpha: volume fraction in [0, 1].
    """
    alpn = float(np.sum(np.maximum(0.0, fill_nodes)))
    alpd = float(np.sum(np.abs(fill_nodes)))
    if alpd > _EM20:
        return float(np.clip(alpn / alpd, 0.0, 1.0))
    return 0.0


def elements_vof_from_nodal_fill(
    fill: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Vectorized calculation of element volume fractions from nodal fill values.

    Args:
        fill: (n_nodes,) nodal fill function values.
        connectivity: (n_elem, 4) element vertex indices.

    Returns:
        alpha: (n_elem,) element volume fractions.
    """
    elem_fills = fill[connectivity]  # (n_elem, 4)
    alpn = np.sum(np.maximum(0.0, elem_fills), axis=1)
    alpd = np.sum(np.abs(elem_fills), axis=1)
    alpha = np.zeros(len(connectivity), dtype=np.float64)
    mask = alpd > _EM20
    alpha[mask] = np.clip(alpn[mask] / alpd[mask], 0.0, 1.0)
    return alpha


# =============================================================================
# Bi-material VOF Evolution Equation
# =============================================================================

def bimat_vof_evolution(
    alph_old: np.ndarray,
    vol_old: np.ndarray,
    vol_new: np.ndarray,
    face_fluxes: np.ndarray,
    flu1: np.ndarray,
    dt: float,
    strain_rate_trace: Optional[np.ndarray] = None,
    stress_trace: Optional[np.ndarray] = None,
    rho_new: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve bi-material volume fraction for phase 1.

    Ported from $OR_SRC/engine/source/ale/bimat/balph2.F
    lines 170-230:
      For element I:
        If ALPH_old in (0, 1):
          ALPH_N = (VOLO - DT1 * 0.5 * (FLU1 + sum(FLUX))) / VOLN
          ALPH_NN = ALPH_N * (1 + tr(D) * DT1)
          If tr(sigma) > 0:
            ALPH_N = min(ALPH_N, ALPH_NN)
          Else:
            ALPH_N = max(ALPH_N, ALPH_NN)
          If ALPH_N > 0 and rho_new <= 0:
            ALPH_N = 0
          ALPH = clip(ALPH_N, 0, 1)
        Else if ALPH_old == 1:
          ALPH_N = (VOLO - DT1 * 0.5 * (FLU1 + sum(FLUX))) / VOLN
          ALPH_NN = ALPH_N * (1 + tr(D) * DT1)
          ALPH_N = max(ALPH_N, ALPH_NN, ALPH_old)
          If rho_new <= 0: ALPH_N = 0
          ALPH = min(1, ALPH_N)
        Else:
          ALPH = ALPH_old

      Small cutoff: if ALPH < 1e-15: ALPH = 0

    Args:
        alph_old: (n_elem,) previous volume fractions.
        vol_old: (n_elem,) previous phase volume V_1^old.
        vol_new: (n_elem,) new total element volume V^new.
        face_fluxes: (n_elem, 4) phase face volume fluxes.
        flu1: (n_elem,) auxiliary upwind sum.
        dt: time step.
        strain_rate_trace: (n_elem,) tr(D) = D11 + D22 + D33.
        stress_trace: (n_elem,) tr(sigma) = sig11 + sig22 + sig33.
        rho_new: (n_elem,) updated phase density.

    Returns:
        alph_new: (n_elem,) updated volume fraction.
        dalph: (n_elem,) change in volume fraction (alph_new - alph_old).
    """
    n_elem = len(alph_old)
    alph_new = np.copy(alph_old)
    dalph = np.zeros(n_elem, dtype=np.float64)

    sum_flux = np.sum(face_fluxes, axis=1)

    for i in range(n_elem):
        a_old = alph_old[i]
        vn = max(vol_new[i], _EM30)
        vo = vol_old[i]
        fl_sum = 0.5 * (flu1[i] + sum_flux[i])

        tr_d = strain_rate_trace[i] if strain_rate_trace is not None else 0.0
        tr_sig = stress_trace[i] if stress_trace is not None else 0.0
        rho = rho_new[i] if rho_new is not None else 1000.0

        if 0.0 < a_old < 1.0:
            alph_n = (vo - dt * fl_sum) / vn
            alph_nn = alph_n * (1.0 + tr_d * dt)
            if tr_sig > 0.0:
                alph_n = min(alph_n, alph_nn)
            else:
                alph_n = max(alph_n, alph_nn)

            if 0.0 < alph_n <= _ZEP99 and rho <= 0.0:
                alph_n = 0.0

            dalph[i] = alph_n - a_old
            alph_new[i] = float(np.clip(alph_n, 0.0, 1.0))

        elif a_old >= 1.0 and alph_new[i] < _ZEP999:
            alph_n = (vo - dt * fl_sum) / vn
            alph_nn = alph_n * (1.0 + tr_d * dt)
            alph_n = max(alph_n, alph_nn, a_old)
            if rho <= 0.0:
                alph_n = 0.0
            alph_new[i] = min(1.0, alph_n)
            dalph[i] = alph_new[i] - a_old
        else:
            alph_new[i] = a_old
            dalph[i] = 0.0

        # Small volume fraction cutoff
        if alph_new[i] < _EM15:
            alph_new[i] = 0.0
            dalph[i] = 0.0

    return alph_new, dalph


def bimat_two_material_constraint(
    alph1: np.ndarray,
    alph2: np.ndarray,
    c11: float = 1.0,
    c12: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce the multi-material packing constraint alpha_1 + alpha_2 <= 1.

    Ported from $OR_SRC/engine/source/ale/bimat/balph2.F
    lines 335-343:
      ALPHT = ALPH1 + ALPH2
      If ALPHT > 1:
        DA = (ALPHT - 1) / (c11 * ALPH2 + c12 * ALPH1)
        ALPH1 = ALPH1 * (1 - c12 * DA)
        ALPH2 = ALPH2 * (1 - c11 * DA)

    Args:
        alph1: (n_elem,) volume fraction of phase 1.
        alph2: (n_elem,) volume fraction of phase 2.
        c11: compressibility weight for phase 1.
        c12: compressibility weight for phase 2.

    Returns:
        alph1_adj, alph2_adj: adjusted volume fractions.
    """
    a1 = np.copy(alph1)
    a2 = np.copy(alph2)
    alph_tot = a1 + a2

    over = alph_tot > 1.0
    if np.any(over):
        denom = c11 * a2[over] + c12 * a1[over]
        denom = np.maximum(denom, 1e-15)
        da = (alph_tot[over] - 1.0) / denom
        a1[over] *= (1.0 - c12 * da)
        a2[over] *= (1.0 - c11 * da)

    a1 = np.clip(a1, 0.0, 1.0)
    a2 = np.clip(a2, 0.0, 1.0)
    # Cutoff small values
    a1[a1 < _EM15] = 0.0
    a2[a2 < _EM15] = 0.0
    return a1, a2


# =============================================================================
# Interface-Weighted Multi-Material Face Fluxes (AMULF2)
# =============================================================================

def bimat_face_fluxes(
    fill: np.ndarray,
    dfill: np.ndarray,
    connectivity: np.ndarray,
    flux_total: np.ndarray,
    vol: np.ndarray,
    alph: np.ndarray,
    neighbor_elem: np.ndarray,
    upwind_param: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute interface-weighted face volume fluxes for a bi-material phase.

    Ported from $OR_SRC/engine/source/ale/bimat/amulf2.F
    lines 68-181:
      For each quad face (k=1..4) connecting nodes NC_A and NC_B:
        ALPN = max(0, FILL_A) + max(0, FILL_A - DFILL_A) + max(0, FILL_B) + max(0, FILL_B - DFILL_B)
        ALPD = |FILL_A| + |FILL_A - DFILL_A| + |FILL_B| + |FILL_B - DFILL_B|
        alpha_face = ALPN / ALPD (1.0 if ALPD == 0)
        FLUB_k = FLUX_total_k * alpha_face
        If FLUX_total_k >= 0:
          FLUB_k = max(FLUB_k, FLUX_total_k - (1 - alpha_1)*VOL_1)
          FLUB_k = min(FLUB_k, alpha_1 * VOL_1)
        Else:
          neighbor J:
          FLUB_k = -max(-FLUB_k, -FLUX_total_k - (1 - alpha_2)*VOL_2)
          FLUB_k = -min(-FLUB_k, alpha_2 * VOL_2)

      Upwind splitting:
        FLUX_k = FLUB_k - eta * |FLUB_k|
        FLU1   = sum (FLUB_k + eta * |FLUB_k|)

    Args:
        fill: (n_nodes,) nodal fill function.
        dfill: (n_nodes,) nodal fill increment.
        connectivity: (n_elem, 4) element connectivity.
        flux_total: (n_elem, 4) total element face volume fluxes.
        vol: (n_elem,) element volumes.
        alph: (n_elem,) phase volume fraction.
        neighbor_elem: (n_elem, 4) neighbor connectivity.
        upwind_param: upwind factor eta.

    Returns:
        flux_phase: (n_elem, 4) phase face volume fluxes.
        flu1_phase: (n_elem,) phase auxiliary upwind flux sum.
    """
    n_elem = len(connectivity)
    flux_phase = np.zeros((n_elem, 4), dtype=np.float64)
    flu1_phase = np.zeros(n_elem, dtype=np.float64)

    # 4 quad edges: (0, 1), (1, 2), (2, 3), (3, 0)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    flub = np.zeros((n_elem, 4), dtype=np.float64)

    for ed_idx, (loc_a, loc_b) in enumerate(edge_pairs):
        na = connectivity[:, loc_a]
        nb = connectivity[:, loc_b]

        fa = fill[na]
        fb = fill[nb]
        dfa = dfill[na]
        dfb = dfill[nb]

        alpn = (np.maximum(0.0, fa) + np.maximum(0.0, fa - dfa) +
                np.maximum(0.0, fb) + np.maximum(0.0, fb - dfb))
        alpd = (np.abs(fa) + np.abs(fa - dfa) +
                np.abs(fb) + np.abs(fb - dfb))

        alph_face = np.ones(n_elem, dtype=np.float64)
        valid = alpd > _EM20
        alph_face[valid] = alpn[valid] / alpd[valid]

        fl_k = flux_total[:, ed_idx]
        flub_k = fl_k * alph_face

        alp1_vol = alph * vol
        calp1_vol = (1.0 - alph) * vol

        # Outgoing flux
        out = fl_k >= 0.0
        flub_k[out] = np.maximum(flub_k[out], fl_k[out] - calp1_vol[out])
        flub_k[out] = np.minimum(flub_k[out], alp1_vol[out])

        # Incoming flux (fl_k < 0) bounded by neighbor state
        inc = ~out
        if np.any(inc):
            for i in np.where(inc)[0]:
                nbr = neighbor_elem[i, ed_idx]
                if nbr >= 0:
                    alp2_vol = alph[nbr] * vol[nbr]
                    calp2_vol = (1.0 - alph[nbr]) * vol[nbr]
                else:
                    alp2_vol = alp1_vol[i]
                    calp2_vol = calp1_vol[i]

                val = flub_k[i]
                fl = fl_k[i]
                val = -max(-val, -fl - calp2_vol)
                val = -min(-val, alp2_vol)
                flub_k[i] = val

        flub[:, ed_idx] = flub_k

    # Upwind splitting
    eta = float(np.clip(upwind_param, 0.0, 1.0))
    abs_flub = np.abs(flub)
    flux_phase = flub - eta * abs_flub
    flu1_phase = np.sum(flub + eta * abs_flub, axis=1)

    return flux_phase, flu1_phase


# =============================================================================
# Mixture Properties and Nodal Accumulation (BCUMU2, BLERO2)
# =============================================================================

def bimat_mixture_properties(
    alph1: np.ndarray,
    alph2: np.ndarray,
    sig1: np.ndarray,
    sig2: np.ndarray,
    eint1: np.ndarray,
    eint2: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray,
    bulk1: Optional[np.ndarray] = None,
    bulk2: Optional[np.ndarray] = None,
    temp1: Optional[np.ndarray] = None,
    temp2: Optional[np.ndarray] = None,
    plas1: Optional[np.ndarray] = None,
    plas2: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute homogenized mixture properties for bi-material elements.

    Ported from $OR_SRC/engine/source/ale/bimat/bcumu2.F
    lines 110-140:
      SIGT(I, J)  = SIG1(I, J)*ALPH1(I) + SIG2(I, J)*ALPH2(I)
      EINTT(I)    = EINT1(I)*ALPH1(I)   + EINT2(I)*ALPH2(I)
      RHOT(I)     = RHO1(I)*ALPH1(I)    + RHO2(I)*ALPH2(I)
      BULKT(I)    = BULK1(I)*ALPH1(I)   + BULK2(I)*ALPH2(I)
      TEMPT(I)    = TEMP1(I)*ALPH1(I)   + TEMP2(I)*ALPH2(I)
      PLAST(I)    = PLAS1(I)*ALPH1(I)   + PLAS2(I)*ALPH2(I)

    Args:
        alph1: (n_elem,) volume fraction of phase 1.
        alph2: (n_elem,) volume fraction of phase 2.
        sig1, sig2: (n_elem, 6) Cauchy stresses.
        eint1, eint2: (n_elem,) internal energies.
        rho1, rho2: (n_elem,) densities.
        bulk1, bulk2: optional (n_elem,) bulk moduli.
        temp1, temp2: optional (n_elem,) temperatures.
        plas1, plas2: optional (n_elem,) effective plastic strains.

    Returns:
        dict containing homogenized mixture arrays:
          'sig_mix': (n_elem, 6),
          'eint_mix': (n_elem,),
          'rho_mix': (n_elem,),
          'bulk_mix': (n_elem,),
          'temp_mix': (n_elem,),
          'plas_mix': (n_elem,).
    """
    a1 = alph1[:, np.newaxis]
    a2 = alph2[:, np.newaxis]

    sig_mix = a1 * sig1 + a2 * sig2
    eint_mix = alph1 * eint1 + alph2 * eint2
    rho_mix = alph1 * rho1 + alph2 * rho2

    bulk_mix = (alph1 * bulk1 + alph2 * bulk2) if (bulk1 is not None and bulk2 is not None) else None
    temp_mix = (alph1 * temp1 + alph2 * temp2) if (temp1 is not None and temp2 is not None) else None
    plas_mix = (alph1 * plas1 + alph2 * plas2) if (plas1 is not None and plas2 is not None) else None

    return {
        "sig_mix": sig_mix,
        "eint_mix": eint_mix,
        "rho_mix": rho_mix,
        "bulk_mix": bulk_mix,
        "temp_mix": temp_mix,
        "plas_mix": plas_mix,
    }


def bimat_nodal_force_accumulation(
    element_forces: np.ndarray,
    alph: np.ndarray,
    connectivity: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Accumulate volume-fraction-weighted element forces to global nodal force vector.

    Ported from $OR_SRC/engine/source/ale/bimat/bcumu2.F
    lines 82-97:
      F_node += alph_elem * F_elem_node

    Args:
        element_forces: (n_elem, 4, 2) nodal forces for each quad element.
        alph: (n_elem,) volume fraction.
        connectivity: (n_elem, 4) element connectivity.
        n_nodes: total number of nodes in the mesh.

    Returns:
        nodal_forces: (n_nodes, 2) accumulated global nodal forces.
    """
    nodal_forces = np.zeros((n_nodes, 2), dtype=np.float64)
    weighted_forces = alph[:, np.newaxis, np.newaxis] * element_forces
    for j in range(4):
        node_indices = connectivity[:, j]
        np.add.at(nodal_forces, node_indices, weighted_forces[:, j, :])
    return nodal_forces


# =============================================================================
# Nodal Fill Convection and Update (BAFIL2, BMULTN)
# =============================================================================

def bimat_nodal_fill_convection(
    v: np.ndarray,
    w: np.ndarray,
    fill: np.ndarray,
    dalph: np.ndarray,
    connectivity: np.ndarray,
    coords: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute convective variation of nodal fill function across quad elements.

    Ported from $OR_SRC/engine/source/ale/bimat/bafil2.F
    lines 63-189:
      Relative velocity v_d = v - w
      Mean relative velocity in element weighted by (fill + 1)
      Local derivatives ds, dt from quadrilateral isoparametric mapping
      Element filling rate FA = -DALPH * sum(|FILL_n|) / (dt * NP)
      Convective variation DF accumulated to DFILL.

    Args:
        v: (n_nodes, 2) material velocity.
        w: (n_nodes, 2) mesh grid velocity.
        fill: (n_nodes,) current nodal fill function in [-1, 1].
        dalph: (n_elem,) volume fraction increments.
        connectivity: (n_elem, 4) element connectivity.
        coords: (n_nodes, 2) nodal coordinates.
        dt: time step.

    Returns:
        dfill: (n_nodes,) accumulated convective fill increments.
        node_count: (n_nodes,) count of elements sharing each node.
    """
    n_nodes = len(fill)
    dfill = np.zeros(n_nodes, dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.int64)

    nc1 = connectivity[:, 0]
    nc2 = connectivity[:, 1]
    nc3 = connectivity[:, 2]
    nc4 = connectivity[:, 3]

    fi1 = fill[nc1]
    fi2 = fill[nc2]
    fi3 = fill[nc3]
    fi4 = fill[nc4]

    # Element filling rate adjustment
    abf = np.abs(fi1) + np.abs(fi2) + np.abs(fi3) + np.abs(fi4)
    np_nodes = ((fi1 > 0).astype(int) + (fi2 > 0).astype(int) +
                (fi3 > 0).astype(int) + (fi4 > 0).astype(int))
    dn = dt * np_nodes
    fa = np.zeros(len(connectivity), dtype=np.float64)
    mask = dn > 0.0
    fa[mask] = -dalph[mask] * abf[mask] / dn[mask]

    # Relative velocity: vd = v - w
    vd = v - w
    vdy1 = vd[nc1, 0]
    vdz1 = vd[nc1, 1]
    vdy2 = vd[nc2, 0]
    vdz2 = vd[nc2, 1]
    vdy3 = vd[nc3, 0]
    vdz3 = vd[nc3, 1]
    vdy4 = vd[nc4, 0]
    vdz4 = vd[nc4, 1]

    # Mean relative velocity in element
    p1 = fi1 + 1.0
    p2 = fi2 + 1.0
    p3 = fi3 + 1.0
    p4 = fi4 + 1.0
    pt = np.maximum(_EM15, p1 + p2 + p3 + p4)
    vdy = (vdy1 * p1 + vdy2 * p2 + vdy3 * p3 + vdy4 * p4) / pt
    vdz = (vdz1 * p1 + vdz2 * p2 + vdz3 * p3 + vdz4 * p4) / pt

    # Element geometry derivatives
    x1, y1 = coords[nc1, 0], coords[nc1, 1]
    x2, y2 = coords[nc2, 0], coords[nc2, 1]
    x3, y3 = coords[nc3, 0], coords[nc3, 1]
    x4, y4 = coords[nc4, 0], coords[nc4, 1]

    psy = -x1 + x2 + x3 - x4
    psz = -y1 + y2 + y3 - y4
    pty = -x1 - x2 + x3 + x4
    ptz = -y1 - y2 + y3 + y4

    pst = psy * ptz - psz * pty
    pts = -pst
    safe_pst = np.where(np.abs(pst) > 1e-14, pst, 1e-14)
    safe_pts = np.where(np.abs(pts) > 1e-14, pts, 1e-14)

    ds0 = -4.0 * (pty * vdz - ptz * vdy) / safe_pts
    dt0 = -4.0 * (psy * vdz - psz * vdy) / safe_pst

    # Node 1 variation
    ds = np.where(fi1 >= 0.0, -4.0 * (pty * vdz1 - ptz * vdy1) / safe_pts, ds0)
    dt_val = np.where(fi1 >= 0.0, -4.0 * (psy * vdz1 - psz * vdy1) / safe_pst, dt0)
    ds = np.maximum(0.0, 2.0 * ds)
    dt_val = np.maximum(0.0, 2.0 * dt_val)
    df1 = 0.25 * ((-2.0 * ds - 2.0 * dt_val + ds * dt_val * dt) * fi1 +
                  (2.0 * ds - ds * dt_val * dt) * fi2 +
                  (ds * dt_val * dt) * fi3 +
                  (2.0 * dt_val - ds * dt_val * dt) * fi4)

    # Node 2 variation
    ds = np.where(fi2 >= 0.0, -4.0 * (pty * vdz2 - ptz * vdy2) / safe_pts, ds0)
    dt_val = np.where(fi2 >= 0.0, -4.0 * (psy * vdz2 - psz * vdy2) / safe_pst, dt0)
    ds = np.minimum(0.0, 2.0 * ds)
    dt_val = np.maximum(0.0, 2.0 * dt_val)
    df2 = 0.25 * ((-2.0 * ds + ds * dt_val * dt) * fi1 +
                  (2.0 * ds - 2.0 * dt_val - ds * dt_val * dt) * fi2 +
                  (2.0 * dt_val + ds * dt_val * dt) * fi3 -
                  (ds * dt_val * dt) * fi4)

    # Node 3 variation
    ds = np.where(fi3 >= 0.0, -4.0 * (pty * vdz3 - ptz * vdy3) / safe_pts, ds0)
    dt_val = np.where(fi3 >= 0.0, -4.0 * (psy * vdz3 - psz * vdy3) / safe_pst, dt0)
    ds = np.minimum(0.0, 2.0 * ds)
    dt_val = np.minimum(0.0, 2.0 * dt_val)
    df3 = 0.25 * ((ds * dt_val * dt) * fi1 +
                  (-2.0 * dt_val - ds * dt_val * dt) * fi2 +
                  (2.0 * ds + 2.0 * dt_val + ds * dt_val * dt) * fi3 +
                  (-2.0 * ds - ds * dt_val * dt) * fi4)

    # Node 4 variation
    ds = np.where(fi4 >= 0.0, -4.0 * (pty * vdz4 - ptz * vdy4) / safe_pts, ds0)
    dt_val = np.where(fi4 >= 0.0, -4.0 * (psy * vdz4 - psz * vdy4) / safe_pst, dt0)
    ds = np.maximum(0.0, 2.0 * ds)
    dt_val = np.minimum(0.0, 2.0 * dt_val)
    df4 = 0.25 * ((-2.0 * dt_val + ds * dt_val * dt) * fi1 -
                  (ds * dt_val * dt) * fi2 +
                  (2.0 * ds + ds * dt_val * dt) * fi3 +
                  (-2.0 * ds + 2.0 * dt_val - ds * dt_val * dt) * fi4)

    # Accumulate onto nodes
    for j, df_j in [(0, df1), (1, df2), (2, df3), (3, df4)]:
        nodes = connectivity[:, j]
        np.add.at(dfill, nodes, df_j - fa)
        np.add.at(node_count, nodes, 1)

    return dfill, node_count


def bimat_nodal_fill_update(
    fill: np.ndarray,
    dfill: np.ndarray,
    node_count: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Update and clamp nodal fill function values.

    Ported from $OR_SRC/engine/source/ale/bimat/bmultn.F
    lines 51-70:
      For node N:
        If node_count(N) > 0:
          DFILL(N) = DT2 * DFILL(N) / node_count(N)
          FILL(N)  = clip(FILL(N) + DFILL(N), -1.0, 1.0)

    Args:
        fill: (n_nodes,) current fill function values.
        dfill: (n_nodes,) accumulated variations.
        node_count: (n_nodes,) number of adjacent elements.
        dt: time step.

    Returns:
        fill_updated: (n_nodes,) updated fill values in [-1, 1].
    """
    fill_new = np.copy(fill)
    mask = node_count > 0
    delta = np.zeros_like(dfill)
    delta[mask] = dt * dfill[mask] / node_count[mask]
    fill_new[mask] = np.clip(fill[mask] + delta[mask], -1.0, 1.0)
    return fill_new


# =============================================================================
# Full Bi-Material VOF Manager Class
# =============================================================================

class BiMaterialVOF:
    """Bi-material VOF Engine for 2D Quad ALE Elements.

    Ported from OpenRadioss Fortran sources:
    - engine/source/ale/bimat/balph2.F
    - engine/source/ale/bimat/amulf2.F
    - engine/source/ale/bimat/bcumu2.F
    - engine/source/ale/bimat/bafil2.F
    - engine/source/ale/bimat/bmultn.F
    - engine/source/ale/bimat/blero2.F

    Manages a 2-phase bi-material domain, tracking:
    1. Continuous nodal level-set-like fill function phi in [-1, 1]
    2. Element volume fractions alpha_1, alpha_2
    3. Phase face fluxes and volume advection
    4. Homogenized mixture properties (stress, density, internal energy)
    """

    def __init__(
        self,
        connectivity: np.ndarray,
        coords: np.ndarray,
        initial_nodal_fill: np.ndarray,
        rho1: float,
        rho2: float,
        bulk1: float,
        bulk2: float,
        upwind_param: float = 0.0,
        c11: float = 1.0,
        c12: float = 1.0,
    ) -> None:
        self.connectivity = np.array(connectivity, dtype=np.int64)
        self.coords = np.array(coords, dtype=np.float64)
        self.n_elem = len(connectivity)
        self.n_nodes = len(coords)

        self.rho1_ref = rho1
        self.rho2_ref = rho2
        self.bulk1_ref = bulk1
        self.bulk2_ref = bulk2
        self.upwind_param = upwind_param
        self.c11 = c11
        self.c12 = c12

        # Nodal fill function
        self.fill = np.array(initial_nodal_fill, dtype=np.float64)
        self.dfill = np.zeros(self.n_nodes, dtype=np.float64)

        # Initial element volume fractions from nodal fill
        self.alpha1 = elements_vof_from_nodal_fill(self.fill, self.connectivity)
        self.alpha2 = 1.0 - self.alpha1

        # Element volumes
        self.elem_volumes = self._compute_quad_areas()
        self.vol1 = self.alpha1 * self.elem_volumes
        self.vol2 = self.alpha2 * self.elem_volumes

        # Phase masses
        self.mass1 = self.vol1 * self.rho1_ref
        self.mass2 = self.vol2 * self.rho2_ref

        # State arrays
        self.rho1 = np.full(self.n_elem, self.rho1_ref, dtype=np.float64)
        self.rho2 = np.full(self.n_elem, self.rho2_ref, dtype=np.float64)
        self.sig1 = np.zeros((self.n_elem, 6), dtype=np.float64)
        self.sig2 = np.zeros((self.n_elem, 6), dtype=np.float64)
        self.eint1 = np.zeros(self.n_elem, dtype=np.float64)
        self.eint2 = np.zeros(self.n_elem, dtype=np.float64)

        # Mixture arrays
        self.mixture_state: Dict[str, np.ndarray] = {}
        self.compute_mixture_state()

    def _compute_quad_areas(self) -> np.ndarray:
        """Compute quadrilateral element areas via cross product."""
        p0 = self.coords[self.connectivity[:, 0]]
        p1 = self.coords[self.connectivity[:, 1]]
        p2 = self.coords[self.connectivity[:, 2]]
        p3 = self.coords[self.connectivity[:, 3]]
        area1 = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                             (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
        area2 = 0.5 * np.abs((p2[:, 0] - p0[:, 0]) * (p3[:, 1] - p0[:, 1]) -
                             (p2[:, 1] - p0[:, 1]) * (p3[:, 0] - p0[:, 0]))
        return area1 + area2

    def compute_bimat_face_fluxes(
        self,
        flux_total: np.ndarray,
        neighbor_elem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute face volume fluxes for both phases using AMULF2.

        Returns:
            flux1, flu1_1, flux2, flu1_2
        """
        fl1, flu1_1 = bimat_face_fluxes(
            fill=self.fill,
            dfill=self.dfill,
            connectivity=self.connectivity,
            flux_total=flux_total,
            vol=self.elem_volumes,
            alph=self.alpha1,
            neighbor_elem=neighbor_elem,
            upwind_param=self.upwind_param,
        )
        fl2, flu1_2 = bimat_face_fluxes(
            fill=-self.fill,  # Inverted fill for phase 2
            dfill=-self.dfill,
            connectivity=self.connectivity,
            flux_total=flux_total,
            vol=self.elem_volumes,
            alph=self.alpha2,
            neighbor_elem=neighbor_elem,
            upwind_param=self.upwind_param,
        )
        return fl1, flu1_1, fl2, flu1_2

    def step_advection(
        self,
        flux_total: np.ndarray,
        neighbor_elem: np.ndarray,
        v_mat: np.ndarray,
        w_mesh: np.ndarray,
        dt: float,
        strain_rate_trace: Optional[np.ndarray] = None,
    ) -> None:
        """Perform one complete ALE advection and VOF evolution step.

        1. Computes phase face fluxes via AMULF2
        2. Advances volume fractions via BALPH2
        3. Enforces two-material packing constraint
        4. Convects and updates nodal fill function via BAFIL2 + BMULTN
        5. Updates phase volumes, densities, and mixture state via BLERO2 + BCUMU2
        """
        # 1. Phase face fluxes
        fl1, flu1_1, fl2, flu1_2 = self.compute_bimat_face_fluxes(flux_total, neighbor_elem)

        # 2. VOF evolution for phase 1 and phase 2
        sig_tr1 = np.sum(self.sig1[:, :3], axis=1)
        sig_tr2 = np.sum(self.sig2[:, :3], axis=1)

        a1_new, da1 = bimat_vof_evolution(
            alph_old=self.alpha1,
            vol_old=self.vol1,
            vol_new=self.elem_volumes,
            face_fluxes=fl1,
            flu1=flu1_1,
            dt=dt,
            strain_rate_trace=strain_rate_trace,
            stress_trace=sig_tr1,
            rho_new=self.rho1,
        )

        a2_new, da2 = bimat_vof_evolution(
            alph_old=self.alpha2,
            vol_old=self.vol2,
            vol_new=self.elem_volumes,
            face_fluxes=fl2,
            flu1=flu1_2,
            dt=dt,
            strain_rate_trace=strain_rate_trace,
            stress_trace=sig_tr2,
            rho_new=self.rho2,
        )

        # 3. Two-material packing constraint
        self.alpha1, self.alpha2 = bimat_two_material_constraint(
            a1_new, a2_new, c11=self.c11, c12=self.c12
        )

        # 4. Nodal fill convection & update
        dfill, node_count = bimat_nodal_fill_convection(
            v=v_mat,
            w=w_mesh,
            fill=self.fill,
            dalph=da1,
            connectivity=self.connectivity,
            coords=self.coords,
            dt=dt,
        )
        self.dfill = dfill
        self.fill = bimat_nodal_fill_update(self.fill, dfill, node_count, dt)

        # 5. Phase volumes and densities update (BLERO2)
        self.vol1 = self.alpha1 * self.elem_volumes
        self.vol2 = self.alpha2 * self.elem_volumes

        # Mass change from face fluxes
        net_fl1 = np.sum(fl1, axis=1)
        net_fl2 = np.sum(fl2, axis=1)
        self.mass1 = np.maximum(0.0, self.mass1 - dt * net_fl1 * self.rho1)
        self.mass2 = np.maximum(0.0, self.mass2 - dt * net_fl2 * self.rho2)

        mask1 = self.vol1 > _EM30
        mask2 = self.vol2 > _EM30
        self.rho1[mask1] = self.mass1[mask1] / self.vol1[mask1]
        self.rho1[~mask1] = self.rho1_ref
        self.rho2[mask2] = self.mass2[mask2] / self.vol2[mask2]
        self.rho2[~mask2] = self.rho2_ref

        # 6. Compute mixture properties
        self.compute_mixture_state()

    def compute_mixture_state(self) -> None:
        """Update homogenized mixture state from current phase states."""
        self.mixture_state = bimat_mixture_properties(
            alph1=self.alpha1,
            alph2=self.alpha2,
            sig1=self.sig1,
            sig2=self.sig2,
            eint1=self.eint1,
            eint2=self.eint2,
            rho1=self.rho1,
            rho2=self.rho2,
        )
