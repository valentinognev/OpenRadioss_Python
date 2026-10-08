"""Thermal Contact Interface Conduction and Radiation for OpenRadioss (pyradioss).

Upstream OpenRadioss Fortran References:
----------------------------------------
- TYPE2 tied thermal interface conduction (I2THERM):
  ``engine/source/interfaces/interf/i2therm.F``
- TYPE7 penalty surface thermal contact, conduction, radiation, and friction heating (I7THERM):
  ``engine/source/interfaces/int07/i7therm.F``
- TYPE11 edge-to-edge penalty thermal contact (I11THERM):
  ``engine/source/interfaces/int11/i11therm.F``
- TYPE21 stamping drawbead/tool-blank thermal contact with distance-decay conductivity (I21THERM):
  ``engine/source/interfaces/int21/i21therm.F``
- TYPE25 general contact interface conduction with harmonic mean conductivity (I25THERM):
  ``engine/source/interfaces/int25/i25therm.F``
- Thermal acceleration factor and global parameters:
  ``common_source/modules/mat_elem/glob_therm_mod.F90``

Physics & Formulation:
----------------------
1. Interface Heat Conduction (Fourier Law across contact gap):
   Heat exchange between secondary node s and main surface m:
       Phi = Area_c * (T_m - T_s) * dt * THEACCFACT / R_th
   where R_th is thermal resistance (RSTIF + gap resistance), Area_c is contact area,
   and THEACCFACT is the thermal acceleration factor.
   Reaction on master segment is distributed using interpolation/barycentric shape functions:
       Phi_k = -Phi * H_k,   sum(H_k) = 1  =>  sum(Phi_k) + Phi = 0
   Exact conservation of thermal energy is maintained.

2. Interface Radiation (Stefan-Boltzmann factored form):
   Between surfaces at temperatures T_m and T_s separated by distance PENRAD <= DRAD:
       Phi_rad = FRAD * Area_c * (T_m^4 - T_s^4) * dt * THEACCFACT
               = FRAD * Area_c * (T_m^2 + T_s^2) * (T_m + T_s) * (T_m - T_s) * dt * THEACCFACT

3. Friction Heat Dissipation & Partitioning:
   Mechanical frictional dissipation E_frict is partitioned between secondary and main:
       Phi_s = FHEATS * E_frict
       Phi_m = FHEATM * E_frict
       Phi_m,k = Phi_m * H_k

4. Pressure-Dependent Thermal Resistance (IFUNCTK):
   Contact conductivity scales with normal contact pressure P = XTHE * |Fn| / Area_c:
       RSTIFF = RSTIF / max(EM30, fct(P))

5. Distance-Decay Conductivity (FCOND, TYPE21 & TYPE25):
   As contact gap opens between 0 and DDCOND:
       h_cond = fct(penrad / DDCOND) / R_th
       Phi_cond = Area_c * (T_m - T_s) * dt * h_cond * THEACCFACT
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM15, EM20, EM30
from ..common.fastmath import cross3, norm3


def _eval_function_or_curve(fct: Any, x_val: float) -> float:
    """Evaluate a function or curve at abscissa x_val (equivalent to finter.F)."""
    if fct is None:
        return 1.0
    if hasattr(fct, "eval"):
        return float(fct.eval(x_val))
    if hasattr(fct, "interpolate"):
        return float(fct.interpolate(x_val))
    if callable(fct):
        return float(fct(x_val))
    if hasattr(fct, "x") and hasattr(fct, "y"):
        return float(np.interp(x_val, fct.x, fct.y))
    return 1.0


# ============================================================================
# 1. TYPE2 Tied Thermal Interface Conduction (i2therm.F)
# ============================================================================

def thermal_contact_type2(
    x: np.ndarray,
    temp: np.ndarray,
    slave_nodes: np.ndarray,
    master_segs: np.ndarray,
    weights: np.ndarray,
    kthe: float,
    dt: float,
    theaccfact: float = 1.0,
    areas: Optional[np.ndarray] = None,
    node_weights: Optional[np.ndarray] = None,
    active_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute thermal conduction across /INTER/TYPE2 tied contact interface.

    Ported from $OR_SRC/engine/source/interfaces/interf/i2therm.F

    Parameters
    ----------
    x : np.ndarray
        Nodal coordinates (N_nodes, 3).
    temp : np.ndarray
        Nodal temperatures (N_nodes,).
    slave_nodes : np.ndarray
        Secondary node indices (N_pairs,).
    master_segs : np.ndarray
        Master segment corner node indices (N_pairs, 4). For triangle segments,
        master_segs[:, 2] == master_segs[:, 3].
    weights : np.ndarray
        Interpolation / shape function weights H_1, H_2, H_3, H_4 (N_pairs, 4).
    kthe : float
        Thermal contact conductivity KTHE [W/(m^2*K)].
    dt : float
        Explicit time step dt.
    theaccfact : float
        Thermal acceleration factor (default 1.0).
    areas : Optional[np.ndarray]
        Prescribed secondary tributary areas (N_pairs,). If None, calculated from segment.
    node_weights : Optional[np.ndarray]
        Nodal weighting factors W (N_pairs,), default 1.
    active_mask : Optional[np.ndarray]
        Active pair mask (N_pairs,), default all True.

    Returns
    -------
    fthe : np.ndarray
        Nodal thermal energy increments [J] (N_nodes,).
    condn : np.ndarray
        Accumulated nodal thermal conductance [W/K] (N_nodes,).
    heat_transferred : float
        Total thermal energy transferred from master to slave [J].
    """
    n_nodes = len(temp)
    fthe = np.zeros(n_nodes, dtype=float)
    condn = np.zeros(n_nodes, dtype=float)

    n_pairs = len(slave_nodes)
    if n_pairs == 0 or dt <= 0.0 or kthe <= 0.0:
        return fthe, condn, 0.0

    slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
    master_segs = np.asarray(master_segs, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)

    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
    else:
        active = np.ones(n_pairs, dtype=bool)

    if not np.any(active):
        return fthe, condn, 0.0

    act_idx = np.where(active)[0]
    s_nodes = slave_nodes[act_idx]
    m_segs = master_segs[act_idx]
    h_w = weights[act_idx]

    ix1 = m_segs[:, 0]
    ix2 = m_segs[:, 1]
    ix3 = m_segs[:, 2]
    ix4 = m_segs[:, 3]

    # Segment area calculation matching i2therm.F lines 93-104:
    # AX1 = X(1, IX3) - X(1, IX1)
    # AX2 = X(1, IX4) - X(1, IX2)
    # AX  = AY1*AZ2 - AZ1*AY2 ...
    # AREAM = 1/8 * SQRT(AX^2 + AY^2 + AZ^2)
    d13 = x[ix3] - x[ix1]
    d24 = x[ix4] - x[ix2]
    cross_vec = np.cross(d13, d24)
    aream = 0.125 * np.linalg.norm(cross_vec, axis=1)

    if areas is not None:
        areas_act = np.asarray(areas, dtype=float)[act_idx]
        areac = np.where(areas_act > 0.0, np.minimum(areas_act, aream), aream)
    else:
        areac = aream

    # Temperatures
    temps = temp[s_nodes]
    tempm = (
        h_w[:, 0] * temp[ix1]
        + h_w[:, 1] * temp[ix2]
        + h_w[:, 2] * temp[ix3]
        + h_w[:, 3] * temp[ix4]
    )

    # Thermal conductance and heat flux increment (i2therm.F lines 110-113)
    # CONDINT = AREAC * KTHE * THEACCFACT
    # PHI = AREAC * (TEMPM - TEMPS) * DT1 * KTHE * THEACCFACT
    condint = areac * kthe * theaccfact
    phi = condint * (tempm - temps) * dt

    # Master partition: PHI_k = -PHI * H_k (i2therm.F lines 115-118)
    phi1 = -phi * h_w[:, 0]
    phi2 = -phi * h_w[:, 1]
    phi3 = -phi * h_w[:, 2]
    phi4 = -phi * h_w[:, 3]

    # Accumulate into fthe
    np.add.at(fthe, s_nodes, phi)
    np.add.at(fthe, ix1, phi1)
    np.add.at(fthe, ix2, phi2)
    np.add.at(fthe, ix3, phi3)
    np.add.at(fthe, ix4, phi4)

    # Accumulate conductance CONDN (i2therm.F lines 122, 130-133)
    w_factor = 1.0
    if node_weights is not None:
        w_factor = np.asarray(node_weights, dtype=float)[act_idx]
    np.add.at(condn, s_nodes, condint * w_factor)
    np.add.at(condn, ix1, np.abs(h_w[:, 0]) * condint)
    np.add.at(condn, ix2, np.abs(h_w[:, 1]) * condint)
    np.add.at(condn, ix3, np.abs(h_w[:, 2]) * condint)
    np.add.at(condn, ix4, np.abs(h_w[:, 3]) * condint)

    heat_transferred = float(np.sum(phi))
    return fthe, condn, heat_transferred


# ============================================================================
# 2. TYPE7 Penalty Surface Thermal Interface (i7therm.F)
# ============================================================================

def thermal_contact_type7(
    x: np.ndarray,
    temp: np.ndarray,
    slave_nodes: np.ndarray,
    master_segs: np.ndarray,
    weights: np.ndarray,
    kthe: float,
    dt: float,
    theaccfact: float = 1.0,
    iform: int = 1,
    tint: float = 293.15,
    gapv: Optional[np.ndarray] = None,
    frad: float = 0.0,
    drad: float = 0.0,
    fni: Optional[np.ndarray] = None,
    fct_k: Optional[Any] = None,
    xthe: float = 1.0,
    mat_cond: Optional[np.ndarray] = None,
    fheats: float = 0.0,
    fheatm: float = 0.0,
    efrict: Optional[np.ndarray] = None,
    areas: Optional[np.ndarray] = None,
    distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute thermal conduction, radiation, and friction heating for /INTER/TYPE7.

    Ported from $OR_SRC/engine/source/interfaces/int07/i7therm.F

    Parameters
    ----------
    x : np.ndarray
        Nodal coordinates (N_nodes, 3).
    temp : np.ndarray
        Nodal temperatures (N_nodes,).
    slave_nodes : np.ndarray
        Secondary node indices (N_pairs,).
    master_segs : np.ndarray
        Master segment corners (N_pairs, 4). Triangles have seg[:, 2] == seg[:, 3].
    weights : np.ndarray
        Barycentric weights H_1, H_2, H_3, H_4 (N_pairs, 4) of projection.
    kthe : float
        Interface thermal conductance KTHE (or thermal resistance RSTIF = 1 / KTHE).
    dt : float
        Explicit time step dt.
    theaccfact : float
        Thermal acceleration factor (default 1.0).
    iform : int
        0 = exchange with ambient/interior tint, 1 = exchange between slave and master surface.
    tint : float
        Prescribed ambient temperature for iform = 0.
    gapv : Optional[np.ndarray]
        Contact gap per pair (N_pairs,).
    frad : float
        Radiation coefficient FRAD = emissivity * sigma.
    drad : float
        Radiation cutoff distance (radiation active if gapv <= penrad <= drad).
    fni : Optional[np.ndarray]
        Normal contact force per pair (N_pairs,) for pressure-dependent conductivity.
    fct_k : Optional[Any]
        Function/curve for conductivity vs contact pressure P = XTHE * |FNI| / Area_c.
    xthe : float
        Scale factor on contact pressure for conductivity function (default 1.0).
    mat_cond : Optional[np.ndarray]
        Thermal conductivity of element material per pair (N_pairs,).
    fheats : float
        Fraction of friction energy dissipated into secondary node.
    fheatm : float
        Fraction of friction energy dissipated into master surface.
    efrict : Optional[np.ndarray]
        Frictional energy dissipation per pair over dt (N_pairs,).
    areas : Optional[np.ndarray]
        Contact areas (N_pairs,). If None, computed as 0.5 * |normal|.
    distances : Optional[np.ndarray]
        Signed normal distances (N_pairs,). If None, computed from node and segment center.

    Returns
    -------
    fthe : np.ndarray
        Nodal thermal energy increments [J] (N_nodes,).
    condint : np.ndarray
        Pair thermal conductance [W/K] (N_pairs,).
    ledger : Dict[str, float]
        Dictionary with 'conduction', 'radiation', and 'friction' cumulative work/heat.
    """
    n_nodes = len(temp)
    fthe = np.zeros(n_nodes, dtype=float)
    n_pairs = len(slave_nodes)
    condint = np.zeros(n_pairs, dtype=float)
    ledger = {"conduction": 0.0, "radiation": 0.0, "friction": 0.0}

    if n_pairs == 0 or dt <= 0.0:
        return fthe, condint, ledger

    slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
    master_segs = np.asarray(master_segs, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)

    ix1 = master_segs[:, 0]
    ix2 = master_segs[:, 1]
    ix3 = master_segs[:, 2]
    ix4 = master_segs[:, 3]

    x1 = x[ix1]
    x2 = x[ix2]
    x3 = x[ix3]
    x4 = x[ix4]
    xs = x[slave_nodes]

    # Surface vector (*2) i7therm.F lines 103-107:
    # SX1 = (Y1-Y3)*(Z2-Z4) - (Z1-Z3)*(Y2-Y4)
    # NORM = SQRT(SX1^2 + SY1^2 + SZ1^2)
    v13 = x1 - x3
    v24 = x2 - x4
    s_vec = np.cross(v13, v24)
    norm_s = np.linalg.norm(s_vec, axis=1)
    norm_clamped = np.maximum(norm_s, EM15)

    # Segment center (i7therm.F lines 111-119)
    is_tri = (ix3 == ix4)
    center = np.where(
        is_tri[:, None],
        (x1 + x2 + x3) / 3.0,
        (x1 + x2 + x3 + x4) / 4.0,
    )

    if distances is not None:
        dist = np.asarray(distances, dtype=float)
    else:
        # Vector between center and secondary node: SX2 = center - XI (lines 112-118, 125)
        sc = center - xs
        dist = np.sum(sc * s_vec, axis=1) / norm_clamped

    penrad = np.abs(dist)

    # Contact area (i7therm.F lines 133-137)
    if areas is not None:
        areas_arr = np.asarray(areas, dtype=float)
        areac = np.where(areas_arr > 0.0, areas_arr, 0.5 * norm_s)
    else:
        areac = 0.5 * norm_s

    gap_arr = np.asarray(gapv, dtype=float) if gapv is not None else np.zeros(n_pairs, dtype=float)

    rstif = 1.0 / max(EM30, kthe) if kthe > 0.0 else 1.0 / EM30

    # Temperatures
    ts = temp[slave_nodes]
    if iform == 0:
        tm = np.full(n_pairs, tint, dtype=float)
    else:
        tm = (
            weights[:, 0] * temp[ix1]
            + weights[:, 1] * temp[ix2]
            + weights[:, 2] * temp[ix3]
            + weights[:, 3] * temp[ix4]
        )

    # Thermal contact resistance (i7therm.F lines 274-275, 312-313)
    if fct_k is not None and fni is not None:
        fni_arr = np.abs(np.asarray(fni, dtype=float))
        p_cont = xthe * fni_arr / np.maximum(areac, EM20)
        f_val = np.array([_eval_function_or_curve(fct_k, p) for p in p_cont])
        rstiff = rstif / np.maximum(EM30, f_val)
    else:
        rstiff = np.full(n_pairs, rstif, dtype=float)

    # Material gap resistance TSTIFM (i7therm.F lines 153, 278)
    if mat_cond is not None:
        m_cond = np.asarray(mat_cond, dtype=float)
        tstifm = np.where(m_cond > 0.0, np.maximum(dist, 0.0) / np.maximum(m_cond, EM20), 0.0)
    else:
        tstifm = np.zeros(n_pairs, dtype=float)

    tstift = tstifm + rstiff

    # Regime selection: Radiation vs Conduction (i7therm.F lines 140-163, 181-193)
    is_rad = (penrad <= drad) & (penrad >= gap_arr) & (drad > 0.0) & (frad > 0.0)

    # Stefan-Boltzmann radiation: FRAD * AREAC * (TM^4 - TS^4) * DT * THEACCFACT
    # Note: (TM^2 + TS^2) * (TM + TS) * (TM - TS) == TM^4 - TS^4
    phi_rad = frad * areac * (tm**4 - ts**4) * dt * theaccfact

    # Conduction: AREAC * (TM - TS) * DT * THEACCFACT / TSTIFT
    phi_cond = areac * (tm - ts) * dt * theaccfact / np.maximum(tstift, EM20)
    condint_vals = areac * theaccfact / np.maximum(tstift, EM20)

    phi = np.where(is_rad, phi_rad, phi_cond)
    condint[:] = np.where(is_rad, 0.0, condint_vals)

    ledger["conduction"] = float(np.sum(np.where(~is_rad, phi_cond, 0.0)))
    ledger["radiation"] = float(np.sum(np.where(is_rad, phi_rad, 0.0)))

    # Master partition: PHI_k = -PHI * H_k (i7therm.F lines 194-197)
    phi1 = -phi * weights[:, 0]
    phi2 = -phi * weights[:, 1]
    phi3 = -phi * weights[:, 2]
    phi4 = -phi * weights[:, 3]

    # Friction heating (i7therm.F lines 201-207)
    if efrict is not None and (fheats != 0.0 or fheatm != 0.0):
        ef = np.asarray(efrict, dtype=float)
        phi_fric_s = fheats * ef
        phi_fric_m = fheatm * ef
        phi += phi_fric_s
        if iform == 1:
            phi1 += phi_fric_m * weights[:, 0]
            phi2 += phi_fric_m * weights[:, 1]
            phi3 += phi_fric_m * weights[:, 2]
            phi4 += phi_fric_m * weights[:, 3]
        ledger["friction"] = float(np.sum(ef * (fheats + fheatm)))

    # Assemble into fthe
    np.add.at(fthe, slave_nodes, phi)
    if iform == 1:
        np.add.at(fthe, ix1, phi1)
        np.add.at(fthe, ix2, phi2)
        np.add.at(fthe, ix3, phi3)
        np.add.at(fthe, ix4, phi4)

    return fthe, condint, ledger


# ============================================================================
# 3. TYPE11 Edge-to-Edge Thermal Interface (i11therm.F)
# ============================================================================

def thermal_contact_type11(
    temp: np.ndarray,
    slave_edge_nodes: np.ndarray,
    master_edge_nodes: np.ndarray,
    hs: np.ndarray,
    hm: np.ndarray,
    kthe: float,
    dt: float,
    areac: np.ndarray,
    penrad: np.ndarray,
    gapv: Optional[np.ndarray] = None,
    frad: float = 0.0,
    drad: float = 0.0,
    iform: int = 1,
    tint: float = 293.15,
    mat_cond: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute thermal conduction and radiation across /INTER/TYPE11 edge-to-edge contact.

    Ported from $OR_SRC/engine/source/interfaces/int11/i11therm.F

    Parameters
    ----------
    temp : np.ndarray
        Nodal temperatures (N_nodes,).
    slave_edge_nodes : np.ndarray
        Secondary edge end node indices (N_pairs, 2) [S1, S2].
    master_edge_nodes : np.ndarray
        Master edge end node indices (N_pairs, 2) [M1, M2].
    hs : np.ndarray
        Secondary edge interpolation weights (N_pairs, 2) [HS1, HS2], sum to 1.
    hm : np.ndarray
        Master edge interpolation weights (N_pairs, 2) [HM1, HM2], sum to 1.
    kthe : float
        Thermal interface conductivity KTHE.
    dt : float
        Explicit time step dt.
    areac : np.ndarray
        Contact tributary area for each edge pair (N_pairs,).
    penrad : np.ndarray
        Signed distance / penetration (N_pairs,).
    gapv : Optional[np.ndarray]
        Edge gap per pair (N_pairs,).
    frad : float
        Radiation coefficient.
    drad : float
        Radiation cutoff distance.
    iform : int
        0 = exchange with tint, 1 = exchange between slave and master edge.
    tint : float
        Prescribed ambient temperature if iform = 0.
    mat_cond : Optional[np.ndarray]
        Material conductivity per pair.

    Returns
    -------
    fthe : np.ndarray
        Nodal thermal energy increments [J] (N_nodes,).
    condint : np.ndarray
        Pair thermal conductance [W/K] (N_pairs,).
    heat_transferred : float
        Total thermal energy transferred [J].
    """
    n_nodes = len(temp)
    fthe = np.zeros(n_nodes, dtype=float)
    n_pairs = len(slave_edge_nodes)
    condint = np.zeros(n_pairs, dtype=float)

    if n_pairs == 0 or dt <= 0.0:
        return fthe, condint, 0.0

    s1 = slave_edge_nodes[:, 0]
    s2 = slave_edge_nodes[:, 1]
    m1 = master_edge_nodes[:, 0]
    m2 = master_edge_nodes[:, 1]

    hs1 = hs[:, 0]
    hs2 = hs[:, 1]
    hm1 = hm[:, 0]
    hm2 = hm[:, 1]

    ts1 = temp[s1]
    ts2 = temp[s2]
    ts = hs1 * ts1 + hs2 * ts2

    if iform == 0:
        tm = np.full(n_pairs, tint, dtype=float)
    else:
        tm1 = temp[m1]
        tm2 = temp[m2]
        tm = hm1 * tm1 + hm2 * tm2

    gap_arr = np.asarray(gapv, dtype=float) if gapv is not None else np.zeros(n_pairs, dtype=float)
    rstif = 1.0 / max(EM30, kthe) if kthe > 0.0 else 1.0 / EM30

    if mat_cond is not None:
        cond_val = np.asarray(mat_cond, dtype=float)
        dist = penrad + gap_arr
        tstifm = np.where(cond_val > 0.0, np.maximum(dist, 0.0) / np.maximum(cond_val, EM20), 0.0)
    else:
        tstifm = np.zeros(n_pairs, dtype=float)

    tstift = tstifm + rstif

    # i11therm.F lines 229-262:
    # Conduction if penrad <= 0
    # Radiation if 0 < penrad <= drad
    is_cond = (penrad <= 0.0)
    is_rad = (~is_cond) & (penrad <= drad) & (drad > 0.0) & (frad > 0.0)

    phi_cond = areac * (tm - ts) * dt / np.maximum(tstift, EM20)
    phi_rad = frad * areac * (tm**4 - ts**4) * dt

    phi = np.where(is_cond, phi_cond, np.where(is_rad, phi_rad, 0.0))
    condint[:] = np.where(is_cond, areac / np.maximum(tstift, EM20), 0.0)

    # Nodal distribution (i11therm.F lines 264-267)
    phis1 = hs1 * phi
    phis2 = hs2 * phi
    np.add.at(fthe, s1, phis1)
    np.add.at(fthe, s2, phis2)

    if iform == 1:
        phim1 = -hm1 * phi
        phim2 = -hm2 * phi
        np.add.at(fthe, m1, phim1)
        np.add.at(fthe, m2, phim2)

    heat_transferred = float(np.sum(phi))
    return fthe, condint, heat_transferred


# ============================================================================
# 4. TYPE21 Drawbead/Tool-Blank Thermal Contact (i21therm.F)
# ============================================================================

def thermal_contact_type21(
    temp: np.ndarray,
    slave_nodes: np.ndarray,
    master_segs: np.ndarray,
    weights: np.ndarray,
    kthe: float,
    dt: float,
    theaccfact: float = 1.0,
    areac: Optional[np.ndarray] = None,
    penrad: Optional[np.ndarray] = None,
    gapv: Optional[np.ndarray] = None,
    dcond: float = 0.0,
    fcond: Optional[Any] = None,
    frad: float = 0.0,
    drad: float = 0.0,
    iform: int = 1,
    fheat: float = 0.0,
    efrict: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute thermal contact with distance-decay conductivity for /INTER/TYPE21.

    Ported from $OR_SRC/engine/source/interfaces/int21/i21therm.F

    Parameters
    ----------
    temp : np.ndarray
        Nodal temperatures (N_nodes,).
    slave_nodes : np.ndarray
        Secondary node indices (N_pairs,).
    master_segs : np.ndarray
        Master segment corners (N_pairs, 4).
    weights : np.ndarray
        Master interpolation weights H_1..H_4 (N_pairs, 4).
    kthe : float
        Thermal interface conductivity KTHE.
    dt : float
        Explicit time step dt.
    theaccfact : float
        Thermal acceleration factor (default 1.0).
    areac : Optional[np.ndarray]
        Contact area per pair (N_pairs,).
    penrad : Optional[np.ndarray]
        Distance to contact surface (N_pairs,).
    gapv : Optional[np.ndarray]
        Gap per pair (N_pairs,).
    dcond : float
        Maximum distance for conduction decay (DDCOND = max(DCOND - GAPV, EM20)).
    fcond : Optional[Any]
        Conductance decay curve vs normalized distance DD = PENRAD / DDCOND.
    frad : float
        Radiation coefficient.
    drad : float
        Radiation cutoff distance.
    iform : int
        1 = update master surface nodes, 0 = secondary only.
    fheat : float
        Friction heating partition factor.
    efrict : Optional[np.ndarray]
        Friction energy dissipation per pair over dt.

    Returns
    -------
    fthe : np.ndarray
        Nodal thermal energy increments [J] (N_nodes,).
    condint : np.ndarray
        Pair thermal conductance [W/K] (N_pairs,).
    heat_transferred : float
        Total thermal energy transferred [J].
    """
    n_nodes = len(temp)
    fthe = np.zeros(n_nodes, dtype=float)
    n_pairs = len(slave_nodes)
    condint = np.zeros(n_pairs, dtype=float)

    if n_pairs == 0 or dt <= 0.0:
        return fthe, condint, 0.0

    slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
    master_segs = np.asarray(master_segs, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)

    ix1 = master_segs[:, 0]
    ix2 = master_segs[:, 1]
    ix3 = master_segs[:, 2]
    ix4 = master_segs[:, 3]

    ts = temp[slave_nodes]
    tm = (
        weights[:, 0] * temp[ix1]
        + weights[:, 1] * temp[ix2]
        + weights[:, 2] * temp[ix3]
        + weights[:, 3] * temp[ix4]
    )

    area_arr = np.asarray(areac, dtype=float) if areac is not None else np.ones(n_pairs, dtype=float)
    pen_arr = np.asarray(penrad, dtype=float) if penrad is not None else np.zeros(n_pairs, dtype=float)
    gap_arr = np.asarray(gapv, dtype=float) if gapv is not None else np.zeros(n_pairs, dtype=float)

    rstif = 1.0 / max(EM30, kthe) if kthe > 0.0 else 1.0 / EM30
    ddcond = np.maximum(dcond - gap_arr, EM20)

    # 3 Distance regimes (i21therm.F lines 124-162):
    # 1. Close distance / penetration: PENRAD <= 0
    #    TSTIFT = max(DIST, 0) / COND + RSTIF
    #    PHI = AREAC * (TM - TS) * DT * THEACCFACT / TSTIFT
    # 2. Conduction + Radiation: 0 < PENRAD <= DDCOND
    #    HCOND = fcond(PENRAD / DDCOND) / TSTIFT
    #    PHI = AREAC * (TM - TS) * DT * HCOND * THEACCFACT + FRAD * AREAC * (TM^4 - TS^4) * DT * THEACCFACT
    # 3. Pure Radiation: DDCOND < PENRAD <= DRAD
    phi = np.zeros(n_pairs, dtype=float)

    # Zone 1: Close distance
    mask_close = (pen_arr <= 0.0)
    if np.any(mask_close):
        c1 = area_arr[mask_close] * theaccfact / rstif
        phi[mask_close] = c1 * (tm[mask_close] - ts[mask_close]) * dt
        condint[mask_close] = c1

    # Zone 2: Distance-decay conduction + radiation
    mask_decay = (~mask_close) & (pen_arr <= ddcond)
    if np.any(mask_decay):
        dd = pen_arr[mask_decay] / ddcond[mask_decay]
        if fcond is not None:
            h_factor = np.array([_eval_function_or_curve(fcond, d) for d in dd]) / rstif
        else:
            h_factor = (1.0 - dd) / rstif
        c2 = area_arr[mask_decay] * h_factor * theaccfact
        phi_cond2 = c2 * (tm[mask_decay] - ts[mask_decay]) * dt
        phi_rad2 = (
            frad
            * area_arr[mask_decay]
            * (tm[mask_decay] ** 4 - ts[mask_decay] ** 4)
            * dt
            * theaccfact
        )
        phi[mask_decay] = phi_cond2 + phi_rad2
        condint[mask_decay] = c2

    # Zone 3: Radiation only
    mask_rad = (~mask_close) & (~mask_decay) & (pen_arr <= drad) & (drad > 0.0) & (frad > 0.0)
    if np.any(mask_rad):
        phi[mask_rad] = (
            frad
            * area_arr[mask_rad]
            * (tm[mask_rad] ** 4 - ts[mask_rad] ** 4)
            * dt
            * theaccfact
        )

    # Master partition: PHI_k = -PHI * H_k
    phi1 = -phi * weights[:, 0]
    phi2 = -phi * weights[:, 1]
    phi3 = -phi * weights[:, 2]
    phi4 = -phi * weights[:, 3]

    # Friction heating (i21therm.F lines 172-181)
    if efrict is not None and fheat > 0.0:
        ef = np.asarray(efrict, dtype=float)
        phim = fheat * ef
        phi += fheat * ef * theaccfact
        if iform == 1:
            phi1 += phim * weights[:, 0]
            phi2 += phim * weights[:, 1]
            phi3 += phim * weights[:, 2]
            phi4 += phim * weights[:, 3]

    np.add.at(fthe, slave_nodes, phi)
    if iform == 1:
        np.add.at(fthe, ix1, phi1)
        np.add.at(fthe, ix2, phi2)
        np.add.at(fthe, ix3, phi3)
        np.add.at(fthe, ix4, phi4)

    heat_transferred = float(np.sum(phi))
    return fthe, condint, heat_transferred


# ============================================================================
# 5. TYPE25 General Interface Conduction (i25therm.F)
# ============================================================================

def thermal_contact_type25(
    x: np.ndarray,
    temp: np.ndarray,
    slave_nodes: np.ndarray,
    master_segs: np.ndarray,
    weights: np.ndarray,
    kthe: float,
    dt: float,
    theaccfact: float = 1.0,
    iform: int = 1,
    tint: float = 293.15,
    gapv: Optional[np.ndarray] = None,
    dcond: float = 0.0,
    fcond: Optional[Any] = None,
    frad: float = 0.0,
    drad: float = 0.0,
    cond_slave: Optional[np.ndarray] = None,
    cond_master: Optional[np.ndarray] = None,
    fheats: float = 0.0,
    fheatm: float = 0.0,
    efrict: Optional[np.ndarray] = None,
    areas: Optional[np.ndarray] = None,
    distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute general interface thermal contact with harmonic mean conductivity (/INTER/TYPE25).

    Ported from $OR_SRC/engine/source/interfaces/int25/i25therm.F

    Parameters
    ----------
    x : np.ndarray
        Nodal coordinates (N_nodes, 3).
    temp : np.ndarray
        Nodal temperatures (N_nodes,).
    slave_nodes : np.ndarray
        Secondary node indices (N_pairs,).
    master_segs : np.ndarray
        Master segment corners (N_pairs, 4).
    weights : np.ndarray
        Barycentric weights H_1..H_4 (N_pairs, 4).
    kthe : float
        Contact conductance KTHE.
    dt : float
        Explicit time step dt.
    theaccfact : float
        Thermal acceleration factor (default 1.0).
    iform : int
        0 = exchange with tint, 1 = exchange between slave and master.
    tint : float
        Ambient temperature if iform = 0.
    gapv : Optional[np.ndarray]
        Contact gap (N_pairs,).
    dcond : float
        Conductivity decay threshold.
    fcond : Optional[Any]
        Decay curve vs DD = PENRAD / DDCOND.
    frad : float
        Radiation coefficient.
    drad : float
        Radiation distance threshold.
    cond_slave : Optional[np.ndarray]
        Material thermal conductivity for secondary elements (N_pairs,).
    cond_master : Optional[np.ndarray]
        Material thermal conductivity for master elements (N_pairs,).
    fheats : float
        Friction heating fraction to secondary.
    fheatm : float
        Friction heating fraction to master.
    efrict : Optional[np.ndarray]
        Friction energy dissipation per pair over dt.
    areas : Optional[np.ndarray]
        Contact areas.
    distances : Optional[np.ndarray]
        Normal distances.

    Returns
    -------
    fthe : np.ndarray
        Nodal thermal energy increments [J] (N_nodes,).
    condint : np.ndarray
        Pair thermal conductance [W/K] (N_pairs,).
    ledger : Dict[str, float]
        Dictionary with 'conduction', 'radiation', and 'friction' heat components.
    """
    n_nodes = len(temp)
    fthe = np.zeros(n_nodes, dtype=float)
    n_pairs = len(slave_nodes)
    condint = np.zeros(n_pairs, dtype=float)
    ledger = {"conduction": 0.0, "radiation": 0.0, "friction": 0.0}

    if n_pairs == 0 or dt <= 0.0:
        return fthe, condint, ledger

    slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
    master_segs = np.asarray(master_segs, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)

    ix1 = master_segs[:, 0]
    ix2 = master_segs[:, 1]
    ix3 = master_segs[:, 2]
    ix4 = master_segs[:, 3]

    x1 = x[ix1]
    x2 = x[ix2]
    x3 = x[ix3]
    x4 = x[ix4]
    xs = x[slave_nodes]

    v13 = x1 - x3
    v24 = x2 - x4
    s_vec = np.cross(v13, v24)
    norm_s = np.linalg.norm(s_vec, axis=1)
    norm_clamped = np.maximum(norm_s, EM15)

    is_tri = (ix3 == ix4)
    center = np.where(
        is_tri[:, None],
        (x1 + x2 + x3) / 3.0,
        (x1 + x2 + x3 + x4) / 4.0,
    )

    if distances is not None:
        dist = np.asarray(distances, dtype=float)
    else:
        sc = center - xs
        dist = -np.sum(sc * s_vec, axis=1) / norm_clamped

    gap_arr = np.asarray(gapv, dtype=float) if gapv is not None else np.zeros(n_pairs, dtype=float)
    penrad = dist - gap_arr

    if areas is not None:
        areas_arr = np.asarray(areas, dtype=float)
        areac = np.where(areas_arr > 0.0, areas_arr, 0.5 * norm_s)
    else:
        areac = 0.5 * norm_s

    ts = temp[slave_nodes]
    if iform == 0:
        tm = np.full(n_pairs, tint, dtype=float)
    else:
        tm = (
            weights[:, 0] * temp[ix1]
            + weights[:, 1] * temp[ix2]
            + weights[:, 2] * temp[ix3]
            + weights[:, 3] * temp[ix4]
        )

    # Harmonic mean conductivity (i25therm.F lines 155-177):
    # COND = 2 * CONDS * CONDM / (CONDM + CONDS)
    cs = np.asarray(cond_slave, dtype=float) if cond_slave is not None else np.zeros(n_pairs, dtype=float)
    if iform == 1 and cond_master is not None:
        cm = np.asarray(cond_master, dtype=float)
        denom = cm + cs
        harm_cond = np.where(denom > EM20, 2.0 * cs * cm / denom, np.maximum(cs, cm))
    else:
        harm_cond = cs

    rstif = 1.0 / max(EM30, kthe) if kthe > 0.0 else 1.0 / EM30
    ddcond = np.maximum(dcond - gap_arr, EM20)

    phi = np.zeros(n_pairs, dtype=float)

    # Distance regimes (i25therm.F lines 150-233):
    # 1. PENRAD <= 0: Direct conduction
    mask_close = (penrad <= 0.0)
    if np.any(mask_close):
        cond_k = harm_cond[mask_close]
        tstifm = np.where(cond_k > 0.0, np.abs(dist[mask_close]) / np.maximum(cond_k, EM20), 0.0)
        tstift = tstifm + rstif
        c1 = areac[mask_close] * theaccfact / np.maximum(tstift, EM20)
        phi[mask_close] = c1 * (tm[mask_close] - ts[mask_close]) * dt
        condint[mask_close] = c1

    # 2. 0 < PENRAD <= DDCOND: Conduction decay + radiation
    mask_decay = (~mask_close) & (penrad <= ddcond)
    if np.any(mask_decay):
        cond_k = harm_cond[mask_decay]
        tstifm = np.where(cond_k > 0.0, np.maximum(gap_arr[mask_decay], 0.0) / np.maximum(cond_k, EM20), 0.0)
        tstift = tstifm + rstif
        dd = penrad[mask_decay] / ddcond[mask_decay]
        if fcond is not None:
            h_decay = np.array([_eval_function_or_curve(fcond, d) for d in dd]) / np.maximum(tstift, EM20)
        else:
            h_decay = (1.0 - dd) / np.maximum(tstift, EM20)
        c2 = areac[mask_decay] * h_decay * theaccfact
        phi_cond2 = c2 * (tm[mask_decay] - ts[mask_decay]) * dt
        phi_rad2 = (
            frad
            * areac[mask_decay]
            * (tm[mask_decay] ** 4 - ts[mask_decay] ** 4)
            * dt
            * theaccfact
        )
        phi[mask_decay] = phi_cond2 + phi_rad2
        condint[mask_decay] = c2

    # 3. Radiation only: DDCOND < PENRAD <= DRAD
    mask_rad = (~mask_close) & (~mask_decay) & (penrad <= drad) & (drad > 0.0) & (frad > 0.0)
    if np.any(mask_rad):
        phi[mask_rad] = (
            frad
            * areac[mask_rad]
            * (tm[mask_rad] ** 4 - ts[mask_rad] ** 4)
            * dt
            * theaccfact
        )

    ledger["conduction"] = float(np.sum(np.where(mask_close, phi, 0.0)))
    ledger["radiation"] = float(np.sum(np.where(mask_rad, phi, 0.0)))

    # Master partition
    phi1 = -phi * weights[:, 0]
    phi2 = -phi * weights[:, 1]
    phi3 = -phi * weights[:, 2]
    phi4 = -phi * weights[:, 3]

    # Friction heating (i25therm.F lines 243-253)
    if efrict is not None and (fheats > 0.0 or fheatm > 0.0):
        ef = np.asarray(efrict, dtype=float)
        phim = fheatm * ef
        phi += fheats * ef * theaccfact
        if iform == 1:
            phi1 += phim * weights[:, 0]
            phi2 += phim * weights[:, 1]
            phi3 += phim * weights[:, 2]
            phi4 += phim * weights[:, 3]
        ledger["friction"] = float(np.sum(ef * (fheats + fheatm)))

    np.add.at(fthe, slave_nodes, phi)
    if iform == 1:
        np.add.at(fthe, ix1, phi1)
        np.add.at(fthe, ix2, phi2)
        np.add.at(fthe, ix3, phi3)
        np.add.at(fthe, ix4, phi4)

    return fthe, condint, ledger
