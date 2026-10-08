# pyradioss/engine/ale_2d.py
"""
2D Arbitrary Lagrangian-Eulerian (ALE) Quad Formulation.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/ale2d/aconv2.F: 2D ALE convection update
- engine/source/ale/ale2d/aflux2.F: 2D ALE face flux computation & outward normals
- engine/source/ale/ale2d/agrad2.F: 2D ALE gradient reconstruction & face projections
- engine/source/ale/ale2d/arezo2.F: 2D ALE rezoning/remapping of element variables
- engine/source/ale/ale2d/amomt2.F: 2D ALE momentum convective forces on nodes
- engine/source/ale/ale2d/adiff2.F: 2D ALE finite-volume diffusion with harmonic interpolation
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple, Dict, Set, List, Any


# -----------------------------------------------------------------------------
# 2D Quad Face / Edge Definitions
# Fortran origin: engine/source/ale/ale2d/aflux2.F lines 80-97, 128-140
# -----------------------------------------------------------------------------
# 0-based node indices for the 4 edges of a 4-node quadrilateral:
# Edge 0: [0, 1] (NC1 -> NC2)
# Edge 1: [1, 2] (NC2 -> NC3)
# Edge 2: [2, 3] (NC3 -> NC4)
# Edge 3: [3, 0] (NC4 -> NC1)
QUAD_EDGES = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
], dtype=np.int64)


def build_quad_edge_connectivity(connectivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build element-element neighbor connectivity table across all 4 quad edges.

    Ported from $OR_SRC/engine/source/ale/ale2d/aflux2.F
    lines 187-211 and common_source/modules/ale/ale_connectivity_mod.F.

    Args:
        connectivity: (n_elem, 4) node indices for each 2D quad element.

    Returns:
        neighbor_elem: (n_elem, 4) int64, neighbor element index (-1 if boundary).
        neighbor_edge: (n_elem, 4) int64, edge index on neighbor element (-1 if boundary).
    """
    n_elem = len(connectivity)
    neighbor_elem = np.full((n_elem, 4), -1, dtype=np.int64)
    neighbor_edge = np.full((n_elem, 4), -1, dtype=np.int64)

    edge_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for e in range(n_elem):
        conn_e = connectivity[e]
        for ed_idx in range(4):
            n1 = int(conn_e[QUAD_EDGES[ed_idx, 0]])
            n2 = int(conn_e[QUAD_EDGES[ed_idx, 1]])
            key = (min(n1, n2), max(n1, n2))
            if key in edge_map:
                e_prev, ed_prev = edge_map[key]
                neighbor_elem[e, ed_idx] = e_prev
                neighbor_edge[e, ed_idx] = ed_prev
                neighbor_elem[e_prev, ed_prev] = e
                neighbor_edge[e_prev, ed_prev] = ed_idx
            else:
                edge_map[key] = (e, ed_idx)

    return neighbor_elem, neighbor_edge


def compute_quad_edge_normals(xe: np.ndarray, axisymmetric: bool = False) -> np.ndarray:
    """Compute outward area-normal vectors for all 4 edges of quad elements.

    Ported from $OR_SRC/engine/source/ale/ale2d/aflux2.F
    lines 128-153:
      N1_y = (Z2 - Z1)
      N1_z = -(Y2 - Y1)
      ...
    For standard 2D Cartesian (x, y):
      Edge (0 -> 1): vector (dx, dy) = (x1 - x0, y1 - y0).
      Outward normal: N = (dy, -dx). Length |N| is the edge length.
    If axisymmetric (N2D == 1), normals are scaled by the mean radius 0.5 * (y0 + y1).

    Args:
        xe: (n_elem, 4, 2) nodal coordinates for each quad element.
        axisymmetric: if True, scale by mean radius (Y coordinate = radius).

    Returns:
        normals: (n_elem, 4, 2) outward normal vectors.
    """
    n_elem = len(xe)
    normals = np.zeros((n_elem, 4, 2), dtype=np.float64)
    if n_elem == 0:
        return normals

    p0 = xe[:, 0]  # (n_elem, 2)
    p1 = xe[:, 1]
    p2 = xe[:, 2]
    p3 = xe[:, 3]

    # Edge 0: 0 -> 1: (y1 - y0, -(x1 - x0))
    normals[:, 0, 0] = p1[:, 1] - p0[:, 1]
    normals[:, 0, 1] = -(p1[:, 0] - p0[:, 0])

    # Edge 1: 1 -> 2: (y2 - y1, -(x2 - x1))
    normals[:, 1, 0] = p2[:, 1] - p1[:, 1]
    normals[:, 1, 1] = -(p2[:, 0] - p1[:, 0])

    # Edge 2: 2 -> 3: (y3 - y2, -(x3 - x2))
    normals[:, 2, 0] = p3[:, 1] - p2[:, 1]
    normals[:, 2, 1] = -(p3[:, 0] - p2[:, 0])

    # Edge 3: 3 -> 0: (y0 - y3, -(x0 - x3))
    normals[:, 3, 0] = p0[:, 1] - p3[:, 1]
    normals[:, 3, 1] = -(p0[:, 0] - p3[:, 0])

    if axisymmetric:
        # aflux2.F lines 142-153: multiply by 0.5 * (Y_a + Y_b)
        r0 = p0[:, 1]
        r1 = p1[:, 1]
        r2 = p2[:, 1]
        r3 = p3[:, 1]
        normals[:, 0] *= (0.5 * (r0 + r1))[:, None]
        normals[:, 1] *= (0.5 * (r1 + r2))[:, None]
        normals[:, 2] *= (0.5 * (r2 + r3))[:, None]
        normals[:, 3] *= (0.5 * (r3 + r0))[:, None]

    return normals


def compute_quad_areas(x: np.ndarray, connectivity: np.ndarray, axisymmetric: bool = False) -> np.ndarray:
    """Compute areas (or axisymmetric volumes) of 4-node quad elements.

    Ported from $OR_SRC/engine/source/ale/grid/alew6.F
    lines 124-138:
      A1 = Y2*(Z3-Z4) + Y3*(Z4-Z2) + Y4*(Z2-Z3)
      A2 = Y2*(Z4-Z1) + Y4*(Z1-Z2) + Y1*(Z2-Z4)
      Area = 0.5 * (A1 + A2)

    Args:
        x: (n_nodes, 2) nodal coordinates.
        connectivity: (n_elem, 4) element connectivity.
        axisymmetric: if True, computes volume V = Area * 2 * pi * r_centroid (or per rad).

    Returns:
        areas: (n_elem,) positive element areas (or volumes).
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return np.empty(0, dtype=np.float64)

    xe = x[connectivity]  # (n_elem, 4, 2)
    p0, p1, p2, p3 = xe[:, 0], xe[:, 1], xe[:, 2], xe[:, 3]

    # Shoelace / diagonal cross product
    # Area = 0.5 * |(x2 - x0)*(y3 - y1) - (x3 - x1)*(y2 - y0)|
    d1 = p2 - p0
    d2 = p3 - p1
    areas = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    areas = np.maximum(areas, 1e-30)

    if axisymmetric:
        # Centroid radius r_c = 1/4 * sum(y_i)
        r_c = np.mean(xe[:, :, 1], axis=1)
        areas = areas * np.maximum(r_c, 1e-15)

    return areas


def ale_2d_compute_gradients(x: np.ndarray,
                             connectivity: np.ndarray,
                             neighbor_elem: np.ndarray,
                             axisymmetric: bool = False) -> np.ndarray:
    """Compute 2D face gradient geometric projection factors.

    Ported from $OR_SRC/engine/source/ale/ale2d/agrad2.F
    lines 95-161:
      For each edge k connecting element I and neighbor IV_k:
        D_k = - YC(I) + sum_{nodes of IV_k} Y
        DD_k = D_k_y^2 + D_k_z^2
        GRAD(I, k) = 4 * (D_k . N_k) / max(1e-15, DD_k)

    Args:
        x: (n_nodes, 2) nodal coordinates.
        connectivity: (n_elem, 4) quad connectivity.
        neighbor_elem: (n_elem, 4) neighbor element index (-1 if boundary).
        axisymmetric: if True, applies axisymmetric scaling to edge normals.

    Returns:
        grad: (n_elem, 4) directional derivative weighting factors across the 4 edges.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return np.empty((0, 4), dtype=np.float64)

    xe = x[connectivity]  # (n_elem, 4, 2)
    xc = np.mean(xe, axis=1)  # (n_elem, 2) centroids
    normals = compute_quad_edge_normals(xe, axisymmetric=axisymmetric)  # (n_elem, 4, 2)

    grad = np.zeros((n_elem, 4), dtype=np.float64)

    for e in range(n_elem):
        xc_e = xc[e]
        for k in range(4):
            nbr = neighbor_elem[e, k]
            if nbr >= 0:
                d = xc[nbr] - xc_e  # vector from element centroid to neighbor centroid
            else:
                # Boundary face: use edge midpoint
                edge_nodes = QUAD_EDGES[k]
                x_mid = 0.5 * (xe[e, edge_nodes[0]] + xe[e, edge_nodes[1]])
                d = x_mid - xc_e

            dd = float(np.dot(d, d))
            dd_safe = max(1e-15, dd)
            grad[e, k] = float(np.dot(d, normals[e, k])) / dd_safe

    return grad


def ale_2d_compute_fluxes(q_elem: np.ndarray,
                          x: np.ndarray,
                          velocity: np.ndarray,
                          w_grid: np.ndarray,
                          connectivity: np.ndarray,
                          neighbor_elem: Optional[np.ndarray] = None,
                          neighbor_edge: Optional[np.ndarray] = None,
                          upwl: float = 1.0,
                          reduc: float = 0.0,
                          axisymmetric: bool = False) -> Dict[str, np.ndarray]:
    """Compute upwind convective fluxes across quad edges in 2D ALE.

    Ported from $OR_SRC/engine/source/ale/ale2d/aflux2.F
    lines 78-236.

    Relative velocity on edge: V_rel = 0.5 * ((V_a - W_a) + (V_b - W_b)).
    Volume flux: F_vol_k = V_rel_k . N_k.
    Upwind flux arrays matching Fortran:
      FLUX(e, k) = F_vol_k - upwl * |F_vol_k|
      FLU1(e)    = sum_k (F_vol_k + upwl * |F_vol_k|)

    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) element-centered state variables.
        x: (n_nodes, 2) nodal coordinates.
        velocity: (n_nodes, 2) material velocity V.
        w_grid: (n_nodes, 2) grid velocity W.
        connectivity: (n_elem, 4) quad connectivity.
        neighbor_elem: optional (n_elem, 4) neighbor connectivity.
        neighbor_edge: optional (n_elem, 4) neighbor edge connectivity.
        upwl: upwind parameter (1.0 = donor cell upwind, 0.0 = centered).
        reduc: boundary face reduction factor (0.0 = closed/slip boundary).
        axisymmetric: if True, applies axisymmetric radius scaling.

    Returns:
        dict with:
            'vol_fluxes': (n_elem, 4) outward volume fluxes across each edge.
            'flux_q': (n_elem, 4) or (n_elem, 4, n_vars) fluxes of quantity q.
            'net_flux_q': (n_elem,) or (n_elem, n_vars) net outgoing flux.
            'FLUX': (n_elem, 4) matching Fortran aflux2.F line 225.
            'FLU1': (n_elem,) matching Fortran aflux2.F line 230.
    """
    n_elem = len(connectivity)
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_vars = q_2d.shape[1]

    if neighbor_elem is None or neighbor_edge is None:
        neighbor_elem, neighbor_edge = build_quad_edge_connectivity(connectivity)

    xe = x[connectivity]  # (n_elem, 4, 2)
    ve = velocity[connectivity]  # (n_elem, 4, 2)
    we = w_grid[connectivity]  # (n_elem, 4, 2)
    v_rel_nodes = ve - we  # (n_elem, 4, 2)

    normals = compute_quad_edge_normals(xe, axisymmetric=axisymmetric)  # (n_elem, 4, 2)

    # 1. Edge relative velocity and volume fluxes
    vol_fluxes = np.zeros((n_elem, 4), dtype=np.float64)
    for k in range(4):
        n1 = QUAD_EDGES[k, 0]
        n2 = QUAD_EDGES[k, 1]
        v_edge = 0.5 * (v_rel_nodes[:, n1] + v_rel_nodes[:, n2])
        vol_fluxes[:, k] = np.sum(v_edge * normals[:, k], axis=1)

    # Boundary edges: apply reduc factor (aflux2.F lines 187-211)
    boundary_mask = (neighbor_elem < 0)
    vol_fluxes[boundary_mask] *= reduc

    # Enforce strict antisymmetry on internal edges
    for e in range(n_elem):
        for k in range(4):
            nbr = neighbor_elem[e, k]
            if nbr > e:
                nbr_k = neighbor_edge[e, k]
                avg_f = 0.5 * (vol_fluxes[e, k] - vol_fluxes[nbr, nbr_k])
                vol_fluxes[e, k] = avg_f
                vol_fluxes[nbr, nbr_k] = -avg_f

    # 2. Upwind flux calculation for quantity q
    flux_q = np.zeros((n_elem, 4, n_vars), dtype=np.float64)
    for e in range(n_elem):
        for k in range(4):
            vf = vol_fluxes[e, k]
            if abs(vf) < 1e-30:
                continue
            nbr = neighbor_elem[e, k]
            if vf > 0.0:
                # Outflow: donor is e
                donor_q = q_2d[e]
            else:
                # Inflow: donor is neighbor (or e if boundary)
                donor_q = q_2d[nbr] if nbr >= 0 else q_2d[e]

            flux_q[e, k] = donor_q * vf

    # Strict antisymmetry on flux_q
    for e in range(n_elem):
        for k in range(4):
            nbr = neighbor_elem[e, k]
            if nbr > e:
                nbr_k = neighbor_edge[e, k]
                avg_q = 0.5 * (flux_q[e, k] - flux_q[nbr, nbr_k])
                flux_q[e, k] = avg_q
                flux_q[nbr, nbr_k] = -avg_q

    # Fortran aflux2.F lines 225-234:
    f_flux = vol_fluxes - upwl * np.abs(vol_fluxes)
    f_flu1 = np.sum(vol_fluxes + upwl * np.abs(vol_fluxes), axis=1)

    net_flux_q = np.sum(flux_q, axis=1)
    res_flux_q = flux_q[:, :, 0] if is_1d else flux_q
    res_net_q = net_flux_q[:, 0] if is_1d else net_flux_q

    return {
        "vol_fluxes": vol_fluxes,
        "flux_q": res_flux_q,
        "net_flux_q": res_net_q,
        "FLUX": f_flux,
        "FLU1": f_flu1,
    }


def ale_2d_advect(vtot: np.ndarray,
                  phi: np.ndarray,
                  flux_dict: Dict[str, np.ndarray],
                  dt: float,
                  neighbor_elem: Optional[np.ndarray] = None) -> np.ndarray:
    """Convective time update of state variables in 2D ALE.

    Ported from $OR_SRC/engine/source/ale/ale2d/aconv2.F
    lines 68-93:
      VL(I, k) = PHI(IV_k) * FLUX(IE, k)  (or PHI(IE) if IV_k == 0)
      delta = 0.5 * dt * (- PHI(IE) * FLU1(I) - sum_k VL(I, k))
      vtot_new = vtot + delta
      vtot_new = max(1e-20, vtot_new)

    Args:
        vtot: (n_elem,) extensive quantity (e.g. mass = rho * V, total internal energy).
        phi: (n_elem,) intensive quantity (e.g. density rho, specific energy eint).
        flux_dict: dictionary returned by ale_2d_compute_fluxes (containing FLUX, FLU1).
        dt: explicit time step duration.
        neighbor_elem: (n_elem, 4) neighbor connectivity.

    Returns:
        vtot_new: (n_elem,) updated extensive state variable after convection.
    """
    n_elem = len(vtot)
    f_flux = flux_dict["FLUX"]  # (n_elem, 4)
    f_flu1 = flux_dict["FLU1"]  # (n_elem,)

    vl = np.zeros((n_elem, 4), dtype=np.float64)
    for e in range(n_elem):
        for k in range(4):
            if neighbor_elem is not None and neighbor_elem[e, k] >= 0:
                iv = neighbor_elem[e, k]
                vl[e, k] = phi[iv] * f_flux[e, k]
            else:
                vl[e, k] = phi[e] * f_flux[e, k]

    # aconv2.F line 90:
    delta = 0.5 * dt * (- phi * f_flu1 - np.sum(vl, axis=1))
    vtot_new = np.maximum(1e-20, vtot + delta)
    return vtot_new


def ale_2d_remap(var: np.ndarray,
                 phi: np.ndarray,
                 swept_fluxes: np.ndarray,
                 vol: np.ndarray,
                 dt: float,
                 neighbor_elem: np.ndarray,
                 jmult: int = 0) -> np.ndarray:
    """Rezoning / remapping of 2D element variables.

    Ported from $OR_SRC/engine/source/ale/ale2d/arezo2.F
    lines 68-106:
      VAR(I) = VAR(I) + 0.5 * dt * (PHI(IE) * sum(FLUX) - sum(PHI(IV_k) * FLUX_k)) / VOL

    Args:
        var: (n_elem,) variable to be updated (e.g. density, specific internal energy).
        phi: (n_elem,) intensive flux carrier variable (often equal to var).
        swept_fluxes: (n_elem, 4) swept volume fluxes across edges.
        vol: (n_elem,) element volume / area.
        dt: time step.
        neighbor_elem: (n_elem, 4) neighbor connectivity.
        jmult: multi-material flag (1: update volume before division).

    Returns:
        var_new: (n_elem,) updated remapped variable.
    """
    n_elem = len(var)
    var_new = var.copy()

    for e in range(n_elem):
        sum_flux = float(np.sum(swept_fluxes[e]))
        sum_phi_flux = 0.0
        for k in range(4):
            nbr = neighbor_elem[e, k]
            phi_nbr = phi[nbr] if nbr >= 0 else phi[e]
            sum_phi_flux += phi_nbr * swept_fluxes[e, k]

        num = phi[e] * sum_flux - sum_phi_flux
        if jmult == 0:
            if vol[e] > 0.0:
                var_new[e] += 0.5 * dt * num / vol[e]
        else:
            vol_n = vol[e] - dt * sum_flux
            if vol_n > 1e-15:
                var_new[e] += 0.5 * dt * num / max(1e-15, vol_n)

    return var_new


def ale_2d_momentum_forces(rho: np.ndarray,
                           areas: np.ndarray,
                           v_nodes: np.ndarray,
                           w_nodes: np.ndarray,
                           x_nodes: np.ndarray,
                           connectivity: np.ndarray,
                           gamma: float = 0.0,
                           supg: bool = False,
                           off: float = 1.0) -> Tuple[np.ndarray, float]:
    """Compute 2D ALE momentum convective forces on nodes.

    Ported from $OR_SRC/engine/source/ale/ale2d/amomt2.F
    lines 105-225 (standard formulation) & lines 226-372 (SUPG formulation).

    Calculates nodal forces F_conv that account for convective momentum transport:
      T_1i: convective force in x (dim 0) on local node i (1..4)
      T_2i: convective force in y (dim 1) on local node i (1..4)

    Energy accounting:
      Calculates the net mechanical power of convective forces:
        P_conv = sum_{nodes} F_node . V_node
      which is returned for tracking in the engine energy ledger.

    Args:
        rho: (n_elem,) fluid density.
        areas: (n_elem,) quad element areas.
        v_nodes: (n_nodes, 2) material velocity.
        w_nodes: (n_nodes, 2) grid velocity.
        x_nodes: (n_nodes, 2) nodal coordinates.
        connectivity: (n_elem, 4) quad connectivity.
        gamma: upwind parameter (PM(15, MAT)).
        supg: if True, applies Streamline Upwind Petrov-Galerkin (UPWM == 3).
        off: active element flag (1.0 = active).

    Returns:
        f_nodes: (n_nodes, 2) assembled convective forces on all nodes.
        power_conv: float, total mechanical power of convective forces.
    """
    n_nodes = len(x_nodes)
    n_elem = len(connectivity)
    f_nodes = np.zeros((n_nodes, 2), dtype=np.float64)
    if n_elem == 0:
        return f_nodes, 0.0

    xe = x_nodes[connectivity]  # (n_elem, 4, 2)
    ve = v_nodes[connectivity]  # (n_elem, 4, 2)
    we = w_nodes[connectivity]  # (n_elem, 4, 2)
    v_rel = ve - we             # (n_elem, 4, 2)

    # 1. Element centroid relative velocity: VDY, VDZ (amomt2.F lines 139-142)
    v_rel_c = 0.25 * np.sum(v_rel, axis=1)  # (n_elem, 2)
    vdy = v_rel_c[:, 0]
    vdz = v_rel_c[:, 1]

    # Quarter mass: XMS = 0.25 * RHO * AREA
    xms = 0.25 * rho * areas  # (n_elem,)

    # 2. Geometric shape function derivatives for standard quad (amomt2.F lines 202-225)
    # y1..y4 = x[:, 0], z1..z4 = x[:, 1]
    y1, y2, y3, y4 = xe[:, 0, 0], xe[:, 1, 0], xe[:, 2, 0], xe[:, 3, 0]
    z1, z2, z3, z4 = xe[:, 0, 1], xe[:, 1, 1], xe[:, 2, 1], xe[:, 3, 1]

    # Hourglass / gradient vectors:
    # PY1 = 0.5 * (y2 - y4), PZ1 = 0.5 * (z2 - z4)
    # PY2 = 0.5 * (y3 - y1), PZ2 = 0.5 * (z3 - z1)
    py1 = 0.5 * (y2 - y4)
    pz1 = 0.5 * (z2 - z4)
    py2 = 0.5 * (y3 - y1)
    pz2 = 0.5 * (z3 - z1)

    # Velocity gradient components at centroid:
    # DYY = sum(V_rel_x * B_x), DZZ = sum(V_rel_y * B_y)...
    # In amomt2.F: DYY, DZZ, DYZ, DZY
    # DYY = -(z2 - z4)*v1x + -(z3 - z1)*v2x + ...
    dyy = - (z2 - z4) * v_rel[:, 0, 0] - (z3 - z1) * v_rel[:, 1, 0] - (z4 - z2) * v_rel[:, 2, 0] - (z1 - z3) * v_rel[:, 3, 0]
    dyz =   (y2 - y4) * v_rel[:, 0, 0] + (y3 - y1) * v_rel[:, 1, 0] + (y4 - y2) * v_rel[:, 2, 0] + (y1 - y3) * v_rel[:, 3, 0]
    dzy = - (z2 - z4) * v_rel[:, 0, 1] - (z3 - z1) * v_rel[:, 1, 1] - (z4 - z2) * v_rel[:, 2, 1] - (z1 - z3) * v_rel[:, 3, 1]
    dzz =   (y2 - y4) * v_rel[:, 0, 1] + (y3 - y1) * v_rel[:, 1, 1] + (y4 - y2) * v_rel[:, 2, 1] + (y1 - y3) * v_rel[:, 3, 1]

    # Normalize by area
    safe_area = np.maximum(areas, 1e-30)
    dyy = dyy / (2.0 * safe_area)
    dyz = dyz / (2.0 * safe_area)
    dzy = dzy / (2.0 * safe_area)
    dzz = dzz / (2.0 * safe_area)

    # Convective centroid force components F1, F2:
    f1 = (vdy * dyy + vdz * dyz) * xms * off
    f2 = (vdy * dzy + vdz * dzz) * xms * off

    # Upwind weighting factors A1, A2:
    a1_val = py1 * vdy + pz1 * vdz
    a2_val = py2 * vdy + pz2 * vdz
    a1 = np.sign(a1_val) * gamma
    a2 = np.sign(a2_val) * gamma

    # Convective forces distributed to the 4 nodes:
    # T11..T14 for x-direction, T21..T24 for y-direction (amomt2.F lines 215-224)
    t11 = (1.0 + a1) * f1
    t12 = (1.0 + a2) * f1
    t13 = (1.0 - a1) * f1
    t14 = (1.0 - a2) * f1

    t21 = (1.0 + a1) * f2
    t22 = (1.0 + a2) * f2
    t23 = (1.0 - a1) * f2
    t24 = (1.0 - a2) * f2

    # Assemble onto global nodes
    for e in range(n_elem):
        conn = connectivity[e]
        # Node 0
        f_nodes[conn[0], 0] += t11[e]
        f_nodes[conn[0], 1] += t21[e]
        # Node 1
        f_nodes[conn[1], 0] += t12[e]
        f_nodes[conn[1], 1] += t22[e]
        # Node 2
        f_nodes[conn[2], 0] += t13[e]
        f_nodes[conn[2], 1] += t23[e]
        # Node 3
        f_nodes[conn[3], 0] += t14[e]
        f_nodes[conn[3], 1] += t24[e]

    # Mechanical work rate (power) of convective forces
    power_conv = float(np.sum(f_nodes * v_nodes))
    return f_nodes, power_conv


def ale_2d_diffuse(phi_elem: np.ndarray,
                   grad_geom: np.ndarray,
                   alpha_diff: np.ndarray,
                   areas: np.ndarray,
                   dt: float,
                   neighbor_elem: np.ndarray) -> np.ndarray:
    """Finite-volume diffusion with harmonic interpolation on 2D quad mesh.

    Ported from $OR_SRC/engine/source/ale/ale2d/adiff2.F
    lines 81-127:
      Harmonic interpolation:
        AA_face_k = (alpha_e * alpha_nbr) / max(1e-20, alpha_e + alpha_nbr)
      Time evolution:
        DPHI_e = sum_k AA_face_k * (phi_nbr - phi_e) * GRAD(e, k)
        phi_new_e = phi_e + 2 * DPHI_e * dt / max(VOL_e, 1e-20)

    Args:
        phi_elem: (n_elem,) element scalar field to diffuse (e.g. k, epsilon, internal energy).
        grad_geom: (n_elem, 4) geometric gradient projection factors from ale_2d_compute_gradients.
        alpha_diff: (n_elem,) diffusivity coefficient at element centers.
        areas: (n_elem,) element areas/volumes.
        dt: explicit time step duration.
        neighbor_elem: (n_elem, 4) neighbor connectivity.

    Returns:
        phi_new: (n_elem,) updated scalar field after diffusion.
    """
    n_elem = len(phi_elem)
    if n_elem == 0:
        return phi_elem.copy()

    dphi = np.zeros(n_elem, dtype=np.float64)

    for e in range(n_elem):
        ae = alpha_diff[e]
        qe = phi_elem[e]
        for k in range(4):
            nbr = neighbor_elem[e, k]
            if nbr >= 0:
                anbr = alpha_diff[nbr]
                qnbr = phi_elem[nbr]
            else:
                # Ghost cell / insulated boundary: value matches e
                anbr = ae
                qnbr = qe

            # Harmonic mean: 2 * (a * b) / (a + b) (adiff2.F uses (a*b)/(a+b) with factor 2 below)
            aa_face = (ae * anbr) / max(1e-20, ae + anbr)
            dphi[e] += aa_face * (qnbr - qe) * grad_geom[e, k]

    # adiff2.F line 121: DPHI = TWO * DPHI * DT / VOL
    delta = 2.0 * dphi * dt / np.maximum(areas, 1e-20)
    phi_new = phi_elem + delta
    return phi_new
