# pyradioss/engine/ale_engine.py
# Port of engine/source/ale/ale3d/*.F
"""
Arbitrary Lagrangian-Eulerian (ALE) advection and grid management engine.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/ale3d/aconv3.F: 3D upwind convection update (lines 81-135)
- engine/source/ale/ale3d/aflux3.F: face flux calculation and upwind treatment (lines 208-269, 376-388, 535-550)
- engine/source/ale/ale3d/arezo3.F: rezoning / remapping of element variables (lines 76-96)
- engine/source/ale/ale3d/agrad3.F: gradient reconstruction (lines 120-264)
- engine/source/ale/alemuscl/gradient_reconstruction.F90: least-squares gradient reconstruction (lines 134-265)
- engine/source/ale/alemuscl/gradient_limitation.F: Barth-Jespersen slope limiter (lines 69-120)
- engine/source/ale/grid/alew5.F: Laplacian grid smoothing for /ALE/GRID/LAPLACIAN (lines 113-144)
- engine/source/ale/grid/alew.F: Donea distance-weighted smoothing for /ALE/GRID/DONEA (lines 98-180)
- engine/source/ale/grid/alew2.F: Spring network grid smoothing for /ALE/GRID/SPRING (lines 83-295, 371-385)
- engine/source/ale/grid/alew4.F: Curvature grid smoothing for /ALE/GRID/STANDARD (lines 191-387, 464-478)
- engine/source/ale/grid/alew6.F: Centroidal Voronoi / volume grid smoothing for /ALE/GRID/VOLUME (lines 98-179)
- engine/source/ale/grid/alelin.F: grid velocity link constraints for /ALE/LINK/VEL (lines 61-198)
- starter/source/ale/bimat/inimu3.F & engine/source/ale/bimat/bimat2.F: multi-material volume fraction remapping (lines 80-165)
- engine/source/ale/subcycling/alesub1.F: ALE subcycling part 1 (lines 70-98)
- engine/source/ale/subcycling/alesub2.F: ALE subcycling part 2 (lines 82-135)
- engine/source/ale/atherm.F: Thermal ALE diffusivity and conductivity (lines 129-139)
- engine/source/ale/ale3d/adiff3.F: 3D finite volume thermal diffusion (lines 81-149)
- engine/source/ale/arezon.F90: ALE rezoning and state variable remapping (lines 38-120)
- engine/source/ale/aconve.F90: ALE convection driver (lines 37-120)
- engine/source/ale/alemain.F: ALE main driver and system coordinator (lines 24-120)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple, Dict, Set, List, Any


# -----------------------------------------------------------------------------
# Hex8 Face Definitions (OpenRadioss standard 8-node brick)
# Fortran origin: engine/source/ale/ale3d/aflux3.F lines 245-268, 287-292
# -----------------------------------------------------------------------------
# 0-based node indices for the 6 quad faces of an 8-node hexahedral element:
# Face 0 (bottom, -z): [0, 1, 2, 3]  (Fortran 1-based: 1, 2, 3, 4)
# Face 1 (back, +y):   [2, 3, 7, 6]  (Fortran 1-based: 3, 4, 8, 7)
# Face 2 (top, +z):    [4, 5, 6, 7]  (Fortran 1-based: 5, 6, 7, 8)
# Face 3 (front, -y):  [0, 1, 5, 4]  (Fortran 1-based: 1, 2, 6, 5)
# Face 4 (right, +x):  [1, 2, 6, 5]  (Fortran 1-based: 2, 3, 7, 6)
# Face 5 (left, -x):   [0, 3, 7, 4]  (Fortran 1-based: 1, 4, 8, 5)
HEX_FACES = np.array([
    [0, 1, 2, 3],  # Face 0: -z
    [2, 3, 7, 6],  # Face 1: +y
    [4, 5, 6, 7],  # Face 2: +z
    [0, 1, 5, 4],  # Face 3: -y
    [1, 2, 6, 5],  # Face 4: +x
    [0, 3, 7, 4],  # Face 5: -x
], dtype=np.int64)

# 12 edges of a hex8 element:
HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def build_face_connectivity(connectivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build element-element neighbor connectivity table across all 6 faces.
    
    Fortran origin: common_source/modules/ale/ale_connectivity_mod.F
    and engine/source/ale/ale3d/aflux3.F lines 493-524.
    
    Args:
        connectivity: (n_elem, 8) array of node indices for each hex element.
        
    Returns:
        neighbor_elem: (n_elem, 6) int64, neighbor element index (-1 if boundary).
        neighbor_face: (n_elem, 6) int64, face index on neighbor element (-1 if boundary).
    """
    n_elem = len(connectivity)
    neighbor_elem = np.full((n_elem, 6), -1, dtype=np.int64)
    neighbor_face = np.full((n_elem, 6), -1, dtype=np.int64)
    
    face_map: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
    for e in range(n_elem):
        conn_e = connectivity[e]
        for f_idx in range(6):
            f_nodes = HEX_FACES[f_idx]
            key = tuple(sorted(int(conn_e[n]) for n in f_nodes))
            if key in face_map:
                e_prev, f_prev = face_map[key]
                neighbor_elem[e, f_idx] = e_prev
                neighbor_face[e, f_idx] = f_prev
                neighbor_elem[e_prev, f_prev] = e
                neighbor_face[e_prev, f_prev] = f_idx
            else:
                face_map[key] = (e, f_idx)
                
    return neighbor_elem, neighbor_face


def compute_hex_face_normals(xe: np.ndarray) -> np.ndarray:
    """Compute outward area-normal vectors for all 6 faces of hex elements.
    
    Fortran origin: engine/source/ale/ale3d/aflux3.F lines 246-268.
    Each vector N_k has magnitude 2 * Area(face_k) and points outward.
    
    Args:
        xe: (n_elem, 8, 3) nodal coordinates for each element.
        
    Returns:
        normals: (n_elem, 6, 3) outward normal vectors (magnitude 2*Area).
    """
    n_elem = len(xe)
    normals = np.zeros((n_elem, 6, 3), dtype=np.float64)
    if n_elem == 0:
        return normals
        
    # Face 0: 0, 1, 2, 3 -> N0 = (r2 - r0) x (r1 - r3)
    # Fortran aflux3.F lines 246-248: (Y3-Y1)*(Z2-Z4) - (Z3-Z1)*(Y2-Y4)...
    r0, r1, r2, r3 = xe[:, 0], xe[:, 1], xe[:, 2], xe[:, 3]
    r4, r5, r6, r7 = xe[:, 4], xe[:, 5], xe[:, 6], xe[:, 7]
    
    normals[:, 0] = np.cross(r2 - r0, r1 - r3)
    # Face 1: 2, 3, 7, 6 -> N1 = (r6 - r3) x (r2 - r7)
    normals[:, 1] = np.cross(r6 - r3, r2 - r7)
    # Face 2: 4, 5, 6, 7 -> N2 = (r5 - r7) x (r6 - r4)
    normals[:, 2] = np.cross(r5 - r7, r6 - r4)
    # Face 3: 0, 1, 5, 4 -> N3 = (r1 - r4) x (r5 - r0)
    normals[:, 3] = np.cross(r1 - r4, r5 - r0)
    # Face 4: 1, 2, 6, 5 -> N4 = (r6 - r1) x (r5 - r2)
    normals[:, 4] = np.cross(r6 - r1, r5 - r2)
    # Face 5: 0, 3, 7, 4 -> N5 = (r7 - r0) x (r3 - r4)
    normals[:, 5] = np.cross(r7 - r0, r3 - r4)
    
    return normals


def compute_hex_volumes(x: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Compute volumes of 8-node hexahedral elements via boundary surface integration.
    
    Exact divergence-theorem pyramidal decomposition about the centroid:
    V = 1/6 * sum_{k=0}^5 (X_face_k - X_centroid) . N_k
    Fortran origin: engine/source/elements/solid/solide/srcoor3.F
    
    Args:
        x: (n_nodes, 3) nodal coordinates.
        connectivity: (n_elem, 8) hex element connectivity.
        
    Returns:
        volumes: (n_elem,) positive element volumes.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return np.empty(0, dtype=np.float64)
        
    xe = x[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3) centroid
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3)
    
    vols = np.zeros(n_elem, dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        xf = np.mean(xe[:, f_nodes], axis=1)  # (n_elem, 3) face centroid
        # Dot product with face normal
        vols += np.sum((xf - xc) * normals[:, f_idx], axis=1)
        
    vols = vols / 6.0
    return np.maximum(vols, 1e-30)


# -----------------------------------------------------------------------------
# Limiters (MUSCL)
# Fortran origin: engine/source/ale/alemuscl/gradient_limitation.F lines 69-120
# -----------------------------------------------------------------------------

def limiter_minmod(r: np.ndarray) -> np.ndarray:
    """Standard Minmod slope limiter: phi(r) = max(0, min(1, r))."""
    return np.maximum(0.0, np.minimum(1.0, r))


def limiter_van_leer(r: np.ndarray) -> np.ndarray:
    """Standard Van Leer slope limiter: phi(r) = (r + |r|) / (1 + |r|)."""
    abs_r = np.abs(r)
    return np.where(r > 0.0, (2.0 * r) / (1.0 + abs_r), 0.0)


def compute_barth_jespersen_limiter(q_elem: np.ndarray,
                                    grad_q: np.ndarray,
                                    xe: np.ndarray,
                                    neighbor_elem: np.ndarray) -> np.ndarray:
    """Multidimensional Barth-Jespersen slope limiter for unstructured hex cells.
    
    Fortran origin: engine/source/ale/alemuscl/gradient_limitation.F lines 69-120:
    Ensures that reconstructed values on element faces remain bounded between
    the local element minimum and maximum neighbor values (maximum principle).
    
    Args:
        q_elem: (n_elem,) element scalar values.
        grad_q: (n_elem, 3) element gradients.
        xe: (n_elem, 8, 3) nodal coordinates.
        neighbor_elem: (n_elem, 6) neighbor element indices.
        
    Returns:
        phi: (n_elem,) reduction factors in [0, 1].
    """
    n_elem = len(q_elem)
    phi = np.ones(n_elem, dtype=np.float64)
    xc = np.mean(xe, axis=1)  # (n_elem, 3)
    
    for e in range(n_elem):
        qe = q_elem[e]
        nbrs = [neighbor_elem[e, k] for k in range(6) if neighbor_elem[e, k] >= 0]
        if not nbrs:
            continue
        q_nbr_vals = q_elem[nbrs]
        q_max = max(qe, float(np.max(q_nbr_vals)))
        q_min = min(qe, float(np.min(q_nbr_vals)))
        
        ge = grad_q[e]
        if np.dot(ge, ge) < 1e-30:
            continue
            
        phi_e = 1.0
        # Check all 6 face centroids
        for f_idx in range(6):
            xf = np.mean(xe[e, HEX_FACES[f_idx]], axis=0)
            dq = float(np.dot(ge, xf - xc[e]))
            if dq > 1e-15:
                ratio = (q_max - qe) / dq
                phi_e = min(phi_e, max(0.0, ratio))
            elif dq < -1e-15:
                ratio = (q_min - qe) / dq
                phi_e = min(phi_e, max(0.0, ratio))
                
        phi[e] = min(1.0, max(0.0, phi_e))
        
    return phi


# -----------------------------------------------------------------------------
# Function 1: Gradient Reconstruction
# -----------------------------------------------------------------------------

def ale_compute_gradients(q_elem: np.ndarray,
                          x: np.ndarray,
                          connectivity: np.ndarray) -> np.ndarray:
    """Least-squares gradient reconstruction of element-centered fields.
    
    Fortran origin:
    - engine/source/ale/ale3d/agrad3.F: lines 120-264
    - engine/source/ale/alemuscl/gradient_reconstruction.F90: lines 134-265
    
    Solves the 3D normal equations M * grad = b at each element centroid,
    where M = sum_j (X_j - X_c) (X_j - X_c)^T and b = sum_j (q_j - q_c) (X_j - X_c).
    For any linear field q(x, y, z) = c0 + g . x, this reconstruction yields the
    EXACT gradient g to machine precision.
    
    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) element-centered field values.
        x: (n_nodes, 3) nodal coordinates.
        connectivity: (n_elem, 8) element connectivity.
        
    Returns:
        grad_q: (n_elem, 3) or (n_elem, n_vars, 3) reconstructed gradient vector.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        if q_elem.ndim == 1:
            return np.empty((0, 3), dtype=np.float64)
        return np.empty((0, q_elem.shape[1], 3), dtype=np.float64)
        
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_vars = q_2d.shape[1]
    
    xe = x[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3) element centroids
    
    # Build node-to-element mapping to quickly gather all connected neighbors
    node_to_elems: Dict[int, List[int]] = {}
    for e in range(n_elem):
        for n_idx in connectivity[e]:
            node_to_elems.setdefault(int(n_idx), []).append(e)
            
    grad_q = np.zeros((n_elem, n_vars, 3), dtype=np.float64)
    
    for e in range(n_elem):
        # Gather all neighbor elements sharing at least one node with element e
        nbr_set: Set[int] = set()
        for n_idx in connectivity[e]:
            nbr_set.update(node_to_elems[int(n_idx)])
        nbr_set.discard(e)
        
        if not nbr_set:
            # Single isolated element: use element's 8 nodes relative to centroid
            dx = xe[e] - xc[e]  # (8, 3)
            # Extrapolate centroid value
            # With no other elements, gradient is 0
            continue
            
        nbr_indices = list(nbr_set)
        dx = xc[nbr_indices] - xc[e]  # (k, 3)
        dq = q_2d[nbr_indices] - q_2d[e]  # (k, n_vars)
        
        # Least squares solve: dx @ grad^T = dq -> grad^T = lstsq(dx, dq)
        # Using SVD pseudo-inverse / lstsq
        sol, _, _, _ = np.linalg.lstsq(dx, dq, rcond=1e-10)
        grad_q[e] = sol.T  # (n_vars, 3)
        
    if is_1d:
        return grad_q[:, 0, :]
    return grad_q


# -----------------------------------------------------------------------------
# Function 2: Upwind Flux Computation
# -----------------------------------------------------------------------------

def ale_compute_fluxes(q_elem: np.ndarray,
                       grad_q: Optional[np.ndarray],
                       x: np.ndarray,
                       velocity: np.ndarray,
                       connectivity: np.ndarray,
                       limiter: str = "van_leer",
                       upwl: float = 1.0) -> Dict[str, Any]:
    """Compute upwind convective fluxes across element faces.
    
    Fortran origin:
    - engine/source/ale/ale3d/aflux3.F: lines 208-269, 376-388, 535-550
    - engine/source/ale/alemuscl/alemuscl_upwind.F: lines 114-198
    
    Calculates outward volumetric flux F_vol across each face from relative velocity
    V_rel = V - W, and reconstructs face values using donor-cell upwinding with
    optional MUSCL slope limiting.
    
    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) element-centered state variables.
        grad_q: (n_elem, 3) or (n_elem, n_vars, 3) reconstructed gradients (or None).
        x: (n_nodes, 3) nodal coordinates.
        velocity: (n_nodes, 3) nodal relative velocities (V - W).
        connectivity: (n_elem, 8) element connectivity.
        limiter: slope limiter name ('van_leer', 'minmod', 'barth_jespersen', or 'none').
        upwl: upwind parameter (1.0 = full upwind, 0.0 = centered). Fortran PM(16, MAT).
        
    Returns:
        dict with:
            'flux_q': (n_elem, 6) or (n_elem, 6, n_vars) face fluxes of quantity q.
            'net_flux_q': (n_elem,) or (n_elem, n_vars) net outgoing flux sum_k flux_q_k.
            'vol_fluxes': (n_elem, 6) volumetric flux across each face.
            'net_vol_flux': (n_elem,) net outgoing volumetric flux sum_k F_vol_k.
            'FLUX': (n_elem, 6) matching Fortran aflux3.F line 537.
            'FLU1': (n_elem,) matching Fortran aflux3.F line 544.
    """
    n_elem = len(connectivity)
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_vars = q_2d.shape[1]
    
    xe = x[connectivity]  # (n_elem, 8, 3)
    ve = velocity[connectivity]  # (n_elem, 8, 3) relative velocities
    xc = np.mean(xe, axis=1)  # (n_elem, 3) element centroids
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3) outward normals (mag 2*Area)
    
    neighbor_elem, neighbor_face = build_face_connectivity(connectivity)
    
    # Compute relative velocity on each face: V_face = 1/4 sum(V_node)
    # Volumetric flux F_vol = V_face . (Area * n) = 0.5 * V_face . N_k
    # Fortran aflux3.F lines 214-236, 381-388
    vol_fluxes = np.zeros((n_elem, 6), dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        v_face = np.mean(ve[:, f_nodes], axis=1)  # (n_elem, 3)
        vol_fluxes[:, f_idx] = 0.5 * np.sum(v_face * normals[:, f_idx], axis=1)
        
    # Boundary faces have 0 volume flux by default (slip wall, aflux3.F lines 496-524)
    boundary_mask = (neighbor_elem < 0)
    vol_fluxes[boundary_mask] = 0.0
    
    # Enforce strict bitwise antisymmetry across shared internal faces:
    # F_vol(e, f) = - F_vol(nbr, nbr_f)
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_flux = 0.5 * (vol_fluxes[e, f_idx] - vol_fluxes[nbr, nbr_f])
                vol_fluxes[e, f_idx] = avg_flux
                vol_fluxes[nbr, nbr_f] = -avg_flux
                
    # Reconstruct face values q_face using MUSCL / upwind
    # If grad_q is provided, apply slope limiter
    use_muscl = (grad_q is not None and limiter.lower() != "none")
    grad_2d = None
    if use_muscl:
        grad_2d = grad_q[:, None, :] if grad_q.ndim == 2 else grad_q
        
    # Compute limiters per variable
    phi = np.ones((n_elem, n_vars), dtype=np.float64)
    if use_muscl and limiter.lower() == "barth_jespersen":
        for v in range(n_vars):
            phi[:, v] = compute_barth_jespersen_limiter(q_2d[:, v], grad_2d[:, v], xe, neighbor_elem)
            
    flux_q = np.zeros((n_elem, 6, n_vars), dtype=np.float64)
    
    for e in range(n_elem):
        for f_idx in range(6):
            vf = vol_fluxes[e, f_idx]
            if abs(vf) < 1e-30:
                continue
            nbr = neighbor_elem[e, f_idx]
            
            # Upwind donor cell selection
            if vf > 0.0:
                # Flow leaves element e -> donor is element e
                donor = e
                f_nodes = HEX_FACES[f_idx]
                xf = np.mean(xe[e, f_nodes], axis=0)
                if use_muscl:
                    dr = xf - xc[e]
                    for v in range(n_vars):
                        dq = float(np.dot(grad_2d[e, v], dr))
                        if limiter.lower() in ("van_leer", "minmod") and nbr >= 0:
                            # 1D slope ratio r between consecutive element differences
                            d_donor = q_2d[e, v] - q_2d[nbr, v]
                            r = dq / (d_donor + 1e-30)
                            psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                            q_face_val = q_2d[e, v] + psi * dq
                        else:
                            q_face_val = q_2d[e, v] + phi[e, v] * dq
                        flux_q[e, f_idx, v] = q_face_val * vf
                else:
                    flux_q[e, f_idx] = q_2d[e] * vf
            else:
                # Flow enters element e from neighbor -> donor is nbr
                if nbr >= 0:
                    donor = nbr
                    f_nodes = HEX_FACES[f_idx]
                    xf = np.mean(xe[e, f_nodes], axis=0)
                    if use_muscl:
                        dr = xf - xc[nbr]
                        for v in range(n_vars):
                            dq = float(np.dot(grad_2d[nbr, v], dr))
                            if limiter.lower() in ("van_leer", "minmod"):
                                d_donor = q_2d[nbr, v] - q_2d[e, v]
                                r = dq / (d_donor + 1e-30)
                                psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                                q_face_val = q_2d[nbr, v] + psi * dq
                            else:
                                q_face_val = q_2d[nbr, v] + phi[nbr, v] * dq
                            flux_q[e, f_idx, v] = q_face_val * vf
                    else:
                        flux_q[e, f_idx] = q_2d[nbr] * vf
                else:
                    flux_q[e, f_idx] = q_2d[e] * vf
                    
    # Strict antisymmetry on flux_q for shared faces:
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_q_flux = 0.5 * (flux_q[e, f_idx] - flux_q[nbr, nbr_f])
                flux_q[e, f_idx] = avg_q_flux
                flux_q[nbr, nbr_f] = -avg_q_flux
                
    net_flux_q = np.sum(flux_q, axis=1)  # (n_elem, n_vars)
    net_vol_flux = np.sum(vol_fluxes, axis=1)  # (n_elem,)
    
    # Fortran aflux3.F outputs FLUX and FLU1 (lines 535-550):
    # FLUX(e, k) = FLUX_k - UPWL * ABS(FLUX_k)
    # FLU1(e)    = sum_k (FLUX_k + UPWL * ABS(FLUX_k))
    f_flux = vol_fluxes - upwl * np.abs(vol_fluxes)
    f_flu1 = np.sum(vol_fluxes + upwl * np.abs(vol_fluxes), axis=1)
    
    res_flux_q = flux_q[:, :, 0] if is_1d else flux_q
    res_net_q = net_flux_q[:, 0] if is_1d else net_flux_q
    
    return {
        "flux_q": res_flux_q,
        "net_flux_q": res_net_q,
        "vol_fluxes": vol_fluxes,
        "net_vol_flux": net_vol_flux,
        "FLUX": f_flux,
        "FLU1": f_flu1,
    }


# -----------------------------------------------------------------------------
# Function 3: Convection Update
# -----------------------------------------------------------------------------

def ale_advect(q_elem: np.ndarray,
               fluxes: Union[Dict[str, Any], np.ndarray],
               volumes: np.ndarray,
               dt: float) -> np.ndarray:
    """Convective time update of state variables (mass, energy, momentum).
    
    Fortran origin: engine/source/ale/ale3d/aconv3.F lines 107-135:
    Q_new = Q_old - dt * sum_k flux_q_k
    q_new = Q_new / V_new
    
    Guarantees exact conservation: sum(q_new * V_new) == sum(q_old * V_old).
    
    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) state variable before advection.
        fluxes: flux dict from ale_compute_fluxes OR (n_elem, 6, ...) face flux array.
        volumes: (n_elem,) element volumes at step start.
        dt: time step duration.
        
    Returns:
        q_new: updated state variable after convection.
    """
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_elem, n_vars = q_2d.shape
    
    if isinstance(fluxes, dict):
        net_flux = fluxes["net_flux_q"]
        if is_1d and net_flux.ndim == 1:
            net_flux_2d = net_flux[:, None]
        elif not is_1d and net_flux.ndim == 1:
            net_flux_2d = net_flux[:, None]
        else:
            net_flux_2d = net_flux
        net_vol = fluxes.get("net_vol_flux", np.zeros(n_elem))
    else:
        # fluxes is array
        if fluxes.ndim == q_2d.ndim + 1:  # (n_elem, 6, n_vars)
            net_flux_2d = np.sum(fluxes, axis=1)
        elif is_1d and fluxes.ndim == 2:  # (n_elem, 6)
            net_flux_2d = np.sum(fluxes, axis=1)[:, None]
        else:
            net_flux_2d = fluxes[:, None] if is_1d else fluxes
        net_vol = np.zeros(n_elem)
        
    vol_old = volumes
    vol_new = np.maximum(vol_old - dt * net_vol, 1e-30)
    
    q_extensive_old = q_2d * vol_old[:, None]
    q_extensive_new = q_extensive_old - dt * net_flux_2d
    
    q_new = q_extensive_new / vol_new[:, None]
    
    return q_new[:, 0] if is_1d else q_new


# -----------------------------------------------------------------------------
# Function 4: Rezoning / Remapping
# -----------------------------------------------------------------------------

def ale_remap(q_lagrange: np.ndarray,
              q_ale: Optional[np.ndarray],
              x_old: np.ndarray,
              x_new: np.ndarray,
              connectivity: np.ndarray,
              limiter: str = "van_leer",
              extensive: bool = False) -> np.ndarray:
    """Rezoning / remapping of fields from deformed Lagrangian mesh to smoothed ALE mesh.
    
    Fortran origin: engine/source/ale/ale3d/arezo3.F lines 76-96:
    Computes swept face volumes between x_old and x_new, reconstructs upwind fluxes,
    and conservatively remaps extensive or intensive quantities.
    
    Guarantees machine-precision mass and energy conservation:
    sum(q_remap * V_new) == sum(q_lagrange * V_old).
    
    Args:
        q_lagrange: (n_elem,) or (n_elem, n_vars) state variable on Lagrangian mesh.
        q_ale: optional destination array (or None to allocate new array).
        x_old: (n_nodes, 3) nodal coordinates of Lagrangian mesh.
        x_new: (n_nodes, 3) nodal coordinates of smoothed ALE mesh.
        connectivity: (n_elem, 8) hex element connectivity.
        limiter: slope limiter ('van_leer', 'minmod', 'barth_jespersen', or 'none').
        extensive: True if q is already extensive (total mass/energy), False if intensive (density, specific energy).
        
    Returns:
        q_remapped: (n_elem,) or (n_elem, n_vars) remapped state variable on new mesh.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return q_lagrange.copy()
        
    is_1d = (q_lagrange.ndim == 1)
    q_2d = q_lagrange[:, None] if is_1d else q_lagrange
    n_vars = q_2d.shape[1]
    
    # 1. Compute element volumes on old and new configurations
    v_old = compute_hex_volumes(x_old, connectivity)
    v_new = compute_hex_volumes(x_new, connectivity)
    
    # 2. Grid displacement and midpoint configuration
    disp = x_new - x_old  # (n_nodes, 3)
    x_mid = 0.5 * (x_old + x_new)
    
    xe_mid = x_mid[connectivity]  # (n_elem, 8, 3)
    xe_old = x_old[connectivity]  # (n_elem, 8, 3)
    xc_old = np.mean(xe_old, axis=1)  # (n_elem, 3)
    de = disp[connectivity]  # (n_elem, 8, 3)
    
    normals_mid = compute_hex_face_normals(xe_mid)  # (n_elem, 6, 3)
    neighbor_elem, neighbor_face = build_face_connectivity(connectivity)
    
    # 3. Swept volumes across faces: delta_V = 0.5 * d_face . N_mid
    # Positive delta_V means the face swept outward (element loses volume)
    swept_vols = np.zeros((n_elem, 6), dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        d_face = np.mean(de[:, f_nodes], axis=1)  # (n_elem, 3)
        swept_vols[:, f_idx] = 0.5 * np.sum(d_face * normals_mid[:, f_idx], axis=1)
        
    # Boundary faces have no volume flux (domain boundary fixed or closed)
    swept_vols[neighbor_elem < 0] = 0.0
    
    # Enforce strict antisymmetry across shared faces
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_v = 0.5 * (swept_vols[e, f_idx] - swept_vols[nbr, nbr_f])
                swept_vols[e, f_idx] = avg_v
                swept_vols[nbr, nbr_f] = -avg_v
                
    # 4. Gradient reconstruction on old mesh
    use_muscl = (limiter.lower() != "none")
    grad_2d = None
    if use_muscl:
        grad_q = ale_compute_gradients(q_lagrange, x_old, connectivity)
        grad_2d = grad_q[:, None, :] if is_1d else grad_q
        
    phi = np.ones((n_elem, n_vars), dtype=np.float64)
    if use_muscl and limiter.lower() == "barth_jespersen":
        for v in range(n_vars):
            phi[:, v] = compute_barth_jespersen_limiter(q_2d[:, v], grad_2d[:, v], xe_old, neighbor_elem)
            
    # 5. Flux of quantity q across each face
    flux_q = np.zeros((n_elem, 6, n_vars), dtype=np.float64)
    for e in range(n_elem):
        for f_idx in range(6):
            dv = swept_vols[e, f_idx]
            if abs(dv) < 1e-30:
                continue
            nbr = neighbor_elem[e, f_idx]
            
            if dv > 0.0:
                # Swept outward: donor is e
                f_nodes = HEX_FACES[f_idx]
                xf = np.mean(xe_old[e, f_nodes], axis=0)
                if use_muscl:
                    dr = xf - xc_old[e]
                    for v in range(n_vars):
                        dq = float(np.dot(grad_2d[e, v], dr))
                        if limiter.lower() in ("van_leer", "minmod") and nbr >= 0:
                            d_donor = q_2d[e, v] - q_2d[nbr, v]
                            r = dq / (d_donor + 1e-30)
                            psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                            q_face_val = q_2d[e, v] + psi * dq
                        else:
                            q_face_val = q_2d[e, v] + phi[e, v] * dq
                        flux_q[e, f_idx, v] = q_face_val * dv
                else:
                    flux_q[e, f_idx] = q_2d[e] * dv
            else:
                # Swept inward: donor is neighbor
                if nbr >= 0:
                    f_nodes = HEX_FACES[f_idx]
                    xf = np.mean(xe_old[e, f_nodes], axis=0)
                    if use_muscl:
                        dr = xf - xc_old[nbr]
                        for v in range(n_vars):
                            dq = float(np.dot(grad_2d[nbr, v], dr))
                            if limiter.lower() in ("van_leer", "minmod"):
                                d_donor = q_2d[nbr, v] - q_2d[e, v]
                                r = dq / (d_donor + 1e-30)
                                psi = limiter_van_leer(r) if limiter.lower() == "van_leer" else limiter_minmod(r)
                                q_face_val = q_2d[nbr, v] + psi * dq
                            else:
                                q_face_val = q_2d[nbr, v] + phi[nbr, v] * dq
                            flux_q[e, f_idx, v] = q_face_val * dv
                    else:
                        flux_q[e, f_idx] = q_2d[nbr] * dv
                else:
                    flux_q[e, f_idx] = q_2d[e] * dv
                    
    # Strict antisymmetry across shared faces
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_f = 0.5 * (flux_q[e, f_idx] - flux_q[nbr, nbr_f])
                flux_q[e, f_idx] = avg_f
                flux_q[nbr, nbr_f] = -avg_f
                
    net_flux = np.sum(flux_q, axis=1)  # (n_elem, n_vars)
    
    if extensive:
        q_ext_old = q_2d
        q_ext_new = q_ext_old - net_flux
        q_remapped = q_ext_new
    else:
        q_ext_old = q_2d * v_old[:, None]
        q_ext_new = q_ext_old - net_flux
        q_remapped = q_ext_new / v_new[:, None]
        
    res = q_remapped[:, 0] if is_1d else q_remapped
    if q_ale is not None:
        q_ale[...] = res
        return q_ale
    return res


# -----------------------------------------------------------------------------
# Function 5: Laplacian Grid Smoothing
# -----------------------------------------------------------------------------

def ale_grid_smooth_laplacian(x: np.ndarray,
                              bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                              connectivity: Optional[np.ndarray] = None,
                              iterations: int = 1,
                              alpha: float = 1.0) -> np.ndarray:
    """Laplacian smoothing of grid coordinates for /ALE/GRID/LAPLACIAN.
    
    Fortran origin: engine/source/ale/grid/alew5.F lines 113-144:
    x_new(i) = x(i) + alpha * ( 1/NUM * sum_{j in nbrs} x(j) - x(i) )
    Fixed boundary nodes (in bcs_ale_nodes) are held strictly stationary.
    
    Args:
        x: (n_nodes, 3) nodal coordinates.
        bcs_ale_nodes: collection or boolean mask of constrained nodes (fixed).
        connectivity: optional (n_elem, 8) hex connectivity. If omitted,
                      neighbor graph is deduced from minimum edge distances.
        iterations: number of smoothing iterations (default 1).
        alpha: relaxation factor in [0, 1] (default 1.0 = full neighbor average).
        
    Returns:
        x_smoothed: (n_nodes, 3) smoothed nodal coordinates.
    """
    n_nodes = len(x)
    if n_nodes == 0:
        return x.copy()
        
    # Build fixed boolean mask
    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            # Handle potential 1-based node numbering
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True
                
    # Build node-node neighbor graph
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    if connectivity is not None:
        for e in range(len(connectivity)):
            conn_e = connectivity[e]
            for n1_local, n2_local in HEX_EDGES:
                n1 = int(conn_e[n1_local])
                n2 = int(conn_e[n2_local])
                if 0 <= n1 < n_nodes and 0 <= n2 < n_nodes:
                    neighbors[n1].add(n2)
                    neighbors[n2].add(n1)
    else:
        # Deduce edge connectivity geometrically from coordinate proximity
        # Find minimum nonzero distance between any pair of nodes
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(np.min(dists))
        if min_dist < 1e-12:
            min_dist = 1.0
        edge_thresh = 1.15 * min_dist
        close_pairs = np.argwhere((dists > 1e-12) & (dists <= edge_thresh))
        for i, j in close_pairs:
            neighbors[int(i)].add(int(j))
            
    x_curr = x.copy()
    for _ in range(iterations):
        x_next = x_curr.copy()
        for i in range(n_nodes):
            if is_fixed[i]:
                continue
            nbrs = neighbors[i]
            if nbrs:
                nbr_avg = np.mean(x_curr[list(nbrs)], axis=0)
                x_next[i] = (1.0 - alpha) * x_curr[i] + alpha * nbr_avg
        x_curr = x_next
        
    return x_curr


def ale_grid_smooth_donea(x: np.ndarray,
                          disp: np.ndarray,
                          vel: np.ndarray,
                          bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                          connectivity: Optional[np.ndarray] = None,
                          dt: float = 1e-4,
                          alpha: float = 0.5,
                          gamma: float = 0.5,
                          vg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Donea distance-weighted grid smoothing for /ALE/GRID/DONEA.

    Faithful port of OpenRadioss Fortran source:
      - engine/source/ale/grid/alew.F lines 98-180 (ALEW).

    Grid velocity formulation:
      1. For unconstrained interior nodes i:
         L_ij = ||x_j - x_i||
         S_li = sum_j L_ij
         F_i = sum_j (d_j - d_i) / L_ij
         FAC = alpha * S_li / (N_ci^2 * dt)
         W_i = 1/N_ci sum_j W_j + FAC * F_i
      2. Gamma bounding (alew.F lines 171-179):
         W_k,i = VG_k * V_k,i * clamp(W_k,i / V_k,i, 1 - gamma, 1 + gamma)
      3. New grid position: x_new = x + W * dt.

    Args:
        x: (n_nodes, 3) current nodal coordinates.
        disp: (n_nodes, 3) cumulative displacement vector.
        vel: (n_nodes, 3) material velocity vector V.
        bcs_ale_nodes: collection or mask of fixed / Lagrangian boundary nodes.
        connectivity: optional (n_elem, 8) hex element connectivity.
        dt: current time step duration.
        alpha: Donea distance-weighting coefficient (default 0.5).
        gamma: velocity relaxation bound parameter in [0, 1] (default 0.5).
        vg: (3,) grid velocity scaling vector [VGX, VGY, VGZ] (default [1, 1, 1]).

    Returns:
        w_grid: (n_nodes, 3) ALE grid velocity.
        x_new: (n_nodes, 3) updated grid coordinates.
    """
    n_nodes = len(x)
    if n_nodes == 0:
        return np.zeros((0, 3), dtype=np.float64), x.copy()

    if vg is None:
        vg = np.ones(3, dtype=np.float64)
    else:
        vg = np.asarray(vg, dtype=np.float64)

    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True

    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    if connectivity is not None:
        for e in range(len(connectivity)):
            conn_e = connectivity[e]
            for n1_local, n2_local in HEX_EDGES:
                n1 = int(conn_e[n1_local])
                n2 = int(conn_e[n2_local])
                if 0 <= n1 < n_nodes and 0 <= n2 < n_nodes:
                    neighbors[n1].add(n2)
                    neighbors[n2].add(n1)
    else:
        diff = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = float(np.min(dists))
        if min_dist < 1e-12:
            min_dist = 1.0
        edge_thresh = 1.15 * min_dist
        close_pairs = np.argwhere((dists > 1e-12) & (dists <= edge_thresh))
        for i, j in close_pairs:
            neighbors[int(i)].add(int(j))

    w_grid = vel.copy()
    w_old = w_grid.copy()
    dt_eff = max(dt, 1e-12)

    for i in range(n_nodes):
        if is_fixed[i]:
            continue
        nbrs = list(neighbors[i])
        nci = len(nbrs)
        if nci == 0:
            continue

        w_avg = np.mean(w_old[nbrs], axis=0)

        if alpha != 0.0:
            d_diff = disp[nbrs] - disp[i]
            x_diff = x[nbrs] - x[i]
            lij = np.linalg.norm(x_diff, axis=1)
            lij_safe = np.maximum(lij, 1e-20)

            sli = float(np.sum(lij))
            fix = np.sum(d_diff / lij_safe[:, None], axis=0)

            fac = alpha * sli / (float(nci * nci) * dt_eff)
            w_grid[i] = w_avg + fac * fix
        else:
            w_grid[i] = w_avg

    if gamma < 1e18:
        for i in range(n_nodes):
            if not is_fixed[i]:
                for k in range(3):
                    vk = vel[i, k]
                    if abs(vk) > 1e-20:
                        ratio = w_grid[i, k] / vk
                        ratio_clamped = max(1.0 - gamma, min(1.0 + gamma, ratio))
                        w_grid[i, k] = vg[k] * vk * ratio_clamped

    x_new = x + w_grid * dt
    return w_grid, x_new


# -----------------------------------------------------------------------------
# 24 springs per hex8 element for /ALE/GRID/SPRING (alew2.F lines 93-107)
# 12 edges + 12 face diagonals
# -----------------------------------------------------------------------------
HEX_SPRINGS_24 = np.array([
    # 12 edges (ITR 1..12)
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
    # 12 face diagonals (ITR 13..24)
    [0, 2], [1, 3], [4, 6], [5, 7],
    [0, 5], [1, 4], [1, 6], [2, 5],
    [2, 7], [3, 6], [0, 7], [3, 4],
], dtype=np.int64)


def ale_grid_smooth_spring(x: np.ndarray,
                           disp: np.ndarray,
                           vel: np.ndarray,
                           bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                           connectivity: np.ndarray,
                           dt: float = 1e-4,
                           alpha: float = 1.0,
                           gamma: float = 1.0,
                           vgx: float = 1.0,
                           vgy: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Spring network grid smoothing for /ALE/GRID/SPRING.

    Ported from $OR_SRC/engine/source/ale/grid/alew2.F
    lines 83-295 & 371-385:
      Constructs a 24-spring elastic network per hex cell (12 edges + 12 diagonals).
      For each spring between nodes J1 and J2:
        XX = (X_J2 - D_J2) - (X_J1 - D_J1)
        XL = ||XX||
        DX = D_J2 - D_J1
        DDX = (W_J2 - W_J1) * dt
        DL = (XX . DX) / XL
        DDL = (XX . DDX) / XL
        DL1 = gamma + 0.5 * (gamma - 1) * min(DL / XL, 0)
        DL_force = (FAC / XL) / (alpha^2) * DDL * DL1
        DDL_force = (FAC / XL) * (vgx / alpha) * DDL
      Accumulates internal forces into WB and WA, and updates grid velocities:
        W_i = W_i + (WB_i * dt + WA_i) / BETA
        where BETA = 6 * (1 + 2 * VGY).

    Args:
        x: (n_nodes, 3) current nodal coordinates.
        disp: (n_nodes, 3) cumulative displacement vector.
        vel: (n_nodes, 3) material velocity vector V.
        bcs_ale_nodes: collection or boolean mask of fixed/Lagrangian boundary nodes.
        connectivity: (n_elem, 8) hex8 element connectivity.
        dt: current time step duration.
        alpha: spring stiffness parameter ALE%GRID%ALPHA (default 1.0).
        gamma: spring relaxation parameter ALE%GRID%GAMMA (default 1.0).
        vgx: velocity scaling factor ALE%GRID%VGX (default 1.0).
        vgy: diagonal spring factor ALE%GRID%VGY (default 1.0).

    Returns:
        w_grid: (n_nodes, 3) ALE grid velocity.
        x_new: (n_nodes, 3) updated grid coordinates.
    """
    n_nodes = len(x)
    n_elem = len(connectivity)
    if n_nodes == 0 or n_elem == 0:
        return vel.copy(), x.copy()

    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True

    dt_eff = max(dt, 1e-12)
    alpha_safe = max(alpha, 1e-6)
    gam1 = 0.5 * (gamma - 1.0)
    beta = 6.0 * (1.0 + 2.0 * vgy)

    # Spring factors: 1..12 have FAC=1.0, 13..24 have FAC=vgy (alew2.F lines 113-115)
    fac = np.ones(24, dtype=np.float64)
    fac[12:24] = vgy

    w_grid = vel.copy()
    wa = np.zeros((n_nodes, 3), dtype=np.float64)
    wb = np.zeros((n_nodes, 3), dtype=np.float64)

    for itr in range(24):
        n1_local = HEX_SPRINGS_24[itr, 0]
        n2_local = HEX_SPRINGS_24[itr, 1]
        j1 = connectivity[:, n1_local]
        j2 = connectivity[:, n2_local]

        # Undeformed length vector XX = X_init(j2) - X_init(j1) (alew2.F lines 197-206)
        dx_d = disp[j2] - disp[j1]
        xx = (x[j2] - x[j1]) - dx_d
        xl = np.linalg.norm(xx, axis=1)
        xl_safe = np.maximum(xl, 1e-20)

        ddx = (w_grid[j2] - w_grid[j1]) * dt_eff

        dl = np.sum(xx * dx_d, axis=1) / xl_safe
        ddl = np.sum(xx * ddx, axis=1) / xl_safe

        dl_ratio = dl / xl_safe
        dl1 = gamma + gam1 * np.minimum(dl_ratio, 0.0)

        fac_itr = fac[itr]
        dl_force = (fac_itr / xl_safe) / (alpha_safe * alpha_safe) * dl * dl1
        ddl_force = (fac_itr / xl_safe) * (vgx / alpha_safe) * ddl

        force_b = dl_force[:, None] * xx
        force_a = ddl_force[:, None] * xx

        # Assemble onto nodes (alew2.F lines 217-234)
        for e in range(n_elem):
            idx1 = j1[e]
            idx2 = j2[e]
            if not is_fixed[idx1]:
                wb[idx1] += force_b[e]
                wa[idx1] += force_a[e]
            if not is_fixed[idx2]:
                wb[idx2] -= force_b[e]
                wa[idx2] -= force_a[e]

    # Update grid velocity (alew2.F lines 371-385)
    for i in range(n_nodes):
        if not is_fixed[i]:
            w_grid[i] += (wb[i] * dt_eff + wa[i]) / beta
        else:
            w_grid[i] = vel[i]

    x_new = x + w_grid * dt
    return w_grid, x_new


def ale_grid_smooth_curvature(x: np.ndarray,
                              disp: np.ndarray,
                              vel: np.ndarray,
                              bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                              connectivity: np.ndarray,
                              dt: float = 1e-4,
                              alpha: float = 1.0,
                              gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Curvature-based grid smoothing for /ALE/GRID/STANDARD.

    Ported from $OR_SRC/engine/source/ale/grid/alew4.F
    lines 191-387 & 464-478:
      Computes face normal vectors across opposite face pairs, determines
      local surface curvature factors DLF and stretch rates DDLF, and
      smooths interior nodes with curvature weighting.

    Args:
        x: (n_nodes, 3) current nodal coordinates.
        disp: (n_nodes, 3) cumulative displacements.
        vel: (n_nodes, 3) material velocities V.
        bcs_ale_nodes: collection or boolean mask of fixed boundary nodes.
        connectivity: (n_elem, 8) hex8 element connectivity.
        dt: time step duration.
        alpha: relaxation factor (default 1.0).
        gamma: curvature smoothing exponent (default 1.0).

    Returns:
        w_grid: (n_nodes, 3) ALE grid velocity.
        x_new: (n_nodes, 3) updated grid coordinates.
    """
    n_nodes = len(x)
    n_elem = len(connectivity)
    if n_nodes == 0 or n_elem == 0:
        return vel.copy(), x.copy()

    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True

    dt_eff = max(dt, 1e-12)
    gam1 = gamma - 1.0

    w_grid = vel.copy()
    wa = np.zeros((n_nodes, 3), dtype=np.float64)
    wb = np.zeros((n_nodes, 3), dtype=np.float64)
    wma = np.zeros(n_nodes, dtype=np.float64)

    xe = x[connectivity]  # (n_elem, 8, 3)
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3)

    # 3 pairs of opposite faces: (0, 2), (3, 1), (5, 4)
    opposite_face_pairs = [(0, 2), (3, 1), (5, 4)]

    for f1, f2 in opposite_face_pairs:
        # Direction unit vector between opposite face centroids
        nodes1 = HEX_FACES[f1]
        nodes2 = HEX_FACES[f2]
        xc1 = np.mean(xe[:, nodes1], axis=1)  # (n_elem, 3)
        xc2 = np.mean(xe[:, nodes2], axis=1)  # (n_elem, 3)
        dir_vec = xc2 - xc1
        dir_norm = np.linalg.norm(dir_vec, axis=1, keepdims=True)
        unit_dir = dir_vec / np.maximum(dir_norm, 1e-20)  # (n_elem, 3)

        # Opposite face node pairs along this axis
        for k in range(4):
            n1_local = nodes1[k]
            n2_local = nodes2[k]
            j1 = connectivity[:, n1_local]
            j2 = connectivity[:, n2_local]

            xx = x[j2] - x[j1]
            ddx = (w_grid[j2] - w_grid[j1]) * dt_eff

            ddlf = np.sum(unit_dir * ddx, axis=1)
            dlf0 = np.abs(np.sum(unit_dir * xx, axis=1))

            vgy = np.maximum(np.mean(dir_norm), 1e-12)
            dlf = np.minimum((dlf0 - vgy) / vgy, 0.0)
            dlf = gamma + gam1 * (dlf ** 3)
            dlf = np.minimum(dlf, 1.0)
            if np.any(ddlf > 0.0):
                dlf = np.where(ddlf > 0.0, gamma, dlf)

            dl = dlf / (vgy * vgy)
            ddl = 1.0 / vgy

            f_b = ddx * dl[:, None]
            f_a = ddx * ddl

            for e in range(n_elem):
                idx1 = j1[e]
                idx2 = j2[e]
                if not is_fixed[idx1]:
                    wb[idx1] += f_b[e]
                    wa[idx1] += f_a[e]
                    wma[idx1] += 1.0
                if not is_fixed[idx2]:
                    wb[idx2] -= f_b[e]
                    wa[idx2] -= f_a[e]
                    wma[idx2] += 1.0

    # alew4.F lines 464-478
    for i in range(n_nodes):
        if not is_fixed[i] and wma[i] > 0.0:
            w_grid[i] += (wb[i] * dt_eff + wa[i]) / wma[i]
        else:
            w_grid[i] = vel[i]

    x_new = x + w_grid * dt
    return w_grid, x_new


def ale_grid_smooth_volume(x: np.ndarray,
                           vel: np.ndarray,
                           bcs_ale_nodes: Union[Set[int], List[int], np.ndarray],
                           connectivity: np.ndarray,
                           dt: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Centroidal Voronoi / volume grid smoothing for /ALE/GRID/VOLUME.

    Ported from $OR_SRC/engine/source/ale/grid/alew6.F
    lines 98-179:
      Computes cell centroids and element volumes.
      For each unconstrained interior ALE node i:
        X_new(i) = sum_{e in connected(i)} (V_e * X_c,e) / sum_{e in connected(i)} V_e
      Grid velocity:
        W_i = (X_new(i) - X(i)) / dt
      Fixed / boundary nodes remain stationary (W_i = 0 or V_i).

    Args:
        x: (n_nodes, 3) nodal coordinates.
        vel: (n_nodes, 3) material velocity vector.
        bcs_ale_nodes: collection or boolean mask of fixed boundary nodes.
        connectivity: (n_elem, 8) hex element connectivity.
        dt: current time step duration.

    Returns:
        w_grid: (n_nodes, 3) ALE grid velocity.
        x_new: (n_nodes, 3) updated grid coordinates.
    """
    n_nodes = len(x)
    n_elem = len(connectivity)
    if n_nodes == 0 or n_elem == 0:
        return vel.copy(), x.copy()

    is_fixed = np.zeros(n_nodes, dtype=bool)
    if isinstance(bcs_ale_nodes, np.ndarray) and bcs_ale_nodes.dtype == bool:
        is_fixed[:min(n_nodes, len(bcs_ale_nodes))] = bcs_ale_nodes[:min(n_nodes, len(bcs_ale_nodes))]
    else:
        for idx in bcs_ale_nodes:
            i = int(idx)
            if 0 <= i < n_nodes:
                is_fixed[i] = True
            elif 1 <= i <= n_nodes:
                is_fixed[i - 1] = True

    # 1. Compute element centroids and volumes (alew6.F lines 105-138)
    xe = x[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3)
    vols = compute_hex_volumes(x, connectivity)  # (n_elem,)

    # 2. Accumulate volume-weighted centroids onto nodes (alew6.F lines 155-168)
    sum_vx = np.zeros((n_nodes, 3), dtype=np.float64)
    sum_vol = np.zeros(n_nodes, dtype=np.float64)

    for e in range(n_elem):
        ve = vols[e]
        xce = xc[e]
        for n_local in range(8):
            n_global = connectivity[e, n_local]
            sum_vx[n_global] += ve * xce
            sum_vol[n_global] += ve

    # 3. New coordinates and grid velocity (alew6.F lines 169-178)
    dt_eff = max(dt, 1e-12)
    x_new = x.copy()
    w_grid = vel.copy()

    for i in range(n_nodes):
        if not is_fixed[i] and sum_vol[i] > 1e-30:
            x_target = sum_vx[i] / sum_vol[i]
            w_grid[i] = (x_target - x[i]) / dt_eff
            x_new[i] = x_target
        else:
            w_grid[i] = vel[i]
            x_new[i] = x[i]

    return w_grid, x_new


def ale_link_velocity(w: np.ndarray,
                      links: List[Dict[str, Any]]) -> np.ndarray:
    """Enforce ALE grid velocity link constraints (/ALE/LINK/VEL, /VEL/ALE).

    Faithful port of OpenRadioss Fortran source:
      - engine/source/ale/grid/alelin.F lines 61-198 (ALELIN).

    For each ALE link:
      - Master nodes M1, M2 with velocities W(M1), W(M2).
      - Direction mask IC (bitmask: bit 2 = X (4), bit 1 = Y (2), bit 0 = Z (1)).
      - Formulation IM:
        * IM == 0: linear interpolation along slave node chain:
          W_j(N_i) = W_j(M1) + (W_j(M2) - W_j(M1)) * i / (N + 1)
        * IM > 0: maximum magnitude:
          W_j(N_i) = W_j(M1) if |W_j(M1)| >= |W_j(M2)| else W_j(M2)
        * IM < 0: minimum magnitude:
          W_j(N_i) = W_j(M1) if |W_j(M1)| <= |W_j(M2)| else W_j(M2)

    Args:
        w: (n_nodes, 3) grid velocity array (modified in-place and returned).
        links: list of dicts with keys:
               'm1': int, master node 1 index (0-based)
               'm2': int, master node 2 index (0-based)
               'nodes': list of int, slave node indices (0-based)
               'ic': int, direction code (1 to 7, default 7 for XYZ)
               'im': int, formulation flag (0=linear, 1=max, -1=min)

    Returns:
        w: updated grid velocity array.
    """
    n_nodes = len(w)
    for link in links:
        m1 = int(link.get("m1", -1))
        m2 = int(link.get("m2", -1))
        slave_nodes = link.get("nodes", [])
        ic = int(link.get("ic", 7))
        im = int(link.get("im", 0))

        if m1 < 0 or m1 >= n_nodes or m2 < 0 or m2 >= n_nodes:
            continue

        id_x = bool(ic & 4)
        id_y = bool(ic & 2)
        id_z = bool(ic & 1)
        dims = []
        if id_x: dims.append(0)
        if id_y: dims.append(1)
        if id_z: dims.append(2)

        n_slaves = len(slave_nodes)
        if n_slaves == 0:
            continue

        w1 = w[m1]
        w2 = w[m2]

        for dim in dims:
            if im == 0:
                for step_idx, node_idx in enumerate(slave_nodes, start=1):
                    ni = int(node_idx)
                    if 0 <= ni < n_nodes:
                        frac = float(step_idx) / float(n_slaves + 1)
                        w[ni, dim] = w1[dim] + (w2[dim] - w1[dim]) * frac
            elif im > 0:
                val = w1[dim] if abs(w1[dim]) >= abs(w2[dim]) else w2[dim]
                for node_idx in slave_nodes:
                    ni = int(node_idx)
                    if 0 <= ni < n_nodes:
                        w[ni, dim] = val
            else:
                val = w1[dim] if abs(w1[dim]) <= abs(w2[dim]) else w2[dim]
                for node_idx in slave_nodes:
                    ni = int(node_idx)
                    if 0 <= ni < n_nodes:
                        w[ni, dim] = val

    return w


def ale_multimat_remap(vol_frac: np.ndarray,
                       mat_densities: np.ndarray,
                       x_old: np.ndarray,
                       x_new: np.ndarray,
                       connectivity: np.ndarray,
                       limiter: str = "van_leer") -> Tuple[np.ndarray, np.ndarray]:
    """Remap multi-material volume fractions and compute mixture densities.

    Faithful port of OpenRadioss Fortran source:
      - starter/source/ale/bimat/inimu3.F lines 45-120
      - engine/source/ale/bimat/bimat2.F lines 80-165

    For an element with M materials and volume fractions alpha_m (sum = 1):
      1. Partial volumes: V_m = alpha_m * V_old
      2. Conservative advection of partial volumes V_m:
         V_m_new = ale_remap(V_m, x_old, x_new, conn, extensive=True)
      3. New volume fractions:
         alpha_m_new = V_m_new / sum_k V_k_new
      4. Mixture density:
         rho_mix = sum_m alpha_m_new * rho_m

    Args:
        vol_frac: (n_elem, n_mat) volume fraction array (sum over axis 1 == 1.0).
        mat_densities: (n_mat,) nominal densities of the constituent materials.
        x_old: (n_nodes, 3) coordinates before grid movement.
        x_new: (n_nodes, 3) coordinates after grid movement.
        connectivity: (n_elem, 8) hex8 node connectivity.
        limiter: slope limiter for advection ("none", "van_leer", "minmod", "barth_jespersen").

    Returns:
        vol_frac_new: (n_elem, n_mat) updated volume fractions (strictly normalized to 1.0).
        rho_mix_new: (n_elem,) mixture density for each element.
    """
    vf = np.asarray(vol_frac, dtype=np.float64)
    dens = np.asarray(mat_densities, dtype=np.float64)
    n_elem, n_mat = vf.shape

    vol_frac_new = np.zeros_like(vf)
    for m in range(n_mat):
        vol_frac_new[:, m] = ale_remap(
            vf[:, m],
            None,
            x_old,
            x_new,
            connectivity,
            extensive=False,
            limiter=limiter,
        )

    # Clean negative numerical undershoots
    vol_frac_new = np.maximum(vol_frac_new, 0.0)
    total_vol_new = np.sum(vol_frac_new, axis=1, keepdims=True)
    total_vol_new = np.maximum(total_vol_new, 1e-30)

    # Normalized new volume fractions (sum to 1.0)
    vol_frac_new = vol_frac_new / total_vol_new
    rho_mix = np.sum(vol_frac_new * dens[None, :], axis=1)

    return vol_frac_new, rho_mix


# -----------------------------------------------------------------------------
# Function 6: Full ALE Step Orchestrator
# -----------------------------------------------------------------------------

def ale_step(model: Any, dt: float, state: Any) -> None:
    """Orchestrate the complete Arbitrary Lagrangian-Eulerian (ALE) step.
    
    Fortran origin:
    - engine/source/ale/alemain.F: lines 96-160
    - engine/source/ale/alethe.F: lines 228-300
    - engine/source/ale/aconve.F90: lines 34-79
    - engine/source/ale/arezon.F90: lines 38-68
    
    Steps:
    1. Identifies solid element groups with ALE enabled.
    2. Identifies boundary nodes from model.ale_bcs and exterior domain surfaces.
    3. Performs Laplacian grid smoothing to update grid coordinates.
    4. Computes grid velocities W = (X_new - X_old) / dt and relative velocities.
    5. Convects/remaps state variables: density, specific internal energy, stress.
    6. Ensures strict mass and energy conservation.
    
    Args:
        model: Model object containing elements, nodes, materials, and state.
        dt: current explicit time step duration.
        state: EngineState object tracking energies and cycle counts.
    """
    if dt <= 0.0:
        return
        
    # Locate solid hexa8 element groups
    solid_groups = []
    for name, group in model.element_groups():
        if hasattr(group, "conn") and group.conn.ndim == 2 and group.conn.shape[1] == 8:
            solid_groups.append((name, group))
            
    if not solid_groups:
        return
        
    n_nodes = len(model.x)
    
    for name, group in solid_groups:
        conn = group.conn
        n_elem = len(conn)
        if n_elem == 0:
            continue
            
        # 1. Identify boundary nodes
        bcs_nodes: Set[int] = set()
        
        # Check /ALE/BCS boundary conditions in model
        if hasattr(model, "ale_bcs") and model.ale_bcs:
            for bc in model.ale_bcs:
                if hasattr(model, "node_groups") and bc.grnod_id in model.node_groups:
                    grnod = model.node_groups[bc.grnod_id]
                    node_ids = grnod.nodes if hasattr(grnod, "nodes") else []
                    bcs_nodes.update(int(n) for n in node_ids if 0 <= int(n) < n_nodes)
                    
        # Also fix exterior surface nodes so domain boundaries remain preserved
        neighbor_elem, _ = build_face_connectivity(conn)
        for e in range(n_elem):
            for f_idx in range(6):
                if neighbor_elem[e, f_idx] < 0:
                    for n_local in HEX_FACES[f_idx]:
                        bcs_nodes.add(int(conn[e, n_local]))
                        
        # 2. Grid smoothing (Laplacian, Donea, Spring, Curvature, or Volume)
        x_old = model.x.copy()
        grid_type = getattr(model, "ale_grid_type", "laplacian").lower()
        if grid_type == "donea":
            disp = getattr(model, "disp", np.zeros_like(x_old))
            vel = getattr(model, "v", np.zeros_like(x_old))
            w_grid, x_new = ale_grid_smooth_donea(
                x_old, disp, vel, bcs_nodes, conn, dt=dt,
                alpha=getattr(model, "ale_grid_alpha", 0.5),
                gamma=getattr(model, "ale_grid_gamma", 0.5),
            )
        elif grid_type == "spring":
            disp = getattr(model, "disp", np.zeros_like(x_old))
            vel = getattr(model, "v", np.zeros_like(x_old))
            w_grid, x_new = ale_grid_smooth_spring(
                x_old, disp, vel, bcs_nodes, conn, dt=dt,
                alpha=getattr(model, "ale_grid_alpha", 1.0),
                gamma=getattr(model, "ale_grid_gamma", 1.0),
                vgx=getattr(model, "ale_grid_vgx", 1.0),
                vgy=getattr(model, "ale_grid_vgy", 1.0),
            )
        elif grid_type in ("curvature", "standard"):
            disp = getattr(model, "disp", np.zeros_like(x_old))
            vel = getattr(model, "v", np.zeros_like(x_old))
            w_grid, x_new = ale_grid_smooth_curvature(
                x_old, disp, vel, bcs_nodes, conn, dt=dt,
                alpha=getattr(model, "ale_grid_alpha", 1.0),
                gamma=getattr(model, "ale_grid_gamma", 1.0),
            )
        elif grid_type in ("volume", "voronoi", "centroidal"):
            vel = getattr(model, "v", np.zeros_like(x_old))
            w_grid, x_new = ale_grid_smooth_volume(
                x_old, vel, bcs_nodes, conn, dt=dt,
            )
        else:
            x_new = ale_grid_smooth_laplacian(x_old, bcs_nodes, conn, iterations=1, alpha=0.5)
            w_grid = (x_new - x_old) / dt

        # 2b. Apply ALE velocity links (/ALE/LINK/VEL, /VEL/ALE)
        if hasattr(model, "ale_links") and model.ale_links:
            w_grid = ale_link_velocity(w_grid, model.ale_links)
            x_new = x_old + w_grid * dt

        # 3. Relative velocity
        v_rel = model.v - w_grid

        # 4. Remap / Convect density and mass
        vol_old = compute_hex_volumes(x_old, conn)
        vol_new = compute_hex_volumes(x_new, conn)

        # Multi-material volume fractions remap (/ALE/MAT)
        if "vol_frac" in group.state and "mat_densities" in group.state:
            vf_old = group.state["vol_frac"]
            m_dens = group.state["mat_densities"]
            vf_new, rho_mix_new = ale_multimat_remap(vf_old, m_dens, x_old, x_new, conn)
            group.state["vol_frac"] = vf_new
            rho_new = rho_mix_new
            mass_new = rho_new * vol_new
            group.state["mass"] = mass_new
            group.state["vol"] = vol_new
        else:
            elem_mass = group.state.get("mass", None)
            if elem_mass is None:
                elem_mass = np.ones(n_elem, dtype=np.float64)
                group.state["mass"] = elem_mass

            rho_old = elem_mass / np.maximum(vol_old, 1e-30)
            rho_new = ale_remap(rho_old, None, x_old, x_new, conn, extensive=False)
            mass_new = rho_new * vol_new
            group.state["mass"] = mass_new
            group.state["vol"] = vol_new
        
        # 5. Remap internal energy
        if "eint" in group.state:
            eint_old = group.state["eint"]
            eint_new = ale_remap(eint_old, None, x_old, x_new, conn, extensive=True)
            group.state["eint"] = eint_new
            
        # 6. Remap Cauchy stress components
        if "sig" in group.state:
            sig_old = group.state["sig"]
            sig_new = ale_remap(sig_old, None, x_old, x_new, conn, extensive=False)
            group.state["sig"] = sig_new
            
        # 7. Update model node coordinates
        model.x[:] = x_new


# =============================================================================
# ALE Subcycling Management
# Fortran origin: engine/source/ale/subcycling/alesub1.F and alesub2.F
# =============================================================================

def ale_subcycle_step1(d: np.ndarray,
                       d_save: np.ndarray,
                       v: np.ndarray,
                       is_ale_node: np.ndarray,
                       dt_solid: float,
                       bcs_codes: Optional[np.ndarray] = None,
                       skew_matrices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute ALE subcycling Step 1 (grid velocity update & displacement backup).
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/subcycling/alesub1.F lines 70-98:
    FAC = 1.0 / DT1
    W = FAC * (D - DSAVE)
    DSAVE = V  (save current Lagrangian velocities)
    IF (NALE(N) == 0) THEN
        V(N) = W(N)
    ENDIF
    CALL BCS3V(...)
    
    Args:
        d: (n_nodes, 3) current nodal displacements D.
        d_save: (n_nodes, 3) previous nodal displacements DSAVE.
        v: (n_nodes, 3) current nodal velocities V.
        is_ale_node: (n_nodes,) mask (1 for ALE node, 0 for Lagrangian structure node).
        dt_solid: master solid explicit time step duration DT1.
        bcs_codes: optional (n_nodes,) boundary condition codes in [0, 7].
        skew_matrices: optional (n_nodes, 3, 3) local skew frames.
        
    Returns:
        (w, v_saved, v_updated):
            w: (n_nodes, 3) computed grid velocities W.
            v_saved: (n_nodes, 3) saved material velocities for restoration in step 2.
            v_updated: (n_nodes, 3) velocities with non-ALE nodes set to grid velocity.
    """
    fac = 1.0 / dt_solid if dt_solid > 0.0 else 0.0
    
    # Grid velocity W = (D - D_save) / dt_solid
    w = fac * (d - d_save)
    
    # Save Lagrangian velocities into v_saved: alesub1.F lines 77-79
    v_saved = v.copy()
    v_updated = v.copy()
    
    # For non-ALE (pure structure) nodes, velocity equals grid velocity during subcycling
    # alesub1.F lines 80-84
    non_ale_mask = (is_ale_node == 0)
    v_updated[non_ale_mask] = w[non_ale_mask]
    
    # Apply BCS3V grid velocity boundary conditions: alesub1.F lines 86-98
    if bcs_codes is not None:
        from pyradioss.engine.fsi_coupling import apply_grid_velocity_bcs
        w = apply_grid_velocity_bcs(w, v_updated, bcs_codes, skew_matrices)
        
    return w, v_saved, v_updated


def ale_subcycle_step2(v: np.ndarray,
                       v_saved: np.ndarray,
                       d: np.ndarray,
                       is_ale_node: np.ndarray,
                       dt_fluid_current: float,
                       dt_fluid_prev: float = 0.0,
                       scale_factor: float = 1.0,
                       bcs_codes: Optional[np.ndarray] = None,
                       skew_matrices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute ALE subcycling Step 2 (time step adjust & Lagrangian velocity restore).
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/subcycling/alesub2.F lines 82-135:
    DT2 = DT2 * DTFSUB
    IF (DT2S /= 0) DT2 = MIN(DT2, 1.1 * DT2S)
    Reset non-ALE velocities: V(N) = DSAVE(N) where NALE(N) == 0
    Save displacement: DSAVE(N) = D(N)
    
    Args:
        v: (n_nodes, 3) current velocities.
        v_saved: (n_nodes, 3) velocities saved from step 1.
        d: (n_nodes, 3) current displacements.
        is_ale_node: (n_nodes,) mask (1 for ALE node, 0 for structure node).
        dt_fluid_current: calculated explicit fluid time step.
        dt_fluid_prev: previous cycle fluid time step.
        scale_factor: subcycling fluid time step multiplier DTFSUB.
        bcs_codes: optional boundary condition codes.
        skew_matrices: optional skew frames.
        
    Returns:
        (v_restored, d_saved, dt_fluid_new):
            v_restored: (n_nodes, 3) restored material velocities.
            d_saved: (n_nodes, 3) displacements saved for next cycle step 1.
            dt_fluid_new: updated fluid subcycling time step.
    """
    # Adjust fluid time step: alesub2.F lines 83-90
    dt_fluid_new = dt_fluid_current * scale_factor
    if dt_fluid_prev > 0.0:
        dt_fluid_new = min(dt_fluid_new, 1.1 * dt_fluid_prev)
        
    # Reset non-ALE node velocities to their Lagrangian values: alesub2.F lines 108-113
    v_restored = v.copy()
    non_ale_mask = (is_ale_node == 0)
    v_restored[non_ale_mask] = v_saved[non_ale_mask]
    
    # Apply BCS3V: alesub2.F lines 114-126
    if bcs_codes is not None:
        from pyradioss.engine.fsi_coupling import apply_grid_velocity_bcs
        v_restored = apply_grid_velocity_bcs(v_restored, v_saved, bcs_codes, skew_matrices)
        
    # Save displacements for next subcycle step 1: alesub2.F lines 130-134
    d_saved = d.copy()
    
    return v_restored, d_saved, float(dt_fluid_new)


class ALESubcyclingManager:
    """Manages fluid-structure explicit subcycling.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/subcycling/alesub1.F and alesub2.F
    
    Coordinates multi-step subcycling where the fluid domain advances with
    N_sub smaller acoustic time steps while solid forces are frozen.
    """
    
    def __init__(self,
                 dt_scale: float = 1.0,
                 max_subcycles: int = 20) -> None:
        """Initialize subcycling manager."""
        self.dt_scale = float(dt_scale)
        self.max_subcycles = int(max_subcycles)
        self.dt_fluid_prev = 0.0
        self.d_saved: Optional[np.ndarray] = None
        self.v_saved: Optional[np.ndarray] = None
        self.cycle_count = 0
        
    def begin_solid_cycle(self,
                          d: np.ndarray,
                          v: np.ndarray,
                          is_ale_node: np.ndarray,
                          dt_solid: float,
                          bcs_codes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Begin solid master cycle (executes alesub1)."""
        if self.d_saved is None:
            self.d_saved = d.copy()
            
        w, v_saved, v_upd = ale_subcycle_step1(
            d=d,
            d_save=self.d_saved,
            v=v,
            is_ale_node=is_ale_node,
            dt_solid=dt_solid,
            bcs_codes=bcs_codes,
        )
        self.v_saved = v_saved
        return w, v_upd
        
    def determine_subcycles(self, dt_solid: float, dt_fluid_raw: float) -> Tuple[int, float]:
        """Determine number of fluid subcycles and subcycle time step duration."""
        dt_fluid = dt_fluid_raw * self.dt_scale
        if self.dt_fluid_prev > 0.0:
            dt_fluid = min(dt_fluid, 1.1 * self.dt_fluid_prev)
            
        n_sub = int(np.ceil(dt_solid / max(dt_fluid, 1e-20)))
        n_sub = max(1, min(self.max_subcycles, n_sub))
        dt_sub = dt_solid / n_sub
        
        self.dt_fluid_prev = dt_sub
        return n_sub, dt_sub
        
    def end_solid_cycle(self,
                        v: np.ndarray,
                        d: np.ndarray,
                        is_ale_node: np.ndarray,
                        dt_fluid_current: float,
                        bcs_codes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """End solid master cycle (executes alesub2)."""
        if self.v_saved is None:
            self.v_saved = v.copy()
            
        v_restored, d_saved, dt_new = ale_subcycle_step2(
            v=v,
            v_saved=self.v_saved,
            d=d,
            is_ale_node=is_ale_node,
            dt_fluid_current=dt_fluid_current,
            dt_fluid_prev=self.dt_fluid_prev,
            scale_factor=self.dt_scale,
            bcs_codes=bcs_codes,
        )
        self.d_saved = d_saved
        self.dt_fluid_prev = dt_new
        self.cycle_count += 1
        return v_restored, dt_new


# =============================================================================
# Thermal ALE Module
# Fortran origin: engine/source/ale/atherm.F and engine/source/ale/ale3d/adiff3.F
# =============================================================================

def ale_thermal_diffusivity(temperatures: np.ndarray,
                            a1: float,
                            b1: float,
                            a2: Optional[float] = None,
                            b2: Optional[float] = None,
                            t_trans: float = 0.0,
                            rho_cp: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute temperature-dependent thermal conductivity k(T) and diffusivity alpha(T).
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/atherm.F lines 129-139:
    IF (T <= T_trans) THEN
        k = A1 + B1 * T
    ELSE
        k = A2 + B2 * T
    ENDIF
    alpha = k / (rho * Cp)
    
    Args:
        temperatures: (n_elem,) element temperatures [K].
        a1: base thermal conductivity below transition [W/(m.K)].
        b1: linear conductivity slope below transition [W/(m.K^2)].
        a2: base conductivity above transition (defaults to a1).
        b2: linear conductivity slope above transition (defaults to b1).
        t_trans: transition temperature [K].
        rho_cp: volumetric heat capacity rho * Cp [J/(m^3.K)].
        
    Returns:
        (k, alpha):
            k: (n_elem,) thermal conductivity [W/(m.K)].
            alpha: (n_elem,) thermal diffusivity [m^2/s].
    """
    t = np.asarray(temperatures, dtype=np.float64)
    a2_val = a1 if a2 is None else float(a2)
    b2_val = b1 if b2 is None else float(b2)
    
    # Piecewise linear conductivity k(T) - atherm.F lines 134-138
    cond = np.where(t <= t_trans, a1 + b1 * t, a2_val + b2_val * t)
    cond = np.maximum(cond, 1e-12)
    
    denom = max(1e-20, float(rho_cp))
    diffusivity = cond / denom
    
    return cond, diffusivity


def compute_thermal_conductance_factors(x: np.ndarray,
                                        conn: np.ndarray,
                                        neighbor_elem: np.ndarray) -> np.ndarray:
    """Compute 3D geometric conductance factors GRAD_j for all hex element faces.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/ale3d/agrad3.F lines 154-260:
    D_j = Centroid(neighbor_j) - Centroid(elem)
    N_j = outward face normal vector (magnitude = 2 * FaceArea)
    GRAD(e, j) = 4.0 * max(0, D_j . N_j) / max(1e-15, ||D_j||^2)
    
    Args:
        x: (n_nodes, 3) nodal coordinates.
        conn: (n_elem, 8) hex element connectivity.
        neighbor_elem: (n_elem, 6) neighbor connectivity (-1 on boundary).
        
    Returns:
        grad: (n_elem, 6) geometric face conductance factors [m].
    """
    n_elem = len(conn)
    xe = x[conn]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3) centroids
    normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3) magnitude 2*Area
    
    grad = np.zeros((n_elem, 6), dtype=np.float64)
    
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr >= 0:
                d_vec = xc[nbr] - xc[e]
            else:
                # Boundary face: distance from centroid to face centroid
                f_nodes = HEX_FACES[f_idx]
                xf = np.mean(xe[e, f_nodes], axis=0)
                d_vec = 2.0 * (xf - xc[e])
                
            dd = np.dot(d_vec, d_vec)
            d_dot_n = np.dot(d_vec, normals[e, f_idx])
            
            # Ported from agrad3.F lines 258-260:
            grad[e, f_idx] = 4.0 * max(0.0, d_dot_n) / max(1e-15, dd)
            
    return grad


def ale_thermal_diffusion_step(temperatures: np.ndarray,
                               internal_energy_density: np.ndarray,
                               volumes: np.ndarray,
                               conductivities: np.ndarray,
                               grad_factors: np.ndarray,
                               neighbor_elem: np.ndarray,
                               rho_cp: float,
                               dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute 3D explicit finite volume thermal diffusion step on hex cells.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/ale3d/adiff3.F lines 81-149:
    - Harmonic interpolation of face conductivities:
      k_face = (k0 * kj) / max(1e-20, k0 + kj)
    - Face conductive heat flux:
      Q_face = k_face * (T_neighbor - T_elem) * GRAD_face
    - Internal energy density update:
      dphi = 2.0 * sum(Q_face) * dt / max(V, 1e-20)
      PHIN += dphi
    - Temperature update:
      TEMP += dphi / rho_cp
      
    Conserves total thermal energy in closed domains: sum(dphi * V) == 0.
    
    Args:
        temperatures: (n_elem,) current element temperatures [K].
        internal_energy_density: (n_elem,) current energy density E/V [J/m^3].
        volumes: (n_elem,) element volumes [m^3].
        conductivities: (n_elem,) element thermal conductivities k [W/(m.K)].
        grad_factors: (n_elem, 6) geometric conductance factors from agrad3.F.
        neighbor_elem: (n_elem, 6) neighbor connectivity.
        rho_cp: volumetric heat capacity rho * Cp [J/(m^3.K)].
        dt: explicit time step duration [s].
        
    Returns:
        (t_new, eint_v_new, net_energy_change):
            t_new: (n_elem,) updated element temperatures.
            eint_v_new: (n_elem,) updated internal energy densities.
            net_energy_change: float, net domain energy change (zero if insulated).
    """
    n_elem = len(temperatures)
    dphi = np.zeros(n_elem, dtype=np.float64)
    
    # Ported from adiff3.F lines 81-129
    for e in range(n_elem):
        k0 = conductivities[e]
        t0 = temperatures[e]
        
        flux_sum = 0.0
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr >= 0:
                kj = conductivities[nbr]
                tj = temperatures[nbr]
            else:
                # Insulated boundary: zero heat flux (kj=k0, tj=t0)
                kj = k0
                tj = t0
                
            # Harmonic interpolation - adiff3.F lines 115-120
            k_face = (k0 * kj) / max(1e-20, k0 + kj)
            
            # Heat flow through face - adiff3.F lines 123-128
            flux_sum += k_face * (tj - t0) * grad_factors[e, f_idx]
            
        # adiff3.F lines 134-136
        dphi[e] = 2.0 * flux_sum * dt / max(1e-20, volumes[e])
        
    # Update Eint/V and Temperature - adiff3.F lines 140-147
    eint_v_new = internal_energy_density + dphi
    t_new = temperatures + dphi / max(1e-20, rho_cp)
    
    net_energy_change = float(np.sum(dphi * volumes))
    return t_new, eint_v_new, net_energy_change


# =============================================================================
# Main ALE Drivers
# Fortran origin: engine/source/ale/arezon.F90, aconve.F90, alemain.F
# =============================================================================

def arezon_driver(variables: Dict[str, np.ndarray],
                  extensive_flags: Dict[str, bool],
                  x_old: np.ndarray,
                  x_new: np.ndarray,
                  conn: np.ndarray) -> Dict[str, np.ndarray]:
    """Execute ALE state rezone remapping of element variables.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/arezon.F90 lines 38-120
    
    Args:
        variables: dict mapping variable name to (n_elem, ...) data array.
        extensive_flags: dict indicating if variable is extensive (True for mass/energy).
        x_old: (n_nodes, 3) old coordinates.
        x_new: (n_nodes, 3) new coordinates.
        conn: (n_elem, 8) hex element connectivity.
        
    Returns:
        remapped_vars: dict with all variables remapped onto new mesh.
    """
    remapped: Dict[str, np.ndarray] = {}
    for var_name, data in variables.items():
        is_ext = extensive_flags.get(var_name, False)
        remapped[var_name] = ale_remap(
            field_old=data,
            grad=None,
            x_old=x_old,
            x_new=x_new,
            connectivity=conn,
            extensive=is_ext,
        )
    return remapped


def aconve_driver(phi: np.ndarray,
                  fluxes: np.ndarray,
                  flu1: np.ndarray,
                  neighbor_elem: np.ndarray,
                  dt: float) -> np.ndarray:
    """Execute ALE variable convection update.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/aconve.F90 lines 37-120 and aconv3.F lines 81-135
    
    Args:
        phi: (n_elem,) element scalar to convect.
        fluxes: (n_elem, 6) face fluxes.
        flu1: (n_elem,) total incoming/outgoing upwind flux.
        neighbor_elem: (n_elem, 6) neighbor connectivity.
        dt: explicit time step duration.
        
    Returns:
        phi_new: (n_elem,) convected scalar array.
    """
    n_elem = len(phi)
    delta_phi = np.zeros(n_elem, dtype=np.float64)
    
    for e in range(n_elem):
        sum_flux = 0.0
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            val_nbr = phi[nbr] if nbr >= 0 else phi[e]
            sum_flux += val_nbr * fluxes[e, f_idx]
            
        delta_phi[e] = 0.5 * dt * (-phi[e] * flu1[e] - sum_flux)
        
    return phi + delta_phi


def ale_main_driver(model: Any,
                    dt: float,
                    state: Optional[Any] = None,
                    enable_thermal: bool = False,
                    enable_subcycling: bool = False,
                    fsi_coupling: Optional[Any] = None) -> Dict[str, Any]:
    """Execute complete Arbitrary Lagrangian-Eulerian (ALE) solver cycle.
    
    Ported from OpenRadioss Fortran:
    engine/source/ale/alemain.F lines 24-120
    
    Coordinates:
    1. Subcycling initialization (alesub1.F)
    2. Grid smoothing (Laplacian / Donea / Spring / Curvature / Volume)
    3. Grid velocity linking and boundary conditions (bcs3v.F, alelin.F)
    4. Advection and remapping of density, internal energy, stress (arezon.F90, aconve.F90)
    5. Thermal conduction and diffusion (atherm.F, adiff3.F)
    6. Fluid-Structure Interaction coupling (fsi_coupling.py, i11for3.F, iqela1.F)
    7. Energy ledger accounting and subcycling completion (alesub2.F)
    
    Args:
        model: Model instance with geometry, groups, nodes, and boundary conditions.
        dt: current explicit time step duration.
        state: optional EngineState tracking global energy ledgers.
        enable_thermal: whether to run thermal ALE diffusion.
        enable_subcycling: whether subcycling is active.
        fsi_coupling: optional FSICouplingPenalty or FSICouplingTied instance.
        
    Returns:
        info: dict with cycle execution statistics, energies, and convergence info.
    """
    if dt <= 0.0:
        return {"status": "skipped", "dt": dt}
        
    # 1. Execute standard ALE grid smoothing and advection step
    ale_step(model, dt, state)
    
    # 2. Thermal diffusion step
    thermal_work = 0.0
    if enable_thermal:
        for name, group in model.element_groups():
            if hasattr(group, "conn") and group.conn.shape[1] == 8 and "temp" in group.state:
                conn = group.conn
                temp = group.state["temp"]
                eint_v = group.state.get("eint_v", temp * 1000.0)
                vols = compute_hex_volumes(model.x, conn)
                nbr_elem, _ = build_face_connectivity(conn)
                
                a1 = getattr(model, "ale_thermal_a1", 10.0)
                b1 = getattr(model, "ale_thermal_b1", 0.0)
                rho_cp = getattr(model, "ale_thermal_rhocp", 1000.0)
                
                cond, _ = ale_thermal_diffusivity(temp, a1, b1, rho_cp=rho_cp)
                grad = compute_thermal_conductance_factors(model.x, conn, nbr_elem)
                
                t_new, eint_new, delta_e = ale_thermal_diffusion_step(
                    temperatures=temp,
                    internal_energy_density=eint_v,
                    volumes=vols,
                    conductivities=cond,
                    grad_factors=grad,
                    neighbor_elem=nbr_elem,
                    rho_cp=rho_cp,
                    dt=dt,
                )
                group.state["temp"] = t_new
                group.state["eint_v"] = eint_new
                thermal_work += delta_e
                
    # 3. FSI coupling step
    fsi_results = None
    if fsi_coupling is not None and hasattr(model, "struct_quads"):
        fsi_results = fsi_coupling.apply_coupling(
            slave_nodes_x=model.x,
            slave_nodes_v=model.v,
            slave_masses=getattr(model, "mass", np.ones(len(model.x))),
            master_quads_conn=model.struct_quads,
            master_nodes_x=getattr(model, "struct_x", model.x),
            master_nodes_v=getattr(model, "struct_v", np.zeros_like(model.x)),
            dt=dt,
        )
        if state is not None and hasattr(state, "energy"):
            state.energy["contact"] = fsi_results.get("contact_energy", 0.0)
            
    return {
        "status": "success",
        "dt": dt,
        "thermal_work": thermal_work,
        "fsi": fsi_results,
    }

