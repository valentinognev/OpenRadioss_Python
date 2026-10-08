# pyradioss/engine/ale_multimaterial.py
"""
ALE Law 51 Multi-material Subsystem and Youngs Interface Reconstruction.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/ale51/afluxt.F: multi-material volume flux driver and species advection
- engine/source/ale/ale51/ale51_init.F: multi-material flux initialization and upwind driver
- engine/source/ale/ale51/ale51_upwind2.F: 2D upwind & donor-cell volume flux splitting
- engine/source/ale/ale51/ale51_upwind3.F: 3D upwind & donor-cell volume flux splitting
- engine/source/ale/ale51/ale51_antidiff2.F: 2D anti-diffusive volume fraction flux computation
- engine/source/ale/ale51/ale51_antidiff3.F: 3D anti-diffusive volume fraction flux computation
- engine/source/ale/alemuscl/ale51_gradient_reconstruction2.F: 2D species gradient reconstruction
- engine/source/ale/alemuscl/ale51_gradient_reconstruction.F: 3D species gradient reconstruction
- engine/source/ale/alemuscl/gradient_reconstruction2.F: least-squares gradient reconstruction (2D)
- engine/source/ale/alemuscl/gradient_reconstruction.F90: least-squares gradient reconstruction (3D)
- engine/source/ale/alemuscl/gradient_limitation2.F: Barth-Jespersen gradient limiter (2D)
- engine/source/ale/alemuscl/gradient_limitation.F: Barth-Jespersen gradient limiter (3D)
- engine/source/materials/mat/mat051/sigeps51.F90: multi-material iterative pressure relaxation & mixture laws
- common_source/modules/multimat_param_mod.F90: multi-material data structures & phase parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List, Any, Callable

import numpy as np

_EM20 = 1.0e-20
_EM15 = 1.0e-15
_EM12 = 1.0e-12
_EM4 = 1.0e-4
_EP10 = 1.0e10
_THIRD = 1.0 / 3.0
_HALF = 0.5
_FOURTH = 0.25


# =============================================================================
# Youngs (1982) Interface Reconstruction
# =============================================================================

def youngs_gradient_2d(
    alpha: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute volume fraction gradient using Youngs' 2D 9-point finite difference stencil.

    Reference:
    Youngs, D. L. (1982), "Time-Dependent Multi-material Flow with Large Fluid Distortion",
    in Numerical Methods for Fluid Dynamics, Academic Press.
    Also used in OpenRadioss MUSCL/interface schemes.

    For cell (i, j):
      grad_x = [(a_{i+1,j+1} + 2 a_{i+1,j} + a_{i+1,j-1}) - (a_{i-1,j+1} + 2 a_{i-1,j} + a_{i-1,j-1})] / (8 * dx)
      grad_y = [(a_{i+1,j+1} + 2 a_{i,j+1} + a_{i-1,j+1}) - (a_{i+1,j-1} + 2 a_{i,j-1} + a_{i-1,j-1})] / (8 * dy)

    Args:
        alpha: (ny, nx) 2D array of volume fractions in [0, 1].
        dx: cell spacing in x.
        dy: cell spacing in y.

    Returns:
        gx, gy: (ny, nx) arrays of gradient components.
    """
    ny, nx = alpha.shape
    # Pad boundary with replicated values (zero normal derivative at boundaries)
    a_pad = np.pad(alpha, ((1, 1), (1, 1)), mode="edge")

    # 9-point weighted stencil
    top_right = a_pad[2:, 2:]
    top_mid = a_pad[2:, 1:-1]
    top_left = a_pad[2:, :-2]
    mid_right = a_pad[1:-1, 2:]
    mid_left = a_pad[1:-1, :-2]
    bot_right = a_pad[:-2, 2:]
    bot_mid = a_pad[:-2, 1:-1]
    bot_left = a_pad[:-2, :-2]

    gx = ((top_right + 2.0 * mid_right + bot_right) -
          (top_left + 2.0 * mid_left + bot_left)) / (8.0 * max(dx, 1e-14))

    gy = ((top_right + 2.0 * top_mid + top_left) -
          (bot_right + 2.0 * bot_mid + bot_left)) / (8.0 * max(dy, 1e-14))

    return gx, gy


def youngs_gradient_3d(
    alpha: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute volume fraction gradient using Youngs' 3D 27-point finite difference stencil.

    Reference:
    Youngs, D. L. (1984), "An Interface Tracking Method for a 3D Eulerian Hydrodynamics Code".

    Args:
        alpha: (nz, ny, nx) 3D array of volume fractions in [0, 1].
        dx, dy, dz: grid cell dimensions.

    Returns:
        gx, gy, gz: (nz, ny, nx) arrays of gradient components.
    """
    a_pad = np.pad(alpha, ((1, 1), (1, 1), (1, 1)), mode="edge")
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)

    # In x: difference between right plane (i+1) and left plane (i-1) weighted across y and z
    diff_x = (a_pad[1:-1, 1:-1, 2:] - a_pad[1:-1, 1:-1, :-2]) * 4.0
    for dz_idx, wz in [(-1, 1.0), (1, 1.0)]:
        z_slice = 1 + dz_idx
        diff_x += (a_pad[z_slice:z_slice + alpha.shape[0], 1:-1, 2:] -
                   a_pad[z_slice:z_slice + alpha.shape[0], 1:-1, :-2]) * (2.0 * wz)
    for dy_idx, wy in [(-1, 1.0), (1, 1.0)]:
        y_slice = 1 + dy_idx
        diff_x += (a_pad[1:-1, y_slice:y_slice + alpha.shape[1], 2:] -
                   a_pad[1:-1, y_slice:y_slice + alpha.shape[1], :-2]) * (2.0 * wy)
    for dz_idx, wz in [(-1, 1.0), (1, 1.0)]:
        for dy_idx, wy in [(-1, 1.0), (1, 1.0)]:
            z_slice = 1 + dz_idx
            y_slice = 1 + dy_idx
            diff_x += (a_pad[z_slice:z_slice + alpha.shape[0], y_slice:y_slice + alpha.shape[1], 2:] -
                       a_pad[z_slice:z_slice + alpha.shape[0], y_slice:y_slice + alpha.shape[1], :-2]) * (wz * wy)

    gx = diff_x / (32.0 * max(dx, 1e-14))

    # In y:
    diff_y = (a_pad[1:-1, 2:, 1:-1] - a_pad[1:-1, :-2, 1:-1]) * 4.0
    for dz_idx, wz in [(-1, 1.0), (1, 1.0)]:
        z_slice = 1 + dz_idx
        diff_y += (a_pad[z_slice:z_slice + alpha.shape[0], 2:, 1:-1] -
                   a_pad[z_slice:z_slice + alpha.shape[0], :-2, 1:-1]) * (2.0 * wz)
    for dx_idx, wx in [(-1, 1.0), (1, 1.0)]:
        x_slice = 1 + dx_idx
        diff_y += (a_pad[1:-1, 2:, x_slice:x_slice + alpha.shape[2]] -
                   a_pad[1:-1, :-2, x_slice:x_slice + alpha.shape[2]]) * (2.0 * wx)
    for dz_idx, wz in [(-1, 1.0), (1, 1.0)]:
        for dx_idx, wx in [(-1, 1.0), (1, 1.0)]:
            z_slice = 1 + dz_idx
            x_slice = 1 + dx_idx
            diff_y += (a_pad[z_slice:z_slice + alpha.shape[0], 2:, x_slice:x_slice + alpha.shape[2]] -
                       a_pad[z_slice:z_slice + alpha.shape[0], :-2, x_slice:x_slice + alpha.shape[2]]) * (wz * wx)

    gy = diff_y / (32.0 * max(dy, 1e-14))

    # In z:
    diff_z = (a_pad[2:, 1:-1, 1:-1] - a_pad[:-2, 1:-1, 1:-1]) * 4.0
    for dy_idx, wy in [(-1, 1.0), (1, 1.0)]:
        y_slice = 1 + dy_idx
        diff_z += (a_pad[2:, y_slice:y_slice + alpha.shape[1], 1:-1] -
                   a_pad[:-2, y_slice:y_slice + alpha.shape[1], 1:-1]) * (2.0 * wy)
    for dx_idx, wx in [(-1, 1.0), (1, 1.0)]:
        x_slice = 1 + dx_idx
        diff_z += (a_pad[2:, 1:-1, x_slice:x_slice + alpha.shape[2]] -
                   a_pad[:-2, 1:-1, x_slice:x_slice + alpha.shape[2]]) * (2.0 * wx)
    for dy_idx, wy in [(-1, 1.0), (1, 1.0)]:
        for dx_idx, wx in [(-1, 1.0), (1, 1.0)]:
            y_slice = 1 + dy_idx
            x_slice = 1 + dx_idx
            diff_z += (a_pad[2:, y_slice:y_slice + alpha.shape[1], x_slice:x_slice + alpha.shape[2]] -
                       a_pad[:-2, y_slice:y_slice + alpha.shape[1], x_slice:x_slice + alpha.shape[2]]) * (wy * wx)

    gz = diff_z / (32.0 * max(dz, 1e-14))

    return gx, gy, gz


def youngs_interface_normal(gradient: np.ndarray) -> np.ndarray:
    """Compute unit outward interface normal vector pointing outward from fluid 1.

    In Youngs' formulation:
      n = -grad(alpha) / |grad(alpha)|

    Args:
        gradient: (..., dim) array of volume fraction gradients.

    Returns:
        normal: (..., dim) unit normal vector (zeros if gradient norm is zero).
    """
    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    normal = np.zeros_like(gradient)
    mask = norm[..., 0] > 1e-15
    normal[mask] = -gradient[mask] / norm[mask]
    return normal


def youngs_plane_constant_2d(normal: np.ndarray, alpha: float) -> float:
    """Compute plane constant c for Youngs line n_x * x + n_y * y = c in [0, 1]^2.

    The line truncates a fraction alpha of the unit square [0, 1]^2.
    Formula:
    Let n1 = |n_x|, n2 = |n_y|. Rotate/reflect coordinates to quadrant n1, n2 >= 0.
    Normalized such that n1 + n2 > 0.
    For c in [0, n1 + n2]:
      - If c < min(n1, n2): Area = c^2 / (2 * n1 * n2)
      - If min(n1, n2) <= c <= max(n1, n2): Area = (c - 0.5 * min(n1, n2)) / max(n1, n2)
      - If c > max(n1, n2): Area = 1.0 - (n1 + n2 - c)^2 / (2 * n1 * n2)

    Args:
        normal: (2,) unit normal vector.
        alpha: volume fraction in [0, 1].

    Returns:
        c: scalar line constant in original coordinates.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return -1e10
    if alpha >= 1.0:
        return 1e10

    nx, ny = float(normal[0]), float(normal[1])
    # Absolute values for quadrant mapping
    n1, n2 = abs(nx), abs(ny)
    if n1 + n2 < 1e-14:
        return 0.5

    # Target area in standard quadrant
    # If nx < 0, coordinate transformation x' = 1 - x adds offset |nx|
    # Target area A in [0, 1]^2 for n1 * x' + n2 * y' <= c'
    m1, m2 = min(n1, n2), max(n1, n2)
    m1_m2_2 = 2.0 * m1 * m2

    if m1 < 1e-14:
        # 1D line perpendicular to an axis
        c_prime = alpha * m2
    else:
        crit1 = 0.5 * m1 / m2  # Area when c' = m1
        crit2 = 1.0 - crit1    # Area when c' = m2
        if alpha < crit1:
            c_prime = math.sqrt(alpha * m1_m2_2)
        elif alpha <= crit2:
            c_prime = alpha * m2 + 0.5 * m1
        else:
            c_prime = m1 + m2 - math.sqrt((1.0 - alpha) * m1_m2_2)

    # Shift back to original coordinates:
    # n_x * x + n_y * y = n1 * x' + n2 * y' - offset
    # where x' = x if nx >= 0 else 1 - x
    c = c_prime
    if nx < 0.0:
        c -= n1
    if ny < 0.0:
        c -= n2
    return c


# =============================================================================
# Unstructured Least-Squares Gradient Reconstruction
# =============================================================================

def least_squares_gradient_2d(
    alpha: np.ndarray,
    elem_centers: np.ndarray,
    neighbor_elem: np.ndarray,
    face_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute 2D least-squares gradient of volume fraction for quad elements.

    Ported from $OR_SRC/engine/source/ale/alemuscl/gradient_reconstruction2.F
    lines 88-143:
      For each element II:
        mat(2, 2) = sum_{k=1..4} (x_L - x_K) (x_L - x_K)^T
        rhs(2)    = sum_{k=1..4} (val_K - val_L) * (x_L - x_K)
        sol       = mat^{-1} * rhs
        grad      = -sol

    Args:
        alpha: (n_elem,) volume fraction in each element.
        elem_centers: (n_elem, 2) centroid coordinates (y, z) or (x, y).
        neighbor_elem: (n_elem, 4) neighbor element index (-1 for boundary).
        face_centers: (n_elem, 4, 2) face midpoint coordinates for boundary treatment.

    Returns:
        grad: (n_elem, 2) gradient of volume fraction.
    """
    n_elem = len(alpha)
    grad = np.zeros((n_elem, 2), dtype=np.float64)

    for i in range(n_elem):
        val_k = alpha[i]
        xk = elem_centers[i]

        mat = np.zeros((2, 2), dtype=np.float64)
        rhs = np.zeros(2, dtype=np.float64)

        for k in range(4):
            vois_id = neighbor_elem[i, k]
            if vois_id >= 0:
                val_l = alpha[vois_id]
                xl = elem_centers[vois_id]
            else:
                # Boundary face: mirror centroid across face center
                if face_centers is not None:
                    xf = face_centers[i, k]
                    xl = 2.0 * xf - xk
                else:
                    xl = xk
                val_l = val_k

            dx = xl - xk
            rhs += (val_k - val_l) * dx
            mat += np.outer(dx, dx)

        det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        if abs(det) > 1e-15:
            inv_det = 1.0 / det
            sol0 = inv_det * (mat[1, 1] * rhs[0] - mat[0, 1] * rhs[1])
            sol1 = inv_det * (-mat[1, 0] * rhs[0] + mat[0, 0] * rhs[1])
            grad[i, 0] = -sol0
            grad[i, 1] = -sol1

    return grad


def least_squares_gradient_3d(
    alpha: np.ndarray,
    elem_centers: np.ndarray,
    neighbor_elem: np.ndarray,
    face_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute 3D least-squares gradient of volume fraction for hex elements.

    Ported from $OR_SRC/engine/source/ale/alemuscl/gradient_reconstruction.F90
    lines 134-265.

    Args:
        alpha: (n_elem,) volume fraction.
        elem_centers: (n_elem, 3) element centroids.
        neighbor_elem: (n_elem, 6) neighbor element indices (-1 if boundary).
        face_centers: (n_elem, 6, 3) optional face centers for boundary mirroring.

    Returns:
        grad: (n_elem, 3) gradient vector.
    """
    n_elem = len(alpha)
    grad = np.zeros((n_elem, 3), dtype=np.float64)

    for i in range(n_elem):
        val_k = alpha[i]
        xk = elem_centers[i]

        mat = np.zeros((3, 3), dtype=np.float64)
        rhs = np.zeros(3, dtype=np.float64)

        for k in range(6):
            vois_id = neighbor_elem[i, k]
            if vois_id >= 0:
                val_l = alpha[vois_id]
                xl = elem_centers[vois_id]
            else:
                if face_centers is not None:
                    xf = face_centers[i, k]
                    xl = 2.0 * xf - xk
                else:
                    xl = xk
                val_l = val_k

            dx = xl - xk
            rhs += (val_k - val_l) * dx
            mat += np.outer(dx, dx)

        det = np.linalg.det(mat)
        if abs(det) > 1e-15:
            sol = np.linalg.solve(mat, rhs)
            grad[i] = -sol

    return grad


def gradient_limiter_barth_jespersen_2d(
    grad: np.ndarray,
    alpha: np.ndarray,
    elem_centers: np.ndarray,
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    neighbor_elem: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Barth-Jespersen slope limiter for 2D quad element gradients.

    Ported from $OR_SRC/engine/source/ale/alemuscl/gradient_limitation2.F
    lines 71-115.

    Ensures the linearly extrapolated value at all 4 vertices lies within the
    range of neighbor element centroids:
      phi_min = min_{neighbors}(alpha_j)
      phi_max = max_{neighbors}(alpha_j)
      alpha_node = alpha_i + grad_i . (x_node - x_elem)
      If alpha_node > alpha_i:
        r = min(1.0, beta * (phi_max - alpha_i) / (alpha_node - alpha_i))
      Else:
        r = min(1.0, beta * (phi_min - alpha_i) / (alpha_node - alpha_i))
      reduc = min_{nodes}(r)
      grad_limited = reduc * grad

    Args:
        grad: (n_elem, 2) unconstrained gradients.
        alpha: (n_elem,) element volume fractions.
        elem_centers: (n_elem, 2) centroid coordinates.
        node_coords: (n_nodes, 2) nodal coordinates.
        connectivity: (n_elem, 4) element vertex indices.
        neighbor_elem: (n_elem, 4) neighbor element index.
        beta: limiter parameter (default: 1.0).

    Returns:
        grad_limited: (n_elem, 2) limited gradients.
    """
    n_elem = len(alpha)
    n_nodes = len(node_coords)
    grad_limited = np.copy(grad)

    # Ported from ale51_gradient_reconstruction2.F lines 191-207:
    # NODE_MAX_VALUE and NODE_MIN_VALUE for each node from connected elements
    node_min = np.full(n_nodes, np.inf, dtype=np.float64)
    node_max = np.full(n_nodes, -np.inf, dtype=np.float64)

    for e in range(n_elem):
        for j in range(connectivity.shape[1]):
            nid = connectivity[e, j]
            if nid < n_nodes:
                node_min[nid] = min(node_min[nid], alpha[e])
                node_max[nid] = max(node_max[nid], alpha[e])

    # If any neighbor is provided outside the element list, incorporate into node bounds
    if neighbor_elem is not None:
        for e in range(n_elem):
            for k in range(connectivity.shape[1]):
                nbr = neighbor_elem[e, k]
                if 0 <= nbr < n_elem:
                    for j in range(connectivity.shape[1]):
                        nid = connectivity[e, j]
                        node_min[nid] = min(node_min[nid], alpha[nbr])
                        node_max[nid] = max(node_max[nid], alpha[nbr])

    # Ported from gradient_limitation2.F lines 71-114
    for i in range(n_elem):
        if np.linalg.norm(grad[i]) < 1e-14:
            continue

        val_elem = alpha[i]
        xk = elem_centers[i]
        reduc = 1.0

        for j in range(connectivity.shape[1]):
            node_id = connectivity[i, j]
            xn = node_coords[node_id]
            dx = xn - xk
            val_node = val_elem + np.dot(grad[i], dx)
            diff = val_node - val_elem

            if diff > 1e-14:
                n_max = node_max[node_id]
                r = min(beta * (n_max - val_elem) / diff, 1.0)
            elif diff < -1e-14:
                n_min = node_min[node_id]
                r = min(beta * (n_min - val_elem) / diff, 1.0)
            else:
                r = 1.0
            reduc = min(reduc, max(0.0, r))

        grad_limited[i] *= reduc

    return grad_limited


# =============================================================================
# ALE Law 51 Upwind and Anti-diffusion Volume Fluxes
# =============================================================================

def ale51_upwind_flux(
    flux_total: np.ndarray,
    upwind_param: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute donor-acceptor upwinded face volume fluxes and momentum flux helpers.

    Ported from $OR_SRC/engine/source/ale/ale51/ale51_upwind2.F
    lines 162-176 and ale51_upwind3.F:
      FLUX(II, K) = FLUX_1(I) - UPWL * |FLUX_1(I)|
      QMV(II, 5)  = FLUX_1(I) + UPWL * |FLUX_1(I)|
      FLU1        = sum_{K} QMV(II, 4+K)

    Args:
        flux_total: (n_elem, n_faces) outgoing volume fluxes through faces.
        upwind_param: upwind weighting coefficient eta (0 = centered, 1 = full upwind).

    Returns:
        flux_upwind: (n_elem, n_faces) upwind-adjusted fluxes.
        qmv: (n_elem, n_faces) complementary donor-acceptor momentum fluxes.
        flu1: (n_elem,) sum of QMV fluxes per element.
    """
    eta = float(np.clip(upwind_param, 0.0, 1.0))
    abs_flux = np.abs(flux_total)
    flux_upwind = flux_total - eta * abs_flux
    qmv = flux_total + eta * abs_flux
    flu1 = np.sum(qmv, axis=-1)
    return flux_upwind, qmv, flu1


def ale51_antidiffusion(
    flux_saved: np.ndarray,
    alpha_mat: np.ndarray,
    elem_volumes: np.ndarray,
    dt: float,
    upwind_sm: float = 0.0,
    neighbor_elem: Optional[np.ndarray] = None,
    neighbor_face: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute species volume fluxes with anti-diffusive limiter.

    Ported from $OR_SRC/engine/source/ale/ale51/ale51_antidiff2.F
    lines 90-194 and ale51_antidiff3.F lines 148-260:
      VOL0 = VOL * (1 / DT1)
      For each species itrimat in 1..trimat:
        AV0  = alpha(elem, itrimat) * VOL0
        UAV0 = VOL0 - AV0
        For each outgoing face (flux_save > 0):
          FF(K) = alpha(donor) * flux_save(K)
          ALPHI = sum FF(K)
          PHI0  = sum flux_save(K)
        Limiting:
          If ALPHI > AV0 > 0:
            AAA = AV0 / ALPHI
            FF(K) = FF(K) * AAA
          Else if (PHI0 - ALPHI) > UAV0 > 0:
            AAA = UAV0 / (PHI0 - ALPHI)
            FF(K) = flux_save(K) + (FF(K) - flux_save(K)) * AAA
        Upwind blending:
          FF(K) = 0.5 * [ FF(K)*(1 - UPWSM) + alpha(elem)*flux_save(K)*(1 + UPWSM) ]
          FLUX(elem, K, itrimat) = FF(K)
          FLUX(neighbor, K_opp, itrimat) = -FF(K)

    Args:
        flux_saved: (n_elem, n_faces) geometric face volume fluxes.
        alpha_mat: (n_elem, n_phases) volume fractions for each species.
        elem_volumes: (n_elem,) element volumes.
        dt: time step.
        upwind_sm: anti-diffusion upwind blending factor (ALE%UPWIND%UPWSM).
        neighbor_elem: (n_elem, n_faces) neighbor connectivity.
        neighbor_face: (n_elem, n_faces) corresponding face indices on neighbor.

    Returns:
        flux_mat: (n_elem, n_faces, n_phases) species volumetric fluxes.
    """
    n_elem, n_faces = flux_saved.shape
    n_phases = alpha_mat.shape[1]
    flux_mat = np.zeros((n_elem, n_faces, n_phases), dtype=np.float64)

    udt = 1.0 / dt if dt > 0.0 else 0.0

    for i in range(n_elem):
        vol0 = elem_volumes[i] * udt
        for m in range(n_phases):
            av0 = alpha_mat[i, m] * vol0
            uav0 = vol0 - av0

            alphi = 0.0
            phi0 = 0.0
            ff = np.zeros(n_faces, dtype=np.float64)

            # Outgoing faces (flux_saved > 0)
            for k in range(n_faces):
                f_k = flux_saved[i, k]
                if f_k > 0.0:
                    # In standard donor-cell, material is taken from current element
                    ff[k] = alpha_mat[i, m] * f_k
                    alphi += ff[k]
                    phi0 += f_k

            # Anti-diffusive limiter
            ualphi = phi0 - alphi
            if alphi > av0 and av0 > 0.0:
                aaa = av0 / max(alphi, 1e-30)
                for k in range(n_faces):
                    if flux_saved[i, k] > 0.0:
                        ff[k] *= aaa
            elif ualphi > uav0 and uav0 > 0.0:
                aaa = uav0 / max(ualphi, 1e-30)
                for k in range(n_faces):
                    if flux_saved[i, k] > 0.0:
                        ff[k] = flux_saved[i, k] + (ff[k] - flux_saved[i, k]) * aaa

            # Upwind blending and assignment
            for k in range(n_faces):
                if flux_saved[i, k] > 0.0:
                    val = 0.5 * (ff[k] * (1.0 - upwind_sm) +
                                 alpha_mat[i, m] * flux_saved[i, k] * (1.0 + upwind_sm))
                    flux_mat[i, k, m] = val

    # Enforce anti-symmetry on neighbor faces if neighbor table is provided
    if neighbor_elem is not None and neighbor_face is not None:
        for i in range(n_elem):
            for k in range(n_faces):
                if flux_saved[i, k] > 0.0:
                    nbr = neighbor_elem[i, k]
                    nbr_f = neighbor_face[i, k]
                    if nbr >= 0 and nbr_f >= 0:
                        for m in range(n_phases):
                            flux_mat[nbr, nbr_f, m] = -flux_mat[i, k, m]

    return flux_mat


# =============================================================================
# Law 51 Multi-Material Pressure Relaxation and Mixture Properties
# =============================================================================

@dataclass
class PhaseState:
    """State of an individual material phase in a multi-material element."""
    volume: float               # Phase volume V_k
    mass: float                 # Phase mass M_k
    energy: float               # Phase internal energy E_{int, k}
    density: float              # Phase density rho_k = M_k / V_k
    pressure: float             # Phase pressure P_k
    sound_speed: float          # Phase acoustic wave speed c_k
    shear_modulus: float = 0.0  # Phase shear modulus G_k
    viscosity: float = 0.0      # Phase dynamic viscosity mu_k
    deviatoric_stress: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))


def law51_pressure_relaxation(
    volumes: np.ndarray,
    masses: np.ndarray,
    energies: np.ndarray,
    densities: np.ndarray,
    pressures: np.ndarray,
    sound_speeds: np.ndarray,
    elem_volume: float,
    pext: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-4,
    eos_evaluator: Optional[Callable[[int, float, float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Multi-material pressure relaxation algorithm for Law 51 elements.

    Ported faithfully from $OR_SRC/engine/source/materials/mat/mat051/sigeps51.F90
    lines 1360-1571.

    Equilibrates the pressures of all active phases within an element to an
    equilibrium relaxation pressure P* such that the sum of submaterial volume
    variations is zero (volume conservation) while booking thermodynamically
    consistent internal energy work:
      COEF_k = V_k / (rho_k * c_k^2)
      P* = sum(COEF_k * P_k) / sum(COEF_k)
      Delta V_k = COEF_k * (P_k - P*)
      Step limiter: if Delta V_k < 0: step = min(step, -0.5 * V_k / Delta V_k)
      Delta V_k = step * Delta V_k
      Delta E_{int, k} = -(P* + Pext) * Delta V_k   (energy accounting booked!)
      V_k = V_k + Delta V_k
      Re-evaluate phase EOS until sum(|Delta V_k| / V_k) < tol.

    Args:
        volumes: (n_phases,) current phase volumes.
        masses: (n_phases,) current phase masses.
        energies: (n_phases,) current phase internal energies.
        densities: (n_phases,) current phase densities.
        pressures: (n_phases,) initial phase pressures.
        sound_speeds: (n_phases,) current acoustic sound speeds.
        elem_volume: total element volume.
        pext: external / reference pressure.
        max_iter: maximum relaxation iterations.
        tol: relative convergence tolerance.
        eos_evaluator: optional callback fn(phase_idx, rho, e_spec, v) -> (P, sound_speed).
                       If None, an adiabatic acoustic EOS is used.

    Returns:
        v_relaxed: (n_phases,) equilibrated phase volumes.
        e_relaxed: (n_phases,) updated phase internal energies.
        p_relaxed: (n_phases,) updated phase pressures.
        p_eq: equilibrium mixture pressure.
        converged: boolean flag indicating convergence.
    """
    n_phases = len(volumes)
    v = np.copy(volumes)
    e = np.copy(energies)
    p = np.copy(pressures)
    rho = np.copy(densities)
    ssp = np.copy(sound_speeds)

    p_eq = 0.0
    converged = False

    for iteration in range(1, max_iter + 1):
        # 1. Compute phase compliance coefficients: COEF_k = V_k / (rho_k * c_k^2)
        coef = np.zeros(n_phases, dtype=np.float64)
        for k in range(n_phases):
            if v[k] > 0.0 and ssp[k] > 0.0 and rho[k] > 0.0:
                bulk = rho[k] * ssp[k] * ssp[k]
                coef[k] = v[k] / bulk

        sum_coef = np.sum(coef)
        if sum_coef <= 0.0:
            p_eq = np.sum(v * p) / max(elem_volume, 1e-30)
            converged = True
            break

        # 2. Equilibrium relaxation pressure
        p_eq = np.sum(coef * p) / sum_coef

        # 3. Volume increments: Delta V_k = COEF_k * (P_k - P*)
        dv = coef * (p - p_eq)

        # 4. Step limiter: prevent phase volume from decreasing by more than 50%
        step = 1.0
        for k in range(n_phases):
            if dv[k] < 0.0 and v[k] > 0.0:
                step = min(step, -0.5 * v[k] / dv[k])

        dv *= step

        # 5. Energy accounting: book thermodynamically consistent work
        # dE_{int, k} = -(P* + Pext) * dV_k
        work = -(p_eq + pext) * dv
        for k in range(n_phases):
            if v[k] > 0.0:
                e[k] += work[k]

        # 6. Update volumes and normalize to exact element volume
        v += dv
        total_v = np.sum(v)
        if total_v > 0.0:
            scale = elem_volume / total_v
            v *= scale

        # 7. Update densities and re-evaluate EOS
        for k in range(n_phases):
            if v[k] > 0.0:
                rho[k] = masses[k] / v[k]
                e_spec = e[k] / max(masses[k], 1e-30)
                if eos_evaluator is not None:
                    p[k], ssp[k] = eos_evaluator(k, rho[k], e_spec, v[k])
                else:
                    # Acoustic adiabatic EOS: P = P_eq - bulk * (dV / V)
                    bulk = rho[k] * ssp[k] * ssp[k]
                    p[k] = p_eq

        # 8. Check convergence: sum(|Delta V_k| / V_k) < tol
        err = 0.0
        for k in range(n_phases):
            if v[k] > 0.0:
                err += abs(dv[k]) / v[k]

        if err < tol:
            converged = True
            break

    return v, e, p, float(p_eq), converged


def law51_mixture_properties(
    volumes: np.ndarray,
    masses: np.ndarray,
    pressures: np.ndarray,
    sound_speeds: np.ndarray,
    deviatoric_stresses: Optional[np.ndarray] = None,
    viscosities: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute homogenized mixture properties for Law 51 multi-material element.

    Ported from $OR_SRC/engine/source/materials/mat/mat051/sigeps51.F90
    lines 1570-1610:
      Total volume V = sum V_k,  alpha_k = V_k / V
      Total mass   M = sum M_k,  rho_mix = M / V = sum alpha_k * rho_k
      Pressure     P = sum alpha_k * P_k
      Sound speed  c_mix^2 = sum (M_k / M) * c_k^2
      Viscosity    mu_mix  = sum (M_k / M) * mu_k
      Dev stress   s_mix   = sum alpha_k * s_k
      Cauchy stress sigma  = -P * I + s_mix

    Args:
        volumes: (n_phases,) phase volumes.
        masses: (n_phases,) phase masses.
        pressures: (n_phases,) phase pressures.
        sound_speeds: (n_phases,) phase sound speeds.
        deviatoric_stresses: (n_phases, 6) optional 6-component deviatoric stresses.
        viscosities: (n_phases,) optional dynamic viscosities.

    Returns:
        dict of mixture properties:
          'volume': total volume,
          'mass': total mass,
          'density': mixture density,
          'volume_fractions': array of alpha_k,
          'pressure': mixture pressure,
          'sound_speed': mixture sound speed,
          'viscosity': mixture dynamic viscosity,
          'deviatoric_stress': (6,) mixture deviatoric stress,
          'cauchy_stress': (6,) mixture total Cauchy stress [sxx, syy, szz, sxy, syz, szx].
    """
    total_vol = float(np.sum(volumes))
    total_mass = float(np.sum(masses))

    if total_vol <= 0.0:
        return {
            "volume": 0.0,
            "mass": 0.0,
            "density": 0.0,
            "volume_fractions": np.zeros_like(volumes),
            "pressure": 0.0,
            "sound_speed": 0.0,
            "viscosity": 0.0,
            "deviatoric_stress": np.zeros(6, dtype=np.float64),
            "cauchy_stress": np.zeros(6, dtype=np.float64),
        }

    alpha = volumes / total_vol
    rho_mix = total_mass / total_vol

    # Volume-weighted pressure
    p_mix = float(np.sum(alpha * pressures))

    # Mass-weighted sound speed squared
    if total_mass > 0.0:
        mass_frac = masses / total_mass
        c2_mix = float(np.sum(mass_frac * (sound_speeds ** 2)))
        c_mix = math.sqrt(max(c2_mix, 0.0))
        mu_mix = float(np.sum(mass_frac * viscosities)) if viscosities is not None else 0.0
    else:
        c_mix = 0.0
        mu_mix = 0.0

    # Volume-weighted deviatoric stress
    dev_mix = np.zeros(6, dtype=np.float64)
    if deviatoric_stresses is not None:
        for k in range(len(volumes)):
            dev_mix += alpha[k] * deviatoric_stresses[k]

    # Total Cauchy stress: sigma = -P * I + s
    # Vector form: [xx, yy, zz, xy, yz, zx]
    cauchy = np.copy(dev_mix)
    cauchy[0] -= p_mix
    cauchy[1] -= p_mix
    cauchy[2] -= p_mix

    return {
        "volume": total_vol,
        "mass": total_mass,
        "density": rho_mix,
        "volume_fractions": alpha,
        "pressure": p_mix,
        "sound_speed": c_mix,
        "viscosity": mu_mix,
        "deviatoric_stress": dev_mix,
        "cauchy_stress": cauchy,
    }


# =============================================================================
# Full Law 51 Multi-Material Engine Class
# =============================================================================

class Law51MultiMaterial:
    """Multi-material ALE Law 51 Manager with Youngs Interface Reconstruction.

    Ported from OpenRadioss Fortran sources:
    - engine/source/ale/ale51/afluxt.F
    - engine/source/ale/ale51/ale51_upwind2.F & ale51_upwind3.F
    - engine/source/ale/ale51/ale51_antidiff2.F & ale51_antidiff3.F
    - engine/source/materials/mat/mat051/sigeps51.F90

    Manages multi-material elements containing up to N phases, performing:
    1. Interface normal reconstruction via Youngs method
    2. Anti-diffusive species face volume flux computation
    3. Convective volume and mass advection
    4. Multi-phase pressure relaxation
    5. Homogenized mixture stress and wave speed evaluation
    """

    def __init__(
        self,
        n_elem: int,
        n_phases: int,
        elem_volumes: np.ndarray,
        initial_volume_fractions: np.ndarray,
        phase_densities: np.ndarray,
        phase_sound_speeds: np.ndarray,
        phase_bulk_moduli: Optional[np.ndarray] = None,
        phase_viscosities: Optional[np.ndarray] = None,
        upwind_param: float = 0.0,
        upwind_sm: float = 0.0,
        pext: float = 0.0,
    ) -> None:
        self.n_elem = n_elem
        self.n_phases = n_phases
        self.elem_volumes = np.copy(elem_volumes)
        self.upwind_param = upwind_param
        self.upwind_sm = upwind_sm
        self.pext = pext

        # (n_elem, n_phases) arrays
        self.alpha = np.copy(initial_volume_fractions)
        # Normalize initial volume fractions
        row_sums = np.sum(self.alpha, axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        self.alpha /= row_sums

        # Phase volumes: V_{e, k} = alpha_{e, k} * V_e
        self.phase_volumes = self.alpha * self.elem_volumes[:, np.newaxis]

        # Phase reference properties
        self.rho0 = np.array(phase_densities, dtype=np.float64)
        self.ssp0 = np.array(phase_sound_speeds, dtype=np.float64)
        if phase_bulk_moduli is not None:
            self.bulk0 = np.array(phase_bulk_moduli, dtype=np.float64)
        else:
            self.bulk0 = self.rho0 * (self.ssp0 ** 2)

        self.visc0 = (np.array(phase_viscosities, dtype=np.float64)
                      if phase_viscosities is not None
                      else np.zeros(n_phases, dtype=np.float64))

        # Phase masses: M_{e, k} = V_{e, k} * rho0_k
        self.phase_masses = self.phase_volumes * self.rho0[np.newaxis, :]
        self.phase_energies = np.zeros((n_elem, n_phases), dtype=np.float64)
        self.phase_pressures = np.zeros((n_elem, n_phases), dtype=np.float64)
        self.phase_densities = np.tile(self.rho0, (n_elem, 1))
        self.phase_sound_speeds = np.tile(self.ssp0, (n_elem, 1))
        self.phase_deviatoric_stresses = np.zeros((n_elem, n_phases, 6), dtype=np.float64)

        # Mixture arrays
        self.mixture_pressure = np.zeros(n_elem, dtype=np.float64)
        self.mixture_density = np.zeros(n_elem, dtype=np.float64)
        self.mixture_sound_speed = np.zeros(n_elem, dtype=np.float64)
        self.mixture_cauchy_stress = np.zeros((n_elem, 6), dtype=np.float64)

        # Total energy accounting ledger (work done by pressure relaxation)
        self.energy_work_booked = 0.0

        # Initial mixture state
        self.compute_mixture_state()

    def reconstruct_interfaces_2d(
        self,
        elem_centers: np.ndarray,
        neighbor_elem: np.ndarray,
        node_coords: Optional[np.ndarray] = None,
        connectivity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute interface normals for all species using Youngs least-squares reconstruction.

        Args:
            elem_centers: (n_elem, 2) centroid coordinates.
            neighbor_elem: (n_elem, 4) neighbor connectivity.
            node_coords: optional (n_nodes, 2) vertex coordinates for limiting.
            connectivity: optional (n_elem, 4) element connectivity.

        Returns:
            normals: (n_elem, n_phases, 2) unit normal vectors.
        """
        normals = np.zeros((self.n_elem, self.n_phases, 2), dtype=np.float64)
        for m in range(self.n_phases):
            grad_m = least_squares_gradient_2d(self.alpha[:, m], elem_centers, neighbor_elem)
            if node_coords is not None and connectivity is not None:
                grad_m = gradient_limiter_barth_jespersen_2d(
                    grad_m, self.alpha[:, m], elem_centers, node_coords, connectivity, neighbor_elem
                )
            normals[:, m] = youngs_interface_normal(grad_m)
        return normals

    def compute_species_fluxes(
        self,
        total_face_fluxes: np.ndarray,
        dt: float,
        neighbor_elem: Optional[np.ndarray] = None,
        neighbor_face: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute species volume fluxes through all element faces via anti-diffusion.

        Ported from engine/source/ale/ale51/afluxt.F lines 270-292.

        Args:
            total_face_fluxes: (n_elem, n_faces) geometric face fluxes.
            dt: time step.
            neighbor_elem: (n_elem, n_faces) neighbor indices.
            neighbor_face: (n_elem, n_faces) corresponding face indices.

        Returns:
            species_fluxes: (n_elem, n_faces, n_phases) volume flux for each species.
        """
        return ale51_antidiffusion(
            flux_saved=total_face_fluxes,
            alpha_mat=self.alpha,
            elem_volumes=self.elem_volumes,
            dt=dt,
            upwind_sm=self.upwind_sm,
            neighbor_elem=neighbor_elem,
            neighbor_face=neighbor_face,
        )

    def advect_species(
        self,
        species_fluxes: np.ndarray,
        dt: float,
    ) -> None:
        """Advect phase volumes and masses based on face fluxes.

        Ported from engine/source/ale/ale51/afluxt.F and arezon.F90:
          V_{e, k}^{new} = V_{e, k}^{old} - dt * sum_{faces} FLUX_{e, face, k}
          M_{e, k}^{new} = M_{e, k}^{old} - dt * sum_{faces} (rho_{donor, k} * FLUX_{e, face, k})

        Args:
            species_fluxes: (n_elem, n_faces, n_phases) species volumetric face fluxes.
            dt: time step.
        """
        # Sum outgoing species flux for each element
        net_flux = np.sum(species_fluxes, axis=1)  # (n_elem, n_phases)
        vol_change = dt * net_flux

        # Update volumes
        self.phase_volumes = np.maximum(0.0, self.phase_volumes - vol_change)
        self.elem_volumes = np.sum(self.phase_volumes, axis=1)

        # Update volume fractions
        mask = self.elem_volumes > 1e-30
        self.alpha[mask] = self.phase_volumes[mask] / self.elem_volumes[mask, np.newaxis]

        # Update masses proportionally to volume change and phase density
        # Donor density: rho_{e, k}
        mass_change = np.zeros_like(self.phase_masses)
        for k in range(species_fluxes.shape[1]):
            flux_k = species_fluxes[:, k, :]  # (n_elem, n_phases)
            # If flux_k > 0: current element is donor
            mass_change += dt * np.where(flux_k > 0.0, flux_k * self.phase_densities, 0.0)

        self.phase_masses = np.maximum(0.0, self.phase_masses - mass_change)

        # Update densities
        for m in range(self.n_phases):
            act = self.phase_volumes[:, m] > 1e-30
            self.phase_densities[act, m] = self.phase_masses[act, m] / self.phase_volumes[act, m]
            self.phase_densities[~act, m] = self.rho0[m]

    def equilibrate_pressures(
        self,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        """Perform multi-material iterative pressure relaxation across all elements.

        Ported from engine/source/materials/mat/mat051/sigeps51.F90 lines 1360-1571.
        """
        total_work = 0.0
        for i in range(self.n_elem):
            if self.elem_volumes[i] <= 0.0:
                continue

            v_i = self.phase_volumes[i]
            m_i = self.phase_masses[i]
            e_i = self.phase_energies[i]
            rho_i = self.phase_densities[i]
            p_i = self.phase_pressures[i]
            ssp_i = self.phase_sound_speeds[i]

            e_before = float(np.sum(e_i))

            v_rel, e_rel, p_rel, p_eq, conv = law51_pressure_relaxation(
                volumes=v_i,
                masses=m_i,
                energies=e_i,
                densities=rho_i,
                pressures=p_i,
                sound_speeds=ssp_i,
                elem_volume=self.elem_volumes[i],
                pext=self.pext,
                max_iter=max_iter,
                tol=tol,
            )

            self.phase_volumes[i] = v_rel
            self.phase_energies[i] = e_rel
            self.phase_pressures[i] = p_rel
            self.mixture_pressure[i] = p_eq

            e_after = float(np.sum(e_rel))
            total_work += (e_after - e_before)

        self.energy_work_booked += total_work
        # Update alpha from relaxed volumes
        row_sums = np.sum(self.phase_volumes, axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        self.alpha = self.phase_volumes / row_sums

        self.compute_mixture_state()

    def compute_mixture_state(self) -> None:
        """Compute homogenized mixture stress, density, and acoustic wave speed."""
        for i in range(self.n_elem):
            props = law51_mixture_properties(
                volumes=self.phase_volumes[i],
                masses=self.phase_masses[i],
                pressures=self.phase_pressures[i],
                sound_speeds=self.phase_sound_speeds[i],
                deviatoric_stresses=self.phase_deviatoric_stresses[i],
                viscosities=self.visc0,
            )
            self.mixture_density[i] = props["density"]
            self.mixture_pressure[i] = props["pressure"]
            self.mixture_sound_speed[i] = props["sound_speed"]
            self.mixture_cauchy_stress[i] = props["cauchy_stress"]
