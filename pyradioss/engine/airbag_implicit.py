"""pyradioss.engine.airbag_implicit — Implicit coupled monitored volume airbag solver.

Fortran source references:
- ``engine/source/airbag/monv_imp0.F``:
  - Subroutine IMP_PVGA (lines 2233-2294): Incremental pressure from volume change.
  - Subroutine MONV_KD, MONV_KEDI, MONV_KEDJ (lines 956-1260): Airbag tangent stiffness matrix.
  - Subroutine MV_MATV (lines 1767-1899): Airbag matrix-vector directional derivative.
  - Subroutine MONV_IMP, MONV_M3 (lines 559-670): Coupled implicit monitored volume management.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pyradioss.model.model import Model


# Ported from $OR_SRC/engine/source/airbag/monv_imp0.F: IMP_PVGA (lines 2233-2294)
def compute_implicit_pressure_increment(
    p_old: float,
    e_old: float,
    v_old: float,
    v_new: float,
    v_inc: float = 0.0,
    gamma: float = 1.4,
    p_max: float = 1e30,
    p_ext: float = 0.0,
    de_out: float = 0.0,
    dt: float = 1e-3,
    v_eps: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute incremental pressure change dP for implicit iteration.

    Matches OpenRadioss ``IMP_PVGA``:
    Calculates thermodynamic internal energy E and pressure P for a candidate
    volume V_new, and returns (dP, P, E).

    Args:
        p_old: Pressure at start of time step.
        e_old: Internal energy at start of time step.
        v_old: Volume at start of time step.
        v_new: Candidate new volume.
        v_inc: Incompressible / offset volume VINC.
        gamma: Ratio of specific heats Cp/Cv.
        p_max: Burst pressure threshold.
        p_ext: External ambient pressure.
        de_out: Energy outflow rate through vents/leaks.
        dt: Time step size.
        v_eps: Volume regularizing increment.

    Returns:
        (dP, P_new, E_new): Incremental pressure, new pressure, and new internal energy.
    """
    v_eff_old = max(v_old - v_inc, 1e-12)
    v_curr = v_new + v_eps
    v_eff_new = max(v_curr - v_inc, 1e-12)
    dv = v_curr - v_old

    gamma = max(gamma, 1.001)
    fac = 0.5 * (gamma - 1.0) * dv

    num = (1.0 - fac / v_eff_old) * e_old - de_out * dt
    denom = 1.0 + fac / v_eff_new

    if denom > 1e-6:
        energy = max(num / denom, 0.0)
    else:
        # Fallback to closed-form isentropic law for large volume changes (matching volpvga.F)
        ratio = v_eff_old / v_eff_new
        energy = max(0.0, e_old * (ratio ** (gamma - 1.0)) - de_out * dt)

    pres = (gamma - 1.0) * energy / v_eff_new

    # Burst pressure check (monv_imp0.F line 2284)
    if p_max > 0.0 and pres > p_max:
        pres = p_ext

    dp = pres - p_old
    return float(dp), float(pres), float(energy)


# Ported from $OR_SRC/engine/source/airbag/monv_imp0.F: MONV_KD, MONV_KEDI, MONV_KEDJ (lines 956-1150)
def compute_airbag_tangent_stiffness(
    mv: Any,
    model: Model,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute diagonal tangent stiffness matrix contributions for airbag pressure forces.

    Matches OpenRadioss ``MONV_KD`` / ``MONV_KEDI``:
    In implicit equilibrium, the airbag nodal force F = P(V) * A(x) leads to the
    tangent stiffness Jacobian K = -dF/dx = -(dP/dV) * A * A^T - P * (dA/dx).
    For ideal gas with adiabatic index gamma:
    dP/dV = -gamma * P / V_eff.

    Args:
        mv: Monitored volume object.
        model: Model containing surfaces.
        x: Current nodal coordinates (N, 3).

    Returns:
        (k_diag, nodal_areas):
            - k_diag: (N, 3) diagonal stiffness terms for each node.
            - nodal_areas: (N, 3) tributary normal area vector for each node.
    """
    n_nodes = len(x)
    k_diag = np.zeros((n_nodes, 3), dtype=np.float64)
    nodal_areas = np.zeros((n_nodes, 3), dtype=np.float64)

    surf = model.surfaces.get(getattr(mv, "surf_id", 0)) if hasattr(model, "surfaces") else None
    if surf is None or surf.segments is None or len(surf.segments) == 0:
        return k_diag, nodal_areas

    # Current volume and pressure
    vol = max(getattr(mv, "volume", 1e-9), 1e-9)
    v_inc = getattr(mv, "vinc", 0.0)
    v_eff = max(vol - v_inc, 1e-9)
    pres = getattr(mv, "pressure", getattr(mv, "pext", 0.0))
    gamma = getattr(mv, "gamma", 1.4)
    if gamma <= 1.0:
        gamma = 1.4

    # dP/dV factor: GAMAV2 = gamma * P / V_eff^2
    dp_dv = gamma * pres / v_eff

    # Accumulate facet normal area vectors and compute geometric sensitivities
    for seg in surf.segments:
        n1, n2, n3 = seg[0], seg[1], seg[2]
        is_tri = (len(seg) < 4) or (seg[3] == n3) or (seg[3] < 0)

        x1 = x[n1]; x2 = x[n2]; x3 = x[n3]
        v31 = x3 - x1

        if not is_tri:
            n4 = seg[3]
            x4 = x[n4]
            v42 = x4 - x2
            # Facet normal area vector
            xn = 0.5 * np.cross(v31, v42)
            # Tributary area per node: 1/4
            a_node = xn / 4.0
            nodes = [n1, n2, n3, n4]
        else:
            v21 = x2 - x1
            xn = 0.5 * np.cross(v21, v31)
            # Tributary area per node: 1/3
            a_node = xn / 3.0
            nodes = [n1, n2, n3]

        # Accumulate nodal areas
        for nid in nodes:
            if 0 <= nid < n_nodes:
                nodal_areas[nid] += a_node

    # Nodal diagonal stiffness: K_diag,i = dp_dv * a_node,i^2
    for nid in range(n_nodes):
        a_i = nodal_areas[nid]
        k_diag[nid] = dp_dv * (a_i ** 2)

    return k_diag, nodal_areas


# Ported from $OR_SRC/engine/source/airbag/monv_imp0.F: MV_MATV (lines 1767-1899)
def apply_airbag_matvec(
    mv: Any,
    model: Model,
    x: np.ndarray,
    u: np.ndarray,
    dt: float = 1e-3,
) -> np.ndarray:
    """Evaluate airbag matrix-vector product K_bag * u for implicit iteration.

    Matches OpenRadioss ``MV_MATV``:
    Given displacement perturbation u, updates perturbed coordinates x* = x + u,
    computes perturbed volume V*, evaluates pressure increment dP from ``IMP_PVGA``,
    and applies directional pressure forces:
    F_tan = K_bag * u = sum_facets (dP / n_nodes) * Normal_facet.

    Args:
        mv: Monitored volume object.
        model: Finite element model with surfaces.
        x: Current coordinates (N, 3).
        u: Displacement increment / perturbation (N, 3).
        dt: Time step size.

    Returns:
        (N, 3) tangent force increment array.
    """
    n_nodes = len(x)
    f_tan = np.zeros((n_nodes, 3), dtype=np.float64)

    surf = model.surfaces.get(getattr(mv, "surf_id", 0)) if hasattr(model, "surfaces") else None
    if surf is None or surf.segments is None or len(surf.segments) == 0:
        return f_tan

    x_perturbed = x + u

    # Compute perturbed volume V*
    v_old = getattr(mv, "volume_old", getattr(mv, "volume", 1e-9))
    v_inc = getattr(mv, "vinc", 0.0)
    p_old = getattr(mv, "pressure", 0.0)
    e_old = getattr(mv, "energy", 0.0)
    gamma = getattr(mv, "gamma", 1.4)
    p_max = getattr(mv, "pmax", 1e30)
    p_ext = getattr(mv, "pext", 0.0)
    de_out = getattr(mv, "de_out", 0.0)

    v_perturbed = 0.0
    facet_normals = []

    for seg in surf.segments:
        n1, n2, n3 = seg[0], seg[1], seg[2]
        is_tri = (len(seg) < 4) or (seg[3] == n3) or (seg[3] < 0)

        xp1 = x_perturbed[n1]; xp2 = x_perturbed[n2]; xp3 = x_perturbed[n3]
        v31 = xp3 - xp1

        if not is_tri:
            n4 = seg[3]
            xp4 = x_perturbed[n4]
            v42 = xp4 - xp2
            xn = 0.5 * np.cross(v31, v42)
            vol_facet = np.dot(xp1 + xp2 + xp3 + xp4, xn) / 12.0
        else:
            v21 = xp2 - xp1
            xn = 0.5 * np.cross(v21, v31)
            vol_facet = np.dot(xp1 + xp2 + xp3, xn) / 9.0

        v_perturbed += vol_facet
        facet_normals.append((is_tri, seg, xn))

    # Evaluate incremental pressure dP from IMP_PVGA
    dp, _, _ = compute_implicit_pressure_increment(
        p_old=p_old,
        e_old=e_old,
        v_old=v_old,
        v_new=v_perturbed,
        v_inc=v_inc,
        gamma=gamma,
        p_max=p_max,
        p_ext=p_ext,
        de_out=de_out,
        dt=dt,
    )

    if abs(dp) < 1e-20:
        return f_tan

    # Distribute tangent force increment dP * Area / N_nodes
    for is_tri, seg, xn in facet_normals:
        if is_tri:
            dpi = dp / 3.0
            fx, fy, fz = dpi * xn[0], dpi * xn[1], dpi * xn[2]
            for nid in (seg[0], seg[1], seg[2]):
                if 0 <= nid < n_nodes:
                    f_tan[nid, 0] += fx
                    f_tan[nid, 1] += fy
                    f_tan[nid, 2] += fz
        else:
            dpi = dp / 4.0
            fx, fy, fz = dpi * xn[0], dpi * xn[1], dpi * xn[2]
            for nid in (seg[0], seg[1], seg[2], seg[3]):
                if 0 <= nid < n_nodes:
                    f_tan[nid, 0] += fx
                    f_tan[nid, 1] += fy
                    f_tan[nid, 2] += fz

    return f_tan


# Ported from $OR_SRC/engine/source/airbag/monv_imp0.F: coupled implicit solve
def solve_implicit_airbag_step(
    mv: Any,
    model: Model,
    x: np.ndarray,
    f_structural: np.ndarray,
    k_structural_diag: np.ndarray,
    dt: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, bool, int]:
    """Solve coupled implicit equilibrium step for structural fabric and enclosed airbag.

    Solves the nonlinear equilibrium system:
    R(u) = F_ext - F_structural(x + u) + F_bag(x + u) = 0
    using Newton-Raphson iteration with diagonal preconditioner and exact MV_MATV
    airbag tangent stiffness.

    Args:
        mv: Monitored volume object.
        model: Model containing surfaces and materials.
        x: (N, 3) current nodal coordinates.
        f_structural: (N, 3) structural internal/external force imbalance at start of step.
        k_structural_diag: (N, 3) diagonal structural stiffness matrix.
        dt: Time step size.
        max_iter: Maximum Newton-Raphson iterations.
        tol: Relative force residual tolerance.

    Returns:
        (u, converged, iterations):
            - u: (N, 3) converged nodal displacement increments.
            - converged: bool.
            - iterations: int number of Newton steps.
    """
    n_nodes = len(x)
    u = np.zeros((n_nodes, 3), dtype=np.float64)

    # Airbag diagonal stiffness
    k_bag_diag, _ = compute_airbag_tangent_stiffness(mv, model, x)

    # Initial residual
    r0_norm = float(np.linalg.norm(f_structural))
    if r0_norm < 1e-12:
        return u, True, 0

    converged = False
    n_iters = 0

    for it in range(max_iter):
        n_iters = it + 1
        # Airbag tangent force response K_bag * u
        f_bag_tan = apply_airbag_matvec(mv, model, x, u, dt=dt)

        # Residual R = F_structural - K_structural * u + F_bag_tan
        res = f_structural - k_structural_diag * u + f_bag_tan
        r_norm = float(np.linalg.norm(res))

        if r_norm <= tol * r0_norm or r_norm < 1e-10:
            converged = True
            break

        # Total diagonal tangent stiffness: K_eff = K_structural + K_bag
        k_eff = k_structural_diag + k_bag_diag
        k_eff = np.where(np.abs(k_eff) < 1e-12, 1.0, k_eff)

        # Newton displacement correction
        du = res / k_eff
        u += du

    return u, converged, n_iters
