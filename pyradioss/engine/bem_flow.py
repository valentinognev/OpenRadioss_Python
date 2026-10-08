"""Boundary Element Method (BEM) Incompressible Potential Flow Solver.

Upstream OpenRadioss Fortran References:
----------------------------------------
- Main Incompressible Potential Flow Solver (INCPFLOW):
  ``engine/source/fluid/incpflow.F`` (lines 38-1218)
- Surface Triangle Gradient Kernel (TRGRAD):
  ``engine/source/fluid/incpflow.F`` (lines 1224-1304)
- BEM Collocation & Galerkin System Assembly and Linear Solution (BEMSOLV):
  ``engine/source/fluid/bemsolv.F`` (lines 35-219)
- Boundary Element Double-Layer Integration (INTHTG):
  ``engine/source/fluid/bemsolv.F`` (lines 312-431)
- Boundary Element Single-Layer Integration (INTGTG):
  ``engine/source/fluid/bemsolv.F`` (lines 438-578)
- Analytical Panel Integration (INTANL):
  ``engine/source/fluid/bemsolv.F`` (lines 729-890)
- Fluid-Structure Interaction Driver (FLOW0):
  ``engine/source/fluid/flow0.F`` (lines 36-193)
- BEM Flow Card Reader:
  ``starter/source/loads/bem/hm_read_bem.F`` (lines 280-755)

Physics & Formulation:
----------------------
1. Incompressible Potential Flow:
   Governing Laplace equation for velocity potential Phi in fluid domain Omega:
       nabla^2 Phi = 0
   Velocity field is given by:
       u = nabla Phi

2. Boundary Integral Representation (Green's Second Identity):
   On closed structural boundary Gamma with unit outward normal n (pointing into fluid):
       c(x) Phi(x) + integral_Gamma dG/dn(x, y) Phi(y) dGamma(y)
           = integral_Gamma G(x, y) q(y) dGamma(y) + Phi_inf(x)
   where G(x, y) = 1 / (4 * pi * |x - y|) is the free-space 3D fundamental solution,
   q = dPhi/dn is the normal flux (structural boundary normal velocity v . n),
   and c(x) is the free-term coefficient (1/2 for smooth surface, solid angle / (4*pi) at corners).

3. Discretized BEM Matrix System (bemsolv.F lines 68-180):
   Linear elements in potential Phi, constant elements in normal flux Q:
       H * Phi = G * Q - Phi_inf
   where:
   - H_ij: double-layer panel integrals (INTHTG)
   - G_ij: single-layer panel integrals (INTGTG)
   - Diagonal row-sum correction enforces the rigid-body mode:
       H_ii = - sum_{j != i} H_ij
   - Far-field closure condition replaces the last column of H with -G_inf:
       H_{:, N} = - G_{:, NEL+1}
   The linear system is solved via LU decomposition or standard linear solver.

4. Surface Velocities (incpflow.F lines 522-561):
   For each triangle panel, tangential velocity is obtained from surface gradient of Phi (TRGRAD),
   and normal velocity is given by Q:
       u_elem = grad_Gamma Phi + Q * n_unit
   Nodal velocities U are obtained by area-weighted averaging over adjacent panels:
       U(N_k) += u_elem * (Area / 3) / NodArea(N_k)

5. Internal / Field Point Potential and Velocity (incpflow.F lines 674-752):
   At arbitrary query points x_q:
       Phi(x_q) = sum_panels integral [ G(x_q, y) q(y) - dG/dn(x_q, y) Phi(y) ] dGamma(y) + Phi_inf
       u(x_q)   = sum_panels integral [ grad_x G(x_q, y) q(y) - grad_x dG/dn(x_q, y) Phi(y) ] dGamma(y) + v_inf

6. Unsteady Bernoulli Pressure (incpflow.F lines 1058-1132):
   Pressure is computed from the unsteady Bernoulli equation:
       P = P_stag - rho * (dPhi/dt + a_inf . (x - x_ref)) - 0.5 * rho * |u|^2
   where P_stag is ambient/stagnation pressure, rho is fluid density, and a_inf is far-field acceleration.

7. Structural Forces and Work Ledger (incpflow.F lines 1137-1166):
   Segment pressure force F_e = P_mean * Area * n is distributed equally to the 3 corner nodes:
       F_node(N_k) += (1/3) * P_mean * Area * n_unit
   External work done by the fluid on the structure is accumulated into the energy ledger:
       d WFEXT = sum_nodes (F_node . v_node) * dt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM15, EM20, EM30


# ============================================================================
# Gauss Quadrature Rules on Flat Triangles (bemsolv.F lines 110-148, 337-375)
# ============================================================================

# 1-point rule (order 1, centroid)
_PG_1 = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
_WPG_1 = np.array([1.0], dtype=float)

# 4-point rule (order 2)
_PG_4 = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [0.6, 0.2],
    [0.2, 0.6],
    [0.2, 0.2],
], dtype=float)
_WPG_4 = np.array([-0.5625, 0.520833333333333, 0.520833333333333, 0.520833333333333], dtype=float)

# 7-point rule (order 5)
_PG_7 = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [0.79742699, 0.10128651],
    [0.10128651, 0.79742699],
    [0.10128651, 0.10128651],
    [0.05971587, 0.47014206],
    [0.47014206, 0.05971587],
    [0.47014206, 0.47014206],
], dtype=float)
_WPG_7 = np.array([
    0.22500000,
    0.12593918, 0.12593918, 0.12593918,
    0.13239415, 0.13239415, 0.13239415,
], dtype=float)

# 13-point rule (order 7)
_PG_13 = np.array([
    [0.06513010, 0.06513010],
    [0.86973979, 0.06513010],
    [0.06513010, 0.86973979],
    [0.31286550, 0.04869031],
    [0.63844419, 0.31286550],
    [0.04869031, 0.63844419],
    [0.63844419, 0.04869031],
    [0.31286550, 0.63844419],
    [0.04869031, 0.31286550],
    [0.26034597, 0.26034597],
    [0.47930807, 0.26034597],
    [0.26034597, 0.47930807],
    [1.0 / 3.0, 1.0 / 3.0],
], dtype=float)
_WPG_13 = np.array([
    0.05334724, 0.05334724, 0.05334724,
    0.07711376, 0.07711376, 0.07711376,
    0.07711376, 0.07711376, 0.07711376,
    0.17561526, 0.17561526, 0.17561526,
    -0.14957004,
], dtype=float)


# ============================================================================
# Dataclasses & Parameter Containers
# ============================================================================

@dataclass
class BemFlowParams:
    """Parameters for /BEM/FLOW incompressible potential flow solver.

    Fortran origin: ``starter/source/loads/bem/hm_read_bem.F`` lines 645-720.
    """

    id: int = 1
    title: str = ""
    rho: float = 1000.0  # Fluid mass density [kg/m^3] (RFLOW(5))
    iform: int = 1  # 1 = Collocation BEM, 2 = Galerkin BEM
    ilvout: int = 0  # Print/verbosity level

    # Subcycling time step (RFLOW(3))
    dtsub: float = 0.0

    # Stagnation / Reference Pressure (RFLOW(1), RFLOW(2), IFLOW(23))
    ifpa: int = 0  # Function ID for stagnation pressure vs time
    sfpa: float = 0.0  # Scale factor for stagnation pressure
    scalt_pa: float = 1.0  # Time scale factor
    pa_const: float = 0.0  # Constant stagnation pressure if no function

    # Far-field velocity (RFLOW(7), RFLOW(8), RFLOW(9:11), IFLOW(24))
    ifvini: int = 0  # Function ID for far-field speed vs time
    sfvini: float = 0.0  # Scale factor for far-field speed
    scalt_vi: float = 1.0  # Time scale factor
    dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    v_inf_const: float = 0.0  # Constant far-field speed if no function

    # Internal / Auxiliary node testing
    tole: float = 1.0e-3  # Solid angle tolerance (RFLOW(6))
    itest: int = 1  # 1 = internal points, 2 = external points


# ============================================================================
# Geometric & Integral Helper Functions
# ============================================================================

def compute_triangle_normals_and_areas(
    x: np.ndarray,
    elem: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normal vectors, element areas, and nodal tributary areas.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 240-278.

    Parameters
    ----------
    x : np.ndarray
        Nodal coordinates (N_nodes, 3).
    elem : np.ndarray
        Triangle element node indices (N_elem, 3), 0-based.

    Returns
    -------
    norm_vec : np.ndarray
        Normal vector (N_elem, 3) = (x2-x1) x (x3-x1) (magnitude = 2 * Area).
    elarea : np.ndarray
        Element area (N_elem,) = 0.5 * |norm_vec|.
    nodarea : np.ndarray
        Nodal tributary area (N_nodes,) = sum_elem (Area / 6).
    """
    n_nodes = len(x)
    n_elem = len(elem)

    n1 = elem[:, 0]
    n2 = elem[:, 1]
    n3 = elem[:, 2]

    x1 = x[n1]
    x2 = x[n2]
    x3 = x[n3]

    v12 = x2 - x1
    v13 = x3 - x1

    # Cross product (x2 - x1) x (x3 - x1) (incpflow.F lines 267-272)
    norm_vec = np.cross(v12, v13)
    area2 = np.linalg.norm(norm_vec, axis=1)
    elarea = 0.5 * area2

    nodarea = np.zeros(n_nodes, dtype=float)
    tributary = area2 / 6.0
    np.add.at(nodarea, n1, tributary)
    np.add.at(nodarea, n2, tributary)
    np.add.at(nodarea, n3, tributary)

    return norm_vec, elarea, nodarea


def trgrad(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    phi1: float,
    phi2: float,
    phi3: float,
) -> np.ndarray:
    """Compute surface gradient of potential on a 3D triangle panel.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 1224-1304.

    Parameters
    ----------
    x1, x2, x3 : np.ndarray
        Corner coordinates of the triangle (3,).
    phi1, phi2, phi3 : float
        Scalar potentials at the three vertices.

    Returns
    -------
    grad : np.ndarray
        3D Cartesian gradient vector (3,).
    """
    # Local orthonormal basis on triangle (incpflow.F lines 1245-1275)
    vx1 = x2 - x1
    ex1 = np.linalg.norm(vx1)
    if ex1 > EM20:
        vx1 = vx1 / ex1
    else:
        return np.zeros(3, dtype=float)

    vx2 = x3 - x1
    ex2 = np.dot(vx1, vx2)
    vx2 = vx2 - ex2 * vx1
    ey2 = np.linalg.norm(vx2)
    if ey2 > EM20:
        vx2 = vx2 / ey2
    else:
        return np.zeros(3, dtype=float)

    # Local coordinates: (0, 0), (ex1, 0), (ex2, ey2)
    # Jacobian = ex1 * ey2 (incpflow.F lines 1283-1294)
    jac = ex1 * ey2
    if abs(jac) <= EM20:
        return np.zeros(3, dtype=float)

    jj11 = ey2 / jac  # 1 / ex1
    jj12 = -ex2 / jac
    jj21 = 0.0
    jj22 = ex1 / jac  # 1 / ey2

    # Local 2D gradient
    grlx = (phi2 - phi1) * jj11 + (phi3 - phi1) * jj21
    grly = (phi2 - phi1) * jj12 + (phi3 - phi1) * jj22

    # Global 3D gradient (incpflow.F lines 1299-1301)
    grad = grlx * vx1 + grly * vx2
    return grad


def int_h_tg(
    xs: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x0: np.ndarray,
    d2: float,
    normal_vec: np.ndarray,
    is_vertex: bool,
) -> np.ndarray:
    """Numerical integration of double-layer potential kernel over triangle.

    Ported from $OR_SRC/engine/source/fluid/bemsolv.F lines 312-431.

    Parameters
    ----------
    xs : np.ndarray
        Source / collocation point (3,).
    x1, x2, x3 : np.ndarray
        Corner coordinates of the panel (3,).
    x0 : np.ndarray
        Panel centroid (3,).
    d2 : float
        Squared minimum distance from centroid to corners.
    normal_vec : np.ndarray
        Normal vector = (x2-x1) x (x3-x1) (magnitude = 2 * Area).
    is_vertex : bool
        True if the source point coincides with one of the vertices.

    Returns
    -------
    rval : np.ndarray
        Double-layer matrix contributions to the 3 panel vertices (3,).
    """
    if is_vertex:
        return np.zeros(3, dtype=float)

    r2_source = np.sum((x0 - xs) ** 2)

    # Adaptive Gauss quadrature selection (bemsolv.F lines 383-395)
    if r2_source > 100.0 * d2:
        pg = _PG_1
        wpg = _WPG_1
    elif r2_source > 25.0 * d2:
        pg = _PG_4
        wpg = _WPG_4
    elif r2_source > 4.0 * d2:
        pg = _PG_7
        wpg = _WPG_7
    else:
        pg = _PG_13
        wpg = _WPG_13

    rval = np.zeros(3, dtype=float)
    inv_4pi = 0.25 / math.pi

    for i in range(len(wpg)):
        w = wpg[i]
        eta1 = pg[i, 0]
        eta2 = pg[i, 1]
        val1 = 1.0 - eta1 - eta2
        val2 = eta1
        val3 = eta2

        xg = val1 * x1 + val2 * x2 + val3 * x3
        r_vec = xg - xs
        r2 = np.sum(r_vec ** 2)

        if r2 > EM20:
            r_norm = math.sqrt(r2)
            r3 = r2 * r_norm
            # Kernel: dG/dn = - 1 / (4*pi*R^3) * dot(normal, R_vec)
            # Note: normal_vec is 2 * Area * unit_normal, so 0.5 * dot(normal_vec, r_vec) gives
            # the panel surface element area * unit_normal (bemsolv.F line 419-423)
            valphi = -inv_4pi / r3 * np.dot(normal_vec, r_vec)
            rval[0] += 0.5 * w * val1 * valphi
            rval[1] += 0.5 * w * val2 * valphi
            rval[2] += 0.5 * w * val3 * valphi

    return rval


def int_g_tg(
    xs: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x0: np.ndarray,
    d2: float,
    jac: float,
) -> float:
    """Numerical integration of single-layer potential kernel over triangle.

    Ported from $OR_SRC/engine/source/fluid/bemsolv.F lines 438-578.

    Parameters
    ----------
    xs : np.ndarray
        Source / collocation point (3,).
    x1, x2, x3 : np.ndarray
        Corner coordinates of the panel (3,).
    x0 : np.ndarray
        Panel centroid (3,).
    d2 : float
        Squared minimum distance from centroid to corners.
    jac : float
        Panel Jacobian = 2 * Area = |(x2-x1) x (x3-x1)|.

    Returns
    -------
    rval : float
        Single-layer matrix contribution to the panel flux.
    """
    r2_source = np.sum((x0 - xs) ** 2)

    # Adaptive Gauss quadrature selection (bemsolv.F lines 542-554)
    if r2_source > 100.0 * d2:
        pg = _PG_1
        wpg = _WPG_1
    elif r2_source > 25.0 * d2:
        pg = _PG_4
        wpg = _WPG_4
    elif r2_source > 4.0 * d2:
        pg = _PG_7
        wpg = _WPG_7
    else:
        pg = _PG_13
        wpg = _WPG_13

    rval = 0.0
    inv_4pi = 0.25 / math.pi

    for i in range(len(wpg)):
        w = wpg[i]
        eta1 = pg[i, 0]
        eta2 = pg[i, 1]
        val1 = 1.0 - eta1 - eta2
        val2 = eta1
        val3 = eta2

        xg = val1 * x1 + val2 * x2 + val3 * x3
        r_vec = xg - xs
        r2 = np.sum(r_vec ** 2)

        if r2 > EM20:
            r = math.sqrt(r2)
            valphi = inv_4pi / r
            rval += 0.5 * w * valphi * jac

    return rval


def solid_angle_triangle(
    xq: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
) -> float:
    """Compute solid angle subtended by a triangle at query point xq.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 607-655.

    Returns
    -------
    sarea : float
        Signed solid angle in steradians.
    """
    v1 = x1 - xq
    v2 = x2 - xq
    v3 = x3 - xq

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    n3 = np.cross(v3, v1)

    r1 = np.linalg.norm(n1)
    r2 = np.linalg.norm(n2)
    r3 = np.linalg.norm(n3)

    if r1 <= EM20 or r2 <= EM20 or r3 <= EM20:
        return 0.0

    ss1 = -np.dot(n1, n2) / (r1 * r2)
    ss2 = -np.dot(n2, n3) / (r2 * r3)
    ss3 = -np.dot(n3, n1) / (r3 * r1)

    ss1 = np.clip(ss1, -1.0, 1.0)
    ss2 = np.clip(ss2, -1.0, 1.0)
    ss3 = np.clip(ss3, -1.0, 1.0)

    sarea = math.acos(ss1) + math.acos(ss2) + math.acos(ss3) - math.pi

    # Orientation check using panel normal and center (incpflow.F lines 640-654)
    normal = np.cross(x2 - x1, x3 - x1)
    xc = (x1 + x2 + x3) / 3.0
    ss = np.dot(normal, xc - xq)
    if ss < 0.0:
        sarea = -sarea

    return sarea


# ============================================================================
# BEM System Assembly and Solution (bemsolv.F)
# ============================================================================

def assemble_bem_system(
    x: np.ndarray,
    elem: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble BEM influence matrices HBEM and GBEM for potential flow.

    Ported from $OR_SRC/engine/source/fluid/bemsolv.F lines 68-180.

    Parameters
    ----------
    x : np.ndarray
        Nodal coordinates (N_nodes, 3).
    elem : np.ndarray
        Triangle element connectivity (N_elem, 3), 0-based.

    Returns
    -------
    hbem : np.ndarray
        Influence matrix H (N_nodes, N_nodes).
    gbem : np.ndarray
        Influence matrix G (N_nodes, N_elem + 1).
    """
    n_nodes = len(x)
    n_elem = len(elem)

    hbem = np.zeros((n_nodes, n_nodes), dtype=float)
    gbem = np.zeros((n_nodes, n_elem + 1), dtype=float)

    norm_vec, elarea, _ = compute_triangle_normals_and_areas(x, elem)

    # Centroids and corner squared distances for all elements
    n1_all = elem[:, 0]
    n2_all = elem[:, 1]
    n3_all = elem[:, 2]

    x1_all = x[n1_all]
    x2_all = x[n2_all]
    x3_all = x[n3_all]

    x0_all = (x1_all + x2_all + x3_all) / 3.0
    d2_all = np.minimum(
        np.sum((x0_all - x1_all) ** 2, axis=1),
        np.minimum(
            np.sum((x0_all - x2_all) ** 2, axis=1),
            np.sum((x0_all - x3_all) ** 2, axis=1),
        ),
    )

    area2_all = 2.0 * elarea

    # Loop over elements and collocation nodes (bemsolv.F lines 79-124)
    for iel in range(n_elem):
        n1 = n1_all[iel]
        n2 = n2_all[iel]
        n3 = n3_all[iel]

        x1 = x1_all[iel]
        x2 = x2_all[iel]
        x3 = x3_all[iel]
        x0 = x0_all[iel]
        d2 = d2_all[iel]
        jac = area2_all[iel]
        n_vec = norm_vec[iel]

        for jn in range(n_nodes):
            xs = x[jn]
            is_v = (jn == n1 or jn == n2 or jn == n3)

            # Double-layer potential integral
            rval_h = int_h_tg(xs, x1, x2, x3, x0, d2, n_vec, is_v)
            hbem[jn, n1] += rval_h[0]
            hbem[jn, n2] += rval_h[1]
            hbem[jn, n3] += rval_h[2]

            # Single-layer potential integral
            rval_g = int_g_tg(xs, x1, x2, x3, x0, d2, jac)
            gbem[jn, iel] += rval_g
            gbem[jn, n_elem] += rval_g

    # Rigid-body mode / row-sum correction on diagonal (bemsolv.F lines 125-131):
    # HBEM(IN, IN) = - sum_{JN != IN} HBEM(IN, JN)
    for i in range(n_nodes):
        diag_sum = np.sum(hbem[i, :]) - hbem[i, i]
        hbem[i, i] = -diag_sum

    # Far-field closure condition (bemsolv.F lines 175-177):
    # HBEM(IN, NN) = - GBEM(IN, NEL+1)
    hbem[:, -1] = -gbem[:, n_elem]

    return hbem, gbem


def solve_bem_system(
    hbem: np.ndarray,
    gbem: np.ndarray,
    q: np.ndarray,
    phi_inf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve BEM linear system H * Phi = G * Q - Phi_inf for boundary potential.

    Ported from $OR_SRC/engine/source/fluid/bemsolv.F lines 185-214.

    Parameters
    ----------
    hbem : np.ndarray
        BEM double-layer matrix H (N_nodes, N_nodes).
    gbem : np.ndarray
        BEM single-layer matrix G (N_nodes, N_elem + 1).
    q : np.ndarray
        Normal flux / velocity on elements (N_elem,).
    phi_inf : Optional[np.ndarray]
        Far-field potential at nodes (N_nodes,).

    Returns
    -------
    phi : np.ndarray
        Nodal velocity potentials (N_nodes,).
    """
    n_nodes = hbem.shape[0]
    n_elem = len(q)

    # Extended flux vector with zero far-field closure element (bemsolv.F line 187)
    q_ext = np.zeros(n_elem + 1, dtype=float)
    q_ext[:n_elem] = q

    rhs = gbem @ q_ext

    if phi_inf is not None:
        rhs -= phi_inf[:n_nodes]

    # Solve linear system H * phi = rhs
    # Note: BEM matrices with far-field closure are regular and well-conditioned
    phi = np.linalg.solve(hbem, rhs)
    return phi


# ============================================================================
# Post-Processing: Velocities, Pressure, and Forces (incpflow.F)
# ============================================================================

def compute_surface_velocities(
    x: np.ndarray,
    elem: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    elarea: np.ndarray,
    nodarea: np.ndarray,
    norm_vec: np.ndarray,
) -> np.ndarray:
    """Compute surface velocities by combining tangential gradient and normal flux.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 513-561.

    Returns
    -------
    u : np.ndarray
        Nodal velocities (N_nodes, 3).
    """
    n_nodes = len(x)
    n_elem = len(elem)
    u = np.zeros((n_nodes, 3), dtype=float)

    for i in range(n_elem):
        n1 = elem[i, 0]
        n2 = elem[i, 1]
        n3 = elem[i, 2]

        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]

        # Tangential gradient via TRGRAD (incpflow.F lines 538-540)
        grad = trgrad(x1, x2, x3, phi[n1], phi[n2], phi[n3])

        nr = norm_vec[i]
        area = elarea[i]
        area2 = 2.0 * area

        if area2 > EM20:
            unit_n = nr / area2
        else:
            unit_n = np.zeros(3, dtype=float)

        # Total panel velocity = tangential + normal (incpflow.F lines 546-548)
        u_elem = grad + q[i] * unit_n

        # Area-weighted scatter to corner nodes (incpflow.F lines 552-560)
        w_nod = (area / 3.0)
        u[n1] += u_elem * w_nod / np.maximum(nodarea[n1], EM20)
        u[n2] += u_elem * w_nod / np.maximum(nodarea[n2], EM20)
        u[n3] += u_elem * w_nod / np.maximum(nodarea[n3], EM20)

    return u


def evaluate_field_points(
    xq: np.ndarray,
    x_surf: np.ndarray,
    elem: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    norm_vec: np.ndarray,
    elarea: np.ndarray,
    v_inf: Optional[np.ndarray] = None,
    phi_inf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate velocity potential and velocity at arbitrary query points.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 674-752.

    Parameters
    ----------
    xq : np.ndarray
        Field query points (N_query, 3).
    x_surf : np.ndarray
        Surface node coordinates (N_surf, 3).
    elem : np.ndarray
        Triangle element connectivity (N_elem, 3).
    phi : np.ndarray
        Surface potentials (N_surf,).
    q : np.ndarray
        Surface normal fluxes (N_elem,).
    norm_vec : np.ndarray
        Element normal vectors (N_elem, 3).
    elarea : np.ndarray
        Element areas (N_elem,).
    v_inf : Optional[np.ndarray]
        Far-field velocity (3,).
    phi_inf : Optional[np.ndarray]
        Far-field potential at query points (N_query,).

    Returns
    -------
    phi_q : np.ndarray
        Potentials at query points (N_query,).
    u_q : np.ndarray
        Velocities at query points (N_query, 3).
    """
    n_query = len(xq)
    n_elem = len(elem)
    phi_q = np.zeros(n_query, dtype=float)
    u_q = np.zeros((n_query, 3), dtype=float)

    n1_all = elem[:, 0]
    n2_all = elem[:, 1]
    n3_all = elem[:, 2]

    x1_all = x_surf[n1_all]
    x2_all = x_surf[n2_all]
    x3_all = x_surf[n3_all]
    x0_all = (x1_all + x2_all + x3_all) / 3.0
    d2_all = np.minimum(
        np.sum((x0_all - x1_all) ** 2, axis=1),
        np.minimum(
            np.sum((x0_all - x2_all) ** 2, axis=1),
            np.sum((x0_all - x3_all) ** 2, axis=1),
        ),
    )

    inv_4pi = 0.25 / math.pi

    for i in range(n_query):
        pt = xq[i]
        for j in range(n_elem):
            n1 = n1_all[j]
            n2 = n2_all[j]
            n3 = n3_all[j]

            x1 = x1_all[j]
            x2 = x2_all[j]
            x3 = x3_all[j]
            x0 = x0_all[j]
            d2 = d2_all[j]
            area = elarea[j]
            area2 = 2.0 * area
            nr = norm_vec[j] / max(area2, EM20)
            qm = q[j]

            r2_source = np.sum((x0 - pt) ** 2)
            if r2_source > 100.0 * d2:
                pg = _PG_1
                wpg = _WPG_1
            elif r2_source > 25.0 * d2:
                pg = _PG_4
                wpg = _WPG_4
            elif r2_source > 4.0 * d2:
                pg = _PG_7
                wpg = _WPG_7
            else:
                pg = _PG_13
                wpg = _WPG_13

            for k in range(len(wpg)):
                w = wpg[k]
                eta1 = pg[k, 0]
                eta2 = pg[k, 1]
                val1 = 1.0 - eta1 - eta2
                val2 = eta1
                val3 = eta2

                xm = val1 * x1 + val2 * x2 + val3 * x3
                phim = val1 * phi[n1] + val2 * phi[n2] + val3 * phi[n3]

                r_vec = pt - xm
                r2 = np.sum(r_vec ** 2)
                if r2 > EM20:
                    r = math.sqrt(r2)
                    r3 = r2 * r
                    fac = inv_4pi / r3
                    dot_nr = np.dot(r_vec, nr)
                    fac2 = 3.0 / r2 * dot_nr

                    phis = inv_4pi / r
                    qs = fac * dot_nr

                    dps = -fac * r_vec
                    dqs = fac * (nr - r_vec * fac2)

                    phi_q[i] += area * w * (qm * phis - phim * qs)
                    u_q[i] += area * w * (qm * dps - phim * dqs)

        if phi_inf is not None:
            phi_q[i] += phi_inf[i]
        if v_inf is not None:
            u_q[i] += v_inf

    return phi_q, u_q


def compute_unsteady_bernoulli_pressure(
    phi: np.ndarray,
    phi_old: np.ndarray,
    u: np.ndarray,
    rho: float,
    dt: float,
    pa: float = 0.0,
    a_inf: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    x_ref: Optional[np.ndarray] = None,
    ncycle: int = 2,
) -> np.ndarray:
    """Compute unsteady Bernoulli pressure P = PA - rho * dPhi/dt - 0.5 * rho * |u|^2.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 1108-1132.

    Parameters
    ----------
    phi : np.ndarray
        Current potential (N,).
    phi_old : np.ndarray
        Previous potential (N,).
    u : np.ndarray
        Velocities (N, 3).
    rho : float
        Fluid density.
    dt : float
        Time step dt.
    pa : float
        Stagnation / reference pressure.
    a_inf : Optional[np.ndarray]
        Far-field acceleration vector (3,).
    x : Optional[np.ndarray]
        Node coordinates (N, 3).
    x_ref : Optional[np.ndarray]
        Reference point coordinates (3,).
    ncycle : int
        Current cycle count. If cycle <= 1, time derivative is set to zero.

    Returns
    -------
    pres : np.ndarray
        Nodal pressures (N,).
    """
    n = len(phi)
    if dt <= 0.0:
        return np.zeros(n, dtype=float)

    if ncycle > 1:
        phip = (phi - phi_old) / dt
        if a_inf is not None and x is not None and x_ref is not None:
            dx = x - x_ref
            phip += np.dot(dx, a_inf)
    else:
        phip = np.zeros(n, dtype=float)

    u_sq = np.sum(u ** 2, axis=1)
    pres = pa - rho * phip - 0.5 * rho * u_sq
    return pres


def compute_boundary_forces_and_work(
    elem: np.ndarray,
    pres: np.ndarray,
    norm_vec: np.ndarray,
    elarea: np.ndarray,
    v: Optional[np.ndarray] = None,
    dt: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Compute consistent structural boundary forces and work ledger accumulation.

    Ported from $OR_SRC/engine/source/fluid/incpflow.F lines 1137-1166.

    Parameters
    ----------
    elem : np.ndarray
        Triangle element connectivity (N_elem, 3).
    pres : np.ndarray
        Nodal pressures (N_nodes,).
    norm_vec : np.ndarray
        Element normal vectors (N_elem, 3) = (x2-x1) x (x3-x1) (magnitude = 2 * Area).
    elarea : np.ndarray
        Element areas (N_elem,).
    v : Optional[np.ndarray]
        Nodal velocities (N_nodes, 3).
    dt : float
        Time step dt.

    Returns
    -------
    f_node : np.ndarray
        Nodal force vectors (N_nodes, 3).
    dwfext : float
        External work done by fluid on structure over this cycle [J].
    """
    n_nodes = len(pres)
    n_elem = len(elem)
    f_node = np.zeros((n_nodes, 3), dtype=float)

    n1 = elem[:, 0]
    n2 = elem[:, 1]
    n3 = elem[:, 2]

    # Consistent pressure coefficient: COEF = 1/6 * 1/3 * (PRES(N1) + PRES(N2) + PRES(N3))
    # Combined with NORM = 2 * Area * unit_n, each node gets:
    # F = COEF * NORM = (1/18 * sum_P) * (2 * Area * unit_n) = (P_mean * Area / 3) * unit_n
    p_mean = (pres[n1] + pres[n2] + pres[n3]) / 3.0
    coef = p_mean / 6.0

    # Nodal force from each element = COEF * NORM
    f_elem_node = coef[:, None] * norm_vec

    np.add.at(f_node, n1, f_elem_node)
    np.add.at(f_node, n2, f_elem_node)
    np.add.at(f_node, n3, f_elem_node)

    # Work ledger accumulation (incpflow.F lines 1157-1165):
    # WFEXT += sum_elem COEF * dot(NORM, v_node) * dt
    dwfext = 0.0
    if v is not None and dt > 0.0:
        dwfext = float(np.sum(f_node * v)) * dt

    return f_node, dwfext


# ============================================================================
# Main BEM Incompressible Potential Flow Solver Class
# ============================================================================

class BemIncompressibleFlow:
    """Full BEM Incompressible Potential Flow Solver (/BEM/FLOW).

    Integrates ``incpflow.F``, ``bemsolv.F``, and ``flow0.F``.
    """

    def __init__(
        self,
        params: BemFlowParams,
        elem: np.ndarray,
        x0: np.ndarray,
        auxiliary_nodes: Optional[np.ndarray] = None,
    ):
        """Initialize the BEM flow solver.

        Parameters
        ----------
        params : BemFlowParams
            BEM flow configuration and fluid parameters.
        elem : np.ndarray
            Triangle surface mesh connectivity (N_elem, 3), 0-based.
        x0 : np.ndarray
            Initial surface node coordinates (N_surf, 3).
        auxiliary_nodes : Optional[np.ndarray]
            Optional internal/auxiliary field query node indices or coordinates.
        """
        self.params = params
        self.elem = np.asarray(elem, dtype=np.int64)
        self.n_nodes = len(x0)
        self.n_elem = len(self.elem)

        # Normalize far-field direction
        norm_dir = np.linalg.norm(self.params.dir)
        if norm_dir > EM20:
            self.dir = self.params.dir / norm_dir
        else:
            self.dir = np.array([1.0, 0.0, 0.0], dtype=float)

        # State arrays
        self.phi = np.zeros(self.n_nodes, dtype=float)
        self.phi_old = np.zeros(self.n_nodes, dtype=float)
        self.pres = np.zeros(self.n_nodes, dtype=float)
        self.u = np.zeros((self.n_nodes, 3), dtype=float)

        # Work energy ledger (double precision, flow0.F lines 68, 192)
        self.wfext: float = 0.0

        # Subcycling and assembly flags
        self.last_assemble_time: float = -1.0e30
        self.hbem: Optional[np.ndarray] = None
        self.gbem: Optional[np.ndarray] = None

        # Previous far-field velocity for acceleration computation
        self.vg_old = np.zeros(3, dtype=float)
        self.ncycle = 0

    def step(
        self,
        x: np.ndarray,
        v: np.ndarray,
        dt: float,
        time: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Execute one time step of BEM incompressible flow solver.

        Ported from $OR_SRC/engine/source/fluid/incpflow.F

        Parameters
        ----------
        x : np.ndarray
            Current nodal coordinates (N_nodes, 3).
        v : np.ndarray
            Current nodal velocities (N_nodes, 3).
        dt : float
            Current time step dt.
        time : float
            Current solver time TT.

        Returns
        -------
        f_fluid : np.ndarray
            Nodal forces exerted by the fluid on the structure (N_nodes, 3).
        pres : np.ndarray
            Nodal pressures (N_nodes,).
        dwfext : float
            Work done on structure over this step [J].
        """
        self.ncycle += 1
        if dt <= 0.0:
            return np.zeros((self.n_nodes, 3), dtype=float), self.pres, 0.0

        # 0. Geometry
        norm_vec, elarea, nodarea = compute_triangle_normals_and_areas(x, self.elem)

        # Store previous potentials
        self.phi_old[:] = self.phi

        # 1. Far-field kinematics
        v_speed = self.params.v_inf_const
        if self.params.ifvini > 0 and self.params.sfvini != 0.0:
            v_speed = self.params.sfvini  # or evaluate curve
        vg = self.dir * v_speed
        if dt > 0.0:
            ag = (vg - self.vg_old) / dt
        else:
            ag = np.zeros(3, dtype=float)
        self.vg_old = vg.copy()

        # Far-field potential at surface nodes: PHI_INF = vg . (x - x_ref)
        x_ref = x[-1] if len(x) > 0 else np.zeros(3, dtype=float)
        phi_inf = np.dot(x - x_ref, vg)

        # 2. Surface normal fluxes Q (incpflow.F lines 336-357)
        # VX = 1/3 * sum(VL) ... Q = dot(VX, normal) / (2 * Area)
        n1 = self.elem[:, 0]
        n2 = self.elem[:, 1]
        n3 = self.elem[:, 2]
        v_elem = (v[n1] + v[n2] + v[n3]) / 3.0
        area2 = 2.0 * elarea
        unit_n = np.where(area2[:, None] > EM20, norm_vec / np.maximum(area2[:, None], EM20), 0.0)
        q = np.sum(v_elem * unit_n, axis=1)

        # 3. BEM System Assembly (with subcycling if enabled)
        need_assemble = True
        if self.params.dtsub > 0.0 and self.hbem is not None and self.gbem is not None:
            if time < self.last_assemble_time + self.params.dtsub:
                need_assemble = False

        if need_assemble:
            self.hbem, self.gbem = assemble_bem_system(x, self.elem)
            self.last_assemble_time = time

        # 4. Linear System Solution (bemsolv.F)
        self.phi = solve_bem_system(self.hbem, self.gbem, q, phi_inf)

        # 5. Velocities (incpflow.F lines 513-561)
        self.u = compute_surface_velocities(x, self.elem, self.phi, q, elarea, nodarea, norm_vec)

        # 6. Pressure (incpflow.F lines 1058-1132)
        pa = self.params.pa_const
        if self.params.ifpa > 0 and self.params.sfpa != 0.0:
            pa = self.params.sfpa
        self.pres = compute_unsteady_bernoulli_pressure(
            self.phi,
            self.phi_old,
            self.u,
            self.params.rho,
            dt,
            pa=pa,
            a_inf=ag,
            x=x,
            x_ref=x_ref,
            ncycle=self.ncycle,
        )

        # 7. Structural Boundary Forces and Work Ledger (incpflow.F lines 1137-1166)
        f_fluid, dwfext = compute_boundary_forces_and_work(
            self.elem,
            self.pres,
            norm_vec,
            elarea,
            v=v,
            dt=dt,
        )
        self.wfext += dwfext

        return f_fluid, self.pres, dwfext
