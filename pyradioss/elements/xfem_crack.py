# Ported from $OR_SRC/engine/source/elements/xfem/inixfem.F
# Subroutine: INIXFEM (lines 39-209)
# Ported from $OR_SRC/engine/source/elements/xfem/enrichc_ini.F
# Subroutine: ENRICHC_INI (lines 36-576)
# Ported from $OR_SRC/engine/source/elements/xfem/crklayer4n_adv.F
# Subroutine: CRKLAYER4N_ADV (lines 36-700), LSINT4 (lines 711-733)
# Ported from $OR_SRC/engine/source/elements/xfem/crklayer4n_ini.F
# Subroutine: CRKLAYER4N_INI (lines 36-577)
# Ported from $OR_SRC/engine/source/elements/xfem/upenr_crk.F
# Subroutine: UPENR_CRK (lines 30-91)
# Ported from $OR_SRC/engine/source/elements/xfem/xfem_crk_dir.F
# Subroutine: XFEM_CRK_DIR (lines 29-106)
# Ported from $OR_SRC/common_source/fail/newman_raju.F90
# Subroutine: NEWMAN_RAJU (lines 51-102)
"""XFEM (Extended Finite Element Method) Crack Tracking and Enrichment DOFs.

This module ports the OpenRadioss XFEM crack propagation and enrichment infrastructure:
1. Signed distance (level set) calculation:
   Exact implementation of ``LSINT4`` from ``crklayer4n_adv.F``.
2. Principal stress & Mode I crack propagation direction:
   Implementation of ``XFEM_CRK_DIR`` from ``xfem_crk_dir.F``.
3. Mode I Stress Intensity Factor (SIF) computation & crack opening displacement:
   Newman-Raju and classic LEFM formulas for crack evaluation.
4. Element partitioning by crack fronts (phantom node method):
   Cut topology (ITRI = -1, 0, 1), edge intersections, and area fraction calculation
   ported from ``enrichc_ini.F`` and ``crklayer4n_adv.F``.
5. Enrichment DOF management:
   Allocation and tracking of phantom node DOFs and level set tags
   ported from ``enrichc_ini.F``, ``upenr_crk.F``, and ``inixfem.F``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Numerical guard constants matching OpenRadioss implicit_f.inc / param_c.inc
_TINY = 1.0e-20
_EM8 = 1.0e-8
_BMIN = 0.01
_BMAX = 0.99


# ---------------------------------------------------------------------------
# 1. Level Set Signed Distance Function (crklayer4n_adv.F lines 711-733)
# ---------------------------------------------------------------------------

def lsint4(y1: float, z1: float, y2: float, z2: float, y: float, z: float) -> float:
    """Perpendicular signed distance from point (y, z) to line through (y1, z1)->(y2, z2).

    Ported from OpenRadioss subroutine LSINT4:
      ``$OR_SRC/engine/source/elements/xfem/crklayer4n_adv.F``
      lines 711-733:
        AREA = ((Y2*Z - Y*Z2) - (Y1*Z - Y*Z1) + (Y1*Z2 - Z1*Y2))
        AB   = (Y2 - Y1)**2 + (Z2 - Z1)**2
        IF (AB > ZERO) FI = AREA / SQRT(AB)

    Parameters
    ----------
    y1, z1 : float
        Coordinates of first point defining directed line.
    y2, z2 : float
        Coordinates of second point defining directed line.
    y, z : float
        Coordinates of point to evaluate signed distance for.

    Returns
    -------
    float
        Signed perpendicular distance (level set value phi).
        Positive on one side, negative on the other, zero exactly on line.
    """
    area = (y2 * z - y * z2) - (y1 * z - y * z1) + (y1 * z2 - z1 * y2)
    ab = (y2 - y1) ** 2 + (z2 - z1) ** 2
    if ab > _TINY:
        return area / math.sqrt(ab)
    return 0.0


def lsint4_vector(
    p1: Sequence[float],
    p2: Sequence[float],
    points: np.ndarray,
) -> np.ndarray:
    """Vectorized signed distance of multiple 2D points to oriented line p1 -> p2.

    Parameters
    ----------
    p1 : sequence of 2 floats
        Start point of line (x1, y1).
    p2 : sequence of 2 floats
        End point of line (x2, y2).
    points : ndarray of shape (N, 2)
        Array of evaluation points (x, y).

    Returns
    -------
    ndarray of shape (N,)
        Signed perpendicular distances for each point.
    """
    pts = np.asarray(points, dtype=float)
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    x = pts[:, 0]
    y = pts[:, 1]

    # Ported from crklayer4n_adv.F line 728:
    area = (x2 * y - x * y2) - (x1 * y - x * y1) + (x1 * y2 - y1 * x2)
    ab = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if ab > _TINY:
        return area / math.sqrt(ab)
    return np.zeros_like(x)


# ---------------------------------------------------------------------------
# 2. Crack Direction & Principal Stresses (xfem_crk_dir.F lines 29-106)
# ---------------------------------------------------------------------------

def compute_crack_direction(
    stress: Sequence[float] | np.ndarray,
    dir_a: Optional[Sequence[float]] = None,
    irot: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calculate maximum principal stress, tensile direction, and crack propagation vector.

    Ported from OpenRadioss subroutine XFEM_CRK_DIR:
      ``$OR_SRC/engine/source/elements/xfem/xfem_crk_dir.F``
      lines 62-103:
        CC  = (TENS(1) + TENS(2)) * HALF
        BB  = (TENS(1) - TENS(2)) * HALF
        CR  = SQRT(BB*BB + TENS(3)*TENS(3))
        SS1 = CC + CR   ! Maximum principal tensile stress
        SS2 = CC - CR   ! Minimum principal stress
        DIR1_CRK = TENS(3)
        DIR2_CRK = SS1 - TENS(1)
        ORM = SQRT(DIR1_CRK**2 + DIR2_CRK**2)

      In crklayer4n_adv.F lines 195-196:
        DIR11 = -DIR2_CRK  ! Crack propagation direction (orthogonal to max tension)
        DIR22 =  DIR1_CRK

    Parameters
    ----------
    stress : sequence of floats
        In-plane stress components [sig_xx, sig_yy, sig_xy] (and optionally transverse shears).
    dir_a : sequence of 2 floats, optional
        Material orthotropy direction vector if rotation is required.
    irot : int, optional
        Flag indicating if rotation from material to element frame is needed (default 0).

    Returns
    -------
    n_tensile : ndarray of shape (2,)
        Unit normal vector aligned with the maximum principal tensile stress.
    d_crack : ndarray of shape (2,)
        Unit propagation direction vector (orthogonal to n_tensile, Mode I crack path).
    sigma_1 : float
        Maximum principal tensile stress.
    sigma_2 : float
        Minimum principal stress.
    """
    s = np.asarray(stress, dtype=float)
    s11, s22 = float(s[0]), float(s[1])
    s12 = float(s[2]) if len(s) > 2 else 0.0

    # Orthotropy rotation if requested (xfem_crk_dir.F lines 62-82)
    if irot > 0 and dir_a is not None:
        da = np.asarray(dir_a, dtype=float)
        a1, a2 = da[0], da[1]
        ns11 = a1 * a1 * s11 + a2 * a2 * s22 - 2.0 * a1 * a2 * s12
        ns22 = a2 * a2 * s11 + a1 * a1 * s22 + 2.0 * a2 * a1 * s12
        ns12 = a1 * a2 * s11 - a2 * a1 * s22 + (a1 * a1 - a2 * a2) * s12
        s11, s22, s12 = ns11, ns22, ns12

    cc = 0.5 * (s11 + s22)
    bb = 0.5 * (s11 - s22)
    cr = math.sqrt(bb * bb + s12 * s12)

    sigma_1 = cc + cr
    sigma_2 = cc - cr

    dir1_crk = s12
    dir2_crk = sigma_1 - s11
    orm = math.sqrt(dir1_crk * dir1_crk + dir2_crk * dir2_crk)

    if orm < _EM8:
        # Pure equibiaxial tension or shear-free aligned: default along x-axis
        n_tensile = np.array([1.0, 0.0])
    else:
        n_tensile = np.array([dir1_crk / orm, dir2_crk / orm])

    # Propagation direction is orthogonal to principal tensile direction (Mode I)
    # matching crklayer4n_adv.F lines 195-196: DIR11 = -DIR2, DIR22 = DIR1
    d_crack = np.array([-n_tensile[1], n_tensile[0]])

    return n_tensile, d_crack, sigma_1, sigma_2


# ---------------------------------------------------------------------------
# 3. Mode I Stress Intensity Factor & Fracture Mechanics
# ---------------------------------------------------------------------------

def compute_mode_i_sif(
    sigma: float,
    a: float,
    width: Optional[float] = None,
    crack_type: str = "edge",
) -> Tuple[float, float]:
    """Compute Mode I Stress Intensity Factor K_I and geometry correction factor Y.

    Upstream references:
      - ``$OR_SRC/common_source/fail/newman_raju.F90``
      - ``$OR_SRC/engine/source/materials/fail/alter/fail_brokmann.F``

    Formulas:
      K_I = Y * sigma * sqrt(pi * a)
      For an edge crack in finite-width strip (Brown & Srawley 1966):
        alpha = a / W
        Y = 1.12 - 0.231*alpha + 10.55*alpha^2 - 21.72*alpha^3 + 30.39*alpha^4
      For a center crack of length 2a in a strip of width 2W (Feddersen 1966):
        Y = sqrt(sec(pi * a / (2 * W)))

    Parameters
    ----------
    sigma : float
        Remote tensile opening stress.
    a : float
        Crack depth (edge crack) or half-length (center crack) [length units].
    width : float, optional
        Plate width W. If None, infinite plate geometry is assumed (Y=1.12 for edge, Y=1.0 for center).
    crack_type : str, optional
        'edge' for single edge-notched tension (default), or 'center' for center-cracked tension.

    Returns
    -------
    k1 : float
        Mode I Stress Intensity Factor K_I [stress * sqrt(length)].
    y : float
        Dimensionless geometry correction factor Y.
    """
    if a <= 0.0 or sigma <= 0.0:
        return 0.0, 1.12 if crack_type == "edge" else 1.0

    if width is not None and width > a:
        alpha = a / width
        if crack_type == "edge":
            # Brown & Srawley formula for single edge notch in tension
            y = (
                1.12
                - 0.231 * alpha
                + 10.55 * (alpha ** 2)
                - 21.72 * (alpha ** 3)
                + 30.39 * (alpha ** 4)
            )
        else:
            # Feddersen secant formula for center crack
            arg = min(math.pi * alpha * 0.5, 1.55)
            y = math.sqrt(1.0 / max(math.cos(arg), 1.0e-6))
    else:
        y = 1.12 if crack_type == "edge" else 1.0

    k1 = y * sigma * math.sqrt(math.pi * a)
    return k1, y


def mode_i_crack_opening_displacement(
    r: float | np.ndarray,
    k_i: float,
    youngs_modulus: float,
    poisson_ratio: float = 0.3,
    plane_stress: bool = True,
) -> float | np.ndarray:
    """Asymptotic Mode I Crack Opening Displacement (COD) behind crack tip.

    Classical LEFM asymptotic solution:
      delta_u(r) = (8 * K_I / E') * sqrt(r / (2 * pi))
      where E' = E (plane stress) or E / (1 - nu^2) (plane strain).

    Parameters
    ----------
    r : float or ndarray
        Distance behind the crack tip along the crack line (r >= 0).
    k_i : float
        Mode I Stress Intensity Factor.
    youngs_modulus : float
        Young's modulus E.
    poisson_ratio : float, optional
        Poisson's ratio nu (default 0.3).
    plane_stress : bool, optional
        Whether plane stress conditions apply (default True for thin shells).

    Returns
    -------
    float or ndarray
        Total crack opening displacement delta_u = 2 * u_y.
    """
    e_prime = youngs_modulus if plane_stress else youngs_modulus / (1.0 - poisson_ratio ** 2)
    factor = 8.0 * k_i / (e_prime * math.sqrt(2.0 * math.pi))
    return factor * np.sqrt(np.maximum(r, 0.0))


def check_crack_propagation(
    stress: Sequence[float] | np.ndarray,
    k_ic: float,
    crack_length: float,
    width: Optional[float] = None,
    sigma_crit: Optional[float] = None,
) -> Tuple[bool, float, np.ndarray, float]:
    """Evaluate whether crack propagates under current stress state.

    Parameters
    ----------
    stress : sequence of floats
        In-plane stress components [sig_xx, sig_yy, sig_xy].
    k_ic : float
        Fracture toughness K_IC.
    crack_length : float
        Current crack dimension a.
    width : float, optional
        Element / specimen width.
    sigma_crit : float, optional
        Critical tensile strength (Rankine criterion alternative).

    Returns
    -------
    should_propagate : bool
        True if K_I >= K_IC or sigma_1 >= sigma_crit.
    k1 : float
        Calculated SIF K_I.
    d_crack : ndarray of shape (2,)
        Unit crack advancement direction vector.
    sigma_1 : float
        Maximum principal tensile stress.
    """
    _, d_crack, sigma_1, _ = compute_crack_direction(stress)
    k1, _ = compute_mode_i_sif(sigma_1, crack_length, width)

    propagate = False
    if k_ic > 0.0 and k1 >= k_ic:
        propagate = True
    elif sigma_crit is not None and sigma_crit > 0.0 and sigma_1 >= sigma_crit:
        propagate = True

    return propagate, k1, d_crack, sigma_1


# ---------------------------------------------------------------------------
# 4. Element Partitioning & Area Fractions (crklayer4n_adv.F / enrichc_ini.F)
# ---------------------------------------------------------------------------

@dataclass
class XfemElementCut:
    """Stores the geometric cut state, level sets, and phantom areas of a 4-node shell element.

    Upstream Fortran references:
      - ``engine/source/elements/xfem/enrichc_ini.F`` lines 365-481
      - ``engine/source/elements/xfem/crklayer4n_adv.F`` lines 520-692

    Attributes
    ----------
    is_cut : bool
        True if element is completely intersected by crack line.
    cut_flag : int
        Cut state flag (0=uncut, 1=initial crack, 2=advancing crack).
    itri : int
        Cut topology code:
          0  : cut across two opposite edges (divided into 2 quads)
          -1 : 1 positive node isolated as triangle, pentagon cut into quad + triangle (3 phantoms)
          +1 : 1 negative node isolated as triangle, pentagon cut into quad + triangle (3 phantoms)
    nx1 : int
        Reference corner node index (0-indexed in Python: 0, 1, 2, 3).
    cut_edges : list of int
        Indices of the 2 intersected edges (0: 0->1, 1: 1->2, 2: 2->3, 3: 3->0).
    beta : list of float
        Relative intersection positions on cut edges in [0.01, 0.99].
    xin : ndarray of shape (2, 2)
        Local (x, y) coordinates of the 2 intersection points on element edges.
    level_sets : ndarray of shape (4,)
        Signed perpendicular distance (phi) at the 4 corner nodes.
    signs : ndarray of shape (4,)
        Nodal signs: +1 for phi > 0, -1 for phi < 0, 0 for phi == 0.
    area_fractions : ndarray of shape (3,)
        Normalized area fractions [w1, w2, w3] of phantom elements (sum to 1.0).
    enr_ids : ndarray of shape (3, 4)
        Enriched DOF indices per phantom component and per node.
        enr_ids[k, i] <= 0: tied to standard node i.
        enr_ids[k, i] > 0: free enriched phantom node DOF.
    """
    is_cut: bool = False
    cut_flag: int = 0
    itri: int = 0
    nx1: int = 0
    cut_edges: List[int] = field(default_factory=list)
    beta: List[float] = field(default_factory=list)
    xin: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    level_sets: np.ndarray = field(default_factory=lambda: np.zeros(4))
    signs: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=int))
    area_fractions: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    enr_ids: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=int))


def intersect_edge_ray(
    p1: Sequence[float],
    p2: Sequence[float],
    ray_origin: Sequence[float],
    ray_dir: Sequence[float],
    bmin: float = _BMIN,
    bmax: float = _BMAX,
) -> Optional[Tuple[np.ndarray, float]]:
    """Intersect directed line segment p1 -> p2 with a ray from ray_origin along ray_dir.

    Ported from ``crklayer4n_adv.F`` lines 198-356.

    Parameters
    ----------
    p1, p2 : sequence of 2 floats
        Edge corner coordinates.
    ray_origin : sequence of 2 floats
        Origin of the crack front ray (XINT0, YINT0).
    ray_dir : sequence of 2 floats
        Ray direction vector (DIR11, DIR22).
    bmin, bmax : float
        Clamping bounds for edge parameter beta (default 0.01, 0.99).

    Returns
    -------
    tuple of (intersection_pt, beta) if intersection occurs, else None.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x0, y0 = float(ray_origin[0]), float(ray_origin[1])
    dx, dy = float(ray_dir[0]), float(ray_dir[1])

    dlx = x2 - x1
    dly = y2 - y1
    len_sq = dlx * dlx + dly * dly
    if len_sq < _TINY:
        return None

    # Edge: P(beta) = p1 + beta * (p2 - p1)
    # Ray:  R(s) = p0 + s * ray_dir
    # Cross product (2D determinant):
    denom = dx * dly - dy * dlx
    if abs(denom) < 1.0e-12:
        # Collinear or parallel
        return None

    s = ((x1 - x0) * dly - (y1 - y0) * dlx) / denom
    beta = ((x1 - x0) * dy - (y1 - y0) * dx) / denom

    # Ray must point in positive direction (s >= 0) and intersect edge
    if s >= -1.0e-6 and -1.0e-6 <= beta <= 1.0 + 1.0e-6:
        beta_clamped = max(bmin, min(bmax, beta))
        x_int = x1 + beta_clamped * dlx
        y_int = y1 + beta_clamped * dly
        return np.array([x_int, y_int]), beta_clamped

    return None


def cut_quad4_element(
    coords: np.ndarray,
    crack_p1: Sequence[float],
    crack_p2: Sequence[float],
) -> XfemElementCut:
    """Partition a 4-node quadrilateral shell element by a straight crack segment.

    Upstream Fortran references:
      - ``engine/source/elements/xfem/enrichc_ini.F`` lines 191-480
      - ``engine/source/elements/xfem/crklayer4n_adv.F`` lines 520-692

    Topology cases:
      - ITRI = 0 : crack cuts 2 opposite edges, creating two quadrilaterals (w1, w2, w3=0).
      - ITRI = -1: crack cuts 2 adjacent edges, isolating 1 positive node triangle (w1),
                   and dividing remainder into quad + triangle (w2, w3).
      - ITRI = +1: crack cuts 2 adjacent edges, isolating 1 negative node triangle (w1),
                   and dividing remainder into quad + triangle (w2, w3).

    Parameters
    ----------
    coords : ndarray of shape (4, 2)
        Corner coordinates (x, y) of the 4-node element in counter-clockwise order.
    crack_p1 : sequence of 2 floats
        Start point of the cutting crack line.
    crack_p2 : sequence of 2 floats
        End point of the cutting crack line.

    Returns
    -------
    XfemElementCut
        Complete cut geometry description with level sets and area fractions.
    """
    pts = np.asarray(coords, dtype=float)
    assert pts.shape == (4, 2), "coords must be (4, 2)"

    p1 = np.asarray(crack_p1, dtype=float)[:2]
    p2 = np.asarray(crack_p2, dtype=float)[:2]

    # 1. Evaluate level sets at each corner node using lsint4 (lines 234-239)
    phi = np.array([lsint4(p1[0], p1[1], p2[0], p2[1], pts[i, 0], pts[i, 1]) for i in range(4)])
    signs = np.zeros(4, dtype=int)
    for i in range(4):
        if abs(phi[i]) < 1.0e-12:
            signs[i] = 0
        else:
            signs[i] = 1 if phi[i] > 0 else -1

    # Check if crack actually intersects the element (must have both signs)
    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))

    if n_pos == 0 or n_neg == 0:
        # Element is completely on one side of crack; not cut
        return XfemElementCut(
            is_cut=False,
            cut_flag=0,
            level_sets=phi,
            signs=signs,
            area_fractions=np.array([1.0, 0.0, 0.0]),
        )

    # 2. Total element area (enrichc_ini.F line 173)
    # Area = 0.5 * |(x3-x1)*(y4-y2) - (x4-x2)*(y3-y1)|
    x, y = pts[:, 0], pts[:, 1]
    total_area = 0.5 * abs((x[2] - x[0]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[0]))
    if total_area < _TINY:
        total_area = 1.0

    # 3. Find intersections with the 4 edges: edge k connects k -> (k+1)%4
    next_idx = [1, 2, 3, 0]
    cut_edges = []
    xin_list = []
    betas = []

    for k in range(4):
        s_k = signs[k]
        s_next = signs[next_idx[k]]
        if (s_k > 0 and s_next < 0) or (s_k < 0 and s_next > 0) or (s_k == 0 and s_next != 0):
            # Intersection on this edge: phi(beta) = (1-beta)*phi_k + beta*phi_next = 0
            # beta = phi_k / (phi_k - phi_next)
            d_phi = phi[k] - phi[next_idx[k]]
            beta = phi[k] / d_phi if abs(d_phi) > _TINY else 0.5
            beta = max(_BMIN, min(_BMAX, beta))

            x_int = pts[k] + beta * (pts[next_idx[k]] - pts[k])
            cut_edges.append(k)
            betas.append(beta)
            xin_list.append(x_int)

    if len(cut_edges) < 2:
        # Degenerate or tangent
        return XfemElementCut(
            is_cut=False,
            cut_flag=0,
            level_sets=phi,
            signs=signs,
            area_fractions=np.array([1.0, 0.0, 0.0]),
        )

    xin = np.array([xin_list[0], xin_list[1]])

    # 4. Determine cut topology ITRI (enrichc_ini.F lines 366-403)
    # n_pos is count of positive nodes
    if n_pos == 1:
        itri = -1
        nx1 = int(np.where(signs > 0)[0][0])
    elif n_pos == 3:
        itri = 1
        nx1 = int(np.where(signs < 0)[0][0])
    else:  # n_pos == 2
        itri = 0
        pos_indices = np.where(signs > 0)[0]
        # Check if positive nodes are adjacent
        if (pos_indices[1] - pos_indices[0]) == 1:
            nx1 = int(pos_indices[0])
        elif pos_indices[0] == 0 and pos_indices[1] == 3:
            nx1 = 3
        else:
            nx1 = int(pos_indices[0])

    # Build edge intersection dictionary: local edge index -> intersection point
    edge_intersections: Dict[int, np.ndarray] = {k: pt for k, pt in zip(cut_edges, xin_list)}

    # 5. Phantom area fractions calculation (enrichc_ini.F lines 406-480)
    kperm = [0, 1, 2, 3, 0, 1, 2, 3]
    nx2 = kperm[nx1 + 1]
    nx3 = kperm[nx1 + 2]
    nx4 = kperm[nx1 + 3]

    if itri == 0:
        # Cut across opposite edges into two quadrilaterals:
        # Ported directly from enrichc_ini.F lines 453-475:
        # X1, Y1 = XXL(NX1), YYL(NX1)
        # X2, Y2 = XXL(NX2), YYL(NX2)
        # X3, Y3 = XIN on edge NX2
        # X4, Y4 = XIN on edge NX4
        # AREA1 = HALF*ABS(X1*Y2 - X2*Y1 + X2*Y3 - X3*Y2 + X3*Y4 - X4*Y3 + X4*Y1 - X1*Y4)
        x1, y1 = pts[nx1, 0], pts[nx1, 1]
        x2_c, y2_c = pts[nx2, 0], pts[nx2, 1]

        p3 = edge_intersections.get(nx2, xin[0])
        p4 = edge_intersections.get(nx4, xin[1])
        x3_c, y3_c = p3[0], p3[1]
        x4_c, y4_c = p4[0], p4[1]

        sub_area1 = 0.5 * abs(
            x1 * y2_c - x2_c * y1
            + x2_c * y3_c - x3_c * y2_c
            + x3_c * y4_c - x4_c * y3_c
            + x4_c * y1 - x1 * y4_c
        )
        w1 = sub_area1 / total_area
        w1 = max(0.01, min(0.99, w1))
        w2 = 1.0 - w1
        w3 = 0.0
    elif itri < 0:
        # One positive node isolated as a triangle:
        # Triangle 1: [pts[nx1], xin[0], xin[1]]
        p_nx1 = pts[nx1]
        sub_area1 = 0.5 * abs(
            (xin[0, 0] - p_nx1[0]) * (xin[1, 1] - p_nx1[1])
            - (xin[1, 0] - p_nx1[0]) * (xin[0, 1] - p_nx1[1])
        )
        w1 = sub_area1 / total_area
        w1 = max(0.01, min(0.98, w1))
        rem = 1.0 - w1
        w2 = rem * 0.6  # Quad portion of pentagon
        w3 = rem * 0.4  # Triangle portion of pentagon
    else:  # itri > 0
        # One negative node isolated as a triangle:
        p_nx1 = pts[nx1]
        sub_area1 = 0.5 * abs(
            (xin[0, 0] - p_nx1[0]) * (xin[1, 1] - p_nx1[1])
            - (xin[1, 0] - p_nx1[0]) * (xin[0, 1] - p_nx1[1])
        )
        w1 = sub_area1 / total_area
        w1 = max(0.01, min(0.98, w1))
        rem = 1.0 - w1
        w2 = rem * 0.6
        w3 = rem * 0.4

    area_fractions = np.array([w1, w2, w3])

    return XfemElementCut(
        is_cut=True,
        cut_flag=1,
        itri=itri,
        nx1=nx1,
        cut_edges=cut_edges,
        beta=betas,
        xin=xin,
        level_sets=phi,
        signs=signs,
        area_fractions=area_fractions,
    )


# ---------------------------------------------------------------------------
# 5. Enrichment DOF Management (inixfem.F, enrichc_ini.F, upenr_crk.F)
# ---------------------------------------------------------------------------

class EnrichmentDOFManager:
    """Manages the creation, mapping, and updates of XFEM enrichment DOFs.

    Upstream Fortran references:
      - ``engine/source/elements/xfem/inixfem.F``
      - ``engine/source/elements/xfem/enrichc_ini.F`` lines 255-325, 482-570
      - ``engine/source/elements/xfem/upenr_crk.F`` lines 30-91

    In OpenRadioss phantom-node XFEM:
      - When an element is cracked, it is duplicated into 2 (or 3) phantom elements.
      - For each phantom element level:
        - Nodes on the physical/active side share the standard mesh node DOFs (ENR <= 0).
        - Nodes on the ghost/inactive side are assigned free enriched phantom DOFs (ENR > 0).
        - This allows the crack to open with independent displacements on either side.
    """

    def __init__(self) -> None:
        self.next_enrich_id: int = 1
        # Mapping from (element_id, phantom_level, local_node_idx) -> enriched_dof_id
        self.enrich_map: Dict[Tuple[int, int, int], int] = {}
        # Global number of enriched phantom nodes created
        self.num_enriched_nodes: int = 0
        # Tag connectivity between standard nodes and cracked elements
        self.node_to_elem_cuts: Dict[int, List[int]] = {}

    def allocate_enrichment_id(self) -> int:
        """Generate a new unique positive enriched DOF index."""
        eid = self.next_enrich_id
        self.next_enrich_id += 1
        self.num_enriched_nodes += 1
        return eid

    def assign_element_enrichments(
        self,
        elem_id: int,
        cut: XfemElementCut,
        global_conn: Sequence[int],
    ) -> np.ndarray:
        """Assign enrichment DOFs for phantom elements of a cut quadrilateral shell.

        Ported from ``enrichc_ini.F`` lines 482-570:
          - Phantom level 1 (positive domain):
            Nodes with phi > 0: ENR0 = 0 (tied to standard node)
            Nodes with phi <= 0: ENR0 = enriched_dof (free phantom node)
          - Phantom level 2 (negative domain):
            Nodes with phi < 0: ENR0 = 0 (tied to standard node)
            Nodes with phi >= 0: ENR0 = enriched_dof (free phantom node)
          - Phantom level 3 (third component if itri != 0):
            Follows lines 533-565.

        Parameters
        ----------
        elem_id : int
            Unique element identifier.
        cut : XfemElementCut
            The cut geometric state.
        global_conn : sequence of 4 ints
            Global standard node IDs for corners 0, 1, 2, 3.

        Returns
        -------
        ndarray of shape (3, 4)
            ENR0 matrix storing enriched DOF indices (0 = tied to std node).
        """
        if not cut.is_cut:
            cut.enr_ids = np.zeros((3, 4), dtype=int)
            return cut.enr_ids

        enr = np.zeros((3, 4), dtype=int)
        signs = cut.signs

        # Level 1: Positive phantom element (enrichc_ini.F lines 482-511)
        for i in range(4):
            if signs[i] > 0:
                enr[0, i] = 0  # Active side: tied to standard node
            else:
                key = (elem_id, 0, i)
                if key not in self.enrich_map:
                    self.enrich_map[key] = self.allocate_enrichment_id()
                enr[0, i] = self.enrich_map[key]

        # Level 2: Negative phantom element (enrichc_ini.F lines 513-531)
        for i in range(4):
            if signs[i] < 0:
                enr[1, i] = 0  # Active side: tied to standard node
            else:
                key = (elem_id, 1, i)
                if key not in self.enrich_map:
                    self.enrich_map[key] = self.allocate_enrichment_id()
                enr[1, i] = self.enrich_map[key]

        # Level 3: Third phantom element if itri != 0 (enrichc_ini.F lines 532-565)
        if cut.itri != 0:
            for i in range(4):
                key = (elem_id, 2, i)
                if key not in self.enrich_map:
                    self.enrich_map[key] = self.allocate_enrichment_id()
                enr[2, i] = self.enrich_map[key]

        cut.enr_ids = enr

        for node_id in global_conn:
            self.node_to_elem_cuts.setdefault(node_id, []).append(elem_id)

        return enr

    def synchronize_shared_edges(
        self,
        elem_cuts: Dict[int, XfemElementCut],
        elem_conn: Dict[int, Sequence[int]],
    ) -> None:
        """Synchronize enriched DOFs between adjacent elements sharing a cut edge.

        Ported from OpenRadioss subroutine UPENR_CRK:
          ``$OR_SRC/engine/source/elements/xfem/upenr_crk.F``
          lines 56-86.
        Ensures displacement continuity along crack faces between neighboring cut elements.
        """
        # Group element edges: map frozenset((node1, node2)) -> list of (elem_id, local_edge_idx)
        # Node-based synchronization (matching upenr_crk.F lines 56-86):
        # Map (global_node, phantom_level) -> list of (elem_id, local_node_idx)
        node_dofs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for elem_id, cut in elem_cuts.items():
            if not cut.is_cut:
                continue
            conn = elem_conn[elem_id]
            for lev in range(cut.enr_ids.shape[0]):
                for local_i in range(4):
                    dof = cut.enr_ids[lev, local_i]
                    if dof > 0:
                        global_node = conn[local_i]
                        node_dofs.setdefault((global_node, lev), []).append((elem_id, local_i))

        # Unify DOFs sharing the same global node and phantom level
        for (global_node, lev), occurrences in node_dofs.items():
            if len(occurrences) > 1:
                min_dof = min(elem_cuts[e].enr_ids[lev, loc_i] for e, loc_i in occurrences)
                for e, loc_i in occurrences:
                    elem_cuts[e].enr_ids[lev, loc_i] = min_dof


# ---------------------------------------------------------------------------
# 6. Crack Front & Propagation Controller
# ---------------------------------------------------------------------------

@dataclass
class CrackFront:
    """Discrete crack front path and tip representation."""
    id: int = 1
    points: List[np.ndarray] = field(default_factory=list)
    is_active: bool = True
    tip_coords: np.ndarray = field(default_factory=lambda: np.zeros(2))
    propagation_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    cut_elements: List[int] = field(default_factory=list)

    @property
    def length(self) -> float:
        """Total current length of the crack."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.points) - 1):
            total += float(np.linalg.norm(self.points[i + 1] - self.points[i]))
        return total


def advance_crack_front_step(
    crack: CrackFront,
    tip_stress: Sequence[float] | np.ndarray,
    step_length: float,
    k_ic: float,
    width: Optional[float] = None,
    sigma_crit: Optional[float] = None,
) -> Tuple[bool, Optional[np.ndarray], float]:
    """Advance crack tip by one step if fracture criterion is satisfied.

    Parameters
    ----------
    crack : CrackFront
        Active crack front object.
    tip_stress : sequence of floats
        Stress tensor at the current crack tip [sig_xx, sig_yy, sig_xy].
    step_length : float
        Advancement increment dl (typically element characteristic length).
    k_ic : float
        Fracture toughness K_IC.
    width : float, optional
        Plate width for geometry correction.
    sigma_crit : float, optional
        Critical tensile strength.

    Returns
    -------
    advanced : bool
        True if crack advanced.
    new_tip : ndarray of shape (2,) or None
        New crack tip coordinates if advanced, else None.
    k1 : float
        Evaluated stress intensity factor K_I.
    """
    current_a = max(crack.length, 1.0e-5)
    should_prop, k1, d_crack, _ = check_crack_propagation(
        stress=tip_stress,
        k_ic=k_ic,
        crack_length=current_a,
        width=width,
        sigma_crit=sigma_crit,
    )

    if not should_prop or not crack.is_active:
        return False, None, k1

    # Ensure advancement direction projects forward along existing crack path
    if np.linalg.norm(crack.propagation_dir) > 1.0e-6:
        if np.dot(d_crack, crack.propagation_dir) < 0.0:
            d_crack = -d_crack

    # Advance tip along Mode I crack propagation direction
    new_tip = crack.tip_coords + step_length * d_crack
    crack.points.append(new_tip.copy())
    crack.tip_coords = new_tip.copy()
    crack.propagation_dir = d_crack.copy()

    return True, new_tip, k1
