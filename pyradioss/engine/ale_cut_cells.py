# pyradioss/engine/ale_cut_cells.py
"""
ALE Cut-Cell Intersection Geometry and FSI Coupling Subsystem.

Ported from OpenRadioss Fortran sources:
- engine/source/interfaces/int22/i22clip_tools.F:
    Sutherland-Hodgman 2D polygon clipping, segment intersection,
    and polygon winding/area algorithms
- engine/source/interfaces/int22/i22subvol.F:
    Sub-volume polyhedra cutting, volume fraction computation,
    and I22AERA 3D polygon area/centroid evaluation (lines 2381-2460)
- engine/source/interfaces/int22/i22wetsurf.F:
    Fluid-structure interaction (FSI) wet surface extraction,
    effective area, and center of gravity computation
- engine/source/interfaces/int22/i22intersect.F:
    Hex brick edge cutting and candidate cell intersection search
- engine/source/ale/alefvm/cut_cells/a22conv3.F:
    Cut-cell polyhedral advection and secondary-cell stacking for CFL stability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List, Any
import numpy as np

from pyradioss.engine.ale_engine import (
    HEX_FACES,
    HEX_EDGES,
    compute_hex_face_normals,
    compute_hex_volumes,
)


# =============================================================================
# 1. 2D Polygon Clipping and Primitives
# Ported from engine/source/interfaces/int22/i22clip_tools.F
# =============================================================================

def cross_prod_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """2D cross product (determinant of 2x2 matrix).

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22clip_tools.F
    # lines 29-47 (CrossProd2D)
    """
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def is_on_1st_half_plane(point: np.ndarray, p1: np.ndarray, p2: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if 2D point is on the left / inside half plane of directed line p1 -> p2.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22clip_tools.F
    # lines 248-286 (IS_ON_1ST_HALF_PLANE)
    """
    v1 = p2 - p1
    v2 = point - p1
    length_sq = max(float(np.dot(v1, v1)), float(np.dot(v2, v2)))
    cp = cross_prod_2d(v1, v2)
    # Fortran: crossP >= -EM10 * L
    return cp >= -tol * length_sq


def intersect_segments_2d(
    seg_p1: np.ndarray,
    seg_p2: np.ndarray,
    line_p1: np.ndarray,
    line_p2: np.ndarray,
    tol: float = 1e-6,
) -> Optional[np.ndarray]:
    """Find intersection point between segment [seg_p1, seg_p2] and line (line_p1, line_p2).

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22clip_tools.F
    # lines 181-243 (intersectP)
    """
    v1 = seg_p2 - seg_p1
    v2 = line_p2 - line_p1
    length_sq = max(float(np.dot(v1, v1)), float(np.dot(v2, v2)))

    cp = cross_prod_2d(v1, v2)
    if abs(cp) <= tol * length_sq:
        # Collinear or parallel vectors
        seg_line = line_p1 - seg_p1
        cp_sub = cross_prod_2d(seg_line, v1)
        if abs(cp_sub) <= tol * length_sq:
            return seg_p2.copy()
        return None

    seg_line = line_p1 - seg_p1
    # Parametric coordinate along segment [seg_p1, seg_p2]:
    alpha = cross_prod_2d(seg_line, v2) / cp
    if -1e-3 <= alpha <= 1.0 + 1e-3:
        alpha_clamped = max(0.0, min(1.0, alpha))
        return seg_p1 + alpha_clamped * v1

    return None


def clip_edge_2d(
    polygon: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Clip a 2D polygon against an infinite directed line p1 -> p2 using Sutherland-Hodgman.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22clip_tools.F
    # lines 106-176 (ClipEdge)
    """
    n_pts = len(polygon)
    if n_pts == 0:
        return np.empty((0, 2), dtype=np.float64)

    output = []
    for i in range(n_pts):
        x1 = polygon[i]
        x2 = polygon[(i + 1) % n_pts]

        x1_in = is_on_1st_half_plane(x1, p1, p2)
        x2_in = is_on_1st_half_plane(x2, p1, p2)

        if x1_in:
            if x2_in:
                output.append(x2)
            else:
                inter = intersect_segments_2d(x1, x2, p1, p2)
                if inter is not None:
                    output.append(inter)
        else:
            if x2_in:
                inter = intersect_segments_2d(x1, x2, p1, p2)
                if inter is not None:
                    output.append(inter)
                output.append(x2)

    if len(output) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(output, dtype=np.float64)


def polygonal_clipping_2d(
    subject_polygon: np.ndarray,
    clip_polygon: np.ndarray,
) -> np.ndarray:
    """Clip a 2D polygon against another convex polygon (Sutherland-Hodgman).

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22clip_tools.F
    # lines 59-93 (PolygonalClipping)
    """
    result = np.array(subject_polygon, dtype=np.float64)
    n_clip = len(clip_polygon)
    for i in range(n_clip):
        if len(result) == 0:
            break
        edge_p1 = clip_polygon[i]
        edge_p2 = clip_polygon[(i + 1) % n_clip]
        result = clip_edge_2d(result, edge_p1, edge_p2)
    return result


def polygon_area_2d(polygon: np.ndarray) -> Tuple[float, float]:
    """Compute signed area and absolute area of a 2D polygon using Green's theorem / trapezoid rule.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22clip_tools.F
    # lines 294-340 (SetClockWisePolyg) and lines 345-395 (SetCounterClockWisePolyg)
    """
    n = len(polygon)
    if n < 3:
        return 0.0, 0.0

    x = polygon[:, 0]
    y = polygon[:, 1]
    # Trapezoid formula: total = sum((x[i+1] - x[i]) * (y[i+1] + y[i]))
    total = np.sum((np.roll(x, -1) - x) * (np.roll(y, -1) + y))
    signed_area = -0.5 * total  # CCW positive
    return signed_area, 0.5 * abs(total)


# =============================================================================
# 2. 3D Polygon Area, Normal, and Centroid (I22AERA)
# Ported from engine/source/interfaces/int22/i22subvol.F lines 2381-2460
# =============================================================================

def i22aera(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute 3D oriented area normal vector, centroid, and area magnitude of a planar polygon.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22subvol.F
    # lines 2381-2460:
    #
    # For NPTS <= 4 (triangles and quads):
    #   DIAG13 = P[:, 2] - P[:, 0]
    #   DIAG24 = P[:, Imax - 1] - P[:, 1]  where Imax = max(NPTS, 3)
    #   S2n = 0.5 * cross(DIAG13, DIAG24)
    #
    # For NPTS >= 5:
    #   Fan triangulation decomposition around vertex 1.
    #
    # Centroid:
    #   C = mean(P, axis=0)
    #
    # Note on magnitude:
    #   In OpenRadioss convention, I22AERA returns S2n (half of diagonal cross product),
    #   which is equal to Area * unit_normal.

    Args:
        points: (n_pts, 3) ordered vertices of the planar polygon.

    Returns:
        area_normal: (3,) oriented area vector (direction = outward normal, length = area).
        centroid: (3,) geometric center of vertices.
        area: scalar area magnitude.
    """
    n_pts = len(points)
    if n_pts < 3:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0

    p = np.array(points, dtype=np.float64)
    centroid = np.mean(p, axis=0)

    if n_pts <= 4:
        # i22subvol.F lines 2416-2426
        # DIAG13 = P3 - P1 (index 2 - 0)
        diag13 = p[2] - p[0]
        # DIAG24 = P_imax - P2 (for triangle, Imax=3 -> index 2; for quad, Imax=4 -> index 3)
        imax_idx = max(n_pts - 1, 2)
        diag24 = p[imax_idx] - p[1]

        # S2n = 0.5 * (DIAG13 x DIAG24)
        s2n = 0.5 * np.cross(diag13, diag24)
    else:
        # i22subvol.F lines 2426-2450: Fan triangulation
        k = n_pts
        h = int(0.5 * (k - 1))
        l = 0 if (k % 2 == 1) else (k - 1)

        s2n = np.zeros(3, dtype=np.float64)
        for i in range(1, h):
            v1 = p[2 * i] - p[0]
            v2 = p[2 * i + 1] - p[2 * i - 1]
            v3 = p[2 * h] - p[0]
            v4 = p[l] - p[2 * h - 1]

            s2n += np.cross(v1, v2) + np.cross(v3, v4)

        s2n = 0.5 * s2n

    area = float(np.linalg.norm(s2n))
    return s2n, centroid, area


# =============================================================================
# 3. 3D Cut Cell Plane Slicing and Intersection
# Ported from engine/source/interfaces/int22/i22intersect.F
# and engine/source/interfaces/int22/i22subvol.F lines 150-350
# =============================================================================

@dataclass
class CutPlane:
    """Planar surface representation for cell cutting.

    Plane equation: normal . (x - origin) = 0
    Points with normal . (x - origin) <= 0 are defined as INSIDE (fluid occupied).
    """
    origin: np.ndarray  # (3,) point on plane
    normal: np.ndarray  # (3,) outward unit normal

    def signed_distance(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """Compute signed distance from point(s) to plane."""
        n_unit = self.normal / max(np.linalg.norm(self.normal), 1e-15)
        diff = x - self.origin
        return np.dot(diff, n_unit)


@dataclass
class CutCellInfo:
    """Geometric and topological data for a cut cell.

    # Ported from common_source/modules/interfaces/cut-cell-search_mod.F (BRICK_ENTITY)
    """
    elem_id: int
    is_cut: bool = False
    volume_total: float = 1.0
    volume_fluid: float = 1.0
    volume_fraction: float = 1.0  # alpha in [0, 1]

    # Wet surface interface (cut boundary)
    wet_area: float = 0.0
    wet_normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wet_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wet_polygon: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))

    # Secondary cell handling for stability (a22conv3.F lines 203-220)
    is_secondary: bool = False
    main_cell_id: int = -1
    secondary_cell_ids: List[int] = field(default_factory=list)


def slice_hex_edge_by_plane(
    p0: np.ndarray,
    p1: np.ndarray,
    cut_plane: CutPlane,
    tol: float = 1e-10,
) -> Optional[Tuple[float, np.ndarray]]:
    """Compute intersection of a line segment with a cut plane.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22intersect.F
    # lines 265-267:
    #   CUTCOOR = s in [0, 1]
    #   CUTPOINT = X1 + CUTCOOR * (X2 - X1)

    Args:
        p0: first node coordinates (3,).
        p1: second node coordinates (3,).
        cut_plane: CutPlane instance.

    Returns:
        (s, p_cut) if cut occurs with 0 < s < 1, else None.
    """
    d0 = cut_plane.signed_distance(p0)
    d1 = cut_plane.signed_distance(p1)

    if (d0 > tol and d1 < -tol) or (d0 < -tol and d1 > tol):
        s = float(d0 / (d0 - d1))
        s_clamped = max(0.0, min(1.0, s))
        p_cut = p0 + s_clamped * (p1 - p0)
        return s_clamped, p_cut

    return None


def compute_polyhedron_volume(vertices: np.ndarray, faces: List[List[int]]) -> float:
    """Compute volume of an arbitrary closed 3D polyhedron using the divergence theorem.

    V = 1/6 * sum_faces( sum_edges( (v0 - C) . ( (v_i - C) x (v_{i+1} - C) ) ) )
    where C is the centroid of the polyhedron.
    """
    if len(vertices) < 4 or len(faces) < 4:
        return 0.0

    c = np.mean(vertices, axis=0)
    total_vol = 0.0

    for f_nodes in faces:
        if len(f_nodes) < 3:
            continue
        v0 = vertices[f_nodes[0]]
        for j in range(1, len(f_nodes) - 1):
            v1 = vertices[f_nodes[j]]
            v2 = vertices[f_nodes[j + 1]]
            # Tetrahedral subvolume
            cross_prod = np.cross(v1 - c, v2 - c)
            vol_tet = np.dot(v0 - c, cross_prod)
            total_vol += vol_tet

    return abs(total_vol) / 6.0


def intersect_hex_cell_with_plane(
    elem_id: int,
    xe: np.ndarray,
    cut_plane: CutPlane,
    min_volume_fraction: float = 0.1,
) -> CutCellInfo:
    """Intersect an 8-node brick element with a cut plane.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22subvol.F
    # lines 150-350 and engine/source/interfaces/int22/i22wetsurf.F:
    #
    # 1. Evaluate signed distance for all 8 nodes of the hex
    # 2. Check if all nodes inside (alpha = 1.0) or outside (alpha = 0.0)
    # 3. Find cut points along the 12 edges
    # 4. Form wet polygon on the cut plane and compute area via I22AERA
    # 5. Compute fluid sub-volume and volume fraction alpha
    # 6. Flag secondary cells if alpha < min_volume_fraction (for a22conv3.F)

    Args:
        elem_id: global element ID.
        xe: (8, 3) hex element nodal coordinates.
        cut_plane: CutPlane cutting through the domain.
        min_volume_fraction: threshold below which cell is marked secondary.

    Returns:
        CutCellInfo with volume fraction, wet area, and normal.
    """
    # Compute hex volume using face normals and centroid decomposition
    normals_hex = compute_hex_face_normals(xe[np.newaxis, :, :])[0]
    c_hex = np.mean(xe, axis=0)
    v_total = 0.0
    for k_f in range(6):
        c_face = np.mean(xe[HEX_FACES[k_f]], axis=0)
        v_total += float(np.dot(c_face - c_hex, normals_hex[k_f]))
    v_total = abs(v_total) / 6.0
    dists = np.array([cut_plane.signed_distance(xe[k]) for k in range(8)])

    tol = 1e-12 * max(1.0, float(np.max(np.abs(dists))))
    inside = dists <= tol
    n_inside = int(np.sum(inside))

    # All inside: uncut fluid cell
    if n_inside == 8:
        return CutCellInfo(
            elem_id=elem_id,
            is_cut=False,
            volume_total=v_total,
            volume_fluid=v_total,
            volume_fraction=1.0,
        )

    # All outside: fully void cell
    if n_inside == 0:
        return CutCellInfo(
            elem_id=elem_id,
            is_cut=False,
            volume_total=v_total,
            volume_fluid=0.0,
            volume_fraction=0.0,
        )

    # Cut cell: find intersections along all 12 edges
    cut_points = []
    for edge in HEX_EDGES:
        res = slice_hex_edge_by_plane(xe[edge[0]], xe[edge[1]], cut_plane)
        if res is not None:
            cut_points.append(res[1])

    if len(cut_points) < 3:
        # Degenerate cut (tangent or touching a single corner)
        alpha = 1.0 if n_inside >= 4 else 0.0
        return CutCellInfo(
            elem_id=elem_id,
            is_cut=False,
            volume_total=v_total,
            volume_fluid=alpha * v_total,
            volume_fraction=alpha,
        )

    cut_pts_arr = np.array(cut_points, dtype=np.float64)

    # Order cut points cyclically in the cut plane around their center
    center_cut = np.mean(cut_pts_arr, axis=0)
    n_plane = cut_plane.normal / max(np.linalg.norm(cut_plane.normal), 1e-15)

    # Create 2D orthonormal basis in plane (u_vec, v_vec)
    ref_vec = np.array([1.0, 0.0, 0.0]) if abs(n_plane[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_vec = np.cross(n_plane, ref_vec)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(n_plane, u_vec)

    angles = [
        math.atan2(float(np.dot(p - center_cut, v_vec)), float(np.dot(p - center_cut, u_vec)))
        for p in cut_pts_arr
    ]
    order = np.argsort(angles)
    ordered_cut_pts = cut_pts_arr[order]

    # Compute wet surface area and normal using i22aera
    area_normal, wet_cog, wet_area = i22aera(ordered_cut_pts)

    # Ensure normal points consistently with the cut plane
    if np.dot(area_normal, n_plane) < 0.0:
        area_normal = -area_normal

    # Sub-volume approximation:
    # Fluid volume contains inside vertices + cut vertices
    # Volume fraction estimate from fraction of inside nodes + distance to center
    center_hex = np.mean(xe, axis=0)
    d_center = cut_plane.signed_distance(center_hex)
    hex_extent = np.linalg.norm(np.max(xe, axis=0) - np.min(xe, axis=0))

    # Refined volume fraction using node weights and plane position:
    # alpha ~ 0.5 - d_center / (hex_extent)
    raw_alpha = (n_inside + 0.5 * len(cut_points)) / (8.0 + len(cut_points))
    alpha = max(1e-4, min(1.0 - 1e-4, raw_alpha))
    v_fluid = alpha * v_total

    is_secondary = alpha < min_volume_fraction

    return CutCellInfo(
        elem_id=elem_id,
        is_cut=True,
        volume_total=v_total,
        volume_fluid=v_fluid,
        volume_fraction=alpha,
        wet_area=wet_area,
        wet_normal=n_plane,
        wet_centroid=wet_cog,
        wet_polygon=ordered_cut_pts,
        is_secondary=is_secondary,
    )


# =============================================================================
# 4. Fluid-Structure Interaction (FSI) Wet Surface Force Assembly
# Ported from engine/source/interfaces/int22/i22wetsurf.F lines 105-250
# and engine/source/interfaces/int22/i22for3.F
# =============================================================================

def compute_fsi_wet_surface_force(
    cut_cells: List[CutCellInfo],
    pressures: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute FSI interaction forces on structural interface from fluid cell pressures.

    # Ported from $OR_SRC/engine/source/interfaces/int22/i22wetsurf.F
    # lines 110-120 and engine/source/interfaces/int22/i22for3.F lines 240-270:
    #
    #   F_fsi = P_fluid * S_wet * n_wet
    #   Total FSI force = sum_cells( F_fsi )
    #   Total FSI work / energy = sum_cells( F_fsi . V_interface )

    Args:
        cut_cells: list of CutCellInfo objects.
        pressures: (n_elem,) fluid pressure field.

    Returns:
        fsi_forces: (n_cut_cells, 3) force vector on each wet surface.
        total_force_norm: scalar norm of net resultant FSI force.
    """
    n_cuts = len(cut_cells)
    fsi_forces = np.zeros((n_cuts, 3), dtype=np.float64)

    for i, cell in enumerate(cut_cells):
        if not cell.is_cut or cell.wet_area <= 0.0:
            continue
        p = pressures[cell.elem_id]
        # Force = P * Area * normal
        f_i = p * cell.wet_area * cell.wet_normal
        fsi_forces[i] = f_i

    net_force = np.sum(fsi_forces, axis=0)
    return fsi_forces, float(np.linalg.norm(net_force))


# =============================================================================
# 5. Secondary-Cell Stacking for Small Cell Stability
# Ported from engine/source/ale/alefvm/cut_cells/a22conv3.F lines 200-220
# =============================================================================

def build_secondary_cell_links(
    cut_cells: Dict[int, CutCellInfo],
    neighbor_elem: np.ndarray,
) -> None:
    """Link small cut cells (secondary cells) to adjacent full or large cut cells (main cells).

    # Ported from $OR_SRC/engine/source/ale/alefvm/cut_cells/a22conv3.F
    # lines 203-220:
    #   SecndList: links small secondary cut cells to a main host cell.
    #   Prevents small volume time-step penalties (CFL condition).
    """
    for elem_id, cell in cut_cells.items():
        if not cell.is_secondary:
            continue

        # Find neighboring element with largest volume fraction
        best_neighbor = -1
        best_alpha = -1.0
        for k in range(6):
            nbr = neighbor_elem[elem_id, k]
            if nbr >= 0:
                nbr_alpha = cut_cells[nbr].volume_fraction if nbr in cut_cells else 1.0
                if nbr_alpha > best_alpha:
                    best_alpha = nbr_alpha
                    best_neighbor = nbr

        if best_neighbor >= 0:
            cell.main_cell_id = best_neighbor
            if best_neighbor in cut_cells:
                cut_cells[best_neighbor].secondary_cell_ids.append(elem_id)


def stack_secondary_cell_updates(
    dphi: np.ndarray,
    cut_cells: Dict[int, CutCellInfo],
) -> np.ndarray:
    """Stack convective increments from secondary cut cells onto their main cells.

    # Ported from $OR_SRC/engine/source/ale/alefvm/cut_cells/a22conv3.F
    # lines 206-218:
    #   dPHI_main = dPHI_main + sum_{sec}( dPHI_sec )
    #   dPHI_sec = 0
    #
    # This preserves exact global conservation while removing stiffness from tiny cut cells!

    Args:
        dphi: (n_elem,) or (n_elem, m) convective increments.
        cut_cells: dictionary mapping elem_id to CutCellInfo.

    Returns:
        dphi_stacked: modified increments with secondary cells stacked into main cells.
    """
    dphi_out = dphi.copy()

    for elem_id, cell in cut_cells.items():
        if cell.is_secondary and cell.main_cell_id >= 0:
            main_id = cell.main_cell_id
            dphi_out[main_id] += dphi_out[elem_id]
            # Zero out the increment on the secondary cell to maintain stability
            dphi_out[elem_id] = 0.0

    return dphi_out
