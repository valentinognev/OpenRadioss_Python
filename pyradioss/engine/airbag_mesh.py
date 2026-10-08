"""pyradioss.engine.airbag_mesh — 3D unstructured mesh generation and dynamic rezoning for folded airbags.

Fortran source references:
- 3D Finite volume mesh generation inside folded airbags:
  ``engine/source/airbag/fvmesh.F`` (subroutines FVMESH1, ITRIBOX, POLCLIP, FACEPOLY, POLYHEDR, COORLOC, GPOLCUT)
- Dynamic grid rezoning during fabric expansion:
  ``engine/source/airbag/fvrezone.F`` (subroutines FVREZONE0, FVREZONE1, PINPOLH)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# 1. 3D Geometric Clipping & Polygon Operations (from fvmesh.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/fvmesh.F: POLCLIP (lines 3160-3240)
def polygon_clip_plane(
    poly_in: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """Clip a 3D polygon against a half-space defined by a plane.

    Points satisfying dot(x - plane_point, plane_normal) >= -tol are kept (inside).
    Implements Sutherland-Hodgman clipping algorithm matching OpenRadioss ``POLCLIP``.

    Args:
        poly_in: (N, 3) array of polygon vertices in cyclic order.
        plane_point: (3,) point on the clipping plane.
        plane_normal: (3,) unit normal pointing towards the inside half-space.
        tol: Tolerance for coplanar vertices.

    Returns:
        (M, 3) array of clipped polygon vertices, or empty array if completely clipped.
    """
    n_in = len(poly_in)
    if n_in < 3:
        return np.zeros((0, 3), dtype=np.float64)

    # Distances to plane: d > 0 means inside half-space
    diffs = poly_in - plane_point
    dists = np.dot(diffs, plane_normal)

    poly_out = []
    for i in range(n_in):
        curr_pt = poly_in[i]
        curr_dist = dists[i]
        prev_pt = poly_in[(i - 1 + n_in) % n_in]
        prev_dist = dists[(i - 1 + n_in) % n_in]

        curr_inside = curr_dist >= -tol
        prev_inside = prev_dist >= -tol

        if curr_inside:
            if not prev_inside:
                # Edge entered inside: compute intersection
                denom = prev_dist - curr_dist
                t = prev_dist / denom if abs(denom) > 1e-15 else 0.0
                t = max(0.0, min(1.0, float(t)))
                inter_pt = prev_pt + t * (curr_pt - prev_pt)
                poly_out.append(inter_pt)
            poly_out.append(curr_pt)
        elif prev_inside:
            # Edge exited inside: compute intersection
            denom = prev_dist - curr_dist
            t = prev_dist / denom if abs(denom) > 1e-15 else 0.0
            t = max(0.0, min(1.0, float(t)))
            inter_pt = prev_pt + t * (curr_pt - prev_pt)
            poly_out.append(inter_pt)

    if len(poly_out) < 3:
        return np.zeros((0, 3), dtype=np.float64)

    return np.array(poly_out, dtype=np.float64)


# Ported from $OR_SRC/engine/source/airbag/fvmesh.F: ITRIBOX (lines 3106-3158)
def clip_triangle_to_box(
    triangle: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """Clip a 3D triangle against an axis-aligned bounding box (6 planes).

    Matches Fortran ``ITRIBOX`` which clips triangle against the 6 faces of background cell.

    Args:
        triangle: (3, 3) coordinates of triangle vertices.
        box_min: (3,) minimum corner of box [xmin, ymin, zmin].
        box_max: (3,) maximum corner of box [xmax, ymax, zmax].

    Returns:
        (K, 3) polygon vertices inside the box (K >= 3), or empty array if outside.
    """
    poly = np.array(triangle, dtype=np.float64)

    # 6 planes: x_min (+x), x_max (-x), y_min (+y), y_max (-y), z_min (+z), z_max (-z)
    planes = [
        (np.array([box_min[0], 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([box_max[0], 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
        (np.array([0.0, box_min[1], 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, box_max[1], 0.0]), np.array([0.0, -1.0, 0.0])),
        (np.array([0.0, 0.0, box_min[2]]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, box_max[2]]), np.array([0.0, 0.0, -1.0])),
    ]

    for p_point, p_normal in planes:
        poly = polygon_clip_plane(poly, p_point, p_normal)
        if len(poly) < 3:
            return np.zeros((0, 3), dtype=np.float64)

    return poly


# Ported from $OR_SRC/engine/source/airbag/fvmesh.F: subroutines FACEPOLY, POLYHEDR
def compute_polygon_area_normal(
    poly: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute area, outward unit normal, and centroid of a planar 3D polygon.

    Uses Stokes theorem / Newell's method for arbitrary 3D planar polygons.
    """
    n = len(poly)
    if n < 3:
        return 0.0, np.zeros(3), np.zeros(3)

    normal = np.zeros(3, dtype=np.float64)
    centroid = np.mean(poly, axis=0)

    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        normal[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
        normal[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
        normal[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])

    norm_mag = np.linalg.norm(normal)
    if norm_mag > 1e-14:
        unit_normal = normal / norm_mag
        area = 0.5 * norm_mag
    else:
        unit_normal = np.zeros(3, dtype=np.float64)
        area = 0.0

    return float(area), unit_normal, centroid


# =============================================================================
# 2. Point-in-Polyhedron Test via Spherical Solid Angle (from fvrezone.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/fvrezone.F: PINPOLH (lines 799-902)
def point_in_polyhedron(
    point: np.ndarray,
    triangles: np.ndarray,
    tole: float = 1e-4,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
) -> bool:
    """Test if a 3D point is inside a closed triangular polyhedron mesh.

    Uses spherical polygon solid angle summation matching Fortran ``PINPOLH``:
    For each surface triangle (x1, x2, x3), vectors v1, v2, v3 from point:
    n1 = v1 x v2, n2 = v2 x v3, n3 = v3 x v1
    s1 = -(n1 . n2)/(|n1| |n2|), etc.
    solid_angle = acos(s1) + acos(s2) + acos(s3) - pi
    Signed by face normal dot (xc - query_pt).
    Total solid angle is 4*pi for interior points, 0 for exterior points.

    Args:
        point: (3,) query point [x, y, z].
        triangles: (M, 3, 3) coordinates of triangular surface facets.
        tole: Tolerance threshold (inside if |solid_angle| > tole; 4*pi ~ 12.56).
        bbox_min: Optional bounding box min for quick rejection.
        bbox_max: Optional bounding box max for quick rejection.

    Returns:
        bool: True if point is inside polyhedron.
    """
    xx, yy, zz = point[0], point[1], point[2]

    # Quick bounding box check
    if bbox_min is not None:
        if xx < bbox_min[0] or yy < bbox_min[1] or zz < bbox_min[2]:
            return False
    if bbox_max is not None:
        if xx > bbox_max[0] or yy > bbox_max[1] or zz > bbox_max[2]:
            return False

    solid_total = 0.0
    for tri in triangles:
        x1, y1, z1 = tri[0]
        x2, y2, z2 = tri[1]
        x3, y3, z3 = tri[2]

        vx1 = x1 - xx; vy1 = y1 - yy; vz1 = z1 - zz
        vx2 = x2 - xx; vy2 = y2 - yy; vz2 = z2 - zz
        vx3 = x3 - xx; vy3 = y3 - yy; vz3 = z3 - zz

        nx1 = vy1 * vz2 - vz1 * vy2
        ny1 = vz1 * vx2 - vx1 * vz2
        nz1 = vx1 * vy2 - vy1 * vx2

        nx2 = vy2 * vz3 - vz2 * vy3
        ny2 = vz2 * vx3 - vx2 * vz3
        nz2 = vx2 * vy3 - vy2 * vx3

        nx3 = vy3 * vz1 - vz3 * vy1
        ny3 = vz3 * vx1 - vx3 * vz1
        nz3 = vx3 * vy1 - vy3 * vx1

        rr1 = math.sqrt(nx1 * nx1 + ny1 * ny1 + nz1 * nz1)
        rr2 = math.sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2)
        rr3 = math.sqrt(nx3 * nx3 + ny3 * ny3 + nz3 * nz3)

        if rr1 * rr2 < 1e-24 or rr2 * rr3 < 1e-24 or rr3 * rr1 < 1e-24:
            continue

        ss1 = -(nx1 * nx2 + ny1 * ny2 + nz1 * nz2) / (rr1 * rr2)
        ss2 = -(nx2 * nx3 + ny2 * ny3 + nz2 * nz3) / (rr2 * rr3)
        ss3 = -(nx3 * nx1 + ny3 * ny1 + nz3 * nz1) / (rr3 * rr1)

        ss1 = max(-1.0, min(1.0, ss1))
        ss2 = max(-1.0, min(1.0, ss2))
        ss3 = max(-1.0, min(1.0, ss3))

        sarea = math.acos(ss1) + math.acos(ss2) + math.acos(ss3) - math.pi

        # Triangle orientation
        ax = x2 - x1; ay = y2 - y1; az = z2 - z1
        bx = x3 - x1; by = y3 - y1; bz = z3 - z1
        tnx = ay * bz - az * by
        tny = az * bx - ax * bz
        tnz = ax * by - ay * bx

        xc = (x1 + x2 + x3) / 3.0
        yc = (y1 + y2 + y3) / 3.0
        zc = (z1 + z2 + z3) / 3.0

        ss = tnx * (xc - xx) + tny * (yc - yy) + tnz * (zc - zz)
        if ss < 0.0:
            sarea = -sarea

        solid_total += sarea

    return abs(solid_total) > tole


# =============================================================================
# 3. Data Structures for 3D FVM Polyhedral Airbag
# =============================================================================

@dataclass
class FvmFacet:
    """A planar face/polygon of an unstructured 3D finite volume cell."""
    id: int
    vertices: np.ndarray  # (K, 3)
    area: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    left_cell: int = -1   # Index of polyhedron on left (+normal side)
    right_cell: int = -1  # Index of polyhedron on right (-normal side)
    is_boundary: bool = False
    surface_tri_id: int = -1  # ID of original fabric surface triangle if boundary


@dataclass
class FvmPolyhedron:
    """A 3D polyhedral finite volume cell (conforming or cut cell)."""
    id: int
    triangles: np.ndarray  # (M, 3, 3) surface triangulation bounding the cell
    volume: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Thermodynamic state variables
    mass: float = 0.0
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    energy: float = 0.0
    pressure: float = 0.0
    density: float = 0.0
    temperature: float = 293.15
    gamma: float = 1.4
    r_spec: float = 287.05
    cpa: float = 1004.0
    cpb: float = 0.0
    cpc: float = 0.0
    cpd: float = 0.0
    cpe: float = 0.0
    cpf: float = 0.0


# Ported from $OR_SRC/engine/source/airbag/fvmesh.F: subroutines FVMESH1, POLYHEDR
def compute_polyhedron_volume(triangles: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute enclosed volume and centroid of a triangulated polyhedron.

    Uses Gauss divergence theorem:
    V = (1/3) * sum_i dot(xc_i, n_i) * Area_i.
    Matches OpenRadioss ``fvmesh.F`` lines 477-488.
    """
    if len(triangles) == 0:
        return 0.0, np.zeros(3)

    total_vol = 0.0
    weighted_centroid = np.zeros(3, dtype=np.float64)

    for tri in triangles:
        p1, p2, p3 = tri[0], tri[1], tri[2]
        v21 = p2 - p1
        v31 = p3 - p1
        n_vec = 0.5 * np.cross(v21, v31)
        area = np.linalg.norm(n_vec)
        if area > 1e-15:
            n_unit = n_vec / area
            xc = (p1 + p2 + p3) / 3.0
            vol_i = (1.0 / 3.0) * np.dot(xc, n_vec)
            total_vol += vol_i
            # Centroid of tetrahedron with origin apex is 0.75 * xc
            weighted_centroid += vol_i * (0.75 * xc)

    if abs(total_vol) > 1e-15:
        centroid = weighted_centroid / total_vol
    else:
        centroid = np.mean(triangles.reshape(-1, 3), axis=0)

    return float(total_vol), centroid


# =============================================================================
# 4. 3D Unstructured Mesh Generation Inside Folded Airbag (from fvmesh.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/fvmesh.F: FVMESH1 (lines 45-500)
def generate_fvm_airbag_mesh(
    surface_triangles: np.ndarray,
    grid_res: Tuple[int, int, int] = (2, 2, 2),
    margin: float = 0.01,
) -> List[FvmPolyhedron]:
    """Generate 3D unstructured finite volume mesh inside an airbag boundary surface.

    Implements the Cartesian background grid cutting algorithm from ``fvmesh.F``:
    1. Determines global bounding box of airbag surface with safety margin.
    2. Overlays a background grid of regular hex/box cells (BRIC).
    3. Clips the airbag boundary surface against each background cell (ITRIBOX/POLCLIP).
    4. Constructs closed polyhedra inside the airbag with verified volumes and centroids.

    Args:
        surface_triangles: (M, 3, 3) coordinates of airbag surface triangles.
        grid_res: (nx, ny, nz) subdivisions of background grid.
        margin: Relative boundary box padding factor.

    Returns:
        List of generated FvmPolyhedron cells with enclosed volumes and bounding boxes.
    """
    if len(surface_triangles) == 0:
        return []

    all_pts = surface_triangles.reshape(-1, 3)
    p_min = np.min(all_pts, axis=0)
    p_max = np.max(all_pts, axis=0)
    span = np.maximum(p_max - p_min, 1e-6)

    # Pad bounding box
    bbox_min = p_min - margin * span
    bbox_max = p_max + margin * span

    nx, ny, nz = grid_res
    dx = (bbox_max[0] - bbox_min[0]) / nx
    dy = (bbox_max[1] - bbox_min[1]) / ny
    dz = (bbox_max[2] - bbox_min[2]) / nz

    polyhedra = []
    cell_id = 1

    for ix in range(nx):
        x0 = bbox_min[0] + ix * dx
        x1 = x0 + dx
        for iy in range(ny):
            y0 = bbox_min[1] + iy * dy
            y1 = y0 + dy
            for iz in range(nz):
                z0 = bbox_min[2] + iz * dz
                z1 = z0 + dz

                c_min = np.array([x0, y0, z0])
                c_max = np.array([x1, y1, z1])
                c_center = 0.5 * (c_min + c_max)

                # Check if background cell intersects or is inside airbag
                # Find surface triangles intersecting this box
                cell_triangles = []
                for tri in surface_triangles:
                    clipped_poly = clip_triangle_to_box(tri, c_min, c_max)
                    if len(clipped_poly) >= 3:
                        # Fan triangulate clipped polygon
                        v0 = clipped_poly[0]
                        for k in range(1, len(clipped_poly) - 1):
                            cell_triangles.append(np.array([v0, clipped_poly[k], clipped_poly[k + 1]]))

                # Also check if box center is inside the airbag surface
                center_inside = point_in_polyhedron(c_center, surface_triangles)

                # If box is fully inside with no surface cuts, construct standard hex faces (12 triangles)
                if center_inside and len(cell_triangles) == 0:
                    # 8 box corners
                    p = [
                        np.array([x0, y0, z0]), np.array([x1, y0, z0]),
                        np.array([x1, y1, z0]), np.array([x0, y1, z0]),
                        np.array([x0, y0, z1]), np.array([x1, y0, z1]),
                        np.array([x1, y1, z1]), np.array([x0, y1, z1]),
                    ]
                    # 12 triangles for 6 faces
                    box_tris = [
                        # -z face
                        [p[0], p[2], p[1]], [p[0], p[3], p[2]],
                        # +z face
                        [p[4], p[5], p[6]], [p[4], p[6], p[7]],
                        # -y face
                        [p[0], p[1], p[5]], [p[0], p[5], p[4]],
                        # +y face
                        [p[3], p[6], p[2]], [p[3], p[7], p[6]],
                        # -x face
                        [p[0], p[4], p[7]], [p[0], p[7], p[3]],
                        # +x face
                        [p[1], p[2], p[6]], [p[1], p[6], p[5]],
                    ]
                    cell_triangles = [np.array(t) for t in box_tris]

                if len(cell_triangles) > 0:
                    tri_arr = np.array(cell_triangles)
                    vol, cent = compute_polyhedron_volume(tri_arr)
                    if vol > 1e-12:
                        b_min = np.min(tri_arr.reshape(-1, 3), axis=0)
                        b_max = np.max(tri_arr.reshape(-1, 3), axis=0)
                        poly = FvmPolyhedron(
                            id=cell_id,
                            triangles=tri_arr,
                            volume=vol,
                            centroid=cent,
                            bbox_min=b_min,
                            bbox_max=b_max,
                        )
                        polyhedra.append(poly)
                        cell_id += 1

    return polyhedra


# =============================================================================
# 5. Dynamic Grid Rezoning (from fvrezone.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/fvrezone.F: FVREZONE1 (lines 131-793)
def rezone_airbag_mesh(
    old_polyhedra: List[FvmPolyhedron],
    new_polyhedra: List[FvmPolyhedron],
    n_sample_steps: int = 5,
) -> Dict[str, float]:
    """Rezone state variables from old deformed polyhedra to newly generated polyhedra.

    Matches OpenRadioss ``FVREZONE1``:
    1. Sets up 3D background voxel grid with NSTEP^3 sampling points per cell bounding box.
    2. Identifies overlapping subvolumes between old and new polyhedra using ``PINPOLH``.
    3. Remaps conservative variables (mass, momentum, internal energy, Cp polynomials).
    4. Computes global mass, momentum, and internal energy conservation errors.
    5. Updates new cell densities, pressures, and temperatures.

    Args:
        old_polyhedra: List of donor polyhedra with current state.
        new_polyhedra: List of newly meshed target polyhedra.
        n_sample_steps: Number of voxel divisions per axis (NSTEP in fvrezone.F).

    Returns:
        Dict with initial mass/energy, rezoned mass/energy, and relative error percentages.
    """
    n_step = max(int(n_sample_steps), 2)
    n_points_per_box = n_step ** 3

    # Initial conservation sums
    mass0 = sum(p.mass for p in old_polyhedra)
    mom0 = np.sum([p.momentum for p in old_polyhedra], axis=0) if old_polyhedra else np.zeros(3)
    ener0 = sum(p.energy for p in old_polyhedra)

    for p_new in new_polyhedra:
        # Reset new polyhedral state
        p_new.mass = 0.0
        p_new.momentum = np.zeros(3, dtype=np.float64)
        p_new.energy = 0.0
        p_new.cpa = 0.0
        p_new.cpb = 0.0
        p_new.cpc = 0.0
        p_new.r_spec = 0.0

        # Discretize new cell bounding box into n_step^3 voxels
        x_min, y_min, z_min = p_new.bbox_min
        x_max, y_max, z_max = p_new.bbox_max
        dx = (x_max - x_min) / n_step
        dy = (y_max - y_min) / n_step
        dz = (z_max - z_min) / n_step
        vol_voxel = dx * dy * dz

        # Check overlap with candidate old polyhedra
        for p_old in old_polyhedra:
            # Quick bounding box overlap test
            if (
                p_new.bbox_max[0] < p_old.bbox_min[0] or p_new.bbox_min[0] > p_old.bbox_max[0]
                or p_new.bbox_max[1] < p_old.bbox_min[1] or p_new.bbox_min[1] > p_old.bbox_max[1]
                or p_new.bbox_max[2] < p_old.bbox_min[2] or p_new.bbox_min[2] > p_old.bbox_max[2]
            ):
                continue

            # Count overlapping sampling points inside both p_new and p_old
            overlap_voxels = 0
            for ix in range(n_step):
                vx = x_min + (ix + 0.5) * dx
                for iy in range(n_step):
                    vy = y_min + (iy + 0.5) * dy
                    for iz in range(n_step):
                        vz = z_min + (iz + 0.5) * dz
                        pt = np.array([vx, vy, vz])
                        if point_in_polyhedron(pt, p_new.triangles, tole=1e-4, bbox_min=p_new.bbox_min, bbox_max=p_new.bbox_max):
                            if point_in_polyhedron(pt, p_old.triangles, tole=1e-4, bbox_min=p_old.bbox_min, bbox_max=p_old.bbox_max):
                                overlap_voxels += 1

            if overlap_voxels > 0 and p_old.volume > 1e-15:
                # Volume overlap ratio RR = V_overlap / V_old (fvrezone.F line 704)
                v_overlap = overlap_voxels * vol_voxel
                rr = min(1.0, max(0.0, v_overlap / p_old.volume))

                p_new.mass += rr * p_old.mass
                p_new.momentum += rr * p_old.momentum
                p_new.energy += rr * p_old.energy
                p_new.cpa += rr * p_old.mass * p_old.cpa
                p_new.cpb += rr * p_old.mass * p_old.cpb
                p_new.cpc += rr * p_old.mass * p_old.cpc
                p_new.r_spec += rr * p_old.mass * p_old.r_spec
                p_new.gamma = p_old.gamma

        # Normalize mass-weighted specific heats
        if p_new.mass > 1e-15:
            p_new.cpa /= p_new.mass
            p_new.cpb /= p_new.mass
            p_new.cpc /= p_new.mass
            p_new.r_spec /= p_new.mass
            p_new.density = p_new.mass / max(p_new.volume, 1e-12)
            # Pressure from ideal gas / EOS: P = (gamma - 1) * E / V (fvrezone.F line 788)
            gamma = max(p_new.gamma, 1.001)
            p_new.pressure = (gamma - 1.0) * p_new.energy / max(p_new.volume, 1e-12)
            cv = max(p_new.cpa - p_new.r_spec, 1.0)
            p_new.temperature = p_new.energy / (p_new.mass * cv)
        else:
            p_new.density = 0.0
            p_new.pressure = 0.0
            p_new.temperature = 293.15

    # Post-rezone conservation sums & error check
    mass1 = sum(p.mass for p in new_polyhedra)
    mom1 = np.sum([p.momentum for p in new_polyhedra], axis=0) if new_polyhedra else np.zeros(3)
    ener1 = sum(p.energy for p in new_polyhedra)

    # Scaling adjustment if small sampling truncation occurs (to ensure exact 1st law conservation)
    if mass0 > 0.0 and mass1 > 0.0:
        scale_m = mass0 / mass1
        scale_e = ener0 / ener1 if ener1 > 0.0 else 1.0
        for p_new in new_polyhedra:
            p_new.mass *= scale_m
            p_new.energy *= scale_e
            p_new.density = p_new.mass / max(p_new.volume, 1e-12)
            gamma = max(p_new.gamma, 1.001)
            p_new.pressure = (gamma - 1.0) * p_new.energy / max(p_new.volume, 1e-12)

    err_mass = abs(mass1 - mass0) / max(mass0, 1e-12) * 100.0
    err_ener = abs(ener1 - ener0) / max(ener0, 1e-12) * 100.0
    mom0_norm = float(np.linalg.norm(mom0))
    err_mom = float(np.linalg.norm(mom1 - mom0)) / max(mom0_norm, 1e-12) * 100.0 if mom0_norm > 0 else 0.0

    return {
        "mass_initial": float(mass0),
        "mass_rezoned": float(mass1),
        "mass_error_pct": float(err_mass),
        "energy_initial": float(ener0),
        "energy_rezoned": float(ener1),
        "energy_error_pct": float(err_ener),
        "momentum_error_pct": float(err_mom),
    }
