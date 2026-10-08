"""
Weiler-Atherton 2D polygon clipping algorithm and polygon utilities.

Fortran source citations:
- polygon_mod:          $OR_SRC/common_source/tools/clipping/polygon_mod.F90
- polygon_clipping_mod: $OR_SRC/common_source/tools/clipping/polygon_clipping_mod.F90

Theory notes:
- The Weiler-Atherton algorithm clips a subject polygon ("ClippedPolygon") against a
  clipping boundary polygon ("ClippingPolygon").
- Supports arbitrary 2D polygons (convex and concave) and can produce multiple disjoint
  result polygons.
- Edges are parameterized with alpha in [0, 1] for the clipped polygon and beta in [0, 1]
  for the clipping polygon.
- Intersections are classified as entering (+1) or leaving (-1) using 2D cross-product
  orientation tests against edge normals.
- Traversal follows the subject polygon entering into the interior, and follows the
  clipping boundary at leaving intersections, closing loops into distinct output polygons.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PolygonPoint:
    """2D point representation matching Fortran polygon_point_ (y, z coordinates)."""
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.y, self.z], dtype=np.float64)


class Polygon:
    """Polygon representation matching Fortran polygon_ in polygon_mod.F90."""

    def __init__(self, points: list[PolygonPoint] | np.ndarray | None = None):
        self.points: list[PolygonPoint] = []
        if points is not None:
            if isinstance(points, np.ndarray):
                for pt in points:
                    self.points.append(PolygonPoint(float(pt[0]), float(pt[1])))
            else:
                for pt in points:
                    if isinstance(pt, PolygonPoint):
                        self.points.append(PolygonPoint(pt.y, pt.z))
                    else:
                        self.points.append(PolygonPoint(float(pt[0]), float(pt[1])))
        self.area: float = 0.0
        self.diag: float = 0.0
        if len(self.points) >= 3:
            self.compute_properties()

    @property
    def numpoint(self) -> int:
        return len(self.points)

    def add_point(self, pt: PolygonPoint | tuple[float, float]) -> None:
        if isinstance(pt, PolygonPoint):
            self.points.append(PolygonPoint(pt.y, pt.z))
        else:
            self.points.append(PolygonPoint(float(pt[0]), float(pt[1])))

    def compute_properties(self) -> None:
        """Compute polygon area and bounding box diagonal."""
        n = len(self.points)
        if n < 3:
            self.area = 0.0
            self.diag = 0.0
            return

        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]
        self.diag = float(np.sqrt((max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2))

        # Shoelace formula matching polygon_SetClockWise
        total = 0.0
        for i in range(n - 1):
            total += (self.points[i + 1].y - self.points[i].y) * (self.points[i + 1].z + self.points[i].z)
        # Close loop if not closed
        if self.points[0].y != self.points[-1].y or self.points[0].z != self.points[-1].z:
            total += (self.points[0].y - self.points[-1].y) * (self.points[0].z + self.points[-1].z)
        self.area = 0.5 * abs(total)

    def to_numpy(self) -> np.ndarray:
        return np.array([[p.y, p.z] for p in self.points], dtype=np.float64)


def intersect_pt(
    p1: PolygonPoint, p2: PolygonPoint, q1: PolygonPoint, q2: PolygonPoint, tol: float = 1.0e-6
) -> tuple[PolygonPoint | None, float, float]:
    """Compute intersection between segment [P1, P2] and [Q1, Q2].

    Ported from $OR_SRC/common_source/tools/clipping/polygon_clipping_mod.F90 (intersectPt)

    Parameters
    ----------
    p1, p2 : PolygonPoint
        Endpoints of first segment [P1, P2].
    q1, q2 : PolygonPoint
        Endpoints of second segment [Q1, Q2].
    tol : float, optional
        Relative tolerance for parallelism test (default 1e-6).

    Returns
    -------
    intersection : PolygonPoint or None
        Intersection point if exists on both segments, else None.
    alpha : float
        Position along [P1, P2] in [0, 1].
    beta : float
        Position along [Q1, Q2] in [0, 1].
    """
    denom = (q2.z - q1.z) * (p2.y - p1.y) - (q2.y - q1.y) * (p2.z - p1.z)
    len_p2 = (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2
    len_q2 = (q2.y - q1.y) ** 2 + (q2.z - q1.z) ** 2
    scale_factor = float(np.sqrt(len_p2 * len_q2))

    if abs(denom) < tol * scale_factor or scale_factor <= 1.0e-30:
        return None, -1.0, -1.0

    numer_a = (q2.y - q1.y) * (p1.z - q1.z) - (q2.z - q1.z) * (p1.y - q1.y)
    numer_b = (p2.y - p1.y) * (p1.z - q1.z) - (p2.z - p1.z) * (p1.y - q1.y)

    alpha = numer_a / denom
    beta = numer_b / denom

    if 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0:
        ix = p1.y + alpha * (p2.y - p1.y)
        iz = p1.z + alpha * (p2.z - p1.z)
        return PolygonPoint(ix, iz), alpha, beta

    return None, alpha, beta


def polygon_set_clockwise(poly: Polygon) -> None:
    """Set polygon orientation to counter-clockwise and compute area.

    Ported from $OR_SRC/common_source/tools/clipping/polygon_clipping_mod.F90 (polygon_SetClockWise)
    """
    n = len(poly.points)
    if n < 3:
        return

    # Ensure closed ring
    pts = poly.points.copy()
    if pts[0].y != pts[-1].y or pts[0].z != pts[-1].z:
        pts.append(PolygonPoint(pts[0].y, pts[0].z))
    n = len(pts)

    total = 0.0
    for i in range(n - 1):
        total += (pts[i + 1].y - pts[i].y) * (pts[i + 1].z + pts[i].z)

    poly.area = 0.5 * abs(total)
    # Fortran: if total > 0 then reverse order (set CCW)
    if total > 0.0:
        pts.reverse()
    poly.points = pts


def polygon_is_point_inside(poly: Polygon, pt: PolygonPoint) -> bool:
    """Ray-casting point-in-polygon test.

    Ported from $OR_SRC/common_source/tools/clipping/polygon_clipping_mod.F90 (polygon_is_point_inside)
    """
    pts = poly.points
    npt = len(pts)
    if npt < 3:
        return False

    # Closed polygon loop
    if pts[0].y != pts[-1].y or pts[0].z != pts[-1].z:
        pts = pts + [PolygonPoint(pts[0].y, pts[0].z)]
        npt = len(pts)

    num_inter_pt = 0
    em20 = 1.0e-20
    for i in range(npt - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        dy = p2.y - p1.y
        dz = p2.z - p1.z

        if abs(dz) > em20:
            lam = (pt.z - p1.z) / dz
            if 0.0 <= lam <= 1.0:
                if (p1.y - pt.y) + lam * dy > em20:
                    num_inter_pt += 1

    return (num_inter_pt % 2) != 0


def clipping_weiler_atherton(
    clipped_poly: Polygon | np.ndarray, clipping_poly: Polygon | np.ndarray
) -> list[Polygon]:
    """Clip subject polygon against clipping polygon using the Weiler-Atherton algorithm.

    Ported from $OR_SRC/common_source/tools/clipping/polygon_clipping_mod.F90 (Clipping_Weiler_Atherton)

    Parameters
    ----------
    clipped_poly : Polygon or np.ndarray
        Subject polygon to clip.
    clipping_poly : Polygon or np.ndarray
        Clipping window / boundary polygon.

    Returns
    -------
    result_polygons : list of Polygon
        List of resulting clipped polygons.
    """
    poly1 = Polygon(clipped_poly) if not isinstance(clipped_poly, Polygon) else clipped_poly
    poly2 = Polygon(clipping_poly) if not isinstance(clipping_poly, Polygon) else clipping_poly

    if len(poly1.points) < 3 or len(poly2.points) < 3:
        return []

    # Ensure both are closed polygons
    pts1 = poly1.points.copy()
    if pts1[0].y != pts1[-1].y or pts1[0].z != pts1[-1].z:
        pts1.append(PolygonPoint(pts1[0].y, pts1[0].z))
    pts2 = poly2.points.copy()
    if pts2[0].y != pts2[-1].y or pts2[0].z != pts2[-1].z:
        pts2.append(PolygonPoint(pts2[0].y, pts2[0].z))

    num_edges_1 = len(pts1) - 1
    num_edges_2 = len(pts2) - 1

    # Edge data structures: list of points on each edge
    # [ { 'alpha': float, 'coor': PolygonPoint, 'id': int, 'orient': int (+1 entering, -1 leaving, 0 vertex) } ]
    edges_1: list[list[dict]] = [[] for _ in range(num_edges_1)]
    edges_2: list[list[dict]] = [[] for _ in range(num_edges_2)]

    for i in range(num_edges_1):
        edges_1[i].append({'alpha': 0.0, 'coor': pts1[i], 'id': 0, 'orient': 0})
    for j in range(num_edges_2):
        edges_2[j].append({'alpha': 0.0, 'coor': pts2[j], 'id': 0, 'orient': 0})

    total_int_pt = 0
    em10 = 1.0e-10

    for i in range(num_edges_1):
        p1 = pts1[i]
        p2 = pts1[i + 1]
        for j in range(num_edges_2):
            q1 = pts2[j]
            q2 = pts2[j + 1]

            # Bounding box test
            if max(p1.y, p2.y) < min(q1.y, q2.y) or max(q1.y, q2.y) < min(p1.y, p2.y):
                continue
            if max(p1.z, p2.z) < min(q1.z, q2.z) or max(q1.z, q2.z) < min(p1.z, p2.z):
                continue

            ipt, alpha, beta = intersect_pt(p1, p2, q1, q2)
            if ipt is None:
                continue

            # Skip vertex hits
            if alpha < em10 or alpha > 1.0 - em10 or beta < em10 or beta > 1.0 - em10:
                continue

            total_int_pt += 1

            # Determine entering (+1) or leaving (-1) from q1q2 orientation
            ny = -(q2.z - q1.z)
            nz = +(q2.y - q1.y)
            vy = p2.y - p1.y
            vz = p2.z - p1.z
            dot = ny * vy + nz * vz
            orient1 = 1 if dot > 0.0 else -1

            # From p1p2 orientation for edge 2
            ny2 = -(p2.z - p1.z)
            nz2 = +(p2.y - p1.y)
            vy2 = q2.y - q1.y
            vz2 = q2.z - q1.z
            dot2 = ny2 * vy2 + nz2 * vz2
            orient2 = 1 if dot2 > 0.0 else -1

            edges_1[i].append({'alpha': alpha, 'coor': ipt, 'id': total_int_pt, 'orient': orient1})
            edges_2[j].append({'alpha': beta, 'coor': ipt, 'id': total_int_pt, 'orient': orient2})

    # Add end points
    for i in range(num_edges_1):
        edges_1[i].append({'alpha': 1.0, 'coor': pts1[i + 1], 'id': 0, 'orient': 0})
        edges_1[i].sort(key=lambda item: item['alpha'])
    for j in range(num_edges_2):
        edges_2[j].append({'alpha': 1.0, 'coor': pts2[j + 1], 'id': 0, 'orient': 0})
        edges_2[j].sort(key=lambda item: item['alpha'])

    # Case of no intersections
    if total_int_pt == 0:
        if polygon_is_point_inside(poly2, pts1[0]):
            res = Polygon(pts1)
            res.compute_properties()
            return [res]
        elif polygon_is_point_inside(poly1, pts2[0]):
            res = Polygon(pts2)
            res.compute_properties()
            return [res]
        else:
            return []

    # Map intersection IDs to positions
    # (edge_idx, pt_idx)
    id_map_1: dict[int, tuple[int, int]] = {}
    id_map_2: dict[int, tuple[int, int]] = {}
    entering_1: list[tuple[int, int, int]] = []  # (edge_idx, pt_idx, point_id)

    for i, edge in enumerate(edges_1):
        for k, pt in enumerate(edge):
            pid = pt['id']
            if pid > 0:
                id_map_1[pid] = (i, k)
                if pt['orient'] == 1:
                    entering_1.append((i, k, pid))

    for j, edge in enumerate(edges_2):
        for k, pt in enumerate(edge):
            pid = pt['id']
            if pid > 0:
                id_map_2[pid] = (j, k)

    # Weiler-Atherton traversal starting from each entering point
    visited_entering: set[int] = set()
    result_polys: list[Polygon] = []

    for start_edge, start_k, start_pid in entering_1:
        if start_pid in visited_entering:
            continue

        poly_pts: list[PolygonPoint] = []
        cur_list = 1  # 1 = clipped, 2 = clipping
        cur_edge = start_edge
        cur_k = start_k
        start_pt = edges_1[cur_edge][cur_k]['coor']
        poly_pts.append(start_pt)
        visited_entering.add(start_pid)

        max_steps = 2 * (num_edges_1 + num_edges_2 + total_int_pt + 10)
        closed = False

        for _ in range(max_steps):
            cur_pt_info = edges_1[cur_edge][cur_k] if cur_list == 1 else edges_2[cur_edge][cur_k]
            orient = cur_pt_info['orient']

            # Move to next point
            if cur_list == 1:
                if orient == -1:
                    # Leaving: switch to list 2 at the corresponding point
                    pid = cur_pt_info['id']
                    cur_edge, cur_k = id_map_2[pid]
                    cur_list = 2
                    # Step forward along list 2
                    cur_k += 1
                    if cur_k >= len(edges_2[cur_edge]):
                        cur_edge = (cur_edge + 1) % num_edges_2
                        cur_k = 1
                else:
                    cur_k += 1
                    if cur_k >= len(edges_1[cur_edge]):
                        cur_edge = (cur_edge + 1) % num_edges_1
                        cur_k = 1
            else:  # cur_list == 2
                if orient == 1:
                    # Entering: switch back to list 1
                    pid = cur_pt_info['id']
                    cur_edge, cur_k = id_map_1[pid]
                    cur_list = 1
                    visited_entering.add(pid)
                    cur_k += 1
                    if cur_k >= len(edges_1[cur_edge]):
                        cur_edge = (cur_edge + 1) % num_edges_1
                        cur_k = 1
                else:
                    cur_k += 1
                    if cur_k >= len(edges_2[cur_edge]):
                        cur_edge = (cur_edge + 1) % num_edges_2
                        cur_k = 1

            next_info = edges_1[cur_edge][cur_k] if cur_list == 1 else edges_2[cur_edge][cur_k]
            pt = next_info['coor']

            # Check if we returned to start
            if abs(pt.y - start_pt.y) < 1.0e-12 and abs(pt.z - start_pt.z) < 1.0e-12:
                poly_pts.append(PolygonPoint(start_pt.y, start_pt.z))
                closed = True
                break

            poly_pts.append(pt)
            if next_info['id'] > 0 and next_info['orient'] == 1 and cur_list == 1:
                visited_entering.add(next_info['id'])

        if closed and len(poly_pts) >= 4:
            p = Polygon(poly_pts)
            p.compute_properties()
            if p.area > 1.0e-20:
                result_polys.append(p)

    return result_polys
