"""Hydrostatic fluid surface pressure loading (/LOAD/PFLUID).

Upstream OpenRadioss Fortran References:
----------------------------------------
- Starter Card Readers:
  ``$OR_SRC/starter/source/loads/general/pfluid/hm_read_pfluid.F``
  ``$OR_SRC/starter/source/loads/general/pfluid/hm_preread_pfluid.F``
  ``hm_cfg_files/config/CFG/radioss120/LOADS/pfluid.cfg``
- Engine Time-Integration Kernel:
  ``$OR_SRC/engine/source/loads/general/pfluid/pfluid.F``

Physics Overview:
-----------------
1. Hydrostatic Pressure Field:
   - For a fluid of density rho_f under effective acceleration a_eff:
     Effective gravity vector:
       g_eff = g - a_container
       |g_eff| = norm(g_eff)
       d_depth = -g_eff / |g_eff| (unit vector in direction of increasing depth)
   - Depth of surface segment centroid x_c below liquid free surface reference x0 (or altitude z0):
       h = max(0.0, (x0 - x_c) . d_depth)
   - Fluid pressure:
       P = P0 + rho_f * |g_eff| * h   (if submerged, h > 0)
       P = P0                         (if above liquid level, h <= 0)
     where P0 is the atmospheric / free-surface pressure.

2. Segment Force Integration:
   - For each 3D boundary segment (quadrilateral or triangle) with area A and unit normal n:
       F_seg = P * A * n
   - Nodal force distribution:
       F_node = F_seg / M   (M=4 for quad, M=3 for triangle)
     assembled into global external force vector F_ext.

3. External Work Accumulation:
   - Centroid velocity v_c = (1/M) * sum(v_node_i)
   - Instantaneous power:
       P_ext = sum( F_seg . v_c )
   - Cumulative external work:
       dW = P_ext * dt
       W_ext += dW
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class PfluidSegmentResult:
    """Hydrostatic pressure evaluation result for a single surface segment."""

    segment_index: int
    nodes: Tuple[int, ...]
    centroid: np.ndarray
    normal: np.ndarray
    area: float
    depth: float
    pressure: float
    force_vector: np.ndarray
    power: float = 0.0


@dataclass
class PfluidLoadParams:
    """Parameters for /LOAD/PFLUID hydrostatic surface pressure loading.

    Upstream Fortran origin:
      - ``starter/source/loads/general/pfluid/hm_read_pfluid.F``
      - ``engine/source/loads/general/pfluid/pfluid.F``
    """

    id: int = 1
    title: str = ""
    surf_id: int = 0
    segments: List[Sequence[int]] = field(default_factory=list)  # Segment node IDs
    rho_f: float = 1000.0  # Fluid density (kg/m^3 in SI)
    z0: float = 0.0  # Liquid free surface reference coordinate (m)
    p0: float = 0.0  # Atmospheric / free surface pressure (Pa)
    gravity: Union[Sequence[float], np.ndarray] = (0.0, 0.0, -9.81)  # Gravity vector g
    a_container: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0)  # Container acceleration
    ref_point: Optional[Union[Sequence[float], np.ndarray]] = None  # Free surface ref point x0
    depth_dir: Optional[Union[Sequence[float], np.ndarray]] = None  # Direction of depth
    sensor_id: int = 0  # Sensor ID for gating
    tstart: float = 0.0  # Activation start time
    tstop: float = 1.0e30  # Deactivation stop time
    scale: float = 1.0  # Pressure scale factor
    flip_normal: bool = False  # If True, reverse normal direction
    is_active: bool = True

    def get_effective_gravity(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute effective gravity vector g_eff, magnitude, and unit depth direction.

        Returns
        -------
        g_eff : np.ndarray (3,)
        g_mag : float
        d_depth : np.ndarray (3,)
        """
        g = np.asarray(self.gravity, dtype=np.float64)
        a = np.asarray(self.a_container, dtype=np.float64)
        g_eff = g - a
        g_mag = float(np.linalg.norm(g_eff))

        if self.depth_dir is not None:
            d = np.asarray(self.depth_dir, dtype=np.float64)
            d_norm = np.linalg.norm(d)
            d_depth = d / d_norm if d_norm > 1e-12 else np.array([0.0, 0.0, -1.0])
        elif g_mag > 1e-12:
            # Depth increases opposite to effective gravity direction
            d_depth = -g_eff / g_mag
        else:
            d_depth = np.array([0.0, 0.0, -1.0])

        return g_eff, g_mag, d_depth

    def get_free_surface_point(self) -> np.ndarray:
        """Return free surface reference point x0."""
        if self.ref_point is not None:
            return np.asarray(self.ref_point, dtype=np.float64)
        return np.array([0.0, 0.0, float(self.z0)], dtype=np.float64)

    def check_active(self, t: float, sensor_val: Optional[float] = None) -> bool:
        """Check if load is active based on time and sensor."""
        if not self.is_active:
            return False
        if t < self.tstart or t > self.tstop:
            return False
        if self.sensor_id > 0 and sensor_val is not None:
            if sensor_val <= 0.0:
                return False
        return True


# Alias for load naming convention
PfluidLoad = PfluidLoadParams


@dataclass
class PfluidStepResult:
    """Output summary of /LOAD/PFLUID pressure integration."""

    dt: float = 0.0
    time: float = 0.0
    nodal_forces: Dict[int, np.ndarray] = field(default_factory=dict)  # node_id -> [Fx, Fy, Fz]
    total_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    total_power: float = 0.0
    incremental_work: float = 0.0
    cumul_work: float = 0.0
    segment_results: List[PfluidSegmentResult] = field(default_factory=list)


def compute_segment_normal_and_area(
    pts: Sequence[np.ndarray],
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute centroid, area, and unit normal vector for a 3D polygonal segment.

    Parameters
    ----------
    pts : sequence of 3D points [x, y, z]

    Returns
    -------
    centroid : np.ndarray (3,)
    area : float
    normal : np.ndarray (3,) unit vector
    """
    m = len(pts)
    if m < 3:
        raise ValueError(f"Segment must have at least 3 vertices, got {m}")

    centroid = np.mean(pts, axis=0)

    if m == 4:
        # 4-node quadrilateral: cross product of diagonals (x3 - x1) x (x4 - x2)
        d13 = pts[2] - pts[0]
        d24 = pts[3] - pts[1]
        cross = np.cross(d13, d24)
        norm = float(np.linalg.norm(cross))
        area = 0.5 * norm
        normal = cross / norm if norm > 1e-15 else np.array([0.0, 0.0, 1.0])
        return centroid, area, normal

    elif m == 3:
        # 3-node triangle: (x2 - x1) x (x3 - x1)
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross = np.cross(v1, v2)
        norm = float(np.linalg.norm(cross))
        area = 0.5 * norm
        normal = cross / norm if norm > 1e-15 else np.array([0.0, 0.0, 1.0])
        return centroid, area, normal

    # General polygon: fan triangulation from centroid
    total_area = 0.0
    normal_accum = np.zeros(3, dtype=np.float64)
    for i in range(m):
        v1 = pts[i] - centroid
        v2 = pts[(i + 1) % m] - centroid
        cr = np.cross(v1, v2)
        a = 0.5 * np.linalg.norm(cr)
        total_area += a
        normal_accum += cr
    norm = np.linalg.norm(normal_accum)
    normal = normal_accum / norm if norm > 1e-15 else np.array([0.0, 0.0, 1.0])
    return centroid, float(total_area), normal


class PfluidEngine:
    """Manages evaluation and external work tracking for /LOAD/PFLUID hydrostatic loads."""

    def __init__(self, loads: Optional[Sequence[PfluidLoadParams]] = None) -> None:
        self.loads: List[PfluidLoadParams] = list(loads or [])
        self.cumul_work: float = 0.0

    def add_load(self, load: PfluidLoadParams) -> None:
        self.loads.append(load)

    def reset(self) -> None:
        self.cumul_work = 0.0

    def compute_step(
        self,
        dt: float,
        time: float,
        coords: Union[np.ndarray, Dict[int, Sequence[float]]],
        vels: Optional[Union[np.ndarray, Dict[int, Sequence[float]]]] = None,
        sensor_active: Optional[Dict[int, bool]] = None,
    ) -> PfluidStepResult:
        """Evaluate hydrostatic pressure loading across all segments for time step dt.

        Parameters
        ----------
        dt : float
            Time step increment (s).
        time : float
            Current time (s).
        coords : array or dict
            Nodal coordinates {node_id: [x, y, z]} or array of shape (N, 3).
        vels : array or dict, optional
            Nodal velocities {node_id: [vx, vy, vz]} or array of shape (N, 3).
        sensor_active : dict, optional
            Mapping sensor_id -> bool indicating if sensor is active.

        Returns
        -------
        PfluidStepResult
            Result containing nodal forces and external work increments.
        """
        res = PfluidStepResult(dt=dt, time=time)

        def _get_coord(nid: int) -> np.ndarray:
            if isinstance(coords, dict):
                return np.asarray(coords[nid][:3], dtype=np.float64)
            elif isinstance(coords, np.ndarray):
                idx = nid - 1 if (nid > 0 and len(coords) >= nid) else nid
                return np.asarray(coords[idx][:3], dtype=np.float64)
            raise TypeError("Unsupported coords type")

        def _get_vel(nid: int) -> np.ndarray:
            if vels is None:
                return np.zeros(3, dtype=np.float64)
            if isinstance(vels, dict):
                return np.asarray(vels.get(nid, [0.0, 0.0, 0.0])[:3], dtype=np.float64)
            elif isinstance(vels, np.ndarray):
                idx = nid - 1 if (nid > 0 and len(vels) >= nid) else nid
                return np.asarray(vels[idx][:3], dtype=np.float64)
            return np.zeros(3, dtype=np.float64)

        for load in self.loads:
            sens_val = None
            if load.sensor_id > 0 and sensor_active is not None:
                sens_val = 1.0 if sensor_active.get(load.sensor_id, True) else 0.0

            if not load.check_active(time, sens_val):
                continue

            g_eff, g_mag, d_depth = load.get_effective_gravity()
            x0 = load.get_free_surface_point()

            for s_idx, seg_nodes in enumerate(load.segments):
                m = len(seg_nodes)
                if m < 3:
                    continue

                pts = [_get_coord(nid) for nid in seg_nodes]
                centroid, area, normal = compute_segment_normal_and_area(pts)

                if load.flip_normal:
                    normal = -normal

                # Depth below free surface along gravity direction
                # h = (x0 - centroid) . d_depth
                depth = float(np.dot(x0 - centroid, d_depth))

                if depth > 0.0:
                    # Submerged: hydrostatic pressure P = P0 + rho * g_eff * h
                    pressure = float((load.p0 + load.rho_f * g_mag * depth) * load.scale)
                else:
                    # Above liquid level: only top surface / atmospheric pressure P0
                    depth = 0.0
                    pressure = float(max(0.0, load.p0) * load.scale)

                # Segment normal force vector: F = P * Area * normal
                f_seg = pressure * area * normal
                res.total_force += f_seg

                # Nodal distribution: F_node = F_seg / M
                f_node = f_seg / m
                for nid in seg_nodes:
                    if nid not in res.nodal_forces:
                        res.nodal_forces[nid] = np.zeros(3, dtype=np.float64)
                    res.nodal_forces[nid] += f_node

                # Centroid velocity and power
                v_centroid = np.mean([_get_vel(nid) for nid in seg_nodes], axis=0)
                seg_power = float(np.dot(f_seg, v_centroid))
                res.total_power += seg_power

                res.segment_results.append(
                    PfluidSegmentResult(
                        segment_index=s_idx,
                        nodes=tuple(seg_nodes),
                        centroid=centroid,
                        normal=normal,
                        area=area,
                        depth=depth,
                        pressure=pressure,
                        force_vector=f_seg,
                        power=seg_power,
                    )
                )

        # External work increment dW = P_ext * dt
        res.incremental_work = res.total_power * dt
        self.cumul_work += res.incremental_work
        res.cumul_work = self.cumul_work

        return res


def pfluid_subroutine(
    iloadp: np.ndarray,
    rload: np.ndarray,
    lloadp: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    fext: np.ndarray,
    dt: float,
    time: float = 0.0,
    rho_f: float = 1000.0,
    g: float = 9.81,
    z0: float = 0.0,
) -> float:
    """Emulate OpenRadioss Fortran kernel `pfluid.F` for testing and parity verification.

    Parameters
    ----------
    iloadp : np.ndarray
      Integer parameters from starter
    rload : np.ndarray
      Real parameters from starter
    lloadp : np.ndarray
      Segment node IDs (4 nodes per segment)
    x : np.ndarray, shape (3, num_nodes)
      Nodal coordinates
    v : np.ndarray, shape (3, num_nodes)
      Nodal velocities
    fext : np.ndarray, shape (3, num_nodes)
      External force array (modified in place)
    dt : float
      Time step
    time : float
      Current time
    rho_f : float
      Fluid density
    g : float
      Gravitational acceleration
    z0 : float
      Free surface altitude

    Returns
    -------
    wfext : float
      External work increment (J)
    """
    num_segs = len(lloadp) // 4
    wfext = 0.0

    for i in range(num_segs):
        n1 = int(lloadp[4 * i]) - 1
        n2 = int(lloadp[4 * i + 1]) - 1
        n3 = int(lloadp[4 * i + 2]) - 1
        n4 = int(lloadp[4 * i + 3]) - 1

        if n4 >= 0 and n4 != n1 and n4 != n2 and n4 != n3:
            # Quad
            x_c = 0.25 * (x[:, n1] + x[:, n2] + x[:, n3] + x[:, n4])
            d13 = x[:, n3] - x[:, n1]
            d24 = x[:, n4] - x[:, n2]
            cross = np.cross(d13, d24)
            norm = float(np.linalg.norm(cross))
            area = 0.5 * norm
            normal = cross / norm if norm > 1e-15 else np.array([0.0, 0.0, 1.0])
            m = 4
            nodes = [n1, n2, n3, n4]
        else:
            # Tri
            x_c = (1.0 / 3.0) * (x[:, n1] + x[:, n2] + x[:, n3])
            v1 = x[:, n2] - x[:, n1]
            v2 = x[:, n3] - x[:, n1]
            cross = np.cross(v1, v2)
            norm = float(np.linalg.norm(cross))
            area = 0.5 * norm
            normal = cross / norm if norm > 1e-15 else np.array([0.0, 0.0, 1.0])
            m = 3
            nodes = [n1, n2, n3]

        # Depth below z0
        depth = max(0.0, z0 - float(x_c[2]))
        pressure = rho_f * g * depth
        f_seg = pressure * area * normal
        f_node = f_seg / m

        for nid in nodes:
            fext[:, nid] += f_node

        v_centroid = np.mean([v[:, nid] for nid in nodes], axis=0)
        wfext += float(np.dot(f_seg, v_centroid)) * dt

    return wfext
