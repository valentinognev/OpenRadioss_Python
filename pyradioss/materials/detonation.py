"""Detonation Front Surfaces (/DFS/) and Explosive Burn Kinetics.

Ported from OpenRadioss Fortran source:
- `/DFS/DETPOINT` (point detonator source):
  `$OR_SRC/starter/source/initial_conditions/detonation/read_dfs_detpoint.F`
  Subroutine: `READ_DFS_DETPOINT`
- `/DFS/DETLINE` (line detonator source):
  `$OR_SRC/starter/source/initial_conditions/detonation/read_dfs_detline.F`
  Subroutine: `READ_DFS_DETLINE`
- `/DFS/DETPLAN` (planar wave detonation front):
  `$OR_SRC/starter/source/initial_conditions/detonation/read_dfs_detplan.F`
  Subroutine: `READ_DFS_DETPLAN`
- Eikonal wavefront propagation:
  `$OR_SRC/starter/source/initial_conditions/detonation/eikonal_solver.F90`
  Subroutine: `EIKONAL_SOLVER`
- Explosive burn fraction kinetics:
  `$OR_SRC/engine/source/materials/mat/mat051/compute_bfrac.F`
  `$OR_SRC/engine/source/materials/mat/mat005/m5law.F` (lines 108-121)

Physics & Formulation:
----------------------
1. Point Detonator (/DFS/DETPOINT):
   A spherical detonation wave emanates from point x0 = (x0, y0, z0) with detonation
   velocity D_cj starting at ignition time t0:
     r = ||x - x0||
     t_arr(x) = t0 + r / D_cj

2. Line Detonator (/DFS/DETLINE):
   A detonation wave emanates from a straight line segment [p1, p2].
   If instantaneous line ignition (v_line = inf or None):
     d_seg = distance(x, segment[p1, p2])
     t_arr(x) = t0 + d_seg / D_cj
   If progressive line burning from p1 to p2 at velocity v_line:
     p(s) = p1 + s * (p2 - p1) for s in [0, 1]
     t_ign(s) = t0 + s * L / v_line
     t_arr(x) = min_{s in [0, 1]} ( t_ign(s) + ||x - p(s)|| / D_cj )

3. Planar Detonator (/DFS/DETPLAN):
   A planar detonation wave starts at plane origin x_p with unit normal n
   at time t0, propagating in direction n with speed D_cj:
     d = dot(x - x_p, n)
     t_arr(x) = t0 + max(0, d) / D_cj

4. Multi-Detonator Wave Interaction:
   When multiple detonators are present, the earliest arrival wave ignites the material:
     t_arr(x) = min_{i} t_arr_i(x)

5. Burn Fraction Kinetics (compute_bfrac.F lines 52-61, m5law.F lines 108-121):
   For an explosive element with characteristic size dx = V^(1/3):
     burn_fraction = clamp( (t - t_arr) / (1.5 * dx / D_cj), 0.0, 1.0 )
   At t <= t_arr: burn_fraction = 0.0 (unreacted explosive).
   At t >= t_arr + 1.5 * dx / D_cj: burn_fraction = 1.0 (fully reacted detonation gas).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_TINY = 1.0e-20
_INF = 1.0e30


def clamp(val: float | np.ndarray, low: float = 0.0, high: float = 1.0) -> float | np.ndarray:
    """Clamp value(s) to [low, high]."""
    if isinstance(val, np.ndarray):
        return np.clip(val, low, high)
    return max(low, min(high, float(val)))


def burn_fraction(
    t: float | np.ndarray,
    t_arr: float | np.ndarray,
    dx: float | np.ndarray,
    d_cj: float,
) -> float | np.ndarray:
    """Compute JWL / explosive burn fraction matching compute_bfrac.F lines 52-61.

    Formula:
      dt_burn = 1.5 * dx / D_cj
      bfrac = clamp((t - t_arr) / dt_burn, 0.0, 1.0)

    Parameters
    ----------
    t : float or ndarray
        Current simulation time.
    t_arr : float or ndarray
        Detonation wave arrival time at element / point.
    dx : float or ndarray
        Characteristic element length (e.g. V^(1/3)).
    d_cj : float
        Chapman-Jouguet detonation velocity (m/s).

    Returns
    -------
    float or ndarray
        Burn fraction in [0.0, 1.0].
    """
    d_cj_safe = max(float(d_cj), _TINY)
    dt_burn = np.maximum(1.5 * np.asarray(dx, dtype=float) / d_cj_safe, _TINY)
    delta_t = np.asarray(t, dtype=float) - np.asarray(t_arr, dtype=float)
    frac = delta_t / dt_burn
    frac = np.where(np.isclose(frac, 1.0, atol=1e-12, rtol=1e-12), 1.0, frac)
    frac = np.where(np.isclose(frac, 0.0, atol=1e-12, rtol=1e-12), 0.0, frac)
    res = np.clip(frac, 0.0, 1.0)
    if np.ndim(res) == 0:
        return float(res)
    return res


@dataclass
class DetPoint:
    """Point-source explosive detonator (/DFS/DETPOINT).

    Fortran reference:
      `starter/source/initial_conditions/detonation/read_dfs_detpoint.F`
    """

    x0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    t0: float = 0.0
    d_cj: float = 7000.0  # default Chapman-Jouguet velocity (m/s)
    id: int = 1
    mat_id: int = 0
    name: str = "DETPOINT"

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float).reshape(3)
        if self.d_cj <= 0.0:
            self.d_cj = 7000.0

    @classmethod
    def from_coords(
        cls,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        t0: float = 0.0,
        d_cj: float = 7000.0,
        id: int = 1,
        mat_id: int = 0,
    ) -> DetPoint:
        """Construct DetPoint from individual coordinate scalars."""
        return cls(x0=np.array([x, y, z], dtype=float), t0=t0, d_cj=d_cj, id=id, mat_id=mat_id)

    def arrival_time(self, x: np.ndarray | Sequence[float]) -> float | np.ndarray:
        """Calculate detonation arrival time at target point(s) x.

        Parameters
        ----------
        x : ndarray of shape (3,) or (N, 3)
            Target spatial coordinates.

        Returns
        -------
        float or ndarray
            Arrival time: t0 + ||x - x0|| / D_cj.
        """
        x_arr = np.asarray(x, dtype=float)
        is_1d = (x_arr.ndim == 1)
        pts = np.atleast_2d(x_arr)

        diff = pts - self.x0.reshape(1, 3)
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        t_arr = self.t0 + dist / self.d_cj

        return float(t_arr[0]) if is_1d else t_arr

    def burn_fraction(
        self,
        x: np.ndarray | Sequence[float],
        t: float,
        dx: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute burn fraction at point(s) x at time t."""
        t_arr = self.arrival_time(x)
        return burn_fraction(t, t_arr, dx, self.d_cj)


@dataclass
class DetLine:
    """Line-source explosive detonator (/DFS/DETLINE).

    Fortran reference:
      `starter/source/initial_conditions/detonation/read_dfs_detline.F`
    """

    p1: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    p2: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    t0: float = 0.0
    d_cj: float = 7000.0
    v_line: Optional[float] = None  # None or inf means simultaneous ignition along line
    id: int = 1
    mat_id: int = 0
    name: str = "DETLINE"

    def __post_init__(self) -> None:
        self.p1 = np.asarray(self.p1, dtype=float).reshape(3)
        self.p2 = np.asarray(self.p2, dtype=float).reshape(3)
        if self.d_cj <= 0.0:
            self.d_cj = 7000.0

    @classmethod
    def from_endpoints(
        cls,
        p1: Sequence[float],
        p2: Sequence[float],
        t0: float = 0.0,
        d_cj: float = 7000.0,
        v_line: Optional[float] = None,
        id: int = 1,
        mat_id: int = 0,
    ) -> DetLine:
        """Construct DetLine from two endpoint 3-vectors."""
        return cls(
            p1=np.asarray(p1, dtype=float),
            p2=np.asarray(p2, dtype=float),
            t0=t0,
            d_cj=d_cj,
            v_line=v_line,
            id=id,
            mat_id=mat_id,
        )

    def arrival_time(self, x: np.ndarray | Sequence[float]) -> float | np.ndarray:
        """Calculate detonation arrival time at target point(s) x from line segment.

        Supports both instantaneous line ignition (v_line is None or inf) and
        progressive burning along the line at speed v_line.
        """
        x_arr = np.asarray(x, dtype=float)
        is_1d = (x_arr.ndim == 1)
        pts = np.atleast_2d(x_arr)
        n_pts = len(pts)

        seg_vec = self.p2 - self.p1
        seg_len_sq = float(np.sum(seg_vec**2))
        seg_len = math.sqrt(max(seg_len_sq, _TINY))

        # 1. Simultaneous line ignition
        if self.v_line is None or math.isinf(self.v_line) or self.v_line <= 0.0:
            if seg_len_sq < _TINY:
                # Degenerate line segment -> behaves as point p1
                diff = pts - self.p1.reshape(1, 3)
                d_seg = np.sqrt(np.sum(diff**2, axis=-1))
            else:
                p1_to_x = pts - self.p1.reshape(1, 3)
                proj = np.sum(p1_to_x * seg_vec.reshape(1, 3), axis=-1) / seg_len_sq
                s_clamped = np.clip(proj, 0.0, 1.0)[:, np.newaxis]
                closest_pts = self.p1.reshape(1, 3) + s_clamped * seg_vec.reshape(1, 3)
                d_seg = np.sqrt(np.sum((pts - closest_pts)**2, axis=-1))
            t_arr = self.t0 + d_seg / self.d_cj
            return float(t_arr[0]) if is_1d else t_arr

        # 2. Progressive burn along line at speed v_line
        # Discretize segment or evaluate exact stationary point
        v_l = float(self.v_line)
        # We sample s in [0, 1] with adaptive resolution
        n_samples = max(21, int(seg_len / max(0.01, seg_len / 50.0)))
        s_vals = np.linspace(0.0, 1.0, n_samples)
        line_pts = self.p1.reshape(1, 3) + s_vals[:, np.newaxis] * seg_vec.reshape(1, 3)  # (M, 3)
        t_ign = self.t0 + (s_vals * seg_len) / v_l  # (M,)

        # Distance from each line sample to target points: shape (N, M)
        diff = pts[:, np.newaxis, :] - line_pts[np.newaxis, :, :]  # (N, M, 3)
        dist = np.sqrt(np.sum(diff**2, axis=-1))  # (N, M)
        t_matrix = t_ign[np.newaxis, :] + dist / self.d_cj
        t_arr = np.min(t_matrix, axis=-1)

        return float(t_arr[0]) if is_1d else t_arr

    def burn_fraction(
        self,
        x: np.ndarray | Sequence[float],
        t: float,
        dx: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute burn fraction at point(s) x at time t."""
        t_arr = self.arrival_time(x)
        return burn_fraction(t, t_arr, dx, self.d_cj)


@dataclass
class DetPlan:
    """Planar detonation front wave (/DFS/DETPLAN).

    Fortran reference:
      `starter/source/initial_conditions/detonation/read_dfs_detplan.F`
    """

    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    t0: float = 0.0
    d_cj: float = 7000.0
    id: int = 1
    mat_id: int = 0
    ignite_behind: bool = True  # Points behind the plane ignite at t0
    name: str = "DETPLAN"

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        norm_raw = np.asarray(self.normal, dtype=float).reshape(3)
        n_len = math.sqrt(float(np.sum(norm_raw**2)))
        if n_len < _TINY:
            self.normal = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self.normal = norm_raw / n_len
        if self.d_cj <= 0.0:
            self.d_cj = 7000.0

    @classmethod
    def from_origin_normal(
        cls,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        nx: float = 1.0,
        ny: float = 0.0,
        nz: float = 0.0,
        t0: float = 0.0,
        d_cj: float = 7000.0,
        id: int = 1,
        mat_id: int = 0,
        ignite_behind: bool = True,
    ) -> DetPlan:
        """Construct DetPlan from origin and normal components."""
        return cls(
            origin=np.array([x, y, z], dtype=float),
            normal=np.array([nx, ny, nz], dtype=float),
            t0=t0,
            d_cj=d_cj,
            id=id,
            mat_id=mat_id,
            ignite_behind=ignite_behind,
        )

    def arrival_time(self, x: np.ndarray | Sequence[float]) -> float | np.ndarray:
        """Calculate planar detonation wave arrival time at point(s) x.

        Parameters
        ----------
        x : ndarray of shape (3,) or (N, 3)
            Target spatial coordinates.

        Returns
        -------
        float or ndarray
            Arrival time: t0 + max(0, (x - x_plane) . n) / D_cj.
        """
        x_arr = np.asarray(x, dtype=float)
        is_1d = (x_arr.ndim == 1)
        pts = np.atleast_2d(x_arr)

        diff = pts - self.origin.reshape(1, 3)
        proj = np.sum(diff * self.normal.reshape(1, 3), axis=-1)

        if self.ignite_behind:
            d_eff = np.maximum(0.0, proj)
        else:
            d_eff = proj

        t_arr = self.t0 + d_eff / self.d_cj
        return float(t_arr[0]) if is_1d else t_arr

    def burn_fraction(
        self,
        x: np.ndarray | Sequence[float],
        t: float,
        dx: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute burn fraction at point(s) x at time t."""
        t_arr = self.arrival_time(x)
        return burn_fraction(t, t_arr, dx, self.d_cj)


class DetonationSystem:
    """Manages multi-detonator interactions across explosive domains."""

    def __init__(
        self,
        detonators: Optional[Sequence[Union[DetPoint, DetLine, DetPlan]]] = None,
        default_d_cj: float = 7000.0,
    ) -> None:
        self.detonators: List[Union[DetPoint, DetLine, DetPlan]] = list(detonators or [])
        self.default_d_cj = default_d_cj

    def add(self, det: Union[DetPoint, DetLine, DetPlan]) -> None:
        """Add a detonator to the system."""
        self.detonators.append(det)

    def add_point(
        self,
        x: float,
        y: float,
        z: float,
        t0: float = 0.0,
        d_cj: Optional[float] = None,
        id: int = 1,
        mat_id: int = 0,
    ) -> DetPoint:
        """Add and return a DetPoint detonator."""
        vel = d_cj if d_cj is not None else self.default_d_cj
        det = DetPoint.from_coords(x, y, z, t0=t0, d_cj=vel, id=id, mat_id=mat_id)
        self.add(det)
        return det

    def add_line(
        self,
        p1: Sequence[float],
        p2: Sequence[float],
        t0: float = 0.0,
        d_cj: Optional[float] = None,
        v_line: Optional[float] = None,
        id: int = 1,
        mat_id: int = 0,
    ) -> DetLine:
        """Add and return a DetLine detonator."""
        vel = d_cj if d_cj is not None else self.default_d_cj
        det = DetLine.from_endpoints(p1, p2, t0=t0, d_cj=vel, v_line=v_line, id=id, mat_id=mat_id)
        self.add(det)
        return det

    def add_plane(
        self,
        origin: Sequence[float],
        normal: Sequence[float],
        t0: float = 0.0,
        d_cj: Optional[float] = None,
        id: int = 1,
        mat_id: int = 0,
        ignite_behind: bool = True,
    ) -> DetPlan:
        """Add and return a DetPlan detonator."""
        vel = d_cj if d_cj is not None else self.default_d_cj
        det = DetPlan(
            origin=np.asarray(origin, dtype=float),
            normal=np.asarray(normal, dtype=float),
            t0=t0,
            d_cj=vel,
            id=id,
            mat_id=mat_id,
            ignite_behind=ignite_behind,
        )
        self.add(det)
        return det

    def arrival_time(self, x: np.ndarray | Sequence[float]) -> float | np.ndarray:
        """Calculate effective arrival time as min over all active detonators.

        t_arr(x) = min_{i} t_arr_i(x)
        """
        if not self.detonators:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                return _INF
            return np.full(len(x_arr), _INF, dtype=float)

        x_arr = np.asarray(x, dtype=float)
        is_1d = (x_arr.ndim == 1)
        pts = np.atleast_2d(x_arr)

        times = np.array([det.arrival_time(pts) for det in self.detonators])
        min_t = np.min(times, axis=0)

        return float(min_t[0]) if is_1d else min_t

    def burn_fraction(
        self,
        x: np.ndarray | Sequence[float],
        t: float,
        dx: float | np.ndarray,
        d_cj: Optional[float] = None,
    ) -> float | np.ndarray:
        """Compute combined burn fraction at coordinates x at time t."""
        t_arr = self.arrival_time(x)
        vel = d_cj if d_cj is not None else self.default_d_cj
        return burn_fraction(t, t_arr, dx, vel)


def arrival_time(
    detonators: Union[DetPoint, DetLine, DetPlan, Sequence[Any]],
    x: np.ndarray | Sequence[float],
) -> float | np.ndarray:
    """Calculate arrival time from a detonator or sequence of detonators."""
    if isinstance(detonators, (DetPoint, DetLine, DetPlan)):
        return detonators.arrival_time(x)
    system = DetonationSystem(detonators)
    return system.arrival_time(x)
