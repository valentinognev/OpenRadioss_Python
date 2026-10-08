# $OR_SRC/starter/source/loads/sph/hm_read_sphio.F
# Subroutine: HM_READ_SPHIO (lines 44-610)
# $OR_SRC/engine/source/elements/sph/sponof1.F
# Subroutine: SPONOF1 (lines 39-558)
# $OR_SRC/engine/source/elements/sph/sponof2.F
# Subroutine: SPONOF2 (lines 39-820)
"""Smoothed Particle Hydrodynamics (SPH) Inflow and Outflow Boundary Conditions.

Faithful port of OpenRadioss Fortran source:
  - ``starter/source/loads/sph/hm_read_sphio.F``:
      Reads /SPH/INOUT (/SPH/INFLOW, /SPH/OUTFLOW) boundary definitions,
      parses inlet/outlet surface geometries, flow velocity, density, and energy.
  - ``engine/source/elements/sph/sponof1.F`` (SPONOF1):
      SPH inlet condition (ITYPE = 1). Injects SPH particles across the boundary
      surface with prescribed velocity, density, and specific energy.
  - ``engine/source/elements/sph/sponof2.F`` (SPONOF2):
      SPH outlet / silent boundary condition (ITYPE = 2, 3). Deactivates particles
      crossing the outflow control surface to avoid non-physical pressure reflection.

Physics & Formulations:
-----------------------
1. Inflow Boundary Condition (/SPH/INFLOW, ITYPE = 1):
   - Defined by a planar injection surface S with area A, origin x0, and unit normal n.
   - Prescribed inflow velocity v_in = V_N * n (or arbitrary 3D vector v_in(t)).
   - Particle spacing dx_in: determines initial spatial lattice and volume vol_p = dx_in^3.
   - Particle mass: m_p = rho_in * vol_p.
   - Mass flow rate:
       m_dot = rho_in * A * ||v_in||
   - Injection interval:
       dt_inject = dx_in / max(||v_in||, 1e-20)
     Every dt_inject, a new discrete layer of particles is introduced into the domain
     with initial position, velocity v_in, density rho_in, and internal energy e_in.

2. Outflow Boundary Condition (/SPH/OUTFLOW, ITYPE = 2, 3):
   - Defined by a planar control surface with reference point x_plane and unit normal n_out.
   - Signed distance for each particle:
       d = (x_p - x_plane) . n_out
   - When d > 0 (or d > DIST buffer distance), the particle has crossed the outflow boundary.
   - Deactivation:
       Particles crossing the outflow boundary are removed from active neighbor search,
       density summations, and force integration, preventing artificial boundary reflections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_TINY = 1.0e-20


# ---------------------------------------------------------------------------
# Inflow Parameter & Controller Classes
# ---------------------------------------------------------------------------

@dataclass
class SphInflowParams:
    """Configuration parameters for an SPH inflow boundary (/SPH/INFLOW).

    Upstream Fortran reference:
      ``starter/source/loads/sph/hm_read_sphio.F`` lines 137-140, 276-289, 317-322.
      ``engine/source/elements/sph/sponof1.F`` lines 133-228.
    """
    id: int = 1
    part_id: Optional[int] = None
    title: str = "SPH_INFLOW"
    # Planar geometry
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    width: float = 1.0
    height: float = 1.0
    radius: Optional[float] = None  # if circular nozzle
    # Flow properties
    velocity: Union[float, Sequence[float], Callable[[float], np.ndarray]] = 1.0
    density: float = 1000.0
    internal_energy: float = 0.0
    # Particle discretization
    particle_spacing: float = 0.1
    smoothing_length: Optional[float] = None
    time_delay: float = 0.0


class SphInflow:
    """SPH Inflow Boundary generator and particle injector.

    Implements periodic generation of discrete SPH particle layers matching
    ``sponof1.F`` lines 131-235.
    """

    def __init__(self, params: SphInflowParams | Dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(params, SphInflowParams):
            self.params = params
        elif isinstance(params, dict):
            p_dict = dict(params)
            p_dict.update(kwargs)
            self.params = SphInflowParams(**p_dict)
        elif params is None:
            self.params = SphInflowParams(**kwargs)
        else:
            raise TypeError(f"Invalid parameters type: {type(params)}")

        # Normalize normal vector
        norm = np.asarray(self.params.normal, dtype=np.float64)
        n_len = np.linalg.norm(norm)
        self.normal = norm / n_len if n_len > _TINY else np.array([1.0, 0.0, 0.0])

        self.origin = np.asarray(self.params.origin, dtype=np.float64)
        self.dx = float(self.params.particle_spacing)
        self.rho0 = float(self.params.density)
        self.e0 = float(self.params.internal_energy)

        # Build orthonormal basis (e1, e2, normal) on the inflow plane
        self.e1, self.e2 = self._build_plane_basis(self.normal)

        # Calculate area and particle mass
        self.area = self._compute_area()
        # Single particle volume in 3D: dx^3
        self.vol_particle = self.dx ** 3
        self.mass_particle = self.rho0 * self.vol_particle
        self.h_particle = (
            float(self.params.smoothing_length)
            if self.params.smoothing_length is not None
            else 1.2 * self.dx
        )

        # Internal state tracking
        self.time_accum = 0.0
        self.total_injected_particles = 0
        self.total_injected_mass = 0.0
        self.current_time = 0.0

        # Precompute base particle coordinates on the 2D planar injection grid
        self._base_grid_offsets = self._generate_base_grid()
        self.particles_per_layer = len(self._base_grid_offsets)

    @staticmethod
    def _build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct two orthogonal unit vectors spanning the plane perpendicular to normal."""
        ref = np.array([0.0, 1.0, 0.0]) if abs(normal[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(normal, ref)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        e2 = e2 / np.linalg.norm(e2)
        return e1, e2

    def _compute_area(self) -> float:
        """Compute surface area of inflow injection plane."""
        if self.params.radius is not None and self.params.radius > 0.0:
            return math.pi * (self.params.radius ** 2)
        return float(self.params.width * self.params.height)

    def _generate_base_grid(self) -> np.ndarray:
        """Generate 2D relative offsets (du, dv) for particles in a single layer."""
        offsets = []
        if self.params.radius is not None and self.params.radius > 0.0:
            r = self.params.radius
            nx = int(math.ceil(2.0 * r / self.dx))
            xs = np.linspace(-r + 0.5 * self.dx, r - 0.5 * self.dx, nx)
            for u in xs:
                for v in xs:
                    if u * u + v * v <= r * r:
                        offsets.append((u, v))
        else:
            w = self.params.width
            h = self.params.height
            nx = max(1, int(round(w / self.dx)))
            ny = max(1, int(round(h / self.dx)))
            xs = np.linspace(-0.5 * w + 0.5 * self.dx, 0.5 * w - 0.5 * self.dx, nx)
            ys = np.linspace(-0.5 * h + 0.5 * self.dx, 0.5 * h - 0.5 * self.dx, ny)
            for u in xs:
                for v in ys:
                    offsets.append((u, v))

        if not offsets:
            offsets.append((0.0, 0.0))
        return np.asarray(offsets, dtype=np.float64)

    def get_velocity_vector(self, time: float = 0.0) -> np.ndarray:
        """Compute the inflow velocity vector v_in(t) in 3D."""
        v_param = self.params.velocity
        if callable(v_param):
            res = np.asarray(v_param(time), dtype=np.float64)
            if res.ndim == 0 or len(res) == 1:
                return float(res) * self.normal
            return res
        if isinstance(v_param, (list, tuple, np.ndarray)):
            arr = np.asarray(v_param, dtype=np.float64)
            if len(arr) == 3:
                return arr
            if len(arr) == 1:
                return float(arr[0]) * self.normal
        return float(v_param) * self.normal

    @property
    def speed(self) -> float:
        """Current magnitude of inflow velocity."""
        return float(np.linalg.norm(self.get_velocity_vector(self.current_time)))

    @property
    def dt_inject(self) -> float:
        """Time interval between consecutive particle layer injections."""
        v_mag = max(self.speed, _TINY)
        return self.dx / v_mag

    @property
    def mass_flow_rate(self) -> float:
        """Theoretical continuous mass flow rate m_dot = rho * A * v_in."""
        return self.rho0 * self.area * self.speed

    def generate_single_layer(
        self,
        time: float = 0.0,
        offset_distance: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Generate a single discrete layer of SPH particles at the inflow boundary.

        Parameters
        ----------
        time : float
            Simulation time for velocity evaluation.
        offset_distance : float
            Offset along flow normal direction from the origin.

        Returns
        -------
        dict with particle arrays ('pos', 'vel', 'mass', 'rho', 'energy', 'h').
        """
        n_pts = len(self._base_grid_offsets)
        v_vec = self.get_velocity_vector(time)

        # Base 3D positions on the plane
        pos = np.zeros((n_pts, 3), dtype=np.float64)
        pos += self.origin[None, :]
        pos += self._base_grid_offsets[:, 0:1] * self.e1[None, :]
        pos += self._base_grid_offsets[:, 1:2] * self.e2[None, :]
        pos += offset_distance * self.normal[None, :]

        vel = np.broadcast_to(v_vec, (n_pts, 3)).copy()
        mass = np.full(n_pts, self.mass_particle, dtype=np.float64)
        rho = np.full(n_pts, self.rho0, dtype=np.float64)
        energy = np.full(n_pts, self.e0, dtype=np.float64)
        h_arr = np.full(n_pts, self.h_particle, dtype=np.float64)

        return {
            "pos": pos,
            "vel": vel,
            "mass": mass,
            "rho": rho,
            "energy": energy,
            "h": h_arr,
        }

    def update(self, dt: float, time: Optional[float] = None) -> Optional[Dict[str, np.ndarray]]:
        """Advance time and inject new layers of particles if dt_inject is reached.

        Parameters
        ----------
        dt : float
            Time step increment.
        time : float, optional
            Current simulation time.

        Returns
        -------
        dict or None:
            If new particles are generated, returns concatenated dict of particle
            arrays; otherwise returns None.
        """
        if time is not None:
            self.current_time = time
        else:
            self.current_time += dt

        if self.current_time < self.params.time_delay:
            return None

        self.time_accum += dt
        dt_inj = self.dt_inject

        if self.time_accum < dt_inj:
            return None

        # Determine number of complete particle layers to inject
        num_layers = int(self.time_accum / dt_inj)
        self.time_accum -= num_layers * dt_inj

        all_layers: List[Dict[str, np.ndarray]] = []
        for k in range(num_layers):
            # Stagger layers by their flow advance: (k + 0.5) * dx
            dist_offset = (k + 0.5) * self.dx
            layer = self.generate_single_layer(
                time=self.current_time,
                offset_distance=dist_offset,
            )
            all_layers.append(layer)
            self.total_injected_particles += len(layer["pos"])
            self.total_injected_mass += float(np.sum(layer["mass"]))

        # Concatenate injected layers
        merged = {
            k: np.concatenate([layer[k] for layer in all_layers], axis=0)
            for k in all_layers[0].keys()
        }
        return merged


# ---------------------------------------------------------------------------
# Outflow Parameter & Controller Classes
# ---------------------------------------------------------------------------

@dataclass
class SphOutflowParams:
    """Configuration parameters for an SPH outflow boundary (/SPH/OUTFLOW).

    Upstream Fortran reference:
      ``starter/source/loads/sph/hm_read_sphio.F`` lines 141-150, 323-334.
      ``engine/source/elements/sph/sponof2.F`` lines 130-220.
    """
    id: int = 1
    part_id: Optional[int] = None
    title: str = "SPH_OUTFLOW"
    point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    distance_buffer: float = 0.0  # DIST parameter: boundary buffer thickness


class SphOutflow:
    """SPH Outflow Boundary and non-reflecting deactivator.

    Identifies and filters particles that cross the designated boundary plane,
    preventing non-physical shock reflections (sponof2.F lines 130-220).
    """

    def __init__(self, params: SphOutflowParams | Dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(params, SphOutflowParams):
            self.params = params
        elif isinstance(params, dict):
            p_dict = dict(params)
            p_dict.update(kwargs)
            self.params = SphOutflowParams(**p_dict)
        elif params is None:
            self.params = SphOutflowParams(**kwargs)
        else:
            raise TypeError(f"Invalid parameters type: {type(params)}")

        norm = np.asarray(self.params.normal, dtype=np.float64)
        n_len = np.linalg.norm(norm)
        self.normal = norm / n_len if n_len > _TINY else np.array([1.0, 0.0, 0.0])
        self.point = np.asarray(self.params.point, dtype=np.float64)
        self.dist = float(self.params.distance_buffer)

        # Statistics
        self.total_deactivated_particles = 0
        self.total_deactivated_mass = 0.0

    def compute_signed_distance(self, pos: np.ndarray) -> np.ndarray:
        """Compute signed distance from particles to the outflow plane.

        Positive values mean particle is in the outflow half-space.
        Formula: d = (pos - point) . normal
        """
        pos_arr = np.asarray(pos, dtype=np.float64)
        diff = pos_arr - self.point[None, :]
        return np.dot(diff, self.normal)

    def check_particles(
        self,
        pos: np.ndarray,
        mass: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify particles crossing the outflow plane.

        Parameters
        ----------
        pos : ndarray of shape (N, 3)
            Particle spatial coordinates.
        mass : ndarray of shape (N,), optional
            Particle masses for accounting.
        active_mask : ndarray of shape (N,), optional
            Current alive mask (1/True = alive, 0/False = deactivated).

        Returns
        -------
        crossed_mask : ndarray of shape (N,) bool
            True for particles that have newly crossed the outflow plane.
        new_active_mask : ndarray of shape (N,) bool
            Updated active status (False for particles outside domain).
        """
        d = self.compute_signed_distance(pos)
        crossed = d > self.dist

        if active_mask is not None:
            is_active = np.asarray(active_mask, dtype=bool)
            newly_crossed = crossed & is_active
            new_active = is_active & (~crossed)
        else:
            newly_crossed = crossed
            new_active = ~crossed

        n_new = int(np.sum(newly_crossed))
        if n_new > 0:
            self.total_deactivated_particles += n_new
            if mass is not None:
                m_arr = np.asarray(mass, dtype=np.float64)
                self.total_deactivated_mass += float(np.sum(m_arr[newly_crossed]))

        return newly_crossed, new_active

    def filter_particles(self, particles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Filter out particles that have crossed the outflow plane.

        Parameters
        ----------
        particles : dict
            Dictionary containing particle arrays ('pos', 'vel', 'mass', etc.)

        Returns
        -------
        dict with deactivated particles removed.
        """
        if "pos" not in particles or len(particles["pos"]) == 0:
            return particles

        pos = particles["pos"]
        mass = particles.get("mass", None)
        _, keep_mask = self.check_particles(pos, mass=mass)

        filtered = {}
        for k, v in particles.items():
            if isinstance(v, np.ndarray) and len(v) == len(pos):
                filtered[k] = v[keep_mask]
            else:
                filtered[k] = v

        return filtered


# ---------------------------------------------------------------------------
# Composite Boundary Manager
# ---------------------------------------------------------------------------

class SphBoundaryManager:
    """Manages SPH inflow and outflow boundary conditions during time stepping."""

    def __init__(self) -> None:
        self.inflows: List[SphInflow] = []
        self.outflows: List[SphOutflow] = []

    def add_inflow(self, inflow: SphInflow | SphInflowParams | Dict[str, Any]) -> SphInflow:
        """Add an inflow boundary condition."""
        if not isinstance(inflow, SphInflow):
            inflow = SphInflow(inflow)
        self.inflows.append(inflow)
        return inflow

    def add_outflow(self, outflow: SphOutflow | SphOutflowParams | Dict[str, Any]) -> SphOutflow:
        """Add an outflow boundary condition."""
        if not isinstance(outflow, SphOutflow):
            outflow = SphOutflow(outflow)
        self.outflows.append(outflow)
        return outflow

    def step(
        self,
        particles: Dict[str, np.ndarray],
        dt: float,
        time: Optional[float] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Apply inflow injections and outflow deactivations for one time step.

        Parameters
        ----------
        particles : dict
            Active particles dictionary containing 'pos', 'vel', 'mass', etc.
        dt : float
            Time step.
        time : float, optional
            Current simulation time.

        Returns
        -------
        updated_particles : dict
            Updated particles container with injected particles added and
            exited particles removed.
        stats : dict
            Step statistics (new_injected, newly_deactivated, total_mass, etc.)
        """
        injected_this_step = 0
        injected_mass_step = 0.0

        # 1. Process all inflow boundaries
        for inflow in self.inflows:
            new_layer = inflow.update(dt, time=time)
            if new_layer is not None and len(new_layer["pos"]) > 0:
                n_new = len(new_layer["pos"])
                injected_this_step += n_new
                injected_mass_step += float(np.sum(new_layer["mass"]))

                if len(particles.get("pos", [])) == 0:
                    particles = new_layer
                else:
                    for k in new_layer.keys():
                        if k in particles and isinstance(particles[k], np.ndarray):
                            particles[k] = np.concatenate([particles[k], new_layer[k]], axis=0)

        # 2. Process all outflow boundaries
        deact_this_step = 0
        deact_mass_step = 0.0
        for outflow in self.outflows:
            if "pos" in particles and len(particles["pos"]) > 0:
                n_before = len(particles["pos"])
                particles = outflow.filter_particles(particles)
                n_after = len(particles["pos"])
                diff = n_before - n_after
                deact_this_step += diff

        stats = {
            "injected_count": injected_this_step,
            "injected_mass": injected_mass_step,
            "deactivated_count": deact_this_step,
            "active_particle_count": len(particles.get("pos", [])),
            "total_mass": float(np.sum(particles.get("mass", 0.0))),
        }
        return particles, stats


# ---------------------------------------------------------------------------
# Builder Functions
# ---------------------------------------------------------------------------

def build_sph_inflow(data: Any = None, **kwargs: Any) -> SphInflow:
    """Factory to construct an SphInflow instance."""
    if isinstance(data, SphInflow):
        return data
    d: Dict[str, Any] = {}
    if isinstance(data, SphInflowParams):
        return SphInflow(data)
    if isinstance(data, dict):
        d.update(data)
    elif hasattr(data, "__dict__"):
        d.update(data.__dict__)
    d.update(kwargs)
    return SphInflow(d)


def build_sph_outflow(data: Any = None, **kwargs: Any) -> SphOutflow:
    """Factory to construct an SphOutflow instance."""
    if isinstance(data, SphOutflow):
        return data
    d: Dict[str, Any] = {}
    if isinstance(data, SphOutflowParams):
        return SphOutflow(data)
    if isinstance(data, dict):
        d.update(data)
    elif hasattr(data, "__dict__"):
        d.update(data.__dict__)
    d.update(kwargs)
    return SphOutflow(d)
