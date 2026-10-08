"""Thermal boundary conditions: Convection cooling and Stefan-Boltzmann radiation.

Upstream OpenRadioss Fortran References:
----------------------------------------
- Starter Card Readers:
  ``$OR_SRC/starter/source/loads/thermic/hm_read_convec.F``
  ``$OR_SRC/starter/source/loads/thermic/hm_read_radiation.F``
  ``$OR_SRC/starter/source/loads/thermic/hm_preread_convec.F``
  ``$OR_SRC/starter/source/loads/thermic/hm_preread_radiation.F``
- Engine Time-Integration Kernels:
  ``$OR_SRC/engine/source/constraints/thermic/convec.F``
  ``$OR_SRC/engine/source/constraints/thermic/radiation.F``

Physics Overview:
-----------------
1. Convection Heat Transfer (/CONVEC, /THERM_LOAD/CONVEC):
   - Newton's law of cooling across surface boundary segments:
     q_conv = h * (T_surf - T_inf)
     where h is the convection heat transfer coefficient (constant or h(T)),
     T_surf is the average surface segment temperature, and T_inf is the fluid/ambient bulk temperature.
   - Total segment heat loss rate:
     P_loss = q_conv * Area
   - Nodal heat loss rate distributed to M segment nodes:
     Q_node = -q_conv * Area / M = h * Area * (T_inf - T_surf) / M
   - Upstream Fortran `convec.F` calculates:
     FLUX = AREA * H * (T_INF - TE) * DT
     FTHE(node) += FLUX / M

2. Stefan-Boltzmann Thermal Radiation (/RADIATION, /THERM_LOAD/RADIATION):
   - Stefan-Boltzmann radiation to ambient enclosure:
     q_rad = epsilon * sigma * (T_surf^4 - T_inf^4)
     where epsilon is the surface emissivity in [0, 1],
     sigma is the Stefan-Boltzmann constant (default 5.670374419e-8 W/(m^2 K^4) in SI),
     and T_surf, T_inf are absolute temperatures in Kelvin.
   - Total segment radiation heat loss rate:
     P_loss = q_rad * Area
   - Nodal heat loss rate distributed to M segment nodes:
     Q_node = -q_rad * Area / M = epsilon * sigma * Area * (T_inf^4 - T_surf^4) / M
   - Upstream Fortran `radiation.F` calculates:
     FLUX = AREA * (EMI * SIGMA) * (T_INF^4 - TE^4) * DT
     FTHE(node) += FLUX / M

3. Thermal Energy Balance & Conservation:
   - Cumulative dissipated thermal energy over time:
     E_conv(t) = integral( sum(q_conv * Area) dt )
     E_rad(t)  = integral( sum(q_rad * Area) dt )
   - In a closed system with lumped thermal mass M_th * cp:
     M_th * cp * dT/dt = sum(Q_node) = -P_conv - P_rad
     Delta(M_th * cp * T) + E_conv + E_rad = 0 (First Law of Thermodynamics)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Stefan-Boltzmann constant in SI units: W / (m^2 * K^4) = J / (s * m^2 * K^4)
STEFAN_BOLTZMANN: float = 5.670374419e-8


def compute_segment_area(
    nodes: Sequence[int],
    coords: Union[np.ndarray, Dict[int, Sequence[float]]],
) -> float:
    """Compute 3D or 2D surface area of a boundary segment.

    Matches geometry kernels in OpenRadioss `convec.F` and `radiation.F`:
    - 4 nodes (quadrilateral):
      Area = 0.5 * ||(x_3 - x_1) x (x_4 - x_2)||
    - 3 nodes (triangle):
      Area = 0.5 * ||(x_2 - x_1) x (x_3 - x_1)||
    - 2 nodes (2D line):
      Length = ||x_2 - x_1||
    """
    n = len(nodes)
    if n < 2:
        return 0.0

    def _get_xyz(nid: int) -> np.ndarray:
        if isinstance(coords, np.ndarray):
            # 1-based or 0-based array index
            idx = nid - 1 if (nid > 0 and len(coords) >= nid) else nid
            return np.asarray(coords[idx][:3], dtype=np.float64)
        elif isinstance(coords, dict):
            return np.asarray(coords[nid][:3], dtype=np.float64)
        raise TypeError(f"Unsupported coords type: {type(coords)}")

    pts = [_get_xyz(nid) for nid in nodes]

    if n == 4:
        # Cross product of diagonals: (x3 - x1) x (x4 - x2)
        d13 = pts[2] - pts[0]
        d24 = pts[3] - pts[1]
        nx = d13[1] * d24[2] - d13[2] * d24[1]
        ny = d13[2] * d24[0] - d13[0] * d24[2]
        nz = d13[0] * d24[1] - d13[1] * d24[0]
        return float(0.5 * math.sqrt(nx * nx + ny * ny + nz * nz))

    elif n == 3:
        # Triangle cross product: (x2 - x1) x (x3 - x1)
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        nx = v1[1] * v2[2] - v1[2] * v2[1]
        ny = v1[2] * v2[0] - v1[0] * v2[2]
        nz = v1[0] * v2[1] - v1[1] * v2[0]
        return float(0.5 * math.sqrt(nx * nx + ny * ny + nz * nz))

    elif n == 2:
        # 2D line segment length
        diff = pts[1] - pts[0]
        return float(math.sqrt(np.dot(diff, diff)))

    # General polygon: fan triangulation from pts[0]
    total_area = 0.0
    for i in range(1, n - 1):
        v1 = pts[i] - pts[0]
        v2 = pts[i + 1] - pts[0]
        cross = np.cross(v1, v2)
        total_area += 0.5 * np.linalg.norm(cross)
    return float(total_area)


def compute_convec_flux(
    t_surf: float,
    t_inf: float,
    h: float,
) -> float:
    """Compute linear convection heat flux q_conv (W / m^2).

    Convention:
    q_conv > 0 means heat leaves the body (cooling).
    q_conv < 0 means heat enters the body (heating).
    """
    return float(h * (t_surf - t_inf))


def compute_radiation_flux(
    t_surf: float,
    t_inf: float,
    emissivity: float,
    sigma: float = STEFAN_BOLTZMANN,
) -> float:
    """Compute Stefan-Boltzmann radiation heat flux q_rad (W / m^2).

    q_rad = epsilon * sigma * (T_surf^4 - T_inf^4)
    Convention:
    q_rad > 0 means heat leaves the body (radiative cooling).
    q_rad < 0 means heat enters the body (radiative heating).
    """
    ts = max(0.0, float(t_surf))
    ti = max(0.0, float(t_inf))
    return float(emissivity * sigma * (ts**4 - ti**4))


@dataclass
class ConvecParams:
    """Parameters and definition of /CONVEC (/THERM_LOAD/CONVEC) thermal boundary condition.

    Upstream Fortran origin:
      - ``starter/source/loads/thermic/hm_read_convec.F``
      - ``engine/source/constraints/thermic/convec.F``
    """

    id: int = 1
    title: str = ""
    surf_id: int = 0
    h: Union[float, Callable[[float], float]] = 0.0  # Heat transfer coeff or h(T)
    t_inf: Union[float, Callable[[float], float]] = 300.0  # Ambient temperature or T_inf(t)
    funct_id: int = 0  # Optional curve ID for T_inf(t)
    sens_id: int = 0  # Optional sensor ID
    xscale: float = 1.0  # Time scale (ASCALE)
    scale: float = 1.0  # Temperature magnitude scale (FSCALE / FCY)
    tstart: float = 0.0  # Activation start time
    tstop: float = 1.0e30  # Deactivation stop time
    segments: List[Sequence[int]] = field(default_factory=list)  # Segment node lists
    areas: Optional[Sequence[float]] = None  # Explicit segment areas if known
    area: float = 0.0  # Single/lumped surface area (used if segments is empty)
    is_active: bool = True

    def get_h(self, t_surf: float = 300.0) -> float:
        """Evaluate convection heat transfer coefficient h."""
        if callable(self.h):
            return float(self.h(t_surf))
        return float(self.h)

    def get_t_inf(self, t: float = 0.0) -> float:
        """Evaluate ambient fluid temperature T_inf at time t."""
        if callable(self.t_inf):
            return float(self.scale * self.t_inf(t * self.xscale))
        return float(self.scale * self.t_inf)

    def check_active(self, t: float) -> bool:
        """Return True if convection condition is active at time t."""
        return self.is_active and (self.tstart <= t <= self.tstop)


# Alias for load naming convention
ConvecLoad = ConvecParams


@dataclass
class RadiationParams:
    """Parameters and definition of /RADIATION (/THERM_LOAD/RADIATION) boundary condition.

    Upstream Fortran origin:
      - ``starter/source/loads/thermic/hm_read_radiation.F``
      - ``engine/source/constraints/thermic/radiation.F``
    """

    id: int = 1
    title: str = ""
    surf_id: int = 0
    emissivity: Union[float, Callable[[float], float]] = 1.0  # Surface emissivity epsilon in [0, 1]
    sigma: float = STEFAN_BOLTZMANN  # Stefan-Boltzmann constant
    t_inf: Union[float, Callable[[float], float]] = 300.0  # Ambient radiation temp or T_inf(t)
    funct_id: int = 0  # Optional curve ID
    sens_id: int = 0  # Optional sensor ID
    xscale: float = 1.0  # Time scale (ASCALE)
    scale: float = 1.0  # Temperature magnitude scale (FSCALE / FCY)
    tstart: float = 0.0  # Activation start time
    tstop: float = 1.0e30  # Deactivation stop time
    segments: List[Sequence[int]] = field(default_factory=list)  # Segment node lists
    areas: Optional[Sequence[float]] = None  # Explicit segment areas if known
    area: float = 0.0  # Single/lumped surface area (used if segments is empty)
    is_active: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.emissivity, (int, float)):
            self.emissivity = float(min(max(self.emissivity, 0.0), 1.0))
        if self.sigma <= 0.0:
            self.sigma = STEFAN_BOLTZMANN

    def get_emissivity(self, t_surf: float = 300.0) -> float:
        """Evaluate surface emissivity."""
        if callable(self.emissivity):
            return float(min(max(self.emissivity(t_surf), 0.0), 1.0))
        return float(self.emissivity)

    def get_t_inf(self, t: float = 0.0) -> float:
        """Evaluate ambient radiation temperature T_inf at time t."""
        if callable(self.t_inf):
            return float(self.scale * self.t_inf(t * self.xscale))
        return float(self.scale * self.t_inf)

    def check_active(self, t: float) -> bool:
        """Return True if radiation condition is active at time t."""
        return self.is_active and (self.tstart <= t <= self.tstop)


# Alias for load naming convention
RadiationLoad = RadiationParams


@dataclass
class ThermalStepResult:
    """Output summary of a thermal boundary condition step."""

    dt: float = 0.0
    time: float = 0.0
    q_conv_nodal: Dict[int, float] = field(default_factory=dict)  # Q_node (W) for convection
    q_rad_nodal: Dict[int, float] = field(default_factory=dict)  # Q_node (W) for radiation
    total_nodal_power: Dict[int, float] = field(default_factory=dict)  # Q_node_total (W)
    power_conv_lost: float = 0.0  # Instantaneous convection heat loss power (W)
    power_rad_lost: float = 0.0  # Instantaneous radiation heat loss power (W)
    energy_conv_dissipated: float = 0.0  # Delta E_conv (J) in current step
    energy_rad_dissipated: float = 0.0  # Delta E_rad (J) in current step
    cumul_energy_conv: float = 0.0  # Cumulative E_conv (J)
    cumul_energy_rad: float = 0.0  # Cumulative E_rad (J)


class ThermalLoadsManager:
    """Manages evaluation and cumulative energy tracking for thermal loads.

    Implements the global thermal balance matching OpenRadioss `glob_therm_mod.F90`:
      - `glob_therm%heat_conv`: cumulative convection heat transferred
      - `glob_therm%heat_radia`: cumulative radiation heat transferred
    """

    def __init__(
        self,
        convec_loads: Optional[Sequence[ConvecParams]] = None,
        radiation_loads: Optional[Sequence[RadiationParams]] = None,
    ) -> None:
        self.convec_loads: List[ConvecParams] = list(convec_loads or [])
        self.radiation_loads: List[RadiationParams] = list(radiation_loads or [])
        self.cumul_energy_conv: float = 0.0  # Total heat dissipated via convection (J)
        self.cumul_energy_rad: float = 0.0  # Total heat dissipated via radiation (J)

    def add_convec(self, load: ConvecParams) -> None:
        self.convec_loads.append(load)

    def add_radiation(self, load: RadiationParams) -> None:
        self.radiation_loads.append(load)

    def reset_energy(self) -> None:
        self.cumul_energy_conv = 0.0
        self.cumul_energy_rad = 0.0

    def compute_step(
        self,
        dt: float,
        time: float,
        temps: Union[np.ndarray, Dict[int, float]],
        coords: Optional[Union[np.ndarray, Dict[int, Sequence[float]]]] = None,
    ) -> ThermalStepResult:
        """Evaluate all thermal boundary loads for time step dt.

        Parameters
        ----------
        dt : float
            Time step increment (s).
        time : float
            Current simulation time (s).
        temps : array or dict
            Nodal temperature map {node_id: T_kelvin} or 1D array.
        coords : array or dict, optional
            Nodal coordinates for segment area computation.

        Returns
        -------
        ThermalStepResult
            Contains nodal thermal load rates Q_node (W) and energy increments.
        """
        res = ThermalStepResult(dt=dt, time=time)

        def _get_temp(nid: int) -> float:
            if isinstance(temps, dict):
                return float(temps.get(nid, 300.0))
            elif isinstance(temps, np.ndarray):
                idx = nid - 1 if (nid > 0 and len(temps) >= nid) else nid
                return float(temps[idx])
            return 300.0

        # -------------------------------------------------------------
        # 1. Evaluate Convection Loads
        # -------------------------------------------------------------
        for cl in self.convec_loads:
            if not cl.check_active(time):
                continue

            t_inf = cl.get_t_inf(time)

            # Case A: Explicit segments defined
            if cl.segments:
                for s_idx, seg_nodes in enumerate(cl.segments):
                    m = len(seg_nodes)
                    if m == 0:
                        continue
                    # Compute segment area
                    if cl.areas is not None and len(cl.areas) > s_idx:
                        area = float(cl.areas[s_idx])
                    elif coords is not None:
                        area = compute_segment_area(seg_nodes, coords)
                    else:
                        area = cl.area / len(cl.segments) if cl.area > 0 else 1.0

                    t_surf = sum(_get_temp(n) for n in seg_nodes) / m
                    h_val = cl.get_h(t_surf)
                    q_flux = compute_convec_flux(t_surf, t_inf, h_val)
                    p_loss = q_flux * area

                    res.power_conv_lost += p_loss
                    q_node = -p_loss / m
                    for n in seg_nodes:
                        res.q_conv_nodal[n] = res.q_conv_nodal.get(n, 0.0) + q_node
                        res.total_nodal_power[n] = res.total_nodal_power.get(n, 0.0) + q_node

            # Case B: Lumped single area (e.g. single node or surface-less test)
            elif cl.area > 0.0:
                t_surf = _get_temp(1)
                h_val = cl.get_h(t_surf)
                q_flux = compute_convec_flux(t_surf, t_inf, h_val)
                p_loss = q_flux * cl.area
                res.power_conv_lost += p_loss
                q_node = -p_loss
                res.q_conv_nodal[1] = res.q_conv_nodal.get(1, 0.0) + q_node
                res.total_nodal_power[1] = res.total_nodal_power.get(1, 0.0) + q_node

        # -------------------------------------------------------------
        # 2. Evaluate Radiation Loads
        # -------------------------------------------------------------
        for rl in self.radiation_loads:
            if not rl.check_active(time):
                continue

            t_inf = rl.get_t_inf(time)

            # Case A: Explicit segments defined
            if rl.segments:
                for s_idx, seg_nodes in enumerate(rl.segments):
                    m = len(seg_nodes)
                    if m == 0:
                        continue
                    # Compute segment area
                    if rl.areas is not None and len(rl.areas) > s_idx:
                        area = float(rl.areas[s_idx])
                    elif coords is not None:
                        area = compute_segment_area(seg_nodes, coords)
                    else:
                        area = rl.area / len(rl.segments) if rl.area > 0 else 1.0

                    t_surf = sum(_get_temp(n) for n in seg_nodes) / m
                    eps = rl.get_emissivity(t_surf)
                    q_flux = compute_radiation_flux(t_surf, t_inf, eps, rl.sigma)
                    p_loss = q_flux * area

                    res.power_rad_lost += p_loss
                    q_node = -p_loss / m
                    for n in seg_nodes:
                        res.q_rad_nodal[n] = res.q_rad_nodal.get(n, 0.0) + q_node
                        res.total_nodal_power[n] = res.total_nodal_power.get(n, 0.0) + q_node

            # Case B: Lumped single area
            elif rl.area > 0.0:
                t_surf = _get_temp(1)
                eps = rl.get_emissivity(t_surf)
                q_flux = compute_radiation_flux(t_surf, t_inf, eps, rl.sigma)
                p_loss = q_flux * rl.area
                res.power_rad_lost += p_loss
                q_node = -p_loss
                res.q_rad_nodal[1] = res.q_rad_nodal.get(1, 0.0) + q_node
                res.total_nodal_power[1] = res.total_nodal_power.get(1, 0.0) + q_node

        # -------------------------------------------------------------
        # 3. Energy Balance Update
        # -------------------------------------------------------------
        res.energy_conv_dissipated = res.power_conv_lost * dt
        res.energy_rad_dissipated = res.power_rad_lost * dt

        self.cumul_energy_conv += res.energy_conv_dissipated
        self.cumul_energy_rad += res.energy_rad_dissipated

        res.cumul_energy_conv = self.cumul_energy_conv
        res.cumul_energy_rad = self.cumul_energy_rad

        return res


def convec_subroutine(
    ibcv: np.ndarray,
    fconv: np.ndarray,
    x: np.ndarray,
    temp: np.ndarray,
    fthe: np.ndarray,
    dt: float,
    time: float = 0.0,
    theaccfact: float = 1.0,
) -> float:
    """Emulate Fortran subroutine CONVEC from `convec.F`.

    Parameters
    ----------
    ibcv : np.ndarray, shape (6, num_conv)
      IBCV(1..4, i) = segment nodes n1, n2, n3, n4 (1-based, n4=0 for triangle)
      IBCV(5, i) = function index
      IBCV(6, i) = sensor index
    fconv : np.ndarray, shape (6, num_conv)
      FCONV(1, i) = FCY (temperature magnitude / scale)
      FCONV(2, i) = 1 / FCX (time scale)
      FCONV(3, i) = H (convection heat transfer coeff)
      FCONV(4, i) = STARTT
      FCONV(5, i) = STOPT
      FCONV(6, i) = OFFG (active > 0)
    x : np.ndarray, shape (3, num_nodes)
      Nodal coordinates
    temp : np.ndarray, shape (num_nodes,)
      Nodal temperatures
    fthe : np.ndarray, shape (num_nodes,)
      Thermal force/heat increment vector (modified in place)
    dt : float
      Time increment
    time : float
      Current time
    theaccfact : float
      Thermal acceleration factor (default 1.0)

    Returns
    -------
    heat_conv : float
      Heat transferred to the body (J), matching Fortran `glob_therm%heat_conv`.
    """
    num_conv = ibcv.shape[1]
    heat_conv = 0.0
    t_scaled = time * theaccfact

    for nl in range(num_conv):
        offg = fconv[5, nl]
        if offg <= 0.0:
            continue
        startt = fconv[3, nl]
        stopt = fconv[4, nl]
        if t_scaled < startt or t_scaled > stopt:
            continue

        n1 = int(ibcv[0, nl]) - 1
        n2 = int(ibcv[1, nl]) - 1
        n3 = int(ibcv[2, nl]) - 1
        n4 = int(ibcv[3, nl]) - 1

        fcy = fconv[0, nl]
        h = fconv[2, nl]
        t_inf = fcy

        if n4 >= 0:
            # 4-node quad: cross product of diagonals (x3-x1) x (x4-x2)
            d13 = x[:, n3] - x[:, n1]
            d24 = x[:, n4] - x[:, n2]
            cross = np.cross(d13, d24)
            area = 0.5 * float(np.linalg.norm(cross))
            te = 0.25 * float(temp[n1] + temp[n2] + temp[n3] + temp[n4])
            flux = area * h * (t_inf - te) * dt * theaccfact
            heat_conv += flux
            flux_node = 0.25 * flux
            fthe[n1] += flux_node
            fthe[n2] += flux_node
            fthe[n3] += flux_node
            fthe[n4] += flux_node
        elif n3 >= 0:
            # 3-node tri: cross product (x2-x1) x (x3-x1)
            v1 = x[:, n2] - x[:, n1]
            v2 = x[:, n3] - x[:, n1]
            cross = np.cross(v1, v2)
            area = 0.5 * float(np.linalg.norm(cross))
            te = (1.0 / 3.0) * float(temp[n1] + temp[n2] + temp[n3])
            flux = area * h * (t_inf - te) * dt * theaccfact
            heat_conv += flux
            flux_node = (1.0 / 3.0) * flux
            fthe[n1] += flux_node
            fthe[n2] += flux_node
            fthe[n3] += flux_node
        else:
            # 2-node line
            diff = x[:, n2] - x[:, n1]
            area = float(np.linalg.norm(diff))
            te = 0.5 * float(temp[n1] + temp[n2])
            flux = area * h * (t_inf - te) * dt * theaccfact
            heat_conv += flux
            flux_node = 0.5 * flux
            fthe[n1] += flux_node
            fthe[n2] += flux_node

    return heat_conv


def radiation_subroutine(
    ibcr: np.ndarray,
    fradia: np.ndarray,
    x: np.ndarray,
    temp: np.ndarray,
    fthe: np.ndarray,
    dt: float,
    time: float = 0.0,
    theaccfact: float = 1.0,
) -> float:
    """Emulate Fortran subroutine RADIATION from `radiation.F`.

    Parameters
    ----------
    ibcr : np.ndarray, shape (6, num_rad)
      IBCR(1..4, i) = segment nodes n1, n2, n3, n4 (1-based, n4=0 for triangle)
      IBCR(5, i) = function index
      IBCR(6, i) = sensor index
    fradia : np.ndarray, shape (6, num_rad)
      FRADIA(1, i) = FCY (temperature magnitude)
      FRADIA(2, i) = 1 / FCX (time scale)
      FRADIA(3, i) = EMISIG (epsilon * sigma)
      FRADIA(4, i) = STARTT
      FRADIA(5, i) = STOPT
      FRADIA(6, i) = OFFG (active > 0)
    x : np.ndarray, shape (3, num_nodes)
      Nodal coordinates
    temp : np.ndarray, shape (num_nodes,)
      Nodal temperatures
    fthe : np.ndarray, shape (num_nodes,)
      Thermal force/heat increment vector (modified in place)
    dt : float
      Time increment
    time : float
      Current time
    theaccfact : float
      Thermal acceleration factor

    Returns
    -------
    heat_radia : float
      Heat transferred to the body (J), matching Fortran `glob_therm%heat_radia`.
    """
    num_rad = ibcr.shape[1]
    heat_radia = 0.0
    t_scaled = time * theaccfact

    for nl in range(num_rad):
        offg = fradia[5, nl]
        if offg <= 0.0:
            continue
        startt = fradia[3, nl]
        stopt = fradia[4, nl]
        if t_scaled < startt or t_scaled > stopt:
            continue

        n1 = int(ibcr[0, nl]) - 1
        n2 = int(ibcr[1, nl]) - 1
        n3 = int(ibcr[2, nl]) - 1
        n4 = int(ibcr[3, nl]) - 1

        fcy = fradia[0, nl]
        emisig = fradia[2, nl]
        t_inf = fcy

        if n4 >= 0:
            d13 = x[:, n3] - x[:, n1]
            d24 = x[:, n4] - x[:, n2]
            cross = np.cross(d13, d24)
            area = 0.5 * float(np.linalg.norm(cross))
            te = 0.25 * float(temp[n1] + temp[n2] + temp[n3] + temp[n4])
            flux = area * emisig * (t_inf**4 - te**4) * dt * theaccfact
            heat_radia += flux
            flux_node = 0.25 * flux
            fthe[n1] += flux_node
            fthe[n2] += flux_node
            fthe[n3] += flux_node
            fthe[n4] += flux_node
        elif n3 >= 0:
            v1 = x[:, n2] - x[:, n1]
            v2 = x[:, n3] - x[:, n1]
            cross = np.cross(v1, v2)
            area = 0.5 * float(np.linalg.norm(cross))
            te = (1.0 / 3.0) * float(temp[n1] + temp[n2] + temp[n3])
            flux = area * emisig * (t_inf**4 - te**4) * dt * theaccfact
            heat_radia += flux
            flux_node = (1.0 / 3.0) * flux
            fthe[n1] += flux_node
            fthe[n2] += flux_node
            fthe[n3] += flux_node
        else:
            diff = x[:, n2] - x[:, n1]
            area = float(np.linalg.norm(diff))
            te = 0.5 * float(temp[n1] + temp[n2])
            flux = area * emisig * (t_inf**4 - te**4) * dt * theaccfact
            heat_radia += flux
            flux_node = 0.5 * flux
            fthe[n1] += flux_node
            fthe[n2] += flux_node

    return heat_radia


def simulate_lumped_cooling(
    m_therm: float,
    cp: float,
    t_init: float,
    convec: Optional[ConvecParams] = None,
    radiation: Optional[RadiationParams] = None,
    t_end: float = 10.0,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Simulate transient cooling of a lumped thermal capacitance M_therm * c_p.

    Solves:
      M_therm * c_p * dT/dt = -P_conv - P_rad
    and tracks energy balance:
      Delta E_thermal + E_conv + E_rad == 0

    Returns
    -------
    dict with history arrays: 'times', 'temps', 'e_conv', 'e_rad', 'e_thermal', 'energy_error'
    """
    times = []
    temps = []
    e_conv_hist = []
    e_rad_hist = []
    e_err_hist = []

    mgr = ThermalLoadsManager(
        convec_loads=[convec] if convec else [],
        radiation_loads=[radiation] if radiation else [],
    )

    t = 0.0
    temp = float(t_init)
    c_th = float(m_therm * cp)
    e_init = c_th * temp

    while t <= t_end + 1e-12:
        step_res = mgr.compute_step(dt=dt, time=t, temps={1: temp})

        times.append(t)
        temps.append(temp)
        e_conv_hist.append(mgr.cumul_energy_conv)
        e_rad_hist.append(mgr.cumul_energy_rad)

        # Thermal energy change in the mass: E_th(t) - E_th(0)
        delta_e_mass = c_th * (temp - t_init)
        # First law: delta_e_mass + E_dissipated = 0
        err = abs(delta_e_mass + mgr.cumul_energy_conv + mgr.cumul_energy_rad)
        e_err_hist.append(err)

        # Explicit Euler temperature update
        p_total = step_res.total_nodal_power.get(1, 0.0)
        temp += (p_total / c_th) * dt
        t += dt

    return {
        "times": np.array(times),
        "temps": np.array(temps),
        "e_conv": np.array(e_conv_hist),
        "e_rad": np.array(e_rad_hist),
        "energy_error": np.array(e_err_hist),
    }
