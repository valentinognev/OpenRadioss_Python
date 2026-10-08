"""Thermal Rigid Wall (/RWALL/THERM) — Frictional Contact Heating and Interface Conduction.

Upstream OpenRadioss Fortran References:
----------------------------------------
- Card Reader:
  ``$OR_SRC/starter/source/constraints/general/rwall/hm_read_rwall_therm.F``
- Engine Solver:
  ``$OR_SRC/engine/source/constraints/general/rwall/rgwal0.F`` (RGWALT)
- CFG Schema:
  ``$OR_SRC/hm_cfg_files/config/CFG/radioss110/RWALL/therm_plane.cfg``

Physics & Formulation:
----------------------
1. Thermal Rigid Wall Parameters (/RWALL/THERM):
   - Initial wall temperature T_wall0 (Fscale_T in input card)
   - Optional time-dependent wall temperature function IFUNC (fct_IDt)
   - Thermal contact resistance R_th (Thermalresistance in card) or conductance h_cont = 1 / R_th
     matching hm_read_rwall_therm.F line 411: TSTIF = ONE / TSTIF
   - Thermal effusivities:
     e_struct = sqrt(k_struct * rho_struct * Cp_struct)
     e_wall   = sqrt(k_wall   * rho_wall   * Cp_wall)
   - Frictional heat partition coefficient:
     f_struct = e_struct / (e_struct + e_wall)
     (or explicit user fraction fheat in [0, 1])

2. Frictional Heat Generation (Coulomb sliding, slide=2):
   - Normal contact force:
     F_N = m_node * |dv_n| / dt
   - Frictional dissipation power:
     P_fric = mu * |F_N| * |v_rel_tan|
     (For frictionless slide=0 or mu=0: P_fric = 0)
   - Frictional heat flux partitioning:
     Q_struct_fric = f_struct * P_fric
     Q_wall_fric   = (1 - f_struct) * P_fric

3. Interface Conductive Heat Transfer:
   - Heat transfer across contact interface:
     Q_cond = h_cont * A_contact * (T_wall - T_node)
     where h_cont = 1 / R_th if R_th > 0

4. Nodal Temperature Update:
   - Net heat flux into structural node:
     Q_node_net = Q_struct_fric + Q_cond
   - Nodal temperature change over time step dt:
     dT_node = Q_node_net * dt / (m_node * Cp_struct)
     T_node_new = T_node + dT_node

5. First-Law Energy Conservation:
   - Total frictional work generated:
     E_fric = P_fric * dt
   - Thermal internal energy partitioning:
     E_th_struct = Q_struct_fric * dt
     E_th_wall   = Q_wall_fric * dt
     E_th_struct + E_th_wall == E_fric (exact conservation to machine precision)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM20


@dataclass
class RwallThermalParams:
    """Parameters for a thermal rigid wall (/RWALL/THERM).

    Upstream Fortran: ``hm_read_rwall_therm.F`` lines 155-200, 399-440.
    """

    id: int = 1
    title: str = ""
    slide: int = 2  # 0=frictionless sliding, 1=tied, 2=Coulomb friction
    fric: float = 0.0  # Coulomb friction coefficient mu
    dist: float = 0.0  # Search distance / offset

    # Wall thermal properties
    t_wall0: float = 293.15  # Initial wall temperature [K] (Fscale_T)
    fscale_t: float = 293.15  # Temperature scale factor
    fct_idt: Optional[int] = None  # Function ID for wall temperature T(t)

    k_w: float = 50.0  # Wall thermal conductivity [W/(m*K)]
    cp_w: float = 500.0  # Wall specific heat capacity [J/(kg*K)]
    rho_w: float = 7800.0  # Wall mass density [kg/m^3]

    # Structure thermal properties
    k_struct: float = 50.0  # Structural thermal conductivity [W/(m*K)]
    cp_struct: float = 500.0  # Structural specific heat capacity [J/(kg*K)]
    rho_struct: float = 7800.0  # Structural mass density [kg/m^3]

    # Friction heat partition fraction (None -> calculated from effusivities)
    f_struct: Optional[float] = None

    # Interface thermal contact parameters
    thermal_resistance: float = 0.0  # Thermal resistance R_th [K*m^2/W]
    h_cont: float = 0.0  # Thermal conductance h_cont [W/(m^2*K)] = 1/R_th
    area_contact: float = 1.0  # Nominal contact area [m^2]

    # Plane geometry
    point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    # Node sets
    grnod_id: Optional[int] = None
    exclude_grnod_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.t_wall0 <= 0.0 and self.fscale_t > 0.0:
            self.t_wall0 = self.fscale_t
        elif self.fscale_t <= 0.0 and self.t_wall0 > 0.0:
            self.fscale_t = self.t_wall0

        # Match Fortran hm_read_rwall_therm.F line 411: TSTIF = ONE / TSTIF
        if self.h_cont <= 0.0 and self.thermal_resistance > 0.0:
            self.h_cont = 1.0 / self.thermal_resistance
        elif self.thermal_resistance <= 0.0 and self.h_cont > 0.0:
            self.thermal_resistance = 1.0 / self.h_cont

        self.point = np.asarray(self.point, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(self.normal)
        if norm > EM20:
            self.normal = self.normal / norm

    @property
    def effusivity_wall(self) -> float:
        """Thermal effusivity of the rigid wall e_w = sqrt(k_w * rho_w * Cp_w)."""
        return math.sqrt(max(0.0, self.k_w * self.rho_w * self.cp_w))

    @property
    def effusivity_struct(self) -> float:
        """Thermal effusivity of the structure e_s = sqrt(k_s * rho_s * Cp_s)."""
        return math.sqrt(max(0.0, self.k_struct * self.rho_struct * self.cp_struct))

    @property
    def partition_struct(self) -> float:
        """Fraction of frictional heat dissipated into the structural contacting node."""
        if self.f_struct is not None:
            return max(0.0, min(1.0, float(self.f_struct)))
        e_s = self.effusivity_struct
        e_w = self.effusivity_wall
        denom = e_s + e_w
        if denom > EM20:
            return e_s / denom
        return 0.5

    @property
    def partition_wall(self) -> float:
        """Fraction of frictional heat dissipated into the rigid wall."""
        return 1.0 - self.partition_struct

    @property
    def conductance(self) -> float:
        """Interface contact thermal conductance h_cont [W/(m^2*K)]."""
        if self.h_cont > 0.0:
            return self.h_cont
        if self.thermal_resistance > 0.0:
            return 1.0 / self.thermal_resistance
        return 0.0


class RwallThermal:
    """Thermal Rigid Wall engine solver matching hm_read_rwall_therm.F and rgwalt.F."""

    def __init__(self, params: Union[RwallThermalParams, Dict[str, Any], Any]) -> None:
        if isinstance(params, RwallThermalParams):
            self.params = params
        elif isinstance(params, dict):
            self.params = RwallThermalParams(**params)
        else:
            # Extract from general object (e.g. RigidWall entity)
            p_dict: Dict[str, Any] = {}
            for k in (
                "id",
                "title",
                "slide",
                "fric",
                "dist",
                "t_wall0",
                "fscale_t",
                "fct_idt",
                "k_w",
                "cp_w",
                "rho_w",
                "k_struct",
                "cp_struct",
                "rho_struct",
                "f_struct",
                "thermal_resistance",
                "h_cont",
                "area_contact",
                "point",
                "normal",
                "grnod_id",
                "exclude_grnod_id",
            ):
                if hasattr(params, k):
                    p_dict[k] = getattr(params, k)
            self.params = RwallThermalParams(**p_dict)

        self.t_wall: float = float(self.params.t_wall0)
        self.node_temperatures: Dict[int, float] = {}

        # Cumulative energy ledgers [J]
        self.e_fric_total: float = 0.0
        self.e_th_struct: float = 0.0
        self.e_th_wall: float = 0.0
        self.e_cond_total: float = 0.0  # Net conduction from wall to structure

    def get_wall_temperature(
        self,
        t: float = 0.0,
        functs: Optional[Dict[int, Callable[[float], float]]] = None,
    ) -> float:
        """Compute the current rigid wall temperature at time t."""
        if (
            self.params.fct_idt is not None
            and functs is not None
            and self.params.fct_idt in functs
        ):
            f_val = functs[self.params.fct_idt](t)
            return float(self.params.fscale_t * f_val)
        return float(self.t_wall)

    def compute_frictional_power(
        self,
        f_normal: float,
        v_rel_tan: float,
        fric: Optional[float] = None,
    ) -> float:
        """Compute frictional dissipation power: P_fric = mu * |F_N| * |v_rel_tan|.

        If slide != 2 or mu <= 0, friction is inactive and returns 0.0.
        """
        if self.params.slide != 2:
            return 0.0
        mu = self.params.fric if fric is None else fric
        if mu <= 0.0:
            return 0.0
        fn_mag = abs(float(f_normal))
        vt_mag = abs(float(v_rel_tan))
        return float(mu * fn_mag * vt_mag)

    def compute_conductive_flux(
        self,
        t_node: float,
        t_wall: float,
        area: Optional[float] = None,
    ) -> float:
        """Compute conductive heat transfer rate: Q_cond = h_cont * Area * (T_wall - T_node)."""
        h = self.params.conductance
        if h <= 0.0:
            return 0.0
        a = self.params.area_contact if area is None else float(area)
        return float(h * a * (t_wall - t_node))

    def apply_contact_step(
        self,
        node_id: int,
        f_normal: float,
        v_rel_tan: float,
        node_mass: float,
        dt: float,
        t: float = 0.0,
        functs: Optional[Dict[int, Callable[[float], float]]] = None,
        cp_node: Optional[float] = None,
        area_node: Optional[float] = None,
        node_temp_init: Optional[float] = None,
    ) -> Dict[str, float]:
        """Process frictional contact heating and interface conduction for a single node.

        Returns a dictionary of all thermal quantities for the step.
        """
        if dt <= 0.0:
            return {
                "p_fric": 0.0,
                "e_fric": 0.0,
                "q_node": 0.0,
                "q_wall": 0.0,
                "q_cond": 0.0,
                "dt_node": 0.0,
                "t_node": self.node_temperatures.get(node_id, 293.15),
            }

        # Retrieve or initialize node temperature
        if node_id not in self.node_temperatures:
            init_t = node_temp_init if node_temp_init is not None else 293.15
            self.node_temperatures[node_id] = float(init_t)
        t_node = self.node_temperatures[node_id]

        t_w = self.get_wall_temperature(t=t, functs=functs)

        # 1. Frictional dissipation power and thermal partition
        p_fric = self.compute_frictional_power(f_normal, v_rel_tan)
        e_fric = p_fric * dt

        f_struct = self.params.partition_struct
        q_node_fric = f_struct * p_fric
        q_wall_fric = (1.0 - f_struct) * p_fric
        e_th_node = q_node_fric * dt
        e_th_wall = q_wall_fric * dt

        # 2. Interface conduction
        q_cond = self.compute_conductive_flux(t_node, t_w, area=area_node)
        e_cond = q_cond * dt

        # 3. Nodal temperature update
        cp = self.params.cp_struct if cp_node is None else float(cp_node)
        m = max(float(node_mass), EM20)
        q_node_net = q_node_fric + q_cond
        dt_node = (q_node_net * dt) / (m * cp)
        t_node_new = t_node + dt_node
        self.node_temperatures[node_id] = t_node_new

        # 4. Energy ledgers update
        self.e_fric_total += e_fric
        self.e_th_struct += e_th_node
        self.e_th_wall += e_th_wall
        self.e_cond_total += e_cond

        return {
            "p_fric": p_fric,
            "e_fric": e_fric,
            "q_node_fric": q_node_fric,
            "q_wall_fric": q_wall_fric,
            "q_cond": q_cond,
            "q_node_net": q_node_net,
            "dt_node": dt_node,
            "t_node": t_node_new,
            "t_wall": t_w,
            "e_th_node": e_th_node,
            "e_th_wall": e_th_wall,
        }

    def apply_thermal_coupling(
        self,
        node_indices: Sequence[int],
        f_normal_arr: Sequence[float],
        v_rel_tan_arr: Sequence[float],
        mass_arr: Sequence[float],
        dt: float,
        t: float = 0.0,
        functs: Optional[Dict[int, Callable[[float], float]]] = None,
        cp_arr: Optional[Sequence[float]] = None,
        area_arr: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, float]]:
        """Batch process frictional contact and conduction for multiple contacting nodes."""
        results = []
        n_nodes = len(node_indices)
        for idx in range(n_nodes):
            nid = int(node_indices[idx])
            fn = float(f_normal_arr[idx])
            vt = float(v_rel_tan_arr[idx])
            m = float(mass_arr[idx])
            cp_i = float(cp_arr[idx]) if cp_arr is not None else None
            area_i = float(area_arr[idx]) if area_arr is not None else None
            res = self.apply_contact_step(
                nid,
                fn,
                vt,
                m,
                dt,
                t=t,
                functs=functs,
                cp_node=cp_i,
                area_node=area_i,
            )
            results.append(res)
        return results

    def get_energy_summary(self) -> Dict[str, float]:
        """Return cumulative thermal energy balances."""
        return {
            "e_fric_total": self.e_fric_total,
            "e_th_struct": self.e_th_struct,
            "e_th_wall": self.e_th_wall,
            "e_cond_total": self.e_cond_total,
            "conservation_error": abs(
                self.e_fric_total - (self.e_th_struct + self.e_th_wall)
            ),
        }

    def reset(self) -> None:
        """Reset cumulative energy ledgers."""
        self.t_wall = float(self.params.t_wall0)
        self.node_temperatures.clear()
        self.e_fric_total = 0.0
        self.e_th_struct = 0.0
        self.e_th_wall = 0.0
        self.e_cond_total = 0.0
