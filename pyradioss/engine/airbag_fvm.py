"""pyradioss.engine.airbag_fvm — Finite Volume Airbag Physics Engine (/MONVOL/FVMBAG).

Implements complete multi-chamber finite volume airbag thermodynamics,
internal orifice communication (subcritical and choked sonic flow),
external venting (isenthalpic orifice and Wang-Nakhimovich fabric porosity),
gas injectors with temperature-dependent Cp polynomials and jetting,
and boundary surface pressure force assembly.

Fortran source references:
- ``engine/source/airbag/fvbag1.F``: Main FVM airbag scheme and flux balance.
- ``engine/source/airbag/fvbag2.F``: Multi-chamber state broadcast.
- ``engine/source/airbag/fvtemp.F``: Temperature inversion from specific energy.
- ``engine/source/airbag/fvinjt6.F``, ``fvinjt8.F``: Injector curves and enthalpies.
- ``engine/source/airbag/fvvent0.F``: Vent hole and porosity area calculations.
- ``engine/source/airbag/porfor5.F``: Autoliv / Wang-Nakhimovich fabric porosity.
- ``engine/source/airbag/airbagb1.F``: Multi-chamber orifice communication and choked flow.
- ``engine/source/airbag/volpfv.F``: Surface pressure force assembly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pyradioss.model.model import Model
from pyradioss.model.entities import (
    MonvolFvmbag,
    FvmChamber,
    FvmOrifice,
    FvmVent,
    FvmPorousSurface,
    FvmInjector,
)


# =============================================================================
# 1. Thermodynamics & Equation of State
# =============================================================================

def compute_cp(
    T: float,
    cpa: float,
    cpb: float = 0.0,
    cpc: float = 0.0,
    cpd: float = 0.0,
    cpe: float = 0.0,
    cpf: float = 0.0,
) -> float:
    """Evaluate specific heat at constant pressure Cp(T) from polynomial.
    
    Cp(T) = cpa + cpb*T + cpc*T^2 + cpd*T^3 + cpe/T^2 + cpf*T^4
    Matches OpenRadioss fvbag1.F / fvinjt8.F.
    """
    T = max(float(T), 1e-6)
    cp = cpa + cpb * T + cpc * (T ** 2) + cpd * (T ** 3) + cpf * (T ** 4)
    if cpe != 0.0:
        cp += cpe / (T ** 2)
    return max(cp, 1.0)


def compute_cv(
    T: float,
    cpa: float,
    cpb: float = 0.0,
    cpc: float = 0.0,
    cpd: float = 0.0,
    cpe: float = 0.0,
    cpf: float = 0.0,
    r_spec: float = 287.0,
) -> float:
    """Evaluate specific heat at constant volume Cv(T) = Cp(T) - R_spec."""
    cp = compute_cp(T, cpa, cpb, cpc, cpd, cpe, cpf)
    return max(cp - r_spec, 1.0)


def compute_gamma(
    T: float,
    cpa: float,
    cpb: float = 0.0,
    cpc: float = 0.0,
    cpd: float = 0.0,
    cpe: float = 0.0,
    cpf: float = 0.0,
    r_spec: float = 287.0,
) -> float:
    """Isentropic expansion coefficient gamma(T) = Cp(T) / Cv(T)."""
    cp = compute_cp(T, cpa, cpb, cpc, cpd, cpe, cpf)
    cv = max(cp - r_spec, 1.0)
    return cp / cv


def compute_specific_enthalpy(
    T: float,
    cpa: float,
    cpb: float = 0.0,
    cpc: float = 0.0,
    cpd: float = 0.0,
    cpe: float = 0.0,
    cpf: float = 0.0,
) -> float:
    """Specific enthalpy h(T) = int_0^T Cp(T') dT'.
    
    h(T) = cpa*T + 0.5*cpb*T^2 + (1/3)*cpc*T^3 + 0.25*cpd*T^4 - cpe/T + 0.2*cpf*T^5
    Matches fvinjt6.F / fvtemp.F.
    """
    T = max(float(T), 1e-6)
    h = (
        cpa * T
        + 0.5 * cpb * (T ** 2)
        + (1.0 / 3.0) * cpc * (T ** 3)
        + 0.25 * cpd * (T ** 4)
        + 0.2 * cpf * (T ** 5)
    )
    if cpe != 0.0:
        h -= cpe / T
    return h


def compute_specific_internal_energy(
    T: float,
    cpa: float,
    cpb: float = 0.0,
    cpc: float = 0.0,
    cpd: float = 0.0,
    cpe: float = 0.0,
    cpf: float = 0.0,
    r_spec: float = 287.0,
) -> float:
    """Specific internal energy e(T) = h(T) - R_spec*T."""
    T = max(float(T), 1e-6)
    return compute_specific_enthalpy(T, cpa, cpb, cpc, cpd, cpe, cpf) - r_spec * T


def solve_temperature(
    e_spec: float,
    cpa: float,
    cpb: float = 0.0,
    cpc: float = 0.0,
    cpd: float = 0.0,
    cpe: float = 0.0,
    cpf: float = 0.0,
    r_spec: float = 287.0,
    t_guess: float = 293.15,
) -> float:
    """Invert specific internal energy e_spec = e(T) for temperature T.
    
    Matches Fortran routine ``fvtemp.F``:
    - Constant Cv (cpb=cpc=cpd=cpe=cpf=0): exact algebraic inversion.
    - Linear Cp (cpc=cpd=cpe=cpf=0): quadratic root.
    - Polynomial Cp: Newton-Raphson iteration with fvtemp fixed-point fallback.
    """
    if e_spec <= 0.0:
        return 0.0

    cva = cpa - r_spec
    # Case 1: Pure constant Cp/Cv
    if cpb == 0.0 and cpc == 0.0 and cpd == 0.0 and cpe == 0.0 and cpf == 0.0:
        if cva > 0.0:
            return e_spec / cva
        return max(t_guess, 1.0)

    # Case 2: Quadratic internal energy (linear Cp)
    if cpc == 0.0 and cpd == 0.0 and cpe == 0.0 and cpf == 0.0:
        if cpb > 0.0:
            disc = max(0.0, cva * cva + 2.0 * cpb * e_spec)
            return (math.sqrt(disc) - cva) / cpb
        elif cva > 0.0:
            return e_spec / cva

    # Case 3: General polynomial (Newton-Raphson iteration)
    T = max(t_guess, 10.0)
    for _ in range(25):
        f = compute_specific_internal_energy(T, cpa, cpb, cpc, cpd, cpe, cpf, r_spec) - e_spec
        df = compute_cv(T, cpa, cpb, cpc, cpd, cpe, cpf, r_spec)
        if abs(df) < 1e-12:
            break
        delta = f / df
        T_new = max(T - delta, 1.0)
        if abs(T_new - T) < 1e-6 * max(T, 1.0):
            return T_new
        T = T_new

    # Fallback to fvtemp.F fixed-point iteration if Newton fails
    T = max(t_guess, 10.0)
    for _ in range(10):
        h_mean = (
            cpa
            + 0.5 * cpb * T
            + (1.0 / 3.0) * cpc * (T ** 2)
            + 0.25 * cpd * (T ** 3)
            + 0.2 * cpf * (T ** 4)
        )
        if cpe != 0.0:
            h_mean -= cpe / (T ** 2)
        denom = h_mean - r_spec
        if denom > 0.0:
            T = max(e_spec / denom, 1.0)
        else:
            break
    return T


# =============================================================================
# 2. Orifice & Venting Dynamics
# =============================================================================

def compute_orifice_flow(
    p1: float,
    rho1: float,
    t1: float,
    gamma1: float,
    r_spec1: float,
    cp_poly1: Tuple[float, float, float, float, float, float],
    p2: float,
    area: float,
    cd: float = 1.0,
    dt: float = 1e-3,
    vol1: float = 1.0,
    vol2: float = 1.0,
) -> Tuple[float, float, float, bool]:
    """Compute isentropic compressible orifice flow between two pressures P1 > P2.
    
    Implements the exact physics from OpenRadioss ``airbagb1.F`` (lines 140-190)
    and ``fvbag1.F`` (lines 950-1040):
    - Subcritical subsonic expansion when P2 > Pcrit.
    - Sonic choked throat flow when P2 <= Pcrit.
    - Numerical stability and volume transfer limiters.
    
    Returns:
        (mass_flow_rate, energy_flow_rate, velocity, is_choked)
    """
    if area <= 0.0 or p1 <= 0.0 or rho1 <= 0.0 or p1 <= p2:
        return 0.0, 0.0, 0.0, False

    gamma = max(gamma1, 1.001)
    eta = (gamma - 1.0) / gamma
    inv_eta = 1.0 / eta  # gamma / (gamma - 1)

    # Critical pressure ratio for sonic choking
    r_crit = (2.0 / (gamma + 1.0)) ** inv_eta
    p_crit = p1 * r_crit

    # Choked flow detection
    is_choked = (p2 <= p_crit)
    p2_eff = max(p2, p_crit)

    # Pressure expansion ratio
    rp = p2_eff / p1
    rp_eta = rp ** eta

    # Isentropic throat velocity: u = sqrt( 2*gamma/(gamma-1) * (P1/rho1) * (1 - (P2*/P1)^eta) )
    term = max(0.0, 1.0 - rp_eta)
    u_sq = (2.0 / eta) * (p1 / rho1) * term
    u = math.sqrt(max(0.0, u_sq))

    # Numerical stability limiters from airbagb1.F:
    # 1. Half-volume emptying bound
    if dt > 0.0:
        u_max1 = 0.5 * vol1 / max(1e-20, area * dt)
        u = min(u, u_max1)

    # Throat density: rho2* = rho1 * (P2*/P1)^(1/gamma)
    rho2_eff = rho1 * (rp ** (1.0 / gamma))

    # Mass flow rate
    mass_flow = cd * area * u * rho2_eff

    # Maximum mass flow stability limit: cannot transfer more than 50% mass in one dt
    if dt > 0.0:
        max_mass_flow = 0.5 * (rho1 * vol1) / dt
        mass_flow = min(mass_flow, max_mass_flow)

    # Enthalpy transported by flowing gas: h(T1)
    cpa, cpb, cpc, cpd, cpe, cpf = cp_poly1
    h1 = compute_specific_enthalpy(t1, cpa, cpb, cpc, cpd, cpe, cpf)
    energy_flow = mass_flow * h1

    return mass_flow, energy_flow, u, is_choked


def compute_vent_flow(
    p: float,
    rho: float,
    t: float,
    gamma: float,
    r_spec: float,
    cp_poly: Tuple[float, float, float, float, float, float],
    p_ext: float,
    area: float,
    cd: float = 1.0,
    dt: float = 1e-3,
    vol: float = 1.0,
) -> Tuple[float, float, float, bool]:
    """Compute external vent hole flow to atmosphere (Pext)."""
    return compute_orifice_flow(
        p1=p,
        rho1=rho,
        t1=t,
        gamma1=gamma,
        r_spec1=r_spec,
        cp_poly1=cp_poly,
        p2=p_ext,
        area=area,
        cd=cd,
        dt=dt,
        vol1=vol,
        vol2=1e30,
    )


def compute_wang_nakhimovich_porosity(
    eps1: float,
    eps2: float,
    tan_phi: float,
    p: float,
    p_ext: float,
    lr1: float,
    fthk: float,
    c1: float,
    c2: float,
    c3: float,
) -> float:
    """Autoliv / Wang-Nakhimovich fabric porosity formula.
    
    Matches OpenRadioss ``porfor5.F``:
    lambda1 = 1 + eps1, lambda2 = 1 + eps2, rs = lambda1 * lambda2
    Apor0 = (lr1 - fthk)^2
    Apor1 = (lr1*lambda1 - fthk/sqrt(lambda2)) * (lr1*lambda2 - fthk/sqrt(lambda1))
    deltaA = max(Apor1 - Apor0, 0)
    deltaP = max(P/Pext - 1, 0)
    cos_phi = 1 / sqrt(1 + tan_phi^2)
    SVTFAC = (c1 * Apor0 * deltaP^c2 + c3 * deltaA) * cos_phi / (rs * lr1^2)
    """
    lbd1 = 1.0 + eps1
    lbd2 = 1.0 + eps2
    rs = lbd1 * lbd2
    if rs <= 1.0 or lr1 <= 0.0 or p_ext <= 0.0:
        return 0.0

    delta_p = max(p / p_ext - 1.0, 0.0)
    apor0 = max(0.0, lr1 - fthk) ** 2
    term1 = max(0.0, lr1 * lbd1 - fthk / math.sqrt(max(lbd2, 1e-6)))
    term2 = max(0.0, lr1 * lbd2 - fthk / math.sqrt(max(lbd1, 1e-6)))
    apor1 = term1 * term2
    delta_a = max(apor1 - apor0, 0.0)

    cos_phi = 1.0 / math.sqrt(1.0 + tan_phi * tan_phi)

    p_term = (delta_p ** c2) if delta_p > 0.0 and c2 > 0.0 else (1.0 if c2 == 0.0 else 0.0)
    svtfac = (c1 * apor0 * p_term + c3 * delta_a) * cos_phi / (rs * lr1 * lr1)
    return max(0.0, svtfac)


# =============================================================================
# 3. Geometry & Surface Integrals
# =============================================================================

def compute_surface_geometry(
    surf: Any, x: np.ndarray
) -> Tuple[float, float, List[np.ndarray], List[float], List[np.ndarray]]:
    """Compute enclosed volume, surface area, facet normal area vectors,
    facet areas, and facet centroid coordinates from current nodal coordinates x.
    
    Uses Gauss divergence theorem matching ``fvmesh0.F`` and ``get_volume_area.F90``.
    Supports 3-node triangles and 4-node quadrilaterals.
    """
    if surf is None or surf.segments is None or len(surf.segments) == 0 or x is None or len(x) == 0:
        return 0.0, 0.0, [], [], []

    total_vol = 0.0
    total_area = 0.0
    facet_normals = []
    facet_areas = []
    facet_centroids = []
    n_nodes = len(x)

    for seg in surf.segments:
        n1 = seg[0]
        n2 = seg[1]
        n3 = seg[2]
        is_tri = (len(seg) < 4) or (seg[3] == n3) or (seg[3] < 0)

        if n1 >= n_nodes or n2 >= n_nodes or n3 >= n_nodes or n1 < 0 or n2 < 0 or n3 < 0:
            continue

        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]

        if not is_tri:
            n4 = seg[3]
            if n4 >= n_nodes or n4 < 0:
                continue
            x4 = x[n4]
            v31 = x3 - x1
            v42 = x4 - x2
            xn = 0.5 * np.cross(v31, v42)
            xc = 0.25 * (x1 + x2 + x3 + x4)
            vol_facet = np.dot(xc, xn) / 3.0
        else:
            v21 = x2 - x1
            v31 = x3 - x1
            xn = 0.5 * np.cross(v21, v31)
            xc = (x1 + x2 + x3) / 3.0
            vol_facet = np.dot(xc, xn) / 3.0

        area_facet = float(np.linalg.norm(xn))
        total_vol += float(vol_facet)
        total_area += area_facet
        facet_normals.append(xn)
        facet_areas.append(area_facet)
        facet_centroids.append(xc)

    return total_vol, total_area, facet_normals, facet_areas, facet_centroids


# =============================================================================
# 4. Multi-Chamber FVM Airbag Manager
# =============================================================================

class FvmAirbagManager:
    """Engine manager for /MONVOL/FVMBAG1 and /MONVOL/FVMBAG2 airbags.
    
    Coordinates:
    - Multi-chamber thermodynamics (mass, energy, pressure, temperature).
    - Gas injectors (mass flow, temperature, jet momentum).
    - Internal orifice communications between chambers (choked sonic flow).
    - External vent holes and fabric porosity (Wang-Nakhimovich).
    - Application of surface pressure forces to finite element nodes.
    """

    @staticmethod
    def initialize(mv: MonvolFvmbag, model: Model) -> None:
        """Starter initialization for FVM airbag chambers, orifices, and vents."""
        if getattr(mv, "is_initialized", False):
            return

        if hasattr(model, "x0") and model.x0 is not None and len(model.x0) > 0:
            x0 = model.x0
        elif hasattr(model, "x") and model.x is not None and len(model.x) > 0:
            x0 = model.x
        elif hasattr(model, "nodes") and model.nodes:
            max_nid = max(model.nodes.keys())
            x0 = np.zeros((max_nid + 1, 3), dtype=np.float64)
            for nid, node in model.nodes.items():
                x0[nid] = (node.x, node.y, node.z)
        else:
            x0 = np.zeros((0, 3), dtype=np.float64)

        # 1. Setup chambers
        # If chambers not explicitly pre-populated, construct default chambers:
        if not mv.chambers:
            # Chamber 1: Main / External chamber
            c1 = FvmChamber(
                id=1,
                surf_id=mv.surf_id_ex or mv.surf_id,
                mat_id=mv.mat_id,
                pext=mv.pext,
                t_initial=mv.t_initial or mv.t0 or 293.15,
                iequil=mv.iequil,
            )
            mv.chambers[1] = c1

            # Chamber 2: Internal / Secondary chamber if surf_id_in is specified
            if mv.surf_id_in > 0:
                c2 = FvmChamber(
                    id=2,
                    surf_id=mv.surf_id_in,
                    mat_id=mv.mat_id,
                    pext=mv.pext,
                    t_initial=mv.t_initial or mv.t0 or 293.15,
                    iequil=mv.iequil,
                )
                mv.chambers[2] = c2

                # Setup default inter-chamber orifice connecting Chamber 1 and 2
                if not mv.orifices:
                    surf_in = model.surfaces.get(mv.surf_id_in)
                    v_in, a_in, _, _, _ = compute_surface_geometry(surf_in, x0)
                    mv.orifices.append(
                        FvmOrifice(
                            id=1,
                            surf_id=mv.surf_id_in,
                            chamber1_id=1,
                            chamber2_id=2,
                            area=a_in,
                            cd=0.8,
                            is_open=True,
                            title="Chamber 1-2 Communication",
                        )
                    )

        # 2. Initialize each chamber's material, geometry, and thermodynamic state
        for ch_id, ch in mv.chambers.items():
            # Material properties
            mat = model.materials.get(ch.mat_id)
            if mat is not None:
                params = getattr(mat, "params", {})
                ch.cpa = float(params.get("CPA", getattr(mat, "cpa", 1004.0)))
                ch.cpb = float(params.get("CPB", getattr(mat, "cpb", 0.0)))
                ch.cpc = float(params.get("CPC", getattr(mat, "cpc", 0.0)))
                ch.cpd = float(params.get("CPD", getattr(mat, "cpd", 0.0)))
                ch.cpe = float(params.get("CPE", getattr(mat, "cpe", 0.0)))
                ch.cpf = float(params.get("CPF", getattr(mat, "cpf", 0.0)))
                ch.r_spec = float(getattr(mat, "r_spec", params.get("R_igc", 287.0) / max(params.get("MW", 1.0), 1e-6)))
            else:
                ch.cpa = 1004.0
                ch.cpb = 0.0
                ch.cpc = 0.0
                ch.r_spec = 287.0

            ch.gamma = compute_gamma(ch.t_initial, ch.cpa, ch.cpb, ch.cpc, ch.cpd, ch.cpe, ch.cpf, ch.r_spec)

            # Geometry
            surf = model.surfaces.get(ch.surf_id)
            v0, a0, _, _, _ = compute_surface_geometry(surf, x0)
            ch.volume = max(v0, 1e-9)
            ch.volume_old = ch.volume
            ch.area = a0

            # Initial thermodynamic equilibrium state
            if ch.mass <= 0.0:
                if ch.iequil == 0 or ch.iequil == -1:
                    if ch.pext > 0.0 and ch.r_spec > 0.0:
                        ch.mass = ch.pext * ch.volume / (ch.r_spec * ch.t_initial)
                    else:
                        rho0 = getattr(mat, "rho0", 1.0) if mat else 1.0
                        ch.mass = rho0 * ch.volume
                else:
                    ch.mass = 1e-12

            ch.temperature = ch.t_initial
            ch.pressure = ch.pext
            ch.density = ch.mass / ch.volume
            e_spec = compute_specific_internal_energy(
                ch.temperature, ch.cpa, ch.cpb, ch.cpc, ch.cpd, ch.cpe, ch.cpf, ch.r_spec
            )
            ch.energy = ch.mass * e_spec

        # 3. Initialize vents
        for vent in mv.vents:
            if vent.area <= 0.0:
                if vent.surf_id > 0 and vent.surf_id in model.surfaces:
                    surf_v = model.surfaces[vent.surf_id]
                    _, a_v, _, _, _ = compute_surface_geometry(surf_v, x0)
                    vent.area = a_v * vent.avent
                elif vent.avent > 0.0:
                    vent.area = vent.avent

        # 4. Set global airbag aggregates
        mv.volume = sum(c.volume for c in mv.chambers.values())
        mv.mass = sum(c.mass for c in mv.chambers.values())
        mv.energy = sum(c.energy for c in mv.chambers.values())
        mv.temperature = mv.chambers[1].temperature if 1 in mv.chambers else mv.t_initial
        mv.pressure = mv.chambers[1].pressure if 1 in mv.chambers else mv.pext
        mv.is_initialized = True

    @staticmethod
    def update_geometry(mv: MonvolFvmbag, model: Model, x: np.ndarray) -> None:
        """Update volumes and surface areas of all airbag chambers."""
        if not getattr(mv, "is_initialized", False):
            FvmAirbagManager.initialize(mv, model)

        for ch in mv.chambers.values():
            ch.volume_old = ch.volume
            surf = model.surfaces.get(ch.surf_id)
            if surf is not None:
                v, a, _, _, _ = compute_surface_geometry(surf, x)
                ch.volume = max(v, 1e-9)
                ch.area = a

        mv.volume = sum(c.volume for c in mv.chambers.values())

    @staticmethod
    def step_thermodynamics(
        mv: MonvolFvmbag, model: Model, dt: float, current_time: float
    ) -> None:
        """Integrate conservation of mass and energy for each chamber over time step dt."""
        if not getattr(mv, "is_initialized", False):
            FvmAirbagManager.initialize(mv, model)

        dt = max(float(dt), 1e-12)

        # Track increments per chamber: delta_m, delta_E, delta_W, delta_Q
        delta_m = {ch_id: 0.0 for ch_id in mv.chambers}
        delta_e = {ch_id: 0.0 for ch_id in mv.chambers}
        delta_w = {ch_id: 0.0 for ch_id in mv.chambers}

        # Track gas mixture contributions per chamber: sum(m_in * prop)
        mix_cpa = {ch_id: 0.0 for ch_id in mv.chambers}
        mix_cpb = {ch_id: 0.0 for ch_id in mv.chambers}
        mix_cpc = {ch_id: 0.0 for ch_id in mv.chambers}
        mix_rmw = {ch_id: 0.0 for ch_id in mv.chambers}

        # ---------------------------------------------------------------------
        # A. Gas Injectors
        # ---------------------------------------------------------------------
        for inj in mv.injectors:
            # Check sensor if specified
            if inj.sens_id > 0 and hasattr(model, "sensors") and inj.sens_id in model.sensors:
                sens = model.sensors[inj.sens_id]
                if not getattr(sens, "is_active", True):
                    continue

            target_ch_id = inj.chamber_id if inj.chamber_id in mv.chambers else 1
            ch = mv.chambers[target_ch_id]

            # Determine mass flow rate
            if inj.fct_mass > 0 and hasattr(model, "functions") and inj.fct_mass in model.functions:
                f_m = model.functions[inj.fct_mass]
                m_dot = float(f_m.eval(current_time)) * inj.scale_mass
            else:
                m_dot = inj.mass_flow

            # Determine injection temperature
            if inj.fct_temp > 0 and hasattr(model, "functions") and inj.fct_temp in model.functions:
                f_t = model.functions[inj.fct_temp]
                t_inj = float(f_t.eval(current_time)) * inj.scale_temp
            else:
                t_inj = inj.temperature

            if m_dot > 0.0:
                dm_inj = m_dot * dt
                h_inj = compute_specific_enthalpy(
                    t_inj, inj.cpa, inj.cpb, inj.cpc, inj.cpd, inj.cpe, inj.cpf
                )
                de_inj = dm_inj * h_inj

                delta_m[target_ch_id] += dm_inj
                delta_e[target_ch_id] += de_inj

                mix_cpa[target_ch_id] += dm_inj * inj.cpa
                mix_cpb[target_ch_id] += dm_inj * inj.cpb
                mix_cpc[target_ch_id] += dm_inj * inj.cpc
                mix_rmw[target_ch_id] += dm_inj * inj.r_spec

        # ---------------------------------------------------------------------
        # B. Internal Orifices Between Chambers
        # ---------------------------------------------------------------------
        for orif in mv.orifices:
            if not orif.is_open or current_time < orif.tstart or current_time > orif.tstop:
                continue

            c1_id = orif.chamber1_id
            c2_id = orif.chamber2_id
            if c1_id not in mv.chambers or c2_id not in mv.chambers:
                continue

            ch1 = mv.chambers[c1_id]
            ch2 = mv.chambers[c2_id]

            # Check membrane burst pressure if specified
            if orif.pdef > 0.0:
                if abs(ch1.pressure - ch2.pressure) < orif.pdef:
                    continue

            # Identify upstream and downstream chambers
            if ch1.pressure >= ch2.pressure:
                up_id, down_id = c1_id, c2_id
                up_ch, down_ch = ch1, ch2
            else:
                up_id, down_id = c2_id, c1_id
                up_ch, down_ch = ch2, ch1

            cp_poly_up = (
                up_ch.cpa, up_ch.cpb, up_ch.cpc, up_ch.cpd, up_ch.cpe, up_ch.cpf
            )
            m_dot, e_dot, u_orif, is_choked = compute_orifice_flow(
                p1=up_ch.pressure,
                rho1=up_ch.density,
                t1=up_ch.temperature,
                gamma1=up_ch.gamma,
                r_spec1=up_ch.r_spec,
                cp_poly1=cp_poly_up,
                p2=down_ch.pressure,
                area=orif.area,
                cd=orif.cd,
                dt=dt,
                vol1=up_ch.volume,
                vol2=down_ch.volume,
            )

            dm = m_dot * dt
            de = e_dot * dt

            delta_m[up_id] -= dm
            delta_e[up_id] -= de

            delta_m[down_id] += dm
            delta_e[down_id] += de

            # Mixing into downstream chamber
            mix_cpa[down_id] += dm * up_ch.cpa
            mix_cpb[down_id] += dm * up_ch.cpb
            mix_cpc[down_id] += dm * up_ch.cpc
            mix_rmw[down_id] += dm * up_ch.r_spec

        # ---------------------------------------------------------------------
        # C. External Vents
        # ---------------------------------------------------------------------
        for vent in mv.vents:
            if not vent.is_open or current_time < vent.tstart or current_time > vent.tstop:
                continue

            target_ch_id = vent.chamber_id if vent.chamber_id in mv.chambers else 1
            ch = mv.chambers[target_ch_id]

            # Check membrane burst pressure
            if vent.dpdef > 0.0 and (ch.pressure - ch.pext) < vent.dpdef:
                continue

            # Determine dynamic vent area scaling factors
            a_eff = vent.area
            if vent.fct_t > 0 and hasattr(model, "functions") and vent.fct_t in model.functions:
                a_eff *= float(model.functions[vent.fct_t].eval(current_time)) * vent.fscale_t
            if vent.fct_p > 0 and hasattr(model, "functions") and vent.fct_p in model.functions:
                dp = ch.pressure - ch.pext
                a_eff *= float(model.functions[vent.fct_p].eval(dp)) * vent.fscale_p

            if a_eff <= 0.0 or ch.pressure <= ch.pext:
                continue

            cp_poly = (ch.cpa, ch.cpb, ch.cpc, ch.cpd, ch.cpe, ch.cpf)
            m_dot, e_dot, _, _ = compute_vent_flow(
                p=ch.pressure,
                rho=ch.density,
                t=ch.temperature,
                gamma=ch.gamma,
                r_spec=ch.r_spec,
                cp_poly=cp_poly,
                p_ext=ch.pext,
                area=a_eff,
                cd=vent.cd,
                dt=dt,
                vol=ch.volume,
            )

            dm_vent = m_dot * dt
            de_vent = e_dot * dt

            delta_m[target_ch_id] -= dm_vent
            delta_e[target_ch_id] -= de_vent

        # ---------------------------------------------------------------------
        # D. Boundary Work & Convection Heat Loss
        # ---------------------------------------------------------------------
        for ch_id, ch in mv.chambers.items():
            dv = ch.volume - ch.volume_old
            # P*dV work done by expanding gas
            p_work = 0.5 * (ch.pressure + ch.pext) * dv if dv != 0.0 else 0.0
            delta_w[ch_id] = p_work

            # Convection loss
            q_conv = 0.0
            if mv.hconv > 0.0:
                q_conv = mv.hconv * ch.area * (ch.temperature - ch.t_initial) * dt

            delta_e[ch_id] -= (p_work + q_conv)

        # ---------------------------------------------------------------------
        # E. Update State of Each Chamber
        # ---------------------------------------------------------------------
        for ch_id, ch in mv.chambers.items():
            m_old = ch.mass
            m_new = max(m_old + delta_m[ch_id], 1e-12)
            e_new = max(ch.energy + delta_e[ch_id], 1e-12)

            # Update gas mixture parameters if new gas arrived
            m_in = sum(v for k, v in delta_m.items() if k == ch_id and v > 0)
            if mix_cpa[ch_id] > 0.0:
                ch.cpa = (m_old * ch.cpa + mix_cpa[ch_id]) / m_new
                ch.cpb = (m_old * ch.cpb + mix_cpb[ch_id]) / m_new
                ch.cpc = (m_old * ch.cpc + mix_cpc[ch_id]) / m_new
                ch.r_spec = (m_old * ch.r_spec + mix_rmw[ch_id]) / m_new

            ch.mass = m_new
            ch.energy = e_new
            ch.density = ch.mass / max(ch.volume, 1e-9)

            # Solve for new temperature
            e_spec = ch.energy / ch.mass
            ch.temperature = solve_temperature(
                e_spec,
                ch.cpa,
                ch.cpb,
                ch.cpc,
                ch.cpd,
                ch.cpe,
                ch.cpf,
                ch.r_spec,
                t_guess=ch.temperature,
            )

            # Update thermodynamic properties
            ch.gamma = compute_gamma(
                ch.temperature,
                ch.cpa,
                ch.cpb,
                ch.cpc,
                ch.cpd,
                ch.cpe,
                ch.cpf,
                ch.r_spec,
            )

            # Equation of state: P = rho * R_spec * T
            ch.pressure = ch.density * ch.r_spec * ch.temperature

        # ---------------------------------------------------------------------
        # F. Update Global Airbag Quantities
        # ---------------------------------------------------------------------
        mv.mass = sum(c.mass for c in mv.chambers.values())
        mv.energy = sum(c.energy for c in mv.chambers.values())
        if 1 in mv.chambers:
            mv.pressure = mv.chambers[1].pressure
            mv.temperature = mv.chambers[1].temperature
        else:
            first_c = next(iter(mv.chambers.values()))
            mv.pressure = first_c.pressure
            mv.temperature = first_c.temperature

    @staticmethod
    def apply_forces(
        mv: MonvolFvmbag, model: Model, x: np.ndarray, f: np.ndarray
    ) -> None:
        """Assemble surface pressure forces onto finite element nodes.
        
        Matches OpenRadioss ``volpfv.F`` and ``fvbag1.F`` (lines 2249-2278):
        - External surfaces of Chamber k: P_net = P_k - P_ext.
        - Internal partition surfaces between Chamber 1 and 2: P_net = P1 - P2.
        - Nodal distribution:
            1/3 of facet normal force to each node of a triangle.
            1/4 of facet normal force to each node of a quad.
        """
        if not getattr(mv, "is_initialized", False):
            FvmAirbagManager.initialize(mv, model)

        # 1. External chamber surface forces
        for ch_id, ch in mv.chambers.items():
            surf = model.surfaces.get(ch.surf_id)
            if surf is None or surf.segments is None or len(surf.segments) == 0:
                continue

            p_eff = ch.pressure - ch.pext
            if abs(p_eff) < 1e-12:
                continue

            for seg in surf.segments:
                n1 = seg[0]
                n2 = seg[1]
                n3 = seg[2]
                is_tri = (len(seg) < 4) or (seg[3] == n3) or (seg[3] < 0)

                x1 = x[n1]
                x2 = x[n2]
                x3 = x[n3]

                if not is_tri:
                    n4 = seg[3]
                    x4 = x[n4]
                    v31 = x3 - x1
                    v42 = x4 - x2
                    xn = 0.5 * np.cross(v31, v42)
                    fn = 0.25 * p_eff * xn
                    f[n1] += fn
                    f[n2] += fn
                    f[n3] += fn
                    f[n4] += fn
                else:
                    v21 = x2 - x1
                    v31 = x3 - x1
                    xn = 0.5 * np.cross(v21, v31)
                    fn = (1.0 / 3.0) * p_eff * xn
                    f[n1] += fn
                    f[n2] += fn
                    f[n3] += fn

        # 2. Internal partition / orifice surface forces
        if mv.surf_id_in > 0 and 1 in mv.chambers and 2 in mv.chambers:
            surf_in = model.surfaces.get(mv.surf_id_in)
            if surf_in is not None and surf_in.segments is not None:
                p1 = mv.chambers[1].pressure
                p2 = mv.chambers[2].pressure
                delta_p = p1 - p2
                if abs(delta_p) >= 1e-12:
                    for seg in surf_in.segments:
                        n1 = seg[0]
                        n2 = seg[1]
                        n3 = seg[2]
                        is_tri = (len(seg) < 4) or (seg[3] == n3) or (seg[3] < 0)

                        x1 = x[n1]
                        x2 = x[n2]
                        x3 = x[n3]

                        if not is_tri:
                            n4 = seg[3]
                            x4 = x[n4]
                            v31 = x3 - x1
                            v42 = x4 - x2
                            xn = 0.5 * np.cross(v31, v42)
                            fn = 0.25 * delta_p * xn
                            f[n1] += fn
                            f[n2] += fn
                            f[n3] += fn
                            f[n4] += fn
                        else:
                            v21 = x2 - x1
                            v31 = x3 - x1
                            xn = 0.5 * np.cross(v21, v31)
                            fn = (1.0 / 3.0) * delta_p * xn
                            f[n1] += fn
                            f[n2] += fn
                            f[n3] += fn

        # 3. Injector jetting reaction forces
        for inj in mv.injectors:
            apply_injector_jetting_forces(inj, f, model)


# Top-level functional interface matching airbag.py
def update_fvmbag_volume(mv: MonvolFvmbag, model: Model, x: np.ndarray) -> None:
    """Update volume of FVM airbag."""
    FvmAirbagManager.update_geometry(mv, model, x)


def update_fvmbag_thermodynamics(
    mv: MonvolFvmbag, model: Model, dt: float, current_time: float
) -> None:
    """Update thermodynamic state of FVM airbag."""
    FvmAirbagManager.step_thermodynamics(mv, model, dt, current_time)


def apply_fvmbag_forces(
    mv: MonvolFvmbag, model: Model, x: np.ndarray, f: np.ndarray
) -> None:
    """Apply FVM airbag pressure forces to nodal force array."""
    FvmAirbagManager.apply_forces(mv, model, x, f)


def apply_injector_jetting_forces(
    injector: FvmInjector, fext: np.ndarray, model: Model
) -> None:
    """Project injector jetting recoil / thrust reaction forces onto finite element nodes.

    Upstream Fortran:
    - engine/source/airbag/fvbag1.F (lines 904-933): momentum flux DQI = -DMI * V_jet * n.
    - engine/source/airbag/fvinjt6.F (lines 70-131): mass flow and gas velocity.
    - engine/source/airbag/volpres.F (lines 216-332): force distribution on surface facets.

    Reaction force on nozzle boundary has magnitude F_thrust = m_dot * v_jet,
    directed along the outward surface normal n (opposite to the injected jet velocity).
    """
    if injector.sens_id > 0 and hasattr(model, "sensors") and injector.sens_id in model.sensors:
        sens = model.sensors[injector.sens_id]
        if not getattr(sens, "is_active", True):
            return

    current_time = getattr(model, "t", getattr(model, "current_time", 0.0))

    # Mass flow rate m_dot
    if injector.fct_mass > 0 and hasattr(model, "functions") and injector.fct_mass in model.functions:
        f_m = model.functions[injector.fct_mass]
        m_dot = float(f_m.eval(current_time)) * injector.scale_mass
    else:
        m_dot = injector.mass_flow

    # Jet velocity v_jet
    if injector.fct_vel > 0 and hasattr(model, "functions") and injector.fct_vel in model.functions:
        f_v = model.functions[injector.fct_vel]
        v_jet = float(f_v.eval(current_time)) * injector.scale_vel
    else:
        v_jet = injector.scale_vel

    if m_dot <= 0.0 or v_jet <= 0.0:
        return

    f_thrust = m_dot * v_jet

    # Check if injector has associated surface
    surf = model.surfaces.get(injector.surf_id) if hasattr(model, "surfaces") else None
    x = getattr(model, "x", getattr(model, "x0", None))

    if surf is not None and surf.segments is not None and len(surf.segments) > 0 and x is not None:
        seg_normals = []
        total_area = 0.0
        for seg in surf.segments:
            n1, n2, n3 = seg[0], seg[1], seg[2]
            is_tri = (len(seg) < 4) or (seg[3] == n3) or (seg[3] < 0)
            x1, x2, x3 = x[n1], x[n2], x[n3]
            v31 = x3 - x1
            if not is_tri:
                n4 = seg[3]
                x4 = x[n4]
                v42 = x4 - x2
                xn = 0.5 * np.cross(v31, v42)
            else:
                v21 = x2 - x1
                xn = 0.5 * np.cross(v21, v31)
            area_seg = float(np.linalg.norm(xn))
            total_area += area_seg
            seg_normals.append((seg, is_tri, xn, area_seg))

        if total_area > 1e-15:
            p_jet = f_thrust / total_area
            for seg, is_tri, xn, area_seg in seg_normals:
                if not is_tri:
                    n1, n2, n3, n4 = seg[0], seg[1], seg[2], seg[3]
                    fn = 0.25 * p_jet * xn
                    fext[n1] += fn
                    fext[n2] += fn
                    fext[n3] += fn
                    fext[n4] += fn
                else:
                    n1, n2, n3 = seg[0], seg[1], seg[2]
                    fn = (1.0 / 3.0) * p_jet * xn
                    fext[n1] += fn
                    fext[n2] += fn
                    fext[n3] += fn
    else:
        n_vec = np.array(getattr(injector, "normal", (0.0, 0.0, 1.0)), dtype=float)
        norm_val = np.linalg.norm(n_vec)
        if norm_val > 1e-12:
            n_vec = n_vec / norm_val
        force_vec = f_thrust * n_vec

        nodes = None
        if surf is not None:
            if getattr(surf, "nodes", None) is not None and len(surf.nodes) > 0:
                nodes = surf.nodes
            elif getattr(surf, "segments", None) is not None and len(surf.segments) > 0:
                nodes = np.unique(surf.segments)
        if nodes is not None and len(nodes) > 0:
            fn_node = force_vec / len(nodes)
            for nid in nodes:
                if 0 <= nid < len(fext):
                    fext[nid] += fn_node


# =============================================================================
# 7. Shock-Capturing Upwind Flux Limiter (from fv_up_switch.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/fv_up_switch.F: lines 905-1008
def compute_upwind_face_flux(
    rho1: float,
    re1: float,
    u1: np.ndarray,
    gamma1: float,
    cp_poly1: Tuple[float, float, float, float, float, float],
    r_spec1: float,
    rho2: float,
    re2: float,
    u2: np.ndarray,
    gamma2: float,
    cp_poly2: Tuple[float, float, float, float, float, float],
    r_spec2: float,
    normal: np.ndarray,
    area: float,
    v_grid: np.ndarray = np.zeros(3),
    porosity: float = 1.0,
) -> Tuple[float, np.ndarray, float, Tuple[float, float, float, float, float, float], float]:
    """Compute shock-capturing upwind advective fluxes across a cell face.

    Matches OpenRadioss ``fv_up_switch.F`` (lines 905-1008):
    Computes relative velocity between gas and moving boundary mesh:
    V_gas = 0.5*(u1 + u2)
    V_rel = V_gas - V_grid
    ss_ = dot(normal, V_rel)
    alpha = 1.0 if ss_ > 0 else 0.0 (upwind direction)
    rho_m = alpha*rho1 + (1-alpha)*rho2
    rem = alpha*gamma1*re1 + (1-alpha)*gamma2*re2

    Args:
        rho1: Density of left cell (+normal).
        re1: Volumetric internal energy (E/V) of left cell.
        u1: (3,) velocity of left cell.
        gamma1: Ratio of specific heats for left cell.
        cp_poly1: (cpa, cpb, cpc, cpd, cpe, cpf) for left cell.
        r_spec1: Gas constant for left cell.
        rho2: Density of right cell (-normal).
        re2: Volumetric internal energy (E/V) of right cell.
        u2: (3,) velocity of right cell.
        gamma2: Ratio of specific heats for right cell.
        cp_poly2: (cpa, cpb, cpc, cpd, cpe, cpf) for right cell.
        r_spec2: Gas constant for right cell.
        normal: (3,) unit outward normal from cell 1 to cell 2.
        area: Geometric surface area of the face.
        v_grid: (3,) velocity of the face mesh (grid velocity).
        porosity: Effective porosity ratio (0 to 1).

    Returns:
        (mass_flux, momentum_flux, energy_flux, cp_flux, rgas_flux):
            - mass_flux: kg/s transferred across face.
            - momentum_flux: (3,) N momentum vector transferred.
            - energy_flux: W total enthalpy/energy transferred.
            - cp_flux: 6-tuple of Cp polynomial fluxes.
            - rgas_flux: Gas constant flux.
    """
    if area <= 0.0 or porosity <= 0.0:
        return 0.0, np.zeros(3), 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0

    eff_area = area * max(0.0, min(1.0, porosity))

    # Gas velocity at face: average of left and right cells
    v_gas = 0.5 * (u1 + u2)
    # Relative velocity: v_gas - v_grid (fv_up_switch.F line 964)
    v_rel = v_gas - v_grid
    ss_ = float(np.dot(normal, v_rel))

    # Directional upwind switch: alpha = 1 if flowing 1 -> 2, 0 if flowing 2 -> 1
    alpha = 1.0 if ss_ > 0.0 else 0.0

    # Upwind state reconstruction
    rho_m = alpha * rho1 + (1.0 - alpha) * rho2
    ux_m = alpha * rho1 * u1[0] + (1.0 - alpha) * rho2 * u2[0]
    uy_m = alpha * rho1 * u1[1] + (1.0 - alpha) * rho2 * u2[1]
    uz_m = alpha * rho1 * u1[2] + (1.0 - alpha) * rho2 * u2[2]
    re_m = alpha * gamma1 * re1 + (1.0 - alpha) * gamma2 * re2

    # Fluxes: MASSFLOW = rho_m * ss_ * eff_area
    mass_flow = rho_m * ss_ * eff_area
    mom_flux = np.array([ux_m, uy_m, uz_m]) * ss_ * eff_area
    energy_flux = re_m * ss_ * eff_area

    # Cp polynomial interpolation
    cpa_m = alpha * cp_poly1[0] + (1.0 - alpha) * cp_poly2[0]
    cpb_m = alpha * cp_poly1[1] + (1.0 - alpha) * cp_poly2[1]
    cpc_m = alpha * cp_poly1[2] + (1.0 - alpha) * cp_poly2[2]
    cpd_m = alpha * cp_poly1[3] + (1.0 - alpha) * cp_poly2[3]
    cpe_m = alpha * cp_poly1[4] + (1.0 - alpha) * cp_poly2[4]
    cpf_m = alpha * cp_poly1[5] + (1.0 - alpha) * cp_poly2[5]
    rgas_m = alpha * r_spec1 + (1.0 - alpha) * r_spec2

    cp_flux = (
        mass_flow * cpa_m,
        mass_flow * cpb_m,
        mass_flow * cpc_m,
        mass_flow * cpd_m,
        mass_flow * cpe_m,
        mass_flow * cpf_m,
    )
    rgas_flux = mass_flow * rgas_m

    return float(mass_flow), mom_flux, float(energy_flux), cp_flux, float(rgas_flux)


# =============================================================================
# 8. Artificial Gas Bulk Viscosity & CFL Time Step (from fv_up_switch.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/fv_up_switch.F: lines 1208-1220, 1255-1262
def compute_gas_viscosity_and_cfl(
    volume: float,
    mass: float,
    energy: float,
    gamma: float,
    dm: float,
    dt: float,
    u_gas: np.ndarray,
    char_length: float,
    qa: float = 1.5,
    qb: float = 0.06,
    cfl_coef: float = 0.9,
    u_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute artificial gas bulk viscosity QVISC and CFL-stable time step.

    Matches OpenRadioss ``fv_up_switch.F``:
    AL = VOLU^(1/3)
    DD = max(0, DM / MASS)
    SSP = sqrt((gamma - 1)*gamma * ENERGY / MASS)
    QVISC = RHO * DD * AL * (QA^2 * AL * DD + QB * SSP)
    QX = QB * SSP + AL * QA^2 * DD
    SSP_eff = QX + sqrt(QX^2 + SSP^2)
    V_rel = |u_gas - u_grid|
    DT_CFL = CFL_COEF * char_length / (SSP_eff + V_rel)

    Args:
        volume: Polyhedron volume.
        mass: Gas mass in cell.
        energy: Internal energy in cell.
        gamma: Ratio of specific heats.
        dm: Mass increment / change during step.
        dt: Current time step.
        u_gas: (3,) Gas velocity vector in cell.
        char_length: Characteristic length of cell.
        qa: Quadratic bulk viscosity coefficient (default 1.5).
        qb: Linear bulk viscosity coefficient (default 0.06).
        cfl_coef: Courant number multiplier (default 0.9).
        u_grid: Optional (3,) grid velocity vector.

    Returns:
        (q_visc, ssp_eff, dt_cfl): Artificial viscosity pressure, effective wave speed, and CFL time step.
    """
    if volume <= 0.0 or mass <= 0.0 or energy <= 0.0 or gamma <= 1.0:
        return 0.0, 340.0, 1e-3

    rho = mass / volume
    al = volume ** (1.0 / 3.0)
    dd = max(0.0, abs(dm) / mass)

    # Sound speed: c_s = sqrt((gamma - 1) * gamma * E / m)
    ssp_sq = max(1e-6, (gamma - 1.0) * gamma * energy / mass)
    ssp = math.sqrt(ssp_sq)

    # Artificial bulk viscosity: QVISC = rho * DD * AL * (QA^2 * AL * DD + QB * SSP)
    q_visc = rho * dd * al * ((qa ** 2) * al * dd + qb * ssp)

    # Wave speed shift with artificial damping
    qx = qb * ssp + al * (qa ** 2) * dd
    ssp_eff = qx + math.sqrt(qx * qx + ssp * ssp)

    # Relative velocity
    if u_grid is not None:
        v_rel = float(np.linalg.norm(u_gas - u_grid))
    else:
        v_rel = float(np.linalg.norm(u_gas))

    # CFL time step bound
    denom = max(1e-12, ssp_eff + v_rel)
    dt_cfl = cfl_coef * char_length / denom

    return float(q_visc), float(ssp_eff), float(dt_cfl)


# =============================================================================
# 9. Membrane Anti-Hourglass Viscosity for Fabric Shells (from mhvis3.F)
# =============================================================================

# Ported from $OR_SRC/engine/source/airbag/mhvis3.F: lines 109-364
def compute_membrane_hourglass_viscosity(
    nodes_v: np.ndarray,
    thk0: float,
    rho: float,
    area: float,
    sound_speed: float,
    dt: float,
    h4: float = 0.1,
    hvisc: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Compute anti-hourglass damping forces and dissipated energy for fabric membranes.

    Matches OpenRadioss ``mhvis3.F`` for 4-node quadrilateral fabric membrane elements:
    Calculates in-plane and hyperbolic/V-shape out-of-plane anti-hourglass modes,
    linear and quadratic viscous forces, and accumulates dissipated hourglass energy.

    Args:
        nodes_v: (4, 3) velocity vectors of the 4 membrane nodes.
        thk0: Initial shell thickness.
        rho: Fabric density.
        area: Element area.
        sound_speed: Dilatational wave speed in fabric continuum.
        dt: Time step size.
        h4: Hourglass parameter (GEO 17 or PM 91, default 0.1).
        hvisc: Viscosity scale (HVISC, default 1.0).

    Returns:
        (f_hour, e_hour):
            - f_hour: (4, 3) anti-hourglass damping forces on the 4 nodes.
            - e_hour: Dissipated hourglass energy increment dE = dt * sum(v_i . f_i).
    """
    if len(nodes_v) < 4 or area <= 0.0 or thk0 <= 0.0 or rho <= 0.0:
        return np.zeros((4, 3), dtype=np.float64), 0.0

    vx1, vx2, vx3, vx4 = nodes_v[0, 0], nodes_v[1, 0], nodes_v[2, 0], nodes_v[3, 0]
    vy1, vy2, vy3, vy4 = nodes_v[0, 1], nodes_v[1, 1], nodes_v[2, 1], nodes_v[3, 1]
    vz1, vz2, vz3, vz4 = nodes_v[0, 2], nodes_v[1, 2], nodes_v[2, 2], nodes_v[3, 2]

    # In-plane membrane hourglass velocity modes (mhvis3.F lines 235-236)
    hg1 = vx1 - vx2 + vx3 - vx4
    hg2 = vy1 - vy2 + vy3 - vy4

    # Out-of-plane V-shape bending hourglass modes (mhvis3.F lines 310-332)
    hg4_1 = +vz1 + vz2 - vz3 - vz4
    hg4_2 = +vz1 - vz2 - vz3 + vz4

    # Viscous coefficients (mhvis3.F lines 145-153)
    fac = 0.25 * rho * thk0
    h4l = fac * math.sqrt(max(1e-20, hvisc * h4 * area)) * sound_speed
    h4q = math.sqrt(max(1e-20, hvisc * h4)) * h4l * 100.0

    # Linear and quadratic damping forces
    f_hour = np.zeros((4, 3), dtype=np.float64)

    # In-plane damping
    f_inplane_x = hg1 * (h4l + h4q * abs(hg1))
    f_inplane_y = hg2 * (h4l + h4q * abs(hg2))
    f_hour[0, 0] += f_inplane_x;  f_hour[0, 1] += f_inplane_y
    f_hour[1, 0] -= f_inplane_x;  f_hour[1, 1] -= f_inplane_y
    f_hour[2, 0] += f_inplane_x;  f_hour[2, 1] += f_inplane_y
    f_hour[3, 0] -= f_inplane_x;  f_hour[3, 1] -= f_inplane_y

    # V-shape mode 1 (mhvis3.F lines 315-320)
    f_v1 = hg4_1 * (h4l + h4q * abs(hg4_1))
    f_hour[0, 2] += f_v1
    f_hour[1, 2] += f_v1
    f_hour[2, 2] -= f_v1
    f_hour[3, 2] -= f_v1

    # V-shape mode 2 (mhvis3.F lines 327-332)
    f_v2 = hg4_2 * (h4l + h4q * abs(hg4_2))
    f_hour[0, 2] += f_v2
    f_hour[1, 2] -= f_v2
    f_hour[2, 2] -= f_v2
    f_hour[3, 2] += f_v2

    # Hourglass work / energy dissipation rate: dE = dt * sum(v_i . f_i) (mhvis3.F line 347)
    e_hour = dt * float(np.sum(nodes_v * f_hour))

    return f_hour, float(max(0.0, e_hour))


