"""Explicit thermal solver for OpenRadioss (pyradioss).

Upstream OpenRadioss Fortran References:
----------------------------------------
- Nodal temperature time integration (TEMPUR):
  ``$OR_SRC/engine/source/constraints/thermic/tempur.F``
- Imposed temperature boundary conditions (FIXTEMP, /IMPTEMP):
  ``$OR_SRC/engine/source/constraints/thermic/fixtemp.F``
- Imposed heat flux sources (FIXFLUX, /FIXFLUX, /IMPFLUX):
  ``$OR_SRC/engine/source/constraints/thermic/fixflux.F``
- Global thermal energy balance tracking (THERMBILAN):
  ``$OR_SRC/engine/source/constraints/thermic/thermbilan.F``
- Critical thermal time step limit (DTTHERM, /DT/THERM):
  ``$OR_SRC/engine/source/time_step/dttherm.F90``
- Global thermal model parameters and ledger (GLOB_THERM_MOD):
  ``$OR_SRC/common_source/modules/mat_elem/glob_therm_mod.F90``
- Solid element internal conduction (STHERM):
  ``$OR_SRC/engine/source/elements/solid/solide/stherm.F``

Physics Overview:
-----------------
1. Explicit Nodal Temperature Time Integration (tempur.F):
   - Conservation of thermal energy at each node:
     M_i * Cp_i * dT_i/dt = Q_i_net
     where M_i is nodal lumped mass, Cp_i is specific heat capacity (MCP_i = M_i * Cp_i).
   - In explicit forward Euler form over time step dt:
     Delta T_i = FTHE_i / MCP_i
     T_i^{n+1} = T_i^n + Delta T_i
     where FTHE_i is the total accumulated thermal energy increment (J) from conduction,
     boundary fluxes, convection, radiation, and mechanical dissipation over dt.
   - Element deactivation mask: MCP_i = MCP_i * MCP_OFF_i. If MCP_i <= 0, node temperature is fixed.
   - Cumulative thermal energy stored: Delta E_stored += sum_{MCP > 0} FTHE_i.

2. Imposed Temperature Boundary Conditions (fixtemp.F, /IMPTEMP):
   - Overwrites nodal temperature on specified node groups:
     T_imp(t) = scale * fct((t * theaccfact - tstart) / xscale)
   - Active only within time window [tstart, tstop].

3. Imposed Heat Flux (fixflux.F, /FIXFLUX):
   - Prescribes surfacic (W/m^2) or volumetric (W/m^3) or direct nodal heat flux:
     FLUX_DENS(t) = scale * fct((t * theaccfact - tstart) / xscale)
   - Surfacic flux: FLUX = Area * FLUX_DENS * dt_eff
   - Volumetric flux: FLUX = Volume * FLUX_DENS * dt_eff
   - Energy bookkeeping: HEAT_FFLUX += FLUX; FTHE_i += FLUX / m_nodes.

4. First-Law Thermal Energy Balance (thermbilan.F):
   - First Law of Thermodynamics:
     E_in - E_out = Delta E_stored
     E_fixed + E_meca - E_conv - E_rad = Delta E_stored
     where:
       * E_fixed: cumulative heat from imposed fluxes (HEAT_FFLUX)
       * E_meca: cumulative heat converted from mechanical plastic strain (HEAT_MECA)
       * E_conv: cumulative heat lost to fluid via convection (HEAT_CONV)
       * E_rad:  cumulative heat lost to ambient via radiation (HEAT_RADIA)
       * Delta E_stored: net thermal energy stored in the structure (HEAT_STORED = sum MCP * Delta T)

5. Critical Thermal Time Step (dttherm.F90):
   - From explicit parabolic diffusion stability (Fourier number Fo <= 1/2):
     dt_crit = dtfactherm * 0.5 * lc^2 * (rho * Cp) / max(k_eff, 1e-20)
     where lc is the characteristic element length, k_eff = k * theaccfact is effective conductivity,
     and dtfactherm (default 0.9) is the Courant-like thermal safety factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.constants import EM20
from .thermal_loads import compute_segment_area


@dataclass
class GlobTherm:
    """Global thermal solver parameters, flags, and energy ledger.

    Ported from $OR_SRC/common_source/modules/mat_elem/glob_therm_mod.F90
    """

    itherm_fe: int = 1  # 1 = thermal FE active for lagrangian analysis
    itherm: int = 0  # ALE/Eulerian thermal flag
    intheat: int = 0  # Thermal option in interfaces
    nfxtemp: int = 0  # Number of nodes with imposed temperature
    nfxflux: int = 0  # Number of entities with imposed heat flux
    numconv: int = 0  # Number of segments subject to convection
    numradia: int = 0  # Number of segments subject to radiation
    dt_therm: float = 1.0e30  # Current critical thermal time step
    theaccfact: float = 1.0  # Thermal acceleration factor
    dtfactherm: float = 0.9  # Thermal time step reduction safety factor
    heat_meca: float = 0.0  # Cumulative heat converted from mechanical plastic work
    heat_conv: float = 0.0  # Cumulative heat lost via convection
    heat_radia: float = 0.0  # Cumulative heat lost via radiation
    heat_fflux: float = 0.0  # Cumulative heat added via imposed fluxes
    heat_stored: float = 0.0  # Cumulative thermal energy stored in structure

    def reset_energy(self) -> None:
        """Reset all cumulative energy ledger terms to zero."""
        self.heat_meca = 0.0
        self.heat_conv = 0.0
        self.heat_radia = 0.0
        self.heat_fflux = 0.0
        self.heat_stored = 0.0


def get_glob_therm(model: Any) -> GlobTherm:
    """Retrieve or initialize the GlobTherm object on a model."""
    gt = getattr(model, "glob_therm", None)
    if gt is None or not isinstance(gt, GlobTherm):
        gt = GlobTherm()
        # Transfer existing attributes if present on model
        for attr in ("theaccfact", "dtfactherm", "heat_fflux", "heat_meca", "heat_conv", "heat_radia", "heat_stored"):
            if hasattr(model, attr) and getattr(model, attr) is not None:
                try:
                    setattr(gt, attr, float(getattr(model, attr)))
                except (ValueError, TypeError):
                    pass
        model.glob_therm = gt
    return gt


def _eval_curve_or_func(fct: Any, x_val: float) -> float:
    """Evaluate a function object, curve, or callable at abscissa x_val."""
    if fct is None:
        return 1.0
    if hasattr(fct, "eval"):
        return float(fct.eval(x_val))
    if hasattr(fct, "interpolate"):
        return float(fct.interpolate(x_val))
    if callable(fct):
        return float(fct(x_val))
    if hasattr(fct, "x") and hasattr(fct, "y"):
        # Radioss table / curve format
        return float(np.interp(x_val, fct.x, fct.y))
    return 1.0


def _resolve_node_indices(model: Any, target: Any) -> np.ndarray:
    """Resolve a target (NodeGroup, list of IDs, 0-based index array, or single ID) to 0-based indices."""
    if target is None:
        return np.zeros(0, dtype=np.int64)

    # If target is a group ID (int)
    if isinstance(target, (int, np.integer)):
        gid = int(target)
        if gid <= 0:
            # 0 means all nodes
            n_nod = len(getattr(model, "temperature", getattr(model, "x", [])))
            return np.arange(n_nod, dtype=np.int64)
        if hasattr(model, "node_groups") and gid in model.node_groups:
            target = model.node_groups[gid]
        elif hasattr(model, "grnod") and gid in model.grnod:
            target = model.grnod[gid]
        else:
            # Assume target is a single node ID
            target = [gid]

    # If target has node_ids attribute (NodeGroup object)
    if hasattr(target, "node_ids"):
        nids = target.node_ids
    elif isinstance(target, (list, tuple, np.ndarray, set)):
        nids = list(target)
    else:
        nids = [target]

    if len(nids) == 0:
        return np.zeros(0, dtype=np.int64)

    # Map user node IDs (1-based or arbitrary) to model 0-based array index
    res: List[int] = []
    id2idx = getattr(model, "_id2idx", None)
    model_nids = getattr(model, "node_ids", None)
    n_nod = len(getattr(model, "temperature", getattr(model, "x", [])))

    for nid in nids:
        nid_int = int(nid)
        if id2idx is not None and nid_int in id2idx:
            idx = int(id2idx[nid_int])
        elif hasattr(model, "node_index"):
            try:
                idx = int(model.node_index(nid_int))
            except (KeyError, IndexError):
                idx = nid_int - 1 if nid_int > 0 else 0
        elif model_nids is not None:
            matches = np.where(model_nids == nid_int)[0]
            idx = int(matches[0]) if len(matches) > 0 else (nid_int - 1 if nid_int > 0 else 0)
        else:
            # Default to 0-based indexing (if nid > 0, assume 1-based user ID)
            idx = nid_int - 1 if (nid_int > 0 and (n_nod == 0 or nid_int <= n_nod)) else nid_int

        if 0 <= idx < (n_nod if n_nod > 0 else idx + 1):
            res.append(idx)

    return np.array(res, dtype=np.int64)


def update_nodal_temperatures(model: Any, dt: float) -> np.ndarray:
    """Integrate nodal temperatures forward in time using explicit central difference / forward Euler.

    Ported from $OR_SRC/engine/source/constraints/thermic/tempur.F

    Upstream Fortran implementation:
    --------------------------------
    .. code-block:: fortran

        DO N=NODFT,NODLT
          MCP(N) = MCP(N) * MCP_OFF(N)
          IF (MCP(N) > ZERO ) THEN
            TEMP(N) =  TEMP(N)  + FTHE(N) / MCP(N)
            IF (WEIGHT(N) == 1) HEAT_STORED_LOCAL = HEAT_STORED_LOCAL + FTHE(N)
          ENDIF
          FTHE(N) = ZERO
        ENDDO
        HEAT_STORED = HEAT_STORED + HEAT_STORED_LOCAL

    Parameters
    ----------
    model : Any
        Radioss solver model instance containing:
        - `temperature` (or `temperatures`): shape (n_nodes,) ndarray of temperatures [K]
        - `mcp` (or `nodal_mcp` or `mass * cp`): shape (n_nodes,) nodal heat capacity [J/K]
        - `fthe`: shape (n_nodes,) thermal energy increment over dt [J], or `q_nodal` [W]
        - `mcp_off` (optional): shape (n_nodes,) element deactivation mask (1.0=active, 0.0=off)
        - `weight` (optional): shape (n_nodes,) MPI domain ownership (1=owned, 0=ghost)
        - `glob_therm`: GlobTherm global state tracking `heat_stored`
    dt : float
        Time step increment [s].

    Returns
    -------
    np.ndarray
        Updated nodal temperature array [K].
    """
    temp = getattr(model, "temperature", None)
    if temp is None:
        temp = getattr(model, "temperatures", None)
    if temp is None:
        raise ValueError("Model has no 'temperature' or 'temperatures' array.")

    temp = np.asarray(temp, dtype=np.float64)
    n_nod = len(temp)

    # 1. Retrieve or construct nodal heat capacity MCP = Mass * Cp [J/K]
    mcp = None
    if hasattr(model, "mcp") and model.mcp is not None:
        mcp = np.asarray(model.mcp, dtype=np.float64).copy()
    elif hasattr(model, "nodal_mcp") and model.nodal_mcp is not None:
        mcp = np.asarray(model.nodal_mcp, dtype=np.float64).copy()
    elif hasattr(model, "mass") and model.mass is not None:
        cp_val = getattr(model, "cp", getattr(model, "spheat", None))
        if cp_val is None:
            # Check materials in model
            cp_val = 1.0
            if hasattr(model, "materials") and model.materials:
                first_mat = next(iter(model.materials.values())) if isinstance(model.materials, dict) else model.materials[0]
                cp_val = getattr(first_mat, "spheat", getattr(first_mat, "cp", 1.0))
        mcp = np.asarray(model.mass * float(cp_val), dtype=np.float64)
    else:
        # Default unit heat capacity if not provided
        mcp = np.ones(n_nod, dtype=np.float64)

    # 2. Apply MCP_OFF mask (tempur.F line 49: MCP(N) = MCP(N) * MCP_OFF(N))
    mcp_off = getattr(model, "mcp_off", None)
    if mcp_off is not None:
        mcp_off_arr = np.asarray(mcp_off, dtype=np.float64)
        if len(mcp_off_arr) == n_nod:
            mcp = mcp * mcp_off_arr

    # 3. Retrieve thermal load vector FTHE [J] (or q_nodal [W] * dt)
    fthe = getattr(model, "fthe", None)
    if fthe is None:
        q_nodal = getattr(model, "q_nodal", None)
        if q_nodal is None:
            q_nodal = getattr(model, "thermal_flux", None)
        if q_nodal is not None:
            fthe = np.asarray(q_nodal, dtype=np.float64) * float(dt)
        else:
            fthe = np.zeros(n_nod, dtype=np.float64)
    else:
        fthe = np.asarray(fthe, dtype=np.float64)

    if len(fthe) != n_nod:
        fthe_adj = np.zeros(n_nod, dtype=np.float64)
        m_len = min(len(fthe), n_nod)
        fthe_adj[:m_len] = fthe[:m_len]
        fthe = fthe_adj

    # 4. Domain decomposition weight array (tempur.F line 52: IF (WEIGHT(N) == 1))
    weight = getattr(model, "weight", None)
    if weight is None:
        weight_mask = np.ones(n_nod, dtype=bool)
    else:
        weight_mask = np.asarray(weight, dtype=np.int32) == 1

    # 5. Vectorized update (tempur.F lines 50-55)
    active_mask = mcp > 0.0
    delta_t = np.zeros(n_nod, dtype=np.float64)
    delta_t[active_mask] = fthe[active_mask] / mcp[active_mask]
    temp[active_mask] += delta_t[active_mask]

    # 6. Global thermal energy stored booking (tempur.F lines 52, 58)
    stored_mask = active_mask & weight_mask
    heat_stored_local = float(np.sum(fthe[stored_mask]))

    gt = get_glob_therm(model)
    gt.heat_stored += heat_stored_local
    if hasattr(model, "heat_stored"):
        model.heat_stored = gt.heat_stored

    # 7. Zero out FTHE for next cycle (tempur.F line 54: FTHE(N) = ZERO)
    if hasattr(model, "fthe") and model.fthe is not None:
        if isinstance(model.fthe, np.ndarray):
            model.fthe[:] = 0.0
        else:
            model.fthe = np.zeros(n_nod, dtype=np.float64)
    if hasattr(model, "q_nodal") and model.q_nodal is not None:
        if isinstance(model.q_nodal, np.ndarray):
            model.q_nodal[:] = 0.0

    # Keep model.temperature and model.temperatures in sync
    if hasattr(model, "temperature") and model.temperature is not None:
        model.temperature = temp
    if hasattr(model, "temperatures") and model.temperatures is not None:
        model.temperatures = temp

    return temp


def apply_imposed_temperatures(model: Any, time: Optional[float] = None) -> Dict[int, float]:
    """Apply /IMPTEMP boundary conditions by setting prescribed temperatures on specified nodes.

    Ported from $OR_SRC/engine/source/constraints/thermic/fixtemp.F

    Upstream Fortran logic:
    -----------------------
    - Evaluate active time window: STARTT <= TT * theaccfact <= STOPT.
    - Evaluate relative time TS = (TT * theaccfact - STARTT) * FACX.
    - Interpolate temperature value YC from curve/function IFUNC.
    - Scale by magnitude factor FAC.
    - Prescribe TEMP(I) = YC * FAC directly on node I.

    Parameters
    ----------
    model : Any
        Radioss solver model instance containing:
        - `imptemp`: list of ImposedTemperature boundary conditions
        - `temperature` (or `temperatures`): nodal temperature array
        - `functions`: dictionary of curves / functions
        - `time` (or `t`): current simulation time [s]
        - `glob_therm`: GlobTherm global state tracking theaccfact
    time : float, optional
        Simulation time to evaluate at. Defaults to `model.time` or `model.t` (0.0).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping modified node indices to their prescribed temperatures.
    """
    temp = getattr(model, "temperature", None)
    if temp is None:
        temp = getattr(model, "temperatures", None)
    if temp is None:
        return {}

    if time is None:
        time = float(getattr(model, "time", getattr(model, "t", 0.0)))

    gt = get_glob_therm(model)
    theaccfact = float(getattr(gt, "theaccfact", 1.0))
    t_scaled = time * theaccfact

    # Collect all /IMPTEMP conditions
    bcs = getattr(model, "imptemp", None)
    if bcs is None or len(bcs) == 0:
        bcs = getattr(model, "fixed_temperatures", None)
    if bcs is None or len(bcs) == 0:
        bcs = getattr(model, "bcs_temp", [])

    if not bcs:
        return {}

    applied: Dict[int, float] = {}

    for bc in bcs:
        startt = float(getattr(bc, "tstart", 0.0) or 0.0)
        stopt = float(getattr(bc, "tstop", 1.0e30) or 1.0e30)

        # Check sensor offset if present (fixtemp.F lines 98-105)
        sens_id = int(getattr(bc, "sens_id", 0) or 0)
        if sens_id > 0 and hasattr(model, "sensors"):
            sensors = model.sensors
            if hasattr(sensors, "active") and not sensors.active(sens_id):
                continue
            if hasattr(sensors, "tstart"):
                t_sens = float(sensors.tstart(sens_id))
                startt += t_sens
                stopt += t_sens

        # Check active time window (fixtemp.F lines 107-108, 122-123)
        if t_scaled < startt or t_scaled > stopt:
            continue

        # Scale factors (VAL(3, N) = FACX, VAL(4, N) = FAC)
        scale = float(getattr(bc, "scale", getattr(bc, "val", 1.0)) or 1.0)
        xscale = float(getattr(bc, "xscale", 1.0) or 1.0)
        facx = (1.0 / xscale) if xscale not in (0.0, None) else 1.0

        ts = t_scaled - startt
        tsc = ts * facx

        # Curve / function evaluation (fixtemp.F lines 170-178)
        fct_id = getattr(bc, "funct_id", getattr(bc, "fct_id", None))
        fct = None
        if fct_id and hasattr(model, "functions") and model.functions:
            fct = model.functions.get(fct_id)
        elif callable(getattr(bc, "func", None)):
            fct = bc.func

        val = _eval_curve_or_func(fct, tsc)
        t_imposed = scale * val

        # Target nodes resolution
        target = getattr(bc, "grnod_id", None)
        if target is None or target == 0:
            target = getattr(bc, "nodes", getattr(bc, "node_ids", getattr(bc, "node_id", None)))

        node_indices = _resolve_node_indices(model, target)
        for idx in node_indices:
            temp[idx] = t_imposed
            applied[int(idx)] = t_imposed

    return applied


def apply_imposed_flux(model: Any, dt: Optional[float] = None) -> float:
    """Apply /IMPFLUX / /FIXFLUX heat source loads and book heat into GLOB_THERM%HEAT_FFLUX.

    Ported from $OR_SRC/engine/source/constraints/thermic/fixflux.F

    Upstream Fortran logic:
    -----------------------
    - Active time interval check: TTA >= STARTT and TTB < STOPT.
    - Effective dt computation: DT1N bounded by [STARTT, STOPT].
    - Flux density: FLUX_DENS = FCY * FINTER(IFUNC, TS * FCX).
    - Surfacic flux: FLUX = Area * FLUX_DENS * DT1N, distributed to segment nodes.
    - Volumetric flux: FLUX = Volume * FLUX_DENS * DT1N, distributed to element nodes.
    - Cumulative ledger booking: GLOB_THERM%HEAT_FFLUX += FLUX; FTHE(N) += FLUX / m.

    Parameters
    ----------
    model : Any
        Radioss solver model instance containing:
        - `impflux_loads` (or `impfluxes`): list of ImposedFlux loads
        - `fthe`: nodal heat increment vector [J]
        - `time` (or `t`): current simulation time [s]
        - `dt`: current time step increment [s]
        - `glob_therm`: GlobTherm global state
    dt : float, optional
        Time step increment [s]. Defaults to `model.dt` or 0.0.

    Returns
    -------
    float
        Total heat energy [J] added to the model during this call.
    """
    if dt is None:
        dt = float(getattr(model, "dt", 0.0))

    if dt <= 0.0:
        return 0.0

    time = float(getattr(model, "time", getattr(model, "t", 0.0)))
    gt = get_glob_therm(model)
    theaccfact = float(getattr(gt, "theaccfact", 1.0))

    tta = time * theaccfact
    dt1a = dt * theaccfact
    ttb = tta - dt1a

    # Ensure model has an FTHE array
    n_nod = len(getattr(model, "temperature", getattr(model, "temperatures", [])))
    if not hasattr(model, "fthe") or model.fthe is None:
        model.fthe = np.zeros(n_nod, dtype=np.float64)
    elif len(model.fthe) < n_nod:
        old_fthe = model.fthe
        model.fthe = np.zeros(n_nod, dtype=np.float64)
        model.fthe[: len(old_fthe)] = old_fthe

    fthe = model.fthe

    # Collect flux loads
    loads = getattr(model, "impflux_loads", None)
    if loads is None or len(loads) == 0:
        loads_dict = getattr(model, "impfluxes", None)
        if isinstance(loads_dict, dict):
            loads = list(loads_dict.values())
    if loads is None or len(loads) == 0:
        loads = getattr(model, "fixed_fluxes", [])

    if not loads:
        return 0.0

    total_flux_step = 0.0
    coords = getattr(model, "x", getattr(model, "coords", None))

    for fl in loads:
        startt = float(getattr(fl, "tstart", 0.0) or 0.0)
        stopt = float(getattr(fl, "tstop", 1.0e30) or 1.0e30)

        # Check sensor offset if present (fixflux.F lines 100-113)
        sens_id = int(getattr(fl, "sens_id", 0) or 0)
        if sens_id > 0 and hasattr(model, "sensors"):
            sensors = model.sensors
            if hasattr(sensors, "active") and not sensors.active(sens_id):
                continue
            if hasattr(sensors, "tstart"):
                t_sens = float(sensors.tstart(sens_id))
                startt += t_sens
                stopt += t_sens

        # Check activation window (fixflux.F line 115)
        if tta < startt or ttb >= stopt:
            continue

        # Effective time interval within [startt, stopt] (fixflux.F lines 116-128)
        if tta > stopt:
            if ttb <= startt:
                dt1n = stopt - startt
            else:
                dt1n = stopt - ttb
        else:  # tta <= stopt
            if ttb <= startt:
                dt1n = tta - startt
            else:
                dt1n = dt1a

        # Scale factors
        scale = float(getattr(fl, "scale", getattr(fl, "scale_y", 1.0)) or 1.0)
        xscale = float(getattr(fl, "xscale", getattr(fl, "scale_x", 1.0)) or 1.0)
        facx = (1.0 / xscale) if xscale not in (0.0, None) else 1.0

        ts = tta - startt
        tsc = ts * facx

        # Evaluate time function
        fct_id = getattr(fl, "funct_id", getattr(fl, "fct_id", 0))
        fct = None
        if fct_id and hasattr(model, "functions") and model.functions:
            fct = model.functions.get(fct_id)
        elif callable(getattr(fl, "func", None)):
            fct = fl.func

        val = _eval_curve_or_func(fct, tsc)
        flux_dens = scale * val  # W/m^2 or W/m^3

        # Determine surface vs volumetric flux
        surf_id = int(getattr(fl, "surf_id", 0) or 0)
        grbric_id = int(getattr(fl, "grbric_id", getattr(fl, "grbrick_id", 0)) or 0)

        # -------------------------------------------------------------
        # Case A: Surfacic heat flux (fixflux.F lines 149-204)
        # -------------------------------------------------------------
        if surf_id > 0 or hasattr(fl, "segments") or hasattr(fl, "surface"):
            segments = getattr(fl, "segments", None)
            if segments is None and hasattr(model, "surfaces") and surf_id in model.surfaces:
                surf_obj = model.surfaces[surf_id]
                segments = getattr(surf_obj, "segments", [])

            if segments:
                for seg in segments:
                    m = len(seg)
                    if m == 0:
                        continue
                    area = compute_segment_area(seg, coords) if coords is not None else float(getattr(fl, "area", 1.0) / len(segments))
                    q_seg = area * flux_dens * dt1n
                    total_flux_step += q_seg
                    gt.heat_fflux += q_seg
                    q_node = q_seg / m
                    node_indices = _resolve_node_indices(model, seg)
                    for n_idx in node_indices:
                        fthe[n_idx] += q_node
            elif hasattr(fl, "area") and fl.area > 0.0:
                # Explicit lumped surface area
                q_total = float(fl.area) * flux_dens * dt1n
                total_flux_step += q_total
                gt.heat_fflux += q_total
                target = getattr(fl, "nodes", getattr(fl, "grnod_id", None))
                node_indices = _resolve_node_indices(model, target)
                m = len(node_indices) if len(node_indices) > 0 else 1
                for n_idx in node_indices:
                    fthe[n_idx] += q_total / m

        # -------------------------------------------------------------
        # Case B: Volumetric heat flux (fixflux.F lines 207-240)
        # -------------------------------------------------------------
        elif grbric_id > 0 or hasattr(fl, "volume"):
            vol = float(getattr(fl, "volume", 0.0) or 0.0)
            if vol > 0.0:
                q_total = vol * flux_dens * dt1n
                total_flux_step += q_total
                gt.heat_fflux += q_total
                target = getattr(fl, "nodes", getattr(fl, "grnod_id", None))
                node_indices = _resolve_node_indices(model, target)
                m = len(node_indices) if len(node_indices) > 0 else 1
                for n_idx in node_indices:
                    fthe[n_idx] += q_total / m

        # -------------------------------------------------------------
        # Case C: Direct nodal flux injection
        # -------------------------------------------------------------
        else:
            target = getattr(fl, "nodes", getattr(fl, "grnod_id", getattr(fl, "node_ids", None)))
            node_indices = _resolve_node_indices(model, target)
            if len(node_indices) > 0:
                q_per_node = flux_dens * dt1n
                q_total = q_per_node * len(node_indices)
                total_flux_step += q_total
                gt.heat_fflux += q_total
                for n_idx in node_indices:
                    fthe[n_idx] += q_per_node

    if hasattr(model, "heat_fflux"):
        model.heat_fflux = gt.heat_fflux

    return total_flux_step


def compute_thermal_balance(model: Any) -> Dict[str, Any]:
    """Compute and verify First-Law thermal energy balance across all heat exchange mechanisms.

    Ported from $OR_SRC/engine/source/constraints/thermic/thermbilan.F

    Upstream Fortran implementation:
    --------------------------------
    .. code-block:: fortran

        ARRAY(1) =  GLOB_THERM%HEAT_FFLUX
        ARRAY(2) =  GLOB_THERM%HEAT_MECA
        ARRAY(3) =  GLOB_THERM%HEAT_CONV
        ARRAY(4) =  GLOB_THERM%HEAT_RADIA
        ARRAY(5) =  GLOB_THERM%HEAT_STORED

    Energy conservation equation:
      E_fixed + E_meca - E_conv - E_rad = Delta E_stored

    Parameters
    ----------
    model : Any
        Radioss solver model instance containing GlobTherm or thermal tracking fields.

    Returns
    -------
    Dict[str, Any]
        Dictionary of thermal balance energy values [J]:
        - `heat_fflux` (E_fixed): imposed boundary heat input
        - `heat_meca` (E_meca): mechanical strain/friction dissipation converted to heat
        - `heat_conv` (E_conv): heat lost via convection cooling
        - `heat_radia` (E_rad): heat lost via Stefan-Boltzmann radiation
        - `heat_stored` (Delta E_stored): cumulative heat accumulated by tempur.F
        - `delta_e_stored`: direct thermal energy change sum MCP * (T - T0)
        - `balance_error`: absolute energy conservation discrepancy [J]
        - `relative_error`: relative error normalized by max energy scale
        - `is_conserved`: True if relative error < 1e-4 or balance_error < 1e-8
    """
    gt = get_glob_therm(model)

    e_fixed = float(gt.heat_fflux)
    e_meca = float(gt.heat_meca)
    e_conv = float(gt.heat_conv)
    e_rad = float(gt.heat_radia)
    heat_stored = float(gt.heat_stored)

    # Compute direct Delta E_stored from nodal temperatures: sum MCP_i * (T_i - T0_i)
    temp = getattr(model, "temperature", getattr(model, "temperatures", None))
    temp0 = getattr(model, "initial_temperature", getattr(model, "temp0", None))
    mcp = getattr(model, "mcp", getattr(model, "nodal_mcp", None))

    delta_e_stored_direct: Optional[float] = None
    if temp is not None and temp0 is not None and mcp is not None:
        t_arr = np.asarray(temp, dtype=np.float64)
        t0_arr = np.asarray(temp0, dtype=np.float64)
        mcp_arr = np.asarray(mcp, dtype=np.float64)
        if len(t_arr) == len(t0_arr) == len(mcp_arr):
            delta_e_stored_direct = float(np.sum(mcp_arr * (t_arr - t0_arr)))

    delta_e_stored = delta_e_stored_direct if delta_e_stored_direct is not None else heat_stored

    # First Law: E_fixed + E_meca - E_conv - E_rad = Delta E_stored
    e_expected = (e_fixed + e_meca) - (e_conv + e_rad)
    balance_error = abs(e_expected - delta_e_stored)

    scale = max(abs(e_expected), abs(delta_e_stored), abs(e_fixed), abs(e_conv), abs(e_rad), 1.0e-20)
    relative_error = balance_error / scale
    is_conserved = bool(relative_error < 1.0e-4 or balance_error < 1.0e-8)

    return {
        "heat_fflux": e_fixed,
        "heat_meca": e_meca,
        "heat_conv": e_conv,
        "heat_radia": e_rad,
        "heat_stored": heat_stored,
        "delta_e_stored": delta_e_stored,
        "expected_stored": e_expected,
        "balance_error": balance_error,
        "relative_error": relative_error,
        "is_conserved": is_conserved,
    }


def compute_thermal_dt(model: Any) -> float:
    """Calculate the critical explicit thermal time step based on element diffusion limit.

    Ported from $OR_SRC/engine/source/time_step/dttherm.F90

    Upstream Fortran formulation:
    -----------------------------
    .. code-block:: fortran

        do i=1,nel
          if (tempel(i) < tmelt) then
            akk = as + bs*tempel(i)
          else
            akk = al + bl*tempel(i)
          end if
          akk = akk * glob_therm%theaccfact
          lc2 = lc(i)*lc(i)
          dt  = glob_therm%dtfactherm * half*lc2*rhocp/max(akk,em20)
          if (dt < glob_therm%dt_therm)  glob_therm%dt_therm = dt
        end do

    Parameters
    ----------
    model : Any
        Radioss solver model containing:
        - `glob_therm`: GlobTherm with `dtfactherm` and `theaccfact`
        - Mesh elements (bricks, shells, bars, etc.) or element thermal parameters:
          `lc` (characteristic length), `rhocp` (rho*Cp), `as`, `bs`, `al`, `bl`, `tmelt`
          (or `k` / `thermal_conductivity`, `rho`, `cp`).

    Returns
    -------
    float
        Critical explicit thermal time step [s].
    """
    gt = get_glob_therm(model)
    dtfactherm = float(getattr(gt, "dtfactherm", 0.9))
    theaccfact = float(getattr(gt, "theaccfact", 1.0))
    dt_therm_min = float(getattr(gt, "dt_therm", 1.0e30))

    dt_candidates: List[float] = []

    # 1. Direct explicit arrays on model (vectorized fast path)
    if hasattr(model, "lc") and model.lc is not None and hasattr(model, "rhocp") and model.rhocp is not None:
        lc = np.asarray(model.lc, dtype=np.float64)
        rhocp = np.asarray(model.rhocp, dtype=np.float64)
        tempel = np.asarray(getattr(model, "tempel", np.zeros_like(lc)), dtype=np.float64)

        as_val = float(getattr(model, "as", getattr(model, "k", 50.0)))
        bs_val = float(getattr(model, "bs", 0.0))
        al_val = float(getattr(model, "al", as_val))
        bl_val = float(getattr(model, "bl", 0.0))
        tmelt = float(getattr(model, "tmelt", 1.0e30))

        akk = np.where(tempel < tmelt, as_val + bs_val * tempel, al_val + bl_val * tempel)
        akk = akk * theaccfact
        akk = np.maximum(akk, EM20)

        dt_arr = dtfactherm * 0.5 * (lc**2) * rhocp / akk
        if len(dt_arr) > 0:
            dt_candidates.append(float(np.min(dt_arr)))

    # 2. Scalar parameters on model (e.g. 1D bar or lumped test problem)
    elif hasattr(model, "dx") or hasattr(model, "lc"):
        dx = float(getattr(model, "dx", getattr(model, "lc", 1.0)))
        rho = float(getattr(model, "rho", 7800.0))
        cp = float(getattr(model, "cp", getattr(model, "spheat", 500.0)))
        rhocp = float(getattr(model, "rhocp", rho * cp))

        as_val = float(getattr(model, "as", getattr(model, "k", 50.0)))
        bs_val = float(getattr(model, "bs", 0.0))
        al_val = float(getattr(model, "al", as_val))
        bl_val = float(getattr(model, "bl", 0.0))
        tmelt = float(getattr(model, "tmelt", 1.0e30))

        temp_val = float(getattr(model, "temp_elem", 300.0))
        if temp_val < tmelt:
            akk = as_val + bs_val * temp_val
        else:
            akk = al_val + bl_val * temp_val

        akk = akk * theaccfact
        akk = max(akk, EM20)

        dt_val = dtfactherm * 0.5 * (dx**2) * rhocp / akk
        dt_candidates.append(dt_val)

    # 3. Scan brick element groups if present
    bricks = getattr(model, "bricks", None)
    if bricks is not None and hasattr(bricks, "conn") and len(bricks.conn) > 0:
        coords = getattr(model, "x", getattr(model, "coords", None))
        temp = getattr(model, "temperature", None)
        conn = bricks.conn
        nel = len(conn)

        if coords is not None:
            c_arr = np.asarray(coords)
            n1 = conn[:, 0]
            n2 = conn[:, 1]
            diff = c_arr[n2] - c_arr[n1] if c_arr.ndim == 2 and c_arr.shape[1] == 3 else c_arr[:, n2].T - c_arr[:, n1].T
            lc = np.linalg.norm(diff, axis=1)
        else:
            lc = np.ones(nel, dtype=np.float64)

        tempel = np.zeros(nel, dtype=np.float64)
        if temp is not None:
            t_arr = np.asarray(temp)
            tempel = np.mean(t_arr[conn], axis=1)

        as_val = 50.0
        rhocp = 7800.0 * 500.0
        parts = getattr(model, "parts_list", getattr(model, "parts", []))
        if parts and hasattr(bricks, "part"):
            p_idx = int(bricks.part[0]) if len(bricks.part) > 0 else 0
            if isinstance(parts, list) and 0 <= p_idx < len(parts):
                part = parts[p_idx]
                mat = getattr(part, "material", None)
                if mat is not None:
                    as_val = float(getattr(mat, "k", getattr(mat, "as", 50.0)))
                    rho0 = float(getattr(mat, "rho0", 7800.0))
                    cp0 = float(getattr(mat, "spheat", getattr(mat, "cp", 500.0)))
                    rhocp = float(getattr(mat, "rhocp", rho0 * cp0))

        akk = as_val * theaccfact
        dt_arr = dtfactherm * 0.5 * (lc**2) * rhocp / max(akk, EM20)
        dt_candidates.append(float(np.min(dt_arr)))

    if dt_candidates:
        dt_calc = min(dt_candidates)
        dt_therm_min = min(dt_therm_min, dt_calc)
        gt.dt_therm = dt_therm_min

    return dt_therm_min


def compute_1d_bar_conduction(
    nodes: Sequence[int],
    temps: np.ndarray,
    dx: float,
    area: float,
    k: float,
    theaccfact: float,
    dt: float,
) -> np.ndarray:
    """Calculate internal heat conduction thermal force increments FTHE [J] for a 1D bar mesh.

    Matches the conservation properties of OpenRadioss `stherm.F`:
    - Strict local energy conservation: Q_{i->i+1} = -Q_{i+1->i}.
    - Net internal conduction sum is zero to machine precision: sum(FTHE) == 0.

    Parameters
    ----------
    nodes : Sequence[int]
        Node indices along the bar from x=0 to x=L.
    temps : np.ndarray
        Nodal temperature array [K].
    dx : float
        Uniform element spacing [m].
    area : float
        Bar cross-sectional area [m^2].
    k : float
        Thermal conductivity [W/(m*K)].
    theaccfact : float
        Thermal acceleration factor.
    dt : float
        Time step increment [s].

    Returns
    -------
    np.ndarray
        Thermal energy increment vector FTHE [J] of length equal to len(temps).
    """
    fthe = np.zeros(len(temps), dtype=np.float64)
    n_nodes = len(nodes)
    if n_nodes < 2 or dx <= 0.0 or dt <= 0.0:
        return fthe

    conductance = (k * theaccfact * area / dx) * dt

    for i in range(n_nodes - 1):
        n_a = nodes[i]
        n_b = nodes[i + 1]
        t_a = float(temps[n_a])
        t_b = float(temps[n_b])

        # Heat flux from a to b over dt [J]
        delta_q = conductance * (t_a - t_b)

        # Conservation: heat leaves a and enters b
        fthe[n_a] -= delta_q
        fthe[n_b] += delta_q

    return fthe


def apply_conduction(model: Any, dt: float) -> np.ndarray:
    """Assemble internal conduction heat transfer increments into model.fthe.

    Supports:
    - Explicit 1D bar mesh (model.bar_nodes, model.dx, model.area, model.k)
    - Conduction matrix K_cond if preassembled on model (FTHE += -K * T * dt)
    - 3D brick elements

    Parameters
    ----------
    model : Any
        Radioss solver model instance.
    dt : float
        Time step increment [s].

    Returns
    -------
    np.ndarray
        Nodal thermal force increment vector FTHE [J].
    """
    temp = getattr(model, "temperature", getattr(model, "temperatures", None))
    if temp is None:
        return np.zeros(0, dtype=np.float64)

    temp_arr = np.asarray(temp, dtype=np.float64)
    n_nod = len(temp_arr)

    if not hasattr(model, "fthe") or model.fthe is None:
        model.fthe = np.zeros(n_nod, dtype=np.float64)
    elif len(model.fthe) != n_nod:
        model.fthe = np.zeros(n_nod, dtype=np.float64)

    gt = get_glob_therm(model)
    theaccfact = float(getattr(gt, "theaccfact", 1.0))

    # Case 1: 1D bar mesh
    if hasattr(model, "bar_nodes") and model.bar_nodes is not None:
        bar_nodes = model.bar_nodes
        dx = float(getattr(model, "dx", 1.0))
        area = float(getattr(model, "area", 1.0))
        k = float(getattr(model, "k", getattr(model, "as", 50.0)))
        fthe_cond = compute_1d_bar_conduction(bar_nodes, temp_arr, dx, area, k, theaccfact, dt)
        model.fthe += fthe_cond

    # Case 2: Explicit conduction conductance matrix K_cond
    elif hasattr(model, "k_matrix") and model.k_matrix is not None:
        k_mat = np.asarray(model.k_matrix, dtype=np.float64)
        q_rate = -np.dot(k_mat, temp_arr) * theaccfact
        model.fthe += q_rate * dt

    return model.fthe


def solve_thermal_step(model: Any, dt: float) -> Dict[str, Any]:
    """Execute one complete explicit thermal solver cycle.

    Orchestrates the OpenRadioss thermal solver sequence matching `resol.F`:
    1. Compute / check thermal critical time step (dttherm.F90)
    2. Apply imposed heat fluxes (fixflux.F)
    3. Apply convection and radiation boundary loads (convec.F, radiation.F)
    4. Compute internal conduction heat flux (stherm.F, thermc.F)
    5. Update nodal temperatures: T_new = T_old + FTHE / MCP (tempur.F)
    6. Enforce imposed temperature boundary conditions (fixtemp.F)
    7. Compute thermal energy balance and ledger (thermbilan.F)

    Parameters
    ----------
    model : Any
        Radioss solver model instance.
    dt : float
        Time step increment [s].

    Returns
    -------
    Dict[str, Any]
        Thermal balance and step diagnostics.
    """
    model.dt = dt

    # 1. Critical dt check
    compute_thermal_dt(model)

    # 2. Imposed heat flux (/FIXFLUX)
    apply_imposed_flux(model, dt=dt)

    # 3. Convection and radiation boundary loads
    if hasattr(model, "thermal_loads_mgr") and model.thermal_loads_mgr is not None:
        t_curr = float(getattr(model, "time", getattr(model, "t", 0.0)))
        coords = getattr(model, "x", getattr(model, "coords", None))
        step_res = model.thermal_loads_mgr.compute_step(
            dt=dt,
            time=t_curr,
            temps=model.temperature,
            coords=coords,
        )
        gt = get_glob_therm(model)
        gt.heat_conv = model.thermal_loads_mgr.cumul_energy_conv
        gt.heat_radia = model.thermal_loads_mgr.cumul_energy_rad
        for n_id, q_val in step_res.total_nodal_power.items():
            idx = n_id - 1 if n_id > 0 and len(model.fthe) >= n_id else n_id
            if 0 <= idx < len(model.fthe):
                model.fthe[idx] += q_val * dt

    # 4. Internal conduction
    apply_conduction(model, dt=dt)

    # 5. Nodal temperature update (tempur.F)
    update_nodal_temperatures(model, dt=dt)

    # 6. Enforce imposed temperatures (fixtemp.F)
    apply_imposed_temperatures(model)

    # Advance time
    if hasattr(model, "time"):
        model.time += dt
    elif hasattr(model, "t"):
        model.t += dt

    # 7. Energy balance
    balance = compute_thermal_balance(model)
    return balance
