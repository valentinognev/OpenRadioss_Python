"""LAW5 / JWL — Jones-Wilkins-Lee High Pressure Gas Expansion Equation of State (/MAT/LAW5, /MAT/JWL).

Function: SIGEPS_05 / M5LAW (lines 1-186) & MJWL (lines 1-189)
Upstream Fortran origins:
- ``$OR_SRC/engine/source/materials/mat/mat005/m5law.F`` (solid constitutive update & sound speed, subroutine M5LAW / SIGEPS05)
- ``$OR_SRC/engine/source/materials/mat/mat005/mjwl.F`` (energy, afterburning & stress state, subroutine MJWL)
- ``$OR_SRC/starter/source/materials/mat/mat005/hm_read_mat05.F`` (starter card reader & defaults, subroutine HM_READ_MAT05)
- ``$OR_SRC/hm_cfg_files/config/CFG/radioss110/MAT/matl5_jwl.cfg`` (CFG attributes & format)

Theory & Formulation
--------------------
LAW5 models the high-pressure expansion of detonation products using the empirical
Jones-Wilkins-Lee (JWL) equation of state.

1. Relative volume (m5law.F line 102):
   v = DF = rho0 / rho = 1 / (1 + mu)
   where mu = rho / rho0 - 1 = -tr(eps_vol).

2. Burn fraction kinetics BFRAC (m5law.F lines 108-121):
   Time control:
     b_1 = D * (t - t_b) / (1.5 * Delta_x)  for t > t_b if I_bfrac != 1
   Volumetric control:
     b_2 = B_HE * (1 - v)  if I_bfrac != 2
     where B_HE = rho_ref * D^2 / P_CJ.
   Combined burn fraction:
     BFRAC = max(BFRAC_old, max(b_1, b_2))
     Clamped to [0, 1]. If BFRAC < 1e-4, BFRAC = 0.

3. Detonation pressure P_JWL (m5law.F lines 124-145):
   P_JWL = A * (1 - omega / (R_1 * v)) * exp(-R_1 * v)
         + B * (1 - omega / (R_2 * v)) * exp(-R_2 * v)
         + omega * E_int / V
   where V = v * V_0, and E_int is current internal energy.

4. Unreacted blend & cavitation cutoff (m5law.F lines 135-148):
   If K_unreacted == 0:
     P = P_0 + P_JWL
   Else:
     P = (1 - BFRAC) * (P_0 + K_unreacted * mu) + BFRAC * P_JWL
   Cavitation cutoff:
     P = max(0, P) - P_sh

5. Hydrodynamic fluid stress state (m5law.F lines 174-182, mjwl.F lines 180-184):
   Deviatoric shear stress is zero (G = 0).
   sig = -P * I  (in Voigt 6-vector: [-P, -P, -P, 0, 0, 0]).

6. Afterburning (mjwl.F lines 78-156):
   Computes E_int <- E_int + (lambda - lambda_old) * E_add * V_0 - (P + P_sh) * Delta_V.
   - Q_opt = 0: instantaneous release at t > T_begin
   - Q_opt = 1: constant afterburning rate from T_begin to T_end
   - Q_opt = 2: linear afterburning rate from T_begin to T_end
   - Q_opt = 3: Miller's extension, rate dependent on pressure

7. Sound speed calculation (m5law.F lines 156-172):
   SSP = A * exp(-R_1 * v) * (-omega / (R_1 * v) + R_1 * v - omega)
       + B * exp(-R_2 * v) * (-omega / (R_2 * v) + R_2 * v - omega)
       + omega * E_int / V + (P + P_sh) * omega
   SSP = SSP * v
   If K_unreacted == 0:
     c = sqrt(|SSP| / rho0), bounded below by D * (1 - BFRAC)
   Else:
     c = sqrt(|BFRAC * (SSP / rho0) + (1 - BFRAC) * (K_unreacted / rho0)|)

8. Consistent tangent (purely volumetric hydrodynamic):
   K_eff = rho0 * c^2
   D = K_eff * (1 (x) 1) (upper-left 3x3 block all K_eff, all others 0).
"""

from __future__ import annotations

import numpy as np

from pyradioss.model.entities import Material

_EM04 = 1e-4
_EM20 = 1e-20
_INF = 1e30


def _get_val(d: dict, *keys, default=None):
    """Retrieve matching key from dictionary with smart alias conflict resolution."""
    present = [d[k] for k in keys if k in d and d[k] is not None]
    if not present:
        return default
    if len(present) == 1:
        return present[0]
    try:
        if all(np.array_equal(v, present[0]) for v in present):
            return present[0]
    except Exception:
        pass
    try:
        non_zeros = [v for v in present if np.any(np.asarray(v) != 0)]
        if len(non_zeros) == 1:
            return non_zeros[0]
    except Exception:
        pass
    return present[0]


def _extract_params(mat: Material | dict) -> dict:
    """Robust parameter extraction supporting CFG uppercase and lowercase names.

    Fortran origin: ``starter/source/materials/mat/mat005/hm_read_mat05.F``.
    """
    if hasattr(mat, "params") and mat.params is not None:
        p = mat.params
    elif isinstance(mat, dict):
        p = mat.get("params", mat)
    else:
        p = getattr(mat, "params", {})

    # Initial density and reference density
    rho0_val = _get_val(p, "MAT_RHO", "rho0", "rho", "density")
    if rho0_val is None:
        rho0_val = getattr(mat, "rho0", 1.0) or 1.0
    rho0 = float(rho0_val)
    if rho0 <= 0.0:
        rho0 = 1.0

    rhor_val = _get_val(p, "Refer_Rho", "rhor", "ref_rho")
    rhor = float(rhor_val) if rhor_val is not None else 0.0
    if rhor == 0.0:
        rhor = rho0  # hm_read_mat05.F line 215: IF(RHOR == ZERO) RHOR=RHO0

    # EOS JWL parameters
    a = float(_get_val(p, "MAT_A", "a", default=0.0))
    b = float(_get_val(p, "MAT_B", "b", default=0.0))
    r1 = float(_get_val(p, "MAT_PDIR1", "r1", "pdir1", default=0.0))
    r2 = float(_get_val(p, "MAT_PDIR2", "r2", "pdir2", default=0.0))
    omega = float(_get_val(p, "Omega", "omega", "w", default=0.0))

    # Detonation velocity & Chapman-Jouguet state
    d = float(_get_val(p, "MAT_D", "d", "vdet", default=0.0))
    pcj = float(_get_val(p, "MAT_PC", "pc", "pcj", default=0.0))
    e0 = float(_get_val(p, "MAT_E0", "e0", default=0.0))
    eadd = float(_get_val(p, "MAT_E", "eadd", "e", default=0.0))

    # Burn fraction method & afterburning option
    ibfrac = int(_get_val(p, "MAT_IBFRAC", "ibfrac", default=0))
    qopt = int(_get_val(p, "QOPT", "qopt", default=0))
    if qopt < 0 or qopt > 3:
        qopt = 0  # hm_read_mat05.F lines 153-155

    # Initial pressure, pressure shift, unreacted bulk modulus
    p0 = float(_get_val(p, "LAW5_P0", "p0", "c0", default=0.0))
    psh = float(_get_val(p, "LAW5_PSH", "psh", default=0.0))
    bulk = float(_get_val(p, "BUNREACTED", "bulk", "bunreacted", default=0.0))

    # Derived Chapman-Jouguet parameters (hm_read_mat05.F lines 230, 235)
    b_he = (rhor * (d ** 2) / pcj) if pcj > 0.0 else 0.0
    v_cj = (1.0 - 1.0 / b_he) if b_he != 0.0 else 0.0

    # Interface stiffness C1 (hm_read_mat05.F lines 208-212)
    if bulk > 0.0:
        c1 = bulk
    else:
        c1 = omega * (pcj + e0)

    # Afterburning parameters (hm_read_mat05.F lines 163-205)
    tbegin = float(_get_val(p, "TSTART", "tstart", "tbegin", default=0.0))
    tend_raw = _get_val(p, "TSTOP", "tstop", "tend")
    tend = float(tend_raw) if tend_raw is not None else _INF
    if tend == 0.0:
        tend = _INF

    if eadd > 0.0 and tbegin == tend:
        qopt = 0  # Dirac instantaneous release

    a_mil = float(_get_val(p, "LAW5_A", "a_mil", default=0.0))
    m_mil = float(_get_val(p, "LAW5_M", "m_mil", default=0.0))
    n_mil = float(_get_val(p, "LAW5_N", "n_mil", default=0.0))
    alpha_unit = float(_get_val(p, "alpha_unit", default=1.0))
    if alpha_unit == 0.0:
        alpha_unit = 1.0  # hm_read_mat05.F line 200

    # Reaction rates
    reaction_rate = 0.0
    reaction_rate2 = 0.0
    if qopt == 1:
        dt_span = tend - tbegin
        reaction_rate = 1.0 / dt_span if dt_span > 0.0 else 0.0
    elif qopt == 2:
        dt_span = tend - tbegin
        denom = dt_span ** 2 if dt_span > 0.0 else 1.0
        reaction_rate = 2.0 / denom
        reaction_rate2 = (tbegin ** 2) / denom

    params = {
        "rho0": rho0,
        "rhor": rhor,
        "a": a,
        "b": b,
        "r1": r1,
        "r2": r2,
        "omega": omega,
        "w": omega,
        "d": d,
        "vdet": d,
        "pcj": pcj,
        "pc": pcj,
        "e0": e0,
        "eadd": eadd,
        "ibfrac": ibfrac,
        "qopt": qopt,
        "p0": p0,
        "psh": psh,
        "bulk": bulk,
        "bunreacted": bulk,
        "b_he": b_he,
        "v_cj": v_cj,
        "c1": c1,
        "tbegin": tbegin,
        "tstart": tbegin,
        "tend": tend,
        "tstop": tend,
        "reaction_rate": reaction_rate,
        "reaction_rate2": reaction_rate2,
        "a_mil": a_mil,
        "m_mil": m_mil,
        "n_mil": n_mil,
        "alpha_unit": alpha_unit,
        # Mirrored uppercase CFG keys
        "MAT_RHO": rho0,
        "Refer_Rho": rhor,
        "MAT_A": a,
        "MAT_B": b,
        "MAT_PDIR1": r1,
        "MAT_PDIR2": r2,
        "Omega": omega,
        "MAT_D": d,
        "MAT_PC": pcj,
        "MAT_E0": e0,
        "MAT_E": eadd,
        "MAT_IBFRAC": ibfrac,
        "QOPT": qopt,
        "LAW5_P0": p0,
        "LAW5_PSH": psh,
        "BUNREACTED": bulk,
        "TSTART": tbegin,
        "TSTOP": tend,
        "LAW5_A": a_mil,
        "LAW5_M": m_mil,
        "LAW5_N": n_mil,
    }
    return params


def build_law5(rec) -> Material:
    """Card parsing and validation -> Material for LAW5 (JWL explosive).

    Fortran origin: ``starter/source/materials/mat/mat005/hm_read_mat05.F``.
    """
    if isinstance(rec, Material):
        mat_id = rec.id
        title = rec.title
        p = _extract_params(rec)
        density = p["rho0"]
    elif isinstance(rec, dict):
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        p = _extract_params(rec)
        density = p["rho0"]
    else:
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        p = _extract_params(rec)
        density = p["rho0"]

    return Material(id=mat_id, law=5, rho0=density, title=title, params=p)


build_law05 = build_law5
build_jwl = build_law5


def extra_shapes(mat=None, nip: int = 1) -> dict[str, tuple[int, ...]]:
    """Buffer allocation shapes for LAW5 per-element state.

    Fortran origin: ``hm_read_mat05.F`` lines 251-258:
    - G_TB, L_TB: detonation arrival time
    - G_BFRAC, L_BFRAC: burn fraction
    - G_ABURN, L_ABURN: afterburning progress fraction
    - eint: internal energy
    """
    return {
        "bfrac": (),
        "aburn": (),
        "eint": (),
        "tb": (),
    }


def needs_env(mat=None) -> bool:
    """LAW5 requires runtime environment views (time, density, volume, mesh size)."""
    return True


def sound_speed(
    mat: Material | dict,
    rho: np.ndarray | float | None = None,
    extra: dict | None = None,
    env: dict | None = None,
) -> np.ndarray | float:
    """Compute JWL sound speed per m5law.F lines 156-172."""
    p = _extract_params(mat)
    rho0 = p["rho0"]
    a = p["a"]
    b = p["b"]
    r1 = p["r1"]
    r2 = p["r2"]
    omega = p["omega"]
    d = p["d"]
    psh = p["psh"]
    bulk = p["bulk"]

    if extra is None:
        extra = {}
    if env is None:
        env = {}

    # Density and relative volume v
    if rho is not None:
        is_scalar = np.isscalar(rho)
        rho_arr = np.atleast_1d(np.asarray(rho, dtype=float))
        v = rho0 / np.maximum(_EM20, rho_arr)
    else:
        v_val = _get_val(extra, "v", "df", default=_get_val(env, "v", "df", default=None))
        if v_val is not None:
            is_scalar = np.isscalar(v_val)
            v = np.atleast_1d(np.asarray(v_val, dtype=float)).copy()
            rho_arr = rho0 / np.maximum(_EM20, v)
        else:
            vol_given = extra.get("vol", extra.get("voln", env.get("vol", env.get("voln", None))))
            vol0_given = extra.get("vol0", env.get("vol0", None))
            if vol_given is not None and vol0_given is not None:
                is_scalar = np.isscalar(vol_given)
                vol_arr_tmp = np.atleast_1d(np.asarray(vol_given, dtype=float))
                vol0_arr_tmp = np.atleast_1d(np.asarray(vol0_given, dtype=float))
                v = vol_arr_tmp / np.maximum(_EM20, vol0_arr_tmp)
                rho_arr = rho0 / np.maximum(_EM20, v)
            else:
                rho_val = _get_val(extra, "rho", default=_get_val(env, "rho", default=rho0))
                is_scalar = np.isscalar(rho_val)
                rho_arr = np.atleast_1d(np.asarray(rho_val, dtype=float))
                v = rho0 / np.maximum(_EM20, rho_arr)

    nel = len(v)
    v = np.maximum(v, _EM20)
    mu = 1.0 / v - 1.0

    # Volume and internal energy
    vol0 = extra.get("vol0", env.get("vol0", 1.0))
    vol0_arr = np.broadcast_to(np.atleast_1d(np.asarray(vol0, dtype=float)), (nel,))
    vol_n = extra.get("vol", extra.get("voln", env.get("vol", env.get("voln", None))))
    if vol_n is None:
        vol_n_arr = v * vol0_arr
    else:
        vol_n_arr = np.broadcast_to(np.atleast_1d(np.asarray(vol_n, dtype=float)), (nel,))
    vol_n_arr = np.maximum(vol_n_arr, _EM20)

    eint = extra.get("eint", env.get("eint", None))
    if eint is None:
        eint_arr = p["e0"] * vol0_arr
    else:
        eint_arr = np.broadcast_to(np.atleast_1d(np.asarray(eint, dtype=float)), (nel,))

    bfrac = extra.get("bfrac", env.get("bfrac", 0.0))
    bfrac_arr = np.broadcast_to(np.atleast_1d(np.asarray(bfrac, dtype=float)), (nel,))

    # m5law.F lines 124-132:
    # R1V = A*W/(R1*DF), WDR1V = A - R1V
    # R2V = B*W/(R2*DF), WDR2V = B - R2V
    # DR1V = W*EINT / MAX(EM20, VOLN)
    # ER1V = EXP(-R1*DF), ER2V = EXP(-R2*DF)
    er1v = np.exp(-r1 * v)
    er2v = np.exp(-r2 * v)
    w_r1_v = omega / np.maximum(_EM20, r1 * v)
    w_r2_v = omega / np.maximum(_EM20, r2 * v)
    wdr1v = a * (1.0 - w_r1_v)
    wdr2v = b * (1.0 - w_r2_v)
    dr1v = omega * eint_arr / vol_n_arr

    # Detonation pressure
    p_jwl = wdr1v * er1v + wdr2v * er2v + dr1v
    if bulk == 0.0:
        p_tot = p["p0"] + p_jwl
    else:
        p_tot = (1.0 - bfrac_arr) * (p["p0"] + bulk * mu) + bfrac_arr * p_jwl
    p_tot = np.maximum(0.0, p_tot) - psh

    # SSP calculation: m5law.F lines 156-161:
    # SSP = A*ER1V*((-W/DF/R1) + R1*DF - W) + B*ER2V*((-W/DF/R2) + R2*DF - W) + DR1V + (P + PSH)*W
    # SSP = SSP * DF
    term1 = a * er1v * (-w_r1_v + r1 * v - omega)
    term2 = b * er2v * (-w_r2_v + r2 * v - omega)
    ssp = (term1 + term2 + dr1v + (p_tot + psh) * omega) * v

    # Sound speed: m5law.F lines 162-172
    if bulk == 0.0:
        c = np.sqrt(np.abs(ssp) / rho0)
        c = np.maximum(c, d * (1.0 - bfrac_arr))
    else:
        c2 = bfrac_arr * (ssp / rho0) + (1.0 - bfrac_arr) * (bulk / rho0)
        c = np.sqrt(np.abs(c2))

    if is_scalar:
        return float(c[0])
    return c


def solid_update(
    first,
    second,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Constitutive stress update for LAW5 (JWL explosive model).

    Ports:
    - ``engine/source/materials/mat/mat005/m5law.F``
    - ``engine/source/materials/mat/mat005/mjwl.F``

    Supports both calling conventions:
    - Standard: ``solid_update(sig, epsp, deps, mat, dt, extra=None, env=None)``
    - Dispatched: ``solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None, env=None)``

    Parameters
    ----------
    sig : (n, 6) ndarray
        Old Cauchy / Jaumann-rotated stress tensor.
    epsp : (n,) ndarray or None
        Plastic strain (unused for LAW5, preserved).
    deps : (n, 6) ndarray
        Strain increment tensor (engineering shear: dgamma_xy, dgamma_yz, dgamma_zx).
    mat : Material or dict
        Material entity or parameters dictionary.
    dt : float
        Time step increment.
    extra : dict, optional
        Extra persistent state buffers:
        - "bfrac": burn fraction (n,)
        - "aburn": afterburning reaction progress (n,)
        - "eint": internal energy per element (n,)
        - "tb": detonation ignition time per element (n,)
        - "rho": current density (n,)
        - "vol": current volume (n,)
        - "vol0": initial volume (n,)
    env : dict, optional
        Global solver environment views:
        - "time": current simulation time t
        - "deltax": characteristic element size Delta_x

    Returns
    -------
    sig : (n, 6) ndarray
        Updated Cauchy stress tensor [-P, -P, -P, 0, 0, 0].
    epsp : (n,) ndarray or None
        Updated plastic strain (None or preserved).
    c : (n,) ndarray
        Per-element sound speed.
    """
    # 0. Dispatch argument signature
    if isinstance(first, (Material, dict)) or not isinstance(first, np.ndarray):
        # Convention: solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None, env=None)
        mat = first
        sig = second
        deps = args[0] if len(args) > 0 else kwargs.get("deps")
        epsp = args[1] if len(args) > 1 else kwargs.get("epsp", None)
        dt = float(args[2] if len(args) > 2 else kwargs.get("dt", 0.0))
        extra = args[3] if len(args) > 3 else kwargs.get("extra", None)
        env = args[4] if len(args) > 4 else kwargs.get("env", None)
    else:
        # Convention: solid_update(sig, epsp, deps, mat, dt, extra=None, env=None)
        sig = first
        epsp = second
        deps = args[0] if len(args) > 0 else kwargs.get("deps")
        mat = args[1] if len(args) > 1 else kwargs.get("mat")
        dt = float(args[2] if len(args) > 2 else kwargs.get("dt", 0.0))
        extra = args[3] if len(args) > 3 else kwargs.get("extra", None)
        env = args[4] if len(args) > 4 else kwargs.get("env", None)

    if extra is None:
        extra = {}
    if env is None:
        env = {}

    p = _extract_params(mat)
    rho0 = p["rho0"]
    rhor = p["rhor"]
    a = p["a"]
    b = p["b"]
    r1 = p["r1"]
    r2 = p["r2"]
    omega = p["omega"]
    d = p["d"]
    pcj = p["pcj"]
    e0 = p["e0"]
    eadd = p["eadd"]
    ibfrac = p["ibfrac"]
    qopt = p["qopt"]
    p0 = p["p0"]
    psh = p["psh"]
    bulk = p["bulk"]
    b_he = p["b_he"]
    tbegin = p["tbegin"]
    tend = p["tend"]
    rr = p["reaction_rate"]
    rr2 = p["reaction_rate2"]
    a_mil = p["a_mil"]
    m_mil = p["m_mil"]
    n_mil = p["n_mil"]
    alpha_unit = p["alpha_unit"]

    # Check dimensionality
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, 6)
        if deps is not None:
            deps = deps.reshape(1, 6)
        if epsp is not None and np.ndim(epsp) > 0:
            epsp = epsp.reshape(1)

    nel = sig.shape[0]
    if nel == 0:
        c_empty = np.zeros(0, dtype=float)
        return (sig.reshape(6,) if is_1d else sig), epsp, c_empty

    # Time and step
    current_time = float(_get_val(env, "time", "tt", "t", default=_get_val(extra, "time", "t", default=0.0)))
    if dt <= 0.0:
        dt = float(_get_val(extra, "dt", default=_get_val(env, "dt", default=0.0)))

    # Mesh size Delta x
    deltax_val = _get_val(env, "deltax", "dx", default=_get_val(extra, "deltax", "dx", default=None))
    if deltax_val is None:
        deltax = np.ones(nel, dtype=float)
    else:
        deltax = np.broadcast_to(np.atleast_1d(np.asarray(deltax_val, dtype=float)), (nel,)).copy()

    # Initial and current volumes
    vol0_val = _get_val(extra, "vol0", default=_get_val(env, "vol0", default=1.0))
    vol0 = np.broadcast_to(np.atleast_1d(np.asarray(vol0_val, dtype=float)), (nel,)).copy()

    # Relative volume v = DF = rho0 / rho = 1 / (1 + mu)
    # Volumetric strain increment deps_vol = deps_xx + deps_yy + deps_zz
    if deps is not None:
        deps_vol = deps[:, 0] + deps[:, 1] + deps[:, 2]
    else:
        deps_vol = np.zeros(nel, dtype=float)

    current_rho = _get_val(extra, "rho", default=_get_val(env, "rho", default=None))
    if current_rho is not None:
        rho_arr = np.broadcast_to(np.atleast_1d(np.asarray(current_rho, dtype=float)), (nel,))
        v = rho0 / np.maximum(_EM20, rho_arr)
    else:
        df_prev = _get_val(extra, "df", default=_get_val(extra, "v", default=None))
        if df_prev is not None:
            df_prev_arr = np.broadcast_to(np.atleast_1d(np.asarray(df_prev, dtype=float)), (nel,))
            v = df_prev_arr * (1.0 + deps_vol)
        else:
            vol_given = _get_val(extra, "vol", "voln", default=_get_val(env, "vol", "voln", default=None))
            if vol_given is not None:
                vol_arr_tmp = np.broadcast_to(np.atleast_1d(np.asarray(vol_given, dtype=float)), (nel,))
                v = (vol_arr_tmp / np.maximum(_EM20, vol0)) * (1.0 + deps_vol)
            else:
                mu_val = _get_val(extra, "mu", default=_get_val(env, "mu", default=None))
                if mu_val is not None:
                    mu_arr = np.broadcast_to(np.atleast_1d(np.asarray(mu_val, dtype=float)), (nel,))
                    v = 1.0 / np.maximum(_EM20, 1.0 + mu_arr)
                else:
                    v = 1.0 + deps_vol

    v = np.maximum(v, _EM20)
    mu = 1.0 / v - 1.0

    # Current element volume V = v * V0
    vol = v * vol0
    vol = np.maximum(vol, _EM20)

    # 1. Burn Fraction kinetics (m5law.F lines 105-121)
    bfrac_old = _get_val(extra, "bfrac", default=_get_val(env, "bfrac", default=None))
    if bfrac_old is None:
        bfrac = np.zeros(nel, dtype=float)
    else:
        bfrac = np.broadcast_to(np.atleast_1d(np.asarray(bfrac_old, dtype=float)), (nel,)).copy()

    tb_val = _get_val(extra, "tb", default=_get_val(env, "tb", default=0.0))
    tb_arr = np.broadcast_to(np.atleast_1d(np.asarray(tb_val, dtype=float)), (nel,)).copy()
    # Handle Fortran convention TB = -TBURN(I) if tb is negative
    tb_phys = np.where(tb_arr < 0.0, -tb_arr, tb_arr)

    # Time control: b1 = D * (t - t_b) / (1.5 * deltax) for t > t_b if ibfrac != 1
    b1 = np.zeros(nel, dtype=float)
    if ibfrac != 1 and d > 0.0:
        mask_time = current_time > tb_phys
        b1 = np.where(mask_time, d * (current_time - tb_phys) / (1.5 * np.maximum(_EM20, deltax)), 0.0)

    # Volumetric control: b2 = B_HE * (1 - v) if ibfrac != 2
    b2 = np.zeros(nel, dtype=float)
    if ibfrac != 2 and b_he > 0.0:
        b2 = b_he * (1.0 - v)

    # m5law.F lines 109-121:
    # If BFRAC < 1.0, BFRAC = max(bfrac, bfrac1, bfrac2), threshold EM04 (1e-4), clamped to [0, 1].
    # If BFRAC >= 1.0, update is skipped (irreversible complete detonation).
    mask_active = (bfrac < 1.0)
    b_cand = np.maximum(bfrac, np.maximum(b1, b2))
    b_cand = np.where(b_cand < _EM04, 0.0, b_cand)
    b_cand = np.clip(b_cand, 0.0, 1.0)
    bfrac = np.where(mask_active, b_cand, 1.0)
    extra["bfrac"] = bfrac.copy()

    # 2. Internal energy & Afterburning (mjwl.F lines 69-156)
    eint_val = _get_val(extra, "eint", default=_get_val(env, "eint", default=None))
    if eint_val is None:
        eint = e0 * vol0.copy()
    else:
        eint = np.broadcast_to(np.atleast_1d(np.asarray(eint_val, dtype=float)), (nel,)).copy()

    # Old pressure from previous stress state: p_old = -tr(sig_old) / 3
    p_old = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    p_eff_old = p_old - psh

    # Delta_V = vol * tr(deps)
    dvol = vol * deps_vol

    # First half-step PdV work on Eint (mjwl.F lines 70-73):
    # EINC = 0.5 * DVOL * (POLD - PSH)
    eint += -0.5 * (p_old + psh) * dvol

    aburn_val = _get_val(extra, "aburn", default=_get_val(env, "aburn", default=None))
    if aburn_val is None:
        aburn = np.zeros(nel, dtype=float)
    else:
        aburn = np.broadcast_to(np.atleast_1d(np.asarray(aburn_val, dtype=float)), (nel,)).copy()

    if eadd > 0.0:
        if qopt == 0:
            # Instantaneous release (mjwl.F lines 94-109)
            lam = np.where(current_time > tbegin, 1.0, 0.0)
            de_ab = np.maximum(0.0, lam - aburn) * eadd * np.maximum(_EM20, vol0)
            eint += de_ab
            aburn = np.where(current_time > tbegin, 1.0, aburn)
        elif qopt == 1:
            # Constant afterburning rate (mjwl.F lines 110-126)
            if current_time <= tbegin:
                lam = np.zeros(nel, dtype=float)
            elif current_time > tend:
                lam = np.ones(nel, dtype=float)
            else:
                lam = np.clip((current_time - tbegin) * rr, 0.0, 1.0)
            de_ab = np.maximum(0.0, lam - aburn) * eadd * np.maximum(_EM20, vol0)
            eint += de_ab
            aburn = np.maximum(aburn, lam)
        elif qopt == 2:
            # Linear afterburning rate (mjwl.F lines 127-143)
            if current_time <= tbegin:
                lam = np.zeros(nel, dtype=float)
            elif current_time > tend:
                lam = np.ones(nel, dtype=float)
            else:
                lam = 0.5 * rr * (current_time ** 2) - rr * tbegin * current_time + rr2
                lam = np.clip(lam, 0.0, 1.0)
            de_ab = np.maximum(0.0, lam - aburn) * eadd * np.maximum(_EM20, vol0)
            eint += de_ab
            aburn = np.maximum(aburn, lam)
        elif qopt == 3:
            # Miller's extension: pressure dependent rate (mjwl.F lines 144-156)
            mask_mil = (p_eff_old > 0.0) & (dt > 1e-20)
            term_m = (1.0 + aburn) ** m_mil
            term_n = (alpha_unit * np.maximum(_EM20, p_eff_old)) ** n_mil
            dlam = np.where(mask_mil, dt * a_mil * term_m * term_n, 0.0)
            lam = np.clip(aburn + dlam, 0.0, 1.0)
            de_ab = np.maximum(0.0, lam - aburn) * eadd * np.maximum(_EM20, vol0)
            eint += de_ab
            aburn = lam

    extra["aburn"] = aburn.copy()

    # 3. JWL Pressure calculation (m5law.F lines 124-149)
    # R1V = A*W/(R1*DF), WDR1V = A - R1V
    # R2V = B*W/(R2*DF), WDR2V = B - R2V
    # DR1V = W*EINT / MAX(EM20, VOLN)
    # ER1V = EXP(-R1*DF), ER2V = EXP(-R2*DF)
    er1v = np.exp(-r1 * v)
    er2v = np.exp(-r2 * v)
    w_r1_v = omega / np.maximum(_EM20, r1 * v)
    w_r2_v = omega / np.maximum(_EM20, r2 * v)
    wdr1v = a * (1.0 - w_r1_v)
    wdr2v = b * (1.0 - w_r2_v)
    dr1v = omega * eint / vol

    p_jwl = wdr1v * er1v + wdr2v * er2v + dr1v

    if bulk == 0.0:
        p_tot = p0 + p_jwl
    else:
        p_tot = (1.0 - bfrac) * (p0 + bulk * mu) + bfrac * p_jwl

    # Cavitation cutoff: m5law.F line 147, mjwl.F line 172
    p_tot = np.maximum(0.0, p_tot) - psh

    # Second half-step PdV work on Eint (mjwl.F lines 176-177):
    # Total PdV work over time step is -0.5 * (p_old + p_tot + 2*psh) * dvol
    eint += -0.5 * (p_tot + psh) * dvol
    extra["eint"] = eint.copy()

    # 4. Stress update: fluid has no deviatoric shear stress (m5law.F lines 176-182)
    # sig = -P * I
    sig[:, 0] = -p_tot
    sig[:, 1] = -p_tot
    sig[:, 2] = -p_tot
    sig[:, 3] = 0.0
    sig[:, 4] = 0.0
    sig[:, 5] = 0.0

    # partial derivative at constant volume (m5law.F line 152)
    dpde = omega / v
    extra["dpde"] = dpde.copy()

    # 5. Sound speed calculation (m5law.F lines 156-172)
    term1 = a * er1v * (-w_r1_v + r1 * v - omega)
    term2 = b * er2v * (-w_r2_v + r2 * v - omega)
    ssp = (term1 + term2 + dr1v + (p_tot + psh) * omega) * v

    if bulk == 0.0:
        c = np.sqrt(np.abs(ssp) / rho0)
        c = np.maximum(c, d * (1.0 - bfrac))
    else:
        c2 = bfrac * (ssp / rho0) + (1.0 - bfrac) * (bulk / rho0)
        c = np.sqrt(np.abs(c2))

    # Persist state views
    extra["df"] = v.copy()
    extra["v"] = v.copy()
    extra["rho"] = (rho0 / v).copy()
    extra["vol"] = vol.copy()

    if is_1d:
        return sig.reshape(6,), epsp, float(c[0])
    return sig, epsp, c


def shell_update(*args, **kwargs):
    """LAW5 is defined strictly for solid and SPH elements."""
    raise NotImplementedError("LAW5 (JWL explosive) is implemented for solid/SPH elements only.")


def consistent_solid_tangent(
    first,
    second=None,
    *args,
    **kwargs,
) -> np.ndarray:
    """Purely volumetric algorithmic tangent for LAW5 hydrodynamic fluid.

    Supports both conventions:
    - ``consistent_solid_tangent(sig, epsp, mat, extra=None, env=None)``
    - ``consistent_solid_tangent(mat, sig, epsp=None, epsp_incr=None, dt=0.0, extra=None, env=None)``

    D = K_eff * (1 (x) 1) where K_eff = rho0 * c^2.
    The upper-left 3x3 block is all K_eff, all other components (deviatoric/shear) are 0.
    """
    if isinstance(first, (Material, dict)):
        mat = first
        sig = second
        extra = kwargs.get("extra", None)
        env = kwargs.get("env", None)
        dict_args = [a for a in args if isinstance(a, dict)]
        if extra is None and len(dict_args) >= 1:
            extra = dict_args[0]
        if env is None and len(dict_args) >= 2:
            env = dict_args[1]
    elif isinstance(second, (Material, dict)):
        sig = first
        mat = second
        extra = kwargs.get("extra", None)
        env = kwargs.get("env", None)
        dict_args = [a for a in args if isinstance(a, dict)]
        if extra is None and len(dict_args) >= 1:
            extra = dict_args[0]
        if env is None and len(dict_args) >= 2:
            env = dict_args[1]
    else:
        sig = first
        mat = args[0] if len(args) > 0 else kwargs.get("mat")
        extra = kwargs.get("extra", None)
        env = kwargs.get("env", None)
        dict_args = [a for a in args[1:] if isinstance(a, dict)]
        if extra is None and len(dict_args) >= 1:
            extra = dict_args[0]
        if env is None and len(dict_args) >= 2:
            env = dict_args[1]

    if extra is None:
        extra = {}
    if env is None:
        env = {}

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, 6)
    nel = sig.shape[0]
    if nel == 0:
        return np.zeros((0, 6, 6), dtype=float)

    p = _extract_params(mat)
    rho0 = p["rho0"]

    # Compute current sound speed
    c = sound_speed(mat, extra=extra, env=env)
    c_arr = np.broadcast_to(np.atleast_1d(np.asarray(c, dtype=float)), (nel,))

    # Algorithmic bulk modulus K_eff = rho0 * c^2
    k_eff = rho0 * (c_arr ** 2)

    D = np.zeros((nel, 6, 6), dtype=float)
    # Pure volumetric response: upper-left 3x3 block is all K_eff
    for i in range(3):
        for j in range(3):
            D[:, i, j] = k_eff

    if is_1d:
        return D.reshape(6, 6)
    return D


solid_tangent = consistent_solid_tangent


def _register():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    except (ImportError, ValueError):
        try:
            from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        except ImportError:
            return
    for k in (5, "5", "LAW5", "JWL", "MAT_JWL"):
        MAT_PHYSICS_REGISTRY[k] = build_law5


build_law05 = build_law5
build_jwl = build_law5
solid_tangent = consistent_solid_tangent


_register()
