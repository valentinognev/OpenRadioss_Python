"""LAW4 — Hydrodynamic Johnson-Cook (/MAT/LAW4, /MAT/HYD_JCOOK).

Function: SIGEPS_04 / M4LAW (lines 1-244)
Upstream Fortran origins:
- ``$OR_SRC/engine/source/materials/mat/mat004/m4law.F`` (solid constitutive update, subroutine M4LAW / SIGEPS04)
- ``$OR_SRC/starter/source/materials/mat/mat004/hm_read_mat04.F`` (starter card reader & defaults, subroutine HM_READ_MAT04)
- ``$OR_SRC/common_source/eos/gruneisen.F`` (Mie-Grüneisen equation of state, subroutine GRUNEISEN)
- ``$OR_SRC/hm_cfg_files/config/CFG/radioss110/MAT/matl4_hyd_jcook.cfg`` (CFG attributes & format)

Theory
------
LAW4 models elastic-plastic hydrodynamic behavior using a Johnson-Cook yield surface
with isotropic power-law hardening, strain rate sensitivity, and thermal softening,
coupled with a hydrodynamic equation of state (linear bulk modulus, embedded polynomial EOS, or Grüneisen EOS).

1. Deviatoric elastic trial (m4law.F lines 90-95, 106-113):
   P_old = -tr(sigma_old) / 3
   DAV = -tr(Delta eps) / 3
   s_trial = sigma_old + P_old * I + 2G * (Delta eps + DAV * I)

2. Strain rate measure EPD (m4law.F lines 116-117):
   For D = Delta eps / dt:
   eps_dot = max(|D_11|, |D_22|, |D_33|, 0.5|D_12|, 0.5|D_23|, 0.5|D_31|) for dt > 0, else 0

3. Strain rate factor C_E (m4law.F lines 140-144):
   C_E = 1.0  if eps_dot <= eps_dot_0
   C_E = 1.0 + C * ln(eps_dot / eps_dot_0)  if eps_dot > eps_dot_0

4. Temperature & thermal softening factor C_T (m4law.F lines 128-138):
   If T >= T_melt:
       sigma_y = 0, QH = 0, scale = 0 (deviatoric stress relaxed to 0)
   Else if T > T_0:
       m_eff = 1.0 if T > T_max else m
       T* = clamp((T - T_0) / max(T_melt - T_0, 1e-20), 0.0, 1.0)
       C_T = 1.0 - (T*)^m_eff
   Else:
       C_T = 1.0

5. Hardening C_H (m4law.F lines 146-152):
   if eps_p <= 0: C_H = A
   else if eps_p > eps_p_max: C_H = 0
   else: C_H = A + B * eps_p^N

6. Yield stress sigma_y (m4law.F line 154):
   sigma_y = min(sig_max, C_H) * C_E * C_T

7. Hardening modulus QH = d(sigma_y)/d(eps_p) (m4law.F lines 158-164):
   If N >= 1.0: QH = (B * N * eps_p^(N - 1)) * C_E * C_T
   Else if eps_p > 0: QH = (B * N / eps_p^(1 - N)) * C_E * C_T
   Else: QH = 0.0

8. Radial return (IPLA=0, 1, 2, m4law.F lines 168-219):
   sigma_vm = sqrt(3 J_2)
   If sigma_vm > sigma_y:
       scale = sigma_y / sigma_vm
       s = scale * s
       Delta eps_p = (1.0 - scale) * sigma_vm / (3G + QH)  (IPLA=0)
       eps_p += Delta eps_p

9. Adiabatic temperature rise (m4law.F lines 232-235):
   If rho * C_p > 0:
       Delta T = sigma_y * Delta eps_p / (rho * C_p)
       T += Delta T

10. Hydrodynamic pressure & pressure cutoff (m4law.F line 91, eosmain.F line 229, eospolyno.F line 104):
    P_new = K * (rho / rho0 - 1.0) if rho provided else P_old + 3K * DAV
    P_new = max(P_new, P_min)  (P_min is negative tension cutoff, e.g. -1e30)
    sigma = s - P_new * I

11. Sound speed (m4law.F lines 99-102):
    c = sqrt((K + 4/3 * G) / rho0)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyradioss.model.entities import EquationOfState, Material

_EM15 = 1e-15
_EM20 = 1e-20
_INF = 1e20


def _ensure_params(mat: Material) -> dict:
    """Ensure material params contain both CFG and direct keys with robust defaults."""
    if hasattr(mat, "params") and mat.params is not None:
        p = mat.params
    elif isinstance(mat, dict):
        p = mat.get("params", mat)
    else:
        p = getattr(mat, "params", {})

    # Extract or calculate E and nu
    e = float(p.get("E") if p.get("E") is not None else (p.get("MAT_E") or 0.0))
    nu = float(p.get("nu") if p.get("nu") is not None else (p.get("MAT_NU") or 0.0))

    if (1.0 + nu) != 0.0:
        g_calc = e / (2.0 * (1.0 + nu))
    else:
        g_calc = 0.0

    if (1.0 - 2.0 * nu) != 0.0:
        k_calc = e / (3.0 * (1.0 - 2.0 * nu))
    else:
        k_calc = 0.0

    g = float(p.get("G") if p.get("G") is not None else g_calc)
    k = float(p.get("K") if p.get("K") is not None else k_calc)

    # Yield and hardening parameters
    a = float(p.get("A") if p.get("A") is not None else (p.get("MAT_SIGY") or 0.0))
    b = float(p.get("B") if p.get("B") is not None else (p.get("MAT_BETA") or 0.0))

    n_val = p.get("N") if p.get("N") is not None else (p.get("n") if p.get("n") is not None else p.get("MAT_HARD"))
    n = float(n_val if n_val is not None else 0.0)
    if n == 0.0 or n == 1.0:
        n = 1.0001

    eps_max_val = p.get("eps_max") if p.get("eps_max") is not None else p.get("MAT_EPS")
    eps_max = float(eps_max_val if eps_max_val is not None else 0.0)
    if eps_max == 0.0:
        eps_max = _INF

    sig_max_val = p.get("sig_max") if p.get("sig_max") is not None else p.get("MAT_SIG")
    sig_max = float(sig_max_val if sig_max_val is not None else 0.0)
    if sig_max == 0.0:
        sig_max = _INF

    # Strain rate parameters
    c = float(p.get("C") if p.get("C") is not None else (p.get("MAT_SRC") or 0.0))
    srp_val = p.get("eps0") if p.get("eps0") is not None else (p.get("eps_dot_0") if p.get("eps_dot_0") is not None else p.get("MAT_SRP"))
    if c == 0.0:
        eps0 = 1.0
    else:
        eps0 = float(srp_val) if srp_val is not None and float(srp_val) != 0.0 else 1.0

    # Temperature parameters
    m_val = p.get("m") if p.get("m") is not None else p.get("MAT_M")
    m = float(m_val) if m_val is not None and float(m_val) != 0.0 else 1.0

    tmelt_val = p.get("Tmelt") if p.get("Tmelt") is not None else p.get("MAT_TMELT")
    tmelt = float(tmelt_val if tmelt_val is not None else 0.0)
    if tmelt == 0.0:
        tmelt = _INF

    tmax_val = p.get("Tmax") if p.get("Tmax") is not None else p.get("MAT_TMAX")
    tmax = float(tmax_val if tmax_val is not None else 0.0)
    if tmax == 0.0:
        tmax = _INF

    rho_cp = float(p.get("rho_cp") if p.get("rho_cp") is not None else (p.get("MAT_SPHEAT") or 0.0))

    t0_val = p.get("T0") if p.get("T0") is not None else p.get("MAT_T0")
    t0 = float(t0_val) if t0_val is not None and float(t0_val) > 0.0 else 300.0

    # Pressure cutoff
    pmin_val = p.get("pmin") if p.get("pmin") is not None else p.get("MAT_PC")
    pmin = float(pmin_val) if pmin_val is not None and float(pmin_val) != 0.0 else -1e30

    # Embedded polynomial EOS parameters
    psh = float(p.get("psh") if p.get("psh") is not None else (p.get("MAT_PSH") or 0.0))
    e0 = float(p.get("e0") if p.get("e0") is not None else (p.get("MAT_EA") or p.get("MAT_E0") or 0.0))
    c0 = float(p.get("c0") if p.get("c0") is not None else (p.get("MAT_C0") or 0.0))
    c1 = float(p.get("c1") if p.get("c1") is not None else (p.get("MAT_C1") or 0.0))
    c2 = float(p.get("c2") if p.get("c2") is not None else (p.get("MAT_C2") or 0.0))
    c3 = float(p.get("c3") if p.get("c3") is not None else (p.get("MAT_C3") or 0.0))
    c4 = float(p.get("c4") if p.get("c4") is not None else (p.get("MAT_C4") or 0.0))
    c5 = float(p.get("c5") if p.get("c5") is not None else (p.get("MAT_C5") or 0.0))

    # Populate direct keys
    p["E"] = e
    p["nu"] = nu
    p["G"] = g
    p["K"] = k
    p["A"] = a
    p["B"] = b
    p["N"] = n
    p["n"] = n
    p["eps_max"] = eps_max
    p["sig_max"] = sig_max
    p["C"] = c
    p["eps0"] = eps0
    p["m"] = m
    p["Tmelt"] = tmelt
    p["Tmax"] = tmax
    p["rho_cp"] = rho_cp
    p["T0"] = t0
    p["pmin"] = pmin
    p["psh"] = psh
    p["e0"] = e0
    p["c0"] = c0
    p["c1"] = c1
    p["c2"] = c2
    p["c3"] = c3
    p["c4"] = c4
    p["c5"] = c5

    # Populate CFG keys
    p["MAT_E"] = e
    p["MAT_NU"] = nu
    p["MAT_SIGY"] = a
    p["MAT_BETA"] = b
    p["MAT_HARD"] = n
    p["MAT_EPS"] = eps_max
    p["MAT_SIG"] = sig_max
    p["MAT_SRC"] = c
    p["MAT_SRP"] = eps0
    p["MAT_M"] = m
    p["MAT_TMELT"] = tmelt
    p["MAT_TMAX"] = tmax
    p["MAT_SPHEAT"] = rho_cp
    p["MAT_T0"] = t0
    p["MAT_PC"] = pmin
    p["MAT_PSH"] = psh
    p["MAT_EA"] = e0
    p["MAT_E0"] = e0
    p["MAT_C0"] = c0
    p["MAT_C1"] = c1
    p["MAT_C2"] = c2
    p["MAT_C3"] = c3
    p["MAT_C4"] = c4
    p["MAT_C5"] = c5

    return p


def build_law04(rec) -> Material:
    """Card parsing and validation -> Material for LAW4 (/MAT/LAW4, /MAT/HYD_JCOOK)."""
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params is not None else {}
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
        eos = rec.eos
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = rec.get("id", 1)
        title = rec.get("title", "LAW4")
        density = None
        for k in ("density", "rho0", "rho", "MAT_RHO"):
            if k in rec and rec[k] is not None:
                density = float(rec[k])
                break
        if density is None:
            for k in ("density", "rho0", "rho", "MAT_RHO"):
                if k in p and p[k] is not None:
                    density = float(p[k])
                    break
        if density is None:
            density = 0.0
        eos = rec.get("eos")
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "LAW4")
        density = float(getattr(rec, "rho0", getattr(rec, "density", 0.0)))
        eos = getattr(rec, "eos", None)

    if density <= 0.0:
        raise ValueError(f"/MAT/LAW4/{mat_id}: Initial density rho0 must be > 0 (got {density})")

    e_val = p.get("E") if p.get("E") is not None else p.get("MAT_E")
    if e_val is None:
        raise ValueError(f"/MAT/LAW4/{mat_id}: Young's modulus E must be > 0 (not provided)")
    e = float(e_val)
    if e <= 0.0:
        raise ValueError(f"/MAT/LAW4/{mat_id}: Young's modulus E must be > 0 (got {e})")

    nu_val = p.get("nu") if p.get("nu") is not None else p.get("MAT_NU")
    if nu_val is None:
        raise ValueError(f"/MAT/LAW4/{mat_id}: Poisson's ratio nu must be in [0, 0.5) (not provided)")
    nu = float(nu_val)
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"/MAT/LAW4/{mat_id}: Poisson's ratio nu must be in [0, 0.5) (got {nu})")

    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))

    a = float(p.get("A") if p.get("A") is not None else (p.get("MAT_SIGY") or 0.0))
    b = float(p.get("B") if p.get("B") is not None else (p.get("MAT_BETA") or 0.0))

    n_raw = p.get("N") if p.get("N") is not None else (p.get("n") if p.get("n") is not None else p.get("MAT_HARD"))
    n = float(n_raw if n_raw is not None else 0.0)
    if n == 0.0 or n == 1.0:
        n = 1.0001

    eps_max_raw = p.get("eps_max") if p.get("eps_max") is not None else p.get("MAT_EPS")
    eps_max = float(eps_max_raw if eps_max_raw is not None else 0.0)
    if eps_max == 0.0:
        eps_max = _INF

    sig_max_raw = p.get("sig_max") if p.get("sig_max") is not None else p.get("MAT_SIG")
    sig_max = float(sig_max_raw if sig_max_raw is not None else 0.0)
    if sig_max == 0.0:
        sig_max = _INF

    c_rate = float(p.get("C") if p.get("C") is not None else (p.get("MAT_SRC") or 0.0))
    srp_raw = p.get("eps0") if p.get("eps0") is not None else (p.get("eps_dot_0") if p.get("eps_dot_0") is not None else p.get("MAT_SRP"))
    if c_rate == 0.0:
        eps0 = 1.0
    else:
        eps0 = float(srp_raw) if srp_raw is not None and float(srp_raw) != 0.0 else 1.0

    m_raw = p.get("m") if p.get("m") is not None else p.get("MAT_M")
    m = float(m_raw) if m_raw is not None and float(m_raw) != 0.0 else 1.0

    tmelt_raw = p.get("Tmelt") if p.get("Tmelt") is not None else p.get("MAT_TMELT")
    tmelt = float(tmelt_raw if tmelt_raw is not None else 0.0)
    if tmelt == 0.0:
        tmelt = _INF

    tmax_raw = p.get("Tmax") if p.get("Tmax") is not None else p.get("MAT_TMAX")
    tmax = float(tmax_raw if tmax_raw is not None else 0.0)
    if tmax == 0.0:
        tmax = _INF

    rho_cp = float(p.get("rho_cp") if p.get("rho_cp") is not None else (p.get("MAT_SPHEAT") or 0.0))

    t0_raw = p.get("T0") if p.get("T0") is not None else p.get("MAT_T0")
    t0 = float(t0_raw) if t0_raw is not None and float(t0_raw) > 0.0 else 300.0

    pmin_raw = p.get("pmin") if p.get("pmin") is not None else p.get("MAT_PC")
    pmin = float(pmin_raw) if pmin_raw is not None and float(pmin_raw) != 0.0 else -1e30

    # Embedded polynomial EOS
    c0 = float(p.get("c0") if p.get("c0") is not None else (p.get("MAT_C0") or 0.0))
    c1 = float(p.get("c1") if p.get("c1") is not None else (p.get("MAT_C1") or 0.0))
    c2 = float(p.get("c2") if p.get("c2") is not None else (p.get("MAT_C2") or 0.0))
    c3 = float(p.get("c3") if p.get("c3") is not None else (p.get("MAT_C3") or 0.0))
    c4 = float(p.get("c4") if p.get("c4") is not None else (p.get("MAT_C4") or 0.0))
    c5 = float(p.get("c5") if p.get("c5") is not None else (p.get("MAT_C5") or 0.0))
    e0 = float(p.get("e0") if p.get("e0") is not None else (p.get("MAT_EA") or p.get("MAT_E0") or 0.0))
    psh = float(p.get("psh") if p.get("psh") is not None else (p.get("MAT_PSH") or 0.0))

    if isinstance(eos, dict):
        eos = EquationOfState(
            kind=eos.get("kind", "POLYNOMIAL"),
            params=eos.get("params", eos),
            rho0=density,
        )

    # Detect Grüneisen EOS parameters (/EOS/GRUNEISEN, gruneisen.F)
    has_gruneisen = (
        str(p.get("eos_kind", p.get("eos_type", ""))).upper() in ("GRUNEISEN", "GRUN", "2")
        or (isinstance(rec, dict) and str(rec.get("eos_kind", rec.get("kind", ""))).upper() in ("GRUNEISEN", "GRUN"))
        or any(k in p for k in ("gamma0", "s1", "eos_gamma0", "eos_s1", "EOS_GRUN_1", "GAMMA0"))
    )

    has_embedded_eos = any(
        k in p for k in ("c0", "c1", "c2", "c3", "c4", "c5", "e0", "psh",
                         "MAT_C0", "MAT_C1", "MAT_C2", "MAT_C3", "MAT_C4", "MAT_C5",
                         "MAT_EA", "MAT_E0", "MAT_PSH")
    )
    if eos is None and has_gruneisen:
        # Grüneisen parameters: c (bulk sound speed), s1, s2, s3 (Hugoniot slopes), gamma0, a
        c_sound = p.get("c_sound") or p.get("eos_c") or p.get("C_GRUN")
        if c_sound is None:
            c_val = p.get("c")
            if c_val is not None and float(c_val) != c_rate:
                c_sound = float(c_val)
            else:
                c_sound = np.sqrt(k / density) if density > 0.0 else 0.0
        grun_params = {
            "c": float(c_sound),
            "s1": float(p.get("s1") if p.get("s1") is not None else (p.get("eos_s1") or p.get("S1") or 0.0)),
            "s2": float(p.get("s2") if p.get("s2") is not None else (p.get("eos_s2") or p.get("S2") or 0.0)),
            "s3": float(p.get("s3") if p.get("s3") is not None else (p.get("eos_s3") or p.get("S3") or 0.0)),
            "gamma0": float(p.get("gamma0") if p.get("gamma0") is not None else (p.get("eos_gamma0") or p.get("GAMMA0") or 0.0)),
            "a": float(p.get("a_grun") if p.get("a_grun") is not None else (p.get("eos_a") or p.get("A_GRUN") or 0.0)),
            "e0": float(p.get("e0") if p.get("e0") is not None else (p.get("eos_e0") or 0.0)),
            "rho0_card": float(p.get("rho0_card") or density),
        }
        eos = EquationOfState(kind="GRUNEISEN", params=grun_params, rho0=density)
    elif eos is None and has_embedded_eos:
        eos_params = {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
            "e0": e0,
            "psh": psh,
            "pmin": pmin,
        }
        eos = EquationOfState(kind="POLYNOMIAL", params=eos_params, rho0=density)

    params = {
        "E": e,
        "nu": nu,
        "G": g,
        "K": k,
        "A": a,
        "B": b,
        "N": n,
        "n": n,
        "eps_max": eps_max,
        "sig_max": sig_max,
        "C": c_rate,
        "eps0": eps0,
        "m": m,
        "Tmelt": tmelt,
        "Tmax": tmax,
        "rho_cp": rho_cp,
        "T0": t0,
        "pmin": pmin,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "e0": e0,
        "psh": psh,
        "MAT_E": e,
        "MAT_NU": nu,
        "MAT_SIGY": a,
        "MAT_BETA": b,
        "MAT_HARD": n,
        "MAT_EPS": eps_max,
        "MAT_SIG": sig_max,
        "MAT_SRC": c_rate,
        "MAT_SRP": eps0,
        "MAT_M": m,
        "MAT_TMELT": tmelt,
        "MAT_TMAX": tmax,
        "MAT_SPHEAT": rho_cp,
        "MAT_T0": t0,
        "MAT_PC": pmin,
        "MAT_PSH": psh,
        "MAT_EA": e0,
        "MAT_E0": e0,
        "MAT_C0": c0,
        "MAT_C1": c1,
        "MAT_C2": c2,
        "MAT_C3": c3,
        "MAT_C4": c4,
        "MAT_C5": c5,
    }
    for k, v in p.items():
        if k not in params:
            params[k] = v

    mat = Material(id=mat_id, law=4, rho0=density, title=title, params=params, eos=eos)
    _ensure_params(mat)
    return mat


def solid_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray = None, dt: float = 0.0, extra: dict = None):
    """Vectorized 3D solid stress update for LAW4 (hydrodynamic Johnson-Cook).

    Ports ``engine/source/materials/mat/mat004/m4law.F`` (IPLA=0 branch).

    Parameters
    ----------
    mat : Material
        Material entity with parameters.
    sig : (n, 6) ndarray
        Old (Jaumann-rotated) stress [xx, yy, zz, xy, yz, zx].
    deps : (n, 6) ndarray
        Strain increment tensor (engineering shear: dgamma_xy, dgamma_yz, dgamma_zx).
    epsp : (n,) ndarray, optional
        Accumulated equivalent plastic strain.
    dt : float, optional
        Time step increment.
    extra : dict, optional
        Extra state views, e.g. "temp" (temperature) and "rho" (current density).

    Returns
    -------
    sig : (n, 6) ndarray
        Updated stress.
    epsp : (n,) ndarray
        Updated accumulated plastic strain.
    c : (n,) ndarray or None
        Sound speed array c = sqrt((K + 4/3*G)/rho0).
    """
    if sig.shape[0] == 0:
        return sig, epsp, None

    nel = sig.shape[0]
    p = _ensure_params(mat)
    G = float(p["G"])
    K = float(p["K"])
    A = float(p["A"])
    B = float(p["B"])
    N = float(p["N"])
    eps_max = float(p["eps_max"])
    sig_max = float(p["sig_max"])
    c_rate = float(p["C"])
    eps0 = float(p["eps0"])
    m = float(p["m"])
    Tmelt = float(p["Tmelt"])
    Tmax = float(p["Tmax"])
    rho_cp = float(p["rho_cp"])
    T0 = float(p["T0"])
    pmin = float(p["pmin"])
    rho0 = float(getattr(mat, "rho0", 1.0) or 1.0)

    sig = sig.copy()
    if epsp is None:
        epsp = np.zeros(nel, dtype=float)
    else:
        epsp = np.array(epsp, dtype=float, copy=True)

    # 1. Hydrostatic vs deviatoric split (m4law.F lines 90-95, 106-113)
    # Fortran: P = -THIRD*(SIG1+SIG2+SIG3), DAV = -THIRD*(D1+D2+D3)
    # SIG1_dev = SIG1 + P + 2G*(D1 + DAV)
    P_old = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    dav = -(deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0

    s = np.empty_like(sig)
    s[:, 0] = sig[:, 0] + P_old + 2.0 * G * (deps[:, 0] + dav)
    s[:, 1] = sig[:, 1] + P_old + 2.0 * G * (deps[:, 1] + dav)
    s[:, 2] = sig[:, 2] + P_old + 2.0 * G * (deps[:, 2] + dav)
    s[:, 3] = sig[:, 3] + G * deps[:, 3]
    s[:, 4] = sig[:, 4] + G * deps[:, 4]
    s[:, 5] = sig[:, 5] + G * deps[:, 5]

    # 2. Strain rate measure EPD (m4law.F lines 116-117)
    if dt > 0.0:
        D = deps / dt
        epd = np.maximum.reduce([
            np.abs(D[:, 0]),
            np.abs(D[:, 1]),
            np.abs(D[:, 2]),
            0.5 * np.abs(D[:, 3]),
            0.5 * np.abs(D[:, 4]),
            0.5 * np.abs(D[:, 5]),
        ])
    else:
        epd = np.zeros(nel, dtype=float)

    # 3. Strain rate factor C_E (m4law.F lines 140-144)
    if extra is not None and "ce" in extra and extra["ce"] is not None:
        raw_ce = extra["ce"]
        if np.isscalar(raw_ce):
            ce = np.full(nel, float(raw_ce), dtype=float)
        else:
            ce = np.array(raw_ce, dtype=float, copy=True)
    elif c_rate > 0.0 and eps0 > 0.0:
        ratio = np.maximum(epd / eps0, 1.0)
        ce = 1.0 + c_rate * np.log(ratio)
    else:
        ce = np.ones(nel, dtype=float)

    # 4. Temperature T
    if extra is not None and "temp" in extra and extra["temp"] is not None:
        raw_t = extra["temp"]
        if np.isscalar(raw_t):
            T = np.full(nel, float(raw_t), dtype=float)
        else:
            T = np.array(raw_t, dtype=float, copy=True)
            if T.shape != (nel,):
                T = np.full(nel, float(T.flat[0]) if T.size > 0 else T0, dtype=float)
    else:
        T = np.full(nel, T0, dtype=float)

    # 5. Thermal softening factor C_T (m4law.F lines 128-138)
    ct = np.ones(nel, dtype=float)
    melted = T >= Tmelt
    above_t0 = (T > T0) & (~melted)
    if np.any(above_t0):
        T_sub = T[above_t0]
        m_eff = np.where(T_sub > Tmax, 1.0, m)
        denom = max(Tmelt - T0, 1e-20)
        tstar = np.clip((T_sub - T0) / denom, 0.0, 1.0)
        ct[above_t0] = 1.0 - (tstar ** m_eff)

    # 6. Hardening C_H (m4law.F lines 146-152)
    ch = np.empty(nel, dtype=float)
    le_zero = epsp <= 0.0
    gt_max = epsp > eps_max
    normal = (~le_zero) & (~gt_max)

    ch[le_zero] = A
    ch[gt_max] = 0.0
    ch[normal] = A + B * (epsp[normal] ** N)

    # 7. Yield stress sigma_y (m4law.F line 154)
    sig_y = np.minimum(sig_max, ch) * ce * ct
    sig_y[melted] = 0.0

    # 8. Hardening modulus QH (m4law.F lines 158-164)
    qh = np.zeros(nel, dtype=float)
    if N >= 1.0:
        qh = (B * N * np.maximum(epsp, 0.0) ** (N - 1.0)) * ce * ct
    else:
        pos = epsp > 0.0
        qh[pos] = (B * N / np.maximum(epsp[pos], _EM15) ** (1.0 - N)) * ce[pos] * ct[pos]
    qh[melted] = 0.0
    qh[ch >= sig_max] = 0.0
    qh[gt_max] = 0.0

    # 9. von Mises equivalent sigma_vm = sqrt(3 J2)
    j2 = (0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2)
          + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    sig_vm = np.sqrt(3.0 * np.maximum(j2, 0.0))

    # 10. Radial return (IPLA=0, 1, 2, m4law.F lines 168-219)
    ipla = int(extra.get("ipla", p.get("ipla", p.get("IPLA", 0)))) if extra is not None else int(p.get("ipla", p.get("IPLA", 0)))
    scale = np.ones(nel, dtype=float)
    dpla = np.zeros(nel, dtype=float)

    if np.any(melted):
        scale[melted] = 0.0
        s[melted] = 0.0

    unmelted = ~melted
    yielding = unmelted & (sig_vm > sig_y)
    if np.any(yielding):
        sc = np.where(sig_vm[yielding] > 0.0, sig_y[yielding] / sig_vm[yielding], 0.0)
        scale[yielding] = sc
        for c_idx in range(6):
            s[yielding, c_idx] *= sc

        if ipla == 2:
            denom = 3.0 * G
        else:
            denom = 3.0 * G + qh[yielding]
            denom = np.where(denom > _EM15, denom, _EM15)

        dpla[yielding] = (1.0 - sc) * sig_vm[yielding] / denom

        if ipla == 1:
            ak = sig_y[yielding] + dpla[yielding] * qh[yielding]
            sc2 = np.minimum(1.0, ak / np.maximum(sig_vm[yielding], _EM15))
            ratio = np.where(sc > 0.0, sc2 / np.maximum(sc, _EM15), 0.0)
            for c_idx in range(6):
                s[yielding, c_idx] *= ratio

        epsp[yielding] += dpla[yielding]

    # 11. Temperature rise (m4law.F lines 232-235)
    if rho_cp > 0.0:
        delta_T = sig_y * dpla / rho_cp
        T += delta_T
        if extra is not None:
            if isinstance(extra.get("temp"), np.ndarray) and extra["temp"].shape == T.shape:
                extra["temp"][:] = T
            else:
                extra["temp"] = T.copy()

    # 12. Hydrodynamic pressure update & cutoff (m4law.F line 91, eosmain.F line 229, gruneisen.F)
    # Pressure P is positive in compression, negative in tension.
    if getattr(mat, "eos", None) is not None:
        from pyradioss.materials import eos as eos_mod

        if extra is not None and "rho" in extra and extra["rho"] is not None:
            rho_curr = np.asarray(extra["rho"], dtype=float)
            mu = rho_curr / rho0 - 1.0
            dv = 1.0 / (1.0 + np.maximum(mu, -0.999999)) - 1.0
        elif extra is not None and "mu" in extra and extra["mu"] is not None:
            mu = np.asarray(extra["mu"], dtype=float)
            dv = 1.0 / (1.0 + np.maximum(mu, -0.999999)) - 1.0
        else:
            # Volumetric strain increment: tr(deps) = -3 * dav
            mu = 3.0 * dav
            dv = -mu

        e_int = None
        p_eos_old = None
        if extra is not None:
            if "e_eos" in extra and extra["e_eos"] is not None:
                e_int = np.atleast_1d(np.asarray(extra["e_eos"], dtype=float))
            elif "eint" in extra and extra["eint"] is not None:
                e_int = np.atleast_1d(np.asarray(extra["eint"], dtype=float))
            if "p_eos" in extra and extra["p_eos"] is not None:
                p_eos_old = np.atleast_1d(np.asarray(extra["p_eos"], dtype=float))

        if e_int is None:
            e0 = float(getattr(mat.eos, "params", {}).get("e0", 0.0))
            e_int = np.full(nel, e0, dtype=float)
        if p_eos_old is None:
            p_eos_old = P_old

        # Deviatoric work rate de_dev = s_mid : deps
        s_mid = 0.5 * (s + (sig + P_old[:, None]))
        de_dev = np.einsum("nk,nk->n", s_mid, deps)

        mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
        dv_arr = np.atleast_1d(np.asarray(dv, dtype=float))
        if mu_arr.shape != (nel,):
            mu_arr = np.full(nel, float(mu_arr.flat[0]), dtype=float)
        if dv_arr.shape != (nel,):
            dv_arr = np.full(nel, float(dv_arr.flat[0]), dtype=float)

        P_eos, e_eos, c2 = eos_mod.update(
            mat.eos, mu_arr, dv_arr, e_int, p_eos_old, de_dev
        )

        if extra is not None:
            if "e_eos" in extra and isinstance(extra["e_eos"], np.ndarray):
                extra["e_eos"][:] = e_eos
            elif "e_eos" in extra:
                extra["e_eos"] = e_eos
            if "p_eos" in extra and isinstance(extra["p_eos"], np.ndarray):
                extra["p_eos"][:] = P_eos
            elif "p_eos" in extra:
                extra["p_eos"] = P_eos

        P_new = np.asarray(P_eos, dtype=float)
        if P_new.ndim == 0:
            P_new = np.full(nel, float(P_new), dtype=float)

        # Sound speed combining EOS bulk stiffness c2 and shear G:
        # Fortran m4law.F line 100: DPDM = DPDM + 4/3*G; SSP = sqrt(abs(DPDM)/RHO0)
        c_val = np.sqrt(np.maximum(c2 + (4.0 / 3.0) * G / rho0, 0.0))
        c = np.asarray(c_val, dtype=float)
        if c.ndim == 0 or c.shape != (nel,):
            c = np.full(nel, float(c), dtype=float)
    else:
        if extra is not None and "rho" in extra and extra["rho"] is not None:
            P_new = K * (extra["rho"] / rho0 - 1.0)
        else:
            # Incremental hypoelastic bulk update: dP = -K * tr(deps) = 3 * K * dav
            P_new = P_old + 3.0 * K * dav
        c_val = np.sqrt(max((K + (4.0 / 3.0) * G) / rho0, 0.0))
        c = np.full(nel, c_val, dtype=float)

    # Pressure cutoff: P >= Pmin (Pmin is negative tension cutoff, e.g. -1e30)
    P_new = np.maximum(P_new, pmin)

    # Recombine deviatoric stress and pressure into Cauchy stress: sigma = s - P * I
    sig[:, 0] = s[:, 0] - P_new
    sig[:, 1] = s[:, 1] - P_new
    sig[:, 2] = s[:, 2] - P_new
    sig[:, 3] = s[:, 3]
    sig[:, 4] = s[:, 4]
    sig[:, 5] = s[:, 5]

    return sig, epsp, c


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """Plane-stress shell update is not supported for LAW4."""
    raise NotImplementedError("LAW4 (hydrodynamic Johnson-Cook) is implemented for solid/SPH elements only.")


def consistent_solid_tangent(mat: Material, sig: np.ndarray, epsp: np.ndarray,
                            epsp_incr: np.ndarray, extra: dict = None,
                            dt: float = 0.0) -> np.ndarray:
    """Algorithmic consistent elastoplastic tangent matrix for solids (Simo & Hughes).

    Matches the J2 radial-return tangent formulation in ``law02_johnson_cook.py``:
    D = C - a * (C - K 1(x)1) + b * (N (x) N)
    where a = 3G * dep / q_tr, b = 6G^2 * (dep / q_tr - 1 / (3G + H)).
    """
    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 6, 6))

    p = _ensure_params(mat)
    G = float(p["G"])
    Kb = float(p["K"])
    rho0 = float(getattr(mat, "rho0", 1.0) or 1.0)
    if getattr(mat, "eos", None) is not None:
        from pyradioss.materials import eos as eos_mod
        mu_tan = 0.0
        e_tan = float(getattr(mat.eos, "params", {}).get("e0", 0.0))
        if extra is not None:
            if "mu" in extra and extra["mu"] is not None:
                mu_tan = float(np.mean(extra["mu"]))
            elif "rho" in extra and extra["rho"] is not None:
                mu_tan = float(np.mean(extra["rho"])) / rho0 - 1.0
            if "e_eos" in extra and extra["e_eos"] is not None:
                e_tan = float(np.mean(extra["e_eos"]))
        try:
            c_eos = float(eos_mod.sound_speed(mat.eos, mu_tan, e_tan))
            if c_eos > 0.0:
                Kb = rho0 * (c_eos ** 2)
        except Exception:
            pass
    B = float(p["B"])
    N = float(p["N"])
    sig_max = float(p["sig_max"])
    c_rate = float(p.get("C", 0.0))
    eps0 = float(p.get("eps0", 1.0))
    T0 = float(p.get("T0", 300.0))
    Tmelt = float(p.get("Tmelt", _INF))
    Tmax = float(p.get("Tmax", _INF))
    m = float(p.get("m", 1.0))

    lam = Kb - 2.0 * G / 3.0
    C = np.array([
        [lam + 2.0 * G, lam, lam, 0.0, 0.0, 0.0],
        [lam, lam + 2.0 * G, lam, 0.0, 0.0, 0.0],
        [lam, lam, lam + 2.0 * G, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, G, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, G, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, G],
    ], dtype=float)

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    KeeT = Kb * np.outer(ee, ee)

    D = np.broadcast_to(C, (n, 6, 6)).copy()

    # 1. Melted state: when T >= Tmelt, shear strength is 0, leaving only bulk volumetric modulus K 1 (x) 1
    melted = np.zeros(n, dtype=bool)
    if extra is not None and "temp" in extra and extra["temp"] is not None:
        raw_t = extra["temp"]
        if np.isscalar(raw_t):
            T_arr = np.full(n, float(raw_t), dtype=float)
        else:
            T_arr = np.array(raw_t, dtype=float, copy=True)
            if T_arr.shape != (n,):
                T_arr = np.full(n, float(T_arr.flat[0]) if T_arr.size > 0 else T0, dtype=float)
        melted = T_arr >= Tmelt

    if np.any(melted):
        D[melted] = KeeT

    # 2. Broken state: when epsp > eps_max, shear strength is 0, leaving only bulk volumetric modulus K 1 (x) 1
    eps_max = float(p.get("eps_max", _INF))
    broken = np.zeros(n, dtype=bool)
    if epsp is not None and eps_max < _INF:
        broken = epsp > eps_max
        if np.any(broken):
            D[broken] = KeeT

    active_mask = (~melted) & (~broken)

    if epsp_incr is None:
        return D
    plastic = (epsp_incr > 0.0) & active_mask
    if not np.any(plastic):
        return D

    idx = np.where(plastic)[0]
    s = sig[idx].copy()
    pm = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    s[:, 0] -= pm
    s[:, 1] -= pm
    s[:, 2] -= pm

    snorm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
                    + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    snorm = np.maximum(snorm, 1e-30)
    Nv = s / snorm[:, None]
    q = np.sqrt(1.5) * snorm
    dep = epsp_incr[idx]
    q_tr = q + 3.0 * G * dep

    epsp_cur = epsp[idx]
    H = np.zeros(len(idx), dtype=float)
    if N >= 1.0:
        H = B * N * np.maximum(epsp_cur, 0.0) ** (N - 1.0)
    else:
        pos = epsp_cur > 0.0
        H[pos] = B * N / np.maximum(epsp_cur[pos], 1e-20) ** (1.0 - N)

    # Thermal and strain rate scaling of hardening modulus (m4law.F line 160)
    ct = np.ones(len(idx), dtype=float)
    if extra is not None and "temp" in extra and extra["temp"] is not None:
        raw_t = extra["temp"]
        if np.isscalar(raw_t):
            T_idx = np.full(len(idx), float(raw_t), dtype=float)
        else:
            t_arr = np.array(raw_t, dtype=float, copy=True)
            T_idx = t_arr[idx] if t_arr.shape == (n,) else np.full(len(idx), float(t_arr.flat[0]), dtype=float)
        above_t0 = (T_idx > T0) & (T_idx < Tmelt)
        if np.any(above_t0):
            T_sub = T_idx[above_t0]
            m_eff = np.where(T_sub > Tmax, 1.0, m)
            denom = max(Tmelt - T0, 1e-20)
            tstar = np.clip((T_sub - T0) / denom, 0.0, 1.0)
            ct[above_t0] = np.maximum(1.0 - (tstar ** m_eff), 0.0)

    ce = np.ones(len(idx), dtype=float)
    if extra is not None and "ce" in extra and extra["ce"] is not None:
        raw_ce = extra["ce"]
        if np.isscalar(raw_ce):
            ce = np.full(len(idx), float(raw_ce), dtype=float)
        else:
            ce_arr = np.array(raw_ce, dtype=float, copy=True)
            ce = ce_arr[idx] if ce_arr.shape == (n,) else np.full(len(idx), float(ce_arr.flat[0]), dtype=float)
    elif extra is not None and "epd" in extra and extra["epd"] is not None:
        raw_epd = extra["epd"]
        if np.isscalar(raw_epd):
            epd_idx = np.full(len(idx), float(raw_epd), dtype=float)
        else:
            epd_arr = np.array(raw_epd, dtype=float, copy=True)
            epd_idx = epd_arr[idx] if epd_arr.shape == (n,) else np.full(len(idx), float(epd_arr.flat[0]), dtype=float)
        if c_rate > 0.0 and eps0 > 0.0:
            ratio = np.maximum(epd_idx / eps0, 1.0)
            ce = 1.0 + c_rate * np.log(ratio)

    H = H * ce * ct

    ch = (p.get("A", 0.0) + B * np.maximum(epsp_cur, 0.0) ** N) * ce * ct
    capped = (ch >= sig_max) | (epsp_cur > eps_max)
    H[capped] = 0.0

    a = 3.0 * G * dep / q_tr
    b = 6.0 * G * G * (dep / q_tr - 1.0 / (3.0 * G + np.maximum(H, 0.0)))

    C_minus_vol = C - KeeT
    NN = np.einsum("mi,mj->mij", Nv, Nv)
    D[idx] = (C[None, :, :]
              - a[:, None, None] * C_minus_vol[None, :, :]
              + b[:, None, None] * NN)
    return D


def sound_speed(
    mat: Material | dict,
    rho: float | np.ndarray | None = None,
    extra: dict | None = None,
) -> float | np.ndarray:
    """Acoustic longitudinal wave speed for LAW4 (hydrodynamic Johnson-Cook).

    Cites m4law.F line 100:
    c = sqrt((K_eff + 4/3 * G) / rho0)
    """
    p = _ensure_params(mat)
    g = float(p["G"])
    k = float(p["K"])
    rho0 = float(getattr(mat, "rho0", 1.0) or (p.get("rho0", 1.0) if isinstance(p, dict) else 1.0) or 1.0)
    current_rho = rho0 if rho is None else rho

    if getattr(mat, "eos", None) is not None:
        from pyradioss.materials import eos as eos_mod
        mu_val = 0.0
        e_val = float(getattr(mat.eos, "params", {}).get("e0", 0.0))
        if extra is not None:
            if "mu" in extra and extra["mu"] is not None:
                mu_val = float(np.mean(extra["mu"]))
            elif "rho" in extra and extra["rho"] is not None:
                mu_val = float(np.mean(extra["rho"])) / rho0 - 1.0
            if "e_eos" in extra and extra["e_eos"] is not None:
                e_val = float(np.mean(extra["e_eos"]))
        try:
            c_eos = float(eos_mod.sound_speed(mat.eos, mu_val, e_val))
            if c_eos > 0.0:
                k = rho0 * (c_eos ** 2)
        except Exception:
            pass

    c_sq = (k + (4.0 / 3.0) * g) / current_rho
    c = np.sqrt(np.maximum(0.0, c_sq))
    if isinstance(c, np.ndarray) and c.ndim == 0:
        return float(c)
    return c


solid_tangent = consistent_solid_tangent
tangent = consistent_solid_tangent


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW4 uses an incremental hypoelastic rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Persistent history variables for LAW4."""
    return {
        "temp": () if nip is None else (nip,),
        "e_eos": () if nip is None else (nip,),
        "p_eos": () if nip is None else (nip,),
    }


def _register():
    """Register LAW4 constructors in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        MAT_PHYSICS_REGISTRY["LAW4"] = build_law04
        MAT_PHYSICS_REGISTRY["HYD_JCOOK"] = build_law04
        MAT_PHYSICS_REGISTRY["JCOOK_HYD"] = build_law04
        MAT_PHYSICS_REGISTRY["4"] = build_law04
    except Exception:
        pass


_register()
