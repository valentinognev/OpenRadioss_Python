r"""LAW10 — Soil and crushable material with Drucker-Prager yield and compaction EOS (/MAT/LAW10, /MAT/SOIL, /MAT/DPRAG).

Fortran origins:
- ``$OR_SRC/engine/source/materials/mat/mat010/m10law.F``
  (Subroutine M10LAW, lines 28-220; historically cited as sigeps10.F)
- ``$OR_SRC/common_source/eos/compaction.F90``
  (Compaction equation of state, lines 103-180)
- ``$OR_SRC/starter/source/materials/mat/mat010/hm_read_mat10.F``
  (Starter card reader, defaults & parameter estimation, lines 97-290)
- ``$OR_SRC/hm_cfg_files/config/CFG/radioss2020/MAT/matl10_law10.cfg``
  (CFG attributes & card format)

Theory
------
LAW10 models geological materials (soils, rocks, sand) and crushable concrete using:
1. A pressure-dependent Drucker-Prager yield surface:
   \(J_2 = A_0 + A_1 P_{tot} + A_2 P_{tot}^2\)
   capped by a von Mises limit \(A_{max}\), with tensile fracture pressure \(P_{min}\)
   and pressure axis root \(P^*\) closure.
2. A compaction equation of state for volumetric behavior:
   \(P(\mu) = c_0 + c_1 \mu + (c_2 + c_3 \mu) \mu |\mu|\)
   with maximum historic compaction memory \(\mu_{bak}\) and constant or continuous
   unloading bulk modulus \(B_{unl}\).
3. Shock bulk viscosity for shock smoothing (mqviscb / mmain.F90 lines 1329-1342):
   \(P_{tot} = P_{new} + P_{sh} + q_{vis}\)
4. Radial return projection of the deviatoric trial stress onto the yield envelope:
   \(r = \sqrt{G_0 / (J_2 + 10^{-14})}\)
   with Cauchy stress \(\sigma = s_{new} - (P_{new} + q_{vis}) I\).
5. Acoustic wave speed:
   \(c = \sqrt{(K_{eff} + \frac{4}{3} G) / \rho_0}\) where \(K_{eff} = \max(c_1, B_{unl})\).

Solids only (SOLID_ISOTROPIC + SPH).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyradioss.model.entities import Material

_EM14 = 1e-14
_EM20 = 1e-20
_EP20 = 1e20
_INF = 1e30


# -----------------------------------------------------------------------------
# Parameter extraction and validation (hm_read_mat10.F)
# -----------------------------------------------------------------------------

def _ensure_params(mat: Material | dict) -> dict[str, Any]:
    """Ensure material params contain both CFG and direct keys with robust defaults.

    Cites hm_read_mat10.F lines 97-290.
    """
    if hasattr(mat, "params") and mat.params is not None:
        p = mat.params
    elif isinstance(mat, dict):
        p = mat.get("params", mat)
    else:
        p = getattr(mat, "params", {})

    # Density rho0
    rho0_val = (
        getattr(mat, "rho0", None)
        if hasattr(mat, "rho0") and getattr(mat, "rho0") is not None
        else (
            p.get("rho0")
            if p.get("rho0") is not None
            else (
                p.get("density")
                if p.get("density") is not None
                else (p.get("rho") if p.get("rho") is not None else p.get("MAT_RHO"))
            )
        )
    )
    if rho0_val is None or float(rho0_val) <= 0.0:
        raise ValueError(f"LAW10: Density rho0 must be > 0 (got {rho0_val})")
    rho0 = float(rho0_val)

    # Young's modulus E
    e_val = p.get("E") if p.get("E") is not None else p.get("MAT_E")
    if e_val is None or float(e_val) <= 0.0:
        raise ValueError(f"LAW10: Young's modulus E must be > 0 (got {e_val})")
    e = float(e_val)

    # Poisson's ratio nu in [0, 0.5)
    nu_val = p.get("nu") if p.get("nu") is not None else p.get("MAT_NU")
    if nu_val is None:
        raise ValueError("LAW10: Poisson's ratio nu must be defined")
    nu = float(nu_val)
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"LAW10: Poisson's ratio nu must be in [0, 0.5) (got {nu})")

    # Elastic shear and bulk moduli
    g_val = p.get("G") if p.get("G") is not None else p.get("MAT_G")
    g = float(g_val) if g_val is not None and float(g_val) > 0.0 else e / (2.0 * (1.0 + nu))
    k_val = p.get("K") if p.get("K") is not None else p.get("MAT_K")
    k = float(k_val) if k_val is not None and float(k_val) > 0.0 else e / (3.0 * (1.0 - 2.0 * nu))

    # Drucker-Prager coefficients A0, A1, A2
    a0 = float(p.get("A0") if p.get("A0") is not None else (p.get("MAT_A0") or 0.0))
    a1 = float(p.get("A1") if p.get("A1") is not None else (p.get("MAT_A1") or 0.0))
    a2 = float(p.get("A2") if p.get("A2") is not None else (p.get("MAT_A2") or 0.0))

    # von Mises cap Amax (defaults to 1e20 if 0 or None, hm_read_mat10.F line 222)
    amax_val = (
        p.get("Amax")
        if p.get("Amax") is not None
        else (
            p.get("amax")
            if p.get("amax") is not None
            else (p.get("AMAX") if p.get("AMAX") is not None else p.get("MAT_AMAX"))
        )
    )
    amax = float(amax_val) if amax_val is not None and float(amax_val) != 0.0 else _EP20

    # Tension fracture pressure pmin (defaults to -1e30 if 0 or None, hm_read_mat10.F line 223, m10law.F line 98)
    pmin_val = (
        p.get("pmin")
        if p.get("pmin") is not None
        else (
            p.get("MAT_PC")
            if p.get("MAT_PC") is not None
            else (
                p.get("PMIN")
                if p.get("PMIN") is not None
                else (
                    p.get("p_min")
                    if p.get("p_min") is not None
                    else (
                        p.get("MAT_PMIN")
                        if p.get("MAT_PMIN") is not None
                        else (p.get("pfrac") if p.get("pfrac") is not None else p.get("PFRAC"))
                    )
                )
            )
        )
    )
    pmin = float(pmin_val) if pmin_val is not None and float(pmin_val) != 0.0 else -_INF

    # External pressure / shift
    pext = float(
        p.get("pext")
        if p.get("pext") is not None
        else (p.get("PEXT") if p.get("PEXT") is not None else (p.get("MAT_PEXT") or 0.0))
    )
    psh = float(
        p.get("psh")
        if p.get("psh") is not None
        else (p.get("MAT_PSH") if p.get("MAT_PSH") is not None else (p.get("PSH") or 0.0))
    )

    # Compaction EOS coefficients c0, c1, c2, c3
    c0 = float(
        p.get("c0")
        if p.get("c0") is not None
        else (p.get("EOS_COM_C0") if p.get("EOS_COM_C0") is not None else (p.get("MAT_C0") or 0.0))
    )

    c1_val = (
        p.get("c1")
        if p.get("c1") is not None
        else (p.get("EOS_COM_C1") if p.get("EOS_COM_C1") is not None else p.get("MAT_C1"))
    )
    c1 = float(c1_val) if c1_val is not None and float(c1_val) != 0.0 else k

    c2 = float(
        p.get("c2")
        if p.get("c2") is not None
        else (p.get("EOS_COM_C2") if p.get("EOS_COM_C2") is not None else (p.get("MAT_C2") or 0.0))
    )
    c3 = float(
        p.get("c3")
        if p.get("c3") is not None
        else (p.get("EOS_COM_C3") if p.get("EOS_COM_C3") is not None else (p.get("MAT_C3") or 0.0))
    )

    # Unloading bulk modulus bunl & maximum compaction mue_max
    bunl_val = (
        p.get("bunl")
        if p.get("bunl") is not None
        else (
            p.get("EOS_COM_B")
            if p.get("EOS_COM_B") is not None
            else (
                p.get("BUNL")
                if p.get("BUNL") is not None
                else (p.get("MAT_BULK") if p.get("MAT_BULK") is not None else p.get("MAT_BUNL"))
            )
        )
    )
    bunl = float(bunl_val) if bunl_val is not None else 0.0

    mue_max_val = (
        p.get("mue_max")
        if p.get("mue_max") is not None
        else (
            p.get("EOS_COM_Mue_max")
            if p.get("EOS_COM_Mue_max") is not None
            else (
                p.get("XMUMX")
                if p.get("XMUMX") is not None
                else (
                    p.get("mue_mx")
                    if p.get("mue_mx") is not None
                    else (
                        p.get("MAT_SIG")
                        if p.get("MAT_SIG") is not None
                        else (
                            p.get("MAT_XMUMX")
                            if p.get("MAT_XMUMX") is not None
                            else (p.get("mu_max") if p.get("mu_max") is not None else p.get("MU_MAX"))
                        )
                    )
                )
            )
        )
    )
    mue_max = float(mue_max_val) if mue_max_val is not None else 0.0

    # Auto-estimation per hm_read_mat10.F lines 229-260
    if mue_max == 0.0 and bunl != 0.0:
        if c3 == 0.0:
            if c2 == 0.0:
                mue_max = _EP20
            else:
                mue_max = (bunl - c1) / (2.0 * c2)
        else:
            det = math.sqrt(max(0.0, c2**2 + 3.0 * c3 * (bunl - c1)))
            mue_max = (det - c2) / (3.0 * c3)
    elif mue_max != 0.0 and bunl == 0.0:
        if c3 == 0.0:
            if c2 == 0.0:
                bunl = c1
            else:
                bunl = c1 + 2.0 * c2 * mue_max
        else:
            # hm_read_mat10.F line 256: BUNL = C1 + TWO*C2*XMUMX + THREE*C3*C3*XMUMX**TWO
            bunl = c1 + 2.0 * c2 * mue_max + 3.0 * (c3**2) * (mue_max**2)

    if bunl == 0.0:
        bunl = c1
    if mue_max == 0.0:
        mue_max = _EP20

    # Unload slope formulation flag iform (1: constant unload slope, 2: continuous)
    iform_val = p.get("iform") if p.get("iform") is not None else p.get("IFORM", 1)
    iform = int(iform_val) if iform_val is not None else 1
    if iform not in (1, 2):
        iform = 1

    # Pressure root pstar (hm_read_mat10.F lines 202-220)
    if a2 == 0.0 and a1 != 0.0:
        pstar = -a0 / a1
    elif a2 != 0.0:
        delta = a1**2 - 4.0 * a0 * a2
        if delta >= 0.0:
            pstar = (-a1 + math.sqrt(delta)) / (2.0 * a2)
        else:
            pstar = -a1 / (2.0 * a2)
    else:
        pstar = -_INF

    # Populate direct keys
    p["rho0"] = rho0
    p["E"] = e
    p["nu"] = nu
    p["G"] = g
    p["K"] = k
    p["A0"] = a0
    p["A1"] = a1
    p["A2"] = a2
    p["Amax"] = amax
    p["pmin"] = pmin
    p["pext"] = pext
    p["psh"] = psh
    p["c0"] = c0
    p["c1"] = c1
    p["c2"] = c2
    p["c3"] = c3
    p["bunl"] = bunl
    p["mue_max"] = mue_max
    p["iform"] = iform
    p["pstar"] = pstar

    # Populate CFG keys
    p["MAT_RHO"] = rho0
    p["MAT_E"] = e
    p["MAT_NU"] = nu
    p["MAT_A0"] = a0
    p["MAT_A1"] = a1
    p["MAT_A2"] = a2
    p["MAT_AMAX"] = amax
    p["MAT_PC"] = pmin
    p["PEXT"] = pext
    p["MAT_PSH"] = psh
    p["EOS_COM_C0"] = c0
    p["EOS_COM_C1"] = c1
    p["EOS_COM_C2"] = c2
    p["EOS_COM_C3"] = c3
    p["EOS_COM_B"] = bunl
    p["EOS_COM_Mue_max"] = mue_max
    p["MAT_BULK"] = bunl
    p["MAT_BUNL"] = bunl
    p["MAT_SIG"] = mue_max
    p["MAT_XMUMX"] = mue_max
    p["MAT_PMIN"] = pmin
    p["PFRAC"] = pmin
    p["MAT_G"] = g
    p["MAT_K"] = k

    return p


def build_law10(rec: Any) -> Material:
    """Card parsing and validation -> Material for LAW10 (/MAT/LAW10, /MAT/SOIL, /MAT/DPRAG).

    Cites starter/source/materials/mat/mat010/hm_read_mat10.F.
    """
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params is not None else {}
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = rec.get("id", 1)
        title = rec.get("title", "LAW10")
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
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "LAW10")
        density = float(getattr(rec, "rho0", getattr(rec, "density", 0.0)))

    params = dict(p)
    if density > 0.0:
        params["rho0"] = density
        params["MAT_RHO"] = density

    # Populate and validate
    _ensure_params(params)
    density = params["rho0"]

    mat = Material(id=mat_id, law=10, rho0=density, title=title, params=params)
    return mat


# -----------------------------------------------------------------------------
# Sound speed calculation
# -----------------------------------------------------------------------------

def sound_speed(
    mat: Material | dict,
    rho: float | np.ndarray | None = None,
    extra: dict | None = None,
) -> float | np.ndarray:
    """Longitudinal acoustic wave speed for LAW10 solids.

    Cites m10law.F lines 152-155, compaction.F90 lines 152-155,
    and matl10_law10.cfg DRAWABLES:
    c = sqrt((K_eff + 4/3 * G) / rho0) where K_eff = max(c1, bunl).
    """
    p = _ensure_params(mat)
    g = float(p["G"])
    c1 = float(p["c1"])
    c2 = float(p["c2"])
    c3 = float(p["c3"])
    bunl = float(p["bunl"])
    iform = int(p["iform"])
    mue_max = float(p["mue_max"])
    rho0 = float(p["rho0"])

    # Dynamic DPDM per compaction.F90 lines 152-156 & m10law.F lines 152-155
    if extra is not None and "mu" in extra and extra["mu"] is not None:
        mu_val = np.asarray(extra["mu"], dtype=float)
        mu_pos = np.maximum(0.0, mu_val)
        dpdm = c1 + mu_pos * (2.0 * c2 + 3.0 * c3 * mu_val)
        if iform == 1:
            b_eff = bunl
        else:
            mu_bak = np.asarray(extra.get("mu_bak", mu_val), dtype=float)
            alpha = np.where(mue_max > 0.0, mu_bak / mue_max, 1.0)
            b_eff = alpha * bunl + (1.0 - alpha) * c1
        k_eff = np.maximum(b_eff, dpdm)
    else:
        k_eff = max(c1, bunl)

    g43 = (4.0 / 3.0) * g

    if rho is None and extra is not None and "rho" in extra and extra["rho"] is not None:
        rho = extra["rho"]

    current_rho = rho0 if rho is None else rho
    c_sq = (k_eff + g43) / current_rho
    c = np.sqrt(np.maximum(0.0, c_sq))
    if isinstance(c, np.ndarray) and c.ndim == 0:
        return float(c)
    return c


# -----------------------------------------------------------------------------
# Constitutive stress update (solid only)
# -----------------------------------------------------------------------------

def solid_update(
    mat: Material | dict,
    sig: np.ndarray,
    eps_dot: np.ndarray | None = None,
    dt: float = 0.0,
    *args,
    d_eps: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    extra: dict | None = None,
    return_tuple: bool = False,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None, np.ndarray | float | None]:
    """Vectorized 3D solid stress update for LAW10 (soil/concrete Drucker-Prager + Compaction EOS).

    Ports:
    - ``engine/source/materials/mat/mat010/m10law.F``
    - ``common_source/eos/compaction.F90``

    Parameters
    ----------
    mat : Material or dict
        Material entity or parameters dict.
    sig : (n, 6) ndarray
        Old (Jaumann-rotated) Cauchy stress [xx, yy, zz, xy, yz, zx].
    eps_dot : (n, 6) ndarray, optional
        Strain rate tensor or strain increment if called positionally.
    dt : float, optional
        Time step increment.
    *args :
        Additional positional arguments (e.g. `epsp`, `dt`, `extra`).
    d_eps, deps : (n, 6) ndarray, optional
        Explicit strain increment tensor (engineering shear).
    epsp : (n,) ndarray, optional
        Equivalent plastic strain history.
    extra : dict, optional
        Extra state views (e.g. 'mu_bak', 'epxe', 'p_old', 'rho', 'off').
    return_tuple : bool, default False
        If True, returns (sig_new, epsp, sound_speed) for kernel dispatch.
        Otherwise returns sig_new.

    Returns
    -------
    sig_new : (n, 6) ndarray (or tuple if return_tuple=True)
        Updated Cauchy stress tensor [xx, yy, zz, xy, yz, zx].
    """
    # 0. Handle argument permutations
    # Element kernels may call: solid_update(mat, sig, deps, epsp, dt, extra)
    # Direct tests may call: solid_update(mat, sig, eps_dot, dt, extra)
    if len(args) >= 1:
        if isinstance(dt, (int, float)):
            # dt is really dt, args[0] might be extra
            if extra is None and isinstance(args[0], dict):
                extra = args[0]
        else:
            # Positional call: (mat, sig, deps, epsp, dt, extra)
            actual_deps = eps_dot
            epsp = dt
            dt = float(args[0])
            if len(args) >= 2 and isinstance(args[1], dict):
                extra = args[1]
            if deps is None and d_eps is None:
                deps = actual_deps

    if extra is None:
        extra = {}

    single = (sig.ndim == 1)
    if single:
        sig = sig[None, :]

    n = sig.shape[0]
    if n == 0:
        sig_ret = sig.copy()
        if return_tuple:
            return sig_ret, epsp, None
        return sig_ret

    if dt <= 0.0:
        sig_ret = sig.copy()
        if single:
            sig_ret = sig_ret[0]
        if return_tuple:
            c_val = sound_speed(mat, extra=extra)
            return sig_ret, epsp, c_val
        return sig_ret

    # Extract strain increment d_eps
    if d_eps is not None:
        d_e = np.asarray(d_eps, dtype=float)
    elif deps is not None:
        d_e = np.asarray(deps, dtype=float)
    elif eps_dot is not None:
        d_e = np.asarray(eps_dot, dtype=float) * dt
    else:
        d_e = np.zeros_like(sig, dtype=float)

    if single and d_e.ndim == 1:
        d_e = d_e[None, :]

    # Material parameters
    p = _ensure_params(mat)
    g = float(p["G"])
    a0 = float(p["A0"])
    a1 = float(p["A1"])
    a2 = float(p["A2"])
    amax = float(p["Amax"])
    pmin = float(p["pmin"])
    psh = float(p["psh"])
    c0 = float(p["c0"])
    c1 = float(p["c1"])
    c2 = float(p["c2"])
    c3 = float(p["c3"])
    bunl = float(p["bunl"])
    mue_max = float(p["mue_max"])
    iform = int(p["iform"])
    pstar = float(p["pstar"])
    rho0 = float(p["rho0"])

    # Extra state initialization
    if "mu_bak" not in extra or extra["mu_bak"] is None:
        extra["mu_bak"] = np.zeros(n, dtype=float)
    if "epxe" not in extra or extra["epxe"] is None:
        extra["epxe"] = np.zeros(n, dtype=float)
    if "p_old" not in extra or extra["p_old"] is None:
        extra["p_old"] = np.zeros(n, dtype=float)

    mu_bak = np.array(extra["mu_bak"], dtype=float, copy=True)
    if mu_bak.shape != (n,):
        mu_bak = np.full(n, float(mu_bak.flat[0]) if mu_bak.size > 0 else 0.0, dtype=float)

    off = np.asarray(extra.get("off", np.ones(n, dtype=float)), dtype=float)
    if off.shape != (n,):
        off = np.full(n, float(off.flat[0]) if off.size > 0 else 1.0, dtype=float)

    # 1. Old pressure & deviatoric trial stress (m10law.F lines 132, 141-147, 161-167)
    p_old = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    tr_deps = d_e[:, 0] + d_e[:, 1] + d_e[:, 2]

    s_tr = np.empty_like(sig, dtype=float)
    s_tr[:, 0] = sig[:, 0] + p_old + 2.0 * g * (d_e[:, 0] - tr_deps / 3.0)
    s_tr[:, 1] = sig[:, 1] + p_old + 2.0 * g * (d_e[:, 1] - tr_deps / 3.0)
    s_tr[:, 2] = sig[:, 2] + p_old + 2.0 * g * (d_e[:, 2] - tr_deps / 3.0)
    s_tr[:, 3] = sig[:, 3] + g * d_e[:, 3]
    s_tr[:, 4] = sig[:, 4] + g * d_e[:, 4]
    s_tr[:, 5] = sig[:, 5] + g * d_e[:, 5]

    # 2. Volumetric strain / compaction mu (compaction.F90 lines 124-148)
    if "rho" in extra and extra["rho"] is not None:
        rho_arr = np.asarray(extra["rho"], dtype=float)
        mu = rho_arr / rho0 - 1.0
    elif "mu_total" in extra and extra["mu_total"] is not None:
        mu = np.asarray(extra["mu_total"], dtype=float)
    else:
        mu_prev = np.asarray(extra.get("mu", 0.0), dtype=float)
        if mu_prev.shape != (n,):
            mu_prev = np.full(n, float(mu_prev.flat[0]) if mu_prev.size > 0 else 0.0, dtype=float)
        mu = mu_prev - tr_deps

    if mu.shape != (n,):
        mu = np.full(n, float(mu.flat[0]) if mu.size > 0 else 0.0, dtype=float)
    extra["mu"] = mu
    extra["mu_old"] = mu

    # 3. Compaction EOS pressure (compaction.F90 lines 125-148, 161-163)
    mu2 = mu * np.maximum(0.0, mu)
    p_loading = c0 + c1 * mu + (c2 + c3 * mu) * mu2
    p_bak = c0 + c1 * mu_bak + (c2 + c3 * mu_bak) * (mu_bak**2)

    if iform == 1:
        b_eff = np.full(n, bunl, dtype=float)
    else:
        alpha = np.where(mue_max > 0.0, mu_bak / mue_max, 1.0)
        b_eff = alpha * bunl + (1.0 - alpha) * c1

    p_unl = p_bak - (mu_bak - mu) * b_eff
    p_eos = np.where(mu_bak > 0.0, np.minimum(p_loading, p_unl), p_loading)
    p_eos = np.maximum(p_eos, pmin) * off
    p_new = p_eos - psh

    # Shock / bulk viscosity pressure (mqviscb / mmain.F90 lines 1329-1342)
    q_vis = np.asarray(
        extra.get("qvis", extra.get("q_bulk", extra.get("q", 0.0))),
        dtype=float,
    )
    if q_vis.shape != (n,):
        q_vis = np.full(n, float(q_vis.flat[0]) if q_vis.size > 0 else 0.0, dtype=float)

    p_tot = p_new + psh + q_vis

    # Historic compaction update (compaction.F90 line 177)
    mu_bak = np.minimum(mue_max, np.maximum(mu_bak, mu))

    # 4. Drucker-Prager Yield Envelope & Projection Factor (m10law.F lines 173-193)
    j2 = (
        0.5 * (s_tr[:, 0] ** 2 + s_tr[:, 1] ** 2 + s_tr[:, 2] ** 2)
        + s_tr[:, 3] ** 2
        + s_tr[:, 4] ** 2
        + s_tr[:, 5] ** 2
    )
    g0 = a0 + a1 * p_tot + a2 * (p_tot**2)
    g0 = np.clip(g0, 0.0, amax)
    g0 = np.where(p_new <= pmin, 0.0, g0)
    g0 = np.where(p_tot <= pstar, 0.0, g0)

    yield2 = j2 - g0
    ratio = np.where((yield2 <= 0.0) & (g0 > 0.0), 1.0, np.sqrt(g0 / (j2 + _EM14)))
    ratio = np.where(g0 <= 0.0, 0.0, ratio)

    # 5. Deviatoric stress update (m10law.F lines 199-204)
    s_new = ratio[:, None] * s_tr * off[:, None]

    # 6. Total Cauchy stress recombination (sigma = s - (p_new + q_vis) * I)
    p_eff = p_new + q_vis
    sig_new = np.empty_like(sig, dtype=float)
    sig_new[:, 0] = s_new[:, 0] - p_eff
    sig_new[:, 1] = s_new[:, 1] - p_eff
    sig_new[:, 2] = s_new[:, 2] - p_eff
    sig_new[:, 3] = s_new[:, 3]
    sig_new[:, 4] = s_new[:, 4]
    sig_new[:, 5] = s_new[:, 5]

    # 7. Plastic strain increment & state update (m10law.F lines 205, 212-216)
    # Fortran: DPLA = (1 - RATIO) * SQRT(3 * J2) / (3 * G)
    denom = max(_EM20, 3.0 * g)
    dpla = (1.0 - ratio) * np.sqrt(3.0 * np.maximum(0.0, j2)) / denom

    epxe_cur = np.asarray(extra["epxe"], dtype=float)
    if epxe_cur.shape != (n,):
        epxe_cur = np.full(n, float(epxe_cur.flat[0]) if epxe_cur.size > 0 else 0.0, dtype=float)
    extra["epxe"] = epxe_cur + dpla
    extra["epsq"] = mu_bak  # m10law.F line 215: EPSQ(I) = MU_BAK(I)
    extra["mu_bak"] = mu_bak
    extra["p_old"] = p_new
    extra["sigy"] = np.sqrt(3.0 * g0)
    extra["g0"] = g0
    extra["dpla"] = dpla
    extra["ratio"] = ratio
    extra["j2"] = j2
    extra["p_new"] = p_new
    extra["ptot"] = p_tot
    extra["q_vis"] = q_vis

    if epsp is not None:
        epsp[:] = extra["epxe"]

    if single:
        if return_tuple:
            c_speed = sound_speed(mat, extra=extra)
            return sig_new[0], extra["epxe"], c_speed
        return sig_new[0]

    if return_tuple:
        c_speed = sound_speed(mat, extra=extra)
        return sig_new, extra["epxe"], c_speed

    return sig_new


def shell_update(mat: Any, sig: Any, *args: Any, **kwargs: Any) -> Any:
    """Plane-stress shell update is not supported for LAW10."""
    raise NotImplementedError("LAW10 (soil/Drucker-Prager) is implemented for 3D solid elements only.")


# -----------------------------------------------------------------------------
# Consistent tangent for implicit analysis
# -----------------------------------------------------------------------------

def consistent_solid_tangent(
    mat: Material | dict,
    sig: np.ndarray,
    eps_dot: np.ndarray | None = None,
    dt: float = 0.0,
    *args,
    epsp: np.ndarray | None = None,
    epsp_incr: np.ndarray | None = None,
    extra: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Algorithmic consistent elastoplastic tangent matrix for solids in Voigt notation.

    Voigt convention: [xx, yy, zz, xy, yz, zx] with engineering shear.

    Parameters
    ----------
    mat : Material or dict
        Material definition.
    sig : (n, 6) ndarray
        Current stress state.
    eps_dot, dt :
        Optional rate and time step.
    epsp, epsp_incr :
        Optional plastic strain history / increments.
    extra : dict, optional
        Extra state views (e.g. 'ratio', 'j2', 'p_new', 'p_old', 'mu', 'mu_bak').
    symmetric : bool, optional (via kwargs)
        If True, returns symmetrized tangent matrix 0.5 * (D + D^T).
        Default is False (exact algorithmic consistent tangent).

    Returns
    -------
    D : (n, 6, 6) ndarray
        Consistent tangent stiffness matrix.
    """
    if extra is None:
        for a in args:
            if isinstance(a, dict):
                extra = a
                break

    single = (sig.ndim == 1)
    if single:
        sig = sig[None, :]

    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=sig.dtype)

    p = _ensure_params(mat)
    g = float(p["G"])
    c0 = float(p["c0"])
    c1 = float(p["c1"])
    c2 = float(p["c2"])
    c3 = float(p["c3"])
    bunl = float(p["bunl"])
    k_eff = max(c1, bunl)
    a0 = float(p["A0"])
    a1 = float(p["A1"])
    a2 = float(p["A2"])
    amax = float(p["Amax"])
    pmin = float(p["pmin"])
    psh = float(p["psh"])
    pstar = float(p["pstar"])
    iform = int(p["iform"])
    mue_max = float(p["mue_max"])

    # Base Voigt projector tensors
    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    c_dev = np.zeros((6, 6), dtype=float)
    c_dev[0, 0] = c_dev[1, 1] = c_dev[2, 2] = (4.0 / 3.0) * g
    c_dev[0, 1] = c_dev[0, 2] = c_dev[1, 0] = c_dev[1, 2] = c_dev[2, 0] = c_dev[2, 1] = -(2.0 / 3.0) * g
    c_dev[3, 3] = c_dev[4, 4] = c_dev[5, 5] = g

    d_tangent = np.zeros((n, 6, 6), dtype=float)

    # Current stress state decomposition
    p_cur = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    p_tot = p_cur + psh
    s_cur = sig.copy()
    s_cur[:, 0] += p_cur
    s_cur[:, 1] += p_cur
    s_cur[:, 2] += p_cur

    mu = extra.get("mu") if extra is not None else None
    mu_bak = extra.get("mu_bak") if extra is not None else None

    # Element loop
    for i in range(n):
        # 1. Tangent bulk modulus K_t = dP/dmu from Compaction EOS
        if mu is not None:
            mu_arr = np.asarray(mu, dtype=float)
            mu_i = float(mu_arr.flat[i if i < mu_arr.size else 0])
            if mu_bak is not None:
                mu_bak_arr = np.asarray(mu_bak, dtype=float)
                mu_bak_i = float(mu_bak_arr.flat[i if i < mu_bak_arr.size else 0])
            else:
                mu_bak_i = 0.0

            if iform == 1:
                b_eff = bunl
            else:
                alpha = (mu_bak_i / mue_max) if mue_max > 0.0 else 1.0
                b_eff = alpha * bunl + (1.0 - alpha) * c1

            mu2 = mu_i * max(0.0, mu_i)
            p_loading = c0 + c1 * mu_i + (c2 + c3 * mu_i) * mu2
            p_bak = c0 + c1 * mu_bak_i + (c2 + c3 * mu_bak_i) * (mu_bak_i**2)
            p_unl = p_bak - (mu_bak_i - mu_i) * b_eff

            if mu_bak_i > 0.0 and p_unl < p_loading:
                k_t = b_eff
            elif mu_i > 0.0:
                k_t = c1 + 2.0 * c2 * mu_i + 3.0 * c3 * (mu_i**2)
            else:
                k_t = c1
        else:
            k_t = k_eff

        keet = k_t * np.outer(ee, ee)
        c_elastic = keet + c_dev

        # 2. Yield surface & ratio evaluation
        s_i = s_cur[i]
        j2_i = (
            0.5 * (s_i[0] ** 2 + s_i[1] ** 2 + s_i[2] ** 2)
            + s_i[3] ** 2
            + s_i[4] ** 2
            + s_i[5] ** 2
        )
        p_i = p_cur[i]
        ptot_i = p_tot[i]

        g0_uncapped = a0 + a1 * ptot_i + a2 * (ptot_i**2)
        g0_val = min(max(0.0, g0_uncapped), amax)
        if p_i <= pmin or ptot_i <= pstar:
            g0_val = 0.0

        if extra is not None and "ratio" in extra:
            ratio_arr = np.asarray(extra["ratio"], dtype=float)
            r_i = float(ratio_arr.flat[i if i < ratio_arr.size else 0])
        else:
            if j2_i <= g0_val and g0_val > 0.0:
                r_i = 1.0
            elif g0_val <= 0.0:
                r_i = 0.0
            else:
                r_i = math.sqrt(g0_val / (j2_i + _EM14))

        # 3. Tangent stiffness matrix partition
        if r_i >= 1.0:
            # Fully elastic regime: isotropic Hookean tensor KeeT + C_dev
            d_tangent[i] = c_elastic
        elif r_i <= 0.0:
            # Collapsed yield envelope (G0=0 or P <= pmin or P_tot <= pstar):
            # Deviatoric tangent drops to zero, only KeeT remains
            d_tangent[i] = keet
        else:
            # Elastoplastic radial return tangent:
            # d_sigma = keet + r * C_dev - r * G * (s_hat (x) s_hat) + D_coupling
            norm_s = math.sqrt(max(j2_i, _EM20))
            s_hat = s_i / norm_s
            nn = np.outer(s_hat, s_hat)

            # Non-associated Drucker-Prager pressure-coupling term:
            # Yield function F = J2 - G0(P_tot) = 0 depends on pressure P_tot,
            # while plastic flow is purely deviatoric (radial return in deviatoric plane).
            # dG0/dP_tot = A1 + 2*A2*P_tot when 0 < G0 < Amax (0 at von Mises cap cutoff).
            if g0_uncapped >= amax or g0_uncapped <= 0.0:
                dg0_dptot = 0.0
            else:
                dg0_dptot = a1 + 2.0 * a2 * ptot_i

            d_coupling = - (k_t * dg0_dptot / (2.0 * max(g0_val, _EM14))) * np.outer(s_i, ee)
            d_tangent[i] = keet + r_i * c_dev - g * r_i * nn + d_coupling

        if kwargs.get("symmetric", False):
            d_tangent[i] = 0.5 * (d_tangent[i] + d_tangent[i].T)

    if single:
        return d_tangent[0]
    return d_tangent


solid_tangent = consistent_solid_tangent
tangent = consistent_solid_tangent


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW10 uses an incremental hypoelastic rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Persistent history variables for LAW10 (m10law.F lines 76-77; compaction.F90 line 176)."""
    return {
        "mu_bak": () if nip is None else (nip,),
        "epxe": () if nip is None else (nip,),
        "mu": () if nip is None else (nip,),
    }


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

def _register() -> None:
    """Register LAW10 constructors in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

        MAT_PHYSICS_REGISTRY["LAW10"] = build_law10
        MAT_PHYSICS_REGISTRY["SOIL"] = build_law10
        MAT_PHYSICS_REGISTRY["DPRAG"] = build_law10
        MAT_PHYSICS_REGISTRY["DPRAG1"] = build_law10
        MAT_PHYSICS_REGISTRY["10"] = build_law10
        MAT_PHYSICS_REGISTRY[10] = build_law10
    except Exception:
        pass


_register()
