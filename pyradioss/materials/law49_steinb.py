r"""LAW49 — Steinberg-Guinan high-strain-rate / shock plasticity model (/MAT/LAW49, /MAT/STEINB).

Fortran origins:
- ``$OR_SRC/engine/source/materials/mat/mat049/sigeps49.F``
  (engine constitutive update, implemented as ``m49law.F``)
- ``$OR_SRC/starter/source/materials/mat/mat049/hm_read_mat49.F``
  (starter card reader, defaults & parameter estimation)

Theory & Constitutive Formulation:
----------------------------------
LAW49 models metallic materials under high pressure, high strain rate, and shock deformation
where the shear modulus and yield strength exhibit strong pressure hardening and thermal
softening, with work hardening that saturates with plastic strain (Steinberg, Cochran & Guinan 1980):

- Shear Modulus:
  G(P, T) = G0 * (1 + A * P / V^(1/3) - B * (T - 300)) * Q_C
  where V = rho0 / rho = 1 / eta, A = b1 = G'_P / G0, B = h = -G'_T / G0.

- Yield Strength:
  Y(eps_p, P, T) = Y0 * (1 + beta * eps_p)^n * (G(P, T) / G0) * F_rate
  bounded by sigma_max and melting cutoff (Y = 0 for T >= Tmelt).

- Johnson-Cook Rate Dependence (at high strain rates):
  F_rate = 1 + C * ln(max(eps_dot, eps_dot0) / eps_dot0)  (when C > 0).

1. Pressure & Temperature Scaling:
   \(P = -\frac{1}{3} \text{tr}(\boldsymbol{\sigma})\) (compression positive convention)
   \(D_{av} = -\frac{1}{3} \text{tr}(\Delta\boldsymbol{\varepsilon})\)
   \(\eta = \rho / \rho_0\) (density compression ratio)
   \(Q_A = P \cdot \eta^{1/3}\)
   \(Q_B = 1 - h \cdot (\theta - T_0)\)
   \(E_{\text{melt}} = \rho C_p \cdot T_{\text{melt}}\)

2. Melt Softening Factor \(Q_C\):
   If \(E_{\text{melt}} \le 0\) or \(f \le 0\): \(Q_C = 1\)
   Else if \(E_{\text{spe}} \ge E_{\text{melt}}\): \(Q_C = 0\)
   Else:
       \(Q_C = \exp\left( \frac{f \cdot E_{\text{spe}}}{E_{\text{spe}} - E_{\text{melt}}} \right)\)

3. Pressure- and Temperature-dependent Shear Modulus:
   \(G = G_0 \left( b_1 Q_A + Q_B \right) Q_C\)
   If \(\theta \ge T_{\text{melt}}\): \(G = 0\) (liquid phase)

4. Yield Stress with Cold-Work Hardening:
   \(Q_D = \left( b_2 Q_A + Q_B \right) Q_C\)
   Cold work hardening function \(Q_E\):
   - If \(\varepsilon_p \le 0\): \(Q_E = \sigma_0\)
   - If \(\varepsilon_p > \varepsilon_{p,\max}\): \(Q_E = \sigma_0 (1 + \beta \varepsilon_{p,\max})^n\)
   - Else: \(Q_E = \sigma_0 (1 + \beta \varepsilon_p)^n\)
   Yield strength:
   \(Y = \min(\sigma_{\max}, Q_E) \cdot Q_D\)
   If \(\theta \ge T_{\text{melt}}\): \(Y = 0\)

5. Deviatoric Elastic Trial & Radial Return:
   \(s_{ij}^{\text{trial}} = s_{ij}^{\text{old}} + 2 G \cdot \text{off} \cdot (\Delta\varepsilon_{ij} + D_{av}\delta_{ij})\)  (normal components)
   \(s_{ij}^{\text{trial}} = s_{ij}^{\text{old}} + G \cdot \text{off} \cdot \Delta\varepsilon_{ij}\)  (shear components)
   \(AJ_2 = \sqrt{3 J_2(s^{\text{trial}})}\)
   Hardening modulus:
   \(Q_H = Q_D \cdot \sigma_0 \beta n (1 + \beta \varepsilon_p)^{n-1}\)
   \(\text{SCALE} = 1\) if \(AJ_2 \le Y\), else \(Y / AJ_2\)
   \(\Delta\varepsilon_p = \frac{(1 - \text{SCALE}) AJ_2}{\max(3G + Q_H, 10^{-15})}\)
   \(Y \leftarrow Y + \Delta\varepsilon_p \cdot Q_H\)
   \(s_{ij} = \text{SCALE} \cdot s_{ij}^{\text{trial}}\)
   \(\varepsilon_p \leftarrow (\varepsilon_p + \Delta\varepsilon_p) \cdot \text{off}\)

6. Adiabatic Temperature Rise:
   If \(\rho C_p > 0\):
       \(\Delta\theta = \frac{Y \cdot \Delta\varepsilon_p}{\rho C_p}\)
       \(\theta \leftarrow \theta + \Delta\theta\)

7. Volumetric Pressure Update:
   Hypoelastic / EOS update:
   \(P_{\text{new}} = P_{\text{old}} + 3 K D_{av}\), capped at \(P_{\text{new}} \ge P_{\min}\).
   \(\sigma_{ij} = s_{ij} - P_{\text{new}} \delta_{ij}\).

8. Longitudinal Acoustic Wave Speed:
   \(c = \sqrt{\frac{|K + \frac{4}{3} G|}{\rho_0}}\).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EM15 = 1e-15
_EM20 = 1e-20
_EP20 = 1e20


@dataclass
class Law49Params:
    """Parameters for /MAT/LAW49 (Steinberg-Guinan shock plasticity)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e0: float = 0.0
    nu: float = 0.0
    g0: float = 0.0
    bulk: float = 0.0
    sig0: float = 0.0
    beta: float = 0.0
    n: float = 0.0
    eps_max: float = _EP20
    sigma_max: float = _EP20
    t0: float = 300.0
    tmelt: float = _EP20
    rhoc_p: float = 0.0
    pmin: float = -_EP20
    b1: float = 0.0
    b2: float = 0.0
    h: float = 0.0
    f: float = 0.0
    c_rate: float = 0.0
    eps0: float = 1.0
    title: str = ""

    @property
    def G0(self) -> float:
        return self.g0

    @property
    def bulk_k(self) -> float:
        return self.bulk

    @property
    def E(self) -> float:
        return self.e0

    @property
    def G(self) -> float:
        return self.g0

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def sigy(self) -> float:
        return self.sig0

    @property
    def y0(self) -> float:
        return self.sig0

    @property
    def Y0(self) -> float:
        return self.sig0

    @property
    def A(self) -> float:
        return self.b1

    @property
    def B(self) -> float:
        return self.h


def _get_params(mat: Any) -> Law49Params:
    """Extract Law49Params from Material, Law49Params, dict, or MatLaw49."""
    if isinstance(mat, Law49Params):
        return mat

    if hasattr(mat, "law49_params") and isinstance(mat.law49_params, Law49Params):
        return mat.law49_params

    p: dict[str, Any] = {}
    rho0_val = None
    title = ""

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho0_val = getattr(mat, "rho0", None)
        title = getattr(mat, "title", "")
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        rho0_val = mat.get("rho0", mat.get("rho", mat.get("MAT_RHO")))
        title = mat.get("title", "")
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
        rho0_val = getattr(mat, "rho0", getattr(mat, "rho", None))
        title = getattr(mat, "title", "")
    elif hasattr(mat, "__dict__"):
        p = dict(mat.__dict__)
        rho0_val = getattr(mat, "rho0", getattr(mat, "rho", None))
        title = getattr(mat, "title", "")

    # Density rho0
    if rho0_val is None or float(rho0_val) <= 0.0:
        for k in ("rho0", "rho", "MAT_RHO", "density", "Refer_Rho"):
            if k in p and p[k] is not None and float(p[k]) > 0.0:
                rho0_val = p[k]
                break
    rho0 = float(rho0_val) if rho0_val is not None and float(rho0_val) > 0.0 else 1.0

    # Reference density refer_rho (defaults to rho0 if 0 or None)
    refer_rho_val = p.get("refer_rho", p.get("Refer_Rho", p.get("rhor", rho0)))
    refer_rho = float(refer_rho_val) if refer_rho_val is not None and float(refer_rho_val) > 0.0 else rho0

    # Elastic modulus e0 and nu
    e0_val = p.get("e0", p.get("E0", p.get("E", p.get("MAT_E", p.get("MAT_E0", p.get("e"))))))
    e0 = float(e0_val) if e0_val is not None else 0.0

    nu_val = p.get("nu", p.get("NU", p.get("MAT_NU", p.get("pr", p.get("poisson")))))
    nu = float(nu_val) if nu_val is not None else 0.0

    # Derived G0 and bulk modulus if not explicitly set
    if "g0" in p and p["g0"] is not None and float(p["g0"]) > 0.0:
        g0 = float(p["g0"])
    elif "G" in p and p["G"] is not None and float(p["G"]) > 0.0:
        g0 = float(p["G"])
    elif (1.0 + nu) != 0.0 and e0 != 0.0:
        g0 = e0 / (2.0 * (1.0 + nu))
    else:
        g0 = 0.0

    if "bulk" in p and p["bulk"] is not None and float(p["bulk"]) > 0.0:
        bulk = float(p["bulk"])
    elif "K" in p and p["K"] is not None and float(p["K"]) > 0.0:
        bulk = float(p["K"])
    elif (1.0 - 2.0 * nu) != 0.0 and e0 != 0.0:
        bulk = e0 / (3.0 * (1.0 - 2.0 * nu))
    else:
        bulk = 0.0

    if e0 == 0.0 and g0 > 0.0:
        if bulk > 0.0 and (3.0 * bulk + g0) > 0.0:
            e0 = 9.0 * bulk * g0 / (3.0 * bulk + g0)
            if nu == 0.0:
                nu = (3.0 * bulk - 2.0 * g0) / (2.0 * (3.0 * bulk + g0))
        elif nu > 0.0:
            e0 = 2.0 * g0 * (1.0 + nu)
    elif g0 == 0.0 and e0 > 0.0 and (1.0 + nu) != 0.0:
        g0 = e0 / (2.0 * (1.0 + nu))
    elif bulk == 0.0 and e0 > 0.0 and (1.0 - 2.0 * nu) != 0.0:
        bulk = e0 / (3.0 * (1.0 - 2.0 * nu))

    # Yield and hardening parameters
    sig0_val = p.get(
        "sig0",
        p.get("sigy", p.get("MAT_SIGY", p.get("Y0", p.get("y0", p.get("Y_0", p.get("sigma_0", p.get("sigma_y0", p.get("A", p.get("a"))))))))),
    )
    sig0 = float(sig0_val) if sig0_val is not None else 0.0

    beta_val = p.get("beta", p.get("MAT_BETA", p.get("cb", p.get("B", p.get("b")))))
    beta = float(beta_val) if beta_val is not None else 0.0

    n_val = p.get("n", p.get("N", p.get("MAT_HARD", p.get("cn", p.get("hard")))))
    n = float(n_val) if n_val is not None else 0.0

    eps_max_val = p.get("eps_max", p.get("MAT_EPS", p.get("epmx", p.get("eps_p_max"))))
    eps_max = float(eps_max_val) if eps_max_val is not None and float(eps_max_val) > 0.0 else _EP20

    sigma_max_val = p.get("sigma_max", p.get("MAT_SIG", p.get("sigmx", p.get("sigm", p.get("sig_max")))))
    sigma_max = float(sigma_max_val) if sigma_max_val is not None and float(sigma_max_val) > 0.0 else _EP20

    # Thermal & melt parameters
    t0_val = p.get("t0", p.get("MAT_T0", p.get("T0")))
    t0 = float(t0_val) if t0_val is not None and float(t0_val) > 0.0 else 300.0

    tmelt_val = p.get("tmelt", p.get("MAT_TMELT", p.get("Tmelt")))
    tmelt = float(tmelt_val) if tmelt_val is not None and float(tmelt_val) > 0.0 else _EP20

    rhoc_p_val = p.get("rhoc_p", p.get("MAT_SPHEAT", p.get("sph", p.get("rho_cp", p.get("rhocp", p.get("Cv", p.get("cv")))))))
    rhoc_p = float(rhoc_p_val) if rhoc_p_val is not None else 0.0

    pmin_val = p.get("pmin", p.get("MAT_PC", p.get("pc", p.get("Pmin"))))
    pmin = float(pmin_val) if pmin_val is not None and float(pmin_val) != 0.0 else -_EP20

    # Pressure and temperature coefficients
    b1_val = p.get("b1", p.get("MAT_B1", p.get("cb1", p.get("A", p.get("a_press")))))
    b1 = float(b1_val) if b1_val is not None else 0.0

    b2_val = p.get("b2", p.get("MAT_B2", p.get("cb2", p.get("b1", p.get("A", p.get("a_press"))))))
    b2 = float(b2_val) if b2_val is not None else b1

    h_val = p.get("h", p.get("MAT_H", p.get("ch", p.get("B", p.get("b_temp")))))
    h = float(h_val) if h_val is not None else 0.0

    f_val = p.get("f", p.get("MAT_F", p.get("cf")))
    f = float(f_val) if f_val is not None else 0.0

    c_rate_val = p.get("c_rate", p.get("C", p.get("MAT_C", p.get("rate_c", 0.0))))
    c_rate = float(c_rate_val) if c_rate_val is not None else 0.0

    eps0_val = p.get("eps0", p.get("eps_dot0", p.get("MAT_EPS0", 1.0)))
    eps0 = float(eps0_val) if eps0_val is not None and float(eps0_val) > 0.0 else 1.0

    return Law49Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e0=e0,
        nu=nu,
        g0=g0,
        bulk=bulk,
        sig0=sig0,
        beta=beta,
        n=n,
        eps_max=eps_max,
        sigma_max=sigma_max,
        t0=t0,
        tmelt=tmelt,
        rhoc_p=rhoc_p,
        pmin=pmin,
        b1=b1,
        b2=b2,
        h=h,
        f=f,
        c_rate=c_rate,
        eps0=eps0,
        title=title,
    )


_ensure_params = _get_params


def build_law49(mat_def: Any) -> Material:
    """Card parsing, defaults & normalization factory for LAW49 (Steinberg-Guinan).

    Cites:
    - ``starter/source/materials/mat/mat049/hm_read_mat49.F``
    - ``engine/source/materials/mat/mat049/m49law.F``
    """
    params_obj = _get_params(mat_def)

    mat_id = 1
    if hasattr(mat_def, "id") and getattr(mat_def, "id") is not None:
        mat_id = getattr(mat_def, "id")
    elif isinstance(mat_def, dict) and "id" in mat_def:
        mat_id = mat_def["id"]

    title = params_obj.title or "LAW49"

    # Populate normalized parameter dictionary
    p = {
        "rho0": params_obj.rho0,
        "rho": params_obj.rho0,
        "refer_rho": params_obj.refer_rho,
        "Refer_Rho": params_obj.refer_rho,
        "e0": params_obj.e0,
        "E0": params_obj.e0,
        "E": params_obj.e0,
        "nu": params_obj.nu,
        "g0": params_obj.g0,
        "G": params_obj.g0,
        "bulk": params_obj.bulk,
        "K": params_obj.bulk,
        "sig0": params_obj.sig0,
        "sigy": params_obj.sig0,
        "beta": params_obj.beta,
        "n": params_obj.n,
        "eps_max": params_obj.eps_max,
        "sigma_max": params_obj.sigma_max,
        "t0": params_obj.t0,
        "tmelt": params_obj.tmelt,
        "rhoc_p": params_obj.rhoc_p,
        "pmin": params_obj.pmin,
        "b1": params_obj.b1,
        "b2": params_obj.b2,
        "h": params_obj.h,
        "f": params_obj.f,
        # CFG parameter names
        "MAT_RHO": params_obj.rho0,
        "MAT_E": params_obj.e0,
        "MAT_E0": params_obj.e0,
        "MAT_NU": params_obj.nu,
        "MAT_SIGY": params_obj.sig0,
        "MAT_BETA": params_obj.beta,
        "MAT_HARD": params_obj.n,
        "MAT_EPS": params_obj.eps_max,
        "MAT_SIG": params_obj.sigma_max,
        "MAT_T0": params_obj.t0,
        "MAT_TMELT": params_obj.tmelt,
        "MAT_SPHEAT": params_obj.rhoc_p,
        "MAT_PC": params_obj.pmin,
        "MAT_B1": params_obj.b1,
        "MAT_B2": params_obj.b2,
        "MAT_H": params_obj.h,
        "MAT_F": params_obj.f,
    }

    if isinstance(mat_def, Material):
        mat = mat_def
        mat.law = 49
        mat.rho0 = params_obj.rho0
        mat.params.update(p)
    else:
        mat = Material(id=mat_id, law=49, rho0=params_obj.rho0, title=title, params=p)

    setattr(mat, "law49_params", params_obj)
    mat.params["law49_params"] = params_obj
    mat.sound_speed_solid = lambda rho=None, extra=None: sound_speed_solid(mat, rho=rho, extra=extra)
    return mat


def extra_shapes(mat: Any, nip: int | None = None) -> dict[str, Tuple[int, ...]]:
    """Persistent state arrays required by LAW49 solid elements."""
    return {
        "theta": (nip,) if nip is not None else (),
        "espe": (nip,) if nip is not None else (),
        "epxe": (nip,) if nip is not None else (),
        "dpla": (nip,) if nip is not None else (),
    }


def sound_speed_solid(
    mat: Any,
    rho: float | np.ndarray | None = None,
    extra: dict | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    """Longitudinal acoustic sound speed for LAW49 solid elements.

    Cites m49law.F lines 133-134:
    c = sqrt(|bulk + 4/3 * G| / rho0).
    """
    p = _get_params(mat)
    rho0 = p.rho0
    current_rho = rho0 if rho is None else rho

    G = p.g0
    if extra is not None and "g" in extra and extra["g"] is not None:
        G = np.asarray(extra["g"], dtype=float)

    c_sq = np.abs(p.bulk + (4.0 / 3.0) * G) / np.maximum(current_rho, _EM20)
    c = np.sqrt(c_sq)
    if isinstance(c, np.ndarray) and c.ndim == 0:
        return float(c)
    return c


sound_speed_solid_law49 = sound_speed_solid
sound_speed = sound_speed_solid


def solid_update(
    mat: Any,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict[str, Any] | None = None,
    return_tuple: bool = False,
    **kwargs: Any,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray | float]:
    """Constitutive stress update for LAW49 (Steinberg-Guinan shock plasticity).

    Ports upstream ``engine/source/materials/mat/mat049/m49law.F``.

    Parameters
    ----------
    mat : Material, Law49Params, or dict
        Material properties.
    sig : (6,) or (n, 6) ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx] (Voigt notation).
    deps : (6,) or (n, 6) ndarray
        Strain increment tensor (engineering shear: dgamma_xy, dgamma_yz, dgamma_zx).
    epsp : (n,) or float, optional
        Accumulated equivalent plastic strain history.
    dt : float, optional
        Time step increment.
    extra : dict, optional
        Persistent element state dictionary (e.g. 'theta', 'temp', 'espe', 'eint', 'rho', 'off').
    return_tuple : bool, default False
        If True, returns (sig, epsp, c). Otherwise returns sig.

    Returns
    -------
    sig_new : (6,) or (n, 6) ndarray (or tuple if return_tuple=True)
        Updated Cauchy stress tensor.
    """
    # Check if called as element group style: solid_update(group, x, u, ur, dt, fint, mint)
    if not (isinstance(sig, np.ndarray) and (deps is None or isinstance(deps, np.ndarray))):
        if hasattr(mat, "elements") or hasattr(mat, "nel") or hasattr(mat, "nodes") or kwargs.get("fint") is not None:
            fint = kwargs.get("fint", None)
            return fint

    p = _get_params(mat)

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)

    if is_1d:
        sig_arr = sig_arr.reshape(1, 6)
        deps_arr = deps_arr.reshape(1, 6)

    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.asarray(epsp, dtype=float).copy()
        if epsp_arr.ndim == 0:
            epsp_arr = np.full(nel, float(epsp_arr))

    if extra is None:
        extra = {}

    off = np.asarray(extra.get("off", np.ones(nel, dtype=float)), dtype=float)
    if off.shape != (nel,):
        off = np.full(nel, float(off.flat[0]) if off.size > 0 else 1.0, dtype=float)

    # Material constants
    rho0 = p.rho0
    g0 = p.g0
    bulk = p.bulk
    sig0 = p.sig0
    beta = p.beta
    n = p.n
    eps_max = p.eps_max
    sigma_max = p.sigma_max
    t0 = p.t0
    tmelt = p.tmelt
    rhoc_p = p.rhoc_p
    pmin = p.pmin
    b1 = p.b1
    b2 = p.b2
    h = p.h
    f = p.f

    # 1. Hydrostatic pressure P and mean strain rate increment Dav (m49law.F lines 90-93)
    P_old = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
    Dav = -(deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0
    Emelt = rhoc_p * tmelt

    # Density ratio DF = rho / rho0 (m49law.F line 98)
    if "DF" in extra and extra["DF"] is not None:
        df = np.asarray(extra["DF"], dtype=float)
    elif "df" in extra and extra["df"] is not None:
        df = np.asarray(extra["df"], dtype=float)
    elif "rho" in extra and extra["rho"] is not None:
        df = np.asarray(extra["rho"], dtype=float) / rho0
    else:
        df = np.ones(nel, dtype=float)
    if df.shape != (nel,):
        df = np.full(nel, float(df.flat[0]) if df.size > 0 else 1.0, dtype=float)

    # Current temperature theta (m49law.F line 99)
    if "theta" in extra and extra["theta"] is not None:
        theta = np.asarray(extra["theta"], dtype=float).copy()
        if np.all(theta == 0.0) and t0 > 0.0:
            theta = np.full(nel, t0, dtype=float)
    elif "temp" in extra and extra["temp"] is not None:
        theta = np.asarray(extra["temp"], dtype=float).copy()
        if np.all(theta == 0.0) and t0 > 0.0:
            theta = np.full(nel, t0, dtype=float)
    elif "temperature" in extra and extra["temperature"] is not None:
        theta = np.asarray(extra["temperature"], dtype=float).copy()
        if np.all(theta == 0.0) and t0 > 0.0:
            theta = np.full(nel, t0, dtype=float)
    else:
        theta = np.full(nel, t0, dtype=float)
    if theta.shape != (nel,):
        theta = np.full(nel, float(theta.flat[0]) if theta.size > 0 else t0, dtype=float)

    # Specific internal energy espe (m49law.F lines 100-106)
    if "espe" in extra and extra["espe"] is not None and np.any(extra["espe"] != 0.0):
        espe = np.asarray(extra["espe"], dtype=float)
    elif "eint" in extra and extra["eint"] is not None:
        espe_raw = np.asarray(extra["eint"], dtype=float)
        if "vol0" in extra and extra["vol0"] is not None:
            v0 = np.asarray(extra["vol0"], dtype=float)
            espe = np.where(v0 > _EM20, espe_raw / np.maximum(v0, _EM20), espe_raw)
        else:
            espe = espe_raw
    elif "espe" in extra and extra["espe"] is not None:
        espe = np.asarray(extra["espe"], dtype=float)
    else:
        espe = np.zeros(nel, dtype=float)
    if espe.shape != (nel,):
        espe = np.full(nel, float(espe.flat[0]) if espe.size > 0 else 0.0, dtype=float)

    # 2. Scaling variables QA, QB, QC (m49law.F lines 98-106)
    qa = P_old * np.cbrt(df)
    qb = 1.0 - h * (theta - t0)

    qc = np.ones(nel, dtype=float)
    if Emelt > 0.0 and f > 0.0:
        melted_energy = (espe >= Emelt)
        unmelted_energy = ~melted_energy
        qc[melted_energy] = 0.0
        if np.any(unmelted_energy):
            denom = espe[unmelted_energy] - Emelt
            qc[unmelted_energy] = np.exp(f * espe[unmelted_energy] / denom)

    # 3. Current shear modulus G and yield scaling factor QD (m49law.F lines 107-108)
    G = g0 * (b1 * qa + qb) * qc
    qd = (b2 * qa + qb) * qc

    # 4. Cold work hardening QE (m49law.F lines 109-115)
    qe = np.empty(nel, dtype=float)
    le_zero = (epsp_arr <= 0.0)
    gt_max = (epsp_arr > eps_max)
    mid = (~le_zero) & (~gt_max)

    qe[le_zero] = sig0
    qe[gt_max] = sig0 * ((1.0 + beta * eps_max) ** n)
    qe[mid] = sig0 * ((1.0 + beta * epsp_arr[mid]) ** n)

    # Nominal yield stress YLD
    yld = np.minimum(sigma_max, qe) * qd

    # High strain rate Johnson-Cook rate multiplier (if c_rate > 0)
    rate_fac = 1.0
    if p.c_rate > 0.0:
        if "eps_dot" in extra and extra["eps_dot"] is not None:
            eps_rate = np.asarray(extra["eps_dot"], dtype=float)
        elif dt > 0.0:
            deps_dev = deps_arr.copy()
            deps_dev[:, :3] += Dav[:, None]
            deps_eq = np.sqrt(
                (2.0 / 3.0)
                * (
                    deps_dev[:, 0] ** 2 + deps_dev[:, 1] ** 2 + deps_dev[:, 2] ** 2
                    + 2.0 * (deps_dev[:, 3] ** 2 + deps_dev[:, 4] ** 2 + deps_dev[:, 5] ** 2)
                )
            )
            eps_rate = deps_eq / dt
        else:
            eps_rate = np.zeros(nel, dtype=float)

        ratio = np.maximum(eps_rate, p.eps0) / max(p.eps0, _EM20)
        rate_fac = 1.0 + p.c_rate * np.log(ratio)
        rate_fac = np.maximum(rate_fac, 0.0)
        yld = yld * rate_fac

    # 5. Deviatoric elastic trial stress (m49law.F lines 121-130)
    g1 = G * off
    g2 = 2.0 * g1

    s_tr = np.empty_like(sig_arr)
    s_tr[:, 0] = sig_arr[:, 0] + P_old + g2 * (deps_arr[:, 0] + Dav)
    s_tr[:, 1] = sig_arr[:, 1] + P_old + g2 * (deps_arr[:, 1] + Dav)
    s_tr[:, 2] = sig_arr[:, 2] + P_old + g2 * (deps_arr[:, 2] + Dav)
    s_tr[:, 3] = sig_arr[:, 3] + g1 * deps_arr[:, 3]
    s_tr[:, 4] = sig_arr[:, 4] + g1 * deps_arr[:, 4]
    s_tr[:, 5] = sig_arr[:, 5] + g1 * deps_arr[:, 5]

    # 6. Acoustic sound speed (m49law.F lines 132-135, before melting reset G=0 at line 147)
    c_solid = sound_speed_solid(p, rho=rho0, extra={"g": G})
    if np.isscalar(c_solid):
        c_solid = np.full(nel, float(c_solid), dtype=float)

    # von Mises equivalent trial stress AJ2 (m49law.F lines 139-140)
    j2 = (
        0.5 * (s_tr[:, 0] ** 2 + s_tr[:, 1] ** 2 + s_tr[:, 2] ** 2)
        + s_tr[:, 3] ** 2
        + s_tr[:, 4] ** 2
        + s_tr[:, 5] ** 2
    )
    aj2 = np.sqrt(3.0 * np.maximum(j2, 0.0))

    # 7. Check melting and radial return (m49law.F lines 145-168)
    melted = (theta >= tmelt)
    unmelted = ~melted

    qh = np.zeros(nel, dtype=float)
    scale = np.zeros(nel, dtype=float)

    if np.any(unmelted):
        u_idx = unmelted
        if n >= 1.0:
            qh[u_idx] = qd[u_idx] * sig0 * beta * n * ((1.0 + beta * epsp_arr[u_idx]) ** (n - 1.0))
        else:
            pos = u_idx & (epsp_arr > 0.0)
            qh[pos] = qd[pos] * sig0 * beta * n / ((1.0 + beta * epsp_arr[pos]) ** (1.0 - n))
        if p.c_rate > 0.0:
            if isinstance(rate_fac, np.ndarray):
                qh[u_idx] *= rate_fac[u_idx]
            else:
                qh[u_idx] *= rate_fac

        elastic = u_idx & (aj2 <= yld)
        scale[elastic] = 1.0

        plastic = u_idx & (aj2 > yld)
        if np.any(plastic):
            scale[plastic] = np.where(aj2[plastic] > 0.0, yld[plastic] / aj2[plastic], 0.0)

    # Melted state: G=0, YLD=0, scale=0 (m49law.F lines 146-150)
    scale[melted] = 0.0
    G[melted] = 0.0
    yld[melted] = 0.0
    qh[melted] = 0.0

    # 8. Plastic strain increment DPLA (m49law.F line 173)
    denom = np.maximum(3.0 * G + qh, _EM15)
    dpla = (1.0 - scale) * aj2 / denom
    dpla = np.where(melted, 0.0, dpla)

    # Actual yield stress after return (m49law.F line 175)
    yld_actual = yld + dpla * qh

    # Return deviatoric stress (m49law.F lines 176-181)
    s_new = scale[:, None] * s_tr * off[:, None]

    # Accumulate equivalent plastic strain (m49law.F lines 182-183: EPXE = (EPXE + DPLA) * OFF)
    epsp_new = (epsp_arr + dpla) * off

    # 9. Temperature rise due to plastic work (m49law.F lines 198-207)
    if rhoc_p > 0.0:
        d_theta = yld_actual * dpla / rhoc_p
        theta_new = theta + d_theta
    else:
        d_theta = np.zeros(nel, dtype=float)
        theta_new = theta

    # 10. Volumetric pressure update & tensile cutoff (P_min)
    if "P_eos" in extra and extra["P_eos"] is not None:
        P_new = np.asarray(extra["P_eos"], dtype=float)
    elif "rho" in extra and extra["rho"] is not None:
        mu = np.asarray(extra["rho"], dtype=float) / rho0 - 1.0
        P_new = bulk * mu
    else:
        P_new = P_old + 3.0 * bulk * Dav

    P_new = np.maximum(P_new, pmin) * off

    # Recombine deviatoric stress and pressure into Cauchy stress
    sig_new = s_new.copy()
    sig_new[:, 0] -= P_new
    sig_new[:, 1] -= P_new
    sig_new[:, 2] -= P_new
    sig_new *= off[:, None]

    # Store state variables in extra
    if extra is not None:
        extra["espe"] = espe
        extra["sigy"] = yld_actual
        extra["yld"] = yld_actual
        extra["defp"] = epsp_new
        extra["epxe"] = epsp_new
        extra["dpla"] = dpla
        extra["theta"] = theta_new
        extra["temp"] = theta_new
        extra["temperature"] = theta_new
        extra["dtheta"] = d_theta
        extra["g"] = G
        extra["qd"] = qd
        extra["qc"] = qc
        extra["qh"] = qh
        extra["scale"] = scale
        extra["p"] = P_new
        extra["s"] = s_new
        extra["c_solid"] = c_solid
        extra["sound_speed"] = c_solid

    # In-place mutations if arrays were provided
    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = sig_new[0] if is_1d else sig_new
        except Exception:
            pass

    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = epsp_new[0] if is_1d else epsp_new
        except Exception:
            pass

    out_sig = sig_new[0] if is_1d else sig_new
    out_epsp = epsp_new[0] if is_1d else epsp_new
    out_c = float(c_solid[0]) if is_1d else c_solid

    if return_tuple:
        return out_sig, out_epsp, out_c
    return out_sig


solid_update_law49 = solid_update


def shell_update(*args: Any, **kwargs: Any) -> Any:
    """Plane-stress shell update is not supported for LAW49."""
    raise NotImplementedError("LAW49 (Steinberg-Guinan) is implemented for 3D solid and SPH elements only.")


shell_update_law49 = shell_update


# -----------------------------------------------------------------------------
# State Copy Helper
# -----------------------------------------------------------------------------

def _copy_extra(extra: dict[str, Any] | None) -> dict[str, Any] | None:
    """Deep copy dictionary of state variables for LAW49."""
    if extra is None:
        return None
    res: dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        elif hasattr(v, "copy"):
            try:
                res[k] = v.copy()
            except Exception:
                res[k] = v
        else:
            res[k] = v
    return res


# -----------------------------------------------------------------------------
# Algorithmic Consistent Tangent Stiffness Tensor
# -----------------------------------------------------------------------------

def tangent_law49_solid(
    mat: Any,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    eps_dot: np.ndarray | None = None,
    dt: float = 0.0,
    *args: Any,
    epsp: np.ndarray | None = None,
    epsp_incr: np.ndarray | None = None,
    extra: dict | None = None,
    symmetric: bool = False,
    h: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent elastoplastic tangent stiffness matrix for LAW49 in Voigt notation.

    Voigt convention: [xx, yy, zz, xy, yz, zx] with engineering shear.

    Parameters
    ----------
    mat : Material or dict
        Material definition.
    sig : (6,) or (n, 6) ndarray, optional
        Stress state (old stress if deps is provided, or current stress).
    deps : (6,) or (n, 6) ndarray, optional
        Strain increment tensor.
    eps_dot, dt :
        Optional rate and time step.
    epsp, epsp_incr :
        Optional plastic strain history.
    extra : dict, optional
        Extra state views (e.g. 'temp', 'e_spe', 'off', 'g').
    symmetric : bool, default False
        If True, returns symmetrized matrix 0.5 * (D + D^T).
    h : float, default 1e-7
        Perturbation step size for numerical algorithmic tangent.

    Returns
    -------
    D : (6, 6) or (n, 6, 6) ndarray
        Consistent tangent stiffness tensor.
    """
    # 0. Disambiguate keyword arguments
    if "deps" in kwargs and deps is None:
        deps = kwargs.pop("deps")
    if "d_eps" in kwargs and deps is None:
        deps = kwargs.pop("d_eps")
    if "eps" in kwargs and deps is None:
        deps = kwargs.pop("eps")
    if "sig" in kwargs and sig is None:
        sig = kwargs.pop("sig")
    if "epsp" in kwargs and epsp is None:
        epsp = kwargs.pop("epsp")
    if "epsp_incr" in kwargs and epsp_incr is None:
        epsp_incr = kwargs.pop("epsp_incr")
    if "extra" in kwargs and extra is None:
        extra = kwargs.pop("extra")
    if "dt" in kwargs:
        dt = float(kwargs.pop("dt"))
    if "h" in kwargs:
        h = float(kwargs.pop("h"))
    if "symmetric" in kwargs:
        symmetric = bool(kwargs.pop("symmetric"))

    # 1. Disambiguate positional arguments
    if deps is None and eps_dot is not None:
        deps_check = np.asarray(eps_dot)
        if (deps_check.ndim == 1 and deps_check.shape[0] == 6) or (deps_check.ndim == 2 and deps_check.shape[1] == 6):
            deps = deps_check
            eps_dot = None

    if len(args) >= 1:
        if isinstance(args[0], dict) and extra is None:
            extra = args[0]
        elif isinstance(args[0], (int, float)):
            dt = float(args[0])
        elif isinstance(args[0], np.ndarray):
            arr = np.asarray(args[0])
            if (arr.ndim == 1 and arr.shape[0] == 6) or (arr.ndim == 2 and arr.shape[1] == 6):
                if deps is None:
                    deps = arr
            elif epsp_incr is None:
                epsp_incr = args[0]
    if len(args) >= 2:
        if isinstance(args[1], dict) and extra is None:
            extra = args[1]
        elif isinstance(args[1], (int, float)):
            dt = float(args[1])
        elif epsp_incr is None:
            epsp_incr = args[1]
    if len(args) >= 3 and isinstance(args[2], dict) and extra is None:
        extra = args[2]

    # 2. Sizing and dimensionality
    if sig is not None and deps is not None:
        n_sig = 1 if np.ndim(sig) <= 1 else np.asarray(sig).shape[0]
        n_deps = 1 if np.ndim(deps) <= 1 else np.asarray(deps).shape[0]
        nel = max(n_sig, n_deps)
        single = (np.ndim(sig) <= 1 and np.ndim(deps) <= 1)
    elif sig is not None:
        nel = 1 if np.ndim(sig) <= 1 else np.asarray(sig).shape[0]
        single = (np.ndim(sig) <= 1)
    elif deps is not None:
        nel = 1 if np.ndim(deps) <= 1 else np.asarray(deps).shape[0]
        single = (np.ndim(deps) <= 1)
    else:
        nel = 1
        single = True

    if nel == 0:
        return np.empty((0, 6, 6), dtype=float)

    if sig is not None:
        sig_arr = np.asarray(sig, dtype=float).copy()
        if sig_arr.ndim == 1:
            sig_arr = sig_arr.reshape(1, -1)
        if sig_arr.shape[0] == 1 and nel > 1:
            sig_arr = np.repeat(sig_arr, nel, axis=0)
    else:
        sig_arr = np.zeros((nel, 6), dtype=float)

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float).copy()
        if deps_arr.ndim == 1:
            deps_arr = deps_arr.reshape(1, -1)
        if deps_arr.shape[0] == 1 and nel > 1:
            deps_arr = np.repeat(deps_arr, nel, axis=0)
    else:
        deps_arr = None

    # Element deletion / deactivation status
    off_arr = np.ones(nel, dtype=float)
    if extra is not None:
        for k in ("off", "off49"):
            if k in extra and extra[k] is not None:
                val = np.asarray(extra[k], dtype=float).flatten()
                if len(val) == 1 and nel > 1:
                    off_arr = np.full(nel, val[0], dtype=float)
                else:
                    off_arr = val.copy()
                break
    deleted_mask = (off_arr <= 0.0)

    analytical_requested = bool(
        kwargs.get("analytical", False) or kwargs.get("analytic", False) or kwargs.get("method") == "analytical"
    )

    # 3. If deps is provided and not explicitly requesting pure analytical:
    if deps_arr is not None and not analytical_requested:
        D = np.zeros((nel, 6, 6), dtype=float)
        active = ~deleted_mask
        h_step = float(h)
        dt_call = dt if dt > 0.0 else 1.0
        if np.any(active):
            for j in range(6):
                ej = np.zeros_like(deps_arr)
                ej[:, j] = h_step

                ext_p = _copy_extra(extra)
                ext_m = _copy_extra(extra)

                sp = solid_update_law49(mat, sig_arr.copy(), deps=deps_arr + ej, dt=dt_call, extra=ext_p, epsp=epsp)
                sm = solid_update_law49(mat, sig_arr.copy(), deps=deps_arr - ej, dt=dt_call, extra=ext_m, epsp=epsp)

                if sp.ndim == 1:
                    sp = sp.reshape(1, 6)
                if sm.ndim == 1:
                    sm = sm.reshape(1, 6)

                D[:, :, j] = (sp[:, :6] - sm[:, :6]) / (2.0 * h_step)

        if np.any(deleted_mask):
            D[deleted_mask] = 0.0

        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))

        return D[0] if single else D

    # 4. Pure analytical consistent tangent (from sig_arr and extra)
    p = _get_params(mat)
    G = p.g0
    K = p.bulk

    if extra is not None and "g" in extra and extra["g"] is not None:
        G_arr = np.atleast_1d(np.asarray(extra["g"], dtype=float))
    else:
        G_arr = np.full(nel, G, dtype=float)

    # Elastic tangent tensor C (6x6) in Voigt notation: [xx, yy, zz, xy, yz, zx]
    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    KeeT = K * np.outer(ee, ee)

    I_dev = np.diag([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5])
    I_dev[0, 1] = I_dev[0, 2] = I_dev[1, 0] = I_dev[1, 2] = I_dev[2, 0] = I_dev[2, 1] = -1.0 / 3.0

    D = np.zeros((nel, 6, 6), dtype=float)
    for i in range(nel):
        D[i] = KeeT + 2.0 * G_arr[i] * I_dev

    if epsp_incr is not None:
        epsp_incr_arr = np.atleast_1d(np.asarray(epsp_incr, dtype=float))
        if epsp_incr_arr.shape != (nel,):
            epsp_incr_arr = np.full(nel, float(epsp_incr_arr.flat[0]), dtype=float)

        plastic = epsp_incr_arr > 0.0
        if np.any(plastic):
            p_old = (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
            s = sig_arr.copy()
            s[:, 0] -= p_old
            s[:, 1] -= p_old
            s[:, 2] -= p_old

            epsp_arr = np.zeros(nel, dtype=float) if epsp is None else np.atleast_1d(np.asarray(epsp, dtype=float))

            for i in range(nel):
                if not plastic[i] or G_arr[i] <= 0.0:
                    continue
                g_i = G_arr[i]
                dep = epsp_incr_arr[i]

                s_i = s[i]
                snorm = math.sqrt(
                    s_i[0] ** 2 + s_i[1] ** 2 + s_i[2] ** 2
                    + 2.0 * (s_i[3] ** 2 + s_i[4] ** 2 + s_i[5] ** 2)
                )
                snorm = max(snorm, 1e-30)
                Nv = s_i / snorm
                q = math.sqrt(1.5) * snorm
                q_tr = q + 3.0 * g_i * dep

                cur_epsp = epsp_arr[i] if len(epsp_arr) > i else 0.0
                if p.n >= 1.0:
                    qh = p.sig0 * p.beta * p.n * ((1.0 + p.beta * cur_epsp) ** (p.n - 1.0))
                else:
                    qh = (
                        p.sig0 * p.beta * p.n / ((1.0 + p.beta * max(cur_epsp, _EM15)) ** (1.0 - p.n))
                        if cur_epsp > 0.0
                        else 0.0
                    )
                if cur_epsp >= p.eps_max:
                    qh = 0.0

                a = 3.0 * g_i * dep / q_tr
                b = 6.0 * g_i * g_i * (dep / q_tr - 1.0 / max(3.0 * g_i + qh, _EM15))

                NN = np.outer(Nv, Nv)
                D[i] = KeeT + 2.0 * g_i * (1.0 - a) * I_dev + b * NN

    if np.any(deleted_mask):
        D[deleted_mask] = 0.0

    if symmetric:
        D = 0.5 * (D + np.swapaxes(D, -1, -2))

    return D[0] if single else D


consistent_solid_tangent = tangent_law49_solid
solid_tangent = tangent_law49_solid
solid_tangent_law49 = tangent_law49_solid


def tangent(group: Any = None, x: Any = None, epsp_incr: Any = None) -> Any:
    """Stiffness tangent dispatch for element groups or implicit solver."""
    if group is None:
        return None
    mat = getattr(group, "mat", None) or getattr(group, "material", None)
    if mat is not None:
        return solid_tangent(mat, epsp_incr=epsp_incr)
    return None


def _register() -> None:
    """Register LAW49 with pyradioss material physics registry."""
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

        for key in (
            49,
            "49",
            "LAW49",
            "STEINB",
            "STEINBERG",
            "STEINBERG_GUINAN",
            "MAT_LAW49",
            "MAT_STEINB",
            "MAT_STEINBERG",
            "LAW49_STEINB",
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law49
    except Exception:
        pass


_register()
