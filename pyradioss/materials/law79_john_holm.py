# $OR_SRC/engine/source/materials/mat/mat079/sigeps79.F
# Function: SIGEPS79 (lines 28-311)
# $OR_SRC/starter/source/materials/mat/mat079/hm_read_mat79.F
# Function: HM_READ_MAT79 (lines 38-250)
r"""LAW79 — Johnson-Holmquist (JH-2) ceramic/brittle damage model (/MAT/LAW79, /MAT/JOHN_HOLM).

Fortran origins:
- ``engine/source/materials/mat/mat079/sigeps79.F`` (solid constitutive update)
- ``starter/source/materials/mat/mat079/hm_read_mat79.F`` (starter card reader, defaults & parameter estimation)
- ``hm_cfg_files/config/CFG/radioss2023/MAT/matl79_79.cfg`` (CFG attributes & card format)

Theory
------
LAW79 models brittle materials (ceramics, glass, rocks) subjected to large strains,
high strain rates, and high pressures based on the Johnson-Holmquist (JH-2) formulation:

1. Normalized Strengths:
   Intact normalized strength:
   \(\sigma_i^* = A (P^* + T^*)^N (1 + C \ln(\max(\dot{\varepsilon}, \dot{\varepsilon}_0) / \dot{\varepsilon}_0))\)   (if \(P^* + T^* > 0\), else 0)
   Fractured normalized strength:
   \(\sigma_f^* = \min\left( B (P^*)^M (1 + C \ln(\max(\dot{\varepsilon}, \dot{\varepsilon}_0) / \dot{\varepsilon}_0)), \sigma_{f,\max}^* \right)\) (if \(P^* > 0\), else 0)
   Current normalized strength:
   \(\sigma^* = (1 - D) \sigma_i^* + D \sigma_f^*\)
   Actual yield strength:
   \(\sigma_y = \sigma^* \cdot \sigma_{HEL}\) with \(\sigma_{HEL} = 1.5 (HEL - P_{HEL})\).

2. Deviatoric Elastic Trial & Radial Return:
   \(P_{old} = -\frac{1}{3} \text{tr}(\boldsymbol{\sigma}^{old})\)
   \(D_{av} = \frac{1}{3} \text{tr}(\Delta\boldsymbol{\varepsilon})\)
   \(s_{ij}^{trial} = \sigma_{ij}^{old} + P_{old} \delta_{ij} + 2 G (\Delta\varepsilon_{ij} - D_{av}\delta_{ij})\) (normal)
   \(s_{ij}^{trial} = \sigma_{ij}^{old} + G \Delta\varepsilon_{ij}\) (shear, engineering shear)
   \(J_2 = \frac{1}{2} \mathbf{s}^{trial} : \mathbf{s}^{trial}\), \(\sigma_{vm} = \sqrt{3 J_2}\)
   \(\sigma^* = \sigma_{vm} / \sigma_{HEL}\)
   If \(\sigma^* < \sigma_y\): \(\text{SCALE} = 1.0\) else \(\text{SCALE} = \sigma_y / \sigma^* = (\sigma_y \sigma_{HEL}) / \sigma_{vm}\).
   \(s_{ij} = \text{SCALE} \cdot s_{ij}^{trial}\).

3. Damage Evolution:
   Plastic strain to failure:
   \(\varepsilon_p^f = D_1 (P^* + T^*)^{D_2}\) (if \(P^* + T^* \ge 0\), else 0)
   Plastic strain increment:
   \(\Delta\varepsilon_p = \frac{(1 - \text{SCALE}) \sigma_{vm}}{3 \sqrt{3} G}\)
   Damage accumulation:
   \(D \leftarrow \min(1.0, D + \Delta\varepsilon_p / \varepsilon_p^f)\) (if \(\varepsilon_p^f > 0\), else \(D = 1.0\) if yielding).

4. Bulking Pressure Increment (internal shear strain energy loss -> dilation):
   When \(D > D_{old}\), \(\mu > 0\) and \(\text{off} = 1\):
   \(\Delta U = \frac{\sigma_{y,old}^2 - \sigma_{y,curr}^2}{6 G} \cdot \sigma_{HEL}^2\)
   If \(\Delta U > 0\):
   \(\Delta P = -P_1 + \sqrt{(\Delta P_{old} + P_1)^2 + 2 \beta K_1 \Delta U}\) with \(P_1 = K_1 \mu\).

5. Equation of State Pressure:
   \(P = K_1 \mu + \Delta P\)
   If \(\mu > 0\): \(P \leftarrow P + K_2 \mu^2 + K_3 \mu^3\)
   Else if \(IDEL \ne 1\): \(P \leftarrow \max(P, -T^* P_{HEL} (1 - D))\)
   \(P^* = P / P_{HEL}\).

6. Element Deletion (IDEL):
   - IDEL = 0: No deletion
   - IDEL = 1: Deletion if \(P^* + T^* < 0\) (hydrostatic tension)
   - IDEL = 2: Deletion if \(\varepsilon_p > \varepsilon_{\max}\)
   - IDEL = 3: Deletion if \(D \ge 1.0\)
   When deletion is triggered: \(\text{off} \leftarrow 0.8\), decaying to 0.

7. Longitudinal Acoustic Wave Speed:
   \(\frac{\partial P}{\partial \mu} = K_1 + 2 K_2 \mu + 3 K_3 \mu^2\) (\(\mu > 0\)) else \(K_1\)
   \(c = \sqrt{\frac{\frac{\partial P}{\partial \mu} + \frac{4}{3} G}{\rho_0}}\).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pyradioss.model.entities import Material

_EP20 = 1.0e20
_EM20 = 1.0e-20
_EM30 = 1.0e-30


# -----------------------------------------------------------------------------
# Parameter dataclass
# -----------------------------------------------------------------------------

@dataclass
class Law79Params:
    """Parameters for /MAT/LAW79 (Johnson-Holmquist JH-2 ceramic/brittle model).

    Cites starter/source/materials/mat/mat079/hm_read_mat79.F.
    """

    rho0: float = 0.0
    refer_rho: float = 0.0
    shear: float = 0.0
    a: float = 0.0
    b: float = 0.0
    m: float = 1.0
    n: float = 1.0
    c: float = 0.0
    eps0: float = 1.0
    sigfmax: float = _EP20
    fcut: float = 0.0
    t: float = 0.0
    hel: float = 0.0
    phel: float = 0.0
    d1: float = 0.0
    d2: float = 1.0
    idel: int = 0
    epsmax: float = _EP20
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    beta: float = 1.0
    title: str = ""

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.eps0 == 0.0:
            self.eps0 = 1.0
        if self.sigfmax == 0.0:
            self.sigfmax = _EP20
        if self.epsmax == 0.0:
            self.epsmax = _EP20
        self.idel = max(0, min(int(self.idel), 3))

    @property
    def shel(self) -> float:
        """Normalized shear strength scaling: shel = 1.5 * (hel - phel)."""
        return 1.5 * (self.hel - self.phel)

    @property
    def tstar(self) -> float:
        """Normalized tensile limit: tstar = t / phel."""
        return (self.t / self.phel) if self.phel != 0.0 else 0.0

    @property
    def young(self) -> float:
        """Derived Young's modulus E = 9 * K1 * G / (3 * K1 + G)."""
        denom = 3.0 * self.k1 + self.shear
        return (9.0 * self.k1 * self.shear / denom) if denom != 0.0 else 0.0

    @property
    def nu(self) -> float:
        """Derived Poisson's ratio nu = (3 * K1 - 2 * G) / (6 * K1 + 2 * G)."""
        denom = 6.0 * self.k1 + 2.0 * self.shear
        return ((3.0 * self.k1 - 2.0 * self.shear) / denom) if denom != 0.0 else 0.0

    @property
    def G(self) -> float:
        return self.shear

    @property
    def E(self) -> float:
        return self.young

    @property
    def K(self) -> float:
        return self.k1

    @property
    def bulk(self) -> float:
        return self.k1

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def t0(self) -> float:
        return self.t

    @property
    def sig0(self) -> float:
        return self.a


# -----------------------------------------------------------------------------
# Parameter extraction & factory (hm_read_mat79.F)
# -----------------------------------------------------------------------------

def _extract_param(
    d: Dict[str, Any],
    keys: Tuple[str, ...],
    default: Any = 0.0,
) -> Any:
    """Look up first matching key in dictionary."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_params(mat: Any) -> Law79Params:
    """Extract Law79Params from Material, Law79Params, MatLaw79, or dict."""
    if isinstance(mat, Law79Params):
        return mat

    if hasattr(mat, "law79_params") and isinstance(mat.law79_params, Law79Params):
        return mat.law79_params

    p: Dict[str, Any] = {}
    mat_id = 1
    title = ""
    rho0_val = None

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        mat_id = mat.id
        title = mat.title
        rho0_val = getattr(mat, "rho0", None)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        mat_id = mat.get("id", 1)
        title = mat.get("title", "")
        rho0_val = mat.get("rho0") or mat.get("rho") or mat.get("density")
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
        mat_id = getattr(mat, "id", 1)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho", getattr(mat, "rho0", None))
    else:
        # Generic object (e.g. MatLaw79)
        mat_id = getattr(mat, "id", 1)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho", getattr(mat, "rho0", None))
        for attr in (
            "rho", "refer_rho", "g", "shear", "a", "b", "m", "n", "c",
            "eps0", "sigma_fmax", "sigfmax", "fcut", "t0", "t", "hel",
            "phel", "d1", "d2", "idel", "epsmax", "k1", "k2", "k3", "beta",
        ):
            if hasattr(mat, attr):
                p[attr] = getattr(mat, attr)

    # 1. Density
    if rho0_val is None or float(rho0_val) == 0.0:
        rho0_val = _extract_param(p, ("rho0", "rho", "density", "MAT_RHO", "Refer_Rho"), 0.0)
    rho0 = float(rho0_val)

    refer_rho_val = _extract_param(p, ("refer_rho", "Refer_Rho", "rho_ref", "refer_density"), rho0)
    refer_rho = float(refer_rho_val) if refer_rho_val is not None and float(refer_rho_val) != 0.0 else rho0

    # 2. Shear modulus G
    shear_val = _extract_param(p, ("shear", "G", "tau_shear", "g", "MAT_G"), 0.0)
    shear = float(shear_val)

    # 3. Strength constants a, b, m, n
    a = float(_extract_param(p, ("a", "A", "MAT_A", "aa"), 0.0))
    b = float(_extract_param(p, ("b", "B", "MAT_B", "bb"), 0.0))
    m = float(_extract_param(p, ("m", "M", "MAT_M", "mm"), 1.0))
    n = float(_extract_param(p, ("n", "N", "MAT_N", "nn"), 1.0))

    # 4. Rate parameters c, eps0, sigfmax, fcut
    c = float(_extract_param(p, ("c", "C", "MAT_C", "cc"), 0.0))
    eps0_val = _extract_param(p, ("eps0", "EPS0", "epsp_0", "MAT_Epsilon_F", "epsilon_f"), 1.0)
    eps0 = float(eps0_val) if eps0_val is not None and float(eps0_val) != 0.0 else 1.0
    if c == 0.0 and (eps0_val is None or float(eps0_val) == 0.0):
        eps0 = 1.0

    sigfmax_val = _extract_param(p, ("sigfmax", "SIGFMAX", "sigma_f_max", "sigma_fmax", "MAT_SIG1max_t"), _EP20)
    sigfmax = float(sigfmax_val) if sigfmax_val is not None and float(sigfmax_val) not in (0.0, 1.0e30) else _EP20

    fcut_val = _extract_param(p, ("fcut", "FCUT", "asrate", "ASRATE", "MAT_FCUT"), 0.0)
    fcut = float(fcut_val)

    # 5. Limits: T0, HEL, PHEL
    t_val = _extract_param(p, ("t", "T", "T0", "t0", "tmax", "TMAX", "MAT_T0"), 0.0)
    t = float(t_val)
    hel = float(_extract_param(p, ("hel", "HEL", "MAT_E", "e"), 0.0))
    phel = float(_extract_param(p, ("phel", "PHEL", "MAT_EPS", "eps"), 0.0))

    # 6. Damage & deletion: D1, D2, IDEL, EPSMAX
    d1 = float(_extract_param(p, ("d1", "D1"), 0.0))
    d2 = float(_extract_param(p, ("d2", "D2"), 1.0))
    idel_val = _extract_param(p, ("idel", "IDEL"), 0)
    idel = max(0, min(int(idel_val), 3))

    epsmax_val = _extract_param(p, ("epsmax", "EPSMAX"), _EP20)
    epsmax = float(epsmax_val) if epsmax_val is not None and float(epsmax_val) not in (0.0, 1.0e30) else _EP20

    # 7. Pressure coefficients K1, K2, K3, BETA
    k1 = float(_extract_param(p, ("k1", "K1", "MAT_BULK", "bulk"), 0.0))
    k2 = float(_extract_param(p, ("k2", "K2"), 0.0))
    k3 = float(_extract_param(p, ("k3", "K3"), 0.0))
    beta_val = _extract_param(p, ("beta", "BETA", "MAT_Beta"), 1.0)
    beta = float(beta_val)

    return Law79Params(
        rho0=rho0,
        refer_rho=refer_rho,
        shear=shear,
        a=a,
        b=b,
        m=m,
        n=n,
        c=c,
        eps0=eps0,
        sigfmax=sigfmax,
        fcut=fcut,
        t=t,
        hel=hel,
        phel=phel,
        d1=d1,
        d2=d2,
        idel=idel,
        epsmax=epsmax,
        k1=k1,
        k2=k2,
        k3=k3,
        beta=beta,
        title=title,
    )


def build_law79(mat_def: Any) -> Material:
    """Card parsing, validation and factory for /MAT/LAW79 (/MAT/JOHN_HOLM).

    Cites:
    - starter/source/materials/mat/mat079/hm_read_mat79.F lines 95-249.
    """
    params_obj = _get_params(mat_def)

    # Validation checks (hm_read_mat79.F lines 164-198)
    if params_obj.phel > params_obj.hel and params_obj.hel > 0.0:
        raise ValueError(
            f"LAW79: Pressure at HEL (PHEL={params_obj.phel}) cannot exceed HEL ({params_obj.hel})"
        )
    if params_obj.shear <= 0.0:
        raise ValueError(f"LAW79: Shear modulus must be positive (got {params_obj.shear})")
    if params_obj.k1 <= 0.0:
        raise ValueError(f"LAW79: Bulk modulus K1 must be positive (got {params_obj.k1})")
    if params_obj.eps0 <= 0.0:
        raise ValueError(f"LAW79: Reference strain rate eps0 must be positive (got {params_obj.eps0})")
    if not (0.0 <= params_obj.beta <= 1.0):
        raise ValueError(f"LAW79: Bulking coefficient beta must be in [0, 1] (got {params_obj.beta})")

    # Construct material params dict
    mat_id = getattr(mat_def, "id", 1) if not isinstance(mat_def, dict) else mat_def.get("id", 1)
    title = params_obj.title or (getattr(mat_def, "title", "LAW79") if not isinstance(mat_def, dict) else mat_def.get("title", "LAW79"))

    p: Dict[str, Any] = {
        "rho0": params_obj.rho0,
        "refer_rho": params_obj.refer_rho,
        "Refer_Rho": params_obj.refer_rho,
        "MAT_RHO": params_obj.rho0,
        "shear": params_obj.shear,
        "G": params_obj.shear,
        "tau_shear": params_obj.shear,
        "a": params_obj.a,
        "MAT_A": params_obj.a,
        "b": params_obj.b,
        "MAT_B": params_obj.b,
        "m": params_obj.m,
        "MAT_M": params_obj.m,
        "n": params_obj.n,
        "MAT_N": params_obj.n,
        "c": params_obj.c,
        "MAT_C": params_obj.c,
        "eps0": params_obj.eps0,
        "MAT_Epsilon_F": params_obj.eps0,
        "sigfmax": params_obj.sigfmax,
        "MAT_SIG1max_t": params_obj.sigfmax,
        "fcut": params_obj.fcut,
        "MAT_FCUT": params_obj.fcut,
        "t": params_obj.t,
        "t0": params_obj.t,
        "MAT_T0": params_obj.t,
        "hel": params_obj.hel,
        "MAT_E": params_obj.hel,
        "phel": params_obj.phel,
        "MAT_EPS": params_obj.phel,
        "d1": params_obj.d1,
        "D1": params_obj.d1,
        "d2": params_obj.d2,
        "D2": params_obj.d2,
        "idel": params_obj.idel,
        "IDEL": params_obj.idel,
        "epsmax": params_obj.epsmax,
        "EPSMAX": params_obj.epsmax,
        "k1": params_obj.k1,
        "K1": params_obj.k1,
        "k2": params_obj.k2,
        "K2": params_obj.k2,
        "k3": params_obj.k3,
        "K3": params_obj.k3,
        "beta": params_obj.beta,
        "MAT_Beta": params_obj.beta,
        # Derived
        "shel": params_obj.shel,
        "tstar": params_obj.tstar,
        "E": params_obj.young,
        "nu": params_obj.nu,
        "law79_params": params_obj,
    }

    mat = Material(
        id=mat_id,
        law=79,
        rho0=params_obj.rho0,
        title=title,
        law_name="LAW79",
        params=p,
    )
    setattr(mat, "law79_params", params_obj)
    return mat


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Persistent state arrays required by LAW79 solid elements."""
    return {
        "deltap": (nip,) if nip is not None else (),
        "sigy_old": (nip,) if nip is not None else (),
        "dmg": (nip,) if nip is not None else (),
        "off": (nip,) if nip is not None else (),
        "off79": (nip,) if nip is not None else (),
        "mu": (nip,) if nip is not None else (),
        "amu": (nip,) if nip is not None else (),
        "uvar": (nip, 2) if nip is not None else (2,),
        "epsd_filtered": (nip,) if nip is not None else (),
    }


# -----------------------------------------------------------------------------
# Acoustic Sound Speed (sigeps79.F lines 287-292)
# -----------------------------------------------------------------------------

def sound_speed_solid_law79(
    mat: Union[Material, Law79Params, dict],
    rho: Optional[Union[float, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[float, np.ndarray]:
    """Longitudinal acoustic wave speed for LAW79 solids.

    Cites sigeps79.F lines 287-292:
    dpdmu = K1 + 2*K2*mu + 3*K3*mu^2 (if mu > 0) else K1
    c = sqrt((dpdmu + 4/3*G) / rho0).
    """
    p = _get_params(mat)
    k1 = p.k1
    k2 = p.k2
    k3 = p.k3
    g = p.shear
    rho0 = p.refer_rho if rho is None else rho

    mu = 0.0
    if extra is not None:
        if "rho" in extra and extra["rho"] is not None:
            rho_arr = np.asarray(extra["rho"], dtype=float)
            mu = rho_arr / rho0 - 1.0
        elif "vol" in extra and "vol0" in extra and extra["vol"] is not None and extra["vol0"] is not None:
            v = np.asarray(extra["vol"], dtype=float)
            v0 = np.asarray(extra["vol0"], dtype=float)
            mu = v0 / np.maximum(v, 1e-30) - 1.0
        elif "mu" in extra and extra["mu"] is not None:
            mu = extra["mu"]
        elif "amu" in extra and extra["amu"] is not None:
            mu = extra["amu"]

    mu_arr = np.asarray(mu, dtype=float)
    dpdmu = np.where(mu_arr > 0.0, k1 + 2.0 * k2 * mu_arr + 3.0 * k3 * (mu_arr ** 2), k1)
    g43 = (4.0 / 3.0) * g

    c_sq = (dpdmu + g43) / np.maximum(rho0, _EM20)
    c = np.sqrt(np.maximum(c_sq, 0.0))
    if np.ndim(c) == 0:
        return float(c)
    return c


sound_speed_solid = sound_speed_solid_law79
sound_speed = sound_speed_solid_law79


# -----------------------------------------------------------------------------
# Constitutive stress update (solid only, sigeps79.F)
# -----------------------------------------------------------------------------

def solid_update_law79(
    mat: Union[Material, Law79Params, dict],
    sig: np.ndarray,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, Union[float, np.ndarray], Union[float, np.ndarray]]]:
    """Vectorized 3D solid constitutive update for LAW79 (Johnson-Holmquist JH-2).

    Ports exact physics from ``engine/source/materials/mat/mat079/sigeps79.F``.

    Parameters
    ----------
    mat : Material, Law79Params, or dict
        Material definition.
    sig : (6,) or (n, 6) ndarray
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx] (Voigt).
    deps : (6,) or (n, 6) ndarray, optional
        Strain increment tensor (engineering shear).
    epsp : float or (n,) ndarray, optional
        Current equivalent plastic strain.
    dt : float, default 0.0
        Time step increment.
    extra : dict, optional
        State variables ('mu', 'amu', 'deltap', 'dmg', 'off', 'epsd', 'uvar', etc.).
    return_tuple : bool, default False
        If True, returns (sig_new, epsp_new, sound_speed).

    Returns
    -------
    sig_new : (6,) or (n, 6) ndarray
        Updated Cauchy stress tensor.
    """
    # 0. Handle argument aliases & keyword fallbacks
    if deps is None:
        if "d_eps" in kwargs:
            deps = kwargs["d_eps"]
        elif "eps_dot" in kwargs:
            deps = kwargs["eps_dot"] * dt
        elif len(args) >= 1 and isinstance(args[0], np.ndarray):
            deps = args[0]
        else:
            deps = np.zeros_like(sig)

    if extra is None:
        if len(args) >= 2 and isinstance(args[1], dict):
            extra = args[1]
        elif "extra" in kwargs:
            extra = kwargs["extra"]
        else:
            extra = {}

    p_obj = _get_params(mat)
    g = p_obj.shear
    a = p_obj.a
    b = p_obj.b
    m_exp = p_obj.m
    n_exp = p_obj.n
    c_rate = p_obj.c
    eps0 = p_obj.eps0
    sigfmax = p_obj.sigfmax
    fcut = p_obj.fcut
    tstar = p_obj.tstar
    phel = p_obj.phel
    shel = p_obj.shel
    d1 = p_obj.d1
    d2 = p_obj.d2
    idel = p_obj.idel
    epsmax = p_obj.epsmax
    k1 = p_obj.k1
    k2 = p_obj.k2
    k3 = p_obj.k3
    beta = p_obj.beta
    rho0 = p_obj.refer_rho

    # Dimensionality
    is_1d = (np.ndim(sig) == 1)
    sig_arr = np.atleast_2d(np.asarray(sig, dtype=float)).copy()
    deps_arr = np.atleast_2d(np.asarray(deps, dtype=float)).copy()
    n = sig_arr.shape[0]

    if n == 0:
        sig_ret = sig.copy()
        if return_tuple:
            return sig_ret, epsp, None
        return sig_ret

    # 1. Recover internal state variables (sigeps79.F lines 115-123)
    off = np.asarray(extra.get("off", np.ones(n, dtype=float)), dtype=float).copy()
    if off.shape != (n,):
        off = np.full(n, float(off.flat[0]) if off.size > 0 else 1.0, dtype=float)

    # Initialisation of computation on time step:
    # IF (OFF(I) < EM01) OFF(I) = ZERO
    # IF (OFF(I) < ONE)  OFF(I) = OFF(I)*FOUR_OVER_5
    off = np.where(off < 0.1, 0.0, off)
    off = np.where(off < 1.0, off * 0.8, off)

    # Bulking pressure deltap
    if "deltap" in extra and extra["deltap"] is not None:
        deltap = np.atleast_1d(np.asarray(extra["deltap"], dtype=float)).copy()
    elif "uvar" in extra and extra["uvar"] is not None:
        uvar_arr = np.asarray(extra["uvar"], dtype=float)
        deltap = np.array([uvar_arr[0]]) if uvar_arr.ndim == 1 else uvar_arr[:, 0].copy()
    else:
        deltap = np.zeros(n, dtype=float)
    if deltap.shape != (n,):
        deltap = np.full(n, float(deltap.flat[0]) if deltap.size > 0 else 0.0, dtype=float)

    # Previous step normalized yield stress sigyold
    if "sigy_old" in extra and extra["sigy_old"] is not None:
        sigy_old = np.atleast_1d(np.asarray(extra["sigy_old"], dtype=float)).copy()
    elif "uvar" in extra and extra["uvar"] is not None:
        uvar_arr = np.asarray(extra["uvar"], dtype=float)
        if uvar_arr.ndim == 1:
            sigy_old = np.array([uvar_arr[1] / shel]) if shel > 0.0 else np.array([a])
        else:
            sigy_old = (uvar_arr[:, 1] / shel).copy() if shel > 0.0 else np.full(n, a, dtype=float)
    else:
        sigy_old = np.full(n, a, dtype=float)
    if sigy_old.shape != (n,):
        sigy_old = np.full(n, float(sigy_old.flat[0]) if sigy_old.size > 0 else a, dtype=float)

    # Damage dmg
    if "dmg" in extra and extra["dmg"] is not None:
        dmg = np.atleast_1d(np.asarray(extra["dmg"], dtype=float)).copy()
    elif "damage" in extra and extra["damage"] is not None:
        dmg = np.atleast_1d(np.asarray(extra["damage"], dtype=float)).copy()
    else:
        dmg = np.zeros(n, dtype=float)
    if dmg.shape != (n,):
        dmg = np.full(n, float(dmg.flat[0]) if dmg.size > 0 else 0.0, dtype=float)
    if "dmg_old" in extra and extra["dmg_old"] is not None:
        dmg_old = np.atleast_1d(np.asarray(extra["dmg_old"], dtype=float)).copy()
        if dmg_old.shape != (n,):
            dmg_old = np.full(n, float(dmg_old.flat[0]) if dmg_old.size > 0 else 0.0, dtype=float)
    else:
        dmg_old = dmg.copy()

    # Volumetric strain mu = rho / rho0 - 1
    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]
    if "rho" in extra and extra["rho"] is not None:
        rho_arr = np.asarray(extra["rho"], dtype=float)
        mu = rho_arr / rho0 - 1.0
    elif "vol" in extra and "vol0" in extra and extra["vol"] is not None and extra["vol0"] is not None:
        v = np.asarray(extra["vol"], dtype=float)
        v0 = np.asarray(extra["vol0"], dtype=float)
        mu = v0 / np.maximum(v, 1e-30) - 1.0
    elif "mu" in extra and extra["mu"] is not None:
        mu = np.asarray(extra["mu"], dtype=float).copy()
    elif "amu" in extra and extra["amu"] is not None:
        mu = np.asarray(extra["amu"], dtype=float).copy()
    else:
        mu_prev = np.asarray(extra.get("mu_prev", extra.get("mu_total", 0.0)), dtype=float)
        if mu_prev.shape != (n,):
            mu_prev = np.full(n, float(mu_prev.flat[0]) if mu_prev.size > 0 else 0.0, dtype=float)
        mu = mu_prev - tr_deps

    if mu.shape != (n,):
        mu = np.full(n, float(mu.flat[0]) if mu.size > 0 else 0.0, dtype=float)
    mu2 = mu * mu

    # 2. Deviatoric elastic stresses and equivalent trial stress (sigeps79.F lines 128-140)
    dav = tr_deps / 3.0
    p_old = -(sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0

    s_tr = np.empty_like(sig_arr)
    s_tr[:, 0] = sig_arr[:, 0] + p_old + 2.0 * g * (deps_arr[:, 0] - dav)
    s_tr[:, 1] = sig_arr[:, 1] + p_old + 2.0 * g * (deps_arr[:, 1] - dav)
    s_tr[:, 2] = sig_arr[:, 2] + p_old + 2.0 * g * (deps_arr[:, 2] - dav)
    s_tr[:, 3] = sig_arr[:, 3] + g * deps_arr[:, 3]
    s_tr[:, 4] = sig_arr[:, 4] + g * deps_arr[:, 4]
    s_tr[:, 5] = sig_arr[:, 5] + g * deps_arr[:, 5]

    j2 = (
        0.5 * (s_tr[:, 0] ** 2 + s_tr[:, 1] ** 2 + s_tr[:, 2] ** 2)
        + s_tr[:, 3] ** 2
        + s_tr[:, 4] ** 2
        + s_tr[:, 5] ** 2
    )
    vm = np.sqrt(3.0 * np.maximum(0.0, j2))

    # 3. Computation of pressure (sigeps79.F lines 145-154)
    p_new = k1 * mu + deltap
    mu_pos = mu > 0.0
    p_new = np.where(mu_pos, p_new + k2 * mu2 + k3 * mu2 * mu, p_new)

    if idel != 1:
        pmin = -tstar * phel * (1.0 - dmg)
        p_new = np.where(~mu_pos, np.maximum(p_new, pmin), p_new)

    pstar = p_new / phel if phel > 0.0 else np.zeros(n, dtype=float)

    # 4. Computation of yield stress (sigeps79.F lines 159-186)
    if n_exp == 0.0:
        sigyi = np.full(n, a, dtype=float)
    else:
        pt = pstar + tstar
        sigyi = np.where(pt > 0.0, a * (np.maximum(0.0, pt) ** n_exp), 0.0)

    if m_exp == 0.0:
        sigyf = np.full(n, b, dtype=float)
    else:
        sigyf = np.where(pstar > 0.0, b * (np.maximum(0.0, pstar) ** m_exp), 0.0)

    # Strain rate epsd and enhancement ce
    if "epsd" in extra and extra["epsd"] is not None:
        epsd = np.asarray(extra["epsd"], dtype=float).copy()
    elif dt > 0.0:
        vm_deps = np.sqrt(
            2.0 / 3.0 * (
                (deps_arr[:, 0] - dav) ** 2
                + (deps_arr[:, 1] - dav) ** 2
                + (deps_arr[:, 2] - dav) ** 2
                + 0.5 * (deps_arr[:, 3] ** 2 + deps_arr[:, 4] ** 2 + deps_arr[:, 5] ** 2)
            )
        )
        epsd = vm_deps / dt
    else:
        epsd = np.zeros(n, dtype=float)

    if epsd.shape != (n,):
        epsd = np.full(n, float(epsd.flat[0]) if epsd.size > 0 else 0.0, dtype=float)

    if fcut > 0.0 and fcut < _EP20 and dt > 0.0:
        alpha_rate = min(1.0, 2.0 * math.pi * fcut * dt)
        epsd_prev = np.asarray(extra.get("epsd_filtered", epsd), dtype=float)
        epsd = alpha_rate * epsd + (1.0 - alpha_rate) * epsd_prev
        extra["epsd_filtered"] = epsd

    if c_rate == 0.0:
        ce = np.ones(n, dtype=float)
    else:
        ce = np.where(epsd <= eps0, 1.0, 1.0 + c_rate * np.log(np.maximum(epsd, eps0) / eps0))

    sigyi = ce * sigyi
    sigyf = np.minimum(ce * sigyf, sigfmax)
    sigy = (1.0 - dmg) * sigyi + dmg * sigyf

    # 5. Radial return (sigeps79.F lines 191-208)
    sigstar = vm / shel if shel > 0.0 else np.zeros(n, dtype=float)
    yield_actual = sigy * shel

    scale = np.ones(n, dtype=float)
    over_yield = (sigstar >= sigy) & (vm > 0.0)
    scale = np.where(over_yield, sigy / np.maximum(sigstar, _EM30), scale)
    scale = np.where((sigstar >= sigy) & (vm <= 0.0), 0.0, scale)

    s_new = s_tr.copy()
    active = (off == 1.0)
    for k in range(6):
        s_new[:, k] = np.where(active, scale * s_tr[:, k], s_new[:, k])

    # 6. Update plastic strain and damage (sigeps79.F lines 215-260)
    if d2 == 0.0:
        epfail = np.full(n, d1, dtype=float)
    else:
        pt = pstar + tstar
        epfail = np.where(pt >= 0.0, d1 * (np.maximum(0.0, pt) ** d2), 0.0)

    dpla = np.zeros(n, dtype=float)
    epsp_cur = np.zeros(n, dtype=float) if epsp is None else np.asarray(epsp, dtype=float).copy()
    if epsp_cur.shape != (n,):
        epsp_cur = np.full(n, float(epsp_cur.flat[0]) if epsp_cur.size > 0 else 0.0, dtype=float)

    for i in range(n):
        if active[i]:
            if epfail[i] > 0.0:
                dpla[i] = (1.0 - scale[i]) * vm[i] / (3.0 * math.sqrt(3.0) * g)
                epsp_cur[i] += dpla[i]
                dmg[i] = min(1.0, dmg[i] + dpla[i] / epfail[i])
            elif scale[i] < 1.0:
                dmg[i] = 1.0

            # Element deletion check
            if idel == 1:
                if (pstar[i] + tstar) < 0.0:
                    off[i] = 0.8
            elif idel == 2:
                if epsp_cur[i] > epsmax:
                    off[i] = 0.8
            elif idel == 3:
                if dmg[i] >= 1.0:
                    off[i] = 0.8

    # 7. Compute pressure increment (bulking, sigeps79.F lines 265-276)
    for i in range(n):
        if (dmg[i] > dmg_old[i]) and (mu[i] > 0.0) and (off[i] == 1.0):
            p1 = k1 * mu[i]
            yield_curr = (1.0 - dmg[i]) * sigyi[i] + dmg[i] * sigyf[i]
            deltau = (sigy_old[i] ** 2 - yield_curr ** 2) / (6.0 * g)
            if deltau > 0.0:
                deltau *= (shel ** 2)
                radicand = (deltap[i] + p1) ** 2 + 2.0 * beta * k1 * deltau
                deltap[i] = -p1 + math.sqrt(max(0.0, radicand))

    # 8. Update stress tensor and sound speed (sigeps79.F lines 281-294)
    sig_new = np.empty_like(sig_arr)
    sig_new[:, 0] = s_new[:, 0] - p_new
    sig_new[:, 1] = s_new[:, 1] - p_new
    sig_new[:, 2] = s_new[:, 2] - p_new
    sig_new[:, 3] = s_new[:, 3]
    sig_new[:, 4] = s_new[:, 4]
    sig_new[:, 5] = s_new[:, 5]

    for k in range(6):
        sig_new[:, k] = np.where(off <= 0.0, 0.0, sig_new[:, k])

    dpdmu = np.where(mu > 0.0, k1 + 2.0 * k2 * mu + 3.0 * k3 * mu2, k1)
    g43 = (4.0 / 3.0) * g
    c_sound = np.sqrt(np.maximum(0.0, (dpdmu + g43) / np.maximum(rho0, _EM20)))

    # Store state into extra
    uvar_all = np.column_stack([deltap, sigy * shel])
    extra["deltap"] = float(deltap[0]) if is_1d else deltap
    extra["sigy_old"] = float(sigy[0]) if is_1d else sigy
    extra["uvar"] = uvar_all[0] if is_1d else uvar_all
    extra["dmg"] = float(dmg[0]) if is_1d else dmg
    extra["off"] = float(off[0]) if is_1d else off
    extra["off79"] = float(off[0]) if is_1d else off
    extra["mu"] = float(mu[0]) if is_1d else mu
    extra["amu"] = float(mu[0]) if is_1d else mu
    extra["p"] = float(p_new[0]) if is_1d else p_new
    extra["pstar"] = float(pstar[0]) if is_1d else pstar
    extra["epsp"] = float(epsp_cur[0]) if is_1d else epsp_cur
    extra["dpla"] = float(dpla[0]) if is_1d else dpla
    extra["vm"] = float(vm[0]) if is_1d else vm
    extra["scale"] = float(scale[0]) if is_1d else scale
    extra["c_solid"] = float(c_sound[0]) if is_1d else c_sound
    extra["sound_speed"] = float(c_sound[0]) if is_1d else c_sound

    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = epsp_cur[0] if is_1d else epsp_cur
        except Exception:
            pass

    out_sig = sig_new[0] if is_1d else sig_new
    out_epsp = epsp_cur[0] if is_1d else epsp_cur
    out_c = float(c_sound[0]) if is_1d else c_sound

    if return_tuple:
        return out_sig, out_epsp, out_c

    return out_sig


solid_update = solid_update_law79


# -----------------------------------------------------------------------------
# Plane-stress Shell Update (Not supported for brittle solids)
# -----------------------------------------------------------------------------

def shell_update_law79(*args: Any, **kwargs: Any) -> Any:
    """Plane-stress shell update is not supported for LAW79."""
    raise NotImplementedError("LAW79 (Johnson-Holmquist) is implemented for 3D solid and SPH elements only.")


shell_update = shell_update_law79


# -----------------------------------------------------------------------------
# State Copy Helper
# -----------------------------------------------------------------------------

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy dictionary of state variables for LAW79."""
    if extra is None:
        return None
    res: Dict[str, Any] = {}
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

def tangent_law79_solid(
    mat: Union[Material, Law79Params, dict],
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    *args: Any,
    epsp: Optional[Union[float, np.ndarray]] = None,
    epsp_incr: Optional[Union[float, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
    symmetric: bool = False,
    h: float = 1.0e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent elastoplastic tangent stiffness matrix for LAW79 in Voigt notation.

    Returns (6, 6) or (n, 6, 6) tensor relating d(sigma) to d(epsilon).
    """
    # Disambiguate arguments
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

    # Sizing
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

    # Off / element deletion status
    off_arr = np.ones(nel, dtype=float)
    if extra is not None:
        for k in ("off", "off79"):
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

    # 1. Finite-difference numerical consistent tangent if deps is provided
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

                sp = solid_update_law79(mat, sig_arr.copy(), deps=deps_arr + ej, dt=dt_call, extra=ext_p, epsp=epsp)
                sm = solid_update_law79(mat, sig_arr.copy(), deps=deps_arr - ej, dt=dt_call, extra=ext_m, epsp=epsp)

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

    # 2. Analytical tangent
    p = _get_params(mat)
    G = p.shear
    K = p.k1

    ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    KeeT = K * np.outer(ee, ee)

    I_dev = np.diag([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5])
    I_dev[0, 1] = I_dev[0, 2] = I_dev[1, 0] = I_dev[1, 2] = I_dev[2, 0] = I_dev[2, 1] = -1.0 / 3.0

    D = np.zeros((nel, 6, 6), dtype=float)
    for i in range(nel):
        D[i] = KeeT + 2.0 * G * I_dev

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

            for i in range(nel):
                if not plastic[i] or G <= 0.0:
                    continue
                dep = epsp_incr_arr[i]
                s_i = s[i]
                snorm = math.sqrt(
                    s_i[0] ** 2 + s_i[1] ** 2 + s_i[2] ** 2
                    + 2.0 * (s_i[3] ** 2 + s_i[4] ** 2 + s_i[5] ** 2)
                )
                snorm = max(snorm, 1.0e-30)
                Nv = s_i / snorm
                q = math.sqrt(1.5) * snorm
                q_tr = q + 3.0 * G * dep

                a_coeff = 3.0 * G * dep / q_tr
                b_coeff = 6.0 * G * G * (dep / q_tr - 1.0 / max(3.0 * G, 1.0e-15))

                NN = np.outer(Nv, Nv)
                D[i] = KeeT + 2.0 * G * (1.0 - a_coeff) * I_dev + b_coeff * NN

    if np.any(deleted_mask):
        D[deleted_mask] = 0.0

    if symmetric:
        D = 0.5 * (D + np.swapaxes(D, -1, -2))

    return D[0] if single else D


consistent_solid_tangent = tangent_law79_solid
solid_tangent = tangent_law79_solid
solid_tangent_law79 = tangent_law79_solid
tangent = tangent_law79_solid


# -----------------------------------------------------------------------------
# Module registration with pyradioss material physics registry
# -----------------------------------------------------------------------------

def _register() -> None:
    """Register LAW79 with pyradioss material physics registry."""
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

        for key in (
            79,
            "79",
            "LAW79",
            "JOHN_HOLM",
            "JOHNSON_HOLMQUIST",
            "MAT_LAW79",
            "MAT_JOHN_HOLM",
            "LAW79_JOHN_HOLM",
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law79
    except Exception:
        pass


_register()
