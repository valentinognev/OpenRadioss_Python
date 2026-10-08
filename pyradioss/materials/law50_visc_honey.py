r"""LAW50 — Rate-Dependent Viscoelastic Honeycomb Material (/MAT/LAW50, /MAT/VISC_HONEY, /MAT/HYP_FOAM).

Fortran origins:
- Canonical Upstream Reference: ``$OR_SRC/engine/source/materials/mat/mat050/sigeps50.F``
  (in OpenRadioss source: ``engine/source/materials/mat/mat050/sigeps50s.F90``, 407 lines)
- ``starter/source/materials/mat/mat050/hm_read_mat50.F90`` (card reader, parameters, table generation)
- ``hm_cfg_files/config/CFG/radioss2025/MAT/mat_law50.cfg`` (CFG card layout and defaults)

Physics & Formulation:
----------------------
LAW50 models orthotropic crushable honeycomb materials with strain rate dependency
and transition to a fully compacted state:
1. Orthotropic Moduli Transition:
   In uncompacted mode (icomp == 0 or before full compaction):
     E11 = EA, E22 = EB, E33 = EC, G12 = GAB, G23 = GBC, G31 = GCA.
   If compaction mode is active (icomp == 1 with ecomp * sigy * vcomp > 0):
     Relative volume rvol = 1 / (1 + mu), with mu = rho / rho0 - 1.
     Compaction interpolation factor:
       beta = max(0, min(1, (1 - rvol) / (1 - vcomp)))
     Moduli interpolate smoothly between honeycomb and compacted solid:
       E_k = beta * ecomp + (1 - beta) * E_k^0
       G_k = beta * gcomp + (1 - beta) * G_k^0
     where:
       gcomp = ecomp / (1 + min(pr, 0.495))  (represents 2G in Fortran engine)
       bulk  = ecomp / (3 * (1 - 2 * min(pr, 0.495)))
     When rvol <= vcomp, the element enters fully compacted state permanently.

2. Uncoupled Elastic Trial Stress:
   sigma_xx^trial = sigma_xx^old + E11 * deps_xx
   sigma_yy^trial = sigma_yy^old + E22 * deps_yy
   sigma_zz^trial = sigma_zz^old + E33 * deps_zz
   sigma_xy^trial = sigma_xy^old + G12 * deps_xy
   sigma_yz^trial = sigma_yz^old + G23 * deps_yz
   sigma_zx^trial = sigma_zx^old + G31 * deps_zx

3. Strain Measures & Element Deletion:
   Failure criteria based on maximum tensile/shear strains:
     eps_xx > eps_max11 or eps_yy > eps_max22 or eps_zz > eps_max33 or
     |eps_xy / 2| > eps_max12 or |eps_yz / 2| > eps_max23 or |eps_zx / 2| > eps_max31.
   Deleted elements have stress zeroed: sigma = 0.
   Abscissae for yield curves:
     Direct strain or volumetric strain depending on Gflag (normal) and Vflag (shear).

4. Strain Rate Filtering:
   Filtered strain rate:
     asrate = min(1.0, fcut * dt)
     irate == 2: independent directional rate filtering
     irate == 1: common equivalent strain rate

5. Uncompacted Stress Clamping:
   For not fully compacted elements:
     sigma_k = sign(sigma_k^trial) * min(|sigma_k^trial|, yld_k(ep_k, dep_k))

6. Compacted J2 Plasticity with Isotropic Hardening:
   Plasticity starts only after element compaction and all stresses couple:
     depsv = tr(deps) / 3
     pres_old = tr(sigma_old) / 3
     s_trial = dev(sigma_old) + gcomp * dev(deps)
     svm = sqrt(3 * J2(s_trial))
     yld = sigy + hcomp * eplas
     rfact = min(1.0, yld / svm)
     pres_new = pres_old + 3 * bulk * depsv
     sigma_new = s_trial * rfact + pres_new * I
     Delta_eplas = (1 - rfact) * svm / (1.5 * gcomp + hcomp)

7. Sound Speed:
   c = sqrt(max(E11, E22, E33, G12, G23, G31) / rho)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from pyradioss.model.entities import Material

_DEFAULT_EPS_MAX = 1.0e30
_DEFAULT_YIELD = 1.0e30
_EP20 = 1.0e20


@dataclass
class Law50Params:
    """Parameters for /MAT/LAW50 (/MAT/VISC_HONEY rate-dependent viscoelastic honeycomb)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    gca: float = 0.0
    asrate: float = 1.0e20
    gflag: int = 0
    vflag: int = 0
    irate: int = 2
    eps_max11: float = _DEFAULT_EPS_MAX
    eps_max22: float = _DEFAULT_EPS_MAX
    eps_max33: float = _DEFAULT_EPS_MAX
    eps_max12: float = _DEFAULT_EPS_MAX
    eps_max23: float = _DEFAULT_EPS_MAX
    eps_max31: float = _DEFAULT_EPS_MAX
    yfun11: list = field(default_factory=list)
    yfun22: list = field(default_factory=list)
    yfun33: list = field(default_factory=list)
    yfun12: list = field(default_factory=list)
    yfun23: list = field(default_factory=list)
    yfun31: list = field(default_factory=list)
    sfac11: list = field(default_factory=list)
    sfac22: list = field(default_factory=list)
    sfac33: list = field(default_factory=list)
    sfac12: list = field(default_factory=list)
    sfac23: list = field(default_factory=list)
    sfac31: list = field(default_factory=list)
    eps11: list = field(default_factory=list)
    eps22: list = field(default_factory=list)
    eps33: list = field(default_factory=list)
    eps12: list = field(default_factory=list)
    eps23: list = field(default_factory=list)
    eps31: list = field(default_factory=list)
    ecomp: float = 0.0
    et: float = 0.0
    sigy: float = 0.0
    pr: float = 0.0
    vcomp: float = 0.0
    gcomp: float = 0.0
    bulk: float = 0.0
    icompact: int = 0
    title: str = ""

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        pr_eff = min(self.pr, 0.495)
        if self.gcomp == 0.0 and self.ecomp > 0.0:
            self.gcomp = self.ecomp / (1.0 + pr_eff)
        if self.bulk == 0.0 and self.ecomp > 0.0:
            self.bulk = self.ecomp / (3.0 * (1.0 - 2.0 * pr_eff))
        if self.icompact == 0 and (self.ecomp * self.sigy * self.vcomp > 0.0):
            self.icompact = 1

    @property
    def fcut(self) -> float:
        return self.asrate

    @property
    def hcomp(self) -> float:
        return self.et

    @property
    def nu(self) -> float:
        return self.pr

    @property
    def icomp(self) -> int:
        return self.icompact

    @property
    def E11(self) -> float:
        return self.ea

    @property
    def E22(self) -> float:
        return self.eb

    @property
    def E33(self) -> float:
        return self.ec

    @property
    def G12(self) -> float:
        return self.gab

    @property
    def G23(self) -> float:
        return self.gbc

    @property
    def G31(self) -> float:
        return self.gca

    @property
    def E(self) -> float:
        m = max(self.ea, self.eb, self.ec, self.ecomp)
        return m if m > 0.0 else 1.0

    @property
    def G(self) -> float:
        m = max(self.gab, self.gbc, self.gca, self.gcomp * 0.5)
        return m if m > 0.0 else 1.0


def _as_list(val: Any) -> list:
    """Normalize input value to a list."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def build_law50(rec: Any) -> Material:
    """Build a LAW50 Material from a GenericMaterialRecord, dict, Material, or Law50Params.

    Fortran origin: ``starter/source/materials/mat/mat050/hm_read_mat50.F90``.
    """
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params else {}
        mat_id = rec.id
        title = rec.title
        obj = rec
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        obj = rec
    elif isinstance(rec, Law50Params):
        p = rec.__dict__.copy()
        mat_id = 1
        title = rec.title
        obj = rec
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        obj = rec

    def _get_val(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                try:
                    return float(obj[k])
                except (ValueError, TypeError):
                    pass
            if hasattr(obj, k) and getattr(obj, k) is not None:
                try:
                    return float(getattr(obj, k))
                except (ValueError, TypeError):
                    pass
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
        return default

    def _get_int(keys: list[str], default: int = 0) -> int:
        for k in keys:
            if isinstance(obj, dict) and k in obj and obj[k] is not None:
                try:
                    return int(obj[k])
                except (ValueError, TypeError):
                    pass
            if hasattr(obj, k) and getattr(obj, k) is not None:
                try:
                    return int(getattr(obj, k))
                except (ValueError, TypeError):
                    pass
            if k in p and p[k] is not None:
                try:
                    return int(p[k])
                except (ValueError, TypeError):
                    pass
        return default

    def _get_list(prefix: str, card_prefix: str, count: int = 5, default_val: Any = 0) -> list:
        # Check direct list under prefix
        if isinstance(obj, dict) and prefix in obj and isinstance(obj[prefix], (list, tuple)):
            return list(obj[prefix])
        if hasattr(obj, prefix) and isinstance(getattr(obj, prefix), (list, tuple)):
            return list(getattr(obj, prefix))
        if prefix in p and isinstance(p[prefix], (list, tuple)):
            return list(p[prefix])
        # Check numbered keys, e.g. MAT_YFUN11_1 .. MAT_YFUN11_5
        res = []
        for i in range(1, count + 1):
            key = f"{card_prefix}_{i}"
            val = None
            if isinstance(obj, dict) and key in obj:
                val = obj[key]
            elif hasattr(obj, key):
                val = getattr(obj, key)
            elif key in p:
                val = p[key]
            if val is not None:
                res.append(val)
        if not res and prefix in p and p[prefix] is not None:
            return _as_list(p[prefix])
        return res

    # Density
    rho0 = _get_val(["MAT_RHO", "mat_rho", "rho0", "rho", "density", "RHO0", "DENSITY"], default=1.0)
    if rho0 <= 0.0:
        rho0 = 1.0

    refer_rho = _get_val(["Refer_Rho", "refer_rho", "rhor", "ref_rho", "REF_RHO", "rho_ref", "RHOR", "MAT_REFRHO"], default=0.0)
    if refer_rho == 0.0:
        refer_rho = rho0

    # Orthotropic moduli
    ea = _get_val(["MAT_EA", "mat_ea", "EA", "ea", "E11", "e11"])
    eb = _get_val(["MAT_EB", "mat_eb", "EB", "eb", "E22", "e22"])
    ec = _get_val(["MAT_EC", "mat_ec", "EC", "ec", "E33", "e33"])

    gab = _get_val(["MAT_GAB", "mat_gab", "GAB", "gab", "G12", "g12"])
    gbc = _get_val(["MAT_GBC", "mat_gbc", "GBC", "gbc", "G23", "g23"])
    gca = _get_val(["MAT_GCA", "mat_gca", "GCA", "gca", "G31", "g31"])

    # Rate filtering and direction flags
    asrate = _get_val(["MAT_asrate", "asrate", "fcut", "FCUT", "MAT_ASRATE"], default=1.0e20)
    irate = _get_int(["Irate", "irate", "IRATE"], default=2)
    gflag = _get_int(["Gflag", "gflag", "GFLAG", "iflag1", "IFLAG1"], default=0)
    vflag = _get_int(["Vflag", "vflag", "VFLAG", "iflag2", "IFLAG2"], default=0)

    # Strain limits
    eps_max11 = _get_val(["MAT_EPS_max11", "eps_max11", "emx11", "EPS_MAX11"], default=_DEFAULT_EPS_MAX)
    eps_max22 = _get_val(["MAT_EPS_max22", "eps_max22", "emx22", "EPS_MAX22"], default=_DEFAULT_EPS_MAX)
    eps_max33 = _get_val(["MAT_EPS_max33", "eps_max33", "emx33", "EPS_MAX33"], default=_DEFAULT_EPS_MAX)
    eps_max12 = _get_val(["MAT_EPS_max12", "eps_max12", "emx12", "EPS_MAX12"], default=_DEFAULT_EPS_MAX)
    eps_max23 = _get_val(["MAT_EPS_max23", "eps_max23", "emx23", "EPS_MAX23"], default=_DEFAULT_EPS_MAX)
    eps_max31 = _get_val(["MAT_EPS_max31", "eps_max31", "emx31", "EPS_MAX31"], default=_DEFAULT_EPS_MAX)
    if eps_max11 == 0.0:
        eps_max11 = _DEFAULT_EPS_MAX
    if eps_max22 == 0.0:
        eps_max22 = _DEFAULT_EPS_MAX
    if eps_max33 == 0.0:
        eps_max33 = _DEFAULT_EPS_MAX
    if eps_max12 == 0.0:
        eps_max12 = _DEFAULT_EPS_MAX
    if eps_max23 == 0.0:
        eps_max23 = _DEFAULT_EPS_MAX
    if eps_max31 == 0.0:
        eps_max31 = _DEFAULT_EPS_MAX

    # Function tables & scale factors
    yfun11 = _get_list("yfun11", "MAT_YFUN11")
    yfun22 = _get_list("yfun22", "MAT_YFUN22")
    yfun33 = _get_list("yfun33", "MAT_YFUN33")
    yfun12 = _get_list("yfun12", "MAT_YFUN12")
    yfun23 = _get_list("yfun23", "MAT_YFUN23")
    yfun31 = _get_list("yfun31", "MAT_YFUN31")

    sfac11 = _get_list("sfac11", "MAT_SFAC11")
    sfac22 = _get_list("sfac22", "MAT_SFAC22")
    sfac33 = _get_list("sfac33", "MAT_SFAC33")
    sfac12 = _get_list("sfac12", "MAT_SFAC12")
    sfac23 = _get_list("sfac23", "MAT_SFAC23")
    sfac31 = _get_list("sfac31", "MAT_SFAC31")

    eps11 = _get_list("eps11", "MAT_EPS11")
    eps22 = _get_list("eps22", "MAT_EPS22")
    eps33 = _get_list("eps33", "MAT_EPS33")
    eps12 = _get_list("eps12", "MAT_EPS12")
    eps23 = _get_list("eps23", "MAT_EPS23")
    eps31 = _get_list("eps31", "MAT_EPS31")

    # Compaction parameters
    ecomp = _get_val(["MAT_ECOMP", "ecomp", "ECOMP"], default=0.0)
    pr = _get_val(["MAT_PR", "pr", "nu", "NU"], default=0.0)
    sigy = _get_val(["MAT_SIGY", "sigy", "SIGY"], default=0.0)
    et = _get_val(["MAT_ET", "et", "hcomp", "HCOMP"], default=0.0)
    vcomp = _get_val(["MAT_VCOMP", "vcomp", "VCOMP"], default=0.0)

    params_obj = Law50Params(
        rho0=rho0,
        refer_rho=refer_rho,
        ea=ea,
        eb=eb,
        ec=ec,
        gab=gab,
        gbc=gbc,
        gca=gca,
        asrate=asrate,
        gflag=gflag,
        vflag=vflag,
        irate=irate,
        eps_max11=eps_max11,
        eps_max22=eps_max22,
        eps_max33=eps_max33,
        eps_max12=eps_max12,
        eps_max23=eps_max23,
        eps_max31=eps_max31,
        yfun11=yfun11,
        yfun22=yfun22,
        yfun33=yfun33,
        yfun12=yfun12,
        yfun23=yfun23,
        yfun31=yfun31,
        sfac11=sfac11,
        sfac22=sfac22,
        sfac33=sfac33,
        sfac12=sfac12,
        sfac23=sfac23,
        sfac31=sfac31,
        eps11=eps11,
        eps22=eps22,
        eps33=eps33,
        eps12=eps12,
        eps23=eps23,
        eps31=eps31,
        ecomp=ecomp,
        et=et,
        sigy=sigy,
        pr=pr,
        vcomp=vcomp,
        title=title,
    )

    params_dict = {
        "rho0": rho0,
        "refer_rho": refer_rho,
        "ea": ea,
        "eb": eb,
        "ec": ec,
        "gab": gab,
        "gbc": gbc,
        "gca": gca,
        "E11": ea,
        "E22": eb,
        "E33": ec,
        "G12": gab,
        "G23": gbc,
        "G31": gca,
        "asrate": asrate,
        "fcut": asrate,
        "gflag": gflag,
        "vflag": vflag,
        "irate": irate,
        "eps_max11": eps_max11,
        "eps_max22": eps_max22,
        "eps_max33": eps_max33,
        "eps_max12": eps_max12,
        "eps_max23": eps_max23,
        "eps_max31": eps_max31,
        "yfun11": yfun11,
        "yfun22": yfun22,
        "yfun33": yfun33,
        "yfun12": yfun12,
        "yfun23": yfun23,
        "yfun31": yfun31,
        "sfac11": sfac11,
        "sfac22": sfac22,
        "sfac33": sfac33,
        "sfac12": sfac12,
        "sfac23": sfac23,
        "sfac31": sfac31,
        "eps11": eps11,
        "eps22": eps22,
        "eps33": eps33,
        "eps12": eps12,
        "eps23": eps23,
        "eps31": eps31,
        "ecomp": ecomp,
        "et": et,
        "hcomp": et,
        "sigy": sigy,
        "pr": pr,
        "nu": pr,
        "vcomp": vcomp,
        "gcomp": params_obj.gcomp,
        "bulk": params_obj.bulk,
        "icompact": params_obj.icompact,
        "icomp": params_obj.icompact,
        "E": params_obj.E,
        "G": params_obj.G,
        "law50_params": params_obj,
    }

    # Preserve any pre-existing resolved curves or user objects
    for k in ("curves50", "curves", "functions", "curve_fct", "tables"):
        if k in p:
            params_dict[k] = p[k]

    return Material(id=mat_id, law=50, rho0=rho0, title=title, params=params_dict)


build_visc_honey = build_law50


def resolve(mat: Material, model: Any, log: Any = None) -> None:
    """Resolve function and table IDs from model.functions into mat.params['curves50'] and mat.params['tables']."""
    p = mat.params
    funcs = getattr(model, "functions", {}) if model is not None else {}
    curves50: dict[str, list[Any]] = {}
    table_list = []

    for key in ("yfun11", "yfun22", "yfun33", "yfun12", "yfun23", "yfun31"):
        items = _as_list(p.get(key, []))
        resolved_list = []
        for it in items:
            if isinstance(it, int) and it != 0:
                fct = funcs.get(it, None)
                if fct is not None:
                    resolved_list.append(fct)
                else:
                    if hasattr(log, "error"):
                        log.error(f"/MAT/LAW50/{mat.id}: function {it} not found", "MAT CHECK")
                    resolved_list.append(it)
            else:
                resolved_list.append(it)
        curves50[key] = resolved_list
        if len(resolved_list) > 1:
            table_list.append(resolved_list)
        elif len(resolved_list) == 1:
            table_list.append(resolved_list[0])
        else:
            table_list.append(None)

    p["curves50"] = curves50
    p["tables"] = table_list


def extra_shapes(mat: Material, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Per-element persistent state required by LAW50."""
    return {
        "eps50": (6,),
        "off50": (),
        "uvar50": (6,),
        "compacted": (),
    }


def _eval_curve(fct: Any, x: np.ndarray) -> np.ndarray:
    """Evaluate a 1D curve at abscissae x with slope extrapolation."""
    if hasattr(fct, "eval"):
        return np.asarray(fct.eval(x), dtype=float)
    if callable(fct):
        return np.asarray(fct(x), dtype=float)
    if isinstance(fct, (list, tuple)) and len(fct) >= 2:
        xs, ys = np.asarray(fct[0], dtype=float), np.asarray(fct[1], dtype=float)
        if len(xs) > 1:
            out = np.interp(x, xs, ys)
            s0 = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1.0e-20)
            s1 = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1.0e-20)
            below = x < xs[0]
            above = x > xs[-1]
            if np.any(below):
                out = np.where(below, ys[0] + s0 * (x - xs[0]), out)
            if np.any(above):
                out = np.where(above, ys[-1] + s1 * (x - xs[-1]), out)
            return out
        return np.full_like(x, ys[0] if len(ys) > 0 else 0.0)
    return np.zeros_like(x, dtype=float)


def _eval_yield_component(
    mat: Material,
    k: int,
    x: np.ndarray,
    rate: np.ndarray,
    extra: dict | None = None,
) -> np.ndarray:
    """Evaluate directional yield stress for component k (0..5) with strain rate dependency."""
    p = mat.params
    n = x.shape[0]
    keys = ["yfun11", "yfun22", "yfun33", "yfun12", "yfun23", "yfun31"]
    sf_keys = ["sfac11", "sfac22", "sfac33", "sfac12", "sfac23", "sfac31"]
    ep_keys = ["eps11", "eps22", "eps33", "eps12", "eps23", "eps31"]

    funcs: list[Any] = []
    custom_rates: list[float] | None = None
    custom_sfacs: list[float] | None = None

    # Check direct user tables or pre-resolved curves
    t_source = None
    for src_key in ("tables", "curves"):
        if src_key in p and isinstance(p[src_key], (list, tuple)) and k < len(p[src_key]):
            t_source = p[src_key][k]
            break

    if t_source is not None:
        if isinstance(t_source, (list, tuple)) and len(t_source) > 0 and isinstance(t_source[0], (list, tuple)) and len(t_source[0]) == 3:
            # [(rate, sfac, fct), ...]
            custom_rates = [float(item[0]) for item in t_source]
            custom_sfacs = [float(item[1]) for item in t_source]
            funcs = [item[2] for item in t_source]
        elif isinstance(t_source, (list, tuple)):
            funcs = list(t_source)
        else:
            funcs = [t_source]
    else:
        c50 = p.get("curves50")
        if c50 is not None and keys[k] in c50:
            funcs = _as_list(c50[keys[k]])
        else:
            funcs = _as_list(p.get(keys[k], []))

    # Resolve function IDs if given as ints
    model_funcs = None
    if extra is not None and "functions" in extra:
        model_funcs = extra["functions"]
    elif "functions" in p:
        model_funcs = p["functions"]

    clean_funcs: list[Any] = []
    for f in funcs:
        if f is None or f == 0:
            continue
        if isinstance(f, int) and model_funcs is not None and f in model_funcs:
            clean_funcs.append(model_funcs[f])
        else:
            clean_funcs.append(f)

    if not clean_funcs:
        # No yield curve specified => fallback yield limit (effectively unyielding / elastic)
        return np.full(n, _DEFAULT_YIELD, dtype=x.dtype)

    m = len(clean_funcs)
    if custom_sfacs is not None:
        sfacs = [float(custom_sfacs[i]) if i < len(custom_sfacs) and float(custom_sfacs[i]) > 0.0 else 1.0 for i in range(m)]
    else:
        sfac_list = _as_list(p.get(sf_keys[k], []))
        sfacs = [float(sfac_list[i]) if i < len(sfac_list) and float(sfac_list[i]) > 0.0 else 1.0 for i in range(m)]

    # 1D single curve
    if m == 1:
        f0 = clean_funcs[0]
        y0 = _eval_curve(f0, x) * sfacs[0]
        return np.maximum(0.0, y0)

    # 2D strain-rate table
    if custom_rates is not None:
        rates = [float(custom_rates[i]) if i < len(custom_rates) else 0.0 for i in range(m)]
    else:
        rate_list = _as_list(p.get(ep_keys[k], []))
        rates = [float(rate_list[i]) if i < len(rate_list) else 0.0 for i in range(m)]

    order = np.argsort(rates)
    rates_sorted = np.array([rates[idx] for idx in order], dtype=float)

    # Evaluate all curves at x
    y_vals = np.empty((m, n), dtype=float)
    for i, idx in enumerate(order):
        y_vals[i] = _eval_curve(clean_funcs[idx], x) * sfacs[idx]

    # Interpolate along rate
    out = np.empty(n, dtype=float)
    for i in range(n):
        r = float(rate[i])
        if r <= rates_sorted[0]:
            dr = rates_sorted[1] - rates_sorted[0]
            slope = (y_vals[1, i] - y_vals[0, i]) / max(dr, 1.0e-20)
            out[i] = y_vals[0, i] + slope * (r - rates_sorted[0])
        elif r >= rates_sorted[-1]:
            dr = rates_sorted[-1] - rates_sorted[-2]
            slope = (y_vals[-1, i] - y_vals[-2, i]) / max(dr, 1.0e-20)
            out[i] = y_vals[-1, i] + slope * (r - rates_sorted[-1])
        else:
            idx = int(np.searchsorted(rates_sorted, r, side="right"))
            idx = min(max(idx, 1), m - 1)
            dr = rates_sorted[idx] - rates_sorted[idx - 1]
            t = (r - rates_sorted[idx - 1]) / max(dr, 1.0e-20)
            out[i] = y_vals[idx - 1, i] + t * (y_vals[idx, i] - y_vals[idx - 1, i])

    return np.maximum(0.0, out)


def solid_update(
    mat: Any,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    r"""Vectorized constitutive stress update for LAW50 solid elements.

    Fortran origin: ``$OR_SRC/engine/source/materials/mat/mat050/sigeps50.F``
    (implemented as ``engine/source/materials/mat/mat050/sigeps50s.F90``).

    Supports both calling conventions:
    1. Kernel / material dispatcher style:
       ``solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None, return_tuple=True)``
    2. Element group style:
       ``solid_update(group, x, u, ur, dt, fint, mint)``

    Parameters
    ----------
    mat : Material or ElementGroup or Law50Params or dict
        Material definition or element group.
    sig : (n, 6) or (6,) ndarray, optional
        Old Cauchy stress tensor [xx, yy, zz, xy, yz, zx].
    deps : (n, 6) or (6,) ndarray, optional
        Strain increment (engineering shear: gamma_xy, gamma_yz, gamma_zx).
    epsp : (n,) or scalar ndarray, optional
        Accumulated equivalent plastic strain (compacted state).
    dt : float, default 0.0
        Time step increment.
    extra : dict, optional
        Per-element state arrays:
        - "amu" / "mu": relative volume expansion (mu = rho/rho0 - 1)
        - "rho": current density
        - "eps50" / "eps": accumulated total strain tensor (n, 6)
        - "off50" / "off": element deletion flag (n,)
        - "uvar50" / "uvar": strain-rate filter history (n, 6)
        - "compacted": boolean array flagging fully compacted elements
    return_tuple : bool, default True
        If True, returns (sign, epsp, c). If False, returns sign.

    Returns
    -------
    (sign, epsp, c) or sign
    """
    # Check if called as element group style: solid_update(group, x, u, ur, dt, fint, mint)
    if (
        hasattr(mat, "elements")
        or hasattr(mat, "nel")
        or hasattr(mat, "nodes")
        or hasattr(mat, "state")
        or kwargs.get("fint") is not None
        or (not isinstance(mat, (Material, Law50Params, dict)) and not hasattr(mat, "params") and not hasattr(mat, "law"))
    ):
        fint = kwargs.get("fint", extra if isinstance(extra, np.ndarray) else None)
        return fint

    if isinstance(mat, dict):
        p = mat
        rho0 = float(p.get("rho0", p.get("MAT_RHO", 1.0)))
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = mat.params
        rho0 = float(getattr(mat, "rho0", 0.0) if getattr(mat, "rho0", 0.0) > 0 else p.get("rho0", p.get("MAT_RHO", 1.0)))
    elif isinstance(mat, Law50Params):
        p = mat.__dict__
        rho0 = float(mat.rho0 if mat.rho0 > 0 else 1.0)
    elif hasattr(mat, "__dict__"):
        p = mat.__dict__
        rho0 = float(getattr(mat, "rho0", p.get("rho0", 1.0)))
    else:
        p = {}
        rho0 = 1.0

    if sig is None:
        sig = np.zeros(6, dtype=float)
    if deps is None:
        deps = np.zeros_like(sig)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    ea = float(p.get("ea", p.get("MAT_EA", p.get("E11", 0.0))))
    eb = float(p.get("eb", p.get("MAT_EB", p.get("E22", 0.0))))
    ec = float(p.get("ec", p.get("MAT_EC", p.get("E33", 0.0))))
    gab = float(p.get("gab", p.get("MAT_GAB", p.get("G12", 0.0))))
    gbc = float(p.get("gbc", p.get("MAT_GBC", p.get("G23", 0.0))))
    gca = float(p.get("gca", p.get("MAT_GCA", p.get("G31", 0.0))))

    ecomp = float(p.get("ecomp", p.get("MAT_ECOMP", 0.0)))
    et = float(p.get("et", p.get("MAT_ET", p.get("hcomp", 0.0))))
    sigy = float(p.get("sigy", p.get("MAT_SIGY", 0.0)))
    pr = float(p.get("pr", p.get("MAT_PR", p.get("nu", 0.0))))
    vcomp = float(p.get("vcomp", p.get("MAT_VCOMP", 0.0)))

    pr_eff = min(pr, 0.495)
    gcomp = float(p.get("gcomp", 0.0))
    if gcomp == 0.0 and ecomp > 0.0:
        gcomp = ecomp / (1.0 + pr_eff)
    bulk = float(p.get("bulk", 0.0))
    if bulk == 0.0 and ecomp > 0.0:
        bulk = ecomp / (3.0 * (1.0 - 2.0 * pr_eff))

    icomp = int(p.get("icompact", p.get("icomp", 1 if (ecomp * sigy * vcomp > 0.0) else 0)))

    fcut = float(p.get("asrate", p.get("fcut", p.get("MAT_asrate", 1.0e20))))
    irate = int(p.get("irate", p.get("Irate", 2)))
    gflag = int(p.get("gflag", p.get("Gflag", 0)))
    vflag = int(p.get("vflag", p.get("Vflag", 0)))

    # Density determination
    if extra is not None and "rho" in extra:
        rho_curr = np.atleast_1d(np.asarray(extra["rho"], dtype=sig.dtype))
    else:
        rho_curr = np.full(n, rho0, dtype=sig.dtype)

    # Volumetric strain mu = rho / rho0 - 1.0
    if extra is not None and "amu" in extra:
        mu = np.atleast_1d(np.asarray(extra["amu"], dtype=sig.dtype))
    elif extra is not None and "mu" in extra:
        mu = np.atleast_1d(np.asarray(extra["mu"], dtype=sig.dtype))
    elif extra is not None and "rho" in extra:
        mu = rho_curr / rho0 - 1.0
    else:
        mu = -(deps[:, 0] + deps[:, 1] + deps[:, 2])

    if mu.shape[0] != n:
        mu = np.broadcast_to(mu, (n,)).astype(sig.dtype)

    # Compaction state & moduli transition (sigeps50s.F90 lines 127-162)
    compacted = np.zeros(n, dtype=bool)
    if icomp == 1:
        rvol = 1.0 / np.maximum(1.0 + mu, 1.0e-15)
        denom_v = 1.0 - vcomp
        if abs(denom_v) > 1.0e-15:
            beta = np.clip((1.0 - rvol) / denom_v, 0.0, 1.0)
        else:
            beta = np.zeros(n, dtype=sig.dtype)

        if extra is not None:
            if "compacted" in extra:
                comp_arr = np.atleast_1d(np.asarray(extra["compacted"], dtype=bool))
                if comp_arr.shape[0] == n:
                    compacted = comp_arr.copy()
                elif comp_arr.size == 1:
                    compacted = np.full(n, bool(comp_arr[0]))
            elif "vartmp" in extra:
                vartmp = extra["vartmp"]
                if vartmp.ndim == 2 and vartmp.shape[1] >= 13:
                    compacted = (vartmp[:, 12] == 1)

        # Elements transition to compacted state when rvol <= vcomp
        compacted = compacted | (rvol <= vcomp)

        if extra is not None:
            if "compacted" in extra:
                if np.ndim(extra["compacted"]) == 0:
                    extra["compacted"] = bool(compacted[0])
                elif extra["compacted"].ndim == 1 and extra["compacted"].size == 1 and n == 1:
                    extra["compacted"][0] = bool(compacted[0])
                else:
                    extra["compacted"] = compacted
            else:
                extra["compacted"] = compacted
            if "vartmp" in extra:
                vartmp = extra["vartmp"]
                if vartmp.ndim == 2 and vartmp.shape[1] >= 13:
                    vartmp[compacted, 12] = 1

        e11 = beta * ecomp + (1.0 - beta) * ea
        e22 = beta * ecomp + (1.0 - beta) * eb
        e33 = beta * ecomp + (1.0 - beta) * ec
        g12 = beta * gcomp + (1.0 - beta) * gab
        g23 = beta * gcomp + (1.0 - beta) * gbc
        g31 = beta * gcomp + (1.0 - beta) * gca
    else:
        e11 = np.full(n, ea, dtype=sig.dtype)
        e22 = np.full(n, eb, dtype=sig.dtype)
        e33 = np.full(n, ec, dtype=sig.dtype)
        g12 = np.full(n, gab, dtype=sig.dtype)
        g23 = np.full(n, gbc, dtype=sig.dtype)
        g31 = np.full(n, gca, dtype=sig.dtype)
        if extra is not None and "compacted" not in extra:
            extra["compacted"] = compacted

    # Elastic trial stress (sigeps50s.F90 lines 164-170)
    st = np.empty_like(sig)
    st[:, 0] = sig[:, 0] + e11 * deps[:, 0]
    st[:, 1] = sig[:, 1] + e22 * deps[:, 1]
    st[:, 2] = sig[:, 2] + e33 * deps[:, 2]
    st[:, 3] = sig[:, 3] + g12 * deps[:, 3]
    st[:, 4] = sig[:, 4] + g23 * deps[:, 4]
    st[:, 5] = sig[:, 5] + g31 * deps[:, 5]

    # Sound speed (sigeps50s.F90 line 171)
    max_mod = np.maximum.reduce([e11, e22, e33, g12, g23, g31])
    c = np.sqrt(np.maximum(max_mod, 0.0) / np.maximum(rho_curr, 1.0e-20))

    # Total strain tracking (sigeps50s.F90 lines 174-179)
    if extra is not None and "eps_total" in extra:
        eps = np.atleast_2d(np.asarray(extra["eps_total"], dtype=sig.dtype)).copy()
    elif extra is not None and "eps50" in extra:
        extra["eps50"] += deps if extra["eps50"].shape == deps.shape else deps[0]
        eps = np.atleast_2d(extra["eps50"])
    elif extra is not None and "eps" in extra:
        extra["eps"] += deps if extra["eps"].shape == deps.shape else deps[0]
        eps = np.atleast_2d(extra["eps"])
    else:
        eps = deps.copy()
        if extra is not None:
            extra["eps50"] = eps.copy()

    if is_1d and eps.ndim == 1:
        eps = eps.reshape(1, -1)

    # Element deletion check
    eps_max11 = float(p.get("eps_max11", _DEFAULT_EPS_MAX))
    eps_max22 = float(p.get("eps_max22", _DEFAULT_EPS_MAX))
    eps_max33 = float(p.get("eps_max33", _DEFAULT_EPS_MAX))
    eps_max12 = float(p.get("eps_max12", _DEFAULT_EPS_MAX))
    eps_max23 = float(p.get("eps_max23", _DEFAULT_EPS_MAX))
    eps_max31 = float(p.get("eps_max31", _DEFAULT_EPS_MAX))

    rupture = (
        (eps[:, 0] > eps_max11)
        | (eps[:, 1] > eps_max22)
        | (eps[:, 2] > eps_max33)
        | (np.abs(eps[:, 3] * 0.5) > eps_max12)
        | (np.abs(eps[:, 4] * 0.5) > eps_max23)
        | (np.abs(eps[:, 5] * 0.5) > eps_max31)
    )

    off = None
    if extra is not None:
        if "off50" in extra:
            off = np.atleast_1d(np.asarray(extra["off50"], dtype=sig.dtype)).copy()
        elif "off" in extra:
            off = np.atleast_1d(np.asarray(extra["off"], dtype=sig.dtype)).copy()
        else:
            off = np.ones(n, dtype=sig.dtype)
            extra["off50"] = off

    if off is not None:
        if off.shape[0] == 1 and n > 1:
            off = np.broadcast_to(off, (n,)).copy()
        off[rupture] = 0.0
        dead = (off == 0.0)
        if extra is not None:
            if "off50" in extra:
                extra["off50"] = off
            if "off" in extra:
                extra["off"] = off
    else:
        dead = rupture

    # Strain definition (sigeps50s.F90 lines 183-208)
    if gflag == 1:
        ep1, ep2, ep3 = eps[:, 0], eps[:, 1], eps[:, 2]
    elif gflag == -1:
        ep1, ep2, ep3 = -eps[:, 0], -eps[:, 1], -eps[:, 2]
    else:
        ep1, ep2, ep3 = mu, mu, mu

    if vflag == 1:
        ep4, ep5, ep6 = eps[:, 3], eps[:, 4], eps[:, 5]
    elif vflag == -1:
        ep4, ep5, ep6 = -eps[:, 3], -eps[:, 4], -eps[:, 5]
    else:
        ep4, ep5, ep6 = mu, mu, mu

    # Strain rate definition & filtering (sigeps50s.F90 lines 212-259)
    fcut_eff = 1.0e20 if (fcut <= 0.0 or fcut >= 1.0e19) else fcut
    if dt > 0.0:
        asrate = min(1.0, fcut_eff * dt)
    else:
        asrate = 0.0

    if extra is not None and "rate" in extra:
        epsp_rate = np.atleast_2d(np.asarray(extra["rate"], dtype=sig.dtype))
    elif extra is not None and "epsp_rate" in extra:
        epsp_rate = np.atleast_2d(np.asarray(extra["epsp_rate"], dtype=sig.dtype))
    elif extra is not None and "epsp" in extra:
        epsp_rate = np.atleast_2d(np.asarray(extra["epsp"], dtype=sig.dtype))
    elif dt > 0.0:
        epsp_rate = deps / dt
    else:
        epsp_rate = np.zeros_like(deps)

    uvar = None
    uvar_is_1d = False
    if extra is not None:
        if "uvar50" in extra:
            uvar = extra["uvar50"]
        elif "uvar" in extra:
            uvar = extra["uvar"]

    if uvar is None:
        uvar = np.zeros((n, 6), dtype=sig.dtype)
        if extra is not None:
            extra["uvar50"] = uvar
    elif uvar.ndim == 1:
        uvar_is_1d = True
        uvar = uvar.reshape(1, -1)

    if irate == 2:
        for k_dir in range(6):
            uvar[:, k_dir] = asrate * epsp_rate[:, k_dir] + (1.0 - asrate) * uvar[:, k_dir]
        dep = np.abs(uvar)
        dep1, dep2, dep3, dep4, dep5, dep6 = dep[:, 0], dep[:, 1], dep[:, 2], dep[:, 3], dep[:, 4], dep[:, 5]
        epsd = (dep1**2 + dep2**2 + dep3**2) + 0.5 * (dep4**2 + dep5**2 + dep6**2)
    else:
        eq_rate = np.sqrt(
            epsp_rate[:, 0]**2 + epsp_rate[:, 1]**2 + epsp_rate[:, 2]**2
            + 0.5 * (epsp_rate[:, 3]**2 + epsp_rate[:, 4]**2 + epsp_rate[:, 5]**2)
        )
        uvar[:, 0] = asrate * eq_rate + (1.0 - asrate) * uvar[:, 0]
        dep1 = dep2 = dep3 = dep4 = dep5 = dep6 = uvar[:, 0]
        epsd = uvar[:, 0].copy()

    if uvar_is_1d and extra is not None:
        if "uvar50" in extra and extra["uvar50"].ndim == 1:
            extra["uvar50"][:] = uvar[0]
        elif "uvar" in extra and extra["uvar"].ndim == 1:
            extra["uvar"][:] = uvar[0]

    if extra is not None:
        extra["epsd50"] = epsd

    # Plastic strain initialization
    if epsp is not None:
        eplas = np.atleast_1d(np.asarray(epsp, dtype=sig.dtype)).copy()
        if eplas.shape[0] != n:
            eplas = np.broadcast_to(eplas, (n,)).copy()
    else:
        eplas = np.zeros(n, dtype=sig.dtype)

    # Directional yield stress calculation
    ep_all = [ep1, ep2, ep3, ep4, ep5, ep6]
    dep_all = [dep1, dep2, dep3, dep4, dep5, dep6]
    yld_all = [_eval_yield_component(mat, k_comp, ep_all[k_comp], dep_all[k_comp], extra=extra) for k_comp in range(6)]

    # Stress update: uncompacted elements (sigeps50s.F90 lines 354-371)
    sign_stress = np.empty_like(st)
    not_comp = ~compacted

    for k_comp in range(6):
        sign_stress[not_comp, k_comp] = np.copysign(
            np.minimum(np.abs(st[not_comp, k_comp]), yld_all[k_comp][not_comp]),
            st[not_comp, k_comp],
        )

    # Plasticity treatment for fully compacted elements (sigeps50s.F90 lines 376-401)
    if icomp == 1 and np.any(compacted):
        depsv = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
        pres = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0

        dev_x = sig[:, 0] + gcomp * (deps[:, 0] - depsv) - pres
        dev_y = sig[:, 1] + gcomp * (deps[:, 1] - depsv) - pres
        dev_z = sig[:, 2] + gcomp * (deps[:, 2] - depsv) - pres
        dev_xy = sig[:, 3] + gcomp * deps[:, 3] * 0.5
        dev_yz = sig[:, 4] + gcomp * deps[:, 4] * 0.5
        dev_zx = sig[:, 5] + gcomp * deps[:, 5] * 0.5

        j2 = 0.5 * (dev_x**2 + dev_y**2 + dev_z**2) + dev_xy**2 + dev_yz**2 + dev_zx**2
        svm = np.sqrt(np.maximum(3.0 * j2, 0.0))

        yld_c = sigy + et * eplas
        safe_svm = np.maximum(svm, 1.0e-30)
        rfact = np.where(svm > 1.0e-30, np.minimum(1.0, yld_c / safe_svm), 1.0)
        pres_new = pres + 3.0 * bulk * depsv

        sign_stress[compacted, 0] = dev_x[compacted] * rfact[compacted] + pres_new[compacted]
        sign_stress[compacted, 1] = dev_y[compacted] * rfact[compacted] + pres_new[compacted]
        sign_stress[compacted, 2] = dev_z[compacted] * rfact[compacted] + pres_new[compacted]
        sign_stress[compacted, 3] = dev_xy[compacted] * rfact[compacted]
        sign_stress[compacted, 4] = dev_yz[compacted] * rfact[compacted]
        sign_stress[compacted, 5] = dev_zx[compacted] * rfact[compacted]

        denom = max(1.5 * gcomp + et, 1.0e-20)
        eplas[compacted] += ((1.0 - rfact[compacted]) * svm[compacted]) / denom

    # Element deletion: zero stresses
    if np.any(dead):
        sign_stress[dead] = 0.0

    if is_1d:
        sign_stress = sign_stress[0]
        c = c[0]
        eplas = float(eplas[0]) if hasattr(eplas, "__getitem__") else float(eplas)

    if return_tuple:
        return sign_stress, eplas, c
    return sign_stress


def sound_speed_solid(
    mat: Any,
    rho: float | np.ndarray | None = None,
    extra: dict | None = None,
    compacted: bool = False,
    **kwargs: Any,
) -> float | np.ndarray:
    """Acoustic longitudinal wave speed for LAW50.

    Fortran origin: ``engine/source/materials/mat/mat050/sigeps50s.F90``:
        SOUNDSP = SQRT(MAX(E11,E22,E33,G12,G23,G31)/RHO)
    """
    p = getattr(mat, "params", mat)
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = p.get("rho0", p.get("rho", p.get("MAT_RHO", 1.0))) if isinstance(p, dict) else getattr(p, "rho0", 1.0)
    rho_val = rho if rho is not None else rho0

    ea = float(p.get("ea", p.get("MAT_EA", p.get("E11", 0.0))) if isinstance(p, dict) else getattr(p, "ea", 0.0))
    eb = float(p.get("eb", p.get("MAT_EB", p.get("E22", 0.0))) if isinstance(p, dict) else getattr(p, "eb", 0.0))
    ec = float(p.get("ec", p.get("MAT_EC", p.get("E33", 0.0))) if isinstance(p, dict) else getattr(p, "ec", 0.0))
    gab = float(p.get("gab", p.get("MAT_GAB", p.get("G12", 0.0))) if isinstance(p, dict) else getattr(p, "gab", 0.0))
    gbc = float(p.get("gbc", p.get("MAT_GBC", p.get("G23", 0.0))) if isinstance(p, dict) else getattr(p, "gbc", 0.0))
    gca = float(p.get("gca", p.get("MAT_GCA", p.get("G31", 0.0))) if isinstance(p, dict) else getattr(p, "gca", 0.0))
    ecomp = float(p.get("ecomp", p.get("MAT_ECOMP", 0.0)) if isinstance(p, dict) else getattr(p, "ecomp", 0.0))
    gcomp = float(p.get("gcomp", 0.0) if isinstance(p, dict) else getattr(p, "gcomp", 0.0))
    if gcomp == 0.0 and ecomp > 0.0:
        pr = float(p.get("pr", p.get("MAT_PR", p.get("nu", 0.0))) if isinstance(p, dict) else getattr(p, "pr", 0.0))
        gcomp = ecomp / (1.0 + min(pr, 0.495))

    vcomp = float(p.get("vcomp", p.get("MAT_VCOMP", 0.0)) if isinstance(p, dict) else getattr(p, "vcomp", 0.0))
    sigy = float(p.get("sigy", p.get("MAT_SIGY", 0.0)) if isinstance(p, dict) else getattr(p, "sigy", 0.0))
    icomp = int(p.get("icompact", p.get("icomp", 1 if (ecomp * sigy * vcomp > 0.0) else 0)) if isinstance(p, dict) else getattr(p, "icompact", 0))

    if extra is not None and ("amu" in extra or "mu" in extra) and icomp == 1:
        mu_val = extra.get("amu", extra.get("mu"))
        rvol = 1.0 / (1.0 + np.asarray(mu_val, dtype=float))
        denom_v = 1.0 - vcomp
        beta = np.clip((1.0 - rvol) / denom_v, 0.0, 1.0) if abs(denom_v) > 1e-15 else 0.0
        e11_val = beta * ecomp + (1.0 - beta) * ea
        e22_val = beta * ecomp + (1.0 - beta) * eb
        e33_val = beta * ecomp + (1.0 - beta) * ec
        g12_val = beta * gcomp + (1.0 - beta) * gab
        g23_val = beta * gcomp + (1.0 - beta) * gbc
        g31_val = beta * gcomp + (1.0 - beta) * gca
        mod_max = np.maximum.reduce([e11_val, e22_val, e33_val, g12_val, g23_val, g31_val])
    else:
        is_compact = compacted or bool(kwargs.get("compacted", False))
        if not is_compact and extra is not None:
            comp_val = extra.get("compacted", False)
            if isinstance(comp_val, np.ndarray):
                is_compact = bool(np.any(comp_val))
            else:
                is_compact = bool(comp_val)

        if is_compact and ecomp > 0.0:
            mod_max = max(ecomp, gcomp)
        else:
            mod_max = max(ea, eb, ec, gab, gbc, gca)
            if mod_max <= 0.0 and ecomp > 0.0:
                mod_max = max(ecomp, gcomp)

    r = rho_val
    if isinstance(r, np.ndarray):
        return np.sqrt(np.maximum(mod_max, 0.0) / np.maximum(r, 1.0e-20))
    return math.sqrt(max(mod_max, 0.0) / max(float(r), 1.0e-20))


def consistent_solid_tangent(
    mat: Any,
    sig: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    epsp_incr: np.ndarray | None = None,
    extra: dict | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent tangent stiffness tensor (n, 6, 6) for LAW50 solid elements.

    Supports uncoupled orthotropic response in honeycomb state, and J2 isotropic
    elastoplastic tangent in compacted state.
    """
    if sig is None:
        sig = np.zeros((1, 6), dtype=float)

    n = sig.shape[0] if hasattr(sig, "shape") and sig.ndim > 1 else (1 if sig is not None else 0)
    if n == 0:
        return np.empty((0, 6, 6), dtype=float if sig is None else sig.dtype)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)

    if isinstance(mat, dict):
        p = mat
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = mat.params
    elif isinstance(mat, Law50Params):
        p = mat.__dict__
    elif hasattr(mat, "__dict__"):
        p = mat.__dict__
    else:
        p = {}
    ea = float(p.get("ea", p.get("MAT_EA", p.get("E11", 0.0))))
    eb = float(p.get("eb", p.get("MAT_EB", p.get("E22", 0.0))))
    ec = float(p.get("ec", p.get("MAT_EC", p.get("E33", 0.0))))
    gab = float(p.get("gab", p.get("MAT_GAB", p.get("G12", 0.0))))
    gbc = float(p.get("gbc", p.get("MAT_GBC", p.get("G23", 0.0))))
    gca = float(p.get("gca", p.get("MAT_GCA", p.get("G31", 0.0))))

    ecomp = float(p.get("ecomp", p.get("MAT_ECOMP", 0.0)))
    et = float(p.get("et", p.get("MAT_ET", p.get("hcomp", 0.0))))
    sigy = float(p.get("sigy", p.get("MAT_SIGY", 0.0)))
    pr = float(p.get("pr", p.get("MAT_PR", p.get("nu", 0.0))))
    vcomp = float(p.get("vcomp", p.get("MAT_VCOMP", 0.0)))

    pr_eff = min(pr, 0.495)
    gcomp = float(p.get("gcomp", 0.0))
    if gcomp == 0.0 and ecomp > 0.0:
        gcomp = ecomp / (1.0 + pr_eff)
    bulk = float(p.get("bulk", 0.0))
    if bulk == 0.0 and ecomp > 0.0:
        bulk = ecomp / (3.0 * (1.0 - 2.0 * pr_eff))

    icomp = int(p.get("icompact", p.get("icomp", 1 if (ecomp * sigy * vcomp > 0.0) else 0)))

    compacted = np.zeros(n, dtype=bool)
    if icomp == 1 and extra is not None:
        if "compacted" in extra:
            comp_arr = np.atleast_1d(np.asarray(extra["compacted"], dtype=bool))
            if comp_arr.shape[0] == n:
                compacted = comp_arr
            elif comp_arr.size == 1:
                compacted = np.full(n, bool(comp_arr[0]))

    D = np.zeros((n, 6, 6), dtype=sig.dtype)

    # 1. Uncompacted elements: orthotropic uncoupled tangent
    not_comp = ~compacted
    if np.any(not_comp):
        D[not_comp, 0, 0] = ea
        D[not_comp, 1, 1] = eb
        D[not_comp, 2, 2] = ec
        D[not_comp, 3, 3] = gab
        D[not_comp, 4, 4] = gbc
        D[not_comp, 5, 5] = gca

    # 2. Compacted elements: J2 elastoplastic tangent
    if np.any(compacted):
        g_shear = gcomp * 0.5  # gcomp is 2G
        k_bulk = bulk

        # Deviatoric stress
        pres = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        s_xx = sig[:, 0] - pres
        s_yy = sig[:, 1] - pres
        s_zz = sig[:, 2] - pres
        s_xy = sig[:, 3]
        s_yz = sig[:, 4]
        s_zx = sig[:, 5]

        j2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2) + s_xy**2 + s_yz**2 + s_zx**2
        svm = np.sqrt(np.maximum(3.0 * j2, 0.0))

        if epsp is not None:
            eplas = np.atleast_1d(np.asarray(epsp, dtype=sig.dtype))
        else:
            eplas = np.zeros(n, dtype=sig.dtype)

        yld = sigy + et * eplas

        for i in range(n):
            if not compacted[i]:
                continue
            # Elastic isotropic tensor
            c11 = k_bulk + (4.0 / 3.0) * g_shear
            c12 = k_bulk - (2.0 / 3.0) * g_shear
            D[i, 0, 0] = D[i, 1, 1] = D[i, 2, 2] = c11
            D[i, 0, 1] = D[i, 1, 0] = c12
            D[i, 0, 2] = D[i, 2, 0] = c12
            D[i, 1, 2] = D[i, 2, 1] = c12
            D[i, 3, 3] = D[i, 4, 4] = D[i, 5, 5] = g_shear

            # Plastic correction if yielding
            if svm[i] > yld[i] and svm[i] > 1.0e-20:
                s_vec = np.array([s_xx[i], s_yy[i], s_zz[i], 2.0 * s_xy[i], 2.0 * s_yz[i], 2.0 * s_zx[i]])
                flow = (3.0 / (2.0 * svm[i])) * s_vec
                denom = 3.0 * g_shear + et
                factor = (4.0 * g_shear**2) / max(denom, 1.0e-20)
                D[i] -= factor * np.outer(flow, flow)

    # 3. Ruptured or deleted elements
    off = None
    if extra is not None:
        off = extra.get("off50", extra.get("off", None))
    if off is not None:
        dead = (np.atleast_1d(np.asarray(off)).reshape(-1) == 0.0)
        D[dead] = 0.0

    return D


sound_speed = sound_speed_solid


def shell_update(
    mat: Material,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shell update rejection guard: LAW50 is 3D solid only."""
    raise NotImplementedError("LAW50 (/MAT/VISC_HONEY) is implemented for solid elements only.")


# Canonical alias exports
solid_update_law50 = solid_update
shell_update_law50 = shell_update
sound_speed_solid_law50 = sound_speed_solid
tangent_law50_solid = consistent_solid_tangent
solid_tangent = consistent_solid_tangent


def tangent(group: Any = None, x: Any = None, epsp_incr: Any = None) -> Any:
    """Stiffness tangent dispatch for element groups or implicit solver."""
    if group is None:
        return None
    mat = getattr(group, "mat", None) or getattr(group, "material", None)
    if mat is not None:
        return consistent_solid_tangent(mat, epsp_incr=epsp_incr)
    if isinstance(group, (Material, Law50Params, dict)) or hasattr(group, "params"):
        return consistent_solid_tangent(group, epsp_incr=epsp_incr)
    return None
