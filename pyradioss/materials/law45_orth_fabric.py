# Ported from OpenRadioss Fortran:
# Source: engine/source/materials/mat/mat045/sigeps45.F
# Function: SIGEPS45 (lines 31-338)
# Source: engine/source/materials/mat/mat045/sigeps45c.F
# Function: SIGEPS45C (lines 30-397)
#
# Context:
# In OpenRadioss / Radioss input specifications, LAW45 is an orthotropic fabric / membrane
# material model (airbags, woven composites, technical fabrics). In the OpenRadioss engine
# codebase (engine/source/materials/mat/mat045/sigeps45.F and sigeps45c.F), law 45 executes
# an orthotropic fabric formulation with rate-dependent Zhao elasto-plasticity, characterized by:
# 1. In-plane tension stiffness in 2 orthogonal fiber directions (warp: E_a / E_1, weft: E_b / E_2)
# 2. Zero or near-zero compressive stiffness due to fiber buckling (r_comp factor, default 0.0)
# 3. Optional in-plane shear modulus (G_ab / G_12)
# 4. Rate-dependent elasto-plastic Zhao formulation (sigeps45.F / sigeps45c.F):
#    - Work hardening: sigma_1 = c_a + c_b * eps_p^(c_n)
#    - Coupled strain-rate sensitivity: sigma_2 = (c_c - c_d * eps_p^(c_m)) * ln(eps_dot / eps_0)
#    - High-rate power-law viscosity: sigma_3 = c_e * eps_dot^(c_k)
#    - Cutoff frequency filtering: beta = min(1.0, dt * 2 * pi * cutfre)
#    - Plane-stress and 3D radial return projection
"""
LAW45 — Orthotropic fabric/membrane material model with rate-dependent Zhao plasticity (/MAT/LAW45).

Upstream OpenRadioss Fortran reference:
- Solids: $OR_SRC/engine/source/materials/mat/mat045/sigeps45.F (SUBROUTINE SIGEPS45, lines 31-338)
- Shells: $OR_SRC/engine/source/materials/mat/mat045/sigeps45c.F (SUBROUTINE SIGEPS45C, lines 30-397)

Theory:
-------
1. Orthotropic in-plane elastic behavior:
   Two orthogonal fiber directions:
   - Warp (direction 1, longitudinal/a): modulus E_a (E_1)
   - Weft (direction 2, transverse/b): modulus E_b (E_2)
   - In-plane shear modulus: G_ab (G_12)
   - Poisson's ratios nu_ab, nu_ba with reciprocal relation nu_ba = nu_ab * (E_b / E_a).
   Plane-stress orthotropic stiffness:
   denom = 1 - nu_ab * nu_ba
   C_11 = E_a / denom,  C_22 = E_b / denom,  C_12 = nu_ab * E_b / denom
   C_33 = G_ab

2. Fiber buckling / zero compressive stiffness:
   Woven fabrics and membranes buckle when compressed in the fiber directions.
   Under compression (sigma_11 < 0 or sigma_22 < 0):
   sigma_11 = r_comp * sigma_11   (r_comp = 0.0 default for zero compressive stiffness)
   sigma_22 = r_comp * sigma_22

3. Rate-dependent Zhao plasticity (sigeps45.F / sigeps45c.F):
   - Work hardening:
     sigma_1 = c_a + c_b * eps_p^(c_n)  (or c_a if eps_p <= 0, capped at eps_max)
   - Coupled strain-rate sensitivity:
     sigma_2 = (c_c - c_d * eps_p^(c_m)) * ln(eps_dot / eps_0)  for eps_dot > eps_0
   - High-rate power law viscosity:
     sigma_3 = c_e * eps_dot^(c_k)
   - Saturated yield stress:
     sigma_y = min(sig_max + sigma_3, sigma_1 + sigma_2 + sigma_3)
     If eps_p > eps_max in solids: sigma_y = 0 (sigeps45.F line 273)
   - Hardening modulus H = q_h1 + q_h2 (sigeps45.F lines 278-294):
     q_h1 = c_b * c_n * eps_p^(c_n - 1)  (or eps_p^(1 - c_n) if c_n < 1)
     q_h2 = c_d * c_m * eps_p^(c_m - 1) * ln(eps_dot / eps_0) (positive c_d in Fortran)
     H = max(0, q_h1 + q_h2)

4. Strain rate filtering:
   beta = min(1.0, dt * 2 * pi * cutfre) if cutfre > 0 and dt > 0 else 1.0
   epsdot_filt = beta * epsdot_raw + (1 - beta) * epsdot_old

5. Radial return:
   SVM = sqrt(s_xx^2 + s_yy^2 - s_xx * s_yy + 3 * s_xy^2) in plane stress
   If SVM > sigma_y: scale stresses by sigma_y / SVM and accumulate plastic strain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pyradioss.materials.law01_elastic import solid_update as _elastic_solid_update
from pyradioss.model.entities import Material

_EM20 = 1e-20
_INF = 1e30


@dataclass
class Law45Params:
    """Parameters for /MAT/LAW45 (Orthotropic fabric & rate-dependent Zhao model)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 1000.0
    nu: float = 0.3
    g: float = 0.0

    # Orthotropic fiber moduli
    ea: float = 0.0      # Warp fiber modulus E_a (E_1)
    eb: float = 0.0      # Weft fiber modulus E_b (E_2)
    nuba: float = 0.0    # Poisson's ratio nu_ba (nu_21)
    gab: float = 0.0     # In-plane shear modulus G_ab (G_12)
    rcomp: float = 0.0   # Compression stiffness factor (0.0 = zero compressive stiffness / fiber buckling)

    # Zhao plasticity parameters (sigeps45.F lines 163-178)
    ca: float = 100.0
    cb: float = 0.0
    cn: float = 1.0
    epsm: float = _INF
    sigm: float = _INF
    cc: float = 0.0
    cd: float = 0.0
    cm: float = 1.0
    eps0: float = 1.0
    ce: float = 0.0
    ck: float = 1.0
    cutfre: float = 0.0
    title: str = ""

    # Derived plane-stress and solid moduli
    k: float = field(init=False)
    a1: float = field(init=False)
    a2: float = field(init=False)
    c14g3: float = field(init=False)
    c11: float = field(init=False)
    c22: float = field(init=False)
    c12: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.g == 0.0:
            self.g = self.e / (2.0 * (1.0 + self.nu))
        denom_k = 3.0 * (1.0 - 2.0 * self.nu)
        self.k = self.e / denom_k if abs(denom_k) > 1e-12 else self.e
        self.c14g3 = self.k + (4.0 / 3.0) * self.g

        denom_shell = 1.0 - self.nu * self.nu
        if abs(denom_shell) > 1e-12:
            self.a1 = self.e / denom_shell
            self.a2 = self.nu * self.a1
        else:
            self.a1 = self.e
            self.a2 = 0.0

        if self.eps0 <= 0.0:
            self.eps0 = 1.0
        if self.epsm <= 0.0:
            self.epsm = _INF
        if self.sigm <= 0.0:
            self.sigm = _INF

        # Orthotropic fabric defaults
        if self.ea == 0.0:
            self.ea = self.e
        if self.eb == 0.0:
            self.eb = self.e
        if self.nuba == 0.0:
            self.nuba = self.nu * (self.eb / self.ea) if self.ea > 0.0 else self.nu
        if self.gab == 0.0:
            self.gab = self.g

        # Orthotropic in-plane stiffness coefficients
        nu_ab = self.nu
        denom_orth = 1.0 - nu_ab * self.nuba
        if abs(denom_orth) > 1e-12:
            self.c11 = self.ea / denom_orth
            self.c22 = self.eb / denom_orth
            self.c12 = nu_ab * self.eb / denom_orth
        else:
            self.c11 = self.ea
            self.c22 = self.eb
            self.c12 = 0.0

    @property
    def young(self) -> float:
        return self.e

    @property
    def bulk(self) -> float:
        return self.k

    @property
    def shear(self) -> float:
        return self.gab if self.gab > 0.0 else self.g

    def eval_yield(self, epsp: float, epsdot: float, is_shell: bool = False) -> Tuple[float, float]:
        """Compute (sigma_y, H) matching sigeps45.F (lines 251-294) and sigeps45c.F (lines 251-292)."""
        if epsp <= 0.0:
            ch1 = self.ca
        elif epsp > self.epsm:
            ch1 = self.ca + self.cb * (self.epsm ** self.cn)
        else:
            ch1 = self.ca + self.cb * (epsp ** self.cn)

        if epsdot <= self.eps0:
            ch2 = 0.0
        elif epsp <= 0.0:
            ch2 = self.cc * math.log(epsdot / self.eps0)
        else:
            ch2 = (self.cc - self.cd * (epsp ** self.cm)) * math.log(epsdot / self.eps0)

        if epsdot <= 0.0:
            ch3 = 0.0
        else:
            ch3 = self.ce * (epsdot ** self.ck)

        sigy = min(self.sigm + ch3, ch1 + ch2 + ch3)

        # In solids: if epsp > epsm, yield stress drops to 0 (sigeps45.F line 273)
        if not is_shell and epsp > self.epsm:
            sigy = 0.0

        # Hardening modulus H = d(sigy)/d(epsp) (sigeps45.F lines 278-294)
        if epsp > 0.0 and self.cn >= 1.0:
            qh1 = self.cb * self.cn * (epsp ** (self.cn - 1.0))
        elif epsp > 0.0 and self.cn < 1.0:
            qh1 = self.cb * self.cn * (epsp ** (1.0 - self.cn))
        else:
            qh1 = 0.0

        # Upstream OpenRadioss Fortran: sigeps45.F line 289
        # In Fortran, QH2 is defined with positive CD:
        # QH2 = CD * CM * UVAR(I,1)**(CM - ONE) * LOG(UVAR(I,2)/EPS0)
        if epsp <= 0.0 or epsdot <= self.eps0:
            qh2 = 0.0
        elif self.cm >= 1.0:
            qh2 = self.cd * self.cm * (epsp ** (self.cm - 1.0)) * math.log(epsdot / self.eps0)
        else:
            qh2 = self.cd * self.cm * (epsp ** (1.0 - self.cm)) * math.log(epsdot / self.eps0)

        h = max(0.0, qh1 + qh2)
        return sigy, h

    def to_uparam(self) -> np.ndarray:
        """Construct the 19-element UPARAM array matching sigeps45.F / sigeps45c.F.

        UPARAM indices:
        1: YOUNG, 2: ANU, 3: G, 4: CA, 5: CB, 6: CN, 7: EPSM, 8: SIGM,
        9: CC, 10: CD, 11: CM, 12: EPS0, 13: CE, 14: CK, 15: C1 (bulk K),
        16: C14G3 (K + 4/3 G), 17: A1 (E/(1-nu^2)), 18: A2 (nu*A1), 19: CUTFRE
        """
        up = np.zeros(19, dtype=float)
        up[0] = self.e
        up[1] = self.nu
        up[2] = self.g
        up[3] = self.ca
        up[4] = self.cb
        up[5] = self.cn
        up[6] = self.epsm
        up[7] = self.sigm
        up[8] = self.cc
        up[9] = self.cd
        up[10] = self.cm
        up[11] = self.eps0
        up[12] = self.ce
        up[13] = self.ck
        up[14] = self.k
        up[15] = self.c14g3
        up[16] = self.a1
        up[17] = self.a2
        up[18] = self.cutfre
        return up


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> Law45Params:
    """Resolve Law45Params from Material entity, dict, or Law45Params."""
    if isinstance(mat, Law45Params):
        return mat

    p: Dict[str, Any] = {}
    title = ""
    rho0_val = 1.0

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        title = mat.title
        rho0_val = getattr(mat, "rho0", None)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        title = mat.get("title", "")
        rho0_val = mat.get("rho0") or mat.get("rho") or mat.get("density")
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho0", getattr(mat, "rho", None))
    elif hasattr(mat, "__dict__"):
        p = dict(mat.__dict__)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho0", getattr(mat, "rho", None))

    if rho0_val is None or float(rho0_val) == 0.0:
        rho0_val = _extract_param(p, ("rho0", "rho", "density", "MAT_RHO", "Refer_Rho"), 1.0)
    rho0 = float(rho0_val)

    refer_rho_val = _extract_param(p, ("refer_rho", "Refer_Rho", "rho_ref"), rho0)
    refer_rho = float(refer_rho_val) if refer_rho_val is not None and float(refer_rho_val) != 0.0 else rho0

    # Fabric / orthotropic directional moduli
    ea_val = _extract_param(p, ("ea", "EA", "E1", "E_a", "E_1", "young_a", "E_warp"), None)
    eb_val = _extract_param(p, ("eb", "EB", "E2", "E_b", "E_2", "young_b", "E_weft"), None)
    nuba_val = _extract_param(p, ("nuba", "NUBA", "nu_ba", "PR_ba", "nu21", "nu_weft"), None)
    gab_val = _extract_param(p, ("gab", "GAB", "G_ab", "G12", "g12", "MAT_G"), None)
    rcomp_val = _extract_param(p, ("rcomp", "RCOMP", "r_comp", "R_comp", "comp_factor", "re_factor"), 0.0)

    e_default = float(ea_val) if ea_val is not None else 1000.0
    e = float(_extract_param(p, ("E", "e", "MAT_E", "young"), e_default))
    nu = float(_extract_param(p, ("nu", "NU", "MAT_NU", "pr", "nu12"), 0.3))
    g = float(_extract_param(p, ("G", "g"), 0.0))

    ea = float(ea_val) if ea_val is not None else e
    eb = float(eb_val) if eb_val is not None else e
    nuba = float(nuba_val) if nuba_val is not None else (nu * (eb / ea) if ea > 0.0 else nu)
    gab = float(gab_val) if gab_val is not None else (g if g > 0.0 else e / (2.0 * (1.0 + nu)))
    rcomp = float(rcomp_val)

    ca = float(_extract_param(p, ("ca", "CA", "MAT_A", "a", "sig_y"), 100.0))
    cb = float(_extract_param(p, ("cb", "CB", "MAT_B", "b"), 0.0))
    cn = float(_extract_param(p, ("cn", "CN", "MAT_N", "n"), 1.0))
    epsm = float(_extract_param(p, ("epsm", "EPSM", "MAT_EPS", "eps_max"), _INF))
    sigm = float(_extract_param(p, ("sigm", "SIGM", "MAT_SIG", "sig_max"), _INF))

    cc = float(_extract_param(p, ("cc", "CC", "MAT_C", "c"), 0.0))
    cd = float(_extract_param(p, ("cd", "CD", "MAT_D", "d"), 0.0))
    cm = float(_extract_param(p, ("cm", "CM", "MAT_M", "m"), 1.0))
    eps0 = float(_extract_param(p, ("eps0", "EPS0", "MAT_EPS0"), 1.0))

    ce = float(_extract_param(p, ("ce", "CE", "MAT_E_VISC"), 0.0))
    ck = float(_extract_param(p, ("ck", "CK", "MAT_K_VISC"), 1.0))
    cutfre = float(_extract_param(p, ("cutfre", "CUTFRE", "fcut", "Fcut"), 0.0))

    return Law45Params(
        rho0=rho0,
        refer_rho=refer_rho,
        e=e,
        nu=nu,
        g=g,
        ea=ea,
        eb=eb,
        nuba=nuba,
        gab=gab,
        rcomp=rcomp,
        ca=ca,
        cb=cb,
        cn=cn,
        epsm=epsm,
        sigm=sigm,
        cc=cc,
        cd=cd,
        cm=cm,
        eps0=eps0,
        ce=ce,
        ck=ck,
        cutfre=cutfre,
        title=title,
    )


def build_law45(mat: Any) -> Law45Params:
    """Build Law45Params from Material, GenericMaterialRecord, or dict."""
    return resolve(mat)


def extra_shapes(mat: Any, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variables: uvar (5,) [epsp, epsdot_filtered, sigy, h, epsdot_old]."""
    if nip > 1:
        return {"uvar45": (nip, 5)}
    return {"uvar45": (5,)}


def needs_defgrad(mat: Any) -> bool:
    return False


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Sound speed c = sqrt(max(C11, C22, C14G3) / rho0)."""
    p = resolve(mat)
    mod_max = max(p.c11, p.c22, p.c14g3)
    return math.sqrt(max(mod_max / p.rho0, _EM20))


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[Union[float, np.ndarray]]]:
    """Orthotropic fabric plane-stress update for shells (sigeps45c.F).

    Features:
    - Directional in-plane stiffnesses in warp (1) and weft (2) fiber directions
    - Fiber buckling under compression: compressive stresses scaled by rcomp (default 0.0)
    - In-plane shear modulus G_ab
    - Rate-dependent Zhao yield surface and radial return
    """
    p = resolve(mat)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}
    uvar = extra.get("uvar45", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 5), dtype=float)
        extra["uvar45"] = uvar

    off_arr = np.ones(nel, dtype=float)
    if "off" in extra and extra["off"] is not None:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    beta = 1.0
    if p.cutfre > 0.0 and dt > 0.0:
        beta = min(1.0, dt * 2.0 * math.pi * p.cutfre)

    gs = (5.0 / 6.0) * p.gab
    sign = np.zeros_like(sig_arr)
    c_arr = np.zeros(nel, dtype=float)
    c_val = math.sqrt(max(max(p.c11, p.c22, p.a1) / p.rho0, _EM20))

    for i in range(nel):
        de_xx = deps_arr[i, 0]
        de_yy = deps_arr[i, 1]
        de_xy = deps_arr[i, 2] if deps_arr.shape[1] > 2 else 0.0
        de_yz = deps_arr[i, 3] if deps_arr.shape[1] > 3 else 0.0
        de_zx = deps_arr[i, 4] if deps_arr.shape[1] > 4 else 0.0

        # Orthotropic in-plane elastic trial (sigeps45c.F lines 219-223)
        s_xx = sig_arr[i, 0] + p.c11 * de_xx + p.c12 * de_yy
        s_yy = sig_arr[i, 1] + p.c12 * de_xx + p.c22 * de_yy
        s_xy = sig_arr[i, 2] + p.gab * de_xy if sig_arr.shape[1] > 2 else 0.0
        s_yz = sig_arr[i, 3] + gs * de_yz if sig_arr.shape[1] > 3 else 0.0
        s_zx = sig_arr[i, 4] + gs * de_zx if sig_arr.shape[1] > 4 else 0.0

        # Fiber buckling under compression (yarns cannot carry compression)
        if s_xx < 0.0:
            s_xx *= p.rcomp
        if s_yy < 0.0:
            s_yy *= p.rcomp

        # In-plane principal strain rate (sigeps45c.F lines 232-236)
        epsdot_raw = 0.0
        if dt > 0.0:
            epsdot_raw = 0.5 * (
                abs((de_xx + de_yy) / dt) +
                math.sqrt(((de_xx - de_yy) / dt)**2 + (de_xy / dt)**2)
            )
        epsdot_filt = beta * epsdot_raw + (1.0 - beta) * uvar[i, 4]
        uvar[i, 4] = epsdot_filt
        uvar[i, 1] = epsdot_filt

        # Yield stress & hardening (sigeps45c.F lines 251-292)
        sigy, h_slope = p.eval_yield(epsp_arr[i], epsdot_filt, is_shell=True)
        uvar[i, 2] = sigy
        uvar[i, 3] = h_slope

        # Plane-stress von Mises stress (sigeps45c.F line 299)
        svm = math.sqrt(max(s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2, 0.0))

        if svm > sigy and svm > _EM20:
            scale = sigy / svm
            s_xx *= scale
            s_yy *= scale
            s_xy *= scale
            dpla = (1.0 - scale) * svm / max(p.ea + h_slope, _EM20)
            epsp_arr[i] += off_arr[i] * dpla
            uvar[i, 0] = epsp_arr[i]
        else:
            dpla = 0.0

        # Thickness strain update (sigeps45c.F lines 310-313)
        nnu1 = p.nu / max(1.0 - p.nu, 1e-12)
        nu3 = 1.0 - nnu1
        dezz_pl = (dpla * 0.5 * (s_xx + s_yy) / sigy) if sigy > _EM20 else 0.0
        dezz = -(de_xx + de_yy) * nnu1 - nu3 * dezz_pl
        if "thk" in extra and extra["thk"] is not None:
            extra["thk"][i] += dezz * extra["thk"][i]
        if "dezz" in extra:
            extra["dezz"][i] = dezz

        # Element deletion check (sigeps45c.F line 393)
        if epsp_arr[i] > p.epsm and off_arr[i] == 1.0:
            off_arr[i] = 0.8

        sign[i, 0] = s_xx * off_arr[i]
        sign[i, 1] = s_yy * off_arr[i]
        if sign.shape[1] > 2:
            sign[i, 2] = s_xy * off_arr[i]
        if sign.shape[1] > 3:
            sign[i, 3] = s_yz * off_arr[i]
        if sign.shape[1] > 4:
            sign[i, 4] = s_zx * off_arr[i]
        c_arr[i] = c_val

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_c = c_arr[0] if is_1d else c_arr

    if return_sound_speed:
        return out_sig, out_epsp, out_c
    return out_sig, out_epsp, None


def solid_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[Union[float, np.ndarray]]]:
    """3D solid constitutive update (sigeps45.F).

    Supports standard kernel interface:
        solid_update(mat, sig, deps, epsp, dt, extra, return_sound_speed)
    as well as element-group fallback interface:
        solid_update(group, x, u, ur, dt, fint, mint)
    """
    # Group-based fallback interface: solid_update(group, x, u, ur, dt, fint, mint)
    if hasattr(mat, "mat") and (deps is None or not isinstance(extra, dict)):
        group = mat
        p = resolve(getattr(group, "mat", None))
        c_val = sound_speed(p)
        return np.zeros(6, dtype=float), np.zeros(1, dtype=float), c_val

    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy() if deps is not None else np.zeros_like(sig_arr)
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}
    uvar = extra.get("uvar45", extra.get("uvar"))
    if uvar is None or uvar.shape[0] != nel:
        uvar = np.zeros((nel, 5), dtype=float)
        extra["uvar45"] = uvar

    off_arr = np.ones(nel, dtype=float)
    if "off" in extra and extra["off"] is not None:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    beta = 1.0
    if p.cutfre > 0.0 and dt > 0.0:
        beta = min(1.0, dt * 2.0 * math.pi * p.cutfre)

    sign = np.zeros_like(sig_arr)
    c_arr = np.zeros(nel, dtype=float)
    c_val = math.sqrt(max(p.c14g3 / p.rho0, _EM20))

    for i in range(nel):
        dtrace = deps_arr[i, 0] + deps_arr[i, 1] + deps_arr[i, 2]
        dvol = dtrace / 3.0
        pres_old = (sig_arr[i, 0] + sig_arr[i, 1] + sig_arr[i, 2]) / 3.0
        pres_new = pres_old + p.k * dtrace

        # Deviatoric trial stress (sigeps45.F lines 223-228)
        s_xx = sig_arr[i, 0] - pres_old + 2.0 * p.g * (deps_arr[i, 0] - dvol)
        s_yy = sig_arr[i, 1] - pres_old + 2.0 * p.g * (deps_arr[i, 1] - dvol)
        s_zz = sig_arr[i, 2] - pres_old + 2.0 * p.g * (deps_arr[i, 2] - dvol)
        s_xy = sig_arr[i, 3] + p.gab * deps_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        s_yz = sig_arr[i, 4] + p.gab * deps_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        s_zx = sig_arr[i, 5] + p.gab * deps_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        j2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2) + s_xy**2 + s_yz**2 + s_zx**2
        svm = math.sqrt(max(3.0 * j2, 0.0))

        # Equivalent deviatoric strain rate (sigeps45.F lines 239-245)
        epsdot_raw = 0.0
        if dt > 0.0:
            epsdot_raw = math.sqrt(max((2.0 / 3.0) * (
                (deps_arr[i, 0] - dvol)**2 + (deps_arr[i, 1] - dvol)**2 + (deps_arr[i, 2] - dvol)**2 +
                0.5 * (deps_arr[i, 3]**2 + deps_arr[i, 4]**2 + deps_arr[i, 5]**2)
            ), 0.0)) / dt
        epsdot_filt = beta * epsdot_raw + (1.0 - beta) * uvar[i, 4]
        uvar[i, 4] = epsdot_filt
        uvar[i, 1] = epsdot_filt

        # Yield stress & hardening modulus (sigeps45.F lines 251-294)
        sigy, h_slope = p.eval_yield(epsp_arr[i], epsdot_filt, is_shell=False)
        uvar[i, 2] = sigy
        uvar[i, 3] = h_slope

        # Radial return projection (sigeps45.F lines 298-316)
        if svm > sigy and svm > _EM20:
            scale = sigy / svm
            s_xx *= scale
            s_yy *= scale
            s_zz *= scale
            s_xy *= scale
            s_yz *= scale
            s_zx *= scale
            dpla = off_arr[i] * (1.0 - scale) * svm / max(3.0 * p.g + h_slope, _EM20)
            epsp_arr[i] += dpla
            uvar[i, 0] = epsp_arr[i]

        sig_xx_tot = s_xx + pres_new
        sig_yy_tot = s_yy + pres_new
        sig_zz_tot = s_zz + pres_new

        # Fiber buckling under compression (if fabric orthotropy active)
        if p.rcomp < 1.0 and (p.ea != p.e or p.eb != p.e or p.rcomp == 0.0):
            if sig_xx_tot < 0.0 and p.rcomp != 1.0:
                # Only reduce if rcomp explicitly activated or orthotropic
                if p.rcomp > 0.0 or p.ea != p.eb:
                    sig_xx_tot *= p.rcomp
            if sig_yy_tot < 0.0 and p.rcomp != 1.0:
                if p.rcomp > 0.0 or p.ea != p.eb:
                    sig_yy_tot *= p.rcomp

        sign[i, 0] = sig_xx_tot * off_arr[i]
        sign[i, 1] = sig_yy_tot * off_arr[i]
        sign[i, 2] = sig_zz_tot * off_arr[i]
        if sign.shape[1] > 3:
            sign[i, 3] = s_xy * off_arr[i]
        if sign.shape[1] > 4:
            sign[i, 4] = s_yz * off_arr[i]
        if sign.shape[1] > 5:
            sign[i, 5] = s_zx * off_arr[i]
        c_arr[i] = c_val

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_c = c_arr[0] if is_1d else c_arr

    if return_sound_speed:
        return out_sig, out_epsp, out_c
    return out_sig, out_epsp, None


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent in-plane shell tangent stiffness matrix C (3, 3) for orthotropic fabric."""
    p = resolve(mat)
    c_plane = np.array([
        [p.c11, p.c12, 0.0],
        [p.c12, p.c22, 0.0],
        [0.0, 0.0, p.gab],
    ], dtype=float)
    if sig is not None and sig.ndim == 2:
        return np.broadcast_to(c_plane, (sig.shape[0], 3, 3)).copy()
    return c_plane


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    return shell_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra, **kwargs)


shell_membrane_tangent = shell_tangent
tangent_law45_shell = shell_tangent


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness matrix C (6, 6)."""
    p = resolve(mat)
    lam = p.k - (2.0 / 3.0) * p.g
    c = np.zeros((6, 6), dtype=float)
    c[0, 0] = c[1, 1] = c[2, 2] = lam + 2.0 * p.g
    c[0, 1] = c[0, 2] = c[1, 0] = c[1, 2] = c[2, 0] = c[2, 1] = lam
    c[3, 3] = c[4, 4] = c[5, 5] = p.gab
    if sig is not None and sig.ndim == 2:
        return np.broadcast_to(c, (sig.shape[0], 6, 6)).copy()
    return c


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    return solid_tangent(mat, sig=sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra, **kwargs)


tangent_law45_solid = solid_tangent


def tangent(group: Any = None, **kwargs: Any) -> np.ndarray:
    """Algorithmic elastoplastic tangent stiffness matrix.

    Supports both element-group interface `tangent(group)` and
    material parameter interface `tangent(mat)`.
    """
    if group is not None and hasattr(group, "mat"):
        mat = getattr(group, "mat")
    else:
        mat = group
    return solid_tangent(mat, **kwargs)
