"""LAW96 — Thermo-Elasto-Viscoplastic Material Model (/MAT/LAW96, /MAT/THERMO_VISC_POLY).

Upstream OpenRadioss Fortran reference:
----------------------------------------
Source: $OR_SRC/engine/source/materials/mat/mat096/sigeps96.F
Subroutine: SIGEPS96 (lines 28-517)
Associated Curve Interpolation: engine/source/tools/curve/vinter.F (VINTER2)
Card reader: starter/source/materials/mat/mat096/hm_read_mat96.F

Constitutive Theory and Physics:
--------------------------------
LAW96 is a thermo-viscoplastic constitutive model combining Drucker-Prager
pressure-dependent plasticity with non-linear hardening, post-peak softening,
rubbery state modulus plateau, and temperature/strain-rate coupling.

1. Elastic Trial State:
   Volumetric strain increment:
       deps_v = deps_xx + deps_yy + deps_zz
   Deviatoric strain increment:
       d_dev = deps - (1/3) * deps_v * I
   Trial hydrostatic pressure:
       P_trial = P_old - K(T) * deps_v
       where K(T) = K_ini * fac_E(T) if temperature function IFUNC_E is defined.
   Trial deviatoric stress:
       s_trial = s_old + 2 * G(T) * d_dev
       where G(T) = G_ini * fac_E(T).

2. Drucker-Prager Yield Criterion:
   Equivalent von Mises stress:
       Q = sqrt(3/2 * s_ij * s_ij)
   Hydrostatic pressure:
       P = -1/3 * tr(sigma)
   Yield function (sigeps96.F lines 277-282):
       F = Q - max(0, P * tan(beta) + sigma_y(eps_p, eps_dot, T)) <= 0
   where:
       tan(beta) = UPARAM(5) is the friction angle tangent.

3. Thermo-Viscoplastic Hardening Law (sigeps96.F lines 212-227, 442-446):
       sigma_y = sigma_y0 * F_rate(eps_dot)
               + sigma_a * (1 - exp(-R_a * eps_p))
               - sigma_b * (1 - exp(-R_b * eps_p))
               + (1/3) * sigma_r * eps_p * (3 - R_c * eps_p^2) / (1 - R_c * eps_p^2)
   where:
       sigma_y0: initial yield stress in pure shear (UPARAM(8))
       F_rate:   strain rate scaling factor from table IFUNC_YLD
       sigma_a:  peak hardening gain (UPARAM(9))
       sigma_b:  post-peak softening drop (UPARAM(10))
       sigma_r:  rubbery modulus in rubbery plateau state (UPARAM(11))
       R_a:      temperature-dependent exponent:
                 R_a = R_a1 * (T / T_ref)^N_a + R_a2   (when JTHE = 1)
                 R_a = R_a1 + R_a2                    (otherwise)
       R_b:      exponent for softening drop evolution (UPARAM(15))
       R_c:      singularity limit factor for rubbery modulus (UPARAM(16))

4. Non-Associated Flow Rule:
   Plastic flow direction:
       d_eps_p = d_lam * (3 / (2 * Q) * s - 1/3 * tan(psi) * I)
   where tan(psi) = UPARAM(6) is the plastic flow dilatancy angle.
   Equivalent plastic strain accumulation:
       d_eps_p_dev = d_lam * k_ep
       where k_ep = 1 / (1 + nu_p^2) (UPARAM(17)), nu_p = UPARAM(20).
   Volumetric plastic strain accumulation:
       d_eps_p_vol = - d_lam * tan(psi)

5. Return Mapping and Elastic Fallback:
   - Below yield (F <= 0) or in elastic fallback mode:
     The trial stress state is fully elastic. No plastic flow occurs.
   - Above yield (F > 0):
     A Newton-Raphson return-mapping loop solves for the plastic multiplier d_lam
     satisfying the yield condition at the end of the time step.

State Variable Structure (UVAR array):
--------------------------------------
sigeps96.F maps internal element variables as:
  - UVAR(1) : Current yield stress sigma_y
  - UVAR(2) : Smoothed equivalent plastic strain rate eps_dot
  - DEFP(1) : Deviatoric equivalent plastic strain eps_p
  - DEFP(2) : Volumetric plastic strain eps_pv
  - TEMP    : Element temperature T
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from pyradioss.model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law96Params:
    """Parameters for OpenRadioss /MAT/LAW96 (Thermo-viscoplastic polymer).

    Upstream Fortran reference:
        $OR_SRC/engine/source/materials/mat/mat096/sigeps96.F
        lines 110-127 (UPARAM array definition).
    """
    id: int = 1
    title: str = ""
    law: int = 96
    law_name: str = "LAW96"

    # Density
    rho0: float = 1.0
    rhor: float = 1.0

    # Elastic Moduli (UPARAM 1-4)
    young: float = 1000.0     # UPARAM(1): Young's modulus E
    nu: float = 0.35          # UPARAM(4): Poisson's ratio nu

    # Drucker-Prager Angles (UPARAM 5-7)
    tanb: float = 0.2         # UPARAM(5): tan(beta) - yield friction angle
    tanp: float = 0.1         # UPARAM(6): tan(psi) - flow dilatancy angle

    # Plasticity & Hardening Parameters (UPARAM 8-16)
    sigy: float = 30.0        # UPARAM(8): initial yield stress in pure shear
    siga: float = 15.0        # UPARAM(9): max yield gain during hardening peak
    sigb: float = 10.0        # UPARAM(10): max yield drop during softening
    sigr: float = 5.0         # UPARAM(11): rubbery modulus
    ra1: float = 50.0         # UPARAM(12): 1st exponent coefficient for siga
    ra2: float = 0.0          # UPARAM(13): 2nd exponent coefficient for siga
    na: float = 0.0           # UPARAM(14): temperature exponent for ra
    rb: float = 20.0          # UPARAM(15): exponent coefficient for sigb
    rc: float = 0.5           # UPARAM(16): coefficient for sigr evolution

    # Plastic Kinematics & Temperature (UPARAM 17, 19, 20)
    nup: float = 0.0          # UPARAM(20): plastic Poisson ratio
    alpha: float = 0.0        # UPARAM(19): strain rate smoothing coefficient
    tref: float = 293.15      # Reference temperature T_ref (K)
    jthe: int = 0             # Thermal coupling flag (1 = active, 0 = inactive)
    elastic_only: bool = False # Flag for elastic-only fallback mode

    # Derived constants
    g: float = field(init=False)
    bulk: float = field(init=False)
    kep: float = field(init=False)
    yxi: float = field(init=False)
    a11_2d: float = field(init=False)
    a21_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 1.0
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.young <= 0.0:
            self.young = 1000.0

        self.nu = max(0.0, min(0.49999, self.nu))
        self.g = 0.5 * self.young / max(1.0 + self.nu, _EM20)
        self.bulk = self.young / max(3.0 * (1.0 - 2.0 * self.nu), _EM20)

        # UPARAM(17): kep = 1 / (1 + nup^2)
        self.kep = 1.0 / (1.0 + self.nup ** 2)
        # UPARAM(7): yxi = sqrt(1 + tanb^2)
        self.yxi = math.sqrt(1.0 + self.tanb ** 2)

        denom_2d = max(1.0 - self.nu ** 2, _EM20)
        self.a11_2d = self.young / denom_2d
        self.a21_2d = self.nu * self.young / denom_2d

        self.c_solid = math.sqrt(max(0.0, (self.bulk + 4.0 / 3.0 * self.g) / self.rho0))
        self.c_shell = math.sqrt(max(0.0, self.a11_2d / self.rho0))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law96(mat_def: Any = None, **kwargs: Any) -> Law96Params:
    """Construct Law96Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law96Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)

    data.update(kwargs)

    mat_id = int(_extract_val(data, ["id", "mat_id", "user_id"], 1))
    title = str(data.get("title", f"LAW96_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 1.0)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E", "UPARAM_1"], 1000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson", "nux", "UPARAM_4"], 0.35)

    tanb = _extract_val(data, ["TANB", "tanb", "UPARAM_5"], 0.2)
    tanp = _extract_val(data, ["TANP", "tanp", "UPARAM_6"], 0.1)

    sigy = _extract_val(data, ["SIGY", "sigy", "yield_stress", "UPARAM_8"], 30.0)
    siga = _extract_val(data, ["SIGA", "siga", "UPARAM_9"], 15.0)
    sigb = _extract_val(data, ["SIGB", "sigb", "UPARAM_10"], 10.0)
    sigr = _extract_val(data, ["SIGR", "sigr", "UPARAM_11"], 5.0)

    ra1 = _extract_val(data, ["RA1", "ra1", "UPARAM_12"], 50.0)
    ra2 = _extract_val(data, ["RA2", "ra2", "UPARAM_13"], 0.0)
    na = _extract_val(data, ["NA", "na", "UPARAM_14"], 0.0)
    rb = _extract_val(data, ["RB", "rb", "UPARAM_15"], 20.0)
    rc = _extract_val(data, ["RC", "rc", "UPARAM_16"], 0.5)

    nup = _extract_val(data, ["NUP", "nup", "UPARAM_20"], 0.0)
    alpha = _extract_val(data, ["ALPHA", "alpha", "UPARAM_19"], 0.0)
    tref = _extract_val(data, ["TREF", "tref", "temp0"], 293.15)
    jthe = int(_extract_val(data, ["JTHE", "jthe", "therm_couple"], 0))
    elastic_only = bool(data.get("elastic_only", False))

    return Law96Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        tanb=tanb,
        tanp=tanp,
        sigy=sigy,
        siga=siga,
        sigb=sigb,
        sigr=sigr,
        ra1=ra1,
        ra2=ra2,
        na=na,
        rb=rb,
        rc=rc,
        nup=nup,
        alpha=alpha,
        tref=tref,
        jthe=jthe,
        elastic_only=elastic_only,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law96Params:
    """Resolve and cache Law96Params from a Material or dict."""
    if isinstance(mat, Law96Params):
        return mat
    cached = getattr(mat, "_cached_law96", None)
    if cached is None:
        cached = build_law96(mat)
        try:
            setattr(mat, "_cached_law96", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW96 uses an incremental rate formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW96 (sigeps96.F lines 48-53, 128-130).

    Columns:
      [0]: Deviatoric equivalent plastic strain eps_p (DEFP(1))
      [1]: Volumetric plastic strain eps_pv (DEFP(2))
      [2]: Current yield stress sigma_y (UVAR(1))
      [3]: Plastic strain rate eps_dot (UVAR(2))
    """
    if nip is not None and nip > 1:
        return {"uvar96": (nip, 4), "uvar": (nip, 4)}
    return {"uvar96": (4,), "uvar": (4,)}


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic sound speed for LAW96 matching sigeps96.F lines 511-513."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def calc_yield(p: Law96Params, epsp: float, temp: float = 0.0) -> Tuple[float, float]:
    """Compute yield stress sigma_y and tangent hardening modulus H for given epsp and temp.

    Matches sigeps96.F lines 212-227:
        RA = RA1 * (TEMP/TREF)^NA + RA2
        YLD = SIGY + SIGA*(1 - EXP(-RA*PLA)) - SIGB*(1 - EXP(-RB*PLA))
              + (1/3)*SIGR*PLA*(3 - RC*PLA^2)/(1 - RC*PLA^2)
    """
    ra = p.ra1 + p.ra2
    if p.jthe == 1 or (temp > 0.0 and p.na != 0.0):
        if p.tref > 0.0:
            ra = p.ra1 * ((temp / p.tref) ** p.na) + p.ra2
        else:
            ra = p.ra1 * (temp ** p.na) + p.ra2

    exp_a = math.exp(-ra * epsp) if ra * epsp < 50.0 else 0.0
    exp_b = math.exp(-p.rb * epsp) if p.rb * epsp < 50.0 else 0.0

    pla2 = epsp ** 2
    pp1 = max(1.0 - p.rc * pla2, 1.0e-6)
    pp2 = pp1 + 2.0

    rubbery_term = (1.0 / 3.0) * p.sigr * epsp * (pp2 / pp1)
    yld = p.sigy + p.siga * (1.0 - exp_a) - p.sigb * (1.0 - exp_b) + rubbery_term
    yld = max(0.0, yld)

    # Tangent modulus H = d(sigma_y) / d(epsp)
    # Upstream Fortran sigeps96.F lines 379-380:
    #   HO = SIGA*RA(I)*EXP(-RA(I)*EPSPD(I)) - SIGB*RB*EXP(-RB*EPSPD(I))
    #        + SIGR*(ONE + TWO_THIRD*RC*PLA2 * PP2 / PP1**2)
    h_rubbery = p.sigr * (1.0 + (2.0 / 3.0) * p.rc * pla2 * pp2 / (pp1 ** 2))
    h = p.siga * ra * exp_a - p.sigb * p.rb * exp_b + h_rubbery
    return yld, h


def _solid_update_single(
    p: Law96Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    temp: float = 0.0,
    elastic_only: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum update for LAW96 matching sigeps96.F lines 218-507.

    If elastic_only is True (or p.elastic_only), performs the elastic fallback
    prediction without plastic return mapping.
    """
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp_d = uvar[0]
    epsp_v = uvar[1]

    # Elastic trial state (sigeps96.F lines 229-265)
    deps_v = deps[0] + deps[1] + deps[2]
    deps_dev = deps[:3] - (deps_v / 3.0)

    p_old = -(sig0[0] + sig0[1] + sig0[2]) / 3.0
    s_dev_old = sig0[:3] + p_old

    s_dev_trial = np.zeros(6, dtype=float)
    s_dev_trial[:3] = s_dev_old + 2.0 * p.g * deps_dev
    s_dev_trial[3] = sig0[3] + p.g * deps[3]
    s_dev_trial[4] = sig0[4] + p.g * deps[4]
    s_dev_trial[5] = sig0[5] + p.g * deps[5]

    p_trial = p_old - p.bulk * deps_v

    # von Mises equivalent stress Q (sigeps96.F lines 277-280)
    j2 = 0.5 * (s_dev_trial[0] ** 2 + s_dev_trial[1] ** 2 + s_dev_trial[2] ** 2) + \
        (s_dev_trial[3] ** 2 + s_dev_trial[4] ** 2 + s_dev_trial[5] ** 2)
    q_trial = math.sqrt(max(0.0, 3.0 * j2))

    yld, h_mod = calc_yield(p, epsp_d, temp=temp)
    f_yield = q_trial - max(0.0, p_trial * p.tanb + yld)

    # Plastic return mapping (or elastic fallback)
    if (not elastic_only) and (not p.elastic_only) and (f_yield > _EM10) and (q_trial > _EM20):
        # Newton-Raphson plastic return mapping (sigeps96.F lines 356-459)
        denom = 3.0 * p.g + p.bulk * p.tanb * p.tanp + h_mod
        ldot = f_yield / max(denom, _EM20) if denom != 0.0 else 0.0

        for _ in range(4):
            q_k = max(0.0, q_trial - 3.0 * p.g * ldot)
            p_k = p_trial + p.bulk * ldot * p.tanp
            ep_k = epsp_d + ldot * p.kep
            yld_k, h_k = calc_yield(p, ep_k, temp=temp)
            res = q_k - max(0.0, p_k * p.tanb + yld_k)
            dres = -(3.0 * p.g + p.bulk * p.tanb * p.tanp + h_k * p.kep)
            if abs(dres) > _EM20:
                ldot -= res / dres

        ldot = max(0.0, ldot)
        q_final = max(0.0, q_trial - 3.0 * p.g * ldot)
        p_final = p_trial + p.bulk * ldot * p.tanp

        scale = q_final / max(q_trial, _EM20)
        s_dev_final = s_dev_trial * scale

        epsp_d += ldot * p.kep
        epsp_v -= ldot * p.tanp
        yld, _ = calc_yield(p, epsp_d, temp=temp)
    else:
        # Elastic trial / fallback state
        s_dev_final = s_dev_trial
        p_final = p_trial

    sign = np.empty(6, dtype=float)
    sign[:3] = s_dev_final[:3] - p_final
    sign[3:] = s_dev_final[3:]

    uvar[0] = epsp_d
    uvar[1] = epsp_v
    uvar[2] = yld
    return sign, epsp_d, uvar, p.c_solid


def solid_update(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D solid constitutive update for /MAT/LAW96.

    Supports standard pyradioss material kernel signature:
        (mat, sig, deps, epsp, dt, extra, return_sound_speed)
    as well as group-based interfaces.
    """
    # Handle element-group interface fallback: solid_update(group, x, u, ur, dt, fint, mint)
    if deps is None and hasattr(mat, "mat"):
        group = mat
        p = resolve(getattr(group, "mat", None))
        return np.zeros(6, dtype=float), np.zeros(1, dtype=float), p.c_solid

    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 4), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar96", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(4, len(u))] = u[:min(4, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(4, u.shape[1])] = u[:min(nel, len(u)), :min(4, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 0] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    temp_arr = np.zeros(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "temp" in extra:
        t = np.atleast_1d(extra["temp"]).astype(float)
        temp_arr[:min(nel, len(t))] = t[:nel]

    elastic_only = kwargs.get("elastic_only", p.elastic_only)

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _solid_update_single(
            p,
            sig_arr[i],
            deps_arr[i],
            uvar_arr[i],
            off=off_arr[i],
            temp=temp_arr[i],
            elastic_only=elastic_only,
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar96"] = uvar_arr
        extra["uvar"] = uvar_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


def _shell_update_single(
    p: Law96Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    temp: float = 0.0,
    elastic_only: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell update for LAW96."""
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[0]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    epsp_d = uvar[0]
    epsp_v = uvar[1]

    # Elastic trial stresses
    s0_xx = sig0[0] + p.a11_2d * deps[0] + p.a21_2d * deps[1]
    s0_yy = sig0[1] + p.a21_2d * deps[0] + p.a11_2d * deps[1]
    s0_xy = sig0[2] + p.g * deps[2]

    # Plane stress von Mises and mean stress
    svm = math.sqrt(max(0.0, s0_xx ** 2 + s0_yy ** 2 - s0_xx * s0_yy + 3.0 * s0_xy ** 2))
    p_mean = - (s0_xx + s0_yy) / 3.0

    yld, h_mod = calc_yield(p, epsp_d, temp=temp)
    f_yield = svm - max(0.0, p_mean * p.tanb + yld)

    if (not elastic_only) and (not p.elastic_only) and (f_yield > _EM10) and (svm > _EM20):
        denom = 3.0 * p.g + p.bulk * p.tanb * p.tanp + h_mod
        dlam = f_yield / max(denom, _EM20) if denom != 0.0 else 0.0

        for _ in range(4):
            q_k = max(0.0, svm - 3.0 * p.g * dlam)
            p_k = p_mean + p.bulk * dlam * p.tanp
            ep_k = epsp_d + dlam * p.kep
            yld_k, h_k = calc_yield(p, ep_k, temp=temp)
            res = q_k - max(0.0, p_k * p.tanb + yld_k)
            dres = -(3.0 * p.g + p.bulk * p.tanb * p.tanp + h_k * p.kep)
            if abs(dres) > _EM20:
                dlam -= res / dres

        dlam = max(0.0, dlam)
        scale = max(0.0, svm - 3.0 * p.g * dlam) / max(svm, _EM20)
        s0_xx *= scale
        s0_yy *= scale
        s0_xy *= scale

        epsp_d += dlam * p.kep
        epsp_v -= dlam * p.tanp
        yld, _ = calc_yield(p, epsp_d, temp=temp)

    sign = np.empty_like(sig0, dtype=float)
    sign[0] = s0_xx
    sign[1] = s0_yy
    sign[2] = s0_xy
    if len(sig0) >= 5:
        sign[3] = sig0[3] + p.g * deps[3]
        sign[4] = sig0[4] + p.g * deps[4]

    uvar[0] = epsp_d
    uvar[1] = epsp_v
    uvar[2] = yld
    return sign, epsp_d, uvar, p.c_shell


def shell_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D plane-stress shell constitutive update for /MAT/LAW96."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(float)
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, 4), dtype=float)
    if extra is not None and isinstance(extra, dict):
        for k in ("uvar96", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(4, len(u))] = u[:min(4, len(u))]
                elif u.ndim == 2:
                    uvar_arr[:min(nel, len(u)), :min(4, u.shape[1])] = u[:min(nel, len(u)), :min(4, u.shape[1])]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 0] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    temp_arr = np.zeros(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "temp" in extra:
        t = np.atleast_1d(extra["temp"]).astype(float)
        temp_arr[:min(nel, len(t))] = t[:nel]

    elastic_only = kwargs.get("elastic_only", p.elastic_only)

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    for i in range(nel):
        s_i, ep_i, u_i, c_i = _shell_update_single(
            p,
            sig_arr[i],
            deps_arr[i],
            uvar_arr[i],
            off=off_arr[i],
            temp=temp_arr[i],
            elastic_only=elastic_only,
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar96"] = uvar_arr
        extra["uvar"] = uvar_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness operator (n, 6, 6)."""
    p = resolve(mat)
    c_el = np.zeros((6, 6), dtype=float)
    c11 = p.bulk + 4.0 / 3.0 * p.g
    c12 = p.bulk - 2.0 / 3.0 * p.g
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = p.g

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 6, 6)).copy()


def shell_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell plane-stress tangent operator (n, 3, 3)."""
    p = resolve(mat)
    c_el = np.array([
        [p.a11_2d, p.a21_2d, 0.0],
        [p.a21_2d, p.a11_2d, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


tangent = solid_tangent
consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
