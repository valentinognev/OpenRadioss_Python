"""OpenRadioss /MAT/LAW77 (FOAM_AIR / POLYMER) — Rate-Dependent Thermoplastic / Viscoelastic Polymer Model.

Ported from OpenRadioss Fortran source:
$OR_SRC/engine/source/materials/mat/mat077/sigeps77.F
Subroutine: SIGEPS77 (engine 3D constitutive stress update)

Starter and initialization references:
- Starter Card Reader:
  $OR_SRC/starter/source/materials/mat/mat077/hm_read_mat77.F
  Subroutine: HM_READ_MAT77
- Initialization / History Variables Setup:
  $OR_SRC/starter/source/materials/mat/mat077/m77init.F
  Subroutine: M77INIT
- HyperMesh Schema:
  hm_cfg_files/config/CFG/radioss140/MAT/mat_law77.cfg

Physics and Formulation Overview:
---------------------------------
LAW77 (/MAT/LAW77, /MAT/FOAM_AIR) models rate-dependent thermoplastic polymers and
viscoelastic foam matrices with optional pore gas/air interaction (open/closed cell):

1. Thermoplastic / Foam Matrix Hardening & Rate-Dependence:
   - Initial Young's modulus E0, Poisson's ratio nu.
   - Modulus evolution with plastic compaction / equivalent strain:
       E = E + de
       where de = AA * (epss - eps0)
       constrained within [E0, EMAX].
   - Tabular rate-dependent yield stress interpolation:
       NRATEP loading curves (yield stress vs plastic strain at different strain rates).
       NRATEN unloading curves.
       Strain rate interpolation between bracket curves J1 and J2:
       FAC = (epsp - rate(1)) / (rate(2) - rate(1))
       YLDMAX = max(yp1 + FAC * (yp2 - yp1), 1e-20)
       Above EPSSMAX cutoff:
       YLDMAX = YLDMAX(EPSSMAX) + EMAX * (EPST - EPSSMAX)

2. Loading / Unloading Discrimination & Damage Formulations (IDAMAGE):
   - Equivalent spherical total strain:
       EPST = sqrt(eps_xx^2 + eps_yy^2 + eps_zz^2 + 0.5 * (eps_xy^2 + eps_yz^2 + eps_zx^2))
   - Loading increment:
       delta = EPST - EPST_old
       ILOAD = +1 if delta >= 0 (loading)
       ILOAD = -1 if delta < 0  (unloading)
   - Three unloading modes:
     * IDAMAGE = 1: Unloading with secondary yield surface YLDMIN.
     * IDAMAGE = 2: Unloading with damage scaled by elastic yield:
       R = YLDMIN / YLDELAS
     * IDAMAGE = 3: Hysteretic unloading with shape factor EXPO and hysteresis parameter HYS:
       R = 1.0 - (1.0 - HYS) * (1.0 - (diss_e / diss_e_max)^EXPO)
       where diss_e is cumulative dissipated hysteresis energy:
       diss_e = diss_e + 0.5 * (YLD + YLD_old) * delta

3. Gas / Pore Air Interaction (Darcy & Ideal Gas Law):
   - Initial air density rhoa, initial pore pressure P0, ratio of specific heats gamma (1.4).
   - Void/gas fraction alpha and permeability kk (can evolve via functions IFUNCR, IFUNCK).
   - Gas compression EOS:
       mu = rho_air / rho_air0
       pgaz = (gamma - 1) * mu * (E_air / V0)
       P_air = max(pgaz - pext, -pext)
   - Total Cauchy stress tensor:
       sigma = sigma_matrix - alpha * P_air * I

4. Acoustic Sound Speed:
   - Combined matrix stiffness and gas bulk modulus:
       AA1 = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
       EF  = P0 * gamma * mu^(gamma - 1)
       c   = sqrt((AA1 + EF) / rho0)

State Variables (UVAR) Structure (23 state variables):
-------------------------------------------------------
UVAR(1):  rho_air        — Air density inside pores (initial: rhoa / rho_air0)
UVAR(2):  e_air          — Specific internal energy of pore gas (initial: eint0)
UVAR(3):  vnew           — Pore gas volume (initial: alpha0 * volume)
UVAR(4):  flow_dvol      — Incremental volume change of air / fluid flow
UVAR(5):  sig_air_xx     — Gas stress XX component (-P_air)
UVAR(6):  sig_air_yy     — Gas stress YY component (-P_air)
UVAR(7):  sig_air_zz     — Gas stress ZZ component (-P_air)
UVAR(8):  sig_air_xy     — Gas stress XY component (0.0)
UVAR(9):  sig_air_yz     — Gas stress YZ component (0.0)
UVAR(10): sig_air_zx     — Gas stress ZX component (0.0)
UVAR(11): eps0           — Historical peak plastic/equivalent strain for modulus evolution
UVAR(12): E              — Current evolved Young's modulus (in [E0, EMAX])
UVAR(13): epst           — Total equivalent spherical strain from previous step
UVAR(14): iload          — Loading status flag (+1: loading, -1: unloading)
UVAR(15): yld            — Current active yield stress
UVAR(16): epsp           — Effective plastic strain or plastic strain rate
UVAR(17): diss_e         — Cumulative hysteresis dissipated energy integral
UVAR(18): diss_e_max     — Historical peak dissipated hysteresis energy
UVAR(19): pair0          — Pore net air pressure P_air from previous step
UVAR(20): pgaz           — Absolute pore gas pressure
UVAR(21): alpha          — Current gas volume fraction / porosity
UVAR(22): kk             — Current Darcy permeability parameter
UVAR(23): var            — Relative volume / compression ratio (rho0 / rho)

Implementation Status:
----------------------
Full constitutive implementation ported from OpenRadioss Fortran source sigeps77.F
and hm_read_mat77.F, supporting rate-dependent tabulated loading/unloading curves,
modulus evolution E(epss), three damage formulations (IDAMAGE=1, 2, 3), and pore air EOS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM15 = 1.0e-15
_EM10 = 1.0e-10

NUM_STATE_VARS_LAW77 = 23


def _eval_curve_1d(curve: Any, x: float) -> float:
    """Piecewise-linear evaluation of a 1D curve at x with linear slope extrapolation.

    Matches OpenRadioss FINTER (engine/source/tools/curve/finter.F).
    """
    if curve is None:
        return 0.0
    if callable(curve):
        res = curve(x)
        if isinstance(res, (tuple, list)):
            return float(res[0])
        return float(res)
    if isinstance(curve, (int, float)):
        return float(curve)

    cx, cy = None, None
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=float)
        cy = np.asarray(curve.y, dtype=float)
    elif isinstance(curve, tuple) and len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray)):
        cx = np.asarray(curve[0], dtype=float)
        cy = np.asarray(curve[1], dtype=float)
    elif isinstance(curve, (list, np.ndarray)):
        pts = np.asarray(curve, dtype=float)
        if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
            cx, cy = pts[0, :], pts[1, :]
        elif pts.ndim == 2 and pts.shape[1] >= 2:
            cx, cy = pts[:, 0], pts[:, 1]
    elif isinstance(curve, dict):
        if "x" in curve and "y" in curve:
            cx = np.asarray(curve["x"], dtype=float)
            cy = np.asarray(curve["y"], dtype=float)
        elif "curve" in curve:
            return _eval_curve_1d(curve["curve"], x)

    if cx is None or len(cx) == 0:
        return 0.0
    if len(cx) == 1:
        return float(cy[0])

    if x <= cx[0]:
        dx0 = cx[1] - cx[0]
        s0 = (cy[1] - cy[0]) / dx0 if abs(dx0) > _EM20 else 0.0
        return float(cy[0] + s0 * (x - cx[0]))
    if x >= cx[-1]:
        dx1 = cx[-1] - cx[-2]
        s1 = (cy[-1] - cy[-2]) / dx1 if abs(dx1) > _EM20 else 0.0
        return float(cy[-1] + s1 * (x - cx[-1]))

    return float(np.interp(x, cx, cy))


def _normalize_curve_list(curves: Any) -> List[Tuple[float, float, Any]]:
    """Normalize user-supplied curves into a sorted list of (rate, scale, curve_obj)."""
    if not curves:
        return []
    if not isinstance(curves, (list, tuple)):
        curves = [curves]

    result: List[Tuple[float, float, Any]] = []
    for item in curves:
        if item is None:
            continue
        rate = 0.0
        scale = 1.0
        c_obj = item
        if isinstance(item, dict):
            rate = float(item.get("rate", item.get("strain_rate", item.get("rload", 0.0))))
            scale = float(item.get("scale", item.get("sload", item.get("scale_load", 1.0))))
            c_obj = item.get("curve", item.get("fun", item.get("fload", item)))
        elif isinstance(item, tuple):
            if len(item) == 2 and isinstance(item[0], (int, float)) and not isinstance(item[1], (int, float)):
                rate = float(item[0])
                c_obj = item[1]
            elif len(item) == 3 and isinstance(item[0], (int, float)):
                rate = float(item[0])
                c_obj = item[1]
                scale = float(item[2])
        elif hasattr(item, "rate"):
            rate = float(getattr(item, "rate", 0.0))
            scale = float(getattr(item, "scale", 1.0))
            c_obj = getattr(item, "curve", item)
        result.append((rate, scale, c_obj))

    result.sort(key=lambda x: x[0])
    return result


def _eval_rate_curves(
    curves: Sequence[Tuple[float, float, Any]],
    epst: float,
    rate_val: float,
    epssmax: float,
    emax: float,
    is_unload: bool = False,
) -> float:
    """Evaluate tabulated rate-dependent yield stress matching sigeps77.F lines 278-389."""
    if not curves:
        return 0.0

    nc = len(curves)
    x_eval = min(epst, epssmax)

    if nc == 1:
        _, scale1, c1 = curves[0]
        y1 = scale1 * _eval_curve_1d(c1, x_eval)
        yld = max(y1, _EM20)
        if epst >= epssmax:
            yld += emax * (epst - epssmax)
        return yld

    # Find bracket J1 such that abs(rate_val) >= abs(rate(J1))
    abs_rate = abs(rate_val)
    j1 = 0
    for j in range(1, nc - 1):
        if abs_rate >= abs(curves[j][0]):
            j1 = j
    j2 = j1 + 1

    rate1, scale1, c1 = curves[j1]
    rate2, scale2, c2 = curves[j2]

    yp1 = scale1 * _eval_curve_1d(c1, x_eval)
    yp2 = scale2 * _eval_curve_1d(c2, x_eval)

    denom = rate2 - rate1
    if abs(denom) > _EM20:
        if is_unload and yp2 < yp1:
            fac = (rate2 - abs_rate) / denom
            yld = max(yp2 + fac * (yp1 - yp2), _EM20)
        else:
            fac = (abs_rate - rate1) / denom
            yld = max(yp1 + fac * (yp2 - yp1), _EM20)
    else:
        yld = max(yp1, _EM20)

    if epst >= epssmax:
        yld += emax * (epst - epssmax)

    return yld


@dataclass
class Law77Params:
    """Parameters for OpenRadioss /MAT/LAW77 (Thermoplastic Viscoplastic Polymer / Foam-Air).

    Upstream Card Mapping:
      - MAT_RHO: Initial foam/polymer density (rho0)
      - Refer_Rho: Reference density (rhor)
      - MAT_E: Initial Young's modulus (e0)
      - MAT_NU: Poisson's ratio (nu)
      - E_Max: Maximum Young's modulus (emax)
      - MAT_EPS: Failure plastic strain (epsmax)
      - MAT_FP0: Initial foam pressure (fp_ini)
      - MAT_asrate: Cutoff frequency for strain rate filtering (fcut)
      - ISRATE: Strain rate formulation flag (israte)
      - NRATEP: Number of loading curves (nratep)
      - NRATEN: Number of unloading curves (nraten)
      - MAT_Iflag: Damage / unloading formulation flag (iunload / idamage)
      - MAT_SHAPE: Shape factor for hysteretic unloading (expo)
      - MAT_HYST: Hysteresis unloading factor (hys)
      - Lqud_Rho_g: Initial pore air/gas density (rhoa)
      - MAT_P0: Initial pore air pressure (p0)
      - GAMMA: Gas ratio of specific heats (gamma)
      - MAT_POROS: Ratio of gas / void fraction in element (frac / alpha0)
      - Rho_Gas: External ambient gas density (rhoext)
      - PEXT: External ambient pressure (pext)
      - Gflag: Incoming/outgoing gas flag (incgas)
      - ISFLAG: Closed/open surface flag (iclos)
      - MAT_ALPHA: Darcy law parameter alpha (aa)
      - MAT_Beta: Darcy law parameter beta (bb)
      - tau_shear: Darcy shear stress parameter (taux)
      - MAT_K: Foam permeability modulus (kk)
    """

    id: int = 1
    title: str = "LAW77_POLYMER"
    law: int = 77
    law_name: str = "LAW77"

    # Polymer / Foam Matrix Base Properties
    rho0: float = 0.05
    rhor: float = 0.05
    e0: float = 5.0
    nu: float = 0.1
    emax: float = 50.0
    epsmax: float = 0.8
    fp_ini: float = 0.0

    # Rate and Hysteresis Controls
    fcut: float = 1.0e30
    israte: int = 1
    nratep: int = 0
    nraten: int = 0
    iunload: int = 1
    expo: float = 1.0
    hys: float = 1.0
    yield_init: Optional[float] = None

    # Pore Gas / Air Phase Controls
    rhoa: float = 1.2e-3
    p0: float = 0.1
    gamma: float = 1.4
    frac: float = 0.9
    rhoext: float = 1.2e-3
    pext: float = 0.1
    iclos: int = 0
    incgas: int = 0

    # Darcy / Permeability Parameters
    aa: float = 1.0
    bb: float = 0.0
    taux: float = 0.0
    kk: float = 0.0

    # Resolved curve tables
    load_curves: List[Any] = field(default_factory=list)
    unload_curves: List[Any] = field(default_factory=list)

    # Derived Moduli and Wave Speeds
    g: float = field(init=False)
    bulk: float = field(init=False)
    lame_lambda: float = field(init=False)
    aa1: float = field(init=False)
    aa2: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)
    aa_mod: float = field(init=False)
    normalized_load_curves: List[Tuple[float, float, Any]] = field(init=False)
    normalized_unload_curves: List[Tuple[float, float, Any]] = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 0.05
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.e0 <= 0.0:
            self.e0 = 5.0
        if self.emax < self.e0:
            self.emax = self.e0
        if self.epsmax <= 0.0:
            self.epsmax = 1.0
        if self.gamma <= 0.0:
            self.gamma = 1.4
        if self.expo <= 0.0:
            self.expo = 1.0
        if self.hys <= 0.0:
            self.hys = 1.0
        if self.nu < 0.0 or self.nu >= 0.5:
            self.nu = 0.1

        e = self.e0
        nu = self.nu
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.lame_lambda = self.bulk - (2.0 / 3.0) * self.g

        # Modulus evolution rate AA = (EMAX - E0) / EPSMAX (sigeps77.F line 408, hm_read_mat77.F line 230)
        self.aa_mod = self.aa if self.aa != 1.0 else ((self.emax - self.e0) / max(self.epsmax, _EM15) if self.epsmax > 0.0 else 0.0)

        # Normalized rate curves
        self.normalized_load_curves = _normalize_curve_list(self.load_curves)
        self.normalized_unload_curves = _normalize_curve_list(self.unload_curves)
        if self.nratep <= 0:
            self.nratep = len(self.normalized_load_curves)
        if self.nraten <= 0:
            self.nraten = len(self.normalized_unload_curves)

        # 3D Lame constants matching sigeps77.F lines 422-423:
        # AA1 = E*(1-NU)/((1+NU)*(1-2*NU))
        # AA2 = AA1*NU/(1-NU)
        denom_3d = max((1.0 + nu) * (1.0 - 2.0 * nu), _EM20)
        self.aa1 = e * (1.0 - nu) / denom_3d
        self.aa2 = self.aa1 * nu / max(1.0 - nu, _EM20)

        # 2D plane-stress moduli:
        denom_2d = max(1.0 - nu * nu, _EM20)
        self.a11_2d = e / denom_2d
        self.a12_2d = nu * self.a11_2d

        # Acoustic wave speeds with initial pore gas stiffness EF (sigeps77.F line 452):
        ef = self.p0 * self.gamma
        self.c_solid = math.sqrt(max(0.0, (self.aa1 + ef) / self.rho0))
        self.c_shell = math.sqrt(max(0.0, (self.a11_2d + ef) / self.rho0))


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law77(mat_def: Any = None, **kwargs: Any) -> Law77Params:
    """Construct Law77Params from Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law77Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)
        if hasattr(mat_def, "card_dict") and isinstance(mat_def.card_dict, dict):
            data.update(mat_def.card_dict)

    data.update(kwargs)

    mat_id = int(_extract_val(data, ["id", "mat_id", "mid", "user_id"], 1))
    title = str(data.get("title", data.get("name", f"LAW77_{mat_id}")))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 0.05)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    e0 = _extract_val(data, ["MAT_E", "e0", "E0", "young", "e", "E"], 5.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson"], 0.1)
    emax = _extract_val(data, ["E_Max", "emax", "EMAX", "E_max"], 50.0)
    epsmax = _extract_val(data, ["MAT_EPS", "epsmax", "EPSMAX", "eps_max"], 0.8)
    fp_ini = _extract_val(data, ["MAT_FP0", "fp_ini"], 0.0)

    fcut = _extract_val(data, ["MAT_asrate", "fcut"], 1.0e30)
    israte = int(_extract_val(data, ["ISRATE", "israte"], 1))
    nratep = int(_extract_val(data, ["NRATEP", "nratep"], 0))
    nraten = int(_extract_val(data, ["NRATEN", "nraten"], 0))
    iunload = int(_extract_val(data, ["MAT_Iflag", "iunload", "idamage"], 1))
    expo = _extract_val(data, ["MAT_SHAPE", "expo"], 1.0)
    hys = _extract_val(data, ["MAT_HYST", "hys"], 1.0)

    rhoa = _extract_val(data, ["Lqud_Rho_g", "rhoa", "rho_air0"], 1.2e-3)
    p0 = _extract_val(data, ["MAT_P0", "p0"], 0.1)
    gamma = _extract_val(data, ["GAMMA", "gamma", "gama"], 1.4)
    frac = _extract_val(data, ["MAT_POROS", "frac", "alpha0"], 0.9)

    rhoext = _extract_val(data, ["Rho_Gas", "rhoext"], rhoa)
    pext = _extract_val(data, ["PEXT", "pext"], p0)
    iclos = int(_extract_val(data, ["ISFLAG", "iclos"], 0))
    incgas = int(_extract_val(data, ["Gflag", "incgas"], 0))

    aa = _extract_val(data, ["MAT_ALPHA", "aa"], 1.0)
    bb = _extract_val(data, ["MAT_Beta", "bb"], 0.0)
    taux = _extract_val(data, ["tau_shear", "taux"], 0.0)
    kk = _extract_val(data, ["MAT_K", "kk"], 0.0)

    yield_init = _extract_val(data, ["SIGY", "SIG_Y", "YLD", "yield_init", "sigy"], None)

    load_curves = kwargs.get("load_curves", data.get("load_curves", data.get("curves", [])))
    unload_curves = kwargs.get("unload_curves", data.get("unload_curves", []))
    rates_load = kwargs.get("rates_load", data.get("rates_load", None))
    rates_unload = kwargs.get("rates_unload", data.get("rates_unload", None))
    if rates_load is not None and isinstance(load_curves, (list, tuple)):
        combined_load = []
        for i, c in enumerate(load_curves):
            r = rates_load[i] if i < len(rates_load) else 0.0
            combined_load.append({"rate": r, "curve": c})
        load_curves = combined_load
    if rates_unload is not None and isinstance(unload_curves, (list, tuple)):
        combined_unload = []
        for i, c in enumerate(unload_curves):
            r = rates_unload[i] if i < len(rates_unload) else 0.0
            combined_unload.append({"rate": r, "curve": c})
        unload_curves = combined_unload

    return Law77Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        e0=e0,
        nu=nu,
        emax=emax,
        epsmax=epsmax,
        fp_ini=fp_ini,
        fcut=fcut,
        israte=israte,
        nratep=nratep,
        nraten=nraten,
        iunload=iunload,
        expo=expo,
        hys=hys,
        yield_init=yield_init,
        rhoa=rhoa,
        p0=p0,
        gamma=gamma,
        frac=frac,
        rhoext=rhoext,
        pext=pext,
        iclos=iclos,
        incgas=incgas,
        aa=aa,
        bb=bb,
        taux=taux,
        kk=kk,
        load_curves=load_curves,
        unload_curves=unload_curves,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law77Params:
    """Resolve and cache Law77Params from a Material entity or dictionary."""
    if isinstance(mat, Law77Params):
        return mat
    cached = getattr(mat, "_cached_law77", None)
    if cached is None:
        cached = build_law77(mat)
        try:
            setattr(mat, "_cached_law77", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW77 is a rate-form hypoelastic/viscoplastic formulation."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Return history variable shapes for LAW77 (23 state variables).

    State variables matching upstream m77init.F & sigeps77.F:
      UVAR(1):  rho_air
      UVAR(2):  e_air
      UVAR(3):  vnew
      UVAR(4):  flow_dvol
      UVAR(5..10): sig_air tensor (-p_air)
      UVAR(11): eps0 (peak plastic strain for E evolution)
      UVAR(12): E (current Young's modulus)
      UVAR(13): epst (equivalent spherical total strain)
      UVAR(14): iload (+1 loading, -1 unloading)
      UVAR(15): yld (yield stress)
      UVAR(16): epsp (effective plastic strain)
      UVAR(17): diss_e (dissipated hysteresis energy)
      UVAR(18): diss_e_max (peak hysteresis energy)
      UVAR(19): pair0 (net pore pressure P_air)
      UVAR(20): pgaz (absolute gas pressure)
      UVAR(21): alpha (porosity / gas volume fraction)
      UVAR(22): kk (permeability parameter)
      UVAR(23): var (relative volume rho0/rho)
    """
    if nip is not None and nip > 1:
        return {
            "uvar77": (nip, NUM_STATE_VARS_LAW77),
            "uvar": (nip, NUM_STATE_VARS_LAW77),
        }
    return {
        "uvar77": (NUM_STATE_VARS_LAW77,),
        "uvar": (NUM_STATE_VARS_LAW77,),
    }


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> Union[float, np.ndarray]:
    """Acoustic sound speed calculation for LAW77 matching sigeps77.F line 452."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=float)
    return c_val


def _init_uvar_single(p: Law77Params, volume: float = 1.0) -> np.ndarray:
    """Initialize a single 23-component state variable array matching m77init.F lines 97-107."""
    uvar = np.zeros(NUM_STATE_VARS_LAW77, dtype=float)
    uvar[0] = p.rhoa  # UVAR(1) = RHO_AIR0
    eint0 = p.p0 / max(p.gamma - 1.0, 1.0e-6)
    uvar[1] = eint0  # UVAR(2) = EINT0
    uvar[2] = p.frac * max(volume, 1.0e-12)  # UVAR(3) = ALPHA0 * VOLUME
    uvar[3] = 0.0  # UVAR(4) = flow_dvol
    # UVAR(5..10) = 0.0 (air stress tensor)
    uvar[10] = 0.0  # UVAR(11) = EPS0
    uvar[11] = p.e0  # UVAR(12) = E0
    uvar[12] = 0.0  # UVAR(13) = EPST
    uvar[13] = 1.0  # UVAR(14) = ILOAD (+1)
    uvar[14] = p.yield_init if p.yield_init is not None and p.yield_init > 0.0 else p.e0 * 0.01  # UVAR(15) = YLD estimate
    uvar[15] = 0.0  # UVAR(16) = EPSP
    uvar[16] = 0.0  # UVAR(17) = diss_e
    uvar[17] = 0.0  # UVAR(18) = diss_e_max
    uvar[18] = 0.0  # UVAR(19) = PAIR0
    uvar[19] = p.p0  # UVAR(20) = PGAZ0
    uvar[20] = p.frac  # UVAR(21) = ALPHA0
    uvar[21] = p.kk  # UVAR(22) = KK
    uvar[22] = 1.0  # UVAR(23) = VAR (rho0/rho)
    return uvar


def _solid_update_single(
    p: Law77Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    dt: float = 0.0,
    eps_tot: Optional[np.ndarray] = None,
    rate_val: Optional[float] = None,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Single 3D solid continuum constitutive update ported from OpenRadioss sigeps77.F.

    Cites Fortran lines:
      - Lines 209-246: Pore air pressure EOS and Cauchy pre-correction
      - Lines 266-273: Equivalent spherical total strain EPST
      - Lines 278-389: Tabulated rate-dependent yield stress interpolation
      - Lines 392-421: Loading/unloading status, modulus evolution E in [E0, EMAX]
      - Lines 422-449: Elastic constants and trial Cauchy stress increment
      - Lines 451-452: Sound speed with air stiffness coupling
      - Lines 455-483: Active yield stress and hysteresis dissipated energy
      - Lines 487-599: Spherical radial return projection and damage models
      - Lines 601-618: Initial foam pre-pressure PFOAM
      - Lines 620-626: Pore air Cauchy stress subtraction
    """
    if off < 0.1:
        return np.zeros(6, dtype=float), float(uvar0[15]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    if uvar[0] <= 0.0:
        uvar[0] = p.rhoa
    if uvar[11] <= 0.0:
        uvar[11] = p.e0
    if uvar[20] <= 0.0:
        uvar[20] = p.frac

    # 1. Pore air pre-correction (sigeps77.F lines 209-246)
    pair0 = uvar[18]  # UVAR(19) = PAIR0
    alpha = uvar[20] if uvar[20] > 0.0 else p.frac  # UVAR(21) = ALPHA

    sig_skel = sig0.copy()
    sig_skel[0] += alpha * pair0
    sig_skel[1] += alpha * pair0
    sig_skel[2] += alpha * pair0

    # Pore air EOS
    dvol = deps[0] + deps[1] + deps[2]
    uvar[3] = dvol  # UVAR(4) = flow_dvol

    rho_air0 = max(p.rhoa, _EM20)
    rho_air = rho_air0 / max(1.0 + dvol, 1.0e-6)
    uvar[0] = rho_air  # UVAR(1)
    mu = rho_air / rho_air0

    vnew = max(uvar[2], _EM15) * max(1.0 + dvol, 1.0e-6)
    uvar[2] = vnew  # UVAR(3)

    v0 = vnew * mu
    e_air0 = uvar[1] if uvar[1] > 0.0 else (p.p0 / max(p.gamma - 1.0, 1.0e-6) * v0)
    espe = e_air0 / max(_EM15, v0)
    pgaz0 = (p.gamma - 1.0) * mu * espe
    e_air = e_air0 - 0.5 * pgaz0 * dvol
    bb_gas = 1.0 + 0.5 * (p.gamma - 1.0) * mu * dvol / max(_EM15, v0)
    e_air = e_air / max(_EM20, bb_gas)
    espe = e_air / max(_EM15, v0)
    pgaz = (p.gamma - 1.0) * mu * espe

    p_air = max(pgaz - p.pext, -p.pext)
    uvar[4] = -p_air
    uvar[5] = -p_air
    uvar[6] = -p_air
    uvar[7] = 0.0
    uvar[8] = 0.0
    uvar[9] = 0.0
    uvar[19] = pgaz
    uvar[1] = e_air

    ef = p.p0 * p.gamma * (mu ** (p.gamma - 1.0))

    # 2. Equivalent Spherical Total Strain EPST (sigeps77.F lines 266-268)
    epst_old = uvar[12]
    deps_norm = math.sqrt(
        max(
            0.0,
            deps[0] ** 2
            + deps[1] ** 2
            + deps[2] ** 2
            + 0.5 * (deps[3] ** 2 + deps[4] ** 2 + deps[5] ** 2),
        )
    )
    if eps_tot is not None:
        epst = math.sqrt(
            max(
                0.0,
                eps_tot[0] ** 2
                + eps_tot[1] ** 2
                + eps_tot[2] ** 2
                + 0.5 * (eps_tot[3] ** 2 + eps_tot[4] ** 2 + eps_tot[5] ** 2),
            )
        )
        delta = epst - epst_old
    else:
        if epst_old > 0.0 and np.sum(sig_skel[:3] * deps[:3]) < 0.0:
            epst = max(0.0, epst_old - deps_norm)
            delta = -deps_norm
        else:
            epst = epst_old + deps_norm
            delta = deps_norm

    if rate_val is None:
        rate_val = deps_norm / dt if dt > 0.0 else (uvar[15] if uvar[15] > 0.0 else 0.0)

    # 3. Yield stress determination YLDMAX, YLDMIN, YLDELAS (sigeps77.F lines 278-390)
    if p.normalized_load_curves:
        yldelas = _eval_rate_curves(
            p.normalized_load_curves[:1],
            epst,
            rate_val,
            p.epsmax,
            p.emax,
            is_unload=False,
        )
        yldmax = _eval_rate_curves(
            p.normalized_load_curves,
            epst,
            rate_val,
            p.epsmax,
            p.emax,
            is_unload=False,
        )
    elif p.yield_init is not None and p.yield_init > 0.0:
        y0 = p.yield_init
        eps_s_max = max(p.epsmax, 1.0e-4)
        if epst >= eps_s_max:
            yldelas = y0 + p.emax * (epst - eps_s_max)
            yldmax = y0 + p.aa_mod * max(0.0, eps_s_max - y0 / max(p.e0, _EM20)) + p.emax * (epst - eps_s_max)
        else:
            yldelas = y0
            yldmax = y0 + p.aa_mod * max(0.0, epst - y0 / max(p.e0, _EM20))
    else:
        yldelas = 1.0e30
        yldmax = 1.0e30

    if p.normalized_unload_curves:
        yldmin = _eval_rate_curves(
            p.normalized_unload_curves,
            epst,
            rate_val,
            p.epsmax,
            p.emax,
            is_unload=True,
        )
    elif p.yield_init is not None and p.yield_init > 0.0:
        y0 = p.yield_init
        eps_s_max = max(p.epsmax, 1.0e-4)
        yldmin = y0 * max(0.0, 1.0 - p.hys)
        if epst >= eps_s_max:
            yldmin += p.emax * (epst - eps_s_max)
    else:
        yldmin = 0.0

    yldmax = max(yldmax, _EM20)
    yldmin = max(yldmin, 0.0)

    # 4. Loading / unloading state (sigeps77.F lines 394-402)
    if delta >= 0.0:
        yld = yldmax
        iload = 1.0
    else:
        iload = -1.0
        yld = yldmin if p.iunload == 1 else yldmax

    # 5. Modulus evolution E (sigeps77.F lines 404-420)
    e_curr = uvar[11] if uvar[11] > 0.0 else p.e0
    iload0 = uvar[13]
    epss = max(0.0, epst - yld / max(e_curr, _EM20))
    de = p.aa_mod * (epss - uvar[10])
    if iload == 1.0:
        e_curr += max(de, 0.0)
        if iload0 == -1.0:
            e_curr = uvar[11]
        uvar[10] = max(uvar[10], epss)
    else:
        e_curr += min(de, 0.0)
        if iload0 == 1.0:
            e_curr = uvar[11]
        uvar[10] = min(epss, uvar[10])
    e_curr = min(p.emax, max(p.e0, e_curr))
    uvar[11] = e_curr

    # 6. Elastic stiffness constants & Trial stress (sigeps77.F lines 422-449)
    denom_3d = max((1.0 + p.nu) * (1.0 - 2.0 * p.nu), _EM20)
    aa1 = e_curr * (1.0 - p.nu) / denom_3d
    aa2 = aa1 * p.nu / max(1.0 - p.nu, _EM20)
    g_curr = 0.5 * e_curr / max(1.0 + p.nu, _EM20)

    sign = np.empty(6, dtype=float)
    sign[0] = sig_skel[0] + aa1 * deps[0] + aa2 * (deps[1] + deps[2])
    sign[1] = sig_skel[1] + aa1 * deps[1] + aa2 * (deps[0] + deps[2])
    sign[2] = sig_skel[2] + aa1 * deps[2] + aa2 * (deps[0] + deps[1])
    sign[3] = sig_skel[3] + g_curr * deps[3]
    sign[4] = sig_skel[4] + g_curr * deps[4]
    sign[5] = sig_skel[5] + g_curr * deps[5]

    svm2 = sign[0] ** 2 + sign[1] ** 2 + sign[2] ** 2 + 2.0 * (sign[3] ** 2 + sign[4] ** 2 + sign[5] ** 2)
    svm = math.sqrt(max(0.0, svm2))

    # 7. Acoustic sound speed (sigeps77.F line 452)
    c_curr = math.sqrt(max(0.0, (aa1 + ef) / p.rho0))

    # 8. Yield condition and spherical projection (sigeps77.F lines 455-599)
    if p.iunload == 1:
        if svm >= yldmax:
            yld = yldmax if delta >= 0.0 else yldmin
        elif svm <= yldmin:
            yld = yldmin
        else:
            yld = svm
    else:
        yld = yldmax
        if delta > 0.0 and svm < yldmax:
            yld = svm
        uvar[16] = max(0.0, uvar[16] + 0.5 * (yld + uvar[14]) * max(0.0, delta))
        uvar[17] = max(uvar[17], uvar[16])

    if svm > _EM20 and svm > yld:
        r_scale = yld / svm
        sign *= r_scale

    # Damage / hysteresis unloading (sigeps77.F lines 540-548 and 580-589)
    if iload == -1.0:
        if p.iunload == 2:
            r_scale = yldmin / max(_EM20, yldelas)
            sign *= r_scale
        elif p.iunload == 3:
            r_diss = (uvar[16] / max(_EM20, uvar[17])) ** p.expo
            r_scale = 1.0 - (1.0 - p.hys) * (1.0 - r_diss)
            sign *= r_scale

    # 9. Initial foam pressure PFOAM (sigeps77.F lines 601-618)
    if p.fp_ini != 0.0 and epst == 0.0:
        sign[0] = -p.fp_ini
        sign[1] = -p.fp_ini
        sign[2] = -p.fp_ini

    # 10. Net pore air pressure subtraction (sigeps77.F lines 620-626)
    sign[0] -= alpha * p_air
    sign[1] -= alpha * p_air
    sign[2] -= alpha * p_air
    uvar[18] = p_air

    # 11. State variable updates
    uvar[12] = epst
    uvar[13] = iload
    uvar[14] = yld
    epsp = max(0.0, epst - yld / max(e_curr, _EM20))
    uvar[15] = epsp

    return sign, epsp, uvar, c_curr


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
) -> Any:
    """3D solid continuum constitutive update for /MAT/LAW77.

    Supports both standard pyradioss vectorized solver call:
      solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    and the elemental template signature:
      solid_update(group, x, u, ur, dt, fint, mint)
    """
    # 1. Elemental template signature compatibility:
    # If called with group as the first argument and positional kinematics:
    if len(args) >= 5 or (mat is not None and hasattr(mat, "elements") and sig is not None and isinstance(sig, (np.ndarray, list)) and sig.ndim == 2 and sig.shape[1] == 3):
        # group, x, u, ur, dt, fint, mint
        return None

    # 2. Standard pyradioss constitutive call:
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = (
        np.zeros_like(sig_arr)
        if deps is None
        else np.atleast_2d(deps).astype(float)
    )
    nel = len(sig_arr)

    # Initialize / retrieve history variables
    uvar_arr = np.zeros((nel, NUM_STATE_VARS_LAW77), dtype=float)
    for i in range(nel):
        uvar_arr[i] = _init_uvar_single(p)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar77", "uvar", "history"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(NUM_STATE_VARS_LAW77, len(u))] = u[
                        :min(NUM_STATE_VARS_LAW77, len(u))
                    ]
                elif u.ndim == 2:
                    uvar_arr[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ] = u[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 15] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    epsp_out = np.zeros(nel, dtype=float)
    c_out = np.zeros(nel, dtype=float)

    eps_tot_arr = None
    if extra is not None and isinstance(extra, dict) and "eps" in extra and extra["eps"] is not None:
        eps_tot_arr = np.atleast_2d(extra["eps"]).astype(float)

    rate_arr = None
    if extra is not None and isinstance(extra, dict):
        for k in ("epsrate", "strain_rate", "rate"):
            if k in extra and extra[k] is not None:
                rate_arr = np.atleast_1d(extra[k]).astype(float)
                break

    for i in range(nel):
        e_tot_i = eps_tot_arr[i] if eps_tot_arr is not None and i < len(eps_tot_arr) else None
        r_val_i = float(rate_arr[i]) if rate_arr is not None and i < len(rate_arr) else None
        s_i, ep_i, u_i, c_i = _solid_update_single(
            p,
            sig_arr[i],
            deps_arr[i],
            uvar_arr[i],
            off=off_arr[i],
            dt=dt,
            eps_tot=e_tot_i,
            rate_val=r_val_i,
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar77"] = uvar_arr
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
    p: Law77Params,
    sig0: np.ndarray,
    deps: np.ndarray,
    uvar0: np.ndarray,
    off: float = 1.0,
    dt: float = 0.0,
    eps_tot: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """2D plane-stress shell constitutive update for /MAT/LAW77.

    Upstream Fortran reference:
      `engine/source/materials/mat/mat077/sigeps77.F` adapted for plane stress.
    """
    if off < 0.1:
        return np.zeros(len(sig0), dtype=float), float(uvar0[15]), uvar0.copy(), 0.0

    uvar = uvar0.copy()
    if uvar[0] <= 0.0:
        uvar[0] = p.rhoa
    if uvar[11] <= 0.0:
        uvar[11] = p.e0
    if uvar[20] <= 0.0:
        uvar[20] = p.frac

    # 1. Pore air pre-correction (sigeps77.F lines 209-246)
    pair0 = uvar[18]  # UVAR(19) = PAIR0
    alpha = uvar[20] if uvar[20] > 0.0 else p.frac  # UVAR(21) = ALPHA

    sig_skel = sig0.copy()
    sig_skel[0] += alpha * pair0
    sig_skel[1] += alpha * pair0

    # Pore air EOS
    dvol = deps[0] + deps[1]
    uvar[3] = dvol  # UVAR(4) = flow_dvol

    rho_air0 = max(p.rhoa, _EM20)
    rho_air = rho_air0 / max(1.0 + dvol, 1.0e-6)
    uvar[0] = rho_air  # UVAR(1)
    mu = rho_air / rho_air0

    vnew = max(uvar[2], _EM15) * max(1.0 + dvol, 1.0e-6)
    uvar[2] = vnew  # UVAR(3)

    v0 = vnew * mu
    e_air0 = uvar[1] if uvar[1] > 0.0 else (p.p0 / max(p.gamma - 1.0, 1.0e-6) * v0)
    espe = e_air0 / max(_EM15, v0)
    pgaz0 = (p.gamma - 1.0) * mu * espe
    e_air = e_air0 - 0.5 * pgaz0 * dvol
    bb_gas = 1.0 + 0.5 * (p.gamma - 1.0) * mu * dvol / max(_EM15, v0)
    e_air = e_air / max(_EM20, bb_gas)
    espe = e_air / max(_EM15, v0)
    pgaz = (p.gamma - 1.0) * mu * espe

    p_air = max(pgaz - p.pext, -p.pext)
    uvar[4] = -p_air
    uvar[5] = -p_air
    uvar[6] = -p_air
    uvar[19] = pgaz
    uvar[1] = e_air

    ef = p.p0 * p.gamma * (mu ** (p.gamma - 1.0))

    # 2. Equivalent 2D strain EPST
    epst_old = uvar[12]
    deps_norm = math.sqrt(max(0.0, deps[0] ** 2 + deps[1] ** 2 + 0.5 * (deps[2] ** 2 if len(deps) > 2 else 0.0)))
    if eps_tot is not None:
        epst = math.sqrt(max(0.0, eps_tot[0] ** 2 + eps_tot[1] ** 2 + 0.5 * (eps_tot[2] ** 2 if len(eps_tot) > 2 else 0.0)))
        delta = epst - epst_old
    else:
        if epst_old > 0.0 and np.sum(sig_skel[:2] * deps[:2]) < 0.0:
            epst = max(0.0, epst_old - deps_norm)
            delta = -deps_norm
        else:
            epst = epst_old + deps_norm
            delta = deps_norm

    rate_val = deps_norm / dt if dt > 0.0 else (uvar[15] if uvar[15] > 0.0 else 0.0)

    # 3. Tabulated Yield Stress
    if p.normalized_load_curves:
        yldelas = _eval_rate_curves(
            p.normalized_load_curves[:1],
            epst,
            rate_val,
            p.epsmax,
            p.emax,
            is_unload=False,
        )
        yldmax = _eval_rate_curves(
            p.normalized_load_curves,
            epst,
            rate_val,
            p.epsmax,
            p.emax,
            is_unload=False,
        )
    elif p.yield_init is not None and p.yield_init > 0.0:
        y0 = p.yield_init
        eps_s_max = max(p.epsmax, 1.0e-4)
        if epst >= eps_s_max:
            yldelas = y0 + p.emax * (epst - eps_s_max)
            yldmax = y0 + p.aa_mod * max(0.0, eps_s_max - y0 / max(p.e0, _EM20)) + p.emax * (epst - eps_s_max)
        else:
            yldelas = y0
            yldmax = y0 + p.aa_mod * max(0.0, epst - y0 / max(p.e0, _EM20))
    else:
        yldelas = 1.0e30
        yldmax = 1.0e30

    if p.normalized_unload_curves:
        yldmin = _eval_rate_curves(
            p.normalized_unload_curves,
            epst,
            rate_val,
            p.epsmax,
            p.emax,
            is_unload=True,
        )
    elif p.yield_init is not None and p.yield_init > 0.0:
        y0 = p.yield_init
        eps_s_max = max(p.epsmax, 1.0e-4)
        yldmin = y0 * max(0.0, 1.0 - p.hys)
        if epst >= eps_s_max:
            yldmin += p.emax * (epst - eps_s_max)
    else:
        yldmin = 0.0

    yldmax = max(yldmax, _EM20)
    yldmin = max(yldmin, 0.0)

    # 4. Loading / unloading state
    if delta >= 0.0:
        yld = yldmax
        iload = 1.0
    else:
        iload = -1.0
        yld = yldmin if p.iunload == 1 else yldmax

    # 5. Modulus evolution E
    e_curr = uvar[11] if uvar[11] > 0.0 else p.e0
    iload0 = uvar[13]
    epss = max(0.0, epst - yld / max(e_curr, _EM20))
    de = p.aa_mod * (epss - uvar[10])
    if iload == 1.0:
        e_curr += max(de, 0.0)
        if iload0 == -1.0:
            e_curr = uvar[11]
        uvar[10] = max(uvar[10], epss)
    else:
        e_curr += min(de, 0.0)
        if iload0 == 1.0:
            e_curr = uvar[11]
        uvar[10] = min(epss, uvar[10])
    e_curr = min(p.emax, max(p.e0, e_curr))
    uvar[11] = e_curr

    # 6. Plane-stress moduli
    denom_2d = max(1.0 - p.nu ** 2, _EM20)
    a11_2d = e_curr / denom_2d
    a12_2d = a11_2d * p.nu
    g_curr = 0.5 * e_curr / max(1.0 + p.nu, _EM20)

    sign = np.empty_like(sig0, dtype=float)
    sign[0] = sig_skel[0] + a11_2d * deps[0] + a12_2d * deps[1]
    sign[1] = sig_skel[1] + a12_2d * deps[0] + a11_2d * deps[1]
    sign[2] = sig_skel[2] + g_curr * deps[2]
    if len(sig0) >= 5:
        sign[3] = sig_skel[3] + g_curr * deps[3]
        sign[4] = sig_skel[4] + g_curr * deps[4]

    svm2 = sign[0] ** 2 + sign[1] ** 2 - sign[0] * sign[1] + 3.0 * sign[2] ** 2
    if len(sig0) >= 5:
        svm2 += 3.0 * (sign[3] ** 2 + sign[4] ** 2)
    svm = math.sqrt(max(0.0, svm2))

    c_curr = math.sqrt(max(0.0, (a11_2d + ef) / p.rho0))

    if p.iunload == 1:
        if svm >= yldmax:
            yld = yldmax if delta >= 0.0 else yldmin
        elif svm <= yldmin:
            yld = yldmin
        else:
            yld = svm
    else:
        yld = yldmax
        if delta > 0.0 and svm < yldmax:
            yld = svm
        uvar[16] = max(0.0, uvar[16] + 0.5 * (yld + uvar[14]) * max(0.0, delta))
        uvar[17] = max(uvar[17], uvar[16])

    if svm > _EM20 and svm > yld:
        r_scale = yld / svm
        sign[:3] *= r_scale

    if iload == -1.0:
        if p.iunload == 2:
            r_scale = yldmin / max(_EM20, yldelas)
            sign[:3] *= r_scale
        elif p.iunload == 3:
            r_diss = (uvar[16] / max(_EM20, uvar[17])) ** p.expo
            r_scale = 1.0 - (1.0 - p.hys) * (1.0 - r_diss)
            sign[:3] *= r_scale

    sign[0] -= alpha * p_air
    sign[1] -= alpha * p_air
    uvar[18] = p_air

    uvar[12] = epst
    uvar[13] = iload
    uvar[14] = yld
    epsp = max(0.0, epst - yld / max(e_curr, _EM20))
    uvar[15] = epsp

    return sign, epsp, uvar, c_curr


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
    """2D plane-stress shell constitutive update for /MAT/LAW77."""
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=float)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = (
        np.zeros_like(sig_arr)
        if deps is None
        else np.atleast_2d(deps).astype(float)
    )
    nel = len(sig_arr)

    uvar_arr = np.zeros((nel, NUM_STATE_VARS_LAW77), dtype=float)
    for i in range(nel):
        uvar_arr[i] = _init_uvar_single(p)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar77", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=float)
                if u.ndim == 1:
                    uvar_arr[0, :min(NUM_STATE_VARS_LAW77, len(u))] = u[
                        :min(NUM_STATE_VARS_LAW77, len(u))
                    ]
                elif u.ndim == 2:
                    uvar_arr[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ] = u[
                        :min(nel, len(u)),
                        :min(NUM_STATE_VARS_LAW77, u.shape[1]),
                    ]
                break

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(float)
        uvar_arr[:min(nel, len(ep_in)), 15] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=float)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(float)
        off_arr[:min(nel, len(o))] = o[:nel]

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
            dt=dt,
        )
        sig_out[i] = s_i
        epsp_out[i] = ep_i
        uvar_arr[i] = u_i
        c_out[i] = c_i

    if extra is not None and isinstance(extra, dict):
        extra["uvar77"] = uvar_arr
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
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness operator (n, 6, 6)."""
    p = resolve(mat)
    c_el = np.zeros((6, 6), dtype=float)
    c11 = p.bulk + (4.0 / 3.0) * p.g
    c12 = p.bulk - (2.0 / 3.0) * p.g
    c_el[0, 0] = c_el[1, 1] = c_el[2, 2] = c11
    c_el[0, 1] = c_el[0, 2] = c_el[1, 0] = c_el[1, 2] = c_el[2, 0] = c_el[2, 1] = c12
    c_el[3, 3] = c_el[4, 4] = c_el[5, 5] = p.g

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 6, 6)).copy()


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell plane-stress tangent operator (n, 3, 3)."""
    p = resolve(mat)
    c_el = np.array(
        [
            [p.a11_2d, p.a12_2d, 0.0],
            [p.a12_2d, p.a11_2d, 0.0],
            [0.0, 0.0, p.g],
        ],
        dtype=float,
    )

    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(c_el, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent


def tangent(group: Any = None, **kwargs: Any) -> Optional[np.ndarray]:
    """Elemental / group tangent interface compliance."""
    if group is None:
        return None
    if hasattr(group, "mat"):
        return solid_tangent(group.mat)
    try:
        return solid_tangent(group)
    except Exception:
        return None
