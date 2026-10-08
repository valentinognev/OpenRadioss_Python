"""Classic Tabulated Failure Model (/FAIL/TAB).

Ported from OpenRadioss Fortran source:
- Solids:
  `$OR_SRC/engine/source/materials/fail/tabulated/fail_tab_old_s.F`
  Subroutine: `FAIL_TAB_OLD_S`
- Shells:
  `$OR_SRC/engine/source/materials/fail/tabulated/fail_tab_old_c.F`
  Subroutine: `FAIL_TAB_OLD_C`
- Starter card reader:
  `$OR_SRC/starter/source/materials/fail/tabulated/hm_read_fail_tab_old.F`
  Subroutine: `HM_READ_FAIL_TAB_OLD`

Failure model index: IRUPT = 37.

Physics & Formulation:
----------------------
Classic tabulated failure criterion with stress-triaxiality-dependent plastic strain
at failure, optional strain-rate sensitivity, element length scaling, temperature scaling,
and non-linear damage accumulation:

1. Stress Triaxiality:
   - Solids (3D continuum):
       P = (sig_xx + sig_yy + sig_zz) / 3
       s_xx = sig_xx - P,  s_yy = sig_yy - P,  s_zz = sig_zz - P
       J_2 = 0.5 * (s_xx^2 + s_yy^2 + s_zz^2) + sig_xy^2 + sig_yz^2 + sig_zx^2
       sigma_vm = sqrt(3 * J_2)
       eta = P / max(sigma_vm, 1e-20)
   - Shells (plane-stress):
       P = (sig_xx + sig_yy) / 3
       sigma_vm = sqrt(sig_xx^2 + sig_yy^2 - sig_xx * sig_yy + 3 * sig_xy^2)
       eta = P / max(sigma_vm, 1e-20)

2. Failure Strain Interpolation:
   - Single curve:
       eps_f = Fscale * f(eta)
   - Multi-rate curves:
       Linear interpolation between adjacent strain rate curves at eps_dot:
       FAC = (eps_dot - rate_1) / (rate_2 - rate_1)
       eps_f = max(EF1 + FAC * (EF2 - EF1), 1e-20)
   - Element length scaling (if fct_IDel > 0):
       lambda = L / EL_REF
       eps_f = eps_f * Fscale_el * f_el(lambda)
   - Temperature scaling (if fct_IDt > 0):
       eps_f = eps_f * FscaleT * f_temp(T*)
   - Constant threshold fallback (if no curve specified):
       eps_f = max(Fscale, eps_f_param, 1e-20)

3. Non-linear Damage Accumulation (sigeps_tab_old lines 167-169):
   - Parameters:
       DN: hardening/softening exponent (default 1.0)
       DD: damage base parameter (default 0.999)
       DCRIT: critical damage threshold (default 1.0)
   - Accumulation exponent:
       If DD > 0 and DN > 0:
           DP = DN * DD^(1 - 1/DN)
       Else:
           DP = 1.0
   - Damage rate:
       dD = DP * max(d_epsp, 0) / max(eps_f, 1e-20)
       D = min(DCRIT, D + dD)
   - Rupture condition:
       Element / layer broken when D >= DCRIT.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

_TINY = 1.0e-20


def _get_param(params: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    """Retrieve first matching numeric parameter value or default."""
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    return default


def _compute_dp(dd: float, dn: float) -> float:
    """Compute DP damage multiplier: DP = DN * DD ** (1 - 1/DN)."""
    if dn <= 0.0 or math.isclose(dn, 1.0, rel_tol=1e-12):
        return 1.0
    if dd <= 0.0:
        return 1.0
    inv_dn = 1.0 / dn
    exp_term = 1.0 - inv_dn
    try:
        return float(dn * (max(dd, _TINY) ** exp_term))
    except (ValueError, OverflowError, ZeroDivisionError):
        return 1.0


def _eval_curve(curve: Any, x: np.ndarray | float) -> np.ndarray:
    """Evaluate a curve or table at abscissa x."""
    x_arr = np.asarray(x, dtype=float)
    if curve is None:
        return np.ones_like(x_arr)

    if hasattr(curve, "x") and hasattr(curve, "y"):
        return np.interp(x_arr, np.asarray(curve.x, dtype=float), np.asarray(curve.y, dtype=float))
    if callable(curve):
        try:
            res = curve(x_arr)
            return np.asarray(res, dtype=float)
        except Exception:
            return np.ones_like(x_arr)
    if isinstance(curve, (list, tuple)) and len(curve) == 2:
        cx, cy = np.asarray(curve[0], dtype=float), np.asarray(curve[1], dtype=float)
        return np.interp(x_arr, cx, cy)

    return np.ones_like(x_arr)


def _evaluate_failure_strain(
    p: Dict[str, Any],
    triax: np.ndarray,
    epsp_rate: Optional[np.ndarray] = None,
    tstar: Optional[np.ndarray] = None,
    length: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Interpolate failure strain eps_f(eta, eps_dot) matching fail_tab_old_s.F."""
    fscale = _get_param(p, ["fscale", "Fscale", "FSCALE", "xscale1"], 1.0)
    default_eps_f = _get_param(p, ["eps_f", "EPS_F", "fcrit", "FCRIT"], 0.2)

    # 1. Curve / Table lookup
    table = p.get("table", p.get("fct_id_tab", p.get("curve", p.get("curves"))))
    rates = p.get("rates", p.get("rate", p.get("eps_dot")))
    rate_curves = p.get("rate_curves", p.get("load_curves"))

    # Multi-rate interpolation
    if rate_curves and rates and len(rate_curves) > 1 and len(rates) == len(rate_curves):
        rates_arr = np.asarray(rates, dtype=float)
        rate_val = epsp_rate if epsp_rate is not None else np.zeros_like(triax)
        ef_arr = np.zeros_like(triax)
        for i in range(len(triax)):
            r_i = float(rate_val[i])
            # Find bracket
            j1 = 0
            for j in range(1, len(rates_arr) - 1):
                if r_i >= rates_arr[j]:
                    j1 = j
            j2 = min(j1 + 1, len(rates_arr) - 1)
            ef1 = float(_eval_curve(rate_curves[j1], triax[i]))
            ef2 = float(_eval_curve(rate_curves[j2], triax[i]))
            denom = max(abs(rates_arr[j2] - rates_arr[j1]), _TINY)
            fac = np.clip((r_i - rates_arr[j1]) / denom, 0.0, 1.0)
            ef_arr[i] = max(ef1 + fac * (ef2 - ef1), _TINY)
        eps_f = ef_arr * fscale
    elif table is not None:
        eps_f = np.maximum(_eval_curve(table, triax) * fscale, _TINY)
    else:
        eps_f = np.full_like(triax, max(default_eps_f, _TINY))

    # 2. Element length scaling: SC_EL * f_el(L / EL_REF)
    fct_el = p.get("fct_idel", p.get("fct_id_el", p.get("fct_el")))
    if fct_el is not None and length is not None:
        sc_el = _get_param(p, ["scale_el", "Scale_el", "fscale_el", "Fscale_el"], 1.0)
        el_ref = max(_get_param(p, ["el_ref", "El_ref", "ei_ref", "EI_ref"], 1.0), _TINY)
        lambda_val = length / el_ref
        fac_el = sc_el * _eval_curve(fct_el, lambda_val)
        eps_f *= np.maximum(fac_el, _TINY)

    # 3. Temperature scaling: SC_TEMP * f_temp(T*)
    fct_temp = p.get("fct_idt", p.get("fct_id_t", p.get("fct_temp")))
    if fct_temp is not None and tstar is not None:
        sc_temp = _get_param(p, ["scale_temp", "Scale_temp", "fscalet", "FscaleT"], 1.0)
        fac_temp = sc_temp * _eval_curve(fct_temp, tstar)
        eps_f *= np.maximum(fac_temp, _TINY)

    return eps_f


def solid_step(
    fail: Any,
    sig: np.ndarray,
    d_epsp: np.ndarray,
    deps: np.ndarray,
    dt: float,
    dama: np.ndarray,
    tstar: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Advance classic Tab failure (/FAIL/TAB) for a 3D solid slice.

    Fortran reference: fail_tab_old_s.F lines 115-185.
    """
    p = getattr(fail, "params", {})
    dcrit = _get_param(p, ["dcrit", "Dcrit", "DCRIT", "d_crit"], 1.0)
    dd = _get_param(p, ["d", "D", "dd", "DD"], 0.999)
    if math.isclose(dd, 1.0, rel_tol=1e-12):
        dd = 0.999
    dn = _get_param(p, ["n", "N", "dn", "DN"], 1.0)
    if math.isclose(dn, 0.0, abs_tol=1e-12):
        dn = 1.0

    dp = _compute_dp(dd, dn)

    sig_arr = np.asarray(sig, dtype=float)
    n = len(sig_arr)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    # Hydrostatic pressure P = (sxx + syy + szz) / 3
    p_hydro = (sxx + syy + szz) / 3.0
    dev_xx = sxx - p_hydro
    dev_yy = syy - p_hydro
    dev_zz = szz - p_hydro

    # von Mises equivalent stress
    j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + sxy**2 + syz**2 + szx**2
    svm = np.sqrt(np.maximum(_TINY, 3.0 * j2))

    # Stress triaxiality SIGM = P / max(svm, 1e-20)
    triax = p_hydro / np.maximum(svm, _TINY)

    d_epsp_arr = np.maximum(0.0, np.asarray(d_epsp, dtype=float))
    epsp_rate = d_epsp_arr / max(dt, 1.0e-12) if dt > 0.0 else np.zeros_like(d_epsp_arr)

    eps_f = _evaluate_failure_strain(p, triax, epsp_rate=epsp_rate, tstar=tstar)

    # Damage accumulation: UVAR(I,1) = UVAR(I,1) + DP * DPLA / EPSF
    d_dama = dp * d_epsp_arr / np.maximum(eps_f, _TINY)
    dama[:] = np.minimum(dcrit, dama + d_dama)

    return dama >= dcrit


def shell_step(
    fail: Any,
    sig: np.ndarray,
    d_epsp: np.ndarray,
    deps: np.ndarray,
    dt: float,
    dama: np.ndarray,
    tstar: Optional[np.ndarray] = None,
    eps_tot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Advance classic Tab failure (/FAIL/TAB) for a plane-stress shell layer.

    Fortran reference: fail_tab_old_c.F lines 126-155.
    """
    p = getattr(fail, "params", {})
    dcrit = _get_param(p, ["dcrit", "Dcrit", "DCRIT", "d_crit"], 1.0)
    dd = _get_param(p, ["d", "D", "dd", "DD"], 0.999)
    if math.isclose(dd, 1.0, rel_tol=1e-12):
        dd = 0.999
    dn = _get_param(p, ["n", "N", "dn", "DN"], 1.0)
    if math.isclose(dn, 0.0, abs_tol=1e-12):
        dn = 1.0

    dp = _compute_dp(dd, dn)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    # Plane-stress mean stress and von Mises
    p_hydro = (sxx + syy) / 3.0
    svm = np.sqrt(np.maximum(_TINY, sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2))
    triax = p_hydro / np.maximum(svm, _TINY)

    d_epsp_arr = np.maximum(0.0, np.asarray(d_epsp, dtype=float))
    epsp_rate = d_epsp_arr / max(dt, 1.0e-12) if dt > 0.0 else np.zeros_like(d_epsp_arr)

    eps_f = _evaluate_failure_strain(p, triax, epsp_rate=epsp_rate, tstar=tstar)

    d_dama = dp * d_epsp_arr / np.maximum(eps_f, _TINY)
    dama[:] = np.minimum(dcrit, dama + d_dama)

    return dama >= dcrit
