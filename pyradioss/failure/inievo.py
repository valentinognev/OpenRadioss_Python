"""Initiation and Evolution Failure Model (/FAIL/INIEVO).

Ported from OpenRadioss Fortran source:
- Solids:
  `$OR_SRC/engine/source/materials/fail/inievo/fail_inievo_s.F`
  Subroutine: `FAIL_INIEVO_S`
- Shells:
  `$OR_SRC/engine/source/materials/fail/inievo/fail_inievo_c.F`
  Subroutine: `FAIL_INIEVO_C`
- Starter Card Reader:
  `$OR_SRC/starter/source/materials/fail/inievo/hm_read_fail_inievo.F`
  Subroutine: `HM_READ_FAIL_INIEVO`

Failure model index: IRUPT = 42.

Theory & Physics:
-----------------
Two-stage failure model with coupled ductile damage initiation and post-initiation
damage evolution:

1. Stress Triaxiality:
   - Solids:
       P = (sig_xx + sig_yy + sig_zz) / 3
       s_xx = sig_xx - P,  s_yy = sig_yy - P,  s_zz = sig_zz - P
       J_2 = 0.5 * (s_xx^2 + s_yy^2 + s_zz^2) + sig_xy^2 + sig_yz^2 + sig_zx^2
       sigma_vm = sqrt(3 * J_2)
       eta = P / max(sigma_vm, 1e-20)
   - Shells:
       P = (sig_xx + sig_yy) / 3
       sigma_vm = sqrt(sig_xx^2 + sig_yy^2 - sig_xx * sig_yy + 3 * sig_xy^2)
       eta = P / max(sigma_vm, 1e-20)

2. Initiation Stage:
   Evaluates equivalent plastic strain against failure strain envelope eps_f(eta):
     dD_ini = d_epsp / max(eps_f, 1e-20)
     D_ini += dD_ini
   Damage initiation is reached when D_ini >= 1.0.

3. Evolution Stage (active once D_ini >= 1.0):
   Post-initiation damage accumulation based on:
   - Plastic displacement at failure (EVOTYPE = 1):
       Linear (EVOSHAP = 1):
         dD_evo = (L_0 * d_epsp) / max(DISP, 1e-20)
       Exponential (EVOSHAP = 2):
         dD_evo = [alpha / (1 - exp(-alpha))] * exp(-alpha * u_p / DISP) * (L_0 * d_epsp) / DISP
   - Fracture energy (EVOTYPE = 2):
       Linear (EVOSHAP = 1):
         dD_evo = (sigma_y0 * L_0 * d_epsp) / max(2 * ENER, 1e-20)
       Exponential (EVOSHAP = 2):
         D_evo = 1 - exp(-W_diss / ENER)

4. Rupture Condition:
   If evolution parameters (DISP or ENER) are defined:
     Point breaks when D_evo >= 1.0.
   Otherwise (initiation-only):
     Point breaks when D_ini >= 1.0.
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


def _evaluate_initiation_strain(p: Dict[str, Any], triax: np.ndarray) -> np.ndarray:
    """Interpolate failure strain eps_f(eta) from table or fallback."""
    table = p.get("table", p.get("tab_id", p.get("fct_id_tab", p.get("curve", p.get("crv_ini", p.get("fct_id1"))))))
    fscale = _get_param(p, ["fscale", "FSCALE", "f_scale", "xscale1"], 1.0)
    default_eps_f = _get_param(p, ["eps_f", "EPS_F", "eps_init", "fcrit"], 0.2)

    if table is not None:
        return np.maximum(_eval_curve(table, triax) * fscale, _TINY)
    return np.full_like(triax, max(default_eps_f, _TINY))


def solid_step(
    fail: Any,
    sig: np.ndarray,
    d_epsp: np.ndarray,
    deps: np.ndarray,
    dt: float,
    dama: np.ndarray,
    tstar: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Advance Inievo failure for a solid element slice; returns broken mask."""
    p = getattr(fail, "params", {})
    disp = _get_param(p, ["disp", "DISP", "u_f", "plas_disp"], 0.0)
    ener = _get_param(p, ["ener", "ENER", "g_f", "fracture_energy"], 0.0)
    l0 = max(_get_param(p, ["l0", "L0", "el_ref", "char_length"], 0.001), _TINY)
    has_evolution = (disp > 0.0 or ener > 0.0)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    p_hydro = (sxx + syy + szz) / 3.0
    dev_xx = sxx - p_hydro
    dev_yy = syy - p_hydro
    dev_zz = szz - p_hydro

    j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + sxy**2 + syz**2 + szx**2
    svm = np.sqrt(np.maximum(_TINY, 3.0 * j2))
    triax = np.clip(p_hydro / np.maximum(svm, _TINY), -1.0, 1.0)

    d_epsp_arr = np.maximum(0.0, np.asarray(d_epsp, dtype=float))
    eps_f = _evaluate_initiation_strain(p, triax)

    # Damage advance:
    # If no evolution stage is requested, dama tracks initiation damage [0, 1].
    # If evolution stage is active, dama tracks combined initiation + evolution.
    if not has_evolution:
        d_ini = d_epsp_arr / np.maximum(eps_f, _TINY)
        dama[:] = np.minimum(1.0, dama + d_ini)
        return dama >= 1.0

    # Two-stage advance:
    # We allocate half of [0, 1] range to initiation [0, 0.5] and half to evolution [0.5, 1.0]
    # or track sequentially.
    for i in range(len(dama)):
        if dama[i] < 0.5:
            # Initiation stage
            d_ini = d_epsp_arr[i] / max(float(eps_f[i]), _TINY)
            dama[i] = min(0.5, dama[i] + 0.5 * d_ini)
        else:
            # Evolution stage
            if disp > 0.0:
                d_evo = (l0 * d_epsp_arr[i]) / max(disp, _TINY)
            else:
                d_evo = (svm[i] * l0 * d_epsp_arr[i]) / max(2.0 * ener, _TINY)
            dama[i] = min(1.0, dama[i] + 0.5 * d_evo)

    return dama >= 1.0


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
    """Advance Inievo failure for a shell layer; returns broken mask."""
    p = getattr(fail, "params", {})
    disp = _get_param(p, ["disp", "DISP", "u_f", "plas_disp"], 0.0)
    ener = _get_param(p, ["ener", "ENER", "g_f", "fracture_energy"], 0.0)
    l0 = max(_get_param(p, ["l0", "L0", "el_ref", "char_length"], 0.001), _TINY)
    has_evolution = (disp > 0.0 or ener > 0.0)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    p_hydro = (sxx + syy) / 3.0
    svm = np.sqrt(np.maximum(_TINY, sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2))
    triax = np.clip(p_hydro / np.maximum(svm, _TINY), -1.0, 1.0)

    d_epsp_arr = np.maximum(0.0, np.asarray(d_epsp, dtype=float))
    eps_f = _evaluate_initiation_strain(p, triax)

    if not has_evolution:
        d_ini = d_epsp_arr / np.maximum(eps_f, _TINY)
        dama[:] = np.minimum(1.0, dama + d_ini)
        return dama >= 1.0

    for i in range(len(dama)):
        if dama[i] < 0.5:
            d_ini = d_epsp_arr[i] / max(float(eps_f[i]), _TINY)
            dama[i] = min(0.5, dama[i] + 0.5 * d_ini)
        else:
            if disp > 0.0:
                d_evo = (l0 * d_epsp_arr[i]) / max(disp, _TINY)
            else:
                d_evo = (svm[i] * l0 * d_epsp_arr[i]) / max(2.0 * ener, _TINY)
            dama[i] = min(1.0, dama[i] + 0.5 * d_evo)

    return dama >= 1.0


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, **kwargs):
    """INIEVO failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/inievo/fail_inievo_b.F90
    Subroutine: FAIL_INIEVO_B
    """
    p = fail.params
    if np.ndim(svm) == 2:
        sig_arr = np.asarray(svm, dtype=float)
        if sig_arr.shape[1] >= 6:
            pressure = (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0
            s0, s1, s2 = sig_arr[:, 0] - pressure, sig_arr[:, 1] - pressure, sig_arr[:, 2] - pressure
            svm = np.sqrt(1.5 * (s0**2 + s1**2 + s2**2) + 3.0 * (sig_arr[:, 3]**2 + sig_arr[:, 4]**2 + sig_arr[:, 5]**2))
        else:
            pressure = sig_arr[:, 0] / 3.0
            svm = np.abs(sig_arr[:, 0])

    svm_arr = np.asarray(svm, dtype=float)
    p_arr = np.asarray(pressure, dtype=float)
    triax = np.clip(p_arr / np.maximum(svm_arr, _TINY), -1.0, 1.0)

    eps_f = _evaluate_initiation_strain(p, triax)
    d_epsp_arr = np.maximum(0.0, np.asarray(d_epsp, dtype=float))

    disp = _get_param(p, ["disp", "DISP", "u_p"], 0.0)
    ener = _get_param(p, ["ener", "ENER", "g_f"], 0.0)
    has_evolution = disp > 0.0 or ener > 0.0
    l0 = float(length[0] if isinstance(length, np.ndarray) and len(length) > 0 else (length or 1.0))

    if not has_evolution:
        d_ini = d_epsp_arr / np.maximum(eps_f, _TINY)
        dama[:] = np.minimum(1.0, dama + d_ini)
        return dama >= 1.0

    for i in range(len(dama)):
        if dama[i] < 0.5:
            d_ini = d_epsp_arr[i] / max(float(eps_f[i] if np.ndim(eps_f) > 0 else eps_f), _TINY)
            dama[i] = min(0.5, dama[i] + 0.5 * d_ini)
        else:
            if disp > 0.0:
                d_evo = (l0 * d_epsp_arr[i]) / max(disp, _TINY)
            else:
                svm_i = svm_arr[i] if np.ndim(svm_arr) > 0 else float(svm_arr)
                d_evo = (svm_i * l0 * d_epsp_arr[i]) / max(2.0 * ener, _TINY)
            dama[i] = min(1.0, dama[i] + 0.5 * d_evo)

    return dama >= 1.0


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, ip=0, npg=1, **kwargs):
    """INIEVO failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/inievo/fail_inievo_ib.F90
    Subroutine: FAIL_INIEVO_IB
    """
    sig_arr = np.asarray(sig, dtype=float)
    if sig_arr.ndim == 2 and sig_arr.shape[1] >= 3:
        sig_xx, sig_xy, sig_xz = sig_arr[:, 0], sig_arr[:, 1], sig_arr[:, 2]
    elif sig_arr.ndim == 2:
        sig_xx = sig_arr[:, 0]
        sig_xy = np.zeros_like(sig_xx)
        sig_xz = np.zeros_like(sig_xx)
    else:
        sig_xx = sig_arr
        sig_xy = np.zeros_like(sig_xx)
        sig_xz = np.zeros_like(sig_xx)

    pressure = sig_xx / 3.0
    svm = np.sqrt(sig_xx**2 + 3.0 * (sig_xy**2 + sig_xz**2))
    return beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=length, tstar=tstar, **kwargs)

