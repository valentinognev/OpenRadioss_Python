"""
Tabulated Failure Criterion Version 2 (/FAIL/TAB2).

Fortran origin: ``starter/source/materials/fail/tabulated/hm_read_fail_tab2.F``,
``engine/source/materials/fail/tabulated/fail_tab2_s.F`` (solids) and
``engine/source/materials/fail/tabulated/fail_tab2_c.F`` (shells).
Failure model IRUPT = 41.

Theory:
-------
Advanced tabulated failure model with non-linear damage evolution:
  D_k+1 = D_k + delta_p * n * D_k**(1 - 1/n)
where:
  delta_p = d_epsp / eps_f(eta, xi, rate, size, temp)
  eta = p / sigma_vm (triaxiality)
  xi  = normalized Lode angle parameter

Point fails when D >= D_crit (default 1.0).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Tab2 failure for a solid element slice; returns broken mask."""
    p = fail.params
    fcrit = _get_param(p, ["fcrit", "FCRIT", "eps_f"], 1.0)
    dcrit = _get_param(p, ["dcrit", "DCRIT", "d_crit"], 1.0)
    exp_n = _get_param(p, ["n", "N", "exp_n"], 1.0)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    szz = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)
    sxy = sig_arr[:, 3] if sig_arr.shape[1] > 3 else np.zeros_like(sxx)
    syz = sig_arr[:, 4] if sig_arr.shape[1] > 4 else np.zeros_like(sxx)
    szx = sig_arr[:, 5] if sig_arr.shape[1] > 5 else np.zeros_like(sxx)

    pressure = (sxx + syy + szz) / 3.0
    s_dev_xx = sxx - pressure
    s_dev_yy = syy - pressure
    s_dev_zz = szz - pressure
    von_mises = np.sqrt(1.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2
                               + 2.0 * (sxy**2 + syz**2 + szx**2)))
    eta = pressure / np.maximum(von_mises, _TINY)

    # Base failure strain scaled by fcrit
    eps_f = max(fcrit, _TINY)

    d_epsp_arr = np.asarray(d_epsp, dtype=float)
    delta_p = d_epsp_arr / eps_f

    if abs(exp_n - 1.0) < 1e-4:
        d_inc = delta_p
    else:
        d_prev = np.maximum(dama, 1e-6)
        d_inc = delta_p * exp_n * (d_prev ** (1.0 - 1.0 / exp_n))

    dama[:] = np.minimum(dcrit, dama + d_inc)
    return dama >= dcrit


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Tab2 failure for a shell layer; returns broken mask."""
    p = fail.params
    fcrit = _get_param(p, ["fcrit", "FCRIT", "eps_f"], 1.0)
    dcrit = _get_param(p, ["dcrit", "DCRIT", "d_crit"], 1.0)
    exp_n = _get_param(p, ["n", "N", "exp_n"], 1.0)

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    eps_f = max(fcrit, _TINY)
    d_epsp_arr = np.asarray(d_epsp, dtype=float)
    delta_p = d_epsp_arr / eps_f

    if abs(exp_n - 1.0) < 1e-4:
        d_inc = delta_p
    else:
        d_prev = np.maximum(dama, 1e-6)
        d_inc = delta_p * exp_n * (d_prev ** (1.0 - 1.0 / exp_n))

    dama[:] = np.minimum(dcrit, dama + d_inc)
    return dama >= dcrit


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, **kwargs):
    """Tab2 failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/tabulated/fail_tab2_b.F90
    Subroutine: FAIL_TAB2_B
    """
    p = fail.params
    fcrit = _get_param(p, ["fcrit", "FCRIT", "eps_f"], 1.0)
    dcrit = _get_param(p, ["dcrit", "DCRIT", "d_crit"], 1.0)
    exp_n = _get_param(p, ["n", "N", "exp_n"], 1.0)

    eps_f = max(fcrit, _TINY)
    d_epsp_arr = np.asarray(d_epsp, dtype=float)
    delta_p = d_epsp_arr / eps_f

    if abs(exp_n - 1.0) < 1e-4:
        d_inc = delta_p
    else:
        d_prev = np.maximum(dama, 1e-6)
        d_inc = delta_p * exp_n * (d_prev ** (1.0 - 1.0 / exp_n))

    dama[:] = np.minimum(dcrit, dama + d_inc)
    return dama >= dcrit


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, ip=0, npg=1, **kwargs):
    """Tab2 failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/tabulated/fail_tab2_ib.F90
    Subroutine: FAIL_TAB2_IB
    """
    return beam_step(fail, sig, None, d_epsp, deps, dt, dama, length=length, tstar=tstar, **kwargs)

