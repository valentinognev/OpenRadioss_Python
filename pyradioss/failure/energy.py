"""
Specific Energy Failure Model (/FAIL/ENERGY).

Fortran origin: ``starter/source/materials/fail/energy/hm_read_fail_energy.F``,
``engine/source/materials/fail/energy/fail_energy_s.F`` (solids) and
``engine/source/materials/fail/energy/fail_energy_c.F`` (shells).
Failure model IRUPT = 11.

Theory:
-------
Accumulates specific plastic work density:
  e_sp += sigma : d_eps_p

Linear damage between initiation energy E1 and failure energy E2:
  D = 0                      for e_sp <= E1
  D = (e_sp - E1)/(E2 - E1)  for E1 < e_sp <= E2
  D = 1                      for e_sp > E2

Point fails when D >= 1.0.
"""

from __future__ import annotations

import numpy as np

_INF = 1e30
_TINY = 1e-20


def _get_param(params: dict, keys: list[str], default: float = _INF) -> float:
    for k in keys:
        if k in params:
            val = params[k]
            if val is not None:
                return float(val)
    return default


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance Energy failure for a solid element slice; returns broken mask."""
    p = fail.params
    e1 = _get_param(p, ["e1", "E1", "e_init"], 1.0e20)
    e2 = _get_param(p, ["e2", "E2", "e_fail"], 2.0e20)
    if e2 <= e1:
        e2 = e1 + _TINY

    # Approximate specific work increment using von Mises equivalent stress * d_epsp
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

    d_work = von_mises * np.asarray(d_epsp, dtype=float)

    # In dama we store the accumulated specific work until e1, then normalized damage
    # To keep dama in [0, 1], we map dama from work:
    # Let work = dama * e2 if dama <= 1
    current_work = dama * e2 + d_work
    d_norm = np.where(current_work <= e1, 0.0, (current_work - e1) / (e2 - e1))
    dama[:] = np.clip(d_norm, 0.0, 1.0)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Energy failure for a shell layer; returns broken mask."""
    p = fail.params
    e1 = _get_param(p, ["e1", "E1", "e_init"], 1.0e20)
    e2 = _get_param(p, ["e2", "E2", "e_fail"], 2.0e20)
    if e2 <= e1:
        e2 = e1 + _TINY

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    von_mises = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
    d_work = von_mises * np.asarray(d_epsp, dtype=float)

    current_work = dama * e2 + d_work
    d_norm = np.where(current_work <= e1, 0.0, (current_work - e1) / (e2 - e1))
    dama[:] = np.clip(d_norm, 0.0, 1.0)
    return dama >= 1.0


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, forces=None, moments=None, strains=None, curvatures=None, area=None, epsd=None, **kwargs):
    """Specific energy failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/energy/fail_energy_b.F
    Subroutine: FAIL_ENERGY_B
    """
    p = fail.params
    e1 = _get_param(p, ["e1", "E1", "e_init"], 1.0e20)
    e2 = _get_param(p, ["e2", "E2", "e_fail"], 2.0e20)
    if e2 <= e1:
        e2 = e1 + _TINY

    if forces is not None and strains is not None:
        f_arr = np.asarray(forces, dtype=float)
        s_arr = np.asarray(strains, dtype=float)
        dW = f_arr[:, 0] * s_arr[:, 0]
        if f_arr.shape[1] > 1 and s_arr.shape[1] > 1:
            dW = dW + f_arr[:, 1] * s_arr[:, 1]
        if f_arr.shape[1] > 2 and s_arr.shape[1] > 2:
            dW = dW + f_arr[:, 2] * s_arr[:, 2]
        if moments is not None and curvatures is not None:
            m_arr = np.asarray(moments, dtype=float)
            k_arr = np.asarray(curvatures, dtype=float)
            dW = dW + m_arr[:, 0] * k_arr[:, 0]
            if m_arr.shape[1] > 1 and k_arr.shape[1] > 1:
                dW = dW + m_arr[:, 1] * k_arr[:, 1]
            if m_arr.shape[1] > 2 and k_arr.shape[1] > 2:
                dW = dW + m_arr[:, 2] * k_arr[:, 2]
        area_val = max(float(area), _TINY) if area is not None else 1.0
        d_work = dW / area_val
    else:
        svm_arr = np.asarray(svm, dtype=float) if np.ndim(svm) > 0 or not np.isnan(svm) else np.zeros(len(dama))
        d_work = svm_arr * np.asarray(d_epsp, dtype=float)

    current_work = dama * e2 + d_work
    d_norm = np.where(current_work <= e1, 0.0, (current_work - e1) / (e2 - e1))
    dama[:] = np.clip(d_norm, 0.0, 1.0)
    return dama >= 1.0


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, epsd=None, ip=0, npg=1, **kwargs):
    """Specific energy failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/energy/fail_energy_ib.F
    Subroutine: FAIL_ENERGY_IB
    """
    p = fail.params
    e1 = _get_param(p, ["e1", "E1", "e_init"], 1.0e20)
    e2 = _get_param(p, ["e2", "E2", "e_fail"], 2.0e20)
    if e2 <= e1:
        e2 = e1 + _TINY

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    if sig_arr.ndim == 2 and deps_arr.ndim == 2 and sig_arr.shape[1] >= 3 and deps_arr.shape[1] >= 3:
        d_work = (sig_arr[:, 0] * deps_arr[:, 0] +
                  sig_arr[:, 1] * deps_arr[:, 1] +
                  sig_arr[:, 2] * deps_arr[:, 2])
    elif sig_arr.ndim == 2 and deps_arr.ndim == 2:
        d_work = np.sum(sig_arr * deps_arr, axis=1)
    else:
        svm = np.abs(sig_arr.ravel())
        d_work = svm * np.asarray(d_epsp, dtype=float)

    current_work = dama * e2 + d_work
    d_norm = np.where(current_work <= e1, 0.0, (current_work - e1) / (e2 - e1))
    dama[:] = np.clip(d_norm, 0.0, 1.0)
    return dama >= 1.0

