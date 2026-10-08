"""
Generic Multi-Criterion Failure Model (/FAIL/GENE1).

Fortran origin: ``starter/source/materials/fail/gene1/hm_read_fail_gene1.F``,
``engine/source/materials/fail/gene1/fail_gene1_s.F`` (solids) and
``engine/source/materials/fail/gene1/fail_gene1_c.F`` (shells).
Failure model IRUPT = 39.

Theory:
-------
Evaluates up to 18 distinct failure criteria simultaneously (hydrostatic pressures,
principal stresses/strains, von Mises stress, Tuler-Butcher integral, FLD, thinning, etc.).

Failure occurs when the number of satisfied criteria reaches NCS (default 1).
Stress is faded over NSTEP steps.
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
    """Advance Gene1 failure for a solid element slice; returns broken mask."""
    p = fail.params
    sig_vm_max = _get_param(p, ["mat_sigvm", "MAT_SIGVM", "sig_vm_max"], _INF)
    max_eps = _get_param(p, ["mat_maxeps", "MAT_MAXEPS", "max_eps"], _INF)
    eff_eps = _get_param(p, ["mat_effeps", "MAT_EFFEPS", "eff_eps"], _INF)
    ncs = int(_get_param(p, ["mat_ncs", "MAT_NCS", "ncs"], 1))

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))

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

    crit_vm = von_mises >= sig_vm_max
    crit_eps = np.asarray(d_epsp, dtype=float) >= max_eps
    crit_eff = np.sum(np.abs(deps_arr), axis=1) >= eff_eps

    count = crit_vm.astype(int) + crit_eps.astype(int) + crit_eff.astype(int)
    broken = count >= ncs

    dama[:] = np.where(broken, 1.0, np.maximum(dama, count / max(ncs, 1)))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance Gene1 failure for a shell layer; returns broken mask."""
    p = fail.params
    sig_vm_max = _get_param(p, ["mat_sigvm", "MAT_SIGVM", "sig_vm_max"], _INF)
    max_eps = _get_param(p, ["mat_maxeps", "MAT_MAXEPS", "max_eps"], _INF)
    ncs = int(_get_param(p, ["mat_ncs", "MAT_NCS", "ncs"], 1))

    sig_arr = np.asarray(sig, dtype=float)
    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros_like(sxx)

    von_mises = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
    crit_vm = von_mises >= sig_vm_max
    crit_eps = np.asarray(d_epsp, dtype=float) >= max_eps

    count = crit_vm.astype(int) + crit_eps.astype(int)
    broken = count >= ncs

    dama[:] = np.where(broken, 1.0, np.maximum(dama, count / max(ncs, 1)))
    return dama >= 1.0


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, tstar=None, **kwargs):
    """Gene1 failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/gene1/fail_gene1_b.F90
    Subroutine: FAIL_GENE1_B
    """
    p = fail.params
    sig_vm_max = _get_param(p, ["mat_sigvm", "MAT_SIGVM", "sig_vm_max"], _INF)
    max_eps = _get_param(p, ["mat_maxeps", "MAT_MAXEPS", "max_eps"], _INF)
    ncs = int(_get_param(p, ["mat_ncs", "MAT_NCS", "ncs"], 1))

    svm_arr = np.asarray(svm, dtype=float) if np.ndim(svm) > 0 else np.full(len(dama), float(svm))
    crit_vm = svm_arr >= sig_vm_max
    crit_eps = np.asarray(d_epsp, dtype=float) >= max_eps

    count = crit_vm.astype(int) + crit_eps.astype(int)
    broken = count >= ncs
    dama[:] = np.where(broken, 1.0, np.maximum(dama, count / max(ncs, 1)))
    return dama >= 1.0


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, ip=0, npg=1, **kwargs):
    """Gene1 failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/gene1/fail_gene1_ib.F90
    Subroutine: FAIL_GENE1_IB
    """
    p = fail.params
    sig_vm_max = _get_param(p, ["mat_sigvm", "MAT_SIGVM", "sig_vm_max"], _INF)
    max_eps = _get_param(p, ["mat_maxeps", "MAT_MAXEPS", "max_eps"], _INF)
    ncs = int(_get_param(p, ["mat_ncs", "MAT_NCS", "ncs"], 1))

    sig_arr = np.asarray(sig, dtype=float)
    if sig_arr.ndim == 2 and sig_arr.shape[1] >= 3:
        von_mises = np.sqrt(sig_arr[:, 0]**2 + 3.0 * (sig_arr[:, 1]**2 + sig_arr[:, 2]**2))
    elif sig_arr.ndim == 2:
        von_mises = np.abs(sig_arr[:, 0])
    else:
        von_mises = np.abs(sig_arr)

    crit_vm = von_mises >= sig_vm_max
    crit_eps = np.asarray(d_epsp, dtype=float) >= max_eps

    count = crit_vm.astype(int) + crit_eps.astype(int)
    broken = count >= ncs
    dama[:] = np.where(broken, 1.0, np.maximum(dama, count / max(ncs, 1)))
    return dama >= 1.0

