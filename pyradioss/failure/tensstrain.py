"""
Tensile Strain Failure Model (/FAIL/TENSSTRAIN).

Fortran origin: ``starter/source/materials/fail/tensstrain/hm_read_fail_tensstrain.F``,
``engine/source/materials/fail/tensstrain/fail_tensstrain_s.F`` (solids) and
``engine/source/materials/fail/tensstrain/fail_tensstrain_c.F`` (shells).
Failure model IRUPT = 10.

Theory:
-------
Tensile strain threshold model with initiation threshold eps_t1 and complete
rupture threshold eps_t2.

Damage evolution:
  D = 0                                for eps_max <= eps_t1
  D = (eps_max - eps_t1)/(eps_t2 - eps_t1) for eps_t1 < eps_max < eps_t2
  D = 1                                for eps_max >= eps_t2

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
    """Advance TensStrain failure for a solid element slice; returns broken mask."""
    p = fail.params
    eps_t1 = _get_param(p, ["eps_t1", "Epsilon_t1", "epst1", "et1"], _INF)
    eps_t2 = _get_param(p, ["eps_t2", "Epsilon_t2", "epst2", "et2"], eps_t1 * 1.5 if eps_t1 < _INF else _INF)

    deps_arr = np.asarray(deps, dtype=float) if deps is not None else np.zeros((len(dama), 6))
    n = len(deps_arr)
    # Principal tensile strains
    eps_3x3 = np.zeros((n, 3, 3), dtype=float)
    eps_3x3[:, 0, 0] = deps_arr[:, 0]
    eps_3x3[:, 1, 1] = deps_arr[:, 1]
    eps_3x3[:, 2, 2] = deps_arr[:, 2] if deps_arr.shape[1] > 2 else 0.0
    eps_3x3[:, 0, 1] = eps_3x3[:, 1, 0] = deps_arr[:, 3] / 2.0 if deps_arr.shape[1] > 3 else 0.0
    eps_3x3[:, 1, 2] = eps_3x3[:, 2, 1] = deps_arr[:, 4] / 2.0 if deps_arr.shape[1] > 4 else 0.0
    eps_3x3[:, 2, 0] = eps_3x3[:, 0, 2] = deps_arr[:, 5] / 2.0 if deps_arr.shape[1] > 5 else 0.0

    eigvals = np.linalg.eigvalsh(eps_3x3)
    eps_max = eigvals[:, 2]  # maximum principal strain

    d = np.where(eps_max <= eps_t1, 0.0,
                 np.where(eps_max >= eps_t2, 1.0, (eps_max - eps_t1) / max(eps_t2 - eps_t1, _TINY)))

    dama[:] = np.minimum(1.0, np.maximum(dama, d))
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance TensStrain failure for a shell layer; returns broken mask."""
    p = fail.params
    eps_t1 = _get_param(p, ["eps_t1", "Epsilon_t1", "epst1", "et1"], _INF)
    eps_t2 = _get_param(p, ["eps_t2", "Epsilon_t2", "epst2", "et2"], eps_t1 * 1.5 if eps_t1 < _INF else _INF)

    strain_arr = eps_tot if eps_tot is not None else deps
    deps_arr = np.asarray(strain_arr, dtype=float) if strain_arr is not None else np.zeros((len(dama), 3))

    exx = deps_arr[:, 0]
    eyy = deps_arr[:, 1]
    exy = deps_arr[:, 2] / 2.0 if deps_arr.shape[1] > 2 else np.zeros_like(exx)

    center = (exx + eyy) / 2.0
    radius = np.sqrt(((exx - eyy) / 2.0) ** 2 + exy ** 2)
    eps_max = center + radius

    d = np.where(eps_max <= eps_t1, 0.0,
                 np.where(eps_max >= eps_t2, 1.0, (eps_max - eps_t1) / max(eps_t2 - eps_t1, _TINY)))

    dama[:] = np.minimum(1.0, np.maximum(dama, d))
    return dama >= 1.0


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, strains=None, epsd=None, **kwargs):
    """Tensile strain failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/tensstrain/fail_tensstrain_b.F
    Subroutine: FAIL_TENSSTRAIN_B
    """
    p = fail.params
    eps_t1 = _get_param(p, ["eps_t1", "Epsilon_t1", "epst1", "et1"], _INF)
    eps_t2 = _get_param(p, ["eps_t2", "Epsilon_t2", "epst2", "et2"], eps_t1 * 1.5 if eps_t1 < _INF else _INF)

    if strains is not None:
        eps_xx = np.asarray(strains, dtype=float)[:, 0] if np.asarray(strains).ndim == 2 else np.asarray(strains, dtype=float)
    elif deps is not None:
        eps_xx = np.asarray(deps, dtype=float)[:, 0] if np.asarray(deps).ndim == 2 else np.asarray(deps, dtype=float)
    else:
        eps_xx = np.asarray(d_epsp, dtype=float)

    eps_eq = np.abs(eps_xx)
    d = np.where(eps_eq <= eps_t1, 0.0,
                 np.where(eps_eq >= eps_t2, 1.0, (eps_eq - eps_t1) / max(eps_t2 - eps_t1, _TINY)))
    dama[:] = np.minimum(1.0, np.maximum(dama, d))
    return dama >= 1.0


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, epsd=None, ip=0, npg=1, **kwargs):
    """Tensile strain failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/tensstrain/fail_tensstrain_ib.F
    Subroutine: FAIL_TENSSTRAIN_IB
    """
    p = fail.params
    eps_t1 = _get_param(p, ["eps_t1", "Epsilon_t1", "epst1", "et1"], _INF)
    eps_t2 = _get_param(p, ["eps_t2", "Epsilon_t2", "epst2", "et2"], eps_t1 * 1.5 if eps_t1 < _INF else _INF)

    if deps is not None:
        eps_xx = np.asarray(deps, dtype=float)[:, 0] if np.asarray(deps).ndim == 2 else np.asarray(deps, dtype=float)
    else:
        eps_xx = np.asarray(d_epsp, dtype=float)

    eps_eq = np.abs(eps_xx)
    d = np.where(eps_eq <= eps_t1, 0.0,
                 np.where(eps_eq >= eps_t2, 1.0, (eps_eq - eps_t1) / max(eps_t2 - eps_t1, _TINY)))
    dama[:] = np.minimum(1.0, np.maximum(dama, d))
    return dama >= 1.0

