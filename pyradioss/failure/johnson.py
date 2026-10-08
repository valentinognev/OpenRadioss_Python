"""
Johnson–Cook failure criterion (/FAIL/JOHNSON).

Fortran origin: ``engine/source/materials/fail/johnson_cook/fail_johnson_s.F``
(solids) and ``fail_johnson_c.F`` (shells); reader
``starter/source/materials/fail/johnson_cook/hm_read_fail_johnson.F``.

Theory (Johnson & Cook, Eng. Fracture Mech. 21 (1985) 31-48)
------------------------------------------------------------
Ductile fracture by accumulation of plastic strain, with a failure
strain that depends on the **stress triaxiality**
sigma* = sigma_m / sigma_vm (mean over von Mises stress — the classic
observation that hydrostatic tension accelerates void growth) and on the
strain rate:

    eps_f = ( D1 + D2 * exp(D3 * sigma*) ) * ( 1 + D4 * ln(rate/rate_0) )
                                           * ( 1 + D5 * T* )

    D += d_eps_p / eps_f         (linear damage accumulation)

The point breaks at D >= 1. The thermal term (1 + D5*T*, M6) uses the
homologous temperature from the material's ADIABATIC temperature state
(the LAW2 thermal card — plastic work heats the point, see
law02_johnson_cook.py); D5 > 0 raises the failure strain with
temperature (hot metal is more ductile — the Johnson-Cook 1985
calibration convention). Without a thermal card the term is dropped
(warned by the Starter).

Conventions matching the original:
* sigma* uses the CURRENT stress state, tension positive. Note D3 is
  usually NEGATIVE for metals in this convention (failure strain drops
  with triaxiality); some references flip the sign of both.
* the rate factor is clamped at 1 below the reference rate (no rate
  term unless D4 > 0 and the rate exceeds eps_dot_0);
* no damage grows without plastic flow (d_eps_p = 0 -> D frozen), so
  the criterion is inert on elastic materials — the Starter warns.
* a NON-POSITIVE failure strain FREEZES the damage instead of failing
  the point (M40): the reference floors eps_f at EPSF_MIN (default 0)
  and accumulates only where eps_f > 0 (fail_johnson.F ``EPSF =
  MAX(EPSF,EPSF_MIN)``, ``IF (EPSF>ZERO) DFMAX = DFMAX + DPLA/EPSF``;
  identical in fail_johnson_c.F).  With D1 < 0 (e.g. the RD-V-0700
  calibration D1=-0.1, D2=0.6, D3=-1.2) eps_f goes negative beyond
  sigma* = ln(-D2/D1)/D3 = 1.493: such highly triaxial points can
  NEVER fail through this criterion — the reference keeps them alive
  (its own Starter only warns when the root lies inside |sigma*| <= 1).
  The port used to divide by max(eps_f, 1e-20), instant-deleting those
  points on their first plastic increment — the exact opposite.
  Damage is capped at 1 (``DFMAX = MIN(ONE,DFMAX)``), which the
  deletion threshold D >= 1 makes output-only for solids.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _rate_factor(fail, deps=None, dt=0.0, dev_from_6=True, d_epsp=None, epsd=None):
    """1 + D4*ln(rate/rate0), clamped at 1 below the reference rate."""
    D4 = fail.params.get("D4", 0.0)
    if D4 == 0.0:
        return 1.0
    if epsd is not None:
        rate = np.asarray(epsd, dtype=float)
    elif d_epsp is not None:
        rate = np.maximum(d_epsp, 0.0) / max(dt, _TINY)
    elif deps is not None and (np.ndim(deps) == 0 or (isinstance(deps, np.ndarray) and deps.ndim == 1 and len(deps) not in (3, 6))):
        rate = np.maximum(deps, 0.0) / max(dt, _TINY)
    else:
        deps_arr = np.asarray(deps, dtype=float)
        if dev_from_6 and deps_arr.ndim == 2 and deps_arr.shape[1] >= 6:
            tr3 = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0
            ee = (deps_arr[:, 0] - tr3) ** 2 + (deps_arr[:, 1] - tr3) ** 2 \
                + (deps_arr[:, 2] - tr3) ** 2 \
                + 0.5 * (deps_arr[:, 3] ** 2 + deps_arr[:, 4] ** 2 + deps_arr[:, 5] ** 2)
            rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, _TINY)
        elif not dev_from_6 and deps_arr.ndim == 2 and deps_arr.shape[1] >= 3:
            dzz = -(deps_arr[:, 0] + deps_arr[:, 1])
            tr3 = (deps_arr[:, 0] + deps_arr[:, 1] + dzz) / 3.0
            ee = (deps_arr[:, 0] - tr3) ** 2 + (deps_arr[:, 1] - tr3) ** 2 \
                + (dzz - tr3) ** 2 + 0.5 * deps_arr[:, 2] ** 2
            rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, _TINY)
        else:
            rate = np.maximum(deps_arr, 0.0) / max(dt, _TINY)
    eps0 = max(fail.params.get("eps_dot_0", 1.0), _TINY)
    r = np.maximum(rate / eps0, 1.0)
    return 1.0 + D4 * np.log(r)


def _thermal_factor(fail, tstar):
    """1 + D5*T* (M6) — 1 when no D5 or no temperature state."""
    D5 = fail.params.get("D5", 0.0)
    if D5 == 0.0 or tstar is None:
        return 1.0
    return 1.0 + D5 * tstar


def _accumulate(fail, dama, d_epsp, eps_f):
    """In-place damage update, the fail_johnson.F contract (M40):

        EPSF  = MAX(EPSF, EPSF_MIN)          (EPSF_MIN default 0)
        IF (EPSF > ZERO) DFMAX = DFMAX + DPLA/EPSF
        DFMAX = MIN(ONE, DFMAX)

    i.e. a non-positive failure strain FREEZES the damage (the point
    cannot fail there) rather than failing it instantly — see the
    module docstring's D1 < 0 note.  ``dama`` may be a slice view of
    the element-group state array: in-place ops only."""
    eps_f = np.maximum(eps_f, fail.params.get("eps_f_min", 0.0))
    grow = eps_f > 0.0
    dama += np.where(grow, np.maximum(d_epsp, 0.0), 0.0) \
        / np.where(grow, eps_f, 1.0)
    np.minimum(dama, 1.0, out=dama)


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """3-D damage step. sig (m, 6) Voigt; ``tstar`` (M6) = homologous
    temperature array of the slice, None without a thermal material.
    Returns the broken mask."""
    p = fail.params
    sm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    s0, s1, s2 = sig[:, 0] - sm, sig[:, 1] - sm, sig[:, 2] - sm
    vm = np.sqrt(1.5 * (s0 ** 2 + s1 ** 2 + s2 ** 2)
                 + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
    triax = sm / np.maximum(vm, _TINY)
    d1 = float(p.get("D1", p.get("d1", 0.1)))
    d2 = float(p.get("D2", p.get("d2", 0.5)))
    d3 = float(p.get("D3", p.get("d3", -1.5)))
    eps_f = (d1 + d2 * np.exp(np.clip(d3 * triax, -100.0, 100.0))) \
        * _rate_factor(fail, deps, dt, True, d_epsp=d_epsp) \
        * _thermal_factor(fail, tstar)
    _accumulate(fail, dama, d_epsp, eps_f)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Plane-stress damage step for one layer. sig (m, 3) = [xx, yy, xy]."""
    p = fail.params
    sm = (sig[:, 0] + sig[:, 1]) / 3.0            # sigma_zz = 0
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2
                 + 3.0 * sig[:, 2] ** 2)
    triax = sm / np.maximum(vm, _TINY)
    d1 = float(p.get("D1", p.get("d1", 0.1)))
    d2 = float(p.get("D2", p.get("d2", 0.5)))
    d3 = float(p.get("D3", p.get("d3", -1.5)))
    eps_f = (d1 + d2 * np.exp(np.clip(d3 * triax, -100.0, 100.0))) \
        * _rate_factor(fail, deps, dt, False, d_epsp=d_epsp) \
        * _thermal_factor(fail, tstar)
    _accumulate(fail, dama, d_epsp, eps_f)
    return dama >= 1.0


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, epsd=None, **kwargs):
    """Johnson-Cook failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/johnson_cook/fail_johnson_b.F
    Subroutine: FAIL_JOHNSON_B
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
    triax = p_arr / np.maximum(svm_arr, _TINY)

    d1 = float(p.get("D1", p.get("d1", 0.1)))
    d2 = float(p.get("D2", p.get("d2", 0.5)))
    d3 = float(p.get("D3", p.get("d3", -1.5)))

    eps_f = (d1 + d2 * np.exp(np.clip(d3 * triax, -100.0, 100.0))) \
        * _rate_factor(fail, deps, dt, False, d_epsp=d_epsp, epsd=epsd) \
        * _thermal_factor(fail, tstar)
    _accumulate(fail, dama, d_epsp, eps_f)
    return dama >= 1.0


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, epsd=None, ip=0, npg=1, **kwargs):
    """Johnson-Cook failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/johnson_cook/fail_johnson_ib.F
    Subroutine: FAIL_JOHNSON_IB
    """
    p = fail.params
    sig_arr = np.asarray(sig, dtype=float)
    if sig_arr.ndim == 1:
        sig_xx = sig_arr
        sig_xy = np.zeros_like(sig_xx)
        sig_xz = np.zeros_like(sig_xx)
    elif sig_arr.shape[1] == 1:
        sig_xx = sig_arr[:, 0]
        sig_xy = np.zeros_like(sig_xx)
        sig_xz = np.zeros_like(sig_xx)
    elif sig_arr.shape[1] == 3:
        sig_xx = sig_arr[:, 0]
        sig_xy = sig_arr[:, 1]
        sig_xz = sig_arr[:, 2]
    else:
        sig_xx = sig_arr[:, 0]
        sig_xy = sig_arr[:, 3]
        sig_xz = sig_arr[:, 5]

    pressure = sig_xx / 3.0
    svm = np.sqrt(sig_xx**2 + 3.0 * (sig_xy**2 + sig_xz**2))
    triax = pressure / np.maximum(svm, _TINY)

    d1 = float(p.get("D1", p.get("d1", 0.1)))
    d2 = float(p.get("D2", p.get("d2", 0.5)))
    d3 = float(p.get("D3", p.get("d3", -1.5)))

    eps_f = (d1 + d2 * np.exp(np.clip(d3 * triax, -100.0, 100.0))) \
        * _rate_factor(fail, deps, dt, False, d_epsp=d_epsp, epsd=epsd) \
        * _thermal_factor(fail, tstar)
    _accumulate(fail, dama, d_epsp, eps_f)
    return dama >= 1.0


def xfem_step(fail, sig, d_epsp, deps, dt, dama, elcrkini, tstar=None, eps_tot=None, dadv=1.0, is_phantom=False):
    """Johnson-Cook failure step with XFEM crack tracking.

    # Ported from $OR_SRC/engine/source/materials/fail/johnson_cook/fail_johnson_xfem.F
    Subroutine: FAIL_JOHNSON_XFEM
    """
    shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)

    elcrkini_arr = np.asarray(elcrkini, dtype=int)
    broken = np.zeros(len(dama), dtype=bool)
    for i in range(len(dama)):
        if is_phantom:
            if dama[i] >= 1.0:
                broken[i] = True
        else:
            if elcrkini_arr[i] == 0 and dama[i] >= 1.0:
                elcrkini[i] = -1
                broken[i] = True
            elif elcrkini_arr[i] == 2 and dama[i] >= dadv:
                elcrkini[i] = 1
                broken[i] = True
            elif dama[i] >= 1.0:
                broken[i] = True

    return broken

