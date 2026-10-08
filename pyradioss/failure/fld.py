"""
Forming Limit Diagram failure model (/FAIL/FLD).

Fortran origin: ``starter/source/materials/fail/fld/hm_read_fail_fld.F`` and
``engine/source/materials/fail/fld/fail_fld_c.F``.

Physics:
The Forming Limit Diagram (FLD) criterion predicts onset of localized necking /
fracture in sheet metal shells based on the minor and major in-plane principal strains
(eps_min, eps_maj).

Principal strains:
    S1 = 0.5 * (eps_xx + eps_yy)
    S2 = 0.5 * (eps_xx - eps_yy)
    Q  = sqrt(S2^2 + (0.5 * eps_xy)^2)
    eps_maj = S1 + Q
    eps_min = S1 - Q

Limit major strain EM is evaluated from the FLD curve function f(eps_min):
- If Istrain == 0 (true strain):
    EM = f(eps_min)
- If Istrain == 1 (engineering strain):
    eps_min_eng = exp(eps_min) - 1
    EM_eng = f(eps_min_eng)
    EM = ln(1 + EM_eng)

Damage ratio:
    D = eps_maj / max(EM, 1e-20)
    dama = max(dama, D)

Failure condition:
    Point breaks when dama >= 1.0 (unless Ifail_sh == 4, where damage is monitored
    without element deletion).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Solid step: FLD is a shell/plane-stress criterion; no-op for solids."""
    return np.zeros(len(dama), dtype=bool)


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Plane-stress FLD damage step for one layer of a shell slice.

    Parameters
    ----------
    fail : FailureModel
        The /FAIL/FLD failure model instance.
    sig : np.ndarray (m, 3)
        Plane-stress tensor [sig_xx, sig_yy, sig_xy].
    d_epsp : np.ndarray (m,) or float
        Plastic strain increment of the cycle.
    deps : np.ndarray (m, 3)
        In-plane strain increment [deps_xx, deps_yy, deps_xy].
    dt : float
        Current time step.
    dama : np.ndarray (m,)
        Persistent damage array (modified in place).
    tstar : np.ndarray (m,) or None
        Homologous temperature (unused by FLD).
    eps_tot : np.ndarray (m, 3) or None
        Total in-plane strain tensor [eps_xx, eps_yy, eps_xy]. If None,
        approximated from deps or plastic strain.
    """
    p = fail.params
    func = p.get("function")
    if func is None:
        return dama >= 1.0

    if eps_tot is not None:
        e = eps_tot
    else:
        # Fallback if total strain is not explicitly supplied
        e = deps if deps.ndim == 2 else deps[None, :]

    eps_xx = e[:, 0]
    eps_yy = e[:, 1]
    eps_xy = e[:, 2]

    e12 = 0.5 * eps_xy
    s1 = 0.5 * (eps_xx + eps_yy)
    s2 = 0.5 * (eps_xx - eps_yy)
    q = np.sqrt(s2 ** 2 + e12 ** 2)

    emaj = s1 + q
    emin = s1 - q

    # Ensure emin <= emaj
    swap = emin > emaj
    if np.any(swap):
        emaj_s = emaj.copy()
        emaj = np.where(swap, emin, emaj)
        emin = np.where(swap, emaj_s, emin)

    istrain = p.get("istrain", 0)
    if istrain == 1:
        # Engineering strain input: convert true minor strain to engineering
        emin_eng = np.exp(np.clip(emin, -100.0, 100.0)) - 1.0
        em_eng = np.interp(emin_eng, func.x, func.y)
        em = np.log(np.maximum(em_eng + 1.0, _TINY))
    else:
        # True strain input
        em = np.interp(emin, func.x, func.y)

    dam = emaj / np.maximum(em, _TINY)
    np.maximum(dama, dam, out=dama)

    ifail_sh = p.get("ifail_sh", getattr(fail, "ifail_sh", 1))
    if ifail_sh == 4:
        # Ifail_sh = 4: calculation only, no element deletion
        return np.zeros(len(dama), dtype=bool)

    return dama >= 1.0


def thick_shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None, pla=None):
    """Thick shell Forming Limit Diagram failure step.

    Ported from $OR_SRC/engine/source/materials/fail/fld/fail_fld_tsh.F
    Subroutine: FAIL_FLD_TSH
    """
    p = fail.params
    func = p.get("function")
    if func is None:
        return dama >= 1.0

    if eps_tot is not None:
        e = eps_tot
    else:
        e = deps if deps.ndim == 2 else deps[None, :]

    eps_xx = e[:, 0]
    eps_yy = e[:, 1]
    eps_xy = e[:, 2]

    e12 = 0.5 * eps_xy
    s1 = 0.5 * (eps_xx + eps_yy)
    s2 = 0.5 * (eps_xx - eps_yy)
    q = np.sqrt(s2 ** 2 + e12 ** 2)

    emaj = s1 + q
    emin = s1 - q

    swap = emin > emaj
    if np.any(swap):
        emaj_s = emaj.copy()
        emaj = np.where(swap, emin, emaj)
        emin = np.where(swap, emaj_s, emin)

    beta = emin / np.maximum(emaj, _TINY)

    istrain = p.get("istrain", p.get("ieng", 0))
    if istrain == 1:
        # Engineering strain input
        emin_eng = np.exp(np.clip(emin, -100.0, 100.0)) - 1.0
        em_eng = np.interp(emin_eng, func.x, func.y)
        em = np.log(np.maximum(em_eng + 1.0, _TINY))
        dam = emaj / np.maximum(em, _TINY)
    elif istrain == 2:
        # Non-linear strain path formulation
        em = np.interp(beta, func.x, func.y)
        plastic = pla if pla is not None else d_epsp
        dam = np.asarray(plastic, dtype=float) / np.maximum(em, _TINY)
    else:
        # True strain input
        em = np.interp(emin, func.x, func.y)
        dam = emaj / np.maximum(em, _TINY)

    np.maximum(dama, dam, out=dama)

    # Optional zone index calculation (zones 1 to 6 matching fail_fld_tsh.F lines 202-281)
    fact_loosemetal = float(p.get("fact_loosemetal", 0.02))
    rani = float(p.get("rani", 1.0))
    fact_margin = float(p.get("fact_margin", 0.1))
    imargin = int(p.get("imargin", 0))

    r1 = fact_loosemetal
    r2 = rani / (rani + 1.0)
    zones = np.full(len(dama), 4, dtype=int)  # default zone 4 = safe
    for i in range(len(dama)):
        em_val = em[i] if np.ndim(em) > 0 else float(em)
        emaj_val = emaj[i]
        emin_val = emin[i]
        margin_thresh = em_val * (1.0 - fact_margin) if imargin == 3 else em_val - fact_margin

        if emaj_val >= em_val:
            zones[i] = 6  # failure
        elif emaj_val >= margin_thresh:
            zones[i] = 5  # margin
        elif emaj_val**2 + emin_val**2 < r1**2:
            zones[i] = 1  # loose metal
        elif emaj_val >= abs(emin_val):
            zones[i] = 4  # safe
        elif emaj_val >= r2 * abs(emin_val):
            zones[i] = 3  # compression
        else:
            zones[i] = 2  # wrinkle tendency

    fail.fld_zone = zones

    ifail_sh = p.get("ifail_sh", getattr(fail, "ifail_sh", 1))
    if ifail_sh == 4:
        return np.zeros(len(dama), dtype=bool)

    return dama >= 1.0


def xfem_step(fail, sig, d_epsp, deps, dt, dama, elcrkini, tstar=None, eps_tot=None, dadv=1.0, is_phantom=False):
    """Forming Limit Diagram failure step with XFEM crack tracking.

    Ported from $OR_SRC/engine/source/materials/fail/fld/fail_fld_xfem.F
    Subroutine: FAIL_FLD_XFEM
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

