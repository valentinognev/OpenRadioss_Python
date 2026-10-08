"""
Bi-quadratic failure criterion (/FAIL/BIQUAD).

Fortran origin: ``engine/source/materials/fail/biquad/fail_biquad_s.F``
(solids) / ``fail_biquad_c.F`` (shells); reader
``starter/source/materials/fail/biquad/hm_read_fail_biquad.F``.

Theory
------
Like Johnson–Cook failure, plastic strain accumulates damage weighted by
a triaxiality-dependent failure strain,

    D += d_eps_p / eps_f(sigma*),      break at D >= 1,

but eps_f(sigma*) is given by TWO parabolas fitted through five
calibration points at the canonical triaxialities of standard coupon
tests:

    sigma* :  -1/3        0        1/3        2/3         1
    eps_f  :   c1        c2        c3         c4         c5
            (compression) (shear) (uniaxial) (plane strain) (biaxial)

    parabola 1 through (-1/3, c1), (0, c2), (1/3, c3)   used sigma* <= 1/3
    parabola 2 through (1/3, c3), (2/3, c4), (1, c5)    used sigma* >  1/3

The two meet at the uniaxial point c3, so eps_f is continuous. Outside
[-1/3, 1] each parabola extrapolates (Radioss does the same); a small
positive floor guards against a parabola dipping through zero far outside
its fitted range.

Presets and Options
-------------------
- M_flag (biquad_coefficients.F lines 70–122): Built-in material presets
  deriving c1, c2, c4, c5 from c3:
    1: Mild Steel (default if M_flag > 0 or c1=c2=c4=c5=0)
    2: DP600
    3: Boron
    4: Aluminium AA5182
    5: Aluminium AA6082-T6
    6: Plastic PA6GF30
    7: Plastic PP T40
    99: User scaling factors e1..e4
- S_flag (fail_biquad_s.F lines 176–205):
    1: Raw parabola through (1/3, c3), (2/3, c4), (1, c5)
    2: Split high parabolas meeting at plane strain triaxiality
       sigma* = 1/sqrt(3) with zero slope (default in Radioss)
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_FLOOR = 1e-6      # eps_f floor way outside the fitted range


def _parabola(x0, y0, x1, y1, x2, y2):
    """Coefficients (a, b, c) of y = a x^2 + b x + c through 3 points."""
    # Lagrange form condensed for the equally-spaced points used here
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) \
        / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1
         + x0 * x1 * (x0 - x1) * y2) / denom
    return a, b, c


def fit(params: dict) -> None:
    """Pre-compute the parabola coefficients from c1..c5 and M_Flag/S_Flag."""
    c3 = params.get("c3", 0.0)
    
    # M_flag presets (biquad_coefficients.F)
    m_flag = params.get("m_flag", 0)
    if m_flag > 0 or (params.get("c1", 0.0) == 0.0 and params.get("c2", 0.0) == 0.0 
                      and params.get("c4", 0.0) == 0.0 and params.get("c5", 0.0) == 0.0):
        if m_flag == 2:    # DP600
            c1, c2, c4, c5 = 4.3 * c3, 1.4 * c3, 0.6 * c3, 1.6 * c3
        elif m_flag == 3:  # Boron
            c1, c2, c4, c5 = 5.2 * c3, 3.1 * c3, 0.8 * c3, 3.5 * c3
        elif m_flag == 4:  # AA5182
            c1, c2, c4, c5 = 5.0 * c3, 1.0 * c3, 0.4 * c3, 0.8 * c3
        elif m_flag == 5:  # AA6082-T6
            c1, c2, c4, c5 = 7.8 * c3, 3.5 * c3, 0.6 * c3, 2.8 * c3
        elif m_flag == 6:  # PA6GF30
            c1, c2, c4, c5 = 3.6 * c3, 0.6 * c3, 0.5 * c3, 0.6 * c3
        elif m_flag == 7:  # PP T40
            c1, c2, c4, c5 = 10.0 * c3, 2.7 * c3, 0.6 * c3, 0.7 * c3
        elif m_flag == 99: # user scaling factors
            e1, e2 = params.get("e1", 0.0), params.get("e2", 0.0)
            e3, e4 = params.get("e3", 0.0), params.get("e4", 0.0)
            c1, c2, c4, c5 = e1 * c3, e2 * c3, e3 * c3, e4 * c3
        else:              # m_flag == 1 or anything else -> Mild Steel
            c1, c2, c4, c5 = 3.5 * c3, 1.6 * c3, 0.6 * c3, 1.5 * c3
        # write back the resolved params so they can be inspected
        params.update({"c1": c1, "c2": c2, "c4": c4, "c5": c5})
    else:
        c1, c2, c4, c5 = (params.get(k, 0.0) for k in ("c1", "c2", "c4", "c5"))

    params["plow"] = _parabola(-1.0 / 3.0, c1, 0.0, c2, 1.0 / 3.0, c3)
    
    # S_flag = 2 creates two high parabolas meeting at plane strain with zero slope
    # plane strain triax = 1/sqrt(3) ~= 0.57735
    s_flag = params.get("s_flag", 2)
    if s_flag == 3:
        inst = params.get("inst_start", 0.0)
        if inst <= 0.0 or inst >= c4:
            s_flag = 2

    if s_flag == 2:
        # P1 = (1/3, c3), S1 = (1/sqrt(3), c4), P2 = (2/3, c5)
        # matching biquad_coefficients.F and fail_biquad_s.F:180-202
        sqr3 = np.sqrt(3.0)
        s1x = 1.0 / sqr3
        raw_ah, raw_bh, raw_ch = _parabola(1.0 / 3.0, c3, s1x, c4, 2.0 / 3.0, c5)
        s1y = raw_ah * s1x**2 + raw_bh * s1x + raw_ch
        
        # Parabola 2a through P1=(1/3, c3) with zero slope at S1=(s1x, s1y)
        p1x, p1y = 1.0 / 3.0, c3
        a1 = (p1y - s1y) / (p1x - s1x)**2
        b1 = -2.0 * a1 * s1x
        c1_c = a1 * s1x**2 + s1y
        params["phigh_1"] = (a1, b1, c1_c)
        
        # Parabola 2b through P2=(2/3, c5) with zero slope at S1=(s1x, s1y)
        p2x, p2y = 2.0 / 3.0, c5
        a2 = (p2y - s1y) / (p2x - s1x)**2
        b2 = -2.0 * a2 * s1x
        c2_c = a2 * s1x**2 + s1y
        params["phigh_2"] = (a2, b2, c2_c)
    else:
        sqr3 = np.sqrt(3.0)
        params["phigh"] = _parabola(1.0 / 3.0, c3, 1.0 / sqr3, c4, 2.0 / 3.0, c5)


def eps_f(fail, triax: np.ndarray) -> np.ndarray:
    """Failure strain at the given triaxiality (vectorized)."""
    params = getattr(fail, "params", fail)
    if "plow" not in params:
        fit(params)
    al, bl, cl = params["plow"]
    low = triax <= 1.0 / 3.0
    
    if "phigh" in params:
        ah, bh, ch = params["phigh"]
        e = np.where(low,
                     al * triax ** 2 + bl * triax + cl,
                     ah * triax ** 2 + bh * triax + ch)
    else:
        ah1, bh1, ch1 = params["phigh_1"]
        ah2, bh2, ch2 = params["phigh_2"]
        s1x = 1.0 / np.sqrt(3.0)
        high1 = (triax > 1.0 / 3.0) & (triax <= s1x)
        high2 = (triax > s1x)
        e = np.where(low, al * triax ** 2 + bl * triax + cl, 0.0)
        e = np.where(high1, ah1 * triax ** 2 + bh1 * triax + ch1, e)
        e = np.where(high2, ah2 * triax ** 2 + bh2 * triax + ch2, e)
        
    return np.maximum(e, _FLOOR)


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """3-D damage step (deps/dt unused: no rate term in BIQUAD)."""
    sm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    s0, s1, s2 = sig[:, 0] - sm, sig[:, 1] - sm, sig[:, 2] - sm
    vm = np.sqrt(1.5 * (s0 ** 2 + s1 ** 2 + s2 ** 2)
                 + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
    triax = sm / np.maximum(vm, _TINY)
    triax = np.clip(triax, -2.0 / 3.0, 2.0 / 3.0)
    dama += np.maximum(d_epsp, 0.0) / eps_f(fail, triax)
    np.minimum(dama, 1.0, out=dama)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Plane-stress damage step for one layer."""
    sm = (sig[:, 0] + sig[:, 1]) / 3.0
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2
                 + 3.0 * sig[:, 2] ** 2)
    triax = sm / np.maximum(vm, _TINY)
    dama += np.maximum(d_epsp, 0.0) / eps_f(fail, triax)
    np.minimum(dama, 1.0, out=dama)
    return dama >= 1.0


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, **kwargs):
    """Bi-quadratic failure step for standard beams (TYPE 3).

    # Ported from $OR_SRC/engine/source/materials/fail/biquad/fail_biquad_b.F
    Subroutine: FAIL_BIQUAD_B
    """
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
    triax = np.clip(triax, -2.0 / 3.0, 2.0 / 3.0)
    dama += np.maximum(d_epsp, 0.0) / eps_f(fail, triax)
    np.minimum(dama, 1.0, out=dama)
    return dama >= 1.0


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, ip=0, npg=1, **kwargs):
    """Bi-quadratic failure step for integrated beam integration point (TYPE 18).

    # Ported from $OR_SRC/engine/source/materials/fail/biquad/fail_biquad_ib.F
    Subroutine: FAIL_BIQUAD_IB
    """
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
    svm = np.sqrt(3.0 * (0.5 * sig_xx**2 + sig_xy**2 + sig_xz**2))
    triax = pressure / np.maximum(svm, _TINY)
    triax = np.clip(triax, -2.0 / 3.0, 2.0 / 3.0)
    dama += np.maximum(d_epsp, 0.0) / eps_f(fail, triax)
    np.minimum(dama, 1.0, out=dama)
    return dama >= 1.0

