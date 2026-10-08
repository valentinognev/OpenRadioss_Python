"""
LAW57 — Barlat 1991 6-Parameter & Barlat-Lian 1989 3-Parameter Anisotropic Plasticity (/MAT/LAW57, /MAT/BARLAT3).

Upstream Fortran reference:
  - 3D solid constitutive kernel: engine/source/materials/mat/mat057/sigeps57.F
  - 2D shell plane-stress constitutive kernel: engine/source/materials/mat/mat057/sigeps57c.F90
  - Starter reader & setup: starter/source/materials/mat/mat057/hm_read_mat57.F90
  - Newton-Raphson solver for parameter p: starter/source/materials/mat/mat057/calculp2.F90
  - CFG card definition: hm_cfg_files/config/CFG/radioss140/MAT/matl57_BARLAT3.cfg

Theory
------
Barlat 1991 anisotropic yield criterion for 3D solids:
    Phi = |s1 - s2|^m + |s2 - s3|^m + |s3 - s1|^m = 2 * sigma_y^m
    sigma_eq = (0.5 * Phi)**(1 / m)
where s1, s2, s3 are the principal stresses of the transformed stress tensor:
    s_xx = [c*(sigma_xx - sigma_yy) - b*(sigma_zz - sigma_xx)] / 3
    s_yy = [a*(sigma_yy - sigma_zz) - c*(sigma_xx - sigma_yy)] / 3
    s_zz = [b*(sigma_zz - sigma_xx) - a*(sigma_yy - sigma_zz)] / 3
    s_yz = f * sigma_yz
    s_zx = g * sigma_zx
    s_xy = h * sigma_xy

Barlat-Lian (1989) non-quadratic anisotropic yield function in plane stress:
    Phi = 0.5 * (a * |K1 + K2|^m + a * |K1 - K2|^m + c * |2*K2|^m)
    sigma_eq = Phi^(1/m)

where in-plane stress invariants are defined by:
    K1 = (sigma_xx + h_bar * sigma_yy) / 2
    K2 = sqrt(((sigma_xx - h_bar * sigma_yy) / 2)^2 + p^2 * sigma_xy^2)

Anisotropy constants derived from Lankford coefficients (R00, R45, R90):
    r = R00 / (1 + R00)
    h_orig = R90 / (1 + R90)
    c = 2 * sqrt(r * h_orig)
    a = 2 - c
    h_bar = sqrt(r / h_orig)
    p = calculp2(a, c, h_bar, 1.0, m, R45)

Isotropic recovery:
    R00 = R45 = R90 = 1.0 => r = 0.5, h_orig = 0.5, c = 1.0, a = 1.0, h_bar = 1.0, p = 1.0.
    For m = 2, Barlat-Lian reduces identically to von Mises plane stress.

Mixed Isotropic / Kinematic Hardening:
    F_isokin = CHARD in [0, 1]
    sigma_y = (1 - F_isokin) * Y(pla, rate) + F_isokin * Y0(rate)
    Backstress tensor alpha updates along plastic normal:
    dalpha_xx = (2/3) * H_k * (2 * norm_xx + norm_yy) * dlam
    dalpha_yy = (2/3) * H_k * (2 * norm_yy + norm_xx) * dlam
    dalpha_xy = (2/3) * H_k * norm_xy * dlam

Dynamic Young's Modulus Degradation:
    E(pla) = E0 - (E0 - Einf) * (1 - exp(-CE * pla))   (or tabulated ifunce)

Tensile Failure Damage & Element Deletion:
    FAIL = clamp((EPSR2 - epst) / (EPSR2 - EPSR1), 0.0, 1.0)
    where epst = 0.5 * (eps_xx + eps_yy + sqrt((eps_xx - eps_yy)^2 + eps_xy^2))
    Element eroded/deleted if pla >= EPSMAX or epst >= EPSR2.

Shell Thickness Thinning:
    deps_zz = -nu / (1 - nu) * (deps_xx + deps_yy) - (deps_xx^p + deps_yy^p)
    thk_new = thk + deps_zz * thkly * off
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM30 = 1.0e-30
_INF = 1.0e30


# ============================================================================
# Parameter Structures & Anisotropy Solver
# ============================================================================

class BarlatParams(NamedTuple):
    """Barlat 1989 yield criterion anisotropic parameters."""
    r: float
    h: float
    c: float
    a: float
    h_bar: float
    p: float


class Barlat1991Params(NamedTuple):
    """Barlat 1991 yield criterion 6-parameter anisotropic constants."""
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    f: float = 1.0
    g: float = 1.0
    h: float = 1.0
    m: float = 6.0


def calculp2(a: float, c: float, h_bar: float, p: float = 1.0,
             m: float = 6.0, r45: float = 1.0) -> float:
    """Newton-Raphson solver for parameter p in Barlat-Lian (1989).

    Upstream Fortran reference:
      starter/source/materials/mat/mat057/calculp2.F90
    """
    two = 2.0
    fourth = 0.25
    half = 0.5
    one = 1.0
    nmax = 10

    m2 = m - two
    c1 = (two ** m) * c
    c2 = fourth * (one - h_bar)
    c2 = c2 * c2
    c3 = fourth * (one + h_bar)
    c4 = half * (one + h_bar)
    pp = float(p)

    for _ in range(nmax):
        gama = math.sqrt(c2 + fourth * pp * pp)
        if gama < _EM30:
            break
        alpha = c3 - gama
        beta = c3 + gama
        c5 = two * c2 / gama
        aba2 = abs(alpha) ** m2
        abb2 = abs(beta) ** m2
        ca = aba2 * (c4 - c5) * (one + r45)
        cb = abb2 * (c4 + c5) * (one + r45)
        ca1 = aba2 * alpha
        cb1 = abb2 * beta
        abg1 = c1 * (gama ** (m - one))
        f = a * (alpha * (ca1 - ca) + beta * (cb1 - cb)) + gama * abg1
        df = a * ((m - one) * (ca - cb) - (ca1 - cb1) * (m + (one + r45) * c5 / gama)) + m * abg1
        df = half * df * pp / gama
        if abs(df) < _EM30:
            break
        pp = pp - f / df

    return pp


def barlat_params(r0: float = 1.0, r45: float = 1.0, r90: float = 1.0,
                  m: float = 6.0) -> BarlatParams:
    """Compute Barlat 1989 anisotropic constants from Lankford coefficients.

    Follows starter/source/materials/mat/mat057/hm_read_mat57.F90 lines 160-252.
    """
    r00_val = float(r0) if float(r0) > 0.0 else 1.0
    r45_val = float(r45) if float(r45) > 0.0 else 1.0
    r90_val = float(r90) if float(r90) > 0.0 else 1.0
    m_val = float(m) if float(m) > 0.0 else 6.0

    r = r00_val / (1.0 + r00_val)
    h_orig = r90_val / (1.0 + r90_val)
    c = 2.0 * math.sqrt(r * h_orig)
    a = 2.0 - c
    h_bar = math.sqrt(r / h_orig)
    p = calculp2(a, c, h_bar, 1.0, m_val, r45_val)

    return BarlatParams(r=r, h=h_orig, c=c, a=a, h_bar=h_bar, p=p)


@dataclass
class Law57Params:
    """Strongly-typed parameters for /MAT/LAW57 (/MAT/BARLAT3)."""
    id: int = 1
    law: int = 57
    title: str = "LAW57_BARLAT"
    rho0: float = 1.0
    rhor: float = 1.0
    E: float = 210000.0
    nu: float = 0.3
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    m: float = 6.0
    fisokin: float = 0.0
    epsmax: float = _INF
    epsr1: float = _INF
    epsr2: float = 2.0 * _INF
    asrate: float = 0.0
    israte: int = 0
    vp: int = 0
    ifunce: int = 0
    einf: float = 0.0
    ce: float = 0.0
    g5: float = 0.0
    shf: float = 5.0 / 6.0
    sigy0: float = 1.0e30
    funct_ids: List[int] = field(default_factory=list)
    yfac: List[float] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)
    curve_x: List[np.ndarray] = field(default_factory=list)
    curve_y: List[np.ndarray] = field(default_factory=list)
    curve_s: List[np.ndarray] = field(default_factory=list)
    E_curve_x: Optional[np.ndarray] = None
    E_curve_y: Optional[np.ndarray] = None
    E_curve_s: Optional[np.ndarray] = None
    a_barlat: float = 1.0
    b_barlat: float = 1.0
    c_barlat: float = 1.0
    f_barlat: float = 1.0
    g_barlat: float = 1.0
    h_barlat: float = 1.0

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def bulk(self) -> float:
        return self.E / max(3.0 * (1.0 - 2.0 * self.nu), 1.0e-15)

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def A11(self) -> float:
        return self.E / max(1.0 - self.nu ** 2, 1.0e-15)

    @property
    def A12(self) -> float:
        return self.nu * self.A11

    @property
    def barlat(self) -> BarlatParams:
        return barlat_params(self.r00, self.r45, self.r90, self.m)

    @property
    def barlat1991(self) -> Barlat1991Params:
        if (self.a_barlat == 1.0 and self.b_barlat == 1.0 and self.c_barlat == 1.0 and
            self.f_barlat == 1.0 and self.g_barlat == 1.0 and self.h_barlat == 1.0 and
            (self.r00 != 1.0 or self.r45 != 1.0 or self.r90 != 1.0)):
            a, b, c, f, g, h = calibrate_barlat1991(self.r00, self.r45, self.r90, self.m)
            return Barlat1991Params(a=a, b=b, c=c, f=f, g=g, h=h, m=self.m)
        return Barlat1991Params(
            a=self.a_barlat,
            b=self.b_barlat,
            c=self.c_barlat,
            f=self.f_barlat,
            g=self.g_barlat,
            h=self.h_barlat,
            m=self.m,
        )

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def nu0(self) -> float:
        return self.nu

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


def _get_params(mat: Any) -> Law57Params:
    """Extract or adapt Law57Params from a Law57Params, Material, or dict."""
    if isinstance(mat, Law57Params):
        return mat
    if hasattr(mat, "params") and isinstance(mat.params, Law57Params):
        return mat.params
    return build_law57(mat)


# ============================================================================
# Barlat Equivalent Stress & Evaluation
# ============================================================================

def barlat_equivalent_stress(
    sig: np.ndarray,
    a: float,
    c: float,
    h_bar: float,
    p: float,
    m: float,
) -> np.ndarray:
    """Compute Barlat-Lian (1989) equivalent stress for 2D plane stress.

    Input sig: (n, 3) or (3,) in-plane stress [sig_xx, sig_yy, sig_xy].
    Output: equivalent stress array of shape (n,) or scalar float.
    """
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2]

    normsig = np.sqrt(sxx * sxx + syy * syy + 2.0 * sxy * sxy)
    normsig = np.maximum(normsig, 1.0)

    k1 = 0.5 * (sxx + h_bar * syy) / normsig
    k2 = np.sqrt((0.5 * (sxx - h_bar * syy)) ** 2 + (p * sxy) ** 2) / normsig

    phi = (a * (np.abs(k1 + k2) ** m)
           + a * (np.abs(k1 - k2) ** m)
           + c * (np.abs(2.0 * k2) ** m))

    pos = phi > 0.0
    seq = np.zeros_like(phi)
    seq[pos] = np.exp((1.0 / m) * np.log(0.5 * phi[pos])) * normsig[pos]

    if is_1d:
        return float(seq[0])
    return seq


# ============================================================================
# Barlat 1991 3D Anisotropic Yield Surface & Normal Gradient
# ============================================================================

def barlat1991_yield_function(
    sig: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    f: float = 1.0,
    g: float = 1.0,
    h: float = 1.0,
    m: float = 6.0,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Evaluate Barlat 1991 3D yield function value and equivalent stress.

    Upstream Fortran reference:
      $OR_SRC/engine/source/materials/mat/mat057/sigeps57.F

    Theory
    ------
    Yield surface:
        Phi = |s1 - s2|^m + |s2 - s3|^m + |s3 - s1|^m = 2 * sigma_y^m
        sigma_eq = (0.5 * Phi)**(1 / m)

    where s1, s2, s3 are the principal stresses of the transformed stress tensor s:
        s_xx = [c*(sigma_xx - sigma_yy) - b*(sigma_zz - sigma_xx)] / 3
        s_yy = [a*(sigma_yy - sigma_zz) - c*(sigma_xx - sigma_yy)] / 3
        s_zz = [b*(sigma_zz - sigma_xx) - a*(sigma_yy - sigma_zz)] / 3
        s_yz = f * sigma_yz
        s_zx = g * sigma_zx
        s_xy = h * sigma_xy

    Returns
    -------
    phi : yield function value |s1-s2|^m + |s2-s3|^m + |s3-s1|^m
    seq : Barlat equivalent stress (0.5 * phi)**(1 / m)
    """
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]

    n = sig_arr.shape[0]
    phi_res = np.zeros(n, dtype=float)
    seq_res = np.zeros(n, dtype=float)

    for i in range(n):
        sxx = ((c + b) * sig_arr[i, 0] - c * sig_arr[i, 1] - b * sig_arr[i, 2]) / 3.0
        syy = (-c * sig_arr[i, 0] + (a + c) * sig_arr[i, 1] - a * sig_arr[i, 2]) / 3.0
        szz = (-b * sig_arr[i, 0] - a * sig_arr[i, 1] + (a + b) * sig_arr[i, 2]) / 3.0
        sxy = h * sig_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        syz = f * sig_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        szx = g * sig_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        S = np.array([
            [sxx, sxy, szx],
            [sxy, syy, syz],
            [szx, syz, szz]
        ], dtype=float)

        vals = np.linalg.eigvalsh(S)
        s1, s2, s3 = vals[0], vals[1], vals[2]

        d12 = abs(s1 - s2)
        d23 = abs(s2 - s3)
        d31 = abs(s3 - s1)

        phi = d12**m + d23**m + d31**m
        phi_res[i] = phi
        if phi > 0.0:
            seq_res[i] = (0.5 * phi) ** (1.0 / m)

    if is_1d:
        return float(phi_res[0]), float(seq_res[0])
    return phi_res, seq_res


def barlat1991_equivalent_stress(
    sig: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    f: float = 1.0,
    g: float = 1.0,
    h: float = 1.0,
    m: float = 6.0,
) -> Union[float, np.ndarray]:
    """Compute Barlat 1991 equivalent stress (0.5 * Phi)**(1 / m).

    Upstream Fortran reference: sigeps57.F
    """
    _, seq = barlat1991_yield_function(sig, a, b, c, f, g, h, m)
    return seq


def barlat1991_gradient(
    sig: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    f: float = 1.0,
    g: float = 1.0,
    h: float = 1.0,
    m: float = 6.0,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """Compute Barlat 1991 equivalent stress and its gradient d(sigma_eq)/d(sigma).

    Upstream Fortran reference:
      $OR_SRC/engine/source/materials/mat/mat057/sigeps57.F

    Returns
    -------
    seq  : equivalent stress (float if 1D, (n,) if 2D)
    grad : normal vector to yield surface in stress space (shape (6,) or (n, 6)).
           Voigt order: [xx, yy, zz, xy, yz, zx].
    """
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]

    n = sig_arr.shape[0]
    seq_res = np.zeros(n, dtype=float)
    grad_res = np.zeros((n, 6), dtype=float)

    for i in range(n):
        sxx = ((c + b) * sig_arr[i, 0] - c * sig_arr[i, 1] - b * sig_arr[i, 2]) / 3.0
        syy = (-c * sig_arr[i, 0] + (a + c) * sig_arr[i, 1] - a * sig_arr[i, 2]) / 3.0
        szz = (-b * sig_arr[i, 0] - a * sig_arr[i, 1] + (a + b) * sig_arr[i, 2]) / 3.0
        sxy = h * sig_arr[i, 3] if sig_arr.shape[1] > 3 else 0.0
        syz = f * sig_arr[i, 4] if sig_arr.shape[1] > 4 else 0.0
        szx = g * sig_arr[i, 5] if sig_arr.shape[1] > 5 else 0.0

        S = np.array([
            [sxx, sxy, szx],
            [sxy, syy, syz],
            [szx, syz, szz]
        ], dtype=float)

        vals, vecs = np.linalg.eigh(S)
        s1, s2, s3 = vals[0], vals[1], vals[2]

        d12 = s1 - s2
        d23 = s2 - s3
        d31 = s3 - s1

        ad12 = abs(d12)
        ad23 = abs(d23)
        ad31 = abs(d31)

        phi = ad12**m + ad23**m + ad31**m
        if phi <= 0.0:
            continue

        seq = (0.5 * phi) ** (1.0 / m)
        seq_res[i] = seq

        if seq < _EM20:
            continue

        psi1 = m * (ad12**(m - 1.0)) * math.copysign(1.0, d12) if ad12 > _EM30 else 0.0
        psi2 = m * (ad23**(m - 1.0)) * math.copysign(1.0, d23) if ad23 > _EM30 else 0.0
        psi3 = m * (ad31**(m - 1.0)) * math.copysign(1.0, d31) if ad31 > _EM30 else 0.0

        dphi_ds1 = psi1 - psi3
        dphi_ds2 = -psi1 + psi2
        dphi_ds3 = -psi2 + psi3

        M = (dphi_ds1 * np.outer(vecs[:, 0], vecs[:, 0]) +
             dphi_ds2 * np.outer(vecs[:, 1], vecs[:, 1]) +
             dphi_ds3 * np.outer(vecs[:, 2], vecs[:, 2]))

        fac = (seq**(1.0 - m)) / (2.0 * m)

        grad_res[i, 0] = fac * (((c + b) * M[0, 0] - c * M[1, 1] - b * M[2, 2]) / 3.0)
        grad_res[i, 1] = fac * ((-c * M[0, 0] + (a + c) * M[1, 1] - a * M[2, 2]) / 3.0)
        grad_res[i, 2] = fac * ((-b * M[0, 0] - a * M[1, 1] + (a + b) * M[2, 2]) / 3.0)
        grad_res[i, 3] = fac * (2.0 * h * M[0, 1])
        grad_res[i, 4] = fac * (2.0 * f * M[1, 2])
        grad_res[i, 5] = fac * (2.0 * g * M[2, 0])

    if is_1d:
        return float(seq_res[0]), grad_res[0]
    return seq_res, grad_res


def barlat1991_r_values(
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    f: float = 1.0,
    g: float = 1.0,
    h: float = 1.0,
    m: float = 6.0,
) -> Tuple[float, float, float]:
    """Compute Lankford anisotropy parameters (r0, r45, r90) from Barlat 1991 constants.

    Theory:
        Uniaxial tension at 0 deg:  sig = [sig, 0, 0, 0, 0, 0]
            r0 = deps_yy^p / deps_zz^p = N_yy / N_zz
        Uniaxial tension at 90 deg: sig = [0, sig, 0, 0, 0, 0]
            r90 = deps_xx^p / deps_zz^p = N_xx / N_zz
        Uniaxial tension at 45 deg: sig = [sig/2, sig/2, 0, sig/2, 0, 0]
            r45 = (0.5*(N_xx + N_yy) - 0.5*N_xy) / N_zz
    """
    ref_s = 100.0
    _, g0 = barlat1991_gradient(np.array([ref_s, 0.0, 0.0, 0.0, 0.0, 0.0]), a, b, c, f, g, h, m)
    r0 = float(g0[1] / g0[2]) if abs(g0[2]) > _EM30 else 1.0

    _, g90 = barlat1991_gradient(np.array([0.0, ref_s, 0.0, 0.0, 0.0, 0.0]), a, b, c, f, g, h, m)
    r90 = float(g90[0] / g90[2]) if abs(g90[2]) > _EM30 else 1.0

    _, g45 = barlat1991_gradient(np.array([0.5 * ref_s, 0.5 * ref_s, 0.0, 0.5 * ref_s, 0.0, 0.0]), a, b, c, f, g, h, m)
    eps_w = 0.5 * (g45[0] + g45[1]) - 0.5 * g45[3]
    r45 = float(eps_w / g45[2]) if abs(g45[2]) > _EM30 else 1.0

    return r0, r45, r90


def calibrate_barlat1991(
    r00: float = 1.0,
    r45: float = 1.0,
    r90: float = 1.0,
    m: float = 6.0,
) -> Tuple[float, float, float, float, float, float]:
    """Calibrate Barlat 1991 constants (a, b, c, f, g, h) from Lankford parameters (r00, r45, r90).

    Returns (a, b, c, f, g, h) with c=1.0, f=1.0, g=1.0.
    """
    r0_tgt = float(r00) if float(r00) > 0.0 else 1.0
    r45_tgt = float(r45) if float(r45) > 0.0 else 1.0
    r90_tgt = float(r90) if float(r90) > 0.0 else 1.0
    m_val = float(m) if float(m) > 0.0 else 6.0

    if abs(r0_tgt - 1.0) < 1.0e-6 and abs(r45_tgt - 1.0) < 1.0e-6 and abs(r90_tgt - 1.0) < 1.0e-6:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    try:
        from scipy.optimize import least_squares

        def res(x: Sequence[float]) -> List[float]:
            a_val, b_val, h_val = x
            r0, r45, r90 = barlat1991_r_values(a_val, b_val, 1.0, 1.0, 1.0, h_val, m_val)
            return [r0 - r0_tgt, r45 - r45_tgt, r90 - r90_tgt]

        sol = least_squares(res, [1.0, 1.0, 1.0], bounds=([0.05, 0.05, 0.05], [10.0, 10.0, 10.0]))
        a_fit, b_fit, h_fit = sol.x
        return float(a_fit), float(b_fit), 1.0, 1.0, 1.0, float(h_fit)
    except Exception:
        # Fallback approximation based on Hill48 equivalence
        h_fit = math.sqrt(max(0.01, (1.0 + 2.0 * r45_tgt) / (1.0 + 2.0 * r0_tgt)))
        b_fit = math.sqrt(max(0.01, 1.0 / (1.0 + r0_tgt)))
        a_fit = math.sqrt(max(0.01, 1.0 / (1.0 + r90_tgt)))
        return float(a_fit), float(b_fit), 1.0, 1.0, 1.0, float(h_fit)


# ============================================================================
# Curve Evaluation Utilities
# ============================================================================

def _curve_eval(cx: np.ndarray, cy: np.ndarray, cs: np.ndarray,
                e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear evaluation with end-slope extrapolation (Radioss FINTER)."""
    e_arr = np.asarray(e, dtype=float)
    if len(cx) <= 1:
        val = cy[0] if len(cy) > 0 else 0.0
        return np.full_like(e_arr, val), np.zeros_like(e_arr)

    idx = np.minimum(np.maximum(np.searchsorted(cx, e_arr, side="right") - 1, 0),
                     len(cx) - 2)
    val = cy[idx] + cs[idx] * (e_arr - cx[idx])
    return val, cs[idx]


def _eval_young(p: Law57Params, pla: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute current degraded Young's modulus E, A11, A12, G.

    Follows sigeps57c.F90 lines 200-216.
    """
    pla_arr = np.asarray(pla, dtype=float)
    e0 = p.E
    nu = p.nu
    e = np.full_like(pla_arr, e0, dtype=float)

    if p.ifunce > 0 and p.E_curve_x is not None and len(p.E_curve_x) > 0:
        val, _ = _curve_eval(p.E_curve_x, p.E_curve_y, p.E_curve_s, pla_arr)
        e = val
    elif p.ce > 0.0 and p.einf > 0.0:
        pos = pla_arr > 0.0
        if np.any(pos):
            e[pos] = e0 - (e0 - p.einf) * (1.0 - np.exp(-p.ce * pla_arr[pos]))

    a11 = e / max(1.0 - nu * nu, 1.0e-15)
    a12 = nu * a11
    g = e / (2.0 * (1.0 + nu))
    return e, a11, a12, g


def _eval_yield_stress(p: Law57Params, epsp: np.ndarray, rate: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate tabulated yield stress, hardening slope, and initial yield stress.

    Follows sigeps57c.F90 lines 183-187 and 271-281.
    """
    cxs = p.curve_x
    cys = p.curve_y
    css = p.curve_s
    nfun = len(cxs)

    epsp_arr = np.asarray(epsp, dtype=float)
    zeros_arr = np.zeros_like(epsp_arr)

    if nfun == 0:
        sigy0 = p.sigy0
        zeros = np.zeros_like(epsp_arr)
        return np.full_like(epsp_arr, sigy0), zeros, np.full_like(epsp_arr, sigy0)

    if nfun == 1:
        y, h = _curve_eval(cxs[0], cys[0], css[0], epsp_arr)
        y0, _ = _curve_eval(cxs[0], cys[0], css[0], zeros_arr)
        return y, h, y0

    rates = np.asarray(p.rates, dtype=float)
    rate_arr = np.asarray(rate, dtype=float)

    vals = np.empty((nfun, len(epsp_arr)), dtype=float)
    slps = np.empty((nfun, len(epsp_arr)), dtype=float)
    vals0 = np.empty((nfun, len(epsp_arr)), dtype=float)
    for k in range(nfun):
        vals[k], slps[k] = _curve_eval(cxs[k], cys[k], css[k], epsp_arr)
        vals0[k], _ = _curve_eval(cxs[k], cys[k], css[k], zeros_arr)

    j = np.clip(np.searchsorted(rates, rate_arr, side="right") - 1, 0, nfun - 2)
    denom = np.maximum(rates[j + 1] - rates[j], _EM20)
    w = np.clip((rate_arr - rates[j]) / denom, 0.0, 1.0)

    cols = np.arange(len(epsp_arr))
    sy = (1.0 - w) * vals[j, cols] + w * vals[j + 1, cols]
    h_slope = (1.0 - w) * slps[j, cols] + w * slps[j + 1, cols]
    sy0 = (1.0 - w) * vals0[j, cols] + w * vals0[j + 1, cols]

    return sy, h_slope, sy0


# ============================================================================
# Sound Speed & Allocations
# ============================================================================

def sound_speed_shell_law57(params: Any, rho0: Optional[float] = None,
                            extra: Optional[Dict[str, Any]] = None) -> float:
    """Return thin-shell acoustic sound speed: sqrt(E / ((1 - nu^2) * rho0)).

    Upstream Fortran reference: sigeps57c.F90 line 603.
    """
    rho_val = float(rho0) if rho0 is not None else getattr(params, "rho0", getattr(params, "rho", getattr(params, "density", None)))
    p = _get_params(params)
    if rho_val is None:
        rho_val = p.rho0
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        try:
            r_ex = float(np.asarray(extra["rho"]).flatten()[0])
            if r_ex > 0.0:
                rho_val = r_ex
        except Exception:
            pass

    rho_val = max(rho_val, _EM20)
    a11 = p.A11
    return float(math.sqrt(max(0.0, a11 / rho_val)))


sound_speed_shell = sound_speed_shell_law57


def sound_speed_solid_law57(params: Any, rho0: Optional[float] = None,
                            extra: Optional[Dict[str, Any]] = None) -> float:
    """Return 3D solid acoustic longitudinal wave speed: sqrt((K + 4/3*G) / rho0).

    Upstream Fortran reference: sigeps57.F
    """
    rho_val = float(rho0) if rho0 is not None else getattr(params, "rho0", getattr(params, "rho", getattr(params, "density", None)))
    p = _get_params(params)
    if rho_val is None:
        rho_val = p.rho0
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        try:
            r_ex = float(np.asarray(extra["rho"]).flatten()[0])
            if r_ex > 0.0:
                rho_val = r_ex
        except Exception:
            pass
    rho_val = max(rho_val, _EM20)
    k = p.bulk
    g = p.G
    return float(math.sqrt(max(0.0, (k + (4.0 / 3.0) * g) / rho_val)))


def sound_speed(params: Any, rho0: Optional[float] = None,
                extra: Optional[Dict[str, Any]] = None,
                is_shell: bool = False, **kwargs: Any) -> float:
    """Sound speed dispatcher for LAW57."""
    if is_shell:
        return sound_speed_shell_law57(params, rho0=rho0, extra=extra)
    return sound_speed_solid_law57(params, rho0=rho0, extra=extra)


class SolidUpdateResult(tuple):
    """Result tuple for solid_update containing (sig, epsp, c_sound)."""
    def __new__(cls, sig: Any, epsp: Any, c_sound: Any):
        return super().__new__(cls, (sig, epsp, c_sound))

    @property
    def sig(self) -> Any:
        return self[0]

    @property
    def epsp(self) -> Any:
        return self[1]

    @property
    def pla(self) -> Any:
        return self[1]

    @property
    def c_sound(self) -> Any:
        return self[2]


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Union[SolidUpdateResult, Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
    """3D solid constitutive update for Barlat 1991 anisotropic plasticity with radial return.

    Upstream Fortran reference:
      $OR_SRC/engine/source/materials/mat/mat057/sigeps57.F

    Yield surface:
        Phi = |s1 - s2|^m + |s2 - s3|^m + |s3 - s1|^m = 2 * sigma_y^m
        sigma_eq = (0.5 * Phi)**(1 / m)
    Radial return:
        Newton-Raphson iteration on plastic multiplier dlam.
        Isochoric plastic flow: tr(deps_p) = 0 => volumetric response is purely elastic.
    """
    p = _get_params(mat)

    sig_arr = np.asarray(sig, dtype=float).copy()
    deps_arr = np.asarray(deps, dtype=float).copy()
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]
        deps_arr = deps_arr[None, :]

    n = sig_arr.shape[0]

    # Plastic strain extraction
    if extra is not None and "pla57" in extra and extra["pla57"] is not None:
        pla = np.asarray(extra["pla57"], dtype=float).copy().flatten()
    elif extra is not None and "pla" in extra and extra["pla"] is not None:
        pla = np.asarray(extra["pla"], dtype=float).copy().flatten()
    elif extra is not None and "epsp" in extra and extra["epsp"] is not None:
        pla = np.asarray(extra["epsp"], dtype=float).copy().flatten()
    elif epsp is not None:
        pla = np.asarray(epsp, dtype=float).copy().flatten()
    else:
        pla = np.zeros(n, dtype=float)

    if len(pla) == 1 and n > 1:
        pla = np.full(n, float(pla[0]), dtype=float)
    elif len(pla) == 0:
        pla = np.zeros(n, dtype=float)

    # Elastic trial step
    g = p.G
    k = p.bulk
    lam = k - (2.0 / 3.0) * g

    # Volumetric strain trace and trial stress
    tr_deps = deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]

    sig_tr = np.zeros_like(sig_arr)
    sig_tr[:, 0] = sig_arr[:, 0] + lam * tr_deps + 2.0 * g * deps_arr[:, 0]
    sig_tr[:, 1] = sig_arr[:, 1] + lam * tr_deps + 2.0 * g * deps_arr[:, 1]
    sig_tr[:, 2] = sig_arr[:, 2] + lam * tr_deps + 2.0 * g * deps_arr[:, 2]
    sig_tr[:, 3] = sig_arr[:, 3] + g * deps_arr[:, 3]
    if sig_arr.shape[1] > 4:
        sig_tr[:, 4] = sig_arr[:, 4] + g * deps_arr[:, 4]
    if sig_arr.shape[1] > 5:
        sig_tr[:, 5] = sig_arr[:, 5] + g * deps_arr[:, 5]

    # Anisotropic parameters
    bp91 = p.barlat1991
    a_val = float(kwargs.get("a_barlat", kwargs.get("a", bp91.a)))
    b_val = float(kwargs.get("b_barlat", kwargs.get("b", bp91.b)))
    c_val = float(kwargs.get("c_barlat", kwargs.get("c", bp91.c)))
    f_val = float(kwargs.get("f_barlat", kwargs.get("f", bp91.f)))
    g_val = float(kwargs.get("g_barlat", kwargs.get("g", bp91.g)))
    h_val = float(kwargs.get("h_barlat", kwargs.get("h", bp91.h)))
    m_val = float(kwargs.get("m", bp91.m))

    # Determine strain rate for hardening curves
    epsd = np.zeros(n, dtype=float)
    if dt > 0.0:
        tr3 = tr_deps / 3.0
        exx = deps_arr[:, 0] - tr3
        eyy = deps_arr[:, 1] - tr3
        ezz = deps_arr[:, 2] - tr3
        exy = 0.5 * deps_arr[:, 3]
        eyz = 0.5 * deps_arr[:, 4] if deps_arr.shape[1] > 4 else 0.0
        ezx = 0.5 * deps_arr[:, 5] if deps_arr.shape[1] > 5 else 0.0
        ee = exx**2 + eyy**2 + ezz**2 + 2.0 * (exy**2 + eyz**2 + ezx**2)
        epsd = np.sqrt(np.maximum(0.0, (2.0 / 3.0) * ee)) / max(dt, _EM20)

    yld, _, _ = _eval_yield_stress(p, pla, epsd)

    # Newton-Raphson return mapping
    sig_out = np.zeros_like(sig_tr)
    epsp_out = pla.copy()
    dpla = np.zeros(n, dtype=float)

    max_iter = 30
    tol = 1.0e-7

    for i in range(n):
        s_tr_i = sig_tr[i]
        seq_tr, _ = barlat1991_gradient(s_tr_i, a_val, b_val, c_val, f_val, g_val, h_val, m_val)
        sy_i = yld[i]

        if seq_tr <= sy_i:
            sig_out[i] = s_tr_i
            continue

        dlam = 0.0
        s_cur = s_tr_i.copy()

        for it in range(max_iter):
            seq_i, N_i = barlat1991_gradient(s_cur, a_val, b_val, c_val, f_val, g_val, h_val, m_val)
            sy_cur, h_cur, _ = _eval_yield_stress(p, np.array([epsp_out[i] + dlam]), np.array([epsd[i]]))
            res = seq_i - sy_cur[0]

            if abs(res) < tol * max(sy_i, 1.0) or abs(res) < 1.0e-9:
                break

            N_star = np.array([N_i[0], N_i[1], N_i[2], 0.5 * N_i[3], 0.5 * N_i[4], 0.5 * N_i[5]])
            denom = 2.0 * g * float(np.dot(N_i, N_star)) + max(float(h_cur[0]), 0.0)
            ddlam = res / max(denom, 1.0e-12)
            dlam += ddlam
            s_cur -= 2.0 * g * ddlam * N_star

        sig_out[i] = s_cur
        dpla[i] = dlam
        epsp_out[i] += dlam

    # Sound speed calculation
    c_sound_val = sound_speed_solid_law57(p, extra=extra)
    c_sound = np.full(n, c_sound_val, dtype=float)

    # State update in extra
    if extra is not None:
        for k in ("pla57", "pla", "epsp"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = epsp_out
            elif k in extra:
                extra[k] = epsp_out[0] if is_1d else epsp_out
        if not any(k in extra for k in ("pla57", "pla", "epsp")):
            extra["pla57"] = epsp_out[0] if is_1d else epsp_out.copy()

        for k in ("sig", "sig57"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k][:] = sig_out[0] if (extra[k].ndim == 1 and is_1d) else sig_out
            elif k in extra:
                extra[k] = sig_out[0] if is_1d else sig_out

    if is_1d:
        sig_res = sig_out[0]
        epsp_res = float(epsp_out[0])
        c_res = float(c_sound[0])
    else:
        sig_res = sig_out
        epsp_res = epsp_out
        c_res = c_sound

    if not return_sound_speed:
        return sig_res, epsp_res
    return SolidUpdateResult(sig_res, epsp_res, c_res)


def solid_update_law57(*args: Any, **kwargs: Any) -> Any:
    """Solid update entry point with backward compatibility for shell-only materials."""
    if len(args) == 0:
        raise NotImplementedError(
            "LAW57 (/MAT/LAW57, /MAT/BARLAT3) is a 2D plane-stress anisotropic model "
            "implemented for shell elements only. 3D solid elements are not supported."
        )
    if len(args) >= 1:
        mat = args[0]
        if not getattr(mat, "is_3d", False) and not (isinstance(mat, dict) and mat.get("is_3d")):
            raise NotImplementedError(
                "LAW57 (/MAT/LAW57, /MAT/BARLAT3) is a 2D plane-stress anisotropic model "
                "implemented for shell elements only. 3D solid elements are not supported."
            )
    return solid_update(*args, **kwargs)


solid_update_3d = solid_update
solid_update_barlat1991 = solid_update
solid_update_barlat = solid_update


def solid_tangent(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    epsp_incr: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    symmetric: bool = False,
    h: float = 1.0e-7,
    **kwargs: Any,
) -> np.ndarray:
    """3D algorithmic consistent tangent tensor for LAW57 Barlat 1991 (6x6).

    When deps is given, computes the algorithmic tangent via central finite differences:
        D_ij = (sigma_i(deps + h*e_j) - sigma_i(deps - h*e_j)) / (2*h)
    Otherwise returns the isotropic elastic stiffness matrix C_el (6x6).
    """
    p = _get_params(mat)

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float).copy()
        is_1d = (deps_arr.ndim == 1)
        if is_1d:
            deps_arr = deps_arr[None, :]
        n = deps_arr.shape[0]

        sig_arr = np.asarray(sig, dtype=float).copy() if sig is not None else np.zeros((n, 6), dtype=float)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[None, :]
        if sig_arr.shape[0] == 1 and n > 1:
            sig_arr = np.repeat(sig_arr, n, axis=0)

        epsp_arr = np.asarray(epsp, dtype=float).copy() if epsp is not None else np.zeros(n, dtype=float)
        if epsp_arr.ndim == 0:
            epsp_arr = np.full(n, float(epsp_arr))

        D = np.zeros((n, 6, 6), dtype=float)
        h_step = float(h)

        for j in range(6):
            ej = np.zeros_like(deps_arr)
            ej[:, j] = h_step

            ext_p = _copy_extra(extra)
            ext_m = _copy_extra(extra)

            res_p = solid_update(p, sig_arr.copy(), deps_arr + ej, epsp=epsp_arr.copy(), dt=dt, extra=ext_p, return_sound_speed=False)
            res_m = solid_update(p, sig_arr.copy(), deps_arr - ej, epsp=epsp_arr.copy(), dt=dt, extra=ext_m, return_sound_speed=False)

            sp = res_p[0] if isinstance(res_p, tuple) else res_p
            sm = res_m[0] if isinstance(res_m, tuple) else res_m

            if sp.ndim == 1:
                sp = sp[None, :]
                sm = sm[None, :]

            D[:, :, j] = (sp[:, :6] - sm[:, :6]) / (2.0 * h_step)

        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))

        return D[0] if is_1d else D

    # Elastic 6x6 tangent
    k = p.bulk
    g = p.G
    lam = k - (2.0 / 3.0) * g

    c_mat = np.array([
        [lam + 2.0 * g, lam, lam, 0.0, 0.0, 0.0],
        [lam, lam + 2.0 * g, lam, 0.0, 0.0, 0.0],
        [lam, lam, lam + 2.0 * g, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g],
    ], dtype=float)

    if sig is not None and np.ndim(sig) > 1:
        n = np.shape(sig)[0]
        return np.broadcast_to(c_mat, (n, 6, 6)).copy()
    return c_mat


consistent_solid_tangent = solid_tangent
solid_tangent_law57 = solid_tangent
tangent_law57_solid = solid_tangent
tangent = solid_tangent


LAW_DISPATCH_METADATA = {
    "plane_stress": True,
    "solid": True,
    "shell": True,
}


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Extra allocations required in element layer state for LAW57."""
    if nip:
        return {
            "pla57": (nip,),
            "sigb57": (nip, 3),
            "off57": (nip,),
            "epsd57": (nip,),
            "thk57": (nip,),
            "dmg57": (nip, 3),
            "eps57": (nip, 3),
        }
    return {
        "pla57": (),
        "sigb57": (3,),
        "off57": (),
        "epsd57": (),
        "thk57": (),
        "dmg57": (3,),
        "eps57": (3,),
    }


# ============================================================================
# Shell Constitutive Update: shell_update_law57 (sigeps57c.F90)
# ============================================================================

class ShellUpdateResult(tuple):
    """Result tuple for shell_update containing (sig, epsp, c_sound).

    Supports unpacking as either 3 items (sig, epsp, c_sound) or
    accessing .sig, .epsp, .pla, .c_sound attributes.
    """
    def __new__(cls, sig: Any, epsp: Any, c_sound: Any):
        return super().__new__(cls, (sig, epsp, c_sound))

    @property
    def sig(self) -> Any:
        return self[0]

    @property
    def epsp(self) -> Any:
        return self[1]

    @property
    def pla(self) -> Any:
        return self[1]

    @property
    def c_sound(self) -> Any:
        return self[2]


def shell_update_law57(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    *args: Any,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Plane-stress constitutive update for shells under Barlat-Lian (1989) plasticity.

    Follows engine/source/materials/mat/mat057/sigeps57c.F90.
    """
    p = _get_params(mat)

    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)
    if not isinstance(deps, np.ndarray):
        deps = np.array(deps, dtype=float)

    sig_orig = sig

    orig_shape = sig.shape
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]
        deps = deps[None, :]

    n = sig.shape[0]

    # Positional argument disambiguation
    if len(args) >= 3:
        epsp = args[0]
        dt = float(args[1])
        extra = args[2]
    elif len(args) == 2:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
            extra = args[1]
        elif isinstance(args[1], dict) or args[1] is None:
            dt = float(args[0]) if isinstance(args[0], (int, float)) else 0.0
            extra = args[1]
        else:
            epsp = args[0]
            dt = float(args[1])
    elif len(args) == 1:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
        elif isinstance(args[0], dict):
            extra = args[0]
        else:
            epsp = args[0]

    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "extra" in kwargs and kwargs["extra"] is not None:
        extra = kwargs["extra"]
    if "epsp" in kwargs and kwargs["epsp"] is not None:
        epsp = kwargs["epsp"]

    epsp_orig = epsp

    # History variables extraction
    # 1. Equivalent plastic strain (pla)
    if extra is not None and "pla57" in extra and extra["pla57"] is not None:
        pla = np.asarray(extra["pla57"], dtype=float).copy().flatten()
    elif extra is not None and "pla" in extra and extra["pla"] is not None:
        pla = np.asarray(extra["pla"], dtype=float).copy().flatten()
    elif extra is not None and "epsp" in extra and extra["epsp"] is not None:
        pla = np.asarray(extra["epsp"], dtype=float).copy().flatten()
    elif epsp is not None:
        pla = np.asarray(epsp, dtype=float).copy().flatten()
    else:
        pla = np.zeros(n, dtype=float)

    if len(pla) == 1 and n > 1:
        pla = np.full(n, float(pla[0]), dtype=float)
    elif len(pla) == 0:
        pla = np.zeros(n, dtype=float)

    # 2. Backstress tensor (sigb)
    if extra is not None and "sigb57" in extra and extra["sigb57"] is not None:
        sigb = np.asarray(extra["sigb57"], dtype=float).copy()
    elif extra is not None and "sigb" in extra and extra["sigb"] is not None:
        sigb = np.asarray(extra["sigb"], dtype=float).copy()
    else:
        sigb = np.zeros((n, 3), dtype=float)

    if sigb.ndim == 1:
        if n == 1:
            sigb = sigb[None, :3]
        else:
            sigb = np.tile(sigb[:3], (n, 1))
    elif sigb.shape[0] != n:
        sigb = np.tile(sigb[0, :3], (n, 1))

    # 3. Element deletion status (off)
    if extra is not None and "off57" in extra and extra["off57"] is not None:
        off = np.asarray(extra["off57"], dtype=float).copy().flatten()
    elif extra is not None and "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).copy().flatten()
    elif extra is not None and "layfail" in extra and extra["layfail"] is not None:
        off = np.asarray(extra["layfail"], dtype=float).copy().flatten()
    else:
        off = np.ones(n, dtype=float)

    if len(off) == 1 and n > 1:
        off = np.full(n, float(off[0]), dtype=float)
    elif len(off) == 0:
        off = np.ones(n, dtype=float)

    # Gradual deletion update (sigeps57c.F90 lines 164-167)
    for i in range(n):
        if off[i] < 0.1:
            off[i] = 0.0
        elif off[i] < 1.0:
            off[i] = off[i] * 0.8

    # 4. Damage array (dmg: [global, plastic, tensile])
    if extra is not None and "dmg57" in extra and extra["dmg57"] is not None:
        dmg = np.asarray(extra["dmg57"], dtype=float).copy()
    elif extra is not None and "dmg" in extra and extra["dmg"] is not None:
        dmg = np.asarray(extra["dmg"], dtype=float).copy()
    else:
        dmg = np.zeros((n, 3), dtype=float)

    if dmg.ndim == 1:
        if n == 1:
            dmg = dmg[None, :3]
        else:
            dmg = np.tile(dmg[:3], (n, 1))
    elif dmg.shape[0] != n:
        dmg = np.tile(dmg[0, :3], (n, 1))

    # 5. Total strain (eps)
    if extra is not None and "eps57" in extra and extra["eps57"] is not None:
        eps_tot = np.asarray(extra["eps57"], dtype=float).copy()
    elif extra is not None and "eps" in extra and extra["eps"] is not None:
        eps_tot = np.asarray(extra["eps"], dtype=float).copy()
    else:
        eps_tot = np.zeros((n, 3), dtype=float)

    if eps_tot.ndim == 1:
        eps_tot = eps_tot[None, :3]
    if eps_tot.shape[0] != n:
        eps_tot = np.tile(eps_tot[0, :3], (n, 1))
    eps_tot[:, :3] += deps[:, :3]

    # Dynamic Young's modulus evaluation (sigeps57c.F90 lines 200-216)
    young_arr, a11_arr, a12_arr, shear_arr = _eval_young(p, pla)
    nu = p.nu
    shf = float(p.shf)
    if extra is not None and "shf" in extra and extra["shf"] is not None:
        shf = float(extra["shf"])
    gs_arr = shear_arr * shf

    # Trial stress calculation with prior tensile damage (sigeps57c.F90 lines 223-230)
    dmg_fact = np.maximum(1.0 - dmg[:, 2], _EM20)
    signxx = sig[:, 0] / dmg_fact + a11_arr * deps[:, 0] + a12_arr * deps[:, 1]
    signyy = sig[:, 1] / dmg_fact + a12_arr * deps[:, 0] + a11_arr * deps[:, 1]
    signxy = sig[:, 2] / dmg_fact + shear_arr * deps[:, 2]

    has_shear = (sig.shape[1] >= 5 and deps.shape[1] >= 5)
    if has_shear:
        signyz = sig[:, 3] / dmg_fact + gs_arr * deps[:, 3]
        signzx = sig[:, 4] / dmg_fact + gs_arr * deps[:, 4]
    else:
        signyz = np.zeros(n, dtype=float)
        signzx = np.zeros(n, dtype=float)

    # Shift trial stress by backstress (sigeps57c.F90 lines 233-235)
    signxx = signxx - sigb[:, 0]
    signyy = signyy - sigb[:, 1]
    signxy = signxy - sigb[:, 2]

    # Total strain rate computation & filtering (sigeps57c.F90 lines 170-180)
    if extra is not None and "epsd_pg" in extra and extra["epsd_pg"] is not None:
        epsd_inst = np.asarray(extra["epsd_pg"], dtype=float).flatten()
        if len(epsd_inst) == 1 and n > 1:
            epsd_inst = np.full(n, float(epsd_inst[0]), dtype=float)
    elif dt > 0.0:
        epspxx = deps[:, 0] / dt
        epspyy = deps[:, 1] / dt
        epspxy = deps[:, 2] / dt
        epsd_inst = 0.5 * (np.abs(epspxx + epspyy)
                           + np.sqrt((epspxx - epspyy) ** 2 + epspxy * epspxy))
    else:
        epsd_inst = np.zeros(n, dtype=float)

    epsd_prev = np.zeros(n, dtype=float)
    if extra is not None and "epsd57" in extra and extra["epsd57"] is not None:
        epsd_prev = np.asarray(extra["epsd57"], dtype=float).flatten()
    elif extra is not None and "epsd" in extra and extra["epsd"] is not None:
        epsd_prev = np.asarray(extra["epsd"], dtype=float).flatten()

    israte_val = p.israte
    if extra is not None and "israte" in extra and extra["israte"] is not None:
        israte_val = int(extra["israte"])
    vp_val = p.vp
    if extra is not None and "vp" in extra and extra["vp"] is not None:
        vp_val = int(extra["vp"])

    if vp_val == 0:
        if israte_val == 0:
            epsd = epsd_inst.copy()
        else:
            if extra is not None and "asrate" in extra and extra["asrate"] is not None:
                alpha_rate = float(extra["asrate"])
            else:
                alpha_rate = p.asrate
                if dt > 0.0 and alpha_rate > 1.0:
                    alpha_rate = min(1.0, alpha_rate * dt)
            epsd = alpha_rate * epsd_inst + (1.0 - alpha_rate) * epsd_prev
    else:
        epsd = epsd_prev.copy()

    # Barlat anisotropic parameters
    bp = p.barlat
    a_param = bp.a
    c_param = bp.c
    h_bar = bp.h_bar
    p_param = bp.p
    m_param = p.m

    # Initial yield stress at pla=0 for kinematic hardening (sigeps57c.F90 lines 183-187)
    _, _, yld0 = _eval_yield_stress(p, np.zeros(n, dtype=float), epsd)

    # Current yield stress at pla (sigeps57c.F90 lines 271-281)
    yld, dyld_dp, _ = _eval_yield_stress(p, pla, epsd)

    fisokin = p.fisokin
    yld = (1.0 - fisokin) * yld + fisokin * yld0
    hk = fisokin * dyld_dp
    dyld_dp = (1.0 - fisokin) * dyld_dp

    # Trial equivalent stress (sigeps57c.F90 lines 243-265)
    normsig = np.sqrt(signxx * signxx + signyy * signyy + 2.0 * signxy * signxy)
    normsig = np.maximum(normsig, 1.0)

    k1 = 0.5 * (signxx + h_bar * signyy) / normsig
    k2 = np.sqrt((0.5 * (signxx - h_bar * signyy)) ** 2 + (p_param * signxy) ** 2) / normsig

    phi_k = (a_param * (np.abs(k1 + k2) ** m_param)
             + a_param * (np.abs(k1 - k2) ** m_param)
             + c_param * (np.abs(2.0 * k2) ** m_param))
    pos = phi_k > 0.0
    seq = np.zeros(n, dtype=float)
    seq[pos] = np.exp((1.0 / m_param) * np.log(0.5 * phi_k[pos])) * normsig[pos]

    # Plastic check
    phi = (seq / np.maximum(yld, _EM20)) ** 2 - 1.0
    yielding = (phi >= 0.0) & (off == 1.0)

    dpla = np.zeros(n, dtype=float)
    deplzz = np.zeros(n, dtype=float)
    niter = 3

    if np.any(yielding):
        idx = np.where(yielding)[0]

        for _ in range(niter):
            for i in idx:
                s_ratio = seq[i] / normsig[i]
                p1 = s_ratio ** (1.0 - m_param)

                kp = k1[i] + k2[i]
                km = k1[i] - k2[i]
                t1 = abs(kp) ** (m_param - 1.0)
                t2 = abs(km) ** (m_param - 1.0)
                s1 = math.copysign(1.0, kp)
                s2 = math.copysign(1.0, km)

                dseq_dk1 = p1 * (a_param / 2.0) * (s1 * t1 + s2 * t2)
                dseq_dk2 = p1 * ((a_param / 2.0) * (s1 * t1 - s2 * t2)
                                 + c_param * (abs(2.0 * k2[i]) ** (m_param - 1.0)))

                dk1_dsxx = 0.5
                dk1_dsyy = h_bar / 2.0
                denom_k2 = max(normsig[i] * 4.0 * k2[i], _EM20)
                dk2_dsxx = (signxx[i] - h_bar * signyy[i]) / denom_k2
                dk2_dsyy = -h_bar * (signxx[i] - h_bar * signyy[i]) / denom_k2
                dk2_dsxy = (p_param ** 2) * signxy[i] / max(normsig[i] * k2[i], _EM20)

                dseq_dsxx = dseq_dk1 * dk1_dsxx + dseq_dk2 * dk2_dsxx
                dseq_dsyy = dseq_dk1 * dk1_dsyy + dseq_dk2 * dk2_dsyy
                dseq_dsxy = dseq_dk2 * dk2_dsxy

                dphi_dseq = 2.0 * (seq[i] / max(yld[i] ** 2, _EM20))
                normxx = dphi_dseq * dseq_dsxx
                normyy = dphi_dseq * dseq_dsyy
                normxy = dphi_dseq * dseq_dsxy

                dsigxx_dlam = -a11_arr[i] * normxx - a12_arr[i] * normyy
                dsigyy_dlam = -a11_arr[i] * normyy - a12_arr[i] * normxx
                dsigxy_dlam = -shear_arr[i] * normxy

                dphidsig_dsigdlam = normxx * dsigxx_dlam + normyy * dsigyy_dlam + normxy * dsigxy_dlam

                dphi_dyld = -2.0 * (seq[i] ** 2 / max(yld[i] ** 3, _EM20))
                dphi_dpla = dphi_dyld * dyld_dp[i]

                sig_dphidsig = signxx[i] * normxx + signyy[i] * normyy + signxy[i] * normxy
                dpla_dlam = sig_dphidsig / max(yld[i], _EM20)

                dphi_dsigbxx = -normxx
                dphi_dsigbyy = -normyy
                dphi_dsigbxy = -normxy

                dsigbxx_dlam = (2.0 / 3.0) * hk[i] * (2.0 * normxx + normyy)
                dsigbyy_dlam = (2.0 / 3.0) * hk[i] * (2.0 * normyy + normxx)
                dsigbxy_dlam = (2.0 / 3.0) * hk[i] * normxy

                dphidsig_dsigbdlam = (dphi_dsigbxx * dsigbxx_dlam
                                      + dphi_dsigbyy * dsigbyy_dlam
                                      + dphi_dsigbxy * dsigbxy_dlam)

                dphi_dlam = dphidsig_dsigdlam + dphi_dpla * dpla_dlam + dphidsig_dsigbdlam
                if abs(dphi_dlam) < _EM20:
                    dphi_dlam = math.copysign(_EM20, dphi_dlam)

                dlam = -phi[i] / dphi_dlam

                ddep = dpla_dlam * dlam
                dpla[i] = max(0.0, dpla[i] + ddep)
                pla[i] += ddep

                deplzz[i] -= dlam * normxx + dlam * normyy

                # Total stress advance
                signxx[i] += sigb[i, 0] + dsigxx_dlam * dlam
                signyy[i] += sigb[i, 1] + dsigyy_dlam * dlam
                signxy[i] += sigb[i, 2] + dsigxy_dlam * dlam

                # Backstress advance
                sigb[i, 0] += dsigbxx_dlam * dlam
                sigb[i, 1] += dsigbyy_dlam * dlam
                sigb[i, 2] += dsigbxy_dlam * dlam

                # Shifted stress
                signxx[i] -= sigb[i, 0]
                signyy[i] -= sigb[i, 1]
                signxy[i] -= sigb[i, 2]

                # Update equivalent stress
                ns = math.sqrt(signxx[i]**2 + signyy[i]**2 + 2.0 * signxy[i]**2)
                normsig[i] = max(ns, 1.0)
                k1[i] = 0.5 * (signxx[i] + h_bar * signyy[i]) / normsig[i]
                k2[i] = math.sqrt((0.5 * (signxx[i] - h_bar * signyy[i]))**2
                                  + (p_param * signxy[i])**2) / normsig[i]
                pk = (a_param * (abs(k1[i] + k2[i]) ** m_param)
                      + a_param * (abs(k1[i] - k2[i]) ** m_param)
                      + c_param * (abs(2.0 * k2[i]) ** m_param))
                if pk > 0.0:
                    seq[i] = math.exp((1.0 / m_param) * math.log(0.5 * pk)) * normsig[i]
                else:
                    seq[i] = 0.0

            # Re-evaluate yield stress at current pla for yielding elements
            yld_new, dyld_dp_new, _ = _eval_yield_stress(p, pla[idx], epsd[idx])
            yld[idx] = (1.0 - fisokin) * yld_new + fisokin * yld0[idx]
            hk[idx] = fisokin * dyld_dp_new
            dyld_dp[idx] = (1.0 - fisokin) * dyld_dp_new
            phi[idx] = (seq[idx] / np.maximum(yld[idx], _EM20)) ** 2 - 1.0

        # Post-plasticity updates (sigeps57c.F90 lines 501-508)
        for i in idx:
            dmg[i, 1] = min(1.0, pla[i] / max(p.epsmax, _EM20))
            if pla[i] > p.epsmax and off[i] == 1.0:
                off[i] = 0.8

    if vp_val == 1:
        dpdt = dpla / max(dt, _EM20)
        if extra is not None and "asrate" in extra and extra["asrate"] is not None:
            alpha_rate = float(extra["asrate"])
        else:
            alpha_rate = p.asrate
            if dt > 0.0 and alpha_rate > 1.0:
                alpha_rate = min(1.0, alpha_rate * dt)
        epsd = alpha_rate * dpdt + (1.0 - alpha_rate) * epsd

    # Reconstitute total stresses with backstresses (sigeps57c.F90 lines 524-528)
    signxx = signxx + sigb[:, 0]
    signyy = signyy + sigb[:, 1]
    signxy = signxy + sigb[:, 2]

    # Tensile damage softening (sigeps57c.F90 lines 531-550)
    if p.epsr1 > 0.0 and p.epsr2 > 0.0 and p.epsr2 > p.epsr1:
        epst = 0.5 * (eps_tot[:, 0] + eps_tot[:, 1]
                      + np.sqrt((eps_tot[:, 0] - eps_tot[:, 1]) ** 2 + eps_tot[:, 2] ** 2))
        d_tensile = 1.0 - (p.epsr2 - epst) / (p.epsr2 - p.epsr1)
        d_tensile = np.clip(d_tensile, 0.0, 1.0)
        dmg[:, 2] = np.maximum(dmg[:, 2], d_tensile)

        scale_dam = 1.0 - dmg[:, 2]
        signxx = scale_dam * signxx
        signyy = scale_dam * signyy
        signxy = scale_dam * signxy
        if has_shear:
            signyz = scale_dam * signyz
            signzx = scale_dam * signzx

    # Shell thickness thinning increment (sigeps57c.F90 lines 596-601)
    deelzz = -nu * (signxx - sig[:, 0] + signyy - sig[:, 1]) / np.maximum(young_arr, _EM20)
    depszz = deelzz + deplzz

    thkly = np.ones(n, dtype=float)
    if extra is not None and "thkly" in extra and extra["thkly"] is not None:
        thkly = np.asarray(extra["thkly"], dtype=float).flatten()
    elif extra is not None and "thklyl" in extra and extra["thklyl"] is not None:
        thkly = np.asarray(extra["thklyl"], dtype=float).flatten()
    elif extra is not None and "thk0" in extra and extra["thk0"] is not None:
        thkly = np.asarray(extra["thk0"], dtype=float).flatten()
    elif extra is not None and "thkn" in extra and extra["thkn"] is not None:
        thkly = np.asarray(extra["thkn"], dtype=float).flatten()
    elif extra is not None and "thk" in extra and extra["thk"] is not None:
        thkly = np.asarray(extra["thk"], dtype=float).flatten()

    thk_val = thkly.copy()
    if extra is not None and "thk57" in extra and extra["thk57"] is not None:
        t_arr = np.asarray(extra["thk57"], dtype=float).flatten()
        if np.any(t_arr > 0.0):
            thk_val = t_arr.copy()
    elif extra is not None and "thk" in extra and extra["thk"] is not None:
        t_arr = np.asarray(extra["thk"], dtype=float).flatten()
        if np.any(t_arr > 0.0):
            thk_val = t_arr.copy()

    if len(thkly) == 1 and n > 1:
        thkly = np.full(n, float(thkly[0]), dtype=float)
    if len(thk_val) == 1 and n > 1:
        thk_val = np.full(n, float(thk_val[0]), dtype=float)

    thk_val = thk_val + depszz * thkly * off

    # Element deletion check: off < 0.1 or pla >= epsmax or epst >= epsr2
    deleted = (off <= 0.0) | (pla >= p.epsmax)
    if p.epsr2 < _INF:
        epst_del = 0.5 * (eps_tot[:, 0] + eps_tot[:, 1]
                          + np.sqrt((eps_tot[:, 0] - eps_tot[:, 1]) ** 2 + eps_tot[:, 2] ** 2))
        deleted = deleted | (epst_del >= p.epsr2)

    if np.any(deleted):
        off[deleted] = 0.0
        signxx[deleted] = 0.0
        signyy[deleted] = 0.0
        signxy[deleted] = 0.0
        signyz[deleted] = 0.0
        signzx[deleted] = 0.0

    # Global damage (sigeps57c.F90 line 605)
    dmg[:, 0] = np.maximum(dmg[:, 1], dmg[:, 2])

    # Sound speed (sigeps57c.F90 line 603)
    rho_cur = np.full(n, p.rho0, dtype=float)
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        r_arr = np.asarray(extra["rho"], dtype=float).flatten()
        if len(r_arr) == 1 and n > 1:
            rho_cur = np.full(n, float(r_arr[0]), dtype=float)
        elif len(r_arr) == n:
            rho_cur = r_arr
    c_sound = np.sqrt(a11_arr / np.maximum(rho_cur, _EM20))

    # Assemble output stress array
    if has_shear:
        sig_out = np.column_stack([signxx, signyy, signxy, signyz, signzx])
    else:
        sig_out = np.column_stack([signxx, signyy, signxy])

    # Write state back to extra dict
    if extra is not None:
        for k in ("pla57", "pla"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = pla
            elif k in extra:
                extra[k] = pla[0] if is_1d else pla
        if not any(k in extra for k in ("pla57", "pla")):
            extra["pla57"] = pla[0] if is_1d else pla.copy()

        for k in ("sigb57", "sigb"):
            if k in extra and isinstance(extra[k], np.ndarray):
                if extra[k].ndim == 1:
                    extra[k][:3] = sigb[0, :3]
                else:
                    extra[k][:] = sigb
            elif k in extra:
                extra[k] = sigb[0, :3] if is_1d else sigb
        if not any(k in extra for k in ("sigb57", "sigb")):
            extra["sigb57"] = sigb[0, :3] if is_1d else sigb.copy()

        found_off = False
        for k in ("off57", "off", "layfail"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = off
                found_off = True
            elif k in extra:
                extra[k] = off[0] if is_1d else off
                found_off = True
        if not found_off:
            extra["off57"] = off[0] if is_1d else off

        for k in ("dmg57", "dmg"):
            if k in extra and isinstance(extra[k], np.ndarray):
                if extra[k].ndim == 1:
                    extra[k][:3] = dmg[0, :3]
                else:
                    extra[k][:] = dmg
            elif k in extra:
                extra[k] = dmg[0, :3] if is_1d else dmg
        if not any(k in extra for k in ("dmg57", "dmg")):
            extra["dmg57"] = dmg[0, :3] if is_1d else dmg.copy()

        for k in ("eps57", "eps"):
            if k in extra and isinstance(extra[k], np.ndarray):
                if extra[k].ndim == 1:
                    extra[k][:3] = eps_tot[0, :3]
                else:
                    extra[k][:, :3] = eps_tot[:, :3]
            elif k in extra:
                extra[k] = eps_tot[0, :3] if is_1d else eps_tot
        if not any(k in extra for k in ("eps57", "eps")):
            extra["eps57"] = eps_tot[0, :3] if is_1d else eps_tot.copy()

        if "thk57" in extra:
            if isinstance(extra["thk57"], np.ndarray):
                extra["thk57"].flat = thk_val
            else:
                extra["thk57"] = thk_val[0] if is_1d else thk_val
        elif "thk" in extra:
            if isinstance(extra["thk"], np.ndarray):
                extra["thk"].flat = thk_val
            else:
                extra["thk"] = thk_val[0] if is_1d else thk_val
        else:
            extra["thk57"] = thk_val[0] if is_1d else thk_val.copy()

        for k in ("epsd57", "epsd"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = epsd
            elif k in extra:
                extra[k] = epsd[0] if is_1d else epsd
        if not any(k in extra for k in ("epsd57", "epsd")):
            extra["epsd57"] = epsd[0] if is_1d else epsd.copy()

        extra["depszz"] = depszz[0] if is_1d else depszz
        extra["seq"] = seq[0] if is_1d else seq

    if is_1d:
        sig_res = sig_out[0]
        pla_res = float(pla[0])
        c_res = float(c_sound[0])
    else:
        sig_res = sig_out
        pla_res = pla
        c_res = c_sound

    if not return_sound_speed:
        return sig_res, pla_res
    return ShellUpdateResult(sig_res, pla_res, c_res)


shell_update = shell_update_law57


# ============================================================================
# Algorithmic Consistent Tangent Operator: tangent_law57_shell
# ============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep-copy arrays and nested structures in extra dictionary."""
    if extra is None:
        return None
    res: Dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        elif hasattr(v, "copy"):
            try:
                res[k] = v.copy()
            except Exception:
                res[k] = v
        else:
            res[k] = v
    return res


def tangent_law57_shell(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    epsp_incr: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    symmetric: bool = False,
    h: float = 1.0e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent plane-stress membrane tangent tensor (NEL, 3, 3) or (3, 3).

    For elastic points, returns the plane-stress elastic matrix C_el.
    When deps is provided, computes the exact algorithmic consistent tangent
    via directional central finite difference perturbation:
        D_ij = (sigma_i(deps + h*e_j) - sigma_i(deps - h*e_j)) / (2*h)
    """
    p = _get_params(mat)

    # Disambiguate keyword arguments
    if "deps" in kwargs and deps is None:
        deps = kwargs.pop("deps")
    if "eps" in kwargs and deps is None:
        deps = kwargs.pop("eps")
    if "sig" in kwargs and sig is None:
        sig = kwargs.pop("sig")
    if "epsp" in kwargs and epsp is None:
        epsp = kwargs.pop("epsp")
    if "epsp_incr" in kwargs and epsp_incr is None:
        epsp_incr = kwargs.pop("epsp_incr")
    if "extra" in kwargs and extra is None:
        extra = kwargs.pop("extra")
    if "dt" in kwargs:
        dt = float(kwargs.pop("dt"))
    if "h" in kwargs:
        h = float(kwargs.pop("h"))
    if "symmetric" in kwargs:
        symmetric = bool(kwargs.pop("symmetric"))

    # Disambiguate positional arguments
    # If called as tangent(mat, sig, epsp, epsp_incr, extra):
    # then deps is actually epsp!
    if deps is not None and not isinstance(deps, dict):
        deps_check = np.asarray(deps)
        if deps_check.ndim == 0 or (deps_check.ndim == 1 and deps_check.shape[0] not in (3, 5)) or (deps_check.ndim == 2 and deps_check.shape[1] not in (3, 5)):
            if epsp_incr is None and epsp is not None:
                epsp_incr = epsp
            epsp = deps
            deps = None
    if len(args) >= 1:
        if isinstance(args[0], dict) and extra is None:
            extra = args[0]
        elif isinstance(args[0], (int, float)):
            dt = float(args[0])
        elif isinstance(args[0], np.ndarray):
            arr = np.asarray(args[0])
            if (arr.ndim == 1 and arr.shape[0] in (3, 5)) or (arr.ndim == 2 and arr.shape[1] in (3, 5)):
                if deps is None:
                    deps = arr
            elif epsp_incr is None:
                epsp_incr = args[0]
    if len(args) >= 2:
        if isinstance(args[1], dict) and extra is None:
            extra = args[1]
        elif epsp_incr is None:
            epsp_incr = args[1]
    if len(args) >= 3 and isinstance(args[2], dict) and extra is None:
        extra = args[2]

    # Sizing and dimensionality
    n_sig = 1 if (sig is None or np.ndim(sig) <= 1) else np.asarray(sig).shape[0]
    n_deps = 1 if (deps is None or np.ndim(deps) <= 1) else np.asarray(deps).shape[0]
    nel = max(n_sig, n_deps)
    single = (sig is None or np.ndim(sig) <= 1) and (deps is None or np.ndim(deps) <= 1)

    if extra is not None:
        for k in ("pla57", "pla", "off57", "off", "layfail", "thk57", "thk"):
            if k in extra and extra[k] is not None:
                v = np.asarray(extra[k])
                if v.ndim >= 1 and len(v) > nel:
                    nel = len(v)
                    single = False

    if sig is not None:
        sig_arr = np.asarray(sig, dtype=float).copy()
        if sig_arr.ndim == 1:
            sig_arr = sig_arr.reshape(1, -1)
        if sig_arr.shape[0] == 1 and nel > 1:
            sig_arr = np.repeat(sig_arr, nel, axis=0)
    else:
        sig_arr = np.zeros((nel, 3), dtype=float)

    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float).copy()
        if deps_arr.ndim == 1:
            deps_arr = deps_arr.reshape(1, -1)
        if deps_arr.shape[0] == 1 and nel > 1:
            deps_arr = np.repeat(deps_arr, nel, axis=0)
    else:
        deps_arr = None

    if epsp is not None:
        epsp_arr = np.asarray(epsp, dtype=float).copy().flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)
    elif extra is not None and "pla57" in extra and extra["pla57"] is not None:
        epsp_arr = np.asarray(extra["pla57"], dtype=float).copy().flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)
    elif extra is not None and "pla" in extra and extra["pla"] is not None:
        epsp_arr = np.asarray(extra["pla"], dtype=float).copy().flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)
    elif extra is not None and "epsp" in extra and extra["epsp"] is not None:
        epsp_arr = np.asarray(extra["epsp"], dtype=float).copy().flatten()
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)
    else:
        epsp_arr = np.zeros(nel, dtype=float)

    # Element deletion / erosion status
    off_arr = np.ones(nel, dtype=float)
    if extra is not None:
        for k in ("off57", "off", "layfail"):
            if k in extra and extra[k] is not None:
                val = np.asarray(extra[k], dtype=float).flatten()
                if len(val) == 1 and nel > 1:
                    off_arr = np.full(nel, val[0], dtype=float)
                else:
                    off_arr = val.copy()
                break

    deleted_mask = (off_arr <= 0.0) | (epsp_arr >= p.epsmax)

    if extra is not None:
        for k in ("dmg57", "dmg"):
            if k in extra and extra[k] is not None:
                dmg_val = np.asarray(extra[k], dtype=float)
                if dmg_val.ndim == 1 and len(dmg_val) == 3:
                    dmg_val = dmg_val.reshape(1, 3)
                if dmg_val.shape[0] == nel:
                    deleted_mask = deleted_mask | (dmg_val[:, 2] >= 1.0 - 1e-12)
                break
        if p.epsr2 < _INF:
            for k in ("eps57", "eps"):
                if k in extra and extra[k] is not None:
                    eps_tot_val = np.asarray(extra[k], dtype=float)
                    if eps_tot_val.ndim == 1:
                        eps_tot_val = eps_tot_val.reshape(1, -1)
                    if eps_tot_val.shape[0] == 1 and nel > 1:
                        eps_tot_val = np.repeat(eps_tot_val, nel, axis=0)
                    if eps_tot_val.shape[0] == nel:
                        epst = 0.5 * (eps_tot_val[:, 0] + eps_tot_val[:, 1]
                                      + np.sqrt((eps_tot_val[:, 0] - eps_tot_val[:, 1]) ** 2 + eps_tot_val[:, 2] ** 2))
                        deleted_mask = deleted_mask | (epst >= p.epsr2)
                    break

    # If deps is provided, compute numerical algorithmic perturbation
    if deps_arr is not None:
        D = np.zeros((nel, 3, 3), dtype=float)
        active = ~deleted_mask
        h_step = float(h)

        if np.any(active):
            for j in range(3):
                ej = np.zeros_like(deps_arr)
                ej[:, j] = h_step

                ext_p = _copy_extra(extra)
                ext_m = _copy_extra(extra)

                res_p = shell_update_law57(p, sig_arr.copy(), deps_arr + ej, epsp=epsp_arr.copy(), dt=dt, extra=ext_p, return_sound_speed=False)
                res_m = shell_update_law57(p, sig_arr.copy(), deps_arr - ej, epsp=epsp_arr.copy(), dt=dt, extra=ext_m, return_sound_speed=False)

                sp = res_p[0] if isinstance(res_p, tuple) else res_p
                sm = res_m[0] if isinstance(res_m, tuple) else res_m

                if sp.ndim == 1:
                    sp = sp.reshape(1, -1)
                    sm = sm.reshape(1, -1)

                D[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h_step)

        if np.any(deleted_mask):
            D[deleted_mask] = 0.0

        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))

        return D[0] if single else D

    # Pure elastic membrane tangent (with degraded modulus if active)
    _, a11_arr, a12_arr, shear_arr = _eval_young(p, epsp_arr)
    C_el = np.zeros((nel, 3, 3), dtype=float)
    for i in range(nel):
        if not deleted_mask[i]:
            C_el[i, 0, 0] = a11_arr[i]
            C_el[i, 0, 1] = a12_arr[i]
            C_el[i, 1, 0] = a12_arr[i]
            C_el[i, 1, 1] = a11_arr[i]
            C_el[i, 2, 2] = shear_arr[i]

    if symmetric:
        C_el = 0.5 * (C_el + np.swapaxes(C_el, -1, -2))

    return C_el[0] if single else C_el


consistent_shell_tangent = tangent_law57_shell
shell_tangent = tangent_law57_shell
shell_membrane_tangent = tangent_law57_shell


# ============================================================================
# Material Construction & Registry
# ============================================================================

def build_law57(rec: Any = None, **kwargs: Any) -> Law57Params:
    """Physics constructor for /MAT/LAW57 (/MAT/BARLAT3).

    Extracts parameters per hm_read_mat57.F90:
      - rho0, rhor
      - E, nu
      - ifunce, einf, ce
      - r00, r45, r90, fisokin, m
      - epsmax, epsr1, epsr2
      - asrate, israte, vp
      - tabulated yield curves FunctionIds, ABG_cpa, ABG_cpb
    """
    if rec is None:
        p: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        _title = str(kwargs.get("title", "LAW57_BARLAT"))
    elif isinstance(rec, Law57Params):
        return rec
    elif isinstance(rec, dict):
        base = rec.get("params", rec)
        p = {**base, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        _title = str(rec.get("title", kwargs.get("title", "LAW57_BARLAT")))
    elif hasattr(rec, "params"):
        base = rec.params if isinstance(rec.params, dict) else rec.params.__dict__
        p = {**base, **kwargs}
        if hasattr(rec, "rho0") and "rho0" not in p and "rho" not in p:
            p["rho0"] = getattr(rec, "rho0")
        elif hasattr(rec, "density") and "density" not in p and "rho0" not in p:
            p["density"] = getattr(rec, "density")
        elif hasattr(rec, "rho") and "rho" not in p and "rho0" not in p:
            p["rho"] = getattr(rec, "rho")
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW57_BARLAT")))
    elif hasattr(rec, "__dict__"):
        p = {**rec.__dict__, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW57_BARLAT")))
    else:
        p = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        _title = str(kwargs.get("title", "LAW57_BARLAT"))

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _geti(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return int(p[k])
                except (ValueError, TypeError):
                    pass
        return int(default)

    rho0 = _get(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], 1.0)
    rhor = _get(["Refer_Rho", "rhor", "MAT_REFRHO"], rho0)
    if rhor <= 0.0:
        rhor = rho0

    e0 = _get(["MAT_E", "E", "e", "young", "Young", "E0"], 210000.0)
    nu = _get(["MAT_NU", "NU", "nu"], 0.3)
    if nu >= 0.5:
        nu = 0.499
    if nu < 0.0:
        nu = 0.0

    ifunce = _geti(["MAT_fct_IDE", "ifunce", "IFUNCE", "funct_e"], 0)
    einf = _get(["MAT_EA", "einf", "Einf", "EINF"], 0.0)
    ce = _get(["MAT_CE", "ce", "CE"], 0.0)

    r00 = _get(["MAT_R00", "r00", "R00"], 1.0)
    if r00 <= 0.0:
        r00 = 1.0
    r45 = _get(["MAT_R45", "r45", "R45"], 1.0)
    if r45 <= 0.0:
        r45 = 1.0
    r90 = _get(["MAT_R90", "r90", "R90"], 1.0)
    if r90 <= 0.0:
        r90 = 1.0

    fisokin = _get(["MAT_CHARD", "chard", "CHARD", "fisokin", "FISOKIN"], 0.0)
    fisokin = max(0.0, min(1.0, fisokin))

    m = _get(["MAT_M", "m", "M"], 6.0)
    if m <= 0.0:
        m = 6.0

    epsmax = _get(["MAT_EPS", "epsmax", "EPSMAX", "eps_max", "epsp_max"], _INF)
    if epsmax <= 0.0:
        epsmax = _INF
    epsr1 = _get(["MAT_EPST1", "epsr1", "EPSR1", "epst1", "eps_t1"], _INF)
    if epsr1 <= 0.0:
        epsr1 = _INF
    epsr2 = _get(["MAT_EPST2", "epsr2", "EPSR2", "epst2", "eps_t2"], 2.0 * _INF)
    if epsr2 <= 0.0:
        epsr2 = 2.0 * _INF

    asrate = _get(["Fcut", "fcut", "FCUT", "asrate", "ASRATE"], 0.0)
    israte = _geti(["Fsmooth", "fsmooth", "FSMOOTH", "israte", "ISRATE"], 0)
    vp = _geti(["MAT_VP", "vp", "VP"], 0)
    vp = max(0, min(1, vp))

    if vp == 0:
        if asrate != 0.0:
            israte = 1
        elif israte != 0:
            asrate = 10000.0
        else:
            asrate = 0.0
    else:
        israte = 1
        if asrate == 0.0:
            asrate = 10000.0

    sigy0 = _get(["sigy0", "SIGY0", "MAT_SIGY", "sigy"], 1.0e30)
    g5 = _get(["G5", "g5"], 0.0)
    shf = _get(["shf", "SHF"], 5.0 / 6.0)

    a_barlat = _get(["a_barlat", "A_BARLAT", "a_bar", "a91", "a"], 1.0)
    b_barlat = _get(["b_barlat", "B_BARLAT", "b_bar", "b91", "b"], 1.0)
    c_barlat = _get(["c_barlat", "C_BARLAT", "c_bar", "c91", "c"], 1.0)
    f_barlat = _get(["f_barlat", "F_BARLAT", "f_bar", "f91", "f"], 1.0)
    g_barlat = _get(["g_barlat", "G_BARLAT", "g_bar", "g91", "g"], 1.0)
    h_barlat = _get(["h_barlat", "H_BARLAT", "h_bar", "h91", "h"], 1.0)

    # Process curves
    funct_ids: List[int] = []
    yfacs: List[float] = []
    rates: List[float] = []
    cxs: List[np.ndarray] = []
    cys: List[np.ndarray] = []
    css: List[np.ndarray] = []

    if "curves" in p and isinstance(p["curves"], (list, tuple)):
        for item in p["curves"]:
            if hasattr(item, "fct_id"):
                f_int = int(getattr(item, "fct_id", 0))
                if f_int != 0:
                    funct_ids.append(f_int)
                    yfacs.append(float(getattr(item, "fscale", 1.0)) if float(getattr(item, "fscale", 1.0)) != 0.0 else 1.0)
                    rates.append(float(getattr(item, "eps", 0.0)))
                continue
            if isinstance(item, dict) and ("fct_id" in item or "funct_id" in item):
                f_int = int(item.get("fct_id", item.get("funct_id", 0)))
                if f_int != 0:
                    funct_ids.append(f_int)
                    yfacs.append(float(item.get("fscale", 1.0)) if float(item.get("fscale", 1.0)) != 0.0 else 1.0)
                    rates.append(float(item.get("eps", 0.0)))
                continue
            if isinstance(item, (list, tuple, np.ndarray)):
                if len(item) == 3:
                    x_c, y_c, r_c = item
                elif len(item) == 2:
                    x_c, y_c = item
                    r_c = 0.0
                else:
                    continue
                x_arr = np.asarray(x_c, dtype=float)
                y_arr = np.asarray(y_c, dtype=float)
                s_arr = (np.diff(y_arr) / np.maximum(np.diff(x_arr), _EM20)
                         if len(x_arr) > 1 else np.zeros(0, dtype=float))
                cxs.append(x_arr)
                cys.append(y_arr)
                css.append(s_arr)
                rates.append(float(r_c))

    if "FunctionIds" in p:
        fids = list(p["FunctionIds"]) if isinstance(p["FunctionIds"], (list, tuple, np.ndarray)) else [p["FunctionIds"]]
        yfs = list(p.get("ABG_cpa", [1.0] * len(fids)))
        rts = list(p.get("ABG_cpb", [0.0] * len(fids)))
        for f, y, r_c in zip(fids, yfs, rts):
            f_int = int(f)
            if f_int != 0:
                funct_ids.append(f_int)
                yfacs.append(float(y) if float(y) != 0.0 else 1.0)
                rates.append(float(r_c))

    e_cx = p.get("E_curve_x")
    e_cy = p.get("E_curve_y")
    e_cs = p.get("E_curve_s")
    if "E_curve" in p and p["E_curve"] is not None:
        ec = p["E_curve"]
        if isinstance(ec, (list, tuple)) and len(ec) == 2:
            e_cx = np.asarray(ec[0], dtype=float)
            e_cy = np.asarray(ec[1], dtype=float)
    if e_cx is not None and e_cy is not None:
        e_cx = np.asarray(e_cx, dtype=float)
        e_cy = np.asarray(e_cy, dtype=float)
        if e_cs is None:
            e_cs = (np.diff(e_cy) / np.maximum(np.diff(e_cx), _EM20)
                    if len(e_cx) > 1 else np.zeros(0, dtype=float))

    params = Law57Params(
        id=_id,
        title=_title,
        rho0=rho0,
        rhor=rhor,
        E=e0,
        nu=nu,
        r00=r00,
        r45=r45,
        r90=r90,
        m=m,
        fisokin=fisokin,
        epsmax=epsmax,
        epsr1=epsr1,
        epsr2=epsr2,
        asrate=asrate,
        israte=israte,
        vp=vp,
        ifunce=ifunce,
        einf=einf,
        ce=ce,
        g5=g5,
        shf=shf,
        sigy0=sigy0,
        funct_ids=funct_ids,
        yfac=yfacs,
        rates=rates,
        curve_x=cxs,
        curve_y=cys,
        curve_s=css,
        E_curve_x=e_cx,
        E_curve_y=e_cy,
        E_curve_s=e_cs,
        a_barlat=a_barlat,
        b_barlat=b_barlat,
        c_barlat=c_barlat,
        f_barlat=f_barlat,
        g_barlat=g_barlat,
        h_barlat=h_barlat,
    )
    return params


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT references into stored numpy arrays in Law57Params."""
    p = _get_params(mat)

    if hasattr(model, "functions"):
        functions = model.functions
    elif isinstance(model, dict):
        functions = model
    else:
        functions = {}

    cxs: List[np.ndarray] = []
    cys: List[np.ndarray] = []
    css: List[np.ndarray] = []

    for fid, yf in zip(p.funct_ids, p.yfac):
        fct = functions.get(fid)
        if fct is not None:
            if hasattr(fct, "x"):
                x_arr = np.asarray(fct.x, dtype=float).copy()
                y_arr = np.asarray(fct.y, dtype=float).copy() * yf
            else:
                x_arr = np.asarray(fct[0], dtype=float).copy()
                y_arr = np.asarray(fct[1], dtype=float).copy() * yf
            if hasattr(fct, "slope"):
                s_arr = np.asarray(fct.slope, dtype=float).copy() * yf
            elif len(x_arr) > 1:
                s_arr = np.diff(y_arr) / np.maximum(np.diff(x_arr), _EM20)
            else:
                s_arr = np.zeros(0, dtype=float)
            cxs.append(x_arr)
            cys.append(y_arr)
            css.append(s_arr)

    if len(cxs) > 0:
        p.curve_x = cxs
        p.curve_y = cys
        p.curve_s = css

    if p.ifunce > 0:
        fct_e = functions.get(p.ifunce)
        if fct_e is not None:
            if hasattr(fct_e, "x"):
                x_e = np.asarray(fct_e.x, dtype=float).copy()
                y_e = np.asarray(fct_e.y, dtype=float).copy()
            else:
                x_e = np.asarray(fct_e[0], dtype=float).copy()
                y_e = np.asarray(fct_e[1], dtype=float).copy()
            s_e = (np.diff(y_e) / np.maximum(np.diff(x_e), _EM20)
                   if len(x_e) > 1 else np.zeros(0, dtype=float))
            p.E_curve_x = x_e
            p.E_curve_y = y_e
            p.E_curve_s = s_e

    if p.sigy0 >= 1.0e29 and len(p.curve_y) > 0 and len(p.curve_y[0]) > 0:
        p.sigy0 = float(p.curve_y[0][0])

    if hasattr(mat, "params") and not isinstance(mat.params, Law57Params):
        mat.params = p


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in ("57", 57, "LAW57", "BARLAT", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT"):
            MAT_PHYSICS_REGISTRY[key] = build_law57
    except Exception:
        pass


_register()
