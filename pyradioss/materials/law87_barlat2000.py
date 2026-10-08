r"""
/MAT/LAW87 (/MAT/BARLAT2000 / /MAT/BARLAT_2000 / /MAT/BARLAT2000_2D)
Barlat Yld2000-2d Anisotropic Plasticity Model for Shells.

Reference upstream files:
  - Canonical Reference: C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat087\sigeps87.F
    (engine/source/materials/mat/mat087/sigeps87c.F90)
  - Starter reader: starter/source/materials/mat/mat087/hm_read_mat87.F90
  - Swift-Voce hardening: engine/source/materials/mat/mat087/mat87c_swift_voce.F90
  - Tabulated hardening: engine/source/materials/mat/mat087/mat87c_tabulated.F90
  - Hansel hardening: engine/source/materials/mat/mat087/mat87c_hansel.F90
  - 3-Dir Ortho Tabulated hardening: engine/source/materials/mat/mat087/mat87c_tabulated_3dir_ortho.F90
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material, MatLaw87

_EM20 = 1.0e-20
_EM30 = 1.0e-30
_INF = 1.0e30


@dataclass
class Law87Params:
    """Strongly-typed parameters for /MAT/LAW87 (/MAT/BARLAT2000)."""

    id: int = 1
    law: int = 87
    title: str = "LAW87_BARLAT2000"
    rho0: float = 1.0
    refer_rho: float = 1.0
    e: float = 210000.0
    nu: float = 0.3
    iflag: int = 0  # 0: Tabulated, 1: Swift-Voce, 2: Hansel, 3: 3-Dir Ortho
    iflagsr: int = 0  # 0: total strain rate from deps, 1: plastic strain rate from uvar
    invc: float = 0.0  # Cowper-Symonds 1/C
    invp: float = 0.0  # Cowper-Symonds 1/p
    flag_fit: int = 0  # 0: direct alphas, 1: Lankford/yield stress fitting
    al1: float = 1.0
    al2: float = 1.0
    al3: float = 1.0
    al4: float = 1.0
    al5: float = 1.0
    al6: float = 1.0
    al7: float = 1.0
    al8: float = 1.0
    fisokin: float = 0.0  # Mixed hardening factor in [0, 1] (0 = pure iso, 1 = pure kin)
    ikin: int = 1  # 1: Chaboche-Rousselier, 2: Prager
    expa: float = 2.0  # Yield exponent a (2.0 for Mises, ~6.0 BCC, ~8.0 FCC)
    fcut: float = 0.0
    israte: int = 0
    nrate: int = 0
    aswift: float = 0.0
    nexp: float = 0.0
    alpha: float = 0.0  # Weighting factor between Swift and Voce (0 = pure Voce, 1 = pure Swift)
    epso: float = 0.0
    qvoce: float = 0.0
    beta: float = 0.0
    ko: float = 0.0
    ckh: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    akh: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    funct_ids: list[int] = field(default_factory=list)
    yfac: list[float] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    curves: list[Any] = field(default_factory=list)
    yield_table: Optional[Any] = None
    table_id: int = 0
    tables: list[Any] = field(default_factory=list)
    fscale0: float = 1.0
    fscale45: float = 1.0
    fscale90: float = 1.0
    epsd0: float = 0.0
    epsd45: float = 0.0
    epsd90: float = 0.0
    # Hansel parameters (iflag=2)
    ahs: float = 0.0
    bhs: float = 0.0
    mhs: float = 0.0
    eps0hs: float = 0.0
    nhs: float = 0.0
    hmart: float = 0.0
    k1: float = 1.0
    k2: float = 0.0
    temp0: float = 293.15
    tref: float = 293.15
    eta: float = 0.9
    cp: float = 450.0
    am: float = 0.0
    bm: float = 0.0
    cm: float = 0.0
    dm: float = 0.0
    ppm: float = 0.0
    qm: float = 0.0
    e0mart: float = 0.0
    vm0: float = 0.0
    shf: float = 5.0 / 6.0
    a_hansel: float = 0.0
    b_hansel: float = 0.0
    c_hansel: float = 0.0

    @property
    def a1(self) -> float:
        return self.e / max(1.0 - self.nu ** 2, _EM20)

    @property
    def a2(self) -> float:
        return self.nu * self.a1

    @property
    def g(self) -> float:
        return 0.5 * self.e / max(1.0 + self.nu, _EM20)

    @property
    def E(self) -> float:
        return self.e

    @property
    def nu0(self) -> float:
        return self.nu

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def a(self) -> float:
        return self.expa

    @property
    def unsc(self) -> float:
        return 1.0 / self.invc if self.invc > 0.0 else 0.0

    @property
    def unsp(self) -> float:
        return 1.0 / self.invp if self.invp > 0.0 else 0.0

    def __post_init__(self):
        if self.a_hansel != 0.0 and self.am == 0.0:
            self.am = self.a_hansel
        if self.b_hansel != 0.0 and self.bm == 0.0:
            self.bm = self.b_hansel
        if self.c_hansel != 0.0 and self.cm == 0.0:
            self.cm = self.c_hansel

        if isinstance(self.ckh, (int, float)):
            self.ckh = [float(self.ckh), 0.0, 0.0, 0.0]
        else:
            self.ckh = list(self.ckh) + [0.0] * max(0, 4 - len(self.ckh))

        if isinstance(self.akh, (int, float)):
            self.akh = [float(self.akh), 0.0, 0.0, 0.0]
        else:
            self.akh = list(self.akh) + [0.0] * max(0, 4 - len(self.akh))

    @property
    def akck(self) -> float:
        return sum(a * c for a, c in zip(self.akh, self.ckh))

    # Linear transformation matrices Lp (L') and Lpp (L'')
    @property
    def lp11(self) -> float:
        return 2.0 * self.al1 / 3.0

    @property
    def lp12(self) -> float:
        return -self.al1 / 3.0

    @property
    def lp21(self) -> float:
        return -self.al2 / 3.0

    @property
    def lp22(self) -> float:
        return 2.0 * self.al2 / 3.0

    @property
    def lp66(self) -> float:
        return self.al7

    @property
    def lpp11(self) -> float:
        return (-2.0 * self.al3 + 2.0 * self.al4 + 8.0 * self.al5 - 2.0 * self.al6) / 9.0

    @property
    def lpp12(self) -> float:
        return (self.al3 - 4.0 * self.al4 - 4.0 * self.al5 + 4.0 * self.al6) / 9.0

    @property
    def lpp21(self) -> float:
        return (4.0 * self.al3 - 4.0 * self.al4 - 4.0 * self.al5 + self.al6) / 9.0

    @property
    def lpp22(self) -> float:
        return (-2.0 * self.al3 + 8.0 * self.al4 + 2.0 * self.al5 - 2.0 * self.al6) / 9.0

    @property
    def lpp66(self) -> float:
        return self.al8

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


def _get_params(mat: Any) -> Law87Params:
    """Extract or build Law87Params from a Law87Params, MatLaw87, Material, or dict."""
    if isinstance(mat, Law87Params):
        return mat
    if hasattr(mat, "params") and isinstance(mat.params, Law87Params):
        return mat.params
    return build_law87(mat)


def build_law87(mat_def: Any = None, **kwargs: Any) -> Law87Params:
    """Build a Law87Params instance from a dict, MatLaw87, Material, or keyword arguments."""
    p = Law87Params()

    if isinstance(mat_def, Law87Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)

    data.update(kwargs)

    if not data:
        return p

    def _get(keys: Sequence[str], default: Any) -> Any:
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        return default

    p.id = int(_get(["id", "mid", "mat_id"], p.id))
    p.title = str(_get(["title", "name", "titr"], p.title))
    p.rho0 = float(_get(["rho0", "rho", "MAT_RHO", "RHO", "initial_density"], p.rho0))
    p.refer_rho = float(_get(["refer_rho", "rhor", "Refer_Rho", "REF_RHO"], p.rho0))
    p.e = float(_get(["e", "E", "young", "MAT_E", "young_modulus"], p.e))
    p.nu = float(_get(["nu", "poisson", "MAT_NU", "poisson_ratio"], p.nu))
    p.iflag = int(_get(["iflag", "MAT_Iflag", "IFLAG"], p.iflag))
    p.iflagsr = int(_get(["iflagsr", "vp", "Vflag", "VFLAG"], p.iflagsr))
    p.invc = float(_get(["invc", "strain1", "STRAIN1", "c_rate", "C"], p.invc))
    p.invp = float(_get(["invp", "exp1", "MAT_EXP1", "p_rate", "P"], p.invp))
    p.flag_fit = int(_get(["flag_fit", "ifit", "Ifit", "IFIT"], p.flag_fit))

    # Read alpha anisotropic parameters
    alphas_cand = _get(["alphas", "MAT_ALPHA"], None)
    if alphas_cand is None and "alpha" in data and isinstance(data["alpha"], (list, tuple, np.ndarray)) and len(data["alpha"]) >= 8:
        alphas_cand = data["alpha"]

    if alphas_cand is not None and isinstance(alphas_cand, (list, tuple, np.ndarray)) and len(alphas_cand) >= 8:
        p.al1 = float(alphas_cand[0])
        p.al2 = float(alphas_cand[1])
        p.al3 = float(alphas_cand[2])
        p.al4 = float(alphas_cand[3])
        p.al5 = float(alphas_cand[4])
        p.al6 = float(alphas_cand[5])
        p.al7 = float(alphas_cand[6])
        p.al8 = float(alphas_cand[7])
    else:
        p.al1 = float(_get(["al1", "alpha1", "MAT_ALPHA1", "a1"], p.al1))
        p.al2 = float(_get(["al2", "alpha2", "MAT_ALPHA2", "a2"], p.al2))
        p.al3 = float(_get(["al3", "alpha3", "MAT_ALPHA3", "a3"], p.al3))
        p.al4 = float(_get(["al4", "alpha4", "MAT_ALPHA4", "a4"], p.al4))
        p.al5 = float(_get(["al5", "alpha5", "MAT_ALPHA5", "a5"], p.al5))
        p.al6 = float(_get(["al6", "alpha6", "MAT_ALPHA6", "a6"], p.al6))
        p.al7 = float(_get(["al7", "alpha7", "MAT_ALPHA7", "a7"], p.al7))
        p.al8 = float(_get(["al8", "alpha8", "MAT_ALPHA8", "a8"], p.al8))

    p.fisokin = float(_get(["fisokin", "chard", "MAT_kin", "CHARD", "f_isokin"], p.fisokin))
    p.ikin = int(_get(["ikin", "MAT_IKIN", "IKIN"], p.ikin))
    p.expa = float(_get(["expa", "exp_a", "A", "a_exp", "EXPA", "m"], p.expa))
    p.fcut = float(_get(["fcut", "Fcut", "FCUT"], p.fcut))
    p.israte = int(_get(["israte", "fsmooth", "MAT_FSMOOTH", "ISRATE"], p.israte))
    p.nrate = int(_get(["nrate", "MAT_NRATE", "NRATE"], p.nrate))

    # Swift-Voce parameters
    p.aswift = float(_get(["aswift", "MAT_A", "a_swift", "ASWIFT"], p.aswift))
    p.nexp = float(_get(["nexp", "n_hard", "FScale33", "NEXP", "n"], p.nexp))
    alpha_val = _get(["alpha_vol", "MAT_VOL", "ALPHA_VOL", "alpha_swift_voce"], None)
    if alpha_val is not None:
        p.alpha = float(alpha_val)
    elif "alpha" in data and isinstance(data["alpha"], (int, float, np.floating)):
        p.alpha = float(data["alpha"])
    p.epso = float(_get(["epso", "eps0", "FScale22", "EPS0", "epso"], p.epso))
    p.qvoce = float(_get(["qvoce", "MAT_PR", "q_voce", "QVOCE"], p.qvoce))
    p.beta = float(_get(["beta", "MAT_T0", "BETA"], p.beta))
    p.ko = float(_get(["ko", "k0", "MAT_NUt", "KO", "K0"], p.ko))

    # Kinematic hardening parameters
    ckh_in = _get(["ckh", "MAT_CRC"], None)
    if ckh_in is not None:
        if isinstance(ckh_in, (int, float)):
            p.ckh = [float(ckh_in), 0.0, 0.0, 0.0]
        elif isinstance(ckh_in, (list, tuple, np.ndarray)):
            p.ckh = [float(ckh_in[k]) if k < len(ckh_in) else 0.0 for k in range(4)]
    else:
        def_c = p.ckh if isinstance(p.ckh, (list, tuple)) else [float(p.ckh), 0.0, 0.0, 0.0]
        c1 = float(_get(["ckh1", "crc1", "MAT_CRC1"], def_c[0]))
        c2 = float(_get(["ckh2", "crc2", "MAT_CRC2"], def_c[1]))
        c3 = float(_get(["ckh3", "crc3", "MAT_CRC3"], def_c[2]))
        c4 = float(_get(["ckh4", "crc4", "MAT_CRC4"], def_c[3]))
        p.ckh = [c1, c2, c3, c4]

    akh_in = _get(["akh", "MAT_CRA"], None)
    if akh_in is not None:
        if isinstance(akh_in, (int, float)):
            p.akh = [float(akh_in), 0.0, 0.0, 0.0]
        elif isinstance(akh_in, (list, tuple, np.ndarray)):
            p.akh = [float(akh_in[k]) if k < len(akh_in) else 0.0 for k in range(4)]
    else:
        def_a = p.akh if isinstance(p.akh, (list, tuple)) else [float(p.akh), 0.0, 0.0, 0.0]
        a1 = float(_get(["akh1", "cra1", "MAT_CRA1"], def_a[0]))
        a2 = float(_get(["akh2", "cra2", "MAT_CRA2"], def_a[1]))
        a3 = float(_get(["akh3", "cra3", "MAT_CRA3"], def_a[2]))
        a4 = float(_get(["akh4", "cra4", "MAT_CRA4"], def_a[3]))
        p.akh = [a1, a2, a3, a4]

    # Tabulated yield curves / tables
    if "curves" in data:
        p.curves = data["curves"]
    if "yield_table" in data:
        p.yield_table = data["yield_table"]
    if "tables" in data:
        p.tables = data["tables"]
    p.table_id = int(_get(["table_id", "tab_id0", "TAB_ID0", "table"], p.table_id))

    p.fscale0 = float(_get(["fscale0", "FSCALE0"], p.fscale0))
    p.fscale45 = float(_get(["fscale45", "FSCALE45"], p.fscale45))
    p.fscale90 = float(_get(["fscale90", "FSCALE90"], p.fscale90))
    p.epsd0 = float(_get(["epsd0", "EPSD0"], p.epsd0))
    p.epsd45 = float(_get(["epsd45", "EPSD45"], p.epsd45))
    p.epsd90 = float(_get(["epsd90", "EPSD90"], p.epsd90))

    # Hansel parameters (iflag=2)
    p.am = float(_get(["am", "MAT_AM", "a_hansel"], p.am))
    p.bm = float(_get(["bm", "MAT_BM", "b_hansel"], p.bm))
    p.cm = float(_get(["cm", "MAT_CM", "c_hansel"], p.cm))
    p.ahs = float(_get(["ahs", "MAT_AHS"], p.ahs))
    p.bhs = float(_get(["bhs", "MAT_BHS"], p.bhs))
    p.mhs = float(_get(["mhs", "MAT_MHS"], p.mhs))
    p.eps0hs = float(_get(["eps0hs", "MAT_EPS0HS"], p.eps0hs))
    p.nhs = float(_get(["nhs", "MAT_NHS"], p.nhs))
    p.hmart = float(_get(["hmart", "MAT_HMART"], p.hmart))
    p.k1 = float(_get(["k1", "MAT_K1"], p.k1))
    p.k2 = float(_get(["k2", "MAT_K2"], p.k2))
    p.temp0 = float(_get(["temp0", "MAT_TEMP0"], p.temp0))
    p.tref = float(_get(["tref", "MAT_TREF"], p.tref))
    p.eta = float(_get(["eta", "MAT_ETA"], p.eta))
    p.cp = float(_get(["cp", "MAT_CP"], p.cp))
    p.dm = float(_get(["dm", "MAT_DM"], p.dm))
    p.ppm = float(_get(["ppm", "pm", "MAT_PM"], p.ppm))
    p.qm = float(_get(["qm", "MAT_QM"], p.qm))
    p.e0mart = float(_get(["e0mart", "MAT_E0MART"], p.e0mart))
    p.vm0 = float(_get(["vm0", "MAT_VM0"], p.vm0))

    return p


# ============================================================================
# Barlat Yld2000-2d Equivalent Stress
# ============================================================================

def barlat2000_equivalent_stress(
    sig: np.ndarray,
    p: Law87Params,
) -> Union[float, np.ndarray]:
    """Compute Barlat Yld2000-2d equivalent stress for in-plane stress tensor.

    Follows sigeps87c.F90 and mat87c_swift_voce.F90 lines 251-297:
      X'  = L'  * sig
      X'' = L'' * sig
      xp1, xp2   = principal stresses of X'
      xpp1, xpp2 = principal stresses of X''
      phi = |xp1 - xp2|^a + |2*xpp2 + xpp1|^a + |2*xpp1 + xpp2|^a
      seq = (0.5 * phi)^(1/a)
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

    # Transformed stress tensors
    xpxx = (p.lp11 * sxx + p.lp12 * syy) / normsig
    xpyy = (p.lp21 * sxx + p.lp22 * syy) / normsig
    xpxy = (p.lp66 * sxy) / normsig

    xppxx = (p.lpp11 * sxx + p.lpp12 * syy) / normsig
    xppyy = (p.lpp21 * sxx + p.lpp22 * syy) / normsig
    xppxy = (p.lpp66 * sxy) / normsig

    # Principal values of X' and X''
    r_p = np.sqrt(0.25 * (xpxx - xpyy) ** 2 + xpxy ** 2)
    c_p = 0.5 * (xpxx + xpyy)
    xp1 = c_p + r_p
    xp2 = c_p - r_p

    r_pp = np.sqrt(0.25 * (xppxx - xppyy) ** 2 + xppxy ** 2)
    c_pp = 0.5 * (xppxx + xppyy)
    xpp1 = c_pp + r_pp
    xpp2 = c_pp - r_pp

    # Yield function components
    phip = np.abs(xp1 - xp2) ** p.expa
    phipp = (np.abs(2.0 * xpp2 + xpp1) ** p.expa
             + np.abs(2.0 * xpp1 + xpp2) ** p.expa)

    phi = 0.5 * (phip + phipp)
    pos = phi > 0.0
    seq = np.zeros_like(phi)
    seq[pos] = (phi[pos] ** (1.0 / p.expa)) * normsig[pos]

    if is_1d:
        return float(seq[0])
    return seq


def barlat2000_yield_surface(
    sig: np.ndarray,
    p: Any,
    sigma_y: Optional[float] = None,
) -> Union[float, np.ndarray]:
    r"""Evaluate the Barlat Yld2000-2d yield surface.

    .. math::
        \Phi = |s_1' - s_2'|^m + |2s_2'' + s_1''|^m + |2s_1'' + s_2''|^m = 2 \bar{\sigma}^m

    If ``sigma_y`` is provided, evaluates:
    .. math::
        \phi = |s_1' - s_2'|^m + |2s_2'' + s_1''|^m + |2s_1'' + s_2''|^m - 2\sigma_y^m

    Upstream Fortran reference:
      C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat087\sigeps87.F
      (engine/source/materials/mat/mat087/mat87c_swift_voce.F90 lines 263-296)

    Parameters
    ----------
    sig : (3,) or (n, 3) ndarray
        In-plane stress components [sigma_xx, sigma_yy, sigma_xy].
    p : Law87Params or Material or dict
        Material parameters with Barlat 2000 coefficients.
    sigma_y : float, optional
        Current yield stress.

    Returns
    -------
    phi : float or ndarray
        Yield surface value.
    """
    params = _get_params(p)
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2]

    normsig = np.sqrt(sxx * sxx + syy * syy + 2.0 * sxy * sxy)
    normsig = np.maximum(normsig, 1.0)

    # Transformed stress tensors
    xpxx = (params.lp11 * sxx + params.lp12 * syy) / normsig
    xpyy = (params.lp21 * sxx + params.lp22 * syy) / normsig
    xpxy = (params.lp66 * sxy) / normsig

    xppxx = (params.lpp11 * sxx + params.lpp12 * syy) / normsig
    xppyy = (params.lpp21 * sxx + params.lpp22 * syy) / normsig
    xppxy = (params.lpp66 * sxy) / normsig

    # Principal values of X' and X''
    r_p = np.sqrt(0.25 * (xpxx - xpyy) ** 2 + xpxy ** 2)
    c_p = 0.5 * (xpxx + xpyy)
    xp1 = c_p + r_p
    xp2 = c_p - r_p

    r_pp = np.sqrt(0.25 * (xppxx - xppyy) ** 2 + xppxy ** 2)
    c_pp = 0.5 * (xppxx + xppyy)
    xpp1 = c_pp + r_pp
    xpp2 = c_pp - r_pp

    # Yield function components
    phip = np.abs(xp1 - xp2) ** params.expa
    phipp = (np.abs(2.0 * xpp2 + xpp1) ** params.expa
             + np.abs(2.0 * xpp1 + xpp2) ** params.expa)

    phi_raw = (phip + phipp) * (normsig ** params.expa)

    if sigma_y is not None:
        res = phi_raw - 2.0 * (float(sigma_y) ** params.expa)
    else:
        res = phi_raw

    if is_1d:
        return float(res[0])
    return res


def barlat2000_gradient(
    sig: np.ndarray,
    p: Any,
) -> np.ndarray:
    r"""Compute the analytical gradient of Barlat Yld2000-2d equivalent stress w.r.t in-plane stress.

    Flow direction normal:
        d(sigma_eq) / d(sigma) = [d_seq/d_sigxx, d_seq/d_sigyy, d_seq/d_sigxy]

    Upstream Fortran reference:
      C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat087\sigeps87.F
      (engine/source/materials/mat/mat087/mat87c_swift_voce.F90 lines 385-465)

    Parameters
    ----------
    sig : (3,) or (n, 3) ndarray
        In-plane stress components [sigma_xx, sigma_yy, sigma_xy].
    p : Law87Params or Material or dict
        Material parameters with Barlat 2000 coefficients.

    Returns
    -------
    grad : (3,) or (n, 3) ndarray
        Partial derivatives [d_seq/d_sigxx, d_seq/d_sigyy, d_seq/d_sigxy].
    """
    params = _get_params(p)
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    if is_1d:
        sig_arr = sig_arr[None, :]

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2]

    normsig = np.sqrt(sxx * sxx + syy * syy + 2.0 * sxy * sxy)
    normsig = np.maximum(normsig, 1.0)

    # Transformed stress tensors (normalized)
    xpxx = (params.lp11 * sxx + params.lp12 * syy) / normsig
    xpyy = (params.lp21 * sxx + params.lp22 * syy) / normsig
    xpxy = (params.lp66 * sxy) / normsig

    xppxx = (params.lpp11 * sxx + params.lpp12 * syy) / normsig
    xppyy = (params.lpp21 * sxx + params.lpp22 * syy) / normsig
    xppxy = (params.lpp66 * sxy) / normsig

    # Principal values of X' and X''
    r_p = np.sqrt(0.25 * (xpxx - xpyy) ** 2 + xpxy ** 2)
    c_p = 0.5 * (xpxx + xpyy)
    xp1 = c_p + r_p
    xp2 = c_p - r_p

    r_pp = np.sqrt(0.25 * (xppxx - xppyy) ** 2 + xppxy ** 2)
    c_pp = 0.5 * (xppxx + xppyy)
    xpp1 = c_pp + r_pp
    xpp2 = c_pp - r_pp

    # Derivatives of X' principal values w.r.t X' components
    mr_p = np.maximum(r_p, _EM20)
    dxp1dxpxx = 0.5 * (1.0 + (xpxx - xpyy) / (2.0 * mr_p))
    dxp1dxpyy = 0.5 * (1.0 - (xpxx - xpyy) / (2.0 * mr_p))
    dxp1dxpxy = xpxy / mr_p
    dxp2dxpxx = 0.5 * (1.0 - (xpxx - xpyy) / (2.0 * mr_p))
    dxp2dxpyy = 0.5 * (1.0 + (xpxx - xpyy) / (2.0 * mr_p))
    dxp2dxpxy = -xpxy / mr_p

    # Derivatives of X'' principal values w.r.t X'' components
    mr_pp = np.maximum(r_pp, _EM20)
    dxpp1dxppxx = 0.5 * (1.0 + (xppxx - xppyy) / (2.0 * mr_pp))
    dxpp1dxppyy = 0.5 * (1.0 - (xppxx - xppyy) / (2.0 * mr_pp))
    dxpp1dxppxy = xppxy / mr_pp
    dxpp2dxppxx = 0.5 * (1.0 - (xppxx - xppyy) / (2.0 * mr_pp))
    dxpp2dxppyy = 0.5 * (1.0 + (xppxx - xppyy) / (2.0 * mr_pp))
    dxpp2dxppxy = -xppxy / mr_pp

    # Chain rule to stress components
    dxp1dsigxx = dxp1dxpxx * params.lp11 + dxp1dxpyy * params.lp21
    dxp1dsigyy = dxp1dxpxx * params.lp12 + dxp1dxpyy * params.lp22
    dxp1dsigxy = dxp1dxpxy * params.lp66

    dxp2dsigxx = dxp2dxpxx * params.lp11 + dxp2dxpyy * params.lp21
    dxp2dsigyy = dxp2dxpxx * params.lp12 + dxp2dxpyy * params.lp22
    dxp2dsigxy = dxp2dxpxy * params.lp66

    dxpp1dsigxx = dxpp1dxppxx * params.lpp11 + dxpp1dxppyy * params.lpp21
    dxpp1dsigyy = dxpp1dxppxx * params.lpp12 + dxpp1dxppyy * params.lpp22
    dxpp1dsigxy = dxpp1dxppxy * params.lpp66

    dxpp2dsigxx = dxpp2dxppxx * params.lpp11 + dxpp2dxppyy * params.lpp21
    dxpp2dsigyy = dxpp2dxppxx * params.lpp12 + dxpp2dxppyy * params.lpp22
    dxpp2dsigxy = dxpp2dxppxy * params.lpp66

    # Derivative of phip w.r.t X' principal values
    diff_p = xp1 - xp2
    sgn_p = np.where(diff_p >= 0.0, 1.0, -1.0)
    dphipdxp1 = params.expa * (np.abs(diff_p) ** (params.expa - 1.0)) * sgn_p
    dphipdxp2 = -dphipdxp1

    # Derivative of phipp w.r.t X'' principal values
    term_pp1 = 2.0 * xpp2 + xpp1
    sgn_pp1 = np.where(term_pp1 >= 0.0, 1.0, -1.0)
    term_pp2 = 2.0 * xpp1 + xpp2
    sgn_pp2 = np.where(term_pp2 >= 0.0, 1.0, -1.0)

    dphippdxpp1 = (params.expa * (np.abs(term_pp1) ** (params.expa - 1.0)) * sgn_pp1
                   + 2.0 * params.expa * (np.abs(term_pp2) ** (params.expa - 1.0)) * sgn_pp2)
    dphippdxpp2 = (params.expa * (np.abs(term_pp2) ** (params.expa - 1.0)) * sgn_pp2
                   + 2.0 * params.expa * (np.abs(term_pp1) ** (params.expa - 1.0)) * sgn_pp1)

    # Assemble d(phip)/d(sig) and d(phipp)/d(sig)
    dphipdsigxx = dphipdxp1 * dxp1dsigxx + dphipdxp2 * dxp2dsigxx
    dphipdsigyy = dphipdxp1 * dxp1dsigyy + dphipdxp2 * dxp2dsigyy
    dphipdsigxy = dphipdxp1 * dxp1dsigxy + dphipdxp2 * dxp2dsigxy

    dphippdsigxx = dphippdxpp1 * dxpp1dsigxx + dphippdxpp2 * dxpp2dsigxx
    dphippdsigyy = dphippdxpp1 * dxpp1dsigyy + dphippdxpp2 * dxpp2dsigyy
    dphippdsigxy = dphippdxpp1 * dxpp1dsigxy + dphippdxpp2 * dxpp2dsigxy

    # Yield function components
    phip = np.abs(diff_p) ** params.expa
    phipp = np.abs(term_pp1) ** params.expa + np.abs(term_pp2) ** params.expa

    s_phi = 0.5 * (phip + phipp)
    dseqdphi = np.where(s_phi > 0.0, (0.5 / params.expa) * (s_phi ** (1.0 / params.expa - 1.0)), 0.0)

    dseqdsigxx = dseqdphi * (dphipdsigxx + dphippdsigxx)
    dseqdsigyy = dseqdphi * (dphipdsigyy + dphippdsigyy)
    dseqdsigxy = dseqdphi * (dphipdsigxy + dphippdsigxy)

    grad = np.column_stack([dseqdsigxx, dseqdsigyy, dseqdsigxy])
    if is_1d:
        return grad[0]
    return grad


def barlat2000_yield_and_gradient(
    sig: np.ndarray,
    p: Any,
    sigma_y: Optional[float] = None,
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    """Compute both the Barlat 2000 yield function and its stress gradient."""
    phi = barlat2000_yield_surface(sig, p, sigma_y=sigma_y)
    grad = barlat2000_gradient(sig, p)
    return phi, grad


# ============================================================================
# Yield Stress Evaluation
# ============================================================================

def _eval_curve_1d(curve: Any, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise linear evaluation of a 1D curve returning (y, dy/dx)."""
    x_arr = np.asarray(x_val, dtype=float)
    if hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=float)
        cy = np.asarray(curve.y, dtype=float)
    elif hasattr(curve, "data"):
        d = np.asarray(curve.data, dtype=float)
        cx = d[:, 0]
        cy = d[:, 1]
    elif isinstance(curve, (list, tuple, np.ndarray)):
        arr = np.asarray(curve, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            cx = arr[:, 0]
            cy = arr[:, 1]
        elif len(curve) == 2 and isinstance(curve[0], (list, tuple, np.ndarray)):
            cx = np.asarray(curve[0], dtype=float)
            cy = np.asarray(curve[1], dtype=float)
        else:
            return np.ones_like(x_arr), np.zeros_like(x_arr)
    else:
        return np.ones_like(x_arr), np.zeros_like(x_arr)

    if len(cx) <= 1:
        v = cy[0] if len(cy) > 0 else 1.0
        return np.full_like(x_arr, v), np.zeros_like(x_arr)

    cs = np.diff(cy) / np.maximum(np.diff(cx), _EM20)
    idx = np.minimum(np.maximum(np.searchsorted(cx, x_arr, side="right") - 1, 0), len(cx) - 2)
    val = cy[idx] + cs[idx] * (x_arr - cx[idx])
    return val, cs[idx]


def _eval_yield_stress(
    p: Law87Params,
    pla: np.ndarray,
    epsd: np.ndarray,
    sign: Optional[np.ndarray] = None,
    temp: Optional[np.ndarray] = None,
    uvar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate yield stress Y, hardening slope dY/dpla, and kinematic modulus hk.

    Returns:
      (yld, dylddp, hk) as 1D numpy arrays of length nel.
    """
    nel = len(pla)
    yld = np.zeros(nel, dtype=float)
    dylddp = np.zeros(nel, dtype=float)
    hk = np.zeros(nel, dtype=float)

    # 1. Swift-Voce hardening
    if p.iflag == 1:
        swift = np.zeros(nel, dtype=float)
        dswiftdp = np.zeros(nel, dtype=float)
        eff_pla = pla + p.epso
        pos_s = eff_pla > 0.0
        if np.any(pos_s):
            swift[pos_s] = p.aswift * (eff_pla[pos_s] ** p.nexp)
            dswiftdp[pos_s] = p.aswift * p.nexp * (eff_pla[pos_s] ** (p.nexp - 1.0))

        voce = p.ko + p.qvoce * (1.0 - np.exp(-p.beta * pla))
        dvocedp = p.qvoce * p.beta * np.exp(-p.beta * pla)

        yld_iso = p.alpha * swift + (1.0 - p.alpha) * voce
        dylddp_iso = p.alpha * dswiftdp + (1.0 - p.alpha) * dvocedp

        # Initial yield stress for mixed hardening
        yld0_val = (1.0 - p.alpha) * p.ko
        if p.epso > 0.0:
            yld0_val += p.alpha * p.aswift * (p.epso ** p.nexp)

        yld = (1.0 - p.fisokin) * yld_iso + p.fisokin * yld0_val
        hk = p.fisokin * dylddp_iso
        dylddp = (1.0 - p.fisokin) * dylddp_iso

        # Cowper-Symonds strain rate multiplier
        if p.unsp > 0.0 and p.unsc > 0.0:
            pos_e = epsd > 0.0
            frate = np.ones(nel, dtype=float)
            frate[pos_e] = 1.0 + (p.unsc * epsd[pos_e]) ** p.unsp
            yld *= frate
            hk *= frate
            dylddp *= frate
        elif p.invc > 0.0 and p.invp > 0.0:
            pos_e = epsd > 0.0
            frate = np.ones(nel, dtype=float)
            p_exp = 1.0 / p.invp if p.invp < 1.0 else p.invp
            c_val = p.invc if p.invc > 1.0 else 1.0 / p.invc
            frate[pos_e] = 1.0 + (epsd[pos_e] / c_val) ** p_exp
            yld *= frate
            hk *= frate
            dylddp *= frate

    # 2. Tabulated yield stress
    elif p.iflag == 0:
        if p.yield_table is not None or p.curves:
            src = None
            is_rate_table = False
            if p.curves:
                is_single = False
                try:
                    num_arr = np.asarray(p.curves, dtype=float)
                    if num_arr.ndim == 2 and num_arr.shape[1] >= 2:
                        is_single = True
                        src = num_arr
                except (ValueError, TypeError):
                    pass
                if not is_single:
                    if len(p.curves) > 1 and len(p.rates) >= len(p.curves):
                        is_rate_table = True
                    else:
                        src = p.curves[0] if len(p.curves) > 0 else None
            else:
                src = p.yield_table

            if is_rate_table:
                r_arr = np.asarray(p.rates, dtype=float)
                vals = []
                slps = []
                for c in p.curves:
                    v, s = _eval_curve_1d(c, pla)
                    vals.append(v)
                    slps.append(s)
                vals = np.array(vals)  # (ncurves, nel)
                slps = np.array(slps)
                idx = np.searchsorted(r_arr, epsd) - 1
                idx = np.clip(idx, 0, len(r_arr) - 2)
                r0 = r_arr[idx]
                r1 = r_arr[idx + 1]
                t_fact = np.clip((epsd - r0) / np.maximum(r1 - r0, _EM20), 0.0, 1.0)
                v_cur = (1.0 - t_fact) * vals[idx, np.arange(nel)] + t_fact * vals[idx + 1, np.arange(nel)]
                s_cur = (1.0 - t_fact) * slps[idx, np.arange(nel)] + t_fact * slps[idx + 1, np.arange(nel)]
                v0_cur = (1.0 - t_fact) * vals[idx, 0] + t_fact * vals[idx + 1, 0]
            else:
                v_cur, s_cur = _eval_curve_1d(src, pla)
                v0_cur, _ = _eval_curve_1d(src, np.zeros_like(pla))

            yld = (1.0 - p.fisokin) * v_cur + p.fisokin * v0_cur
            hk = p.fisokin * s_cur
            dylddp = (1.0 - p.fisokin) * s_cur
        else:
            sigy0 = p.ko if p.ko > 0.0 else (p.aswift if p.aswift > 0.0 else 1.0)
            yld.fill(sigy0)
            dylddp.fill(0.0)
            hk.fill(0.0)

    # 3. Hansel hardening (iflag=2)
    elif p.iflag == 2:
        t_arr = temp if temp is not None else np.full(nel, p.temp0)
        vm_arr = uvar[:, 0] if (uvar is not None and uvar.shape[1] >= 1) else np.full(nel, p.vm0)

        ahs_val = p.ahs if p.ahs > 0.0 else (p.ko if p.ko > 0.0 else 200.0)
        bhs_val = p.bhs if p.bhs > 0.0 else max(ahs_val + 100.0, 350.0)
        mhs_val = p.mhs if p.mhs > 0.0 else 1.0
        nhs_val = p.nhs if p.nhs > 0.0 else 0.5

        expo0 = (p.eps0hs ** nhs_val) if p.eps0hs > 0.0 else 0.0
        aexp0 = (bhs_val - ahs_val) * math.exp(-mhs_val * expo0)
        atemp = p.k1 + p.k2 * t_arr
        yld0_val = (bhs_val - aexp0) * atemp + p.hmart * vm_arr

        eff_p = pla + p.eps0hs
        pos_p = eff_p > 0.0
        expo = np.zeros_like(eff_p)
        dexpo = np.zeros_like(eff_p)
        expo[pos_p] = eff_p[pos_p] ** nhs_val
        if nhs_val == 1.0:
            dexpo[pos_p] = 1.0
        elif nhs_val != 0.0:
            dexpo[pos_p] = nhs_val * (eff_p[pos_p] ** (nhs_val - 1.0))
        aexp = (bhs_val - ahs_val) * np.exp(-mhs_val * expo)

        yld_cur = (bhs_val - aexp) * atemp + p.hmart * vm_arr
        dylddp_cur = mhs_val * dexpo * aexp * atemp

        yld = (1.0 - p.fisokin) * yld_cur + p.fisokin * yld0_val
        hk = p.fisokin * dylddp_cur
        dylddp = (1.0 - p.fisokin) * dylddp_cur

    # 4. 3-Direction Orthotropic Tabulated (iflag=3)
    elif p.iflag == 3:
        if sign is not None and len(p.tables) >= 3:
            sxx = sign[:, 0]
            syy = sign[:, 1]
            sxy = sign[:, 2]
            mohr_r = np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
            cos2th = np.where(mohr_r > _EM20, (0.5 * (sxx - syy)) / mohr_r, 1.0)
            cos4th = 2.0 * cos2th ** 2 - 1.0

            y0, dy0 = _eval_curve_1d(p.tables[0], pla)
            y45, dy45 = _eval_curve_1d(p.tables[1], pla)
            y90, dy90 = _eval_curve_1d(p.tables[2], pla)

            q1 = (y0 + 2.0 * y45 + y90) / 4.0
            q2 = (y0 - y90) / 2.0
            q3 = (y0 - 2.0 * y45 + y90) / 4.0

            yld_cur = q1 + q2 * cos2th + q3 * cos4th

            dq1 = (dy0 + 2.0 * dy45 + dy90) / 4.0
            dq2 = (dy0 - dy90) / 2.0
            dq3 = (dy0 - 2.0 * dy45 + dy90) / 4.0
            dylddp_cur = dq1 + dq2 * cos2th + dq3 * cos4th

            y0_0, _ = _eval_curve_1d(p.tables[0], np.zeros_like(pla))
            y45_0, _ = _eval_curve_1d(p.tables[1], np.zeros_like(pla))
            y90_0, _ = _eval_curve_1d(p.tables[2], np.zeros_like(pla))
            q1_0 = (y0_0 + 2.0 * y45_0 + y90_0) / 4.0
            q2_0 = (y0_0 - y90_0) / 2.0
            q3_0 = (y0_0 - 2.0 * y45_0 + y90_0) / 4.0
            yld_0 = q1_0 + q2_0 * cos2th + q3_0 * cos4th

            yld = (1.0 - p.fisokin) * yld_cur + p.fisokin * yld_0
            hk = p.fisokin * dylddp_cur
            dylddp = (1.0 - p.fisokin) * dylddp_cur
        else:
            sigy0 = p.ko if p.ko > 0.0 else (p.aswift if p.aswift > 0.0 else 1.0)
            yld.fill(sigy0)
            dylddp.fill(0.0)
            hk.fill(0.0)

    else:
        sigy0 = p.ko if p.ko > 0.0 else (p.aswift if p.aswift > 0.0 else 1.0)
        yld.fill(sigy0)
        dylddp.fill(0.0)
        hk.fill(0.0)

    return yld, dylddp, hk


# ============================================================================
# Shell Constitutive Update
# ============================================================================

def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Plane-stress constitutive update for /MAT/LAW87 (/MAT/BARLAT2000).

    Follows engine/source/materials/mat/mat087/sigeps87c.F90 and mat87c_swift_voce.F90.

    Parameters
    ----------
    mat : Material, Law87Params, or dict
    sig : (3,) or (n, 3) or (n, 5) ndarray
        Old Cauchy stress [xx, yy, xy, (yz, zx)].
    deps : (3,) or (n, 3) or (n, 5) ndarray
        Engineering strain increment [xx, yy, xy, (yz, zx)].
    epsp : (n,) or float, optional
        Accumulated equivalent plastic strain history.
    dt : float, default 0.0
        Time step increment.
    extra : dict, optional
        State dict containing 'uvar87', 'pla87', 'sigb87', 'off87', 'thk', 'thkly', etc.
    return_tuple : bool, default False
        If True, returns (sig, epsp, soundsp). Otherwise returns (sig, epsp).
    """
    p = _get_params(mat)

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)

    if is_1d:
        sig_arr = sig_arr[None, :]
        deps_arr = deps_arr[None, :]

    n = sig_arr.shape[0]
    ncomp = max(sig_arr.shape[1], deps_arr.shape[1], 3)
    has_shear = (ncomp >= 5)

    if deps_arr.shape[0] == 1 and n > 1:
        deps_arr = np.repeat(deps_arr, n, axis=0)

    if extra is None:
        extra = {}

    # 1. Equivalent plastic strain (pla)
    if "pla87" in extra and extra["pla87"] is not None:
        pla = np.asarray(extra["pla87"], dtype=float).copy().flatten()
    elif "pla" in extra and extra["pla"] is not None:
        pla = np.asarray(extra["pla"], dtype=float).copy().flatten()
    elif "epsp" in extra and extra["epsp"] is not None:
        pla = np.asarray(extra["epsp"], dtype=float).copy().flatten()
    elif epsp is not None:
        pla = np.asarray(epsp, dtype=float).copy().flatten()
    else:
        pla = np.zeros(n, dtype=float)

    if len(pla) == 1 and n > 1:
        pla = np.full(n, pla[0], dtype=float)
    elif len(pla) == 0:
        pla = np.zeros(n, dtype=float)

    # 2. Backstress tensor (sigb) - 12 components per element (up to 4 CR branches x 3)
    if "sigb87" in extra and extra["sigb87"] is not None:
        sigb = np.asarray(extra["sigb87"], dtype=float).copy()
    elif "sigb" in extra and extra["sigb"] is not None:
        sigb = np.asarray(extra["sigb"], dtype=float).copy()
    else:
        sigb = np.zeros((n, 12), dtype=float)

    if sigb.ndim == 1:
        if sigb.shape[0] < 12:
            sigb_padded = np.zeros(12, dtype=float)
            sigb_padded[:sigb.shape[0]] = sigb
            sigb = sigb_padded
        sigb = np.tile(sigb[None, :], (n, 1))
    elif sigb.shape[1] < 12:
        sigb_padded = np.zeros((n, 12), dtype=float)
        sigb_padded[:, :sigb.shape[1]] = sigb
        sigb = sigb_padded

    # 3. Element deletion status (off)
    if "off87" in extra and extra["off87"] is not None:
        off = np.asarray(extra["off87"], dtype=float).copy().flatten()
    elif "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).copy().flatten()
    elif "layfail" in extra and extra["layfail"] is not None:
        off = np.asarray(extra["layfail"], dtype=float).copy().flatten()
    else:
        off = np.ones(n, dtype=float)

    if len(off) == 1 and n > 1:
        off = np.full(n, off[0], dtype=float)

    # 4. User variables (uvar)
    if "uvar87" in extra and extra["uvar87"] is not None:
        uvar = np.asarray(extra["uvar87"], dtype=float).copy()
    elif "uvar" in extra and extra["uvar"] is not None:
        uvar = np.asarray(extra["uvar"], dtype=float).copy()
    else:
        uvar = np.zeros((n, 1), dtype=float)

    if uvar.ndim == 1:
        uvar = uvar[:, None] if len(uvar) == n else np.zeros((n, 1), dtype=float)

    # Elastic stiffness moduli
    a1 = p.a1
    a2 = p.a2
    g = p.g
    gs = g * p.shf
    young = p.e
    nu = p.nu

    # 1. Elastic trial stresses
    sigoxx = sig_arr[:, 0]
    sigoyy = sig_arr[:, 1]
    sigoxy = sig_arr[:, 2]

    signxx = sigoxx + a1 * deps_arr[:, 0] + a2 * deps_arr[:, 1]
    signyy = sigoyy + a2 * deps_arr[:, 0] + a1 * deps_arr[:, 1]
    signxy = sigoxy + g * deps_arr[:, 2]

    if has_shear:
        sigoyz = sig_arr[:, 3]
        sigozx = sig_arr[:, 4]
        signyz = sigoyz + gs * deps_arr[:, 3]
        signzx = sigozx + gs * deps_arr[:, 4]
    else:
        signyz = np.zeros(n, dtype=float)
        signzx = np.zeros(n, dtype=float)

    # 2. Backstress accumulation and shift
    sigbxx = np.zeros(n, dtype=float)
    sigbyy = np.zeros(n, dtype=float)
    sigbxy = np.zeros(n, dtype=float)
    if p.fisokin > 0.0:
        for j in range(4):
            sigbxx += sigb[:, 3 * j + 0]
            sigbyy += sigb[:, 3 * j + 1]
            sigbxy += sigb[:, 3 * j + 2]
        signxx -= sigbxx
        signyy -= sigbyy
        signxy -= sigbxy

    # Strain rate evaluation
    if p.iflagsr == 0:
        if dt > 0.0:
            epspxx = deps_arr[:, 0] / dt
            epspyy = deps_arr[:, 1] / dt
            epspxy = deps_arr[:, 2] / dt
            epsd = 0.5 * (np.abs(epspxx + epspyy)
                          + np.sqrt((epspxx - epspyy) ** 2 + epspxy ** 2))
        else:
            epsd = np.zeros(n, dtype=float)
    else:
        epsd = uvar[:, 0].copy()

    # Initial trial equivalent stress
    normsig = np.sqrt(signxx ** 2 + signyy ** 2 + 2.0 * signxy ** 2)
    normsig = np.maximum(normsig, 1.0)

    xpxx = (p.lp11 * signxx + p.lp12 * signyy) / normsig
    xpyy = (p.lp21 * signxx + p.lp22 * signyy) / normsig
    xpxy = (p.lp66 * signxy) / normsig

    xppxx = (p.lpp11 * signxx + p.lpp12 * signyy) / normsig
    xppyy = (p.lpp21 * signxx + p.lpp22 * signyy) / normsig
    xppxy = (p.lpp66 * signxy) / normsig

    r_p = np.sqrt(0.25 * (xpxx - xpyy) ** 2 + xpxy ** 2)
    c_p = 0.5 * (xpxx + xpyy)
    xp1 = c_p + r_p
    xp2 = c_p - r_p

    r_pp = np.sqrt(0.25 * (xppxx - xppyy) ** 2 + xppxy ** 2)
    c_pp = 0.5 * (xppxx + xppyy)
    xpp1 = c_pp + r_pp
    xpp2 = c_pp - r_pp

    phip = np.abs(xp1 - xp2) ** p.expa
    phipp = (np.abs(2.0 * xpp2 + xpp1) ** p.expa
             + np.abs(2.0 * xpp1 + xpp2) ** p.expa)

    sum_phi = 0.5 * (phip + phipp)
    pos = sum_phi > 0.0
    seq = np.zeros(n, dtype=float)
    seq[pos] = (sum_phi[pos] ** (1.0 / p.expa)) * normsig[pos]

    # Initial yield stress
    temp_arr = extra.get("temp", np.full(n, p.temp0))
    yld, dylddp, hk = _eval_yield_stress(p, pla, epsd, np.column_stack([signxx, signyy, signxy]), temp_arr, uvar)

    phi = (seq / np.maximum(yld, _EM20)) ** 2 - 1.0

    # Yield check
    yielding = np.where((phi >= 0.0) & (off == 1.0))[0]

    dpla = np.zeros(n, dtype=float)
    deplzz = np.zeros(n, dtype=float)
    etse = np.ones(n, dtype=float)

    # 3-iteration Newton-Raphson return mapping
    if len(yielding) > 0:
        dsigbxxdp = np.zeros(n, dtype=float)
        dsigbyydp = np.zeros(n, dtype=float)
        dsigbxydp = np.zeros(n, dtype=float)
        if p.ikin == 1 and p.fisokin > 0.0:
            for j in range(4):
                dsigbxxdp += p.ckh[j] * sigb[:, 3 * j + 0]
                dsigbyydp += p.ckh[j] * sigb[:, 3 * j + 1]
                dsigbxydp += p.ckh[j] * sigb[:, 3 * j + 2]

        niter_val = kwargs.get("niter", None)
        if niter_val is not None:
            niter = int(niter_val)
            tol = float(kwargs.get("tol", 0.0))
        elif kwargs.get("fortran_parity", False):
            niter = 3
            tol = 0.0
        else:
            niter = 25
            tol = 1e-5

        for _iter in range(niter):
            if tol > 0.0:
                active = yielding[np.abs(phi[yielding]) >= tol]
                if _iter > 0 and len(active) == 0:
                    break
            else:
                active = yielding

            for i in active:
                # Derivatives of X' principal values
                mr_p = max(r_p[i], _EM20)
                dxp1dxpxx = 0.5 * (1.0 + (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp1dxpyy = 0.5 * (1.0 - (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp1dxpxy = xpxy[i] / mr_p
                dxp2dxpxx = 0.5 * (1.0 - (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp2dxpyy = 0.5 * (1.0 + (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp2dxpxy = -xpxy[i] / mr_p

                # Derivatives of X'' principal values
                mr_pp = max(r_pp[i], _EM20)
                dxpp1dxppxx = 0.5 * (1.0 + (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp1dxppyy = 0.5 * (1.0 - (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp1dxppxy = xppxy[i] / mr_pp
                dxpp2dxppxx = 0.5 * (1.0 - (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp2dxppyy = 0.5 * (1.0 + (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp2dxppxy = -xppxy[i] / mr_pp

                # Chain to stress components
                dxp1dsigxx = dxp1dxpxx * p.lp11 + dxp1dxpyy * p.lp21
                dxp1dsigyy = dxp1dxpxx * p.lp12 + dxp1dxpyy * p.lp22
                dxp1dsigxy = dxp1dxpxy * p.lp66

                dxp2dsigxx = dxp2dxpxx * p.lp11 + dxp2dxpyy * p.lp21
                dxp2dsigyy = dxp2dxpxx * p.lp12 + dxp2dxpyy * p.lp22
                dxp2dsigxy = dxp2dxpxy * p.lp66

                dxpp1dsigxx = dxpp1dxppxx * p.lpp11 + dxpp1dxppyy * p.lpp21
                dxpp1dsigyy = dxpp1dxppxx * p.lpp12 + dxpp1dxppyy * p.lpp22
                dxpp1dsigxy = dxpp1dxppxy * p.lpp66

                dxpp2dsigxx = dxpp2dxppxx * p.lpp11 + dxpp2dxppyy * p.lpp21
                dxpp2dsigyy = dxpp2dxppxx * p.lpp12 + dxpp2dxppyy * p.lpp22
                dxpp2dsigxy = dxpp2dxppxy * p.lpp66

                diff_p = xp1[i] - xp2[i]
                sgn_p = 1.0 if diff_p >= 0.0 else -1.0
                dphipdxp1 = p.expa * (abs(diff_p) ** (p.expa - 1.0)) * sgn_p
                dphipdxp2 = -dphipdxp1

                term_pp1 = 2.0 * xpp2[i] + xpp1[i]
                sgn_pp1 = 1.0 if term_pp1 >= 0.0 else -1.0
                term_pp2 = 2.0 * xpp1[i] + xpp2[i]
                sgn_pp2 = 1.0 if term_pp2 >= 0.0 else -1.0

                dphippdxpp1 = (p.expa * (abs(term_pp1) ** (p.expa - 1.0)) * sgn_pp1
                               + 2.0 * p.expa * (abs(term_pp2) ** (p.expa - 1.0)) * sgn_pp2)
                dphippdxpp2 = (p.expa * (abs(term_pp2) ** (p.expa - 1.0)) * sgn_pp2
                               + 2.0 * p.expa * (abs(term_pp1) ** (p.expa - 1.0)) * sgn_pp1)

                dphipdsigxx = dphipdxp1 * dxp1dsigxx + dphipdxp2 * dxp2dsigxx
                dphipdsigyy = dphipdxp1 * dxp1dsigyy + dphipdxp2 * dxp2dsigyy
                dphipdsigxy = dphipdxp1 * dxp1dsigxy + dphipdxp2 * dxp2dsigxy

                dphippdsigxx = dphippdxpp1 * dxpp1dsigxx + dphippdxpp2 * dxpp2dsigxx
                dphippdsigyy = dphippdxpp1 * dxpp1dsigyy + dphippdxpp2 * dxpp2dsigyy
                dphippdsigxy = dphippdxpp1 * dxpp1dsigxy + dphippdxpp2 * dxpp2dsigxy

                s_phi = 0.5 * (phip[i] + phipp[i])
                dseqdphi = (0.5 / p.expa) * (s_phi ** (1.0 / p.expa - 1.0)) if s_phi > 0.0 else 0.0

                dseqdsigxx = dseqdphi * (dphipdsigxx + dphippdsigxx)
                dseqdsigyy = dseqdphi * (dphipdsigyy + dphippdsigyy)
                dseqdsigxy = dseqdphi * (dphipdsigxy + dphippdsigxy)

                myld = max(yld[i], _EM20)
                dphidseq = 2.0 * (seq[i] / (myld ** 2))
                normxx = dphidseq * dseqdsigxx
                normyy = dphidseq * dseqdsigyy
                normxy = dphidseq * dseqdsigxy

                dsigxxdlam = -a1 * normxx - a2 * normyy
                dsigyydlam = -a1 * normyy - a2 * normxx
                dsigxydlam = -g * normxy

                dphidsig_dsigdlam = normxx * dsigxxdlam + normyy * dsigyydlam + normxy * dsigxydlam

                dphidyld = -2.0 * (seq[i] ** 2 / (myld ** 3))
                dphidpla = dphidyld * dylddp[i]
                sig_dphidsig = signxx[i] * normxx + signyy[i] * normyy + signxy[i] * normxy
                dpladlam = sig_dphidsig / myld

                # Kinematic hardening contribution
                if p.fisokin > 0.0:
                    if p.ikin == 1:
                        dsigbxxdlam = p.fisokin * (p.akck * (2.0 * normxx + normyy) - dsigbxxdp[i] * dpladlam)
                        dsigbyydlam = p.fisokin * (p.akck * (2.0 * normyy + normxx) - dsigbyydp[i] * dpladlam)
                        dsigbxydlam = p.fisokin * (p.akck * normxy - dsigbxydp[i] * dpladlam)
                    elif p.ikin == 2:
                        dsigbxxdlam = (2.0 / 3.0) * hk[i] * (2.0 * normxx + normyy)
                        dsigbyydlam = (2.0 / 3.0) * hk[i] * (2.0 * normyy + normxx)
                        dsigbxydlam = (2.0 / 3.0) * hk[i] * normxy
                    else:
                        dsigbxxdlam = dsigbyydlam = dsigbxydlam = 0.0
                    dphidsigb_dsigbdlam = -normxx * dsigbxxdlam - normyy * dsigbyydlam - normxy * dsigbxydlam
                else:
                    dsigbxxdlam = dsigbyydlam = dsigbxydlam = 0.0
                    dphidsigb_dsigbdlam = 0.0

                dphidlam = dphidsig_dsigdlam + dphidpla * dpladlam + dphidsigb_dsigbdlam
                if abs(dphidlam) < _EM20:
                    dphidlam = _EM20 if dphidlam >= 0.0 else -_EM20

                dlam = -phi[i] / dphidlam
                ddep = dpladlam * dlam

                dpla[i] = max(0.0, dpla[i] + ddep)
                pla[i] += ddep
                deplzz[i] -= (dlam * normxx + dlam * normyy)

                signxx[i] += dsigxxdlam * dlam
                signyy[i] += dsigyydlam * dlam
                signxy[i] += dsigxydlam * dlam

                if p.fisokin > 0.0:
                    # Update backstresses
                    signxx[i] += sigbxx[i]
                    signyy[i] += sigbyy[i]
                    signxy[i] += sigbxy[i]
                    sigbxx[i] += dsigbxxdlam * dlam
                    sigbyy[i] += dsigbyydlam * dlam
                    sigbxy[i] += dsigbxydlam * dlam
                    signxx[i] -= sigbxx[i]
                    signyy[i] -= sigbyy[i]
                    signxy[i] -= sigbxy[i]

                    if p.ikin == 1:
                        for j in range(4):
                            fac_a = p.akh[j] * p.ckh[j]
                            c_j = p.ckh[j]
                            sigb[i, 3 * j + 0] += p.fisokin * (fac_a * (2.0 * normxx + normyy) * dlam - c_j * sigb[i, 3 * j + 0] * ddep)
                            sigb[i, 3 * j + 1] += p.fisokin * (fac_a * (2.0 * normyy + normxx) * dlam - c_j * sigb[i, 3 * j + 1] * ddep)
                            sigb[i, 3 * j + 2] += p.fisokin * (fac_a * normxy * dlam - c_j * sigb[i, 3 * j + 2] * ddep)
                    elif p.ikin == 2:
                        sigb[i, 0] += dsigbxxdlam * dlam
                        sigb[i, 1] += dsigbyydlam * dlam
                        sigb[i, 2] += dsigbxydlam * dlam

                # Update normalized stresses and equivalent stress
                ns = math.sqrt(signxx[i] ** 2 + signyy[i] ** 2 + 2.0 * signxy[i] ** 2)
                normsig[i] = max(ns, 1.0)
                xpxx[i] = (p.lp11 * signxx[i] + p.lp12 * signyy[i]) / normsig[i]
                xpyy[i] = (p.lp21 * signxx[i] + p.lp22 * signyy[i]) / normsig[i]
                xpxy[i] = (p.lp66 * signxy[i]) / normsig[i]

                xppxx[i] = (p.lpp11 * signxx[i] + p.lpp12 * signyy[i]) / normsig[i]
                xppyy[i] = (p.lpp21 * signxx[i] + p.lpp22 * signyy[i]) / normsig[i]
                xppxy[i] = (p.lpp66 * signxy[i]) / normsig[i]

                r_p[i] = math.sqrt(0.25 * (xpxx[i] - xpyy[i]) ** 2 + xpxy[i] ** 2)
                c_p = 0.5 * (xpxx[i] + xpyy[i])
                xp1[i] = c_p + r_p[i]
                xp2[i] = c_p - r_p[i]

                r_pp[i] = math.sqrt(0.25 * (xppxx[i] - xppyy[i]) ** 2 + xppxy[i] ** 2)
                c_pp = 0.5 * (xppxx[i] + xppyy[i])
                xpp1[i] = c_pp + r_pp[i]
                xpp2[i] = c_pp - r_pp[i]

                phip[i] = abs(xp1[i] - xp2[i]) ** p.expa
                phipp[i] = (abs(2.0 * xpp2[i] + xpp1[i]) ** p.expa
                            + abs(2.0 * xpp1[i] + xpp2[i]) ** p.expa)

                sum_phi_i = 0.5 * (phip[i] + phipp[i])
                if sum_phi_i > 0.0:
                    seq[i] = (sum_phi_i ** (1.0 / p.expa)) * normsig[i]
                else:
                    seq[i] = 0.0

            # Re-evaluate yield stress
            if len(active) > 0:
                y_up, dy_up, hk_up = _eval_yield_stress(p, pla[active], epsd[active],
                                                        np.column_stack([signxx[active], signyy[active], signxy[active]]),
                                                        temp_arr[active], uvar[active])
                yld[active] = y_up
                dylddp[active] = dy_up
                hk[active] = hk_up
                phi[active] = (seq[active] / np.maximum(yld[active], _EM20)) ** 2 - 1.0

        for i in yielding:
            h_tot = dylddp[i] + hk[i]
            etse[i] = h_tot / (h_tot + young)

    # Re-add backstress to total stress tensor
    if p.fisokin > 0.0:
        signxx += sigbxx
        signyy += sigbyy
        signxy += sigbxy

    # Thickness thinning increment
    deelzz = -nu * (signxx - sigoxx + signyy - sigoyy) / young
    depszz = deelzz + deplzz

    thkly = np.ones(n, dtype=float)
    if "thkly" in extra and extra["thkly"] is not None:
        thkly = np.asarray(extra["thkly"], dtype=float).flatten()
    elif "thk0" in extra and extra["thk0"] is not None:
        thkly = np.asarray(extra["thk0"], dtype=float).flatten()
    elif "thk" in extra and extra["thk"] is not None:
        thkly = np.asarray(extra["thk"], dtype=float).flatten()

    thk_val = thkly.copy()
    if "thk87" in extra and extra["thk87"] is not None:
        thk_val = np.asarray(extra["thk87"], dtype=float).flatten().copy()
    elif "thk" in extra and extra["thk"] is not None:
        thk_val = np.asarray(extra["thk"], dtype=float).flatten().copy()

    if len(thkly) == 1 and n > 1:
        thkly = np.full(n, thkly[0], dtype=float)
    if len(thk_val) == 1 and n > 1:
        thk_val = np.full(n, thk_val[0], dtype=float)

    thk_val = thk_val + depszz * thkly * off

    # Sound speed calculation
    rho_cur = np.full(n, p.rho0, dtype=float)
    if "rho" in extra and extra["rho"] is not None:
        r_in = np.asarray(extra["rho"], dtype=float).flatten()
        if len(r_in) == 1 and n > 1:
            rho_cur = np.full(n, r_in[0], dtype=float)
        elif len(r_in) == n:
            rho_cur = r_in
    soundsp = np.sqrt(a1 / np.maximum(rho_cur, _EM20))

    # Assemble stress output
    if has_shear:
        sig_out = np.column_stack([signxx, signyy, signxy, signyz, signzx])
    else:
        sig_out = np.column_stack([signxx, signyy, signxy])

    # Record state back to extra
    if extra is not None:
        for k in ("pla87", "pla", "epsp"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = pla
            elif k in extra:
                extra[k] = pla[0] if is_1d else pla
        if not any(k in extra for k in ("pla87", "pla")):
            extra["pla87"] = pla[0] if is_1d else pla.copy()

        for k in ("sigb87", "sigb"):
            if k in extra and isinstance(extra[k], np.ndarray):
                if extra[k].ndim == 1:
                    extra[k][:min(12, len(extra[k]))] = sigb[0, :min(12, len(extra[k]))]
                else:
                    extra[k][:, :12] = sigb[:, :12]
            elif k in extra:
                extra[k] = sigb[0] if is_1d else sigb
        if not any(k in extra for k in ("sigb87", "sigb")):
            extra["sigb87"] = sigb[0] if is_1d else sigb.copy()

        for k in ("off87", "off", "layfail"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = off
            elif k in extra:
                extra[k] = off[0] if is_1d else off
        if not any(k in extra for k in ("off87", "off", "layfail")):
            extra["off87"] = off[0] if is_1d else off

        for k in ("thk87", "thk"):
            if k in extra and isinstance(extra[k], np.ndarray):
                extra[k].flat = thk_val
            elif k in extra:
                extra[k] = thk_val[0] if is_1d else thk_val
        if not any(k in extra for k in ("thk87", "thk")):
            extra["thk87"] = thk_val[0] if is_1d else thk_val.copy()

        if p.iflagsr == 1:
            dpdt = dpla / max(dt, _EM20)
            uvar[:, 0] = dpdt
        elif dt > 0.0:
            uvar[:, 0] = epsd

        for k_uvar in ("uvar87", "uvar"):
            if k_uvar in extra and isinstance(extra[k_uvar], np.ndarray):
                if extra[k_uvar].ndim == 1:
                    n_copy = min(len(extra[k_uvar]), uvar.shape[1] if uvar.ndim > 1 else len(uvar))
                    if uvar.ndim > 1:
                        extra[k_uvar][:n_copy] = uvar[0, :n_copy]
                    else:
                        extra[k_uvar][:n_copy] = uvar[:n_copy]
                else:
                    ncols = min(extra[k_uvar].shape[1], uvar.shape[1] if uvar.ndim > 1 else 1)
                    if uvar.ndim > 1:
                        extra[k_uvar][:, :ncols] = uvar[:, :ncols]
                    else:
                        extra[k_uvar][:, 0] = uvar.flatten()
            elif k_uvar in extra:
                extra[k_uvar] = uvar[0] if is_1d else uvar
        if not any(k_uvar in extra for k_uvar in ("uvar87", "uvar")):
            extra["uvar87"] = uvar[0] if is_1d else uvar
        extra["depszz"] = depszz[0] if is_1d else depszz
        extra["seq"] = seq[0] if is_1d else seq
        extra["etse"] = etse[0] if is_1d else etse

    if is_1d:
        res_sig = sig_out[0]
        res_pla = float(pla[0])
        res_snd = float(soundsp[0])
    else:
        res_sig = sig_out
        res_pla = pla
        res_snd = soundsp

    if return_tuple:
        return res_sig, res_pla, res_snd
    return res_sig, res_pla


shell_update_law87 = shell_update


# ============================================================================
# Solid Update (Rejection)
# ============================================================================

def solid_update(mat: Any, sig: np.ndarray | None = None, deps: np.ndarray | None = None, epsp: Any = None,
                 dt: float = 0.0, extra: Any = None, **kwargs: Any) -> Any:
    """Solid constitutive update for /MAT/LAW87 (rejected: shells only)."""
    if hasattr(mat, "elements") or hasattr(mat, "nel") or hasattr(mat, "nodes") or kwargs.get("fint") is not None:
        return kwargs.get("fint", None)
    raise NotImplementedError("/MAT/LAW87 is for shell elements only.")


solid_update_law87 = solid_update


# ============================================================================
# Sound Speed
# ============================================================================

def sound_speed_shell(mat: Any, rho: Optional[Union[float, np.ndarray]] = None,
                      extra: Any = None, **kwargs: Any) -> Union[float, np.ndarray]:
    """Acoustic sound speed in thin shells c = sqrt(E / ((1 - nu^2) * rho))."""
    p = _get_params(mat)
    if rho is None:
        rho = p.rho0
    rho_arr = np.asarray(rho, dtype=float)
    c2 = p.a1 / np.maximum(rho_arr, _EM20)
    res = np.sqrt(c2)
    return float(res) if res.ndim == 0 else res


sound_speed = sound_speed_shell
sound_speed_shell_law87 = sound_speed_shell


# ============================================================================
# Consistent Algorithmic Tangent Operator
# ============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if extra is None:
        return None
    res: Dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        else:
            res[k] = v
    return res


def consistent_shell_tangent(
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

    When deps is provided, computes the exact algorithmic consistent tangent
    via directional central finite difference perturbation:
        D_ij = (sigma_i(deps + h*e_j) - sigma_i(deps - h*e_j)) / (2*h)
    Otherwise returns the analytical elastoplastic or elastic plane-stress tangent.
    """
    p = _get_params(mat)

    if "deps" in kwargs and deps is None:
        deps = kwargs.pop("deps")
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

    # Sizing
    n_sig = 1 if (sig is None or np.ndim(sig) <= 1) else np.asarray(sig).shape[0]
    n_deps = 1 if (deps is None or np.ndim(deps) <= 1) else np.asarray(deps).shape[0]
    nel = max(n_sig, n_deps)
    single = (sig is None or np.ndim(sig) <= 1) and (deps is None or np.ndim(deps) <= 1)

    if sig is not None:
        sig_arr = np.asarray(sig, dtype=float).copy()
        if sig_arr.ndim == 1:
            sig_arr = sig_arr.reshape(1, -1)
        if sig_arr.shape[0] == 1 and nel > 1:
            sig_arr = np.repeat(sig_arr, nel, axis=0)
    else:
        sig_arr = np.zeros((nel, 3), dtype=float)

    # If deps is provided, compute numerical consistent tangent
    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float).copy()
        if deps_arr.ndim == 1:
            deps_arr = deps_arr.reshape(1, -1)
        if deps_arr.shape[0] == 1 and nel > 1:
            deps_arr = np.repeat(deps_arr, nel, axis=0)

        epsp_arr = np.asarray(epsp, dtype=float).flatten() if epsp is not None else np.zeros(nel, dtype=float)
        if len(epsp_arr) == 1 and nel > 1:
            epsp_arr = np.full(nel, epsp_arr[0], dtype=float)

        D = np.zeros((nel, 3, 3), dtype=float)
        h_step = float(h)
        for j in range(3):
            ej = np.zeros_like(deps_arr)
            ej[:, j] = h_step

            ext_p = _copy_extra(extra)
            ext_m = _copy_extra(extra)

            res_p, _ = shell_update(p, sig_arr.copy(), deps_arr + ej, epsp=epsp_arr.copy(), dt=dt, extra=ext_p)
            res_m, _ = shell_update(p, sig_arr.copy(), deps_arr - ej, epsp=epsp_arr.copy(), dt=dt, extra=ext_m)

            if res_p.ndim == 1:
                res_p = res_p.reshape(1, -1)
                res_m = res_m.reshape(1, -1)

            D[:, :, j] = (res_p[:, :3] - res_m[:, :3]) / (2.0 * h_step)

        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    # Pure elastic matrix C_el
    C_el = np.array([
        [p.a1, p.a2, 0.0],
        [p.a2, p.a1, 0.0],
        [0.0, 0.0, p.g],
    ], dtype=float)

    # Analytical elastoplastic tangent if on yield surface or active plastic increment
    is_plastic = False
    if epsp_incr is not None and np.any(np.asarray(epsp_incr) > 0.0):
        is_plastic = True
    else:
        seq_val = barlat2000_equivalent_stress(sig_arr, p)
        pla_val = np.asarray(epsp, dtype=float).flatten() if epsp is not None else np.zeros(nel, dtype=float)
        epsd_val = np.zeros(nel, dtype=float)
        yld_val, dylddp_val, hk_val = _eval_yield_stress(p, pla_val, epsd_val, sig_arr)
        if np.any(seq_val >= yld_val * (1.0 - 1e-4)):
            is_plastic = True

    if is_plastic:
        # Compute gradient n = d(seq)/d(sig)
        D = np.zeros((nel, 3, 3), dtype=float)
        pla_arr = np.asarray(epsp, dtype=float).flatten() if epsp is not None else np.zeros(nel, dtype=float)
        epsd_arr = np.zeros(nel, dtype=float)
        yld_arr, dyld_arr, hk_arr = _eval_yield_stress(p, pla_arr, epsd_arr, sig_arr)

        h_step = float(h)
        for i in range(nel):
            s_i = sig_arr[i, :3]
            # Normal via central difference of equivalent stress
            n_vec = np.zeros(3, dtype=float)
            for k in range(3):
                e_k = np.zeros(3, dtype=float)
                e_k[k] = h_step
                s_plus = barlat2000_equivalent_stress(s_i + e_k, p)
                s_minus = barlat2000_equivalent_stress(s_i - e_k, p)
                n_vec[k] = (s_plus - s_minus) / (2.0 * h_step)

            m_vec = C_el @ n_vec
            denom = n_vec @ m_vec + dyld_arr[i] + hk_arr[i]
            if denom > _EM20:
                D[i] = C_el - np.outer(m_vec, m_vec) / denom
            else:
                D[i] = C_el

        if symmetric:
            D = 0.5 * (D + np.swapaxes(D, -1, -2))
        return D[0] if single else D

    C_broad = np.broadcast_to(C_el, (nel, 3, 3)).copy()
    return C_broad[0] if single else C_broad


tangent_law87_shell = consistent_shell_tangent
shell_membrane_tangent = consistent_shell_tangent
shell_tangent = consistent_shell_tangent


def tangent(group: Any = None, x: Any = None, epsp_incr: Any = None, **kwargs: Any) -> Any:
    """Stiffness tangent dispatch for element groups or implicit solver."""
    if group is None:
        return None
    mat = getattr(group, "mat", None) or getattr(group, "material", None)
    if mat is not None:
        return consistent_shell_tangent(mat, epsp_incr=epsp_incr, **kwargs)
    if isinstance(group, (Material, Law87Params, dict)) or hasattr(group, "params"):
        return consistent_shell_tangent(group, epsp_incr=epsp_incr, **kwargs)
    return None


# ============================================================================
# Extra Shapes & Model Resolution
# ============================================================================

def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state shapes required by LAW87."""
    nuvar = 7 if isinstance(mat, MatLaw87) else 1
    if nip is not None and nip > 1:
        return {"uvar87": (nip, nuvar)}
    return {"uvar87": (nuvar,)}


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Resolve /FUNCT and /TABLE references into curves/tables in mat.params."""
    p = _get_params(mat)
    functions = getattr(model, "functions", {})
    tables = getattr(model, "tables", {})

    if p.table_id > 0 and p.yield_table is None:
        if p.table_id in tables:
            p.yield_table = tables[p.table_id]
        elif p.table_id in functions:
            p.yield_table = functions[p.table_id]

    if hasattr(mat, "curves") and mat.curves:
        for c_entry in mat.curves:
            fid = getattr(c_entry, "fct_id", 0)
            if fid in functions:
                p.curves.append(functions[fid])
                p.rates.append(getattr(c_entry, "epsp", 0.0))
                p.yfac.append(getattr(c_entry, "fscale", 1.0))


# ============================================================================
# Registration Helper
# ============================================================================

def _register() -> None:
    """Register LAW87 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (87, "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D",
                  "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT2000", "MAT_BARLAT_2000",
                  "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000", "LAW87_BARLAT2000"):
            MAT_PHYSICS_REGISTRY[k] = build_law87
    except Exception:
        pass


_register()
