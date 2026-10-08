# $OR_SRC/engine/source/materials/mat/mat102/sigeps102.F
# Function: SIGEPS102 (lines 28-163)
# $OR_SRC/starter/source/materials/mat/mat102/hm_read_mat102.F
# Function: HM_READ_MAT102 (lines 38-301)
r"""LAW102 — Extended Drucker-Prager 2nd formulation material model (/MAT/LAW102, /MAT/DPRAG2).

Fortran origins:
- ``starter/source/materials/mat/mat102/hm_read_mat102.F`` (starter card reader, defaults, Mohr-Coulomb to Drucker-Prager conversion)
- ``engine/source/materials/mat/mat102/sigeps102.F`` (3D continuum solid constitutive update)
- ``hm_cfg_files/config/CFG/radioss2020/MAT/mat102_DPRAG2.cfg`` (CFG attributes & card format)

Theory & Constitutive Formulation
----------------------------------
LAW102 models geological materials, concrete, soil, rock, and polymers using a pressure-dependent
parabolic/linear Drucker-Prager yield criterion fitted from Mohr-Coulomb parameters:
- Cohesion \(C\) (Mohr-Coulomb intercept, stress units)
- Friction angle \(\phi\) (Mohr-Coulomb internal friction angle, input in degrees)
- Formulation flag ``IFORM``:
  - ``IFORM = 1``: Circumscribed criterion (matching Mohr-Coulomb at outer corners)
    \(k = \frac{6 C \cos\phi}{\sqrt{3}(3 - \sin\phi)}, \quad \alpha = \frac{2 \sin\phi}{\sqrt{3}(3 - \sin\phi)}\)
  - ``IFORM = 2``: Middle criterion (default, matching intermediate triaxial state)
    \(k = \frac{6 C \cos\phi}{\sqrt{3}(3 + \sin\phi)}, \quad \alpha = \frac{2 \sin\phi}{\sqrt{3}(3 + \sin\phi)}\)
  - ``IFORM = 3``: Inscribed criterion (matching Mohr-Coulomb inner circle)
    \(k = \frac{3 C \cos\phi}{\sqrt{9 + 3\sin^2\phi}}, \quad \alpha = \frac{\sin\phi}{\sqrt{9 + 3\sin^2\phi}}\)
  - ``IFORM = 4``: Original Mohr-Coulomb formulation with Lode angle \(\theta\) and 3rd invariant \(I_3\)
- Parabolic Drucker-Prager yield function coefficients:
  \(A_0 = k^2, \quad A_1 = 6 k \alpha, \quad A_2 = 9 \alpha^2\)
- Tensile pressure root \(P^*\) (closure of the yield envelope along the hydrostatic axis):
  - If \(A_2 = 0\) and \(A_1 \ne 0\): \(P^* = -A_0 / A_1\)
  - Else if \(A_2 \ne 0\): \(\Delta = A_1^2 - 4 A_0 A_2\); if \(\Delta \ge 0\), \(P^* = (-A_1 + \sqrt{\Delta}) / (2 A_2)\), else \(P^* = -A_1 / (2 A_2)\)
  - Else \(P^* = -\infty\)
- Yield envelope evaluation:
  For ``IFORM`` in \(\{1, 2, 3\}\):
    \(G_0(P_{\text{tot}}) = A_0 + A_1 P_{\text{tot}} + A_2 P_{\text{tot}}^2\)
    \(G_0 = \min(A_{\max}, \max(0, G_0))\)
    Tensile and minimum pressure cutoffs: if \(P_{\text{tot}} \le P_{\min}\) or \(P_{\text{tot}} \le P^*\), \(G_0 = 0\).
    \(\text{YIELD2} = J_2 - G_0\)
  For ``IFORM = 4``:
    \(I_3 = \det(\mathbf{T}) = T_{yy} T_{zz} T_{xx} - T_{yy} T_{zx}^2 - T_{zz} T_{xy}^2 - T_{yz}^2 T_{xx} + 2 T_{yz} T_{xy} T_{zx}\)
    \(\cos(3\theta) = \frac{9 I_3}{2 \sqrt{3} J_2^{3/2}}\) clamped to \([0, 1]\)
    \(\theta = \arccos(\cos(3\theta))\)
    \(G_0 = \max(0, -P_{\text{tot}} \sin\phi + \sqrt{J_2} (\cos\theta - \frac{1}{\sqrt{3}} \sin\theta \sin\phi) - C \cos\phi)\)
    If \(P_{\text{tot}} \le P_{\min}\), \(G_0 = 0\).
    \(\text{YIELD2} = J_2 - G_0\)
- Radial return projection:
  If \(\text{YIELD2} \le 0\) and \(G_0 > 0\): \(\text{ratio} = 1\) (elastic)
  Else: \(\text{ratio} = \sqrt{\frac{G_0}{J_2 + 10^{-14}}}\)
- Stress update:
  \(\sigma_{ij}^{\text{new}} = \text{ratio} \cdot T_{ij} \cdot \text{off} - P_{\text{new}} \delta_{ij}\)
- Plastic strain accumulation:
  \(\Delta\varepsilon_p = (1 - \text{ratio}) \sqrt{J_2} / \max(10^{-20}, 3 G)\)
- Acoustic wave speed:
  \(c_{\text{solid}} = \sqrt{(K + \frac{4}{3}G)/\rho_0}\)
- Element compatibility:
  3D continuum solids (Hexa8, Tetra4, Penta6, Pyra5) and SPH.
  Rejects shells (ANCMSG 305) and 1D elements (ANCMSG 306).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

_EM14 = 1.0e-14
_EM20 = 1.0e-20
_DEFAULT_AMAX = 1.0e30
_DEFAULT_PMIN = -1.0e30


@dataclass
class DPrag2Params:
    """Consolidated material parameters for /MAT/LAW102 (/MAT/DPRAG2)."""
    id: int = 1
    title: str = ""
    law: int = 102
    law_name: str = "LAW102"
    rho: float = 0.0
    iform: int = 2
    e: float = 0.0
    nu: float = 0.0
    c: float = 0.0
    phi: float = 0.0          # degrees
    phi_rad: float = 0.0      # radians
    amax: float = _DEFAULT_AMAX
    pmin: float = _DEFAULT_PMIN
    # Computed parameters matching hm_read_mat102.F
    g: float = 0.0
    bulk: float = 0.0
    k_yield: float = 0.0
    alpha: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    pstar: float = -float("inf")
    # Cap plasticity parameters
    r_cap: float = 0.0
    pa: float = 0.0
    k_cap: float = 0.0
    w_cap: float = 0.0
    d_cap: float = 0.0
    pa0: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def E(self) -> float:
        return self.e

    @property
    def nu_val(self) -> float:
        return self.nu

    @property
    def G(self) -> float:
        return self.g

    @property
    def K(self) -> float:
        return self.bulk

    @property
    def k(self) -> float:
        return self.bulk

    @property
    def sound_speed(self) -> float:
        if self.rho > 0.0:
            return math.sqrt(max(0.0, self.bulk + 4.0 / 3.0 * self.g) / self.rho)
        return 0.0


class DPrag2Constants(NamedTuple):
    g: float
    bulk: float
    phi_rad: float
    k_yield: float
    alpha: float
    a0: float
    a1: float
    a2: float
    pstar: float
    iform: int


def compute_dprag2_constants(
    e: float = 0.0,
    nu: float = 0.0,
    c: float = 0.0,
    phi_deg: float = 0.0,
    iform: int = 2,
    amax: float = _DEFAULT_AMAX,
    pmin: float = _DEFAULT_PMIN,
    a0: float = 0.0,
    a1: float = 0.0,
    a2: float = 0.0,
    phi: Optional[float] = None,
) -> DPrag2Constants:
    """Compute Drucker-Prager yield constants and limits matching hm_read_mat102.F.

    Parameters
    ----------
    e : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    c : float
        Mohr-Coulomb cohesion (stress units).
    phi_deg : float
        Internal friction angle in degrees.
    iform : int
        Formulation flag (1: Circumscribed, 2: Middle, 3: Inscribed, 4: Original Mohr-Coulomb).
    amax : float
        Yield criteria limit on J2.
    pmin : float
        Minimum pressure / tensile cutoff.
    a0, a1, a2 : float
        Direct parabolic Drucker-Prager coefficients (used if C=0 and PHI=0).
    phi : float, optional
        Alias for phi_deg.

    Returns
    -------
    DPrag2Constants
        NamedTuple (g, bulk, phi_rad, k_yield, alpha, a0, a1, a2, pstar, iform_sanitized)
    """
    if phi is not None and phi_deg == 0.0:
        phi_deg = phi

    phi_rad = phi_deg * math.pi / 180.0
    if iform <= 0 or iform > 4:
        iform = 2

    # Elastic moduli
    g = e / (2.0 * (1.0 + nu)) if (1.0 + nu) != 0.0 else 0.0
    bulk = e / (3.0 * (1.0 - 2.0 * nu)) if (1.0 - 2.0 * nu) != 0.0 else 0.0

    sin_phi = math.sin(phi_rad)
    cos_phi = math.cos(phi_rad)
    sqrt3 = math.sqrt(3.0)

    k_yield = 0.0
    alpha = 0.0

    if c != 0.0 or phi_deg != 0.0 or (a0 == 0.0 and a1 == 0.0 and a2 == 0.0):
        if iform == 1:
            # Circumscribed criterion (hm_read_mat102.F:144-145)
            denom = sqrt3 * (3.0 - sin_phi)
            if abs(denom) > 1.0e-30:
                k_yield = 6.0 * c * cos_phi / denom
                alpha = 2.0 * sin_phi / denom
        elif iform == 2:
            # Middle criterion (hm_read_mat102.F:147-148)
            denom = sqrt3 * (3.0 + sin_phi)
            if abs(denom) > 1.0e-30:
                k_yield = 6.0 * c * cos_phi / denom
                alpha = 2.0 * sin_phi / denom
        elif iform == 3:
            # Inscribed criterion (hm_read_mat102.F:150-151)
            denom = math.sqrt(9.0 + 3.0 * sin_phi * sin_phi)
            if abs(denom) > 1.0e-30:
                k_yield = 3.0 * c * cos_phi / denom
                alpha = sin_phi / denom
        elif iform == 4:
            # Original Mohr-Coulomb (sigeps102.F:111)
            k_yield = 1.0 / sqrt3
            alpha = 0.0

        a0 = k_yield * k_yield
        a1 = 6.0 * k_yield * alpha
        a2 = 9.0 * alpha * alpha

    # Pressure root PSTAR (hm_read_mat102.F:167-185)
    pstar = -float("inf")
    if a2 == 0.0 and a1 != 0.0:
        pstar = -a0 / a1
    elif a2 != 0.0:
        delta = a1 * a1 - 4.0 * a0 * a2
        if delta >= 0.0:
            delta_sqrt = math.sqrt(delta)
            pstar = (-a1 + delta_sqrt) / (2.0 * a2)
        else:
            pstar = -a1 / (2.0 * a2)
    else:
        pstar = -float("inf")

    # Limit sanitization (hm_read_mat102.F:187-188)
    if amax == 0.0:
        amax = _DEFAULT_AMAX
    if pmin == 0.0:
        pmin = _DEFAULT_PMIN

    return DPrag2Constants(g, bulk, phi_rad, k_yield, alpha, a0, a1, a2, pstar, iform)


def build_law102(mat: Any = None, **kwargs) -> DPrag2Params:
    """Extract and validate parameters for /MAT/LAW102 (/MAT/DPRAG2) from a model entity or kwargs."""
    if isinstance(mat, DPrag2Params):
        return mat

    if mat is None:
        mat_dict = kwargs
    elif isinstance(mat, dict):
        mat_dict = {**mat, **kwargs}
    else:
        mat_dict = {}

    mat_id = getattr(mat, "id", mat_dict.get("id", kwargs.get("id", 1)))
    title = getattr(mat, "title", mat_dict.get("title", kwargs.get("title", "")))

    rho = getattr(mat, "rho", mat_dict.get("rho", kwargs.get("rho", 0.0)))
    if rho == 0.0:
        rho = getattr(mat, "rho0", mat_dict.get("rho0", kwargs.get("rho0", mat_dict.get("rho_i", kwargs.get("rho_i", 0.0)))))

    iform = getattr(mat, "iform", mat_dict.get("iform", kwargs.get("iform", 2)))
    e = getattr(mat, "e", mat_dict.get("e", kwargs.get("e", 0.0)))
    if e == 0.0:
        e = getattr(mat, "E", mat_dict.get("E", kwargs.get("E", 0.0)))
    nu = getattr(mat, "nu", mat_dict.get("nu", kwargs.get("nu", 0.0)))
    c = getattr(mat, "c", mat_dict.get("c", kwargs.get("c", 0.0)))
    phi = getattr(mat, "phi", mat_dict.get("phi", kwargs.get("phi", mat_dict.get("phi_deg", kwargs.get("phi_deg", 0.0)))))
    amax = getattr(mat, "amax", mat_dict.get("amax", kwargs.get("amax", mat_dict.get("a_max", kwargs.get("a_max", _DEFAULT_AMAX)))))
    pmin = getattr(mat, "pmin", mat_dict.get("pmin", kwargs.get("pmin", mat_dict.get("p_min", kwargs.get("p_min", _DEFAULT_PMIN)))))
    a0 = getattr(mat, "a0", mat_dict.get("a0", kwargs.get("a0", 0.0)))
    a1 = getattr(mat, "a1", mat_dict.get("a1", kwargs.get("a1", 0.0)))
    a2 = getattr(mat, "a2", mat_dict.get("a2", kwargs.get("a2", 0.0)))

    # Cap plasticity parameters
    r_cap = getattr(mat, "r_cap", mat_dict.get("r_cap", kwargs.get("r_cap", mat_dict.get("R_cap", kwargs.get("R_cap", mat_dict.get("r", kwargs.get("r", mat_dict.get("R", kwargs.get("R", 0.0)))))))))
    pa = getattr(mat, "pa", mat_dict.get("pa", kwargs.get("pa", mat_dict.get("Pa", kwargs.get("Pa", mat_dict.get("p_a", kwargs.get("p_a", mat_dict.get("P_a", kwargs.get("P_a", 0.0)))))))))
    pa0 = getattr(mat, "pa0", mat_dict.get("pa0", kwargs.get("pa0", mat_dict.get("Pa0", kwargs.get("Pa0", 0.0)))))
    k_cap = getattr(mat, "k_cap", mat_dict.get("k_cap", kwargs.get("k_cap", mat_dict.get("K_cap", kwargs.get("K_cap", mat_dict.get("kc", kwargs.get("kc", mat_dict.get("k_c", kwargs.get("k_c", 0.0)))))))))
    w_cap = getattr(mat, "w_cap", mat_dict.get("w_cap", kwargs.get("w_cap", mat_dict.get("W_cap", kwargs.get("W_cap", mat_dict.get("w", kwargs.get("w", mat_dict.get("W", kwargs.get("W", 0.0)))))))))
    d_cap = getattr(mat, "d_cap", mat_dict.get("d_cap", kwargs.get("d_cap", mat_dict.get("D_cap", kwargs.get("D_cap", mat_dict.get("d", kwargs.get("d", mat_dict.get("D", kwargs.get("D", 0.0)))))))))

    # Check params dictionary fallback
    params = getattr(mat, "params", {})
    if isinstance(params, dict):
        if rho == 0.0:
            rho = params.get("rho", params.get("rho_i", params.get("rho0", 0.0)))
        if e == 0.0:
            e = params.get("e", params.get("E", 0.0))
        if nu == 0.0:
            nu = params.get("nu", 0.0)
        if c == 0.0:
            c = params.get("c", 0.0)
        if phi == 0.0:
            phi = params.get("phi", params.get("phi_deg", 0.0))
        if amax == _DEFAULT_AMAX:
            amax = params.get("amax", params.get("a_max", _DEFAULT_AMAX))
        if pmin == _DEFAULT_PMIN:
            pmin = params.get("pmin", params.get("p_min", _DEFAULT_PMIN))
        if a0 == 0.0:
            a0 = params.get("a0", 0.0)
        if a1 == 0.0:
            a1 = params.get("a1", 0.0)
        if a2 == 0.0:
            a2 = params.get("a2", 0.0)
        if iform == 2 and "iform" in params:
            iform = params.get("iform", 2)
        if r_cap == 0.0:
            r_cap = params.get("r_cap", params.get("R_cap", params.get("r", params.get("R", 0.0))))
        if pa == 0.0:
            pa = params.get("pa", params.get("Pa", params.get("p_a", params.get("P_a", 0.0))))
        if pa0 == 0.0:
            pa0 = params.get("pa0", params.get("Pa0", 0.0))
        if k_cap == 0.0:
            k_cap = params.get("k_cap", params.get("K_cap", params.get("kc", params.get("k_c", 0.0))))
        if w_cap == 0.0:
            w_cap = params.get("w_cap", params.get("W_cap", params.get("w", params.get("W", 0.0))))
        if d_cap == 0.0:
            d_cap = params.get("d_cap", params.get("D_cap", params.get("d", params.get("D", 0.0))))

    r_cap = float(r_cap)
    pa = float(pa)
    pa0 = float(pa0)
    k_cap = float(k_cap)
    w_cap = float(w_cap)
    d_cap = float(d_cap)

    if pa0 == 0.0 and pa > 0.0:
        pa0 = pa
    if pa == 0.0 and pa0 > 0.0:
        pa = pa0

    g, bulk, phi_rad, k_yield, alpha, a0, a1, a2, pstar, iform_sanitized = compute_dprag2_constants(
        e=e,
        nu=nu,
        c=c,
        phi_deg=phi,
        iform=iform,
        amax=amax,
        pmin=pmin,
        a0=a0,
        a1=a1,
        a2=a2,
    )

    return DPrag2Params(
        id=mat_id,
        title=title,
        law=102,
        law_name="LAW102",
        rho=rho,
        iform=iform_sanitized,
        e=e,
        nu=nu,
        c=c,
        phi=phi,
        phi_rad=phi_rad,
        amax=amax,
        pmin=pmin,
        g=g,
        bulk=bulk,
        k_yield=k_yield,
        alpha=alpha,
        a0=a0,
        a1=a1,
        a2=a2,
        pstar=pstar,
        r_cap=r_cap,
        pa=pa,
        k_cap=k_cap,
        w_cap=w_cap,
        d_cap=d_cap,
        pa0=pa0,
        extra=params if isinstance(params, dict) else {},
    )


def init_history(n: int = 1, params: Optional[DPrag2Params] = None) -> np.ndarray:
    """Initialize history variables for LAW102 solid elements.

    State variables per integration point:
    - Col 0: Cumulative equivalent plastic strain (PLA)
    - Col 1: Previous cycle hydrostatic pressure (POLD)
    - Col 2: Cap position parameter (Pa) (when cap is active)
    - Col 3: Cumulative plastic volumetric strain (epspv) (when cap is active)
    """
    if params is not None and params.r_cap > 0.0 and params.pa > 0.0:
        hist = np.zeros((n, 4), dtype=np.float64)
        hist[:, 2] = params.pa0 if params.pa0 > 0.0 else params.pa
        return hist
    return np.zeros((n, 2), dtype=np.float64)


def _solid_update_single_core(
    params: DPrag2Params,
    deps: np.ndarray,
    sig_old: np.ndarray,
    history: np.ndarray,
    rho: Optional[float] = None,
    rho0: Optional[float] = None,
    off: float = 1.0,
    pnew: Optional[float] = None,
    psh: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Single-element 3D solid continuum constitutive update matching sigeps102.F.

    Parameters
    ----------
    params : DPrag2Params
        Material parameters.
    deps : np.ndarray
        Strain increments [deps_xx, deps_yy, deps_zz, deps_xy, deps_yz, deps_zx].
    sig_old : np.ndarray
        Previous Cauchy stress tensor [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_zx].
    history : np.ndarray
        History array of shape (2,) [pla, pold].
    rho : float, optional
        Current element density (unused if None, defaults to rho0).
    rho0 : float, optional
        Initial reference density (defaults to params.rho).
    off : float
        Element activation switch (1.0 active, 0.0 inactive).
    pnew : float, optional
        Current hydrostatic pressure from EOS. If None, calculated via linear bulk elasticity.
    psh : float
        Artificial shock viscosity pressure (default 0.0).

    Returns
    -------
    sig_new : np.ndarray
        Updated Cauchy stress tensor [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_zx].
    history_new : np.ndarray
        Updated history array [pla, pold].
    ssp : float
        Instantaneous acoustic dilatational sound speed.
    """
    g = params.g
    gg = 2.0 * g
    bulk = params.bulk
    rho_ref = rho0 if (rho0 is not None and rho0 > 0.0) else params.rho
    if rho_ref <= 0.0:
        rho_ref = 1.0

    # Previous pressure and mean strain increment (sigeps102.F:80-81)
    pold = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
    scrt = (deps[0] + deps[1] + deps[2]) / 3.0

    # Deviatoric trial stress tensor (sigeps102.F:89-94)
    t1 = sig_old[0] + pold + gg * (deps[0] - scrt)
    t2 = sig_old[1] + pold + gg * (deps[1] - scrt)
    t3 = sig_old[2] + pold + gg * (deps[2] - scrt)
    t4 = sig_old[3] + g * deps[3]
    t5 = sig_old[4] + g * deps[4]
    t6 = sig_old[5] + g * deps[5]

    # Pressure treatment: if pnew not supplied, linear bulk elasticity pnew = pold - 3 * bulk * scrt
    if pnew is None:
        pnew = pold - 3.0 * bulk * scrt

    # Sound speed (sigeps102.F:100-101)
    dpdm = bulk + (4.0 / 3.0) * g
    ssp = math.sqrt(max(0.0, dpdm) / rho_ref)

    # Second deviatoric stress invariant J2 (sigeps102.F:107)
    aj2 = 0.5 * (t1 * t1 + t2 * t2 + t3 * t3) + t4 * t4 + t5 * t5 + t6 * t6

    ptot = pnew + psh

    use_cap = (params.r_cap > 0.0 and params.pa > 0.0)

    if use_cap:
        pa_init = params.pa0 if params.pa0 > 0.0 else params.pa
        if len(history) >= 4:
            pa_cur = history[2]
            epspv_cur = history[3]
        else:
            pa_cur = pa_init
            epspv_cur = 0.0
        if pa_cur <= 0.0:
            pa_cur = pa_init

        k_cap = params.k_cap
        if k_cap <= 0.0:
            if params.phi_rad > 0.0 or params.c > 0.0:
                k_cap = math.tan(params.phi_rad) + params.c / max(1.0e-12, pa_cur)
            elif params.k_yield > 0.0:
                k_cap = (math.sqrt(3.0) * params.k_yield + 3.0 * math.sqrt(3.0) * params.alpha * pa_cur) / max(1.0e-12, pa_cur)
            else:
                k_cap = 1.0

        r_cap = params.r_cap
        b = k_cap * pa_cur
        q_tr = math.sqrt(max(0.0, 3.0 * aj2))

        if ptot <= pa_cur:
            # Shear yield surface: Fs = q - p*tan(phi) - c
            if params.phi_rad > 0.0 or params.c > 0.0:
                q_yield = ptot * math.tan(params.phi_rad) + params.c
            else:
                g0 = params.a0 + params.a1 * ptot + params.a2 * ptot * ptot
                q_yield = math.sqrt(max(0.0, 3.0 * g0))

            if ptot <= params.pmin or ptot <= params.pstar:
                q_yield = 0.0
            q_yield = max(0.0, q_yield)
            if params.amax < _DEFAULT_AMAX:
                q_yield = min(q_yield, math.sqrt(max(0.0, 3.0 * params.amax)))

            fs = q_tr - q_yield
            if fs <= 0.0 and q_yield > 0.0:
                ratio = 1.0
            elif q_yield <= 0.0:
                ratio = 0.0
            else:
                ratio = q_yield / (q_tr + _EM14)

            sign_xx = ratio * t1 * off - pnew
            sign_yy = ratio * t2 * off - pnew
            sign_zz = ratio * t3 * off - pnew
            sign_xy = ratio * t4 * off
            sign_yz = ratio * t5 * off
            sign_zx = ratio * t6 * off

            dpla = (1.0 - ratio) * math.sqrt(max(0.0, aj2)) / max(_EM20, 3.0 * g)
            pla_new = history[0] + dpla
            pa_new = pa_cur
            epspv_new = epspv_cur
            history_new = np.array([pla_new, pnew, pa_new, epspv_new], dtype=np.float64)
            sig_new = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=np.float64)
            return sig_new, history_new, ssp
        else:
            # Cap yield surface: Fc = (p - Pa)^2 / R^2 + (q / (k*Pa))^2 - 1
            dp_0 = ptot - pa_cur
            fc = (dp_0 / r_cap) ** 2 + (q_tr / b) ** 2 - 1.0
            if fc <= 0.0:
                # Inside cap (elastic)
                ratio = 1.0
                sign_xx = t1 * off - pnew
                sign_yy = t2 * off - pnew
                sign_zz = t3 * off - pnew
                sign_xy = t4 * off
                sign_yz = t5 * off
                sign_zx = t6 * off
                pla_new = history[0]
                pa_new = pa_cur
                epspv_new = epspv_cur
                history_new = np.array([pla_new, pnew, pa_new, epspv_new], dtype=np.float64)
                sig_new = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=np.float64)
                return sig_new, history_new, ssp
            else:
                # Plastic return projection onto elliptic cap
                c_p = bulk / (r_cap * r_cap)
                c_q = (3.0 * g) / (b * b)
                x = 0.0
                for _ in range(30):
                    denom_p = 1.0 + c_p * x
                    denom_q = 1.0 + c_q * x
                    term_p = (dp_0 / denom_p) ** 2 / (r_cap * r_cap)
                    term_q = (q_tr / denom_q) ** 2 / (b * b)
                    val = term_p + term_q - 1.0
                    if abs(val) < 1.0e-12:
                        break
                    dval = -2.0 * (c_p * term_p / denom_p + c_q * term_q / denom_q)
                    if abs(dval) < 1.0e-20:
                        break
                    dx = -val / dval
                    x = max(0.0, x + dx)

                dp_fin = dp_0 / (1.0 + c_p * x)
                ptot_fin = pa_cur + dp_fin
                p_final = ptot_fin - psh
                q_final = q_tr / (1.0 + c_q * x)

                ratio = q_final / (q_tr + _EM14) if q_tr > 0.0 else 0.0
                sign_xx = ratio * t1 * off - p_final
                sign_yy = ratio * t2 * off - p_final
                sign_zz = ratio * t3 * off - p_final
                sign_xy = ratio * t4 * off
                sign_yz = ratio * t5 * off
                sign_zx = ratio * t6 * off

                dpla = (1.0 - ratio) * math.sqrt(max(0.0, aj2)) / max(_EM20, 3.0 * g)
                pla_new = history[0] + dpla
                depspv = max(0.0, (pnew - p_final) / max(_EM20, bulk))
                epspv_new = epspv_cur + depspv

                if params.w_cap > 0.0:
                    if params.d_cap > 0.0:
                        pa_new = pa_init * (1.0 + params.w_cap * (epspv_new ** params.d_cap))
                    else:
                        pa_new = pa_init * math.exp(params.w_cap * epspv_new)
                else:
                    pa_new = pa_cur

                history_new = np.array([pla_new, p_final, pa_new, epspv_new], dtype=np.float64)
                sig_new = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=np.float64)
                return sig_new, history_new, ssp

    # Standard non-cap formulation (sigeps102.F)
    iform = params.iform

    if iform == 4:
        # Original Mohr-Coulomb formulation (sigeps102.F:110-122)
        k_mc = 1.0 / math.sqrt(3.0)
        i3 = (
            t2 * t3 * t1
            - t2 * t6 * t6
            - t3 * t4 * t4
            - t5 * t5 * t1
            + 2.0 * t5 * t4 * t6
        )
        sqrt_j2 = math.sqrt(max(0.0, aj2))
        denom_j2 = 2.0 * math.sqrt(3.0) * (sqrt_j2 * sqrt_j2 * sqrt_j2)
        if denom_j2 > 1.0e-30:
            cos3t = 9.0 * i3 / denom_j2
        else:
            cos3t = 0.0
        cos3t_clamped = max(0.0, min(1.0, cos3t))
        theta = math.acos(cos3t_clamped)

        sin_phi = math.sin(params.phi_rad)
        cos_phi = math.cos(params.phi_rad)
        g0 = -ptot * sin_phi + sqrt_j2 * (math.cos(theta) - k_mc * math.sin(theta) * sin_phi) - params.c * cos_phi
        g0 = max(0.0, g0)
        if ptot <= params.pmin:
            g0 = 0.0
        yield2 = aj2 - g0
    else:
        # Drucker-Prager yield function (sigeps102.F:126-133)
        g0 = params.a0 + params.a1 * ptot + params.a2 * ptot * ptot
        g0 = min(params.amax, g0)
        g0 = max(0.0, g0)
        if ptot <= params.pmin:
            g0 = 0.0
        if ptot <= params.pstar:
            g0 = 0.0
        yield2 = aj2 - g0

    # Projection factor on yield surface (sigeps102.F:140-146)
    if yield2 <= 0.0 and g0 > 0.0:
        ratio = 1.0
    else:
        ratio = math.sqrt(max(0.0, g0) / (aj2 + _EM14))

    # Stress tensor update (sigeps102.F:151-156)
    sign_xx = ratio * t1 * off - pnew
    sign_yy = ratio * t2 * off - pnew
    sign_zz = ratio * t3 * off - pnew
    sign_xy = ratio * t4 * off
    sign_yz = ratio * t5 * off
    sign_zx = ratio * t6 * off

    # Plastic strain increment (sigeps102.F:157-159)
    dpla = (1.0 - ratio) * math.sqrt(max(0.0, aj2)) / max(_EM20, 3.0 * g)
    pla_new = history[0] + dpla

    sig_new = np.array([sign_xx, sign_yy, sign_zz, sign_xy, sign_yz, sign_zx], dtype=np.float64)
    if len(history) >= 4:
        history_new = np.array([pla_new, pnew, history[2], history[3]], dtype=np.float64)
    else:
        history_new = np.array([pla_new, pnew], dtype=np.float64)

    return sig_new, history_new, ssp


def solid_update_single(
    params: DPrag2Params,
    sig_or_deps: np.ndarray,
    deps_or_sig: np.ndarray,
    history: Optional[np.ndarray] = None,
    rho: Optional[float] = None,
    rho0: Optional[float] = None,
    off: float = 1.0,
    pnew: Optional[float] = None,
    psh: float = 0.0,
    *,
    epsp: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Single-element 3D solid continuum constitutive update matching sigeps102.F.

    Supports both solver convention:
      solid_update_single(params, sig_old, deps, epsp=epsp) -> (sig_new, epsp_new)
    and low-level Fortran array convention:
      solid_update_single(params, deps, sig_old, history) -> (sig_new, history_new, ssp)
    """
    use_cap = (params.r_cap > 0.0 and params.pa > 0.0)

    if epsp is not None:
        sig_old = np.asarray(sig_or_deps, dtype=np.float64)
        deps = np.asarray(deps_or_sig, dtype=np.float64)
        pold = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
        if use_cap:
            pa_in = float(kwargs.get("pa", params.pa0 if params.pa0 > 0.0 else params.pa))
            epspv_in = float(kwargs.get("epspv", 0.0))
            hist = np.array([float(epsp), pold, pa_in, epspv_in], dtype=np.float64)
        else:
            hist = np.array([float(epsp), pold], dtype=np.float64)
        sig_new, hist_new, ssp = _solid_update_single_core(
            params, deps, sig_old, hist, rho=rho, rho0=rho0, off=off, pnew=pnew, psh=psh
        )
        if kwargs.get("return_history", False) or kwargs.get("return_all", False):
            return sig_new, hist_new
        return sig_new, float(hist_new[0])

    if history is not None:
        deps = np.asarray(sig_or_deps, dtype=np.float64)
        sig_old = np.asarray(deps_or_sig, dtype=np.float64)
        hist = np.asarray(history, dtype=np.float64)
        if use_cap and len(hist) < 4:
            pa_in = params.pa0 if params.pa0 > 0.0 else params.pa
            hist = np.array([hist[0], hist[1], pa_in, 0.0], dtype=np.float64)
        return _solid_update_single_core(
            params, deps, sig_old, hist, rho=rho, rho0=rho0, off=off, pnew=pnew, psh=psh
        )

    sig_old = np.asarray(sig_or_deps, dtype=np.float64)
    deps = np.asarray(deps_or_sig, dtype=np.float64)
    pold = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
    if use_cap:
        pa_in = params.pa0 if params.pa0 > 0.0 else params.pa
        hist = np.array([0.0, pold, pa_in, 0.0], dtype=np.float64)
    else:
        hist = np.array([0.0, pold], dtype=np.float64)
    return _solid_update_single_core(
        params, deps, sig_old, hist, rho=rho, rho0=rho0, off=off, pnew=pnew, psh=psh
    )


def solid_update_array(
    params: DPrag2Params,
    deps: np.ndarray,
    sig_old: np.ndarray,
    history: np.ndarray,
    rho: Optional[np.ndarray] = None,
    rho0: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
    pnew: Optional[np.ndarray] = None,
    psh: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 3D solid continuum constitutive update matching sigeps102.F.

    Parameters
    ----------
    params : DPrag2Params
        Material parameters.
    deps : np.ndarray
        Array of shape (nel, 6).
    sig_old : np.ndarray
        Array of shape (nel, 6).
    history : np.ndarray
        Array of shape (nel, 2).
    rho : np.ndarray, optional
        Array of shape (nel,).
    rho0 : np.ndarray, optional
        Array of shape (nel,).
    off : np.ndarray, optional
        Array of shape (nel,).
    pnew : np.ndarray, optional
        Array of shape (nel,).
    psh : float or np.ndarray, optional
        Array of shape (nel,).

    Returns
    -------
    sig_new : np.ndarray
        Array of shape (nel, 6).
    history_new : np.ndarray
        Array of shape (nel, 2).
    ssp : np.ndarray
        Array of shape (nel,).
    """
    deps = np.asarray(deps, dtype=np.float64)
    sig_old = np.asarray(sig_old, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    nel = deps.shape[0]

    if off is None:
        off_arr = np.ones(nel, dtype=np.float64)
    else:
        off_arr = np.asarray(off, dtype=np.float64)

    if psh is None:
        psh_arr = np.zeros(nel, dtype=np.float64)
    elif isinstance(psh, (int, float)):
        psh_arr = np.full(nel, float(psh), dtype=np.float64)
    else:
        psh_arr = np.asarray(psh, dtype=np.float64)

    g = params.g
    gg = 2.0 * g
    bulk = params.bulk

    rho_ref = params.rho if params.rho > 0.0 else 1.0
    if rho0 is not None:
        rho0_arr = np.where(rho0 > 0.0, rho0, rho_ref)
    else:
        rho0_arr = np.full(nel, rho_ref, dtype=np.float64)

    # Old pressure & mean strain (sigeps102.F:80-81)
    pold = -(sig_old[:, 0] + sig_old[:, 1] + sig_old[:, 2]) / 3.0
    scrt = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0

    # Deviatoric trial stresses (sigeps102.F:89-94)
    t1 = sig_old[:, 0] + pold + gg * (deps[:, 0] - scrt)
    t2 = sig_old[:, 1] + pold + gg * (deps[:, 1] - scrt)
    t3 = sig_old[:, 2] + pold + gg * (deps[:, 2] - scrt)
    t4 = sig_old[:, 3] + g * deps[:, 3]
    t5 = sig_old[:, 4] + g * deps[:, 4]
    t6 = sig_old[:, 5] + g * deps[:, 5]

    # Linear bulk pressure if pnew not provided
    if pnew is None:
        pnew_arr = pold - 3.0 * bulk * scrt
    else:
        pnew_arr = np.asarray(pnew, dtype=np.float64)

    # Sound speed (sigeps102.F:100-101)
    dpdm = bulk + (4.0 / 3.0) * g
    ssp = np.sqrt(np.maximum(0.0, dpdm) / rho0_arr)

    if params.r_cap > 0.0 and params.pa > 0.0:
        sig_new = np.empty((nel, 6), dtype=np.float64)
        history_new = np.empty((nel, 4), dtype=np.float64)
        pa_init = params.pa0 if params.pa0 > 0.0 else params.pa
        for i in range(nel):
            h_in = history[i]
            if len(h_in) < 4:
                h_in = np.array([h_in[0], h_in[1], pa_init, 0.0], dtype=np.float64)
            s_i, h_i, c_i = _solid_update_single_core(
                params,
                deps[i],
                sig_old[i],
                h_in,
                rho=rho[i] if rho is not None else None,
                rho0=rho0_arr[i],
                off=off_arr[i],
                pnew=pnew_arr[i] if pnew is not None else None,
                psh=psh_arr[i],
            )
            sig_new[i] = s_i
            history_new[i] = h_i
            ssp[i] = c_i
        return sig_new, history_new, ssp

    # Second invariant J2 (sigeps102.F:107)
    aj2 = 0.5 * (t1 * t1 + t2 * t2 + t3 * t3) + t4 * t4 + t5 * t5 + t6 * t6

    ptot = pnew_arr + psh_arr
    iform = params.iform

    if iform == 4:
        # Original Mohr-Coulomb (sigeps102.F:110-122)
        k_mc = 1.0 / math.sqrt(3.0)
        i3 = (
            t2 * t3 * t1
            - t2 * t6 * t6
            - t3 * t4 * t4
            - t5 * t5 * t1
            + 2.0 * t5 * t4 * t6
        )
        sqrt_j2 = np.sqrt(np.maximum(0.0, aj2))
        denom_j2 = 2.0 * math.sqrt(3.0) * (sqrt_j2 * sqrt_j2 * sqrt_j2)
        cos3t = np.where(denom_j2 > 1.0e-30, 9.0 * i3 / np.maximum(denom_j2, 1.0e-30), 0.0)
        cos3t_clamped = np.clip(cos3t, 0.0, 1.0)
        theta = np.arccos(cos3t_clamped)

        sin_phi = math.sin(params.phi_rad)
        cos_phi = math.cos(params.phi_rad)
        g0 = -ptot * sin_phi + sqrt_j2 * (np.cos(theta) - k_mc * np.sin(theta) * sin_phi) - params.c * cos_phi
        g0 = np.maximum(0.0, g0)
        g0 = np.where(ptot <= params.pmin, 0.0, g0)
        yield2 = aj2 - g0
    else:
        # Drucker-Prager (sigeps102.F:126-133)
        g0 = params.a0 + params.a1 * ptot + params.a2 * ptot * ptot
        g0 = np.minimum(params.amax, g0)
        g0 = np.maximum(0.0, g0)
        g0 = np.where(ptot <= params.pmin, 0.0, g0)
        g0 = np.where(ptot <= params.pstar, 0.0, g0)
        yield2 = aj2 - g0

    # Projection factor (sigeps102.F:140-146)
    elastic_mask = (yield2 <= 0.0) & (g0 > 0.0)
    ratio = np.where(elastic_mask, 1.0, np.sqrt(np.maximum(0.0, g0) / (aj2 + _EM14)))

    # Stress tensor update (sigeps102.F:151-156)
    sig_new = np.empty((nel, 6), dtype=np.float64)
    sig_new[:, 0] = ratio * t1 * off_arr - pnew_arr
    sig_new[:, 1] = ratio * t2 * off_arr - pnew_arr
    sig_new[:, 2] = ratio * t3 * off_arr - pnew_arr
    sig_new[:, 3] = ratio * t4 * off_arr
    sig_new[:, 4] = ratio * t5 * off_arr
    sig_new[:, 5] = ratio * t6 * off_arr

    # Plastic strain increment (sigeps102.F:157-159)
    dpla = (1.0 - ratio) * np.sqrt(np.maximum(0.0, aj2)) / max(_EM20, 3.0 * g)
    if history.shape[1] >= 4:
        history_new = np.empty((nel, 4), dtype=np.float64)
        history_new[:, 0] = history[:, 0] + dpla
        history_new[:, 1] = pnew_arr
        history_new[:, 2] = history[:, 2]
        history_new[:, 3] = history[:, 3]
    else:
        history_new = np.empty((nel, 2), dtype=np.float64)
        history_new[:, 0] = history[:, 0] + dpla
        history_new[:, 1] = pnew_arr

    return sig_new, history_new, ssp


def solid_update(
    mat: Any = None,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    eps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    *,
    epsp: Optional[np.ndarray] = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> Any:
    """3D solid continuum Extended Drucker-Prager stress update.

    Supports both standard solver group call:
      solid_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    and direct array call:
      solid_update(params, deps, sig_old, history=history)
    """
    # Check if caller used direct array call: solid_update(params, deps, sig_old, history=...)
    if isinstance(mat, DPrag2Params) and sig is not None and deps is not None and "history" in kwargs:
        deps_in = np.asarray(sig, dtype=np.float64)
        sig_in = np.asarray(deps, dtype=np.float64)
        hist_in = np.asarray(kwargs["history"], dtype=np.float64)
        return solid_update_array(
            mat,
            deps_in,
            sig_in,
            hist_in,
            rho=kwargs.get("rho"),
            rho0=kwargs.get("rho0"),
            off=kwargs.get("off"),
            pnew=kwargs.get("pnew"),
            psh=kwargs.get("psh", 0.0),
        )

    # Standard element / materials.solid_update call:
    params = mat if isinstance(mat, DPrag2Params) else build_law102(mat, **kwargs)

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    is_1d = (sig.ndim == 1)

    sig_arr = sig[np.newaxis, :] if is_1d else sig
    deps_arr = deps[np.newaxis, :] if (deps is not None and is_1d) else (deps if deps is not None else np.zeros_like(sig_arr))
    nel = len(sig_arr)
    epsp_arr = epsp[np.newaxis] if (epsp is not None and is_1d) else (epsp if epsp is not None else np.zeros(nel))

    # Retrieve history variables from extra
    hist = None
    if extra is not None:
        for k in ("uvar102", "uvar", "history"):
            if k in extra and extra[k] is not None:
                hist = extra[k]
                break

    use_cap = (params.r_cap > 0.0 and params.pa > 0.0)
    if hist is None or len(hist) == 0:
        hist = init_history(nel, params=params)
    elif hist.ndim == 1:
        hist = hist[np.newaxis, :]

    if use_cap and hist.shape[1] < 4:
        pa_init = params.pa0 if params.pa0 > 0.0 else params.pa
        hist_4 = np.zeros((len(hist), 4), dtype=np.float64)
        hist_4[:, :min(hist.shape[1], 4)] = hist[:, :min(hist.shape[1], 4)]
        hist_4[:, 2] = pa_init
        hist = hist_4

    rho_arr = extra.get("rho") if extra else None
    rho0_arr = extra.get("rho0") if extra else None
    off_arr = extra.get("off") if extra else None
    pnew_arr = extra.get("pnew") if extra else None
    psh_arr = extra.get("psh", 0.0) if extra else 0.0

    sig_new, hist_new, ssp = solid_update_array(
        params,
        deps_arr,
        sig_arr,
        hist,
        rho=rho_arr,
        rho0=rho0_arr,
        off=off_arr,
        pnew=pnew_arr,
        psh=psh_arr,
    )

    if extra is not None:
        extra["uvar102"] = hist_new
        extra["uvar"] = hist_new
        extra["history"] = hist_new
        if use_cap:
            extra["pa"] = hist_new[0, 2] if is_1d else hist_new[:, 2]
            extra["epspv"] = hist_new[0, 3] if is_1d else hist_new[:, 3]

    sig_out = sig_new[0] if is_1d else sig_new
    epsp_out = hist_new[0, 0] if is_1d else hist_new[:, 0]
    c_out = float(ssp[0]) if is_1d else ssp

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = sig_out
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = epsp_out
        except Exception:
            pass

    return sig_out, epsp_out, c_out



def sound_speed(params: Any, rho: Optional[Any] = None) -> Any:
    """Dilatational acoustic sound speed for /MAT/LAW102 solids."""
    p = params if isinstance(params, DPrag2Params) else build_law102(params)
    stiff = max(0.0, p.bulk + (4.0 / 3.0) * p.g)
    if rho is not None:
        r = np.asarray(rho, dtype=np.float64)
        r_val = np.where(r > 0.0, r, p.rho)
        c = np.where(r_val > 0.0, np.sqrt(np.maximum(0.0, stiff / np.maximum(1e-20, r_val))), 0.0)
        return float(c) if r.ndim == 0 else c
    if p.rho <= 0.0:
        return 0.0
    return float(np.sqrt(np.maximum(0.0, stiff / p.rho)))


def solid_tangent(
    params: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    history: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Algorithmic consistent tangent stiffness tensor for /MAT/LAW102 solids.

    Returns (6, 6) or (N, 6, 6) tensor.
    """
    p = params if isinstance(params, DPrag2Params) else build_law102(params)
    g = p.g
    bulk = p.bulk

    # Standard isotropic 6x6 elasticity tensor in Voigt notation [xx, yy, zz, xy, yz, zx]
    c_mat = np.zeros((6, 6), dtype=np.float64)
    c11 = bulk + 4.0 / 3.0 * g
    c12 = bulk - 2.0 / 3.0 * g

    c_mat[0, 0] = c11
    c_mat[1, 1] = c11
    c_mat[2, 2] = c11

    c_mat[0, 1] = c12
    c_mat[0, 2] = c12
    c_mat[1, 0] = c12
    c_mat[1, 2] = c12
    c_mat[2, 0] = c12
    c_mat[2, 1] = c12

    c_mat[3, 3] = g
    c_mat[4, 4] = g
    c_mat[5, 5] = g

    stress = sig if sig is not None else kwargs.get("stress")
    strain_inc = deps if deps is not None else kwargs.get("strain_inc")

    if stress is not None and np.ndim(stress) > 1:
        n = np.shape(stress)[0]
        return np.broadcast_to(c_mat, (n, 6, 6)).copy()

    if strain_inc is None or stress is None or history is None:
        return c_mat

    # Check if plastic yield occurred via numerical perturbation
    h = 1.0e-7
    tangent = np.zeros((6, 6), dtype=np.float64)
    deps0 = np.asarray(strain_inc, dtype=np.float64)
    sig0 = np.asarray(stress, dtype=np.float64)
    hist0 = np.asarray(history, dtype=np.float64)

    base_sig, _, _ = _solid_update_single_core(p, deps0, sig0, hist0)

    for j in range(6):
        deps_p = deps0.copy()
        deps_p[j] += h
        sig_p, _, _ = _solid_update_single_core(p, deps_p, sig0, hist0)
        tangent[:, j] = (sig_p - base_sig) / h

    return 0.5 * (tangent + tangent.T)


def shell_update(*args: Any, **kwargs: Any) -> Any:
    """LAW102 does not support 2D shell elements."""
    raise NotImplementedError(
        "/MAT/LAW102 (/MAT/DPRAG2) is a 3D solid continuum material; "
        "shell elements are rejected with message ANCMSG 305."
    )


def shell_layer_tangent(*args: Any, **kwargs: Any) -> Any:
    """LAW102 does not support shell layer tangents."""
    raise NotImplementedError(
        "/MAT/LAW102 (/MAT/DPRAG2) does not support shell elements."
    )


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Extra history shapes needed for LAW102."""
    if mat is not None:
        p = mat if isinstance(mat, DPrag2Params) else build_law102(mat)
        if p.r_cap > 0.0 and p.pa > 0.0:
            return {"uvar102": (4,)}
    return {"uvar102": (2,)}


tangent = solid_tangent
consistent_solid_tangent = solid_tangent


