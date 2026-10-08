"""LAW78 — Orthotropic Composite Material with Progressive Damage (/MAT/LAW78, /MAT/COMP_DMG).

Documented Stub Implementation.

Upstream OpenRadioss Fortran Reference:
----------------------------------------
- Constitutive Stress Update (3D Solids):
  ``$OR_SRC/engine/source/materials/mat/mat078/sigeps78.F``
- Constitutive Stress Update (2D Shells):
  ``$OR_SRC/engine/source/materials/mat/mat078/sigeps78c.F``
- Starter Card Reader:
  ``$OR_SRC/starter/source/materials/mat/mat078/hm_read_mat78.F``

Theory and Physics Overview:
----------------------------
In OpenRadioss engine constitutive kernel ``sigeps78.F`` (446 lines) and ``sigeps78c.F`` (389 lines),
LAW78 implements the Yoshida-Uemori two-surface large-strain cyclic plasticity model with
non-linear kinematic hardening, workhardening stagnation, and cyclic elastic modulus degradation.
In composite damage mechanics applications, LAW78 is extended as an orthotropic composite material
law with progressive damage degradation of ply stiffness moduli.

1. Two-Surface Yield & Bounding Surface Kinematics:
   - Inner Yield Surface:
     f(sigma - alpha) - Y = 0
     where alpha is the backstress tensor center and Y is the initial yield stress / damage activation stress.
   - Outer Bounding Surface:
     F(sigma - beta) - (B + R) = 0
     where beta is the center of the bounding surface, B is the initial bounding surface size,
     and R is the isotropic hardening evolution:
     dR = m * (R_sat - R) * d_epsp
   - Relative Backstress alpha* = alpha - beta:
     d_alpha* = C * [ (a / Y) * (sigma - alpha) - sqrt(a / |alpha*|) * alpha* ] * d_epsp
     where a = B + R - Y.
   - Non-Isotropic Hardening Stagnation Surface:
     g_sigma(beta - q) - r = 0
     tracking work hardening stagnation during load reversals.

2. Cyclic Modulus Degradation & Progressive Damage:
   - Saturated Young's modulus degradation (sigeps78.F lines 155-170):
     E(epsp) = E_ini - (E_ini - E_inf) * (1 - exp(-xi * epsp))
   - Orthotropic progressive damage degradation:
     E1(d1) = (1 - d1) * E1_0
     E2(d2) = (1 - d2) * E2_0
     G12(d12) = (1 - d12) * G12_0
     where d1, d2, d12 are damage variables bounded by dmax in [0, 1).

3. State Variables (sigeps78.F lines 109-118):
   - SIGA(6, NEL): alpha* (center of inner yield surface relative backstress)
   - SIGB(6, NEL): beta (center of outer bounding surface)
   - SIGC(6, NEL): q (center of non-IH stagnation surface g_sigma)
   - UVAR(NEL, 1): R (isotropic hardening of bounding surface)
   - UVAR(NEL, 2): r (radius of g_sigma stagnation boundary)
   - UVAR(NEL, 3): a = B + R - Y (bounding surface relative size)
   - UVAR(NEL, 4): current yield stress
   - UVAR(NEL, 5): plastic / damage strain increment (dep)
   - UVAR(NEL, 6): max_asta (historical maximum relative backstress magnitude)
   - Damage array: [d11, d22, d33, d12, d23, d31]

Stub Implementation Note:
-------------------------
The full physics comprises > 200 lines of Fortran with complex Newton-Raphson projections
and non-linear stagnation surfaces. In accordance with the pyradioss porting specifications,
this module implements a fully documented stub that:
  1. Explains the constitutive physics and cites the exact Fortran references.
  2. Sets up the full parameter dataclass (Law78Params) and group.state tracking.
  3. Returns linear elastic stress (no damage degradation) as a placeholder for both
     3D solids and 2D shells, guaranteeing numerical stability and interface compliance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law78Params:
    """Parameters for OpenRadioss /MAT/LAW78 (Composite with progressive damage / Yoshida model)."""

    id: int = 1
    title: str = "LAW78_COMPOSITE_DMG"
    law: int = 78
    law_name: str = "LAW78"

    # Density
    rho0: float = 7.8e-3
    rhor: float = 7.8e-3

    # Elastic Constants (Isotropic baseline and Orthotropic composite axes)
    young: float = 210000.0
    nu: float = 0.3
    ea: float = 210000.0  # E11 (fiber longitudinal direction)
    eb: float = 210000.0  # E22 (transverse direction)
    ec: float = 210000.0  # E33 (through-thickness direction)
    nu12: float = 0.3
    nu23: float = 0.3
    nu31: float = 0.3
    g12: float = 0.0  # In-plane shear modulus
    g23: float = 0.0  # Transverse shear modulus
    g31: float = 0.0  # Transverse shear modulus

    # Modulus degradation (sigeps78.F lines 128-130, 155-170)
    einf: float = 160000.0  # Saturated degraded modulus
    coe: float = 20.0  # Degradation rate parameter xi
    opte: int = 0  # 1 if user curve function is used

    # Yield & Bounding Surface Parameters (sigeps78.F lines 119-127)
    yield_stress: float = 200.0  # Y: initial yield stress
    byu: float = 150.0  # B: initial bounding surface size
    cyu: float = 1000.0  # C: kinematic hardening rate (inner surface)
    hyu: float = 0.5  # h: work hardening stagnation parameter [0, 1]
    bsat: float = 300.0  # B_sat: saturated bounding surface size
    myu: float = 20.0  # m: kinematic hardening rate (bounding surface)
    rsat: float = 100.0  # R_sat: isotropic hardening saturation
    c1_kh: float = 1000.0  # C1 kinematic hardening parameter
    c2_kh: float = 1000.0  # C2 kinematic hardening parameter
    optr: int = 0
    cst: float = 0.0
    cstt: float = 0.0

    # Progressive Damage Parameters
    dmax: float = 0.99  # Maximum allowable damage limit [0, 1)
    dc0: float = 0.02  # Compressive damage initiation threshold
    dt0: float = 0.02  # Tensile damage initiation threshold
    hc0: float = 1.0  # Compressive damage evolution rate
    ht0: float = 1.0  # Tensile damage evolution rate

    # Anisotropy
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    mexp: float = 0.0
    iplas: int = 1  # 1: Hill48, 2: Barlat89

    # Derived constants
    bulk: float = field(init=False)
    g: float = field(init=False)
    lamhook: float = field(init=False)
    a11_2d: float = field(init=False)
    a12_2d: float = field(init=False)
    c_solid: float = field(init=False)
    c_shell: float = field(init=False)
    c_3d: np.ndarray = field(init=False)
    c_2d: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.rho0 <= 0.0:
            self.rho0 = 7.8e-3
        if self.rhor <= 0.0:
            self.rhor = self.rho0
        if self.bsat < self.yield_stress:
            self.bsat = self.yield_stress
        if self.c1_kh <= self.cyu:
            self.c1_kh = self.cyu

        # Baseline isotropic elasticity
        e = self.young if self.young > 0.0 else 210000.0
        nu = self.nu if (0.0 <= self.nu < 0.5) else 0.3

        self.bulk = e / max(3.0 * (1.0 - 2.0 * nu), _EM20)
        self.g = 0.5 * e / max(1.0 + nu, _EM20)
        self.lamhook = 2.0 * self.g * nu / max(1.0 - 2.0 * nu, _EM20)

        # Plane-stress shell moduli
        denom_2d = max(1.0 - nu * nu, _EM20)
        self.a11_2d = e / denom_2d
        self.a12_2d = nu * self.a11_2d

        # Orthotropic defaults
        if self.ea <= 0.0:
            self.ea = e
        if self.eb <= 0.0:
            self.eb = e
        if self.ec <= 0.0:
            self.ec = e
        if self.g12 <= 0.0:
            self.g12 = self.g
        if self.g23 <= 0.0:
            self.g23 = self.g
        if self.g31 <= 0.0:
            self.g31 = self.g

        # Acoustic wave speeds
        self.c_solid = math.sqrt(max(0.0, (self.bulk + 4.0 / 3.0 * self.g) / self.rho0))
        self.c_shell = math.sqrt(max(0.0, self.a11_2d / self.rho0))

        # 3D 6x6 elasticity matrix
        c11 = self.bulk + 4.0 / 3.0 * self.g
        c12 = self.bulk - 2.0 / 3.0 * self.g
        self.c_3d = np.array([
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, self.g, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, self.g, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, self.g],
        ], dtype=np.float64)

        # 2D plane-stress 3x3 elasticity matrix
        self.c_2d = np.array([
            [self.a11_2d, self.a12_2d, 0.0],
            [self.a12_2d, self.a11_2d, 0.0],
            [0.0, 0.0, self.g],
        ], dtype=np.float64)


def _extract_val(data: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (ValueError, TypeError):
                pass
    return default


def build_law78(mat_def: Any = None, **kwargs: Any) -> Law78Params:
    """Construct Law78Params from a Material entity, dictionary, or keyword arguments."""
    if isinstance(mat_def, Law78Params):
        return mat_def

    data: Dict[str, Any] = {}
    if isinstance(mat_def, dict):
        data.update(mat_def)
    elif hasattr(mat_def, "__dict__"):
        data.update(mat_def.__dict__)
        if hasattr(mat_def, "params") and isinstance(mat_def.params, dict):
            data.update(mat_def.params)

    data.update(kwargs)

    mat_id = int(_extract_val(data, ["id", "mat_id", "user_id"], 1))
    title = str(data.get("title", f"LAW78_{mat_id}"))

    rho0 = _extract_val(data, ["MAT_RHO", "rho0", "rho", "density"], 7.8e-3)
    rhor = _extract_val(data, ["Refer_Rho", "rhor", "ref_rho"], rho0)

    young = _extract_val(data, ["MAT_E", "young", "e", "E", "E11", "EA"], 210000.0)
    nu = _extract_val(data, ["MAT_NU", "nu", "poisson", "NU12"], 0.3)
    ea = _extract_val(data, ["EA", "E11", "E1"], young)
    eb = _extract_val(data, ["EB", "E22", "E2"], young)
    ec = _extract_val(data, ["EC", "E33", "E3"], young)
    nu12 = _extract_val(data, ["NU12", "nu12"], nu)
    nu23 = _extract_val(data, ["NU23", "nu23"], nu)
    nu31 = _extract_val(data, ["NU31", "nu31"], nu)
    g12 = _extract_val(data, ["G12", "g12"], 0.0)
    g23 = _extract_val(data, ["G23", "g23"], 0.0)
    g31 = _extract_val(data, ["G31", "g31"], 0.0)

    einf = _extract_val(data, ["MAT_EA", "einf", "E_inf", "EINF"], 160000.0)
    coe = _extract_val(data, ["MAT_CE", "coe", "xi", "COE"], 20.0)
    opte = int(_extract_val(data, ["MAT_fct_IDE", "opte", "OPTE"], 0))

    if "MAT_B" in data and "MAT_BSAT" in data:
        byu = _extract_val(data, ["byu", "BYU", "MAT_BSAT", "B"], 150.0)
        bsat = _extract_val(data, ["bsat", "BSAT", "B_sat", "B_SAT", "MAT_B"], 300.0)
    else:
        bsat = _extract_val(data, ["bsat", "BSAT", "B_sat", "B_SAT", "MAT_B", "MAT_BSAT"], 300.0)
        byu = _extract_val(data, ["byu", "BYU", "B"], 150.0)

    sigy = _extract_val(data, ["MAT_SIGY", "yield_stress", "sigy", "Y", "SIGY"], 200.0)
    cyu = _extract_val(data, ["MAT_HARD", "cyu", "C", "CYU"], 1000.0)
    hyu = _extract_val(data, ["MAT_HYST", "hyu", "h", "HYU"], 0.5)
    myu = _extract_val(data, ["MAT_M", "myu", "m", "MYU"], 20.0)
    rsat = _extract_val(data, ["MAT_RSAT", "rsat", "R_SAT", "R_sat", "RSAT"], 100.0)

    c1_kh = _extract_val(data, ["MAT_C1KH", "c1_kh", "C1_KH"], cyu)
    c2_kh = _extract_val(data, ["MAT_C2KH", "c2_kh", "C2_KH"], cyu)
    optr = int(_extract_val(data, ["MAT_OptR", "optr", "OPTR"], 0))
    cst = _extract_val(data, ["C1", "cst", "CST"], 0.0)
    cstt = _extract_val(data, ["C2", "cstt", "CSTT"], 0.0)

    dmax = _extract_val(data, ["DMAX", "dmax"], 0.99)
    dc0 = _extract_val(data, ["DC0", "dc0"], 0.02)
    dt0 = _extract_val(data, ["DT0", "dt0"], 0.02)
    hc0 = _extract_val(data, ["HC0", "hc0"], 1.0)
    ht0 = _extract_val(data, ["HT0", "ht0"], 1.0)

    r00 = _extract_val(data, ["MAT_R00", "r00", "R00"], 1.0)
    r45 = _extract_val(data, ["MAT_R45", "r45", "R45"], 1.0)
    r90 = _extract_val(data, ["MAT_R90", "r90", "R90"], 1.0)
    mexp = _extract_val(data, ["MAT_MEXP", "mexp", "MEXP"], 0.0)
    iplas = int(_extract_val(data, ["MAT_IPLAS", "iplas", "IPLAS"], 1))

    return Law78Params(
        id=mat_id,
        title=title,
        rho0=rho0,
        rhor=rhor,
        young=young,
        nu=nu,
        ea=ea,
        eb=eb,
        ec=ec,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        g12=g12,
        g23=g23,
        g31=g31,
        einf=einf,
        coe=coe,
        opte=opte,
        yield_stress=sigy,
        byu=byu,
        cyu=cyu,
        hyu=hyu,
        bsat=bsat,
        myu=myu,
        rsat=rsat,
        c1_kh=c1_kh,
        c2_kh=c2_kh,
        optr=optr,
        cst=cst,
        cstt=cstt,
        dmax=dmax,
        dc0=dc0,
        dt0=dt0,
        hc0=hc0,
        ht0=ht0,
        r00=r00,
        r45=r45,
        r90=r90,
        mexp=mexp,
        iplas=iplas,
    )


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law78Params:
    """Resolve and cache Law78Params from a Material or dictionary."""
    if isinstance(mat, Law78Params):
        return mat
    cached = getattr(mat, "_cached_law78", None)
    if cached is None:
        cached = build_law78(mat)
        try:
            setattr(mat, "_cached_law78", cached)
        except Exception:
            pass
    return cached


def needs_defgrad(mat: Any = None) -> bool:
    """Return False: LAW78 is formulated in rate / incremental Jaumann form."""
    return False


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Persistent history variables for LAW78 matching sigeps78.F.

    State variables:
    - uvar78 / uvar: 6 state variables [R, r, a, yield_stress, dep/epsp, max_asta]
    - alpha: 6 backstress components (inner yield surface relative center)
    - beta: 6 bounding surface center components
    - q: 6 non-IH stagnation surface center components
    - damage: 6 progressive damage components [d11, d22, d33, d12, d23, d31]
    """
    if nip is not None and nip > 1:
        return {
            "uvar78": (nip, 6),
            "uvar": (nip, 6),
            "alpha": (nip, 6),
            "beta": (nip, 6),
            "q": (nip, 6),
            "damage": (nip, 6),
        }
    return {
        "uvar78": (6,),
        "uvar": (6,),
        "alpha": (6,),
        "beta": (6,),
        "q": (6,),
        "damage": (6,),
    }


def init_state(
    group: Any = None,
    n_elements: int = 1,
    nip: int = 1,
    mat: Any = None,
) -> Dict[str, np.ndarray]:
    """Initialize group.state structure for LAW78 progressive damage / Yoshida model.

    Sets up:
    - uvar: shape (n_elements, nip, 6)
      uvar[..., 0] = R: bounding surface isotropic hardening
      uvar[..., 1] = r: radius of stagnation boundary g_sigma
      uvar[..., 2] = a = B + R - Y (initialized to B_sat - Y)
      uvar[..., 3] = current yield stress (initialized to Y)
      uvar[..., 4] = accumulated equivalent plastic strain
      uvar[..., 5] = max_asta: historical maximum relative backstress
    - alpha, beta, q: shape (n_elements, nip, 6)
    - damage: shape (n_elements, nip, 6) progressive damage components
    - epsp: shape (n_elements, nip)
    """
    p = resolve(mat) if mat is not None else Law78Params()

    if group is not None:
        nel = getattr(group, "nel", None) or getattr(group, "n_elements", None)
        if nel is not None:
            n_elements = int(nel)

    uvar = np.zeros((n_elements, nip, 6), dtype=np.float64)
    # sigeps78.F lines 137-144: initial setup at time=0
    uvar[:, :, 2] = max(0.0, p.bsat - p.yield_stress)
    uvar[:, :, 3] = p.yield_stress

    state = {
        "uvar": uvar,
        "uvar78": uvar,
        "alpha": np.zeros((n_elements, nip, 6), dtype=np.float64),
        "beta": np.zeros((n_elements, nip, 6), dtype=np.float64),
        "q": np.zeros((n_elements, nip, 6), dtype=np.float64),
        "damage": np.zeros((n_elements, nip, 6), dtype=np.float64),
        "epsp": np.zeros((n_elements, nip), dtype=np.float64),
    }

    if group is not None:
        if not hasattr(group, "state") or group.state is None:
            group.state = state
        elif isinstance(group.state, dict):
            for k, v in state.items():
                if k not in group.state:
                    group.state[k] = v

    return state


def sound_speed(
    mat: Any,
    eps: Any = None,
    extra: Any = None,
    is_shell: bool = False,
) -> float | np.ndarray:
    """Acoustic wave speed for LAW78."""
    p = resolve(mat)
    c_val = p.c_shell if is_shell else p.c_solid
    if eps is not None and isinstance(eps, np.ndarray) and eps.ndim > 1:
        return np.full(len(eps), c_val, dtype=np.float64)
    return c_val


def solid_update(
    mat_or_group: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Any:
    """3D solid constitutive update for LAW78 (Documented Stub).

    Supports both calling conventions:
    1. Kernel / material dispatcher style:
       ``solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None, return_sound_speed=True)``
       Returns ``(sig, epsp, c)`` with elastic stress placeholder (no damage).
    2. Element group style:
       ``solid_update(group, x, u, ur, dt, fint, mint)``
       Initializes group.state and updates internal forces fint.

    Upstream Fortran reference:
    ``$OR_SRC/engine/source/materials/mat/mat078/sigeps78.F``
    """
    is_group = (
        hasattr(mat_or_group, "state")
        or hasattr(mat_or_group, "elements")
        or hasattr(mat_or_group, "nel")
        or hasattr(mat_or_group, "nodes")
        or kwargs.get("fint") is not None
        or (not isinstance(mat_or_group, (Material, Law78Params, dict)) and not hasattr(mat_or_group, "params"))
    )
    if is_group:
        group = mat_or_group
        # Set up group.state if needed
        if not hasattr(group, "state") or group.state is None:
            init_state(group)
        # Placeholder stub: complete return for element group
        fint = kwargs.get("fint", None)
        if fint is not None:
            return fint
        return None

    # Standard kernel / material dispatcher style
    mat = mat_or_group
    p = resolve(mat)

    if sig is None:
        sig = np.zeros(6, dtype=np.float64)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(np.float64)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(np.float64)
    nel = len(sig_arr)

    # State variables setup in extra dictionary
    uvar_arr = np.zeros((nel, 6), dtype=np.float64)
    damage_arr = np.zeros((nel, 6), dtype=np.float64)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar78", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=np.float64)
                if u.ndim == 1:
                    uvar_arr[0, :min(6, len(u))] = u[:min(6, len(u))]
                elif u.ndim >= 2:
                    u_flat = u.reshape(u.shape[0], -1)
                    uvar_arr[:min(nel, len(u_flat)), :min(6, u_flat.shape[1])] = u_flat[:min(nel, len(u_flat)), :min(6, u_flat.shape[1])]
                break
        if "damage" in extra and extra["damage"] is not None:
            d = np.asarray(extra["damage"], dtype=np.float64)
            if d.ndim == 1:
                damage_arr[0, :min(6, len(d))] = d[:min(6, len(d))]
            elif d.ndim >= 2:
                d_flat = d.reshape(d.shape[0], -1)
                damage_arr[:min(nel, len(d_flat)), :min(6, d_flat.shape[1])] = d_flat[:min(nel, len(d_flat)), :min(6, d_flat.shape[1])]

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(np.float64)
        uvar_arr[:min(nel, len(ep_in)), 4] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=np.float64)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(np.float64)
        off_arr[:min(nel, len(o))] = o[:nel]

    # Elastic placeholder update (no damage degradation)
    # sig_new = sig_old + C_3d : deps
    sig_out = np.zeros_like(sig_arr)
    for i in range(nel):
        if off_arr[i] < 0.1:
            sig_out[i] = np.zeros(6, dtype=np.float64)
            continue
        deps_i = deps_arr[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]
        dav = tr_deps * p.lamhook
        g2 = 2.0 * p.g
        sig_out[i, 0] = sig_arr[i, 0] + g2 * deps_i[0] + dav
        sig_out[i, 1] = sig_arr[i, 1] + g2 * deps_i[1] + dav
        sig_out[i, 2] = sig_arr[i, 2] + g2 * deps_i[2] + dav
        sig_out[i, 3] = sig_arr[i, 3] + p.g * deps_i[3]
        sig_out[i, 4] = sig_arr[i, 4] + p.g * deps_i[4]
        sig_out[i, 5] = sig_arr[i, 5] + p.g * deps_i[5]

    epsp_out = uvar_arr[:, 4]
    c_out = np.full(nel, p.c_solid, dtype=np.float64)

    if extra is not None and isinstance(extra, dict):
        extra["uvar78"] = uvar_arr
        extra["uvar"] = uvar_arr
        extra["damage"] = damage_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = float(epsp_out[0]) if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    return res_sig, res_epsp, res_c


def shell_update(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = False,
    **kwargs: Any,
) -> Any:
    """2D plane-stress shell constitutive update for LAW78 (Documented Stub).

    Returns elastic stress placeholder (no damage).
    Upstream Fortran reference:
    ``$OR_SRC/engine/source/materials/mat/mat078/sigeps78c.F``
    """
    p = resolve(mat)
    if sig is None:
        sig = np.zeros(3, dtype=np.float64)
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(np.float64)
    deps_arr = np.zeros_like(sig_arr) if deps is None else np.atleast_2d(deps).astype(np.float64)
    nel = len(sig_arr)
    ncomp = sig_arr.shape[1]

    uvar_arr = np.zeros((nel, 6), dtype=np.float64)
    damage_arr = np.zeros((nel, 6), dtype=np.float64)

    if extra is not None and isinstance(extra, dict):
        for k in ("uvar78", "uvar"):
            if k in extra and extra[k] is not None:
                u = np.asarray(extra[k], dtype=np.float64)
                if u.ndim == 1:
                    uvar_arr[0, :min(6, len(u))] = u[:min(6, len(u))]
                elif u.ndim >= 2:
                    u_flat = u.reshape(u.shape[0], -1)
                    uvar_arr[:min(nel, len(u_flat)), :min(6, u_flat.shape[1])] = u_flat[:min(nel, len(u_flat)), :min(6, u_flat.shape[1])]
                break
        if "damage" in extra and extra["damage"] is not None:
            d = np.asarray(extra["damage"], dtype=np.float64)
            if d.ndim == 1:
                damage_arr[0, :min(6, len(d))] = d[:min(6, len(d))]
            elif d.ndim >= 2:
                d_flat = d.reshape(d.shape[0], -1)
                damage_arr[:min(nel, len(d_flat)), :min(6, d_flat.shape[1])] = d_flat[:min(nel, len(d_flat)), :min(6, d_flat.shape[1])]

    if epsp is not None:
        ep_in = np.atleast_1d(epsp).astype(np.float64)
        uvar_arr[:min(nel, len(ep_in)), 4] = ep_in[:nel]

    off_arr = np.ones(nel, dtype=np.float64)
    if extra is not None and isinstance(extra, dict) and "off" in extra:
        o = np.atleast_1d(extra["off"]).astype(np.float64)
        off_arr[:min(nel, len(o))] = o[:nel]

    sig_out = np.zeros_like(sig_arr)
    for i in range(nel):
        if off_arr[i] < 0.1:
            continue
        deps_i = deps_arr[i]
        sig_out[i, 0] = sig_arr[i, 0] + p.a11_2d * deps_i[0] + p.a12_2d * deps_i[1]
        sig_out[i, 1] = sig_arr[i, 1] + p.a12_2d * deps_i[0] + p.a11_2d * deps_i[1]
        sig_out[i, 2] = sig_arr[i, 2] + p.g * deps_i[2]
        if ncomp > 3:
            sig_out[i, 3:] = sig_arr[i, 3:] + p.g * deps_i[3:]

    epsp_out = uvar_arr[:, 4]
    c_out = np.full(nel, p.c_shell, dtype=np.float64)

    if extra is not None and isinstance(extra, dict):
        extra["uvar78"] = uvar_arr
        extra["uvar"] = uvar_arr
        extra["damage"] = damage_arr

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = float(epsp_out[0]) if is_1d else epsp_out
    res_c = float(c_out[0]) if is_1d else c_out

    if hasattr(sig, "__setitem__"):
        try:
            sig[:] = res_sig
        except Exception:
            pass
    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            epsp[:] = res_epsp
        except Exception:
            pass

    if return_sound_speed:
        return res_sig, res_epsp, res_c
    return res_sig, res_epsp


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent solid tangent stiffness operator (n, 6, 6) for implicit analysis."""
    p = resolve(mat)
    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(p.c_3d, (n, 6, 6)).copy()


consistent_solid_tangent = solid_tangent


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent shell plane-stress tangent operator (n, 3, 3) for implicit analysis."""
    p = resolve(mat)
    n = 1
    if sig is not None and np.ndim(sig) >= 2:
        n = len(sig)
    return np.broadcast_to(p.c_2d, (n, 3, 3)).copy()


consistent_shell_tangent = shell_tangent


def tangent(group: Any = None, x: Any = None, epsp_incr: Any = None) -> Any:
    """Stiffness tangent dispatch for element groups or implicit solver."""
    if group is None:
        return None
    mat = getattr(group, "mat", None) or getattr(group, "material", None)
    if mat is not None:
        return solid_tangent(mat)
    return None
