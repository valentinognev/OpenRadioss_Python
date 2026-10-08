# $OR_SRC/starter/source/initial_conditions/inicrack/hm_read_inicrack.F
# Function: HM_READ_INICRACK (lines 40-138)
# $OR_SRC/starter/source/materials/fail/windshield_alter/brokmann_crack_init.F90
# Subroutine: BROKMANN_CRACK_INIT (lines 48-374)
# $OR_SRC/engine/source/materials/fail/alter/fail_brokmann.F
# Subroutine: FAIL_BROKMANN (lines 35-188)
# $OR_SRC/common_source/fail/newman_raju.F90
# Subroutine: NEWMAN_RAJU (lines 51-102)
"""Initial crack definition (/INICRACK) for fracture mechanics and failure models.

Fortran origin
--------------
* starter card reader:
    ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``
    (reads /INICRACK cards, element partitioning segments, and geometric data);
* Brokmann crack initialization:
    ``starter/source/materials/fail/windshield_alter/brokmann_crack_init.F90``
    (initializes random or deterministic surface micro-cracks on glass shells,
    computes initial aspect ratio a/c, Weibull fracture stresses, initial SIFs);
* engine subcritical crack growth & failure:
    ``engine/source/materials/fail/alter/fail_brokmann.F``
    (Mohr's circle crack opening stress, Paris law subcritical propagation,
    K_1A / K_1C evaluation, and critical stress intensity factor rupture test);
* Newman-Raju geometry correction:
    ``common_source/fail/newman_raju.F90`` (lines 51-102).

Theory & Physics
----------------
1. Initial Crack State:
   A semi-elliptical surface crack is defined by:
     - depth a0 (in thickness direction)
     - half-length c0 (in surface length direction)
     - aspect ratio: a0 / c0
     - center point (x0, y0, z0)
     - crack plane normal direction vector n = (nx, ny, nz)
     - residual pre-stress sigma_ini at the crack tip.

2. Shell Surface Projection & Orientation Mapping:
   For shell elements with unit normal e3:
     n_proj = n - (n . e3) * e3
     v = n_proj / ||n_proj||
   The in-plane orientation angle theta relative to element local x-axis e1 is:
     cos(theta) = v . e1
     sin(theta) = v . e2
     theta = atan2(sin(theta), cos(theta))
   In the Brokmann failure model (/FAIL/ALTER Brokmann), this is stored as:
     CR_ANG = theta
     theta_mohr = 2 * CR_ANG = 2 * theta
   The crack opening stress rotated into the crack plane is:
     sigma_n = 0.5 * (sig_xx + sig_yy)
             + 0.5 * (sig_xx - sig_yy) * cos(2 * theta)
             + sig_xy * sin(2 * theta)

3. Newman-Raju Stress Intensity Factors:
   With effective opening stress sigma_eff = max(0, sigma_n - sigma_ini):
     K_1A = Y_A * sigma_eff * sqrt(pi * a)   (deepest tip in depth direction, phi = pi/2)
     K_1C = Y_C * sigma_eff * sqrt(pi * c)   (surface tip along length, phi = 0)
   where Y_A and Y_C are the Newman-Raju (1981) geometry boundary correction factors.

4. Paris Law Subcritical Crack Growth:
   When K_th <= K_1 < K_IC:
     v_A = v0 * (K_1A / K_IC)^n
     v_C = v0 * (K_1C / K_IC)^n
     da = v_A * dt
     dc = v_C * dt
   Failure occurs when K_1A >= K_IC or K_1C >= K_IC (or crack penetrates thickness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_TINY = 1.0e-20
_EP06 = 1.0e6   # conversion to micrometers (EP06 in Fortran)
_EM6 = 1.0e-6   # conversion from micrometers to meters (EM6 in Fortran)


# ---------------------------------------------------------------------------
# Newman-Raju geometry correction factor (common_source/fail/newman_raju.F90)
# ---------------------------------------------------------------------------

def newman_raju(c: float, a: float, t: float, b: float, fpi: float) -> float:
    """Newman-Raju (1981) geometry correction factor for a semi-elliptical surface crack.

    Upstream Fortran reference:
      ``common_source/fail/newman_raju.F90`` lines 73-99.

    Parameters
    ----------
    c : float
        Crack half-length along the surface [micrometers or consistent L].
    a : float
        Crack depth into thickness [micrometers or consistent L].
    t : float
        Plate / shell thickness [micrometers or consistent L].
    b : float
        Plate / element half-width [micrometers or consistent L].
    fpi : float
        Angular position factor:
          0.5 -> deepest point (phi = pi/2, tip in depth direction)
          0.0 -> surface point (phi = 0, tip in surface direction).

    Returns
    -------
    float
        Dimensionless geometry correction factor Y.
    """
    if fpi == 0.5:
        sinp = 1.0
        cosp = 0.0
    elif fpi == 0.0:
        sinp = 0.0
        cosp = 1.0
    else:
        sinp = math.sin(fpi * math.pi)
        cosp = math.cos(fpi * math.pi)

    ac = a / max(c, _TINY)
    at = a / max(t, _TINY)
    q = 1.0 + 1.464 * (ac ** 1.65)

    m1 = 1.13 - 0.09 * ac
    m2 = -0.54 + 0.89 / (0.2 + ac)
    m3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * ((1.0 - ac) ** 24)
    g = 1.0 + (0.1 + 0.35 * (at ** 2)) * ((1.0 - sinp) ** 2)

    fphi = (ac ** 2 * cosp ** 2 + sinp ** 2) ** 0.25

    # Finite-width correction (newman_raju.F90 line 96)
    fw = math.cos(math.pi * c / (2.0 * max(b, _TINY)) * math.sqrt(max(at, 0.0)))

    f = (m1 + m2 * (at ** 2) + m3 * (at ** 4)) * fphi * g / math.sqrt(abs(fw) + _TINY)
    y = math.sqrt(1.0 / max(q, _TINY)) * f
    return y


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class IniCrackSegment:
    """Segment definition for element edge partitioning / X-FEM (/INICRACK).

    Upstream reference:
      ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F`` lines 112-125.
    """
    node_id1: int
    node_id2: int
    ratio: float = 0.0


@dataclass
class IniCrackData:
    """Initial crack specification for fracture mechanics and failure models (/INICRACK).

    Upstream Fortran references:
      - ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``
      - ``starter/source/materials/fail/windshield_alter/brokmann_crack_init.F90``
      - ``engine/source/materials/fail/alter/fail_brokmann.F``

    Attributes
    ----------
    id : int
        Crack identifier (from /INICRACK/id).
    part_id : int, optional
        Target part ID associated with the initial crack.
    element_id : int, optional
        Target shell / solid element ID.
    title : str
        Crack descriptive title.
    center : tuple of 3 floats
        Crack center coordinates (x0, y0, z0) in global space.
    a0 : float
        Initial crack depth a0 (into plate thickness) [micrometers or model units].
    c0 : float
        Initial crack half-length c0 (surface direction) [micrometers or model units].
    normal : tuple of 3 floats
        Crack plane normal direction vector in 3D global coordinates.
    sigma_ini : float
        Initial surface residual stress or pre-stress at crack tip [stress units].
    angle : float
        In-plane crack orientation angle theta relative to element local x-axis [radians].
    p1 : tuple of 3 floats
        Crack start endpoint coordinates (for segment-based definitions).
    p2 : tuple of 3 floats
        Crack end endpoint coordinates (for segment-based definitions).
    segments : list of IniCrackSegment
        Optional edge partitioning segments.
    open_flag : int
        Crack opening condition flag (0 = open, 1 = cohesive/closed).
    """
    id: int = 1
    part_id: Optional[int] = None
    element_id: Optional[int] = None
    title: str = ""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    a0: float = 100.0
    c0: float = 100.0
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    sigma_ini: float = 0.0
    angle: float = 0.0
    p1: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    p2: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    segments: List[IniCrackSegment] = field(default_factory=list)
    open_flag: int = 0

    @property
    def depth(self) -> float:
        """Initial crack depth a0."""
        return self.a0

    @depth.setter
    def depth(self, val: float) -> None:
        self.a0 = float(val)

    @property
    def half_length(self) -> float:
        """Initial crack half-length c0."""
        return self.c0

    @half_length.setter
    def half_length(self, val: float) -> None:
        self.c0 = float(val)

    @property
    def length(self) -> float:
        """Total surface crack length 2 * c0."""
        return 2.0 * self.c0

    @property
    def aspect_ratio(self) -> float:
        """Crack aspect ratio a0 / c0."""
        return self.a0 / max(self.c0, _TINY)

    @property
    def x0(self) -> float:
        return self.center[0]

    @property
    def y0(self) -> float:
        return self.center[1]

    @property
    def z0(self) -> float:
        return self.center[2]


# Type alias for model consistency
IniCrack = IniCrackData


# ---------------------------------------------------------------------------
# Vector Projection and In-Plane Orientation Mapping
# ---------------------------------------------------------------------------

def project_crack_orientation(
    crack_normal: Sequence[float],
    shell_normal: Sequence[float],
    shell_x_axis: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float]:
    """Project crack normal vector onto shell element surface and calculate orientation angle.

    Parameters
    ----------
    crack_normal : sequence of 3 floats
        Crack plane normal direction vector in 3D global coordinates.
    shell_normal : sequence of 3 floats
        Shell element surface normal unit vector e3.
    shell_x_axis : sequence of 3 floats, optional
        Local shell element x-axis direction e1. If None, an orthogonal local
        reference frame is constructed automatically.

    Returns
    -------
    proj_vec : ndarray of shape (3,)
        Normalized in-plane crack orientation vector lying on the shell surface.
    theta : float
        In-plane crack angle in radians in [-pi, pi] relative to the local shell x-axis.
    """
    n = np.asarray(crack_normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm > _TINY:
        n = n / n_norm
    else:
        n = np.array([1.0, 0.0, 0.0])

    e3 = np.asarray(shell_normal, dtype=float)
    e3_norm = np.linalg.norm(e3)
    if e3_norm > _TINY:
        e3 = e3 / e3_norm
    else:
        e3 = np.array([0.0, 0.0, 1.0])

    # Project n onto plane perpendicular to e3: v_proj = n - (n . e3) * e3
    dot_n_e3 = np.dot(n, e3)
    v_proj = n - dot_n_e3 * e3
    v_len = np.linalg.norm(v_proj)

    # Establish local in-plane coordinate axes (e1, e2)
    if shell_x_axis is not None:
        e1_raw = np.asarray(shell_x_axis, dtype=float)
        e1_proj = e1_raw - np.dot(e1_raw, e3) * e3
        e1_len = np.linalg.norm(e1_proj)
        if e1_len > _TINY:
            e1 = e1_proj / e1_len
        else:
            e1 = _default_tangent(e3)
    else:
        e1 = _default_tangent(e3)

    e2 = np.cross(e3, e1)
    e2 = e2 / max(np.linalg.norm(e2), _TINY)

    if v_len < 1.0e-12:
        # Crack normal is parallel to shell normal; default to local e1 direction
        return e1.copy(), 0.0

    v_unit = v_proj / v_len
    cos_th = float(np.clip(np.dot(v_unit, e1), -1.0, 1.0))
    sin_th = float(np.clip(np.dot(v_unit, e2), -1.0, 1.0))
    theta = math.atan2(sin_th, cos_th)

    return v_unit, theta


def _default_tangent(normal: np.ndarray) -> np.ndarray:
    """Construct an arbitrary unit tangent vector orthogonal to normal."""
    n = normal / max(np.linalg.norm(normal), _TINY)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tan = np.cross(n, ref)
    return tan / max(np.linalg.norm(tan), _TINY)


def map_crack_angle(theta: float) -> Tuple[float, float]:
    """Map in-plane physical crack angle to Brokmann state variables.

    Upstream Fortran reference:
      ``engine/source/materials/fail/alter/fail_brokmann.F`` line 114:
      ``CR_ANG = UVAR(I,18) * TWO``

    Parameters
    ----------
    theta : float
        Physical crack orientation angle [radians].

    Returns
    -------
    cr_ang : float
        Internal state angle UVAR(18) = theta.
    theta_mohr : float
        Mohr's circle rotation angle = 2 * theta.
    """
    cr_ang = float(theta)
    theta_mohr = 2.0 * cr_ang
    return cr_ang, theta_mohr


# ---------------------------------------------------------------------------
# Stress Intensity Factors and Failure Criteria
# ---------------------------------------------------------------------------

def compute_crack_sif(
    a: float,
    c: float,
    t: float,
    b: float,
    sigma: float,
    phi: float = math.pi / 2.0,
    unit_fac_l: float = 1.0,
    in_micrometers: bool = True,
) -> Tuple[float, float]:
    """Compute Newman-Raju geometry correction factor Y and Mode I SIF K_I.

    Upstream Fortran reference:
      - ``common_source/fail/newman_raju.F90`` lines 51-102
      - ``engine/source/materials/fail/alter/fail_brokmann.F`` lines 136-139

    Formula:
      K_1 = Y * sigma * sqrt(pi * crack_dim * 1e-6)  [Pa*sqrt(m)]

    Parameters
    ----------
    a : float
        Crack depth [micrometers or model units].
    c : float
        Crack half-length [micrometers or model units].
    t : float
        Plate thickness [micrometers or model units].
    b : float
        Plate half-width [micrometers or model units].
    sigma : float
        Tensile crack-opening stress [stress units].
    phi : float, optional
        Parametric angle:
          pi/2 -> deepest tip in depth direction (phi = 0.5 for newman_raju)
          0.0  -> surface point in length direction (phi = 0.0).
        Default is pi/2.
    unit_fac_l : float, optional
        Length unit conversion factor (default 1.0).
    in_micrometers : bool, optional
        If True, a and c are in micrometers and scaled by 1e-6 in sqrt. Default is True.

    Returns
    -------
    k1 : float
        Stress intensity factor K_I [stress * sqrt(L)].
    y : float
        Newman-Raju dimensionless geometry correction factor.
    """
    fpi = 0.5 if abs(phi - math.pi / 2.0) < 1.0e-5 or phi == 0.5 else 0.0
    y = newman_raju(c, a, t, b, fpi)

    if sigma <= 0.0 or a <= 0.0 or c <= 0.0:
        return 0.0, y

    # Tip dimension: a for deepest tip (phi=pi/2), c for surface point (phi=0)
    dim = a if fpi == 0.5 else c
    len_scale = _EM6 if in_micrometers else 1.0

    k1 = y * sigma * math.sqrt(math.pi * dim * len_scale * unit_fac_l)
    return k1, y


def compute_initial_sif(
    crack: IniCrackData,
    thickness: float,
    width: float,
    sigma: float,
    unit_fac_l: float = 1.0,
    in_micrometers: bool = True,
) -> Dict[str, float]:
    """Compute initial SIF at deepest point (K_1A) and surface point (K_1C).

    Parameters
    ----------
    crack : IniCrackData
        Initial crack definition containing a0, c0, sigma_ini.
    thickness : float
        Plate thickness [micrometers or model units].
    width : float
        Plate half-width [micrometers or model units].
    sigma : float
        Applied tensile normal stress [stress units].
    unit_fac_l : float, optional
        Unit length scale factor. Default is 1.0.
    in_micrometers : bool, optional
        Whether crack dimensions are in micrometers. Default is True.

    Returns
    -------
    dict with keys:
      'K1_A' : SIF at deepest point
      'K1_C' : SIF at surface point
      'Y_A'  : geometry factor at deepest point
      'Y_C'  : geometry factor at surface point
      'sigma_eff' : effective opening stress (sigma - sigma_ini)
      'aspect_ratio' : a0 / c0
      'depth_ratio'  : a0 / thickness
    """
    sigma_eff = max(0.0, sigma - crack.sigma_ini)

    k1_a, y_a = compute_crack_sif(
        a=crack.a0,
        c=crack.c0,
        t=thickness,
        b=width,
        sigma=sigma_eff,
        phi=math.pi / 2.0,
        unit_fac_l=unit_fac_l,
        in_micrometers=in_micrometers,
    )
    k1_c, y_c = compute_crack_sif(
        a=crack.a0,
        c=crack.c0,
        t=thickness,
        b=width,
        sigma=sigma_eff,
        phi=0.0,
        unit_fac_l=unit_fac_l,
        in_micrometers=in_micrometers,
    )

    return {
        "K1_A": k1_a,
        "K1_C": k1_c,
        "Y_A": y_a,
        "Y_C": y_c,
        "sigma_eff": sigma_eff,
        "aspect_ratio": crack.aspect_ratio,
        "depth_ratio": crack.a0 / max(thickness, _TINY),
    }


# ---------------------------------------------------------------------------
# Linking with Failure Models (/FAIL/ALTER Brokmann)
# ---------------------------------------------------------------------------

def init_brokmann_crack_state(
    crack: IniCrackData,
    thickness: float,
    width: float,
    uvar: Optional[np.ndarray] = None,
    fac_l: float = 1.0,
) -> np.ndarray:
    """Initialize internal state variables for /FAIL/ALTER Brokmann crack growth.

    Upstream Fortran reference:
      ``starter/source/materials/fail/windshield_alter/brokmann_crack_init.F90``
      lines 362-366:
        uvar(15) = fail_b   : failure flag (0 = alive / growing, 1 = failed)
        uvar(16) = cr_len   : surface half-length c [micrometers]
        uvar(17) = cr_depth : crack depth a [micrometers]
        uvar(18) = cr_ang   : random / specified crack angle theta [radians]
        uvar(19) = thk0     : initial thickness [micrometers]
        uvar(20) = aldt0    : initial width [micrometers]
        uvar(21) = sig_cos  : crack-opening stress (saved for filtering)

    Python 0-based indexing:
      [14] = fail_b   (0.0)
      [15] = cr_len   (c0)
      [16] = cr_depth (a0)
      [17] = cr_ang   (angle)
      [18] = thk0
      [19] = aldt0
      [20] = sig_cos  (sigma_ini)

    Parameters
    ----------
    crack : IniCrackData
        Crack geometry and pre-stress specification.
    thickness : float
        Plate / shell thickness in model units.
    width : float
        Element size / width in model units.
    uvar : ndarray, optional
        Pre-allocated state variable array of length >= 21. If None, created.
    fac_l : float, optional
        Conversion factor from model length units to meters. Default is 1.0.

    Returns
    -------
    ndarray of shape (21,) or larger containing initialized failure states.
    """
    if uvar is None:
        uvar = np.zeros(21, dtype=float)
    else:
        if len(uvar) < 21:
            extended = np.zeros(21, dtype=float)
            extended[:len(uvar)] = uvar
            uvar = extended

    fac_lenm = _EP06 * fac_l  # conversion to micrometers

    # Scale geometric values to micrometers if defined in model units
    c_um = crack.c0 if crack.c0 > 1.0 else crack.c0 * fac_lenm
    a_um = crack.a0 if crack.a0 > 1.0 else crack.a0 * fac_lenm
    thk_um = thickness * fac_lenm
    aldt_um = width * fac_lenm

    uvar[14] = 0.0          # FAIL_B = 0 (alive / subcritical growth allowed)
    uvar[15] = c_um         # CR_LEN (c in micrometers)
    uvar[16] = a_um         # CR_DEPTH (a in micrometers)
    uvar[17] = crack.angle  # CR_ANG (in radians)
    uvar[18] = thk_um       # THK0
    uvar[19] = aldt_um      # ALDT0
    uvar[20] = crack.sigma_ini  # SIG_COS initial pre-stress

    return uvar


def advance_brokmann_crack(
    crack: IniCrackData,
    stress: Sequence[float] | np.ndarray,
    thickness: float,
    width: float,
    dt: float,
    k_ic: float,
    k_th: float,
    v0: float,
    exp_n: float,
    alpha: float = 0.9,
    fac_l: float = 1.0,
    fac_m: float = 1.0,
    fac_t: float = 1.0,
    uvar: Optional[np.ndarray] = None,
) -> Tuple[bool, float, float, Dict[str, float]]:
    """Advance crack by one time step under applied in-plane stress tensor.

    Upstream Fortran reference:
      ``engine/source/materials/fail/alter/fail_brokmann.F`` lines 108-172.

    Parameters
    ----------
    crack : IniCrackData
        Current crack definition.
    stress : sequence of 3 floats
        In-plane stress tensor [sig_xx, sig_yy, sig_xy] in model units.
    thickness : float
        Shell thickness in model units.
    width : float
        Element width in model units.
    dt : float
        Time increment [model units].
    k_ic : float
        Fracture toughness K_IC [stress * sqrt(L)].
    k_th : float
        Subcritical threshold SIF K_th [stress * sqrt(L)].
    v0 : float
        Reference subcritical crack growth velocity [L / T].
    exp_n : float
        Paris-Erdogan exponent.
    alpha : float, optional
        Exponential averaging filter factor (default 0.9).
    fac_l, fac_m, fac_t : float, optional
        Unit conversion factors.
    uvar : ndarray, optional
        Internal history variable vector.

    Returns
    -------
    is_failed : bool
        True if crack SIF exceeds K_IC (rupture).
    new_a : float
        Updated crack depth a in micrometers.
    new_c : float
        Updated crack half-length c in micrometers.
    diag : dict
        Diagnostic values (K1_A, K1_C, Y_A, Y_C, sig_cos, V_A, V_C).
    """
    if uvar is None:
        uvar = init_brokmann_crack_state(crack, thickness, width, fac_l=fac_l)

    s_xx = float(stress[0])
    s_yy = float(stress[1])
    s_xy = float(stress[2]) if len(stress) > 2 else 0.0

    # 1. Rotated opening stress (fail_brokmann.F lines 114-117)
    cr_ang_2 = 2.0 * uvar[17]
    sig_cos_inst = (
        0.5 * (s_xx + s_yy)
        + 0.5 * (s_xx - s_yy) * math.cos(cr_ang_2)
        + s_xy * math.sin(cr_ang_2)
    )

    # Exponential filter: sig_cos = sig_cos * alpha + prev * (1 - alpha)
    prev_sig_cos = uvar[20]
    sig_cos = sig_cos_inst * alpha + prev_sig_cos * (1.0 - alpha)
    uvar[20] = sig_cos

    # Residual stress subtraction (line 131)
    sig_eff = max(0.0, sig_cos - crack.sigma_ini)

    c_cur = uvar[15]
    a_cur = uvar[16]
    thk_m = uvar[18]
    aldt_m = uvar[19]

    # Newman-Raju geometry correction
    ya = newman_raju(c_cur, a_cur, thk_m, aldt_m, 0.5)
    yc = newman_raju(c_cur, a_cur, thk_m, aldt_m, 0.0)

    # Stress intensity factors (lines 138-139)
    kcm = k_ic * math.sqrt(fac_l)
    ktm = k_th * math.sqrt(fac_l)

    k1_a = ya * sig_eff * math.sqrt(math.pi * a_cur * _EM6 * fac_l)
    k1_c = yc * sig_eff * math.sqrt(math.pi * c_cur * _EM6 * fac_l)

    # Check rupture criterion 1 (lines 142-149)
    is_failed = False
    if k1_a >= kcm or k1_c >= kcm or a_cur >= thk_m:
        is_failed = True
        uvar[14] = 1.0  # element failed

    # Paris-Erdogan subcritical growth (lines 151-158)
    v_a = 0.0
    v_c = 0.0
    if not is_failed and kcm > 0.0:
        if ktm <= k1_a < kcm:
            v_a = v0 * ((k1_a / kcm) ** exp_n)
        if ktm <= k1_c and k1_a < kcm:
            v_c = v0 * ((k1_c / kcm) ** exp_n)

    fac_lenm = _EP06 * fac_l
    da = v_a * dt * fac_lenm
    dc = v_c * dt * fac_lenm

    new_a = a_cur + da
    new_c = c_cur + dc

    uvar[15] = new_c
    uvar[16] = new_a

    crack.a0 = new_a
    crack.c0 = new_c

    diag = {
        "K1_A": k1_a,
        "K1_C": k1_c,
        "Y_A": ya,
        "Y_C": yc,
        "sig_cos": sig_cos,
        "sig_eff": sig_eff,
        "V_A": v_a,
        "V_C": v_c,
        "da": da,
        "dc": dc,
    }
    return is_failed, new_a, new_c, diag


# ---------------------------------------------------------------------------
# Card Parsing & Builder Helpers
# ---------------------------------------------------------------------------

def build_inicrack(data: Any = None, **kwargs: Any) -> IniCrackData:
    """Construct an IniCrackData instance from dictionary, entity, or keyword arguments."""
    if isinstance(data, IniCrackData):
        return data

    d: Dict[str, Any] = {}
    if isinstance(data, dict):
        d.update(data)
    elif hasattr(data, "__dict__"):
        d.update(data.__dict__)

    d.update(kwargs)

    crack_id = int(d.get("id", d.get("ID", d.get("crack_id", 1))))
    title = str(d.get("title", d.get("TITLE", f"INICRACK_{crack_id}")))
    part_id = d.get("part_id", d.get("pid", None))
    if part_id is not None:
        part_id = int(part_id)
    element_id = d.get("element_id", d.get("eid", None))
    if element_id is not None:
        element_id = int(element_id)

    # Center location
    center = d.get("center", (d.get("x0", 0.0), d.get("y0", 0.0), d.get("z0", 0.0)))
    if isinstance(center, (list, np.ndarray)) and len(center) >= 3:
        center = (float(center[0]), float(center[1]), float(center[2]))

    # Dimensions
    a0 = float(d.get("a0", d.get("a", d.get("depth", d.get("CR_DEPTH", 100.0)))))
    c0 = float(d.get("c0", d.get("c", d.get("half_length", d.get("CR_LEN", 100.0)))))

    # Normal / orientation
    normal = d.get("normal", d.get("norm", (1.0, 0.0, 0.0)))
    if isinstance(normal, (list, np.ndarray)) and len(normal) >= 3:
        normal = (float(normal[0]), float(normal[1]), float(normal[2]))

    sigma_ini = float(d.get("sigma_ini", d.get("SIG_INI", d.get("prestress", 0.0))))
    angle = float(d.get("angle", d.get("CR_ANG", d.get("theta", 0.0))))

    p1 = tuple(d.get("p1", (0.0, 0.0, 0.0)))
    p2 = tuple(d.get("p2", (0.0, 0.0, 0.0)))
    segments = d.get("segments", [])
    open_flag = int(d.get("open_flag", 0))

    return IniCrackData(
        id=crack_id,
        part_id=part_id,
        element_id=element_id,
        title=title,
        center=center,
        a0=a0,
        c0=c0,
        normal=normal,
        sigma_ini=sigma_ini,
        angle=angle,
        p1=p1,
        p2=p2,
        segments=segments,
        open_flag=open_flag,
    )
