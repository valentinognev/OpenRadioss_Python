r"""
OpenRadioss /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D) — 3D Orthotropic Elastic Composite.

Upstream OpenRadioss Fortran source references:
- Engine constitutive kernel:
  $OR_SRC/engine/source/materials/mat/mat012/m12law.F
  Subroutine: M12LAW (referenced as sigeps12.F for LAW12)
- Starter card reader and property initialization:
  $OR_SRC/starter/source/materials/mat/mat012/hm_read_mat12.F
  Subroutine: HM_READ_MAT12
- Local fiber orientation transformations:
  engine/source/materials/mat/mat014/m14ama.F
  engine/source/materials/mat/mat014/m14gtf.F
  engine/source/materials/mat/mat014/m14ftg.F
- HyperMesh configuration:
  hm_cfg_files/config/CFG/radioss2020/MAT/3d_comp_12.cfg

Constitutive Model:
-------------------
LAW12 models 3D orthotropic elastic composite solid elements with optional fiber reinforcement:
1. 3D Orthotropic Elasticity (9 independent engineering constants):
   - Young's moduli: E11, E22, E33 (MAT_EA, MAT_EB, MAT_EC)
   - Poisson's ratios: nu12, nu23, nu31 (MAT_PRAB, MAT_PRBC, MAT_PRCA)
     Reciprocity relations: nu21 = nu12*E22/E11, nu32 = nu23*E33/E22, nu13 = nu31*E11/E33
   - Shear moduli: G12, G23, G31 (MAT_GAB, MAT_GBC, MAT_GCA)
2. Inversion of 3D compliance matrix to stiffness tensor D:
   - D11, D12, D13, D22, D23, D33, G12, G23, G31
   - Incremental Hooke's law in local orthotropic axes:
       Delta sigma_1 = D11*Deps_1 + D12*Deps_2 + D13*Deps_3
       Delta sigma_2 = D12*Deps_1 + D22*Deps_2 + D23*Deps_3
       Delta sigma_3 = D13*Deps_1 + D23*Deps_2 + D33*Deps_3
       Delta sigma_12 = G12*Deps_12
       Delta sigma_23 = G23*Deps_23
       Delta sigma_31 = G31*Deps_31
3. Fiber Composite Reinforcement:
   - Fiber volume fraction: alpha (MAT_ALPHA)
   - Fiber Young's modulus: Efib (MAT_EFIB)
   - Fiber strain and stress tracking: sigma_f = Efib * eps_f
4. Acoustic Sound Speed:
   - c = sqrt(max(D11, D22, D33) / rho0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30


def _inv(val: float) -> float:
    """Safely invert a stress value (returns 0.0 for zero or infinity)."""
    if val <= 0.0 or math.isinf(val) or val >= 1.0e29:
        return 0.0
    return 1.0 / val


def _inv_prod(v1: float, v2: float) -> float:
    """Safely compute 1 / (v1 * v2) for Tsai-Wu diagonal terms."""
    if (
        v1 <= 0.0
        or v2 <= 0.0
        or math.isinf(v1)
        or math.isinf(v2)
        or v1 >= 1.0e29
        or v2 >= 1.0e29
    ):
        return 0.0
    prod = v1 * v2
    if prod <= 0.0 or math.isinf(prod):
        return 0.0
    return 1.0 / prod


def _tw_cross(v1: float, v2: float, v3: float, v4: float) -> float:
    """Safely compute Tsai-Wu cross term: -0.5 / sqrt(v1 * v2 * v3 * v4)."""
    if any(v <= 0.0 or math.isinf(v) or v >= 1.0e29 for v in (v1, v2, v3, v4)):
        return 0.0
    prod = v1 * v2 * v3 * v4
    if prod <= 0.0 or math.isinf(prod):
        return 0.0
    return -0.5 / math.sqrt(prod)


# ============================================================================
# Coordinate Transformations (m14ama, m14gtf, m14ftg)
# ============================================================================

def m14ama(
    rx: np.ndarray,
    ry: np.ndarray,
    rz: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    a: Optional[np.ndarray] = None,
    jcvt: int = 0,
    jsph: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local orthotropic axes triad [A, B, C].

    Fortran origin: engine/source/materials/mat/mat014/m14ama.F
    """
    rx = np.atleast_1d(rx).astype(float)
    ry = np.atleast_1d(ry).astype(float)
    rz = np.atleast_1d(rz).astype(float)
    sx = np.atleast_1d(sx).astype(float)
    sy = np.atleast_1d(sy).astype(float)
    sz = np.atleast_1d(sz).astype(float)
    nel = rx.shape[0]

    if jsph == 1 and a is not None:
        ax = a[:, 0]
        ay = a[:, 1]
        az = a[:, 2]
        bx = a[:, 3]
        by = a[:, 4]
        bz = a[:, 5]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        return ax, ay, az, bx, by, bz, cx, cy, cz

    if jcvt == 0:
        rr = 1.0 / np.maximum(np.sqrt(rx**2 + ry**2 + rz**2), _EM20)
        rx_n = rx * rr
        ry_n = ry * rr
        rz_n = rz * rr

        tx = ry_n * sz - rz_n * sy
        ty = rz_n * sx - rx_n * sz
        tz = rx_n * sy - ry_n * sx
        rt = 1.0 / np.maximum(np.sqrt(tx**2 + ty**2 + tz**2), _EM20)
        tx_n = tx * rt
        ty_n = ty * rt
        tz_n = tz * rt

        sx_n = ty_n * rz_n - tz_n * ry_n
        sy_n = tz_n * rx_n - tx_n * rz_n
        sz_n = tx_n * ry_n - ty_n * rx_n
        rs = 1.0 / np.maximum(np.sqrt(sx_n**2 + sy_n**2 + sz_n**2), _EM20)
        sx_n *= rs
        sy_n *= rs
        sz_n *= rs

        if a is not None:
            ax = a[:, 0] * rx_n + a[:, 1] * sx_n + a[:, 2] * tx_n
            ay = a[:, 0] * ry_n + a[:, 1] * sy_n + a[:, 2] * ty_n
            az = a[:, 0] * rz_n + a[:, 1] * sz_n + a[:, 2] * tz_n

            bx = a[:, 3] * rx_n + a[:, 4] * sx_n + a[:, 5] * tx_n
            by = a[:, 3] * ry_n + a[:, 4] * sy_n + a[:, 5] * ty_n
            bz = a[:, 3] * rz_n + a[:, 4] * sz_n + a[:, 5] * tz_n
        else:
            ax, ay, az = rx_n, ry_n, rz_n
            bx, by, bz = sx_n, sy_n, sz_n

        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        return ax, ay, az, bx, by, bz, cx, cy, cz

    # Co-rotational default: identity axes
    ax = np.ones(nel, dtype=float)
    ay = np.zeros(nel, dtype=float)
    az = np.zeros(nel, dtype=float)
    bx = np.zeros(nel, dtype=float)
    by = np.ones(nel, dtype=float)
    bz = np.zeros(nel, dtype=float)
    cx = np.zeros(nel, dtype=float)
    cy = np.zeros(nel, dtype=float)
    cz = np.ones(nel, dtype=float)
    return ax, ay, az, bx, by, bz, cx, cy, cz


def m14gtf(
    sig: np.ndarray,
    d: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform global stresses and strain increments to material axes.

    Fortran origin: engine/source/materials/mat/mat014/m14gtf.F
    """
    sig = np.atleast_2d(sig).astype(float)
    d = np.atleast_2d(d).astype(float)
    nel = sig.shape[0]

    s1, s2, s3, s4, s5, s6 = (
        sig[:, 0],
        sig[:, 1],
        sig[:, 2],
        sig[:, 3],
        sig[:, 4],
        sig[:, 5],
    )
    d1, d2, d3, d4, d5, d6 = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]

    e1 = ax * ax * d1 + ay * ay * d2 + az * az * d3 + ax * ay * d4 + ay * az * d5 + az * ax * d6
    e2 = bx * bx * d1 + by * by * d2 + bz * bz * d3 + bx * by * d4 + by * bz * d5 + bz * bx * d6
    e3 = cx * cx * d1 + cy * cy * d2 + cz * cz * d3 + cx * cy * d4 + cy * cz * d5 + cz * cx * d6
    e4 = 2.0 * (
        ax * bx * d1
        + ay * by * d2
        + az * bz * d3
        + (ax * by + ay * bx) * d4 * 0.5
        + (ay * bz + az * by) * d5 * 0.5
        + (az * bx + ax * bz) * d6 * 0.5
    )
    e5 = 2.0 * (
        bx * cx * d1
        + by * cy * d2
        + bz * cz * d3
        + (bx * cy + by * cx) * d4 * 0.5
        + (by * cz + bz * cy) * d5 * 0.5
        + (bz * cx + bx * cz) * d6 * 0.5
    )
    e6 = 2.0 * (
        cx * ax * d1
        + cy * ay * d2
        + cz * az * d3
        + (cx * ay + cy * ax) * d4 * 0.5
        + (cy * az + cz * ay) * d5 * 0.5
        + (cz * ax + cx * az) * d6 * 0.5
    )

    t1 = ax * ax * s1 + ay * ay * s2 + az * az * s3 + 2.0 * ax * ay * s4 + 2.0 * ay * az * s5 + 2.0 * az * ax * s6
    t2 = bx * bx * s1 + by * by * s2 + bz * bz * s3 + 2.0 * bx * by * s4 + 2.0 * by * bz * s5 + 2.0 * bz * bx * s6
    t3 = cx * cx * s1 + cy * cy * s2 + cz * cz * s3 + 2.0 * cx * cy * s4 + 2.0 * cy * cz * s5 + 2.0 * cz * cx * s6
    t4 = ax * bx * s1 + ay * by * s2 + az * bz * s3 + (ax * by + ay * bx) * s4 + (ay * bz + az * by) * s5 + (az * bx + ax * bz) * s6
    t5 = bx * cx * s1 + by * cy * s2 + bz * cz * s3 + (bx * cy + by * cx) * s4 + (by * cz + bz * cy) * s5 + (bz * cx + bx * cz) * s6
    t6 = cx * ax * s1 + cy * ay * s2 + cz * az * s3 + (cx * ay + cy * ax) * s4 + (cy * az + cz * ay) * s5 + (cz * ax + cx * az) * s6

    T = np.column_stack([t1, t2, t3, t4, t5, t6])
    E = np.column_stack([e1, e2, e3, e4, e5, e6])
    return T, E


def m14ftg(
    T: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
) -> np.ndarray:
    """Transform material axes stresses to global coordinate system.

    Fortran origin: engine/source/materials/mat/mat014/m14ftg.F
    """
    T = np.atleast_2d(T).astype(float)
    t1, t2, t3, t4, t5, t6 = T[:, 0], T[:, 1], T[:, 2], T[:, 3], T[:, 4], T[:, 5]

    s1 = ax * ax * t1 + bx * bx * t2 + cx * cx * t3 + 2.0 * ax * bx * t4 + 2.0 * bx * cx * t5 + 2.0 * cx * ax * t6
    s2 = ay * ay * t1 + by * by * t2 + cy * cy * t3 + 2.0 * ay * by * t4 + 2.0 * by * cy * t5 + 2.0 * cy * ay * t6
    s3 = az * az * t1 + bz * bz * t2 + cz * cz * t3 + 2.0 * az * bz * t4 + 2.0 * bz * cz * t5 + 2.0 * cz * az * t6
    s4 = ax * ay * t1 + bx * by * t2 + cx * cy * t3 + (ax * by + bx * ay) * t4 + (bx * cy + cx * by) * t5 + (cx * ay + ax * cy) * t6
    s5 = ay * az * t1 + by * bz * t2 + cy * cz * t3 + (ay * bz + by * az) * t4 + (by * cz + cy * bz) * t5 + (cy * az + ay * cz) * t6
    s6 = az * ax * t1 + bz * bx * t2 + cz * cx * t3 + (az * bx + bz * ax) * t4 + (bz * cx + cz * bx) * t5 + (cz * ax + az * cx) * t6

    return np.column_stack([s1, s2, s3, s4, s5, s6])


# ============================================================================
# Material Builder (hm_read_mat12.F)
# ============================================================================

def build_law12(rec: Any = None, **kwargs: Any) -> Material:
    """Construct /MAT/LAW12 (/MAT/3D_COMP) material with derived constants.

    Fortran source: starter/source/materials/mat/mat012/hm_read_mat12.F
    CFG: hm_cfg_files/config/CFG/radioss2020/MAT/3d_comp_12.cfg
    """
    if rec is None:
        p_in: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        _title = str(kwargs.get("title", "LAW12_3D_COMP"))
    elif isinstance(rec, dict):
        base_params = rec.get("params", rec)
        p_in = {**base_params, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        _title = str(rec.get("title", kwargs.get("title", "LAW12_3D_COMP")))
    elif hasattr(rec, "params"):
        p_in = {**rec.params, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW12_3D_COMP")))
    elif hasattr(rec, "__dataclass_fields__"):
        base_dict = {
            k: getattr(rec, k) for k in rec.__dataclass_fields__ if hasattr(rec, k)
        }
        p_in = {**base_dict, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW12_3D_COMP")))
    else:
        p_in = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        _title = str(kwargs.get("title", "LAW12_3D_COMP"))

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in p_in and p_in[k] is not None:
                try:
                    return float(p_in[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _geti(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            if k in p_in and p_in[k] is not None:
                try:
                    return int(p_in[k])
                except (ValueError, TypeError):
                    pass
        return int(default)

    # Card 1: rho0, refer_rho
    rho0 = _get(["MAT_RHO", "rho0", "density", "rho"], 0.0)
    refer_rho = _get(["Refer_Rho", "refer_rho", "RHO_O", "rhor", "rho_ref"], 0.0)
    if refer_rho <= 0.0:
        refer_rho = rho0

    # Card 2: E11, E22, E33
    e11 = _get(["MAT_EA", "E11", "e11", "ea", "MAT_E11"], 0.0)
    e22 = _get(["MAT_EB", "E22", "e22", "eb", "MAT_E22"], 0.0)
    e33 = _get(["MAT_EC", "E33", "e33", "ec", "MAT_E33"], 0.0)
    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0:
        raise ValueError(
            f"E11, E22, and E33 must be > 0 (got E11={e11}, E22={e22}, E33={e33})"
        )

    # Card 3: nu12, nu23, nu31
    nu12 = _get(["MAT_PRAB", "nu12", "NU12", "prab", "MAT_NU12"], 0.0)
    nu23 = _get(["MAT_PRBC", "nu23", "NU23", "prbc", "MAT_NU23"], 0.0)
    nu31 = _get(["MAT_PRCA", "nu31", "NU31", "prca", "MAT_NU31"], 0.0)

    # Card 4: G12, G23, G31
    g12 = _get(["MAT_GAB", "G12", "g12", "gab", "MAT_G12"], 0.0)
    g23 = _get(["MAT_GBC", "G23", "g23", "gbc", "MAT_G23"], 0.0)
    g31 = _get(["MAT_GCA", "G31", "g31", "gca", "MAT_G31"], 0.0)

    # Card 5: sigt1, sigt2, sigt3, delta
    sigt1 = _get(["MAT_SIGT1", "sigt1", "sig_t1", "sigma_t1"], 0.0)
    sigt2 = _get(["MAT_SIGT2", "sigt2", "sig_t2", "sigma_t2"], 0.0)
    sigt3 = _get(["MAT_SIGT3", "sigt3", "sig_t3", "sigma_t3"], 0.0)
    delta = _get(["MAT_DAMAGE", "delta", "damage", "DELTA"], 0.05)

    if sigt1 <= 0.0:
        sigt1 = _INF
    if sigt2 <= 0.0:
        sigt2 = sigt1
    if sigt3 <= 0.0:
        sigt3 = sigt1
    if delta <= 0.0:
        delta = 0.05

    # Card 6: cb (b), cn (n), fmax, wplaref
    cb = _get(["MAT_BETA", "cb", "b", "B", "beta"], 0.0)
    cn = _get(["MAT_HARD", "cn", "n", "N", "hard"], 1.0)
    if cn <= 0.0:
        cn = 1.0
    fmax = _get(["MAT_SIG", "fmax", "FMAX", "sig"], 1.0e10)
    if fmax <= 0.0:
        fmax = 1.0e10
    fmax = max(1.0001, fmax)
    wplaref = _get(["WPREF", "wplaref", "wpref", "WPLAREF"], 1.0)
    if wplaref <= 0.0:
        wplaref = 1.0

    # Card 7: sigyt1, sigyt2, sigyc1, sigyc2
    sigyt1 = _get(["MAT_SIGYT1", "sigyt1", "sig_1yt", "sigma_1yt"], 0.0)
    sigyt2 = _get(["MAT_SIGYT2", "sigyt2", "sig_2yt", "sigma_2yt"], 0.0)
    sigyc1 = _get(["MAT_SIGYC1", "sigyc1", "sig_1yc", "sigma_1yc"], 0.0)
    sigyc2 = _get(["MAT_SIGYC2", "sigyc2", "sig_2yc", "sigma_2yc"], 0.0)

    if sigyt1 <= 0.0:
        sigyt1 = _INF
    if sigyc1 <= 0.0:
        sigyc1 = sigyt1
    if sigyt2 <= 0.0:
        sigyt2 = sigyt1
    if sigyc2 <= 0.0:
        sigyc2 = sigyc1

    # Card 8: sigyt12, sigyc12, sigyt23, sigyc23
    sigyt12 = _get(["MAT_SIGT12", "sigyt12", "sigt12", "sig_12yt", "sigma_12yt"], 0.0)
    sigyc12 = _get(["MAT_SIGC12", "sigyc12", "sigc12", "sig_12yc", "sigma_12yc"], 0.0)
    sigyt23 = _get(["MAT_SIGT23", "sigyt23", "sigt23", "sig_23yt", "sigma_23yt"], 0.0)
    sigyc23 = _get(["MAT_SIGC23", "sigyc23", "sigc23", "sig_23yc", "sigma_23yc"], 0.0)

    if sigyt12 <= 0.0:
        sigyt12 = _INF
    if sigyc12 <= 0.0:
        sigyc12 = _INF
    if sigyt23 <= 0.0:
        sigyt23 = _INF
    if sigyc23 <= 0.0:
        sigyc23 = _INF

    # Card 9: sigyt3, sigyc3, sigyt13, sigyc13
    sigyt3 = _get(["MAT_SIGYT3", "sigyt3", "sig_3yt", "sigma_3yt"], 0.0)
    sigyc3 = _get(["MAT_SIGYC3", "sigyc3", "sig_3yc", "sigma_3yc"], 0.0)
    sigyt13 = _get(["MAT_SIGYT13", "sigyt13", "sigt13", "sig_13yt", "sigma_13yt"], 0.0)
    sigyc13 = _get(["MAT_SIGYC13", "sigyc13", "sigc13", "sig_13yc", "sigma_13yc"], 0.0)

    if sigyt3 <= 0.0:
        sigyt3 = sigyt2
    if sigyc3 <= 0.0:
        sigyc3 = sigyc2
    if sigyt13 <= 0.0:
        sigyt13 = sigyt12
    if sigyc13 <= 0.0:
        sigyc13 = sigyc12

    # Card 10: alpha, efib, c, eps0, ICC
    alpha = _get(["MAT_ALPHA", "alpha", "ALPHA"], 0.0)
    if alpha >= 1.0:
        alpha = 0.99
    efib = _get(["MAT_EFIB", "efib", "EFIB", "ef", "Ef"], 0.0)
    c_rate = _get(["MAT_SRC", "c", "C", "src"], 0.0)
    eps0 = _get(["MAT_SRP", "eps0", "srp", "eps_rate_0", "EPS_RATE_0"], 1.0)
    if eps0 <= 0.0 or c_rate == 0.0:
        eps0 = 1.0
    icc = _geti(["STRFLAG", "icc", "ICC", "strflag"], 1)
    if icc <= 0:
        icc = 1

    # Orthotropic compliance matrix inversion (hm_read_mat12.F:221-248)
    c11 = 1.0 / e11
    c22 = 1.0 / e22
    c33 = 1.0 / e33
    c12 = -nu12 / e11
    c13 = -nu31 / e33
    c23 = -nu23 / e22

    detc = (
        c11 * c22 * c33
        - c11 * (c23**2)
        - (c12**2) * c33
        + 2.0 * c12 * c13 * c23
        - (c13**2) * c22
    )
    if detc <= 0.0:
        raise ValueError(f"Orthotropic compliance determinant DETC={detc} <= 0")

    d11 = (c22 * c33 - c23**2) / detc
    d12 = -(c12 * c33 - c13 * c23) / detc
    d13 = (c12 * c23 - c13 * c22) / detc
    d22 = (c11 * c33 - c13**2) / detc
    d23 = -(c11 * c23 - c13 * c12) / detc
    d33 = (c11 * c22 - c12**2) / detc
    d21, d31, d32 = d12, d13, d23

    # Verification: A = C @ D = I (hm_read_mat12.F:250-255)
    a11 = c11 * d11 + c12 * d21 + c13 * d31
    a12 = c11 * d12 + c12 * d22 + c13 * d32
    a13 = c11 * d13 + c12 * d23 + c13 * d33
    a22 = c12 * d12 + c22 * d22 + c23 * d23
    a23 = c12 * d13 + c22 * d23 + c23 * d33
    a33 = c13 * d13 + c23 * d23 + c33 * d33

    # Tsai-Wu yield constants (hm_read_mat12.F:291-306)
    f1 = _inv(sigyt1) - _inv(sigyc1)
    f2 = _inv(sigyt2) - _inv(sigyc2)
    f3 = _inv(sigyt3) - _inv(sigyc3)
    f4 = _inv(sigyt12) - _inv(sigyc12)
    f5 = _inv(sigyt23) - _inv(sigyc23)
    f6 = _inv(sigyt13) - _inv(sigyc13)

    f11 = _inv_prod(sigyt1, sigyc1)
    f22 = _inv_prod(sigyt2, sigyc2)
    f33 = _inv_prod(sigyt3, sigyc3)
    f44 = _inv_prod(sigyt12, sigyc12)
    f55 = _inv_prod(sigyt23, sigyc23)
    f66 = _inv_prod(sigyt13, sigyc13)

    f12 = _tw_cross(sigyt1, sigyc1, sigyt2, sigyc2)
    f23 = _tw_cross(sigyt2, sigyc2, sigyt3, sigyc3)
    f13 = _tw_cross(sigyt1, sigyc1, sigyt3, sigyc3)

    # Verification constants for yield surface (hm_read_mat12.F:306-307)
    ft1 = f11 * f22 - 4.0 * (f12**2)
    ft2 = (f22**2) - 4.0 * (f23**2)

    # Sound speed (hm_read_mat12.F:265-266)
    c1_stiff = max(d11, d22, d33)
    r_ref = refer_rho if refer_rho > 0.0 else rho0
    ssp = math.sqrt(c1_stiff / max(r_ref, 1.0e-20))

    # Time step parameter for solid elements (hm_read_mat12.F:351-352)
    dmin = min(d11 * d22 - d12**2, d22 * d33 - d23**2, d11 * d33 - d13**2)
    pm105 = dmin / (c1_stiff**2) if c1_stiff > 0.0 else 0.0

    params: Dict[str, Any] = {
        "rho0": rho0,
        "refer_rho": refer_rho,
        "rhor": refer_rho,
        "density": rho0,
        "MAT_RHO": rho0,
        # Orthotropic moduli
        "E11": e11,
        "E22": e22,
        "E33": e33,
        "MAT_EA": e11,
        "MAT_EB": e22,
        "MAT_EC": e33,
        "nu12": nu12,
        "nu23": nu23,
        "nu31": nu31,
        "MAT_PRAB": nu12,
        "MAT_PRBC": nu23,
        "MAT_PRCA": nu31,
        "G12": g12,
        "G23": g23,
        "G31": g31,
        "MAT_GAB": g12,
        "MAT_GBC": g23,
        "MAT_GCA": g31,
        # Tensile damage
        "sigt1": sigt1,
        "sigt2": sigt2,
        "sigt3": sigt3,
        "delta": delta,
        "MAT_SIGT1": sigt1,
        "MAT_SIGT2": sigt2,
        "MAT_SIGT3": sigt3,
        "MAT_DAMAGE": delta,
        # Hardening
        "cb": cb,
        "cn": cn,
        "fmax": fmax,
        "wplaref": wplaref,
        "MAT_BETA": cb,
        "MAT_HARD": cn,
        "MAT_SIG": fmax,
        "WPREF": wplaref,
        # Yield strengths
        "sigyt1": sigyt1,
        "sigyt2": sigyt2,
        "sigyc1": sigyc1,
        "sigyc2": sigyc2,
        "sigyt12": sigyt12,
        "sigyc12": sigyc12,
        "sigyt23": sigyt23,
        "sigyc23": sigyc23,
        "sigyt3": sigyt3,
        "sigyc3": sigyc3,
        "sigyt13": sigyt13,
        "sigyc13": sigyc13,
        "MAT_SIGYT1": sigyt1,
        "MAT_SIGYT2": sigyt2,
        "MAT_SIGYC1": sigyc1,
        "MAT_SIGYC2": sigyc2,
        "MAT_SIGT12": sigyt12,
        "MAT_SIGC12": sigyc12,
        "MAT_SIGT23": sigyt23,
        "MAT_SIGC23": sigyc23,
        "MAT_SIGYT3": sigyt3,
        "MAT_SIGYC3": sigyc3,
        "MAT_SIGYT13": sigyt13,
        "MAT_SIGYC13": sigyc13,
        # Fibers & rate
        "alpha": alpha,
        "efib": efib,
        "c": c_rate,
        "eps0": eps0,
        "ICC": icc,
        "MAT_ALPHA": alpha,
        "MAT_EFIB": efib,
        "MAT_SRC": c_rate,
        "MAT_SRP": eps0,
        "STRFLAG": icc,
        # Compliance & Stiffness
        "C11": c11,
        "C22": c22,
        "C33": c33,
        "C12": c12,
        "C13": c13,
        "C23": c23,
        "DETC": detc,
        "D11": d11,
        "D12": d12,
        "D13": d13,
        "D22": d22,
        "D23": d23,
        "D33": d33,
        "D21": d21,
        "D31": d31,
        "D32": d32,
        # Verification of C @ D = I (hm_read_mat12.F:250-255)
        "A11": a11,
        "A12": a12,
        "A13": a13,
        "A22": a22,
        "A23": a23,
        "A33": a33,
        # Tsai-Wu coefficients
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "F5": f5,
        "F6": f6,
        "F11": f11,
        "F22": f22,
        "F33": f33,
        "F44": f44,
        "F55": f55,
        "F66": f66,
        "F12": f12,
        "F23": f23,
        "F13": f13,
        "FT1": ft1,
        "FT2": ft2,
        # Sound speed & equivalents
        "C1": c1_stiff,
        "SSP": ssp,
        "ssp": ssp,
        "c_sound": ssp,
        "DMIN": dmin,
        "PM105": pm105,
        "E": max(e11, e22, e33),
        "MAT_E": max(e11, e22, e33),
        "nu": (nu12 + nu23 + nu31) / 3.0,
        "MAT_NU": (nu12 + nu23 + nu31) / 3.0,
        "G": (g12 + g23 + g31) / 3.0,
        "MAT_G": (g12 + g23 + g31) / 3.0,
        "b": cb,
        "n": cn,
    }

    return Material(
        id=_id,
        law=12,
        law_name="LAW12",
        rho0=rho0,
        title=_title,
        params=params,
    )


# ============================================================================
# Auxiliary Functions: extra_shapes & sound_speed
# ============================================================================

def extra_shapes(mat: Any = None, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """State variable shapes for LAW12 3D composite.

    dam12: (5,) [dam1, dam2, dam3, WVEC, flags]
    epe12: (3,) [epe1, epe2, epe3] total strains
    epc12: (3,) [epc1, epc2, epc3] crack strains
    wpla12: () plastic work
    off12: () failure degradation factor
    epsf12: () fiber strain
    sigf12: () fiber stress
    """
    if nip is not None and nip > 1:
        return {
            "dam12": (nip, 5),
            "epe12": (nip, 3),
            "epc12": (nip, 3),
            "wpla12": (nip,),
            "off12": (nip,),
            "epsf12": (nip,),
            "sigf12": (nip,),
            "tsaiwu12": (nip,),
        }
    return {
        "dam12": (5,),
        "epe12": (3,),
        "epc12": (3,),
        "wpla12": (),
        "off12": (),
        "epsf12": (),
        "sigf12": (),
        "tsaiwu12": (),
    }


def sound_speed(
    mat: Any,
    rho: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute Courant sound speed for LAW12.

    Fortran: SSP = SQRT(MAX(D11, D22, D33) / RHO)
    """
    p = getattr(mat, "params", {}) or {}
    d11 = float(p.get("D11", 0.0))
    d22 = float(p.get("D22", 0.0))
    d33 = float(p.get("D33", 0.0))
    c1 = max(d11, d22, d33)
    if c1 <= 0.0:
        c1 = float(p.get("C1", max(float(p.get("E11", 1.0)), 1.0)))
    r = (
        rho
        if (rho is not None and rho > 0.0)
        else float(getattr(mat, "rho0", 0.0) or p.get("rho0", 1.0))
    )
    return math.sqrt(c1 / max(r, 1.0e-20))


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Raise error as LAW12 is solid-only."""
    raise NotImplementedError(
        "/MAT/LAW12 (3D_COMP) is implemented for 3D solid elements only."
    )


# ============================================================================
# Core Element Constitutive Integration (m12law.F)
# ============================================================================

def _update_one_element(
    p: Dict[str, Any],
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float,
    dt: float,
    dam: np.ndarray,
    epe: np.ndarray,
    epc: np.ndarray,
    wpla: float,
    off: float,
    epsf: float,
    sigf: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Constitutive integration for a single 3D element following m12law.F.

    Returns:
      (sig_out, epsp_out, dam_out, epe_out, epc_out, wpla_out, off_out, epsf_out, sigf_out)
    """
    # 1. Decode damage history flags (m12law.F:153-162)
    dam5 = float(dam[4])
    if dam5 >= 10000.0:
        idam = int(dam5) - 10000
        kd1 = idam // 1000
        kdx = idam - kd1 * 1000
        kd2 = kdx // 100
        kdx = kdx - kd2 * 100
        kd3 = kdx // 10
        kd4 = kdx - kd3 * 10
    else:
        kd1, kd2, kd3, kd4 = 0, 0, 0, 0

    d11 = float(p["D11"])
    d12 = float(p["D12"])
    d13 = float(p["D13"])
    d22 = float(p["D22"])
    d23 = float(p["D23"])
    d33 = float(p["D33"])
    g12 = float(p["G12"])
    g23 = float(p["G23"])
    g31 = float(p["G31"])

    sigt1 = float(p["sigt1"])
    sigt2 = float(p["sigt2"])
    sigt3 = float(p["sigt3"])
    delta = float(p["delta"])

    cb = float(p["cb"])
    cn = float(p["cn"])
    fmax = float(p["fmax"])
    wplaref = float(p["wplaref"])

    f1 = float(p["F1"])
    f2 = float(p["F2"])
    f3 = float(p["F3"])
    f4 = float(p["F4"])
    f5 = float(p["F5"])
    f6 = float(p["F6"])
    f11 = float(p["F11"])
    f22 = float(p["F22"])
    f33 = float(p["F33"])
    f44 = float(p["F44"])
    f55 = float(p["F55"])
    f66 = float(p["F66"])
    f12 = float(p["F12"])
    f23 = float(p["F23"])
    f13 = float(p["F13"])

    alpha = float(p["alpha"])
    efib = float(p["efib"])
    c_rate = float(p["c"])
    eps0 = float(p["eps0"])
    icc = int(p["ICC"])

    deps1 = float(deps[0])
    deps2 = float(deps[1])
    deps3 = float(deps[2])
    deps4 = float(deps[3])
    deps5 = float(deps[4])
    deps6 = float(deps[5])

    # 2. Strain rate & rate enhancement (m12law.F:208-228)
    if dt > 0.0:
        d_rate1 = deps1 / dt
        d_rate2 = deps2 / dt
        d_rate3 = deps3 / dt
        d_rate4 = deps4 / dt
        d_rate5 = deps5 / dt
        d_rate6 = deps6 / dt
        epsp_rate = max(
            abs(d_rate1),
            abs(d_rate2),
            abs(d_rate3),
            0.5 * abs(d_rate4),
            0.5 * abs(d_rate5),
            0.5 * abs(d_rate6),
        )
    else:
        epsp_rate = 0.0

    if epsp_rate > eps0 and c_rate > 0.0 and eps0 > 0.0:
        rate_fac = 1.0 + c_rate * math.log(epsp_rate / eps0)
    else:
        rate_fac = 1.0

    if icc in (1, 3):
        sigmx = fmax * rate_fac
    else:
        sigmx = fmax

    cb_eff = cb * rate_fac
    ca_eff = 1.0 * rate_fac
    wpla_term = (wpla**cn) if wpla > 0.0 else 0.0
    sigmy = min(sigmx, ca_eff + cb_eff * wpla_term)

    if sigmy >= sigmx and off == 1.0:
        off = 0.99
        kd4 = 2

    # 3. Total strain in crack directions (m12law.F:260-263)
    epe1 = float(epe[0]) + deps1
    epe2 = float(epe[1]) + deps2
    epe3 = float(epe[2]) + deps3

    # 4. Old stress and elastic trial stresses (m12law.F:338-354)
    so1 = float(sig[0])
    so2 = float(sig[1])
    so3 = float(sig[2])
    so4 = float(sig[3])
    so5 = float(sig[4])
    so6 = float(sig[5])

    t1 = so1 + d11 * deps1 + d12 * deps2 + d13 * deps3
    t2 = so2 + d12 * deps1 + d22 * deps2 + d23 * deps3
    t3 = so3 + d13 * deps1 + d23 * deps2 + d33 * deps3
    t4 = so4 + g12 * deps4
    t5 = so5 + g23 * deps5
    t6 = so6 + g31 * deps6

    # 5. Fibers stress (m12law.F:399-407)
    if alpha > 0.0:
        epsf += deps1
        sigf = efib * epsf

    # 6. General failure degradation (m12law.F:415-418)
    if off < 0.1:
        off = 0.0
    elif off < 1.0:
        off = off * 0.8

    # 7. Tensile damage in directions 1, 2, 3 (m12law.F:422-501)
    dam1 = float(dam[0])
    dam2 = float(dam[1])
    dam3 = float(dam[2])

    epc1 = float(epc[0])
    epc2 = float(epc[1])
    epc3 = float(epc[2])

    # Direction 1
    wvec1 = (1.0 - dam1) * sigt1
    if t1 > wvec1:
        if epc1 == 0.0:
            epc1 = max(epe1, 0.0)
        else:
            epc1 = max(epc1 + deps1, 0.0)
        t1 = wvec1
        t2 -= d12 * deps1 * dam1
        t3 -= d13 * deps1 * dam1
        if kd1 == 0:
            kd1 = 1
        dam1 = min(dam1 + delta, 1.0)
        if dam1 >= 1.0 and kd1 != 2:
            kd1 = 2

    if deps1 < 0.0 and dam1 > 0.0:
        epc1 = max(epc1 + deps1, 0.0)

    # Direction 2
    wvec2 = (1.0 - dam2) * sigt2
    if t2 > wvec2:
        if epc2 == 0.0:
            epc2 = max(epe2, 0.0)
        else:
            epc2 = max(epc2 + deps2, 0.0)
        t1 -= d12 * deps2 * dam2
        t2 = wvec2
        t3 -= d23 * deps2 * dam2
        if kd2 == 0:
            kd2 = 1
        dam2 = min(dam2 + delta, 1.0)
        if dam2 >= 1.0 and kd2 != 2:
            kd2 = 2

    if deps2 < 0.0 and dam2 > 0.0:
        epc2 = max(epc2 + deps2, 0.0)

    # Direction 3
    wvec3 = (1.0 - dam3) * sigt3
    if t3 > wvec3:
        if epc3 == 0.0:
            epc3 = max(epe3, 0.0)
        else:
            epc3 = max(epc3 + deps3, 0.0)
        t1 -= d13 * deps3 * dam3
        t2 -= d23 * deps3 * dam3
        t3 = wvec3
        if kd3 == 0:
            kd3 = 1
        dam3 = min(dam3 + delta, 1.0)
        if dam3 >= 1.0 and kd3 != 2:
            kd3 = 2

    if deps3 < 0.0 and dam3 > 0.0:
        epc3 = max(epc3 + deps3, 0.0)

    # 8. Crack open condition: no compression across open crack (m12law.F:506-522)
    if t1 < 0.0 and epc1 > 0.0:
        t1 = 0.0
        t2 -= d12 * deps1 * dam1
        t3 -= d13 * deps1 * dam1

    if t2 < 0.0 and epc2 > 0.0:
        t1 -= d12 * deps2 * dam2
        t2 = 0.0
        t3 -= d23 * deps2 * dam2

    if t3 < 0.0 and epc3 > 0.0:
        t3 = 0.0
        t1 -= d13 * deps3 * dam3
        t2 -= d23 * deps3 * dam3

    # 9. Tsai-Wu 3D plastic return (m12law.F:526-636)
    wvec_tw = (
        f1 * t1
        + f2 * t2
        + f3 * t3
        + f11 * (t1**2)
        + f22 * (t2**2)
        + f33 * (t3**2)
        + f44 * (t4**2)
        + f55 * (t5**2)
        + f66 * (t6**2)
        + 2.0 * f12 * t1 * t2
        + 2.0 * f13 * t1 * t3
        + 2.0 * f23 * t2 * t3
    )
    dam4 = wvec_tw

    if wvec_tw > sigmy and off == 1.0:
        if kd4 == 0:
            kd4 = 1

        dp1 = f1 + 2.0 * f11 * so1 + 2.0 * f12 * so2 + 2.0 * f13 * so3
        dp2 = f2 + 2.0 * f22 * so2 + 2.0 * f12 * so1 + 2.0 * f23 * so3
        dp3 = f3 + 2.0 * f33 * so3 + 2.0 * f13 * so1 + 2.0 * f23 * so2
        dp4 = 2.0 * f44 * so4
        dp5 = 2.0 * f55 * so5
        dp6 = 2.0 * f66 * so6

        ds1 = t1 - so1
        ds2 = t2 - so2
        ds3 = t3 - so3
        ds4 = t4 - so4
        ds5 = t5 - so5
        ds6 = t6 - so6

        num = (
            dp1 * ds1
            + dp2 * ds2
            + dp3 * ds3
            + dp4 * ds4
            + dp5 * ds5
            + dp6 * ds6
        )

        plas = (wpla ** (cn - 1.0)) if wpla > 0.0 else 1.0

        denom = (
            dp1 * (d11 * dp1 + d12 * dp2 + d13 * dp3)
            + dp2 * (d12 * dp1 + d22 * dp2 + d23 * dp3)
            + dp3 * (d13 * dp1 + d23 * dp2 + d33 * dp3)
            + 2.0 * dp4 * g12 * dp4
            + 2.0 * dp5 * g23 * dp5
            + 2.0 * dp6 * g31 * dp6
            + (
                so1 * dp1
                + so2 * dp2
                + so3 * dp3
                + 2.0 * so4 * dp4
                + 2.0 * so5 * dp5
                + 2.0 * so6 * dp6
            )
            * cn
            * cb_eff
            * plas
        )

        if denom != 0.0 and num > 0.0:
            lamda = num / denom
        else:
            lamda = 0.0

        if lamda != 0.0:
            l_dp1 = lamda * dp1
            l_dp2 = lamda * dp2
            l_dp3 = lamda * dp3
            l_dp4 = lamda * dp4
            l_dp5 = lamda * dp5
            l_dp6 = lamda * dp6

            epe1 -= l_dp1
            epe2 -= l_dp2
            epe3 -= l_dp3

            t1 -= d11 * l_dp1 + d12 * l_dp2 + d13 * l_dp3
            t2 -= d12 * l_dp1 + d22 * l_dp2 + d23 * l_dp3
            t3 -= d13 * l_dp1 + d23 * l_dp2 + d33 * l_dp3
            t4 -= 2.0 * g12 * l_dp4
            t5 -= 2.0 * g23 * l_dp5
            t6 -= 2.0 * g31 * l_dp6

            dwpla = 0.5 * (
                l_dp1 * (t1 + so1)
                + l_dp2 * (t2 + so2)
                + l_dp3 * (t3 + so3)
                + 2.0 * l_dp4 * (t4 + so4)
                + 2.0 * l_dp5 * (t5 + so5)
                + 2.0 * l_dp6 * (t6 + so6)
            )
            dwpla = max(dwpla, 0.0) / wplaref
            wpla = max(wpla + dwpla, 0.0)

    # 10. Stresses scaled by off (m12law.F:648-654)
    s_out = np.array(
        [
            t1 * off,
            t2 * off,
            t3 * off,
            t4 * off,
            t5 * off,
            t6 * off,
        ],
        dtype=float,
    )

    dam5 = float(kd1 * 1000 + kd2 * 100 + kd3 * 10 + kd4 + 10000)
    dam_out = np.array([dam1, dam2, dam3, dam4, dam5], dtype=float)
    epe_out = np.array([epe1, epe2, epe3], dtype=float)
    epc_out = np.array([epc1, epc2, epc3], dtype=float)

    return s_out, wpla, dam_out, epe_out, epc_out, wpla, off, epsf, sigf


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """3D solid constitutive update for LAW12 (/MAT/3D_COMP).

    Fortran origin: engine/source/materials/mat/mat012/m12law.F
    """
    p = getattr(mat, "params", {}) or {}
    is_1d = (sig.ndim == 1)
    s_in = np.atleast_2d(sig).copy()
    d_in = np.atleast_2d(deps).copy()
    n = s_in.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=float)
    else:
        ep = np.atleast_1d(epsp).astype(float).copy()
        if len(ep) == 1 and n > 1:
            ep = np.full(n, ep[0], dtype=float)

    if extra is None:
        extra = {}

    # Extract/initialize state arrays
    dam = extra.get("dam12", extra.get("dam"))
    if dam is None:
        dam = np.zeros((n, 5), dtype=float)
    else:
        dam = np.atleast_2d(dam).astype(float).copy()
        if dam.shape[0] != n:
            dam = np.zeros((n, 5), dtype=float)

    epe = extra.get("epe12", extra.get("epe"))
    if epe is None:
        epe = np.zeros((n, 3), dtype=float)
    else:
        epe = np.atleast_2d(epe).astype(float).copy()
        if epe.shape[0] != n:
            epe = np.zeros((n, 3), dtype=float)

    epc = extra.get("epc12", extra.get("epc"))
    if epc is None:
        epc = np.zeros((n, 3), dtype=float)
    else:
        epc = np.atleast_2d(epc).astype(float).copy()
        if epc.shape[0] != n:
            epc = np.zeros((n, 3), dtype=float)

    wpla = extra.get("wpla12", extra.get("wpla"))
    if wpla is None:
        wpla = np.zeros(n, dtype=float)
    else:
        wpla = np.atleast_1d(wpla).astype(float).copy()
        if len(wpla) == 1 and n > 1:
            wpla = np.full(n, wpla[0], dtype=float)

    off = extra.get("off12", extra.get("off"))
    if off is None:
        off = np.ones(n, dtype=float)
    else:
        off = np.atleast_1d(off).astype(float).copy()
        if len(off) == 1 and n > 1:
            off = np.full(n, off[0], dtype=float)

    epsf = extra.get("epsf12", extra.get("epsf"))
    if epsf is None:
        epsf = np.zeros(n, dtype=float)
    else:
        epsf = np.atleast_1d(epsf).astype(float).copy()
        if len(epsf) == 1 and n > 1:
            epsf = np.full(n, epsf[0], dtype=float)

    sigf = extra.get("sigf12", extra.get("sigf"))
    if sigf is None:
        sigf = np.zeros(n, dtype=float)
    else:
        sigf = np.atleast_1d(sigf).astype(float).copy()
        if len(sigf) == 1 and n > 1:
            sigf = np.full(n, sigf[0], dtype=float)

    tsaiwu = extra.get("tsaiwu12", extra.get("tsaiwu"))
    if tsaiwu is None:
        tsaiwu = np.zeros(n, dtype=float)
    else:
        tsaiwu = np.atleast_1d(tsaiwu).astype(float).copy()
        if len(tsaiwu) == 1 and n > 1:
            tsaiwu = np.full(n, tsaiwu[0], dtype=float)

    # Optional local orthotropic coordinate system transformation
    axes = extra.get("axes", extra.get("frame", extra.get("A")))
    has_rot = False
    if axes is not None:
        axes_arr = np.asarray(axes, dtype=float)
        if axes_arr.ndim == 2 and axes_arr.shape == (n, 9):
            has_rot = True
            ax, ay, az = axes_arr[:, 0], axes_arr[:, 1], axes_arr[:, 2]
            bx, by, bz = axes_arr[:, 3], axes_arr[:, 4], axes_arr[:, 5]
            cx, cy, cz = axes_arr[:, 6], axes_arr[:, 7], axes_arr[:, 8]
        elif axes_arr.ndim == 3 and axes_arr.shape == (n, 3, 3):
            has_rot = True
            ax, ay, az = axes_arr[:, 0, 0], axes_arr[:, 0, 1], axes_arr[:, 0, 2]
            bx, by, bz = axes_arr[:, 1, 0], axes_arr[:, 1, 1], axes_arr[:, 1, 2]
            cx, cy, cz = axes_arr[:, 2, 0], axes_arr[:, 2, 1], axes_arr[:, 2, 2]
        elif axes_arr.ndim == 2 and axes_arr.shape == (3, 3):
            has_rot = True
            ax = np.full(n, axes_arr[0, 0])
            ay = np.full(n, axes_arr[0, 1])
            az = np.full(n, axes_arr[0, 2])
            bx = np.full(n, axes_arr[1, 0])
            by = np.full(n, axes_arr[1, 1])
            bz = np.full(n, axes_arr[1, 2])
            cx = np.full(n, axes_arr[2, 0])
            cy = np.full(n, axes_arr[2, 1])
            cz = np.full(n, axes_arr[2, 2])

    if has_rot:
        s_mat, d_mat = m14gtf(s_in, d_in, ax, ay, az, bx, by, bz, cx, cy, cz)
    else:
        s_mat, d_mat = s_in, d_in

    s_out = np.zeros_like(s_mat)
    ep_out = np.zeros(n, dtype=float)

    for i in range(n):
        (
            s_i,
            ep_i,
            dam_i,
            epe_i,
            epc_i,
            wpla_i,
            off_i,
            epsf_i,
            sigf_i,
        ) = _update_one_element(
            p,
            s_mat[i],
            d_mat[i],
            ep[i],
            dt,
            dam[i],
            epe[i],
            epc[i],
            wpla[i],
            off[i],
            epsf[i],
            sigf[i],
        )
        s_out[i] = s_i
        ep_out[i] = ep_i
        dam[i] = dam_i
        epe[i] = epe_i
        epc[i] = epc_i
        wpla[i] = wpla_i
        off[i] = off_i
        epsf[i] = epsf_i
        sigf[i] = sigf_i
        # Historical Tsai-Wu utilization factor (m12law.F:533)
        wvec_i = dam_i[3]
        sigmx_i = float(p.get("fmax", 1.0e10))
        cb_i = float(p.get("cb", 0.0))
        cn_i = float(p.get("cn", 1.0))
        sigmy_i = min(sigmx_i, 1.0 + cb_i * (wpla_i**cn_i if wpla_i > 0 else 0.0))
        tsaiwu[i] = max(min(wvec_i / max(sigmy_i, 1.0e-20), 1.0), float(tsaiwu[i]))

    if has_rot:
        s_out = m14ftg(s_out, ax, ay, az, bx, by, bz, cx, cy, cz)

    # Store back updated state variables into extra
    if extra is not None:
        state_vars = [
            ("dam12", dam),
            ("epe12", epe),
            ("epc12", epc),
            ("wpla12", wpla),
            ("off12", off),
            ("epsf12", epsf),
            ("sigf12", sigf),
            ("tsaiwu12", tsaiwu),
            ("tsaiwu", tsaiwu),
            ("dam", dam),
            ("off", off),
            ("wpla", wpla),
        ]
        for k, v in state_vars:
            if k in extra and isinstance(extra[k], np.ndarray):
                try:
                    extra[k][:] = v.reshape(extra[k].shape)
                    continue
                except Exception:
                    pass
            extra[k] = v.copy()

    c_val = sound_speed(mat)
    c_arr = np.full(n, float(c_val), dtype=float)

    if epsp is not None and isinstance(epsp, np.ndarray):
        try:
            epsp[:] = ep_out.reshape(epsp.shape)
        except Exception:
            pass

    if is_1d:
        return s_out[0], float(ep_out[0]), float(c_arr[0])
    return s_out, ep_out, c_arr


# ============================================================================
# Algorithmic Tangent (consistent_solid_tangent)
# ============================================================================

def consistent_solid_tangent(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    epsp_incr: Any = None,
    deps: Optional[np.ndarray] = None,
    symmetric: bool = False,
    h: float = 1.0e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Exact (n, 6, 6) algorithmic 3D solid tangent tensor.

    In the elastic regime, returns the orthotropic elasticity matrix:
      [[D11, D12, D13, 0,   0,   0  ],
       [D12, D22, D23, 0,   0,   0  ],
       [D13, D23, D33, 0,   0,   0  ],
       [0,   0,   0,   G12, 0,   0  ],
       [0,   0,   0,   0,   G23, 0  ],
       [0,   0,   0,   0,   0,   G31]]
    Under damage / plasticity, computes the consistent algorithmic tangent.
    """
    if deps is None and "deps" in kwargs:
        deps = kwargs["deps"]
    if "symmetric" in kwargs and kwargs["symmetric"] is not None:
        symmetric = bool(kwargs["symmetric"])
    if "h" in kwargs and kwargs["h"] is not None:
        h = float(kwargs["h"])

    p = getattr(mat, "params", {}) or {}
    sig_arr = np.asarray(sig, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    sig_2d = sig_arr[None, :] if is_1d else sig_arr
    n = sig_2d.shape[0]
    if n == 0:
        return np.empty((0, 6, 6), dtype=float)

    d11 = float(p.get("D11", 0.0))
    d12 = float(p.get("D12", 0.0))
    d13 = float(p.get("D13", 0.0))
    d22 = float(p.get("D22", 0.0))
    d23 = float(p.get("D23", 0.0))
    d33 = float(p.get("D33", 0.0))
    g12 = float(p.get("G12", 0.0))
    g23 = float(p.get("G23", 0.0))
    g31 = float(p.get("G31", 0.0))

    D_mat = np.array(
        [
            [d11, d12, d13, 0.0, 0.0, 0.0],
            [d12, d22, d23, 0.0, 0.0, 0.0],
            [d13, d23, d33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g31],
        ],
        dtype=float,
    )

    if extra is None:
        extra = {}

    dam = extra.get("dam12", extra.get("dam"))
    if dam is None:
        dam = np.zeros((n, 5), dtype=float)
    else:
        dam = np.atleast_2d(dam).astype(float)
        if dam.shape[0] != n:
            dam = np.zeros((n, 5), dtype=float)

    epe = extra.get("epe12", extra.get("epe"))
    if epe is None:
        epe = np.zeros((n, 3), dtype=float)
    else:
        epe = np.atleast_2d(epe).astype(float)
        if epe.shape[0] != n:
            epe = np.zeros((n, 3), dtype=float)

    epc = extra.get("epc12", extra.get("epc"))
    if epc is None:
        epc = np.zeros((n, 3), dtype=float)
    else:
        epc = np.atleast_2d(epc).astype(float)
        if epc.shape[0] != n:
            epc = np.zeros((n, 3), dtype=float)

    wpla = extra.get("wpla12", extra.get("wpla"))
    if wpla is None:
        wpla = np.zeros(n, dtype=float)
    else:
        wpla = np.atleast_1d(wpla).astype(float)
        if len(wpla) == 1 and n > 1:
            wpla = np.full(n, wpla[0], dtype=float)

    off = extra.get("off12", extra.get("off"))
    if off is None:
        off = np.ones(n, dtype=float)
    else:
        off = np.atleast_1d(off).astype(float)
        if len(off) == 1 and n > 1:
            off = np.full(n, off[0], dtype=float)

    epsf = extra.get("epsf12", extra.get("epsf"))
    if epsf is None:
        epsf = np.zeros(n, dtype=float)
    else:
        epsf = np.atleast_1d(epsf).astype(float)
        if len(epsf) == 1 and n > 1:
            epsf = np.full(n, epsf[0], dtype=float)

    sigf = extra.get("sigf12", extra.get("sigf"))
    if sigf is None:
        sigf = np.zeros(n, dtype=float)
    else:
        sigf = np.atleast_1d(sigf).astype(float)
        if len(sigf) == 1 and n > 1:
            sigf = np.full(n, sigf[0], dtype=float)

    if epsp is None:
        ep = np.zeros(n, dtype=float)
    else:
        ep = np.atleast_1d(epsp).astype(float)
        if len(ep) == 1 and n > 1:
            ep = np.full(n, ep[0], dtype=float)

    deps_2d = None
    if deps is not None:
        deps_arr = np.asarray(deps, dtype=float)
        deps_2d = deps_arr[None, :] if deps_arr.ndim == 1 else deps_arr

    # Coordinate transformation if axes provided in extra
    axes = extra.get("axes", extra.get("frame", extra.get("A")))
    has_rot = False
    if axes is not None:
        axes_arr = np.asarray(axes, dtype=float)
        if axes_arr.ndim == 2 and axes_arr.shape == (n, 9):
            has_rot = True
            ax, ay, az = axes_arr[:, 0], axes_arr[:, 1], axes_arr[:, 2]
            bx, by, bz = axes_arr[:, 3], axes_arr[:, 4], axes_arr[:, 5]
            cx, cy, cz = axes_arr[:, 6], axes_arr[:, 7], axes_arr[:, 8]
        elif axes_arr.ndim == 3 and axes_arr.shape == (n, 3, 3):
            has_rot = True
            ax, ay, az = axes_arr[:, 0, 0], axes_arr[:, 0, 1], axes_arr[:, 0, 2]
            bx, by, bz = axes_arr[:, 1, 0], axes_arr[:, 1, 1], axes_arr[:, 1, 2]
            cx, cy, cz = axes_arr[:, 2, 0], axes_arr[:, 2, 1], axes_arr[:, 2, 2]
        elif axes_arr.ndim == 2 and axes_arr.shape == (3, 3):
            has_rot = True
            ax = np.full(n, axes_arr[0, 0])
            ay = np.full(n, axes_arr[0, 1])
            az = np.full(n, axes_arr[0, 2])
            bx = np.full(n, axes_arr[1, 0])
            by = np.full(n, axes_arr[1, 1])
            bz = np.full(n, axes_arr[1, 2])
            cx = np.full(n, axes_arr[2, 0])
            cy = np.full(n, axes_arr[2, 1])
            cz = np.full(n, axes_arr[2, 2])

    if has_rot:
        s_mat, d_mat = m14gtf(
            sig_2d,
            deps_2d if deps_2d is not None else np.zeros((n, 6), dtype=float),
            ax, ay, az, bx, by, bz, cx, cy, cz,
        )
    else:
        s_mat = sig_2d
        d_mat = deps_2d if deps_2d is not None else np.zeros((n, 6), dtype=float)

    tangents = np.zeros((n, 6, 6), dtype=float)

    for i in range(n):
        off_i = off[i]
        if off_i <= 0.0 or off_i < 0.1:
            continue

        base_d = d_mat[i]
        s_i = s_mat[i]
        dam_i = dam[i].copy()
        is_damaged = (
            dam_i[0] > 0.0
            or dam_i[1] > 0.0
            or dam_i[2] > 0.0
            or epc[i, 0] > 0.0
            or epc[i, 1] > 0.0
            or epc[i, 2] > 0.0
        )

        # Trial stresses under base_d
        t1 = s_i[0] + d11 * base_d[0] + d12 * base_d[1] + d13 * base_d[2]
        t2 = s_i[1] + d12 * base_d[0] + d22 * base_d[1] + d23 * base_d[2]
        t3 = s_i[2] + d13 * base_d[0] + d23 * base_d[1] + d33 * base_d[2]
        t4 = s_i[3] + g12 * base_d[3]
        t5 = s_i[4] + g23 * base_d[4]
        t6 = s_i[5] + g31 * base_d[5]

        sigt1 = float(p.get("sigt1", 0.0))
        sigt2 = float(p.get("sigt2", 0.0))
        sigt3 = float(p.get("sigt3", 0.0))
        will_crack = (
            (sigt1 > 0.0 and t1 > sigt1)
            or (sigt2 > 0.0 and t2 > sigt2)
            or (sigt3 > 0.0 and t3 > sigt3)
        )

        # Tsai-Wu yield evaluation at trial stress
        f1 = float(p.get("F1", 0.0))
        f2 = float(p.get("F2", 0.0))
        f3 = float(p.get("F3", 0.0))
        f11 = float(p.get("F11", 0.0))
        f22 = float(p.get("F22", 0.0))
        f33 = float(p.get("F33", 0.0))
        f44 = float(p.get("F44", 0.0))
        f55 = float(p.get("F55", 0.0))
        f66 = float(p.get("F66", 0.0))
        f12 = float(p.get("F12", 0.0))
        f23 = float(p.get("F23", 0.0))
        f13 = float(p.get("F13", 0.0))

        wvec_tw = (
            f1 * t1
            + f2 * t2
            + f3 * t3
            + f11 * (t1**2)
            + f22 * (t2**2)
            + f33 * (t3**2)
            + f44 * (t4**2)
            + f55 * (t5**2)
            + f66 * (t6**2)
            + 2.0 * f12 * t1 * t2
            + 2.0 * f13 * t1 * t3
            + 2.0 * f23 * t2 * t3
        )

        cb_val = float(p.get("cb", 0.0))
        cn_val = float(p.get("cn", 1.0))
        fmax_val = float(p.get("fmax", 1.0e10))
        sigmy = min(fmax_val, 1.0 + cb_val * (wpla[i] ** cn_val if wpla[i] > 0.0 else 0.0))

        if not has_rot and not is_damaged and not will_crack and wvec_tw < sigmy and off_i == 1.0:
            tangents[i] = D_mat.copy()
            if symmetric:
                tangents[i] = 0.5 * (tangents[i] + tangents[i].T)
            continue

        # Algorithmic tangent via consistent numerical perturbation
        for j in range(6):
            if has_rot:
                d_p_g = (deps_2d[i] if deps_2d is not None else np.zeros(6, dtype=float)).copy()
                d_m_g = d_p_g.copy()
                d_p_g[j] += h
                d_m_g[j] -= h

                _, d_p_m = m14gtf(
                    sig_2d[i : i + 1],
                    d_p_g[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )
                _, d_m_m = m14gtf(
                    sig_2d[i : i + 1],
                    d_m_g[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )

                s_p_m, _, _, _, _, _, _, _, _ = _update_one_element(
                    p,
                    s_i.copy(),
                    d_p_m[0],
                    ep[i],
                    dt,
                    dam_i.copy(),
                    epe[i].copy(),
                    epc[i].copy(),
                    wpla[i],
                    off_i,
                    epsf[i],
                    sigf[i],
                )
                s_m_m, _, _, _, _, _, _, _, _ = _update_one_element(
                    p,
                    s_i.copy(),
                    d_m_m[0],
                    ep[i],
                    dt,
                    dam_i.copy(),
                    epe[i].copy(),
                    epc[i].copy(),
                    wpla[i],
                    off_i,
                    epsf[i],
                    sigf[i],
                )

                s_p = m14ftg(
                    s_p_m[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )[0]
                s_m = m14ftg(
                    s_m_m[None, :],
                    ax[i : i + 1], ay[i : i + 1], az[i : i + 1],
                    bx[i : i + 1], by[i : i + 1], bz[i : i + 1],
                    cx[i : i + 1], cy[i : i + 1], cz[i : i + 1],
                )[0]
            else:
                d_p = base_d.copy()
                d_m = base_d.copy()
                d_p[j] += h
                d_m[j] -= h

                s_p, _, _, _, _, _, _, _, _ = _update_one_element(
                    p,
                    s_i.copy(),
                    d_p,
                    ep[i],
                    dt,
                    dam_i.copy(),
                    epe[i].copy(),
                    epc[i].copy(),
                    wpla[i],
                    off_i,
                    epsf[i],
                    sigf[i],
                )
                s_m, _, _, _, _, _, _, _, _ = _update_one_element(
                    p,
                    s_i.copy(),
                    d_m,
                    ep[i],
                    dt,
                    dam_i.copy(),
                    epe[i].copy(),
                    epc[i].copy(),
                    wpla[i],
                    off_i,
                    epsf[i],
                    sigf[i],
                )

            tangents[i, :, j] = (s_p - s_m) / (2.0 * h)

        if symmetric:
            tangents[i] = 0.5 * (tangents[i] + tangents[i].T)

    if is_1d:
        return tangents[0]
    return tangents


solid_tangent = consistent_solid_tangent


def tangent(group: Any = None, sig: Optional[np.ndarray] = None, **kwargs: Any) -> Optional[np.ndarray]:
    """Elemental / group tangent interface compliance."""
    if group is None:
        return None
    mat = getattr(group, "mat", group)
    if sig is None:
        sig = np.zeros(6, dtype=float)
    try:
        return solid_tangent(mat, sig, **kwargs)
    except Exception:
        return None


# ============================================================================
# Registry Hook
# ============================================================================

def _register() -> None:
    """Register LAW12 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY

        for key in (
            12,
            "12",
            "LAW12",
            "3D_COMP",
            "COMP_3D",
            "MAT_LAW12",
            "MAT_3D_COMP",
            "MAT_COMP_3D",
            "3PARBI",
            "MAT_3PARBI",
            "LAW12_3PARBI",
            "LAW12_3D_COMP",
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law12
    except Exception:
        pass


_register()
