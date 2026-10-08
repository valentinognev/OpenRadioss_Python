"""
pyradioss.accel.gpu_kernels — CuPy/CUDA accelerated compute backend (WS14).

CuPy mirrors of the hot kernel blocks mirroring the existing numba backend
architecture in pyradioss.accel.jit_kernels.

Kernel blocks mirrored:
    hexa_pre      <-> elements/solid_hexa8._pre     (srcoor3/sdefo3/srota3)
    hexa_post     <-> elements/solid_hexa8._post    (sbulk3/sfint3/shour3/sdlen3)
    hexa_hgphys   <-> elements/solid_hexa8._phys_hourglass_law70 (shour3)
    tetra10_pre   <-> elements/solid_tetra10._pre   (t10coor/t10defo)
    tetra10_post  <-> elements/solid_tetra10._post  (t10fint/t10dlen)
    shell_pre     <-> elements/shell_bt4._pre       (ccoor3/cdefo3)
    shell_post    <-> elements/shell_bt4._post      (czforc3/chvis3)
    t7_narrow     <-> contact/inter_type7._narrow   (i7dst3)
    t24_narrow    <-> contact/inter_type24._narrow  (i7dst3)
    scatter3      <-> parallel/asspar.F             (asspar)
    law70_*       <-> materials/law70_tabfoam       (sigeps70)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HAS_CUPY",
    "to_gpu",
    "to_cpu",
    "hexa_pre",
    "hexa_post",
    "hexa_hgphys",
    "shell_pre",
    "shell_post",
    "_tri_closest",
    "t7_narrow",
    "t24_narrow",
    "scatter3",
    "scatter3_colored",
    "law70_tab2d",
    "law70_enorm",
    "law70_snorm",
    "law70_elastic_stress",
    "tetra10_pre",
    "tetra10_post",
    "qbat_pre_flat",
    "qbat_post_flat",
    "qbat_pre",
    "qbat_post",
    "qeph_pre",
    "qeph_post",
]

try:
    import cupy as cp
    try:
        import cupyx
    except ImportError:
        cupyx = None
    HAS_CUPY = bool(cp.cuda.is_available()) if hasattr(cp, "cuda") and hasattr(cp.cuda, "is_available") else True
except Exception:
    cp = None
    cupyx = None
    HAS_CUPY = False

from ...common.constants import EM20, EP30

# ---- Module-level constants matching OpenRadioss tables -------------------
_XI8_DATA = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
]
_H4_DATA = [
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, -1, 1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, 1, -1, 1, -1],
]
_FACES6_DATA = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
]
_HG_PHYS = 0.03
_DT_HG_SF = 0.9

_DN_DXI_T10_DATA = [
    [[-0.44721360, 0.00000000, 0.00000000],
     [ 0.00000000,-0.44721360, 0.00000000],
     [ 0.00000000, 0.00000000,-0.44721360],
     [-1.34164079,-1.34164079,-1.34164079],
     [ 0.55278640, 0.55278640, 0.00000000],
     [ 0.00000000, 0.55278640, 0.55278640],
     [ 0.55278640, 0.00000000, 0.55278640],
     [ 1.78885438,-0.55278640,-0.55278640],
     [-0.55278640, 1.78885438,-0.55278640],
     [-0.55278640,-0.55278640, 1.78885438]],
    [[ 1.34164079, 0.00000000, 0.00000000],
     [ 0.00000000,-0.44721360, 0.00000000],
     [ 0.00000000, 0.00000000,-0.44721360],
     [ 0.44721360, 0.44721360, 0.44721360],
     [ 0.55278640, 2.34164079, 0.00000000],
     [ 0.00000000, 0.55278640, 0.55278640],
     [ 0.55278640, 0.00000000, 2.34164079],
     [-1.78885438,-2.34164079,-2.34164079],
     [-0.55278640, 0.00000000,-0.55278640],
     [-0.55278640,-0.55278640, 0.00000000]],
    [[-0.44721360, 0.00000000, 0.00000000],
     [ 0.00000000, 1.34164079, 0.00000000],
     [ 0.00000000, 0.00000000,-0.44721360],
     [ 0.44721360, 0.44721360, 0.44721360],
     [ 2.34164079, 0.55278640, 0.00000000],
     [ 0.00000000, 0.55278640, 2.34164079],
     [ 0.55278640, 0.00000000, 0.55278640],
     [ 0.00000000,-0.55278640,-0.55278640],
     [-2.34164079,-1.78885438,-2.34164079],
     [-0.55278640,-0.55278640, 0.00000000]],
    [[-0.44721360, 0.00000000, 0.00000000],
     [ 0.00000000,-0.44721360, 0.00000000],
     [ 0.00000000, 0.00000000, 1.34164079],
     [ 0.44721360, 0.44721360, 0.44721360],
     [ 0.55278640, 0.55278640, 0.00000000],
     [ 0.00000000, 2.34164079, 0.55278640],
     [ 2.34164079, 0.00000000, 0.55278640],
     [ 0.00000000,-0.55278640,-0.55278640],
     [-0.55278640, 0.00000000,-0.55278640],
     [-2.34164079,-2.34164079,-1.78885438]],
]
_WIP_T10_DATA = [0.25, 0.25, 0.25, 0.25]
_FACES_T10_DATA = [
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2],
]


# ---- Array transfer helpers ------------------------------------------------

def to_gpu(array):
    """Transfer an array to GPU memory (CuPy array).

    If array is already a CuPy array, it is returned as is.
    If cupy is not installed or CUDA is unavailable, raises RuntimeError.
    """
    if array is None:
        return None
    if cp is None:
        raise RuntimeError("CuPy/CUDA is not available")
    if isinstance(array, cp.ndarray):
        return array
    return cp.asarray(array)


def to_cpu(array):
    """Transfer an array from GPU memory to CPU (NumPy array).

    If array has a .get() method (CuPy array), calls it.
    Otherwise returns as a NumPy array.
    """
    if array is None:
        return None
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def _get_xp(*arrays):
    """Return cupy if any input is a CuPy array or if cupy is available, else numpy."""
    if cp is not None:
        return cp
    return np


# ============================================================================
# solid_hexa8 GPU kernels
# ============================================================================

def hexa_pre(xe, ve, sig, dt, off, lc_scale):
    """Mirror of solid_hexa8._pre on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/solid/srcoor3.F
    # Ported from $OR_SRC/engine/source/elements/solid/sdefo3.F
    # Ported from $OR_SRC/engine/source/elements/solid/srota3.F
    """
    xp = _get_xp(xe)
    is_np = isinstance(xe, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xe_g = cp.asarray(xe)
        ve_g = cp.asarray(ve)
        sig_g = cp.asarray(sig)
        off_g = cp.asarray(off)
        lc_scale_g = cp.asarray(lc_scale)
    else:
        xe_g = xe
        ve_g = ve
        sig_g = sig
        off_g = off
        lc_scale_g = lc_scale

    n = xe_g.shape[0]
    if n == 0:
        dndx = xp.empty((0, 8, 3), dtype=xe_g.dtype)
        vol = xp.empty(0, dtype=xe_g.dtype)
        lc = xp.empty(0, dtype=xe_g.dtype)
        deps = xp.empty((0, 6), dtype=xe_g.dtype)
        trD = xp.empty(0, dtype=xe_g.dtype)
        return (to_cpu(dndx), to_cpu(vol), to_cpu(lc), to_cpu(deps), to_cpu(trD)) if is_np else (dndx, vol, lc, deps, trD)

    XI8 = xp.asarray(_XI8_DATA, dtype=xe_g.dtype) / 8.0
    DN_DXI_T = XI8.T  # (3, 8)

    # Jacobian J[a,b] = sum_i dN_i/dxi_a * x_i,b: (3, 8) @ (n, 8, 3) -> (n, 3, 3)
    J = DN_DXI_T @ xe_g
    j00, j01, j02 = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
    j10, j11, j12 = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
    j20, j21, j22 = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]

    A = j11 * j22 - j12 * j21
    B = j12 * j20 - j10 * j22
    C = j10 * j21 - j11 * j20
    det = j00 * A + j01 * B + j02 * C
    v8 = 8.0 * det
    vol = xp.where(v8 > EM20, v8, EM20)

    idet = xp.divide(1.0, det)
    i00 = A * idet
    i01 = (j02 * j21 - j01 * j22) * idet
    i02 = (j01 * j12 - j02 * j11) * idet
    i10 = B * idet
    i11 = (j00 * j22 - j02 * j20) * idet
    i12 = (j02 * j10 - j00 * j12) * idet
    i20 = C * idet
    i21 = (j01 * j20 - j00 * j21) * idet
    i22 = (j00 * j11 - j01 * j10) * idet

    row0 = xp.stack([i00, i01, i02], axis=-1)
    row1 = xp.stack([i10, i11, i12], axis=-1)
    row2 = xp.stack([i20, i21, i22], axis=-1)
    Jinv = xp.stack([row0, row1, row2], axis=-2)
    dndx = XI8 @ Jinv.transpose(0, 2, 1)

    # Characteristic length: vol / max face area
    FACES6 = xp.asarray(_FACES6_DATA, dtype=xp.int64)
    n0, n1, n2, n3 = FACES6[:, 0], FACES6[:, 1], FACES6[:, 2], FACES6[:, 3]
    d1 = xe_g[:, n2, :] - xe_g[:, n0, :]
    d2 = xe_g[:, n3, :] - xe_g[:, n1, :]
    c_prod = xp.cross(d1, d2)
    areas = 0.5 * xp.sqrt(xp.sum(c_prod * c_prod, axis=-1))
    amax = xp.max(areas, axis=1)
    lc = (vol / xp.where(amax > EM20, amax, EM20)) * lc_scale_g

    # Velocity gradient L[b,c] = sum_i ve[i,b] * dndx[i,c] -> ve.T @ dndx
    L = ve_g.transpose(0, 2, 1) @ dndx
    l00, l01, l02 = L[:, 0, 0], L[:, 0, 1], L[:, 0, 2]
    l10, l11, l12 = L[:, 1, 0], L[:, 1, 1], L[:, 1, 2]
    l20, l21, l22 = L[:, 2, 0], L[:, 2, 1], L[:, 2, 2]

    alive = off_g > 0.0
    tr = l00 + l11 + l22
    vmax = xp.max(xp.abs(ve_g), axis=(1, 2))
    gmax = xp.max(xp.abs(dndx), axis=(1, 2))
    tr = xp.where(xp.abs(tr) <= 1e-14 * (vmax * gmax), 0.0, tr)
    trD = xp.where(alive, tr, 0.0)

    deps = xp.zeros((n, 6), dtype=xe_g.dtype)
    deps[:, 0] = xp.where(alive, l00 * dt, 0.0)
    deps[:, 1] = xp.where(alive, l11 * dt, 0.0)
    deps[:, 2] = xp.where(alive, l22 * dt, 0.0)
    deps[:, 3] = xp.where(alive, (l01 + l10) * dt, 0.0)
    deps[:, 4] = xp.where(alive, (l12 + l21) * dt, 0.0)
    deps[:, 5] = xp.where(alive, (l02 + l20) * dt, 0.0)

    # Jaumann rotation of sig in place
    wxy = 0.5 * (l01 - l10) * dt
    wyz = 0.5 * (l12 - l21) * dt
    wxz = 0.5 * (l02 - l20) * dt

    sxx = sig_g[:, 0].copy()
    syy = sig_g[:, 1].copy()
    szz = sig_g[:, 2].copy()
    sxy = sig_g[:, 3].copy()
    syz = sig_g[:, 4].copy()
    szx = sig_g[:, 5].copy()

    sig_g[:, 0] = sxx + 2.0 * (wxy * sxy + wxz * szx)
    sig_g[:, 1] = syy + 2.0 * (-wxy * sxy + wyz * syz)
    sig_g[:, 2] = szz + 2.0 * (-wxz * szx - wyz * syz)
    sig_g[:, 3] = sxy + wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig_g[:, 4] = syz + wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig_g[:, 5] = szx + wxz * (szz - sxx) + wxy * syz - wyz * sxy

    if is_np:
        sig[...] = to_cpu(sig_g)
        return to_cpu(dndx), to_cpu(vol), to_cpu(lc), to_cpu(deps), to_cpu(trD)
    return dndx, vol, lc, deps, trD


def hexa_post(xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
              qa, qb, c, hcoef, alive, qvw_pend, dt, dtfac):
    """Mirror of solid_hexa8._post on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/solid/sbulk3.F
    # Ported from $OR_SRC/engine/source/elements/solid/sfint3.F
    # Ported from $OR_SRC/engine/source/elements/solid/shour3.F
    # Ported from $OR_SRC/engine/source/elements/solid/sdlen3.F
    """
    xp = _get_xp(xe)
    is_np = isinstance(xe, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xe_g = cp.asarray(xe)
        ve_g = cp.asarray(ve)
        dndx_g = cp.asarray(dndx)
        vol_g = cp.asarray(vol)
        lc_g = cp.asarray(lc)
        rho_g = cp.asarray(rho)
        trD_g = cp.asarray(trD)
        deps_g = cp.asarray(deps)
        sig_g = cp.asarray(sig)
        sig_old_g = cp.asarray(sig_old)
        qa_g = cp.asarray(qa)
        qb_g = cp.asarray(qb)
        c_g = cp.asarray(c)
        alive_g = cp.asarray(alive)
        qvw_pend_g = cp.asarray(qvw_pend)
        dtfac_g = cp.asarray(dtfac)
    else:
        xe_g = xe
        ve_g = ve
        dndx_g = dndx
        vol_g = vol
        lc_g = lc
        rho_g = rho
        trD_g = trD
        deps_g = deps
        sig_g = sig
        sig_old_g = sig_old
        qa_g = qa
        qb_g = qb
        c_g = c
        alive_g = alive
        qvw_pend_g = qvw_pend
        dtfac_g = dtfac

    n = xe_g.shape[0]
    live = alive_g.astype(bool) if hasattr(alive_g, "astype") else alive_g

    # Bulk viscosity
    compressing = (trD_g < 0.0) & live
    qv = xp.where(compressing, rho_g * lc_g * (qa_g * qa_g * lc_g * trD_g * trD_g - qb_g * c_g * trD_g), 0.0)

    # Stress matrix with viscous pressure
    S = xp.empty((n, 3, 3), dtype=xe_g.dtype)
    S[:, 0, 0] = sig_g[:, 0] - qv
    S[:, 1, 1] = sig_g[:, 1] - qv
    S[:, 2, 2] = sig_g[:, 2] - qv
    S[:, 0, 1] = S[:, 1, 0] = sig_g[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig_g[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig_g[:, 5]

    fe = (dndx_g @ S) * (-vol_g)[:, None, None]

    # Hourglass forces
    H4 = xp.asarray(_H4_DATA, dtype=xe_g.dtype)
    hx = H4 @ xe_g  # (n, 4, 3)
    gamma = H4[None, :, :] - hx @ dndx_g.transpose(0, 2, 1)  # (n, 4, 8)
    qd = gamma @ ve_g  # (n, 4, 3)
    ah = xp.where(live, hcoef * rho_g * c_g * (vol_g ** (2.0 / 3.0)) / 4.0, 0.0)
    fhg = (gamma.transpose(0, 2, 1) @ qd) * (-ah)[:, None, None]
    fe += fhg
    dehour = -xp.sum(fhg * ve_g, axis=(1, 2)) * dt

    # Energy increments
    w_visc = 0.5 * vol_g * qv * (-trD_g * dt) + qvw_pend_g * (-trD_g)
    qvw_new = 0.5 * vol_g * qv * dt
    sig_mid = 0.5 * (sig_old_g + sig_g)
    deint0 = vol_g * xp.sum(sig_mid * deps_g, axis=1)

    # Critical dt
    Q = xp.where(compressing, qb_g * c_g + qa_g * lc_g * xp.abs(trD_g), 0.0)
    denom = Q + xp.sqrt(Q * Q + c_g * c_g)
    dt_crit = xp.where(live & (denom > EM20), dtfac_g * lc_g / denom, EP30)

    if is_np:
        return to_cpu(fe), to_cpu(dt_crit), to_cpu(w_visc), to_cpu(qvw_new), to_cpu(deint0), to_cpu(dehour)
    return fe, dt_crit, w_visc, qvw_new, deint0, dehour


def hexa_hgphys(xe, ve, dndx, vol, c, mask, mass, vol0, q, dt):
    """Mirror of solid_hexa8._phys_hourglass_law70 on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/solid/shour3.F
    """
    xp = _get_xp(xe)
    is_np = isinstance(xe, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xe_g = cp.asarray(xe)
        ve_g = cp.asarray(ve)
        dndx_g = cp.asarray(dndx)
        vol_g = cp.asarray(vol)
        c_g = cp.asarray(c)
        mask_g = cp.asarray(mask)
        mass_g = cp.asarray(mass)
        vol0_g = cp.asarray(vol0)
        q_g = cp.asarray(q)
    else:
        xe_g = xe
        ve_g = ve
        dndx_g = dndx
        vol_g = vol
        c_g = c
        mask_g = mask
        mass_g = mass
        vol0_g = vol0
        q_g = q

    H4 = xp.asarray(_H4_DATA, dtype=xe_g.dtype)
    hx = H4 @ xe_g  # (n, 4, 3)
    gamma = H4[None, :, :] - hx @ dndx_g.transpose(0, 2, 1)  # (n, 4, 8)
    v0 = xp.maximum(vol0_g, EM20)
    aa1 = (mass_g / v0) * c_g * c_g
    traceS = xp.sum(dndx_g * dndx_g, axis=(1, 2))
    kstiff = xp.where(mask_g, _HG_PHYS * aa1 * vol_g * traceS, 0.0)

    # Accumulate modal displacement in place for masked (LAW70) elements
    q_inc = (gamma @ ve_g) * dt
    q_g += xp.where(mask_g[:, None, None], q_inc, 0.0)

    f_hg = (gamma.transpose(0, 2, 1) @ q_g) * (-kstiff)[:, None, None]
    dehour = -xp.sum(f_hg * ve_g, axis=(1, 2)) * dt

    gnorm = xp.sum(gamma * gamma, axis=(1, 2))
    den = xp.maximum(2.0 * kstiff * gnorm, EM20)
    mm = xp.maximum(mass_g, EM20)
    dt_hg = xp.where(kstiff > 0.0, _DT_HG_SF * xp.sqrt(mm / den), EP30)

    if is_np:
        q[...] = to_cpu(q_g)
        return to_cpu(f_hg), to_cpu(dehour), to_cpu(dt_hg)
    return f_hg, dehour, dt_hg


# ============================================================================
# shell_bt4 GPU kernels
# ============================================================================

def shell_pre(xe, ve, vre, off):
    """Mirror of shell_bt4._pre on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/shell/ccoor3.F
    # Ported from $OR_SRC/engine/source/elements/shell/cdefo3.F
    """
    xp = _get_xp(xe)
    is_np = isinstance(xe, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xe_g = cp.asarray(xe)
        ve_g = cp.asarray(ve)
        vre_g = cp.asarray(vre)
        off_g = cp.asarray(off)
    else:
        xe_g = xe
        ve_g = ve
        vre_g = vre
        off_g = off

    n = xe_g.shape[0]
    r31 = xe_g[:, 2, :] - xe_g[:, 0, :]
    r42 = xe_g[:, 3, :] - xe_g[:, 1, :]
    e3 = xp.cross(r31, r42)
    nrm3 = xp.maximum(xp.sqrt(xp.sum(e3 * e3, axis=-1, keepdims=True)), EM20)
    e3 = e3 / nrm3

    s1 = xe_g[:, 1, :] - xe_g[:, 0, :]
    dot = xp.sum(s1 * e3, axis=-1, keepdims=True)
    e1 = s1 - dot * e3
    nrm1 = xp.maximum(xp.sqrt(xp.sum(e1 * e1, axis=-1, keepdims=True)), EM20)
    e1 = e1 / nrm1
    e2 = xp.cross(e3, e1)
    E = xp.stack([e1, e2, e3], axis=-1)

    cx = xp.mean(xe_g, axis=1, keepdims=True)
    d = xe_g - cx
    xl = xp.empty((n, 4, 2), dtype=xe_g.dtype)
    xl[:, :, 0] = xp.sum(d * e1[:, None, :], axis=-1)
    xl[:, :, 1] = xp.sum(d * e2[:, None, :], axis=-1)

    A = 0.5 * ((xl[:, 2, 0] - xl[:, 0, 0]) * (xl[:, 3, 1] - xl[:, 1, 1])
               + (xl[:, 1, 0] - xl[:, 3, 0]) * (xl[:, 2, 1] - xl[:, 0, 1]))
    twoA = xp.maximum(2.0 * A, EM20)
    inv2A = 1.0 / twoA

    B1 = xp.empty((n, 4), dtype=xe_g.dtype)
    B1[:, 0] = (xl[:, 1, 1] - xl[:, 3, 1]) * inv2A
    B1[:, 1] = (xl[:, 2, 1] - xl[:, 0, 1]) * inv2A
    B1[:, 2] = (xl[:, 3, 1] - xl[:, 1, 1]) * inv2A
    B1[:, 3] = (xl[:, 0, 1] - xl[:, 2, 1]) * inv2A

    B2 = xp.empty((n, 4), dtype=xe_g.dtype)
    B2[:, 0] = (xl[:, 3, 0] - xl[:, 1, 0]) * inv2A
    B2[:, 1] = (xl[:, 0, 0] - xl[:, 2, 0]) * inv2A
    B2[:, 2] = (xl[:, 1, 0] - xl[:, 3, 0]) * inv2A
    B2[:, 3] = (xl[:, 2, 0] - xl[:, 0, 0]) * inv2A
    area = xp.where(A > EM20, A, EM20)

    side0 = (xl[:, 1, 0] - xl[:, 0, 0])**2 + (xl[:, 1, 1] - xl[:, 0, 1])**2
    side1 = (xl[:, 2, 0] - xl[:, 1, 0])**2 + (xl[:, 2, 1] - xl[:, 1, 1])**2
    side2 = (xl[:, 3, 0] - xl[:, 2, 0])**2 + (xl[:, 3, 1] - xl[:, 2, 1])**2
    side3 = (xl[:, 0, 0] - xl[:, 3, 0])**2 + (xl[:, 0, 1] - xl[:, 3, 1])**2
    lmax = xp.maximum(xp.maximum(side0, side1), xp.maximum(side2, side3))
    lden = xp.maximum(xp.sqrt(lmax), EM20)
    lc = area / lden

    V = xp.empty((n, 4, 5), dtype=xe_g.dtype)
    V[:, :, :3] = ve_g @ E
    V[:, :, 3:] = (vre_g @ E)[:, :, :2]

    Bt = xp.empty((n, 2, 4), dtype=xe_g.dtype)
    Bt[:, 0, :] = B1
    Bt[:, 1, :] = B2
    M = Bt @ V
    m00, m10 = M[:, 0, 0], M[:, 1, 0]
    m01, m11 = M[:, 0, 1], M[:, 1, 1]
    m02, m12 = M[:, 0, 2], M[:, 1, 2]
    m03, m13 = M[:, 0, 3], M[:, 1, 3]
    m04, m14 = M[:, 0, 4], M[:, 1, 4]
    thx_m = xp.mean(V[:, :, 3], axis=1)
    thy_m = xp.mean(V[:, :, 4], axis=1)

    alive = off_g > 0.0
    dm = xp.zeros((n, 3), dtype=xe_g.dtype)
    kap = xp.zeros((n, 3), dtype=xe_g.dtype)
    gs = xp.zeros((n, 2), dtype=xe_g.dtype)
    dm[:, 0] = xp.where(alive, m00, 0.0)
    dm[:, 1] = xp.where(alive, m11, 0.0)
    dm[:, 2] = xp.where(alive, m01 + m10, 0.0)
    kap[:, 0] = xp.where(alive, m04, 0.0)
    kap[:, 1] = xp.where(alive, -m13, 0.0)
    kap[:, 2] = xp.where(alive, m14 - m03, 0.0)
    gs[:, 0] = xp.where(alive, m02 + thy_m, 0.0)
    gs[:, 1] = xp.where(alive, m12 - thx_m, 0.0)

    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = xp.empty((n, 4), dtype=xe_g.dtype)
    gam[:, 0] = 1.0 - hx * B1[:, 0] - hy * B2[:, 0]
    gam[:, 1] = -1.0 - hx * B1[:, 1] - hy * B2[:, 1]
    gam[:, 2] = 1.0 - hx * B1[:, 2] - hy * B2[:, 2]
    gam[:, 3] = -1.0 - hx * B1[:, 3] - hy * B2[:, 3]
    bb = xp.sum(B1 * B1 + B2 * B2, axis=1)

    if is_np:
        return (to_cpu(E), to_cpu(area), to_cpu(lc), to_cpu(B1), to_cpu(B2),
                to_cpu(bb), to_cpu(gam), to_cpu(V), to_cpu(dm), to_cpu(kap), to_cpu(gs))
    return E, area, lc, B1, B2, bb, gam, V, dm, kap, gs


def shell_post(E, area, B1, B2, gam, V, Nres, Mres, qres, Q,
               k_m, k_w, hqm, hqb, hqr, dt):
    """Mirror of shell_bt4._post on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/shell/czforc3.F
    # Ported from $OR_SRC/engine/source/elements/shell/chvis3.F
    """
    xp = _get_xp(area)
    is_np = isinstance(area, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        E_g = cp.asarray(E)
        area_g = cp.asarray(area)
        B1_g = cp.asarray(B1)
        B2_g = cp.asarray(B2)
        gam_g = cp.asarray(gam)
        V_g = cp.asarray(V)
        Nres_g = cp.asarray(Nres)
        Mres_g = cp.asarray(Mres)
        qres_g = cp.asarray(qres)
        Q_g = cp.asarray(Q)
        k_m_g = cp.asarray(k_m)
        k_w_g = cp.asarray(k_w)
        hqm_g = cp.asarray(hqm)
        hqb_g = cp.asarray(hqb)
        hqr_g = cp.asarray(hqr)
    else:
        E_g = E
        area_g = area
        B1_g = B1
        B2_g = B2
        gam_g = gam
        V_g = V
        Nres_g = Nres
        Mres_g = Mres
        qres_g = qres
        Q_g = Q
        k_m_g = k_m
        k_w_g = k_w
        hqm_g = hqm
        hqb_g = hqb
        hqr_g = hqr

    n = area_g.shape[0]
    qd = xp.empty((n, 5), dtype=area_g.dtype)
    qd[:, 0] = xp.sum(gam_g * V_g[:, :, 0], axis=1)
    qd[:, 1] = xp.sum(gam_g * V_g[:, :, 1], axis=1)
    qd[:, 2] = xp.sum(gam_g * V_g[:, :, 2], axis=1)
    qd[:, 3] = V_g[:, 0, 3] - V_g[:, 1, 3] + V_g[:, 2, 3] - V_g[:, 3, 3]
    qd[:, 4] = V_g[:, 0, 4] - V_g[:, 1, 4] + V_g[:, 2, 4] - V_g[:, 3, 4]

    # Update elastic hourglass state Q in place
    Q_g[:, 0] += k_m_g * qd[:, 0] * dt
    Q_g[:, 1] += k_m_g * qd[:, 1] * dt
    Q_g[:, 2] += k_w_g * qd[:, 2] * dt
    Q_g[:, 3] = 0.0
    Q_g[:, 4] = 0.0

    F = xp.empty((n, 5), dtype=area_g.dtype)
    F[:, 0] = Q_g[:, 0] + qd[:, 0] * hqm_g * xp.abs(qd[:, 0])
    F[:, 1] = Q_g[:, 1] + qd[:, 1] * hqm_g * xp.abs(qd[:, 1])
    F[:, 2] = Q_g[:, 2] + qd[:, 2] * hqb_g * xp.abs(qd[:, 2])
    F[:, 3] = qd[:, 3] * hqr_g * xp.abs(qd[:, 3])
    F[:, 4] = qd[:, 4] * hqr_g * xp.abs(qd[:, 4])
    dehg = xp.sum(F * qd, axis=1) * dt

    h_raw = xp.array([1.0, -1.0, 1.0, -1.0], dtype=area_g.dtype)
    fl = xp.empty((n, 4, 3), dtype=area_g.dtype)
    ml = xp.empty((n, 4, 3), dtype=area_g.dtype)

    fl[:, :, 0] = -area_g[:, None] * (B1_g * Nres_g[:, 0:1] + B2_g * Nres_g[:, 2:3]) - gam_g * F[:, 0:1]
    fl[:, :, 1] = -area_g[:, None] * (B2_g * Nres_g[:, 1:2] + B1_g * Nres_g[:, 2:3]) - gam_g * F[:, 1:2]
    fl[:, :, 2] = -area_g[:, None] * (B1_g * qres_g[:, 0:1] + B2_g * qres_g[:, 1:2]) - gam_g * F[:, 2:3]

    ml[:, :, 0] = -area_g[:, None] * (-B2_g * Mres_g[:, 1:2] - B1_g * Mres_g[:, 2:3] - 0.25 * qres_g[:, 1:2]) - h_raw[None, :] * F[:, 3:4]
    ml[:, :, 1] = -area_g[:, None] * (B1_g * Mres_g[:, 0:1] + B2_g * Mres_g[:, 2:3] + 0.25 * qres_g[:, 0:1]) - h_raw[None, :] * F[:, 4:5]
    ml[:, :, 2] = 0.0

    fg = fl @ E_g.transpose(0, 2, 1)
    mg = ml @ E_g.transpose(0, 2, 1)

    if is_np:
        Q[...] = to_cpu(Q_g)
        return to_cpu(fg), to_cpu(mg), to_cpu(dehg)
    return fg, mg, dehg


# ============================================================================
# TYPE7 / TYPE24 contact narrow phase GPU kernels
# ============================================================================

def _tri_closest(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Closest point of point P to triangle ABC (Ericson algorithm).
    # Ported from $OR_SRC/engine/source/inter/i7dst3.F
    """
    xp = _get_xp(px, ax)
    abx = bx - ax; aby = by - ay; abz = bz - az
    acx = cx - ax; acy = cy - ay; acz = cz - az
    apx = px - ax; apy = py - ay; apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    bpx = px - bx; bpy = py - by; bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    cpx = px - cx; cpy = py - cy; cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    if np.isscalar(px):
        denom = max(va + vb + vc, EM20)
        v = vb / denom
        w = vc / denom
        if (vc <= 0.0) and (d1 >= 0.0) and (d3 <= 0.0):
            den = max(d1 - d3, EM20)
            v = d1 / den
            w = 0.0
        if (vb <= 0.0) and (d2 >= 0.0) and (d6 <= 0.0):
            den = max(d2 - d6, EM20)
            v = 0.0
            w = d2 / den
        if (va <= 0.0) and (d4 - d3 >= 0.0) and (d5 - d6 >= 0.0):
            den = max((d4 - d3) + (d5 - d6), EM20)
            t = (d4 - d3) / den
            v = 1.0 - t
            w = t
        if (d1 <= 0.0) and (d2 <= 0.0):
            v = 0.0
            w = 0.0
        if (d3 >= 0.0) and (d4 <= d3):
            v = 1.0
            w = 0.0
        if (d6 >= 0.0) and (d5 <= d6):
            v = 0.0
            w = 1.0
        u = 1.0 - v - w
        qx = u * ax + v * bx + w * cx
        qy = u * ay + v * by + w * cy
        qz = u * az + v * bz + w * cz
        return qx, qy, qz, u, v, w

    denom = xp.maximum(va + vb + vc, EM20)
    v = vb / denom
    w = vc / denom

    onAB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denAB = xp.maximum(d1 - d3, EM20)
    v = xp.where(onAB, d1 / denAB, v)
    w = xp.where(onAB, 0.0, w)

    onAC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denAC = xp.maximum(d2 - d6, EM20)
    v = xp.where(onAC, 0.0, v)
    w = xp.where(onAC, d2 / denAC, w)

    onBC = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    denBC = xp.maximum((d4 - d3) + (d5 - d6), EM20)
    t = (d4 - d3) / denBC
    v = xp.where(onBC, 1.0 - t, v)
    w = xp.where(onBC, t, w)

    atA = (d1 <= 0.0) & (d2 <= 0.0)
    v = xp.where(atA, 0.0, v)
    w = xp.where(atA, 0.0, w)

    atB = (d3 >= 0.0) & (d4 <= d3)
    v = xp.where(atB, 1.0, v)
    w = xp.where(atB, 0.0, w)

    atC = (d6 >= 0.0) & (d5 <= d6)
    v = xp.where(atC, 0.0, v)
    w = xp.where(atC, 1.0, w)

    u = 1.0 - v - w
    qx = u * ax + v * bx + w * cx
    qy = u * ay + v * by + w * cy
    qz = u * az + v * bz + w * cz
    return qx, qy, qz, u, v, w


def t7_narrow(x, ni, seg):
    """Closest point of each candidate node on its quad segment.
    # Ported from $OR_SRC/engine/source/inter/i7dst3.F
    """
    xp = _get_xp(x, ni, seg)
    is_np = isinstance(x, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        x_g = cp.asarray(x)
        ni_g = cp.asarray(ni)
        seg_g = cp.asarray(seg)
    else:
        x_g = x
        ni_g = ni
        seg_g = seg

    m = len(ni_g)
    if m == 0:
        d = xp.empty(0, dtype=x_g.dtype)
        pt = xp.empty((0, 3), dtype=x_g.dtype)
        w = xp.empty((0, 4), dtype=x_g.dtype)
        return (to_cpu(d), to_cpu(pt), to_cpu(w)) if is_np else (d, pt, w)

    px, py, pz = x_g[ni_g, 0], x_g[ni_g, 1], x_g[ni_g, 2]
    i0, i1, i2, i3 = seg_g[:, 0], seg_g[:, 1], seg_g[:, 2], seg_g[:, 3]

    # Triangle 0 (0, 1, 2)
    qx0, qy0, qz0, u0, v0, w0 = _tri_closest(
        px, py, pz,
        x_g[i0, 0], x_g[i0, 1], x_g[i0, 2],
        x_g[i1, 0], x_g[i1, 1], x_g[i1, 2],
        x_g[i2, 0], x_g[i2, 1], x_g[i2, 2],
    )
    dx0, dy0, dz0 = px - qx0, py - qy0, pz - qz0
    d0 = xp.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)

    # Triangle 1 (0, 2, 3)
    qx1, qy1, qz1, u1, v1, w1 = _tri_closest(
        px, py, pz,
        x_g[i0, 0], x_g[i0, 1], x_g[i0, 2],
        x_g[i2, 0], x_g[i2, 1], x_g[i2, 2],
        x_g[i3, 0], x_g[i3, 1], x_g[i3, 2],
    )
    dx1, dy1, dz1 = px - qx1, py - qy1, pz - qz1
    d1 = xp.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)

    better = d1 < d0
    best_d = xp.where(better, d1, d0)
    best_pt = xp.empty((m, 3), dtype=x_g.dtype)
    best_pt[:, 0] = xp.where(better, qx1, qx0)
    best_pt[:, 1] = xp.where(better, qy1, qy0)
    best_pt[:, 2] = xp.where(better, qz1, qz0)

    best_w = xp.empty((m, 4), dtype=x_g.dtype)
    best_w[:, 0] = xp.where(better, u1, u0)
    best_w[:, 1] = xp.where(better, 0.0, v0)
    best_w[:, 2] = xp.where(better, v1, w0)
    best_w[:, 3] = xp.where(better, w1, 0.0)

    if is_np:
        return to_cpu(best_d), to_cpu(best_pt), to_cpu(best_w)
    return best_d, best_pt, best_w


# Contact narrow phase alias: TYPE24 uses the same algorithm and signature as TYPE7
t24_narrow = t7_narrow


# ============================================================================
# Force scatter-assembly (Fortran asspar)
# ============================================================================

def scatter3(target, idx, values):
    """Accumulate (m, 3) values into target (N, 3) at rows idx (m,).
    # Ported from $OR_SRC/engine/source/parallel/asspar.F
    """
    if len(target) == 0 or len(idx) == 0:
        return
    xp = _get_xp(target, idx, values)
    if xp is cp and cp is not None and isinstance(target, cp.ndarray):
        if cupyx is not None and hasattr(cupyx, "scatter_add"):
            cupyx.scatter_add(target, idx, values)
        elif hasattr(cp, "add") and hasattr(cp.add, "at"):
            cp.add.at(target, idx, values)
        else:
            for c_idx in range(3):
                target[:, c_idx] += cp.bincount(idx, weights=values[:, c_idx], minlength=target.shape[0])
    else:
        for c_idx in range(3):
            target[:, c_idx] += np.bincount(idx, weights=values[:, c_idx], minlength=target.shape[0])


def scatter3_colored(target, idx, values, color_indices, color_offsets, npe):
    """Node-colored force accumulation on GPU.
    # Ported from $OR_SRC/engine/source/parallel/asspar.F
    """
    scatter3(target, idx, values)


# ============================================================================
# LAW70 tabulated-foam numeric leaves
# ============================================================================

def law70_tab2d(xg, rates, Y, x, r):
    """Bilinear (strain, rate) lookup with end-slope extrapolation.
    # Ported from $OR_SRC/engine/source/materials/law70/sigeps70.F
    """
    xp = _get_xp(xg, rates, Y, x, r)
    is_np = isinstance(x, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xg_g = cp.asarray(xg)
        rates_g = cp.asarray(rates)
        Y_g = cp.asarray(Y)
        x_g = cp.asarray(x)
        r_g = cp.asarray(r)
    else:
        xg_g = xg
        rates_g = rates
        Y_g = Y
        x_g = x
        r_g = r

    m = xg_g.shape[0]
    nr = rates_g.shape[0]
    i = xp.searchsorted(xg_g, x_g, side="right")
    i = xp.clip(i, 1, m - 1)
    t = (x_g - xg_g[i - 1]) / xp.maximum(xg_g[i] - xg_g[i - 1], 1e-20)

    if nr == 1:
        out = Y_g[i - 1, 0] + t * (Y_g[i, 0] - Y_g[i - 1, 0])
    else:
        j = xp.searchsorted(rates_g, r_g, side="right")
        j = xp.clip(j, 1, nr - 1)
        u = (r_g - rates_g[j - 1]) / xp.maximum(rates_g[j] - rates_g[j - 1], 1e-20)
        y0 = Y_g[i - 1, j - 1] + t * (Y_g[i, j - 1] - Y_g[i - 1, j - 1])
        y1 = Y_g[i - 1, j] + t * (Y_g[i, j] - Y_g[i - 1, j])
        out = y0 + u * (y1 - y0)

    return to_cpu(out) if is_np else out


def law70_enorm(v):
    """Tensor norm of a Voigt strain (0.5 on engineering shears).
    # Ported from $OR_SRC/engine/source/materials/law70/sigeps70.F
    """
    xp = _get_xp(v)
    is_np = isinstance(v, np.ndarray) if hasattr(np, "ndarray") else False
    v_g = cp.asarray(v) if (xp is cp and cp is not None) else v
    out = xp.sqrt(v_g[:, 0]**2 + v_g[:, 1]**2 + v_g[:, 2]**2
                  + 0.5 * (v_g[:, 3]**2 + v_g[:, 4]**2 + v_g[:, 5]**2))
    return to_cpu(out) if is_np else out


def law70_snorm(v):
    """Frobenius norm of a Voigt stress (2.0 on shears).
    # Ported from $OR_SRC/engine/source/materials/law70/sigeps70.F
    """
    xp = _get_xp(v)
    is_np = isinstance(v, np.ndarray) if hasattr(np, "ndarray") else False
    v_g = cp.asarray(v) if (xp is cp and cp is not None) else v
    out = xp.sqrt(v_g[:, 0]**2 + v_g[:, 1]**2 + v_g[:, 2]**2
                  + 2.0 * (v_g[:, 3]**2 + v_g[:, 4]**2 + v_g[:, 5]**2))
    return to_cpu(out) if is_np else out


def law70_elastic_stress(aa1, aa2, g, e):
    """C(E):eps for per-element moduli (Voigt, engineering shear).
    # Ported from $OR_SRC/engine/source/materials/law70/sigeps70.F
    """
    xp = _get_xp(aa1, aa2, g, e)
    is_np = isinstance(e, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        aa1_g = cp.asarray(aa1)
        aa2_g = cp.asarray(aa2)
        g_g = cp.asarray(g)
        e_g = cp.asarray(e)
    else:
        aa1_g = aa1
        aa2_g = aa2
        g_g = g
        e_g = e

    out = xp.empty_like(e_g)
    out[:, 0] = aa1_g * e_g[:, 0] + aa2_g * (e_g[:, 1] + e_g[:, 2])
    out[:, 1] = aa1_g * e_g[:, 1] + aa2_g * (e_g[:, 0] + e_g[:, 2])
    out[:, 2] = aa1_g * e_g[:, 2] + aa2_g * (e_g[:, 0] + e_g[:, 1])
    out[:, 3] = g_g * e_g[:, 3]
    out[:, 4] = g_g * e_g[:, 4]
    out[:, 5] = g_g * e_g[:, 5]

    return to_cpu(out) if is_np else out


# ============================================================================
# solid_tetra10 GPU kernels
# ============================================================================

def tetra10_pre(xe, ve, sig, dt, off):
    """Mirror of solid_tetra10._pre on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/solid/tetra10/t10coor.F
    # Ported from $OR_SRC/engine/source/elements/solid/tetra10/t10defo.F
    """
    xp = _get_xp(xe)
    is_np = isinstance(xe, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xe_g = cp.asarray(xe)
        ve_g = cp.asarray(ve)
        sig_g = cp.asarray(sig)
        off_g = cp.asarray(off)
    else:
        xe_g = xe
        ve_g = ve
        sig_g = sig
        off_g = off

    n = xe_g.shape[0]
    dndx = xp.empty((n, 4, 10, 3), dtype=xe_g.dtype)
    vol = xp.empty((n, 4), dtype=xe_g.dtype)
    deps = xp.empty((n, 4, 6), dtype=xe_g.dtype)
    trD = xp.empty((n, 4), dtype=xe_g.dtype)

    DN_DXI_T10 = xp.asarray(_DN_DXI_T10_DATA, dtype=xe_g.dtype)
    WIP_T10 = xp.asarray(_WIP_T10_DATA, dtype=xe_g.dtype)
    FACES_T10 = xp.asarray(_FACES_T10_DATA, dtype=xp.int64)

    alive = off_g > 0.0
    vmax = xp.max(xp.abs(ve_g), axis=(1, 2))

    for k in range(4):
        DN_k = DN_DXI_T10[k]
        J = DN_k.T @ xe_g  # (n, 3, 3)
        j00, j01, j02 = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
        j10, j11, j12 = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
        j20, j21, j22 = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]

        A = j11 * j22 - j12 * j21
        B = j12 * j20 - j10 * j22
        C = j10 * j21 - j11 * j20
        det = j00 * A + j01 * B + j02 * C
        vol[:, k] = det / 6.0

        idet = xp.divide(1.0, det)
        i00 = A * idet
        i01 = (j02 * j21 - j01 * j22) * idet
        i02 = (j01 * j12 - j02 * j11) * idet
        i10 = B * idet
        i11 = (j00 * j22 - j02 * j20) * idet
        i12 = (j02 * j10 - j00 * j12) * idet
        i20 = C * idet
        i21 = (j01 * j20 - j00 * j21) * idet
        i22 = (j00 * j11 - j01 * j10) * idet

        row0 = xp.stack([i00, i01, i02], axis=-1)
        row1 = xp.stack([i10, i11, i12], axis=-1)
        row2 = xp.stack([i20, i21, i22], axis=-1)
        Jinv = xp.stack([row0, row1, row2], axis=-2)
        dndx[:, k] = DN_k @ Jinv.transpose(0, 2, 1)

        L = ve_g.transpose(0, 2, 1) @ dndx[:, k]
        l00, l01, l02 = L[:, 0, 0], L[:, 0, 1], L[:, 0, 2]
        l10, l11, l12 = L[:, 1, 0], L[:, 1, 1], L[:, 1, 2]
        l20, l21, l22 = L[:, 2, 0], L[:, 2, 1], L[:, 2, 2]

        tr = l00 + l11 + l22
        gmax = xp.max(xp.abs(dndx[:, k]), axis=(1, 2))
        tr = xp.where(xp.abs(tr) <= 1e-14 * (vmax * gmax), 0.0, tr)
        trD[:, k] = xp.where(alive, tr, 0.0)

        deps[:, k, 0] = xp.where(alive, l00 * dt, 0.0)
        deps[:, k, 1] = xp.where(alive, l11 * dt, 0.0)
        deps[:, k, 2] = xp.where(alive, l22 * dt, 0.0)
        deps[:, k, 3] = xp.where(alive, (l01 + l10) * dt, 0.0)
        deps[:, k, 4] = xp.where(alive, (l12 + l21) * dt, 0.0)
        deps[:, k, 5] = xp.where(alive, (l02 + l20) * dt, 0.0)

        wxy = 0.5 * (l01 - l10) * dt
        wyz = 0.5 * (l12 - l21) * dt
        wxz = 0.5 * (l02 - l20) * dt

        sxx = sig_g[:, k, 0].copy()
        syy = sig_g[:, k, 1].copy()
        szz = sig_g[:, k, 2].copy()
        sxy = sig_g[:, k, 3].copy()
        syz_s = sig_g[:, k, 4].copy()
        szx = sig_g[:, k, 5].copy()

        sig_g[:, k, 0] = sxx + 2.0 * (wxy * sxy + wxz * szx)
        sig_g[:, k, 1] = syy + 2.0 * (-wxy * sxy + wyz * syz_s)
        sig_g[:, k, 2] = szz + 2.0 * (-wxz * szx - wyz * syz_s)
        sig_g[:, k, 3] = sxy + wxy * (syy - sxx) + wxz * syz_s + wyz * szx
        sig_g[:, k, 4] = syz_s + wyz * (szz - syy) - wxy * szx - wxz * sxy
        sig_g[:, k, 5] = szx + wxz * (szz - sxx) + wxy * syz_s - wyz * sxy

    vol_tot = xp.maximum(xp.sum(vol * WIP_T10[None, :], axis=1), EM20)
    f0, f1, f2 = FACES_T10[:, 0], FACES_T10[:, 1], FACES_T10[:, 2]
    e1 = xe_g[:, f1, :] - xe_g[:, f0, :]
    e2 = xe_g[:, f2, :] - xe_g[:, f0, :]
    c_prod = xp.cross(e1, e2)
    areas = 0.5 * xp.sqrt(xp.sum(c_prod * c_prod, axis=-1))
    amax = xp.max(areas, axis=1)
    lc = 3.0 * vol_tot / xp.maximum(amax, EM20)

    if is_np:
        sig[...] = to_cpu(sig_g)
        return to_cpu(dndx), to_cpu(vol), to_cpu(vol_tot), to_cpu(lc), to_cpu(deps), to_cpu(trD)
    return dndx, vol, vol_tot, lc, deps, trD


def tetra10_post(xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig, sig_old,
                 qa, qb, c, alive, qvw_pend, dt, dtfac):
    """Mirror of solid_tetra10._post on GPU via CuPy.
    # Ported from $OR_SRC/engine/source/elements/solid/tetra10/t10fint.F
    # Ported from $OR_SRC/engine/source/elements/solid/tetra10/t10dlen.F
    """
    xp = _get_xp(xe)
    is_np = isinstance(xe, np.ndarray) if hasattr(np, "ndarray") else False

    if xp is cp and cp is not None:
        xe_g = cp.asarray(xe)
        dndx_g = cp.asarray(dndx)
        vol_g = cp.asarray(vol)
        vol_tot_g = cp.asarray(vol_tot)
        lc_g = cp.asarray(lc)
        rho_g = cp.asarray(rho)
        trD_g = cp.asarray(trD)
        deps_g = cp.asarray(deps)
        sig_g = cp.asarray(sig)
        sig_old_g = cp.asarray(sig_old)
        qa_g = cp.asarray(qa)
        qb_g = cp.asarray(qb)
        c_g = cp.asarray(c)
        alive_g = cp.asarray(alive)
        qvw_pend_g = cp.asarray(qvw_pend)
        dtfac_g = cp.asarray(dtfac)
    else:
        xe_g = xe
        dndx_g = dndx
        vol_g = vol
        vol_tot_g = vol_tot
        lc_g = lc
        rho_g = rho
        trD_g = trD
        deps_g = deps
        sig_g = sig
        sig_old_g = sig_old
        qa_g = qa
        qb_g = qb
        c_g = c
        alive_g = alive
        qvw_pend_g = qvw_pend
        dtfac_g = dtfac

    n = xe_g.shape[0]
    live = alive_g.astype(bool) if hasattr(alive_g, "astype") else alive_g

    WIP_T10 = xp.asarray(_WIP_T10_DATA, dtype=xe_g.dtype)
    fe = xp.zeros((n, 10, 3), dtype=xe_g.dtype)
    wv = xp.zeros(n, dtype=xe_g.dtype)
    qvn = xp.zeros(n, dtype=xe_g.dtype)
    de = xp.zeros(n, dtype=xe_g.dtype)
    trD_min = trD_g[:, 0].copy()

    for k in range(4):
        compressing = (trD_g[:, k] < 0.0) & live
        qv = xp.where(
            compressing,
            rho_g * lc_g * (qa_g * qa_g * lc_g * trD_g[:, k] * trD_g[:, k] - qb_g * c_g * trD_g[:, k]),
            0.0,
        )

        S_k = xp.empty((n, 3, 3), dtype=xe_g.dtype)
        S_k[:, 0, 0] = sig_g[:, k, 0] - qv
        S_k[:, 1, 1] = sig_g[:, k, 1] - qv
        S_k[:, 2, 2] = sig_g[:, k, 2] - qv
        S_k[:, 0, 1] = S_k[:, 1, 0] = sig_g[:, k, 3]
        S_k[:, 1, 2] = S_k[:, 2, 1] = sig_g[:, k, 4]
        S_k[:, 0, 2] = S_k[:, 2, 0] = sig_g[:, k, 5]

        wv_k = -WIP_T10[k] * vol_g[:, k]
        fe += (dndx_g[:, k] @ S_k) * wv_k[:, None, None]

        wk_vol = WIP_T10[k] * vol_g[:, k]
        wv += wk_vol * 0.5 * qv * (-trD_g[:, k] * dt)
        qvn += wk_vol * 0.5 * qv * dt

        sig_mid_k = 0.5 * (sig_old_g[:, k, :] + sig_g[:, k, :])
        de += wk_vol * xp.sum(sig_mid_k * deps_g[:, k, :], axis=1)

        trD_min = xp.minimum(trD_min, trD_g[:, k])

    sum_neg_trD = xp.sum(-trD_g, axis=1)
    wv += qvw_pend_g * sum_neg_trD / 4.0

    w_visc = wv
    qvw_new = qvn
    deint0 = de + wv

    compressing_any = trD_min < 0.0
    Q = xp.where(compressing_any, qb_g * c_g + qa_g * lc_g * xp.abs(trD_min), 0.0)
    denom = Q + xp.sqrt(Q * Q + c_g * c_g)
    dt_crit = xp.where(live & (denom > EM20), dtfac_g * lc_g / denom, EP30)

    if is_np:
        return to_cpu(fe), to_cpu(dt_crit), to_cpu(w_visc), to_cpu(qvw_new), to_cpu(deint0)
    return fe, dt_crit, w_visc, qvw_new, deint0


# ---- Fallbacks / delegating mirrors for shell formulations -----------------
try:
    from ..jit_kernels.shells_qbat import qbat_pre_flat, qbat_post_flat, qbat_pre, qbat_post
except Exception:
    qbat_pre_flat = None
    qbat_post_flat = None
    qbat_pre = None
    qbat_post = None

try:
    from ..jit_kernels.shells_qeph import qeph_pre, qeph_post
except Exception:
    qeph_pre = None
    qeph_post = None
