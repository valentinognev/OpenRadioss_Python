# pyradioss/engine/sph_engine.py
# Port of engine/source/elements/sph/*.F
"""
Smoothed Particle Hydrodynamics (SPH) Physics Engine.

Faithful port of OpenRadioss Fortran source:
  - weight.F  (lines 33-66: WEIGHT0, lines 78-119: WEIGHT1): cubic B-spline kernel and gradient
  - spdens.F  (lines 101-167, 373-376): density summation rho_a = sum_b m_b W_ab
  - spdefo3.F (lines 63-70) & spdens.F (lines 188-202): rate of deformation D_ab
  - sppro3.F  (lines 72-118) & spforcp.F (lines 240-304): SPH forces (pressure + Monaghan 1992 artificial viscosity)
  - spstab.F  (lines 80-129, 212-250) & spforcp.F (lines 259-277): Monaghan-Gray tensile instability stabilization
  - spcompl.F (lines 156-202, 266-325): zeroth-order Shepard and first-order MLS kernel gradient corrections
  - sphreq.F  (lines 34-40) & mdtsph.F (lines 96-135): critical time step dt = CFL * h / c_s
  - spsym.F   (lines 34-150: SPSYMP): symmetry plane boundary handling and ghost particles
  - sptemp.F  (lines 32-231: SPGRADT, lines 241-483: SPLAPLT, lines 657-784: SPGTSYM): SPH thermal conduction
  - soltosph.F (lines 39-507: SOLTOSPHF, lines 523-1311: SOLTOSPHP), soltospha.F (lines 39-439), soltosph_on1.F: solid-to-SPH adaptive conversion
"""

import numpy as np
from typing import Any, Dict, Optional
from scipy.spatial import cKDTree


def cubic_bspline_kernel(r, h):
    """Cubic B-spline SPH kernel W(r, h) in 3D.

    Port of OpenRadioss engine/source/elements/sph/weight.F lines 33-66 (WEIGHT0).

    In 3D with compact support radius 2h:
      W(r, h) = (1 / (pi * h^3)) * (1 - 1.5*(r/h)^2 + 0.75*(r/h)^3)   if 0 <= r <= h
              = (1 / (4 * pi * h^3)) * (2 - r/h)^3                    if h < r <= 2h
              = 0                                                     if r > 2h

    Args:
        r: Distance between particles (float or np.ndarray >= 0).
        h: Smoothing length (float or np.ndarray > 0).

    Returns:
        Kernel value W(r, h) (float or np.ndarray).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    q = r_arr / h_arr
    inv_pi = 1.0 / np.pi
    h3 = h_arr ** 3

    w = np.zeros_like(q, dtype=np.float64)

    # Branch 1: 0 <= q <= 1 (weight.F lines 52-56)
    m1 = (q >= 0.0) & (q <= 1.0)
    if np.any(m1):
        q1 = q[m1]
        h3_1 = h3[m1] if np.ndim(h_arr) > 0 else h3
        w[m1] = (1.0 - 1.5 * q1 * q1 + 0.75 * q1 * q1 * q1) * inv_pi / h3_1

    # Branch 2: 1 < q <= 2 (weight.F lines 57-62)
    m2 = (q > 1.0) & (q <= 2.0)
    if np.any(m2):
        q2 = q[m2]
        h3_2 = h3[m2] if np.ndim(h_arr) > 0 else h3
        w[m2] = 0.25 * (2.0 - q2) ** 3 * inv_pi / h3_2

    # Branch 3: q > 2 -> 0.0 (weight.F line 63)
    if np.ndim(r) == 0 and np.ndim(h) == 0:
        return float(w)
    return w


def cubic_bspline_grad(rij, h):
    """Gradient of cubic B-spline kernel grad_a W_ab with respect to r_a.

    Port of OpenRadioss engine/source/elements/sph/weight.F lines 78-119 (WEIGHT1).

    grad_a W_ab = (dW/dr / r) * (x_a - x_b) = WPRIMR * rij
    where WPRIMR is:
      (-3 + 2.25 * (r/h)) / (pi * h^5)             if 0 <= r <= h
      -0.75 * (2 - r/h)^2 / (pi * h^4 * r)         if h < r <= 2h
      0                                            if r > 2h

    Args:
        rij: Vector x_i - x_j (shape (3,) or (N, 3)).
        h: Smoothing length (float or np.ndarray of shape (N,)).

    Returns:
        grad_i W_ij (shape matching rij: (3,) or (N, 3)).
    """
    rij_arr = np.asarray(rij, dtype=np.float64)
    is_1d = (rij_arr.ndim == 1)
    if is_1d:
        rij_arr = rij_arr.reshape(1, 3)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(len(rij_arr), float(h_arr))

    r = np.linalg.norm(rij_arr, axis=1)
    wprimr = np.zeros(len(rij_arr), dtype=np.float64)
    inv_pi = 1.0 / np.pi

    # Branch 1: r <= h (weight.F lines 97-103)
    m1 = (r <= h_arr)
    if np.any(m1):
        r1 = r[m1]
        h1 = h_arr[m1]
        rh = r1 / h1
        ih3 = inv_pi / (h1 ** 3)
        wprimr[m1] = (-3.0 + 2.25 * rh) * ih3 / (h1 * h1)

    # Branch 2: h < r <= 2h (weight.F lines 104-110)
    m2 = (r > h_arr) & (r <= 2.0 * h_arr)
    if np.any(m2):
        r2 = r[m2]
        h2 = h_arr[m2]
        rhm = (2.0 - r2 / h2) / h2
        rhm2 = rhm * rhm * inv_pi
        wprimr[m2] = -0.75 * rhm2 / (h2 * h2 * np.maximum(r2, 1e-30))

    # WGRAD = WPRIMR * rij (weight.F lines 115-117)
    grad = wprimr[:, None] * rij_arr
    if is_1d:
        return grad[0]
    return grad


def wendland_c2_kernel(r, h):
    """Wendland C2 smoothing kernel in 3D with compact support radius 2h.

    Normalisation in 3D:
      alpha_3D = 21 / (16 * pi * h^3)
    For q = r / h:
      W(r, h) = alpha_3D * (1 - q/2)^4 * (1 + 2q)    if 0 <= q <= 2
              = 0                                    if q > 2

    Args:
        r: Inter-particle distance (float or np.ndarray >= 0).
        h: Smoothing length (float or np.ndarray > 0).

    Returns:
        Kernel value W(r, h).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    q = r_arr / h_arr
    alpha = 21.0 / (16.0 * np.pi * (h_arr ** 3))

    w = np.zeros_like(q, dtype=np.float64)
    mask = (q >= 0.0) & (q <= 2.0)
    if np.any(mask):
        qm = q[mask]
        alpha_m = alpha[mask] if np.ndim(h_arr) > 0 else alpha
        term1 = (1.0 - 0.5 * qm) ** 4
        term2 = 1.0 + 2.0 * qm
        w[mask] = alpha_m * term1 * term2

    if np.ndim(r) == 0 and np.ndim(h) == 0:
        return float(w)
    return w


def wendland_c2_grad(rij, h):
    """Gradient of Wendland C2 kernel in 3D with respect to r_i.

    grad_i W_ij = (1/r * dW/dr) * rij
    where dW/dr = - (105 / (16 * pi * h^4)) * q * (1 - q/2)^3
    and 1/r * dW/dr = - (105 / (16 * pi * h^5)) * (1 - q/2)^3   for 0 <= q <= 2

    Args:
        rij: Vector x_i - x_j (shape (3,) or (N, 3)).
        h: Smoothing length (float or np.ndarray of shape (N,)).

    Returns:
        grad_i W_ij (shape matching rij: (3,) or (N, 3)).
    """
    rij_arr = np.asarray(rij, dtype=np.float64)
    is_1d = (rij_arr.ndim == 1)
    if is_1d:
        rij_arr = rij_arr.reshape(1, 3)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(len(rij_arr), float(h_arr))

    r = np.linalg.norm(rij_arr, axis=1)
    q = r / h_arr
    coeff = np.zeros(len(rij_arr), dtype=np.float64)

    mask = (q >= 0.0) & (q <= 2.0)
    if np.any(mask):
        qm = q[mask]
        hm = h_arr[mask]
        factor = -105.0 / (16.0 * np.pi * (hm ** 5))
        coeff[mask] = factor * ((1.0 - 0.5 * qm) ** 3)

    grad = coeff[:, None] * rij_arr
    if is_1d:
        return grad[0]
    return grad


def quintic_spline_kernel(r, h):
    """Quintic B-spline SPH kernel in 3D with compact support radius 3h.

    Normalisation in 3D:
      alpha_3D = 1 / (120 * pi * h^3)
    For q = r / h:
      W(r, h) = alpha_3D * ((3-q)^5 - 6*(2-q)^5 + 15*(1-q)^5)  if 0 <= q <= 1
              = alpha_3D * ((3-q)^5 - 6*(2-q)^5)               if 1 < q <= 2
              = alpha_3D * (3-q)^5                             if 2 < q <= 3
              = 0                                              if q > 3

    Args:
        r: Inter-particle distance (float or np.ndarray >= 0).
        h: Smoothing length (float or np.ndarray > 0).

    Returns:
        Kernel value W(r, h).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    q = r_arr / h_arr
    alpha = 1.0 / (120.0 * np.pi * (h_arr ** 3))

    w = np.zeros_like(q, dtype=np.float64)

    m1 = (q >= 0.0) & (q <= 1.0)
    if np.any(m1):
        q1 = q[m1]
        a1 = alpha[m1] if np.ndim(h_arr) > 0 else alpha
        w[m1] = a1 * ((3.0 - q1)**5 - 6.0 * (2.0 - q1)**5 + 15.0 * (1.0 - q1)**5)

    m2 = (q > 1.0) & (q <= 2.0)
    if np.any(m2):
        q2 = q[m2]
        a2 = alpha[m2] if np.ndim(h_arr) > 0 else alpha
        w[m2] = a2 * ((3.0 - q2)**5 - 6.0 * (2.0 - q2)**5)

    m3 = (q > 2.0) & (q <= 3.0)
    if np.any(m3):
        q3 = q[m3]
        a3 = alpha[m3] if np.ndim(h_arr) > 0 else alpha
        w[m3] = a3 * ((3.0 - q3)**5)

    if np.ndim(r) == 0 and np.ndim(h) == 0:
        return float(w)
    return w


def quintic_spline_grad(rij, h):
    """Gradient of Quintic B-spline SPH kernel in 3D with respect to r_i.

    Args:
        rij: Vector x_i - x_j (shape (3,) or (N, 3)).
        h: Smoothing length (float or np.ndarray of shape (N,)).

    Returns:
        grad_i W_ij (shape matching rij: (3,) or (N, 3)).
    """
    rij_arr = np.asarray(rij, dtype=np.float64)
    is_1d = (rij_arr.ndim == 1)
    if is_1d:
        rij_arr = rij_arr.reshape(1, 3)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(len(rij_arr), float(h_arr))

    r = np.linalg.norm(rij_arr, axis=1)
    q = r / h_arr
    alpha = 1.0 / (120.0 * np.pi * (h_arr ** 3))

    dw_dq = np.zeros(len(rij_arr), dtype=np.float64)

    m1 = (q >= 0.0) & (q <= 1.0)
    if np.any(m1):
        q1 = q[m1]
        dw_dq[m1] = -5.0 * (3.0 - q1)**4 + 30.0 * (2.0 - q1)**4 - 75.0 * (1.0 - q1)**4

    m2 = (q > 1.0) & (q <= 2.0)
    if np.any(m2):
        q2 = q[m2]
        dw_dq[m2] = -5.0 * (3.0 - q2)**4 + 30.0 * (2.0 - q2)**4

    m3 = (q > 2.0) & (q <= 3.0)
    if np.any(m3):
        q3 = q[m3]
        dw_dq[m3] = -5.0 * (3.0 - q3)**4

    r_safe = np.maximum(r, 1e-30)
    coeff = (alpha / (h_arr * r_safe)) * dw_dq
    coeff[r < 1e-30] = 0.0

    grad = coeff[:, None] * rij_arr
    if is_1d:
        return grad[0]
    return grad


def sph_shepard_correction(pos, h_arr, mass, rho):
    """Compute Shepard (order 0) kernel correction factors c_i = sum_j V_j W_ij.

    Port of OpenRadioss engine/source/elements/sph/spcompl.F lines 156-202 (NZERO).
    Restores zeroth-order consistency: sum_j V_j W_ij^corr = 1.0.

    Args:
        pos: Particle positions, shape (N, 3).
        h_arr: Smoothing lengths, shape (N,) or float.
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.

    Returns:
        c_i: Shepard normalisation factor for each particle, shape (N,).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))
    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))
    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    vols = m / np.maximum(r_rho, 1e-20)
    # Self-contribution
    c = vols * (1.0 / (np.pi * (h ** 3)))

    if n <= 1:
        return c

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * float(np.max(h)), output_type='ndarray')
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])
        valid = r <= 2.0 * hij
        if np.any(valid):
            iv = i_idx[valid]
            jv = j_idx[valid]
            w = cubic_bspline_kernel(r[valid], hij[valid])
            np.add.at(c, iv, vols[jv] * w)
            np.add.at(c, jv, vols[iv] * w)

    return np.maximum(c, 1e-12)


def sph_mls_gradient_correction(pos, h_arr, mass, rho):
    """Compute Moving Least Squares (order 1) gradient correction matrix M_i.

    Port of OpenRadioss engine/source/elements/sph/spcompl.F lines 266-325 (NUN).
    Correction matrix:
      M_i = - sum_j V_j (grad_i W_ij) (x) (x_i - x_j)
    Corrected gradient:
      grad_tilde_i W_ij = M_i^{-1} grad_i W_ij
    Restores linear consistency: sum_j V_j (x_j - x_i) . grad_tilde_i W_ij = I (3x3 identity).

    Args:
        pos: Particle positions, shape (N, 3).
        h_arr: Smoothing lengths, shape (N,) or float.
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.

    Returns:
        M_inv: Inverse correction matrix array, shape (N, 3, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))
    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))
    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    vols = m / np.maximum(r_rho, 1e-20)
    M = np.zeros((n, 3, 3), dtype=np.float64)

    if n <= 1:
        return np.repeat(np.eye(3)[None, :, :], n, axis=0)

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * float(np.max(h)), output_type='ndarray')
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])
        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            iv = i_idx[valid]
            jv = j_idx[valid]
            rij_v = rij[valid]
            hij_v = hij[valid]
            grad_w = cubic_bspline_grad(rij_v, hij_v)

            outer_i = grad_w[:, :, None] * rij_v[:, None, :]
            term_i = -vols[jv, None, None] * outer_i
            term_j = -vols[iv, None, None] * outer_i

            np.add.at(M, iv, term_i)
            np.add.at(M, jv, term_j)

    M_inv = np.zeros_like(M)
    for i in range(n):
        mat = M[i]
        det = np.linalg.det(mat)
        if abs(det) > 1e-4:
            M_inv[i] = np.linalg.inv(mat)
        else:
            reg = mat + 1e-3 * np.eye(3)
            det_reg = np.linalg.det(reg)
            if abs(det_reg) > 1e-6:
                M_inv[i] = np.linalg.inv(reg)
            else:
                M_inv[i] = np.eye(3)

    return M_inv


def sph_tensile_stabilization(pos, vel, mass, rho, stress, h_arr, zstab=1.0, dp=None, dt=1e-6):
    """Compute Monaghan-Gray tensile instability stabilization forces.

    Faithful port of OpenRadioss Fortran:
      - spstab.F lines 80-129 (SPSTABW): STAB(7, N) = zstab / max(1e-30, W0(dp)^4)
      - spstab.F lines 212-250 (SPSTABS): Principal tensile stress projection C_ij
      - spforcp.F lines 259-277: Stabilization pair forces T_ij

    When particles are under tensile stress (sigma_k > 0), standard SPH exhibits
    unphysical particle clumping / clustering (tensile instability). This algorithm
    introduces an artificial repulsive force along the tensile principal directions
    scaled by (W(r)/W(dp))^4.

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.
        stress: Particle Cauchy stress tensors, shape (N, 3, 3) or (N, 6) [xx, yy, zz, xy, yz, zx].
        h_arr: Smoothing lengths, shape (N,) or float.
        zstab: Tensile stabilization parameter (default 1.0, from /PROP/SPH ZSTAB).
        dp: Initial particle spacing (if None, defaults to 2/3 * h as in spstab.F line 123).
        dt: Time step increment for energy accounting.

    Returns:
        f_stab: Stabilization force array, shape (N, 3).
        w_stab: Numerical work / energy done by stabilization forces.
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n <= 1 or zstab <= 0.0:
        return np.zeros((n, 3), dtype=np.float64), 0.0

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))
    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))
    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    # Convert stress to (N, 3, 3)
    stress_arr = np.asarray(stress, dtype=np.float64)
    if stress_arr.ndim == 2 and stress_arr.shape[1] == 6:
        sig_tensor = np.zeros((n, 3, 3), dtype=np.float64)
        sig_tensor[:, 0, 0] = stress_arr[:, 0]
        sig_tensor[:, 1, 1] = stress_arr[:, 1]
        sig_tensor[:, 2, 2] = stress_arr[:, 2]
        sig_tensor[:, 0, 1] = sig_tensor[:, 1, 0] = stress_arr[:, 3]
        sig_tensor[:, 1, 2] = sig_tensor[:, 2, 1] = stress_arr[:, 4]
        sig_tensor[:, 0, 2] = sig_tensor[:, 2, 0] = stress_arr[:, 5]
    elif stress_arr.ndim == 3 and stress_arr.shape[1:] == (3, 3):
        sig_tensor = stress_arr
    else:
        sig_tensor = np.zeros((n, 3, 3), dtype=np.float64)

    # 1. Compute principal stresses and tensile projection tensor C (spstab.F lines 234-250)
    C_tensor = np.zeros((n, 3, 3), dtype=np.float64)
    stab_active = np.zeros(n, dtype=bool)

    for i in range(n):
        evals, evecs = np.linalg.eigh(sig_tensor[i])
        if np.any(evals > 0.0):
            stab_active[i] = True
            for k in range(3):
                if evals[k] > 0.0:
                    rk = -evals[k]
                    vk = evecs[:, k]
                    C_tensor[i] += rk * np.outer(vk, vk)

    if not np.any(stab_active):
        return np.zeros((n, 3), dtype=np.float64), 0.0

    # 2. Reference kernel value at initial interparticle distance W0(dp) (spstab.F lines 121-126)
    stab7 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if stab_active[i]:
            dd = (dp / h[i]) if (dp is not None and dp > 1e-30) else (2.0 / 3.0)
            w0 = cubic_bspline_kernel(dd, 1.0)
            stab7[i] = zstab / max(1e-30, w0 ** 4)

    # 3. Inter-particle stabilization forces (spforcp.F lines 259-277)
    forces = np.zeros((n, 3), dtype=np.float64)
    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * float(np.max(h)), output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        active_pairs = (stab7[i_idx] > 0.0) & (stab7[j_idx] > 0.0)
        if np.any(active_pairs):
            iv = i_idx[active_pairs]
            jv = j_idx[active_pairs]
            rij = pos_arr[iv] - pos_arr[jv]
            r = np.linalg.norm(rij, axis=1)
            hij = 0.5 * (h[iv] + h[jv])
            valid = (r <= 2.0 * hij) & (r > 1e-30)
            if np.any(valid):
                iv = iv[valid]
                jv = jv[valid]
                rij_v = rij[valid]
                hij_v = hij[valid]
                r_v = r[valid]

                grad_w = cubic_bspline_grad(rij_v, hij_v)
                w_val = cubic_bspline_kernel(r_v, hij_v)

                vol_i = m[iv] / np.maximum(r_rho[iv], 1e-20)
                vol_j = m[jv] / np.maximum(r_rho[jv], 1e-20)
                vij_vol = vol_i * vol_j

                ww = w_val * (hij_v ** 3)
                wr = 0.5 * (stab7[iv] + stab7[jv])
                wi = (ww ** 4) * wr

                cx = np.einsum('ijk,ik->ij', C_tensor[iv], grad_w)
                dx = np.einsum('ijk,ik->ij', C_tensor[jv], grad_w)

                t_force = (vij_vol * wi)[:, None] * (cx + dx)

                np.add.at(forces, iv, t_force)
                np.subtract.at(forces, jv, t_force)

    vel_arr = np.asarray(vel, dtype=np.float64)
    w_stab = float(np.sum(forces * vel_arr) * dt)
    return forces, w_stab


def sph_density_sum(pos, mass, h_arr):
    """Compute SPH density by summation: rho_a = sum_b m_b * W(r_ab, h_ab).

    Port of OpenRadioss engine/source/elements/sph/spdens.F (lines 101-167, 373-376).
    Uses scipy.spatial.cKDTree for efficient neighbor searching within 2h.

    Args:
        pos: Particle positions, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        h_arr: Smoothing lengths, shape (N,) or float.

    Returns:
        rho: Densities, shape (N,).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))

    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    # Self-contribution: W(0, h_a) = 1 / (pi * h_a^3) (weight.F lines 56, 101-102)
    rho = m * (1.0 / (np.pi * h ** 3))

    if n == 1:
        return rho

    max_h = float(np.max(h))
    search_radius = 2.0 * max_h

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)

        # Average smoothing length: DIJ = 0.5 * (DI + DJ) (spdens.F line 147)
        hij = 0.5 * (h[i_idx] + h[j_idx])
        valid = r <= 2.0 * hij
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            r_v = r[valid]
            hij_v = hij[valid]
            w = cubic_bspline_kernel(r_v, hij_v)
            np.add.at(rho, i_v, m[j_v] * w)
            np.add.at(rho, j_v, m[i_v] * w)

    return rho


def sph_defo_rate(pos, vel, mass, rho, h_arr):
    """Compute SPH strain rate tensor D_ab (rate of deformation).

    Port of OpenRadioss engine/source/elements/sph/spdefo3.F (lines 63-70)
    and spdens.F (lines 188-202).

    Velocity gradient tensor:
      L_a = - sum_b (m_b / rho_b) * (v_a - v_b) (x) grad_a W_ab
    Rate of deformation (symmetric strain rate):
      D_a = 0.5 * (L_a + L_a^T)

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.
        h_arr: Smoothing lengths, shape (N,) or float.

    Returns:
        D: Rate of deformation tensor array, shape (N, 3, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))

    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))

    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    L = np.zeros((n, 3, 3), dtype=np.float64)
    if n == 1:
        return L

    max_h = float(np.max(h))
    search_radius = 2.0 * max_h

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])

        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            rij_v = rij[valid]
            hij_v = hij[valid]

            grad_w = cubic_bspline_grad(rij_v, hij_v)
            vij = vel_arr[i_v] - vel_arr[j_v]

            # outer product: v_ij (x) grad_i W_ij (shape (P, 3, 3))
            outer = vij[:, :, None] * grad_w[:, None, :]

            # spdens.F lines 188-202:
            # VJ = m_j / rho_j
            # L_i += - (m_j / rho_j) * (v_i - v_j) (x) grad_i W_ij
            vol_j = m[j_v] / np.maximum(r_rho[j_v], 1e-20)
            vol_i = m[i_v] / np.maximum(r_rho[i_v], 1e-20)

            term_i = -vol_j[:, None, None] * outer
            # For particle j: (v_j - v_i) (x) grad_j W_ji = (-vij) (x) (-grad_w) = vij (x) grad_w
            term_j = -vol_i[:, None, None] * outer

            np.add.at(L, i_v, term_i)
            np.add.at(L, j_v, term_j)

    # Symmetric deformation rate tensor: D = 0.5 * (L + L^T) (spdefo3.F lines 63-70)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    return D


def sph_forces(pos, vel, mass, rho, pressure, h_arr, alpha_visc=1.0, beta_visc=2.0, c_s=None, gamma=1.4, stress=None, zstab=0.0, dp=None):
    """Compute SPH force array (pressure gradient + Monaghan 1992 artificial viscosity + optional tensile stabilization).

    Port of OpenRadioss engine/source/elements/sph/sppro3.F (lines 72-118),
    spforcp.F (lines 240-304), and spstab.F (lines 234-277).

    Pressure force (spforcp.F lines 251-256):
      F_p,ij = - (m_i * m_j / (rho_i * rho_j)) * (p_i + p_j) * grad_i W_ij

    Monaghan 1992 artificial viscosity (spforcp.F lines 280-304):
      mu_ij = h_ij * (v_ij . x_ij) / (|x_ij|^2 + 0.01 * h_ij^2)   if v_ij . x_ij < 0 else 0
      Pi_ij = (beta_visc * mu_ij^2 - alpha_visc * c_s_bar * mu_ij) * 2 / (rho_i + rho_j)
      F_v,ij = - m_i * m_j * Pi_ij * grad_i W_ij

    Tensile stabilization (spstab.F lines 234-277, spforcp.F lines 259-277):
      T_ij = V_i * V_j * W_I * (C_i + C_j) . grad_i W_ij  (if zstab > 0 and stress tensile)

    Total force:
      F_i = sum_j (F_p,ij + F_v,ij + T_ij)
      By Newton's third law, F_ji = - F_ij, guaranteeing exact momentum conservation.

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.
        pressure: Particle pressures, shape (N,) or float.
        h_arr: Smoothing lengths, shape (N,) or float.
        alpha_visc: Monaghan linear viscosity coefficient (default 1.0).
        beta_visc: Monaghan quadratic viscosity coefficient (default 2.0).
        c_s: Sound speed array, shape (N,) or None (computed from EOS if None).
        gamma: Ratio of specific heats for ideal gas (default 1.4).
        stress: Cauchy stress tensor array (N, 3, 3) or (N, 6) for tensile stabilization.
        zstab: Tensile stabilization parameter (default 0.0, disabled).
        dp: Initial particle spacing (default 2/3 * h).

    Returns:
        forces: Force array, shape (N, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    n = len(pos_arr)
    forces = np.zeros((n, 3), dtype=np.float64)
    if n <= 1:
        return forces

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))

    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))

    p = np.asarray(pressure, dtype=np.float64)
    if p.ndim == 0:
        p = np.full(n, float(p))

    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    # Sound speed: c_s = sqrt(gamma * max(p, 0) / rho) (mdtsph.F line 98)
    if c_s is None:
        c_s = np.sqrt(np.maximum(gamma * p / np.maximum(r_rho, 1e-20), 1e-20))
    else:
        c_s = np.asarray(c_s, dtype=np.float64)
        if c_s.ndim == 0:
            c_s = np.full(n, float(c_s))

    max_h = float(np.max(h))
    search_radius = 2.0 * max_h

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])

        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            rij_v = rij[valid]
            r_v = r[valid]
            hij_v = hij[valid]

            grad_w = cubic_bspline_grad(rij_v, hij_v)

            # 1. Pressure forces (spforcp.F lines 251-256):
            # VIJ = (m_i / rho_i) * (m_j / rho_j)
            vol_i = m[i_v] / np.maximum(r_rho[i_v], 1e-20)
            vol_j = m[j_v] / np.maximum(r_rho[j_v], 1e-20)
            vij_vol = vol_i * vol_j
            # AX + BX = - (p_i + p_j) * grad_W
            f_p_coeff = -vij_vol * (p[i_v] + p[j_v])
            f_p = f_p_coeff[:, None] * grad_w

            # 2. Monaghan 1992 artificial viscosity (spforcp.F lines 280-304):
            vij_vel = vel_arr[i_v] - vel_arr[j_v]
            vr_dot = np.sum(vij_vel * rij_v, axis=1)

            # Viscosity active only under compression: vr_dot < 0
            compress = vr_dot < 0.0
            f_v = np.zeros_like(f_p)
            if np.any(compress):
                i_c = i_v[compress]
                j_c = j_v[compress]
                hij_c = hij_v[compress]
                rij_c = rij_v[compress]
                r_c = r_v[compress]
                vr_c = vr_dot[compress]
                grad_w_c = grad_w[compress]

                # mu_ij = hij * (v_ij . r_ij) / (|r_ij|^2 + 0.01 * hij^2) (spforcp.F lines 286, 290)
                mu_ij = hij_c * vr_c / (r_c * r_c + 0.01 * hij_c * hij_c)
                ssp_bar = 0.5 * (c_s[i_c] + c_s[j_c])
                rho_bar = 0.5 * (r_rho[i_c] + r_rho[j_c])

                # PIJ = (QA * mu^2 - QB * ssp * mu) / rho_bar (spforcp.F line 294)
                # QA = beta_visc, QB = alpha_visc
                pij = (beta_visc * mu_ij * mu_ij - alpha_visc * ssp_bar * mu_ij) / np.maximum(rho_bar, 1e-20)

                # FACT = m_i * m_j * PIJ (spforcp.F line 295)
                # FV = -FACT * grad_W (spforcp.F lines 301-303)
                f_v_coeff = - (m[i_c] * m[j_c] * pij)
                f_v[compress] = f_v_coeff[:, None] * grad_w_c

            f_total = f_p + f_v
            np.add.at(forces, i_v, f_total)
            np.subtract.at(forces, j_v, f_total)

    # 3. Monaghan-Gray tensile instability stabilization (spstab.F lines 234-277, spforcp.F lines 259-277)
    if zstab > 0.0 and stress is not None:
        f_stab, _ = sph_tensile_stabilization(pos, vel, mass, rho, stress, h_arr, zstab=zstab, dp=dp, dt=0.0)
        forces += f_stab

    return forces


def sph_critical_dt(h_arr, rho, pressure, gamma, cfl=0.6):
    """Compute SPH critical time step from CFL stability condition.

    Port of OpenRadioss engine/source/elements/sph/sphreq.F and mdtsph.F (lines 96-135).
    dt = cfl * h / c_s, where c_s = sqrt(gamma * p / rho).

    Args:
        h_arr: Smoothing lengths (float or np.ndarray).
        rho: Densities (float or np.ndarray).
        pressure: Pressures (float or np.ndarray).
        gamma: Ratio of specific heats for ideal gas EOS (LAW6).
        cfl: Courant-Friedrichs-Lewy safety coefficient (default 0.6).

    Returns:
        dt: Critical time step (float or np.ndarray).
    """
    h = np.asarray(h_arr, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)

    # Sound speed: c_s = sqrt(gamma * max(p, 0) / max(rho, 1e-20)) (mdtsph.F line 98)
    c_s = np.sqrt(np.maximum(gamma * p / np.maximum(r, 1e-20), 1e-20))
    dt = cfl * h / c_s

    if np.ndim(h_arr) == 0 and np.ndim(rho) == 0 and np.ndim(pressure) == 0:
        return float(dt)
    return dt


def sph_step(model, dt, state, fint=None):
    """Execute one SPH time step and assemble forces into the model.

    Integrates SPH physics into the explicit time loop of pyradioss/engine/engine.py:
      1. Evaluates density by summation (spdens.F)
      2. Computes pressure from EOS / material state (LAW6 / perfect gas)
      3. Computes internal and artificial viscous forces (sppro3.F, spforcp.F)
      4. Assembles forces into fint
      5. Books numerical dissipation / energy into state (EN ledger)

    Args:
        model: Radioss Model instance with sph_cells.
        dt: Current time step increment.
        state: EngineState instance for energy tracking.
        fint: Optional global internal force array (if None, taken from model.fint).
    """
    if not hasattr(model, 'sph_cells') or not model.sph_cells:
        return

    cells = model.sph_cells

    # Identify node indices for SPH particles
    if hasattr(cells, 'conn'):
        node_idx = cells.conn.reshape(-1)
    elif hasattr(cells, 'node_ids'):
        node_idx = np.asarray(cells.node_ids, dtype=np.int64)
    elif hasattr(cells, 'ids'):
        node_idx = np.asarray(cells.ids, dtype=np.int64)
    else:
        node_idx = np.arange(len(cells))

    n = len(node_idx)
    if n == 0:
        return

    # Extract positions, velocities, masses
    if hasattr(cells, 'pos'):
        pos = cells.pos
    elif hasattr(model, 'x'):
        pos = model.x[node_idx]
    else:
        return

    if hasattr(cells, 'vel'):
        vel = cells.vel
    elif hasattr(model, 'v'):
        vel = model.v[node_idx]
    else:
        vel = np.zeros_like(pos)

    if hasattr(cells, 'mass'):
        mass = cells.mass
    elif hasattr(model, 'mass'):
        mass = model.mass[node_idx]
    else:
        mass = np.ones(n, dtype=np.float64)

    # Smoothing lengths
    if hasattr(cells, 'h'):
        h_arr = cells.h
    elif hasattr(cells, 'state') and "h" in cells.state:
        h_arr = cells.state["h"]
    else:
        h_arr = np.full(n, 0.1, dtype=np.float64)

    # Density computation: summation
    rho = sph_density_sum(pos, mass, h_arr)
    if hasattr(cells, 'state'):
        cells.state["rho"] = rho

    # Pressure from EOS (LAW6: p = (gamma - 1) * rho * u)
    gamma = getattr(cells, 'gamma', 1.4)
    if hasattr(cells, 'pressure'):
        pressure = cells.pressure
    elif hasattr(cells, 'state') and "pressure" in cells.state:
        pressure = cells.state["pressure"]
    elif hasattr(cells, 'state') and "u" in cells.state:
        pressure = (gamma - 1.0) * rho * cells.state["u"]
    else:
        pressure = np.zeros(n, dtype=np.float64)

    # Viscosity parameters
    alpha = getattr(cells, 'alpha_visc', 1.0)
    beta = getattr(cells, 'beta_visc', 2.0)

    # Tensile stabilization parameters (spstab.F)
    stress = getattr(cells, 'stress', None)
    if stress is None and hasattr(cells, 'state') and "stress" in cells.state:
        stress = cells.state["stress"]
    zstab = getattr(cells, 'zstab', 0.0)
    if zstab == 0.0 and hasattr(cells, 'state') and "zstab" in cells.state:
        zstab = float(cells.state["zstab"])
    dp = getattr(cells, 'dp', None)

    # Compute SPH forces
    f_sph = sph_forces(
        pos, vel, mass, rho, pressure, h_arr,
        alpha_visc=alpha, beta_visc=beta, gamma=gamma,
        stress=stress, zstab=zstab, dp=dp,
    )

    # Assemble into fint
    if fint is None:
        fint = getattr(model, 'fint', None)
    if fint is not None:
        fint[node_idx] += f_sph

    # Book artificial viscosity work into numerical dissipation ledger (EN)
    # Energy accounting: spforcp.F lines 519-521: WVIS = 0.5 * (FV . (VI - VJ))
    if state is not None:
        # Rate of work on particles: W_dot = sum(F_sph . v)
        w_sph = float(np.sum(f_sph * vel) * dt)
        if hasattr(state, 'e_num'):
            state.e_num = getattr(state, 'e_num', 0.0) + w_sph


# ---------------------------------------------------------------------------
# SPH Symmetry Planes and Ghost Particles
# ---------------------------------------------------------------------------
# Ported from $OR_SRC/engine/source/elements/sph/spsym.F
# (lines 34-150: SPSYMP) and sptemp.F (lines 657-784: SPGTSYM)


class SPHSymmetryPlane:
    """SPH symmetry plane boundary.

    Ported from OpenRadioss engine/source/elements/sph/spsym.F (lines 34-150: SPSYMP)
    and engine/source/elements/sph/sptemp.F (lines 657-784: SPGTSYM).

    Handles planar symmetry boundaries with slip or no-slip condition:
      - Slip condition (ISLIDE=1, default): normal velocity component is reversed (v_n -> -v_n),
        tangential velocity component is preserved (v_t -> v_t).
      - No-slip condition (ISLIDE=0): full velocity vector is reversed (v -> -v).
    """

    def __init__(self, point, normal, islide: int = 1):
        """Initialize symmetry plane.

        Args:
            point: (3,) array-like, point x0 on the symmetry plane.
            normal: (3,) array-like, normal unit vector n directed into the fluid domain.
            islide: Boundary slip flag: 1 for slip (default), 0 for no-slip.
        """
        self.point = np.asarray(point, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-15:
            raise ValueError("Symmetry plane normal vector must be non-zero.")
        self.normal = n / norm_n
        self.islide = int(islide)

    def signed_distance(self, pos: np.ndarray) -> np.ndarray:
        """Compute signed distance of particles to the symmetry plane.

        d = (x - x0) . n  (spsym.F line 116)
        Positive distance indicates the particle is on the active domain side.

        Args:
            pos: Particle positions, shape (N, 3) or (3,).

        Returns:
            d: Signed distance, shape (N,) or float.
        """
        pos_arr = np.asarray(pos, dtype=np.float64)
        return np.sum((pos_arr - self.point) * self.normal, axis=-1)

    def reflect_position(self, pos: np.ndarray) -> np.ndarray:
        """Compute position of mirrored ghost particles across the plane.

        x_s = x - 2 * d * n  (spsym.F lines 119-121)

        Args:
            pos: Particle positions, shape (N, 3) or (3,).

        Returns:
            pos_sym: Mirrored positions, shape matching pos.
        """
        pos_arr = np.asarray(pos, dtype=np.float64)
        d = self.signed_distance(pos_arr)
        if pos_arr.ndim == 1:
            return pos_arr - 2.0 * d * self.normal
        return pos_arr - 2.0 * d[:, None] * self.normal

    def reflect_velocity(self, vel: np.ndarray) -> np.ndarray:
        """Compute velocity of mirrored ghost particles across the plane.

        Port of spsym.F lines 122-131:
          - If ISLIDE == 0 (no-slip): v_s = -v
          - If ISLIDE == 1 (slip): vn = v . n; v_s = v - 2 * vn * n

        Args:
            vel: Particle velocities, shape (N, 3) or (3,).

        Returns:
            vel_sym: Mirrored velocities, shape matching vel.
        """
        vel_arr = np.asarray(vel, dtype=np.float64)
        if self.islide == 0:
            return -vel_arr
        vn = np.sum(vel_arr * self.normal, axis=-1)
        if vel_arr.ndim == 1:
            return vel_arr - 2.0 * vn * self.normal
        return vel_arr - 2.0 * vn[:, None] * self.normal

    def reflect_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Reflect field gradient (e.g. temperature gradient) across symmetry plane.

        Port of OpenRadioss sptemp.F lines 742-751 (SPGTSYM):
          gn = g . n
          g_sym = g - 2 * gn * n

        Args:
            grad: Field gradient vectors, shape (N, 3) or (3,).

        Returns:
            grad_sym: Reflected gradients, shape matching grad.
        """
        g_arr = np.asarray(grad, dtype=np.float64)
        gn = np.sum(g_arr * self.normal, axis=-1)
        if g_arr.ndim == 1:
            return g_arr - 2.0 * gn * self.normal
        return g_arr - 2.0 * gn[:, None] * self.normal


def reflect_sph_gradient(grad: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect vector or scalar gradient across a plane with given normal.

    Port of OpenRadioss engine/source/elements/sph/sptemp.F lines 742-751 (SPGTSYM).

    g_sym = g - 2 * (g . n) * n

    Args:
        grad: Gradient array, shape (N, 3) or (3,).
        normal: Plane normal, shape (3,).

    Returns:
        grad_sym: Reflected gradient array.
    """
    g_arr = np.asarray(grad, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    gn = np.sum(g_arr * n, axis=-1)
    if g_arr.ndim == 1:
        return g_arr - 2.0 * gn * n
    return g_arr - 2.0 * gn[:, None] * n


def create_sph_ghost_particles(pos, vel, mass, rho, h_arr, planes, pressure=None, temp=None, cutoff=None):
    """Generate mirrored ghost particles near symmetry planes.

    Port of OpenRadioss engine/source/elements/sph/spsym.F lines 97-150.
    For each symmetry plane, particles within the compact support cutoff (0 < d <= 2h)
    generate mirror ghost particles with reflected positions and velocities.

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,).
        rho: Particle densities, shape (N,).
        h_arr: Smoothing lengths, shape (N,).
        planes: List of SPHSymmetryPlane instances.
        pressure: Optional particle pressures, shape (N,).
        temp: Optional particle temperatures, shape (N,).
        cutoff: Optional custom cutoff distance array or scalar (default 2*h).

    Returns:
        ghost_dict: Dictionary containing:
          - 'pos': Ghost particle positions (M, 3)
          - 'vel': Ghost particle velocities (M, 3)
          - 'mass': Ghost particle masses (M,)
          - 'rho': Ghost particle densities (M,)
          - 'h': Ghost particle smoothing lengths (M,)
          - 'pressure': Ghost particle pressures (M,) or None
          - 'temp': Ghost particle temperatures (M,) or None
          - 'source_idx': Original particle index for each ghost (M,)
    """
    if not planes:
        return {
            'pos': np.empty((0, 3)), 'vel': np.empty((0, 3)),
            'mass': np.empty(0), 'rho': np.empty(0), 'h': np.empty(0),
            'pressure': None if pressure is None else np.empty(0),
            'temp': None if temp is None else np.empty(0),
            'source_idx': np.empty(0, dtype=np.int64),
        }

    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    m_arr = np.asarray(mass, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    h_a = np.asarray(h_arr, dtype=np.float64)

    p_arr = np.asarray(pressure, dtype=np.float64) if pressure is not None else None
    t_arr = np.asarray(temp, dtype=np.float64) if temp is not None else None

    all_pos, all_vel, all_m, all_rho, all_h, all_idx = [], [], [], [], [], []
    all_p = [] if p_arr is not None else None
    all_t = [] if t_arr is not None else None

    for plane in planes:
        d = plane.signed_distance(pos_arr)
        max_d = 2.0 * h_a if cutoff is None else cutoff
        mask = (d > 1e-12) & (d <= max_d)
        if not np.any(mask):
            continue

        src_idx = np.where(mask)[0]
        g_pos = plane.reflect_position(pos_arr[src_idx])
        g_vel = plane.reflect_velocity(vel_arr[src_idx])

        all_pos.append(g_pos)
        all_vel.append(g_vel)
        all_m.append(m_arr[src_idx])
        all_rho.append(rho_arr[src_idx])
        all_h.append(h_a[src_idx])
        all_idx.append(src_idx)

        if p_arr is not None:
            all_p.append(p_arr[src_idx])
        if t_arr is not None:
            all_t.append(t_arr[src_idx])

    if not all_pos:
        return {
            'pos': np.empty((0, 3)), 'vel': np.empty((0, 3)),
            'mass': np.empty(0), 'rho': np.empty(0), 'h': np.empty(0),
            'pressure': None if pressure is None else np.empty(0),
            'temp': None if temp is None else np.empty(0),
            'source_idx': np.empty(0, dtype=np.int64),
        }

    return {
        'pos': np.concatenate(all_pos, axis=0),
        'vel': np.concatenate(all_vel, axis=0),
        'mass': np.concatenate(all_m, axis=0),
        'rho': np.concatenate(all_rho, axis=0),
        'h': np.concatenate(all_h, axis=0),
        'pressure': np.concatenate(all_p, axis=0) if all_p is not None else None,
        'temp': np.concatenate(all_t, axis=0) if all_t is not None else None,
        'source_idx': np.concatenate(all_idx, axis=0),
    }


# ---------------------------------------------------------------------------
# SPH Thermal Conduction (sptemp.F)
# ---------------------------------------------------------------------------
# Ported from $OR_SRC/engine/source/elements/sph/sptemp.F
# (lines 32-231: SPGRADT, lines 241-483: SPLAPLT, lines 657-784: SPGTSYM, lines 790-825: SPTEMPEL)


def sph_temperature_gradient(pos, temp, mass, rho, h_arr, planes=None):
    """Compute SPH temperature gradient grad(T) for thermal conduction.

    Port of OpenRadioss engine/source/elements/sph/sptemp.F lines 32-231 (SPGRADT).

    Discretization:
      grad(T)_i = sum_j (m_j / rho_j) * (T_j - T_i) * grad_i W_ij

    Args:
        pos: Particle positions, shape (N, 3).
        temp: Particle temperatures, shape (N,).
        mass: Particle masses, shape (N,).
        rho: Particle densities, shape (N,).
        h_arr: Smoothing lengths, shape (N,).
        planes: Optional list of SPHSymmetryPlane instances.

    Returns:
        grad_t: Temperature gradient vectors, shape (N, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    t_arr = np.asarray(temp, dtype=np.float64)
    m_arr = np.asarray(mass, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    h_a = np.asarray(h_arr, dtype=np.float64)

    n = len(pos_arr)
    grad_t = np.zeros((n, 3), dtype=np.float64)
    if n <= 1:
        return grad_t

    # Internal pair interactions
    max_h = float(np.max(h_a))
    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * max_h, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h_a[i_idx] + h_a[j_idx])

        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            rij_v = rij[valid]
            hij_v = hij[valid]

            grad_w = cubic_bspline_grad(rij_v, hij_v)
            v_j = m_arr[j_v] / np.maximum(rho_arr[j_v], 1e-20)
            v_i = m_arr[i_v] / np.maximum(rho_arr[i_v], 1e-20)
            dt_ij = t_arr[j_v] - t_arr[i_v]

            term_i = (v_j * dt_ij)[:, None] * grad_w
            term_j = (v_i * (-dt_ij))[:, None] * (-grad_w)

            np.add.at(grad_t, i_v, term_i)
            np.add.at(grad_t, j_v, term_j)

    # Symmetry plane ghost particle interactions
    if planes:
        ghosts = create_sph_ghost_particles(pos_arr, np.zeros_like(pos_arr), m_arr, rho_arr, h_a, planes, temp=t_arr)
        if len(ghosts['pos']) > 0:
            tree_g = cKDTree(ghosts['pos'])
            g_pairs = tree.query_ball_tree(tree_g, r=2.0 * max_h)
            for i_p, g_indices in enumerate(g_pairs):
                if not g_indices:
                    continue
                g_idx = np.array(g_indices, dtype=np.int64)
                rij_g = pos_arr[i_p] - ghosts['pos'][g_idx]
                r_g = np.linalg.norm(rij_g, axis=1)
                hij_g = 0.5 * (h_a[i_p] + ghosts['h'][g_idx])
                val_g = (r_g <= 2.0 * hij_g) & (r_g > 1e-30)
                if np.any(val_g):
                    sel_g = g_idx[val_g]
                    rij_sel = rij_g[val_g]
                    hij_sel = hij_g[val_g]
                    grad_wg = cubic_bspline_grad(rij_sel, hij_sel)
                    v_g = ghosts['mass'][sel_g] / np.maximum(ghosts['rho'][sel_g], 1e-20)
                    dt_g = ghosts['temp'][sel_g] - t_arr[i_p]
                    np.add.at(grad_t, i_p, np.sum((v_g * dt_g)[:, None] * grad_wg, axis=0))

    return grad_t


def sph_thermal_conduction(pos, temp, mass, rho, h_arr, conductivity, specific_heat, dt=None, planes=None):
    """Compute SPH thermal conduction rate and update temperature.

    Faithful port of OpenRadioss engine/source/elements/sph/sptemp.F lines 241-483 (SPLAPLT).
    Discretizes heat diffusion equation:
      rho * c_v * dT/dt = div(k * grad(T))
    using the conservative Cleary & Monaghan (1999) / Brookshaw formulation:
      dT_i/dt = (1 / (rho_i * c_v,i)) * sum_j (m_j / rho_j) * (4 * k_i * k_j / (k_i + k_j))
                * (T_i - T_j) * (rij . grad_i W_ij) / (|rij|^2 + 0.01 * hij^2)

    Conservation properties:
      The pairwise heat exchange q_ij = - q_ji, guaranteeing exact thermal energy conservation:
        sum_i m_i * c_v,i * dT_i/dt = 0

    Args:
        pos: Particle positions, shape (N, 3).
        temp: Particle temperatures, shape (N,).
        mass: Particle masses, shape (N,).
        rho: Particle densities, shape (N,).
        h_arr: Smoothing lengths, shape (N,).
        conductivity: Thermal conductivity k (float or (N,)).
        specific_heat: Specific heat capacity c_v (float or (N,)).
        dt: Optional time step increment to advance temperature: T_new = T + dt * dT_dt.
        planes: Optional list of SPHSymmetryPlane instances.

    Returns:
        dT_dt: Temperature time derivative, shape (N,).
        q_rates: Heat rate on each particle dQ/dt = m * c_v * dT/dt, shape (N,).
        temp_new: Updated temperature array (if dt is provided) or current temp.
        e_exchange: Total rate of heat energy transferred across particle pairs.
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    t_arr = np.asarray(temp, dtype=np.float64)
    m_arr = np.asarray(mass, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    h_a = np.asarray(h_arr, dtype=np.float64)

    n = len(pos_arr)
    dT_dt = np.zeros(n, dtype=np.float64)
    q_rates = np.zeros(n, dtype=np.float64)
    if n <= 1:
        t_new = t_arr.copy() if dt is not None else t_arr
        return dT_dt, q_rates, t_new, 0.0

    k_arr = np.asarray(conductivity, dtype=np.float64)
    if k_arr.ndim == 0:
        k_arr = np.full(n, float(k_arr))

    cv_arr = np.asarray(specific_heat, dtype=np.float64)
    if cv_arr.ndim == 0:
        cv_arr = np.full(n, float(cv_arr))

    max_h = float(np.max(h_a))
    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * max_h, output_type='ndarray')

    e_exchange = 0.0

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h_a[i_idx] + h_a[j_idx])

        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            rij_v = rij[valid]
            r_v = r[valid]
            hij_v = hij[valid]

            grad_w = cubic_bspline_grad(rij_v, hij_v)
            r_dot_grad = np.sum(rij_v * grad_w, axis=1)

            # Harmonic mean thermal conductivity: 4 * ki * kj / (ki + kj) (sptemp.F line 406)
            k_eff = 4.0 * k_arr[i_v] * k_arr[j_v] / np.maximum(k_arr[i_v] + k_arr[j_v], 1e-20)
            vol_i = m_arr[i_v] / np.maximum(rho_arr[i_v], 1e-20)
            vol_j = m_arr[j_v] / np.maximum(rho_arr[j_v], 1e-20)

            # Pairwise heat rate from particle j to particle i:
            # Note r_dot_grad < 0. When Ti > Tj, Ti - Tj > 0, so (Ti - Tj) * r_dot_grad < 0.
            # Heat leaves particle i: q_ij = vol_i * vol_j * k_eff * (Ti - Tj) * r_dot_grad / (r^2 + 0.01*h^2)
            denom = r_v * r_v + 0.01 * hij_v * hij_v
            q_ij = vol_i * vol_j * k_eff * (t_arr[i_v] - t_arr[j_v]) * r_dot_grad / denom

            np.add.at(q_rates, i_v, q_ij)
            np.add.at(q_rates, j_v, -q_ij)
            e_exchange += float(np.sum(np.abs(q_ij)))

    # Symmetry plane interactions
    if planes:
        ghosts = create_sph_ghost_particles(pos_arr, np.zeros_like(pos_arr), m_arr, rho_arr, h_a, planes, temp=t_arr)
        if len(ghosts['pos']) > 0:
            tree_g = cKDTree(ghosts['pos'])
            g_pairs = tree.query_ball_tree(tree_g, r=2.0 * max_h)
            for i_p, g_indices in enumerate(g_pairs):
                if not g_indices:
                    continue
                g_idx = np.array(g_indices, dtype=np.int64)
                rij_g = pos_arr[i_p] - ghosts['pos'][g_idx]
                r_g = np.linalg.norm(rij_g, axis=1)
                hij_g = 0.5 * (h_a[i_p] + ghosts['h'][g_idx])
                val_g = (r_g <= 2.0 * hij_g) & (r_g > 1e-30)
                if np.any(val_g):
                    sel_g = g_idx[val_g]
                    rij_sel = rij_g[val_g]
                    r_sel = r_g[val_g]
                    hij_sel = hij_g[val_g]
                    grad_wg = cubic_bspline_grad(rij_sel, hij_sel)
                    r_dot_g = np.sum(rij_sel * grad_wg, axis=1)

                    k_eff_g = 4.0 * k_arr[i_p] * k_arr[ghosts['source_idx'][sel_g]] / np.maximum(
                        k_arr[i_p] + k_arr[ghosts['source_idx'][sel_g]], 1e-20
                    )
                    vol_i = m_arr[i_p] / np.maximum(rho_arr[i_p], 1e-20)
                    vol_g = ghosts['mass'][sel_g] / np.maximum(ghosts['rho'][sel_g], 1e-20)
                    denom_g = r_sel * r_sel + 0.01 * hij_sel * hij_sel
                    q_g = vol_i * vol_g * k_eff_g * (t_arr[i_p] - ghosts['temp'][sel_g]) * r_dot_g / denom_g
                    q_rates[i_p] += np.sum(q_g)

    # Temperature rate of change: dT/dt = (dQ/dt) / (m * cv)
    heat_capacity = m_arr * cv_arr
    dT_dt = q_rates / np.maximum(heat_capacity, 1e-20)

    t_new = t_arr.copy()
    if dt is not None and dt > 0.0:
        t_new += dt * dT_dt

    return dT_dt, q_rates, t_new, e_exchange


# ---------------------------------------------------------------------------
# Solid-to-SPH Adaptive Conversion (soltosph*.F)
# ---------------------------------------------------------------------------
# Ported from $OR_SRC/engine/source/elements/sph/soltosph.F
# (lines 39-507: SOLTOSPHF, lines 523-1311: SOLTOSPHP)
# and soltospha.F (lines 39-439: SOLTOSPHA), soltosph_on1.F (lines 37-275: SOLTOSPH_ON1)


# Gauss / sub-particle coordinate matrix for Hexahedron (soltosph.F lines 93-120)
# A_GAUSS(particle_index, n_dir): coordinate in [-1, 1]
HEX_GAUSS_COORDS = {
    1: np.array([0.0]),
    2: np.array([-0.5, 0.5]),
    3: np.array([-2.0 / 3.0, 0.0, 2.0 / 3.0]),
}

# Barycentric coordinate distribution for Tetrahedron (soltosph.F lines 122-150)
TET_BARYCENTRIC_COORDS = {
    1: np.array([[0.25, 0.25, 0.25, 0.25]]),
    2: np.array([
        [0.583333333333333, 0.138888888888889, 0.138888888888889, 0.138888888888889],
        [0.138888888888889, 0.583333333333333, 0.138888888888889, 0.138888888888889],
        [0.138888888888889, 0.138888888888889, 0.583333333333333, 0.138888888888889],
        [0.138888888888889, 0.138888888888889, 0.138888888888889, 0.583333333333333],
    ]),
}


def hex8_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate 8-node trilinear hexahedron shape functions at reference coords (xi, eta, zeta).

    Port of OpenRadioss engine/source/elements/sph/soltosph.F lines 282-289:
      phi_k = (1/8) * (1 +- xi) * (1 +- eta) * (1 +- zeta)
    """
    signs = np.array([
        [-1, -1, -1],  # node 1
        [-1, -1,  1],  # node 2
        [ 1, -1,  1],  # node 3
        [ 1, -1, -1],  # node 4
        [-1,  1, -1],  # node 5
        [-1,  1,  1],  # node 6
        [ 1,  1,  1],  # node 7
        [ 1,  1, -1],  # node 8
    ], dtype=np.float64)

    phi = 0.125 * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta)
    return phi


def tet4_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate 4-node linear tetrahedron shape functions.

    Port of OpenRadioss engine/source/elements/sph/soltosph.F lines 221-224:
      phi_1 = xi, phi_2 = eta, phi_3 = zeta, phi_4 = 1 - xi - eta - zeta
    """
    return np.array([xi, eta, zeta, 1.0 - xi - eta - zeta], dtype=np.float64)


def interpolate_solid_field(solid_type: str, node_values: np.ndarray, xi: float, eta: float, zeta: float) -> np.ndarray:
    """Interpolate nodal values (coordinates, velocities) inside solid element to sub-particle point.

    Port of OpenRadioss engine/source/elements/sph/soltospha.F lines 245-247 and 318-323.

    Args:
        solid_type: 'hex8' (or 'brick') or 'tet4' (or 'tetra').
        node_values: Nodal values, shape (8, D) or (4, D).
        xi, eta, zeta: Reference coordinates in parent element.

    Returns:
        val: Interpolated value, shape (D,).
    """
    vals = np.asarray(node_values, dtype=np.float64)
    stype = solid_type.lower()
    if 'hex' in stype or 'brick' in stype:
        phi = hex8_shape_functions(xi, eta, zeta)
    elif 'tet' in stype:
        phi = tet4_shape_functions(xi, eta, zeta)
    else:
        raise ValueError(f"Unsupported solid element type for SPH conversion: {solid_type}")
    return np.tensordot(phi, vals, axes=(0, 0))


class SolidToSPHConverter:
    """Solid-to-SPH adaptive element conversion manager.

    Ported from OpenRadioss Fortran sources:
      - engine/source/elements/sph/soltosph.F (SOLTOSPHF, SOLTOSPHP)
      - engine/source/elements/sph/soltospha.F (SOLTOSPHA)
      - engine/source/elements/sph/soltosph_on1.F (SOLTOSPH_ON1)

    When solid elements (bricks or tetrahedra) undergo extreme deformation, failure,
    or contact penetration, they are adaptively replaced by clouds of active SPH particles.
    Preserves:
      - Total mass: sum(m_p) = M_solid
      - Total momentum: sum(m_p * v_p) = P_solid
      - Internal energy and stress state
    Dissipated kinetic energy from remeshing is booked into hourglass/numerical dissipation (EN ledger):
      E_hour += 0.5 * M_solid * v_solid^2 - E_k,sph (soltosph_on1.F line 265).
    """

    def __init__(self, n_dir: int = 2, h_factor: float = 1.2):
        """Initialize converter.

        Args:
            n_dir: Particles per spatial direction (default 2 -> 2^3=8 particles per hex).
            h_factor: Smoothing length multiplier wrt particle spacing (default 1.2).
        """
        self.n_dir = int(n_dir)
        self.h_factor = float(h_factor)
        self.converted_solids: set = set()
        self.accumulated_e_hour: float = 0.0

    def convert_element(
        self,
        solid_id: int,
        solid_type: str,
        node_coords: np.ndarray,
        node_velocities: np.ndarray,
        solid_mass: float,
        solid_rho: float,
        solid_energy: float = 0.0,
        solid_stress: Optional[np.ndarray] = None,
        solid_plastic_strain: float = 0.0,
        node_masses: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Convert a single solid element into an active SPH particle cloud.

        Port of soltosph.F lines 212-330, soltospha.F lines 212-329, and soltosph_on1.F lines 185-265.

        Args:
            solid_id: Element identifier.
            solid_type: 'hex8' or 'tet4'.
            node_coords: Element corner coordinates, shape (8, 3) or (4, 3).
            node_velocities: Element nodal velocities, shape (8, 3) or (4, 3).
            solid_mass: Total solid element mass.
            solid_rho: Solid density.
            solid_energy: Element internal energy.
            solid_stress: Optional Cauchy stress tensor (6,) or (3, 3).
            solid_plastic_strain: Effective plastic strain.
            node_masses: Optional lumped nodal masses for exact kinetic energy accounting.

        Returns:
            particle_data: Dictionary containing created SPH particle arrays:
              - 'pos': Positions (N_p, 3)
              - 'vel': Velocities (N_p, 3)
              - 'mass': Masses (N_p,)
              - 'rho': Densities (N_p,)
              - 'h': Smoothing lengths (N_p,)
              - 'energy': Internal energies (N_p,)
              - 'stress': Stresses (N_p, 6)
              - 'plastic_strain': Plastic strains (N_p,)
              - 'delta_e_hour': Discretization kinetic energy loss booked to numerical dissipation
        """
        stype = solid_type.lower()
        coords = np.asarray(node_coords, dtype=np.float64)
        vels = np.asarray(node_velocities, dtype=np.float64)

        if 'hex' in stype or 'brick' in stype:
            n_pts_1d = HEX_GAUSS_COORDS.get(self.n_dir, HEX_GAUSS_COORDS[2])
            n_p = len(n_pts_1d) ** 3
            ref_pts = []
            for xi in n_pts_1d:
                for eta in n_pts_1d:
                    for zeta in n_pts_1d:
                        ref_pts.append((xi, eta, zeta))
        elif 'tet' in stype:
            b_coords = TET_BARYCENTRIC_COORDS.get(self.n_dir, TET_BARYCENTRIC_COORDS[1])
            n_p = len(b_coords)
            ref_pts = [(b[0], b[1], b[2]) for b in b_coords]
        else:
            raise ValueError(f"Unknown solid type {solid_type}")

        # Sub-particle positions and velocities
        pos_p = np.zeros((n_p, 3), dtype=np.float64)
        vel_p = np.zeros((n_p, 3), dtype=np.float64)

        for i, (xi, eta, zeta) in enumerate(ref_pts):
            pos_p[i] = interpolate_solid_field(stype, coords, xi, eta, zeta)
            vel_p[i] = interpolate_solid_field(stype, vels, xi, eta, zeta)

        # Mass conservation: sum(m_p) = M_solid (soltosph_on1.F line 211)
        m_particle = solid_mass / n_p
        masses = np.full(n_p, m_particle, dtype=np.float64)
        densities = np.full(n_p, solid_rho, dtype=np.float64)

        # Characteristic volume and smoothing length
        vol_solid = solid_mass / max(solid_rho, 1e-20)
        vol_p = vol_solid / n_p
        dp = vol_p ** (1.0 / 3.0)
        h_particles = np.full(n_p, self.h_factor * dp, dtype=np.float64)

        # State transfer: internal energy, stress, plastic strain (soltosph.F lines 1105-1275)
        e_p = np.full(n_p, solid_energy / n_p, dtype=np.float64)
        pl_p = np.full(n_p, solid_plastic_strain, dtype=np.float64)

        if solid_stress is not None:
            s_arr = np.asarray(solid_stress, dtype=np.float64)
            if s_arr.shape == (3, 3):
                s6 = np.array([s_arr[0, 0], s_arr[1, 1], s_arr[2, 2], s_arr[0, 1], s_arr[1, 2], s_arr[0, 2]])
            else:
                s6 = s_arr.reshape(-1)[:6]
            stresses = np.tile(s6, (n_p, 1))
        else:
            stresses = np.zeros((n_p, 6), dtype=np.float64)

        # Kinetic energy accounting (soltosph_on1.F lines 196-200, 263-265)
        # E_k,solid = sum 0.5 * m_node * v_node^2
        if node_masses is not None:
            m_nod = np.asarray(node_masses, dtype=np.float64)
            e_k_solid = 0.5 * float(np.sum(m_nod[:, None] * (vels ** 2)))
        else:
            e_k_solid = 0.5 * float(np.sum((solid_mass / len(coords)) * (vels ** 2)))

        e_k_sph = 0.5 * float(np.sum(masses[:, None] * (vel_p ** 2)))
        delta_e_hour = max(0.0, e_k_solid - e_k_sph)
        self.accumulated_e_hour += delta_e_hour
        self.converted_solids.add(solid_id)

        return {
            'pos': pos_p,
            'vel': vel_p,
            'mass': masses,
            'rho': densities,
            'h': h_particles,
            'energy': e_p,
            'stress': stresses,
            'plastic_strain': pl_p,
            'delta_e_hour': delta_e_hour,
        }

