"""
8-node corotational hexahedral solid element with assumed strain formulation.

Fortran source citations:
- s8zforc3:  $OR_SRC/engine/source/elements/solid/solide8z/s8zforc3.F
- s8zdefo3:  $OR_SRC/engine/source/elements/solid/solide8z/s8zdefo3.F
- s8zderi3:  $OR_SRC/engine/source/elements/solid/solide8z/s8zderi3.F
- s8zfint3:  $OR_SRC/engine/source/elements/solid/solide8z/s8zfint3.F
- srepiso3:  $OR_SRC/engine/source/elements/solid/solide/srepiso3.F
- sortho3:   $OR_SRC/engine/source/elements/solid/solide/sortho3.F

Theory notes (Belytschko & Bindeman 1993, Flanagan & Belytschko 1981):
* Corotational / Convected Frame:
  At each cycle, an element-fixed orthogonal coordinate frame R = [e1, e2, e3] is computed
  from the natural axes of the hexahedron (srepiso3.F) and iteratively orthogonalized (sortho3.F).
  Nodal coordinates and velocities are projected into the corotational frame:
      x_loc = x_glob . R,   v_loc = v_glob . R
  Because the corotational frame rotates with the rigid-body motion of the element, large
  rotations are handled geometrically without artificial stress rotation drift.
* Assumed Strain Kinematics (s8zdefo3.F, s8zderi3.F):
  Deformation rate D is computed in the corotational frame from the velocity gradient
  L = v_loc^T . gradN. Assumed volumetric dilatation (ICP=1) mitigates volumetric locking
  in nearly-incompressible regimes.
* Internal Forces (s8zfint3.F):
  Nodal forces are integrated in the corotational frame:
      f_loc = - V * (sigma_loc . gradN)
  and transformed back to the global frame:
      f_glob = f_loc . R^T
* Bulk Viscosity (sbulk3.F):
  Compressive volumetric shocks are damped with quadratic and linear bulk viscosity pressure.
* Critical Time Step:
  Courant-Friedrichs-Lewy stability condition based on characteristic element height lc and
  acoustic wave speed c: dt = lc / (c + Q).
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import DEFAULT_DT_SCALE, DEFAULT_QA, DEFAULT_QB, EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

# Node sign pattern of the trilinear hexa (Radioss /BRICK node ordering:
# nodes 1-4 = bottom face counter-clockwise, 5-8 = top face).
_XI = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)

_DN_DXI = _XI / 8.0
_DN_DXI_T = np.ascontiguousarray(_DN_DXI.T)  # (3, 8)

# The 4 hourglass base vectors of Flanagan-Belytschko (their table 2)
_H = np.array([
    [1, 1, -1, -1, -1, -1, 1, 1],    # mode 1
    [1, -1, -1, 1, -1, 1, 1, -1],    # mode 2
    [1, -1, 1, -1, 1, -1, 1, -1],    # mode 3
    [-1, 1, -1, 1, 1, -1, 1, -1],    # mode 4
], dtype=np.float64)

_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
], dtype=np.int64)


# ----------------------------------------------------------------------------
# Corotational Frame Construction (srepiso3.F + sortho3.F)
# ----------------------------------------------------------------------------

def _corotational_frame(xe: np.ndarray) -> np.ndarray:
    """Compute orthonormal corotational frame R = [e1, e2, e3] for each element.

    Ported from:
    - $OR_SRC/engine/source/elements/solid/solide/srepiso3.F
    - $OR_SRC/engine/source/elements/solid/solide/sortho3.F

    xe : (n, 8, 3) global nodal coordinates.
    Returns: R of shape (n, 3, 3) where columns are orthonormal unit vectors [e1, e2, e3].
    """
    n = len(xe)
    if n == 0:
        return np.empty((0, 3, 3), dtype=np.float64)

    # srepiso3.F: diagonal vectors connecting opposite corners
    x17 = xe[:, 6] - xe[:, 0]  # node 7 - node 1 (0-indexed: 6 - 0)
    x28 = xe[:, 7] - xe[:, 1]  # node 8 - node 2 (0-indexed: 7 - 1)
    x35 = xe[:, 4] - xe[:, 2]  # node 5 - node 3 (0-indexed: 4 - 2)
    x46 = xe[:, 5] - xe[:, 3]  # node 6 - node 4 (0-indexed: 5 - 3)

    a17 = x17 + x46
    a28 = x28 + x35

    rx = x17 + x28 - x35 - x46  # axis along xi
    sx = a17 + a28              # axis along eta
    tx = a17 - a28              # axis along zeta

    # sortho3.F: normalize r, s, t
    nr = np.maximum(norm3(rx), EM20)[:, None]
    ns = np.maximum(norm3(sx), EM20)[:, None]
    nt = np.maximum(norm3(tx), EM20)[:, None]

    u = rx / nr
    v = sx / ns
    w = tx / nt

    # 3 iterative orthogonalization steps
    for _ in range(3):
        e1 = cross3(v, w) + u
        e2 = cross3(w, u) + v
        e3 = cross3(u, v) + w
        u = e1 / np.maximum(norm3(e1), EM20)[:, None]
        v = e2 / np.maximum(norm3(e2), EM20)[:, None]
        w = e3 / np.maximum(norm3(e3), EM20)[:, None]

    # Final Gram-Schmidt orthogonalization
    e1 = u
    e3_raw = cross3(e1, v)
    e3 = e3_raw / np.maximum(norm3(e3_raw), EM20)[:, None]
    e2 = cross3(e3, e1)

    # Assemble R where columns are [e3, e1, e2] matching srcoor3.F lines 281-283
    # (JHBE=14/24 passes R12, R13, R11 so dir1=e3, dir2=e1, dir3=e2)
    R = np.stack([e3, e1, e2], axis=-1)  # (n, 3, 3)
    return R


def _geometry(xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid Jacobian, volume and shape function gradients.

    Ported from $OR_SRC/engine/source/elements/solid/solide8z/s8zderi3.F

    xe : (n, 8, 3) nodal coordinates.
    Returns (dndx (n, 8, 3), vol (n,)).
    """
    n = len(xe)
    if n == 0:
        return np.empty((0, 8, 3)), np.empty(0)

    J = _DN_DXI_T @ xe  # (n, 3, 3)
    detJ, Jinv = det_inv33(J)
    bad = detJ <= EM20

    if np.any(bad):
        J_safe = np.where(bad[:, None, None], np.eye(3)[None, :, :], J)
        _, Jinv = det_inv33(J_safe)
        Jinv[bad] = 0.0
        vol = np.where(detJ <= EM20, EM20, 8.0 * detJ)
        dndx = _DN_DXI @ Jinv.transpose(0, 2, 1)
        dndx[bad] = 0.0
        return dndx, vol

    vol = np.where(detJ <= EM20, EM20, 8.0 * detJ)
    dndx = _DN_DXI @ Jinv.transpose(0, 2, 1)
    return dndx, vol


def _char_length(xe: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Characteristic element length lc = V / max face area (sdlen3.F)."""
    d1 = xe[:, _FACES[:, 2]] - xe[:, _FACES[:, 0]]
    d2 = xe[:, _FACES[:, 3]] - xe[:, _FACES[:, 1]]
    a = 0.5 * norm3(cross3(d1, d2))
    return vol / np.maximum(a.max(axis=1), EM20)


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def init_group(group, model, log) -> tuple[np.ndarray, np.ndarray, None]:
    """Initialize element buffer and lumped masses for corotational hexa group.

    Ported from $OR_SRC/starter/source/elements/solid/sinit3.F
    """
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.empty((0, 6)),
            epsp=np.empty(0),
            vol0=np.empty(0),
            mass=np.empty(0),
            eint=np.empty(0),
            ehour=np.empty(0),
            off=np.empty(0),
            dtfac=np.empty(0),
            R=np.empty((0, 3, 3)),
            chk_fail=False,
            mat_extra={},
        )
        return np.empty(0, dtype=np.int64), np.empty(0), None

    xe = model.x0[group.conn]  # (n, 8, 3)
    R0 = _corotational_frame(xe)
    xe_loc = np.einsum("nia,nab->nib", xe, R0)
    dndx0, vol = _geometry(xe_loc)

    bad = vol <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/BRICK {eid}: zero or negative initial volume", "SOLID_HEXA8Z")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol

    lc0 = _char_length(xe, vol)

    group.state.update(
        sig=np.zeros((n, 6)),    # Cauchy stress in corotational frame
        epsp=np.zeros(n),        # equivalent plastic strain
        vol0=vol.copy(),         # initial volume
        mass=mass,               # element mass
        eint=np.zeros(n),        # internal energy
        ehour=np.zeros(n),       # hourglass energy
        off=np.ones(n),          # 1 alive / 0 deleted
        dtfac=np.ones(n),        # dt factor
        R=R0,                    # corotational rotation matrix
        chk_fail=False,
        mat_extra={},
    )

    group._model = model
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


# ----------------------------------------------------------------------------
# Engine Forces and Kinematics Cycle
# ----------------------------------------------------------------------------

def forces(group, x: np.ndarray, v: np.ndarray, vr: np.ndarray, dt: float,
           fint: np.ndarray | None, mint: np.ndarray | None) -> np.ndarray:
    """Evaluate corotational 8-node solid element internal forces and update state.

    Ported from:
    - s8zforc3: $OR_SRC/engine/source/elements/solid/solide8z/s8zforc3.F
    - s8zdefo3: $OR_SRC/engine/source/elements/solid/solide8z/s8zdefo3.F
    - s8zderi3: $OR_SRC/engine/source/elements/solid/solide8z/s8zderi3.F
    - s8zfint3: $OR_SRC/engine/source/elements/solid/solide8z/s8zfint3.F

    Parameters
    ----------
    group : ElementGroup
        Solid element group.
    x : np.ndarray of shape (Nnodes, 3)
        Global nodal positions.
    v : np.ndarray of shape (Nnodes, 3)
        Global nodal velocities.
    vr : np.ndarray
        Rotational velocities (unused for solids).
    dt : float
        Current engine time step.
    fint : np.ndarray of shape (Nnodes, 3)
        Global internal force accumulator (negative work convention).
    mint : np.ndarray
        Global moment accumulator (unused for solids).

    Returns
    -------
    dt_crit : np.ndarray of shape (Nelem,)
        Per-element critical time step.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0)

    alive = st["off"] > 0.5
    if not np.any(alive):
        return np.full(n, EP30)

    xe = x[conn]  # (n, 8, 3)
    ve = v[conn]  # (n, 8, 3)

    # 1. Corotational frame (srepiso3.F + sortho3.F)
    R = _corotational_frame(xe)  # (n, 3, 3)
    st["R"] = R

    # 2. Project coordinates and velocities into corotational frame
    # x_loc = xe @ R, v_loc = ve @ R
    xe_loc = np.einsum("nia,nab->nib", xe, R)
    ve_loc = np.einsum("nia,nab->nib", ve, R)

    # 3. Geometry and shape gradients in corotational frame (s8zderi3.F)
    dndx, vol = _geometry(xe_loc)
    lc = _char_length(xe, vol)

    # 4. Deformation kinematics in corotational frame (s8zdefo3.F)
    # Velocity gradient L_ab = sum_i ve_loc[i, a] * dndx[i, b]
    L = np.einsum("nia,nib->nab", ve_loc, dndx)

    # Rate of deformation D (symmetric part)
    dxx = L[:, 0, 0]
    dyy = L[:, 1, 1]
    dzz = L[:, 2, 2]
    dxy = L[:, 0, 1] + L[:, 1, 0]
    dyz = L[:, 1, 2] + L[:, 2, 1]
    dzx = L[:, 0, 2] + L[:, 2, 0]

    trD = dxx + dyy + dzz
    # Assumed volumic strain correction (s8zdefo3.F lines 189-196)
    # DVCA = sum_k (PXC_k * dvx_k + PYC_k * dvy_k + PZC_k * dvz_k)
    pxc = dndx[:, :4, 0]
    pyc = dndx[:, :4, 1]
    pzc = dndx[:, :4, 2]
    dvx = ve_loc[:, :4, 0] - ve_loc[:, 4:, 0]
    dvy = ve_loc[:, :4, 1] - ve_loc[:, 4:, 1]
    dvz = ve_loc[:, :4, 2] - ve_loc[:, 4:, 2]
    dvca = np.sum(pxc * dvx + pyc * dvy + pzc * dvz, axis=1)
    trD = np.where(alive, dvca, 0.0)

    # Strain increment over cycle
    deps = np.stack([dxx, dyy, dzz, dxy, dyz, dzx], axis=-1) * dt

    # 5. Spin tensor in corotational frame (residual spin after rigid rotation)
    wxy = 0.5 * (L[:, 0, 1] - L[:, 1, 0]) * dt
    wyz = 0.5 * (L[:, 1, 2] - L[:, 2, 1]) * dt
    wxz = 0.5 * (L[:, 0, 2] - L[:, 2, 0]) * dt

    sig = st["sig"]
    # Jaumann rotation of residual spin in local frame (srota3.F)
    sxx, syy, szz = sig[:, 0].copy(), sig[:, 1].copy(), sig[:, 2].copy()
    sxy, syz, szx = sig[:, 3].copy(), sig[:, 4].copy(), sig[:, 5].copy()
    sig[:, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    sig_old = sig.copy()

    # 6. Material law evaluation
    rho = st["mass"] / np.maximum(vol, EM20)
    c = np.zeros(n)
    qa = np.full(n, DEFAULT_QA)
    qb = np.full(n, DEFAULT_QB)
    hcoef = np.full(n, 0.1)

    for sl, mat, prop in st["slices"]:
        extra = {}
        if hasattr(mat, "solid_update"):
            sig_new, epsp_new, c_new = mat.solid_update(sig[sl], deps[sl], st["epsp"][sl], dt)
            sig[sl] = sig_new
            st["epsp"][sl] = epsp_new
            if c_new is not None:
                c[sl] = c_new
        elif hasattr(materials, "solid_update") and getattr(mat, "law", None) is not None:
            _, _, c_new = materials.solid_update(
                mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra or None
            )
            if c_new is not None:
                c[sl] = c_new
        else:
            # Linear elastic isotropic Hooke's law fallback
            K = getattr(mat, "K", 0.0)
            G = getattr(mat, "G", 0.0)
            lam = K - 2.0 * G / 3.0
            tr_eps = deps[sl, 0] + deps[sl, 1] + deps[sl, 2]
            sig[sl, 0] += lam * tr_eps + 2.0 * G * deps[sl, 0]
            sig[sl, 1] += lam * tr_eps + 2.0 * G * deps[sl, 1]
            sig[sl, 2] += lam * tr_eps + 2.0 * G * deps[sl, 2]
            sig[sl, 3] += G * deps[sl, 3]
            sig[sl, 4] += G * deps[sl, 4]
            sig[sl, 5] += G * deps[sl, 5]

        if hasattr(mat, "sound_speed_solid"):
            c[sl] = mat.sound_speed_solid()
        elif c[sl] == 0.0:
            K = getattr(mat, "K", 0.0)
            G = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum((K + 4.0 * G / 3.0) / np.maximum(rho[sl], EM20), 0.0))
        params = getattr(prop, "params", {})
        qa[sl] = params.get("qa", DEFAULT_QA)
        qb[sl] = params.get("qb", DEFAULT_QB)
        hcoef[sl] = params.get("h", 0.1)

    # 7. Bulk viscosity pressure (sbulk3.F)
    compressing = (trD < 0.0) & alive
    qvisc = np.where(
        compressing,
        rho * lc * (qa ** 2 * lc * trD ** 2 - qb * c * trD),
        0.0
    )

    # 8. Internal forces in corotational frame (s8zfint3.F)
    S = np.empty((n, 3, 3))
    S[:, 0, 0] = sig[:, 0] - qvisc
    S[:, 1, 1] = sig[:, 1] - qvisc
    S[:, 2, 2] = sig[:, 2] - qvisc
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]

    # fe_loc[i, b] = -vol * sum_c dndx[i, c] * S[c, b]
    fe_loc = (dndx @ S) * (-vol)[:, None, None]

    # Hourglass stabilization in local frame (shour3.F)
    # gamma_ai = h_ai - (sum_j h_aj x_j) . gradN_i
    hx = _H @ xe_loc                                            # (n, 4, 3)
    gamma = _H[None, :, :] - hx @ dndx.transpose(0, 2, 1)        # (n, 4, 8)
    qdot = gamma @ ve_loc                                       # (n, 4, 3) modal velocities
    ah = hcoef * rho * c * vol ** (2.0 / 3.0) / 4.0 * alive     # (n,)
    f_hg_loc = (gamma.transpose(0, 2, 1) @ qdot) * (-ah)[:, None, None]  # (n, 8, 3)
    fe_loc += f_hg_loc

    # 9. Transform local nodal forces back to global coordinates:
    # f_glob = fe_loc @ R^T
    fe_glob = np.einsum("nia,nba->nib", fe_loc, R)

    # 10. Energy accounting
    sig_avg = 0.5 * (sig_old + sig)
    deint0 = vol * (
        sig_avg[:, 0] * deps[:, 0] + sig_avg[:, 1] * deps[:, 1] + sig_avg[:, 2] * deps[:, 2]
        + sig_avg[:, 3] * deps[:, 3] + sig_avg[:, 4] * deps[:, 4] + sig_avg[:, 5] * deps[:, 5]
    )
    w_visc = -qvisc * trD * vol * dt
    dehour = np.sum(-f_hg_loc * ve_loc, axis=(1, 2)) * dt

    st["eint"] += deint0 + w_visc
    st["ehour"] += dehour

    # Zero forces for deleted elements
    fe_glob = np.where(alive[:, None, None], fe_glob, 0.0)

    # 11. Critical Courant time step (sdlen3.F)
    Q = qb * c + qa * lc * np.maximum(-trD, 0.0)
    dt_crit = DEFAULT_DT_SCALE * lc / np.maximum(Q + np.sqrt(Q ** 2 + c ** 2), EM20)
    dt_crit = np.where(alive, dt_crit, EP30)

    # 12. Scatter internal forces to global array
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe_glob.reshape(-1, 3))

    return dt_crit
