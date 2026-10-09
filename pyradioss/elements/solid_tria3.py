"""
3-node Constant Strain 2D Triangle Solid Element (/TRIA / /TRIA3).

**This module is a declared port extension, not a port**, registered with its
justification and upstream-search evidence in ``docs/PORT_EXTENSIONS.md`` and
kept honest by ``tests/test_p2_tria3_provenance.py``. There is no open-source
kernel to transcribe: ``engine/source/elements/solid_2d/`` holds only ``quad/``
and ``quad4/`` with no ``tria/``, while the ``tria/`` under ``starter/`` holds
just two deck readers, ``t3grhead.F``/``t3grtails.F``. /TRIA3 is a declared
element of the input format whose kernel ships only in the closed-source tree,
so the only honest citation is **Simcenter Radioss (closed source)**. Nothing
here was read off a Fortran routine; this module must never be listed as a port.

Theory notes:
* 3 nodes in the (Y, Z) plane with 2 translational DOFs per node (Y=radial, Z=axial).
* In-plane area: A = 0.5 * |(y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1)|.
* Constant strain field across the triangle:
      dN1/dy = (z2 - z3) / (2A),  dN1/dz = (y3 - y2) / (2A)
      dN2/dy = (z3 - z1) / (2A),  dN2/dz = (y1 - y3) / (2A)
      dN3/dy = (z1 - z2) / (2A),  dN3/dz = (y2 - y1) / (2A)
* Axisymmetric formulation (N2D=1): centroid radius r_c = (y1 + y2 + y3) / 3,
    1-radian volume V = A * r_c, hoop strain rate
    D_theta = (v_{y,1} + v_{y,2} + v_{y,3}) / (3 * r_c), and the radial hoop
    force f_{y,i}^{hoop} = - 1/3 * A * sigma_theta on the Y DOFs.
* Plane strain formulation (N2D=2): volume per unit thickness is A, and the
    out-of-plane strain D_theta = 0.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3


def _geometry_tria(xe: np.ndarray, n2d: int = 2):
    """Compute in-plane shape gradients, area, and volume for 3-node 2D triangle.

    xe: (n, 3, 3) nodal coordinates in (X, Y, Z). Y is axis 1, Z is axis 2.
    Returns:
        dndx: (n, 3, 2) Cartesian gradients in (Y, Z)
        area: (n,) triangle cross-sectional area
        vol: (n,) element volume
        rc: (n,) centroid radius
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 3, 2)), np.zeros(0), np.zeros(0), np.zeros(0)

    y1, y2, y3 = xe[:, 0, 1], xe[:, 1, 1], xe[:, 2, 1]
    z1, z2, z3 = xe[:, 0, 2], xe[:, 1, 2], xe[:, 2, 2]

    # Twice triangle area: 2A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1)
    a2 = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    area = 0.5 * np.abs(a2)
    safe_a2 = np.where(np.abs(a2) > EM20, a2, 1.0)

    dndx = np.zeros((n, 3, 2), dtype=np.float64)
    # dN/dy
    dndx[:, 0, 0] = (z2 - z3) / safe_a2
    dndx[:, 1, 0] = (z3 - z1) / safe_a2
    dndx[:, 2, 0] = (z1 - z2) / safe_a2
    # dN/dz
    dndx[:, 0, 1] = (y3 - y2) / safe_a2
    dndx[:, 1, 1] = (y1 - y3) / safe_a2
    dndx[:, 2, 1] = (y2 - y1) / safe_a2

    rc = (y1 + y2 + y3) / 3.0

    if n2d == 1:
        # Axisymmetric (1 radian revolution)
        vol = area * np.maximum(rc, 0.0)
    else:
        # Plane strain
        vol = area.copy()

    return dndx, area, vol, rc


def _char_length(xe: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Characteristic minimum altitude length for 2D triangle: lc = 2 * A / max_edge."""
    n = len(xe)
    if n == 0:
        return np.zeros(0)
    y = xe[:, :, 1]
    z = xe[:, :, 2]

    e1 = (y[:, 1] - y[:, 0])**2 + (z[:, 1] - z[:, 0])**2
    e2 = (y[:, 2] - y[:, 1])**2 + (z[:, 2] - z[:, 1])**2
    e3 = (y[:, 0] - y[:, 2])**2 + (z[:, 0] - z[:, 2])**2
    max_e2 = np.maximum.reduce([e1, e2, e3])
    max_e = np.sqrt(np.maximum(max_e2, EM20))
    return np.where(max_e > 0.0, 2.0 * area / max_e, 0.0)


def init_group(group, model, log):
    """Starter initialization for 3-node 2D triangle element group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 6)),
            epsp=np.zeros(0),
            vol0=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.zeros(0),
            qvw_pend=np.zeros(0),
            dtfac=np.zeros(0),
            chk_fail=False,
            dama=np.zeros(0),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    n2d = getattr(model, "n2d", 2) or 2
    xe = model.x0[conn]
    dndx0, area0, vol0, rc0 = _geometry_tria(xe, n2d)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/TRIA3 {eid}: zero or negative area/volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    lc0 = _char_length(xe, area0)
    dtfac = np.full(n, 1.0)

    group.state.update(
        sig=np.zeros((n, 6)),
        epsp=np.zeros(n),
        vol0=vol0.copy(),
        area0=area0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        dtfac=dtfac,
        n2d=n2d,
        chk_fail=False,
        dama=np.zeros(n),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 3.0, 3)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 3-node 2D triangle elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    n2d = st.get("n2d", 2)
    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    dndx, area, vol, rc = _geometry_tria(xe, n2d)
    vol_safe = np.maximum(vol, EM20)
    lc = _char_length(xe, area)

    # Probe cycle 0
    if dt == 0.0 or v is None:
        rho = st["mass"] / vol_safe
        c = np.zeros(n)
        for sl, mat, _ in st.get("slices", []):
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        return np.where(c > 0.0, st["dtfac"] * lc / c, EP30)

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    # In-plane velocities in (Y, Z)
    v_yz = ve[:, :, 1:3]  # (n, 3, 2)

    # In-plane velocity gradient L[n, b, c] = sum_i v_yz[n, i, b] * dndx[n, i, c]
    L = np.einsum("nib,nic->nbc", v_yz, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))

    Dyy = D[:, 0, 0]
    Dzz = D[:, 1, 1]
    Dyz2 = 2.0 * D[:, 0, 1]

    # Hoop strain rate (N2D=1)
    if n2d == 1:
        safe_rc = np.maximum(rc, EM20)
        vy_avg = np.mean(v_yz[:, :, 0], axis=1)
        Dxx = vy_avg / safe_rc
    else:
        Dxx = np.zeros_like(Dyy)

    trD = Dxx + Dyy + Dzz

    # Jaumann spin
    Wyz = 0.5 * (L[:, 0, 1] - L[:, 1, 0]) * dt

    deps = np.zeros((n, 6), dtype=np.float64)
    deps[:, 0] = Dxx * dt
    deps[:, 1] = Dyy * dt
    deps[:, 2] = Dzz * dt
    deps[:, 4] = Dyz2 * dt

    if not np.all(alive):
        deps[~alive] = 0.0
        trD[~alive] = 0.0

    # Jaumann stress rotation in (Y, Z) plane
    syy = sig[:, 1].copy()
    szz = sig[:, 2].copy()
    syz = sig[:, 4].copy()

    sig[:, 1] += 2.0 * Wyz * syz
    sig[:, 2] += -2.0 * Wyz * syz
    sig[:, 4] += Wyz * (szz - syy)

    # Constitutive update
    c_sound = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)
    rho = st["mass"] / vol_safe

    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "qa", 1.1) if prop else 1.1
        qb[sl] = getattr(prop, "qb", 0.05) if prop else 0.05

        law = getattr(mat, "law", 1)
        if law == 0:
            sig[sl] = 0.0
            continue

        K_sl = getattr(mat, "K", 0.0)
        G_sl = getattr(mat, "G", 0.0)
        c_sound[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))

        materials.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt, None)

    # Bulk viscosity
    compressing = (trD < 0.0) & alive
    qvisc = np.where(
        compressing,
        rho * lc * (qa**2 * lc * trD**2 - qb * c_sound * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, 0] -= qvisc
    sig_tot[:, 1] -= qvisc
    sig_tot[:, 2] -= qvisc

    # Internal force in (Y, Z):
    S2 = np.empty((n, 2, 2), dtype=np.float64)
    S2[:, 0, 0] = sig_tot[:, 1]
    S2[:, 0, 1] = sig_tot[:, 4]
    S2[:, 1, 0] = sig_tot[:, 4]
    S2[:, 1, 1] = sig_tot[:, 2]

    # f_yz[n, i, b] = - vol[n] * sum_c S2[n, b, c] * dndx[n, i, c]
    f_yz = -vol[:, None, None] * np.einsum("nbc,nic->nib", S2, dndx)

    # Axisymmetric hoop force on Y DOF: f_y^{hoop} = - 1/3 * area * sig_theta
    if n2d == 1:
        f_hoop_y = - (1.0 / 3.0) * area * sig_tot[:, 0]
        f_yz[:, :, 0] += f_hoop_y[:, None]

    fe = np.zeros((n, 3, 3), dtype=np.float64)
    fe[:, :, 1:3] = f_yz

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = vol * 0.5 * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    deint0 = vol * np.einsum("na,na->n", sig_mid, deps) + w_visc
    st["qvw_pend"] = vol * 0.5 * qvisc * dt
    st["eint"] += deint0

    # Courant time step
    Q = np.where(compressing, qb * c_sound + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit


# ----------------------------------------------------------------------------
# Implicit element matrices: tangent, kgeo, consistent_mass (M614 Component 1B)
# ----------------------------------------------------------------------------
# Port extension -- no upstream counterpart (see the module docstring and
# docs/PORT_EXTENSIONS.md): the standard CST consistent matrices, built from the
# same shape functions _geometry_tria() gives above.
# ----------------------------------------------------------------------------

def _edofs(conn: np.ndarray) -> np.ndarray:
    """(n, 6) global scalar DOF slot ids, node-major [uy, uz] * 3."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 6), dtype=np.int64)
    edofs = np.empty((n, 6), dtype=np.int64)
    for i in range(3):
        edofs[:, 2 * i + 0] = conn[:, i] * 6 + 1
        edofs[:, 2 * i + 1] = conn[:, i] * 6 + 2
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for 3-node 2D CST triangle (n, 6, 6).

    Port extension -- no upstream counterpart (see the module docstring).

    Returns (ke, edofs):
      ke: (n, 6, 6) in-plane element stiffness matrix (uy, uz)
      edofs: (n, 6)
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 6, 6), dtype=float), np.zeros((0, 6), dtype=np.int64)

    xe = x[conn]
    n2d = int(np.asarray(st.get("n2d", 2)).flat[0])
    dndx, area, vol, rc = _geometry_tria(xe, n2d)

    ke = np.zeros((n, 6, 6), dtype=np.float64)

    # Constant B-matrix (n, 3, 6): [eps_yy, eps_zz, gamma_yz]
    B = np.zeros((n, 3, 6), dtype=np.float64)
    for i in range(3):
        B[:, 0, 2 * i + 0] = dndx[:, i, 0]  # dN_i/dy
        B[:, 1, 2 * i + 1] = dndx[:, i, 1]  # dN_i/dz
        B[:, 2, 2 * i + 0] = dndx[:, i, 1]
        B[:, 2, 2 * i + 1] = dndx[:, i, 0]

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        E = float(getattr(mat, "E", 1.0e10) or 1.0e10)
        nu = float(getattr(mat, "nu", 0.3) or 0.3)

        denom = (1.0 + nu) * (1.0 - 2.0 * nu)
        if abs(denom) < 1e-12:
            denom = 1e-12
        C11 = E * (1.0 - nu) / denom
        C12 = E * nu / denom
        G = E / (2.0 * (1.0 + nu))

        D2d = np.array([
            [C11, C12, 0.0],
            [C12, C11, 0.0],
            [0.0, 0.0, G],
        ], dtype=np.float64)

        Bs = B[sl]
        DB = np.einsum("ab,nbj->naj", D2d, Bs)
        ke[sl] += vol[sl, None, None] * np.einsum("nai,naj->nij", Bs, DB)

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for 2D triangle (n, 6, 6).

    Port extension -- no upstream counterpart (see the module docstring).

    Returns (ke, edofs): ke (n, 6, 6), edofs (n, 6).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 6, 6), dtype=float), np.zeros((0, 6), dtype=np.int64)

    xe = x[conn]
    n2d = int(np.asarray(st.get("n2d", 2)).flat[0])
    dndx, _, vol, _ = _geometry_tria(xe, n2d)

    ke = np.zeros((n, 6, 6), dtype=np.float64)
    sig = st["sig"]  # (n, 6)

    S = np.empty((n, 2, 2), dtype=np.float64)
    S[:, 0, 0] = sig[:, 1]
    S[:, 1, 1] = sig[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 4]

    g = vol[:, None, None] * np.einsum("nac,ncd,nbd->nab", dndx, S, dndx)

    for a in range(3):
        for b in range(3):
            val = g[:, a, b]
            ke[:, 2 * a + 0, 2 * b + 0] += val
            ke[:, 2 * a + 1, 2 * b + 1] += val

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


_M_TRIA3_2D = np.array([
    [2.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0],
], dtype=np.float64) / 12.0


def consistent_mass(group, x=None):
    """Analytical 2D triangle consistent mass matrix (6x6 in plane).

    Port extension -- no upstream counterpart (see the module docstring).

    Returns (me, edofs): me (n, 6, 6), edofs (n, 6).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 6, 6), dtype=float), np.zeros((0, 6), dtype=np.int64)

    m = st["mass"]
    me = np.zeros((n, 6, 6), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            val = m * _M_TRIA3_2D[a, b]
            me[:, 2 * a + 0, 2 * b + 0] = val
            me[:, 2 * a + 1, 2 * b + 1] = val

    dead = (st["off"] <= 0.0)
    if np.any(dead):
        me[dead] = 0.0

    return me, _edofs(conn)

