"""
General 6-DOF springs — /PROP/TYPE8 SPR_GENE (``spring_beam`` owns TYPE13).

Fortran origin
--------------
* engine : ``engine/source/elements/spring/rforc3.F`` dispatches IGTYP==8
  through ``r2coor3`` (frame) + ``r2def3.F`` (the 6 uncoupled DOF force
  laws); each DOF accumulates ``F_i = K_i d_i (+ C_i d_i_dot)`` from the
  relative motion of the two nodes resolved in a local skew frame
  (``EXX = SKEW(1,ISK)`` …).
* starter: ``starter/source/properties/spring/hm_read_prop08.F``
  (the K_i/C_i and the per-DOF function slots).

Ported physics (the LINEAR stiffness/damping core)
--------------------------------------------------
Six uncoupled degrees of freedom in a local orthonormal frame
(e1, e2, e3) — three translations, three rotations::

    delta_i = ((x2 - x1) - (X2 - X1)) . e_i          i = 1,2,3  (stretch)
    theta_i = theta_i + ((wr2 - wr1) . e_i) * dt     i = 4,5,6  (twist, accum)
    F_i = K_i delta_i + C_i (v2 - v1) . e_i
    M_i = K_i theta_i + C_i (wr2 - wr1) . e_i

Translations use the TOTAL form; rotations are RATE-integrated (the port
carries no accumulated nodal-rotation vector), exactly as r2def3
accumulates ``RX = RXOLD + …``. The nonlinear per-DOF channels (force
functions, hardening, rupture, rate smoothing, sensor gating) live in
:mod:`spring_common` and plug into ``forces6`` through the ``t8`` state.

Local frame: the skew frame ``SKEW(:,skew_ID)`` — e1/e2/e3 ARE the skew's
X'/Y'/Z' axes, read verbatim as ``r2def3.F`` reads them. ``skew_ID = 0``
uses the GLOBAL frame; a /SKEW/MOV frame is RELOADED every cycle.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30

#: property TYPE numbers handled by this module (TYPE8 general springs;
#: 13 stays a member because spring.init_group consults it for the mass
#: skip, but TYPE13 elements are built by spring_beam, never here)
SPRING_PROP_TYPES = frozenset({8, 13})


def _slice_frame(prop, xe, log, skews=None):
    """Local orthonormal frame (e1, e2, e3) for the elements of one slice,
    each returned as an (m, 3) array.  See the module docstring.

    TYPE8 with a ``skew_ID`` (M39) takes the /SKEW's axes verbatim:
    r2def3.F reads ``EXX = SKEW(1,ISK) ... EZZ = SKEW(9,ISK)`` and resolves
    the relative motion on them, i.e. e1/e2/e3 ARE the skew's X'/Y'/Z'.
    """
    m = len(xe)
    if m == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    skew_row = int(prop.params.get("skew_row", 0) or 0)
    if skew_row and skews is not None:
        if hasattr(skews, "axes") and 0 <= skew_row < len(skews.axes):
            a = skews.axes[skew_row]                 # rows = X', Y', Z'
            return (np.tile(a[0], (m, 1)), np.tile(a[1], (m, 1)),
                    np.tile(a[2], (m, 1)))
        elif log is not None:
            log.warning(f"/PROP/TYPE8/{getattr(prop, 'id', '?')}: "
                        f"skew_row={skew_row} out of bounds for skews (total {len(getattr(skews, 'axes', []))})",
                        "SPRING INIT")
    # global frame (TYPE8 skew 0, or the fallback)
    e1 = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
    e2 = np.tile(np.array([0.0, 1.0, 0.0]), (m, 1))
    e3 = np.tile(np.array([0.0, 0.0, 1.0]), (m, 1))
    return e1, e2, e3


def init6(group, model, log, idx6, massn, inertn):
    """Build the 6-DOF spring state for the ``idx6`` elements of a spring
    group and add their half/half nodal mass + rotational inertia into the
    caller's ``massn`` / ``inertn`` arrays (laid out like
    ``group.conn.reshape(-1)``)."""
    if idx6 is None or len(idx6) == 0:
        return
    st = group.state
    m6 = len(idx6)
    conn6 = group.conn[idx6, :2]
    xe = model.x0[conn6]                       # (m6, 2, 3)
    e1 = np.zeros((m6, 3))
    e2 = np.zeros((m6, 3))
    e3 = np.zeros((m6, 3))
    k6 = np.zeros((m6, 6))
    c6 = np.zeros((m6, 6))
    mass = np.zeros(m6)
    inertia = np.zeros(m6)
    iequil = np.zeros(m6, dtype=np.int64)

    # position of each idx6 element within group order -> its slice params
    pos = {int(e): j for j, e in enumerate(idx6)}
    skews = getattr(model, "skews", None)
    skew_row = np.zeros(m6, dtype=np.int64)     # 0 = global / element frame
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 4) not in SPRING_PROP_TYPES:
            continue
        rng = np.arange(group.n)[sl]
        local = np.array([pos[int(e)] for e in rng if int(e) in pos],
                         dtype=np.int64)
        if not len(local):
            continue
        p = prop.params
        se1, se2, se3 = _slice_frame(prop, xe[local], log, skews)
        e1[local], e2[local], e3[local] = se1, se2, se3
        if getattr(prop, "type", 8) == 8:
            skew_row[local] = int(p.get("skew_row", 0) or 0)
        for i in range(6):
            k6[local, i] = float(p.get(f"k{i + 1}", 0.0))
            c6[local, i] = float(p.get(f"c{i + 1}", 0.0))
        mass[local] = float(p.get("mass", 0.0))
        inertia[local] = float(p.get("inertia", 0.0))
        iequil[local] = int(p.get("iequil", 0) or 0)

    L0 = np.stack([
        np.einsum("mb,mb->m", xe[:, 1] - xe[:, 0], e1),
        np.einsum("mb,mb->m", xe[:, 1] - xe[:, 0], e2),
        np.einsum("mb,mb->m", xe[:, 1] - xe[:, 0], e3),
    ], axis=1)                                 # initial (x2-x1) . e_i

    # a TYPE8 on a /SKEW/MOV must re-read its frame every cycle (the skew
    # turns with its nodes — r2def3 reloads SKEW(:,ISK) on every call).
    # ``mov`` selects those elements; when nothing moves the frame stays
    # the init-time one and forces6 skips the refresh entirely.
    mov = np.zeros(m6, dtype=bool)
    if skews is not None:
        for r in np.unique(skew_row[skew_row > 0]):
            if hasattr(skews, "is_moving_row") and skews.is_moving_row(int(r)):
                mov |= skew_row == r

    st["gen6"] = dict(
        idx=np.asarray(idx6, dtype=np.int64), conn=conn6,
        e1=e1, e2=e2, e3=e3, k6=k6, c6=c6, mass=mass, inertia=inertia,
        iequil=iequil,
        L0=L0, theta=np.zeros((m6, 3)),
        force=np.zeros((m6, 3)), moment=np.zeros((m6, 3)),
        eint=np.zeros(m6),
        skews=skews, skew_row=skew_row, skew_mov=mov,
    )
    # half/half lumped mass + inertia into the caller's per-(elem,node)
    # arrays: node_idx = conn.reshape(-1) => slot stride*e (node0), stride*e+1 (node1)
    if massn is not None and inertn is not None:
        stride = group.conn.shape[1] if hasattr(group, "conn") and group.conn.ndim == 2 else 2
        for j, e in enumerate(idx6):
            if stride * e + 1 < len(massn):
                massn[stride * e] += mass[j] / 2.0
                massn[stride * e + 1] += mass[j] / 2.0
            if stride * e + 1 < len(inertn):
                inertn[stride * e] += inertia[j] / 2.0
                inertn[stride * e + 1] += inertia[j] / 2.0


def forces6(group, x, v, vr, dt, fint, mint, idx6):
    """6-DOF spring forces for the ``idx6`` elements; scatters translation
    forces into ``fint`` and moments into ``mint`` and returns their
    per-element critical time step (aligned to ``idx6``)."""
    if idx6 is None or len(idx6) == 0:
        return np.zeros(0)
    g = group.state.get("gen6")
    if g is None:
        return np.zeros(len(idx6))

    conn = g["conn"]
    n1, n2 = conn[:, 0], conn[:, 1]
    # a TYPE8 on a /SKEW/MOV reloads its axes from the (already updated,
    # engine step 0b) skew rows — r2def3 re-reads SKEW(:,ISK) every call,
    # so the spring's 6 DOFs turn with the skew's nodes.  The total-form
    # delta below then measures d(t).e(t) - d(0).e(0), which is exactly
    # r2def3's total branch (lines 305-307: X21DP*EXX - X0DP).
    mov = g["skew_mov"]
    if mov.any() and g.get("skews") is not None:
        skews = g["skews"]
        rows = g["skew_row"][mov]
        valid = (rows >= 0) & (rows < len(skews.axes))
        if valid.any():
            a = skews.axes[rows[valid]]        # (k, 3, 3)
            mov_indices = np.where(mov)[0][valid]
            g["e1"][mov_indices] = a[:, 0]
            g["e2"][mov_indices] = a[:, 1]
            g["e3"][mov_indices] = a[:, 2]
    e1, e2, e3 = g["e1"], g["e2"], g["e3"]

    if v is None:
        v = np.zeros_like(x)
    if vr is None:
        vr = np.zeros_like(x)
    dt_val = max(float(dt), 0.0) if dt is not None else 0.0

    # relative translation (total form) and its rate, per local axis
    d = (x[n2] - x[n1])
    dv = v[n2] - v[n1]
    delta = np.stack([np.einsum("mb,mb->m", d, e1) - g["L0"][:, 0],
                      np.einsum("mb,mb->m", d, e2) - g["L0"][:, 1],
                      np.einsum("mb,mb->m", d, e3) - g["L0"][:, 2]], axis=1)
    dvl = np.stack([np.einsum("mb,mb->m", dv, e1),
                    np.einsum("mb,mb->m", dv, e2),
                    np.einsum("mb,mb->m", dv, e3)], axis=1)
    F = g["k6"][:, :3] * delta + g["c6"][:, :3] * dvl
    g["force"] = F

    # relative rotation (rate-integrated) and its rate, per local axis
    dw = vr[n2] - vr[n1]
    dwl = np.stack([np.einsum("mb,mb->m", dw, e1),
                    np.einsum("mb,mb->m", dw, e2),
                    np.einsum("mb,mb->m", dw, e3)], axis=1)
    g["theta"] += dwl * dt_val
    M = g["k6"][:, 3:] * g["theta"] + g["c6"][:, 3:] * dwl
    g["moment"] = M

    # assemble to the nodes: global force / moment vectors
    fvec = F[:, 0:1] * e1 + F[:, 1:2] * e2 + F[:, 2:3] * e3
    mvec = M[:, 0:1] * e1 + M[:, 1:2] * e2 + M[:, 2:3] * e3

    if fint is not None:
        np.add.at(fint, n1, fvec)
        np.add.at(fint, n2, -fvec)

    # Rotational equilibrium (Fortran r2cum3.F:125-140):
    # When iequil == 1, a moment arm correction MM = 0.5 * (d x fvec) is added
    # to both nodes, ensuring exact angular momentum conservation:
    # sum(M) + sum(r x F) = 0.
    ieq = g.get("iequil")
    if ieq is not None and np.any(ieq == 1):
        has_eq = (ieq == 1)
        arm = 0.5 * np.cross(d, fvec)
        mvec1 = mvec.copy()
        mvec2 = -mvec.copy()
        mvec1[has_eq] += arm[has_eq]
        mvec2[has_eq] += arm[has_eq]
        if mint is not None:
            np.add.at(mint, n1, mvec1)
            np.add.at(mint, n2, mvec2)
    else:
        if mint is not None:
            np.add.at(mint, n1, mvec)
            np.add.at(mint, n2, -mvec)

    # internal energy (elastic + damping work, like the axial spring)
    g["eint"] += (np.einsum("mi,mi->m", F, dvl) + np.einsum("mi,mi->m", M, dwl)) * dt_val

    # critical time step with exact Fortran r2len3.F:182-185 damping reduction:
    # dt_tr  = mass / (sqrt(C^2 + mass * K) + C)
    # dt_rot = inertia / (sqrt(C^2 + inertia * K) + C)
    mass = np.maximum(g["mass"], 0.0)
    inertia = np.maximum(g["inertia"], 0.0)
    kt = np.maximum(g["k6"][:, :3], 0.0)
    ct = np.maximum(g["c6"][:, :3], 0.0)
    kr = np.maximum(g["k6"][:, 3:], 0.0)
    cr = np.maximum(g["c6"][:, 3:], 0.0)

    dt_tr = np.full((len(idx6), 3), EP30)
    for i in range(3):
        active = (mass > 0.0) & ((kt[:, i] > 0.0) | (ct[:, i] > 0.0))
        if np.any(active):
            m_a = mass[active]
            k_a = kt[active, i]
            c_a = ct[active, i]
            denom = np.sqrt(c_a * c_a + m_a * k_a) + c_a
            dt_tr[active, i] = m_a / np.maximum(denom, EM20)

    dt_rot = np.full((len(idx6), 3), EP30)
    for i in range(3):
        active = (inertia > 0.0) & ((kr[:, i] > 0.0) | (cr[:, i] > 0.0))
        if np.any(active):
            in_a = inertia[active]
            k_a = kr[active, i]
            c_a = cr[active, i]
            denom = np.sqrt(c_a * c_a + in_a * k_a) + c_a
            dt_rot[active, i] = in_a / np.maximum(denom, EM20)

    return np.minimum(dt_tr.min(axis=1), dt_rot.min(axis=1))
