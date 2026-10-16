# pyradioss/elements/spring_pretensioner.py
# Ported from $OR_SRC/engine/source/elements/spring/ruser32.F
# Ported from $OR_SRC/engine/source/elements/spring/ruser32ke3.F
# Ported from $OR_SRC/engine/source/elements/spring/ruser32mat3.F
# Ported from $OR_SRC/starter/source/properties/spring/hm_read_prop32.F
"""
2-Node Pretensioner Spring Element (/SPRING + /PROP/SPR_PRE or /PROP/TYPE32).

Fortran origins are the header comments above. Before activation
``F += STIF0*Ldot*dt``; after, the stroke ``X`` accumulates and ``FF`` is
imposed per ``ITYP``; energy uses the ``r5def3.F`` left rule
``dE = F_old*Ldot*dt``; ``dt = sqrt(M / K)``.
Open point (deferred to the ``spring.py`` owner — P4.8 touches that file):
``RINI32``'s E(0) (``_rini32_eint`` here, verified against the oracle T01
heads) cannot land in ``eint`` yet — the group-build zeroes ``eint`` after
init runs, and a first-cycle top-up trips the engine's energy guard instead.
Until the handoff lands, runs start at E(0) = 0 and the global IE carries
that offset (100500 on c52, exactly accounted).
Porting notes (exact spots where the port stops, all cited):
  - The ``IMPL_S > 0`` tangent refinements of ``ruser32.F`` (ITYP 1/2/4) are
    not ported: no implicit driver calls this channel, so ``UVAR(4)`` stays
    ``STIF0``.
  - Sensor gating reads the port ``Sensors`` state; ``GET_U_SENS`` returns
    ``TT - LT`` only for ``TT > LT`` (``usensor.F``), so upstream stays
    inactive on the exact firing cycle — a one-cycle knife-edge difference
    on sensor-gated decks (c52 has none).
  - ``preload_axial.F90`` (``/PRELOAD/AXIAL``) is not called by the
    ``rforc3.F:1061`` TYPE32 branch, so it plays no part here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3


def _safe_param(params: dict, key: str, default: float = 0.0) -> float:
    """Safely extract float parameter from prop.params with fallback."""
    val = params.get(key)
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _eval_curves(model: Any, fct_ids: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Per-element ``GET_U_FUNC`` (``ruser32.F``): missing/zero function -> 0."""
    out = np.zeros(len(np.asarray(xs, dtype=float)))
    if model is None or not hasattr(model, "functions"):
        return out
    for k, (f_id, x) in enumerate(zip(np.asarray(fct_ids).tolist(),
                                      np.asarray(xs, dtype=float).tolist())):
        out[k] = _eval_one(model.functions.get(int(f_id)) if f_id else None,
                           float(x))
    return out


def _eval_one(func: Any, x: float) -> float:
    """Scalar ``GET_U_FUNC`` (``ruser32.F``): a missing function reads as 0."""
    if func is None:
        return 0.0
    return func.eval(x) if hasattr(func, "eval") else func(x)


def _area_eint(d1: float, x: float, x2: float, y: float, y2: float, stif0: float) -> float:
    """AREA() of ``hm_read_prop32.F``: positive area under (X,Y)-(X2,Y2) for X <= D1."""
    if d1 >= x2:
        return 0.0
    if d1 >= x:
        fd1 = y + (y2 - y) / (x2 - x) * (d1 - x)
        if y < 0.0 and y2 > 0.0:
            x0 = x - y * (x2 - x) / (y2 - y)
            if d1 <= x0:
                return (x2 - x0) * y2 / 2.0
            return ((x2 - d1) * fd1 + (x2 - d1) * (y2 - fd1) / 2.0
                    + float(np.float64(fd1) / np.float64(stif0)) * fd1 / 2.0)
        if y >= 0.0 and y2 > 0.0:
            return ((x2 - d1) * fd1 + (x2 - d1) * (y2 - fd1) / 2.0
                    + float(np.float64(fd1) / np.float64(stif0)) * fd1 / 2.0)
        return 0.0
    if y >= 0.0 and y2 > 0.0:
        return (x2 - x) * y + (x2 - x) * (y2 - y) / 2.0
    return 0.0


def _curve_points(model: Any, fct_id: int):
    """Raw (x, y) of a ``/FUNCT`` table, else None (registry functs lack one)."""
    if model is None or not hasattr(model, "functions"):
        return None
    func = model.functions.get(int(fct_id)) if fct_id else None
    if func is None or not hasattr(func, "x") or not hasattr(func, "y"):
        return None
    return np.asarray(func.x, dtype=float), np.asarray(func.y, dtype=float)


def _rini32_eint(ityp: int, f1: float, d1: float, stif0: float,
                 pts1, pts2, s_d: float, s_f: float) -> float:
    """RINI32 (``hm_read_prop32.F``): Starter initial internal energy per ITYP.

    ``pts1/pts2`` are raw ``(x, y)`` tables (None when missing/table-free);
    scales are the stored inverted GEO values. A single-point table skips the
    two-point pre-block (upstream would read past it); the ``D1 > 0``
    pre-block is transcribed though ``D1 = -|D1|`` keeps it dormant.
    """
    if ityp == 1:
        if d1 == 0.0:
            return float(np.float64(f1) / np.float64(stif0)) * f1 / 2.0
        return abs(d1 * f1) / 2.0
    if ityp == 3:
        f0 = s_f * pts2[1][0] if pts2 is not None else 0.0
        return float(np.float64(f0) / np.float64(stif0)) * f0 / 2.0
    if pts1 is None or len(pts1[0]) == 0:
        return 0.0
    xs = np.asarray(pts1[0], dtype=float) * s_d
    ypre = np.asarray(pts1[1], dtype=float) * s_f
    if ityp == 4:
        f0 = s_f * pts2[1][0] if pts2 is not None else 0.0
        ys = np.asarray(pts1[1], dtype=float) * f0
        dloc = -1e30 if d1 == 0.0 else d1
    else:
        ys = ypre
        dloc = d1
        if dloc == 0.0 and ys[0] <= 0.0:
            dloc = -1e30
    e = 0.0
    if dloc > 0.0 and len(xs) > 1 and ypre[0] > 0.0 and xs[0] > dloc:
        y = ypre[0] + (ypre[1] - ypre[0]) / (xs[1] - xs[0]) * (dloc - xs[0])
        e += _area_eint(dloc, dloc, xs[0], y, ypre[0], stif0)
    for j in range(len(xs) - 1):
        if xs[j] < 0.0:
            e += _area_eint(dloc, xs[j], xs[j + 1], ys[j], ys[j + 1], stif0)
    return e


def _latch(force: float, ff: float, ilock: int, locked: float,
           x: Optional[float] = None, d1: Optional[float] = None) -> Tuple[float, float]:
    """Pretension lock + impose (``ruser32.F`` ITYP tails; ITYP1 passes no D1)."""
    if d1 is not None and x < d1 and d1 != 0.0:
        locked = 1.0
    if force > ff and ilock == 2:
        locked = 1.0
    if ff > 0.0 and locked == 0.0:
        force = max(ff, force)
    return force, locked


def init_pretensioner_type32(group, model, log, idx32: np.ndarray, massn: Optional[np.ndarray] = None, inertn: Optional[np.ndarray] = None) -> None:
    """TYPE32 state buffers (``hm_read_prop32.F`` params + ``ruser32.F`` UVARs).

    ``massn``/``inertn`` ride along for the shared init signature.
    """
    st = group.state
    n = group.n
    if len(idx32) == 0:
        return

    # Ensure TYPE32 state arrays exist.
    if "stif0" not in st:
        for key in ("stif0", "stif1", "f1", "d1",
                    "uvar1", "uvar2", "uvar3", "uvar4"):
            st[key] = np.zeros(n)
        for key in ("scale_t", "scale_d", "scale_f"):
            st[key] = np.ones(n)
        for key in ("ityp", "sens_id", "fct_id1", "fct_id2", "ilock"):
            st[key] = np.zeros(n, dtype=np.int64)

    _doubles = (("stif0", "stiff0", 0.0), ("stif1", "stiff1", 0.0), ("f1", None, 0.0), ("d1", None, 0.0), ("scale_t", None, 1.0), ("scale_d", None, 1.0), ("scale_f", None, 1.0))
    _ints = (("ityp", 1), ("sens_id", 0), ("fct_id1", 0), ("fct_id2", 0), ("ilock", 0))
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 4) != 32:
            continue
        p = getattr(prop, "params", {}) or {}
        for key, alt, default in _doubles:
            st[key][sl] = _safe_param(p, key, _safe_param(p, alt, default) if alt else default)
        for key, default in _ints:
            st[key][sl] = int(_safe_param(p, key, default))
        if "mass" in st:
            st["mass"][sl] = _safe_param(p, "mass", 0.0)
        if "k" in st:
            st["k"][sl] = st["stif0"][sl] + st["stif1"][sl]
        st["uvar4"][sl] = st["stif0"][sl]

    # RINI32 check (MSGID 406): a zero-length pretensioner is a Starter error.
    if log is not None and hasattr(model, "x0"):
        for e in np.asarray(idx32)[norm3(model.x0[group.conn[idx32][:, 1]] - model.x0[group.conn[idx32][:, 0]]) == 0.0].tolist():
            log.error(f"/SPRING {int(group.ids[e])}: zero initial length (RINI32 MSG406)", "SPRING INIT")


def forces_pretensioner_type32(group, x: np.ndarray, v: Optional[np.ndarray], dt: Optional[float], fint: Optional[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Axial TYPE32 forces (``ruser32.F``): scatter to ``fint``, book work with
    the pre-update force (``r5def3.F``), return the per-element critical step."""
    st = group.state
    conn = group.conn[idx]
    if len(conn) == 0:
        return np.empty(0)

    model = st.get("model")
    t = getattr(model, "t", 0.0) if model is not None else 0.0
    sensors = getattr(model, "sensors_state", None) if model is not None else None

    # Element orientation and length: L = |x2 - x1|
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    # Elongation rate Ldot = (v2 - v1) . a (ruser32.F VX(I))
    if v is None:
        Ldot = np.zeros(len(conn))
    else:
        Ldot = np.einsum("nb,nb->n", v[conn[:, 1]] - v[conn[:, 0]], a)

    F = st["force"][idx].copy()
    stif0 = st["stif0"][idx]
    stif1 = st["stif1"][idx]
    scale_t = st["scale_t"][idx]
    scale_d = st["scale_d"][idx]
    scale_f = st["scale_f"][idx]
    ityp = st["ityp"][idx]
    f1 = st["f1"][idx]
    d1 = st["d1"][idx]
    ilock = st["ilock"][idx]
    sens_id = st["sens_id"][idx]

    uvar1 = st["uvar1"][idx]  # relative stroke X
    uvar2 = st["uvar2"][idx]  # activation flag
    uvar3 = st["uvar3"][idx]  # locked flag
    uvar4 = st.get("uvar4", np.zeros(group.n))[idx]  # tangent stiffness

    # Sensor gating (see module docstring for the TACTI == 0 knife-edge).
    tacti = np.zeros(len(idx))
    iact = np.ones(len(idx), dtype=bool)
    if sensors is not None:
        for i, s_id in enumerate(sens_id):
            if s_id <= 0:
                tacti[i] = t
            elif sensors.active(s_id):
                tacti[i] = max(0.0, t - sensors.fire_time.get(s_id, 0.0))
            else:
                iact[i] = False
    else:
        tacti[:] = t

    dt_val = dt if (dt is not None and dt > 0.0) else 0.0

    # Inactive sensor (ruser32.F:162-171) vs active (ruser32.F:173-184).
    not_act = ~iact
    if np.any(not_act):
        uvar2[not_act] = 0.0
        F[not_act] += stif0[not_act] * dt_val * Ldot[not_act]
        uvar4[not_act] = stif0[not_act]
        st["k"][idx[not_act]] = stif0[not_act]

    act = iact
    if np.any(act):
        mask_just_act = act & (uvar2 == 0.0)
        uvar1[mask_just_act] = 0.0
        uvar2[mask_just_act] = 1.0

        uvar1[act] += dt_val * Ldot[act]
        F[act] += stif0[act] * dt_val * Ldot[act]
        uvar4[act] = stif0[act]
        st["k"][idx[act]] = stif0[act]

        # Evaluate pretension laws per ITYP (ruser32.F lines 186-250)
        for it in (1, 2, 3, 4):
            mask = act & (ityp == it)
            if not np.any(mask):
                continue

            X = uvar1[mask]
            cur_F = F[mask]
            cur_ilock = ilock[mask]
            cur_d1 = d1[mask]
            cur_uvar3 = uvar3[mask]

            if it == 1:
                # FF = F1 + STIF1 * X (ruser32.F:187-202); lock only on ILOCK==2.
                FF = f1[mask] + stif1[mask] * X
                cur_uvar3 = np.where((cur_F > FF) & (cur_ilock == 2), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0.0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)

            elif it == 2:
                # FF = Scale_f * func1(X * Scale_d) (ruser32.F:203-219).
                gidx = np.where(mask)[0]
                FF = scale_f[mask] * _eval_curves(
                    model, st["fct_id1"][idx[gidx]], X * scale_d[gidx])
                # Lock condition: stroke reached D1 or force exceeded FF (ILOCK==2)
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > FF) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0.0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)

            elif it == 3:
                # F0 = Scale_f * func2(tacti * Scale_t) (ruser32.F:220-230).
                gidx = np.where(mask)[0]
                F0 = scale_f[mask] * _eval_curves(
                    model, st["fct_id2"][idx[gidx]], tacti[mask] * scale_t[gidx])
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > F0) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((F0 > 0.0) & (cur_uvar3 == 0.0), np.maximum(F0, cur_F), cur_F)

            elif it == 4:
                # FF = F0(tacti) * func1(X * Scale_d) (ruser32.F:231-250).
                gidx = np.where(mask)[0]
                F0 = scale_f[mask] * _eval_curves(
                    model, st["fct_id2"][idx[gidx]], tacti[mask] * scale_t[gidx])
                FF = F0 * _eval_curves(
                    model, st["fct_id1"][idx[gidx]], X * scale_d[gidx])
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > FF) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0.0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)

            F[mask] = cur_F
            uvar3[mask] = cur_uvar3

    # Element deletion check
    alive = st.get("off", np.ones(group.n, dtype=float))[idx] > 0.0
    F = np.where(alive, F, 0.0)

    # Work with the pre-update force (r5def3.F books before RUSER32 updates it).
    F_old = st["force"][idx].copy()
    if dt_val > 0.0:
        st["eint"][idx] += np.where(alive, F_old * Ldot * dt_val, 0.0)

    st["force"][idx] = F
    st["uvar1"][idx] = uvar1
    st["uvar2"][idx] = uvar2
    st["uvar3"][idx] = uvar3
    if "uvar4" in st:
        st["uvar4"][idx] = uvar4

    # Scatter along the axis (+F*a on node 1, -F*a on node 2).
    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, conn[:, 0], fvec)
        np.add.at(fint, conn[:, 1], -fvec)

    # Explicit step: omega = 2*sqrt(K/M), dt = 2/omega = sqrt(M/K).
    mass = np.maximum(st["mass"][idx], EM20)
    k_dt = np.maximum(st["k"][idx], 0.0)
    pos_k = (st["k"][idx] > 0.0) & (st["mass"][idx] > 0.0)
    omega = 2.0 * np.sqrt(np.where(pos_k, k_dt / mass, 1.0))
    dt_crit = 2.0 / omega
    return np.where(alive, np.where(pos_k, dt_crit, EP30), EP30)


def driver(group, model, ctx) -> np.ndarray:
    """TYPE32 channel: ``driver(group, model, ctx) -> np.ndarray``.

    Shared 1-D contract (``rforc3.F:1061`` via ``RUSER32``): unpack ``ctx``
    and run the pretensioner force channel; returns the per-element step.
    """
    st = group.state
    if group.n == 0:
        return np.empty(0)
    st["model"] = model
    v = ctx.get("v")
    idx = st["idx32"] if "idx32" in st else np.arange(group.n, dtype=np.int64)
    return forces_pretensioner_type32(group, np.asarray(ctx["x"], dtype=float),
                                      None if v is None else np.asarray(v, dtype=float),
                                      float(ctx.get("dt", 0.0)), ctx["fint"], np.asarray(idx))


def implicit_stiffness_type32(group, x: np.ndarray, idx: np.ndarray, ikgeo: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """6x6 tangent per element: material part from ``UVAR(4)`` (``ruser32mat3.F``)
    plus the initial-stress geometric part (``ruser32ke3.F`` via ``tkeg3``)."""
    conn = group.conn[idx]
    if len(conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)

    st = group.state
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    # Tangent stiffness: Kt = uvar4 (ruser32mat3.F line 46)
    uvar4 = st.get("uvar4", st["stif0"])
    k_mat = uvar4[idx]

    # Material tangent Kt * [[aaT, -aaT], [-aaT, aaT]] (ruser32mat3.F, r4sumg3.F).
    aat = a[:, :, None] * a[:, None, :]  # (N, 3, 3)
    ke = np.zeros((len(idx), 6, 6), dtype=np.float64)
    km = k_mat[:, None, None] * aat
    ke[:, :3, :3] = km
    ke[:, 3:, 3:] = km
    ke[:, :3, 3:] = ke[:, 3:, :3] = -km

    # Geometric part (F/L) * [[P, -P], [-P, P]], P = I - aaT (ruser32ke3.F, tkeg3).
    if ikgeo == 1:
        kg = (st["force"][idx] / L)[:, None, None] * (np.eye(3)[None] - aat)
        ke[:, :3, :3] += kg
        ke[:, 3:, 3:] += kg
        ke[:, :3, 3:] -= kg
        ke[:, 3:, :3] -= kg

    edofs = (conn[:, :, None] * 3 + np.arange(3)).reshape(len(idx), 6).astype(np.int64)

    return ke, edofs


class SpringPretensioner:
    """Standalone TYPE32 model (``ruser32.F`` channel; RINI32's E(0) is the
    group channel's job, so this starts at zero work)."""

    def __init__(
        self,
        stif0: float,
        stif1: float = 0.0,
        f1: float = 0.0,
        d1: float = 0.0,
        e1: float = 0.0,
        ityp: int = 1,
        scale_t: float = 1.0,
        scale_d: float = 1.0,
        scale_f: float = 1.0,
        sens_id: int = 0,
        ilock: int = 0,
        mass: float = 0.0,
        func1: Optional[Any] = None,
        func2: Optional[Any] = None,
    ):
        self.stif0 = float(stif0)
        self.stif1 = float(stif1)
        self.f1 = float(f1)
        self.d1 = -abs(float(d1)) if d1 != 0.0 else 0.0
        self.e1 = float(e1)
        self.ityp = int(ityp)
        self.scale_t = float(scale_t) if scale_t != 0.0 else 1.0
        self.scale_d = float(scale_d) if scale_d != 0.0 else 1.0
        self.scale_f = float(scale_f) if scale_f != 0.0 else 1.0
        self.sens_id = int(sens_id)
        self.ilock = int(ilock)
        self.mass = float(mass)
        self.func1 = func1
        self.func2 = func2

        # State variables (UVAR 1 to 4 in ruser32.F)
        self.uvar1: float = 0.0  # relative stroke X
        self.uvar2: float = 0.0  # activation flag (0 or 1)
        self.uvar3: float = 0.0  # locked flag (0 or 1)
        self.uvar4: float = self.stif0  # tangent stiffness Kt
        self.force: float = 0.0
        self.eint: float = 0.0

    def step(
        self,
        L: float,
        Ldot: float,
        dt: float,
        t: float = 0.0,
        sensor_active: bool = True,
        fire_time: float = 0.0,
    ) -> float:
        """Advance pretensioner by one time increment dt and return axial force F."""
        if dt <= 0.0:
            return self.force

        f_old = self.force

        # Determine activation state and active time
        if self.sens_id > 0 and sensor_active:
            tacti, iact = max(0.0, t - fire_time), True
        elif self.sens_id > 0:
            tacti, iact = 0.0, False
        else:
            tacti, iact = t, True

        if not iact:
            # Inactive sensor (ruser32.F lines 162-171)
            self.uvar2 = 0.0
            self.force += self.stif0 * dt * Ldot
            self.uvar4 = self.stif0
        else:
            # Active sensor (ruser32.F lines 173-184)
            if self.uvar2 == 0.0:
                self.uvar1 = 0.0
                self.uvar2 = 1.0

            self.uvar1 += dt * Ldot
            self.force += self.stif0 * dt * Ldot
            self.uvar4 = self.stif0

            X = self.uvar1

            if self.ityp == 1:
                # Linear: FF = F1 + STIF1 * X (ruser32.F line 191)
                FF = self.f1 + self.stif1 * X
                self.force, self.uvar3 = _latch(self.force, FF, self.ilock, self.uvar3)

            elif self.ityp == 2:
                # Non-linear displacement: FF = Scale_f * func1(X * Scale_d)
                FF = self.scale_f * _eval_one(self.func1, X * self.scale_d)
                self.force, self.uvar3 = _latch(self.force, FF, self.ilock, self.uvar3, X, self.d1)

            elif self.ityp == 3:
                # Time-dependent: F0 = Scale_f * func2(tacti * Scale_t)
                F0 = self.scale_f * _eval_one(self.func2, tacti * self.scale_t)
                self.force, self.uvar3 = _latch(self.force, F0, self.ilock, self.uvar3, X, self.d1)

            elif self.ityp == 4:
                # Combined: FF = F0(tacti) * func1(X * Scale_d)
                F0 = self.scale_f * _eval_one(self.func2, tacti * self.scale_t)
                FF = F0 * _eval_one(self.func1, X * self.scale_d)
                self.force, self.uvar3 = _latch(self.force, FF, self.ilock, self.uvar3, X, self.d1)

        # Work (r5def3.F left rule): booked with the pre-update force.
        self.eint += f_old * Ldot * dt
        return self.force

    def critical_dt(self) -> float:
        """Compute critical time step dt = 2 / omega = sqrt(M / K)."""
        if self.mass <= 0.0 or self.uvar4 <= 0.0:
            return EP30
        omega = 2.0 * math.sqrt(self.uvar4 / self.mass)
        return 2.0 / omega
