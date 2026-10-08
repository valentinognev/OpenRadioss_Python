# pyradioss/elements/spring_pretensioner.py
# Ported from $OR_SRC/engine/source/elements/spring/ruser32.F
# Ported from $OR_SRC/engine/source/elements/spring/ruser32ke3.F
# Ported from $OR_SRC/engine/source/elements/spring/ruser32mat3.F
# Ported from $OR_SRC/starter/source/properties/spring/hm_read_prop32.F
"""
2-Node Pretensioner Spring Element (/SPRING + /PROP/SPR_PRE or /PROP/TYPE32).

Upstream Fortran origins:
  - engine/source/elements/spring/ruser32.F:
      Computes pretensioner spring forces, sensor activation, displacement/force
      locking, and 4 formulations of pretension laws (ITYP=1 linear, ITYP=2 f(x),
      ITYP=3 f(t), ITYP=4 f(t)*f(x)).
  - engine/source/elements/spring/ruser32ke3.F:
      Element tangent stiffness matrix (material tangent and initial-stress geometric stiffness).
  - engine/source/elements/spring/ruser32mat3.F:
      Tangent stiffness selection (UVAR 4) for explicit time step and implicit solver.
  - starter/source/properties/spring/hm_read_prop32.F:
      Property reading and parameter initialization (mass, stif0, stif1, f1, d1, e1,
      scale factors, sensor ID, lock flag).

Physics overview:
  1. Before sensor activation (t < t_fire):
     The spring behaves with initial stiffness STIF0:
       F += STIF0 * Ldot * dt
  2. After sensor activation (t >= t_fire):
     The relative displacement X = L(t) - L(t_fire) accumulates:
       X += Ldot * dt
       F += STIF0 * Ldot * dt
     Target pretension force FF is computed depending on ITYP:
       - ITYP 1: FF = F1 + STIF1 * X
       - ITYP 2: FF = Scale_f * func1(X * Scale_d)
       - ITYP 3: FF = Scale_f * func2(t_act * Scale_t)
       - ITYP 4: FF = Scale_f * func2(t_act * Scale_t) * func1(X * Scale_d)
     Locking (ILOCK):
       - If X < D1 (with D1 < 0 max retraction limit): locked.
       - If ILOCK == 2 and F > FF: locked (external tension exceeds pretension).
       - When not locked and FF > 0: F = max(FF, F)
  3. Energy accounting:
     dE_int = 0.5 * (F_old + F) * Ldot * dt
  4. Time step:
     omega = 2 * sqrt(K / M) => dt_crit = 2 / omega = sqrt(M / K)
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


def init_pretensioner_type32(group, model, log, idx32: np.ndarray, massn: Optional[np.ndarray] = None, inertn: Optional[np.ndarray] = None) -> None:
    """Initialize state buffers for TYPE32 pretensioner spring elements.

    Ported from $OR_SRC/starter/source/properties/spring/hm_read_prop32.F
    and engine/source/elements/spring/ruser32.F.

    Args:
        group: ElementGroup for springs.
        model: Global Model container.
        log: MessageLog for warnings/errors.
        idx32: Array of element indices belonging to TYPE32.
        massn: Nodal mass array to accumulate spring lumped mass into (M/2 per node).
        inertn: Optional nodal inertia array.
    """
    st = group.state
    n = group.n
    if len(idx32) == 0:
        return

    # Ensure TYPE32 state arrays exist
    if "stif0" not in st:
        st["stif0"] = np.zeros(n)
        st["stif1"] = np.zeros(n)
        st["ityp"] = np.zeros(n, dtype=np.int64)
        st["f1"] = np.zeros(n)
        st["d1"] = np.zeros(n)
        st["scale_t"] = np.ones(n)
        st["scale_d"] = np.ones(n)
        st["scale_f"] = np.ones(n)
        st["sens_id"] = np.zeros(n, dtype=np.int64)
        st["fct_id1"] = np.zeros(n, dtype=np.int64)
        st["fct_id2"] = np.zeros(n, dtype=np.int64)
        st["ilock"] = np.zeros(n, dtype=np.int64)
        st["uvar1"] = np.zeros(n)  # relative stroke X
        st["uvar2"] = np.zeros(n)  # activation flag (0 before, 1 after)
        st["uvar3"] = np.zeros(n)  # lock flag (0 unlocked, 1 locked)
        st["uvar4"] = np.zeros(n)  # tangent stiffness Kt

    for sl, mat, prop in st["slices"]:
        pt = getattr(prop, "type", 4)
        if pt == 32:
            p = getattr(prop, "params", {}) or {}
            st["stif0"][sl] = _safe_param(p, "stif0", _safe_param(p, "stiff0", 0.0))
            st["stif1"][sl] = _safe_param(p, "stif1", _safe_param(p, "stiff1", 0.0))
            st["ityp"][sl] = int(_safe_param(p, "ityp", 1))
            st["f1"][sl] = _safe_param(p, "f1", 0.0)
            st["d1"][sl] = _safe_param(p, "d1", 0.0)
            st["scale_t"][sl] = _safe_param(p, "scale_t", 1.0)
            st["scale_d"][sl] = _safe_param(p, "scale_d", 1.0)
            st["scale_f"][sl] = _safe_param(p, "scale_f", 1.0)
            st["sens_id"][sl] = int(_safe_param(p, "sens_id", 0))
            st["fct_id1"][sl] = int(_safe_param(p, "fct_id1", 0))
            st["fct_id2"][sl] = int(_safe_param(p, "fct_id2", 0))
            st["ilock"][sl] = int(_safe_param(p, "ilock", 0))
            m_val = _safe_param(p, "mass", 0.0)
            if "mass" in st:
                st["mass"][sl] = m_val
            if "k" in st:
                st["k"][sl] = st["stif0"][sl] + st["stif1"][sl]
            st["uvar4"][sl] = st["stif0"][sl]


def forces_pretensioner_type32(group, x: np.ndarray, v: Optional[np.ndarray], dt: Optional[float], fint: Optional[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Compute axial forces for TYPE32 pretensioner spring elements.

    Ported from $OR_SRC/engine/source/elements/spring/ruser32.F.

    Args:
        group: ElementGroup containing the springs.
        x: Global nodal coordinate array, shape (num_nodes, 3).
        v: Global nodal velocity array, shape (num_nodes, 3).
        dt: Current time step increment.
        fint: Global nodal internal force array, shape (num_nodes, 3).
        idx: Element indices for TYPE32 springs.

    Returns:
        dt_crit: Array of critical time steps for active pretensioner elements.
    """
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

    tacti = np.zeros(len(idx))
    iact = np.ones(len(idx), dtype=bool)

    # Sensor activation evaluation (ruser32.F lines 160-174)
    if sensors is not None:
        for i, s_id in enumerate(sens_id):
            if s_id > 0:
                if sensors.active(s_id):
                    tf = sensors.fire_time.get(s_id, 0.0)
                    tacti[i] = max(0.0, t - tf)
                    iact[i] = True
                else:
                    tacti[i] = 0.0
                    iact[i] = False
            else:
                tacti[i] = t
    else:
        tacti[:] = t

    dt_val = dt if (dt is not None and dt > 0.0) else 0.0

    # Case 1: Inactive sensor (ruser32.F lines 162-171)
    not_act = ~iact
    if np.any(not_act):
        uvar2[not_act] = 0.0
        F[not_act] += stif0[not_act] * dt_val * Ldot[not_act]
        uvar4[not_act] = stif0[not_act]
        st["k"][idx[not_act]] = stif0[not_act]

    # Case 2: Active sensor (ruser32.F lines 173-184)
    act = iact
    if np.any(act):
        # On first cycle of activation: reset stroke X to 0 and set uvar2 = 1
        mask_just_act = act & (uvar2 == 0.0)
        uvar1[mask_just_act] = 0.0
        uvar2[mask_just_act] = 1.0

        # Accumulate stroke X and elastic baseline force
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
                # Linear pretensioner: FF = F1 + STIF1 * X (ruser32.F lines 187-202)
                FF = f1[mask] + stif1[mask] * X
                # Lock condition: external tension exceeds pretension (ILOCK==2)
                cur_uvar3 = np.where((cur_F > FF) & (cur_ilock == 2), 1.0, cur_uvar3)
                # Apply pretension if unlocked
                cur_F = np.where((FF > 0.0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)

            elif it == 2:
                # Non-linear displacement: FF = Scale_f * func1(X * Scale_d) (ruser32.F lines 203-219)
                FF = np.zeros(len(X))
                for local_i, global_i in enumerate(np.where(mask)[0]):
                    f_id = st["fct_id1"][idx[global_i]]
                    func = model.functions.get(f_id) if model is not None and hasattr(model, "functions") else None
                    if func is not None:
                        val = func.eval(X[local_i] * scale_d[global_i]) if hasattr(func, "eval") else func(X[local_i] * scale_d[global_i])
                        FF[local_i] = scale_f[global_i] * val
                # Lock condition: stroke reached D1 or force exceeded FF (ILOCK==2)
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > FF) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0.0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)

            elif it == 3:
                # Time-dependent: F0 = Scale_f * func2(tacti * Scale_t) (ruser32.F lines 220-230)
                F0 = np.zeros(len(X))
                for local_i, global_i in enumerate(np.where(mask)[0]):
                    f_id = st["fct_id2"][idx[global_i]]
                    func = model.functions.get(f_id) if model is not None and hasattr(model, "functions") else None
                    if func is not None:
                        val = func.eval(tacti[global_i] * scale_t[global_i]) if hasattr(func, "eval") else func(tacti[global_i] * scale_t[global_i])
                        F0[local_i] = scale_f[global_i] * val
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > F0) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((F0 > 0.0) & (cur_uvar3 == 0.0), np.maximum(F0, cur_F), cur_F)

            elif it == 4:
                # Combined: FF = F0(tacti) * func1(X * Scale_d) (ruser32.F lines 231-250)
                F0 = np.zeros(len(X))
                FF = np.zeros(len(X))
                for local_i, global_i in enumerate(np.where(mask)[0]):
                    f2_id = st["fct_id2"][idx[global_i]]
                    f1_id = st["fct_id1"][idx[global_i]]
                    f2 = model.functions.get(f2_id) if model is not None and hasattr(model, "functions") else None
                    f1_obj = model.functions.get(f1_id) if model is not None and hasattr(model, "functions") else None
                    if f2 is not None:
                        val2 = f2.eval(tacti[global_i] * scale_t[global_i]) if hasattr(f2, "eval") else f2(tacti[global_i] * scale_t[global_i])
                        F0[local_i] = scale_f[global_i] * val2
                    if f1_obj is not None:
                        val1 = f1_obj.eval(X[local_i] * scale_d[global_i]) if hasattr(f1_obj, "eval") else f1_obj(X[local_i] * scale_d[global_i])
                        FF[local_i] = F0[local_i] * val1
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > FF) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0.0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)

            F[mask] = cur_F
            uvar3[mask] = cur_uvar3

    # Element deletion check
    alive = st.get("off", np.ones(group.n, dtype=float))[idx] > 0.0
    F = np.where(alive, F, 0.0)

    # Internal energy tracking: dE_int = 0.5 * (F_old + F) * Ldot * dt
    F_old = st["force"][idx].copy()
    if dt_val > 0.0:
        st["eint"][idx] += np.where(alive, 0.5 * (F_old + F) * Ldot * dt_val, 0.0)

    st["force"][idx] = F
    st["uvar1"][idx] = uvar1
    st["uvar2"][idx] = uvar2
    st["uvar3"][idx] = uvar3
    if "uvar4" in st:
        st["uvar4"][idx] = uvar4

    # Assemble into fint: node 1 pulled towards node 2 (+F*a), node 2 pulled towards node 1 (-F*a)
    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, conn[:, 0], fvec)
        np.add.at(fint, conn[:, 1], -fvec)

    # Critical explicit time step: omega = 2 * sqrt(k / M) => dt = 2 / omega = sqrt(M / k)
    mass = np.maximum(st["mass"][idx], EM20)
    k_dt = np.maximum(st["k"][idx], 0.0)
    pos_k = (st["k"][idx] > 0.0) & (st["mass"][idx] > 0.0)
    omega = 2.0 * np.sqrt(np.where(pos_k, k_dt / mass, 1.0))
    dt_crit = 2.0 / omega
    return np.where(alive, np.where(pos_k, dt_crit, EP30), EP30)


def implicit_stiffness_type32(group, x: np.ndarray, idx: np.ndarray, ikgeo: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble 6x6 element tangent stiffness for TYPE32 pretensioner spring elements.

    Ported from $OR_SRC/engine/source/elements/spring/ruser32ke3.F
    and ruser32mat3.F.

    Args:
        group: ElementGroup containing the springs.
        x: Current node coordinates.
        idx: Element indices for TYPE32 springs.
        ikgeo: Flag for geometric stiffness (1=include initial stress stiffness).

    Returns:
        ke: Array of (len(idx), 6, 6) element tangent stiffness matrices.
        edofs: Array of (len(idx), 6) DOF indices.
    """
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

    # Material tangent matrix: Ke_mat = Kt * [[a a^T, -a a^T], [-a a^T, a a^T]] (r4sumg3.F)
    aat = a[:, :, None] * a[:, None, :]  # (N, 3, 3)
    ke = np.zeros((len(idx), 6, 6), dtype=np.float64)
    km = k_mat[:, None, None] * aat
    ke[:, :3, :3] = km
    ke[:, :3, 3:] = -km
    ke[:, 3:, :3] = -km
    ke[:, 3:, 3:] = km

    # Geometric stiffness matrix: Kg = (F / L) * [[I - a a^T, -(I - a a^T)], [-(I - a a^T), I - a a^T]] (tkeg3.F)
    if ikgeo == 1:
        F = st["force"][idx]
        f_over_l = F / L
        eye3 = np.eye(3, dtype=np.float64)[None, :, :]
        proj = eye3 - aat  # I - a a^T
        kg = f_over_l[:, None, None] * proj
        ke[:, :3, :3] += kg
        ke[:, :3, 3:] -= kg
        ke[:, 3:, :3] -= kg
        ke[:, 3:, 3:] += kg

    stride = 3
    edofs = np.empty((len(idx), 6), dtype=np.int64)
    edofs[:, 0] = conn[:, 0] * stride
    edofs[:, 1] = conn[:, 0] * stride + 1
    edofs[:, 2] = conn[:, 0] * stride + 2
    edofs[:, 3] = conn[:, 1] * stride
    edofs[:, 4] = conn[:, 1] * stride + 1
    edofs[:, 5] = conn[:, 1] * stride + 2

    return ke, edofs


class SpringPretensioner:
    """Object-oriented TYPE32 pretensioner spring element model.

    Faithful standalone model ported from OpenRadioss Fortran:
      - engine/source/elements/spring/ruser32.F (RUSER32)
      - starter/source/properties/spring/hm_read_prop32.F (HM_READ_PROP32)
      - engine/source/elements/spring/ruser32ke3.F (RUSER32KE3)
      - engine/source/elements/spring/ruser32mat3.F (RUSER32MAT3)
    """

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
        if self.sens_id > 0:
            if sensor_active:
                tacti = max(0.0, t - fire_time)
                iact = True
            else:
                tacti = 0.0
                iact = False
        else:
            tacti = t
            iact = True

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
                if self.force > FF and self.ilock == 2:
                    self.uvar3 = 1.0
                if FF > 0.0 and self.uvar3 == 0.0:
                    self.force = max(FF, self.force)

            elif self.ityp == 2:
                # Non-linear displacement: FF = Scale_f * func1(X * Scale_d)
                val1 = 0.0
                if self.func1 is not None:
                    val1 = self.func1.eval(X * self.scale_d) if hasattr(self.func1, "eval") else self.func1(X * self.scale_d)
                FF = self.scale_f * val1
                if (X < self.d1 and self.d1 != 0.0) or (self.force > FF and self.ilock == 2):
                    self.uvar3 = 1.0
                if FF > 0.0 and self.uvar3 == 0.0:
                    self.force = max(FF, self.force)

            elif self.ityp == 3:
                # Time-dependent: F0 = Scale_f * func2(tacti * Scale_t)
                val2 = 0.0
                if self.func2 is not None:
                    val2 = self.func2.eval(tacti * self.scale_t) if hasattr(self.func2, "eval") else self.func2(tacti * self.scale_t)
                F0 = self.scale_f * val2
                if (X < self.d1 and self.d1 != 0.0) or (self.force > F0 and self.ilock == 2):
                    self.uvar3 = 1.0
                if F0 > 0.0 and self.uvar3 == 0.0:
                    self.force = max(F0, self.force)

            elif self.ityp == 4:
                # Combined: FF = F0(tacti) * func1(X * Scale_d)
                val2 = 0.0
                val1 = 0.0
                if self.func2 is not None:
                    val2 = self.func2.eval(tacti * self.scale_t) if hasattr(self.func2, "eval") else self.func2(tacti * self.scale_t)
                F0 = self.scale_f * val2
                if self.func1 is not None:
                    val1 = self.func1.eval(X * self.scale_d) if hasattr(self.func1, "eval") else self.func1(X * self.scale_d)
                FF = F0 * val1
                if (X < self.d1 and self.d1 != 0.0) or (self.force > FF and self.ilock == 2):
                    self.uvar3 = 1.0
                if FF > 0.0 and self.uvar3 == 0.0:
                    self.force = max(FF, self.force)

        # Work integration: dE_int = 0.5 * (F_old + F) * Ldot * dt
        self.eint += 0.5 * (f_old + self.force) * Ldot * dt
        return self.force

    def critical_dt(self) -> float:
        """Compute critical time step dt = 2 / omega = sqrt(M / K)."""
        if self.mass <= 0.0 or self.uvar4 <= 0.0:
            return EP30
        omega = 2.0 * math.sqrt(self.uvar4 / self.mass)
        return 2.0 / omega
