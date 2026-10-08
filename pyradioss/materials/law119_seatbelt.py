"""
LAW119 — Seatbelt material model (/MAT/LAW119, /MAT/SH_SEATBELT, /PROP/SEATBELT).

Upstream Fortran Reference:
  - ``engine/source/materials/mat/mat119/sigeps119c.F`` (Subroutine SIGEPS119C, lines 33-152)
  - ``engine/source/materials/mat/mat119/law119_membrane.F`` (Subroutine LAW119_MEMBRANE, lines 34-399)
  - ``engine/source/tools/seatbelts/redef_seatbelt.F90`` (Subroutine REDEF_SEATBELT, lines 47-563)
  - ``engine/source/tools/seatbelts/update_slipring.F`` (Subroutine UPDATE_SLIPRING, lines 40-700)
  - ``engine/source/tools/seatbelts/retractor_table_inv.F90`` (Subroutine RETRACTOR_TABLE_INV, lines 30-195)
  - ``starter/source/materials/mat/mat119/hm_read_mat119.F`` (Subroutine HM_READ_MAT119, lines 38-274)
  - ``starter/source/tools/seatbelts/hm_read_retractor.F`` (Subroutine HM_READ_RETRACTOR, lines 45-385)
  - ``starter/source/tools/seatbelts/hm_read_slipring.F`` (Subroutine HM_READ_SLIPRING, lines 45-786)

Features:
  1. 1D Cable/Belt Elements (/PROP/SEATBELT):
     - Strictly tension-only: zero compression stress/force to accommodate belt folding/slack.
     - Nonlinear loading curve (FUN_L) and unloading curve (FUN_UL) with plastic offset tracking.
     - Viscous damping in tension: F_damp = DAMP1 * v_rel.
     - Folding resistance / ribbon bending moment model around edges and D-rings.
     - Sliding friction model: Euler-Eytelwein capstan relation across sliprings:
       T2 = T1 * exp(mu_eff * theta), with dynamic velocity decay.
     - Retractor and pretensioner behavior: sensor trigger, lock threshold, pretensioner
       pull-in force/displacement curves, and optional load-limiter energy absorption.
  2. 2D Shell Seatbelt Elements (/MAT/LAW119, /MAT/SH_SEATBELT):
     - In-plane orthotropic membrane behavior with principal strain tension/compression tagging.
     - Compression branch uses RCOMP (yarn wrinkling/buckling).
     - Hysteresis loading/unloading/reloading with peak equivalent strain tracking.
     - Optional coating layer (ECOAT, NUCOAT, TCOAT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np

from ..model.entities import Material


def _eval_funct(func: Any, x: float) -> float:
    """Evaluate curve function at scalar abscissa x."""
    if func is None or func == 0:
        return 0.0
    if hasattr(func, "eval"):
        return float(func.eval(x))
    if callable(func):
        return float(func(x))
    if isinstance(func, (int, float)):
        return float(func)
    return 0.0


def _eval_funct_deriv(func: Any, x: float) -> float:
    """Evaluate derivative dy/dx of curve function at scalar abscissa x."""
    if func is None or func == 0:
        return 0.0
    if hasattr(func, "slope") and hasattr(func, "x") and len(func.x) > 1:
        x_arr = func.x
        slopes = func.slope
        if x <= x_arr[0]:
            return float(slopes[0])
        if x >= x_arr[-1]:
            return float(slopes[-1])
        idx = int(np.searchsorted(x_arr, x, side="right") - 1)
        idx = max(0, min(len(slopes) - 1, idx))
        return float(slopes[idx])
    if hasattr(func, "eval"):
        h = max(1e-6, abs(x) * 1e-5)
        return float((func.eval(x + h) - func.eval(x - h)) / (2.0 * h))
    if callable(func):
        h = max(1e-6, abs(x) * 1e-5)
        return float((func(x + h) - func(x - h)) / (2.0 * h))
    return 0.0


# ==============================================================================
# 1D Seatbelt Cable / Spring Mechanics
# ==============================================================================

@dataclass
class RetractorState:
    """Internal state of a seatbelt retractor and pretensioner.

    Corresponds to /RETRACTOR in starter/source/tools/seatbelts/hm_read_retractor.F
    and engine/source/tools/seatbelts/retractor_table_inv.F90.
    """
    locked: bool = False
    payout: float = 0.0
    pretens_active: bool = False
    pretens_time: float = 0.0
    pretens_pull: float = 0.0
    ret_force: float = 0.0


def sliding_friction(
    t1: float,
    theta: float,
    mu: float,
    mu_stat: float | None = None,
    v_rel: float = 0.0,
    decay: float = 0.0,
) -> tuple[float, float]:
    """Euler-Eytelwein capstan friction model for seatbelt sliding across a slipring or guide.

    Upstream reference:
      ``starter/source/tools/seatbelts/hm_read_slipring.F`` (lines 95-180)
      ``engine/source/tools/seatbelts/update_slipring.F`` (lines 98-350)

    Formula:
      mu_eff = mu + (mu_stat - mu) * exp(-decay * |v_rel|)
      T2 = T1 * exp(sign(v_rel) * mu_eff * theta)
      F_fric = |T2 - T1|

    Parameters
    ----------
    t1 : float
        Belt tension on upstream strand (must be >= 0).
    theta : float
        Wrap angle in radians (>= 0).
    mu : float
        Dynamic friction coefficient.
    mu_stat : float, optional
        Static friction coefficient (if None, defaults to mu).
    v_rel : float
        Relative belt sliding velocity through slipring.
    decay : float
        Exponential decay parameter from static to dynamic friction.

    Returns
    -------
    t2 : float
        Downstream belt tension.
    f_fric : float
        Friction force transferred to slipring / ring body.
    """
    if t1 <= 0.0 or theta <= 0.0:
        return max(0.0, t1), 0.0

    mu_d = max(0.0, mu)
    mu_s = max(mu_d, mu_stat if mu_stat is not None else mu_d)

    if decay > 0.0 and abs(v_rel) > 0.0:
        mu_eff = mu_d + (mu_s - mu_d) * math.exp(-decay * abs(v_rel))
    else:
        mu_eff = mu_s if abs(v_rel) < 1e-12 else mu_d

    sign_v = 1.0 if v_rel >= 0.0 else -1.0
    exponent = max(-50.0, min(50.0, sign_v * mu_eff * theta))
    ratio = math.exp(exponent)
    t2 = t1 * ratio
    f_fric = abs(t2 - t1)
    return t2, f_fric


# Alias
slipring_friction = sliding_friction


def folding_moment(theta: float, k_fold: float = 0.0, m_max: float = 0.0) -> float:
    """Folding resistance / ribbon bending moment for 1D/2D seatbelt folding.

    Upstream reference:
      ``engine/source/tools/seatbelts/redef_seatbelt.F90`` (Ibend / Itors terms)

    Parameters
    ----------
    theta : float
        Bending / folding angle in radians.
    k_fold : float
        Bending stiffness per unit angle.
    m_max : float
        Maximum allowable bending moment (plastic hinge).

    Returns
    -------
    M : float
        Restoring folding moment.
    """
    if k_fold <= 0.0 or abs(theta) <= 1e-12:
        return 0.0
    m_lin = k_fold * abs(theta)
    if m_max > 0.0:
        m_lin = min(m_max, m_lin)
    return math.copysign(m_lin, theta)


def retractor_update(
    state: RetractorState,
    pullout_disp: float,
    pullout_vel: float = 0.0,
    dt: float = 0.0,
    sensor_lock: bool = False,
    sensor_pretens: bool = False,
    pretens_curve: Any = None,
    pretens_force: float = 0.0,
    lock_threshold: float = 0.0,
    load_limit: float = 0.0,
    functions: dict[int, Any] | None = None,
) -> tuple[float, RetractorState]:
    """Update seatbelt retractor and pretensioner state.

    Upstream reference:
      ``starter/source/tools/seatbelts/hm_read_retractor.F`` (lines 110-230)
      ``engine/source/tools/seatbelts/retractor_table_inv.F90`` (lines 40-190)

    Behaviors:
      1. Locking:
         Triggered if sensor_lock is True, or if pullout exceeds lock_threshold.
         Once locked, spool cannot freely pay out.
      2. Pretensioner:
         Triggered by sensor_pretens. Advances pretens_time, applies pull-in
         displacement or tension force up to pretens_force.
      3. Load Limiter:
         If locked belt tension exceeds load_limit, the retractor yields and
         pays out belt at constant force (energy absorption).

    Returns
    -------
    force : float
        Retractor holding / pulling force.
    state : RetractorState
        Updated state.
    """
    # Check lock conditions
    if sensor_lock:
        state.locked = True
    elif lock_threshold > 0.0 and (pullout_disp >= lock_threshold or pullout_vel > 2.0):
        state.locked = True

    # Check pretensioner activation
    if sensor_pretens and not state.pretens_active:
        state.pretens_active = True
        state.locked = True

    force = 0.0
    if state.pretens_active:
        state.pretens_time += max(0.0, dt)
        f_pre = pretens_force
        if pretens_curve is not None:
            func = functions.get(pretens_curve) if (functions and pretens_curve in functions) else pretens_curve
            f_val = _eval_funct(func, state.pretens_time)
            if f_val != 0.0:
                f_pre = f_val
        force = max(force, f_pre)
        state.pretens_pull += max(0.0, -pullout_vel * dt)

    if state.locked:
        # Belt is locked: resisting pullout
        if pullout_disp > 0.0:
            k_ret = 10000.0  # nominal retractor lock stiffness
            force = max(force, k_ret * pullout_disp)
        if load_limit > 0.0 and force > load_limit:
            # Load limiter activation: plastic spool payout
            force = load_limit
            state.payout += max(0.0, pullout_vel * dt)
    else:
        # Free payout: small rewind spring tension
        force = min(force, 5.0)
        state.payout = max(0.0, pullout_disp)

    state.ret_force = force
    return force, state


def cable_update(
    mat: Any,
    L: float,
    L0: float,
    v_rel: float = 0.0,
    state: dict[str, Any] | None = None,
    dt: float = 0.0,
    functions: dict[int, Any] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Compute 1D seatbelt cable force, tangent stiffness, and internal state.

    Upstream Fortran reference:
      ``engine/source/tools/seatbelts/redef_seatbelt.F90`` (Subroutine REDEF_SEATBELT)

    Physics:
      - Strictly tension-only: if delta_L <= 0 or elastic strain <= 0, force = 0, k_tan = 0.
        Compressive deformation corresponds to slack/folding without carrying force.
      - Tension branch: follows nonlinear loading curve FUN_L (scaled by Fcoeft1, Xcoeft1)
        or linear stiffness STIFF1.
      - Hysteresis unloading: follows FUN_UL tracking maximum strain and permanent plastic offset dpx.
      - Rate sensitivity: factor 1 + C_RATE * ln(max(1, |eps_dot| / eps0)).
      - Damping: F_damp = DAMP1 * v_rel (tension only).
    """
    if state is None:
        state = {
            "yield_f": 0.0,
            "eps_max": 0.0,
            "dpx": 0.0,
            "force_old": 0.0,
            "eint": 0.0,
            "eps_old": 0.0,
            "delta_old": 0.0,
        }

    p = getattr(mat, "params", {})
    lmin = getattr(mat, "lmin", float(p.get("LMIN", p.get("lmin", 0.0))))
    l_ref = max(L0, lmin) if lmin > 0.0 else L0
    if l_ref <= 0.0:
        l_ref = 1.0

    delta = L - L0
    eps = delta / l_ref

    # Rate sensitivity
    c_rate = getattr(mat, "c_rate", float(p.get("C_RATE", p.get("c_rate", 0.0))))
    eps0_rate = getattr(mat, "eps0_rate", float(p.get("EPS0", p.get("eps0", 1.0))))
    rate_fac = 1.0
    if abs(dt) > 0.0 and c_rate > 0.0 and eps0_rate > 0.0:
        eps_dot = abs(v_rel) / l_ref
        rate_fac = 1.0 + c_rate * math.log(max(1.0, eps_dot / eps0_rate))

    dpx = state.get("dpx", 0.0)
    yield_f = state.get("yield_f", 0.0)
    f_old = state.get("force_old", 0.0)
    eps_ela = eps - dpx

    # Tension-only test: zero compression stress / force
    is_tension = (delta > 0.0 and eps_ela > 0.0)

    if not is_tension:
        # Belt is slack / folding: zero force and zero stiffness
        F = 0.0
        k_tan = 0.0
        if dpx <= 0.0:
            dpx = 0.0
    else:
        # Retrieve loading and unloading curves
        func_l_id = getattr(mat, "fun_l", p.get("FUN_L", p.get("fun_l", 0)))
        func_ul_id = getattr(mat, "fun_ul", p.get("FUN_UL", p.get("fun_ul", 0)))
        fscale1 = getattr(mat, "fscale1", float(p.get("Fcoeft1", p.get("fscale1", 1.0))))
        fscale2 = getattr(mat, "fscale2", float(p.get("Fcoeft2", p.get("fscale2", 1.0))))
        stiff1 = getattr(mat, "e11", float(p.get("STIFF1", p.get("stiff1", p.get("E11", 1000.0)))))
        damp1 = float(p.get("DAMP1", p.get("damp1", 0.0)))

        f_load = None
        f_unload = None
        if functions is not None:
            if func_l_id in functions:
                f_load = functions[func_l_id]
            if func_ul_id in functions:
                f_unload = functions[func_ul_id]
        if f_load is None and (hasattr(func_l_id, "eval") or callable(func_l_id)):
            f_load = func_l_id
        if f_unload is None and (hasattr(func_ul_id, "eval") or callable(func_ul_id)):
            f_unload = func_ul_id
        if f_unload is None and f_load is not None:
            f_unload = f_load

        if f_load is not None:
            x_eval = max(0.0, eps)
            f_curve = fscale1 * _eval_funct(f_load, x_eval) * rate_fac
            k_load = (fscale1 * _eval_funct_deriv(f_load, x_eval) * rate_fac) / l_ref

            x_unl = max(0.0, eps)
            k_unl_deriv = _eval_funct_deriv(f_unload, x_unl)
            if k_unl_deriv <= 0.0:
                k_unl_deriv = _eval_funct_deriv(f_load, x_unl)
            if k_unl_deriv <= 0.0:
                k_unl_deriv = stiff1 if stiff1 > 0.0 else 1.0
            k_unl = (fscale2 * k_unl_deriv * rate_fac) / l_ref
            if k_unl <= 0.0:
                k_unl = max(stiff1, 1.0) / l_ref

            eps_max = state.get("eps_max", 0.0)
            if f_curve >= yield_f and eps >= eps_max:
                # Primary loading
                F = f_curve
                yield_f = F
                state["eps_max"] = eps
                k_tan = k_load
                dpx = max(0.0, eps - F / (k_unl * l_ref))
            else:
                # Hysteresis unloading / reloading
                deps = eps - state.get("eps_old", eps)
                F = f_old + k_unl * (deps * l_ref)
                F = max(0.0, min(yield_f, F))
                k_tan = k_unl if F > 0.0 else 0.0
        else:
            # Linear elastic in tension
            k_eff = stiff1 if stiff1 > 0.0 else 1.0
            F = k_eff * max(0.0, delta) * rate_fac
            k_tan = k_eff * rate_fac

        # Viscous damping in tension
        if damp1 > 0.0 and F > 0.0:
            f_damp = damp1 * v_rel
            F = max(0.0, F + f_damp)

    # Energy accounting: dE = 0.5 * (F_old + F) * d_delta
    delta_old = state.get("delta_old", delta)
    d_delta = delta - delta_old
    dE = 0.5 * (f_old + F) * d_delta
    state["eint"] = state.get("eint", 0.0) + dE

    state["yield_f"] = yield_f
    state["dpx"] = dpx
    state["force_old"] = F
    state["eps_old"] = eps
    state["delta_old"] = delta

    return F, k_tan, state


spring_update = cable_update
belt_1d_update = cable_update


def compute_force(
    delta_L: float,
    v_rel: float = 0.0,
    L0: float = 1.0,
    state: dict[str, Any] | None = None,
    dt: float = 0.0,
    functions: dict[int, Any] | None = None,
    mat: Any = None,
) -> tuple[float, dict[str, Any]]:
    """Compute axial force F and update state for 1D seatbelt."""
    m = mat if mat is not None else Law119Seatbelt(id=1, params={})
    F, _, updated_state = cable_update(
        mat=m,
        L=L0 + delta_L,
        L0=L0,
        v_rel=v_rel,
        state=state,
        dt=dt,
        functions=functions,
    )
    return F, updated_state


# ==============================================================================
# Material Class
# ==============================================================================

class Law119Seatbelt(Material):
    """LAW119 (/MAT/SH_SEATBELT, /PROP/SEATBELT) seatbelt material."""

    def __init__(self, id: int, rho0: float = 0.0, title: str = "", params: dict[str, Any] | None = None):
        p = params or {}
        super().__init__(id=id, law=119, rho0=rho0, title=title, params=p)
        self.e11 = float(p.get("STIFF1", p.get("stiff1", p.get("E11", p.get("e11", p.get("E", 1.0))))))
        self.e22 = float(p.get("E22", p.get("e22", 0.0)))
        self.nu12 = float(p.get("NU12", p.get("nu12", p.get("nu", 0.19))))
        self.g12 = float(p.get("G12", p.get("g12", 0.0)))
        self.re = float(p.get("RE", p.get("re", p.get("rcomp", p.get("RCOMP", 1.0)))))
        self.rcomp = max(1e-3, self.re if self.re > 0.0 else 1.0)
        self.lmin = float(p.get("LMIN", p.get("lmin", 0.0)))
        self.fun_l = p.get("FUN_L", p.get("fun_l", 0))
        self.fun_ul = p.get("FUN_UL", p.get("fun_ul", 0))
        self.fscale1 = float(p.get("Fcoeft1", p.get("fscale1", p.get("fcoeft1", 1.0))))
        self.fscale2 = float(p.get("Fcoeft2", p.get("fscale2", p.get("fcoeft2", 1.0))))
        self.fscale22 = float(p.get("Fcoeft22", p.get("fscale22", p.get("fcoeft22", 1.0))))
        self.ireload = int(p.get("Ireload", p.get("ireload", 0)))
        self.ecoat = float(p.get("ECOAT", p.get("ecoat", 0.0)))
        self.nucoat = float(p.get("NUCOAT", p.get("nucoat", self.nu12)))
        self.tcoat = float(p.get("TCOAT", p.get("tcoat", 0.0)))
        self.c_rate = float(p.get("C_RATE", p.get("c_rate", 0.0)))
        self.eps0_rate = float(p.get("EPS0", p.get("eps0", p.get("eps0_rate", 1.0))))

        # Derived orthotropic constants
        fscalet_input = p.get("Fcoeft22", p.get("fscale22", p.get("fcoeft22", None)))
        if fscalet_input is not None and float(fscalet_input) > 0.0:
            fscalet = float(fscalet_input)
        elif self.e11 > 0.0 and self.e22 > 0.0:
            fscalet = self.e22 / self.e11
        else:
            fscalet = 0.1
        self.fscalet = fscalet
        self.nu21 = self.nu12 * fscalet
        self.det = 1.0 / max(1e-12, 1.0 - self.nu12 * self.nu21)
        self.a11 = self.e11 * self.det
        self.a22 = self.a11 * fscalet
        self.a12 = self.a11 * self.nu21

        # Fallback shear modulus if zero
        if self.g12 <= 0.0:
            self.g12 = self.e11 / (2.0 * (1.0 + self.nu12))

    @property
    def G(self) -> float:
        return self.g12

    def sound_speed_shell(self) -> float:
        """Characteristic acoustic sound speed for Courant time step."""
        c1 = max(self.e11, self.e22) * self.det
        rho = max(self.rho0, 1e-12)
        return float(math.sqrt(c1 / rho))

    def cable_update(
        self,
        L: float,
        L0: float,
        v_rel: float = 0.0,
        state: dict[str, Any] | None = None,
        dt: float = 0.0,
        functions: dict[int, Any] | None = None,
    ) -> tuple[float, float, dict[str, Any]]:
        return cable_update(self, L, L0, v_rel=v_rel, state=state, dt=dt, functions=functions)

    def spring_update(
        self,
        L: float,
        L0: float,
        v_rel: float = 0.0,
        state: dict[str, Any] | None = None,
        dt: float = 0.0,
        functions: dict[int, Any] | None = None,
    ) -> tuple[float, float, dict[str, Any]]:
        return cable_update(self, L, L0, v_rel=v_rel, state=state, dt=dt, functions=functions)

    def compute_force(
        self,
        delta_L: float,
        v_rel: float = 0.0,
        L0: float = 1.0,
        state: dict[str, Any] | None = None,
        dt: float = 0.0,
        functions: dict[int, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        return compute_force(delta_L, v_rel=v_rel, L0=L0, state=state, dt=dt, functions=functions, mat=self)

    def sliding_friction(
        self,
        t1: float,
        theta: float,
        mu: float,
        mu_stat: float | None = None,
        v_rel: float = 0.0,
        decay: float = 0.0,
    ) -> tuple[float, float]:
        return sliding_friction(t1, theta, mu, mu_stat=mu_stat, v_rel=v_rel, decay=decay)

    def retractor_update(
        self,
        state: RetractorState,
        pullout_disp: float,
        pullout_vel: float = 0.0,
        dt: float = 0.0,
        sensor_lock: bool = False,
        sensor_pretens: bool = False,
        pretens_curve: Any = None,
        pretens_force: float = 0.0,
        lock_threshold: float = 0.0,
        load_limit: float = 0.0,
        functions: dict[int, Any] | None = None,
    ) -> tuple[float, RetractorState]:
        return retractor_update(
            state,
            pullout_disp,
            pullout_vel=pullout_vel,
            dt=dt,
            sensor_lock=sensor_lock,
            sensor_pretens=sensor_pretens,
            pretens_curve=pretens_curve,
            pretens_force=pretens_force,
            lock_threshold=lock_threshold,
            load_limit=load_limit,
            functions=functions,
        )


# ==============================================================================
# 2D Shell Mechanics (sigeps119c.F & law119_membrane.F)
# ==============================================================================

def extra_shapes(mat: Any, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Persistent state per shell integration point for LAW119 seatbelt."""
    return {
        "eps119": (nip, 3) if nip else (3,),
        "uv119": (nip, 10) if nip else (10,),
    }


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """(3, 3) plane-stress orthotropic membrane tangent matrix."""
    a11 = getattr(mat, "a11", mat.params.get("A11", 1.0))
    a22 = getattr(mat, "a22", mat.params.get("A22", 1.0))
    a12 = getattr(mat, "a12", mat.params.get("A12", 0.0))
    g12 = getattr(mat, "g12", mat.params.get("G12", 0.0))
    return np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=float)


def consistent_shell_tangent(mat: Any, sig: np.ndarray | None = None, deps: np.ndarray | None = None, extra: dict | None = None) -> np.ndarray:
    """(3, 3) or (n, 3, 3) consistent shell tangent tensor for LAW119."""
    c_el = shell_membrane_tangent(mat)
    if extra is not None and "eps119" in extra:
        eps = extra["eps119"]
        if eps.ndim == 2:
            n = eps.shape[0]
            s = 0.5 * (eps[:, 0] + eps[:, 1])
            d = 0.5 * (eps[:, 0] - eps[:, 1])
            r = np.sqrt(eps[:, 2]**2 + d**2)
            p1 = s + r
            p2 = s - r
            is_tension = (p1 > 0.0) & (p1 >= -p2)
            rcomp = getattr(mat, "rcomp", 1.0)
            beta = np.where(is_tension, 1.0, rcomp)[:, None, None]
            return np.broadcast_to(c_el, (n, 3, 3)) * beta
    return c_el


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized layer update for LAW119 shell elements (sigeps119c.F & law119_membrane.F)."""
    sig_shape = sig.shape
    if sig.ndim == 1:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        return sig.reshape(sig_shape), epsp

    p = getattr(mat, "params", {})
    a11 = getattr(mat, "a11", float(p.get("A11", p.get("E", 1.0))))
    a22 = getattr(mat, "a22", float(p.get("A22", p.get("E", 1.0))))
    a12 = getattr(mat, "a12", float(p.get("A12", 0.0)))
    g12 = getattr(mat, "g12", float(p.get("G12", p.get("G", 0.0))))
    nu12 = getattr(mat, "nu12", float(p.get("NU12", 0.19)))
    nu21 = getattr(mat, "nu21", nu12 * 0.1)
    det = getattr(mat, "det", 1.0 / max(1e-12, 1.0 - nu12 * nu21))
    fscalet = getattr(mat, "fscalet", 0.1)
    rcomp = getattr(mat, "rcomp", max(1e-3, float(p.get("RE", p.get("re", 1.0)))))
    fscale1 = getattr(mat, "fscale1", float(p.get("Fcoeft1", 1.0)))
    fscale2 = getattr(mat, "fscale2", float(p.get("Fcoeft2", 1.0)))
    ireload = getattr(mat, "ireload", int(p.get("Ireload", 0)))
    c_rate = getattr(mat, "c_rate", float(p.get("C_RATE", 0.0)))
    eps0_rate = getattr(mat, "eps0_rate", float(p.get("EPS0", 1.0)))

    # Functions retrieval
    func = getattr(mat, "fun_l", p.get("FUN_L", p.get("fun_l", 0)))
    fund = getattr(mat, "fun_ul", p.get("FUN_UL", p.get("fun_ul", 0)))
    functions = getattr(mat, "functions", None)
    if functions is None and extra is not None:
        functions = extra.get("functions")

    f_l = None
    f_ul = None
    if functions is not None:
        if func in functions:
            f_l = functions[func]
        if fund in functions:
            f_ul = functions[fund]
    if f_l is None and (hasattr(func, "eval") or callable(func)):
        f_l = func
    if f_ul is None and (hasattr(fund, "eval") or callable(fund)):
        f_ul = fund

    if extra is None:
        extra = {}
    eps = extra.get("eps119")
    if not isinstance(eps, np.ndarray) or eps.shape[0] != n:
        eps = np.zeros((n, 3), dtype=float)
        extra["eps119"] = eps

    uv = extra.get("uv119")
    if not isinstance(uv, np.ndarray) or uv.shape[0] != n:
        uv = np.zeros((n, 10), dtype=float)
        extra["uv119"] = uv

    # Accumulate in-plane strain
    eps[:, :min(3, deps.shape[1])] += deps[:, :min(3, deps.shape[1])]
    sigo = sig.copy()

    # Principal strain calculation (law119_membrane.F:126-146)
    s = 0.5 * (eps[:, 0] + eps[:, 1])
    d = 0.5 * (eps[:, 0] - eps[:, 1])
    r = np.sqrt(eps[:, 2]**2 + d**2)
    p1 = s + r
    p2 = s - r
    is_tension = (p1 > 0.0) & (p1 >= -p2)
    beta = np.where(is_tension, 1.0, rcomp)

    # In-plane shear stresses
    sig[:, 2] = g12 * eps[:, 2] * beta
    if sig.shape[1] >= 5 and deps.shape[1] >= 5:
        sig[:, 3] += g12 * deps[:, 3] * beta
        sig[:, 4] += g12 * deps[:, 4] * beta

    # Direct in-plane stresses
    if f_l is None:
        # Linear elastic orthotropic fabric
        sig[:, 0] = (a11 * eps[:, 0] + a12 * eps[:, 1]) * beta
        sig[:, 1] = (a12 * eps[:, 0] + a22 * eps[:, 1]) * beta
    else:
        # Nonlinear fabric behavior
        epsq = np.sqrt((eps[:, 0]**2 + eps[:, 1]**2) / (1.0 + nu21**2))

        # Strain-rate scaling factor
        rate_fac = 1.0
        if dt > 0.0 and c_rate > 0.0 and eps0_rate > 0.0:
            depsq = np.abs(epsq - uv[:, 7])
            eps_dot = depsq / dt
            rate_fac = 1.0 + c_rate * np.log(np.maximum(1.0, eps_dot / eps0_rate))

        if f_ul is None:
            # Nonlinear loading curve without separate unloading curve
            etl = np.array([_eval_funct_deriv(f_l, float(eq)) for eq in epsq]) * fscale1 * rate_fac
            a11_eff = etl * det
            a22_eff = a11_eff * fscalet
            a12_eff = a11_eff * nu21

            sig_tens_xx = sigo[:, 0] + a11_eff * deps[:, 0] + a12_eff * deps[:, 1]
            sig_tens_yy = sigo[:, 1] + a12_eff * deps[:, 0] + a22_eff * deps[:, 1]
            sig_comp_xx = (a11 * eps[:, 0] + a12 * eps[:, 1]) * rcomp
            sig_comp_yy = (a12 * eps[:, 0] + a22 * eps[:, 1]) * rcomp

            sig[:, 0] = np.where(is_tension, sig_tens_xx, sig_comp_xx)
            sig[:, 1] = np.where(is_tension, sig_tens_yy, sig_comp_yy)
        else:
            # Full hysteresis (loading, unloading, reloading) - law119_membrane.F:254-379
            xint = float(p.get("XINT", 1.0))
            yint = float(p.get("YINT", 1.0))

            for i in range(n):
                eq_i = epsq[i]
                if not is_tension[i]:
                    # Compression branch
                    sig[i, 0] = (a11 * eps[i, 0] + a12 * eps[i, 1]) * rcomp
                    sig[i, 1] = (a12 * eps[i, 0] + a22 * eps[i, 1]) * rcomp
                    uv[i, 0] = 1e-20  # EMAX
                    uv[i, 1] = 0.0    # SMAX
                    uv[i, 2] = 0.0    # EMINRL
                    uv[i, 3] = 1e-20  # EMAXRL
                    uv[i, 4] = 0.0    # SMINRL
                    uv[i, 5] = 0.0    # SMAXRL
                    uv[i, 9] = -1.0   # compression flag
                else:
                    dw = eq_i - uv[i, 7]
                    svm = math.sqrt(max(0.0, sigo[i, 0]**2 + sigo[i, 1]**2 - sigo[i, 0] * sigo[i, 1]))

                    if dw < 0.0 and uv[i, 9] >= 0.0:
                        # Unloading branch
                        emax_rl = max(1e-20, uv[i, 3])
                        smax_rl = max(1e-20, uv[i, 5])
                        x_val = eq_i * xint / emax_rl
                        etu = _eval_funct_deriv(f_ul, x_val)
                        etx = etu * (smax_rl / emax_rl) * (xint / max(1e-20, yint))
                        if eq_i > 0.0:
                            etx = max(etx, svm / eq_i)
                        a11_eff = etx * det * fscale2 * rate_fac
                        a22_eff = a11_eff * fscalet
                        a12_eff = a11_eff * nu21

                        sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                        sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                        new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))
                        uv[i, 2] = eq_i      # EMINRL
                        uv[i, 4] = new_svm   # SMINRL

                    elif svm >= uv[i, 1] or uv[i, 9] == -1.0:
                        # Loading branch
                        etl = _eval_funct_deriv(f_l, eq_i) * fscale1 * rate_fac
                        a11_eff = etl * det
                        a22_eff = a11_eff * fscalet
                        a12_eff = a11_eff * nu21

                        sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                        sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                        new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))

                        uv[i, 0] = max(1e-20, eq_i)    # EMAX
                        uv[i, 1] = new_svm             # SMAX
                        uv[i, 2] = eq_i                # EMINRL
                        uv[i, 3] = max(1e-20, eq_i)    # EMAXRL
                        uv[i, 4] = new_svm             # SMINRL
                        uv[i, 5] = new_svm             # SMAXRL
                        uv[i, 9] = 0.0                 # clear compression flag

                    else:
                        # Reloading branch
                        emax_val = max(1e-20, uv[i, 0])
                        smax_val = max(1e-20, uv[i, 1])
                        emin_rl = uv[i, 2]
                        smin_rl = uv[i, 4]

                        if ireload == 1:
                            # Reloading follows loading curve
                            etl = _eval_funct_deriv(f_l, eq_i) * fscale1 * rate_fac
                            ht = (smax_val - smin_rl) / max(1e-20, emax_val - emin_rl)
                            hf = smax_val / emax_val
                            etx = etl * ht / max(1e-20, hf)
                            a11_eff = etx * det
                            a22_eff = a11_eff * fscalet
                            a12_eff = a11_eff * nu21

                            sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                            sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                            new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))
                            uv[i, 3] = max(1e-20, eq_i)
                            uv[i, 5] = new_svm
                        else:
                            # Reloading follows unloading curve
                            emax_rl = max(1e-20, uv[i, 3])
                            x_val = eq_i * xint / emax_rl
                            etu = _eval_funct_deriv(f_ul, x_val)
                            ht = smax_val / emax_val
                            hf = yint / max(1e-20, xint)
                            etx = fscale2 * etu * ht / max(1e-20, hf) * rate_fac
                            a11_eff = etx * det
                            a22_eff = a11_eff * fscalet
                            a12_eff = a11_eff * nu21

                            sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                            sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                            new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))
                            uv[i, 3] = max(1e-20, emax_val)
                            uv[i, 5] = smax_val

        uv[:, 7] = epsq

    # Coating layer contribution (sigeps119c.F:116-132)
    ecoat = getattr(mat, "ecoat", float(p.get("ECOAT", 0.0)))
    tcoat = getattr(mat, "tcoat", float(p.get("TCOAT", 0.0)))
    nucoat = getattr(mat, "nucoat", float(p.get("NUCOAT", nu12)))
    if ecoat > 0.0 and tcoat > 0.0:
        a1c = ecoat / max(1e-12, 1.0 - nucoat**2)
        a2c = a1c * nucoat
        gc = ecoat / (2.0 * (1.0 + nucoat))
        sig_coat_xx = a1c * eps[:, 0] + a2c * eps[:, 1]
        sig_coat_yy = a2c * eps[:, 0] + a1c * eps[:, 1]
        sig_coat_xy = gc * eps[:, 2]

        thk = extra.get("thk") if extra is not None else None
        if thk is not None:
            w_c = np.clip(2.0 * tcoat / np.maximum(thk, 1e-12), 0.0, 1.0)
            sig[:, 0] = (1.0 - w_c) * sig[:, 0] + w_c * sig_coat_xx
            sig[:, 1] = (1.0 - w_c) * sig[:, 1] + w_c * sig_coat_yy
            sig[:, 2] = (1.0 - w_c) * sig[:, 2] + w_c * sig_coat_xy
        else:
            sig[:, 0] += sig_coat_xx * (tcoat / max(1e-3, getattr(mat, "lmin", 1.0)))
            sig[:, 1] += sig_coat_yy * (tcoat / max(1e-3, getattr(mat, "lmin", 1.0)))
            sig[:, 2] += sig_coat_xy * (tcoat / max(1e-3, getattr(mat, "lmin", 1.0)))

    return sig.reshape(sig_shape), epsp


# ==============================================================================
# Dispatcher Functions & Template API
# ==============================================================================

def solid_update(*args: Any, **kwargs: Any) -> Any:
    r"""Solid update stub per template.

    Function: SIGEPS_119 (lines 33-152) in
    $OR_SRC/engine/source/materials/mat/mat119/sigeps119c.F

    LAW119 is formulated specifically for 2D shell seatbelts and 1D cable/spring seatbelts.
    Solid elements are not supported (stubbed per template).
    """
    if len(args) == 7 or "fint" in kwargs:
        # Template call: solid_update(group, x, u, ur, dt, fint, mint)
        # TODO: port from $OR_SRC/engine/source/materials/mat/mat119/sigeps119c.F
        return None
    # Constitutive call: solid_update(mat, sig, deps, ...)
    raise NotImplementedError("LAW119 (/MAT/SH_SEATBELT) is implemented for shell and seatbelt elements only.")


def tangent(*args: Any, **kwargs: Any) -> Any:
    r"""Consistent tangent operator for LAW119 per template.

    Function: SIGEPS_119 (lines 33-152) in
    $OR_SRC/engine/source/materials/mat/mat119/sigeps119c.F
    """
    if len(args) == 1 and hasattr(args[0], "mat"):
        # Template call: tangent(group)
        # TODO: port from $OR_SRC/engine/source/materials/mat/mat119/sigeps119c.F
        return None
    mat = args[0] if args else kwargs.get("mat")
    if mat is not None:
        return consistent_shell_tangent(mat, **kwargs)
    return None


def sound_speed(mat: Any, rho: float | None = None, extra: dict | None = None, is_shell: bool = True) -> float:
    """Characteristic sound speed for LAW119."""
    if hasattr(mat, "sound_speed_shell"):
        return mat.sound_speed_shell()
    e11 = getattr(mat, "e11", mat.params.get("E11", mat.params.get("E", 1.0)))
    e22 = getattr(mat, "e22", mat.params.get("E22", 0.0))
    det = getattr(mat, "det", 1.0)
    c1 = max(e11, e22) * det
    r0 = rho if rho is not None else getattr(mat, "rho0", 1.0)
    return float(math.sqrt(max(c1, 1e-12) / max(r0, 1e-12)))


def build_law119(rec: Any) -> Law119Seatbelt:
    """Constructor for /MAT/LAW119 (/MAT/SH_SEATBELT) 2D shell seatbelt material."""
    if isinstance(rec, dict):
        p = rec.get("params", rec)
        rec_id = int(rec.get("id", p.get("id", 1)))
        rec_rho = float(rec.get("rho0", rec.get("rho", rec.get("density", p.get("rho0", p.get("rho", 0.0))))))
        rec_title = str(rec.get("title", p.get("title", "")))
    else:
        p = getattr(rec, "params", {})
        rec_id = getattr(rec, "id", 1)
        rec_rho = getattr(rec, "density", getattr(rec, "rho0", 0.0))
        rec_title = getattr(rec, "title", "")

    stiff1 = float(p.get("STIFF1", p.get("stiff1", p.get("E11", 0.0))))
    damp1 = float(p.get("DAMP1", p.get("damp1", 0.0)))
    re = float(p.get("RE", p.get("re", 1.0)))
    lmin = float(p.get("LMIN", p.get("lmin", 0.0)))
    fun_l = p.get("FUN_L", p.get("fun_l", 0))
    fun_ul = p.get("FUN_UL", p.get("fun_ul", 0))
    fcoeft1 = float(p.get("Fcoeft1", p.get("fcoeft1", p.get("fscale1", 1.0))))
    fcoeft2 = float(p.get("Fcoeft2", p.get("fcoeft2", p.get("fscale2", 1.0))))
    ireload = int(p.get("Ireload", p.get("ireload", 0)))
    e22 = float(p.get("E22", p.get("e22", 0.0)))
    nu12 = float(p.get("NU12", p.get("nu12", 0.19)))
    g12 = float(p.get("G12", p.get("g12", 0.0)))
    e = e22 if e22 > 0.0 else (stiff1 if stiff1 > 0.0 else 1.0)
    nu = nu12 if (0.0 <= nu12 < 0.5) else 0.19

    params = {
        "E": e,
        "nu": nu,
        "STIFF1": stiff1,
        "stiff1": stiff1,
        "E11": stiff1,
        "e11": stiff1,
        "damp1": damp1,
        "re": re,
        "RE": re,
        "RCOMP": re,
        "lmin": lmin,
        "fun_l": fun_l,
        "fun_ul": fun_ul,
        "fcoeft1": fcoeft1,
        "fscale1": fcoeft1,
        "fcoeft2": fcoeft2,
        "fscale2": fcoeft2,
        "ireload": ireload,
        "e22": e22,
        "E22": e22,
        "nu12": nu12,
        "NU12": nu12,
        "g12": g12,
        "G12": g12,
        "fcoeft22": float(p.get("Fcoeft22", p.get("fcoeft22", p.get("fscale22", 0.0)))),
        "fscale22": float(p.get("Fcoeft22", p.get("fcoeft22", p.get("fscale22", 0.0)))),
        "ecoat": float(p.get("ECOAT", p.get("ecoat", 0.0))),
        "nucoat": float(p.get("NUCOAT", p.get("nucoat", 0.0))),
        "tcoat": float(p.get("TCOAT", p.get("tcoat", 0.0))),
        "c_rate": float(p.get("C_RATE", p.get("c_rate", 0.0))),
        "eps0_rate": float(p.get("EPS0", p.get("eps0", p.get("eps0_rate", 1.0)))),
    }
    return Law119Seatbelt(id=rec_id, rho0=rec_rho, title=rec_title, params=params)


def _register() -> None:
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW119"] = build_law119
    MAT_PHYSICS_REGISTRY["SH_SEATBELT"] = build_law119
    MAT_PHYSICS_REGISTRY["MAT_LAW119"] = build_law119
    MAT_PHYSICS_REGISTRY["MAT_SH_SEATBELT"] = build_law119


_register()
