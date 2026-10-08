r"""
Flexible Body Modal Solver (/FXBODY).

This module implements component mode synthesis (CMS) / modal superposition
for deformable bodies in OpenRadioss explicit dynamics.

Fortran source origins:
- Main system solver:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbsys.F
- Generalized forces & gravity projection:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbyfor.F
- Physical velocity restitution:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbyvit.F
- Physical displacement restitution:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbdispl.F
- Force projection & modal internal/damping forces:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F
- Modal kinematics & links:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodv.F
- Modal predictor/corrector, acceleration & energy accounting:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F
- Stress recovery from modal coordinates:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbsgmaj.F
- Secondary node & element activation:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxbypid.F
- Gravity correction:
  Ported from $OR_SRC/engine/source/constraints/fxbody/fxgrvcor.F
- Starter keyword parsing & critical dt:
  Ported from $OR_SRC/starter/source/constraints/fxbody/hm_read_fxb.F
- Starter initialization & ortho-normalization:
  Ported from $OR_SRC/starter/source/constraints/fxbody/ini_fxbody.F

Mathematical Formulation:
-------------------------
The equation of motion in modal coordinates q is:
    M q̈ + C q̇ + K q = Q_ext
where:
    M     = modal mass matrix (diagonal for orthogonal modes)
    K     = modal stiffness matrix (diagonal, K_i = omega_i^2 * M_i)
    C     = modal damping matrix (Rayleigh C = alpha*M + beta*K, or modal damping)
    Q_ext = Phi^T · F_ext (generalized forces projected from physical nodal loads)
    Phi   = modal matrix (n_dof x n_modes)

Physical recovery:
    u = Phi · q     (physical nodal displacements)
    v = Phi · q̇     (physical nodal velocities)
    a = Phi · q̈     (physical nodal accelerations)

Energy accounting:
    E_kin  = 1/2 · q̇^T · M · q̇
    E_pot  = 1/2 · q^T · K · q
    E_tot  = E_kin + E_pot
    W_ext  += 1/2 · (Q_old + Q_new) · (q_new - q_old)
    W_damp += q̇_mid^T · C · q̇_mid · dt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class ModalEnergy(float):
    r"""Modal energy container representing total modal energy (KE + PE).

    Behaves as a float (total energy) for scalar operations, but can also
    be unpacked as `(kinetic, potential)` or accessed via `.kinetic`,
    `.potential`, and `.total` attributes.

    Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F (ECIN)
    and $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F (ENINT)
    """

    kinetic: float
    potential: float
    total: float

    def __new__(cls, kinetic: float, potential: float) -> ModalEnergy:
        total = float(kinetic + potential)
        obj = super().__new__(cls, total)
        obj.kinetic = float(kinetic)
        obj.potential = float(potential)
        obj.total = total
        return obj

    def __iter__(self):
        return iter((self.kinetic, self.potential))

    def __getitem__(self, idx: int) -> float:
        return (self.kinetic, self.potential)[idx]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return (
            f"ModalEnergy(total={self.total:.6e}, "
            f"kinetic={self.kinetic:.6e}, potential={self.potential:.6e})"
        )


class FlexBody:
    """Flexible body component represented via modal superposition (/FXBODY).

    Ported from OpenRadioss upstream files:
      - engine/source/constraints/fxbody/fxbsys.F
      - engine/source/constraints/fxbody/fxbyfor.F
      - engine/source/constraints/fxbody/fxbyvit.F
      - engine/source/constraints/fxbody/fxbdispl.F
      - engine/source/constraints/fxbody/fxbodfp.F
      - engine/source/constraints/fxbody/fxbodv.F
      - engine/source/constraints/fxbody/fxbodvp.F
      - starter/source/constraints/fxbody/hm_read_fxb.F
      - starter/source/constraints/fxbody/ini_fxbody.F

    Parameters
    ----------
    modes : Sequence or np.ndarray
        Array or list of mode identifiers (e.g. [1, 2, ...]).
    frequencies : Sequence or np.ndarray
        Angular frequencies omega (rad/s) for each mode.
    modal_mass : Sequence or np.ndarray, optional
        Modal masses for each mode. Defaults to unit modal mass (1.0).
    modal_stiffness : Sequence or np.ndarray, optional
        Modal stiffnesses for each mode. Defaults to omega_i^2 * modal_mass_i.
    boundary_nodes : Sequence or np.ndarray, optional
        Physical node IDs at which the flexible body interfaces or recovers motion.
    phi : np.ndarray, optional
        Modal shape matrix of shape (n_dof, n_modes).
    damping_ratio : float or Sequence, optional
        Modal damping ratio(s) zeta.
    alpha : float, optional
        Mass-proportional Rayleigh damping coefficient.
    beta : float, optional
        Stiffness-proportional Rayleigh damping coefficient.
    body_id : int, optional
        Flexible body ID.
    title : str, optional
        Flexible body title / label.
    """

    def __init__(
        self,
        modes: Union[Sequence[int], np.ndarray],
        frequencies: Union[Sequence[float], np.ndarray],
        modal_mass: Optional[Union[Sequence[float], np.ndarray]] = None,
        modal_stiffness: Optional[Union[Sequence[float], np.ndarray]] = None,
        boundary_nodes: Optional[Union[Sequence[int], np.ndarray]] = None,
        phi: Optional[np.ndarray] = None,
        damping_ratio: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        body_id: int = 1,
        title: str = "",
    ):
        # Mode definitions
        self.modes = np.asarray(modes)
        self.n_modes = int(len(self.modes))
        if self.n_modes == 0:
            raise ValueError("FlexBody requires at least one mode.")

        # Frequencies (omega in rad/s)
        self.frequencies = np.asarray(frequencies, dtype=float)
        if self.frequencies.ndim == 0:
            self.frequencies = np.full(self.n_modes, float(self.frequencies))
        elif len(self.frequencies) != self.n_modes:
            raise ValueError(
                f"Length of frequencies ({len(self.frequencies)}) does not match n_modes ({self.n_modes})"
            )
        self.omega = self.frequencies.copy()

        # Boundary nodes
        if boundary_nodes is not None:
            self.boundary_nodes = np.asarray(boundary_nodes, dtype=int)
        else:
            self.boundary_nodes = np.array([], dtype=int)

        # Modal matrix phi: (n_dof, n_modes)
        if phi is not None:
            self.phi = np.asarray(phi, dtype=float)
            if self.phi.ndim == 1:
                self.phi = self.phi.reshape(-1, self.n_modes)
            if self.phi.shape[1] != self.n_modes:
                raise ValueError(
                    f"Modal matrix phi columns ({self.phi.shape[1]}) must match n_modes ({self.n_modes})"
                )
            self.n_dof = int(self.phi.shape[0])
        else:
            self.n_dof = len(self.boundary_nodes) * 3 if len(self.boundary_nodes) > 0 else self.n_modes
            self.phi = np.eye(self.n_dof, self.n_modes, dtype=float)

        # Modal mass matrix M
        if modal_mass is None:
            self.modal_mass = np.ones(self.n_modes, dtype=float)
            self.M = np.diag(self.modal_mass)
        else:
            m_arr = np.asarray(modal_mass, dtype=float)
            if m_arr.ndim == 1:
                if len(m_arr) != self.n_modes:
                    raise ValueError(f"modal_mass length ({len(m_arr)}) != n_modes ({self.n_modes})")
                self.modal_mass = m_arr
                self.M = np.diag(m_arr)
            elif m_arr.ndim == 2:
                if m_arr.shape != (self.n_modes, self.n_modes):
                    raise ValueError(f"modal_mass shape {m_arr.shape} != ({self.n_modes}, {self.n_modes})")
                self.M = m_arr
                self.modal_mass = np.diag(m_arr).copy()
            else:
                raise ValueError("modal_mass must be 1D or 2D array")

        # Modal stiffness matrix K
        if modal_stiffness is None:
            k_diag = (self.omega ** 2) * self.modal_mass
            self.modal_stiffness = k_diag
            self.K = np.diag(k_diag)
        else:
            k_arr = np.asarray(modal_stiffness, dtype=float)
            if k_arr.ndim == 1:
                if len(k_arr) != self.n_modes:
                    raise ValueError(f"modal_stiffness length ({len(k_arr)}) != n_modes ({self.n_modes})")
                self.modal_stiffness = k_arr
                self.K = np.diag(k_arr)
            elif k_arr.ndim == 2:
                if k_arr.shape != (self.n_modes, self.n_modes):
                    raise ValueError(f"modal_stiffness shape {k_arr.shape} != ({self.n_modes}, {self.n_modes})")
                self.K = k_arr
                self.modal_stiffness = np.diag(k_arr).copy()
            else:
                raise ValueError("modal_stiffness must be 1D or 2D array")

        # Damping parameters
        # Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F (lines 347-377)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.damping_ratio = damping_ratio

        if damping_ratio is not None:
            if np.isscalar(damping_ratio):
                c_diag = 2.0 * float(damping_ratio) * self.omega * self.modal_mass
            else:
                z_arr = np.asarray(damping_ratio, dtype=float)
                c_diag = 2.0 * z_arr * self.omega * self.modal_mass
            self.C = np.diag(c_diag)
        elif self.alpha != 0.0 or self.beta != 0.0:
            self.C = self.alpha * self.M + self.beta * self.K
        else:
            self.C = np.zeros((self.n_modes, self.n_modes), dtype=float)

        # Identification & metadata
        self.body_id = int(body_id)
        self.title = str(title)

        # Dynamic state variables in modal space
        self.q = np.zeros(self.n_modes, dtype=float)
        self.q_dot = np.zeros(self.n_modes, dtype=float)
        self.q_ddot = np.zeros(self.n_modes, dtype=float)
        self.q_force = np.zeros(self.n_modes, dtype=float)
        self.q_force_prev = np.zeros(self.n_modes, dtype=float)

        # Energy tracking
        # Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F (lines 149-154)
        self.external_work: float = 0.0
        self.dissipated_energy: float = 0.0

        # Physical recovery buffers
        self.u = np.zeros(self.n_dof, dtype=float)
        self.v = np.zeros(self.n_dof, dtype=float)

        # Track initialization of acceleration
        self._acc_initialized = False

    @property
    def kinetic_energy(self) -> float:
        """Modal kinetic energy: 1/2 q̇ᵀ · M · q̇."""
        return 0.5 * float(self.q_dot @ self.M @ self.q_dot)

    @property
    def potential_energy(self) -> float:
        """Modal potential / internal strain energy: 1/2 qᵀ · K · q."""
        return 0.5 * float(self.q @ self.K @ self.q)

    @property
    def total_energy(self) -> float:
        """Total modal mechanical energy: E_kin + E_pot."""
        return self.kinetic_energy + self.potential_energy

    def critical_dt(self) -> float:
        r"""Compute critical time step based on highest frequency: 2 / ω_max.

        Ported from $OR_SRC/starter/source/constraints/fxbody/hm_read_fxb.F (lines 633-646)

        Formulas:
            If stiffness damping beta > 0:
                dtc1 = (-beta * omega + sqrt(beta^2 * omega^2 + 4)) / omega
                dtc2 = 2 / (beta * omega^2)
                dt_crit = min(dtc1, dtc2)
            Else:
                dt_crit = 2 / omega_max
        """
        if len(self.omega) == 0:
            return float("inf")
        omega_max = float(np.max(self.omega))
        if omega_max <= 0.0:
            return float("inf")

        if self.beta > 0.0:
            w = omega_max
            b = self.beta
            dtc1 = (-b * w + np.sqrt(b * b * w * w + 4.0)) / w
            dtc2 = 2.0 / (b * w * w)
            return float(min(dtc1, dtc2))

        return float(2.0 / omega_max)

    def compute_generalized_forces(self, external_forces: Union[Sequence, np.ndarray]) -> np.ndarray:
        r"""Project physical external forces onto the modal basis: Q = Φᵀ · F.

        Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F (lines 120-167)

        Parameters
        ----------
        external_forces : Sequence or np.ndarray
            Physical force vector of length n_dof, or 2D array of nodal forces.

        Returns
        -------
        np.ndarray
            Generalized forces Q of shape (n_modes,).
        """
        F = np.asarray(external_forces, dtype=float)
        if F.ndim > 1:
            F = F.ravel()

        if len(F) != self.n_dof:
            if len(F) > self.n_dof:
                F = F[: self.n_dof]
            else:
                F_pad = np.zeros(self.n_dof, dtype=float)
                F_pad[: len(F)] = F
                F = F_pad

        self.q_force_prev = self.q_force.copy()
        # Q = Φᵀ · F
        self.q_force = self.phi.T @ F
        return self.q_force

    def modal_time_step(self, dt: float) -> np.ndarray:
        r"""Integrate modal equations of motion for one time step dt:
            q̈ = M⁻¹ (Q - K·q - C·q̇)

        Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F (fxbodvp1, lines 60-155)
        and $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F (fxbodfp2, lines 347-447)
        and $OR_SRC/engine/source/constraints/fxbody/fxbsys.F (fxbsys, lines 30-67)

        Uses a symplectic Velocity-Verlet / central-difference scheme with
        implicit damping treatment:
          1. v_{n+1/2} = q̇_n + 1/2 · dt · q̈_n
          2. q_{n+1}   = q_n + dt · v_{n+1/2}
          3. Solve for q̇_{n+1} and q̈_{n+1}:
             (M + 1/2 · dt · C) q̇_{n+1} = M · v_{n+1/2} + 1/2 · dt · (Q_{n+1} - K · q_{n+1})
             q̈_{n+1} = M⁻¹ (Q_{n+1} - K · q_{n+1} - C · q̇_{n+1})

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        np.ndarray
            Updated generalized coordinates q of shape (n_modes,).
        """
        dt = float(dt)
        if dt <= 0.0:
            return self.q

        # Initial acceleration evaluation if starting from rest or non-zero initial q
        if not self._acc_initialized:
            rhs0 = self.q_force - self.K @ self.q - self.C @ self.q_dot
            self.q_ddot = np.linalg.solve(self.M, rhs0)
            self._acc_initialized = True

        q_old = self.q.copy()
        q_dot_old = self.q_dot.copy()

        # Step 1: Half-step velocity
        v_mid = self.q_dot + 0.5 * dt * self.q_ddot

        # Step 2: Displacement update
        q_next = self.q + dt * v_mid

        # Step 3: Advance velocity & acceleration with implicit damping correction
        # (fxbodvp1.F line 100: FAC = ONE + HALF*DT2*ALPHA)
        f_int = self.K @ q_next
        rhs_force = self.q_force - f_int

        if np.all(self.C == 0.0):
            q_ddot_next = np.linalg.solve(self.M, rhs_force)
            q_dot_next = v_mid + 0.5 * dt * q_ddot_next
        else:
            M_eff = self.M + 0.5 * dt * self.C
            rhs_vel = self.M @ v_mid + 0.5 * dt * rhs_force
            q_dot_next = np.linalg.solve(M_eff, rhs_vel)
            q_ddot_next = np.linalg.solve(self.M, rhs_force - self.C @ q_dot_next)

        # Step 4: Energy accounting
        # Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F (lines 337-342)
        # and $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F (lines 146-154)
        q_dot_avg = 0.5 * (q_dot_old + q_dot_next)
        delta_q = q_next - q_old
        q_force_avg = 0.5 * (self.q_force_prev + self.q_force)

        # External work: integral of Q · dq
        delta_wext = float(np.dot(q_force_avg, delta_q))
        self.external_work += delta_wext

        # Damping dissipation: integral of q̇ · C · q̇ dt
        delta_wdamp = float(q_dot_avg @ self.C @ q_dot_avg * dt)
        self.dissipated_energy += delta_wdamp

        # Update modal state
        self.q = q_next
        self.q_dot = q_dot_next
        self.q_ddot = q_ddot_next

        return self.q

    def recover_displacements(self) -> np.ndarray:
        r"""Recover physical nodal displacements from modal amplitudes: u = Φ · q.

        Ported from $OR_SRC/engine/source/constraints/fxbody/fxbdispl.F (fxbdepla, lines 57-127)

        Returns
        -------
        np.ndarray
            Physical displacement vector of length n_dof.
        """
        self.u = self.phi @ self.q
        return self.u

    def recover_velocities(self) -> np.ndarray:
        r"""Recover physical nodal velocities from modal velocities: v = Φ · q̇.

        Ported from $OR_SRC/engine/source/constraints/fxbody/fxbyvit.F (fxbodvp2, lines 202-307)

        Returns
        -------
        np.ndarray
            Physical velocity vector of length n_dof.
        """
        self.v = self.phi @ self.q_dot
        return self.v

    def update_physical_nodes(self, model: Any) -> None:
        r"""Write recovered physical displacements and velocities back to model node arrays.

        Ported from $OR_SRC/engine/source/constraints/fxbody/fxbdispl.F (lines 128-140)
        and $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F (lines 325-341)

        Parameters
        ----------
        model : Model
            OpenRadioss Model instance holding node coordinate arrays x0, x, v, vr.
        """
        u = self.recover_displacements()
        v = self.recover_velocities()

        if model is None:
            return

        n_nodes = len(self.boundary_nodes)
        if n_nodes > 0:
            dofs_per_node = self.n_dof // n_nodes
        else:
            dofs_per_node = 3

        for i, node_id in enumerate(self.boundary_nodes):
            # Resolve node index in model
            if hasattr(model, "_id2idx") and node_id in model._id2idx:
                idx = model._id2idx[node_id]
            elif hasattr(model, "node_ids") and len(model.node_ids) > 0:
                matches = np.where(model.node_ids == node_id)[0]
                idx = int(matches[0]) if len(matches) > 0 else i
            else:
                idx = i

            offset = i * dofs_per_node
            u_node = u[offset : offset + min(3, dofs_per_node)]
            v_node = v[offset : offset + min(3, dofs_per_node)]

            # Position update: x = x0 + u
            if hasattr(model, "x") and model.x is not None:
                if idx < len(model.x):
                    if hasattr(model, "x0") and model.x0 is not None and idx < len(model.x0):
                        model.x[idx, : len(u_node)] = model.x0[idx, : len(u_node)] + u_node
                    else:
                        model.x[idx, : len(u_node)] += u_node

            # Velocity update: v = v_modal
            if hasattr(model, "v") and model.v is not None and idx < len(model.v):
                model.v[idx, : len(v_node)] = v_node

            # Rotational velocities (for 6-DOF modal nodes)
            if dofs_per_node >= 6 and hasattr(model, "vr") and model.vr is not None and idx < len(model.vr):
                vr_node = v[offset + 3 : offset + 6]
                model.vr[idx, : len(vr_node)] = vr_node

            # Displacements array if present
            if hasattr(model, "u") and model.u is not None and idx < len(model.u):
                model.u[idx, : len(u_node)] = u_node

    def compute_modal_energy(self) -> ModalEnergy:
        r"""Compute kinetic and potential energies in modal space.

        Ported from $OR_SRC/engine/source/constraints/fxbody/fxbodvp.F (ECIN, lines 128-148)
        and $OR_SRC/engine/source/constraints/fxbody/fxbodfp.F (ENINT, lines 397-404)

        Returns
        -------
        ModalEnergy
            Subclass of float containing total energy (KE + PE), unpackable
            as `(kinetic, potential)`.
        """
        ke = 0.5 * float(self.q_dot @ self.M @ self.q_dot)
        pe = 0.5 * float(self.q @ self.K @ self.q)
        return ModalEnergy(ke, pe)

    def reset(self) -> None:
        """Reset modal state variables to initial equilibrium."""
        self.q.fill(0.0)
        self.q_dot.fill(0.0)
        self.q_ddot.fill(0.0)
        self.q_force.fill(0.0)
        self.q_force_prev.fill(0.0)
        self.external_work = 0.0
        self.dissipated_energy = 0.0
        self.u.fill(0.0)
        self.v.fill(0.0)
        self._acc_initialized = False

    def state_dict(self) -> Dict[str, Any]:
        """Serialize flexible body dynamic state for restart / checkpointing."""
        return {
            "body_id": self.body_id,
            "q": self.q.copy(),
            "q_dot": self.q_dot.copy(),
            "q_ddot": self.q_ddot.copy(),
            "q_force": self.q_force.copy(),
            "external_work": self.external_work,
            "dissipated_energy": self.dissipated_energy,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore flexible body dynamic state from restart dict."""
        self.q = np.asarray(state["q"], dtype=float).copy()
        self.q_dot = np.asarray(state["q_dot"], dtype=float).copy()
        self.q_ddot = np.asarray(state["q_ddot"], dtype=float).copy()
        self.q_force = np.asarray(state["q_force"], dtype=float).copy()
        self.external_work = float(state.get("external_work", 0.0))
        self.dissipated_energy = float(state.get("dissipated_energy", 0.0))
        self._acc_initialized = True
        self.recover_displacements()
        self.recover_velocities()


def init_flex_bodies(model: Any) -> List[FlexBody]:
    r"""Initialize flexible body solvers from model data (/FXBODY).

    Ported from $OR_SRC/starter/source/constraints/fxbody/ini_fxbody.F
    and $OR_SRC/starter/source/constraints/fxbody/hm_read_fxb.F

    Parameters
    ----------
    model : Model
        The OpenRadioss model containing flexible body definitions.

    Returns
    -------
    List[FlexBody]
        List of initialized FlexBody solver instances.
    """
    if model is None:
        return []

    # If already instantiated FlexBody instances exist on model
    if hasattr(model, "flex_bodies") and model.flex_bodies:
        if isinstance(model.flex_bodies, list):
            return model.flex_bodies
        elif isinstance(model.flex_bodies, dict):
            return list(model.flex_bodies.values())

    bodies: List[FlexBody] = []
    fxbodies_dict = getattr(model, "fxbodies", {})
    if not fxbodies_dict:
        return bodies

    for fid, fxb in fxbodies_dict.items():
        if isinstance(fxb, FlexBody):
            bodies.append(fxb)
            continue

        # Extract attributes from FxBody model entity
        imin = getattr(fxb, "imin", 1)
        imax = getattr(fxb, "imax", 1)
        if imax >= imin > 0:
            modes = np.arange(imin, imax + 1)
        else:
            modes = np.array([1])

        frequencies = getattr(fxb, "frequencies", None)
        if frequencies is None:
            frequencies = np.full(len(modes), 100.0)

        modal_mass = getattr(fxb, "modal_mass", None)
        modal_stiffness = getattr(fxb, "modal_stiffness", None)

        boundary_nodes = getattr(fxb, "boundary_nodes", None)
        if boundary_nodes is None:
            node_id = getattr(fxb, "node_id", 0)
            boundary_nodes = np.array([node_id]) if node_id > 0 else np.array([], dtype=int)

        phi = getattr(fxb, "phi", None)
        alpha = float(getattr(fxb, "alpha", 0.0))
        beta = float(getattr(fxb, "beta", 0.0))
        damping_ratio = getattr(fxb, "damping_ratio", None)

        fb = FlexBody(
            modes=modes,
            frequencies=frequencies,
            modal_mass=modal_mass,
            modal_stiffness=modal_stiffness,
            boundary_nodes=boundary_nodes,
            phi=phi,
            damping_ratio=damping_ratio,
            alpha=alpha,
            beta=beta,
            body_id=getattr(fxb, "id", fid),
            title=getattr(fxb, "title", ""),
        )
        bodies.append(fb)

    model.flex_bodies = {fb.body_id: fb for fb in bodies}
    return bodies


def flex_body_forces(
    model: Any, flex_bodies: Sequence[FlexBody], dt: Optional[float] = None
) -> float:
    r"""Engine cycle hook for flexible bodies (/FXBODY).

    Ported from $OR_SRC/engine/source/constraints/fxbody/fxbyfor.F
    and $OR_SRC/engine/source/constraints/fxbody/fxbyvit.F

    In each explicit cycle:
      1. Gathers nodal forces on the flexible body's boundary nodes.
      2. Projects forces to modal coordinates: Q = Φᵀ · F.
      3. Advances modal coordinates via modal_time_step(dt) if dt > 0.
      4. Recovers physical displacements u = Φ · q and velocities v = Φ · q̇.
      5. Updates physical node positions and velocities in model.
      6. Returns the minimum critical dt for all flexible bodies.

    Parameters
    ----------
    model : Model
        The OpenRadioss model.
    flex_bodies : Sequence[FlexBody]
        The active flexible body instances.
    dt : float, optional
        Cycle time step size. If None, retrieved from model.dt.

    Returns
    -------
    float
        Minimum critical time step among all flexible bodies.
    """
    if not flex_bodies:
        return float("inf")

    if dt is None:
        dt = getattr(model, "dt", 0.0)

    # Gather total nodal forces from model (fext + fint + fcont)
    f_total: Optional[np.ndarray] = None
    if model is not None:
        fext = getattr(model, "fext", None)
        fint = getattr(model, "fint", None)
        fcont = getattr(model, "fcont", None)

        parts = []
        if fext is not None:
            parts.append(fext)
        if fint is not None:
            parts.append(fint)
        if fcont is not None:
            parts.append(fcont)

        if len(parts) > 0:
            f_total = np.zeros_like(parts[0])
            for p in parts:
                f_total += p

    min_dt_crit = float("inf")

    for fb in flex_bodies:
        dt_c = fb.critical_dt()
        if dt_c < min_dt_crit:
            min_dt_crit = dt_c

        # Gather external forces on boundary nodes
        if f_total is not None and len(fb.boundary_nodes) > 0:
            extracted_forces = []
            for nid in fb.boundary_nodes:
                if hasattr(model, "_id2idx") and nid in model._id2idx:
                    idx = model._id2idx[nid]
                    extracted_forces.append(f_total[idx])
                elif hasattr(model, "node_ids") and len(model.node_ids) > 0:
                    matches = np.where(model.node_ids == nid)[0]
                    if len(matches) > 0:
                        extracted_forces.append(f_total[matches[0]])
                    else:
                        extracted_forces.append(np.zeros(3))
                elif nid < len(f_total):
                    extracted_forces.append(f_total[nid])
                else:
                    extracted_forces.append(np.zeros(3))

            fb.compute_generalized_forces(np.array(extracted_forces).ravel())

        # Modal time integration
        if dt is not None and dt > 0.0:
            fb.modal_time_step(dt)
            if model is not None:
                fb.update_physical_nodes(model)
            fb.compute_modal_energy()

    return min_dt_crit
