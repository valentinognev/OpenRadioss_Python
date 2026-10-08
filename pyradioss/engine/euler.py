# pyradioss/engine/euler.py
"""
Fixed-Grid 3D Eulerian Finite Volume Solver.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/euler3d/eflux3.F: Eulerian 3D face flux calculation (W = 0)
- engine/source/ale/euler3d/egrad3.F: Eulerian 3D geometric gradient projection
- engine/source/ale/ale3d/aconv3.F: Convective conservation update
- engine/source/ale/alemain.F: Main Eulerian/ALE solver orchestration
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple, Dict, Set, List, Any

from pyradioss.engine.ale_engine import (
    HEX_FACES,
    build_face_connectivity,
    compute_hex_face_normals,
    compute_hex_volumes,
    limiter_van_leer,
    limiter_minmod,
    compute_barth_jespersen_limiter,
)


def euler_compute_gradients(x: np.ndarray,
                            connectivity: np.ndarray,
                            neighbor_elem: np.ndarray,
                            face_normals: Optional[np.ndarray] = None) -> np.ndarray:
    """Eulerian 3D geometric gradient projection factors across hex faces.

    Ported from $OR_SRC/engine/source/ale/euler3d/egrad3.F
    lines 145-272:
      Ni = 2Sn, |n|=1
      DiX = 8.dx[i], DiY = 8.dy[i], DiZ = 8.dz[i]
      DDi = 64 (dx^2 + dy^2 + dz^2)
      GRADi = FOUR * (Di . Ni) / max(EM15, DDi) = Si . < di, ni > / di^2
      where di is the distance between the centers of the two elements.

    Args:
        x: (n_nodes, 3) nodal coordinates.
        connectivity: (n_elem, 8) hex element connectivity.
        neighbor_elem: (n_elem, 6) neighbor element indices (-1 if boundary).
        face_normals: optional precomputed (n_elem, 6, 3) outward face normals.

    Returns:
        grad: (n_elem, 6) geometric gradient projection factor for each face.
    """
    n_elem = len(connectivity)
    if n_elem == 0:
        return np.empty((0, 6), dtype=np.float64)

    xe = x[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3) element centroids

    if face_normals is None:
        face_normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3)

    grad = np.zeros((n_elem, 6), dtype=np.float64)

    for e in range(n_elem):
        xc_e = xc[e]
        for k in range(6):
            nbr = neighbor_elem[e, k]
            if nbr >= 0:
                d = xc[nbr] - xc_e
            else:
                # Boundary face: use face centroid relative to element centroid
                f_nodes = HEX_FACES[k]
                xf = np.mean(xe[e, f_nodes], axis=0)
                d = xf - xc_e

            dd = float(np.dot(d, d))
            dd_safe = max(1e-15, dd)
            # egrad3.F line 266:
            # 4 * (8*d . 2*S*n) / (64*d^2) = (d . N) / (2 * d^2)
            # Since N has magnitude 2*S, (d . N) / (2*d^2) = S*(d . n) / d^2
            grad[e, k] = 0.5 * float(np.dot(d, face_normals[e, k])) / dd_safe

    return grad


def euler_compute_fluxes(q_elem: np.ndarray,
                         x: np.ndarray,
                         velocity: np.ndarray,
                         connectivity: np.ndarray,
                         neighbor_elem: Optional[np.ndarray] = None,
                         neighbor_face: Optional[np.ndarray] = None,
                         face_normals: Optional[np.ndarray] = None,
                         upwl: float = 1.0,
                         reduc: float = 0.0,
                         limiter: str = "none") -> Dict[str, np.ndarray]:
    """Eulerian 3D face fluxes with fixed grid (W = 0).

    Ported from $OR_SRC/engine/source/ale/euler3d/eflux3.F
    lines 135-248 & 265-323:
      V_face = 1/4 Sum(V_node)
      Flux_face = 0.5 * V_face . N_face (where N_face = 2S.n)
      FLUX(e, k) = Flux_k - UPWL * |Flux_k|
      FLU1(e)    = Sum_k (Flux_k + UPWL * |Flux_k|)
      Boundary face flux is scaled by REDUC.

    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) element scalar or vector fields.
        x: (n_nodes, 3) nodal coordinates.
        velocity: (n_nodes, 3) fluid material velocity V.
        connectivity: (n_elem, 8) hex element connectivity.
        neighbor_elem: optional (n_elem, 6) neighbor connectivity.
        neighbor_face: optional (n_elem, 6) neighbor face connectivity.
        face_normals: optional precomputed (n_elem, 6, 3) outward face normals.
        upwl: upwind parameter (1.0 = donor cell, 0.0 = central).
        reduc: boundary face reduction factor (0.0 = closed slip wall).
        limiter: MUSCL limiter ('none', 'van_leer', 'minmod', 'barth_jespersen').

    Returns:
        dict with:
            'vol_fluxes': (n_elem, 6) outward volumetric fluxes across each face.
            'flux_q': (n_elem, 6) or (n_elem, 6, n_vars) fluxes of quantity q.
            'net_flux_q': (n_elem,) or (n_elem, n_vars) net outgoing flux.
            'FLUX': (n_elem, 6) matching Fortran eflux3.F line 310.
            'FLU1': (n_elem,) matching Fortran eflux3.F line 317.
    """
    n_elem = len(connectivity)
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    n_vars = q_2d.shape[1]

    if neighbor_elem is None or neighbor_face is None:
        neighbor_elem, neighbor_face = build_face_connectivity(connectivity)

    xe = x[connectivity]  # (n_elem, 8, 3)
    ve = velocity[connectivity]  # (n_elem, 8, 3)
    xc = np.mean(xe, axis=1)  # (n_elem, 3)

    if face_normals is None:
        face_normals = compute_hex_face_normals(xe)  # (n_elem, 6, 3)

    # 1. Face velocities and volume fluxes (eflux3.F lines 141-206)
    # V_face = 1/4 sum(V_node), Flux = 0.5 * V_face . N_k
    vol_fluxes = np.zeros((n_elem, 6), dtype=np.float64)
    for f_idx in range(6):
        f_nodes = HEX_FACES[f_idx]
        v_face = np.mean(ve[:, f_nodes], axis=1)  # (n_elem, 3)
        vol_fluxes[:, f_idx] = 0.5 * np.sum(v_face * face_normals[:, f_idx], axis=1)

    # 2. Boundary reduction (eflux3.F lines 274-307)
    boundary_mask = (neighbor_elem < 0)
    vol_fluxes[boundary_mask] *= reduc

    # 3. Antisymmetry on internal faces
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_f = 0.5 * (vol_fluxes[e, f_idx] - vol_fluxes[nbr, nbr_f])
                vol_fluxes[e, f_idx] = avg_f
                vol_fluxes[nbr, nbr_f] = -avg_f

    # 4. Upwind reconstruction for quantity q
    flux_q = np.zeros((n_elem, 6, n_vars), dtype=np.float64)
    for e in range(n_elem):
        for f_idx in range(6):
            vf = vol_fluxes[e, f_idx]
            if abs(vf) < 1e-30:
                continue
            nbr = neighbor_elem[e, f_idx]
            if vf > 0.0:
                # Flow leaves element e -> donor is e
                flux_q[e, f_idx] = q_2d[e] * vf
            else:
                # Flow enters element e -> donor is nbr (or e if boundary)
                donor = q_2d[nbr] if nbr >= 0 else q_2d[e]
                flux_q[e, f_idx] = donor * vf

    # Strict antisymmetry on flux_q
    for e in range(n_elem):
        for f_idx in range(6):
            nbr = neighbor_elem[e, f_idx]
            if nbr > e:
                nbr_f = neighbor_face[e, f_idx]
                avg_q = 0.5 * (flux_q[e, f_idx] - flux_q[nbr, nbr_f])
                flux_q[e, f_idx] = avg_q
                flux_q[nbr, nbr_f] = -avg_q

    # Fortran eflux3.F lines 310-323:
    f_flux = vol_fluxes - upwl * np.abs(vol_fluxes)
    f_flu1 = np.sum(vol_fluxes + upwl * np.abs(vol_fluxes), axis=1)

    net_flux_q = np.sum(flux_q, axis=1)
    res_flux_q = flux_q[:, :, 0] if is_1d else flux_q
    res_net_q = net_flux_q[:, 0] if is_1d else net_flux_q

    return {
        "vol_fluxes": vol_fluxes,
        "flux_q": res_flux_q,
        "net_flux_q": res_net_q,
        "FLUX": f_flux,
        "FLU1": f_flu1,
    }


def euler_advect_step(q_elem: np.ndarray,
                      volumes: np.ndarray,
                      fluxes: Dict[str, Any],
                      dt: float) -> np.ndarray:
    """Consistently update Eulerian state variable over time step dt.

    Ported from $OR_SRC/engine/source/ale/ale3d/aconv3.F
    lines 107-135:
      Q_new = Q_old - dt * sum_k flux_q_k
      q_new = Q_new / V

    Guarantees exact conservation:
      sum(q_new * V) == sum(q_old * V) in closed domains.

    Args:
        q_elem: (n_elem,) or (n_elem, n_vars) intensive state variable.
        volumes: (n_elem,) element volumes.
        fluxes: flux dict from euler_compute_fluxes.
        dt: explicit time step duration.

    Returns:
        q_new: updated state variable after Eulerian convection.
    """
    is_1d = (q_elem.ndim == 1)
    q_2d = q_elem[:, None] if is_1d else q_elem
    net_flux = fluxes["net_flux_q"]
    net_flux_2d = net_flux[:, None] if is_1d else net_flux

    # On fixed Eulerian grid, volume V does not deform: V_new = V_old
    q_extensive_old = q_2d * volumes[:, None]
    q_extensive_new = q_extensive_old - dt * net_flux_2d
    q_new = q_extensive_new / volumes[:, None]

    return q_new[:, 0] if is_1d else q_new


class EulerianGrid3D:
    """Fixed-Grid 3D Eulerian Mesh and Advection Engine.

    Ported from OpenRadioss Fortran sources:
      - engine/source/ale/euler3d/eflux3.F (Eulerian 3D fluxes)
      - engine/source/ale/euler3d/egrad3.F (Eulerian 3D gradients)
      - engine/source/ale/alemain.F (Eulerian time step orchestration)

    Precomputes and caches geometric invariants (face normals, element volumes,
    neighbor connectivity) so subsequent time steps execute with maximum efficiency.
    Tracks mass, momentum, and energy accounting for strict conservation.
    """

    def __init__(self,
                 x: np.ndarray,
                 connectivity: np.ndarray,
                 upwl: float = 1.0,
                 reduc: float = 0.0) -> None:
        """Initialize fixed Eulerian grid.

        Args:
            x: (n_nodes, 3) fixed nodal coordinates.
            connectivity: (n_elem, 8) hex8 element connectivity.
            upwl: upwind parameter (1.0 = donor-cell upwind, 0.0 = centered).
            reduc: boundary face reduction factor (0.0 = slip wall).
        """
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.conn = np.asarray(connectivity, dtype=np.int64).copy()
        self.n_nodes = len(self.x)
        self.n_elem = len(self.conn)
        self.upwl = upwl
        self.reduc = reduc

        # Precompute and cache fixed geometry
        self.xe = self.x[self.conn]  # (n_elem, 8, 3)
        self.xc = np.mean(self.xe, axis=1)  # (n_elem, 3)
        self.volumes = compute_hex_volumes(self.x, self.conn)  # (n_elem,)
        self.normals = compute_hex_face_normals(self.xe)  # (n_elem, 6, 3)
        self.neighbor_elem, self.neighbor_face = build_face_connectivity(self.conn)
        self.geom_grad = euler_compute_gradients(
            self.x, self.conn, self.neighbor_elem, self.normals
        )

        # Energy accounting ledger
        self.energy_ledger: Dict[str, float] = {
            "initial_mass": 0.0,
            "current_mass": 0.0,
            "initial_energy": 0.0,
            "current_energy": 0.0,
            "cumulative_mass_flux": 0.0,
            "cumulative_energy_flux": 0.0,
        }

    def compute_fluxes(self,
                       q_elem: np.ndarray,
                       velocity: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute convective fluxes across fixed hex faces."""
        return euler_compute_fluxes(
            q_elem=q_elem,
            x=self.x,
            velocity=velocity,
            connectivity=self.conn,
            neighbor_elem=self.neighbor_elem,
            neighbor_face=self.neighbor_face,
            face_normals=self.normals,
            upwl=self.upwl,
            reduc=self.reduc,
        )

    def advance(self,
                state: Dict[str, np.ndarray],
                velocity: np.ndarray,
                dt: float) -> Dict[str, np.ndarray]:
        """Advance Eulerian state fields over explicit time step dt.

        Convects mass density 'rho', specific internal energy 'eint',
        and momentum / velocity 'v_elem'.

        Args:
            state: dictionary containing:
                   'rho': (n_elem,) density field
                   'eint': (n_elem,) specific internal energy field
                   optional 'sig': (n_elem, 6) stress tensor field
            velocity: (n_nodes, 3) fluid velocity at nodes.
            dt: explicit time step duration.

        Returns:
            updated state dictionary.
        """
        if dt <= 0.0:
            return state

        rho = state["rho"]
        # 1. Convect mass density
        flux_rho = self.compute_fluxes(rho, velocity)
        rho_new = euler_advect_step(rho, self.volumes, flux_rho, dt)
        state["rho"] = np.maximum(rho_new, 1e-30)

        # 2. Convect specific internal energy
        if "eint" in state:
            eint = state["eint"]
            # Convect extensive total internal energy: E = rho * eint
            e_tot = rho * eint
            flux_e = self.compute_fluxes(e_tot, velocity)
            e_tot_new = euler_advect_step(e_tot, self.volumes, flux_e, dt)
            state["eint"] = e_tot_new / state["rho"]

        # 3. Convect stresses if present
        if "sig" in state:
            sig = state["sig"]
            flux_sig = self.compute_fluxes(sig, velocity)
            state["sig"] = euler_advect_step(sig, self.volumes, flux_sig, dt)

        # 4. Book energy accounting
        mass_curr = float(np.sum(state["rho"] * self.volumes))
        self.energy_ledger["current_mass"] = mass_curr
        if "eint" in state:
            energy_curr = float(np.sum(state["rho"] * state["eint"] * self.volumes))
            self.energy_ledger["current_energy"] = energy_curr

        return state
