# pyradioss/engine/ale_fvm.py
"""
Finite Volume Method (FVM) ALE Shock Physics Solver.

Ported from OpenRadioss Fortran sources:
- engine/source/ale/alefvm/alefvm_main.F:
    Main FVM ALE collocated solver orchestration and nodal velocity mapping
- engine/source/ale/alefvm/alefvm_scheme.F:
    Finite volume momentum time update and state evolution
- engine/source/ale/alefvm/alefvm_sfint3.F:
    Acoustic Riemann solver with Dellacherie-Omnes-Raviart low-Mach correction
    and internal force assembly from face pressures
- engine/source/ale/alefvm/alefvm_stress.F:
    Face normal computation, acoustic impedance, Mach numbers, and total stress tensor
- engine/source/ale/alefvm/alefvm_aflux3.F:
    3D ALE face relative velocities, acoustic Riemann interface velocities,
    and convective face flux computation
- engine/source/ale/alefvm/alefvm_eflux3.F:
    Fixed-grid Eulerian face fluxes (W = 0)
- engine/source/ale/alefvm/alefvm_expand_mom2.F:
    Expansion of cell momentum to 8 element nodes
- engine/source/ale/alefvm/alefvm_accele.F:
    Nodal acceleration reset for collocated ALE FVM
- common_source/modules/ale/alefvm_mod.F:
    Global ALE FVM buffer and parameter declarations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, List, Any
import numpy as np

from pyradioss.engine.ale_engine import (
    HEX_FACES,
    build_face_connectivity,
    compute_hex_face_normals,
    compute_hex_volumes,
)


# =============================================================================
# Solver Constants
# Ported from engine/source/ale/alefvm/alefvm_aflux3.F lines 107-115
# and engine/source/ale/alefvm/alefvm_sfint3.F lines 206-275
# =============================================================================

SOLVER_FEM = 1               # Standard FEM (bypasses FVM)
SOLVER_U_AVG = 2            # Centered FVM: velocity arithmetic average
SOLVER_RHOU_AVG = 3         # Centered FVM: momentum-weighted velocity average
SOLVER_ROE_AVG = 4          # Centered FVM: Roe density-weighted average
SOLVER_GODUNOV_ACOUSTIC = 5 # Godunov acoustic Riemann solver with low-Mach correction
SOLVER_INTERPOLATED = 6     # Centered geometric interpolation


@dataclass
class ALEFVMParams:
    """Parameters governing the ALE FVM collocated scheme.

    # Ported from common_source/modules/ale/alefvm_mod.F lines 100-117
    # and starter/source/materials/ale/read_euler_mat.F
    """
    enabled: bool = True
    solver_type: int = SOLVER_GODUNOV_ACOUSTIC
    upwind_factor: float = 1.0    # UPWL parameter (1.0 = full donor cell upwind, 0.0 = centered)
    reduc_boundary: float = 0.0   # Boundary flux reduction (0.0 = slip wall, 1.0 = inflow/outflow)
    low_mach_fix: bool = True     # Dellacherie-Omnes-Raviart low-Mach correction
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class ALEFVMState:
    """State storage for ALE FVM cell and face variables.

    # Ported from common_source/modules/ale/alefvm_mod.F lines 93-98 (ALEFVM_BUFFER_)
    """
    n_elem: int
    n_nodes: int

    # Cell-centered conserved and derived variables
    vol: np.ndarray        # (n_elem,) element volume
    rho: np.ndarray        # (n_elem,) density
    mom: np.ndarray        # (n_elem, 3) total cell momentum (rho * vol * u)
    ssp: np.ndarray        # (n_elem,) speed of sound c
    pressure: np.ndarray   # (n_elem,) thermodynamic pressure P
    sig: np.ndarray        # (n_elem, 6) Cauchy stress tensor [s11, s22, s33, s12, s23, s31]

    # Forces
    fint_cell: np.ndarray  # (n_elem, 3) cell internal force from face pressures
    fext_cell: np.ndarray  # (n_elem, 3) external force / gravity
    fcell: np.ndarray      # (n_elem, 3) total net force on cell

    # Face-centered intermediate buffers (6 faces per hex8)
    # Fortran origin: F_FACE(3, 6, n_elem) in alefvm_stress.F and alefvm_sfint3.F
    face_normals: np.ndarray   # (n_elem, 6, 3) outward face area-normal vectors (magnitude = 2 * Area)
    face_areas: np.ndarray     # (n_elem, 6) face areas S = 0.5 * ||N||
    face_u_n: np.ndarray       # (n_elem, 6) outward normal velocities
    face_pressures: np.ndarray # (n_elem, 6) numerical flux pressures Pf
    face_fluxes: np.ndarray    # (n_elem, 6) volumetric convective face fluxes

    # Nodal expansion buffers
    # Fortran origin: VERTEX(1:4, n_nodes) in alefvm_mod.F
    vertex_mom: np.ndarray     # (n_nodes, 3) accumulated nodal momentum
    vertex_count: np.ndarray   # (n_nodes,) element contribution count

    # Energy ledger for strict conservation tracking
    energy_kinetic: float = 0.0
    energy_internal_work: float = 0.0

    @classmethod
    def initialize(
        cls,
        n_elem: int,
        n_nodes: int,
        vol: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        ssp: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
    ) -> "ALEFVMState":
        """Factory method to initialize state with specified dimensions and initial fields."""
        v = np.ones(n_elem, dtype=np.float64) if vol is None else np.array(vol, dtype=np.float64)
        r = np.ones(n_elem, dtype=np.float64) if rho is None else np.array(rho, dtype=np.float64)
        c = np.ones(n_elem, dtype=np.float64) * 340.0 if ssp is None else np.array(ssp, dtype=np.float64)
        p = np.zeros(n_elem, dtype=np.float64) if pressure is None else np.array(pressure, dtype=np.float64)

        if velocity is None:
            u = np.zeros((n_elem, 3), dtype=np.float64)
        else:
            u = np.array(velocity, dtype=np.float64)

        mass = r * v
        mom = u * mass[:, np.newaxis]

        sig = np.zeros((n_elem, 6), dtype=np.float64)
        # Pressure is isotropic part: sigma_ii = -p
        sig[:, 0] = -p
        sig[:, 1] = -p
        sig[:, 2] = -p

        return cls(
            n_elem=n_elem,
            n_nodes=n_nodes,
            vol=v,
            rho=r,
            mom=mom,
            ssp=c,
            pressure=p,
            sig=sig,
            fint_cell=np.zeros((n_elem, 3), dtype=np.float64),
            fext_cell=np.zeros((n_elem, 3), dtype=np.float64),
            fcell=np.zeros((n_elem, 3), dtype=np.float64),
            face_normals=np.zeros((n_elem, 6, 3), dtype=np.float64),
            face_areas=np.zeros((n_elem, 6), dtype=np.float64),
            face_u_n=np.zeros((n_elem, 6), dtype=np.float64),
            face_pressures=np.zeros((n_elem, 6), dtype=np.float64),
            face_fluxes=np.zeros((n_elem, 6), dtype=np.float64),
            vertex_mom=np.zeros((n_nodes, 3), dtype=np.float64),
            vertex_count=np.zeros(n_nodes, dtype=np.float64),
        )


# =============================================================================
# Face Geometry and Normal Vectors
# Ported from engine/source/ale/alefvm/alefvm_stress.F lines 184-209
# =============================================================================

def compute_alefvm_face_normals(xe: np.ndarray) -> np.ndarray:
    """Compute outward face area-normal vectors for hex elements.

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_stress.F
    # lines 184-209:
    # Face 1 (-z): cross((X3-X1), (X2-X4))
    # Face 2 (+y): cross((X7-X4), (X3-X8))
    # Face 3 (+z): cross((X6-X8), (X7-X5))
    # Face 4 (-y): cross((X2-X5), (X6-X1))
    # Face 5 (+x): cross((X7-X2), (X6-X3))
    # Face 6 (-x): cross((X8-X1), (X4-X5))
    #
    # Vector magnitude ||N_k|| = 2 * Area_k.

    Args:
        xe: (n_elem, 8, 3) nodal coordinates.

    Returns:
        normals: (n_elem, 6, 3) outward normal vectors.
    """
    n_elem = xe.shape[0]
    normals = np.zeros((n_elem, 6, 3), dtype=np.float64)
    if n_elem == 0:
        return normals

    p = xe  # 0-indexed: node 0..7 correspond to Fortran 1..8

    # Face 0 (bottom, -z, nodes 1,2,3,4 -> indices 0,1,2,3)
    d1 = p[:, 2] - p[:, 0]  # X3 - X1
    d2 = p[:, 1] - p[:, 3]  # X2 - X4
    normals[:, 0] = np.cross(d1, d2)

    # Face 1 (back, +y, nodes 3,4,8,7 -> indices 2,3,7,6)
    # Fortran: cross( (X7 - X4), (X3 - X8) ) -> indices (6 - 3), (2 - 7)
    d1 = p[:, 6] - p[:, 3]
    d2 = p[:, 2] - p[:, 7]
    normals[:, 1] = np.cross(d1, d2)

    # Face 2 (top, +z, nodes 5,6,7,8 -> indices 4,5,6,7)
    # Fortran: cross( (X6 - X8), (X7 - X5) ) -> indices (5 - 7), (6 - 4)
    d1 = p[:, 5] - p[:, 7]
    d2 = p[:, 6] - p[:, 4]
    normals[:, 2] = np.cross(d1, d2)

    # Face 3 (front, -y, nodes 1,2,6,5 -> indices 0,1,5,4)
    # Fortran: cross( (X2 - X5), (X6 - X1) ) -> indices (1 - 4), (5 - 0)
    d1 = p[:, 1] - p[:, 4]
    d2 = p[:, 5] - p[:, 0]
    normals[:, 3] = np.cross(d1, d2)

    # Face 4 (right, +x, nodes 2,3,7,6 -> indices 1,2,6,5)
    # Fortran: cross( (X7 - X2), (X6 - X3) ) -> indices (6 - 1), (5 - 2)
    d1 = p[:, 6] - p[:, 1]
    d2 = p[:, 5] - p[:, 2]
    normals[:, 4] = np.cross(d1, d2)

    # Face 5 (left, -x, nodes 1,4,8,5 -> indices 0,3,7,4)
    # Fortran: cross( (X8 - X1), (X4 - X5) ) -> indices (7 - 0), (3 - 4)
    d1 = p[:, 7] - p[:, 0]
    d2 = p[:, 3] - p[:, 4]
    normals[:, 5] = np.cross(d1, d2)

    return normals


def alefvm_prepare_stress_buffer(
    state: ALEFVMState,
    x: np.ndarray,
    connectivity: np.ndarray,
    svis: Optional[np.ndarray] = None,
    qvis: Optional[np.ndarray] = None,
) -> None:
    """Prepare face buffers, acoustic impedances, Mach numbers, and normal velocities.

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_stress.F
    # lines 120-250:
    # 1. Total Cauchy stress: S = SIG + SVIS - QVIS * I
    # 2. Centroid pressure: P = -1/3 * (S11 + S22 + S33)
    # 3. Acoustic impedance: Z = rho * c
    # 4. Mach number: M = ||u|| / c
    # 5. Outward normal velocity on face k: U_N = (u . N_k) / ||N_k||
    """
    n_elem = state.n_elem
    if n_elem == 0:
        return

    xe = x[connectivity]  # (n_elem, 8, 3)
    normals = compute_alefvm_face_normals(xe)  # (n_elem, 6, 3)
    state.face_normals = normals

    # Total stress tensor
    s_tot = state.sig.copy()
    if svis is not None:
        s_tot += svis
    if qvis is not None:
        s_tot[:, 0] -= qvis
        s_tot[:, 1] -= qvis
        s_tot[:, 2] -= qvis

    # Centroid pressure P = -1/3 * tr(sigma_total)
    state.pressure = - (s_tot[:, 0] + s_tot[:, 1] + s_tot[:, 2]) / 3.0

    # Cell velocity u = MOM / (rho * vol)
    mass = np.maximum(state.rho * state.vol, 1e-30)
    u_cell = state.mom / mass[:, np.newaxis]  # (n_elem, 3)

    # Face areas: S = 0.5 * ||N||
    norms = np.linalg.norm(normals, axis=2)  # (n_elem, 6)
    norms_safe = np.maximum(norms, 1e-15)
    state.face_areas = 0.5 * norms

    # Face normal velocities: U_N = (u . N) / ||N||
    # Dot product of u_cell with each face normal
    u_dot_n = np.einsum("ei,eki->ek", u_cell, normals)
    state.face_u_n = u_dot_n / norms_safe


# =============================================================================
# Face Pressure and Internal Force Assembly
# Ported from engine/source/ale/alefvm/alefvm_sfint3.F lines 206-301
# =============================================================================

def alefvm_compute_internal_forces(
    state: ALEFVMState,
    neighbor_elem: np.ndarray,
    neighbor_face: np.ndarray,
    solver_type: int = SOLVER_GODUNOV_ACOUSTIC,
    low_mach_fix: bool = True,
) -> np.ndarray:
    """Compute numerical interface pressures and assemble internal cell forces.

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_sfint3.F
    # lines 206-301:
    #
    # Godunov Acoustic Riemann Problem (ISOLVER = 5):
    #   Z1 = rho1 * c1, Z2 = rho2 * c2
    #   DENOM = Z1 + Z2
    #   Mf = min(M1, M2)
    #   theta = min(1.0, Mf)   (Dellacherie low-Mach number fix)
    #   Pf = (Z1 * P2 + Z2 * P1) / DENOM + theta * (Z1 * Z2 * (U1N1 - U2N1) / DENOM)
    #
    # Sliding rigid wall BC (no neighbor):
    #   Pf = P1 + theta * 0.5 * Z1 * U1N1
    #
    # Average solver (ISOLVER = 1..4):
    #   Pf = 0.5 * (P1 + P2)  (or P1 if boundary)
    #
    # Face force:
    #   FFACE = -0.5 * Pf * N = -Pf * Area * n_outward
    #
    # Internal cell force:
    #   FINT_CELL = sum(FFACE, faces)

    Args:
        state: ALEFVMState data object.
        neighbor_elem: (n_elem, 6) neighbor element indices (-1 if boundary).
        neighbor_face: (n_elem, 6) face index on neighbor element (-1 if boundary).
        solver_type: numerical scheme (5 for Godunov acoustic).
        low_mach_fix: apply Dellacherie low-Mach velocity dissipation correction.

    Returns:
        fint_cell: (n_elem, 3) internal forces.
    """
    n_elem = state.n_elem
    fint = np.zeros((n_elem, 3), dtype=np.float64)
    if n_elem == 0:
        return fint

    p = state.pressure
    rho = state.rho
    c = state.ssp
    z = rho * c  # Acoustic impedance
    mass = np.maximum(rho * state.vol, 1e-30)
    u_cell = state.mom / mass[:, np.newaxis]
    u_mag = np.linalg.norm(u_cell, axis=1)
    mach = u_mag / np.maximum(c, 1e-12)

    normals = state.face_normals
    face_u_n = state.face_u_n
    face_pf = np.zeros((n_elem, 6), dtype=np.float64)

    for e in range(n_elem):
        p1 = p[e]
        z1 = z[e]
        m1 = mach[e]

        for k in range(6):
            nbr = neighbor_elem[e, k]
            u1n1 = face_u_n[e, k]

            if nbr >= 0:
                p2 = p[nbr]
                z2 = z[nbr]
                m2 = mach[nbr]
                nbr_k = neighbor_face[e, k]
                # Normal velocity on neighbor face pointing outward from neighbor
                # Normal relative to cell e points in opposite direction:
                u2n1 = -face_u_n[nbr, nbr_k]

                if solver_type == SOLVER_GODUNOV_ACOUSTIC:
                    denom = max(z1 + z2, 1e-15)
                    mf = min(m1, m2)
                    theta = min(1.0, mf) if low_mach_fix else 1.0
                    # alefvm_sfint3.F line 230:
                    pf = (z1 * p2 + z2 * p1) / denom + theta * (z1 * z2 * (u1n1 - u2n1) / denom)
                else:
                    # Arithmetic pressure average (alefvm_sfint3.F line 260)
                    pf = 0.5 * (p1 + p2)
            else:
                # Boundary face (sliding wall BC)
                if solver_type == SOLVER_GODUNOV_ACOUSTIC:
                    theta = min(1.0, m1) if low_mach_fix else 1.0
                    # alefvm_sfint3.F line 238:
                    pf = p1 + theta * 0.5 * z1 * u1n1
                else:
                    pf = p1

            face_pf[e, k] = pf

            # Face force: FFACE = -0.5 * Pf * N = -Pf * Area * n
            # alefvm_sfint3.F line 241-243:
            fface = -0.5 * pf * normals[e, k]
            fint[e] += fface

    state.face_pressures = face_pf
    state.fint_cell = fint
    state.fcell = fint + state.fext_cell

    return fint


# =============================================================================
# 3D ALE Face Relative Velocities and Flux Computation
# Ported from engine/source/ale/alefvm/alefvm_aflux3.F lines 302-665, 777-791
# =============================================================================

def alefvm_compute_face_fluxes(
    state: ALEFVMState,
    x: np.ndarray,
    connectivity: np.ndarray,
    neighbor_elem: np.ndarray,
    grid_velocity: Optional[np.ndarray] = None,
    solver_type: int = SOLVER_GODUNOV_ACOUSTIC,
    upwl: float = 1.0,
    reduc: float = 0.0,
    dt1: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute 3D ALE face relative velocities, acoustic interface velocities, and fluxes.

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_aflux3.F
    # lines 302-665 & 732-791:
    #
    # 1. Face grid velocity W_face = 1/4 * sum(W_node) for the 4 nodes of the face
    # 2. Interface fluid velocity u*:
    #    - CASE(1): u* = 0.5 * (u1 + u2) - W_face
    #    - CASE(2): u* = (rho1*u1 + rho2*u2) / (rho1 + rho2) - W_face
    #    - CASE(3): u* = (sqrt(rho1)*u1 + sqrt(rho2)*u2) / (sqrt(rho1) + sqrt(rho2)) - W_face
    #    - CASE(5) Godunov Acoustic:
    #        u* = (Z1*u1 + Z2*u2) / (Z1 + Z2) + ((P1 - P2) / (Z1 + Z2)) * (N / ||N||) - W_face
    # 3. Geometric face volume flux:
    #    FLUX = 0.5 * (u* . N) = u* . (Area * n)
    # 4. Boundary face treatment:
    #    FLUX = FLUX * REDUC if neighbor == -1 (slip wall: REDUC = 0)
    # 5. Upwind blending:
    #    FLUX_upw = FLUX - UPWL * |FLUX|
    #    FLU1 = sum(FLUX + UPWL * |FLUX|)

    Args:
        state: ALEFVMState container.
        x: (n_nodes, 3) nodal coordinates.
        connectivity: (n_elem, 8) hex element connectivity.
        neighbor_elem: (n_elem, 6) element neighbor indices.
        grid_velocity: (n_nodes, 3) ALE mesh velocity W (None for Eulerian W=0).
        solver_type: interface velocity scheme (1..5).
        upwl: donor-cell upwind factor (1.0 = full upwind).
        reduc: boundary flux multiplier (0.0 = slip wall).
        dt1: previous time step (if 0.0, initializes with centered velocity).

    Returns:
        dict with:
          'flux_raw': (n_elem, 6) raw volumetric fluxes
          'flux_upwind': (n_elem, 6) upwind donor fluxes
          'flu1': (n_elem,) sum of positive upwind fluxes
          'face_v_rel': (n_elem, 6, 3) relative face velocities
    """
    n_elem = state.n_elem
    n_nodes = len(x)
    if n_elem == 0:
        return {
            "flux_raw": np.empty((0, 6)),
            "flux_upwind": np.empty((0, 6)),
            "flu1": np.empty(0),
            "face_v_rel": np.empty((0, 6, 3)),
        }

    mass = np.maximum(state.rho * state.vol, 1e-30)
    u_cell = state.mom / mass[:, np.newaxis]  # (n_elem, 3)
    p = state.pressure
    rho = state.rho
    c = state.ssp
    z = rho * c  # (n_elem,) acoustic impedance

    normals = state.face_normals  # (n_elem, 6, 3)
    norms = np.linalg.norm(normals, axis=2)
    norms_safe = np.maximum(norms, 1e-15)
    unit_n = normals / norms_safe[:, :, np.newaxis]

    # Grid velocities on faces W_face
    # Fortran alefvm_aflux3.F lines 176-201
    w_face = np.zeros((n_elem, 6, 3), dtype=np.float64)
    if grid_velocity is not None:
        for f_idx in range(6):
            f_nodes = HEX_FACES[f_idx]
            # (n_elem, 4, 3) -> mean over 4 nodes
            w_face[:, f_idx] = np.mean(grid_velocity[connectivity[:, f_nodes]], axis=1)

    face_v_rel = np.zeros((n_elem, 6, 3), dtype=np.float64)
    flux_raw = np.zeros((n_elem, 6), dtype=np.float64)
    flux_upw = np.zeros((n_elem, 6), dtype=np.float64)
    flu1 = np.zeros(n_elem, dtype=np.float64)

    for e in range(n_elem):
        u1 = u_cell[e]
        rho1 = rho[e]
        p1 = p[e]
        z1 = z[e]

        for k in range(6):
            nbr = neighbor_elem[e, k]
            if nbr >= 0:
                u2 = u_cell[nbr]
                rho2 = rho[nbr]
                p2 = p[nbr]
                z2 = z[nbr]
            else:
                # Solid wall reflection / sliding boundary
                u2 = u1.copy()
                rho2 = rho1
                p2 = p1
                z2 = z1

            # Interface velocity u*
            if solver_type == SOLVER_U_AVG:
                # alefvm_aflux3.F line 310
                u_star = 0.5 * (u1 + u2)
            elif solver_type == SOLVER_RHOU_AVG:
                # alefvm_aflux3.F line 353
                u_star = (rho1 * u1 + rho2 * u2) / max(rho1 + rho2, 1e-15)
            elif solver_type == SOLVER_ROE_AVG:
                # alefvm_aflux3.F line 398
                sr1 = math.sqrt(rho1)
                sr2 = math.sqrt(rho2)
                u_star = (sr1 * u1 + sr2 * u2) / max(sr1 + sr2, 1e-15)
            elif solver_type == SOLVER_GODUNOV_ACOUSTIC:
                # alefvm_aflux3.F lines 491-527
                if dt1 == 0.0 or nbr < 0:
                    u_star = (rho1 * u1 + rho2 * u2) / max(rho1 + rho2, 1e-15)
                else:
                    denom = max(z1 + z2, 1e-15)
                    # Acoustic Riemann interface velocity:
                    # u* = (Z1*u1 + Z2*u2)/(Z1+Z2) + ((P1 - P2)/(Z1+Z2)) * n
                    u_star = (z1 * u1 + z2 * u2) / denom + ((p1 - p2) / denom) * unit_n[e, k]
            else:
                u_star = 0.5 * (u1 + u2)

            # Relative face velocity: V_face = u* - W_face
            v_rel = u_star - w_face[e, k]
            face_v_rel[e, k] = v_rel

            # Volumetric face flux: FLUX = 0.5 * (V_face . N) = V_face . (Area * n)
            # alefvm_aflux3.F line 659
            flx = 0.5 * float(np.dot(v_rel, normals[e, k]))

            # Boundary face treatment: slip wall blocks normal flux
            # alefvm_aflux3.F lines 748-774
            if nbr < 0:
                flx *= reduc

            flux_raw[e, k] = flx

            # Upwind treatment (alefvm_aflux3.F lines 778-790)
            flux_upw[e, k] = flx - upwl * abs(flx)
            flu1[e] += flx + upwl * abs(flx)

    state.face_fluxes = flux_raw

    return {
        "flux_raw": flux_raw,
        "flux_upwind": flux_upw,
        "flu1": flu1,
        "face_v_rel": face_v_rel,
    }


# =============================================================================
# Momentum Time Integration and Nodal Expansion
# Ported from engine/source/ale/alefvm/alefvm_scheme.F lines 102-220
# and engine/source/ale/alefvm/alefvm_expand_mom2.F lines 89-107
# =============================================================================

def alefvm_update_momentum(
    state: ALEFVMState,
    dt: float,
    dt_prev: float = 0.0,
) -> np.ndarray:
    """Update cell momentum using net cell forces (FVM time integration).

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_scheme.F
    # lines 102-121:
    #   if DT_prev == 0:
    #       dMOM = 0.5 * dt * FCELL
    #   else:
    #       dMOM = dt * FCELL
    #   MOM = MOM + dMOM

    Args:
        state: ALEFVMState object.
        dt: current time step dt2.
        dt_prev: previous time step dt1.

    Returns:
        dmom: (n_elem, 3) momentum increment.
    """
    factor = 0.5 * dt if dt_prev == 0.0 else dt
    dmom = factor * state.fcell
    state.mom += dmom

    # Book internal work for strict energy accounting
    mass = np.maximum(state.rho * state.vol, 1e-30)
    u_cell = state.mom / mass[:, np.newaxis]
    state.energy_internal_work += float(np.sum(state.fint_cell * u_cell) * dt)
    state.energy_kinetic = 0.5 * float(np.sum(mass * np.sum(u_cell**2, axis=1)))

    return dmom


def alefvm_expand_momentum_to_nodes(
    state: ALEFVMState,
    connectivity: np.ndarray,
    nodal_mass: np.ndarray,
    is_ale_node: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Expand cell momentum to element nodes and compute nodal velocities.

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_expand_mom2.F
    # lines 89-107:
    #   For each cell e and each of its 8 nodes k:
    #     VERTEX(1:3, node_k) += MOM(e) / 8.0
    #     VERTEX(4, node_k) += 1.0
    #
    # And engine/source/ale/alefvm/alefvm_main.F lines 196-211:
    #   V_node = VERTEX(1:3) / MSNF(node)

    Args:
        state: ALEFVMState object.
        connectivity: (n_elem, 8) hex element connectivity.
        nodal_mass: (n_nodes,) lump mass for each node MSNF.
        is_ale_node: (n_nodes,) boolean / integer mask (1 if ALE node, 0 if fixed/lagrangian).

    Returns:
        nodal_velocity: (n_nodes, 3) updated nodal velocities.
    """
    n_nodes = state.n_nodes
    vertex_mom = np.zeros((n_nodes, 3), dtype=np.float64)
    vertex_count = np.zeros(n_nodes, dtype=np.float64)

    # 1/8th of cell momentum to each vertex
    mom_eighth = state.mom / 8.0

    for k in range(8):
        node_indices = connectivity[:, k]
        np.add.at(vertex_mom, node_indices, mom_eighth)
        np.add.at(vertex_count, node_indices, 1.0)

    state.vertex_mom = vertex_mom
    state.vertex_count = vertex_count

    # Nodal velocity: V = VERTEX / MSNF
    nodal_velocity = np.zeros((n_nodes, 3), dtype=np.float64)
    active_mask = (nodal_mass > 0.0) & (vertex_count > 0.0)
    if is_ale_node is not None:
        active_mask &= (is_ale_node > 0)

    nodal_velocity[active_mask] = vertex_mom[active_mask] / nodal_mass[active_mask, np.newaxis]
    return nodal_velocity


def alefvm_reset_accelerations(
    n_nodes: int,
    is_ale_node: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reset nodal accelerations for ALE FVM nodes.

    # Ported from $OR_SRC/engine/source/ale/alefvm/alefvm_accele.F
    # lines 68-73:
    #   if NALE(n) /= 0:
    #     A(1:3, n) = 0
    """
    accel = np.zeros((n_nodes, 3), dtype=np.float64)
    return accel


# =============================================================================
# ALE FVM Advection / Transport Step
# Ported from engine/source/ale/alefvm/cut_cells/a22conv3.F lines 150-194
# and engine/source/ale/ale3d/aconv3.F lines 81-135
# =============================================================================

def alefvm_advect_scalar(
    phi_elem: np.ndarray,
    flux_upwind: np.ndarray,
    flu1: np.ndarray,
    neighbor_elem: np.ndarray,
    dt: float,
) -> np.ndarray:
    """First-order upwind convective transport of an intensive/extensive cell variable.

    # Ported from $OR_SRC/engine/source/ale/alefvm/cut_cells/a22conv3.F
    # lines 164-192:
    #   dPHI = sum_faces( VALVOIS * UpwFLUX_face ) + VALEL * FLU1
    #   dPHI = -0.5 * dt * dPHI
    #   PHI_new = PHI + dPHI

    Args:
        phi_elem: (n_elem,) or (n_elem, m) cell variable.
        flux_upwind: (n_elem, 6) upwind fluxes (FLUX - UPWL * |FLUX|).
        flu1: (n_elem,) sum of positive incoming fluxes.
        neighbor_elem: (n_elem, 6) neighbor indices (-1 for boundary).
        dt: time step.

    Returns:
        dphi: change in variable due to convection.
    """
    n_elem = len(phi_elem)
    is_1d = (phi_elem.ndim == 1)
    if is_1d:
        phi = phi_elem[:, np.newaxis]
    else:
        phi = phi_elem

    n_comp = phi.shape[1]
    dphi = np.zeros_like(phi)

    for e in range(n_elem):
        val_el = phi[e]
        accum = val_el * flu1[e]

        for k in range(6):
            nbr = neighbor_elem[e, k]
            val_vois = phi[nbr] if nbr >= 0 else val_el
            accum += val_vois * flux_upwind[e, k]

        dphi[e] = -0.5 * dt * accum

    if is_1d:
        return dphi.ravel()
    return dphi


# =============================================================================
# High-Level Orchestrator Class: ALEFVMSolver
# Ported from engine/source/ale/alefvm/alefvm_main.F lines 43-227
# =============================================================================

class ALEFVMSolver:
    """Arbitrary Lagrangian-Eulerian Finite Volume Method explicit solver.

    Full collocated Godunov FVM engine adhering strictly to OpenRadioss alefvm:
    - Acoustic Riemann problem with low-Mach damping
    - Outward face area normals via diagonal cross product
    - Donor cell upwind / centered convective fluxes
    - Energy accounting and nodal momentum expansion
    """

    def __init__(self, params: Optional[ALEFVMParams] = None):
        self.params = params or ALEFVMParams()

    def step(
        self,
        state: ALEFVMState,
        x: np.ndarray,
        connectivity: np.ndarray,
        neighbor_elem: np.ndarray,
        neighbor_face: np.ndarray,
        nodal_mass: np.ndarray,
        dt: float,
        dt_prev: float = 0.0,
        grid_velocity: Optional[np.ndarray] = None,
        is_ale_node: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute a full ALE FVM integration step.

        Steps:
        1. Compute face normals, acoustic impedances, and normal velocities
        2. Compute numerical interface pressures and internal forces (sfint3)
        3. Add external / gravity forces
        4. Compute 3D face fluxes (aflux3 / eflux3)
        5. Update momentum by net forces: dMOM = dt * FCELL (scheme)
        6. Convectively transport density and momentum across faces (a22conv3)
        7. Expand cell momentum to nodes for post-treatment / velocity update
        8. Reset nodal accelerations (accele)
        """
        # Step 1: Stress & geometry buffer preparation
        alefvm_prepare_stress_buffer(state, x, connectivity)

        # Step 2: Internal force calculation
        fint = alefvm_compute_internal_forces(
            state,
            neighbor_elem,
            neighbor_face,
            solver_type=self.params.solver_type,
            low_mach_fix=self.params.low_mach_fix,
        )

        # Step 3: Body force / gravity
        if np.any(self.params.gravity != 0.0):
            mass = state.rho * state.vol
            state.fext_cell = mass[:, np.newaxis] * self.params.gravity[np.newaxis, :]
            state.fcell = state.fint_cell + state.fext_cell

        # Step 4: Fluxes
        flux_data = alefvm_compute_face_fluxes(
            state,
            x,
            connectivity,
            neighbor_elem,
            grid_velocity=grid_velocity,
            solver_type=self.params.solver_type,
            upwl=self.params.upwind_factor,
            reduc=self.params.reduc_boundary,
            dt1=dt_prev,
        )

        # Step 5: Momentum update
        dmom_force = alefvm_update_momentum(state, dt=dt, dt_prev=dt_prev)

        # Step 6: Convective transport of mass and momentum
        # Advect mass: delta(mass) = alefvm_advect(rho, ...)
        d_mass = alefvm_advect_scalar(
            state.rho,
            flux_data["flux_upwind"],
            flux_data["flu1"],
            neighbor_elem,
            dt=dt,
        )
        new_mass = np.maximum(state.rho * state.vol + d_mass, 1e-30)
        state.rho = new_mass / state.vol

        # Advect momentum:
        dmom_conv = alefvm_advect_scalar(
            state.mom / state.vol[:, np.newaxis],
            flux_data["flux_upwind"],
            flux_data["flu1"],
            neighbor_elem,
            dt=dt,
        )
        state.mom += dmom_conv

        # Step 7: Expand momentum to nodes
        v_node = alefvm_expand_momentum_to_nodes(
            state,
            connectivity,
            nodal_mass,
            is_ale_node=is_ale_node,
        )

        # Step 8: Reset accelerations
        a_node = alefvm_reset_accelerations(len(x), is_ale_node=is_ale_node)

        return {
            "fint": fint,
            "fcell": state.fcell,
            "fluxes": flux_data["flux_raw"],
            "dmom_force": dmom_force,
            "dmom_conv": dmom_conv,
            "nodal_velocity": v_node,
            "nodal_accel": a_node,
            "energy_kinetic": state.energy_kinetic,
            "energy_internal_work": state.energy_internal_work,
        }
