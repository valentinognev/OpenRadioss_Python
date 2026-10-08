# pyradioss/engine/multifluid.py
"""
Multi-Material Compressible Finite Volume Method (FVM) Fluid Solver.

Ported from OpenRadioss Fortran sources:
- engine/source/multifluid/multi_pressure_equilibrium.F:
    Newton-Raphson multi-fluid pressure equilibrium relaxation and energy reset.
- engine/source/multifluid/multi_muscl_fluxes_computation.F:
    2nd-order MUSCL Riemann flux solver with HLLC / Rusanov schemes and submaterial fluxes.
- engine/source/multifluid/multi_muscl_gradients.F:
    Least-squares spatial gradient reconstruction with Barth-Jespersen slope limiter.
- engine/source/multifluid/multi_timeevolution.F:
    Conservative state time evolution orchestration.
- engine/source/multifluid/multi_evolve_global.F & multi_update_global.F:
    Global mixture conservative variable time integration.
- engine/source/multifluid/multi_evolve_partial.F & multi_update_partial.F:
    Submaterial volume, mass, and internal energy partial time integration.
- engine/source/multifluid/ns_fvm_diffusion.F:
    Navier-Stokes viscous stress tensor diffusion with strict energy accounting.
- engine/source/multifluid/multi_compute_dt.F:
    CFL-limited acoustic/convective time step computation.
- engine/source/multifluid/multi_computevolume.F:
    Cell volume and geometric factor evaluation.
- engine/source/multifluid/multi_fvm2fem.F:
    FVM-to-FEM fluid pressure force assembly.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple, Dict, List, Any
from dataclasses import dataclass, field
import numpy as np


# =============================================================================
# 1. Equation of State (EOS) Representation
# =============================================================================

class MaterialEOS:
    """Equation of State model for single-phase or multi-material components.

    Ported from OpenRadioss Fortran sources:
    - starter/source/materials/eos/hm_read_eos.F
    - common_source/eos/idealgas.F
    - common_source/eos/stiffgas.F
    - engine/source/multifluid/multi_pressure_equilibrium.F
    """

    def __init__(
        self,
        kind: str = "IDEAL-GAS",
        gamma: float = 1.4,
        p_star: float = 0.0,
        psh: float = 0.0,
        pmin: float = -1e30,
        rho0: float = 1.0,
        **kwargs: Any,
    ):
        self.kind = kind.upper()
        self.gamma = float(gamma)
        self.p_star = float(p_star)
        self.psh = float(psh)
        self.pmin = float(pmin)
        self.rho0 = float(rho0)
        self.extra_params = kwargs

    def compute_pressure(
        self,
        rho: Union[float, np.ndarray],
        eint_vol: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute pressure P from density rho and volumetric internal energy eint_vol (rho * e).

        # Ported from $OR_SRC/common_source/eos/idealgas.F
        # and engine/source/multifluid/multi_pressure_equilibrium.F lines 291-305
        """
        if self.kind in ("IDEAL-GAS", "IDEAL_GAS"):
            # P = (gamma - 1) * rho * e - psh
            p = (self.gamma - 1.0) * eint_vol - self.psh
            return np.maximum(p, self.pmin)
        elif self.kind in ("STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS"):
            # P = (gamma - 1) * rho * e - gamma * p_star - psh
            p = (self.gamma - 1.0) * eint_vol - self.gamma * self.p_star - self.psh
            return np.maximum(p, self.pmin)
        else:
            p = (self.gamma - 1.0) * eint_vol - self.psh
            return np.maximum(p, self.pmin)

    def compute_sound_speed(
        self,
        rho: Union[float, np.ndarray],
        pres: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute sound speed c from density rho and pressure P.

        # Ported from $OR_SRC/engine/source/multifluid/multi_pressure_equilibrium.F lines 178-184
        """
        rho_safe = np.maximum(rho, 1e-12)
        if self.kind in ("STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS"):
            arg = (self.gamma * (pres + self.psh + self.p_star)) / rho_safe
        else:
            arg = (self.gamma * (pres + self.psh)) / rho_safe
        return np.sqrt(np.maximum(arg, 1e-20))

    def gruneisen(
        self,
        rho: Union[float, np.ndarray] = 1.0,
        eint_vol: Union[float, np.ndarray] = 0.0,
    ) -> Union[float, np.ndarray]:
        """Evaluate Grüneisen coefficient Gamma = (1/rho) * (dP/de)_rho.

        # Ported from $OR_SRC/engine/source/multifluid/multi_pressure_equilibrium.F lines 350-353
        """
        if np.isscalar(rho):
            return self.gamma - 1.0
        return np.full_like(rho, self.gamma - 1.0, dtype=np.float64)


# =============================================================================
# 2. FVM Mesh Representation
# =============================================================================

class FVMMesh:
    """Finite Volume Mesh supporting 1D, 2D, and 3D cell-face connectivity.

    Ported from OpenRadioss Fortran sources:
    - common_source/modules/ale/ale_connectivity_mod.F
    - engine/source/multifluid/multi_computevolume.F
    - engine/source/multifluid/multi_face_data_elem.F
    """

    def __init__(
        self,
        n_cells: int,
        cell_centers: np.ndarray,
        cell_volumes: np.ndarray,
        face_left: np.ndarray,
        face_right: np.ndarray,
        face_normals: np.ndarray,
        face_areas: np.ndarray,
        face_centers: np.ndarray,
        face_w: Optional[np.ndarray] = None,
        dim: int = 1,
    ):
        self.dim = dim
        self.n_cells = int(n_cells)
        self.cell_centers = np.asarray(cell_centers, dtype=np.float64)  # (n_cells, 3)
        self.cell_volumes = np.asarray(cell_volumes, dtype=np.float64)  # (n_cells,)

        self.n_faces = len(face_left)
        self.face_left = np.asarray(face_left, dtype=np.int64)          # (n_faces,)
        self.face_right = np.asarray(face_right, dtype=np.int64)        # (n_faces,) -1 if boundary
        self.face_normals = np.asarray(face_normals, dtype=np.float64)  # (n_faces, 3) unit normal
        self.face_areas = np.asarray(face_areas, dtype=np.float64)      # (n_faces,)
        self.face_centers = np.asarray(face_centers, dtype=np.float64)  # (n_faces, 3)

        if face_w is None:
            self.face_w = np.zeros((self.n_faces, 3), dtype=np.float64)
        else:
            self.face_w = np.asarray(face_w, dtype=np.float64)

        # Build cell-to-faces lookup tables:
        # cell_face_indices[i]: list of face indices attached to cell i
        # cell_face_directions[i]: +1 if cell i is face_left (outward normal), -1 if face_right
        # cell_neighbors[i]: list of neighboring cell indices
        self.cell_face_indices: List[List[int]] = [[] for _ in range(self.n_cells)]
        self.cell_face_directions: List[List[float]] = [[] for _ in range(self.n_cells)]
        self.cell_neighbors: List[List[int]] = [[] for _ in range(self.n_cells)]

        for f_idx in range(self.n_faces):
            c_l = self.face_left[f_idx]
            c_r = self.face_right[f_idx]
            if 0 <= c_l < self.n_cells:
                self.cell_face_indices[c_l].append(f_idx)
                self.cell_face_directions[c_l].append(1.0)
                if c_r >= 0:
                    self.cell_neighbors[c_l].append(c_r)
            if 0 <= c_r < self.n_cells:
                self.cell_face_indices[c_r].append(f_idx)
                self.cell_face_directions[c_r].append(-1.0)
                if c_l >= 0:
                    self.cell_neighbors[c_r].append(c_l)

    @classmethod
    def create_1d(cls, x_nodes: np.ndarray) -> FVMMesh:
        """Create a 1D FVM mesh from 1D nodal coordinates.

        Args:
            x_nodes: 1D array of node positions along x, shape (n_nodes,).
        """
        x = np.asarray(x_nodes, dtype=np.float64)
        n_cells = len(x) - 1
        if n_cells < 1:
            raise ValueError("1D mesh requires at least 2 nodes")

        dx = x[1:] - x[:-1]
        x_mid = 0.5 * (x[:-1] + x[1:])
        cell_centers = np.zeros((n_cells, 3), dtype=np.float64)
        cell_centers[:, 0] = x_mid
        cell_volumes = dx.copy()

        # Interior faces + 2 boundary faces = n_cells + 1 faces
        n_faces = n_cells + 1
        face_left = np.full(n_faces, -1, dtype=np.int64)
        face_right = np.full(n_faces, -1, dtype=np.int64)
        face_normals = np.zeros((n_faces, 3), dtype=np.float64)
        face_centers = np.zeros((n_faces, 3), dtype=np.float64)
        face_areas = np.ones(n_faces, dtype=np.float64)

        # Left boundary face at x[0]:
        face_left[0] = 0
        face_right[0] = -1
        face_normals[0, 0] = -1.0  # Outward normal pointing left (-x)
        face_centers[0, 0] = x[0]

        # Interior faces between cell i-1 and cell i:
        for i in range(1, n_cells):
            face_left[i] = i - 1
            face_right[i] = i
            face_normals[i, 0] = 1.0   # Pointing from i-1 to i (+x)
            face_centers[i, 0] = x[i]

        # Right boundary face at x[-1]:
        face_left[n_cells] = n_cells - 1
        face_right[n_cells] = -1
        face_normals[n_cells, 0] = 1.0  # Outward normal pointing right (+x)
        face_centers[n_cells, 0] = x[-1]

        return cls(
            n_cells=n_cells,
            cell_centers=cell_centers,
            cell_volumes=cell_volumes,
            face_left=face_left,
            face_right=face_right,
            face_normals=face_normals,
            face_areas=face_areas,
            face_centers=face_centers,
            dim=1,
        )

    @classmethod
    def create_uniform_1d(cls, n_cells: int, x_min: float = 0.0, x_max: float = 1.0) -> FVMMesh:
        """Create a uniform 1D mesh with n_cells on [x_min, x_max]."""
        x_nodes = np.linspace(x_min, x_max, n_cells + 1)
        return cls.create_1d(x_nodes)


# =============================================================================
# 3. Cell State Data Structure
# =============================================================================

@dataclass
class CellState:
    """Conserved and primitive state arrays across all cells.

    Ported from OpenRadioss Fortran sources:
    - common_source/modules/ale/multi_fvm_mod.F90
    - engine/source/multifluid/multi_timeevolution.F
    """

    n_cells: int
    n_mat: int = 1
    # Global mixture variables
    rho: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    eint: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    pres: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    sound_speed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    vol: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    # Multi-material partial variables: shape (n_mat, n_cells)
    phase_alpha: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    phase_rho: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    phase_eint: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    phase_pres: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    phase_sound_speed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))

    # Equations of state (one per material)
    eos_list: List[MaterialEOS] = field(default_factory=list)

    def __post_init__(self):
        if len(self.rho) == 0 and self.n_cells > 0:
            self.rho = np.ones(self.n_cells, dtype=np.float64)
            self.vel = np.zeros((self.n_cells, 3), dtype=np.float64)
            self.eint = np.zeros(self.n_cells, dtype=np.float64)
            self.pres = np.zeros(self.n_cells, dtype=np.float64)
            self.sound_speed = np.ones(self.n_cells, dtype=np.float64)
            self.vol = np.ones(self.n_cells, dtype=np.float64)

        if self.phase_alpha.size == 0 and self.n_cells > 0:
            self.phase_alpha = np.zeros((self.n_mat, self.n_cells), dtype=np.float64)
            self.phase_alpha[0, :] = 1.0
            self.phase_rho = np.tile(self.rho, (self.n_mat, 1))
            self.phase_eint = np.tile(self.eint, (self.n_mat, 1))
            self.phase_pres = np.tile(self.pres, (self.n_mat, 1))
            self.phase_sound_speed = np.tile(self.sound_speed, (self.n_mat, 1))

        if not self.eos_list:
            self.eos_list = [MaterialEOS() for _ in range(self.n_mat)]

    def copy(self) -> CellState:
        """Create a deep copy of the cell state arrays."""
        return CellState(
            n_cells=self.n_cells,
            n_mat=self.n_mat,
            rho=self.rho.copy(),
            vel=self.vel.copy(),
            eint=self.eint.copy(),
            pres=self.pres.copy(),
            sound_speed=self.sound_speed.copy(),
            vol=self.vol.copy(),
            phase_alpha=self.phase_alpha.copy(),
            phase_rho=self.phase_rho.copy(),
            phase_eint=self.phase_eint.copy(),
            phase_pres=self.phase_pres.copy(),
            phase_sound_speed=self.phase_sound_speed.copy(),
            eos_list=list(self.eos_list),
        )


# =============================================================================
# 4. Multi-Fluid Pressure Equilibrium
# =============================================================================

def pressure_equilibrium(
    cell_state: CellState,
    max_iter: int = 50,
    tol: float = 1e-4,
    pshift: float = 0.0,
) -> CellState:
    """Iterative multi-phase pressure equilibrium solve with energy reset.

    Ported from $OR_SRC/engine/source/multifluid/multi_pressure_equilibrium.F
    lines 151-758:
      Stage 1: Multi-phase pressure relaxation (Newton-Raphson on sum(alpha_m) = 1)
      Stage 2: Global energy reset ensuring sum(alpha_m * eint_m) = eint_global
      Stage 3: Mixture sound speed and pressure evaluation

    Args:
        cell_state: Current cell state with phase volume fractions, densities, energies.
        max_iter: Maximum Newton iterations per cell.
        tol: Relative pressure increment convergence tolerance.
        pshift: Pressure shift for cavitation or stiffened fluid laws.

    Returns:
        Updated cell_state with relaxed pressures, phase fractions, and sound speeds.
    """
    n_mat = cell_state.n_mat
    n_cells = cell_state.n_cells

    # Monofluid case: direct EOS evaluation
    if n_mat == 1:
        eos = cell_state.eos_list[0]
        cell_state.pres = eos.compute_pressure(cell_state.rho, cell_state.eint)
        cell_state.sound_speed = eos.compute_sound_speed(cell_state.rho, cell_state.pres)
        cell_state.phase_alpha[0, :] = 1.0
        cell_state.phase_rho[0, :] = cell_state.rho
        cell_state.phase_eint[0, :] = cell_state.eint
        cell_state.phase_pres[0, :] = cell_state.pres
        cell_state.phase_sound_speed[0, :] = cell_state.sound_speed
        return cell_state

    # Multifluid Newton-Raphson relaxation:
    for c in range(n_cells):
        v_cell = cell_state.vol[c]
        alpha = cell_state.phase_alpha[:, c]
        rho_m = cell_state.phase_rho[:, c]
        eint_m = cell_state.phase_eint[:, c]  # volumetric internal energy of phase m

        # Filter active materials: alpha > 0 and mass fraction > 1e-14
        rho_tot = cell_state.rho[c]
        y_m = np.where(rho_tot > 0, (alpha * rho_m) / rho_tot, 0.0)
        active_indices = [m for m in range(n_mat) if alpha[m] > 1e-12 and y_m[m] > 1e-14]

        # Single active material in this cell:
        if len(active_indices) <= 1:
            idx = active_indices[0] if active_indices else 0
            cell_state.phase_eint[idx, c] = cell_state.eint[c]
            eos = cell_state.eos_list[idx]
            p_val = float(eos.compute_pressure(rho_m[idx], cell_state.eint[c]))
            c_val = float(eos.compute_sound_speed(rho_m[idx], p_val))
            cell_state.phase_pres[idx, c] = p_val
            cell_state.phase_sound_speed[idx, c] = c_val
            cell_state.pres[c] = p_val
            cell_state.sound_speed[c] = c_val
            continue

        # Multiple phases present in cell:
        mass_norm = alpha * rho_m  # normalized mass per cell volume
        tau = np.where(rho_m > 0, 1.0 / rho_m, 0.0)  # specific volume
        tau0 = tau.copy()
        spe_eint = np.where(rho_m > 0, eint_m / rho_m, 0.0)  # specific internal energy
        spe_eint0 = spe_eint.copy()

        pres_m = np.zeros(n_mat, dtype=np.float64)
        ssp_m = np.zeros(n_mat, dtype=np.float64)
        grun_m = np.zeros(n_mat, dtype=np.float64)

        for m in active_indices:
            eos = cell_state.eos_list[m]
            pres_m[m] = float(eos.compute_pressure(rho_m[m], eint_m[m]))
            c_s = float(eos.compute_sound_speed(rho_m[m], pres_m[m]))
            ssp_m[m] = c_s ** 2
            grun_m[m] = float(eos.gruneisen(rho_m[m], eint_m[m]))

        # Initial relaxed pressure guess P0 = sum(alpha_m * P_m)
        p_relaxed = float(np.sum(alpha[active_indices] * pres_m[active_indices]))

        # --- Stage 1: Newton-Raphson Outer Relaxation Loop ---
        for it in range(max_iter):
            coef_tot = 0.0
            p_incr_sum = 0.0

            for m in active_indices:
                fp_m = pres_m[m] - p_relaxed
                grun = grun_m[m]
                t_m = tau[m]
                dpde = grun / t_m
                dpdtau = -(ssp_m[m] - grun * (pres_m[m] + pshift) * t_m) / (t_m ** 2)
                denom = dpdtau - (p_relaxed + pshift) * dpde
                if abs(denom) < 1e-20:
                    denom = math.copysign(1e-20, denom)
                coef1 = mass_norm[m] / denom
                coef_tot += coef1 * (1.0 + grun)
                p_incr_sum -= fp_m * coef1

            if abs(coef_tot) < 1e-20:
                break

            pres_incr = p_incr_sum / coef_tot

            # Specific volume increments & step limiting:
            lim_tau = 1.0
            tau_incr = np.zeros(n_mat, dtype=np.float64)
            for m in active_indices:
                grun = grun_m[m]
                t_m = tau[m]
                dpde = grun / t_m
                dpdtau = -(ssp_m[m] - grun * (pres_m[m] + pshift) * t_m) / (t_m ** 2)
                denom = dpdtau - (p_relaxed + pshift) * dpde
                if abs(denom) < 1e-20:
                    denom = math.copysign(1e-20, denom)
                coef1 = 1.0 / denom
                tau_incr[m] = coef1 * (pres_incr * (1.0 + grun) + (pres_m[m] - p_relaxed))

                if tau[m] - tau_incr[m] < 0.0 and tau_incr[m] > 0.0:
                    lim_tau = min(lim_tau, 0.5 * tau[m] / tau_incr[m])

            # Apply state update:
            for m in active_indices:
                tau[m] -= lim_tau * tau_incr[m]
                rho_m[m] = 1.0 / tau[m]
                alpha[m] = mass_norm[m] * tau[m]
                spe_eint[m] = spe_eint0[m] - (p_relaxed + pshift) * (tau[m] - tau0[m])
                eint_m[m] = rho_m[m] * spe_eint[m]

            p_relaxed -= pres_incr

            # Recompute EOS values:
            for m in active_indices:
                eos = cell_state.eos_list[m]
                pres_m[m] = float(eos.compute_pressure(rho_m[m], eint_m[m]))
                c_s = float(eos.compute_sound_speed(rho_m[m], pres_m[m]))
                ssp_m[m] = c_s ** 2
                grun_m[m] = float(eos.gruneisen(rho_m[m], eint_m[m]))

            if abs(pres_incr) < tol * (1.0 + abs(p_relaxed)):
                break

        # --- Stage 2: Energy Reset (sum(alpha_m * eint_m) = eint_global) ---
        # Ported from multi_pressure_equilibrium.F lines 583-686
        e_global = cell_state.eint[c]
        for it_e in range(10):
            fe = -e_global + np.sum(alpha[active_indices] * eint_m[active_indices])
            coef_e = 0.0
            p_incr_e = 0.0

            for m in active_indices:
                grun = grun_m[m]
                one_over_grun = 1.0 / grun if grun > 0 else 0.0
                fp_m = pres_m[m] - p_relaxed
                p_incr_e += alpha[m] * one_over_grun * fp_m
                coef_e += alpha[m] * one_over_grun

            if abs(coef_e) > 1e-20:
                pres_incr = (fe - p_incr_e) / coef_e
                p_relaxed -= pres_incr

                for m in active_indices:
                    grun = grun_m[m]
                    if grun > 0:
                        fp_m = pres_m[m] - p_relaxed
                        lim = 1.0
                        if (fp_m + pres_incr) > 0:
                            lim = max(0.0, min(1.0, 0.5 * eint_m[m] * grun / (fp_m + pres_incr)))
                        eint_m[m] = max(0.0, eint_m[m] - lim * (fp_m + pres_incr) / grun)

                for m in active_indices:
                    eos = cell_state.eos_list[m]
                    pres_m[m] = float(eos.compute_pressure(rho_m[m], eint_m[m]))
                    c_s = float(eos.compute_sound_speed(rho_m[m], pres_m[m]))
                    ssp_m[m] = c_s ** 2

                if abs(pres_incr) < tol * (1.0 + abs(p_relaxed)):
                    break

        # --- Stage 3: Store Final Values and Mixture Properties ---
        # Ported from multi_pressure_equilibrium.F lines 704-748
        norm_alpha = alpha.copy()
        sum_a = np.sum(norm_alpha[active_indices])
        if sum_a > 0:
            norm_alpha[active_indices] /= sum_a

        cell_state.phase_alpha[:, c] = norm_alpha
        cell_state.phase_rho[:, c] = rho_m
        cell_state.phase_eint[:, c] = eint_m
        cell_state.phase_pres[:, c] = pres_m
        cell_state.phase_sound_speed[:, c] = np.sqrt(ssp_m)

        # Mixture density, pressure, and sound speed:
        mix_p = float(np.sum(norm_alpha[active_indices] * pres_m[active_indices]))
        mix_ssp = float(np.sum(norm_alpha[active_indices] * rho_m[active_indices] * ssp_m[active_indices]))
        rho_c = cell_state.rho[c]
        mix_c = math.sqrt(mix_ssp / max(rho_c, 1e-12)) if rho_c > 0 else 1e-10

        cell_state.pres[c] = mix_p
        cell_state.sound_speed[c] = mix_c

    return cell_state


# =============================================================================
# 5. MUSCL Spatial Gradient Reconstruction
# =============================================================================

def limiter_sweby(phi: float, beta: float = 1.0) -> float:
    """Barth-Jespersen / Sweby slope limiter function.

    Ported from $OR_SRC/engine/source/multifluid/multi_muscl_gradients.F
    lines 631-640 (SUBROUTINE LIMITER):
      LIM = MAX(ZERO, MAX(MIN(ONE, BETA * PHI), MIN(PHI, BETA)))
    """
    return max(0.0, max(min(1.0, beta * phi), min(phi, beta)))


def muscl_gradients(
    cell_state: CellState,
    mesh: FVMMesh,
    beta: float = 1.0,
    use_limiter: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute least-squares spatial gradients with Barth-Jespersen slope limiter.

    Ported from $OR_SRC/engine/source/multifluid/multi_muscl_gradients.F
    lines 163-620:
      For each cell K:
        M = sum_F (x_L - x_K) (x_L - x_K)^T
        RHS = sum_F (phi_K - phi_L) (x_L - x_K)
        grad = -M^{-1} RHS = M^{-1} sum_F (phi_L - phi_K) (x_L - x_K)
        Limiter: Delta_F = grad . (x_F - x_K)
        Phi_F = 0.5 * (phi_extrema - phi_K) / Delta_F
        grad_limited = min_F(limiter(Phi_F, beta)) * grad

    Args:
        cell_state: Conserved and primitive state arrays.
        mesh: FVMMesh with connectivity and geometry.
        beta: Compression / sharpness parameter for Sweby limiter (default 1.0).
        use_limiter: Whether to apply the Barth-Jespersen slope limiter.

    Returns:
        dict containing gradients for:
        - 'vel': (n_cells, 3, 3) where [c, i, :] is grad(vel_i) for cell c
        - 'rho': (n_cells, 3) grad(rho)
        - 'eint': (n_cells, 3) grad(eint)
        - 'pres': (n_cells, 3) grad(pres)
        - 'phase_alpha': (n_mat, n_cells, 3)
        - 'phase_rho': (n_mat, n_cells, 3)
        - 'phase_eint': (n_mat, n_cells, 3)
    """
    n_cells = mesh.n_cells
    n_mat = cell_state.n_mat

    grad_vel = np.zeros((n_cells, 3, 3), dtype=np.float64)
    grad_rho = np.zeros((n_cells, 3), dtype=np.float64)
    grad_eint = np.zeros((n_cells, 3), dtype=np.float64)
    grad_pres = np.zeros((n_cells, 3), dtype=np.float64)

    grad_p_alpha = np.zeros((n_mat, n_cells, 3), dtype=np.float64)
    grad_p_rho = np.zeros((n_mat, n_cells, 3), dtype=np.float64)
    grad_p_eint = np.zeros((n_mat, n_cells, 3), dtype=np.float64)

    for i in range(n_cells):
        xk = mesh.cell_centers[i]
        face_list = mesh.cell_face_indices[i]
        if not face_list:
            continue

        # Assemble least-squares matrix M:
        M = np.zeros((3, 3), dtype=np.float64)
        dx_list = []
        xf_list = []
        neighbors_valid = []

        for f_idx in face_list:
            c_l = mesh.face_left[f_idx]
            c_r = mesh.face_right[f_idx]
            j = c_r if c_l == i else c_l
            xf = mesh.face_centers[f_idx]

            if j >= 0:
                xl = mesh.cell_centers[j]
                neighbors_valid.append(j)
            else:
                # Boundary face: mirror centroid xl = 2 * xf - xk
                xl = 2.0 * xf - xk
                neighbors_valid.append(-1)

            dx = xl - xk
            M += np.outer(dx, dx)
            dx_list.append(dx)
            xf_list.append(xf)

        # Invert matrix M with pseudoinverse to handle 1D/2D/3D mesh degeneracies:
        pinv_M = np.linalg.pinv(M, rcond=1e-10)

        # Helper function to reconstruct a scalar field phi across cell i:
        def reconstruct_scalar(
            phi_c: float,
            phi_neighbors: List[float],
            beta_lim: float = 1.0,
            force_positive: bool = False,
        ) -> np.ndarray:
            max_val = phi_c
            min_val = phi_c
            rhs = np.zeros(3, dtype=np.float64)

            for idx_f, f_idx in enumerate(face_list):
                j = neighbors_valid[idx_f]
                dx = dx_list[idx_f]
                if j >= 0:
                    phi_l = phi_neighbors[idx_f]
                    max_val = max(max_val, phi_l)
                    min_val = min(min_val, phi_l)
                    rhs += (phi_l - phi_c) * dx

            g = pinv_M @ rhs
            if not use_limiter:
                return g

            # Barth-Jespersen face limitation:
            phi_lim = 1.0
            for idx_f, xf in enumerate(xf_list):
                delta = float(np.dot(g, xf - xk))
                lim_f = 1.0
                if delta > 1e-14:
                    lim_f = 0.5 * (max_val - phi_c) / delta
                elif delta < -1e-14:
                    lim_f = 0.5 * (min_val - phi_c) / delta

                if force_positive and (phi_c + delta <= 0.0):
                    lim_f = 0.0
                else:
                    lim_f = limiter_sweby(lim_f, beta_lim)

                phi_lim = min(phi_lim, lim_f)

            return g * phi_lim

        # Reconstruct velocity components (u, v, w):
        for comp in range(3):
            val_c = cell_state.vel[i, comp]
            val_nb = [
                cell_state.vel[j, comp] if j >= 0 else val_c
                for j in neighbors_valid
            ]
            grad_vel[i, comp, :] = reconstruct_scalar(val_c, val_nb, beta)

        # Monofluid scalars:
        if n_mat == 1:
            # Density:
            rho_c = cell_state.rho[i]
            rho_nb = [cell_state.rho[j] if j >= 0 else rho_c for j in neighbors_valid]
            grad_rho[i, :] = reconstruct_scalar(rho_c, rho_nb, beta, force_positive=True)

            # Internal energy:
            e_c = cell_state.eint[i]
            e_nb = [cell_state.eint[j] if j >= 0 else e_c for j in neighbors_valid]
            grad_eint[i, :] = reconstruct_scalar(e_c, e_nb, beta, force_positive=True)

            # Pressure:
            p_c = cell_state.pres[i]
            p_nb = [cell_state.pres[j] if j >= 0 else p_c for j in neighbors_valid]
            grad_pres[i, :] = reconstruct_scalar(p_c, p_nb, beta)
        else:
            # Multi-material partial scalars:
            for m in range(n_mat):
                # Volume fraction:
                a_c = cell_state.phase_alpha[m, i]
                a_nb = [cell_state.phase_alpha[m, j] if j >= 0 else a_c for j in neighbors_valid]
                grad_p_alpha[m, i, :] = reconstruct_scalar(a_c, a_nb, beta)

                # Phase density:
                prho_c = cell_state.phase_rho[m, i]
                prho_nb = [cell_state.phase_rho[m, j] if j >= 0 else prho_c for j in neighbors_valid]
                grad_p_rho[m, i, :] = reconstruct_scalar(prho_c, prho_nb, beta, force_positive=True)

                # Phase internal energy:
                peint_c = cell_state.phase_eint[m, i]
                peint_nb = [cell_state.phase_eint[m, j] if j >= 0 else peint_c for j in neighbors_valid]
                grad_p_eint[m, i, :] = reconstruct_scalar(peint_c, peint_nb, beta, force_positive=True)

            # Mixture gradients from phase gradients:
            grad_rho[i, :] = np.sum(grad_p_alpha[:, i, :] * cell_state.phase_rho[:, i, None] +
                                    cell_state.phase_alpha[:, i, None] * grad_p_rho[:, i, :], axis=0)
            grad_eint[i, :] = np.sum(cell_state.phase_alpha[:, i, None] * grad_p_eint[:, i, :], axis=0)

    return {
        "vel": grad_vel,
        "rho": grad_rho,
        "eint": grad_eint,
        "pres": grad_pres,
        "phase_alpha": grad_p_alpha,
        "phase_rho": grad_p_rho,
        "phase_eint": grad_p_eint,
    }


# =============================================================================
# 6. MUSCL Numerical Flux Solver (HLLC & Rusanov)
# =============================================================================

def muscl_fluxes(
    left_state: Dict[str, Any],
    right_state: Dict[str, Any],
    normal: Optional[np.ndarray] = None,
    area: float = 1.0,
    w_n: float = 0.0,
    method: str = "hllc",
    lowmach: bool = False,
    pshift: float = 0.0,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Compute numerical convective flux across a face interface.

    Ported from $OR_SRC/engine/source/multifluid/multi_muscl_fluxes_computation.F
    lines 477-734:
      HLLC approximate Riemann solver with optional Low-Mach correction and submaterial fluxes.

    Args:
        left_state: Dict containing reconstructed primitive values for left cell (II):
                    'rho', 'vel' (3,), 'eint', 'pres', 'sound_speed',
                    optional 'phase_alpha', 'phase_rho', 'phase_eint', 'phase_pres'.
        right_state: Dict containing reconstructed primitive values for right cell (JJ).
        normal: (3,) unit outward normal from left to right.
        area: Face surface area.
        w_n: Grid velocity projected onto face normal (w . n), 0 for Eulerian.
        method: 'hllc' (OpenRadioss default) or 'rusanov'.
        lowmach: Enable low Mach number pressure correction.
        pshift: Pressure shift.

    Returns:
        (flux_global, sub_fluxes):
          - flux_global: (5,) array [F_rho, F_mom_x, F_mom_y, F_mom_z, F_E]
          - sub_fluxes: dict containing 'vol', 'mass', 'ener' of shape (n_mat,) if multi-material
    """
    if normal is None:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        normal = np.asarray(normal, dtype=np.float64)
    nx, ny, nz = normal

    rho_l = max(float(left_state["rho"]), 1e-12)
    rho_r = max(float(right_state["rho"]), 1e-12)

    vel_l = np.asarray(left_state["vel"], dtype=np.float64)
    vel_r = np.asarray(right_state["vel"], dtype=np.float64)
    vn_l = float(np.dot(vel_l, normal))
    vn_r = float(np.dot(vel_r, normal))
    v2_l = float(np.dot(vel_l, vel_l))
    v2_r = float(np.dot(vel_r, vel_r))

    p_l = float(left_state["pres"])
    p_r = float(right_state["pres"])
    c_l = max(float(left_state["sound_speed"]), 1e-10)
    c_r = max(float(right_state["sound_speed"]), 1e-10)

    eint_l = float(left_state["eint"])
    eint_r = float(right_state["eint"])
    etot_l = eint_l + 0.5 * rho_l * v2_l
    etot_r = eint_r + 0.5 * rho_r * v2_r

    # Conserved state vectors: U = [rho, rho*u, rho*v, rho*w, E]
    u_l = np.array([rho_l, rho_l * vel_l[0], rho_l * vel_l[1], rho_l * vel_l[2], etot_l], dtype=np.float64)
    u_r = np.array([rho_r, rho_r * vel_r[0], rho_r * vel_r[1], rho_r * vel_r[2], etot_r], dtype=np.float64)

    # Physical flux vectors: F = [rho*vn, rho*u*vn + P*nx, rho*v*vn + P*ny, rho*w*vn + P*nz, (E + P + psh)*vn]
    f_l = np.array([
        rho_l * vn_l,
        rho_l * vel_l[0] * vn_l + p_l * nx,
        rho_l * vel_l[1] * vn_l + p_l * ny,
        rho_l * vel_l[2] * vn_l + p_l * nz,
        (etot_l + p_l + pshift) * vn_l,
    ], dtype=np.float64)

    f_r = np.array([
        rho_r * vn_r,
        rho_r * vel_r[0] * vn_r + p_r * nx,
        rho_r * vel_r[1] * vn_r + p_r * ny,
        rho_r * vel_r[2] * vn_r + p_r * nz,
        (etot_r + p_r + pshift) * vn_r,
    ], dtype=np.float64)

    # Multi-material phase data setup:
    is_multi = "phase_alpha" in left_state and "phase_alpha" in right_state
    sub_fluxes = None
    if is_multi:
        n_mat = len(left_state["phase_alpha"])
        sub_vol = np.zeros(n_mat, dtype=np.float64)
        sub_mass = np.zeros(n_mat, dtype=np.float64)
        sub_ener = np.zeros(n_mat, dtype=np.float64)

    if method.lower() == "rusanov":
        # Rusanov (Local Lax-Friedrichs) flux
        s_max = max(abs(vn_l) + c_l, abs(vn_r) + c_r)
        flux = 0.5 * (f_l + f_r) - 0.5 * s_max * (u_r - u_l) - w_n * (0.5 * (u_l + u_r))
        flux *= area
        if is_multi:
            vn_avg = 0.5 * (vn_l + vn_r) - w_n
            for m in range(n_mat):
                a_up = left_state["phase_alpha"][m] if vn_avg >= 0 else right_state["phase_alpha"][m]
                r_up = left_state["phase_rho"][m] if vn_avg >= 0 else right_state["phase_rho"][m]
                e_up = left_state["phase_eint"][m] if vn_avg >= 0 else right_state["phase_eint"][m]
                sub_vol[m] = a_up * vn_avg * area
                sub_mass[m] = a_up * r_up * vn_avg * area
                sub_ener[m] = a_up * e_up * vn_avg * area
            sub_fluxes = {"vol": sub_vol, "mass": sub_mass, "ener": sub_ener}
        return flux, sub_fluxes

    # --- HLLC Riemann Solver (OpenRadioss Fortran default) ---
    # Upstream source: multi_muscl_fluxes_computation.F lines 481-694
    s_l = min(vn_l - c_l, vn_r - c_r)
    s_r = max(vn_l + c_l, vn_r + c_r)

    denom_star = rho_l * (s_l - vn_l) - rho_r * (s_r - vn_r)
    if abs(denom_star) < 1e-14:
        denom_star = math.copysign(1e-14, denom_star)

    s_star = (p_r - p_l + rho_l * vn_l * (s_l - vn_l) - rho_r * vn_r * (s_r - vn_r)) / denom_star

    # Intermediate pressure P*
    p_star2 = p_l + rho_l * (s_star - vn_l) * (s_l - vn_l)

    if lowmach:
        mach_l = abs(vn_l) / c_l
        mach_r = abs(vn_r) / c_r
        min_mach = min(mach_l, mach_r)
        theta = min_mach if min_mach < 0.1 else 1.0
        p_star = (1.0 - theta) * 0.5 * (p_l + p_r) + theta * p_star2
    else:
        p_star = p_star2

    pp = np.array([0.0, p_star * nx, p_star * ny, p_star * nz, s_star * (p_star + pshift)], dtype=np.float64)

    # 4 Wave regimes depending on grid velocity w_n:
    if s_l > w_n:
        # Supersonic flow from left
        flux = (f_l - w_n * u_l) * area
        if is_multi:
            for m in range(n_mat):
                a_m = left_state["phase_alpha"][m]
                r_m = left_state["phase_rho"][m]
                e_m = left_state["phase_eint"][m]
                sub_vol[m] = a_m * (vn_l - w_n) * area
                sub_mass[m] = a_m * r_m * (vn_l - w_n) * area
                sub_ener[m] = a_m * e_m * (vn_l - w_n) * area

    elif s_l <= w_n <= s_star:
        # Left intermediate star state
        denom_l = s_star - s_l
        if abs(denom_l) < 1e-14:
            denom_l = math.copysign(1e-14, denom_l)
        u_l_star = (f_l - s_l * u_l - pp) / denom_l
        f_l_star = u_l_star * s_star + pp
        flux = (f_l_star - w_n * u_l_star) * area

        if is_multi:
            for m in range(n_mat):
                a_m = left_state["phase_alpha"][m]
                r_m = left_state["phase_rho"][m]
                e_m = left_state["phase_eint"][m]
                p_m = left_state["phase_pres"][m]
                r_star = r_m * (vn_l - s_l) / denom_l
                if a_m > 0 and r_m > 0 and r_star > 0:
                    e_spec = e_m / r_m - p_m * (1.0 / r_star - 1.0 / r_m)
                    e_spec = max(0.0, e_spec)
                else:
                    e_spec = 0.0
                sub_vol[m] = a_m * (s_star - w_n) * area
                sub_mass[m] = a_m * r_star * (s_star - w_n) * area
                sub_ener[m] = a_m * r_star * e_spec * (s_star - w_n) * area

    elif s_star < w_n <= s_r:
        # Right intermediate star state
        denom_r = s_star - s_r
        if abs(denom_r) < 1e-14:
            denom_r = math.copysign(1e-14, denom_r)
        u_r_star = (f_r - s_r * u_r - pp) / denom_r
        f_r_star = u_r_star * s_star + pp
        flux = (f_r_star - w_n * u_r_star) * area

        if is_multi:
            for m in range(n_mat):
                a_m = right_state["phase_alpha"][m]
                r_m = right_state["phase_rho"][m]
                e_m = right_state["phase_eint"][m]
                p_m = right_state["phase_pres"][m]
                r_star = r_m * (vn_r - s_r) / denom_r
                if a_m > 0 and r_m > 0 and r_star > 0:
                    e_spec = e_m / r_m - p_m * (1.0 / r_star - 1.0 / r_m)
                    e_spec = max(0.0, e_spec)
                else:
                    e_spec = 0.0
                sub_vol[m] = a_m * (s_star - w_n) * area
                sub_mass[m] = a_m * r_star * (s_star - w_n) * area
                sub_ener[m] = a_m * r_star * e_spec * (s_star - w_n) * area

    else:
        # Supersonic flow from right
        flux = (f_r - w_n * u_r) * area
        if is_multi:
            for m in range(n_mat):
                a_m = right_state["phase_alpha"][m]
                r_m = right_state["phase_rho"][m]
                e_m = right_state["phase_eint"][m]
                sub_vol[m] = a_m * (vn_r - w_n) * area
                sub_mass[m] = a_m * r_m * (vn_r - w_n) * area
                sub_ener[m] = a_m * e_m * (vn_r - w_n) * area

    if is_multi:
        sub_fluxes = {"vol": sub_vol, "mass": sub_mass, "ener": sub_ener}

    return flux, sub_fluxes


# =============================================================================
# 7. Time Evolution Integration
# =============================================================================

def time_evolution(
    cell_state: CellState,
    fluxes: np.ndarray,
    dt: float,
    mesh: Optional[FVMMesh] = None,
    sub_fluxes: Optional[Dict[str, np.ndarray]] = None,
    sources: Optional[np.ndarray] = None,
    pshift: float = 0.0,
) -> CellState:
    """Explicit Euler conservative state update.

    Ported from OpenRadioss Fortran sources:
    - $OR_SRC/engine/source/multifluid/multi_evolve_global.F (lines 122-194)
    - $OR_SRC/engine/source/multifluid/multi_update_global.F (lines 79-98)
    - $OR_SRC/engine/source/multifluid/multi_evolve_partial.F (lines 125-215)
    - $OR_SRC/engine/source/multifluid/multi_update_partial.F (lines 75-103)

    Args:
        cell_state: CellState to update.
        fluxes: (n_faces, 5) conservative fluxes for each face.
        dt: Time step.
        mesh: Optional FVMMesh providing cell-face topology and volumes.
        sub_fluxes: Optional dict containing 'vol', 'mass', 'ener' arrays of shape (n_faces, n_mat).
        sources: Optional (n_cells, 5) body force / external work source terms.
        pshift: Pressure shift.

    Returns:
        Updated cell_state instance.
    """
    n_cells = cell_state.n_cells
    n_mat = cell_state.n_mat

    vol_old = cell_state.vol
    vol_new = mesh.cell_volumes if mesh is not None else vol_old

    # Accumulate net global fluxes for each cell:
    net_flux = np.zeros((n_cells, 5), dtype=np.float64)
    if mesh is not None:
        for f_idx in range(mesh.n_faces):
            c_l = mesh.face_left[f_idx]
            c_r = mesh.face_right[f_idx]
            flx = fluxes[f_idx]
            if 0 <= c_l < n_cells:
                net_flux[c_l] += flx
            if 0 <= c_r < n_cells:
                net_flux[c_r] -= flx

    # Total energy per unit volume:
    v2_old = np.sum(cell_state.vel ** 2, axis=1)
    etot_old = cell_state.eint + 0.5 * cell_state.rho * v2_old

    u_old = np.column_stack([
        cell_state.rho,
        cell_state.rho * cell_state.vel[:, 0],
        cell_state.rho * cell_state.vel[:, 1],
        cell_state.rho * cell_state.vel[:, 2],
        etot_old,
    ])

    src = sources if sources is not None else np.zeros_like(u_old)

    # Conserved integrated update: U_new * V_new = U_old * V_old - dt * sum(F) + dt * S
    u_integrated = u_old * vol_old[:, None] - dt * net_flux + dt * src
    vol_safe = np.maximum(vol_new, 1e-16)
    u_new = u_integrated / vol_safe[:, None]

    # Extract updated primitives:
    rho_new = np.maximum(u_new[:, 0], 1e-12)
    vel_new = u_new[:, 1:4] / rho_new[:, None]
    v2_new = np.sum(vel_new ** 2, axis=1)
    eint_new = np.maximum(u_new[:, 4] - 0.5 * rho_new * v2_new, 0.0)

    cell_state.rho = rho_new
    cell_state.vel = vel_new
    cell_state.eint = eint_new

    # Update multi-material partial states:
    if n_mat > 1 and sub_fluxes is not None and mesh is not None:
        net_sub_vol = np.zeros((n_mat, n_cells), dtype=np.float64)
        net_sub_mass = np.zeros((n_mat, n_cells), dtype=np.float64)
        net_sub_ener = np.zeros((n_mat, n_cells), dtype=np.float64)
        sum_vol_flux = np.zeros(n_cells, dtype=np.float64)

        flx_vol = sub_fluxes["vol"]    # (n_faces, n_mat)
        flx_mass = sub_fluxes["mass"]  # (n_faces, n_mat)
        flx_ener = sub_fluxes["ener"]  # (n_faces, n_mat)

        for f_idx in range(mesh.n_faces):
            c_l = mesh.face_left[f_idx]
            c_r = mesh.face_right[f_idx]
            f_v = flx_vol[f_idx]
            f_m = flx_mass[f_idx]
            f_e = flx_ener[f_idx]
            tot_v = np.sum(f_v)

            if 0 <= c_l < n_cells:
                net_sub_vol[:, c_l] += f_v
                net_sub_mass[:, c_l] += f_m
                net_sub_ener[:, c_l] += f_e
                sum_vol_flux[c_l] += tot_v
            if 0 <= c_r < n_cells:
                net_sub_vol[:, c_r] -= f_v
                net_sub_mass[:, c_r] -= f_m
                net_sub_ener[:, c_r] -= f_e
                sum_vol_flux[c_r] -= tot_v

        for m in range(n_mat):
            alpha_m = cell_state.phase_alpha[m]
            rho_m = cell_state.phase_rho[m]
            eint_m = cell_state.phase_eint[m]
            pres_m = cell_state.phase_pres[m]

            # Partial volume, mass, energy updates (multi_evolve_partial.F lines 197-213):
            vol_partial_old = alpha_m * vol_old
            mass_partial_old = vol_partial_old * rho_m
            ener_partial_old = vol_partial_old * eint_m

            vol_partial_new = vol_partial_old - dt * (net_sub_vol[m] - alpha_m * sum_vol_flux)
            mass_partial_new = mass_partial_old - dt * net_sub_mass[m]
            ener_partial_new = ener_partial_old - dt * (net_sub_ener[m] + alpha_m * (pres_m + pshift) * sum_vol_flux)

            vol_partial_new = np.maximum(vol_partial_new, 0.0)
            mass_partial_new = np.maximum(mass_partial_new, 0.0)
            ener_partial_new = np.maximum(ener_partial_new, 0.0)

            # Filtering small volume fractions (< 1e-8) as in multi_update_partial.F lines 85-89:
            alpha_new = vol_partial_new / vol_safe
            mask_small = alpha_new < 1e-8
            alpha_new[mask_small] = 0.0

            rho_m_new = np.where(vol_partial_new > 1e-14, mass_partial_new / np.maximum(vol_partial_new, 1e-14), cell_state.eos_list[m].rho0)
            eint_m_new = np.where(vol_partial_new > 1e-14, ener_partial_new / np.maximum(vol_partial_new, 1e-14), 0.0)

            cell_state.phase_alpha[m] = alpha_new
            cell_state.phase_rho[m] = rho_m_new
            cell_state.phase_eint[m] = eint_m_new

        # Re-normalize volume fractions to exactly 1.0:
        sum_alpha = np.sum(cell_state.phase_alpha, axis=0)
        mask_valid = sum_alpha > 0
        cell_state.phase_alpha[:, mask_valid] /= sum_alpha[mask_valid]

    # Re-evaluate pressure equilibrium:
    pressure_equilibrium(cell_state, pshift=pshift)

    return cell_state


# =============================================================================
# 8. Navier-Stokes Viscous Stress Diffusion
# =============================================================================

def viscous_diffusion(
    cell_state: CellState,
    mesh: FVMMesh,
    dt: float,
    dynamic_viscosity: float = 0.0,
    kinematic_viscosity: Optional[float] = None,
) -> CellState:
    """Navier-Stokes viscous stress diffusion on velocity with strict energy conservation.

    Ported from $OR_SRC/engine/source/multifluid/ns_fvm_diffusion.F
    lines 166-300:
      F_visc = sum_faces Area * mu_face * (v_L - v_K) / dist_KL
      (rho * V) * dv/dt = F_visc
      Energy balance (lines 290-297):
      E_total = E_total_old -> eint_new = eint_old + 0.5 * rho * (|v_old|^2 - |v_new|^2)

    Args:
        cell_state: CellState instance with vel, rho, eint.
        mesh: FVMMesh containing faces and distances.
        dt: Time step.
        dynamic_viscosity: Dynamic shear viscosity mu [Pa.s].
        kinematic_viscosity: Optional kinematic viscosity nu = mu / rho.

    Returns:
        Updated cell_state with diffused velocity and updated internal energy.
    """
    if dynamic_viscosity <= 0.0 and (kinematic_viscosity is None or kinematic_viscosity <= 0.0):
        return cell_state

    n_cells = mesh.n_cells
    f_visc = np.zeros((n_cells, 3), dtype=np.float64)

    # Compute cell dynamic viscosity:
    if kinematic_viscosity is not None and kinematic_viscosity > 0.0:
        mu_cell = kinematic_viscosity * cell_state.rho
    else:
        mu_cell = np.full(n_cells, dynamic_viscosity, dtype=np.float64)

    # Loop over faces:
    for f_idx in range(mesh.n_faces):
        c_l = mesh.face_left[f_idx]
        c_r = mesh.face_right[f_idx]
        if c_l < 0 or c_r < 0:
            continue  # Boundary face treatment handled separately or zero-traction

        xk = mesh.cell_centers[c_l]
        xl = mesh.cell_centers[c_r]
        dist = float(np.linalg.norm(xl - xk))
        if dist < 1e-12:
            continue

        mu_f = 0.5 * (mu_cell[c_l] + mu_cell[c_r])
        surf = mesh.face_areas[f_idx]

        # Viscous force F = Area * mu * (v_r - v_l) / dist
        f_face = surf * mu_f * (cell_state.vel[c_r] - cell_state.vel[c_l]) / dist
        f_visc[c_l] += f_face
        f_visc[c_r] -= f_face

    # Velocity update:
    mass_cell = cell_state.rho * mesh.cell_volumes
    mass_safe = np.maximum(mass_cell, 1e-16)

    vel_old = cell_state.vel.copy()
    v2_old = np.sum(vel_old ** 2, axis=1)

    vel_new = vel_old + (dt / mass_safe[:, None]) * f_visc
    v2_new = np.sum(vel_new ** 2, axis=1)

    # Energy accounting (ns_fvm_diffusion.F lines 290-297):
    # Kinetic energy dissipated by viscosity is converted to internal energy:
    d_ekin = 0.5 * cell_state.rho * (v2_new - v2_old)
    cell_state.eint = np.maximum(0.0, cell_state.eint - d_ekin)
    cell_state.vel = vel_new

    # Update pressure and sound speed with new internal energy:
    pressure_equilibrium(cell_state)

    return cell_state


# =============================================================================
# 9. CFL-Limited FVM Time Step Computation
# =============================================================================

def compute_fvm_dt(
    cell_state: CellState,
    mesh: FVMMesh,
    cfl: float = 0.8,
    wgrid: Optional[np.ndarray] = None,
) -> float:
    """Compute CFL-limited time step for explicit FVM integration.

    Ported from $OR_SRC/engine/source/multifluid/multi_compute_dt.F
    lines 91-134:
      For each cell K:
        For each face f:
          v_rel_n = (vel_K - w_f) . n_f
          1/dt_K = max_f (surf / vol * (sound_speed + |v_rel_n|) / CFL)
      dt = min_K dt_K

    Args:
        cell_state: Current cell state with vel, sound_speed, vol.
        mesh: FVMMesh with face normals, areas, volumes.
        cfl: Courant-Friedrichs-Lewy safety number (default 0.8).
        wgrid: Optional mesh node velocity array.

    Returns:
        Allowable time step dt.
    """
    n_cells = mesh.n_cells
    dt_inv_max = 0.0

    for i in range(n_cells):
        v_c = cell_state.vol[i]
        c_s = cell_state.sound_speed[i]
        vel_i = cell_state.vel[i]
        face_list = mesh.cell_face_indices[i]
        dirs = mesh.cell_face_directions[i]

        for idx_f, f_idx in enumerate(face_list):
            normal = mesh.face_normals[f_idx] * dirs[idx_f]
            surf = mesh.face_areas[f_idx]
            w_f = mesh.face_w[f_idx]
            v_rel_n = float(np.dot(vel_i - w_f, normal))
            speed = c_s + abs(v_rel_n)
            dt_inv = (surf / max(v_c, 1e-16)) * speed / cfl
            dt_inv_max = max(dt_inv_max, dt_inv)

    if dt_inv_max > 0.0:
        return 1.0 / dt_inv_max
    return 1e-6


# =============================================================================
# 10. FVM-to-FEM Force Mapping
# =============================================================================

def fvm2fem_forces(
    cell_state: CellState,
    mesh: FVMMesh,
    nodes: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Map cell FVM fluid pressure to nodal FEM force vectors.

    Ported from $OR_SRC/engine/source/multifluid/multi_fvm2fem.F
    lines 163-248:
      F_node = - 0.25 * P_cell * sum_faces (n_face * Area_face)

    Args:
        cell_state: CellState providing cell pressures.
        mesh: FVMMesh.
        nodes: (n_nodes, 3) nodal coordinates.
        connectivity: (n_cells, n_nodes_per_elem) node indices per element.

    Returns:
        (n_nodes, 3) nodal force array.
    """
    n_nodes = len(nodes)
    nodal_forces = np.zeros((n_nodes, 3), dtype=np.float64)

    for i in range(min(mesh.n_cells, len(connectivity))):
        p_c = cell_state.pres[i]
        conn = connectivity[i]
        n_elem_nodes = len(conn)
        factor = 1.0 / float(n_elem_nodes)

        face_list = mesh.cell_face_indices[i]
        dirs = mesh.cell_face_directions[i]
        elem_force = np.zeros(3, dtype=np.float64)

        for idx_f, f_idx in enumerate(face_list):
            norm = mesh.face_normals[f_idx] * dirs[idx_f]
            surf = mesh.face_areas[f_idx]
            elem_force -= norm * surf * p_c

        for node_id in conn:
            nodal_forces[node_id] += factor * elem_force

    return nodal_forces


# =============================================================================
# 11. MultifluidSolver Orchestrator Class
# =============================================================================

class MultifluidSolver:
    """Comprehensive Multi-Material Compressible Finite Volume Solver.

    Coordinates mesh, cell state, spatial reconstruction, numerical fluxes,
    pressure relaxation, and time integration according to OpenRadioss FVM physics.

    Ported from OpenRadioss Fortran sources:
    - engine/source/multifluid/multi_timeevolution.F
    - engine/source/ale/alemain.F
    """

    def __init__(
        self,
        mesh: FVMMesh,
        eos_list: Optional[List[MaterialEOS]] = None,
        beta: float = 1.0,
        cfl: float = 0.8,
        method: str = "hllc",
        lowmach: bool = False,
        dynamic_viscosity: float = 0.0,
        pshift: float = 0.0,
        boundary_conditions: Optional[Dict[int, str]] = None,
    ):
        self.mesh = mesh
        self.beta = float(beta)
        self.cfl = float(cfl)
        self.method = method.lower()
        self.lowmach = bool(lowmach)
        self.dynamic_viscosity = float(dynamic_viscosity)
        self.pshift = float(pshift)
        self.boundary_conditions = boundary_conditions or {}

        n_mat = len(eos_list) if eos_list is not None else 1
        self.n_mat = n_mat
        self.eos_list = eos_list or [MaterialEOS() for _ in range(n_mat)]

        self.state = CellState(
            n_cells=mesh.n_cells,
            n_mat=n_mat,
            vol=mesh.cell_volumes.copy(),
            eos_list=self.eos_list,
        )
        self.time = 0.0
        self.cycle = 0

    def initialize_state(
        self,
        rho: np.ndarray,
        vel: np.ndarray,
        eint: np.ndarray,
        phase_alpha: Optional[np.ndarray] = None,
        phase_rho: Optional[np.ndarray] = None,
        phase_eint: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize simulation state across all mesh cells."""
        self.state.rho = np.asarray(rho, dtype=np.float64).copy()
        self.state.vel = np.asarray(vel, dtype=np.float64).copy()
        self.state.eint = np.asarray(eint, dtype=np.float64).copy()
        self.state.vol = self.mesh.cell_volumes.copy()

        if self.n_mat > 1:
            if phase_alpha is not None:
                self.state.phase_alpha = np.asarray(phase_alpha, dtype=np.float64).copy()
            if phase_rho is not None:
                self.state.phase_rho = np.asarray(phase_rho, dtype=np.float64).copy()
            if phase_eint is not None:
                self.state.phase_eint = np.asarray(phase_eint, dtype=np.float64).copy()

        pressure_equilibrium(self.state, pshift=self.pshift)

    def compute_dt(self) -> float:
        """Compute maximum allowable time step."""
        return compute_fvm_dt(self.state, self.mesh, cfl=self.cfl)

    def reconstruct_face_states(
        self,
        grads: Dict[str, np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Reconstruct left and right primitive states at each mesh face interface.

        # Ported from multi_muscl_fluxes_computation.F lines 278-476
        """
        n_faces = self.mesh.n_faces
        left_states: List[Dict[str, Any]] = []
        right_states: List[Dict[str, Any]] = []

        g_vel = grads["vel"]
        g_rho = grads["rho"]
        g_eint = grads["eint"]
        g_pres = grads["pres"]

        for f in range(n_faces):
            c_l = self.mesh.face_left[f]
            c_r = self.mesh.face_right[f]
            xf = self.mesh.face_centers[f]

            # Left state reconstruction from cell c_l:
            xk = self.mesh.cell_centers[c_l]
            dx_l = xf - xk

            vel_l = self.state.vel[c_l] + np.dot(g_vel[c_l], dx_l)
            rho_l = max(1e-12, float(self.state.rho[c_l] + np.dot(g_rho[c_l], dx_l)))
            eint_l = max(0.0, float(self.state.eint[c_l] + np.dot(g_eint[c_l], dx_l)))

            if self.n_mat == 1:
                eos = self.eos_list[0]
                p_l = float(eos.compute_pressure(rho_l, eint_l))
                c_l = float(eos.compute_sound_speed(rho_l, p_l))
                st_l = {
                    "rho": rho_l,
                    "vel": vel_l,
                    "eint": eint_l,
                    "pres": p_l,
                    "sound_speed": c_l,
                }
            else:
                a_l = np.maximum(0.0, self.state.phase_alpha[:, c_l] + np.dot(grads["phase_alpha"][:, c_l, :], dx_l))
                if np.sum(a_l) > 0:
                    a_l /= np.sum(a_l)
                prho_l = np.maximum(1e-12, self.state.phase_rho[:, c_l] + np.dot(grads["phase_rho"][:, c_l, :], dx_l))
                peint_l = np.maximum(0.0, self.state.phase_eint[:, c_l] + np.dot(grads["phase_eint"][:, c_l, :], dx_l))
                ppres_l = np.zeros(self.n_mat, dtype=np.float64)
                pssp_l = np.zeros(self.n_mat, dtype=np.float64)
                for m in range(self.n_mat):
                    eos = self.eos_list[m]
                    ppres_l[m] = float(eos.compute_pressure(prho_l[m], peint_l[m]))
                    pssp_l[m] = float(eos.compute_sound_speed(prho_l[m], ppres_l[m])) ** 2
                p_l = float(np.sum(a_l * ppres_l))
                c_l = math.sqrt(float(np.sum(a_l * prho_l * pssp_l) / rho_l))
                st_l = {
                    "rho": rho_l,
                    "vel": vel_l,
                    "eint": eint_l,
                    "pres": p_l,
                    "sound_speed": c_l,
                    "phase_alpha": a_l,
                    "phase_rho": prho_l,
                    "phase_eint": peint_l,
                    "phase_pres": ppres_l,
                }

            # Right state reconstruction:
            if c_r >= 0:
                xl = self.mesh.cell_centers[c_r]
                dx_r = xf - xl
                vel_r = self.state.vel[c_r] + np.dot(g_vel[c_r], dx_r)
                rho_r = max(1e-12, float(self.state.rho[c_r] + np.dot(g_rho[c_r], dx_r)))
                eint_r = max(0.0, float(self.state.eint[c_r] + np.dot(g_eint[c_r], dx_r)))

                if self.n_mat == 1:
                    eos = self.eos_list[0]
                    p_r = float(eos.compute_pressure(rho_r, eint_r))
                    c_r = float(eos.compute_sound_speed(rho_r, p_r))
                    st_r = {
                        "rho": rho_r,
                        "vel": vel_r,
                        "eint": eint_r,
                        "pres": p_r,
                        "sound_speed": c_r,
                    }
                else:
                    a_r = np.maximum(0.0, self.state.phase_alpha[:, c_r] + np.dot(grads["phase_alpha"][:, c_r, :], dx_r))
                    if np.sum(a_r) > 0:
                        a_r /= np.sum(a_r)
                    prho_r = np.maximum(1e-12, self.state.phase_rho[:, c_r] + np.dot(grads["phase_rho"][:, c_r, :], dx_r))
                    peint_r = np.maximum(0.0, self.state.phase_eint[:, c_r] + np.dot(grads["phase_eint"][:, c_r, :], dx_r))
                    ppres_r = np.zeros(self.n_mat, dtype=np.float64)
                    pssp_r = np.zeros(self.n_mat, dtype=np.float64)
                    for m in range(self.n_mat):
                        eos = self.eos_list[m]
                        ppres_r[m] = float(eos.compute_pressure(prho_r[m], peint_r[m]))
                        pssp_r[m] = float(eos.compute_sound_speed(prho_r[m], ppres_r[m])) ** 2
                    p_r = float(np.sum(a_r * ppres_r))
                    c_r = math.sqrt(float(np.sum(a_r * prho_r * pssp_r) / rho_r))
                    st_r = {
                        "rho": rho_r,
                        "vel": vel_r,
                        "eint": eint_r,
                        "pres": p_r,
                        "sound_speed": c_r,
                        "phase_alpha": a_r,
                        "phase_rho": prho_r,
                        "phase_eint": peint_r,
                        "phase_pres": ppres_r,
                    }
            else:
                # Boundary face treatment:
                bc_type = self.boundary_conditions.get(f, "wall").lower()
                normal = self.mesh.face_normals[f]
                w_f = self.mesh.face_w[f]
                w_n = float(np.dot(w_f, normal))

                if bc_type == "wall":
                    # Reflecting solid wall: normal velocity inverted relative to grid normal velocity
                    vn_l = float(np.dot(vel_l, normal))
                    vel_refl = vel_l - 2.0 * (vn_l - w_n) * normal
                    st_r = dict(st_l)
                    st_r["vel"] = vel_refl
                else:  # 'outflow' / 'transmissive'
                    st_r = dict(st_l)

            left_states.append(st_l)
            right_states.append(st_r)

        return left_states, right_states

    def step(self, dt: Optional[float] = None) -> float:
        """Advance the fluid state by one explicit time step."""
        if dt is None:
            dt = self.compute_dt()

        # 1. MUSCL spatial gradients:
        grads = muscl_gradients(self.state, self.mesh, beta=self.beta)

        # 2. Reconstruct states at face interfaces:
        left_states, right_states = self.reconstruct_face_states(grads)

        # 3. Numerical flux calculation across all faces:
        n_faces = self.mesh.n_faces
        fluxes = np.zeros((n_faces, 5), dtype=np.float64)

        if self.n_mat > 1:
            sub_vol = np.zeros((n_faces, self.n_mat), dtype=np.float64)
            sub_mass = np.zeros((n_faces, self.n_mat), dtype=np.float64)
            sub_ener = np.zeros((n_faces, self.n_mat), dtype=np.float64)
            sub_fluxes = {"vol": sub_vol, "mass": sub_mass, "ener": sub_ener}
        else:
            sub_fluxes = None

        for f in range(n_faces):
            norm = self.mesh.face_normals[f]
            area = self.mesh.face_areas[f]
            w_n = float(np.dot(self.mesh.face_w[f], norm))

            flx, sflx = muscl_fluxes(
                left_states[f],
                right_states[f],
                normal=norm,
                area=area,
                w_n=w_n,
                method=self.method,
                lowmach=self.lowmach,
                pshift=self.pshift,
            )
            fluxes[f] = flx
            if self.n_mat > 1 and sflx is not None:
                sub_vol[f] = sflx["vol"]
                sub_mass[f] = sflx["mass"]
                sub_ener[f] = sflx["ener"]

        # 4. Conservative state update:
        time_evolution(
            self.state,
            fluxes,
            dt,
            mesh=self.mesh,
            sub_fluxes=sub_fluxes,
            pshift=self.pshift,
        )

        # 5. Viscous stress diffusion (if enabled):
        if self.dynamic_viscosity > 0.0:
            viscous_diffusion(
                self.state,
                self.mesh,
                dt,
                dynamic_viscosity=self.dynamic_viscosity,
            )

        self.time += dt
        self.cycle += 1
        return dt

    def run(self, t_end: float, dt: Optional[float] = None, max_cycles: int = 100000) -> None:
        """Run simulation until reaching t_end."""
        while self.time < t_end and self.cycle < max_cycles:
            step_dt = self.compute_dt() if dt is None else dt
            step_dt = min(step_dt, t_end - self.time)
            if step_dt <= 1e-15:
                break
            self.step(step_dt)

    def get_conserved_totals(self) -> Dict[str, float]:
        """Compute globally integrated conserved quantities (bilan).

        # Ported from multi_bilan.F
        """
        vol = self.mesh.cell_volumes
        mass = float(np.sum(self.state.rho * vol))
        mom = np.sum(self.state.rho[:, None] * self.state.vel * vol[:, None], axis=0)
        e_kin = float(0.5 * np.sum(self.state.rho * np.sum(self.state.vel ** 2, axis=1) * vol))
        e_int = float(np.sum(self.state.eint * vol))
        e_tot = e_kin + e_int
        return {
            "mass": mass,
            "momentum_x": float(mom[0]),
            "momentum_y": float(mom[1]),
            "momentum_z": float(mom[2]),
            "kinetic_energy": e_kin,
            "internal_energy": e_int,
            "total_energy": e_tot,
        }
