"""
/STACK & /PLY — Composite Ply Layup Stack Model (Classical Lamination Theory).

Upstream Fortran reference:
  - $OR_SRC/starter/source/properties/composite_options/stack/lecstack_ply.F
    Subroutine LECSTACK_PLY (lines 46-367)
  - $OR_SRC/starter/source/properties/composite_options/stack/preplyxfem.F
    Subroutine PREPLYXFEM (lines 29-94)
  - $OR_SRC/starter/source/stack/hm_read_stack.F
    Subroutine HM_READ_STACK (lines 44-595)
  - $OR_SRC/starter/source/elements/shell/coque/lcgeo19.F
    Subroutine LCGEO19 (lines 37-181)

Theory & Formulation
--------------------
A composite laminate is represented by a /STACK containing a series of /PLY
layers stacked through the shell thickness h from bottom to top (z in [-h/2, +h/2]).

1. Ply Definition (/PLY, lcgeo19.F):
   Each ply has:
   - Ply ID and Material ID (typically orthotropic elastic LAW25, LAW58, LAW14, etc.)
   - Thickness t_p > 0
   - Fiber orientation angle theta_p (in degrees relative to shell reference coordinate axis)
   - Number of through-thickness integration points N_int (default 1 or 2)
   - Angle between orthotropic directions alpha_i (default 90 deg)

2. Stack Geometry (/STACK, hm_read_stack.F, lecstack_ply.F):
   - Reference surface position IPOS:
       IPOS = 0: Mid-surface reference (ZSHIFT = -0.5), z in [-h/2, +h/2] (default)
       IPOS = 3: Top surface reference (ZSHIFT = -1.0), z in [-h, 0]
       IPOS = 4: Bottom surface reference (ZSHIFT = 0.0), z in [0, +h]
       IPOS = 2: Custom offset ZSHIFT in [-1, +1]
   - Total thickness:
       h = sum(t_p)
   - Layer boundaries:
       z_bottom = ZSHIFT * h
       z_p = z_bottom + sum_{k=1}^{p-1} t_k
       z_{p+1} = z_p + t_p

3. Classical Lamination Theory (CLT) ABD Matrices:
   In plane stress, the orthotropic reduced stiffness matrix in principal material
   directions (1 = fiber, 2 = transverse) is:
       Q11 = E1 / (1 - nu12 * nu21)
       Q22 = E2 / (1 - nu12 * nu21)
       Q12 = nu12 * E2 / (1 - nu12 * nu21)
       Q66 = G12
   where nu21 = nu12 * E2 / E1.

   For orientation angle theta_p:
       m = cos(theta_p), n = sin(theta_p)
       Qbar_11 = Q11*m^4 + 2*(Q12 + 2*Q66)*m^2*n^2 + Q22*n^4
       Qbar_22 = Q11*n^4 + 2*(Q12 + 2*Q66)*m^2*n^2 + Q22*m^4
       Qbar_12 = (Q11 + Q22 - 4*Q66)*m^2*n^2 + Q12*(m^4 + n^4)
       Qbar_16 = (Q11 - Q12 - 2*Q66)*m^3*n + (Q12 - Q22 + 2*Q66)*m*n^3
       Qbar_26 = (Q11 - Q12 - 2*Q66)*m*n^3 + (Q12 - Q22 + 2*Q66)*m^3*n
       Qbar_66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m^2*n^2 + Q66*(m^4 + n^4)

   The laminate constitutive equations relate in-plane force resultants N and moment
   resultants M to mid-surface strains eps^0 and curvatures kappa:
       [ N ]   [ A   B ] [ eps^0 ]
       [   ] = [       ] [       ]
       [ M ]   [ B   D ] [ kappa ]

   Where:
       A_ij = sum_p Qbar_ij^p * (z_{p+1} - z_p)           (Extensional stiffness)
       B_ij = 0.5 * sum_p Qbar_ij^p * (z_{p+1}^2 - z_p^2) (Bending-extension coupling)
       D_ij = (1/3) * sum_p Qbar_ij^p * (z_{p+1}^3 - z_p^3)(Bending stiffness)

4. Properties of the ABD Matrices:
   - For symmetric laminates (symmetric layup about mid-surface), B_ij = 0 identically.
   - For balanced laminates (equal +theta and -theta plies), A16 = A26 = 0.
   - For cross-ply laminates (0/90 plies only), A16 = A26 = B16 = B26 = D16 = D26 = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================================
# Classical Lamination Theory (CLT) Core Physics Functions
# ============================================================================

def reduced_stiffness_matrix(
    e1: float,
    e2: float,
    nu12: float,
    g12: float,
) -> np.ndarray:
    """Compute the 3x3 reduced plane-stress stiffness matrix Q in principal material axes.

    Voigt order: [11, 22, 12] corresponding to [sigma_1, sigma_2, tau_12].

    Parameters
    ----------
    e1 : float
        Longitudinal Young's modulus (fiber direction).
    e2 : float
        Transverse Young's modulus.
    nu12 : float
        Major Poisson's ratio (-eps_2 / eps_1 for uniaxial tension in 1).
    g12 : float
        In-plane shear modulus.

    Returns
    -------
    np.ndarray of shape (3, 3)
        Reduced stiffness matrix Q.
    """
    if e1 <= 0.0 or e2 <= 0.0 or g12 <= 0.0:
        raise ValueError(f"Elastic moduli must be positive: E1={e1}, E2={e2}, G12={g12}")

    nu21 = nu12 * e2 / e1
    denom = 1.0 - nu12 * nu21
    if denom <= 0.0:
        raise ValueError(f"Thermodynamic instability: 1 - nu12*nu21 = {denom} <= 0")

    q11 = e1 / denom
    q22 = e2 / denom
    q12 = nu12 * e2 / denom
    q66 = float(g12)

    return np.array([
        [q11, q12, 0.0],
        [q12, q22, 0.0],
        [0.0, 0.0, q66],
    ], dtype=np.float64)


def rotate_reduced_stiffness(q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Transform reduced stiffness matrix Q by orientation angle theta (degrees).

    Voigt stress: [sigma_x, sigma_y, tau_xy].
    Voigt engineering strain: [eps_x, eps_y, gamma_xy] where gamma_xy = 2 * eps_xy.

    Parameters
    ----------
    q : np.ndarray of shape (3, 3)
        Principal reduced stiffness matrix.
    theta_deg : float
        Fiber orientation angle in degrees relative to laminate x-axis.

    Returns
    -------
    np.ndarray of shape (3, 3)
        Transformed reduced stiffness matrix Q_bar.
    """
    theta_rad = math.radians(theta_deg)
    m = math.cos(theta_rad)
    n = math.sin(theta_rad)

    m2 = m * m
    n2 = n * n
    m4 = m2 * m2
    n4 = n2 * n2
    m2n2 = m2 * n2
    mn = m * n

    q11 = q[0, 0]
    q12 = q[0, 1]
    q22 = q[1, 1]
    q66 = q[2, 2]

    qbar11 = q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2n2 + q22 * n4
    qbar22 = q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2n2 + q22 * m4
    qbar12 = (q11 + q22 - 4.0 * q66) * m2n2 + q12 * (m4 + n4)
    qbar16 = (q11 - q12 - 2.0 * q66) * (m2 * mn) + (q12 - q22 + 2.0 * q66) * (n2 * mn)
    qbar26 = (q11 - q12 - 2.0 * q66) * (n2 * mn) + (q12 - q22 + 2.0 * q66) * (m2 * mn)
    qbar66 = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2n2 + q66 * (m4 + n4)

    return np.array([
        [qbar11, qbar12, qbar16],
        [qbar12, qbar22, qbar26],
        [qbar16, qbar26, qbar66],
    ], dtype=np.float64)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PlyDefinition:
    """Definition of an individual composite ply layer (/PLY).

    Upstream Fortran reference:
      $OR_SRC/starter/source/properties/composite_options/stack/lecstack_ply.F
      and lcgeo19.F (lines 79-92).

    Attributes
    ----------
    id : int
        Ply identification number.
    mat_id : int
        Material ID (/MAT) assigned to this ply.
    thickness : float
        Ply physical thickness t_ply > 0.
    angle : float
        Ply orientation angle in degrees (delta_phi in OpenRadioss).
    n_int : int, default 1
        Number of through-thickness integration points for this ply (Npt_ply).
    title : str, default ""
        Ply descriptive title.
    weight : float, default 1.0
        Volume fraction or weighting factor (F_weight_i).
    p_thick_fail : float, default 1.0
        Thickness failure criterion (P_thick_fail_lam).
    drape_id : int, default 0
        Draping identification number.
    alpha : float, default 90.0
        Angle between orthotropy directions (alpha_i, default 90 degrees).
    """
    id: int = 1
    mat_id: int = 1
    thickness: float = 1.0
    angle: float = 0.0
    n_int: int = 1
    title: str = ""
    weight: float = 1.0
    p_thick_fail: float = 1.0
    drape_id: int = 0
    alpha: float = 90.0

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError(f"Ply thickness must be strictly positive, got {self.thickness}")
        if self.n_int < 1:
            self.n_int = 1

    @property
    def t_ply(self) -> float:
        """Alias for thickness."""
        return self.thickness

    @property
    def theta_ply(self) -> float:
        """Alias for angle in degrees."""
        return self.angle

    @property
    def theta(self) -> float:
        """Alias for angle in degrees."""
        return self.angle


@dataclass
class StackDefinition:
    """Composite ply layup stack (/STACK, /PROP/TYPE52, /PROP/PCOMPP).

    Upstream Fortran reference:
      $OR_SRC/starter/source/properties/composite_options/stack/lecstack_ply.F
      and hm_read_stack.F (lines 44-595).

    Attributes
    ----------
    id : int
        Stack property ID.
    title : str
        Stack descriptive title.
    plies : List[PlyDefinition]
        Ordered list of plies from bottom surface to top surface.
    ipos : int, default 0
        Reference surface position:
        - 0: Mid-surface reference (z in [-h/2, +h/2], ZSHIFT = -0.5)
        - 3: Top surface reference (z in [-h, 0], ZSHIFT = -1.0)
        - 4: Bottom surface reference (z in [0, +h], ZSHIFT = 0.0)
        - 2: Custom offset defined by z_shift
    z_shift : Optional[float], optional
        Custom offset factor ZSHIFT = z0 / h. If None, derived from ipos.
    shfsr : float, default 5.0 / 6.0
        Transverse shear correction factor (FSHEAR / SHFSR).
    """
    id: int = 1
    title: str = "STACK"
    plies: List[PlyDefinition] = field(default_factory=list)
    ipos: int = 0
    z_shift: Optional[float] = None
    shfsr: float = 5.0 / 6.0

    def __post_init__(self) -> None:
        if self.z_shift is None:
            if self.ipos == 0:
                self.z_shift = -0.5
            elif self.ipos == 3:
                self.z_shift = -1.0
            elif self.ipos == 4:
                self.z_shift = 0.0
            else:
                self.z_shift = -0.5

    # ------------------------------------------------------------------------
    # Geometric Properties
    # ------------------------------------------------------------------------

    @property
    def total_thickness(self) -> float:
        """Total laminate thickness h = sum(t_ply)."""
        return float(sum(p.thickness for p in self.plies))

    @property
    def h(self) -> float:
        """Alias for total_thickness."""
        return self.total_thickness

    @property
    def num_plies(self) -> int:
        """Total number of plies in the stack."""
        return len(self.plies)

    @property
    def num_integration_points(self) -> int:
        """Total number of through-thickness integration points."""
        return sum(p.n_int for p in self.plies)

    @property
    def ply_interfaces(self) -> np.ndarray:
        """Through-thickness z-coordinates of ply interfaces (bottom to top).

        Shape: (num_plies + 1,).
        For mid-surface reference (ipos=0), interfaces span [-h/2, +h/2].
        """
        h_tot = self.total_thickness
        z0 = (self.z_shift if self.z_shift is not None else -0.5) * h_tot

        z_coords = [z0]
        curr_z = z0
        for p in self.plies:
            curr_z += p.thickness
            z_coords.append(curr_z)

        return np.array(z_coords, dtype=np.float64)

    # ------------------------------------------------------------------------
    # Integration Point Distribution
    # ------------------------------------------------------------------------

    def integration_points(self, rule: str = "gauss") -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Compute through-thickness positions z_k and weights w_k across all plies.

        Parameters
        ----------
        rule : str, default "gauss"
            Quadrature rule per ply:
            - "gauss" / "legendre": Gauss-Legendre quadrature (default in FE shell theory)
            - "lobatto": Gauss-Lobatto quadrature (includes ply boundary points)
            - "simpson": Simpson rule (applicable for n_int >= 2)
            - "uniform": Uniformly spaced points across each ply

        Returns
        -------
        z_pts : np.ndarray of shape (N_int_total,)
            Through-thickness coordinate z_k for each integration point.
        weights : np.ndarray of shape (N_int_total,)
            Quadrature weight w_k associated with each integration point (sum(weights) == h).
        ply_indices : list of int of length N_int_total
            0-based index of the ply containing each integration point.
        """
        z_pts: List[float] = []
        weights: List[float] = []
        ply_indices: List[int] = []

        interfaces = self.ply_interfaces
        rule_lower = rule.lower()

        for idx, ply in enumerate(self.plies):
            z_bot = float(interfaces[idx])
            z_top = float(interfaces[idx + 1])
            t_p = ply.thickness
            n_p = ply.n_int

            if n_p == 1:
                # Single mid-point integration
                z_pts.append(0.5 * (z_bot + z_top))
                weights.append(t_p)
                ply_indices.append(idx)
            elif rule_lower in ("gauss", "legendre"):
                # Standard Gauss-Legendre points on [-1, 1]
                xi, w = np.polynomial.legendre.leggauss(n_p)
                for xi_k, w_k in zip(xi, w):
                    # Map [-1, 1] -> [z_bot, z_top]
                    z_k = 0.5 * (z_top + z_bot) + 0.5 * t_p * xi_k
                    wt_k = 0.5 * t_p * w_k
                    z_pts.append(float(z_k))
                    weights.append(float(wt_k))
                    ply_indices.append(idx)
            elif rule_lower in ("lobatto", "gauss-lobatto"):
                # Gauss-Lobatto points on [-1, 1]
                if n_p == 2:
                    xi = [-1.0, 1.0]
                    w = [1.0, 1.0]
                elif n_p == 3:
                    xi = [-1.0, 0.0, 1.0]
                    w = [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
                elif n_p == 4:
                    sq5 = math.sqrt(5.0)
                    xi = [-1.0, -1.0 / sq5, 1.0 / sq5, 1.0]
                    w = [1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0]
                else:
                    xi = np.linspace(-1.0, 1.0, n_p)
                    w = np.full(n_p, 2.0 / n_p)
                for xi_k, w_k in zip(xi, w):
                    z_k = 0.5 * (z_top + z_bot) + 0.5 * t_p * xi_k
                    wt_k = 0.5 * t_p * w_k
                    z_pts.append(float(z_k))
                    weights.append(float(wt_k))
                    ply_indices.append(idx)
            elif rule_lower == "simpson":
                xi = np.linspace(-1.0, 1.0, n_p)
                if n_p == 2:
                    w = [1.0, 1.0]
                elif n_p == 3:
                    w = [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
                elif n_p == 4:
                    w = [3.0 / 8.0, 9.0 / 8.0, 9.0 / 8.0, 3.0 / 8.0]
                else:
                    w = np.ones(n_p)
                    w /= np.sum(w) / 2.0
                for xi_k, w_k in zip(xi, w):
                    z_k = 0.5 * (z_top + z_bot) + 0.5 * t_p * xi_k
                    wt_k = 0.5 * t_p * w_k
                    z_pts.append(float(z_k))
                    weights.append(float(wt_k))
                    ply_indices.append(idx)
            else:  # uniform
                xi = np.linspace(-1.0 + 1.0 / n_p, 1.0 - 1.0 / n_p, n_p)
                w_uniform = t_p / n_p
                for xi_k in xi:
                    z_k = 0.5 * (z_top + z_bot) + 0.5 * t_p * xi_k
                    z_pts.append(float(z_k))
                    weights.append(w_uniform)
                    ply_indices.append(idx)

        return np.array(z_pts, dtype=np.float64), np.array(weights, dtype=np.float64), ply_indices

    # ------------------------------------------------------------------------
    # Classical Lamination Theory (CLT) ABD Matrix Computation
    # ------------------------------------------------------------------------

    def compute_abd(
        self,
        materials: Union[
            Dict[int, Any],
            Tuple[float, float, float, float],
            np.ndarray,
        ],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Classical Lamination Theory (CLT) ABD stiffness matrices.

        Equations (lecstack_ply.F, hm_read_stack.F):
            A_ij = sum_p C_ij^p * (z_{p+1} - z_p)
            B_ij = 0.5 * sum_p C_ij^p * (z_{p+1}^2 - z_p^2)
            D_ij = (1/3) * sum_p C_ij^p * (z_{p+1}^3 - z_p^3)
        where C_ij^p is the rotated plane-stress reduced stiffness matrix Q_bar^p.

        Parameters
        ----------
        materials : dict, tuple, or np.ndarray
            Material properties mapping:
            - If dict: maps mat_id -> (E1, E2, nu12, G12) or dict with 'E1', 'E2', 'nu12', 'G12'
            - If tuple: single material (E1, E2, nu12, G12) shared by all plies
            - If np.ndarray of shape (3, 3): single reduced stiffness matrix Q shared by all plies

        Returns
        -------
        A : np.ndarray of shape (3, 3)
            Extensional stiffness matrix (N/m).
        B : np.ndarray of shape (3, 3)
            Extension-bending coupling matrix (N). (Identically zero for symmetric layups)
        D : np.ndarray of shape (3, 3)
            Bending stiffness matrix (N*m).
        """
        if len(self.plies) == 0:
            return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))

        interfaces = self.ply_interfaces

        a_mat = np.zeros((3, 3), dtype=np.float64)
        b_mat = np.zeros((3, 3), dtype=np.float64)
        d_mat = np.zeros((3, 3), dtype=np.float64)

        for p_idx, ply in enumerate(self.plies):
            z_p = interfaces[p_idx]
            z_p1 = interfaces[p_idx + 1]

            # Resolve reduced stiffness Q for this ply
            if isinstance(materials, dict):
                mat_def = materials.get(ply.mat_id)
                if mat_def is None:
                    # Try matching by index or fallback to first available
                    mat_def = next(iter(materials.values()))
                if isinstance(mat_def, (tuple, list)):
                    e1, e2, nu12, g12 = mat_def[:4]
                    q = reduced_stiffness_matrix(e1, e2, nu12, g12)
                elif hasattr(mat_def, "E1") and hasattr(mat_def, "E2"):
                    q = reduced_stiffness_matrix(mat_def.E1, mat_def.E2, mat_def.nu12, mat_def.G12)
                elif isinstance(mat_def, dict):
                    e1 = mat_def.get("E1", mat_def.get("e1", 1.0))
                    e2 = mat_def.get("E2", mat_def.get("e2", e1))
                    nu12 = mat_def.get("nu12", mat_def.get("NU12", 0.3))
                    g12 = mat_def.get("G12", mat_def.get("g12", 0.5 * e1 / (1.0 + nu12)))
                    q = reduced_stiffness_matrix(e1, e2, nu12, g12)
                elif isinstance(mat_def, np.ndarray) and mat_def.shape == (3, 3):
                    q = mat_def
                else:
                    raise TypeError(f"Unrecognized material format for mat_id={ply.mat_id}: {mat_def}")
            elif isinstance(materials, (tuple, list)):
                e1, e2, nu12, g12 = materials[:4]
                q = reduced_stiffness_matrix(e1, e2, nu12, g12)
            elif isinstance(materials, np.ndarray) and materials.shape == (3, 3):
                q = materials
            else:
                raise TypeError(f"Unsupported materials input: {type(materials)}")

            # Rotate Q to ply angle
            q_bar = rotate_reduced_stiffness(q, ply.angle)

            # Accumulate layer contributions
            dz1 = z_p1 - z_p
            dz2 = z_p1 * z_p1 - z_p * z_p
            dz3 = z_p1**3 - z_p**3

            a_mat += q_bar * dz1
            b_mat += q_bar * (0.5 * dz2)
            d_mat += q_bar * ((1.0 / 3.0) * dz3)

        return a_mat, b_mat, d_mat

    def abd_matrix(self, materials: Any) -> np.ndarray:
        """Compute the full 6x6 laminate stiffness matrix [[A, B], [B, D]]."""
        a, b, d = self.compute_abd(materials)
        return np.block([
            [a, b],
            [b, d],
        ])

    # ------------------------------------------------------------------------
    # Layup Classification & Engineering Constants
    # ------------------------------------------------------------------------

    def is_symmetric(self, tol: float = 1.0e-6) -> bool:
        """Check if the laminate layup is geometrically and materially symmetric.

        A laminate is symmetric about mid-surface if for each ply at distance +z,
        there is an identical ply with same thickness, material, and orientation at -z.
        """
        n = len(self.plies)
        if n == 0:
            return True
        for i in range(n // 2):
            p_bot = self.plies[i]
            p_top = self.plies[n - 1 - i]
            if abs(p_bot.thickness - p_top.thickness) > tol:
                return False
            if p_bot.mat_id != p_top.mat_id:
                return False
            if abs(p_bot.angle - p_top.angle) > tol:
                return False
        return True

    def is_balanced(self, tol: float = 1.0e-6) -> bool:
        """Check if the laminate layup is balanced.

        A laminate is balanced if every off-axis +theta ply is balanced by an identical
        -theta ply of the same material and thickness. (Ensures A16 = A26 = 0).
        """
        # Collect non-zero / non-90 plies
        pairs: List[Tuple[float, float, int]] = []  # (angle, thickness, mat_id)
        for p in self.plies:
            ang_mod = (p.angle % 180.0)
            if abs(ang_mod) < tol or abs(ang_mod - 90.0) < tol or abs(ang_mod - 180.0) < tol:
                continue
            pairs.append((ang_mod, p.thickness, p.mat_id))

        matched = [False] * len(pairs)
        for i, (ang_i, t_i, mat_i) in enumerate(pairs):
            if matched[i]:
                continue
            # Search for matching -angle
            target_ang = (180.0 - ang_i) % 180.0
            found = False
            for j in range(i + 1, len(pairs)):
                if not matched[j]:
                    ang_j, t_j, mat_j = pairs[j]
                    if (abs(ang_j - target_ang) < tol and abs(t_i - t_j) < tol and mat_i == mat_j):
                        matched[i] = True
                        matched[j] = True
                        found = True
                        break
            if not found:
                return False

        return True

    def effective_engineering_constants(self, materials: Any) -> Dict[str, float]:
        """Compute effective in-plane engineering constants Ex, Ey, Gxy, nu_xy, nu_yx.

        Derived from the inverted in-plane compliance matrix [a] = [A]^-1 * h.

        Returns
        -------
        dict with keys: 'Ex', 'Ey', 'Gxy', 'nu_xy', 'nu_yx', 'h'
        """
        a_mat, _, _ = self.compute_abd(materials)
        h_tot = self.total_thickness
        if h_tot <= 0.0:
            return {"Ex": 0.0, "Ey": 0.0, "Gxy": 0.0, "nu_xy": 0.0, "nu_yx": 0.0, "h": 0.0}

        a_inv = np.linalg.inv(a_mat)
        ex = 1.0 / (h_tot * a_inv[0, 0])
        ey = 1.0 / (h_tot * a_inv[1, 1])
        gxy = 1.0 / (h_tot * a_inv[2, 2])
        nu_xy = -a_inv[0, 1] / a_inv[0, 0]
        nu_yx = -a_inv[0, 1] / a_inv[1, 1]

        return {
            "Ex": float(ex),
            "Ey": float(ey),
            "Gxy": float(gxy),
            "nu_xy": float(nu_xy),
            "nu_yx": float(nu_yx),
            "h": float(h_tot),
        }

    def __repr__(self) -> str:
        return (
            f"StackDefinition(id={self.id}, title={self.title!r}, "
            f"num_plies={len(self.plies)}, total_thickness={self.total_thickness:.4f})"
        )


# ============================================================================
# Factory / Helper Functions
# ============================================================================

def build_stack(
    id: int = 1,
    title: str = "STACK",
    plies: Optional[Sequence[Union[PlyDefinition, Tuple[Any, ...], Dict[str, Any]]]] = None,
    ipos: int = 0,
    z_shift: Optional[float] = None,
    shfsr: float = 5.0 / 6.0,
    **kwargs: Any,
) -> StackDefinition:
    """Convenience builder function for StackDefinition.

    Supports initializing plies from:
    - List of PlyDefinition objects
    - List of tuples: (ply_id, mat_id, thickness, angle, [n_int])
    - List of dicts with matching keys
    """
    ply_list: List[PlyDefinition] = []
    if plies is not None:
        for idx, p in enumerate(plies):
            if isinstance(p, PlyDefinition):
                ply_list.append(p)
            elif isinstance(p, (tuple, list)):
                p_id = int(p[0]) if len(p) > 0 else (idx + 1)
                mat_id = int(p[1]) if len(p) > 1 else 1
                thick = float(p[2]) if len(p) > 2 else 1.0
                ang = float(p[3]) if len(p) > 3 else 0.0
                n_int = int(p[4]) if len(p) > 4 else 1
                ply_list.append(PlyDefinition(
                    id=p_id, mat_id=mat_id, thickness=thick, angle=ang, n_int=n_int
                ))
            elif isinstance(p, dict):
                p_id = int(p.get("id", p.get("ply_id", idx + 1)))
                mat_id = int(p.get("mat_id", p.get("mid", 1)))
                thick = float(p.get("thickness", p.get("t", p.get("t_ply", 1.0))))
                ang = float(p.get("angle", p.get("delta_phi", p.get("theta", 0.0))))
                n_int = int(p.get("n_int", p.get("Npt_ply", 1)))
                ply_list.append(PlyDefinition(
                    id=p_id, mat_id=mat_id, thickness=thick, angle=ang, n_int=n_int
                ))

    return StackDefinition(
        id=int(id),
        title=str(title),
        plies=ply_list,
        ipos=int(ipos),
        z_shift=z_shift,
        shfsr=float(shfsr),
    )
