"""Composite fiber draping angles and thickness variation (/DRAPE).

Upstream OpenRadioss Fortran References:
----------------------------------------
- Starter Card Readers & Modules:
  ``$OR_SRC/starter/source/properties/composite_options/drape/hm_read_drape.F``
  ``$OR_SRC/starter/source/properties/composite_options/drape/shellthk_upd.F``
  ``$OR_SRC/starter/share/modules1/drape_mod.F``
  ``$OR_SRC/hm_cfg_files/config/CFG/radioss2017/TABLE/drape.cfg``
  ``$OR_SRC/hm_cfg_files/config/CFG/radioss2022/TABLE/drape.cfg``
  ``$OR_SRC/hm_cfg_files/config/CFG/radioss2022/TABLE/drape_ply_slice.cfg``

Physics & Formulation:
----------------------
In composite manufacturing (resin transfer molding, thermoforming, tape layup),
flat prepreg fabrics or dry preforms undergo in-plane trellising shear and distortion
when draped over complex 3D tool surfaces:

1. Fiber Orientation Draping:
   - Draped fiber angles:
     alpha_1: local warp / primary fiber direction
     alpha_2: local weft / secondary fiber direction
   - Angle change (rotation):
     delta_alpha = alpha_1 - alpha_1_nominal
   - Local orthotropy frame adjustment:
     phi_effective = phi_nominal + delta_alpha

2. Trellising Shear Angle:
   - In the undeformed flat fabric, the fiber axes are orthogonal (alpha_1 - alpha_2 = 90 deg).
   - Draping trellising in-plane shear angle:
     theta_shear = |alpha_1 - alpha_2 - 90 deg|
   - Under pin-joint net kinematic compaction (volume conservation):
     A_sheared = A_0 * cos(theta_shear)
     t_effective = t_nominal / cos(theta_shear)
   - Alternatively, prescribed thinning factor tau or thickness t_drape:
     t_effective = t_nominal * tau
     or
     t_effective = t_nominal * (1.0 + delta_t / t_nominal)

3. Composite Shell & Ply Stack Updates:
   - Updates shell element thicknesses and composite layer angles matching `shellthk_upd.F`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class ElementDrape:
    """Draping parameters for a single shell element.

    Parameters
    ----------
    elem_id : int
        Shell element identifier.
    part_id : int
        Part or property identifier.
    alpha_1 : float
        Draped warp / primary fiber angle (default 0.0 degrees).
    alpha_2 : float
        Draped weft / secondary fiber angle (default 90.0 degrees).
    delta_alpha : float
        Explicit fiber orientation angle change (degrees).
    thickness : Optional[float]
        Explicit draped element thickness (t_drape).
    thinning_factor : Optional[float]
        Thinning ratio tau = t / t_nominal.
    angle_unit : str
        "deg" or "rad" (default "deg").
    """

    elem_id: int
    part_id: int = 0
    alpha_1: float = 0.0
    alpha_2: float = 90.0
    delta_alpha: float = 0.0
    thickness: Optional[float] = None
    thinning_factor: Optional[float] = None
    angle_unit: str = "deg"

    @property
    def shear_angle_deg(self) -> float:
        """In-plane trellising shear angle theta_shear = |alpha_1 - alpha_2 - 90| in degrees."""
        if self.angle_unit == "rad":
            a1 = math.degrees(self.alpha_1)
            a2 = math.degrees(self.alpha_2)
        else:
            a1 = float(self.alpha_1)
            a2 = float(self.alpha_2)

        # Difference from orthogonal 90 degrees
        diff = abs(abs(a1 - a2) - 90.0)
        return float(diff)

    @property
    def shear_angle_rad(self) -> float:
        """In-plane trellising shear angle in radians."""
        return math.radians(self.shear_angle_deg)

    @property
    def trellising_thinning_factor(self) -> float:
        """Kinematic pin-joint net thickness factor 1 / cos(theta_shear)."""
        th_rad = self.shear_angle_rad
        cos_th = math.cos(th_rad)
        if abs(cos_th) < 1e-6:
            return 10.0  # Cap extreme shear locking
        return float(1.0 / cos_th)

    def get_delta_alpha_deg(self) -> float:
        """Return fiber angle change in degrees."""
        if self.delta_alpha != 0.0:
            if self.angle_unit == "rad":
                return math.degrees(self.delta_alpha)
            return float(self.delta_alpha)
        # Derived from alpha_1 relative to 0
        if self.angle_unit == "rad":
            return math.degrees(self.alpha_1)
        return float(self.alpha_1)


@dataclass
class DrapeParams:
    """Parameters and lookup table for /DRAPE composite draping.

    Upstream Fortran origin:
      - ``starter/source/properties/composite_options/drape/hm_read_drape.F``
      - ``starter/source/properties/composite_options/drape/shellthk_upd.F``
    """

    id: int = 1
    title: str = ""
    part_id: int = 0
    ply_id: int = 0
    entries: Dict[int, ElementDrape] = field(default_factory=dict)

    def add_entry(
        self,
        elem_id: int,
        alpha_1: float = 0.0,
        alpha_2: float = 90.0,
        delta_alpha: float = 0.0,
        thickness: Optional[float] = None,
        thinning_factor: Optional[float] = None,
        angle_unit: str = "deg",
        part_id: int = 0,
    ) -> ElementDrape:
        """Add or update an element draping entry."""
        pid = part_id if part_id > 0 else self.part_id
        entry = ElementDrape(
            elem_id=elem_id,
            part_id=pid,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            delta_alpha=delta_alpha,
            thickness=thickness,
            thinning_factor=thinning_factor,
            angle_unit=angle_unit,
        )
        self.entries[elem_id] = entry
        return entry

    def get_drape(self, elem_id: int) -> Optional[ElementDrape]:
        """Retrieve draping entry for an element."""
        return self.entries.get(elem_id, None)

    def compute_effective_thickness(
        self,
        elem_id: int,
        t_nominal: float,
    ) -> float:
        """Compute effective draped thickness for element elem_id.

        Priority order:
        1. Explicit thickness if given
        2. Explicit thinning factor: t_eff = t_nominal * thinning_factor
        3. Trellising shear model: t_eff = t_nominal / cos(theta_shear)
        4. Nominal thickness if no drape entry
        """
        entry = self.entries.get(elem_id, None)
        if entry is None:
            return float(t_nominal)

        if entry.thickness is not None and entry.thickness > 0.0:
            return float(entry.thickness)

        if entry.thinning_factor is not None and entry.thinning_factor > 0.0:
            return float(t_nominal * entry.thinning_factor)

        # Trellising shear thinning/thickening: t = t_nom / cos(theta_shear)
        return float(t_nominal * entry.trellising_thinning_factor)

    def compute_effective_angle(
        self,
        elem_id: int,
        phi_nominal_deg: float,
    ) -> float:
        """Compute effective fiber angle: phi_eff = phi_nominal + delta_alpha."""
        entry = self.entries.get(elem_id, None)
        if entry is None:
            return float(phi_nominal_deg)
        return float(phi_nominal_deg + entry.get_delta_alpha_deg())


# Alias for load/table naming convention
DrapeTable = DrapeParams


def compute_drape_shear_angle(
    alpha_1: float,
    alpha_2: float,
    angle_unit: str = "deg",
) -> float:
    """Calculate trellising shear angle theta_shear = |alpha_1 - alpha_2 - 90 deg|.

    Returns
    -------
    float : shear angle in degrees.
    """
    if angle_unit == "rad":
        a1 = math.degrees(alpha_1)
        a2 = math.degrees(alpha_2)
    else:
        a1 = float(alpha_1)
        a2 = float(alpha_2)
    return float(abs(abs(a1 - a2) - 90.0))


def compute_trellising_thickness(
    t_nominal: float,
    theta_shear: float,
    angle_unit: str = "deg",
) -> float:
    """Calculate draped thickness under trellising shear: t = t_nominal / cos(theta_shear)."""
    th_rad = math.radians(theta_shear) if angle_unit == "deg" else float(theta_shear)
    cos_th = math.cos(th_rad)
    if abs(cos_th) < 1e-6:
        return float(t_nominal * 10.0)
    return float(t_nominal / cos_th)


def apply_draping_to_ply(
    ply: Any,
    drape: DrapeParams,
    elem_id: int,
) -> Any:
    """Update a composite Ply entity using element draping data.

    Updates:
      - `ply.thick`: effective draped thickness
      - `ply.orientangle`: effective fiber angle phi_effective = phi + delta_alpha
    """
    nominal_thick = getattr(ply, "thick", 1.0)
    nominal_angle = getattr(ply, "orientangle", 0.0)

    eff_thick = drape.compute_effective_thickness(elem_id, nominal_thick)
    eff_angle = drape.compute_effective_angle(elem_id, nominal_angle)

    # Return updated or cloned object
    if hasattr(ply, "__dict__"):
        try:
            ply.thick = eff_thick
            ply.orientangle = eff_angle
            return ply
        except Exception:
            pass

    return {
        "ply_id": getattr(ply, "id", getattr(ply, "ply_id", 1)),
        "thick": eff_thick,
        "orientangle": eff_angle,
    }


def apply_draping_to_shell_mesh(
    elements: Union[Dict[int, Any], Sequence[Any]],
    drape: DrapeParams,
    thickness_map: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Batch update shell element thicknesses matching `shellthk_upd.F`.

    Parameters
    ----------
    elements : dict of {elem_id: elem} or list of elements
    drape : DrapeParams
        Draping table containing per-element draping data.
    thickness_map : dict, optional
        Optional dictionary of initial element thicknesses.

    Returns
    -------
    Dict[int, float] : map of {elem_id: updated_effective_thickness}.
    """
    updated_thks: Dict[int, float] = {}

    if isinstance(elements, dict):
        elem_iter = elements.items()
    else:
        elem_iter = [(getattr(e, "id", idx + 1), e) for idx, e in enumerate(elements)]

    for eid, elem in elem_iter:
        if thickness_map and eid in thickness_map:
            t_nom = float(thickness_map[eid])
        elif hasattr(elem, "thick"):
            t_nom = float(getattr(elem, "thick", 1.0))
        elif hasattr(elem, "thickness"):
            t_nom = float(getattr(elem, "thickness", 1.0))
        elif isinstance(elem, dict):
            t_nom = float(elem.get("thick", elem.get("thickness", 1.0)))
        else:
            t_nom = 1.0

        t_eff = drape.compute_effective_thickness(eid, t_nom)
        updated_thks[eid] = t_eff

        # Apply in-place if element has attributes
        if hasattr(elem, "thick"):
            try:
                elem.thick = t_eff
            except Exception:
                pass
        elif isinstance(elem, dict) and "thick" in elem:
            elem["thick"] = t_eff

    return updated_thks
