"""
/PROP/TYPE5 & /PROP/RIVET — Fastener / Rivet Connection Property.

Upstream Fortran reference:
  - $OR_SRC/starter/source/properties/rivet/hm_read_prop05.F
    Subroutine HM_READ_PROP05 (lines 35-135)
  - $OR_SRC/engine/source/elements/rivet/rivet1.F
    Subroutine RIVET1 (lines 28-240)

Theory & Formulation
--------------------
A rivet connects two structural nodes (N1, N2) with specified tensile, shear,
and elongation failure criteria.

1. Parameter Card Layout (hm_read_prop05.F lines 83-115):
   - FN (NFORCE) : Maximum normal / tensile failure force
   - FT (TFORCE) : Maximum shear / tangential failure force
   - DX (LENGTH) : Characteristic maximum rivet length before elongation rupture
   - WFLAG (IROT): Rotational coupling flag:
       0 = Translation only (degrees of freedom 1, 2, 3)
       1 = Translation + Rotation (degrees of freedom 1..6)
   - IMODE       : Formulation & failure mode:
       1 = Rigid Body formulation with quadratic interaction failure (default)
       2 = Rigid Link formulation

2. Internal Storage in OpenRadioss GEO Array (hm_read_prop05.F lines 110-114):
       GEO(1) = FN^2
       GEO(2) = FT^2
       GEO(3) = DX^2
       GEO(4) = WFLAG + 0.1
       GEO(5) = IMODE + 0.1

3. Quadratic Failure Interaction Criterion (rivet1.F lines 177-199):
   For normal force F_n and tangential force F_t:
       alpha = sqrt((F_n / FN)^2 + (F_t / FT)^2)
   The rivet fails / breaks when:
       (F_n / FN)^2 + (F_t / FT)^2 >= 1.0   (alpha >= 1.0)
   Or when elongation exceeds maximum length DX:
       dist = ||x2 - x1|| > DX   (for DX > 0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.messages import MessageLog
from ..model.entities import Property
from .deck_reader import Card, KeywordBlock, parse_fortran_float

_EP15 = 1.0e-15


@dataclass
class Prop5Rivet:
    """Fastener / Rivet connection property (/PROP/TYPE5, /PROP/RIVET).

    Attributes
    ----------
    id : int
        Property identification number.
    title : str
        Property descriptive title.
    fn : float
        Maximum normal tensile failure force FN (NFORCE).
    ft : float
        Maximum shear failure force FT (TFORCE).
    dx : float
        Maximum rivet length DX (LENGTH) before rupture.
    wflag : int, default 0
        Rotational coupling flag (0 = TRANS only, 1 = TRANS + ROT).
    imod : int, default 1
        Formulation and failure mode (1 = quadratic interaction, 2 = rigid link).
    mass : float, default 0.0
        Optional mass of the rivet fastener.
    stiffness : float, default 0.0
        Optional elastic stiffness.
    """
    id: int = 1
    title: str = "RIVET"
    fn: float = 0.0
    ft: float = 0.0
    dx: float = 0.0
    wflag: int = 0
    imod: int = 1
    mass: float = 0.0
    stiffness: float = 0.0

    def __post_init__(self) -> None:
        if self.imod <= 0:
            self.imod = 1
        if self.wflag not in (0, 1):
            self.wflag = 1 if self.wflag > 0 else 0

    @property
    def nforce(self) -> float:
        """Alias for fn."""
        return self.fn

    @property
    def tforce(self) -> float:
        """Alias for ft."""
        return self.ft

    @property
    def length(self) -> float:
        """Alias for dx."""
        return self.dx

    @property
    def irot(self) -> int:
        """Alias for wflag."""
        return self.wflag

    @property
    def fn_fail(self) -> float:
        """Alias for fn."""
        return self.fn

    @property
    def ft_fail(self) -> float:
        """Alias for ft."""
        return self.ft

    @property
    def fn2(self) -> float:
        """Square of normal failure force FN^2."""
        val = self.fn if self.fn > 0.0 else _EP15
        return val * val

    @property
    def ft2(self) -> float:
        """Square of shear failure force FT^2."""
        val = self.ft if self.ft > 0.0 else _EP15
        return val * val

    @property
    def dx2(self) -> float:
        """Square of maximum length DX^2."""
        val = self.dx if self.dx > 0.0 else _EP15
        return val * val

    def evaluate_failure(
        self,
        fn: float,
        ft: float,
        dist: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Evaluate the quadratic interaction failure criterion and elongation limit.

        Criterion (rivet1.F line 198):
            alpha = sqrt((fn / FN)^2 + (ft / FT)^2)
            failed = (alpha >= 1.0) or (dist > DX if DX > 0)

        Parameters
        ----------
        fn : float
            Acting normal / tensile force.
        ft : float
            Acting shear / tangential force.
        dist : float, optional
            Current distance between connected nodes.

        Returns
        -------
        (is_failed, alpha) : tuple of (bool, float)
            is_failed: True if the rivet has failed.
            alpha: failure interaction index (>= 1.0 indicates failure).
        """
        fn_cap = max(self.fn, _EP15)
        ft_cap = max(self.ft, _EP15)

        term_n = (fn / fn_cap) ** 2
        term_t = (ft / ft_cap) ** 2
        alpha = math.sqrt(term_n + term_t)

        failed = (alpha >= 1.0)

        if dist is not None and self.dx > 0.0:
            if dist > self.dx:
                failed = True

        return failed, alpha

    def evaluate_force_vector(
        self,
        f_vector: Sequence[float] | np.ndarray,
        normal_axis: Sequence[float] | np.ndarray,
        dist: Optional[float] = None,
    ) -> Tuple[bool, float, float, float]:
        """Decompose 3D force vector into normal and tangential components and evaluate failure.

        Follows rivet1.F lines 177-199:
            n_vec = normal_axis / ||normal_axis||
            F_n = F @ n_vec
            F_t_vec = F - F_n * n_vec
            F_t = ||F_t_vec||
            alpha = sqrt((F_n / FN)^2 + (F_t / FT)^2)

        Parameters
        ----------
        f_vector : array_like of shape (3,)
            Acting 3D force vector [Fx, Fy, Fz].
        normal_axis : array_like of shape (3,)
            Rivet axial orientation vector (e.g. x2 - x1).
        dist : float, optional
            Current length of the rivet.

        Returns
        -------
        (is_failed, alpha, fn, ft) : tuple of (bool, float, float, float)
        """
        f_arr = np.asarray(f_vector, dtype=float)
        n_arr = np.asarray(normal_axis, dtype=float)

        n_norm = np.linalg.norm(n_arr)
        if n_norm > _EP15:
            n_unit = n_arr / n_norm
            fn = float(np.dot(f_arr, n_unit))
            f_t_vec = f_arr - fn * n_unit
            ft = float(np.linalg.norm(f_t_vec))
        else:
            # Degenerate coincident nodes
            fn = float(np.linalg.norm(f_arr))
            ft = 0.0

        current_dist = dist if dist is not None else float(n_norm)
        is_failed, alpha = self.evaluate_failure(fn, ft, dist=current_dist)

        return is_failed, alpha, fn, ft

    def to_geo(self) -> np.ndarray:
        """Return the OpenRadioss GEO array representation.

        Matches hm_read_prop05.F lines 110-114:
            GEO(1) = FN^2
            GEO(2) = FT^2
            GEO(3) = DX^2
            GEO(4) = WFLAG + 0.1
            GEO(5) = IMODE + 0.1
        """
        pun = 0.1
        fn_val = self.fn if self.fn > 0.0 else _EP15
        ft_val = self.ft if self.ft > 0.0 else _EP15
        dx_val = self.dx if self.dx > 0.0 else _EP15

        geo = np.zeros(15, dtype=np.float64)
        geo[0] = fn_val * fn_val
        geo[1] = ft_val * ft_val
        geo[2] = dx_val * dx_val
        geo[3] = float(self.wflag) + pun
        geo[4] = float(self.imod) + pun
        return geo

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "fn": self.fn,
            "ft": self.ft,
            "dx": self.dx,
            "wflag": self.wflag,
            "imod": self.imod,
            "mass": self.mass,
            "stiffness": self.stiffness,
            "nforce": self.fn,
            "tforce": self.ft,
            "length": self.dx,
        }


# Type aliases
PropRivet = Prop5Rivet


# ============================================================================
# Card Reader & Parser
# ============================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(parse_fortran_float(str(val)))
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def parse_prop_rivet(block: KeywordBlock, log: Optional[MessageLog] = None) -> Property:
    """Parse a /PROP/TYPE5 or /PROP/RIVET card into a Property entity.

    Follows OpenRadioss hm_read_prop05.F.

    Parameters
    ----------
    block : KeywordBlock
        Deck reader keyword block.
    log : MessageLog, optional
        Message logging facility.

    Returns
    -------
    Property
        Property entity populated with named rivet parameters.
    """
    prop_id = block.user_id or 1
    title = getattr(block, "title", None) or f"PROP_RIVET_{prop_id}"

    cards = [c for c in block.cards if not c.is_blank and not c.raw.strip().startswith("#")]

    fn, ft, dx = 0.0, 0.0, 0.0
    wflag = 0
    imod = 1
    mass = 0.0
    stiffness = 0.0

    if not cards:
        if log:
            log.warning(f"/PROP/TYPE5/{prop_id}: missing data cards, using defaults", block.source)
    elif len(cards) >= 2:
        # Standard 2-card format (hm_read_prop05.F):
        # Card 1: WFLAG  IMODE
        # Card 2: NFORCE TFORCE LENGTH
        c0 = cards[0].tokens()
        c1 = cards[1].tokens()

        wflag = _safe_int(c0[0]) if len(c0) > 0 else 0
        imod = _safe_int(c0[1]) if len(c0) > 1 else 1

        fn = _safe_float(c1[0]) if len(c1) > 0 else 0.0
        ft = _safe_float(c1[1]) if len(c1) > 1 else 0.0
        dx = _safe_float(c1[2]) if len(c1) > 2 else 0.0
    else:
        # Single card formats
        c0 = cards[0].tokens()
        if len(c0) == 4:
            # Legacy 4-parameter format: MASS STIFFNESS FN FT
            mass = _safe_float(c0[0])
            stiffness = _safe_float(c0[1])
            fn = _safe_float(c0[2])
            ft = _safe_float(c0[3])
        elif len(c0) >= 5:
            # 5-parameter format: NFORCE TFORCE LENGTH WFLAG IMODE
            fn = _safe_float(c0[0])
            ft = _safe_float(c0[1])
            dx = _safe_float(c0[2])
            wflag = _safe_int(c0[3])
            imod = _safe_int(c0[4])
        elif len(c0) == 3:
            # 3-parameter format: NFORCE TFORCE LENGTH
            fn = _safe_float(c0[0])
            ft = _safe_float(c0[1])
            dx = _safe_float(c0[2])
        elif len(c0) == 2:
            # 2-parameter format: NFORCE TFORCE
            fn = _safe_float(c0[0])
            ft = _safe_float(c0[1])
        else:
            fn = _safe_float(c0[0]) if len(c0) > 0 else 0.0

    if imod <= 0:
        imod = 1

    rivet_obj = Prop5Rivet(
        id=prop_id,
        title=title,
        fn=fn,
        ft=ft,
        dx=dx,
        wflag=wflag,
        imod=imod,
        mass=mass,
        stiffness=stiffness,
    )

    params: Dict[str, Any] = {
        "fn": fn,
        "ft": ft,
        "dx": dx,
        "wflag": wflag,
        "imod": imod,
        "mass": mass,
        "stiffness": stiffness,
        "nforce": fn,
        "tforce": ft,
        "length": dx,
        "fn_fail": fn,
        "ft_fail": ft,
        "rivet": rivet_obj,
    }

    prop = Property(id=prop_id, type=5, title=title, params=params)
    prop.rivet = rivet_obj  # type: ignore[attr-defined]
    return prop


# Alias for dispatch compatibility
parse_prop_type5 = parse_prop_rivet
