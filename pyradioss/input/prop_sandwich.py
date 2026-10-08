"""Sandwich Shell Property (/PROP/TYPE11, /PROP/SH_SANDW, /PROP/SANDWICH).

Upstream OpenRadioss Fortran References:
----------------------------------------
- /PROP/TYPE11 (SH_SANDW, SANDWICH):
  ``$OR_SRC/starter/source/properties/shell/hm_read_prop11.F``
  ``$OR_SRC/hm_cfg_files/config/CFG/radioss2026/PROP/prop_p11_sh_sandw.cfg``

Mechanics Formulation of Sandwich Shells:
-----------------------------------------
1. Through-Thickness Geometry:
   A 3-layer sandwich shell comprises:
   - Layer 1 (bottom skin / face sheet 1): thickness t_1, material ID mat_1, fiber angle phi_1.
   - Layer 2 (core): thickness t_core, material ID mat_core, fiber angle phi_core.
   - Layer 3 (top skin / face sheet 2): thickness t_2, material ID mat_2, fiber angle phi_2.

   Total sandwich thickness:
       h = t_1 + t_core + t_2

   Distance between face-sheet centroids:
       d = t_core + (t_1 + t_2) / 2

   Through-thickness z-coordinate bounds relative to the mid-surface (z = 0):
   - Skin 1: z in [-h/2, -h/2 + t_1], with centroid z_c1 = -h/2 + t_1 / 2
   - Core:   z in [-h/2 + t_1, h/2 - t_2], with centroid z_cc = 0.5 * (z_min,core + z_max,core)
   - Skin 2: z in [h/2 - t_2, h/2], with centroid z_c2 = h/2 - t_2 / 2

2. Equivalent Bending Stiffness (Classical Sandwich Theory):
   For general asymmetric skins (E_1, t_1, E_2, t_2, E_core, t_core):
       A_ext = E_1 * t_1 + E_core * t_core + E_2 * t_2
       z_neutral = (E_1 * t_1 * z_c1 + E_core * t_core * z_cc + E_2 * t_2 * z_c2) / A_ext
       d_1 = |z_c1 - z_neutral|
       d_2 = |z_c2 - z_neutral|
       d_c = |z_cc - z_neutral|

       D_eff = E_1 * t_1 * (d_1)^2 + E_2 * t_2 * (d_2)^2 + E_core * t_core * (d_c)^2
               + (E_1 * t_1^3) / 12 + (E_2 * t_2^3) / 12 + (E_core * t_core^3) / 12

   For symmetric face sheets (t_1 = t_2 = t_s, E_1 = E_2 = E_s, z_neutral = 0, d_1 = d_2 = d/2):
       D_eff = (E_s * t_s * d^2) / 2 + (E_core * t_core^3) / 12 (+ E_s * t_s^3 / 6)

3. Equivalent Transverse Shear Stiffness:
       G_eff = G_core * d^2 / t_core
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..common.messages import MessageLog
from ..model.entities import Property
from .deck_reader import Card, KeywordBlock, parse_fortran_float


class CallableFloat(float):
    """Float subclass that allows both attribute access and function call syntax.

    Supports both ``prop.total_thickness`` (float) and ``prop.total_thickness()``
    seamlessly for compatibility across different calling conventions.
    """

    def __call__(self) -> float:
        return float(self)


@dataclass
class SandwichLayer:
    """Individual layer of a sandwich shell property (face skin or core).

    Upstream Fortran origin:
      - ``starter/source/properties/shell/hm_read_prop11.F``

    Parameters
    ----------
    layer_id : int
        1-based layer index (1: skin1, 2: core, 3: skin2).
    name : str
        Layer descriptor (e.g., 'skin1', 'core', 'skin2').
    thickness : float
        Layer thickness t_i.
    mat_id : int
        Material ID for this layer.
    phi : float
        Fiber orientation angle in degrees.
    nip : int
        Number of through-thickness integration points (default 1).
    z_min : float
        Bottom z-coordinate relative to laminate mid-surface.
    z_max : float
        Top z-coordinate relative to laminate mid-surface.
    weight : float
        Layer failure weight factor (F_weight, default 1.0).
    """

    layer_id: int = 1
    name: str = ""
    thickness: float = 0.0
    mat_id: int = 1
    phi: float = 0.0
    nip: int = 1
    z_min: float = 0.0
    z_max: float = 0.0
    weight: float = 1.0

    @property
    def z_mid(self) -> float:
        """Mid-surface z-coordinate of this layer."""
        val = 0.5 * (self.z_min + self.z_max)
        return 0.0 if math.isclose(val, 0.0, abs_tol=1e-15) else val


@dataclass
class Prop11Sandwich:
    """Sandwich shell property (/PROP/TYPE11, /PROP/SH_SANDW, /PROP/SANDWICH).

    Upstream Fortran origin:
      - ``starter/source/properties/shell/hm_read_prop11.F``
      - ``hm_cfg_files/config/CFG/radioss2026/PROP/prop_p11_sh_sandw.cfg``

    Parameters
    ----------
    id : int
        Property identifier.
    title : str
        Descriptive title.
    skin1 : SandwichLayer
        Bottom face sheet (layer 1).
    core : SandwichLayer
        Sandwich core (layer 2).
    skin2 : SandwichLayer
        Top face sheet (layer 3).
    layers : List[SandwichLayer]
        All layers in order from bottom to top.
    ishell : int
        Shell element formulation flag (default 24: QEPH).
    ismstr : int
        Small strain formulation flag (default 2).
    ish3n : int
        3-node shell element formulation flag (default 1).
    idrill : int
        Drilling DOF stiffness flag (default 0).
    thick_fail : float
        Percentage of layer thickness failing before element deletion.
    """

    id: int = 1
    title: str = ""
    skin1: Optional[SandwichLayer] = None
    core: Optional[SandwichLayer] = None
    skin2: Optional[SandwichLayer] = None
    layers: List[SandwichLayer] = field(default_factory=list)

    # Shell formulation parameters
    ishell: int = 24
    ismstr: int = 2
    ish3n: int = 1
    idrill: int = 0
    thick_fail: float = 0.0

    # Hourglass & damping
    hm: float = 0.01
    hf: float = 0.01
    hr: float = 0.01
    dm: float = 0.0
    dn: float = 0.0

    # Geometry & plasticity
    thick: float = 0.0
    area_shear: float = 5.0 / 6.0
    ithick: int = 0
    iplas: int = 1

    # Orthotropic reference vector
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    ip: int = 0

    def __post_init__(self) -> None:
        if self.skin1 is None:
            self.skin1 = SandwichLayer(layer_id=1, name="skin1", thickness=0.0, mat_id=1)
        if self.core is None:
            self.core = SandwichLayer(layer_id=2, name="core", thickness=0.0, mat_id=1)
        if self.skin2 is None:
            self.skin2 = SandwichLayer(layer_id=3, name="skin2", thickness=0.0, mat_id=1)

        if not self.layers:
            self.layers = [self.skin1, self.core, self.skin2]
        else:
            if len(self.layers) >= 1:
                self.skin1 = self.layers[0]
            if len(self.layers) >= 2:
                self.core = self.layers[1]
            if len(self.layers) >= 3:
                self.skin2 = self.layers[2]

        self.compute_layer_bounds()

    def compute_layer_bounds(self) -> None:
        """Compute through-thickness z-bounds for all layers relative to mid-surface."""
        t_tot = sum(layer.thickness for layer in self.layers)
        if self.thick <= 0.0 or math.isclose(self.thick, 0.0):
            self.thick = t_tot

        h = t_tot if t_tot > 0.0 else self.thick
        z_curr = -0.5 * h

        for layer in self.layers:
            layer.z_min = z_curr
            layer.z_max = z_curr + layer.thickness
            z_curr = layer.z_max

    @property
    def total_thickness(self) -> CallableFloat:
        """Total sandwich thickness h = t_1 + t_core + t_2."""
        t_tot = sum(layer.thickness for layer in self.layers)
        val = t_tot if t_tot > 0.0 else self.thick
        return CallableFloat(val)

    @property
    def core_thickness(self) -> CallableFloat:
        """Thickness of core layer t_core."""
        val = self.core.thickness if self.core else 0.0
        return CallableFloat(val)

    @property
    def face_sheet_distance_d(self) -> CallableFloat:
        """Distance d between centroids of face sheets: d = t_core + 0.5 * (t_1 + t_2)."""
        t1 = self.skin1.thickness if self.skin1 else 0.0
        tc = self.core.thickness if self.core else 0.0
        t2 = self.skin2.thickness if self.skin2 else 0.0
        return CallableFloat(tc + 0.5 * (t1 + t2))

    @property
    def d(self) -> CallableFloat:
        """Alias for face_sheet_distance_d."""
        return self.face_sheet_distance_d

    def get_layer_at_z(self, z: float) -> SandwichLayer:
        """Return the layer containing through-thickness coordinate z."""
        if not self.layers:
            raise ValueError("No layers defined in sandwich shell")

        eps = 1e-12
        for layer in self.layers[:-1]:
            if z <= layer.z_max + eps:
                return layer
        return self.layers[-1]

    def equivalent_bending_stiffness(
        self,
        e_skin1: float,
        e_skin2: float,
        e_core: float = 0.0,
        include_skin_bending: bool = True,
    ) -> float:
        """Compute equivalent bending stiffness D_eff per unit width.

        Classical Sandwich Theory (Plantema / Allen / Zenkert):
        --------------------------------------------------------
        For general asymmetric face sheets (E_1, t_1, E_2, t_2, E_core, t_core):
            A_ext = E_1 * t_1 + E_core * t_core + E_2 * t_2
            z_neutral = (E_1*t_1*z_c1 + E_core*t_c*z_cc + E_2*t_2*z_c2) / A_ext
            d_1 = |z_c1 - z_neutral|
            d_2 = |z_c2 - z_neutral|
            d_c = |z_cc - z_neutral|

            D_eff = E_1 * t_1 * d_1^2 + E_2 * t_2 * d_2^2 + E_core * t_core * d_c^2
                    + (E_1 * t_1^3) / 12 + (E_2 * t_2^3) / 12 + (E_core * t_core^3) / 12

        For symmetric face sheets (t_1 = t_2 = t_s, E_1 = E_2 = E_s):
            D_eff = (E_s * t_s * d^2) / 2 + (E_core * t_core^3) / 12 (+ E_s * t_s^3 / 6)
        """
        t1 = self.skin1.thickness if self.skin1 else 0.0
        tc = self.core.thickness if self.core else 0.0
        t2 = self.skin2.thickness if self.skin2 else 0.0

        z1 = self.skin1.z_mid if self.skin1 else 0.0
        zc = self.core.z_mid if self.core else 0.0
        z2 = self.skin2.z_mid if self.skin2 else 0.0

        a1 = e_skin1 * t1
        ac = e_core * tc
        a2 = e_skin2 * t2
        a_tot = a1 + ac + a2

        if a_tot > 0.0:
            z_neutral = (a1 * z1 + ac * zc + a2 * z2) / a_tot
        else:
            z_neutral = 0.0

        d1 = z1 - z_neutral
        dc = zc - z_neutral
        d2 = z2 - z_neutral

        # Core bending contribution
        d_core = (e_core * (tc ** 3)) / 12.0 + ac * (dc ** 2)

        # Parallel-axis theorem terms for face sheets
        d_skins_parallel = a1 * (d1 ** 2) + a2 * (d2 ** 2)

        # Self-bending terms for face sheets (t^3 / 12)
        if include_skin_bending:
            d_skin_self = (e_skin1 * (t1 ** 3) + e_skin2 * (t2 ** 3)) / 12.0
        else:
            d_skin_self = 0.0

        return float(d_skins_parallel + d_skin_self + d_core)

    def equivalent_shear_stiffness(self, g_core: float) -> float:
        """Compute equivalent transverse shear stiffness factor.

        Classical Sandwich Theory (Plantema / Allen):
            G_eff = G_core * d^2 / t_core
        """
        tc = float(self.core_thickness)
        d = float(self.face_sheet_distance_d)
        if tc > 0.0:
            return float(g_core * (d ** 2) / tc)
        return 0.0


# ============================================================================
# Card parsing helpers
# ============================================================================

def _pf(val: Any, default: float = 0.0) -> float:
    if val in (None, ""):
        return default
    try:
        return parse_fortran_float(str(val))
    except (ValueError, TypeError):
        return default


def _pi(val: Any, default: int = 0) -> int:
    if val in (None, ""):
        return default
    try:
        return int(parse_fortran_float(str(val)))
    except (ValueError, TypeError):
        return default


def _is_layer_token_card(toks: Sequence[str]) -> bool:
    """Return True if tokens look like a layer definition line rather than a property header."""
    if not toks:
        return False
    # Layer definition: Prop_phi (float), Prop_Thick (float), Prop_Zi (float), Prop_mi (int)
    # A header card typically starts with integer flags (Ishell, NIP) or hourglass floats (0.01)
    # If there are >= 3 numeric tokens and the second is a small positive thickness (< 0.5)
    try:
        t0 = parse_fortran_float(toks[0])
        t1 = parse_fortran_float(toks[1])
        if 0.0 < t1 < 10.0 and len(toks) >= 3:
            return True
    except (ValueError, TypeError, IndexError):
        pass
    return False


def _parse_layer_card(
    card: Card,
    fixed: bool,
    layer_idx: int,
) -> SandwichLayer:
    """Parse one layer line into a SandwichLayer."""
    name_map = {1: "skin1", 2: "core", 3: "skin2"}
    name = name_map.get(layer_idx, f"layer_{layer_idx}")

    phi = 0.0
    thick = 0.0
    zi = 0.0
    mat_id = 1
    nip = 1
    weight = 1.0

    # Fixed format check: %20lg%20lg%20lg%10d%10d%20lg
    if fixed and len(card.raw.rstrip()) >= 40:
        fields = card.cut((20, 20, 20, 10, 10, 20))
        phi = _pf(fields[0], 0.0)
        thick = _pf(fields[1], 0.0)
        zi = _pf(fields[2], 0.0) if len(fields) > 2 else 0.0
        mat_id = _pi(fields[3], 1) if len(fields) > 3 else 1
        nip = _pi(fields[4], 1) if len(fields) > 4 else 1
        weight = _pf(fields[5], 1.0) if len(fields) > 5 else 1.0
    else:
        toks = card.tokens()
        if len(toks) >= 4:
            phi = _pf(toks[0], 0.0)
            thick = _pf(toks[1], 0.0)
            zi = _pf(toks[2], 0.0)
            mat_id = _pi(toks[3], 1)
            nip = _pi(toks[4], 1) if len(toks) > 4 else 1
            weight = _pf(toks[5], 1.0) if len(toks) > 5 else 1.0
        elif len(toks) == 3:
            phi = _pf(toks[0], 0.0)
            thick = _pf(toks[1], 0.0)
            mat_id = _pi(toks[2], 1)
            nip = 1
            weight = 1.0
        elif len(toks) == 2:
            phi = _pf(toks[0], 0.0)
            thick = _pf(toks[1], 0.0)
        elif len(toks) == 1:
            thick = _pf(toks[0], 0.0)

    return SandwichLayer(
        layer_id=layer_idx,
        name=name,
        thickness=thick,
        mat_id=mat_id,
        phi=phi,
        nip=max(1, nip),
        weight=weight,
    )


def parse_sandwich_card(block: KeywordBlock, log: MessageLog) -> Property:
    """Parse /PROP/TYPE11 (SH_SANDW, SANDWICH) sandwich shell property.

    Upstream OpenRadioss Fortran:
      - ``starter/source/properties/shell/hm_read_prop11.F``
    """
    from .prop_reader import _data_cards

    title, cards, fixed = _data_cards(block)
    prop11 = Prop11Sandwich(id=block.user_id, title=title)

    # Filter blank cards and comments
    non_blank_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not non_blank_cards:
        prop11.compute_layer_bounds()
        return _make_property(block.user_id, title, prop11)

    # Check whether the cards start directly with layer cards (<= 3 cards) or standard header cards (>= 4 cards)
    has_headers = len(non_blank_cards) >= 4

    if has_headers:
        # Card 0: Ishell Ismstr Ish3n Idrill [spacing 20] P_Thick_Fail
        # Layout: %10d%10d%10d%10d%20lg
        c0 = non_blank_cards[0]
        f0 = c0.cut((10, 10, 10, 10, 20, 20)) if fixed and len(c0.raw) >= 40 else c0.tokens()
        if len(f0) > 0:
            prop11.ishell = _pi(f0[0], 24)
        if len(f0) > 1:
            prop11.ismstr = _pi(f0[1], 2)
        if len(f0) > 2:
            prop11.ish3n = _pi(f0[2], 1)
        if len(f0) > 3:
            prop11.idrill = _pi(f0[3], 0)
        if len(f0) > 4:
            # In fixed cut, field 4 is 20-col blank, field 5 is P_Thick_Fail
            val_fail = f0[5] if (fixed and len(f0) > 5) else f0[4]
            prop11.thick_fail = _pf(val_fail, 0.0)

        # Card 1: Hm Hf Hr Dm Dn
        # Layout: %20lg%20lg%20lg%20lg%20lg
        if len(non_blank_cards) > 1:
            c1 = non_blank_cards[1]
            f1 = c1.cut((20, 20, 20, 20, 20)) if fixed and len(c1.raw) >= 60 else c1.tokens()
            if len(f1) > 0:
                prop11.hm = _pf(f1[0], 0.01)
            if len(f1) > 1:
                prop11.hf = _pf(f1[1], 0.01)
            if len(f1) > 2:
                prop11.hr = _pf(f1[2], 0.01)
            if len(f1) > 3:
                prop11.dm = _pf(f1[3], 0.0)
            if len(f1) > 4:
                prop11.dn = _pf(f1[4], 0.0)

        # Card 2: NIP ISTRAIN THICK AREA_SHEAR [spacing 10] ITHICK IPLAS
        # Layout: %10d%10d%20lg%20lg%10s%10d%10d
        if len(non_blank_cards) > 2:
            c2 = non_blank_cards[2]
            f2 = c2.cut((10, 10, 20, 20, 10, 10, 10)) if fixed and len(c2.raw) >= 60 else c2.tokens()
            if len(f2) > 2:
                prop11.thick = _pf(f2[2], 0.0)
            if len(f2) > 3:
                prop11.area_shear = _pf(f2[3], 5.0 / 6.0)
            if len(f2) > 5:
                prop11.ithick = _pi(f2[5], 0)
            if len(f2) > 6:
                prop11.iplas = _pi(f2[6], 1)

        # Card 3: V_X V_Y V_Z SKEW_CSID Iorth Ipos Ip
        # Layout: %20lg%20lg%20lg%10d%10d%10d%10d
        if len(non_blank_cards) > 3:
            c3 = non_blank_cards[3]
            f3 = c3.cut((20, 20, 20, 10, 10, 10, 10)) if fixed and len(c3.raw) >= 60 else c3.tokens()
            if len(f3) > 0:
                prop11.vx = _pf(f3[0], 1.0)
            if len(f3) > 1:
                prop11.vy = _pf(f3[1], 0.0)
            if len(f3) > 2:
                prop11.vz = _pf(f3[2], 0.0)
            if len(f3) > 3:
                prop11.skew_id = _pi(f3[3], 0)
            if len(f3) > 4:
                prop11.iorth = _pi(f3[4], 0)
            if len(f3) > 5:
                prop11.ipos = _pi(f3[5], 0)
            if len(f3) > 6:
                prop11.ip = _pi(f3[6], 0)

        layer_cards = non_blank_cards[4:]
    else:
        layer_cards = non_blank_cards

    # Parse layers
    parsed_layers: List[SandwichLayer] = []
    for idx, card in enumerate(layer_cards, start=1):
        parsed_layers.append(_parse_layer_card(card, fixed, idx))

    if parsed_layers:
        prop11.layers = parsed_layers
        if len(parsed_layers) >= 1:
            prop11.skin1 = parsed_layers[0]
            prop11.skin1.name = "skin1"
        if len(parsed_layers) >= 2:
            prop11.core = parsed_layers[1]
            prop11.core.name = "core"
        if len(parsed_layers) >= 3:
            prop11.skin2 = parsed_layers[2]
            prop11.skin2.name = "skin2"

    prop11.compute_layer_bounds()
    return _make_property(block.user_id, title, prop11)


def _make_property(prop_id: int, title: str, prop11: Prop11Sandwich) -> Property:
    """Package Prop11Sandwich into model.entities.Property."""
    params: Dict[str, Any] = {
        "thick": float(prop11.total_thickness),
        "t_total": float(prop11.total_thickness),
        "skin1": prop11.skin1,
        "core": prop11.core,
        "skin2": prop11.skin2,
        "layers": prop11.layers,
        "n_layers": len(prop11.layers),
        "nip": max(1, sum(l.nip for l in prop11.layers)) if prop11.layers else 1,
        "ishell": prop11.ishell,
        "ismstr": prop11.ismstr,
        "ish3n": prop11.ish3n,
        "idrill": prop11.idrill,
        "thick_fail": prop11.thick_fail,
        "p_thick_fail": prop11.thick_fail,
        "hm": prop11.hm,
        "hf": prop11.hf,
        "hr": prop11.hr,
        "dm": prop11.dm,
        "dn": prop11.dn,
        "area_shear": prop11.area_shear,
        "ithick": prop11.ithick,
        "iplas": prop11.iplas,
        "vx": prop11.vx,
        "vy": prop11.vy,
        "vz": prop11.vz,
        "skew_id": prop11.skew_id,
        "iorth": prop11.iorth,
        "ipos": prop11.ipos,
        "ip": prop11.ip,
        "prop11": prop11,
        "prop_sandwich": prop11,
    }

    return Property(
        id=prop_id,
        type=11,
        title=title,
        params=params,
    )


# Function aliases
parse_prop11_sandwich = parse_sandwich_card
parse_sandwich = parse_sandwich_card
