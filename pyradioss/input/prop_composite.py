"""Multi-layer Composite Shell (/PROP/TYPE17) and Thermal Shell (/PROP/TYPE19).

Upstream OpenRadioss Fortran References:
----------------------------------------
- /PROP/TYPE17 (SH_COMP, STACK):
  ``$OR_SRC/starter/source/properties/shell/hm_read_prop17.F``
  ``$OR_SRC/hm_cfg_files/config/CFG/radioss2022/PROP/prop_p17_stack.cfg``
- /PROP/TYPE19 (THERM_SHELL, SH_THERM):
  ``$OR_SRC/starter/source/properties/shell/hm_read_prop19.F``

Physics & Stacking Formulations:
--------------------------------
1. Multi-Layer Composite Shell (/PROP/TYPE17):
   - Shell laminate consisting of N_layers plies (up to 100 plies).
   - Each layer i has thickness t_i, fiber angle phi_i, material ID mat_id_i, and integration points nip_i.
   - Total laminate thickness:
     t_total = sum(t_i)
   - Through-thickness coordinates:
     Lower coordinate: z_lower,i = Z0 + sum_{k=1}^{i-1} t_k
     Upper coordinate: z_upper,i = z_lower,i + t_i
     Midplane coordinate: z_mid,i = 0.5 * (z_lower,i + z_upper,i)
     When Z0 is not specified, symmetric midplane reference Z0 = -0.5 * t_total is used.
   - For nip=2 Gauss integration points per layer:
     z_{i,1} = z_mid,i - t_i / (2 * sqrt(3))
     z_{i,2} = z_mid,i + t_i / (2 * sqrt(3))

2. Thermal Shell (/PROP/TYPE19):
   - Shell property with through-thickness thermal conduction.
   - Parameters: thickness t, thermal conductivity k_th, heat capacity C_p,
     thermal expansion coefficient alpha_th.
   - N_temp through-thickness temperature integration points across [-t/2, +t/2].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..common.messages import MessageLog
from ..model.entities import Property
from .deck_reader import Card, KeywordBlock, parse_fortran_float


@dataclass
class Prop17Layer:
    """Individual layer within a /PROP/TYPE17 multi-layer composite shell.

    Parameters
    ----------
    layer_id : int
        1-based layer index.
    mat_id : int
        Material ID for this layer.
    thick : float
        Layer thickness t_i.
    phi : float
        Fiber orientation angle phi_i in degrees.
    nip : int
        Number of through-thickness integration points for this layer (1 or 2).
    z_mid : float
        Through-thickness midplane coordinate of this layer.
    z_lower : float
        Bottom z-coordinate of this layer.
    z_upper : float
        Top z-coordinate of this layer.
    gauss_z : List[float]
        Through-thickness integration point z-coordinates.
    """

    layer_id: int
    mat_id: int
    thick: float
    phi: float = 0.0
    nip: int = 1
    z_mid: float = 0.0
    z_lower: float = 0.0
    z_upper: float = 0.0
    gauss_z: List[float] = field(default_factory=list)


@dataclass
class Prop17CompositeShell:
    """Multi-layer composite shell property (/PROP/TYPE17, /PROP/STACK, /PROP/SH_COMP).

    Upstream Fortran origin:
      - ``starter/source/properties/shell/hm_read_prop17.F``
    """

    id: int = 1
    title: str = ""
    thick: float = 0.0  # Total thickness (auto-computed from layers if 0)
    z0: Optional[float] = None  # Reference plane offset (default -thick/2)
    layers: List[Prop17Layer] = field(default_factory=list)

    # Shell formulation parameters
    ishell: int = 1
    ismstr: int = 2
    ish3n: int = 2
    idrill: int = 0
    area_shear: float = 5.0 / 6.0
    ithick: int = 0
    iplas: int = 1
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0

    def __post_init__(self) -> None:
        self.compute_z_coordinates()

    @property
    def n_layers(self) -> int:
        """Number of layers in the laminate."""
        return len(self.layers)

    @property
    def total_thickness(self) -> float:
        """Total laminate thickness."""
        if self.layers:
            return float(sum(layer.thick for layer in self.layers))
        return float(self.thick)

    def add_layer(
        self,
        thick: float,
        phi: float = 0.0,
        mat_id: int = 1,
        nip: int = 1,
    ) -> Prop17Layer:
        """Add a layer and recompute through-thickness coordinates."""
        lid = len(self.layers) + 1
        layer = Prop17Layer(
            layer_id=lid,
            mat_id=mat_id,
            thick=float(thick),
            phi=float(phi),
            nip=int(nip) if nip > 0 else 1,
        )
        self.layers.append(layer)
        self.compute_z_coordinates()
        return layer

    def get_layer(self, index: int) -> Prop17Layer:
        """Retrieve layer by 0-based or 1-based index."""
        idx = index - 1 if (index > 0 and index <= len(self.layers)) else index
        return self.layers[idx]

    def compute_z_coordinates(self) -> None:
        """Compute through-thickness integration coordinates for all layers."""
        if not self.layers:
            return

        tot_thk = sum(l.thick for l in self.layers)
        if self.thick <= 0.0 or math.isclose(self.thick, 0.0):
            self.thick = tot_thk

        z_curr = float(self.z0) if self.z0 is not None else -0.5 * tot_thk

        for layer in self.layers:
            layer.z_lower = z_curr
            layer.z_upper = z_curr + layer.thick
            layer.z_mid = 0.5 * (layer.z_lower + layer.z_upper)

            if layer.nip == 2:
                # 2-point Gauss integration per layer
                offset = layer.thick / (2.0 * math.sqrt(3.0))
                layer.gauss_z = [layer.z_mid - offset, layer.z_mid + offset]
            else:
                # 1 midpoint per layer
                layer.gauss_z = [layer.z_mid]

            z_curr = layer.z_upper


@dataclass
class Prop19ThermalShell:
    """Thermal shell property (/PROP/TYPE19, /PROP/THERM_SHELL, /PROP/SH_THERM).

    Upstream Fortran origin:
      - ``starter/source/properties/shell/hm_read_prop19.F``
    """

    id: int = 1
    title: str = ""
    thick: float = 1.0  # Shell thickness t (m)
    k_th: float = 50.0  # Thermal conductivity k_th (W / (m * K))
    c_p: float = 500.0  # Specific heat capacity C_p (J / (kg * K))
    alpha_th: float = 1.2e-5  # Thermal expansion coefficient alpha_th (1 / K)
    n_temp: int = 3  # Number of through-thickness temperature integration points (3 or 5)
    mat_id: int = 0
    orientangle: float = 0.0
    density: float = 7800.0

    def __post_init__(self) -> None:
        if self.n_temp < 1:
            self.n_temp = 3

    def temperature_points(self) -> List[float]:
        """Compute through-thickness coordinates z_k in [-thick/2, +thick/2]."""
        n = self.n_temp
        h = self.thick
        if n == 1:
            return [0.0]
        # Uniformly distributed through-thickness stations from -h/2 to +h/2
        return [float(-0.5 * h + i * h / (n - 1)) for i in range(n)]


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


def _extract_tokens(cards: Sequence[Card]) -> List[List[str]]:
    """Extract list of tokens per card."""
    lines = []
    for c in cards:
        if c.is_blank:
            continue
        toks = c.tokens()
        if toks:
            lines.append(toks)
    return lines


def parse_prop17_composite(block: KeywordBlock, log: MessageLog) -> Property:
    """Parse /PROP/TYPE17 (SH_COMP, STACK) multi-layer composite shell property."""
    from .prop_reader import _data_cards

    title, cards, _fixed = _data_cards(block)
    prop17 = Prop17CompositeShell(id=block.user_id, title=title)

    lines = _extract_tokens(cards)

    # Line 1: Ishell Ismstr ISH3N Idrill plyxfem Z0 Vinterply
    if len(lines) > 0:
        row = lines[0]
        if len(row) > 0:
            prop17.ishell = _pi(row[0], 1)
        if len(row) > 1:
            prop17.ismstr = _pi(row[1], 2)
        if len(row) > 2:
            prop17.ish3n = _pi(row[2], 2)
        if len(row) > 3:
            prop17.idrill = _pi(row[3], 0)
        if len(row) > 5:
            val_z0 = _pf(row[5], 0.0)
            if row[5] not in ("", "0", "0.0"):
                prop17.z0 = val_z0

    # Line 2: Hm Hf Hr Dm Dn
    # Line 3: P_Thick_Fail THICK AREA_SHEAR ITHICK IPLAS (or N_layers)
    n_layers_hint = 0
    if len(lines) > 2:
        row = lines[2]
        if len(row) > 1:
            prop17.thick = _pf(row[1], 0.0)
        if len(row) > 2:
            prop17.area_shear = _pf(row[2], 5.0 / 6.0)
        if len(row) > 3:
            prop17.ithick = _pi(row[3], 0)
        if len(row) > 4:
            prop17.iplas = _pi(row[4], 1)

    # Line 4: V_X V_Y V_Z SKEW_CSID Iorth Ipos Ip
    if len(lines) > 3:
        row = lines[3]
        if len(row) > 0:
            prop17.vx = _pf(row[0], 0.0)
        if len(row) > 1:
            prop17.vy = _pf(row[1], 0.0)
        if len(row) > 2:
            prop17.vz = _pf(row[2], 0.0)
        if len(row) > 3:
            prop17.skew_id = _pi(row[3], 0)
        if len(row) > 4:
            prop17.iorth = _pi(row[4], 0)
        if len(row) > 5:
            prop17.ipos = _pi(row[5], 0)

    # Subsequent lines: layer definitions
    # Format per layer: THICK_i PHI_i MAT_ID_i [NIP_i]
    # or plies defined sequentially
    layer_lines = lines[4:] if len(lines) > 4 else []
    for row in layer_lines:
        if len(row) == 0:
            continue
        # Check if this row is a layer card (first item is numeric thickness)
        try:
            thk = parse_fortran_float(row[0])
        except (ValueError, TypeError):
            continue

        phi = _pf(row[1], 0.0) if len(row) > 1 else 0.0
        mat_id = _pi(row[2], 1) if len(row) > 2 else 1
        nip = _pi(row[3], 1) if len(row) > 3 else 1

        prop17.add_layer(thick=thk, phi=phi, mat_id=mat_id, nip=nip)

    prop17.compute_z_coordinates()

    # Pack into Property params dict
    params = {
        "thick": prop17.total_thickness,
        "t_total": prop17.total_thickness,
        "n_layers": prop17.n_layers,
        "layers": prop17.layers,
        "prop17": prop17,
        "nip": max(1, sum(l.nip for l in prop17.layers)) if prop17.layers else 1,
        "ishell": prop17.ishell,
        "ismstr": prop17.ismstr,
        "ish3n": prop17.ish3n,
        "idrill": prop17.idrill,
        "area_shear": prop17.area_shear,
        "z0": prop17.z0,
    }

    return Property(
        id=block.user_id,
        type=17,
        title=title,
        params=params,
    )


def parse_prop19_thermal(block: KeywordBlock, log: MessageLog) -> Property:
    """Parse /PROP/TYPE19 (THERM_SHELL, SH_THERM) thermal shell property."""
    from .prop_reader import _data_cards

    title, cards, _fixed = _data_cards(block)
    prop19 = Prop19ThermalShell(id=block.user_id, title=title)

    lines = _extract_tokens(cards)

    # Card 1: material, thickness1, orientangle, grsh4n_ID, grsh3n_ID, integrationpoints, orientangle2
    if len(lines) > 0:
        row = lines[0]
        if len(row) > 0:
            prop19.mat_id = _pi(row[0], 0)
        if len(row) > 1:
            prop19.thick = _pf(row[1], 1.0)
        if len(row) > 2:
            prop19.orientangle = _pf(row[2], 0.0)
        if len(row) > 5:
            prop19.n_temp = _pi(row[5], 3)

    # Card 2 or thermal params: k_th, c_p, alpha_th
    if len(lines) > 1:
        row = lines[1]
        if len(row) > 0:
            prop19.k_th = _pf(row[0], 50.0)
        if len(row) > 1:
            prop19.c_p = _pf(row[1], 500.0)
        if len(row) > 2:
            prop19.alpha_th = _pf(row[2], 1.2e-5)

    params = {
        "thick": prop19.thick,
        "k_th": prop19.k_th,
        "c_p": prop19.c_p,
        "alpha_th": prop19.alpha_th,
        "n_temp": prop19.n_temp,
        "z_coords": prop19.temperature_points(),
        "prop19": prop19,
        "mat_id": prop19.mat_id,
    }

    return Property(
        id=block.user_id,
        type=19,
        title=title,
        params=params,
    )
