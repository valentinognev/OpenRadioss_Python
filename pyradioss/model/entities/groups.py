"""Part groups, sets and subsets.

Split out of the former single-module ``pyradioss/model/entities.py`` (task
P1.4).  The blocks below are byte-identical to their originals, with one
mechanical exception: relative ``from . import`` statements gained a dot,
because code that was one module deep now sits one package level deeper.
Nothing else about any block changed.

This module is self-contained: the split was checked and leaves no cross-file
reference of any kind, so no sibling submodule is imported here.  A future
cross-file reference should be an ``if TYPE_CHECKING:`` import (annotations
are lazy here, via ``from __future__ import annotations``) unless the
reference is actually evaluated at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math

import numpy as np


@dataclass
class SmsGlobal:
    """/SMS or /AMS (M101): Selective Mass Scaling global parameters.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_sms.F``.
    """
    grpart_id: int = 0
    dt_target: float = 0.0


@dataclass
class Subset:
    """/SUBSET (M113): Hierarchical model component subset."""
    id: int
    title: str = ""
    assembly_ids: List[int] = field(default_factory=list)


@dataclass
class SetGeneric:
    """/SET (M137): Generic ID collection set."""
    id: int
    set_type: str
    title: str = ""
    ids: List[int] = field(default_factory=list)
    key: str = ""
    seg_nodes: List[List[int]] = field(default_factory=list)


# ============================================================================
# M201 Entities: TH_SUBS, THPART, WAV_SHA, SENSORS, PCOMPP, TYPE51
# ============================================================================

@dataclass
class ThSubs:
    """``/TH/SUBS/id`` (M201): Substructure Time History output block."""
    id: int = 0
    title: str = ""
    prefix: str = "TH"
    vars: List[str] = field(default_factory=list)
    subs_ids: List[int] = field(default_factory=list)


@dataclass
class ThPartGroup:
    """``/THPART/GR.../id`` (M201): Group-based Time History part output block."""
    id: int = 0
    title: str = ""
    elem_type: str = "SHEL"  # 'BEAM', 'BRIC', 'QUAD', 'SH3N', 'SHEL', 'SPRI', 'TRUS'
    grelem_id: int = 0
