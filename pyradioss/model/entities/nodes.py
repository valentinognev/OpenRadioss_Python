"""Node groups, surfaces and other geometric entities.

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


# ============================================================================
# Groups & surfaces (Fortran: groupdef_mod.F, IGRNOD/IGRSURF)
# ============================================================================

@dataclass
class Box:
    """A /BOX volume, used by /GRNOD/BOX to select nodes (M37: the three
    real geometries of ``starter/source/model/box/rdbox.F``).

    ``kind`` selects the geometry and which fields are meaningful:

    * ``'RECTA'`` — axis-aligned box between two diagonal corners.
      Corners come either from the coordinate cards (``corner_min``/
      ``corner_max`` filled at read time) or from two NODES (``node1``/
      ``node2`` > 0 — resolved against the mesh at group-resolution
      time, the cfg recta.cfg N1/N2 fields); ``iskew`` > 0 evaluates
      limits in the local skew frame;
    * ``'CYLIN'`` — finite cylinder: axis segment ``p1``->``p2``,
      ``diameter``; a node is inside when its axis projection falls
      between the caps and its distance from the axis is <= D/2
      (rdbox.F INSIDE_CYLINDER, boundaries inclusive);
    * ``'SPHER'`` — sphere: center ``p1``, ``diameter``.
    """

    id: int
    corner_min: Optional[np.ndarray] = None  # (3,) RECTA
    corner_max: Optional[np.ndarray] = None  # (3,) RECTA
    title: str = ""
    kind: str = "RECTA"                      # 'RECTA' | 'CYLIN' | 'SPHER'
    p1: Optional[np.ndarray] = None          # (3,) CYLIN base / SPHER center
    p2: Optional[np.ndarray] = None          # (3,) CYLIN axis end
    diameter: float = 0.0                    # CYLIN / SPHER
    iskew: int = 0                           # CYLIN / SPHER
    node1: int = 0                           # RECTA/CYLIN corner/axis node
    node2: int = 0
    box_ids: List[int] = field(default_factory=list) # /BOX/BOX (M132): positive for union, negative for subtraction


@dataclass
class NodeGroup:
    """A /GRNOD node group. After Starter resolution, ``node_idx`` holds
    dense 0-based node indices (Fortran IGRNOD(IGR)%ENTITY).

    M37 adds the remaining real-deck subtypes (48 % of the official
    corpus): nodes of surfaces (/GRNOD/SURF — hm_surfnod.F), recursive
    group-of-groups (/GRNOD/GRNOD — hm_grogronod.F, iterative fixpoint
    with cycle detection; NEGATIVE ids REMOVE the referenced group's
    nodes, and removal wins over addition whatever the order — the
    BUFTMP = -1 convention), nodes of element groups (/GRNOD/GRSHEL|
    GRSH3N|GRBRIC|... — hm_elngr*.F) and generated id ranges
    (/GRNOD/GENE first..last [+ GEN_INCR increment])."""

    id: int
    title: str = ""
    # Unresolved content, as read from the deck:
    node_ids: Union[List[int], np.ndarray] = field(default_factory=list)   # /GRNOD/NODE
    part_ids: List[int] = field(default_factory=list)   # /GRNOD/PART
    box_ids: List[int] = field(default_factory=list)    # /GRNOD/BOX
    surf_ids: List[int] = field(default_factory=list)   # /GRNOD/SURF (M37)
    line_ids: List[int] = field(default_factory=list)   # /GRNOD/LINE (M136)
    grnod_ids: List[int] = field(default_factory=list)  # /GRNOD/GRNOD, signed
    # /GRNOD/GRSHEL|GRSH3N|GRBRIC|GRTRUS|GRBEAM|GRSPRI: (family, group id)
    # pairs — family is the canonical element-group key ('SHEL', 'SH3N',
    # 'BRIC', ...), see Model.egroups (M37)
    egroup_refs: List[tuple] = field(default_factory=list)
    # /GRNOD/GENE (+ GEN_INCR): (first_id, last_id, incr) user-id ranges
    gene_ranges: List[tuple] = field(default_factory=list)
    # Resolved by the Starter:
    node_idx: Optional[np.ndarray] = None


@dataclass
class Surface:
    """A /SURF contact surface: a set of 3/4-node segments.

    Fortran: IGRSURF(ISU)%NODES(NSEG,4). Segments from /SURF/PART are the
    free (outer) faces of the part's elements — extracted by the Starter,
    like the Fortran surface-from-part builder in starter/source/model/sets.

    Since M4 every segment also records its *provenance* — which element it
    is a face of (Fortran IGRSURF%ELTYP/ELEM). Contact needs this twice:

    * the Radioss penalty stiffness and variable-gap formulas are written
      in terms of the parent element (shell thickness, solid volume...);
    * element deletion (/FAIL, M3): a segment whose parent element has
      GBUF%OFF = 0 must drop out of the main surface, so freshly created
      crack faces stop carrying contact forces (the IDEL treatment of the
      original interfaces).
    """

    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)         # /SURF/PART
    seg_nodes: List[List[int]] = field(default_factory=list)  # /SURF/SEG (user ids)
    # M37 subtypes:
    # /SURF/SURF — surface-of-surfaces (hm_read_surfsurf.F): the listed
    # surfaces' segments are CONCATENATED, resolved by iterative fixpoint
    # with cycle detection; a NEGATIVE id includes the surface with its
    # segment node order REVERSED (n4 n3 n2 n1 — the normal flips).
    surf_ids: List[int] = field(default_factory=list)
    # /SURF/GRSHEL | /SURF/GRSH3N — every element of the element group
    # becomes a segment (hm_surfgr2/surftage): (family, group id) pairs.
    egroup_refs: List[tuple] = field(default_factory=list)
    # Resolved by the Starter: (nseg, 4) 0-based node indices; triangles
    # repeat the 3rd node in the 4th slot (Radioss convention).
    segments: Optional[np.ndarray] = None
    # Provenance, parallel to ``segments`` (resolved by the Starter):
    # seg_gtype[i] = element-group attribute on Model ('shells', 'bricks',
    # 'tetras', 'sh3n') or '' for explicit /SURF/SEG segments;
    # seg_elem[i]  = row in that group (-1 for explicit segments).
    seg_gtype: Optional[np.ndarray] = None    # (nseg,) dtype '<U8'
    seg_elem: Optional[np.ndarray] = None     # (nseg,) int64
    # /SURF/PLANE (M92): infinite plane defined by point P1 and normal point P2
    plane_p1: Optional[np.ndarray] = None     # (3,) float [X_A, Y_A, Z_A]
    plane_p2: Optional[np.ndarray] = None     # (3,) float [X_B, Y_B, Z_B]
    # M115 extensions:
    mat_ids: List[int] = field(default_factory=list)   # /SURF/MAT
    prop_ids: List[int] = field(default_factory=list)  # /SURF/PROP
    box_ids: List[int] = field(default_factory=list)   # /SURF/BOX
    modifier: str = ""                                 # 'EXT', 'ALL', 'FREE'
    # /SURF/ELLIPSE (M132): ellipsoidal quadric surface
    ellipse_center: Optional[np.ndarray] = None        # (3,) [Xc, Yc, Zc]
    ellipse_semiaxes: Optional[np.ndarray] = None      # (3,) [a, b, c]
    ellipse_skew: int = 0
    # /SURF/CYL & /SURF/SPHER & /SURF/SUB (M134):
    cyl_center: Optional[np.ndarray] = None            # (3,) [X0, Y0, Z0]
    cyl_axis: Optional[np.ndarray] = None              # (3,) [Ax, Ay, Az]
    cyl_radius: float = 0.0
    cyl_length: float = 0.0
    spher_center: Optional[np.ndarray] = None          # (3,) [Xc, Yc, Zc]
    spher_radius: float = 0.0
    subset_surf_ids: List[int] = field(default_factory=list)


@dataclass
class Line:
    """A /LINE edge set: 2-node segments, the sides of /INTER/TYPE11
    edge-to-edge contact.

    Fortran: IGRSLIN(ISL)%NODES(NSEG,2) built by
    ``starter/source/model/sets/hm_read_lines.F``. The port supports

    * ``/LINE/SURF`` — every unique edge of the segments of the listed
      surfaces (with element provenance carried over from the surface, so
      edges of deleted elements drop out, exactly like surface segments);
    * ``/LINE/SEG``  — explicit node pairs;
    * ``/LINE/EDGE`` (M37) — only the BORDER edges of the listed
      surfaces: edges used by exactly ONE segment (``linedge.F``
      'REMOVAL OF INTERNAL SEGMENTS (EXCEPT BORDERS)' — interior edges,
      shared by two segments, are removed entirely, which turns the
      free boundary of a shell patch into a line);
    * ``/LINE/LINE`` (M37) — line-of-lines: the listed lines' edges
      concatenated (hm_lines_of_lines.F, fixpoint + cycle detection);
    * ``/LINE/PART`` (M37) — every 1-D element (truss/beam/spring) of
      the listed parts becomes an edge (elem_1D_line_buffer.F);
    * ``/LINE/BEAM``, ``/LINE/TRUSS``, ``/LINE/SPRING`` (M134) — direct 1D sets;
    * ``/LINE/BOX``, ``/LINE/CIRC``, ``/LINE/ALL`` (M134) — geometric/boundary lines.
    """

    id: int
    title: str = ""
    surf_ids: List[int] = field(default_factory=list)         # /LINE/SURF
    seg_nodes: List[List[int]] = field(default_factory=list)  # /LINE/SEG (user ids)
    edge_surf_ids: List[int] = field(default_factory=list)    # /LINE/EDGE (M37)
    line_ids: List[int] = field(default_factory=list)         # /LINE/LINE (M37)
    part_ids: List[int] = field(default_factory=list)         # /LINE/PART (M37)
    # M134 extensions:
    beam_ids: List[int] = field(default_factory=list)         # /LINE/BEAM
    truss_ids: List[int] = field(default_factory=list)        # /LINE/TRUSS
    spring_ids: List[int] = field(default_factory=list)       # /LINE/SPRING
    box_ids: List[int] = field(default_factory=list)          # /LINE/BOX
    circ_center: Optional[np.ndarray] = None                  # /LINE/CIRC center
    circ_radius: float = 0.0                                  # /LINE/CIRC radius
    circ_axis: Optional[np.ndarray] = None                    # /LINE/CIRC normal
    all_boundary: bool = False                                # /LINE/ALL
    # Resolved by the Starter: (nseg, 2) node indices + provenance
    # (same convention as Surface.seg_gtype/seg_elem).
    segments: Optional[np.ndarray] = None
    seg_gtype: Optional[np.ndarray] = None
    seg_elem: Optional[np.ndarray] = None


@dataclass
class EntityGroup:
    """An ELEMENT (or part) group: /GRSHEL, /GRSH3N, /GRBRIC, /GRTRUS,
    /GRBEAM, /GRSPRI, /GRQUAD and /GRPART (M37).

    Fortran origin: the IGRSH4N/IGRSH3N/IGRBRIC/... GROUP_ structures of
    ``groupdef_mod.F`` read by ``starter/source/groups/hm_lecgre.F``
    (direct element lists and parts) and ``hm_grogro.F`` (recursive
    group-of-groups with the same negative-id removal convention and
    iterative-fixpoint cycle detection as /GRNOD/GRNOD).

    ``family`` is the canonical element-family key ('SHEL', 'SH3N',
    'BRIC', 'QUAD', 'TRUS', 'BEAM', 'SPRI', 'PART' — GRBRIC covers ALL
    solids, bricks and tetras alike, exactly like IGRBRIC spans IXS).
    After resolution ``members`` lists (model element-group attribute,
    row indices) pairs — the port's dense equivalent of GROUP%ENTITY —
    and for family 'PART' ``part_ids_resolved`` holds the part ids.
    """

    id: int
    family: str
    title: str = ""
    # Unresolved content:
    elem_ids: List[int] = field(default_factory=list)   # direct element ids
    part_ids: List[int] = field(default_factory=list)   # /GR*/PART
    group_ids: List[int] = field(default_factory=list)  # group-of-groups, signed
    box_ids: List[int] = field(default_factory=list)    # /GR*/BOX (M136)
    surf_ids: List[int] = field(default_factory=list)   # /GR*/SURF (M136)
    # Resolved by the Starter:
    members: Optional[list] = None            # [(gtype attr, rows ndarray)]
    part_ids_resolved: Optional[list] = None  # family 'PART' only


@dataclass
class InitialVelocity:
    """/INIVEL/TRA: initial translational velocity on a node group.
    /INIVEL/AXIS (M5): initial *rotational* velocity field about an axis,
    v += omega * (d x (x0 - P)) — how a spinning body is initialized
    (Fortran: starter/source/initial_conditions/inivel/hm_read_inivel.F).

    kind='TRA' uses ``v``; kind='AXIS' uses ``omega``, ``axis`` (unit
    direction d) and ``origin`` (point P on the axis).

    ``frame_id`` / ``dir`` (M39): with a /FRAME the AXIS card's rotation
    runs about the frame's ``dir`` axis THROUGH THE FRAME ORIGIN, and its
    Vxt/Vyt/Vzt are components IN the frame — the Starter resolves both
    into ``axis``/``origin``/``v`` (hm_read_inivel.F 415-453 rotates Vt by
    the frame, 581-621 builds ``V = Vt + VR * (d x (X - O))``).  Without a
    frame the axis passes through the GLOBAL origin along the global
    ``dir``, which is that code's IFRA == 0 branch."""

    id: int
    grnod_id: int
    v: np.ndarray  # (3,)  (TRA)
    title: str = ""
    kind: str = "TRA"
    omega: float = 0.0
    axis: Optional[np.ndarray] = None    # (3,) unit vector (AXIS)
    origin: Optional[np.ndarray] = None  # (3,) point on the axis (AXIS)
    frame_id: int = 0                    # /FRAME (AXIS); 0 = global
    dir: int = 1                         # IDIR 1/2/3 = the frame's X'/Y'/Z'
    iskew: int = 0


@dataclass
class NodeMergeOption:
    """/MERGE/NODE, /MERGE (M130): Node group merge tolerance option."""
    id: int
    title: str = ""
    tol: float = 0.0
    grnod_id: int = 0
    merge_type: int = 0


@dataclass
class InivelAxis:
    """/INIVEL/AXIS (M112): Axisymmetric initial velocity around frame axis.

    Fortran origin: ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` / CFG ``inivel_axis.cfg``.
    """
    id: int
    title: str = ""
    dir: str = "Z"
    frame_id: int = 0
    grnod_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vr: float = 0.0
    tstart: float = 0.0
    sens_id: int = 0


@dataclass
class InivelFvm:
    """/INIVEL/FVM (M112): FVM airbag initial velocity on brick/quad/tria groups.

    Fortran origin: ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` / CFG ``inivel_fvm.cfg``.
    """
    id: int
    title: str = ""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    skew_id: int = 0
    tstart: float = 0.0
    sens_id: int = 0


@dataclass
class InivelNodeItem:
    """Single node entry for /INIVEL/NODE (M112)."""
    node_id: int
    skew_id: int = 0
    vxt: float = 0.0
    vyt: float = 0.0
    vzt: float = 0.0
    vxr: float = 0.0
    vyr: float = 0.0
    vzr: float = 0.0


@dataclass
class InivelNode:
    """/INIVEL/NODE (M112): Nodal vector initial velocities.

    Fortran origin: ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` / CFG ``inivel_node.cfg``.
    """
    id: int
    title: str = ""
    items: List[InivelNodeItem] = field(default_factory=list)


@dataclass
class InivelPart:
    """/INIVEL/PART (M137): Initial velocity on a part."""
    id: int
    title: str = ""
    part_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vr: float = 0.0
    skew_id: int = 0
    tstart: float = 0.0
    sens_id: int = 0


@dataclass
class InivelSph:
    """/INIVEL/SPH (M137): Initial velocity on SPH particles."""
    id: int
    title: str = ""
    grsph_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0


@dataclass
class SurfSurf:
    """``/SURF/SURF/id`` or ``/SURFSURF/id`` (M202): Surface composed of other surfaces or surface-to-surface interaction."""
    id: int = 0
    title: str = ""
    surf_ids: list[int] = field(default_factory=list)
    surf1_id: int = 0
    surf2_id: int = 0
    iflag: int = 0
    gap: float = 0.0
    fric: float = 0.0


BoxRect = Box
BoxCyl = Box
BoxSphere = Box
BoxCylin = Box
BoxSpher = Box
