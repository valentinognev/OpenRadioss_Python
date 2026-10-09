"""Task P2.3 — solid → 2-node degenerations (the ``sfor_n2s4`` family).

Fortran origin (all under ``engine/source/elements/solid/``)::

    solide/ssort_n4.F     the sort: classify which pair of a quad4 has merged
    solide/sfor_n2s4.F    hexa8 / wedge6 self-contact; calls ssort_n4
    solide/sfor_ns2s4.F90 the striking node against that quad (the HJ weights)
    solide/sfor_4n2s4.F90 four nodes striking a quad; drives sfor_ns2s4
    solide4/sfor_n2stria.F tetra4, the sfor_n2s3 entry point

Why this module exists
----------------------
An explicit solid can be crushed flat by its own load.  Upstream does not delete
it and does not stop integrating it — it *degenerates* it: the collapsed
element is re-described as a 2-node segment so it keeps carrying mass and keeps
absorbing energy instead of dividing by a vanishing area.  The branch structure
that decides which nodes survive lives in ``ssort_n4``.

``ssort_n4`` IS THE NODE RE-ORDERING
------------------------------------
It does not permute ``X1..X4``.  It writes an integer code — ``2`` (flattened),
``3``/``4``/``5``/``6`` (one pair merged) or ``0`` (already a line) — and every
downstream ``SELECT CASE`` re-orders the nodes by that code.  The ladder is
branch-for-branch and in the Fortran's own order, because the order *is* the
result: ``X4==X3`` is tested before ``X2==X1`` before ``X4==X1`` before
``X3==X2``, and the first branch hit ``CYCLE``s out.  Re-deriving a rule that
picks "the" merged pair on some other criterion gives the same *set* of nodes in
several cases and the wrong order in all of them — a segment with its ends
swapped, which upstream's own plan calls an infinite energy engine.

A note on the two ``IFC == 0`` line collapses
--------------------------------------------
The ladder reaches code ``0`` from two branches, and upstream lets both stand:
``X4==X3`` followed by ``X2==X1`` (the line runs 1 → 3) and ``X4==X1`` followed
by ``X3==X2`` (the line runs 1 → 2).  Which one it was is not recoverable from
the code alone, which is exactly why :func:`line_pair` walks the same ladder
rather than inferring the pair from ``0``.  Upstream only needs the code because
it *skips* those elements; a port that has to reduce them needs the pair too.

Deviation from upstream (one, deliberate)
-----------------------------------------
``sfor_ns2s4`` divides by ``S2``, the squared area of the triangle it projects
onto, with no guard.  This port uses ``max(EM20, S2)``, the same
``one/max(em20, ...)`` idiom the routine itself uses two loops earlier to
normalize the normal.  It changes nothing upstream computes and keeps a collinear
quad from producing ``inf`` weights.

What is NOT here
----------------
The forces.  ``sfor_n2s4`` and ``sfor_ns2s4`` go on to build ``FN``, book
``E_DISTOR`` and scatter into ``FOR_T``/``FORC_N``; that is the distortion-energy
path, which reads the element's stiffness history and is a different task.  This
module owns the topology and the weights that path consumes.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from ..common.constants import EM20

#: ``ssort_n4`` codes.  ``IFC_MERGE_LINE`` is 0 because upstream reuses 0 both
#: for "already on a line" and for "caller had not set a code yet".
IFC_MERGE_LINE = 0
IFC_FLAT = 2
IFC_MERGE_34 = 3
IFC_MERGE_14 = 4
IFC_MERGE_23 = 5
IFC_MERGE_12 = 6

#: ``sfor_4n2s4``'s ``ITGSUB``: the quad has collapsed onto a line or a point,
#: so there is no face left to strike.
ITGSUB_LINE = -1

#: ``sfor_ns2s4.F90:119`` — the ``(XB, XC, XA)`` triangle each code selects.
#: Codes 1-4 mean "striking node k is near quad node k"; 12/23/34/14 mean the
#: quad has already lost that pair.  This is the routine's own comment line,
#: transcribed; the code below writes the same coordinates in the same order.
_TRIANGLES: dict[int, tuple[int, int, int]] = {
    1: (1, 2, 4),
    2: (2, 3, 1),
    3: (3, 4, 2),
    4: (4, 1, 3),
    34: (1, 2, 3),
    14: (2, 3, 4),
    23: (4, 1, 2),
    12: (3, 4, 1),
}

#: ``sfor_ns2s4.F90:294-335`` — the node each of ``(LA, LB, LC)`` lands on.
#: The node the triangle drops gets zero, so the four weights still sum to one.
#:
#: The tuple is ``(node for LA, node for LB, node for LC)``, transcribed from the
#: eight ``SELECT CASE`` branches — not derived from the triangle.  The two differ:
#: code 3 puts ``LA`` on node 1 while its triangle ``3-4-2`` has ``XA`` at node 2,
#: so ``sum(hj_j * X_j)`` is *not* the striking point for every code.  That is
#: upstream's formulation, and a port that "fixes" it stops matching the Fortran.
_HJ_SLOTS: dict[int, tuple[int, int, int]] = {
    1: (4, 1, 2),     # hj1=LB, hj2=LC, hj4=LA, hj3=0
    2: (1, 2, 3),     # hj1=LA, hj2=LB, hj3=LC, hj4=0
    3: (1, 3, 4),     # hj1=LA, hj3=LB, hj4=LC, hj2=0
    4: (3, 1, 4),     # hj3=LA, hj1=LC, hj4=LB, hj2=0
    34: (3, 1, 2),    # hj3=LA, hj1=LB, hj2=LC, hj4=0
    14: (4, 2, 3),    # hj4=LA, hj2=LB, hj3=LC, hj1=0
    23: (2, 1, 4),    # hj2=LA, hj1=LC, hj4=LB, hj3=0
    12: (1, 3, 4),    # hj1=LA, hj3=LB, hj4=LC, hj2=0
}

#: Default relative tolerance for "every node of this element lies on the
#: surviving line".  Upstream decides coincidence with an exact ``DMIN == ZERO``;
#: a reduction that then has to *place mass* cannot use exact arithmetic on
#: coordinates a deck wrote in decimal, so the on-line test is scale-relative.
_DEFAULT_ON_LINE_TOL = 1e-9


class LineReduction(NamedTuple):
    """A solid collapsed onto a line, as the 2-node segment it becomes."""

    ids: np.ndarray       # (2,) 1-based node numbers, in the ladder's own order
    weights: np.ndarray   # (2,) mass fractions; they sum to 1
    code: int             # the ``ssort_n4`` code the ladder returned


class Striking(NamedTuple):
    """The four ``ns = n1 .. n4`` passes of ``sfor_4n2s4`` for one quad."""

    itgsub: int             # the quad's own collapse code, or ``ITGSUB_LINE``
    struck: np.ndarray      # (4,) bool, one per striking node
    index: np.ndarray       # (4,) the striking node number, 1..4
    itg: np.ndarray         # (4,) the triangle code each pass used
    weights: np.ndarray     # (4, 4) barycentric weights on the quad's nodes
    pene: np.ndarray        # (4,) penetration each pass measured


# ---------------------------------------------------------------------------
# the sort
# ---------------------------------------------------------------------------
def _dmin(quad: np.ndarray, i: int, j: int) -> np.ndarray:
    """``ABS(Xj - Xi)`` summed over the axes — the Fortran's ``DMIN``.

    ``ssort_n4`` and ``sfor_4n2s4`` both test ``DMIN == ZERO``, i.e. *exact*
    coincidence, not a tolerance.  Transcribed as written.
    """
    d = quad[:, j - 1] - quad[:, i - 1]
    return np.abs(d).sum(axis=-1)


def _ladder(quad: np.ndarray):
    """The pair ladder of ``ssort_n4`` (its second loop), over a batch.

    Returns ``(code, pair)``: ``code`` is the ``IFC1`` the Fortran writes and
    ``pair`` is the 2-node line it collapsed onto, as node numbers ``(p, q)``
    with ``p < q``, or ``(-1, -1)`` when the quad is not on a line.
    """
    code = np.full(len(quad), IFC_MERGE_LINE, dtype=np.int64)
    pair = np.full((len(quad), 2), -1, dtype=np.int64)

    d34 = _dmin(quad, 4, 3)
    d12 = _dmin(quad, 2, 1)
    d14 = _dmin(quad, 4, 1)
    d23 = _dmin(quad, 3, 2)

    # branch 1 — X4 == X3
    hit = d34 == 0.0
    code = np.where(hit, IFC_MERGE_34, code)
    line = hit & (d12 == 0.0)
    code = np.where(line, IFC_MERGE_LINE, code)
    pair[line] = (1, 3)                    # 1 == 2 and 4 == 3

    # branch 2 — X2 == X1
    hit = (d34 != 0.0) & (d12 == 0.0)
    code = np.where(hit, IFC_MERGE_12, code)

    # branch 3 — X4 == X1
    hit = (d34 != 0.0) & (d12 != 0.0) & (d14 == 0.0)
    code = np.where(hit, IFC_MERGE_14, code)
    line = hit & (d23 == 0.0)
    code = np.where(line, IFC_MERGE_LINE, code)
    pair[line] = (1, 2)                    # 4 == 1 and 3 == 2

    # branch 4 — X3 == X2
    hit = (d34 != 0.0) & (d12 != 0.0) & (d14 != 0.0) & (d23 == 0.0)
    code = np.where(hit, IFC_MERGE_23, code)

    return code, pair


def ssort_n4(quad: np.ndarray, xi: np.ndarray, *, ifc: np.ndarray,
             marge: np.ndarray, stif: np.ndarray) -> np.ndarray:
    """``ssort_n4.F`` — the sort, both loops, over a batch of quads.

    Parameters
    ----------
    quad : (n, 4, 3) float
        The quad's four nodes.
    xi : (n, 3) float
        The striking point each quad is measured against.
    ifc : (n,) int
        The caller's code.  A positive value skips both loops (``IFC1(I)>0
        CYCLE``); zero enters them.
    marge : (n,) float
        The flattening margin of loop 1.
    stif : (n,) float
        The nodal stiffness.  Loop 1 needs ``STIF > ZERO`` to flatten; loop 2
        does not consult it.

    Returns
    -------
    (n,) int — the ``IFC1`` codes: the caller's, ``2`` for a quad flattened
    inside MARGE, ``3``/``4``/``5``/``6`` for a merged pair, ``0`` for a quad
    already on a line.
    """
    quad = np.asarray(quad, dtype=np.float64)
    ifc = np.asarray(ifc, dtype=np.int64).copy()

    # --- loop 1: flattened inside MARGE (ssort_n4.F:64-80)
    r = quad[:, 1] + quad[:, 2] - quad[:, 0] - quad[:, 3]
    s = quad[:, 2] + quad[:, 3] - quad[:, 0] - quad[:, 1]
    n = np.cross(r, s)
    scale = 1.0 / np.maximum(EM20, np.sqrt((n * n).sum(axis=-1)))
    # PENE = ABS(((X3 - XI) . N) * NORM)
    pene1 = np.abs(((quad[:, 2] - xi) * n).sum(axis=-1) * scale)
    flat = (ifc <= 0) & (pene1 < np.asarray(marge)) & (np.asarray(stif) > 0.0)
    ifc = np.where(flat, IFC_FLAT, ifc)

    # --- loop 2: the pair ladder (ssort_n4.F:83-127), code 2 excluded
    todo = ifc == 0
    if todo.any():
        code, _ = _ladder(quad[todo])
        ifc[todo] = code
    return ifc


def line_pair(quad: np.ndarray) -> Optional[tuple[int, int]]:
    """The 2-node line a quad4 has collapsed onto, or ``None``.

    Walks the ladder of :func:`_ladder` and reports the pair it chose, so the
    two ways of reaching code ``0`` stay distinguishable — upstream does not
    need them to, because it skips those quads.
    """
    _, pair = _ladder(np.asarray(quad, dtype=np.float64)[None, :, :])
    p, q = pair[0]
    return None if p < 0 else (int(p), int(q))


# ---------------------------------------------------------------------------
# the reductions
# ---------------------------------------------------------------------------
def _reduce(nodes: np.ndarray, tol: float) -> Optional[LineReduction]:
    """Reduce a solid to the 2-node segment its bottom face has collapsed onto.

    The ladder always inspects four nodes: for a quad that is the whole element,
    and for a wedge or a hexa it is the bottom face 1-2-3-4, which is the same
    quad in Radioss order (see ``solid_hexa8``: nodes 1-4 bottom, 5-8 top).  The
    weights, though, are measured over *every* node the caller passed.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    code, pair = _ladder(nodes[:4][None, :, :])
    p, q = (int(v) for v in pair[0])
    if p < 0:
        return None

    xa, xb = nodes[p - 1], nodes[q - 1]
    d = xb - xa
    length2 = float(d @ d)
    if length2 < EM20:                 # every node in one place: a point, not a line
        return None
    length = float(np.sqrt(length2))

    # Every node must lie on that line, or the element is a solid with a flat
    # face, not a solid collapsed onto a line.
    offset = nodes - xa
    perp = np.linalg.norm(np.cross(np.broadcast_to(d, offset.shape), offset), axis=-1)
    if perp.max() > tol * length:
        return None

    # Lumped mass projected onto the segment: total mass is the weights' sum and
    # the centroid is reproduced wherever the element really is.  For every
    # collapse the ladder can detect the two coincident groups are equal in size,
    # so this is a half-and-half split — but it is computed, not assumed, and a
    # hand-written 0.5 moves the centroid of a hexa whose top face survived.
    t = (offset @ d) / length2
    w_hi = float(t.mean())
    return LineReduction(ids=np.array([p, q]),
                         weights=np.array([1.0 - w_hi, w_hi]),
                         code=int(code[0]))


def four_to_two(nodes: np.ndarray, *, tol: float = _DEFAULT_ON_LINE_TOL
                ) -> Optional[LineReduction]:
    """A quad4 collapsed onto a line becomes its 2-node segment.

    ``None`` when the quad is intact, or when its four nodes sit in one place —
    that is a point, and a zero-length segment would divide by zero everywhere
    downstream.
    """
    return _reduce(nodes, tol)


def six_to_two(nodes: np.ndarray, *, tol: float = _DEFAULT_ON_LINE_TOL
               ) -> Optional[LineReduction]:
    """A 6-node wedge collapsed onto a line becomes its 2-node segment.

    The ladder is run on the bottom quad 1-2-3-4, which in Radioss wedge order
    is the face whose collapse decides the element; the weights are then taken
    over all six nodes, so nodes 5 and 6 are counted in the mass.
    """
    return _reduce(nodes, tol)


def eight_to_two(nodes: np.ndarray, *, tol: float = _DEFAULT_ON_LINE_TOL
                 ) -> Optional[LineReduction]:
    """An 8-node hexa collapsed onto a line becomes its 2-node segment.

    Same face, same ladder; the weights are taken over all eight nodes, so the
    result carries the element's total mass and reproduces its centroid.
    """
    return _reduce(nodes, tol)


# ---------------------------------------------------------------------------
# four_to_four_striking — sfor_4n2s4.F90 driving sfor_ns2s4.F90
# ---------------------------------------------------------------------------
def itgsub(quad: np.ndarray, *, ifc: int = 1, ll: Optional[float] = None) -> int:
    """``sfor_4n2s4.F90:118-167`` — which pair of the struck quad has merged.

    ``-1`` (``ITGSUB_LINE``) when the quad is already on a line or a point, when
    its characteristic length has vanished, or when the caller's ``IFC1`` is
    zero; ``0`` when nothing has merged; otherwise the merged pair read as a two
    digit number — 34, 12, 14 or 23 — in the branch order the Fortran tests
    them.
    """
    quad = np.asarray(quad, dtype=np.float64)[None, :, :]
    if ifc == 0 or (ll is not None and ll < EM20):
        return ITGSUB_LINE

    d34 = _dmin(quad, 4, 3)[0]
    d12 = _dmin(quad, 2, 1)[0]
    d14 = _dmin(quad, 4, 1)[0]
    d23 = _dmin(quad, 3, 2)[0]

    if d34 == 0.0:
        return ITGSUB_LINE if d12 == 0.0 else 34
    if d12 == 0.0:
        return 12
    if d14 == 0.0:
        return ITGSUB_LINE if d23 == 0.0 else 14
    if d23 == 0.0:
        return 23
    return 0


def triangle_nodes(code: int) -> np.ndarray:
    """The ``(XB, XC, XA)`` node numbers a striking code re-orders the quad into.

    ``sfor_ns2s4.F90:119-200``.  The order matters: ``LA``/``LB``/``LC`` are the
    barycentric coordinates of ``XA``/``XB``/``XC`` in that order, so transposing
    a pair here silently re-points the weights.
    """
    return np.array(_TRIANGLES[int(code)], dtype=np.int64)


def _ifde_ladder(striking: np.ndarray) -> np.ndarray:
    """``sfor_4n2s4.F90:187-229`` — the same ladder run on the striking quad.

    Codes ``2``/``3``/``4`` name a merged pair; the negative codes ``-2``/``-3``/
    ``-4`` mark a striking group that has collapsed onto a line or a point, and
    disqualify the passes that would strike through it.
    """
    code = np.zeros(len(striking), dtype=np.int64)
    d34 = _dmin(striking, 4, 3)
    d12 = _dmin(striking, 2, 1)
    d14 = _dmin(striking, 4, 1)
    d23 = _dmin(striking, 3, 2)

    hit = d34 == 0.0
    code = np.where(hit, 4, code)
    code = np.where(hit & (d12 == 0.0), -2, code)

    hit = (d34 != 0.0) & (d12 == 0.0)
    code = np.where(hit, 2, code)

    hit = (d34 != 0.0) & (d12 != 0.0) & (d14 == 0.0)
    code = np.where(hit, 4, code)
    code = np.where(hit & (d23 == 0.0), -3, code)

    hit = (d34 != 0.0) & (d12 != 0.0) & (d14 != 0.0) & (d23 == 0.0)
    code = np.where(hit, 3, code)
    return code


def _barycentric(xa, xb, xc, xi):
    """``sfor_ns2s4.F90:226-257`` — ``(LA, LB, LC)`` of ``xi`` on the triangle.

    Transcribed, including the sign convention of the edge cross product, so
    the coordinates come out the way the routine's own ``HJ`` mapping expects.
    The one addition is the ``max(EM20, S2)`` guard noted in the module
    docstring.
    """
    xab, yab, zab = xb - xa
    xca, yca, zca = xa - xc

    xia, yia, zia = xa - xi
    xib, yib, zib = xb - xi
    xic, yic, zic = xc - xi

    sx = -yab * zca + zab * yca
    sy = -zab * xca + xab * zca
    sz = -xab * yca + yab * xca
    s2 = sx * sx + sy * sy + sz * sz

    sax = yib * zic - zib * yic
    say = zib * xic - xib * zic
    saz = xib * yic - yib * xic
    la = (sx * sax + sy * say + sz * saz) / max(EM20, s2)

    sbx = yic * zia - zic * yia
    sby = zic * xia - xic * zia
    sbz = xic * yia - yic * xia
    lb = (sx * sbx + sy * sby + sz * sbz) / max(EM20, s2)

    return la, lb, 1.0 - la - lb


def _clamp(la: float, lb: float, lc: float):
    """``sfor_ns2s4.F90:259-290`` — pull negative barycentric coordinates back
    onto the triangle, in the routine's own branch order."""
    if la < 0.0:
        if lb < 0.0:
            return 0.0, 0.0, 1.0
        if lc < 0.0:
            return 0.0, 1.0, 0.0
        aaa = lb + lc
        return 0.0, lb / aaa, lc / aaa
    if lb < 0.0:
        if lc < 0.0:
            return 1.0, 0.0, 0.0
        aaa = lc + la
        return la / aaa, 0.0, lc / aaa
    if lc < 0.0:
        aaa = la + lb
        return la / aaa, lb / aaa, 0.0
    return la, lb, lc


def four_to_four_striking(quad: np.ndarray, striking: np.ndarray, *,
                          marge: float, penmin: float = 0.0,
                          stif0: float = 1.0, ll: Optional[float] = None) -> Striking:
    """``sfor_4n2s4.F90`` driving ``sfor_ns2s4.F90``, for one quad and four nodes.

    The four ``ns = n1 .. n4`` passes are kept independent, as upstream runs
    them: each gets its own ``IFC2``, and the ``ITGSUB`` a pass borrows is handed
    back afterwards, so a node that strikes does not change the next node's
    triangle.

    Parameters
    ----------
    quad : (4, 3) float
        The quad being struck, counter-clockwise 1-2-3-4.
    striking : (4, 3) float
        The four nodes striking it.
    marge : float
        A striking node within MARGE of the quad's plane is a candidate.
    penmin : float
        Minimum penetration; the depth is counted only up to it.
    stif0 : float
        Nodal stiffness; at or below zero nothing strikes.
    ll : float, optional
        The quad's characteristic length.  Below ``EM20`` the quad is treated as
        already degenerate, as upstream does.

    Returns
    -------
    Striking — per passing node: whether it struck, the triangle code the pass
    used, its barycentric weights on the quad's four nodes, and the penetration
    it measured.
    """
    quad = np.asarray(quad, dtype=np.float64)
    striking = np.asarray(striking, dtype=np.float64)

    codes = np.zeros(4, dtype=np.int64)
    struck = np.zeros(4, dtype=bool)
    weights = np.zeros((4, 4), dtype=np.float64)
    pene = np.zeros(4, dtype=np.float64)

    code = itgsub(quad, ifc=1, ll=ll)
    if code == ITGSUB_LINE:
        return Striking(itgsub=code, struck=struck,
                        index=np.arange(1, 5, dtype=np.int64),
                        itg=codes, weights=weights, pene=pene)

    # the quad's unit normal (sfor_4n2s4.F90:169-185)
    r = quad[1] + quad[2] - quad[0] - quad[3]
    s = quad[2] + quad[3] - quad[0] - quad[1]
    n = np.cross(r, s)
    n = n * (1.0 / max(EM20, float(np.sqrt(n @ n))))

    ifde = int(_ifde_ladder(striking[None, ...])[0])

    for k in range(4):
        # IF (ITGSUB(I)==-1 .OR. ABS(IFDE_S(I))==K) CYCLE — pass 1 has no such
        # gate upstream, and neither does it get one here.
        if k > 0 and abs(ifde) == k + 1:
            continue
        # IF (DMIN==ZERO) CYCLE — an exactly merged node cannot strike.
        if float(np.abs(striking[k] - quad[k]).sum()) == 0.0:
            continue

        dn = abs(float(n @ (striking[k] - quad[k])))
        if not (dn < marge and stif0 > 0.0):
            continue

        # IF (ITGSUB(I)==0) ITGSUB(I)=K — on an intact quad the pass number is
        # the triangle code.
        use = code if code != 0 else k + 1
        codes[k] = use

        xb, xc, xa = (quad[i - 1] for i in triangle_nodes(use))
        # PENE = MAX(0, -((XB - XI).N - PENMIN))   (sfor_ns2s4.F90:220-222)
        pene[k] = max(0.0, penmin - float(n @ (xb - striking[k])))
        if pene[k] == 0.0:
            continue

        la, lb, lc = _clamp(*_barycentric(xa, xb, xc, striking[k]))
        hj = np.zeros(4, dtype=np.float64)
        i_la, i_lb, i_lc = _HJ_SLOTS[use]
        hj[i_la - 1], hj[i_lb - 1], hj[i_lc - 1] = la, lb, lc
        weights[k] = hj
        struck[k] = True

    return Striking(itgsub=code, struck=struck,
                    index=np.arange(1, 5, dtype=np.int64),
                    itg=codes, weights=weights, pene=pene)
