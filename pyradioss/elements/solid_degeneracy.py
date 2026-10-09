"""Task P2.6 — degeneracy detection and the length correction it buys.

Fortran origin (all under ``engine/source/elements/solid/solide/`` and
``engine/source/elements/thickshell/solidec/``)::

    solide/degenes8.F     the count: how many of a brick's 8 nodes repeat
    solide/idege8.F       one degenerate quad: is a corner pair coincident,
                          and how big is the quad (A = |R x S|^2)
    solide/idege.F        the same scan, vectorized over the 6 faces
    solide/sdlen_dege.F   the FAC ladder and the DELTAX/LAT correction
    solide/sldege.F       the same correction keyed on the face areas
    solide/nodedege.F     append a position unless it is already there
    solide/deges4v.F      the fully collapsed case: the tetra of the first
                          four DISTINCT positions
    solide/tetra4v.F      |det| / 6
    thickshell/solidec/dim_tshedg.F    the degenerate-direction edge list
    thickshell/solidec/ind_tshedg.F    the same list, with the node pairs
    thickshell/solidec/tshcdcom_dim.F  which elements touch those edges
    thickshell/solidec/tshcdcom_ini.F  the element numbers behind that mask

Why this module exists
----------------------
Two different things are called "degenerate" in this tree, and confusing
them is the bug this module is written to prevent.

**The connectivity mark.**  ``starter/initialization.py`` runs a /BRICK with
5, 6 or 7 distinct node ids through the *same* 8-node kernel, connectivity as
written, repeats included — that is upstream's own way of writing a wedge or
a pyramid as a brick (HEXA_DEGE, RD_V_0240's ``Modele_HEXA_P14``).  A brick
with 4 distinct nodes is moved to /TETRA4 by the Starter, so it never reaches
the brick kernel at all.  :func:`detect` speaks exactly that language:
:func:`degenes8` is the repeat count, and ``detect`` is true precisely on
the elements the Starter chose to run as collapsed hexas.

**The geometry mark.**  A deck can also flatten a face by giving every corner
its own node, so no id repeats and the connectivity mark is silent.  That is
what ``nodedege`` is for: upstream compares POSITIONS, not ids, everywhere it
needs to know that two corners are really one point.

:func:`detect` takes the union, so it can never contradict the Starter — it
can only catch a collapse the Starter's id test cannot see.  It deliberately
does **not** test the Jacobian or the volume: a brick crushed flat is not a
different element, it is the same brick at a different time, and upstream
never deletes it.

Why nothing is deleted
----------------------
Upstream has no "delete a degenerate solid" rule.  ``check_solid_geometric_
erosion`` (``sgeodel3.F``, already ported in ``engine/element_erosion.py``)
is the only solid deletion criterion and it is a *property*-driven one
(``/DEL`` col_min / defv_min / asp_max), not a degeneracy one.  What upstream
does with a degenerate brick is correct its characteristic length, because
``lc = V / A_max`` goes to zero as the brick flattens and a zero ``lc`` is a
zero time step.  :func:`degeneracy_factors` and :func:`characteristic_length`
are that correction.  Turning the mask into ``OFF = 0`` is the element
cycle's job and no caller exists yet.

The two length ladders, and which one is here
---------------------------------------------
``sdlen3.F`` computes ``DELTAX = 4*VOL/sqrt(AREAM)`` where ``AREAM`` is the
largest face area — and because ``slen.F``'s ``AREA`` is ``|R x S|^2`` with
``R = P13 - P24`` and ``S = P13 + P24``, i.e. ``(4*area)^2``, that is exactly
``VOL / A_max``.  Two different guards can overwrite ``AREAM``:

* ``sldege.F`` (guarded by ``IDTS6 > 0``) keys on the face AREAS:
  ``IDEG = #{faces with AREA < EM30}``, and any degenerate element gets
  ``AREAM = EM20`` and a fresh ladder.
* ``sdlen_dege.F`` keys on the CONNECTIVITY: it calls ``degenes8`` and
  corrects only ``IDEGE > 0`` elements.

:func:`degeneracy_factors` ports the **connectivity** one (``sdlen_dege``),
because it is the one that agrees with the Starter's mark and therefore with
:func:`detect`.  The difference that matters: ``sdlen_dege`` seeds ``AMAX``
with ``EM20``, so the FIRST of the six faces always wins and a later face
only displaces it when its ``A`` is strictly larger; ``sldege`` seeds it with
face 1's area and adds the ``IT4``/``deges4v`` fallback for the fully
collapsed case.  Both are noted in the docstrings of the helpers.

The FAC ladder (``sdlen_dege.F:120-131``)
    ``IDEGE > 2 -> 1/9``, ``IDEGE == 2 -> 1/4``, ``IDEGE == 1 -> 1``.  It is
    applied to ``AREAM`` only when ``IT == 0``, i.e. when the winning face is
    NOT itself degenerate (``IDEGE8`` sets ``IT = IDE``, the flag of the face
    that beat ``AMAX``).  Since ``sqrt(FAC)`` lands in the denominator,
    ``FAC = 1/4`` doubles the length and ``FAC = 1/9`` triples it — the same
    2x / 3x the engine's ``solid_hexa8.init_group`` applies through
    ``lc_scale``.  ``init_group`` derives ``lc_scale`` from ``IDEGE`` alone
    and cannot see the geometry; :func:`characteristic_length` can, and is the
    faithful one.  ``tests/test_p2_solid_degeneracy.py`` holds the two counts
    to each other.

What is NOT here
----------------
The forces, and the moment the element is finally deleted.  The unslaved-edge
consequences (``ALPHA_DC``, the whole ``tshcdcom`` -> ``SHCDCOM`` chain that
follows ``IENUNL`` in ``resol_init.F``) belong to the thickshell/solid-shell
path and are wired when an element cycle consumes this module.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..common.constants import EM20

#: ``ONE_OVER_9`` / ``FOURTH`` / ``THIRD`` as spelled in ``sdlen_dege.F`` and
#: ``sldege.F``.
ONE_OVER_9 = 1.0 / 9.0
FOURTH = 0.25
THIRD = 1.0 / 3.0
FOUR = 4.0
ONE = 1.0

#: The six faces of a brick, in the order ``sdlen3.F`` scans them (and the
#: order ``idege.F`` is called in from ``sldege.F``):
#: ``(1,2,3,4) (5,6,7,8) (1,2,6,5) (2,3,7,6) (3,4,8,7) (4,1,5,8)``,
#: 0-based here.  The ORDER is load-bearing: ``IDEGE8`` keeps the first face
#: it is given (``AMAX`` starts at ``EM20``) and only a strictly larger ``A``
#: displaces it.
_HEXA_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
], dtype=np.int64)

#: ``ind_tshedg.F``'s ``JHBE == 15`` branch — the four "unslave" directions of
#: an 8-node solid: node ``i`` against node ``i+4``.  (The ``JHBE == 14``
#: branch is the solid-shell stress-strain variants ``ICSTR`` 100/10/1 and the
#: ``ISOLNOD == 6`` branch is the 6-node wedge; neither is a brick.)
_UNSLAVE_PAIRS = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)


# ---------------------------------------------------------------------------
# small accessors
# ---------------------------------------------------------------------------

def _conn(mesh) -> np.ndarray:
    """Connectivity of whatever ``mesh`` is: an ``(n, k)`` index array or
    anything carrying a ``.conn`` (an :class:`ElementGroup`)."""
    conn = getattr(mesh, "conn", mesh)
    conn = np.asarray(conn)
    if not np.issubdtype(conn.dtype, np.integer):
        raise TypeError("connectivity must be an integer node-index array, "
                        f"got dtype {conn.dtype}")
    if conn.ndim == 1:
        conn = conn[None, :]
    return conn


def _coords(group, x=None) -> np.ndarray:
    """Nodal coordinates: the explicit ``x``, else the group's own model."""
    if x is not None:
        return np.asarray(x, dtype=float)
    model = getattr(group, "_model", None)
    if model is None or getattr(model, "x", None) is None:
        raise ValueError("no coordinates: pass x= or build the group with "
                         "solid_hexa8.init_group")
    return np.asarray(model.x, dtype=float)


def _rows(conn: np.ndarray, ids) -> np.ndarray:
    conn = _conn(conn)
    if ids is None:
        return np.arange(conn.shape[0])
    return np.asarray(ids, dtype=np.int64).reshape(-1)


# ---------------------------------------------------------------------------
# degenes8 -- the connectivity mark
# ---------------------------------------------------------------------------

def degenes8(group) -> np.ndarray:
    """Number of repeated nodes of each brick (``degenes8.F``, ``IDEGE``).

    The Fortran's whole body is "for each node, is it repeated elsewhere in
    ``IXS(1..8)?" summed and then halved::

        IDEGE(I) = 0
        DO J=1,8
          NC(J)=IXS(J+1,I)
        ENDDO
        DO J=1,8
          NJ=NC(J) ; NC(J)=0
          IF (INTAB(8,NC,NJ)) IDEGE(I)=IDEGE(I)+1
          NC(J)=NJ
        ENDDO
        IDEGE(I)=IDEGE(I)/2

    The halving is not a fudge: a repeated node is found once from each slot it
    occupies, so a node written twice contributes 2 and the count is the number
    of *pairs*.  Integer division is upstream's, so a node written three times
    gives ``IDEGE = 1`` (``3/2``), not 2.  That count is what the Starter's
    collapsed-hexa mark turns on — ``distinct ids in (5, 6, 7)`` — and it is
    the same count ``solid_hexa8.init_group`` recomputes inline to build
    ``lc_scale``.

    Returns:
        ``(n,)`` int64 array.  ``0`` on an ordinary brick.
    """
    conn = _conn(group)
    if conn.size == 0:
        return np.zeros(conn.shape[0], dtype=np.int64)
    eq = conn[:, :, None] == conn[:, None, :]
    diag = np.arange(conn.shape[1])
    eq[:, diag, diag] = False
    # "is it repeated elsewhere": a node counts once no matter how many extra
    # slots it also occupies, which is INTAB's test.
    return (eq.any(axis=2).sum(axis=1) // 2).astype(np.int64)


def distinct_positions(xe: np.ndarray) -> np.ndarray:
    """How many of each element's nodes sit on a position already taken.

    ``nodedege.F``'s test, batched: it appends a point unless one of the
    points already collected is at *exactly* the same coordinates — the
    Fortran's ``XIJ == ZERO .AND. YIJ == ZERO .AND. ZIJ == ZERO``, with no
    tolerance, and none here either.
    """
    xe = np.asarray(xe, dtype=float)
    if xe.ndim == 2:
        xe = xe[None, :, :]
    same = (xe[:, :, None, :] == xe[:, None, :, :]).all(axis=3)
    seen = np.zeros(same.shape[:2], dtype=bool)
    count = np.zeros(xe.shape[0], dtype=np.int64)
    for k in range(xe.shape[1]):
        count += ~(seen & same[:, k, :]).any(axis=1)
        seen |= same[:, k, :]
    return count


def detect(group, x=None) -> np.ndarray:
    """Which elements of ``group`` are degenerate — ``(n,)`` bool.

    True on exactly the elements the Starter runs as collapsed hexas
    (``degenes8 > 0``, i.e. 5, 6 or 7 distinct node ids), plus — when
    coordinates are supplied — the elements that are degenerate in geometry
    alone, which is ``nodedege``'s case: 8 distinct node ids that share
    positions.

    The union is deliberate and one-directional.  The Starter's mark can only
    ever be a subset, so this function can never contradict a Starter that
    already ran; it only catches a collapse the id test cannot see.  What it
    must never become is a *different* rule (a Jacobian or volume threshold),
    because then a Starter that marked an element collapsed would leave the
    Engine looking at a flag it never sets.
    """
    degen = degenes8(group) > 0
    if x is not None:
        conn = _conn(group)
        if conn.size:
            degen = degen | (distinct_positions(_coords(group, x)[conn])
                             < conn.shape[1])
    return degen


# ---------------------------------------------------------------------------
# idege8 / idege -- the per-face degenerate-quad scan
# ---------------------------------------------------------------------------

def idege8_area(xq: np.ndarray) -> np.ndarray:
    """``IDEGE8``'s ``A`` — a SQUARED area — for a batch of quads.

    ``idege8.F``::

        RX = X2+X3-X1-X4   SX = X3+X4-X1-X2
        NX = RY*SZ - RZ*SY           (N = R x S)
        A  = NX*NX+NY*NY+NZ*NZ       "A: (2*AREA)^2"

    The comment in the Fortran is off by a factor of four: with
    ``R = P13 - P24`` and ``S = P13 + P24``, ``R x S = -2 P13 x P24`` and
    ``|P13 x P24| = 2*area``, so ``A = (4*area)^2``.  That is the same
    normalization ``slen.F`` uses, which is why ``sdlen3.F``'s
    ``4*VOL/sqrt(AREAM)`` comes out as ``VOL/A_max``.  The Fortran wins over
    its own comment; this is ``|R x S|^2``.

    Args:
        xq: ``(m, 4, 3)`` quad corners, in ``(n1, n2, n3, n4)`` order.

    Returns:
        ``(m,)`` float array.
    """
    xq = np.asarray(xq, dtype=float)
    if xq.ndim == 2:
        xq = xq[None, :, :]
    r = xq[:, 1] + xq[:, 2] - xq[:, 0] - xq[:, 3]
    s = xq[:, 2] + xq[:, 3] - xq[:, 0] - xq[:, 1]
    n = np.cross(r, s)
    return np.einsum("ij,ij->i", n, n)


def idege8_flag(xq: np.ndarray) -> np.ndarray:
    """``IDEGE8``'s ``IDE`` — is this quad degenerate on its own?

    The Fortran walks the quad's four PERIMETER edges and stops at the first
    coincidence (``X12``, ``X23``, ``X34``, ``X41``); it does not compare all
    six pairs, and neither does this.  Exact equality, as in the Fortran.
    """
    xq = np.asarray(xq, dtype=float)
    if xq.ndim == 2:
        xq = xq[None, :, :]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        if np.any((xq[:, a] == xq[:, b]).all(axis=1)):
            return np.ones(xq.shape[0], dtype=np.int64)
    return np.zeros(xq.shape[0], dtype=np.int64)


def _idege8_scan(amax: np.ndarray, fac: np.ndarray, it: np.ndarray,
                 xq: np.ndarray):
    """One vectorized ``IDEGE8`` call over a batch of quads.

    ``IDEGE8`` takes ``AMAX`` and ``IT`` in/out and only ever *raises* the
    running maximum::

        IF (A > AMAX) THEN ; IT = IDE ; AMAX = A ; END IF

    ``FAC`` is declared but never assigned in ``IDEGE8`` — it passes through
    unchanged.  ``sdlen_dege.F`` relies on that: the ladder value survives the
    six calls and only ``IT`` (the winning face's own flag) changes.
    """
    a = idege8_area(xq)
    win = a > amax
    amax = np.where(win, a, amax)
    it = np.where(win, idege8_flag(xq), it)
    return amax, fac, it


# ---------------------------------------------------------------------------
# nodedege / deges4v / tetra4v -- the fully collapsed case
# ---------------------------------------------------------------------------

def nodedege(xyz: np.ndarray, point, n: int) -> int:
    """``nodedege.F``: append ``point`` to ``xyz[:, :n]`` unless it is already
    there, and return the new count.

    The Fortran is a two-line loop over the points collected so far and
    ``RETURN``s on the first exact coincidence — so ``n`` grows only when the
    point is genuinely new.  Kept in that shape (rather than vectorized over a
    whole element) because ``deges4v`` needs the early ``GOTO`` on reaching
    four distinct positions, which is the order-dependence the routine exists
    for: the tetra it builds is the one formed by the FIRST four distinct
    corners, in the order they appear in the connectivity.
    """
    p = np.asarray(point, dtype=float)
    for j in range(n):
        if np.all(xyz[:, j] == p):
            return n
    xyz[:, n] = p
    return n + 1


def tetra4v(xq: np.ndarray) -> float:
    """``tetra4v.F`` — the volume ``|det| / 6`` of a tetra from its 4 corners.

    ``RX, SX, TX = -(X4-X1), -(X4-X2), -(X4-X3)`` and
    ``TSX, TSY, TSZ = (X4-X3) x (X4-X2)``, so the triple product is the usual
    ``|(X4-X1) . ((X4-X2) x (X4-X3))|``.
    """
    q = np.asarray(xq, dtype=float).reshape(4, 3)
    a = -(q[3] - q[0])
    b = -(q[3] - q[1])
    c = -(q[3] - q[2])
    ts = np.cross(q[3] - q[2], q[3] - q[1])
    return float(abs(np.dot(a, ts)) / 6.0)


def deges4v(xe: np.ndarray) -> float:
    """``deges4v.F`` — the volume of the tetra formed by a brick's first four
    DISTINCT positions, or 0 if it never reaches four.

    ``sdlen_dege.F`` calls this only when ``IDEGE > 3``, i.e. when the
    connectivity has lost so many distinct nodes that the element's own volume
    is not a usable reference any more; below that it uses ``VOLG`` directly.
    """
    xe = np.asarray(xe, dtype=float).reshape(8, 3)
    xyz = np.zeros((3, 4), dtype=float)
    n = 1
    xyz[:, 0] = xe[0]
    for k in range(1, 8):
        n = nodedege(xyz, xe[k], n)
        if n == 4:
            break
    if n != 4:
        return 0.0
    return tetra4v(xyz.T)


# ---------------------------------------------------------------------------
# sdlen_dege -- the FAC ladder and the AREAM correction
# ---------------------------------------------------------------------------

def degeneracy_factors(group, x=None) -> tuple[np.ndarray, np.ndarray]:
    """``sdlen_dege.F``'s ``(AREAM, FAC)`` for a brick group.

    ``AREAM`` is upstream's degenerate-element stand-in for the largest face
    area; ``FAC`` is the ladder factor that gets applied to it when the
    winning face is not itself degenerate.  Together they give
    ``DELTAX = 4*V_G/sqrt(FAC*AREAM)``, which is what keeps a crushed brick's
    time step finite and positive.

    The port, in order (``sdlen_dege.F:115-142``):

    * ``FAC = 1/9`` if ``IDEGE > 2``, ``1/4`` if ``IDEGE == 2``, ``1``
      otherwise;
    * ``AREAM = EM20`` and ``IT = 0`` for every degenerate element;
    * six ``IDEGE8`` calls, one per face, in ``_HEXA_FACES`` order;
    * ``AREAM *= FAC`` where ``IT == 0``.

    ``sldege.F`` reaches the same expression from the other side — it counts
    faces with ``AREA < EM30`` instead of counting repeated nodes, and adds
    the ``IT4`` / ``deges4v`` fallback for the fully collapsed case.  Its
    ``IDEGE`` ladder is also one notch different (``>= 2 -> 1/9``, else
    ``1/4``, with no ``IDEGE == 1 -> 1`` arm).  The connectivity version is
    ported here because it is the one that agrees with the Starter's mark and
    therefore with :func:`detect`; the discrepancy is not a choice, it is the
    difference between the two callers upstream too.

    Returns:
        ``(AREAM, FAC)``, both ``(n,)`` float arrays.  For a non-degenerate
        element ``AREAM`` stays at its ``EM20`` seed and ``FAC`` at 1, exactly
        as upstream leaves it — those entries carry no meaning.
    """
    conn = _conn(group)
    n_el = conn.shape[0]
    idege = degenes8(group)
    aream = np.full(n_el, EM20, dtype=float)
    fac = np.ones(n_el, dtype=float)
    degen = idege > 0
    if n_el == 0 or not degen.any():
        return aream, fac

    sel = np.nonzero(degen)[0]
    fac[sel] = np.where(idege[sel] > 2, ONE_OVER_9, np.where(idege[sel] > 1,
                                                              FOURTH, ONE))
    it = np.zeros(sel.size, dtype=np.int64)
    xe = _coords(group, x)[conn[sel]]
    for face in _HEXA_FACES:
        # FAC passes through IDEGE8 unchanged, so only AREAM and IT move.
        aream[sel], _fac, it = _idege8_scan(
            aream[sel], fac[sel], it, xe[:, face, :])
    aream[sel] = np.where(it == 0, fac[sel] * aream[sel], aream[sel])
    return aream, fac


def characteristic_length(group, x=None) -> np.ndarray:
    """``sdlen3.F`` / ``sdlen_dege.F``'s ``DELTAX`` (Fortran ``LAT``).

    ``DELTAX = 4*VOL*XIOFF/sqrt(AREAM)``, and since ``slen.F``'s ``AREA`` is
    ``(4*area)^2`` the non-degenerate branch is exactly ``VOL / A_max`` —
    ``solid_hexa8._char_length``, which is what the engine already uses.  Only
    the degenerate branch is new: :func:`degeneracy_factors`' ``AREAM`` and
    ``FAC`` replace the face areas, and ``V_G`` becomes the
    :func:`deges4v` tetra volume once ``IDEGE > 3``.

    This is the answer to "what does degeneracy buy you".  ``V / A_max``
    tends to zero as an element is crushed, and a zero characteristic length is
    a zero time step, so an element that has degenerated past the face areas
    would otherwise freeze the cycle rather than be integrated.

    The ``IDEGE > 3`` arm (``V_G`` from :func:`deges4v`) cannot fire on a
    Starter-built group: four or more repeated nodes means 4 distinct ids,
    which the Starter has already promoted to a /TETRA4.  It is ported
    because ``sdlen_dege`` guards on it and a group assembled by hand — or by
    a future caller that skips the promotion — can reach it.
    """
    from .solid_hexa8 import _char_length, _geometry

    conn = _conn(group)
    if conn.shape[0] == 0:
        return np.empty(0, dtype=float)
    xe = _coords(group, x)[conn]
    _dndx, vol = _geometry(xe)
    lc = _char_length(xe, vol)

    idege = degenes8(group)
    degen = idege > 0
    if not degen.any():
        return lc

    aream, _fac = degeneracy_factors(group, x)
    sel = np.nonzero(degen)[0]
    v_g = np.array([deges4v(xe[i]) if idege[i] > 3 else vol[i]
                    for i in sel], dtype=float)
    lat = FOUR * v_g / np.sqrt(aream[sel])
    lc[sel] = lat
    return lc


# ---------------------------------------------------------------------------
# dim_tshedg / ind_tshedg -- the degenerate-direction edge list
# ---------------------------------------------------------------------------

def assemble_isotropic(mesh, ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """``dim_tshedg.F`` + ``ind_tshedg.F``: the ``IENUNL`` degenerate-direction
    edge list of the selected elements.

    Upstream runs the two routines back to back over the same data — ``dim_``
    counts the edges (``NEDG``), ``ind_`` fills them (``IENUNL(2,NEDG)``) —
    and then hands the pair to the ``tshcdcom`` chain, which walks the graph
    those edges define to find the shortest path between the degenerate
    nodes.  The two halves are one list, so this is one function returning
    what ``ind_`` writes.

    The scan, for the ``JHBE == 15`` (8-node solid) branch:

    * the four unslave directions ``(i, i+4)`` for ``i = 0..3``;
    * a direction is emitted only when BOTH its nodes are still unclaimed
      (``ITAG(N1) == 0 .AND. ITAG(N2) == 0``), after which both are claimed;
    * elements in the order given, directions in the order listed above.

    The ``ITAG`` rule is the whole point: the result is a *matching*, so a
    node belongs to at most one degenerate direction and the graph walk that
    consumes it cannot take the same direction twice.  The Fortran has no
    guard against ``N1 == N2`` — a brick collapsed along a direction produces
    a self-loop, burns one tag slot and is otherwise recorded verbatim.  That
    is upstream's behaviour and it is kept here.

    Args:
        mesh: an ``(n, k)`` connectivity array, or anything with a ``.conn``
            (an :class:`ElementGroup`).
        ids: which elements to scan, in order.  ``None`` means all of them.

    Returns:
        ``(2, NEDG)`` int64 array of node-index pairs; ``(2, 0)`` when
        nothing is selected.
    """
    conn = _conn(mesh)
    rows = _rows(conn, ids)
    if rows.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    tagged: dict[int, int] = {}
    pairs = []
    for e in rows:
        nodes = conn[e]
        for a, b in _UNSLAVE_PAIRS.T:
            n1, n2 = int(nodes[a]), int(nodes[b])
            if n1 not in tagged and n2 not in tagged:
                pairs.append((n1, n2))
                tagged[n1] = len(pairs)
                tagged[n2] = len(pairs)
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).T


def tag_elements(ienunl: np.ndarray, mesh) -> np.ndarray:
    """``tshcdcom_dim.F``'s element mask — ``(n,)`` bool.

    ``tshcdcom_dim`` tags every node that appears in ``IENUNL``
    (``ITAGS(N) = I``) and then marks the elements that touch a tagged node
    (``ISEND(J) = 1``), before exchanging the mask with the neighbouring
    ranks.  On one rank the exchange is the identity, so the whole routine is
    the local mask: an element is tagged when any of its nodes is one end of a
    degenerate direction.  What is left for a multi-rank build is the
    ``spmd_exch_dttsh`` call — ``mpi4py`` is not installed in this tree, so
    ``comm.mpi_world_size()`` is always 1 and there is nothing to exchange.

    ``tshcdcom_ini.F`` is the other half of the same routine and carries no
    geometry at all: it compacts the mask into the list of element numbers
    that own a tag, per rank.  ``assemble_isotropic``'s caller needs that
    list; it is one boolean gather over this mask.

    Args:
        ienunl: a ``(2, NEDG)`` array from :func:`assemble_isotropic`.
        mesh: the same mesh that produced it.

    Returns:
        ``(n,)`` bool array, one entry per element of ``mesh``.
    """
    conn = _conn(mesh)
    ienunl = np.asarray(ienunl, dtype=np.int64)
    if ienunl.size == 0 or conn.size == 0:
        return np.zeros(conn.shape[0], dtype=bool)
    ends = np.unique(ienunl)
    return (conn[:, :, None] == ends[None, None, :]).any(axis=2).any(axis=1)


def tagged_ids(mask: np.ndarray, ids) -> np.ndarray:
    """``tshcdcom_ini.F``: compact a :func:`tag_elements` mask into the
    element ids it marks."""
    ids = np.asarray(ids, dtype=np.int64)
    return ids[np.asarray(mask, dtype=bool)]
