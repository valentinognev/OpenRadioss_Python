"""Task P2.7 — the remeshing remap: ``sreploc3`` / ``srepiso3`` family.

Fortran origin (all under ``engine/source/elements/solid/solide/``)::

    srepiso3.F    the R / S / T vectors of a brick, from its 8 nodal positions
    sreploc3.F    the orthogonalization: E1 = R, E3 = E1 x S, E2 = E3 x E1
    srepiso12.F   srepiso3 plus the IF (OFF(I) <= ONE) CYCLE skip
    srepisot3.F   srepiso3 again, single precision, for the anim path

and the consumers that fix the frame CONVENTION rather than just its value::

    srcoor3.F              engine/elements/solid/solide/  (the convected frame)
    ale/alefvm/scoor3_fvm.F engine/ale/                   (the ALE / Euler one)

Who consumes this
-----------------
**Phase 11's ALE and Euler remeshers**, and nothing else today.  The remap has
no caller in this tree yet; the wiring is Phase 11 Task P11.15
(``plan/12_phase11_ale_airbag_fsi.md``, "wire the remesher"), which is gated on
this module existing.  That gate is why the module is declared rather than
wired here: an unwired remap is dead code, and the reviewer is expected to
reject it *unless* this declaration is present — so it is present, in this
docstring, in ``UPDATES.md`` §1.18.0, and in ``remap_state``'s own docstring.

Nothing in ``elements/__init__.py``, ``engine.py`` or any other caller is
touched: dispatching the tensor remap into an element cycle is a later task's
decision, and Phase 11 owns it.

What the four routines actually are
-----------------------------------
None of them interpolates a field onto a new mesh.  All four build an
orthonormal element frame out of a brick's nodal geometry, and the callers
use that frame to re-express state that was stored in one frame after the mesh
moved into another.  So the port is two pieces that compose:

* :func:`local_frame` is ``srepiso3`` + ``sreploc3`` — the frame.
* :func:`remap_isotropic` is the ``R M Rᵀ`` tensor rule that
  ``srcoor3``/``scoor3_fvm`` apply once the frame exists.
* :func:`remap_state` is the composition, and is what Phase 11 calls.

Three things the Fortran says that a port must not "improve"
------------------------------------------------------------

**The zero guard is exact, not an epsilon.**  ``sreploc3`` reads
``IF (SUMA > ZERO) SUMA = ONE/SUMA`` after ``SUMA`` has *already* been
assigned ``SQRT(...)``.  When the length is zero the reciprocal is skipped and
``SUMA`` stays at the zero length, so ``E1 = R * 0 = 0``.  A collapsed brick
gets a **zero frame**, not an identity frame and not a NaN.  Substituting
``max(norm, EM20)`` — as ``solid_hexa8z._corotational_frame`` does for
``sortho3`` — would be a different routine.

**``sreploc3`` never reads ``T``.**  It takes ``RX..TZ`` and uses only ``R``
and ``S``; ``T`` is passed and ignored.  The frame is therefore blind to the
third diagonal, and :func:`local_frame` is too.

**The engine's dummy-argument order is permuted.**  ``sreploc3.F`` declares
``E1X, E1Y, E1Z, E2X, E2Y, E2Z, E3X, E3Y, E3Z`` (line 58) while every caller
passes them ``E1X, E2X, E3X, E1Y, E2Y, E3Y, E1Z, E2Z, E3Z``
(``s10ke3.F:210``, ``s10forc3.F:533``, ``s4forc3.F:416``, ``scoor3.F:194``).
Fortran matches positionally, so inside the routine the caller's ``E2X``
variable receives the component the routine calls ``E1Y``.  The pair is
self-consistent — every caller passes and consumes in the same order — so the
computed frame is correct; but it does mean **a caller's variable names are
relabelled**, and reading the engine file alone will suggest the columns are
transposed when they are not.  :func:`local_frame` returns the frame with the
ROWS as the directions (``E1``, ``E2``, ``E3`` top to bottom), which is the
convention the consumers mean: ``srcoor3.F:301-306`` stores
``GAMA(:,1:3) = (R11, R21, R31)`` — dir1's components — and applies a vector as
``x' = Gᵀ x`` (``srcoor3.F:562-574``), i.e. rows are directions.

The unit brick, as a sanity anchor
---------------------------------
``sreploc3`` on an axis-aligned unit brick gives ``E1 = (0,1,0)``,
``E3 = (1,0,0)``, ``E2 = (0,0,1)`` — a cyclic triad, **not** the identity.
``tests/test_p2_solid_remesh.py`` pins that number, so a frame that comes out
looking like the identity is known-wrong rather than plausible.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np



def _norm(v: np.ndarray) -> np.ndarray:
    """Euclidean length along the last axis."""
    return np.sqrt(np.einsum("...i,...i->...", v, v))


def _normalize(v: np.ndarray) -> np.ndarray:
    """``sreploc3``'s normalization, including its exact-zero behaviour.

    ``IF (SUMA > ZERO) SUMA = ONE/SUMA`` after ``SUMA = SQRT(...)``: a zero
    length is left alone, so the product is the zero vector.  Upstream's guard
    is exact and this reproduces it rather than clamping to an epsilon.
    """
    n = _norm(v)
    scale = np.where(n > 0.0, 1.0 / np.where(n > 0.0, n, 1.0), 0.0)
    return v * scale[..., None]


def local_frame(xe: np.ndarray, off: Optional[np.ndarray] = None) -> np.ndarray:
    """The orthonormal element frame — ``srepiso3.F`` then ``sreploc3.F``.

    xe : (nel, 8, 3) nodal coordinates, OpenRadioss HEXA ordering (nodes 1-4
        the bottom face counter-clockwise, 5-8 the top face).
    off : optional (nel,) deletion flag.  When given, ``srepiso12.F``'s
        ``IF (OFF(I) <= ONE) CYCLE`` is honoured: an element that has not been
        remeshed yet gets a **zero** frame rather than one of its own, because
        upstream skips the loop body and leaves ``RX..TZ`` untouched.

    Returns (nel, 3, 3) with the ROWS being ``E1``, ``E2``, ``E3`` as unit
    vectors in global coordinates — the convention ``srcoor3``'s ``GAMA``
    implies (dir1's components go in ``GAMA(:,1:3)``).

    The three vectors come straight from ``srepiso3``::

        X17 = P7 - P1   X28 = P8 - P2   X35 = P5 - P3   X46 = P6 - P4
        A17 = X17 + X46                       A28 = X28 + X35
        R   = X17 + X28 - X35 - X46
        S   = A17 + A28
        T   = A17 - A28

    and ``sreploc3`` then normalizes ``R``, crosses it with ``S`` and crosses
    that with ``R`` again — ``T`` is passed in and never read.
    """
    xe = np.asarray(xe, dtype=np.float64)
    if xe.ndim != 3 or xe.shape[1:] != (8, 3):
        raise ValueError(f"xe must be (nel, 8, 3), got {xe.shape}")

    p1, p2, p3, p4, p5, p6, p7, p8 = (xe[:, i] for i in range(8))

    x17 = p7 - p1
    x28 = p8 - p2
    x35 = p5 - p3
    x46 = p6 - p4

    a17 = x17 + x46
    a28 = x28 + x35

    r = x17 + x28 - x35 - x46
    s = a17 + a28
    # T = a17 - a28 is computed by srepiso3 and then never read by sreploc3.

    e1 = _normalize(r)
    e3 = _normalize(np.cross(e1, s))
    e2 = _normalize(np.cross(e3, e1))

    frame = np.stack([e1, e2, e3], axis=-2)        # (nel, 3, 3), rows = dirs

    if off is not None:
        off = np.asarray(off, dtype=np.float64).reshape(-1)
        if off.shape[0] != frame.shape[0]:
            raise ValueError(f"off must have {frame.shape[0]} entries, "
                             f"got {off.shape[0]}")
        frame = np.where((off > 1.0)[:, None, None], frame, 0.0)

    return frame


def remap_isotropic(tensor: np.ndarray,
                    rotation: np.ndarray) -> np.ndarray:
    """``R M Rᵀ`` — the isotropic (second-order) tensor transform.

    This is the rule the callers of the ``srep*`` frame apply to a state
    tensor: ``srcoor3`` rotates geometry as ``x' = Rᵀ x`` and the matching
    tensor rule for a symmetric second-order field is ``σ' = R σ Rᵀ`` with
    ``R``'s rows the directions (``sordeft3.F``'s ``_rot3`` in
    ``solid_orthotropic.py`` is the same contraction).

    Being *isotropic* is the whole point: ``R σ Rᵀ`` commutes with the
    spherical part and leaves the trace alone, so the remap has no preferred
    global orientation.  ``tests/test_p2_solid_remesh.py`` proves that
    (covariance under a rotated frame, trace and spherical-part invariance,
    symmetry preservation, and the two-step push equalling the one-step).

    tensor   : (..., 3, 3)
    rotation : (3, 3), broadcast against the leading axes.
    """
    t = np.asarray(tensor, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    if t.shape[-2:] != (3, 3):
        raise ValueError(f"tensor must be (..., 3, 3), got {t.shape}")
    if r.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {r.shape}")
    return r @ t @ np.swapaxes(r, -1, -2)


def remap_state(old: Mapping[str, np.ndarray],
                new: Optional[Mapping[str, np.ndarray]],
                nodes_old: np.ndarray,
                nodes_new: np.ndarray) -> dict:
    """Re-express per-element state after the mesh has moved. **Phase 11's ALE
    and Euler remeshers are the declared consumer of this function** (Phase 11
    Task P11.15 wires the call); nothing in this tree calls it yet.

    For every element the frame is rebuilt from the OLD geometry and from the
    NEW geometry, and the state is carried from the old frame into the new one
    by the change of frame::

        Q  = F_newᵀ @ F_old
        σ' = Q σ Qᵀ        (second-order: stress, strain)
        v' = Q v           (first order: velocity, displacement)
        s' = s             (scalars: density, OFF, damage, temperature)

    The frame's ROWS are the directions (``srcoor3.F:301-306``), so local
    components are ``v_local = Fᵀ v_global`` (``srcoor3.F:562-574``) and the
    change of frame carries the transposed product above.  One consequence is
    worth stating because it is easy to get backwards: under a **rigid**
    rotation ``R`` of the mesh, ``F_new = F_old Rᵀ``, so ``Q`` collapses to
    exactly ``R`` — a rigidly rotated element frame must leave the state
    rotated by ``R`` and by nothing else.

    Scalars are copied, never rotated — a scalar has no direction, and a port
    that rotated ``dens`` is the classic remeshing bug, so the copy is a test
    rather than an assumption.

    old       : dict of per-element arrays, dispatched on the trailing shape
                only — no key-name guessing: ``(nel, 3, 3)`` is a second-order
                tensor, ``(nel, 3)`` a first-order vector, anything else a
                scalar.
    new       : optional dict to fill.  Must carry the same keys and matching
                leading dimension; the SAME object is returned, so a caller
                that pre-allocated its buffers gets them written in place.
                Keys the caller added and this module does not know about are
                carried through untouched.
    nodes_old : (nel, 8, 3) element geometry before the remesh.
    nodes_new : (nel, 8, 3) element geometry after the remesh.

    Returns the ``new`` dict (a fresh one when ``new`` was ``None``).
    """
    f_old = local_frame(nodes_old)
    f_new = local_frame(nodes_new)
    if f_old.shape != f_new.shape:
        raise ValueError(f"nodes_old and nodes_new disagree on element count: "
                         f"{f_old.shape[0]} vs {f_new.shape[0]}")
    # v_local = Fᵀ v_global (srcoor3.F:562-574), so old-frame components
    # become new-frame components through Q = F_newᵀ F_old.  Under a rigid
    # rotation of the mesh (F_new = F_old Rᵀ) this collapses to exactly R.
    q = np.einsum("ica,icb->iab", f_new, f_old)

    # fill the caller's own dict when given, so pre-allocated buffers are
    # written in place; keys it carries that `old` does not are left alone.
    target = {} if new is None else new

    for key, value in old.items():
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[-2:] == (3, 3):
            remapped = remap_isotropic(arr, q)            # second order
        elif arr.ndim >= 2 and arr.shape[-1] == 3:
            remapped = np.einsum("iab,ib->ia", q, arr)   # first order
        else:
            remapped = arr.copy()                         # scalar: no direction

        existing = target.get(key)
        if existing is not None and np.shape(existing) != remapped.shape:
            raise ValueError(
                f"target {key!r} has shape {np.shape(existing)}, remap "
                f"produced {remapped.shape}")
        target[key] = remapped

    return target

