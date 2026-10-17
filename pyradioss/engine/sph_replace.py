# pyradioss/engine/sph_replace.py
"""SPH particle replacement: conservative filter plus local-frame carry.

Port of OpenRadioss Fortran sources:

- ``engine/source/elements/sph/spreploc.F`` (SPREPLOC, lines 29-142):
  rotation carry of the two local material directions ``REPLOC(:,1:6)`` by
  the spin increment ``(WXX, WYY, WZZ)`` into ``GAMA(:,1:6)``.  Called from
  ``spstres.F`` line 235, after ``SPDEFO3`` has produced the spin rates and
  ``SROTA3`` has rotated the stresses.  With ``ISORTH == 0`` (isotropic)
  ``GAMA`` is the identity frame (lines 63-71); otherwise each direction
  ``U`` is carried by the small-angle Rodrigues step (lines 84-128)::

      NR = |W|;  CR = 1 - NR**2/2;  SR = NR;  R = W / max(EM20, NR)
      PS = U.R;  V = PS*R;  Wv = U - V;  T = R x Wv
      U' = V + CR*Wv + SR*T;  U' /= max(EM20, |U'|)

  transcribed here as :func:`carry_state`.
- ``engine/source/elements/sph/soltosph_hour.F`` (SIG_HEPH1, lines 33-102):
  the per-particle hourglass state ``FHOUR(NEL,3,4)`` / ``SIG_HEPH`` that
  travels with the particle set; carried here as :func:`hourglass_state`.
- ``engine/source/elements/sph/sponof2.F`` (SPONOF2): the outflow silent
  boundary that deactivates exited particles — the removal side of
  :func:`replace`.

:func:`replace` itself implements the Phase 5 Task P5.1 contract
(``plan/06_phase5_elements_special.md``): replacement is a **volume- and
momentum-conserving filter**.  Particles rejected by the boundary keep no
state behind; their mass, momentum, internal energy and hourglass weight
are resampled onto the surviving set with mass weights, so the totals are
unchanged — nothing is created or destroyed, only re-attributed.
A replacement that merely drops particles passes neither conservation test.

Particle protocol (``sph_engine`` carries these as ``sph_cells`` state
arrays; this module stays duck-typed so no existing module is touched):

- ``particles.mass`` ``(N,)``, ``particles.vel`` ``(N, 3)``,
  ``particles.pos`` ``(N, 3)``, ``particles.ids`` ``(N,)``,
  optional ``particles.energy`` ``(N,)``,
  optional ``particles.hg`` ``(N, 3, 4)`` and ``particles.hg_w`` ``(N,)``;
  a ``take(idx)`` method returning the same kind of object narrowed to
  ``idx``.  Plain ``dict`` containers with the same keys are accepted too.
- ``boundary``: ``None`` (keep all), or an object with
  ``keep_mask(pos) -> bool[N]``, or an outflow-like object with
  ``compute_signed_distance(pos)`` plus a ``dist``/``distance_buffer``
  attribute (``SphOutflow``-compatible), or a callable
  ``boundary(pos) -> bool[N]``.  Anything else keeps all.
- ``model`` is accepted and ignored (reserved for threading engine state
  through a future restart path); pass ``None``.
"""

from __future__ import annotations

import numpy as np

#: Fortran ``EM20`` guard used by SPREPLOC (implicit_f.inc): the largest
#: value still treated as zero in the axis/renormalisation denominators.
_EM20 = 1.0e-20


# ---------------------------------------------------------------------------
# Field access (attribute object or plain dict — both mutated in place)
# ---------------------------------------------------------------------------

def _get(particles, name):
    """Read a particle field, or ``None`` when the container lacks it."""
    if isinstance(particles, dict):
        return particles.get(name)
    return getattr(particles, name, None)


def _put(particles, name, idx, values):
    """Write ``values`` into ``particles.<name>[idx]`` in place."""
    if isinstance(particles, dict):
        particles[name][idx] = values
    else:
        getattr(particles, name)[idx] = values


def _count(particles) -> int:
    pos = _get(particles, "pos")
    if pos is not None:
        return len(np.asarray(pos))
    mass = _get(particles, "mass")
    return 0 if mass is None else len(np.asarray(mass))


# ---------------------------------------------------------------------------
# SPREPLOC transcription: local-frame rotation carry
# ---------------------------------------------------------------------------

def carry_state(reploc, spin, isorth=1):
    """Carry the SPH local material frame through a spin increment.

    Faithful port of ``spreploc.F`` lines 63-139.  ``reploc`` holds the two
    local directions ``REPLOC(:,1:6)``; ``spin`` holds ``(WXX, WYY, WZZ)``
    per particle; the result is the rotated ``GAMA(:,1:6)``.

    Args:
        reploc: Local directions, shape ``(N, 6)`` or ``(6,)``.
        spin: Spin increment vectors, shape ``(N, 3)`` or ``(3,)``.
        isorth: Orthotropy flag (``ISORTH``).  ``0`` writes the identity
            frame (lines 63-71); nonzero applies the Rodrigues step.

    Returns:
        gama: Carried frame, same shape as ``reploc``.
    """
    r = np.asarray(reploc, dtype=np.float64)
    w = np.asarray(spin, dtype=np.float64)
    single = (r.ndim == 1)
    r2 = r.reshape(1, 6) if single else r
    n = len(r2)
    w2 = np.asarray(w, dtype=np.float64)
    if w2.ndim == 1:
        w2 = np.broadcast_to(w2, (n, 3)).copy()

    if int(isorth) == 0:
        # spreploc.F lines 63-71: GAMA = identity frame.
        out = np.zeros_like(r2)
        out[:, 0] = 1.0
        out[:, 4] = 1.0
        return out[0] if single else out

    # spreploc.F lines 74-82: NR = |W|; CR = 1-NR^2/2; SR = NR; R = W/NR.
    nr = np.linalg.norm(w2, axis=1)
    cr = 1.0 - 0.5 * nr * nr
    sr = nr.copy()
    axis = w2 / np.maximum(nr, _EM20)[:, None]

    out = np.empty_like(r2)
    for cols in ((0, 1, 2), (3, 4, 5)):
        # spreploc.F lines 84-128, applied to each of the two directions.
        u = r2[:, cols[0]:cols[2] + 1]
        ps = np.sum(u * axis, axis=1)
        v = ps[:, None] * axis
        wv = u - v
        t = np.cross(axis, wv)
        unew = v + cr[:, None] * wv + sr[:, None] * t
        # spreploc.F lines 101-104 / 124-127: renormalise to a unit vector.
        unew /= np.maximum(np.linalg.norm(unew, axis=1), _EM20)[:, None]
        out[:, cols[0]:cols[2] + 1] = unew

    return out[0] if single else out


# ---------------------------------------------------------------------------
# Hourglass state carry (soltosph_hour.F layout: FHOUR(NEL,3,4))
# ---------------------------------------------------------------------------

def hourglass_state(hg, keep, mass_old=None, mass_new=None):
    """Carry hourglass generalized coordinates onto the surviving set.

    The ``FHOUR(NEL,3,4)`` history of ``soltosph_hour.F`` is per-particle
    state, so it follows the particle: the carried array is the survivors'
    own rows.  The *extensive* hourglass mass (``hg_w``) is not handled
    here — :func:`replace` resamples it conservatively instead.

    Args:
        hg: Hourglass coordinates, shape ``(N, 3, 4)``.
        keep: Survivor indices, shape ``(K,)``.
        mass_old / mass_new: Accepted for signature symmetry with the
            resampling path; unused (intensive state needs no resampling).

    Returns:
        Carried coordinates, shape ``(K, 3, 4)``.
    """
    del mass_old, mass_new
    return np.asarray(hg, dtype=np.float64)[np.asarray(keep)].copy()


def hg_mass(particles, keep=None) -> float:
    """Total hourglass-mode mass of a particle set.

    Uses the resampled ``hg_w`` weights when present (exactly conserved by
    :func:`replace`); otherwise falls back to the ``hg`` Frobenius energy,
    which is diagnostic only and exact solely for an unchanged set.

    Args:
        particles: Particle container with optional ``hg`` / ``hg_w``.
        keep: Optional index subset to total over.

    Returns:
        Total hourglass-mode mass (0.0 when no hourglass state exists).
    """
    w = _get(particles, "hg_w")
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        if keep is not None:
            w = w[np.asarray(keep)]
        return float(w.sum())
    hg = _get(particles, "hg")
    if hg is None:
        return 0.0
    hg = np.asarray(hg, dtype=np.float64)
    if keep is not None:
        hg = hg[np.asarray(keep)]
    return float(np.sum(hg * hg))


# ---------------------------------------------------------------------------
# Replacement: boundary filter + conservative resampling (P5.1)
# ---------------------------------------------------------------------------

def _keep_mask(boundary, pos) -> np.ndarray:
    """Resolve the survivor boolean mask for the supported boundary kinds."""
    n = len(pos)
    if boundary is None:
        return np.ones(n, dtype=bool)
    if hasattr(boundary, "keep_mask"):
        return np.asarray(boundary.keep_mask(pos), dtype=bool)
    if hasattr(boundary, "compute_signed_distance"):
        d = np.asarray(boundary.compute_signed_distance(pos),
                       dtype=np.float64)
        dist = getattr(boundary, "dist",
                       getattr(boundary, "distance_buffer", 0.0))
        return d <= float(dist)
    if callable(boundary):
        return np.asarray(boundary(pos), dtype=bool)
    return np.ones(n, dtype=bool)


def replace(particles, boundary=None, model=None) -> np.ndarray:
    """Filter relocated particles through a boundary, conserving totals.

    Particles rejected by ``boundary`` are deactivated (their rows are
    zeroed: outflow semantics per ``sponof2.F``); their mass, momentum,
    internal energy and hourglass weight are resampled onto the survivors
    with mass weights, so total mass, total momentum, total internal
    energy and total hourglass-mode mass are unchanged.  Survivor rows are
    updated **in place**; use ``particles.take(keep)`` for the narrowed set.

    Args:
        particles: Particle container (see module docstring).
        boundary: Boundary selecting survivors (see module docstring).
        model: Reserved, ignored; pass ``None``.

    Returns:
        keep: Surviving particle ids, shape ``(K,)``, in original order.
    """
    del model
    n = _count(particles)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    pos = np.asarray(_get(particles, "pos"), dtype=np.float64)
    mask = np.asarray(_keep_mask(boundary, pos), dtype=bool).reshape(n)
    keep = np.flatnonzero(mask)
    if len(keep) == n:
        ids = _get(particles, "ids")
        if ids is not None:
            return np.asarray(ids, dtype=np.int64).copy()
        return np.arange(n, dtype=np.int64)

    # Pre-update copies: mass[drop]/vel are the ORIGINAL removed state;
    # mass[keep] is scaled below, so m_keep_before is re-derived from it.
    mass = np.asarray(_get(particles, "mass"), dtype=np.float64).copy()
    vel = np.asarray(_get(particles, "vel"), dtype=np.float64).copy()
    m_keep_total = float(mass[keep].sum())
    if not np.isfinite(m_keep_total) or m_keep_total <= 0.0:
        # Nothing to resample onto: deactivate everything.
        _put(particles, "mass", np.arange(n), np.zeros(n))
        _put(particles, "vel", np.arange(n), np.zeros((n, 3)))
        return np.zeros(0, dtype=np.int64)

    drop = np.flatnonzero(~mask)
    m_tot = float(mass.sum())
    scale = m_tot / m_keep_total
    m_keep_before = mass[keep].copy()

    # Mass-weighted resampling: survivor i absorbs (m_i / M_keep) of every
    # removed total.
    _put(particles, "mass", keep, mass[keep] * scale)
    _put(particles, "mass", drop, np.zeros(len(drop)))

    # Momentum: m'_i v'_i = m_i v_i + (m_i / M_keep) P_rem, i.e.
    # v'_i = (v_i + P_rem / M_keep) / scale — exact by construction.
    p_removed = (mass[drop][:, None] * vel[drop]).sum(axis=0)
    _put(particles, "vel", keep,
         (vel[keep] + p_removed[None, :] / m_keep_total) / scale)
    _put(particles, "vel", drop, np.zeros((len(drop), 3)))

    # Internal energy, mass-weighted: e'_i = (e_i + E_rem / M_keep) / scale.
    energy = _get(particles, "energy")
    if energy is not None:
        e = np.asarray(energy, dtype=np.float64)
        e_removed = float((mass[drop] * e[drop]).sum())
        _put(particles, "energy", keep,
             (e[keep] + e_removed / m_keep_total) / scale)
        _put(particles, "energy", drop, np.zeros(len(drop)))

    # Hourglass weight, mass-weighted (keeps hg_mass exact).
    hg_w = _get(particles, "hg_w")
    if hg_w is not None:
        w = np.asarray(hg_w, dtype=np.float64)
        w_removed = float(w[drop].sum())
        _put(particles, "hg_w", keep,
              w[keep] + w_removed * (m_keep_before / m_keep_total))
        _put(particles, "hg_w", drop, np.zeros(len(drop)))

    # Hourglass coordinates follow the particle (intensive history).
    hg = _get(particles, "hg")
    if hg is not None:
        carried = hourglass_state(hg, keep)
        _put(particles, "hg", keep, carried)
        _put(particles, "hg", drop,
              np.zeros((len(drop),) + np.asarray(hg).shape[1:]))

    ids = _get(particles, "ids")
    if ids is not None:
        return np.asarray(ids, dtype=np.int64)[keep].copy()
    return keep.copy()
