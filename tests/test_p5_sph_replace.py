"""Task P5.1 — SPH particle replacement (spreploc.F).

Failing tests first (TDD Step 1). The conservation tests are the
specification: replacement is a volume- and momentum-conserving filter —
upstream resamples the density field onto the surviving particle set — so a
replacement that merely drops particles passes neither.

Upstream Fortran references:
- ``engine/source/elements/sph/spreploc.F`` (SPREPLOC, lines 29-142):
  rotation carry of the two local material directions ``REPLOC(:,1:6)`` by
  the spin increment ``(WXX, WYY, WZZ)`` into ``GAMA(:,1:6)``; called from
  ``spstres.F`` line 235.
- ``engine/source/elements/sph/soltosph_hour.F`` (SIG_HEPH1): the
  hourglass-stress state ``FHOUR(NEL,3,4)`` / ``SIG_HEPH`` carried with the
  particle set.
- ``engine/source/elements/sph/sponof2.F`` (SPONOF2): the outflow silent
  boundary that deactivates exited particles (the removal side).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.engine import sph_replace


# ---------------------------------------------------------------------------
# Minimal particle container implementing the protocol ``sph_replace``
# consumes (mass / vel / pos / ids arrays plus ``take``).  ``sph_engine``
# carries these as ``sph_cells`` state arrays; the protocol keeps this task
# decoupled from that layout (new file only — no existing module is touched).
# ---------------------------------------------------------------------------

class _ParticleSet:
    """Duck-typed particle set: the arrays ``replace`` filters and resamples."""

    def __init__(self, pos, vel, mass, ids=None, energy=None,
                 hg=None, hg_w=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.mass = np.asarray(mass, dtype=np.float64)
        n = len(self.pos)
        self.ids = (np.arange(n, dtype=np.int64) if ids is None
                    else np.asarray(ids, dtype=np.int64))
        self.energy = (None if energy is None
                       else np.asarray(energy, dtype=np.float64))
        # Hourglass state: FHOUR-like generalized coordinates (N,3,4) plus
        # per-particle hourglass weights (N,); see soltosph_hour.F.
        self.hg = None if hg is None else np.asarray(hg, dtype=np.float64)
        self.hg_w = (None if hg_w is None
                     else np.asarray(hg_w, dtype=np.float64))

    def take(self, keep):
        keep = np.asarray(keep, dtype=np.int64)
        return _ParticleSet(
            pos=self.pos[keep],
            vel=self.vel[keep],
            mass=self.mass[keep],
            ids=self.ids[keep],
            energy=None if self.energy is None else self.energy[keep],
            hg=None if self.hg is None else self.hg[keep],
            hg_w=None if self.hg_w is None else self.hg_w[keep],
        )


def _particle_set(n=1000, seed=0, hourglass=False, mass_spread=1.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    vel = rng.normal(0.0, 1.0, size=(n, 3))
    mass = np.full(n, 1.0 / n) * rng.uniform(1.0, mass_spread, size=n)
    energy = rng.uniform(0.0, 10.0, size=n)
    hg = rng.normal(0.0, 1e-3, size=(n, 3, 4)) if hourglass else None
    hg_w = rng.uniform(0.5, 1.5, size=n) if hourglass else None
    return _ParticleSet(pos, vel, mass, energy=energy, hg=hg, hg_w=hg_w)


class _OpenBox:
    """Boundary that keeps every particle (the plan's ``_open_box()``)."""

    def keep_mask(self, pos):
        return np.ones(len(np.asarray(pos)), dtype=bool)


def _open_box():
    return _OpenBox()


class _ClipPlane:
    """Boundary removing every particle with ``x > xmax`` (real removal)."""

    def __init__(self, xmax=0.0):
        self.xmax = float(xmax)

    def keep_mask(self, pos):
        return np.asarray(pos)[:, 0] <= self.xmax


# ---------------------------------------------------------------------------
# Plan acceptance tests (Step 1 of P5.1)
# ---------------------------------------------------------------------------

def test_replacement_preserves_total_mass_and_momentum():
    ps = _particle_set(n=1000, seed=0)
    keep = sph_replace.replace(ps, boundary=_open_box(), model=None)
    after = ps.take(keep)
    assert after.mass.sum() == pytest.approx(ps.mass.sum(), rel=1e-12)
    assert (after.mass[:, None] * after.vel).sum(0) == pytest.approx(
        (ps.mass[:, None] * ps.vel).sum(0), rel=1e-12)


def test_replacement_conserves_the_hourglass_mode_mass():
    ps = _particle_set(n=1000, seed=0, hourglass=True)
    keep = sph_replace.replace(ps, boundary=_open_box(), model=None)
    assert np.isclose(sph_replace.hg_mass(ps, keep), sph_replace.hg_mass(ps),
                      rtol=1e-10)


# ---------------------------------------------------------------------------
# Removal actually happens: totals must still be conserved exactly because
# the removed particles' mass/momentum/energy are resampled onto survivors.
# ---------------------------------------------------------------------------

def test_removal_resamples_mass_momentum_and_energy_onto_survivors():
    ps = _particle_set(n=1000, seed=0, hourglass=True)
    m0 = ps.mass.sum()
    p0 = (ps.mass[:, None] * ps.vel).sum(0)
    e0 = (ps.mass * ps.energy).sum()
    hg0 = sph_replace.hg_mass(ps)

    keep = sph_replace.replace(ps, boundary=_ClipPlane(xmax=0.0), model=None)
    assert 0 < len(keep) < 1000  # removal really happened
    assert np.all(ps.pos[keep][:, 0] <= 0.0)
    after = ps.take(keep)
    assert after.mass.sum() == pytest.approx(m0, rel=1e-12)
    assert (after.mass[:, None] * after.vel).sum(0) == pytest.approx(
        p0, rel=1e-12)
    assert (after.mass * after.energy).sum() == pytest.approx(e0, rel=1e-12)
    assert sph_replace.hg_mass(after) == pytest.approx(hg0, rel=1e-10)


def test_replace_returns_surviving_ids_in_order():
    ps = _particle_set(n=200, seed=1)
    keep = sph_replace.replace(ps, boundary=_ClipPlane(xmax=0.0), model=None)
    assert keep.dtype.kind == "i"
    assert np.all(np.diff(keep) > 0)
    assert np.array_equal(keep, np.flatnonzero(ps.pos[:, 0] <= 0.0))


# ---------------------------------------------------------------------------
# carry_state: faithful SPREPLOC rotation (spreploc.F lines 63-139)
# ---------------------------------------------------------------------------

def test_carry_state_is_identity_frame_without_orthotropy():
    rng = np.random.default_rng(3)
    reploc = rng.normal(size=(50, 6))
    gama = sph_replace.carry_state(reploc, np.zeros((50, 3)), isorth=0)
    assert gama.shape == (50, 6)
    assert np.allclose(gama[:, 0], 1.0)
    assert np.allclose(gama[:, 1:4], 0.0)
    assert np.allclose(gama[:, 4], 1.0)
    assert np.allclose(gama[:, 5], 0.0)


def test_carry_state_matches_the_rodrigues_formula_and_normalizes():
    # spreploc.F lines 74-105: NR=|W|; CR=1-NR^2/2; SR=NR; Rodrigues step.
    w = np.array([[0.0, 0.0, 0.1]])
    reploc = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    gama = sph_replace.carry_state(reploc, w, isorth=1)
    nr = 0.1
    cr, sr = 1.0 - 0.5 * nr * nr, nr
    # rotate (1,0,0) about z: axis r=(0,0,1); PS = U.r = 0; V = (0,0,0);
    # W = U - V = (1,0,0); T = r x W = (0,1,0);
    # U' = V + CR*W + SR*T = (CR, SR, 0), then normalized (lines 98-104).
    expected0 = np.array([cr, sr, 0.0])
    expected0 /= np.linalg.norm(expected0)
    assert np.allclose(gama[0, :3], expected0, rtol=1e-12, atol=1e-14)
    # each carried direction stays a unit vector
    assert np.linalg.norm(gama[0, :3]) == pytest.approx(1.0, rel=1e-12)
    assert np.linalg.norm(gama[0, 3:]) == pytest.approx(1.0, rel=1e-12)


def test_carry_state_zero_spin_is_a_noop_on_unit_frames():
    rng = np.random.default_rng(5)
    n = 30
    a = rng.normal(size=(n, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(n, 3))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    reploc = np.concatenate([a, b], axis=1)
    gama = sph_replace.carry_state(reploc, np.zeros((n, 3)), isorth=1)
    assert np.allclose(gama, reploc, rtol=1e-12, atol=1e-12)
