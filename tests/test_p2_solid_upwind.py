"""Task P2.4 — the volume-upwind strain filter.

Fortran under test: ``$OR_SRC/engine/source/elements/solid/solide/upwind_v.F``
(the ``GAM`` ladder), its default ``ALE%UPWIND%UPWM = 0``
(``common_source/modules/ale/ale_mod.F:325``), and the SUPG rung thresholds
``PE <= EM3`` / ``PE < 3`` of ``upwind_v.F:104-107``.

The plan's reviewer focus for this task (plan/03_phase2_elements_solid.md:698):
*the default is off; a deck that does not request upwinding is byte-identical
before and after*. That is the property most of these tests pin, so "identical"
is asserted with :func:`numpy.array_equal`, not ``allclose``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.elements import solid_upwind

#: A group of eight identical unit-cube hexahedra: 8 elements, one
#: integration point each, volume 1, characteristic length 1, rho 1.
N = 8


def _unit_hexa8_group(upwm: int = 0, cupwm: float = 1.0, **state) -> object:
    """A solid element group carrying only the upwind filter's inputs."""
    conn = np.tile(np.arange(8, dtype=np.int64), (N, 1))
    base = {
        "vol0": np.ones(N),
        "mass": np.ones(N),
        "rho": np.ones(N),
        "vis": np.full(N, 0.5),
        "deltax": np.ones(N),
        "upwm": upwm,
        "cupwm": cupwm,
    }
    base.update(state)

    class _Group:
        pass

    g = _Group()
    g.n = N
    g.ids = np.arange(1, N + 1)
    g.conn = conn
    g.state = base
    return g


def _spike(sign: float = 1.0) -> np.ndarray:
    """One element at 1e6 volumetric rate, its seven peers at rest."""
    d = np.zeros((N, 6))
    d[0, :3] = sign * 1e6
    return d


# ---------------------------------------------------------------------------
# default off
# ---------------------------------------------------------------------------
def test_spike_is_untouched_when_the_property_flag_is_off():
    """A deck that never asks for upwinding comes out the other side identical."""
    d = _spike()
    g = _unit_hexa8_group(upwm=0)
    assert np.array_equal(solid_upwind.filter_rate(g, d, 1e-6), d)


def test_should_upwind_is_all_false_by_default():
    g = _unit_hexa8_group(upwm=0)
    mask = solid_upwind.should_upwind(g, _spike(), 1e-6)
    assert mask.shape == (N,)
    assert not mask.any()


def test_filter_rate_does_not_modify_its_input():
    g = _unit_hexa8_group(upwm=2)
    d = _spike()
    ref = d.copy()
    solid_upwind.filter_rate(g, d, 1e-6)
    assert np.array_equal(d, ref)


# ---------------------------------------------------------------------------
# a smooth field is never touched, flag on or off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("upwm", [0, 2, 3])
def test_upwind_is_a_no_op_on_a_smooth_strain_field(upwm):
    """Uniform tension: the volumetric rate is the same in every element."""
    g = _unit_hexa8_group(upwm=upwm)
    d = np.tile(np.array([1e-3, 0.0, 0.0]), (N, 1))
    assert np.array_equal(solid_upwind.filter_rate(g, d, 1e-6), d)


def test_should_upwind_rejects_a_field_where_every_element_transports_alike():
    g = _unit_hexa8_group(upwm=3)
    d = np.tile(np.array([1e-3, 0.0, 0.0]), (N, 1))
    assert not solid_upwind.should_upwind(g, d, 1e-6).any()


# ---------------------------------------------------------------------------
# a volumetric spike is reduced, and only volumetrically
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_upwind_activates_on_a_volumetric_spike(sign):
    g = _unit_hexa8_group(upwm=3)
    d = _spike(sign)
    out = solid_upwind.filter_rate(g, d, 1e-6)
    assert np.abs(out).max() < np.abs(d).max()
    # ... and it is the spiking element that moved, not the quiet peers.
    assert np.array_equal(out[1:], d[1:])


def test_should_upwind_selects_the_spike_and_only_the_spike():
    g = _unit_hexa8_group(upwm=3)
    mask = solid_upwind.should_upwind(g, _spike(), 1e-6)
    assert mask.tolist() == [True] + [False] * (N - 1)


def test_the_deviatoric_columns_are_never_filtered():
    """upwind_v.F is the *volume* variant: the shear part is the law's business."""
    g = _unit_hexa8_group(upwm=3)
    d = _spike()
    d[0, 3] = -4.2e5
    d[:, 4:] = 1.7e3
    out = solid_upwind.filter_rate(g, d, 1e-6)
    assert np.array_equal(out[:, 3:], d[:, 3:])


def test_the_filter_relaxes_the_trace_towards_the_group_mean():
    """The damping is a step towards the peers, not a rescale of the field."""
    g = _unit_hexa8_group(upwm=3, cupwm=1.0)
    d = _spike()
    out = solid_upwind.filter_rate(g, d, 1e-6)
    mean_tr = d[:, :3].mean()
    tr0 = d[0, :3].sum()
    tr_out = out[0, :3].sum()
    assert mean_tr < tr_out < tr0


def test_filtering_never_raises_the_peak():
    """The blend is a convex combination towards a group mean — bounded."""
    g = _unit_hexa8_group(upwm=3)
    rng = np.random.default_rng(7)
    d = rng.normal(scale=1.0e3, size=(N, 6))
    d[3, :3] += 5.0e6
    out = solid_upwind.filter_rate(g, d, 1e-6)
    assert np.abs(out).max() <= np.abs(d).max()


# ---------------------------------------------------------------------------
# the UPWM ladder itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("upwm", [0, 1])
def test_upwm_levels_below_two_do_not_filter_the_strain_rate(upwm):
    """Levels 0/1 set GAM to a sign coefficient, never a rate (amomt3.F:391-410)."""
    d = _spike()
    g = _unit_hexa8_group(upwm=upwm)
    assert np.array_equal(solid_upwind.filter_rate(g, d, 1e-6), d)


def test_a_stronger_cupwm_damps_the_spike_more():
    d = _spike()
    weak = solid_upwind.filter_rate(_unit_hexa8_group(upwm=2, cupwm=0.1), d, 1e-6)
    strong = solid_upwind.filter_rate(_unit_hexa8_group(upwm=2, cupwm=1.0), d, 1e-6)
    assert strong[0, 0] < weak[0, 0]


def test_the_supg_ladder_switches_on_the_local_peclet_number():
    """PE = FAC*|tr| with FAC = HALF*rho/max(EM20,vis) (upwind_v.F:103-107).

    Three rungs, selected by how advective the element is: PE <= EM3 takes the
    DELTAX**2 rung, EM3 < PE < 3 the GAM**2/V rung, PE >= 3 the GAM/V rung.
    Viscous elements sit on the first and the advective ones on the last, so
    sweeping ``vis`` alone walks the whole ladder and the damping must grow
    monotonically across it — a port that collapsed the branches to one
    constant would return three equal peaks.
    """
    dt = 1e-6
    d = _spike()

    def peak(vis):
        g = _unit_hexa8_group(upwm=3, vis=np.full(N, vis))
        return solid_upwind.filter_rate(g, d, dt)[0, 0]

    viscous = peak(1.0e13)          # PE <= EM3: the DELTAX**2 rung barely filters
    assert viscous == pytest.approx(d[0, 0])
    assert peak(1.0e6) < viscous    # EM3 < PE < 3, the GAM**2/V rung
    assert peak(1.0e-12) < peak(1.0e6)   # PE >= 3, the GAM/V rung


def test_supg_and_taylor_galerkin_are_different_strengths():
    d = _spike()
    tg = solid_upwind.filter_rate(_unit_hexa8_group(upwm=2), d, 1e-6)
    supg = solid_upwind.filter_rate(_unit_hexa8_group(upwm=3), d, 1e-6)
    assert tg[0, 0] != pytest.approx(supg[0, 0])


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def test_a_three_column_strain_rate_is_accepted():
    g = _unit_hexa8_group(upwm=3)
    d = np.zeros((N, 3))
    d[0, :3] = 1e6
    out = solid_upwind.filter_rate(g, d, 1e-6)
    assert out.shape == (N, 3)
    assert np.abs(out).max() < 1e6


def test_a_zero_or_negative_step_is_not_an_upwind_step():
    g = _unit_hexa8_group(upwm=3)
    d = _spike()
    assert np.array_equal(solid_upwind.filter_rate(g, d, 0.0), d)


def test_a_non_matrix_strain_rate_is_refused():
    g = _unit_hexa8_group(upwm=3)
    with pytest.raises(ValueError):
        solid_upwind.filter_rate(g, np.zeros(6), 1e-6)
    with pytest.raises(ValueError):
        solid_upwind.should_upwind(g, np.zeros((N, 2)), 1e-6)