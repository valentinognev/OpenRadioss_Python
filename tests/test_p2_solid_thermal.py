"""Task P2.8 — solid thermal strain (the ``stherm`` family and ``mmain``).

What the tests hold down, in the order the plan lists them:

* the plan's own sample — isotropic heating produces the free expansion
  ``alpha * dT``, element for element;
* the plan's second sample — a **fully constrained** element heated by the
  same ``dT`` produces a **non-zero force**, and that force is pinned to its
  exact value rather than merely to ``> 0``;
* the ORDER, which is what the constrained case exists to pin: upstream
  subtracts the thermal strain from the **strain rate before** the
  constitutive update and books the thermal energy against the stress it
  reads at that point — the stress **before** the update, at half weight.
  Booking it after the update, or at full weight, moves the number by
  ``sigma . alpha . dT``; that sensitivity is the test.

Three things the Fortran says that a port must not "improve"
-----------------------------------------------------------

**The expansion coefficient of a solid is ONE scalar, not three.**  ``mmain``
evaluates ``alpha = FINTER(IFUNC_ALPHA, TEMPEL, ...) * FSCAL_ALPHA`` once per
element and subtracts the same ``ETH`` from ``DXX``, ``DYY`` and ``DZZ``
(``mmain.F90:776-783``) — isotropic expansion, full stop.  The plan's snippet
passes ``alpha=np.array([12e-6]*3)``; that is not a shape upstream produces
(the orthotropic per-direction ``ETHXX/ETHYY/ETHZZ`` exists only in the SHELL
path, ``materials/mat_share/thermexpc.F``, which this module is not), so a
port that broadcast it would be inventing an anisotropy that no solid element
has.  The tests pin the scalar and reject the triple.

**The energy is booked against the PRE-update stress, at half weight.**
``mmain.F90:786-787`` reads ``SIGKK = SIG(1) + SIG(2) + SIG(3)`` — the stress
the element is *entering* the step with — and then books
``EINTTH(I) = EINTTH(I) - HALF*SIGKK(I)*ETH(I)``.  The constitutive call is
200 lines further down.  The consequence is observable and is a test: the
**first** thermal step of a stressed-so-far-cold element books **exactly
zero**, because the stress read there is still the un-thermal one.

**The thermal strain is killed by ``OFF``.**  ``ETH(I) = ALPHA *
(TEMPEL(I)-TEMPEL0(I)) * OFF(I)`` — a deleted element expands by nothing.

Fortran read before any of this was written
-------------------------------------------
``engine/source/materials/mat_share/mmain.F90:758-791`` (the thermal-strain
block of the material cycle: the gate, ``ETH``, the ``FORTH``/``EPSTH``
accumulation, the strain-rate subtraction, the volume and ``AMU`` companions,
and the ``EINTTH`` booking), and the five routines the plan names —
``engine/source/elements/solid/solide/stherm.F``,
``engine/source/elements/solid/solide4/s4therm.F``,
``engine/source/elements/solid/solide4/s4therm-itet1.F``,
``engine/source/elements/thickshell/solidec/sctherm.F`` and
``engine/source/elements/thickshell/solide6c/s6ctherm.F`` — read in full and
found to be the **heat-conduction source term** of each element family, not
the thermal strain: each builds ``KC = (CA + CB*TEL)*DT*VOL*THEACCFACT``,
forms the conduction flux from the PRE-COMPUTED shape gradients
``PX1..PZ4`` and writes the nodal heat source ``FPHI``.  None of them reads
``ALPHA``, none of them touches a stress, and none of them books a mechanical
energy.  The thermal **strain** they are named for lives in ``mmain``, above.
``solid_thermal``'s docstring records this; the conduction routines are a
separate piece of work.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.elements import solid_hexa8, solid_thermal
from pyradioss.model.model import ElementGroup, Model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Mat:
    """A minimal /MAT/LAW1 elastic material (E = 210 GPa, nu = 0.3, steel).

    ``K`` and ``G`` are the Lamé constants ``mmain``'s thermal stress needs;
    for an isotropic law ``C:m = 3K`` with ``m = [1, 1, 1, 0, 0, 0]``.
    """

    def __init__(self, law=1, rho0=1000.0, E=2.1e11, nu=0.3):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.fail = None
        self.eos = None
        self.params = {"E": E, "nu": nu, "sig_y": 200.0e6,
                       "eps_max": 1e30, "eps_p_max": 1e30}

    def sound_speed_solid(self):
        return float(np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0))


class _Prop:
    def __init__(self, qa=1.1, qb=0.05, h=0.1, isolid=1, icontrol=0):
        self.params = {"qa": qa, "qb": qb, "h": h, "isolid": isolid,
                       "icontrol": icontrol}


class _Log:
    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, msg, tag=""):
        self.errors.append((msg, tag))

    def warning(self, msg, tag=""):
        pass

    def info(self, msg, tag=""):
        self.infos.append((msg, tag))


E_STEEL, NU_STEEL = 2.1e11, 0.3
K_STEEL = E_STEEL / (3.0 * (1.0 - 2.0 * NU_STEEL))      # 1.75e11
ALPHA = 12.0e-6                                          # steel, 1/K
D_T = 100.0                                              # K
ETH = ALPHA * D_T                                       # 1.2e-3 free expansion


def _group(x, conn, mat=None, prop=None, init=True):
    """An :class:`ElementGroup` over ``x``/``conn``, initialized the way the
    engine does — the shape every other ``elements/solid_*`` task builds."""
    conn = np.asarray(conn, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    model = Model()
    model.x0 = x.copy()
    model.x = x.copy()
    model.v = np.zeros((len(model.x), 3))
    model.vr = np.zeros((len(model.x), 3))
    group = ElementGroup(ids=np.arange(1, conn.shape[0] + 1, dtype=np.int64),
                         conn=conn,
                         part=np.zeros(conn.shape[0], dtype=np.int64))
    group.state["slices"] = [(slice(0, conn.shape[0]),
                              mat if mat is not None else _Mat(),
                              prop if prop is not None else _Prop())]
    if init:
        solid_hexa8.init_group(group, model, _Log())
    return group


def _unit_hexa8_group(nelem=1, mat=None):
    """``nelem`` unit cubes, node ``8*e + k`` — the canonical hexa geometry,
    so the exact force values below are readable by hand."""
    base = 0.5 * (solid_hexa8._XI + 1.0)                # unit cube, radioxs order
    x = np.tile(base, (nelem, 1, 1)).reshape(-1, 3)
    conn = (np.arange(nelem * 8, dtype=np.int64)
            .reshape(nelem, 8))
    return _group(x, conn, mat=mat)


def _thermal_stress(dT=D_T, alpha=ALPHA, mat=None):
    """``C : (alpha*dT*I)`` for an isotropic material: ``3K*alpha*dT`` in every
    normal component, compressive for a heating step (``mmain.F90:781-783``
    subtracts a POSITIVE ``ETH`` from the strain rate, so a constrained
    element is put into compression)."""
    k = K_STEEL if mat is None else mat.K
    return -3.0 * k * alpha * dT


# ---------------------------------------------------------------------------
# 1. the plan's first sample: isotropic heating -> free expansion
# ---------------------------------------------------------------------------
def test_isotropic_heating_produces_free_expansion():
    g = _unit_hexa8_group()
    d = solid_thermal.thermal_strain(g, dT=D_T, alpha=ALPHA)
    assert np.allclose(d, ALPHA * D_T)


def test_free_expansion_is_alpha_times_dT_elementwise():
    """A two-element group, two different temperatures: the strain is
    per-element, not a group-wide constant."""
    g = _unit_hexa8_group(nelem=2)
    dT = np.array([0.0, 250.0])
    d = solid_thermal.thermal_strain(g, dT=dT, alpha=ALPHA)
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(ALPHA * 250.0)


def test_alpha_is_one_scalar_per_element_not_a_per_direction_triple():
    """``mmain.F90:776`` evaluates ONE alpha per element and subtracts the same
    ETH from all three normals.  A length-3 alpha would be the SHELL path's
    orthotropic ETHXX/ETHYY/ETHZZ (``thermexpc.F``); refusing it here keeps the
    solid coupling isotropic instead of silently inventing an anisotropy."""
    g = _unit_hexa8_group()
    with pytest.raises(ValueError, match="ISOTROPIC"):
        solid_thermal.thermal_strain(g, dT=D_T,
                                     alpha=np.array([ALPHA] * 3))


def test_a_deleted_element_expands_by_nothing():
    """``ETH(I) = ALPHA*(TEMPEL-TEMPEL0)*OFF(I)`` — mmain.F90:778."""
    g = _unit_hexa8_group(nelem=2)
    g.state["off"][1] = 0.0
    d = solid_thermal.thermal_strain(g, dT=D_T, alpha=ALPHA)
    assert d[0] == pytest.approx(ETH)
    assert d[1] == 0.0


def test_a_zero_expansion_coefficient_is_exactly_zero_strain():
    g = _unit_hexa8_group()
    d = solid_thermal.thermal_strain(g, dT=D_T, alpha=0.0)
    assert np.all(d == 0.0)


def test_cooling_is_a_contraction():
    g = _unit_hexa8_group()
    d = solid_thermal.thermal_strain(g, dT=-D_T, alpha=ALPHA)
    assert np.allclose(d, -ETH)


# ---------------------------------------------------------------------------
# 2. the strain-rate rule: subtract BEFORE the constitutive update
# ---------------------------------------------------------------------------
def test_effective_strain_increment_removes_thermal_strain_from_the_normals():
    g = _unit_hexa8_group(nelem=2)
    eth = solid_thermal.thermal_strain(g, dT=D_T, alpha=ALPHA)
    deps = np.zeros((2, 6))
    deps[:, 3:] = 1.0e-3                    # a pure shear increment
    got = solid_thermal.effective_strain_increment(deps, eth)
    assert np.allclose(got[:, :3], -ETH)    # the three normals lose ETH
    assert np.allclose(got[:, 3:], deps[:, 3:])   # the shears are untouched


def test_effective_strain_increment_leaves_a_free_element_alone():
    """An element that ALREADY expanded by exactly ETH sees no effective
    strain — the free-expansion state is the zero of the subtracted
    increment, which is what makes ``thermal_strain`` observable at all."""
    g = _unit_hexa8_group()
    eth = solid_thermal.thermal_strain(g, dT=D_T, alpha=ALPHA)
    deps = np.zeros((g.n, 6))
    deps[:, 0] = ETH
    got = solid_thermal.effective_strain_increment(deps, eth)
    assert np.allclose(got[:, 0], 0.0, atol=1e-18)


def test_effective_strain_increment_does_not_mutate_its_input():
    g = _unit_hexa8_group()
    eth = solid_thermal.thermal_strain(g, dT=D_T, alpha=ALPHA)
    deps = np.zeros((g.n, 6))
    before = deps.copy()
    solid_thermal.effective_strain_increment(deps, eth)
    assert np.array_equal(deps, before)


# ---------------------------------------------------------------------------
# 3. the plan's second sample: a constrained element feels a force
# ---------------------------------------------------------------------------
def test_constrained_heating_produces_force():
    g = _unit_hexa8_group()
    f = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)
    assert np.abs(f).sum() > 0.0


def test_constrained_thermal_force_pins_the_exact_stress_value():
    """The thermal stress of an isotropic element blocked from expanding is
    ``3K*alpha*dT`` in every normal component, and the internal force of a
    uniform stress on the unit cube is ``-V*sigma.gradN_i`` with
    ``gradN_i = xi_i/4`` — so node 1 (xi = (-1,-1,-1)) carries
    ``+3K*alpha*dT/4`` on every axis and node 5 its opposite."""
    g = _unit_hexa8_group()
    f = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)
    assert f.shape == (1, 8, 3)
    scale = 3.0 * K_STEEL * ETH / 4.0
    xi = solid_hexa8._XI
    assert np.allclose(f[0], scale * xi)


def test_constrained_thermal_force_is_self_equilibrated():
    """A uniform stress on a closed element exerts no net force: sum over the
    8 nodes is zero, because sum_i gradN_i = 0.  Without this the thermal
    force would inject momentum into the explicit cycle."""
    g = _unit_hexa8_group(nelem=3)
    f = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)
    assert np.allclose(f.sum(axis=1), 0.0, atol=1e-3)


def test_heating_pushes_outward_and_cooling_pulls_inward():
    """Symmetry of the sign: the thermal stress is ``-3K*alpha*dT``, so the
    force on node 1 flips with dT."""
    g = _unit_hexa8_group()
    hot = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)
    cold = solid_thermal.forces(g, dT=-D_T, alpha=ALPHA, dt=1.0e-3)
    assert np.allclose(hot, -cold)


def test_the_time_step_enters_through_mmans_strain_rate_floor():
    """Upstream subtracts ``ETH/MAX(DT1, EM20)`` from the strain RATE and the
    constitutive update integrates that over ``DT1`` again
    (``mmain.F90:781``), so any positive dt reproduces the same stress.  The
    ``MAX(..., EM20)`` floor is what stops a zero dt from dividing by zero:
    the increment is then ``DT1 * (-ETH/EM20)``, i.e. exactly zero."""
    g = _unit_hexa8_group()
    a = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)
    b = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=2.0e-3)
    assert np.allclose(a, b)
    floored = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=0.0)
    assert np.all(np.isfinite(floored))
    assert np.abs(floored).sum() == 0.0


def test_a_deleted_element_exerts_no_thermal_force():
    g = _unit_hexa8_group(nelem=2)
    g.state["off"][1] = 0.0
    f = solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)
    assert np.abs(f[0]).sum() > 0.0
    assert np.abs(f[1]).sum() == 0.0


# ---------------------------------------------------------------------------
# 4. the energy booking — and the ORDER it pins
# ---------------------------------------------------------------------------
def _iso_stress(value, nelem=1):
    return np.tile(np.array([value, value, value, 0.0, 0.0, 0.0]),
                   (nelem, 1))


def test_energy_books_half_of_the_pre_update_stress_trace():
    """``EINTTH(I) -= HALF*SIGKK(I)*ETH(I)`` with ``SIGKK`` read at
    ``mmain.F90:786`` — the stress the element ENTERS the step with."""
    g = _unit_hexa8_group()
    sig = _iso_stress(_thermal_stress())
    e = solid_thermal.energy(g, sig, dT=D_T, alpha=ALPHA)
    # volume of the unit cube is 1, tr(sig) = 3 * (-3K*ETH)
    assert e == pytest.approx(-0.5 * (3.0 * _thermal_stress()) * ETH)


def test_a_constrained_heated_element_books_positive_energy():
    """Heating a body that cannot expand does work ON it: the ledger gains the
    strain energy the constraint stores."""
    g = _unit_hexa8_group()
    e = solid_thermal.energy(g, _iso_stress(_thermal_stress()),
                             dT=D_T, alpha=ALPHA)
    assert e > 0.0


def test_the_first_thermal_step_books_exactly_zero():
    """The order, observed: at the first thermal step the stress read at
    ``mmain.F90:786`` is still the un-thermal one (zero here), so the booking
    is zero.  A port that read the POST-update stress would book the whole of
    ``tr(sigma).alpha.dT`` on the very first step."""
    g = _unit_hexa8_group()
    assert solid_thermal.energy(g, _iso_stress(0.0),
                                dT=D_T, alpha=ALPHA) == 0.0


def test_the_booking_order_is_the_pre_update_stress_at_half_weight():
    """Both halves of ``-HALF*SIGKK*ETH`` are load-bearing, and each of the two
    plausible alternatives moves the number by the ``sigma.alpha.dT`` the plan
    names."""
    g = _unit_hexa8_group()
    sig = _iso_stress(_thermal_stress())
    eth = solid_thermal.thermal_strain(g, dT=D_T, alpha=ALPHA)[0]
    booked = solid_thermal.energy(g, sig, dT=D_T, alpha=ALPHA)

    pre_at_full_weight = -3.0 * _thermal_stress() * eth          # sig . a.dT
    post_update = -3.0 * (2.0 * _thermal_stress()) * eth        # sig_new . a.dT
    assert booked == pytest.approx(0.5 * pre_at_full_weight)
    assert booked != pytest.approx(post_update)
    assert abs(pre_at_full_weight - booked) == pytest.approx(abs(booked))


def test_energy_scales_with_the_element_volume():
    """The ledger entry is a WORK, so it carries the volume — one unit cube
    books exactly what a group of three books times three."""
    one = solid_thermal.energy(_unit_hexa8_group(nelem=1),
                               _iso_stress(_thermal_stress()),
                               dT=D_T, alpha=ALPHA)
    three = solid_thermal.energy(_unit_hexa8_group(nelem=3),
                                 _iso_stress(_thermal_stress(), 3),
                                 dT=D_T, alpha=ALPHA)
    assert three == pytest.approx(3.0 * one)


def test_energy_of_a_deleted_element_is_not_booked():
    """``ETH`` is already ``* OFF`` (``mmain.F90:778``), so a deleted element
    books nothing and a two-element group books what one of them would."""
    g = _unit_hexa8_group(nelem=2)
    g.state["off"][1] = 0.0
    sig = _iso_stress(_thermal_stress(), 2)
    both = solid_thermal.energy(g, sig, dT=D_T, alpha=ALPHA)
    one = solid_thermal.energy(_unit_hexa8_group(nelem=1), sig[:1],
                               dT=D_T, alpha=ALPHA)
    assert both == pytest.approx(one)


def test_energy_of_an_unstressed_element_is_zero_for_any_temperature():
    g = _unit_hexa8_group()
    assert solid_thermal.energy(g, _iso_stress(0.0),
                                dT=5000.0, alpha=ALPHA) == 0.0


# ---------------------------------------------------------------------------
# 5. degenerate groups must not crash the module
# ---------------------------------------------------------------------------
def test_an_empty_group_returns_empty_arrays_not_an_exception():
    empty = _group(np.zeros((0, 3)), np.zeros((0, 8), dtype=np.int64),
                   init=False)
    assert empty.n == 0
    assert solid_thermal.thermal_strain(empty, dT=D_T,
                                        alpha=ALPHA).shape == (0,)
    assert solid_thermal.forces(empty, dT=D_T, alpha=ALPHA,
                                dt=1.0e-3).shape == (0, 8, 3)
    assert solid_thermal.energy(empty, np.zeros((0, 6)),
                                dT=D_T, alpha=ALPHA) == 0.0


def test_a_group_without_a_material_slice_raises_rather_than_guessing():
    """``forces`` needs ``C:m = 3K``.  Inventing a stiffness because the slice
    is missing would put a stress nobody asked for into the cycle."""
    g = _unit_hexa8_group()
    del g.state["slices"]
    with pytest.raises(KeyError):
        solid_thermal.forces(g, dT=D_T, alpha=ALPHA, dt=1.0e-3)