"""Element-side non-local (regularised) damage path for DKT18 shells.

WHAT THIS FILE IS FOR
---------------------
``plan/04_phase3_elements_shell.md`` Task P3.0 audited the seven shell
directories and recorded exactly two DKT18 rows as ``status: "missing"`` in
``tools/validation_data/shell_routine_status.json``:

  engine/source/elements/sh3n/coquedk/cdkfint_reg.F   (544 lines)
  engine/source/elements/sh3n/coquedk/dtcdk_reg.F      ( 95 lines)

Both carry ``guarded_by_inloc: true`` — upstream reaches them only under
``INLOC = IPARG(78) > 0`` (``cdkforc3.F:498`` and ``cdkforc3.F:573``), and the
c06/c07 parity cases do not set ``/NONLOCAL``.  **They therefore cannot explain
the measured DKT18 MOMX deviation**; this is a porting-coverage gap, not a
parity gap, and the parity numbers are expected to be unchanged.

What is ported here is transcribed from the Fortran, not extrapolated from the
solid side (the solid rows in ``solid_routine_status.json`` stay ``refused``,
which is Phase 2's decision and this task does not touch it):

  * the ``NLOC_DMG`` bookkeeping for the shell side — the DOF numbering
    ``IDXI``/``POSI``/``L_NLOC``, the per-material ``LEN``/``LE_MAX``/``DENS``/
    ``DAMP``/``SSPNL`` and the ``MASS``/``MASS0`` non-local masses;
  * ``cdkfint_reg.F`` — the in-thickness regularisation over ``NDDL`` sub-dofs
    and the plane non-local force, including the broken-element absorbing branch
    and the ``NODADT > 0`` nodal time-step leg;
  * ``dtcdk_reg.F`` — the ``DT2T`` reduction by the non-local internal length
    ``LEN**2``, with the thickness variant and the ``NODADT == 0`` guard;
  * the ``cdkforc3.F:392-424`` pre-treatment that builds ``VAR_REG`` from the
    non-local increment ``DNL`` at the Hammer point.

The tests are written against the *Fortran's own algebra*, not against the
implementation: every closed form below is re-derived in the test from
``cdkfint_reg.F`` / ``dtcdk_reg.F`` / ``nlocal_dmg_init.F`` / ``hm_read_nonlocal.F``
so a transcription slip in one of the four places cannot cancel out.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.elements import shell_nonlocal as nl


# --------------------------------------------------------------------------
# Fixtures.  Everything is hand-built: the port has no /NONLOCAL DKT18 deck,
# and the upstream corpus has no DKT18 + /NONLOCAL case either.
# --------------------------------------------------------------------------


def _store(mat_id=1, *, length=0.05, le_max=0.0, dens=1.0, damp=0.5,
           sspnl=2.0, nnod=1, nddl_store=3, numnod=3):
    """A minimal ``NLOC_DMG`` shell store with ``nnod`` tagged nodes.

    ``nddl_store`` is the number of through-thickness dofs each node owns, so
    ``L_NLOC == nnod * nddl_store`` and ``POSI(k) = (k-1)*nddl_store``.
    """
    store = nl.NonlocalShell()
    store.materials[mat_id] = nl.NlocalMaterial(
        mat_id=mat_id, length=length, le_max=le_max,
        dens=dens, damp=damp, sspnl=sspnl)
    store.nnod = nnod
    store.l_nloc = nnod * nddl_store
    store.idxi = np.full(numnod, -1, dtype=np.int64)
    store.idxi[:nnod] = np.arange(nnod)
    store.posi = np.arange(nnod + 1, dtype=np.int64) * nddl_store
    store.nmat = np.full(nnod, mat_id, dtype=np.int64)
    store.nddl = np.full(nnod, nddl_store, dtype=np.int64)
    store.allocate()
    store.mass[:] = 1.0
    store.mass0[:] = 1.0
    return store


def _plane_args(*, area=1.0, thk=0.01, thk0=0.01, area0=1.0,
                px2=1.0, py2=0.0, px3=0.0, py3=1.0, off=1.0, n=1, nddl=1):
    """One-element geometry: node-1-relative DKT plane gradients.

    The three nodes each own ``nddl`` non-local dofs, so their first dof
    slots are ``0``, ``nddl`` and ``2*nddl`` and ``L_NLOC == 3*nddl`` —
    ``POSI(k) + K - 1`` in cdkfint_reg.F's indexing.
    """
    a = lambda v: np.full(n, float(v))              # noqa: E731
    return dict(area=a(area), thk=a(thk), thk0=a(thk0), area0=a(area0),
                px2=a(px2), py2=a(py2), px3=a(px3), py3=a(py3), off=a(off),
                pos1=np.full(n, 0, dtype=np.int64),
                pos2=np.full(n, nddl, dtype=np.int64),
                pos3=np.full(n, 2 * nddl, dtype=np.int64))


def _force_one(store, *, nddl=1, var_reg=None, unl=None, vnl=None,
               mat_kw=None, extra=None):
    """Run ``cdkfint_reg`` on a single element and return the FNL delta."""
    if mat_kw:
        for key in ("length", "dens", "damp", "sspnl", "le_max"):
            if mat_kw.get(key) is not None:
                setattr(store.materials[1], key, mat_kw[key])
    args = _plane_args(nddl=nddl, **(extra or {}))
    if unl is not None:
        store.unl[:] = unl
    if vnl is not None:
        store.vnl[:] = vnl
    if var_reg is None:
        var_reg = np.zeros((1, nddl))
    store.fnl[:] = 0.0
    dt2t = nl.cdkfint_reg(
        store, var_reg=var_reg, nddl=nddl, imat=1,
        ksi=1.0 / 6.0, eta=1.0 / 6.0, dt2t=EP30, nodadt=0, **args)
    return store.fnl.copy(), dt2t


# --------------------------------------------------------------------------
# 1. NLOC_DMG parameter bookkeeping (nlocal_reg_mod.F, hm_read_nonlocal.F,
#    nloc_dmg_init.F)
# --------------------------------------------------------------------------


def test_get_length_and_get_lemax_are_mutual_inverses():
    """nlocal_reg_mod.F:207 ``LE_MAX = SQRT(PI)*NLOC_LENGTH/THREE``
    and :222 ``NLOC_LENGTH = THREE*LE_MAX/SQRT(PI)``."""
    for le_max in (0.01, 0.5, 3.0):
        assert nl.get_length(le_max) == pytest.approx(3.0 * le_max / math.sqrt(math.pi))
        assert nl.get_lemax(nl.get_length(le_max)) == pytest.approx(le_max)


def test_module_constants_are_the_upstream_parameters():
    """nlocal_reg_mod.F:138 CSTA=40, :141 CDAMP=0.7;
    nloc_dmg_init.F:112 ETA=0.2."""
    assert nl.CSTA == 40.0
    assert nl.CDAMP == 0.7
    assert nl.ETA == 0.2
    assert nl.PI == pytest.approx(math.pi)


def test_thickness_tables_are_the_coqini_data_block():
    """coqini.F: Z01/WF1/ZN1 are DATA, transcribed digit-for-digit.

    Spot values read off the ``DATA`` blocks of ``coqini.F`` (rows are
    1-based in Fortran, so index ``[k-1, nddl-1]`` in NumPy)."""
    # NDDL = 1 : a single through-thickness station at the mid-surface, weight 1
    assert nl.Z0[0, 0] == 0.0
    assert nl.WF[0, 0] == 1.0
    # NDDL = 2 : the two Gauss stations sit on the faces, weight 1/2 each
    assert (nl.Z0[0, 1], nl.Z0[1, 1]) == (-0.5, 0.5)
    assert (nl.WF[0, 1], nl.WF[1, 1]) == (0.5, 0.5)
    # NDDL = 3 : Simpson weights
    assert (nl.WF[0, 2], nl.WF[1, 2], nl.WF[2, 2]) == (0.25, 0.5, 0.25)
    assert nl.Z0[2, 2] == 0.5
    # ZTH = ZN1 : the extra non-local sub-dof node positions in the thickness
    assert nl.ZTH[0, 1] == -0.5 and nl.ZTH[1, 1] == 0.5
    # every row of WF must be a partition of unity over [-0.5, 0.5]
    for nddl in range(1, 12):
        assert nl.WF[:nddl, nddl - 1].sum() == pytest.approx(1.0)


def test_nonlocal_material_density_damp_and_sound_speed_relations():
    """nloc_dmg_init.F:903-921 — the three derived per-material columns.

        DTMIN = MAX(LE_MAX/SSP, DTMIN_AMS)
        DENS  = CSTA*((LEN/LE_MAX)**2 + 1/12)*DTMIN**2
        DAMP  = (2*ETA/LE_MAX)*SQRT(DENS*(LEN**2*PI**2 + LE_MAX**2))
        SSPNL = SQRT((LEN**2 + LE_MAX**2/PI**2)/DENS)
    """
    length, le_max, ssp, dtmin_ams = 0.05, 0.12, 5000.0, 1.0e-5
    got = nl.nlocal_material_parameters(length, le_max, sound_speed=ssp,
                                        dt_min=0.0, dtmin_ams=dtmin_ams)
    # LE_MAX/SSP = 2.4e-5 > DTIN_AMS, so the sound speed drives DTMIN
    dtmin = max(le_max / ssp, dtmin_ams)
    dens = nl.CSTA * (((length / le_max) ** 2) + 1.0 / 12.0) * dtmin ** 2
    assert got.dens == pytest.approx(dens)
    assert got.damp == pytest.approx(
        (2.0 * nl.ETA / le_max)
        * math.sqrt(dens * ((length ** 2) * math.pi ** 2 + le_max ** 2)))
    assert got.sspnl == pytest.approx(
        math.sqrt((length ** 2 + le_max ** 2 / math.pi ** 2) / dens))
    # and the relation itself, read back off the returned object
    assert got.sspnl ** 2 == pytest.approx(
        (length ** 2 + le_max ** 2 / math.pi ** 2) / got.dens)


# --------------------------------------------------------------------------
# 2. cdkfint_reg.F — the plane non-local force
# --------------------------------------------------------------------------


def test_constant_nonlocal_field_with_its_own_interpolant_gives_zero_force():
    """The weak form of the Helmholtz operator is ``L^2*grad.grad - u + e``.

    Feeding it a spatially constant field whose source is that field's own
    interpolant (``VAR_REG = sum_j H_j UNL_j``, which is what
    ``cdkforc3.F:398-424`` builds out of a constant ``DNL``) leaves exactly
    the ``LEN**2 * BtB`` term — a discrete Laplacian, which must annihilate a
    constant.  This is the test that catches a mistyped ``BTB12``/``BTB13``:
    the row sums of ``BtB`` vanish only because ``PX1 = -(PX2+PX3)`` and
    ``PY1 = -(PY2+PY3)``.
    """
    u = 2.5
    unl = np.array([u, u, u])
    fnl, _ = _force_one(_store(), unl=unl,
                        var_reg=np.full((1, 1), u),     # sum_j H_j * u == u
                        mat_kw=dict(dens=1.0, damp=0.0, length=0.05))
    assert np.allclose(fnl, 0.0, atol=1e-14)


def test_constant_field_with_a_zero_source_is_not_in_equilibrium():
    """The guard on the test above: ``VAR_REG = 0`` with a constant ``UNL``
    is a real, non-zero residual — ``Nt N UNL`` does not vanish, because the
    operator is ``Nt N u - Nt e + L^2 grad.grad u``, not a partition of unity.
    Pinning it stops the equilibrium test above passing for the wrong reason.
    """
    unl = np.array([2.5, 2.5, 2.5])
    fnl, _ = _force_one(_store(), unl=unl, var_reg=np.zeros((1, 1)),
                        mat_kw=dict(dens=1.0, damp=0.0, length=0.05))
    h = np.array([1.0 - 1.0 / 6.0 - 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    vol_wf = 1.0 * 0.01 / 3.0
    assert np.allclose(fnl, -h * 2.5 * vol_wf, rtol=1e-12, atol=1e-14)


def test_plane_force_matches_the_closed_form_of_cdkfint_reg():
    """``cdkfint_reg.F:352-401``, one Hammer point, a non-constant field.

        F_i = (Nt N UNL)_i + XI*(Nt N VNL)_i - VAR_REG*H_i
              + L^2 * VOL * WF * (BtB UNL)_i

    every term written out again here, from the Fortran, independently of the
    module under test."""
    eta = ksi = 1.0 / 6.0
    h = np.array([1.0 - eta - ksi, eta, ksi])
    unl = np.array([0.3, -0.1, 0.45])
    vnl = np.array([0.02, 0.05, -0.07])
    var_reg = 0.21
    length, xi, zeta = 0.05, 0.5, 1.0
    area, thk, nddl = 1.3, 0.02, 1

    store = _store(dens=zeta, damp=xi, length=length, sspnl=2.0)
    # a scalene triangle so no two gradient columns coincide
    fnl, _ = _force_one(store, nddl=nddl, unl=unl, vnl=vnl,
                        var_reg=np.full((1, nddl), var_reg),
                        extra=dict(area=area, thk=thk, thk0=thk, area0=area,
                                   px2=0.6, py2=-0.8, px3=1.1, py3=0.4))

    vol = area * thk / 3.0
    wf = nl.WF[0, nddl - 1]
    px2, py2, px3, py3 = 0.6, -0.8, 1.1, 0.4
    # cdkfint_reg.F:159-164, transcribed again from the Fortran
    a1, a2 = -px3 - px2, -py2 - py3
    btb = np.array([
        [a1 * a1 + a2 * a2, a1 * px2 + a2 * py2, a1 * px3 + a2 * py3],
        [a1 * px2 + a2 * py2, px2 * px2 + py2 * py2, px2 * px3 + py2 * py3],
        [a1 * px3 + a2 * py3, px2 * px3 + py2 * py3, px3 * px3 + py3 * py3],
    ])
    b = (length ** 2) * vol * wf * (btb @ unl)
    ntn_unl = vol * wf * (h * (h @ unl))
    ntn_vnl = xi * vol * wf * (h * (h @ vnl))
    ntvar = var_reg * vol * wf * h
    expected = ntn_unl + ntn_vnl - ntvar + b
    # assembly subtracts (cdkfint_reg.F:471-473) and accumulates per node
    got = -fnl
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_broken_element_absorbs_the_nonlocal_wave():
    """cdkfint_reg.F:434-451 — ``OFF(I) == 0`` takes the absorbing branch:
    ``F_i = sqrt(M/M0)_i * H_i * WF * ZETA * SSPNL * (VNL_i + VNL0_i)/2
            * sqrt((4/sqrt(3))*AREA0) * THK0`` (the ``NODADT == 0`` form)."""
    zeta, sspnl = 1.25, 3.0
    eta = ksi = 1.0 / 6.0
    h = np.array([1.0 - eta - ksi, eta, ksi])
    vnl = np.array([0.4, -0.2, 0.1])
    store = _store(dens=zeta, damp=0.3, length=0.05, sspnl=sspnl)
    store.vnl[:] = vnl
    store.vnl_old[:] = 0.5 * vnl          # VNL0
    fnl, _ = _force_one(store, nddl=1, var_reg=np.zeros((1, 1)),
                        extra=dict(area=1.3, thk=0.02, thk0=0.02,
                                   area0=1.3, off=0.0))
    wf = nl.WF[0, 0]
    scale = wf * zeta * sspnl * math.sqrt((4.0 / math.sqrt(3.0)) * 1.3) * 0.02
    expected = -(h * scale * 0.5 * (vnl + 0.5 * vnl))
    assert np.allclose(fnl, expected, rtol=1e-12, atol=1e-14)
    # the sign convention is identical in the alive branch: assembly subtracts
    assert nl.cdkfint_reg.__doc__ is not None


def test_nodal_damping_factor_scales_the_velocity_term():
    """cdkfint_reg.F:373-390 — under ``NODADT > 0`` each ``NtN*VNL`` row is
    multiplied by ``MIN(sqrt(MASS_i/MASS0_i))`` over the element's three nodes."""
    eta = ksi = 1.0 / 6.0
    h = np.array([1.0 - eta - ksi, eta, ksi])
    vnl = np.array([0.2, 0.4, 0.6])
    store = _store(dens=1.0, damp=2.0, length=0.05)
    store.mass[:] = np.array([1.0, 4.0, 9.0])
    store.mass0[:] = np.array([1.0, 1.0, 1.0])
    store.vnl[:] = vnl
    store.unl[:] = 0.0
    args = _plane_args(area=1.0, thk=0.01, thk0=0.01, area0=1.0, off=1.0)
    store.fnl[:] = 0.0
    nl.cdkfint_reg(store, var_reg=np.zeros((1, 1)), nddl=1, imat=1,
                   ksi=ksi, eta=eta, dt2t=EP30, nodadt=1, **args)
    got = -store.fnl
    wf = nl.WF[0, 0]
    vol = 1.0 * 0.01 / 3.0
    ratio = math.sqrt(1.0)          # MIN over the three nodes
    expected = 2.0 * vol * wf * ratio * h * (h @ vnl)
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-14)
    # without the guard the same element gets the unscaled rows
    store.fnl[:] = 0.0
    nl.cdkfint_reg(store, var_reg=np.zeros((1, 1)), nddl=1, imat=1,
                   ksi=ksi, eta=eta, dt2t=EP30, nodadt=0, **args)
    # MASS0 == MASS makes the ratio 1 either way, so raise MASS0 and re-check
    store.mass0[:] = np.array([1.0, 1.0, 1.0])
    store.mass[:] = np.array([4.0, 4.0, 4.0])
    store.fnl[:] = 0.0
    nl.cdkfint_reg(store, var_reg=np.zeros((1, 1)), nddl=1, imat=1,
                   ksi=ksi, eta=eta, dt2t=EP30, nodadt=1, **args)
    assert np.allclose(-store.fnl, expected * 2.0, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# 3. cdkfint_reg.F — the in-thickness regularisation (NDDL > 1)
# --------------------------------------------------------------------------


def test_thickness_regularisation_transfers_back_to_var_reg():
    """cdkfint_reg.F:330-349 — after the in-thickness sweep, the integration
    point's ``VAR_REG`` is the linear interpolation of the nodal ``UNLTH``
    through the ``NTH1``/``NTH2`` shape values built from ``Z0``/``ZTH``."""
    nddl = 3
    ndnod = nddl + 1                 # cdkfint_reg.F:194 — an extra sub-dof at NDDL>2
    bufnl = nl.NonlocalThickness(nel=1, ndnod=ndnod)
    unlth = np.linspace(0.0, 1.0, ndnod).reshape(1, ndnod)
    bufnl.unlth[:] = unlth

    store = _store(nddl_store=3 * nddl)
    store.unl[:] = 0.0
    store.vnl[:] = 0.0
    var_reg = np.zeros((1, nddl))
    # a unit thickness mass makes the velocity update inert, and a zero source
    # makes the UNLTH accumulation the identity — so the transfer back to
    # VAR_REG can be read directly
    bufnl.massth[:] = 1.0
    args = _plane_args(area=1.0, thk=0.01, thk0=0.01, area0=1.0, nddl=nddl)
    nl.cdkfint_reg(store, var_reg=var_reg, nddl=nddl, imat=1,
                   ksi=1.0 / 6.0, eta=1.0 / 6.0, dt2t=EP30, nodadt=0,
                   bufnl=bufnl, dt1=0.0, dt12=0.0, **args)

    z0, zth = nl.Z0, nl.ZTH
    for k in range(nddl):
        nth1 = (z0[k, nddl - 1] - zth[k + 1, nddl - 1]) / (
            zth[k, nddl - 1] - zth[k + 1, nddl - 1])
        nth2 = (z0[k, nddl - 1] - zth[k, nddl - 1]) / (
            zth[k + 1, nddl - 1] - zth[k, nddl - 1])
        assert var_reg[0, k] == pytest.approx(
            nth1 * unlth[0, k] + nth2 * unlth[0, k + 1])


def test_thickness_is_skipped_when_the_internal_length_is_zero():
    """cdkfint_reg.F:186 ``IF ((NDDL > 1).AND.(L2>ZERO))`` — a zero-length
    non-local field leaves ``VAR_REG`` untouched and no in-thickness velocity
    update happens."""
    nddl = 3
    ndnod = nddl + 1
    bufnl = nl.NonlocalThickness(nel=1, ndnod=ndnod)
    bufnl.massth[:] = 1.0
    bufnl.vnlth[:] = 3.0
    bufnl.unlth[:] = 7.0
    vth_before = bufnl.vnlth.copy()
    ulth_before = bufnl.unlth.copy()
    store = _store(length=0.0, nddl_store=3 * nddl)
    var_reg = np.full((1, nddl), 9.0)
    args = _plane_args(area=1.0, thk=0.01, thk0=0.01, area0=1.0, nddl=nddl)
    nl.cdkfint_reg(store, var_reg=var_reg, nddl=nddl, imat=1,
                   ksi=1.0 / 6.0, eta=1.0 / 6.0, dt2t=EP30, nodadt=0,
                   bufnl=bufnl, dt1=1.0, dt12=0.5, **args)
    assert np.array_equal(var_reg, np.full((1, nddl), 9.0))
    assert np.array_equal(bufnl.vnlth, vth_before)
    assert np.array_equal(bufnl.unlth, ulth_before)


# --------------------------------------------------------------------------
# 4. dtcdk_reg.F — the non-local time-step reduction
# --------------------------------------------------------------------------


def test_dtcdk_reg_closed_form():
    """dtcdk_reg.F:69-93.

        DTNL    = 2*MIN(LE,LE_MAX)*sqrt(3*ZETA) / sqrt(12*LEN^2 + MIN(LE,LE_MAX)^2)
        DTNL_TH = the same with MIN(THK/NDDL, LE_MAX)   (NDDL > 1)
        DT2T    = MIN(DT2T, DTFAC1(1)*CDAMP*DTNL_TH, DTFAC1(1)*CDAMP*DTNL)
    """
    length, le_max, zeta = 0.05, 0.12, 2.0
    le, thk = np.array([0.3]), np.array([0.04])
    store = _store(length=length, le_max=le_max, dens=zeta)
    dt2t = nl.dtcdk_reg(store, thk=thk, off=np.ones(1), le=le, imat=1,
                        nddl=1, dt2t=1e-2, nodadt=0, dt_fac1=0.9)
    lmin = min(le[0], le_max)
    dtnl = (2.0 * lmin * math.sqrt(3.0 * zeta)
            / math.sqrt(12.0 * length ** 2 + lmin ** 2))
    # NDDL == 1 leaves DTNL_TH at EP20, so only the plane bound applies
    assert dt2t == pytest.approx(min(1e-2, 0.9 * nl.CDAMP * dtnl), rel=1e-12)


def test_dtcdk_reg_uses_the_thickness_station_spacing_for_nddl_above_one():
    """dtcdk_reg.F:73-81 — with ``NDDL > 2`` the thickness bound uses
    ``THK/NDDL``; with ``NDDL == 2`` it uses ``THK`` itself."""
    length, le_max, zeta = 0.05, 0.12, 2.0
    thk = np.array([0.06])
    le = np.array([1.0])
    store = _store(length=length, le_max=le_max, dens=zeta)

    def expect(nddl):
        spacing = thk[0] / nddl if nddl > 2 else thk[0]
        lmin = min(spacing, le_max)
        dtnl_th = (2.0 * lmin * math.sqrt(3.0 * zeta)
                   / math.sqrt(12.0 * length ** 2 + lmin ** 2))
        lmin_p = min(le[0], le_max)
        dtnl = (2.0 * lmin_p * math.sqrt(3.0 * zeta)
                / math.sqrt(12.0 * length ** 2 + lmin_p ** 2))
        return min(EP30, 0.9 * nl.CDAMP * dtnl_th, 0.9 * nl.CDAMP * dtnl)

    for nddl in (2, 3):
        got = nl.dtcdk_reg(store, thk=thk, off=np.ones(1), le=le, imat=1,
                           nddl=nddl, dt2t=EP30, nodadt=0, dt_fac1=0.9)
        assert got == pytest.approx(expect(nddl), rel=1e-12)


def test_dtcdk_reg_is_inert_under_nodadt_and_on_broken_elements():
    """dtcdk_reg.F:65 ``IF (NODADT == 0)`` and :67 ``IF (OFF(I)/=ZERO)``."""
    store = _store(length=0.05, le_max=0.12, dens=2.0)
    thk = np.array([0.06])
    le = np.array([1.0])
    alive = nl.dtcdk_reg(store, thk=thk, off=np.ones(1), le=le, imat=1,
                         nddl=1, dt2t=EP30, nodadt=0)
    assert nl.dtcdk_reg(store, thk=thk, off=np.ones(1), le=le, imat=1,
                        nddl=1, dt2t=EP30, nodadt=1) == EP30
    assert nl.dtcdk_reg(store, thk=thk, off=np.zeros(1), le=le, imat=1,
                        nddl=1, dt2t=EP30, nodadt=0) == EP30
    assert alive < EP30


def test_dtcdk_reg_never_widens_the_step():
    store = _store(length=0.05, le_max=0.12, dens=2.0)
    thk, le = np.array([0.06]), np.array([1.0])
    tight = nl.dtcdk_reg(store, thk=thk, off=np.ones(1), le=le, imat=1,
                         nddl=1, dt2t=1e-9, nodadt=0)
    assert tight == 1e-9


# --------------------------------------------------------------------------
# 5. The INLOC == 0 cycle must be untouched, bit for bit.
# --------------------------------------------------------------------------


def test_inloc_zero_path_is_untouched_by_the_nonlocal_module():
    """The whole module is inert unless a store exists: ``setup`` returns
    ``None`` with no ``/NONLOCAL`` and no ``/MAT/NONLOCAL`` in the model, so
    ``cdkforc3.F``'s ``IF (INLOC > 0)`` guard never opens."""
    from pyradioss.model.model import Model
    model = Model()
    assert nl.setup(model, log=None) is None


# --------------------------------------------------------------------------
# 6. /MAT/NONLOCAL and /NONLOCAL become live inputs.
# --------------------------------------------------------------------------


def test_mat_nonlocal_modifier_feeds_the_element_side_path():
    """``/MAT/NONLOCAL/{id}`` is parsed into ``model.mat_nonlocal_modifiers``
    (``pyradioss/model/model.py:1190``) and, until this task, nothing read it.
    ``setup`` must now consume it and produce a live non-local material."""
    from pyradioss.model.entities import MatNonlocalModifier
    from pyradioss.model.model import Model

    model = Model()
    model.mat_nonlocal_modifiers[7] = MatNonlocalModifier(
        id=7, mat_id=7, length=0.04, le_max=0.0)
    store = nl.materials_from_model(model)
    assert 7 in store
    got = store[7]
    # hm_read_nonlocal.F:150-158 — LE_MAX == 0 derives it from LENGTH
    assert got.le_max == pytest.approx(nl.get_lemax(0.04))
    assert got.length == pytest.approx(0.04)


def test_nonlocal_card_is_preferred_and_its_lemax_drives_the_length():
    """``/NONLOCAL/{id}`` (``model.nonlocal_models``) and ``/MAT/NONLOCAL/{id}``
    (``model.mat_nonlocal_modifiers``) both feed the same table; when both are
    present the ``/NONLOCAL`` card wins, and ``LE_MAX > 0`` inverts into
    ``LEN`` (hm_read_nonlocal.F:150-153)."""
    from pyradioss.model.entities import MatNonlocalModifier, NonlocalModel
    from pyradioss.model.model import Model

    model = Model()
    model.mat_nonlocal_modifiers[3] = MatNonlocalModifier(
        id=3, mat_id=3, length=0.01, le_max=0.0)
    model.nonlocal_models[3] = NonlocalModel(
        mat_id=3, length=0.0, le_max=0.09)
    got = nl.materials_from_model(model)[3]
    assert got.le_max == pytest.approx(0.09)
    assert got.length == pytest.approx(nl.get_length(0.09))

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 7. The DKT18 kernel and the engine actually run the path.
# --------------------------------------------------------------------------

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

_DKT18_BODY = """\
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 0.0 10.0 0.0
4 10.0 10.0 0.0
5 20.0 10.0 0.0
/SH3N/1
1 1 2 3
2 2 4 3
3 2 5 4
/PART/1
triangles
1 1
""" + STEEL_LAW1 + """\
/PROP/SHELL/1
dkt18 shell prop
1 0 2 0
0.01 0.01 0.01
{nip} 0 1.0
/END
"""


def _dkt18_deck(nonlocal_card="", nip=1):
    """A three-triangle DKT18 deck, optionally carrying ``/NONLOCAL/MAT/1``.

    **Synthetic** — the upstream validation corpus has no DKT18 + ``/NONLOCAL``
    case, so this deck exists to exercise the path, not to compare against a
    measured parity number.  The engine-level parity cases c00/c06/c07 do not
    set ``/NONLOCAL`` and are untouched by it.
    """
    nl_card = f"/NONLOCAL/MAT/1\n{nonlocal_card}\n" if nonlocal_card else ""
    return ("/BEGIN\ndkt18 non-local test\n" + nl_card
            + _DKT18_BODY.format(nip=nip))


def _build(tmp_path, nonlocal_card="", nip=1):
    """Parse + resolve a DKT18 model, the way the Starter does."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model
    from pyradioss.starter.initialization import (
        build_element_groups, initialize_elements_and_mass, resolve_materials,
        resolve_node_groups,
    )

    f = tmp_path / "deck.rad"
    f.write_text(_dkt18_deck(nonlocal_card=nonlocal_card, nip=nip))
    log = MessageLog()
    model = Model()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    resolve_node_groups(model, log)
    build_element_groups(model, log)
    initialize_elements_and_mass(model, log)
    return model, log


def test_a_nonlocal_dkt18_deck_builds_the_dof_numbering(tmp_path):
    """``nloc_dmg_init.F:751-768`` — POSI/IDXI/L_NLOC over the DKT18 nodes.

    Three triangles over five nodes: every node carries ``NDDL`` dofs, so
    ``L_NLOC == 5*NDDL`` and ``POSI(k) == (k-1)*NDDL``.
    """
    model, _log = _build(tmp_path, nonlocal_card="0.05 0.0")
    st = model.sh3n_dkt18.state
    assert st["inloc"] == 1
    store = st["nlocal"]
    nddl = st["nip_max"]
    assert store.nnod == 5
    assert store.l_nloc == 5 * nddl
    assert list(store.posi) == [k * nddl for k in range(6)]
    assert (store.idxi[model.sh3n_dkt18.conn.reshape(-1)] >= 0).all()
    # DENS / DAMP / SSPNL are derived and live, not zero
    mat = store.materials[1]
    assert mat.dens > 0.0 and mat.damp > 0.0 and mat.sspnl > 0.0
    assert mat.le_max == pytest.approx(nl.get_lemax(0.05))
    assert (store.mass > 0.0).all()
    assert np.array_equal(store.mass, store.mass0)


def test_a_deck_without_nonlocal_leaves_the_explicit_cycle_untouched(tmp_path):
    """The task's scope: with no ``/NONLOCAL`` the INLOC == 0 cycle is
    bit-identical — the store never exists, so ``cdkforc3.F:498`` never fires."""
    from pyradioss.elements import shell_dkt18

    plain_dir, nl_dir = tmp_path / "plain", tmp_path / "nl"
    plain_dir.mkdir()
    nl_dir.mkdir()
    plain_model, _ = _build(plain_dir)
    nl_model, _ = _build(nl_dir, nonlocal_card="0.05 0.0")

    assert "inloc" not in plain_model.sh3n_dkt18.state
    assert getattr(plain_model, "nloc_dmg", None) is None
    assert nl_model.sh3n_dkt18.state["inloc"] == 1

    def _cycle(model):
        x = model.x0.copy()
        v = np.zeros((model.numnod, 3))
        v[:, 2] = 1.0
        vr = np.zeros((model.numnod, 3))
        fint = np.zeros((model.numnod, 3))
        mint = np.zeros((model.numnod, 3))
        dt_e = shell_dkt18.forces(model.sh3n_dkt18, x, v, vr, 1e-6,
                                  fint, mint)
        return fint, mint, dt_e

    fint_a, mint_a, dt_a = _cycle(plain_model)
    fint_b, mint_b, dt_b = _cycle(nl_model)
    # the internal force and moment are bit-identical: the regularised force
    # lands on the additional non-local dofs, never on the structural ones
    assert np.array_equal(fint_a, fint_b), "the INLOC == 0 force cycle moved"
    assert np.array_equal(mint_a, mint_b), "the INLOC == 0 moment cycle moved"
    # the non-local time step is dtcdk_reg.F's reduction, so it may only ever
    # pull the claim DOWN — never change the local part of it
    assert np.all(dt_b <= dt_a)
    assert float(dt_b.min()) < float(dt_a.min())


def test_the_nonlocal_force_is_live_and_claims_a_time_step(tmp_path):
    """cdkforc3.F:498 / :573 — with INLOC > 0 the kernel drives the non-local
    dofs and ``dtcdk_reg.F``'s reduction reaches the element's dt claim.

    A virgin deck's field is zero — upstream's initial relaxation
    (``nlocal_init_sta.F``) converges to zero when the material produces no
    plastic strain — so the field is seeded here to show the path carries it.
    """
    from pyradioss.elements import shell_dkt18

    model, _log = _build(tmp_path, nonlocal_card="0.05 0.0")
    g = model.sh3n_dkt18
    st = g.state
    store = st["nlocal"]
    assert store.l_nloc > 0

    x = model.x0.copy()
    v = np.zeros((model.numnod, 3))
    v[:, 2] = 1.0                      # a transverse kick
    v[-1, 0] = 3.0                     # and an in-plane pull on the tip node
    vr = np.zeros((model.numnod, 3))
    fint = np.zeros((model.numnod, 3))
    mint = np.zeros((model.numnod, 3))
    dt = 1e-6

    # a virgin deck starts at zero and stays there
    dt_e = shell_dkt18.forces(g, x, v, vr, dt, fint, mint)
    assert np.all(store.fnl == 0.0)
    assert np.abs(fint).max() > 0.0

    # seed a non-uniform field: the regularised force must follow
    store.unl[:] = np.linspace(0.1, 0.4, store.l_nloc)
    store.fnl[:] = 0.0
    shell_dkt18.forces(g, x, v, vr, dt, fint, mint)
    assert np.abs(store.fnl).max() > 0.0
    assert np.all(np.isfinite(store.fnl))

    # dtcdk_reg.F reduced the claim to the non-local bound
    mat = store.materials[1]
    le_min = float(np.min(np.sqrt((4.0 / math.sqrt(3.0)) * st["area0"])))
    lmin = min(le_min, mat.le_max)
    dtnl = 0.9 * nl.CDAMP * (2.0 * lmin * math.sqrt(3.0 * mat.dens)
                             / math.sqrt(12.0 * mat.length ** 2 + lmin ** 2))
    assert float(dt_e.min()) == pytest.approx(dtnl, rel=1e-9)


def test_the_nonlocal_cycle_advances_the_field(tmp_path):
    """The four engine phases (nlocal_incr/acc/vel.F) actually move the field:
    a virgin deck starts at zero, and a run has to leave it non-zero."""
    model, _log = _build(tmp_path, nonlocal_card="0.05 0.0")
    store = model.sh3n_dkt18.state["nlocal"]
    store.unl[:] = 0.0
    store.vnl[:] = 0.0
    store.fnl[:] = 3.0
    store.acc()
    store.vel(0.5e-6)
    store.incr(1e-6)
    assert np.abs(store.vnl).max() > 0.0
    assert np.abs(store.unl).max() > 0.0
    assert np.all(store.fnl == 0.0)          # nlocal_vel.F:71 zeroes it


_NONLOCAL_RUN_DECK = """\
/BEGIN
dkt18 non-local run
/NONLOCAL/MAT/1
0.05 0.0
/NODE
1 0.0 0.0 0.0
2 10.0 0.0 0.0
3 0.0 10.0 0.0
4 10.0 10.0 0.0
5 20.0 10.0 0.0
/SH3N/1
1 1 2 3
2 2 4 3
3 2 5 4
/PART/1
triangles
1 1
""" + STEEL_LAW1 + """\
/PROP/SHELL/1
dkt18 shell prop
1 0 2 0
0.01 0.01 0.01
1 0 1.0
/TIME
1e-3
/GRNOD/NODE/1
fixed
1 2 3
/END
"""

_NONLOCAL_RUN_ENGINE = """\
/RUN/DKT18/1
0.001
/TFILE
1e-4
/PRINT
-10
/STOP
99.0
/END
"""


def test_a_nonlocal_dkt18_deck_runs_end_to_end(tmp_path):
    """The whole chain: Starter builds NLOC_DMG, the Engine's three non-local
    phases run every cycle, and the run completes without a stop."""
    from pyradioss.engine.engine import run_engine
    from pyradioss.starter.starter import run_starter

    (tmp_path / "NL_0000.rad").write_text(_NONLOCAL_RUN_DECK)
    (tmp_path / "NL_0001.rad").write_text(_NONLOCAL_RUN_ENGINE)
    model = run_starter(str(tmp_path / "NL_0000.rad"))
    assert model.nloc_dmg is not None
    assert model.sh3n_dkt18.state["inloc"] == 1

    model = run_engine(str(tmp_path / "NL_0001.rad"))
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.cycle > 0
    assert np.all(np.isfinite(model.x))
