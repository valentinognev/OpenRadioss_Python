"""Task P2.2 — the ``solidez`` orthotropic solid continuum (``szforc3.F``).

WHAT THIS FILE IS FOR
---------------------
``pyradioss/elements/solid_orthotropic.py`` is the first port of upstream's
**orthotropic solid continuum** — the ``solidez`` family, ``/PROP/SOL_ORTH``
(Isolid 11/12), driven by ``engine/source/elements/solid/solidez/szforc3.F``.
The P2.0 reconciliation audit recorded the whole family as ``missing`` and
noted the misattribution that hid it: ``solid_cohesive.py`` cites
``szforc3.F`` while implementing a cohesive element, and
``/PROP/SOL_ORTH`` parsed but routed to ``tshells``.  These tests are the
gate on the new module: they pin the *behaviour* the plan names, and they pin
the algebra that decides everything downstream — ``gettransv.F`` /
``cbatran3v.F`` / ``mstiforthv.F`` / ``mmodul.F`` — against hand
computations, so a kernel that passes the isotropic smoke test and is wrong
for every real orthotropic part cannot pass here.

THE TWO BEHAVIOURS THE PLAN NAMES
---------------------------------
1. **The isotropic limit is the isotropic kernel.**  A group whose moduli
   reduce to the isotropic Lamé tensor carries no orthotropy
   (``ISORTH == 0``, upstream ``szforc3.F:499`` / ``:933``), so the solidez
   cycle *is* the ``sforc3`` cycle and ``forces`` must return the same
   nodal-force array — compared here with :func:`numpy.array_equal`, i.e.
   bit for bit, not within a tolerance.
2. **A uniaxial strain returns ``C11`` times the strain and no lateral
   stress.**  For the decoupled moduli an orthotropic card can carry
   (``mmodul.F`` MTN=28/68 sets ``CC(1,2) = CC(1,3) = CC(2,3) = ZERO``), the
   strain state ``(eps11, 0, 0, 0, 0, 0)`` must give
   ``sigma_11 = C11 * eps11`` and ``sigma_22 = sigma_33 = 0``.

   NOTE the plan's sample passes ``nu12 = nu13 = nu23 = 0.3`` together with
   that assertion.  Those two cannot both hold: any orthotropic stiffness
   with a non-zero normal coupling term ``C12`` returns
   ``sigma_22 = C12 * eps11 != 0`` under a strain-controlled uniaxial
   strain, so ``atol = 1e-3`` on the lateral components is unreachable with
   ``nu != 0``.  The assertion above is the one kept; the coupled case is
   covered separately by
   :func:`test_coupled_orthotropic_moduli_match_the_lame_nu_form`, which
   checks the ``mmodul.F`` MTN=25/107/112 formula term by term.

WHAT IS *NOT* CLAIMED HERE
--------------------------
``szhour3_or.F`` / ``szsvm_or.F`` / ``szhour_ctl.F`` — the ORTHOTROPIC
hourglass law, the consumer of the ``CC``/``CG``/``G33`` moduli this module
builds — are **not** transcribed.  The orthotropic branch therefore runs the
isotropic viscous Flanagan–Belytschko hourglass (``szhour3.F``, the routine
upstream itself calls at ``szforc3.F:970`` for ``ISORTH == 0``).  The moduli
themselves *are* ported and tested; the hourglass law that consumes them is
named as absent in the module docstring and in ``UPDATES.md``.  The parity
case the plan asks for is likewise recorded as not added: RD-E-2100 exists on
this machine but contains no orthotropic material, and Task P2.11 owns the
``"solids_ortho"`` registration that would let a deck reach this kernel at
all.
"""

from __future__ import annotations

import copy
import inspect
import pathlib
import re

import numpy as np
import pytest

from pyradioss.elements import solid_hexa8, solid_orthotropic
from pyradioss.model.model import ElementGroup, Model


# ----------------------------------------------------------------------------
# Test doubles: a LAW1 material, an orthotropic one, a /PROP/SOL_ORTH property
# ----------------------------------------------------------------------------

class _Mat:
    """Minimal material object: the fields the solid kernels read."""

    def __init__(self, law=1, rho0=1000.0, E=2.0e11, nu=0.3, **params):
        self.law = law
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.fail = None
        self.eos = None
        self.params = dict(params)
        self.params.setdefault("eps_max", 1e30)
        self.params.setdefault("eps_p_max", 1e30)

    def sound_speed_solid(self):
        return np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)


class _Prop:
    """A /PROP/SOL_ORTH-shaped property (``PropType6`` fields the kernel reads)."""

    def __init__(self, orthtrop=0, vx=0.0, vy=0.0, vz=1.0, phi=0.0,
                 qa=1.1, qb=0.05, h=0.1):
        self.type = 6
        self.orthtrop = orthtrop
        self.vx, self.vy, self.vz = vx, vy, vz
        self.phi = phi
        self.params = {"qa": qa, "qb": qb, "h": h}


class _Log:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg, tag=""):
        self.errors.append((tag, msg))

    def warning(self, msg, tag=""):
        self.warnings.append((tag, msg))

    def info(self, msg, tag=""):
        pass


#: MTN=28 of ``mmodul.F``: CC(1,1..3) from the three moduli, CC(1,2) =
#: CC(1,3) = CC(2,3) = ZERO, G3 from the three shear moduli.  The decoupled
#: family, and the one that makes behaviour 2 above expressible.
MTN_DIRECT = 28


def _ortho_params(c11, c22, c33, g12, g23, g31):
    return {
        "MAT_M11": c11, "MAT_M22": c22, "MAT_M33": c33,
        "MAT_G12": g12, "MAT_G23": g23, "MAT_G31": g31,
    }


def _unit_hexa8_group(mat=None, prop=None, model=None, group=None):
    """A one-element unit-cube group at the reference configuration."""
    if model is None:
        model = Model()
        x0 = 0.5 * (solid_hexa8._XI + 1.0)
        model.x0 = x0.copy()
        model.x = x0.copy()
        model.v = np.zeros((8, 3))
        model.vr = np.zeros((8, 3))
    if group is None:
        group = ElementGroup(
            ids=np.array([1], dtype=np.int64),
            conn=np.arange(8, dtype=np.int64)[None, :],
            part=np.array([0], dtype=np.int64),
        )
    if mat is None:
        mat = _Mat(law=1, rho0=1000.0, E=2.0e11, nu=0.3)
    if prop is None:
        prop = _Prop()
    group.state["slices"] = [(slice(0, group.n), mat, prop)]
    return model, group


def _loading_state(model, seed=3):
    """A fixed pseudo-random velocity field: a genuinely deforming state."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(8, 3)) * 1.0e2


def _run(kernel, model, group, v, dt=1.0e-3):
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dtc = kernel.forces(group, model.x, v, model.vr, dt, fint, mint)
    return fint, dtc


# ----------------------------------------------------------------------------
# 1. The isotropic limit — behaviour 1
# ----------------------------------------------------------------------------

def test_isotropic_limit_forces_matches_solid_hexa8_bitwise():
    """A group with no orthotropy runs the sforc3 cycle: the same fint,
    bit for bit.  Upstream's own reduction: szforc3.F only rotates the rate
    into the material frame (`IF (... OR.ISORTH/=0)`, line 499) and only
    calls the orthotropic hourglass (`ELSEIF (ISORTH>0)`, line 933) when
    ISORTH > 0."""
    model, ref_group = _unit_hexa8_group()
    solid_hexa8.init_group(ref_group, model, _Log())
    v = _loading_state(model)
    ref, _ = _run(solid_hexa8, model, ref_group, v)

    model2, got_group = _unit_hexa8_group()
    solid_orthotropic.init_group(got_group, model2, _Log())
    got, _ = _run(solid_orthotropic, model2, got_group, v)

    assert np.array_equal(ref, got), (
        "the isotropic limit must reproduce solid_hexa8.forces bit for bit; "
        f"max |diff| = {np.abs(ref - got).max():.3e}"
    )


def test_isotropic_limit_tangent_and_kgeo_match_solid_hexa8():
    model, ref_group = _unit_hexa8_group()
    solid_hexa8.init_group(ref_group, model, _Log())
    ref_group.state["sig"][:] = np.array([[1.0e8, 2.0e7, 3.0e7,
                                          4.0e6, 5.0e6, 6.0e6]])
    ke_ref, dofs_ref = solid_hexa8.tangent(ref_group, model.x)
    kg_ref, _ = solid_hexa8.kgeo(ref_group, model.x)

    model2, got_group = _unit_hexa8_group()
    solid_orthotropic.init_group(got_group, model2, _Log())
    got_group.state["sig"][:] = np.array([[1.0e8, 2.0e7, 3.0e7,
                                           4.0e6, 5.0e6, 6.0e6]])
    ke_got, dofs_got = solid_orthotropic.tangent(got_group, model2.x)
    kg_got, _ = solid_orthotropic.kgeo(got_group, model2.x)

    assert np.array_equal(dofs_ref, dofs_got)
    assert np.array_equal(ke_ref, ke_got)
    assert np.array_equal(kg_ref, kg_got)


# ----------------------------------------------------------------------------
# 2. The uniaxial stiffness — behaviour 2
# ----------------------------------------------------------------------------

def test_uniaxial_strain_returns_C11_and_no_lateral_stress():
    """Behaviour 2: strain (1e-3, 0, 0, 0, 0, 0) against C11 = 2.0e11 with
    the decoupled moduli of mmodul.F MTN=28/68 gives sigma_11 = 2.0e8 and
    zero lateral stress."""
    C11 = 2.0e11
    mat = _Mat(law=MTN_DIRECT, rho0=1000.0, **_ortho_params(C11, C11, C11, 0.0, 0.0, 0.0))
    model, group = _unit_hexa8_group(mat=mat)
    solid_orthotropic.init_group(group, model, _Log())

    eps = np.zeros((1, 6))
    eps[0, 0] = 1.0e-3
    got = solid_orthotropic.stress_from_strain(group, eps)

    assert got.shape == (1, 6)
    assert np.allclose(got[0, 0], C11 * 1.0e-3, rtol=1.0e-12)
    assert np.allclose(got[0, 1:], 0.0, atol=1.0e-3)


def test_coupled_orthotropic_moduli_match_the_lame_nu_form():
    """mmodul.F MTN=25/107/112 — the compliance construction with its
    singular guard — term by term against the Fortran formulas:
    ``C11 = E1/max(EM20, 1-nu12*nu21)``,
    ``C12 = 0.5*(nu21*C11 + nu12*C22)``, ``C13 = C23 = 0``."""
    E1, E2, E3 = 2.0e11, 3.0e11, 4.0e11
    nu12, nu21 = 0.3, 0.2
    g12, g23, g31 = 5.0e10, 6.0e10, 7.0e10
    mat = _Mat(
        law=25, rho0=1000.0,
        MAT_E11=E1, MAT_E22=E2, MAT_E33=E3,
        MAT_PRAB=nu12, MAT_PRBA=nu21,
        MAT_G12=g12, MAT_G23=g23, MAT_G31=g31,
    )
    model, group = _unit_hexa8_group(mat=mat)
    solid_orthotropic.init_group(group, model, _Log())

    s1 = 1.0 - nu12 * nu21
    c11 = E1 / s1
    c22 = E2 / s1
    c12 = 0.5 * (nu21 * c11 + nu12 * c22)
    D = group.state["D"]
    assert D.shape == (1, 6, 6)
    assert np.allclose(D[0, 0, 0], c11)
    assert np.allclose(D[0, 1, 1], c22)
    assert np.allclose(D[0, 2, 2], E3)
    assert np.allclose(D[0, 0, 1], c12)
    assert np.allclose(D[0, 0, 2], 0.0)
    assert np.allclose(D[0, 1, 2], 0.0)
    assert np.allclose(D[0, 3, 3], g12)
    assert np.allclose(D[0, 4, 4], g23)
    assert np.allclose(D[0, 5, 5], g31)

    # ... and the singular guard: nu12*nu21 -> 1 sends C11/C22 to the EM20 cap
    mat2 = _Mat(law=25, rho0=1000.0, MAT_E11=E1, MAT_E22=E2, MAT_E33=E3,
                MAT_PRAB=1.0, MAT_PRBA=1.0, MAT_G12=g12, MAT_G23=g23, MAT_G31=g31)
    model2, group2 = _unit_hexa8_group(mat=mat2)
    solid_orthotropic.init_group(group2, model2, _Log())
    assert np.all(np.isfinite(group2.state["D"]))
    assert np.allclose(group2.state["D"][0, 0, 0], E1 / 1.0e-20)


def test_isotropic_material_reproduces_the_lame_moduli():
    """mmodul.F's ELSE branch: C1 = 3E/(1+nu), LAMDA = C1*nu,
    GG = C1*(1-2*nu), CC11 = LAMDA + GG, CC12 = LAMDA,
    G33(J,J) = GG/2, CG = 0.  CC11 = LAMDA + GG is exactly lam + 2G, which
    is what makes the isotropic-limit test above a real limit."""
    E, nu = 2.0e11, 0.3
    mat = _Mat(law=1, rho0=1000.0, E=E, nu=nu)
    model, group = _unit_hexa8_group(mat=mat)
    solid_orthotropic.init_group(group, model, _Log())
    D = group.state["D"][0]

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    assert np.allclose(D[0, 0], lam + 2.0 * G)
    assert np.allclose(D[1, 1], lam + 2.0 * G)
    assert np.allclose(D[2, 2], lam + 2.0 * G)
    assert np.allclose(D[0, 1], lam)
    assert np.allclose(D[0, 2], lam)
    assert np.allclose(D[3, 3], G)
    assert np.allclose(D[4, 4], G)
    assert np.allclose(D[5, 5], G)
    # the isotropic block has no normal-shear coupling (mmodul.F: CG = 0)
    assert np.allclose(D[:3, 3:], 0.0)


# ----------------------------------------------------------------------------
# 3. gettransv.F — the cosine products, against a hand computation
# ----------------------------------------------------------------------------

def test_gettransv_matches_its_fortran_formulas():
    """gettransv.F: E1 = GAMA(:,1:3), E2 = GAMA(:,4:6), E3 = E1 x E2, then
    QC(i,·) = (E1^2, E2^2, E3^2), QGC(i,·) = (E1*E2, E2*E3, E1*E3),
    QG(i,·) = (E1*E2'+E2*E1', E2*E3'+E3*E2', E3*E1'+E1*E3') and
    QCG(i,·) = 2*(E1*E1', E2*E2', E3*E3'), with the primed component the
    next one cyclically (J -> J+1, 3 -> 1)."""
    c = np.sqrt(0.5)
    gama = np.array([[c, c, 0.0, 0.0, 0.0, 1.0]])     # 45 deg about z
    qc, qcg, qgc, qg = solid_orthotropic.gettransv(gama)

    e1 = np.array([c, c, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    e3 = np.cross(e1, e2)

    nxt = lambda j: (j + 1) % 3                        # noqa: E731
    for j in range(3):
        k = nxt(j)
        # the products' SECOND index is the direction, the third the component
        assert np.allclose(qc[0, :, j], [e1[j] ** 2, e2[j] ** 2, e3[j] ** 2])
        assert np.allclose(qcg[0, :, j], [2 * e1[j] * e1[k], 2 * e2[j] * e2[k],
                                          2 * e3[j] * e3[k]])
        assert np.allclose(qgc[0, :, j], [e1[j] * e2[j], e2[j] * e3[j],
                                          e1[j] * e3[j]])
        assert np.allclose(qg[0, :, j], [e1[j] * e2[k] + e2[j] * e1[k],
                                         e2[j] * e3[k] + e3[j] * e2[k],
                                         e3[j] * e1[k] + e1[j] * e3[k]])


def test_gettransv_with_an_identity_frame_reduces_to_qc_and_qg_identity():
    """The record that pins the assembly: with ``GAMA = (e1, e2)`` the Fortran
    gives ``QC = QG = I`` and ``QCG = QGC = 0``.  That is what collapses
    :func:`mstiforthv` to ``CG = 0``, ``G33 = G``, ``CC`` unchanged — and it
    is a property worth asserting, because the four products are built from
    the same three direction vectors by different index pairs and a
    transposition error in any one of them would still look plausible."""
    gama = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    qc, qcg, qgc, qg = solid_orthotropic.gettransv(gama)
    assert np.allclose(qc[0], np.eye(3))
    assert np.allclose(qg[0], np.eye(3))
    assert np.allclose(qcg, 0.0)
    assert np.allclose(qgc, 0.0)


# ----------------------------------------------------------------------------
# 4. cbatran3v.F and mstiforthv.F — the sandwich and the moduli composition
# ----------------------------------------------------------------------------

def test_cbatran3v_is_the_sandwich_product():
    """cbatran3v.F: K(·,I,J) = sum_pq VQI(p,I) KK(p,q) VQJ(q,J) — i.e.
    VQI^T KK VQJ, NOT VQI KK VQJ.  ISYM=1 fills only the upper triangle from
    a symmetric KK; ISYM=0 fills all nine."""
    rng = np.random.default_rng(11)
    vqi = rng.normal(size=(1, 3, 3))
    vqj = rng.normal(size=(1, 3, 3))
    kk = rng.normal(size=(1, 3, 3))

    got = solid_orthotropic.cbatran3v(vqi, kk, vqj, isym=0)
    assert np.allclose(got[0], vqi[0].T @ kk[0] @ vqj[0])

    sym = 0.5 * (kk + kk.transpose(0, 2, 1))
    got1 = solid_orthotropic.cbatran3v(vqi, sym, vqj, isym=1)
    full = vqi[0].T @ sym[0] @ vqj[0]
    for i in range(3):
        for j in range(i, 3):
            assert np.isclose(got1[0, i, j], full[i, j])


def test_mstiforthv_composition_matches_the_fortran_weights():
    """mstiforthv.F: CG = HALF*A + TWO*B (A from QC·CC·QCG, B from
    QGC·G·QG), G33 = FOURTH*A + B (A from QCG·CC·QCG), and CC = A + FOUR*B
    (A from QC·CC·QC), then the strict-upper triangle is mirrored.  With the
    identity cosine frame that reduces to CG = 0, G33 = G, CC unchanged."""
    cc = np.zeros((1, 3, 3))
    cc[0, 0, 0], cc[0, 1, 1], cc[0, 2, 2] = 1.0e11, 2.0e11, 3.0e11
    cc[0, 0, 1] = cc[0, 1, 0] = 4.0e10
    cc[0, 1, 2] = cc[0, 2, 1] = 5.0e10
    cc[0, 0, 2] = cc[0, 2, 0] = 6.0e10
    g3 = np.array([[7.0e10, 8.0e10, 9.0e10]])
    gama = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    qc, qcg, qgc, qg = solid_orthotropic.gettransv(gama)

    cc_o, cg_o, g33_o = solid_orthotropic.mstiforthv(cc.copy(), g3.copy(),
                                                     qc, qcg, qgc, qg)
    assert np.allclose(cg_o, 0.0)
    assert np.allclose(np.diag(g33_o[0]), g3[0])
    assert np.allclose(cc_o, cc)


def test_mmod_norm_divides_every_modulus_by_the_same_G():
    """mmod_norm.F: CC, G33 and CG are all divided by GG, the upper triangle
    first and mirrored."""
    cc = np.ones((2, 3, 3))
    cg = 2.0 * np.ones((2, 3, 3))
    g33 = 3.0 * np.ones((2, 3, 3))
    cc_o, cg_o, g33_o = solid_orthotropic.mmod_norm(cc, cg, g33,
                                                    np.array([2.0, 4.0]))
    assert np.allclose(cc_o[0], 0.5)
    assert np.allclose(cg_o[0], 1.0)
    assert np.allclose(g33_o[0], 1.5)
    assert np.allclose(cc_o[1], 0.25)


# ----------------------------------------------------------------------------
# 5. The rotation order — sordeft3.F / sroto3.F / szordef3.F
# ----------------------------------------------------------------------------

def _frame(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([np.cos(t), np.sin(t), 0.0,
                     -np.sin(t), np.cos(t), 0.0])


def test_sordeft3_into_the_material_frame_and_sroto3_back_out():
    """``sordeft3.F`` carries a tensor INTO the material frame as
    ``R M Rᵀ`` — ``M'_ab = Σ_cd e_a(c) M_c(d) e_b(d)``, the ``e_a`` the ROWS of
    the frame — and ``sroto3.F``'s operator ``Rᵀ σ R`` carries it back.

    Neither is an involution (``R M Rᵀ`` twice gives ``R M Rᵀ Rᵀ Mᵀ Rᵀ``), so
    "rotate there and back" is the wrong assertion — it would fail for the
    right operator.  What is pinned: the order, against a hand-computed 90°
    case; that the pair are exact inverses of each other, which is the property
    the cycle depends on; and that both preserve the trace.
    """
    # a 90 deg rotation about z puts material axis 1 on +y and axis 2 on -x:
    # e_1 = (0, 1, 0), e_2 = (-1, 0, 0), e_3 = (0, 0, 1)
    g90 = np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    m = np.zeros((1, 3, 3))
    m[0, 0, 0] = 1.0                                  # m_xx
    m[0, 0, 1] = 1.0                                  # m_xy
    got = solid_orthotropic.sordeft3(m, g90)[0]
    # M'_ab = (e_a)_c M_cd (e_b)_d: m_xx has only an x component and only
    # e_2 has one (-1), so it lands on the (2, 2) slot
    assert np.isclose(got[1, 1], 1.0)
    # m_xy: only (e_2)_x (e_1)_y = -1 survives -> slot (2, 1)
    assert np.isclose(got[1, 0], -1.0)
    assert np.isclose(got[0, 0], 0.0)
    assert np.isclose(got[2, 2], 0.0)

    # the pair are exact inverses -- the property the cycle relies on
    for theta in (0.0, 17.0, 30.0, 47.0, 90.0):
        g = _frame(theta)
        t = np.arange(1.0, 10.0).reshape(1, 3, 3)
        t = t + t.transpose(0, 2, 1)
        assert np.allclose(
            solid_orthotropic.sordeft3(
                _as_tensor(solid_orthotropic.sroto3(_from_tensor(t), g)), g),
            t, atol=1e-12), theta
        v = np.arange(1.0, 7.0).reshape(1, 6) * 1.0e8
        assert np.allclose(
            solid_orthotropic.sroto3(
                _from_tensor(solid_orthotropic.sordeft3(_as_tensor(v), g)), g),
            v, rtol=1e-12, atol=1e-6), theta

    # the trace is the rotation invariant of both
    rng = np.random.default_rng(5)
    for theta in (0.0, 30.0, 47.0, 90.0):
        g = _frame(theta)
        mm = rng.normal(size=(1, 9)).reshape(1, 3, 3)
        mm = mm + mm.transpose(0, 2, 1)
        sv = rng.normal(size=(1, 6))
        assert np.isclose(np.trace(solid_orthotropic.sordeft3(mm, g)[0]),
                          np.trace(mm[0])), theta
        rot = solid_orthotropic.sroto3(sv, g)
        assert np.isclose(rot[0, 0] + rot[0, 1] + rot[0, 2],
                          sv[0, 0] + sv[0, 1] + sv[0, 2]), theta


def _as_tensor(v):
    """(1, 6) Voigt -> (1, 3, 3), the input shape :func:`sordeft3` takes."""
    t = np.zeros((1, 3, 3))
    t[0, 0, 0], t[0, 1, 1], t[0, 2, 2] = v[0, 0], v[0, 1], v[0, 2]
    t[0, 0, 1] = t[0, 1, 0] = v[0, 3]
    t[0, 1, 2] = t[0, 2, 1] = v[0, 4]
    t[0, 0, 2] = t[0, 2, 0] = v[0, 5]
    return t


def _from_tensor(t):
    """(1, 3, 3) -> (1, 6) Voigt, the output shape :func:`sroto3` returns."""
    v = np.empty((1, 6))
    v[0, 0], v[0, 1], v[0, 2] = t[0, 0, 0], t[0, 1, 1], t[0, 2, 2]
    v[0, 3], v[0, 4], v[0, 5] = t[0, 0, 1], t[0, 1, 2], t[0, 0, 2]
    return v


def test_sordeft3_at_zero_angle_is_the_identity():
    g = _frame(0.0)
    m = np.arange(1.0, 10.0).reshape(1, 3, 3)
    assert np.allclose(solid_orthotropic.sordeft3(m, g), m)


# ----------------------------------------------------------------------------
# 6. scoor_cp2sp.F — what it actually is
# ----------------------------------------------------------------------------

def test_scoor_cp2sp_is_a_coordinate_split_not_a_rotation():
    """The plan calls scoor_cp2sp.F "the strain-rate-to-cotangent
    transform".  Read, it is neither: it copies X0(I,1..8)/Y0/Z0 (DOUBLE
    PRECISION) into the 24 separate single-precision X1..Z8 arrays.  This
    test pins THAT, so the citation cannot drift back to the description."""
    x0 = np.arange(1.0, 9.0)[None, :] * 0.5
    y0 = np.arange(2.0, 10.0)[None, :] * 0.25
    z0 = np.arange(3.0, 11.0)[None, :] * 0.125
    cols = solid_orthotropic.scoor_cp2sp(x0, y0, z0)
    assert len(cols) == 24
    for j in range(8):
        assert np.array_equal(cols[j], x0[:, j])
        assert np.array_equal(cols[8 + j], y0[:, j])
        assert np.array_equal(cols[16 + j], z0[:, j])


# ----------------------------------------------------------------------------
# 7. sz_dt1.F90 — the hourglass length, verbatim
# ----------------------------------------------------------------------------

def test_sz_dt1_length_matches_the_fortran_formula():
    """sz_dt1.F90: for gfac < 1, deltax1 = 1/sqrt(d) with
    d = 4*sqrt(third*max(-p,0)) - (2/3)*aa, aa = -(pxx+pyy+pzz) and
    p = bb - third*aa^2;  for gfac >= 1 it is exactly zero."""
    rng = np.random.default_rng(7)
    p = rng.normal(size=(4, 3, 4))
    gfac = 0.5
    got = solid_orthotropic.sz_dt1(p[..., 0], p[..., 1], p[..., 2], gfac)

    pxx = 2.0 * np.sum(p[..., 0] ** 2, axis=1)
    pyy = 2.0 * np.sum(p[..., 1] ** 2, axis=1)
    pzz = 2.0 * np.sum(p[..., 2] ** 2, axis=1)
    pxy = 2.0 * np.sum(p[..., 0] * p[..., 1], axis=1)
    pxz = 2.0 * np.sum(p[..., 0] * p[..., 2], axis=1)
    pyz = 2.0 * np.sum(p[..., 1] * p[..., 2], axis=1)
    aa = -(pxx + pyy + pzz)
    bb = gfac * (pxx * pyy + pxx * pzz + pyy * pzz - pxy ** 2 - pxz ** 2 - pyz ** 2)
    pp = bb - aa ** 2 / 3.0
    d = 4.0 * np.sqrt(np.maximum(-pp, 0.0) / 3.0) - 2.0 * aa / 3.0
    assert np.allclose(got, 1.0 / np.sqrt(d))

    zeros = solid_orthotropic.sz_dt1(p[..., 0], p[..., 1], p[..., 2], 1.0)
    assert np.array_equal(zeros, np.zeros(4))


# ----------------------------------------------------------------------------
# 8. The orthotropic branch: it must actually do something
# ----------------------------------------------------------------------------

def _law17_mat(e1, e2, e3, g, rho0=1000.0, nu=0.2):
    """A /MAT/LAW17 (3D orthotropic elastic) material — the port's real
    orthotropic SOLID law, ``mat017/m17law.F``.  The forces tests need it
    because a synthetic law number reaches no stress update at all, and a
    cycle whose stress never leaves zero is indistinguishable from a kernel
    that does nothing.

    ``nu`` goes in the params dict under all three of LAW17's Poisson names:
    the law reads ``mat.params``, not the ``mat.nu`` field, so setting only
    the field silently gives the 0.3 default and an "isotropic" card with a
    different nu in each direction.
    """
    return _Mat(law=17, rho0=rho0, e1=e1, e2=e2, e3=e3,
                g12=g, g23=g, g31=g,
                nu12=nu, nu23=nu, nu31=nu)


def test_orthotropic_group_differs_from_the_isotropic_one():
    """An anisotropic material in a ROTATED material frame must produce a
    different force than the same material axis-aligned.

    The isotropic-limit test above is only worth anything if the kernel is not
    hard-wired to the isotropic path, and the material must actually be
    anisotropic for the rotation to have anything to bite on — an isotropic
    one is rotation-invariant, so this uses three distinct Young's moduli.
    """
    v = None
    fints = {}
    for name, prop in (("aligned", _Prop(orthtrop=1, vx=1.0, vy=0.0, vz=0.0)),
                       ("rotated", _Prop(orthtrop=1, vx=1.0, vy=0.0, vz=0.0,
                                        phi=30.0))):
        model, group = _unit_hexa8_group(
            mat=_law17_mat(2.0e11, 1.4e11, 0.8e11, 0.3e11), prop=prop)
        solid_orthotropic.init_group(group, model, _Log())
        if v is None:
            v = _loading_state(model)
        fints[name], _ = _run(solid_orthotropic, model, group, v)
    assert not np.allclose(fints["aligned"], fints["rotated"])


def test_isotropic_material_in_a_rotated_frame_matches_the_isotropic_kernel():
    """An ISOTROPIC material is rotation-invariant: rotating its material
    frame must not change the force at all.  This is the check that the
    rotation is applied to the moduli and not to anything else (the strain,
    the geometry, the hourglass), which a rotated anisotropic material alone
    cannot distinguish.

    The card has to be isotropic *consistently*: equal E's and equal nu's alone
    are not enough, because LAW17 also carries the shear moduli G12/G23/G31
    independently, and an isotropic material needs ``a - b == G`` on the normal
    block.  Equal E and nu with an unrelated G gives ``a - b = 2G``, which is
    genuinely anisotropic — and then a rotated frame legitimately changes the
    force, so the assertion would be testing the wrong thing."""
    nu = 0.2
    e = 2.0e11
    g = e / (2.0 * (1.0 + nu))               # the one G that makes it isotropic
    v = None
    fints = {}
    for name, prop in (("aligned", _Prop(orthtrop=1, vx=1.0, vy=0.0, vz=0.0)),
                       ("rotated", _Prop(orthtrop=1, vx=1.0, vy=0.0, vz=0.0,
                                        phi=37.0))):
        model, group = _unit_hexa8_group(
            mat=_law17_mat(e, e, e, g, nu=nu), prop=prop)
        solid_orthotropic.init_group(group, model, _Log())
        if v is None:
            v = _loading_state(model)
        fints[name], _ = _run(solid_orthotropic, model, group, v)
    assert np.allclose(fints["aligned"], fints["rotated"], rtol=1e-12)


def test_orthotropic_stress_update_uses_the_rotated_frame():
    """With IORTH > 0 the rate is rotated into the material frame before the
    law and the stress back afterwards (szforc3.F: SZTORTH3/SORDEFT3, then
    MMAIN with MFXX..MFZZ and GAMA, then SROTO3).  ``material_frame`` puts
    material axis 1 on the element z axis for ``(vx,vy,vz) = (0,0,1)``, axis 2
    on x and axis 3 on y, so an element-frame pull along z is a pull along
    material axis 1 and picks up C11, not C33."""
    C = 2.0e11
    model, group = _unit_hexa8_group(
        mat=_Mat(law=MTN_DIRECT, rho0=1000.0,
                 **_ortho_params(C, C, 0.5 * C, 0.1 * C, 0.1 * C, 0.1 * C)),
        prop=_Prop(orthtrop=1, vx=0.0, vy=0.0, vz=1.0))
    solid_orthotropic.init_group(group, model, _Log())

    gama = group.state["gama"][0]
    assert np.allclose(gama[0:3], [0.0, 0.0, 1.0])          # e1 = element z
    assert np.allclose(gama[3:6], [1.0, 0.0, 0.0])          # e2 = element x
    # ... so material axis 3 (G3 = e1 x e2) is element y

    # The frame's rows are the material directions in ELEMENT coordinates, so
    # an element-frame direction x_j loads material axis j:
    #   element z -> material 1 -> C11, element x -> material 2 -> C22,
    #   element y -> material 3 -> C33.
    sig_z = solid_orthotropic.stress_from_strain(
        group, np.array([[0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0]]))
    assert np.isclose(sig_z[0, 2], C * 1.0e-3, rtol=1e-12)
    assert np.allclose(sig_z[0, [0, 1, 3, 4, 5]], 0.0, atol=1e-6)

    # element x == material 2  ->  sigma_xx = C22 = C
    sig_x = solid_orthotropic.stress_from_strain(
        group, np.array([[1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.isclose(sig_x[0, 0], C * 1.0e-3, rtol=1e-12)
    assert np.allclose(sig_x[0, [1, 2, 3, 4, 5]], 0.0, atol=1e-6)

    # element y == material 3  ->  sigma_yy = C33 = C/2
    sig_y = solid_orthotropic.stress_from_strain(
        group, np.array([[0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0]]))
    assert np.isclose(sig_y[0, 1], 0.5 * C * 1.0e-3, rtol=1e-12)


# ----------------------------------------------------------------------------
# 9. tangent / kgeo / dt_claim — required, and never zero
# ----------------------------------------------------------------------------

def test_tangent_is_the_consistent_stiffness_and_never_zero():
    """K_e = V Bᵀ D B + K_hg, with D the element-frame Voigt moduli.

    Checked against the expression rather than a single hand number, because
    the entry ``K[0,0]`` is node 0's x translation only and the one-point
    B-matrix's row mixes all eight nodes — the two differ by the shear-modulus
    terms a scalar assertion would hide.  The constitutive part is isolated by
    removing the hourglass block the port shares with ``solid_hexa8``.
    """
    C11 = 2.0e11
    mat = _Mat(law=MTN_DIRECT, rho0=1000.0,
               **_ortho_params(C11, 3.0e11, 4.0e11, 5.0e10, 6.0e10, 7.0e10))
    model, group = _unit_hexa8_group(mat=mat)
    solid_orthotropic.init_group(group, model, _Log())
    ke, edofs = solid_orthotropic.tangent(group, model.x)

    assert ke.shape == (1, 24, 24)
    assert edofs.shape == (1, 24)
    assert np.array_equal(edofs, solid_hexa8._edofs(group.conn))
    assert np.any(ke != 0.0), "a zero tangent is the failure this test exists for"
    assert np.allclose(ke, ke.transpose(0, 2, 1))

    # subtract the shared hourglass block and what remains must be exactly
    # V B^T D B for the moduli the port built
    dndx, vol = solid_hexa8._geometry(model.x[group.conn])
    ix = np.arange(8)
    gx, gy, gz = dndx[:, :, 0], dndx[:, :, 1], dndx[:, :, 2]
    B = np.zeros((1, 6, 24))
    B[:, 0, 3 * ix + 0] = gx
    B[:, 1, 3 * ix + 1] = gy
    B[:, 2, 3 * ix + 2] = gz
    B[:, 3, 3 * ix + 0] = gy
    B[:, 3, 3 * ix + 1] = gx
    B[:, 4, 3 * ix + 1] = gz
    B[:, 4, 3 * ix + 2] = gy
    B[:, 5, 3 * ix + 0] = gz
    B[:, 5, 3 * ix + 2] = gx
    expect = vol[:, None, None] * np.einsum("nji,njk,nkl->nil", B,
                                            group.state["D"], B, optimize=True)
    _c, _g, GG, k_hg, _ks = solid_hexa8._hg_operators(group, model.x)
    for b in range(3):
        expect[:, (3 * ix + b)[:, None], (3 * ix + b)[None, :]] += k_hg[:, None, None] * GG
    assert np.allclose(ke, expect, rtol=1e-12)
    # ... and the axial entry is the one-point B's share of C11, not 2*C11
    assert ke[0, 0, 0] > 0.0


def test_tangent_is_never_zero_for_any_orthotropic_group():
    for name, params in (
        ("decoupled", _ortho_params(2.0e11, 2.0e11, 2.0e11, 0.0, 0.0, 0.0)),
        ("shear-only", _ortho_params(0.0, 0.0, 0.0, 5.0e10, 5.0e10, 5.0e10)),
        ("laminated", _ortho_params(1.0e11, 1.2e11, 1.4e11, 2.0e10, 3.0e10, 4.0e10)),
    ):
        model, group = _unit_hexa8_group(mat=_Mat(law=MTN_DIRECT, rho0=1000.0, **params))
        solid_orthotropic.init_group(group, model, _Log())
        ke, _ = solid_orthotropic.tangent(group, model.x)
        assert np.any(ke != 0.0), f"{name}: tangent is identically zero"


def test_kgeo_is_zero_without_stress_and_the_initial_stress_form_with_it():
    mat = _Mat(law=MTN_DIRECT, rho0=1000.0,
               **_ortho_params(2.0e11, 2.0e11, 2.0e11, 5.0e10, 5.0e10, 5.0e10))
    model, group = _unit_hexa8_group(mat=mat)
    solid_orthotropic.init_group(group, model, _Log())

    ke0, _ = solid_orthotropic.kgeo(group, model.x)
    assert ke0.shape == (1, 24, 24)
    assert np.allclose(ke0, 0.0)

    group.state["sig"][:] = np.array([[1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ke1, _ = solid_orthotropic.kgeo(group, model.x)
    assert np.any(ke1 != 0.0)
    assert np.allclose(ke1, ke1.transpose(0, 2, 1))

    # uniaxial tension along x: g_ab = V gradN_a . S . gradN_b with S carrying
    # 1e8 on xx alone, so g_ab = 1e8 V (dN_a/dx)(dN_b/dx).  The port spreads g
    # into the three 8x8 diagonal blocks of the 24x24 addressing, so the block
    # (b, b) holds the g of translation direction b and the off-diagonal
    # blocks stay zero — a stress in ONE direction does NOT make the other two
    # blocks vanish, it makes them all equal to the same g.
    dndx, vol = solid_hexa8._geometry(model.x[group.conn])
    s = np.zeros((1, 3, 3))
    s[:, 0, 0] = 1.0e8
    g = vol * np.einsum("nac,ncd,nbd->nab", dndx, s, dndx)
    ix = np.arange(8)
    for b in range(3):
        block = ke1[0][np.ix_(3 * ix + b, 3 * ix + b)]
        assert np.allclose(block, g[0]), b
    assert np.allclose(ke1[0, 0, 1], 0.0)          # cross-direction is zero
    assert np.isclose(g[0, 0, 0], 1.0e8 * vol[0] * 0.25 ** 2)


def test_dt_claim_is_positive_and_matches_the_isotropic_kernel():
    model, ref_group = _unit_hexa8_group()
    solid_hexa8.init_group(ref_group, model, _Log())
    ref = solid_hexa8.forces(ref_group, model.x, None, model.vr, 0.0, None, None)

    model2, got_group = _unit_hexa8_group()
    solid_orthotropic.init_group(got_group, model2, _Log())
    got = solid_orthotropic.dt_claim(got_group, model2.x)
    assert got.shape == (1,)
    assert np.all(got > 0.0)
    assert np.allclose(got, ref)


def test_dt_claim_shortens_as_the_material_stiffens():
    """dt = length / c and c = sqrt(D11 / rho): a stiffer material must claim
    a shorter step.  Both groups carry ``IORTH = 0``, so this is the
    *isotropic* branch — the one that delegates to ``solid_hexa8`` — which is
    exactly where a dt claim that silently ignored the moduli would still
    look plausible."""
    soft, _ = _dt_claim_for(2.0e11)
    stiff, _ = _dt_claim_for(2.0e12)
    assert soft > 0.0
    assert stiff < soft


def test_dt_claim_shortens_as_the_material_stiffens_when_orthotropic():
    """The same, through the orthotropic branch (``IORTH > 0``), where the
    claim is built from the element-frame moduli rather than delegated."""
    soft, _ = _dt_claim_for(2.0e11, orthotropic=True)
    stiff, _ = _dt_claim_for(2.0e12, orthotropic=True)
    assert soft > 0.0
    assert stiff < soft


def _dt_claim_for(E, orthotropic=False):
    # The isotropic branch (IORTH == 0) delegates to solid_hexa8, which reads
    # mat.E / mat.K / mat.G -- so E must be set here as well as in the ortho
    # card, or both claims come out of the same default and compare equal.
    mat = _Mat(law=MTN_DIRECT, rho0=1000.0, E=E,
               **_ortho_params(E, E, E, 0.3 * E, 0.3 * E, 0.3 * E))
    prop = _Prop(orthtrop=1, vx=0.0, vy=0.0, vz=1.0) if orthotropic else _Prop()
    model, group = _unit_hexa8_group(mat=mat, prop=prop)
    solid_orthotropic.init_group(group, model, _Log())
    return solid_orthotropic.dt_claim(group, model.x), group


# ----------------------------------------------------------------------------
# 10. Empty / degenerate groups, and the module's exported surface
# ----------------------------------------------------------------------------

def test_empty_group_is_handled_everywhere():
    model = Model()
    model.x0 = np.zeros((0, 3))
    model.x = np.zeros((0, 3))
    model.v = np.zeros((0, 3))
    model.vr = np.zeros((0, 3))
    group = ElementGroup(ids=np.empty(0, dtype=np.int64),
                         conn=np.empty((0, 8), dtype=np.int64),
                         part=np.empty(0, dtype=np.int64))
    group.state["slices"] = []
    log = _Log()

    nids, masses, _ = solid_orthotropic.init_group(group, model, log)
    assert len(nids) == 0 and len(masses) == 0
    assert log.errors == []
    assert solid_orthotropic.forces(group, model.x, None, model.vr, 1e-3,
                                    None, None).size == 0
    ke, dofs = solid_orthotropic.tangent(group, model.x)
    assert ke.shape == (0, 24, 24) and dofs.shape == (0, 24)
    kg, _ = solid_orthotropic.kgeo(group, model.x)
    assert kg.shape == (0, 24, 24)
    assert solid_orthotropic.dt_claim(group, model.x).shape == (0,)


def test_module_exports_the_registration_surface_p2_11_needs():
    for name in ("init_group", "forces", "dt_claim", "tangent", "kgeo"):
        assert callable(getattr(solid_orthotropic, name)), name


def test_registration_is_still_p2_11s_job():
    """This task creates the module; it does NOT register it.  The guard is
    here so the wave-1/2 split stays honest: P2.11 owns the group."""
    from pyradioss import elements
    assert not any("orthotropic" in name for name in elements.KERNELS)
    assert elements.SOLID_ISOLID_GROUPS.get(11) is None
    assert elements.SOLID_ISOLID_GROUPS.get(12) is None


@pytest.mark.parametrize("shape", [(0,), (1, 2, 3)])
def test_cbatran3v_and_gettransv_tolerate_degenerate_input(shape):
    n = shape[0]
    rng = np.random.default_rng(1)
    out = solid_orthotropic.cbatran3v(
        np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), isym=0)
    assert out.shape == (n, 3, 3)
    qc, qcg, qgc, qg = solid_orthotropic.gettransv(np.zeros((n, 6)))
    assert qc.shape == qcg.shape == qgc.shape == qg.shape == (n, 3, 3)
    assert rng is not None


def test_hourglass_moduli_are_finite_and_stored():
    """The (CC, CG, G33) triple szsvm_or.F / szhour3_or.F consume is built and
    stored even though those consumers are not ported — it is the module's
    export for them."""
    C = 2.0e11
    model, group = _unit_hexa8_group(
        mat=_Mat(law=MTN_DIRECT, rho0=1000.0,
                 **_ortho_params(C, C, C, 5.0e10, 5.0e10, 5.0e10)),
        prop=_Prop(orthtrop=1, vx=0.0, vy=1.0, vz=0.0))
    solid_orthotropic.init_group(group, model, _Log())
    mods = solid_orthotropic.hourglass_moduli(group)
    for key in ("cc", "cg", "g33"):
        assert mods[key].shape == (1, 3, 3), key
        assert np.all(np.isfinite(mods[key])), key
    for key in ("cc", "cg", "g33"):
        assert np.array_equal(mods[key], group.state[key])


def test_nothing_in_the_module_cites_the_isolid_24_variant():
    """``mmodul24c.F`` is the Isolid=24 (HEPH) reinforcement variant: it reads
    PM(50) and the PM(53..55) armature fractions and assembles through
    ``C33STIF2EL``.  It must not leak into the base kernel's defaults -- but the
    module is expected to NAME it, to record that it was considered and
    excluded.  So the check is on the CODE, not the prose: Isolid=24 gets no
    branch of its own, and its symbols appear only inside docstrings."""
    source = pathlib.Path(inspect.getsourcefile(solid_orthotropic)).read_text()
    # named in the docstring, as an explicit exclusion
    assert "mmodul24c" in source
    # no branch of its own in either dispatch table
    assert "Isolid=24" not in solid_orthotropic.DIRECT_BRANCHES
    assert "Isolid=24" not in solid_orthotropic.NU_BRANCHES
    # its symbols survive only in docstrings: strip them and require them gone
    code = re.sub(r'\"\"\".*?\"\"\"', "", source, flags=re.S)
    code = re.sub(r"'''.*?'''", "", code, flags=re.S)
    for symbol in ("mmodul24c", "c33stif2el", "m1tot_stab24"):
        assert symbol not in code.lower(), symbol
