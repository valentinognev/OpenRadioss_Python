"""P3.3 — Orthotropic QEPH stabilization: ``czfintn.F`` CZFINTN_OR (HM/HF).

Upstream map
------------
``engine/source/elements/shell/coquez/czforc3.F:678-724`` dispatches
``IMPL_S > 0 .OR. IORTH == 0`` to ``CZFINTN1`` (isotropic) and everything
else to ``CZFINTN_OR``.  ``IORTH`` is set by MATERIAL type alone —
``cmatc3.F:211-217`` sets it for ``MTN == 19`` — and the moduli it consumes
are built by ``GEPM_LC`` (``cmatc3.F:477-513``, from ``PM(33..39)`` =
E11, E22, N12, N21, G12, G23, G31 per ``hm_read_mat19.F:165-179``), rotated
into the element frame by ``CCTOGLOB`` (``cmatc3.F:412-470``) and halved by a
twelfth into the bending set ``HF`` (``cmatc3.F:225-231``).

Two upstream facts this module pins, because they are the whole task:

* the substitution list is NOT symmetric.  ``czfintn.F:1228-1248`` (the
  IORTH==0 ELSE inside ``CZFINTN_OR``) uses ``C1M = A11`` for BOTH x and y
  membrane/bending slots — the port's isotropic rows are ``CZFINTN1``'s at
  ``czfintn.F:184-227``; ``czfintn.F:1164-1185`` (``IORTH>0``) uses ``HM(1)``
  on the x rows and ``HM(2)`` on the y rows, and the bending rows use
  ``HF(1)``/``HF(2)`` (a TWELFTH of them) rather than ``A11``;
* ``CCTOGLOB``'s ``CM(4)`` is ``s^2c^2(C11+C22-2C12-2C66) + (c^4+s^4)C66``
  (``cmatc3.F:446-448``), NOT the classical-laminate
  ``s^2c^2(C11+C22+2C12) + (c^4+s^4)C66``.  On the RD-V-0230 fabric at 45
  degrees upstream gives 0.200 and the textbook gives 0.225.  Upstream wins.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_qeph, shell_qeph_ortho
from pyradioss.input import mat_reader
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)

# Every deck here parses /MAT/FABRI through the cfg-driven generic reader,
# which needs OpenRadioss's hm_cfg_files (not vendored — license; CI fetches
# a sparse checkout, ci.yml). Skip loudly without them (PORTING_GUIDE M37).
pytestmark = pytest.mark.skipif(
    mat_reader.catalogue().schema("FABRI") is None,
    reason="hm_cfg_files CFG tree not found — set PYRADIOSS_HM_CFG "
           "(see ci.yml / PORTING_GUIDE M37)")

DT = 1.0e-3
T = 0.1

# the RD-V-0230 /MAT/FABRI/1 constants (E11 != E22 — the whole point):
# E11 = 0.3, E22 = 0.5, NU12 = 0, G12 = G23 = G31 = 0.05
FABRI = ("/MAT/FABRI/1\nfabric\n8e-7\n"
         "0.3 0.5 0.0\n"
         "0.05 0.05 0.05\n")
STEEL = "/MAT/LAW1/1\nsteel elastic\n7.8e-6\n210. 0.3\n"

DIST = "1 0 0 0\n2 1.3 0.1 0\n3 1.1 0.9 0\n4 -0.2 1.2 0\n"

# GEPM_LC (cmatc3.F:477-513) on the FABRI constants
E11, E22, NU12 = 0.3, 0.5, 0.0
NU21 = NU12 * E22 / E11                 # hm_read_mat19.F:121
DETC = 1.0 - NU12 * NU21                # GEPM_LC's NU12 local (cmatc3.F:498)
HM_FIBRE = (E11 / DETC, E22 / DETC, NU21 * E11 / DETC, 0.05)


def _deck(prop, mat=FABRI, ishell=24, nip=1, vx=1.0, vy=0.0):
    return ("/BEGIN\np33 qeph ortho\n/NODE\n" + DIST +
            "/SHELL/1\n1 1 2 3 4\n"
            "/PART/1\nplate\n1 1\n" + mat +
            prop + f"/END\n")


def _ortho_prop(vx=1.0, vy=0.0, nip=1):
    return ("/PROP/SH_ORTH/1\nfabric prop\n"
            f"24 11 0 0 0\n"
            "0 0 0 0 0\n"
            f"{nip} 0 {T}\n"
            f"{vx} {vy} 0 0\n")


def _shell_prop(nip=1):
    return ("/PROP/SHELL/1\nplate prop\n"
            "24 0 0 0\n0 0 0 0 0\n"
            f"{nip} 0 {T}\n")


def _build(tmp_path, prop, mat=FABRI, name="K"):
    f = tmp_path / f"{name}_0000.rad"
    f.write_text(_deck(prop, mat=mat))
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    assert model.shells_qeph is not None and model.shells_qeph.n == 1
    return model


def _group_law19(tmp_path, nip=3, vx=1.0, vy=0.0):
    m = _build(tmp_path, _ortho_prop(vx, vy, nip=nip))
    return m, m.shells_qeph


def _dglas(model, anisotropic, comp=2):
    """DGLAS (n, 12) on the hourglass h = (1,-1,1,-1) velocity mode, both
    branches.  ``comp`` is the velocity component carrying the mode: 2 (z)
    exercises transverse shear only, 0 (x) the membrane rows."""
    g = model.shells_qeph
    v = np.zeros_like(model.x)
    v[g.conn, comp] = np.array([1.0, -1.0, 1.0, -1.0])   # h = (1,-1,1,-1)
    G = shell_qeph._geometry(model.x[g.conn])
    v13, v24, vhi, rl, _plat, _vqn, _di, _db = shell_qeph._kinematics(
        G, v[g.conn], np.zeros_like(v)[g.conn], DT, g.state["npt1"])
    _vdef, vhg = shell_qeph._rates(G, v13, v24, vhi, rl, np.ones(g.n, bool))
    st = dict(g.state)
    if not anisotropic:
        st["hm"] = None
        st["hc"] = None
    return shell_qeph_ortho.elastic_increment(G, st, vhg, DT, shell_qeph._CVIS)


# ---------------------------------------------------------------------------
# GEPM_LC — materials.shell_moduli
# ---------------------------------------------------------------------------

def test_shell_moduli_are_the_gepm_lc_quadruple(tmp_path):
    """cmatc3.F:477-513: (C11, C22, C12, G12) with C11 = E11/(1-N12*N21),
    C12 = N21*C11 — the FOUR entries, not the isotropic (A11, A12, G)."""
    _m, g = _group_law19(tmp_path)
    mat = g.state["slices"][0][1]
    c11, c22, c12, g12, hc1, hc2 = materials.shell_moduli.shell_moduli(mat, 19)
    assert (c11, c22, c12, g12) == pytest.approx(HM_FIBRE)
    # the two-argument call carries the UNFOLDED transverse pair — SHF is
    # folded once, by the caller (group_moduli), exactly as GEPM_LC folds
    # HC(1) = G23*SHF / HC(2) = G31*SHF (cmatc3.F:508-509) a single time
    assert (hc1, hc2) == pytest.approx((0.05, 0.05))
    assert materials.shell_moduli.shell_moduli(
        mat, 19, shf=5.0 / 6.0)[4:] == pytest.approx(
            (0.05 * 5.0 / 6.0, 0.05 * 5.0 / 6.0))
    # and the isotropic collapse the deferred path was running, for contrast
    assert c11 == pytest.approx(0.3)
    assert float(g.state["a11"][0]) == pytest.approx(max(E11, E22) / DETC)


def test_shell_moduli_is_none_for_a_non_orthotropic_law(tmp_path):
    m = _build(tmp_path, _shell_prop(), mat=STEEL, name="L1")
    mat = m.shells_qeph.state["slices"][0][1]
    assert materials.shell_moduli.shell_moduli(mat, mat.law) is None


# ---------------------------------------------------------------------------
# CCTOGLOB — the fiber -> element rotation, transcribed verbatim
# ---------------------------------------------------------------------------

def test_cctoglob_is_the_identity_at_zero_degrees():
    """DIR = (1,0): CCTOGLOB must return HM unchanged (cmatc3.F:429-447)."""
    hm = np.array([list(HM_FIBRE)])
    rot, hc = shell_qeph_ortho.cctoglob(hm, np.array([[0.0416667, 0.0416667]]),
                                        np.array([[1.0, 0.0]]))
    assert rot[0] == pytest.approx(HM_FIBRE, abs=1e-15)
    assert hc[0] == pytest.approx([0.0416667, 0.0416667], abs=1e-15)


def test_cctoglob_swaps_the_axes_at_ninety_degrees():
    """DIR = (0,1): C11 <-> C22, C12 and C66 untouched (cmatc3.F:429-447)."""
    hm = np.array([list(HM_FIBRE)])
    rot, _hc = shell_qeph_ortho.cctoglob(
        hm, np.array([[0.04, 0.04]]), np.array([[0.0, 1.0]]))
    assert rot[0] == pytest.approx([E22, E11, HM_FIBRE[2], HM_FIBRE[3]], abs=1e-15)


def test_cctoglob_at_45_degrees_is_upstream_not_the_textbook():
    """cmatc3.F:429-448 on E11=0.3, E22=0.5, NU12=0, G12=0.05 at 45 deg.

    Upstream gives (0.25, 0.25, 0.15, 0.20).  The classical-laminate
    Q66 would be 0.225 — the divergence is UPSTREAM's, and the port
    transcribes it (see the module docstring)."""
    c = 1.0 / np.sqrt(2.0)
    hm = np.array([list(HM_FIBRE)])
    rot, _hc = shell_qeph_ortho.cctoglob(
        hm, np.array([[0.04, 0.04]]), np.array([[c, c]]))
    assert rot[0] == pytest.approx([0.25, 0.25, 0.15, 0.20], abs=1e-12)
    textbook_q66 = 0.25 * (0.3 + 0.5) + 0.5 * 0.05
    assert textbook_q66 == pytest.approx(0.225)
    assert not np.isclose(rot[0, 3], textbook_q66)


def test_cctoglob_mixes_the_transverse_moduli():
    """cmatc3.F:452-458: HC(1) = M2*HC(1) + N2*HC(2) (and HC(2) the mirror),
    which at 45 deg collapses both onto their mean."""
    hc = np.array([[0.06, 0.02]])
    _rot, out = shell_qeph_ortho.cctoglob(np.array([list(HM_FIBRE)]), hc,
                                          np.array([[1.0, 0.0]]))
    assert out[0] == pytest.approx([0.06, 0.02])
    _rot, out = shell_qeph_ortho.cctoglob(np.array([list(HM_FIBRE)]), hc,
                                          np.array([[0.0, 1.0]]))
    assert out[0] == pytest.approx([0.02, 0.06])


# ---------------------------------------------------------------------------
# the group carries the rotated moduli
# ---------------------------------------------------------------------------

def test_law19_group_carries_the_rotated_ortho_moduli(tmp_path):
    """cmatc3.F:222-232: GEPM_LC -> CCTOGLOB -> HF = HM/12, stored per
    element; the port keeps HM and derives HF inside the increment.

    The element's own fiber cosines are whatever
    ``shell_ortho.build_group_ortho`` resolved from the property V in the
    element's co-rotational frame, so the expected HM is the rotation of
    the fiber-frame quadruple by those cosines."""
    for vx, vy in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        m = _build(tmp_path, _ortho_prop(vx, vy), name=f"K{vx}{vy}")
        g = m.shells_qeph
        cs = g.state["ortho"]
        shf = g.state["shf"]
        assert g.state["hm"] is not None and g.state["hc"] is not None
        want, want_hc = shell_qeph_ortho.cctoglob(
            np.array([list(HM_FIBRE)]),
            np.array([[0.05 * shf[0], 0.05 * shf[0]]]), cs)
        assert g.state["hm"][0] == pytest.approx(want[0], abs=1e-14)
        assert g.state["hc"][0] == pytest.approx(want_hc[0], abs=1e-14)
    # and the three properties really do resolve to three DIFFERENT
    # fibre angles on the same material — the rotation is not a no-op
    ang = []
    for vx, vy in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        m = _build(tmp_path, _ortho_prop(vx, vy), name=f"A{vx}{vy}")
        c, s = m.shells_qeph.state["ortho"][0]
        ang.append(float(np.degrees(np.arctan2(s, c))))
    assert ang[2] - ang[0] == pytest.approx(45.0, abs=0.1)


def test_non_law19_qeph_group_stays_on_the_isotropic_moduli(tmp_path):
    """cmatc3.F:211-217 gates IORTH on MTN alone — a LAW1 QEPH group must
    keep hm = None and the a11/a12/gsr/a11sr/a12sr it had before."""
    m = _build(tmp_path, _shell_prop(), mat=STEEL, name="L1")
    g = m.shells_qeph
    assert g.state["hm"] is None and g.state["hc"] is None
    assert g.state["a11"][0] == pytest.approx(210.0 / (1.0 - 0.3 ** 2))
    assert g.state["a11sr"][0] == pytest.approx(np.sqrt(210.0 / 0.91))
    assert g.state["gsr"][0] == pytest.approx(np.sqrt(210.0 / (2.0 * 1.3)))


def test_law19_group_damper_moduli_follow_czfintn_or(tmp_path):
    """czfintn.F:1136-1140: GSR = sqrt(HM(4)), A11SR = sqrt((HM1+HM2)/2),
    A12SR = sqrt(MAX(EM20, HM3)) — NOT the isotropic sqrt(A11)/sqrt(nu)*A11SR.
    HM(3) is the ELEMENT-frame C12 (post-CCTOGLOB): nonzero here even
    though the fibre-frame C12 vanishes, because the deck's fibre sits
    slightly off the element axes."""
    _m, g = _group_law19(tmp_path)
    hm = g.state["hm"][0]
    assert g.state["gsr"][0] == pytest.approx(np.sqrt(hm[3]))
    assert g.state["a11sr"][0] == pytest.approx(
        np.sqrt(0.5 * (hm[0] + hm[1])))
    assert g.state["a12sr"][0] == pytest.approx(
        np.sqrt(max(1e-20, hm[2])))


# ---------------------------------------------------------------------------
# the increment itself — the adapted plan Step-1 acceptance test #2
# ---------------------------------------------------------------------------

def test_qeph_anisotropic_stabilisation_differs_from_isotropic(tmp_path):
    """The port's two DGLAS branches on a deck whose E11 != E22.

    czfintn.F:1148-1185 (IORTH>0) vs the CZFINTN1 rows it replaces: the
    asymmetric substitution (HM(1) on x, HM(2) on y, HF = HM/12 on
    bending) makes the two visibly different, and both are energetically
    admissible.  The mode rides on the x-component: a z-driven hourglass
    exercises only transverse shear, where HC == GS on this deck (all
    three shear moduli are 0.05) and the branches legitimately agree."""
    m, g = _group_law19(tmp_path, nip=3)
    dg_ortho = _dglas(m, True, comp=0)
    dg_iso = _dglas(m, False, comp=0)
    assert not np.allclose(dg_ortho, dg_iso)
    # the x-membrane row (DGLAS(1)) carries HM(1) ~= 0.30 orthotropic
    # against A11 = 0.50 isotropic, so the orthotropic increment is smaller
    assert abs(dg_ortho[0, 0]) < abs(dg_iso[0, 0])
    # the energy the stabilization books stays non-negative
    eint, ehour = g.state["eint"].sum(), g.state["ehour"].sum()
    assert eint == 0.0 and ehour == 0.0       # not yet stepped
    v = np.zeros_like(m.x)
    v[g.conn, 2] = np.array([1.0, -1.0, 1.0, -1.0])
    f = np.zeros_like(m.x)
    mm = np.zeros_like(m.x)
    shell_qeph.forces(g, m.x, v, np.zeros_like(v), DT, f, mm)
    assert g.state["eint"].sum() >= 0.0
    assert g.state["ehour"].sum() >= 0.0
    assert g.state["hgstr"].any()


def test_numba_mirror_is_bypassed_for_ortho_groups(tmp_path):
    """The numba ``qeph_post`` mirror only knows CZFINTN1's isotropic rows.
    An hm-carrying group must get the numpy ``_post`` (CZFINTN_OR branch)
    under every backend — never the isotropic mirror, silently."""
    from pyradioss import accel
    try:
        accel.select_backend("numba")
        if accel.get("qeph_post") is None:
            pytest.skip("numba backend unavailable")
        m_nb = _build(tmp_path, _ortho_prop(1.0, 0.0, nip=3), name="NB")
        g_nb = m_nb.shells_qeph
        assert g_nb.state["hm"] is not None
        v = np.zeros_like(m_nb.x)
        v[g_nb.conn, 0] = np.array([1.0, -1.0, 1.0, -1.0])
        f_nb = np.zeros_like(m_nb.x)
        shell_qeph.forces(g_nb, m_nb.x, v, np.zeros_like(v), DT, f_nb,
                          np.zeros_like(v))
        hg_nb = g_nb.state["hgstr"].copy()
    finally:
        accel.select_backend("numpy")
    m_np = _build(tmp_path, _ortho_prop(1.0, 0.0, nip=3), name="NP")
    g_np = m_np.shells_qeph
    v = np.zeros_like(m_np.x)
    v[g_np.conn, 0] = np.array([1.0, -1.0, 1.0, -1.0])
    f_np = np.zeros_like(m_np.x)
    shell_qeph.forces(g_np, m_np.x, v, np.zeros_like(v), DT, f_np,
                      np.zeros_like(v))
    assert np.array_equal(hg_nb, g_np.state["hgstr"])
    assert np.array_equal(f_nb, f_np)


def test_isotropic_branch_is_untouched_by_the_ortho_path(tmp_path):
    """A LAW1 QEPH group produces the same DGLAS with and without hm —
    i.e. the selection really is gated on IORTH (cmatc3.F:211-217)."""
    m = _build(tmp_path, _shell_prop(), mat=STEEL, name="L1")
    g = m.shells_qeph
    before = g.state["gsr"].copy(), g.state["a11sr"].copy(), g.state["a12sr"].copy()
    assert g.state["hm"] is None
    dg_a = _dglas(m, True)
    dg_b = _dglas(m, False)
    assert np.array_equal(dg_a, dg_b)
    assert all(np.array_equal(x, y) for x, y in zip(
        before, (g.state["gsr"], g.state["a11sr"], g.state["a12sr"])))