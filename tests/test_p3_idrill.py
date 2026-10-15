"""P3.5 — Drilling stiffness ``Idrill``/``Isrot`` across BT/QBAT/QEPH.

Upstream map (read from the Fortran, not extrapolated)
------------------------------------------------------
* Reader: ``starter/source/properties/shell/hm_read_prop01.F:145`` reads
  the card field ``Idrill`` into ``ISROT``; ``:214``
  ``IF(ISROT==0)ISROT=IDRIL_D`` (the ``/DEF_SHELL`` default,
  ``hm_read_defshell.F``); ``:215`` ``IF(ISROT==2) ISROT = 0``;
  ``:216`` ``IGEO(20)=ISROT``.
* Magnitude: ``engine/source/elements/sh3n/coquedk/cncoef3.F:86-91`` —
  ``KFAC = EM03`` (1e-3) explicit for quads, ``KRZ(I) = KFAC*G(I)``
  (``CNCOEF3`` for QBAT, ``CNCOEF3B`` for QEPH — same factor).
* QBAT (``IDRIL = IPARG(41)``, ``cbaforc3.F:335``): ``CBADEFRZ``
  (``cbadef.F:844``) on the strain side, ``CBAFORRZ`` (``cbafori.F:358``)
  on the force side; with ``IDRIL > 0`` the ``CBAENER``/``CBAFORCT``/
  ``CBAENERS`` constant-shear treatment is SKIPPED (``cbaforc3.F:718,
  855, 989``); drill moments leave through ``CBAPROJ`` (``VQ(:,3)*VMZ``).
* QEPH (``ISROT = IPARG(41)``, ``czforc3.F:300``): ``CZDERIRZ``
  (``czdef.F:299``) + ``CZDEFRZ`` (``czdef.F:212``) on the strain side
  (``RLZ`` = nodal spin about the normal over area, ``czcorc.F:518-534``),
  ``CZFINTCRZ`` (``czfintce.F:111``) on the force side, drill moments
  leave through ``CZPROJN`` (``czproj.F:139-154`` flat branch).
* BT (``coque``): there is NO drilling term. ``ISROT`` is read into an
  unused variable (``shell_internal_forces.F90:510``) and neither
  ``cforc3.F`` nor ``cfint3.F`` mentions ``ISROT``/``IDRIL``/``RLZ``/``VMZ``
  anywhere. A BT "drill moment" would be an invented formula, which
  ``AGENTS.md`` forbids — so the port surfaces ``isrot`` on BT props and
  the kernel ignores it, and the test below pins that.

Documented cut kept visible: the drill *hourglass* stabilization rows
(QEPH ``CZFINTNRZ``'s ``VHGZK``/``VHGZE`` modes, ``czfintn.F:565``) are
not ported — the constant drill stiffness is; the ``CZFINTN1``
stabilization keeps running unchanged.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, shell_qbat, shell_qeph
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.keywords import parse_starter_deck
from pyradioss.input.keywords.properties import resolve_isrot
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)

DT = 1.0e-3
SPIN = 0.01

NODES = ("1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n")
MAT = "/MAT/LAW1/1\nsteel\n7.8e-9\n210. 0.3\n"


def _deck(ishell, idrill_field, nip=3, extra="", pre_prop=""):
    flags = f"{ishell} 0 0 {idrill_field}" if idrill_field is not None \
        else f"{ishell} 0 0"
    return ("/BEGIN\np35 idrill\n/NODE\n" + NODES +
            "/SHELL/1\n1 1 2 3 4\n/PART/1\nplate\n1 1\n" + MAT + pre_prop +
            f"/PROP/SHELL/1\nplate prop\n{flags}\n0 0 0 0 0\n"
            f"{nip} 0 0.1\n" + extra + "/END\n")


def _build(tmp_path, name, **kw):
    f = tmp_path / f"{name}_0000.rad"
    f.write_text(_deck(**kw))
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model


def _forces(mod, group, spin=0.0):
    x = np.asarray(mod.x, dtype=float)
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    if spin:
        vr[int(group.conn[0, 0]), 2] = spin   # node-0 spin about the normal
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    if group is mod.shells:
        shell_bt4.forces(group, x, v, vr, DT, fint, mint)
    elif group is mod.shells_qbat:
        shell_qbat.forces(group, x, v, vr, DT, fint, mint)
    else:
        shell_qeph.forces(group, x, v, vr, DT, fint, mint)
    return fint, mint


# ---------------------------------------------------------------------------
# reader half (hm_read_prop01.F:145,214-216)
# ---------------------------------------------------------------------------

def test_idrill2_is_mapped_to_zero_isrot(tmp_path):
    """``hm_read_prop01.F:215`` — ``IF(ISROT==2) ISROT = 0``."""
    assert resolve_isrot(2) == 0
    m = _build(tmp_path, "R2", ishell=12, idrill_field=2)
    params = m.properties[1].params
    assert params["idrill"] == 2
    assert params["isrot"] == 0


def test_idrill1_passes_through_and_blank_defaults_to_zero(tmp_path):
    m = _build(tmp_path, "R1", ishell=24, idrill_field=1)
    assert m.properties[1].params["isrot"] == 1
    m0 = _build(tmp_path, "R0", ishell=0, idrill_field=0)
    assert m0.properties[1].params["isrot"] == 0
    mblank = _build(tmp_path, "RB", ishell=0, idrill_field=None)
    assert mblank.properties[1].params["isrot"] == 0


def test_idrill0_falls_back_to_def_shell_default(tmp_path):
    """``hm_read_prop01.F:214`` — ``IF(ISROT==0)ISROT=IDRIL_D``."""
    assert resolve_isrot(0, {"idrill": 1}) == 1
    m = _build(tmp_path, "RD", ishell=12, idrill_field=0,
               pre_prop="/DEF_SHELL\n0 0 0 0 0 0 1\n")
    assert m.properties[1].params["isrot"] == 1


# ---------------------------------------------------------------------------
# kernel half — QBAT / QEPH resist a node spin, BT ignores it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ishell,group_name", [(12, "shells_qbat"),
                                               (24, "shells_qeph")])
def test_drilling_stiffness_resists_a_node_spin(tmp_path, ishell,
                                                group_name):
    """``CBAFORRZ`` / ``CZFINTCRZ``: spinning node 0 about its normal
    (``KRZ = 1e-3 * G``, ``cncoef3.F:86-91``) produces a drill moment."""
    m = _build(tmp_path, f"S{ishell}", ishell=ishell, idrill_field=1)
    g = getattr(m, group_name)
    assert g is not None and g.n == 1
    f0, m0 = _forces(m, g)
    f1, m1 = _forces(m, g, spin=SPIN)
    dm = m1 - m0
    assert np.abs(dm).sum() > 0.0
    assert dm[int(g.conn[0, 0]), 2] != 0.0      # drill moment about normal
    assert np.abs(f1 - f0).sum() > 0.0          # + in-plane corrections


@pytest.mark.parametrize("ishell,group_name", [(12, "shells_qbat"),
                                               (24, "shells_qeph")])
def test_drill_response_is_linear_and_books_internal_energy(
        tmp_path, ishell, group_name):
    """One cycle from a zero drill state is linear in the spin, and the
    trapezoidal drill work lands in ``eint`` (``CBAFORRZ``/``CZFINTCRZ``
    book ``EINT`` on both half-steps)."""
    ma = _build(tmp_path, f"L{ishell}a", ishell=ishell, idrill_field=1)
    ga = getattr(ma, group_name)
    fa, ma1 = _forces(ma, ga, spin=SPIN)
    ea = float(np.asarray(ga.state["eint"]).sum())

    mb = _build(tmp_path, f"L{ishell}b", ishell=ishell, idrill_field=1)
    gb = getattr(mb, group_name)
    fb, mb1 = _forces(mb, gb, spin=2.0 * SPIN)
    eb = float(np.asarray(gb.state["eint"]).sum())

    # virgin drill state both times: response exactly linear in the spin
    np.testing.assert_allclose(mb1, 2.0 * ma1, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(fb, 2.0 * fa, rtol=1e-12, atol=0.0)
    assert eb == pytest.approx(4.0 * ea)     # trapezoidal work is quadratic
    assert ea > 0.0


@pytest.mark.parametrize("ishell,group_name", [(12, "shells_qbat"),
                                               (24, "shells_qeph")])
def test_idrill0_leaves_a_node_spin_answered(tmp_path, ishell, group_name):
    """Gating: with ``isrot == 0`` the drill path is off, so a pure
    normal spin on a flat element changes nothing — every existing
    (``Idrill == 0``) deck is untouched."""
    m = _build(tmp_path, f"Z{ishell}", ishell=ishell, idrill_field=0)
    g = getattr(m, group_name)
    assert g.state["slices"][0][2].params["isrot"] == 0
    f0, m0 = _forces(m, g)
    f1, m1 = _forces(m, g, spin=SPIN)
    assert np.array_equal(f1, f0) and np.array_equal(m1, m0)


@pytest.mark.parametrize("ishell,group_name", [(12, "shells_qbat"),
                                               (24, "shells_qeph")])
def test_rigid_rotation_grows_no_drill_stress(tmp_path, ishell, group_name):
    """Objectivity: a rigid rotation (``v = Om x r``, ``vr = Om``) is no
    drill deformation, so the drill path adds nothing over ``isrot=0`` —
    while a single-node spin does (contrast
    ``test_drilling_stiffness_resists_a_node_spin``)."""
    def _run(name, isrot, rigid):
        m = _build(tmp_path, name, ishell=ishell, idrill_field=isrot)
        g = getattr(m, group_name)
        x = np.asarray(m.x, dtype=float)
        v = np.zeros_like(x)
        vr = np.zeros_like(x)
        if rigid == "spin":
            vr[int(g.conn[0, 0]), 2] = SPIN
        elif rigid == "rigid":
            v[:, 0] = -SPIN * x[:, 1]
            v[:, 1] = SPIN * x[:, 0]
            vr[:, 2] = SPIN
        f = np.zeros_like(x)
        mm = np.zeros_like(x)
        if group_name == "shells_qbat":
            from pyradioss.elements import shell_qbat as mod
        else:
            from pyradioss.elements import shell_qeph as mod
        mod.forces(g, x, v, vr, DT, f, mm)
        return f, mm

    f_rigid1, m_rigid1 = _run(f"G{ishell}r1", 1, "rigid")
    f_rigid0, m_rigid0 = _run(f"G{ishell}r0", 0, "rigid")
    np.testing.assert_allclose(m_rigid1, m_rigid0, rtol=1e-9, atol=1e-12)
    f_spin1, m_spin1 = _run(f"G{ishell}s1", 1, "spin")
    assert np.abs(m_spin1 - m_rigid1).sum() > 0.0


def test_bt_ignores_isrot_like_upstream(tmp_path):
    """``coque/cforc3.F`` + ``cfint3.F`` carry no ``ISROT`` term
    (``shell_internal_forces.F90:510`` reads it into an unused
    variable), so a BT prop accepts ``isrot`` and the kernel ignores
    it — including a node spin."""
    m = _build(tmp_path, "BT", ishell=0, idrill_field=1)
    g = m.shells
    assert g is not None and g.n == 1
    assert g.state["slices"][0][2].params["isrot"] == 1
    f0, m0 = _forces(m, g)
    f1, m1 = _forces(m, g, spin=SPIN)
    assert np.array_equal(f1, f0) and np.array_equal(m1, m0)
