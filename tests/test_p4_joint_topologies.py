"""Task P4.6 — ``/PROP/TYPE33`` topology matrix and the nine joint readers.

Fortran origin (starter side)::

  starter/source/properties/spring/hm_read_prop33.F          (ITYP dispatch 1..9)
  starter/source/properties/spring/hm_read_prop33_sph_jnt.F  (ITYP 1, spherical)
  starter/source/properties/spring/hm_read_prop33_rev_jnt.F  (ITYP 2, revolute)
  starter/source/properties/spring/hm_read_prop33_cyl_jnt.F  (ITYP 3, cylindrical)
  starter/source/properties/spring/hm_read_prop33_plan_jnt.F (ITYP 4, planar)
  starter/source/properties/spring/hm_read_prop33_univ_jnt.F (ITYP 5, universal)
  starter/source/properties/spring/hm_read_prop33_trans_jnt.F (ITYP 6, translation)
  starter/source/properties/spring/hm_read_prop33_old_jnt.F  (ITYP 7, oldham)
  starter/source/properties/spring/hm_read_prop33_fix_jnt.F  (ITYP 8, fixed)
  starter/source/properties/spring/hm_read_prop33_free_jnt.F (ITYP 9, free)

Fortran origin (element side)::

  engine/source/elements/joint/ruser33.F     (RUSER33 + DEF_FDOF free-DOF table)
  engine/source/elements/joint/rgjoint.F     (RGJOINT driver: RSKEW33/RUSER33/
                                             RDTIME33/RBILAN33/RANIM33/RCUM33)
  engine/source/elements/joint/joint_block_stiffness.F

Acceptance (plan P4.6): ``joint_topologies.TOPOLOGIES`` carries one entry per
joint kind with ``dofs``, ``free_dofs`` and ``ncmax``, and every topology is
either routed to an engine kernel or refused with its reader cited.

IMPORTANT — plan-text discrepancy, resolved in favour of the Fortran sources
(``PORTING_GUIDE`` rule: the Fortran wins). The Step-1 sketch in
``plan/05_phase4_elements_1d.md`` P4.6 parametrizes ``("planar", 1)`` and
``("translation", 3)``. Every source of truth says the opposite:

* ``hm_read_prop33_plan_jnt.F`` stores free stiffnesses ``Kty, Ktz, Krx``
  (``SET_U_GEO(5,6,7)``) → Ty, Tz, Rx free → **3** DOFs;
* ``hm_read_prop33_trans_jnt.F`` stores only ``Ktx`` (``SET_U_GEO(4)``) → Tx
  free → **1** DOF;
* ``DEF_FDOF`` in ``ruser33.F``: ``JTYP==4 → FDOF(2,3,4)`` (3 free),
  ``JTYP==6 → FDOF(1)`` (1 free);
* the port's own ``engine/kjoint.py``: ``PLANAR → {1, 2, 3}``,
  ``SLIDER/PRISMATIC → {0}``.

The tests below therefore assert planar → 3 and translation → 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import joint_topologies
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def _joint_deck(pid: int, ityp: int, c3: str, c4: str = "",
                c5: str = "", c6: str = "", fixed: bool = False) -> str:
    """Free- (or fixed-) format /PROP/TYPE33 deck for one joint type.

    Card image (``hm_read_prop33.F`` reads ``type``/``SkewFlag`` first, then
    each sub-reader reads its subset in echo order ``Idsk1, Idsk2, Xk, Cr,
    Kn, K<free..>, Ifun<free..>[, C<free..>, IfunC<free..>]``):

      1: joint_type skew_flag
      2: idsk1 idsk2 xk cr
      3: kn K<free..>
      4: Ifun<free..>
      5: [C<free..>]          (optional — old-format decks omit it)
      6: [IfunC<free..>]      (optional — old-format decks omit it)
    """
    c1 = f"{ityp:10d}{0:10d}" if fixed else f"{ityp} 0"
    c2 = f"{5:10d}{6:10d}{2.0e6:20.6e}{0.1:20.6e}" if fixed else "5 6 2.0e6 0.1"
    cards = "\n".join(c for c in (c1, c2, c3, c4, c5, c6) if c != "")
    head = "/BEGIN\nP4.6 joint\n       2026         0\n" if fixed else "/BEGIN\nP4.6 joint\n"
    return f"{head}/PROP/TYPE33/{pid}\nJoint {ityp}\n{cards}\n/END\n"


# ---------------------------------------------------------------------------
# Topology matrix
# ---------------------------------------------------------------------------

def test_topologies_has_exactly_the_nine_kinds():
    assert set(joint_topologies.TOPOLOGIES) == {
        "fixed", "free", "planar", "revolute", "spherical",
        "cylindrical", "universal", "translation", "old",
    }


@pytest.mark.parametrize("kind,dofs", [
    ("fixed", 0), ("planar", 3), ("revolute", 1),
    ("spherical", 3), ("cylindrical", 2),
    ("universal", 2), ("translation", 1),
    ("free", 6), ("old", 2),
])
def test_joint_topology_frees_the_right_dofs(kind, dofs):
    t = joint_topologies.TOPOLOGIES[kind]
    assert len(t.free_dofs) == dofs
    assert t.dofs == dofs


@pytest.mark.parametrize("kind,free", [
    ("fixed", ()),
    ("revolute", (3,)),
    ("spherical", (3, 4, 5)),
    ("cylindrical", (0, 3)),
    ("planar", (1, 2, 3)),
    ("universal", (4, 5)),
    ("translation", (0,)),
    ("old", (1, 2)),
    ("free", (0, 1, 2, 3, 4, 5)),
])
def test_free_dof_sets_match_def_fdof(kind, free):
    """0-based (Tx,Ty,Tz,Rx,Ry,Rz); cf. ``DEF_FDOF`` 1-based ``FDOF(1..6)``."""
    assert tuple(joint_topologies.TOPOLOGIES[kind].free_dofs) == free


@pytest.mark.parametrize("kind,ncmax", [
    ("fixed", 6), ("planar", 3), ("revolute", 5),
    ("spherical", 3), ("cylindrical", 4),
    ("universal", 4), ("translation", 5),
    ("old", 4), ("free", 0),
])
def test_ncmax_counts_the_knn_blocked_slots(kind, ncmax):
    """``ncmax`` = KNN-blocked relative-DOF slots (``SET_U_GEO(4..9)``)."""
    t = joint_topologies.TOPOLOGIES[kind]
    assert t.ncmax == ncmax
    assert t.ncmax == 6 - t.dofs


@pytest.mark.parametrize("ityp,kind", [
    (1, "spherical"), (2, "revolute"), (3, "cylindrical"),
    (4, "planar"), (5, "universal"), (6, "translation"),
    (7, "old"), (8, "fixed"), (9, "free"),
])
def test_ityp_dispatch_matches_hm_read_prop33(ityp, kind):
    """Dispatch order of ``hm_read_prop33.F:103-129``."""
    assert joint_topologies.by_ityp(ityp).kind == kind


def test_unknown_ityp_is_refused_with_citation():
    with pytest.raises(KeyError):
        joint_topologies.by_ityp(10)


def test_every_topology_is_routed_or_refused():
    for kind in joint_topologies.TOPOLOGIES:
        assert kind in joint_topologies.routed() or \
               kind in joint_topologies.refused()


def test_refused_kinds_cite_the_upstream_gap():
    for kind, cite in joint_topologies.refused().items():
        assert cite.startswith("engine/source/elements/joint/")


# ---------------------------------------------------------------------------
# The nine readers
# ---------------------------------------------------------------------------

#: ITYP -> (card3 "kn Kfree..", card4 "Ifun..", subset attribute checks).
JOINT_CARDS = {
    1: ("1.0e7 1000.0 2000.0 3000.0", "11 12 13",
        {"kn": 1.0e7, "krx": 1000.0, "kry": 2000.0, "krz": 3000.0,
         "xr_fun": 11, "yr_fun": 12, "zr_fun": 13}),
    2: ("1.0e7 1000.0", "11", {"kn": 1.0e7, "krx": 1000.0, "xr_fun": 11}),
    3: ("1.0e7 4000.0 1000.0", "14 11",
        {"kn": 1.0e7, "ktx": 4000.0, "krx": 1000.0,
         "xt_fun": 14, "xr_fun": 11}),
    4: ("1.0e7 5000.0 6000.0 1000.0", "15 16 11",
        {"kn": 1.0e7, "kty": 5000.0, "ktz": 6000.0, "krx": 1000.0,
         "yt_fun": 15, "zt_fun": 16, "xr_fun": 11}),
    5: ("1.0e7 2000.0 3000.0", "12 13",
        {"kn": 1.0e7, "kry": 2000.0, "krz": 3000.0,
         "yr_fun": 12, "zr_fun": 13}),
    6: ("1.0e7 4000.0", "14", {"kn": 1.0e7, "ktx": 4000.0, "xt_fun": 14}),
    7: ("1.0e7 5000.0 6000.0", "15 16",
        {"kn": 1.0e7, "kty": 5000.0, "ktz": 6000.0,
         "yt_fun": 15, "zt_fun": 16}),
    8: ("1.0e7", "", {"kn": 1.0e7}),
    9: ("0.0 4000.0 5000.0 6000.0 1000.0 2000.0 3000.0",
        "14 15 16 11 12 13",
        {"ktx": 4000.0, "kty": 5000.0, "ktz": 6000.0,
         "krx": 1000.0, "kry": 2000.0, "krz": 3000.0,
         "xt_fun": 14, "yt_fun": 15, "zt_fun": 16,
         "xr_fun": 11, "yr_fun": 12, "zr_fun": 13}),
}


@pytest.mark.parametrize("ityp", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_all_nine_joint_types_parse(tmp_path, ityp):
    c3, c4, expect = JOINT_CARDS[ityp]
    model, log = _parse_starter(tmp_path, _joint_deck(100 + ityp, ityp, c3, c4))
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert (100 + ityp) in model.prop_type33s
    p = model.prop_type33s[100 + ityp]
    assert p.joint_type == ityp
    assert p.id_sk1 == 5 and p.id_sk2 == 6
    assert p.xk == pytest.approx(2.0e6)
    assert p.cr == pytest.approx(0.1)
    for attr, val in expect.items():
        assert getattr(p, attr) == pytest.approx(val), attr
    assert (100 + ityp) in model.properties
    assert model.properties[100 + ityp].type == 33


def test_spherical_fixed_format_keeps_the_m186_card_image(tmp_path):
    """The pre-P4.6 generic 6-card image is the spherical card image."""
    c1 = f"{1:10d}{1:10d}"
    c2 = f"{10:10d}{20:10d}{1.0e6:20.6e}{0.05:20.6e}"
    c3 = f"{1.0e7:20.6e}{5.0e4:20.6e}{6.0e4:20.6e}{7.0e4:20.6e}"
    c4 = f"{101:10d}{102:10d}{103:10d}"
    deck = ("/BEGIN\nP4.6 sph fixed\n       2026         0\n"
            "/PROP/TYPE33/33\nSpherical\n"
            f"{c1}\n{c2}\n{c3}\n{c4}\n/END\n")
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    p = model.prop_type33s[33]
    assert (p.kn, p.krx, p.kry, p.krz) == pytest.approx((1.0e7, 5.0e4, 6.0e4, 7.0e4))
    assert (p.xr_fun, p.yr_fun, p.zr_fun) == (101, 102, 103)


def test_free_joint_reads_no_blocking_stiffness(tmp_path):
    model, log = _parse_starter(
        tmp_path, _joint_deck(109, 9, *JOINT_CARDS[9][:2]))
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    p = model.prop_type33s[109]
    assert p.kn == pytest.approx(0.0)


def test_cr_defaults_to_fiveem2_when_zero(tmp_path):
    """``IF (CR==ZERO) CR = FIVEEM2`` (all nine readers)."""
    deck = _joint_deck(102, 2, "1.0e7 1000.0", "11").replace("2.0e6 0.1", "2.0e6 0.0")
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.prop_type33s[102].cr == pytest.approx(0.05)


def test_unknown_joint_type_is_an_error(tmp_path):
    """``hm_read_prop33.F`` ELSE branch → ANCMSG 409."""
    model, log = _parse_starter(tmp_path, _joint_deck(110, 10, "1.0e7", ""))
    assert len(log.errors) == 1
    assert 110 not in model.prop_type33s


@pytest.mark.parametrize("ityp", [1, 2, 3, 4, 5, 6, 7])
def test_zero_kn_is_an_error(tmp_path, ityp):
    """ANCMSG 387 — every reader that reads ``Kn`` (all but free)."""
    c3 = "0.0 " + JOINT_CARDS[ityp][0].split(" ", 1)[1]
    model, log = _parse_starter(tmp_path, _joint_deck(200 + ityp, ityp, c3, JOINT_CARDS[ityp][1]))
    assert len(log.errors) == 1


def test_free_joint_needs_no_kn(tmp_path):
    """``hm_read_prop33_free_jnt.F`` reads no ``Kn`` — no 387 error."""
    model, log = _parse_starter(
        tmp_path, _joint_deck(209, 9, "0.0 0.0 0.0 0.0 0.0 0.0 0.0", ""))
    assert len(log.errors) == 0, f"Errors: {log.errors}"


def test_zero_kn_is_an_error_fixed(tmp_path):
    c3 = f"{0.0:20.6e}"
    deck = ("/BEGIN\nP4.6 fix\n       2026         0\n/PROP/TYPE33/208\nFixed\n"
            f"{8:10d}{0:10d}\n{5:10d}{6:10d}{2.0e6:20.6e}{0.1:20.6e}\n{c3}\n/END\n")
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1


def test_missing_skew_is_an_error(tmp_path):
    """ANCMSG 386."""
    deck = ("/BEGIN\nP4.6 noskew\n/PROP/TYPE33/108\nRevolute\n"
            "2 0\n0 6 2.0e6 0.1\n1.0e7 1000.0\n11\n/END\n")
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1


def test_cr_out_of_range_is_an_error(tmp_path):
    """ANCMSG 388."""
    deck = _joint_deck(102, 2, "1.0e7 1000.0", "11").replace("2.0e6 0.1", "2.0e6 1.5")
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1


def test_stiffness_defaults_to_one_when_function_given(tmp_path):
    """``IF(KXX==ZERO.AND.IFUN_XX/=0) KXX = ONE`` (all readers)."""
    model, log = _parse_starter(tmp_path, _joint_deck(106, 6, "1.0e7 0.0", "14"))
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.prop_type33s[106].ktx == pytest.approx(1.0)


def test_damping_cards_are_optional(tmp_path):
    """Old-format decks omit the ``C``/``Cf`` cards (``OFLAG`` echo)."""
    model, log = _parse_starter(
        tmp_path, _joint_deck(104, 4, *JOINT_CARDS[4][:2]))
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    p = model.prop_type33s[104]
    assert (p.cty, p.ctz, p.crx) == pytest.approx((0.0, 0.0, 0.0))
    assert (p.cty_fun, p.ctz_fun, p.crx_fun) == (0, 0, 0)


def test_damping_cards_parse_when_present(tmp_path):
    model, log = _parse_starter(
        tmp_path, _joint_deck(104, 4, "1.0e7 5000.0 6000.0 1000.0",
                              "15 16 11", "50.0 60.0 70.0", "25 26 21"))
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    p = model.prop_type33s[104]
    assert (p.cty, p.ctz, p.crx) == pytest.approx((50.0, 60.0, 70.0))
    assert (p.cty_fun, p.ctz_fun, p.crx_fun) == (25, 26, 21)


def test_engine_param_keys_cover_free_and_blocked_dofs(tmp_path):
    """``KJointEngine.from_prop`` consumes these ``Property.params`` keys."""
    model, log = _parse_starter(
        tmp_path, _joint_deck(103, 3, *JOINT_CARDS[3][:2]))
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    params = model.properties[103].params
    assert params["joint_type"] == 3
    assert params["ktx"] == pytest.approx(4000.0)
    assert params["krx"] == pytest.approx(1000.0)
    assert params["xt_fun"] == 14 and params["xr_fun"] == 11
    assert params["kn"] == pytest.approx(1.0e7)
