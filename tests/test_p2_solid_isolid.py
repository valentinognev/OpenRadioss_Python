"""
P2.11 — the ``/PROP/SOLID`` ``Isolid`` card is routed or refused, never skipped.

Fortran origins read before writing this file:

* ``starter/source/elements/reader/hm_read_solid.F`` — the solid element card
  reader (``/BRICK``, ``/TETRA4``, ``/PENTA6``, ``/PYRA5``).
* ``starter/source/properties/solid/hm_read_prop06.F`` — the ``ISOLID`` card
  itself: the accepted set (line 285), the pre-5.1 aliases 101/102/104/112
  (lines 257-273), the ``IHBE == 0 -> /DEF_SOLID`` substitution (line 255), and
  the ``ANCMSG(533)`` reset to Isolid=1 for anything else (lines 285-293).
* ``starter/source/properties/solid/hm_read_prop14.F`` — ``/PROP/TYPE14``
  (``/PENTA6``): accepted set at line 252, aliases 5 -> 1 (line 233) and
  19 -> 17 (line 242).
* ``starter/source/properties/thickshell/hm_read_prop20.F`` line 155,
  ``hm_read_prop21.F`` line 155, ``hm_read_prop22.F`` line 184 — the
  thick-shell property readers and their narrower accepted sets.
* ``engine/source/elements/forint.F`` — the group loop this port mirrors. The
  solid branch is a ``SELECT``-style chain keyed on ``ISOLNOD`` and ``JHBE``:
  ``S6ZFORC3`` at :720, ``S6CFORC3``/``SCFORC3`` at :773, ``S8SFORC3``/
  ``S8EFORC3`` at :877, ``S8CFORC3`` at :956, ``S8ZFORC3`` at :1026,
  ``SUFORC3`` at :1105, ``SZFORC3``/``SFORC3`` at :1124, ``S8FORC3`` at :1220,
  and ``Q4FORC2`` for the 2D quad at :1269.

The failure this task exists to close: an ``Isolid`` the port does not model
used to vanish without a word. A refusal with its upstream citation is the
only acceptable alternative.
"""


def test_every_idsolid_value_is_routed_or_refused():
    from pyradioss.elements import SOLID_ISOLID_GROUPS, KERNELS
    from pyradioss.input.checks import unsupported_isolid
    for idsolid in range(0, 50):
        if idsolid in SOLID_ISOLID_GROUPS:
            assert KERNELS[SOLID_ISOLID_GROUPS[idsolid]]
        else:
            assert idsolid in unsupported_isolid()


def test_every_refused_idsolid_cites_an_upstream_fortran_file():
    """A refusal the reader cannot act on is a silent skip wearing a hat."""
    import re

    from pyradioss.elements import SOLID_ISOLID_GROUPS
    from pyradioss.input.checks import unsupported_isolid, unsupported_isolid_reason
    for idsolid in sorted(unsupported_isolid()):
        reason = unsupported_isolid_reason(idsolid)
        assert reason, f"Isolid={idsolid} is refused without a reason"
        assert re.search(r"hm_read_[a-z0-9]+\.F|forint\.F|elbuf_ini\.F|"
                         r"hm_read_properties\.F|hm_read_defsolid\.F",
                         reason), (f"Isolid={idsolid} refusal does not name the "
                                  f"upstream Fortran file: {reason!r}")
    # A routed value must never also be listed as refused.
    assert not (set(SOLID_ISOLID_GROUPS) & set(unsupported_isolid()))


def test_every_routed_idsolid_group_reaches_the_engine():
    """``Model.element_groups()`` is the only list the Engine walks.

    A dispatch entry pointing at a group name that list never yields is a
    silent skip: the Starter partitions those elements out of ``bricks`` and
    the Engine then never integrates them.
    """
    from pyradioss.elements import SOLID_ISOLID_GROUPS
    from pyradioss.model.model import Model

    class _StubGroup:
        n = 1

    model = Model()
    for name in set(SOLID_ISOLID_GROUPS.values()):
        setattr(model, name, _StubGroup())
    reachable = {name for name, _ in model.element_groups()}
    for idsolid, gname in SOLID_ISOLID_GROUPS.items():
        assert gname in reachable, (
            f"Isolid={idsolid} routes to group {gname!r}, which "
            f"Model.element_groups() never yields")
