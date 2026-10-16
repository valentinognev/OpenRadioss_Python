"""Task P4.8 — the spring ``IGTYP`` dispatch is exhaustive and loud.

WHAT THIS FILE IS FOR
---------------------
``engine/source/elements/spring/rforc3.F`` dispatches on
``IGTYP = IGEO(11,I0)`` through one ``IF``/``ELSEIF`` ladder (:243 TYPE4,
:321 TYPE26, :387 TYPE27, :452 TYPE8, :558 TYPE12, :626 TYPE13, :749
TYPE25, :862 TYPE29-31, :1061 TYPE32, :1160 TYPE33/45, :1185 TYPE35/36,
:1293 TYPE44, :1391 TYPE46; ``forint.F:1451`` sends TYPE23 to
``R23FORC3`` before ``RFORC3`` is even called).  The ladder has no
terminal ``ELSE`` — an ``IGTYP`` with no branch silently computes zero
force there — and the Starter only errors on a property ``TYPE`` it
cannot read at all (``hm_read_prop_generic.F:200-203``, ``MSGID=1647``).

So the port transcribes the ladder twice, in complementary maps that
mirror the Phase-2 ``Isolid`` gate (``tests/test_p2_solid_isolid.py``):

* ``pyradioss.elements.IGTYP_GROUPS`` — every ladder branch, ``IGTYP``
  by ``IGTYP``, to the kernel module that owns it;
* ``pyradioss.input.checks.unsupported_igtyp()`` — every other value
  in ``0..49``, refused with the upstream spring file that lacks the
  branch.

RULES THIS FILE ENFORCES
------------------------
* the union of the two maps covers ``0..49`` with no silent value;
* every refusal citation names a file under
  ``engine/source/elements/spring/``;
* the routed set is EXACTLY the ladder — no invented ``IGTYP`` (there
  is no ``IGTYP == 6`` or ``IGTYP == 19`` branch upstream: ``R6DEF3``
  serves TYPE25 from ``rforc3.F:785`` and ``R3TORS`` is the TYPE12
  branch's ``TANI`` output helper at ``:609``), and no ladder branch
  left out;
* no value is both routed and refused;
* every routed group names a kernel module this repository imports
  (``pyradioss.elements.<group>`` or, for the joint branch driven by
  ``engine/kjoint.py``, ``pyradioss.engine.<group>``).
"""

from __future__ import annotations

import importlib

#: The ``rforc3.F`` ladder branches plus the ``forint.F`` TYPE23 driver:
#: the only IGTYP values with an upstream spring computation.
LADDER_IGTYPS = frozenset({
    4, 8, 12, 13, 23, 25, 26, 27,
    29, 30, 31, 32, 33, 35, 36, 44, 45, 46,
})


def test_igtyp_is_exhaustive():
    from pyradioss.elements import IGTYP_GROUPS
    from pyradioss.input import checks
    known = set(IGTYP_GROUPS) | set(checks.unsupported_igtyp())
    for igtyp in range(0, 50):
        assert igtyp in known, f"IGTYP {igtyp} silently unhandled"


def test_unsupported_igtyp_cites_upstream():
    from pyradioss.input import checks
    for igtyp, cite in checks.unsupported_igtyp().items():
        assert cite.startswith("engine/source/elements/spring/")


def test_routed_set_is_exactly_the_ladder():
    """No invented IGTYP, no missing branch: the routed keys ARE the
    ``rforc3.F`` ELSEIF targets plus forint's TYPE23."""
    from pyradioss.elements import IGTYP_GROUPS
    assert set(IGTYP_GROUPS) == set(LADDER_IGTYPS)


def test_no_igtyp_is_both_routed_and_refused():
    from pyradioss.elements import IGTYP_GROUPS
    from pyradioss.input import checks
    overlap = set(IGTYP_GROUPS) & set(checks.unsupported_igtyp())
    assert not overlap, f"IGTYP values both routed and refused: {sorted(overlap)}"


def test_routed_groups_name_real_kernels():
    """Every routing target resolves to a module this repository imports —
    a dispatch entry pointing at a module that does not exist is a
    silent skip wearing a routing table."""
    from pyradioss.elements import IGTYP_GROUPS
    for igtyp, group in IGTYP_GROUPS.items():
        for pkg in ("pyradioss.elements", "pyradioss.engine"):
            try:
                importlib.import_module(f"{pkg}.{group}")
                break
            except ImportError:
                continue
        else:  # pragma: no cover - the failure is the point
            raise AssertionError(
                f"IGTYP {igtyp} routes to group {group!r}, which is neither "
                "a pyradioss.elements nor a pyradioss.engine module"
            )
