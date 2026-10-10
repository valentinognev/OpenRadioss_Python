"""
pyradioss.input.idsolid — the ``/PROP/SOLID`` ``Isolid`` refusal table.

The Starter's group loop (``engine/source/elements/forint.F``) sends every
element group to exactly one force routine, chosen from ``ISOLNOD`` and
``JHBE``. ``JHBE`` is the ``Isolid`` field of the ``/PROP/SOLID`` card.

This port models a subset of those formulations.
``pyradioss.elements.SOLID_ISOLID_GROUPS`` routes the ones it does; this
module is the complete, explicit answer for every value it does **not**: the
value, and the upstream file (with the line) that says what the Fortran
starter does with it. Nothing may be added to ``SOLID_ISOLID_GROUPS`` without
either appearing here or routing to a real kernel — an unrouted, unrefused
value is a silent skip, which is what turned 440 corpus decks into "SKIPS".

The starter's own normalization, for reference:

===============================  =========================================
``hm_read_prop06.F:255``         ``IF (IHBE == 0) IHBE = IHBE_DS``
``hm_read_prop06.F:257``         101 -> 1   (pre-5.1 format)
``hm_read_prop06.F:262``         102 -> 2
``hm_read_prop06.F:266``         104 -> 24
``hm_read_prop06.F:270``         112 -> 12
``hm_read_prop06.F:285``         anything outside {1,2,14,17,18,24} ->
                                 ``ANCMSG(533)``, ``IHBE = 1``
``hm_read_prop14.F:233``         5  -> 1    (/PROP/TYPE14, /PENTA6)
``hm_read_prop14.F:242``         19 -> 17   (/PROP/TYPE14, /PENTA6)
===============================  =========================================

Values 1001-1049 (``hm_read_prop15.F:194-197``, /PROP/TYPE15) are not
formulations either: they encode the number of integration points
(``NPTS = IHBE - 1000``) on the multilayer solid property, which this port
does not read, so they are refused there rather than here.
"""

from functools import lru_cache
from typing import Dict, FrozenSet, Optional

_PROP06 = "starter/source/properties/solid/hm_read_prop06.F"
_PROP14 = "starter/source/properties/solid/hm_read_prop14.F"
_PROP15 = "starter/source/properties/solid/hm_read_prop15.F"
_PROP20 = "starter/source/properties/thickshell/hm_read_prop20.F"
_PROP43 = "starter/source/properties/solid/hm_read_prop43.F"
_PROPS = "starter/source/properties/hm_read_properties.F"
_FORINT = "engine/source/elements/forint.F"


def _rejected(value: int) -> str:
    """The Starter itself throws this value away (``ANCMSG(533)``)."""
    return (f"Not an Isolid value: {_PROP06}:285 accepts only "
            f"{{1,2,14,17,18,24}} and resets {value} to 1 with ANCMSG(533); "
            f"{_PROP14}:252 does not list it either.")


def _alias(value: int, target: int) -> str:
    """A pre-5.1 spelling the Starter rewrites before dispatch."""
    return (f"Pre-5.1 alias: hm_read_prop06.F rewrites {value} to Isolid="
            f"{target} with JCVT=2. This port does not apply that rewrite, so "
            f"{value} must not reach a kernel.")


def _user_solid_type(value: int) -> str:
    """Not a formulation: a user-solid property *type*."""
    return (f"Not an Isolid value (hm_read_prop06.F:285 / hm_read_prop14.F:"
            f"252); {value} is a user-solid property type ({_PROPS}:585), "
            f"routed by forint.F:1105 to SUFORC3.")


#: Every ``Isolid`` value this port does not route to a dedicated kernel,
#: mapped to the upstream citation that says so.  0-49 is the whole
#: formulation-flag range the dispatch table has to answer for; 101, 102, 104,
#: 112 and 222 are the pre-5.1 spellings the Starter still reads.
UNSUPPORTED_ISOLID: Dict[int, str] = {
    0: ("Not a formulation: hm_read_prop06.F:255 `IF (IHBE == 0) IHBE = "
        "IHBE_DS` takes the /DEF_SOLID default (hm_read_defsolid.F:85). The "
        "port resolves that default while reading the property, so such parts "
        "stay in the residual 'bricks' group (solid_hexa8, sforc3.F) instead "
        "of being split into a dedicated group."),
    1: ("The default /BRICK formulation: hm_read_prop06.F:285 accepts it and "
        "forint.F:1156-1159 sends it to SFORC3, which is the residual 'bricks' "
        "group (solid_hexa8). Naming it in SOLID_ISOLID_GROUPS would make "
        "_dispatch_solid_formulations partition those elements *out* of "
        "'bricks' and nothing would iterate them."),
    3: _rejected(3),
    4: (f"Only /PROP/TYPE15 keeps it (hm_read_prop15.F:192 leaves 4 alone); "
        f"that multilayer solid property is not read by this port. Not a "
        f"/PROP/SOLID value ({_PROP06}:285)."),
    5: (f"Legacy alias: hm_read_prop14.F:233-234 rewrites 5 to 1 on "
        f"/PROP/TYPE14; the port does not apply that rewrite, so 5 must not "
        f"reach a kernel."),
    6: _rejected(6), 7: _rejected(7), 8: _rejected(8), 9: _rejected(9),
    10: _rejected(10), 11: _rejected(11),
    12: ("Accepted (legacy 8-point fully-integrated brick, hm_read_prop06.F:381 "
         "NPT=8, routed by forint.F:1220 to S8FORC3; 112 is its pre-5.1 "
         "alias, hm_read_prop06.F:270). This port reaches that same S8FORC3 "
         "kernel through Isolid=2 only and splits no group for 12."),
    13: ("Accepted on /PROP/TYPE14 (hm_read_prop14.F:252) and given NPT=8 "
         "(hm_read_prop06.F:381), so forint.F:1220 routes it to S8FORC3. No "
         "dedicated port group is split for it."),
    18: ("Accepted (hm_read_prop06.F:286, warned about by ANCMSG(1160)); it "
         "only sets IINT=2 (hm_read_prop06.F:250). forint.F:877 keys the EAS "
         "branch on JHBE==17 alone, so 18 reaches no branch of its own. The "
         "port routes 17 to 'bricks_eas' (solid_hexa8_eas, s8eforc3.F) and "
         "refuses 18 rather than guessing the IINT=2 variant."),
    19: (f"Legacy alias: hm_read_prop14.F:242-243 rewrites 19 to 17 on "
         f"/PROP/TYPE14; the port does not apply that rewrite, so 19 must "
         f"not reach a kernel."),
    20: (f"20 is a property *type* (/PROP/TYPE20 = TSHELL, {_PROPS}:485-491), "
         f"not an Isolid value: that reader's ISOLID card accepts only "
         f"14/15/16 ({_PROP20}:155)."),
    22: ("The 2D /QUAD enhanced-assumed-strain formulation: "
         "hm_read_prop06.F:385-388 forces NPT=4 and forint.F:1269 routes "
         "JHBE==22 to Q4FORC2. This port routes that path through Iquad=2 "
         "(QUAD_IQUAD_GROUPS -> 'quads_full', q4forc2.F) and splits no group "
         "for Isolid=22."),
    23: ("Not an Isolid value (hm_read_prop06.F:285 / hm_read_prop14.F:252); "
         "23 is the shell-element type tag in the starter's "
         "`elbuf_init/elbuf_ini.F`:1622, a different flag entirely."),
    25: _rejected(25), 26: _rejected(26), 27: _rejected(27),
    28: _rejected(28),
    29: (f"Not an Isolid value: {_PROP06}:285 accepts only {{1,2,14,17,18,24}}. "
         f"29+ are user-solid property types ({_PROPS}:585, routed by "
         f"forint.F:1105 to SUFORC3), not formulations."),
    30: _user_solid_type(30), 31: _user_solid_type(31),
    32: _user_solid_type(32),
    33: _rejected(33),
    34: (f"Not an Isolid value (hm_read_prop06.F:285 / hm_read_prop14.F:252); "
         f"34 is the /SPH property type ({_PROPS}), a different element "
         f"family."),
    35: _rejected(35), 36: _rejected(36), 37: _rejected(37),
    38: _rejected(38), 39: _rejected(39), 40: _rejected(40),
    41: _rejected(41), 42: _rejected(42),
    43: (f"43 is a property *type*, not an Isolid value: /PROP/TYPE43 (SOLID "
         f"SPOTWELD) has no ISOLID card at all — {_PROP43} reads only Ismstr "
         f"and THICK — so no Isolid=43 can reach the dispatch. The entry that "
         f"used to route it to 'solid_connect' was a silent skip: "
         f"Model.element_groups() never yields that group, so every element "
         f"it took out of 'bricks' was never integrated."),
    44: _rejected(44), 45: _rejected(45), 46: _rejected(46),
    47: _rejected(47), 48: _rejected(48), 49: _rejected(49),
    # Pre-5.1 spellings the Starter still reads and remaps before dispatch.
    101: _alias(101, 1), 102: _alias(102, 2),
    104: _alias(104, 24), 112: _alias(112, 12),
    222: (f"Thickness-coded thick shell accepted on /PROP/TYPE14 "
          f"(hm_read_prop14.F:252); {_FORINT}:1026 routes JHBE==222 to the "
          f"same S8ZFORC3 as JHBE==14, with NPTS decoded from the value "
          f"(hm_read_prop15.F:197). No dedicated port group is split for it."),
}


def unsupported_isolid() -> FrozenSet[int]:
    """The ``Isolid`` values this port does not route to a dedicated kernel.

    The complement of ``pyradioss.elements.SOLID_ISOLID_GROUPS`` over the
    formulation-flag range, plus the legacy aliases the Starter remaps before
    dispatch. Every member carries an upstream citation in
    :data:`UNSUPPORTED_ISOLID`; see :func:`unsupported_isolid_reason`.
    """
    return frozenset(UNSUPPORTED_ISOLID)


@lru_cache(maxsize=1)
def solid_group_fragments() -> FrozenSet[str]:
    """Name fragments matching every solid group, derived from the formulation
    tables: ``/DT/SOLID`` and ``/DEL/SOLID`` walk group *names*, so a group
    added to a table answers to them without a second hand-kept list. Cached —
    the tables are import-time constants of ``elements``, which imports this
    module's package, so the derivation cannot run at import time."""
    from ..elements import (PENTA_ISOLID_GROUPS, QUAD_IQUAD_GROUPS,
                            SOLID_ISOLID_GROUPS)
    return frozenset(
        n.split("_", 1)[0]
        for table in (SOLID_ISOLID_GROUPS, PENTA_ISOLID_GROUPS,
                      QUAD_IQUAD_GROUPS)
        for n in table.values()) | {"bricks", "quad", "tria"}


def unsupported_isolid_reason(isolid: int) -> Optional[str]:
    """Why ``Isolid=``*isolid* is refused, naming the upstream Fortran — or
    ``None`` when it is refused without a recorded reason (a bug)."""
    return UNSUPPORTED_ISOLID.get(isolid)


def warn_unsupported_def_solid_isolid(isolid: int, log) -> None:
    """Report an ``/DEF_SOLID`` Isolid this port does not model.

    ``hm_read_defsolid.F:85`` stores the card verbatim and every property
    reader substitutes it for its own Isolid when the card is silent
    (``hm_read_prop06.F:255``), so a bad default propagates to *every* property
    that omits the field. Saying so where the default is read is the only
    place it can still be traced.
    """
    reason = unsupported_isolid_reason(isolid)
    if reason:
        log.warning(f" /DEF_SOLID ISOLID={isolid} IS NOT ROUTED TO A DEDICATED "
                    f"FORMULATION KERNEL BY THIS PORT -- {reason}")
