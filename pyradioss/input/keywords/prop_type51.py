# -*- coding: utf-8 -*-
"""
/PROP/TYPE51 (/PROP/SH_COH) reader — split per the P1.7 budget.

Moved verbatim from ``keywords/properties.py`` alongside ``resolve_isrot``,
the engine-``ISROT`` rule shared by every shell-prop reader
(``hm_read_prop01.F:214-215``); ``properties.py`` re-imports both names so
every existing ``from .properties import ...`` keeps working.
"""

from __future__ import annotations

from ..deck_reader import KeywordBlock
from ...common.messages import MessageLog
from ...model.model import Model
from .misc import _fixed_data, _title_and_data
from ..prop_rivet import _safe_float, _safe_int


def resolve_isrot(idrill_raw, def_shell=None):
    """Resolve the card ``Idrill`` field to engine ``ISROT``.

    ``starter/source/properties/shell/hm_read_prop01.F:145`` reads the
    card field into ``ISROT``; ``:214`` ``IF(ISROT==0)ISROT=IDRIL_D``
    (the ``/DEF_SHELL`` default from ``hm_read_defshell.F``);
    ``:215`` ``IF(ISROT==2) ISROT = 0``; ``:216`` stores ``IGEO(20)``.
    The kernels gate their drilling branches on the resolved value.
    """
    try:
        isrot = int(idrill_raw or 0)
    except (TypeError, ValueError):
        isrot = 0
    if isrot == 0:
        try:
            isrot = int((def_shell or {}).get("idrill", 0) or 0)
        except (TypeError, ValueError):
            isrot = 0
    if isrot == 2:
        isrot = 0
    return isrot


def read_prop_type51(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE51`` or ``/PROP/SH_COH`` (M188): Cohesive shell / composite stack property."""
    from ...model.entities import PropType51, Property
    prop_id = block.user_id or 0
    title, cards = _fixed_data(block) if block.fixed else _title_and_data(block)
    valid_cards = [c for c in cards if not c.is_blank and not c.raw.strip().startswith("#")]
    if not valid_cards:
        log.error(f"/PROP/TYPE51/{prop_id}: missing data card", block.source)
        return

    ishell, ismstr, ish3n, idrill = 0, 0, 0, 0
    pthk, zshift = 0.0, 0.0
    hm, hf, hr, dm, dn = 0.0, 0.0, 0.0, 0.0, 0.0
    ashear = 0.833333
    iint, ithick = 0, 0
    failexp = 0.0
    p_thick_fail = 0.0
    vx, vy, vz = 1.0, 0.0, 0.0
    idsk, iorth, ipos, irp = 0, 0, 0, 0

    if block.fixed:
        if len(valid_cards) > 0:
            c0 = valid_cards[0].cut("PROP_TYPE51_1")
            ishell = _safe_int(c0[0]) if len(c0) > 0 else 0
            ismstr = _safe_int(c0[1]) if len(c0) > 1 else 0
            ish3n = _safe_int(c0[2]) if len(c0) > 2 else 0
            idrill = _safe_int(c0[3]) if len(c0) > 3 else 0
            pthk = _safe_float(c0[4]) if len(c0) > 4 else 0.0
            zshift = _safe_float(c0[5]) if len(c0) > 5 else (_safe_float(c0[4]) if len(c0) > 4 else 0.0)
        if len(valid_cards) > 1:
            c1 = valid_cards[1].cut("PROP_TYPE51_2")
            hm = _safe_float(c1[0]) if len(c1) > 0 else 0.0
            hf = _safe_float(c1[1]) if len(c1) > 1 else 0.0
            hr = _safe_float(c1[2]) if len(c1) > 2 else 0.0
            dm = _safe_float(c1[3]) if len(c1) > 3 else 0.0
            dn = _safe_float(c1[4]) if len(c1) > 4 else 0.0
        if len(valid_cards) > 2:
            c2 = valid_cards[2].cut("PROP_TYPE51_3")
            ashear = _safe_float(c2[0]) if len(c2) > 0 and c2[0].strip() else 0.833333
            iint = _safe_int(c2[1]) if len(c2) > 1 else 0
            ithick = _safe_int(c2[2]) if len(c2) > 2 else 0
            failexp = _safe_float(c2[3]) if len(c2) > 3 else 0.0
        if len(valid_cards) > 3:
            c3 = valid_cards[3].cut("PROP_TYPE51_4")
            vx = _safe_float(c3[0]) if len(c3) > 0 and c3[0].strip() else 1.0
            vy = _safe_float(c3[1]) if len(c3) > 1 else 0.0
            vz = _safe_float(c3[2]) if len(c3) > 2 else 0.0
            idsk = _safe_int(c3[3]) if len(c3) > 3 else 0
            iorth = _safe_int(c3[4]) if len(c3) > 4 else 0
            ipos = _safe_int(c3[5]) if len(c3) > 5 else 0
            irp = _safe_int(c3[6]) if len(c3) > 6 else 0
    else:
        if len(valid_cards) > 0:
            t0 = valid_cards[0].tokens()
            ishell = _safe_int(t0[0]) if len(t0) > 0 else 0
            ismstr = _safe_int(t0[1]) if len(t0) > 1 else 0
            ish3n = _safe_int(t0[2]) if len(t0) > 2 else 0
            idrill = _safe_int(t0[3]) if len(t0) > 3 else 0
            if len(t0) > 5:
                pthk = _safe_float(t0[4])
                zshift = _safe_float(t0[5])
            elif len(t0) > 4:
                zshift = _safe_float(t0[4])
                pthk = zshift
        if len(valid_cards) > 1:
            t1 = valid_cards[1].tokens()
            hm = _safe_float(t1[0]) if len(t1) > 0 else 0.0
            hf = _safe_float(t1[1]) if len(t1) > 1 else 0.0
            hr = _safe_float(t1[2]) if len(t1) > 2 else 0.0
            dm = _safe_float(t1[3]) if len(t1) > 3 else 0.0
            dn = _safe_float(t1[4]) if len(t1) > 4 else 0.0
        if len(valid_cards) > 2:
            t2 = valid_cards[2].tokens()
            if len(t2) >= 4:
                ashear = _safe_float(t2[0])
                iint = _safe_int(t2[1])
                ithick = _safe_int(t2[2])
                failexp = _safe_float(t2[3])
            elif len(t2) >= 3 and _safe_float(t2[1]) < 1.0 and _safe_int(t2[0]) > 0:
                iint = _safe_int(t2[0])
                ashear = _safe_float(t2[1])
                ithick = _safe_int(t2[2])
            else:
                ashear = _safe_float(t2[0]) if len(t2) > 0 else 0.833333
                iint = _safe_int(t2[1]) if len(t2) > 1 else 0
                ithick = _safe_int(t2[2]) if len(t2) > 2 else 0
                failexp = _safe_float(t2[3]) if len(t2) > 3 else 0.0
        if len(valid_cards) > 3:
            t3 = valid_cards[3].tokens()
            vx = _safe_float(t3[0]) if len(t3) > 0 else 1.0
            vy = _safe_float(t3[1]) if len(t3) > 1 else 0.0
            vz = _safe_float(t3[2]) if len(t3) > 2 else 0.0
            idsk = _safe_int(t3[3]) if len(t3) > 3 else 0
            iorth = _safe_int(t3[4]) if len(t3) > 4 else 0
            ipos = _safe_int(t3[5]) if len(t3) > 5 else 0
            if len(t3) > 8:
                p_thick_fail = _safe_float(t3[6])
                failexp = _safe_float(t3[7])
                irp = _safe_int(t3[8])
            elif len(t3) > 6:
                irp = _safe_int(t3[6])

    p51 = PropType51(
        id=prop_id, ishell=ishell, ismstr=ismstr, ish3n=ish3n, idrill=idrill,
        pthk=pthk, zshift=zshift, hm=hm, hf=hf, hr=hr, dm=dm, dn=dn,
        ashear=ashear, iint=iint, ithick=ithick, failexp=failexp,
        vx=vx, vy=vy, vz=vz, idsk=idsk, iorth=iorth, ipos=ipos, irp=irp,
        title=title
    )
    model.prop_type51s[prop_id] = p51
    model.props_type51[prop_id] = p51
    model.properties[prop_id] = Property(
        id=prop_id, type=51, title=title,
        params={
            "ishell": ishell, "ismstr": ismstr, "ish3n": ish3n, "idrill": idrill,
            "isrot": resolve_isrot(idrill, getattr(model, "def_shell", None)),
            "pthk": pthk, "zshift": zshift, "z0": zshift,
            "hm": hm, "hf": hf, "hr": hr, "dm": dm, "dn": dn,
            "ashear": ashear, "iint": iint, "ithick": ithick,
            "failexp": failexp, "fexp": failexp, "p_thick_fail": p_thick_fail,
            "vx": vx, "vy": vy, "vz": vz, "idsk": idsk, "iorth": iorth, "ipos": ipos, "irp": irp
        }
    )




read_prop_p51 = read_prop_type51

