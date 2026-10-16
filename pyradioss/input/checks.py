"""
pyradioss.input.checks — input and starter sanity checks for materials, properties, and elements.
"""

from ..starter.checks import (
    check_mat_law106,
    check_mat_jcook_alm,
    check_mat_johns_cook_alm,
    check_mat_law104,
    check_mat_drucker,
    check_mat_johns_voce_drucker,
    check_mat_plas_druck,
    check_mat_law103,
    check_mat_hensel_spittel,
    check_mat_plas_hens,
    check_mat_law102,
    check_mat_dprag2,
    check_mat_law101,
    check_mat_law100,
    check_mat_law95,
    check_mat_law93,
    check_mat_law94,
    check_mat_law92,
    check_mat_law88,
    check_mat_law87,
    check_materials,
    check_model,
)
# The Isolid dispatch refusal table (P2.11) lives in idsolid.py; it is
# re-exported here because that is where a caller looks for a refusal gate.
from .idsolid import (UNSUPPORTED_ISOLID, unsupported_isolid,
                      unsupported_isolid_reason, warn_unsupported_def_solid_isolid)
# The P2.13 solid reconciliation registry (the `refused`/`ported` rows of the
# per-routine audit) lives in solid_routines.py; it is re-exported here for the
# same reason idsolid is: this is where a caller looks for a refusal gate.
from .solid_routines import (ported_solid_routines,
                             refused_solid_routine_reason,
                             refused_solid_routines)
# P4.8 spring gate: ladder in elements/spring.py (IGTYP_GROUPS), re-exported here.
from ..elements.spring import IGTYP_GROUPS

_RFORC3 = "engine/source/elements/spring/rforc3.F"
_EXTRA_REFUSAL = {
    19: "; R3TORS at :609 serves the TYPE12 branch's TANI output, it is not a dispatch target (TYPE19 torsion here is a port-side extension)",
    28: "; NSTRAND xelems run under engine/source/elements/xelem/xforc28.F, computed here by nstrand.py rather than the spring ladder",
    **{v: "; the Starter ships no such /PROP reader (hm_read_prop_generic.F:200-203 raises MSGID=1647)"
       for v in (7, 24, 38, 39, 40, 41, 42, 47, 48, 49)},
}


def unsupported_igtyp() -> dict[int, str]:
    """IGTYP values with no rforc3.F ladder branch, refused with an upstream citation."""
    return {v: f"{_RFORC3}:243-1489 has no IGTYP == {v} branch{_EXTRA_REFUSAL.get(v, '')}"
            for v in range(50) if v not in IGTYP_GROUPS}

check_all = check_model

__all__ = sorted(n for n in globals() if not n.startswith("_"))
