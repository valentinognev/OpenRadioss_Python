"""
The P2.13 reconciliation registry: every upstream solid routine this port does
not carry, and why.

:mod:`pyradioss.input.idsolid` refuses *formulation flags* — the ``Isolid=``
value a deck writes. This module refuses *routines* — the individual Fortran
files of ``engine/source/elements/solid`` that the phase's audit closed as
``refused`` rather than ported.

The distinction matters because the two failure modes are different. An
unrouted ``Isolid`` reaches a reader that must decide what to do with the part.
An unreached routine is usually reached only by a caller this port does not run
at all (a thickshell kernel, a visualization writer, an ALE advection step), so
the honest answer is not a warning at read time but a **queryable record**: a
maintainer asking "why does this port not have ``s8zkeb3``?" gets the upstream
file and the reason, from code, instead of from a JSON file they have to know to
open.

The table is read from ``tools/validation_data/solid_routine_status.json`` at
call time rather than copied here, so the registry cannot drift from the audit
that produced it — a second hand-kept list of 198 refusals would drift within a
week, and a drifted refusal list is worse than none.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

#: The audit table P2.0 produced and P2.13 closed.  Resolved from this file, not
#: from the working directory, so a caller in any process finds it.
_STATUS_JSON = (
    Path(__file__).resolve().parents[2]
    / "tools" / "validation_data" / "solid_routine_status.json"
)


@lru_cache(maxsize=1)
def refused_solid_routines() -> Dict[str, str]:
    """Every ``refused`` row of the solid reconciliation audit.

    Maps the upstream path (``engine/source/elements/solid/...``, relative to
    ``$OR_SRC``) to the deviation note explaining the refusal.  Each note cites
    the upstream file that leaves the routine unreached, or records that no such
    routine is reachable at all — a refusal without one of those two is the
    unowned row this audit exists to eliminate, and
    ``tests/test_p2_solid_audit_close.py`` fails if any is missing.

    Cached: the table is an import-time-sized file on disk, and the registry is
    queried per element group.  A test that edits the table must call
    :func:`refused_solid_routines.cache_clear`.
    """
    rows = json.loads(_STATUS_JSON.read_text())
    return {
        row["fortran"]: row["deviation_note"]
        for row in rows
        if row["status"] == "refused"
    }


@lru_cache(maxsize=1)
def ported_solid_routines() -> Dict[str, str]:
    """Every ``ported`` row: upstream path -> ``module.symbol`` that carries it.

    The inverse question to :func:`refused_solid_routines`, and the same
    table-derived construction.
    """
    rows = json.loads(_STATUS_JSON.read_text())
    return {
        row["fortran"]: f"{row['port_module']}.{row['port_symbol']}"
        for row in rows
        if row["status"] == "ported"
    }


def refused_solid_routine_reason(path: str) -> str:
    """Why *path* is refused — ``""`` when it is not in the audit at all.

    An unknown path is reported as unknown rather than raised: this is a lookup
    a caller uses while deciding what to warn about, and the answer for "this
    routine was never in scope" is legitimately "no reason recorded".
    """
    return refused_solid_routines().get(path, "")


__all__ = [
    "ported_solid_routines",
    "refused_solid_routine_reason",
    "refused_solid_routines",
]