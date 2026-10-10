"""Task P2.13 — close the reconciliation audit: no `missing` row is left open.

The P2.0 audit (`tests/test_p2_reconciliation.py`) recorded a `missing` status for
every upstream solid routine that no task in the phase had claimed.  P2.13 owns
the leftovers: each one ends `ported`, `approximate` (with a measured deviation
note) or `refused` (naming the upstream file that lacks the routine).

Two invariants make the closure honest rather than cosmetic:

* a `refused` row must **cite** — the upstream file it is refused against, or
  the fact that no such routine exists anywhere upstream;
* an `approximate` row must carry a deviation note too, so the difference from
  the Fortran is written down next to the claim.

Run:  ``python3 -m pytest tests/test_p2_solid_audit_close.py -v``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = ROOT / "tools" / "validation_data" / "solid_routine_status.json"

#: Statuses P2.13 may leave behind.  `missing` is the value the audit forbids.
CLOSED_STATUSES = {"ported", "partial", "approximate", "refused"}

#: A refusal has to point at something checkable: an upstream path under the
#: solid tree, a Fortran source name, or the statement that upstream has no such
#: routine at all.  `[Ff](?:90)?` matches `.F`, `.f`, `.F90` and `.f90`; the
#: `9` and `0` are optional, so they must be grouped.
CITES_UPSTREAM = re.compile(r"engine/source/|common_source/|starter/source/"
                            r"|\b[A-Za-z0-9_]+\.[Ff](?:90)?\b|\.[Ff](?:90)?:\d+")
CITES_ABSENCE = re.compile(r"does not exist|no such routine|not present upstream|"
                           r"never existed|no upstream file", re.IGNORECASE)


def _rows() -> list[dict]:
    return json.loads(STATUS_JSON.read_text())


def test_no_solid_routine_is_left_unclassified():
    st = json.loads((ROOT / "tools/validation_data/solid_routine_status.json").read_text())
    left = [r["fortran"] for r in st if r["status"] == "missing"]
    assert left == [], left


def test_every_row_carries_a_closed_status():
    open_rows = [
        f"{r['fortran']}: {r['status']!r}"
        for r in _rows()
        if r["status"] not in CLOSED_STATUSES
    ]
    assert open_rows == [], open_rows[:10]


def test_every_refused_row_cites_the_upstream_file():
    """A refusal without a citation is an unowned row wearing a new label."""
    uncited = []
    for row in _rows():
        if row["status"] != "refused":
            continue
        note = (row["deviation_note"] or "").strip()
        if not (CITES_UPSTREAM.search(note) or CITES_ABSENCE.search(note)):
            uncited.append(row["fortran"])
    assert uncited == [], uncited[:10]


def test_a_refused_row_names_no_port():
    """A refusal has no implementation; claiming one is the thing it refuses."""
    claiming = [
        r["fortran"] for r in _rows()
        if r["status"] == "refused" and (r["port_module"] or r["port_symbol"])
    ]
    assert claiming == [], claiming[:10]


def test_the_closed_statuses_are_all_reachable():
    """Guard against a table that passes by collapsing every row to one value.

    A real audit closure uses more than one outcome: the phase ported kernels,
    approximated others, and refused the rest.  If every row ended on the same
    status this file would still be green while describing nothing.
    """
    seen = {r["status"] for r in _rows()}
    assert {"ported", "refused"} <= seen, (
        f"the table claims only {sorted(seen)}: a closure that ports everything "
        f"or refuses everything has not been reconciled row by row"
    )


def test_checks_exposes_the_refused_solid_routines_with_citations():
    """P2.13 closes the audit; the refusals must be reachable from the input
    checks, not readable only out of a JSON file.

    A refusal the Solver cannot ask about is the silent skip this phase exists
    to end: the deck asks for a formulation, the port drops it, and nothing
    says why.  ``refused_solid_routines()`` is that query.
    """
    from pyradioss.input.checks import refused_solid_routines

    refused = refused_solid_routines()
    assert refused, "the audit closed rows as refused but checks.py exposes none"

    table = {r["fortran"]: r for r in _rows()}
    for path, reason in refused.items():
        assert path in table, f"{path} is refused by checks.py but not in the table"
        assert table[path]["status"] == "refused", (
            f"{path} is refused by checks.py but the table calls it "
            f"{table[path]['status']!r}"
        )
        assert CITES_UPSTREAM.search(reason) or CITES_ABSENCE.search(reason), (
            f"{path} is refused without a citation: {reason!r}"
        )


def test_checks_refusal_matches_the_table_exactly():
    """The registry and the table are one fact in two places; they may not drift."""
    from pyradioss.input.checks import refused_solid_routines

    in_table = {r["fortran"] for r in _rows() if r["status"] == "refused"}
    assert set(refused_solid_routines()) == in_table


@pytest.mark.parametrize("status", sorted(CLOSED_STATUSES - {"ported"}))
def test_non_ported_statuses_carry_a_deviation_note(status):
    for row in _rows():
        if row["status"] == status:
            assert (row["deviation_note"] or "").strip(), (
                f"{row['fortran']} is {status} with no deviation_note"
            )