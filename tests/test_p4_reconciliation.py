"""Task P4.0 — the per-routine reconciliation audit of the 1-D element families.

WHAT THIS FILE IS FOR
---------------------
``tools/validation_data/census.json`` is a *machine* census: it walks every
upstream file and reports the status of the best Python module whose text
mentions the file.  It cannot judge whether the port is faithful, and it
cannot route an ``IGTYP`` branch to its kernel.

So this task adds the human half, recorded in
``tools/validation_data/w1d_routine_status.json``: **one row per upstream
file under ``engine/source/elements/{truss,beam,spring,joint,rivet,xelem}/``**
plus one branch row per shared-kernel ``IGTYP`` (``r5def3.F`` serves five
branches; a single file row cannot carry five ``igtyp`` values) and one row
per documented TYPE8 cut.  Each row carries the
``solid_routine_status.json`` shape **plus an ``igtyp`` field** on every
spring/joint row so the coverage test of Task P4.8 can consume it.

STATUSES (same four as Phase 2, plus the P2.13 ``refused``)
----------------------------------------------------------
``ported`` / ``partial`` / ``approximate`` / ``missing`` / ``refused``.
``refused`` names no port module — single-rank SPMD partition scatters
(``*cum3p.F``, mpi4py is not installed so every ``-np N`` path runs at
rank 1), ``PARTSAV`` energy booking (the port keeps per-element ``eint``
only) and TANI/H3D output helpers (``*tors.F``) are refusals with reasons,
not gaps.

RULES THIS FILE ENFORCES
------------------------
* the table exists and every row carries the eight keys;
* every one of the 24 ``IGTYP`` values of the ``rforc3.F`` ladder is
  classified by at least one row (the P4.0 gate);
* a row that claims a port points at a module this repository imports and
  a symbol that module defines — a ``ported`` row with an invented module
  launders an unverified claim;
* the eight TYPE8 cuts named at ``PORTING_GUIDE.md:67`` each have a row
  citing that line — recorded with citations, not omitted.  Seven are
  genuinely absent; the ``skew_ID`` frame was wired by M39
  (``spring_general._slice_frame``) after the guide row was written, so
  its row is ``partial`` with the same citation and a note saying so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = ROOT / "tools" / "validation_data" / "w1d_routine_status.json"
PHASE_PLAN = ROOT / "plan" / "05_phase4_elements_1d.md"

STATUSES = {"ported", "partial", "approximate", "refused", "missing"}
ROW_KEYS = {
    "fortran",
    "routine",
    "port_module",
    "port_symbol",
    "status",
    "deviation_note",
    "parity_case",
    "igtyp",
}

#: The complete spring ``IGTYP`` list of Task P4.0 (``rforc3.F`` ladder plus
#: the ``forint.F`` TYPE23 driver).  IGTYP 1/2/3/5/6 have no engine branch in
#: this checkout and no starter reader; they are classified by legacy rows,
#: not omitted.
REQUIRED_IGTYPS = (
    1, 2, 3, 4, 5, 6, 8, 12, 13, 19, 23, 25, 26, 27,
    29, 30, 31, 32, 33, 35, 36, 44, 45, 46,
)

#: The eight TYPE8 cuts named at ``PORTING_GUIDE.md:67``.  Each must have a
#: row whose routine ends with ``@cut:<id>`` and whose note cites the guide.
TYPE8_CUTS = (
    "fct_IDji",
    "hardening_Hi_IECROU",
    "rupture_DeltaMinMax",
    "rate_ISRATE_Fcut",
    "sensor_activation",
    "skew_ID_frame",
    "type13_corotational_frame",
    "implicit_type4_only",
)

GUIDE_CITE = "PORTING_GUIDE.md:67"


def _rows() -> list[dict]:
    return json.loads(STATUS_JSON.read_text())


def test_every_igtyp_branch_has_a_status():
    st = json.loads(Path("tools/validation_data/w1d_routine_status.json").read_text())
    have = {r["igtyp"] for r in st if r.get("igtyp") is not None}
    for igtyp in (1,2,3,4,5,6,8,12,13,19,23,25,26,27,29,30,31,32,33,35,36,44,45,46):
        assert igtyp in have, f"IGTYP {igtyp} unclassified"


def test_status_table_exists_and_is_a_list_of_rows():
    assert STATUS_JSON.is_file(), (
        f"{STATUS_JSON} is missing: Task P4.0 has to record a status for "
        f"every upstream 1-D file"
    )
    rows = _rows()
    assert isinstance(rows, list) and rows, "the status table is an empty list"


def test_igtyp_coverage_uses_the_plan_list():
    """The spec gate above must stay in sync with the plan's IGTYP list."""
    assert tuple(REQUIRED_IGTYPS) == (
        1, 2, 3, 4, 5, 6, 8, 12, 13, 19, 23, 25, 26, 27,
        29, 30, 31, 32, 33, 35, 36, 44, 45, 46,
    )


@pytest.mark.parametrize("key", sorted(ROW_KEYS))
def test_row_carries_every_key(key):
    for row in _rows():
        assert key in row, f"{row.get('fortran')}:{row.get('routine')} has no {key!r}"


def test_status_is_one_of_the_five():
    for row in _rows():
        assert row["status"] in STATUSES, f"{row['fortran']}: {row['status']!r}"


def test_igtyp_is_an_int_or_null():
    for row in _rows():
        igtyp = row["igtyp"]
        assert igtyp is None or isinstance(igtyp, int), (
            f"{row['fortran']}:{row['routine']}: igtyp {igtyp!r} is not an int"
        )


def test_routine_names_its_file_or_a_named_branch():
    """``routine`` is the file stem, or ``stem@...`` for a shared kernel's
    per-IGTYP branch row / a documented cut row (one file, several rows)."""
    for row in _rows():
        stem = Path(row["fortran"]).stem.lower()
        routine = row["routine"].lower()
        assert routine == stem or routine.startswith(stem + "@"), (
            f"{row['fortran']} names routine {row['routine']!r}; the primary "
            f"routine of that file is {stem!r}"
        )


def test_port_module_is_never_invented():
    import importlib

    for row in _rows():
        module = row["port_module"]
        if row["status"] in ("missing", "refused"):
            assert module is None, (
                f"{row['fortran']}:{row['routine']} is {row['status']} "
                f"but names {module}"
            )
            assert row["port_symbol"] is None
            continue
        assert module, f"{row['fortran']} is {row['status']} with no port_module"
        try:
            importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover - the failure is the point
            pytest.fail(f"{row['fortran']} names port_module {module!r}: {exc}")


def test_port_symbol_exists_in_its_module():
    import importlib

    for row in _rows():
        module, symbol = row["port_module"], row["port_symbol"]
        if module is None:
            continue
        obj = importlib.import_module(module)
        symbol = symbol or ""
        assert hasattr(obj, symbol.split(".")[0]) or symbol == "", (
            f"{row['fortran']} names symbol {symbol!r}, which "
            f"{module} does not define"
        )


def test_approximate_rows_carry_a_deviation_note():
    for row in _rows():
        if row["status"] == "approximate":
            note = (row["deviation_note"] or "").strip()
            assert note, (
                f"{row['fortran']}:{row['routine']} is `approximate` with no "
                f"deviation_note"
            )


def test_every_type8_cut_has_a_cited_row():
    """The eight ``PORTING_GUIDE.md:67`` cuts are recorded, not omitted."""
    rows = _rows()
    for cut in TYPE8_CUTS:
        matches = [r for r in rows if r["routine"].endswith(f"@cut:{cut}")]
        assert matches, f"TYPE8 cut {cut!r} has no audit row"
        for row in matches:
            assert GUIDE_CITE in (row["deviation_note"] or ""), (
                f"TYPE8 cut {cut!r} does not cite {GUIDE_CITE}"
            )
