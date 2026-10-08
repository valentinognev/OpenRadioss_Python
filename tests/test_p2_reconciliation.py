"""Task P2.0 — the per-routine reconciliation audit of the claimed solid kernels.

WHAT THIS FILE IS FOR
---------------------
``tools/validation_data/census.json`` is a *machine* census: it walks every
upstream file, parses the provenance block upstream writes into it, and reports
the status of the best Python module whose text mentions the file.  It is not a
judgement — as generated, **all 5,887 of its statuses read ``missing``**, including
kernels this port has carried for years (``solid/sforc3.F`` →
``pyradioss.elements.solid_hexa8.forces``).  A census that says "nothing is
ported" cannot be used to decide what Phase 2 still owes, and it cannot be used
to prove that anything *is* ported.

So this task adds the human half, recorded in
``tools/validation_data/solid_routine_status.json``: **one row per upstream file
under ``engine/source/elements/solid/``, 415 of them**, each carrying the routine
it implements, the Python module and symbol that implements it, and one of four
statuses.  Phase 2's later tasks and Phase 12/17 read that file, not the census.

FOUR STATUSES, AND THE RULE THAT DECIDES ``approximate``
-------------------------------------------------------
``ported``
    the Python module performs the routine's job with the same formulation.
``partial``
    it performs part of the job — a branch, a subset of the outputs — and the
    rest is absent.  A shell kernel that implements the membrane but not the
    drilling penalty is ``partial``, not ``ported``.
``approximate``
    **the same quantity by a different formulation.**  This is the word that
    gets abused, so the boundary is drawn here: a shorter derivation of the same
    formula is still ``ported`` (a hexahedron's Jacobian evaluated with one
    cofactor expansion instead of a determinant call is the same formulation);
    computing a stress from an isotropic modulus where upstream builds an
    orthotropic compliance is not — that is ``approximate`` and it needs a
    ``deviation_note`` naming the difference.
``missing``
    no module implements it.

RULES THIS FILE ENFORCES
------------------------
* every one of the 415 census files has exactly one row;
* a row's ``port_module`` names a module that exists in this repository and is
  not invented — a ``ported`` row with a module path that does not import is
  worse than a ``missing`` row, because it launders an unverified claim;
* every ``approximate`` row carries a ``deviation_note`` that is not empty;
* every ``missing`` row is **covered by a later task in
  ``plan/03_phase2_elements_solid.md``** — its path has to sit in a directory or
  file list that Task P2.1 … P2.13 names.  A gap with nobody dispatched against
  it is exactly what Phase 1's reconciliation work exists to surface, so the
  check is here rather than in prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = ROOT / "tools" / "validation_data" / "solid_routine_status.json"
CENSUS_JSON = ROOT / "tools" / "validation_data" / "census.json"
PHASE_PLAN = ROOT / "plan" / "03_phase2_elements_solid.md"

SOLID_PREFIX = "engine/source/elements/solid/"
STATUSES = {"ported", "partial", "approximate", "missing"}
ROW_KEYS = {
    "fortran",
    "routine",
    "port_module",
    "port_symbol",
    "status",
    "deviation_note",
    "parity_case",
}
#: Tasks that may close a ``missing`` row.  P2.0 is excluded: it is the audit
#: itself, and letting it cover its own findings would make the check vacuous.
COVERING_TASKS = {f"P2.{n}" for n in range(1, 14)}


def _rows() -> list[dict]:
    return json.loads(STATUS_JSON.read_text())


def _solid_census_paths() -> list[str]:
    files = json.loads(CENSUS_JSON.read_text())["files"]
    return sorted(p for p in files if p.startswith(SOLID_PREFIX))


def _task_sections() -> dict[str, str]:
    """Map ``P2.n`` to the text of its section in the phase file."""
    text = PHASE_PLAN.read_text()
    parts = re.split(r"(?m)^### Task ", text)
    sections: dict[str, str] = {}
    for part in parts[1:]:
        task_id = part.split(":", 1)[0].strip()
        # the section ends at the next `---` rule that closes the task block
        sections[task_id] = part.split("\n---", 1)[0]
    return sections


def _plan_coverage() -> tuple[set[str], set[str]]:
    """Directories and file names that Tasks P2.1..P2.13 name.

    A backticked token is read as a directory when it is a real directory of the
    upstream solid tree, and as a file when it carries a Fortran extension.  The
    plan writes both as inline code (`` `solidez` ``, `` `sfor_n2s4.F` ``), so
    no prose matching is involved.
    """
    census_dirs = {p[len(SOLID_PREFIX):].split("/", 1)[0] for p in _solid_census_paths()}
    dirs: set[str] = set()
    files: set[str] = set()
    for task_id, body in _task_sections().items():
        if task_id not in COVERING_TASKS:
            continue
        for token in re.findall(r"`([^`\n]+)`", body):
            token = token.strip()
            bare = token.rsplit("/", 1)[-1]
            if bare in census_dirs:
                dirs.add(bare)
            if re.fullmatch(r"[A-Za-z0-9_.\-]+\.(?:F|F90|f|f90)", bare):
                files.add(bare)
    return dirs, files


def _is_covered(path: str) -> bool:
    dirs, files = _plan_coverage()
    relative = path[len(SOLID_PREFIX):]
    return relative.split("/", 1)[0] in dirs or path.rsplit("/", 1)[-1] in files


def test_status_table_exists_and_is_a_list_of_rows():
    assert STATUS_JSON.is_file(), (
        f"{STATUS_JSON} is missing: Task P2.0 has to record a status for every "
        f"upstream solid file"
    )
    rows = _rows()
    assert isinstance(rows, list) and rows, "the status table is an empty list"


def test_every_upstream_solid_routine_has_a_status():
    rows = _rows()
    seen = [r["fortran"] for r in rows]
    missing = [p for p in _solid_census_paths() if p not in set(seen)]
    assert missing == [], f"{len(missing)} unclassified, e.g. {missing[:5]}"


def test_every_census_solid_file_has_exactly_one_row():
    counted = {p: 0 for p in _solid_census_paths()}
    for row in _rows():
        counted[row["fortran"]] = counted.get(row["fortran"], 0) + 1
    duplicated = sorted(p for p, n in counted.items() if n > 1)
    assert duplicated == [], f"duplicate rows for {duplicated[:5]}"
    unknown = sorted({r["fortran"] for r in _rows()} - set(counted))
    assert unknown == [], f"rows for files the census does not list: {unknown[:5]}"


@pytest.mark.parametrize("key", sorted(ROW_KEYS))
def test_row_carries_every_key(key):
    for row in _rows():
        assert key in row, f"{row.get('fortran')} has no {key!r}"


def test_status_is_one_of_the_four():
    for row in _rows():
        assert row["status"] in STATUSES, f"{row['fortran']}: {row['status']!r}"


def test_routine_name_matches_the_file():
    for row in _rows():
        stem = Path(row["fortran"]).stem.lower()
        assert row["routine"].lower() == stem, (
            f"{row['fortran']} names routine {row['routine']!r}; the primary "
            f"routine of that file is {stem!r}"
        )


def test_port_module_is_never_invented():
    """A `ported` row must point at a module this repository really has."""
    import importlib

    for row in _rows():
        module = row["port_module"]
        if row["status"] == "missing":
            assert module is None, f"{row['fortran']} is missing but names {module}"
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
                f"{row['fortran']} is `approximate` with no deviation_note: the "
                f"difference from the Fortran is the whole content of the status"
            )


def test_every_missing_row_is_dispatched_to_a_later_task():
    """No gap without an owner.

    A `missing` row is covered when its path sits in a directory or file that
    Task P2.1 … P2.13 names.  Anything else is a gap the phase file never
    dispatched: closing the audit (P2.13) can record a refusal, but it cannot
    port code that no task claims.
    """
    uncovered = [
        r["fortran"] for r in _rows()
        if r["status"] == "missing" and not _is_covered(r["fortran"])
    ]
    assert uncovered == [], (
        f"{len(uncovered)} missing routines are covered by no task in "
        f"{PHASE_PLAN.name}: e.g. {uncovered[:8]}"
    )


def test_plan_coverage_extraction_finds_the_named_families():
    """Guard against a coverage parser that silently matches nothing."""
    dirs, files = _plan_coverage()
    assert "solidez" in dirs, "P2.2 names the solidez family; the parser missed it"
    assert "solid_q1np" in dirs, "P2.9 names solid_q1np"
    assert "solide8s" in dirs, "P2.10 names solide8s"
    assert "upwind.f" in {f.lower() for f in files}, "P2.4 names upwind.F"