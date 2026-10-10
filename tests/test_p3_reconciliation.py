"""Task P3.0 — the per-routine reconciliation audit of the shell kernels.

WHAT THIS FILE IS FOR
---------------------
``plan/04_phase3_elements_shell.md`` §Task P3.0 asks for one row per upstream
shell routine across the seven directories in its §Scope table (124 files:
``shell/`` common, ``coque``, ``coqueba``, ``coquez``, ``sh3n/coque3n``,
``coquedk``, ``coquedk6``), in the same record shape as Phase 2's
``tools/validation_data/solid_routine_status.json``, plus a ``parity``
sub-record per family seeded from the measured M41 cases.

Two things this file enforces beyond "the file parses":

* **a claim that names a Python module is checked.**  A ``ported`` row whose
  ``port_module`` does not import is worse than a ``missing`` row, because it
  launders an unverified claim — the same rule P2.0 applies to the solid table;
* **every ``partial``/``approximate`` row carries a ``deviation_note``**, so a
  known-wrong result can never sit in the table unnamed.

THE DKT18 QUESTION
------------------
``VALIDATION.md`` §7 item 2 calls DKT18 "the only unported shell formulation in
the family" as of M41.  That is stale — ``pyradioss/elements/shell_dkt18.py``
plus ``pyradioss/accel/jit_kernels/shells_dkt18.py`` carry the whole explicit
cycle — and the residual gap is depth, not existence: c06 0.2813 and c07
0.2581 rel-RMS, **worst significant channel MOMX on both**, against a 0.05
tolerance.  The plan's Step 3 asks which routine explains a MOMX deviation:
a *force* that is wrong while the *strain* is right points at ``cdkfint3.F``;
both wrong points at ``cdkdefo3.F``/``cdkderi3.F``.

Those are **hypotheses, not conclusions** — the port has no per-routine oracle
channel that separates force from strain, so this audit records, for every
DKT18 routine, the hypothesis its role makes it a candidate for, and stops
there.  No shell physics is touched by this task.

WHY THE DKT18 ASSERTION IS SCOPED
---------------------------------
Two DKT18 routines are genuinely unported: ``cdkfint_reg.F`` and
``dtcdk_reg.F``, the non-local (regularised) damage force and the non-local
element time step.  Upstream guards both behind ``INLOC = IPARG(78) > 0``
(``cdkforc3.F:498``), so they do not run on c06/c07 and cannot explain a MOMX
deviation there.

Every row therefore carries a declared ``guarded_by_inloc`` flag, the DKT18
assertion is made over the ``INLOC=0`` explicit cycle, and
``test_dkt18_nonlocal_branch_is_still_recorded_as_missing`` keeps the excluded
rows pinned — still ``missing``, still noted, still owned by the queued
non-local shell force and timestep work.  The scope is a property the table
declares, not a filename pattern, so a newly missing explicit-cycle routine
cannot hide behind it.  The same seven routines are missing in every shell
family, and P2.0 already records the solid one
(``solid/solide/sfint_reg.F``) the same way.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = ROOT / "tools" / "validation_data" / "shell_routine_status.json"
CENSUS_JSON = ROOT / "tools" / "validation_data" / "census.json"
PARITY_M41_JSON = ROOT / "tools" / "validation_data" / "parity_m41.json"

STATUSES = {"ported", "partial", "approximate", "missing"}
ROW_KEYS = {
    "family",
    "fortran",
    "routine",
    "port_module",
    "port_symbol",
    "status",
    "deviation_note",
    "parity_case",
    "parity",
    "hypothesis",
    "guarded_by_inloc",
}

#: Upstream directory -> family slug, from ``plan/04_phase3_elements_shell.md``
#: §Scope.  ``common`` is the four files directly under ``elements/shell/``.
FAMILY_PREFIX = {
    "common": "engine/source/elements/shell/",
    "bt": "engine/source/elements/shell/coque/",
    "qbat": "engine/source/elements/shell/coqueba/",
    "qeph": "engine/source/elements/shell/coquez/",
    "tri3": "engine/source/elements/sh3n/coque3n/",
    "dkt18": "engine/source/elements/sh3n/coquedk/",
    "dkt6": "engine/source/elements/sh3n/coquedk6/",
}
#: File counts the plan's §Scope table publishes for those directories.
PLANNED_COUNTS = {
    "common": 4,
    "bt": 26,
    "qbat": 30,
    "qeph": 18,
    "tri3": 25,
    "dkt18": 11,
    "dkt6": 10,
}

#: The measured M41 shell-family cases, from ``parity_m41.json``
#: (``tolerance_rel_rms`` 0.05).
DKT18_CASES = {"c06": 0.2813, "c07": 0.2581}

#: The upstream non-local branch, as P3.0 found it: seven routines across the
#: seven directories, every one of them unported.  Pinned here so the scope
#: decision cannot quietly widen.
NONLOCAL_ROUTINES = {
    "engine/source/elements/shell/coque/cfint_reg.F",
    "engine/source/elements/shell/coqueba/cbafint_reg.F",
    "engine/source/elements/shell/coqueba/dtcba_reg.F",
    "engine/source/elements/sh3n/coque3n/c3fint_reg.F",
    "engine/source/elements/sh3n/coquedk/cdkfint_reg.F",
    "engine/source/elements/sh3n/coquedk/dtcdk_reg.F",
    "engine/source/elements/sh3n/coquedk6/cdk6fint_reg.F",
}
#: The dispatch firstmate named as the owner of that path.
NONLOCAL_OWNER = "non-local shell force and timestep work"


def _rows() -> list[dict]:
    return json.loads(STATUS_JSON.read_text())


def _census_shell_paths() -> set[str]:
    files = json.loads(CENSUS_JSON.read_text())["files"]
    return {p for p in files if p.startswith(tuple(FAMILY_PREFIX.values()))}


def _dkt18_rows() -> list[dict]:
    return [r for r in _rows() if r["family"] == "dkt18"]


def _dkt18_explicit_rows() -> list[dict]:
    """The INLOC=0 explicit cycle — everything cdkforc3.F runs on c06/c07."""
    return [r for r in _dkt18_rows() if not r["guarded_by_inloc"]]


# --------------------------------------------------------------------------
# Step 1 of the plan.  Scoped to the INLOC=0 cycle per the owner decision of
# 2026-10-10; the scope itself is pinned by the tests below.
# --------------------------------------------------------------------------


def test_dkt18_parity_gap_is_localised_not_global():
    import json
    from pathlib import Path
    st = json.loads(Path("tools/validation_data/shell_routine_status.json").read_text())
    dk = [r for r in st if r["family"] == "dkt18" and not r["guarded_by_inloc"]]
    missing = [r["fortran"] for r in dk if r["status"] == "missing"]
    assert missing == [], f"DKT18 has {len(missing)} unported routines: {missing}"


def test_dkt18_scope_excludes_only_the_nonlocal_branch():
    explicit = {r["fortran"] for r in _dkt18_explicit_rows()}
    excluded = {r["fortran"] for r in _dkt18_rows()} - explicit
    assert excluded == {
        "engine/source/elements/sh3n/coquedk/cdkfint_reg.F",
        "engine/source/elements/sh3n/coquedk/dtcdk_reg.F",
    }
    assert len(explicit) == 9, f"the DKT18 explicit cycle has {len(explicit)} routines"


def test_dkt18_nonlocal_branch_is_still_recorded_as_missing():
    """The scoped assertion must not delete the gap it steps around."""
    rows = [r for r in _dkt18_rows() if r["guarded_by_inloc"]]
    assert rows, "the non-local rows are gone from the table"
    for row in rows:
        assert row["status"] == "missing", (
            f"{row['fortran']} is {row['status']}: the non-local branch was "
            f"recorded as unported and is not reclassified by the scope decision"
        )
        assert NONLOCAL_OWNER in row["deviation_note"], (
            f"{row['fortran']} names no owner for the non-local path"
        )


# --------------------------------------------------------------------------
# The table is a per-routine audit, not a summary.
# --------------------------------------------------------------------------


def test_status_table_exists_and_is_a_list_of_rows():
    assert STATUS_JSON.is_file(), (
        f"{STATUS_JSON} is missing — Task P3.0's audit table has not been built"
    )
    rows = _rows()
    assert isinstance(rows, list) and rows, "the status table is an empty list"


@pytest.mark.parametrize("key", sorted(ROW_KEYS))
def test_row_carries_every_key(key):
    for row in _rows():
        assert key in row, f"{row.get('fortran')} has no {key!r}"


def test_status_is_one_of_the_four():
    for row in _rows():
        assert row["status"] in STATUSES, f"{row['fortran']}: {row['status']!r}"


def test_guarded_by_inloc_is_a_declared_boolean():
    for row in _rows():
        assert isinstance(row["guarded_by_inloc"], bool), (
            f"{row['fortran']}: guarded_by_inloc is "
            f"{row['guarded_by_inloc']!r}, not a bool"
        )


def test_the_inloc_flag_marks_exactly_the_known_nonlocal_routines():
    flagged = {r["fortran"] for r in _rows() if r["guarded_by_inloc"]}
    assert flagged == NONLOCAL_ROUTINES, (
        "the INLOC>0 branch moved: "
        f"added {sorted(flagged - NONLOCAL_ROUTINES)}, "
        f"dropped {sorted(NONLOCAL_ROUTINES - flagged)}"
    )


def test_every_inloc_guarded_row_is_missing_and_owned():
    for row in _rows():
        if row["guarded_by_inloc"]:
            assert row["status"] == "missing", f"{row['fortran']} is {row['status']}"
            assert NONLOCAL_OWNER in row["deviation_note"], (
                f"{row['fortran']} names no owner for the non-local path"
            )


@pytest.mark.parametrize("family", sorted(PLANNED_COUNTS))
def test_family_covers_its_upstream_directory_exactly_once(family):
    prefix = FAMILY_PREFIX[family]
    expected = {p for p in _census_shell_paths() if p.startswith(prefix)}
    if family == "common":
        # ``common`` is the files sitting directly in elements/shell/, not the
        # family sub-directories below it.
        expected = {p for p in expected if "/" not in p[len(prefix):]}
    rows = [r["fortran"] for r in _rows() if r["family"] == family]
    assert len(rows) == len(set(rows)), f"{family}: a file has two rows"
    missing = sorted(expected - set(rows))
    unknown = sorted(set(rows) - expected)
    assert missing == [], (
        f"{family}: {len(missing)} census files have no row, e.g. {missing[:5]}"
    )
    assert unknown == [], f"{family}: rows for files the census does not list: {unknown[:5]}"
    assert len(rows) == PLANNED_COUNTS[family], (
        f"{family}: {len(rows)} rows, the plan's §Scope table says "
        f"{PLANNED_COUNTS[family]}"
    )


def test_every_census_shell_file_has_exactly_one_row():
    counted: dict[str, int] = {}
    for row in _rows():
        counted[row["fortran"]] = counted.get(row["fortran"], 0) + 1
    repeated = sorted(p for p, n in counted.items() if n != 1)
    assert repeated == [], f"files with a row count other than 1: {repeated[:5]}"
    orphans = sorted(_census_shell_paths() - set(counted))
    assert orphans == [], f"{len(orphans)} census shell files have no row, e.g. {orphans[:5]}"


def test_routine_name_matches_the_file():
    for row in _rows():
        name = row["fortran"].rsplit("/", 1)[-1].split(".")[0].lower()
        assert row["routine"].lower() == name, (
            f"{row['fortran']} carries routine {row['routine']!r}"
        )


# --------------------------------------------------------------------------
# A claim that names a module is checked, not believed.
# --------------------------------------------------------------------------


def test_port_module_is_never_invented():
    """A non-null ``port_module`` must be a module this repository can import."""
    for row in _rows():
        module = row["port_module"]
        if module is None:
            continue
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - the failure IS the report
            pytest.fail(f"{row['fortran']} names port_module {module!r}: {exc}")


def test_ported_and_partial_rows_name_a_symbol():
    for row in _rows():
        if row["status"] in ("ported", "partial", "approximate"):
            assert row["port_module"], f"{row['fortran']}: {row['status']} without a module"
            assert row["port_symbol"], f"{row['fortran']}: {row['status']} without a symbol"


def test_port_symbol_exists_in_its_module():
    for row in _rows():
        module, symbol = row["port_module"], row["port_symbol"]
        if module is None or symbol is None:
            continue
        mod = importlib.import_module(module)
        assert hasattr(mod, symbol), (
            f"{row['fortran']} names {module}.{symbol}, which does not exist"
        )


def test_missing_rows_have_no_port_claim():
    for row in _rows():
        if row["status"] == "missing":
            assert row["port_module"] is None, f"{row['fortran']} is missing but names a module"
            assert row["port_symbol"] is None, f"{row['fortran']} is missing but names a symbol"


# --------------------------------------------------------------------------
# No known-wrong result sits in the table unnamed.
# --------------------------------------------------------------------------


def test_every_partial_or_approximate_row_carries_a_deviation_note():
    for row in _rows():
        if row["status"] in ("partial", "approximate"):
            assert row["deviation_note"], (
                f"{row['fortran']} is {row['status']} with no deviation_note naming the "
                f"difference from upstream"
            )


def test_ported_rows_say_what_was_not_carried_over():
    """``ported`` is a claim of same-formulation, not of completeness."""
    for row in _rows():
        if row["status"] == "ported":
            assert row["deviation_note"], (
                f"{row['fortran']} is ported with no note of what upstream does "
                f"that the port does not"
            )


# --------------------------------------------------------------------------
# The parity sub-record: one per family, seeded from the measured cases.
# --------------------------------------------------------------------------


def test_every_family_carries_one_parity_sub_record():
    for family in FAMILY_PREFIX:
        rows = [r for r in _rows() if r["family"] == family]
        assert rows, f"no rows for family {family!r}"
        records = [json.dumps(r["parity"], sort_keys=True) for r in rows]
        assert len(set(records)) == 1, f"{family}: rows disagree about their parity record"
        sub = rows[0]["parity"]
        assert sub["family"] == family
        for key in ("cases", "worst_channel", "source"):
            assert key in sub, f"{family} parity record has no {key!r}"


def test_dkt18_parity_record_matches_the_measured_m41_cases():
    sub = _dkt18_rows()[0]["parity"]
    for case, rel_rms in DKT18_CASES.items():
        assert case in sub["cases"], f"the DKT18 parity record has no {case} row"
        assert abs(sub["cases"][case] - rel_rms) < 5e-4, (
            f"{case} recorded as {sub['cases'][case]}, parity_m41.json measures {rel_rms}"
        )
    assert sub["worst_channel"] == "MOMX"


def test_parity_cases_name_measured_cases_not_invented_ones():
    measured = {
        (r.get("case") or "").split("_")[0]
        for r in json.loads(PARITY_M41_JSON.read_text())["results"]
    }
    for row in _rows():
        for case in row["parity"]["cases"]:
            assert case in measured, (
                f"{row['fortran']}: parity case {case!r} is not in parity_m41.json"
            )


# --------------------------------------------------------------------------
# DKT18: the MOMX localisation is recorded as a hypothesis, per routine.
# --------------------------------------------------------------------------


def test_every_dkt18_row_carries_a_momx_hypothesis():
    rows = _dkt18_rows()
    assert rows
    for row in rows:
        assert row["hypothesis"], (
            f"{row['fortran']} carries no MOMX hypothesis — Step 3 of Task P3.0 asks "
            f"for one per routine"
        )


def test_hypotheses_are_labelled_as_hypotheses():
    for row in _rows():
        if row["hypothesis"]:
            assert row["hypothesis"].lstrip().lower().startswith("hypothesis:"), (
                f"{row['fortran']}: a hypothesis must read as one"
            )


def test_only_dkt18_rows_carry_a_hypothesis():
    for row in _rows():
        if row["family"] != "dkt18":
            assert not row["hypothesis"], (
                f"{row['fortran']} is {row['family']}: the MOMX localisation is a "
                f"DKT18 question only"
            )


def test_force_and_strain_hypotheses_are_named_by_the_plan():
    """The plan names the two discriminating signatures; the table must use them."""
    text = STATUS_JSON.read_text()
    assert "cdkfint3.F" in text and "cdkdefo3.F" in text and "cdkderi3.F" in text
    rows = {r["fortran"]: r for r in _dkt18_rows()}
    strain_hyp = rows["engine/source/elements/sh3n/coquedk/cdkdefo3.F"]["hypothesis"]
    assert "strain" in strain_hyp.lower()