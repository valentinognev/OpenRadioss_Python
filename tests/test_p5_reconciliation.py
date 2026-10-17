"""Task P5.0 — the per-routine reconciliation audit of the special elements.

WHAT THIS FILE IS FOR
---------------------
``plan/06_phase5_elements_special.md`` §Task P5.0 asks for one row per
upstream routine across ``engine/source/elements/{sph,xfem,ige3d,elbuf}/``
(46 + 40 + 15 + 6 files), plus ``starter/source/elements/{sph,ige3d,xfem}/``
and ``starter/source/elements/elbuf_init/``, plus the two
``common_source/modules`` type-definition files the phase scope names
(``xfem2def_mod.F``, ``nlocal_reg_mod.F``) — 176 rows in
``tools/validation_data/special_routine_status.json``.

Each row carries the standard record shape (the seven P2.0 keys plus
``family``) **plus a ``subfamily`` field** (``geometry``, ``stress``,
``enrichment``, ``crack``, ``state``, ``bucket``, ``integrator``) so that
Phase 17 can report per-capability coverage rather than per-file.

TWO CLAIMS VERIFIED HERE, NOT ASSUMED
-------------------------------------
(a) ``elements/iga3d.py`` is NOT a genuine ``onebasisfun.F`` port: the
    module routes around ``onebasisfun.F``/``ig3donebasis.F`` through the
    ders-family (``ders_one_basis_fun``/``ders_basis_funs``) evaluated on
    single-span open-uniform knots built from the element's own knot span.
    The ``onebasisfun.F`` row is therefore ``missing`` and says so.
(b) The SPH conservation tests (``test_sph_pressure_forces_antisymmetric``,
    ``test_sph_mass_conservation``) pass because the force assembly is
    pairwise-antisymmetric *by construction* — which is the correct upstream
    ``spforcp.F`` structure — but that same structure means the tests cannot
    catch a wrong pressure/viscosity magnitude, a wrong EOS, or any of the
    unported state machinery (``spreploc``, ``splissv``, ``spmall3`` …).
    The audit record states what the tests do and do not pin.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = ROOT / "tools" / "validation_data" / "special_routine_status.json"
CENSUS_JSON = ROOT / "tools" / "validation_data" / "census.json"

STATUSES = {"ported", "partial", "approximate", "missing", "refused"}
ROW_KEYS = {
    "family",
    "fortran",
    "routine",
    "port_module",
    "port_symbol",
    "status",
    "deviation_note",
    "parity_case",
    "subfamily",
}
FAMILIES = {"sph", "xfem", "ige3d", "elbuf", "common"}
SUBFAMILIES = {
    "geometry",
    "stress",
    "enrichment",
    "crack",
    "state",
    "bucket",
    "integrator",
}

#: Upstream directory -> (family, planned file count), from
#: ``plan/06_phase5_elements_special.md`` §Scope and the starter tree.
ENGINE_DIRS = {
    "engine/source/elements/sph/": ("sph", 46),
    "engine/source/elements/xfem/": ("xfem", 40),
    "engine/source/elements/ige3d/": ("ige3d", 15),
    "engine/source/elements/elbuf/": ("elbuf", 6),
}
STARTER_DIRS = {
    "starter/source/elements/sph/": ("sph", 16),
    "starter/source/elements/ige3d/": ("ige3d", 24),
    "starter/source/elements/xfem/": ("xfem", 12),
    "starter/source/elements/elbuf_init/": ("elbuf", 15),
}
#: The two type-definition modules the phase scope names (census holds the
#: ``.F`` spelling; the plan text writes ``.F90``).
COMMON_FILES = {
    "common_source/modules/xfem2def_mod.F",
    "common_source/modules/nlocal_reg_mod.F",
}


def _rows() -> list[dict]:
    return json.loads(STATUS_JSON.read_text())


def _scope_census_paths() -> set[str]:
    files = json.loads(CENSUS_JSON.read_text())["files"]
    prefixes = tuple(list(ENGINE_DIRS) + list(STARTER_DIRS))
    return {p for p in files if p.startswith(prefixes)} | (
        COMMON_FILES & set(files)
    )


# --------------------------------------------------------------------------
# Step 1 of the plan, verbatim.
# --------------------------------------------------------------------------


def test_sph_subfamilies_are_all_covered():
    st = json.loads(Path("tools/validation_data/special_routine_status.json").read_text())
    sph = [r for r in st if r["family"] == "sph"]
    assert {r["subfamily"] for r in sph} >= {
        "geometry","stress","enrichment","state","bucket","integrator"}


# --------------------------------------------------------------------------
# The table is a per-routine audit, not a summary.
# --------------------------------------------------------------------------


def test_status_table_exists_and_is_a_list_of_rows():
    assert STATUS_JSON.is_file(), (
        f"{STATUS_JSON} is missing — Task P5.0's audit table has not been built"
    )
    rows = _rows()
    assert isinstance(rows, list) and rows, "the status table is an empty list"


@pytest.mark.parametrize("key", sorted(ROW_KEYS))
def test_row_carries_every_key(key):
    for row in _rows():
        assert key in row, f"{row.get('fortran')} has no {key!r}"


def test_status_is_one_of_the_five():
    for row in _rows():
        assert row["status"] in STATUSES, f"{row['fortran']}: {row['status']!r}"


def test_subfamily_is_a_declared_value():
    for row in _rows():
        assert row["subfamily"] in SUBFAMILIES, (
            f"{row['fortran']}: subfamily {row['subfamily']!r} is not declared"
        )


def test_family_is_declared():
    for row in _rows():
        assert row["family"] in FAMILIES, (
            f"{row['fortran']}: family {row['family']!r} is not declared"
        )


def test_routine_name_matches_the_file():
    for row in _rows():
        name = row["fortran"].rsplit("/", 1)[-1].split(".")[0].lower()
        assert row["routine"].lower() == name, (
            f"{row['fortran']} carries routine {row['routine']!r}"
        )


def test_every_census_scope_file_has_exactly_one_row():
    counted: dict[str, int] = {}
    for row in _rows():
        counted[row["fortran"]] = counted.get(row["fortran"], 0) + 1
    for path in _scope_census_paths():
        assert counted.get(path, 0) == 1, (
            f"{path}: has {counted.get(path, 0)} audit rows, expected exactly 1"
        )
    orphans = sorted(set(counted) - _scope_census_paths())
    assert orphans == [], f"rows for files outside the P5.0 scope: {orphans[:5]}"


@pytest.mark.parametrize("prefix", sorted({**ENGINE_DIRS, **STARTER_DIRS}))
def test_directory_covers_its_planned_file_count(prefix):
    dirs = {**ENGINE_DIRS, **STARTER_DIRS}
    family, count = dirs[prefix]
    rows = [r for r in _rows() if r["fortran"].startswith(prefix)]
    assert len(rows) == count, (
        f"{prefix}: {len(rows)} rows, the plan's §Scope tree holds {count}"
    )
    assert {r["family"] for r in rows} == {family}, (
        f"{prefix}: rows leak into another family"
    )


def test_common_modules_are_classified():
    rows = {r["fortran"]: r for r in _rows() if r["fortran"] in COMMON_FILES}
    assert set(rows) == COMMON_FILES, (
        f"missing common-module rows: {sorted(COMMON_FILES - set(rows))}"
    )
    for fortran, row in rows.items():
        assert row["family"] == "common", f"{fortran} is not family 'common'"


# --------------------------------------------------------------------------
# A claim that names a module is checked, not believed.
# --------------------------------------------------------------------------


def test_port_module_is_never_invented():
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


def test_missing_and_refused_rows_name_no_port():
    for row in _rows():
        if row["status"] in ("missing", "refused"):
            assert row["port_module"] is None, (
                f"{row['fortran']} is {row['status']} but names a module"
            )
            assert row["port_symbol"] is None, (
                f"{row['fortran']} is {row['status']} but names a symbol"
            )


def test_every_non_missing_row_carries_a_deviation_note():
    for row in _rows():
        if row["status"] in ("ported", "partial", "approximate"):
            assert row["deviation_note"], (
                f"{row['fortran']} is {row['status']} with no deviation_note"
            )


# --------------------------------------------------------------------------
# Claim (a): the IGA basis is a ders-family substitute, not onebasisfun.F.
# --------------------------------------------------------------------------


def test_iga_basis_claim_is_resolved_with_evidence():
    """``onebasisfun.F`` must be recorded ``missing`` with the substitute named.

    ``pyradioss/elements/iga3d.py`` defines ``ders_one_basis_fun`` (from
    ``dersonebasisfun.F``) and ``ders_basis_funs`` (from ``dersbasisfuns.F``)
    and builds single-span open-uniform knots from the element knot span —
    it defines no ``one_basis_fun`` symbol. The audit must record that,
    not a ported basis.
    """
    rows = {r["fortran"]: r for r in _rows()}
    basis = rows["engine/source/elements/ige3d/onebasisfun.F"]
    assert basis["status"] == "missing", (
        f"onebasisfun.F is {basis['status']}: iga3d.py defines no "
        f"one_basis_fun symbol (grep finds only docstring mentions)"
    )
    note = basis["deviation_note"] or ""
    assert "dersonebasisfun" in note and "iga3d" in note, (
        "the onebasisfun.F row does not name the ders-family substitute"
    )
    span_caller = rows["engine/source/elements/ige3d/ig3donebasis.F"]
    assert span_caller["status"] == "missing", (
        "ig3donebasis.F (the span-search caller of onebasisfun.F) must be "
        "missing alongside it"
    )


# --------------------------------------------------------------------------
# Claim (b): SPH conservation tests pin structure, not magnitudes.
# --------------------------------------------------------------------------


def test_sph_result_changing_machinery_is_recorded_missing():
    """``spreploc``/``splissv``/``spmall3`` change results and have no port."""
    rows = {r["fortran"]: r for r in _rows()}
    for stem in ("spreploc.F", "splissv.F", "spmall3.F"):
        fortran = f"engine/source/elements/sph/{stem}"
        assert rows[fortran]["status"] == "missing", (
            f"{fortran} is {rows[fortran]['status']}: the plan names it as "
            f"result-changing with no port counterpart"
        )


def test_sph_test_strength_is_stated_not_assumed():
    """The audit states what the conservation tests do and do not pin."""
    notes = " ".join((r["deviation_note"] or "") for r in _rows())
    for test_id in (
        "test_sph_pressure_forces_antisymmetric",
        "test_sph_mass_conservation",
    ):
        assert test_id in notes, (
            f"no audit note cites {test_id}: claim (b) is assumed, not evidenced"
        )
    assert "structural" in notes, (
        "no audit note states the antisymmetry-is-structural limitation"
    )
