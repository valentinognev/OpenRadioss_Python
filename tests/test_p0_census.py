"""Task P1.3 — the machine-readable Fortran<->Python coverage census.

WHAT THIS FILE IS FOR
---------------------
Until this task landed, "what has the port covered?" had no answer a machine
could give.  The answer lived in prose: milestone rows in ``README.md``, the
readiness narrative in ``docs/STATE.md``, and the reviewer's memory.  Phase 2
needs to *dispatch* work — ``plan/03_phase2_elements_solid.md:718`` runs
``tools/census.py --area engine/source/elements/solid`` and the phase reviewer
then reads the table and asserts no row remains ``missing`` — and prose cannot
be filtered by directory.

So this task adds a reader, not a judgement: :func:`tools.census.scan` walks the
upstream tree and parses the provenance block that upstream already writes into
every source file.

THE PROVENANCE BLOCK IS READ, NOT INVENTED
------------------------------------------
The layout is not a guess.  It is read off
``$OR_SRC/engine/source/elements/solid/solide/sforc3.F:24-60``, which this
repository's ``plan/02_phase1_foundation.md:206-209`` names as the reference,
and it looks like this::

    !||====================================================================
    !||    sforc3                 ../engine/source/elements/solid/solide/sforc3.F
    !||--- called by ------------------------------------------------------
    !||    alemain                ../engine/source/ale/alemain.F
    !||    forint                 ../engine/source/elements/forint.F
    !||--- calls      -----------------------------------------------------
    !||    aleflow                ../engine/source/ale/porous/aleflow.F
    !||--- uses       -----------------------------------------------------
    !||    ale_connectivity_mod   ../common_source/modules/ale/ale_connectivity_mod.F
    !||====================================================================

That is: a ``!||====`` fence, a self line naming the routine and its own path,
then any number of ``--- <section> ---`` markers each followed by
``name<spaces>path`` rows, closed by a second ``!||====`` fence.  Sections are
optional — upstream files that call nothing (``engine/share/resol/initbuf.F``
has ``called by`` and no ``calls``) exist, so a missing section is empty, not
an error.  Measured over the tree, the only three section markers upstream emits
are ``called by`` (:data:`SECTION_CALLED_BY`), ``calls``
(:data:`SECTION_CALLS`) and ``uses`` (:data:`SECTION_USES`).

WHAT `status` IS NOT
--------------------
The dangerous field is ``status``, because ``ported`` is a claim about physics
and a wrong one is invisible: nothing crashes, the table just lies.  So status
is read from exactly one place — ``tools/validation_data/port_status.json`` —
and **a file absent from that allowlist is ``missing``, never ``ported``**.  The
allowlist ships empty; populating it is a later phase's job, and it is the only
thing in this program that may promote a row.

Note what that means for :func:`test_census_claims_nothing_it_cannot_prove`:
with an empty allowlist every row is ``missing``, so the census is, at first,
an honest statement that *nothing* has been verified as ported.  That is the
correct conservative default, and it is why :func:`test_census_maps_python_modules_from_citations`
matters more than it looks: ``python_module`` is filled from the port's own
docstrings, so the census still says *where* the port says it implements each
file even while it refuses to say *that it does*.

THE SCOPE OF A `!||` BLOCK IS NOT A CALL GRAPH WE OWN
----------------------------------------------------
``calls``/``called_by`` come from upstream's header, which upstream generates.
The census reports them; it does not verify them and does not attempt to
re-derive them from the Fortran bodies.  Anything else would be a second,
disagreeing call graph, and the disagreement would be undetectable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import census  # noqa: E402
# The reference file the layout was read from, and the two probes the plan's
# Step-1 sketch names.  Both live under engine/source, so a tree that lost the
# engine subtree would fail here rather than quietly return a short census.
_SFORC3 = "engine/source/elements/solid/solide/sforc3.F"
_RESOL = "engine/source/engine/resol.F"

# The plan's Step-1 sketch pins the time-history check to
# ``engine/source/output/th/th1t.F90``.  That file does not exist in this
# upstream tree: zero hits for ``th1t`` across all 5,150 commits, so the plan
# asserted against a name that is not there (its own sketch was written
# defensively — ``... if ... in c.files else None`` — and would still have
# failed the ``assert anim is not None`` that follows).  The time-history
# output driver that *does* exist is ``th_time_output.F``, so that is what the
# classification is asserted against.  The absent name is kept as a second
# constant and asserted absent below, so the discrepancy stays recorded rather
# than quietly forgotten.
_TH1T = "engine/source/output/th/th1t.F90"
_TH_OUTPUT = "engine/source/output/th/th_time_output.F"


@pytest.fixture(scope="module")
def census_data():
    """One scan for the whole module — :func:`scan` walks ~5.9k files."""
    return census.scan()


# --- the two assertions the plan prescribes verbatim -----------------------


def test_census_finds_every_fortran_source_file(census_data):
    """The census must cover the whole upstream tree, not a curated subset.

    Measured: 5,887 Fortran files under ``$OR_SRC``.  The threshold is 5000
    because the plan's sketch says 5000; the real number sits well above it, so
    a regression that silently skips a subtree still trips this.
    """
    assert len(census_data.files) > 5000
    assert census_data.files[_RESOL].lines > 100
    assert "forint" in census_data.files[_SFORC3].called_by


def test_census_classifies_the_output_tree_as_missing(census_data):
    """Time-history output is not ported, and the census must say so.

    This is the assertion that pins the *conservative* direction.  It is
    written as ``in {"missing", "partial"}`` rather than ``== "missing"`` so a
    later phase may legitimately promote the row to ``partial`` with a cited
    entry in ``port_status.json`` — but it can never become ``ported`` on
    silence.
    """
    anim = census_data.files.get(_TH_OUTPUT)
    assert anim is not None
    assert anim.status in {"missing", "partial"}


def test_census_does_not_invent_a_file_upstream_does_not_have(census_data):
    """A census that fabricates a row is worse than one that omits it.

    The plan's sketch named ``th1t.F90``; the upstream tree has no such file in
    any of its 5,150 commits.  A tool that satisfied the sketch by synthesising
    the row would put a record into ``census.json`` that names a routine no
    engineer can go and read — and phase exit gates are written to *count*
    these rows.  Asserting the row is absent keeps the census honest about the
    tree it describes.
    """
    assert _TH1T not in census_data.files
    assert not (census.or_src() / _TH1T).exists()


# --- the parsed block ------------------------------------------------------


def test_block_parser_reads_the_sforc3_header_verbatim():
    """The three sections of the reference file, parsed from its real text.

    Written against the literal upstream lines (line 24-60 of ``sforc3.F``) so
    a parser that invents a layout fails here rather than only in production.
    """
    text = "\n".join(
        [
            "!||====================================================================",
            "!||    sforc3                 ../engine/source/elements/solid/solide/sforc3.F",
            "!||--- called by ------------------------------------------------------",
            "!||    alemain                ../engine/source/ale/alemain.F",
            "!||    forint                 ../engine/source/elements/forint.F",
            "!||--- calls      -----------------------------------------------------",
            "!||    aleflow                ../engine/source/ale/porous/aleflow.F",
            "!||--- uses       -----------------------------------------------------",
            "!||    ale_connectivity_mod   ../common_source/modules/ale/ale_connectivity_mod.F",
            "!||====================================================================",
            "!||    SUBROUTINE SFORC3(NELT,...)",
        ]
    )
    parsed = census.parse_provenance(text)
    assert parsed is not None
    assert parsed.routine == "sforc3"
    assert parsed.self_path == _SFORC3
    assert parsed.called_by == ("alemain", "forint")
    assert parsed.calls == ("aleflow",)
    assert parsed.uses == ("ale_connectivity_mod",)


def test_block_parser_treats_an_absent_section_as_empty():
    """``engine/share/resol/initbuf.F`` has ``called by`` and no ``calls``.

    An upstream file that calls nothing is normal, so a missing section must
    yield an empty tuple.  Anything that raised, or invented members, would
    make the census unreadable on a large fraction of the tree.
    """
    text = "\n".join(
        [
            "!||====================================================================",
            "!||    initbuf                      ../engine/share/resol/initbuf.F",
            "!||--- called by ------------------------------------------------------",
            "!||    aconve                           ../engine/source/ale/aconve.F90",
            "!||====================================================================",
        ]
    )
    parsed = census.parse_provenance(text)
    assert parsed is not None
    assert parsed.routine == "initbuf"
    assert parsed.called_by == ("aconve",)
    assert parsed.calls == ()
    assert parsed.uses == ()


def test_block_parser_returns_none_when_there_is_no_block():
    """A file with no ``!||`` fence is simply not annotated.

    Returning ``None`` (rather than an empty parse) keeps "upstream did not
    annotate this" distinguishable from "upstream annotated it with nothing".
    """
    assert census.parse_provenance("      PROGRAM MAIN\n      END\n") is None


def test_paths_are_normalised_to_the_tree_root(census_data):
    """Every key is relative to ``$OR_SRC`` and carries no ``../`` prefix.

    The self path inside the block is written ``../engine/...`` — relative to
    the file's own directory, not the tree root — so a naive copy would key
    every file by a path that does not resolve.  Normalising here is what lets
    a docstring citation match it later.
    """
    for path in census_data.files:
        assert not path.startswith("../"), path
        assert not path.startswith("/"), path
        assert ".." not in Path(path).parts, path


def test_line_counts_are_positive(census_data):
    """``lines`` counts real source lines, so it is never zero."""
    assert all(r.lines > 0 for r in census_data.files.values())


# --- python_module ---------------------------------------------------------


def test_census_maps_python_modules_from_citations(census_data):
    """``python_module`` comes from docstrings, never from a hand-written table.

    Task P1.2 normalised every citation to the ``$OR_SRC`` form, which is what
    makes this join possible at all.  At least one citation must have landed or
    the mapping silently degrades to all-``None`` and nobody notices.
    """
    mapped = {p: r.python_module for p, r in census_data.files.items()
              if r.python_module is not None}
    assert mapped, "no docstring citation resolved to an upstream path"

    # The spring element cites its upstream source six times; it is the
    # densest citation in the tree and must resolve to the module that owns it.
    spring = "engine/source/elements/spring/rforc3.F"
    assert spring in mapped, f"{spring} is cited by the port but did not map"
    assert mapped[spring].startswith("pyradioss.")
    candidate = _REPO_ROOT / mapped[spring].replace(".", "/")
    assert candidate.with_suffix(".py").exists() or (candidate / "__init__.py").exists()


def test_python_module_is_none_when_nothing_cites_the_file(census_data):
    """Most upstream files are uncited, and the census must not invent modules."""
    cited = sum(1 for r in census_data.files.values() if r.python_module)
    assert 0 < cited < len(census_data.files)


def test_python_module_never_names_the_census_itself(census_data):
    """A reader or a test cannot be the implementation of a Fortran file.

    This file and ``tools/census.py`` both quote upstream paths in prose, so an
    unrestricted repository-wide citation scan handed ``sforc3.F`` to
    ``tests.test_p0_census`` — the census claiming that its own test implements
    the Fortran it reads.  Nothing else here would have caught it: the status
    column stays ``missing`` either way, and the mapped path is a real file.
    Only the *identity* of the claimant is wrong, and phase exit gates read
    these rows.
    """
    for path, rec in census_data.files.items():
        if rec.python_module is None:
            continue
        assert rec.python_module.startswith("pyradioss."), f"{path}: {rec.python_module}"
        assert not rec.python_module.startswith("tools"), path
        assert not rec.python_module.startswith("tests"), path


# --- status is the allowlist, and only the allowlist -----------------------


def test_status_comes_only_from_the_allowlist(census_data):
    """An unlisted file is ``missing`` — never ``ported``.

    This is the rule that makes the ``ported`` row trustworthy, and it is the
    one a future contributor is most likely to break by "just fixing" the
    default.  With the shipped-empty allowlist every row must be ``missing``.
    """
    allowlist = census.load_port_status()
    assert allowlist == {}, "the allowlist must ship empty; phases populate it"
    for path, rec in census_data.files.items():
        assert rec.status == "missing", f"{path} was promoted without an entry"


def test_census_claims_nothing_it_cannot_prove(census_data):
    """No ``ported`` or ``stub`` row may exist while the allowlist is empty."""
    statuses = {r.status for r in census_data.files.values()}
    assert statuses <= {"missing"}


def test_committed_census_json_agrees_with_a_fresh_scan(census_data):
    """The committed record must be the one the tool produces now.

    A census that is regenerated by hand, or committed from a different tree
    state, is exactly the rot this program keeps getting burned by.  Compare
    the file set and the statuses — not the timestamps — so the check is
    deterministic.
    """
    committed_path = census.CENSUS_JSON
    if not committed_path.exists():
        pytest.skip("census.json has not been generated yet")
    committed = json.loads(committed_path.read_text())
    assert set(committed["files"]) == set(census_data.files)
    for path, rec in committed["files"].items():
        assert rec["status"] == census_data.files[path].status, path


def test_committed_artifacts_are_byte_reproducible(census_data, tmp_path):
    """Regenerating must reproduce both artifacts exactly, byte for byte.

    A census that comes out differently each run cannot be diffed, so a phase
    reviewer cannot see what a task actually changed in coverage — the diff
    would be noise from dict ordering or a timestamp, and would train everyone
    to regenerate without reading.  Comparing the rendered bytes is the only
    check that catches a nondeterministic ordering.
    """
    md = tmp_path / "CENSUS.md"
    js = tmp_path / "census.json"
    assert census.main(["--render", str(md), "--json", str(js), "--quiet"]) == 0

    assert census.CENSUS_MD.exists(), "plan/CENSUS.md has not been generated"
    assert census.CENSUS_JSON.exists(), "census.json has not been generated"
    assert md.read_text() == census.CENSUS_MD.read_text()
    assert js.read_text() == census.CENSUS_JSON.read_text()


# --- area, render, CLI -----------------------------------------------------


def test_area_is_a_prefix_of_the_path(census_data):
    """``area`` is the subsystem a file belongs to, so it prefixes ``path``.

    ``--area engine/source/elements/solid`` is how the phase-2 exit gate
    narrows the table (``plan/03_phase2_elements_solid.md:718``); that only
    works if the field is a path prefix rather than a hand-assigned label.
    """
    for path, rec in census_data.files.items():
        assert rec.area == "" or path.startswith(rec.area), path
        assert ".." not in rec.area


def test_area_filter_matches_whole_path_components(census_data):
    """``--area .../solid`` must not capture ``.../solid_2d``.

    A raw string prefix says ``"…/solid_2d/quad".startswith("…/solid")``, so the
    filter returned 46 rows from the 2D-solid scope inside the 3D-solid view.
    That is not a cosmetic overlap: the phase-2 exit gate
    (``plan/03_phase2_elements_solid.md:718``) runs exactly this filter and then
    counts the ``missing`` rows to decide whether the phase is finished, so a
    leak there decides the gate on rows that are not in scope.
    """
    view = census_data.select(area="engine/source/elements/solid")
    assert view.files, "the solid area must not be empty"
    for path in view.files:
        assert census._is_under(path, "engine/source/elements/solid"), path

    # The leaked sibling is a real directory in this tree, so the test has teeth.
    sibling = [p for p in census_data.files
               if p.startswith("engine/source/elements/solid_2d/")]
    assert sibling, "expected the solid_2d subtree to exist for this to matter"
    assert not set(sibling) & set(view.files)


def test_render_emits_a_markdown_table_with_a_status_column(census_data):
    """The rendered table is what a phase reviewer actually reads."""
    text = census.render(census_data)
    assert text.startswith("# Fortran")
    assert "| area |" in text or "| area " in text
    assert "status" in text
    assert str(len(census_data.files)) in text


def test_generated_records_name_the_tree_indirectly(census_data):
    """``CENSUS.md`` and ``census.json`` must not spell out this box's home.

    Both artifacts are committed records, and this repository's rule is that a
    record which names ``/home/<user>/...`` is false on every other machine.
    The first generated ``CENSUS.md`` did exactly that and
    ``tests/test_p0_no_stale_machine_paths.py`` failed on it — caught by the
    gate built for this class of rot, one task late.  Pin the rule here so the
    generator cannot reintroduce it.
    """
    text = census.render(census_data)
    assert "$OR_SRC" in text
    assert str(census_data.or_src) not in text


def test_render_narrows_to_an_area(census_data):
    """``render(c, area=...)`` must not leak rows from other subsystems."""
    text = census.render(census_data, area="engine/source/engine")
    assert "resol.F" in text
    assert "sforc3.F" not in text


def test_main_writes_both_artifacts(tmp_path):
    """The CLI writes what the plan's commands ask for."""
    md = tmp_path / "CENSUS.md"
    js = tmp_path / "census.json"
    rc = census.main(["--render", str(md), "--json", str(js)])
    assert rc == 0
    assert md.exists() and js.exists()

    payload = json.loads(js.read_text())
    assert len(payload["files"]) > 5000
    rec = payload["files"]["engine/source/elements/solid/solide/sforc3.F"]
    assert rec["lines"] > 0
    assert "forint" in rec["called_by"]
    assert rec["area"] == "engine/source/elements/solid/solide"


def test_main_status_missing_prints_only_the_missing_rows(capsys):
    """``--status-missing`` is the "what is left to do" view.

    It must filter to ``missing`` *and* report the filtered count, so a phase
    reviewer does not read "0 rows" as "the whole tree is ported".
    """
    rc = census.main(["--status-missing"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "missing" in out
    assert str(len(census.scan().files)) in out


def test_main_rejects_an_unknown_area(capsys):
    """A typo'd ``--area`` must fail loudly, not render an empty table."""
    with pytest.raises(SystemExit) as exc:
        census.main(["--area", "engine/source/no_such_area"])
    assert exc.value.code == 2


def test_upstream_tree_is_never_written_to(census_data, tmp_path):
    """The upstream tree is read-only; scanning must not touch it.

    ``$OR_SRC`` is upstream AGPL source the port only ever reads.  Guard the
    property rather than the intent: snapshot the newest mtime under the tree
    before and after a scan and require it unchanged.
    """
    src = census.or_src()
    before = census._newest_mtime(src)
    census.scan()
    assert census._newest_mtime(src) == before


def test_census_does_not_invent_fortran_extensions(census_data):
    """Only upstream's Fortran suffixes are scanned.

    The tree also holds C, C++ and headers.  Including them would silently
    change what "5,887 files" means between runs whenever upstream adds a C
    file, so the suffix set is pinned here.
    """
    assert census.FORTRAN_SUFFIXES == frozenset({".F", ".F90", ".f", ".f90", ".for"})
    for path in census_data.files:
        assert Path(path).suffix in census.FORTRAN_SUFFIXES


def test_module_declares_the_documented_section_markers():
    """The three section markers are upstream's, and the parser names them.

    Measured over the whole tree, upstream emits exactly these three and no
    other.  If upstream adds a fourth, this test should fail so the parser can
    be taught to read it rather than dropping its rows on the floor.
    """
    assert census.SECTION_CALLED_BY == "called by"
    assert census.SECTION_CALLS == "calls"
    assert census.SECTION_USES == "uses"


def test_forbidden_fake_status_is_not_in_the_source():
    """A guard against the failure mode this task exists to prevent.

    Reading the *tool's own source* is unusual in this repository, but the
    specific regression it catches — someone hard-coding ``ported`` to make a
    demo table look finished — is invisible in every other test here, because
    every other assertion is satisfied by the allowlist being empty.
    """
    src = census.__file__
    text = Path(src).read_text()
    # Any occurrence must be inside the allowlist vocabulary, never a literal
    # assignment of a row's status.
    assert not re.search(r'status\s*=\s*"ported"', text)
    assert not re.search(r'status\s*=\s*"stub"', text)
