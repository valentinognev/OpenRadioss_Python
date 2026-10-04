"""Task P0.18 — a record may not state an exact suite count without saying when.

THE ROT THIS EXISTS FOR
-----------------------
Three review rounds in a row flagged the same class of defect in the records:
the exact test counts written into ``docs/STATE.md``, ``UPDATES.md`` and
``plan/01_phase0_oracle_and_licensing.md`` aged ``68 → 70 → 71 → …`` and each
round a reviewer had to flag them again.  Nothing was enforcing the rule,
because there was no rule — only a convention everybody already broke.

**A record that states an exact count is a record that will be wrong.**  So the
rule this file enforces is about *binding*, not about the number:

    A suite count in a record is acceptable only if the statement says WHICH
    measurement moment it describes.

and the two treatments that follow from it are the ones the records now use:

* **evidence** (a before/after showing what a change did) — keep the number,
  bind it to the date/commit it was measured at, and let it age in place;
* **decoration** (a "baseline" block that merely restates the suite size) —
  drop the bare number, keep the **command** that produces the current one, and
  keep the dated measurement beside it as history.

WHAT COUNTS AS A COUNT, AND WHAT COUNTS AS A BINDING
----------------------------------------------------
:data:`SUITE_COUNT` is deliberately narrow — a number standing directly beside
a pytest RESULT word (``14594 passed``, ``71 passed, 0 skipped``, ``collected
14617``).  An earlier draft of this file also matched the bare word ``tests``,
which flagged 127 lines, of which 117 were milestone rows whose "34 new tests"
is a *delta* that does not rot the way a suite size does, and the rest were
"35 files, 10,356" and "937 files".  A rot scanner that cries wolf is disabled
by its first false positive, so the pattern is the narrow one.

A **binding** is any of (:data:`ISO_DATE`, a 7–40 hex **commit sha**,
:data:`AS_OF`), and it may sit:

* on the count's own line or within :data:`WINDOW` lines either side — a
  measurement is usually introduced by its date or closed by its commit, and a
  rule that only looked in one direction would force every sentence to be
  rewritten to suit the rule; or
* in the **nearest preceding markdown heading**, which is what makes a dated
  *log entry* legitimate rather than a violation.  This is the "a measurement
  inside a dated entry describes that moment" half of the rule, and it is why
  every ``## <version> - …`` heading in ``UPDATES.md`` and every dated audit
  heading in ``docs/`` carries a date.

Two shapes are not claims at all and are skipped, each for a stated reason:

* **a fenced block tagged with a language** (`````python``) — a code sample.
  ``plan/02_phase1_foundation.md`` shows ``Ledger(passed=10, …)`` as the API a
  future task must write; that is a specimen of code, not a statement about
  any suite.  Requiring the language tag is what keeps this from becoming a
  laundering route: an untagged fence is a specimen only if its opener
  attributes it, and that is the (already pinned) rule of
  ``tests/test_p0_no_stale_machine_paths.py``; this file does not weaken it,
  it simply does not duplicate it.
* **the count of a milestone row's own contribution** (``34 new tests``) — a
  delta, not a suite size.

§BASELINE IS HELD TO A STRICTER RULE
------------------------------------
Heading inheritance would be a hole in exactly the place the rot happened:
``docs/STATE.md`` §Baseline is the onboarding document, its heading has carried
a date since P0.17, and a bare "the suite has N tests" dropped in there would
inherit that date and pass.  So inside §Baseline a count must bind **locally**
(:func:`test_the_baseline_block_carries_no_inherited_binding`), which is why the
block now states commands and dated measurements instead of bare numbers, and
why :func:`test_the_baseline_names_the_command_not_only_a_number` pins the
command half of the treatment.

A PER-MODULE SIZE IS A CLAIM ABOUT NOW, AND ONLY A COMMAND CHECKS NOW
--------------------------------------------------------------------
:data:`SUITE_COUNT` above is deliberately narrow, and the narrowness had a hole
in it: it needs a pytest RESULT word, so it cannot see a figure whose unit is
``tests`` and whose subject is a module — ``**22** at 2026-10-04``.  Two records
carried exactly that, one in ``UPDATES.md`` and one in
``plan/01_phase0_oracle_and_licensing.md``, both **bound** to a date, both
looking honest, and both **wrong** (the measured figure was 24).  A binding rule
cannot catch that: 22 *was* a dated claim, it was simply not true, and nothing
short of re-running the count can tell.

So the treatment for a **size** is the decoration treatment stated above, and
the rule below is its teeth: a figure about a named ``tests/*.py`` module must
carry the command that re-measures it (:data:`_RECOUNT`), in the two lines
either side.  A date says *when the claim was made*; the command is the only
thing that says *whether it is still true*, which is the whole difference
between the figure being evidence and being decoration.

Two shapes are exempt, each for a stated reason:

* **a milestone table row** (``| M701 | …``) — the §"What is implemented"
  preamble already defines every row figure as *that milestone's own*
  measurement, so the row is a dated measurement of a past tree by
  construction, not a claim about the module today.
* **a contribution marker** (``New``/``added``/``created``/``fresh``) — the
  figure is what a task *delivered*, and the dated entry that records the task
  binds it.  Same category as the ``34 new tests`` delta skipped above.

WHAT IS DELIBERATELY OUT OF REACH
---------------------------------
A count with **no** RESULT word **and no named module** (``the reader library
is 1,210 lines``), and a count in ``.superpowers/`` reports — those are working
notes, not records.  The record scope is imported from the sibling gate rather
than re-listed, so a record added to ``docs/`` or ``plan/`` tomorrow is scanned
by both rules without anyone remembering to update either file.
"""

from __future__ import annotations

import re

from tests.test_p0_no_stale_machine_paths import (
    STATE,
    _quoted_example_lines,
    _record_sources,
)

#: ``<number> <pytest RESULT>``, or the two spellings where the word comes
#: first (``collected 14617``).  Thousands separators are allowed because two of
#: the dated audits write ``5,498 passed``.
RESULT = (r"(?:passed|failed|skipped|deselected|xfailed|xpassed|errors?"
          r"|collected)")
SUITE_COUNT = re.compile(
    rf"\b\d[\d,]*\s+(?:{RESULT})\b"
    rf"|\b(?:passed|collected|deselected|skipped)\s+\d[\d,]*\b")

#: An ISO date — the cheapest binding and the one every record already writes.
ISO_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")

#: A 7–40 character lowercase hex token: a git sha.  Uppercase is excluded on
#: purpose (an md5 digest is a different kind of fact, and ``20d4059`` in a
#: banner would otherwise pass as a commit).
COMMIT = re.compile(r"\b[0-9a-f]{7,40}\b")

#: Words that say the figure describes *a moment* rather than *now*.  Weaker
#: than a date and accepted only beside one of them is not required — the point
#: is that a record may LABEL history in prose, which is what the brief asks of
#: a dated entry, without inventing a date for it.
AS_OF = re.compile(
    r"\b(?:as of|at the time|histor\w*|superseded|withdrawn|at that date)\b",
    re.I)

#: How far either side of a count a binding may sit.
WINDOW = 2

#: The languages whose fenced block is a CODE SAMPLE rather than prose.
CODE_INFO = re.compile(r"^```\s*[A-Za-z0-9_+#-]+\s*$")

#: ``tests/<module>.py`` — the *subject* of a size figure.  Scoped to that shape
#: on purpose: a figure naming a module is re-derivable (run pytest on that one
#: file), which is what makes the command requirement below reasonable.  A
#: figure like "their 20 tests stay green" describes a set of behaviours in a
#: dated entry and has no single command.
NAMED_MODULE = re.compile(r"\btests/[A-Za-z0-9_]+\.py\b")

#: The unit a size is stated in.  :data:`SUITE_COUNT` cannot see it, because that
#: pattern requires a pytest RESULT word and this shape has none — the blind spot
#: this whole section exists to close.
SIZE_UNIT = re.compile(r"\b\d[\d,]*\s*\*?\*?\s*(?:tests?|test\s+cases?)\b", re.I)

#: The command that re-measures a module's size.  Named as a *shape* (the flag),
#: not the whole incantation, so an invocation written differently still counts.
_RECOUNT = re.compile(r"--collect-only")

#: What ends the clause a figure lives in.  ``.w`` guards against a row that
#: simply runs on: ``… and ``tools/x.py``: 7 tests`` is a second clause.
_SIZE_CLAUSE_BREAK = re.compile(r"[;:!?]|\.\w")

#: How far after the module's name its figure may sit and still be *about* it.
_SIZE_GAP = 70

#: A milestone row of the §"What is implemented" table, whose preamble makes
#: every row figure that milestone's own measurement.
_MILESTONE_ROW = re.compile(r"^\s*\|\s*M\d")

#: A figure that states what a task DELIVERED.  Bound by the dated entry that
#: records the task, in the same category as the ``34 new tests`` delta.
CONTRIBUTION = re.compile(r"\b(?:new|added|created|fresh)\b", re.I)


def _size_about_a_named_module(line: str) -> bool:
    """Does ``line`` state how many tests a named module has?

    The module name and the figure must be in the **same clause**: ``… and
    ``tools/x.py``: 7 tests`` is two claims, and the second one is about a
    different file.
    """
    for named in NAMED_MODULE.finditer(line):
        span = line[named.end():][:_SIZE_GAP]
        brk = _SIZE_CLAUSE_BREAK.search(span)
        if brk:
            span = span[:brk.start()]
        if SIZE_UNIT.search(span):
            return True
    return False


def unreproducible_module_sizes(lines: list[str]) -> list[int]:
    """Indices of lines stating a module's size with no way to re-check it.

    Binding is already handled by :func:`unbound`; this is the other half.  A
    size changes the moment anybody adds a test to the module, so the date
    beside it binds only the past — and the figure this rule was written for was
    bound to 2026-10-04, looked impeccable, and was wrong by two.
    """
    fenced, quoted = _fenced(lines), _quoted_example_lines(lines)
    out = []
    for index, line in enumerate(lines):
        if index in fenced or index in quoted:
            continue
        if _MILESTONE_ROW.match(line) or CONTRIBUTION.search(line):
            continue          # a past tree's figure, or a task's own output
        if not _size_about_a_named_module(line):
            continue
        window = "\n".join(lines[max(0, index - WINDOW):index + WINDOW + 1])
        if not _RECOUNT.search(window):
            out.append(index)
    return out


def _fenced(lines: list[str]) -> set[int]:
    """0-based indices of lines inside a LANGUAGE-TAGGED fenced block.

    An untagged fence is deliberately **not** skipped: quoting a claim is not
    the same as making it true, and
    ``tests/test_p0_no_stale_machine_paths.py`` already holds the untagged case
    to its own (stricter, attributed-opener) rule.  This helper only exempts
    the one shape that cannot be a claim at all — source code.
    """
    skipped: set[int] = set()
    inside = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            skipped.add(index)
            inside = not inside if inside or CODE_INFO.match(stripped) else False
            continue
        if inside:
            skipped.add(index)
    return skipped


def _heading_before(lines: list[str], index: int) -> str:
    """The nearest preceding markdown heading, or ``""``."""
    for above in reversed(lines[:index]):
        if above.startswith("#"):
            return above
    return ""


def _is_baseline(lines: list[str], index: int) -> bool:
    """Is line ``index`` inside ``docs/STATE.md`` §Baseline?"""
    for above in reversed(lines[:index]):
        if above.startswith("#"):
            return above.startswith("## Baseline")


def unbound(lines: list[str]) -> list[int]:
    """Indices of lines carrying a suite count with NO binding anywhere.

    The heading is consulted for the binding, and a milestone row's own
    contribution count (``34 new tests`` — no RESULT word, so it never matches)
    needs no exemption.
    """
    fenced = _fenced(lines)
    quoted = _quoted_example_lines(lines)
    out = []
    for index, line in enumerate(lines):
        if not SUITE_COUNT.search(line):
            continue
        if index in fenced or index in quoted:
            continue
        window = "\n".join(lines[max(0, index - WINDOW):index + WINDOW + 1])
        if ISO_DATE.search(window) or COMMIT.search(window) or AS_OF.search(window):
            continue
        if ISO_DATE.search(_heading_before(lines, index)) \
                or COMMIT.search(_heading_before(lines, index)):
            continue
        out.append(index)
    return out


def _locally_unbound(lines: list[str]) -> list[int]:
    """§Baseline only: the same rule with the heading NOT consulted."""
    fenced = _fenced(lines)
    quoted = _quoted_example_lines(lines)
    out = []
    for index, line in enumerate(lines):
        if not _is_baseline(lines, index) or not SUITE_COUNT.search(line):
            continue
        if index in fenced or index in quoted:
            continue
        window = "\n".join(lines[max(0, index - WINDOW):index + WINDOW + 1])
        if ISO_DATE.search(window) or COMMIT.search(window) or AS_OF.search(window):
            continue
        out.append(index)
    return out


# ---------------------------------------------------------------------------
# 1. the rule, over every record
# ---------------------------------------------------------------------------
def test_no_record_states_an_undated_suite_count():
    """Every suite count in every record names the moment it was true.

    The scan runs over :func:`_record_sources` — the same scope the
    machine-facts gate uses, imported rather than re-listed, so the two rules
    can never drift apart on WHICH files they read.
    """
    offenders = []
    for path in _record_sources():
        for index in unbound(_lines(path)):
            offenders.append(
                f"{_label(path)}:{index + 1}: {_lines(path)[index].strip()[:90]}")
    assert offenders == [], (
        "these record lines state an exact suite count without saying when it "
        "was true (an ISO date, a commit sha, or an explicit as-of label must "
        "be on the line, within two lines either side, or in the nearest "
        "preceding heading): " + "; ".join(offenders)
        + ". Either bind it (this is a dated measurement — say the date or the "
          "commit), or drop the number and keep the command that produces the "
          "current one. See docs/STATE.md §Baseline for the worked example.")


# ---------------------------------------------------------------------------
# 2. ... and §Baseline, which is where the rot happened, holds it locally
# ---------------------------------------------------------------------------
def test_the_baseline_block_carries_no_inherited_binding():
    """§Baseline may not lean on its heading's date.

    The heading has carried a date since P0.17, so without this test a bare
    "the suite is N tests" dropped into §Baseline would pass by inheritance —
    which is the rot, reintroduced through the fix's own mechanism.
    """
    lines = _lines(STATE)
    offenders = [f"docs/STATE.md:{i + 1}: {lines[i].strip()[:90]}"
                 for i in _locally_unbound(lines)]
    assert offenders == [], (
        "docs/STATE.md §Baseline states a suite count that binds only through "
        "its heading: " + "; ".join(offenders)
        + ". §Baseline must bind every figure on its own line — put the date "
          "and the commit next to the number, or state the command instead.")


def test_the_baseline_names_the_command_not_only_a_number():
    """The replacement for a decorative number is the command that makes it.

    The rule is only useful if the record still answers the question a reader
    asked, so §Baseline must keep both pytest invocations AND the collection
    command: a reader who needs the current figure runs them.
    """
    text = "\n".join(_lines(STATE))
    start = text.index("## Baseline")
    end = text.index("\n## ", start)
    baseline = text[start:end]
    for command in ('-m pytest -q -m "not slow"',
                    "--collect-only",
                    "PYRADIOSS_BACKEND=numpy",
                    "PYRADIOSS_ORACLE_REQUIRED=1"):
        assert command in baseline, (
            f"docs/STATE.md §Baseline no longer carries {command!r}: the "
            "replacement for a bare count is the command that produces the "
            "current one, so it has to be there")


# ---------------------------------------------------------------------------
# 2b. ... and a per-module SIZE is a claim about NOW, so it needs a command
# ---------------------------------------------------------------------------
def test_a_module_size_figure_names_the_command_that_re_measures_it() -> None:
    """A figure with no command cannot be checked, and a date is not a check.

    The round-4 defect this rule exists for: ``UPDATES.md`` and
    ``plan/01_phase0_oracle_and_licensing.md`` both stated that
    ``tests/test_p0_no_stale_machine_paths.py`` holds ``**22**`` tests, bound to
    ``2026-10-04``, and the measured figure is **24**.  Every existing rule
    passed it — :data:`SUITE_COUNT` needs a pytest RESULT word, so the shape was
    invisible, and the date it carried was a perfectly good binding.  A binding
    answers "when was this true"; nothing but the command answers "is it still
    true".
    """
    offenders = []
    for path in _record_sources():
        lines = _lines(path)
        offenders += [
            f"{_label(path)}:{i + 1}: {lines[i].strip()[:90]}"
            for i in unreproducible_module_sizes(lines)]
    assert offenders == [], (
        "these record lines state how many tests a named test module has, with "
        "no command that re-measures it: " + "; ".join(offenders)
        + ". A module's size is a claim about NOW — it changes on the next "
          "added test — so the date beside it binds only the past. Quote the "
          "command (`.venv/bin/python -m pytest -q --collect-only <file>`), "
          "drop the number, or keep it explicitly as a milestone/contribution "
          "figure of a past tree.")

    # ... and the rule can still tell the four apart, or it is decoration:
    checked = [
        "`tests/test_p0_no_stale_machine_paths.py` (24 tests as of 2026-10-04; "
        "re-count with `.venv/bin/python -m pytest -q --collect-only "
        "tests/test_p0_no_stale_machine_paths.py`).",
        # a milestone row: the table's preamble binds it to that milestone
        "| M701 | Oracle records (`tests/test_p0_oracle_provenance.py`, 10 "
        "tests). |",
        # a task's own output, in the dated entry that records the task
        "- **New** `tests/test_p0_paths.py` (35 tests) — every resolver per tier.",
        # not a size of a named module: nothing to re-derive, so no command owed
        "- their 20 tests stay green and neither file was touched.",
    ]
    for line in checked:
        assert unreproducible_module_sizes([line]) == [], (
            f"{line!r} is not an unreproducible size claim")

    for line in ("`tests/test_p0_paths.py` 60 tests (was 35): every pair.",
                 "`tests/test_p0_oracle_provenance.py` (10 tests)."):
        assert len(unreproducible_module_sizes([line])) == 1, (
            f"{line!r} states a module's size with no command to re-check it "
            "and must be reported — this is the exact shape the binding rule "
            "could not see")


# ---------------------------------------------------------------------------
# 3. the check can still fail
# ---------------------------------------------------------------------------
def test_the_binding_rule_can_still_fail():
    """A checker that accepts everything is worse than none; drive it.

    Six shapes, and the answers must differ: a dated count, a commit-bound
    count, a heading-bound count (the dated-log-entry case), a bare count (the
    rot), a language-tagged code sample (not a claim), and a count in
    ``§Baseline`` that only its heading binds (the rot's own hiding place).
    """
    dated = ["- Measured 2026-10-04: `14594 passed, 15 skipped`."]
    assert unbound(dated) == [], "a dated figure is bound"

    committed = ["- The fast tier measured `14594 passed`.",
                 "  All at commit `13deef2`."]
    assert unbound(committed) == [], "a commit-bound figure is bound"

    logged = ["## 1.7.0 - Fix wave 2 (2026-10-04)",
              "",
              "- the oracle modules collect **71** and measure `71 passed, 0",
              "  skipped` both ways"]
    assert unbound(logged) == [], (
        "a dated log entry binds its own measurements: that is the whole point "
        "of dating the entry")

    bare = ["- The fast tier measures `14594 passed, 15 skipped` on this box."]
    assert len(unbound(bare)) == 1, (
        "an undated bare count is the rot this file exists for and must be "
        "reported")

    labelled = ["- At the time: `14594 passed, 15 skipped`."]
    assert unbound(labelled) == [], (
        "a record may label history in prose instead of dating it")

    code = ["The ledger API a future task must write::", "",
            "```python", "base = Ledger(passed=10, failed=(), skipped=0)", "```"]
    assert SUITE_COUNT.search(code[3]), "the sample really does match the rule"
    assert unbound(code) == [], (
        "a language-tagged code block is a specimen, not a claim about a suite")

    fence_no_lang = ["Measured facts::", "", "```", "14594 passed", "```"]
    assert len(unbound(fence_no_lang)) == 1, (
        "an UNTAGGED fence gets no free pass here: quoting is not dating, and "
        "the sibling gate holds that shape to its stricter attributed-opener "
        "rule. If this ever becomes 0, _fenced grew too permissive")


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _label(path) -> str:
    from tests.test_p0_no_stale_machine_paths import REPO

    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.name
