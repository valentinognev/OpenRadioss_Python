"""Task P0.15 — the repository must not assert machine facts that are not true here.

This repo was migrated off a Windows/conda machine. Files kept asserting
things about THAT machine — most of all ``tools/validate_vs_fortran.py``'s
claim, twice, that the oracle binaries carry
``DT_RPATH=/home/valentin/anaconda/lib``.  The rebuilt binaries carry **no**
``DT_RPATH`` and no ``DT_RUNPATH`` at all (``readelf -d``, measured), so the
claim was false in the one file every parity verdict is read out of.  A second
false record sat in ``pyproject.toml``'s ``accel`` comment (numpy 2.5.2 against
the lock's 2.5.3), and a third in ``tools/oracle/build_oracle.sh`` (a
``command -v cmake`` version that no longer resolves that way).

In this project a false recorded fact is a defect, not a typo: every parity and
reproducibility claim downstream inherits its authority (AGENTS.md; plan
§1.1).  So the rule enforced here is the one that generalises:

    **A machine-specific path or version may appear in the tooling/packaging
    surface only if it is true on the box reading it, or is explicitly marked as
    a statement about another machine.**

Two halves, because both are needed and neither is enough:

* existence (``test_a_machine_path_this_box_does_not_have_is_marked_history``)
  — a ``/home/...`` path that does not exist here must be labelled history.
  A path that DOES exist is not thereby correctly described, which is why the
  second half exists.
* measurement (``test_the_harness_records_the_rpath_the_binaries_actually_carry``,
  ``test_the_build_script_names_the_tool_versions_this_box_has``,
  ``test_packaging_version_claims_agree_with_the_recorded_pins``) — where a
  file claims a *measured* fact about the oracle binaries, the build
  toolchain, or the installed distributions, the claim is compared against the
  thing itself (``readelf`` via the harness's own ELF parser, ``--version``
  banners, ``requirements-lock.txt``).

Deliberately NOT "forbid the string ``anaconda``": the pre-migration record is
legitimate provenance and must stay readable —
``tools/validation_data/oracle_provenance.json`` keeps naming
``/home/valentin/anaconda`` precisely because it records that prefix as
**absent**, and tests/test_p0_oracle_provenance.py asserts that against the
disk.  A test that forbade the word would force that provenance to be deleted
or lied about.

And deliberately not "forbid the paragraph": the RPATH hazard is real and must
stay explained — glibc searches ``DT_RPATH`` before ``LD_LIBRARY_PATH``, so a
writer reachable that way would be dlopen'd by ``h3d_dl.c:660-666``.  Deleting
the paragraph would make this file green and the documentation worse, so
:func:`test_the_rpath_hazard_is_still_explained_and_stated_as_measured` requires
both the reasoning tokens AND an explicit statement of what the binaries carry
today — which is the sentence that has to agree with ``readelf``.

And the rule needs one thing spelled out, because getting it wrong is how a
rot scanner dies.  A line of text can be a SENTENCE about this machine or a
SPECIMEN of what some program printed, and only the second may quote a fact that
is false here — so :func:`_quoted_example_lines` distinguishes them, and it needs
**two** conditions, not one.  Fix round 1 exempted any indented run under a ``::``
on structure alone, and the review of it planted a false path and a false version
under exactly such a block and watched every rule skip every line.  A block is a
specimen only if its opener *attributes* it (:data:`SPECIMEN_OPENERS`) — and even
then a machine PATH gets no exemption at all (:func:`_path_claims`), because a
path is a location and quoting cannot make one exist.
:func:`test_a_stale_claim_written_inside_a_literal_block_is_still_a_claim` pins
both, and the honest ``_gcc_version`` specimen stays ignored.

Three self-tests at the end (:func:`test_the_history_rule_can_still_fail`,
:func:`test_the_version_rule_can_still_fail`,
:func:`test_a_stale_claim_written_inside_a_literal_block_is_still_a_claim`) drive
both checkers over synthetic text: a checker that accepts everything is worse
than no checker, and these are the ones that would rot into that.

WIDENED TO THE RECORDS (round 3), because a gate that cannot see the records
cannot see the files a future agent reads FIRST.  Rounds 1 and 2 scanned
``tools/**`` plus ``pyproject.toml`` and the lock, which is where the four false
claims of this task actually lived — but ``docs/STATE.md`` §Baseline is the
onboarding document, ``UPDATES.md`` is the mandatory change log, and
``plan/`` is the roadmap: a claim in any of them is the claim that gets quoted.
The Phase 0 whole-branch review measured the consequence (STATE.md carrying four
fields no rule could reach) and this round is the fix, so the scope is now the
tooling surface **plus every record file** (:func:`_record_sources`).

Widening is only legitimate if the records stop being a wall of noise, and they
nearly were: the discriminators below grew exactly as far as the records forced
them to, each for one named category —

* **a link target is an address, not a location** (:func:`_in_uri_target`).  The
  dated bug report carries 18 ``C:/Users/pmqua/…`` permalinks (on 12 lines) into
  the pre-migration checkout, and ``Path("C:/…").exists()`` measures nothing:
  there is no such drive here and there never will be.  Only the ``](… )`` span
  is exempt — the link TEXT stays prose and stays checked.
* **a Unicode comparison operator is a requirement, not a measurement**
  (:data:`_CONSTRAINT_TAIL`): ``cmake ≥ 3.15`` says what the build needs, which
  is the same sentence as the ASCII ``cmake >= 3.15`` the rule already skips.
* **the old side of a migration arrow is superseded by construction**
  (``python 3.14.6→3.12.3``, :func:`_version_claim_offenders`); the new side is
  still compared against the lock.
* **a version belonging to the OTHER supported machine is legitimate whenever the
  lock records it** (:func:`_recorded_pins`).  §[A] is the maintainer's Windows
  box and its pins are part of the record by design, which is what lets
  ``AGENTS.md``'s ``numpy 2.4.6`` stand while ``numpy 2.5.2`` does not.  §[A] is
  DESCRIPTIVE — no rule compares it to the running interpreter — and this rule
  does not either: it compares prose to the RECORD.
* **``$OR_SRC`` / ``$OR_ROOT`` / ``OR_BUILD`` placeholders** in documented
  commands were already outside :data:`MACHINE_PATH` (the path root follows the
  variable name, so there is nothing ``/home``-shaped to match).  That was luck
  with a regex's good manners, so it is now pinned by
  :func:`test_a_shell_variable_placeholder_is_not_a_machine_path` instead of
  being left to a future edit of the lookbehind.

ROUND 4, after the review of round 3, which found the gate green here and red
in CI on ``ubuntu-latest``.  Three corrections, each of which is a rule this
file now states rather than a rule it merely has:

* **a record may not name a path out of THIS box's ``$HOME``**
  (:func:`test_no_record_may_name_a_path_out_of_this_machines_home_directory`).
  Nineteen record lines did, so the whole gate was a snapshot of one checkout.
  The fix is in the records — ``$OR_SRC`` / ``$OR_ROOT`` / ``$OR_BUILD`` /
  ``$HOME/…``, i.e. the indirection ``plan/00_ORCHESTRATION.md`` §4.1 already
  documents — and the test re-points ``$HOME`` and masks the disk answer under
  this box's home, so the same scan runs as if the reader were somebody else.
  Two properties survive it, both asserted: a placeholder is accepted on every
  box, and a bare absolute path under a ``$HOME`` that is not this one is still
  caught.
* **the link-target exemption is a SPAN, not the rest of the line**
  (:func:`_in_uri_target`).  It found the first ``](`` to the left of a match
  and treated everything after it as the target, so
  ``see [x](C:/a/b) and … /home/valentin/anaconda/lib`` reported ``[]`` — the
  exact opposite of what the helper's own docstring promised.  The target must
  now close before the match ends, which is why ``end`` is a required argument.
* **a requirement FLOOR is not a measurement** (:func:`_is_floor`).
  ``numpy>=1.22`` was already skipped as an operator-prefixed requirement, but
  the prose spelling — ``requires numpy 1.21 at minimum``, ``numpy 1.21+``,
  ``cmake 3.15 or newer`` — fell through to the pin comparison and to
  ``<tool> --version``.  A floor is satisfied by every value the lock could
  hold.  The window is one CLAUSE, so a floor word elsewhere in the sentence
  cannot exempt a real measurement.

ROUND 5, after the review of round 4, which fixed nineteen record lines and
still left three hits — all of them in ``tools/oracle/``, all of them true here
and false on ``ubuntu-latest``: a **hardcoded ``$OR_SRC`` default in code** in
``toolchain_probe.py`` (the real defect: on a box without the variable the probe
resolved to nothing and recorded ``extlib_url_reachable: false``), a docstring,
and a build-script aside.  They survived three rounds for one structural reason,
and :func:`test_the_foreign_home_scan_covers_the_whole_scanned_scope` is the
fix: **the portability simulation read the RECORDS only**, so ``tools/`` was
never examined as if it were somebody else's machine.  The same scan now runs
over :func:`_scanned_sources` — the whole gate scope — and a planted
non-vacuity test
(:func:`test_a_machine_path_planted_in_a_tooling_file_is_caught_by_the_simulation`)
proves it fires on the exact shape it exists for.

What is deliberately STILL out of reach, stated here so nobody mistakes silence
for green: **backslash Windows paths** (``C:\\OpenRadioss\\exec``,
``.venv\\Scripts\\python.exe``, ``C:\\Users\\pmqua\\…``).  :data:`MACHINE_PATH`
is ``$HOME``-shaped, the Windows box this project still supports is not this
one, and a rule that declared those false would be inventing a measurement
nobody can take — task P1.0 owns that rewrite.  ``AGENTS.md`` itself **is** in
scope for everything this gate can measure (that was measured: zero hits, so it
is free), which means a Linux-shaped false claim introduced there is caught from
now on.  ``plan/README.md`` §8 is not in that category — see the hits this
round surfaced for its owner.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
DOCS = REPO / "docs"
PLAN = REPO / "plan"
PYPROJECT = REPO / "pyproject.toml"
LOCK = REPO / "requirements-lock.txt"
AGENTS = REPO / "AGENTS.md"
README = REPO / "README.md"
UPDATES = REPO / "UPDATES.md"
STATE = DOCS / "STATE.md"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
#: Suffixes whose prose the scan reads.  ``.md`` joined the three code/script
#: suffixes in round 3 — the records are markdown, and a scope that lists
#: ``docs/`` but not ``.md`` is the same unnamed scope this function's docstring
#: is written to prevent.
SCANNED_SUFFIXES = (".py", ".sh", ".txt", ".md")

#: The root records a future agent reads before anything else, plus the two
#: directories of them.  ``AGENTS.md`` is deliberately in: it is the most
#: read-first file in the repo, it is known to be Windows-shaped (task P1.0
#: owns the rewrite), and including it was measured to cost **zero** hits —
#: every path in it is backslash-shaped, which :data:`MACHINE_PATH` cannot see.
#: So it buys coverage for any Linux-shaped claim added there, and costs
#: nothing today.
RECORD_FILES = (README, UPDATES, AGENTS, STATE)
RECORD_GLOBS = ("docs/**/*.md", "plan/**/*.md")


def _record_sources() -> list[Path]:
    """Every record file, in a stable order, that exists.

    The GLOB is deliberate rather than a list of filenames: ``docs/`` and
    ``plan/`` grow, and a record added tomorrow must be scanned without anyone
    remembering to add it here.  :func:`test_the_records_a_future_agent_reads
    _first_are_in_scope` fails if a file present on disk is missing from this
    list, which is the regression that undoes the widening silently.  Order is
    stable and duplicates are dropped (``STATE.md`` is named above AND matched
    by the glob, and a record scanned twice is a finding reported twice).
    """
    files = list(RECORD_FILES)
    for pattern in RECORD_GLOBS:
        files.extend(sorted(REPO.glob(pattern)))
    seen: set[Path] = set()
    return [p for p in files if p.is_file() and not (p in seen or seen.add(p))]


def _scanned_sources() -> list[Path]:
    """The maintained files held to the rule, in a stable order.

    IN: the parity harness, the oracle build/runtime scripts, the two packaging
    records at the repo root, and — since round 3 — **the records themselves**
    (:func:`_record_sources`): ``docs/**``, ``plan/**``, ``README.md``,
    ``UPDATES.md`` and ``AGENTS.md``.  Rounds 1–2 scoped ``tools/**`` alone,
    which is where this task's four false claims lived but NOT where a reader
    starts: the phase review measured ``docs/STATE.md`` §Baseline carrying four
    fields nothing could reach, precisely because the gate never opened the
    onboarding document.

    OUT, each for a stated reason — an unnamed scope is how a rule stops being
    enforced:

    * ``tools/validation_data/*.json`` — machine-of-record DATA, harvested from
      whatever box produced it.  A record of a foreign machine is legitimate
      there and is already held to this disk by
      ``tests/test_p0_oracle_provenance.py``.
    * ``tests/`` — RE-MEASURED 2026-10-03 (P0.15 fix round 2) by running both
      live rules over ``sorted(REPO.glob("tests/**/*.py"))`` (1042 modules at
      that measurement; reproduce with the sweep quoted in the round-2 report).
      The existence rule reports **13 hits on 11 lines** and the
      toolchain-version rule **20 hits on 20 lines**, and **none of them is a
      stale claim**: 4 of the existence hits (2 lines) are the ``/home/someone``
      / ``/Users/someone`` placeholders in detector regexes — this file's and
      ``test_p0_harness_portable.py:187``'s — honest placeholders for what a
      pattern matches; the other 9 are ``/home/valentin/anaconda…`` named as the
      pre-migration record (this file's module docstring, its self-test
      fixture, and ``test_p0_toolchain.py``'s quotations of the old record),
      every one of them inside a :data:`HISTORY_MARKERS` window.  The 20
      version-rule hits are 15 in this file's own fixtures and 5 in
      ``test_p0_toolchain.py``'s quotations.  So widening today means flagging
      honest placeholders and honest history in files this task does not own —
      a gate that cries wolf.  The exclusion therefore STAYS; widening it is a
      separate decision for the owners of those two modules (add the glob here,
      and exempt this file as the scanner, which by construction quotes what it
      hunts).
    * ``/opt`` and ``/usr`` roots.  ``/opt/OpenRadioss`` is upstream's own
      documented prefix and ``/usr/bin/cmake`` is the system toolchain; neither
      is a per-machine fact.  A per-machine fact in this repo is ``$HOME``-shaped
      (or a ``/mnt/...`` bind), which is what :data:`MACHINE_PATH` matches.
    * backslash Windows paths, wherever they appear (``AGENTS.md``,
      ``README.md``'s documented fallback) — see the module docstring: this box
      cannot measure them and task P1.0 owns them.
    """
    scripts = sorted(
        p for p in TOOLS.rglob("*")
        if p.is_file() and p.suffix in (".py", ".sh", ".txt")
    )
    return [PYPROJECT, LOCK, *_record_sources(), *scripts]


#: An absolute path rooted in a per-machine location.  ``$HOME``-shaped on both
#: platforms, plus the ``/mnt`` bind this box's checkout lives under.
MACHINE_PATH = re.compile(
    r"(?<![\w.$~-])(/(?:home|mnt|media|Users)/[A-Za-z0-9._+@-]+"
    r"(?:/[A-Za-z0-9._+@%~-]+)*)"
)

#: ``scheme:`` — the first token of a URI (``C:``, ``file:``, ``https:``).
_URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:")

#: What closes a link target: the ``)`` of ``](…)``, or the ``>`` of ``](<…>)``.
_LINK_CLOSE = re.compile(r"[)>]")


def _in_uri_target(line: str, start: int, end: int) -> bool:
    """Is the span ``line[start:end]`` inside a markdown/HTML link TARGET?

    ``[starter_keywords.py:14847](C:/Users/pmqua/…/starter_keywords.py:14847)``
    — the span after ``](`` up to the closing ``)``.  Such a span is an address
    in a link namespace: an editor permalink into whichever checkout the report
    was written against.  ``Path("C:/Users/pmqua/…").exists()`` is not a
    measurement of anything (there is no ``C:`` here and there never will be),
    so reporting it would be the rule asserting something it cannot know.

    Added in round 3 for the records: the dated bug report carries 18 such
    permalinks on 12 lines and every one of them is a locator into the
    pre-migration checkout.  Four properties keep this from being a laundering
    route, and the round-3 review found the first version missing three of them:

    * only the TARGET is exempt — the link text, where a claim would actually be
      written, is outside the span and still read by both rules;
    * a scheme is REQUIRED, so a bare relative target (``](tools/foo.py)``) and
      an ordinary absolute path in prose are unaffected;
    * **the target must be CLOSED before the match ends** (``end``).  This is
      the reviewer's case, verbatim::

          see [x](C:/a/b) and … /home/valentin/anaconda/lib      ->  []

      The old code took the *first* ``](`` to the left of a match and treated
      the whole remainder of the line as the target, so one URI-ish token
      anywhere earlier on the line exempted every later path — which
      contradicts the very docstring above ("up to the closing ``)``").  The
      closing delimiter is therefore located by search from ``](`` and the match
      must end at or before it;
    * an unterminated ``](`` is not a link at all, so a target that never closes
      buys nothing.

    ``end`` is required, not optional, for the same reason: given only ``start``
    there is no way to ask the third property.  :func:`_path_claims` passes the
    real ``match.end()``; :func:`test_the_uri_exemption_is_a_span_not_the_rest_of
    _the_line` pins the signature so "optional again" fails with a name.
    """
    head = line[:start]
    open_at = head.rfind("](")
    if open_at < 0:
        return False
    target = head[open_at + 2:]
    if "(" in target:          # a further, unclosed target began before this
        return False           # match — the span we found is not the real one
    if not _URI_SCHEME.match(target.lstrip("<")):
        return False
    close = _LINK_CLOSE.search(line, open_at + 2)
    return close is not None and end <= close.start()


#: Phrases that mark a statement as being about ANOTHER machine (or another
#: time), which is what makes naming a path that is not here legitimate.  A
#: provenance record says so; a stale claim, however confident, does not.
#:
#: The vocabulary is deliberately NOT extended with the correction register
#: ("corrected", "false", "withdrawn").  It would buy four fewer reported lines
#: in the records and cost the rule its teeth: "the corrected prefix is
#: /home/…" would launder any claim on earth, because the label only has to be
#: in the same 3-line window.  An honest record that quotes a removed claim
#: already has a word this list accepts — ``pre-migration``, ``previous record``,
#: ``no longer`` — and a report that names the exact line is cheaper than a
#: vocabulary that cannot be trusted.  (The four lines that cost this decision
#: are listed in the round-3 report as findings for their owners.)
HISTORY_MARKERS = (
    "pre-migration", "pre migration", "previous record", "prior record",
    "old record", "superseded", "no longer", "not exist", "no such file",
    "recorded as absent", "is absent", "used to", "is gone",
    "foreign machine", "another machine", "other box", "history",
    "historically", "machine_of_record", "the box that", "supersede",
)


#: How many lines after a claim the "this is another machine's record" label
#: may sit.  A claim and its label are the same sentence in every honest record
#: in this repo (checked: the pre-migration provenance in
#: ``tools/validation_data/oracle_provenance.json`` and the corrected cmake note
#: in ``tools/oracle/build_oracle.sh``), and a wider window re-opens the hole
#: this number closes: one historical aside anywhere in a paragraph would
#: exempt every other claim in it, which is how a rule stops being one.
LABEL_WINDOW_LINES = 2


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


#: Words that make a block's OPENER an attribution — it says the lines below
#: are text quoted FROM something (a producer's output, an illustration of a
#: format) rather than facts of their own.  Substring matching, lower-cased.
#:
#: The bare structural discriminator ("a ``::`` line opens a literal block") is
#: a loophole, and the review of fix round 1 proved it: the four lines
#:
#:     Measured facts::
#:
#:             cmake 4.4.3
#:             DT_RPATH=/home/valentin/anaconda/lib
#:
#: contain a false path AND a false version, and every rule skipped every one of
#: them.  Structure alone cannot tell a specimen from a claim, because both are
#: indented text under a ``::``.
#:
#: Attribution is the semantic half, and it is what an *honest* specimen opener
#: already says without being asked: ``_gcc_version``'s docstring in
#: ``tools/oracle/toolchain_probe.py`` writes "The banner shapes, one row per
#: distro, look like this::" — that sentence is what makes the banner table
#: below it a specimen rather than an assertion.  An opener that asserts
#: ("Measured facts::") attributes the block to nobody and to nothing, so it
#: gets no exemption.
#:
#: The polarity matters and is stated here because it is easy to invert: an
#: attributing opener ⇒ quotation ⇒ EXEMPT.  (Inverting it would exempt nothing
#: and flag the Red Hat banner, which is exactly the false positive this whole
#: mechanism exists to remove.)
SPECIMEN_OPENERS = (
    "print", "output", "banner", "sample", "specimen", "usage", "invocation",
    "example", "e.g.", "i.e.", "for instance", "such as", "looks like",
    "looks as", "resemble", "shape of", "form of", "quot", "parse", "parser",
    "reads", "extract", "template", "format", "verbatim", "like this",
)

#: What a CAPTURED banner looks like, as opposed to a sentence naming a
#: version: a parenthesised packaging (``(Red Hat 4.8.5-44)``), the word
#: ``version`` (``cmake version 3.28.3``), or a build date (``20150623``).
#: Needed because a specimen is *allowed* to be false here — the Red Hat
#: banner is not this box's compiler — so for tool versions the discriminator
#: cannot be "is it true", only "is it output".
_OUTPUT_SHAPE = (
    re.compile(r"\([^)]*\d[^)]*\)"),
    re.compile(r"\bversion\b", re.I),
    re.compile(r"\b\d{6,8}\b"),
)


def _attributes_a_specimen(opener: str) -> bool:
    """Does this opener say the block below is text QUOTED from something?"""
    text = opener.lower()
    return any(marker in text for marker in SPECIMEN_OPENERS)


def _looks_like_captured_output(line: str) -> bool:
    """Does ``line`` have the shape of a program's printed output?"""
    return any(pattern.search(line) for pattern in _OUTPUT_SHAPE)


def _fence_opener(lines: list[str], index: int) -> str:
    """The prose a fenced block was opened under: its info string, else the
    nearest non-blank line above it.  A bare ``````` carries no attribution of
    its own, so the attribution has to come from the sentence that introduced
    it."""
    parts = []
    info = lines[index].strip()[3:]
    if info:
        parts.append(info)
    for above in reversed(lines[:index]):
        stripped = above.strip()
        if not stripped or stripped.startswith("```"):
            continue
        parts.append(stripped)
        break
    return " ".join(parts)


def _quoted_example_lines(lines: list[str]) -> set[int]:
    """0-based indices of SPECIMEN lines — a quotation, as opposed to prose.

    Two conditions, both required, and the second is the one round 1 was
    missing:

    * **structure** — a reST literal block (a line ending in ``::`` followed by
      lines indented deeper than the opener) or a fenced block; and
    * **attribution** — the opener says whose text the block quotes
      (:data:`SPECIMEN_OPENERS`).

    In this repo the shape that satisfies both is where a *sample of what a
    tool prints* lives, and a sample is not a claim about anything:
    ``_gcc_version``'s docstring in ``tools/oracle/toolchain_probe.py`` opens
    "The banner shapes, one row per distro, look like this::" and then lists the
    banners it parses, including this box's own::

        GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0  Debian/Ubuntu
        gfortran (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)     Red Hat

    The first row is this box's compiler and the second is a Red Hat banner
    quoted from nowhere near it.  Without this distinction the version rule
    flagged the examples as stale facts about this box, which is the classic
    failure mode of a rot scanner: it cannot tell a sentence from a specimen, and
    it cries wolf until the specimen is deleted.  (The row count and the exact
    sentences there are the owning file's business and have moved more than
    once; the self-test finds the block by content, not by line number.)

    Structure alone was a loophole (see :data:`SPECIMEN_OPENERS`), and it is
    worth being explicit about what the remaining guarantee is: a false claim
    can hide here only if BOTH the opener attributes the block to a producer
    AND the line itself is shaped like that producer's output.  That is not
    airtight — a liar who writes "output of ``cmake --version``::" over a
    false banner gets through — and it cannot be, because the honest Red Hat
    specimen and the dishonest block are the same sentence up to the words.
    What the rule does buy is that laundering a claim costs *deliberate
    framing*, and the plain "Measured facts::" block now fails loudly.

    The region ends at the first non-blank line indented no deeper than the
    opener, the opener line itself always stays PROSE, and membership still
    requires blank-or-deeper indentation.  So an ordinary sentence — a version
    written in a comment or a docstring line — is never exempt, including one
    written directly below an example block.
    """
    quoted: set[int] = set()
    open_indent: int | None = None
    fence: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if fence is not None:
            quoted.add(index)
            if stripped.startswith(fence):
                fence = None
            continue
        if open_indent is not None:
            # Tested before the fence branch: a reST literal block is literal,
            # so a ``` inside one is part of the quoted text, not a new block.
            if not stripped or indent > open_indent:
                quoted.add(index)
                continue
            open_indent = None
        if stripped.startswith("```") and _attributes_a_specimen(
                _fence_opener(lines, index)):
            fence = stripped[:3]
            quoted.add(index)
            continue
        if stripped.endswith("::") and _attributes_a_specimen(stripped):
            open_indent = indent
    return quoted


def _label_window(lines: list[str], index: int) -> str:
    """The claim's own line plus :data:`LABEL_WINDOW_LINES` lines below it."""
    return "\n".join(lines[index:index + 1 + LABEL_WINDOW_LINES]).lower()


def _marked_history(window: str) -> bool:
    return any(marker in window for marker in HISTORY_MARKERS)


def _path_claims(lines: list[str]) -> list[tuple[int, str, str]]:
    """``(index, path, label_window)`` for every machine path on every line.

    One entry per (line, path); the window travels with it so the caller never
    has to rediscover which lines a claim belongs to.

    Specimen lines are NOT skipped here, and that is deliberate and asymmetric
    with the version rule.  A machine path is a **location**, never a format:
    there is no reading under which ``DT_RPATH=/home/valentin/anaconda/lib``
    inside an indented block illustrates anything.  It is either true here, or
    it is a claim about a machine — and if it is about another machine, the
    block says so with a :data:`HISTORY_MARKERS` label, which is what the label
    window is for.  So quoting buys a path nothing at all, and the exemption
    cannot be spent on laundering: the reviewer's planted block is caught by
    this rule whether or not it is indented under a ``::``.

    The one shape skipped here is a path inside a link TARGET
    (:func:`_in_uri_target`), which is an address rather than a location; the
    link text beside it is prose and is read.
    """
    claims = []
    for index, line in enumerate(lines):
        for match in MACHINE_PATH.finditer(line):
            if _in_uri_target(line, match.start(), match.end()):
                continue      # an address in a link namespace (round 3)
            path = match.group(1).rstrip(".,;:)'\"")
            claims.append((index, path, _label_window(lines, index)))
    return claims


def _stale_paths_in(lines: list[str]) -> list[tuple[int, str]]:
    """``(line index, path)`` for the paths that are false here and unmarked."""
    return [(index, path) for index, path, window in _path_claims(lines)
            if not Path(path).exists() and not _marked_history(window)]


def _existence_offenders(paths: list[Path]) -> list[str]:
    """``file:line: path`` for every path claim that is false here and unmarked.

    The scan the first live test performs, extracted so a regression test can
    drive the SAME scan over a scratch copy of a record file (outside the repo)
    and prove the rule catches what is planted in it.  A scope regression is
    otherwise invisible: narrowing the scope does not raise, it silences.
    """
    offenders = []
    for path in paths:
        for index, stale in _stale_paths_in(_lines(path)):
            label = (path.relative_to(REPO).as_posix()
                     if path.is_relative_to(REPO) else path.name)
            offenders.append(f"{label}:{index + 1}: {stale}")
    return offenders


# ---------------------------------------------------------------------------
# 1. a path this box does not have must be marked as another machine's
# ---------------------------------------------------------------------------
def test_a_machine_path_this_box_does_not_have_is_marked_history():
    """The generic half of the rule, over :func:`_scanned_sources`.

    Catches ``validate_vs_fortran.py``'s two ``/home/valentin/anaconda/lib``
    claims — a directory that does not exist here, asserted as a measurement
    of this box's oracle.  A block that names a path which DOES exist passes
    regardless of whether the sentence about it is accurate; that is the
    measured half's job, and pretending otherwise is why this rule is not the
    only one.

    Since round 3 the scan covers the RECORDS (:func:`_record_sources`) as well
    as the tooling, so this is also the rule that reads ``docs/STATE.md``,
    ``UPDATES.md``, ``plan/**`` and ``README.md``.  The hits it reports there
    are findings for those files' owners, not defects in the rule: the fix is a
    label (``pre-migration``, ``previous record``, ``no longer``) or a path that
    exists here.
    """
    offenders = _existence_offenders(_scanned_sources())
    assert offenders == [], (
        "these lines assert a machine path that does not exist on this box, "
        "with no label within "
        f"{LABEL_WINDOW_LINES} lines saying the statement describes another "
        "machine or another record (accepted labels: "
        f"{', '.join(HISTORY_MARKERS[:6])}...): " + "; ".join(offenders)
        + ". Either the path exists here, or the sentence says which machine / "
        "which record it describes.")


# ---------------------------------------------------------------------------
# 1b. ... and the same rule on a box whose $HOME is not this box's
# ---------------------------------------------------------------------------

#: This box's ``$HOME``, measured at import time (before anything patches it).
#: ``Path(__file__).resolve()`` puts the repository itself under ``/mnt/...`` on
#: this checkout, so masking ``$HOME`` cannot hide the repo's own files.
THIS_BOX_HOME = str(Path(os.path.expanduser("~")))

#: The ``$HOME`` the simulation pretends this box has.
FOREIGN_HOME = "/home/some-other-user"


@contextlib.contextmanager
def _a_box_whose_home_is_not_this_one():
    """Pretend the reader's ``$HOME`` is somebody else's, for the scan only.

    What a differently-homed box actually differs in is the DISK under its home
    directory: nothing under ``/home/<that user>/`` exists on it.  So that is
    exactly what this masks — ``Path.exists`` answers "no" for any path under
    :data:`THIS_BOX_HOME` and defers to the real filesystem for everything
    else (``/usr/bin/cmake``, ``/opt/OpenRadioss``, the repository under
    ``/mnt/...``).  ``$HOME`` itself is re-pointed so anything that resolves a
    ``~`` sees the other box.

    Nothing in the working tree is touched and no copy of the repo is made: the
    records are READ, and only the answer to "is this path here" changes, for
    the duration of one ``with`` block.
    """
    real_exists = Path.exists
    real_home = os.environ.get("HOME")
    prefix = THIS_BOX_HOME + os.sep

    def foreign_exists(self, *args, **kwargs):
        if str(self).startswith(prefix):
            return False
        return real_exists(self, *args, **kwargs)

    Path.exists = foreign_exists
    os.environ["HOME"] = FOREIGN_HOME
    try:
        yield
    finally:
        Path.exists = real_exists
        if real_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = real_home


def test_no_record_may_name_a_path_out_of_this_machines_home_directory():
    """The records must read true on a box that is not this one.

    Round 3 shipped a gate that was green here and **red in CI**
    (``.github/workflows/ci.yml`` runs the fast tier on ``ubuntu-latest``, whose
    ``$HOME`` is ``/home/runner``): 19 record lines named this box's absolute
    ``/home/valentin/...`` paths, so a reader on any other machine would be told
    about a directory that is not there.  A gate whose verdict is true of one
    checkout is not a gate — it is a snapshot.

    The fix is in the records, not in the rule: a record that says "the mirror
    is ``$OR_BUILD``" (or ``$HOME/…``, or ``~``) is true everywhere, while
    "``/home/valentin/OpenRadioss_build``" is true on exactly one machine.  This
    test is what keeps it fixed, and it runs the SAME
    :func:`_existence_offenders` over the SAME :func:`_record_sources` the live
    test uses — only the disk answers differently.

    Two properties had to survive, and both are asserted here so the fix cannot
    be "make the rule stop looking":

    * **a ``$HOME``-relative or ``$OR_*``-shaped path is not a claim.**  It is
      a placeholder with nothing to stat, so it is accepted on every box.  It
      was already outside :data:`MACHINE_PATH`
      (:func:`test_a_shell_variable_placeholder_is_not_a_machine_path` pins the
      regex), and it stays outside it — this test is the reason that has to be
      true rather than lucky.
    * **a bare absolute path under a ``$HOME`` that is not this box's is still
      caught.**  ``/home/some-other-user/OpenRadioss_build`` is reported even
      though ``$HOME`` is *set*, because the rule's job is to catch a claim of a
      location that does not exist, and setting a variable is not evidence that
      the directory is there.
    """
    with _a_box_whose_home_is_not_this_one():
        offenders = _existence_offenders(_record_sources())
    assert offenders == [], (
        "these record lines name a path inside THIS box's $HOME ("
        f"{THIS_BOX_HOME}), so they are false on every other machine — and "
        f"the gate is red in CI for that reason, not only here. Offenders: "
        + "; ".join(offenders)
        + ". Use the project's own indirection instead: $OR_SRC / $OR_ROOT / "
          "$OR_BUILD / $PYRADIOSS_HM_CFG (plan/00_ORCHESTRATION.md §4.1), or "
          "$HOME/<name> where a home-relative directory is meant.")

    with _a_box_whose_home_is_not_this_one():
        # property 1: the documented indirection is accepted
        assert _path_claims([
            "the mirror is $OR_BUILD; the install prefix is $OR_ROOT;",
            "the checkout is $HOME/OpenCourant and the venv is ~/wt-p0/.venv",
        ]) == [], "a shell variable or a ~ is a placeholder, not a location"
        assert _stale_paths_in([
            "the mirror is $OR_BUILD and the tree is $HOME/OpenCourant"]) == [], (
            "and therefore nothing to report about it")

        # property 2: a bare absolute path under a foreign $HOME is a claim
        foreign = f"{FOREIGN_HOME}/OpenRadioss_build"
        assert _stale_paths_in([f"the mirror is at {foreign}"]) == [(0, foreign)], (
            "a bare absolute path is a location claim and is still caught when "
            "it does not exist — setting $HOME does not make it true")


# ---------------------------------------------------------------------------
# 1c. ... and the same simulation over the WHOLE scanned scope, not the records
# ---------------------------------------------------------------------------
def test_the_foreign_home_scan_covers_the_whole_scanned_scope():
    """The portability simulation must read what the live gate reads.

    :func:`test_no_record_may_name_a_path_out_of_this_machines_home_directory`
    simulates a foreign ``$HOME`` over :func:`_record_sources` — the RECORDS.
    That was the right scope when the nineteen hits were all in records, but the
    rule's job is bigger than that: the round-4 hits this box could not see were
    in ``tools/oracle/`` (a hardcoded ``$OR_SRC`` **default in code**, a docstring
    and a build-script aside), and they survived three rounds precisely because
    the simulated scan never looked at ``tools/``.  A portability proof that is
    narrower than the thing it proves is a snapshot with extra steps.

    So the same scan runs over :func:`_scanned_sources` — records, tooling and
    the two packaging records — under the same simulated home, and must report
    nothing.  Two properties are asserted so the fix cannot be "narrow the scan
    again":

    * the simulated file set IS the live gate's file set (same helper, so the
      two cannot drift apart by construction);
    * a bare absolute path under this box's ``$HOME`` in ANY scanned file — the
      shape the three round-4 hits had — is reported, driven through the real
      scan over a scratch copy outside the repository.
    """
    with _a_box_whose_home_is_not_this_one():
        simulated = _existence_offenders(_scanned_sources())
    assert simulated == [], (
        "these lines are true on this box and false on every other one, so the "
        "gate is red in CI for exactly this reason: " + "; ".join(simulated)
        + ". Name it through the project's own indirection ($OR_SRC / $OR_ROOT / "
          "$OR_BUILD / $HOME/<name>), or derive it (pyradioss.paths).")

    scripts = sorted(
        p for p in TOOLS.rglob("*")
        if p.is_file() and p.suffix in (".py", ".sh", ".txt"))
    assert _scanned_sources() == [PYPROJECT, LOCK, *_record_sources(), *scripts], (
        "the portability scan and the live gate must read the same files, in "
        "the same order; if this fires, one of them has been narrowed")


def test_a_machine_path_planted_in_a_tooling_file_is_caught_by_the_simulation(
        tmp_path):
    """The round-4 shape, planted OUTSIDE the repository, is reported.

    Non-vacuity for the test above, and the shape it exists for: the three hits
    the simulation missed were a **hardcoded default in code** in
    ``tools/oracle/toolchain_probe.py``, which is on this box, so the live
    existence rule had nothing to say about it.  The real file is copied to
    ``tmp_path`` — never planted in the working tree — one line carrying this
    box's absolute path is appended, and the SAME scan runs under the SAME
    simulated home.  Without the fix the copy reports ``[]``.
    """
    real = TOOLS / "oracle" / "toolchain_probe.py"
    gone = f"{THIS_BOX_HOME}/Projects/OpenRadioss/OpenCourant"
    copy = tmp_path / real.name
    copy.write_text(real.read_text(encoding="utf-8")
                    + f'\nOR_SRC = "{gone}"\n', encoding="utf-8")

    with _a_box_whose_home_is_not_this_one():
        planted = _existence_offenders([copy])
        untouched = _existence_offenders([real])
    assert untouched == [], (
        "the real file must be clean for this test to mean anything: "
        f"{untouched}")
    assert planted == [f"{copy.name}:{len(_lines(copy))}: {gone}"], (
        "a machine path planted in a tooling file must be reported under the "
        f"simulated home; got {planted}")


# ---------------------------------------------------------------------------
# 2. the harness's RPATH claim, compared with the binaries
# ---------------------------------------------------------------------------
def test_the_harness_records_the_rpath_the_binaries_actually_carry():
    """Every ``DT_RPATH``/``DT_RUNPATH`` directory the tooling claims must be one
    the installed oracle binaries actually carry.

    The claim under test: "both binaries carry
    ``DT_RPATH=/home/valentin/anaconda/lib``".  Measured with the harness's own
    ELF parser (:func:`tools.validate_vs_fortran.elf_search_paths`, which reads
    the dynamic section directly — no subprocess, so this is the same code path
    the hazard probe uses): both binaries report ``[]``.  ``readelf -d`` says the
    same (no tag 15, no tag 29).

    A claim marked as history is exempt on purpose — the pre-migration binaries
    really did carry that conda prefix, and the provenance JSON says so.

    Skips (never silently passes) when no oracle is configured here.
    """
    from tests.test_p0_oracle_build import _require_live_oracle

    _require_live_oracle(runtime_env=False)
    from tools import validate_vs_fortran as V

    oracle = V.oracle_paths()
    measured = {key: V.elf_search_paths(oracle[key]) for key in ("starter", "engine")}
    union = {p for entries in measured.values() for p in entries}
    for key, entries in measured.items():
        for entry in entries:
            assert Path(entry).is_dir(), (
                f"the harness probe reports {entry} as an RPATH of {key} but it "
                "is not a directory — the ELF parser is wrong, not the disk")

    claims = []
    for path in _scanned_sources():
        lines = _lines(path)
        for index, claimed, window in _path_claims(lines):
            if claimed in union:
                continue
            line, above = lines[index], lines[index - 1] if index else ""
            if "DT_RPATH" not in line and "DT_RUNPATH" not in line \
                    and "DT_RPATH" not in above and "DT_RUNPATH" not in above:
                continue          # a path, but not claimed as an RPATH entry
            if _marked_history(window):
                continue
            claims.append(f"{path.relative_to(REPO)}:{index + 1}: {claimed}")
    assert claims == [], (
        "these lines claim an RPATH entry the oracle binaries do not carry "
        f"(measured: {measured}): " + "; ".join(claims)
        + ". glibc searches DT_RPATH before LD_LIBRARY_PATH, so the claim is "
        "load-bearing: state what readelf actually shows (no DT_RPATH and no "
        "DT_RUNPATH at all), or mark the statement as the pre-migration record.")


def test_the_rpath_hazard_is_still_explained_and_stated_as_measured():
    """The hazard paragraph must keep its reasoning AND state today's fact.

    Two failure modes this closes:

    * the paragraph deleted to make it stop being false (the reasoning is the
      point: ``DT_RPATH`` is searched before ``LD_LIBRARY_PATH``, so the writer
      would be dlopen'd by ``h3d_dl.c:660-666`` whatever the harness exports);
    * the sentence deleted while the reasoning stays, leaving a reader unable to
      tell which of the two is true today.  Each docstring must therefore carry
      the reasoning tokens AND an explicit negation — "neither binary carries
      DT_RPATH ... at all" — which is the statement ``readelf`` confirms.
    """
    from tools import validate_vs_fortran as V

    # ``fortran_env`` cites the trial as ``:660-666`` (the file is named once,
    # above it); ``binary_rpath_hazards`` spells the citation out.
    required = {"binary_rpath_hazards": ("DT_RPATH", "LD_LIBRARY_PATH",
                                         "h3d_dl.c:660-666"),
                "fortran_env": ("DT_RPATH", "LD_LIBRARY_PATH", ":660-666")}
    for name, tokens in required.items():
        doc = getattr(V, name).__doc__ or ""
        for token in tokens:
            assert token in doc, (
                f"{name}'s docstring no longer explains the {token} half of the "
                "hazard; the reasoning must survive a correction to the "
                "measured value")
        negation = re.search(r"\b(no|not|neither|none|without)\b[^.]*DT_RPATH",
                             doc, re.I)
        assert negation, (
            f"{name}'s docstring does not say whether the binaries carry a "
            "DT_RPATH. The measured answer today is: neither of them does "
            "(readelf -d on starter_linux64_gf and engine_linux64_gf). Say so.")


# ---------------------------------------------------------------------------
# 3. packaging prose against the recorded pins
# ---------------------------------------------------------------------------
#: ``<distribution> <version>`` in a comment — an exact version adjacent to the
#: name it belongs to.  Operator-prefixed forms are deliberately NOT matched:
#: ``numpy<2.6,>=1.22`` is a constraint the upstream project declares, not a
#: claim about what is installed, and ``"numba>=0.59"`` is a floor in the extra
#: itself.  Case-insensitive since round 3, because the records write ``NumPy
#: 2.5.2`` and ``Python 3.14.6`` with the distribution's own capitalisation —
#: the same claim, spelled the way the vendor spells it.
VERSION_CLAIM = re.compile(
    r"\b(?P<pkg>numpy|scipy|numba|llvmlite|mpi4py|pytest|python)\s+"
    r"(?P<ver>\d+\.\d+(?:\.\d+)?)\b", re.I)

#: A migration arrow.  The value on its LEFT is the superseded one by
#: construction (``python 3.14.6→3.12.3`` is a record of what this box stopped
#: being), so it is provenance; the value on the right is the live claim and is
#: compared as usual.  This is the arrow-delta category of the records, and it
#: is why ``numpy 2.5.2→2.5.3`` in ``UPDATES.md`` passes while a bare
#: ``numpy 2.5.2`` does not.
_MIGRATION_ARROW = re.compile(r"\s*(?:→|->|=>|⟶|-->)\s*")

#: A FLOOR written in prose instead of with an operator.  :data:`_CONSTRAINT_TAIL`
#: already skips ``numpy>=1.22`` for the toolchain rule, and
#: :data:`VERSION_CLAIM` never matches that form — but the prose spelling
#: ("requires numpy 1.21 at minimum") has no operator, so it fell through to the
#: pin comparison and was reported as a false measurement.  A floor is satisfied
#: by every pin the lock could hold, so comparing one with the lock is a
#: requirement measured against reality.
_FLOOR_MARKERS = (
    "at minimum", "at least", "or newer", "or later", "or better",
    "or above", "or higher", "or greater", "and up", "and newer",
    "and later", "minimum", "floor",
)

#: Where a floor word still counts as describing THIS version: the clause it
#: sits in.  One clause, not the whole line — otherwise a floor mentioned
#: anywhere in a sentence would exempt a measurement elsewhere in it, which is
#: the laundering shape :func:`_in_uri_target` was just fixed for.  Brackets and
#: parentheses are breaks too, because a parenthetical "(at least 1.20 too)"
#: is not a statement about the version it follows.
_FLOOR_CLAUSE_BREAK = re.compile(r"[;,:.!?()\[\]]")

#: … and a clause that ASSERTS this version is what is on this box is never a
#: floor, however the sentence hedges it.  ``on this box numpy 1.21 is
#: installed at least`` is a measurement with a trailing "at least": the clause
#: says *this is what is here*, and the floor word only softens the claim.  The
#: clause break cannot catch it — there is no punctuation between the version
#: and the hedge — so this is the second half of the narrowing.
#:
#: ``is`` alone is deliberately NOT the trigger, because "the floor is numpy
#: 1.21" and "the requirement is cmake 3.15 or newer" are requirements; the
#: pattern needs an actual predicate of installation or measurement.
_MEASURED_CLAUSE = re.compile(
    r"\b(?:is|was|are|were)\s+(?:already\s+|still\s+|currently\s+|actually\s+)*"
    r"(?:installed|imported|importable|pinned|resolved|present|measured)\b",
    re.I)

#: How far either side of the version a floor word may sit and still count.
_FLOOR_WINDOW = 32


def _is_floor(line: str, match: re.Match) -> bool:
    """Is this ``pkg version`` a REQUIREMENT floor rather than a measurement?

    Two shapes, both scoped to the version's own clause:

    * ``numpy 1.21+`` — the ``+`` suffix, checked on the character right after
      the version;
    * a :data:`_FLOOR_MARKERS` word within :data:`_FLOOR_WINDOW` characters
      before the distribution name or after the version, **without crossing a
      clause boundary**: on the left that is the text after the last break, on
      the right the text before the first — and **not at all** when either
      clause carries a :data:`_MEASURED_CLAUSE` assertion.

    So ``requires numpy 1.21 at minimum`` and ``at minimum numpy 1.21`` are
    floors, while ``numpy 1.21 was installed; at least 1.20 is required``,
    ``numpy 1.21 is the measured version (at least 1.20 too)`` and ``on this
    box numpy 1.21 is installed at least`` are claims — pinned from both sides
    by :func:`test_a_requirement_floor_is_not_read_as_a_measurement`.
    """
    if line[match.end():].lstrip()[:1] == "+":
        return True
    head = line[:match.start()][-_FLOOR_WINDOW:]
    tail = line[match.end():][:_FLOOR_WINDOW]
    breaks = list(_FLOOR_CLAUSE_BREAK.finditer(head))
    if breaks:
        head = head[breaks[-1].end():]
    breaks = list(_FLOOR_CLAUSE_BREAK.finditer(tail))
    if breaks:
        tail = tail[:breaks[0].start()]
    if _MEASURED_CLAUSE.search(head) or _MEASURED_CLAUSE.search(tail):
        return False          # "… is installed at least" is a hedged claim
    return any(marker in span.lower() for span in (head, tail)
               for marker in _FLOOR_MARKERS)


_PIN_LINE = re.compile(r"^#\s*pin:\s*([A-Za-z0-9_.\-]+)==([^\s#]+)")


def _recorded_pins() -> dict[str, set[str]]:
    """Every version ``requirements-lock.txt`` records, per distribution.

    Both sections: ``[A]`` the Windows .venv (flat ``name==version`` lines) and
    ``[B]`` this box (the machine-verified ``# pin:`` lines).  A packaging
    comment may name a version from either box's record — what it may not do is
    invent one.
    """
    pins: dict[str, set[str]] = {}
    in_b = False
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.startswith("# [B]") or line.strip().startswith("# [B] at"):
            in_b = True
        match = _PIN_LINE.match(line)
        if match:
            pins.setdefault(match.group(1).lower(), set()).add(match.group(2))
        elif in_b or not line.startswith("#"):
            flat = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
            if flat:
                pins.setdefault(flat.group(1).lower(), set()).add(flat.group(2))
    return pins


def _version_agrees(claimed: str, recorded: set[str]) -> bool:
    """Is ``claimed`` consistent with a recorded pin?

    Consistent means equal, or a dotted-prefix of one (a comment may say
    ``0.68`` where the pin says ``0.68.0``); ``2.5.2`` against a ``2.5.3`` pin
    is not, and that is exactly the rot this kills.
    """
    for pin in recorded:
        if claimed == pin or pin.startswith(claimed + ".") \
                or claimed.startswith(pin + "."):
            return True
    return False


def _version_claim_offenders(lines: list[str], pins: dict[str, set[str]],
                             label: str, comment_only: bool = True
                             ) -> list[str]:
    """The ``label:<line>`` entries for prose naming an unrecorded version.

    Extracted so a self-test can drive the real rule over synthetic text (see
    :func:`test_the_version_rule_can_still_fail`).  ``comment_only`` reads just
    the ``#`` comment lines, which is what ``pyproject.toml`` needs: the
    requirement arrays themselves declare floors, not measurements.  The RECORDS
    (round 3) pass ``comment_only=False``, because in markdown every line is
    prose and a requirement reads ``numpy>=1.22``, which :data:`VERSION_CLAIM`
    does not match anyway.

    One exemption, and it is narrow: a version on the LEFT of a migration arrow
    (:data:`_MIGRATION_ARROW`) is the value the box moved away from, so it is a
    record rather than a claim.  The right-hand side is compared as usual, which
    is what keeps ``python 3.14.6→3.12.3`` honest — the live claim in it is
    still the one the lock has to carry.

    And the SAME history label the path rule honours, over the same window
    (:func:`_marked_history`), added in round 3: ``docs/STATE.md`` §Baseline
    writes "**Previous machine (Windows, Python 3.14.2 / numpy 2.4.6,
    pre-migration)**" and that is provenance, exactly as the path beside it
    would be.  Without it this rule reported the repo's own correctly labelled
    history section as rot — the crying-wolf failure this file exists to avoid,
    and it changed no pyproject verdict (zero offenders there before and after).
    """
    offenders = []
    for lineno, line in enumerate(lines, 1):
        if comment_only and not line.lstrip().startswith("#"):
            continue              # the requirement arrays themselves, not prose
        window = _label_window(lines, lineno - 1).lower()
        if _marked_history(window):
            continue              # labelled another machine's / another record's
        for match in VERSION_CLAIM.finditer(line):
            pkg, ver = match.group("pkg").lower(), match.group("ver")
            if _MIGRATION_ARROW.match(line[match.end():]):
                continue          # the superseded side of a stated migration
            if _is_floor(line, match):
                continue          # a requirement, not a measurement of the box
            if not _version_agrees(ver, pins[pkg]):
                offenders.append(
                    f"{label}:{lineno}: {pkg} {ver} "
                    f"(lock records {'/'.join(sorted(pins[pkg]))})")
    return offenders


def test_packaging_version_claims_agree_with_the_recorded_pins():
    """``pyproject.toml``'s comments must not contradict ``requirements-lock.txt``.

    The claim under test (P0.7, ``accel``): "Verified installed +
    kernel-compiling on the shared Linux box at numba 0.68.0 / llvmlite 0.50.0 /
    **numpy 2.5.2**" — while the lock's ``[B]`` records ``numpy==2.5.3``
    (Task P0.11, commit 18ce2e3, re-pinned against this ``.venv``).  Two records
    of one resolution, one of them false, in the two files every reader of this
    repo consults for "what is actually installed".

    Dropping the number is a legitimate answer and the one the fix takes: the
    comment then carries nothing that can rot, and points at the lock, whose
    ``# pin:`` lines ``tests/test_p0_optional_deps.py`` already asserts against
    the importable modules.  Naming the true number is equally acceptable — what
    is not acceptable is naming one the lock never recorded.
    """
    pins = _recorded_pins()
    for pkg in ("numpy", "scipy", "numba", "llvmlite", "mpi4py", "pytest"):
        assert pkg in pins, (
            f"{pkg} has no recorded version in requirements-lock.txt, so a "
            "pyproject comment cannot be checked against it; record the pin")

    offenders = _version_claim_offenders(
        PYPROJECT.read_text(encoding="utf-8").splitlines(), pins, "pyproject.toml")
    assert offenders == [], (
        "pyproject.toml prose names a version the lock never recorded: "
        + "; ".join(offenders)
        + ". Name the recorded version, or none at all and a pointer to the lock.")


def test_record_version_claims_agree_with_the_recorded_pins():
    """The same pin comparison over the records — the OTHER supported machine.

    Round 3.  The records are where a reader learns what is installed, and
    ``AGENTS.md`` states it in the imperative: "All dependencies are
    preinstalled (numpy 2.4.6, scipy 1.18.0, numba 0.66.0, pytest 9.1.1 —
    exact pins in ``requirements-lock.txt``)".  Those are the maintainer's
    **Windows** box, which this project deliberately still supports, and they
    ARE in the lock's §[A] — so they are legitimate and must pass
    (:func:`test_a_version_of_the_other_supported_machine_is_allowed` pins that
    direction).  What may not appear is a version the lock never recorded:
    ``plan/README.md`` §8's "Dev box: Python 3.14.6, NumPy 2.5.2" is exactly
    that, and it is a LIVE claim ("numba and mpi4py are NOT installed on this
    box") that this box contradicts.

    §[A] stays DESCRIPTIVE — nothing here compares it to the running
    interpreter.  This rule compares prose to the RECORD, which is the only
    comparison the lock supports.
    """
    pins = _recorded_pins()
    offenders = []
    for path in _record_sources():
        offenders += _version_claim_offenders(
            _lines(path), pins, path.relative_to(REPO).as_posix(),
            comment_only=False)
    assert offenders == [], (
        "these record lines name a distribution version the lock never "
        "recorded (either section): " + "; ".join(offenders)
        + ". A version that belongs to the other supported machine is fine "
        "because the lock records it (§[A] is the Windows box); a version "
        "nobody recorded is not. Name a recorded one, or drop the number and "
        "point at requirements-lock.txt.")


# ---------------------------------------------------------------------------
# 4. the build script's toolchain versions, against the tools
# ---------------------------------------------------------------------------
#: ``<tool> ... <version>`` in a comment, with a comparison-operator tail
#: excluded: "cmake >= 3.15 and < 4 is required" is a REQUIREMENT, not a
#: measurement, and failing it would be the test inventing its own rule.
TOOL_VERSION_CLAIM = re.compile(
    r"\b(?P<tool>cmake|gfortran|gcc|g\+\+|make)\b(?P<tail>[^\n]{0,60}?)"
    r"(?<![\w.])(?P<ver>\d+\.\d+(?:\.\d+)?)\b")

#: The Unicode operators are here for the records, which are written by hand and
#: use them naturally: ``plan/01_phase0_oracle_and_licensing.md`` §Tech stack
#: says "cmake **≥ 3.15** (measured 2026-10-03 on this box: /usr/bin/cmake
#: 3.28.3)" — the same REQUIREMENT as the ASCII form the rule already skips, and
#: the measured value beside it is the claim that gets compared.  Without these
#: three characters the rule reported a requirement as a false measurement, which
#: is the crying-wolf failure mode, not a finding.
_CONSTRAINT_TAIL = (">=", "<=", "==", "!=", "~>", "~=", ">", "<",
                    "≥", "≤", "≫", "≪", "≈", "⩾", "⩽", "⇒")


@functools.lru_cache(maxsize=None)
def _tool_banner(tool: str) -> str | None:
    """``<tool> --version`` for the tool ``PATH`` resolves, or ``None``.

    Cached because the scan is now over ~1.5 MB of records as well: one
    subprocess per tool, not one per version mentioned.  The measurement is a
    property of the box, not of the line being read, so the cache cannot change
    a verdict — it only keeps the gate's cost flat as the records grow.
    """
    exe = shutil.which(tool)
    if exe is None:
        return None
    done = subprocess.run([exe, "--version"], capture_output=True, text=True,
                          timeout=60)
    return done.stdout + done.stderr


def _tool_version_offenders(lines: list[str],
                            label: str) -> tuple[list[str], list[str]]:
    """``(offenders, unchecked)`` for toolchain versions asserted in ``lines``.

    Extracted from the test body so the self-tests can drive the REAL rule over
    synthetic text — the round-2 hole was only visible to a driver like this,
    not to a scan of the tree.

    Three ways a line is not a claim, in order:

    * a constraint tail (``cmake >= 3.15``) is a requirement, not a measurement;
    * a :data:`HISTORY_MARKERS` label in the window marks it another machine's;
    * a line inside an ATTRIBUTED specimen block whose text has the shape of
      captured output (:func:`_looks_like_captured_output`) — the two together,
      because either alone is a loophole (an unattributed block launders
      anything; a bare ``cmake 4.4.3`` in an attributed block is still a
      sentence, not a banner).

    ``unchecked`` lists tools absent from PATH: nothing to measure against, so
    nothing is asserted about them either way.
    """
    offenders: list[str] = []
    unchecked: list[str] = []
    quoted = _quoted_example_lines(lines)
    for index, line in enumerate(lines):
        window = _label_window(lines, index)
        for match in TOOL_VERSION_CLAIM.finditer(line):
            tool, tail, ver = (match.group("tool"), match.group("tail"),
                               match.group("ver"))
            if any(op in tail for op in _CONSTRAINT_TAIL):
                continue              # a requirement or a range, not a claim
            if _is_floor(line, match):
                continue              # a floor in prose ("cmake 3.15 at minimum")
            if _marked_history(window):
                continue              # labelled as another machine's toolchain
            if index in quoted and _looks_like_captured_output(line):
                continue              # a specimen of output, not a claim
            banner = _tool_banner(tool)
            if banner is None:
                unchecked.append(f"{label}:{index + 1}: {tool}")
                continue
            measured = re.search(r"\d+\.\d+(?:\.\d+)?", banner)
            assert measured, f"{tool} --version printed no version: {banner!r}"
            if not _version_agrees(ver, {measured.group(0)}):
                offenders.append(
                    f"{label}:{index + 1}: claims {tool} {ver}, "
                    f"but `{tool} --version` here says {measured.group(0)}")
    return offenders, unchecked


def _toolchain_offenders(paths: list[Path]) -> tuple[list[str], list[str]]:
    """``(offenders, unchecked)`` for every scanned file the rule can read.

    The scan :func:`test_the_build_script_names_the_tool_versions_this_box_has`
    performs, extracted for the same reason as :func:`_existence_offenders` — a
    regression test has to be able to drive the real scan over a scratch copy of
    a record file and see the planted claim, and :data:`SCANNED_SUFFIXES` is
    then stated in ONE place, so widening the scope to a new file type cannot
    silently skip it (which is what the round-3 gap was: a scope the gate could
    not see, not a rule that did not fire).
    """
    offenders: list[str] = []
    unchecked: list[str] = []
    for path in paths:
        if path.suffix not in SCANNED_SUFFIXES:
            continue
        found, skipped = _tool_version_offenders(
            _lines(path), path.relative_to(REPO).as_posix()
            if path.is_relative_to(REPO) else path.name)
        offenders += found
        unchecked += skipped
    return offenders, unchecked


def test_the_build_script_names_the_tool_versions_this_box_has():
    """``tools/`` prose must not describe a toolchain this box does not have.

    The claim this rule was written for, at ``build_oracle.sh:98``: "on this box
    ``command -v cmake`` is conda cmake **4.4.3**, which fails the configure" —
    i.e. the comment argues for pinning ``CMAKE=/usr/bin/cmake`` by asserting
    that the first cmake on PATH is a conda 4.x.  Measured here:
    ``command -v cmake`` is ``/usr/bin/cmake`` and ``cmake --version`` is
    **3.28.3** (so the default already satisfies the script's own
    3.15 <= version < 4 gate, and the argument is belt-and-braces, not a fact
    about this box).

    What it does NOT do is read a quoted sample as a claim:
    ``_gcc_version``'s docstring in ``tools/oracle/toolchain_probe.py`` shows
    ``gfortran (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)`` inside a reST literal
    block to document what it parses, and that banner belongs to no machine this
    repo runs on.  :func:`_quoted_example_lines` draws the line, and only when
    the opener ATTRIBUTES the block and the text has output shape; prose claims —
    every comment, every ordinary docstring line, every unattributed block — stay
    checked.  :func:`test_a_quoted_example_is_not_a_claim_but_prose_beside_it_is`
    and :func:`test_a_stale_claim_written_inside_a_literal_block_is_still_a_claim`
    prove both halves on the real file plus synthetic ones.

    The reasoning the corrected build-script comment carries — CMake 4 rejects
    upstream's ``cmake_minimum_required (VERSION 3.15)`` and the gate therefore
    must be explicit — is untouched by any of this.

    Since round 3 the scan reads the records too (:data:`SCANNED_SUFFIXES`
    gained ``.md``), because a toolchain version quoted in ``docs/STATE.md`` is
    as load-bearing as one in a build script: both are what a reader copies
    into their next command.
    """
    offenders, unchecked = _toolchain_offenders(_scanned_sources())
    assert offenders == [], (
        "toolchain versions asserted in comments that this box contradicts: "
        + "; ".join(offenders)
        + ". Measure it here (`<tool> --version`) and write what it says, or "
        f"label the statement as another machine's. Not checked at all: {unchecked}")


# ---------------------------------------------------------------------------
# 5. the checkers themselves can still fail
# ---------------------------------------------------------------------------
def test_the_history_rule_can_still_fail():
    """Drive :func:`_stale_paths_in` over synthetic text, all three ways.

    A rule that accepts everything is worse than no rule, because it reads as
    coverage.  These are the three answers that must differ: the exact shape of
    the false claim this task removes, the shape that makes it legitimate
    provenance, and a path that simply exists.
    """
    gone = "/home/valentin/anaconda/lib"

    stale = ["    box both binaries carry ``DT_RPATH=" + gone + "`` — a conda",
             "    prefix with nothing to do with the oracle."]
    assert _stale_paths_in(stale) == [(0, gone)], (
        "an unmarked path that does not exist must be reported — this is the "
        "exact shape of the false claim this task removes")

    historical = ["    the pre-migration binaries carried ``DT_RPATH=" + gone + "``;",
                  "    that conda prefix no longer exists on this box, so the",
                  "    rebuilt binaries carry no DT_RPATH at all."]
    assert _stale_paths_in(historical) == [], (
        "a path labelled as another machine's / another record's is provenance, "
        "not rot — the rule must accept it")

    real = ["    the reader library lives in "
            "/home/valentin/Projects/OpenRadioss/OpenCourant (read-only)."]
    assert _stale_paths_in(real) == [], (
        "a path that EXISTS here is not stale whatever the sentence around it "
        "says — being accurate is the measured half's job, not this rule's")

    distant = ["    (the conda-forge cmake 4.x that used to shadow it went with",
               "    /home/valentin/anaconda, which no longer exists;"]
    assert _stale_paths_in(distant) == [], (
        "a label on the same line as the path is enough — that is how the "
        "corrected cmake note in build_oracle.sh is written")

    assert _stale_paths_in(["    see /home/valentin/anaconda/lib"])[0][1] == gone, (
        "one line with no label at all stays a violation")


def test_the_version_rule_can_still_fail():
    """Same discipline for the pin comparison: a false version must be caught."""
    pins = {"numpy": {"2.4.6", "2.5.3"}, "numba": {"0.66.0", "0.68.0"}}
    assert not _version_agrees("2.5.2", pins["numpy"]), (
        "numpy 2.5.2 is what pyproject.toml claimed against a 2.5.3 pin")
    assert _version_agrees("2.5.3", pins["numpy"])
    assert _version_agrees("0.68", pins["numba"]), (
        "a dotted prefix of a recorded pin is not a contradiction")
    assert not _version_agrees("9.9.9", pins["numba"])

    pyproject_stale = [
        "[project.optional-dependencies]",
        "# kernel-compiling on the shared Linux box at numpy 2.5.2."]
    assert _version_claim_offenders(
        pyproject_stale, {"numpy": {"2.5.3"}}, "pyproject.toml"
    ) == ["pyproject.toml:2: numpy 2.5.2 (lock records 2.5.3)"], (
        "the whole rule, not just the comparison it ends in, must reject the "
        "exact comment f5bd4c5 removed")


def test_a_requirement_floor_is_not_read_as_a_measurement():
    """``numpy 1.21 at minimum`` says what the build NEEDS, not what is here.

    The round-3 review found a false-positive class: :data:`_CONSTRAINT_TAIL`
    already skips an operator-prefixed requirement (``numpy>=1.22``,
    ``cmake >= 3.15``) for the toolchain rule, and :data:`VERSION_CLAIM` skips
    the same form by construction — but the **prose** spelling of a floor has no
    operator, so it fell through to the pin comparison and was reported::

        requires numpy 1.21 at minimum   ->  numpy 1.21 (lock records 2.4.6/2.5.3)

    That is the rule asserting something it cannot know: a floor is satisfied by
    *every* pin the lock could hold, so comparing it with the lock compares a
    requirement with a measurement — exactly the category error the operator
    tails exist to prevent.  ``plan/README.md`` §Tech stack writes the very same
    kind of sentence in the operator form ("NumPy ≥ 1.22") and must be allowed
    to keep it.

    The floor window is deliberately one CLAUSE — the text after the last break
    on the left, the text before the first break on the right — not the whole
    line, so the exemption cannot swallow a real claim that happens to sit in
    the same sentence: the cases below that must still FAIL are the ones where
    the floor word is in a different clause from the version.

    A clause break is not enough, though, because a clause can hedge a
    measurement in the same breath.  The reviewer's laundering case is
    ``on this box numpy 1.21 is installed at least``: no punctuation separates
    the version from "at least", the clause-scoped window therefore admitted
    it, and a false *measurement* of the box exited the gate as a
    *requirement*.  :data:`_MEASURED_CLAUSE` closes that — a clause that says
    the version IS installed/present/pinned is a claim whatever follows it.
    """
    pins = {"numpy": {"2.4.6", "2.5.3"}}

    floors = [
        "requires numpy 1.21 at minimum",
        "requires numpy 1.21+",
        "requires numpy 1.21 or newer",
        "requires numpy 1.21 or later",
        "requires numpy 1.21 and up",
        "numpy 1.21 at least",
        "at minimum numpy 1.21",
        "the floor is numpy 1.21 for the accel extra",
        "the requirement is numpy 1.21 and up",
        "numpy>=1.21",          # operator form: never matched in the first place
    ]
    for line in floors:
        assert _version_claim_offenders([line], pins, "x.md",
                                        comment_only=False) == [], (
            f"{line!r} is a REQUIREMENT (a floor), not a measurement of this "
            f"box; it must not be compared with the lock")

    still_caught = [
        "on this box numpy 1.21 is what is installed",         # plain claim
        "the lock pins numpy 1.21, verified here",              # plain claim
        "numpy 1.21 was installed; at least 1.20 is required",  # floor elsewhere
        "numpy 1.21 is the measured version (at least 1.20 too)",  # ditto
        "we measured numpy 1.21",                               # plain claim
        # the reviewer's laundering case: same clause, no punctuation, and the
        # "at least" used to buy the claim an exemption
        "on this box numpy 1.21 is installed at least",
        "numpy 1.21 is present at least",
        "at least numpy 1.21 is installed",
    ]
    for line in still_caught:
        assert _version_claim_offenders([line], pins, "x.md",
                                        comment_only=False) == [
            f"x.md:1: numpy 1.21 (lock records 2.4.6/2.5.3)"], (
            f"{line!r} states a measurement and must still be reported")

    assert _recorded_pins()["numpy"], "the live lock still records numpy"


def test_a_toolchain_floor_is_not_read_as_a_measurement():
    """The same class on the toolchain rule, which has the same operator tail.

    ``cmake 3.15 at minimum`` / ``gfortran 9.2 or newer`` is the build's
    requirement; comparing it with ``cmake --version`` on this box is the same
    category error :data:`_CONSTRAINT_TAIL` was added to prevent, and the
    reviewer's report is that the prose spelling slipped through it.  The
    measured claim beside it must still be reported, which is the whole point.
    """
    got, _ = _tool_version_offenders(["cmake 3.15 at minimum"], "x.md")
    assert got == [], f"a toolchain floor must not be compared with PATH: {got}"
    got, _ = _tool_version_offenders(["gfortran 9.2 or newer"], "x.md")
    assert got == [], f"nor must an 'or newer' floor: {got}"
    got, _ = _tool_version_offenders(["cmake 4.4.3 at minimum"], "x.md")
    assert got == [], (
        "and the NUMBER in a floor is still a requirement, not a measurement -- "
        f"what makes it a floor is the word, not the value; got {got}")
    got, _ = _tool_version_offenders(
        ["on this box `command -v cmake` is cmake 4.4.3, at least per the "
         "conda-forge build notes"], "x.md")
    assert [o for o in got if "claims cmake 4.4.3" in o], (
        "but a measurement whose floor word sits in a different clause is still "
        f"a claim; got {got}")
    got, _ = _tool_version_offenders(
        ["# cmake 4.4.3 is the pre-migration conda-forge build"], "x.md")
    assert got == [], (
        "a history label is still honoured beside the new floor rule")


def test_a_quoted_example_is_not_a_claim_but_prose_beside_it_is():
    """The claim/specimen line, drawn on the real file that tripped it.

    ``tools/oracle/toolchain_probe.py``'s ``_gcc_version`` documents what it
    parses with a reST literal block containing this box's banner and a Red Hat
    one.  The rule must read the block as quoted material and the sentence that
    introduces it as prose — otherwise the version rule calls a parser's example
    a stale machine fact, and the only "fixes" left are deleting the example or
    deleting the rule.

    The three synthetic cases below pin the boundary from the other side, so the
    exemption cannot be stretched into a loophole: a false version in ordinary
    prose is still compared, and so is one written immediately AFTER an example
    block (indenting a claim does not launder it; only quoting it does).
    """
    probe = _lines(TOOLS / "oracle" / "toolchain_probe.py")
    quoted = _quoted_example_lines(probe)

    red_hat = next(i for i, line in enumerate(probe) if "gfortran (GCC)" in line)
    opener = next(i for i in range(red_hat, -1, -1) if probe[i].rstrip().endswith("::"))
    assert red_hat in quoted, (
        "the Red Hat banner in _gcc_version's docstring is a quoted example, "
        "not a claim about this box's compiler")
    assert opener not in quoted, (
        "the line ending in '::' is the sentence that INTRODUCES the example "
        "(it says what the banner looks like); it is prose and stays checked")
    assert set(range(opener + 1, red_hat + 1)) <= quoted, (
        "the whole literal block is quoted: the blank line AND both sample "
        "banners. Note the first one happens to BE this box's banner - it is "
        "still exempt, because a specimen is judged by whether it is a specimen, "
        "not by whether it is currently true. The authoritative compiler record "
        "is toolchain_probe.json, held to this box by tests/test_p0_toolchain.py")

    stale = ["# cmake: prefer the upstream-era 3.28 over the conda 4.x, which",
             "# on this box `command -v cmake` is conda cmake 4.4.3."]
    assert _quoted_example_lines(stale) == set(), "ordinary prose is never quoted"
    found = [m for m in TOOL_VERSION_CLAIM.finditer(stale[1])
             if m.group("tool") == "cmake" and m.group("ver") == "4.4.3"]
    assert found, "a false version in prose must still be seen by the rule"

    quoted_stale = ["# cmake banners look like this::", "",
                    "        cmake version 4.4.3"]
    assert 2 in _quoted_example_lines(quoted_stale), (
        "the same string inside a literal block is a specimen")

    after = quoted_stale + ["# on this box cmake is 4.4.3."]
    assert 3 not in _quoted_example_lines(after), (
        "the first line at the block's indentation level ends the block, so a "
        "claim written right after it is prose again")


def test_a_stale_claim_written_inside_a_literal_block_is_still_a_claim():
    """The round-2 hole, closed and pinned: indentation is not a licence to lie.

    Fix round 1 exempted a reST literal block or a fenced block from the scan on
    STRUCTURE alone, and the review of that round built the block below and ran
    it through the real functions.  Every rule skipped every line::

        Measured facts::

            cmake 4.4.3
            DT_RPATH=/home/valentin/anaconda/lib

    A false path that does not exist here, a false cmake version next to it, and
    zero offenders: a rot scanner that a two-line indent switch disarms.

    Two things close it, and this test holds both:

    * a block is a specimen only if its opener ATTRIBUTES it (a producer's
      output, an illustration of a format).  "Measured facts::" attributes the
      lines to nobody, so they are prose and every rule reads them;
    * a quoted block buys a MACHINE PATH nothing at all (:func:`_path_claims`).
      Quoting cannot make a location exist.

    The last two cases are the cost, stated rather than hidden: a block whose
    opener names a producer and whose text has the shape of that producer's
    output is still a specimen, including when it quotes a version or a path
    that is false here.  That is the residual this rule cannot close without
    deleting ``_gcc_version``'s own documentation — and it now costs a
    deliberately framed sentence instead of an indent.
    """
    gone = "/home/valentin/anaconda/lib"
    planted = ["    Measured facts::", "",
               "        cmake 4.4.3",
               "        DT_RPATH=" + gone]

    assert _quoted_example_lines(planted) == set(), (
        "a block whose opener asserts facts attributes its lines to nobody, so "
        "it is prose — the exemption is not available to it")
    assert _stale_paths_in(planted) == [(3, gone)], (
        "the existence rule must read a planted path inside a literal block; "
        "quoting is not a defence for a location that is not on this box")
    offenders, _ = _tool_version_offenders(planted, "planted.py")
    assert [o for o in offenders if "claims cmake 4.4.3" in o], (
        "the version rule must read a planted version inside a literal block: "
        f"got {offenders}")

    fenced = ["The measured facts on this box:", "",
              "```", "cmake 4.4.3", "DT_RPATH=" + gone, "```"]
    assert _stale_paths_in(fenced) == [(4, gone)], (
        "a fence is not a hole: the same path, fenced, is still read")
    offenders, _ = _tool_version_offenders(fenced, "fenced.txt")
    assert [o for o in offenders if "claims cmake 4.4.3" in o], (
        f"and so is the same version, fenced: got {offenders}")

    labelled = ["    Measured facts::", "",
                "        DT_RPATH=" + gone + " (the pre-migration record)"]
    assert _stale_paths_in(labelled) == [], (
        "the escape hatch survives the tightening: an attributed-to-history "
        "claim inside a block is still provenance, not rot")

    banner = ["GCC prints the packaging in parentheses and its own version::", "",
              "        gfortran (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)"]
    assert 2 in _quoted_example_lines(banner), (
        "the specimen shape must survive: opener attributes, text is output")
    assert _tool_version_offenders(banner, "banner.py")[0] == [], (
        "a captured banner from another machine is not compared with this box's")

    dump = ["A `readelf -d` run on the pre-migration binaries printed::", "",
            "        0x000000000000001e (RUNPATH) Library runpath: [" + gone
            + "]   # pre-migration record"]
    assert 2 in _quoted_example_lines(dump) and _stale_paths_in(dump) == [], (
        "a quoted readelf dump naming the old conda prefix stays provenance: "
        "the opener attributes it, the line is measured output, and the label "
        "is in the window (the window does NOT reach up to the opener — that is "
        "what stops one 'history' at the top of a block from laundering it)")

    tautology = ["The readelf output printed::", "", "        DT_RPATH=" + gone]
    quoted = _quoted_example_lines(tautology)
    assert 2 in quoted and 0 not in quoted, (
        "attribution plus indentation is still what buys the exemption")
    assert _stale_paths_in(tautology) == [(2, gone)], (
        "but it never buys it for a PATH: the value is not output shape, it is a "
        "location, and a location is either on this box or labelled")


# ---------------------------------------------------------------------------
# 6. the widened scope, and the four categories it had to learn
# ---------------------------------------------------------------------------
def test_the_records_a_future_agent_reads_first_are_in_scope():
    """THE REGRESSION TEST for round 3: the scope cannot be narrowed again.

    The gap this round closed was not a rule that failed to fire — it was a
    scope that could not see ``docs/``, ``plan/``, ``UPDATES.md``,
    ``README.md`` or ``AGENTS.md``, so a false claim written into any of them
    left the gate green.  A "cleanup" that trims the scanned-path list back to
    ``tools/`` raises nothing and reports nothing: it silently undoes the whole
    task.  So the list is pinned from three sides —

    * the named files, one by one, by path;
    * EVERY ``docs/**/*.md`` and ``plan/**/*.md`` on disk, so a record added
      tomorrow is scanned without anyone remembering to add it;
    * the tooling surface the rule was born for, so the widening did not buy
      its coverage by giving something up.

    and then, so the membership assertion is not vacuous, the scan those files
    feed is driven over a scratch copy carrying a planted false claim
    (:func:`test_a_false_claim_planted_in_a_record_file_is_caught`).
    """
    scope = _scanned_sources()

    for required in (README, UPDATES, AGENTS, STATE, DOCS / "OPEN_BUGS.md",
                     PLAN / "README.md", PLAN / "00_ORCHESTRATION.md",
                     PLAN / "01_phase0_oracle_and_licensing.md"):
        assert required in scope, (
            f"{required.relative_to(REPO)} is a record a future agent reads "
            "first and it has left the scanned scope — the round-3 widening is "
            "being undone. Add it to RECORD_FILES/RECORD_GLOBS, or state in "
            "_scanned_sources' docstring why a record is exempt.")

    on_disk = {p for pattern in RECORD_GLOBS for p in REPO.glob(pattern)}
    unscanned = sorted(p.relative_to(REPO).as_posix() for p in on_disk
                       if p not in scope)
    assert unscanned == [], (
        "these record files exist and are not scanned: " + ", ".join(unscanned)
        + ". A record is in scope by virtue of being in docs/ or plan/ — add "
          "the glob, do not add a list of filenames.")

    for kept in (PYPROJECT, LOCK, TOOLS / "validate_vs_fortran.py",
                 TOOLS / "oracle" / "build_oracle.sh",
                 TOOLS / "oracle" / "toolchain_probe.py"):
        assert kept in scope, (
            f"{kept.relative_to(REPO)} left the scanned scope. Widening to the "
            "records must not cost the tooling surface — that is where this "
            "task's original four false claims lived.")

    assert ".md" in SCANNED_SUFFIXES, (
        "the records are markdown: a scope that lists docs/ but not the .md "
        "suffix is the unnamed scope this rule was written to prevent")


def test_a_false_claim_planted_in_a_record_file_is_caught(tmp_path):
    """A claim planted in a COPY of ``docs/STATE.md`` outside the repo is caught.

    This is the direction the whole task exists for, and it is measured the way
    the gap was reproduced: the real file is copied to ``tmp_path`` (never
    planted in the working tree), one false line is appended, and BOTH live
    scans are driven over the copy by the same functions the live tests use.

    Before round 3 the answer was that nothing reads ``docs/`` at all — the
    planted line could have been anything and the run stayed green.  Now the
    planted path and the planted cmake version are both reported, by file and
    line, with the reason in the message.
    """
    gone = "/home/valentin/anaconda/lib"
    copy = tmp_path / "STATE.md"
    copy.write_text(STATE.read_text(encoding="utf-8")
                    + f"\nThe oracle reader library is at {gone} on this box, "
                      "and cmake is 4.4.3.\n", encoding="utf-8")

    planted = f"{len(_lines(copy))}: {gone}"
    # the copy is labelled by its bare name and the real file by its repo path,
    # so compare the "line: path" tails — the append is at the end, so every
    # earlier line number is identical between the two files
    tails = lambda offenders: [o.split(":", 1)[1] for o in offenders]  # noqa: E731
    assert tails(_existence_offenders([copy])) == \
        tails(_existence_offenders([STATE])) + [planted], (
        "planting one false claim in a copy of docs/STATE.md must add exactly "
        f"one offender, the planted line; got "
        f"{tails(_existence_offenders([copy]))}")

    toolchain, _ = _toolchain_offenders([copy])
    assert [o for o in toolchain if "claims cmake 4.4.3" in o], (
        "and so must the planted toolchain version: " + "; ".join(toolchain))

    # the planted line number exists only in the copy, so the comparison above
    # can only hold if the copy differs from the real file by the two appended
    assert planted not in tails(_existence_offenders([STATE])) \
        and len(_lines(copy)) == len(_lines(STATE)) + 2, (
        "the two files must differ by exactly the two appended lines")


def test_a_shell_variable_placeholder_is_not_a_machine_path():
    """``$OR_SRC`` and friends are placeholders in documented commands.

    The records are full of them — ``docs/STATE.md`` writes "every ported
    formula cites its upstream file under ``$OR_SRC``", ``UPDATES.md`` writes
    "``$OR_ROOT/bin``" and "``$OR_BUILD/exec/{starter,engine}``", and README
    shows ``export OR_ROOT=<your OpenRadioss install prefix>``.  A placeholder
    is not a filesystem location: there is nothing to stat and nothing that can
    rot, so :data:`MACHINE_PATH` must not produce a claim for it.  It does not,
    because the variable name sits where a path root would be — but that is a
    property of the regex's lookbehind and the character after the root, not a
    documented rule, and an edit that widened the root list would quietly start
    flagging them.  Hence the pin.

    Note what IS still read: an ``export`` that names a real directory
    (``OR_SRC=/home/valentin/Projects/OpenRadioss/OpenCourant``) is a literal
    value in a command a reader may run, so it stays a claim and must exist.
    """
    placeholders = [
        "every formula cites its upstream file under `$OR_SRC`"
        " (`starter/source/…`, `engine/source/…`).",
        "recompiled into `$OR_ROOT/bin` and mirrored at `$OR_BUILD/extlib`.",
        "byte-identical to the build outputs `$OR_BUILD/exec/{starter,engine}`.",
        "so on Linux `export OR_ROOT=<your OpenRadioss install prefix>` is all",
        "`export OR_SRC=${OR_SRC:-/tmp/whatever}` needs, and `%OR_ROOT%\\bin`",
        "the oracle lives under $HOME/OpenRadioss_or/bin on every box.",
    ]
    assert _path_claims(placeholders) == [], (
        "a shell variable is a placeholder, not a path: "
        f"{_path_claims(placeholders)}")

    literal = ["export OR_SRC=/home/valentin/Projects/OpenRadioss/OpenCourant"]
    assert [p for _, p, _ in _path_claims(literal)] == [
        "/home/valentin/Projects/OpenRadioss/OpenCourant"], (
        "but an export that names a real directory is a value a reader may run, "
        "so it stays a claim (and this one exists here, so it passes)")


def test_a_documented_windows_fallback_is_not_a_false_claim():
    """The Windows box this project still supports must stay documentable.

    Two independent reasons the record passes, and both are asserted here so
    neither can be taken away silently:

    * ``README.md`` documents ``C:\\OpenRadioss\\exec`` as an intentional
      fallback — "the Windows compatibility path ... is still tried last" — and
      :data:`MACHINE_PATH` is ``$HOME``-shaped, so a backslash path is not a
      claim this gate can measure.  That is a LIMITATION, not a permission:
      there is no Windows box here to compare against, and declaring those
      paths false would be the rule inventing a measurement.  Task P1.0 owns
      the rewrite of the Windows-shaped files.
    * ``AGENTS.md``'s ``.venv\\Scripts\\python.exe`` is the same shape, and
      ``AGENTS.md`` is nevertheless IN the scanned scope, so the forward-slash
      claims it makes (a ``/home/...`` path added tomorrow, a ``gfortran``
      version) are read.

    Both are checked against the REAL files, not against a paraphrase, so a
    rewrite of either line is what would have to keep this test true.
    """
    readme = _lines(README)
    fallback = next(i for i, line in enumerate(readme)
                    if "compatibility path" in line)
    window = "\n".join(readme[fallback - 1:fallback + 2])
    assert "C:\\OpenRadioss\\exec" in window, (
        "README.md's documented Windows fallback moved; this test is pinned to "
        "the wording that makes it an intentional fallback, not a stale path")
    assert _path_claims(readme[fallback - 1:fallback + 2]) == [], (
        "a backslash Windows path is outside what this box can measure — see "
        "the module docstring; if this ever fires, MACHINE_PATH was widened")

    agents = _lines(AGENTS)
    assert any("C:\\OpenRadioss" in line for line in agents), (
        "AGENTS.md is expected to be Windows-shaped (task P1.0 owns the "
        "rewrite). If it has been rewritten, this test should be replaced by "
        "one that pins the new machine's real paths.")
    assert _path_claims(agents) == [], (
        "AGENTS.md is in the scanned scope, so any $HOME-shaped path in it is "
        f"a claim: {_path_claims(agents)}")


def test_a_version_of_the_other_supported_machine_is_allowed():
    """§[A] is the maintainer's Windows box, and its pins are legitimate.

    ``AGENTS.md`` states the environment in the imperative — "All dependencies
    are preinstalled (numpy 2.4.6, scipy 1.18.0, numba 0.66.0, pytest 9.1.1 —
    exact pins in ``requirements-lock.txt``)" — and those are the WINDOWS box's
    numbers.  This project still supports that box, so they are true *there*,
    they are recorded in §[A], and this rule must accept them.  What it must not
    accept is a version nobody recorded: ``numpy 2.5.2`` was the number purged
    from ``pyproject.toml`` in P0.15 and it is in no section of the lock.

    Driven over the real ``AGENTS.md`` line and the real lock, so both halves
    are measured rather than asserted.
    """
    pins = _recorded_pins()
    assert "2.4.6" in pins["numpy"] and "0.66.0" in pins["numba"] \
        and "1.18.0" in pins["scipy"], (
        f"the lock no longer records the Windows box's pins: {pins}")

    windows = next(line for line in _lines(AGENTS)
                   if "numpy 2.4.6" in line)
    assert _version_claim_offenders(
        [windows], pins, "AGENTS.md", comment_only=False) == [], (
        "AGENTS.md's Windows-box versions are recorded in §[A] and must pass: "
        f"{windows.strip()}")

    for pkg, dead in (("numpy", "2.5.2"), ("numba", "9.9.9"), ("scipy", "1.7.0")):
        assert not _version_agrees(dead, pins[pkg]), (
            f"{pkg} {dead} is recorded nowhere in the lock and must not be "
            "accepted as a version claim")


def test_a_uri_link_target_is_not_a_filesystem_claim():
    """``[x](C:/Users/pmqua/…)`` is an address; the link TEXT is still prose.

    ``docs/BUG_REPORT_2026-09-05.md`` carries 18 such permalinks (on 12 lines)
    into the pre-migration checkout, and they are the only machine paths in it.
    Without :func:`_in_uri_target` the widened scan reported all 18 as false
    claims — the crying-wolf outcome, since ``Path("C:/…").exists()`` is false
    for a reason that has nothing to do with the record being wrong.

    The residual is stated and pinned rather than hidden: only the span after
    ``](`` is exempt, so a claim written in the link TEXT — where a writer would
    actually put one — is still read, and a relative or scheme-less target is
    not exempt at all.
    """
    permalink = ("Location: [starter_keywords.py:14847]"
                 "(C:/Users/pmqua/PycharmProjects/OpenRadioss_Python/pyradioss/"
                 "input/starter_keywords.py:14847).")
    assert _path_claims([permalink]) == [], (
        "a Windows permalink target is an address in a link namespace, not a "
        "filesystem location on this box")
    sole = next(m for m in MACHINE_PATH.finditer(permalink))
    assert _in_uri_target(permalink, sole.start(), sole.end()), (
        "and the span is recognised as the one after '](' specifically")

    assert _stale_paths_in([
        "the reader library is at /home/valentin/anaconda/lib, see "
        "[the old prefix](C:/Users/pmqua/x/anaconda/lib) for the record"
    ])[0][1] == "/home/valentin/anaconda/lib", (
        "the exemption is the TARGET span only: a claim in the link text is "
        "still a claim")

    schemeless = "see [the oracle](/home/valentin/anaconda/lib) for the claim"
    assert not _in_uri_target(schemeless, schemeless.index("/home"),
                              schemeless.index("/home")
                              + len("/home/valentin/anaconda/lib")), (
        "a scheme-less target is not exempt — only a URI is an address")
    addressed = ("see [the oracle](https://ci/home/valentin/anaconda/lib) now")
    assert _in_uri_target(addressed, addressed.index("/home"),
                          addressed.index("/home")
                          + len("/home/valentin/anaconda/lib")), (
        "a scheme is what makes it an address rather than a location")
    assert [claim[1] for claim in _path_claims(
        ["the mirror is at /home/valentin/anaconda/bin here"])] == [
            "/home/valentin/anaconda/bin"], (
        "and a plain absolute path in prose is a claim, whatever follows it")

    bug_report = DOCS / "BUG_REPORT_2026-09-05.md"
    raw = sum(1 for line in _lines(bug_report)
              if "](/C:/Users/pmqua" in line or "](C:/Users/pmqua" in line)
    assert raw, ("the permalinks this test reasons about are gone from "
                 f"{bug_report.name}; replace this test with one that pins "
                 "whatever replaced them")
    assert _existence_offenders([bug_report]) == [], (
        "a dated report full of pre-migration permalinks is not 18 false "
        "claims")


def test_a_uri_token_elsewhere_on_the_line_does_not_exempt_a_path():
    """The reviewer's laundering case, and the reason the exemption is a span.

    ``_in_uri_target`` found the *first* ``](`` to the left of a match and
    treated everything after it as a link target, so one URI-ish token anywhere
    earlier on the line disarmed every rule on every remaining path.  The
    reviewer reproduced it in one line::

        see [x](C:/a/b) and … /home/valentin/anaconda/lib      ->  []

    The path after the ellipsis is prose — it is outside the link, its target
    closed on the first ``)`` — and ``pyradioss/`` has no such directory, so the
    existence rule must report it.  The helper's own docstring says the span
    after ``](`` *up to the closing ``)``* is exempt; the code checked neither
    the closing paren nor where the match ends.

    Both halves are pinned, because the fix must not be "drop the exemption":
    a genuine target is still an address, a bare path in prose is a claim, and
    the boundary cases that used to launder (a second path on the same line, a
    path that runs *past* the closing paren, an unclosed target) are all
    reported.
    """
    gone = "/home/valentin/anaconda/lib"

    laundering = "see [x](C:/a/b) and … " + gone
    assert [claim[1] for claim in _path_claims([laundering])] == [gone], (
        "one link target on the line must not exempt a bare path in prose "
        f"beside it: {_path_claims([laundering])}")
    assert _stale_paths_in([laundering]) == [(0, gone)], (
        "and the existence rule must therefore report it — the defect this test "
        "pins is that it reported nothing")

    # the real thing still passes: an address in a link namespace
    permalink = ("Location: [starter_keywords.py:14847](C:/Users/pmqua/x/"
                 "starter_keywords.py:14847).")
    assert _path_claims([permalink]) == [], (
        "a genuine permalink target is an address, not a filesystem location")

    # the boundary cases that used to launder, with their expected answers
    gone = "/home/valentin/anaconda/lib"
    for line, exempt in (
        # a SECOND path in prose, after a target that has closed: the reviewer's
        # shape, and the one the old code let through
        (f"[a](https://example.invalid/ok) … {gone}", False),
        (f"see [x](C:/a/b) and … {gone}", False),
        # a path in the link TEXT, before any '](' at all
        (f"the mirror is {gone}, see [x](C:/a/b)", False),
        # a ']' + '(' inside the target: the span we found is not the real one
        (f"[a](C:/x[0]({ gone})", False),
        # genuinely inside a closed target: the URI's own path component
        (f"[a](https://example.invalid/{gone})", True),
        (f"[a](https://example.invalid{ gone})", True),
        (f"[a](file://{gone})", True),
        # a target that never closes is not a link
        (f"[a](https://example.invalid/{gone} and prose", False),
        # a scheme-less target is an ordinary path in a link
        (f"[a]({gone})", False),
    ):
        got = [claim[1] for claim in _path_claims([line])]
        assert got == ([] if exempt else [gone]), (
            f"{line!r} -> {got}; only a path genuinely INSIDE a closed "
            f"link target is exempt (expected exempt={exempt})")


def test_the_uri_exemption_is_a_span_not_the_rest_of_the_line():
    """:func:`_in_uri_target` takes the match's END, and bounds the target.

    The hole was a signature problem as much as a logic one: given only
    ``start`` there is no way to know whether the match ends before the target's
    closing ``)``.  So ``end`` is required, ``_path_claims`` passes the real
    ``match.end()``, and the closing delimiter is located by search rather than
    assumed — both stated here so a future edit that makes ``end`` optional
    again (or defaults it to "the rest of the line") is a failure with a name.
    """
    import inspect

    params = list(inspect.signature(_in_uri_target).parameters)
    assert params == ["line", "start", "end"], (
        f"_in_uri_target's parameters are {params}; `end` (the match's end "
        "offset) must be required, not optional — an optional `end` is how "
        "'the whole rest of the line is the target' comes back")

    gone = "/home/valentin/anaconda/lib"
    line = "[a](https://example.invalid/x) … " + gone
    start = line.index("/home/")
    end = start + len(gone)
    assert not _in_uri_target(line, start, end), (
        "a path after the target's closing ')' is prose")
    inside = "[a](https://example.invalid" + gone + ")"
    assert _in_uri_target(inside, inside.index("/home/"), len(inside) - 1), (
        "a path inside the target, up to but not including the ')', is an "
        "address")


def test_the_harness_imports_and_resolves_oracle_paths():
    """Importing the harness must still work after editing it.

    ``tests/test_p0_harness_portable.py`` owns that contract in depth; this is
    the one-line smoke that a documentation edit inside a docstring did not
    turn the module into a syntax error — the same reason the file's own
    no-drive-letter sweep reads the source rather than trusting a reviewer.
    """
    from tools import validate_vs_fortran as V

    paths = V.oracle_paths()
    assert "starter" in paths and "engine" in paths