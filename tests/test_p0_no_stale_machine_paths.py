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
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
PYPROJECT = REPO / "pyproject.toml"
LOCK = REPO / "requirements-lock.txt"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
def _scanned_sources() -> list[Path]:
    """The maintained files held to the rule, in a stable order.

    IN: the parity harness, the oracle build/runtime scripts, and the two
    packaging records at the repo root.

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
    """
    scripts = sorted(
        p for p in TOOLS.rglob("*")
        if p.is_file() and p.suffix in (".py", ".sh", ".txt")
    )
    return [PYPROJECT, LOCK, *scripts]


#: An absolute path rooted in a per-machine location.  ``$HOME``-shaped on both
#: platforms, plus the ``/mnt`` bind this box's checkout lives under.
MACHINE_PATH = re.compile(
    r"(?<![\w.$~-])(/(?:home|mnt|media|Users)/[A-Za-z0-9._+@-]+"
    r"(?:/[A-Za-z0-9._+@%~-]+)*)"
)

#: Phrases that mark a statement as being about ANOTHER machine (or another
#: time), which is what makes naming a path that is not here legitimate.  A
#: provenance record says so; a stale claim, however confident, does not.
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
    """
    claims = []
    for index, line in enumerate(lines):
        for match in MACHINE_PATH.finditer(line):
            path = match.group(1).rstrip(".,;:)'\"")
            claims.append((index, path, _label_window(lines, index)))
    return claims


def _stale_paths_in(lines: list[str]) -> list[tuple[int, str]]:
    """``(line index, path)`` for the paths that are false here and unmarked."""
    return [(index, path) for index, path, window in _path_claims(lines)
            if not Path(path).exists() and not _marked_history(window)]


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
    """
    offenders = []
    for path in _scanned_sources():
        lines = _lines(path)
        for index, stale in _stale_paths_in(lines):
            offenders.append(f"{path.relative_to(REPO)}:{index + 1}: {stale}")
    assert offenders == [], (
        "these lines assert a machine path that does not exist on this box, "
        "with no label within "
        f"{LABEL_WINDOW_LINES} lines saying the statement describes another "
        "machine or another record (accepted labels: "
        f"{', '.join(HISTORY_MARKERS[:6])}...): " + "; ".join(offenders)
        + ". Either the path exists here, or the sentence says which machine / "
        "which record it describes.")


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
#: itself.
VERSION_CLAIM = re.compile(
    r"\b(?P<pkg>numpy|scipy|numba|llvmlite|mpi4py|pytest|python)\s+"
    r"(?P<ver>\d+\.\d+(?:\.\d+)?)\b")

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
                             label: str) -> list[str]:
    """The ``label:<line>`` entries for comment prose naming an unrecorded version.

    Extracted so a self-test can drive the real rule over synthetic text (see
    :func:`test_the_version_rule_can_still_fail`).  Only ``#`` comment lines are
    read: the requirement arrays themselves declare floors, not measurements.
    """
    offenders = []
    for lineno, line in enumerate(lines, 1):
        if not line.lstrip().startswith("#"):
            continue              # the requirement arrays themselves, not prose
        for match in VERSION_CLAIM.finditer(line):
            pkg, ver = match.group("pkg").lower(), match.group("ver")
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


# ---------------------------------------------------------------------------
# 4. the build script's toolchain versions, against the tools
# ---------------------------------------------------------------------------
#: ``<tool> ... <version>`` in a comment, with a comparison-operator tail
#: excluded: "cmake >= 3.15 and < 4 is required" is a REQUIREMENT, not a
#: measurement, and failing it would be the test inventing its own rule.
TOOL_VERSION_CLAIM = re.compile(
    r"\b(?P<tool>cmake|gfortran|gcc|g\+\+|make)\b(?P<tail>[^\n]{0,60}?)"
    r"(?<![\w.])(?P<ver>\d+\.\d+(?:\.\d+)?)\b")

_CONSTRAINT_TAIL = (">=", "<=", "==", "!=", "~>", "~=", ">", "<")


def _tool_banner(tool: str) -> str | None:
    """``<tool> --version`` for the tool ``PATH`` resolves, or ``None``."""
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
    """
    offenders, unchecked = [], []
    for path in _scanned_sources():
        if path.suffix not in (".py", ".sh", ".txt"):
            continue
        found, skipped = _tool_version_offenders(
            _lines(path), path.relative_to(REPO).as_posix())
        offenders += found
        unchecked += skipped
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
# 6. the harness still imports (the fix must not break what P0.12-P0.14 bought)
# ---------------------------------------------------------------------------
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