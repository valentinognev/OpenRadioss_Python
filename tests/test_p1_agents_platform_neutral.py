"""Task P1.0 — ``AGENTS.md`` must be platform-neutral, and must STAY so.

WHY THIS FILE IS A DELIVERABLE AND NOT A VERIFICATION SCRIPT
-----------------------------------------------------------
Task P1.0 rewrote ``AGENTS.md`` to be Linux-first with Windows kept as a
labelled compatibility section (``cb69488``, ``f5c339f``).  Its own test was
written, ran, and was left in the git-ignored scratch directory
(``.superpowers/sdd/task-p1_0_test_agents_md.py``) because that task's ``Files:``
line named ``AGENTS.md`` alone.  So at the moment the rewrite landed **nothing
in the repository stopped the next edit from putting ``C:\\OpenRadioss`` back**,
and an agent contract with no gate is how this repository ended up with a
Windows-only ``AGENTS.md`` in the first place.  This file is that gate, landed.

The plan's own Step 1 test is kept here unchanged (as data, :data:`STALE` /
:data:`REQUIRED`, so it is provably implemented rather than re-typed), and
everything below it is a requirement the same task states in prose and no string
list would catch:

* the ``## STOP`` licensing section is **byte-identical** to ``cf046a3``'s — that
  commit added it deliberately ("a future task reviewed this file specifically
  to confirm a fresh agent would stop"), the one thing a platform rewrite must
  not paraphrase, and it must stay near the TOP of the file or it stops being a
  gate and becomes a section;
* ``AGENTS.md`` matches ``plan/00_ORCHESTRATION.md`` §4, which is the file's own
  declared source of truth — every variable §4.1 defines and every command §4.2
  spells out, with the handful it deliberately does not carry listed as an
  exemption map rather than left as a silent gap;
* **AND the commands it documents are the tool's own.**  Fix round 1: the
  equality with §4.2 was ratifying a command that cannot run — §4.2's
  ``--mode parity --cases … --out …`` reaches
  ``tools/validate_vs_fortran.py``'s argparse as three unrecognized arguments,
  and a gate whose only power on that line was "the two files agree" guaranteed
  the error survived.  Agreement was never the property that mattered.  The
  parity invocation is therefore checked against the harness's **own parser**,
  with ``parity``/``coverage`` stubbed so the check measures the parse and not a
  solver run, and §4.2 is allowed to be wrong only through
  :data:`COMMANDS_CORRECTED_IN_AGENTS`, a declared map that is *itself*
  constrained (a token listed there must NOT appear in ``AGENTS.md``, or the
  escape hatch is just a second exemption list);
* Windows survives as a *labelled* compatibility section: the heading must stay
  (deleting it is the tempting way to satisfy a literal rule) and every
  Windows-shaped path in the file must sit in a window that says it describes the
  other supported machine, using the sibling gate's own discriminator;
* the program vocabulary (``P<phase>.<task>``, waves, reviewers, M700+), the
  model policy (``space-bunny``, never Fast) and the domain rules the plan says
  to keep **verbatim** are still present — the domain rules matched on their
  load-bearing content, because five short tails let "must book" become "should
  book" and "NEVER guess" become "Try not to guess";
* no bare suite count — ``tests/test_p0_record_suite_counts.py``'s own
  ``unbound()`` decides that, so the rule is the repository's, not a copy.

AND THE CHECK CAN STILL FAIL
----------------------------
The last test drives the same checks over the **pre-P1.0** contract and requires
them to report, so "this gate is green" cannot be a property of a checker that
accepts everything.  It uses ``git show cf046a3:AGENTS.md`` where the history is
reachable, and a verbatim excerpt of that file's ``## Environment`` /
``## Terminal rules`` sections so the proof still runs in a shallow CI checkout
(``actions/checkout@v4`` defaults to ``fetch-depth: 1``, where ``cf046a3`` is
not in the clone at all).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENTS = REPO / "AGENTS.md"
ORCHESTRATION = REPO / "plan" / "00_ORCHESTRATION.md"

#: The commit that put the licensing gate where a fresh agent reads it, and the
#: last state of ``AGENTS.md`` before the platform rewrite.
STOP_COMMIT = "cf046a3"
PRE_P1_0_COMMIT = "cf046a3"

#: The plan's Step 1 list (plan/02_phase1_foundation.md, Task P1.0), verbatim —
#: stale first — plus every OTHER Windows-only assumption that file carried and
#: a string list of three would miss.  Each entry below is quoted from
#: ``cf046a3:AGENTS.md`` and each is a *fact about the old machine* or a *rule
#: that only holds on it*; none of them may come back.
STALE = (
    # the interpreter, spelled two ways, and the bare-name rule beside them
    r".venv\Scripts\python.exe",
    r".venv\Scripts\pip.exe",
    "No WSL, no Git Bash, no Unix commands",
    "(ls/rm/cat/grep)",
    "Python 3.12.3 in `.venv`",       # a version in prose; §[B] is the pin
    # the shell
    "Windows 11, PowerShell",
    ".ps1 helpers",
    "-Wait` loops",
    "Windows backslash paths everywhere",
    "No `&&`, no pipes, no redirection",   # pipes are fine on POSIX
    # paths asserted about the old machine
    r"C:\OpenRadioss",
    r"C:\OpenRadioss_old",
    r"C:\Users\pmqua",
    r"E:\openradioss_run",
    r"E:\foxcore_data",
    # backslash spellings of repo-relative paths
    r"tests\data\rd_decks",
    r"examples\tensile_bar",
    r"tests\test_m42_",
    # and the pre-Phase-1 vocabulary the rewrite had to retire
    "Work ONE milestone per conversation",
    "M1–M41",
    "@docs/STATE.md",
    "(validate_vs_fortran.py, validation_data/, lspp_check.py, benchmark.py)",
    "~1105 tests",
)

#: The Windows venv layout in **any** separator, which is the fix for fix round
#: 1's Finding 4: ``.venv/Scripts/python.exe`` is the single most damaging
#: spelling this file could take — a Linux agent handed an interpreter that does
#: not exist here — and it passed BOTH gates, ``STALE`` above because it pins
#: backslash forms only and the sibling rule's :data:`WINDOWS_PATH` because that
#: required a drive letter.  (Both are now closed: the sibling rule carries a
#: ``Scripts``-segment alternative.  These patterns stay because a contract gate
#: that leans on another file's scanner is a contract gate with one more owner.)
#: Patterns, not substrings, because "shape-agnostic" is the whole point;
#: :data:`POSIX_PATHS` pins the direction this must NOT catch.
STALE_PATTERNS = (
    re.compile(r"\.venv[/\\]Scripts[/\\]"),
    # the same layout named without its parent — the lookbehind excludes a word
    # character or a dot, and NOT a separator, so this fires on the pre-P1.0
    # file's own ``.venv\Scripts\python.exe`` as well as on `Scripts/python.exe`
    re.compile(r"(?<![\w.])Scripts[/\\](?:python|pip)\.exe"),
)

#: The POSIX spellings the patterns above must leave alone.  Pinned HERE, in the
#: gate, rather than argued in prose: a rule that cannot tell
#: ``.venv/Scripts/python.exe`` from ``.venv/bin/python`` is a rule that cries
#: wolf, and the two are indistinguishable by shape without a drive letter.
POSIX_PATHS = (
    ".venv/bin/python",
    "tests/data/rd_decks",
    "examples/tensile_bar",
    "tools/validate_vs_fortran.py",
    "tools/oracle/oracle_env.sh",
)

#: The plan's Step 1 required list, verbatim, plus the tokens that carry the
#: Linux-first contract (first row) and the section the plan asks for (rest).
REQUIRED = (
    "OR_SRC", "OR_ROOT", "PYRADIOSS_HM_CFG", '-m "not slow"',
    # §Environment: the interpreter is resolved, never hardcoded
    "$PYTHON", ".venv/bin/python", 'export PYTHON="$PWD/.venv/bin/python"',
    "$OR_BUILD", "OR_STARTER", "OR_ENGINE", "tools/oracle/oracle_env.sh",
    "PYRADIOSS_RD_DECKS", "PYRADIOSS_BACKEND", "PYRADIOSS_ORACLE_REQUIRED",
    # §The oracle and parity evidence
    "READ-ONLY",
    # the PARITY SUBVERB, not §4.2's `--mode parity`: the tool has no such flag
    "validate_vs_fortran.py parity",
    # the labelled compatibility section, and the vocabulary it must teach
    "## Windows compatibility", "compatibility", "§[A]", "§[B]", "Scripts",
    "space-bunny", "Fast", "M700", "P1.0", "wave", "reviewer",
)

#: Domain rules the plan says to keep **verbatim**, as whole rules rather than
#: tails.  Fix round 1's Finding 3 measured what "tail" bought: with five short
#: substrings, ``must book`` → ``should book`` (the energy-ledger rule),
#: ``NEVER guess`` → ``Try not to guess`` (the rule the plan names first) and a
#: dropped ``≥60 s`` all passed every check in the file.  Each fragment below was
#: chosen because no rewording of the sentence carrying it can keep the fragment
#: while changing what the sentence says — and each sits inside a single
#: physical line, so re-wrapping the file does not break it.
DOMAIN_RULES = {
    "the Fortran wins": (
        "NEVER guess Fortran semantics",
        "read the Fortran, don't recall it",
        "the Fortran wins",
    ),
    "a parity run is the evidence": (
        "Numerics changes REQUIRE validation evidence",
        "a parity run against",
        "the real Fortran solver",
    ),
    "never weaken a test": (
        "Never weaken a test",
    ),
    "energy accounting is load-bearing": (
        "Energy accounting is load-bearing",
        "must book its work",
        "in the EN ledger",
    ),
    "pin the backend": (
        "pin the backend: `PYRADIOSS_BACKEND=numpy`",
    ),
    "the 60 s slow-test threshold": (
        "Mark any test ≥60 s serial with `@pytest.mark.slow`",
    ),
}

#: §4.1 variables ``AGENTS.md`` deliberately does not carry, each with the reason
#: it does not have to.  This is an EXEMPTION MAP, not a gap: adding a row to
#: §4.1 without adding it here fails the test below, so the decision to leave a
#: variable out of the contract has to be taken consciously.
VARIABLES_NOT_IN_AGENTS = {
    "PYRADIOSS_SPMD_LOG_ALL":
        "§4.1's own row says 'unset', and §Environment tells the reader the "
        "remaining variables are defined in §4.1 — a diagnostic nobody sets "
        "does not belong in the imperative list",
}

#: The MIRROR of :data:`VARIABLES_NOT_IN_AGENTS`, for the direction round 1 only
#: checked one way: a variable ``AGENTS.md`` sends the reader to §4.1 for which
#: §4.1 has no row.  Finding 2 — ``$OR_BUILD`` was attributed to §4.1, which does
#: not define it, and a check that only asked "does §4.1's variable appear
#: somewhere in AGENTS.md" cannot see an excess.  An agent sent to a section that
#: does not define the variable is strictly worse off than one never sent, so
#: the correct value here is empty and an entry needs the §4.1 row it claims.
VARIABLES_ATTRIBUTED_WITHOUT_A_4_1_ROW: dict[str, str] = {}

#: §4.2 commands ``AGENTS.md`` §Commands does not spell out, same reasoning.
COMMANDS_NOT_IN_AGENTS = {
    "--mode coverage":
        "§Commands carries the parity command, which is the one a numerics "
        "change requires; coverage is a phase-gate command driven from the "
        "plan, not from the contract",
    "tools/census.py":
        "a milestone regenerates the keyword census through the plan's own "
        "task, not from the agent contract",
    "--render":
        "same: it is an argument of the census command above, not a rule",
}

#: §4.2 commands this file **corrects**, each with the tool that says so.
#:
#: The map exists because the alternative — round 1's plain equality with §4.2 —
#: ratified a command that cannot run.  §4.2 line 153 carries
#: ``validate_vs_fortran.py --mode parity --cases <list> --out …``; the tool's
#: parser is ``{parity,coverage} …`` with ``--only`` and no ``--out`` at all
#: (``tools/validate_vs_fortran.py:2251-2267``), so the contract handed every
#: future agent a usage error on the one command a numerics change is *required*
#: to run.  Two constraints keep this from being a laundering hatch, and both
#: are asserted:
#:
#: 1. a token listed here must NOT appear in ``AGENTS.md`` — otherwise it is an
#:    exemption wearing a correction's clothes; and
#: 2. the corrected shape is verified against the tool's own parser by
#:    :func:`test_the_documented_parity_command_is_the_tools_own_interface`, so
#:    "corrected" is measured, not asserted.
#:
#: ``plan/00_ORCHESTRATION.md`` is not this gate's file to edit.  §4.2 lines 153
#: and 154 are the defect and are a finding for the plan's owner.
COMMANDS_CORRECTED_IN_AGENTS = {
    "--mode parity":
        "tools/validate_vs_fortran.py's parser is `parity`/`coverage` "
        "SUBCOMMANDS: `parity --only <names>`. `--mode`, `--cases` and `--out` "
        "are not flags of that tool (validate_vs_fortran.py:2251-2267), and "
        "parity_results.json is written under <workdir>, not to a chosen path "
        "(validate_vs_fortran.py:81). plan/00_ORCHESTRATION.md:153-154 still "
        "carry the wrong commands — the plan's owner has to fix them.",
}

#: Verbatim from ``cf046a3:AGENTS.md`` — lines 30-47 (``## Environment``) and
#: 65, 69 (``## Terminal rules``).  Quoted rather than derived from git so the
#: fail-then-pass proof below runs in a shallow CI checkout, where
#: ``git show cf046a3:AGENTS.md`` cannot resolve.  ``test_…_still_fail_on_the
#: _pre_p1_0_contract`` checks the quote against the real commit wherever the
#: history IS reachable, so the excerpt cannot silently drift.
PRE_P1_0_EXCERPT = """\
- Windows 11, PowerShell. No WSL, no Git Bash, no Unix commands (ls/rm/cat/grep).
- Python 3.12.3 in `.venv` — this box's interpreter, pinned in
  `requirements-lock.txt` §[B]. NEVER bare `python` or `pip`. Always exact
  paths:
  - `.venv\\Scripts\\python.exe`
  - `.venv\\Scripts\\pip.exe`
- All dependencies are preinstalled (numpy 2.4.6, scipy 1.18.0, numba 0.66.0,
  pytest 9.1.1 — exact pins in `requirements-lock.txt`). Do NOT create a venv,
  do NOT run pip install, unless the user explicitly asks.
- The generic `/MAT`–`/PROP` readers need the hm_cfg_files CFG tree; it is
  auto-found at `C:\\OpenRadioss\\hm_cfg_files` (env `PYRADIOSS_HM_CFG` overrides).
- Corpus-deck tests use `tests\\data\\rd_decks` (vendored); env
  `PYRADIOSS_RD_DECKS` points at a fuller extract (see the README there).
- External harvested reference corpus for benchmarking & validation:
  `C:\\Users\\pmqua\\PycharmProjects\\rad_examples_db` (decks, manifest, benchmarks).
- READ-ONLY locations — never write, delete, or extract in place:
  `C:\\OpenRadioss` (reference Fortran install + source), `C:\\OpenRadioss_old`,
  and everything under `E:\\` (`E:\\openradioss_run`, `E:\\foxcore_data`).
- No `&&`, no pipes, no redirection — separate commands.
- Windows backslash paths everywhere.
- START HERE: @docs/STATE.md — what is implemented (M1–M41), the current
- Work ONE milestone per conversation. **Commit at every green sub-step**
  (`git add -A`, `git commit`) — session and quota interruptions are normal here;
- Layout: `pyradioss/` (common, input, model, starter, engine, elements,
  materials, failure, contact, output, implicit, accel, gui), `tests/`
  (+ `tests/data/rd_decks` vendored corpus decks), `tools/`
  (validate_vs_fortran.py, validation_data/, lspp_check.py, benchmark.py),
  `examples/` (runnable decks).
- the fast tier is `~1105 tests`; the milestone table is M1–M41.
"""

#: ``| `OR_SRC` | … |`` — the first cell of §4.1's table.
_VARIABLE_ROW = re.compile(r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|", re.M)

#: ``$VAR`` as written in prose.
_VARIABLE_NAME = re.compile(r"\$([A-Z][A-Z0-9_]*)")

#: The sentence in §Environment that points the reader at §4.1 for a variable.
#: Matched against :func:`_sentences`, not with a character class: ``§4.1``
#: contains a full stop, so ``[^.]*`` would cut the sentence in half — which is
#: how the sentence saying where ``$OR_BUILD`` really lives parses as a claim
#: that says nothing.
_SECTION_4_1 = "§4.1"

#: ``$PYTHON tools/validate_vs_fortran.py …`` — the documented parity
#: invocation, wherever in the contract it is stated.  ``<placeholder>`` is what
#: makes the trailing token substitutable, so the line can be handed to the
#: tool's parser as a real argv.
_PARITY_INVOCATION = re.compile(
    r"tools/validate_vs_fortran\.py(?P<args>[^\n`]*)")
_PLACEHOLDER = re.compile(r"<[^>\n]*>")

#: The one example name the gate substitutes for a documented ``<…>``, and it has
#: to be a real ``examples/`` directory: a contract command naming a case that
#: does not exist is the same defect this test exists to catch, one level up.
SUBSTITUTE_CASE = "tensile_bar"

#: ``$PYTHON -m pytest`` / ``--mode parity`` / ``--render``: the distinctive
#: tokens of §4.2's commands, not the whole incantation, so a command written
#: with different flags is still recognised as present.
_COMMAND_TOKEN = re.compile(
    r"\$PYTHON\s+([A-Za-z0-9_./-]+)|(--mode\s+\w+|--collect-only|--render)")


def _agents_text() -> str:
    return AGENTS.read_text(encoding="utf-8")


def _orchestration_text() -> str:
    return ORCHESTRATION.read_text(encoding="utf-8")


def _lost_domain_rules(section: str) -> dict[str, list[str]]:
    """``{rule: [fragment, …]}`` for every rule §Domain rules no longer states.

    One function, used by the check and by its own non-vacuity proof, so the
    mutation test cannot pass by drifting away from what the gate measures.
    """
    return {rule: [fragment for fragment in fragments
                   if fragment not in section]
            for rule, fragments in DOMAIN_RULES.items()
            if not all(fragment in section for fragment in fragments)}


def _section(text: str, heading: str) -> str:
    """The body of ``heading`` up to the next heading.

    Fenced blocks are skipped while looking for that heading, and they have to
    be: §4.2's command block opens with ``# edit loop``, which is a shell
    comment, and stopping there would silently truncate the section this file
    parses its commands out of.
    """
    rest = text[text.index(heading) + len(heading):]
    fenced, offset = False, 0
    for line in rest.splitlines(keepends=True):
        offset += len(line)
        if line.strip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} ", line.strip()):
            return rest[:offset - len(line)]
    return rest


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True)


def _flat(text: str) -> str:
    """``text`` with its line wrapping collapsed, so a sentence is one string."""
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    """The sentences of ``text``, wrapping collapsed, split at ``". "``.

    Split on the dot AND its trailing space rather than on every dot, because
    this file's cross-references are ``§4.1``, ``§[B]`` and ``§1.3, lines
    58-73`` — a bare-dot split cuts every one of them in half.
    """
    return re.split(r"(?<=\.)\s+", _flat(text))


def _documented_parity_argvs(text: str) -> list[list[str]]:
    """Every ``validate_vs_fortran.py …`` invocation in the contract, as an argv.

    ``$PYTHON`` is dropped — it is the shell's variable, not the tool's — and
    every ``<placeholder>`` becomes :data:`SUBSTITUTE_CASE`, so what comes out is
    a command line a reader could have typed.
    """
    argvs = []
    for match in _PARITY_INVOCATION.finditer(text):
        args = _PLACEHOLDER.sub(SUBSTITUTE_CASE, match.group("args"))
        argv = [token for token in shlex.split(args) if token != "$PYTHON"]
        if argv:
            argvs.append(argv)
    return argvs


def _harness():
    """``tools/validate_vs_fortran.py`` loaded by path, with its two modes stubbed.

    Loaded rather than imported so the gate measures the file that actually
    ships.  ``parity``/``coverage`` are replaced with recorders because a parity
    run launches two solvers per example and this check is about whether the
    documented command line **parses** — the parse is ``main``'s job, the run is
    not this test's business.  Returns ``(call, recorded)``.
    """
    spec = importlib.util.spec_from_file_location(
        "_p1_validation_harness", REPO / "tools" / "validate_vs_fortran.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recorded: list[tuple[str, object]] = []
    module.parity = lambda args: recorded.append(("parity", args)) or 0
    module.coverage = lambda args: recorded.append(("coverage", args)) or 0
    return module.main, recorded


# ---------------------------------------------------------------------------
# 1. the plan's Step 1 test, unchanged
# ---------------------------------------------------------------------------
def test_agents_md_has_no_windows_only_assumptions():
    """The plan's own check, with its three stale and four required strings.

    Kept verbatim in substance and extended (:data:`STALE` /
    :data:`REQUIRED`), because a gate that only remembers the three strings the
    plan happened to name leaves the other eighteen Windows-only assumptions of
    ``cf046a3`` free to come back one at a time.
    """
    text = _agents_text()
    stale = [s for s in STALE if s in text]
    assert stale == [], (
        f"AGENTS.md assumes the pre-migration machine again: {stale}. The "
        "contract is Linux-first; Windows 11 is a labelled compatibility "
        "target (see §Windows compatibility), and no interpreter, shell or "
        "path may be spelled for one platform only.")

    # ... and in the spellings a substring list cannot hold: the Windows venv
    # layout with a forward slash is the same defect as the backslash form and
    # the sibling gate cannot see it either (Finding 4).
    shaped = sorted(pattern.pattern for pattern in STALE_PATTERNS
                    if pattern.search(text))
    assert shaped == [], (
        f"AGENTS.md names the WINDOWS venv layout again, in a spelling the "
        f"string list above misses: {shaped}. `.venv/Scripts/python.exe` does "
        "not exist on this box, and on a drive-less path the string list could "
        "not see it — that was the exact hole task P1.0 exists to close.")
    missing = [r for r in REQUIRED if r not in text]
    assert missing == [], (
        f"AGENTS.md no longer names {missing}: the rewrite is not allowed to "
        "lose content, only to restate it platform-neutrally.")


def test_a_posix_path_is_not_a_windows_path():
    """The other half of :data:`STALE_PATTERNS`, in the gate itself.

    ``STALE_PATTERNS`` cannot tell ``.venv/Scripts/python.exe`` from
    ``.venv/bin/python`` by shape — there is no drive letter to go on — so it is
    told by segment.  A rule that reaches the segment by accident is a rule that
    reports the contract's own paths, and the outcome is the one this file has
    refused three times: everyone ignores the gate.  So the legitimate POSIX
    spellings are pinned HERE, and a future widening of the patterns has to
    break this test before it reaches ``AGENTS.md``.
    """
    for path in POSIX_PATHS:
        matched = [pattern.pattern for pattern in STALE_PATTERNS
                   if pattern.search(path)]
        assert matched == [], (
            f"{path!r} is this box's own path and STALE_PATTERNS flags it as a "
            f"Windows spelling ({matched}); the rule has to key on the "
            "`Scripts` segment, not on the separator")

    for planted in (".venv/Scripts/python.exe", ".venv\\Scripts/pip.exe",
                    "Scripts/python.exe", "run Scripts\\pip.exe"):
        assert any(pattern.search(planted) for pattern in STALE_PATTERNS), (
            f"{planted!r} is the Windows venv layout spelled forward or mixed, "
            "and the shape-agnostic patterns must catch it")


# ---------------------------------------------------------------------------
# 2. the rest of Step 3's required content, which no string list states
# ---------------------------------------------------------------------------
def test_agents_md_carries_the_linux_first_environment_contract():
    """The environment is stated as a variable to resolve, not a path to copy."""
    text = _agents_text()
    for token in (".venv/bin/python", "$PYTHON", "$OR_BUILD", "OR_STARTER",
                  "OR_ENGINE", "tools/oracle/oracle_env.sh",
                  "PYRADIOSS_RD_DECKS", "PYRADIOSS_BACKEND"):
        assert token in text, f"AGENTS.md no longer names {token!r}"

    # ... and it says where the resolver is, which is the rule that stops the
    # next agent from hardcoding a path back in.
    assert "pyradioss/paths.py" in text, (
        "AGENTS.md must point at pyradioss/paths.py as the one place external "
        "paths are resolved — the sentence that makes a hardcoded path a rule "
        "violation rather than a convenience")


def test_agents_md_names_every_variable_the_environment_contract_defines():
    """``AGENTS.md`` and ``plan/00_ORCHESTRATION.md`` §4.1 must agree — BOTH ways.

    ``AGENTS.md`` declares §4.1 "the single source of truth for every variable
    named below", so the two files are a contract and a copy of it.  A copy rots
    silently; this is what stops that.

    Round 1 checked one direction only — every §4.1 row named in the contract —
    and that is how ``$OR_BUILD`` sailed through: the contract sent the reader to
    §4.1 for it and §4.1 has no such row (``grep -n OR_BUILD
    plan/00_ORCHESTRATION.md`` → nothing), which is strictly worse than never
    mentioning the variable.  So both directions are checked now, and both are
    scoped to §Environment rather than the whole file, because a variable named
    in §Project is not the environment contract being defined:

    * every §4.1 row is named in §Environment, unless declared in
      :data:`VARIABLES_NOT_IN_AGENTS` with a reason;
    * every variable the §4.1-attribution SENTENCE sends the reader there for is
      a row §4.1 actually has — the declared excess map is
      :data:`VARIABLES_ATTRIBUTED_WITHOUT_A_4_1_ROW`, and its correct value is
      empty.
    """
    variables = set(_VARIABLE_ROW.findall(_section(_orchestration_text(), "### 4.1")))
    assert len(variables) >= 9, (
        "§4.1's table stopped parsing as a variable table "
        f"(found {sorted(variables)}); this test would be vacuous, not green")

    environment = _section(_agents_text(), "## Environment")
    missing = {v for v in variables if f"${v}" not in environment}
    assert missing == set(VARIABLES_NOT_IN_AGENTS), (
        "AGENTS.md and plan/00_ORCHESTRATION.md §4.1 have drifted: "
        f"{sorted(missing)} is in one and not the other. Name it in AGENTS.md, "
        f"or declare it in VARIABLES_NOT_IN_AGENTS with the reason it does not "
        f"belong in the contract (declared: {sorted(VARIABLES_NOT_IN_AGENTS)})")

    # the reverse direction: the attribution itself must be true
    attributing = [sentence for sentence in _sentences(environment)
                   if f"defined in {_SECTION_4_1}" in sentence]
    assert attributing, (
        f"§Environment no longer contains a sentence attributing variables to "
        f"{_SECTION_4_1}; this test would be vacuous, not green")
    sent = set(_VARIABLE_NAME.findall(attributing[0]))
    assert len(sent) >= 4, (
        f"the {_SECTION_4_1} attribution sentence parses as {sorted(sent)}; it "
        "must carry the variables it points at, or the reverse check below is "
        "measuring an empty set")

    excess = sent - variables - set(VARIABLES_NOT_IN_AGENTS)
    assert excess == set(VARIABLES_ATTRIBUTED_WITHOUT_A_4_1_ROW), (
        f"AGENTS.md sends the reader to {_SECTION_4_1} for {sorted(excess)}, and "
        f"{_SECTION_4_1} has no such row (it defines {sorted(variables)}). Point "
        "each of them at where it really is — §4.1 for the §4.1 variables, "
        "`tools/oracle/oracle_env.sh` for `$OR_BUILD` — or declare it in "
        "VARIABLES_ATTRIBUTED_WITHOUT_A_4_1_ROW with the reason (declared: "
        f"{sorted(VARIABLES_ATTRIBUTED_WITHOUT_A_4_1_ROW)})")

    # ... and the variable that has no §4.1 row must still be told where it lives
    or_build = [sentence for sentence in _sentences(environment)
                if "$OR_BUILD" in sentence]
    assert or_build, (
        "$OR_BUILD is gone from §Environment; it is required — "
        "tools/oracle/oracle_env.sh:56 hard-fails with `:?` when it is unset")
    assert any("tools/oracle/oracle_env.sh" in sentence
               for sentence in or_build), (
        f"§Environment names $OR_BUILD without saying where it is defined, "
        f"which is what Finding 2 was: {or_build}")


def test_agents_md_carries_the_commands_the_environment_contract_defines():
    """Same two-sided comparison, over §4.2's commands — with one asymmetry.

    §4.2 is where a new agent copies its first command from, so a command that
    exists only in the plan is one this repository can lose without noticing.

    **Both** files being wrong is the case round 1's plain equality could not
    see, and §4.2 line 153 is it: ``--mode parity --cases <list> --out …``
    reaches the tool's parser as three unrecognized arguments.  So a §4.2 token
    may be absent from the contract only if it is declared, with a reason, in
    :data:`COMMANDS_NOT_IN_AGENTS` (not carried on purpose) or in
    :data:`COMMANDS_CORRECTED_IN_AGENTS` (corrected because the tool says so —
    and the corrected shape is checked against that tool by
    :func:`test_the_documented_parity_command_is_the_tools_own_interface`).

    A token declared as *corrected* must NOT appear in ``AGENTS.md``.  Without
    that, the map is a second exemption list wearing the first one's coat, and
    the gate would once again be able to certify an error.
    """
    commands = _section(_orchestration_text(), "### 4.2")
    tokens = set()
    for script, flag in _COMMAND_TOKEN.findall(commands):
        tokens.add(script or flag)
    assert len(tokens) >= 5, (
        f"§4.2's command block stopped parsing ({sorted(tokens)}); this test "
        "would be vacuous, not green")

    text = _agents_text()
    missing = {t for t in tokens if t not in text}
    undeclared = missing - set(COMMANDS_NOT_IN_AGENTS) \
        - set(COMMANDS_CORRECTED_IN_AGENTS)
    assert undeclared == set(), (
        "AGENTS.md §Commands and plan/00_ORCHESTRATION.md §4.2 have drifted: "
        f"{sorted(undeclared)} is in one and not the other. Carry the command, "
        "or declare it in COMMANDS_NOT_IN_AGENTS (not carried on purpose) or "
        "in COMMANDS_CORRECTED_IN_AGENTS (the tool says §4.2 is wrong — say "
        "which tool and where)")

    for token, reason in COMMANDS_CORRECTED_IN_AGENTS.items():
        assert token not in text, (
            f"{token!r} is declared as a §4.2 mistake that AGENTS.md corrects, "
            "but it is in AGENTS.md. Either the correction is undone — in which "
            "case delete the declaration — or the contract is still shipping "
            "the flag the tool does not have.")
        assert "validate_vs_fortran.py" in reason and reason.strip(), (
            f"the correction of {token!r} names no tool to be checked against; "
            "'corrected' has to mean measured, not asserted")


def test_the_documented_parity_command_is_the_tools_own_interface():
    """THE fix for Finding 1: the contract's command must be the tool's command.

    ``AGENTS.md`` is an agent contract, so a command in it that errors is a
    defect with a whole workflow behind it — an agent burns a cycle on a usage
    error, and one that reads the failure as "parity cannot be run here" reaches
    exactly the wrong conclusion about a file whose job is to make the oracle
    usable.  Round 1's gate could not catch it, because its deepest check was
    "does ``AGENTS.md`` equal §4.2" and §4.2 carried the same broken command:
    agreement was ratified as truth.

    So the check is against ``tools/validate_vs_fortran.py``'s **own argparse**.
    The command line is lifted out of the contract, ``$PYTHON`` and the ``<…>``
    placeholder are resolved, and the harness is invoked with ``parity`` and
    ``coverage`` replaced by recorders: what is measured is whether the flags
    EXIST, not whether a solver run succeeds, and a run would make a gate this
    load-bearing cost 240 s per example.
    """
    assert (REPO / "examples" / SUBSTITUTE_CASE).is_dir(), (
        f"the gate substitutes {SUBSTITUTE_CASE!r} for a documented <…> and "
        "that example does not exist, so this test cannot measure anything")

    call, recorded = _harness()
    argvs = _documented_parity_argvs(_agents_text())
    assert argvs, (
        "AGENTS.md no longer spells a validate_vs_fortran.py invocation; the "
        "parity command is required content, not a pointer to the skill")

    for argv in argvs:
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                rc = call(argv)
        except SystemExit as exc:
            pytest.fail(
                f"AGENTS.md documents {argv!r}, and the tool's own parser "
                f"REJECTS it (exit {exc.code}). The contract hands every future "
                "agent a command that cannot run; the real interface is "
                "`tools/validate_vs_fortran.py parity --help`:\n"
                f"{stderr.getvalue().rstrip()}")
        assert rc == 0, (
            f"the documented command {argv!r} parsed but did not dispatch "
            f"(rc={rc}); it must name a mode the harness has")
        assert recorded, (
            f"{argv!r} parsed and dispatched nothing, so no mode was reached")
        mode, _ = recorded.pop()
        assert mode == "parity", (
            f"{argv!r} dispatched {mode!r}; a numerics change needs parity")

    # the line in §Commands is the one a reader copies, so it must be the
    # complete command: no `--only` means "every example in examples/", which is
    # a whole-corpus parity run, not the targeted evidence the rule asks for.
    full = _documented_parity_argvs(_section(_agents_text(), "## Commands"))
    assert full, (
        "§Commands no longer carries a validate_vs_fortran.py invocation; the "
        "oracle section pointing at §Commands instead of stating it there "
        "leaves the contract's own command one cross-reference away from an "
        "agent who never opens §Commands")
    call, recorded = _harness()
    call(full[0])
    assert recorded[0][1].only, (
        f"the §Commands invocation {full[0]!r} names no case to run; `--only` "
        "is how the harness is told which examples/ entry to compare, and "
        "without it the run is every example in the corpus")

    # the negative direction, in the same breath: the shape round 1 shipped must
    # still be rejected by the very parser this test consults, so a future edit
    # cannot quietly restore it
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            call(["--mode", "parity", "--cases", SUBSTITUTE_CASE,
                  "--out", "/tmp/parity_p1.json"])
    except SystemExit as exc:
        assert exc.code != 0, (
            "the parser accepted `--mode parity --cases … --out …`; this test "
            "measures the tool's interface and that shape is not it")
    else:
        pytest.fail(
            "tools/validate_vs_fortran.py now accepts `--mode parity --cases … "
            "--out …`, so AGENTS.md and §4.2's old spelling may have become "
            "correct after all — re-measure and update both")


def test_agents_md_keeps_windows_as_a_labelled_compatibility_section():
    """Windows must survive as a SECTION, not as the file's assumption.

    Two halves, and both are needed:

    * the section is still there and still teaches a Windows agent something
      actionable — ``§[A]`` is that box's pin set, ``§[B]`` this one's, and the
      ``Scripts`` variant of the interpreter.  Without this half the file could
      be made "platform-neutral" by deleting the compatibility story, which is
      not neutrality, it is losing the second supported platform;
    * every Windows-shaped path that IS in the file sits in a window that says
      it describes the other machine, decided by the sibling gate's own
      discriminator (``_marked_foreign_machine`` over
      ``_windows_path_claims``) rather than by a second, vaguer copy of it.

    The second half is at zero today and that is stated, not hidden: P1.0 removed
    every literal.  It is the guard for the next edit, and the first half is what
    stops "fixing" it by deleting the section.
    """
    lines = _agents_text().splitlines()
    headings = [line for line in lines if line.startswith("## ")]
    assert any("Windows" in h for h in headings), (
        f"no heading names Windows; the supported second platform is gone: "
        f"{headings}")

    section = _section(_agents_text(), "## Windows compatibility")
    for token in ("§[A]", "§[B]", "Scripts", "compatibility"):
        assert token in section, (
            f"§Windows compatibility no longer carries {token!r}: a Windows "
            "agent reading the contract must still learn the pin sections and "
            "the interpreter spelling, and the section must still say it is a "
            "compatibility target rather than the assumption")

    from tests.test_p0_no_stale_machine_paths import (
        _marked_foreign_machine, _windows_path_claims)

    unlabelled = [(index + 1, path)
                  for index, path, window in _windows_path_claims(lines)
                  if not _marked_foreign_machine(window)]
    assert unlabelled == [], (
        "AGENTS.md names a Windows path with nothing saying it describes the "
        f"other supported machine: {unlabelled}. Either frame it as the "
        "compatibility target, or name the tree through $OR_SRC / $HOME, which "
        "is true on every box.")


def test_agents_md_states_the_model_policy_and_the_program_vocabulary():
    """The file teaches the vocabulary the program actually runs on."""
    text = _agents_text()
    for token in ("space-bunny", "Fast", "M700", "P1.0", "wave", "reviewer"):
        assert token in text, f"AGENTS.md no longer teaches {token!r}"


def test_agents_md_keeps_the_domain_rules():
    """The plan says keep the domain rules verbatim; paraphrasing is changing.

    These are the most load-bearing sentences in the file — this project's whole
    review culture rests on them — so the check is scoped to §Domain rules (a
    fragment quoted anywhere else cannot satisfy a rule) and matched on each
    rule's load-bearing CONTENT, per :data:`DOMAIN_RULES`.

    Round 1 pinned five short tails and was measured against three mutations of
    the real file: ``must book its work`` → ``should book its work``,
    ``NEVER guess`` → ``Try not to guess``, and the ``≥60 s`` threshold deleted
    all passed every check in the file.  ``Never weaken a test`` →
    ``Try not to weaken a test`` was the only one caught, and only because its
    tail happened to be the head.  So the fragments here are the obligation
    words themselves, not the words that follow them.
    """
    headings = [line for line in _agents_text().splitlines()
                if line.startswith("## ")]
    assert any("Domain rules" in h for h in headings), (
        f"§Domain rules is gone: {headings}")

    section = _section(_agents_text(), "## Domain rules")
    lost = _lost_domain_rules(section)
    assert lost == {}, (
        f"the domain rules were lost in the rewrite: {lost}. These are "
        "obligations, not wording — 'must' is not 'should', 'NEVER' is not "
        "'try not to', and a threshold that is not named is not a threshold. "
        "Restore the sentence, or change the rule deliberately and with a "
        "maintainer's decision recorded; the domain rules are not this gate's "
        "to relax.")


def test_agents_md_states_no_bare_suite_count():
    """The repository's own rule, imported rather than copied."""
    from tests.test_p0_record_suite_counts import unbound

    lines = _agents_text().splitlines()
    assert unbound(lines) == [], (
        "AGENTS.md states a suite count with nothing binding it to a moment; "
        f"lines {[i + 1 for i in unbound(lines)]} — cite the command instead")


# ---------------------------------------------------------------------------
# 3. the section that must survive untouched, and stay a gate
# ---------------------------------------------------------------------------
def test_the_stop_licensing_section_is_unchanged_and_still_a_gate():
    """Byte-identical to ``cf046a3``'s, and still near the top of the file.

    Two properties, because either alone is gameable.  Byte-identity stops the
    platform rewrite paraphrasing the notice an agent is expected to stop on;
    the position stops the *next* rewrite burying it — a licensing gate at the
    bottom of a 199-line contract is a section, and the whole point of
    ``cf046a3`` was that a fresh agent meets it in the first screen.

    Skipped where the history is unreachable (a ``fetch-depth: 1`` CI
    checkout), and never silently passed: the tokens it protects are asserted
    unconditionally in the next test.
    """
    if shutil.which("git") is None or _git("cat-file", "-e",
                                           f"{STOP_COMMIT}:AGENTS.md").returncode:
        pytest.skip(f"{STOP_COMMIT} is not reachable in this checkout")

    now = _section(_agents_text(), "## STOP")
    was = _section(_git("show", f"{STOP_COMMIT}:AGENTS.md").stdout, "## STOP")
    assert now.rstrip() == was.rstrip(), (
        "the ## STOP licensing gate was changed by the platform rewrite; it is "
        "the section a fresh agent is expected to stop on, and commit "
        f"{STOP_COMMIT} added it verbatim. First difference:\n"
        + "\n".join(f"  {n}: -{a!r}\n  {n}: +{b!r}"
                    for n, (a, b) in enumerate(
                        zip(was.splitlines(), now.splitlines()), 1)
                    if a != b)[:2000])

    lines = _agents_text().splitlines()
    stop_heading = "## STOP — licensing gate: no new upstream-derived code"
    assert lines.index(stop_heading) <= 15, (
        f"## STOP has moved to line {lines.index(stop_heading) + 1}; commit "
        "cf046a3 put it there so a fresh agent meets it before anything else")


def test_the_stop_section_still_names_its_own_weak_enforcement():
    """The one part of the gate that is a *fact about this repo* must survive.

    Unconditional, and deliberately so: it is what keeps the previous test from
    becoming a gate that only runs where the history is.
    """
    text = _agents_text()
    for token in ("xfail(strict=True)", "an agent may not record one",
                  "GATING, UNRESOLVED", "tests/test_p0_licensing.py"):
        assert token in text, f"## STOP no longer says {token!r}"


# ---------------------------------------------------------------------------
# 4. the check can still fail
# ---------------------------------------------------------------------------
def test_the_platform_neutral_checks_still_fail_on_the_pre_p1_0_contract():
    """Every check above, driven over the contract as it was, must report.

    A gate nobody has watched fail is a gate nobody has watched.  Two sources
    for the same text: the real commit where the history is reachable, and the
    verbatim excerpt above so this runs in a shallow CI checkout too.  The
    excerpt is verified against the commit when that is available, so it cannot
    drift into being a weaker text than the one it stands for.
    """
    def stale_in(text: str) -> list[str]:
        return [s for s in STALE if s in text]

    def missing_in(text: str) -> list[str]:
        return [r for r in REQUIRED if r not in text]

    pre_p1_0 = PRE_P1_0_EXCERPT
    if shutil.which("git") is not None \
            and not _git("cat-file", "-e",
                         f"{PRE_P1_0_COMMIT}:AGENTS.md").returncode:
        real = _git("show", f"{PRE_P1_0_COMMIT}:AGENTS.md").stdout
        for stale in stale_in(PRE_P1_0_EXCERPT):
            assert stale in real, (
                f"the excerpt quotes {stale!r}, which commit "
                f"{PRE_P1_0_COMMIT} no longer contains — the excerpt has drifted "
                "from the file it stands for")
        pre_p1_0 = real

    found = stale_in(pre_p1_0)
    assert len(found) >= 20, (
        f"only {len(found)} of the {len(STALE)} stale strings are in the "
        f"pre-P1.0 contract ({found}); a stale list that no longer matches the "
        "file it was written from is not a gate")

    # ... and the shape-agnostic patterns, which have to fire on the OLD file's
    # backslash spellings too or they are decoration
    shaped = [pattern.pattern for pattern in STALE_PATTERNS
              if pattern.search(pre_p1_0)]
    assert len(shaped) == len(STALE_PATTERNS), (
        f"only {shaped} of the {len(STALE_PATTERNS)} shape-agnostic Windows "
        f"patterns match the pre-P1.0 contract, which carries the very spelling "
        "they exist for; a pattern that never fires is not a gate")

    assert missing_in(pre_p1_0), (
        "the pre-P1.0 contract satisfies every required token, which means the "
        "required list no longer describes the rewrite")
    assert missing_in(_agents_text()) == [], (
        "…and the CURRENT contract must satisfy all of them")

    # the domain rules and the model policy are P1.0's "keep verbatim" list, and
    # the pre-rewrite file states the domain rules in the past tense of
    # milestones — so it must fail the phase-task vocabulary too.
    assert "Work ONE milestone per conversation" in pre_p1_0, (
        "the pre-P1.0 contract is supposed to be the milestone-numbered one; "
        "if this sentence is gone from it, the excerpt is no longer the file "
        "that motivated the rewrite")


def test_the_domain_rule_check_can_still_fail():
    """The three weakenings round 1's gate let through, driven again — in memory.

    Every one of these passed the whole of :data:`DOMAIN_RULES` before fix round
    1, and each was measured on the REAL ``AGENTS.md``: an obligation turned into
    advice is exactly the edit this gate exists to make loud, because the rules
    are what this repository's reviews rest on and nothing else enforces them.
    The mutations live here, in a copy of the live section, so the property is
    re-measured on every run instead of being a claim in a commit message.
    """
    section = _section(_agents_text(), "## Domain rules")
    for rule, mutation, replacement in (
        ("energy accounting is load-bearing",
         "must book its work", "should book its work"),
        ("the Fortran wins",
         "NEVER guess Fortran semantics", "Try not to guess Fortran semantics"),
        ("the 60 s slow-test threshold",
         "Mark any test ≥60 s serial with `@pytest.mark.slow`",
         "Mark any slow test with `@pytest.mark.slow`"),
        ("never weaken a test",
         "Never weaken a test", "Try not to weaken a test"),
        ("a parity run is the evidence",
         "Numerics changes REQUIRE validation evidence",
         "Numerics changes should have validation evidence"),
    ):
        assert mutation in section, (
            f"§Domain rules no longer says {mutation!r}; this mutation can only "
            "prove the check fires if the text it mutates is still there")
        lost = _lost_domain_rules(section.replace(mutation, replacement))
        assert rule in lost, (
            f"rewriting {mutation!r} to {replacement!r} was NOT reported: "
            f"{sorted(lost)}. The gate has stopped seeing this edit, which is "
            "the finding the check exists to close.")
        assert lost[rule], (
            f"{rule!r} is reported without naming a lost fragment; the report "
            "has to say what the edit destroyed")


def test_the_no_bare_count_check_can_still_fail():
    """``unbound()`` is imported, so pin that it is not vacuous here either."""
    from tests.test_p0_record_suite_counts import unbound

    assert unbound(["- the fast tier is `14601 passed, 15 skipped`."]) != [], (
        "a bare, unbound suite count must still be reported — otherwise the "
        "no-bare-count check above is vacuous")
    assert unbound(["- Measured 2026-10-04: `14601 passed, 15 skipped`."]) == [], (
        "and a dated measurement is still legitimate")
