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
* Windows survives as a *labelled* compatibility section: the heading must stay
  (deleting it is the tempting way to satisfy a literal rule) and every
  Windows-shaped path in the file must sit in a window that says it describes the
  other supported machine, using the sibling gate's own discriminator;
* the program vocabulary (``P<phase>.<task>``, waves, reviewers, M700+), the
  model policy (``space-bunny``, never Fast) and the domain rules the plan says
  to keep **verbatim** are still present;
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

import re
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
    "--mode parity",
    # the labelled compatibility section, and the vocabulary it must teach
    "## Windows compatibility", "compatibility", "§[A]", "§[B]", "Scripts",
    "space-bunny", "Fast", "M700", "P1.0", "wave", "reviewer",
)

#: Domain rules the plan says to keep **verbatim**.  A rewrite that paraphrases
#: one of these has changed a rule, not a wording.
DOMAIN_RULES = (
    "the Fortran wins", "EN ledger", "Never weaken a test",
    "@pytest.mark.slow", "PYRADIOSS_BACKEND=numpy",
)

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

#: ``$PYTHON -m pytest`` / ``--mode parity`` / ``--render``: the distinctive
#: tokens of §4.2's commands, not the whole incantation, so a command written
#: with different flags is still recognised as present.
_COMMAND_TOKEN = re.compile(
    r"\$PYTHON\s+([A-Za-z0-9_./-]+)|(--mode\s+\w+|--collect-only|--render)")


def _agents_text() -> str:
    return AGENTS.read_text(encoding="utf-8")


def _orchestration_text() -> str:
    return ORCHESTRATION.read_text(encoding="utf-8")


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
    missing = [r for r in REQUIRED if r not in text]
    assert missing == [], (
        f"AGENTS.md no longer names {missing}: the rewrite is not allowed to "
        "lose content, only to restate it platform-neutrally.")


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
    """``AGENTS.md`` and ``plan/00_ORCHESTRATION.md`` §4.1 must agree.

    ``AGENTS.md`` declares §4.1 "the single source of truth for every variable
    named below", so the two files are a contract and a copy of it.  A copy
    rots silently; this is what stops that.  The comparison is EQUALITY on both
    sides — every §4.1 variable named, and every non-name declared in
    :data:`VARIABLES_NOT_IN_AGENTS` with a reason — so adding a row to §4.1
    fails here until somebody decides what the contract says about it.
    """
    variables = set(_VARIABLE_ROW.findall(_section(_orchestration_text(), "### 4.1")))
    assert len(variables) >= 9, (
        "§4.1's table stopped parsing as a variable table "
        f"(found {sorted(variables)}); this test would be vacuous, not green")

    missing = {v for v in variables if v not in _agents_text()}
    assert missing == set(VARIABLES_NOT_IN_AGENTS), (
        "AGENTS.md and plan/00_ORCHESTRATION.md §4.1 have drifted: "
        f"{sorted(missing)} is in one and not the other. Name it in AGENTS.md, "
        f"or declare it in VARIABLES_NOT_IN_AGENTS with the reason it does not "
        f"belong in the contract (declared: {sorted(VARIABLES_NOT_IN_AGENTS)})")


def test_agents_md_carries_the_commands_the_environment_contract_defines():
    """Same equality, over §4.2's commands.

    §4.2 is where a new agent copies its first command from, so a command that
    exists only in the plan is one this repository can lose without noticing.
    Equality again, with :data:`COMMANDS_NOT_IN_AGENTS` as the declared
    exemptions.
    """
    commands = _section(_orchestration_text(), "### 4.2")
    tokens = set()
    for script, flag in _COMMAND_TOKEN.findall(commands):
        tokens.add(script or flag)
    assert len(tokens) >= 5, (
        f"§4.2's command block stopped parsing ({sorted(tokens)}); this test "
        "would be vacuous, not green")

    missing = {t for t in tokens if t not in _agents_text()}
    assert missing == set(COMMANDS_NOT_IN_AGENTS), (
        "AGENTS.md §Commands and plan/00_ORCHESTRATION.md §4.2 have drifted: "
        f"{sorted(missing)} is in one and not the other. Carry the command, or "
        "declare it in COMMANDS_NOT_IN_AGENTS with the reason "
        f"(declared: {sorted(COMMANDS_NOT_IN_AGENTS)})")


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
    """The plan says keep the domain rules verbatim; paraphrasing is changing."""
    text = _agents_text()
    for rule in DOMAIN_RULES:
        assert rule in text, f"the domain rule {rule!r} was lost in the rewrite"


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


def test_the_no_bare_count_check_can_still_fail():
    """``unbound()`` is imported, so pin that it is not vacuous here either."""
    from tests.test_p0_record_suite_counts import unbound

    assert unbound(["- the fast tier is `14601 passed, 15 skipped`."]) != [], (
        "a bare, unbound suite count must still be reported — otherwise the "
        "no-bare-count check above is vacuous")
    assert unbound(["- Measured 2026-10-04: `14601 passed, 15 skipped`."]) == [], (
        "and a dated measurement is still legitimate")
