"""Task P1.9 — CI that runs on the platform the port is validated on, and
whose verdict is mechanical instead of a judgement call.

What this file is
-----------------
``plan/02_phase1_foundation.md:468-493`` asks for two things and names the
reason: Linux CI is a **gate**, not a convenience, because
``docs/OPEN_BUGS.md`` item 6 records two LAW4 cfg tests that PASS in CI (the
old Windows-shaped pipeline) and FAIL in the Linux dev environment.  The old
pipeline could not catch that: it printed a summary line and a human read it.

So the shape this file pins is:

* a Linux job that runs the project's own gate selection
  (``pytest -q -m "not slow"``, ``AGENTS.md`` §Commands) on a runner, and
* a **ledger gate** — ``tools/regression_ledger.py --check`` against the
  committed record ``tools/validation_data/baseline.json`` (Task P1.8) — whose
  exit code, not a reader's judgement, decides the job; and
* a parity job in its own workflow that runs the real Fortran oracle
  **only when it is configured**, and says so in the log when it is not.

Why the tests parse the YAML instead of grepping it
---------------------------------------------------
The plan's own test (``plan/02_phase1_foundation.md:480-485``) is three
substring assertions.  Kept here as
:func:`test_the_plan_s_own_three_assertions_still_hold`, because a reviewer
should be able to check them against the plan -- but the grep form has a
failure mode that matters for this task in particular: **a commented-out line
satisfies a substring assertion**.  A gate that exists only in a comment is
indistinguishable from a gate that works, which is exactly the class of defect
this task exists to remove.

So every assertion here runs against :func:`_code_only`, a comment-stripped
view of the file that respects quoting, and the load-bearing ones go further
still: the workflow is parsed into jobs and steps by indentation
(:func:`_parse_jobs`), and the ledger gate is asserted on the ``run:`` payload
of a real step -- the command, its flags, and the **order** the two ledger
invocations appear in.  ``grep`` passes on ``# run: python tools/
regression_ledger.py --check ...``; this cannot.

What this file CANNOT detect, stated plainly
--------------------------------------------
**GitHub Actions cannot be executed here.**  There is no Actions runner, no
``act``, and no emulator in this venv.  Everything below is verified by
*parsing* the workflow files, which catches the classic breakages (invalid
YAML, a ``${{ }}`` expression interpolated into shell text, a missing
``permissions:`` block, an action pinned to a moving ref, a ``run:`` quoting
error under ``bash -e``) and catches nothing about the runner itself:

* whether the fast tier finishes inside ``timeout-minutes`` on a 2-core
  runner -- the ledger gate runs it **serially** (see the workflow's own
  comment and :func:`test_the_ledger_gate_does_not_pass_xdist_to_the_ledger`),
  and that is a cost nobody has measured on a runner;
* whether ``tests/test_p0_no_stale_machine_paths.py`` passes on a runner.  It
  is not this file's to own and it is not edited here; the finding is recorded
  in the task report;
* whether the parity job's expressions evaluate as intended.  The gating
  expression is asserted *structurally* (it names ``vars.OR_STARTER`` /
  ``vars.OR_ENGINE`` and not ``secrets.``) and hand-reasoned, not executed.
  The one structural fact asserted about ``secrets`` is a real Actions rule,
  not a preference: **the ``secrets`` context is not available in a job-level
  ``if:``**, so ``if: secrets.OR_STARTER != ''`` would silently evaluate empty
  and skip the job forever -- a gate that never opens and never says why.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"
PARITY = REPO / ".github" / "workflows" / "parity.yml"

#: The committed record Task P1.8 recorded, named repo-relative exactly as
#: the workflow names it.  A gate pointed at some other path is not this gate.
LEDGER_BASELINE = "tools/validation_data/baseline.json"

#: The project's own gate selection (AGENTS.md §Commands, plan §4.2).
FAST_TIER = '-m "not slow"'


# ---------------------------------------------------------------------------
# Reading a workflow without a YAML parser
# ---------------------------------------------------------------------------
# The two helpers below parse the *shape* this task needs -- top-level keys,
# jobs, and each job's steps -- from indentation, which is the part a substring
# assertion cannot reach.  They are a parser, and a parser can be wrong.
#
# It was.  The hand parser below read ``- name: Read the rows back: a deviation
# is a failure`` as three perfectly good fields and every test in this file
# passed, while PyYAML rejected the file outright: an unquoted scalar holding
# ``": "`` is a *nested mapping* to YAML, not a name.  :func:`_unquoted_colon_values`
# catches exactly that one rule, and it is kept as a test of its own
# (:func:`test_the_workflow_yaml_lint_rejects_the_colon_case`) -- but it is a
# ONE-RULE lint, and a round-1 review showed what that costs: an unterminated
# quoted scalar, which PyYAML also rejects, sailed through it.  So validity is
# now decided by a real parser, and pyyaml is a declared test dependency
# (``pyproject.toml``'s ``test`` extra, which both workflows install) rather
# than an ambient convenience.


def _code_only(text: str) -> str:
    """The file with its comments removed, quote-aware.

    ``#`` starts a comment when it is outside quotes and preceded by
    whitespace or at the start of the line -- the same rule YAML uses, which
    is why ``'a # b'`` survives and ``run: x # note`` does not.  This is the
    function that makes "assert the string appears" mean "assert the workflow
    *does* it".
    """
    out = []
    for raw in text.splitlines():
        quote = None
        cut = None
        for i, char in enumerate(raw):
            if quote:
                if char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == "#" and (i == 0 or raw[i - 1] in " \t"):
                cut = i
                break
        out.append(raw if cut is None else raw[:cut])
    return "\n".join(out) + "\n"


_JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*$")
_JOB_ATTR = re.compile(r"^    ([A-Za-z0-9_.-]+):(?:\s+(.*))?$")
_STEP_START = re.compile(r"^      -\s+([A-Za-z0-9_.-]+):\s*(.*)$")
_BLOCK_MARKERS = ("|", ">", "|-", ">-", "|+", ">+")


class Job:
    """One job: its attributes, its steps, and its own source text."""

    def __init__(self, name: str):
        self.name = name
        self.attrs: dict = {}
        self.steps: list = []
        self.text = ""

    @property
    def runs_on(self) -> str:
        return self.attrs.get("runs-on", "")

    @property
    def condition(self) -> str:
        return self.attrs.get("if", "")

    def run_texts(self) -> list:
        """The ``run:`` payload of every step, in file order."""
        return [s["run"] for s in self.steps if s["run"]]

    def all_runs(self) -> str:
        return "\n".join(self.run_texts())

    def step_using(self, needle: str) -> dict:
        """The first step whose run payload (or name) contains ``needle``."""
        for step in self.steps:
            if needle in step["run"] or needle in step["name"]:
                return step
        raise AssertionError(
            f"job {self.name!r} has no step mentioning {needle!r}; steps are "
            f"{[(s['name'], s['run'][:40]) for s in self.steps]}")


def _parse_jobs(text: str) -> dict:
    """``{job name: Job}`` for a workflow's ``jobs:`` mapping.

    Deliberately narrow: it reads job-level scalars (``runs-on``, ``if``,
    ``permissions``, ``env``), step starts (``- name:``/``- uses:``/``- run:``)
    and step-level scalars, plus the body of a ``key: |`` block scalar.  It
    ignores anything it does not understand rather than guessing, and the
    tests below fail loudly if the shape it expects is not there.
    """
    lines = _code_only(text).splitlines()
    jobs: dict = {}
    order: list = []
    in_jobs = False
    job: Job | None = None
    step: dict | None = None
    block: str | None = None
    start = 0

    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()

        if indent == 0:
            in_jobs = stripped == "jobs:"
            if job is not None:
                job.text = "\n".join(lines[start:index])
            job = None
            step = None
            block = None
            continue
        if not in_jobs:
            continue

        if indent == 2 and _JOB_HEADER.match(raw):
            if job is not None:
                job.text = "\n".join(lines[start:index])
            job = Job(_JOB_HEADER.match(raw).group(1))
            jobs[job.name] = job
            step = None
            block = None
            start = index
            continue
        if job is None:
            continue

        if indent == 4:
            match = _JOB_ATTR.match(raw)
            if match:
                job.attrs[match.group(1)] = (match.group(2) or "").strip()
            step = None
            block = None
            continue

        if indent == 6 and stripped.startswith("- "):
            match = _STEP_START.match(raw)
            key = match.group(1) if match else stripped[2:].partition(":")[0]
            value = (match.group(2) if match else "").strip()
            step = {"key": key, "name": value if key == "name" else "",
                    "uses": value if key == "uses" else "",
                    "if": "", "run": "", "env": {}, "last": "", "raw": [raw]}
            job.steps.append(step)
            block = None
            continue

        if step is None:
            continue

        if indent >= 8:
            step["raw"].append(raw)
            if block is not None:
                step[block] = (step[block] + raw.rstrip() + "\n")
                continue
            match = re.match(r"^        ([A-Za-z0-9_.-]+):(?:\s+(.*))?$", raw)
            if match:
                key, value = match.group(1), (match.group(2) or "").strip()
                step["last"] = key
                if key == "name":
                    step["name"] = value
                elif key == "uses":
                    step["uses"] = value
                elif key == "if":
                    step["if"] = value
                elif key == "run":
                    step["run"] = value
                    if value in _BLOCK_MARKERS:
                        block = "run"
                        step["run"] = ""
                elif key == "env":
                    step["env"] = {}
                continue
            # No block scalar open, so this line belongs to the mapping the
            # previous key opened -- the only one this parser reads is `env:`.
            if step["last"] == "env":
                pair = stripped.split(":", 1)
                if len(pair) == 2:
                    step["env"][pair[0].strip()] = pair[1].strip()
            continue
    if job is not None:
        job.text = "\n".join(lines[start:])
    return jobs


def _workflow(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _notice(text: str) -> str:
    return "::notice::" in text or "::notice " in text


_BLOCK_OPEN = re.compile(r"^(\s*)(?:-\s+)?[A-Za-z0-9_.\"'-]+:[ \t]*([|>][-+]?)[ \t]*$")
_MAPPING = re.compile(r"^(\s*)(?:-\s+)?[A-Za-z0-9_.-]+:[ \t]+(\S.*?)[ \t]*$")


def _unquoted_colon_values(text: str) -> list:
    """Lines whose unquoted value contains ``": "`` -- YAML reads those as a
    nested mapping, which is a hard syntax error at the wrong indentation.

    ``- name: Read the rows back: a deviation is a failure`` is the case this
    exists for: it is a perfectly ordinary-looking line, and
    ``yaml.safe_load`` rejects the whole workflow for it.  Lines inside a block
    scalar (``run: |`` and its body) are skipped -- there ``": "`` is just
    shell text -- and a value that starts and ends with the same quote
    character is left alone, because a quoted scalar may contain anything.
    """
    suspects = []
    block_indent = None
    for raw in _code_only(text).splitlines():
        indent = len(raw) - len(raw.lstrip(" "))
        if block_indent is not None:
            if not raw.strip() or indent >= block_indent:
                continue
            block_indent = None
        opening = _BLOCK_OPEN.match(raw)
        if opening:
            block_indent = len(opening.group(1)) + 2
            continue
        line = _MAPPING.match(raw)
        if not line:
            continue
        value = line.group(2)
        if len(value) > 1 and value[0] in "'\"" and value[-1] == value[0]:
            continue
        if ": " in value or value.endswith(":"):
            suspects.append(raw)
    return suspects


def _yaml():
    """The real parser, or an explicit, loud skip.

    The round-1 review's finding was that this file's "valid YAML" test took a
    one-rule fallback branch *everywhere*, including in CI, and that an
    unterminated quoted scalar -- which PyYAML rejects -- passed it.  A test
    named "is valid YAML" has to reject what a YAML parser rejects, so the
    parser is now a declared dependency (``pyyaml`` in ``pyproject.toml``'s
    ``test`` extra, installed by both workflows) and a missing one is a skip
    with a reason, never a silent pass.  Two companion tests below keep that
    skip from ever being the whole story: the dependency must be *declared*,
    and CI must *install the extra*.
    """
    try:
        import yaml
    except ImportError as exc:              # pragma: no cover — see below
        import pytest
        pytest.skip(f"no YAML parser importable ({exc}); this gate is "
                    f"vacuous without one -- install the test extra "
                    f"(pip install -e '.[test]')")
    return yaml


def test_both_workflows_parse_with_a_real_yaml_parser():
    """Valid YAML means a real parser accepted it, not that one rule held.

    PyYAML 6.0.3 is verified in this venv (the exact version a `# pin:` line
    for ``requirements-lock.txt`` §[B] should record -- that file is not this
    task's).  The asserted shape is the one Actions needs: a mapping with a
    non-empty ``jobs:``, and every job carrying ``runs-on`` and at least one
    step -- a workflow that parses but names no runner never runs anything.
    """
    yaml = _yaml()
    for path in (CI, PARITY):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict), path.name
        assert isinstance(doc.get("jobs"), dict) and doc["jobs"], path.name
        for job in doc["jobs"].values():
            assert "runs-on" in job, path.name
            assert job.get("steps"), path.name


def test_the_yaml_lint_still_rejects_the_colon_case_it_was_written_for():
    """The one-rule lint is kept as a test of its own, not as a validity gate.

    It runs with no parser at all, so the defect the round-1 review credited it
    with catching (an unquoted ``": "``, which is a nested mapping) stays
    caught even on a box with no PyYAML.  What it does NOT do is stand in for a
    parse -- that is the test above.
    """
    for path in (CI, PARITY):
        assert not _unquoted_colon_values(path.read_text(encoding="utf-8")), path.name
    planted = ('jobs:\n'
               '  parity:\n'
               '    steps:\n'
               '      - name: Read the rows back: a deviation is a failure\n')
    assert _unquoted_colon_values(planted), "the rule stopped catching its case"


def test_the_yaml_parser_is_a_declared_dependency_that_ci_actually_installs():
    """A gate that needs a package the gate does not install is vacuous.

    Two halves, both load-bearing: ``pyyaml`` is in ``pyproject.toml``'s
    ``test`` extra, and every CI job that runs the suite installs *that extra*
    rather than a hand-written list.  Either half alone leaves a hole -- declared
    but not installed, or installed by hand until the declaration drifts.
    """
    import tomllib
    extras = tomllib.loads(
        (REPO / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["optional-dependencies"]
    assert any(r.lower().startswith("pyyaml") for r in extras["test"]), extras["test"]

    for path in (CI, PARITY):
        jobs = _parse_jobs(_workflow(path))
        # the editable install is the one that carries the extras; the numba
        # step installs an optional backend on purpose and is not one of them
        installs = [s for job in jobs.values() for s in job.steps
                    if "pip install" in s["run"] and "-e " in s["run"]]
        assert installs, path.name
        for step in installs:
            assert '-e ".[test]"' in step["run"] or "-e .[test]" in step["run"], (
                f"{path.name} / {step['name']}: the suite's dependencies must "
                f"come from the declared test extra:\n{step['run']}")


def _parity_gate(jobs: dict) -> Job:
    """The parity job that would actually run: the one gated on ``!= ''``."""
    gates = [j for j in jobs.values()
             if "vars.OR_STARTER" in j.condition and "== ''" not in j.condition]
    assert len(gates) == 1, f"expected one parity gate, got {[g.name for g in gates]}"
    return gates[0]


# ---------------------------------------------------------------------------
# 1. The plan's own test (plan/02_phase1_foundation.md:480-485), verbatim
# ---------------------------------------------------------------------------

def test_the_plan_s_own_three_assertions_still_hold():
    text = Path(".github/workflows/ci.yml").read_text()
    assert "runs-on: ubuntu" in text
    assert "regression_ledger.py" in text
    assert "-m \"not slow\"" in text


# ---------------------------------------------------------------------------
# 2. The Linux job runs the project's own gate selection
# ---------------------------------------------------------------------------

def test_ci_has_a_linux_job_that_runs_the_fast_tier_on_a_runner():
    jobs = _parse_jobs(_workflow(CI))
    assert "fast-tests" in jobs, f"jobs are {sorted(jobs)}"
    fast = jobs["fast-tests"]
    assert fast.runs_on.startswith("ubuntu"), fast.runs_on
    pytest_step = fast.step_using("python -m pytest")
    assert FAST_TIER in pytest_step["run"], pytest_step["run"]
    # the exact AGENTS.md command, as a command: `pytest -q -m "not slow"`
    assert re.search(r"pytest\b[^\n]*\s-q\b", pytest_step["run"]), \
        pytest_step["run"]
    # ... and the reference backend pin, which plan/00_ORCHESTRATION.md §1.1.6
    # requires of any before/after comparison.
    assert pytest_step["env"].get("PYRADIOSS_BACKEND") == "numpy", \
        pytest_step["env"]


def test_the_linux_jobs_are_the_only_ones_and_nothing_silently_disables_them():
    """No ``continue-on-error`` and no ``|| true`` on a job that runs the suite.

    ``continue-on-error: true`` is how a gate becomes a report; ``|| true`` at
    the end of a pytest command does the same thing in shell.  Either one on a
    suite job would make the ledger's exit code theatre.  A step that merely
    uploads an artifact may of course always run -- that is not a gate.
    """
    jobs = _parse_jobs(_workflow(CI))
    for name, job in jobs.items():
        assert job.attrs.get("continue-on-error") != "true", name
        for step in job.steps:
            assert "|| true" not in step["run"], (name, step["name"])
            if "pytest" in step["run"] or "regression_ledger.py" in step["run"]:
                assert step["if"] != "always()", (name, step["name"])


# ---------------------------------------------------------------------------
# 3. The ledger gate -- the mechanical verdict
# ---------------------------------------------------------------------------

def test_ci_gates_on_the_regression_ledger_not_on_a_summary_line():
    """The gate must be a real step that really runs ``--check``.

    Asserted on the step's ``run:`` payload with comments stripped, so the
    three ways to fake this -- a commented-out command, a filename mentioned
    in a comment, a flag passed to the wrong subcommand -- all fail.
    """
    jobs = _parse_jobs(_workflow(CI))
    assert "ledger-gate" in jobs, f"jobs are {sorted(jobs)}"
    gate = jobs["ledger-gate"]
    assert gate.runs_on.startswith("ubuntu"), gate.runs_on

    record = gate.step_using("--record")
    check = gate.step_using("--check")
    assert LEDGER_BASELINE in check["run"], check["run"]
    # `--check` alone would re-run the whole suite a second time; `--results`
    # is the mode Task P1.8 documented as "what CI does".
    assert "--results" in check["run"], check["run"]
    # one suite run, then one comparison, in that order
    assert gate.steps.index(record) < gate.steps.index(check)
    # the record is written outside the checkout, never into it
    assert "RUNNER_TEMP" in record["run"], record["run"]


def test_the_ledger_check_names_the_committed_record_and_nothing_else():
    """The gate is pointed at the record the repository actually ships.

    A ``--check`` with a path that does not exist exits 3 -- which is still a
    red job, but for the wrong reason, and a path that is *generated in the
    same run* compares a run with itself, which is a tautology rather than a
    gate.  So: the committed record, named repo-relative, and it must exist.

    The round-1 Minor finding was that this test also asserted
    ``baseline_ci.json`` must NOT exist, which forbids the strict alternative
    (record in the runner's own environment, drop ``--allow-environment-drift``)
    rather than requiring it to be wired up.  It is now conditional: if such a
    record exists, the gate has to be pointed at it, so the file can never sit
    there unused while the gate accepts a drift it does not need to.
    """
    jobs = _parse_jobs(_workflow(CI))
    check = jobs["ledger-gate"].step_using("--check")
    paths = re.findall(r"--check\s+(\S+)", check["run"])
    assert paths == [LEDGER_BASELINE], paths
    assert (REPO / LEDGER_BASELINE).is_file(), LEDGER_BASELINE

    ci_record = REPO / "tools" / "validation_data" / "baseline_ci.json"
    if ci_record.exists():
        assert paths == ["tools/validation_data/baseline_ci.json"], (
            f"{ci_record.name} exists but the gate still checks "
            f"{LEDGER_BASELINE}: a CI-specific record has to be recorded in the "
            f"environment it is compared in (from the uploaded artifact) and "
            f"then pointed at, or it is dead weight")
        assert "--allow-environment-drift" not in check["run"], (
            "a record made in the runner's own environment does not need the "
            "drift relaxation; keeping it would hide a real mismatch")


def test_the_ledger_gate_says_which_environment_it_compared_against():
    """The comparison crosses environments, so the workflow must say so.

    This is a **documentation** assertion and is labelled as one: a comment
    cannot fail a build.  It is here because the alternative -- a silent
    ``--allow-environment-drift`` -- is exactly the kind of quiet widening a
    reviewer cannot see, and the requirement is that the decision be visible
    *in the workflow*, not only in a commit message.
    """
    jobs = _parse_jobs(_workflow(CI))
    gate = jobs["ledger-gate"]
    assert "--allow-environment-drift" in gate.all_runs(), gate.all_runs()
    comment_text = _workflow(CI)
    body = comment_text.split("ledger-gate:", 1)[1]
    assert "drift" in body, "the gate accepts a drift; say so beside it"
    assert "OR_SRC" in body, "name the axis that makes the drift irreducible"


def test_the_ledger_gate_does_not_pass_xdist_to_the_ledger():
    """``-n auto`` must not reach the ledger's child run.

    ``tools/regression_ledger.py`` writes its report from
    ``pytest_sessionfinish`` (module lines 439-447) into **one** file named by
    ``LEDGER_OUT``.  Under xdist the plugin is loaded in the controller *and in
    every worker*, and each process writes that same file at its own session
    end: last writer wins, so the record can describe a fraction of the run.
    The tool is not this task's to edit, so the gate stays serial and says so.

    Each step's ``run:`` is also asserted non-empty, because this is a set of
    ABSENCE assertions: a parser that failed to open a payload would hand back
    ``""``, which satisfies ``"-n auto" not in ""`` and turns the whole test
    green while asserting nothing.
    """
    jobs = _parse_jobs(_workflow(CI))
    steps = jobs["ledger-gate"].steps
    assert steps, "the ledger-gate job has no steps"
    for step in steps:
        assert step["run"] or step["uses"], (
            f"step {step['name']!r} has neither a run payload nor a uses:")
        assert "-n auto" not in step["run"], step["run"]
        assert "--dist" not in step["run"], step["run"]


def test_the_cfg_cache_key_is_not_immutable_under_a_constant_key():
    """A cache key that never changes freezes its contents for good.

    The round-1 Minor finding: ``key: hm-cfg-files-v1`` with no
    ``restore-keys`` and no upstream ref means the first run to populate the
    cache wins forever, and a schema change in OpenRadioss's ``hm_cfg_files``
    would be invisible to the /MAT-/PROP readers for the life of the
    repository -- silently, because those tests skip rather than fail when the
    tree is unusable.

    The fix is a per-run primary key plus a prefix to restore from, which means
    the fetch step's guard can no longer be ``cache-hit != 'true'``: that
    output is ``false`` even on a restore-key match, so it would clone on every
    run.  Hence the second half.
    """
    for path in (CI, PARITY):
        for job in _parse_jobs(_workflow(path)).values():
            caches = [s for s in job.steps if "actions/cache" in s["uses"]]
            if not caches:
                continue
            assert caches, job.name
            for step in caches:
                raw = "\n".join(step["raw"])
                key = next(ln for ln in step["raw"] if "key:" in ln)
                assert "github.run_id" in key or "${{" in key, (
                    f"{path.name} / {job.name}: the cache key {key.strip()!r} is "
                    f"constant, so the cached tree is never refreshed")
                assert "restore-keys:" in raw, (
                    f"{path.name} / {job.name}: a per-run key with no "
                    f"restore-keys re-clones on every run\n{raw}")
            # the fetch must be guarded by the filesystem, not by cache-hit
            fetches = [s for s in job.steps
                       if "sparse-checkout set" in s["run"]]
            assert fetches, f"{path.name} / {job.name}: the CFG fetch is gone"
            for step in fetches:
                assert "cache-hit" not in step["if"], (
                    f"{path.name} / {job.name}: cache-hit is 'false' on a "
                    f"restore-key match, so this clones every run\n"
                    f"{step['if']}")
                assert "-d or_cfg/hm_cfg_files/config/CFG" in step["run"], \
                    step["run"]


# ---------------------------------------------------------------------------
# 4. The coverage the old CI had, kept
# ---------------------------------------------------------------------------

def test_ci_keeps_two_python_versions_and_numba_on_exactly_one_leg():
    """The value of the pipeline being replaced is not dropped to save lines.

    Two interpreter legs, and the optional accelerated backend exercised on
    one of them: ``tests/test_m7_backends.py``'s parity tests are gated by
    ``skipif(not HAS_NUMBA)``, so a leg without numba does not test the
    accelerated backend at all -- it skips.
    """
    jobs = _parse_jobs(_workflow(CI))
    fast = jobs["fast-tests"]
    versions = re.findall(r'"(\d+\.\d+)"', fast.attrs.get("strategy", "")
                          + _matrix_block(fast))
    assert len(set(versions)) >= 2, f"matrix versions: {versions}"

    numba_steps = [s for s in fast.steps
                   if "pip install numba" in s["run"]]
    assert len(numba_steps) == 1, [s["name"] for s in numba_steps]
    assert "matrix.python-version" in numba_steps[0]["if"], \
        numba_steps[0]["if"]

    # ... and the numba leg still gets the M40 'auto' end-to-end check, which
    # is the only place the 'auto' default can resolve to numba at all.
    auto_steps = [s for s in fast.steps
                  if re.search(r"pytest[^\n]*test_m40_auto_backend", s["run"])
                  and "deselect" not in s["run"]]
    assert auto_steps, "the M40 auto-backend step was dropped"
    assert "matrix.python-version" in auto_steps[0]["if"], \
        auto_steps[0]["if"]


def _matrix_block(job: Job) -> str:
    """The ``strategy:`` sub-tree of a job, as text.

    Read out of the job's own source rather than parsed: it is the one place
    a version number appears, and the tests above only need to know how many
    distinct ones there are.
    """
    lines = [ln for ln in job.text.splitlines()
             if "python-version" in ln or "python_version" in ln]
    return "\n".join(lines)


def test_no_workflow_pins_an_action_to_a_moving_reference():
    """``uses:`` must name a ref that cannot move under us.

    ``@v4``/``@v5`` (what this repository uses) are release tags; ``master``,
    ``main`` and ``latest`` are branches, so the action's code can change
    between two runs of the same commit.  A digest pin is stricter and is the
    next hardening step; it needs verified SHAs, which cannot be obtained
    offline, so the rule enforced here is "not a branch".
    """
    for path in (CI, PARITY):
        for uses in re.findall(r"uses:\s*(\S+)", _code_only(_workflow(path))):
            ref = uses.rsplit("@", 1)
            assert len(ref) == 2, f"{path.name}: {uses} names no ref"
            assert ref[1] not in ("master", "main", "latest", "HEAD"), \
                f"{path.name}: {uses} is pinned to a moving ref"


# ---------------------------------------------------------------------------
# 5. Actions syntax: the classic breakages
# ---------------------------------------------------------------------------

def test_no_workflow_interpolates_a_expression_into_shell_text():
    """``${{ }}`` belongs in ``env:``/``with:``, never inside a ``run:``.

    Actions substitutes an expression *before* the shell sees the line, which
    is exactly the hazard: a value containing a quote, a space or a ``$`` is
    pasted into a command line unescaped.  Passing it through ``env:`` and
    reading ``"$VAR"`` in the script is injection-safe and quoting-safe, and
    it is checkable here -- which a hand review of a 200-line workflow is not.
    """
    for path in (CI, PARITY):
        for job in _parse_jobs(_workflow(path)).values():
            for step in job.steps:
                assert "${{" not in step["run"], \
                    f"{path.name} / {job.name} / {step['name']}: {step['run']}"


def test_both_workflows_declare_a_least_privilege_permissions_block():
    """``permissions:`` is opt-in per workflow, so its absence is default-wide.

    With no ``permissions:`` block a workflow inherits the repository
    default, which may be ``write-all``.  Both files here only ever read the
    repository, so they say so.
    """
    for path in (CI, PARITY):
        text = _code_only(_workflow(path))
        assert re.search(r"^permissions:\s*$", text, re.M), path.name
        assert re.search(r"^ {2}contents:\s*read\s*$", text, re.M), path.name


def test_no_workflow_secrets_a_path_or_a_credential():
    """Paths are configuration; configuration belongs in ``vars``.

    Two reasons, both mechanical: a path in ``vars`` is readable in a
    job-level ``if:`` (which is what gates the parity job), and a *path* is
    not a secret -- the oracle binaries are licensed artefacts, not tokens.
    ``secrets`` in a job-level ``if:`` is not merely discouraged, it does not
    resolve, so such a gate skips forever and says nothing.
    """
    for path in (CI, PARITY):
        text = _code_only(_workflow(path))
        assert "secrets." not in text, path.name


# ---------------------------------------------------------------------------
# 6. The parity job: configured or loudly absent
# ---------------------------------------------------------------------------

def test_the_parity_job_is_gated_on_configuration_and_says_so_when_it_is_not():
    jobs = _parse_jobs(_workflow(PARITY))
    gate = _parity_gate(jobs)
    # gated on the oracle being *configured* ...
    assert "vars.OR_STARTER" in gate.condition, gate.condition
    assert "vars.OR_ENGINE" in gate.condition, gate.condition
    # ... and it would really try: the sweep is a step, not a comment.
    sweep = gate.step_using("validate_vs_fortran")
    assert re.search(r"validate_vs_fortran\s+parity\b", sweep["run"]), \
        sweep["run"]


def test_the_parity_skip_is_visible_in_a_log_somebody_will_read():
    """A skipped job prints nothing at all, so the reason needs its own job.

    ``if:`` false means no runner is ever assigned, so a ``::notice::`` inside
    the parity job would never be executed.  The notice therefore lives in a
    companion job whose condition is the **negation**, which does run, and
    which is green -- so the reason is in the log and the workflow is not red
    for the absence of an oracle it was never promised.
    """
    jobs = _parse_jobs(_workflow(PARITY))
    gate = _parity_gate(jobs)
    others = [j for j in jobs.values() if j is not gate and j.condition]
    assert len(others) == 1, f"expected one companion job, got {[j.name for j in others]}"
    companion = others[0]
    text = companion.all_runs()
    assert _notice(text), text
    # the companion's condition is the OTHER side of the gate's: same two
    # variables, `== ''` where the gate has `!= ''`.
    for var in ("OR_STARTER", "OR_ENGINE"):
        assert var in companion.condition, companion.condition
        assert "== ''" in companion.condition, companion.condition
    assert companion.condition != gate.condition
    # it must be green (no continue-on-error, no failing command) and it must
    # say which variable is missing -- not merely that "something" was.
    assert companion.attrs.get("continue-on-error") != "true"
    assert "OR_STARTER not configured" in text, text
    assert "OR_ENGINE not configured" in text, text
    assert "claims" in text or "not a pass" in text, \
        "the notice must not read as a pass"


def test_the_parity_sweep_is_invoked_as_a_module_not_as_a_path():
    """``python -m tools.validate_vs_fortran``, not ``tools/....py``.

    Running the file by path puts ``tools/`` on ``sys.path`` instead of the
    repository root, and the harness's binary-T01 route does ``from tools
    import compare_t01`` (``validate_vs_fortran.py:1563``) -- the route a runner
    without ``th_to_csv`` is guaranteed to take, since the converter is one of
    the five things it cannot supply.  Measured on this box: the path form
    raises ``ModuleNotFoundError: No module named 'tools'`` as soon as that
    route is reached, so the path form would fail the sweep for a reason that
    has nothing to do with parity.  The module form puts the repository root on
    ``sys.path`` and was run here (``--help`` and a real invocation, both fine).
    """
    gate = _parity_gate(_parse_jobs(_workflow(PARITY)))
    sweep = gate.step_using("validate_vs_fortran")
    assert "-m tools.validate_vs_fortran parity" in sweep["run"], sweep["run"]
    assert not re.search(r"python\s+tools/validate_vs_fortran\.py", sweep["run"]), \
        sweep["run"]


def test_the_parity_job_verifies_the_oracle_before_it_runs_it():
    """Configuration is not the same as availability, and must be checked.

    ``pyradioss.paths`` resolves a resource or fails loudly, naming every
    candidate (``validate_vs_fortran.py:parity`` then exits 2 with that text).
    That is right for a tool and wrong for a CI job, so the job checks the
    binaries are executable **first** -- and, per the round-1 Critical finding,
    an unresolvable oracle is a RED job rather than a green skip.  What it is
    not allowed to be is a *silent* skip, hence the behavioural test below.
    """
    jobs = _parse_jobs(_workflow(PARITY))
    gate = _parity_gate(jobs)
    preflight = gate.step_using("OR_STARTER")
    assert "-x" in preflight["run"] or "-f" in preflight["run"], preflight["run"]
    assert "GITHUB_OUTPUT" in preflight["run"], preflight["run"]

    sweep = gate.step_using("validate_vs_fortran.py")
    assert sweep["if"], "the sweep must be gated on the preflight's output"
    assert "steps." in sweep["if"], sweep["if"]


def _preflight_payload() -> str:
    return _parity_gate(_parse_jobs(_workflow(PARITY))).step_using("OR_STARTER")["run"]


def test_an_unresolvable_oracle_fails_the_job_it_never_exits_zero():
    """Structural half: the unresolvable branch cannot succeed.

    Runs with no shell at all, so it is never vacuous.  The round-1 finding was
    that this branch read ``::notice title=parity skipped`` and ``exit 0``: the
    preflight succeeded, ``resolved=false``, every substantive step was skipped,
    and the job's result was **success** under the check name "parity vs the
    Fortran oracle" -- which is the state a maintainer reaches by following the
    instructions at the top of parity.yml.
    """
    payload = _preflight_payload()
    assert "exit 0" not in payload, (
        f"the preflight must not be able to succeed with an unresolvable "
        f"oracle:\n{payload}")
    assert "::error" in payload, (
        f"an unresolvable oracle has to be an Actions ERROR annotation, so it "
        f"is visible in the checks list and not only in the log:\n{payload}")
    assert "parity skipped" not in payload, (
        f"the word 'skipped' must not survive here: skipping is what the "
        f"companion job does for an UNSET variable, and this branch is not "
        f"that case:\n{payload}")


def test_an_unresolvable_oracle_exits_non_zero_and_annotates(tmp_path):
    """Behavioural half: run the real payload and read its exit code.

    Paths from "the maintainer's own machine" on a runner that does not have
    them -- the default state after the documented setup.  Executed under
    ``bash -e`` exactly as Actions runs it, with ``GITHUB_OUTPUT`` pointed at a
    real file so the ``resolved`` value can be read back too.
    """
    bash = shutil.which("bash")
    if not bash:                             # pragma: no cover — Windows
        pytest.skip("no bash on PATH; the preflight's behaviour is unverified "
                    "here (the structural test above still ran)")
    out = tmp_path / "gh_output"
    out.write_text("", encoding="utf-8")
    env = dict(os.environ,
               OR_STARTER="/home/dev/box/OpenRadioss/exec/starter_linux64_gf",
               OR_ENGINE="/home/dev/box/OpenRadioss/exec/engine_linux64_gf",
               GITHUB_OUTPUT=str(out))
    result = subprocess.run([bash, "-e", "-c", _preflight_payload()],
                            capture_output=True, text=True, env=env,
                            cwd=str(REPO))
    assert result.returncode != 0, (
        f"the job went GREEN on an unresolvable oracle (exit "
        f"{result.returncode})\n--- stdout ---\n{result.stdout}")
    assert result.returncode == 1, result.returncode
    assert "::error title=parity oracle unresolvable::" in result.stdout, \
        result.stdout
    # the substantive steps must still be skipped -- no half-run sweep
    assert "resolved=false" in out.read_text(encoding="utf-8"), \
        out.read_text(encoding="utf-8")
    # and the message must not read as a pass
    assert "verifies no parity" in result.stdout, result.stdout


def test_a_resolvable_oracle_resolves_and_runs_the_sweep(tmp_path):
    """The other half of the same branch: available is not the same as broken.

    A preflight that always failed would be a gate nobody can open.  Two
    executable files are cheap to make, so the payload is driven through both
    states in one test.
    """
    bash = shutil.which("bash")
    if not bash:                             # pragma: no cover — Windows
        pytest.skip("no bash on PATH; the preflight's success path is "
                    "unverified here (the structural test above still ran)")
    starter = tmp_path / "starter_linux64_gf"
    engine = tmp_path / "engine_linux64_gf"
    for path in (starter, engine):
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    out = tmp_path / "gh_output"
    out.write_text("", encoding="utf-8")
    env = dict(os.environ, OR_STARTER=str(starter), OR_ENGINE=str(engine),
               GITHUB_OUTPUT=str(out))
    result = subprocess.run([bash, "-e", "-c", _preflight_payload()],
                            capture_output=True, text=True, env=env,
                            cwd=str(REPO))
    assert result.returncode == 0, (
        f"an available oracle must resolve (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    assert "resolved=true" in out.read_text(encoding="utf-8")


def test_the_parity_sweep_step_can_see_the_oracle_it_was_configured_with():
    """The gate has to be openable, not merely closed (round-1 finding I3).

    ``OR_STARTER``/``OR_ENGINE`` are ``vars`` here, and a ``vars.X`` reference
    is a context expression, not an exported variable: the preflight's ``env:``
    block applies to the preflight step alone.  Without them repeated in the
    SWEEP step's ``env:``, the harness runs with no oracle in its environment,
    ``validate_vs_fortran.parity()`` takes its refusal path
    (``validate_vs_fortran.py:1938-1945``) and returns 2 -- so the parity gate
    could never produce evidence at all, even when configured correctly.

    ``OR_ROOT`` is asserted ABSENT on purpose: ``paths.or_starter()`` would then
    also resolve ``$OR_ROOT/bin/...``, and a tree carrying ``extlib/h3d`` is
    exactly what the harness must refuse to let the solver reach
    (``FORTRAN-FAIL(h3d-writer-in-rpath)``).
    """
    gate = _parity_gate(_parse_jobs(_workflow(PARITY)))
    sweep = gate.step_using("-m tools.validate_vs_fortran")
    for var in ("OR_STARTER", "OR_ENGINE"):
        assert sweep["env"].get(var) == "${{ vars." + var + " }}", sweep["env"]
    assert "OR_ROOT" not in sweep["env"], sweep["env"]


# ---------------------------------------------------------------------------
# 6. The parity VERDICT: an allowlist, and what it must reject
# ---------------------------------------------------------------------------
# Everything below drives the workflow's OWN script -- the ``python -c "..."``
# payload is extracted from the committed parity.yml and executed -- rather
# than re-implementing the classification in the test.  A copy of the rule in
# the test file would keep passing after the rule in the workflow changed,
# which is the failure mode this module exists to remove.

_VERDICT_SCRIPT = re.compile(r'python -c "([^"]*)"')


def _verdict_script() -> str:
    """The exact script the verdict step runs, with ``$PARITY_JSON`` resolved.

    Extracted by regex because that is the only way to run it here: Actions
    substitutes ``${{ }}`` and the shell expands ``$PARITY_JSON`` before Python
    sees them, so the payload as committed is not directly executable.  The one
    substitution performed is the same one the shell performs.
    """
    payload = _parity_gate(_parse_jobs(_workflow(PARITY))).step_using("$PARITY_JSON")
    match = _VERDICT_SCRIPT.search(payload["run"])
    assert match, f"no `python -c \"...\"` in the verdict step:\n{payload['run']}"
    return match.group(1)


def _run_verdict(rows: list, tmp_path: Path) -> subprocess.CompletedProcess:
    """Write ``rows`` as a parity_results.json and run the workflow's script."""
    table = tmp_path / "parity_results.json"
    table.write_text(json.dumps(rows), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c",
         _verdict_script().replace("$PARITY_JSON", str(table))],
        capture_output=True, text=True, cwd=str(REPO))


#: Every class the harness can emit, read from ``row["class"] = ...`` in
#: ``tools/validate_vs_fortran.py`` (the module docstring at :24-38 names the
#: same seven), with the line that emits it.  A reviewer can check this table
#: against that source; the test below then *runs* each one.  The four
#: subtagged ``FORTRAN-FAIL`` rows are not literals in the source -- they are
#: built from :data:`FORTRAN_FAIL_SUBTAGS` by an f-string at :1984 -- so they
#: are verified against that mapping instead (see below).
HARNESS_CLASSES = {
    "MATCH": 2038,                    # both ran, every significant channel <= tol
    "DEVIATION": 2038,                # both ran, some channel > tol
    "PORT-ONLY(implicit)": 1962,      # deck uses the port's /IMPL cards
    "PORT-ONLY(dialect)": 1967,       # no verified translator for the deck
    "PORT-ONLY(starter-reject)": 1970,  # translated deck still refused
    "PYRADIOSS-FAIL": 2017,           # the port failed on its own example
    "SKIPPED-SLOW": 2014,             # --timeout or --budget hit, port side
    "NO-CHANNELS": 2047,              # nothing comparable (also :2088, :2109)
    "FORTRAN-FAIL": 1984,             # bare: engine-fail, th2csv-fail
    "FORTRAN-FAIL(t01-unreadable)": 2067,
}

#: The subtagged FORTRAN-FAIL classes the harness builds at :1984 from this
#: mapping.  Derived from the source, never hand-listed, so adding a status
#: there without deciding its fatality fails the test below.
_FORTRAN_FAIL_SUBTAGS = re.compile(
    r"FORTRAN_FAIL_SUBTAGS\s*=\s*\{(.*?)\n\}", re.S)
_SUBTAG_ENTRY = re.compile(r'"([a-z0-9-]+)"\s*:\s*(None|"([a-z0-9-]+)")')


def _harness_fortran_fail_classes() -> set:
    """The ``FORTRAN-FAIL*`` classes the harness can build, from its own source."""
    source = (REPO / "tools" / "validate_vs_fortran.py").read_text(encoding="utf-8")
    block = _FORTRAN_FAIL_SUBTAGS.search(source)
    assert block, "FORTRAN_FAIL_SUBTAGS is gone; the subtag vocabulary moved"
    classes = set()
    for status, quoted, subtag in _SUBTAG_ENTRY.findall(block.group(1)):
        classes.add("FORTRAN-FAIL" if subtag == "" else f"FORTRAN-FAIL({subtag})")
        assert status, block.group(1)
    return classes


#: The two classes that are not a failure, and why.  MATCH is the thing the
#: job exists to produce; PORT-ONLY(...) is the harness's own vocabulary for a
#: deck the FORTRAN chain cannot be asked about -- a port-only /IMPL card or an
#: old-dialect deck -- which is not a disagreement between two solvers.
NOT_FATAL_CLASSES = ("MATCH", "PORT-ONLY")


def test_every_harness_class_is_decided_and_the_decision_is_fatal_by_default(
        tmp_path):
    """The verdict is an ALLOWLIST over the harness's real vocabulary.

    The round-1 Critical finding was that the old fail list was
    ``DEVIATION | PYRADIOSS-FAIL | FORTRAN-FAIL*`` plus an empty table, so a
    table of ``NO-CHANNELS`` rows -- nothing compared at all -- exited 0.  Every
    class the harness can emit is therefore run through the workflow's own
    script here, and the expectation is derived from :data:`NOT_FATAL_CLASSES`
    rather than from a hand-copied fail list, so adding a class to the harness
    makes this test fail until someone decides what it means.

    ``NO-CHANNELS`` and ``SKIPPED-SLOW`` are the load-bearing rows: both mean
    nothing was compared on that example, and the harness's own comment on the
    first (:2041-2047) is that a MATCH derived from nothing comparable "is the
    worst thing this file could print".
    """
    source = (REPO / "tools" / "validate_vs_fortran.py").read_text(encoding="utf-8")
    for cls in HARNESS_CLASSES:
        assert f'"{cls}"' in source, (
            f"{cls} is in HARNESS_CLASSES but the harness source no longer "
            f"emits it; decide its fatality and update the table")

    vocabulary = set(HARNESS_CLASSES) | _harness_fortran_fail_classes()
    for cls in sorted(vocabulary):
        expect_zero = cls.startswith(NOT_FATAL_CLASSES)
        # a MATCH row has to be present or the table-level rule fires instead
        rows = [{"example": "probe", "class": "MATCH", "max_rel_rms": 0.0}]
        if cls != "MATCH":
            rows.append({"example": "case_under_test", "class": cls})
        result = _run_verdict(rows, tmp_path)
        assert (result.returncode == 0) is expect_zero, (
            f"class {cls!r}: exit {result.returncode}, expected "
            f"{'0 (not fatal)' if expect_zero else '1 (FATAL)'}\n"
            f"--- stdout ---\n{result.stdout}")
        if not expect_zero:
            assert "::error title=parity::" in result.stdout, result.stdout


def test_a_table_that_compared_nothing_fails_in_every_way_it_can_be_empty(
        tmp_path):
    """Three distinct ways to compare nothing, three failures.

    Each is reachable with a CORRECTLY configured oracle and no
    misconfiguration, which is why the round-1 reviewer called this the worse
    of the two Criticals: ``NO-CHANNELS`` is the LAW4-cfg shape
    (``docs/OPEN_BUGS.md`` item 6 -- the two solvers stop producing comparable
    channels) that this milestone exists to instrument.

    * an empty table (``parity()`` exits 0 even when it ran no example);
    * a table whose only rows are ``NO-CHANNELS``;
    * a table of ``PORT-ONLY`` rows and no ``MATCH`` -- "compared nothing" in a
      different hat, and the one a per-row rule alone would wave through.
    """
    for label, rows in (
            ("empty table", []),
            ("NO-CHANNELS only",
             [{"example": "law4_cfg", "class": "NO-CHANNELS"}]),
            ("SKIPPED-SLOW only",
             [{"example": "big_deck", "class": "SKIPPED-SLOW"}]),
            ("PORT-ONLY only",
             [{"example": "imp", "class": "PORT-ONLY(implicit)"},
              {"example": "old", "class": "PORT-ONLY(starter-reject)"}]),
            ("a MATCH plus one NO-CHANNELS",
             [{"example": "tensile_bar", "class": "MATCH", "max_rel_rms": 0.01},
              {"example": "law4_cfg", "class": "NO-CHANNELS"}]),
    ):
        result = _run_verdict(rows, tmp_path)
        assert result.returncode == 1, (
            f"{label}: exit {result.returncode}, expected 1 -- a green parity "
            f"job that compared nothing\n--- stdout ---\n{result.stdout}")


def test_a_real_comparison_still_passes(tmp_path):
    """The allowlist did not become a blanket red: MATCH is a pass.

    A row the harness classifies ``MATCH`` alongside ``PORT-ONLY(...)`` rows --
    the shape a healthy sweep has, ``PORT-ONLY(implicit)`` being 27 of the 81
    rows in the recorded ``parity_m41.json`` -- must exit 0, and must SAY how
    much was compared so a reader of the log can see evidence was produced.
    """
    rows = [{"example": "tensile_bar", "class": "MATCH", "max_rel_rms": 0.004},
            {"example": "imp_bracket", "class": "PORT-ONLY(implicit)"},
            {"example": "old_dialect", "class": "PORT-ONLY(starter-reject)"}]
    result = _run_verdict(rows, tmp_path)
    assert result.returncode == 0, (
        f"a MATCH-bearing table must pass; exit {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    assert "compared 1 of 3 example(s)" in result.stdout, result.stdout
    for row in rows:
        assert row["example"] in result.stdout, result.stdout


def test_a_class_nobody_has_heard_of_is_fatal(tmp_path):
    """The allowlist fails CLOSED.

    A future harness revision that learns a new class must not be able to turn
    the parity job green by emitting it; a row with no ``class`` at all must not
    either.  Both are fatal here, which is the opposite of the old rule's
    shape -- a denylist only knows what it was told about.
    """
    for rows in ([{"example": "future", "class": "SOMETHING-NEW"}],
                 [{"example": "no_class_key_at_all"}],
                 [{"example": "t", "class": None}]):
        result = _run_verdict(rows, tmp_path)
        assert result.returncode == 1, (
            f"{rows} : exit {result.returncode}, expected 1\n{result.stdout}")


def test_the_verdict_runs_as_a_shell_payload_too(tmp_path):
    """The committed ``run:`` block survives ``bash -e``, not just the regex.

    The script above is extracted, so this is the half that proves the *step*
    works: the whole payload, exactly as Actions hands it to bash, with
    ``PARITY_JSON`` exported.  Skipped with a reason where there is no bash --
    the extraction tests above are parser-only and still run there.
    """
    bash = shutil.which("bash")
    if not bash:                             # pragma: no cover — Windows
        pytest.skip("no bash on PATH; the payload's shell quoting is unverified "
                    "here (the extracted-script tests above still ran)")
    payload = _parity_gate(_parse_jobs(_workflow(PARITY))).step_using("$PARITY_JSON")
    table = tmp_path / "parity_results.json"
    table.write_text(json.dumps([{"example": "law4_cfg", "class": "NO-CHANNELS"}]),
                     encoding="utf-8")
    env = dict(os.environ, PARITY_JSON=str(table))
    result = subprocess.run([bash, "-e", "-c", payload["run"]],
                            capture_output=True, text=True, env=env, cwd=str(REPO))
    assert result.returncode == 1, (
        f"the payload exited {result.returncode} on a NO-CHANNELS table, "
        f"expected 1\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")