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

import re
from pathlib import Path

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
# PyYAML is deliberately NOT a dependency of this repository: it is not in
# requirements-lock.txt §[B], the CI jobs install only what the suite needs,
# and a test that needs a package the gate does not install is a test that
# cannot run in CI.  So the two helpers below parse the *shape* this task
# needs -- top-level keys, jobs, and each job's steps -- from indentation,
# which is the part a substring assertion cannot reach.  A real YAML parse is
# still done -- by :func:`test_both_workflows_are_valid_yaml`, which uses a real
# parser when one is importable and a targeted rule when one is not, because
# these helpers are a parser and a parser can be wrong.
#
# It was.  The hand parser below read ``- name: Read the rows back: a deviation
# is a failure`` as three perfectly good fields and every test in this file
# passed, while PyYAML rejected the file outright: an unquoted scalar holding
# ``": "`` is a *nested mapping* to YAML, not a name.  The rule that catches it
# lives in :func:`_unquoted_colon_values` below, and that defect is why the
# YAML check is a test and not a note in the task report.


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


def test_both_workflows_are_valid_yaml():
    """A real parser when one is importable, a targeted rule when one is not.

    PyYAML is not in requirements-lock.txt and must not be added (the CI jobs
    install only what the suite needs, so a test that needs PyYAML could not
    run in CI).  So this **never skips**: with PyYAML importable it parses both
    files and asserts the top-level shape; without it, it applies the rule that
    catches the one syntax error this repository's workflows have actually
    produced (:func:`_unquoted_colon_values`).  Either way something is
    asserted, and the weaker branch says in its message which one ran.
    """
    try:
        import yaml
    except ImportError:
        yaml = None

    for path in (CI, PARITY):
        if yaml is not None:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert isinstance(doc, dict), path.name
            assert isinstance(doc.get("jobs"), dict) and doc["jobs"], path.name
            for job in doc["jobs"].values():
                assert "runs-on" in job, path.name
                assert job.get("steps"), path.name
        else:
            suspects = _unquoted_colon_values(path.read_text(encoding="utf-8"))
            assert not suspects, (
                f"{path.name}: unquoted value(s) containing ': ' -- YAML reads "
                f"these as a nested mapping, so the workflow will not load; "
                f"quote the value:\n  " + "\n  ".join(suspects))


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
    """
    jobs = _parse_jobs(_workflow(CI))
    check = jobs["ledger-gate"].step_using("--check")
    paths = re.findall(r"--check\s+(\S+)", check["run"])
    assert paths == [LEDGER_BASELINE], paths
    assert (REPO / LEDGER_BASELINE).is_file(), LEDGER_BASELINE
    assert not (REPO / "tools" / "validation_data" / "baseline_ci.json").exists(), (
        "a CI-specific record would have to be recorded in the environment it "
        "is compared in; if one exists, the gate should point at it instead of "
        "accepting an environment drift")


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
    """
    jobs = _parse_jobs(_workflow(CI))
    for step in jobs["ledger-gate"].steps:
        assert "-n auto" not in step["run"], step["run"]
        assert "--dist" not in step["run"], step["run"]


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
    """Configuration is not the same as availability, and must not be assumed.

    ``pyradioss/paths.py`` resolves a resource or fails loudly, naming every
    candidate (``validate_vs_fortran.py:parity`` then exits 2 with that text).
    That is right for a tool and wrong for a CI job: a repository variable
    holding a path from the maintainer's machine would turn every pull request
    red.  So the job checks the binaries are executable **first**, and skips
    with a notice if they are not -- which is also why no step may carry
    ``continue-on-error``.
    """
    jobs = _parse_jobs(_workflow(PARITY))
    gate = _parity_gate(jobs)
    preflight = gate.step_using("OR_STARTER")
    assert "-x" in preflight["run"] or "-f" in preflight["run"], preflight["run"]
    assert "GITHUB_OUTPUT" in preflight["run"], preflight["run"]

    sweep = gate.step_using("validate_vs_fortran.py")
    assert sweep["if"], "the sweep must be gated on the preflight's output"
    assert "steps." in sweep["if"], sweep["if"]


def test_the_parity_verdict_is_mechanical_too():
    """``validate_vs_fortran.py parity`` exits 0 on a completed sweep whatever
    the channels said (``:2163``) -- so a job that only runs it would be green
    on a DEVIATION.  The row classes are read back and a deviation is a
    failure."""
    jobs = _parse_jobs(_workflow(PARITY))
    gate = _parity_gate(jobs)
    verdict = gate.step_using("$PARITY_JSON")
    assert verdict["env"].get("PARITY_JSON", "").endswith("parity_results.json"), \
        verdict["env"]
    assert "DEVIATION" in verdict["run"], verdict["run"]
    # ... and an EMPTY table is a failure too: a sweep that compared nothing
    # is not evidence, and `parity()` exits 0 for it.
    assert re.search(r"sys\.exit\(1 if bad\b", verdict["run"]), verdict["run"]
    assert "not rows" in verdict["run"], verdict["run"]
    assert verdict["if"], "the verdict step must share the preflight's gate"