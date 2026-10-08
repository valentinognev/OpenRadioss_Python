#!/usr/bin/env python3
"""The regression ledger: make "no new failures relative to the branch's
recorded baseline" a measurement instead of a judgement call.

``plan/00_ORCHESTRATION.md`` §4.2 makes the fast tier the gate for every
milestone, and every task in this program has to show "no new failures".
Until now that claim was a reader's judgement of a pytest summary line --
and Phase 0 quoted those counts wrong three times, each time costing a
reviewer a finding.  A bare count rots because nobody can say *which* tests
it counted.  This tool replaces the line with a record:

* :func:`collect` runs the fast tier in a **subprocess** and reads a
  machine-readable report out of it -- node ids, not counts;
* :func:`compare` turns two records into a :class:`Delta`;
* ``main`` is the gate: ``--record`` writes a record, ``--check`` refuses
  one that grows a failure.

The one requirement the plan states explicitly
(``plan/02_phase1_foundation.md:462`` and its reviewer checklist at ``:532``)
is that **a renamed test lands in ``unknown``, not in ``new_failures``**.
The cases that requirement has to survive are three, not one, and they are
the reason this module carries a ``selected`` set instead of only failure
lists.

The discrimination
------------------
A node id is ``file.py::a::b::name``.  Given the recorded failures of two
runs there are exactly three ways a recorded failure can *stop* appearing,
and they must not collapse into each other:

1. it is **fixed** -- the same node id is still collected and is green now;
2. it was **renamed** -- a different node id in the *same file* is failing
   now, and the new id did not exist in the baseline;
3. it was **deleted** -- nothing collected now carries its name.

Only (1) is evidence that anything got better, so only (1) may be called
``fixed``.  A rename must not read as a fix either, because "delete the red
test" would otherwise silence it silently -- hence the separate bucket.  The
plan's own example (``:449``) is case (2) with both runs failing, and it
asserts ``unknown == ()``: the pair is *resolved*, not merely quarantined,
which is why :attr:`Delta.renames` carries it and :attr:`Delta.unknown` does
not.

The decision procedure, in the order it is applied (``compare``):

* ``fixed`` = (was failing, is not failing now) AND the id is **still
  collected**.  Existence is read from the current run's collected set -- the
  set of ids this run recorded for :attr:`Ledger.examined_files`, which is
  the union of *both* runs' failing files (see "Why the record keeps a
  restricted selected set"), so it is evidence, not a guess.
* everything else that was failing and is not failing now is **gone**, split
  into *renamed* and *deleted*;
* a gone id is a **rename partner** of a now-failing id when they share a
  file, have different *skeletons* (the node ids with the ``[...]``
  parameters stripped -- so ``t[1]`` -> ``t[2]`` is a new parameter set, not
  a rename) and each is the other's **only** candidate in that file.
  Uniqueness is what makes two simultaneous swaps in one file fall through
  to "unresolved" instead of a coin flip;
* a now-failing id is a **regression** (never a rename candidate) when the
  baseline collected it too -- it is a test that existed and used to pass;
* whatever is left over is a **new failure**.

Every branch that cannot decide falls towards *report more*, never towards
*report less*: an unpairable id stays a new failure and its partner stays
unresolved.

The one cost, named on purpose
------------------------------
"Delete the failing test, add a new failing test in the same file" is
indistinguishable from a rename **in the results** -- nothing was measured,
so nothing can decide it.  The ledger pairs them, prints the pair by name in
every report, and then **refuses the pair**: a gate whose whole job is "no
silent progress claims" must not let the one case it cannot decide through
quietly.  A rename is therefore fatal unless the reader asks for the
permissive reading with ``--allow-renames``, and the remedy is the same as
for a deletion -- ``--record --force``, a deliberate two-command act whose
commit message is where the rename gets justified.  The cost of that ruling
is named in the report: an ordinary rename of a *failing* test now costs an
11-minute recording.  Pinned by
``test_the_documented_cost_of_a_delete_plus_add_in_one_file`` and
``test_a_rename_is_fatal_unless_it_is_allowed``.

Why the record keeps a *restricted* selected set
-----------------------------------------------
Resolving (1) and (3) needs to know whether an id was still collected, which
the collected node-id set answers exactly.  Recording all ~14,600 of them
would put about a megabyte of churn in a committed file for no gain, and
there is a sharper claim: **every id whose existence the diff can ask about
is a failure of one of the two runs being compared.**  A fixed or deleted id
was failing in the baseline, so its file carries a baseline failure; a rename
partner lives in that same file.  So each record keeps the collected ids of
the files named in :attr:`Ledger.examined_files`, which is always the union
of *both* runs' failing files: :func:`collect` is handed the other run's
files as ``examined_files``, and ``--record`` takes them from ``--against``
(default: the committed ``tools/validation_data/baseline.json``, so CI's
one-run-then-compare shape needs no flag).

Widening on the current run only is the defect this tool shipped first, and
it was fatal to the common case: existence read from the *current* run's
failing files cannot answer for a file whose last red test just went green,
so the id was absent, ``fixed`` came out empty and the id fell into
``unknown`` -- the Phase 1 exit gate exited 1 on a tree with **zero**
failures against a record with one, and called the improvement a deletion.
The union costs the size of the files a recorded failure lives in: a red test
in a 4000-test file records that file's 4000 ids.  ``--check --results``
prints, by name, any baseline-failing file whose ids the record it was handed
does not carry, so an unanswerable question is announced instead of being
answered wrongly.

What the record does NOT cover: a deleted **passing** test is invisible, and
so is a rename of one (they are the same hole: neither id is in either
``bad`` set, and ``--strict-renames``-style teeth cannot reach them).  This
is a ledger of failures, and a coverage question is a different tool.

The environment
---------------
Phase 0 measured that this project's counts are environment-dependent --
``docs/STATE.md``'s three rows differ because the oracle is resolvable or
not, and the compare/record is pinned to the NumPy backend.  A baseline
recorded on one box and compared on another is therefore not evidence of
anything, so :func:`environment` measures the axes that move the counts
(interpreter, the optional packages' versions, the backend pin, and whether
``$OR_SRC`` / ``$OR_ROOT`` / the CFG tree resolve) and the record stores the
result as an :func:`environment_key` string.  :func:`compare` refuses to
produce a verdict across a mismatch: ``main --check`` exits **2**, names
both keys and every differing field, and prints no delta at all.  The
refusal is a **pre-flight**: the child's environment is fully determined by
:func:`child_command` without executing anything, so ``--check`` compares the
keys *before* the suite runs (measured: 0.24 s against the committed record)
rather than spending an 11-minute fast tier to decline afterwards.  With
``--allow-environment-drift`` the comparison is *advisory*: new failures
stay fatal (a test failing on ``ubuntu-latest`` is real), while fixes and
deletions stop being fatal (an oracle test that skips because ``$OR_SRC`` is
absent is neither).  That is the shape Task P1.9's CI job needs, where the
oracle is not configured.

Usage
-----
    # record the baseline (this runs the fast tier; ~13 min here)
    python tools/regression_ledger.py --record \\
        tools/validation_data/baseline.json

    # the gate: exit 0 clean, 1 a new failure, 2 another environment
    python tools/regression_ledger.py --check \\
        tools/validation_data/baseline.json

    # reuse a record instead of running the suite again (what CI does)
    python tools/regression_ledger.py --check baseline.json --results now.json

``--check`` writes nothing at all: it reads the baseline and prints.  Only
``--record`` writes, and it refuses to overwrite an existing record without
``--force`` -- a committed record must never be rewritten by accident
(Phase 0's rule).  That refusal is the **first** thing either mode does: it
costs a second, and asking after an 11-minute run whether the answer may be
saved is the expensive order of the two questions.

The collection itself
---------------------
Running pytest from inside pytest is fragile, so :func:`collect` shells out,
following the convention in ``tools/oracle/oracle_selftest.py:835`` and
``tools/benchmark.py:61``: a fresh subprocess, ``cwd`` at the repo root, the
environment captured explicitly, output captured rather than inherited.  The
child loads a plugin (written to a **temporary** directory, never into the
checkout) that reports every node id through ``pytest_runtest_logreport``,
which is the only hook that gives the exact node id for a parametrised test
-- a ``--junit-xml`` round trip would have to rebuild them from
``classname``+``name`` and would mangle any id containing a comma or a
quote.  The child runs with ``PYTHONDONTWRITEBYTECODE=1`` and
``-p no:cacheprovider`` so that a gate run leaves no ``__pycache__``
and no ``.pytest_cache`` behind;
``tests/test_p1_regression_ledger.py`` pins that without running the suite.

The default selection is the project's own gate -- ``-m "not slow"`` with
``PYRADIOSS_BACKEND=numpy`` (``plan/00_ORCHESTRATION.md`` §1.1.6 pins the
backend for any before/after comparison) -- and an already-set
``PYRADIOSS_BACKEND`` in the parent environment is respected rather than
overwritten, so an operator who ran the suite with another pin is compared
against what they actually ran.

The plan's "13 known failures"
------------------------------
``plan/02_phase1_foundation.md:461`` asks the record to include "the 13 known
failures documented in ``docs/STATE.md:34``".  **That text is stale**: Phase 0
found those failures unreproducible and corrected the record, and
``docs/STATE.md:34`` no longer claims them.  The tree this records has no
failing and no erroring node; ``tests/conftest.py``'s ``_KNOWN_XFAIL`` is the
real standing exception and shows up as ``xfailed``, not as a failure.  The
tool records what is there, and
``test_the_recorded_baseline_exists_and_says_what_the_tree_says`` fails if
that ever stops being true.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "SCHEMA",
    "BASELINE_PATH",
    "Ledger",
    "Delta",
    "collect",
    "compare",
    "load",
    "main",
    "environment",
    "environment_key",
    "child_command",
]

REPO = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO / "tools" / "validation_data" / "baseline.json"

#: Bumped only by a deliberate decision; ``load`` refuses anything else
#: rather than guessing at a field it does not know.
SCHEMA = "pyradioss-regression-ledger/1"

#: The fast tier on the measured dev box is ~13 min; 90 min is generous
#: enough for a loaded machine and short enough to be a real timeout.
DEFAULT_TIMEOUT = 5400.0

#: Exit codes.  1 is the gate ("a new failure"), 2 is "not comparable", 3
#: is "the tool could not produce a comparison at all".
EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_ENVIRONMENT = 2
EXIT_ERROR = 3


# ---------------------------------------------------------------------------
# The two records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ledger:
    """One measured run of the suite.

    ``failed``/``errors`` are full node ids, because a count cannot say
    *which* tests it counted and a wrong count is a finding waiting to
    happen.  ``selected`` is the number of collected tests, and the four
    counts partition it exactly (asserted by
    ``test_collect_partitions_every_collected_test_and_sees_the_xfails``).

    ``selected_in_failed_files`` is the collected node ids of the files in
    ``examined_files`` -- the whole of the existence evidence
    :func:`compare` needs, and nothing more; see the module docstring for
    why the union of the two runs' failing files is exact.  ``examined_files``
    is recorded so a comparison can say *out loud* when the record it was
    handed carries no evidence for a file the other run is red in, instead of
    answering "did this id go green or was it deleted" with a guess.  A
    record written before this field existed derives it from
    ``selected_in_failed_files``, which is what those records meant.
    """

    passed: int = 0
    failed: tuple = ()
    skipped: int = 0
    errors: tuple = ()
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0
    selected: int = 0
    selected_in_failed_files: tuple = ()
    examined_files: tuple = ()
    environment: Mapping[str, object] = field(default_factory=dict)
    environment_key: str = ""
    command: str = ""
    recorded: str = ""

    @property
    def bad(self) -> tuple:
        """Every node id that is not green: failures and errors together.

        The rename/deletion question does not care which of the two a node
        is in -- it asks about the id -- so the discrimination runs on this
        set and the result is split again at the end.  Deduplicated in first-
        seen order: ``classify()`` cannot put one id in both lists, but a
        hand-written record can, and an id processed twice would be reported
        twice.
        """
        return tuple(dict.fromkeys(
            tuple(self.failed) + tuple(self.errors)))

    @property
    def examined(self) -> frozenset:
        """The files whose collected ids this record actually carries.

        Falls back to deriving them from ``selected_in_failed_files`` for a
        record that predates the field.
        """
        if self.examined_files:
            return frozenset(self.examined_files)
        return frozenset(node.split("::", 1)[0]
                         for node in self.selected_in_failed_files)


@dataclass(frozen=True)
class Delta:
    """The comparison of a baseline against the current run.

    ``new_failures``/``new_errors`` are the gate.  ``fixed`` and
    ``unknown`` are the two ways a baseline failure may stop appearing that
    are *not* evidence of progress.  ``renames`` is the resolved rename
    pairs: the plan's ``unknown`` bucket is empty for them (``:449``
    asserts exactly that), so they are reported here instead of being
    silently dropped.
    """

    new_failures: tuple = ()
    fixed: tuple = ()
    new_errors: tuple = ()
    unknown: tuple = ()
    renames: tuple = ()
    environment_changed: bool = False
    environment_diff: tuple = ()
    advisory: bool = False

    @property
    def regressions(self) -> tuple:
        return tuple(self.new_failures) + tuple(self.new_errors)


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

#: The variables that change which tests run or how they behave.  Anything
#: not listed is not measured, and the list is deliberately short: a
#: fingerprint full of incidental values reports drift nobody can act on.
_ENV_VARS = ("PYRADIOSS_BACKEND", "PYRADIOSS_ORACLE_REQUIRED",
             "PYRADIOSS_ALLOW_MISSING_DEPS")

#: Resources whose *presence* (not path -- the path differs per machine and
#: comparing it would report drift on every clone) decides whether the oracle
#: tests run or skip.
_RESOURCES = {"OR_SRC": "or_src", "OR_ROOT": "or_root", "hm_cfg": "hm_cfg_dir"}

_PACKAGES = ("numpy", "scipy", "numba", "pytest")


def _version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
    except Exception:                      # absent, or unimportable
        return "absent"
    return str(getattr(mod, "__version__", "unknown"))


def _resolvable(name: str) -> bool:
    """True when ``pyradioss.paths`` resolves ``name`` **and** it exists.

    The resolvers raise rather than return ``None``, and a set-but-stale
    variable raises too (``pyradioss/paths.py``, ``_resolve``) -- both are
    "the resource is not usable here", which is the only question asked.
    """
    from pyradioss import paths
    try:
        return os.path.exists(str(getattr(paths, name)()))
    except Exception:
        return False


def environment(overrides: Mapping[str, Mapping[str, str]] = None) -> dict:
    """Measure the axes on which this project's counts are known to move.

    ``docs/STATE.md``'s three baseline rows differ by two tests, and the
    reason is the oracle: present, the oracle tests run; absent, they skip.
    That is the axis this records, together with the interpreter and the
    optional packages' versions, so a comparison across two boxes is refused
    rather than believed.

    ``overrides`` replaces the measured ``env`` mapping -- the parent process
    is not the child, so :func:`collect` passes the child's own variables in
    order to record the environment the suite actually ran in.  Recording the
    parent's instead would put ``PYRADIOSS_BACKEND=<unset>`` in the record of
    a run that was pinned to numpy, and the key would then be wrong for the
    next comparison as well as for the reader.
    """
    env = {var: os.environ.get(var, "<unset>") for var in _ENV_VARS}
    if overrides:
        env.update(overrides.get("env") or {})
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": {name: _version(name) for name in _PACKAGES},
        "env": env,
        "resources": {label: _resolvable(fn)
                      for label, fn in _RESOURCES.items()},
    }


def environment_key(env: Mapping[str, object]) -> str:
    """A one-line, deterministic digest of an :func:`environment` mapping.

    Two keys that are equal mean the counts are comparable; the diff printed
    alongside them says which field moved.  It is a *digest*, not a hash:
    a reviewer must be able to read it in the CI log.
    """
    if not env:
        return ""
    env_part = ",".join(
        f"{var}={val}" for var, val in sorted(
            (env.get("env") or {}).items())) or "-"
    res_part = ",".join(
        f"{label}={'present' if val else 'absent'}" for label, val in
        sorted((env.get("resources") or {}).items())) or "-"
    pkgs = env.get("packages") or {}
    return (f"{env.get('platform', '?')}/py{env.get('python', '?')}"
            f"/{'+'.join(f'{n}{pkgs[n]}' for n in sorted(pkgs)) or '-'}"
            f"/{env_part}/{res_part}")


def _flatten(env: Mapping[str, object], prefix: str = "") -> dict:
    """``{'packages.numpy': '2.5.3', ...}`` -- one flat mapping so a
    mismatch report can name the field that moved, not just the two blobs."""
    out: dict = {}
    for key, value in env.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def environment_diff(base: Mapping[str, object],
                     now: Mapping[str, object]) -> tuple:
    """Every field that differs, as ``name: baseline=<a> now=<b>`` lines."""
    left, right = _flatten(base), _flatten(now)
    lines = []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name, "<absent>"), right.get(name, "<absent>")
        if a != b:
            lines.append(f"{name}: baseline={a!r} now={b!r}")
    return tuple(lines)


# ---------------------------------------------------------------------------
# collect(): run the suite in a subprocess and read a machine-readable report
# ---------------------------------------------------------------------------

#: Loaded by the child via ``-p <name>``.  It writes its report to the file
#: ``LEDGER_OUT`` names and imports nothing from this repository, so it cannot
#: fail on an import error in the thing it is measuring.
PLUGIN_NAME = "_pyradioss_ledger_plugin"

_PLUGIN_SOURCE = '''\
"""Written by tools/regression_ledger.py into a temporary directory.

Reports every collected node id and every per-phase outcome to the file
LEDGER_OUT names.  It is generated rather than committed so that a gate run
writes nothing into the checkout, which is the same rule the tool enforces on
its own baseline.
"""

import json
import os

_OUT = os.environ["LEDGER_OUT"]
_collected = []
_deselected = []
_reports = {}


def pytest_deselected(items):
    for item in items:
        _deselected.append(item.nodeid)


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        _collected.append(item.nodeid)


def pytest_runtest_logreport(report):
    _reports.setdefault(report.nodeid, []).append(
        [report.when, report.outcome, hasattr(report, "wasxfail")])


def pytest_sessionfinish(session, exitstatus):
    payload = {
        "collected": _collected,
        "deselected": _deselected,
        "reports": [[nodeid, phases] for nodeid, phases in _reports.items()],
        "exitstatus": int(exitstatus),
    }
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
'''


def child_command(pytest_args: Sequence[str] = (),
                  backend: str | None = "numpy") -> tuple:
    """``(argv, env)`` for the child pytest run.

    Split out of :func:`collect` so the *contract* -- the selection, the
    backend pin, ``PYTHONDONTWRITEBYTECODE`` and ``-p no:cacheprovider`` --
    is testable in milliseconds instead of by running the suite.
    ``backend=None`` leaves the parent's own pin alone; a ``backend`` that is
    already set in the parent wins too, because the ledger must describe the
    run that actually happened.
    """
    env = dict(os.environ)
    if backend is not None and "PYRADIOSS_BACKEND" not in env:
        env["PYRADIOSS_BACKEND"] = backend
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    argv = [sys.executable, "-m", "pytest", "-q", "-p", PLUGIN_NAME,
            "-p", "no:cacheprovider", "-m", "not slow"]
    argv.extend(str(a) for a in pytest_args)
    return argv, env


def classify(collected: Sequence[str], deselected: Sequence[str],
             reports: Mapping[str, Sequence[Sequence]]) -> dict:
    """Fold per-phase reports into the six outcome counts.

    The split is the one pytest's own summary line makes, so the record and
    the line a reviewer reads can be compared:

    * a failure in the ``call`` phase is a **failure**; a failure in
      ``setup``/``teardown`` is an **error** (that is the difference between
      "the assertion failed" and "the test never ran");
    * a skip carrying ``wasxfail`` is **xfailed**; a pass carrying it is
      **xpassed** (and an ``xfail(strict=True)`` that passes arrives here as
      a failure, exactly as pytest reports it);
    * a collected node with no report at all is an **error** -- it did not
      pass, and reporting it as anything else is how a count rots.
    """
    deselected_set = set(deselected)
    selected = [node for node in collected if node not in deselected_set]
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0,
              "xfailed": 0, "xpassed": 0}
    failures: list = []
    errors: list = []
    for node in selected:
        phases = reports.get(node) or ()
        if not phases:
            errors.append(node)
            counts["errors"] += 1
            continue
        if any(outcome == "failed" and when == "call" for when, outcome, _ in
               phases):
            failures.append(node)
            counts["failed"] += 1
        elif any(outcome == "failed" for _when, outcome, _xf in phases):
            errors.append(node)
            counts["errors"] += 1
        elif any(outcome == "skipped" and xfail for _w, outcome, xfail in
                 phases):
            counts["xfailed"] += 1
        elif any(outcome == "passed" and xfail for _w, outcome, xfail in
                 phases):
            counts["xpassed"] += 1
        elif any(outcome == "skipped" for _w, outcome, _x in phases):
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return {
        "selected": selected,
        "counts": counts,
        "failed": sorted(failures),
        "errors": sorted(errors),
        "deselected": len(deselected),
    }


def collect(pytest_args: Sequence[str] = (), *,
            timeout: float = DEFAULT_TIMEOUT,
            backend: str | None = "numpy",
            examined_files: Sequence[str] = ()) -> Ledger:
    """Run the suite in a subprocess and return its :class:`Ledger`.

    ``pytest_args`` defaults to the project's own gate selection
    (``-m "not slow"``, see :func:`child_command`); a test passes a path or
    a module name to collect something small.  The child writes its report
    into a temporary directory that is removed on the way out, so the
    checkout is untouched -- that is the property Phase 0 lost three rounds
    to, and it is why the plugin is generated into a temp dir rather than
    committed under ``tools/``.

    ``examined_files`` is the *other* run's failing files: this run records
    the collected ids of those files as well as of its own, because
    "was failing in the baseline and is not failing now" is only a **fix**
    if the id is still collected, and an id in a file that just went green is
    absent from this run's failing files.  Reading existence from the failing
    files alone is the bug this signature exists to prevent: it makes the
    most common event there is -- fix the last red test in a file -- read as a
    deletion.  :func:`main` passes the baseline's bad files for ``--check``
    and the record's future counterpart's for ``--record``.
    """
    argv, env = child_command(pytest_args, backend=backend)
    with tempfile.TemporaryDirectory(prefix="pyradioss-ledger-") as work:
        plugin = Path(work) / (PLUGIN_NAME + ".py")
        plugin.write_text(_PLUGIN_SOURCE, encoding="utf-8")
        report = Path(work) / "report.json"
        env["LEDGER_OUT"] = str(report)
        env["PYTHONPATH"] = os.pathsep.join(
            [work] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        started = time.perf_counter()
        proc = subprocess.run(argv, cwd=str(REPO), env=env, text=True,
                              capture_output=True, errors="replace",
                              timeout=timeout, check=False)
        elapsed = time.perf_counter() - started
        if not report.is_file():
            tail = (proc.stdout + proc.stderr)[-2000:]
            raise RuntimeError(
                f"the child pytest run produced no report (exit "
                f"{proc.returncode}); its last output was:\n{tail}")
        payload = json.loads(report.read_text(encoding="utf-8"))
    parsed = classify(payload["collected"], payload["deselected"],
                      {node: phases for node, phases in payload["reports"]})
    env_record = environment(
        {"env": {var: env.get(var, "<unset>") for var in _ENV_VARS}})
    # The union, and it is the whole point: a file the *other* run is red in
    # is examined even when this run is green there.  What is recorded is the
    # files that actually *contributed* ids -- a record that claims to have
    # examined a file it collected nothing from would answer "did this id go
    # green" with a silence that reads like evidence.
    keep = tuple(sorted(node for node in parsed["selected"]
                        if node.split("::", 1)[0]
                        in {node.split("::", 1)[0] for node in
                            parsed["failed"] + parsed["errors"]}
                        | {str(name) for name in examined_files}))
    examined = tuple(sorted({node.split("::", 1)[0] for node in keep}))
    counts = parsed["counts"]
    return Ledger(
        passed=counts["passed"],
        failed=tuple(parsed["failed"]),
        skipped=counts["skipped"],
        errors=tuple(parsed["errors"]),
        xfailed=counts["xfailed"],
        xpassed=counts["xpassed"],
        deselected=parsed["deselected"],
        selected=len(parsed["selected"]),
        selected_in_failed_files=keep,
        examined_files=tuple(examined),
        environment=env_record,
        environment_key=environment_key(env_record),
        command=_render_command(argv),
        recorded=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def _render_command(argv: Sequence[str]) -> str:
    """The child command as it was run, in full.

    Nothing is filtered out of it: a record that claims a selection it did
    not run is exactly the lie this tool exists to stop, so the interpreter
    path and every flag are kept and only whitespace is quoted.
    """
    return " ".join(f"'{arg}'" if " " in arg else str(arg) for arg in argv)


# ---------------------------------------------------------------------------
# compare(): the discrimination
# ---------------------------------------------------------------------------

def split_node(node: str) -> tuple:
    """``'tests/test_a.py::T::m'`` -> ``('tests/test_a.py', 'T::m')``.

    The file is the container; everything after the first ``::`` is the test
    path inside it.  A rename keeps the container and changes the rest --
    that is the whole of the rule, and it is why a *parametrised* id change
    (``t[1]`` -> ``t[2]``) is not a rename but a deletion plus a
    regression (see the module docstring).
    """
    path, sep, rest = node.partition("::")
    return path, rest if sep else ""


def skeleton(node: str) -> str:
    """The node id with every ``[...]`` parameter suffix removed.

    ``tests/test_a.py::t[1]`` and ``tests/test_a.py::t[2]`` have the same
    skeleton: that is *one* test with two parameter sets, and swapping which
    of them fails is not a rename.  ``t[1]`` and ``u[1]`` do not, and are.
    Without this the ledger would call a parameter change a rename, which is
    the one rename-shaped difference that is really a new test case.
    """
    parts = []
    for segment in node.split("::"):
        parts.append(segment.split("[", 1)[0])
    return "::".join(parts)


def pair_renames(gone: Sequence[str], fresh: Sequence[str]) -> dict:
    """``{old: new}`` for the rename pairs, or ``{}``.

    A pair is only formed in a file that has **exactly one** gone id and
    **exactly one** candidate now-failing id with a different skeleton (so a
    parameter change is not a rename).  Uniqueness is required in *both*
    directions: a file that lost two recorded failures has no way to say
    which of them became ``new_name``, and a wrong pairing is the one error
    this tool cannot afford -- it would report a real regression as a rename.
    The declined case falls through to "new failure" plus "unresolved", which
    can only ever report more.
    """
    old_by_file: dict = {}
    for node in gone:
        old_by_file.setdefault(split_node(node)[0], []).append(node)
    new_by_file: dict = {}
    for node in fresh:
        new_by_file.setdefault(split_node(node)[0], []).append(node)
    pairs: dict = {}
    for path, olds in old_by_file.items():
        if len(olds) != 1:
            continue
        old = olds[0]
        news = [n for n in (new_by_file.get(path) or [])
                if n != old and skeleton(n) != skeleton(old)]
        if len(news) == 1:
            pairs[old] = news[0]
    return pairs


def _unexamined(base: Ledger, now: Ledger) -> tuple:
    """Baseline-failing files whose collected ids ``now`` does not carry.

    A non-empty answer means the existence question cannot be answered for
    the ids in those files, so :func:`compare` puts them in ``unknown``
    (fatal).  That is the safe direction -- it refuses rather than guesses --
    but on its own it reads as a claim about the code, so :func:`main` prints
    the files and the remedy.  The ordinary case is a record written by
    ``--record`` against a different baseline, or one written before this
    rule existed.
    """
    if not base.bad:
        return ()
    bad_files = {node.split("::", 1)[0] for node in base.bad}
    return tuple(sorted(bad_files - set(now.examined)))


def compare(base: Ledger, now: Ledger) -> Delta:
    """Classify the difference between two records.

    Read the module docstring for the rule; the order below is the rule:

    1. still collected and green now -> ``fixed`` (the only "fixed");
    2. the rest of what stopped failing is *gone*, and a gone id pairs with
       a now-failing id only under :func:`pair_renames`' uniqueness rule;
    3. a now-failing id that the baseline also collected is a regression
       outright, never a rename partner;
    4. the leftovers are new failures.

    Step 1 reads existence from ``now.selected_in_failed_files`` -- the
    collected ids of :attr:`Ledger.examined_files`, which
    :func:`collect` fills from the union of both runs' failing files.  If the
    id's file was never examined by this run, the id is not in that set and
    it falls to step 2, i.e. to ``unknown``: unanswerable is reported as
    unanswerable, never as a fix.
    """
    base_bad = set(base.bad)
    now_bad = set(now.bad)
    still_bad = base_bad & now_bad
    gone = sorted(base_bad - still_bad)
    appeared = sorted(now_bad - still_bad)

    now_selected = set(now.selected_in_failed_files)
    base_selected = set(base.selected_in_failed_files)

    # 1. a fix requires the id to still be collected *now*.
    fixed = tuple(node for node in gone if node in now_selected)
    vanished = [node for node in gone if node not in now_selected]

    # 3. a regression: the baseline collected it, so it existed and passed.
    regressed = {node for node in appeared if node in base_selected}
    fresh = [node for node in appeared if node not in base_selected]

    pairs = pair_renames(vanished, fresh)
    renamed_to = set(pairs.values())

    unknown = tuple(node for node in vanished if node not in pairs)
    brand_new = set(fresh) - renamed_to
    flagged = regressed | brand_new

    changed = bool(base.environment_key and now.environment_key
                   and base.environment_key != now.environment_key)
    return Delta(
        new_failures=tuple(sorted(flagged & set(now.failed))),
        new_errors=tuple(sorted(flagged & set(now.errors))),
        fixed=fixed,
        unknown=unknown,
        renames=tuple(sorted(pairs.items())),
        environment_changed=changed,
        environment_diff=environment_diff(base.environment, now.environment)
        if changed else (),
    )


# ---------------------------------------------------------------------------
# The record on disk
# ---------------------------------------------------------------------------

def to_dict(led: Ledger) -> dict:
    # A hand-built Ledger (a test, a reader's experiment) may not carry the
    # field; the record states the same thing either way, so derive it rather
    # than write an empty list that contradicts the ids below it.
    examined = led.examined_files or tuple(sorted(
        {node.split("::", 1)[0] for node in led.selected_in_failed_files}))
    return {
        "schema": SCHEMA,
        "recorded": led.recorded,
        "command": led.command,
        "environment_key": led.environment_key,
        "environment": dict(led.environment),
        "provenance": _provenance(),
        "ledger": {
            "passed": led.passed,
            "failed": list(led.failed),
            "skipped": led.skipped,
            "errors": list(led.errors),
            "xfailed": led.xfailed,
            "xpassed": led.xpassed,
            "deselected": led.deselected,
            "selected": led.selected,
            "selected_in_failed_files": list(led.selected_in_failed_files),
            "examined_files": list(examined),
        },
    }


def _provenance() -> dict:
    """``git`` state, for the reader's benefit only.

    Deliberately **not** part of the environment: every ``--check`` runs at a
    different commit, and a record that called that drift would refuse to
    compare anything.
    """
    out = {}
    git_cmds = (("git_commit", ["git", "rev-parse", "--short", "HEAD"]),
                ("git_dirty", ["git", "status", "--porcelain"]))
    for label, argv in git_cmds:
        try:
            proc = subprocess.run(argv, cwd=str(REPO), capture_output=True,
                                  text=True, errors="replace", timeout=60,
                                  check=False)
        except (OSError, subprocess.SubprocessError):
            out[label] = "unknown"
            continue
        if proc.returncode != 0:
            out[label] = "unknown"
        else:
            out[label] = proc.stdout.strip() or (
                "clean" if label == "git_dirty" else "")
    return out


def load(path) -> Ledger:
    """Read a record, refusing anything this tool does not understand.

    Raises :class:`ValueError` rather than guessing: a record whose
    ``schema`` is unknown may be missing the field the comparison rests on,
    and a quiet misread is how a wrong baseline becomes a green gate.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: not a readable record ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: not a record (a {type(payload).__name__})")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: schema {schema!r} is not {SCHEMA!r}; "
                         "this tool will not guess at a record it does not "
                         "understand -- re-record it")
    body = payload.get("ledger")
    if not isinstance(body, dict):
        raise ValueError(f"{path}: no 'ledger' object")
    missing = {"passed", "failed", "errors", "skipped", "xfailed"} - set(body)
    if missing:
        raise ValueError(f"{path}: ledger is missing {sorted(missing)}")
    selected_in_failed_files = tuple(body.get("selected_in_failed_files", ()))
    led = Ledger(
        passed=int(body.get("passed", 0)),
        failed=tuple(body.get("failed", ())),
        skipped=int(body.get("skipped", 0)),
        errors=tuple(body.get("errors", ())),
        xfailed=int(body.get("xfailed", 0)),
        xpassed=int(body.get("xpassed", 0)),
        deselected=int(body.get("deselected", 0)),
        selected=int(body.get("selected", 0)),
        selected_in_failed_files=selected_in_failed_files,
        # Absent in a record written before the field existed; those records
        # mean exactly "the ids of the files that carried a failure", which is
        # what Ledger.examined derives.  The schema is deliberately NOT
        # bumped: the committed baseline predates the field, and a bump would
        # make the Phase 1 exit gate exit 3 (unreadable) on a record that is
        # perfectly readable.
        examined_files=tuple(body.get("examined_files", ())),
        environment=payload.get("environment") or {},
        environment_key=payload.get("environment_key") or "",
        command=payload.get("command") or "",
        recorded=payload.get("recorded") or "",
    )
    return led


# ---------------------------------------------------------------------------
# main(): the gate
# ---------------------------------------------------------------------------

def _verdict(delta: Delta, allow_renames: bool) -> tuple:
    """``(exit_code, lines)`` for a comparison that is known to be
    environment-compatible (or whose drift the caller accepted).

    Always fatal: a new failure, a new error, and a baseline failure that
    stopped appearing with no explanation -- a deletion, which the plan calls
    "a deliberate act that needs its own ledger entry", so the gate refuses it
    until somebody re-records on purpose.  A *resolved rename* is fatal too,
    **by default** (Finding 6 of the P1.8 review): the plan requires a rename
    not to read as a regression, which this satisfies by resolving the pair
    into ``renames`` with ``unknown == ()`` -- not by waving it through.  The
    one thing that cannot be told apart from a rename by measurement is
    "delete the red test, add a different red test in the same file", so the
    default reading is the one that makes somebody look.  ``--allow-renames``
    takes the permissive reading explicitly; it is deliberately not the
    default, and deliberately *not* the name of the old ``--strict-renames``
    flag -- a flag whose absence has teeth is the safer shape.  Under an
    accepted environment drift only a new failure or error is fatal and the
    rest is not even printed as a finding: an absent ``$OR_SRC`` turns the
    oracle tests into *skips*, so a fix, a deletion and a rename are all
    artefacts of the environment rather than evidence about the code.
    """
    lines: list = []
    fatal = list(delta.regressions) + list(delta.unknown)
    if not allow_renames:
        fatal += [old for old, _new in delta.renames]
    if delta.advisory:
        fatal = list(delta.regressions)
    if delta.new_failures:
        lines.append(f"new failures ({len(delta.new_failures)})")
        lines.extend("  " + node for node in delta.new_failures)
    if delta.new_errors:
        lines.append(f"new errors ({len(delta.new_errors)})")
        lines.extend("  " + node for node in delta.new_errors)
    if delta.advisory:
        lines.append(
            f"suppressed under the accepted environment drift: "
            f"{len(delta.unknown)} unresolved, {len(delta.fixed)} fixed, "
            f"{len(delta.renames)} renamed -- none of those is evidence about "
            "the code in another environment")
    else:
        if delta.renames:
            lines.append(f"renamed ({len(delta.renames)}) -- same file, new "
                         "name; a deliberate act, so justify it in the commit")
            lines.extend(f"  {old}\n-> {new}" for old, new in delta.renames)
        if delta.unknown:
            lines.append(f"unresolved ({len(delta.unknown)}) -- a recorded "
                         "failure that no longer runs and has no rename "
                         "partner; a deletion is deliberate, so re-record")
            lines.extend("  " + node for node in delta.unknown)
        if delta.fixed:
            lines.append(f"fixed ({len(delta.fixed)})")
            lines.extend("  " + node for node in delta.fixed)
    if fatal:
        code = EXIT_REGRESSION
        head = (f"RESULT: REGRESSION -- {len(delta.regressions)} new "
                "failure(s)/error(s)")
        rest = len(fatal) - len(delta.regressions)
        if rest:
            head += f" and {rest} other record(s) the gate will not accept"
    else:
        code = EXIT_OK
        head = "RESULT: OK -- no new failures, no unresolved records"
    return code, [head] + lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regression_ledger.py",
        description="Record and compare the fast tier's node ids.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", metavar="PATH",
                       help="run the suite and write a record to PATH")
    group.add_argument("--check", metavar="PATH",
                       help="compare the current run against the record at "
                            "PATH; writes nothing, ever")
    parser.add_argument("--results", metavar="PATH",
                        help="compare a record already written by --record "
                             "instead of running the suite again (what CI "
                             "does: one suite run, one comparison)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="seconds before the child run is killed "
                             f"(default {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--backend", default="numpy",
                        help="PYRADIOSS_BACKEND for the child run; "
                             "'inherit' leaves the parent's value alone "
                             "(default numpy, the gate's own pin)")
    parser.add_argument("--allow-environment-drift", action="store_true",
                        help="compare across environments and treat only new "
                             "failures as fatal (what CI without the oracle "
                             "needs)")
    parser.add_argument("--allow-renames", action="store_true",
                        help="do not fail on a resolved rename pair; the "
                             "DEFAULT is fatal, because 'delete the red "
                             "test, add a red test in the same file' is "
                             "indistinguishable from a rename and must be "
                             "looked at (Finding 6 of the P1.8 review). "
                             "Suppressed anyway under --allow-environment-drift")
    parser.add_argument("--strict-renames", action="store_true",
                        help="accepted and ignored: a rename is fatal unless "
                             "--allow-renames is passed.  Kept so an old "
                             "command line from the pre-fix tool still runs")
    parser.add_argument("--against", metavar="PATH",
                        help="the record this --record will be compared "
                             "against, so the new record carries the "
                             "collected ids of the files that one is red in "
                             "(default: the committed "
                             "tools/validation_data/baseline.json; needed "
                             "only when that is not the file under test)")
    parser.add_argument("--force", action="store_true",
                        help="let --record overwrite an existing record")
    parser.add_argument("paths", nargs="*",
                        help="paths/modules for the child run, after a '--' "
                             "separator if they start with a dash (default: "
                             "the whole fast tier)")
    return parser


def _split_argv(argv) -> tuple:
    """``argv`` -> ``(own_args, pytest_args)``.

    argparse cannot tell ``-k expr`` meant for the child from a mistyped
    option of this tool, and the child's flags are the ones an operator
    reaches for (``-n auto``, ``--deselect``).  A literal ``--`` separator
    settles it: everything after it goes to the child, everything before it
    to this tool, so neither can eat the other's flags.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1:]
    return argv, []


def _preflight_environment(base: Ledger, pytest_args: Sequence[str],
                           backend: str | None) -> tuple:
    """``(key, diff)`` for the run that *would* happen, without running it.

    The child's environment is fully determined by :func:`child_command` and
    this process, so the environment comparison the gate refuses on is
    decidable in under a second -- measured 0.24 s against the committed
    baseline, whose key it reproduces exactly.  Finding 4 of the P1.8 review
    is that the refusal came *after* the 11-minute run: an expensive way to
    say "wrong box".
    """
    _argv, env = child_command(pytest_args, backend=backend)
    now = environment(
        {"env": {var: env.get(var, "<unset>") for var in _ENV_VARS}})
    key = environment_key(now)
    if not base.environment_key or key == base.environment_key:
        return key, ()
    return key, environment_diff(base.environment, now)


def _examined_for(base: Ledger) -> tuple:
    """The other run's failing files -- the union half of the existence rule."""
    return tuple(sorted({node.split("::", 1)[0] for node in base.bad}))


def _run_record(args, pytest_args, backend) -> int:
    """``--record``: refuse to overwrite, then run, then write.

    The order is the point (Finding 5 of the P1.8 review): "may I replace this
    file?" costs a second and must be asked *first*, because asking it after
    an 11-minute run makes a mistyped command the most expensive keystroke in
    the tool.
    """
    target = Path(args.record)
    if target.exists() and not args.force:
        print(f"ledger: {target} already exists; a committed record is "
              "never rewritten silently -- pass --force to replace it, "
              "and say so in the commit")
        return EXIT_ERROR

    # The record this one will be compared against decides which files' ids it
    # must carry: a fix can only be *seen* for a file whose collected ids the
    # new record knows.  Default: the committed baseline, so `--record` in CI
    # produces a record `--check --results` can actually use.
    counterpart = None
    if args.against:
        try:
            counterpart = load(args.against)
        except ValueError as exc:
            print(f"ledger: {exc}")
            return EXIT_ERROR
    elif BASELINE_PATH.is_file() and args.record != str(BASELINE_PATH):
        try:
            counterpart = load(BASELINE_PATH)
        except ValueError:
            counterpart = None

    # A supplied --results record IS the run to record.  Before this helper
    # existed, main() loaded args.results for --record as well as --check,
    # so recording from another run never re-ran anything; that branch is
    # restored here rather than re-invented, and the overwrite refusal above
    # still comes first so a mistyped --record costs a second either way.
    if args.results:
        try:
            now = load(args.results)
        except ValueError as exc:
            print(f"ledger: {exc}")
            return EXIT_ERROR
    else:
        try:
            now = collect(pytest_args, timeout=args.timeout, backend=backend,
                          examined_files=_examined_for(counterpart)
                          if counterpart is not None else ())
        except subprocess.TimeoutExpired:
            print(f"ledger: the suite did not finish within "
                  f"{args.timeout:g}s")
            return EXIT_ERROR
        except (RuntimeError, OSError) as exc:
            print(f"ledger: {exc}")
            return EXIT_ERROR

    print(f"now      : {now.passed} passed, {len(now.failed)} failed, "
          f"{len(now.errors)} errors, {now.skipped} skipped, "
          f"{now.xfailed} xfailed, {now.xpassed} xpassed, "
          f"{now.deselected} deselected")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_dict(now), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"recorded : {target}")
    print(f"examined : {len(now.examined_files)} file(s) whose collected ids "
          f"this record carries, so a later --check can tell a fix from a "
          f"deletion in them")
    print(f"RESULT: RECORDED -- {now.passed} passed, "
          f"{len(now.failed)} failed, {len(now.errors)} errors, "
          f"{now.skipped} skipped, {now.xfailed} xfailed, "
          f"{now.deselected} deselected")
    print(f"env      : {now.environment_key}")
    return EXIT_OK


def _vanished_passes(base: Ledger, now: Ledger) -> tuple:
    """Passing ids this record saw collected that are not collected now.

    Only over files both runs examined, because only there does the record
    have the before-picture.  A deleted or renamed **passing** test is
    invisible to a failures-ledger -- its id is in neither ``bad`` set -- and
    Finding 6 of the P1.8 review is right that no rename flag can reach it.
    So it is *reported*, not judged: a failures-ledger has no standing to say
    a passing test should not have been removed (the plan never forbade it),
    but a reader of the gate should not have to notice the hole themselves.
    """
    if not base.examined & now.examined:
        return ()
    gone = set(base.selected_in_failed_files) - set(
        now.selected_in_failed_files)
    return tuple(sorted(gone - set(base.bad) - set(now.bad)))


def _run_check(args, pytest_args, backend) -> int:
    """``--check``: read the record, pre-flight the box, run, compare, print."""
    try:
        baseline = load(args.check)
    except ValueError as exc:
        print(f"ledger: {exc}")
        return EXIT_ERROR
    print(f"baseline : {args.check}")
    print(f"recorded : {baseline.recorded or '<undated>'} "
          f"@ {baseline.command or '<no command recorded>'}")
    print(f"            {baseline.passed} passed, "
          f"{len(baseline.failed)} failed, {len(baseline.errors)} "
          f"errors, {baseline.skipped} skipped, "
          f"{baseline.xfailed} xfailed, "
          f"{baseline.deselected} deselected")

    if args.results:
        try:
            now = load(args.results)
        except ValueError as exc:
            print(f"ledger: {exc}")
            return EXIT_ERROR
        missing = _unexamined(baseline, now)
        if missing:
            print(f"warning  : {args.results} carries no collected ids for "
                  f"{len(missing)} file(s) the baseline is red in, so an id "
                  f"in them cannot be told apart from a deletion and is "
                  f"reported as unresolved: {', '.join(missing)}.  Record "
                  f"that run against this baseline (--record ... --against "
                  f"{args.check}) or drop --results to compare a live run")
    else:
        if not args.allow_environment_drift:
            key, diff = _preflight_environment(baseline, pytest_args, backend)
            if diff:
                print(f"env      : baseline {baseline.environment_key}")
                print(f"            now      {key}")
                for line in diff:
                    print(f"            differs: {line}")
                print("RESULT: ENVIRONMENT -- refused before running the "
                      "suite: the child's environment is fully determined "
                      "before it starts, so this costs a second and not an "
                      "11-minute fast tier.  Compare in the environment the "
                      "record was made in, re-record, or pass "
                      "--allow-environment-drift to compare anyway.")
                return EXIT_ENVIRONMENT
        try:
            now = collect(pytest_args, timeout=args.timeout, backend=backend,
                          examined_files=_examined_for(baseline))
        except subprocess.TimeoutExpired:
            print(f"ledger: the suite did not finish within {args.timeout:g}s")
            return EXIT_ERROR
        except (RuntimeError, OSError) as exc:
            print(f"ledger: {exc}")
            return EXIT_ERROR

    print(f"now      : {now.passed} passed, {len(now.failed)} failed, "
          f"{len(now.errors)} errors, {now.skipped} skipped, "
          f"{now.xfailed} xfailed, {now.xpassed} xpassed, "
          f"{now.deselected} deselected")

    delta = compare(baseline, now)
    if delta.environment_changed:
        print(f"env      : baseline {baseline.environment_key}")
        print(f"            now      {now.environment_key}")
        for line in delta.environment_diff:
            print(f"            differs: {line}")
        if not args.allow_environment_drift:
            print("RESULT: ENVIRONMENT -- this run is not comparable with "
                  "the record (see docs/STATE.md: the fast tier is not one "
                  "environment).  No verdict is printed on purpose: compare "
                  "in the environment the record was made in, re-record, or "
                  "pass --allow-environment-drift to compare anyway.")
            return EXIT_ENVIRONMENT
        print("RESULT: ADVISORY -- --allow-environment-drift: only NEW "
              "failures are fatal below; a fix, a deletion and a rename are "
              "not evidence in another environment (an absent $OR_SRC makes "
              "the oracle tests SKIP).")
        delta = replace(delta, advisory=True)

    vanished = _vanished_passes(baseline, now)
    if vanished:
        print(f"passing ids ({len(vanished)}) -- collected when the record "
              f"was made, in a file this run also examined, and no longer "
              f"collected: a renamed or deleted PASSING test.  A ledger of "
              f"failures cannot judge that, so it is reported and not fatal; "
              f"if one of these was meant to stay, that is a coverage "
              f"question, not this gate's")
        print("\n".join("  " + node for node in vanished))

    code, lines = _verdict(delta, args.allow_renames)
    for line in lines:
        print(line)
    return code


def main(argv=None) -> int:
    """``--record PATH`` / ``--check PATH``; exit 1 on any new failure.

    Exit codes: 0 clean, 1 a regression (or an unresolved record, or a rename
    unless ``--allow-renames``), 2 the two runs are not comparable, 3 the tool
    could not produce a comparison.  ``--check`` opens the baseline
    **read-only** and prints; the only branch that writes anything is
    ``--record``, and it needs ``--force`` before it will replace an existing
    record -- which it asks for *before* running anything.

    Both modes refuse the cheap, unanswerable questions first (is this record
    readable, may I overwrite it, is this the right box) and only then spend
    the fast tier.
    """
    own, extra = _split_argv(argv)
    args = _build_parser().parse_args(own)
    pytest_args = list(args.paths) + list(extra)
    backend = None if args.backend == "inherit" else args.backend
    if args.record:
        return _run_record(args, pytest_args, backend)
    return _run_check(args, pytest_args, backend)


if __name__ == "__main__":
    sys.exit(main())