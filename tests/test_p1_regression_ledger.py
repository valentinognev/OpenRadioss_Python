"""Task P1.8 — the regression ledger: "no new failures" must be measurable.

The plan (``plan/02_phase1_foundation.md:421-464``) asks for three
interfaces -- :func:`tools.regression_ledger.collect`, :func:`compare` and
``main`` -- and names one hard requirement for the diff: *a renamed test must
land in ``unknown``, not in ``new_failures````.  This module pins that
requirement and the cases around it.

The first two tests written were the plan's own, verbatim
(``plan/02_phase1_foundation.md:436-452``), and they are kept here marked as
such so a reader can check them against the plan without trusting this file.
What follows is the discrimination the plan leaves to the implementer: a
rename must not read as a **fix** either, and a genuinely **deleted** test
must not read as one.  The rule and its reasoning are in
``tools/regression_ledger.py``'s module docstring; the tests here are named
after the cases they pin, including the one case the heuristic cannot see.

Nothing in this file runs the fast tier -- that is a 13-minute run and it is
what ``collect()`` and ``--check`` are for.  The two ``collect()`` tests
below run two small modules in about two seconds, so the ledger's own tests
cannot be the reason a gate is slow.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pytest

from tools.regression_ledger import (
    BASELINE_PATH,
    Ledger,
    compare,
    environment,
    environment_key,
    load,
    main,
)

REPO = Path(__file__).resolve().parents[1]


def _ledger(failed=(), errors=(), selected=(), examined=(), **kw):
    """A recorded ledger.  ``selected`` is the collected set of the examined
    files -- the only part of the collection the record keeps, and the whole
    of the existence evidence ``compare`` needs (see the module docstring of
    ``tools/regression_ledger.py``).  ``examined`` defaults to the files those
    ids come from, which is what a real ``collect()`` records; pass it
    explicitly only to build a record that claims less than it carries."""
    kw.setdefault("passed", 3)
    return Ledger(failed=tuple(failed), errors=tuple(errors),
                  selected_in_failed_files=tuple(selected),
                  examined_files=tuple(examined) if examined else tuple(
                      sorted({node.split("::", 1)[0] for node in selected})),
                  **kw)


# ---------------------------------------------------------------------------
# The plan's own two tests (plan/02_phase1_foundation.md:436-452), verbatim
# ---------------------------------------------------------------------------

def test_ledger_flags_a_new_failure():
    base = Ledger(passed=10, failed=("tests/test_a.py::t1",), skipped=0,
                  errors=(), xfailed=0)
    now = Ledger(passed=10, failed=("tests/test_a.py::t1",
                                    "tests/test_b.py::t2"), skipped=0,
                 errors=(), xfailed=0)
    d = compare(base, now)
    assert d.new_failures == ("tests/test_b.py::t2",)
    assert d.fixed == ()


def test_ledger_distinguishes_a_rename_from_a_new_failure():
    base = Ledger(passed=1, failed=("tests/test_a.py::old_name",), skipped=0, errors=(), xfailed=0)
    now  = Ledger(passed=1, failed=("tests/test_a.py::new_name",), skipped=0, errors=(), xfailed=0)
    assert compare(base, now).unknown == ()


# ---------------------------------------------------------------------------
# The rename rule, in both directions
# ---------------------------------------------------------------------------

def test_a_rename_of_a_still_failing_test_is_neither_new_nor_fixed():
    base = _ledger(failed=["tests/test_a.py::old_name"],
                   selected=["tests/test_a.py::old_name"])
    now = _ledger(failed=["tests/test_a.py::new_name"],
                  selected=["tests/test_a.py::new_name"])
    d = compare(base, now)
    assert d.new_failures == ()
    assert d.new_errors == ()
    assert d.fixed == ()
    assert d.unknown == ()
    assert d.renames == (("tests/test_a.py::old_name",
                          "tests/test_a.py::new_name"),)


def test_a_rename_cannot_masquerade_as_a_fix():
    """The case a reviewer will build: the failing test's name disappears and
    a different failing test appears in the same file, so
    ``base.failed - now.failed`` is non-empty and the naive reading is
    "fixed".  A fix requires the node id to still be **collected** -- only
    then is it the same test going green."""
    base = _ledger(failed=["tests/test_a.py::test_red"],
                   selected=["tests/test_a.py::test_red"])
    now = _ledger(failed=["tests/test_a.py::test_green"],
                  selected=["tests/test_a.py::test_green"])
    assert compare(base, now).fixed == ()


_RED_MODULE = '''\
"""A module with exactly one red test, written to a temporary directory by
the test below -- never into the checkout, so no run of the suite ever sees
it and no run of the suite is slowed by it."""


def test_red():
    assert False, "the fix this test is about: same id, green next run"


def test_bystander():
    assert True
'''

_GREEN_MODULE = _RED_MODULE.replace(
    'assert False, "the fix this test is about: same id, green next run"',
    'assert True, "the fix this test is about: same id, green next run"')

#: the file pytest's node ids are relative to when it is handed this path
_PROBE_MODULE = "test_p1_ledger_fix.py"


def _probe_module(tmp_path, source=_RED_MODULE):
    module = tmp_path / _PROBE_MODULE
    module.write_text(source, encoding="utf-8")
    return module


def test_a_genuine_fix_is_still_a_fix(tmp_path, capsys):
    """The reviewer's scenario, run for real: **Finding 2 of the P1.8 review.**

    The first draft of this test hand-fed ``now`` a
    ``selected_in_failed_files`` holding the green test's own id -- a value
    ``collect()`` cannot emit, because a file with no failure is not among
    the failing files at all.  It passed while the production path was wrong:
    on a real run, fixing a file's last red test produced ``unknown`` and the
    Phase 1 exit gate exited 1 on a tree with *zero* failures.  A test whose
    fixture is unreachable from the production path certifies nothing, so this
    version runs the suite twice through ``main`` itself: once to record a
    red baseline, once to check a green tree against it.
    """
    module = _probe_module(tmp_path)
    baseline = tmp_path / "baseline.json"

    assert main(["--record", str(baseline), "--", str(module)]) == 0
    recorded = load(baseline)
    assert recorded.failed, "the probe module must be red, or nothing is proved"
    node = recorded.failed[0]
    assert node == f"{_PROBE_MODULE}::test_red", node

    # the fix: the same node id, green, in a file that now carries no failure
    module.write_text(_GREEN_MODULE, encoding="utf-8")

    rc = main(["--check", str(baseline), "--", str(module)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"fixed (1)\n  {node}" in out, out
    assert "\nunresolved (" not in out, out
    assert "REGRESSION" not in out, out


def test_collect_examines_the_files_the_other_run_is_red_in(tmp_path):
    """The mechanism behind the fix above, at the level it lives.

    A green run contributes no failing file of its own, so without the
    ``examined_files`` hand-off its record carries no ids at all and "still
    collected" is unanswerable.  The set is the union of both runs' bad files,
    and what is recorded is the files that actually contributed ids -- an
    examined file with nothing collected from it would make a silence look
    like evidence.
    """
    from tools import regression_ledger as rl
    module = _probe_module(tmp_path, _GREEN_MODULE)

    blind = rl.collect([str(module)], timeout=600)
    assert blind.failed == ()
    assert blind.selected_in_failed_files == ()
    assert blind.examined_files == ()

    widened = rl.collect([str(module)], timeout=600,
                         examined_files=[_PROBE_MODULE])
    assert f"{_PROBE_MODULE}::test_red" in widened.selected_in_failed_files
    assert f"{_PROBE_MODULE}::test_bystander" in widened.selected_in_failed_files
    assert widened.examined_files == (_PROBE_MODULE,)


def test_check_hands_the_baselines_failing_files_to_the_run(tmp_path, capsys,
                                                            monkeypatch):
    """``main`` -- not the test -- derives the union: the files the baseline
    is red in are what the current run must report ids for."""
    from tools import regression_ledger as rl
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_a.py::red"],
                           selected=["tests/test_a.py::red",
                                     "tests/test_a.py::other"],
                           environment_key="k"))
    live = rl.environment()

    def _fake(*_a, **kw):
        # what a real collect returns once the fix is in place
        return rl.Ledger(passed=2, selected_in_failed_files=(
            "tests/test_a.py::other", "tests/test_a.py::red"),
            examined_files=("tests/test_a.py",),
            environment=live, environment_key="k")

    seen = []

    def _spy(*a, **kw):
        seen.append(kw["examined_files"])
        return _fake(*a, **kw)

    monkeypatch.setattr(rl, "collect", _spy)

    # --allow-environment-drift: this test is about the examined_files
    # hand-off, not the environment gate.  Its baseline carries the
    # placeholder key "k", so _preflight_environment correctly refuses
    # before collect ever runs; the flag is what lets this test reach the
    # assertion it exists for.  The refusal itself is asserted by
    # test_the_environment_mismatch_is_refused_before_the_suite_runs.
    assert main(["--check", str(base), "--allow-environment-drift"]) == 0
    assert seen == [("tests/test_a.py",)], seen
    out = capsys.readouterr().out
    assert "fixed (1)" in out and "tests/test_a.py::red" in out, out


def test_a_deleted_failure_is_unresolved_never_a_fix():
    """The plan calls a deletion 'a deliberate act that needs its own ledger
    entry'.  It is not a fix: nothing went green, the test is gone.  The
    ``unknown`` bucket is what refuses to call it either."""
    base = _ledger(failed=["tests/test_a.py::test_red"],
                   selected=["tests/test_a.py::test_red"])
    now = _ledger(failed=[], selected=[])        # the whole file is gone
    d = compare(base, now)
    assert d.unknown == ("tests/test_a.py::test_red",)
    assert d.fixed == ()
    assert d.new_failures == ()


def test_a_test_that_existed_and_started_failing_is_a_new_failure():
    """One file, one test deleted and another one regressed.  The regression
    must not be absorbed by a rename pairing -- this is what the recorded
    selected set buys, and without it the two cases are indistinguishable."""
    base = _ledger(failed=["tests/test_a.py::test_red"],
                   selected=["tests/test_a.py::test_mid",
                             "tests/test_a.py::test_red"])
    now = _ledger(failed=["tests/test_a.py::test_mid"],
                  selected=["tests/test_a.py::test_mid"])
    d = compare(base, now)
    assert d.new_failures == ("tests/test_a.py::test_mid",)
    assert d.renames == ()
    assert d.unknown == ("tests/test_a.py::test_red",)


def test_a_parametrised_id_change_is_a_deletion_not_a_rename():
    """``t[1]`` failing while ``t[2]`` starts failing is one test with two
    parameter sets, not one test renamed: stripping the ``[...]`` gives the
    same skeleton.  Left to the file-and-uniqueness rule alone this would be
    absorbed as a rename, and a new parameter set is a new test case."""
    base = _ledger(failed=["tests/test_a.py::t[1]"],
                   selected=["tests/test_a.py::t[1]"])
    now = _ledger(failed=["tests/test_a.py::t[2]"],
                  selected=["tests/test_a.py::t[2]"])
    d = compare(base, now)
    assert d.new_failures == ("tests/test_a.py::t[2]",)
    assert d.renames == ()
    assert d.unknown == ("tests/test_a.py::t[1]",)


def test_a_renamed_parametrised_test_is_still_a_rename():
    """The complementary case: ``t[1]`` -> ``u[1]`` keeps the parameter and
    changes the name, so the skeletons differ and it pairs."""
    base = _ledger(failed=["tests/test_a.py::t[1]"],
                   selected=["tests/test_a.py::t[1]"])
    now = _ledger(failed=["tests/test_a.py::u[1]"],
                  selected=["tests/test_a.py::u[1]"])
    d = compare(base, now)
    assert d.renames == (("tests/test_a.py::t[1]", "tests/test_a.py::u[1]"),)
    assert d.new_failures == ()


def test_an_error_whose_id_is_still_red_is_not_a_new_failure():
    """An error in the baseline that fails at call phase now is the same
    node id, still red.  It is neither new nor fixed, and reporting it as
    new would make every error->failure transition a gate failure."""
    base = _ledger(failed=[], errors=["tests/test_a.py::boom"],
                   selected=["tests/test_a.py::boom"])
    now = _ledger(failed=["tests/test_a.py::boom"], errors=[],
                  selected=["tests/test_a.py::boom"])
    d = compare(base, now)
    assert d.new_failures == () and d.new_errors == ()
    assert d.fixed == () and d.unknown == ()


def test_a_new_error_is_reported_as_an_error_not_a_failure():
    base = _ledger()
    now = _ledger(errors=["tests/test_a.py::boom"],
                  selected=["tests/test_a.py::boom"])
    d = compare(base, now)
    assert d.new_errors == ("tests/test_a.py::boom",)
    assert d.new_failures == ()


def test_a_new_error_outranks_a_rename_pairing():
    base = _ledger(errors=["tests/test_a.py::old"],
                   selected=["tests/test_a.py::old"])
    now = _ledger(errors=["tests/test_a.py::new"],
                  selected=["tests/test_a.py::new"])
    d = compare(base, now)
    assert d.renames == (("tests/test_a.py::old", "tests/test_a.py::new"),)
    assert d.new_errors == ()
    assert d.new_failures == ()


# ---------------------------------------------------------------------------
# Ambiguity: fall towards reporting more, never less
# ---------------------------------------------------------------------------

def test_two_swaps_in_one_file_are_not_paired():
    """old1->new1 *and* old2->new2 in one file is indistinguishable from a
    rename pair.  The ledger refuses to guess and takes the direction that
    can only report more."""
    base = _ledger(failed=["tests/test_a.py::old1", "tests/test_a.py::old2"],
                   selected=["tests/test_a.py::old1", "tests/test_a.py::old2"])
    now = _ledger(failed=["tests/test_a.py::new1", "tests/test_a.py::new2"],
                  selected=["tests/test_a.py::new1", "tests/test_a.py::new2"])
    d = compare(base, now)
    assert d.renames == ()
    assert d.new_failures == ("tests/test_a.py::new1", "tests/test_a.py::new2")
    assert d.unknown == ("tests/test_a.py::old1", "tests/test_a.py::old2")


def test_a_rename_across_files_is_not_a_rename():
    """Moving a failing test into another module is not name-preserving, so
    it reads as a deletion plus a new failure.  That is the intended
    direction: the gate bites on a moved red test."""
    base = _ledger(failed=["tests/test_a.py::old"],
                   selected=["tests/test_a.py::old"])
    now = _ledger(failed=["tests/test_b.py::old"],
                  selected=["tests/test_b.py::old"])
    d = compare(base, now)
    assert d.renames == ()
    assert d.new_failures == ("tests/test_b.py::old",)
    assert d.unknown == ("tests/test_a.py::old",)


def test_a_rename_among_other_failures_in_the_same_file_is_still_paired():
    base = _ledger(failed=["tests/test_a.py::old", "tests/test_a.py::keep"],
                   selected=["tests/test_a.py::old", "tests/test_a.py::keep"])
    now = _ledger(failed=["tests/test_a.py::keep", "tests/test_a.py::new"],
                  selected=["tests/test_a.py::keep", "tests/test_a.py::new"])
    d = compare(base, now)
    assert d.renames == (("tests/test_a.py::old", "tests/test_a.py::new"),)
    assert d.new_failures == ()
    assert d.unknown == ()


def test_a_failure_that_stays_failing_is_not_reported_at_all():
    base = _ledger(failed=["tests/test_a.py::same"],
                   selected=["tests/test_a.py::same"])
    now = _ledger(failed=["tests/test_a.py::same"],
                  selected=["tests/test_a.py::same"])
    d = compare(base, now)
    assert d.new_failures == () and d.fixed == () and d.unknown == ()
    assert d.renames == ()


def test_the_documented_cost_of_a_delete_plus_add_in_one_file():
    """The one case the heuristic cannot see: "delete the red test, add a new
    red test in the same file" has exactly the signature of a rename, because
    nothing was measured that could tell them apart.  It is reported as a
    rename and *printed by name* in every report; ``--strict-renames`` makes
    it fatal.  Pinned here so the cost is a fact in the suite rather than a
    surprise (see ``test_strict_renames_makes_a_rename_fatal``)."""
    base = _ledger(failed=["tests/test_a.py::test_old"],
                   selected=["tests/test_a.py::test_old"])
    now = _ledger(failed=["tests/test_a.py::test_new"],
                  selected=["tests/test_a.py::test_new"])
    d = compare(base, now)
    assert d.new_failures == ()
    assert d.renames == (("tests/test_a.py::test_old",
                          "tests/test_a.py::test_new"),)


def test_the_restricted_selected_set_is_enough_because_pairs_share_a_file():
    """The record keeps only the collected ids of the files that carry a
    recorded failure.  That is sufficient *because every id whose existence
    the diff asks about belongs to such a file*: a fixed or deleted id was
    failing in the baseline, and a rename partner is in the same file."""
    base = _ledger(failed=["tests/test_a.py::red"],
                   selected=["tests/test_a.py::red"])
    # test_b.py carries no baseline failure, so the record says nothing about
    # its collection -- and nothing is asked about it either.
    now = _ledger(failed=["tests/test_b.py::fresh"],
                  selected=["tests/test_b.py::fresh"])
    d = compare(base, now)
    assert d.new_failures == ("tests/test_b.py::fresh",)
    assert d.unknown == ("tests/test_a.py::red",)
    assert d.renames == ()


# ---------------------------------------------------------------------------
# Environment: a baseline from another box must not become a phantom diff
# ---------------------------------------------------------------------------

def test_the_environment_key_names_the_axes_that_move_the_counts():
    env = environment()
    for key in ("python", "implementation", "platform", "packages",
                "env", "resources"):
        assert key in env, key
    assert "numpy" in env["packages"]
    # the oracle is the axis Phase 0 measured (docs/STATE.md: two rows of the
    # same suite differing by the oracle being resolvable)
    assert set(env["resources"]) >= {"OR_SRC", "OR_ROOT", "hm_cfg"}
    assert all(isinstance(v, bool) for v in env["resources"].values())
    key = environment_key(env)
    assert key and key == environment_key(environment())


def test_the_environment_key_reacts_to_the_backend_pin():
    a = {"python": "3.12.3", "implementation": "CPython", "platform": "Linux",
         "packages": {"numpy": "2.5.3"}, "env": {"PYRADIOSS_BACKEND": "numpy"},
         "resources": {"OR_SRC": True}}
    b = dict(a, env={"PYRADIOSS_BACKEND": "numba"})
    assert environment_key(a) != environment_key(b)


def test_an_environment_mismatch_is_reported_and_never_a_verdict():
    base = _ledger(failed=["tests/test_a.py::t1"],
                   selected=["tests/test_a.py::t1"],
                   environment={"python": "3.12.3",
                                "resources": {"OR_SRC": True}},
                   environment_key="linux/py3.12.3/oracle=present")
    now = _ledger(failed=["tests/test_a.py::t1"],
                  selected=["tests/test_a.py::t1"],
                  environment={"python": "3.10.0",
                               "resources": {"OR_SRC": False}},
                  environment_key="linux/py3.10.0/oracle=absent")
    d = compare(base, now)
    assert d.environment_changed
    assert any(line.startswith("python:") for line in d.environment_diff)
    assert any("OR_SRC" in line for line in d.environment_diff)


def test_a_ledger_with_no_recorded_environment_cannot_drift():
    """An old record (or a hand-written one) carries no key, so the tool does
    not claim a drift it cannot substantiate."""
    d = compare(_ledger(failed=["tests/test_a.py::t1"]), _ledger())
    assert not d.environment_changed
    assert d.environment_diff == ()


def test_an_identical_environment_is_not_a_drift():
    env = {"python": "3.12.3"}
    d = compare(_ledger(environment=env, environment_key="k"),
                _ledger(environment=env, environment_key="k"))
    assert not d.environment_changed


# ---------------------------------------------------------------------------
# main(): the gate
# ---------------------------------------------------------------------------

def _write_record(path: Path, led: Ledger, **extra) -> Path:
    examined = led.examined_files or tuple(sorted(
        {node.split("::", 1)[0] for node in led.selected_in_failed_files}))
    payload = {
        "schema": "pyradioss-regression-ledger/1",
        "recorded": "2026-10-04T12:00:00+00:00",
        "command": "python -m pytest -q -m 'not slow'",
        "environment_key": led.environment_key,
        "environment": dict(led.environment),
        "ledger": {
            "passed": led.passed,
            "failed": list(led.failed),
            "errors": list(led.errors),
            "skipped": led.skipped,
            "xfailed": led.xfailed,
            "xpassed": led.xpassed,
            "deselected": led.deselected,
            "selected": led.selected,
            "selected_in_failed_files": list(led.selected_in_failed_files),
            "examined_files": list(examined),
        },
    }
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _record(tmp_path, name, led, **extra):
    return _write_record(tmp_path / name, led, **extra)


def _pre_field_record(path: Path, led: Ledger) -> Path:
    """The record shape written *before* ``examined_files`` existed: the ids
    are there, the statement of which files they cover is not."""
    _write_record(path, led)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["ledger"]["examined_files"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_check_exits_1_and_names_a_planted_new_failure(tmp_path, capsys):
    base = _record(tmp_path, "baseline.json", _ledger(environment_key="k"))
    now = _record(tmp_path, "now.json",
                  _ledger(failed=["tests/test_p1_x.py::test_planted"],
                          environment_key="k"))
    rc = main(["--check", str(base), "--results", str(now)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "tests/test_p1_x.py::test_planted" in out
    assert "REGRESSION" in out
    assert "1 new failure" in out


def test_check_exits_0_on_an_unchanged_tree(tmp_path, capsys):
    base = _record(tmp_path, "baseline.json", _ledger(environment_key="k"))
    now = _record(tmp_path, "now.json",
                  _ledger(passed=99, skipped=1, xfailed=2, deselected=3,
                          environment_key="k"))
    assert main(["--check", str(base), "--results", str(now)]) == 0
    out = capsys.readouterr().out
    assert "RESULT: OK" in out
    assert "REGRESSION" not in out


def test_check_exits_2_on_an_environment_mismatch(tmp_path, capsys):
    base = _record(tmp_path, "baseline.json",
                   _ledger(environment={"python": "3.12.3"},
                           environment_key="linux/py3.12.3/oracle=present"))
    now = _record(tmp_path, "now.json",
                  _ledger(failed=["tests/test_p1_x.py::test_x"],
                          environment={"python": "3.10.0"},
                          environment_key="linux/py3.10.0/oracle=absent"))
    rc = main(["--check", str(base), "--results", str(now)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "ENVIRONMENT" in out
    assert "oracle=present" in out and "oracle=absent" in out
    # and, crucially, no verdict: a mass of phantom new failures is worse
    # than no answer at all
    assert "new failure" not in out


def test_an_accepted_environment_drift_keeps_new_failures_fatal(tmp_path,
                                                               capsys):
    base = _record(tmp_path, "baseline.json",
                   _ledger(environment={"python": "3.12.3"},
                           environment_key="linux/py3.12.3/oracle=present"))
    now = _record(tmp_path, "now.json",
                  _ledger(failed=["tests/test_p1_x.py::t"], skipped=9000,
                          environment={"python": "3.10.0"},
                          environment_key="linux/py3.10.0/oracle=absent"))
    rc = main(["--check", str(base), "--results", str(now),
               "--allow-environment-drift"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ADVISORY" in out
    assert "tests/test_p1_x.py::t" in out


def test_an_accepted_drift_does_not_turn_the_oracle_skips_into_deletions(
        tmp_path, capsys):
    """The CI shape (Task P1.9): the baseline's oracle test was red, CI skips
    it because ``$OR_SRC`` is absent.  Under an accepted drift that must not
    be reported as a deletion -- or as a fix."""
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_p0_oracle_build.py::test_or"],
                           selected=["tests/test_p0_oracle_build.py::test_or"],
                           environment={"OR_SRC": True},
                           environment_key="linux/py3.12.3/oracle=present"))
    now = _record(tmp_path, "now.json",
                  _ledger(skipped=40, deselected=20,
                          environment={"OR_SRC": False},
                          environment_key="linux/py3.12.3/oracle=absent"))
    rc = main(["--check", str(base), "--results", str(now),
               "--allow-environment-drift"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "ADVISORY" in out
    # the oracle test is named in the record's own summary line, and no node
    # id appears in the verdict: no deletion, no fix, no rename, no new
    # failure
    verdict = out.split("ADVISORY", 1)[1]
    assert "test_oracle" not in verdict
    assert "suppressed" in verdict
    assert "\nnew failures" not in verdict
    assert "\nnew errors" not in verdict
    assert "RESULT: OK" in verdict


def test_a_rename_is_fatal_unless_it_is_allowed(tmp_path, capsys):
    """**Finding 6 of the P1.8 review**: the teeth must be the default.

    ``--strict-renames`` existed and was used *nowhere* -- not in
    ``plan/02_phase1_foundation.md``'s exit gate, not in
    ``.github/workflows/ci.yml`` -- so the default reading of the one case the
    ledger provably cannot decide was the permissive one.  A rename is fatal
    now, and ``--allow-renames`` is the explicit opt-out.  No CI edit is
    needed for this: CI passes ``--allow-environment-drift``, under which
    nothing but a new failure is fatal at all.
    """
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_a.py::old"],
                           selected=["tests/test_a.py::old"]))
    now = _record(tmp_path, "now.json",
                  _ledger(failed=["tests/test_a.py::new"],
                          selected=["tests/test_a.py::new"]))
    assert main(["--check", str(base), "--results", str(now)]) == 1
    out = capsys.readouterr().out
    assert "renamed (1)" in out, out
    assert "tests/test_a.py::old" in out and "tests/test_a.py::new" in out, out
    # the permissive reading is still one flag away, and is named as the
    # exception rather than being the state of the gate
    assert main(["--check", str(base), "--results", str(now),
                 "--allow-renames"]) == 0
    # the old flag name still runs -- an old command line must not break --
    # and does not quietly weaken the default
    assert main(["--check", str(base), "--results", str(now),
                 "--strict-renames"]) == 1


def test_a_renamed_passing_test_is_reported_though_it_cannot_be_judged(
        tmp_path, capsys):
    """Finding 6, second half: a rename of a PASSING test is invisible to a
    failures-ledger -- its id is in neither ``bad`` set, so no rename flag can
    reach it.  The record does carry the before-picture for every file it
    examined, so it is *reported*: named, not fatal.  A ledger of failures has
    no standing to forbid removing a passing test, but the reader of a gate
    should not have to find the hole by hand."""
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_a.py::red"],
                           selected=["tests/test_a.py::red",
                                     "tests/test_a.py::was_green"]))
    now = _record(tmp_path, "now.json",
                  _ledger(failed=[],
                          selected=["tests/test_a.py::red"]))
    rc = main(["--check", str(base), "--results", str(now)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "passing ids (1)" in out, out
    assert "tests/test_a.py::was_green" in out, out


def test_the_environment_mismatch_is_refused_before_the_suite_runs(
        tmp_path, capsys, monkeypatch):
    """**Finding 4 of the P1.8 review.**  The child's environment is fully
    determined by ``child_command()`` without executing anything, so the
    refusal is available for free -- it used to arrive *after* an 11-minute
    fast tier, which is the most expensive way to say "wrong box"."""
    from tools import regression_ledger as rl
    base = _record(tmp_path, "baseline.json",
                   _ledger(environment={"python": "3.10.0"},
                           environment_key="linux/py3.10.0/oracle=present"))
    monkeypatch.setattr(
        rl, "collect",
        lambda *a, **k: pytest.fail("the suite must not run on a "
                                    "mismatched box"))

    started = time.perf_counter()
    rc = main(["--check", str(base)])
    elapsed = time.perf_counter() - started

    assert rc == 2
    assert elapsed < 5.0, f"{elapsed:.1f}s -- the refusal has to be the cheap one"
    out = capsys.readouterr().out
    assert "ENVIRONMENT" in out and "before running the suite" in out, out
    assert "new failure" not in out, out


def test_record_refuses_to_overwrite_before_running_the_suite(tmp_path, capsys,
                                                             monkeypatch):
    """**Finding 5 of the P1.8 review.**  "May I replace this file?" costs a
    second; it used to be asked *after* an 11-minute run, which made a
    mistyped command the most expensive keystroke in the tool."""
    from tools import regression_ledger as rl
    target = _record(tmp_path, "baseline.json", _ledger())
    before = target.read_bytes()
    monkeypatch.setattr(
        rl, "collect",
        lambda *a, **k: pytest.fail("the suite must not run to refuse an "
                                    "overwrite"))

    assert main(["--record", str(target)]) == 3
    assert "--force" in capsys.readouterr().out
    assert target.read_bytes() == before


def test_a_record_carries_the_ids_of_the_record_it_will_be_checked_against(
        tmp_path, monkeypatch):
    """``--record --against BASE``: the new record must be able to answer
    "still collected?" for the ids BASE is red in, or the comparison it exists
    to serve degrades into the Critical it replaced.  The default is the
    committed ``tools/validation_data/baseline.json``, which is what makes
    CI's one-run-then-compare shape work with no extra flag."""
    from tools import regression_ledger as rl
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_z.py::red"],
                           selected=["tests/test_z.py::red", "tests/test_z.py::g"],
                           environment_key="k"))
    target = tmp_path / "fresh.json"

    seen = []

    def _spy(*_a, **kw):
        seen.append(kw["examined_files"])
        return rl.Ledger(
            passed=3, failed=("tests/test_y.py::t",),
            selected_in_failed_files=("tests/test_y.py::t",
                                      "tests/test_z.py::red",
                                      "tests/test_z.py::g"),
            examined_files=("tests/test_y.py", "tests/test_z.py"),
            environment_key="k")

    monkeypatch.setattr(rl, "collect", _spy)

    assert main(["--record", str(target), "--against", str(base)]) == 0
    assert seen == [("tests/test_z.py",)], seen
    back = load(target)
    assert "tests/test_z.py::g" in back.selected_in_failed_files
    assert "tests/test_z.py" in back.examined_files


def test_check_names_the_file_whose_ids_the_current_record_lacks(tmp_path,
                                                                capsys):
    """A record written against a *different* baseline cannot answer "did this
    id go green?" for a file it never looked at.  The verdict is then
    "unresolved", which is fatal and correct -- but on its own it reads as a
    claim about the code, so the file and the remedy are named."""
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_a.py::red"],
                           selected=["tests/test_a.py::red"]))
    # a record from before the field existed: ids, but no statement of which
    # files they were examined for
    now = _pre_field_record(tmp_path / "now.json", _ledger(failed=[]))
    rc = main(["--check", str(base), "--results", str(now)])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "carries no collected ids for 1 file(s)" in out, out
    assert "tests/test_a.py" in out, out
    assert "--against" in out, out
    assert "unresolved" in out and "tests/test_a.py::red" in out, out


def test_a_record_written_before_the_field_was_derived_still_loads(tmp_path):
    """No schema bump: the committed baseline predates ``examined_files`` and
    meant exactly "the ids of the files that carried a failure".  A bump would
    make the Phase 1 exit gate exit 3 (unreadable) on a readable record."""
    led = load(BASELINE_PATH)
    assert led.examined, "the committed record must still say what it examined"
    assert set(led.examined) == {node.split("::", 1)[0]
                                 for node in led.selected_in_failed_files}
    # and it still compares against itself without inventing a difference
    same = compare(led, led)
    assert (same.new_failures, same.fixed, same.unknown, same.renames) == (
        (), (), (), ())


def test_a_deleted_failure_fails_the_gate(tmp_path, capsys):
    base = _record(tmp_path, "baseline.json",
                   _ledger(failed=["tests/test_a.py::red"],
                           selected=["tests/test_a.py::red"]))
    now = _record(tmp_path, "now.json", _ledger(passed=1))
    rc = main(["--check", str(base), "--results", str(now)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "unresolved" in out
    assert "tests/test_a.py::red" in out
    assert "fixed" not in out


def test_check_writes_nothing(tmp_path):
    """Phase 0 lost three rounds to a gate that quietly rewrote a tracked
    file.  ``--check`` is read-only: the record's bytes and its mtime are the
    assertion, and so is the absence of any other file in the directory."""
    base = _record(tmp_path, "baseline.json", _ledger())
    now = _record(tmp_path, "now.json", _ledger())
    before, stamp = base.read_bytes(), base.stat().st_mtime_ns
    names = sorted(p.name for p in tmp_path.iterdir())
    assert main(["--check", str(base), "--results", str(now)]) == 0
    assert base.read_bytes() == before
    assert base.stat().st_mtime_ns == stamp
    assert sorted(p.name for p in tmp_path.iterdir()) == names


def test_check_refuses_a_baseline_it_does_not_understand(tmp_path, capsys):
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"schema": "something-else/9"}),
                   encoding="utf-8")
    # refused before the suite is run: a record this tool cannot read is a
    # question, not a 13-minute run to produce an untrustworthy answer.
    assert main(["--check", str(bad)]) == 3
    assert "schema" in capsys.readouterr().out


@pytest.mark.parametrize("payload", ["not a json document",
                                     json.dumps([1, 2, 3]),
                                     json.dumps(
                                         {"schema": "pyradioss-regression"
                                                  "-ledger/1"}),
                                     json.dumps(
                                         {"schema": "pyradioss-regression"
                                                    "-ledger/1",
                                          "ledger": {"passed": 1}})])
def test_load_refuses_anything_it_cannot_compare(tmp_path, payload):
    p = tmp_path / "b.json"
    p.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load(p)


def test_record_refuses_to_overwrite_a_record_without_force(tmp_path, capsys):
    target = _record(tmp_path, "baseline.json", _ledger())
    now = _record(tmp_path, "now.json", _ledger(passed=7))
    before = target.read_bytes()
    assert main(["--record", str(target), "--results", str(now)]) == 3
    assert "--force" in capsys.readouterr().out
    assert target.read_bytes() == before
    assert main(["--record", str(target), "--results", str(now),
                 "--force"]) == 0
    assert json.loads(target.read_text())["ledger"]["passed"] == 7


def test_a_newly_recorded_file_is_a_record_the_tool_can_read_again(tmp_path):
    target = tmp_path / "fresh.json"
    led = _ledger(failed=["tests/test_a.py::t"],
                  selected=["tests/test_a.py::t"], environment_key="k")
    assert main(["--record", str(target), "--results",
                 str(_record(tmp_path, "src.json", led))]) == 0
    back = load(target)
    assert back.failed == ("tests/test_a.py::t",)
    assert back.selected_in_failed_files == ("tests/test_a.py::t",)


# ---------------------------------------------------------------------------
# collect(): the collection itself
# ---------------------------------------------------------------------------

def test_the_child_run_is_pinned_to_numpy_and_writes_no_bytecode(monkeypatch):
    from tools import regression_ledger as rl
    argv, env = rl.child_command((), backend="numpy")
    assert argv[:3] == [rl.sys.executable, "-m", "pytest"]
    assert env["PYRADIOSS_BACKEND"] == "numpy"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "no:cacheprovider" in argv
    # the gate's own selection: index(3) skips the "-m pytest" of the
    # interpreter invocation itself
    assert argv[argv.index("-m", 3):argv.index("-m", 3) + 2] == [
        "-m", "not slow"]
    # an existing pin is respected, not overwritten: the ledger must describe
    # the run that actually happened
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numba")
    _, env2 = rl.child_command((), backend="numpy")
    assert env2["PYRADIOSS_BACKEND"] == "numba"
    _, env3 = rl.child_command((), backend=None)
    assert env3["PYRADIOSS_BACKEND"] == "numba"
    monkeypatch.delenv("PYRADIOSS_BACKEND")
    _, env4 = rl.child_command((), backend=None)
    assert "PYRADIOSS_BACKEND" not in env4


def test_collect_reproduces_pytests_own_count_of_a_small_module():
    """One cheap end-to-end: ``collect()`` on a one-second module must agree
    with the summary line pytest prints, or the ledger is measuring
    something other than the gate."""
    from tools import regression_ledger as rl
    led = rl.collect(["tests/test_p0_paths.py"], timeout=600)
    assert led.failed == ()
    assert led.errors == ()
    assert (led.passed, led.skipped, led.xfailed, led.deselected) == (
        71, 0, 0, 0)
    assert led.selected == 71
    assert led.environment_key
    assert led.selected_in_failed_files == ()


def test_collect_partitions_every_collected_test_and_sees_the_xfails():
    from tools import regression_ledger as rl
    led = rl.collect(["tests/test_p0_paths.py",
                      "tests/test_m566_law92_arruda_boyce.py"], timeout=600)
    total = (led.passed + len(led.failed) + len(led.errors) + led.skipped
             + led.xfailed + led.xpassed)
    assert total == led.selected
    assert led.failed == () and led.errors == ()
    # 71 paths + 5 solid-update laws; the five Arruda-Boyce entries are in
    # tests/conftest.py's _KNOWN_XFAIL, so they are xfailed, never failures.
    assert led.xfailed == 5, led
    assert led.passed == 76, led
    assert led.selected == 81, led


# ---------------------------------------------------------------------------
# The recorded baseline itself
# ---------------------------------------------------------------------------

def _baseline_payload():
    assert BASELINE_PATH.is_file(), (
        "record the baseline first:\n"
        "  .venv/bin/python tools/regression_ledger.py --record "
        "tools/validation_data/baseline.json")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_the_recorded_baseline_exists_and_says_what_the_tree_says():
    payload = _baseline_payload()
    assert payload["schema"] == "pyradioss-regression-ledger/1"
    led = payload["ledger"]
    assert led["passed"] > 1000, led["passed"]
    assert led["xfailed"] > 0, led["xfailed"]
    # The plan's "13 known failures" (plan/02_phase1_foundation.md:461, and
    # plan/18_phase17_convergence.md:283) is a stale claim: Phase 0 found those
    # unreproducible and corrected the record, and docs/STATE.md:34 no longer
    # carries them.  Whatever the tree really does, it is not 13.
    assert len(led["failed"]) < 13, led["failed"]
    assert not any("M614" in node for node in led["failed"])
    # A recorded failure is a node id, never a count -- that is the whole
    # difference between this file and the paragraph it replaces.
    for node in led["failed"] + led["errors"]:
        assert isinstance(node, str) and "::" in node, node
    # _KNOWN_XFAIL is the real standing exception, and it is xfailed.
    for node in led["failed"] + led["errors"]:
        assert node not in _known_xfail(), node
    assert payload["environment_key"]
    assert payload["recorded"], "a record without a timestamp rots silently"
    assert payload["command"], "a record without its command is not a claim"


def _known_xfail():
    """``tests/conftest.py``'s ``_KNOWN_XFAIL`` -- read, never imported, so a
    broken conftest cannot make this file's gate tests uncollectable."""
    import re
    text = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    block = text.split("_KNOWN_XFAIL = {", 1)[1].split("\n}", 1)[0]
    return set(re.findall(r'"([^"]+)"', block))


def test_the_baseline_is_a_full_ledger_not_a_bare_count():
    """The Phase 0 lesson this file exists for: a bare count rots because
    nobody can tell which tests it counted.  The record carries the node id
    of every failure and error (none right now) *and* the counts."""
    led = _baseline_payload()["ledger"]
    for key in ("passed", "skipped", "xfailed", "xpassed", "deselected",
                "failed", "errors", "selected_in_failed_files", "selected"):
        assert key in led, key
    assert led["selected"] == (
        led["passed"] + len(led["failed"]) + len(led["errors"])
        + led["skipped"] + led["xfailed"] + led["xpassed"])


def test_the_recorded_environment_is_measured_not_declared():
    """The record must describe the box that ran the suite, including whether
    the oracle was resolvable -- the two rows of docs/STATE.md's table differ
    by exactly that."""
    payload = _baseline_payload()
    env = payload["environment"]
    assert set(env) >= {"python", "implementation", "platform", "packages",
                        "env", "resources"}
    assert env["python"] == platform.python_version()
    assert payload["environment_key"] == environment_key(env)
    assert isinstance(env["resources"]["OR_SRC"], bool)
    assert payload["provenance"]["git_commit"] not in ("", "unknown")


def test_the_committed_baseline_loads_through_the_tool_that_reads_it():
    led = load(BASELINE_PATH)
    assert led.passed > 1000
    assert led.environment_key
    # load() is the same parse --check performs, so this is the gate's own
    # entry point against the committed record.
    assert compare(led, led).new_failures == ()
