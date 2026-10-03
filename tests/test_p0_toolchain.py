"""Phase 0 (P0.1) — toolchain probe for the Fortran reference ("oracle") build.

The probe reports the tools the OpenCourant build needs (gfortran, cmake, make,
a *working* OpenMP, python3, docker) plus whether the extlib download host is
reachable.  ``openmp_ok`` is decided by compiling AND running a tiny OpenMP
Fortran program -- grepping a flag string proves nothing about the runtime.

Mirrors the Linux environment contract in $OR_SRC/INSTALL.md:34-42.

``tools/validation_data/toolchain_probe.json`` is a COMMITTED FACT, so this
module is its gate.  Five properties are load-bearing and each has a test:

* **the suite verifies, it never refreshes.**  The probe's read side
  (:func:`probe`, :func:`read_record`, :func:`differences`) writes nothing, and
  the only writer, ``main()``, is not reachable from the gate.  Before this
  module was a gate, ``test_probe_json_is_written`` called ``probe.main([])``:
  every run silently rewrote the record, so the committed copy was a rubber
  stamp -- the suite stayed green over a record naming
  ``/home/valentin/anaconda/bin/gfortran``, a directory this box does not have,
  and every run left a tracked file dirty.
* **drift fails, and the failure names the repair.**  The record is compared
  key-by-key against a fresh probe; the assertion quotes ``REFRESH_COMMAND``
  instead of only reporting a difference.
* **every value is the machine's, and says so.**  A recorded value may not name
  the checkout, the virtualenv or the interpreter that ran the probe --
  ``test_the_record_holds_no_checkout_local_value``.  ``python3`` is the name
  upstream's cmake hardcodes (``starter/CMakeLists.txt:18``), resolved from
  PATH, not ``sys.executable``.  ``gfortran_version`` is the compiler's own
  dotted version, not the packaging in its banner
  (``test_the_recorded_gfortran_version_is_the_compilers_own_version``).
* **the skip surface is one narrow, env-controlled predicate.**  The record
  names absolute paths of a machine, so it is checked where the oracle is
  installed -- the box whose toolchain built it -- and skipped elsewhere (a CI
  runner has no oracle).  That predicate is the repo's own
  ``_require_oracle_box`` / ``_require_live_oracle``, imported rather than
  re-invented, so it carries ``PYRADIOSS_ORACLE_DISABLED=1`` (silence),
  ``PYRADIOSS_ORACLE_REQUIRED=1`` (turn the skip into a failure) and the
  "exported but unresolvable is always a failure" rule that P0.14 added.  A
  box with no compiler is the same story one level down: a live claim that
  needs the toolchain to measure it skips through the *one* narrow
  :func:`_skip_without_a_probeable_toolchain`, the same predicate the gate
  uses, with the missing tool and PATH named in the reason.  Nothing here skips
  silently, and nothing here skips on a condition the gate does not: three
  checks are box-independent and run everywhere -- the record exists, parses and
  carries every declared key; it holds no checkout-local value; and the version
  parser answers every banner shape in a table, offline.
"""

import sys

import pytest

RECORD_REL = "tools/validation_data/toolchain_probe.json"

#: Tools whose absence leaves the record's central claim unmeasurable.  Narrow
#: on purpose: an absent compiler is an environment fact, and it is named in the
#: reason rather than turned into a pass.
REQUIRED_TOOLS = ("gfortran", "cmake", "make")


def _probe_module():
    from tools.oracle import toolchain_probe

    return toolchain_probe


def _skip_without_a_probeable_toolchain(fresh):
    import os

    missing = [tool for tool in REQUIRED_TOOLS if not fresh.get(tool)]
    if missing:
        pytest.skip(
            "the toolchain record cannot be verified: no "
            + ", ".join(missing)
            + " on PATH (shutil.which found none), so the probe cannot measure "
            f"what the record claims. This is an environment fact, not a "
            f"defect in the record. PATH={os.environ.get('PATH')!r}"
        )


def _require_oracle_box():
    """Skip unless this box has the oracle installed; fail when it is broken.

    The record names absolute paths, so it is only meaningful on a machine that
    builds/runs the oracle -- which is exactly what the repo's single oracle
    presence predicate already decides (tests/test_p0_oracle_build.py, imported
    by test_p0_oracle_selftest.py and test_p0_harness_portable.py too).  No new
    env var, no second answer to "is this the oracle box".
    """
    from tests.test_p0_oracle_build import _require_live_oracle

    _require_live_oracle(runtime_env=False)


def test_probe_reports_required_tools():
    fresh = _probe_module().probe()
    _skip_without_a_probeable_toolchain(fresh)
    for key in ("gfortran", "cmake", "make", "python3", "openmp_ok"):
        assert key in fresh
    # the project's own rule for a usable oracle compiler: a GCC major in
    # 11..15 (plan/01_phase0_oracle_and_licensing.md:177)
    assert fresh["gfortran_version"].startswith(("11", "12", "13", "14", "15")), (
        f"gfortran_version {fresh['gfortran_version']!r} is not a GCC major in "
        f"11..15; the probe parsed {fresh['gfortran']!r} --version into it"
    )


def test_the_recorded_gfortran_version_is_the_compilers_own_version():
    """``gfortran_version`` must be the version, not the packaging.

    GCC prints the packaging in parentheses and its own version after it --
    ``GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`` -- and the parse
    used to take the first token with two leading numeric parts, which is the
    Debian *package* version.  The record therefore held ``13.3.0-6ubuntu2~24``
    under a key called ``gfortran_version``: a truncated package string that
    satisfied the project's ``startswith`` 11..15 rule by accident, and that no
    reader would recognise as the compiler's version.

    Four properties, all checkable here:
      * digits and dots only, so a packaging suffix can never reappear;
      * a whitespace-delimited token of the banner the compiler printed, so the
        version is quoted from the compiler and not invented;
      * the plan's own 11..15 rule holds on the RECORDED string;
      * the box's own banner parses to it -- and that last claim is a LIVE one,
        so it skips on the same narrow missing-toolchain predicate the gate
        uses (:func:`_skip_without_a_probeable_toolchain`).  Without that guard
        this test failed on a compiler-less box with ``assert '' == '13.3.0'``
        while the gate skipped, so a failure pointed at a test that had skipped.
    """
    tp = _probe_module()
    record = tp.read_record()
    version = record["gfortran_version"]
    banner = record["gfortran_banner"]

    assert version and all(c.isdigit() or c == "." for c in version), (
        f"recorded gfortran_version {version!r} is not a bare dotted number; "
        f"the banner it came from is {banner!r}")
    assert version in banner.split(), (
        f"recorded gfortran_version {version!r} is not a token of the banner "
        f"the compiler printed, {banner!r}")
    assert version.startswith(("11", "12", "13", "14", "15")), (
        f"recorded gfortran_version {version!r} does not satisfy the project's "
        f"rule (plan/01_phase0_oracle_and_licensing.md:177): a GCC major in "
        f"11..15")

    fresh = tp.probe()
    _skip_without_a_probeable_toolchain(fresh)
    assert fresh["gfortran_version"] == version, (
        "the recorded version and the measured one disagree -- the gate in "
        f"test_the_committed_record_matches_a_fresh_probe reports the drift")


#: Every banner shape a real ``gfortran --version`` is known to print, and the
#: compiler's OWN version each must yield.  A parser repaired against the banner
#: on the box that ran it is a parser that will misparse the next distro, so the
#: class is closed here instead of by one more fix round: the shapes a formula-
#: based distro (``gfortran-13 (Homebrew 13.1) 13.2.0``), a plain upstream GCC,
#: Red Hat (which appends a *date*), a distro that names GCC in the
#: parentheses (``(GCC-14.2.0)``), a bare program name and a bare version all
#: have a stated answer.  Two of these rows are the ones that break a
#: "first dotted-numeric token" rule: the Homebrew banner's first such token is
#: the Homebrew version ``13.1``, and Red Hat's is a truncated package string.
BANNER_SHAPES = (
    pytest.param("GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0", "13.3.0",
                 id="debian-ubuntu"),
    pytest.param("GNU Fortran (GCC) 13.2.0", "13.2.0", id="upstream-gcc"),
    pytest.param("gfortran (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)", "4.8.5",
                 id="red-hat"),
    pytest.param("gfortran-13 (Homebrew 13.1) 13.2.0", "13.2.0",
                 id="homebrew"),
    pytest.param("GNU Fortran (GCC-14.2.0) 14.2.0", "14.2.0",
                 id="fedora-orasl"),
    pytest.param("gfortran-13", "13", id="bare-program-name"),
    pytest.param("13.2.0", "13.2.0", id="version-only"),
    # nothing to parse: the banner is recorded verbatim rather than as an empty
    # string, so a reader sees what the compiler actually said
    pytest.param("GNU Fortran (packaging unknown)", "GNU Fortran (packaging unknown)",
                 id="unparseable-is-verbatim"),
)


#: The two rows that cannot satisfy ``startswith(("11".."15"))``, each with the
#: reason -- named, not silently dropped, and counted in the assertion below so a
#: new row cannot join them by accident.
OUTSIDE_THE_PLAN_RULE = {
    "red-hat": "GCC 4.8.5 predates the 11..15 majors this project accepts",
    "unparseable-is-verbatim": "no version in the banner to parse; it is recorded "
                               "verbatim",
}


@pytest.mark.parametrize(("banner", "expected"), BANNER_SHAPES)
def test_the_version_parse_survives_every_banner_shape(banner, expected):
    """``_gcc_version`` must yield the compiler's version for every known shape.

    Three properties per row, so the table cannot be satisfied by a rule that
    merely returns something plausible:

      * the stated answer, exactly;
      * the answer is *quoted* -- a substring of the banner the compiler
        printed -- so it can never be invented;
      * the answer is never the parenthesised vendor/packaging version, and
        unless the row is the documented verbatim fallback it is digits and
        dots only.
    """
    tp = _probe_module()
    parsed = tp._gcc_version(banner)

    assert parsed == expected, (
        f"{banner!r} -> {parsed!r}, expected {expected!r}; the parenthesised "
        "group is the packaging, never the compiler's own version")
    assert parsed in banner, f"{parsed!r} is not quoted from {banner!r}"
    assert f"({parsed})" not in banner, (
        f"{parsed!r} is the parenthesised vendor version of {banner!r}, not the "
        "compiler's own")
    if expected != banner:  # not the documented verbatim fallback row
        assert all(c.isdigit() or c == "." for c in parsed), (
            f"{parsed!r} carries a packaging suffix; it must be digits and dots")


def test_the_plan_rule_holds_for_every_banner_this_project_accepts():
    """The project's own ``startswith(("11".."15"))`` rule, per banner shape.

    ``plan/01_phase0_oracle_and_licensing.md:177`` requires the recorded
    ``gfortran_version`` to be a GCC major in 11..15, and a parser that can
    return ``13.1`` for a Homebrew compiler -- or a Red Hat packaging string --
    satisfies that rule by accident.  Every row outside :data:`OUTSIDE_THE_PLAN_RULE`
    is checked, and the count makes sure a row cannot be added that escapes it.
    """
    tp = _probe_module()
    rule = ("11", "12", "13", "14", "15")
    checked = 0

    for case in BANNER_SHAPES:
        if case.id in OUTSIDE_THE_PLAN_RULE:
            continue
        banner, expected = case.values
        checked += 1
        parsed = tp._gcc_version(banner)
        assert parsed == expected, f"{case.id}: {banner!r} parsed to {parsed!r}"
        assert parsed.startswith(rule), (
            f"{case.id}: {banner!r} parses to {parsed!r}, which is not a GCC "
            f"major in 11..15")

    ids = {case.id for case in BANNER_SHAPES}
    assert set(OUTSIDE_THE_PLAN_RULE) <= ids, (
        "OUTSIDE_THE_PLAN_RULE exempts rows that do not exist, so a future row "
        f"cannot be checked by accident: {sorted(set(OUTSIDE_THE_PLAN_RULE) - ids)}")
    assert checked == len(BANNER_SHAPES) - len(OUTSIDE_THE_PLAN_RULE), (
        f"{checked} of the {len(BANNER_SHAPES) - len(OUTSIDE_THE_PLAN_RULE)} "
        "rows that must satisfy the plan rule did not get checked")

    # the reviewer's counterexample, named so it cannot be quietly re-broken
    assert tp._gcc_version("gfortran-13 (Homebrew 13.1) 13.2.0") == "13.2.0", (
        "a formula-based distro's banner must not yield the Homebrew 13.1")


def test_network_ok_measures_the_network_and_not_one_url(monkeypatch):
    """A URL that 404s is not a network that is down.

    The record used to carry ``network_ok`` measured by a HEAD against the
    ``url`` in ``$OR_SRC/EXTLIB_VERSION.json``.  That asset 404s here (the
    organisation moved it), so the record said ``network_ok: false`` on a box
    whose network demonstrably works -- a key asserting a machine fact it does
    not hold.  Two facts, two keys:

      * ``network_ok`` -- is a known-good HOST reachable at all (the machine
        fact Task 0.3 needs before it tries to download anything);
      * ``extlib_url_reachable`` -- does that one asset URL still answer (a
        fact about the release, not about the box).

    Driven against a fake ``urlopen`` so the test is offline-deterministic:
    the host answers 200 while the asset 404s, so the two keys MUST differ, and
    the URLs each is asked about are asserted, not assumed.
    """
    import urllib.error
    import urllib.request

    tp = _probe_module()
    extlib = tp._extlib_url()
    assert extlib and extlib != tp.NETWORK_HOST_URL, (
        f"the network host {tp.NETWORK_HOST_URL!r} and the extlib asset "
        f"{extlib!r} are the same URL; then one key would carry both facts")
    requested = []

    class _Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        requested.append(req.full_url)
        if req.full_url == tp.NETWORK_HOST_URL:
            return _Answer()
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert tp.probe_network() is True, (
        "a 404 from one asset URL must not be reported as a dead network")
    assert tp.probe_extlib_url() is False, (
        "the extlib URL answered 404; the key must say so")
    assert requested == [tp.NETWORK_HOST_URL, extlib], requested


def test_the_record_separates_the_network_from_the_extlib_url():
    """Both keys are declared, and both are booleans the probe measures.

    The record must not lose the extlib fact when ``network_ok`` is corrected:
    two keys, both present, so no information the old single key carried is
    dropped on the floor.
    """
    tp = _probe_module()
    record = tp.read_record()

    for key in ("network_ok", "extlib_url_reachable"):
        assert key in tp.RECORD_KEYS, (
            f"{key} is measured but undeclared; a key the gate does not know "
            "about cannot be compared key-by-key")
        assert key in record, f"{RECORD_REL} does not carry {key}"
        assert isinstance(record[key], bool), (
            f"{RECORD_REL} records {key}={record[key]!r}; a reachability answer "
            "is a bool, never a string or a null")


def test_the_record_holds_no_checkout_local_value():
    """Every recorded value must be a property of the machine, not of the tree.

    ``python3`` used to hold ``sys.executable``, i.e. the venv of whichever
    checkout ran the suite.  That made the record change when the worktree
    moved -- a record of the checkout, not of the build machine -- and made the
    gate report "the toolchain moved" when nothing the oracle build invokes had
    moved.  The repair is a key the build actually resolves (``shutil.which`` of
    the ``python3`` upstream hardcodes), and this test is what stops a future
    key from reintroducing the shape.
    """
    tp = _probe_module()
    record = tp.read_record()
    # Only genuinely checkout-local prefixes.  ``sys.base_prefix`` is NOT one of
    # them: /usr is the system interpreter prefix, and /usr/bin/cmake is exactly
    # the kind of value this record exists to hold.
    local = (str(tp.REPO_ROOT), sys.prefix)
    offenders = [
        f"{key}={value!r} names {needle}"
        for key, value in sorted(record.items())
        for needle in local
        if needle and needle in str(value)
    ]
    assert not offenders, (
        f"{RECORD_REL} records checkout-local values, so it describes the "
        f"worktree and not the build machine: {'; '.join(offenders)}. Record "
        f"what the BUILD resolves (shutil.which of the name it invokes), never "
        f"sys.executable or a path under the repository.")


def test_openmp_is_decided_by_compiling_not_by_reading_a_flag():
    from tools.oracle.toolchain_probe import probe

    # openmp_ok must be a real bool from an actual compile+run
    assert isinstance(probe()["openmp_ok"], bool)
    # and the program that decides it is an OpenMP program, not an empty file
    assert "!$omp parallel" in _probe_module().OMP_SOURCE, (
        "the openmp probe compiles a program with no OpenMP directive in it, so "
        "a compiler with no runtime would still pass it")


def test_the_openmp_probe_compiles_the_program_and_then_runs_it(monkeypatch):
    """The decision is ``-fopenmp`` on the command line AND a program that exits.

    Grepping ``-fopenmp`` out of a makefile proves the *flag* is accepted; only
    running the binary proves ``libgomp`` is there.  Driven against a recorded
    ``_run`` so the two steps are visible offline, and the phase-0 reviewer
    checklist item for P0.1 (``openmp_ok`` is decided by compiling *and
    running*) is a test rather than a claim about the source.
    """
    tp = _probe_module()
    calls = []

    def fake_run(cmd, timeout=30):
        calls.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(tp, "_run", fake_run)

    assert tp.probe_openmp("gfortran") is True
    assert len(calls) == 2, f"expected a compile then a run, got {calls}"
    compile_cmd, run_cmd = calls
    assert compile_cmd[:2] == ["gfortran", "-fopenmp"], compile_cmd
    assert "-o" in compile_cmd and compile_cmd[-1].endswith("omp_probe"), (
        f"the probe does not link a program: {compile_cmd}")
    assert run_cmd == [compile_cmd[-1]], (
        f"the binary that was compiled is not the one that was run: {calls}")

    # a failure at EITHER step must make openmp_ok False
    for failing_step, step in ((0, "the compile"), (1, "the run")):
        calls.clear()

        def failing_run(cmd, timeout=30, _step=failing_step, _calls=calls):
            _calls.append(list(cmd))
            return (1, "simulated failure") if len(_calls) == _step + 1 \
                else (0, "")

        monkeypatch.setattr(tp, "_run", failing_run)
        assert tp.probe_openmp("gfortran") is False, (
            f"a failing {step} must make openmp_ok False -- accepting the flag "
            "and running the program are both required")


def test_the_committed_record_is_readable_and_complete():
    """The record must exist, parse, and carry every key the probe measures.

    Runs on EVERY machine, oracle box or not: a missing or truncated file is a
    repository defect wherever it is read, and it is the one thing about the
    record that is not box-specific.
    """
    tp = _probe_module()
    try:
        record = tp.read_record()
    except ValueError as exc:
        pytest.fail(
            f"{exc}\nThe record is committed, so it must be present and "
            f"parsable. Recreate it with:  {tp.REFRESH_COMMAND}"
        )
    missing = [key for key in tp.RECORD_KEYS if key not in record]
    assert not missing, (
        f"{RECORD_REL} is missing {missing}; a record that cannot be compared "
        f"key-by-key is not a record. Recreate it with:  {tp.REFRESH_COMMAND}"
    )


def test_the_committed_record_matches_a_fresh_probe():
    """THE GATE.  The record is a claim about this box; check it.

    WHICH-STALE-RECORD: the committed copy named the conda toolchain of a
    prefix this box lost (``/home/valentin/anaconda/bin/gfortran``,
    ``cmake version 4.4.3``), and nothing compared it to a measurement, so the
    suite was green over a record describing tools that are not installed.
    Here every key is compared against a fresh probe and a difference fails
    with the command that repairs it.
    """
    _require_oracle_box()
    tp = _probe_module()
    fresh = tp.probe()
    record = tp.read_record()

    _skip_without_a_probeable_toolchain(fresh)
    diffs = tp.differences(record, fresh)
    assert diffs == [], (
        f"{RECORD_REL} no longer describes this box:\n" + tp.explain(diffs))


def test_the_read_side_of_the_probe_writes_nothing(tmp_path, monkeypatch):
    """Purity is what keeps a test run from rewriting the record it checks.

    Every read-side entry point is driven against a record path that does not
    exist; if any of them creates it, the file appears and this fails.
    """
    tp = _probe_module()
    target = tmp_path / "toolchain_probe.json"
    monkeypatch.setattr(tp, "JSON_PATH", target)

    tp.probe()
    assert not target.exists(), f"probe() wrote {target}"

    try:
        tp.read_record()
    except ValueError as exc:
        assert str(target) in str(exc), exc
    else:
        raise AssertionError(f"read_record() invented {target}")

    fresh = tp.probe()
    absent = tp.differences({}, fresh)
    assert len(absent) == len(fresh), (
        f"differences() against an empty record returned {len(absent)} lines "
        f"for {len(fresh)} keys; a key the record does not carry must be "
        f"reported, or a shrinking record would compare clean")
    assert not target.exists(), f"a read-side call wrote {target}"


def test_refreshing_is_an_explicit_act_that_writes_the_record(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """``main()`` must still write, or the drift message would quote a command
    that quietly does nothing."""
    tp = _probe_module()
    target = tmp_path / "toolchain_probe.json"
    monkeypatch.setattr(tp, "JSON_PATH", target)

    rc = tp.main([])

    assert rc == 0, f"main() returned {rc}"
    assert target.is_file(), f"main() did not write {target}"
    assert tp.differences(tp.read_record(target), tp.probe()) == [], (
        "main() wrote a record that does not match the probe it just ran")
    out = capsys.readouterr().out
    assert str(target) in out, f"main() did not name the file it wrote: {out!r}"


def test_a_drift_report_names_every_differing_key_and_the_repair():
    """The failure text is part of the contract: it must be actionable without
    reading the probe's source."""
    tp = _probe_module()
    drift = tp.differences({"gfortran": "/gone/gfortran", "make": "/usr/bin/make"},
                           {"gfortran": "/usr/bin/gfortran",
                            "make": "/usr/bin/make"})
    report = tp.explain(drift)

    assert drift == ["gfortran: recorded '/gone/gfortran', "
                     "measured '/usr/bin/gfortran'"], drift
    assert "gfortran: recorded '/gone/gfortran', " \
           "measured '/usr/bin/gfortran'" in report, report
    assert tp.REFRESH_COMMAND in report, report
    assert tp.REFRESH_COMMAND.endswith("tools.oracle.toolchain_probe"), (
        f"the quoted repair {tp.REFRESH_COMMAND!r} does not name the module")
    assert tp.explain([]) == "", "an empty drift must not print a repair"
