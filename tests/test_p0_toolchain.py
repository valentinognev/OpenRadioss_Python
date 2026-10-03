"""Phase 0 (P0.1) — toolchain probe for the Fortran reference ("oracle") build.

The probe reports the tools the OpenCourant build needs (gfortran, cmake, make,
a *working* OpenMP, python3, docker) plus whether the extlib download host is
reachable.  ``openmp_ok`` is decided by compiling AND running a tiny OpenMP
Fortran program -- grepping a flag string proves nothing about the runtime.

Mirrors the Linux environment contract in $OR_SRC/INSTALL.md:34-42.

``tools/validation_data/toolchain_probe.json`` is a COMMITTED FACT, so this
module is its gate.  Four properties are load-bearing and each has a test:

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
  ``_require_live_oracle``, imported rather than re-invented, so it carries
  ``PYRADIOSS_ORACLE_DISABLED=1`` (silence), ``PYRADIOSS_ORACLE_REQUIRED=1``
  (turn the skip into a failure) and the "exported but unresolvable is always a
  failure" rule that P0.14 added.  Nothing here skips silently: the two
  structural checks (the record exists, parses and carries every declared key)
  run on every machine, and the live claims below run everywhere too.
"""

import pathlib
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
    # 11..15 (plan/01_phase0_oracle_and_licensing.md:172)
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

    Three properties, all checkable here:
      * digits and dots only, so a packaging suffix can never reappear;
      * a whitespace-delimited token of the banner the compiler printed, so the
        version is quoted from the compiler and not invented;
      * the plan's own 11..15 rule holds on the RECORDED string.
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
        f"rule (plan/01_phase0_oracle_and_licensing.md:172): a GCC major in "
        f"11..15")
    assert tp.probe()["gfortran_version"] == version, (
        "the recorded version and the measured one disagree -- the gate in "
        f"test_the_committed_record_matches_a_fresh_probe reports the drift")


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