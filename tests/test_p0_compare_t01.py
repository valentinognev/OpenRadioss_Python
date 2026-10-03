"""Gate for P0.10 -- the binary T01 reader and the per-channel rel-RMS scorer.

``tools/compare_t01.py`` turns "two runs" into "one comparable number": it reads
the Fortran engine's binary time-history file (the *T01*, ``ITTYP==3`` Radioss
IEEE) and scores it against another ``T01`` -- in practice the ASCII T01 CSV the
port writes -- with the same verdict vocabulary and the same 5 % threshold the
historical parity tables used, so a verdict produced here is comparable with
``tools/validation_data/parity_m41.json``.

Three independent cross-checks of the reader, all required, because the brief's
own Step 4 (cross-check against upstream's ``th_to_csv``) cannot be run on this
box:

1. **Upstream layout conformance**
   (``test_layout_constants_match_the_cited_upstream_source``) -- every
   structural constant the reader uses is listed in :data:`C.LAYOUT` with the
   upstream ``file:line`` range it comes from and a pattern that must still be
   found there.  A constant that drifts from its source, or a source line that
   moves, fails the test.  Skips loudly when ``$OR_SRC`` is not readable.
2. **A committed golden** (``test_read_t01_*``) -- the T01 the P0.5 golden run
   committed (``tests/data/oracle_smoke/TENSILET01``) must decode to the
   channel set, the sample count and the per-channel maxima recorded in
   ``tools/validation_data/oracle_smoke.json``.  That record was produced by a
   *different* module (``tools/oracle/oracle_selftest.py``), so agreement is a
   real cross-check and a disagreement means one of the two is wrong.
3. **A second producer** (``test_binary_reader_agrees_with_the_port_csv``) --
   the port runs ``examples/tensile_bar`` (the same deck as the golden run) and
   writes its own T01 CSV; the binary reader's channels are scored against it.
   Two independent producers of the same physics, one binary and one ASCII.

Plus ``test_cross_check_against_th_to_csv_when_available``: the brief's own
cross-check, wired to run *if* upstream's converter is ever installed
(``OR_TH_TO_CSV``, or ``$OR_ROOT/{bin,exec}/th_to_csv_*``) and to skip with a
reason naming every place that was looked at when it is not.

The run stamp
-------------
``hist1.F:211`` stamps ``ctime()`` into the T01 header (``timer_c.c:30-40``), so
the raw bytes of two runs of the same deck differ inside a 24-byte window and
nowhere else.  Two tests make that a property of the *reader* and not just a
recorded fact: the stamp is located by content and reported separately, and
overwriting it with a different 24-character timestamp leaves every decoded
channel value bit-identical (``test_read_t01_is_indifferent_to_the_run_stamp``).

Upstream Fortran origins (``$OR_SRC`` = the read-only OpenCourant tree):

* ``common_source/tools/input_output/write_routines.c:499-511`` (``eor_c``) --
  the 4-byte record marker that opens *and* closes every record; the format is
  big-endian.
* ``:520-540`` (``write_r_c``) and ``engine/source/output/tools/ieee.cpp:66-124``
  (``real_to_IEEE_ASCII``) / ``:127-165`` (``IEEE_ASCII_to_real``) -- how a
  channel value is stored: sign, 8-bit exponent, **24**-bit mantissa.
* ``:646-664`` (``write_i_c``) -- big-endian int32 for the header integers.
* ``engine/source/output/th/wrtdes.F:121-133`` -- ``ITTYP==3`` writes every
  value through ``R4``, i.e. single precision even in a double-precision build.
* ``engine/source/output/th/hist1.F:132-144`` -- the format code that fixes the
  title width (``LTITL``) and whether the additional records exist.
* ``engine/source/output/th/hist1.F:201-208`` -- header record 1
  (format code + deck title).
* ``engine/source/output/th/hist1.F:210-234`` with
  ``engine/source/system/timer_c.c:30-40`` -- header record 2, the wall clock.
* ``engine/source/output/th/hist1.F:237-290`` -- the ``TH_VERS>=50`` additional
  records, the last of which is the ``(FAC_MASS, FAC_LENGTH, FAC_TIME)``
  unit-scaling triple.
* ``engine/source/output/th/hist1.F:292-316`` -- the hierarchy record and
  ``NGLOBTH=23``.
* ``engine/source/output/th/hist1.F:357-376`` -- the per-part record and the
  part's curve codes.
* ``engine/source/output/th/hist2.F:302-303``, ``:307-333``, ``:338-477`` --
  the per-step records: ``TT``, the 23 global channels, the parts.
* ``starter/source/output/th/write_thnms1.F90:230-252`` -- the authoritative
  index/name/description table for the 23 global channels.
* ``starter/source/output/th/th_titles.F90:2759-2792`` (``varpa_title``) -- the
  per-part curve codes.
* ``tools/th_to_csv/README.md:1-7`` -- why the converter is optional here: its
  source lives in a separate repository this box cannot reach.
"""

import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

SMOKE_JSON = REPO / "tools" / "validation_data" / "oracle_smoke.json"
PARITY_M41 = REPO / "tools" / "validation_data" / "parity_m41.json"
GOLDEN_T01 = REPO / "tests" / "data" / "oracle_smoke" / "TENSILET01"
DECK_DIR = REPO / "examples" / "tensile_bar"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def smoke():
    assert SMOKE_JSON.is_file(), (
        f"missing golden record {SMOKE_JSON}; regenerate with "
        "`python -m tools.oracle.oracle_selftest --write`")
    return json.loads(SMOKE_JSON.read_text())


@pytest.fixture(scope="module")
def read_golden():
    """The committed golden T01, read by the module under test."""
    from tools.compare_t01 import read_t01

    assert GOLDEN_T01.is_file(), f"missing golden T01 {GOLDEN_T01}"
    return read_t01(GOLDEN_T01)


@pytest.fixture(scope="module")
def port_t01():
    """The port's own T01 CSV for the golden deck -- the second producer.

    One live starter+engine run of ``examples/tensile_bar`` in a private
    scratch directory (the same deck pair the golden run used, so the two
    series describe the same physics), module-scoped because it costs ~4 s and
    three tests below read it.  A port failure FAILS here rather than skipping:
    this is the port's output, not an external resource.
    """
    import tempfile

    work = pathlib.Path(tempfile.mkdtemp(prefix="p010_port_"))
    try:
        for deck in ("TENSILE_0000.rad", "TENSILE_0001.rad"):
            shutil.copy2(DECK_DIR / deck, work / deck)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        for module, deck in (("pyradioss.starter", "TENSILE_0000.rad"),
                             ("pyradioss.engine", "TENSILE_0001.rad")):
            proc = subprocess.run(
                [sys.executable, "-m", module, "-i", deck], cwd=work, env=env,
                capture_output=True, text=True, errors="replace", timeout=600)
            assert proc.returncode == 0, (
                f"the port failed on {deck} (rc={proc.returncode}):\n"
                f"{(proc.stdout + proc.stderr)[-2000:]}")
        csv = work / "TENSILET01.csv"
        assert csv.is_file(), (
            f"the port wrote no T01 CSV: "
            f"{sorted(p.name for p in work.iterdir())}")
        from tools.compare_t01 import read_port_csv

        yield read_port_csv(csv)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# score(): the three decisions the brief forbids guessing
# ---------------------------------------------------------------------------

def _pair(ref_values, port_values, channels=("T", "IE")):
    from tools.compare_t01 import T01

    n = len(ref_values)
    return (T01(channels=list(channels), times=np.arange(float(n)),
                values=np.asarray(ref_values, dtype=float), scaling=[]),
            T01(channels=list(channels), times=np.arange(float(n)),
                values=np.asarray(port_values, dtype=float), scaling=[]))


def test_score_reports_rel_rms_per_channel():
    """The brief's synthetic pair, with the numbers pinned.

    ``rel_rms = sqrt(mean((port-ref)^2)) / max(|ref|)`` per channel (brief Step
    3) with ``MATCH_RMS = 0.05`` (brief interface):

    * identical series -> ``rel_rms == 0.0`` and ``MATCH``;
    * the brief's ``off`` series (IE 100/95 against 100/90) -> the deviation is
      ``[0, 5]``, so ``sqrt(12.5) = 3.5355339...`` over ``max|ref| = 100``:
      ``rel_rms = 0.035355339059327376``.  That is **below** 0.05, so it is a
      MATCH.  The brief's Step 1 snippet asserts ``DEVIATION`` for this pair,
      which its own Step 3 formula and its own 0.05 threshold cannot both
      produce; the formula and the threshold are the authoritative ones (the
      threshold is pinned to ``parity_m41.json``), so this test records the
      number instead of asserting the contradiction.  ``far`` below is the same
      shape with a deviation that does cross the threshold.
    """
    from tools.compare_t01 import score

    ref, same = _pair([[0.0, 100.0], [1.0, 90.0]],
                      [[0.0, 100.0], [1.0, 90.0]])
    result = score(ref, same)
    assert result.worst.rel_rms == 0.0
    assert result.worst.verdict == "MATCH"
    assert set(result.per_channel) == {"T", "IE"}
    assert all(c.verdict == "MATCH" and c.n == 2 for c in
               result.per_channel.values())

    _, off = _pair([[0.0, 100.0], [1.0, 90.0]], [[0.0, 100.0], [1.0, 95.0]])
    ie = score(ref, off).per_channel["IE"]
    assert ie.rel_rms == pytest.approx(math.sqrt(12.5) / 100.0, rel=1e-15)
    assert ie.rel_rms == pytest.approx(0.035355339059327376, rel=1e-15)
    assert ie.max_abs == pytest.approx(5.0)
    assert ie.n == 2
    assert ie.verdict == "MATCH", (
        "0.0354 < MATCH_RMS: the brief's Step 1 snippet calls this pair a "
        "DEVIATION, which its own formula and threshold do not allow")

    _, far = _pair([[0.0, 100.0], [1.0, 90.0]], [[0.0, 100.0], [1.0, 80.0]])
    far_ie = score(ref, far).per_channel["IE"]
    assert far_ie.rel_rms == pytest.approx(math.sqrt(50.0) / 100.0)
    assert far_ie.verdict == "DEVIATION"
    # the T channel is untouched by an IE-only deviation
    assert score(ref, far).per_channel["T"].verdict == "MATCH"
    assert score(ref, far).worst.verdict == "DEVIATION"


def test_channel_order_is_matched_by_name_not_position():
    """Channels are paired by NAME; the column order is irrelevant."""
    from tools.compare_t01 import T01, score

    times = np.array([0.0, 1.0])
    ref = T01(channels=["T", "IE"], times=times,
              values=np.array([[0.0, 100.0], [1.0, 90.0]]), scaling=[])
    # the same two channels, the other way round
    port = T01(channels=["IE", "T"], times=times,
               values=np.array([[100.0, 0.0], [90.0, 1.0]]), scaling=[])
    result = score(ref, port)
    assert result.worst.rel_rms == 0.0
    assert result.worst.verdict == "MATCH"
    assert result.per_channel["IE"].rel_rms == 0.0

    # and the same data in the same order as the reference is identical
    same = T01(channels=["T", "IE"], times=times,
               values=np.array([[0.0, 100.0], [1.0, 90.0]]), scaling=[])
    assert score(ref, same).per_channel == result.per_channel


def test_channel_present_on_one_side_only_is_nodata_not_a_deviation():
    """A channel missing on one side is NODATA and never drives ``worst``."""
    from tools.compare_t01 import T01, score

    times = np.array([0.0, 1.0])
    ref = T01(channels=["T", "IE", "KE"], times=times,
              values=np.array([[0.0, 100.0, 5.0], [1.0, 90.0, 7.0]]),
              scaling=[])
    port = T01(channels=["IE", "T"], times=times,
               values=np.array([[100.0, 0.0], [90.0, 1.0]]), scaling=[])
    result = score(ref, port)
    assert result.per_channel["KE"].verdict == "NODATA"
    assert result.per_channel["KE"].n == 0
    assert math.isinf(result.per_channel["KE"].rel_rms)
    assert result.per_channel["IE"].verdict == "MATCH"
    assert result.worst.verdict == "MATCH", (
        "a NODATA channel must be excluded from worst; the port here has no KE "
        "column at all, which is a fact about the port's /TH request, not a "
        "physics deviation")

    # the reverse: a channel only the port has is NODATA too
    only_port = T01(channels=["IE", "T", "ERR%"], times=times,
                    values=np.array([[100.0, 0.0, 3.0], [90.0, 1.0, 4.0]]),
                    scaling=[])
    assert score(ref, only_port).per_channel["ERR%"].verdict == "NODATA"

    # nothing comparable at all -> worst is itself NODATA, never a crash
    empty = T01(channels=["Z"], times=times,
                values=np.array([[1.0], [2.0]]), scaling=[])
    assert score(ref, empty).worst.verdict == "NODATA"
    assert score(empty, ref).worst.verdict == "NODATA"


def test_all_zero_reference_channel_cannot_divide_by_zero():
    """An all-zero reference channel is scored in absolute units.

    The brief floors the denominator at ``max(|ref|).max() * 1e-12``, which is
    still zero for an identically-zero channel; the floor is therefore applied
    as written *and* the degenerate case falls back to an absolute comparison
    (denominator 1.0).  Two port behaviours then have to be distinguishable:
    a port channel that is exactly zero MATCHES, and one that is not does not.
    """
    from tools.compare_t01 import T01, score

    times = np.array([0.0, 1.0, 2.0])
    ref = T01(channels=["YMOM"], times=times,
              values=np.zeros((3, 1)), scaling=[])
    zero = T01(channels=["YMOM"], times=times, values=np.zeros((3, 1)),
               scaling=[])
    assert score(ref, zero).per_channel["YMOM"].rel_rms == 0.0
    assert score(ref, zero).per_channel["YMOM"].verdict == "MATCH"

    drifted = T01(channels=["YMOM"], times=times,
                  values=np.array([[0.0], [1.0], [0.0]]), scaling=[])
    bad = score(ref, drifted).per_channel["YMOM"]
    assert math.isfinite(bad.rel_rms)
    assert bad.rel_rms == pytest.approx(math.sqrt(1.0 / 3.0))
    assert bad.verdict == "DEVIATION"


def test_match_rms_is_the_tolerance_the_historical_tables_used():
    """``MATCH_RMS`` is pinned to ``parity_m41.json``'s recorded tolerance.

    Comparability with the historical verdicts is the whole reason the constant
    exists, so it is asserted against the record rather than against a literal
    in this file.
    """
    from tools.compare_t01 import MATCH_RMS, VERDICTS

    assert MATCH_RMS == 0.05
    assert VERDICTS == ("MATCH", "DEVIATION", "NODATA")
    record = json.loads(PARITY_M41.read_text())
    assert record["tolerance_rel_rms"] == MATCH_RMS, (
        "the M41 parity tables were recorded at a different tolerance; this "
        "scorer's verdicts would no longer be comparable with them")
    matches = [c for c in record["results"] if c["class"] == "MATCH"]
    deviations = [c for c in record["results"] if c["class"] == "DEVIATION"]
    assert matches and deviations, (
        "the M41 table must carry both verdicts for this to be evidence")
    assert all(c["max_rel_rms"] <= MATCH_RMS for c in matches), (
        "an M41 MATCH above MATCH_RMS would mean this scorer's threshold does "
        "not reproduce the historical verdicts")
    assert all(c["max_rel_rms"] > MATCH_RMS for c in deviations), (
        "an M41 DEVIATION at or below MATCH_RMS likewise")


def test_a_noise_channel_cannot_drive_the_worst_verdict():
    """``worst`` ignores channels that carry no signal -- the harness's rule.

    ``tools/validate_vs_fortran.py:1461-1465`` computes the table's
    ``max_rel_rms`` over the ``significant`` rows only, and marks a channel
    significant when it reaches 1 % of its group's dominant reference peak
    (``:1139-1146``).  Without that, the transverse momentum of a uniaxial pull
    -- ~1e-16 on the reference, ~1e-17 on the port, so ``rel_rms`` ~ 0.5 -- would
    make every deck DEVIATE.  The channel keeps its own (large) number in
    ``per_channel``; only the roll-up skips it.
    """
    from tools.compare_t01 import T01, score

    times = np.array([0.0, 1.0, 2.0])
    ref = T01(channels=["IE", "XMOM", "YMOM"], times=times,
              values=np.array([[0.0, 0.0, 1e-16],
                               [10.0, 1e-4, -1e-16],
                               [20.0, 2e-4, 2e-16]]),
              scaling=[])
    port = T01(channels=["IE", "XMOM", "YMOM"], times=times,
               values=np.array([[0.0, 0.0, 1e-17],
                                [10.0, 1e-4, -1e-17],
                                [20.0, 2e-4, 1e-17]]),
               scaling=[])
    result = score(ref, port)
    assert result.per_channel["YMOM"].rel_rms > 0.4, (
        "the noise channel really does carry a large RELATIVE deviation; that "
        "is the point")
    assert result.worst.verdict == "MATCH"
    assert result.worst.rel_rms == 0.0
    assert result.worst.n == 3


def test_read_port_csv_renames_columns_onto_the_upstream_names(tmp_path):
    """The port's column names are folded onto the upstream short names.

    ``MOMX -> XMOM``, ``EW -> EFW`` (``PORT_ALIASES``, the same pairs as
    ``tools/validate_vs_fortran.py:1036-1046`` seen from the other side) and
    ``P<id>_<VAR> -> P<id>_<code>`` (``PORT_PART_CODES``: the T01 header records
    a part's variables as curve codes, ``varpa_title`` ``th_titles.F90:2759-2762``
    says 1 is INTERNAL ENERGY and 2 KINETIC ENERGY, which is what the port
    computes for ``IE`` and ``KE``, ``time_history.py:93-97``).  A column with no
    upstream counterpart keeps its own name -- that is what makes it NODATA
    rather than silently unmatched.
    """
    from tools.compare_t01 import read_port_csv

    csv = tmp_path / "T01.csv"
    csv.write_text(
        "# pyradioss time history (T01 equivalent)\n"
        "TIME,IE,MOMX,EW,EN,P2_IE,P2_KE,P9_DX\n"
        "0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0\n"
        "1.0,1.5,2.5,3.5,4.5,5.5,6.5,7.5\n")
    got = read_port_csv(csv)
    assert got.channels == ["IE", "XMOM", "EFW", "EN", "P2_1", "P2_2", "P9_DX"]
    assert got.times.tolist() == [0.0, 1.0]
    assert got.values.shape == (2, 7)
    assert got.column("XMOM").tolist() == [2.0, 2.5]
    assert got.scaling == []
    with pytest.raises(KeyError, match="no channel"):
        got.column("MOMX")


def test_read_t01_states_its_known_limitation_rather_than_mis_parsing(tmp_path):
    """A per-step block this reader did not expect must be refused, loudly.

    The record walk is shared with ``tools.oracle.oracle_selftest``, which fixes
    the per-step record count at four (``TT``, the global block, the part block,
    the per-TH-group curves).  A deck that also asks for ``/TH/SUBSET`` curves
    gets a fifth record per step (``hist2.F:478-607``).  This crafts exactly
    that file -- the golden with one extra 4-byte record per step -- and pins
    that the reader REFUSES it with a message about the block shape, rather than
    returning a plausible, wrong series.  Widening the shared walk is a Phase 12
    change; until then the boundary has to be visible.
    """
    from tools.compare_t01 import T01FormatError, read_t01
    from tools.oracle.oracle_selftest import t01_records

    def reframe(payloads):
        out = bytearray()
        for payload in payloads:
            out += len(payload).to_bytes(4, "big")
            out += payload
            out += len(payload).to_bytes(4, "big")
        return bytes(out)

    payloads = [payload for _, payload in t01_records(GOLDEN_T01.read_bytes())]
    header = 14                # the golden's header record count
    steps = (len(payloads) - header) // 4
    assert steps == 100, steps
    widened = list(payloads[:header])
    for k in range(steps):
        widened.extend(payloads[header + 4 * k:header + 4 * k + 4])
        widened.append(b"\x00\x00\x80\x3f")      # one more float: 1.0
    assert len(widened) == header + 5 * steps
    path = tmp_path / "TENSILE_SUBSET_T01"
    path.write_bytes(reframe(widened))

    with pytest.raises(T01FormatError) as caught:
        read_t01(path)
    message = str(caught.value)
    assert "block shape is not uniform" in message or (
        "not a multiple of 4" in message), message


def test_read_port_csv_rejects_a_file_without_a_time_column(tmp_path):
    from tools.compare_t01 import T01FormatError, read_port_csv

    csv = tmp_path / "T01.csv"
    csv.write_text("# a header with no time column\nA,B\n0.0,1.0\n1.0,2.0\n")
    with pytest.raises(T01FormatError, match="TIME"):
        read_port_csv(csv)


def test_the_two_series_are_compared_on_the_port_grid():
    """Two runs never share a sample grid; the comparison must say which.

    ``tools/validation_data/parity_m41.json`` -> ``comparison_note`` records the
    historical semantics ("interpolated onto the port grid ... over the OVERLAP
    window [0, min(fortran, port) final time]"), which is what
    ``tools/validate_vs_fortran.py:1078-1081`` implements.  This scorer keeps
    them, so a verdict here means what an M36..M41 verdict meant.

    Reference 0/10/20 at t = 0/1/2; port 0/5/10/15/25/30 at t = 0/.5/1/1.5/2/2.5.
    The overlap window stops at ``min(t_last) = 2.0``, which leaves the five
    port samples below it; the reference interpolated onto them is 0/5/10/15/20,
    so the port is exact until t = 1.5 and 5.0 too high at t = 2.0.
    """
    from tools.compare_t01 import T01, score

    ref = T01(channels=["IE"], times=np.array([0.0, 1.0, 2.0]),
              values=np.array([[0.0], [10.0], [20.0]]), scaling=[])
    port = T01(channels=["IE"], times=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
               values=np.array([[0.0], [5.0], [10.0], [15.0], [25.0], [30.0]]),
               scaling=[])
    got = score(ref, port).per_channel["IE"]
    assert got.n == 5, "the overlap window stops at min(t_last) = 2.0"
    assert got.max_abs == pytest.approx(5.0)
    assert got.rel_rms == pytest.approx(math.sqrt(5.0) / 20.0)
    assert got.verdict == "DEVIATION"

    # a one-sample series is not a comparison at all (MIN_SAMPLES == 2)
    single = T01(channels=["IE"], times=np.array([0.0]),
                 values=np.array([[0.0]]), scaling=[])
    assert score(ref, single).per_channel["IE"].verdict == "NODATA"


# ---------------------------------------------------------------------------
# Cross-check 1: upstream layout conformance
# ---------------------------------------------------------------------------

def test_layout_constants_match_the_cited_upstream_source():
    """Every structural constant the reader uses is pinned to its source line.

    :data:`tools.compare_t01.LAYOUT` is the machine-readable form of the
    module's citations: ``name -> (upstream file, first line, last line,
    pattern)``.  Each pattern must still be present in that exact line range, so
    a constant that drifts from upstream -- or a source that moves -- fails here
    instead of silently producing a mis-parse.
    """
    from tools import compare_t01 as C

    root = C.upstream_root()
    if root is None:
        pytest.skip(
            "the read-only OpenCourant tree is not readable; set OR_SRC to it "
            "(the citations this test checks are the module's own "
            "file:line references) -- every constant is still documented in the "
            "module docstring, but nothing can be verified against the source")

    assert C.LAYOUT, "the citation table must not be empty"
    for name, (relative, first, last, pattern) in sorted(C.LAYOUT.items()):
        path = root / relative
        assert path.is_file(), f"{name}: {relative} is missing under {root}"
        lines = path.read_text(errors="replace").splitlines()
        assert 1 <= first <= last <= len(lines), (
            f"{name}: {relative}:{first}-{last} is outside the file "
            f"({len(lines)} lines) -- the citation has drifted")
        window = "\n".join(lines[first - 1:last])
        assert re.search(pattern, window), (
            f"{name}: {relative}:{first}-{last} no longer contains "
            f"{pattern!r}; the constant and its source disagree")


def test_every_layout_constant_is_actually_used_by_the_reader():
    """A citation with no reader behind it is documentation rot.

    The reverse direction of the test above: each name in :data:`C.LAYOUT` must
    appear in the module source, so a constant that is documented but not used
    (or added without a citation) is visible.
    """
    from tools import compare_t01 as C

    source = pathlib.Path(C.__file__).read_text()
    for name in C.LAYOUT:
        assert name in source, f"{name} is cited but absent from the module"


# ---------------------------------------------------------------------------
# Cross-check 2: the committed golden
# ---------------------------------------------------------------------------

def test_read_t01_channel_set_matches_the_golden_record(smoke, read_golden):
    """The decoded channel set is the record's channel set.

    23 global channels named from ``write_thnms1.F90:230-252``, then one
    channel per part curve code -- the part's own curve codes as the T01 header
    records them (``hist1.F:367-376``), which for the golden deck are ``[1, 2]``.
    """
    assert GOLDEN_T01.is_file()
    assert read_golden.channels[:23] == [c["name"]
                                         for c in smoke["channel_maxima"]
                                         ["global"]]
    assert len(read_golden.channels) == 25, read_golden.channels
    assert read_golden.channels[23:] == ["P1_1", "P1_2"], (
        "the T01 carries no part id (hist1.F:357-366 writes IPART(4,N), the "
        "title, IPART(7,N), the node bounds and NVAR -- no id), so the part is "
        "named by its 1-based position and its curve code")
    assert smoke["t01"]["part_curve_codes"] == [1, 2]


def test_read_t01_sample_count_and_maxima_match_the_golden_record(smoke,
                                                                  read_golden):
    """Re-derive the recorded maxima from the committed bytes, exactly.

    ``oracle_smoke.json`` was written by ``tools/oracle/oracle_selftest.py`` --
    a different module with a different parser.  Equality here is therefore
    evidence, not a tautology.  Values are compared with ``==``: the stored
    maxima are float32 widened to float64, so any difference is a real one.
    """
    import tools.oracle.oracle_selftest as selftest

    assert read_golden.values.shape == (
        smoke["t01"]["n_steps"], 25), read_golden.values.shape
    assert len(read_golden.times) == smoke["t01"]["n_steps"] == 100
    assert read_golden.times[0] == smoke["t01"]["t_first"] == 0.0
    assert read_golden.times[-1] == smoke["t01"]["t_last"]

    for column, entry in enumerate(smoke["channel_maxima"]["global"]):
        series = read_golden.values[:, column]
        assert float(series.max()) == entry["max"], entry["name"]
        assert float(np.abs(series).max()) == entry["max_abs"], entry["name"]
        assert float(series[-1]) == entry["final"], entry["name"]

    for column, entry in enumerate(smoke["channel_maxima"]["part"]):
        series = read_golden.values[:, 23 + column]
        assert float(series.max()) == entry["max"], entry
        assert float(np.abs(series).max()) == entry["max_abs"], entry
        assert float(series[-1]) == entry["final"], entry

    # and the whole decoded payload, record for record, against the other
    # parser -- proves the reader reuses that framing rather than re-deriving it
    parsed = selftest.parse_t01(GOLDEN_T01.read_bytes())
    assert np.array_equal(read_golden.values[:, :23], parsed["global"])
    assert np.array_equal(read_golden.values[:, 23:], parsed["part"])
    assert np.array_equal(read_golden.times, parsed["time"])


def test_read_t01_reads_the_header_records_and_reports_the_format(smoke,
                                                                  read_golden):
    """Format code, title width and the (absent) scaling record are reported.

    ``hist1.F:132-144`` picks the format code from ``TH_VERS``: 3040 (title
    width 40), 3041 (80), 3050 and 4021 (100, plus the two additional records
    of ``:237-290``).  The golden run used no ``/TH/VERS`` card, and
    ``freform.F:1543`` / ``hm_read_th.F:57`` leave ``TH_VERS`` below 47, so its
    T01 carries format code 3040 and **no** unit-scaling record -- hence
    ``scaling == []``.
    """
    from tools import compare_t01 as C

    assert read_golden.format_code == 3040
    assert read_golden.title_width == 40
    assert read_golden.scaling == [], (
        "a 3040 T01 has no additional records (hist1.F:237 writes them only for "
        "TH_VERS>=50), so there is no FAC_MASS/FAC_LENGTH/FAC_TIME triple")
    assert read_golden.n_records == smoke["t01"]["n_records"]
    assert read_golden.header_records == smoke["t01"]["header_records"]
    assert read_golden.n_steps == smoke["t01"]["n_steps"]
    assert read_golden.values.dtype == np.float64
    assert C.LAYOUT["format_code_table"][0].endswith("hist1.F")


def test_read_t01_locates_the_run_stamp_and_never_decodes_it(smoke, read_golden):
    """The ``ctime`` stamp is located, reported, and excluded from the data."""
    assert read_golden.run_stamp == (smoke["t01"]["run_stamp"]["offset"],
                                     smoke["t01"]["run_stamp"]["length"])
    assert read_golden.run_stamp == (96, 24)
    blob = GOLDEN_T01.read_bytes()
    start, size = read_golden.run_stamp
    stamp = blob[start:start + size].decode("ascii")
    assert stamp.endswith("2026") and len(stamp) == 24, stamp
    # it is header text, not a channel: the first data sample is t = 0
    assert read_golden.times[0] == 0.0


def test_read_t01_is_indifferent_to_the_run_stamp(tmp_path, read_golden):
    """Overwriting the 24 stamp bytes must change nothing that is decoded.

    This is the reader-level half of the P0.5 determinism finding: the raw T01
    bytes move with the wall clock, so a reader that mistook the stamp for
    channel data (or let it shift the record walk) would produce different
    numbers for two runs of one deck.
    """
    from tools.compare_t01 import read_t01

    blob = bytearray(GOLDEN_T01.read_bytes())
    start, size = read_golden.run_stamp
    other = b"Mon Jan  2 03:04:05 1995"
    assert len(other) == size
    blob[start:start + size] = other
    moved = tmp_path / "TENSILET01"
    moved.write_bytes(bytes(blob))
    reread = read_t01(moved)
    assert reread.channels == read_golden.channels
    assert reread.times.tolist() == read_golden.times.tolist()
    assert reread.values.tobytes() == read_golden.values.tobytes()
    assert reread.run_stamp == read_golden.run_stamp, (
        "the stamp is located by content, so its new text must still be found "
        "at the same offset")


def test_read_t01_rejects_a_foreign_file(tmp_path):
    """A file that is not an ``ITTYP==3`` T01 must fail loudly.

    Both record markers are checked, so a truncated or foreign file raises
    instead of yielding plausible garbage.
    """
    from tools.compare_t01 import T01FormatError, read_t01

    blob = GOLDEN_T01.read_bytes()
    truncated = tmp_path / "T01_truncated"
    truncated.write_bytes(blob[:len(blob) // 2])
    with pytest.raises(T01FormatError):
        read_t01(truncated)

    corrupt = bytearray(blob)
    corrupt[0:4] = (999999).to_bytes(4, "big")      # break the first marker
    broken = tmp_path / "T01_badmarker"
    broken.write_bytes(bytes(corrupt))
    with pytest.raises(T01FormatError):
        read_t01(broken)

    empty = tmp_path / "T01_empty"
    empty.write_bytes(b"")
    with pytest.raises(T01FormatError):
        read_t01(empty)


# ---------------------------------------------------------------------------
# Cross-check 3: a second producer -- the port's ASCII T01 CSV
# ---------------------------------------------------------------------------

def test_binary_reader_agrees_with_the_port_csv(read_golden, port_t01):
    """Two independent producers of the same physics, one binary, one ASCII.

    The port runs the golden deck and writes ``TENSILET01.csv``; this scores the
    binary T01 against it.  A reader that mis-walked the records, decoded the
    wrong width, or mistook the run stamp for data would produce numbers that
    are not the port's.

    The channel names line up through ``compare_t01.PORT_ALIASES`` -- the same
    pairs as ``tools/validate_vs_fortran.py:1036-1046`` (``GLOBAL_MAP``) in the
    opposite direction: that table maps a Fortran CSV column onto a port
    column, this one renames a port column onto the upstream short name so the
    two series can be matched by name.  Port columns with **no** upstream
    counterpart (``EN``, ``DE``, ``ERR%`` -- damping dissipation and the
    energy-error percentage, none of which is one of the 23 global channels)
    are left alone on purpose and come out ``NODATA``.

    The tolerance is the program's own: ``MATCH_RMS`` for every channel that
    carries signal, and an absolute bound for the channels that are numerical
    noise on *both* sides (a transverse momentum of ~1e-16 against an axial
    momentum of ~2e-4 is round-off, not physics -- the significance rule
    ``tools/validate_vs_fortran.py:1111-1120,1139-1146`` already documents).
    """
    from tools import compare_t01 as C

    assert read_golden.n_steps == port_t01.times.size == 100, (
        "both producers must have emitted the same number of /TH samples; the "
        "port writing a short series is a PORT bug (see the ROLLING deck "
        "finding), not a reader artefact")

    result = C.score(read_golden, port_t01)

    shared = {name: result.per_channel[name] for name in result.per_channel
              if result.per_channel[name].verdict != "NODATA"}
    assert set(shared) == {"IE", "KE", "HE", "CE", "EFW", "MASS", "XMOM",
                           "YMOM", "ZMOM", "P1_1", "P1_2"}, sorted(
                               result.per_channel)

    def group_of(name):
        if name.startswith("P"):
            return "energy"
        if "MOM" in name:
            return "momentum"
        if name == "MASS":
            return "mass"
        return "energy"

    dominant = {}
    for name, channel in shared.items():
        g = group_of(name)
        dominant[g] = max(dominant.get(g, 0.0),
                          abs(float(_column(read_golden, name).max())))

    significant, noise = {}, {}
    for name, channel in shared.items():
        scale = abs(float(_column(read_golden, name).max()))
        (noise if scale < 1e-9 * dominant[group_of(name)]
         else significant)[name] = channel

    assert set(significant) == {"IE", "KE", "HE", "EFW", "MASS", "XMOM",
                                "P1_1", "P1_2"}, sorted(noise)
    assert set(noise) == {"CE", "YMOM", "ZMOM"}, sorted(significant)
    # CE is identically zero on BOTH sides here (this deck has no contact), so
    # it lands in the noise class through the all-zero reference path -- and the
    # absolute bound is what says the port has no spurious contact energy.
    assert float(_column(read_golden, "CE").max()) == 0.0
    for name, channel in sorted(significant.items()):
        assert channel.rel_rms <= C.MATCH_RMS, (
            f"{name}: rel_rms {channel.rel_rms} exceeds MATCH_RMS "
            f"{C.MATCH_RMS} (n={channel.n})")
        assert channel.verdict == "MATCH"
    for name, channel in sorted(noise.items()):
        assert channel.max_abs <= 1e-9 * dominant[group_of(name)], (
            f"{name} is numerical noise on both sides, so the absolute "
            f"deviation must be too: {channel.max_abs}")

    # the static channel must agree to float32 storage, not merely to 5 %
    assert significant["MASS"].rel_rms < 1e-6, (
        "MASS is a constant of the model (no /DT/NODA/CST in this deck): any "
        "relative deviation above float32 rounding means the reader is "
        "misaligned")

    # the port's own channels with no upstream counterpart are NODATA, and they
    # do not drag `worst` down
    for port_only in ("EN", "DE", "ERR%"):
        assert result.per_channel[port_only].verdict == "NODATA", port_only
    assert result.worst.verdict == "MATCH"
    # both producers wrote 100 samples, but the port's last one (t = 0.1980918)
    # is past the Fortran's last one (t = 0.1980658), so the overlap window --
    # [0, min(t_last)], the M41 rule -- holds 99 of them.
    assert result.worst.n == 99, result.worst
    assert {c.n for c in shared.values()} == {99}, (
        "every compared channel must use the same sample window")


def _column(t01, name):
    return t01.values[:, t01.channels.index(name)]


def test_the_verdict_does_not_depend_on_which_grid_is_chosen(read_golden,
                                                            port_t01):
    """The M41 semantics (port grid) and the reference grid must agree here.

    ``parity_m41.json`` records the port grid; the reference grid is the other
    defensible choice.  The two grids of this deck are 100 samples each and one
    ends ~1.3e-4 s later than the other, so the *verdict* must be the same under
    either -- otherwise the choice of grid would be silently deciding parity.
    """
    from tools import compare_t01 as C

    on_port_grid = C.score(read_golden, port_t01).per_channel
    trend = min(read_golden.times[-1], port_t01.times[-1])
    mask = read_golden.times <= trend
    on_ref_grid = C.score(
        C.T01(channels=read_golden.channels, times=read_golden.times[mask],
              values=read_golden.values[mask], scaling=[]),
        port_t01).per_channel
    for name, channel in on_port_grid.items():
        if channel.verdict == "NODATA":
            continue
        assert on_ref_grid[name].verdict == channel.verdict, name
        assert on_ref_grid[name].rel_rms == pytest.approx(channel.rel_rms,
                                                          rel=0.05), name


# ---------------------------------------------------------------------------
# The brief's own cross-check, wired to run when it becomes possible
# ---------------------------------------------------------------------------

def test_cross_check_against_th_to_csv_when_available(tmp_path):
    """Upstream's converter, when it exists, must agree channel for channel.

    ``tools/th_to_csv/README.md:1-7`` says the converter's source moved to the
    separate ``OpenRadioss/Tools`` repository, which this box cannot reach
    (``git ls-remote https://github.com/OpenRadioss/Tools.git`` ->
    ``Repository not found``), and neither ``starter`` nor ``engine`` builds it.
    So this test SKIPS here -- loudly, naming every location that was tried --
    and turns itself into a real cross-check the moment the binary is installed
    (``OR_TH_TO_CSV``, or ``$OR_ROOT/{bin,exec}/th_to_csv_*``).
    """
    from tools import compare_t01 as C
    from tools.validate_vs_fortran import oracle_paths, read_csv_columns

    resolved = oracle_paths()
    converter = resolved.get("th_to_csv")
    if not converter:
        pytest.skip(
            "th_to_csv is not installed (upstream's own T01->CSV converter; "
            "its source lives in a separate repository, tools/th_to_csv/"
            "README.md:1-7). Places tried, from validate_vs_fortran."
            "oracle_paths(): env OR_TH_TO_CSV and $OR_ROOT/{bin,exec}/"
            "th_to_csv_linux64_gf|th_to_csv_win64.exe. The three independent "
            "cross-checks that replace it are "
            "test_layout_constants_match_the_cited_upstream_source, "
            "test_read_t01_sample_count_and_maxima_match_the_golden_record and "
            "test_binary_reader_agrees_with_the_port_csv.")

    work = tmp_path / "csv"
    work.mkdir()
    shutil.copy2(GOLDEN_T01, work / GOLDEN_T01.name)
    env = dict(os.environ)
    proc = subprocess.run([converter, GOLDEN_T01.name], cwd=work, env=env,
                          capture_output=True, text=True, errors="replace",
                          timeout=300)
    produced = work / (GOLDEN_T01.name + ".csv")
    assert proc.returncode == 0 and produced.is_file(), (
        f"th_to_csv failed on the golden T01 (rc={proc.returncode}):\n"
        f"{(proc.stdout + proc.stderr)[-2000:]}\n"
        f"{sorted(p.name for p in work.iterdir())}")

    header, table = read_csv_columns(str(produced))
    mine = C.read_t01(GOLDEN_T01)
    # column 0 is the time, then one column per channel, in file order
    assert table.shape == (mine.times.size, len(mine.channels) + 1), (
        f"th_to_csv produced {table.shape}, the binary reader decoded "
        f"({mine.times.size}, {len(mine.channels)}) + time")
    assert np.allclose(table[:, 0], mine.times, rtol=0, atol=1e-9), (
        "the time column disagrees -- the reader's record walk is off")
    for column in range(1, table.shape[1]):
        assert np.allclose(table[:, column], mine.values[:, column - 1],
                           rtol=1e-6, atol=1e-9), (
            f"column {column} ({header[column] if column < len(header) else '?'}"
            f") disagrees with the binary reader")