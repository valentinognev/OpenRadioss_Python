"""Gate for the parity-comparison-route restoration (c43 and the BT family).

The comparison route was closed for two independently confirmed causes,
both reproduced here as failing tests:

1. ``th_to_csv`` is not installed in the oracle prefix
   (``$OR_SRC/tools/th_to_csv/`` ships a README only; the converter lives in
   the separate ``OpenRadioss/Tools`` repository), so the harness's CSV route
   degrades to ``FORTRAN-FAIL(th2csv-missing)`` even though the engine wrote
   an admissible T01.
2. ``tools.compare_t01.read_t01`` refuses the decks that matter: the c43
   deck's binary T01 is 7116 records = 1186 steps of 6 while the shared walk
   (``tools.oracle.oracle_selftest.parse_t01``) fixes the per-step stride at
   4.

Both solvers still run; only the comparison is missing.  This file pins the
restored route:

* the converter resolves against the oracle prefix and renders the committed
  golden T01 to the same numbers the binary reader decodes;
* the reader decodes the c43 stride-6 T01 (stride 4 keeps working -- the
  golden tests pin that);
* the stored c43 reference row is re-derived through the restored route
  instead of quoted.

Upstream Fortran origins (``$OR_SRC`` = the read-only OpenCourant tree):
the record layout is the same ``hist1.F`` / ``hist2.F`` / ``wrtdes.F`` /
``write_routines.c`` / ``ieee.cpp`` set ``tools/compare_t01.py`` cites; the
converter renders exactly the channel set ``read_t01`` names so
``tools/validate_vs_fortran.py`` ``GLOBAL_MAP`` resolves on its output.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

GOLDEN_T01 = REPO / "tests" / "data" / "oracle_smoke" / "TENSILET01"
PARITY_M41 = REPO / "tools" / "validation_data" / "parity_m41.json"
C43_DIR = (REPO / "tests" / "data" / "rd_decks" / "rd_e" / "RD-E-1000_Bending"
           / "10_Bending" / "BT" / "BT_type3" / "Sf_0.9")
C45_DIR = (REPO / "tests" / "data" / "rd_decks" / "rd_e" / "RD-E-1000_Bending"
           / "10_Bending" / "BT" / "BT_type4" / "Sf_0.9")

#: The stored c43 reference row this task re-derives rather than quotes
#: (``parity_m41.json``; the coverage is read from the row, not repeated
#: here, because the row's window is what the re-derivation must honor).
STORED_C43_CASE = "c43_E1000_Bending_BT_BT_type3_Sf_0"
STORED_C43_CLASS = "DEVIATION"
STORED_C43_WORST = "HE"
STORED_C43_MAX_REL_RMS = 0.2565737863252257

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _require_deck(directory):
    if not (directory / "ROLLING_0000.rad").is_file():
        pytest.skip(f"deck not in the vendored corpus: {directory}")


def _require_c43_deck():
    _require_deck(C43_DIR)


def _stored_c43_row():
    """The stored c43 row: case id, class, worst, value and its window."""
    import json as _json

    record = _json.loads(PARITY_M41.read_text())
    for row in record["results"]:
        if row["case"] == STORED_C43_CASE:
            return row
    raise AssertionError(
        f"{STORED_C43_CASE} is not in parity_m41.json results")


def _require_oracle():
    from tools.oracle.oracle_selftest import resolve_oracle

    try:
        starter, engine, _ = resolve_oracle()
    except Exception as exc:  # noqa: BLE001 - configuration only
        pytest.skip(f"oracle not resolvable ({exc})")
    if not (starter.is_file() and engine.is_file()):
        pytest.skip("oracle binaries are not built")
    return starter, engine


@pytest.fixture(scope="module")
def c43_fortran_t01(tmp_path_factory):
    """The c43 Fortran T01: 1186 steps of 6 records, live oracle run."""
    _require_c43_deck()
    _require_oracle()
    from tools.oracle.oracle_selftest import run_reference

    work = tmp_path_factory.mktemp("c43_route")
    run_reference("ROLLING", workdir=work, deck_root=C43_DIR, timeout=3600)
    produced = work / "ROLLINGT01"
    assert produced.is_file(), sorted(p.name for p in work.iterdir())
    return produced


@pytest.fixture(scope="function")
def c43_port_csv(tmp_path_factory, monkeypatch):
    """The c43 port T01 CSV, pinned to the numpy backend for comparisons."""
    _require_c43_deck()
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    import tempfile

    work = pathlib.Path(tempfile.mkdtemp(prefix="c43_route_port_"))
    try:
        for deck in ("ROLLING_0000.rad", "ROLLING_0001.rad"):
            shutil.copy2(C43_DIR / deck, work / deck)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        for module, deck in (("pyradioss.starter", "ROLLING_0000.rad"),
                             ("pyradioss.engine", "ROLLING_0001.rad")):
            proc = subprocess.run(
                [sys.executable, "-m", module, "-i", deck], cwd=work, env=env,
                capture_output=True, text=True, errors="replace", timeout=1200)
            assert proc.returncode == 0, (
                f"the port failed on {deck} (rc={proc.returncode}):\n"
                f"{(proc.stdout + proc.stderr)[-2000:]}")
        csv = work / "ROLLINGT01.csv"
        assert csv.is_file(), (
            f"the port wrote no T01 CSV: "
            f"{sorted(p.name for p in work.iterdir())}")
        yield csv
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _require_oracle_prefix():
    """Skip (loudly) on a box with no oracle prefix at all.

    Without ``$OR_ROOT`` there is nowhere a converter could be installed
    against; with one, the restoration requires it to be there.  So this
    skips where the cross-check test skips, and fails where the route
    should have been restored but was not.
    """
    from pyradioss import paths

    try:
        paths.or_root()
    except Exception as exc:  # noqa: BLE001 - configuration only
        pytest.skip(f"no oracle prefix resolvable ({exc})")


def test_converter_is_installed_against_the_oracle_prefix():
    """``oracle_paths()["th_to_csv"]`` resolves to an executable file.

    Before the restoration this is ``None`` (``$OR_ROOT/bin`` holds only the
    two solvers); the harness then reports ``FORTRAN-FAIL(th2csv-missing)``
    for a run whose T01 is admissible evidence.
    """
    _require_oracle_prefix()
    from tools.validate_vs_fortran import oracle_paths

    converter = oracle_paths().get("th_to_csv")
    assert converter is not None, (
        "no th_to_csv converter is installed: $OR_SRC/tools/th_to_csv/ "
        "ships README.md only and $OR_ROOT/{bin,exec}/th_to_csv_* is absent")
    assert os.path.isfile(converter) and os.access(converter, os.X_OK), (
        f"th_to_csv resolved to {converter!r}, which is not executable")


def test_converter_renders_the_golden_t01_like_the_binary_reader(tmp_path):
    """The installed converter and ``read_t01`` decode the same bytes.

    Runs ``th_to_csv <T01>`` the way :func:`th_to_csv_argv` invokes it (one
    positional argument, the T01 basename, in the T01's directory) and
    requires ``<T01>.csv`` with the binary reader's channel set and values.
    """
    _require_oracle_prefix()
    from tools.compare_t01 import read_t01
    from tools.validate_vs_fortran import oracle_paths, read_csv_columns

    converter = oracle_paths().get("th_to_csv")
    assert converter, "no th_to_csv converter is installed"
    work = tmp_path / "csv"
    work.mkdir()
    shutil.copy2(GOLDEN_T01, work / GOLDEN_T01.name)
    proc = subprocess.run([converter, GOLDEN_T01.name], cwd=work,
                          capture_output=True, text=True, errors="replace",
                          timeout=300)
    produced = work / (GOLDEN_T01.name + ".csv")
    assert proc.returncode == 0 and produced.is_file(), (
        f"th_to_csv failed on the golden T01 (rc={proc.returncode}):\n"
        f"{(proc.stdout + proc.stderr)[-2000:]}\n"
        f"{sorted(p.name for p in work.iterdir())}")
    header, table = read_csv_columns(str(produced))
    mine = read_t01(GOLDEN_T01)
    assert table.shape == (mine.times.size, len(mine.channels) + 1), (
        f"th_to_csv produced {table.shape}, the binary reader decoded "
        f"({mine.times.size}, {len(mine.channels)}) + time")
    assert np.allclose(table[:, 0], mine.times, rtol=0, atol=1e-9), (
        "the time column disagrees -- the converter's record walk is off")
    for column in range(1, table.shape[1]):
        assert np.allclose(table[:, column], mine.values[:, column - 1],
                           rtol=1e-6, atol=1e-9), (
            f"column {column} "
            f"({header[column] if column < len(header) else '?'}) disagrees "
            f"with the binary reader")


def test_read_t01_decodes_the_c43_stride_6_t01(c43_fortran_t01):
    """Stride 6 decodes: 1186 steps, 23 globals plus the header's parts.

    The deck's ``/TH`` requests (``/TH/SHEL``, ``/TH/PART``, ``/TH/RBODY``,
    ``/TH/NODE``) write three per-group records per step instead of one, so
    the per-step block is six records.  The shared walk derives that stride
    from the header instead of fixing it at four.
    """
    from tools.compare_t01 import GLOBAL_CHANNEL_NAMES, read_t01
    from tools.oracle.oracle_selftest import parse_t01, t01_records

    blob = c43_fortran_t01.read_bytes()
    records = t01_records(blob)
    parsed = parse_t01(blob)
    assert len(records) == parsed["header_records"] + 7116, (
        f"c43 carries 7116 data records after a "
        f"{parsed['header_records']}-record header, found {len(records)}")
    assert parsed["n_steps"] == 1186, parsed["n_steps"]
    assert parsed["stride_records"] == 6, parsed["stride_records"]
    assert parsed["channels_per_step"]["global"] == 23
    assert parsed["channels_per_step"]["part"] == 9, (
        parsed["channels_per_step"])

    got = read_t01(c43_fortran_t01)
    assert got.n_steps == 1186
    assert got.channels[:23] == list(GLOBAL_CHANNEL_NAMES)
    assert len(got.channels) == 23 + 9, got.channels
    assert np.all(np.diff(got.times) >= 0), "sample times must not go backwards"
    assert got.times[0] == 0.0
    assert np.all(np.isfinite(got.values))


def test_c43_parity_is_rederived_through_the_restored_route(
        c43_fortran_t01, c43_port_csv, tmp_path):
    """The stored c43 row, measured here instead of quoted.

    Three pins, each honest about what the BT decks can discriminate:

    1. the installed converter and the binary reader decode the same c43
       bytes (exact round-trip: ``.17g`` carries float32 losslessly);
    2. the harness CSV route (the M41 semantics) over the full window is
       ``DEVIATION`` on ``HE`` -- and truncated to the stored row's own
       window (its ``port_time_coverage``, 0.9558: the M41 port run aborted
       there on its energy guard) it reproduces the stored row's value, so
       the quoted number is re-derived, not inherited;
    3. the binary fallback route completes on the same pair and agrees in
       class (its denominator is ``max(|ref|)`` where the CSV route uses
       ``max(|fort|, |port|)``, so the two need not agree in value -- on a
       deck whose Fortran hourglass energy is ~zero they cannot -- and the
       test pins the class, not a digit).
    """
    from tools import compare_t01 as C
    from tools.validate_vs_fortran import (
        compare_binary_t01, compare_channels, oracle_paths, read_csv_columns)

    # -- 1. one decode, two renderings --
    _require_oracle_prefix()
    converter = oracle_paths().get("th_to_csv")
    assert converter, "no th_to_csv converter is installed"
    work = tmp_path / "c43csv"
    work.mkdir()
    shutil.copy2(c43_fortran_t01, work / "ROLLINGT01")
    proc = subprocess.run([converter, "ROLLINGT01"], cwd=work,
                          capture_output=True, text=True, errors="replace",
                          timeout=300)
    rendered = work / "ROLLINGT01.csv"
    assert proc.returncode == 0 and rendered.is_file(), (
        f"th_to_csv failed on the c43 T01 (rc={proc.returncode}):\n"
        f"{(proc.stdout + proc.stderr)[-2000:]}")
    header, table = read_csv_columns(str(rendered))
    mine = C.read_t01(c43_fortran_t01)
    assert table.shape == (mine.times.size, len(mine.channels) + 1)
    assert np.allclose(table[:, 0], mine.times, rtol=0, atol=1e-9)
    for column in range(1, table.shape[1]):
        assert np.allclose(table[:, column], mine.values[:, column - 1],
                           rtol=1e-6, atol=1e-9), (
            f"c43 column {column} "
            f"({header[column] if column < len(header) else '?'}) disagrees")

    # -- 2. the CSV route, full window and the stored row's window --
    stored = _stored_c43_row()
    assert stored["class"] == STORED_C43_CLASS
    assert stored["max_rel_rms"] == STORED_C43_MAX_REL_RMS
    fh, fd = read_csv_columns(str(rendered))
    ph, pd = read_csv_columns(str(c43_port_csv))
    full = compare_channels(fh, fd, ph, pd)
    significant = [c for c in full if c.get("significant", True)]
    assert significant, "nothing comparable on the full window"
    top = max(significant, key=lambda c: c["rel_rms"])
    assert top["channel"] == STORED_C43_WORST, (
        f"full-window worst channel {top['channel']} is not the stored "
        f"{STORED_C43_WORST}")
    assert (("MATCH" if top["rel_rms"] <= C.MATCH_RMS else "DEVIATION")
            == STORED_C43_CLASS)

    coverage = stored["port_time_coverage"]
    assert 0.0 < coverage <= 1.0, coverage
    cutoff = coverage * fd[-1, 0]
    truncated = compare_channels(fh, fd[fd[:, 0] <= cutoff + 1e-12],
                                 ph, pd[pd[:, 0] <= cutoff + 1e-12])
    significant = [c for c in truncated if c.get("significant", True)]
    top = max(significant, key=lambda c: c["rel_rms"])
    assert top["channel"] == STORED_C43_WORST, (
        f"stored-window worst channel {top['channel']} is not the stored "
        f"{STORED_C43_WORST}")
    assert top["rel_rms"] == pytest.approx(STORED_C43_MAX_REL_RMS, rel=0.02), (
        "the stored c43 row is not reproduced under its own window -- the "
        "quoted number no longer describes this deck")

    # -- 3. the binary fallback route completes and agrees in class --
    rows, summary = compare_binary_t01(str(c43_fortran_t01), str(c43_port_csv))
    assert summary["worst_verdict"] != "NODATA", (
        f"nothing comparable: {summary['n_compared']} of "
        f"{summary['n_channels']} channels compared")
    assert summary["n_signal"] >= 1
    assert np.isfinite(summary["worst_rel_rms"])
    assert (("MATCH" if summary["worst_rel_rms"] <= C.MATCH_RMS
             else "DEVIATION") == STORED_C43_CLASS)


def test_read_t01_decodes_the_c45_stride_6_t01(tmp_path_factory):
    """Stride 6 on the second BT deck: 1186 steps, 23 globals, 9 parts.

    c45 carries the same ``/TH`` family shape as c43 (``NN = [2, 6, 1]``)
    with its own bytes; one live Fortran run pins that the header-derived
    walk is not fitted to a single file.  The port side of c45 aborts on its
    pre-existing energy guard, so no port CSV is needed here -- only the
    Fortran T01, which the oracle writes before anything aborts.
    """
    _require_deck(C45_DIR)
    _require_oracle()

    from tools.compare_t01 import read_t01
    from tools.oracle.oracle_selftest import (
        channel_maxima, parse_t01, run_reference)

    work = tmp_path_factory.mktemp("c45_route")
    run_reference("ROLLING", workdir=work, deck_root=C45_DIR, timeout=3600)
    produced = work / "ROLLINGT01"
    assert produced.is_file(), sorted(p.name for p in work.iterdir())

    parsed = parse_t01(produced.read_bytes())
    assert parsed["n_steps"] == 1186, parsed["n_steps"]
    assert parsed["stride_records"] == 6, parsed["stride_records"]
    assert parsed["group_nn"] == [2, 6, 1], parsed["group_nn"]
    # the multi-member group path must not raise here either
    maxima = channel_maxima(parsed)
    assert len(maxima["th_group"]) == 2 * 8 + 6 * 11 + 9, (
        len(maxima["th_group"]))

    got = read_t01(produced)
    assert got.n_steps == 1186
    assert len(got.channels) == 23 + 9, got.channels
    assert np.all(np.diff(got.times) >= 0)
    assert np.all(np.isfinite(got.values))
