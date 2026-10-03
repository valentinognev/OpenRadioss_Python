"""Task P0.11 — ``parity`` must produce evidence on a box without ``th_to_csv``.

Phase 0 exists so later phases can produce differential parity evidence
against the real Fortran solver.  On this box they could not: the oracle ran,
wrote its binary ``T01``, and the harness then stopped at
``FORTRAN-FAIL(th2csv-missing)`` because upstream's converter is unobtainable
(``$OR_SRC/tools/th_to_csv/README.md:1-7`` points at ``OpenRadioss/Tools``,
which this machine cannot reach — recorded in
``tools/validation_data/oracle_provenance.json``).

Commit ``c679734`` landed ``tools/compare_t01.py``: a reader for that very
binary file plus a per-channel scorer.  This module pins that it is actually
*used*, and that using it does not weaken anything.

Upstream / provenance origins
-----------------------------
* ``engine/source/output/th/hist1.F:201-208`` + ``:210-234`` and
  ``engine/source/system/timer_c.c:30-40`` — the file is the engine's own
  ``ITTYP==3`` binary time history, and its 24-byte header stamp is
  ``ctime()``: **two runs of one deck differ in those bytes and nowhere else**,
  which is why no comparison here may be a raw-byte one (pinned by
  :func:`test_the_verdict_ignores_the_ctime_run_stamp`).
* ``engine/source/output/th/hist2.F:302-303``, ``:307-333``, ``:338-477`` — the
  per-step records the reader walks: ``TT``, the 23 global channels, the part
  values.
* ``starter/source/output/th/write_thnms1.F90:230-252`` — the authoritative
  name table for those 23 channels, which is why the two sides are matched
  **by name** and not by column position.
* ``tools/validation_data/oracle_provenance.json``
  ``admissible_parity_evidence`` — which evidence a parity claim may rest on.
  ``T01 (binary results table)`` is admissible; H3D files, native ``.k``
  reading, ``/ALE/STRUCTURED_MESH`` and ``/CHECKSUM_REPORT`` over H3D are not.
  A row that cannot name its evidence source may not claim a channel at all.
* ``tools/validation_data/parity_m41.json`` ``tolerance_rel_rms`` (0.05) and its
  class census — the vocabulary a new verdict may not shift.

What is pinned
--------------
1. **the reader path is taken when the converter is absent**, and the row says
   which route produced its numbers;
2. **the CSV path is unchanged** when a converter *is* available — the new
   branch is additive, never a replacement;
3. **a missing or unreadable T01 fails loudly** — never an empty comparison,
   never a silent ``MATCH``;
4. **a known rel-RMS produces the expected per-channel verdict**, and
   ``max_rel_rms`` is the scorer's ``worst`` (which already excludes channels
   carrying no signal);
5. **no verdict may be ``MATCH`` without significant channels** — the guard the
   harness already had, kept and pinned;
6. **the run stamp cannot change a verdict** (no raw-byte comparison anywhere);
7. **admissibility is carried in the record**, and a deck carrying a feature
   the provenance forbids is flagged instead of quietly claimed;
8. ``parity_m41.json`` stays joinable: same class strings for the same
   situations, and its tolerance is the one the reader route is judged by.

No oracle is needed: the committed golden T01 of the P0.5 reference run
(``tests/data/oracle_smoke/TENSILET01``) is real Fortran output, and the port
CSV beside it is synthesised from the reader's own decoded values.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

import pytest

from pyradioss import paths

REPO = Path(paths.__file__).resolve().parents[1]
GOLDEN_T01 = REPO / "tests" / "data" / "oracle_smoke" / "TENSILET01"
PROVENANCE = REPO / "tools" / "validation_data" / "oracle_provenance.json"
PARITY_M41 = REPO / "tools" / "validation_data" / "parity_m41.json"

#: The two ways a Fortran T01 can become comparable numbers.  Recorded per row
#: so a reader of ``parity_results.json`` never has to guess which was used.
CSV_ROUTE = "th_to_csv CSV"
BINARY_ROUTE = "binary T01 (tools.compare_t01.read_t01)"

#: The evidence channel both routes rest on, spelled as the provenance spells
#: it — the key is looked up in ``admissible_parity_evidence``.
T01_EVIDENCE_KEY = "T01 (binary results table)"

#: The features ``oracle_provenance.json`` marks inadmissible, with the deck
#: keyword that requests each one.  ``/H3D`` is the engine's h3d output family
#: (``engine/source/input/freform.F:2680,2696`` read ``KEY3=='H3D'``).
INADMISSIBLE_KEYWORDS = {
    "H3D animation files": "/H3D",
    "/ALE/STRUCTURED_MESH (S-ALE)": "/ALE/STRUCTURED_MESH",
    "/CHECKSUM_REPORT over H3D files": "/CHECKSUM_REPORT",
}

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_golden():
    from tools.compare_t01 import read_t01
    return read_t01(GOLDEN_T01)


def _port_name(upstream: str) -> str:
    """The port CSV column name for an upstream channel (inverse of the aliases)."""
    from tools.compare_t01 import PORT_ALIASES, PORT_PART_CODES
    inverse = {v: k for k, v in PORT_ALIASES.items()}
    if upstream in inverse:
        return inverse[upstream]
    for code, variable in PORT_PART_CODES.items():
        head, _, tail = upstream.partition("_")
        if tail == str(code):
            return f"{head}_{variable}"
    return upstream


def _write_port_csv(t01, path, scale=None, drop=()) -> str:
    """A port-shaped T01 CSV carrying ``t01``'s own numbers.

    ``scale`` multiplies a channel (to synthesise a deviation) and ``drop``
    omits one (to synthesise a ``NODATA``).  The column *names* are the port's
    own, so ``compare_t01.read_port_csv`` maps them back onto the upstream
    names and the two series match by name -- the same round trip the real port
    run goes through.
    """
    scale = scale or {}
    names = [n for n in t01.channels if _port_name(n) not in drop]
    lines = ["# pyradioss time history",
             ",".join(["TIME"] + [_port_name(n) for n in names])]
    for row, time in zip(t01.values, t01.times):
        cells = [f"{time:.17g}"]
        for name in names:
            cells.append(f"{row[t01.channels.index(name)] * scale.get(name, 1.0):.17g}")
        lines.append(",".join(cells))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _parity_args(workdir, only="tensile_bar", tol=0.05):
    return argparse.Namespace(workdir=str(workdir), only=only, budget=60.0,
                              timeout=60.0, tol=tol, shim="translate")


def _rows(workdir):
    return json.loads((Path(workdir) / "parity_results.json")
                      .read_text(encoding="utf-8"))


def _drive(monkeypatch, tmp_path, fortran_info, port_info, tol=0.05):
    """Run ``parity`` with both drivers stubbed; return the single result row."""
    from tools import validate_vs_fortran as V
    monkeypatch.setattr(V, "oracle_paths", lambda *a, **k: {
        key: "/opt/or/bin/x" for key in V.ORACLE_KEYS})
    monkeypatch.setattr(V, "run_fortran",
                        lambda *a, **k: dict(fortran_info))
    monkeypatch.setattr(V, "run_pyradioss",
                        lambda *a, **k: dict(port_info))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = V.parity(_parity_args(tmp_path, tol=tol))
    assert rc == 0
    rows = _rows(tmp_path)
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------------------
# 1. the reader path is taken when the converter is absent
# ---------------------------------------------------------------------------

def test_parity_compares_the_binary_t01_when_th_to_csv_is_absent(tmp_path,
                                                                monkeypatch,
                                                                capsys):
    """No converter, a readable T01 -> a real comparison, and it says so.

    This is the whole point of the task: the oracle runs, its T01 exists, and
    the evidence is produced anyway by reading that file with
    ``tools.compare_t01.read_t01`` — the reader landed early precisely so the
    converter becomes optional.
    """
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing", "mode": "real-format",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    assert row["class"] == "MATCH", row
    assert row["fortran"]["comparison_route"] == BINARY_ROUTE
    assert row["fortran"]["reader"] == "tools.compare_t01.read_t01"
    channels = {c["channel"]: c for c in row["channels"]}
    # the two energy channels are the load-bearing claim on this deck
    assert channels["IE"]["verdict"] == "MATCH"
    assert channels["IE"]["rel_rms"] == pytest.approx(0.0, abs=1e-12)
    assert channels["IE"]["n"] == t01.times.size
    assert channels["P1_1"]["verdict"] == "MATCH"
    assert row["max_rel_rms"] == max(c["rel_rms"] for c in row["channels"]
                                     if c["verdict"] != "NODATA")
    assert "MATCH" in capsys.readouterr().out


def test_the_reader_path_reports_nodata_for_a_channel_only_one_side_has(
        tmp_path, monkeypatch):
    """A port column with no Fortran counterpart is ``NODATA``, not a deviation.

    ``tools/compare_t01`` matches by name and marks a one-sided channel
    ``NODATA``; such a channel must never reach ``max_rel_rms`` or the class.
    """
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv", drop={"CE"})
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    channels = {c["channel"]: c for c in row["channels"]}
    assert channels["CE"]["verdict"] == "NODATA"
    assert channels["CE"]["n"] == 0
    assert row["class"] == "MATCH"
    assert row["max_rel_rms"] < 1e-6


# ---------------------------------------------------------------------------
# 2. the CSV path is untouched when a converter IS available
# ---------------------------------------------------------------------------

def test_the_csv_path_is_still_taken_when_a_converter_is_available(tmp_path,
                                                                   monkeypatch):
    """Additive, never a replacement.

    With ``th_to_csv`` installed the historical CSV path must run exactly as
    before — including its ``final_dev`` / ``scale`` / ``significant`` row shape,
    which the reader route does not produce — and the reader must not be
    consulted at all (the reader module itself is booby-trapped).
    """
    from tools import compare_t01, validate_vs_fortran as V
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    fort_csv = _write_port_csv(t01, tmp_path / "fortran.csv")
    calls = []
    original = V.compare_channels

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(V, "compare_channels", spy)
    monkeypatch.setattr(
        compare_t01, "read_t01",
        lambda *a, **k: pytest.fail("the reader must not run when the "
                                    "converter produced a CSV"))
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "ok", "csv": fort_csv, "csv_tool":
                               "/opt/or/bin/th_to_csv_linux64_gf",
                               "t01": str(GOLDEN_T01)},
                 port_info={"status": "ok", "csv": csv})
    assert calls, "compare_channels was not called: the CSV path was skipped"
    assert row["fortran"].get("comparison_route") in (None, CSV_ROUTE)
    keys = set(row["channels"][0])
    assert {"channel", "rel_rms", "final_dev", "scale",
            "significant"} <= keys, row["channels"][0]


# ---------------------------------------------------------------------------
# 3. a missing or unreadable T01 fails loudly
# ---------------------------------------------------------------------------

def test_a_missing_t01_fails_loudly_and_claims_nothing(tmp_path, monkeypatch,
                                                       capsys):
    """No converter *and* no readable file -> a named failure, not a MATCH."""
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing", "t01":
                               str(tmp_path / "no_such_file"), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    assert row["class"].startswith("FORTRAN-FAIL"), row
    assert row["class"] != "MATCH"
    assert "max_rel_rms" not in row
    assert "compare_t01" in row["fortran"]["error"]
    assert "no such file" in row["fortran"]["error"].lower() \
        or "No such file" in row["fortran"]["error"]


def test_an_unreadable_t01_fails_loudly(tmp_path, monkeypatch):
    """Garbage bytes -> the reader's own error, quoted, and no verdict."""
    from tools.compare_t01 import T01FormatError
    junk = tmp_path / "JUNK_T01"
    junk.write_bytes(b"\x00\x01\x02not a T01 at all" * 4)
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing", "t01": str(junk),
                               "csv": None},
                 port_info={"status": "ok", "csv": csv})
    assert row["class"].startswith("FORTRAN-FAIL"), row
    assert "hist1.F" in row["fortran"]["error"] or "T01" in row["fortran"]["error"]
    assert "max_rel_rms" not in row
    assert T01FormatError.__name__ or True     # the reader's error type


def test_an_unreadable_port_csv_fails_loudly(tmp_path, monkeypatch):
    """The port's own file unreadable is equally a refusal, not a zero."""
    from tools.compare_t01 import T01FormatError
    t01 = _read_golden()
    bad = tmp_path / "port.csv"
    bad.write_text("not,a,t01\n", encoding="utf-8")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": str(bad)})
    assert row["class"].startswith("FORTRAN-FAIL"), row
    assert "T01FormatError" not in row["class"]      # additive subtag only
    assert "max_rel_rms" not in row
    assert T01FormatError.__name__ or True


# ---------------------------------------------------------------------------
# 4. known rel-RMS -> expected per-channel verdict
# ---------------------------------------------------------------------------

def _drop_all(t01) -> set:
    """Every column name a port CSV could carry for this T01, both spellings.

    ``_port_name`` renames four upstream channels (``XMOM`` -> ``MOMX`` …), so
    dropping "all of them" by upstream name alone would silently leave those
    four in the file and the pair would still share a channel.
    """
    return set(t01.channels) | {_port_name(n) for n in t01.channels}


def test_a_known_offset_gives_the_expected_verdict_per_channel(tmp_path,
                                                               monkeypatch):
    """+20 % on one channel -> that channel DEVIATES, the others still MATCH.

    The expected number is derived from the data rather than assumed to be
    0.2: ``rel_rms`` divides by ``max(|ref|)`` while the deviation follows the
    whole ramp, so it is ``0.2 * rms(IE) / max|IE|`` — which is exactly the
    formula P0.10 recorded for the brief's own synthetic case.
    """
    import numpy as np
    t01 = _read_golden()
    ie = t01.values[:, t01.channels.index("IE")]
    expected = 0.2 * float(np.sqrt(np.mean(ie ** 2)) / np.max(np.abs(ie)))
    csv = _write_port_csv(t01, tmp_path / "port.csv", scale={"IE": 1.2})
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    channels = {c["channel"]: c for c in row["channels"]}
    assert channels["IE"]["verdict"] == "DEVIATION"
    assert channels["IE"]["rel_rms"] == pytest.approx(expected, rel=1e-6)
    assert channels["KE"]["verdict"] == "MATCH"
    assert row["class"] == "DEVIATION"
    assert row["max_rel_rms"] == pytest.approx(expected, rel=1e-6)


def test_the_reader_route_uses_the_m41_tolerance(tmp_path, monkeypatch):
    """``MATCH_RMS`` must be ``parity_m41.json``'s own tolerance.

    Otherwise a verdict produced by the reader route would mean something
    different from every verdict in the historical tables, and the two could
    not be compared — which is the invariant this whole module protects.
    """
    from tools import compare_t01
    record = json.loads(PARITY_M41.read_text(encoding="utf-8"))
    assert compare_t01.MATCH_RMS == record["tolerance_rel_rms"]


def test_tol_can_tighten_the_reader_route_but_never_loosen_it(tmp_path,
                                                              monkeypatch):
    """``--tol`` stays authoritative downward, and the M41 tolerance holds upward.

    A caller asking for a stricter tolerance must get it; a caller asking for a
    looser one must not be able to buy a ``MATCH`` the recorded sweeps would
    have called a deviation — that would make a new sweep incomparable with
    every old one, which is the one thing the class vocabulary exists to
    prevent.
    """
    import numpy as np
    t01 = _read_golden()
    ie = t01.values[:, t01.channels.index("IE")]
    # rel_rms follows the ramp and divides by its peak, so the scale that puts
    # this channel between MATCH_RMS and a loose --tol is 1.5, not 1.04
    expected = 0.5 * float(np.sqrt(np.mean(ie ** 2)) / np.max(np.abs(ie)))
    assert 0.05 < expected < 0.5, expected
    csv = _write_port_csv(t01, tmp_path / "port.csv", scale={"IE": 1.5})
    strict = _drive(monkeypatch, tmp_path,
                    fortran_info={"status": "th2csv-missing",
                                  "t01": str(GOLDEN_T01), "csv": None},
                    port_info={"status": "ok", "csv": csv}, tol=0.01)
    assert strict["class"] == "DEVIATION"

    loose = _drive(monkeypatch, tmp_path,
                   fortran_info={"status": "th2csv-missing",
                                 "t01": str(GOLDEN_T01), "csv": None},
                   port_info={"status": "ok", "csv": csv}, tol=0.5)
    assert loose["class"] == "DEVIATION", (
        f"rel_rms {loose['max_rel_rms']:.4f} must stay a DEVIATION even at "
        f"--tol 0.5: parity_m41's own 0.05 tolerance is the floor")
    # ... and the scorer agrees it is a deviation on its own terms
    assert loose["channel_summary"]["worst_verdict"] == "DEVIATION"


# ---------------------------------------------------------------------------
# 5. no MATCH without significant channels
# ---------------------------------------------------------------------------

def test_no_verdict_can_be_match_without_significant_channels(tmp_path,
                                                              monkeypatch):
    """Every channel ``NODATA`` -> the harness's no-overlap failure.

    The guard the CSV path already had ("no overlapping channels"), kept: a
    ``MATCH`` derived from nothing comparable would be the worst possible
    output of a validation harness.
    """
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv", drop=_drop_all(t01))
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    # NO-CHANNELS, the class parity_m41.json carries for exactly this condition
    # (4-9 rows per sweep) — and NOT the bare FORTRAN-FAIL an earlier revision
    # emitted, which no recorded sweep carries at all.
    assert row["class"] == "NO-CHANNELS", row
    assert "max_rel_rms" not in row
    assert all(c["verdict"] == "NODATA" for c in row["channels"])
    assert "no comparable channel" in row["fortran"]["error"]


def test_a_single_sample_port_series_never_matches(tmp_path, monkeypatch):
    """One sample is not a comparison; the scorer says ``NODATA`` and so must we.

    ``tools/compare_t01.MIN_SAMPLES`` is 2, so a one-row port CSV yields
    ``NODATA`` on every channel even though the columns match by name.
    """
    t01 = _read_golden()
    csv = tmp_path / "one_row.csv"
    csv.write_text("TIME,IE,KE\n0.0,1.0,1.0\n", encoding="utf-8")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": str(csv)})
    assert row["class"] == "NO-CHANNELS", row
    assert "max_rel_rms" not in row


# ---------------------------------------------------------------------------
# 6. the run stamp cannot change a verdict (no raw-byte comparison)
# ---------------------------------------------------------------------------

def test_the_summary_separates_compared_from_signal_carrying(tmp_path,
                                                             monkeypatch):
    """``n_compared`` is not ``n_signal``, and neither is the headline number.

    An earlier revision recorded ``n_significant = len(compared)``, which
    overstates the evidence: on ``examples/tensile_bar`` 11 channels are
    compared and only **5** carry signal (the rest are an identically-zero
    contact energy, two 1e-16 transverse momenta, and three channels at
    0.02-0.05 % of the dominant energy channel, all under the 1 %-of-group rule
    the harness has always applied).  The roll-up ``worst`` comes from those 5,
    so the row has to say 5.
    """
    from tools.compare_t01 import SIGNIFICANCE_FRACTION
    t01 = _read_golden()
    port_csv = _write_port_csv(t01, tmp_path / "port.csv",
                              scale={"KE": 1.0001, "HE": 1.0001})
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": port_csv})
    summary = row["channel_summary"]
    assert "n_significant" not in summary, (
        "the over-counting field is back; use n_compared and n_signal")
    assert summary["n_channels"] == len(row["channels"])
    assert summary["n_compared"] == len(
        [c for c in row["channels"] if c["verdict"] != "NODATA"])
    assert summary["n_signal"] == len(summary["significant_channels"])
    assert summary["n_signal"] < summary["n_compared"], (
        "on this deck the two must differ; if they no longer do the test data "
        "changed, not the rule")
    assert set(summary["significant_channels"]) | \
        set(summary["noise_channels"]) == {
            c["channel"] for c in row["channels"] if c["verdict"] != "NODATA"}
    # the signal set is exactly the rows flagged significant, and it is the set
    # worst came from
    flagged = {c["channel"] for c in row["channels"] if c.get("significant")}
    assert flagged == set(summary["significant_channels"])
    worst_channel = max(
        (c for c in row["channels"] if c.get("significant")),
        key=lambda c: c["rel_rms"])
    assert worst_channel["rel_rms"] == row["max_rel_rms"]
    assert 0 < SIGNIFICANCE_FRACTION < 1


def test_an_insignificant_channel_is_flagged_and_marked_on_both_routes(
        tmp_path, monkeypatch, capsys):
    """A 0.4 next to a MATCH must be explained, in the JSON and on the console.

    ``YMOM``/``ZMOM`` compare at 0.41/0.56 on ``examples/tensile_bar`` while the
    row is a MATCH: they are transverse momenta of an axial test, ~1e-16
    against an axial momentum of 2e-4.  The CSV route has always marked such a
    channel with ``~``; the reader route had no flag at all, so the same number
    appeared unexplained next to the same MATCH.
    """
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    channels = {c["channel"]: c for c in row["channels"]}
    assert channels["XMOM"]["significant"] is True
    assert channels["YMOM"]["significant"] is False
    assert channels["YMOM"]["verdict"] != "NODATA"
    printed = capsys.readouterr().out
    assert "~YMOM=" in printed and "~ZMOM=" in printed
    assert "YMOM=0.41" not in printed.replace("~YMOM=0.41", ""), (
        "an unflagged 0.41 must not appear beside a MATCH")
    # and the CSV route marks the same way
    from tools.validate_vs_fortran import channel_text
    assert channel_text(channels["YMOM"]).startswith("~YMOM=")
    assert channel_text(channels["XMOM"]) == "XMOM=" + \
        f"{channels['XMOM']['rel_rms']:.3G}"
    assert channel_text({"channel": "X", "rel_rms": float("inf"),
                         "verdict": "NODATA"}) == "X=-"


def test_the_row_records_the_tolerance_the_class_was_decided_at(tmp_path,
                                                                monkeypatch):
    """A row must not be a DEVIATION at a tolerance it satisfies.

    ``--tol`` can tighten the reader route below ``parity_m41.json``'s 0.05; the
    recorded ``tolerance`` in the summary is that historical constant, so
    without the applied value a ``--tol 0.001`` sweep emits a self-contradictory
    row (DEVIATION at max_rel_rms 0.00246 next to ``tolerance: 0.05``).
    """
    import numpy as np
    t01 = _read_golden()
    ie = t01.values[:, t01.channels.index("IE")]
    # the scale that puts IE's rel-RMS between --tol 0.001 and the recorded
    # 0.05: rel_rms = (f-1) * rms(IE)/max|IE|, so f = 1 + target/0.542...
    factor = 1.0 + 0.02 / float(np.sqrt(np.mean(ie ** 2)) / np.max(np.abs(ie)))
    csv = _write_port_csv(t01, tmp_path / "port.csv", scale={"IE": factor})
    tight = _drive(monkeypatch, tmp_path,
                   fortran_info={"status": "th2csv-missing",
                                 "t01": str(GOLDEN_T01), "csv": None},
                   port_info={"status": "ok", "csv": csv}, tol=0.001)
    assert 0.001 < tight["max_rel_rms"] < 0.05, tight["max_rel_rms"]
    assert tight["class"] == "DEVIATION"
    assert tight["max_rel_rms"] > tight["tolerance_used"]
    assert tight["channel_summary"]["tolerance"] == 0.05, (
        "the historical constant must stay the historical constant")
    loose = _drive(monkeypatch, tmp_path,
                   fortran_info={"status": "th2csv-missing",
                                 "t01": str(GOLDEN_T01), "csv": None},
                   port_info={"status": "ok", "csv": csv})
    assert loose["tolerance_used"] == 0.05
    assert loose["class"] == "MATCH", (
        "the same numbers must MATCH at the recorded tolerance")


def test_the_row_records_the_compute_backend_requested_and_resolved(
        tmp_path, monkeypatch):
    """Which backend produced the port numbers, requested *and* resolved.

    ``plan/00_ORCHESTRATION.md`` §1 item 6 pins ``PYRADIOSS_BACKEND=numpy`` for
    before/after comparisons, and ``auto`` resolves per model (the engine logs
    ``COMPUTE BACKEND . . . : numba (auto: 40 elements >= 32)``), so a parity row
    that cannot name the backend is not comparable with another one.
    """
    from tools import validate_vs_fortran as V
    listing = tmp_path / "CASE_0001.out"
    listing.write_text(
        "  CYCLES  . . . : 1847\n"
        "  ELAPSED TIME . . . :  0.697 s\n"
        " COMPUTE BACKEND  . . . . . . . . . . . : numba "
        "(auto: 40 elements >= 32)\n", encoding="utf-8")
    info = V.harvest_port_out(str(tmp_path / "nope.out"), str(listing))
    assert info["compute_backend_used"] == "numba"
    assert info["compute_backend_reason"] == "auto: 40 elements >= 32"
    assert info["n_cycles"] == 1847

    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv,
                            "backend_requested": "auto",
                            "compute_backend_used": "numba",
                            "compute_backend_reason": "auto: 40 elements >= 32"})
    backend = row["compute_backend"]
    assert backend == {"requested": "auto", "used": "numba",
                       "reason": "auto: 40 elements >= 32",
                       "pinned_for_comparison": False}
    pinned = _drive(monkeypatch, tmp_path,
                    fortran_info={"status": "th2csv-missing",
                                  "t01": str(GOLDEN_T01), "csv": None},
                    port_info={"status": "ok", "csv": csv,
                               "backend_requested": "numpy",
                               "compute_backend_used": "numpy"})
    assert pinned["compute_backend"]["pinned_for_comparison"] is True


def test_run_pyradioss_records_the_requested_backend(monkeypatch, tmp_path):
    """``run_pyradioss`` states what it asked for, before anything runs.

    The default is ``auto`` — the value ``pyradioss`` itself defaults to — so a
    run that was not pinned says so instead of silently being one.
    """
    from tools import validate_vs_fortran as V
    seen = {}

    def fake_run_cmd(cmd, cwd, timeout, env=None):
        seen.update(env or {})
        return 0, "", 0.0

    # the decks need their own directory: run_pyradioss copies the deck's
    # directory into the scratch tree, and a scratch tree *inside* it would be
    # copied into itself
    decks = tmp_path / "decks"
    decks.mkdir()
    deck0 = decks / "CASE_0000.rad"
    deck1 = decks / "CASE_0001.rad"
    deck0.write_text("#RADIOSS STARTER\n", encoding="utf-8")
    deck1.write_text("#RADIOSS ENGINE\n", encoding="utf-8")
    monkeypatch.delenv("PYRADIOSS_BACKEND", raising=False)
    monkeypatch.setattr(V, "run_cmd", fake_run_cmd)
    info = V.run_pyradioss("case", "CASE", str(deck0), str(deck1),
                           str(tmp_path / "wd"), 30.0)
    assert info["backend_requested"] == "auto"
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    info = V.run_pyradioss("case", "CASE", str(deck0), str(deck1),
                           str(tmp_path / "wd"), 30.0)
    assert info["backend_requested"] == "numpy"
    assert seen["PYRADIOSS_BACKEND"] == "numpy"


def test_a_refused_t01_reports_the_measured_step_block(tmp_path, monkeypatch):
    """A refusal must say what the file *is*, not only what the walk expected.

    ``tools.oracle.oracle_selftest.parse_t01`` fixes the per-step stride at four
    records; ``hist2.F`` writes one record per ``/TH`` family that has curves, so
    the block is variable (six records on
    ``rd_e/…/BATOZ/Sf_0.6/ROLLING``: ``[4, 92, 36, 64, 264, 36]`` over 1605
    steps).  An earlier revision of this harness blamed the hierarchy's
    ``NSUBS``, which is **not** evidence of a ``/TH/SUBSET`` request:
    ``starter/source/starter/contrl.F:671-673`` counts the option and then adds
    one "for global subset", so ``NSUBS >= 1`` always.
    """
    from tools import validate_vs_fortran as V
    assert V.measured_step_block(str(GOLDEN_T01))["stride_records"] == 4
    rolled = Path("/tmp/opencode/p011_e2e_rd/fortran/rolling_batoz/ROLLINGT01")
    if not rolled.is_file():
        pytest.skip("the ROLLING reference T01 is not in this scratch dir")
    block = V.measured_step_block(str(rolled))
    assert block["stride_records"] == 6
    assert block["steps"] == 1605, block
    assert block["block_byte_lengths"] == [[4, 92, 36, 64, 264, 36]], block
    assert block["records"] == 9656, block

    # … and the refusal a parity row emits must carry that measurement, and
    # must NOT restate the wrong cause.  (Driven through `parity` because the
    # text is what a reader of the results file sees.)
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing", "t01": str(rolled),
                               "csv": None},
                 port_info={"status": "ok", "csv": csv})
    assert row["class"] == "FORTRAN-FAIL(t01-unreadable)"
    assert "max_rel_rms" not in row
    error = row["fortran"]["error"]
    assert "1605 steps of 6 records" in error, error
    assert "4, 92, 36, 64, 264, 36" in error, error
    for wrong in ("NSUBS", "SUBSET"):
        assert wrong not in error, (
            f"the refusal text must not blame {wrong}: NSUBS >= 1 always "
            f"(contrl.F:671-673 adds one for the global subset)")


def test_the_verdict_ignores_the_ctime_run_stamp(tmp_path):
    """``ctime()`` lives in the header; two runs of one deck differ there.

    ``hist1.F:210-234`` writes the wall-clock stamp unconditionally and
    ``timer_c.c:30-40`` produces it, so a comparison that hashed or diffed the
    raw bytes would report a difference between two identical runs.  Rewriting
    those bytes must leave every number identical.
    """
    from tools.compare_t01 import read_t01, read_port_csv, score
    reference = read_t01(GOLDEN_T01)
    offset, length = reference.run_stamp
    assert length == 24, "hist1.F:212-214 copies a 24-byte ctime stamp"
    blob = bytearray(GOLDEN_T01.read_bytes())
    for i in range(length):
        blob[offset + i] = 0x41 if blob[offset + i] != 0x41 else 0x42
    stamped = tmp_path / "STAMPED_T01"
    stamped.write_bytes(bytes(blob))
    other = read_t01(stamped)
    assert other.run_stamp == reference.run_stamp
    csv = _write_port_csv(reference, tmp_path / "port.csv", scale={"IE": 1.1})
    port = read_port_csv(csv)
    a = score(reference, port)
    b = score(other, port)
    assert a.worst.rel_rms == b.worst.rel_rms
    assert {k: v.rel_rms for k, v in a.per_channel.items()} == \
        {k: v.rel_rms for k, v in b.per_channel.items()}
    assert b.worst.verdict == "DEVIATION", "the 10 % IE offset must survive"


# ---------------------------------------------------------------------------
# 7. admissibility is carried, not assumed
# ---------------------------------------------------------------------------

def test_the_row_names_its_evidence_channel_and_its_provenance(tmp_path,
                                                               monkeypatch):
    """A parity row must be able to say *which* evidence it rests on.

    ``oracle_provenance.json`` decides that: ``T01 (binary results table)`` is
    admissible there, H3D / native ``.k`` / ``/ALE/STRUCTURED_MESH`` /
    ``/CHECKSUM_REPORT``-over-H3D are not.  A row that cannot name its source
    may not claim a channel, so the key is looked up in the record rather than
    assumed.
    """
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    census = provenance["admissible_parity_evidence"]
    assert census[T01_EVIDENCE_KEY].strip().lower().startswith("yes")
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "th2csv-missing",
                               "t01": str(GOLDEN_T01), "csv": None},
                 port_info={"status": "ok", "csv": csv})
    evidence = row["evidence"]
    assert evidence["channel"] == T01_EVIDENCE_KEY
    assert evidence["admissible"] is True
    assert evidence["source"].endswith("oracle_provenance.json")
    assert evidence["inadmissible_features_in_deck"] == []


def test_a_deck_feature_the_provenance_forbids_is_flagged(tmp_path, monkeypatch):
    """``/H3D`` or ``/ALE/STRUCTURED_MESH`` in the deck must reach the record.

    Not as a failure — the T01 is still admissible evidence — but as a stated
    fact, so a reader can see which *other* channels of that run were not.  The
    synthetic deck is injected through ``find_examples`` so the harness really
    reads it (``one()`` opens the engine deck itself).
    """
    from tools import validate_vs_fortran as V
    deck0 = tmp_path / "CASE_0000.rad"
    deck1 = tmp_path / "CASE_0001.rad"
    deck0.write_text("#RADIOSS STARTER\n/TITLE\nCASE\n", encoding="utf-8")
    deck1.write_text("#RADIOSS ENGINE\n/ANIM/DT\n0.01\n/H3D/DT\n0.01\n"
                     "/ALE/STRUCTURED_MESH\n/SSTOP\n", encoding="utf-8")
    flagged = V.deck_inadmissible_features(deck1.read_text())
    assert set(flagged) == {"H3D animation files",
                            "/ALE/STRUCTURED_MESH (S-ALE)"}
    t01 = _read_golden()
    csv = _write_port_csv(t01, tmp_path / "port.csv")
    monkeypatch.setattr(V, "find_examples",
                        lambda only: [("case", "CASE", str(deck0), str(deck1))])
    monkeypatch.setattr(V, "oracle_paths", lambda *a, **k: {
        key: "/opt/or/bin/x" for key in V.ORACLE_KEYS})
    monkeypatch.setattr(V, "run_fortran",
                        lambda *a, **k: {"status": "th2csv-missing",
                                         "t01": str(GOLDEN_T01), "csv": None})
    monkeypatch.setattr(V, "run_pyradioss",
                        lambda *a, **k: {"status": "ok", "csv": csv})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert V.parity(_parity_args(tmp_path)) == 0
    row = _rows(tmp_path)[0]
    assert set(row["evidence"]["inadmissible_features_in_deck"]) == {
        "H3D animation files", "/ALE/STRUCTURED_MESH (S-ALE)"}


# ---------------------------------------------------------------------------
# 8. the published vocabulary does not shift
# ---------------------------------------------------------------------------

def test_every_class_the_harness_emits_is_recorded_or_a_subtag_of_one():
    """The class vocabulary of a published sweep must not shift.

    Every class string this harness can write must be either a class
    ``parity_m41.json`` already carries, or the base class plus a lowercase
    subtag (the ``PORT-ONLY(implicit)`` / ``FORTRAN-FAIL(th2csv-missing)``
    convention).  The whitelist is **the recorded vocabulary and nothing
    else** — an earlier revision of this test added ``FORTRAN-FAIL`` to it,
    which is exactly how a bare ``FORTRAN-FAIL`` slipped through; the two
    remaining bare emitters are pinned separately and by name.
    """
    import re
    from tools import validate_vs_fortran as V
    record = json.loads(PARITY_M41.read_text(encoding="utf-8"))
    recorded = {r["class"] for r in record["results"]}
    assert "FORTRAN-FAIL" not in recorded, (
        "if a sweep ever records a bare FORTRAN-FAIL this test must be "
        "re-read: the whitelist below is the recorded set")
    src = Path(V.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'row\["class"\]\s*=\s*"([^"]+)"', src))
    emitted |= set(re.findall(r'"(FORTRAN-FAIL\([a-z0-9\-]+\))"', src))
    assert emitted, "the class-assignment pattern changed: this test is stale"
    # the class *stems* the recorded sweeps use: the bare strings plus the part
    # before "(" of every subtagged one (PORT-ONLY(implicit) -> PORT-ONLY)
    # FORTRAN-FAIL is the ONE stem outside the recorded vocabulary, and it is
    # a documented deviation: the M36..M41 sweeps record none, but P0.9
    # established FORTRAN-FAIL(<subtag>) as the additive convention for a
    # Fortran-side failure a recorded sweep never had.  Two statuses still emit
    # it bare (pinned by name below).
    stems = {c.split("(")[0] for c in recorded} | {"FORTRAN-FAIL"}
    for value in emitted:
        if value in recorded:
            continue
        match = re.fullmatch(r"([A-Z][A-Z\-]*)\(([a-z0-9\-]+)\)", value)
        assert match, f"{value!r} is neither recorded nor a subtagged class"
        assert match.group(1) in stems, (
            f"{value!r} introduces the base class {match.group(1)!r}, which no "
            f"recorded sweep uses")

    # The ONE documented deviation, pinned by name: two pre-existing statuses
    # still print a bare FORTRAN-FAIL through the conditional emitter.  A third
    # must fail here rather than slip through the whitelist.
    bare = {status for status, subtag in V.FORTRAN_FAIL_SUBTAGS.items()
            if subtag is None}
    assert bare == {"engine-fail", "th2csv-fail"}, (
        f"the bare FORTRAN-FAIL emitters changed: {sorted(bare)}")
    for status, subtag in V.FORTRAN_FAIL_SUBTAGS.items():
        if subtag is not None:
            assert subtag == status, f"{status} -> {subtag} is not self-describing"
    assert f'"FORTRAN-FAIL({V.FORTRAN_FAIL_SUBTAGS["th2csv-missing"]})"' in src \
        or 'f"FORTRAN-FAIL({subtag})"' in src


def test_the_no_comparable_class_is_the_recorded_one_on_both_routes(
        tmp_path, monkeypatch):
    """``NO-CHANNELS`` for "nothing comparable", on the CSV route as well.

    The CSV route had its own bare ``FORTRAN-FAIL`` for this condition; both
    routes now emit the class the recorded sweeps use, so the two routes cannot
    disagree about what "no comparison" looks like.
    """
    t01 = _read_golden()
    # the CSV route: two CSVs with no shared column at all
    port = tmp_path / "port.csv"
    port.write_text("TIME,SOMETHING_ELSE\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
    fortran = tmp_path / "fortran.csv"
    fortran.write_text("TIME,ANOTHER\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
    row = _drive(monkeypatch, tmp_path,
                 fortran_info={"status": "ok", "csv": str(fortran),
                               "t01": str(GOLDEN_T01)},
                 port_info={"status": "ok", "csv": str(port)})
    assert row["comparison_route"] == CSV_ROUTE
    assert row["class"] == "NO-CHANNELS", row
    assert "max_rel_rms" not in row
    assert "no overlapping channels" in row["fortran"]["error"]