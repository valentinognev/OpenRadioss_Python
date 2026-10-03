"""Gate for P0.5 -- the golden reference run of the real Fortran solver.

This module is the FIRST piece of differential evidence in the program: it runs
the oracle built by ``tools/oracle/build_oracle.sh`` on one vendored deck, and
pins the result in ``tools/validation_data/oracle_smoke.json``.  Every later
phase's "Numerics changes REQUIRE validation evidence" rule is anchored on that
file, which is why the oracle's **determinism** is asserted here and not
deferred: if the oracle were not bit-reproducible, no parity claim in the
program would be admissible.

Three layers, in decreasing order of "always applies" (the same split
``tests/test_p0_oracle_build.py`` uses, and the same
``PYRADIOSS_ORACLE_DISABLED`` / ``PYRADIOSS_ORACLE_REQUIRED`` handling):

1. **Structural** (``test_stored_golden_record_is_complete``,
   ``test_stored_golden_t01_reproduces_the_stored_maxima``,
   ``test_stored_golden_deck_bytes_match``) -- always run, no oracle.  They
   check the stored record's completeness and re-derive its per-channel maxima
   from the committed golden ``TENSILET01`` with pure Python, so the T01 parser
   itself is pinned by a test that cannot rot silently.
2. **Live oracle** (``test_oracle_reproduces_reference_t01``) -- the brief's
   test: one live run must reproduce the stored cycle count, T01 size,
   deterministic T01 md5 and per-channel maxima.  SKIP with an actionable
   reason while the oracle is absent, FAIL under ``PYRADIOSS_ORACLE_REQUIRED=1``.
3. **Determinism** (``test_oracle_t01_is_bit_reproducible``,
   ``test_oracle_t01_differs_only_in_the_run_stamp``) -- three runs in separate
   scratch directories.  These FAIL, never skip, when the oracle is present and
   non-deterministic.

The determinism statement, and why it is stated the way it is
-----------------------------------------------------------------
Upstream stamps the wall clock into the T01: ``hist1.F`` writes
``MY_CTIME(ITITLE)`` (a plain ``ctime()``, ``timer_c.c:30-40``) into
``CH80(1:24)`` and then writes that record unconditionally
(``engine/source/output/th/hist1.F:210-234``).  There is no keyword and no
environment variable that suppresses it, so **the raw T01 bytes cannot be
byte-reproducible across runs** and an md5 of them is not a usable anchor.

The anchor is therefore the md5 of the T01 with exactly those 24 stamp bytes
zeroed (``t01.md5_definition`` in the stored record), and the two determinism
tests prove both halves of the claim:

* the three normalized digests are equal  -> the oracle is bit-reproducible;
* every differing raw byte between any two runs lies inside the 24-byte stamp
  window -> the physics payload is reproducible and only the clock moves.

Measured (2026-10-03, oracle of ``oracle_provenance.json``): three runs of
``examples/tensile_bar`` gave normalized md5
``e3688899358f35e825cd640f9bd94964`` every time; the raw md5 of two runs that
shared a wall-clock second was identical (``1ece3c2b...``) and a third run,
one second away, differed in **one** byte -- the seconds digit of the stamp.

Upstream Fortran origins (``$OR_SRC`` = /home/valentin/Projects/OpenRadioss/OpenCourant):

* ``INSTALL.md:110-111``            -- ``./starter_linux64_gf -i <deck> -np 1``
  then ``./engine_linux64_gf -i <deck>``; the engine takes no ``-np`` (see
  ``oracle_provenance.json`` -> ``invocation``, measured).
* ``INSTALL.md:34-42``              -- ``RAD_CFG_PATH`` / ``LD_LIBRARY_PATH``
  (and ``RAD_H3D_PATH``, which this oracle deliberately leaves UNSET).
* ``INSTALL.md:95-101``             -- ``OMP_NUM_THREADS``; pinned to 1 here so
  no OpenMP reduction order can vary between runs.
* ``qa-tests/scripts/or_run_test.py:74-125`` -- upstream's own reference
  driver: the starter is a separate invocation whose return code is checked,
  then the engine(s), each in a private copy of the data directory.
* ``engine/source/output/th/hist1.F:154-188``  -- the T01 is opened per
  ``ITTYP``; ``ITTYP==3`` (:162-173) is the 2022 default and is the format this
  test's parser reads.
* ``engine/source/output/th/hist1.F:210-234`` -- the two header records, the
  second of which carries the ``ctime`` run stamp.
* ``engine/source/system/timer_c.c:30-40``      -- ``my_ctime``: ``time()`` +
  ``ctime()``, the 24 characters copied at :39, no suppression hook.
* ``engine/source/output/th/hist1.F:300-316``   -- ``NGLOBTH=23`` and the
  ``1..23`` curve-code record that delimits the data section.
* ``engine/source/output/th/hist2.F:302-303``   -- the ``TT`` record.
* ``engine/source/output/th/hist2.F:307-333``   -- the 23 global channels, in
  the order they are written.
* ``starter/source/output/th/write_thnms1.F90:228-250`` -- the authoritative
  index -> name -> description table for those 23 channels.
* ``engine/source/output/th/wrtdes.F:121-133``  -- ``ITTYP==3`` writes each
  value as a **single-precision** ``REAL`` (``R4 = A(I)``) even in a
  double-precision build: the stored maxima are float32 widened to float64.
* ``common_source/tools/input_output/write_routines.c:499-511`` (``eor_c``),
  ``:520-540`` (``write_r_c``), ``:646-665`` (``write_i_c``) -- the Radioss
  IEEE format: **big-endian** 4-byte record markers, big-endian int32, and
  ``real_to_IEEE_ASCII`` for the values.
"""

import json
import os
import pathlib
import re
from pathlib import Path

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

SMOKE_JSON = REPO / "tools" / "validation_data" / "oracle_smoke.json"
PROVENANCE = REPO / "tools" / "validation_data" / "oracle_provenance.json"
GOLDEN_DIR = REPO / "tests" / "data" / "oracle_smoke"

ORACLE_DISABLED = os.environ.get("PYRADIOSS_ORACLE_DISABLED") == "1"
ORACLE_REQUIRED = os.environ.get("PYRADIOSS_ORACLE_REQUIRED") == "1"

#: Number of independent live runs the determinism gate needs (the brief asks
#: for "at least TWICE (ideally three times)").
N_REFERENCE_RUNS = 3

_HINT = (
    "run tools/oracle/mirror_and_fetch.sh (acquires extlib) then "
    "tools/oracle/build_oracle.sh"
)


# ---------------------------------------------------------------------------
# Oracle discovery -- pyradioss.paths is the single resolver (plan 00 §4.1);
# the DISABLED/REQUIRED gating matches tests/test_p0_oracle_build.py.
# ---------------------------------------------------------------------------

def _oracle_or_skip(reason: str):
    """One place for the DISABLED / REQUIRED / plain-skip decision.

    ``PYRADIOSS_ORACLE_DISABLED=1`` always wins (an operator asked for silence),
    then ``PYRADIOSS_ORACLE_REQUIRED=1`` turns the absence into a failure (the
    Phase 0 exit gate sets it), and otherwise the test skips with an actionable
    reason -- the same precedence ``tests/test_p0_oracle_build.py`` uses.
    """
    if ORACLE_DISABLED:
        pytest.skip("PYRADIOSS_ORACLE_DISABLED=1")
    if ORACLE_REQUIRED:
        pytest.fail(f"{reason} and PYRADIOSS_ORACLE_REQUIRED=1 is set: {_HINT}")
    pytest.skip(f"oracle not built ({reason}; {_HINT}); set "
                f"PYRADIOSS_ORACLE_REQUIRED=1 to enforce")


def _require_oracle():
    """Skip when the oracle is absent, fail when the gate demands it."""
    from pyradioss import paths

    try:
        starter, engine = paths.or_starter(), paths.or_engine()
    except FileNotFoundError as exc:
        _oracle_or_skip(f"the oracle binaries do not resolve ({exc})")
        return
    if starter.is_file() and engine.is_file():
        return
    _oracle_or_skip(f"the oracle binaries are missing ({starter}, {engine})")


# ---------------------------------------------------------------------------
# The stored record
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke():
    assert SMOKE_JSON.is_file(), (
        f"missing golden record {SMOKE_JSON}; regenerate it with "
        "`python -m tools.oracle.oracle_selftest --write` after running "
        f"{_HINT}"
    )
    return json.loads(SMOKE_JSON.read_text())


@pytest.fixture(scope="module")
def reference_runs(tmp_path_factory):
    """``N_REFERENCE_RUNS`` live oracle runs, each in its own scratch dir.

    Module-scoped and shared: the reproduction test and the two determinism
    tests must look at the *same* runs, and three runs of this deck cost about
    1.2 s of solver time in total (starter 0.30 s, engine 0.08 s each), so it
    stays in the default tier -- no ``@pytest.mark.slow``.
    """
    _require_oracle()
    from tools.oracle.oracle_selftest import run_reference

    base = tmp_path_factory.mktemp("oracle_smoke")
    runs = []
    for i in range(N_REFERENCE_RUNS):
        runs.append(run_reference("TENSILE", workdir=base / f"run{i}"))
    return runs


def test_stored_golden_record_is_complete(smoke):
    """Every field a later parity claim leans on must be present and honest."""
    assert smoke["schema"] == "pyradioss/oracle-smoke/1"

    # -- the oracle is the build from this repo, not a third-party binary --
    oracle = smoke["oracle"]
    assert oracle["built_by"] == "tools/oracle/build_oracle.sh"
    assert oracle["third_party_prebuilt_binary"] is False
    assert (REPO / oracle["provenance"]).is_file(), oracle["provenance"]
    provenance = json.loads((REPO / oracle["provenance"]).read_text())
    assert provenance["upstream_git_sha"] == oracle["upstream_git_sha"]
    for key in ("starter", "engine"):
        entry = oracle[key]
        assert entry["sha256"] and entry["path"], f"oracle.{key} incomplete"

    # -- the deck, by path AND by content --
    deck = smoke["deck"]
    assert deck["run_name"] == "TENSILE"
    for kind in ("starter_deck", "engine_deck"):
        assert (REPO / deck[kind]).is_file(), deck[kind]
        assert re.fullmatch(r"[0-9a-f]{64}", deck[f"{kind}_sha256"])
    assert deck["why"], "the deck choice must be argued, not merely made"

    # -- the invocation, verbatim --
    inv = smoke["invocation"]
    assert inv["starter_argv"][-2:] == ["-np", "1"], inv["starter_argv"]
    assert "-np" not in inv["engine_argv"], (
        "the engine rejects -np (measured: usage dump then SIGSEGV); the "
        "recorded argv must not contain it"
    )
    assert inv["engine_argv"][-2:] == ["-nt", "1"], inv["engine_argv"]

    # -- the h3d safety fact, and the anchor's scope --
    env = smoke["environment"]
    assert env["rad_h3d_path"] is None
    assert env["rad_h3d_path_unset"] is True
    assert env["rad_h3d_path_unset_reason"]
    assert env["omp_num_threads"] == "1"
    # M5: the normalized digest keeps CH80(34:59)/(60:80) -- VERSIO and CPUNAM --
    # so it is bound to the build AND the architecture, not only the build.
    assert "CPUNAM" in env["anchor_scope"], (
        "environment.anchor_scope must say the anchor is architecture-bound: "
        "hist1.F:212-217 writes VERSIO(2) and CPUNAM into the same CH80 "
        "record, right after the stamp this record normalises away"
    )
    assert "architecture" in env["anchor_scope"]

    # -- the run verdict --
    run = smoke["run"]
    assert run["verdict"] == "NORMAL"
    assert run["verdict_banner"] == "NORMAL TERMINATION"
    assert isinstance(run["n_cycles"], int) and run["n_cycles"] > 0
    assert run["starter_msgerrors"] == 0
    assert run["engine_msgerrors"] == 0
    assert run["wall_seconds"]["total"] > 0.0

    # -- the T01 --
    t01 = smoke["t01"]
    assert (REPO / t01["path"]).is_file(), t01["path"]
    assert (REPO / t01["path"]).stat().st_size == t01["size_bytes"]
    assert re.fullmatch(r"[0-9a-f]{32}", t01["md5_normalized"]), (
        "the anchor key must be named md5_normalized: t01.md5_normalized is "
        f"NOT hashlib.md5(t01 bytes) -- that is t01.md5_raw. Got keys "
        f"{sorted(t01)}"
    )
    assert re.fullmatch(r"[0-9a-f]{32}", t01["md5_raw"])
    assert "md5_normalized" in t01["md5_definition"], t01["md5_definition"]
    assert t01["md5_raw_is_reproducible"] is False, (
        "the raw md5 embeds the run's wall clock; if this ever claims to be "
        "reproducible the md5_definition must be revisited"
    )
    assert t01["run_stamp"]["length"] == 24
    assert len(t01["md5_raw_observed"]) >= N_REFERENCE_RUNS
    assert t01["n_steps"] > 1
    assert t01["channels_per_step"]["global"] == 23
    assert t01["header_unparsed_records"] == [], (
        "the header walk should account for every header record of this deck; "
        f"unparsed: {t01['header_unparsed_records']}"
    )

    # -- per-channel maxima: 23 named globals, 2 named part, 2 named TH-group --
    channels = smoke["channel_maxima"]["global"]
    assert len(channels) == 23, len(channels)
    assert [c["index"] for c in channels] == list(range(1, 24))
    assert [c["name"] for c in channels][:4] == ["IE", "KE", "XMOM", "YMOM"]
    for entry in channels:
        assert entry["description"], entry
        for key in ("max", "max_abs", "final"):
            assert isinstance(entry[key], (int, float)), (entry, key)
    assert [c["index"] for c in smoke["channel_maxima"]["part"]] == [1, 2], (
        "the deck's /TH/PART/2 block asks for IE KE (curve codes 1 and 2)"
    )
    # I3: the TH-group channels are named too -- varn1_title
    # (th_titles.F90:168-189) keyed by the codes hist1.F:585-587 records, which
    # for the deck's /TH/NODE/1 ... DX VX are 1 and 4.
    group = smoke["channel_maxima"]["th_group"]
    assert [c["index"] for c in group] == [1, 4], group
    assert [c["name"] for c in group] == ["X-DISPLACEMENT", "X-VELOCITY"], group
    assert t01["th_group_curve_codes"] == [1, 4], (
        "the deck's /TH/NODE/1 ... DX VX card is curve codes 1 and 4"
    )

    # -- anything NOT established is null WITH a reason, never invented --
    for key, value in smoke["not_established"].items():
        assert value is None, f"not_established.{key} must be null, got {value!r}"
        assert smoke["not_established_reasons"].get(key), (
            f"not_established.{key} carries no reason"
        )


def test_stored_golden_deck_bytes_match(smoke):
    """The recorded sha256 must be the bytes on disk, right now."""
    deck = smoke["deck"]
    for kind in ("starter_deck", "engine_deck"):
        path = REPO / deck[kind]
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == deck[f"{kind}_sha256"], (
            f"{deck[kind]} changed: {digest} != {deck[f'{kind}_sha256']}. The "
            "golden run is only about the bytes it was made from; re-run "
            "`python -m tools.oracle.oracle_selftest --write` if the deck was "
            "intentionally replaced."
        )


def test_stored_golden_t01_reproduces_the_stored_maxima(smoke):
    """Re-derive everything from the committed T01 -- no oracle required.

    This pins :mod:`tools.oracle.oracle_selftest`'s parser and the recorded
    maxima against each other, so a later change to either shows up here even
    on a box where the oracle was never built.  It also pins the run stamp by
    CONTENT twice over, which is the only thing that can catch a window that is
    the right length in the wrong place:

    * ``t01.run_stamp.content`` must be built from the bytes the committed
      binary carries (M4/I1) -- a record whose thesis is "every field is
      measured, not asserted" cannot ship a typed-in example of its own header;
    * the 9 bytes right after the window must be ``' RADIOSS '``
      (``hist1.F:212``, ``CH80(25:33)``), so the window is tied to the ctime
      FIELD and not merely to a ``== 24`` literal.
    """
    from tools.oracle.oracle_selftest import (
        channel_maxima,
        deterministic_md5,
        parse_t01,
        run_stamp_text,
        run_stamp_window,
    )

    blob = (REPO / smoke["t01"]["path"]).read_bytes()
    parsed = parse_t01(blob)
    assert len(blob) == smoke["t01"]["size_bytes"]
    assert parsed["n_steps"] == smoke["t01"]["n_steps"]
    assert parsed["header_records"] == smoke["t01"]["header_records"]
    assert parsed["channels_per_step"] == smoke["t01"]["channels_per_step"]
    assert parsed["t_first"] == smoke["t01"]["t_first"]
    assert parsed["t_last"] == smoke["t01"]["t_last"]
    assert parsed["hierarchy"] == smoke["t01"]["hierarchy"]
    assert parsed["part_codes"] == smoke["t01"]["curve_codes_by_group"]["part"]
    assert parsed["group_codes"] == (
        smoke["t01"]["curve_codes_by_group"]["th_group"])

    start, length = run_stamp_window(blob)
    assert parsed["run_stamp"] == (start, length)
    assert (start, length) == (
        smoke["t01"]["run_stamp"]["offset"],
        smoke["t01"]["run_stamp"]["length"],
    )
    # M4: the window is bounded by content on its far side, not only its length.
    assert blob[start + length:start + length + 9] == b" RADIOSS ", (
        f"the 9 bytes after the window are "
        f"{blob[start + length:start + length + 9]!r}, not ' RADIOSS ': the "
        "stamp window does not end where hist1.F:212 says the clock ends"
    )
    # I1: the recorded content is the committed binary's, re-read.
    stamp = run_stamp_text(blob)
    assert stamp == parsed["run_stamp_text"]
    assert stamp == smoke["t01"]["run_stamp"]["content"], (
        "run_stamp.content is not what the committed T01 carries; it must be "
        "built from the bytes read (tools/oracle/oracle_selftest.py "
        "run_stamp_text), never typed in"
    )
    assert blob[start:start + length].decode("ascii") in stamp

    assert deterministic_md5(blob) == smoke["t01"]["md5_normalized"]
    assert channel_maxima(parsed) == smoke["channel_maxima"]


def test_golden_engine_listing_is_committed_and_agrees(smoke):
    """The golden engine listing must be IN the tree, not merely written.

    ``--write`` drops ``<run>_0001.out`` next to the T01, and ``.gitignore:13``
    ignores ``*.out`` repo-wide, so it is force-added once (``git add -f``)
    rather than by editing the shared ignore file.  That is only safe while a
    test holds it to the record: this is the artefact carrying the
    ``ENGINE TERMINATION`` banner, the cycle count and the energy ledger --
    every one of them admissible per ``oracle_provenance.json`` -- and a
    committed-but-drifting copy would be worse than none.
    """
    listing = GOLDEN_DIR / "TENSILE_0001.out"
    assert listing.is_file(), (
        f"{listing} is missing. It is force-added because .gitignore:13 "
        f"ignores *.out; do not 'fix' this by dropping the file."
    )
    text = listing.read_text(errors="replace")
    assert "OpenRadioss Engine" in text[:2000], "not an engine listing"
    assert "CURRENT ENGINE" in text, "not an engine listing"
    assert smoke["run"]["verdict_banner"] in text, (
        f"the committed listing does not carry the recorded banner "
        f"{smoke['run']['verdict_banner']!r}"
    )
    cycles = re.search(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)", text)
    assert cycles and int(cycles.group(1)) == smoke["run"]["n_cycles"], (
        f"the committed listing's cycle count does not match the record's "
        f"{smoke['run']['n_cycles']}"
    )
    assert "** ERROR" not in text, "the golden run reported an error"
    # the listing and the T01 are copied from the SAME run by --write, so the
    # wall-clock second in the listing is the one in the T01's stamp: this is
    # the cross-check that keeps the two committed artefacts a matched pair.
    started = re.search(r"EXECUTION STARTED \.*:\s*([\d/]+)\s+([\d:]+)", text)
    assert started, "the listing carries no execution stamp"
    assert started.group(2).rsplit(":", 1)[1] in smoke["t01"]["run_stamp"][
        "content"], (
        f"the listing's start second {started.group(2)!r} is not in the "
        f"committed T01's run stamp {smoke['t01']['run_stamp']['content']!r}: "
        "the two artefacts are from different runs"
    )


# ---------------------------------------------------------------------------
# Live oracle
# ---------------------------------------------------------------------------

def test_oracle_reproduces_reference_t01(reference_runs, smoke):
    """The brief's test: one live run must reproduce the stored golden run.

    Every expected value comes from ``oracle_smoke.json`` -- there is no second
    hardcoded copy of the cycle count or the digest.
    """
    ref = reference_runs[0]
    stored = smoke

    assert ref["verdict"] == "NORMAL", (
        f"ENGINE TERMINATION was {ref['verdict']!r}:\n{ref['engine_tail']}"
    )
    assert ref["n_cycles"] == stored["run"]["n_cycles"], (
        f"cycle count drifted: {ref['n_cycles']} != "
        f"{stored['run']['n_cycles']}"
    )
    assert ref["starter_msgerrors"] == 0
    assert ref["engine_msgerrors"] == 0
    assert ref["t01_size_bytes"] == stored["t01"]["size_bytes"]
    assert ref["t01_md5_normalized"] == stored["t01"]["md5_normalized"], (
        "the deterministic T01 digest drifted -- the reference run is no "
        "longer the one recorded in oracle_smoke.json. Remember this key is "
        "the STAMP-NORMALISED digest: if the oracle was legitimately rebuilt "
        "or moved to another architecture, re-derive it with `--write` and "
        "say why (see environment.anchor_scope); a mismatch is not by itself "
        "a physics regression."
    )
    assert ref["run_stamp"] == (
        stored["t01"]["run_stamp"]["offset"],
        stored["t01"]["run_stamp"]["length"],
    )
    assert ref["channel_maxima"] == stored["channel_maxima"]
    assert ref["argv"]["starter"] == stored["invocation"]["starter_argv"]
    assert ref["argv"]["engine"] == stored["invocation"]["engine_argv"]
    assert ref["rad_h3d_path"] is None, (
        "RAD_H3D_PATH must stay unset: the reachable libh3dwriter.so is one "
        "parameter short of what the pinned source calls and would write "
        "silently wrong H3D files"
    )


def test_oracle_t01_is_bit_reproducible(reference_runs, smoke):
    """Three live runs, three separate scratch dirs, one digest.

    FAILED, never skipped, when the oracle is present and non-deterministic:
    a non-reproducible oracle makes every parity claim in this program
    inadmissible, so it must be a red test, not a yellow one.
    """
    digests = [r["t01_md5_normalized"] for r in reference_runs]
    assert len(set(digests)) == 1, (
        "the reference T01 is NOT bit-reproducible across "
        f"{len(digests)} runs: {digests}. Determinism is anchored on "
        "t01.md5_normalized (run stamp zeroed); if a digest differs, either "
        "the solver "
        "is not deterministic or the normalization no longer covers every "
        "varying byte."
    )
    assert digests[0] == smoke["t01"]["md5_normalized"]
    for run in reference_runs:
        assert run["verdict"] == "NORMAL"
        assert run["n_cycles"] == smoke["run"]["n_cycles"]
        assert run["t01_size_bytes"] == smoke["t01"]["size_bytes"]


def test_oracle_t01_differs_only_in_the_run_stamp(reference_runs, smoke):
    """Every varying raw byte must lie inside the 24-byte ``ctime`` window.

    This is the half of the determinism claim that says *why* the raw md5 is
    not the anchor: the oracle reproduces itself exactly, and the only bytes
    that move are the wall-clock stamp ``hist1.F:210-234`` writes.
    """
    from tools.oracle.oracle_selftest import diff_offsets

    first = reference_runs[0]
    start, size = first["run_stamp"]
    assert size == smoke["t01"]["run_stamp"]["length"]
    assert start == smoke["t01"]["run_stamp"]["offset"]
    window = range(start, start + size)

    for number, other in enumerate(reference_runs[1:], start=2):
        assert other["t01_size_bytes"] == first["t01_size_bytes"]
        offsets = diff_offsets(first["t01_bytes"], other["t01_bytes"])
        stray = [off for off in offsets if off not in window]
        assert not stray, (
            f"run 1 vs run {number}: {len(stray)} byte(s) differ OUTSIDE the "
            f"{size}-byte ctime run stamp at offset {start} (first at "
            f"{stray[0]}); the physics payload is not reproducible"
        )


def test_three_raw_digests_are_recorded(smoke):
    """The record must carry the determinism evidence, not just a claim about it.

    Note what is *not* asserted: that the raw digests are all equal.  They
    cannot be -- ``hist1.F:211`` stamps the wall clock into the T01 and no
    keyword suppresses it.  What must hold is that the normalized digests are
    all equal and that the raw variation is RECORDED rather than asserted.

    I4, the non-degeneracy half: at least two distinct RAW digests must be in
    the record.  Without it, a ``--write`` whose runs all finished inside one
    wall-clock second would emit ``distinct_md5_raw: [<one digest>]`` while
    still stating ``md5_raw_is_reproducible: false`` -- the record's central
    non-reproducibility claim would be untested prose.  ``build_record`` refuses
    to write such a record and retries until two runs straddle a second; this
    assertion is what keeps that refusal honest against a hand-edited file.
    """
    determinism = smoke["determinism"]
    observed = smoke["t01"]["md5_raw_observed"]

    assert determinism["runs"] >= N_REFERENCE_RUNS
    assert len(determinism["md5"]) == determinism["runs"]
    assert len(set(determinism["md5"])) == 1, (
        f"the recorded determinism digests differ: {determinism['md5']} -- the "
        "oracle was NOT bit-reproducible when this record was written"
    )
    assert determinism["distinct_md5"] == [smoke["t01"]["md5_normalized"]]
    assert determinism["md5_raw"] == observed
    assert smoke["t01"]["md5_raw"] == observed[0]
    assert sorted(set(determinism["md5_raw"])) == determinism["distinct_md5_raw"]
    assert determinism["distinct_md5_raw_count"] == len(
        determinism["distinct_md5_raw"])
    assert len(determinism["distinct_md5_raw"]) >= 2, (
        "only ONE distinct raw digest was recorded, so every run of the "
        "--write finished inside the same wall-clock second and the record's "
        "`md5_raw_is_reproducible: false` was never measured. Re-run "
        "`python -m tools.oracle.oracle_selftest --write` until two runs "
        f"straddle a second; got {determinism['distinct_md5_raw']}"
    )
    assert smoke["t01"]["md5_raw_reason"], (
        "a non-reproducible raw md5 must say why, in the record itself"
    )
    # NOTE: the LIVE runs above are deliberately not required to straddle a
    # second -- three 0.4 s runs can easily share one, and a test that demanded
    # otherwise would be flaky. The non-degeneracy duty belongs to --write
    # (which retries and then refuses) and to the assertion above.
