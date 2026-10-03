"""Task P0.8 — the hashed corpus manifest (``rd_decks_manifest.json``).

The Phase 0 exit gate and every later phase's "numerics changes require
validation evidence" rule need one file that says, without hand-editing:

* **which** deck files are covered by the differential-validation evidence,
* **what they hash to** (so a later run can prove it ran the same bytes),
* **whether** the port is expected to reproduce the oracle on each of them,
* and **what that claim is qualified by** — the families the port still skips
  on the deck, and the fact that the historical verdicts were measured on a
  corpus extract that no longer exists.

That file is ``tools/validation_data/rd_decks_manifest.json``; the loader is
``tools.validate_vs_fortran.load_manifest()``.  It replaces the hand-kept case
list that ``tools/validation_data/inventory.json`` and ``parity_m41.json`` are
keyed by, so it must stay joinable with both of them on ``case_id``.

What is pinned here (the properties that make the manifest worth having):

1. **coverage** — exactly one record per ``*_0000.rad`` starter deck under the
   corpus root, and no record for anything else;
2. **integrity** — every record's file exists and its recorded ``sha256``
   equals a hash computed fresh from the bytes on disk (the *whole* vendored
   corpus: 75 decks / 8.8 MB hashed in ~10 ms, so every record is verified on
   every run rather than a sample);
3. **binding** — a ``deck`` is *relative to a corpus*, so every record carries
   the corpus fingerprint it was hashed from, the header is reachable through
   the public API, and :func:`resolve_manifest_record` refuses to resolve a
   record against a corpus that is not that one.  Otherwise a
   ``PYRADIOSS_RD_DECKS`` override silently joins a verdict to a *different*
   file at the same relative path;
4. **identity** — ``case_id`` values are unique and are the strings the
   evidence files already use (no locally invented ids);
5. **honesty of the envelope flag** — ``in_envelope`` is ``true`` only where a
   measured parity verdict says ``MATCH``; a statically predicted
   ``inventory.json`` classification is never enough; and an ``in_envelope``
   record may never be silent about the families the port skips on it;
6. **honesty of the provenance** — no record claims the historical verdict was
   measured on the bytes it now hashes (it was not: the sweep ran from a
   scratchpad extract that is gone);
7. **joinability** — the manifest intersects ``parity_m41.json`` and
   ``coverage_results_m41.json`` on ``case_id``, so neither side can be
   regenerated into uselessness without a test failing;
8. **no environment, no corpus** — ``load_manifest()`` returns the *recorded*
   hashes without exporting anything or touching a deck, so it works on a
   read-only share and with no corpus at all.

The corpus root is ``pyradioss.paths.rd_decks_dir()`` (``PYRADIOSS_RD_DECKS``
else the vendored ``tests/data/rd_decks``).  The manifest records which corpus
it describes; the skip condition below compares that against the **vendored
path in this checkout** — never against the manifest under test — so a missing
manifest fails instead of skipping.  A different extract (an env override) has
to be re-hashed with ``tools/build_rd_decks_manifest.py``; the
``test_check_is_portable_across_checkout_paths`` test does exactly that on a
copy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from pyradioss import paths

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tools" / "validation_data"
VENDORED = REPO / "tests" / "data" / "rd_decks"
MANIFEST_PATH = DATA / "rd_decks_manifest.json"
PARITY_PATH = DATA / "parity_m41.json"
INVENTORY_PATH = DATA / "inventory.json"
COVERAGE_PATH = DATA / "coverage_results_m41.json"

#: A class the port is *expected* to honour without complaint: a skip of one of
#: these is invisible in ``coverage_verdict`` (which counts non-control skips),
#: so the manifest has to carry the families explicitly.
CONTROL_CLASS = "control"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _doc() -> dict:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest() -> list:
    from tools.validate_vs_fortran import load_manifest
    return load_manifest()


def _resolved_corpus() -> Path:
    return paths.rd_decks_dir()


def _manifest_fingerprint() -> str:
    """The corpus the manifest says it hashed — "" when it cannot be read.

    Read with :func:`pyradioss.paths`-independent code so an absent manifest
    yields "" instead of raising at import time.
    """
    try:
        return _doc().get("corpus_root", {}).get("fingerprint", "")
    except (ValueError, OSError):
        return ""


def _corpus_is_described() -> bool:
    """True when ``paths.rd_decks_dir()`` is the corpus this manifest hashed.

    Two ways that holds, and the difference matters:

    * it IS this checkout's vendored corpus — the normal case;
    * it is a **byte-identical** copy elsewhere (``PYRADIOSS_RD_DECKS``), which
      the corpus fingerprint recognises, so the whole-corpus tests RUN instead
      of skipping.  A copy at another path is still the same corpus.

    Anything else (a different extract) skips here, loudly: the fingerprint
    mismatch is reported by the never-skipped
    ``test_live_corpus_is_the_corpus_the_manifest_describes``, which names both
    fingerprints and the command that re-hashes.  A missing manifest returns
    True on purpose: a missing manifest must FAIL (see
    ``test_manifest_file_exists``), not hide behind eight skips.
    """
    if not MANIFEST_PATH.is_file():
        return True
    try:
        root = _resolved_corpus()
    except (FileNotFoundError, OSError):
        return False
    try:
        if root.resolve() == VENDORED.resolve():
            return True
    except OSError:
        return False
    from tools.validate_vs_fortran import corpus_fingerprint
    expected = _manifest_fingerprint()
    return bool(expected) and corpus_fingerprint(str(root)) == expected


requires_described_corpus = pytest.mark.skipif(
    not _corpus_is_described(),
    reason="rd_decks_dir() is not the corpus this manifest describes "
           "(PYRADIOSS_RD_DECKS override to a different extract?); re-hash it "
           "with python tools/build_rd_decks_manifest.py --root <that dir>")


@pytest.fixture(scope="module")
def records() -> list:
    return _manifest()


@pytest.fixture(scope="module")
def header() -> dict:
    from tools.validate_vs_fortran import load_manifest_doc
    return load_manifest_doc()


@pytest.fixture(scope="module")
def parity_by_case_id() -> dict:
    doc = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    return {r["case_id"]: r for r in doc["results"] if r.get("case_id")}


@pytest.fixture(scope="module")
def coverage_by_case_id() -> dict:
    doc = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    return {c["case_id"]: c for c in doc["cases"] if c.get("case_id")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_doc(tmp_path: Path, doc: dict, name: str = "manifest.json") -> str:
    """A mutated manifest COPY on disk, never the repo artifact."""
    target = tmp_path / name
    target.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return str(target)


def _copy_corpus(tmp_path: Path, decks=None) -> Path:
    """A byte-identical copy of the corpus' starter decks at another path."""
    copy = tmp_path / "corpus_copy"
    for deck in (decks if decks is not None
                 else sorted(VENDORED.rglob("*_0000.rad"))):
        target = copy / deck.relative_to(VENDORED)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deck, target)
    return copy


# --------------------------------------------------------------------------
# 0. the artifact exists at all (never skipped)
# --------------------------------------------------------------------------

def test_manifest_file_exists():
    assert MANIFEST_PATH.is_file(), (
        f"no manifest at {MANIFEST_PATH}; generate it with "
        "python tools/build_rd_decks_manifest.py")
    doc = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    from tools.validate_vs_fortran import MANIFEST_SCHEMA
    assert doc["schema"] == MANIFEST_SCHEMA
    assert doc["decks"], "the manifest has no deck records"


def test_live_corpus_is_the_corpus_the_manifest_describes(header):
    """Never skipped, and the loud half of the skip condition.

    The whole-corpus tests skip when ``rd_decks_dir()`` is some *other*
    extract; without this, that would be eight silent skips.  A byte-identical
    copy passes (the fingerprint matches, so those tests run); anything else
    fails here, naming both fingerprints and the command that re-hashes.
    """
    from tools.validate_vs_fortran import corpus_fingerprint
    live = corpus_fingerprint(str(_resolved_corpus()))
    expected = header["corpus_root"]["fingerprint"]
    assert live == expected, (
        f"the corpus rd_decks_dir() resolves to is NOT the corpus this "
        f"manifest describes ({live} != {expected}). The eight whole-corpus "
        f"tests skip until it is re-hashed:\n"
        f"  python tools/build_rd_decks_manifest.py --root {_resolved_corpus()}")


def test_skip_policy_follows_the_fingerprint_not_the_path(monkeypatch, tmp_path):
    """The skip condition itself, pinned.

    A **byte-identical** corpus copy elsewhere (``PYRADIOSS_RD_DECKS``) is the
    corpus the manifest describes, so the eight whole-corpus tests must RUN
    there, not skip.  Any other extract must not.  Deriving this from the
    vendored *path* — as round 1 did — satisfies neither half and hides a
    drifted corpus behind skips.
    """
    identical = _copy_corpus(tmp_path / "identical")
    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(identical))
    paths.reload()
    try:
        assert _resolved_corpus().resolve() != VENDORED.resolve(), (
            "the override did not take effect")
        assert _corpus_is_described() is True, (
            "a byte-identical corpus copy must not skip the whole-corpus "
            "tests — it is the corpus the manifest hashed")
    finally:
        monkeypatch.delenv("PYRADIOSS_RD_DECKS", raising=False)
        paths.reload()

    foreign = tmp_path / "foreign"
    (foreign / "rd_e" / "SOME_PKG").mkdir(parents=True)
    (foreign / "rd_e" / "SOME_PKG" / "X_0000.rad").write_bytes(b"#RADIOSS\n")
    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(foreign))
    paths.reload()
    try:
        assert _corpus_is_described() is False, (
            "a different extract must not pass as the described corpus")
    finally:
        monkeypatch.delenv("PYRADIOSS_RD_DECKS", raising=False)
        paths.reload()


# --------------------------------------------------------------------------
# 1. coverage: one record per starter deck
# --------------------------------------------------------------------------

@requires_described_corpus
def test_manifest_has_exactly_one_record_per_starter_deck(records):
    root = _resolved_corpus()
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*_0000.rad")}
    assert on_disk, f"no starter decks found under {root}"
    in_manifest = [r["deck"] for r in records]
    assert len(in_manifest) == len(set(in_manifest)), "duplicate deck records"
    assert set(in_manifest) == on_disk, (
        "manifest/corpus drift: "
        f"only in manifest {sorted(set(in_manifest) - on_disk)}, "
        f"only on disk {sorted(on_disk - set(in_manifest))}")


def test_every_corpus_starter_deck_is_covered(records):
    """The manifest must not be a silent subset of the corpus the harness runs.

    Same property as above but counted, and phrased through
    ``paths.rd_decks_dir()`` so it also holds for an env-overridden corpus.
    """
    root = _resolved_corpus()
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*_0000.rad")}
    assert on_disk, f"no starter decks found under {root}"
    missing = on_disk - {r["deck"] for r in records}
    assert not missing, (
        f"{len(missing)}/{len(on_disk)} corpus decks have no manifest record; "
        "regenerate with tools/build_rd_decks_manifest.py")


# --------------------------------------------------------------------------
# 2. integrity: every recorded hash is the hash of the bytes on disk
# --------------------------------------------------------------------------

@requires_described_corpus
def test_every_recorded_sha256_matches_the_file_on_disk(records):
    root = _resolved_corpus()
    for rec in records:
        target = root / rec["hashed_file"]
        assert target.is_file(), f"{rec['deck']}: hashed_file missing: {target}"
        assert _sha256(target) == rec["sha256"], (
            f"{rec['deck']}: sha256 mismatch — the deck changed since the "
            "manifest was generated")
        assert target.stat().st_size == rec["size_bytes"], (
            f"{rec['deck']}: size_bytes mismatch")
    assert records


def test_records_name_the_file_they_hash(records):
    """A reader must never have to guess which bytes ``sha256`` covers."""
    for rec in records:
        assert rec["hashed_file"], rec["deck"]
        assert rec["hashed_file"] == rec["deck"], (
            f"{rec['deck']}: hashed_file names another file — a record that "
            "hashes something else must say so explicitly")
        assert len(rec["sha256"]) == 64
        int(rec["sha256"], 16)          # raises if not hex


# --------------------------------------------------------------------------
# 3. binding: a record is bound to ONE corpus, and says which
# --------------------------------------------------------------------------

def test_header_is_reachable_through_the_public_api(header):
    """``load_manifest()`` returns records only; the header carries the corpus
    identity, the envelope rule and the counts, so it must be reachable."""
    for key in ("schema", "generated", "corpus_root", "hash_algorithm",
                "hashed_file_rule", "envelope_rule", "bytes_verified_rule",
                "joins", "counts", "notes"):
        assert key in header, f"manifest header is missing {key!r}"
    assert header["corpus_root"]["fingerprint"], "no corpus fingerprint"
    assert header["counts"]["decks"] == len(header["decks"])


def test_header_prose_states_the_rules_it_is_quoted_for(header):
    """The header prose IS the contract a reader quotes (``what does
    in_envelope mean?``, ``was the verdict measured on these bytes?``), so the
    two rule keys have to say the specific thing, not merely exist."""
    rule = header["envelope_rule"]
    assert "MATCH" in rule and "in_envelope" in rule
    verified = header["bytes_verified_rule"]
    assert "parity_run_deck_bytes_verified" in verified
    assert "coverage_run_deck_bytes_verified" in verified
    assert header["hash_algorithm"] == "sha256"
    assert "hashed_file" in header["hashed_file_rule"]
    assert header["joins"]["parity"].endswith("parity_m41.json (by case_id)")


def test_every_record_carries_the_corpus_it_was_hashed_from(records, header):
    fingerprint = header["corpus_root"]["fingerprint"]
    for rec in records:
        assert rec["corpus_fingerprint"] == fingerprint, (
            f"{rec['deck']}: record names corpus {rec['corpus_fingerprint']!r} "
            f"while the header says {fingerprint!r}")


@requires_described_corpus
def test_resolve_record_against_the_manifests_own_corpus(records):
    """The default resolution is the manifest's declared corpus, so a record's
    absolute path always hashes to its recorded ``sha256``."""
    from tools.validate_vs_fortran import resolve_manifest_record
    rec = records[0]
    path = Path(resolve_manifest_record(rec))
    assert path.is_absolute() and path.is_file()
    assert _sha256(path) == rec["sha256"]


def test_resolve_record_is_unaffected_by_an_env_override(records, monkeypatch,
                                                          tmp_path):
    """``deck`` is relative, so a ``PYRADIOSS_RD_DECKS`` override that carries a
    DIFFERENT deck at the same relative path must not be joined silently: the
    default resolution stays on the manifest's own corpus and still matches the
    recorded hash."""
    from tools.validate_vs_fortran import resolve_manifest_record
    rec = next(r for r in records if r["in_envelope"])
    alt = tmp_path / "alt_corpus"
    (alt / rec["deck"]).parent.mkdir(parents=True)
    (alt / rec["deck"]).write_bytes(b"#RADIOSS\n/END\n")     # same relpath, other bytes

    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(alt))
    paths.reload()
    try:
        resolved = Path(resolve_manifest_record(rec))
        assert _sha256(resolved) == rec["sha256"], (
            "the default resolution followed PYRADIOSS_RD_DECKS and would "
            "join a verdict to a different file")
        with pytest.raises(ValueError):
            resolve_manifest_record(rec, root=alt)
    finally:
        monkeypatch.delenv("PYRADIOSS_RD_DECKS", raising=False)
        paths.reload()


def test_resolve_record_accepts_a_byte_identical_corpus_elsewhere(records,
                                                                    tmp_path):
    """A byte-identical corpus at another path IS the same corpus: the
    fingerprint matches, so resolution and hash verification must succeed."""
    from tools.validate_vs_fortran import resolve_manifest_record
    rec = records[0]
    copy = _copy_corpus(tmp_path)
    resolved = Path(resolve_manifest_record(rec, root=copy))
    assert resolved == (copy / rec["hashed_file"]).resolve()
    assert _sha256(resolved) == rec["sha256"]


def test_resolve_record_refuses_a_record_without_a_corpus_fingerprint(records):
    """A hand-built record that omits ``corpus_fingerprint`` used to skip the
    I1 guard silently — the one input shape where the protection switches off
    without saying so.  It is now an explicit error, with an explicit opt-out."""
    from tools.validate_vs_fortran import resolve_manifest_record
    bare = dict(records[0])
    bare.pop("corpus_fingerprint")
    with pytest.raises(ValueError, match="corpus_fingerprint"):
        resolve_manifest_record(bare)
    # the opt-out exists, is spelled out in the signature, and still verifies
    # the bytes: it waives the corpus BINDING, not the hash
    resolved = Path(resolve_manifest_record(bare, require_fingerprint=False))
    assert _sha256(resolved) == bare["sha256"]


def test_resolve_record_opt_out_still_refuses_other_bytes(records, tmp_path):
    """``require_fingerprint=False`` must not become a way to skip verification."""
    from tools.validate_vs_fortran import resolve_manifest_record
    bare = dict(records[0])
    bare.pop("corpus_fingerprint")
    other = tmp_path / "other"
    (other / bare["deck"]).parent.mkdir(parents=True)
    (other / bare["deck"]).write_bytes(b"#RADIOSS\n/END\n")
    with pytest.raises(ValueError, match="hashes to"):
        resolve_manifest_record(bare, root=other, require_fingerprint=False)


def test_resolve_record_refuses_a_partial_corpus(records, tmp_path):
    """...and a directory holding only *some* of the decks is not that corpus,
    even when the one deck it holds has the right bytes."""
    from tools.validate_vs_fortran import resolve_manifest_record
    rec = records[0]
    partial = tmp_path / "partial"
    (partial / rec["deck"]).parent.mkdir(parents=True)
    # the checkout's own corpus, NOT rd_decks_dir(): this test must not depend
    # on what the environment points at
    shutil.copyfile(VENDORED / rec["deck"], partial / rec["deck"])
    with pytest.raises(ValueError, match="was hashed from corpus"):
        resolve_manifest_record(rec, root=partial)


# --------------------------------------------------------------------------
# 4. identity: unique, and the same ids the evidence files use
# --------------------------------------------------------------------------

def test_case_ids_are_unique(records):
    ids = [r["case_id"] for r in records if r["case_id"] is not None]
    assert len(ids) == len(set(ids)), "duplicate case_id in the manifest"
    assert ids, "no case_id carried over from inventory.json"


def test_case_ids_match_inventory_json(records):
    known = {c["case_id"]: c
             for c in json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))["cases"]}
    for rec in records:
        if rec["case_id"] is None:
            continue
        case = known.get(rec["case_id"])
        assert case is not None, f"invented case_id {rec['case_id']!r}"
        assert rec["category"] == case["category"]
        assert rec["package"] == case["package"]


def test_records_carry_exactly_the_documented_field_set(records):
    """The loader's MANIFEST_FIELDS is the contract; drift in either direction
    (a record with an undocumented field, or a documented field nobody emits)
    must fail here rather than reach a consumer."""
    from tools.validate_vs_fortran import MANIFEST_FIELDS
    documented = set(MANIFEST_FIELDS)
    for rec in records:
        assert set(rec) == documented, (
            f"{rec.get('deck')}: undocumented {sorted(set(rec) - documented)}, "
            f"missing {sorted(documented - set(rec))}")
        assert isinstance(rec["in_envelope"], bool)
        assert isinstance(rec["size_bytes"], int) and rec["size_bytes"] > 0
        assert rec["deck"].endswith("_0000.rad"), rec["deck"]


def test_loader_rejects_an_undocumented_record_field(tmp_path):
    """The loader's "undocumented extra field" branch is load-bearing: it is
    what stops a record gaining a field no consumer reads.  It has to be
    reachable from the public API, not only from the generator's assert."""
    from tools.validate_vs_fortran import load_manifest, load_manifest_doc
    doc = _doc()
    doc["decks"][0]["parity_verdict_but_prettier"] = "MATCH-ish"
    path = _write_doc(tmp_path, doc, "extra_field.json")
    with pytest.raises(ValueError, match="parity_verdict_but_prettier"):
        load_manifest_doc(path)
    with pytest.raises(ValueError, match="undocumented"):
        load_manifest(path)


def test_loader_rejects_a_record_missing_a_documented_field(tmp_path):
    """The mirror of the above: a *missing* field is equally unusable."""
    from tools.validate_vs_fortran import load_manifest_doc
    doc = _doc()
    doc["decks"][0].pop("in_envelope_reason")
    path = _write_doc(tmp_path, doc, "missing_field.json")
    with pytest.raises(ValueError, match="in_envelope_reason"):
        load_manifest_doc(path)


def test_loader_rejects_an_unknown_schema(tmp_path):
    from tools.validate_vs_fortran import load_manifest_doc
    doc = _doc()
    doc["schema"] = "pyradioss/rd-decks-manifest/999"
    path = _write_doc(tmp_path, doc, "bad_schema.json")
    with pytest.raises(ValueError, match="schema"):
        load_manifest_doc(path)


# --------------------------------------------------------------------------
# 5. the envelope flag is measured, never assumed, and never unqualified
# --------------------------------------------------------------------------

@requires_described_corpus
def test_in_envelope_requires_a_measured_match(records, parity_by_case_id):
    for rec in records:
        verdict = (parity_by_case_id.get(rec["case_id"])
                   if rec["case_id"] else None)
        if rec["in_envelope"]:
            assert verdict is not None, (
                f"{rec['deck']}: in_envelope without any parity evidence")
            assert verdict["class"] == "MATCH", (
                f"{rec['deck']}: in_envelope but the parity class is "
                f"{verdict['class']!r}")
        if verdict is not None and verdict["class"] != "MATCH":
            assert not rec["in_envelope"], rec["deck"]
        assert rec["in_envelope_source"], (
            f"{rec['deck']}: in_envelope_source must always be recorded")
        assert rec["in_envelope_reason"], (
            f"{rec['deck']}: in_envelope_reason must always be recorded")


def test_static_inventory_classification_never_sets_the_envelope(records):
    """A predicted IN_ENVELOPE classification is a *reading* prediction, not a
    parity result: it may inform a human but must never flip the flag."""
    for rec in records:
        if rec["inventory_classification"] == "IN_ENVELOPE":
            assert not rec["in_envelope"] or rec["parity_class"] == "MATCH"


@requires_described_corpus
def test_skipped_families_are_carried_verbatim(records, coverage_by_case_id):
    for rec in records:
        row = coverage_by_case_id.get(rec["case_id"]) if rec["case_id"] else None
        if row is None:
            assert rec["skipped_families"] == {}
            assert rec["coverage_hard_skips"] == []
            assert rec["coverage_blockers"] == []
            continue
        assert rec["coverage_verdict"] == row["verdict"]
        assert rec["skipped_families"] == row["skipped_families"], (
            f"{rec['deck']}: skipped_families does not match the coverage "
            "evidence")
        assert rec["coverage_hard_skips"] == row["hard_skips"]
        assert rec["coverage_blockers"] == row["blockers"]


@requires_described_corpus
def test_in_envelope_records_are_never_silent_about_control_skips(
        records, coverage_by_case_id):
    """``coverage_verdict`` counts only NON-control skips, so a ``SKIPS(2)`` row
    can hide five control-class families.  An in-envelope record must carry the
    families, so the claim cannot be read unqualified."""
    control_seen = 0
    for rec in records:
        if not rec["in_envelope"]:
            continue
        families = rec["skipped_families"]
        assert families, (
            f"{rec['deck']}: in_envelope with an empty skipped_families — "
            "the coverage evidence says the port skips something here")
        for family, info in families.items():
            assert info["class"], f"{rec['deck']}/{family}: no skip class"
            if info["class"] == CONTROL_CLASS:
                control_seen += 1
        row = coverage_by_case_id[rec["case_id"]]
        control_in_evidence = {f for f, i in row["skipped_families"].items()
                               if i["class"] == CONTROL_CLASS}
        assert control_in_evidence <= set(families), (
            f"{rec['deck']}: control-class skips {sorted(control_in_evidence)} "
            "missing from the record")
    assert control_seen, (
        "no in-envelope deck exercises a control-class skip any more — this "
        "test no longer proves anything and should be retired")


# --------------------------------------------------------------------------
# 6. provenance: the verdict was NOT measured on the bytes we now hash
# --------------------------------------------------------------------------

def test_records_never_claim_the_verdict_bytes_were_verified(records):
    """The M41 sweep ran from a session scratchpad extract that no longer
    exists (``.agents/skills/validation-compare/SKILL.md:56-59``); the official
    parity rows carry no deck path at all.  So a record's sha256 is the hash of
    the RE-VENDORED copy, and it must say so rather than let a consumer read
    the hash as 'the bytes MATCH was measured on'."""
    for rec in records:
        assert rec["parity_run_deck_bytes_verified"] is False, rec["deck"]
        assert rec["coverage_run_deck_bytes_verified"] is False, rec["deck"]


def test_manifest_notes_state_the_scratchpad_provenance(header):
    notes = " ".join(header["notes"]).lower()
    assert "scratchpad" in notes
    assert "verification-compare" in notes or "skill.md" in notes
    assert any("parity_run_deck_bytes_verified" in n
               for n in header["notes"]), (
        "the notes must name the field that carries this caveat")


def test_deck_bytes_match_is_derived_not_asserted(tmp_path):
    """The generator decides ``*_deck_bytes_verified`` by hashing the deck the
    evidence row names; this pins the decision, so a future in-place run flips
    the flag on its own."""
    from tools.build_rd_decks_manifest import deck_bytes_match
    deck = tmp_path / "D_0000.rad"
    deck.write_bytes(b"#RADIOSS\n/END\n")
    digest = hashlib.sha256(deck.read_bytes()).hexdigest()
    assert deck_bytes_match(str(deck), digest) is True
    assert deck_bytes_match(str(deck), "0" * 64) is False
    assert deck_bytes_match(str(tmp_path / "gone.rad"), digest) is False


# --------------------------------------------------------------------------
# 7. joinability with the M41 evidence
# --------------------------------------------------------------------------

@requires_described_corpus
def test_manifest_joins_parity_m41(records, parity_by_case_id):
    mine = {r["case_id"] for r in records if r["case_id"]}
    assert mine & set(parity_by_case_id), (
        "the manifest shares no case_id with parity_m41.json — one of the two "
        "has been regenerated into uselessness")


@requires_described_corpus
def test_joined_records_report_the_parity_verdict(records, parity_by_case_id):
    joined = [r for r in records
              if r["case_id"] and r["case_id"] in parity_by_case_id]
    assert joined
    for rec in joined:
        verdict = parity_by_case_id[rec["case_id"]]
        assert rec["parity_case"] == verdict["case"]
        assert rec["parity_class"] == verdict["class"]
        assert rec["parity_provenance"] == verdict["provenance"]
        assert rec["parity_max_rel_rms"] == verdict["max_rel_rms"]


def test_manifest_joins_coverage_m41(records):
    known = {c["case_id"]: c for c in json.loads(
        COVERAGE_PATH.read_text(encoding="utf-8"))["cases"] if c.get("case_id")}
    mine = {r["case_id"] for r in records if r["case_id"]}
    assert mine & set(known), (
        "no case_id in common with coverage_results_m41.json")
    for rec in records:
        if rec["case_id"] in known:
            assert rec["coverage_verdict"] == known[rec["case_id"]]["verdict"]


def test_counts_declare_measured_and_unmeasured_separately(header):
    counts = header["counts"]
    for key in ("decks", "in_envelope", "measured", "unmeasured",
                "with_case_id", "joined_parity_m41", "joined_coverage_m41",
                "bytes_hashed"):
        assert key in counts, f"counts block is missing {key!r}"
    assert counts["measured"] + counts["unmeasured"] == counts["decks"]
    assert counts["measured"] == sum(1 for r in header["decks"]
                                     if r["parity_class"])
    assert counts["unmeasured"] == sum(1 for r in header["decks"]
                                       if not r["parity_class"])


# --------------------------------------------------------------------------
# 8. the loader needs no environment and no corpus
# --------------------------------------------------------------------------

def test_load_manifest_works_with_no_environment_set(monkeypatch):
    from tools.validate_vs_fortran import load_manifest
    for var in ("PYRADIOSS_RD_DECKS", "PYRADIOSS_HM_CFG", "RAD_CFG_PATH",
                "OR_ROOT", "OR_SRC", "OR_BUILD", "OR_STARTER", "OR_ENGINE"):
        monkeypatch.delenv(var, raising=False)
    paths.reload()
    try:
        records = load_manifest()
        assert records
        assert all(r["deck"] and r["sha256"] for r in records)
    finally:
        paths.reload()


def test_load_manifest_does_not_need_the_corpus(monkeypatch):
    """It returns the *recorded* hashes: no deck is opened, so an absent or
    unreachable corpus cannot change the answer."""
    from tools.validate_vs_fortran import load_manifest

    def _gone():
        raise FileNotFoundError("corpus not mounted")

    monkeypatch.setattr(paths, "rd_decks_dir", _gone)
    records = load_manifest()
    assert records and all(r["sha256"] for r in records)


# --------------------------------------------------------------------------
# 9. the generator: portable --check, and no invented identity
# --------------------------------------------------------------------------

def test_check_is_portable_across_checkout_paths(tmp_path):
    """``--check`` must compare what the corpus CONTAINS, not where this
    checkout happens to live: a byte-identical copy of the vendored corpus at
    another absolute path has to verify clean (M1)."""
    from tools import build_rd_decks_manifest as gen
    copy = _copy_corpus(tmp_path / "somewhere_else")
    manifest_copy = tmp_path / "rd_decks_manifest.json"
    manifest_copy.write_bytes(MANIFEST_PATH.read_bytes())
    assert gen.main(["--check", "--root", str(copy),
                     "--out", str(manifest_copy)]) == 0


def test_check_detects_a_changed_deck_in_a_copy(tmp_path):
    """...and must still fail when the copy is NOT identical."""
    from tools import build_rd_decks_manifest as gen
    copy = _copy_corpus(tmp_path / "tampered")
    sorted(copy.rglob("*_0000.rad"))[0].write_bytes(b"#RADIOSS\n/END\n")
    manifest_copy = tmp_path / "rd_decks_manifest.json"
    manifest_copy.write_bytes(MANIFEST_PATH.read_bytes())
    assert gen.main(["--check", "--root", str(copy),
                     "--out", str(manifest_copy)]) == 1


def test_check_names_one_changed_deck_not_seventy_five(tmp_path, capsys):
    """A one-byte corpus change must not print a ``deck changed`` line per
    record: every record carries the corpus fingerprint, so the noise buries the
    one fact that matters (N4)."""
    from tools import build_rd_decks_manifest as gen
    copy = _copy_corpus(tmp_path / "one_byte")
    victim = sorted(copy.rglob("*_0000.rad"))[0]
    victim.write_bytes(b"#RADIOSS\n/END\n")
    manifest_copy = tmp_path / "rd_decks_manifest.json"
    manifest_copy.write_bytes(MANIFEST_PATH.read_bytes())
    assert gen.main(["--check", "--root", str(copy),
                     "--out", str(manifest_copy)]) == 1
    err = capsys.readouterr().err
    changed = [ln for ln in err.splitlines() if "deck changed:" in ln]
    assert len(changed) == 1, f"expected one deck-changed line, got {len(changed)}"
    assert victim.name in changed[0]
    assert "sha256" in changed[0]
    assert "corpus_fingerprint" not in changed[0], (
        "the per-deck line must not repeat the corpus-level fact")
    assert "fingerprint" in err.lower()


@pytest.mark.parametrize("key", ["bytes_verified_rule", "envelope_rule",
                                 "hashed_file_rule", "hash_algorithm",
                                 "joins", "notes"])
def test_check_compares_the_header_prose(tmp_path, key):
    """The header prose is what a reader quotes, so deleting a rule must be
    drift ``--check`` reports — it used to compare neither (N2)."""
    from tools import build_rd_decks_manifest as gen
    doc = _doc()
    doc.pop(key)
    manifest_copy = _write_doc(tmp_path, doc, f"no_{key}.json")
    assert gen.main(["--check", "--root", str(_resolved_corpus()),
                     "--out", manifest_copy]) == 1


def test_check_compares_the_header_prose_content(tmp_path):
    """Not just presence: reworded prose is drift too."""
    from tools import build_rd_decks_manifest as gen
    doc = _doc()
    doc["bytes_verified_rule"] = "everything is fine, trust me"
    manifest_copy = _write_doc(tmp_path, doc, "reworded.json")
    assert gen.main(["--check", "--root", str(_resolved_corpus()),
                     "--out", manifest_copy]) == 1


def test_check_compares_the_generated_envelope_note(tmp_path):
    """The envelope-qualification note is generated FROM the records, so it is
    covered by the ``decks`` comparison only if ``notes`` is compared too."""
    from tools import build_rd_decks_manifest as gen
    doc = _doc()
    doc["notes"] = [n for n in doc["notes"]
                    if "in-envelope decks still skip" not in n]
    manifest_copy = _write_doc(tmp_path, doc, "no_envelope_note.json")
    assert gen.main(["--check", "--root", str(_resolved_corpus()),
                     "--out", manifest_copy]) == 1


def test_generator_nulls_the_identity_of_an_uncatalogued_deck(tmp_path):
    """``case_id``/``category``/``package`` are carried over from
    inventory.json or they are null — a deck the inventory never saw must not
    borrow an id, and must not be called in-envelope without a verdict."""
    from tools import build_rd_decks_manifest as gen

    corpus = tmp_path / "rd_decks"
    deck = corpus / "rd_e" / "NEW_PKG" / "THING_0000.rad"
    deck.parent.mkdir(parents=True)
    deck.write_bytes(b"#RADIOSS\n/END\n")
    (corpus / "rd_e" / "NEW_PKG" / "THING_0001.rad").write_bytes(b"engine\n")

    doc = gen.build(str(corpus))
    assert len(doc["decks"]) == 1, "only the starter deck is a case"
    rec = doc["decks"][0]
    assert rec["deck"] == "rd_e/NEW_PKG/THING_0000.rad"
    assert rec["case_id"] is None
    assert rec["category"] is None and rec["package"] is None
    assert rec["in_envelope"] is False
    assert "no parity_m41.json row" in rec["in_envelope_source"]
    assert rec["sha256"] == hashlib.sha256(deck.read_bytes()).hexdigest()
    assert rec["size_bytes"] == len(b"#RADIOSS\n/END\n")
    assert rec["parity_run_deck_bytes_verified"] is False
    assert rec["coverage_run_deck_bytes_verified"] is False
    assert rec["skipped_families"] == {}
    assert rec["corpus_fingerprint"] == doc["corpus_root"]["fingerprint"]