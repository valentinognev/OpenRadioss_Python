"""Task P0.8 — the hashed corpus manifest (``rd_decks_manifest.json``).

The Phase 0 exit gate and every later phase's "numerics changes require
validation evidence" rule need one file that says, without hand-editing:

* **which** deck files are covered by the differential-validation evidence,
* **what they hash to** (so a later run can prove it ran the same bytes),
* **whether** the port is expected to reproduce the oracle on each of them.

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
3. **identity** — ``case_id`` values are unique and are the strings the
   evidence files already use (no locally invented ids);
4. **honesty of the envelope flag** — ``in_envelope`` is ``true`` only where a
   measured parity verdict says ``MATCH``; a statically predicted
   ``inventory.json`` classification is never enough;
5. **joinability** — the manifest intersects ``parity_m41.json`` and
   ``coverage_results_m41.json`` on ``case_id``, so neither side can be
   regenerated into uselessness without a test failing;
6. **no environment, no corpus** — ``load_manifest()`` returns the *recorded*
   hashes without exporting anything or touching a deck, so it works on a
   read-only share and with no corpus at all.

The corpus root is ``pyradioss.paths.rd_decks_dir()`` (``PYRADIOSS_RD_DECKS``
else the vendored ``tests/data/rd_decks``).  The manifest records which corpus
it describes; if the environment points at a *different* extract, the
whole-corpus properties (1)/(2) skip instead of lying — such a corpus has to be
re-hashed with ``tools/build_rd_decks_manifest.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pyradioss import paths

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tools" / "validation_data"
MANIFEST_PATH = DATA / "rd_decks_manifest.json"
PARITY_PATH = DATA / "parity_m41.json"
INVENTORY_PATH = DATA / "inventory.json"
COVERAGE_PATH = DATA / "coverage_results_m41.json"

REQUIRED_FIELDS = (
    "case_id", "deck", "hashed_file", "sha256", "size_bytes", "category",
    "package", "in_envelope", "in_envelope_source", "in_envelope_reason",
    "inventory_classification", "parity_case", "parity_class",
    "parity_max_rel_rms", "coverage_verdict",
)


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


def _describes_resolved_corpus() -> bool:
    """True when ``paths.rd_decks_dir()`` is the corpus the manifest hashed."""
    vendored = _doc().get("corpus_root", {}).get("vendored")
    if not vendored:
        return False
    try:
        return _resolved_corpus().resolve() == (REPO / vendored).resolve()
    except (FileNotFoundError, OSError):
        return False


requires_vendored_corpus = pytest.mark.skipif(
    not _describes_resolved_corpus(),
    reason="rd_decks_dir() is not the corpus this manifest describes "
           "(PYRADIOSS_RD_DECKS override?); re-hash that extract with "
           "tools/build_rd_decks_manifest.py")


@pytest.fixture(scope="module")
def records() -> list:
    return _manifest()


@pytest.fixture(scope="module")
def parity_by_case_id() -> dict:
    doc = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    return {r["case_id"]: r for r in doc["results"] if r.get("case_id")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 1. coverage: one record per starter deck
# --------------------------------------------------------------------------

@requires_vendored_corpus
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

@requires_vendored_corpus
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
# 3. identity: unique, and the same ids the evidence files use
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


def test_records_carry_the_documented_field_set(records):
    for rec in records:
        for field in REQUIRED_FIELDS:
            assert field in rec, f"{rec.get('deck')}: missing field {field!r}"
        assert isinstance(rec["in_envelope"], bool)
        assert isinstance(rec["size_bytes"], int) and rec["size_bytes"] > 0
        assert rec["deck"].endswith("_0000.rad"), rec["deck"]


# --------------------------------------------------------------------------
# 4. the envelope flag is measured, never assumed
# --------------------------------------------------------------------------

@requires_vendored_corpus
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


# --------------------------------------------------------------------------
# 5. joinability with the M41 evidence
# --------------------------------------------------------------------------

@requires_vendored_corpus
def test_manifest_joins_parity_m41(records, parity_by_case_id):
    mine = {r["case_id"] for r in records if r["case_id"]}
    assert mine & set(parity_by_case_id), (
        "the manifest shares no case_id with parity_m41.json — one of the two "
        "has been regenerated into uselessness")


@requires_vendored_corpus
def test_joined_records_report_the_parity_verdict(records, parity_by_case_id):
    joined = [r for r in records
              if r["case_id"] and r["case_id"] in parity_by_case_id]
    assert joined
    for rec in joined:
        verdict = parity_by_case_id[rec["case_id"]]
        assert rec["parity_case"] == verdict["case"]
        assert rec["parity_class"] == verdict["class"]
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


# --------------------------------------------------------------------------
# 6. the loader needs no environment and no corpus
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
# 7. the generator: an uncatalogued deck must not acquire an invented identity
# --------------------------------------------------------------------------

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

    doc = gen.build(str(corpus), str(tmp_path / "manifest.json"))
    assert len(doc["decks"]) == 1, "only the starter deck is a case"
    rec = doc["decks"][0]
    assert rec["deck"] == "rd_e/NEW_PKG/THING_0000.rad"
    assert rec["case_id"] is None
    assert rec["category"] is None and rec["package"] is None
    assert rec["in_envelope"] is False
    assert "no parity_m41.json row" in rec["in_envelope_source"]
    assert rec["sha256"] == hashlib.sha256(deck.read_bytes()).hexdigest()
    assert rec["size_bytes"] == len(b"#RADIOSS\n/END\n")