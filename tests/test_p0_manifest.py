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
   read-only share and with no corpus at all;
9. **the described corpus still hashes to the recorded fingerprint** — every
   record's ``sha256`` and ``size_bytes`` are re-hashed from
   ``manifest_corpus_root()`` on every run, which is what keeps the whole
   corpus covered instead of skipped.

Which corpus the assertions read
--------------------------------
The corpus every corpus-dependent assertion reads is
``tools.validate_vs_fortran.manifest_corpus_root()`` — the corpus **the
manifest describes**, i.e. the vendored ``tests/data/rd_decks`` in this
checkout.  It is in-tree, so it is always there: no test in this module skips
for want of a corpus, in any configuration, including the sanctioned
``PYRADIOSS_RD_DECKS=<full E: extract>`` setup.

``paths.rd_decks_dir()`` — the corpus the *environment* selects — is used in
exactly one place, :func:`_live_corpus_report`, which is a **report, never a
failure**: it states both corpora with their fingerprints and how many of the
live corpus' decks have no record, and it is pinned by asserting its own text.
A live corpus that is not the described one is something to *say*, not to
fail on: the workaround is a different corpus, not a broken artifact.

Regenerating the manifest is only ever safe for the vendored corpus, and the
generator enforces that (it refuses to overwrite
``tools/validation_data/rd_decks_manifest.json`` from any other root without
an explicit ``--out``); see
``test_generator_refuses_to_overwrite_the_committed_manifest``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import warnings
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
GENERATOR = REPO / "tools" / "build_rd_decks_manifest.py"

#: The command that is safe in every configuration: an explicit root, so the
#: environment cannot redirect it, and a corpus the manifest may describe.
REGENERATE_VENDORED = ("python tools/build_rd_decks_manifest.py --root "
                       "tests/data/rd_decks")

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


def _described_corpus() -> Path:
    """The corpus the manifest describes — what every assertion below reads."""
    from tools.validate_vs_fortran import manifest_corpus_root
    return Path(manifest_corpus_root())


def _live_corpus() -> Path:
    """The corpus the ENVIRONMENT selects (``PYRADIOSS_RD_DECKS`` or vendored).

    Only :func:`_live_corpus_report` looks at this one, and it never fails on
    what it finds.
    """
    return paths.rd_decks_dir()


def _live_corpus_report(live: Path = None, header: dict = None) -> str:
    """State the live-vs-described relationship.  A report, never a verdict.

    Says which corpus the committed manifest describes (path + fingerprint),
    which corpus is live (path + fingerprint), how many of the live corpus'
    starter decks have no record, and — branched, because the two situations
    need opposite actions — how to act:

    * the **vendored** corpus changed → re-hash *it*, with
      ``PYRADIOSS_RD_DECKS`` unset (``--root tests/data/rd_decks``);
    * a **wider extract** is simply selected → that is a workaround, not a
      defect.  If an extract-scoped manifest is really wanted, write it to an
      explicit ``--out`` path and do NOT commit it over the committed file,
      which would null ``corpus_root.vendored`` and break
      ``manifest_corpus_root()`` for everyone else.
    """
    from tools.validate_vs_fortran import corpus_fingerprint, manifest_corpus_root
    live = Path(live) if live is not None else _live_corpus()
    header = header if header is not None else _doc()
    described = manifest_corpus_root(header)
    described_fp = (header.get("corpus_root") or {}).get("fingerprint", "?")
    covered = {r["deck"] for r in header.get("decks", [])}
    try:
        live_fp = corpus_fingerprint(str(live))
    except (ValueError, OSError) as exc:
        return (f"live corpus {live}: cannot fingerprint ({exc}); the manifest "
                f"describes {described} ({described_fp})")
    on_disk = {p.relative_to(live).as_posix()
               for p in live.rglob("*_0000.rad")} if live.is_dir() else set()
    uncovered = sorted(on_disk - covered)
    lines = [
        f"committed manifest describes: {described} ({described_fp})",
        f"live corpus (paths.rd_decks_dir()): {live} ({live_fp})",
        f"decks in the live corpus with no manifest record: {len(uncovered)}",
    ]
    if live_fp == described_fp:
        lines.append(
            "in sync (a byte-identical copy of the described corpus is still "
            "the described corpus)")
        return "\n".join(lines)
    lines.append(
        "DIVERGED - nothing is broken, but the two corpora are not the same:")
    if uncovered:
        lines.append(f"  uncovered decks: {', '.join(uncovered[:5])}"
                     + (" ..." if len(uncovered) > 5 else ""))
    lines.append(
        "  if the VENDORED corpus changed, re-hash it with the environment "
        f"unset:\n      unset PYRADIOSS_RD_DECKS; {REGENERATE_VENDORED}")
    lines.append(
        "  if you want a manifest for THIS extract, write it to your own "
        "path and do not commit it over\n      "
        f"{MANIFEST_PATH.relative_to(REPO)}: that file must keep "
        f"corpus_root.vendored, or manifest_corpus_root()\n      "
        "stops resolving for every other consumer. Use:\n      "
        "      python tools/build_rd_decks_manifest.py --root "
        f"{live} --out <private path>")
    return "\n".join(lines)


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


@pytest.fixture
def committed_manifest():
    """Snapshot the committed manifest and restore it afterwards.

    The generator-refusal tests aim a *harmful* command at that file on
    purpose.  If the guard ever regresses, the artifact must not be left
    damaged by a test run.
    """
    before = MANIFEST_PATH.read_bytes()
    try:
        yield before
    finally:
        if MANIFEST_PATH.read_bytes() != before:
            MANIFEST_PATH.write_bytes(before)


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
        f"no manifest at {MANIFEST_PATH}; generate it with\n"
        f"  {REGENERATE_VENDORED}\n"
        "(an explicit --root, so PYRADIOSS_RD_DECKS cannot redirect it)")
    doc = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    from tools.validate_vs_fortran import MANIFEST_SCHEMA
    assert doc["schema"] == MANIFEST_SCHEMA
    assert doc["decks"], "the manifest has no deck records"


def test_live_corpus_report_states_both_corpora_when_in_sync(header):
    """The live-vs-described relationship is a REPORT; its text is pinned.

    In the default configuration the live corpus IS the described corpus, and
    the report says exactly that — no verdict, nothing skipped.
    """
    text = _live_corpus_report(live=VENDORED, header=header)
    assert str(VENDORED) in text
    assert header["corpus_root"]["fingerprint"] in text
    assert "decks in the live corpus with no manifest record: 0" in text
    assert "in sync" in text
    assert "DIVERGED" not in text


def test_live_corpus_report_branches_the_advice_when_it_diverges(header,
                                                                  tmp_path):
    """A different extract must produce a report that says BOTH corpora and
    then branches the remedy, because the two situations need opposite actions.

    The advice it must NOT give is the one round 2 gave: re-hash the *live*
    extract over the committed manifest, which nulls ``corpus_root.vendored``
    and breaks ``manifest_corpus_root()`` for every other consumer.
    """
    foreign = tmp_path / "extract"
    (foreign / "rd_e" / "SOME_PKG").mkdir(parents=True)
    (foreign / "rd_e" / "SOME_PKG" / "X_0000.rad").write_bytes(b"#RADIOSS\n")
    text = _live_corpus_report(live=foreign, header=header)

    # both corpora, with their fingerprints, and the uncovered count
    assert str(foreign) in text
    assert header["corpus_root"]["fingerprint"] in text
    assert "DIVERGED" in text
    assert "decks in the live corpus with no manifest record: 1" in text
    assert "rd_e/SOME_PKG/X_0000.rad" in text

    # branch 1: the vendored corpus changed -> re-hash IT, env unset
    assert REGENERATE_VENDORED in text
    assert "unset PYRADIOSS_RD_DECKS" in text
    # branch 2: an extract-scoped manifest goes to a private --out, never over
    # the committed file
    assert "--out <private path>" in text
    assert "do not commit it over" in text
    assert str(MANIFEST_PATH.relative_to(REPO)) in text
    assert "corpus_root.vendored" in text
    # ...and it must NOT tell the reader to re-hash the live corpus in place
    assert f"--root {foreign} --out {MANIFEST_PATH}" not in text


def test_live_corpus_report_survives_an_absent_live_corpus(header, tmp_path):
    """A corpus that cannot be read is reported, not raised."""
    text = _live_corpus_report(live=tmp_path / "not_mounted", header=header)
    assert "cannot fingerprint" in text or "DIVERGED" in text


def test_described_corpus_is_the_vendored_one(header, monkeypatch, tmp_path):
    """The corpus every other assertion reads is in-tree, so nothing skips.

    Pinned under an override, because that is where it matters: with
    ``PYRADIOSS_RD_DECKS`` pointing somewhere else, the assertions must still
    read the corpus the manifest describes.
    """
    assert _described_corpus().resolve() == VENDORED.resolve()
    assert header["corpus_root"]["vendored"] == "tests/data/rd_decks"
    foreign = tmp_path / "extract"
    (foreign / "rd_e" / "SOME_PKG").mkdir(parents=True)
    (foreign / "rd_e" / "SOME_PKG" / "X_0000.rad").write_bytes(b"#RADIOSS\n")
    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(foreign))
    paths.reload()
    try:
        assert _live_corpus().resolve() == foreign.resolve()
        assert _described_corpus().resolve() == VENDORED.resolve(), (
            "the corpus the assertions read must not follow "
            "PYRADIOSS_RD_DECKS — that is how eight tests used to skip")
    finally:
        monkeypatch.delenv("PYRADIOSS_RD_DECKS", raising=False)
        paths.reload()


def test_this_module_has_no_skip_markers():
    """0 skipped in every configuration, by construction.

    The whole-corpus assertions read an in-tree corpus, so nothing here may
    reintroduce a conditional skip: a skip would silently turn a coverage
    regression into a green run.  Checked on the AST, so this test's own
    docstring (which has to name the markers) does not trip it.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    marks = sorted({node.attr for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                    and node.attr in ("skip", "skipif", "skiplib")})
    assert not marks, f"conditional skips reintroduced: {marks}"


def test_live_corpus_divergence_is_surfaced_to_the_operator():
    """A report nobody reads is not a report: emit it on every run, so a
    ``PYRADIOSS_RD_DECKS`` override that quietly changes the corpus is visible
    in the warnings summary instead of nowhere."""
    text = _live_corpus_report()
    assert text
    warnings.warn(text, RuntimeWarning, stacklevel=1)




# --------------------------------------------------------------------------
# 1. coverage: one record per starter deck
# --------------------------------------------------------------------------

def test_manifest_has_exactly_one_record_per_starter_deck(records):
    root = _described_corpus()
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*_0000.rad")}
    assert on_disk, f"no starter decks found under {root}"
    in_manifest = [r["deck"] for r in records]
    assert len(in_manifest) == len(set(in_manifest)), "duplicate deck records"
    assert set(in_manifest) == on_disk, (
        "manifest/corpus drift: "
        f"only in manifest {sorted(set(in_manifest) - on_disk)}, "
        f"only on disk {sorted(on_disk - set(in_manifest))}")


# --------------------------------------------------------------------------
# 2. integrity: every recorded hash is the hash of the bytes on disk
# --------------------------------------------------------------------------

def test_every_recorded_sha256_matches_the_file_on_disk(records):
    root = _described_corpus()
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

def test_manifest_joins_parity_m41(records, parity_by_case_id):
    mine = {r["case_id"] for r in records if r["case_id"]}
    assert mine & set(parity_by_case_id), (
        "the manifest shares no case_id with parity_m41.json — one of the two "
        "has been regenerated into uselessness")


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
    assert gen.main(["--check", "--root", str(_described_corpus()),
                     "--out", manifest_copy]) == 1


def test_check_compares_the_header_prose_content(tmp_path):
    """Not just presence: reworded prose is drift too."""
    from tools import build_rd_decks_manifest as gen
    doc = _doc()
    doc["bytes_verified_rule"] = "everything is fine, trust me"
    manifest_copy = _write_doc(tmp_path, doc, "reworded.json")
    assert gen.main(["--check", "--root", str(_described_corpus()),
                     "--out", manifest_copy]) == 1


def test_check_compares_the_generated_envelope_note(tmp_path):
    """The envelope-qualification note is generated FROM the records, so it is
    covered by the ``decks`` comparison only if ``notes`` is compared too."""
    from tools import build_rd_decks_manifest as gen
    doc = _doc()
    doc["notes"] = [n for n in doc["notes"]
                    if "in-envelope decks still skip" not in n]
    manifest_copy = _write_doc(tmp_path, doc, "no_envelope_note.json")
    assert gen.main(["--check", "--root", str(_described_corpus()),
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

# --------------------------------------------------------------------------
# 10. the generator must not damage the committed manifest
# --------------------------------------------------------------------------

def _synthetic_corpus(tmp_path: Path, decks: int = 2) -> Path:
    """A tiny foreign corpus: NOT the vendored one, so writing the committed
    manifest from it would null ``corpus_root.vendored``."""
    corpus = tmp_path / "synthetic" / "rd_decks"
    for i in range(decks):
        deck = corpus / "rd_e" / "SYN" / f"D{i}_0000.rad"
        deck.parent.mkdir(parents=True, exist_ok=True)
        deck.write_bytes(f"#RADIOSS\n/D{i}\n/END\n".encode())
    return corpus


def test_generator_refuses_to_overwrite_the_committed_manifest(tmp_path,
                                                               capsys,
                                                               committed_manifest):
    """The harmful step must not be one command away.

    Regenerating the committed manifest from a foreign corpus writes
    ``corpus_root.vendored: null`` and an absolute
    ``resolved_at_generation``, which silently breaks
    ``manifest_corpus_root()`` — and therefore every consumer — in the
    committed file.  So the generator refuses: an explicit ``--out`` (a private
    path) or an explicit ``--allow-nonportable`` is required.

    ``committed_manifest`` snapshots and restores the committed file: this test
    deliberately aims a harmful command at it, so a regression of the guard
    must not be able to leave the artifact damaged.
    """
    from tools import build_rd_decks_manifest as gen
    corpus = _synthetic_corpus(tmp_path)
    assert gen.main(["--root", str(corpus)]) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert "--out" in err and "--allow-nonportable" in err
    assert str(corpus) in err
    assert "corpus_root.vendored: null" in err
    # the committed manifest must be untouched by the refusal
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == _doc()


def test_generator_refuses_the_documented_bare_command_under_an_override(
        tmp_path, monkeypatch, capsys, committed_manifest):
    """Round 2's remedy was exactly this: ``--root <live dir>`` (or the bare
    command with ``PYRADIOSS_RD_DECKS`` set) writes the committed file.  Both
    must refuse now — including the bare command, which is what the failure
    message used to advise."""
    from tools import build_rd_decks_manifest as gen
    corpus = _synthetic_corpus(tmp_path)
    monkeypatch.setenv("PYRADIOSS_RD_DECKS", str(corpus))
    paths.reload()
    try:
        assert gen.main([]) == 2                      # the bare command
        assert "REFUSING" in capsys.readouterr().err
        assert gen.main(["--root", str(corpus)]) == 2  # the round-2 remedy
        err = capsys.readouterr().err
        assert "REFUSING" in err
        assert "--root tests/data/rd_decks" in err, (
            "the refusal must point at the vendored corpus, not at the live one")
    finally:
        monkeypatch.delenv("PYRADIOSS_RD_DECKS", raising=False)
        paths.reload()
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == _doc()


def test_generator_allows_an_explicit_private_out(tmp_path):
    """An extract-scoped manifest is legitimate — into its OWN file."""
    from tools import build_rd_decks_manifest as gen
    corpus = _synthetic_corpus(tmp_path)
    out = tmp_path / "extract_manifest.json"
    assert gen.main(["--root", str(corpus), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["corpus_root"]["vendored"] is None      # honest about itself
    assert doc["corpus_root"]["fingerprint"]
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == _doc()


def test_generator_allows_an_explicit_override_flag(tmp_path):
    """``--allow-nonportable`` is the deliberate way to say 'yes, I mean the
    committed file'."""
    from tools import build_rd_decks_manifest as gen
    corpus = _synthetic_corpus(tmp_path)
    out = tmp_path / "forced.json"
    assert gen.main(["--root", str(corpus), "--out", str(out),
                     "--allow-nonportable"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["decks"]


def test_check_is_still_allowed_for_any_corpus(tmp_path):
    """The refusal guards WRITING the committed file, not verifying one: a
    portable ``--check`` against a copy must keep working."""
    from tools import build_rd_decks_manifest as gen
    copy = _copy_corpus(tmp_path)
    out = tmp_path / "manifest.json"
    assert gen.main(["--root", str(copy), "--out", str(out)]) == 0
    assert gen.main(["--check", "--root", str(copy), "--out", str(out)]) == 0


def test_check_message_counts_the_records_it_collapsed(tmp_path, capsys):
    """The corpus-fingerprint message must derive its 'not N' from the corpus,
    not hardcode 75 (N8)."""
    from tools import build_rd_decks_manifest as gen
    corpus = _synthetic_corpus(tmp_path, decks=3)
    out = tmp_path / "m.json"
    assert gen.main(["--root", str(corpus), "--out", str(out)]) == 0
    sorted(corpus.rglob("*_0000.rad"))[0].write_bytes(b"#RADIOSS\n/END\n")
    assert gen.main(["--check", "--root", str(corpus), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "not 3" in err, err
    assert "not 75" not in err
