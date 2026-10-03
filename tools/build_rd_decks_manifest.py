#!/usr/bin/env python
"""Generate ``tools/validation_data/rd_decks_manifest.json`` (Phase 0, P0.8).

Why this exists
---------------
The differential-validation evidence files (``inventory.json``,
``parity_m41.json``, ``coverage_results_m41.json``) are keyed by a
hand-collected list of ``case_id`` strings, and nothing on disk says which
*bytes* those cases were.  A green parity table therefore cannot prove which
deck it ran.  The manifest closes that gap: one record per starter deck under
``pyradioss.paths.rd_decks_dir()``, carrying

* ``deck`` / ``hashed_file`` / ``sha256`` / ``size_bytes`` — what was covered
  and exactly which bytes were hashed (``hashed_file`` is ``deck`` for every
  record; the field exists so a record that hashes something else has to say
  so), plus ``corpus_fingerprint`` — *which* corpus those bytes live in, since
  ``deck`` is a relative path and the corpus root comes from the environment;
* ``case_id`` / ``category`` / ``package`` — carried over from
  ``inventory.json`` when the deck is a case it knows, else ``null`` (ids are
  never invented, so ``parity_m41.json`` stays joinable);
* the evidence verdicts: ``parity_*`` from ``parity_m41.json`` and the reader
  census (``coverage_verdict``, ``skipped_families``, ``coverage_hard_skips``,
  ``coverage_blockers``, ``coverage_degrade_warnings``) from
  ``coverage_results_m41.json``;
* ``in_envelope`` **plus** ``in_envelope_source`` / ``in_envelope_reason`` —
  true only where a *measured* parity verdict is ``MATCH``.  A statically
  predicted ``inventory.json`` classification (``IN_ENVELOPE``, i.e. "the port
  should read this deck without skipping a family") is recorded as
  ``inventory_classification`` but never flips the flag: that would make the
  manifest claim more than the oracle can support.  Everything else is false
  with the reason recorded;
* ``parity_run_deck_bytes_verified`` / ``coverage_run_deck_bytes_verified`` —
  **false for every M41 row**.  The sweep ran from a session scratchpad that
  no longer exists (``.agents/skills/validation-compare/SKILL.md:56-59``), and
  the official parity rows carry no deck path at all, so each record pairs the
  hash of the *re-vendored* copy with a verdict measured on the scratchpad
  copy.  The flag is derived (does the evidence row name a file whose bytes
  hash to ours?), never asserted, so a future in-place run flips it itself.

Only the starter deck (``*_0000.rad``) is a *case*: the engine decks are
restart continuations of the same case and are hashed only insofar as the
parent deck identifies them.

Usage
-----
    python tools/build_rd_decks_manifest.py            # (re)generate
    python tools/build_rd_decks_manifest.py --check    # verify, write nothing
    python tools/build_rd_decks_manifest.py --root <corpus> --out <file>

``--check`` compares what the corpus **contains** — the fingerprint, the
counts and the records — never where this checkout lives, so a byte-identical
copy at another absolute path verifies clean; it exits non-zero when a deck,
a hash or a verdict moved.  ``tests/test_p0_manifest.py`` re-verifies the
hashes themselves.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pyradioss import paths
from tools.validate_vs_fortran import (
    MANIFEST_FIELDS,
    MANIFEST_SCHEMA,
    corpus_fingerprint,
)

VALIDATION_DATA = os.path.join(REPO, "tools", "validation_data")
DEFAULT_OUT = os.path.join(VALIDATION_DATA, "rd_decks_manifest.json")
INVENTORY = os.path.join(VALIDATION_DATA, "inventory.json")
PARITY = os.path.join(VALIDATION_DATA, "parity_m41.json")
COVERAGE = os.path.join(VALIDATION_DATA, "coverage_results_m41.json")

#: The one measured verdict that puts a deck inside the validated envelope.
#: Everything else (DEVIATION, NO-CHANNELS, SKIPPED-SLOW, PYRADIOSS-FAIL,
#: PORT-ONLY*) is false — those rows say the port did not reproduce the oracle,
#: or that nothing was compared at all.
ENVELOPE_CLASS = "MATCH"

#: Where the M41 verdicts were actually measured, and why that is not where the
#: bytes now hashed live.  Quoted in the manifest notes so a reader cannot
#: mistake ``sha256`` for "the bytes MATCH was measured on".
PROVENANCE_CAVEAT = (
    "The M41 sweep was run from a session scratchpad extract that no longer "
    "exists (.agents/skills/validation-compare/SKILL.md:56-59: 'The 529-deck "
    "official-corpus sweep has NO in-repo driver (it lived in a session "
    "scratchpad) ... root points at a dead scratchpad'), and the official "
    "parity_m41.json rows carry no deck path at all (reference_used: null; "
    "their Fortran CSVs were carried forward from M37). Every record therefore "
    "pairs the sha256 of the RE-VENDORED copy of a deck with a verdict measured "
    "on the scratchpad copy: parity_run_deck_bytes_verified and "
    "coverage_run_deck_bytes_verified are false on all of them, and the flag is "
    "derived (does the evidence row name a file whose bytes hash to this "
    "record's sha256?) rather than asserted. A MATCH verdict here means 'the "
    "port reproduced the oracle on this deck as it stood in the M41 sweep', "
    "NOT 'these exact bytes were compared'."
)

NOTES = [
    "sha256 is the hash of hashed_file; hashed_file equals deck for every "
    "record (the starter deck itself), stated per record so the hashed bytes "
    "are never a guess.",
    "deck is RELATIVE to a corpus root, so every record also carries "
    "corpus_fingerprint (sha256 over the corpus' starter-deck paths and their "
    "hashes, path-independent). Resolve a record with "
    "tools.validate_vs_fortran.resolve_manifest_record(), which refuses a "
    "corpus whose fingerprint differs; never join rd_decks_dir() / rec['deck'] "
    "directly, a PYRADIOSS_RD_DECKS override would silently read a different "
    "file at the same relative path.",
    "in_envelope is true only where parity_m41.json carries a measured "
    f"verdict of {ENVELOPE_CLASS} for this exact case_id; it means 'the port "
    "reproduced the oracle on this deck'.",
    "inventory_classification is the STATIC reading prediction from "
    "inventory.json (IN_ENVELOPE = no unsupported keyword family). It is "
    "recorded for context and never sets in_envelope.",
    "skipped_families (with a class per family: 'control' or 'soft'), "
    "coverage_hard_skips and coverage_blockers qualify every verdict on this "
    "record: coverage_verdict counts only NON-control skips, so a 'SKIPS(2)' "
    "row can hide five control-class families. The note below lists exactly "
    "which families the in-envelope decks skip, so in_envelope is never read "
    "unqualified.",
    "case_id / category / package are carried over from inventory.json; null "
    "when the corpus holds a deck inventory.json never catalogued. Ids are "
    "never invented.",
    "Admissible oracle evidence channels are listed in "
    "tools/validation_data/oracle_provenance.json; a deck whose parity verdict "
    "leans on an inadmissible channel (H3D, the starter .out include list, "
    "/ALE/STRUCTURED_MESH, /CHECKSUM_REPORT, native .k reading) must not be "
    "recorded as MATCH - none of the vendored decks does.",
    "One record per STARTER deck. Engine decks (*_0001.rad...) are restart "
    "continuations of the same case, not separate cases.",
    PROVENANCE_CAVEAT,
]


def _load(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def starter_decks(root: str) -> List[str]:
    """Every ``*_0000.rad`` under ``root``, as POSIX paths relative to it."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith("_0000.rad"):
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def deck_bytes_match(deck_path: Optional[str], sha256: str) -> bool:
    """True only when the evidence row names an existing file whose bytes hash
    to ``sha256``.

    This is the *derivation* behind ``parity_run_deck_bytes_verified`` /
    ``coverage_run_deck_bytes_verified``.  A row with no deck path, a dead path
    (the M41 scratchpad) or different bytes all answer False; a future in-place
    run answers True without anyone editing a flag.
    """
    if not deck_path or not os.path.isfile(deck_path):
        return False
    return _sha256(deck_path) == sha256


def _evidence_deck(row: dict) -> Optional[str]:
    """The deck path an evidence row names, if it names one at all."""
    for key in ("deck", "deck_path", "starter_deck", "deck0"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _envelope_qualification(records: List[dict]) -> str:
    """A note built FROM the records, never hand-written.

    A hand-written claim about which in-envelope decks skip what goes stale the
    moment a parity verdict moves; this one cannot.
    """
    inside = [r for r in records if r["in_envelope"]]
    if not inside:
        return ("No deck is currently in the validated envelope, so there is "
                "nothing to qualify.")
    groups: Dict[tuple, List[str]] = {}
    for rec in inside:
        families = tuple(sorted(rec["skipped_families"]))
        groups.setdefault(families, []).append(rec["deck"])
    parts = []
    for families, decks in sorted(groups.items(), key=lambda kv: (-len(kv[1]),
                                                                  kv[0])):
        members = [rec for rec in inside if rec["deck"] in decks]
        classes = sorted({info["class"] for rec in members
                          for info in rec["skipped_families"].values()})
        listed = ", ".join(families) if families else "nothing"
        parts.append(f"{len(decks)} of the {len(inside)} in-envelope decks skip "
                     f"[{listed}] (class {', '.join(classes)})")
    return ("What the in-envelope decks still skip — in_envelope is a "
            "reproduction claim about the COMPARED CHANNELS, not a "
            "full-reading claim: " + "; ".join(parts) + ".")


def build(root: str, source: Optional[str] = None) -> dict:
    inventory = {c["case_id"]: c for c in _load(INVENTORY).get("cases", [])}
    parity_rows = _load(PARITY).get("results", [])
    parity_by_id: Dict[str, dict] = {}
    for row in parity_rows:
        cid = row.get("case_id")
        if cid:
            parity_by_id[cid] = row
    coverage_by_id = {c["case_id"]: c for c in _load(COVERAGE).get("cases", [])
                      if c.get("case_id")}

    root = os.path.abspath(root)
    fingerprint = corpus_fingerprint(root)

    records = []
    for rel in starter_decks(root):
        full = os.path.join(root, rel.replace("/", os.sep))
        case = inventory.get(rel)
        verdict = parity_by_id.get(rel) if case else None
        coverage = coverage_by_id.get(rel) if case else None
        digest = _sha256(full)

        if verdict is not None and verdict["class"] == ENVELOPE_CLASS:
            in_envelope = True
            source_row = (f"parity_m41.json results[case={verdict['case']!r}]."
                          f"class == {ENVELOPE_CLASS} "
                          f"(max_rel_rms={verdict['max_rel_rms']!r}, "
                          f"provenance={verdict.get('provenance')!r})")
            reason = ("measured differential parity against the Fortran "
                      "oracle: every compared channel within the M41 "
                      "tolerance. See skipped_families for the families the "
                      "port still skips on this deck.")
        elif verdict is not None:
            in_envelope = False
            source_row = (f"parity_m41.json results[case={verdict['case']!r}]."
                          f"class == {verdict['class']!r}")
            reason = {
                "DEVIATION": ("the port ran and the oracle ran, but at least "
                              "one channel exceeded the M41 tolerance "
                              f"(max_rel_rms={verdict.get('max_rel_rms')!r})."),
                "NO-CHANNELS": ("no comparable channel was produced, so "
                                "reproduction was never measured."),
                "SKIPPED-SLOW": ("the parity run exceeded its time budget, so "
                                 "no verdict was reached."),
                "PYRADIOSS-FAIL": "the port itself failed on this deck.",
            }.get(verdict["class"],
                  f"parity verdict {verdict['class']!r} is not a "
                  "reproduction.")
        else:
            in_envelope = False
            source_row = ("no parity_m41.json row carries this case_id - "
                          "unmeasured, not disproven")
            reason = ("no measured parity verdict exists for this deck; it is "
                      "outside the validated envelope until a parity run "
                      "records MATCH for it.")

        skipped = dict(coverage["skipped_families"]) if coverage else {}
        if in_envelope and skipped:
            classes = sorted({i.get("class", "?") for i in skipped.values()})
            reason += (f" Qualified: the port skips "
                       f"{len(skipped)} famil{'y' if len(skipped) == 1 else 'ies'}"
                       f" on this deck (class {', '.join(classes)}): "
                       f"{', '.join(sorted(skipped))}.")

        records.append({
            # what was covered, and which bytes
            "case_id": rel if case else None,
            "deck": rel,
            "hashed_file": rel,
            "sha256": digest,
            "size_bytes": os.path.getsize(full),
            # which corpus the bytes belong to
            "corpus_fingerprint": fingerprint,
            # identity, carried over from inventory.json
            "category": case["category"] if case else None,
            "package": case["package"] if case else None,
            "inventory_classification": (case["classification"]
                                         if case else None),
            "inventory_classification_strict": (case["classification_strict"]
                                                if case else None),
            # the envelope claim and exactly what it rests on
            "in_envelope": in_envelope,
            "in_envelope_source": source_row,
            "in_envelope_reason": reason,
            # the measured verdict
            "parity_case": verdict["case"] if verdict else None,
            "parity_class": verdict["class"] if verdict else None,
            "parity_max_rel_rms": (verdict.get("max_rel_rms")
                                   if verdict else None),
            "parity_provenance": verdict.get("provenance") if verdict else None,
            # the reader census that qualifies the verdict
            "coverage_verdict": coverage["verdict"] if coverage else None,
            "skipped_families": skipped,
            "coverage_hard_skips": list(coverage["hard_skips"]) if coverage else [],
            "coverage_blockers": list(coverage["blockers"]) if coverage else [],
            "coverage_degrade_warnings": (list(coverage["degrade_warnings"])
                                          if coverage else []),
            # was the verdict measured on THESE bytes? (no, today)
            "parity_run_deck_bytes_verified": deck_bytes_match(
                _evidence_deck(verdict) if verdict else None, digest),
            "coverage_run_deck_bytes_verified": deck_bytes_match(
                _evidence_deck(coverage) if coverage else None, digest),
        })

    parity_ids = set(parity_by_id)
    coverage_ids = set(coverage_by_id)
    mine = {r["case_id"] for r in records if r["case_id"]}
    measured = [r for r in records if r["parity_class"]]
    try:
        inside_repo = os.path.commonpath([root, REPO]) == REPO
    except ValueError:            # different drives on Windows
        inside_repo = False
    doc = {
        "schema": MANIFEST_SCHEMA,
        "generated": datetime.datetime.now().replace(
            microsecond=0).isoformat(),
        "generated_by": "tools/build_rd_decks_manifest.py",
        "corpus_root": {
            "fingerprint": fingerprint,
            "fingerprint_rule": "sha256 over the sorted '<relpath>\\t<sha256>' "
                                "lines of every *_0000.rad under the root — "
                                "path-independent, so the same corpus at "
                                "another absolute path fingerprints equal",
            "resolved_by": "pyradioss.paths.rd_decks_dir()",
            "env_override": "PYRADIOSS_RD_DECKS",
            "source": source or "PYRADIOSS_RD_DECKS or the vendored corpus",
            "vendored": (os.path.relpath(root, REPO).replace(os.sep, "/")
                         if inside_repo else None),
            "resolved_at_generation": root,
        },
        "hash_algorithm": "sha256",
        "hashed_file_rule": "sha256 is the hash of the record's hashed_file; "
                            "hashed_file == deck (the starter deck) for every "
                            "record",
        "envelope_rule": f"in_envelope == (parity_m41.json class == "
                         f"{ENVELOPE_CLASS}) for this case_id; false with a "
                         "recorded reason otherwise. Always read it together "
                         "with skipped_families, which coverage_verdict does "
                         "not count in full.",
        "bytes_verified_rule": "parity_run_deck_bytes_verified / "
                               "coverage_run_deck_bytes_verified are true only "
                               "when the evidence row names an existing deck "
                               "file whose bytes hash to this record's sha256. "
                               "They are false on every M41 row — see notes.",
        "joins": {
            "inventory": "tools/validation_data/inventory.json (by case_id)",
            "parity": "tools/validation_data/parity_m41.json (by case_id)",
            "coverage": "tools/validation_data/coverage_results_m41.json "
                        "(by case_id)",
            "oracle_provenance": "tools/validation_data/"
                                 "oracle_provenance.json (admissible evidence "
                                 "channels)",
            "sweep_driver": ".agents/skills/validation-compare/SKILL.md:56-59 "
                            "(the sweep has no in-repo driver; its root is a "
                            "dead scratchpad)",
        },
        "counts": {
            "decks": len(records),
            "in_envelope": sum(1 for r in records if r["in_envelope"]),
            "measured": len(measured),
            "unmeasured": len(records) - len(measured),
            "with_case_id": len(mine),
            "joined_parity_m41": len(mine & parity_ids),
            "joined_coverage_m41": len(mine & coverage_ids),
            "in_envelope_with_control_skips": sum(
                1 for r in records if r["in_envelope"] and any(
                    i.get("class") == "control"
                    for i in r["skipped_families"].values())),
            "parity_run_deck_bytes_verified": sum(
                1 for r in records if r["parity_run_deck_bytes_verified"]),
            "coverage_run_deck_bytes_verified": sum(
                1 for r in records if r["coverage_run_deck_bytes_verified"]),
            "bytes_hashed": sum(r["size_bytes"] for r in records),
        },
        "notes": NOTES + [_envelope_qualification(records)],
        "decks": records,
    }
    assert not records or set(MANIFEST_FIELDS) == set(records[0]), (
        "the emitted record does not match MANIFEST_FIELDS — update both")
    return doc


def _diff(committed: dict, fresh: dict) -> List[str]:
    """Differences ``--check`` must report.

    Compared: the schema, the corpus **fingerprint**, the counts and the
    records — everything that describes what the corpus contains.  Ignored on
    purpose: ``generated`` (a timestamp) and the ``corpus_root`` provenance
    keys ``source`` / ``vendored`` / ``resolved_at_generation`` (properties of
    *this invocation and this checkout*, not of the corpus).  Comparing them
    would make ``--check`` fail on any checkout at a different path.
    """
    out: List[str] = []
    if committed.get("schema") != fresh.get("schema"):
        out.append("schema differs")
    if (committed.get("corpus_root") or {}).get("fingerprint") != \
            (fresh.get("corpus_root") or {}).get("fingerprint"):
        out.append("corpus fingerprint differs (different decks or bytes)")
    for key in ("counts", "decks"):
        if committed.get(key) == fresh.get(key):
            continue
        if key != "decks":
            out.append(f"{key} differs")
            continue
        old = {r["deck"]: r for r in committed.get("decks", [])}
        new = {r["deck"]: r for r in fresh.get("decks", [])}
        for deck in sorted(set(old) - set(new)):
            out.append(f"deck removed: {deck}")
        for deck in sorted(set(new) - set(old)):
            out.append(f"deck added: {deck}")
        for deck in sorted(set(old) & set(new)):
            if old[deck] != new[deck]:
                changed = [k for k in sorted(set(old[deck]) | set(new[deck]))
                           if old[deck].get(k) != new[deck].get(k)]
                out.append(f"deck changed: {deck} ({', '.join(changed)})")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="corpus root "
                                   "(default: paths.rd_decks_dir())")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed file; write nothing")
    args = ap.parse_args(argv)

    if args.root:
        root, source = os.path.abspath(args.root), "--root"
    else:
        root = str(paths.rd_decks_dir())
        source = ("PYRADIOSS_RD_DECKS" if os.environ.get("PYRADIOSS_RD_DECKS")
                  else "vendored tests/data/rd_decks")
    doc = build(root, source)
    text = json.dumps(doc, indent=2) + "\n"

    if args.check:
        if not os.path.isfile(args.out):
            print(f"FAIL: no manifest at {args.out}", file=sys.stderr)
            return 1
        with open(args.out, encoding="utf-8") as fh:
            committed = json.load(fh)
        drift = _diff(committed, doc)
        if drift:
            print("FAIL: the committed manifest is stale:", file=sys.stderr)
            for line in drift:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"OK: {doc['counts']['decks']} deck records match {root} "
              f"(fingerprint {doc['corpus_root']['fingerprint'][:19]}...)")
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    c = doc["counts"]
    print(f"wrote {args.out}: {c['decks']} decks "
          f"({c['in_envelope']} in envelope, {c['measured']} measured / "
          f"{c['unmeasured']} unmeasured, "
          f"{c['joined_parity_m41']} joined to parity_m41.json, "
          f"{c['bytes_hashed']} bytes hashed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())