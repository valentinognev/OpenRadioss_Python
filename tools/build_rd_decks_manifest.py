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
  record today; the field exists so a record that hashes something else has to
  say so);
* ``case_id`` / ``category`` / ``package`` — carried over from
  ``inventory.json`` when the deck is a case it knows, else ``null`` (ids are
  never invented, so ``parity_m41.json`` stays joinable);
* the evidence verdict: ``parity_*`` from ``parity_m41.json`` and
  ``coverage_verdict`` from ``coverage_results_m41.json``;
* ``in_envelope`` **plus** ``in_envelope_source`` / ``in_envelope_reason`` —
  true only where a *measured* parity verdict is ``MATCH``.  A statically
  predicted ``inventory.json`` classification (``IN_ENVELOPE``, i.e. "the port
  should read this deck without skipping a family") is recorded as
  ``inventory_classification`` but never flips the flag: that would make the
  manifest claim more than the oracle can support.  Everything else is false
  with the reason recorded.

Only the starter deck (``*_0000.rad``) is a *case*: the engine decks are
restart continuations of the same case and are hashed only insofar as the
parent deck identifies them.

Usage
-----
    python tools/build_rd_decks_manifest.py            # (re)generate
    python tools/build_rd_decks_manifest.py --check    # verify, write nothing
    python tools/build_rd_decks_manifest.py --root <corpus> --out <file>

``--check`` exits non-zero when the committed file differs from what the
corpus on disk would produce; ``tests/test_p0_manifest.py`` re-verifies the
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

from pyradioss import paths  # noqa: E402

VALIDATION_DATA = os.path.join(REPO, "tools", "validation_data")
DEFAULT_OUT = os.path.join(VALIDATION_DATA, "rd_decks_manifest.json")
INVENTORY = os.path.join(VALIDATION_DATA, "inventory.json")
PARITY = os.path.join(VALIDATION_DATA, "parity_m41.json")
COVERAGE = os.path.join(VALIDATION_DATA, "coverage_results_m41.json")
PROVENANCE = os.path.join(VALIDATION_DATA, "oracle_provenance.json")

SCHEMA = "pyradioss/rd-decks-manifest/1"

#: The one measured verdict that puts a deck inside the validated envelope.
#: Everything else (DEVIATION, NO-CHANNELS, SKIPPED-SLOW, PYRADIOSS-FAIL,
#: PORT-ONLY*) is false — those rows say the port did not reproduce the oracle,
#: or that nothing was compared at all.
ENVELOPE_CLASS = "MATCH"

NOTES = [
    "sha256 is the hash of hashed_file; hashed_file equals deck for every "
    "record (the starter deck itself), stated per record so the hashed bytes "
    "are never a guess.",
    "in_envelope is true only where parity_m41.json carries a measured "
    f"verdict of {ENVELOPE_CLASS} for this exact case_id; it means 'the port "
    "reproduced the oracle on this deck'.",
    "inventory_classification is the STATIC reading prediction from "
    "inventory.json (IN_ENVELOPE = no unsupported keyword family). It is "
    "recorded for context and never sets in_envelope.",
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


def build(root: str, out_path: str) -> dict:
    inventory = {c["case_id"]: c for c in _load(INVENTORY).get("cases", [])}
    parity_rows = _load(PARITY).get("results", [])
    parity_by_id: Dict[str, dict] = {}
    for row in parity_rows:
        cid = row.get("case_id")
        if cid:
            parity_by_id[cid] = row
    coverage_by_id = {c["case_id"]: c for c in _load(COVERAGE).get("cases", [])
                      if c.get("case_id")}

    records = []
    for rel in starter_decks(root):
        full = os.path.join(root, rel.replace("/", os.sep))
        case = inventory.get(rel)
        verdict = parity_by_id.get(rel) if case else None
        coverage = coverage_by_id.get(rel) if case else None

        if verdict is not None and verdict["class"] == ENVELOPE_CLASS:
            in_envelope = True
            source = (f"parity_m41.json results[case={verdict['case']!r}]."
                      f"class == {ENVELOPE_CLASS} "
                      f"(max_rel_rms={verdict['max_rel_rms']!r}, "
                      f"provenance={verdict.get('provenance')!r})")
            reason = ("measured differential parity against the Fortran "
                      "oracle: every compared channel within the M41 "
                      "tolerance.")
        elif verdict is not None:
            in_envelope = False
            source = (f"parity_m41.json results[case={verdict['case']!r}]."
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
            source = ("no parity_m41.json row carries this case_id - "
                      "unmeasured, not disproven")
            reason = ("no measured parity verdict exists for this deck; it is "
                      "outside the validated envelope until a parity run "
                      "records MATCH for it.")

        records.append({
            "case_id": rel if case else None,
            "deck": rel,
            "hashed_file": rel,
            "sha256": _sha256(full),
            "size_bytes": os.path.getsize(full),
            "category": case["category"] if case else None,
            "package": case["package"] if case else None,
            "in_envelope": in_envelope,
            "in_envelope_source": source,
            "in_envelope_reason": reason,
            "inventory_classification": (case["classification"]
                                         if case else None),
            "inventory_classification_strict": (case["classification_strict"]
                                                if case else None),
            "parity_case": verdict["case"] if verdict else None,
            "parity_class": verdict["class"] if verdict else None,
            "parity_max_rel_rms": verdict.get("max_rel_rms") if verdict else None,
            "parity_provenance": verdict.get("provenance") if verdict else None,
            "coverage_verdict": coverage["verdict"] if coverage else None,
        })

    parity_ids = set(parity_by_id)
    coverage_ids = set(coverage_by_id)
    mine = {r["case_id"] for r in records if r["case_id"]}
    vendored = os.path.relpath(paths.rd_decks_dir(), REPO).replace(os.sep, "/")
    doc = {
        "schema": SCHEMA,
        "generated": datetime.datetime.now().replace(
            microsecond=0).isoformat(),
        "generated_by": "tools/build_rd_decks_manifest.py",
        "corpus_root": {
            "resolved_by": "pyradioss.paths.rd_decks_dir()",
            "env_override": "PYRADIOSS_RD_DECKS",
            "vendored": vendored,
            "resolved_at_generation": os.path.abspath(root),
        },
        "hash_algorithm": "sha256",
        "hashed_file_rule": "sha256 is the hash of the record's hashed_file; "
                            "hashed_file == deck (the starter deck) for every "
                            "record",
        "envelope_rule": f"in_envelope == (parity_m41.json class == "
                         f"{ENVELOPE_CLASS}) for this case_id; false with a "
                         "recorded reason otherwise",
        "joins": {
            "inventory": "tools/validation_data/inventory.json (by case_id)",
            "parity": "tools/validation_data/parity_m41.json (by case_id)",
            "coverage": "tools/validation_data/coverage_results_m41.json "
                        "(by case_id)",
            "oracle_provenance": "tools/validation_data/"
                                 "oracle_provenance.json (admissible evidence "
                                 "channels)",
        },
        "counts": {
            "decks": len(records),
            "in_envelope": sum(1 for r in records if r["in_envelope"]),
            "with_case_id": len(mine),
            "joined_parity_m41": len(mine & parity_ids),
            "joined_coverage_m41": len(mine & coverage_ids),
            "bytes_hashed": sum(r["size_bytes"] for r in records),
        },
        "notes": NOTES,
        "decks": records,
    }
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="corpus root "
                                   "(default: paths.rd_decks_dir())")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed file; write nothing")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root) if args.root else str(paths.rd_decks_dir())
    doc = build(root, args.out)
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
        print(f"OK: {doc['counts']['decks']} deck records match {root}")
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    c = doc["counts"]
    print(f"wrote {args.out}: {c['decks']} decks "
          f"({c['in_envelope']} in envelope, "
          f"{c['joined_parity_m41']} joined to parity_m41.json, "
          f"{c['bytes_hashed']} bytes hashed)")
    return 0


def _diff(committed: dict, fresh: dict) -> List[str]:
    """Differences that matter for ``--check`` (ignoring ``generated``)."""
    out: List[str] = []
    for key in ("schema", "corpus_root", "counts", "decks"):
        if committed.get(key) != fresh.get(key):
            if key == "decks":
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
            else:
                out.append(f"{key} differs")
    return out


if __name__ == "__main__":
    sys.exit(main())