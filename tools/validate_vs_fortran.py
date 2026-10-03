#!/usr/bin/env python
"""
Differential validation of the pyradioss port against the Fortran
OpenRadioss binaries (the RESEARCH_AUDIT's #1 recommendation).

Two modes
=========

parity (default)
    For each example under ``examples/`` (or ``--only name1,name2``):

    1. run the real Fortran Starter + Engine on the example's deck pair
       in a scratch directory, convert the binary T01 time-history with
       ``th_to_csv_win64.exe``;
    2. run the pyradioss Starter + Engine on the *original* deck pair in
       a sibling scratch directory (the port writes its T01 as CSV
       directly);
    3. compare every overlapping time-history channel (global internal /
       kinetic / hourglass / contact energies, external work, mass,
       momentum components, and /TH/PART energies where unambiguous) and
       report the relative RMS deviation and the final-time deviation —
       the actual numbers are always printed, the tolerance only sets
       the MATCH/DEVIATION label.

    Classification per example:

    - ``MATCH``          all compared channels within ``--tol`` (default 5 %)
    - ``DEVIATION``      Fortran and pyradioss both ran; some channel above tol
    - ``PORT-ONLY``      the Fortran chain cannot run this deck. Subtags:
                         ``(implicit)`` engine deck uses the port's /IMPL
                         cards; ``(dialect)`` the real Starter rejects the
                         port's deck dialect (see the DIALECT NOTE below);
                         ``(starter-reject)`` translated deck still refused.
    - ``FORTRAN-FAIL``   Fortran starter passed but engine/converter failed
    - ``PYRADIOSS-FAIL`` the port itself failed on its own example
    - ``SKIPPED-SLOW``   pyradioss side exceeded ``--timeout`` (default 240 s)
                         or the global ``--budget`` (default 2700 s)

    DIALECT NOTE (M36 update): the bundled example decks are now
    generated in the **real fixed 2022 format** directly, by
    ``pyradioss/input/deck_writer.py`` — the M35 per-keyword translator
    below was PROMOTED into that module and extended to every supported
    keyword family; the copy kept here is only a **fallback for
    old-dialect decks** (user decks written in the port's historical
    free-format style).  The harness auto-detects the format
    (:func:`deck_is_real_format` looks for the /BEGIN input-version
    card): real-format decks skip translation entirely and only receive
    :func:`real_deck_fixups` — the deck_writer's *documented residue
    fields* mapped to their real meaning for the Fortran run
    (/RWALL blank search distance -> 1e30; /INTER/TYPE7|11 port gap_max
    in the real Tstart column -> moved to the real GAPMAX field; the
    port-only /TH/SECT block stripped).  See the deck_writer module
    docstring for why those residues exist.

    For OLD-dialect decks the historical behaviour is unchanged: feeding
    them to the Fortran Starter unmodified fails at /BEGIN
    ("INPUT FORMAT 0 NOT SUPPORTED"), so ``--shim translate`` (default)
    re-emits a fixed-format copy per keyword; decks using keywords
    without a verified translator are classified PORT-ONLY(dialect)
    (``--shim begin`` inserts only the /BEGIN version card so the
    per-deck real error is visible; ``--shim none`` runs the deck raw).

coverage
    Run ONLY the pyradioss Starter on the given native ``.rad`` deck(s)
    (e.g. the Ryan Lee k2rad conversions) and tabulate every keyword the
    port skipped, warned about, or errored on — a keyword-coverage
    census of the port against real-world decks.

Examples
========
    python tools/validate_vs_fortran.py parity
    python tools/validate_vs_fortran.py parity --only tensile_bar,box_beam_impact
    python tools/validate_vs_fortran.py coverage E:/openradioss_run/Ryan_Lee_Examples/ton-mm-s/runs/W12_k2rad/W12_0000.rad

Results land in ``<workdir>/parity_results.json`` (parity) /
``<workdir>/coverage_<deck>.json`` (coverage) plus a console table.

Environment: the Fortran binaries and their runtime come from
``C:/OpenRadioss`` and Intel oneAPI, exactly like the proven
E:/openradioss_run/Ryan_Lee_Examples/ton-mm-s/runs/run_batch.ps1 (non-MPI
single-process path: starter_win64.exe -np 1 -nt 1, engine_win64.exe -nt 1).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyradioss.input.deck_reader import read_deck  # noqa: E402

# ----------------------------------------------------------------------------
# Fortran toolchain (mirrors run_batch.ps1, non-MPI path)
# ----------------------------------------------------------------------------

OR_ROOT = r"C:\OpenRadioss"
ONEAPI = r"C:\Program Files (x86)\Intel\oneAPI"
STARTER_EXE = os.path.join(OR_ROOT, "exec", "starter_win64.exe")
ENGINE_EXE = os.path.join(OR_ROOT, "exec", "engine_win64.exe")
TH2CSV_EXE = os.path.join(OR_ROOT, "exec", "th_to_csv_win64.exe")

DEFAULT_WORKDIR = os.environ.get(
    "VALRUNS_DIR",
    os.path.join(os.environ.get("TEMP", REPO), "valruns"))


# ----------------------------------------------------------------------------
# Corpus manifest (Phase 0 / task P0.8)
# ----------------------------------------------------------------------------
#
# The manifest is the generated, hashed answer to "which deck files does the
# differential-validation evidence actually cover, and what are their bytes?".
# It replaces the hand-kept case list that tools/validation_data/inventory.json
# and parity_m41.json are keyed by; those two files stay the authority for
# *ids*, and the manifest carries them over on `case_id` so the three files
# remain joinable.
#
# The loader is deliberately lazy and read-only: it reads the JSON inside the
# function and opens NO deck.  A validation run may legitimately happen with
# the corpus unmounted, and the recorded hashes must still be readable; it
# also means importing this module touches no resource (P0.9 replaces the
# module-scope `C:\...` toolchain literals above with pyradioss.paths).
#
# A record's `deck` is RELATIVE to a corpus, and the corpus is chosen by the
# environment (`PYRADIOSS_RD_DECKS`), so a bare `rd_decks_dir() / rec["deck"]`
# can silently join a verdict to a *different* file that happens to sit at the
# same relative path.  Two things prevent that: every record carries the
# `corpus_fingerprint` it was hashed from, and `resolve_manifest_record()`
# refuses to resolve a record against a corpus whose fingerprint differs.

MANIFEST_SCHEMA = "pyradioss/rd-decks-manifest/1"
MANIFEST_PATH = os.path.join(REPO, "tools", "validation_data",
                             "rd_decks_manifest.json")

#: Fields every record must carry — the record schema, in full.  A record
#: missing one is a broken manifest, not a deck the harness may silently skip;
#: a record carrying an *extra* field is drift the tests must see too.
MANIFEST_FIELDS = (
    # what was covered, and which bytes
    "case_id", "deck", "hashed_file", "sha256", "size_bytes",
    # which corpus the bytes belong to
    "corpus_fingerprint",
    # identity, carried over from inventory.json (never invented)
    "category", "package", "inventory_classification",
    "inventory_classification_strict",
    # the envelope claim and exactly what it rests on
    "in_envelope", "in_envelope_source", "in_envelope_reason",
    # the measured verdict (parity_m41.json) and its provenance
    "parity_case", "parity_class", "parity_max_rel_rms", "parity_provenance",
    # the reader census (coverage_results_m41.json): the families the port
    # still skips on this deck qualify every verdict recorded above
    "coverage_verdict", "skipped_families", "coverage_hard_skips",
    "coverage_blockers", "coverage_degrade_warnings",
    # whether the verdict above was measured on THESE bytes (false for every
    # M41 row: the sweep ran from a scratchpad extract that no longer exists)
    "parity_run_deck_bytes_verified", "coverage_run_deck_bytes_verified",
)


def load_manifest_doc(path: Optional[str] = None) -> dict:
    """Return the whole manifest document: header + ``decks``.

    The header is not decoration: ``corpus_root.fingerprint`` is what binds a
    record's relative ``deck`` to one corpus, ``counts`` is the coverage
    summary, and ``envelope_rule`` / ``notes`` state what ``in_envelope`` is
    allowed to mean.  :func:`load_manifest` returns the records alone for the
    common case.

    Raises ``FileNotFoundError`` when the manifest has not been generated (run
    ``tools/build_rd_decks_manifest.py``) and ``ValueError`` when the file is
    present but is not a manifest this loader understands.
    """
    target = path or MANIFEST_PATH
    with open(target, encoding="utf-8") as fh:
        doc = json.load(fh)
    schema = doc.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(
            f"{target}: schema {schema!r} is not {MANIFEST_SCHEMA!r} — "
            "regenerate it with tools/build_rd_decks_manifest.py")
    records = doc.get("decks")
    if not isinstance(records, list):
        raise ValueError(f"{target}: no 'decks' list")
    documented = set(MANIFEST_FIELDS)
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"{target}: deck record is not an object: {rec!r}")
        missing = sorted(documented - set(rec))
        extra = sorted(set(rec) - documented)
        if missing or extra:
            raise ValueError(
                f"{target}: record {rec.get('deck')!r} does not match the "
                f"record schema (missing {missing}, undocumented {extra}) — "
                "regenerate it with tools/build_rd_decks_manifest.py")
    return doc


def load_manifest(path: Optional[str] = None) -> List[dict]:
    """Return the corpus manifest records, one per starter deck.

    ``path`` defaults to :data:`MANIFEST_PATH`.  Each record names the deck
    (``deck``, a POSIX path relative to the corpus it was hashed from), the
    file its ``sha256`` was computed over (``hashed_file`` — equal to ``deck``
    unless a record says otherwise, so the hashed bytes are never a guess),
    the inventory ``case_id`` it corresponds to (``null`` when the corpus
    holds a deck inventory.json never saw), and the evidence-derived
    ``in_envelope`` flag together with the source of that flag and the
    families the port skips on the deck.

    A record is bound to the corpus named by its ``corpus_fingerprint``, NOT
    to whatever ``paths.rd_decks_dir()`` happens to resolve to; use
    :func:`resolve_manifest_record` to turn one into a path.
    """
    return load_manifest_doc(path)["decks"]


def manifest_corpus_root(doc: Optional[dict] = None) -> str:
    """The corpus the manifest was hashed from, as an absolute path.

    Prefers the checkout-relative ``corpus_root.vendored`` (so the answer does
    not depend on where the manifest was generated) and falls back to the
    absolute ``resolved_at_generation`` recorded at generation time.  This is
    NOT ``paths.rd_decks_dir()``: an env override must not silently re-point a
    record at another extract.
    """
    doc = doc if doc is not None else load_manifest_doc()
    root = doc.get("corpus_root") or {}
    vendored = root.get("vendored")
    if vendored:
        return os.path.abspath(os.path.join(REPO, vendored))
    resolved = root.get("resolved_at_generation")
    if resolved:
        return os.path.abspath(resolved)
    raise ValueError(f"{MANIFEST_PATH}: corpus_root names no location")


def corpus_fingerprint(root: str) -> str:
    """``sha256:<hex>`` over every starter deck in ``root``.

    Path-independent by construction: the digest covers the POSIX-relative
    deck paths and their content hashes, never the root itself, so the same
    corpus at another absolute path (another checkout, another machine, a
    copy in a scratch dir) fingerprints identically — which is what makes it
    usable to *accept* a moved corpus and to *reject* a different one.
    """
    root = os.path.abspath(root)
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith("_0000.rad"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            digest = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            lines.append(f"{rel}\t{digest.hexdigest()}\n")
    if not lines:
        raise ValueError(f"{root}: no *_0000.rad starter deck to fingerprint")
    lines.sort()
    return "sha256:" + hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def resolve_manifest_record(rec: dict, root: Optional[str] = None, *,
                            verify: bool = True) -> str:
    """Resolve one record's ``hashed_file`` inside a corpus, refusing a corpus
    it was not hashed from.

    ``root`` defaults to :func:`manifest_corpus_root` — the corpus the
    manifest names — so the result is independent of ``PYRADIOSS_RD_DECKS``.
    A corpus whose :func:`corpus_fingerprint` differs from the record's
    ``corpus_fingerprint`` raises ``ValueError`` rather than returning a path
    to a different file that happens to share the relative name.  With
    ``verify`` (the default) the resolved file is also re-hashed and must match
    the record's ``sha256``.
    """
    expected = rec.get("corpus_fingerprint")
    target_root = os.path.abspath(root) if root else manifest_corpus_root()
    if expected:
        actual = corpus_fingerprint(target_root)
        if actual != expected:
            raise ValueError(
                f"record {rec.get('deck')!r} was hashed from corpus "
                f"{expected}, but {target_root} is {actual} — resolve it "
                "against the corpus the manifest names, or re-hash this one "
                "with tools/build_rd_decks_manifest.py")
    path = os.path.join(target_root, rec["hashed_file"].replace("/", os.sep))
    if verify:
        if not os.path.isfile(path):
            raise ValueError(f"record {rec.get('deck')!r}: {path} is missing")
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != rec["sha256"]:
            raise ValueError(
                f"record {rec.get('deck')!r}: {path} hashes to "
                f"{digest.hexdigest()}, the manifest says {rec['sha256']}")
    return os.path.abspath(path)


def fortran_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["RAD_CFG_PATH"] = os.path.join(OR_ROOT, "hm_cfg_files")
    env["RAD_H3D_PATH"] = os.path.join(OR_ROOT, "extlib", "h3d", "lib", "win64")
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_AFFINITY"] = "disabled"
    env["KMP_STACKSIZE"] = "400m"
    env["I_MPI_ROOT"] = os.path.join(ONEAPI, "mpi", "latest")
    env["I_MPI_OFI_LIBRARY_INTERNAL"] = "1"
    env["PATH"] = ";".join([
        os.path.join(OR_ROOT, "extlib", "hm_reader", "win64"),
        os.path.join(OR_ROOT, "extlib", "intelOneAPI_runtime", "win64"),
        os.path.join(ONEAPI, "mpi", "latest", "bin"),
        os.path.join(ONEAPI, "mpi", "latest", "libfabric", "bin"),
        env.get("PATH", ""),
    ])
    return env


def run_cmd(cmd: List[str], cwd: str, timeout: float,
            env: Optional[Dict[str, str]] = None) -> Tuple[int, str, float]:
    """Run a command, return (rc, tail-of-output, elapsed). rc=-9 on timeout."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace")
        out = p.stdout or ""
        return p.returncode, out[-4000:], time.time() - t0
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT", time.time() - t0


# ----------------------------------------------------------------------------
# Listing (.out) harvesting — model size, cycle count, self-reported elapsed
# (M36: per-run wall-clock timing is a first-class deliverable; the harness
# wall-clocks each subprocess AND harvests the solvers' own counters so the
# perf records can carry cycles/s and elements*cycles/s.)
# ----------------------------------------------------------------------------

def harvest_fortran_out(starter_out: str, engine_out: str) -> Dict:
    """Counts from the real Starter/Engine listings.

    Starter: ``NUMNOD: NUMBER OF NODAL POINTS. . .   99`` and the
    ``NUMEL*:`` family (summed); ``ELAPSED TIME...........=  1.43 s``.
    Engine: ``TOTAL NUMBER OF CYCLES  :  1420`` and its ELAPSED TIME.
    """
    info: Dict = {}
    if os.path.exists(starter_out):
        txt = open(starter_out, errors="replace").read()
        m = re.search(r"NUMNOD\s*:\s*NUMBER OF NODAL POINTS[ .]*(\d+)", txt)
        if m:
            info["n_nodes"] = int(m.group(1))
        nel = 0
        seen = set()
        for m in re.finditer(r"^\s*(NUMEL\w*)\s*:[^\n]*?(\d+)\s*$",
                             txt, re.M):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                nel += int(m.group(2))
        if seen:
            info["n_elements"] = nel
        m = re.search(r"ELAPSED TIME[ .]*=\s*([\d.]+)", txt)
        if m:
            info["starter_elapsed_self"] = float(m.group(1))
    if os.path.exists(engine_out):
        txt = open(engine_out, errors="replace").read()
        m = re.search(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)", txt)
        if m:
            info["n_cycles"] = int(m.group(1))
        m = re.search(r"ELAPSED TIME[ .]*=\s*([\d.]+)", txt)
        if m:
            info["engine_elapsed_self"] = float(m.group(1))
    return info


def harvest_port_out(starter_out: str, engine_out: str) -> Dict:
    """Counts from the pyradioss listings.

    Starter: ``NUMBER OF NODES . . : 99`` / ``NUMBER OF /BRICK  ELEMENTS``
    (summed) / ``STARTER ELAPSED TIME . . :  0.031 s``.
    Engine: ``CYCLES . . : 1847`` / ``ELAPSED TIME . . :  3.034 s``.
    """
    info: Dict = {}
    if os.path.exists(starter_out):
        txt = open(starter_out, errors="replace").read()
        m = re.search(r"NUMBER OF NODES[ .]*:\s*(\d+)", txt)
        if m:
            info["n_nodes"] = int(m.group(1))
        nel = sum(int(m.group(2)) for m in re.finditer(
            r"NUMBER OF /(\w+)\s*ELEMENTS[ .]*:\s*(\d+)", txt))
        if re.search(r"NUMBER OF /(\w+)\s*ELEMENTS", txt):
            info["n_elements"] = nel
        m = re.search(r"ELAPSED TIME[ .]*:\s*([\d.]+)", txt)
        if m:
            info["starter_elapsed_self"] = float(m.group(1))
    if os.path.exists(engine_out):
        txt = open(engine_out, errors="replace").read()
        m = re.search(r"^\s*CYCLES[ .]*:\s*(\d+)", txt, re.M)
        if m:
            info["n_cycles"] = int(m.group(1))
        m = re.search(r"ELAPSED TIME[ .]*:\s*([\d.]+)", txt)
        if m:
            info["engine_elapsed_self"] = float(m.group(1))
    return info


# ----------------------------------------------------------------------------
# Real-format deck detection + residue fixups (M36)
# ----------------------------------------------------------------------------

def deck_is_real_format(path: str) -> bool:
    """True when the starter deck already carries the fixed-2022 /BEGIN
    input-version card (decks emitted by pyradioss/input/deck_writer.py).
    Such decks are fed to the Fortran Starter without translation."""
    lines = open(path, errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("/BEGIN"):
            for j in range(i + 1, min(i + 4, len(lines))):
                toks = lines[j].split()
                if toks and toks[0].isdigit() and len(toks[0]) >= 3:
                    return True
            return False
    return False


def real_deck_fixups(path: str) -> str:
    """Map the deck_writer's DOCUMENTED RESIDUE fields to their real
    meaning in the Fortran-side deck copy (values preserved; see the
    deck_writer module docstring for why the residues exist):

    * /RWALL/* — the blank d/fric card (the port reads the next card as
      the wall point) becomes d = 1e30, the real equivalent of the
      port's "track all nodes";
    * /INTER/TYPE7 — a value in the Stfac card's 4th field (real Tstart)
      is the PORT's gap_max: moved to the real GAPMAX field (card B,
      2nd field); /INTER/TYPE11 — same field blanked (the real TYPE11
      2020 layout has no gap-cap field);
    * /TH/SECT — the port's section-output request (real Radioss spells
      it /TH/SECTIO and defines sections differently): stripped.
    """
    return real_deck_fixups_text(open(path, errors="replace").read())


def real_deck_fixups_text(text: str) -> str:
    """Text-input variant of :func:`real_deck_fixups` (same mapping)."""
    lines = text.splitlines()
    out: List[str] = []
    i, n = 0, len(lines)

    def is_comment(s: str) -> bool:
        return s.lstrip().startswith(("#", "$"))

    while i < n:
        ln = lines[i]
        s = ln.strip().upper()
        if s.startswith("/TH/SECT/"):
            i += 1
            while i < n and not lines[i].lstrip().startswith("/"):
                i += 1
            out.append("# (/TH/SECT port block stripped for the Fortran "
                       "run — real Radioss spells it /TH/SECTIO)")
            continue
        if s.startswith("/RBODY/"):
            out.append(ln)
            i += 1
            if i < n and not is_comment(lines[i]):
                out.append(lines[i]) # title
                i += 1
                if i < n and not lines[i].lstrip().startswith("/"):
                    card = lines[i]
                    if len(card) <= 50 and card[10:20].strip() and not card[50:].strip():
                        # Port dialect format: master(10) grnod(10) mass(20) icog(10)
                        master = card[:10]
                        grnod = card[10:20]
                        mass = card[20:40] if len(card) >= 40 else " " * 20
                        icog = card[40:50] if len(card) == 50 else " " * 10
                        out.append(master + " " * 30 + mass + grnod + " " * 10 + icog)
                        i += 1
                        if i < n and not lines[i].lstrip().startswith("/") and not is_comment(lines[i]):
                            out.append(lines[i]) # jadd
                            i += 1
                    else:
                        out.append(lines[i])
                        i += 1
                        if i < n and not lines[i].lstrip().startswith("/") and not is_comment(lines[i]):
                            out.append(lines[i]) # jadd
                            i += 1
            continue
        if s.startswith("/RWALL/"):
            out.append(ln)
            i += 1
            replaced = False
            while i < n and not lines[i].lstrip().startswith("/"):
                if (not replaced and lines[i].strip() == ""
                        and not is_comment(lines[i])):
                    out.append(_f20(1e30))     # d: search distance
                    replaced = True
                else:
                    out.append(lines[i])
                i += 1
            continue
        if s.startswith("/INTER/TYPE7/") or s.startswith("/INTER/TYPE11/"):
            is7 = s.startswith("/INTER/TYPE7/")
            blk = [ln]
            i += 1
            while i < n and not lines[i].lstrip().startswith("/"):
                blk.append(lines[i])
                i += 1
            data = [k for k in range(1, len(blk)) if not is_comment(blk[k])]
            nonblank = [k for k in data if blk[k].strip()]
            # nonblank: [title, cardA, cardD(Stfac), ...]
            if len(nonblank) >= 3:
                kd = nonblank[2]
                card = blk[kd].ljust(100)
                tstart = card[60:80].strip()
                if tstart:
                    blk[kd] = card[:60].rstrip()
                    if is7:
                        blanks = [k for k in data if not blk[k].strip()]
                        if blanks:      # card B: Fscalegap | GAPMAX | ...
                            blk[blanks[0]] = " " * 20 + f"{tstart:>20}"
            out.extend(blk)
            continue
        out.append(ln)
        i += 1
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Old-dialect fallback translation
#
# Since M36 the translator lives in pyradioss/input/deck_writer.py (the
# promotion of the M35 per-keyword translator that used to be duplicated
# here — single source of truth).  This wrapper remains as the FALLBACK
# for user decks still written in the port's historical free-format
# dialect: it converts them with the package writer and then applies the
# same documented residue fixups as real-format decks.
# ----------------------------------------------------------------------------

def _f20(v) -> str:
    return f"{float(v):>20.10G}"


def translate_starter_deck(path: str, runname: str) -> Tuple[Optional[str],
                                                             List[str]]:
    """Return (translated text, failure notes).  None text on failure."""
    from pyradioss.input import deck_writer as _dw
    lines = open(path, errors="replace").read().splitlines()
    try:
        text = _dw.starter_deck_from_lines(lines, runname).render()
    except Exception as exc:                      # noqa: BLE001 — reported
        return None, [f"{type(exc).__name__}: {exc}"]
    return real_deck_fixups_text(text), []


def shim_begin_only(path: str) -> str:
    """Insert the 2022 version + unit cards after the /BEGIN title card
    (the minimum to get past the real reader's front door)."""
    lines = open(path, errors="replace").read().splitlines()
    out, i = [], 0
    if not (lines and lines[0].startswith("#RADIOSS STARTER")):
        # the real reader demands '#RADIOSS STARTER' as the first card
        # (ERROR 100201) — two example decks omit it
        out.append("#RADIOSS STARTER")
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip().upper().startswith("/BEGIN"):
            if i + 1 < len(lines) and not lines[i + 1].lstrip().startswith("/"):
                i += 1
                out.append(lines[i])          # keep the run-name card
            out += ["      2022         0",
                    "                  Mg                  mm                   s",
                    "                  Mg                  mm                   s"]
        i += 1
    return "\n".join(out) + "\n"


def strip_engine_stop(path: str) -> str:
    """Drop the port's /STOP block from the engine deck copy: the real
    Engine's reader hits EOF on it (measured: forrtl severe(24) on unit 30);
    everything else passes through unchanged."""
    lines = open(path, errors="replace").read().splitlines()
    out, skip = [], False
    for ln in lines:
        if ln.strip().upper().startswith("/STOP"):
            skip = True
            continue
        if skip and not ln.lstrip().startswith("/"):
            continue                     # /STOP data card(s)
        skip = False
        out.append(ln)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Time-history CSV parsing + comparison
# ----------------------------------------------------------------------------

# Fortran th_to_csv global column -> port column (pyradioss T01 CSV)
GLOBAL_MAP = [
    ("INTERNAL ENERGY", "IE"),
    ("KINETIC ENERGY", "KE"),
    ("HOURGLASS ENERGY", "HE"),
    ("CONTACT ENERGY", "CE"),
    ("EXTERNAL WORK", "EW"),
    ("MASS", "MASS"),
    ("X-MOMENTUM", "MOMX"),
    ("Y-MOMENTUM", "MOMY"),
    ("Z-MOMENTUM", "MOMZ"),
]


def read_csv_columns(path: str) -> Tuple[List[str], "np.ndarray"]:
    import numpy as np
    rows = []
    header: List[str] = []
    with open(path, newline="", errors="replace") as fh:
        for rec in csv.reader(fh):
            rec = [c.strip().strip('"') for c in rec if c.strip() != ""]
            if not rec:
                continue
            if rec[0].startswith("#"):
                continue
            if not header:
                header = rec
                continue
            try:
                rows.append([float(x) for x in rec])
            except ValueError:
                continue
    n = min(len(r) for r in rows) if rows else 0
    data = np.array([r[:n] for r in rows]) if rows else np.zeros((0, 0))
    return header[:n] if n else header, data


def compare_channels(f_hdr, f_dat, p_hdr, p_dat, part_titles=None):
    """Return list of dicts: channel, rel_rms, final_dev (fractions)."""
    import numpy as np
    res = []
    if f_dat.size == 0 or p_dat.size == 0:
        return res
    tf, tp = f_dat[:, 0], p_dat[:, 0]
    tend = min(tf[-1], tp[-1])
    grid = tp[tp <= tend + 1e-12]
    if len(grid) < 3:
        return res

    def col(hdr, dat, name, exact=True):
        for j, h in enumerate(hdr):
            if (h == name) if exact else (name in h):
                return dat[:, j]
        return None

    pairs = []
    for fname, pname in GLOBAL_MAP:
        a, b = col(f_hdr, f_dat, fname), col(p_hdr, p_dat, pname)
        if a is not None and b is not None:
            pairs.append((pname, a, b))
    # derived total energy
    fie, fke = col(f_hdr, f_dat, "INTERNAL ENERGY"), col(f_hdr, f_dat,
                                                         "KINETIC ENERGY")
    pie, pke = col(p_hdr, p_dat, "IE"), col(p_hdr, p_dat, "KE")
    if all(x is not None for x in (fie, fke, pie, pke)):
        pairs.append(("IE+KE", fie + fke, pie + pke))
    # /TH/PART energies: fortran header 'title ... IE'; port 'P<id>_IE'
    for suffix in ("IE", "KE"):
        fcols = [j for j, h in enumerate(f_hdr)
                 if re.search(rf"\s{suffix}\s*$", h)]
        pcols = [j for j, h in enumerate(p_hdr)
                 if re.fullmatch(rf"P\d+_{suffix}", h)]
        if len(fcols) == 1 and len(pcols) == 1:
            pairs.append((f"part_{suffix}", f_dat[:, fcols[0]],
                          p_dat[:, pcols[0]]))

    # group reference scales: a channel that is numerically ~zero on BOTH
    # sides relative to its group's dominant channel (e.g. the transverse
    # momentum of a uniaxial test) is noise, not physics — it is still
    # reported, but marked insignificant and excluded from MATCH/DEVIATION.
    def group_of(nm):
        if nm.startswith("MOM"):
            return "momentum"
        if nm == "MASS":
            return "mass"
        return "energy"

    interp = {}
    for nm, a, b in pairs:
        interp[nm] = (np.interp(grid, tf, a), np.interp(grid, tp, b))
    gref: Dict[str, float] = {}
    for nm, (ai, bi) in interp.items():
        g = group_of(nm)
        gref[g] = max(gref.get(g, 0.0), float(np.max(np.abs(ai))),
                      float(np.max(np.abs(bi))))

    for nm, a, b in pairs:
        ai, bi = interp[nm]
        denom = max(float(np.max(np.abs(ai))), float(np.max(np.abs(bi))))
        ref = gref[group_of(nm)]
        if denom <= 0 or (ref > 0 and denom < 1e-9 * ref):
            continue    # empty on both sides — nothing to compare
        rel_rms = float(np.sqrt(np.mean((ai - bi) ** 2)) / denom)
        final = float(abs(ai[-1] - bi[-1]) / denom)
        # significance rule (documented in VALIDATION.md): a channel drives
        # the MATCH/DEVIATION class only when it carries at least 1 % of its
        # group's dominant scale — e.g. the transverse momentum of an
        # axially loaded mast (0.1 % of the axial momentum) is numerical
        # noise on both sides; it is still printed, prefixed '~'.
        res.append({"channel": nm, "rel_rms": rel_rms, "final_dev": final,
                    "scale": denom,
                    "significant": bool(denom >= 1e-2 * ref)})
    return res


# ----------------------------------------------------------------------------
# Fortran + pyradioss single-example drivers
# ----------------------------------------------------------------------------

def first_starter_error(out_file: str, log_tail: str) -> str:
    if os.path.exists(out_file):
        txt = open(out_file, errors="replace").read()
        m = re.search(r"ERROR ID\s*:\s*\S+\n\*\*[^\n]*\nDESCRIPTION[^\n]*\n"
                      r"((?:--[^\n]*\n|[^\n]*\n){0,3})", txt)
        if m:
            desc = " | ".join(l.strip() for l in m.group(0).splitlines()[:6]
                              if l.strip())
            return desc[:300]
    tail = " | ".join(l.strip() for l in log_tail.splitlines()[-4:]
                      if l.strip())
    return tail[:300]


def run_fortran(name: str, runname: str, deck0: str, deck1: str,
                workdir: str, shim: str) -> Dict:
    """Run starter+engine+th_to_csv. Returns dict with status/csv/error."""
    rd = os.path.join(workdir, "fortran", name)
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    shutil.copytree(os.path.dirname(deck0), rd, dirs_exist_ok=True)
    d0 = os.path.join(rd, os.path.basename(deck0))
    d1 = os.path.join(rd, os.path.basename(deck1))
    info: Dict = {"mode": shim, "dir": rd}

    if shim != "none" and deck_is_real_format(deck0):
        # M36: the deck is already in the real fixed 2022 format (emitted
        # by pyradioss/input/deck_writer.py) — no translation, only the
        # documented residue fixups for the Fortran side.
        info["mode"] = "real-format"
        open(d0, "w").write(real_deck_fixups(deck0))
    elif shim == "translate":
        text, missing = translate_starter_deck(deck0, runname)
        if text is None:
            info.update(status="untranslatable", missing=missing)
            # still probe with the begin shim to capture the real error
            open(d0, "w").write(shim_begin_only(deck0))
            env = fortran_env()
            rc, tail, dt = run_cmd(
                [STARTER_EXE, "-i", os.path.basename(d0), "-np", "1",
                 "-nt", "1"], rd, 180, env)
            info["probe_error"] = first_starter_error(
                os.path.join(rd, f"{runname}_0000.out"), tail)
            return info
        open(d0, "w").write(text)
    elif shim == "begin":
        open(d0, "w").write(shim_begin_only(deck0))
    else:
        shutil.copy(deck0, d0)
    open(d1, "w").write(strip_engine_stop(deck1)
                        if shim != "none" else open(deck1).read())

    env = fortran_env()
    rc, tail, dt = run_cmd([STARTER_EXE, "-i", os.path.basename(d0),
                            "-np", "1", "-nt", "1"], rd, 300, env)
    info["starter_rc"] = rc
    info["starter_time"] = round(dt, 2)
    out0 = os.path.join(rd, f"{runname}_0000.out")
    nerr = 0
    if os.path.exists(out0):
        nerr = len(re.findall(r"^ERROR ID", open(out0, errors="replace")
                              .read(), re.M))
    rst = os.path.join(rd, f"{runname}_0000_0001.rst")
    if rc != 0 or nerr or not os.path.exists(rst):
        info.update(status="starter-reject", nerr=nerr,
                    error=first_starter_error(out0, tail))
        return info

    rc, tail, dt = run_cmd([ENGINE_EXE, "-i", os.path.basename(d1),
                            "-nt", "1"], rd, 900, env)
    info["engine_rc"] = rc
    info["engine_time"] = round(dt, 2)
    info.update(harvest_fortran_out(out0,
                                    os.path.join(rd, f"{runname}_0001.out")))
    t01 = os.path.join(rd, f"{runname}T01")
    normal = "NORMAL TERMINATION" in tail or (
        os.path.exists(os.path.join(rd, f"{runname}_0001.out")) and
        "NORMAL TERMINATION" in open(os.path.join(rd, f"{runname}_0001.out"),
                                     errors="replace").read())
    if not os.path.exists(t01):
        info.update(status="engine-fail",
                    error=tail.splitlines()[-1] if tail else f"rc={rc}")
        return info
    rc2, tail2, _ = run_cmd([TH2CSV_EXE, os.path.basename(t01)], rd, 120, env)
    csvp = t01 + ".csv"
    if not os.path.exists(csvp):
        info.update(status="th2csv-fail", error=tail2[-200:])
        return info
    info.update(status="ok" if normal else "engine-partial", csv=csvp)
    return info


def run_pyradioss(name: str, runname: str, deck0: str, deck1: str,
                  workdir: str, timeout: float) -> Dict:
    rd = os.path.join(workdir, "pyradioss", name)
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    shutil.copytree(os.path.dirname(deck0), rd, dirs_exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    info: Dict = {"dir": rd}
    rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.starter", "-i",
                            os.path.basename(deck0)], rd, timeout, env)
    info["starter_rc"] = rc
    info["starter_time"] = round(dt, 2)
    if rc == -9:
        info["status"] = "timeout"
        return info
    if rc != 0:
        info.update(status="starter-fail", error=tail[-300:])
        return info
    remain = max(20.0, timeout - dt)
    rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.engine", "-i",
                            os.path.basename(deck1)], rd, remain, env)
    info["engine_rc"] = rc
    info["engine_time"] = round(dt, 2)
    info.update(harvest_port_out(
        os.path.join(rd, f"{runname}_0000.out"),
        os.path.join(rd, f"{runname}_0001.out")))
    if rc == -9:
        info["status"] = "timeout"
        return info
    csvp = os.path.join(rd, f"{runname}T01.csv")
    if rc != 0:
        info.update(status="engine-fail", error=tail[-300:])
        return info
    if not os.path.exists(csvp):
        # normal termination without a T-file (e.g. /IMPL runs write no
        # explicit time history) — success, but nothing to compare.
        info.update(status="ok-no-th")
        return info
    info.update(status="ok", csv=csvp)
    return info


# ----------------------------------------------------------------------------
# parity mode
# ----------------------------------------------------------------------------

def find_examples(only: Optional[List[str]]) -> List[Tuple[str, str, str, str]]:
    """[(name, runname, deck0, deck1)] sorted explicit-first."""
    exdir = os.path.join(REPO, "examples")
    items = []
    for name in sorted(os.listdir(exdir)):
        d = os.path.join(exdir, name)
        if not os.path.isdir(d):
            continue
        if only and name not in only:
            continue
        d0 = [f for f in os.listdir(d) if f.endswith("_0000.rad")]
        if not d0:
            continue
        runname = d0[0][:-len("_0000.rad")]
        deck0 = os.path.join(d, d0[0])
        deck1 = os.path.join(d, f"{runname}_0001.rad")
        if not os.path.exists(deck1):
            continue
        implicit = "/IMPL" in open(deck1, errors="replace").read()
        items.append((implicit, name, runname, deck0, deck1))
    items.sort(key=lambda it: (it[0], it[1]))     # explicit first
    return [(n, r, a, b) for _, n, r, a, b in items]


def parity(args) -> int:
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    only = args.only.split(",") if args.only else None
    examples = find_examples(only)
    if not examples:
        print("no examples found", file=sys.stderr)
        return 2
    budget_left = args.budget
    results = []
    retry_queue = []

    def one(name, runname, deck0, deck1, budget_left):
        implicit = "/IMPL" in open(deck1, errors="replace").read()
        row = {"example": name, "runname": runname, "implicit": implicit}

        # ---- Fortran side ------------------------------------------------
        if implicit:
            # the engine controls (/IMPL, /IMPL/FATIG/...) are port
            # extensions; the Fortran chain cannot run them.  Probe the
            # starter anyway so the table carries the real error message.
            f = run_fortran(name, runname, deck0, deck1, workdir,
                            "begin" if args.shim != "none" else "none")
            f["status"] = "implicit-port-card"
            row["class"] = "PORT-ONLY(implicit)"
        else:
            f = run_fortran(name, runname, deck0, deck1, workdir, args.shim)
            if f["status"] == "untranslatable":
                row["class"] = "PORT-ONLY(dialect)"
            elif f["status"] == "starter-reject":
                row["class"] = ("PORT-ONLY(dialect)" if args.shim != "translate"
                                else "PORT-ONLY(starter-reject)")
            elif f["status"] in ("engine-fail", "th2csv-fail"):
                row["class"] = "FORTRAN-FAIL"
        row["fortran"] = {k: v for k, v in f.items() if k != "dir"}

        # ---- pyradioss side ----------------------------------------------
        if budget_left <= 0:
            row["pyradioss"] = {"status": "skipped-budget"}
            row.setdefault("class", "SKIPPED-SLOW")
            return row, budget_left
        t0 = time.time()
        p = run_pyradioss(name, runname, deck0, deck1, workdir,
                          min(args.timeout, max(30, budget_left)))
        budget_left -= time.time() - t0
        row["pyradioss"] = {k: v for k, v in p.items() if k != "dir"}
        if p["status"] == "timeout":
            row["class"] = "SKIPPED-SLOW"
            return row, budget_left
        if p["status"] not in ("ok", "ok-no-th"):
            row["class"] = "PYRADIOSS-FAIL"

        # ---- comparison ----------------------------------------------------
        if f.get("csv") and p.get("csv"):
            fh, fd = read_csv_columns(f["csv"])
            ph, pd = read_csv_columns(p["csv"])
            ch = compare_channels(fh, fd, ph, pd)
            row["channels"] = ch
            sig = [c for c in ch if c.get("significant", True)]
            if sig:
                worst = max(c["rel_rms"] for c in sig)
                row["max_rel_rms"] = worst
                row["class"] = "MATCH" if worst <= args.tol else "DEVIATION"
            else:
                row["class"] = "FORTRAN-FAIL"
                row["fortran"]["error"] = "no overlapping channels"
        elif "class" not in row:
            row["class"] = "FORTRAN-FAIL"
        return row, budget_left

    for name, runname, deck0, deck1 in examples:
        print(f"=== {name} ...", flush=True)
        row, budget_left = one(name, runname, deck0, deck1, budget_left)
        if row.get("pyradioss", {}).get("status") in ("starter-fail",
                                                      "engine-fail"):
            retry_queue.append((name, runname, deck0, deck1))
        results.append(row)
        print(f"    -> {row.get('class')}", flush=True)

    # one retry for pyradioss failures (concurrent-edit protection)
    for name, runname, deck0, deck1 in retry_queue:
        if budget_left <= 0:
            break
        print(f"=== retry {name} ...", flush=True)
        time.sleep(30)
        row, budget_left = one(name, runname, deck0, deck1, budget_left)
        for i, r in enumerate(results):
            if r["example"] == name:
                results[i] = row
        print(f"    -> {row.get('class')} (retry)", flush=True)

    out = os.path.join(workdir, "parity_results.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nresults JSON: {out}\n")

    # ---- console table ------------------------------------------------------
    hdr = f"{'example':<34} {'class':<26} {'maxRMS':>8}  channels"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        chs = r.get("channels", [])
        chtxt = " ".join(
            ("" if c.get("significant", True) else "~") +
            f"{c['channel']}={c['rel_rms']:.3G}" for c in chs)
        rms = f"{r.get('max_rel_rms', float('nan')):.3G}" \
            if "max_rel_rms" in r else "-"
        print(f"{r['example']:<34} {r.get('class', '?'):<26} {rms:>8}  "
              f"{chtxt}")
    return 0


# ----------------------------------------------------------------------------
# coverage mode
# ----------------------------------------------------------------------------

def coverage(args) -> int:
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    from pyradioss.input.starter_keywords import KEYWORD_PARSERS
    rc_all = 0
    for deck in args.decks:
        deck = os.path.abspath(deck)
        name = os.path.splitext(os.path.basename(deck))[0]
        rd = os.path.join(workdir, "coverage", name)
        shutil.rmtree(rd, ignore_errors=True)
        os.makedirs(rd)
        shutil.copytree(os.path.dirname(deck), rd, dirs_exist_ok=True)

        # 1. keyword census straight from the lexer
        census: Dict[str, int] = {}
        for b in read_deck(deck):
            census[b.keyword] = census.get(b.keyword, 0) + 1

        # 2. run the port starter, harvest its messages
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO
        rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.starter",
                                "-i", os.path.basename(deck)], rd,
                               args.timeout, env)
        listing = os.path.join(
            rd, re.sub(r"_0000$", "", name) + "_0000.out")
        text = tail
        if os.path.exists(listing):
            text = open(listing, errors="replace").read() + "\n" + tail
        msgs: Dict[str, List[str]] = {}
        for m in re.finditer(r"\*\*\s*(WARNING|ERROR)[^\n]*\n?([^\n]*)", text):
            line = (m.group(0).replace("\n", " ").strip())[:200]
            kw = "?"
            km = re.search(r"/([A-Z0-9_/\-]+)", line)
            if km:
                kw = km.group(1).split("/")[0]
            msgs.setdefault(kw, []).append(line)

        rows = []
        for kw in sorted(census):
            k0 = kw.split("/")[0]
            ported = k0 in KEYWORD_PARSERS
            note = ""
            for mk, ml in msgs.items():
                for line in ml:
                    if f"/{kw}" in line or (not ported and f"/{k0}" in line):
                        note = line
                        break
                if note:
                    break
            rows.append({"keyword": kw, "blocks": census[kw],
                         "dispatch": "ported" if ported else "UNKNOWN",
                         "message": note})
        result = {"deck": deck, "starter_rc": rc, "elapsed": round(dt, 1),
                  "rows": rows,
                  "other_messages": {k: v[:3] for k, v in msgs.items()},
                  "tail": tail[-1500:]}
        outp = os.path.join(workdir, f"coverage_{name}.json")
        json.dump(result, open(outp, "w"), indent=1)
        print(f"\n### coverage {name}  (starter rc={rc}, {dt:.0f}s)"
              f"  -> {outp}")
        print(f"{'keyword':<28} {'blocks':>6} {'dispatch':<9} message")
        for r in rows:
            print(f"{r['keyword']:<28} {r['blocks']:>6} {r['dispatch']:<9} "
                  f"{r['message'][:90]}")
        if rc not in (0,):
            rc_all = 1
    return rc_all


# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")

    pp = sub.add_parser("parity", help="Fortran vs pyradioss on examples/")
    pp.add_argument("--only", help="comma-separated example names")
    pp.add_argument("--tol", type=float, default=0.05,
                    help="rel RMS tolerance for MATCH (default 0.05)")
    pp.add_argument("--timeout", type=float, default=240,
                    help="per-example pyradioss wall clock cap (s)")
    pp.add_argument("--budget", type=float, default=2700,
                    help="total pyradioss wall clock budget (s)")
    pp.add_argument("--shim", choices=["none", "begin", "translate"],
                    default="translate",
                    help="Fortran-side deck handling (see module docstring)")
    pp.add_argument("--workdir", default=DEFAULT_WORKDIR)

    cp = sub.add_parser("coverage", help="pyradioss starter keyword census")
    cp.add_argument("decks", nargs="+", help=".rad starter decks")
    cp.add_argument("--timeout", type=float, default=900)
    cp.add_argument("--workdir", default=DEFAULT_WORKDIR)

    args = ap.parse_args(argv)
    if args.mode == "coverage":
        return coverage(args)
    if args.mode is None:
        args = ap.parse_args(["parity"] + (argv or sys.argv[1:]))
    return parity(args)


if __name__ == "__main__":
    sys.exit(main())
