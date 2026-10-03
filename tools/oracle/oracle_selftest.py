"""Golden reference run of the real Fortran OpenRadioss, and its determinism proof.

This module is the harness behind ``tests/test_p0_oracle_selftest.py`` and the
producer of ``tools/validation_data/oracle_smoke.json`` -- the seed of every
later parity comparison in this program.  Three things live here and nowhere
else:

1. :func:`run_reference` -- copy a deck pair into a scratch directory, run
   ``$OR_STARTER`` then ``$OR_ENGINE`` with the oracle runtime environment,
   parse the engine listing for the ``ENGINE TERMINATION`` banner and the cycle
   count, md5 the ``<run>T01`` binary and decode its 23 global channels.
2. :func:`parse_t01` / :func:`channel_maxima` -- a pure-Python reader for the
   binary T01, so the stored maxima can be re-derived from the committed golden
   file with no oracle present (the always-running structural tests).
3. :func:`deterministic_md5` -- the anchor.  See "Determinism" below.

Upstream Fortran origins
------------------------
``$OR_SRC`` = /home/valentin/Projects/OpenRadioss/OpenCourant, read-only.

* ``INSTALL.md:105-115``            -- "Run OpenRadioss Starter and Engine from
  the directory that contains the binaries", ``:110``
  ``./starter_linux64_gf -i [Starter input file] -np 1`` and ``:111``
  ``./engine_linux64_gf -i [Engine input file]``.  The engine's argument parser
  has no ``-np``: passing it prints the usage and then dies with a segmentation
  violation (measured; recorded in ``oracle_provenance.json`` ->
  ``invocation``), so this harness passes ``-nt 1`` there and never ``-np``.
* ``qa-tests/scripts/or_run_test.py:74-125`` -- upstream's own reference
  driver: the starter is a *separate* invocation whose return code is checked,
  then the engine(s), each in a private copy of the data directory.  That is
  the same two-step, private-scratch-directory shape :func:`run_reference` uses.
* ``INSTALL.md:34-42``              -- the Linux runtime block:
  ``OPENRADIOSS_PATH`` (retargeted here to ``$OR_BUILD``, the writable mirror),
  ``RAD_CFG_PATH=$OPENRADIOSS_PATH/hm_cfg_files`` and
  ``LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/hm_reader/linux64/``.
  ``RAD_H3D_PATH`` is documented there too and is deliberately NOT set: see
  "The h3d refusal" below.
* ``INSTALL.md:95-101``             -- ``OMP_NUM_THREADS``.  Pinned to 1 here
  (in addition to ``-np 1`` / ``-nt 1``) so no OpenMP reduction order can vary
  between the determinism runs.
* ``engine/source/output/th/hist1.F:154-188`` -- the T01 is opened per
  ``ITTYP``; the 2022 default for a ``/TFILE`` with no ``/VERS`` is ``ITTYP==3``
  (``:162-173``, ``OPEN_C(IFILNAM,LEN_TMP_NAME,0)``), which ``ITTYP==4``/``5``
  (``:174-188``) are converted into.  That is the format :func:`parse_t01`
  reads; any other ``ITTYP`` produces a different framing and is rejected
  loudly rather than mis-parsed.
* ``engine/source/output/th/hist1.F:190-209``  -- header record 1: the format
  code and the deck title.
* ``engine/source/output/th/hist1.F:210-234``  -- header record 2: the
  wall-clock stamp.  ``MY_CTIME(ITITLE)`` fills ``CH80(1:24)``, ``CH80(25:33)``
  is ``' RADIOSS '``, ``CH80(34:59)`` the version and ``CH80(60:80)``
  ``CPUNAM``; the record is written unconditionally.
* ``engine/source/system/timer_c.c:30-40``     -- ``my_ctime``: ``time()`` +
  ``ctime()``, the 24 characters of the result copied out at ``:39``.  There
  is no keyword and no environment variable that suppresses it.
* ``engine/source/output/th/hist1.F:294-308`` -- the 6-integer hierarchy record
  ``(NPART+NTHPART, NUMMAT, NUMGEO, NSUBS, NTHGRP2, NGLOBTH)``.
* ``engine/source/output/th/hist1.F:300-316`` -- ``NGLOBTH=23`` and the
  ``1..NGLOBTH`` integer record written right after the hierarchy record.  This
  module finds the data section through that record, so the number of global
  channels is read from the file, never assumed.
* ``engine/source/output/th/hist1.F:357-376`` -- the per-part description
  (``IPART(4,N)``, the title, ``IPART(7,N)``, the two bounds and ``NVAR``)
  followed by the ``NVAR`` curve codes the deck requested -- this is where
  :func:`_header_codes` reads the part channel codes from.
* ``engine/source/output/th/hist1.F:442-514`` -- the same for subsets (written
  AFTER the parts, which is what makes the header walk positional).
* ``engine/source/output/th/hist1.F:516-600`` -- the same for TH groups: five
  int32 and the title, one int32+title per curve, then the ``NVAR`` curve codes
  a ``/TH/NODE``-style card requested.  This is where the nodal channel codes
  come from.
* ``starter/source/output/th/th_titles.F90:168-189`` -- ``varn1_title``, the
  nodal curve table those codes index into.
* ``starter/source/output/th/th_titles.F90:2759-2792`` -- ``varpa_title``, the
  per-part curve table.
* ``engine/source/output/th/hist2.F:302-303`` -- the ``TT`` record, the first
  record of every step.
* ``engine/source/output/th/hist2.F:307-333`` -- the ``NGLOBTH`` global
  channels, in write order.
* ``starter/source/output/th/write_thnms1.F90:228-250`` -- the authoritative
  index -> short name -> description table for those 23 channels.  This is
  where :data:`GLOBAL_CHANNELS` is transcribed from.
* ``engine/source/output/th/hist2.F:338-477`` -- the per-part values, one
  record for all parts; ``:608-1403`` the TH-group values.
* ``engine/source/output/th/wrtdes.F:121-133`` -- ``ITTYP==3`` writes every
  value as a **single-precision** ``REAL`` (``R4 = A(I)``) even in a
  double-precision build, so all stored maxima are float32 widened to float64.
* ``common_source/tools/input_output/write_routines.c:499-511`` (``eor_c``),
  ``:520-540`` (``write_r_c``), ``:646-664`` (``write_i_c``) -- the Radioss
  IEEE format: **big-endian**.  4-byte record markers, big-endian ``int32``,
  and ``real_to_IEEE_ASCII`` for the values.  Hence ``>i4``/``>f4`` throughout.

Determinism: what is actually reproducible, and what is not
-----------------------------------------------------------
``hist1.F:211`` calls ``time()`` through :func:`my_ctime` and writes the result
into the T01.  Measured 2026-10-03 on this oracle, three runs of
``examples/tensile_bar`` in three scratch directories:

* normalized md5 (the 24 stamp bytes zeroed): identical every time,
  ``e3688899358f35e825cd640f9bd94964``;
* raw md5 of two runs that happened to share a wall-clock second: identical;
* raw md5 of a third run, one second away: differed in **one** byte -- the
  seconds digit of the stamp.

So the solver is bit-reproducible and the *raw T01 bytes* are not.  A raw md5
therefore cannot be the parity anchor: it would be a false failure roughly once
a second.  The anchor is :func:`deterministic_md5`, and the gate asserts THREE
halves so no half can rot: equal normalized digests, every varying raw byte
inside the 24-byte stamp window, and **at least two distinct raw digests
observed** -- without the third, a ``--write`` whose runs all landed in one
second would assert ``md5_raw_is_reproducible: false`` without having measured
it.  :func:`write_record` retries until two runs straddle a second and refuses
to write if none do.

The anchor is also bound to the ARCHITECTURE, not only to the build: the stamp
is the first 24 bytes of an 80-byte record whose remainder carries
``VERSIO(2)`` and ``CPUNAM`` (``hist1.F:212-217``, here ``"linuxa64 gf"``), and
those bytes stay inside the normalized digest.  ``environment.anchor_scope`` in
the record says so.

Key naming (M6): :func:`run_reference` returns ``t01_md5_normalized`` and
``t01_md5_raw``, and the record stores them as ``t01.md5_normalized`` /
``t01.md5_raw``.  The long name is deliberate --
``hashlib.md5(open(t01, "rb").read()).hexdigest()`` is ``t01_md5_raw`` and
comparing THAT to the anchor tests the wall clock, not the physics.

The h3d refusal
---------------
``oracle_env.sh`` deliberately leaves ``RAD_H3D_PATH`` unset: the reachable
``libh3dwriter.so`` (extlib v59) is one parameter short of what the pinned
source calls, so a run that wrote H3D through it would produce silently wrong
files.  Without the variable, ``common_source/output/h3d/h3d_build_cpp/
h3d_dl.c:920-921`` cannot dlopen the writer and returns ``*IERROR = 1``, which
``engine/source/output/h3d/h3d_results/genh3d.F:729-731`` turns into MSGID 274
+ ``ARRET(2)``: H3D output is refused loudly and T01, the A-files, RESTART and
the listings are unaffected.  :func:`oracle_env` removes the variable from the
child environment and :func:`run_reference` reports what it actually was, so
the record carries the fact rather than the intention.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: The stored golden record (the deliverable of this task).
SMOKE_JSON = REPO / "tools" / "validation_data" / "oracle_smoke.json"
#: Where the golden run's admissible artefacts are committed, so the structural
#: tests can re-derive the maxima without an oracle.
GOLDEN_DIR = REPO / "tests" / "data" / "oracle_smoke"
#: The provenance record every parity claim cites.
PROVENANCE = REPO / "tools" / "validation_data" / "oracle_provenance.json"

SCHEMA = "pyradioss/oracle-smoke/1"

#: Deck pair used for the golden run.  ``examples/tensile_bar`` rather than a
#: vendored RD-* deck, for reasons recorded in ``deck.why`` of the record.
DECK_ROOT = REPO / "examples" / "tensile_bar"
RUN_NAME = "TENSILE"

#: ``ctime()`` field width copied by ``my_ctime`` (timer_c.c:36-37).
RUN_STAMP_LENGTH = 24
#: Marker written by ``my_ctime``: ' RADIOSS ' at ``CH80(25:33)``.
_RUN_STAMP_MARKER = b" RADIOSS "

#: The 23 global channels, transcribed from
#: ``starter/source/output/th/write_thnms1.F90:228-250`` -- the authoritative
#: index / short name / description table -- in the order ``hist2.F:307-333``
#: writes them.
GLOBAL_CHANNELS: Tuple[Tuple[int, str, str], ...] = (
    (1, "IE", "INTERNAL ENERGY"),
    (2, "KE", "KINETIC ENERGY"),
    (3, "XMOM", "X-MOMENTUM"),
    (4, "YMOM", "Y-MOMENTUM"),
    (5, "ZMOM", "Z-MOMENTUM"),
    (6, "MASS", "MASS"),
    (7, "DT", "TIME STEP"),
    (8, "RKE", "ROTATION ENERGY"),
    (9, "EFW", "EXTERNAL WORK"),
    (10, "SIE", "SPRING ENERGY"),
    (11, "CE", "CONTACT ENERGY"),
    (12, "HE", "HOURGLASS ENERGY"),
    (13, "CE_ELAST", "ELASTIC CONTACT ENERGY"),
    (14, "CE_FRIC", "FRICTIONAL CONTACT ENERGY"),
    (15, "CE_DAMP", "DAMPING CONTACT ENERGY"),
    (16, "WPLA", "PLASTIC WORK"),
    (17, "DMASS", "ADDED MASS"),
    (18, "DMASS%", "PERCENTAGE ADDED MASS"),
    (19, "M_IN", "INLET MASS"),
    (20, "M_OUT", "OUTLET MASS"),
    (21, "E_IN", "INLET ENERGY"),
    (22, "E_OUT", "OUTLET ENERGY"),
    (23, "DTE_INOUT", "IE+KE+RKE+CE+HE-EFW+E_IN-E_OUT"),
)

#: Per-part curve titles, ``varpa_title`` from
#: ``starter/source/output/th/th_titles.F90:2759-2792`` (1-based code -> title).
#: Only the entries this repo's golden deck can reach are transcribed; a code
#: outside the table is recorded with ``name: null`` and a reason, never guessed.
PART_CHANNEL_TITLES: Dict[int, str] = {
    1: "INTERNAL ENERGY",
    2: "KINETIC ENERGY",
    3: "X-MOMENTUM",
    4: "Y-MOMENTUM",
    5: "Z-MOMENTUM",
    6: "MASS",
    7: "HOURGLASS ENERGY",
    8: "TURBULENT ENERGY",
    21: "SHEAR INTERNAL ENERGY",
    32: "PLASTIC WORK",
}

#: Nodal (``/TH/NODE``) curve titles, ``varn1_title`` from
#: ``starter/source/output/th/th_titles.F90:168-189`` (1-based code -> title),
#: which is the table the TH-group codes index into.  The golden deck's
#: ``/TH/NODE/1 … DX VX`` (``examples/tensile_bar/TENSILE_0000.rad:183-186``)
#: is codes ``[1, 4]`` -> X-DISPLACEMENT, X-VELOCITY, and the decoded maxima
#: agree with the deck's pull (see ``t01.data_blocks_per_step``).
NODAL_CHANNEL_TITLES: Dict[int, str] = {
    1: "X-DISPLACEMENT",
    2: "Y-DISPLACEMENT",
    3: "Z-DISPLACEMENT",
    4: "X-VELOCITY",
    5: "Y-VELOCITY",
    6: "Z-VELOCITY",
    7: "X-ACCELERATION",
    8: "Y-ACCELERATION",
    9: "Z-ACCELERATION",
    10: "X-ROT VELOCITY",
    11: "Y-ROT VELOCITY",
    12: "Z-ROT VELOCITY",
    13: "X-ROT ACCELERATION",
    14: "Y-ROT ACCELERATION",
    15: "Z-ROT ACCELERATION",
    16: "X-COORDINATE",
    17: "Y-COORDINATE",
    18: "Z-COORDINATE",
    19: "TEMPERATURE",
}

#: T01 format code -> the title width the description records use
#: (``hist1.F:132-140``).  Note that the first record's title is hardcoded at 80
#: (``hist1.F:201-208``) whatever this says; see :func:`_title_width`.
_FORMAT_CODE_TITLE_WIDTH: Dict[int, int] = {4021: 100, 3050: 100, 3041: 80,
                                            3040: 40}

#: Per-step records, in ``hist2.F`` write order.
DATA_BLOCK_SOURCES = {
    "TIME": "engine/source/output/th/hist2.F:302-303",
    "GLOBAL": "engine/source/output/th/hist2.F:307-333",
    "PART": "engine/source/output/th/hist2.F:338-477",
    "SUBSET": "engine/source/output/th/hist2.F:478-606",
    "TH_GROUP": "engine/source/output/th/hist2.F:608-1403",
}

#: Where the run stamp sits inside its ``CH80`` record (``hist1.F:210-234``).
#: The window :func:`deterministic_md5` normalises is exactly the ctime field;
#: what FOLLOWS it stays in the digest, which is what makes the anchor
#: build- AND architecture-bound.
RUN_STAMP_TRAILER = (
    "CH80(25:33) = ' RADIOSS ', CH80(34:59) = VERSIO(2)(9:34), "
    "CH80(60:80) = CPUNAM"
)


# ---------------------------------------------------------------------------
# Oracle environment / discovery
# ---------------------------------------------------------------------------

def resolve_oracle() -> Tuple[Path, Path, Path]:
    """``(starter, engine, build_root)`` via :mod:`pyradioss.paths`.

    The single resolver owns the precedence (``$OR_STARTER``, then
    ``$OR_ROOT/bin/starter_linux64_gf`` ...); this function only adds the
    dev-box default ``~/OpenRadioss_or`` that ``paths.or_root()`` already
    carries, and raises ``FileNotFoundError`` when nothing resolves.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from pyradioss import paths

    return paths.or_starter(), paths.or_engine(), paths.or_build()


def oracle_env(build_root: Path) -> Dict[str, str]:
    """The child environment, built here rather than sourced from the caller's
    shell (``tools/oracle/oracle_env.sh`` is meant to be *sourced*, and a
    harness must not depend on the shell that started pytest).

    INSTALL.md:34-42 retargeted from ``OPENRADIOSS_PATH`` to the writable
    mirror, plus ``OMP_STACKSIZE`` (OpenRadioss stacks aggressively) and
    ``OMP_NUM_THREADS=1`` (INSTALL.md:95-101, a determinism guard).

    ``RAD_H3D_PATH`` is **removed**: see the module docstring.  The assertion at
    the end is the machine-checkable form of that decision.
    """
    env = dict(os.environ)
    env.pop("RAD_H3D_PATH", None)
    env["OPENRADIOSS_PATH"] = str(build_root)
    env["RAD_CFG_PATH"] = str(Path(build_root) / "hm_cfg_files")
    env["LD_LIBRARY_PATH"] = (
        str(Path(build_root) / "extlib" / "hm_reader" / "linux64")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["OMP_STACKSIZE"] = "400m"
    env["OMP_NUM_THREADS"] = "1"
    assert "RAD_H3D_PATH" not in env
    return env


# ---------------------------------------------------------------------------
# T01 binary reader
# ---------------------------------------------------------------------------

class T01FormatError(ValueError):
    """The file is not an ``ITTYP==3`` T01 this module knows how to read."""


def t01_records(blob: bytes) -> List[Tuple[int, bytes]]:
    """Split the file into ``(payload_offset, payload)`` records.

    Radioss IEEE framing: a 4-byte **big-endian** byte count, the payload, and
    the same count again (``eor_c``, ``write_routines.c:499-511``).  Both
    markers are checked, so a truncated or foreign file fails here instead of
    yielding plausible garbage.
    """
    records: List[Tuple[int, bytes]] = []
    offset = 0
    total = len(blob)
    while offset < total:
        if offset + 8 > total:
            raise T01FormatError(
                f"trailing {total - offset} byte(s) at {offset}: not a record "
                f"header (this parser reads the ITTYP==3 framing)"
            )
        (length,) = struct.unpack_from(">i", blob, offset)
        if length < 0 or offset + 8 + length > total:
            raise T01FormatError(
                f"record marker {length} at offset {offset} does not fit in "
                f"{total} bytes"
            )
        (trailing,) = struct.unpack_from(">i", blob, offset + 4 + length)
        if trailing != length:
            raise T01FormatError(
                f"record marker mismatch at offset {offset}: leading {length}, "
                f"trailing {trailing}"
            )
        records.append((offset + 4, blob[offset + 4:offset + 4 + length]))
        offset += 8 + length
    return records


def run_stamp_window(blob: bytes) -> Tuple[int, int]:
    """``(offset, length)`` of the ``ctime`` field inside the T01.

    ``hist1.F:210-234`` builds ``CH80`` from ``MY_CTIME`` and then ' RADIOSS ',
    the version and ``CPUNAM``, and writes it as one record.  The record is
    located by that marker (it is the only printable-ASCII record in the file),
    and the stamp is its first :data:`RUN_STAMP_LENGTH` bytes -- the width
    ``my_ctime`` copies (``timer_c.c:39``).
    """
    for offset, payload in t01_records(blob):
        if _RUN_STAMP_MARKER in payload and payload.isascii() and all(
            32 <= c < 127 for c in payload
        ):
            return offset, RUN_STAMP_LENGTH
    raise T01FormatError(
        "no run-stamp record found (a printable record containing "
        f"{_RUN_STAMP_MARKER!r}); hist1.F:210-234 writes one unconditionally, "
        "so its absence means this is not a T01 this module can date"
    )


def run_stamp_text(blob: bytes) -> str:
    """The stamp field AS READ, decoded -- never a literal.

    ``"ctime() -- " + repr(<the bytes the file actually carries>)``.  A record
    whose thesis is "every field is measured, not asserted" cannot ship a
    hardcoded example of its own header; :func:`test_p0_oracle_selftest` asserts
    this string against the committed golden T01 for exactly that reason.
    """
    offset, length = run_stamp_window(blob)
    return "ctime() -- " + repr(blob[offset:offset + length].decode("ascii"))


def deterministic_md5(blob: bytes) -> str:
    """md5 of the T01 with the wall-clock stamp zeroed -- the parity anchor.

    See "Determinism" in the module docstring for why the raw md5 cannot be
    used.  The window is *located*, never hardcoded, so the digest follows the
    file if ``hist1`` changes its header layout.
    """
    offset, length = run_stamp_window(blob)
    normalized = bytearray(blob)
    normalized[offset:offset + length] = b"\x00" * length
    return hashlib.md5(bytes(normalized)).hexdigest()


def _as_ints(payload: bytes) -> List[int]:
    if len(payload) % 4:
        raise T01FormatError(f"payload of {len(payload)} bytes is not int32")
    return [int(v) for v in np.frombuffer(payload, dtype=">i4")]


def _nglobth(records: Sequence[Tuple[int, bytes]]) -> Tuple[int, int]:
    """``(NGLOBTH, index)`` of the ``1..NGLOBTH`` curve-code record.

    ``hist1.F:311-316`` writes ``IWA(I) = I`` for ``I = 1..NGLOBTH`` right after
    the hierarchy record, so the record is found by CONTENT: its payload must be
    a run of big-endian int32 equal to ``1..n`` with ``n >= 8``.  ``NGLOBTH``
    therefore comes from the file instead of being assumed to be 23, and the
    index anchors the header walk (the hierarchy record is the one just before
    it, ``hist1.F:293-308``).
    """
    for index, (_, payload) in enumerate(records):
        if len(payload) < 32 or len(payload) % 4:
            continue
        values = _as_ints(payload)
        if values == list(range(1, len(values) + 1)):
            return len(values), index
    raise T01FormatError(
        "no 1..N integer record found; hist1.F:311-316 writes one right after "
        "the hierarchy record when the T01 is the global history file"
    )


def _title_width(records: Sequence[Tuple[int, bytes]]) -> int:
    """``LTITL``, the title width every description record uses.

    ``hist1.F:132-140`` picks the format code from ``TH_VERS`` and the title
    width with it; the format code is the first int32 of the first record, so
    the width is read, never guessed.  Note the one upstream inconsistency this
    module has to know about: the FIRST record's title is **hardcoded at 80**
    characters (``hist1.F:201-208``, ``EOR_C(84)``) regardless of ``LTITL``, so a
    3040 file carries an 84-byte title record and 40-byte descriptions.
    """
    code = _as_ints(records[0][1][:4])[0]
    try:
        return _FORMAT_CODE_TITLE_WIDTH[code]
    except KeyError:
        raise T01FormatError(
            f"unknown T01 format code {code} in record 0; hist1.F:132-140 "
            f"defines {_sorted(_FORMAT_CODE_TITLE_WIDTH)}"
        ) from None


def _header_codes(records: Sequence[Tuple[int, bytes]], nglo_index: int,
                  ltitl: int) -> Dict[str, List[List[int]]]:
    """Walk ``hist1.F``'s header and return the curve codes of every group.

    The order is ``hist1``'s own, so nothing is pattern-matched:

    * the 6-integer hierarchy record ``(NPART+NTHPART, NUMMAT, NUMGEO, NSUBS,
      NTHGRP2, NGLOBTH)`` (``hist1.F:293-308``) -- the record just before the
      ``1..NGLOBTH`` one;
    * ``NPART+NTHPART`` part blocks (``:318-377``): ``IPART(4,N)``, the title,
      ``IPART(7,N)``, the two bounds and ``NVAR`` (``:357-366``), then the
      ``NVAR`` curve codes (``:367-376``);
    * ``NUMMAT`` material blocks (``:379-411``) and ``NUMGEO`` geometry blocks
      (``:413-441``): an int32 and the title, nothing else;
    * ``NSUBS`` subset blocks (``:442-514``): five int32 then the title
      (``:493-499``), the child list (``:502``), the part list (``:506``), then
      the ``NVAR`` curve codes (``:509-513``);
    * ``NTHGRP2`` TH-group blocks (``:516-600``): five int32 then the title
      (``:545-553``), one int32+title per curve (``:556-584``), then the
      ``NVAR`` curve codes (``:585-587``).

    Anything else the header holds (the two additional records of
    ``hist1.F:237-291``, which only a ``TH_VERS>=50`` file carries, and the
    fluid section of ``:602-663``) sits OUTSIDE this walk and is reported as
    ``skipped`` rather than guessed at.
    """
    hierarchy = _as_ints(records[nglo_index - 1][1])
    if len(hierarchy) != 6:
        raise T01FormatError(
            f"the record before the 1..NGLOBTH one holds {len(hierarchy)} "
            f"int32, not the 6 of hist1.F:293-308"
        )
    n_part_tot, n_mat, n_geo, n_sub, n_grp, _ = hierarchy
    cursor = {"i": nglo_index + 1}

    def _take(index: int) -> bytes:
        if index >= len(records):
            raise T01FormatError(
                f"header ended before record {index}: the hierarchy record "
                f"claims more parts/materials/geometries/subsets/TH groups "
                f"than the file carries"
            )
        return records[index][1]

    def description(n_head: int, n_tail: int, who: str) -> List[int]:
        """One ``... ints, LTITL-character title, ... ints`` record."""
        payload = _take(cursor["i"])
        want = 4 * (n_head + n_tail) + ltitl
        if len(payload) != want:
            raise T01FormatError(
                f"header record {cursor['i']} ({who}) is {len(payload)} bytes, "
                f"expected {want} for hist1.F's description layout"
            )
        title = payload[4 * n_head:4 * n_head + ltitl]
        if not title.isascii():
            raise T01FormatError(
                f"header record {cursor['i']} ({who}) title is not ASCII"
            )
        ints = _as_ints(payload[:4 * n_head]) + _as_ints(payload[4 * n_head
                                                                + ltitl:])
        cursor["i"] += 1
        return ints

    def run(count: int, who: str) -> List[int]:
        """A plain ``count``-int32 record (a code list or a child/part list)."""
        if count <= 0:
            return []
        payload = _take(cursor["i"])
        if len(payload) != 4 * count:
            raise T01FormatError(
                f"header record {cursor['i']} does not hold the {count} int32 "
                f"of {who} (hist1.F)"
            )
        cursor["i"] += 1
        return _as_ints(payload)

    out: Dict[str, List[List[int]]] = {"part": [], "subset": [], "group": []}
    for n in range(n_part_tot):
        who = f"part {n + 1}"
        nvar = description(1, 4, who)[-1]              # hist1.F:357-366
        out["part"].append(run(nvar, f"{who} curve codes"))   # :367-376
    for n in range(n_mat):
        description(1, 0, f"material {n + 1}")         # hist1.F:401-411
    for n in range(n_geo):
        description(1, 0, f"geometry {n + 1}")         # hist1.F:433-441
    for n in range(n_sub):
        who = f"subset {n + 1}"
        head = description(5, 0, who)                  # hist1.F:493-499
        run(head[2], f"{who} children")                # :502
        run(head[3], f"{who} parts")                   # :506
        out["subset"].append(run(head[4], f"{who} curve codes"))   # :509-513
    for n in range(n_grp):
        who = f"TH group {n + 1}"
        head = description(5, 0, who)                  # hist1.F:545-553
        for j in range(head[3]):                       # ITHGRP(4,N) curves
            description(1, 0, f"{who} curve {j + 1}")  # :561-581
        out["group"].append(run(head[4], f"{who} curve codes"))   # :585-587
    out["hierarchy"] = hierarchy
    out["consumed"] = cursor["i"]
    return out


def parse_t01(blob: bytes) -> Dict:
    """Decode an ``ITTYP==3`` T01 into its header facts and its four step blocks.

    Per step ``hist2.F`` writes, in this order: ``TT`` (one value, :302-303),
    the ``NGLOBTH`` global channels (:307-333), the per-part values (:338-477),
    the subset values (:478-606, absent when every subset requests nothing) and
    the TH-group values (:608-1403).  The stride is validated across every step,
    so a shape this module did not expect is an error and not a mis-parsed.

    The returned ``part_codes`` / ``subset_codes`` / ``group_codes`` are per
    GROUP (a list per part / subset / TH group), read out of the header by
    :func:`_header_codes`; ``channels_per_step`` says how many values each block
    carries per step, so a multi-part model's codes and its block width can be
    compared by the caller.
    """
    records = t01_records(blob)
    if len(records) < 8:
        raise T01FormatError(f"only {len(records)} records; not a T01")
    nglo, nglo_index = _nglobth(records)
    glob_bytes = 4 * nglo
    ltitl = _title_width(records)

    start = None
    for index in range(len(records) - 3):
        if (len(records[index][1]) == 4
                and len(records[index + 1][1]) == glob_bytes):
            start = index
            break
    if start is None:
        raise T01FormatError(
            f"no data section: expected a 4-byte TT record followed by a "
            f"{glob_bytes}-byte global block (hist2.F:302-303, :307-333)"
        )
    header = _header_codes(records, nglo_index, ltitl)
    if header["consumed"] != start:
        # Not fatal -- hist1.F:237-291 (TH_VERS>=50 only) and :602-663 (fluid
        # sections) sit here -- but it is the difference between "I read the
        # whole header" and "I read the part I needed", so it is reported.
        header["unparsed_records"] = [
            len(payload) for _, payload in records[header["consumed"]:start]
        ]
    else:
        header["unparsed_records"] = []
    body = records[start:]
    if len(body) % 4:
        raise T01FormatError(
            f"data section has {len(body)} records, not a multiple of 4; the "
            "per-step block shape is not what hist2.F writes"
        )
    n_blocks = len(body) // 4
    shape = tuple(len(body[j][1]) for j in range(4))
    for k in range(1, n_blocks):
        other = tuple(len(body[4 * k + j][1]) for j in range(4))
        if other != shape:
            raise T01FormatError(
                f"per-step block {k} has record lengths {other}, expected "
                f"{shape}; the block shape is not uniform"
            )
    if shape[1] != glob_bytes or shape[0] != 4:
        raise T01FormatError(
            f"per-step block is {shape}, expected a 4-byte TT record followed "
            f"by the {glob_bytes}-byte global block (hist2.F:302-303, :307-333)"
        )

    def column(j: int) -> np.ndarray:
        rows = [np.frombuffer(body[4 * k + j][1], dtype=">f4")
                for k in range(n_blocks)]
        return np.array(rows, dtype=np.float64)

    return {
        "n_nglobth": nglo,
        "format_code": _as_ints(records[0][1][:4])[0],
        "title_width": ltitl,
        "header_records": start,
        "n_records": len(records),
        "run_stamp": run_stamp_window(blob),
        "run_stamp_text": run_stamp_text(blob),
        "hierarchy": header["hierarchy"],
        "header_unparsed_records": header["unparsed_records"],
        "n_steps": n_blocks,
        "channels_per_step": {
            "time": 1,
            "global": nglo,
            "part": shape[2] // 4,
            "th_group": shape[3] // 4,
        },
        "t_first": float(column(0)[0, 0]),
        "t_last": float(column(0)[-1, 0]),
        "time": column(0)[:, 0],
        "global": column(1),
        "part": column(2),
        "th_group": column(3),
        "part_codes": header["part"],
        "subset_codes": header["subset"],
        "group_codes": header["group"],
    }


def _stat(entry: Dict, series: np.ndarray) -> Dict:
    """``max`` / ``max_abs`` / ``final`` of one column.

    Values are float32 in the file (``wrtdes.F:121-133``) widened to float64,
    so these are exact, not rounded.
    """
    entry["max"] = float(series.max())
    entry["max_abs"] = float(np.abs(series).max())
    entry["final"] = float(series[-1])
    return entry


def _named_maxima(block: np.ndarray, codes: Sequence[int], titles: Dict[int, str],
                  table: str, who: str) -> List[Dict]:
    """Maxima of one per-step block, named from its curve codes.

    ``codes`` comes from the T01 header (:func:`_header_codes`) and ``titles``
    from the transcribed upstream table; a code the table does not carry gets
    ``name: null`` plus the reason, never an invented name.
    """
    if len(codes) != block.shape[1]:
        raise T01FormatError(
            f"{len(codes)} {who} curve code(s) for {block.shape[1]} {who} "
            f"channel(s) per step"
        )
    out: List[Dict] = []
    for column, code in enumerate(codes):
        title = titles.get(int(code))
        entry: Dict = {"index": int(code), "name": title}
        if title is None:
            entry["name_reason"] = (
                f"curve code {code} has no entry in the transcribed {table} "
                f"table; no name is invented"
            )
        out.append(_stat(entry, block[:, column]))
    return out


def channel_maxima(parsed: Dict) -> Dict:
    """Per-channel maxima of the golden run: global, per-part and TH-group.

    Exactly the structure stored in ``oracle_smoke.json`` under
    ``channel_maxima``; the structural tests compare the two with ``==``, so a
    key added here must be added to the record too.

    Three naming sources, all upstream tables, all keyed by the curve codes the
    T01 header records:

    * the 23 global channels -- ``write_thnms1.F90:228-250``;
    * the per-part channels -- ``varpa_title``, ``th_titles.F90:2759-2792``;
    * the TH-group (``/TH/NODE`` …) channels -- ``varn1_title``,
      ``th_titles.F90:168-189``.

    Subset channels are the one gap: ``hist2.F`` writes them only when a subset
    requests a curve (``:481``, ``II/=0`` at ``:606``), so there is no block for
    a deck that requests none, and this deck's single subset requests none.  A
    subset-bearing deck will raise here rather than be silently dropped -- add
    the table then.
    """
    nglo = parsed["n_nglobth"]
    if nglo != len(GLOBAL_CHANNELS):
        raise T01FormatError(
            f"the file carries {nglo} global channels but "
            f"{len(GLOBAL_CHANNELS)} are transcribed from "
            "write_thnms1.F90:228-250"
        )
    globals_out: List[Dict] = []
    for column, (index, name, description) in enumerate(GLOBAL_CHANNELS):
        globals_out.append(_stat(
            {"index": index, "name": name, "description": description},
            parsed["global"][:, column]))

    part_codes = parsed["part_codes"]
    if any(codes for codes in parsed["subset_codes"]):
        raise T01FormatError(
            "this T01 carries subset curve codes; per-subset maxima are not "
            "recorded (no per-subset block is decoded) -- transcribe the subset "
            "table and extend channel_maxima rather than losing them silently"
        )
    part_out = _named_maxima(
        parsed["part"], [c for codes in part_codes for c in codes],
        PART_CHANNEL_TITLES, "varpa_title", "part")
    group_out = _named_maxima(
        parsed["th_group"], [c for codes in parsed["group_codes"] for c in codes],
        NODAL_CHANNEL_TITLES, "varn1_title", "TH group")

    return {"global": globals_out, "part": part_out, "th_group": group_out}


def diff_offsets(left: bytes, right: bytes) -> List[int]:
    """Zero-based offsets at which two byte strings differ."""
    if len(left) != len(right):
        raise T01FormatError(
            f"cannot diff T01s of different size: {len(left)} vs {len(right)}"
        )
    array = np.frombuffer(left, dtype=np.uint8) != np.frombuffer(
        right, dtype=np.uint8)
    return [int(i) for i in np.nonzero(array)[0]]


# ---------------------------------------------------------------------------
# Running the oracle
# ---------------------------------------------------------------------------

def _parse_termination(text: str) -> Optional[str]:
    """The ``* TERMINATION`` banner the engine ends with.

    ``NORMAL TERMINATION`` on a clean run, ``ERROR TERMINATION`` when the engine
    aborts (for example MSGID 274 after the h3d refusal).  The LAST banner wins
    -- the engine prints exactly one at the end, and taking the last makes a
    mid-listing banner (there is none today) harmless.
    """
    found = re.findall(r"^\s*([A-Z][A-Z ]*TERMINATION)\s*$", text, re.M)
    return found[-1] if found else None


def _last_int(pattern: str, text: str) -> Optional[int]:
    found = re.findall(pattern, text, re.M)
    return int(found[-1]) if found else None


def _msg_errors(text: str) -> int:
    return len(re.findall(r"^\s*\*\*\s*ERROR", text, re.M))


def run_reference(run_name: str = RUN_NAME,
                  workdir=None,
                  *,
                  deck_root: Path = DECK_ROOT,
                  timeout: float = 900.0) -> Dict:
    """Run the oracle on ``<run_name>``'s deck pair and decode the result.

    ``workdir`` is created if needed and receives private copies of both decks
    plus every artefact the run writes (listings, T01, A-files, RESTART) --
    the private-scratch-directory shape upstream's own reference driver uses
    (``qa-tests/scripts/or_run_test.py:74-125``).  Nothing is written inside the
    repo and nothing is written into the deck's own directory.

    The returned dict is what ``oracle_smoke.json`` records; ``t01_bytes`` is
    kept so a caller can diff two runs without re-reading the files.
    """
    starter, engine, build_root = resolve_oracle()
    for binary in (starter, engine):
        if not binary.is_file():
            raise FileNotFoundError(f"oracle binary missing: {binary}")

    work = Path(workdir) if workdir is not None else Path("/tmp/opencode") / (
        "oracle_selftest_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    work.mkdir(parents=True, exist_ok=True)

    starter_deck = Path(deck_root) / f"{run_name}_0000.rad"
    engine_deck = Path(deck_root) / f"{run_name}_0001.rad"
    for deck in (starter_deck, engine_deck):
        if not deck.is_file():
            raise FileNotFoundError(f"deck missing: {deck}")
        shutil.copy2(deck, work / deck.name)

    env = oracle_env(build_root)
    starter_argv = [str(starter), "-i", starter_deck.name, "-np", "1"]
    # NO -np here: the engine's argument parser does not accept it (measured:
    # usage dump, then SIGSEGV).  INSTALL.md:111 passes no thread flag at all;
    # -nt 1 is the engine's own spelling and pins the reduction order.
    engine_argv = [str(engine), "-i", engine_deck.name, "-nt", "1"]

    timings = {}
    started = time.perf_counter()
    for label, argv in (("starter", starter_argv), ("engine", engine_argv)):
        t0 = time.perf_counter()
        proc = subprocess.run(argv, cwd=work, env=env, capture_output=True,
                              text=True, errors="replace", timeout=timeout)
        timings[label] = time.perf_counter() - t0
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr)[-2000:]
            raise RuntimeError(
                f"{Path(argv[0]).name} exited {proc.returncode} on "
                f"{starter_deck.name}:\n{tail}"
            )
        timings[label + "_tail"] = (proc.stdout + proc.stderr)[-4000:]

    starter_out = work / f"{run_name}_0000.out"
    engine_out = work / f"{run_name}_0001.out"
    if not starter_out.is_file() or not engine_out.is_file():
        raise RuntimeError(
            f"the oracle wrote no listing in {work}: "
            f"{sorted(p.name for p in work.iterdir())}"
        )
    starter_text = starter_out.read_text(errors="replace")
    engine_text = engine_out.read_text(errors="replace")

    banner = _parse_termination(engine_text)
    n_cycles = _last_int(r"TOTAL NUMBER OF CYCLES\s*:\s*(\d+)", engine_text)
    t01 = work / f"{run_name}T01"
    if not t01.is_file():
        raise RuntimeError(
            f"the engine reached {banner!r} but wrote no {t01.name} in {work}; "
            "the T01 is the channel this whole record is about"
        )
    blob = t01.read_bytes()
    parsed = parse_t01(blob)

    return {
        "run_name": run_name,
        "workdir": str(work),
        "deck": {
            "root": str(Path(deck_root).relative_to(REPO)),
            "starter_deck": str(starter_deck.relative_to(REPO)),
            "starter_deck_sha256": hashlib.sha256(
                starter_deck.read_bytes()).hexdigest(),
            "engine_deck": str(engine_deck.relative_to(REPO)),
            "engine_deck_sha256": hashlib.sha256(
                engine_deck.read_bytes()).hexdigest(),
        },
        "argv": {"starter": starter_argv, "engine": engine_argv},
        "rad_h3d_path": env.get("RAD_H3D_PATH"),
        "verdict": banner.split()[0] if banner else None,
        "verdict_banner": banner,
        "starter_verdict_banner": _parse_termination(starter_text),
        "n_cycles": n_cycles,
        "starter_msgerrors": _last_int(r"^\s*(\d+)\s+ERROR\(S\)", starter_text),
        "starter_warnings": _last_int(r"^\s*(\d+)\s+WARNING\(S\)", starter_text),
        "engine_msgerrors": _msg_errors(engine_text),
        "engine_msgerrors_definition":
            "the engine listing carries no 'ERROR(S)' summary; the count of "
            "'** ERROR' lines is used instead (a clean run has none, and a "
            "failed one ends in ERROR TERMINATION)",
        "starter_tail": timings["starter_tail"],
        "engine_tail": timings["engine_tail"],
        "t01_path": str(t01),
        "t01_size_bytes": len(blob),
        # M6: the long name is the point.  ``t01_md5_normalized`` is NOT
        # ``hashlib.md5(open(t01, "rb").read()).hexdigest()``; doing that
        # comparison silently tests the wall clock instead of the physics.
        "t01_md5_normalized": deterministic_md5(blob),
        "t01_md5_raw": hashlib.md5(blob).hexdigest(),
        "t01_bytes": blob,
        "run_stamp": parsed["run_stamp"],
        "run_stamp_text": parsed["run_stamp_text"],
        "t01_facts": {
            "format_code": parsed["format_code"],
            "title_width": parsed["title_width"],
            "header_records": parsed["header_records"],
            "n_records": parsed["n_records"],
            "header_unparsed_records": parsed["header_unparsed_records"],
            "hierarchy": parsed["hierarchy"],
            "n_steps": parsed["n_steps"],
            "t_first": parsed["t_first"],
            "t_last": parsed["t_last"],
            "channels_per_step": parsed["channels_per_step"],
            # per GROUP (a list per part / subset / TH group) -- the form the
            # header walk produces; the record flattens these for its
            # *_curve_codes keys, which a multi-part deck would have to read
            # through curve_codes_by_group instead.
            "part_curve_codes": parsed["part_codes"],
            "subset_curve_codes": parsed["subset_codes"],
            "th_group_curve_codes": parsed["group_codes"],
        },
        "channel_maxima": channel_maxima(parsed),
        "a_files": sorted(p.name for p in work.glob(f"{run_name}A[0-9]*")),
        "restart_files": sorted(p.name for p in work.glob("*.rst")),
        "wall_seconds": {
            "starter": timings["starter"],
            "engine": timings["engine"],
            "total": time.perf_counter() - started,
        },
    }


# ---------------------------------------------------------------------------
# The stored record
# ---------------------------------------------------------------------------

DECK_WHY = (
    "examples/tensile_bar (upstream's own tensile example, vendored in this "
    "repo) rather than a vendored RD-* deck, for four measured reasons. "
    "(1) THREADS: it is run with -np 1 / -nt 1, so no OpenMP reduction order "
    "can vary between the determinism runs; the RD-V-0020 and RD-E-4801 decks "
    "in tests/data/rd_decks ask for /PROC/8, which would put a thread-count-"
    "dependent summation order into the very bytes being pinned. (2) H3D: every "
    "RD-V-0020 engine deck requests /H3D/DT, and this oracle REFUSES h3d output "
    "by design (RAD_H3D_PATH unset -> h3d_dl.c:920-921 -> genh3d.F:729-731 "
    "MSGID 274 + ARRET), so those decks cannot reach NORMAL TERMINATION here at "
    "all. (3) COST: 1420 cycles, starter 0.30 s + engine 0.08 s of solver time "
    "and a 15 168-byte T01, so the three determinism runs cost about 1.2 s and "
    "the gate belongs in the default tier, not behind @pytest.mark.slow. "
    "(4) ADMISSIBILITY: oracle_provenance.json already records this deck pair "
    "as the measured evidence for the T01 / A-file / RESTART / energy-ledger "
    "channels, so the golden record extends proved evidence instead of opening "
    "a new one. Its sha256 is re-hashed by the test on every run."
)


def build_record(reference: Dict, repeats: Sequence[Dict],
                 provenance: Dict) -> Dict:
    """Assemble the stored record from a live reference run.

    ``repeats`` are additional runs of the same binary in their own scratch
    directories; their digests are the determinism evidence.  Two properties are
    required of them and neither is assumed:

    * every normalised digest equal -- otherwise the oracle is not
      bit-reproducible and there is nothing admissible to record;
    * at least TWO distinct raw digests -- otherwise the runs all landed inside
      one wall-clock second, which would make ``md5_raw_is_reproducible: false``
      an unproven claim in a record whose whole point is that its fields are
      measured.  :func:`write_record` is responsible for arranging that.
    """
    digests = ([reference["t01_md5_normalized"]]
               + [r["t01_md5_normalized"] for r in repeats])
    raw = [reference["t01_md5_raw"]] + [r["t01_md5_raw"] for r in repeats]
    if len(set(digests)) != 1:
        raise RuntimeError(
            f"the oracle is not bit-reproducible: {digests}. Refusing to write "
            "a golden record -- no parity claim in this program is admissible "
            "against a non-deterministic oracle."
        )
    if len(set(raw)) < 2:
        raise RuntimeError(
            f"all {len(raw)} runs share one wall-clock second, so the record's "
            "claim that the RAW digest is not reproducible would be unproven: "
            f"{raw}. Re-run `python -m tools.oracle.oracle_selftest --write` "
            "until the runs straddle a second boundary (write_record retries, "
            "but a run this fast can need a few attempts)."
        )
    offs = reference["run_stamp"]
    starter_path, engine_path, _ = resolve_oracle()
    return {
        "schema": SCHEMA,
        "purpose": (
            "The golden reference run of the real Fortran OpenRadioss on one "
            "vendored deck: the anchor every later differential parity claim in "
            "this program is measured against. Regenerate with "
            "`python -m tools.oracle.oracle_selftest --write`."
        ),
        "generated_by": "tools/oracle/oracle_selftest.py --write",
        "generated_on": _dt.date.today().isoformat(),
        "oracle": {
            "provenance": str(PROVENANCE.relative_to(REPO)),
            "upstream_git_sha": provenance["upstream_git_sha"],
            "built_by": "tools/oracle/build_oracle.sh",
            "third_party_prebuilt_binary": False,
            "starter": {
                "path": starter_path.as_posix(),
                "sha256": hashlib.sha256(starter_path.read_bytes()).hexdigest(),
            },
            "engine": {
                "path": engine_path.as_posix(),
                "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
            },
        },
        "deck": dict(reference["deck"], run_name=reference["run_name"],
                     why=DECK_WHY),
        "invocation": {
            "starter_argv": reference["argv"]["starter"],
            "engine_argv": reference["argv"]["engine"],
            "starter_argv_template": ["$OR_STARTER", "-i", "<run>_0000.rad",
                                      "-np", "1"],
            "engine_argv_template": ["$OR_ENGINE", "-i", "<run>_0001.rad",
                                     "-nt", "1"],
            "citation": "$OR_SRC/INSTALL.md:105-115 (see :110 and :111)",
            "why_not_np_on_the_engine": (
                "the engine's argument parser does not accept -np: it prints its "
                "usage and then dies with a segmentation violation (measured). "
                "Only -nt / -nthread work there. See oracle_provenance.json -> "
                "invocation."
            ),
            "shape": (
                "two separate invocations, the starter's return code checked "
                "before the engine runs, each in a private scratch directory -- "
                "the shape upstream's own reference driver uses "
                "($OR_SRC/qa-tests/scripts/or_run_test.py:74-125)"
            ),
        },
        "environment": {
            "rad_h3d_path": reference["rad_h3d_path"],
            "rad_h3d_path_unset": reference["rad_h3d_path"] is None,
            "rad_h3d_path_unset_reason": (
                "the reachable libh3dwriter.so (extlib v59) is one parameter "
                "short of what the pinned source calls, so a run that wrote H3D "
                "through it would produce silently wrong files. With the "
                "variable unset, h3d_dl.c:920-921 cannot dlopen the writer and "
                "returns *IERROR = 1, which genh3d.F:729-731 turns into MSGID "
                "274 + ARRET(2): h3d output is refused loudly. T01, the A-files, "
                "RESTART and the listings are unaffected. See "
                "oracle_provenance.json -> extlib.version_gaps.enforcement."
            ),
            "omp_num_threads": "1",
            "omp_stacksize": "400m",
            "openradioss_path": "$OR_BUILD (the writable mirror, not $OR_SRC)",
            "rad_cfg_path": "$OR_BUILD/hm_cfg_files",
            "ld_library_path": "$OR_BUILD/extlib/hm_reader/linux64",
            "citation": "$OR_SRC/INSTALL.md:34-42 and :95-101",
            "built_by": "tools/oracle/oracle_env.sh is the human-facing source; "
                        "this record's environment is built by "
                        "oracle_selftest.oracle_env() so the harness never "
                        "depends on the caller's shell",
            "anchor_scope": (
                "the normalised T01 digest is bound to the build AND the "
                "architecture, not only to the build: hist1.F:212-217 writes "
                f"{RUN_STAMP_TRAILER} into the same CH80 record, right after "
                "the ctime field this record normalises away. VERSIO(2) is the "
                "build's version banner and CPUNAM the platform string "
                "(here 'linuxa64 gf'), so an oracle rebuilt from another "
                "revision -- or the same revision built for another machine -- "
                "will not reproduce the anchor even though its physics is "
                "unchanged. Treat a digest mismatch as 're-derive with "
                "--write and say why', never as a physics regression."
            ),
        },
        "run": {
            "verdict": reference["verdict"],
            "verdict_banner": reference["verdict_banner"],
            "n_cycles": reference["n_cycles"],
            "starter_verdict_banner": reference["starter_verdict_banner"],
            "starter_msgerrors": reference["starter_msgerrors"],
            "starter_warnings": reference["starter_warnings"],
            "engine_msgerrors": reference["engine_msgerrors"],
            "engine_msgerrors_definition": reference[
                "engine_msgerrors_definition"],
            "a_files": reference["a_files"],
            "restart_files": reference["restart_files"],
            "wall_seconds": reference["wall_seconds"],
            "wall_seconds_note": (
                "one measured run on the dev box; an upper bound only, since "
                "absolute seconds are contention-sensitive. Cycles, verdict and "
                "all digests are contention-insensitive."
            ),
        },
        "t01": {
            "path": str((GOLDEN_DIR / f"{reference['run_name']}T01")
                        .relative_to(REPO)),
            "size_bytes": reference["t01_size_bytes"],
            "format": (
                "ITTYP==3 Radioss IEEE: big-endian 4-byte record markers, "
                "big-endian int32 headers, and SINGLE-PRECISION (float32) "
                "channel values even in this double-precision build "
                "(wrtdes.F:121-133). Stored maxima are float32 widened to "
                "float64, hence exact."
            ),
            "header_records": reference["t01_facts"]["header_records"],
            "n_records": reference["t01_facts"]["n_records"],
            "n_steps": reference["t01_facts"]["n_steps"],
            "t_first": reference["t01_facts"]["t_first"],
            "t_last": reference["t01_facts"]["t_last"],
            "channels_per_step": reference["t01_facts"]["channels_per_step"],
            "format_code": reference["t01_facts"]["format_code"],
            "title_width": reference["t01_facts"]["title_width"],
            "hierarchy": reference["t01_facts"]["hierarchy"],
            "hierarchy_definition": (
                "(NPART+NTHPART, NUMMAT, NUMGEO, NSUBS, NTHGRP2, NGLOBTH) -- "
                "hist1.F:293-308"
            ),
            "header_unparsed_records":
                reference["t01_facts"]["header_unparsed_records"],
            "part_curve_codes": [c for codes
                                 in reference["t01_facts"]["part_curve_codes"]
                                 for c in codes],
            "th_group_curve_codes": [c for codes
                                     in reference["t01_facts"]
                                     ["th_group_curve_codes"]
                                     for c in codes],
            "curve_codes_by_group": {
                "part": reference["t01_facts"]["part_curve_codes"],
                "subset": reference["t01_facts"]["subset_curve_codes"],
                "th_group": reference["t01_facts"]["th_group_curve_codes"],
            },
            "curve_codes_note": (
                "each *_curve_codes list is the codes of every group of that "
                "kind CONCATENATED (this deck has one of each, so the flat list "
                "and the single group's list coincide); curve_codes_by_group is "
                "the authoritative per-group form and is what the header walk "
                "actually produces. It walks hist1.F's own block order "
                "(:357-376 parts, :493-513 subsets, :545-587 TH groups) from "
                "the hierarchy record. header_unparsed_records is empty for "
                "this deck, so the whole header was accounted for."
            ),
            "data_blocks_per_step": [
                {"name": "TIME", "source": DATA_BLOCK_SOURCES["TIME"],
                 "floats": 1, "maxima_recorded": True},
                {"name": "GLOBAL", "source": DATA_BLOCK_SOURCES["GLOBAL"],
                 "floats": reference["t01_facts"]["channels_per_step"]["global"],
                 "maxima_recorded": True},
                {"name": "PART", "source": DATA_BLOCK_SOURCES["PART"],
                 "floats": reference["t01_facts"]["channels_per_step"]["part"],
                 "maxima_recorded": True},
                {"name": "TH_GROUP", "source": DATA_BLOCK_SOURCES["TH_GROUP"],
                 "floats": reference["t01_facts"]["channels_per_step"]
                 ["th_group"],
                 "maxima_recorded": True},
                {"name": "SUBSET", "source": DATA_BLOCK_SOURCES["SUBSET"],
                 "floats": 0,
                 "maxima_recorded": False,
                 "maxima_reason": (
                     "hist2.F writes a per-subset block only when the subset "
                     "requests a curve (:481, II/=0 at :606); this deck's "
                     "single subset requests none, so there is no block and the "
                     "header records an empty code list for it. A subset-bearing "
                     "deck makes channel_maxima() raise rather than be dropped "
                     "silently.")},
            ],
            "run_stamp": {
                "offset": offs[0],
                "length": offs[1],
                "content": reference["run_stamp_text"],
                "citation": "engine/source/output/th/hist1.F:210-234 with "
                            "engine/source/system/timer_c.c:30-40",
                "note": (
                    "OFFSET is located by content (the printable record "
                    "carrying ' RADIOSS '), never hardcoded; LENGTH is the "
                    "constant RUN_STAMP_LENGTH = 24, which is how many "
                    "characters my_ctime copies (timer_c.c:39). The record "
                    "itself is 80 bytes; only its first 24 are the clock."
                ),
                "trailer": RUN_STAMP_TRAILER,
                "trailer_note": (
                    "what FOLLOWS the stamp stays inside the normalised "
                    "digest -- see environment.anchor_scope"
                ),
            },
            "md5_normalized": reference["t01_md5_normalized"],
            "md5_definition": (
                "t01.md5_normalized = md5 of the T01 bytes with the 24-byte "
                "ctime run stamp at t01.run_stamp.offset replaced by 24 NUL "
                "bytes. This is the parity anchor. It is NOT "
                "hashlib.md5(the T01 bytes) -- that is t01.md5_raw and it "
                "moves with the wall clock."
            ),
            "md5_raw": reference["t01_md5_raw"],
            "md5_raw_is_reproducible": False,
            "md5_raw_reason": (
                "hist1.F:211 stamps ctime() into the T01 header "
                "(timer_c.c:30-40, the 24-character copy at :39) and no keyword "
                "or environment variable "
                "suppresses it, so the raw bytes move with the wall clock: "
                "measured, two runs sharing a wall-clock second produced the "
                "same raw md5 and a run one second away differed in exactly one "
                "byte, the seconds digit of the stamp. A minute carry changes "
                "three (offsets 112/114/115), all inside the window. The "
                "physics payload is reproducible -- see determinism below, and "
                "distinct_md5_raw below for the raw variation that proves it."
            ),
            "md5_raw_observed": raw,
        },
        "determinism": {
            "method": (
                f"the reference was run {len(digests)} times in {len(digests)} "
                "separate scratch directories with one thread; the normalized "
                "digest must be identical every time, every raw differing byte "
                "must lie inside the 24-byte run stamp, and at least two "
                "distinct RAW digests must be observed so the "
                "non-reproducibility claim is measured rather than asserted"
            ),
            "runs": len(digests),
            "md5": digests,
            "md5_raw": raw,
            "distinct_md5": sorted(set(digests)),
            "distinct_md5_raw": sorted(set(raw)),
            "distinct_md5_raw_count": len(set(raw)),
            "conclusion": (
                "the oracle is bit-reproducible given a fixed run stamp; the "
                "raw T01 bytes are not, and the normalized digest is therefore "
                "the anchor. tests/test_p0_oracle_selftest.py asserts all three "
                "halves -- equal normalized digests, every raw differing byte "
                "inside the stamp window, and at least two distinct RAW digests "
                "so the non-reproducibility claim is measured -- and FAILS "
                "(never skips) when the oracle is present and non-deterministic."
            ),
        },
        "channel_maxima": reference["channel_maxima"],
        "notes": [
            "The 23 global channels are named from "
            "$OR_SRC/starter/source/output/th/write_thnms1.F90:228-250, the "
            "authoritative index/name/description table, in the order "
            "hist2.F:307-333 writes them.",
            "The part channels are named from varpa_title "
            "($OR_SRC/starter/source/output/th/th_titles.F90:2759-2792) keyed "
            "by the curve codes the deck's /TH/PART block requested, which this "
            "module reads out of the T01 header (hist1.F:367-376).",
            "The TH-group channels are named from varn1_title "
            "($OR_SRC/starter/source/output/th/th_titles.F90:168-189), keyed "
            "by the codes of hist1.F:585-587. The golden deck's "
            "/TH/NODE/1 ... DX VX (examples/tensile_bar/TENSILE_0000.rad:183-186)"
            " is codes [1, 4] -> X-DISPLACEMENT, X-VELOCITY; the decoded "
            "maxima (0.1880658119916916 and 1.0) are the node's pull, which is "
            "what the deck's boundary condition imposes.",
            "MAXIMA ARE THE SEED, NOT A TOLERANCE: a later phase compares its "
            "own T01 against these numbers with a stated tolerance, or -- "
            "better -- against t01.md5_normalized, or the committed "
            "tests/data/oracle_smoke/<run>T01 itself.",
            "Only channels oracle_provenance.json marks admissible are "
            "recorded. H3D is deliberately absent (refused at run time), and "
            "the starter's include-file list is not a channel this record "
            "touches.",
        ],
        "not_established": {
            "th_to_csv": None,
            "h3d": None,
            "starter_include_file_list": None,
            "native_lsdyna_k_deck_reading": None,
            "ale_structured_mesh": None,
            "checksum_report_over_h3d": None,
        },
        "not_established_reasons": {
            "th_to_csv": (
                "upstream's T01->CSV converter is not built on this box: "
                "$OR_ROOT/bin holds only starter_linux64_gf and engine_linux64_gf. "
                "That is why this module carries its own T01 reader rather than "
                "shelling out to th_to_csv. Source: "
                "$OR_SRC/tools/th_to_csv/README.md:1-7."
            ),
            "h3d": (
                "REFUSED at run time, and inadmissible: "
                "oracle_provenance.json -> admissible_parity_evidence['H3D "
                "animation files'] = no, for three independent ABI reasons."
            ),
            "starter_include_file_list": (
                "oracle_provenance.json -> admissible_parity_evidence"
                "['starter .out include-file list'] = no (the harvested v59 "
                "reader does not reproduce a pristine include listing)."
            ),
            "native_lsdyna_k_deck_reading": (
                "oracle_provenance.json -> admissible_parity_evidence['native "
                "LS-DYNA .k deck reading (hm_reader)'] = no: no native .k "
                "fixture has ever been run through this oracle, so nothing is "
                "claimed. This record's deck is Radioss format."
            ),
            "ale_structured_mesh": (
                "oracle_provenance.json -> admissible_parity_evidence"
                "['/ALE/STRUCTURED_MESH (S-ALE)'] = no."
            ),
            "checksum_report_over_h3d": (
                "oracle_provenance.json -> admissible_parity_evidence"
                "['/CHECKSUM_REPORT over H3D files'] = no."
            ),
        },
    }


#: How many times :func:`write_record` will try to straddle a wall-clock second
#: before giving up.  Each attempt is ``1 + repeats`` runs of a deck that takes
#: 0.4 s, so an attempt is ~1.6 s and this bound is ~30 s of writing in the
#: pathological case where every run lands inside the same second.
WRITE_ATTEMPTS = 12


def write_record(*, repeats: int = 3, attempts: int = WRITE_ATTEMPTS,
                 workdir: Optional[Path] = None) -> Dict:
    """Run the reference ``1 + repeats`` times and write the record + artefacts.

    The extra runs exist only to fill ``determinism``; the recorded artefacts are
    the first run's.  Two things abort the write rather than produce a record:

    * the normalised digests disagree -- the oracle is not bit-reproducible, so
      there is nothing admissible to record;
    * every run landed inside ONE wall-clock second -- the record's central
      non-reproducibility claim would then be an assertion instead of a
      measurement.  Attempts are retried (bounded by ``attempts``) because a
      0.4 s deck can plausibly finish four runs inside one second; only a
      genuinely frozen clock reaches the bound, and then the write is refused.
    """
    base = Path(workdir) if workdir else Path(
        "/tmp/opencode/oracle_selftest_record")
    base.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(PROVENANCE.read_text())

    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        suffix = "" if attempt == 1 else f"_try{attempt}"
        reference = run_reference(RUN_NAME, workdir=base / f"run0{suffix}")
        extra = [run_reference(RUN_NAME, workdir=base / f"run{i + 1}{suffix}")
                 for i in range(repeats)]
        try:
            record = build_record(reference, extra, provenance)
        except RuntimeError as exc:
            if "share one wall-clock second" not in str(exc):
                raise
            last_error = exc
            print(f"attempt {attempt}/{attempts}: {exc}", file=sys.stderr)
            continue
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference["t01_path"], GOLDEN_DIR / f"{RUN_NAME}T01")
        shutil.copy2(Path(reference["workdir"]) / f"{RUN_NAME}_0001.out",
                     GOLDEN_DIR / f"{RUN_NAME}_0001.out")
        SMOKE_JSON.parent.mkdir(parents=True, exist_ok=True)
        SMOKE_JSON.write_text(json.dumps(record, indent=2) + "\n")
        return record
    raise RuntimeError(
        f"refusing to write {SMOKE_JSON}: after {attempts} attempts no run "
        f"pair straddled a wall-clock second, so the record could not measure "
        f"the raw digest's non-reproducibility. Last error: {last_error}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"regenerate {SMOKE_JSON} and "
                             f"{GOLDEN_DIR}/ from live oracle runs")
    parser.add_argument("--repeats", type=int, default=3,
                        help="determinism repeats per attempt for --write "
                             "(default 3; the record then carries 4 runs)")
    parser.add_argument("--attempts", type=int, default=WRITE_ATTEMPTS,
                        help="how many times --write retries until two runs "
                             "straddle a wall-clock second (default 12)")
    args = parser.parse_args(argv)

    if not args.write:
        parser.print_help()
        return 0
    record = write_record(repeats=args.repeats, attempts=args.attempts)
    print(f"wrote {SMOKE_JSON}")
    print(f"wrote {GOLDEN_DIR}/{RUN_NAME}T01 "
          f"({record['t01']['size_bytes']} bytes, normalized md5 "
          f"{record['t01']['md5_normalized']})")
    print(f"verdict {record['run']['verdict_banner']!r}, "
          f"{record['run']['n_cycles']} cycles, "
          f"{record['run']['wall_seconds']['total']:.2f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
