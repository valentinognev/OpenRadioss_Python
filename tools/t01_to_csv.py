#!/usr/bin/env python3
# Copyright (C) 2026 the pyradioss contributors.
# This file is part of pyradioss (OpenRadioss_Python), a Python port of the
# OpenRadioss explicit FEM solver, and is licensed under the GNU Affero
# General Public License, version 3 or later (AGPL-3.0-or-later).
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.  See LICENSE for the full text.
#
# Derived from OpenRadioss (Copyright (C) 2026 Siemens, AGPL-3.0-or-later):
#   engine/source/output/th/hist1.F  (header records: format code 132-144,
#     title record 201-208, run stamp 210-234, hierarchy 292-316, part
#     descriptions 357-376, subset descriptions 442-514, TH-group
#     descriptions 516-600)
#   engine/source/output/th/hist2.F  (per-step records: TT 302-303, globals
#     307-333, parts 338-477, subsets 478-607, TH groups 608-1403, each group
#     NN*NVAR values member-major)
#   engine/source/output/th/thkin.F  (member-major order, 51-60)
#   common_source/tools/input_output/write_routines.c  (eor_c 499-511 framing,
#     write_i_c 646-664 big-endian int32, write_r_c 520-540 values)
#   starter/source/output/th/write_thnms1.F90  (the 23 global channel
#     descriptions, 230-252)
#   starter/source/output/th/th_titles.F90  (varpa_title, 2759-2792)
"""A compatible ``th_to_csv``: render a Fortran binary T01 as CSV.

Upstream's own converter lives in the separate ``OpenRadioss/Tools``
repository (``$OR_SRC/tools/th_to_csv/README.md:1-7``), which this box cannot
reach, and neither ``starter`` nor ``engine`` builds it -- so
``$OR_ROOT/bin`` holds only the two solvers and every parity run degrades to
``FORTRAN-FAIL(th2csv-missing)``.  This module is the compatible
reimplementation the harness resolves as ``th_to_csv`` instead: it decodes
the very same ``ITTYP==3`` binary (``hist1.F`` for the header,
``hist2.F`` for the per-step records) and writes ``<T01>.csv`` beside it.

Deliberate scope, stated so a reader never mistakes this for upstream's
binary:

* **Standard library only** (``struct``/``csv``).  The installed copy at
  ``$OR_ROOT/bin/th_to_csv_linux64_gf`` is byte-identical to this file and
  must run with no repository on ``sys.path`` -- so this module imports
  nothing from ``tools`` and duplicates the dozen lines of record framing
  instead of sharing them.  Equivalence with
  :func:`tools.compare_t01.read_t01` is pinned by
  ``tests/test_p0_parity_comparison_route.py`` on the committed golden T01,
  not by sharing code.
* **The named channel set is exactly ``read_t01``'s**: the 23 global
  channels plus one column per part curve code.  Subset and TH-group records
  are walked (the stride is derived from the header, so a six-record step
  converts like a four-record one) but never rendered -- upstream ships no
  name table for them without the object-type dispatch ``hist2.F:608-1403``
  performs, and no channel name is invented here.
* **Header spellings are chosen for the harness**, not copied from
  upstream's binary (which was never observable on this box): global columns
  carry the ``write_thnms1.F90`` descriptions, so
  ``tools/validate_vs_fortran.py`` ``GLOBAL_MAP`` resolves on them, and part
  columns are ``PART <position> IE|KE`` (``varpa_title`` codes 1 and 2),
  so its ``part_IE``/``part_KE`` suffix rule resolves when the deck carries
  exactly one of each.  Values are ``float32`` widened to text at ``.17g``,
  an exact round-trip.

Usage: ``th_to_csv <T01>`` -- one positional argument, no flags (the same
argv ``tools.validate_vs_fortran.th_to_csv_argv`` builds).  The CSV is
written to ``<T01>.csv`` next to the input; exit 0 on success, exit 1 with
the reason on stderr otherwise.
"""

from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

#: ``hist1.F:132-144`` -- format code -> title width (``LTITL``).
FORMAT_CODE_TITLE_WIDTH = {3040: 40, 3041: 80, 3050: 100, 4021: 100}

#: The 23 global channels in write order (``hist2.F:307-333``), transcribed
#: from ``starter/source/output/th/write_thnms1.F90:230-252``: the
#: authoritative index / short name / description table.  The CSV header
#: carries the descriptions, which is what the harness's ``GLOBAL_MAP``
#: matches on.
GLOBAL_CHANNELS = (
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

#: Part curve code -> the CSV suffix (``varpa_title``,
#: ``th_titles.F90:2759-2762``: 1 INTERNAL ENERGY, 2 KINETIC ENERGY).
PART_SUFFIX = {1: "IE", 2: "KE"}


class T01Error(ValueError):
    """The file is not an ``ITTYP==3`` T01 this converter can render."""


def _records(blob: bytes):
    """Split the file into record payloads (``eor_c`` framing, big-endian).

    Every record is a 4-byte big-endian byte count, the payload, and the
    same count again (``write_routines.c:499-511``).  Both markers are
    checked, so a truncated or foreign file fails here.
    """
    out = []
    offset, total = 0, len(blob)
    while offset < total:
        if offset + 8 > total:
            raise T01Error(
                f"trailing {total - offset} byte(s) at {offset}: not a record")
        (length,) = struct.unpack_from(">i", blob, offset)
        if length < 0 or offset + 8 + length > total:
            raise T01Error(
                f"record marker {length} at offset {offset} does not fit "
                f"in {total} bytes")
        (trailing,) = struct.unpack_from(">i", blob, offset + 4 + length)
        if trailing != length:
            raise T01Error(
                f"record marker mismatch at offset {offset}: leading "
                f"{length}, trailing {trailing}")
        out.append(blob[offset + 4:offset + 4 + length])
        offset += 8 + length
    return out


def _as_ints(payload: bytes):
    if len(payload) % 4:
        raise T01Error(f"payload of {len(payload)} bytes is not int32")
    return list(struct.unpack(f">{len(payload) // 4}i", payload))


def _find_hierarchy(records):
    """``(hierarchy, nglo, index)`` of the ``1..NGLOBTH`` curve-code record.

    Found by content (``hist1.F:311-316`` writes ``IWA(I) = I`` right after
    the hierarchy record), accepted only when the record ahead of it yields
    six int32 whose sixth equals ``N`` (``hist1.F:300-316`` writes
    ``IWA(6) = NGLOBTH`` there) -- a part curve-code record is also a run of
    int32 and must not match.
    """
    for index, payload in enumerate(records):
        if len(payload) < 32 or len(payload) % 4 or index == 0:
            continue
        values = _as_ints(payload)
        if values != list(range(1, len(values) + 1)):
            continue
        if len(records[index - 1]) < 24:
            continue
        hierarchy = _as_ints(records[index - 1][:24])
        if len(hierarchy) != 6 or hierarchy[5] != len(values):
            continue
        return tuple(hierarchy), len(values), index
    raise T01Error(
        "no 1..N curve-code record preceded by a hierarchy record was found; "
        "hist1.F:311-316 writes one right after the 6-integer hierarchy "
        "record when the T01 is the global history file")


def _convert(blob: bytes):
    """``(header, rows)`` -- the CSV header and one float row per step."""
    records = _records(blob)
    if len(records) < 8:
        raise T01Error(f"only {len(records)} records; not a T01")
    if len(records[0]) != 84:
        raise T01Error(
            f"header record 1 is {len(records[0])} bytes, not the 84 of "
            "hist1.F:201-208 (EOR_C(84), the format code, 80 title bytes)")
    code = _as_ints(records[0][:4])[0]
    if code not in FORMAT_CODE_TITLE_WIDTH:
        raise T01Error(
            f"unknown T01 format code {code}; hist1.F:132-144 writes "
            f"{sorted(FORMAT_CODE_TITLE_WIDTH)}")
    ltitl = FORMAT_CODE_TITLE_WIDTH[code]

    hierarchy, nglo, code_index = _find_hierarchy(records)
    if nglo != len(GLOBAL_CHANNELS):
        raise T01Error(
            f"the file carries {nglo} global channels but "
            f"{len(GLOBAL_CHANNELS)} are transcribed from "
            "write_thnms1.F90:230-252")
    n_part_tot, n_mat, n_geo, n_sub, n_grp, _ = hierarchy

    cursor = code_index + 1

    def take() -> bytes:
        nonlocal cursor
        if cursor >= len(records):
            raise T01Error(
                "header ended before the hierarchy record's parts/materials/"
                "geometries/subsets/TH groups were all read")
        payload = records[cursor]
        cursor += 1
        return payload

    def description(n_head: int, n_tail: int, who: str):
        payload = take()
        want = 4 * (n_head + n_tail) + ltitl
        if len(payload) != want:
            raise T01Error(
                f"header record for {who} is {len(payload)} bytes, expected "
                f"{want} for hist1.F's description layout")
        return (_as_ints(payload[:4 * n_head])
                + _as_ints(payload[4 * n_head + ltitl:]))

    def run(count: int, who: str):
        if count <= 0:
            return []
        payload = take()
        if len(payload) != 4 * count:
            raise T01Error(
                f"header record does not hold the {count} int32 of {who}")
        return _as_ints(payload)

    part_codes = []
    for n in range(n_part_tot):
        nvar = description(1, 4, f"part {n + 1}")[-1]       # hist1.F:357-366
        part_codes.append(run(nvar, f"part {n + 1} curve codes"))  # :367-376
    for n in range(n_mat):
        description(1, 0, f"material {n + 1}")              # hist1.F:401-411
    for n in range(n_geo):
        description(1, 0, f"geometry {n + 1}")              # hist1.F:433-441
    subset_nvar = 0
    for n in range(n_sub):
        head = description(5, 0, f"subset {n + 1}")         # hist1.F:493-499
        run(head[2], f"subset {n + 1} children")            # :502
        run(head[3], f"subset {n + 1} parts")               # :506
        subset_nvar += len(run(head[4],                        # :509-513
                               f"subset {n + 1} curve codes"))
    group_widths = []
    for n in range(n_grp):
        head = description(5, 0, f"TH group {n + 1}")       # hist1.F:545-553
        for j in range(head[3]):
            description(1, 0, f"TH group {n + 1} curve {j + 1}")  # :561-581
        codes = run(head[4], f"TH group {n + 1} curve codes")  # :585-587
        if codes:
            # hist2.F writes NN*NVAR values member-major per group (thkin.F,
            # hist2.F:1015); the header lists the NVAR variable codes once.
            group_widths.append(head[3] * len(codes))

    glob_bytes = 4 * nglo
    start = None
    for index in range(len(records) - 1):
        if len(records[index]) == 4 and len(records[index + 1]) == glob_bytes:
            start = index
            break
    if start is None:
        raise T01Error(
            "no data section: expected a 4-byte TT record followed by the "
            f"{glob_bytes}-byte global block (hist2.F:302-303, :307-333)")
    n_part = sum(len(codes) for codes in part_codes)
    expected = [4, glob_bytes]
    if n_part:
        expected.append(4 * n_part)
    if subset_nvar:
        expected.append(4 * subset_nvar)
    expected.extend(4 * width for width in group_widths)
    stride = len(expected)
    body = records[start:]
    if len(body) % stride:
        raise T01Error(
            f"data section has {len(body)} records, not a multiple of "
            f"{stride}")
    n_steps = len(body) // stride
    for k in range(n_steps):
        other = tuple(len(body[stride * k + j]) for j in range(stride))
        if other != tuple(expected):
            raise T01Error(
                f"per-step block {k} has record lengths {other}, expected "
                f"{tuple(expected)}; the block shape is not uniform")

    def values(step: int, slot: int):
        payload = body[stride * step + slot]
        return struct.unpack(f">{len(payload) // 4}f", payload)

    slot = 2
    part_slot = slot if n_part else None
    slot += 1 if n_part else 0
    slot += 1 if subset_nvar else 0

    header = ["Time"] + [description_ for _, _, description_ in GLOBAL_CHANNELS]
    for position, codes in enumerate(part_codes, start=1):
        for code in codes:
            header.append(f"PART {position} "
                          f"{PART_SUFFIX.get(int(code), f'CODE{int(code)}')}")

    rows = []
    for step in range(n_steps):
        time = values(step, 0)[0]
        row = [time] + list(values(step, 1))
        if part_slot is not None:
            row.extend(values(step, part_slot))
        rows.append(row)
    return header, rows


def convert_file(t01_path: str) -> str:
    """Render ``t01_path`` to ``<t01_path>.csv``; return the CSV path."""
    source = Path(t01_path)
    header, rows = _convert(source.read_bytes())
    target = Path(str(source) + ".csv")
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(f"{value:.17g}" for value in row)
    return str(target)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: th_to_csv <T01>", file=sys.stderr)
        return 2
    try:
        convert_file(args[0])
    except (T01Error, OSError) as exc:
        print(f"th_to_csv: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
