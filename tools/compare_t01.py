#!/usr/bin/env python
"""Binary T01 reader and per-channel rel-RMS scorer.

Two deliverables, one purpose: turn "two runs" into **one comparable number**.

* :func:`read_t01` decodes the Fortran engine's binary time-history file -- the
  *T01* -- into a :class:`T01` (channel names, sample times, an
  ``(n_times, n_channels)`` value matrix, the header's unit-scaling triples).
  This is the record layout Phase 12 has to write; it lands here because the
  scorer needs it and because upstream's own converter (``th_to_csv``) is not
  obtainable on this box (see "Why this module exists" below).
* :func:`score` compares two :class:`T01` series channel by channel and returns
  one :class:`Score` whose :attr:`Score.worst` is the single number a parity
  claim rests on, with :data:`MATCH_RMS` as the MATCH/DEVIATION threshold.

Why this module exists (and why it reuses another one)
------------------------------------------------------
The brief's own cross-check for the reader was upstream's ``th_to_csv``, whose
**source** lives in a separate repository (``tools/th_to_csv/README.md:1-7``,
``OpenRadioss/Tools``) that this box cannot reach, and which neither
``starter/CMakeLists.txt`` nor ``engine/CMakeLists.txt`` builds.  The reader is
therefore implemented here and validated three ways instead (see
``tests/test_p0_compare_t01.py``): against the cited upstream sources, against
the committed golden T01 of the P0.5 reference run, and against the port's own
ASCII T01 CSV for the same deck.

The **framing** -- the record split, the run-stamp locator, the data-section
walk and the whole-file decode -- is *not* re-implemented here.  It is imported
from :mod:`tools.oracle.oracle_selftest`, which is where it was written and
tested for the golden record (``tests/test_p0_oracle_selftest.py::
test_stored_golden_t01_reproduces_the_stored_maxima``).  One record walk in the
program, two consumers.  What this module adds on top is the part the golden
record has no use for: the channel **names**, the header facts (format code,
title width, unit scaling), upstream's exact value decoder, and the scorer.
Generalising the per-step record stride belongs in that shared walk, not in a
second reader; see "Known limitation" below.

Upstream Fortran origins (``$OR_SRC`` = the read-only OpenCourant tree)
----------------------------------------------------------------------
Every structural constant below is listed in :data:`LAYOUT` with the upstream
``file:line`` range it comes from and a pattern that must still be found there
(``tests/test_p0_compare_t01.py::test_layout_constants_match_the_cited_upstream_source``
checks all of them against the real sources whenever ``$OR_SRC`` is readable):

* ``common_source/tools/input_output/write_routines.c:499-510`` (``eor_c``) --
  every record is framed by a **4-byte big-endian byte count**, written twice
  (opening and closing marker).  Both markers are verified, so a truncated or
  foreign file raises instead of decoding to plausible garbage.
* ``:646-663`` (``write_i_c``) -- header integers are big-endian ``int32``
  (``integer_to_IEEE_ASCII``, ``engine/source/output/tools/ieee.cpp:40-46``).
* ``:520-539`` (``write_r_c``) -> ``real_to_IEEE_ASCII``
  (``ieee.cpp:66-124``) -- a stored value is sign + 8-bit exponent +
  **24**-bit mantissa, big-endian.  The read side is ``read_r_c``
  (``:757-796``) -> ``IEEE_ASCII_to_real`` (``ieee.cpp:127-165``), which
  :func:`decode_radioss_float` is a vectorised transcription of, and which
  decodes this module's header ``FAC`` record.
  **Note:** this is *not* a stock IEEE binary32 -- the mantissa has 24 bits,
  the last of which a stock ``>f4`` read discards.  It does not matter for the
  files the engine writes, because ``wrtdes.F:121-133`` funnels every value
  through ``R4`` (a real*4) first and a binary32 value has an all-zero 24th
  mantissa bit; the channel values themselves are read by the shared walk as
  ``>f4``, and
  ``test_read_t01_sample_count_and_maxima_match_the_golden_record`` asserts the
  two decodings are bit-identical over all 100 samples of the golden file
  instead of taking the argument on trust.
* ``engine/source/output/th/wrtdes.F:121-133`` -- ``ITTYP==3`` writes a value
  record as ``EOR_C(4*L)`` + ``L`` values + ``EOR_C(4*L)``, so a value record is
  always a multiple of 4 bytes.
* ``engine/source/output/th/hist1.F:132-144`` -- the **format code** selects the
  title width: ``3040`` -> ``LTITL=40``, ``3041`` -> 80, ``3050`` and ``4021``
  -> 100 (plus the two additional records of ``:237-290``).
* ``engine/source/output/th/hist1.F:201-208`` -- header record 1 is
  ``EOR_C(84)``, the format code, 80 title bytes, ``EOR_C(84)``.
* ``engine/source/output/th/hist1.F:210-234`` with
  ``engine/source/system/timer_c.c:30-40`` -- header record 2 carries
  ``ctime()``: **the raw bytes of two runs of one deck differ inside those 24
  bytes and nowhere else**, so the stamp is located by content, reported, and
  never decoded as channel data.
* ``engine/source/output/th/hist1.F:237-290`` -- the ``TH_VERS>=50`` additional
  records: the record count, ``LTITL``, and the ``(FAC_MASS, FAC_LENGTH,
  FAC_TIME)`` unit-scaling triple -- the only scaling the T01 carries.
* ``engine/source/output/th/hist1.F:292-316`` -- the hierarchy record
  ``(NPART+NTHPART, NUMMAT, NUMGEO, NSUBS, NTHGRP2, NGLOBTH)`` with
  ``NGLOBTH=23``, immediately followed by the ``1..NGLOBTH`` curve-code record
  that marks the end of the fixed header.
* ``engine/source/output/th/hist1.F:357-376`` -- one part description record,
  ``4 + LTITL + 16`` bytes (``IPART(4,N)``, the title, ``IPART(7,N)``, the two
  node bounds, ``NVAR``), followed by that part's ``NVAR`` curve codes.
  ``:367-376`` skips the codes record when ``NVAR==0``.
* ``engine/source/output/th/hist2.F:302-303``, ``:307-333``, ``:338-477`` --
  the per-step records: ``TT``, the ``NGLOBTH`` global channels, the part
  values.
* ``starter/source/output/th/write_thnms1.F90:230-252`` -- the authoritative
  index / short-name / description table for those 23 global channels.
* ``starter/source/output/th/th_titles.F90:2759-2792`` (``varpa_title``) -- the
  per-part curve codes.
* ``engine/source/input/freform.F:1543`` (``TH_VERS=40`` unless a ``/TH/VERS``
  card says otherwise) and ``starter/source/output/th/hm_read_th.F:57``
  (``TH_VERS=MAX(TH_VERS,41)``) -- why a default T01 carries format code 3040,
  a 40-character title and **no** unit-scaling record, i.e. ``scaling == []``.

The three scoring decisions (they are decisions, so they are stated)
-------------------------------------------------------------------
1. **Channels are matched by NAME.**  Column order is irrelevant.  A channel
   present on one side and absent on the other is ``NODATA`` -- a fact about the
   two ``/TH`` requests, not a physics deviation -- and is excluded from
   :attr:`Score.worst`.
2. **``rel_rms = sqrt(mean((port-ref)^2)) / max(|ref|)``** per channel, over the
   samples both series cover.  The denominator is floored as the brief
   prescribes (``max(|ref|).max() * 1e-12``); because that floor is itself zero
   for an identically-zero reference channel, such a channel falls back to an
   **absolute** comparison (denominator 1.0) so a port channel that is exactly
   zero MATCHES and one that is not does not.
3. **``NODATA`` never drives ``worst``.**  With nothing comparable at all,
   :attr:`Score.worst` is itself a ``NODATA`` score.

Sample grid: the two runs never share one.  The comparison uses the **port's**
grid over the overlap window ``[0, min(ref, port) final time]``, which is what
``tools/validation_data/parity_m41.json`` records as the M36..M41 semantics and
what ``tools/validate_vs_fortran.py:1078-1081`` implements -- kept deliberately
so a verdict produced here means what a historical verdict meant.  Fewer than
:data:`MIN_SAMPLES` comparable samples is ``NODATA`` (the harness uses three,
this scorer two: the brief's own synthetic case scores two samples).

Known limitation (one record walk, shared)
------------------------------------------
``tools.oracle.oracle_selftest.parse_t01`` fixes the per-step record count at
four -- ``TT``, the global block, the part block, and one further block (the
per-TH-group curves, ``hist2.F:608-1403``).  A deck that also requests
``/TH/SUBSET`` curves gets a fifth record per step (``hist2.F:478-607``) and
that parser refuses the file with a loud error rather than mis-parsing it.
Widening it belongs in the shared walk so both consumers benefit; until then
this reader inherits the limitation instead of hiding it.

Deliberate deviation from the brief
-----------------------------------
``scaling`` is typed ``list[tuple[float, float, float]]``, not the brief's
``list[tuple[int, int, int]]``: the only scaling a T01 carries is the
``(FAC_MASS, FAC_LENGTH, FAC_TIME)`` triple, and upstream writes those as
**REAL** (``hist1.F:281-290``, via ``R4`` and ``write_r_c``), so storing them as
integers would be lossy (``FAC_MASS`` is 1e-3 for a Mg/mm deck).  Truncating a
unit factor to an integer would be a silent lie; the type says what the file
says.  The field is empty for every T01 a default deck produces (see the
``TH_VERS`` citations above).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The record framing is imported, not re-derived: one walk in the program.
from tools.oracle.oracle_selftest import (  # noqa: E402
    GLOBAL_CHANNELS,
    RUN_STAMP_LENGTH,
    T01FormatError,
    parse_t01,
    run_stamp_window,
    t01_records,
)

#: Relative-RMS below which a channel is called a match.  **Not** a free
#: parameter: it is the tolerance ``tools/validation_data/parity_m41.json``
#: records as ``tolerance_rel_rms`` (0.05), so a verdict produced here is
#: comparable with the M36..M41 parity tables.
MATCH_RMS = 0.05

#: Fewest comparable samples a channel needs before it is scored at all.  The
#: harness's own guard is three (``tools/validate_vs_fortran.py:1081-1082``); the
#: brief's own synthetic case scores a **two**-sample series, so two is the floor
#: here -- one sample is not a comparison, and every score reports its ``n``, so
#: a two-sample caller can see how thin the evidence is.
MIN_SAMPLES = 2

#: A channel whose reference peak is below this fraction of its group's dominant
#: reference peak carries no signal and is excluded from :attr:`Score.worst` --
#: the harness's own significance rule, and the reason its ``max_rel_rms``
#: (``tools/validate_vs_fortran.py:1461-1465``, which maximises over the
#: ``significant`` rows only) means what it says.  The transverse momentum of a
#: uniaxial test is round-off on both sides; left in, it would make every deck
#: DEVIATE at ``rel_rms ~ 0.5``.  Such channels are still reported in
#: :attr:`Score.per_channel` -- only the roll-up ignores them.
#:
#: The scale is per group, so a group whose *only* channel is noise has no
#: dominant peak to be measured against and stays significant.  That is the
#: harness's behaviour too, and it is the honest one: nothing but its own
#: magnitude says such a channel is noise.
SIGNIFICANCE_FRACTION = 1e-2

#: The relative denominator floor, as the brief prescribes.
RMS_DENOMINATOR_FLOOR = 1e-12

#: The verdict vocabulary.  ``MATCH`` / ``DEVIATION`` are the strings
#: ``tools/validation_data/parity_m41.json`` already carries per case, so they
#: are reused verbatim; ``NODATA`` is this scorer's name for "this channel was
#: not compared", which the historical tables expressed as a row that is absent
#: or as ``FORTRAN-FAIL`` with "no overlapping channels".
Verdict = Literal["MATCH", "DEVIATION", "NODATA"]
VERDICTS: Tuple[str, ...] = ("MATCH", "DEVIATION", "NODATA")

#: ``hist1.F:132-144`` -- format code -> title width (``LTITL``).  Codes 3050
#: and 4021 also carry the two additional records of ``hist1.F:237-290``.
FORMAT_CODE_TABLE: Dict[int, int] = {3040: 40, 3041: 80, 3050: 100, 4021: 100}

#: Width of header record 1: the format code plus 80 title bytes
#: (``hist1.F:201-208``, ``EOR_C(84)``).
HEADER_TITLE_RECORD = 84

#: The 23 global channels' names, in write order, taken from the authoritative
#: table (``write_thnms1.F90:230-252``) that
#: :data:`tools.oracle.oracle_selftest.GLOBAL_CHANNELS` transcribes.
GLOBAL_CHANNEL_NAMES: Tuple[str, ...] = tuple(n for _, n, _ in GLOBAL_CHANNELS)

#: Machine-readable citations: ``name -> (file, first line, last line, pattern)``.
#: ``tests/test_p0_compare_t01.py`` asserts every pattern is still present in
#: that exact line range, so a constant cannot drift away from its source.  A
#: pattern that has to span several lines starts with ``(?s)``: the window the
#: test builds is newline-joined source text, and ``.`` does not cross a
#: newline by default.
LAYOUT: Dict[str, Tuple[str, int, int, str]] = {
    "record_marker": (
        "common_source/tools/input_output/write_routines.c", 499, 510,
        r"integer_to_IEEE_ASCII\(\*len,octet\);\s*write_buffer\(octet,"
        r"sizeof\(unsigned char\),4\)"),
    "int32_header_write": (
        "common_source/tools/input_output/write_routines.c", 646, 663,
        r"integer_to_IEEE_ASCII\(w\[i\+k\],&buf\[i\*4\]\)"),
    "value_write": (
        "common_source/tools/input_output/write_routines.c", 520, 539,
        r"real_to_IEEE_ASCII\(w\[i\+k\],&buf\[i\*4\]\)"),
    "value_read": (
        "common_source/tools/input_output/write_routines.c", 757, 796,
        r"IEEE_ASCII_to_real\(&w\[i\+k\],&buf\[4\*i\]\)"),
    "value_encoding": (
        "engine/source/output/tools/ieee.cpp", 66, 124,
        r"mantisse = \(frexp\(\(double\)reel,&exposant\) - 0\.5\)\*1\.6777216E7"),
    "value_decoding": (
        "engine/source/output/tools/ieee.cpp", 127, 165,
        r"mantisse /= ldexp\(1\.,24\);\s*mantisse \+= 0\.5;"),
    "int32_byte_order": (
        "engine/source/output/tools/ieee.cpp", 40, 46,
        r"octet\[3\] = entier & 0xff;"),
    "value_record_width": (
        "engine/source/output/th/wrtdes.F", 121, 133,
        r"(?s)CALL EOR_C\(4\*L\).*R4 = A\(I\).*CALL EOR_C\(4\*L\)"),
    "format_code_table": (
        "engine/source/output/th/hist1.F", 132, 144,
        r"(?s)ICODE=3040.*LTITL = 40"),
    "header_record_1": (
        "engine/source/output/th/hist1.F", 201, 208,
        r"(?s)CALL EOR_C\(84\).*CALL WRITE_I_C\(ICODE,1\).*CALL WRITE_C_C"
        r"\(ITITLE,80\).*CALL EOR_C\(84\)"),
    "header_record_2_stamp": (
        "engine/source/output/th/hist1.F", 210, 234,
        r"(?s)CALL MY_CTIME\(ITITLE\).*CH80\(25:33\) =' RADIOSS '"),
    "additional_records": (
        "engine/source/output/th/hist1.F", 237, 290,
        r"FAC_MASS,FAC_LENGTH,FAC_TIME"),
    "hierarchy_record": (
        "engine/source/output/th/hist1.F", 292, 316,
        r"(?s)NGLOBTH=23.*IWA\(6\)= NGLOBTH"),
    "part_record": (
        "engine/source/output/th/hist1.F", 357, 376,
        r"(?s)CALL EOR_C\(20\+LTITL\).*CALL WRITE_I_C\(IPART\(4,N\),1\).*"
        r"CALL WRITE_C_C\(ITITLE,LTITL\).*CALL WRITE_I_C\(NVAR,1\).*"
        r"CALL EOR_C\(20\+LTITL\)"),
    "part_curve_codes": (
        "engine/source/output/th/hist1.F", 367, 376,
        r"IF\(NVAR/=0\)CALL WRTDES\(IWA,IWA,NVAR,ITTYP,0\)"),
    "time_record": (
        "engine/source/output/th/hist2.F", 302, 303,
        r"(?s)WA_LOCAL\(1\) = TT.*CALL WRTDES\(WA_LOCAL,WA_LOCAL,1,ITTYP,1\)"),
    "global_record": (
        "engine/source/output/th/hist2.F", 307, 333,
        r"CALL WRTDES\(WA,WA,NGLOBTH,ITTYP,1\)"),
    "part_values_record": (
        "engine/source/output/th/hist2.F", 338, 477,
        r"IF \(II/=0\) CALL WRTDES\(WA,WA,II,ITTYP,1\)"),
    "subset_record": (
        "engine/source/output/th/hist2.F", 478, 607,
        r"(?s)VARIABLES FOR EACH SUBSET.*IF\(II/=0\)CALL WRTDES\(WA,WA,II,ITTYP,1\)"),
    "global_channel_names": (
        "starter/source/output/th/write_thnms1.F90", 230, 252,
        r"(?s)1 IE\s+INTERNAL ENERGY.*23 DTE_INOUT"),
    "part_channel_titles": (
        "starter/source/output/th/th_titles.F90", 2759, 2792,
        r"(?s)varpa_title = \(/.*INTERNAL ENERGY.*KINETIC ENERGY"),
    "run_stamp_source": (
        "engine/source/system/timer_c.c", 30, 40, r"ctime"),
    "th_vers_default": (
        "engine/source/input/freform.F", 1543, 1549, r"TH_VERS=40"),
    "th_vers_floor": (
        "starter/source/output/th/hm_read_th.F", 48, 57,
        r"(?s)TH_VERS = 0.*TH_VERS=MAX\(TH_VERS,41\)"),
    "th_to_csv_is_external": (
        "tools/th_to_csv/README.md", 1, 7, r"OpenRadioss/Tools"),
}

#: Port CSV column -> upstream channel name.  The same pairs as
#: ``tools/validate_vs_fortran.py:1036-1046`` (``GLOBAL_MAP``) in the opposite
#: direction: that table maps a Fortran CSV column onto a port column, this one
#: renames a port column onto the upstream short name so the two series can be
#: matched **by name**.  A port column with no upstream counterpart (``EN``,
#: ``DE``, ``ERR%``) is deliberately left alone and scores ``NODATA``.
PORT_ALIASES: Dict[str, str] = {
    "MOMX": "XMOM",
    "MOMY": "YMOM",
    "MOMZ": "ZMOM",
    "EW": "EFW",
}

#: Port per-part variable -> the part curve code that means the same thing
#: (``varpa_title``, ``th_titles.F90:2759-2762``: 1 INTERNAL ENERGY,
#: 2 KINETIC ENERGY; the port computes exactly those two in
#: ``pyradioss/output/time_history.py:93-97``).  So ``P1_IE`` -- the port's name
#: -- becomes ``P1_1``, the name the binary reader derives from the T01 header.
PORT_PART_CODES: Dict[str, int] = {"IE": 1, "KE": 2}

#: The port's first CSV column (``pyradioss/output/time_history.py:49``).
PORT_TIME_COLUMN = "TIME"


def upstream_root() -> Optional[Path]:
    """The read-only OpenCourant source tree, or ``None``.

    Resolved per call and never at import time (importing this module must
    touch no filesystem).  ``$OR_SRC`` wins; otherwise a checkout that sits
    *beside* this repository is used, which is path-derived and therefore
    machine-neutral -- no absolute path is baked in anywhere in ``tools/``.
    ``None`` means the layout-conformance test skips with a reason; it never
    means the reader is unavailable, because the reader needs no sources.
    """
    probe = "engine/source/output/th/hist1.F"
    candidates = []
    if os.environ.get("OR_SRC"):
        candidates.append(Path(os.environ["OR_SRC"]))
    candidates.append(_REPO.parent / "OpenCourant")
    for root in candidates:
        if (root / probe).is_file():
            return root
    return None


# ---------------------------------------------------------------------------
# The decoded file
# ---------------------------------------------------------------------------

@dataclass
class T01:
    """One decoded time-history series.

    ``channels`` names the columns of ``values`` (column ``j`` of ``values`` is
    ``channels[j]``); ``times`` holds the ``n_times`` sample times;
    ``values`` is ``(n_times, n_channels)``; ``scaling`` holds the header's
    unit-scaling triples (see the module docstring on the element type).
    """

    channels: List[str]
    times: np.ndarray
    values: np.ndarray
    scaling: List[Tuple[float, float, float]] = field(default_factory=list)
    #: Additive header facts the scorer does not need but a reader of the file
    #: does: the ``ITTYP==3`` format code (``hist1.F:201-208``), the title width
    #: it implies (``hist1.F:132-144``), the record/header/step counts, and the
    #: located ``ctime`` run stamp (``hist1.F:210-234``, 24 bytes, never data).
    format_code: int = 0
    title_width: int = 0
    n_records: int = 0
    header_records: int = 0
    n_steps: int = 0
    run_stamp: Tuple[int, int] = (0, 0)

    def column(self, name: str) -> np.ndarray:
        """The ``name`` channel's series.

        Raises ``KeyError`` with the available names rather than returning a
        wrong column: a typo in a channel name is otherwise indistinguishable
        from a channel that is genuinely missing.
        """
        try:
            index = self.channels.index(name)
        except ValueError:
            raise KeyError(
                f"no channel {name!r}; this T01 carries {self.channels}") from None
        return self.values[:, index]


@dataclass(frozen=True)
class ChannelScore:
    """One channel's verdict.

    ``rel_rms`` is the relative RMS deviation, ``max_abs`` the largest absolute
    one, ``n`` the number of samples actually compared (0 for ``NODATA``, whose
    two float fields are ``inf`` so a stray arithmetic use is loud rather than
    quiet).
    """

    rel_rms: float
    max_abs: float
    n: int
    verdict: Verdict


@dataclass(frozen=True)
class Score:
    """``per_channel`` for every name either side mentioned, plus ``worst``."""

    per_channel: Dict[str, ChannelScore]
    worst: ChannelScore


_NODATA = ChannelScore(rel_rms=math.inf, max_abs=math.inf, n=0,
                       verdict="NODATA")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_radioss_float(payload: bytes) -> np.ndarray:
    """Decode a record of Radioss IEEE reals (``IEEE_ASCII_to_real``).

    A vectorised transcription of ``engine/source/output/tools/ieee.cpp:127-165``
    -- the routine ``read_r_c`` (``write_routines.c:757-796``) calls to read
    back what ``write_r_c`` wrote: sign from bit 7 of byte 0, an 8-bit exponent
    split across ``(byte0 & 0x7f) << 1 | byte1 >> 7`` biased by 126, and a
    **24**-bit mantissa ``((byte1 & 0x7f) << 16 | byte2 << 8 | byte3)`` scaled by
    ``2**-24`` and offset by ``0.5``.  An all-zero exponent is a signed zero
    (``ieee.cpp:146-150``).

    Not a stock ``>f4``: the 24th mantissa bit is one a binary32 read would
    drop.  For every value the engine writes the two agree bit for bit, because
    ``wrtdes.F:121-133`` converts through ``R4`` first and a binary32 value has
    an all-zero 24th mantissa bit -- and the shared record walk relies on that
    (it reads the channel values as ``>f4``), which
    ``test_read_t01_sample_count_and_maxima_match_the_golden_record`` asserts
    over all 100 samples of the golden file rather than taking on trust.  This
    function is the upstream-exact decoder, and it is what decodes the header's
    ``FAC`` record; callers that need upstream semantics rather than the
    equivalence should prefer it.
    """
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size % 4:
        raise T01FormatError(
            f"a value record is {raw.size} bytes, not a multiple of 4 "
            "(wrtdes.F:121-133 writes EOR_C(4*L) around L single-precision "
            "values)")
    raw = raw.reshape(-1, 4).astype(np.int32)
    sign = np.where((raw[:, 0] & 0x80) == 0, 1.0, -1.0)
    exponent = ((raw[:, 0] & 0x7F) << 1) + ((raw[:, 1] & 0x80) >> 7)
    mantissa = raw[:, 1] & 0x7F
    mantissa = mantissa * 256 + raw[:, 2]
    mantissa = mantissa * 256 + raw[:, 3]
    out = np.zeros(raw.shape[0], dtype=np.float64)
    nonzero = exponent != 0
    out[nonzero] = (sign[nonzero]
                    * (mantissa[nonzero] / 2.0 ** 24 + 0.5)
                    * 2.0 ** (exponent[nonzero].astype(np.float64) - 126.0))
    return out


def _as_ints(payload: bytes) -> List[int]:
    if len(payload) % 4:
        raise T01FormatError(f"payload of {len(payload)} bytes is not int32")
    return [int(v) for v in np.frombuffer(payload, dtype=">i4")]


def _header_format(records: Sequence[Tuple[int, bytes]]) -> Tuple[int, int]:
    """``(format_code, title_width)`` from header record 1.

    ``hist1.F:201-208`` writes the format code as one int32 followed by 80 title
    bytes; ``hist1.F:132-144`` maps that code to the title width, and codes 3050
    / 4021 to the presence of the additional records of ``:237-290``.  Reading
    the width from the code -- instead of guessing it from a record length --
    is what lets the part records below be located exactly.
    """
    if not records or len(records[0][1]) != HEADER_TITLE_RECORD:
        raise T01FormatError(
            f"header record 1 is {len(records[0][1]) if records else 0} bytes, "
            f"not the {HEADER_TITLE_RECORD} of hist1.F:201-208 "
            "(EOR_C(84), the format code, 80 title bytes)")
    code = _as_ints(records[0][1][:4])[0]
    if code not in FORMAT_CODE_TABLE:
        raise T01FormatError(
            f"unknown T01 format code {code}; hist1.F:132-144 writes "
            f"{sorted(FORMAT_CODE_TABLE)} for TH_VERS below 47, between 47 and "
            "49, from 50 and from 2021")
    return code, FORMAT_CODE_TABLE[code]


def _hierarchy_and_nglobth(records: Sequence[Tuple[int, bytes]]
                            ) -> Tuple[Tuple[int, ...], int, int]:
    """``(hierarchy, nglo, index_of_the_curve_code_record)``.

    The curve-code record is found **by content**, the way
    ``tools.oracle.oracle_selftest`` finds it: a run of big-endian int32 equal
    to ``1..N`` with ``N >= 8``.  ``hist1.F:308-316`` writes the hierarchy record
    (6 int32, ``NGLOBTH`` sixth) immediately before it, with nothing in between,
    so the hierarchy is the record directly ahead of it.  ``NGLOBTH`` therefore
    comes from the file, never assumed to be 23.
    """
    for index, (_, payload) in enumerate(records):
        if len(payload) < 8 * 4 or len(payload) % 4:
            continue
        values = _as_ints(payload)
        if values == list(range(1, len(values) + 1)):
            if index == 0:
                break
            hierarchy = _as_ints(records[index - 1][1][:24])
            if len(hierarchy) != 6:
                continue
            return tuple(hierarchy), len(values), index
    raise T01FormatError(
        "no 1..N curve-code record found; hist1.F:311-316 writes one right after "
        "the hierarchy record of :292-308 when the T01 is the global history "
        "file, and :300-316 fixes NGLOBTH=23")


def _scaling_records(records: Sequence[Tuple[int, bytes]],
                     format_code: int) -> List[Tuple[float, float, float]]:
    """The ``(FAC_MASS, FAC_LENGTH, FAC_TIME)`` triples, if the file has any.

    ``hist1.F:237-290`` writes the additional records only for ``TH_VERS>=50``
    (format codes 3050 / 4021): the number of additional records, ``LTITL``,
    and then three single-precision values.  They sit between the run-stamp
    record and the hierarchy record -- in that order, after header record 2 --
    so they are read positionally, and a file without them (``TH_VERS``
    defaults to 40 / 41) simply has none, which is why ``scaling`` is empty for
    every T01 a default deck produces.
    """
    if format_code not in (3050, 4021):
        return []
    if len(records) < 5:
        raise T01FormatError(
            f"format code {format_code} promises the additional records of "
            "hist1.F:237-290 but the file has too few records")
    count = _as_ints(records[2][1][:4])[0]
    if count != 2:
        raise T01FormatError(
            f"the additional-record count is {count}, not the NRECORD=2 of "
            "hist1.F:240")
    factors = decode_radioss_float(records[4][1])
    if factors.size != 3:
        raise T01FormatError(
            f"the unit-scaling record holds {factors.size} values, not the "
            "FAC_MASS, FAC_LENGTH, FAC_TIME of hist1.F:281-290")
    return [(float(factors[0]), float(factors[1]), float(factors[2]))]


def _part_channels(records: Sequence[Tuple[int, bytes]], after: int,
                   title_width: int, n_parts: int) -> List[List[int]]:
    """Each part's curve codes, in file order (``hist1.F:318-377``).

    A part description is ``4 + LTITL + 16`` bytes -- ``IPART(4,N)``, the
    ``LTITL``-character title, ``IPART(7,N)``, the two node bounds and ``NVAR``
    (``hist1.F:357-366``) -- and is followed by its ``NVAR`` curve codes, which
    ``:376`` skips when ``NVAR==0``.  Parts are written before subsets
    (``:442-514``) and TH groups (``:516-600``), so exactly ``n_parts`` of these
    records start the variable part of the header.
    """
    width = 4 + title_width + 16
    codes: List[List[int]] = []
    index = after
    for _ in range(n_parts):
        if index >= len(records) or len(records[index][1]) != width:
            raise T01FormatError(
                f"expected a {width}-byte part description record "
                "(hist1.F:358-366: 4 + LTITL + 16) at record {index}, found "
                f"{len(records[index][1]) if index < len(records) else 'nothing'}"
                " bytes")
        payload = records[index][1]
        title = payload[4:4 + title_width]
        if not (title.isascii() and all(32 <= c < 127 for c in title)):
            raise T01FormatError(
                f"the part title at record {index} is not printable text, so "
                "this is not a part description record")
        nvar = _as_ints(payload[4 + title_width:])[3]
        index += 1
        if nvar <= 0:
            codes.append([])
            continue
        if index >= len(records) or len(records[index][1]) != 4 * nvar:
            raise T01FormatError(
                f"a part with NVAR={nvar} must be followed by a {4 * nvar}-byte "
                f"curve-code record (hist1.F:367-376); record {index} has "
                f"{len(records[index][1]) if index < len(records) else 'nothing'}"
                " bytes")
        codes.append(_as_ints(records[index][1]))
        index += 1
    return codes


def read_t01(path) -> T01:
    """Decode a binary ``ITTYP==3`` T01 into a :class:`T01`.

    The channel list is the 23 global channels in write order
    (``hist2.F:307-333``, named from ``write_thnms1.F90:230-252``) followed by
    one channel per part curve code, named ``P<position>_<code>`` -- the T01
    holds no part id (``hist1.F:357-366`` writes the part's element count, its
    title, ``IPART(7,N)``, the node bounds and ``NVAR``), so the position in the
    file is the only identity available and the curve code is the only variable
    identity.  Subset, TH-group and per-node curve records are part of the
    per-step block and are walked (the shared walk validates their stride) but
    are **not** named: upstream ships no name table for them without the
    object-type dispatch ``hist2.F:608-1403`` performs, and no channel name is
    invented here.

    ``path`` in, :class:`T01` out; anything that is not an ``ITTYP==3`` T01 raises
    :class:`T01FormatError` (re-exported from
    :mod:`tools.oracle.oracle_selftest`) rather than returning partial data.
    """
    blob = Path(path).read_bytes()
    records = t01_records(blob)
    if len(records) < 8:
        raise T01FormatError(
            f"only {len(records)} records; not a T01 (hist1.F:201-208 alone "
            "writes two, plus the hierarchy and the curve-code record)")
    format_code, title_width = _header_format(records)
    stamp_offset, stamp_length = run_stamp_window(blob)
    if stamp_length != RUN_STAMP_LENGTH:
        raise T01FormatError(
            f"the run stamp is {stamp_length} bytes, not the "
            f"{RUN_STAMP_LENGTH} of timer_c.c:36-37 that hist1.F:212-214 "
            "copies")
    scaling = _scaling_records(records, format_code)
    hierarchy, nglo, code_index = _hierarchy_and_nglobth(records)
    if nglo != len(GLOBAL_CHANNEL_NAMES):
        raise T01FormatError(
            f"the file carries {nglo} global channels but "
            f"{len(GLOBAL_CHANNEL_NAMES)} names are transcribed from "
            "write_thnms1.F90:230-252; no name is invented for the extra ones")
    part_codes = _part_channels(records, code_index + 1, title_width,
                                hierarchy[0])

    # The data section: time, the global block, the part block, one more block.
    # Framing and stride validation are the shared walk's job (see the module
    # docstring's "Known limitation").
    parsed = parse_t01(blob)
    globals_ = parsed["global"]
    parts = parsed["part"]
    n_expected = sum(len(codes) for codes in part_codes)
    if parts.shape[1] != n_expected:
        raise T01FormatError(
            f"the header declares {n_expected} part channels "
            "(hist1.F:367-376) but the data section carries {parts.shape[1]}")

    channels = list(GLOBAL_CHANNEL_NAMES)
    columns = [globals_[:, j] for j in range(globals_.shape[1])]
    for position, codes in enumerate(part_codes, start=1):
        for offset, code in enumerate(codes):
            channels.append(f"P{position}_{int(code)}")
            columns.append(parts[:, offset])
    values = (np.stack(columns, axis=1).astype(np.float64)
              if columns else np.zeros((parsed["n_steps"], 0)))

    times = parsed["time"].astype(np.float64)
    if np.any(np.diff(times) < 0):
        raise T01FormatError(
            "the sample times are not non-decreasing; the record walk found "
            "the data section at the wrong place")

    return T01(channels=channels, times=times, values=values, scaling=scaling,
               format_code=format_code, title_width=title_width,
               n_records=len(records),
               header_records=parsed["header_records"],
               n_steps=parsed["n_steps"],
               run_stamp=(stamp_offset, stamp_length))


# ---------------------------------------------------------------------------
# The port's ASCII T01
# ---------------------------------------------------------------------------

def read_port_csv(path) -> T01:
    """Read the port's T01 CSV into the same :class:`T01` shape.

    The port writes its time history as ASCII (``pyradioss/output/
    time_history.py:77-78`` writes a commented banner and a comma-separated
    header; ``:332`` writes one row per sample), so this is the *second
    producer* the reader is validated against: two independent producers of the
    same physics, one binary and one ASCII.

    Port columns are renamed onto the upstream short names of :data:`GLOBAL_
    CHANNEL_NAMES` through :data:`PORT_ALIASES` and :data:`PORT_PART_CODES`, so
    :func:`score` can match the two series **by name**.  A port column with no
    upstream counterpart (``EN``, ``DE``, ``ERR%``) keeps its own name and
    therefore scores ``NODATA`` -- which is the correct verdict, not a gap.
    """
    header: Optional[List[str]] = None
    rows: List[List[float]] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if header is None:
            header = cells
            continue
        try:
            rows.append([float(cell) for cell in cells])
        except ValueError:
            continue
    if header is None or not rows:
        raise T01FormatError(
            f"{path}: no header and no sample rows; the port writes "
            "'# pyradioss time history' then TIME,... (time_history.py:77-78)")
    width = min(len(header), min(len(row) for row in rows))
    header, rows = header[:width], [row[:width] for row in rows]
    if PORT_TIME_COLUMN not in header:
        raise T01FormatError(
            f"{path}: no {PORT_TIME_COLUMN} column; the port's first column is "
            f"{header[0]!r} (time_history.py:49)")
    times = np.array([row[header.index(PORT_TIME_COLUMN)] for row in rows],
                     dtype=np.float64)
    if np.any(np.diff(times) < 0):
        raise T01FormatError(f"{path}: the sample times are not non-decreasing")

    channels: List[str] = []
    columns: List[np.ndarray] = []
    for index, name in enumerate(header):
        if index == header.index(PORT_TIME_COLUMN):
            continue
        channels.append(_port_channel_name(name))
        columns.append(np.array([row[index] for row in rows], dtype=np.float64))
    values = (np.stack(columns, axis=1) if columns
              else np.zeros((len(rows), 0)))
    return T01(channels=channels, times=times, values=values, scaling=[])


def _port_channel_name(name: str) -> str:
    """The upstream name for a port CSV column, or the port's own if it has none.

    ``P<id>_<VAR>`` part columns are folded onto ``P<id>_<code>`` -- the name the
    binary reader derives from the T01's own curve codes -- using
    :data:`PORT_PART_CODES`.  Everything else goes through :data:`PORT_ALIASES`
    or is returned unchanged.
    """
    if "_" in name and name[0] in "PSN":
        head, _, variable = name.partition("_")
        if variable in PORT_PART_CODES:
            return f"{head}_{PORT_PART_CODES[variable]}"
    return PORT_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _channel_score(reference: np.ndarray, port: np.ndarray) -> ChannelScore:
    """``rel_rms = sqrt(mean((port-ref)^2)) / max(|ref|)`` and its verdict.

    The denominator is floored at ``max(|ref|).max() * 1e-12`` as the brief
    prescribes; for an identically-zero reference channel that floor is zero
    too, so the comparison falls back to absolute units (denominator 1.0) --
    which keeps the two distinguishable outcomes apart instead of dividing by
    zero or calling every deviation a MATCH.
    """
    deviation = port - reference
    scale = float(np.max(np.abs(reference)))
    denominator = max(scale, scale * RMS_DENOMINATOR_FLOOR)
    if denominator == 0.0:
        denominator = 1.0
    rel_rms = float(np.sqrt(np.mean(deviation ** 2)) / denominator)
    max_abs = float(np.max(np.abs(deviation)))
    verdict = "MATCH" if rel_rms <= MATCH_RMS else "DEVIATION"
    return ChannelScore(rel_rms=rel_rms, max_abs=max_abs, n=reference.size,
                        verdict=verdict)


def _group_of(name: str) -> str:
    """Which scale a channel is judged against: momentum, mass, or energy.

    The same grouping as ``tools/validate_vs_fortran.py:1115-1120``, which needs
    it because the transverse momentum of an axially loaded model is numerical
    noise against its axial momentum.  Names here are the upstream short names
    (``XMOM``, ``MASS``, ``IE``, ``P<k>_<code>``), so the prefix test is on
    ``MOM`` either way round.
    """
    if "MOM" in name:
        return "momentum"
    if name == "MASS":
        return "mass"
    return "energy"


def score(ref: T01, port: T01) -> Score:
    """Compare two series channel by channel, matched **by name**.

    The two series are compared on the **port's** grid over the overlap window
    ``[0, min(ref, port) final time]`` -- the M36..M41 semantics recorded in
    ``tools/validation_data/parity_m41.json`` and implemented by
    ``tools/validate_vs_fortran.py:1078-1081``.  A channel only one side has is
    ``NODATA``; a channel with fewer than :data:`MIN_SAMPLES` comparable samples
    is ``NODATA`` too.

    :attr:`Score.worst` is the largest ``rel_rms`` among the channels that were
    compared **and** carry signal (:data:`SIGNIFICANCE_FRACTION` of their
    group's dominant reference peak -- the harness's own rule, so that
    ``worst.rel_rms`` is the number ``parity_m41.json``'s ``max_rel_rms`` is).
    With nothing comparable it is itself a ``NODATA`` score.  Every channel,
    significant or not, stays in :attr:`Score.per_channel`.
    """
    if ref.times.ndim != 1 or port.times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if ref.values.shape[0] != ref.times.size:
        raise ValueError(
            f"ref.values has {ref.values.shape[0]} rows for "
            f"{ref.times.size} times")
    if port.values.shape[0] != port.times.size:
        raise ValueError(
            f"port.values has {port.values.shape[0]} rows for "
            f"{port.times.size} times")

    last = min(float(ref.times[-1]), float(port.times[-1]))
    grid = port.times[port.times <= last + 1e-12]
    comparable = grid.size >= MIN_SAMPLES

    per_channel: Dict[str, ChannelScore] = {}
    significant: List[ChannelScore] = []
    peaks: Dict[str, float] = {}
    for name in sorted(set(ref.channels) | set(port.channels)):
        if name not in ref.channels or name not in port.channels or not comparable:
            per_channel[name] = _NODATA
            continue
        reference = np.interp(grid, ref.times, ref.column(name))
        measured = np.interp(grid, port.times, port.column(name))
        per_channel[name] = _channel_score(reference, measured)
        peaks[name] = float(np.max(np.abs(reference)))

    dominant: Dict[str, float] = {}
    for name, peak in peaks.items():
        group = _group_of(name)
        dominant[group] = max(dominant.get(group, 0.0), peak)
    for name, peak in peaks.items():
        if peak >= SIGNIFICANCE_FRACTION * dominant[_group_of(name)]:
            significant.append(per_channel[name])

    if not significant:
        return Score(per_channel=per_channel, worst=_NODATA)
    return Score(per_channel=per_channel,
                 worst=max(significant, key=lambda c: c.rel_rms))