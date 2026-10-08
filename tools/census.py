#!/usr/bin/env python3
"""Task P1.3 — the machine-readable Fortran<->Python coverage census.

WHAT THIS IS
------------
Upstream OpenRadioss writes a provenance block into the header of every source
file: the routine's name, its own path, and three call-graph sections.  This
module reads those blocks out of every Fortran file under ``$OR_SRC`` and joins
them against what this port says about itself, so "what is covered?" becomes a
queryable table instead of prose.

    python tools/census.py --render plan/CENSUS.md --json tools/validation_data/census.json

The three claims the census makes per upstream file:

``calls`` / ``called_by`` / ``uses``
    Transcribed from upstream's own header.  Not re-derived from the Fortran
    bodies: a second, independently-computed call graph would be free to
    disagree with the first, and the disagreement would be undetectable.

``python_module``
    Found by scanning **this port's own docstrings** for the upstream path.  It
    answers "where does the port claim to implement this file", and it is
    populated automatically — there is no hand-maintained mapping table,
    because a hand-maintained one is a record that ages like every other
    record.  Task P1.2 (normalise every citation to the ``$OR_SRC`` form) is
    what makes this join possible.

``status``
    Read from ``tools/validation_data/port_status.json`` and **nothing else**.
    An upstream file absent from that allowlist is ``missing``, never
    ``ported``.  The allowlist ships empty; populating it is a later phase's
    job, and it is the only thing in this program permitted to promote a row.

The empty allowlist is deliberate rather than a stub.  ``ported`` is a claim
about physics, and a false one is invisible: nothing raises, the table simply
lies.  Starting from "nothing is verified" and being promoted is the only
direction in which the table cannot mislead a phase reviewer, so that is where
it starts.

PROVENANCE BLOCK LAYOUT
-----------------------
Read off ``$OR_SRC/engine/source/elements/solid/solide/sforc3.F:24-60`` — the
reference named by ``plan/02_phase1_foundation.md:206-209`` — not invented::

    !||====================================================================
    !||    sforc3                 ../engine/source/elements/solid/solide/sforc3.F
    !||--- called by ------------------------------------------------------
    !||    alemain                ../engine/source/ale/alemain.F
    !||    forint                 ../engine/source/elements/forint.F
    !||--- calls      -----------------------------------------------------
    !||    aleflow                ../engine/source/ale/porous/aleflow.F
    !||--- uses       -----------------------------------------------------
    !||    ale_connectivity_mod   ../common_source/modules/ale/ale_connectivity_mod.F
    !||====================================================================

A ``!||====`` fence, a self line, then any number of ``--- <section> ---``
markers each followed by ``name<spaces>path`` rows, closed by a second fence.
Sections are optional — ``engine/share/resol/initbuf.F`` has ``called by`` and
no ``calls`` — so an absent section parses as empty rather than as an error.

THE `../` PREFIX IS NOT PART OF THE KEY
---------------------------------------
Rows inside the block carry ``../engine/...``: paths relative to the *file's own
directory*, which is what upstream's generator emits and what makes the block
relocatable.  They are normalised here to tree-root-relative
(``engine/source/...``) because that is the form the port's docstrings cite,
and joining two different spellings of one path would silently map nothing.

READ-ONLY UPSTREAM
------------------
``$OR_SRC`` is upstream AGPL source that this port only ever reads.  This
module opens it for reading and never writes, builds, or fetches into it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pyradioss.paths import or_src  # noqa: E402

# --- vocabulary, all measured off the upstream tree ------------------------

#: Upstream's Fortran suffixes.  The tree also carries C, C++ and headers;
#: including them would silently change what the file count means whenever
#: upstream adds a C file, so the set is closed and pinned by a test.
FORTRAN_SUFFIXES = frozenset({".F", ".F90", ".f", ".f90", ".for"})

#: The three section markers upstream emits.  Surveyed over the whole tree:
#: 9,399 ``called by``, 6,170 ``uses``, 6,038 ``calls``, and no fourth kind.
SECTION_CALLED_BY = "called by"
SECTION_CALLS = "calls"
SECTION_USES = "uses"

_STATUSES = ("ported", "partial", "stub", "missing")
Status = Literal["ported", "partial", "stub", "missing"]

#: Where ``main`` puts things by default; the plan's commands use these paths.
CENSUS_JSON = _REPO_ROOT / "tools" / "validation_data" / "census.json"
CENSUS_MD = _REPO_ROOT / "plan" / "CENSUS.md"
PORT_STATUS = _REPO_ROOT / "tools" / "validation_data" / "port_status.json"

#: ``!||====...`` fence.
_FENCE = "!||===="

from pyradioss.paths import or_src as _resolve_or_src  # noqa: E402

#: ``!||--- called by -------``.  The dashes are cosmetic padding, so match the
#: label between the marker and the run of dashes rather than its column.
_SECTION_RE = re.compile(r"^!\|\|---\s*(?P<name>[a-zA-Z ]+?)\s*-+\s*$")

#: ``!||    sforc3    ../engine/...``.  The name is the first run of
#: non-spaces; the path is the last whitespace-separated token, and it is
#: optional because a header may name a routine it gives no path for.
_ROW_RE = re.compile(r"^!\|\|\s+(?P<name>\S+)(?:\s+(?P<path>\S+))?\s*$")

#: A docstring citation: ``$OR_SRC/engine/source/.../sforc3.F`` with an
#: optional ``:24-60`` line range, which is stripped before joining.
_CITATION_RE = re.compile(
    r"\$OR_SRC/(?P<path>[A-Za-z0-9_./-]+\.(?:F90|F|f90|f|for))(?::[0-9,\-]+)?"
)


@dataclass(frozen=True)
class Provenance:
    """One parsed ``!||`` block.

    ``routine`` is upstream's name for the file (which is not always the file's
    stem — ``s8dlenmax_sm_mod`` lives in ``s8dlenmax_sm.F90``), so it is kept
    separate from the key.  ``self_path`` is normalised to tree-root-relative
    like every other path in the record.
    """

    routine: str
    self_path: str | None
    called_by: tuple[str, ...]
    calls: tuple[str, ...]
    uses: tuple[str, ...]


@dataclass(frozen=True)
class FileRecord:
    """One upstream Fortran file, as the census sees it."""

    path: str
    """Tree-root-relative, e.g. ``engine/source/engine/resol.F``."""

    lines: int
    calls: tuple[str, ...]
    called_by: tuple[str, ...]
    area: str
    """The top-level subsystem: the file's parent directory, so
    ``--area engine/source/elements/solid`` is a prefix match."""

    python_module: str | None
    status: Status

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "lines": self.lines,
            "calls": list(self.calls),
            "called_by": list(self.called_by),
            "area": self.area,
            "python_module": self.python_module,
            "status": self.status,
        }


def _is_under(path: str, prefix: str) -> bool:
    """Whole-path-component containment, so ``solid`` never matches ``solid_2d``."""
    return path == prefix or path.startswith(prefix + "/")


@dataclass
class Census:
    """The whole scan."""

    files: dict[str, FileRecord] = field(default_factory=dict)
    or_src: str = ""

    def counts(self) -> dict[str, int]:
        """File count per status, every key present so a zero is visible."""
        tally = dict.fromkeys(_STATUSES, 0)
        for rec in self.files.values():
            tally[rec.status] += 1
        return tally


    def areas(self) -> dict[str, dict[str, int]]:
        """Per-area counts, recursively collapsed so ``--area`` can filter."""
        out: dict[str, dict[str, int]] = {}
        for rec in self.files.values():
            for prefix in _area_prefixes(rec.area):
                bucket = out.setdefault(prefix, dict.fromkeys(_STATUSES, 0))
                bucket[rec.status] += 1
        return out

    def select(self, area: str | None = None, status: str | None = None) -> "Census":
        """A view narrowed by subsystem and/or status.

        ``area`` is matched as a path prefix against both the file path and its
        area, so ``--area engine`` and ``--area engine/source/elements`` both
        work and neither requires the caller to know the depth.

        Matching is on whole path components, not on raw string prefixes.  The
        raw form leaked: ``--area engine/source/elements/solid`` also matched
        ``solid_2d``, because ``"…/solid_2d/quad".startswith("…/solid")`` is
        true.  That is 46 rows from a different phase's scope appearing in a
        gate that counts rows to decide whether a phase is done.
        """
        norm = area.rstrip("/") if area is not None else None
        out: dict[str, FileRecord] = {}
        for path, rec in self.files.items():
            if norm is not None and not (
                _is_under(path, norm) or _is_under(rec.area, norm)
            ):
                continue
            if status is not None and rec.status != status:
                continue
            out[path] = rec
        return Census(files=out, or_src=self.or_src)


def _area_prefixes(area: str) -> list[str]:
    """Every ancestor of ``area``, longest first, down to the empty root."""
    if not area:
        return [""]
    parts = area.split("/")
    return ["/".join(parts[:i]) for i in range(len(parts), 0, -1)] + [""]


def _normalise(raw: str) -> str:
    """``../engine/source/x.F`` -> ``engine/source/x.F``.

    Upstream writes paths relative to the file's own directory (the block stays
    correct when the tree moves).  A naive copy would key the census by a path
    that resolves nowhere and match no docstring citation.
    """
    path = raw.strip().replace("\\", "/")
    while path.startswith("../"):
        path = path[3:]
    return path.lstrip("./") if path.startswith("./") else path


def parse_provenance(text: str) -> Provenance | None:
    """Parse a ``!||`` block, or return ``None`` if the file has none.

    ``None`` rather than an empty parse, so "upstream did not annotate this"
    stays distinguishable from "upstream annotated it with nothing".
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(_FENCE)), None)
    if start is None:
        return None

    routine: str | None = None
    self_path: str | None = None
    buckets: dict[str, list[str]] = {
        SECTION_CALLED_BY: [],
        SECTION_CALLS: [],
        SECTION_USES: [],
    }
    section: str | None = None

    for line in lines[start + 1:]:
        if line.startswith(_FENCE):
            break
        if not line.startswith("!||"):
            continue

        section_match = _SECTION_RE.match(line.rstrip())
        if section_match:
            label = section_match.group("name").strip()
            section = label if label in buckets else None
            continue

        row = _ROW_RE.match(line.rstrip())
        if row is None:
            continue

        name = row.group("name")
        if routine is None and self_path is None:
            # First row is the self line: routine, then its own path.
            routine = name
            raw = row.group("path")
            if raw:
                self_path = _normalise(raw)
            continue
        if section is not None:
            buckets[section].append(name)

    if routine is None:
        return None
    return Provenance(
        routine=routine,
        self_path=self_path,
        called_by=tuple(buckets[SECTION_CALLED_BY]),
        calls=tuple(buckets[SECTION_CALLS]),
        uses=tuple(buckets[SECTION_USES]),
    )


def or_src() -> Path:
    """Resolve the upstream tree, read-only.

    Routed through :func:`pyradioss.paths.or_src` so this tool honours the same
    resolution order as the rest of the port rather than inventing one.
    """
    return Path(_resolve_or_src())


def _newest_mtime(root: Path) -> float:
    """Newest mtime under ``root`` — used to prove the scan wrote nothing."""
    newest = 0.0
    for path in root.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def build_citation_index(root: Path | None = None) -> dict[str, str]:
    """Map upstream path -> the port module whose docstrings cite it.

    Scanned from the port's own source, never hand-maintained.  When two
    modules cite the same upstream file the lexicographically first wins, so
    the result is deterministic across runs and across filesystems.

    Scoped to the ``pyradioss`` package on purpose.  An unrestricted scan of
    the repository also reads ``tools/census.py`` and this task's own test file,
    both of which quote upstream paths in prose — and the lexicographic
    tie-break then hands ``sforc3.F`` to ``tests.test_p0_census``, so the census
    would claim the tool that reads the census implements the Fortran it reads.
    A ``python_module`` has to name something that could actually be the
    implementation, which is exactly what a test and a reader are not.
    """
    root = root or (_REPO_ROOT / "pyradioss")
    index: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(_REPO_ROOT).with_suffix("")
        module = ".".join(rel.parts)
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        for match in _CITATION_RE.finditer(text):
            cited = _normalise(match.group("path"))
            index.setdefault(cited, module)
    return index


def load_port_status(path: Path | None = None) -> dict[str, str]:
    """Read the status allowlist.

    Returns ``{}`` when the file is absent.  Absent is treated as empty rather
    than as an error, because "no allowlist" and "an empty allowlist" mean the
    same thing here: nothing is promoted, so everything is ``missing``.
    """
    path = path or PORT_STATUS
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object of path -> status")
    bad = {k: v for k, v in data.items() if v not in _STATUSES}
    if bad:
        raise ValueError(f"{path} has unknown statuses: {sorted(bad)}")
    return data


def scan(root: Path | None = None) -> Census:
    """Walk the upstream tree and build the census.

    Every Fortran file is recorded, annotated or not: a file with no ``!||``
    block still has a line count and a status, and both matter — "unannotated"
    and "annotated but unported" are different facts.
    """
    root = Path(root) if root is not None else or_src()
    allowlist = load_port_status()
    citations = build_citation_index()

    files: dict[str, FileRecord] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix not in FORTRAN_SUFFIXES or not path.is_file():
            continue
        rel = _normalise(str(path.relative_to(root)))
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        block = parse_provenance(text)
        lines = len(text.splitlines())
        called_by = block.called_by if block else ()
        calls = block.calls if block else ()
        area = str(Path(rel).parent)

        files[rel] = FileRecord(
            path=rel,
            lines=lines,
            calls=calls,
            called_by=called_by,
            area=area,
            python_module=citations.get(rel),
            status=allowlist.get(rel, "missing"),
        )

    return Census(files=files, or_src=str(root))


def render(census: Census, area: str | None = None, status: str | None = None) -> str:
    """Render the markdown table a phase reviewer reads.

    A summary block comes first — per-area counts and the total — because the
    failure this task exists to prevent is a reviewer reading a short table as
    "all done".  The summary always states the unfiltered total next to the
    filtered one, so a narrowed view can never be mistaken for the whole tree.
    """
    total = len(census.files)
    view = census.select(area=area, status=status)
    tally = view.counts()
    # Name the tree through the project's own indirection, never as an absolute
    # path: CENSUS.md is a committed record, and a record that spells out
    # /home/<user>/... is false on every other machine.  tests/
    # test_p0_no_stale_machine_paths.py fails this file otherwise.
    src = "$OR_SRC"

    out: list[str] = []
    out.append("# Fortran<->Python coverage census")
    out.append("")
    out.append(f"Generated by `tools/census.py` from `{src}` — do not hand-edit.")
    out.append("")
    out.append(
        f"**{len(view.files)} of {total} upstream Fortran files shown**"
        + (f", filtered to `status={status}`." if status else ".")
    )
    if area:
        out.append(f"Area filter: `{area}`.")
    out.append("")
    out.append("`status` comes only from `tools/validation_data/port_status.json`.")
    out.append(
        "A file absent from that allowlist is `missing`, never `ported` — the "
        "allowlist ships empty, so no row claims to be ported yet."
    )
    out.append("")

    if status:
        out.append("## Rows")
        out.append("")
        out.extend(_table(view, limited=None))
        out.append("")
        return "\n".join(out) + "\n"

    out.append("## Per-area summary")
    out.append("")
    out.append("| area | total | " + " | ".join(_STATUSES) + " |")
    out.append("|---|---|" + "---|" * len(_STATUSES))
    for name in sorted(a for a in census.areas() if a):
        counts = census.areas()[name]
        row_total = sum(counts.values())
        out.append(
            f"| `{name}` | {row_total} | "
            + " | ".join(str(counts[s]) for s in _STATUSES)
            + " |"
        )
    out.append("")
    out.append("## Files")
    out.append("")
    out.extend(_table(view, limited=400))
    return "\n".join(out) + "\n"


def _table(view: Census, limited: int | None) -> list[str]:
    """The per-file markdown table, sorted by path for a stable diff."""
    rows = sorted(view.files.items())
    truncated = 0
    if limited is not None and len(rows) > limited:
        truncated = len(rows) - limited
        rows = rows[:limited]

    out = ["| path | lines | area | python_module | status |", "|---|---|---|---|---|"]
    for path, rec in rows:
        module = rec.python_module or "—"
        out.append(f"| `{path}` | {rec.lines} | `{rec.area}` | `{module}` | {rec.status} |")
    if truncated:
        out.append("")
        out.append(
            f"_{truncated} further rows omitted; the JSON artifact carries every "
            "record in full._"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="census.py",
        description="Scan upstream Fortran provenance blocks into a coverage census.",
    )
    parser.add_argument("--render", metavar="PATH", help="write the markdown table here")
    parser.add_argument("--json", metavar="PATH", help="write the JSON records here")
    parser.add_argument("--area", metavar="AREA", help="narrow to a subsystem path prefix")
    parser.add_argument(
        "--status-missing",
        action="store_true",
        help="print only the rows that are not yet ported",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the stdout summary")
    args = parser.parse_args(argv)

    census = scan()

    if args.area and not any(
        p.startswith(args.area.rstrip("/") + "/") for p in census.files
    ):
        parser.error(
            f"no upstream file under {args.area!r}; it is not an area of this tree"
        )

    if args.status_missing:
        view = census.select(area=args.area, status="missing")
        sys.stdout.write(render(census, area=args.area, status="missing"))
        if not args.quiet:
            print(
                f"{len(view.files)} of {len(census.files)} upstream files are "
                "status=missing",
                file=sys.stderr,
            )
        return 0

    if args.render:
        target = Path(args.render)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(census, area=args.area))
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "or_src": "$OR_SRC",
            "total": len(census.files),
            "counts": census.counts(),
            "files": {p: r.as_dict() for p, r in sorted(census.files.items())},
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    if not args.quiet and not (args.render or args.json):
        print(render(census, area=args.area))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
