"""Rewrite Windows OpenRadioss citations in ``pyradioss`` to the ``$OR_SRC`` form.

Task P1.2.  The port's docstrings record, for every routine, the Fortran file it
was ported from.  Those citations were written on a Windows box and carried that
box's absolute paths::

    C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\...\\sforc3.F

That is a fact about the authoring machine, not about the port.  Everywhere else
the repository already refers to the source tree through the ``OR_SRC``
environment variable, resolved by :func:`pyradioss.paths.or_src` — so this module
rewrites the citations to the same spelling and leaves them as the *literal*
text ``$OR_SRC/<relative path>``.  It deliberately does not expand the variable
here: expanding it in the rewriter would freeze whatever checkout this happens
to run in back into the docstrings, which is the defect being removed.

Two roots are recognised, and only two:

``C:\\OpenRadioss\\source\\OpenRadioss-latest-<date>\\<rel>``
    the source checkout — becomes ``$OR_SRC/<rel>``.
``C:\\OpenRadioss\\hm_cfg_files\\<rel>``
    the configuration tree — becomes ``$OR_SRC/hm_cfg_files/<rel>``.

Anything else that looks like a Windows OpenRadioss path is **left alone and
reported** (:func:`find_unrecognised`), because an unrecognised root is a
finding for a human to resolve, not something to rewrite on a guess.  In
particular ``C:\\OpenRadioss\\exec`` in :mod:`pyradioss.gui.postproc` is a live
Windows *runtime* fallback for the converter executables — an install prefix,
not a source citation — and must keep its Windows shape.

Both escaped (``C:\\\\OpenRadioss\\\\...``, in ordinary docstrings) and raw
(``C:\\OpenRadioss\\...``, in raw docstrings and comments) spellings are handled,
because commit 1e134bd raw-stringed the citations and the two forms coexist in
the tree.

Command line::

    python tools/normalize_citations.py --check         # report, change nothing
    python tools/normalize_citations.py pyradioss/model # rewrite in place
    python tools/normalize_citations.py pyradioss/model --stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

__all__ = [
    "KNOWN_ROOTS",
    "EXEMPT_FILES",
    "CITATION_PATTERN",
    "UNRECOGNISED_PATTERN",
    "find_unrecognised",
    "is_exempt",
    "rewrite",
    "rewrite_file",
    "main",
]

#: Modules whose Windows paths are **runtime values, not citations**, and are
#: therefore exempt from the sweep.  Both implement the Windows-compatibility
#: rule (AGENTS.md §4.1 rule 3): they *are* the code that resolves
#: ``C:\OpenRadioss`` on a Windows box, and their docstrings document that
#: candidate list.  Rewriting them would not normalise a citation, it would
#: falsify the compatibility rule — ``C:\OpenRadioss\hm_cfg_files`` (rule 3)
#: is a different candidate from ``$OR_SRC/hm_cfg_files`` (rule 1), and
#: :mod:`pyradioss.paths` lists both.  ``tests/test_p0_paths.py`` pins that
#: precedence, so the exemption is load-bearing, not cosmetic.
EXEMPT_FILES: tuple[str, ...] = (
    "pyradioss/paths.py",
    "pyradioss/gui/postproc.py",
)


def is_exempt(path: Path, root: Path | None = None) -> bool:
    """True when *path* is a Windows-compat implementation module, not a citation."""
    base = Path(root) if root is not None else Path.cwd()
    try:
        rel = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    return rel in EXEMPT_FILES

#: Separator as it appears in *source text*: either a run of backslashes (an
#: escaped docstring, where the file contains ``\\``) or a single one (a raw
#: docstring or a comment).  ``_B`` is spliced into the patterns below.
_B = r"(?:\\+|\\)"

#: Recognised roots, in the order they are tried.  Each is written in its
#: single-backslash (raw-docstring) spelling; the patterns match either.
KNOWN_ROOTS: tuple[str, ...] = (
    r"C:\OpenRadioss\source\OpenRadioss-latest-<date>",
    r"C:\OpenRadioss\hm_cfg_files",
)

#: One path segment: everything up to the next separator or the character that
#: ends the citation (whitespace, a quote, or the bracketing/punctuating
#: characters these citations sit inside — e.g. ``.../fvmesh.F: PORFOR4``).
_SEG = r"[^\\\s\"'`,;:)\]}]+"

#: The relative part of a citation, separators included, captured so it can be
#: re-emitted with forward slashes.
_REL = rf"(?:{_B}{_SEG})*"

CITATION_PATTERN = re.compile(
    # Source checkout.  The dated ``OpenRadioss-latest-<date>`` directory is
    # matched by shape, so a future snapshot date needs no code change.
    r"(?P<src>C:\\+OpenRadioss\\+source\\+OpenRadioss-latest-[0-9]+"
    rf"(?P<src_rel>{_REL}))"
    r"|"
    # Configuration tree.
    r"(?P<cfg>C:\\+OpenRadioss\\+hm_cfg_files"
    rf"(?P<cfg_rel>{_REL}))"
)

#: A Windows path naming an OpenRadioss install that is *not* one of the two
#: recognised citation roots.  The lookahead is zero-width — it must not consume
#: the separator that :data:`_REL` then starts with — and it keeps the two
#: patterns disjoint, so :func:`find_unrecognised` cannot flag its own output.
UNRECOGNISED_PATTERN = re.compile(
    r"[A-Za-z]:"
    rf"{_B}OpenRadioss"
    rf"(?!(?:{_B})(?:source{_B}OpenRadioss-latest-[0-9]+{_B}|hm_cfg_files{_B}))"
    rf"{_REL}"
)


def _to_posix(rel: str) -> str:
    """Re-emit a captured relative path with forward slashes."""
    return "/".join(part for part in re.split(r"\\+|\\", rel) if part)


def rewrite(text: str) -> str:
    """Return *text* with recognised Windows citations replaced by ``$OR_SRC``.

    Unrecognised Windows paths are returned unchanged — see
    :func:`find_unrecognised` for how those are surfaced.
    """

    def _sub(match: re.Match[str]) -> str:
        if match.group("src") is not None:
            rel = _to_posix(match.group("src_rel"))
            return "$OR_SRC/" + rel if rel else "$OR_SRC"
        rel = _to_posix(match.group("cfg_rel"))
        return "$OR_SRC/hm_cfg_files" + ("/" + rel if rel else "")

    return CITATION_PATTERN.sub(_sub, text)


def find_unrecognised(text: str) -> list[str]:
    """Return every Windows OpenRadioss path in *text* that is not a citation."""
    return [m.group(0) for m in UNRECOGNISED_PATTERN.finditer(text)]


def rewrite_file(path: Path, *, check: bool = False, stdout: bool = False) -> bool:
    """Apply :func:`rewrite` to *path*.  Returns True when the text changed."""
    original = path.read_text(encoding="utf-8")
    new = rewrite(original)
    if new == original:
        return False
    if stdout:
        sys.stdout.write(new)
    elif not check:
        path.write_text(new, encoding="utf-8")
    return True


def _iter_paths(targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            out.append(path)
        else:
            raise SystemExit(f"not a Python file or directory: {target}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", default=["pyradioss"])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change; do not write (exit 1 if anything would)",
    )
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args(argv)

    root = Path.cwd()
    changed: list[str] = []
    unrecognised: list[str] = []
    skipped: list[str] = []
    for path in _iter_paths(args.targets or ["pyradioss"]):
        if is_exempt(path, root):
            skipped.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        unrecognised.extend(f"{path}: {u}" for u in find_unrecognised(text))
        if rewrite_file(path, check=args.check, stdout=args.stdout):
            changed.append(str(path))

    for path in skipped:
        print(f"skipped (Windows-compat implementation): {path}", file=sys.stderr)

    if unrecognised:
        print("Unrecognised Windows paths (left alone):", file=sys.stderr)
        for line in unrecognised:
            print(f"  {line}", file=sys.stderr)
    if args.check:
        if changed:
            print("Would rewrite:", file=sys.stderr)
            for path in changed:
                print(f"  {path}", file=sys.stderr)
            return 1
        print("pyradioss: no Windows citations left.", file=sys.stderr)
        return 0
    print(f"rewrote {len(changed)} file(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
