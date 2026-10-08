"""Task P1.2 — every Fortran citation in pyradioss names the source tree portably.

The repo was written on a Windows box, so the docstrings that record where each
routine was ported from carried absolute paths of the form::

    C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\...\\sforc3.F

Those are facts about the *dev machine*, not about the port.  On any box where
the source tree lives elsewhere they are simply wrong, and the citation is the
one thing in the port that is supposed to make the Fortran findable.  The
portable spelling is ``$OR_SRC/<relative path>``: the environment variable the
rest of the tooling already resolves through ``pyradioss.paths.or_src()``, which
picks up an installed prefix, an export, or the layout of whatever box is
running.

This module enforces the rule two ways:

* the **scan** — no module under ``pyradioss/`` may name a Windows OpenRadioss
  reference path in its source, and
* the **rewriter** — ``tools/normalize_citations.rewrite()`` performs exactly
  that substitution and nothing wider, so the sweep that produced the tree is
  reproducible rather than a one-off hand edit.

A Windows path the rewriter does *not* recognise is left alone and reported,
never silently mangled: an unknown root is a finding for a human, not a
rewrite opportunity.  ``test_unrecognised_windows_paths_are_flagged_not_rewritten``
pins that distinction so the tool cannot grow into a blind ``C:\\...`` ->
something-else substitution.

The replacement stays the literal text ``$OR_SRC``.  Expanding it here would
freeze this machine's checkout prefix back into the docstrings — the same class
of defect this test exists to remove — so the test asserts the literal form.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.normalize_citations import (
    CITATION_PATTERN,
    EXEMPT_FILES,
    KNOWN_ROOTS,
    UNRECOGNISED_PATTERN,
    find_unrecognised,
    is_exempt,
    rewrite,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PYRADIOSS = REPO_ROOT / "pyradioss"

#: The Windows OpenRadioss path in ``pyradioss.gui.postproc`` that is a live
#: runtime value rather than a citation: the Windows install prefix the
#: converter executables are looked for under.  It is an install location, not
#: a source-tree reference, so the sweep must not rewrite it — and the scan
#: exempts exactly this one line, by value, with a test of its own.
RUNTIME_WINDOWS_PATH = r"C:\OpenRadioss\exec"


def _sources() -> list[tuple[Path, str]]:
    """Every non-exempt pyradioss module, with its source text."""
    out: list[tuple[Path, str]] = []
    for path in sorted(PYRADIOSS.rglob("*.py")):
        if is_exempt(path, REPO_ROOT):
            continue
        out.append((path, path.read_text(encoding="utf-8")))
    return out


def test_the_exempt_modules_are_the_windows_compat_implementation():
    """The exemption is load-bearing, so pin exactly which files it covers."""
    assert set(EXEMPT_FILES) == {"pyradioss/paths.py", "pyradioss/gui/postproc.py"}
    for rel in EXEMPT_FILES:
        assert (REPO_ROOT / rel).is_file(), rel
        assert is_exempt(REPO_ROOT / rel, REPO_ROOT)


def test_an_exempt_module_still_names_the_windows_path():
    """The exemption must not become a licence to delete the compatibility rule."""
    paths_py = (REPO_ROOT / "pyradioss" / "paths.py").read_text(encoding="utf-8")
    assert r"C:\OpenRadioss" in paths_py, "the Windows compat candidate was removed"
    postproc = (PYRADIOSS / "gui" / "postproc.py").read_text(encoding="utf-8")
    assert RUNTIME_WINDOWS_PATH in postproc


def test_pyradioss_tree_is_not_empty():
    """Guard the scan itself: a bad path would make every scan vacuously green."""
    assert len(list(PYRADIOSS.rglob("*.py"))) > 100


def test_the_two_known_roots_are_representable():
    """The scan has something to look for; neither recognised root may go dead."""
    assert len(KNOWN_ROOTS) == 2
    assert any(r.startswith(r"C:\OpenRadioss\source") for r in KNOWN_ROOTS)
    assert r"C:\OpenRadioss\hm_cfg_files" in KNOWN_ROOTS


# ---------------------------------------------------------------------------
# The scan: the tree must already be normalised.
# ---------------------------------------------------------------------------


def test_no_windows_reference_path_survives_in_pyradioss():
    offenders: list[str] = []
    for path, text in _sources():
        for match in CITATION_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel}:{line}: {match.group(0)}")
    assert not offenders, (
        f"{len(offenders)} Windows OpenRadioss citation(s) still in pyradioss/:\n"
        + "\n".join(offenders[:40])
        + ("\n..." if len(offenders) > 40 else "")
        + "\nRewrite them with tools/normalize_citations.rewrite()."
    )


def test_the_only_windows_path_left_is_the_documented_runtime_prefix():
    """Everything else Windows-shaped is gone; the exec prefix is accounted for."""
    leftovers: list[str] = []
    for path, text in _sources():
        for match in UNRECOGNISED_PATTERN.finditer(text):
            value = match.group(0)
            if value == RUNTIME_WINDOWS_PATH:
                continue
            line = text.count("\n", 0, match.start()) + 1
            rel = path.relative_to(REPO_ROOT)
            leftovers.append(f"{rel}:{line}: {value}")
    assert not leftovers, (
        f"{len(leftovers)} unrecognised Windows path(s) in pyradioss/:\n"
        + "\n".join(leftovers[:40])
    )


def test_the_runtime_prefix_is_still_present_where_it_belongs():
    """The exemption above must not become a licence to delete the fallback."""
    postproc = (PYRADIOSS / "gui" / "postproc.py").read_text(encoding="utf-8")
    assert RUNTIME_WINDOWS_PATH in postproc


def test_citations_that_survive_use_the_portable_form():
    for path, text in _sources():
        rel = path.relative_to(REPO_ROOT)
        assert "/home/valentin" not in text, f"{rel}: a dev checkout path leaked in"
        assert "OpenRadioss-latest-" not in text, f"{rel}: a dated snapshot survived"


# ---------------------------------------------------------------------------
# The rewriter: exactly two mappings, nothing wider.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            r"Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520"
            r"\engine\source\elements\solid\sforc3.F",
            "Ported from $OR_SRC/engine/source/elements/solid/sforc3.F",
        ),
        (
            # Raw docstrings and comments carry a single backslash in the file;
            # ordinary docstrings carry an escaped pair.  Both are source text.
            r"Ported from C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520"
            r"\\common_source\\tools\\graphs\\Graph.cpp",
            "Ported from $OR_SRC/common_source/tools/graphs/Graph.cpp",
        ),
        (
            r"Config lives under C:\OpenRadioss\hm_cfg_files\radioss140",
            "Config lives under $OR_SRC/hm_cfg_files/radioss140",
        ),
        (
            r"C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\profile",
            "$OR_SRC/hm_cfg_files/config/CFG/profile",
        ),
        (
            # A citation ends at the colon of a ``file.F: SUBROUTINE`` reference,
            # so the trailing text is not swallowed into the path.
            r"# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520"
            r"\engine\source\airbag\fvmesh.F: POLCLIP (lines 3160-3240)",
            "# Ported from $OR_SRC/engine/source/airbag/fvmesh.F: POLCLIP (lines 3160-3240)",
        ),
    ],
)
def test_rewrite_maps_the_known_roots(raw, expected):
    assert rewrite(raw) == expected


def test_rewrite_is_idempotent():
    once = rewrite(
        r"See C:\OpenRadioss\source\OpenRadioss-latest-20260520"
        r"\engine\source\engine\resol.F lines 7741-7762."
    )
    assert once == "See $OR_SRC/engine/source/engine/resol.F lines 7741-7762."
    assert rewrite(once) == once


def test_rewrite_leaves_non_windows_text_untouched():
    text = (
        "Cite $OR_SRC/engine/source/elements/solid/sforc3.F and\n"
        "/home/other/checkout/engine/source/x.F and C:/mixed/slashes/y.F\n"
    )
    assert rewrite(text) == text


def test_rewrite_does_not_expand_or_src_into_a_machine_path():
    """The replacement is documentation, not configuration."""
    out = rewrite(
        r"C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\a.F"
    )
    assert out == "$OR_SRC/engine/source/a.F"
    assert str(Path.home()) not in out


def test_rewrite_preserves_surrounding_text_and_line_count():
    raw = (
        "# header\n"
        r"# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520"
        r"\starter\source\elements\sinit3.F" "\n"
        r"# also C:\OpenRadioss\source\OpenRadioss-latest-20260520"
        r"\engine\source\elements\solid\sforc3.F" "\n"
        "# footer\n"
    )
    out = rewrite(raw)
    assert out == (
        "# header\n"
        "# Ported from $OR_SRC/starter/source/elements/sinit3.F\n"
        "# also $OR_SRC/engine/source/elements/solid/sforc3.F\n"
        "# footer\n"
    )
    assert out.count("\n") == raw.count("\n")


def test_unrecognised_windows_paths_are_flagged_not_rewritten():
    """Install-rooted Windows paths that are neither recognised root.

    Scope note: the flagger, like the scan, keys on a Windows path *rooted* at
    ``<drive>:\\OpenRadioss\\`` — a path that merely contains the word
    ``OpenRadioss`` somewhere deeper (``D:\\somewhere\\else\\OpenRadioss\\...``)
    is not an install-rooted citation and is out of scope for both.
    """
    for raw in (
        RUNTIME_WINDOWS_PATH,
        r"C:\OpenRadioss\source\other-snapshot-20240101\engine\source\x.F",
        r"C:\OpenRadioss\unknown_tree\thing.F",
    ):
        assert rewrite(raw) == raw, f"rewrote an unrecognised root: {raw!r}"
        assert find_unrecognised(raw), f"failed to flag an unrecognised root: {raw!r}"


def test_a_windows_path_that_merely_contains_the_word_is_out_of_scope():
    """Not an install root, so neither rewritten nor flagged."""
    raw = r"D:\somewhere\else\OpenRadioss\engine\source\x.F"
    assert rewrite(raw) == raw
    assert find_unrecognised(raw) == []

def test_a_known_root_is_not_reported_as_unrecognised():
    assert (
        find_unrecognised(
            r"C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\x.F"
        )
        == []
    )
    assert find_unrecognised(r"C:\OpenRadioss\hm_cfg_files\radioss140") == []


def test_find_unrecognised_reports_every_occurrence():
    found = find_unrecognised(
        r"C:\OpenRadioss\exec and again C:\OpenRadioss\exec"
    )
    assert len(found) == 2
    assert all(f == RUNTIME_WINDOWS_PATH for f in found)


def test_the_two_patterns_are_disjoint():
    """They must not overlap, or the flagger would flag its own output."""
    citation = r"C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\x.F"
    assert CITATION_PATTERN.search(citation)
    assert not UNRECOGNISED_PATTERN.search(citation)
    assert not UNRECOGNISED_PATTERN.search(rewrite(citation))


# ---------------------------------------------------------------------------
# The sweep the rewriter is for: applying it changes nothing today.
# ---------------------------------------------------------------------------


def test_sweep_is_a_no_op_on_the_current_tree():
    changed = [
        str(path.relative_to(REPO_ROOT))
        for path, text in _sources()
        if rewrite(text) != text
    ]
    assert not changed, f"pyradioss/ still needs the sweep: {changed[:20]}"
