"""P1.1: no pyradioss module may emit an escape-sequence warning.

Windows-path citations such as ``C:\\OpenRadioss\\source\\...`` live in many
docstrings.  In a non-raw string ``\\O`` (and every other Windows escape) is an
*invalid escape sequence*: the bytes are kept verbatim today, but the compiler
warns (DeprecationWarning on 3.12, SyntaxWarning on 3.14+) and a future release
may make it an error.  The fix is to make those docstrings raw, never to rewrite
the citation text.

The check compiles every ``pyradioss/**/*.py`` with warnings recorded and
asserts none of them mentions an escape sequence.  The warning *category* is
deliberately not asserted: it differs across interpreter versions, and the
defect is the message, not the class.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import pyradioss

_PKG = Path(pyradioss.__file__).resolve().parent


def _escape_warnings(path: Path) -> list[str]:
    """Return ``"<rel>:<lineno>: <message>"`` for escape warnings in *path*."""
    source = path.read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings(record=True) as caught:
        # "always" so a warning already shown once in the session is re-emitted.
        warnings.simplefilter("always")
        try:
            compile(source, str(path), "exec")
        except SyntaxError:
            # A file that does not parse is a different test's problem; do not
            # mask it behind a warning assertion here.
            pytest.fail(f"{path} does not compile")
    return [
        f"{path.relative_to(_PKG.parent)}:{w.lineno}: {w.message}"
        for w in caught
        if "escape sequence" in str(w.message)
    ]


def _python_sources() -> list[Path]:
    return sorted(_PKG.rglob("*.py"))


def test_package_is_not_empty():
    """Guard against a wrong _PKG making the sweep vacuously pass."""
    sources = _python_sources()
    assert len(sources) > 100, f"only found {len(sources)} modules under {_PKG}"


def test_no_module_emits_an_escape_warning():
    offenders: list[str] = []
    for path in _python_sources():
        offenders.extend(_escape_warnings(path))
    assert not offenders, (
        "these docstrings hold an invalid escape sequence — make the docstring "
        "a raw string (r\"\"\"...\"\"\") and leave the cited path text unchanged:\n"
        + "\n".join(f"  {line}" for line in offenders)
    )


@pytest.mark.parametrize("relpath", [
    "engine/flex_body.py",
    "materials/law12_comp3d.py",
    "materials/law15_chang.py",
    "materials/law50_visc_honey.py",
    "materials/law87_barlat2000.py",
    "materials/law119_seatbelt.py",
])
def test_known_windows_path_modules_are_clean(relpath):
    """The modules that cite C:\\OpenRadioss must stay clean individually.

    The whole-package sweep above already covers these; this pins the set so a
    future edit that reintroduces an escape names the file directly.
    """
    assert not _escape_warnings(_PKG / relpath), f"{relpath} emits an escape warning"