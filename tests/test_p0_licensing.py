"""Phase 0 / Task 0.0 — the licensing gate.

Upstream OpenRadioss (`OpenCourant`) is AGPL-3.0-or-later
(`OpenCourant/LICENSE.md:1-3`, and every source file header such as
`OpenCourant/engine/source/engine/resol.F:1-7`).  This repository declared
MIT in `LICENSE`, GPL-3.0 in `README.md` and pointed `pyproject.toml` at the
MIT `LICENSE`, so the three artefacts disagree with each other *and* with the
licence of the code that was transcribed from them.

`test_declared_licence_is_consistent` is the consistency gate: it asserts the
declared licence agrees across every artefact.  It is marked
`xfail(strict=True)` on purpose — the contradiction is real and unfixed, and
strictness means that the day somebody reconciles the artefacts the test
XPASSes and FAILS the suite, forcing the marker to be removed deliberately
rather than leaving a stale "known failure" on a repository that has been
repaired.

`test_licensing_record_is_present_and_complete` always passes: the record of
the contradiction and of the four lawful resolutions
(`docs/LICENSING.md`) must not be deletable while the xfail marker hides the
underlying problem. It pins the four real `### Option N` *headings* (not the
bare words "Option N", which also occur in the Decision rationale) and
requires each of the four option sections to carry a non-empty body.

Whoever finally applies option 1 must install the COMPLETE AGPL text as
`LICENSE`, not a stub: the consistency gate matches the canonical title
"GNU AFFERO GENERAL PUBLIC LICENSE", which the full text spells out and a
one-line reference would not.

Path note: the brief reads `Path("LICENSE")` / `Path("pyproject.toml")`
relative to the CWD; this file resolves them from `__file__` instead so the
gate cannot be evaded by running pytest from a subdirectory.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"
LICENSING_DOC = REPO_ROOT / "docs" / "LICENSING.md"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "UNRESOLVED LICENCE CONTRADICTION: LICENSE is MIT, README.md declares "
        "GPL-3.0, pyproject.toml points at the MIT LICENSE, and upstream "
        "OpenCourant (LICENSE.md:1-3, engine/source/engine/resol.F:1-7) is "
        "AGPL-3.0-or-later, so the ported sources are AGPL-covered derivative "
        "works. No maintainer decision is recorded (docs/LICENSING.md "
        "'Decision'), therefore Task 0.0 has NOT chosen and this gate stays "
        "red-by-design, blocking every downstream phase whose entry criterion "
        "is 'Task 0.0 decided'."
    ),
)
def test_declared_licence_is_consistent() -> None:
    meta = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    declared = meta["project"]["license"]
    if isinstance(declared, dict):  # legacy table form: license = {file = "LICENSE"}
        licence_file = REPO_ROOT / declared["file"]
        assert licence_file.is_file(), f"pyproject points at a missing {licence_file}"
        assert "GNU AFFERO GENERAL PUBLIC LICENSE" in licence_file.read_text(
            encoding="utf-8"
        ), (
            f"{declared['file']} does not carry the full AGPL text "
            "(expected the canonical title 'GNU AFFERO GENERAL PUBLIC LICENSE')"
        )
    else:  # PEP 639 SPDX string form: license = "AGPL-3.0-or-later"
        assert declared == "AGPL-3.0-or-later", f"pyproject declares {declared!r}"
    assert "AGPL-3.0-or-later" in readme, "README.md does not declare AGPL-3.0-or-later"


OPTION_HEADINGS = (
    "### Option 1 — Relicense the port to AGPL-3.0-or-later",
    "### Option 2 — Keep the existing files' derived status and relicense only the new program",
    "### Option 3 — Replace the copied expressions",
    "### Option 4 — Obtain a commercial or dual-licence grant from Siemens",
)
MIN_OPTION_BODY_WORDS = 20


def test_licensing_record_is_present_and_complete() -> None:
    assert LICENSING_DOC.is_file(), "docs/LICENSING.md is missing"
    doc = LICENSING_DOC.read_text(encoding="utf-8")
    sections = re.split(r"^#{2,3} ", doc, flags=re.MULTILINE)
    for heading in OPTION_HEADINGS:
        assert f"\n{heading}\n" in f"\n{doc}", f"docs/LICENSING.md lacks heading {heading!r}"
        body = next(
            s for s in sections if s.startswith(heading[len("### ") :] + "\n")
        )
        assert len(body.split()) >= MIN_OPTION_BODY_WORDS, (
            f"section {heading!r} has an empty/trivial body — the record must "
            "not be deletable while the xfail marker hides the contradiction"
        )
    for citation in (
        "OpenCourant/LICENSE.md",
        "engine/source/engine/resol.F",
        "AGPL-3.0-or-later",
        "Siemens",
    ):
        assert citation in doc, f"docs/LICENSING.md does not cite {citation}"
    assert "\n## Decision\n" in f"\n{doc}", "docs/LICENSING.md has no Decision section"
    decision = doc.split("\n## Decision\n", 1)[1].lower()
    status = ("no decision recorded", "decision recorded", "decided on")
    assert any(s in decision for s in status), (
        "the Decision section records no status keyword; expected one of "
        + ", ".join(repr(s) for s in status)
    )
