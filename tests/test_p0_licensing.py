"""Phase 0 / Task 0.0 — the licensing gate.

Upstream OpenRadioss (`OpenCourant`) is AGPL-3.0-or-later
(`OpenCourant/LICENSE.md:1-3`, and every source file header such as
`OpenCourant/engine/source/engine/resol.F:1-7`).  This repository is a
literal transcription of that source, so AGPL section 5(c) makes the whole
work AGPL.

**Decision (maintainer, 2026-10-07): Option 1 — this repository is
AGPL-3.0-or-later as a whole.**  `LICENSE`, `pyproject.toml` and `README.md`
now agree on that, and this file is the gate that proves it: it reads the three
artefacts and asserts the declared licence agrees across all of them, that
`LICENSE` carries the COMPLETE AGPL text rather than a stub or a reference, and
that none of them still claims a permissive licence for this work.

The gate used to be marked `@pytest.mark.xfail(strict=True)` so the
contradiction stayed red by design while no decision was recorded.  The
decision has now been recorded and applied, so the marker is gone and the test
must pass for real.  Any future relicensing must change this test deliberately
in the same commit as the artefacts.

`test_licensing_record_is_present_and_complete` always passes: the record of
the contradiction and of the four lawful resolutions (`docs/LICENSING.md`) must
remain, because it is what tells a later reader *why* the port is AGPL and
which alternatives were rejected. It pins the four real `### Option N`
*headings* (not the bare words "Option N", which also occur in the Decision
rationale) and requires each of the four option sections to carry a non-empty
body.

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
LICENSE = REPO_ROOT / "LICENSE"
LICENSING_DOC = REPO_ROOT / "docs" / "LICENSING.md"

SPDX = "AGPL-3.0-or-later"
AGPL_TITLE = "GNU AFFERO GENERAL PUBLIC LICENSE"

def test_declared_licence_is_consistent() -> None:
    meta = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    licence_text = LICENSE.read_text(encoding="utf-8")

    # 1. pyproject: the declared licence must be the AGPL, in either the
    #    PEP 639 SPDX string form or the legacy `license = {file = ...}`
    #    table form.
    declared = meta["project"]["license"]
    if isinstance(declared, dict):
        licence_file = REPO_ROOT / declared["file"]
        assert licence_file.is_file(), f"pyproject points at a missing {licence_file}"
        assert licence_file == LICENSE, (
            f"pyproject points at {declared['file']}, not the repository LICENSE"
        )
    else:
        assert declared == SPDX, f"pyproject declares {declared!r}, expected {SPDX!r}"

    # 2. LICENSE: the COMPLETE AGPL text, not a stub or a one-line reference.
    #    The section 5(c) paragraph is what makes this the whole-work AGPL
    #    obligation docs/LICENSING.md reasons from, so its presence is the
    #    evidence that the text is complete rather than decorated.
    assert AGPL_TITLE in licence_text, (
        f"LICENSE does not carry the AGPL text (expected the canonical title "
        f"{AGPL_TITLE!r})"
    )
    for clause in (
        "You must license the entire work, as a whole, under this",
        "Remote Network Interaction",
    ):
        assert clause in licence_text, (
            f"LICENSE is missing AGPL clause {clause!r} — the text looks truncated"
        )
    # 3. LICENSE: the grant must be `-or-later`, which the AGPL text itself
    #    does not assert for this work; the notice does.
    assert "either version 3 of the License, or (at your option) any later version" in (
        licence_text
    ), "LICENSE does not grant the AGPL 'or any later version' option"

    # 4. README: must name the same licence...
    assert SPDX in readme, f"README.md does not declare {SPDX}"
    #    ...and must no longer claim a permissive licence for this work. The
    #    patterns are anchored so that "AGPL-3.0-or-later" — which contains
    #    "GPL-3.0" as a substring — is not mistaken for the old GPL-3.0 claim.
    readme_licence = readme.split("\n## License\n", 1)
    assert len(readme_licence) == 2, "README.md has no '## License' section"
    section = readme_licence[1].split("\n## ", 1)[0]
    stale_patterns = {
        "MIT": r"\bMIT\b",
        "GPL (non-affero)": r"(?<!A)\bGPL\b",
        "LGPL": r"\bLGPL\b",
        "BSD": r"\bBSD\b",
    }
    for name, pattern in stale_patterns.items():
        assert not re.search(pattern, section), (
            f"README.md's License section still claims {name} for this work"
        )
    assert SPDX in section, (
        f"README.md's License section does not itself name {SPDX}"
    )


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
