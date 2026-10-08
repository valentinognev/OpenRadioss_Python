"""Task P1.10 — the documentation contract, made checkable.

`plan/02_phase1_foundation.md` Task P1.10 asks for three things and this module
holds each one:

* ``UPDATES.md`` is the mandatory agent changelog and is **newest entry on top**
  (the plan's own check, unchanged);
* ``docs/STATE.md`` carries a ``## Program status`` section pointing at
  ``plan/README.md`` and naming the current phase the way that index states it;
* ``docs/STATE.md`` §Baseline follows the Task P1.8 regression ledger
  (``tools/validation_data/baseline.json``) instead of the three mutually
  inconsistent historical fast-tier/full-suite figures.

The phase is not hardcoded here: the test parses ``plan/README.md`` §3 and
requires the document to agree with it, so the contract survives the next phase
starting.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STATE = REPO / "docs" / "STATE.md"
PLAN_INDEX = REPO / "plan" / "README.md"
UPDATES = REPO / "UPDATES.md"

LEDGER = "tools/validation_data/baseline.json"

#: The three historical figures Task P1.10 retires, in both spellings.
STALE_TRIO = ("11182", "11,182", "13030", "13,030", "9893", "9,893")


def _section(text: str, heading: str) -> str:
    """The body of the markdown section named ``heading``, or ``""``."""
    start = text.find(heading)
    if start < 0:
        return ""
    rest = text[start + len(heading):]
    end = re.search(r"^## ", rest, re.M)
    return rest[: end.start()] if end else rest


def _phase_number_for_file(index: str, stem: str) -> int | None:
    """The phase number ``plan/README.md`` §3 gives the phase file ``stem``."""
    row = re.search(rf"^\|\s*(\d+)\s*\|[^|]*\|\s*`{re.escape(stem)}\.md`", index, re.M)
    return int(row.group(1)) if row else None


# ---------------------------------------------------------------------------
# 1. the changelog is newest-first
# ---------------------------------------------------------------------------
def test_updates_md_is_the_newest_first_changelog() -> None:
    """The plan's check: versioned entries descend, so the newest is on top."""
    text = UPDATES.read_text(encoding="utf-8")
    vers = re.findall(r"^## (\d+\.\d+\.\d+)", text, re.M)
    assert vers, "no versioned entries"
    assert vers == sorted(vers, key=lambda v: [int(x) for x in v.split(".")], reverse=True)


# ---------------------------------------------------------------------------
# 2. the program status section, and the index it defers to
# ---------------------------------------------------------------------------
def test_state_md_has_a_program_status_section_pointing_at_the_plan_index() -> None:
    """An agent must be able to find the program from the onboarding document."""
    section = _section(STATE.read_text(encoding="utf-8"), "## Program status")
    assert section, "docs/STATE.md has no `## Program status` section"
    assert "plan/README.md" in section, "the section does not point at plan/README.md"


def test_program_status_names_the_phase_the_plan_index_states() -> None:
    """The named phase number and phase file must be the same row of §3."""
    section = _section(STATE.read_text(encoding="utf-8"), "## Program status")
    assert section, "docs/STATE.md has no `## Program status` section"

    named = re.search(r"Phase (\d+)", section)
    assert named, "the section names no current phase"
    stem = re.search(r"plan/(\d\d_phase\d+_[A-Za-z0-9_]+)\.md", section)
    assert stem, "the section names no phase file"

    index = _phase_number_for_file(PLAN_INDEX.read_text(encoding="utf-8"), stem.group(1))
    assert index is not None, f"plan/README.md §3 has no row for {stem.group(1)}.md"
    assert int(named.group(1)) == index, (
        f"docs/STATE.md calls phase {named.group(1)} current but "
        f"plan/README.md §3 files {stem.group(1)}.md as phase {index}"
    )


def test_program_status_does_not_declare_a_later_phase_finished() -> None:
    """A record may say a phase *after* the current one is not started."""
    section = _section(STATE.read_text(encoding="utf-8"), "## Program status")
    assert section, "docs/STATE.md has no `## Program status` section"
    current = int(re.search(r"Phase (\d+)", section).group(1))
    for number in re.findall(r"Phase (\d+)[^.\n]{0,60}?\b(?:complete|finished|done)\b", section, re.I):
        assert int(number) <= current, (
            f"docs/STATE.md declares Phase {number} complete while the current phase is {current}"
        )


# ---------------------------------------------------------------------------
# 3. the baseline follows the ledger
# ---------------------------------------------------------------------------
def test_baseline_points_at_the_regression_ledger() -> None:
    """§Baseline defers to the Task P1.8 ledger by path."""
    section = _section(STATE.read_text(encoding="utf-8"), "## Baseline")
    assert section, "docs/STATE.md has no `## Baseline` section"
    assert LEDGER in section, f"§Baseline does not point at {LEDGER}"


def test_state_md_does_not_restore_the_stale_fast_tier_trio() -> None:
    """The three retired figures belonged to the pre-migration Windows box."""
    text = STATE.read_text(encoding="utf-8")
    for figure in STALE_TRIO:
        assert figure not in text, f"docs/STATE.md states the stale figure {figure}"