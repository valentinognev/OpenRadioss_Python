"""Task P1.7 — a per-module line budget that can only shrink.

A port this size grows two ways: a task adds a feature, or a task appends to
the module the feature happened to live in. The first is reviewed on its
merits. The second is how a Fortran routine nobody can hold in their head
becomes a Python module nobody can hold in their head either, one green test
at a time.

So the size of every module is recorded once, in
``tools/validation_data/size_budget.json``, at the line count it has on the
commit that generated the file — **rounded down to that count, never up**. From
then on a module may only shrink; raising an entry is not a way to make this
test pass, it is a way to be caught by it.

Two rules, both enforced here:

* every budgeted module is at or under its recorded size
  (:func:`test_no_module_grew_past_its_budget`), and
* every module that is *not* in the budget — i.e. every module added after the
  budget was generated — is under the recorded default of
  ``new_files_default_max_lines`` (:func:`test_no_oversized_new_module`).

Together they make the linter total: no tracked module under ``pyradioss/`` can
escape a size check. The second rule is what kept the linter honest across the
wave-1 splits (``pyradioss/model/entities.py`` → the ``model/entities``
package, ``pyradioss/input/starter_keywords.py`` → the ``input/keywords``
package): their successors are new files, so each is judged on its own size
rather than inheriting a 53,286- or 97,102-line allowance. The budget was
regenerated from the split tree, so every member of both packages is
enumerated at its own measured size.

The budget file was generated complete — one entry for every
``pyradioss/**/*.py`` present when it was written — but completeness is
deliberately *not* asserted as such, because a module added by a later task is
exactly the case the default rule covers. What is asserted is the stronger
statement that makes the generated file useful: a module above the default must
carry its own entry.

Line counts are read with ``str.splitlines()`` throughout, so the numbers the
budget records are the same numbers the test measures — no ``wc -l`` that
disagrees with a file lacking a final newline.

Scope note on the scan: ``git ls-files --diff-filter=A`` from the plan text is
not a ``git ls-files`` option (git parses it as the unknown option
``diff-filter=A``), and even if it were it would only see files staged in the
current working state, so a module added and committed in one step would slip
through. The scan is therefore over *all* git-tracked modules under
``pyradioss/``, with "new" defined as "absent from the budget" — a property of
the tree, not of the moment the test happens to run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO_ROOT / "tools" / "validation_data" / "size_budget.json"

#: The rule key carried by the budget file itself, rather than a path entry.
RULE_KEY = "new_files_default_max_lines"

#: The size at which a module must have been split, from
#: ``plan/02_phase1_foundation.md`` Task P1.7.
DEFAULT_MAX_LINES = 1200

#: The two wave-1 splits, as the packages that replaced the monolithic modules
#: and the size each parent had on ``c4ccd9d5`` (the base this budget was first
#: generated against).  P1.4/P1.5 have since landed on ``main``: the parents
#: are gone and the budget now enumerates each package member individually, at
#: its own measured size.  The parent figures are kept here because they are
#: the yardstick for :func:`test_the_split_packages_are_not_worse_than_their_parents`
#: — a split that merely moved the bulk around has not split anything.
SPLIT_PACKAGES = {
#: package directory -> (the monolithic module it replaced, that module's size)
    "pyradioss/input/keywords": ("pyradioss/input/starter_keywords.py", 97102),
    "pyradioss/model/entities": ("pyradioss/model/entities.py", 53286),
}

#: ``starter_keywords.py`` survives the P1.5 split as a deprecation shim that
#: re-exports the new package.  It is budgeted as the small alias it is, and
#: :func:`test_the_deprecated_keyword_alias_stays_a_shim` stops it from quietly
#: growing back into a second copy of the readers.
DEPRECATED_ALIAS = "pyradioss/input/starter_keywords.py"
ALIAS_MAX_LINES = 100


def _load_budget() -> dict:
    assert BUDGET_PATH.is_file(), (
        f"missing {BUDGET_PATH.relative_to(REPO_ROOT)}; it records one entry "
        "per module under pyradioss/ and must exist for this linter to mean "
        "anything"
    )
    return json.loads(BUDGET_PATH.read_text(encoding="utf-8"))


def _budgets() -> dict[str, int]:
    """The budget file's path entries, without the rule key."""
    return {k: v for k, v in _load_budget().items() if k != RULE_KEY}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _sizes(modules: list[str]) -> dict[str, int]:
    """``{repo-relative path: line count}`` for each module."""
    return {rel: _line_count(REPO_ROOT / rel) for rel in modules}


def _tracked_modules() -> list[str]:
    """Every git-tracked ``pyradioss/**/*.py``, sorted, repo-root relative."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH; the new-module scan cannot enumerate")
    done = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "pyradioss/"],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, f"git ls-files failed: {done.stderr.strip()}"
    return sorted(p for p in done.stdout.split("\0") if p.endswith(".py"))


def _over_budget(sizes: dict[str, int], budgets: dict[str, int]) -> dict:
    """Budgeted modules whose measured size exceeds their recorded limit."""
    return {rel: (n, budgets[rel]) for rel, n in sizes.items()
            if rel in budgets and n > budgets[rel]}


def _oversized_new(sizes: dict[str, int], budgets: dict[str, int],
                   default: int) -> dict:
    """Unbudgeted (therefore new) modules above the default cap."""
    return {rel: (n, default) for rel, n in sizes.items()
            if rel not in budgets and n > default}


def _report(over: dict, label: str) -> str:
    """The finding list the assertion messages quote, one module per line."""
    return "\n".join(f"  {rel}: {n} lines, {label} {limit}"
                     for rel, (n, limit) in over.items())


# ---------------------------------------------------------------------------
# 1. the scan is not vacuous
# ---------------------------------------------------------------------------


def test_the_scan_sees_the_whole_package() -> None:
    """A wrong root would make every rule below pass for the wrong reason."""
    modules = _tracked_modules()
    assert len(modules) > 100, f"only {len(modules)} modules found under pyradioss/"
    assert "pyradioss/model/entities/materials.py" in modules
    assert "pyradioss/input/keywords/misc.py" in modules
    assert DEPRECATED_ALIAS in modules, (
        f"{DEPRECATED_ALIAS} is the deprecated P1.5 alias and is still tracked"
    )


# ---------------------------------------------------------------------------
# 2. no budgeted module grew
# ---------------------------------------------------------------------------


def test_no_module_grew_past_its_budget() -> None:
    """The rule the budget exists for: a module may shrink, never grow."""
    over = _over_budget(_sizes(_tracked_modules()), _budgets())
    assert over == {}, (
        "module(s) past their recorded size — split the module. An entry is "
        "never raised to absorb a change, because that is the only thing this "
        "linter exists to prevent:\n" + _report(over, "budget")
    )


def test_the_budget_file_declares_the_new_file_default() -> None:
    """The default is data in the file, not a constant only the test knows."""
    budget = _load_budget()
    assert RULE_KEY in budget, f"{BUDGET_PATH.name} does not carry {RULE_KEY}"
    assert budget[RULE_KEY] == DEFAULT_MAX_LINES


def test_every_budget_entry_names_a_tracked_module() -> None:
    """No stale path: an entry for a deleted file keeps an allowance alive."""
    orphans = sorted(set(_budgets()) - set(_tracked_modules()))
    assert not orphans, (
        "budget entries naming no git-tracked module under pyradioss/: "
        + ", ".join(orphans)
    )


def test_every_oversized_module_carries_its_own_entry() -> None:
    """Above the default, a module is grandfathered only by being listed.

    This is what makes the generated file complete in substance rather than in
    spelling: a module above 1200 lines cannot fall back on the default, so it
    must be enumerated — and being enumerated pins it.
    """
    budgets = _budgets()
    undeclared = [
        rel for rel, n in _sizes(_tracked_modules()).items()
        if rel not in budgets and n > DEFAULT_MAX_LINES
    ]
    assert not undeclared, (
        f"module(s) above {DEFAULT_MAX_LINES} lines with no budget entry — "
        "record them at their current size: " + ", ".join(undeclared)
    )


# ---------------------------------------------------------------------------
# 3. no new module is oversized
# ---------------------------------------------------------------------------


def test_no_oversized_new_module() -> None:
    """A module absent from the budget is new, and new modules have a cap."""
    over = _oversized_new(_sizes(_tracked_modules()), _budgets(),
                          _load_budget()[RULE_KEY])
    assert over == {}, (
        "module(s) above the new-file default with no budget entry — split the "
        "module:\n" + _report(over, "cap")
    )


def test_a_new_module_over_the_default_would_fail(tmp_path) -> None:
    """The rule above, driven over a file that is over the cap.

    A gate nobody has watched fail is a gate nobody has watched. The module is
    written to ``tmp_path`` — never into ``pyradioss/`` — and measured by the
    same helpers the live scan uses, so this asserts the comparison and its
    message rather than inventing a second implementation of the rule.
    """
    default = _load_budget()[RULE_KEY]
    probe = tmp_path / "budget_probe.py"
    probe.write_text("x = 1\n" * (default + 1), encoding="utf-8")

    sizes = {"pyradioss/budget_probe.py": _line_count(probe)}
    assert sizes["pyradioss/budget_probe.py"] == default + 1

    over = _oversized_new(sizes, {}, default)
    assert over == {"pyradioss/budget_probe.py": (default + 1, default)}
    assert _report(over, "cap") == \
        "  pyradioss/budget_probe.py: 1201 lines, cap 1200"


def test_a_module_at_the_default_is_allowed() -> None:
    """The boundary is inclusive at 1200 and exclusive at 1201."""
    default = _load_budget()[RULE_KEY]
    assert _oversized_new({"pyradioss/new.py": default}, {}, default) == {}
    assert _oversized_new({"pyradioss/new.py": default + 1}, {}, default) \
        == {"pyradioss/new.py": (default + 1, default)}


def test_a_budgeted_module_that_grew_would_fail() -> None:
    """Rule 1, driven over a module that outgrew its entry."""
    assert _over_budget({"pyradioss/x.py": 101}, {"pyradioss/x.py": 100}) \
        == {"pyradioss/x.py": (101, 100)}
    assert _over_budget({"pyradioss/x.py": 100}, {"pyradioss/x.py": 100}) == {}
    assert _over_budget({"pyradioss/x.py": 80}, {"pyradioss/x.py": 100}) == {}


# ---------------------------------------------------------------------------
# 4. the wave-1 splits stay split
# ---------------------------------------------------------------------------


def test_the_split_packages_are_budgeted_module_by_module() -> None:
    """Every member of a split package is pinned by its own entry.

    This is the rule the pre-split budget could not express.
    ``starter_keywords.py`` was one 97,102-line entry; after P1.5 it is thirteen
    files, and a single package-level allowance would let any one of them grow
    back to the parent's size. Enumerating them is what keeps the split real.
    """
    budgets = _budgets()
    sizes = _sizes(_tracked_modules())
    for package in SPLIT_PACKAGES:
        members = sorted(rel for rel in sizes
                         if rel.startswith(f"{package}/") and rel.endswith(".py"))
        assert members, f"{package} has no budgeted members; did the split land?"
        unbudgeted = [rel for rel in members if rel not in budgets]
        assert not unbudgeted, (
            f"{package} members with no budget entry: " + ", ".join(unbudgeted)
        )


def test_the_split_packages_are_not_worse_than_their_parents() -> None:
    """A split that relocates the bulk has not split anything.

    Measured per file rather than in aggregate: the aggregate legitimately
    rises a little when a monolith becomes a package (``__init__`` and the
    cross-module imports cost lines), but no single successor file should
    approach the parent it came from. That is the property a reader feels.
    """
    sizes = _sizes(_tracked_modules())
    for package, (parent, parent_lines) in SPLIT_PACKAGES.items():
        biggest = max(sizes[rel] for rel in sizes
                      if rel.startswith(f"{package}/") and rel.endswith(".py"))
        assert biggest < parent_lines, (
            f"{package} has a {biggest}-line member, against {parent_lines} for "
            f"the {parent} it replaced; the split did not make it smaller"
        )


def test_the_deprecated_keyword_alias_stays_a_shim() -> None:
    """``starter_keywords.py`` re-exports; it must not re-implement.

    It is budgeted like any other module, so it cannot grow past its entry —
    but an entry is a ceiling, not an intent. This states the intent: the old
    path is a deprecation shim for one release cycle, and a second copy of the
    readers appearing under it is a regression, not growth.
    """
    sizes = _sizes(_tracked_modules())
    assert DEPRECATED_ALIAS in sizes, (
        f"{DEPRECATED_ALIAS} is gone; the P1.5 deprecation cycle has ended and "
        "this test should be retired"
    )
    assert sizes[DEPRECATED_ALIAS] <= ALIAS_MAX_LINES, (
        f"{DEPRECATED_ALIAS} is {sizes[DEPRECATED_ALIAS]} lines, over the "
        f"{ALIAS_MAX_LINES}-line ceiling for a re-export shim: the readers "
        "have been duplicated under the deprecated path"
    )