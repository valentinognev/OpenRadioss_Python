"""The frozen surface of ``pyradioss/implicit/`` — Task P1.6.

What this file is for
---------------------
``pyradioss/implicit/`` carries roughly 25k LOC that upstream OpenRadioss does
**not** have: the implicit Newton / arc-length branch (M8–M15), modal and
complex-modal superposition (M16–M18), random / PSD response and response
spectra (M19), and the spectral-fatigue tower (M20–M34). That extra surface is
why the subsystem has to be *pinned*, not merely documented: the moment a file
can be added or deleted without consequence, the record of what the project is
silently wrong.

The binding decision, encoded here
----------------------------------
* **Decision (maintainer, 2026-10-02)** — recorded in
  ``plan/00_ORCHESTRATION.md`` §3: **keep, but quarantine behind extras.**
* Consequence, binding on every phase: **no new features** under
  ``pyradioss/implicit/`` except what Phase 15
  (``15_phase14_implicit_parity.md``) needs for parity with upstream
  ``engine/source/implicit``.
* Every module listed below must remain **importable and green** for the whole
  program. A phase that breaks one is a phase that does not merge.
* The optional linear-solver backends (CHOLMOD via ``sksparse``, MUMPS via
  ``python-mumps``) keep their existing optional guards. Do **not** promote them
  to hard dependencies.

Provenance of the list below
----------------------------
* **Method:** generated, not typed. It came from a one-shot **census** of the
  live package directory with ``pkgutil.iter_modules`` over
  ``pyradioss.implicit.__path__``, minus the two names in
  ``EXCLUDED_FROM_CENSUS``. Hand-typing the names is the failure mode this file
  exists to prevent: one typo would read as a spurious "added"/"removed" verdict
  forever.
* **Census taken:** 2026-10-04, from the tree at commit ``cf046a3``
  (``docs(gate): the licence gate now meets the agent in AGENTS.md and both
  plan files``), branch ``plan/p1``.
* **Count:** **33** modules. ``plan/02_phase1_foundation.md`` Task P1.6 (line
  354) wrote "the 37 module names currently in ``pyradioss/implicit/``" from a
  snapshot taken before this tree; the discrepancy is recorded in
  ``tests/test_p1_frozen_implicit.py``'s header and in the Task P1.6 report, and
  **the live tree wins** — the plan figure is a stale snapshot, not a
  requirement. There is no missing module: ``git log --diff-filter=D --
  pyradioss/implicit/`` is empty, so nothing was ever deleted from this package
  and the 33 names are simply the 33 that exist.

The scope rule, and why it is a denylist and not a prefix rule
---------------------------------------------------------------
``EXCLUDED_FROM_CENSUS`` names **exactly two** files, each for its own reason:

* ``__init__`` — the package facade. It exists in every state of the tree and
  its surface is the re-export list, which grows as a *consequence* of the
  modules below; freezing the consequence alongside the cause would make every
  new module need two approvals. Named here defensively: ``pkgutil`` skips
  ``__init__`` by design (``pkgutil._iter_file_finder_modules``), so no census
  of this kind ever reports it. Cost of excluding it, stated plainly: an edit
  that only adds a re-export to the facade is not covered by this gate. The
  facade's *importability* is covered — every probe imports the package to get
  at the modules.
* ``_frozen_surface`` — this file. It is the record, not part of the surface it
  records; keeping the mechanism out of its own census is what lets the record
  be edited without re-opening the quarantine decision about itself. (No hash
  circularity is involved — the digest is over *names*, which are known without
  reading this file.)

**Everything else is in the census, including a ``_``-prefixed module.** An
earlier revision of this rule excluded *any* name beginning with ``_``, which
left the quarantine with an open door: a new private module (``_newton_krylov``)
was neither frozen nor imported, and every test still passed. The exclusion is
now an explicit two-name list, pinned by
``tests/test_p1_frozen_implicit.py::test_the_census_excludes_only_the_two_named_modules``
so it cannot be widened silently either.

How the digest is computed, and how a legitimate change is made
----------------------------------------------------------------
``FROZEN_SURFACE_SHA256`` is a digest **over the SORTED names**, one per line,
with a trailing newline::

    sha256("\\n".join(sorted(names)) + "\\n").hexdigest()

It is pinned as a literal, not computed at import, so it is a checksum rather
than an echo of the tree. ``tests/test_p1_frozen_implicit.py::
test_frozen_surface_sha256_is_recomputable`` recomputes that recipe from the
**live** ``pkgutil`` scan, with its own independent implementation, and
compares — an unverified constant would be decoration, and a single shared
helper would be an echo of the tree in two places instead.

* **ADDING a name is a deliberate act and requires the maintainer's approval.**
  It is not a mechanical consequence of writing a new file, and it is not
  something an agent may grant itself. With approval recorded, the legitimate
  procedure is: regenerate this file's ``_SORTED_MODULE_NAMES`` from a live
  ``pkgutil.iter_modules`` scan (minus ``EXCLUDED_FROM_CENSUS``) and replace
  ``FROZEN_SURFACE_SHA256`` with the digest that same scan prints. Edit **this
  file** (``_frozen_surface.py``) together with the new module, in one commit
  whose message quotes the approval and the before/after digest.
* **REMOVING a name is a hard failure**, never a list edit. ``test_implicit_
  surface_is_frozen`` reports it under ``"removed"``, and the list is not to be
  edited to make it pass. A removal is either a mistake or a decision that
  re-opens the quarantine decision itself, and both outrank a test edit.
* **No other file may be edited to make either verdict change**: not the census
  rule, not ``EXCLUDED_FROM_CENSUS``, not the test, not the plan's number. A
  name added to the exclusion list is the same deliberate act as a name added to
  the census, and is pinned to the same two entries.

Interfaces
----------
``FROZEN_MODULES: frozenset[str]`` — the 33 module names, above.
``FROZEN_SURFACE_SHA256: str`` — the sorted-names digest, above.
``EXCLUDED_FROM_CENSUS: frozenset[str]`` — the two names above that are not
part of the frozen surface, and the whole of the exclusion rule.
"""

from __future__ import annotations

#: The names inside ``pyradioss/implicit/`` that are deliberately NOT part of
#: the frozen surface. Exactly two, for the two reasons in the module docstring;
#: a ``_`` prefix is NOT a criterion. Pinned by the test suite, so widening this
#: to silence a new module fails the gate like any other surface change.
EXCLUDED_FROM_CENSUS: frozenset[str] = frozenset({"__init__", "_frozen_surface"})

#: Generated by a pkgutil census on 2026-10-04 (see the module docstring).
#: Sorted so a diff of this file reads as an ordered change, not a reshuffle.
_SORTED_MODULE_NAMES: tuple[str, ...] = (
    "assembly",
    "bfgs",
    "buckling",
    "complex_modal",
    "constraints",
    "contact",
    "convergence",
    "damping_matrix",
    "dofmap",
    "dynamics",
    "evolutionary_fatigue",
    "evolutionary_multi_input",
    "followerload",
    "freq_evolutionary_multi_input",
    "joint_evolutionary_fatigue",
    "joint_nongaussian_fatigue",
    "linesearch",
    "linsolve",
    "modal",
    "modal_response",
    "multi_input_fatigue",
    "multi_input_response",
    "multiaxial_fatigue",
    "nongaussian_fatigue",
    "nongaussian_wigner_ville_fatigue",
    "nonproportional_fatigue",
    "nonstationary_fatigue",
    "random_response",
    "response_spectrum",
    "spectral_fatigue",
    "spectral_nonproportional_fatigue",
    "statics",
    "wigner_ville_fatigue",
)

#: The frozen surface: the module names of ``pyradioss/implicit/`` minus the two
#: excluded above.
FROZEN_MODULES: frozenset[str] = frozenset(_SORTED_MODULE_NAMES)

#: sha256 over the SORTED names, one per line, trailing newline.
FROZEN_SURFACE_SHA256: str = (
    "60611cce763b4bb9acdb34eb4c98377ea2d9b80f7091fdde339e12aaa84b165a"
)
